"""Instrumentos con muestras reales (3.0): las notas del compositor se tocan con un banco de sonidos
General MIDI (FluidR3, licencia MIT, instalado en la imagen) a través de FluidSynth, en vez de con
los sintetizadores de lofigen. El compositor es el mismo; cambia quién toca.

Cómo: durante el render de una sección las notas no se sintetizan, se apuntan como eventos por
capa (acordes, melodía, resto, batería, pad). Al final cada capa se escribe como un fichero MIDI
mínimo (formato 0) y FluidSynth lo convierte en audio (sin su reverberación ni chorus; la sala
la pone el mezclador de siempre). El resultado entra en los mismos buffers, así que el bombeo
lateral, la sala, el filtro «lofi», el realce de graves y la ganancia por pieza siguen igual.

El banco se puede cambiar poniendo un .sf2 en data/soundfont.sf2 (por ejemplo GeneralUser GS).
"""
import os
import shutil
import struct
import subprocess
import tempfile

import numpy as np

SR = 44100
SF2_SISTEMA = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
SF2_USUARIO = "/app/data/soundfont.sf2"

# instrumento del compositor → (programa General MIDI, ganancia relativa, transposición)
PROGRAMAS = {
    "piano": (0, 1.0, 0), "rhodes": (4, 1.0, 0), "vibra": (11, 1.3, 0), "pad": (89, .55, 0), "pulso": (81, .5, 0),
    "chip": (80, .5, 0), "caja": (12, .8, 0), "guitarra": (24, .45, 0), "flauta": (73, .9, 0), "campana": (14, .6, 0),
    "saxo": (65, .4, 0), "trompeta": (59, .5, 0), "clarinete": (71, .6, 0), "koto": (107, .7, 0),
    "shakuhachi": (77, .8, 0), "bonsho": (14, .7, -12), "cuenco": (112, .6, 0),
}
BAJOS = {"sub": 38, "cuerda": 33, "sinte": 38, "triangulo": 32}   # timbres de bajo → programa (synth bass, finger, synth, acoustic)
# golpes de la batería de lofigen (tipo de filtro, frecuencia aproximada) → nota del canal de percusión
PERCUSION = [("bandpass", 1500, 38), ("bandpass", 2600, 37), ("bandpass", 1200, 38), ("bandpass", 1000, 76),
             ("highpass", 7500, 42), ("highpass", 6500, 46), ("highpass", 9000, 82)]


def disponible():
    return shutil.which("fluidsynth") is not None and (os.path.exists(SF2_USUARIO) or os.path.exists(SF2_SISTEMA))


def _banco():
    return SF2_USUARIO if os.path.exists(SF2_USUARIO) else SF2_SISTEMA


def _vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append(0x80 | (n & 0x7F))
        n >>= 7
    return bytes(reversed(out))


def _midi(eventos, tpq=960):
    """Fichero MIDI formato 0 a 60 negras por minuto (1 s = tpq ticks). eventos: [(t, tipo, canal, a, b)]."""
    pista = bytearray()
    pista += _vlq(0) + b"\xff\x51\x03" + (1_000_000).to_bytes(3, "big")   # tempo: 1 s por negra
    ultimo = 0
    for t, tipo, canal, a, b in sorted(eventos, key=lambda e: (e[0], 0 if e[1] == "prog" else 1 if e[1] == "off" else 2)):
        tick = max(0, int(round(t * tpq)))
        pista += _vlq(tick - ultimo)
        ultimo = tick
        if tipo == "prog":
            pista += bytes([0xC0 | canal, a])
        elif tipo == "on":
            pista += bytes([0x90 | canal, a, b])
        else:
            pista += bytes([0x80 | canal, a, 0])
    pista += _vlq(0) + b"\xff\x2f\x00"
    cab = b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpq)
    return cab + b"MTrk" + struct.pack(">I", len(pista)) + bytes(pista)


def _render(eventos, n):
    """Audio mono float64 de n muestras con FluidSynth (sin reverb/chorus propios)."""
    if not eventos:
        return np.zeros(n)
    with tempfile.TemporaryDirectory() as d:
        mid = os.path.join(d, "capa.mid")
        wav = os.path.join(d, "capa.wav")
        open(mid, "wb").write(_midi(eventos))
        subprocess.run(["fluidsynth", "-ni", "-q", "-R", "0", "-C", "0", "-g", "0.6", "-r", str(SR), "-O", "s16", "-T", "wav",
                        "-F", wav, _banco(), mid], stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
        try:
            from scipy.io import wavfile
            sr, x = wavfile.read(wav)
        except Exception:
            return np.zeros(n)
    x = x.astype(np.float64) / 32768
    if x.ndim > 1:
        x = x.mean(axis=1)
    out = np.zeros(n)
    m = min(n, len(x))
    out[:m] = x[:m]
    return out


class Grabador:
    """Apunta las notas de una sección por buffer y las convierte en audio al final.
    vol: el mismo volumen que usarían los sintetizadores (0,02-0,13); se pasa a velocidad MIDI."""

    def __init__(self):
        self.capas = {}        # id(buffer) → (buffer, eventos, canales)

    def _capa(self, buf):
        k = id(buf)
        if k not in self.capas:
            self.capas[k] = (buf, [], {})
        return self.capas[k]

    def _canal(self, capa, programa):
        canales = capa[2]
        if programa not in canales:
            libres = [c for c in range(16) if c != 9 and c not in canales.values()]
            if not libres:
                return None
            canales[programa] = libres[0]
            capa[1].append((0.0, "prog", libres[0], programa, 0))
        return canales[programa]

    @staticmethod
    def _vel(vol, ref, minimo=28, maximo=112):
        return int(np.clip(minimo + (maximo - minimo) * (vol / ref) ** .6, 1, 127))

    def nota(self, buf, inst, m, t, dur, vol):
        prog, gan, transp = PROGRAMAS.get(inst, PROGRAMAS["rhodes"])
        capa = self._capa(buf)
        c = self._canal(capa, prog)
        if c is None:
            return
        m = int(np.clip(m + transp, 0, 127))
        t = max(0.0, t)
        capa[1].append((t, "on", c, m, self._vel(vol * gan, .085)))
        capa[1].append((t + max(dur, .05), "off", c, m, 0))

    def bajo(self, buf, timbre, m, t, dur, vol, nivel):
        capa = self._capa(buf)
        c = self._canal(capa, BAJOS.get(timbre, 32))
        if c is None:
            return
        m = int(np.clip(m, 0, 127))
        t = max(0.0, t)
        capa[1].append((t, "on", c, m, self._vel(vol * nivel, .075, 40, 118)))
        capa[1].append((t + max(dur, .05), "off", c, m, 0))

    def bombo(self, buf, t, vol):
        capa = self._capa(buf)
        t = max(0.0, t)
        capa[1].append((t, "on", 9, 36, self._vel(vol, .34, 50, 120)))
        capa[1].append((t + .3, "off", 9, 36, 0))

    def golpe(self, buf, t, vol, tipo, frec, dur):
        nota = min(PERCUSION, key=lambda p: (p[0] != tipo, abs(p[1] - frec)))[2]
        ref = {38: .13, 37: .08, 42: .028, 46: .025, 76: .09, 82: .008}.get(nota, .1)
        capa = self._capa(buf)
        t = max(0.0, t)
        capa[1].append((t, "on", 9, nota, self._vel(vol, ref, 30, 118)))
        capa[1].append((t + max(dur, .05), "off", 9, nota, 0))

    def volcar(self, ganancias):
        """Renderiza cada capa y la suma a su buffer. ganancias: id(buffer) → factor (nivela con los sintes)."""
        for k, (buf, eventos, _) in self.capas.items():
            buf += _render(eventos, len(buf)) * ganancias.get(k, 1.0)
