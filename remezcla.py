"""Remezcla infinita de los MP3 de la biblioteca (2.9).

El usuario trajo cinco canciones lofi de Pixabay y pidió «continuar estos mismos MP3 de manera
infinita con contenido aleatorio». Esto no compone: analiza cada canción una vez (tempo, rejilla
de compases, nivel por compás) y después emite **segmentos** de 8 a 16 compases alineados a frases de
4, encadenados sin cortes: la mayoría de las veces sigue la canción en línea, otras salta a otra
frase de la misma canción, y de vez en cuando cambia de canción. Los saltos se funden en un compás
(fundido de igual potencia) y cada canción se nivela a -14 dBFS para que no haya saltos de volumen.
Evita intros y finales muy flojos salvo para empezar o cerrar una canción.

Mismo contrato que piezas.Compositor.tramo: devuelve (info, cuerpo float32 mono, cola float32).
"""
import json
import os
import subprocess
import time

import numpy as np
from scipy import signal

SR = 44100


def _decodificar(path, inicio, dur, sr=SR):
    """Trozo de la canción como float32 mono (inicio y dur en segundos)."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, inicio):.4f}", "-t", f"{dur:.4f}", "-i", str(path),
           "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
    x = np.frombuffer(r.stdout, dtype=np.float32).copy()
    n = int(dur * sr)
    if len(x) < n:
        x = np.concatenate([x, np.zeros(n - len(x), dtype=np.float32)])
    return x[:n]


def _titulo(path):
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=title", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, timeout=10)
        t = r.stdout.strip()
        return t.split(" by ")[0] if t else os.path.basename(path)
    except Exception:
        return os.path.basename(path)


def analizar(path):
    """Tempo, rejilla de compases (fase del primer tiempo) y nivel por compás de una canción."""
    sr = 11025
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
                       stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
    x = np.frombuffer(r.stdout, dtype=np.float32)
    dur = len(x) / sr
    b, a = signal.butter(2, 150 / (sr / 2), "low")
    lo = signal.lfilter(b, a, x)
    hop = 128
    env = np.sqrt(np.convolve(lo ** 2, np.ones(256) / 256, "same"))[::hop]
    onset = np.maximum(np.diff(env), 0)
    onset = onset - onset.mean()
    fps = sr / hop
    seg = onset[:int(fps * 90)]
    ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
    # tempo entre 60 y 100 (el lofi vive ahí; evita dobles y mitades), afinado a 0,1 bpm
    bpm = max(range(60, 100), key=lambda v: ac[int(round(60 * fps / v))])
    bpm = float(max(np.arange(bpm - 1, bpm + 1, .1), key=lambda v: ac[int(round(60 * fps / v))]))
    compas = 240 / bpm
    n = int(dur / compas)
    mejor = (0, 0.0)
    for k in range(64):   # fase: donde más bombo cae en el tiempo 1
        off = k / 64 * compas
        idx = ((np.arange(n) * compas + off) * fps).astype(int)
        idx = idx[idx < len(env)]
        e = float(env[idx].mean()) if len(idx) else 0
        if e > mejor[0]:
            mejor = (e, off)
    off = mejor[1]
    n = int((dur - off) / compas)
    niveles = []
    for i in range(n):
        tr = x[int((off + i * compas) * sr):int((off + (i + 1) * compas) * sr)]
        niveles.append(round(20 * float(np.log10(np.sqrt(np.mean(tr ** 2)) + 1e-9)), 1))
    rms = 20 * float(np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9))
    return dict(bpm=round(bpm, 1), compas=compas, fase=off, compases=n, niveles=niveles, rms=rms, pico=float(np.max(np.abs(x)) + 1e-6),
                dur=dur, titulo=_titulo(path))


class Remezcla:
    def __init__(self, carpeta, cache_path=None):
        self.carpeta = carpeta
        self.cache_path = cache_path
        self.cache = {}
        if cache_path:
            try:
                self.cache = json.loads(open(cache_path).read())
            except Exception:
                self.cache = {}
        self.actual = None      # (nombre, análisis)
        self.compas = 0         # siguiente compás a emitir de la canción actual
        self.n = 0              # segmentos emitidos
        self.pieza_n = 0        # canciones encadenadas
        self.ultimo_nombre = None
        self.pendiente = "cambia"   # qué hará el siguiente segmento (sigue / salta / cambia)
        self.emitidos = 0.0     # segundos emitidos de la canción actual

    def _canciones(self):
        return sorted(f for f in os.listdir(self.carpeta) if f.lower().endswith(".mp3"))

    def _analisis(self, nombre):
        path = os.path.join(self.carpeta, nombre)
        clave = f"{nombre}:{int(os.path.getmtime(path))}"
        if clave not in self.cache:
            self.cache = {k: v for k, v in self.cache.items() if not k.startswith(nombre + ":")}
            self.cache[clave] = analizar(path)
            if self.cache_path:
                try:
                    open(self.cache_path, "w").write(json.dumps(self.cache))
                except Exception:
                    pass
        return self.cache[clave]

    def _elegir_cancion(self, rng, evitar=None):
        lista = [c for c in self._canciones() if c != evitar] or self._canciones()
        if not lista:
            raise RuntimeError("no hay MP3 en la biblioteca para remezclar")
        nombre = str(rng.choice(lista))
        return nombre, self._analisis(nombre)

    def _buenos(self, A):
        """Compases con nivel normal (ni intro ni final flojos), alineados a 4."""
        niv = np.array(A["niveles"])
        if not len(niv):
            return [0]
        mediana = float(np.median(niv))
        return [i for i in range(0, A["compases"] - 4, 4) if niv[i:i + 4].mean() > mediana - 6]

    def _decidir(self, rng, variedad):
        """Qué hará el segmento siguiente: seguir en línea, saltar a otra frase o cambiar de canción."""
        p_salto, p_cambio = {"bajo": (.12, .06), "medio": (.25, .12), "alto": (.4, .25)}.get(variedad, (.25, .12))
        A = self.actual[1]
        quedan = A["compases"] - self.compas
        r = rng.random()
        if quedan < 8 or r < p_cambio:
            return "cambia"
        if r < p_cambio + p_salto:
            return "salta"
        return "sigue"

    def tramo(self, opc, seed):
        rng = np.random.default_rng(seed)
        variedad = str(opc.get("variedad", "medio"))
        accion = "cambia" if (self.actual is None or opc.get("salto")) else self.pendiente
        A = self.actual[1] if self.actual else None
        if accion == "salta":
            buenos = [i for i in self._buenos(A) if abs(i - self.compas) >= 8 and i + 8 <= A["compases"]]
            if buenos:
                self.compas = int(rng.choice(buenos))
            else:
                accion = "cambia"
        if accion == "cambia":
            nombre, A = self._elegir_cancion(rng, evitar=self.actual[0] if self.actual else None)
            self.actual = (nombre, A)
            buenos = self._buenos(A)
            self.compas = 0 if (rng.random() < .5 or not buenos) else int(rng.choice(buenos))
            self.pieza_n += 1
            self.emitidos = 0.0
        fundir = accion != "sigue" and self.n > 0
        nombre, A = self.actual
        largo = int(rng.choice([8, 8, 12, 16]))
        largo = max(4, min(largo, A["compases"] - self.compas))
        compas_s = A["compas"]
        inicio = A["fase"] + self.compas * compas_s
        cuerpo = _decodificar(os.path.join(self.carpeta, nombre), inicio, largo * compas_s + compas_s)
        cuerpo *= min(10 ** ((-14 - A["rms"]) / 20), .95 / A.get("pico", 1.0))   # a -14 dBFS sin pasar de 0,95 de pico
        n_c = int(largo * compas_s * SR)
        cola, cuerpo = cuerpo[n_c:], cuerpo[:n_c]
        n_f = min(len(cola), int(compas_s * SR))
        if fundir:   # entra con fundido de igual potencia sobre la cola (el compás que seguía) del anterior
            cuerpo[:n_f] *= np.sin(np.linspace(0, np.pi / 2, n_f, dtype=np.float32))
        desde = self.compas
        self.compas += largo
        self.n += 1
        antes = self.emitidos
        self.emitidos += largo * compas_s
        # lo que hará el siguiente: si sigue en línea, empieza justo donde acaba este y no hace falta cola;
        # si salta o cambia, la cola es el compás que seguía, apagándose
        self.pendiente = self._decidir(rng, variedad)
        cola = cola[:n_f] * np.cos(np.linspace(0, np.pi / 2, n_f, dtype=np.float32)) if self.pendiente != "sigue" else np.zeros(0, dtype=np.float32)
        info = dict(bpm=int(round(A["bpm"])), inst="remezcla", instMel=A["titulo"], bat="original", modo="", tex="", solo_melodia=False,
                    cambios=[accion],
                    pieza=dict(n=self.pieza_n, perfil="remezcla", perfil_nombre=A["titulo"], tono=nombre, variedad=variedad,
                               dur=round(A["dur"], 1), antes=round(A["fase"] + desde * compas_s, 1), seccion=f"compases {desde + 1}-{desde + largo}",
                               idx=self.n, total=0, forma=accion, capa=None, inst2=None, comp="", bajo="", familia="remezcla", parecido=0,
                               remezcla=True, siguiente=self.pendiente))
        return info, cuerpo, cola


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        t = time.time()
        A = analizar(f)
        print(os.path.basename(f), {k: v for k, v in A.items() if k != "niveles"}, f"({time.time() - t:.1f} s)")
