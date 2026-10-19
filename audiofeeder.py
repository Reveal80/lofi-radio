"""Alimentador de audio de la emisión.

Reproduce las canciones de data/tracks en el orden de la lista (orden.json) y
en bucle infinito. Antes de cada canción vuelve a leer el orden y la carpeta,
así que reordenar, añadir o borrar se aplica en directo sin cortar.
Escribe PCM s16le 44,1 kHz estéreo por la salida estándar, así que el
codificador de la emisión nunca se reinicia.

Ficheros de control (en la carpeta de canciones):
  orden.json   orden de emisión (lo escribe el panel); las nuevas van al final
  actual.json  canción que suena ahora (la escribe este proceso)
  saltar       si existe, pasa a la siguiente canción y lo borra
  aleatorio    si existe, la siguiente canción se elige al azar (modo aleatorio del panel)
  infinita.json  {"activo", "estilos", "rotar_min", "bpm"}: radio infinita. Si está activa, en vez de la lista
               se compone música procedural en tiempo real (lofigen.tema_siguiente) por tramos de
               16 compases que se superponen (la resonancia de uno sobre el principio del siguiente).
               El render corre en un proceso aparte para no retrasar nunca la salida de audio.
               Cambiar estilo o tempo, o «saltar», pide un tramo nuevo y corta al siguiente compás.
"""
import json
import multiprocessing
import random
import subprocess
import sys
import time
from pathlib import Path

D = Path(sys.argv[1])
ACTUAL, SALTAR, ORDEN = D / "actual.json", D / "saltar", D / "orden.json"
ALEATORIO = D / "aleatorio"
INFINITA = D / "infinita.json"
SR = 44100
MUESTRA = 4                      # bytes por muestra estéreo s16le
SILENCIO = bytes(SR * MUESTRA // 10)   # 0,1 s
out = sys.stdout.buffer
cola_pendiente = None            # resonancia (float32 mono) que se superpone a lo que suene después


def escribir_actual(datos):
    tmp = ACTUAL.with_suffix(".tmp")
    tmp.write_text(json.dumps(datos))
    tmp.replace(ACTUAL)


def duracion(p):
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(p)], capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


_infinita_cache = None


def leer_infinita():
    """Ajustes de la radio infinita si está activa; None si toca la lista.
    Si el fichero no se puede leer en ese instante, vale lo último leído."""
    global _infinita_cache
    try:
        d = json.loads(INFINITA.read_text())
        _infinita_cache = d if d.get("activo") else None
    except FileNotFoundError:
        _infinita_cache = None
    except Exception:
        pass
    return _infinita_cache


def reproducir(p):
    """Envía una canción entera (o hasta que se pida saltar o se active la radio infinita)."""
    global cola_pendiente
    hijo = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(p), "-f", "s16le",
                             "-ar", "44100", "-ac", "2", "-"],
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
    escribir_actual({"name": p.name, "inicio": time.time(), "dur": duracion(p)})
    buf = bytearray(16384)
    vista = memoryview(buf)
    resto = b""
    ultimo = time.time()
    try:
        while True:
            n = hijo.stdout.readinto(buf)
            if not n:
                return
            datos = resto + bytes(vista[:n])
            corte = len(datos) - len(datos) % MUESTRA   # nunca partir una muestra
            out.write(datos[:corte])
            resto = datos[corte:]
            if time.time() - ultimo > .3:
                ultimo = time.time()
                if SALTAR.exists():
                    SALTAR.unlink(missing_ok=True)
                    return
                if leer_infinita():   # la canción se apaga en 1,5 s y la radio infinita entra encima
                    import numpy as np
                    extra = hijo.stdout.read(int(1.5 * SR) * MUESTRA)
                    pcm = np.frombuffer(extra[:len(extra) - len(extra) % MUESTRA], dtype=np.int16).astype(np.float32) / 32768
                    mono = pcm.reshape(-1, 2).mean(axis=1) if len(pcm) else pcm
                    cola_pendiente = mono * np.linspace(1, 0, len(mono), dtype=np.float32)
                    return
    finally:
        hijo.kill()
        hijo.wait()


# ── radio infinita ──

def _render_worker(req, res):
    """Proceso aparte: compone y renderiza tramos a petición (numpy/scipy no frenan la salida)."""
    import numpy as np
    import lofigen
    while True:
        item = req.get()
        if item is None:
            return
        rid, prev, estilo, bpm, seed = item
        try:
            tema = lofigen.tema_siguiente(prev, seed, estilo, bpm or None)
            cuerpo, cola = lofigen.render_theme(tema, 1, seed, continuo=True)
            res.put((rid, tema, cuerpo.astype(np.float32), cola.astype(np.float32)))
        except Exception as e:   # un tramo fallido no tumba la radio: se pide otro
            print(f"radio infinita: error al componer: {e}", file=sys.stderr)
            res.put((rid, None, None, None))


class RadioInfinita:
    def __init__(self):
        self.proc = self.req = self.res = None
        self.tema = None          # último tramo compuesto (el siguiente hereda de él)
        self.n = 0                # tramos emitidos
        self.rid = 0              # id de la última petición (los resultados viejos se descartan)
        self.seed = random.randrange(1 << 30)

    def arrancar(self):
        if self.proc and self.proc.is_alive():
            return
        ctx = multiprocessing.get_context("fork")
        self.req, self.res = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=_render_worker, args=(self.req, self.res), daemon=True)
        self.proc.start()

    def parar(self):
        if self.proc:
            self.proc.terminate()
            self.proc.join(2)
        self.proc = None
        self.tema = None

    def pedir(self, estilo, bpm):
        self.arrancar()
        self.rid += 1
        self.seed += 1
        self.req.put((self.rid, self.tema, estilo, int(bpm or 0), self.seed))

    def recoger(self, timeout=0):
        """El tramo de la última petición si ya está (tema, cuerpo, cola); None si no."""
        import queue
        try:
            while True:
                rid, tema, cuerpo, cola = self.res.get(timeout=timeout)
                if rid != self.rid:
                    continue
                if tema is None:
                    return None
                self.tema = tema
                return tema, cuerpo, cola
        except queue.Empty:
            return None


def _a_pcm(mono):
    """float32 mono (-1..1) → bytes s16le estéreo. Por encima de 0,7 comprime suave (codo blando):
    donde se superponen dos tramos la suma puede pasar de 1 y así no recorta."""
    import numpy as np
    a = np.abs(mono)
    y = np.where(a > .7, np.sign(mono) * (.7 + (a - .7) * .3), mono)
    x = np.clip(y * 32767, -32768, 32767).astype(np.int16)
    return np.repeat(x, 2).tobytes()


def _elegir_estilo(cfg, actual, desde, rotar):
    """Estilo que toca según la selección (varios = rotación cada rotar_min minutos, en el orden
    de la lista). Si el actual ya no está marcado, pasa al primero. Devuelve (estilo, desde)."""
    estilos = [e for e in (cfg.get("estilos") or [cfg.get("estilo") or "aleatorio"]) if e]
    if actual not in estilos:
        return estilos[0], time.time()
    if rotar and len(estilos) > 1 and time.time() - desde >= max(1, int(cfg.get("rotar_min") or 60)) * 60:
        return estilos[(estilos.index(actual) + 1) % len(estilos)], time.time()
    return actual, desde


def tocar_infinita(cfg):
    """Emite la radio infinita hasta que se desactive. Entre tramos no hay cortes: la cola de
    resonancia del anterior (o el final apagado de la canción que sonaba) se suma al principio
    del siguiente. Cambios de estilo/tempo o «saltar» cortan al compás siguiente con un fundido;
    con varios estilos marcados, el estilo rota cada rotar_min minutos al empezar un tramo."""
    global cola_pendiente
    import numpy as np
    radio = RadioInfinita()
    estilo, desde = _elegir_estilo(cfg, None, 0, False)
    bpm = int(cfg.get("bpm") or 0)
    radio.pedir(estilo, bpm)
    paso = SR // 10                                    # 0,1 s por escritura
    nombres = {}
    try:
        import lofigen
        nombres = {k: e["nombre"] for k, e in lofigen.ESTILOS.items()}
    except Exception:
        pass
    siguiente = None                                   # tramo ya recogido que entra en el corte
    try:
        while True:
            listo, siguiente = siguiente or radio.recoger(timeout=.1), None
            while listo is None:                       # aún no hay tramo: suena la cola o silencio
                if cola_pendiente is not None and len(cola_pendiente):
                    out.write(_a_pcm(cola_pendiente[:paso])); cola_pendiente = cola_pendiente[paso:]
                else:
                    out.write(SILENCIO)
                cfg = leer_infinita()
                if not cfg:
                    return
                e2, desde = _elegir_estilo(cfg, estilo, desde, False)
                if (e2, int(cfg.get("bpm") or 0)) != (estilo, bpm):
                    estilo, bpm = e2, int(cfg.get("bpm") or 0); radio.pedir(estilo, bpm)
                listo = radio.recoger(timeout=.05)
            tema, cuerpo, cola = listo
            if cola_pendiente is not None and len(cola_pendiente):
                n = min(len(cola_pendiente), len(cuerpo))
                cuerpo[:n] += cola_pendiente[:n]
            cola_pendiente = cola
            radio.n += 1
            estilos = cfg.get("estilos") or [estilo]
            rotar_min = max(1, int(cfg.get("rotar_min") or 60))
            # el siguiente se compone mientras suena este (y es donde rota el estilo si toca)
            estilo_sig, desde_sig = _elegir_estilo(cfg, estilo, desde, True)
            radio.pedir(estilo_sig, bpm)
            compas = int(4 * 60 / tema["bpm"] * SR)
            escribir_actual({"name": "radio-infinita", "inicio": time.time(), "dur": len(cuerpo) / SR,
                             "infinita": {"n": radio.n, "estilo": estilo, "estilo_nombre": nombres.get(estilo, estilo),
                                          "estilos": estilos, "bpm": tema["bpm"], "inst": tema.get("inst"),
                                          "instMel": tema.get("instMel"), "bat": tema.get("bat"), "modo": tema.get("modo"),
                                          "tex": tema.get("tex"), "cambios": tema.get("cambios", []),
                                          "siguiente": estilo_sig if estilo_sig != estilo else None,
                                          "rota_en": (desde + rotar_min * 60) if len(estilos) > 1 else 0}})
            estilo, desde = estilo_sig, desde_sig
            pos, ultimo, corte_en, nuevo = 0, time.time(), None, None
            while pos < len(cuerpo):
                out.write(_a_pcm(cuerpo[pos:pos + paso])); pos += paso
                if nuevo is None and time.time() - ultimo > .3:
                    ultimo = time.time()
                    ahora = leer_infinita()
                    if not ahora:                      # vuelve la lista: se apaga en 1,5 s
                        fin = cuerpo[pos:pos + int(1.5 * SR)]
                        out.write(_a_pcm(fin * np.linspace(1, 0, len(fin), dtype=np.float32)))
                        cola_pendiente = None
                        return
                    cfg = ahora
                    e2, desde = _elegir_estilo(cfg, estilo, desde, False)
                    b2 = int(cfg.get("bpm") or 0)
                    if (e2, b2) != (estilo, bpm) or SALTAR.exists():
                        SALTAR.unlink(missing_ok=True)
                        estilo, bpm = e2, b2
                        radio.pedir(estilo, bpm)
                        nuevo = False                  # esperando el tramo con los ajustes nuevos
                if nuevo is False:
                    r = radio.recoger()
                    if r is not None:
                        nuevo = r
                        corte_en = min(len(cuerpo), ((pos // compas) + 1) * compas)   # al compás siguiente
                if corte_en is not None and pos >= corte_en:
                    resto = cuerpo[pos:pos + 4 * SR]
                    cola_pendiente = resto * np.linspace(1, 0, len(resto), dtype=np.float32)
                    break
            if nuevo:                                  # encadena ya con el tramo nuevo
                siguiente = nuevo
    finally:
        radio.parar()
        escribir_actual({})


def en_orden():
    """Misma regla que el panel: orden guardado y, al final, las nuevas por antigüedad."""
    try:
        orden = json.loads(ORDEN.read_text())
    except Exception:
        orden = []
    existen = {p.name: p for p in D.glob("*.mp3")}
    lista = [existen.pop(n) for n in orden if n in existen]
    return lista + sorted(existen.values(), key=lambda q: q.stat().st_mtime)


def main():
    SALTAR.unlink(missing_ok=True)
    actual, pos = None, 0
    escribir_actual({})
    while True:
        if leer_infinita():
            tocar_infinita(leer_infinita())
            actual = None
            continue
        pistas = en_orden()
        if not pistas:
            if actual is not None:
                escribir_actual({})
                actual = None
            out.write(SILENCIO)
            time.sleep(.1)
            continue
        nombres = [p.name for p in pistas]
        # la siguiente a la que acaba de sonar (aunque se haya movido en la lista)
        # (si la que sonaba se ha borrado, la que ocupa ahora su sitio)
        pos = (nombres.index(actual) + 1) if actual in nombres else (pos if actual else 0)
        pos %= len(pistas)
        if ALEATORIO.exists() and len(pistas) > 1:   # al azar, sin repetir la que acaba de sonar
            pos = random.choice([k for k in range(len(pistas)) if nombres[k] != actual])
        actual = nombres[pos]
        reproducir(pistas[pos])


if __name__ == "__main__":
    try:
        main()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
