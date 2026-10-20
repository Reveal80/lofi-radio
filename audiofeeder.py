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
  infinita.json  {"activo", "estilos", "rotar_min", "bpm", "textura", "variedad"…}: radio infinita. Si está activa,
               en vez de la lista se compone música procedural en tiempo real: piezas de 3 a 6 minutos
               (piezas.Compositor) emitidas sección a sección (intro, A, B, descanso, cierre…); los tramos
               se superponen (la resonancia de uno sobre el principio del siguiente).
               El render corre en un proceso aparte para no retrasar nunca la salida de audio.
               Cambiar estilo o tempo pide una pieza nueva que continúa en la misma tonalidad y tempo;
               «saltar» pasa a otra pieza. memoria.json guarda las huellas de las piezas recientes.
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
_infinita_mtime = None


def leer_infinita():
    """Ajustes de la radio infinita si está activa; None si toca la lista. Solo se vuelve a leer el
    fichero si ha cambiado su fecha (se consulta hasta 10 veces por segundo).
    Si el fichero no se puede leer en ese instante, vale lo último leído."""
    global _infinita_cache, _infinita_mtime
    try:
        m = INFINITA.stat().st_mtime_ns
        if m != _infinita_mtime:
            d = json.loads(INFINITA.read_text())
            _infinita_cache, _infinita_mtime = (d if d.get("activo") else None), m
    except FileNotFoundError:
        _infinita_cache = _infinita_mtime = None
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
    """Proceso aparte: compone y renderiza secciones a petición (numpy/scipy no frenan la salida).
    El compositor guarda la pieza en curso y la memoria de piezas recientes."""
    import traceback
    import numpy as np
    import piezas
    comp = piezas.Compositor(str(D / "memoria.json"))
    while True:
        item = req.get()
        if item is None:
            return
        rid, estilo, bpm, fondo, opc, seed = item
        try:
            info, cuerpo, cola = comp.tramo(estilo, bpm, fondo, opc, seed)
            res.put((rid, info, cuerpo.astype(np.float32), cola.astype(np.float32)))
        except Exception as e:   # un tramo fallido no tumba la radio: se pide otro
            print(f"radio infinita: error al componer: {e}\n{traceback.format_exc()}", file=sys.stderr)
            comp.pieza = None
            res.put((rid, None, None, None))


class RadioInfinita:
    def __init__(self):
        self.proc = self.req = self.res = None
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

    def pedir(self, estilo, bpm, fondo=0, opc=None):
        self.arrancar()
        self.rid += 1
        self.seed += 1
        self.req.put((self.rid, estilo, int(bpm or 0), float(fondo), dict(opc or {}), self.seed))

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
    estilos = [e for e in (cfg.get("estilos") or ["aleatorio"]) if e]
    if len(estilos) > 1 and not int(cfg.get("rotar_min") or 0):   # rotar_min 0 = mezclar los estilos
        return "+".join(estilos), desde or time.time()
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
    nombres = {}
    NOMBRES_INST = []
    try:
        import lofigen
        nombres = {k: e["nombre"] for k, e in lofigen.ESTILOS.items()}
        NOMBRES_INST = list(lofigen.INST_NOMBRES)
    except Exception:
        pass
    radio = RadioInfinita()
    estilo, desde = _elegir_estilo(cfg, None, 0, False)
    bpm = int(cfg.get("bpm") or 0)
    fondo = lambda c: .35 if c.get("textura") else 0   # textura de fondo opcional, y baja
    AZAR_MIN = 30                                      # el aleatorio de instrumentos/melodía cambia cada 30 min
    azar = {"inst": (None, 0), "mel": (None, 0)}       # (elección, hasta cuándo vale)
    def inst(c):
        """Opciones para el compositor. Con «Aleatorio» se sortea una combinación y se mantiene 30 minutos."""
        quitar, anadir = tuple(c.get("inst_quitar") or ()), tuple(c.get("inst_anadir") or ())
        melodia = int(c.get("melodia", 1))
        ahora = time.time()
        if c.get("inst_aleatorio"):
            eleccion, hasta = azar["inst"]
            if eleccion is None or ahora >= hasta:
                posibles = [k for k in NOMBRES_INST if k not in quitar] or list(NOMBRES_INST)
                eleccion = tuple(sorted(random.sample(posibles, min(len(posibles), random.randint(2, 4)))))
                azar["inst"] = (eleccion, ahora + AZAR_MIN * 60)
            quitar = tuple(k for k in NOMBRES_INST if k not in eleccion)   # solo suenan los sorteados
            anadir = eleccion
        else:
            azar["inst"] = (None, 0)
        if c.get("melodia_aleatoria"):
            nivel, hasta = azar["mel"]
            if nivel is None or ahora >= hasta:
                nivel = random.choice([0, 1, 1, 2, 2, 3])
                azar["mel"] = (nivel, ahora + AZAR_MIN * 60)
            melodia = nivel
        else:
            azar["mel"] = (None, 0)
        return {"quitar": quitar, "anadir": anadir, "melodia": melodia, "variedad": str(c.get("variedad") or "medio")}
    fon, ins = fondo(cfg), inst(cfg)
    radio.pedir(estilo, bpm, fon, ins)
    paso = SR // 10                                    # 0,1 s por escritura
    entrada_suave = False                              # el próximo tramo entra con fundido (cambio pedido)
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
                if (e2, int(cfg.get("bpm") or 0), fondo(cfg), inst(cfg)) != (estilo, bpm, fon, ins):
                    estilo, bpm, fon, ins = e2, int(cfg.get("bpm") or 0), fondo(cfg), inst(cfg); radio.pedir(estilo, bpm, fon, ins)
                listo = radio.recoger(timeout=.05)
            tema, cuerpo, cola = listo
            compas = int(4 * 60 / tema["bpm"] * SR)
            if entrada_suave:                          # tras un cambio pedido: entra en dos compases
                m = min(2 * compas, len(cuerpo))
                cuerpo[:m] *= np.linspace(.15, 1, m, dtype=np.float32) ** 1.5
                entrada_suave = False
            if cola_pendiente is not None and len(cola_pendiente):
                n = min(len(cola_pendiente), len(cuerpo))
                cuerpo[:n] += cola_pendiente[:n]
            cola_pendiente = cola
            radio.n += 1
            estilos = cfg.get("estilos") or [estilo]
            rotar_min = max(1, int(cfg.get("rotar_min") or 60))
            # el siguiente se compone mientras suena este (y es donde rota el estilo si toca)
            estilo_sig, desde_sig = _elegir_estilo(cfg, estilo, desde, True)
            radio.pedir(estilo_sig, bpm, fon, ins)
            escribir_actual({"name": "radio-infinita", "inicio": time.time(), "dur": len(cuerpo) / SR,
                             "infinita": {"n": radio.n, "estilo": estilo,
                                          "estilo_nombre": " + ".join(nombres.get(x, x) for x in estilo.split("+")),
                                          "bpm": tema["bpm"], "inst": tema.get("inst"),
                                          "instMel": tema.get("instMel"), "bat": tema.get("bat"), "modo": tema.get("modo"),
                                          "siguiente": estilo_sig if estilo_sig != estilo else None,
                                          "solo": bool(tema.get("solo_melodia")),
                                          "azar_inst": list(azar["inst"][0] or []), "azar_inst_hasta": azar["inst"][1],
                                          "azar_mel": azar["mel"][0], "azar_mel_hasta": azar["mel"][1],
                                          "pieza": tema.get("pieza"),
                                          "rota_en": (desde + rotar_min * 60) if len(estilos) > 1 and "+" not in estilo else 0}})
            estilo, desde = estilo_sig, desde_sig
            pos, ultimo, corte_en, nuevo, suave = 0, time.time(), None, None, False
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
                    b2, f2, i2 = int(cfg.get("bpm") or 0), fondo(cfg), inst(cfg)
                    salto = SALTAR.exists()
                    fuerte = lambda d: {k: v for k, v in d.items() if k not in ("melodia", "variedad")}
                    if (e2, b2, f2, fuerte(i2)) != (estilo, bpm, fon, fuerte(ins)) or salto:
                        SALTAR.unlink(missing_ok=True)
                        estilo, bpm, fon, ins = e2, b2, f2, i2
                        # cambio pedido (estilo, instrumentos, tempo): pieza nueva que continúa en la misma
                        # tonalidad y tempo, y entra al acabar la frase de 4 compases con fundido largo;
                        # «saltar» pasa a otra pieza y corta al compás siguiente
                        suave = not salto
                        radio.pedir(estilo, bpm, fon, {**ins, "minimo": suave, "salto": salto})
                        nuevo = False                  # esperando el tramo con los ajustes nuevos
                    elif i2 != ins:                    # solo melodía o variedad: la sección en curso sigue;
                        ins = i2                       # la siguiente se rehace con los ajustes nuevos
                        radio.pedir(estilo, bpm, fon, {**ins, "rehacer": True})
                if nuevo is False:
                    r = radio.recoger()
                    if r is not None:
                        nuevo = r
                        unidad = compas * 4 if suave else compas
                        corte_en = min(len(cuerpo), ((pos // unidad) + 1) * unidad)
                if corte_en is not None and pos >= corte_en:
                    largo = 2 * compas if suave else 4 * SR
                    resto = cuerpo[pos:pos + largo]
                    cola_pendiente = resto * np.linspace(1, 0, len(resto), dtype=np.float32) ** .7
                    entrada_suave = suave
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
        cfg = leer_infinita()
        if cfg:
            tocar_infinita(cfg)
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
