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
"""
import json
import random
import subprocess
import sys
import time
from pathlib import Path

D = Path(sys.argv[1])
ACTUAL, SALTAR, ORDEN = D / "actual.json", D / "saltar", D / "orden.json"
ALEATORIO = D / "aleatorio"
MUESTRA = 4                      # bytes por muestra estéreo s16le
SILENCIO = bytes(44100 * MUESTRA // 10)   # 0,1 s
out = sys.stdout.buffer


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


def reproducir(p):
    """Envía una canción entera (o hasta que se pida saltar)."""
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
    finally:
        hijo.kill()
        hijo.wait()


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
