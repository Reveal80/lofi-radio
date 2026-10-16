"""Alimentador de video de la emisión.

Reproduce la cola de videos (data/videos/cola.json) en orden y en bucle: cada
video se repite durante las horas indicadas y después pasa al siguiente.
Escribe fotogramas en bruto (yuv420p, 1920x1080, 30 fps) por la salida
estándar, así que el codificador de la emisión nunca se reinicia: cambiar de
video, editar la cola o saltar no corta el directo.

Ficheros de control (en la carpeta de videos):
  cola.json    [{"id", "video", "hours", "activo"}, ...] — se relee en caliente;
               los que tienen activo=false no se emiten
  actual.json  lo que está sonando ahora (lo escribe este proceso)
  saltar       si existe, pasa al siguiente video y lo borra
"""
import json
import subprocess
import sys
import time
from pathlib import Path

W, H, FPS = 1920, 1080, 30
FRAME = W * H * 3 // 2
D = Path(sys.argv[1])
COLA, ACTUAL, SALTAR = D / "cola.json", D / "actual.json", D / "saltar"
NEGRO = bytes([16]) * (W * H) + bytes([128]) * (W * H // 2)
out = sys.stdout.buffer


def leer_cola():
    try:
        # solo los videos con «en bucle» activado se emiten
        return [c for c in json.loads(COLA.read_text()) if c.get("video") and c.get("activo") is not False]
    except Exception:
        return []


def escribir_actual(datos):
    tmp = ACTUAL.with_suffix(".tmp")
    tmp.write_text(json.dumps(datos))
    tmp.replace(ACTUAL)


def segundo_en_negro():
    """Sin videos disponibles: un segundo de negro para no cortar la emisión."""
    escribir_actual({})
    for _ in range(FPS):
        out.write(NEGRO)
        time.sleep(1 / FPS)


def reproducir(item):
    """Repite un video hasta cumplir sus horas, saltar o desaparecer de la cola."""
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
          f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={FPS},format=yuv420p")
    hijo = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-re", "-stream_loop", "-1", "-i", str(D / item["video"]),
         "-an", "-vf", vf, "-f", "rawvideo", "-"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
    horas = float(item.get("hours") or 1)
    escribir_actual({"id": item["id"], "video": item["video"], "inicio": time.time(), "horas": horas})
    buf = bytearray(FRAME)
    vista = memoryview(buf)
    fotogramas = 0
    mtime = None
    try:
        while True:
            # solo se reenvían fotogramas completos: cortar a medias descuadraría la imagen
            n = 0
            while n < FRAME:
                r = hijo.stdout.readinto(vista[n:])
                if not r:
                    return fotogramas > 0
                n += r
            out.write(buf)
            fotogramas += 1
            if fotogramas % FPS:
                continue
            # una vez por segundo: ¿saltar?, ¿han cambiado las horas o la cola?
            if SALTAR.exists():
                SALTAR.unlink(missing_ok=True)
                return True
            try:
                m = COLA.stat().st_mtime
            except FileNotFoundError:
                return True
            if m != mtime:
                mtime = m
                igual = next((c for c in leer_cola() if c["id"] == item["id"]), None)
                if igual is None:
                    return True
                if float(igual.get("hours") or 1) != horas:
                    horas = float(igual.get("hours") or 1)
                    escribir_actual({"id": item["id"], "video": item["video"],
                                     "inicio": time.time() - fotogramas / FPS, "horas": horas})
            if fotogramas / FPS >= horas * 3600:
                return True
    finally:
        hijo.kill()
        hijo.wait()


def main():
    SALTAR.unlink(missing_ok=True)
    actual_id, pos = None, -1
    fallos = 0
    while True:
        cola = leer_cola()
        if not cola:
            segundo_en_negro()
            continue
        ids = [c["id"] for c in cola]
        # el siguiente al que sonaba; si se acaba de quitar de la cola, el que ocupa su sitio
        pos = (ids.index(actual_id) + 1) if actual_id in ids else (pos if actual_id else 0)
        pos %= len(cola)
        item = cola[pos]
        actual_id = item["id"]
        if not (D / item["video"]).exists() or not reproducir(item):
            fallos += 1
            if fallos >= len(cola):   # ningún video de la cola se puede reproducir
                segundo_en_negro()
                fallos = 0
            continue
        fallos = 0


if __name__ == "__main__":
    try:
        main()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
