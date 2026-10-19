"""Panel de gestión de la radio lofi 24/7.

Compone música, crea y guarda videos de fondo y controla la emisión
ffmpeg → YouTube (y copia a MediaMTX para verla en la red local).
"""
import json
import logging
import os
import random
import re
import secrets
import shutil
import signal as sig
import subprocess
import threading
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

import escenas
import lofigen

VERSION = "2.3"   # sube 0,1 con cada cambio de código de la imagen (se muestra en la barra superior del panel)
BASE = Path(__file__).parent
TRACKS_DIR = BASE / "data" / "tracks"
VIDEOS_DIR = BASE / "data" / "videos"    # biblioteca de videos + cola
LOGS_DIR = BASE / "logs"
for d in (TRACKS_DIR, VIDEOS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)
# CONFIG_FILE puede apuntar a otra ruta (la imagen Docker la guarda en data/ para que persista)
CONFIG_FILE = Path(os.environ.get("CONFIG_FILE", BASE / "config.json"))
COLA_FILE = VIDEOS_DIR / "cola.json"
TITULOS_FILE = VIDEOS_DIR / "titulos.json"     # nombre visible de cada video (el archivo no cambia)
TITULOS_MUSICA = TRACKS_DIR / "titulos.json"   # títulos editados de las canciones
ORDEN_MUSICA = TRACKS_DIR / "orden.json"       # orden de emisión de las canciones
STREAM_LOG = LOGS_DIR / "stream.log"
STREAM_LOG_ANTERIOR = LOGS_DIR / "stream.log.1"   # registro de la emisión anterior (para analizar caídas)
WATCHDOG_LOG = LOGS_DIR / "watchdog.log"
PREVIEW_URL = os.environ.get("PREVIEW_RTMP", "rtmp://mediamtx:1935/radio")
YOUTUBE_RTMP = os.environ.get("YOUTUBE_RTMP", "rtmp://a.rtmp.youtube.com/live2")   # cambiable solo para pruebas
# Salidas del tee con cola propia (fifo) y reconexión automática ilimitada. Sin esto, un corte
# silencioso de la red (la conexión TCP no da error, solo deja de avanzar) bloqueaba al codificador
# entero: la vista previa se caía, YouTube daba la emisión por terminada y ffmpeg nunca reconectaba.
# rw_timeout (por salida): una conexión o escritura que no avanza en 10 s es un error → el fifo
# reconecta cada 2 s, para siempre, sin parar el codificador ni la otra salida. Comprobado con un
# servidor RTMP pausado (docker pause): la previa sigue y al reanudar hay una publicación nueva.
FIFO_OPTS = "attempt_recovery=1:drop_pkts_on_overflow=1:restart_with_keyframe=1:recovery_wait_time=2:max_recovery_attempts=0"
RW_TIMEOUT = "rw_timeout=10000000"

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 1024 ** 3  # subidas de hasta 1 GB
logging.getLogger("werkzeug").setLevel(logging.WARNING)   # sin una línea por cada sondeo del panel

gen_state = {"running": False, "done": 0, "total": 0, "current": "", "error": ""}
video_state = {"running": False, "error": "", "done": 0, "total": 0, "tipo": "", "destino": ""}
stream = {"proc": None, "mode": "", "log_fh": None, "inicio": 0}
watchdog = {"reinicios": 0, "ultimo": ""}
_lock = threading.Lock()            # estados de los trabajos en segundo plano
_io_lock = threading.RLock()        # lectura-modificación-escritura de los JSON
_stream_lock = threading.RLock()    # arrancar/parar la emisión (panel y vigilante)

# lo que el panel muestra y nunca cambia mientras corre el servidor
ESTILOS_PUB = {k: {"nombre": e["nombre"], "desc": e["desc"]} for k, e in lofigen.ESTILOS.items()}


# --------------------------------------------------------------- JSON en disco
# El panel consulta /api/state cada 3 s: los JSON se cachean por fecha de
# modificación y se escriben de forma atómica (los alimentadores los leen en caliente).

_jcache = {}


def leer_json(p, defecto):
    try:
        m = p.stat().st_mtime_ns
    except OSError:
        return defecto
    c = _jcache.get(p)
    if not c or c[0] != m:
        try:
            c = _jcache[p] = (m, json.loads(p.read_text()))
        except Exception:
            return defecto
    return c[1]


def escribir_json(p, datos):
    tmp = p.with_name(f".{p.name}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(datos, ensure_ascii=False))
    tmp.replace(p)


def load_config():
    return dict(leer_json(CONFIG_FILE, {}))


def actualizar_config(**cambios):
    with _io_lock:
        cfg = load_config()
        cfg.update(cambios)
        escribir_json(CONFIG_FILE, cfg)


def slug(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def libre(carpeta, base, ext, defecto):
    """Ruta libre en la carpeta a partir de un título: base.ext, base-2.ext…"""
    base = slug(base)[:60] or defecto
    p, k = carpeta / f"{base}{ext}", 2
    while p.exists():
        p, k = carpeta / f"{base}-{k}{ext}", k + 1
    return p


def load_titulos(f=TITULOS_FILE):
    return leer_json(f, {})


def set_titulo(archivo, titulo, f=TITULOS_FILE):
    with _io_lock:
        t = dict(load_titulos(f))
        if titulo:
            t[archivo] = titulo
        else:
            t.pop(archivo, None)
        escribir_json(f, t)


# ---------------------------------------------------------------- metadatos
# ffprobe solo una vez por archivo y versión (también se recuerdan los fallos,
# para no relanzarlo cada 3 s con un archivo roto).

_meta = {}


def _cacheado(p, calcular):
    try:
        clave = (p.name, p.stat().st_mtime_ns)
    except OSError:
        return None
    if clave not in _meta:
        try:
            _meta[clave] = calcular(p)
        except Exception:
            _meta[clave] = None
    return _meta[clave]


def _probe_track(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration:format_tags=title,album", "-of", "json", str(p)],
                         capture_output=True, text=True, timeout=10)
    fmt = json.loads(out.stdout)["format"]
    tags = {k.lower(): v for k, v in fmt.get("tags", {}).items()}
    return {"dur": float(fmt.get("duration", 0)),
            "title": tags.get("title") or p.name[:-4].replace("-", " ").capitalize(),
            "tema": (tags.get("album") or "").replace("Tema ", "")}


def track_meta(p):
    """Duración, título y tema desde los metadatos del MP3."""
    return _cacheado(p, _probe_track) or {"dur": 0.0, "title": p.name[:-4], "tema": ""}


def video_meta(p):
    return _cacheado(p, lambda q: float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(q)],
        capture_output=True, text=True, timeout=10).stdout.strip())) or 0.0


# ------------------------------------------------------------- generación

_T_COSAS = ["el café", "la libreta", "un píxel", "la siesta", "el bus nocturno",
            "la tostada", "el casete", "un gato", "la consola", "el paraguas",
            "la nevera", "el examen", "un marcapáginas", "el mando", "la luna"]
_T_ESTADOS = ["en modo avión", "a medio guardar", "sin conexión", "en bucle",
              "fuera de plano", "con lag", "en versión beta", "a 8 bits",
              "en pausa", "sin subtítulos", "a cámara lenta", "del turno de noche"]
_T_FRASES = ["Partida guardada en {c}", "Ultimo nivel antes de cenar",
             "Checkpoint en la cocina", "Se me olvidó el punto de guardado",
             "Capítulo {n} de no hacer nada", "Vida extra para el lunes",
             "Modo historia de mi sofá", "Final alternativo del martes",
             "Logro desbloqueado: madrugar", "Pantalla de carga infinita"]


def titulo_random(rng):
    r = rng.random()
    if r < 0.4:
        return f"{rng.choice(_T_COSAS).capitalize()} {rng.choice(_T_ESTADOS)}"
    if r < 0.7:
        f = rng.choice(_T_FRASES)
        return f.format(c=rng.choice(_T_COSAS), n=rng.randrange(2, 99))
    return f"{rng.choice(_T_COSAS).capitalize()} y {rng.choice(_T_COSAS)} {rng.choice(_T_ESTADOS)}"


def gen_job(count, estilo="aleatorio"):
    rng = random.Random()
    try:
        for i in range(count):
            titulo = titulo_random(rng)
            mp3 = libre(TRACKS_DIR, titulo, ".mp3", "cancion")
            parcial, wav = mp3.with_suffix(".part"), None
            gen_state["current"] = f"componiendo «{titulo}» ({i + 1}/{count})"
            try:
                wav, tema = lofigen.generate_random(random.randrange(1, 1_000_000), TRACKS_DIR, estilo)
                etiqueta = f"{tema['inst']} · {tema['bat']} · {tema.get('modo', 'mayor')} · {tema['bpm']} bpm"
                if estilo != "aleatorio":
                    etiqueta = f"{lofigen.ESTILOS[estilo]['nombre']} · {etiqueta}"
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                                "-f", "mp3", "-b:a", "128k",
                                "-metadata", f"title={titulo}",
                                "-metadata", "artist=Radio Lofi",
                                "-metadata", f"album=Tema {etiqueta}", str(parcial)],
                               check=True)
                parcial.replace(mp3)   # aparece en la lista solo cuando está completo
            finally:
                parcial.unlink(missing_ok=True)
                if wav:
                    wav.unlink(missing_ok=True)
            gen_state["done"] = i + 1
    except Exception as e:
        gen_state["error"] = str(e)
    finally:
        gen_state["running"] = False
        gen_state["current"] = ""


@app.post("/api/generate")
def api_generate():
    data = request.get_json(force=True)
    count = max(1, min(int(data.get("count", 1)), 20))
    estilo = data.get("estilo", "aleatorio")
    if estilo not in lofigen.ESTILOS:
        return jsonify(error="estilo de música no válido"), 400
    with _lock:
        if gen_state["running"]:
            return jsonify(error="ya hay una generación en curso"), 409
        gen_state.update(running=True, done=0, total=count, error="")
    threading.Thread(target=gen_job, args=(count, estilo), daemon=True).start()
    return jsonify(ok=True)


# ------------------------------------------------------------------ video
# Biblioteca de videos en data/videos/*.mp4 (h264 sin audio) con miniatura .jpg.
# La cola (cola.json) dice qué video suena y durante cuántas horas; la lee en
# caliente videofeeder.py, que alimenta la emisión sin reiniciarla.

def hacer_miniatura(mp4):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "1", "-i", str(mp4),
                    "-frames:v", "1", "-vf", "scale=480:-2", str(mp4.with_suffix(".jpg"))],
                   capture_output=True, timeout=60)


def titulo_de(archivo, titulos):
    return titulos.get(archivo) or archivo[:-4].replace("-", " ").capitalize()


def _convertir(cmd, destino, titulo="", terminar=True):
    """Ejecuta ffmpeg hacia un .part y lo publica solo si termina bien
    (así un corte nunca deja un video roto en la biblioteca)."""
    parcial = destino.with_suffix(".part")
    try:
        subprocess.run(cmd + ["-f", "mp4", str(parcial)], check=True, capture_output=True, text=True)
        parcial.replace(destino)
        hacer_miniatura(destino)
        if titulo:
            set_titulo(destino.name, titulo)
    except subprocess.CalledProcessError as e:
        video_state["error"] = (e.stderr or str(e))[-400:]
    except Exception as e:
        video_state["error"] = str(e)
    finally:
        parcial.unlink(missing_ok=True)
        if terminar:
            video_state["running"] = False


def video_job(tipo, cantidad):
    """Genera `cantidad` videos de 20 s del tipo de escena pedido (cada uno distinto)."""
    scene, capa = VIDEOS_DIR / "escena.ppm", VIDEOS_DIR / "capa.ppm"
    try:
        for k in range(cantidad):
            video_state["done"] = k
            capa.unlink(missing_ok=True)
            nombre, particulas = escenas.make_scene(scene, tipo, capa)
            video_state["tipo"] = nombre
            # movimiento suave y cíclico (vuelve al punto de partida a los 20 s, el bucle no salta):
            # leve paneo de cámara + respiración de brillo
            dx, dy = random.choice([(70, 0), (-70, 0), (40, 25), (0, 30)])
            fondo = (f"scale=2040:1148,crop=1920:1080:x='60+{dx}*(1-cos(2*PI*t/20))/2':"
                     f"y='34+{dy}*(1-cos(2*PI*t/20))/2'")
            brillo = "eq=brightness='0.025*sin(2*PI*t/20)':eval=frame,format=yuv420p"
            entradas = ["-loop", "1", "-framerate", "30", "-i", str(scene)]
            if particulas and capa.exists():
                # la capa (dos mosaicos iguales apilados) baja sin parar; su periodo
                # divide los 20 s, así que el bucle tampoco salta
                v = 540 if particulas == "lluvia" else 108
                entradas += ["-loop", "1", "-framerate", "30", "-i", str(capa)]
                filtro = (f"[0]{fondo},format=gbrp[f];"
                          f"[1]crop=1920:1080:0:'1080-mod(t*{v},1080)',format=gbrp[p];"
                          f"[f][p]blend=all_mode=screen,{brillo}")
                cmd = ["ffmpeg", "-y", "-loglevel", "error", *entradas, "-t", "20",
                       "-filter_complex", filtro]
            else:
                cmd = ["ffmpeg", "-y", "-loglevel", "error", *entradas, "-t", "20", "-vf", f"{fondo},{brillo}"]
            usados, titulo, n = set(load_titulos().values()), nombre, 2
            while titulo in usados:   # «Lluvia», «Lluvia 2», «Lluvia 3»…
                titulo, n = f"{nombre} {n}", n + 1
            _convertir(cmd + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"],
                       libre(VIDEOS_DIR, f"escena-{nombre}", ".mp4", "video"), titulo, terminar=False)
            if video_state["error"]:
                break
        video_state["done"] = cantidad
    finally:
        scene.unlink(missing_ok=True)
        capa.unlink(missing_ok=True)
        video_state["running"] = False


def transcode_job(src, nombre, video=None):
    """Guarda el video subido en la biblioteca como MP4 h264 sin audio (máx. 1080p).
    Si ya es h264 yuv420p de 1080p o menos se copia tal cual (sin perder calidad);
    si no, se recodifica con calidad alta (crf 17)."""
    destino = libre(VIDEOS_DIR, nombre, ".mp4", "video")
    titulo = nombre.replace("_", " ").strip()[:80]
    v = video or {}
    copiable = (v.get("codec_name") == "h264" and v.get("pix_fmt") == "yuv420p"
                and (v.get("width") or 9999) <= 1920 and (v.get("height") or 9999) <= 1080)
    try:
        if copiable:
            _convertir(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-map", "0:v:0",
                        "-c:v", "copy", "-an", "-movflags", "+faststart"], destino, titulo, terminar=False)
            if destino.exists():
                return
            video_state["error"] = ""   # la copia no fue posible (contenedor raro): se recodifica
        _convertir(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-map", "0:v:0",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "17",
                    "-vf", "scale='min(1920,iw)':-2,format=yuv420p", "-an", "-movflags", "+faststart"],
                   destino, titulo)
    finally:
        video_state["running"] = False
        src.unlink(missing_ok=True)


IMAGENES = {"mjpeg", "png", "webp", "bmp", "tiff", "ppm", "pgm", "pam"}   # códecs con los que ffprobe ve una imagen


def imagen_job(src, nombre):
    """Convierte una imagen fija en un video de 10 s sin audio para la biblioteca (ocupa muy poco:
    es un solo fotograma repetido). Se escala a 1080p y se rellena con negro si no es 16:9."""
    try:
        _convertir(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", "30", "-i", str(src),
                    "-t", "10", "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,"
                    "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-tune", "stillimage", "-g", "30",
                    "-an", "-movflags", "+faststart"],
                   libre(VIDEOS_DIR, nombre, ".mp4", "imagen"), nombre.replace("_", " ").strip()[:80])
    finally:
        src.unlink(missing_ok=True)


def audio_job(src, nombre, etiquetas):
    """Guarda el audio subido como MP3 en la biblioteca de música. Un MP3 se copia tal cual
    (sin recodificar: no pierde calidad); otros formatos se convierten a 320 kbit/s."""
    parcial = None
    try:
        titulo = (etiquetas.get("title") or nombre.replace("_", " ").replace("-", " ")).strip()[:80]
        mp3 = libre(TRACKS_DIR, titulo, ".mp3", "cancion")
        parcial = mp3.with_suffix(".part")
        if etiquetas.get("_codec") == "mp3":
            codec = ["-c:a", "copy"]
        else:
            codec = ["-ac", "2", "-ar", "44100", "-b:a", "320k"]
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-map", "0:a:0", "-vn", *codec,
                        "-map_metadata", "-1", "-id3v2_version", "3", "-metadata", f"title={titulo}",
                        "-metadata", f"album=Tema {etiquetas.get('artist') or 'subida'}",
                        "-f", "mp3", str(parcial)], check=True, capture_output=True, text=True)
        parcial.replace(mp3)   # aparece en la lista solo cuando está completo
    except subprocess.CalledProcessError as e:
        video_state["error"] = (e.stderr or str(e))[-400:]
    except Exception as e:
        video_state["error"] = str(e)
    finally:
        if parcial:
            parcial.unlink(missing_ok=True)   # solo el suyo: puede haber una composición en curso
        src.unlink(missing_ok=True)
        video_state["running"] = False


_GENERICO = re.compile(r"^\s*\d[\d.,\s]*\s*(pistas|tracks|canciones)|regal[ií]as|derechos de autor"
                       r"|libres? de derechos|royalty|copyright|^\s*pixabay|^\s*música |^\s*music\s*$", re.I)


def titulo_generico(t):
    """True si el título es un encabezado o frase SEO de Pixabay («18 123 pistas de música lofi
    libres de regalías», «Música lofi relajante, lofi sin derechos de autor y fondo…») en vez
    del nombre de una canción: muy largo, con lista de comas o con esas palabras."""
    t = " ".join(str(t or "").split())
    return not t or len(t) > 60 or t.count(",") > 1 or bool(_GENERICO.search(t))


def _repartir(tmp, nombre, titulo="", artista=""):
    """Analiza el archivo recibido y lo manda a Música o a Videos.
    Un MP3 con carátula trae una «imagen adjunta», no un video: va a Música."""
    try:
        info = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(tmp)],
            capture_output=True, text=True, timeout=60).stdout or "{}")
    except Exception:
        info = {}
    pistas = info.get("streams", [])
    hay_video = any(p.get("codec_type") == "video" and not p.get("disposition", {}).get("attached_pic")
                    and p.get("codec_name") not in ("mjpeg", "png") for p in pistas)
    hay_audio = any(p.get("codec_type") == "audio" for p in pistas)
    if hay_video:
        video_state["destino"] = "video"
        v = next(p for p in pistas if p.get("codec_type") == "video" and not p.get("disposition", {}).get("attached_pic"))
        threading.Thread(target=transcode_job, args=(tmp, nombre, v), daemon=True).start()
        return "video"
    if not hay_audio and any(p.get("codec_type") == "video" and p.get("codec_name") in IMAGENES for p in pistas):
        video_state["destino"] = "video"
        threading.Thread(target=imagen_job, args=(tmp, nombre), daemon=True).start()
        return "imagen"
    if hay_audio:
        video_state["destino"] = "musica"
        etiquetas = {k.lower(): v for k, v in info.get("format", {}).get("tags", {}).items()}
        etiquetas["_codec"] = next(p.get("codec_name") for p in pistas if p.get("codec_type") == "audio")
        if titulo_generico(etiquetas.get("title")):
            etiquetas.pop("title", None)   # p. ej. «18 123 pistas de música lofi…» de una importación antigua
        if titulo and not titulo_generico(titulo):
            etiquetas["title"] = titulo
        if artista:
            etiquetas["artist"] = artista
            if etiquetas.get("title"):      # como lo quiere el usuario: «Tokyo Lofi - ZephiraMusic»
                etiquetas["title"] = f"{etiquetas['title']} - {artista}"
        threading.Thread(target=audio_job, args=(tmp, nombre, etiquetas), daemon=True).start()
        return "musica"
    tmp.unlink(missing_ok=True)
    video_state["running"] = False
    return None


def _ocupar():
    """Reserva el procesado de archivos (uno a la vez). Devuelve False si ya hay otro."""
    with _lock:
        if video_state["running"]:
            return False
        video_state.update(running=True, error="", done=0, total=0, tipo="", destino="")
        return True


@app.post("/api/subir")
def api_subir():
    """Subida única: detecta si es video o música y lo manda a su biblioteca."""
    f = request.files.get("archivo")
    if not f:
        return jsonify(error="falta el archivo"), 400
    if not _ocupar():
        return jsonify(error="ya se está procesando otro archivo; espera a que termine"), 409
    tmp = VIDEOS_DIR / "subida_tmp"
    f.save(tmp)
    tipo = _repartir(tmp, Path(f.filename or "archivo").stem)
    if not tipo:
        return jsonify(error="no es un archivo de audio, video ni imagen válido"), 400
    return jsonify(ok=True, tipo=tipo)


# Importar desde Pixabay: su web está tras Cloudflare y no tiene API de música, así que
# el usuario busca y escucha en pixabay.com y aquí solo se descarga el archivo de audio
# de su CDN (cdn.pixabay.com/audio/…), que sí es público.
PIXABAY_CDN = "cdn.pixabay.com"
MAX_IMPORTAR = 150 * 1024 ** 2


@app.post("/api/importar")
def api_importar():
    datos = request.get_json(force=True)
    url = str(datos.get("url", "")).strip()
    titulo = " ".join(str(datos.get("titulo", "")).split())[:80]   # los manda el marcador desde la página
    artista = " ".join(str(datos.get("artista", "")).split())[:60]
    u = urllib.parse.urlparse(url)
    if u.hostname and u.hostname.endswith("pixabay.com") and "/music/" in u.path:
        return jsonify(error="ese es el enlace de la página: Pixabay no deja leerla desde el servidor. "
                             "Usa el marcador «Importar a Radio Lofi» estando en esa página."), 400
    if u.scheme != "https" or u.hostname != PIXABAY_CDN or not u.path.startswith(("/audio/", "/photo/", "/video/")):
        return jsonify(error="pega el enlace del archivo de Pixabay (empieza por https://cdn.pixabay.com/audio/…, /photo/… o /video/…)"), 400
    if not _ocupar():
        return jsonify(error="ya se está procesando otro archivo; espera a que termine"), 409
    tmp = VIDEOS_DIR / "subida_tmp"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (radio-lofi)"})
        with urllib.request.urlopen(req, timeout=30) as r, open(tmp, "wb") as out:
            leido = 0
            while bloque := r.read(1 << 16):
                leido += len(bloque)
                if leido > MAX_IMPORTAR:
                    raise ValueError("el archivo es demasiado grande")
                out.write(bloque)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        video_state["running"] = False
        return jsonify(error=f"no se pudo descargar: {e}"), 502
    # nombre: el de «?filename=…» si viene, si no el del archivo
    nombre = urllib.parse.parse_qs(u.query).get("filename", [Path(u.path).stem])[0]
    nombre = re.sub(r"[-_]\d{4,}$", "", Path(nombre).stem)   # quita el número final de Pixabay
    tipo = _repartir(tmp, nombre, titulo, artista)
    if not tipo:
        return jsonify(error="el enlace no contiene audio, video ni imagen válidos"), 400
    return jsonify(ok=True, tipo=tipo)


@app.post("/api/video")
def api_video():
    """Crea videos de escena: {tipo: noche|amanecer|dia|…|aleatorio, count: 1-5}."""
    data = request.get_json(force=True, silent=True) or {}
    tipo = data.get("tipo", "noche")
    if tipo not in escenas.TIPOS:
        return jsonify(error="tipo de escena no válido"), 400
    cantidad = max(1, min(int(data.get("count", 1) or 1), 5))
    with _lock:
        if video_state["running"]:
            return jsonify(error="ya se está procesando un video"), 409
        video_state.update(running=True, error="", done=0, total=cantidad, tipo=escenas.TIPOS[tipo], destino="video")
    threading.Thread(target=video_job, args=(tipo, cantidad), daemon=True).start()
    return jsonify(ok=True)


def load_cola():
    return leer_json(COLA_FILE, [])


@app.delete("/api/videos/<name>")
def api_video_delete(name):
    if not re.fullmatch(r"[\w-]+\.mp4", name):
        return jsonify(error="nombre no válido"), 400
    # fuera de la cola también (si está sonando, el alimentador pasa al siguiente)
    with _io_lock:
        escribir_json(COLA_FILE, [c for c in load_cola() if c["video"] != name])
    for p in (VIDEOS_DIR / name, (VIDEOS_DIR / name).with_suffix(".jpg")):
        p.unlink(missing_ok=True)
    set_titulo(name, "")
    return jsonify(ok=True)


@app.post("/api/videos/<name>/titulo")
def api_video_titulo(name):
    if not re.fullmatch(r"[\w-]+\.mp4", name) or not (VIDEOS_DIR / name).exists():
        return jsonify(error="video no encontrado"), 404
    titulo = " ".join(str(request.get_json(force=True).get("titulo", "")).split())[:80]
    if not titulo:
        return jsonify(error="el nombre no puede estar vacío"), 400
    set_titulo(name, titulo)
    return jsonify(ok=True, titulo=titulo)


@app.post("/api/cola")
def api_cola():
    """Guarda la lista completa de videos: [{id?, video, hours, activo}].
    Solo los activos se emiten (en este orden); los demás guardan su sitio y horas."""
    nueva = []
    for c in request.get_json(force=True).get("cola", []):
        video = str(c.get("video", ""))
        if not re.fullmatch(r"[\w-]+\.mp4", video) or not (VIDEOS_DIR / video).exists():
            continue
        try:
            horas = min(max(float(c.get("hours", 1)), 1 / 60), 720)   # de 1 minuto a 30 días
        except (TypeError, ValueError):
            horas = 1.0
        nueva.append({"id": str(c.get("id") or secrets.token_hex(4))[:16],
                      "video": video, "hours": round(horas, 5), "activo": c.get("activo") is not False})
    with _io_lock:
        escribir_json(COLA_FILE, nueva)
    return jsonify(ok=True, cola=nueva)


# Modo aleatorio, independiente para videos y para música: cada alimentador elige el siguiente
# elemento al azar si existe el fichero de control «aleatorio» en su carpeta (se relee antes de cada cambio).
ALEATORIO_DIRS = {"video": VIDEOS_DIR, "musica": TRACKS_DIR}


def estado_aleatorio(cfg=None):
    cfg = cfg if cfg is not None else load_config()
    viejo = bool(cfg.get("aleatorio"))   # ajuste único de versiones anteriores a la 2.1 (valía para ambos)
    return {k: bool(cfg.get(f"aleatorio_{k}", viejo)) for k in ALEATORIO_DIRS}


def aplicar_aleatorio(estado):
    for que, d in ALEATORIO_DIRS.items():
        marca = d / "aleatorio"
        if estado.get(que):
            marca.touch()
        else:
            marca.unlink(missing_ok=True)


aplicar_aleatorio(estado_aleatorio())   # al arrancar, según la configuración guardada


@app.post("/api/aleatorio")
def api_aleatorio():
    data = request.get_json(force=True)
    que = data.get("que")
    if que not in ALEATORIO_DIRS:
        return jsonify(error="indica «que»: video o musica"), 400
    activo = bool(data.get("activo"))
    actualizar_config(**{f"aleatorio_{que}": activo})
    estado = estado_aleatorio()
    aplicar_aleatorio(estado)
    return jsonify(ok=True, aleatorio=estado)


# Radio infinita: música procedural compuesta en tiempo real en vez de la lista (audiofeeder.py).
INFINITA_FILE = TRACKS_DIR / "infinita.json"
INFINITA_DEF = {"activo": False, "estilos": ["aleatorio"], "rotar_min": 60, "bpm": 0, "textura": False}   # bpm 0 = automático
ROTAR_MIN = (15, 30, 60, 120, 240)


def estado_infinita(cfg=None):
    cfg = cfg if cfg is not None else load_config()
    d = {**INFINITA_DEF, **(cfg.get("infinita") or {})}
    if "estilo" in d:   # formato de la primera 2.3: un solo estilo
        d["estilos"] = [d.pop("estilo")]
    d["activo"] = bool(d["activo"])
    d["estilos"] = [e for e in d["estilos"] if e in lofigen.ESTILOS] or ["aleatorio"]
    d["rotar_min"] = int(d["rotar_min"]) if d["rotar_min"] in ROTAR_MIN else 60
    d["bpm"] = int(d["bpm"]) if str(d["bpm"]).isdigit() and 50 <= int(d["bpm"]) <= 110 else 0
    d["textura"] = bool(d.get("textura"))   # vinilo/lluvia de fondo; apagado por defecto (sonaba a ruido blanco)
    return d


def aplicar_infinita(d):
    with _io_lock:
        escribir_json(INFINITA_FILE, d)


aplicar_infinita(estado_infinita())   # al arrancar, según la configuración guardada


@app.post("/api/infinita")
def api_infinita():
    """Ajustes de la radio infinita: activo, estilos (lista; varios = rotan cada rotar_min
    minutos en ese orden), rotar_min (15/30/60/120/240) y bpm (50-110, 0 = automático)."""
    data = request.get_json(force=True) or {}
    d = estado_infinita()
    if "activo" in data:
        d["activo"] = bool(data["activo"])
    if "estilos" in data:
        estilos = [e for e in (data["estilos"] or []) if isinstance(e, str)]
        if not estilos or any(e not in lofigen.ESTILOS for e in estilos):
            return jsonify(error="marca al menos un estilo válido"), 400
        d["estilos"] = list(dict.fromkeys(estilos))
    if "rotar_min" in data:
        if data["rotar_min"] not in ROTAR_MIN:
            return jsonify(error="intervalo no válido"), 400
        d["rotar_min"] = int(data["rotar_min"])
    if "textura" in data:
        d["textura"] = bool(data["textura"])
    if "bpm" in data:
        try:
            bpm = int(data["bpm"] or 0)
        except (TypeError, ValueError):
            return jsonify(error="tempo no válido"), 400
        if bpm and not 50 <= bpm <= 110:
            return jsonify(error="el tempo va de 50 a 110 bpm (0 = automático)"), 400
        d["bpm"] = bpm
    actualizar_config(infinita=d)
    aplicar_infinita(d)
    return jsonify(ok=True, infinita=d)


@app.post("/api/youtube_url")
def api_youtube_url():
    """Enlace del canal o del directo de YouTube para el botón «Abrir» del panel (vacío = quitar)."""
    url = str(request.get_json(force=True).get("url", "")).strip()
    if url and not re.fullmatch(r"https://(www\.|m\.)?(youtube\.com|youtu\.be)/\S{1,200}", url):
        return jsonify(error="pega un enlace de YouTube (https://www.youtube.com/@tu-canal, /channel/…, /live/…)"), 400
    actualizar_config(youtube_url=url)
    return jsonify(ok=True, youtube_url=url)


@app.post("/api/cola/saltar")
def api_cola_saltar():
    if not stream_running():
        return jsonify(error="la emisión no está en marcha"), 409
    (VIDEOS_DIR / "saltar").touch()
    return jsonify(ok=True)


def _miniaturas_que_falten():
    for p in VIDEOS_DIR.glob("*.mp4"):
        if not p.with_suffix(".jpg").exists():
            hacer_miniatura(p)


threading.Thread(target=_miniaturas_que_falten, daemon=True).start()


# ------------------------------------------------------------------ música

def canciones_en_orden():
    """Canciones en el orden guardado; las que no están en él (nuevas) van al final,
    de la más antigua a la más reciente. audiofeeder.py usa la misma regla."""
    existen = {p.name: p for p in TRACKS_DIR.glob("*.mp3")}
    lista = [existen.pop(n) for n in leer_json(ORDEN_MUSICA, []) if n in existen]
    return lista + sorted(existen.values(), key=lambda q: q.stat().st_mtime)


@app.delete("/api/track/<name>")
def api_delete(name):
    if not re.fullmatch(r"[\w-]+\.mp3", name):
        return jsonify(error="nombre no válido"), 400
    (TRACKS_DIR / name).unlink(missing_ok=True)
    set_titulo(name, "", TITULOS_MUSICA)
    return jsonify(ok=True)


@app.post("/api/track/<name>/titulo")
def api_track_titulo(name):
    if not re.fullmatch(r"[\w-]+\.mp3", name) or not (TRACKS_DIR / name).exists():
        return jsonify(error="canción no encontrada"), 404
    titulo = " ".join(str(request.get_json(force=True).get("titulo", "")).split())[:80]
    if not titulo:
        return jsonify(error="el nombre no puede estar vacío"), 400
    set_titulo(name, titulo, TITULOS_MUSICA)
    return jsonify(ok=True, titulo=titulo)


@app.post("/api/musica/orden")
def api_musica_orden():
    nombres = [str(n) for n in request.get_json(force=True).get("orden", [])
               if re.fullmatch(r"[\w-]+\.mp3", str(n))]
    with _io_lock:
        escribir_json(ORDEN_MUSICA, nombres)   # el alimentador lo lee en caliente
    return jsonify(ok=True)


@app.post("/api/musica/saltar")
def api_musica_saltar():
    if not stream_running():
        return jsonify(error="la emisión no está en marcha"), 409
    (TRACKS_DIR / "saltar").touch()
    return jsonify(ok=True)


# ---------------------------------------------------------------- emisión

def stream_running():
    return stream["proc"] is not None and stream["proc"].poll() is None


def start_stream(local=False):
    """Arranca la tubería de emisión. Devuelve None si ok o el error.

    Toda emisión se envía además al servidor MediaMTX de la red local, para
    ver y escuchar con VLC exactamente lo mismo que recibe YouTube.
    local=True emite solo a la red local (sin YouTube)."""
    with _stream_lock:
        cfg = load_config()
        key = cfg.get("rtmp_key", "")
        # tasa de bits del video: YouTube recomienda 4500-9000 kbps para 1080p30 (avisa por debajo
        # de 6800). Configurable en config.json («video_kbps») por si la subida no da para tanto.
        try:
            kbps = int(cfg.get("video_kbps") or 6800)
        except (TypeError, ValueError):
            kbps = 6800
        kbps = min(max(kbps, 1000), 12000)
        if stream_running():
            return "la emisión ya está en marcha"
        if not any((VIDEOS_DIR / c["video"]).exists() for c in load_cola() if c.get("activo") is not False):
            return "activa «en bucle» al menos un video (Biblioteca → Videos)"
        if not any(TRACKS_DIR.glob("*.mp3")):
            return "no hay pistas: genera música primero"
        if not local and not key:
            return "guarda la clave de emisión RTMP primero"
        dest = None if local else f"{YOUTUBE_RTMP}/{key}"
        # Una sola codificación repartida con «tee»: el destino principal y la copia
        # local. onfail=ignore: si la copia local falla, YouTube sigue sin cortes.
        salidas = ([f"[f=flv:{RW_TIMEOUT}]{dest}"] if dest else []) + [f"[f=flv:onfail=ignore:{RW_TIMEOUT}]{PREVIEW_URL}"]
        # Audio: audiofeeder.py decodifica las canciones en el orden de la lista y en
        # bucle; anota la que suena y atiende «saltar». Video: videofeeder.py entrega
        # fotogramas en bruto de la cola por el descriptor 3. Ninguno de los dos
        # reinicia el codificador al cambiar de canción o de video.
        feeder = f'python3 -u "{BASE / "audiofeeder.py"}" "{TRACKS_DIR}"'
        vfeeder = f'python3 -u "{BASE / "videofeeder.py"}" "{VIDEOS_DIR}"'
        # -stats_period 5: una línea de progreso cada 5 s (el registro crece poco
        # y el vigilante sabe que la emisión sigue viva si el archivo se actualiza)
        enc = (f'ffmpeg -y -stats_period 5 -thread_queue_size 8 -f rawvideo -pix_fmt yuv420p -s 1920x1080 '
               f'-r 30 -i pipe:3 -thread_queue_size 48 '
               f'-f s16le -ar 44100 -ac 2 -i pipe:0 -map 0:v -map 1:a '
               f'-c:v libx264 -preset veryfast -b:v {kbps}k -maxrate {kbps}k '
               f'-bufsize {kbps * 2}k -g 60 -r 30 -c:a aac -b:a 160k -ar 44100 '
               f'-flags +global_header -f tee -use_fifo 1 -fifo_options "{FIFO_OPTS}" "{"|".join(salidas)}"')
        if stream["log_fh"]:
            stream["log_fh"].close()
        if STREAM_LOG.exists() and STREAM_LOG.stat().st_size:
            STREAM_LOG.replace(STREAM_LOG_ANTERIOR)   # se conserva el final de la emisión anterior
        fh = open(STREAM_LOG, "a")   # en modo «a»: la limpieza diaria puede vaciarlo sin romperlo
        cmd = f"{enc} 3< <({vfeeder}) < <({feeder}); kill 0"
        fh.write("$ " + cmd.replace(key or "~", "<clave>") + "\n")
        fh.flush()
        # bash solo espera al codificador y, cuando este acaba (caída o parada),
        # «kill 0» cierra también los alimentadores en lugar de dejarlos girando
        stream["proc"] = subprocess.Popen(["bash", "-c", cmd],
                                          stdout=fh, stderr=fh, preexec_fn=os.setsid)
        stream["mode"] = "local" if local else "directo"
        stream["inicio"] = time.time()   # para mostrar cuánto lleva en directo
        stream["log_fh"] = fh
        return None


def parar_stream():
    with _stream_lock:
        p = stream["proc"]
        if p is not None and p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), sig.SIGTERM)
            except (ProcessLookupError, PermissionError):
                p.terminate()
            try:
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(p.pid), sig.SIGKILL)
                except Exception:
                    p.kill()
        if stream["log_fh"]:
            stream["log_fh"].close()
            stream["log_fh"] = None
        stream["proc"] = None


@app.post("/api/stream/start")
def api_stream_start():
    local = request.get_json(force=True).get("mode") == "local"
    with _stream_lock:
        err = start_stream(local)
        if err:
            return jsonify(error=err), 409 if "en marcha" in err else 400
        if not local:
            actualizar_config(emitir=True)   # el vigilante la relanzará si se cae
    return jsonify(ok=True)


@app.post("/api/stream/stop")
def api_stream_stop():
    with _stream_lock:
        actualizar_config(emitir=False)   # parada manual: el vigilante no debe relanzar
        parar_stream()
    return jsonify(ok=True)


# --- vigilante 24/7: si la emisión debía estar activa y se ha caído (o se ha
# quedado colgada sin avanzar), la relanza
COLGADA_S = 90   # ffmpeg escribe progreso cada 5 s; 90 s sin hacerlo = colgada (último recurso:
                 # las salidas ya reconectan solas gracias al fifo, ver FIFO_OPTS)


def _anotar(texto):
    marca = time.strftime("%d/%m %H:%M:%S")
    with open(WATCHDOG_LOG, "a") as f:
        f.write(f"{marca} {texto}\n")
    return marca


def _vigilante():
    time.sleep(8)   # deja arrancar el servidor
    while True:
        espera = 15
        try:
            with _stream_lock:
                if load_config().get("emitir"):
                    if stream_running() and stream["mode"] == "directo":
                        try:
                            quieta = time.time() - STREAM_LOG.stat().st_mtime
                        except OSError:
                            quieta = 0
                        if quieta > COLGADA_S:
                            _anotar(f"la emisión no avanza desde hace {int(quieta)} s: se reinicia")
                            parar_stream()
                    if not stream_running():
                        err = start_stream(False)
                        if err is None:
                            watchdog["reinicios"] += 1
                            watchdog["ultimo"] = _anotar("emisión relanzada")
                        else:
                            _anotar(f"no se pudo relanzar: {err}")
                        espera = 45 if err else 20   # margen antes de revisar de nuevo
        except Exception:
            pass
        time.sleep(espera)


threading.Thread(target=_vigilante, daemon=True).start()


# --- limpieza de registros: cada 24 h se vacían (lo anterior se borra)
ROTAR_S = 24 * 3600


def _limpiar_logs():
    while True:
        time.sleep(ROTAR_S)
        for f in (STREAM_LOG, WATCHDOG_LOG):
            try:
                os.truncate(f, 0)   # ffmpeg escribe en modo «a»: sigue al principio del archivo
            except OSError:
                pass
        _anotar("registros de más de 24 h borrados")


threading.Thread(target=_limpiar_logs, daemon=True).start()


@app.get("/api/key")
def api_key_ver():
    """Devuelve la clave guardada para el botón del ojo del panel (solo red local, sin contraseña,
    como el resto del panel; el usuario lo pidió así)."""
    return jsonify(key=load_config().get("rtmp_key", ""))


@app.post("/api/key")
def api_key():
    key = request.get_json(force=True).get("key", "").strip()
    if not re.fullmatch(r"[\w-]{10,60}", key):
        return jsonify(error="formato de clave no válido"), 400
    actualizar_config(rtmp_key=key)
    return jsonify(ok=True)


# ----------------------------------------------------------------- estado

def _cola_de_archivo(path, n=300, maximo=64_000):
    """Últimas n líneas leyendo solo el final del archivo (puede ser grande)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - maximo))
            texto = f.read().decode(errors="replace")
    except OSError:
        return ""
    lineas = texto.replace("\r", "\n").splitlines()
    return "\n".join(l for l in lineas[-n:] if l.strip())


@app.get("/api/logs")
def api_logs():
    key = load_config().get("rtmp_key") or None
    ocultar = lambda t: t.replace(key, "<clave>") if key else t   # ffmpeg imprime la URL con la clave
    return jsonify(stream=ocultar(_cola_de_archivo(STREAM_LOG)), vigilante=_cola_de_archivo(WATCHDOG_LOG),
                   anterior=ocultar(_cola_de_archivo(STREAM_LOG_ANTERIOR)))


# ---------------- estadísticas de recursos (botón «Stats» del panel) ----------------
# Todo se lee de /proc y del cgroup del propio contenedor: no hace falta el socket de Docker.
# Los porcentajes se calculan por diferencia con la lectura anterior (el panel pide cada 2 s).
CGROUP = Path("/sys/fs/cgroup")
_stats_lock = threading.Lock()
_stats_prev = {}
_CLK_TCK = os.sysconf("SC_CLK_TCK")


def _leer(path, defecto=""):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return defecto


def _cgroup_kv(nombre):
    out = {}
    for linea in _leer(CGROUP / nombre).splitlines():
        partes = linea.split()
        if len(partes) == 2 and partes[1].isdigit():
            out[partes[0]] = int(partes[1])
    return out


def _cpu_host():
    """Ticks (total, ocupado) de todas las CPU del servidor y de cada núcleo (/proc/stat)."""
    total = ocupado = 0
    nucleos = []
    for linea in _leer("/proc/stat").splitlines():
        if not linea.startswith("cpu"):
            continue
        nombre, *v = linea.split()
        v = [int(x) for x in v]
        t = sum(v)
        o = t - v[3] - v[4]   # sin idle ni iowait
        if nombre == "cpu":
            total, ocupado = t, o
        else:
            nucleos.append((t, o))
    return total, ocupado, nucleos


def _mem_host():
    m = {}
    for linea in _leer("/proc/meminfo").splitlines():
        k, _, resto = linea.partition(":")
        try:
            m[k] = int(resto.split()[0]) * 1024
        except (ValueError, IndexError):
            pass
    return m


def _procesos():
    """Procesos del contenedor: pid, nombre, ticks de CPU y RSS (de /proc/<pid>/stat)."""
    pagina = os.sysconf("SC_PAGE_SIZE")
    out = {}
    for d in os.scandir("/proc"):
        if not d.name.isdigit():
            continue
        try:
            st = Path(d.path, "stat").read_text()
            cmd = Path(d.path, "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        nombre = st[st.index("(") + 1:st.rindex(")")]
        campos = st[st.rindex(")") + 2:].split()
        ticks = int(campos[11]) + int(campos[12])
        rss = int(campos[21]) * pagina
        args = [a.decode(errors="replace") for a in cmd if a]
        if nombre in ("python", "python3", "bash", "sh"):
            if "-c" in args[1:]:   # «bash -c "ffmpeg … ; kill 0"»: el lanzador de la emisión
                nombre = f"{nombre} (lanzador de la emisión)"
            else:
                script = next((a for a in args[1:] if not a.startswith("-")), None)   # salta -u…
                if script:
                    nombre = f"{nombre} {Path(script).name}"   # app.py / videofeeder.py / audiofeeder.py
        out[int(d.name)] = {"nombre": nombre, "ticks": ticks, "rss": rss}
    return out


def _tam_carpeta(d, ext=None):
    total = n = 0
    try:
        for f in os.scandir(d):
            if f.is_file() and (ext is None or f.name.endswith(ext)):
                total += f.stat().st_size
                n += 1
    except OSError:
        pass
    return total, n


def _gpu():
    """Gráficas visibles en /sys/class/drm; renderD* es lo que necesita ffmpeg para codificar por hardware."""
    vendedores = {"0x8086": "Intel", "0x1002": "AMD", "0x10de": "NVIDIA", "0x1414": "Microsoft (virtual)",
                  "0x1af4": "virtio (virtual)", "0x15ad": "VMware (virtual)"}
    tarjetas = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        if "-" in card.name:
            continue
        dev = card / "device"
        vendor = _leer(dev / "vendor")
        info = {"nombre": card.name, "vendedor": vendedores.get(vendor, vendor or "desconocido"),
                "driver": Path(os.path.realpath(dev / "driver")).name if (dev / "driver").exists() else "",
                "render": sorted(r.name for r in (dev / "drm").glob("renderD*")),
                "mhz": None, "mhz_max": None}
        try:
            cur, mx = _leer(card / "gt_cur_freq_mhz"), _leer(card / "gt_max_freq_mhz")
            if cur and mx:
                info["mhz"], info["mhz_max"] = int(cur), int(mx)
        except ValueError:
            pass
        tarjetas.append(info)
    render = sorted(p.name for p in Path("/dev/dri").glob("renderD*")) if Path("/dev/dri").exists() else []
    return {"tarjetas": tarjetas, "render": render, "codificador": "CPU (libx264)"}


_RE_FFMPEG = re.compile(r"frame=\s*(\d+)\s+fps=\s*([\d.]+)\s+q=\s*([-\d.]+).*?time=(\S+).*?speed=\s*([\d.]+)x")


def _ffmpeg_progreso():
    """Última línea de progreso de ffmpeg en stream.log (-stats_period 5)."""
    cola = _cola_de_archivo(STREAM_LOG, n=20, maximo=6_000)
    m = None
    for m in _RE_FFMPEG.finditer(cola):
        pass
    if not m:
        return None
    return {"frame": int(m.group(1)), "fps": float(m.group(2)), "q": float(m.group(3)),
            "tiempo": m.group(4), "velocidad": float(m.group(5))}


@app.get("/api/stats")
def api_stats():
    with _stats_lock:
        ahora = time.monotonic()
        prev = _stats_prev
        if prev and ahora - prev["t"] < 0.5 and "res" in prev:
            return jsonify(prev["res"])
        ncpu = os.cpu_count() or 1
        cpu_max = _leer(CGROUP / "cpu.max").split()
        limite_cpu = (int(cpu_max[0]) / int(cpu_max[1])) if len(cpu_max) == 2 and cpu_max[0] != "max" else ncpu
        uso_cont = _cgroup_kv("cpu.stat").get("usage_usec", 0)
        host_total, host_ocupado, host_nucleos = _cpu_host()
        procs = _procesos()
        net = {}
        for linea in _leer("/proc/net/dev").splitlines():
            if ":" in linea:
                iface, datos = linea.split(":", 1)
                v = datos.split()
                if iface.strip() != "lo":
                    net[iface.strip()] = (int(v[0]), int(v[8]))   # bytes recibidos, enviados
        rx = sum(a for a, _ in net.values())
        tx = sum(b for _, b in net.values())
        io = {}
        for linea in _leer(CGROUP / "io.stat").splitlines():
            for kv in linea.split()[1:]:
                k, _, v = kv.partition("=")
                if v.isdigit():
                    io[k] = io.get(k, 0) + int(v)

        dt = (ahora - prev["t"]) if prev else 0
        cpu_cont_pct = cpu_host_pct = rx_kbps = tx_kbps = rd_bps = wr_bps = 0.0
        proc_pct = {}
        nucleos_pct = [0.0] * len(host_nucleos)
        if dt > 0:
            cpu_cont_pct = (uso_cont - prev["uso_cont"]) / (dt * 1e6) * 100       # % de un núcleo
            d_total = host_total - prev["host_total"]
            if d_total > 0:
                cpu_host_pct = (host_ocupado - prev["host_ocupado"]) / d_total * 100
            for i, (t, o) in enumerate(host_nucleos):
                if i < len(prev["nucleos"]) and t - prev["nucleos"][i][0] > 0:
                    nucleos_pct[i] = (o - prev["nucleos"][i][1]) / (t - prev["nucleos"][i][0]) * 100
            rx_kbps = (rx - prev["rx"]) * 8 / dt / 1000
            tx_kbps = (tx - prev["tx"]) * 8 / dt / 1000
            rd_bps = (io.get("rbytes", 0) - prev["io"].get("rbytes", 0)) / dt
            wr_bps = (io.get("wbytes", 0) - prev["io"].get("wbytes", 0)) / dt
            for pid, pr in procs.items():
                antes = prev["procs"].get(pid)
                if antes:
                    proc_pct[pid] = (pr["ticks"] - antes["ticks"]) / _CLK_TCK / dt * 100
        _stats_prev.clear()
        _stats_prev.update(t=ahora, uso_cont=uso_cont, host_total=host_total, host_ocupado=host_ocupado,
                           nucleos=host_nucleos, rx=rx, tx=tx, io=io, procs=procs)

        mem = _mem_host()
        mem_cont = int(_leer(CGROUP / "memory.current", "0") or 0)
        mem_stat = _cgroup_kv("memory.stat")
        mem_max = _leer(CGROUP / "memory.max", "max")
        mem_limite = int(mem_max) if mem_max.isdigit() else mem.get("MemTotal", 0)
        try:
            uptime_host = float(_leer("/proc/uptime").split()[0])
            inicio_pid1 = int(Path("/proc/1/stat").read_text().rsplit(")", 1)[1].split()[19]) / _CLK_TCK
            uptime_cont = uptime_host - inicio_pid1
        except (OSError, ValueError, IndexError):
            uptime_host = uptime_cont = 0
        try:
            carga = [float(x) for x in _leer("/proc/loadavg").split()[:3]]
        except ValueError:
            carga = [0, 0, 0]
        disco = shutil.disk_usage(TRACKS_DIR)
        musica, n_musica = _tam_carpeta(TRACKS_DIR, ".mp3")
        videos, n_videos = _tam_carpeta(VIDEOS_DIR, ".mp4")
        logs, _ = _tam_carpeta(LOGS_DIR)
        top = sorted(procs.items(), key=lambda kv: (proc_pct.get(kv[0], 0), kv[1]["rss"]), reverse=True)[:8]
        corriendo = stream_running()
        res = {
            "contenedor": {
                "cpu_pct": round(cpu_cont_pct, 1), "cpu_limite": round(limite_cpu, 2), "ncpu": ncpu,
                "mem": mem_cont, "mem_rss": mem_stat.get("anon", mem_cont), "mem_cache": mem_stat.get("file", 0),
                "mem_limite": mem_limite, "procesos": len(procs), "uptime": round(uptime_cont),
                "pids_max": _leer(CGROUP / "pids.max", "max"),
            },
            "servidor": {
                "cpu_pct": round(cpu_host_pct, 1), "ncpu": ncpu, "carga": carga,
                "nucleos": [round(x, 1) for x in nucleos_pct],
                "mem_total": mem.get("MemTotal", 0), "mem_disponible": mem.get("MemAvailable", 0),
                "swap_total": mem.get("SwapTotal", 0), "swap_libre": mem.get("SwapFree", 0),
                "uptime": round(uptime_host),
            },
            "red": {"rx_kbps": round(rx_kbps), "tx_kbps": round(tx_kbps), "rx_total": rx, "tx_total": tx},
            "disco": {"total": disco.total, "usado": disco.used, "libre": disco.free,
                      "musica": musica, "n_musica": n_musica, "videos": videos, "n_videos": n_videos, "logs": logs,
                      "lectura_bps": round(rd_bps), "escritura_bps": round(wr_bps)},
            "gpu": _gpu(),
            "emision": {"running": corriendo, "mode": stream["mode"] if corriendo else "",
                        "kbps": int(load_config().get("video_kbps") or 6800),
                        "ffmpeg": _ffmpeg_progreso() if corriendo else None},
            "procesos": [{"pid": pid, "nombre": pr["nombre"], "cpu_pct": round(proc_pct.get(pid, 0), 1), "rss": pr["rss"]}
                         for pid, pr in top],
            "version": VERSION,
        }
        _stats_prev["res"] = res
        return jsonify(res)


@app.get("/api/state")
def api_state():
    tit_mus = load_titulos(TITULOS_MUSICA)
    tracks = []
    for p in canciones_en_orden():
        meta = dict(track_meta(p))
        meta["title"] = tit_mus.get(p.name) or meta["title"]
        tracks.append({"name": p.name, **meta})
    titulos = load_titulos()
    videos = []
    for p in VIDEOS_DIR.glob("*.mp4"):
        try:
            st = p.stat()
        except OSError:
            continue
        videos.append((st.st_mtime, {"name": p.name, "title": titulo_de(p.name, titulos),
                                     "dur": video_meta(p), "size": st.st_size}))
    videos = [v for _, v in sorted(videos, key=lambda x: x[0], reverse=True)]
    corriendo = stream_running()
    actual = leer_json(VIDEOS_DIR / "actual.json", {}) if corriendo else {}
    musica = leer_json(TRACKS_DIR / "actual.json", {}) if corriendo else {}
    cfg = load_config()
    return jsonify(
        gen=gen_state,
        video={k: video_state[k] for k in ("running", "error", "done", "total", "tipo", "destino")},
        tipos_escena=escenas.TIPOS,
        estilos_musica=ESTILOS_PUB,
        videos=videos,
        cola=load_cola(),
        actual=actual,
        musica=musica,
        stream={"running": corriendo, "mode": stream["mode"], "auto": bool(cfg.get("emitir")),
                "inicio": stream["inicio"] if corriendo else 0,
                "reinicios": watchdog["reinicios"], "ultimo": watchdog["ultimo"]},
        has_key=bool(cfg.get("rtmp_key")),
        aleatorio=estado_aleatorio(cfg),
        infinita=estado_infinita(cfg),
        youtube_url=str(cfg.get("youtube_url") or ""),
        version=VERSION,
        tracks=tracks)


@app.get("/tracks/<name>")
def serve_track(name):
    # conditional=True habilita peticiones Range (necesarias para adelantar)
    return send_from_directory(TRACKS_DIR, name, conditional=True)


@app.get("/videos/<name>")
def serve_video(name):
    if not re.fullmatch(r"[\w-]+\.(mp4|jpg)", name):
        return Response(status=404)
    return send_from_directory(VIDEOS_DIR, name, conditional=True)


@app.get("/vendor/<name>")
def serve_vendor(name):
    # librerías del panel servidas en local (hls.js): funciona sin acceso a internet
    return send_from_directory(BASE / "vendor", name, max_age=86400)


@app.get("/")
def index():
    return send_from_directory(BASE, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8750)), threaded=True)
