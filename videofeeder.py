"""Alimentador de video de la emisión.

Reproduce la cola de videos (data/videos/cola.json) en orden y en bucle: cada
video se repite durante las horas indicadas y después pasa al siguiente.
Escribe fotogramas en bruto (yuv420p, 1920x1080, 30 fps) por la salida
estándar, así que el codificador de la emisión nunca se reinicia: cambiar de
video, editar la cola o saltar no corta el directo.

Ficheros de control (en la carpeta de videos):
  cola.json    [{"id", "video", "hours", "activo", "niebla"}, ...] — se relee en caliente;
               los que tienen activo=false no se emiten; con niebla=true se superpone una neblina
               animada (ruido Perlin de ffmpeg, mezcla «screen» solo en luma) para que las
               imágenes fijas parezcan moverse
  actual.json  lo que está sonando ahora (lo escribe este proceso)
  saltar       si existe, pasa al siguiente video y lo borra
  aleatorio    si existe, el siguiente video se elige al azar (modo aleatorio del panel)
  infinita.json  {"activo", "velocidad" 1-10, "cantidad" 1-10, "color"}: video infinito tipo Matrix
               generado en tiempo real con numpy (lluvia de glifos) en vez de la cola; se relee cada segundo.
"""
import json
import random
import subprocess
import sys
import time
from pathlib import Path

W, H, FPS = 1920, 1080, 30
FRAME = W * H * 3 // 2
D = Path(sys.argv[1])
COLA, ACTUAL, SALTAR = D / "cola.json", D / "actual.json", D / "saltar"
ALEATORIO = D / "aleatorio"
INFINITA = D / "infinita.json"
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


_infinita_cache = (None, None)   # (mtime_ns, ajustes): solo se relee el fichero si cambia su fecha


def leer_infinita():
    """Ajustes del video infinito si está activo; None si toca la cola (o no se puede leer)."""
    global _infinita_cache
    try:
        m = INFINITA.stat().st_mtime_ns
        if m != _infinita_cache[0]:
            d = json.loads(INFINITA.read_text())
            _infinita_cache = (m, d if d.get("activo") else None)
        return _infinita_cache[1]
    except Exception:
        _infinita_cache = (None, None)
        return None


# ── video infinito: lluvia de letras tipo Matrix ──
# Celdas de 16x24 px (120x45). Un atlas de glifos (katakana de media anchura con VL Gothic y
# letras/dígitos/símbolos con DejaVu Sans Mono, dibujados por ffmpeg una sola vez, más sus espejos
# como en la película) a 32 niveles de brillo + cabeza blanca, en Y, U y V. Cada fotograma es una
# indexación numpy del atlas con la rejilla de (nivel, glifo): ~7 ms (22 % de un núcleo).
CW, CH = 16, 24
COLS, ROWS = W // CW, H // CH
FUENTE = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FUENTE_JP = "/usr/share/fonts/truetype/vlgothic/VL-Gothic-Regular.ttf"
LATINOS = list("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ+-*<>=:")
KANA = list("ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ")
GLIFOS = [(c, FUENTE) for c in LATINOS] + [(c, FUENTE_JP) for c in KANA]
if not Path(FUENTE_JP).exists():   # sin la fuente japonesa (imagen antigua): solo latinos
    GLIFOS = [(c, FUENTE) for c in LATINOS]
NIVELES = 32
COLORES = {   # RGB de la cabeza de color (la cola se apaga hacia negro)
    "verde": (0, 255, 70), "azul": (40, 150, 255), "ambar": (255, 170, 30), "blanco": (225, 225, 225),
    "rojo": (255, 55, 55), "morado": (175, 80, 255), "cian": (40, 230, 230),
}


_glifos_cache = None   # el atlas no cambia: ffmpeg se lanza una sola vez por proceso
_atlas_cache = {}      # (Y, U, V) por color RGB (~3 MB cada uno): encender/apagar o cambiar de color no recalcula


def _glifos_np():
    """Atlas (n, 24, 16) float 0..1: todos los glifos en una sola llamada a ffmpeg, más espejos."""
    global _glifos_cache
    if _glifos_cache is not None:
        return _glifos_cache
    import numpy as np
    n = len(GLIFOS)
    esc = lambda c: c.replace("\\", "\\\\").replace(":", "\\:")   # «:» separa opciones del filtro
    filtros = ",".join(f"drawtext=fontfile={fuente}:text='{esc(c)}':fontsize=21:fontcolor=white:"
                       f"x={i * CW}+({CW}-tw)/2:y=({CH}-th)/2-1" for i, (c, fuente) in enumerate(GLIFOS))
    r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=black:s={n * CW}x{CH}", "-vf", filtros,
                        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, timeout=60)
    if len(r.stdout) != CH * n * CW:
        raise RuntimeError("no se pudieron dibujar los glifos: " + r.stderr.decode(errors="replace")[-300:])
    tira = np.frombuffer(r.stdout, dtype=np.uint8).reshape(CH, n * CW)
    g = [tira[:, i * CW:(i + 1) * CW] for i in range(n)]
    g += [x[:, ::-1] for x, (c, _) in zip(g, GLIFOS) if c.isalpha()]   # espejados, como en la película
    _glifos_cache = np.stack(g).astype(np.float32) / 255
    return _glifos_cache


def _atlas(glifos, rgb):
    """Atlas Y/U/V uint8 (niveles+1, n, alto, ancho) para un color (BT.601, rango limitado)."""
    if rgb in _atlas_cache:
        return _atlas_cache[rgb]
    import numpy as np
    r, g, b = rgb
    yc, uc, vc = 16 + (65.481 * r + 128.553 * g + 24.966 * b) / 255, 128 + (-37.797 * r - 74.203 * g + 112 * b) / 255, \
                 128 + (112 * r - 93.786 * g - 18.214 * b) / 255
    n = len(glifos)
    sub = glifos.reshape(n, CH // 2, 2, CW // 2, 2).mean(axis=(2, 4))
    s = (np.arange(NIVELES) / (NIVELES - 1)) ** 1.7
    Y = np.empty((NIVELES + 1, n, CH, CW), np.uint8)
    U = np.empty((NIVELES + 1, n, CH // 2, CW // 2), np.uint8)
    V = np.empty_like(U)
    for l in range(NIVELES):
        Y[l] = 16 + (yc - 16) * s[l] * glifos
        U[l] = 128 + (uc - 128) * s[l] * sub
        V[l] = 128 + (vc - 128) * s[l] * sub
    Y[NIVELES] = 16 + 219 * glifos   # cabeza: blanca
    U[NIVELES] = 128
    V[NIVELES] = 128
    _atlas_cache[rgb] = (Y, U, V)
    return _atlas_cache[rgb]


def tocar_matrix(cfg):
    """Genera la lluvia de letras hasta que se desactive. Velocidad y color se releen cada segundo."""
    import numpy as np
    rng = np.random.default_rng()
    glifos = _glifos_np()
    ng = len(glifos)
    color = cfg.get("color") if cfg.get("color") in COLORES else "verde"
    Yat, Uat, Vat = _atlas(glifos, COLORES[color])
    F = np.zeros((ROWS, COLS), np.float32)                 # brillo de cada celda (0..1)
    G = rng.integers(0, ng, (ROWS, COLS))                  # glifo de cada celda
    cabeza = np.zeros((ROWS, COLS), bool)
    head = np.full(COLS, -1.0)                              # fila de la cabeza de cada columna (-1 = sin gota)
    vel = np.zeros(COLS)

    def ajustar(c):
        v = min(max(float(c.get("velocidad") or 5), 1), 10)
        q = min(max(float(c.get("cantidad") or 5), 1), 10)
        cps = 3 + 2.7 * v                                   # celdas por segundo (3..30)
        pausa = 420 / q ** 1.3                              # fotogramas medios entre gota y gota de una columna
        return v, q, cps, np.float32(.5 ** (cps / (6 * FPS))), pausa   # la cola se apaga en ~6 celdas

    velocidad, cantidad, cps, decay, pausa = ajustar(cfg)
    espera = rng.uniform(0, 2 * pausa, COLS)
    escribir_actual({"id": "matrix", "video": "matrix", "inicio": time.time(), "horas": 0,
                     "matrix": {"velocidad": velocidad, "cantidad": cantidad, "color": color}})
    mtime = None
    t0 = time.time()
    f = 0
    while True:
        F *= decay
        cabeza[:] = False
        espera -= 1
        nuevo = (head < 0) & (espera <= 0)
        head[nuevo] = 0
        vel[nuevo] = cps / FPS * rng.uniform(.6, 1.4, int(nuevo.sum()))
        activos = np.nonzero(head >= 0)[0]
        prev = np.floor(head[activos]).astype(int)
        head[activos] += vel[activos]
        cur = np.floor(head[activos]).astype(int)
        for c, a, b in zip(activos, prev, cur):
            for r in range(max(a, 0), min(b, ROWS - 1) + 1):
                if r != a or a == 0:
                    F[r, c] = 1.0
                    G[r, c] = rng.integers(ng)
            if b < ROWS:
                cabeza[b, c] = True
            elif b > ROWS + 2:
                head[c] = -1
                espera[c] = rng.uniform(.2, 1.8) * pausa
        m = rng.random((ROWS, COLS)) < .02                  # parpadeo: algunos glifos cambian
        G[m] = rng.integers(0, ng, int(m.sum()))
        L = np.minimum((F * (NIVELES - 1) + .5).astype(np.int64), NIVELES - 1)
        L[cabeza] = NIVELES
        Y = Yat[L, G].transpose(0, 2, 1, 3).reshape(H, W)
        U = Uat[L, G].transpose(0, 2, 1, 3).reshape(H // 2, W // 2)
        V = Vat[L, G].transpose(0, 2, 1, 3).reshape(H // 2, W // 2)
        out.write(Y.tobytes() + U.tobytes() + V.tobytes())
        f += 1
        if f % FPS == 0:                                    # una vez por segundo: ¿cambios?
            SALTAR.unlink(missing_ok=True)
            try:
                m2 = INFINITA.stat().st_mtime
            except FileNotFoundError:
                return
            if m2 != mtime:
                mtime = m2
                c2 = leer_infinita()
                if not c2:
                    return
                p_ant = pausa
                velocidad, cantidad, cps, decay, pausa = ajustar(c2)
                if pausa != p_ant:                          # la cantidad cambia ya: reescala las esperas
                    espera *= pausa / p_ant
                nuevo_color = c2.get("color") if c2.get("color") in COLORES else "verde"
                if nuevo_color != color:
                    color = nuevo_color
                    Yat, Uat, Vat = _atlas(glifos, COLORES[color])
                escribir_actual({"id": "matrix", "video": "matrix", "inicio": t0, "horas": 0,
                                 "matrix": {"velocidad": velocidad, "cantidad": cantidad, "color": color}})
        d = t0 + f / FPS - time.time()                      # ritmo real: 30 fotogramas por segundo
        if d > 0:
            time.sleep(d)
        elif d < -2:                                        # muy retrasado (p. ej. tras una pausa): resincroniza
            t0 = time.time() - f / FPS


def segundo_en_negro():
    """Sin videos disponibles: un segundo de negro para no cortar la emisión."""
    escribir_actual({})
    for _ in range(FPS):
        out.write(NEGRO)
        time.sleep(1 / FPS)


# Neblina, muy sutil (el usuario la quiere «como el humo del tabaco»): Perlin pequeño estirado en
# vertical (hilos), umbral alto para que solo queden los filamentos, vaivén lento por recorte (sin
# costura), escalado a 1080p y mezcla «screen» solo en luma a poca opacidad. ~0,5 núcleos más que
# decodificar a secas; por eso es por video y opcional.
NIEBLA = ("perlin=s=256x216:r=8:octaves=4:persistence=.6:xscale=1.4:yscale=.45:tscale=.05,"
          f"fps={FPS},lutyuv=y=clip((val-134)*7\\,0\\,255),crop=192:108:x=32*(1+sin(t/41)):y=54*(1-sin(t/29)),"
          f"scale={W}:{H}:flags=bilinear,format=gray,format=yuv420p")
NIEBLA_OPACIDAD = .2


def reproducir(item):
    """Repite un video hasta cumplir sus horas, saltar o desaparecer de la cola.
    Devuelve "repetir" si hay que relanzarlo (ha cambiado su neblina)."""
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
          f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={FPS},format=yuv420p")
    niebla = bool(item.get("niebla"))
    if niebla:
        cmd = ["ffmpeg", "-v", "error", "-re", "-stream_loop", "-1", "-i", str(D / item["video"]),
               "-f", "lavfi", "-i", NIEBLA, "-filter_complex",
               # ojo con blend: la opacidad mezcla entre la capa de ARRIBA (primera entrada) y el resultado,
               # así que el video va primero: luma = video + (screen(video, humo) - video) * opacidad;
               # el croma se queda tal cual (modo normal con opacidad 1 = capa de arriba)
               f"[0:v]{vf}[b];[b][1:v]blend=c0_mode=screen:c0_opacity={NIEBLA_OPACIDAD}:c1_mode=normal:c1_opacity=1:"
               f"c2_mode=normal:c2_opacity=1,format=yuv420p", "-an", "-f", "rawvideo", "-"]
    else:
        cmd = ["ffmpeg", "-v", "error", "-re", "-stream_loop", "-1", "-i", str(D / item["video"]),
               "-an", "-vf", vf, "-f", "rawvideo", "-"]
    hijo = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
    horas = float(item.get("hours") or 1)
    escribir_actual({"id": item["id"], "video": item["video"], "inicio": time.time(), "horas": horas, "niebla": niebla})
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
            # una vez por segundo: ¿saltar?, ¿video infinito?, ¿han cambiado las horas o la cola?
            if SALTAR.exists():
                SALTAR.unlink(missing_ok=True)
                return True
            if leer_infinita():
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
                if bool(igual.get("niebla")) != niebla:   # neblina activada/quitada: relanzar este video
                    return "repetir"
                if float(igual.get("hours") or 1) != horas:
                    horas = float(igual.get("hours") or 1)
                    escribir_actual({"id": item["id"], "video": item["video"],
                                     "inicio": time.time() - fotogramas / FPS, "horas": horas, "niebla": niebla})
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
        cfg = leer_infinita()
        if cfg:
            tocar_matrix(cfg)
            escribir_actual({})
            continue
        cola = leer_cola()
        if not cola:
            segundo_en_negro()
            continue
        ids = [c["id"] for c in cola]
        # el siguiente al que sonaba; si se acaba de quitar de la cola, el que ocupa su sitio
        pos = (ids.index(actual_id) + 1) if actual_id in ids else (pos if actual_id else 0)
        pos %= len(cola)
        if ALEATORIO.exists() and len(cola) > 1:   # al azar, sin repetir el que acaba de emitirse
            pos = random.choice([k for k in range(len(cola)) if ids[k] != actual_id])
        item = cola[pos]
        actual_id = item["id"]
        r = (D / item["video"]).exists() and reproducir(item)
        while r == "repetir":   # mismo video con la neblina cambiada
            item = next((c for c in leer_cola() if c["id"] == item["id"]), item)
            r = reproducir(item)
        if not r:
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
