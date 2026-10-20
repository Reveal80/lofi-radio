"""Escenas de ciudad generadas por código para los videos de fondo.

make_scene(ruta, tipo) pinta una imagen 1920x1080 (PPM) del tipo pedido y,
si la escena lleva lluvia o nieve, una capa de partículas 1920x2160 que
app.py hace caer en bucle con ffmpeg. Cada llamada sale distinta.

Tipos: noche, amanecer, dia, atardecer, lluvia, nieve, aurora y aleatorio.
"""
import numpy as np

W, H = 1920, 1080

TIPOS = {   # clave: nombre visible
    "aleatorio": "Aleatorio", "noche": "Noche", "amanecer": "Amanecer", "dia": "Día",
    "atardecer": "Atardecer", "lluvia": "Lluvia", "nieve": "Nieve", "aurora": "Aurora",
}

# variantes de noche: (nombre, cielo arriba, cielo abajo, ventanas)
_NOCHES = [
    ("Medianoche", [10, 12, 34], [42, 34, 74], [[255, 196, 110], [240, 170, 90]]),
    ("Noche violeta", [24, 10, 44], [112, 48, 96], [[255, 190, 120], [255, 140, 170]]),
    ("Noche azul hielo", [6, 18, 36], [34, 86, 116], [[150, 230, 255], [255, 240, 200]]),
    ("Noche neón", [16, 6, 30], [72, 18, 86], [[255, 80, 190], [80, 230, 255], [255, 230, 120]]),
    ("Noche ámbar", [30, 18, 22], [150, 78, 46], [[255, 200, 120]]),
]
CALIDA = [[255, 200, 120], [250, 180, 100]]


def _cfg(tipo, rng):
    """Parámetros de cada tipo de escena (con variación aleatoria)."""
    if tipo == "noche":
        n, top, bot, ven = _NOCHES[int(rng.integers(len(_NOCHES)))]
        return dict(nombre=n, top=top, bot=bot, estrellas=1, luna=.85, ventanas=ven, luz=(.18, .55),
                    edif=[12, 10, 18], reflejo=.35)
    if tipo == "aurora":
        return dict(nombre="Aurora", top=[4, 16, 26], bot=[20, 54, 64], estrellas=1, luna=.6, aurora=True,
                    ventanas=CALIDA, luz=(.15, .4), edif=[8, 12, 16], reflejo=.4)
    if tipo == "amanecer":
        return dict(nombre="Amanecer", top=[46, 62, 128], bot=[250, 168, 128], estrellas=.15, sol=dict(
                    y=(.62, .74), r=(60, 95), color=[255, 214, 170], halo=.55), nubes=(3, 7),
                    tinte_nubes=[255, 190, 170], ventanas=CALIDA, luz=(.05, .18), edif=[40, 34, 62], reflejo=.3)
    if tipo == "dia":
        return dict(nombre="Día", top=[58, 132, 214], bot=[176, 214, 242], sol=dict(
                    y=(.1, .3), r=(45, 70), color=[255, 252, 230], halo=.45), nubes=(5, 11),
                    tinte_nubes=[255, 255, 255], ventanas=[[190, 214, 236], [150, 180, 210]], luz=(.25, .5),
                    edif=[74, 86, 110], lejos=[132, 156, 188], reflejo=.15, suelo=[60, 66, 80])
    if tipo == "atardecer":
        return dict(nombre="Atardecer", top=[58, 30, 92], bot=[252, 126, 64], estrellas=.1, sol=dict(
                    y=(.6, .72), r=(85, 130), color=[255, 168, 90], halo=.7), nubes=(2, 6),
                    tinte_nubes=[255, 150, 120], ventanas=CALIDA, luz=(.1, .3), edif=[40, 20, 46], reflejo=.45)
    if tipo == "lluvia":
        return dict(nombre="Lluvia", top=[26, 32, 44], bot=[74, 84, 100], niebla=.35, nubes=(8, 14),
                    tinte_nubes=[120, 128, 140], ventanas=CALIDA, luz=(.2, .5), edif=[16, 18, 24],
                    reflejo=1, particulas="lluvia")
    if tipo == "nieve":
        return dict(nombre="Nieve", top=[20, 30, 56], bot=[92, 112, 142], estrellas=.2, luna=.4,
                    ventanas=CALIDA, luz=(.2, .5), edif=[22, 26, 40], nieve_tejados=True,
                    suelo=[150, 160, 180], reflejo=0, particulas="nieve")
    raise ValueError(tipo)


def _degradado(top, bot, rng):
    top, bot = np.array(top, float) * rng.uniform(.9, 1.1), np.array(bot, float) * rng.uniform(.9, 1.1)
    y = (np.linspace(0, 1, H) ** rng.uniform(.7, 1.5))[:, None, None]
    return np.broadcast_to(top * (1 - y) + bot * y, (H, W, 3)).copy(), top, bot


def _estrellas(img, rng, cantidad):
    for _ in range(int(rng.integers(300, 1000) * cantidad)):
        sx, sy = int(rng.integers(0, W)), int(rng.integers(0, int(H * .6)))
        b = rng.uniform(50, 210) * (1 - sy / H)
        img[sy, sx] += b
        if rng.random() < .25:
            img[max(sy - 1, 0):sy + 2, max(sx - 1, 0):sx + 2] += b * .3


def _disco(img, yy, xx, rng, x, y, r, color, halo):
    d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    img += np.exp(-np.maximum(d - r, 0) / rng.uniform(60, 130))[..., None] * np.array(color, float) * halo
    img[d < r] = color


def _nubes(img, rng, n, tinte, alto=.5):
    """Nubes suaves: grupos de manchas gaussianas alargadas."""
    capa = np.zeros((H, W))
    for _ in range(n):
        cx, cy = rng.uniform(-.1, 1.1) * W, rng.uniform(.05, alto) * H
        for _ in range(int(rng.integers(4, 9))):
            ox, oy = cx + rng.normal(0, 110), cy + rng.normal(0, 18)
            rx, ry = rng.uniform(90, 220), rng.uniform(25, 55)
            # gaussiana separable y solo en su caja (±4σ): igual resultado, ~20x más rápido
            x0, x1 = max(int(ox - 4 * rx), 0), min(int(ox + 4 * rx) + 1, W)
            y0, y1 = max(int(oy - 4 * ry), 0), min(int(oy + 4 * ry) + 1, H)
            if x1 > x0 and y1 > y0:
                gx = np.exp(-((np.arange(x0, x1) - ox) / rx) ** 2)
                gy = np.exp(-((np.arange(y0, y1) - oy) / ry) ** 2)
                capa[y0:y1, x0:x1] += gy[:, None] * gx[None, :]
    capa = np.clip(capa, 0, 1) * rng.uniform(.55, .85)
    img[:] = img * (1 - capa[..., None]) + np.array(tinte, float) * capa[..., None]


def _aurora(img, yy, xx, rng):
    for _ in range(int(rng.integers(2, 4))):
        base = rng.uniform(.15, .38) * H
        onda = base + 40 * np.sin(xx[0] / rng.uniform(120, 260) + rng.uniform(0, 6)) + 25 * np.sin(xx[0] / 70)
        sobre = onda[None, :] - yy
        banda = np.where(sobre > 0, np.exp(-sobre / rng.uniform(70, 150)), np.exp(-(sobre / 10) ** 2))
        fuerza = .5 + .5 * np.sin(xx[0] / rng.uniform(90, 200) + rng.uniform(0, 6)) ** 2
        color = np.array([[60, 220, 140], [150, 90, 230], [70, 200, 220]][int(rng.integers(3))], float)
        img += (banda * fuerza[None, :])[..., None] * color * rng.uniform(.18, .32)


def _edificios(img, rng, suelo, alto, color, ventanas=None, luz=0, ancho=(50, 150), nieve=False):
    x = int(rng.integers(-40, 0))
    while x < W:
        bw, bh = int(rng.integers(*ancho)), int(rng.integers(*alto))
        x0, x1 = max(x, 0), min(x + bw, W)
        if x1 > x0:
            img[suelo - bh:suelo, x0:x1] = color
            if rng.random() < .3:   # antena
                ax = x0 + (x1 - x0) // 2
                img[suelo - bh - int(rng.integers(10, 40)):suelo - bh, max(ax - 1, 0):ax + 2] = color
            if nieve:
                img[suelo - bh - 4:suelo - bh + 2, x0:x1] = [225, 232, 245]
            if ventanas is not None:
                tam = int(rng.integers(3, 6))
                paso = tam + int(rng.integers(5, 10))
                tinte = np.array(ventanas[int(rng.integers(len(ventanas)))], float)
                for wy in range(suelo - bh + 8, suelo - 8, paso):
                    for wx in range(x0 + 6, x1 - 6, paso):
                        if rng.random() < luz:
                            img[wy:wy + tam, wx:wx + tam] = tinte * rng.uniform(.6, 1.0)
        x += bw + int(rng.integers(0, 14))


def _capa_particulas(tipo, rng):
    """Capa 1920x2160 (dos mosaicos iguales) de lluvia o nieve sobre negro."""
    t = np.zeros((H, W))
    if tipo == "lluvia":
        filas, cols, vals = [], [], []
        for _ in range(int(rng.integers(900, 1500))):
            x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
            largo, b = int(rng.integers(18, 42)), rng.uniform(.25, .7)
            k = np.arange(largo)
            filas.append((y + k) % H)
            cols.append((x + k // 8) % W)
            vals.append(b * k / largo)
        np.maximum.at(t, (np.concatenate(filas), np.concatenate(cols)), np.concatenate(vals))
    else:
        for _ in range(int(rng.integers(900, 1600))):
            x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
            r, b = rng.choice([1, 1, 2, 2, 3]), rng.uniform(.5, 1)
            # los copos que tocan el borde se dibujan también al otro lado, para que
            # no se corten al empalmar los dos mosaicos del bucle
            for dy in (0, -H, H):
                ya, yb = max(y - r + dy, 0), min(y + r + dy, H)
                if yb > ya:
                    t[ya:yb, max(x - r, 0):x + r] = b
    rgb = (np.clip(np.vstack([t, t]), 0, 1)[..., None] * np.array([210, 220, 240])).astype(np.uint8)
    return rgb


def _guardar(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (img.shape[1], img.shape[0]))
        f.write(img.tobytes())


def make_scene(path, tipo="noche", capa_path=None):
    """Pinta la escena. Devuelve (título, tipo de partículas o None).
    Si hay partículas, guarda su capa en capa_path."""
    rng = np.random.default_rng()
    if tipo not in TIPOS or tipo == "aleatorio":
        tipo = str(rng.choice([t for t in TIPOS if t != "aleatorio"]))
    c = _cfg(tipo, rng)
    img, top, bot = _degradado(c["top"], c["bot"], rng)
    yy, xx = np.mgrid[0:H, 0:W]
    if c.get("estrellas"):
        _estrellas(img, rng, c["estrellas"])
    if c.get("aurora") or (tipo == "noche" and rng.random() < .1):
        _aurora(img, yy, xx, rng)
    if c.get("luna") and rng.random() < c["luna"]:
        mx, my, r = rng.uniform(.12, .88) * W, rng.uniform(.1, .32) * H, rng.uniform(30, 85)
        d = np.sqrt((xx - mx) ** 2 + (yy - my) ** 2)
        tono = np.array([240, 232, 210])
        img += np.exp(-np.maximum(d - r, 0) / rng.uniform(50, 110))[..., None] * tono * .28
        disco = d < r
        if rng.random() < .45:   # luna creciente
            ang = rng.uniform(0, 2 * np.pi)
            disco &= np.sqrt((xx - mx - np.cos(ang) * r * .55) ** 2 + (yy - my - np.sin(ang) * r * .55) ** 2) >= r * .95
        img[disco] = tono
    if c.get("sol"):
        s = c["sol"]
        _disco(img, yy, xx, rng, rng.uniform(.15, .85) * W, rng.uniform(*s["y"]) * H,
               rng.uniform(*s["r"]), s["color"], s["halo"])
    if c.get("nubes"):
        _nubes(img, rng, int(rng.integers(*c["nubes"])), c["tinte_nubes"],
               alto=.75 if tipo == "lluvia" else .5)
    if c.get("niebla"):
        img[:] = img * (1 - c["niebla"]) + np.array(c["bot"], float) * c["niebla"]
    # fondo: montañas o ciudad lejana
    suelo = int(H * rng.uniform(.76, .88))
    lejos = np.array(c.get("lejos") or (top * .5 + bot * .5) * .6, float)
    if rng.random() < .45:
        cresta = suelo - 120 - 80 * np.sin(xx[0] / rng.uniform(150, 320) + rng.uniform(0, 6)) - 40 * np.sin(xx[0] / 90)
        img[yy > cresta[None, :]] = lejos
    else:
        _edificios(img, rng, suelo, (140, 320), lejos, ancho=(40, 110))
    cerca = np.array(c["edif"], float)
    _edificios(img, rng, suelo, (70, 300), cerca, c["ventanas"], rng.uniform(*c["luz"]),
               nieve=c.get("nieve_tejados", False))
    img[suelo:] = c.get("suelo", cerca)
    if rng.random() < c.get("reflejo", 0):   # agua o calle mojada
        alto = H - suelo
        img[suelo:] = img[suelo - alto:suelo][::-1] * .35 + np.array(c.get("suelo", cerca), float) * .6
    _guardar(path, img)
    particulas = c.get("particulas")
    if particulas and capa_path:
        _guardar(capa_path, _capa_particulas(particulas, rng))
    return c["nombre"], particulas
