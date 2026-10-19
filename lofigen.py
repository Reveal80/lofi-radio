"""Motor de música lofi hip-hop procedural (render offline, sin IA).

Compone con reglas y azar: tonalidad mayor o menor, progresiones típicas del
género (con novenas a veces), patrón de acompañamiento, melodía, batería con
afinación propia de cada canción, bajo, sala de reverberación y texturas de
vinilo/lluvia/cinta. Estructura de 16 compases (entrada/A/B/respiro) y
guiños de 8 bits en el compás 15. Los estilos (ESTILOS) acotan las elecciones.
"""
from functools import lru_cache

import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 44100
_RNG = np.random.default_rng(0)


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def cents(c):
    return 2 ** (c / 1200)


@lru_cache(maxsize=512)
def _filtro(order, wn, tipo):
    """Coeficientes Butterworth en caché: el mismo corte se usa en miles de notas."""
    return signal.butter(order, wn, tipo)


def lowpass(x, fc, order=2):
    b, a = _filtro(order, min(fc, SR / 2 - 100) / (SR / 2), "low")
    return signal.lfilter(b, a, x)


def highpass(x, fc, order=2):
    b, a = _filtro(order, fc / (SR / 2), "high")
    return signal.lfilter(b, a, x)


def bandpass_q(x, fc, q=1.0):
    """Aproximación al biquad bandpass de Web Audio (ancho = fc/Q)."""
    bw = fc / max(q, 0.3)
    lo = max(40, fc - bw / 2)
    hi = min(SR / 2 - 200, fc + bw / 2)
    b, a = _filtro(1, (lo / (SR / 2), hi / (SR / 2)), "band")
    return signal.lfilter(b, a, x)


def _exp_ramp(v0, v1, n):
    """exponentialRampToValueAtTime: interpolación geométrica."""
    v0, v1 = max(v0, 1e-4), max(v1, 1e-4)
    return np.exp(np.linspace(np.log(v0), np.log(v1), max(n, 1)))


def reverb(x, mezcla, cola):
    """Sala: convolución con ruido que se apaga en `cola` segundos."""
    if mezcla <= 0:
        return x
    n = int(cola * SR)
    ir = _RNG.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    ir = lowpass(ir, 5000)
    ir /= np.sqrt((ir ** 2).sum())
    humedo = signal.fftconvolve(x, ir)[:len(x)]
    return x * (1 - mezcla) + humedo * mezcla * 1.6


def add(buf, x, start):
    i = int(start * SR)
    end = min(i + len(x), len(buf))
    if end > i:
        buf[i:end] += x[:end - i]

# ── los instrumentos (parámetros del fichero de referencia) ──

INST = {
    "rhodes": dict(armonicos=[(1, 1), (2, .18), (3, .05)], filtro=2200,
                   ataque=.012, caida=.4, sostiene=.35),
    "piano":  dict(armonicos=[(1, 1), (2, .35), (3, .12), (4, .05)], filtro=3600,
                   ataque=.004, caida=.25, sostiene=.18),
    "vibra":  dict(armonicos=[(1, 1), (4, .1)], filtro=3000,
                   ataque=.006, caida=.9, sostiene=.45, tremolo=5.5),
    "pad":    dict(onda="sawtooth", voces=3, filtro=950,
                   ataque=.5, caida=1.2, sostiene=.7),
    "pulso":  dict(onda="sawtooth", filtro=1800,
                   ataque=.008, caida=.12, sostiene=.05, punteo=True),
    # (suavizados el 2026-10-19: ataques más lentos y menos agudos, el usuario los oía como «golpes»)
    "caja":   dict(armonicos=[(1, 1), (3, .12)], filtro=2800,
                   ataque=.01, caida=.3, sostiene=.02),
    "chip":   dict(onda="square", filtro=2600,
                   ataque=.008, caida=.09, sostiene=.5),
    # nuevos timbres para que las canciones no suenen todas a «seno con filtro»
    "guitarra": dict(modo="cuerda", filtro=4200, ataque=.002, caida=1.4, sostiene=.25),
    "flauta":   dict(modo="flauta", filtro=3200, ataque=.07, caida=.4, sostiene=.7),
    "campana":  dict(modo="fm", filtro=3800, ataque=.01, caida=1.6, sostiene=.04),
    # vientos (2026-10-19, el usuario asocia el jazz al saxofón): diente de sierra o cuadrada
    # pasada por formantes fijos (como el cuerpo del instrumento), con vibrato tardío, un poco
    # de soplo y «scoop» (la nota entra ligeramente por debajo y sube), que es lo que da el fraseo
    "saxo":     dict(modo="viento", onda="sawtooth", formantes=[(520, 2.5, 1), (1250, 3, .7), (2400, 4, .25)],
                     filtro=3400, vib=(.45, 5.2, .22), soplo=(.07, 2200), scoop=.035, nivel=.95,
                     ataque=.06, caida=.5, sostiene=.62),
    "trompeta": dict(modo="viento", onda="sawtooth", formantes=[(1150, 4, 1), (1900, 5, .6)],   # con sordina
                     filtro=2800, vib=(.3, 5.8, .3), soplo=(.03, 3000), scoop=.02, nivel=.7,
                     ataque=.04, caida=.35, sostiene=.5),
    "clarinete": dict(modo="viento", onda="square", formantes=[(700, 1.8, 1), (1600, 2.5, .35)],
                     filtro=2400, vib=(.2, 5, .35), soplo=(.025, 1800), scoop=.012, nivel=.8,
                     ataque=.05, caida=.4, sostiene=.6),
    # templo japonés (2026-10-19): koto (cuerda pulsada), shakuhachi (flauta de bambú con mucho
    # soplo y entrada «meri» por debajo), bonshō (campana grave inarmónica que resuena segundos)
    # y cuenco tibetano (parciales inarmónicos con batido lento)
    "koto":       dict(modo="cuerda", filtro=3000, ataque=.002, caida=1.0, sostiene=.2),
    "shakuhachi": dict(modo="flauta", filtro=2600, soplo=.4, scoop=.03, ataque=.12, caida=.5, sostiene=.75),
    "bonsho":     dict(modo="fm", fm=(1.2, 1.41, 1.5), filtro=1500, octava=-12, ataque=.004, caida=4.5, sostiene=.15),
    "cuenco":     dict(armonicos=[(1, 1), (2.71, .5), (5.4, .18), (8.9, .06)], filtro=2400, tremolo=2.6,
                       ataque=.03, caida=3.0, sostiene=.3),
}


def _cuerda(f, n):
    """Cuerda pulsada (Karplus-Strong). El filtro promedio añade medio
    muestreo de retardo, así que el periodo es N + 0,5 (por eso el «- .5»:
    sin él las notas agudas salían desafinadas hacia abajo)."""
    N = max(2, int(round(SR / f - .5)))
    if N < 64:
        # notas agudas: pocas muestras por ciclo, así que redondear el periodo
        # desafina; se calcula a 4x de frecuencia de muestreo (filtro recursivo,
        # rápido con periodos cortos) y se reduce después
        N4 = int(round(4 * SR / f - .5))
        a = np.zeros(N4 + 2)
        a[[0, N4, N4 + 1]] = 1, -.4995, -.4995   # .4995 a 4x ≈ la misma caída que .498 a 1x
        x = np.zeros(4 * n + 1)
        x[1:N4 + 1] = _RNG.uniform(-1, 1, N4)
        return signal.resample_poly(signal.lfilter([1], a, x)[1:4 * n + 1], 1, 4)[:n]
    y = np.zeros(n + N + 1)
    y[1:N + 1] = _RNG.uniform(-1, 1, N)
    for s in range(N + 1, n + N + 1, N):
        e = min(s + N, n + N + 1)
        L = e - s
        y[s:e] = .996 * .5 * (y[s - N:s - N + L] + y[s - N - 1:s - N - 1 + L])
    return y[1:n + 1]


def _envolvente(n, a, caida, sost):
    """linearRamp al ataque, exponencial a vol*sost, exponencial a 0,0001."""
    na = min(int(a * SR), n)
    nd = min(int(caida * SR), max(n - na, 1))
    nr = max(n - na - nd, 1)
    return np.concatenate([
        np.linspace(0, 1, na, endpoint=False),
        _exp_ramp(1, max(sost, 1e-4), nd),
        _exp_ramp(max(sost, 1e-4), 1e-4, nr),
    ])[:n]


def tocar_nota(buf, inst, m, t, dur, vol, cinta=False):
    I = INST.get(inst, INST["rhodes"])
    m = m + I.get("octava", 0)
    bamboleo = np.sin(t * .9) * 14 if cinta else 0.0   # la cinta desafina
    n = int(max(dur, I["ataque"] + I["caida"] + .05) * SR)
    tt = np.arange(n) / SR
    out = np.zeros(n)
    modo, f0 = I.get("modo"), hz(m) * cents(bamboleo)
    if modo == "cuerda":
        out = _cuerda(f0, n)
    elif modo == "fm":     # campana: FM con índice que se apaga (índice, relación, caída del índice)
        ind, rel, cai = I.get("fm", (1.6, 3.5, .45))
        out = np.sin(2 * np.pi * f0 * tt + ind * np.exp(-tt / cai) * np.sin(2 * np.pi * f0 * rel * tt))
    elif modo == "viento":  # saxo, trompeta con sordina, clarinete
        prof, vel, tarda = I["vib"]
        vib = 1 + (prof / 100) * np.sin(2 * np.pi * vel * tt) * (1 - np.exp(-tt / tarda))
        vib *= 1 - I.get("scoop", 0) * np.exp(-tt / .06)   # entra un pelo por debajo y sube
        fase = 2 * np.pi * np.cumsum(f0 * vib) / SR
        src = signal.square(fase) if I["onda"] == "square" else signal.sawtooth(fase)
        for fc, q, w in I["formantes"]:
            out += w * bandpass_q(src, fc, q)
        out += .35 * lowpass(src, 900)   # el fundamental, para que no suene hueco
        out /= max(float(np.max(np.abs(out))), 1e-6)
        out *= I.get("nivel", 1)
        nivel_soplo, f_soplo = I["soplo"]
        out += bandpass_q(_RNG.standard_normal(n), f_soplo, 1.5) * nivel_soplo * np.exp(-tt / .25)
    elif modo == "flauta":  # seno con vibrato tardío y algo de soplo
        vib = 1 + .006 * np.sin(2 * np.pi * 5 * tt) * (1 - np.exp(-tt / .3))
        vib *= 1 - I.get("scoop", 0) * np.exp(-tt / .15)
        fase = 2 * np.pi * np.cumsum(f0 * vib) / SR
        out = np.sin(fase) + .25 * np.sin(2 * fase) + .08 * np.sin(3 * fase)
        out += bandpass_q(_RNG.standard_normal(n), min(f0 * 2, 8000), 2) * I.get("soplo", .15)
    elif "onda" in I:
        voces = I.get("voces", 1)
        for v in range(voces):
            det = (v - (voces - 1) / 2) * 9 + bamboleo
            f = hz(m) * cents(det)
            if I["onda"] == "square":
                out += signal.square(2 * np.pi * f * tt)
            else:
                out += signal.sawtooth(2 * np.pi * f * tt)
        out /= voces
    else:
        for h, w in I["armonicos"]:
            det = (_RNG.random() - .5) * 7 + bamboleo
            out += w * np.sin(2 * np.pi * hz(m) * cents(det) * h * tt)
    if I.get("punteo"):   # barrido del filtro: ×2,2 -> 380 Hz en 0,35 s
        hi, lo = lowpass(out, I["filtro"] * 2.2), lowpass(out, 380)
        w = np.exp(-tt / .12)
        out = hi * w + lo * (1 - w)
    else:
        out = lowpass(out, I["filtro"])
    if I.get("tremolo"):
        out *= .65 + .35 * np.sin(2 * np.pi * I["tremolo"] * tt)
    out *= _envolvente(n, I["ataque"], I["caida"], I["sostiene"])
    add(buf, out * vol, t)


_BAJO = "triangulo"   # timbre del bajo de la canción en curso


def bajo_nota(buf, m, t, dur, vol):
    n = int((dur + .05) * SR)
    tt = np.arange(n) / SR
    f = hz(m)
    if _BAJO == "sub":          # seno redondo y profundo
        x = np.sin(2 * np.pi * f * tt) * 1.1
    elif _BAJO == "cuerda":     # bajo eléctrico pulsado
        x = lowpass(_cuerda(f, n), 900) * 2.2
    elif _BAJO == "sinte":      # sierra filtrada, más presente
        x = lowpass(signal.sawtooth(2 * np.pi * f * tt), 650) * .9
    else:                       # triángulo (el de siempre)
        ph = (f * tt) % 1.0
        x = lowpass(4 * np.abs(ph - .5) - 1, 420)
    na = int(.02 * SR)
    env = np.concatenate([np.linspace(0, 1, na, endpoint=False),
                          _exp_ramp(1, 1e-4, max(n - na, 1))])[:n]
    add(buf, x * env * vol, t)

# ── la batería ──

# cada canción afina su batería: bombo más grave o más seco, caja más aguda, etc.
KIT_BASE = dict(k0=115, k1=42, kd=.34, sn=1.0, sd=.16, hh=1.0, shaker=False)
_KIT = dict(KIT_BASE)


def bombo(buf, t, vol, kicks):
    k0, k1, kd = _KIT["k0"], _KIT["k1"], _KIT["kd"]
    n = int((kd + .02) * SR)
    tt = np.arange(n) / SR
    f = np.where(tt < .22, k0 * (k1 / k0) ** (tt / .22), k1)
    env = (1e-4) ** (np.minimum(tt / kd, 1))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * env
    add(buf, x * vol, t)
    kicks.append(t)


def golpe(buf, t, vol, tipo, frec, dur):
    n = int((dur + .02) * SR)
    tt = np.arange(n) / SR
    ruido = _RNG.standard_normal(n)
    if tipo == "bandpass":
        x = bandpass_q(ruido, frec, 1.0)
    else:
        x = highpass(ruido, frec)
    env = (1e-4) ** (np.minimum(tt / dur, 1))
    add(buf, x * env * vol, t)


RITMOS = {
    "boombap": {"K": "x..x....", "S": "..x...x.", "H": "xxxxxxxx"},
    "medio":   {"K": "x.......", "S": "....x...", "H": "x.x.x.x."},
    "shuffle": {"K": "x....x..", "S": "..r...r.", "H": "x.xxx.xo"},
    "roto":    {"K": "x..x..x.", "S": "...x...x", "H": "xx.xx.xx"},
    "suave":   {"K": "x...x...", "S": "..r...r.", "H": ".x.x.x.x"},
    "dilla":   {"K": "x..x.x..", "S": "..x...x.", "H": "x.x.x.xx"},
    "perezoso": {"K": "x.....x.", "S": "....x...", "H": "xxxxxxxx"},
    "doble":   {"K": "x.x...x.", "S": "..x..rx.", "H": "x.xxx.x."},
    "nada":    {"K": "........", "S": "........", "H": "........"},
    "templo":  {"K": "........", "S": "........", "H": "........", "W": "x.....x."},   # W = bloque de madera
}

# ── las texturas (bucle de ruido de fondo, como el buffer de la referencia) ──

def textura(n, tipo):
    lluvia = tipo == "lluvia"
    d = _RNG.uniform(-1, 1, n) * (.22 if lluvia else .05)
    pops = _RNG.random(n) < (.002 if lluvia else .0004)
    d[pops] += _RNG.uniform(-1, 1, pops.sum()) * (.5 if lluvia else .9)
    if lluvia:
        d = lowpass(d, 1600)
        return d * .16
    d = bandpass_q(d, 3200, .5)
    return d * .1

# ── los guiños de 8 bits (compás 15; rasgos de estilo, no melodías de nadie) ──

def arpegio_chip(buf, notas, t, paso, vol):
    for k, m in enumerate(notas):
        tocar_nota(buf, "chip", m, t + k * paso, paso * 1.6, vol)


def guino(buf, T, t, corchea):
    tn = T.get("tonica") or (T["acordes"][0][0] + 12)
    g = T.get("guino", "")
    if g == "cadencia":
        semi = corchea / 2
        acs = [[tn - 4, tn, tn + 3], [tn - 2, tn + 2, tn + 5], [tn, tn + 4, tn + 7]]
        for r in range(2):
            arpegio_chip(buf, acs[0] + [acs[0][0] + 12], t + r * semi * 2, semi / 2, .022)
        for r in range(2):
            arpegio_chip(buf, acs[1] + [acs[1][0] + 12], t + corchea * 2 + r * semi * 2, semi / 2, .022)
        for m in acs[2] + [tn + 12]:
            tocar_nota(buf, "chip", m, t + corchea * 4, corchea * 3.5, .018)
        arpegio_chip(buf, [tn + 12, tn + 16, tn + 19, tn + 24], t + corchea * 4, corchea / 4, .02)
    elif g == "secreto":
        esc = [0, 4, 7, 11, 14, 16, 19, 23, 24]
        arpegio_chip(buf, [tn + x for x in esc], t + corchea * 2, corchea / 3, .02)
    elif g == "monedas":
        for k in (1, 4, 6):
            if _RNG.random() < .8:
                arpegio_chip(buf, [tn + 23, tn + 28], t + k * corchea, corchea / 4, .018)

# ── el compás y la estructura de 16 (port directo de compas()) ──

def render_theme(T, cycles, seed, continuo=False, fondo=1.0):
    """Renderiza el tema. Con continuo=True (radio infinita) no hay entrada filtrada ni fundidos
    y devuelve (cuerpo, cola): los compases exactos y los 4 s de resonancia que se superponen
    al tramo siguiente, para enlazar tramos sin cortes. fondo: nivel de la textura (vinilo, lluvia,
    cinta); 0 la quita (en la radio infinita el usuario la oía como ruido blanco)."""
    global _RNG, _KIT, _BAJO
    _RNG = np.random.default_rng(seed)
    _KIT = {**KIT_BASE, **T.get("kit", {})}
    _BAJO = T.get("bajoTimbre", "triangulo")
    niv = {"acordes": 1, "melodia": 1, "bateria": 1, **T.get("niveles", {})}
    negra = 60 / T["bpm"]
    corchea = negra / 2
    sw = T.get("sw", .12) * corchea
    compas_dur = negra * 4
    total = int((cycles * 16 * compas_dur + 4) * SR)

    bus = np.zeros(total)      # acordes (con bombeo)
    resto = np.zeros(total)    # bajo y guiños
    melb = np.zeros(total)     # melodía (va a la sala con los acordes)
    bat = np.zeros(total)      # batería
    kicks = []
    cinta = T.get("tex") == "cinta"
    inst = T.get("inst", "rhodes")
    inst_mel = T.get("instMel", "vibra")
    R = RITMOS.get(T.get("bat", "boombap"), RITMOS["boombap"])
    vol_ac = (.06 if inst == "pad" else .085) * niv["acordes"]

    for paso in range(cycles * 16):
        t = paso * compas_dur
        c = paso % 16
        parte = "entrada" if c < 2 else "A" if c < 10 else "B" if c < 14 else "respiro"
        lista = T.get("acordesB") if (parte == "B" and T.get("acordesB")) else T["acordes"]
        ac = lista[c % len(lista)]
        cuando = lambda i: t + i * corchea + (sw if i % 2 else 0)
        # acordes, según el patrón de acompañamiento de la canción
        comp = T.get("comp", "largo")
        va = vol_ac * (4 / len(ac)) ** .5   # las novenas (5 notas) no suenan más fuerte
        acorde = lambda i, d, v: [tocar_nota(bus, inst, m, cuando(i), d, v, cinta) for m in ac]
        if comp == "pad":
            acorde(0, negra * 4, va * .9)
        elif comp == "stabs" and parte != "entrada":
            for i in (0, 3, 6):
                acorde(i, corchea * 1.3, va * (1 if i == 0 else .7))
        elif comp == "charleston" and parte != "entrada":
            acorde(0, corchea * 2.6, va)
            acorde(3, corchea * 2.2, va * .75)
        elif comp == "arpegio":
            notas = ac + [ac[1] + 12]
            for i, k in enumerate((0, 1, 2, 3, 4, 3, 2, 1)):
                tocar_nota(bus, inst, notas[k % len(notas)], cuando(i), corchea * 1.8, va * .8, cinta)
        elif comp == "roto" and parte != "entrada":
            tocar_nota(bus, inst, ac[0], cuando(0), negra * 3.5, va, cinta)
            for i, m in zip((2, 3, 4, 6), ac[1:] + ac[1:]):
                tocar_nota(bus, inst, m, cuando(i), corchea * 2.5, va * .75, cinta)
        else:   # «largo»: en el 1, con golpe extra en la corchea 6 de los impares
            acorde(0, negra * 3.7, va)
            if inst != "pad" and parte != "entrada" and c % 2:
                acorde(5, negra * 1.4, va * .6)
        # el bajo, según su estilo (dos octavas abajo)
        if parte != "entrada":
            r = ac[0] - 12   # una octava abajo: audible también en altavoces pequeños
            est = T.get("bajo", "largo")
            if est == "largo":
                bajo_nota(resto, r, cuando(0), negra * 3.5, .075)
            elif est == "rebote":
                bajo_nota(resto, r, cuando(0), negra * .9, .075)
                bajo_nota(resto, r + 12, cuando(3), negra * .6, .05)
                bajo_nota(resto, r, cuando(4), negra * 1.4, .065)
            else:   # caminante
                for j, k in enumerate((0, 2, 4, 6)):
                    bajo_nota(resto, r + (0, 3, 5, 7)[j], cuando(k), negra * .9, .065)
        # la batería: ni en la entrada ni en el respiro (y a veces la B va sin bombo)
        if parte in ("A", "B"):
            sin_bombo = parte == "B" and T.get("break")
            for i in range(8):
                if R["K"][i] == "x" and not sin_bombo:
                    bombo(bat, cuando(i), .34 if i == 0 else .26, kicks)
                if R["S"][i] == "x":
                    golpe(bat, cuando(i), .13, "bandpass", 1500 * _KIT["sn"], _KIT["sd"])
                elif R["S"][i] == "r":
                    golpe(bat, cuando(i), .08, "bandpass", 2600 * _KIT["sn"], .05)
                if R["H"][i] == "x":
                    golpe(bat, cuando(i), .018 if i % 2 else .028, "highpass", 7500 * _KIT["hh"], .045)
                elif R["H"][i] == "o":
                    golpe(bat, cuando(i), .025, "highpass", 6500 * _KIT["hh"], .22)
                if _KIT["shaker"]:   # maraca suave a semicorcheas
                    golpe(bat, cuando(i) + corchea / 2, .008, "highpass", 9000, .03)
                if R.get("W", "........")[i] == "x":   # mokugyo: golpe hueco de madera
                    golpe(bat, cuando(i), .09, "bandpass", 1000, .035)
            if c in (9, 13) and "x" in R["S"]:   # redoble suave antes de cambiar
                golpe(bat, cuando(7), .07, "bandpass", 1200, .3)
        if T.get("bonsho") and c % 8 == 0:   # campana de templo al empezar cada 8 compases
            tocar_nota(resto, "bonsho", T.get("raiz", 50) - 12, t, 5, .13)
        # el gancho: en A tal cual, en B una octava arriba y no siempre entero
        if parte in ("A", "B") and T.get("mel"):
            frase = T["mel"][(c - 2) % len(T["mel"])]
            sube = 12 if parte == "B" and T.get("subeB", True) else 0
            for k, m in enumerate(frase):
                if not m or (parte == "B" and _RNG.random() < .2):
                    continue
                tocar_nota(melb, inst_mel, m + sube, cuando(k), corchea * 1.7, .065 * niv["melodia"], cinta)
        if T.get("guino") and c == 15:
            guino(resto, T, t, corchea)

    # el bombeo: con cada bombo el bus de acordes baja a 0,45 y vuelve en 0,28 s
    pump = np.ones(total)
    rec = int(.28 * SR)
    curva = .45 + .55 * np.linspace(0, 1, rec)
    for tk in kicks:
        i = int(tk * SR)
        j = min(i + rec, total)
        pump[i:j] = np.minimum(pump[i:j], curva[:j - i])

    sala = T.get("sala", (0, 1))
    mix = reverb(bus * pump + melb, *sala) + resto + bat * niv["bateria"]
    if T.get("lofi"):   # carácter de la canción: más o menos «apagada»
        mix = lowpass(mix, T["lofi"])
    if T.get("intro_filtro") and not continuo:   # la entrada suena como a través de una pared y se abre
        fin = int(2 * compas_dur * SR)
        apagado = lowpass(mix[:fin], 450)
        w = np.linspace(0, 1, fin) ** 2
        mix[:fin] = apagado * (1 - w) + mix[:fin] * w
    if fondo:
        mix += textura(total, T.get("tex", "vinilo")) * .5 * T.get("texVol", 1) * fondo
    if not continuo:
        f_in, f_out = int(2.5 * SR), int(3 * SR)
        mix[:f_in] *= np.linspace(0, 1, f_in)
        mix[-f_out:] *= np.linspace(1, 0, f_out)
    # máster: nivela a ~-16 dBFS RMS y redondea los picos con un limitador suave
    mix = mix / (np.sqrt(np.mean(mix ** 2)) + 1e-9) * 10 ** (-16 / 20)
    mix = np.tanh(mix * 1.4) / 1.4
    if continuo:   # sin normalizar al pico: así todos los tramos salen al mismo nivel
        corte = int(cycles * 16 * compas_dur * SR)
        cola = mix[corte:] * np.linspace(1, 0, total - corte)
        return mix[:corte], cola
    return mix / np.max(np.abs(mix)) * .92

# ── compositor de temas nuevos ──

_QUAL = {"maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10), "7": (0, 4, 7, 10),
         "6": (0, 4, 7, 9), "maj9": (0, 4, 7, 11, 14), "m9": (0, 3, 7, 10, 14), "9": (0, 4, 7, 10, 14),
         "5": (0, 7, 12), "sus2": (0, 2, 7), "sus4": (0, 5, 7)}   # abiertos, para el templo
ESCALAS = {   # pentatónicas japonesas (semitonos desde la tónica)
    "hirajoshi": (0, 2, 3, 7, 8), "insen": (0, 1, 5, 7, 10), "yo": (0, 2, 5, 7, 9),
}
_PROGS_MENOR = [   # en menor natural (semitonos desde la tónica)
    [(0, "m7"), (8, "maj7"), (3, "maj7"), (10, "7")],       # i-VI-III-VII
    [(0, "m7"), (5, "m7"), (10, "7"), (3, "maj7")],         # i-iv-VII-III
    [(0, "m7"), (7, "m7"), (8, "maj7"), (5, "m7")],         # i-v-VI-iv
    [(5, "m7"), (10, "7"), (3, "maj7"), (8, "maj7")],       # iv-VII-III-VI
    [(0, "m7"), (0, "m7"), (8, "maj7"), (7, "7")],          # i-i-VI-V
]
_NOVENA = {"maj7": "maj9", "m7": "m9", "7": "9"}
_PROGS = [   # progresiones típicas del género, en grados de la escala mayor
    [(2, "m7"), (7, "7"), (0, "maj7"), (9, "m7")],          # ii-V-I-vi
    [(9, "m7"), (2, "m7"), (7, "7"), (0, "maj7")],          # vi-ii-V-I
    [(5, "maj7"), (4, "m7"), (2, "m7"), (0, "maj7")],       # IV-iii-ii-I
    [(0, "maj7"), (5, "maj7"), (0, "maj7"), (5, "maj7")],   # I-IV con maj7
    [(0, "maj7"), (9, "m7"), (5, "maj7"), (7, "7")],        # I-vi-IV-V
]


def _acordes_de(prog, tonica):
    out = []
    for grado, q in prog:
        root = tonica + grado
        while root > 62:
            root -= 12
        while root < 46:
            root += 12
        out.append([root + s for s in _QUAL[q]])
    return out


_PATRONES_MEL = {
    "escasa": [(0,), (0, 4), (2,), (0, 6), (4,), (0, 3)],
    "normal": [(0, 2, 4), (0, 2, 4, 6), (0, 3, 4, 6), (2, 4, 6), (0, 2, 5), (0, 4, 6)],
    "densa":  [(0, 1, 2, 4, 6), (0, 2, 3, 4, 6), (0, 1, 3, 4, 5, 6), (0, 2, 4, 5, 6), (0, 1, 2, 3, 4)],
}


def _mel_aleatoria(rng, tonica, acordes, menor=False, densidad="normal", grados=None):
    """Gancho: 4 frases de 8 corcheas; la densidad marca cuántas notas lleva."""
    grados = grados or ((0, 2, 3, 5, 7, 8, 10) if menor else (0, 2, 4, 5, 7, 9, 11))
    esc = sorted({tonica + o + d for o in (12, 24) for d in grados})
    idx = min(range(len(esc)), key=lambda i: abs(esc[i] - (tonica + 19)))
    pats = _PATRONES_MEL.get(densidad, _PATRONES_MEL["normal"])
    frases = []
    for bar in range(4):
        frase = [0] * 8
        for o in pats[rng.integers(len(pats))]:
            idx = int(np.clip(idx + rng.choice([-2, -1, -1, 1, 1, 2]),
                              0, len(esc) - 1))
            nota = esc[idx]
            if o in (0, 4):   # tiempos fuertes sobre notas del acorde
                pcs = {p % 12 for p in acordes[bar]}
                cand = [x for x in esc if x % 12 in pcs] or [nota]
                nota = min(cand, key=lambda x: abs(x - nota))
                idx = esc.index(nota)
            frase[o] = nota
        frases.append(frase)
    fin = min((x for x in esc if x % 12 == tonica % 12),
              key=lambda x: abs(x - esc[idx]))
    frases[3] = [fin] + [0] * 7   # la última frase cierra en la tónica
    return frases


# ── estilos: acotan las elecciones del compositor (cada canción sigue siendo distinta) ──
# bpm y swing son rangos; el resto, listas entre las que se elige (repetir = más probable).
# menor/novena/guino/partB/break/intro son probabilidades; «comp» es el patrón de
# acompañamiento de los acordes y «densidad» cuántas notas lleva la melodía.
TODOS = dict(
    inst=["rhodes", "rhodes", "piano", "piano", "vibra", "pad", "pulso", "guitarra", "guitarra"],
    instMel=["vibra", "piano", "caja", "rhodes", "pulso", "chip", "flauta", "campana", "guitarra", "saxo", "trompeta", "clarinete", "shakuhachi", "koto"],
    bat=["boombap", "medio", "shuffle", "roto", "suave", "dilla", "perezoso", "doble"],
    bajo=["largo", "rebote", "caminante"], tex=["vinilo", "vinilo", "lluvia", "cinta"],
    comp=["largo", "pad", "stabs", "charleston", "arpegio", "roto"],
    densidad=["escasa", "normal", "normal", "densa"],
    bajoTimbre=["triangulo", "sub", "cuerda", "sinte"],
)
ESTILOS = {
    "aleatorio": dict(nombre="Aleatorio", desc="Mezcla de todo: cualquier tempo, tonalidad, instrumento y ritmo."),
    "estudiar": dict(nombre="Para estudiar", desc="El lofi clásico: boom-bap tranquilo, rhodes, piano o guitarra y vinilo.",
                     bpm=(70, 84), sw=(.1, .18), menor=.35, inst=["rhodes", "piano", "guitarra"],
                     instMel=["vibra", "piano", "rhodes", "guitarra"], bat=["boombap", "medio", "perezoso", "dilla"],
                     bajo=["largo", "rebote"], tex=["vinilo", "vinilo", "cinta"], comp=["largo", "stabs", "roto"],
                     guino=.08),
    "chill": dict(nombre="Chill", desc="Lento y suave: pads o arpegios, flauta o campanas y batería ligera.",
                  bpm=(62, 74), sw=(.06, .12), menor=.3, inst=["pad", "rhodes", "vibra", "guitarra"],
                  instMel=["vibra", "flauta", "campana", "piano"], bat=["suave", "medio", "perezoso"], bajo=["largo"],
                  tex=["vinilo", "cinta"], comp=["pad", "arpegio", "largo"], densidad=["escasa", "normal"], guino=0, sala=(.15, .4)),
    "jazz": dict(nombre="Jazz hop", desc="Más swing, acordes con novenas, saxo o trompeta con sordina y bajo caminante.",
                 bpm=(80, 96), sw=(.16, .24), menor=.5, novena=.9, inst=["piano", "rhodes", "guitarra"],
                 instMel=["saxo", "saxo", "trompeta", "clarinete", "vibra", "piano"], bat=["shuffle", "dilla", "boombap", "doble"],
                 bajo=["caminante", "caminante", "rebote"], tex=["vinilo"], comp=["charleston", "stabs", "roto"],
                 densidad=["normal", "densa"], guino=0, progs=[0, 1, 2], partB=.85, sala=(.05, .2)),
    "lluvioso": dict(nombre="Día de lluvia", desc="Melancólico y en menor, con lluvia de fondo y ritmo suave.",
                     bpm=(60, 72), sw=(.08, .14), menor=.9, inst=["piano", "rhodes", "guitarra"],
                     instMel=["piano", "clarinete", "vibra", "flauta", "saxo"], bat=["suave", "medio", "perezoso"], bajo=["largo"],
                     tex=["lluvia"], comp=["arpegio", "roto", "largo"], densidad=["escasa", "normal"], guino=0),
    "retro": dict(nombre="Retro 8 bits", desc="Lofi con sintes de videojuego, arpegios rápidos y guiños chiptune.",
                  bpm=(82, 96), sw=(.12, .2), menor=.4, inst=["pulso", "pulso", "rhodes"], instMel=["chip", "chip", "pulso"],
                  bat=["shuffle", "roto", "doble", "boombap"], bajo=["rebote", "caminante"], tex=["vinilo", "cinta"],
                  comp=["arpegio", "stabs"], densidad=["normal", "densa"], guino=1, sala=(.03, .15)),
    "ambiente": dict(nombre="Ambiente", desc="Sin batería: pads, campanas y flauta para fondo.",
                     bpm=(56, 66), sw=(.04, .08), menor=.5, inst=["pad", "pad", "vibra"], instMel=["campana", "flauta", "vibra"],
                     bat=["nada"], bajo=["largo"], tex=["cinta", "lluvia", "vinilo"], comp=["pad", "arpegio"],
                     densidad=["escasa"], guino=0, intro=1, sala=(.3, .55)),
    "templo": dict(nombre="Templo japonés", desc="Escalas pentatónicas japonesas, koto, shakuhachi, cuencos y campana de templo; muy lento y espacioso.",
                   bpm=(50, 60), sw=(0, .04), menor=1.0, novena=0, inst=["koto", "cuenco", "cuenco", "pad"],
                   instMel=["shakuhachi", "shakuhachi", "koto", "campana"], bat=["templo", "templo", "nada"], bajo=["largo"],
                   tex=["lluvia", "cinta"], comp=["arpegio", "largo", "pad"], densidad=["escasa"], guino=0, intro=0,
                   sala=(.4, .65), escalas=["hirajoshi", "hirajoshi", "insen", "yo"], bonsho=.8, partB=.3,
                   progs_propios=[[(0, "5"), (0, "5"), (7, "5"), (0, "5")], [(0, "5"), (3, "5"), (7, "sus2"), (0, "5")],
                                  [(0, "sus2"), (10, "5"), (0, "5"), (7, "5")], [(0, "5"), (5, "sus2"), (0, "5"), (8, "5")]]),
    "nocturno": dict(nombre="Nocturno", desc="Para dormir: muy lento, en menor, cinta antigua y ritmo casi ausente.",
                     bpm=(54, 62), sw=(.04, .1), menor=.85, inst=["pad", "rhodes", "guitarra"],
                     instMel=["flauta", "saxo", "clarinete", "campana"], bat=["nada", "suave"], bajo=["largo"], tex=["cinta"],
                     comp=["pad", "arpegio"], densidad=["escasa"], guino=0, sala=(.25, .5)),
}


def estilo_de(nombre):
    """Estilo por nombre; «chill+jazz» es la mezcla de varios: se suman sus listas de
    instrumentos, baterías…, y los rangos (tempo, swing, sala) y probabilidades se promedian."""
    partes = [x for x in str(nombre or "").split("+") if x in ESTILOS]
    if len(partes) <= 1:
        return ESTILOS.get(partes[0] if partes else nombre, ESTILOS["aleatorio"])
    Es = [ESTILOS[x] for x in partes]
    M = {"nombre": " + ".join(E["nombre"] for E in Es), "desc": "mezcla"}
    for k in ("inst", "instMel", "bat", "bajo", "tex", "comp", "densidad"):
        listas = [E[k] for E in Es if k in E]
        if len(listas) == len(Es):   # si alguno no acota (aleatorio), la mezcla tampoco
            M[k] = sum(listas, [])
    for k in ("bpm", "sw", "sala"):
        rangos = [E[k] for E in Es if k in E]
        if rangos:
            M[k] = (sum(r[0] for r in rangos) / len(rangos), sum(r[1] for r in rangos) / len(rangos))
    for k in ("menor", "novena", "guino", "partB", "break", "intro"):
        vals = [E[k] for E in Es if k in E]
        if vals:
            M[k] = sum(vals) / len(vals)
    progs = [E["progs"] for E in Es if "progs" in E]
    if progs:
        M["progs"] = sorted(set(sum(progs, [])))
    return M


def tema_aleatorio(seed, estilo="aleatorio"):
    """Compone un tema nuevo con las reglas del motor de referencia,
    acotado por el estilo elegido (o mezcla «a+b»)."""
    E = estilo_de(estilo)
    rng = np.random.default_rng(seed)
    elige = lambda clave: str(rng.choice(E.get(clave, TODOS[clave])))
    menor = rng.random() < E.get("menor", .45)
    tonica = int(rng.integers(48, 59))
    if menor:
        progs = _PROGS_MENOR
    else:
        progs = [_PROGS[i] for i in E.get("progs", range(len(_PROGS)))]
    if E.get("progs_propios"):   # el estilo trae sus propias vueltas (acordes abiertos del templo)
        progs = [list(p) for p in E["progs_propios"]]
    novena = rng.random() < E.get("novena", .35) and not E.get("progs_propios")
    con9 = lambda p: [(g, _NOVENA.get(q, q)) for g, q in p] if novena else p
    prog = con9(progs[rng.integers(len(progs))])
    escala = str(rng.choice(E["escalas"])) if E.get("escalas") else None
    acordes = _acordes_de(prog, tonica)
    bpm_min, bpm_max = E.get("bpm", (60, 92))
    sw_min, sw_max = E.get("sw", (.06, .2))
    tema = dict(
        bpm=int(rng.integers(int(bpm_min), int(bpm_max) + 1)),
        sw=round(float(rng.uniform(sw_min, sw_max)), 2),
        inst=elige("inst"), instMel=elige("instMel"), bat=elige("bat"),
        bajo=elige("bajo"), tex=elige("tex"), comp=elige("comp"),
        modo="menor" if menor else "mayor",
        acordes=acordes,
        # batería propia: afinación del bombo, tono de la caja, brillo de los platillos
        kit=dict(k0=float(rng.uniform(92, 135)), k1=float(rng.uniform(38, 52)), kd=float(rng.uniform(.24, .45)),
                 sn=float(rng.uniform(.75, 1.4)), sd=float(rng.uniform(.1, .22)), hh=float(rng.uniform(.8, 1.25)),
                 shaker=bool(rng.random() < .35)),
        lofi=float(rng.uniform(2000, 11000)),          # cuánto se «apaga» el sonido
        bajoTimbre=elige("bajoTimbre"),
        sala=(float(rng.uniform(*E.get("sala", (.05, .4)))), float(rng.uniform(.8, 3.2))),   # reverberación
        niveles=dict(acordes=float(rng.uniform(.75, 1.25)), melodia=float(rng.uniform(.65, 1.35)),
                     bateria=float(rng.uniform(.6, 1.25))),
        texVol=float(rng.uniform(.2, 1.3)),
        intro_filtro=bool(rng.random() < E.get("intro", .4)),
        **({"break": True} if rng.random() < E.get("break", .25) else {}),
        **({"escala": escala} if escala else {}),
        **({"bonsho": True} if rng.random() < E.get("bonsho", 0) else {}),
    )
    if tema["inst"] == "pad" and tema["comp"] in ("stabs", "charleston", "roto"):
        tema["comp"] = "pad"   # un pad no se toca a golpes cortos
    if rng.random() < E.get("partB", .5):   # parte B con otra vuelta de acordes
        otra = con9(progs[rng.integers(len(progs))])
        tema["acordesB"] = _acordes_de(otra[2:] + otra[:2], tonica)
    if rng.random() < E.get("guino", .4):   # guiño de 8 bits en el respiro
        tema["guino"] = str(rng.choice(["cadencia", "secreto", "monedas"]))
        tn = tonica + 12
        while tn < 60:
            tn += 12
        tema["tonica"] = tn
    tema["mel"] = _mel_aleatoria(rng, tonica, acordes, menor, elige("densidad"), ESCALAS.get(escala))
    tema["raiz"] = tonica   # para poder componer otra melodía sobre los mismos acordes (radio infinita)
    return tema


# ── radio infinita: tramos encadenados que cambian poco a poco ──

_GRUPOS = {   # parámetros que viajan juntos al pasar de un tramo al siguiente
    "armonia": ("modo", "acordes", "acordesB", "mel", "guino", "tonica", "raiz", "escala", "bonsho"),
    "inst": ("inst",), "instMel": ("instMel",), "bat": ("bat", "kit", "break"), "bajo": ("bajo", "bajoTimbre"),
    "comp": ("comp",), "tex": ("tex", "texVol"), "sala": ("sala",), "lofi": ("lofi",), "niveles": ("niveles",),
}
_ACOTADOS = ("inst", "instMel", "bat", "bajo", "tex", "comp")   # listas que cada estilo restringe


def tema_siguiente(prev, seed, estilo="aleatorio", bpm=None):
    """Compone el tramo siguiente de la radio infinita: hereda casi todo del anterior y cambia
    uno o dos grupos de parámetros. Si el estilo pedido no admite algo del tramo anterior
    (instrumento, batería…), lo va acercando al estilo de dos en dos cosas por tramo, así el
    cambio de estilo se nota poco a poco. bpm fijo si se da; si no, deriva dentro del estilo."""
    E = estilo_de(estilo)
    rng = np.random.default_rng(seed)
    nuevo = tema_aleatorio(int(rng.integers(1 << 30)), estilo)
    if prev is None:
        T = nuevo
    else:
        T = dict(prev)
        T.pop("intro_filtro", None)
        fuera = [g for g in _ACOTADOS if g in E and prev.get(g) not in E[g]]
        cambios = set(fuera[:2])
        # variedad: 2 o 3 grupos nuevos por tramo, y acordes/tonalidad nuevos como mínimo cada 3 tramos
        n = 2 + int(rng.random() < .4)
        sin_armonia = prev.get("sin_armonia", 0) + 1
        if sin_armonia >= 3 or rng.random() < .3:
            cambios.add("armonia")
        libres = [g for g in _GRUPOS if g not in cambios]
        cambios.update(str(g) for g in rng.choice(libres, size=min(n, len(libres)), replace=False))
        for g in cambios:
            for k in _GRUPOS[g]:
                if k in nuevo:
                    T[k] = nuevo[k]
                else:
                    T.pop(k, None)
            if g in _ACOTADOS:   # que el cambio se note: nunca repetir el valor que ya sonaba
                opciones = [x for x in E.get(g, TODOS[g]) if x != prev.get(g)]
                if opciones:
                    T[g] = str(rng.choice(opciones))
        T["sin_armonia"] = 0 if "armonia" in cambios else sin_armonia
        if "armonia" not in cambios and rng.random() < .7:   # otra melodía sobre los mismos acordes
            dens = str(rng.choice(E.get("densidad", TODOS["densidad"])))
            T["mel"] = _mel_aleatoria(rng, T["raiz"], T["acordes"], T["modo"] == "menor", dens, ESCALAS.get(T.get("escala")))
        if T["inst"] == "pad" and T["comp"] in ("stabs", "charleston", "roto"):
            T["comp"] = "pad"
        if "bat" in cambios or "armonia" in cambios:
            T["sw"] = nuevo["sw"]
        T["bpm"] = prev["bpm"]
    T.pop("guino", None)   # sin guiños de 8 bits en la radio infinita (al usuario le sonaban a videojuego)
    # más armónico: la melodía no sube de dos octavas y una cuarta sobre la raíz, la parte B no
    # salta una octava, y fuera del retro no hay timbres percusivos en la melodía
    techo = T["raiz"] + 29
    T["mel"] = [[(m - 12 if m and m > techo else m) for m in fr] for fr in T["mel"]]
    T["subeB"] = False
    suave = {"caja": "vibra", "chip": "pulso"}
    if T["instMel"] in suave and "chip" not in E.get("instMel", []):
        T["instMel"] = suave[T["instMel"]]
    if bpm:
        T["bpm"] = int(bpm)
    elif prev is not None:   # deriva: hacia el rango del estilo, como mucho 4 bpm por tramo
        lo, hi = (int(x) for x in E.get("bpm", (60, 92)))
        objetivo = min(max(T["bpm"], lo), hi)
        if objetivo != T["bpm"]:
            T["bpm"] += max(-4, min(4, objetivo - T["bpm"]))
        elif rng.random() < .5:
            T["bpm"] = min(max(T["bpm"] + int(rng.integers(-5, 6)), lo), hi)
    T["cambios"] = sorted(cambios) if prev is not None else []
    return T


def generate_random(seed, outdir, estilo="aleatorio", fondo=1.0):
    """Compone y renderiza un tema nuevo del estilo dado; devuelve (ruta WAV, tema).
    fondo 0 = sin textura de vinilo/lluvia/cinta."""
    from pathlib import Path
    tema = tema_aleatorio(seed, estilo)
    audio = render_theme(tema, cycles=2, seed=seed, fondo=fondo)
    out = Path(outdir) / f"nuevo_{seed:06d}.wav"
    wavfile.write(str(out), SR, (audio * 32767).astype(np.int16))
    return out, tema
