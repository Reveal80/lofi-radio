"""Compositor de piezas para la radio infinita (2.8).

Antes la radio encadenaba tramos de 16 compases que heredaban casi todo del anterior y cambiaban
dos o tres parámetros al azar: el oyente oía un bucle de 4 compases sin principio ni fin, con
instrumentos que iban rotando. Esto compone **piezas** de 3 a 6 minutos con identidad propia
(perfil, tonalidad, tempo, instrumentación, armonía, groove), estructura por secciones (intro, A, B,
descanso, puente, cierre) en las que las capas entran y salen con intención, melodías por motivos
con pregunta y respuesta que se desarrollan a lo largo de la pieza, armonía con conducción de voces,
bajo que enlaza los acordes, ritmos con variaciones, rellenos y humanización moderada, y una memoria
de piezas recientes que penaliza repetir combinaciones parecidas.

Usa los sintetizadores, instrumentos y estilos de `lofigen.py` (tocar_nota, bajo_nota, bombo, golpe,
reverb, INST, ESTILOS, tema_aleatorio para los rasgos del estilo: kit, filtro, graves, niveles…).

Modo de prueba:  python3 piezas.py --prueba CARPETA [--variedad bajo|medio|alto]
                 renderiza tres piezas contrastadas (perfiles distintos) en WAV.
"""
import itertools
import json
import sys
import time

import numpy as np

import lofigen as L
from lofigen import SR

VARIEDADES = ("bajo", "medio", "alto")
NOTAS = ["Do", "Do♯", "Re", "Re♯", "Mi", "Fa", "Fa♯", "Sol", "Sol♯", "La", "La♯", "Si"]

# ── perfiles: la identidad de cada pieza ──────────────────────────────────────────────────────
# Cada pieza elige un perfil (entre los que admite el estilo marcado) y de él salen las preferencias
# de instrumentos, groove, familia armónica, acompañamiento, bajo, melodía y filtro. El estilo sigue
# mandando: solo se usan instrumentos y ritmos que el estilo (y el usuario) admitan.
PERFILES = {
    "jazz_calido": dict(
        nombre="Jazz lofi cálido", inst=["rhodes", "piano", "guitarra"], instMel=["saxo", "trompeta", "vibra", "piano", "clarinete"],
        capa=["vibra", "rhodes", "pad"], grooves=["shuffle", "dilla", "boombap", "doble"], armonia=["jazz", "jazz", "pop"],
        comp=["charleston", "stabs", "roto", "largo"], bajo=["caminante", "caminante", "rebote"], bajoTimbre=["cuerda", "triangulo"],
        bpm=(74, 90), sw=(.16, .26), menor=.45, novena=.7, lofi=(3800, 6500), sala=(.08, .22),
        registro=[19, 19, 24], densidad=["normal", "normal", "densa"], dur=(3.0, 5.5)),
    "piano_intimo": dict(
        nombre="Piano íntimo", inst=["piano", "piano", "rhodes"], instMel=["piano", "vibra", "flauta"],
        capa=["pad", "vibra", "campana"], grooves=["suave", "balada", "nada", "medio"], armonia=["pop", "modal", "ambient"],
        comp=["arpegio", "roto", "largo", "balanceo"], bajo=["largo", "pedal", "octavas"], bajoTimbre=["sub", "triangulo"],
        bpm=(62, 76), sw=(.03, .1), menor=.55, novena=.45, lofi=(4200, 7500), sala=(.18, .4),
        registro=[19, 24, 24], densidad=["escasa", "normal"], dur=(3.5, 6.0)),
    "chillhop": dict(
        nombre="Chillhop suave", inst=["rhodes", "guitarra", "piano", "pulso"], instMel=["vibra", "rhodes", "flauta", "guitarra", "pulso"],
        capa=["pad", "vibra", "guitarra"], grooves=["boombap", "dilla", "roto", "perezoso", "doble"], armonia=["pop", "jazz", "modal"],
        comp=["largo", "stabs", "ostinato", "charleston"], bajo=["rebote", "sincopado", "largo", "octavas"], bajoTimbre=["sub", "sinte", "triangulo"],
        bpm=(76, 92), sw=(.1, .2), menor=.35, novena=.55, lofi=(3200, 6000), sala=(.05, .16),
        registro=[14, 19, 19], densidad=["normal", "normal", "escasa"], dur=(3.0, 5.0)),
    "ambient_nocturno": dict(
        nombre="Ambient nocturno", inst=["pad", "rhodes", "koto", "cuenco"], instMel=["flauta", "campana", "vibra", "shakuhachi", "cuenco"],
        capa=["pad", "cuenco", "campana"], grooves=["nada", "suave", "templo", "balada"], armonia=["ambient", "modal", "pop"],
        comp=["pad", "arpegio", "largo"], bajo=["pedal", "largo"], bajoTimbre=["sub", "sub", "triangulo"],
        bpm=(54, 66), sw=(.0, .06), menor=.7, novena=.5, lofi=(2600, 5200), sala=(.3, .55),
        registro=[14, 19, 19], densidad=["escasa", "escasa", "normal"], dur=(4.0, 6.0)),
    "guitarra": dict(
        nombre="Guitarra relajada", inst=["guitarra", "guitarra", "koto"], instMel=["guitarra", "flauta", "clarinete", "vibra"],
        capa=["rhodes", "pad", "flauta"], grooves=["suave", "bossa", "medio", "balada", "perezoso"], armonia=["pop", "modal", "jazz"],
        comp=["arpegio", "bossa", "balanceo", "roto"], bajo=["largo", "rebote", "octavas"], bajoTimbre=["cuerda", "triangulo", "sub"],
        bpm=(66, 84), sw=(.06, .14), menor=.3, novena=.4, lofi=(3800, 7500), sala=(.1, .3),
        registro=[14, 19, 19], densidad=["normal", "escasa"], dur=(3.0, 5.5)),
}
# qué perfiles admite cada estilo (los no listados usan todos y dejan que el estilo filtre)
PERFILES_ESTILO = {
    "estudiar": ["chillhop", "piano_intimo", "guitarra"], "chill": ["chillhop", "guitarra", "ambient_nocturno"],
    "jazz": ["jazz_calido"], "lluvioso": ["piano_intimo", "ambient_nocturno", "guitarra"], "retro": ["chillhop"],
    "ambiente": ["ambient_nocturno"], "templo": ["ambient_nocturno"], "nocturno": ["ambient_nocturno", "piano_intimo"],
    "bossa": ["guitarra"], "cafe": ["guitarra", "piano_intimo", "chillhop"], "invierno": ["ambient_nocturno", "piano_intimo"],
    "neon": ["chillhop"], "bosque": ["guitarra", "ambient_nocturno"], "blues": ["jazz_calido"], "anime": ["chillhop", "piano_intimo"],
}

# ── familias armónicas (índices en las tablas de lofigen) ────────────────────────────────────
FAMILIAS_MAYOR = {"pop": [4, 5, 6, 7, 9, 10], "jazz": [0, 1, 2, 12, 13], "modal": [3, 8, 11, 12], "ambient": [3, 11]}
FAMILIAS_MENOR = {"pop": [0, 1, 2, 4, 7], "jazz": [3, 8, 11], "modal": [6, 9, 10], "ambient": [9, 10]}
PUENTES_MAYOR = [[(5, "maj7"), (7, "7")], [(10, "7"), (5, "maj7")], [(2, "m7"), (4, "m7")], [(5, "maj7"), (5, "m6")]]
PUENTES_MENOR = [[(8, "maj7"), (10, "7")], [(5, "m7"), (7, "7")], [(3, "maj7"), (10, "7")]]

# ── grooves: patrón base de lofigen.RITMOS + variantes de bombo para los compases pares ────────
GROOVES = {
    "boombap":  dict(varK=["x..x..x.", "x..x...x", "x.....x."], hats=1.0),
    "dilla":    dict(varK=["x.x..x..", "x..x.x.x"], hats=.9, tarde=.014),    # la caja va un pelo tarde
    "suave":    dict(varK=["x...x..x", "x......."], hats=.7),
    "medio":    dict(varK=["x.....x.", "x......x"], hats=.6),
    "shuffle":  dict(varK=["x....x.x", "x.x..x.."], hats=1.0),
    "roto":     dict(varK=["x..x..x.", "x.x...x."], hats=.9),
    "bossa":    dict(varK=["x..x..x.", "x..x.x.."], hats=.8),
    "balada":   dict(varK=["x.....x.", "x...x.x."], hats=.7),
    "perezoso": dict(varK=["x.....x.", "x......x"], hats=.8),
    "doble":    dict(varK=["x.x...x.", "x.x..x.."], hats=1.0),
    "templo":   dict(varK=[], hats=0),
    "nada":     dict(varK=[], hats=0),
}

# ── melodía: células rítmicas (corchea de inicio, duración) y contornos (pasos de escala) ────
CELULAS = {
    "escasa": [[(0, 3), (4, 3)], [(0, 2), (3, 3)], [(2, 2), (4, 4)], [(0, 4), (6, 2)], [(0, 6)], [(1, 2), (4, 3)], [(0, 3), (3, 1), (4, 4)]],
    "normal": [[(0, 2), (2, 1), (3, 3)], [(0, 1), (1, 1), (2, 2), (4, 3)], [(0, 3), (3, 1), (4, 2)], [(1, 1), (2, 2), (4, 1), (5, 3)],
               [(0, 2), (3, 2), (6, 2)], [(0, 1), (2, 1), (4, 1), (6, 2)], [(2, 2), (4, 2), (6, 2)], [(0, 1), (1, 1), (3, 1), (4, 3)]],
    "densa":  [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 2), (6, 2)], [(0, 1), (1, 2), (3, 1), (4, 1), (5, 1), (6, 2)],
               [(0, 2), (2, 1), (3, 1), (4, 1), (5, 1), (6, 2)], [(1, 1), (2, 1), (3, 1), (4, 2), (6, 1), (7, 1)]],
}
CONTORNOS = {"arco": (0, 1, 2, 3, 2, 1, 0), "descenso": (0, -1, -2, -3, -4, -5), "ascenso": (0, 1, 2, 3, 4, 5),
             "onda": (0, 2, 1, 3, 2, 4, 3), "vuelta": (0, 3, 2, 1, 0, -1, -2), "repetido": (0, 0, 1, 0, -1, -2, -1),
             "pregunta": (0, 1, 3, 2, 4, 3), "salto": (0, 4, 3, 2, 1, 0, -1), "bajada": (0, -2, -1, -3, -2, -4)}

# ── máster de referencia (medido en las canciones lofi de Pixabay que trajo el usuario): realce de graves,
# nivel del bajo, bombo más largo y con más cuerpo, acordes algo más bajos ──
REF = dict(graves=(1.5, 2.1), bajoNivel=(1.0, 1.4), kv=(1.3, 1.6), kd=(.4, .55), nivAcordes=(.8, 1.05))

# ── variedad: cuánto se separan las piezas entre sí y cuánto se desarrolla cada una ──────────
VARIEDAD = {
    "bajo":  dict(candidatos=4, bpm_salto=6, tonos=(0, 0, 5, 7, 9, 3), formas=[["intro", "A", "A", "B", "A", "cierre"], ["intro", "A", "B", "A", "B", "cierre"], ["intro", "A", "A", "B", "A", "A", "cierre"]],
                  largosA=[12, 16, 16], largosB=[8], variaciones=["final", "ritmo", "eco"], prob_var=.5, relativo=0, puente=0, cambia_inst=0, min_fill=.5),
    "medio": dict(candidatos=8, bpm_salto=12, tonos=(0, 2, 3, 5, 7, 9, 10), formas=[["intro", "A", "A", "B", "descanso", "A", "cierre"], ["intro", "A", "B", "A", "descanso", "B", "A", "cierre"],
                  ["intro", "A", "A", "B", "A", "puente", "A", "cierre"], ["intro", "A", "B", "A", "A", "cierre"]],
                  largosA=[8, 12, 12, 16], largosB=[8, 8, 4], variaciones=["final", "ritmo", "eco", "desplazar", "secuencia", "silencio"], prob_var=.7, relativo=.15, puente=.3, cambia_inst=.3, min_fill=.7),
    "alto":  dict(candidatos=12, bpm_salto=40, tonos=tuple(range(12)), formas=[["intro", "A", "B", "A", "descanso", "B", "puente", "A", "cierre"], ["intro", "A", "A", "puente", "B", "descanso", "A", "B", "cierre"],
                  ["intro", "A", "B", "descanso", "A", "B", "A", "cierre"], ["intro", "A", "A", "B", "puente", "A", "descanso", "A", "cierre"]],
                  largosA=[8, 8, 12, 16], largosB=[4, 8, 8], variaciones=["final", "ritmo", "eco", "desplazar", "secuencia", "silencio", "inversion"], prob_var=.85, relativo=.35, puente=.6, cambia_inst=.6, min_fill=.9),
}


def _nombre_tono(tonica, menor):
    return f"{NOTAS[tonica % 12]} {'menor' if menor else 'mayor'}"


# ══════════════════════════════════════════════════════════════════════════════════════════════
# armonía: progresiones por familia, voicings conducidos y bajo con notas de paso
# ══════════════════════════════════════════════════════════════════════════════════════════════

def _progresion(rng, familia, menor, E, novena):
    """Vuelta de acordes [(semitonos desde la tónica, calidad)] de la familia pedida; si el estilo
    trae las suyas (templo, blues), esas. Las novenas se añaden acorde a acorde, no a todos."""
    if E.get("progs_propios"):
        k = int(rng.integers(len(E["progs_propios"])))
        return [tuple(x) for x in E["progs_propios"][k]], f"propia{k}"
    if menor:
        idx = FAMILIAS_MENOR.get(familia, FAMILIAS_MENOR["pop"])
        tabla = L._PROGS_MENOR
    else:
        idx = FAMILIAS_MAYOR.get(familia, FAMILIAS_MAYOR["pop"])
        if E.get("progs"):   # el estilo acota (jazz hop)
            idx = [i for i in idx if i in E["progs"]] or list(E["progs"])
        tabla = L._PROGS
    i = int(rng.choice(idx))
    prog = [tuple(x) for x in tabla[i]]
    out = []
    for k, (g, q) in enumerate(prog):
        if q in L._NOVENA and rng.random() < novena:
            q = L._NOVENA[q]
        elif q == "maj7" and k == len(prog) - 1 and rng.random() < .2:
            q = "6"
        out.append((g, q))
    return out, f"{'m' if menor else 'M'}{familia}{i}"


def _pcs(tonica, grado, q):
    root = (tonica + grado) % 12
    ints = L._QUAL.get(q, L._QUAL["maj7"])
    if len(ints) > 4 and 7 in ints:   # con novena, la quinta sobra en el voicing
        ints = tuple(x for x in ints if x != 7)
    return root, [(root + x) % 12 for x in ints]


def _voicing(pcs, prev, centro):
    """Notas MIDI del acorde con esas clases de altura, lo más cerca posible del acorde anterior
    (conducción de voces) y alrededor de `centro`; sin segundas menores pegadas."""
    opciones = [[m for m in range(centro - 12, centro + 13) if m % 12 == pc] for pc in pcs]
    mejor = None
    for combo in itertools.product(*opciones):
        notas = sorted(set(combo))
        if len(notas) < len(pcs):
            continue
        coste = (notas[-1] - notas[0]) * .3 + (8 if notas[-1] - notas[0] > 19 else 0)
        if prev:
            coste += sum(min(abs(n - p) for p in prev) for n in notas) + sum(min(abs(n - p) for n in notas) for p in prev)
        else:
            coste += abs(float(np.mean(notas)) - centro) * 2
        for a, b in zip(notas, notas[1:]):
            if b - a == 1:
                coste += 14
            elif b - a == 2 and a < centro - 4:
                coste += 3
        if mejor is None or coste < mejor[0]:
            mejor = (coste, notas)
    return mejor[1]


def _conducir(tonica, prog, centro):
    """Voicings de toda la vuelta, pasando dos veces para que el enlace del último al primero
    también sea suave. Devuelve [(raíz midi para el bajo, [notas midi], pcs)]."""
    prev = None
    out = []
    for vuelta in range(2):
        out = []
        for g, q in prog:
            root, pcs = _pcs(tonica, g, q)
            notas = _voicing(pcs, prev, centro)
            r = tonica + g
            while r > 62:
                r -= 12
            while r < 46:
                r += 12
            out.append((r, notas, pcs))
            prev = notas
    return out


# ══════════════════════════════════════════════════════════════════════════════════════════════
# melodía: motivos de dos compases, pregunta/respuesta y variaciones
# ══════════════════════════════════════════════════════════════════════════════════════════════

def _motivo(rng, densidad):
    """Motivo de 2 compases: célula del compás 1 + célula del 2 recortada (la frase acaba antes
    del final y deja aire), con un contorno y un desplazamiento inicial."""
    celulas = CELULAS.get(densidad, CELULAS["normal"])
    c1 = list(celulas[int(rng.integers(len(celulas)))])
    c2 = [(o, d) for o, d in celulas[int(rng.integers(len(celulas)))] if o <= 3]
    if not c2:
        c2 = [(0, 3)]
    o_fin, _ = c2[-1]
    c2[-1] = (o_fin, int(rng.choice([3, 4, 5, 8 - o_fin])))   # la última, larga
    notas = [(o, d) for o, d in c1] + [(o + 8, d) for o, d in c2]
    nombre = str(rng.choice(list(CONTORNOS)))
    cont = CONTORNOS[nombre]
    pasos = []
    for k in range(len(notas)):
        if k < len(cont):
            pasos.append(cont[k])
        else:   # más notas que contorno: sigue bajando hacia el reposo
            pasos.append(pasos[-1] - int(rng.choice([0, 1, 1, 2])))
    return dict(notas=notas, pasos=pasos, ini=int(rng.integers(-2, 3)), contorno=nombre)


def _variar(rng, m, tipo):
    v = dict(notas=list(m["notas"]), pasos=list(m["pasos"]), ini=m["ini"], contorno=m["contorno"])
    n = len(v["notas"])
    if tipo == "final" and n >= 2:            # otro final: las dos últimas notas cambian de dirección
        d = int(rng.choice([-2, -1, 1, 2]))
        v["pasos"][-1] = v["pasos"][-2] + d
    elif tipo == "ritmo" and n >= 2:          # dos notas se funden en una larga, o una se parte en dos
        j = int(rng.integers(n - 1))
        if rng.random() < .5:
            o, d = v["notas"][j]
            o2, d2 = v["notas"][j + 1]
            v["notas"][j] = (o, o2 + d2 - o); del v["notas"][j + 1]; del v["pasos"][j + 1]
        else:
            o, d = v["notas"][j]
            if d >= 2:
                v["notas"][j] = (o, 1); v["notas"].insert(j + 1, (o + 1, d - 1)); v["pasos"].insert(j + 1, v["pasos"][j] + int(rng.choice([-1, 1])))
    elif tipo == "eco" and n >= 2:            # las dos últimas notas se repiten como eco al final
        o, d = v["notas"][-1]
        if o + d + 2 <= 16:
            v["notas"][-1] = (o, 1)
            v["notas"] += [(o + 1, 1), (o + 2, min(3, 16 - o - 2))]
            v["pasos"] += [v["pasos"][-2], v["pasos"][-1]]
    elif tipo == "desplazar":                 # todo una corchea más tarde (anticipación → retraso)
        if all(o + d <= 15 for o, d in v["notas"]):
            v["notas"] = [(o + 1, d) for o, d in v["notas"]]
    elif tipo == "secuencia":                 # el mismo dibujo un grado más arriba o abajo
        v["ini"] += int(rng.choice([-2, -1, 1, 2]))
    elif tipo == "silencio" and n >= 3:       # empieza más tarde: la primera nota se calla
        del v["notas"][0]; del v["pasos"][0]
    elif tipo == "inversion":                 # el contorno al revés
        v["pasos"] = [-p for p in v["pasos"]]
    return v


def _realizar(frase, esc, centro_idx, acordes2, tonica, final, techo):
    """Pasa un motivo (pasos de escala) a notas MIDI sobre los acordes de sus dos compases:
    los tiempos fuertes caen en notas del acorde; la última nota resuelve según `final`
    («pregunta»: nota del acorde que no sea la raíz; «respuesta»: raíz o tercera)."""
    out = []
    idx_prev = None
    n = len(frase["notas"])
    for k, ((o, d), paso) in enumerate(zip(frase["notas"], frase["pasos"])):
        i = int(np.clip(centro_idx + frase["ini"] + paso, 0, len(esc) - 1))
        root, pcs = acordes2[min(o // 8, len(acordes2) - 1)]
        ultima = k == n - 1
        if ultima and final == "respuesta":
            permitidas = {root % 12, pcs[1] if len(pcs) > 1 else root % 12}
        elif ultima and final == "pregunta":
            permitidas = set(pcs[1:]) or set(pcs)
        elif o % 8 in (0, 4):
            permitidas = set(pcs)
        else:
            permitidas = None
        if permitidas:
            cand = [j for j in range(max(0, i - 4), min(len(esc), i + 5)) if esc[j] % 12 in permitidas]
            if cand:
                i = min(cand, key=lambda j: abs(j - i))
        m = esc[i]
        if idx_prev is not None and abs(m - esc[idx_prev]) > 9:   # sin saltos de más de una sexta
            m += -12 if m > esc[idx_prev] else 12
        if m > techo:
            m -= 12
        idx_prev = i
        out.append((o, d, m))
    return out


# ══════════════════════════════════════════════════════════════════════════════════════════════
# estructura: plan de secciones con capas e intención
# ══════════════════════════════════════════════════════════════════════════════════════════════

def _planificar(rng, V, compases_objetivo, perfil, menor):
    forma = list(V["formas"][int(rng.integers(len(V["formas"])))])
    secciones = []
    nA = 0
    for i, tipo in enumerate(forma):
        if tipo == "intro":
            n = int(rng.choice([2, 4, 4]))
        elif tipo == "A":
            n = int(rng.choice(V["largosA"])); nA += 1
        elif tipo == "B":
            n = int(rng.choice(V["largosB"]))
        elif tipo == "descanso":
            n = int(rng.choice([2, 4]))
        elif tipo == "puente":
            n = 4
        else:   # cierre
            n = int(rng.choice([4, 6]))
        secciones.append(dict(tipo=tipo, compases=n))
    # ajustar al objetivo de duración: alargar o acortar las A
    total = lambda: sum(s["compases"] for s in secciones)
    k = 0
    while total() < compases_objetivo * .9 and k < 20:
        As = [s for s in secciones if s["tipo"] == "A"]
        cortas = [s for s in As if s["compases"] < 16]
        if cortas:
            cortas[k % len(cortas)]["compases"] += 4
        elif len(secciones) < 14:   # todas las A ya largas: otra vuelta A (o B) antes del cierre
            tipo = "B" if sum(1 for s in secciones if s["tipo"] == "B") < 2 and rng.random() < .5 else "A"
            secciones.insert(len(secciones) - 1, dict(tipo=tipo, compases=8 if tipo == "B" else int(rng.choice(V["largosA"]))))
        else:
            break
        k += 1
    As = [s for s in secciones if s["tipo"] == "A"]
    k = 0
    while total() > compases_objetivo * 1.15 and k < 12:
        s = As[k % len(As)]
        if s["compases"] > 8:
            s["compases"] -= 4
        k += 1
    # capas con intención
    primeraA = True
    nombres = {"A": 0, "B": 0}
    for i, s in enumerate(secciones):
        sig = secciones[i + 1]["tipo"] if i + 1 < len(secciones) else None
        t = s["tipo"]
        s["prog"] = "B" if t == "B" else "puente" if t == "puente" else "A"
        s["bat"], s["bajo"], s["mel"], s["capa"], s["pad"], s["hats"] = 1.0, True, True, False, False, 1.0
        s["bat_desde"] = 0            # compás en el que entra la batería
        s["bajo_desde"] = 0
        s["mel_desde"] = 0
        s["fill"] = None
        if t == "intro":
            s.update(bat=0, bajo=s["compases"] >= 4, bajo_desde=2, mel=False, capa=rng.random() < .4, pad=True, hats=0)
        elif t == "A":
            nombres["A"] += 1
            s["nombre"] = f"A{nombres['A']}" if nombres["A"] > 1 else "A"
            if primeraA:
                s.update(bat_desde=int(rng.choice([0, 2, 4])), mel_desde=int(rng.choice([0, 0, 4])), hats=.8)
                primeraA = False
            else:
                s.update(capa=rng.random() < .7, hats=1.0)
                if nombres["A"] >= 3 and rng.random() < .5:
                    s["mel_desde"] = 2
        elif t == "B":
            nombres["B"] += 1
            s["nombre"] = f"B{nombres['B']}" if nombres["B"] > 1 else "B"
            s.update(capa=rng.random() < .6, hats=.6, pad=rng.random() < .4)
        elif t == "descanso":
            s.update(bat=0, bajo=rng.random() < .3, mel=False, capa=True, pad=True, hats=0)
        elif t == "puente":
            s.update(mel=rng.random() < .3, hats=.5, capa=True)
        elif t == "cierre":
            s.update(bat=.6, mel=True, capa=rng.random() < .5, pad=True, hats=.5)
        s.setdefault("nombre", t)
        # relleno al final de la sección, según lo que viene
        if t in ("A", "B", "puente") and sig:
            if sig in ("descanso", "cierre"):
                s["fill"] = "corte"
            elif sig == "B":
                s["fill"] = str(rng.choice(["redoble", "pickup", "redoble"]))
            else:
                s["fill"] = str(rng.choice(["pickup", "charles", "redoble"]))
            if rng.random() > V["min_fill"]:
                s["fill"] = None
        if t == "descanso" and sig in ("A", "B"):
            s["fill"] = "pickup"
    return secciones


def _plan_melodia(rng, secciones, V, nivel):
    """Para cada sección, qué frase (motivo, pregunta/respuesta, variaciones) suena en cada par de
    compases. nivel 0-3: cuánta melodía (0 ninguna, 1 de vez en cuando, 2 normal, 3 continua)."""
    for s in secciones:
        frases = []
        pares = s["compases"] // 2
        if not s["mel"] or nivel == 0:
            s["frases"] = [None] * pares
            continue
        motivo = "N" if s["tipo"] == "B" else "M"
        vuelta = 0
        for p in range(pares):
            if p * 2 < s["mel_desde"]:
                frases.append(None); continue
            if s["tipo"] == "cierre":
                frases.append(("M", "respuesta", ["final"]) if p == 0 else None); continue
            if s["tipo"] == "puente":
                frases.append(("M", "pregunta", ["eco"]) if p == 0 else None); continue
            tipo = "pregunta" if p % 2 == 0 else "respuesta"
            vuelta = p // 2
            vars_ = []
            if vuelta > 0 and rng.random() < V["prob_var"]:
                vars_ = [str(x) for x in rng.choice(V["variaciones"], size=min(vuelta, 2), replace=False)]
            frases.append((motivo, tipo, vars_))
        # cuánta melodía: se quitan pares completos (pregunta+respuesta) para que lo que queda tenga sentido
        if nivel < 3:
            pares_qr = [(i, i + 1) for i in range(0, len(frases) - 1, 2) if frases[i] and frases[i + 1]]
            quitar = {1: .55, 2: .25}[nivel]
            for j, (a, b) in enumerate(pares_qr):
                if j > 0 and rng.random() < quitar:
                    frases[a] = frases[b] = None
        s["frases"] = frases
    return secciones


# ══════════════════════════════════════════════════════════════════════════════════════════════
# la pieza: identidad + plan
# ══════════════════════════════════════════════════════════════════════════════════════════════

def _pref(rng, preferidos, admitidos, evitar=()):
    """Uno de los preferidos del perfil que el estilo admite (los primeros, más probables); si el
    estilo no admite ninguno, uno del estilo. `evitar`: lo ya elegido (que la melodía no repita
    el timbre de los acordes si hay alternativa)."""
    comunes = [x for x in preferidos if x in admitidos]
    base = comunes or list(admitidos)
    sin = [x for x in base if x not in evitar]
    base = sin or base
    if len(base) > 2 and rng.random() < .6:
        base = base[:2]
    return str(rng.choice(base))


def componer(seed, estilo="aleatorio", E=None, variedad="medio", bpm=0, melodia=1, prev=None, continuar=False, perfil=None, capas=3):
    """Compone una pieza entera (sin renderizar): identidad, armonía con voicings, motivos y plan
    de secciones. prev: pieza anterior (para la continuidad de tempo y tonalidad); continuar=True
    (cambio pedido por el usuario) conserva tempo y tonalidad y acorta la intro."""
    rng = np.random.default_rng(seed)
    E = E or L.estilo_de(estilo)
    V = VARIEDAD.get(variedad, VARIEDAD["medio"])
    nombres = [p for p in PERFILES_ESTILO.get(estilo, list(PERFILES)) if p in PERFILES] if "+" not in estilo else list(PERFILES)
    if perfil in PERFILES:
        nombres = [perfil]
    elif prev and len(nombres) > 1 and variedad != "bajo":
        nombres = [p for p in nombres if p != prev.get("perfil")] or nombres
    perfil = str(rng.choice(nombres))
    P = PERFILES[perfil]
    E = dict(E)
    for k in ("kv", "kd", "nivAcordes"):   # el estilo manda; si no acota, el máster de referencia
        E.setdefault({"kv": "bombo"}.get(k, k), REF[k])
    T = L.tema_aleatorio(int(rng.integers(1 << 30)), estilo, E)   # rasgos del estilo: kit, filtro, graves, sala, niveles…
    solo = bool(E.get("solo_melodia"))
    menor = T["modo"] == "menor"
    if not E.get("escalas") and rng.random() < .5:   # el perfil también opina sobre mayor/menor
        menor = rng.random() < P["menor"]
    # tonalidad: cerca de la anterior según la variedad (o la misma si continúa)
    if continuar and prev:
        tonica = prev["tonica"]
    elif prev:
        salto = int(rng.choice(V["tonos"]))
        tonica = 48 + ((prev["tonica"] - 48 + salto) % 12)
    else:
        tonica = int(rng.integers(48, 60))
    if tonica > 58:
        tonica -= 12
    # tempo: el pedido, o dentro del estilo/perfil con un salto acotado respecto a la pieza anterior
    lo, hi = (int(x) for x in E.get("bpm", P["bpm"]))
    if E.get("bpm") is None:
        lo, hi = P["bpm"]
    if bpm:
        tempo = int(bpm)
    else:
        tempo = int(rng.integers(lo, hi + 1))
        if prev and continuar:                      # cambio pedido: mismo tempo (dentro del rango del estilo nuevo)
            tempo = int(min(max(prev["bpm"], lo), hi))
        elif prev and abs(tempo - prev["bpm"]) > V["bpm_salto"]:   # entre piezas: salto acotado por la variedad
            objetivo = prev["bpm"] + int(np.sign(tempo - prev["bpm"])) * V["bpm_salto"]
            tempo = int(min(max(objetivo, lo), hi))
    # instrumentación: preferencias del perfil dentro de lo que admite el estilo (y el usuario)
    inst = _pref(rng, P["inst"], E.get("inst", L.TODOS["inst"]))
    instMel = _pref(rng, P["instMel"], E.get("instMel", L.TODOS["instMel"]), evitar=(inst,))
    # capas: cuántos instrumentos a la vez (2 acordes+melodía, 3 +notas guía, 4 +arpegio/contracanto)
    admitidos = set(E.get("inst", L.TODOS["inst"])) | set(E.get("instMel", L.TODOS["instMel"]))
    capa_opts = [x for x in P["capa"] + list(admitidos) if x not in (inst, instMel) and x in admitidos and x in L.INST]
    if not capas or len(admitidos) > 4:   # sin acotar (estilo aleatorio o muchos activos): 3 o 4 según la pieza
        capas = int(rng.choice([3, 3, 4]))
    capa = str(rng.choice(capa_opts[:3])) if capa_opts and capas >= 3 else None
    inst2_opts = [x for x in P["inst"] + P["instMel"] + list(admitidos) if x not in (inst, instMel, capa, "pad") and x in admitidos and x in L.INST]
    inst2 = str(rng.choice(inst2_opts[:3])) if inst2_opts and capas >= 4 else None
    groove = _pref(rng, P["grooves"], E.get("bat", L.TODOS["bat"]))
    comp = _pref(rng, P["comp"], E.get("comp", L.TODOS["comp"]))
    if inst == "pad" and comp in ("stabs", "charleston", "roto", "ostinato", "bossa"):
        comp = "pad"
    bajo = _pref(rng, P["bajo"], E.get("bajo", L.TODOS["bajo"]))
    bajoTimbre = _pref(rng, P["bajoTimbre"], E.get("bajoTimbre", L.TODOS["bajoTimbre"]))
    sw_lo, sw_hi = E.get("sw", P["sw"])
    sw = float(rng.uniform(sw_lo, sw_hi))
    if groove in ("nada", "templo", "bossa"):
        sw = min(sw, .06)
    # sonido calibrado con las cinco referencias de Pixabay del usuario (2.9): mucho grave, casi nada por
    # encima de 2 kHz; el estilo puede acotarlo (anime) y si no, el perfil (reducido un 40 %)
    lofi = float(rng.uniform(*E["lofi"])) if E.get("lofi") else float(rng.uniform(P["lofi"][0] * .6, P["lofi"][1] * .6))
    sala = (float(rng.uniform(*E.get("sala", P["sala"]))), float(rng.uniform(1.0, 3.0)))
    novena = float(E.get("novena", P["novena"]))
    familia = str(rng.choice(P["armonia"]))
    # armonía: vuelta A, vuelta B (otra de la misma familia, quizá en el relativo) y puente
    progA, idA = _progresion(rng, familia, menor, E, novena)
    tonicaB, menorB = tonica, menor
    if not E.get("progs_propios") and rng.random() < V["relativo"]:
        tonicaB, menorB = (tonica + (3 if menor else -3)), (not menor)
        if tonicaB > 58:
            tonicaB -= 12
        if tonicaB < 46:
            tonicaB += 12
    progB, idB = _progresion(rng, familia, menorB, E, novena)
    if idB == idA and not E.get("progs_propios"):
        progB = progB[len(progB) // 2:] + progB[:len(progB) // 2]
    puente = [tuple(x) for x in (PUENTES_MENOR if menor else PUENTES_MAYOR)[int(rng.integers(len(PUENTES_MENOR if menor else PUENTES_MAYOR)))]]
    centro = tonica + 12 + (12 if T.get("acordesArriba") else 0) + (4 if inst in ("guitarra", "koto") else 0)
    acordes = {"A": _conducir(tonica, progA, centro), "B": _conducir(tonicaB, progB, centro), "puente": _conducir(tonica, puente, centro)}
    # escala y motivos
    grados = L.ESCALAS.get(T.get("escala")) or ((0, 2, 3, 5, 7, 8, 10) if menor else (0, 2, 4, 5, 7, 9, 11))
    esc = sorted({tonica + o + d for o in (12, 24, 36) for d in grados})
    registro = int(rng.choice(E.get("registro", P["registro"])))
    centro_idx = min(range(len(esc)), key=lambda i: abs(esc[i] - (tonica + registro)))
    densidad = _pref(rng, P["densidad"], E.get("densidad", L.TODOS["densidad"]))
    motivos = {"M": _motivo(rng, densidad), "N": _motivo(rng, str(rng.choice(["escasa", "normal", "densa"])))}
    motivos["N"]["ini"] += int(rng.choice([-3, -2, 2, 3]))   # la B contrasta: otro registro
    # duración y plan
    dur_min = float(rng.uniform(*P["dur"]))
    compas_s = 240 / tempo
    secciones = _planificar(rng, V, dur_min * 60 / compas_s, P, menor)
    if continuar:
        secciones[0]["compases"] = 2
    if solo:
        for s in secciones:
            s.update(mel=True, mel_desde=0, bat=0, bajo=False, capa=False, pad=False)
    _plan_melodia(rng, secciones, V, 3 if solo else int(melodia))
    # ¿cambia la instrumentación en alguna sección? (variedad media/alta): la melodía de la B con otro timbre
    instMelB = instMel
    if rng.random() < V["cambia_inst"]:
        otros = [x for x in P["instMel"] if x != instMel and x in E.get("instMel", L.TODOS["instMel"])]
        if otros:
            instMelB = str(rng.choice(otros))
    pieza = dict(
        seed=int(seed), estilo=estilo, perfil=perfil, perfil_nombre=P["nombre"], variedad=variedad,
        bpm=tempo, sw=sw, tonica=tonica, modo="menor" if menor else "mayor", tono=_nombre_tono(tonica, menor),
        inst=inst, instMel=instMel, instMelB=instMelB, capa=capa, inst2=inst2, capas=int(capas), bat=groove, comp=comp, bajo=bajo, bajoTimbre=bajoTimbre,
        kit=T["kit"], lofi=lofi, graves=T.get("graves") or float(rng.uniform(*REF["graves"])),
        bajoNivel=T.get("bajoNivel") if E.get("bajoNivel") else float(rng.uniform(*REF["bajoNivel"])), bajoGrave=True,
        sala=sala, niveles=T["niveles"], tex=T["tex"], texVol=T.get("texVol", 1), bonsho=bool(T.get("bonsho")),
        familia=familia, progA=progA, progB=progB, idA=idA, idB=idB, acordes=acordes, esc=esc, centro_idx=centro_idx,
        registro=registro, motivos=motivos, contorno=motivos["M"]["contorno"], densidad=densidad,
        secciones=secciones, solo=solo, melodia=int(melodia), dur_min=dur_min,
        compases=sum(s["compases"] for s in secciones),
        forma="·".join(s["nombre"] for s in secciones),
    )
    pieza["dur"] = pieza["compases"] * compas_s
    return pieza


def huella(p):
    """Rasgos de la pieza que cuentan para la memoria antirrepetición."""
    return dict(perfil=p["perfil"], prog=p["idA"], tonica=p["tonica"] % 12, modo=p["modo"], bpm=p["bpm"], bat=p["bat"],
                inst=p["inst"], instMel=p["instMel"], contorno=p["contorno"], comp=p["comp"], bajo=p["bajo"],
                celula=tuple(o for o, d in p["motivos"]["M"]["notas"]))


def similitud(h, recientes):
    """0 = nada que ver con las piezas recientes; cuanto más alto, más repetida sonaría.
    Las piezas más recientes pesan más."""
    pesos = dict(perfil=.22, prog=.2, tonica=.1, modo=.05, bat=.14, inst=.12, instMel=.1, contorno=.05, comp=.05, bajo=.03, celula=.06)
    total = 0.0
    for edad, r in enumerate(reversed(recientes)):
        s = sum(w for k, w in pesos.items() if r.get(k) == h.get(k) or (k == "celula" and tuple(r.get(k, ())) == h.get(k)))
        if abs(int(r.get("bpm", 0)) - h["bpm"]) <= 3:
            s += .06
        total += s / (1 + .35 * edad)
    return total


class Compositor:
    """Estado de la radio infinita en el proceso de render: pieza en curso, sección que toca y
    memoria de piezas recientes (se guarda en memoria.json para sobrevivir a reinicios)."""

    def __init__(self, memoria_path=None):
        self.memoria_path = memoria_path
        self.recientes = []
        self.pieza = None
        self.idx = 0
        self.n = 0
        self.segundos = 0.0   # emitidos de la pieza en curso antes de la sección que toca
        self.ultimo = None    # (idx, segundos) antes de la última sección, por si hay que rehacerla
        if memoria_path:
            try:
                self.recientes = json.loads(open(memoria_path).read())[-12:]
            except Exception:
                self.recientes = []

    def _guardar(self):
        if self.memoria_path:
            try:
                open(self.memoria_path, "w").write(json.dumps(self.recientes[-12:]))
            except Exception:
                pass

    def nueva(self, estilo, E, variedad, bpm, melodia, seed, continuar=False, capas=3):
        """Elige entre varias piezas candidatas la que menos se parezca a las recientes."""
        V = VARIEDAD.get(variedad, VARIEDAD["medio"])
        rng = np.random.default_rng(seed)
        mejor = None
        for k in range(V["candidatos"]):
            p = componer(int(rng.integers(1 << 30)), estilo, E, variedad, bpm, melodia, self.pieza, continuar, None, capas)
            s = similitud(huella(p), self.recientes)
            if mejor is None or s < mejor[0]:
                mejor = (s, p)
        p = mejor[1]
        p["parecido"] = round(mejor[0], 2)
        self.recientes.append({k: (list(v) if isinstance(v, tuple) else v) for k, v in huella(p).items()})
        self._guardar()
        self.n += 1
        p["n"] = self.n
        p["clave"] = (estilo, tuple(E.get("inst", ())), tuple(E.get("instMel", ())), bool(E.get("solo_melodia")), capas)
        self.pieza, self.idx, self.segundos = p, 0, 0.0
        return p

    def tramo(self, estilo, bpm, fondo, opc, seed):
        """Siguiente sección a emitir: (info para el panel, cuerpo, cola)."""
        E = L.con_instrumentos(L.estilo_de(estilo), tuple(opc.get("quitar", ())), tuple(opc.get("anadir", ())), False)
        variedad = str(opc.get("variedad", "medio"))
        melodia = int(opc.get("melodia", 1))
        # instrumentos a la vez: los que el usuario tenga activos (2 a 4); con más de 4 activos, 3 o 4 según la pieza
        activos = set(E.get("inst", ())) | set(E.get("instMel", ()))
        capas = min(4, max(2, len(activos))) if E.get("inst") else 0
        clave = (estilo, tuple(E.get("inst", ())), tuple(E.get("instMel", ())), bool(E.get("solo_melodia")), capas)
        p = self.pieza
        if opc.get("rehacer") and self.ultimo and p is not None:   # la misma sección otra vez con ajustes nuevos (melodía…)
            self.idx, self.segundos = self.ultimo
        nueva = p is None or self.idx >= len(p["secciones"]) or opc.get("salto")
        if p is not None and not nueva and opc.get("minimo"):
            if clave != p["clave"] or (bpm and bpm != p["bpm"]):
                nueva, continuar = True, True          # cambio de estilo/instrumentos/tempo: pieza nueva que continúa
            else:                                      # solo cambió la melodía o la variedad: sigue la pieza
                if melodia != p["melodia"]:
                    V = VARIEDAD.get(variedad, VARIEDAD["medio"])
                    _plan_melodia(np.random.default_rng(seed), p["secciones"][self.idx:], V, 3 if p["solo"] else melodia)
                    p["melodia"] = melodia
        else:
            continuar = False
        if p is not None and not nueva and estilo != p["estilo"]:   # rotación de estilos: la pieza va al cierre
            restantes = p["secciones"][self.idx:]
            if len(restantes) > 1:
                cierre = [s for s in p["secciones"] if s["tipo"] == "cierre"]
                p["secciones"] = p["secciones"][:self.idx] + cierre
        if nueva:
            p = self.nueva(estilo, E, variedad, bpm, melodia, seed, continuar, capas)
        S = p["secciones"][self.idx]
        self.ultimo = (self.idx, self.segundos)
        cuerpo, cola = render_seccion(p, S, seed, fondo)
        antes = self.segundos
        self.segundos += len(cuerpo) / SR
        self.idx += 1
        info = dict(bpm=p["bpm"], inst=p["inst"], instMel=p["instMelB"] if S["tipo"] == "B" else p["instMel"], bat=p["bat"], modo=p["modo"],
                    tex=p["tex"], solo_melodia=p["solo"], cambios=[],
                    pieza=dict(n=p["n"], perfil=p["perfil"], perfil_nombre=p["perfil_nombre"], tono=p["tono"], variedad=p["variedad"],
                               dur=round(p["dur"], 1), antes=round(antes, 1), seccion=S["nombre"], idx=self.idx, total=len(p["secciones"]),
                               forma=p["forma"], capa=p["capa"], inst2=p.get("inst2"), comp=p["comp"], bajo=p["bajo"], familia=p["familia"],
                               parecido=p.get("parecido", 0)))
        return info, cuerpo, cola


# ══════════════════════════════════════════════════════════════════════════════════════════════
# render de una sección (continuo: devuelve cuerpo exacto + cola de resonancia de 4 s)
# ══════════════════════════════════════════════════════════════════════════════════════════════

def _ganancia(p, seed):
    """Ganancia fija de la pieza: la que deja una sección A completa a -16 dBFS RMS. Así la intro,
    el descanso y el cierre suenan más bajos de verdad (menos capas = menos nivel) en vez de
    normalizarse cada uno por su cuenta."""
    if "ganancia" in p:
        return p["ganancia"]
    ref = next((S for S in p["secciones"] if S["tipo"] == "A"), p["secciones"][0])
    sonda = {**ref, "compases": 4, "bat_desde": 0, "bajo_desde": 0, "mel_desde": 0, "fill": None, "tipo": "A", "capa": False,
             "frases": [("M", "pregunta", []), ("M", "respuesta", [])]}
    p["ganancia"] = 1.0
    cuerpo, _ = render_seccion(p, sonda, seed, 0.0)
    rms = float(np.sqrt(np.mean(cuerpo ** 2))) + 1e-9
    p["ganancia"] = 10 ** (-16 / 20) / rms
    return p["ganancia"]


class _Salida:
    """Quién toca las notas (los sintetizadores de lofigen). En la 3.0 hubo una variante con muestras
    reales por FluidSynth; el usuario la descartó en la 3.2 («no me ha gustado, que ocupe menos»)."""

    def __init__(self, p):
        self.p = p

    def nota(self, buf, inst, m, t, dur, vol, cinta=False):
        L.tocar_nota(buf, inst, m, t, dur, vol, cinta)

    def bajo(self, buf, m, t, dur, vol):
        L.bajo_nota(buf, m, t, dur, vol)

    def bombo(self, buf, t, vol, kicks):
        L.bombo(buf, t, vol, kicks)

    def golpe(self, buf, t, vol, tipo, frec, dur):
        L.golpe(buf, t, vol, tipo, frec, dur)

    def volcar(self, bus, melb, resto, bat, padb):
        pass


def render_seccion(p, S, seed, fondo=0.0):
    ganancia = _ganancia(p, seed)
    rng = np.random.default_rng(seed)
    sal = _Salida(p)
    L._RNG = np.random.default_rng(seed + 7)
    L._KIT = {**L.KIT_BASE, **p["kit"]}
    L._BAJO = p["bajoTimbre"]
    L._BAJO_NIVEL = float(p["bajoNivel"])
    niv = {"acordes": 1, "melodia": 1, "bateria": 1, **p["niveles"]}
    negra = 60 / p["bpm"]
    corchea = negra / 2
    sw = p["sw"] * corchea
    compas_dur = negra * 4
    nc = S["compases"]
    total = int((nc * compas_dur + 4) * SR)
    bus, resto, melb, bat, padb = (np.zeros(total) for _ in range(5))
    kicks = []
    cinta = p["tex"] == "cinta"
    inst, capa = p["inst"], p["capa"]
    inst_mel = p["instMelB"] if S["tipo"] == "B" else p["instMel"]
    acordes = p["acordes"][S["prog"]]
    nA = len(acordes)
    G = GROOVES.get(p["bat"], GROOVES["boombap"])
    R = L.RITMOS.get(p["bat"], L.RITMOS["boombap"])
    vol_ac = (.06 if inst == "pad" else .085) * niv["acordes"]
    solo = p["solo"]
    humano = lambda s: float(rng.normal(0, s))
    tarde = G.get("tarde", 0)
    capa_prev = None
    ultimo = S["tipo"] == "cierre"
    for c in range(nc):
        t = c * compas_dur
        cuando = lambda i: t + i * corchea + (sw if i % 2 else 0)
        raiz, notas, pcs = acordes[c % nA]
        raiz_sig, _, _ = acordes[(c + 1) % nA]
        final_pieza = ultimo and c >= nc - 2          # los dos últimos compases del cierre: acorde largo y nada más
        va = vol_ac * (4 / len(notas)) ** .5
        acorde = lambda i, d, v: [sal.nota(bus, inst, m, cuando(i) + humano(.004), d, v * (1 + humano(.06)), cinta) for m in notas]
        # ── acordes ──
        if solo:
            pass
        elif final_pieza:
            if c == nc - 2:
                acorde(0, negra * 7.5, va * 1.05)
        elif S["tipo"] == "intro" and inst != "pad":
            acorde(0, negra * 3.8, va * .8)             # la intro: acordes largos y nada más
        else:
            _acompanar(sal, p["comp"], bus, inst, notas, cuando, corchea, negra, va, cinta, acorde, S["tipo"], c)
        # ── pad de fondo (intro, descanso, B…): acorde largo, muy bajo ──
        if S["pad"] and inst != "pad" and not solo and not final_pieza:
            for m in notas[:3]:
                sal.nota(padb, "pad", m, cuando(0), negra * 4.2, .022, False)
        # ── capa: nota guía (tercera o séptima) sostenida, conducida ──
        if S["capa"] and capa and not solo and not final_pieza:
            guias = [m for m in notas if (m - raiz) % 12 in (3, 4, 10, 11)] or notas[1:2]
            m = min(guias, key=lambda x: abs(x - capa_prev)) if capa_prev is not None else guias[0]
            m += 12 if capa in ("flauta", "campana", "vibra", "cuenco") else 0
            capa_prev = m
            if c % 2 == 0 or rng.random() < .5:
                sal.nota(resto, capa, m, cuando(0) + corchea * (0 if c % 2 == 0 else 3), negra * 3.2, .03, cinta)
        # ── cuarto instrumento: arpegio suave una octava arriba (en B, en las A repetidas y en el descanso) ──
        if p.get("inst2") and (S["capa"] or S["tipo"] in ("B", "descanso")) and not solo and not final_pieza:
            arp = [m + 12 for m in notas[:4]]
            patron = (0, 2, 3, 5, 6) if S["tipo"] != "descanso" else (0, 4)
            for j, i in enumerate(patron):
                sal.nota(resto, p["inst2"], arp[(j + c) % len(arp)], cuando(i) + humano(.005), corchea * 1.6, .03 * (1 + humano(.08)), cinta)
        # ── bajo ──
        if S["bajo"] and c >= S["bajo_desde"] and not solo and not final_pieza:
            r = raiz - 12
            if p["bajoGrave"] and r > 40:   # zona sub: 41-55 Hz
                r -= 12
            r_sig = raiz_sig - 12 - (12 if p["bajoGrave"] and raiz_sig - 12 > 40 else 0)
            cambia = (c + 1) % nA != c % nA and raiz_sig != raiz
            _bajo(sal, p["bajo"], resto, r, r_sig, pcs, cuando, negra, corchea, rng, cambia, S["tipo"], c, p["tonica"])
        # ── batería ──
        if S["bat"] and c >= S["bat_desde"] and not solo and not final_pieza:
            fill = S["fill"] if c == nc - 1 else None
            entrada = c == S["bat_desde"] and S["tipo"] == "A"
            _bateria(sal, R, G, bat, cuando, corchea, rng, kicks, S["bat"] * niv["bateria"], S["hats"], fill, c, tarde, humano, entrada)
        if p["bonsho"] and c % 8 == 0 and not solo:
            sal.nota(resto, "bonsho", p["tonica"] - 12, t, 5, .13)
        # ── melodía: frase de dos compases ──
        if c % 2 == 0 and c // 2 < len(S["frases"]) and S["frases"][c // 2] and not final_pieza:
            nombre, tipo, vars_ = S["frases"][c // 2]
            m = p["motivos"][nombre]
            for v in vars_:
                m = _variar(rng, m, v)
            ac2 = [(acordes[c % nA][0], acordes[c % nA][2]), (acordes[(c + 1) % nA][0], acordes[(c + 1) % nA][2])]
            notas_mel = _realizar(m, p["esc"], p["centro_idx"], ac2, p["tonica"], tipo, p["tonica"] + 29)
            vol_mel = (.09 if solo else .062) * niv["melodia"]
            for o, d, midi in notas_mel:
                if o >= 16:
                    continue
                sal.nota(melb, inst_mel, midi, cuando(o) + humano(.006), corchea * d * .95, vol_mel * (1 + humano(.08)), cinta)
    # ── mezcla ──
    sal.volcar(bus, melb, resto, bat, padb)
    if p.get("_debug") is not None:   # niveles por capa (calibración)
        p["_debug"][S["nombre"]] = {k: round(20 * float(np.log10(np.sqrt(np.mean(v ** 2)) + 1e-9)), 1)
                                    for k, v in (("acordes", bus), ("melodia", melb), ("resto", resto), ("bateria", bat), ("pad", padb))}
    pump = np.ones(total)
    rec = int(.28 * SR)
    curva = .45 + .55 * np.linspace(0, 1, rec)
    for tk in kicks:
        i = max(0, int(tk * SR))
        j = min(i + rec, total)
        if j > i:
            pump[i:j] = np.minimum(pump[i:j], curva[:j - i])
    mix = L.reverb(bus * pump + melb + padb, *p["sala"]) + resto + bat
    mix = L.lowpass(mix, p["lofi"])
    if p["graves"]:
        mix += L.lowpass(mix, 80) * p["graves"]
    mix = L.highpass(mix, 28)   # nada por debajo de lo audible: evita que el sub ensucie el limitador
    if fondo and not solo:
        mix += L.textura(total, p["tex"]) * .5 * p["texVol"] * fondo
    mix = mix * ganancia
    rms = float(np.sqrt(np.mean(mix ** 2))) + 1e-9
    tope = 10 ** (-14 / 20)
    if rms > tope:   # por si una sección sale más fuerte que la de referencia
        mix *= tope / rms
    mix = np.tanh(mix * 1.4) / 1.4
    corte = int(nc * compas_dur * SR)
    cola = mix[corte:] * np.linspace(1, 0, total - corte)
    return mix[:corte], cola


def _acompanar(sal, comp, bus, inst, notas, cuando, corchea, negra, va, cinta, acorde, tipo, c):
    ac = notas
    if comp == "ostinato":
        for i in range(8):
            sal.nota(bus, inst, ac[i % 2] + (12 if i % 2 else 0), cuando(i), corchea * 1.2, va * .6, cinta)
    elif comp == "bossa":
        for i, d in ((0, 2.2), (3, 1.4), (4, 2.2), (7, 1.4)):
            acorde(i, corchea * d, va * .8)
    elif comp == "balanceo":
        sal.nota(bus, inst, ac[0], cuando(0), negra * 2, va, cinta)
        acorde(2, corchea * 2.4, va * .7)
        acorde(6, corchea * 2.4, va * .7)
    elif comp == "pad" or inst == "pad":
        acorde(0, negra * 4, va * .9)
    elif comp == "stabs":
        for i in (0, 3, 6):
            acorde(i, corchea * 1.3, va * (1 if i == 0 else .7))
    elif comp == "charleston":
        acorde(0, corchea * 2.6, va)
        acorde(3, corchea * 2.2, va * .75)
    elif comp == "arpegio":
        notas2 = ac + [ac[1] + 12]
        for i, k in enumerate((0, 1, 2, 3, 4, 3, 2, 1)):
            sal.nota(bus, inst, notas2[k % len(notas2)], cuando(i), corchea * 1.8, va * .8, cinta)
    elif comp == "roto":
        sal.nota(bus, inst, ac[0], cuando(0), negra * 3.5, va, cinta)
        for i, m in zip((2, 3, 4, 6), ac[1:] + ac[1:]):
            sal.nota(bus, inst, m, cuando(i), corchea * 2.5, va * .75, cinta)
    else:   # largo
        acorde(0, negra * 3.7, va)
        if inst != "pad" and c % 2:
            acorde(5, negra * 1.4, va * .6)


def _bajo(sal, est, resto, r, r_sig, pcs, cuando, negra, corchea, rng, cambia, tipo, c, tonica):
    """Bajo según su patrón, y una nota de paso hacia la raíz siguiente cuando cambia el acorde."""
    tercera = r + (3 if (r + 3) % 12 in pcs else 4)
    quinta = r + 7
    if est == "largo":
        sal.bajo(resto, r, cuando(0), negra * 3.5, .075)
    elif est == "rebote":
        sal.bajo(resto, r, cuando(0), negra * .9, .075)
        sal.bajo(resto, r + 12, cuando(3), negra * .6, .05)
        sal.bajo(resto, quinta if rng.random() < .5 else r, cuando(4), negra * 1.4, .065)
    elif est == "octavas":
        for i in (0, 2, 4, 6):
            sal.bajo(resto, r + (12 if i in (2, 6) else 0), cuando(i), negra * .8, .06)
    elif est == "sincopado":
        sal.bajo(resto, r, cuando(0), negra * 1.2, .075)
        sal.bajo(resto, r, cuando(3), negra * .7, .055)
        sal.bajo(resto, quinta if rng.random() < .4 else r, cuando(5), negra * 1.2, .065)
    elif est == "pedal":
        ped = tonica - 12
        while ped > r + 5:
            ped -= 12
        sal.bajo(resto, ped, cuando(0), negra * 3.8, .07)
    else:   # caminante: raíz, tercera o quinta, y aproximación cromática o diatónica
        sal.bajo(resto, r, cuando(0), negra * .9, .065)
        sal.bajo(resto, tercera if rng.random() < .6 else quinta, cuando(2), negra * .9, .06)
        sal.bajo(resto, quinta if rng.random() < .6 else r + 12, cuando(4), negra * .9, .06)
        if cambia:
            apro = r_sig + int(rng.choice([-1, 1, -2, 2]))
            sal.bajo(resto, apro, cuando(6), negra * .9, .06)
        else:
            sal.bajo(resto, tercera, cuando(6), negra * .9, .055)
        return
    if cambia and est != "pedal" and rng.random() < .55:   # nota de paso en la última corchea
        apro = r_sig + int(rng.choice([-1, 1, 2, -2, 7, -5]))
        sal.bajo(resto, apro, cuando(7), corchea * .9, .05)


def _bateria(sal, R, G, bat, cuando, corchea, rng, kicks, nivel, hats, fill, c, tarde, humano, entrada):
    K = R["K"]
    if G["varK"] and c % 2 == 1 and rng.random() < .55:   # compases pares: variante de bombo
        K = str(rng.choice(G["varK"]))
    if entrada:   # la batería entra con un pickup: solo la segunda mitad del compás
        K, S_ = "....x..x", "......x."
    else:
        S_ = R["S"]
    hv = hats * G["hats"] * L._KIT["hv"]
    for i in range(8):
        if fill == "corte" and i >= 4:
            break
        if fill in ("redoble", "pickup") and i >= 4:
            break
        if K[i] == "x":
            sal.bombo(bat, cuando(i) + humano(.003), (.34 if i == 0 else .26) * nivel * (1 + humano(.08)), kicks)
        if S_[i] == "x":
            sal.golpe(bat, cuando(i) + tarde + humano(.003), .13 * nivel * (1 + humano(.1)), "bandpass", 1500 * L._KIT["sn"], L._KIT["sd"])
        elif S_[i] == "r":
            sal.golpe(bat, cuando(i) + humano(.003), .08 * nivel, "bandpass", 2600 * L._KIT["sn"], .05)
        if hv > 0:
            if R["H"][i] == "x" and not (rng.random() < .1):
                sal.golpe(bat, cuando(i) + humano(.006), (.018 if i % 2 else .028) * hv * (1 + humano(.15)), "highpass", 7500 * L._KIT["hh"], .045)
            elif R["H"][i] == "o":
                sal.golpe(bat, cuando(i) + humano(.005), .025 * hv, "highpass", 6500 * L._KIT["hh"], .22)
        if i % 2 and S_[i] == "." and rng.random() < .12 * nivel:   # alguna nota fantasma
            sal.golpe(bat, cuando(i) + humano(.004), .03 * nivel, "bandpass", 1500 * L._KIT["sn"], L._KIT["sd"] * .7)
        if R.get("W", "........")[i] == "x":
            sal.golpe(bat, cuando(i), .09, "bandpass", 1000, .035)
    if fill == "redoble":       # semicorcheas de caja creciendo en los dos últimos tiempos
        for k in range(8):
            sal.golpe(bat, cuando(4) + k * corchea / 2 + humano(.003), (.03 + .07 * k / 7) * nivel, "bandpass", 1500 * L._KIT["sn"], .08)
    elif fill == "pickup":      # bombo, bombo, caja
        sal.bombo(bat, cuando(5), .26 * nivel, kicks)
        sal.bombo(bat, cuando(6), .26 * nivel, kicks)
        sal.golpe(bat, cuando(7) + tarde, .13 * nivel, "bandpass", 1500 * L._KIT["sn"], L._KIT["sd"])
    elif fill == "charles":     # charles abierto en la última corchea
        sal.golpe(bat, cuando(7), .035 * max(hats, .5), "highpass", 6500 * L._KIT["hh"], .3)


# ══════════════════════════════════════════════════════════════════════════════════════════════
# piezas enteras (prueba y biblioteca)
# ══════════════════════════════════════════════════════════════════════════════════════════════

def render_pieza(p, seed, fondo=0.0, progreso=None):
    """Renderiza la pieza completa sección a sección, superponiendo la cola de cada una al
    principio de la siguiente (como hace la radio). Devuelve float32 mono normalizado."""
    partes = []
    cola = None
    for k, S in enumerate(p["secciones"]):
        cuerpo, cola_nueva = render_seccion(p, S, seed + k * 101, fondo)
        if cola is not None:
            n = min(len(cola), len(cuerpo))
            cuerpo[:n] += cola[:n]
        partes.append(cuerpo)
        cola = cola_nueva
        if progreso:
            progreso(k + 1, len(p["secciones"]))
    partes.append(cola)
    audio = np.concatenate(partes)
    f_in = int(1.5 * SR)
    audio[:f_in] *= np.linspace(0, 1, f_in)
    return audio / (np.max(np.abs(audio)) + 1e-9) * .92


def generate_pieza(seed, outdir, estilo="aleatorio", variedad="medio", perfil=None, fondo=0.0, prev=None):
    """Compone y renderiza una pieza entera a WAV; devuelve (ruta, pieza)."""
    from pathlib import Path
    from scipy.io import wavfile
    p = componer(seed, estilo, None, variedad, 0, 2, prev, False, perfil)
    audio = render_pieza(p, seed, fondo)
    out = Path(outdir) / f"pieza_{seed:06d}.wav"
    wavfile.write(str(out), SR, (audio * 32767).astype(np.int16))
    return out, p


def piezas_de_prueba(variedad="medio", estilo="aleatorio"):
    """Tres piezas contrastadas: perfiles distintos, tonalidades y tempos separados."""
    rng = np.random.default_rng(int(time.time()))
    nombres = [p for p in PERFILES_ESTILO.get(estilo, list(PERFILES)) if p in PERFILES]
    if len(nombres) < 3:
        nombres = list(PERFILES)
    elegidos = [str(x) for x in rng.choice(nombres, size=3, replace=False)]
    return [(int(rng.integers(1, 1_000_000)), perfil) for perfil in elegidos]


def _cli():
    args = sys.argv[1:]
    if not args or args[0] != "--prueba":
        print(__doc__)
        return
    outdir = args[1] if len(args) > 1 else "."
    variedad = args[args.index("--variedad") + 1] if "--variedad" in args else "medio"
    prev = None
    for k, (seed, perfil) in enumerate(piezas_de_prueba(variedad)):
        t0 = time.time()
        out, p = generate_pieza(seed, outdir, "aleatorio", variedad, perfil, 0.0, prev)
        prev = p
        print(f"{k + 1}/3 {out.name}: {p['perfil_nombre']} · {p['tono']} · {p['bpm']} bpm · {p['inst']}+{p['instMel']}"
              f"{' (+' + p['capa'] + ')' if p['capa'] else ''} · {p['bat']} · {p['forma']} · {p['dur'] / 60:.1f} min "
              f"(render {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    _cli()
