#!/usr/bin/env python3
"""Sube README.md como descripción del repositorio en Docker Hub.

Usa la sesión de `docker login` guardada en ~/.docker/config.json (usuario y
contraseña o token). Ejecutar en el servidor:  python3 hub-readme.py
"""
import base64, json, pathlib, sys, urllib.request

REPO = "therliss/lofi-radio"
CORTA = "Radio lofi 24/7 para YouTube: música generada por código, videos en bucle y emisión con ffmpeg."

cfg = json.load(open(pathlib.Path.home() / ".docker" / "config.json"))
entrada = cfg.get("auths", {}).get("https://index.docker.io/v1/", {}).get("auth")
if not entrada:
    sys.exit("No hay sesión de Docker Hub guardada: ejecuta antes `docker login`.")
usuario, clave = base64.b64decode(entrada).decode().split(":", 1)

def llamada(url, datos, metodo="POST", token=None):
    cab = {"Content-Type": "application/json"}
    if token:
        cab["Authorization"] = "JWT " + token
    req = urllib.request.Request(url, json.dumps(datos).encode(), cab, method=metodo)
    with urllib.request.urlopen(req) as r:
        return json.load(r)

try:
    token = llamada("https://hub.docker.com/v2/users/login", {"username": usuario, "password": clave})["token"]
    readme = pathlib.Path(__file__).with_name("README.md").read_text(encoding="utf-8")
    r = llamada(f"https://hub.docker.com/v2/repositories/{REPO}/",
                {"description": CORTA, "full_description": readme}, "PATCH", token)
    print(f"Descripción publicada en https://hub.docker.com/r/{REPO} ({len(r.get('full_description', ''))} caracteres)")
except urllib.error.HTTPError as e:
    sys.exit(f"Error {e.code} de Docker Hub: {e.read().decode()[:300]}")
