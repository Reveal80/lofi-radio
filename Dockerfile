# Radio Lofi 24/7 — imagen del panel (servicio «python» del docker-compose).
# Lleva dentro todo el código del proyecto; música, videos, registros y config
# van en volúmenes para que persistan y no viajen con la imagen.
FROM python:3.12-slim

# fonts-vlgothic: katakana de media anchura para el video infinito tipo Matrix (8 MB)
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-vlgothic \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir flask==3.1.3 numpy==2.5.4 scipy==1.18.1

WORKDIR /app
COPY app.py audiofeeder.py videofeeder.py escenas.py lofigen.py index.html ./
COPY vendor/ vendor/
COPY apache/ apache/
COPY docker-compose.imagen.yml docker-compose.yml

# Carpetas de datos vacías. La configuración (clave de YouTube) se guarda en
# data/config.json para que persista con el volumen; la imagen no lleva ninguna clave.
RUN mkdir -p data/tracks data/videos logs

ENV PORT=8750 PYTHONUNBUFFERED=1 CONFIG_FILE=/app/data/config.json
EXPOSE 8750
VOLUME ["/app/data", "/app/logs"]
CMD ["python", "app.py"]
