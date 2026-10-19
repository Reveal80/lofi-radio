# Radio Lofi 24/7

Panel web para tener una **radio de música lofi emitiendo 24 horas en YouTube** desde tu propio
servidor. Genera la música por código (sin inteligencia artificial ni servicios externos),
crea videos de escenas de ciudad en bucle y lo emite todo por RTMP con ffmpeg.
Además puedes ver el directo en tu red local, en el propio panel o con VLC, sin pasar por YouTube.

Toda la interfaz está en español.

## Qué incluye

- **Música procedural**: dieciséis estilos (para estudiar, chill, jazz hop con saxo, día de lluvia, nocturno,
  ambiente, retro 8 bits, templo japonés, bossa nova, neo soul, café de mañana, invierno, neón, bosque,
  blues lento y aleatorio). Cada canción es distinta y se guarda como MP3 con su título.
- **Videos de escena**: ciudad de noche, amanecer, día, atardecer, lluvia, nieve o aurora, en bucle
  perfecto de 20 segundos. También puedes subir tus propios MP4, imágenes fijas (se convierten en un video de 10 s
  en bucle) o importar música de Pixabay.
- **Cola de video**: elige qué videos se emiten, cuántas horas cada uno y en qué orden. Se puede
  cambiar en caliente sin cortar el directo.
- **Emisión 24/7**: la salida a YouTube reconecta sola, sin límite de intentos y sin cortar la vista
  previa, y un vigilante relanza la emisión si el codificador se cae. Modo *local* para probar sin clave
  y modo *directo* para YouTube.
- **Vista previa**: reproductor del directo dentro del panel y acceso por RTSP para VLC.
- **Radio infinita**: música lofi compuesta en tiempo real, sin fin y sin repetirse. Elige el estilo y el
  tempo mientras suena; los cambios entran en el siguiente compás y el estilo va cambiando poco a poco.
- **Video infinito**: lluvia de letras tipo Matrix generada en tiempo real, con velocidad y color
  ajustables mientras se emite, como alternativa a la cola de videos.
- **Stats**: uso de CPU, memoria, red y disco del contenedor y del servidor, estado de la gráfica y
  velocidad del codificador, en un panel lateral que se actualiza cada 2 segundos.

## Puesta en marcha

Necesitas Docker con Docker Compose. Crea una carpeta, guarda dentro el archivo
`docker-compose.yml` de más abajo y ejecuta:

```bash
docker compose up -d
```

Abre `http://IP-del-servidor:8080`. En VLC, el directo está en `rtsp://IP-del-servidor:8554/radio`.

Primeros pasos en el panel:

1. **Componer → Música**: genera unas cuantas canciones.
2. **Componer → Video**: genera una o más escenas, y en *Biblioteca → Videos* activa «en bucle».
3. Pulsa **Emitir en local** para comprobar que todo se ve y se oye.
4. Pega tu **clave de emisión de YouTube** (YouTube Studio → Emitir en directo → Clave de emisión)
   y pulsa **Emitir en directo**.

## docker-compose.yml

Son tres servicios: el panel (esta imagen), MediaMTX para la vista previa y Apache como proxy.
No hace falta descargar nada más; la configuración de Apache va dentro del propio archivo.

```yaml
services:
  python:
    image: therliss/lofi-radio:latest
    logging:
      driver: json-file
      options: { max-size: "5m", max-file: "1" }
    volumes:
      - ./data:/app/data   # música, videos y config.json (clave de YouTube)
      - ./logs:/app/logs   # registros de la emisión
    # GPU integrada (opcional): quita el «#» de las tres líneas siguientes para ceder la gráfica del
    # servidor al contenedor (Intel/AMD por VAAPI). Solo si existe /dev/dri/renderD128 en el servidor;
    # si no, el contenedor no arrancaría.
    # devices:
    #   - /dev/dri:/dev/dri
    # group_add: ["video", "render"]
    depends_on:
      - mediamtx
    expose:
      - "8750"
    restart: unless-stopped

  mediamtx:
    image: bluenviron/mediamtx:1.9.3
    logging:
      driver: json-file
      options: { max-size: "5m", max-file: "1" }
    environment:
      MTX_HLSVARIANT: fmp4
      MTX_HLSALWAYSREMUX: "yes"
      MTX_HLSSEGMENTDURATION: 1s
    ports:
      - "${PREVIEW_PORT:-8554}:8554"
    restart: unless-stopped

  apache:
    image: php:8.3-apache
    logging:
      driver: json-file
      options: { max-size: "5m", max-file: "1" }
    depends_on:
      - python
    ports:
      - "${HTTP_PORT:-8080}:80"
    configs:
      - source: radio_conf
        target: /etc/apache2/sites-available/000-default.conf
    command: sh -c "a2enmod proxy proxy_http >/dev/null && exec apache2-foreground"
    restart: unless-stopped

configs:
  radio_conf:
    content: |
      <VirtualHost *:80>
          ServerName localhost
          LimitRequestBody 0
          ProxyTimeout 900
          ProxyPreserveHost On
          ProxyPass        /live/ http://mediamtx:8888/
          ProxyPassReverse /live/ http://mediamtx:8888/
          ProxyPass        / http://python:8750/
          ProxyPassReverse / http://python:8750/
          ErrorLog $${APACHE_LOG_DIR}/error.log
          SetEnvIf Request_URI "^/(api/(state|logs)|live/)" sondeo
          CustomLog $${APACHE_LOG_DIR}/access.log combined env=!sondeo
      </VirtualHost>
```

### Variables opcionales

| Variable | Por defecto | Para qué |
|---|---|---|
| `HTTP_PORT` | `8080` | Puerto del panel web |
| `PREVIEW_PORT` | `8554` | Puerto RTSP para VLC |

Ejemplo: `HTTP_PORT=9000 docker compose up -d`.

### Datos persistentes

| Carpeta | Contenido |
|---|---|
| `./data/tracks` | Canciones MP3, orden de emisión y títulos |
| `./data/videos` | Videos MP4, miniaturas y cola de emisión |
| `./data/config.json` | Clave de emisión de YouTube y estado del vigilante |
| `./logs` | Registros de ffmpeg y del vigilante (se vacían cada 24 h) |

Para hacer copia de seguridad basta con copiar la carpeta `data`.

## Puertos y accesos

| Puerto | Servicio |
|---|---|
| 8080 | Panel web y directo en HLS (`/live/radio/index.m3u8`) |
| 8554 | RTSP para VLC u otros reproductores (`/radio`) |

El panel **no tiene contraseña**: está pensado para una red local. Si lo expones a Internet,
ponle delante un proxy con autenticación o una VPN.

## Requisitos

- Unos 1,5 núcleos de CPU mientras emite (codificación h264 a 1080p y 30 fps por software).
- GPU integrada: el `docker-compose.yml` trae un bloque opcional (`devices` / `group_add`, comentado) para
  ceder la gráfica al contenedor. De momento la emisión codifica siempre por CPU; la codificación por
  hardware (VAAPI) llegará en una versión posterior.
- Alrededor de 1 GB de RAM.
- Subida a Internet de unos 9 Mbit/s para YouTube a 1080p (la emisión va a 6800 kbps, lo que
  recomienda YouTube; se puede bajar con `"video_kbps": 4500` en `data/config.json`).

## Seguridad y privacidad

- La imagen no contiene datos, claves ni cuentas. Todo lo personal vive en tu carpeta `data`.
- La clave de YouTube solo se guarda en tu servidor y solo se usa para el RTMP de YouTube.
- No se llama a ninguna API externa. La única salida a Internet es la emisión RTMP hacia YouTube.

## Actualizar

```bash
docker compose pull
docker compose up -d
```

Los datos se conservan porque están en las carpetas montadas.

## Dentro de la imagen

Python 3.12 con Flask, numpy y scipy, y ffmpeg 7. El código del panel está en `/app`.
Variables de entorno: `PORT` (8750), `CONFIG_FILE` (`/app/data/config.json`) y
`PREVIEW_RTMP` (`rtmp://mediamtx:1935/radio`).
