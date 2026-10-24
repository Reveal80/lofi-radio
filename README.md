# Radio Lofi 24/7

Panel web para tener una **radio de música lofi emitiendo 24 horas en YouTube** desde tu propio
servidor. Genera la música por código (sin inteligencia artificial ni servicios externos),
crea videos de escenas de ciudad en bucle y lo emite todo por RTMP con ffmpeg.
Además puedes ver el directo en tu red local, en el propio panel o con VLC, sin pasar por YouTube.

Toda la interfaz está en español.

## Qué incluye

- **Música procedural**: dieciséis estilos (para estudiar, chill, jazz hop con saxo, día de lluvia, nocturno,
  ambiente, retro 8 bits, templo japonés, bossa nova, café de mañana, invierno, neón, bosque, anime lofi,
  blues lento y aleatorio). Cada canción es distinta y se guarda como MP3 con su título.
- **Videos de escena**: ciudad de noche, amanecer, día, atardecer, lluvia, nieve o aurora, en bucle
  perfecto de 20 segundos. También puedes subir tus propios MP4, imágenes fijas (se convierten en un video de 10 s
  en bucle) o importar música de Pixabay; cada canción importada guarda el enlace a su página original, por si
  hace falta demostrar la licencia ante una reclamación de Content ID.
- **Cola de video**: elige qué videos se emiten, cuántas horas cada uno y en qué orden. Se puede
  cambiar en caliente sin cortar el directo. Cada video puede llevar una **neblina** animada por encima,
  pensada para que las imágenes fijas parezcan moverse.
- **Emisión 24/7**: la salida a YouTube reconecta sola, sin límite de intentos y sin cortar la vista
  previa, y un vigilante relanza la emisión si el codificador se cae. Modo *local* para probar sin clave
  y modo *directo* para YouTube.
- **Vista previa**: reproductor del directo dentro del panel y acceso por RTSP para VLC.
- **Radio infinita**: música lofi compuesta en tiempo real, sin fin y sin repetirse: piezas de 3 a 6 minutos
  con identidad propia (perfil, tonalidad, tempo, instrumentos), estructura (intro, secciones, descanso, cierre),
  melodías con pregunta y respuesta y una memoria que evita repetir combinaciones recientes. Elige el estilo, el
  tempo, los instrumentos y la variedad (baja, media, alta) mientras suena; los cambios entran en el siguiente compás.
  Un modo de prueba compone tres piezas contrastadas en la biblioteca para compararlas.
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
    image: httpd:2.4
    logging:
      driver: json-file
      options: { max-size: "5m", max-file: "1" }
    depends_on:
      - python
    ports:
      - "${HTTP_PORT:-8080}:80"
    configs:
      - source: radio_conf
        target: /usr/local/apache2/conf/extra/radio.conf
    command: ["httpd-foreground", "-c", "ServerName localhost", "-c", "LoadModule proxy_module modules/mod_proxy.so", "-c", "LoadModule proxy_http_module modules/mod_proxy_http.so", "-c", "Include conf/extra/radio.conf"]
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
          ErrorLog /proc/self/fd/2
          SetEnvIf Request_URI "^/(api/(state|logs)|live/)" sondeo
          CustomLog /proc/self/fd/1 combined env=!sondeo
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

## Música propia generada con IA (opcional)

Las canciones con licencia libre (Pixabay y similares) pueden recibir reclamaciones de Content ID en
YouTube aunque su licencia lo permita. Una forma de evitarlo es generar música propia. **Esto no forma
parte de la imagen**: se hace aparte, en un contenedor temporal, y luego se suben los MP3 desde el panel.

Hay dos caminos:

| Opción | Ventajas | Inconvenientes |
|---|---|---|
| Servicio de pago (p. ej. Suno) | Calidad alta y rápido | Plan de pago para uso comercial |
| [MusicGen](https://huggingface.co/facebook/musicgen-small) de Meta en tu servidor | Gratis y local | Lento sin GPU NVIDIA |

### Velocidad orientativa de MusicGen (`musicgen-small`)

| Equipo | Tiempo de cálculo |
|---|---|
| CPU de 2 núcleos (i7-6700) | ~25 s por cada segundo de audio (10 min de música ≈ 4 h) |
| GPU NVIDIA | Minutos por canción; necesaria para muchas horas de música |

Con CPU basta generar 2-3 minutos de material por canción y montarlos a compás hasta 10 minutos
(trozos de 4 compases con fundidos), en vez de generar los 10 minutos enteros.

### Contenedor temporal

```bash
mkdir -p ~/musicgen/out && cd ~/musicgen
docker run --rm -it --cpu-shares 64 -v "$PWD":/work -w /work python:3.12-slim bash
# dentro del contenedor:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers soundfile scipy
```

`--cpu-shares 64` le da prioridad baja para no quitar CPU al codificador del directo.

### Generar una pieza

```python
# gen.py: python gen.py "prompt" salida.wav segundos
import sys, torch, soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration
prompt, salida, seg = sys.argv[1], sys.argv[2], float(sys.argv[3])
proc = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").eval()
inp = proc(text=[prompt], padding=True, return_tensors="pt")
with torch.no_grad():                       # 50 tokens = 1 s; el modelo da como mucho ~30 s por llamada
    y = model.generate(**inp, do_sample=True, guidance_scale=3.0, max_new_tokens=int(min(seg, 30) * 50))
sf.write(salida, y[0, 0].numpy(), 32000)
```

Para piezas más largas se encadenan trozos de 30 s pasando los últimos ~10 s de audio como pista
(`proc(audio=[pista], sampling_rate=32000, text=[prompt], ...)`).

Ejemplo de prompt que funciona bien:

```
lo-fi jazz hop, warm Rhodes electric piano chords and gentle jazz piano, soft brushed drums,
deep mellow bass, 72 bpm, cozy morning cafe, relaxed, muffled and warm
```

### Pasar a MP3 y subir

```bash
ffmpeg -i out/pieza.wav -af "highpass=f=28,lowpass=f=9000,loudnorm=I=-14" -ar 44100 -ac 2 \
  -c:a libmp3lame -b:a 320k -metadata title="Café de mañana" pieza.mp3
```

Después se sube `pieza.mp3` desde el botón «Subir» del panel. Como referencia, el lofi suena bien cuando
tiene la mayor parte de la energía por debajo de 200 Hz y casi nada por encima de 2 kHz.

## Actualizar

```bash
docker compose pull
docker compose up -d
```

Los datos se conservan porque están en las carpetas montadas.

## Dentro de la imagen

Python 3.12 con Flask, numpy y scipy, y ffmpeg 7. El código del panel está en `/app`.
Variables de entorno: `PORT` (8750), `CONFIG_FILE` (`/app/data/config.json`),
`PREVIEW_RTMP` (`rtmp://mediamtx:1935/radio`) y `AUTOR` (opcional: artista con el que se
guardan las canciones subidas; si no se define, se respeta el que traiga el archivo).
