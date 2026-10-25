FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    DISPLAY=:0 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    GALLIUM_DRIVER=llvmpipe \
    SINE_WIDTH=480 \
    SINE_HEIGHT=270

RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 libgl1 libgl1-mesa-dri libglib2.0-0 libportaudio2 \
    libx11-6 libxext6 libxrender1 libxrandr2 libxcursor1 libxi6 \
    mesa-utils-bin novnc websockify x11vnc xvfb xauth curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 runner \
    && mkdir -p /app /output \
    && chown -R runner:runner /app /output

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --requirement requirements.txt
COPY game.py test_game.py README.md ./
USER runner
VOLUME ["/output"]
EXPOSE 6080

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=5 \
  CMD curl --fail --silent http://127.0.0.1:6080/vnc.html > /dev/null || exit 1

CMD ["sh", "-c", "Xvfb :0 -screen 0 1280x720x24 -nolisten tcp & x11vnc -display :0 -forever -shared -nopw -rfbport 5900 >/tmp/x11vnc.log 2>&1 & websockify --web=/usr/share/novnc 6080 localhost:5900 >/tmp/websockify.log 2>&1 & exec python game.py --out /output"]
