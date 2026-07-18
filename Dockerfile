FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMSENSE_DATABASE=/data/events.db

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir ".[media]"

RUN useradd --create-home --uid 10001 streamsense && mkdir -p /data && chown streamsense /data
USER streamsense

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["streamsense", "serve", "--host", "0.0.0.0", "--port", "8000"]
