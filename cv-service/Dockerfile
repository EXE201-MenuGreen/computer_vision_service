# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Build-time libraries needed to compile/install some Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Create an isolated virtual environment for Python dependencies.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install core runtime first, then AI dependencies only for this image.
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
   


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_MODULE=app.main:app \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    APP_WORKERS=1 \
    CONTAINER_ROLE=api \
    CELERY_APP=app.services.worker \
    CELERY_LOG_LEVEL=info \
    CELERY_CONCURRENCY=1 \
    CELERY_QUEUES=cv \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

# Runtime libraries needed by image processing and container health checks.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*



# Copy the virtual environment from the builder stage.
COPY --from=builder /opt/venv /opt/venv

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create a non-root user for safer execution.
RUN useradd --create-home --uid 1001 --shell /bin/bash cvuser

# Copy application code with correct ownership.
COPY --chown=cvuser:cvuser app/ ./app/


USER cvuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import os, sys, urllib.request; cmd=open('/proc/1/cmdline','rb').read().decode('utf-8','ignore'); role=os.getenv('CONTAINER_ROLE','api').lower(); sys.exit(0) if role == 'worker' or 'celery' in cmd else urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"APP_PORT\", \"8000\")}/api/v1/cv/health', timeout=5).read()"

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["api"]
