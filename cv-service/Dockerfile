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

COPY requirements.txt requirements-ml.txt ./

# Create an isolated virtual environment for Python dependencies.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install core runtime first, then AI dependencies only for this image.
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-ml.txt


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

# Runtime shared libraries for OpenCV / ML packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Create a non-root user for safer execution.
RUN useradd --create-home --uid 1001 --shell /bin/bash cvuser

# Copy application code with correct ownership.
COPY --chown=cvuser:cvuser app/ ./app/
COPY --chown=cvuser:cvuser weights/ ./weights/

USER cvuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/cv/health', timeout=5)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "uvloop", "--no-access-log"]
