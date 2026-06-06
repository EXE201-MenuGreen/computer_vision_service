# Food CV Microservice

Python FastAPI · YOLOv8 · EfficientNet · DepthAnything v2 · Docker

## Overview

This service receives an image, runs a computer-vision pipeline to detect and classify food, estimates portions, and looks up nutrition data.

## Project structure

```text
cv-service/
├── app/
│   ├── api/
│   ├── core/
│   ├── db/
│   ├── embeddings/
│   ├── pipeline/
│   ├── registry/
│   ├── schemas/
│   ├── services/
│   └── stages/
├── tests/
├── weights/
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-ml.txt
└── requirements-dev.txt
```

## Requirements files

The project is split into three dependency groups:

- `requirements.txt` — core runtime dependencies for the API
- `requirements-ml.txt` — AI / embedding / inference dependencies
- `requirements-dev.txt` — testing and local development tooling

### Which file should you install?

- For local development or running the full service, install `requirements.txt` and `requirements-ml.txt`.
- For test/lint work, additionally install `requirements-dev.txt`.
- For Docker production builds, the Dockerfile installs `requirements.txt` and `requirements-ml.txt` automatically.

Example installation:

```bash
pip install -r requirements.txt
pip install -r requirements-ml.txt
pip install -r requirements-dev.txt
```

## Quick start

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows CMD:

```bash
.venv\Scripts\activate
```

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

For normal running:

```bash
pip install -r requirements.txt
pip install -r requirements-ml.txt
```

For development and tests:

```bash
pip install -r requirements.txt
pip install -r requirements-ml.txt
pip install -r requirements-dev.txt
```

### 3) Configure environment

Copy `.env.example` to `.env` and adjust values as needed.

Important environment values:

```env
DATABASE_URL=http://your-database-server
DATABASE_READ_URL=http://your-read-endpoint
DATABASE_WRITE_URL=http://your-write-endpoint
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=64
```

### 4) Start Redis

If Redis is needed in your local workflow:

```bash
docker compose up -d redis
```

### 5) Start API service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Docker build

Build the production image:

```bash
docker build -t cv-service .
```

Run the container:

```bash
docker run --rm -p 8000:8000 --env-file .env cv-service
```

## API endpoints

All routes are versioned under `/api/v1`.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/cv/health` | Health check + model status |
| POST | `/api/v1/cv/analyze` | Sync image analysis |
| POST | `/api/v1/cv/analyze/async` | Queue async analysis job |
| GET | `/api/v1/cv/jobs/{job_id}` | Poll async job result |
| POST | `/api/v1/cv/history/query` | Search meal history |
| GET | `/api/v1/cv/history/me` | Get recent meals |
| POST | `/api/v1/cv/admin/cache/clear` | Clear nutrition cache |
| POST | `/api/v1/cv/admin/verified` | Upsert verified nutrition entry |
| DELETE | `/api/v1/cv/admin/verified/{label}` | Delete verified nutrition entry |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | Swagger UI |

## Database connection

The service now uses a deployed database server instead of Supabase SDK.

Set one of the following:

```env
DATABASE_URL=http://localhost:3000
```

Or separate read/write endpoints:

```env
DATABASE_READ_URL=http://localhost:3000
DATABASE_WRITE_URL=http://localhost:3001
```

## Model weights

- `weights/yolov8_food.pt` — YOLO detector
- `weights/efficientnet_food.pt` — EfficientNet classifier
- `DepthAnything-V2-Small` is downloaded automatically on first run

## Running tests

```bash
pytest tests/ -v
```

If you need the development tooling first, install `requirements-dev.txt`.
