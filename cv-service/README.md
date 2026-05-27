# Food CV Microservice

Python FastAPI · YOLOv8 · EfficientNet · DepthAnything v2 · Docker

## Overview

This service receives an image, runs a computer-vision pipeline to detect and classify food, estimates portions, then looks up nutrition data.

### Current architecture

- `client` → `backend` → `AI service` → `database layer`
- The AI service talks to the database layer through **PostgREST HTTP APIs**.
- Redis is used for cache and async job queue.

## Project structure

```
cv-service/
├── app/
│   ├── api/
│   │   ├── cv_router.py
│   │   ├── history_router.py
│   │   └── admin_router.py
│   ├── core/
│   ├── db/
│   ├── embeddings/
│   ├── pipeline/
│   ├── registry/
│   ├── schemas/
│   ├── services/
│   └── stages/
├── scripts/
├── tests/
├── weights/
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── Makefile
└── requirements.txt
```

## Quick start

### 1) Create and activate virtual environment

```bash
python -m venv .venv
```

On Windows CMD:

```bash
.venv\Scripts\activate
```

On PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment

Copy `.env.example` to `.env` and adjust values as needed.

### 4) Start Redis

If you use Docker:

```bash
docker compose up -d redis
```

### 5) Start API service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
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
| GET | `/docs` | Swagger UI (dev only) |

## Database layer

The service no longer depends on Supabase SDK. It uses PostgREST via HTTP.

Required environment variables:

```env
POSTGREST_URL=http://localhost:3000
POSTGREST_API_KEY=your-api-key
POSTGREST_SERVICE_JWT=your-service-jwt
```

## Behavior without database

If PostgREST is unavailable, the AI pipeline can still run image recognition and fall back to USDA or built-in nutrition data.

## Model weights

- `weights/yolov8_food.pt` — YOLO detector
- `weights/efficientnet_food.pt` — EfficientNet classifier
- `DepthAnything-V2-Small` is downloaded automatically on first run

## Running tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```
