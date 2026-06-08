# Food CV Microservice

Python FastAPI · Google Gemini API · Celery · Redis · Docker

## Overview

This service receives an image, forwards it to an external AI API (like Google Gemini) using Celery async task worker with Redis, and returns structured Vietnamese recipe and nutrition data.

---

## Project Structure

```text
cv-service/
├── app/
│   ├── api/            # API routers (cv, history, admin)
│   ├── core/           # Configuration, logging, base schemas
│   ├── db/             # Database clients (Supabase/PostgREST)
│   ├── embeddings/     # Text embeddings for meal history search
│   ├── schemas/        # Pydantic schemas (Pydantic v2)
│   └── services/       # Core services (worker.py, inference_client.py)
├── tests/              # Unit and integration tests (pytest)
├── Dockerfile          # Optimized runtime Dockerfile (slim python)
├── docker-compose.yml  # Docker environment (FastAPI, Redis, Celery)
├── requirements.txt    # Core runtime dependencies
tools
```

---

## Requirements Files

The project has two dependency groups:

- `requirements.txt` — Core runtime dependencies for the API and Celery worker.


### Installation

For running the service locally:
```bash
pip install -r requirements.txt
```

For development and running tests:
```bash
pip install -r requirements.txt
```

---

## Quick Start (Local Development)

### 1) Create and Activate a Virtual Environment

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

Linux/macOS:
```bash
source .venv/bin/activate
```

### 2) Configure Environment

Copy `.env.example` to `.env` and adjust the values. Key values include:

```env
# --- Server Config ---
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000

# --- Security ---
API_SECRET_KEY=your-shared-secret-key  # Key for C# Backend ↔ CV Service auth. Leave empty in DEV to disable auth.

# --- Redis & Celery ---
REDIS_URL=redis://localhost:6379/0
```

### 3) Start Redis & Celery Worker

Start Redis using Docker Compose:
```bash
docker compose up -d redis
```

In a separate terminal, activate the virtual environment and start the Celery worker:
```bash
celery -A app.services.worker worker --loglevel=info --queues=cv
```

### 4) Start API Service

In a separate terminal, activate the virtual environment and start the FastAPI web server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Access Swagger UI documentation at: `http://localhost:8000/docs`.

---

## Running with Docker Compose (Recommended)

To build and run all services (FastAPI, Redis, Celery worker) together in Docker containers:

```bash
# Build the containers without cache
docker compose build --no-cache

# Run the services in the background
docker compose up -d
```

---

## API Endpoints

All routes are versioned under `/api/v1`. Authenticated endpoints require `Authorization: Bearer <API_SECRET_KEY>` header.

| Method | Path | Description |
|--------|------|-------------|
| **GET** | `/api/v1/cv/health` | Health check + model status |
| **POST** | `/api/v1/cv/analyze` | Queue async image analysis job (returns `job_id`) |
| **GET** | `/api/v1/cv/jobs/{job_id}` | Poll async job status and retrieve result |
| **POST** | `/api/v1/cv/history/query` | Search meal history using natural language |
| **GET** | `/api/v1/cv/history/me` | Get recent meal logs for the current user |
| **POST** | `/api/v1/cv/admin/cache/clear` | Clear nutrition cache (admin) |
| **POST** | `/api/v1/cv/admin/verified` | Upsert verified nutrition entry (admin) |
| **DELETE** | `/api/v1/cv/admin/verified/{label}` | Delete verified nutrition entry (admin) |
| **GET** | `/metrics` | Prometheus metrics |

---

## Database Connection

The service connects directly to a database server. Configure one of the following in `.env`:

```env
DATABASE_URL=http://localhost:3000
```

Or separate endpoints for read/write:
```env
DATABASE_READ_URL=http://localhost:3000
DATABASE_WRITE_URL=http://localhost:3001
```

---

## Running Tests

Verify API routes and mocked inference clients locally:

```bash
# Windows
$env:PYTHONPATH="."
pytest tests/ -v

# Linux/macOS
PYTHONPATH=. pytest tests/ -v
```
