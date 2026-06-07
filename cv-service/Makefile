.PHONY: build up down dev test lint shell logs

# ── Docker commands ──────────────────────────────────────────
build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

dev:
	docker compose --profile dev up

logs:
	docker compose logs -f cv-service

# ── Local dev (no Docker) ────────────────────────────────────
run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

worker:
	celery -A app.services.worker worker --loglevel=info --concurrency=1 --queues=cv

# ── Quality ──────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

lint:
	ruff check app/ tests/
	mypy app/ --ignore-missing-imports

# ── Utilities ────────────────────────────────────────────────
shell:
	docker compose exec cv-service /bin/bash

env:
	cp .env.example .env
	@echo ".env created — update USDA_API_KEY and API_SECRET_KEY"
