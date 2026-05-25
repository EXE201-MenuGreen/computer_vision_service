from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.core.config import settings
from app.core.logging import setup_logging, get_logger

# Import stages so they self-register with registries
import app.stages  # noqa: F401

from app.pipeline.pipeline_factory import PipelineFactory
from app.api.cv_router import router as cv_router
from app.api.history_router import router as history_router
from app.api.admin_router import router as admin_router

setup_logging()
logger = get_logger(__name__)


# ── Lifespan: build & load pipeline on startup ─────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup", env=settings.app_env, device=settings.device)

    # Build pipeline from config (reads PIPELINE_* env vars)
    pipeline = PipelineFactory.build(settings.pipeline_config)
    pipeline.load_all()  # blocking — intentional on startup
    app.state.pipeline = pipeline

    yield
    logger.info("shutdown")


# ── App factory ─────────────────────────────────────────────
app = FastAPI(
    title="Food CV Microservice",
    description="Computer Vision pipeline: food detection → portion estimation → nutrition",
    version="1.0.0",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    lifespan=lifespan,
)

# CORS — only allow the C# backend (and localhost for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "Authorization"],
)

# Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app)

# Routers
app.include_router(cv_router)
app.include_router(history_router)
app.include_router(admin_router)


# ── Global exception handler ────────────────────────────────
@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    logger.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── Dev runner ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=not settings.is_production,
        workers=settings.app_workers,
    )
