from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.admin_router import router as admin_router
from app.api.cv_router import router as cv_router
from app.api.history_router import router as history_router
from app.core.config import settings
from app.core.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)
API_PREFIX = "/api/v1"
OPENAPI_URL = "/openapi.json"

API_DESCRIPTION = """
Food CV Microservice API.

Primary flow:
1. Upload a food image to `/api/v1/cv/analyze`.
2. Receive a `job_id`.
3. Poll `/api/v1/cv/jobs/{job_id}` until `status` is `done` or `failed`.

Authentication:
- Protected service endpoints require `Authorization: Bearer <API_SECRET_KEY>`.
- Set `AUTH_ENABLED=false` only for local API testing.
"""

TAGS_METADATA = [
    {
        "name": "Computer Vision",
        "description": "Food image analysis, async job polling, and service health checks.",
    },
    {
        "name": "Meal History",
        "description": "Authenticated user meal history lookup and semantic search.",
    },
    {
        "name": "admin",
        "description": "Admin-only nutrition cache and verified food management.",
    },
]


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("startup", env=settings.app_env, device=settings.device)
    yield
    logger.info("shutdown")


app = FastAPI(
    title="Food CV Microservice",
    description=API_DESCRIPTION,
    version="1.0.0",
    openapi_url=OPENAPI_URL if settings.api_docs_enabled else None,
    docs_url="/docs" if settings.api_docs_enabled else None,
    redoc_url=None,
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan,
)

if settings.api_docs_enabled:
    app.mount("/static", StaticFiles(directory="app/static"), name="static")

    @app.get("/redoc", include_in_schema=False)
    async def redoc_html():
        return get_redoc_html(
            openapi_url=OPENAPI_URL,
            title=f"{app.title} - ReDoc",
            redoc_js_url="/static/redoc.standalone.js",
            with_google_fonts=False,
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "Authorization"],
)

Instrumentator().instrument(app).expose(app)

app.include_router(cv_router, prefix=API_PREFIX)
app.include_router(history_router, prefix=API_PREFIX)
app.include_router(admin_router, prefix=API_PREFIX)


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    logger.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=not settings.is_production,
        workers=settings.app_workers,
    )
