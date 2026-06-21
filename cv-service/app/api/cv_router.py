import io
import asyncio
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, Depends

from app.api.analyze_context import build_user_analysis_context
from app.api.auth import require_api_key
from app.schemas.cv_schemas import JobResponse, JobStatusResponse, HealthResponse, AIInferenceResponse
from app.services.image_validator import validate_and_load_image
from app.services.response_enricher import enrich_ai_response
from app.services.worker import celery_app, enqueue_inference_job, get_job_result
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/cv", tags=["Computer Vision"])


def _is_ai_configured() -> bool:
    provider = settings.ai_provider.lower()
    if provider == "gemini":
        return bool(settings.gemini_api_key)
    if provider == "remote_api":
        return bool(settings.ai_api_base_url and settings.ai_api_key)
    return False


def _is_gemini_configured() -> bool:
    return bool(settings.gemini_api_key)


def _ping_celery_workers() -> bool:
    try:
        responses = celery_app.control.inspect(timeout=1).ping()
        return bool(responses)
    except Exception as exc:
        logger.error("health_check_worker_failed", error=str(exc))
        return False


@router.get(
    "/health",
    summary="API liveness check",
    description="Checks only whether the FastAPI process is alive. Does not touch Redis or Celery.",
)
async def health():
    """Lightweight API liveness check. Does not touch Redis or Celery."""
    return {
        "status": "ok",
        "service": "cv-service",
        "version": "1.0.0",
    }


@router.get(
    "/health/deep",
    response_model=HealthResponse,
    summary="Deep dependency health check",
    description="Manual readiness check for Redis, Celery worker, and AI provider configuration.",
)
async def health_deep():
    """Manual dependency readiness check. Touches Redis and Celery worker."""
    ai_configured = _is_ai_configured()
    gemini_configured = _is_gemini_configured()

    redis_ok = True
    try:
        from redis.asyncio import Redis
        r = Redis.from_url(settings.redis_url, socket_connect_timeout=2)
        await r.ping()
        await r.close()
    except Exception as exc:
        logger.error("health_check_redis_failed", error=str(exc))
        redis_ok = False

    worker_ok = await asyncio.to_thread(_ping_celery_workers) if redis_ok else False
    status_str = "ok" if (redis_ok and worker_ok and ai_configured) else "unhealthy"

    return HealthResponse(
        status=status_str,
        models_loaded=ai_configured,
        redis=redis_ok,
        worker=worker_ok,
        gemini_configured=gemini_configured,
        device=settings.device,
    )


@router.post(
    "/analyze",
    response_model=JobResponse,
    summary="Queue food image analysis",
    description=(
        "Uploads a food image and optional personalization context as multipart/form-data. "
        "Returns a job_id immediately; poll /api/v1/cv/jobs/{job_id} for the result."
    ),
)
async def analyze_async(
    image: UploadFile = File(...),
    user_id: Optional[str] = Form(None, description="User UUID for personalization"),
    dietary_preferences: Optional[str] = Form(
        None,
        description='JSON array, e.g. ["high_protein","low_carb"]',
    ),
    avoid_foods: Optional[str] = Form(
        None,
        description='JSON array of foods to avoid, e.g. ["đồ chiên"]',
    ),
    recent_dishes: Optional[str] = Form(
        None,
        description='JSON array of recent dish/ingredient names; auto-loaded from DB if user_id set',
    ),
    _: None = Depends(require_api_key),
):
    """
    Queue async image analysis. Optional form fields personalize Gemini suggestions.
    """
    pil_image = await validate_and_load_image(image)
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=90)
    image_bytes = buf.getvalue()

    user_context = await build_user_analysis_context(
        user_id=user_id,
        dietary_preferences=dietary_preferences,
        avoid_foods=avoid_foods,
        recent_dishes=recent_dishes,
    )

    job_id = enqueue_inference_job(
        image_bytes,
        image.filename or "image.jpg",
        image.content_type or "image/jpeg",
        user_context_json=user_context.model_dump_json(),
    )
    logger.info("job_queued", job_id=job_id, user_id=user_id)
    return JobResponse(job_id=job_id)


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get analysis job result",
    description="Returns processing while the job is running, done with result when complete, or failed with error.",
)
async def get_job(job_id: str, _: None = Depends(require_api_key)):
    result = get_job_result(job_id)
    if result is None:
        return JobStatusResponse(job_id=job_id, status="processing")
    if result.get("status") == "failed":
        return JobStatusResponse(job_id=job_id, status="failed", error=result.get("error"))
    payload = result.get("result")
    if payload is None:
        return JobStatusResponse(job_id=job_id, status="processing")

    if (
        settings.nutrition_enrichment_enabled
        and payload.get("status") == "done"
        and payload.get("nutrition_breakdown") is None
        and payload.get("nguyen_lieu_tho_quet_duoc")
    ):
        payload = await enrich_ai_response(payload)

    return JobStatusResponse(job_id=job_id, status="done", result=AIInferenceResponse(**payload))
