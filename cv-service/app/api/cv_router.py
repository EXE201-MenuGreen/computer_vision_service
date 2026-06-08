import io
from fastapi import APIRouter, UploadFile, File, Depends

from app.api.auth import require_api_key
from app.schemas.cv_schemas import JobResponse, JobStatusResponse, HealthResponse, AIInferenceResponse
from app.services.image_validator import validate_and_load_image
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/cv", tags=["Computer Vision"])


@router.get("/health", response_model=HealthResponse)
async def health():
    # 1. Check AI Provider configuration readiness
    ai_configured = False
    if settings.ai_provider == "mock":
        ai_configured = True
    elif settings.ai_provider == "gemini":
        ai_configured = bool(settings.gemini_api_key)
    elif settings.ai_provider == "remote_api":
        ai_configured = bool(settings.ai_api_base_url and settings.ai_api_key)

    # 2. Check Redis connection health (critical for Celery job queues)
    redis_ok = True
    try:
        from redis.asyncio import Redis
        r = Redis.from_url(settings.redis_url, socket_connect_timeout=2)
        await r.ping()
        await r.close()
    except Exception as exc:
        logger.error("health_check_redis_failed", error=str(exc))
        redis_ok = False

    status_str = "ok" if (redis_ok and ai_configured) else "unhealthy"

    return HealthResponse(
        status=status_str,
        models_loaded=ai_configured,
        device=settings.device,
    )


@router.post("/analyze", response_model=JobResponse)
async def analyze_async(image: UploadFile = File(...), _: None = Depends(require_api_key)):
    """
    Receive image from backend, forward to external AI API using Bearer key,
    and return a queued job id for async result retrieval.
    """
    pil_image = await validate_and_load_image(image)
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=90)
    image_bytes = buf.getvalue()

    from app.services.worker import enqueue_inference_job
    job_id = enqueue_inference_job(image_bytes, image.filename or "image.jpg", image.content_type or "image/jpeg")
    logger.info("job_queued", job_id=job_id)
    return JobResponse(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str, _: None = Depends(require_api_key)):
    from app.services.worker import get_job_result
    result = get_job_result(job_id)
    if result is None:
        return JobStatusResponse(job_id=job_id, status="processing")
    if result.get("status") == "failed":
        return JobStatusResponse(job_id=job_id, status="failed", error=result.get("error"))
    payload = result.get("result")
    if payload is None:
        return JobStatusResponse(job_id=job_id, status="processing")
    return JobStatusResponse(job_id=job_id, status="done", result=AIInferenceResponse(**payload))
