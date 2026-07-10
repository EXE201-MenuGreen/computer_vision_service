import io
import asyncio
from typing import Any, Dict, Optional

from fastapi import APIRouter, UploadFile, File, Form, Depends

from app.api.analyze_context import build_user_analysis_context
from app.api.auth import require_api_key
from app.schemas.cv_schemas import (
    AIInferenceResponse,
    HealthResponse,
    JobProgressStep,
    JobResponse,
    JobStatusResponse,
)
from app.services.image_validator import validate_and_load_image
from app.services.worker import TASK_HEALTH_CHECK, celery_app, enqueue_inference_job, get_job_result
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
    task = None
    try:
        task = celery_app.send_task(
            TASK_HEALTH_CHECK,
            queue=settings.celery_queue,
        )
        result = task.get(timeout=settings.celery_health_timeout_seconds)
        return isinstance(result, dict) and result.get("status") == "ok"
    except Exception as exc:
        logger.error("health_check_worker_failed", error=str(exc))
        return False
    finally:
        if task is not None:
            try:
                task.forget()
            except Exception:
                pass


def _prepare_image_for_inference(pil_image) -> bytes:
    """Resize large uploads and encode them as JPEG before sending to the worker."""
    max_dimension = settings.image_max_dimension_px
    if max_dimension > 0 and max(pil_image.size) > max_dimension:
        pil_image = pil_image.copy()
        pil_image.thumbnail((max_dimension, max_dimension))

    buf = io.BytesIO()
    pil_image.save(
        buf,
        format="JPEG",
        quality=settings.image_jpeg_quality,
        optimize=True,
    )
    return buf.getvalue()


def _job_progress(status: str, celery_state: str | None) -> tuple[bool, str, list[JobProgressStep]]:
    """Build a user-facing progress description from Celery state."""
    worker_active = celery_state in {"STARTED", "RETRY", "SUCCESS"}

    if status == "queued":
        message = "Job is queued. Waiting for a Celery worker to pick it up."
        steps = [
            JobProgressStep(name="queued", status="active", description="Task is stored in Redis queue."),
            JobProgressStep(name="worker", status="pending", description="Waiting for worker to receive the task."),
            JobProgressStep(name="ai_analysis", status="pending", description="Gemini analysis has not started yet."),
            JobProgressStep(name="result", status="pending", description="Result is not available yet."),
        ]
        return worker_active, message, steps

    if celery_state == "RETRY":
        message = "Worker hit an error and Celery is retrying the job."
        steps = [
            JobProgressStep(name="queued", status="done", description="Task was accepted from Redis queue."),
            JobProgressStep(name="worker", status="active", description="Worker is retrying after a processing error."),
            JobProgressStep(name="ai_analysis", status="active", description="Gemini or nutrition processing is being retried."),
            JobProgressStep(name="result", status="pending", description="Final result is not available yet."),
        ]
        return worker_active, message, steps

    if status == "processing":
        message = "Worker is processing the image with Gemini and nutrition enrichment."
        steps = [
            JobProgressStep(name="queued", status="done", description="Task was accepted from Redis queue."),
            JobProgressStep(name="worker", status="active", description="Celery worker has started this job."),
            JobProgressStep(name="ai_analysis", status="active", description="Gemini analysis and enrichment are running."),
            JobProgressStep(name="result", status="pending", description="Result is not available yet."),
        ]
        return worker_active, message, steps

    if status == "failed":
        message = "Worker failed to complete the job."
        steps = [
            JobProgressStep(name="queued", status="done", description="Task was accepted from Redis queue."),
            JobProgressStep(name="worker", status="failed", description="Worker returned a failure for this job."),
            JobProgressStep(name="ai_analysis", status="failed", description="Gemini or enrichment processing failed."),
            JobProgressStep(name="result", status="failed", description="No successful result is available."),
        ]
        return worker_active, message, steps

    message = "Job completed successfully."
    steps = [
        JobProgressStep(name="queued", status="done", description="Task was accepted from Redis queue."),
        JobProgressStep(name="worker", status="done", description="Worker completed processing."),
        JobProgressStep(name="ai_analysis", status="done", description="Gemini analysis and enrichment completed."),
        JobProgressStep(name="result", status="done", description="Result is available."),
    ]
    return worker_active, message, steps


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
async def health_deep(_: None = Depends(require_api_key)):
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
    """Queue async image analysis. Optional form fields personalize Gemini suggestions."""
    pil_image = await validate_and_load_image(image)
    image_bytes = _prepare_image_for_inference(pil_image)

    user_context = await build_user_analysis_context(
        user_id=user_id,
        dietary_preferences=dietary_preferences,
        avoid_foods=avoid_foods,
        recent_dishes=recent_dishes,
    )

    job_id = enqueue_inference_job(
        image_bytes,
        image.filename or "image.jpg",
        "image/jpeg",
        user_context_json=user_context.model_dump_json(),
    )
    logger.info("job_queued", job_id=job_id, user_id=user_id)
    return JobResponse(job_id=job_id)


@router.post(
    "/analyze-sync",
    response_model=JobStatusResponse,
    summary="Synchronous food image analysis",
    description=(
        "Uploads a food image and waits for the worker to finish inference. "
        "Returns the same JobStatusResponse as /api/v1/cv/jobs/{job_id}; "
        "the result is also persisted in Celery's backend, so a follow-up "
        "GET /api/v1/cv/jobs/{job_id} returns it immediately."
    ),
)
async def analyze_sync(
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
    timeout_seconds: Optional[float] = Form(
        90.0,
        description="Max seconds to wait for worker. Defaults to 90s.",
    ),
    _: None = Depends(require_api_key),
):
    """Run inference in the same request and return the final JobStatusResponse."""
    pil_image = await validate_and_load_image(image)
    image_bytes = _prepare_image_for_inference(pil_image)

    user_context = await build_user_analysis_context(
        user_id=user_id,
        dietary_preferences=dietary_preferences,
        avoid_foods=avoid_foods,
        recent_dishes=recent_dishes,
    )

    job_id = enqueue_inference_job(
        image_bytes,
        image.filename or "image.jpg",
        "image/jpeg",
        user_context_json=user_context.model_dump_json(),
    )
    logger.info("job_queued_sync", job_id=job_id, user_id=user_id, timeout=timeout_seconds)

    # Block until worker finishes (or timeout) so the response is final.
    # `get_job_result` reads from Celery's backend, so subsequent GET /jobs/{id}
    # calls will see the persisted result immediately.
    wait_timeout = timeout_seconds if timeout_seconds and timeout_seconds > 0 else 90.0
    deadline = asyncio.get_event_loop().time() + wait_timeout
    poll_interval = 0.5
    result: Dict[str, Any] = {"status": "processing", "celery_state": "STARTED"}

    while True:
        result = get_job_result(job_id)
        if result.get("status") in {"done", "failed"}:
            break
        if asyncio.get_event_loop().time() >= deadline:
            result = {
                "status": "processing",
                "celery_state": result.get("celery_state"),
                "error": f"Timed out after {wait_timeout}s; client should keep polling.",
            }
            break
        await asyncio.sleep(poll_interval)

    status = result["status"]
    celery_state = result.get("celery_state")
    worker_active, message, steps = _job_progress(status, celery_state)

    if status in {"queued", "processing"}:
        return JobStatusResponse(
            job_id=job_id,
            status=status,
            celery_state=celery_state,
            worker_active=worker_active,
            message=message,
            steps=steps,
            error=result.get("error"),
        )
    if status == "failed":
        return JobStatusResponse(
            job_id=job_id,
            status="failed",
            celery_state=celery_state,
            worker_active=worker_active,
            message=message,
            steps=steps,
            error=result.get("error"),
        )

    payload = result["result"]
    try:
        validated_result = AIInferenceResponse(**payload)
    except Exception as e:
        logger.error(
            "ai_response_validation_failed",
            job_id=job_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return JobStatusResponse(
            job_id=job_id,
            status="failed",
            celery_state=celery_state,
            worker_active=worker_active,
            message="AI response validation failed. The AI service returned incomplete data.",
            steps=steps,
            error=f"AI response validation failed: {type(e).__name__}. Please retry with a clearer food image.",
        )

    return JobStatusResponse(
        job_id=job_id,
        status="done",
        celery_state=celery_state,
        worker_active=worker_active,
        message=message,
        steps=steps,
        result=validated_result,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get analysis job result",
    description="Returns processing while the job is running, done with result when complete, or failed with error.",
)
async def get_job(job_id: str, _: None = Depends(require_api_key)):
    result = get_job_result(job_id)
    status = result["status"]
    celery_state = result.get("celery_state")
    worker_active, message, steps = _job_progress(status, celery_state)

    if status in {"queued", "processing"}:
        return JobStatusResponse(
            job_id=job_id,
            status=status,
            celery_state=celery_state,
            worker_active=worker_active,
            message=message,
            steps=steps,
            error=result.get("error"),
        )
    if status == "failed":
        return JobStatusResponse(
            job_id=job_id,
            status="failed",
            celery_state=celery_state,
            worker_active=worker_active,
            message=message,
            steps=steps,
            error=result.get("error"),
        )

    payload = result["result"]
    
    # Validate and handle Pydantic validation errors gracefully
    try:
        validated_result = AIInferenceResponse(**payload)
    except Exception as e:
        logger.error(
            "ai_response_validation_failed",
            job_id=job_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return JobStatusResponse(
            job_id=job_id,
            status="failed",
            celery_state=celery_state,
            worker_active=worker_active,
            message="AI response validation failed. The AI service returned incomplete data.",
            steps=steps,
            error=f"AI response validation failed: {type(e).__name__}. This indicates the AI model returned incomplete analysis data. Please retry with a clearer food image.",
        )
    
    return JobStatusResponse(
        job_id=job_id,
        status="done",
        celery_state=celery_state,
        worker_active=worker_active,
        message=message,
        steps=steps,
        result=validated_result,
    )
