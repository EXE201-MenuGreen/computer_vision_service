import asyncio
import binascii
from fastapi import APIRouter, UploadFile, File, HTTPException, Request

from app.schemas.cv_schemas import AnalysisResult, JobResponse, JobStatusResponse, HealthResponse
from app.services.image_validator import validate_and_load_image
from app.services.nutrition_service import nutrition_service
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/cv", tags=["Computer Vision"])


def _get_pipeline(request: Request):
    """Retrieve the CVPipeline instance from app state."""
    return request.app.state.pipeline


# ── Health check ────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    pipeline = _get_pipeline(request)
    return HealthResponse(
        status="ok" if pipeline.is_ready else "loading",
        models_loaded=pipeline.is_ready,
        device=settings.device,
    )


# ── Sync analyze (< 5s, suitable for most cases) ────────────
@router.post("/analyze", response_model=AnalysisResult)
async def analyze_sync(request: Request, image: UploadFile = File(...)):
    """
    Receive image, run full CV pipeline, return nutrition result.
    Suitable for CPU inference or small images (< 5s total).
    """
    pipeline = _get_pipeline(request)

    if not pipeline.is_ready:
        raise HTTPException(503, "Models are still loading. Retry in a few seconds.")

    pil_image = await validate_and_load_image(image)

    # Run CV pipeline (detection → classification → depth → postprocess)
    ctx = pipeline.run(pil_image)

    # Nutrition lookup (Option B: outside pipeline)
    nutrition_breakdown = await nutrition_service.lookup_batch(ctx.food_items)
    total = nutrition_service.sum_macros(nutrition_breakdown)

    analysis_result = AnalysisResult(
        request_id=ctx.request_id,
        detected_foods=ctx.food_items,
        nutrition_breakdown=nutrition_breakdown,
        total_macros=total,
        processing_time_ms=ctx.processing_time_ms,
    )

    # Fire-and-forget meal history store (Option 3)
    if settings.meal_history_enabled:
        try:
            from app.api.deps import _decode_user_id
            from app.db import meal_history as meal_history_service
            token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            if token:
                uid = _decode_user_id(token)
                if uid:
                    asyncio.ensure_future(meal_history_service.store_meal(uid, analysis_result))
        except Exception:
            pass

    return analysis_result


# ── Async analyze (for slow GPU warm-up or large batches) ───
@router.post("/analyze/async", response_model=JobResponse)
async def analyze_async(request: Request, image: UploadFile = File(...)):
    """
    Queue the image for async Celery processing.
    Returns job_id immediately — poll /cv/jobs/{job_id} for result.
    """
    pil_image = await validate_and_load_image(image)

    # Re-encode to bytes for Celery serialization
    import io
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=90)
    hex_bytes = binascii.hexlify(buf.getvalue()).decode()

    from app.services.worker import analyze_image_task
    task = analyze_image_task.delay(hex_bytes, image.filename, image.content_type)

    logger.info("job_queued", job_id=task.id)
    return JobResponse(job_id=task.id)


# ── Poll job result ─────────────────────────────────────────
@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    from app.services.worker import celery_app
    task = celery_app.AsyncResult(job_id)

    if task.state == "PENDING":
        return JobStatusResponse(job_id=job_id, status="queued")
    if task.state == "STARTED":
        return JobStatusResponse(job_id=job_id, status="processing")
    if task.state == "SUCCESS":
        return JobStatusResponse(
            job_id=job_id,
            status="done",
            result=AnalysisResult(**task.result),
        )
    if task.state == "FAILURE":
        return JobStatusResponse(job_id=job_id, status="failed", error=str(task.info))

    return JobStatusResponse(job_id=job_id, status=task.state.lower())
