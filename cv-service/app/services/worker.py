"""
Celery worker for async CV inference.
Use when inference time exceeds acceptable HTTP timeout (~5s).

Run with:
  celery -A app.services.worker worker --loglevel=info --concurrency=2
"""
import asyncio
from celery import Celery
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

celery_app = Celery(
    "cv_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,   # results live 1 hour in Redis
    worker_prefetch_multiplier=1,   # process one task at a time (GPU)
)

# ── Build a pipeline for the worker process ─────────────────
_pipeline = None


def _get_pipeline():
    """Lazy-build the pipeline on first task execution."""
    global _pipeline
    if _pipeline is None:
        # Import stages to trigger auto-registration
        import app.stages  # noqa: F401
        from app.pipeline.pipeline_factory import PipelineFactory

        _pipeline = PipelineFactory.build(settings.pipeline_config)
        _pipeline.load_all()
    return _pipeline


@celery_app.task(bind=True, name="cv.analyze_image", max_retries=2)
def analyze_image_task(self, image_bytes_hex: str, filename: str, content_type: str):
    """
    Celery task: receives image bytes (hex-encoded), runs full CV pipeline.
    Returns serialized AnalysisResult dict.
    """
    from PIL import Image
    import io
    import binascii
    from app.services.nutrition_service import nutrition_service
    from app.schemas.cv_schemas import AnalysisResult

    try:
        raw = binascii.unhexlify(image_bytes_hex)
        image = Image.open(io.BytesIO(raw)).convert("RGB")

        # Run CV pipeline
        pipeline = _get_pipeline()
        ctx = pipeline.run(image)

        # Nutrition lookup (Option B: outside pipeline)
        loop = asyncio.new_event_loop()
        nutrition_breakdown = loop.run_until_complete(
            nutrition_service.lookup_batch(ctx.food_items)
        )
        total = nutrition_service.sum_macros(nutrition_breakdown)
        loop.close()

        result = AnalysisResult(
            request_id=ctx.request_id,
            detected_foods=ctx.food_items,
            nutrition_breakdown=nutrition_breakdown,
            total_macros=total,
            processing_time_ms=ctx.processing_time_ms,
        )
        return result.model_dump()

    except Exception as exc:
        logger.error("celery_task_failed", error=str(exc), task_id=self.request.id)
        raise self.retry(exc=exc, countdown=2)
