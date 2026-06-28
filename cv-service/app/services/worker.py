"""
Celery worker and job utility wrappers for remote AI inference.

This module forwards images to the external AI API using the configured Bearer token
via a Celery task. Minimal job status checks use Celery's AsyncResult.
"""
from __future__ import annotations

import asyncio
import binascii
from typing import Any, Dict, Optional

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
    result_expires=3600,            # results live 1 hour in Redis
    worker_prefetch_multiplier=1,   # process one task at a time
    task_default_queue="cv",
)


@celery_app.task(bind=True, name="cv.analyze_image", max_retries=2)
def analyze_image_task(
    self,
    image_bytes_hex: str,
    filename: str,
    content_type: str,
    user_context_json: str = "{}",
) -> dict[str, Any]:
    """
    Celery task: receives image bytes (hex-encoded), forwards to the external AI API.
    Returns the parsed JSON response.
    """
    import json

    from app.schemas.cv_schemas import UserAnalysisContext
    from app.services.inference_client import analyze_image
    from app.services.response_enricher import enrich_ai_response

    image_bytes = binascii.unhexlify(image_bytes_hex)
    try:
        ctx_data = json.loads(user_context_json) if user_context_json else {}
        user_context = UserAnalysisContext(**ctx_data) if ctx_data else None
    except (json.JSONDecodeError, TypeError, ValueError):
        user_context = None

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            analyze_image(image_bytes, filename, content_type, user_context)
        )
        result = loop.run_until_complete(enrich_ai_response(result, user_context))
        return result
    except Exception as exc:
        logger.error("celery_task_failed", error=str(exc), task_id=self.request.id)
        raise self.retry(exc=exc, countdown=2)
    finally:
        loop.close()


def enqueue_inference_job(
    image_bytes: bytes,
    filename: str,
    content_type: str,
    user_context_json: str = "{}",
) -> str:
    """Create a background job for remote AI inference and return the job id."""
    image_bytes_hex = binascii.hexlify(image_bytes).decode("utf-8")
    task = celery_app.send_task(
        "cv.analyze_image",
        args=[image_bytes_hex, filename, content_type, user_context_json],
        queue="cv",
    )
    return task.id


def get_job_result(job_id: str) -> Optional[Dict[str, Any]]:
    """Query Celery's AsyncResult backend for the job result."""
    from celery.result import AsyncResult
    res = AsyncResult(job_id, app=celery_app)
    if res.state == "SUCCESS":
        return {"status": "done", "result": res.result}
    elif res.state == "FAILURE":
        return {"status": "failed", "error": str(res.result)}
    elif res.state in ("PENDING", "RECEIVED", "STARTED", "RETRY"):
        return None  # Represents processing/queued in cv_router
    return None
