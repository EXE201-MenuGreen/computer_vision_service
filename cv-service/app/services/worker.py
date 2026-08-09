"""
Celery worker and job utility wrappers for remote AI inference.

This module forwards images to the external AI API using the configured Bearer token
via a Celery task. Minimal job status checks use Celery's AsyncResult.
"""
from __future__ import annotations

import asyncio
import binascii
import json
from typing import Any, Dict, Optional

from celery import Celery
import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.cv_schemas import UserAnalysisContext
from app.services.inference_client import InferenceClientError, analyze_image
from app.services.prepared_meal_service import analyze_prepared_meal
from app.services.response_utils import normalize_ai_response, ai_circuit_breaker

logger = get_logger(__name__)
TASK_ANALYZE_IMAGE = "cv.analyze_image"
TASK_ANALYZE_PREPARED_MEAL = "cv.analyze_prepared_meal"
TASK_HEALTH_CHECK = "cv.health_check"

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
    result_expires=3600,
    worker_prefetch_multiplier=1,
    task_default_queue=settings.celery_queue,
)


# ── Transient Error Detection ────────────────────────────────────────────

def _is_transient_error(exc: Exception) -> bool:
    """Determine if an error is transient and should be retried."""
    transient_exceptions = (
        httpx.TimeoutException,
        httpx.ConnectError,
        httpx.RemoteProtocolError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
    )
    if isinstance(exc, transient_exceptions):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    if isinstance(exc, InferenceClientError) and exc.is_transient:
        return True
    return False


# ── Celery Tasks ─────────────────────────────────────────────────────────

@celery_app.task(name=TASK_HEALTH_CHECK)
def health_check_task() -> dict[str, str]:
    """Lightweight task used by /health/deep to verify real queue execution."""
    return {"status": "ok"}


@celery_app.task(
    bind=True,
    name=TASK_ANALYZE_IMAGE,
    max_retries=3,
    default_retry_delay=5,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
)
def analyze_image_task(
    self,
    image_bytes_hex: str,
    filename: str,
    content_type: str,
    user_context_json: str = "{}",
) -> dict[str, Any]:
    """Celery task: forwards image bytes to external AI API and returns normalized result."""
    job_id = self.request.id
    image_bytes = binascii.unhexlify(image_bytes_hex)

    try:
        ctx_data = json.loads(user_context_json) if user_context_json else {}
        user_context = UserAnalysisContext(**ctx_data) if ctx_data else None
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("user_context_parse_failed", error=str(e), job_id=job_id)
        user_context = None

    if not ai_circuit_breaker.can_execute():
        logger.warning("circuit_breaker_rejected", job_id=job_id)
        raise InferenceClientError(
            "AI service temporarily unavailable (circuit breaker open)",
            is_transient=True,
        )

    try:
        result = asyncio.run(
            _run_analyze_image(image_bytes, filename, content_type, user_context)
        )
        result = normalize_ai_response(result)
        ai_circuit_breaker.record_success()
        logger.info("analyze_image_success", job_id=job_id)
        return result

    except InferenceClientError as exc:
        ai_circuit_breaker.record_failure()
        if _is_transient_error(exc):
            logger.warning("transient_error_retry", error=str(exc), retry_count=self.request.retries, job_id=job_id)
            raise self.retry(exc=exc)
        logger.error("non_transient_error", error=str(exc), job_id=job_id)
        raise

    except httpx.HTTPStatusError as exc:
        ai_circuit_breaker.record_failure()
        if _is_transient_error(exc):
            logger.warning("http_error_retry", status_code=exc.response.status_code, job_id=job_id)
            raise self.retry(exc=exc)
        logger.error("http_error_non_transient", status_code=exc.response.status_code, job_id=job_id)
        raise

    except Exception as exc:
        ai_circuit_breaker.record_failure()
        if _is_transient_error(exc):
            logger.warning("unknown_transient_error_retry", error=str(type(exc).__name__), job_id=job_id)
            raise self.retry(exc=exc)
        logger.error("unknown_non_transient_error", error=str(type(exc).__name__), job_id=job_id)
        raise


@celery_app.task(bind=True, name=TASK_ANALYZE_PREPARED_MEAL, max_retries=3)
def analyze_prepared_meal_task(
    self,
    image_bytes_hex: str,
    filename: str,
    content_type: str,
) -> dict[str, Any]:
    """Analyze one prepared meal without invoking the ingredient-scan normalizer."""
    job_id = self.request.id
    image_bytes = binascii.unhexlify(image_bytes_hex)
    try:
        if not ai_circuit_breaker.can_execute():
            logger.warning(
                "prepared_meal_circuit_breaker_rejected",
                job_id=job_id,
                analysis_type="prepared_meal",
            )
            raise InferenceClientError(
                "AI service temporarily unavailable (circuit breaker open)",
                is_transient=True,
            )
        result = asyncio.run(analyze_prepared_meal(image_bytes, filename, content_type))
        result["job_id"] = job_id
        ai_circuit_breaker.record_success()
        logger.info("analyze_prepared_meal_success", job_id=job_id, analysis_type="prepared_meal")
        return result
    except Exception as exc:
        ai_circuit_breaker.record_failure()
        if _is_transient_error(exc):
            logger.warning(
                "prepared_meal_transient_error_retry",
                error=type(exc).__name__,
                job_id=job_id,
                analysis_type="prepared_meal",
            )
            raise self.retry(exc=exc, countdown=min(5 * (2 ** self.request.retries), 60))
        logger.error(
            "prepared_meal_non_transient_error",
            error=type(exc).__name__,
            job_id=job_id,
            analysis_type="prepared_meal",
        )
        raise

async def _run_analyze_image(
    image_bytes: bytes,
    filename: str,
    content_type: str,
    user_context: Optional[UserAnalysisContext],
) -> dict[str, Any]:
    """Run analyze_image in async context."""
    return await analyze_image(image_bytes, filename, content_type, user_context)


# ── Job Management ────────────────────────────────────────────────────────

def enqueue_inference_job(
    image_bytes: bytes,
    filename: str,
    content_type: str,
    user_context_json: str = "{}",
) -> str:
    """Create a background job for remote AI inference and return the job id."""
    image_bytes_hex = binascii.hexlify(image_bytes).decode("utf-8")
    task = celery_app.send_task(
        TASK_ANALYZE_IMAGE,
        args=[image_bytes_hex, filename, content_type, user_context_json],
        queue=settings.celery_queue,
    )
    return task.id


def enqueue_prepared_meal_job(image_bytes: bytes, filename: str, content_type: str) -> str:
    """Create a background prepared-meal analysis job and return its id."""
    task = celery_app.send_task(
        TASK_ANALYZE_PREPARED_MEAL,
        args=[binascii.hexlify(image_bytes).decode("utf-8"), filename, content_type],
        queue=settings.celery_queue,
    )
    return task.id


def get_job_result(job_id: str) -> Dict[str, Any]:
    """Query Celery's AsyncResult backend for the job result."""
    from celery.result import AsyncResult
    res = AsyncResult(job_id, app=celery_app)
    if res.state == "SUCCESS":
        return {"status": "done", "celery_state": res.state, "result": res.result}
    elif res.state == "FAILURE":
        return {"status": "failed", "celery_state": res.state, "error": str(res.result)}
    elif res.state in ("PENDING", "RECEIVED"):
        return {"status": "queued", "celery_state": res.state}
    elif res.state in ("STARTED", "RETRY"):
        error = str(res.result) if res.state == "RETRY" and res.result else None
        return {"status": "processing", "celery_state": res.state, "error": error}
    return {"status": "processing", "celery_state": res.state}


def get_circuit_breaker_status() -> dict[str, Any]:
    """Get current circuit breaker status for health check."""
    return ai_circuit_breaker.get_status()
