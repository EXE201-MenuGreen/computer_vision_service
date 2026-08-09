from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.auth import require_api_key
from app.api.cv_router import _job_progress
from app.core.logging import get_logger
from app.schemas.cv_schemas import JobResponse
from app.schemas.meal_scan_schemas import (
    PreparedMealAnalysisResponse,
    PreparedMealJobStatusResponse,
)
from app.services.image_validator import prepare_image_for_inference, validate_and_load_image
from app.services.worker import enqueue_prepared_meal_job, get_job_result

logger = get_logger(__name__)
router = APIRouter(prefix="/cv", tags=["Prepared Meal Scan"])


def _build_status(job_id: str, result: dict[str, Any]) -> PreparedMealJobStatusResponse:
    status = result["status"]
    celery_state = result.get("celery_state")
    worker_active, message, steps = _job_progress(status, celery_state)
    if status != "done":
        # Celery failures/retries may contain provider exception text (including
        # request URLs or upstream payload fragments). Never expose that text to
        # API clients. Timeout text below is generated locally and is safe.
        safe_error = result.get("error") if result.get("error_is_safe") else None
        if status == "failed":
            safe_error = "Prepared-meal analysis failed. Please retry with a clearer food image."
        return PreparedMealJobStatusResponse(
            job_id=job_id,
            status=status,
            celery_state=celery_state,
            worker_active=worker_active,
            message=message,
            steps=steps,
            error=safe_error,
        )
    try:
        payload = PreparedMealAnalysisResponse.model_validate(result["result"])
        # The queue id is the public identity of this operation. Provider/cache
        # metadata must never make the nested result refer to another job.
        payload = payload.model_copy(update={"job_id": job_id})
    except Exception as exc:
        logger.error("prepared_meal_response_validation_failed", job_id=job_id, error_type=type(exc).__name__)
        # Celery succeeded, but an invalid API payload is still a failed result.
        # Rebuild the presentation state so message/steps match that outcome.
        worker_active, message, steps = _job_progress("failed", celery_state)
        return PreparedMealJobStatusResponse(
            job_id=job_id,
            status="failed",
            celery_state=celery_state,
            worker_active=worker_active,
            message=message,
            steps=steps,
            error="AI analysis returned incomplete data. Please retry with a clearer food image.",
        )
    return PreparedMealJobStatusResponse(
        job_id=job_id,
        status="done",
        celery_state=celery_state,
        worker_active=worker_active,
        message=message,
        steps=steps,
        result=payload,
    )


async def _enqueue(image: UploadFile) -> str:
    pil_image = await validate_and_load_image(image)
    return enqueue_prepared_meal_job(
        prepare_image_for_inference(pil_image), image.filename or "meal.jpg", "image/jpeg"
    )


@router.post("/analyze-meal", response_model=JobResponse, summary="Queue prepared-meal analysis")
async def analyze_meal_async(image: UploadFile = File(...), _: None = Depends(require_api_key)):
    job_id = await _enqueue(image)
    logger.info("prepared_meal_job_queued", job_id=job_id, analysis_type="prepared_meal")
    return JobResponse(
        job_id=job_id,
        message="Prepared-meal analysis queued. Poll /cv/meal-jobs/{job_id} for result.",
    )


@router.post(
    "/analyze-meal-sync",
    response_model=PreparedMealJobStatusResponse,
    summary="Synchronous prepared-meal analysis",
)
async def analyze_meal_sync(
    image: UploadFile = File(...),
    timeout_seconds: float = Form(90.0),
    _: None = Depends(require_api_key),
):
    job_id = await _enqueue(image)
    wait_timeout = timeout_seconds if timeout_seconds > 0 else 90.0
    deadline = asyncio.get_event_loop().time() + wait_timeout
    while True:
        result = get_job_result(job_id)
        if result.get("status") in {"done", "failed"}:
            break
        if asyncio.get_event_loop().time() >= deadline:
            result = {
                "status": "processing",
                "celery_state": result.get("celery_state"),
                "error": f"Timed out after {wait_timeout}s; client should keep polling.",
                "error_is_safe": True,
            }
            break
        await asyncio.sleep(0.5)
    return _build_status(job_id, result)


@router.get(
    "/meal-jobs/{job_id}",
    response_model=PreparedMealJobStatusResponse,
    summary="Get prepared-meal job result",
)
async def get_meal_job(job_id: str, _: None = Depends(require_api_key)):
    return _build_status(job_id, get_job_result(job_id))
