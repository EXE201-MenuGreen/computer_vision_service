import binascii
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from celery.exceptions import Retry
from pydantic import ValidationError

from app.schemas.meal_scan_schemas import RawPreparedMealAnalysis
from app.services.inference_client import InferenceClientError
from app.services.worker import (
    TASK_ANALYZE_IMAGE,
    TASK_ANALYZE_PREPARED_MEAL,
    analyze_prepared_meal_task,
    celery_app,
    enqueue_prepared_meal_job,
)


def test_prepared_meal_task_is_registered_separately_from_legacy_task():
    assert TASK_ANALYZE_PREPARED_MEAL == "cv.analyze_prepared_meal"
    assert TASK_ANALYZE_IMAGE == "cv.analyze_image"
    assert TASK_ANALYZE_PREPARED_MEAL in celery_app.tasks
    assert TASK_ANALYZE_IMAGE in celery_app.tasks


def test_enqueue_serializes_image_as_hex_and_routes_to_configured_queue():
    sent = SimpleNamespace(id="meal-job")
    with patch.object(celery_app, "send_task", return_value=sent) as send_task:
        job_id = enqueue_prepared_meal_job(b"image-bytes", "meal.jpg", "image/jpeg")

    assert job_id == "meal-job"
    args = send_task.call_args.kwargs["args"]
    assert send_task.call_args.args[0] == TASK_ANALYZE_PREPARED_MEAL
    assert binascii.unhexlify(args[0]) == b"image-bytes"
    assert args[1:] == ["meal.jpg", "image/jpeg"]


def test_worker_returns_json_serializable_result_and_sets_queue_job_id():
    payload = {"job_id": "provider-job", "analysis_type": "prepared_meal", "value": 1}
    with patch(
        "app.services.worker.ai_circuit_breaker.can_execute", return_value=True
    ), patch(
        "app.services.worker.analyze_prepared_meal", new=AsyncMock(return_value=payload)
    ), patch(
        "app.services.worker.ai_circuit_breaker.record_success"
    ) as success:
        eager_result = analyze_prepared_meal_task.apply(
            args=[binascii.hexlify(b"image").decode(), "meal.jpg", "image/jpeg"],
            task_id="worker-job",
        )
        result = eager_result.get(propagate=True)

    json.dumps(result)
    assert result["job_id"] == "worker-job"
    success.assert_called_once_with()


def test_worker_retries_transient_provider_error():
    error = InferenceClientError("temporary", is_transient=True)
    retry_signal = Retry("retry")
    with patch(
        "app.services.worker.ai_circuit_breaker.can_execute", return_value=True
    ), patch(
        "app.services.worker.analyze_prepared_meal", new=AsyncMock(side_effect=error)
    ), patch.object(
        analyze_prepared_meal_task, "retry", side_effect=retry_signal
    ) as retry:
        with pytest.raises(Retry):
            analyze_prepared_meal_task.run("00", "meal.jpg", "image/jpeg")

    assert retry.call_args.kwargs["exc"] is error


def test_worker_does_not_retry_malformed_ai_payload():
    try:
        RawPreparedMealAnalysis.model_validate({"dish_name": "invalid"})
    except ValidationError as exc:
        malformed = exc

    with patch(
        "app.services.worker.ai_circuit_breaker.can_execute", return_value=True
    ), patch(
        "app.services.worker.analyze_prepared_meal", new=AsyncMock(side_effect=malformed)
    ), patch.object(analyze_prepared_meal_task, "retry") as retry:
        with pytest.raises(ValidationError):
            analyze_prepared_meal_task.run("00", "meal.jpg", "image/jpeg")
    retry.assert_not_called()


def test_worker_retries_when_shared_circuit_breaker_is_open():
    retry_signal = Retry("retry")
    with patch(
        "app.services.worker.ai_circuit_breaker.can_execute", return_value=False
    ), patch(
        "app.services.worker.analyze_prepared_meal", new=AsyncMock()
    ) as analyze, patch.object(
        analyze_prepared_meal_task, "retry", side_effect=retry_signal
    ) as retry:
        with pytest.raises(Retry):
            analyze_prepared_meal_task.run("00", "meal.jpg", "image/jpeg")
    assert isinstance(retry.call_args.kwargs["exc"], InferenceClientError)
    assert retry.call_args.kwargs["exc"].is_transient is True
    analyze.assert_not_awaited()
