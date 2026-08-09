import io
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Header, HTTPException, Request, status
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client():
    from app.api.auth import require_api_key
    from app.main import app

    async def auth(request: Request, authorization: str = Header(default="", alias="Authorization")):
        if authorization != "Bearer test-key":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    app.dependency_overrides[require_api_key] = auth
    with TestClient(app, raise_server_exceptions=False) as value:
        yield value
    app.dependency_overrides.clear()


@pytest.fixture
def image_bytes():
    image = Image.new("RGB", (20, 20), "white")
    stream = io.BytesIO()
    image.save(stream, "JPEG")
    return stream.getvalue()


@pytest.fixture
def done_payload():
    return {
        "job_id": "ai-job", "request_id": "req", "status": "done",
        "analysis_type": "prepared_meal", "dish_name": "Cơm gà",
        "dish_name_key": "com_ga", "dish_confidence": 0.9,
        "estimated_total_grams": 200,
        "ingredients": [{
            "ingredient_id": "i1", "name": "Ức gà", "name_key": "uc_ga",
            "estimated_grams": 200, "detection_confidence": 0.8,
            "nutrition": {"macros": {"calories_kcal": 330, "protein_g": 62, "carbs_g": 0, "fat_g": 7, "fiber_g": 0}, "data_source": "usda", "confidence": 0.75},
        }],
        "total_macros": {"calories_kcal": 330, "protein_g": 62, "carbs_g": 0, "fat_g": 7, "fiber_g": 0},
        "estimation_note": "Giá trị ước tính.",
    }


def test_async_requires_auth(client, image_bytes):
    response = client.post("/api/v1/cv/analyze-meal", files={"image": ("x.jpg", image_bytes, "image/jpeg")})
    assert response.status_code == 401


def test_async_enqueues_separate_task(client, image_bytes):
    with patch("app.api.meal_scan_router.enqueue_prepared_meal_job", return_value="meal-job"):
        response = client.post("/api/v1/cv/analyze-meal", headers={"Authorization": "Bearer test-key"}, files={"image": ("x.jpg", image_bytes, "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["job_id"] == "meal-job"


def test_async_rejects_broken_image(client):
    response = client.post("/api/v1/cv/analyze-meal", headers={"Authorization": "Bearer test-key"}, files={"image": ("x.jpg", b"broken", "image/jpeg")})
    assert response.status_code == 400


def test_async_rejects_unsupported_mime_type(client, image_bytes):
    response = client.post(
        "/api/v1/cv/analyze-meal",
        headers={"Authorization": "Bearer test-key"},
        files={"image": ("x.txt", image_bytes, "text/plain")},
    )
    assert response.status_code == 415


def test_poll_done_validates_prepared_contract(client, done_payload):
    with patch("app.api.meal_scan_router.get_job_result", return_value={"status": "done", "celery_state": "SUCCESS", "result": done_payload}):
        response = client.get("/api/v1/cv/meal-jobs/meal-job", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    assert response.json()["result"]["analysis_type"] == "prepared_meal"
    assert response.json()["result"]["job_id"] == "meal-job"


def test_poll_invalid_payload_returns_safe_failure(client):
    with patch("app.api.meal_scan_router.get_job_result", return_value={"status": "done", "celery_state": "SUCCESS", "result": {"secret": "raw"}}):
        response = client.get("/api/v1/cv/meal-jobs/meal-job", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert "raw" not in response.text
    assert "failed" in response.json()["message"].lower()
    assert all(step["status"] != "done" for step in response.json()["steps"][1:])


@pytest.mark.parametrize(
    ("job_result", "expected_status", "expected_worker_status"),
    [
        ({"status": "queued", "celery_state": "PENDING"}, "queued", "pending"),
        ({"status": "processing", "celery_state": "STARTED"}, "processing", "active"),
        ({"status": "processing", "celery_state": "RETRY"}, "processing", "active"),
    ],
)
def test_poll_reports_non_terminal_states(
    client, job_result, expected_status, expected_worker_status
):
    with patch("app.api.meal_scan_router.get_job_result", return_value=job_result):
        response = client.get(
            "/api/v1/cv/meal-jobs/meal-job",
            headers={"Authorization": "Bearer test-key"},
        )
    assert response.status_code == 200
    assert response.json()["status"] == expected_status
    assert response.json()["steps"][1]["status"] == expected_worker_status


def test_poll_failure_does_not_expose_provider_error(client):
    provider_error = "request failed: https://provider.test?key=SUPER-SECRET"
    with patch(
        "app.api.meal_scan_router.get_job_result",
        return_value={"status": "failed", "celery_state": "FAILURE", "error": provider_error},
    ):
        response = client.get(
            "/api/v1/cv/meal-jobs/meal-job",
            headers={"Authorization": "Bearer test-key"},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert "SUPER-SECRET" not in response.text
    assert "provider.test" not in response.text


def test_sync_success(client, image_bytes, done_payload):
    with patch("app.api.meal_scan_router.enqueue_prepared_meal_job", return_value="meal-job"), patch("app.api.meal_scan_router.get_job_result", return_value={"status": "done", "celery_state": "SUCCESS", "result": done_payload}):
        response = client.post("/api/v1/cv/analyze-meal-sync", headers={"Authorization": "Bearer test-key"}, files={"image": ("x.jpg", image_bytes, "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["status"] == "done"


def test_sync_timeout_returns_processing_and_safe_polling_message(client, image_bytes):
    times = iter([0.0, 1.0])
    fake_loop = type("FakeLoop", (), {"time": lambda self: next(times)})()
    with patch(
        "app.api.meal_scan_router.enqueue_prepared_meal_job", return_value="meal-job"
    ), patch(
        "app.api.meal_scan_router.get_job_result",
        return_value={"status": "queued", "celery_state": "PENDING"},
    ), patch(
        "app.api.meal_scan_router.asyncio.get_event_loop", return_value=fake_loop
    ), patch(
        "app.api.meal_scan_router.asyncio.sleep", new=AsyncMock()
    ):
        response = client.post(
            "/api/v1/cv/analyze-meal-sync",
            headers={"Authorization": "Bearer test-key"},
            files={"image": ("x.jpg", image_bytes, "image/jpeg")},
            data={"timeout_seconds": "0.1"},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "processing"
    assert "keep polling" in response.json()["error"]
