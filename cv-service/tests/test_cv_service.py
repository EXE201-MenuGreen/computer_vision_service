"""
Tests for CV microservice API-based flow.
Run: pytest tests/test_cv_service.py -v
"""
from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def fake_rgb_image() -> bytes:
    img = Image.new("RGB", (320, 240), color=(120, 80, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def client():
    from app.main import app
    from app.api.auth import require_api_key
    from fastapi import Header, HTTPException, status

    async def mock_require_api_key(
        request: Request,
        authorization: str = Header(default="", alias="Authorization"),
    ):
        prefix = "Bearer "
        if not authorization.startswith(prefix):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token.")
        token = authorization.removeprefix(prefix).strip()
        if token != "test-key":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")

    app.dependency_overrides[require_api_key] = mock_require_api_key
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def mock_api_response():
    return {
        "job_id": "job_123",
        "request_id": "req_123",
        "api_version": "v1",
        "status": "done",
        "processing_time_ms": 1234.5,
        "luong_tin_cay_chung": "92%",
        "nguyen_lieu_tho_quet_duoc": [
            {
                "id_nguyen_lieu": "raw_01",
                "ten_nguyen_lieu": "Ức gà tươi sống",
                "ten_nguyen_lieu_ky_thuat": "uc_ga",
                "khoi_luong_uoc_tinh_g": 250,
                "do_chinh_xac_uoc_tinh": "95%",
            }
        ],
        "danh_sach_mon_an_goi_y": [
            {
                "id_mon_an_goi_y": "rec_01",
                "ten_mon_an": "Ức gà áp chảo xào bông cải xanh",
                "ten_mon_an_ky_thuat": "uc_ga_ap_chao_xao_bong_cai_xanh",
                "mo_ta_ngan": "Món ăn giàu protein, ít chất béo.",
                "do_kha_thi": "95%",
                "confidence": 0.95,
                "nguyen_lieu_su_dung": [
                    {
                        "ten": "Ức gà",
                        "ten_ky_thuat": "uc_ga",
                        "khoi_luong_g": 250,
                    }
                ],
                "thong_tin_dinh_duong_mon_an": {
                    "tong_calories": 320,
                    "protein_g": 55.2,
                    "carbs_g": 11.5,
                    "fat_g": 7.3,
                    "fiber_g": 3.8,
                },
            }
        ],
    }


@pytest.fixture
def mock_worker():
    with patch("app.api.cv_router.enqueue_inference_job", return_value="job_123"):
        yield


# ── Health endpoint ─────────────────────────────────────────
def test_health_liveness_does_not_touch_dependencies(client):
    with patch("redis.asyncio.Redis.from_url") as redis_from_url, \
         patch("app.api.cv_router._ping_celery_workers") as worker_ping:
        r = client.get("/api/v1/cv/health")

    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "service": "cv-service",
        "version": "1.0.0",
    }
    redis_from_url.assert_not_called()
    worker_ping.assert_not_called()


def test_health_deep_ok(client):
    class MockRedis:
        @classmethod
        def from_url(cls, *args, **kwargs):
            return cls()
        async def ping(self):
            return True
        async def close(self):
            pass

    with patch("redis.asyncio.Redis.from_url", MockRedis.from_url), \
         patch("app.api.cv_router._ping_celery_workers", return_value=True), \
         patch("app.api.cv_router.settings.ai_provider", "gemini"), \
         patch("app.api.cv_router.settings.gemini_api_key", "test-gemini-key"):
        r = client.get(
            "/api/v1/cv/health/deep",
            headers={"Authorization": "Bearer test-key"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["models_loaded"] is True
        assert data["redis"] is True
        assert data["worker"] is True
        assert data["gemini_configured"] is True


def test_health_deep_unhealthy_when_worker_offline(client):
    class MockRedis:
        @classmethod
        def from_url(cls, *args, **kwargs):
            return cls()
        async def ping(self):
            return True
        async def close(self):
            pass

    with patch("redis.asyncio.Redis.from_url", MockRedis.from_url), \
         patch("app.api.cv_router._ping_celery_workers", return_value=False), \
         patch("app.api.cv_router.settings.ai_provider", "gemini"), \
         patch("app.api.cv_router.settings.gemini_api_key", "test-gemini-key"):
        r = client.get(
            "/api/v1/cv/health/deep",
            headers={"Authorization": "Bearer test-key"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data == {
            "status": "unhealthy",
            "models_loaded": True,
            "redis": True,
            "worker": False,
            "gemini_configured": True,
            "device": "cpu",
            "version": "1.0.0",
        }


def test_ping_celery_workers_fails_when_health_task_does_not_return_ok():
    from unittest.mock import MagicMock

    from app.api.cv_router import _ping_celery_workers

    task = MagicMock()
    task.get.return_value = {"status": "unexpected"}

    with patch("app.api.cv_router.celery_app.send_task", return_value=task):
        assert _ping_celery_workers() is False
    task.forget.assert_called_once()


def test_ping_celery_workers_accepts_real_health_task_result():
    from unittest.mock import MagicMock

    from app.api.cv_router import _ping_celery_workers

    task = MagicMock()
    task.get.return_value = {"status": "ok"}

    with patch("app.api.cv_router.celery_app.send_task", return_value=task):
        assert _ping_celery_workers() is True
    task.forget.assert_called_once()


# ── Auth guard ──────────────────────────────────────────────
def test_analyze_rejects_missing_bearer(client, fake_rgb_image):
    r = client.post(
        "/api/v1/cv/analyze",
        files={"image": ("food.jpg", fake_rgb_image, "image/jpeg")},
    )
    assert r.status_code == 401


# ── Async analyze ───────────────────────────────────────────
def test_analyze_returns_job_id(client, fake_rgb_image, mock_worker):
    r = client.post(
        "/api/v1/cv/analyze",
        headers={"Authorization": "Bearer test-key"},
        files={"image": ("food.jpg", fake_rgb_image, "image/jpeg")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["job_id"] == "job_123"
    assert data["status"] == "queued"


# ── Job polling ─────────────────────────────────────────────
def test_analyze_passes_personalization_context_to_worker(client, fake_rgb_image):
    captured = {}

    def fake_enqueue(*args, **kwargs):
        captured["content_type"] = args[2]
        captured["user_context_json"] = kwargs["user_context_json"]
        return "job_123"

    with patch("app.api.cv_router.enqueue_inference_job", side_effect=fake_enqueue):
        r = client.post(
            "/api/v1/cv/analyze",
            headers={"Authorization": "Bearer test-key"},
            files={"image": ("food.jpg", fake_rgb_image, "image/jpeg")},
            data={
                "user_id": "user-1",
                "dietary_preferences": '["high_protein","low_carb"]',
                "avoid_foods": '["do_chien"]',
                "recent_dishes": '["pho_bo","bun_ca"]',
            },
        )

    assert r.status_code == 200
    assert captured["content_type"] == "image/jpeg"
    context = json.loads(captured["user_context_json"])
    assert context["user_id"] == "user-1"
    assert context["dietary_preferences"] == ["high_protein", "low_carb"]
    assert context["avoid_foods"] == ["do_chien"]
    assert context["recent_dishes"] == ["pho_bo", "bun_ca"]


def test_job_poll_processing(client):
    with patch(
        "app.api.cv_router.get_job_result",
        return_value={"status": "processing", "celery_state": "STARTED"},
    ):
        r = client.get(
            "/api/v1/cv/jobs/job_123",
            headers={"Authorization": "Bearer test-key"},
        )
    assert r.status_code == 200
    assert r.json()["status"] == "processing"
    assert r.json()["celery_state"] == "STARTED"
    assert r.json()["worker_active"] is True
    assert "Worker is processing" in r.json()["message"]
    assert r.json()["steps"][1]["name"] == "worker"
    assert r.json()["steps"][1]["status"] == "active"


def test_job_poll_queued(client):
    with patch(
        "app.api.cv_router.get_job_result",
        return_value={"status": "queued", "celery_state": "PENDING"},
    ):
        r = client.get(
            "/api/v1/cv/jobs/job_123",
            headers={"Authorization": "Bearer test-key"},
        )
    assert r.status_code == 200
    assert r.json()["status"] == "queued"
    assert r.json()["celery_state"] == "PENDING"
    assert r.json()["worker_active"] is False
    assert "Waiting for a Celery worker" in r.json()["message"]
    assert r.json()["steps"][1]["status"] == "pending"


def test_job_poll_done(client, mock_api_response):
    with patch(
        "app.api.cv_router.get_job_result",
        return_value={"status": "done", "celery_state": "SUCCESS", "result": mock_api_response},
    ), \
         patch("app.services.response_enricher.enrich_ai_response") as enrich:
        r = client.get(
            "/api/v1/cv/jobs/job_123",
            headers={"Authorization": "Bearer test-key"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "done"
    assert data["celery_state"] == "SUCCESS"
    assert data["worker_active"] is True
    assert data["steps"][-1]["status"] == "done"
    assert data["result"]["status"] == "done"
    assert data["result"]["nguyen_lieu_tho_quet_duoc"][0]["ten_nguyen_lieu_ky_thuat"] == "uc_ga"
    enrich.assert_not_called()


# ── Inference client ────────────────────────────────────────
@pytest.mark.asyncio
async def test_inference_client_sends_bearer_and_parses_json():
    from app.services.inference_client import analyze_image

    class FakeResp:
        status_code = 200
        text = "{}"

        def raise_for_status(self):
            return None

        def json(self):
            return {"job_id": "job_1", "status": "done"}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.captured_headers = kwargs.get("headers", {})
            self.captured_files = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, files=None, data=None):
            self.captured_files = files
            return FakeResp()

    with patch("app.services.inference_client.settings.ai_provider", "remote_api"), \
         patch("app.services.inference_client.settings.ai_api_base_url", "http://ai-api"), \
         patch("app.services.inference_client.settings.ai_api_key", "secret"), \
         patch("httpx.AsyncClient", FakeClient):
        result = await analyze_image(b"abc", "food.jpg", "image/jpeg", None)

    assert result["job_id"] == "job_1"
    assert result["status"] == "done"


@pytest.mark.asyncio
async def test_inference_client_via_gemini():
    from app.services.inference_client import analyze_image

    class FakeResp:
        status_code = 200
        text = '{"job_id": "job_gemini", "status": "done"}'

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": '{"job_id": "job_gemini", "status": "done", "nguyen_lieu_tho_quet_duoc": []}'
                                }
                            ]
                        }
                    }
                ]
            }

    class FakeClient:
        captured_payload = None

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json=None):
            FakeClient.captured_payload = json
            return FakeResp()

    with patch("app.services.inference_client.settings.ai_provider", "gemini"), \
         patch("app.services.inference_client.settings.gemini_api_key", "secret-key"), \
         patch("app.services.inference_client.settings.gemini_model", "gemini-2.0-flash"), \
         patch("app.services.inference_client.settings.gemini_temperature", 0.9), \
         patch("app.services.inference_client.settings.gemini_top_p", 0.95), \
         patch("httpx.AsyncClient", FakeClient):
        from app.schemas.cv_schemas import UserAnalysisContext
        ctx = UserAnalysisContext(user_id="u1", dietary_preferences=["high_protein"])
        result = await analyze_image(b"abc", "food.jpg", "image/jpeg", ctx)

    assert result["job_id"] == "job_gemini"
    payload = FakeClient.captured_payload
    assert payload is not None
    assert payload["generationConfig"]["temperature"] == 0.9
    assert "high_protein" in payload["contents"][0]["parts"][0]["text"]
    assert result["status"] == "done"


# ── Schema sanity ───────────────────────────────────────────
def test_ai_inference_response_schema():
    from app.schemas.cv_schemas import AIInferenceResponse

    payload = {
        "job_id": "job_1",
        "request_id": "req_1",
        "status": "done",
        "nguyen_lieu_tho_quet_duoc": [],
        "danh_sach_mon_an_goi_y": [],
    }
    model = AIInferenceResponse(**payload)
    assert model.api_version == "v1"
    assert model.status == "done"


# ── Auth helper ────────────────────────────────────────────
@pytest.mark.asyncio
async def test_enrich_ai_response_adds_nutrition_to_each_suggested_dish():
    from app.schemas.cv_schemas import FoodNutrition, MacroNutrients
    from app.services.response_enricher import enrich_ai_response

    payload = {
        "job_id": "job_1",
        "request_id": "req_1",
        "status": "done",
        "nguyen_lieu_tho_quet_duoc": [],
        "danh_sach_mon_an_goi_y": [
            {
                "id_mon_an_goi_y": "rec_01",
                "ten_mon_an": "Uc ga ap chao",
                "ten_mon_an_ky_thuat": "uc_ga_ap_chao",
                "mo_ta_ngan": "Mon an giau protein.",
                "do_kha_thi": "95%",
                "confidence": 0.95,
                "nguyen_lieu_su_dung": [
                    {"ten": "Uc ga", "ten_ky_thuat": "uc_ga", "khoi_luong_g": 200}
                ],
                "thong_tin_dinh_duong_mon_an": {
                    "tong_calories": 0,
                    "protein_g": 0,
                    "carbs_g": 0,
                    "fat_g": 0,
                    "fiber_g": 0,
                },
            }
        ],
    }
    breakdown = [
        FoodNutrition(
            food_label_key="uc_ga",
            food_label_vi="Uc ga",
            estimated_grams=200,
            macros=MacroNutrients(
                calories_kcal=330,
                protein_g=62,
                carbs_g=0,
                fat_g=7,
                fiber_g=0,
            ),
            data_source="usda",
            confidence=0.75,
        )
    ]

    with patch("app.services.response_enricher.settings.nutrition_enrichment_enabled", True), \
         patch("app.services.nutrition_service.nutrition_service.lookup_batch", new=AsyncMock(return_value=breakdown)):
        result = await enrich_ai_response(payload)

    dish = result["danh_sach_mon_an_goi_y"][0]
    assert dish["nutrition_breakdown"][0]["food_label_key"] == "uc_ga"
    assert dish["total_macros"]["protein_g"] == 62
    assert dish["thong_tin_dinh_duong_mon_an"]["tong_calories"] == 330


def test_api_key_dependency_rejects_wrong_key():
    from app.api.auth import require_api_key
    from app.core.config import settings
    from fastapi import HTTPException
    from fastapi.security import HTTPAuthorizationCredentials
    from starlette.requests import Request

    request = Request({"type": "http", "client": ("127.0.0.1", 1234), "headers": []})
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong-key")

    with patch.object(settings, "api_secret_key", "test-secret"):
        with pytest.raises(HTTPException):
            import asyncio
            asyncio.run(require_api_key(request, credentials))


def test_api_key_dependency_rate_limits_valid_key():
    from app.api.auth import _rate_buckets, require_api_key
    from app.core.config import settings
    from fastapi import HTTPException
    from fastapi.security import HTTPAuthorizationCredentials
    from starlette.requests import Request
    import asyncio

    request = Request({"type": "http", "client": ("127.0.0.1", 1234), "headers": []})
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="test-secret")
    _rate_buckets.clear()

    with patch.object(settings, "api_secret_key", "test-secret"), \
         patch.object(settings, "api_rate_limit_per_minute", 1):
        asyncio.run(require_api_key(request, credentials))
        with pytest.raises(HTTPException) as exc:
            asyncio.run(require_api_key(request, credentials))

    assert exc.value.status_code == 429
    _rate_buckets.clear()
