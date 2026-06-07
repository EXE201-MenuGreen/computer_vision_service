"""
Tests for CV microservice API-based flow.
Run: pytest tests/test_cv_service.py -v
"""
from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
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

    async def mock_require_api_key(authorization: str = Header(default="", alias="Authorization")):
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
    with patch("app.services.worker.enqueue_inference_job", return_value="job_123"):
        yield


# ── Health endpoint ─────────────────────────────────────────
def test_health_ok(client):
    r = client.get("/api/v1/cv/health", headers={"Authorization": "Bearer test-key"})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True


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
def test_job_poll_processing(client):
    with patch("app.services.worker.get_job_result", return_value=None):
        r = client.get(
            "/api/v1/cv/jobs/job_123",
            headers={"Authorization": "Bearer test-key"},
        )
    assert r.status_code == 200
    assert r.json()["status"] == "processing"


def test_job_poll_done(client, mock_api_response):
    with patch("app.services.worker.get_job_result", return_value={"status": "done", "result": mock_api_response}):
        r = client.get(
            "/api/v1/cv/jobs/job_123",
            headers={"Authorization": "Bearer test-key"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "done"
    assert data["result"]["status"] == "done"
    assert data["result"]["nguyen_lieu_tho_quet_duoc"][0]["ten_nguyen_lieu_ky_thuat"] == "uc_ga"


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

        async def post(self, url, files=None):
            self.captured_files = files
            return FakeResp()

    with patch("app.services.inference_client.settings.ai_provider", "remote_api"), \
         patch("app.services.inference_client.settings.ai_api_base_url", "http://ai-api"), \
         patch("app.services.inference_client.settings.ai_api_key", "secret"), \
         patch("httpx.AsyncClient", FakeClient):
        result = await analyze_image(b"abc", "food.jpg", "image/jpeg")

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
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json=None):
            return FakeResp()

    with patch("app.services.inference_client.settings.ai_provider", "gemini"), \
         patch("app.services.inference_client.settings.gemini_api_key", "secret-key"), \
         patch("app.services.inference_client.settings.gemini_model", "gemini-2.0-flash"), \
         patch("httpx.AsyncClient", FakeClient):
        result = await analyze_image(b"abc", "food.jpg", "image/jpeg")

    assert result["job_id"] == "job_gemini"
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
def test_api_key_dependency_rejects_wrong_key():
    from app.api.auth import require_api_key
    from app.core.config import settings
    from fastapi import HTTPException

    with patch.object(settings, "api_secret_key", "test-secret"):
        with pytest.raises(HTTPException):
            import asyncio
            asyncio.run(require_api_key("Bearer wrong-key"))
