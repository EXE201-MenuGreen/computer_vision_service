"""
Tests for CV Microservice.
Run: pytest tests/ -v
"""
import io
import pytest
from PIL import Image
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


# ── Fixtures ────────────────────────────────────────────────
@pytest.fixture
def fake_rgb_image() -> bytes:
    """Generate a small fake JPEG for upload tests."""
    img = Image.new("RGB", (320, 240), color=(120, 80, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def mock_pipeline():
    """A mock CVPipeline that returns a canned StageContext."""
    from app.core.base import StageContext
    from app.schemas.cv_schemas import BoundingBox, DetectedFood

    pipeline = MagicMock()
    pipeline.is_ready = True
    pipeline.load_all = MagicMock()

    def fake_run(image):
        ctx = StageContext(image=image)
        ctx.food_items = [
            DetectedFood(
                label="pizza",
                confidence=0.91,
                bbox=BoundingBox(x1=10, y1=10, x2=200, y2=200),
                estimated_grams=250.0,
            )
        ]
        ctx.processing_time_ms = 340.2
        return ctx

    pipeline.run = MagicMock(side_effect=fake_run)
    return pipeline


@pytest.fixture
def client(mock_pipeline):
    """TestClient with pipeline mocked to skip GPU load."""
    with patch(
        "app.main.PipelineFactory.build",
        return_value=mock_pipeline,
    ):
        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ── Health endpoint ─────────────────────────────────────────
def test_health_ok(client):
    r = client.get("/cv/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("ok", "loading")
    assert "models_loaded" in data


# ── Image validation ────────────────────────────────────────
def test_rejects_non_image(client):
    r = client.post(
        "/cv/analyze",
        files={"image": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert r.status_code == 415


def test_rejects_empty_file(client):
    r = client.post(
        "/cv/analyze",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert r.status_code in (400, 422)


# ── Full pipeline (mocked inference) ────────────────────────
def test_analyze_returns_result(client, fake_rgb_image):
    from app.schemas.cv_schemas import MacroNutrients, FoodNutrition

    mock_nutrition = [
        FoodNutrition(
            food_label="pizza",
            estimated_grams=250.0,
            macros=MacroNutrients(
                calories_kcal=665, protein_g=27.5, carbs_g=82.5, fat_g=25.0
            ),
        )
    ]
    mock_total = MacroNutrients(
        calories_kcal=665, protein_g=27.5, carbs_g=82.5, fat_g=25.0
    )

    with patch(
        "app.api.cv_router.nutrition_service"
    ) as mock_ns:
        mock_ns.lookup_batch = AsyncMock(return_value=mock_nutrition)
        mock_ns.sum_macros = MagicMock(return_value=mock_total)

        r = client.post(
            "/cv/analyze",
            files={"image": ("food.jpg", fake_rgb_image, "image/jpeg")},
        )

    assert r.status_code == 200
    data = r.json()
    assert len(data["detected_foods"]) == 1
    assert data["detected_foods"][0]["label"] == "pizza"
    assert data["total_macros"]["calories_kcal"] == 665


# ── Nutrition service unit test ──────────────────────────────
@pytest.mark.asyncio
async def test_nutrition_fallback():
    from app.services.nutrition_service import NutritionService
    from app.schemas.cv_schemas import DetectedFood, BoundingBox

    svc = NutritionService()
    food = DetectedFood(
        label="pho",
        confidence=0.88,
        bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
        estimated_grams=300.0,
    )

    with patch("httpx.AsyncClient.get", side_effect=Exception("network error")):
        breakdown = await svc.lookup_batch([food])

    assert len(breakdown) == 1
    assert breakdown[0].macros.calories_kcal > 0


# ── Macro sum test ───────────────────────────────────────────
def test_sum_macros():
    from app.services.nutrition_service import NutritionService
    from app.schemas.cv_schemas import FoodNutrition, MacroNutrients

    svc = NutritionService()
    items = [
        FoodNutrition(
            food_label="a", estimated_grams=100,
            macros=MacroNutrients(calories_kcal=200, protein_g=10, carbs_g=30, fat_g=5)
        ),
        FoodNutrition(
            food_label="b", estimated_grams=50,
            macros=MacroNutrients(calories_kcal=100, protein_g=5, carbs_g=10, fat_g=3)
        ),
    ]
    total = svc.sum_macros(items)
    assert total.calories_kcal == 300
    assert total.protein_g == 15


# ── Stage registry tests ───────────────────────────────────
def test_registry_contains_default_stages():
    """Verify all default stages are registered."""
    import app.stages  # noqa: F401 — trigger registration
    from app.registry import (
        detector_registry,
        classifier_registry,
        depth_registry,
        postprocessor_registry,
    )

    assert "yolov8" in detector_registry
    assert "mock" in detector_registry
    assert "efficientnet_b4" in classifier_registry
    assert "mock" in classifier_registry
    assert "depth_anything_v2" in depth_registry
    assert "heuristic" in depth_registry
    assert "mock" in depth_registry
    assert "default" in postprocessor_registry


def test_registry_raises_on_unknown_key():
    from app.registry import detector_registry

    with pytest.raises(KeyError, match="No 'detector' stage"):
        detector_registry.get("nonexistent_model_xyz")


# ── Pipeline factory test ──────────────────────────────────
def test_pipeline_factory_builds_mock_pipeline():
    """Build a pipeline with all mock stages — no GPU needed."""
    import app.stages  # noqa: F401
    from app.pipeline.pipeline_factory import PipelineFactory

    config = {
        "detector": "mock",
        "classifier": "mock",
        "depth": "mock",
        "postprocessor": "default",
    }
    pipeline = PipelineFactory.build(config)
    pipeline.load_all()

    assert pipeline.is_ready
    assert len(pipeline.stage_names) == 5  # preprocessor + 4 stages


def test_mock_pipeline_run():
    """Run a full mock pipeline end-to-end."""
    import app.stages  # noqa: F401
    from app.pipeline.pipeline_factory import PipelineFactory

    config = {
        "detector": "mock",
        "classifier": "mock",
        "depth": "mock",
        "postprocessor": "default",
    }
    pipeline = PipelineFactory.build(config)
    pipeline.load_all()

    img = Image.new("RGB", (320, 240), color=(120, 80, 60))
    ctx = pipeline.run(img)

    assert len(ctx.food_items) == 1
    assert ctx.food_items[0].label == "pizza"
    assert ctx.food_items[0].confidence > 0
    assert ctx.processing_time_ms > 0
