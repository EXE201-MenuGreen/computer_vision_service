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
    r = client.get("/api/v1/cv/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("ok", "loading")
    assert "models_loaded" in data


# ── Image validation ────────────────────────────────────────
def test_rejects_non_image(client):
    r = client.post(
        "/api/v1/cv/analyze",
        files={"image": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert r.status_code == 415


def test_rejects_empty_file(client):
    r = client.post(
        "/api/v1/cv/analyze",
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
            "/api/v1/cv/analyze",
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


# ── Weights mapping tests ───────────────────────────────────
def test_weights_paths_are_configured_for_pipeline():
    """Ensure the configured weights paths line up with the code defaults and files."""
    from pathlib import Path
    from app.core.config import settings

    assert settings.food_detection_weights.endswith((".pt", ".pth")) or "best.pt" in settings.food_detection_weights
    assert settings.food_classify_weights.endswith((".pt", ".pth"))
    assert settings.pipeline_detector == "yolov8"
    assert settings.pipeline_classifier in ("efficientnet_b4", "yolo_passthrough", "mock")
    assert settings.pipeline_depth in ("depth_anything_v2", "heuristic", "mock")
    assert settings.pipeline_postprocessor == "default"
    assert Path(settings.food_detection_weights).exists() or settings.food_detection_weights == "yolov8n.pt"


def test_detector_and_classifier_fallback_when_weights_missing():
    """Detector/classifier should still load with fallback models if weights are missing."""
    import app.stages  # noqa: F401
    from app.stages.detection.detector import YoloV8Detector
    from app.stages.classification.classifier import EfficientNetClassifier

    with patch("app.stages.detection.detector.Path.exists", return_value=False), \
         patch("ultralytics.YOLO") as mock_yolo, \
         patch("app.stages.detection.detector.settings.device", "cpu"):
        detector = YoloV8Detector()
        detector.load()
        mock_yolo.assert_called_once_with("yolov8n.pt")

    fake_model = MagicMock()
    fake_model.eval = MagicMock(return_value=fake_model)
    fake_model.to = MagicMock(return_value=fake_model)

    with patch("app.stages.classification.classifier.Path.exists", return_value=False), \
         patch("timm.create_model", return_value=fake_model) as mock_create_model:
        classifier = EfficientNetClassifier()
        classifier.load()
        mock_create_model.assert_called_once_with("efficientnet_b4", pretrained=True, num_classes=101)


# ── Weight-file linkage test ────────────────────────────────
def test_weight_file_linkage_matches_config():
    """Verify detector/classifier config points to expected weight files."""
    from app.core.config import settings
    assert settings.food_detection_weights in ("weights/yolov8_food.pt", "weights/best.pt")
    assert settings.food_classify_weights == "weights/efficientnet_food.pt"


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


# ── Cache and Admin tests ───────────────────────────────────

@pytest.mark.asyncio
async def test_nutrition_service_tier0_bypasses_cache():
    """Verify that Tier 0 (verified food) bypasses Redis cache lookup and is returned directly."""
    from app.services.nutrition_service import NutritionService
    from app.schemas.cv_schemas import DetectedFood, BoundingBox, MacroNutrients
    
    svc = NutritionService()
    food = DetectedFood(
        label="verified_pho",
        confidence=0.9,
        bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
        estimated_grams=100.0,
    )
    
    verified_macros = MacroNutrients(calories_kcal=150, protein_g=10, carbs_g=20, fat_g=3)
    
    with patch("app.services.nutrition_service.get_verified_food", AsyncMock(return_value=verified_macros)), \
         patch("app.services.redis_cache.get_nutrition", AsyncMock()) as mock_cache_get, \
         patch("app.services.redis_cache.set_nutrition", AsyncMock()) as mock_cache_set:
         
        breakdown = await svc.lookup_batch([food])
        
        assert len(breakdown) == 1
        assert breakdown[0].macros.calories_kcal == 150
        assert breakdown[0].data_source == "verified"
        assert breakdown[0].confidence == 1.0
        
        # Verify that cache was NOT checked and NOT set
        mock_cache_get.assert_not_called()
        mock_cache_set.assert_not_called()


@pytest.mark.asyncio
async def test_nutrition_service_tier1_hits_cache():
    """Verify that a Tier 1 (Redis) hit returns cached data without calling pgvector/USDA/fallback."""
    from app.services.nutrition_service import NutritionService
    from app.schemas.cv_schemas import DetectedFood, BoundingBox
    
    svc = NutritionService()
    food = DetectedFood(
        label="cached_food",
        confidence=0.9,
        bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
        estimated_grams=100.0,
    )
    
    cached_data = {
        "macros": {
            "calories_kcal": 220.0,
            "protein_g": 8.0,
            "carbs_g": 30.0,
            "fat_g": 6.0,
            "fiber_g": 2.0,
        },
        "fdc_id": "123456",
        "data_source": "pgvector",
        "confidence": 0.90,
    }
    
    with patch("app.services.nutrition_service.get_verified_food", AsyncMock(return_value=None)), \
         patch("app.services.redis_cache.get_nutrition", AsyncMock(return_value=cached_data)) as mock_cache_get, \
         patch("app.services.nutrition_service.match_food", AsyncMock()) as mock_match_food, \
         patch("app.services.redis_cache.set_nutrition", AsyncMock()) as mock_cache_set:
         
        breakdown = await svc.lookup_batch([food])
        
        assert len(breakdown) == 1
        assert breakdown[0].macros.calories_kcal == 220.0
        assert breakdown[0].data_source == "pgvector"
        assert breakdown[0].confidence == 0.90
        
        mock_cache_get.assert_called_once_with("cached_food")
        mock_match_food.assert_not_called()
        mock_cache_set.assert_not_called()


@pytest.mark.asyncio
async def test_nutrition_service_caches_non_verified_lookups():
    """Verify that a lookup falling through to pgvector/USDA/fallback saves the result to cache."""
    from app.services.nutrition_service import NutritionService
    from app.schemas.cv_schemas import DetectedFood, BoundingBox, MacroNutrients
    
    svc = NutritionService()
    food = DetectedFood(
        label="non_cached_food",
        confidence=0.9,
        bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
        estimated_grams=100.0,
    )
    
    pg_macros = MacroNutrients(calories_kcal=180, protein_g=12, carbs_g=15, fat_g=8)
    
    with patch("app.services.nutrition_service.get_verified_food", AsyncMock(return_value=None)), \
         patch("app.services.redis_cache.get_nutrition", AsyncMock(return_value=None)), \
         patch("app.services.nutrition_service.match_food", AsyncMock(return_value=(pg_macros, "789"))), \
         patch("app.services.redis_cache.set_nutrition", AsyncMock()) as mock_cache_set:
         
        breakdown = await svc.lookup_batch([food])
        
        assert len(breakdown) == 1
        assert breakdown[0].macros.calories_kcal == 180
        assert breakdown[0].data_source == "pgvector"
        
        # Verify that set_nutrition was called to cache the result
        mock_cache_set.assert_called_once()
        args, kwargs = mock_cache_set.call_args
        assert args[0] == "non_cached_food"
        assert args[1]["macros"]["calories_kcal"] == 180.0
        assert args[1]["data_source"] == "pgvector"


def test_admin_upsert_invalidates_cache(client):
    """Verify that adding a verified entry via admin API invalidates the cache for that food label."""
    from unittest.mock import patch, AsyncMock
    
    req_body = {
        "food_label": "new_food",
        "calories_kcal": 250,
        "protein_g": 12,
        "carbs_g": 40,
        "fat_g": 5,
        "verified_by": "admin",
    }
    
    # Mock settings.admin_api_key, service_client, and redis_cache.invalidate
    with patch("app.api.admin_router.settings.admin_api_key", "secret_key"), \
         patch("app.api.admin_router.service_client") as mock_client_factory, \
         patch("app.api.admin_router.redis_cache.invalidate", AsyncMock(return_value=1)) as mock_invalidate:
         
         mock_client = MagicMock()
         mock_client.rpc = MagicMock()
         mock_client_factory.return_value = mock_client
         
         r = client.post(
             "/api/v1/cv/admin/verified",
             headers={"X-Admin-Key": "secret_key"},
             json=req_body,
         )
         
         assert r.status_code == 201
         assert r.json()["food_label"] == "new_food"
         
         # Verify that cache was invalidated for the correct label
         mock_invalidate.assert_called_once_with("new_food")


def test_admin_delete_invalidates_cache(client):
    """Verify that deleting a verified entry via admin API invalidates the cache."""
    from unittest.mock import patch, AsyncMock
    
    with patch("app.api.admin_router.settings.admin_api_key", "secret_key"), \
         patch("app.api.admin_router.service_client") as mock_client_factory, \
         patch("app.api.admin_router.redis_cache.invalidate", AsyncMock(return_value=1)) as mock_invalidate:
         
         mock_client = MagicMock()
         mock_client.table = MagicMock()
         mock_client_factory.return_value = mock_client
         
         r = client.delete(
             "/api/v1/cv/admin/verified/new_food",
             headers={"X-Admin-Key": "secret_key"},
         )
         
         assert r.status_code == 200
         assert r.json()["food_label"] == "new_food"
         
         # Verify that cache was invalidated for the correct label
         mock_invalidate.assert_called_once_with("new_food")

