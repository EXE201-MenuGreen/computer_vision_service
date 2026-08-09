import math
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from app.schemas.cv_schemas import FoodNutrition, MacroNutrients
from app.schemas.meal_scan_schemas import (
    PreparedMealAnalysisResponse,
    PreparedMealJobStatusResponse,
    PreparedMealNutrition,
    RawPreparedMealAnalysis,
)
from app.services.inference_client import InferenceClientError, analyze_prepared_meal_image
from app.services.prepared_meal_service import analyze_prepared_meal, build_prepared_meal_prompt


def test_prompt_is_for_existing_meal_and_never_requests_suggestions():
    prompt = build_prepared_meal_prompt()
    assert "MÓN ĂN ĐÃ HOÀN CHỈNH" in prompt
    assert "KHÔNG gợi ý món ăn khác" in prompt
    assert "KHÔNG trả calories/macros" in prompt


@pytest.mark.parametrize(
    "field,value",
    [
        ("dish_confidence", -0.1),
        ("dish_confidence", 1.1),
        ("dish_confidence", math.nan),
    ],
)
def test_raw_schema_rejects_invalid_confidence(field, value):
    payload = {
        "dish_name": "Cơm gà", "dish_name_key": "com_ga", field: value,
        "ingredients": [{
            "ingredient_id": "i1", "name": "Cơm", "name_key": "rice",
            "estimated_grams": 100, "detection_confidence": 0.8,
        }],
    }
    with pytest.raises(ValidationError):
        RawPreparedMealAnalysis.model_validate(payload)


@pytest.mark.parametrize("grams", [math.inf, -math.inf, math.nan])
def test_raw_schema_rejects_non_finite_grams(grams):
    payload = {
        "dish_name": "Cơm gà",
        "dish_name_key": "com_ga",
        "dish_confidence": 0.9,
        "ingredients": [{
            "ingredient_id": "i1",
            "name": "Cơm",
            "name_key": "rice",
            "estimated_grams": grams,
            "detection_confidence": 0.8,
        }],
    }
    with pytest.raises(ValidationError):
        RawPreparedMealAnalysis.model_validate(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("dish_name"),
        lambda payload: payload["ingredients"][0].update({"estimated_grams": -0.1}),
        lambda payload: payload["ingredients"][0].update({"detection_confidence": 1.1}),
    ],
)
def test_raw_schema_rejects_missing_required_and_out_of_range_fields(mutation):
    payload = {
        "dish_name": "Prepared rice",
        "dish_name_key": "prepared_rice",
        "dish_confidence": 0.9,
        "ingredients": [{
            "ingredient_id": "i1",
            "name": "Rice",
            "name_key": "rice",
            "estimated_grams": 100,
            "detection_confidence": 0.8,
        }],
    }
    mutation(payload)
    with pytest.raises(ValidationError):
        RawPreparedMealAnalysis.model_validate(payload)


def test_final_schemas_enforce_analysis_type_and_status_literals():
    with pytest.raises(ValidationError):
        PreparedMealJobStatusResponse(
            job_id="job", status="unknown", message="invalid"
        )

    invalid_analysis_type = {
        "job_id": "job",
        "request_id": "request",
        "status": "done",
        "analysis_type": "ingredient_scan",
        "dish_name": "Rice",
        "dish_name_key": "rice",
        "dish_confidence": 0.8,
        "estimated_total_grams": 100,
        "ingredients": [{
            "ingredient_id": "i1",
            "name": "Rice",
            "name_key": "rice",
            "estimated_grams": 100,
            "detection_confidence": 0.8,
            "nutrition": {
                "macros": {
                    "calories_kcal": 130,
                    "protein_g": 2.7,
                    "carbs_g": 28,
                    "fat_g": 0.3,
                    "fiber_g": 0.4,
                },
                "data_source": "fallback",
                "confidence": 0.3,
            },
        }],
        "total_macros": {
            "calories_kcal": 130,
            "protein_g": 2.7,
            "carbs_g": 28,
            "fat_g": 0.3,
            "fiber_g": 0.4,
        },
        "estimation_note": "Estimated.",
    }
    with pytest.raises(ValidationError):
        PreparedMealAnalysisResponse.model_validate(invalid_analysis_type)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, math.nan])
def test_final_nutrition_schema_rejects_invalid_confidence(confidence):
    with pytest.raises(ValidationError):
        PreparedMealNutrition(
            macros=MacroNutrients(
                calories_kcal=1, protein_g=1, carbs_g=1, fat_g=1, fiber_g=None
            ),
            data_source="fallback",
            confidence=confidence,
        )


def test_raw_schema_limits_nutrition_lookup_fanout():
    ingredient = {
        "ingredient_id": "i1",
        "name": "Cơm",
        "name_key": "rice",
        "estimated_grams": 10,
        "detection_confidence": 0.8,
    }
    payload = {
        "dish_name": "Cơm",
        "dish_name_key": "com",
        "dish_confidence": 0.9,
        "ingredients": [
            {**ingredient, "ingredient_id": f"i{index}"} for index in range(21)
        ],
    }
    with pytest.raises(ValidationError):
        RawPreparedMealAnalysis.model_validate(payload)


@pytest.mark.asyncio
async def test_prepared_meal_cache_uses_isolated_namespace():
    cached = {"dish_name": "Cơm", "ingredients": []}
    with patch(
        "app.services.inference_client.settings.image_cache_ttl_seconds", 60
    ), patch(
        "app.services.inference_client.get_cached_result",
        new=AsyncMock(return_value=cached),
    ) as cache_get, patch(
        "app.services.inference_client._analyze_image_via_gemini_prompt",
        new=AsyncMock(),
    ) as provider:
        result = await analyze_prepared_meal_image(
            b"same-image", "meal.jpg", "image/jpeg", "prompt"
        )
    assert result == cached
    assert cache_get.await_args.kwargs["namespace"] == "prepared_meal"
    provider.assert_not_awaited()


@pytest.mark.asyncio
async def test_prepared_meal_remote_provider_fails_explicitly_without_fallback():
    with patch(
        "app.services.inference_client.settings.image_cache_ttl_seconds", 0
    ), patch(
        "app.services.inference_client.settings.ai_provider", "remote_api"
    ):
        with pytest.raises(InferenceClientError) as caught:
            await analyze_prepared_meal_image(
                b"image", "meal.jpg", "image/jpeg", "prompt"
            )
    assert caught.value.is_transient is False
    assert "supported only by the Gemini provider" in str(caught.value)


@pytest.mark.asyncio
async def test_service_normalizes_and_uses_only_nutrition_service_macros():
    raw = {
        "job_id": "ai-job", "request_id": "ai-request",
        "dish_name": "Cơm gà", "dish_name_key": "com_ga", "dish_confidence": 0.9,
        "ingredients": [{
            "ingredient_id": "i1", "name": "Ức gà", "name_key": "chicken breast",
            "estimated_grams": 200, "detection_confidence": 0.8,
            "calories_kcal": 99999,
        }],
    }
    nutrition = [FoodNutrition(
        food_label_key="uc_ga", food_label_vi="Ức gà", estimated_grams=200,
        macros=MacroNutrients(calories_kcal=330, protein_g=62, carbs_g=0, fat_g=7, fiber_g=None),
        data_source="usda", confidence=0.75, usda_fdc_id="123",
    )]
    with patch(
        "app.services.prepared_meal_service.analyze_prepared_meal_image",
        new=AsyncMock(return_value=raw),
    ), patch.object(
        __import__("app.services.prepared_meal_service", fromlist=["nutrition_service"]).nutrition_service,
        "lookup_batch", new=AsyncMock(return_value=nutrition),
    ):
        result = await analyze_prepared_meal(b"img", "meal.jpg", "image/jpeg")

    assert result["ingredients"][0]["name_key"] == "uc_ga"
    assert result["ingredients"][0]["nutrition"]["macros"]["calories_kcal"] == 330
    assert result["total_macros"]["calories_kcal"] == 330
    assert result["estimated_total_grams"] == 200
    assert result["ingredients"][0]["nutrition"]["data_source"] == "usda"
    assert "ước tính" in result["estimation_note"]


@pytest.mark.asyncio
async def test_service_aggregates_usda_and_fallback_rows_with_nullable_fiber():
    raw = {
        "dish_name": "Prepared rice",
        "dish_name_key": "prepared_rice",
        "dish_confidence": 0.8,
        "ingredients": [
            {
                "ingredient_id": "i1",
                "name": "Rice",
                "name_key": "rice",
                "estimated_grams": 150,
                "detection_confidence": 0.9,
            },
            {
                "ingredient_id": "i2",
                "name": "Hidden sauce",
                "name_key": "unknown sauce",
                "estimated_grams": 20,
                "detection_confidence": 0.3,
            },
        ],
    }
    breakdown = [
        FoodNutrition(
            food_label_key="com_trang",
            food_label_vi="Rice",
            estimated_grams=150,
            macros=MacroNutrients(
                calories_kcal=195, protein_g=4, carbs_g=42, fat_g=0.5, fiber_g=0.6
            ),
            data_source="usda",
            confidence=0.75,
        ),
        FoodNutrition(
            food_label_key="unknown_sauce",
            food_label_vi="Hidden sauce",
            estimated_grams=20,
            macros=MacroNutrients(
                calories_kcal=40, protein_g=0, carbs_g=4, fat_g=3, fiber_g=None
            ),
            data_source="fallback",
            confidence=0.3,
        ),
    ]
    with patch(
        "app.services.prepared_meal_service.analyze_prepared_meal_image",
        new=AsyncMock(return_value=raw),
    ), patch.object(
        __import__("app.services.prepared_meal_service", fromlist=["nutrition_service"]).nutrition_service,
        "lookup_batch",
        new=AsyncMock(return_value=breakdown),
    ):
        result = await analyze_prepared_meal(b"img", "meal.jpg", "image/jpeg")

    assert result["estimated_total_grams"] == 170
    assert result["total_macros"] == {
        "calories_kcal": 235.0,
        "protein_g": 4.0,
        "carbs_g": 46.0,
        "fat_g": 3.5,
        "fiber_g": 0.6,
    }
    assert [row["nutrition"]["data_source"] for row in result["ingredients"]] == [
        "usda",
        "fallback",
    ]
