"""Tests for AI response normalization and nutrition enrichment."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.schemas.cv_schemas import AIInferenceResponse, FoodNutrition, MacroNutrients, UserAnalysisContext
from app.services.response_enricher import enrich_ai_response


@pytest.fixture
def english_ai_payload() -> dict:
    return {
        "job_id": "job_1",
        "request_id": "req_1",
        "status": "done",
        "nguyen_lieu_tho_quet_duoc": [
            {
                "id_nguyen_lieu": "raw_01",
                "ten_nguyen_lieu": "Chicken breast",
                "ten_nguyen_lieu_ky_thuat": "chicken_breast",
                "khoi_luong_uoc_tinh_g": 200.0,
                "do_chinh_xac_uoc_tinh": "90%",
            }
        ],
        "danh_sach_mon_an_goi_y": [
            {
                "id_mon_an_goi_y": "rec_01",
                "ten_mon_an": "Grilled chicken",
                "ten_mon_an_ky_thuat": "grilled_chicken",
                "mo_ta_ngan": "Simple grilled chicken breast.",
                "do_kha_thi": "90%",
                "confidence": 0.9,
                "nguyen_lieu_su_dung": [
                    {"ten": "Chicken breast", "ten_ky_thuat": "chicken_breast", "khoi_luong_g": 200.0}
                ],
                "thong_tin_dinh_duong_mon_an": {
                    "tong_calories": 999.0,
                    "protein_g": 1.0,
                    "carbs_g": 1.0,
                    "fat_g": 1.0,
                    "fiber_g": 0.0,
                },
            }
        ],
    }


@pytest.mark.asyncio
async def test_enrich_normalizes_english_ingredients(english_ai_payload):
    mock_breakdown = [
        FoodNutrition(
            food_label_key="uc_ga",
            food_label_vi="Ức gà tươi sống",
            estimated_grams=200.0,
            macros=MacroNutrients(calories_kcal=330, protein_g=62, carbs_g=0, fat_g=7),
            data_source="fallback",
            confidence=0.3,
        )
    ]
    with patch(
        "app.services.nutrition_service.nutrition_service.lookup_batch",
        new_callable=AsyncMock,
        return_value=mock_breakdown,
    ), patch(
        "app.services.nutrition_service.nutrition_service.sum_macros",
        return_value=mock_breakdown[0].macros,
    ):
        result = await enrich_ai_response(english_ai_payload)

    ingredient = result["nguyen_lieu_tho_quet_duoc"][0]
    assert ingredient["ten_nguyen_lieu_ky_thuat"] == "uc_ga"
    assert ingredient["ten_nguyen_lieu"] == "Ức gà tươi sống"
    assert "nutrition_breakdown" in result
    assert result["nutrition_breakdown"][0]["food_label_key"] == "uc_ga"


@pytest.mark.asyncio
async def test_enrich_updates_dish_nutrition_from_catalog(english_ai_payload):
    dish_macros = MacroNutrients(calories_kcal=320, protein_g=55, carbs_g=0, fat_g=7, fiber_g=0)
    mock_breakdown = [
        FoodNutrition(
            food_label_key="uc_ga",
            food_label_vi="Ức gà tươi sống",
            estimated_grams=200.0,
            macros=dish_macros,
            data_source="fallback",
            confidence=0.3,
        )
    ]
    with patch(
        "app.services.nutrition_service.nutrition_service.lookup_batch",
        new_callable=AsyncMock,
        return_value=mock_breakdown,
    ), patch(
        "app.services.nutrition_service.nutrition_service.sum_macros",
        return_value=dish_macros,
    ):
        result = await enrich_ai_response(english_ai_payload)

    dish_nutrition = result["danh_sach_mon_an_goi_y"][0]["thong_tin_dinh_duong_mon_an"]
    assert dish_nutrition["tong_calories"] == 320.0
    assert dish_nutrition["protein_g"] == 55.0


@pytest.mark.asyncio
async def test_enrich_picks_random_dish(english_ai_payload):
    english_ai_payload["danh_sach_mon_an_goi_y"].append({
        "id_mon_an_goi_y": "rec_02",
        "ten_mon_an": "Salad ức gà",
        "ten_mon_an_ky_thuat": "salad_uc_ga",
        "mo_ta_ngan": "Salad nhẹ.",
        "do_kha_thi": "90%",
        "confidence": 0.9,
        "nguyen_lieu_su_dung": [
            {"ten": "Ức gà", "ten_ky_thuat": "uc_ga", "khoi_luong_g": 200.0}
        ],
        "thong_tin_dinh_duong_mon_an": {
            "tong_calories": 250,
            "protein_g": 40,
            "carbs_g": 5,
            "fat_g": 5,
            "fiber_g": 2,
        },
    })
    mock_breakdown = [
        FoodNutrition(
            food_label_key="uc_ga",
            food_label_vi="Ức gà tươi sống",
            estimated_grams=200.0,
            macros=MacroNutrients(calories_kcal=330, protein_g=62, carbs_g=0, fat_g=7),
            data_source="fallback",
            confidence=0.3,
        )
    ]
    with patch(
        "app.services.nutrition_service.nutrition_service.lookup_batch",
        new_callable=AsyncMock,
        return_value=mock_breakdown,
    ), patch(
        "app.services.nutrition_service.nutrition_service.sum_macros",
        return_value=mock_breakdown[0].macros,
    ), patch("app.services.response_enricher.random.choice", return_value=english_ai_payload["danh_sach_mon_an_goi_y"][1]):
        result = await enrich_ai_response(english_ai_payload)

    assert result["mon_an_goi_y_chon"]["ten_mon_an"] == "Salad ức gà"
    assert len(result["danh_sach_mon_an_goi_y"]) == 2


@pytest.mark.asyncio
async def test_enrich_skips_non_done_status(english_ai_payload):
    english_ai_payload["status"] = "processing"
    result = await enrich_ai_response(english_ai_payload)
    assert "nutrition_breakdown" not in result


@pytest.mark.asyncio
async def test_enrich_annotates_allergy_safety(english_ai_payload):
    ctx = UserAnalysisContext(
        user_id="u1",
        allergies=["Trứng"],
        allergy_keys=["trung"],
    )
    english_ai_payload["danh_sach_mon_an_goi_y"][0]["nguyen_lieu_su_dung"] = [
        {"ten": "Trứng gà", "ten_ky_thuat": "trung_ga", "khoi_luong_g": 50.0}
    ]
    with patch(
        "app.services.nutrition_service.nutrition_service.lookup_batch",
        new_callable=AsyncMock,
        return_value=[],
    ), patch(
        "app.services.nutrition_service.nutrition_service.sum_macros",
        return_value=MacroNutrients(calories_kcal=0, protein_g=0, carbs_g=0, fat_g=0),
    ):
        result = await enrich_ai_response(english_ai_payload, user_context=ctx)

    dish = result["danh_sach_mon_an_goi_y"][0]
    assert dish["an_toan_cho_user"] is False
    assert "Trứng" in dish["dich_ung_trung"]


def test_ai_inference_response_accepts_enriched_fields(english_ai_payload):
    english_ai_payload["nutrition_breakdown"] = []
    english_ai_payload["total_macros"] = {
        "calories_kcal": 100,
        "protein_g": 10,
        "carbs_g": 5,
        "fat_g": 2,
        "fiber_g": 1,
    }
    model = AIInferenceResponse(**english_ai_payload)
    assert model.total_macros is not None
    assert model.total_macros.protein_g == 10
