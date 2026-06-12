"""Tests for analyze personalization context builder."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.api.analyze_context import build_user_analysis_context


@pytest.mark.asyncio
async def test_build_context_parses_form_json():
    ctx = await build_user_analysis_context(
        user_id="user-1",
        dietary_preferences='["high_protein"]',
        avoid_foods='["đồ chiên"]',
        recent_dishes='["phở bò"]',
    )
    assert ctx.user_id == "user-1"
    assert ctx.dietary_preferences == ["high_protein"]
    assert ctx.avoid_foods == ["đồ chiên"]
    assert ctx.recent_dishes == ["phở bò"]


@pytest.mark.asyncio
async def test_build_context_loads_history_when_user_id_set():
    from app.schemas.cv_schemas import FoodNutrition, MacroNutrients, MealHistoryItem

    history_item = MealHistoryItem(
        id="1",
        request_id="r1",
        analyzed_at="2026-01-01",
        foods=[
            FoodNutrition(
                food_label_key="pho",
                food_label_vi="Phở bò",
                estimated_grams=400,
                macros=MacroNutrients(calories_kcal=400, protein_g=20, carbs_g=50, fat_g=10),
            )
        ],
        total_macros=MacroNutrients(calories_kcal=400, protein_g=20, carbs_g=50, fat_g=10),
        similarity=1.0,
    )
    with patch("app.api.analyze_context.settings.meal_history_enabled", True), \
         patch("app.db.meal_history.get_recent", new_callable=AsyncMock, return_value=[history_item]):
        ctx = await build_user_analysis_context(
            user_id="user-1",
            dietary_preferences=None,
            avoid_foods=None,
            recent_dishes=None,
        )
    assert "Phở bò" in ctx.recent_dishes


@pytest.mark.asyncio
async def test_build_context_loads_user_profile_from_db():
    from app.db.user_profile import UserAllergyRecord, UserCvProfile, UserHealthCondition

    profile = UserCvProfile(
        allergies=[UserAllergyRecord(allergen_key="hai_san", name="Hải sản", severity="high")],
        health_conditions=[
            UserHealthCondition(condition_key="tieu_duong", name="Tiểu đường", notes="")
        ],
        dietary_goal="giam_can",
        avoid_ingredient_keys=["duong_cat"],
        daily_calorie_limit=1800,
    )
    with patch("app.api.analyze_context.settings.user_profile_enabled", True), \
         patch("app.db.user_profile.get_user_cv_profile", new_callable=AsyncMock, return_value=profile):
        ctx = await build_user_analysis_context(
            user_id="user-1",
            dietary_preferences=None,
            avoid_foods=None,
            recent_dishes=None,
        )
    assert ctx.allergies == ["Hải sản"]
    assert ctx.allergy_keys == ["hai_san"]
    assert ctx.health_conditions == ["Tiểu đường"]
    assert ctx.dietary_goal == "giam_can"
    assert "duong_cat" in ctx.avoid_ingredient_keys
    assert ctx.daily_calorie_limit == 1800
    assert "giam_can" in ctx.dietary_preferences
