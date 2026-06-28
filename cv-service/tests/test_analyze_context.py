"""Tests for request-provided analyze personalization context."""
from __future__ import annotations

import pytest

from app.api.analyze_context import build_user_analysis_context


@pytest.mark.asyncio
async def test_build_context_parses_form_json():
    ctx = await build_user_analysis_context(
        user_id="user-1",
        dietary_preferences='["high_protein"]',
        avoid_foods='["do_chien"]',
        recent_dishes='["pho_bo"]',
    )

    assert ctx.user_id == "user-1"
    assert ctx.dietary_preferences == ["high_protein"]
    assert ctx.avoid_foods == ["do_chien"]
    assert ctx.recent_dishes == ["pho_bo"]


@pytest.mark.asyncio
async def test_build_context_does_not_load_history_from_db():
    ctx = await build_user_analysis_context(
        user_id="user-1",
        dietary_preferences=None,
        avoid_foods=None,
        recent_dishes=None,
    )

    assert ctx.user_id == "user-1"
    assert ctx.recent_dishes == []


@pytest.mark.asyncio
async def test_build_context_uses_only_request_provided_profile_fields():
    ctx = await build_user_analysis_context(
        user_id="user-1",
        dietary_preferences=None,
        avoid_foods='["duong_cat"]',
        recent_dishes=None,
    )

    assert ctx.allergies == []
    assert ctx.allergy_keys == []
    assert ctx.health_conditions == []
    assert ctx.health_condition_keys == []
    assert ctx.dietary_goal is None
    assert "duong_cat" in ctx.avoid_ingredient_keys
    assert ctx.daily_calorie_limit is None
