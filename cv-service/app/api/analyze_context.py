"""Build UserAnalysisContext for personalized image analysis."""
from __future__ import annotations

import json
from typing import List, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.cv_schemas import UserAnalysisContext

logger = get_logger(__name__)


def _parse_json_list(raw: Optional[str], field_name: str) -> List[str]:
    if not raw or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("analyze_context_invalid_json", field=field_name, error=str(exc))
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


async def build_user_analysis_context(
    user_id: Optional[str],
    dietary_preferences: Optional[str],
    avoid_foods: Optional[str],
    recent_dishes: Optional[str],
) -> UserAnalysisContext:
    prefs = _parse_json_list(dietary_preferences, "dietary_preferences")
    avoid = _parse_json_list(avoid_foods, "avoid_foods")
    recent = _parse_json_list(recent_dishes, "recent_dishes")

    if user_id and settings.meal_history_enabled and not recent:
        recent = await _load_recent_food_labels(user_id)

    profile_fields = await _load_user_profile_fields(user_id)

    merged_prefs = list(prefs)
    if profile_fields.get("dietary_goal") and profile_fields["dietary_goal"] not in merged_prefs:
        merged_prefs.append(profile_fields["dietary_goal"])

    merged_avoid_keys = list(profile_fields.get("avoid_ingredient_keys") or [])
    from app.services.allergen_checker import resolve_avoid_keys_from_names

    for key in resolve_avoid_keys_from_names(avoid):
        if key not in merged_avoid_keys:
            merged_avoid_keys.append(key)

    return UserAnalysisContext(
        user_id=user_id,
        dietary_preferences=merged_prefs,
        avoid_foods=avoid,
        recent_dishes=recent,
        allergies=profile_fields.get("allergies") or [],
        allergy_keys=profile_fields.get("allergy_keys") or [],
        health_conditions=profile_fields.get("health_conditions") or [],
        health_condition_keys=profile_fields.get("health_condition_keys") or [],
        dietary_goal=profile_fields.get("dietary_goal"),
        avoid_ingredient_keys=merged_avoid_keys,
        daily_calorie_limit=profile_fields.get("daily_calorie_limit"),
    )


async def _load_user_profile_fields(user_id: Optional[str]) -> dict:
    if not user_id or not settings.user_profile_enabled:
        return {}

    from app.db.user_profile import get_user_cv_profile

    profile = await get_user_cv_profile(user_id)
    if profile is None:
        return {}

    return {
        "allergies": profile.allergy_names,
        "allergy_keys": profile.allergy_keys,
        "health_conditions": profile.health_condition_names,
        "health_condition_keys": [c.condition_key for c in profile.health_conditions],
        "dietary_goal": profile.dietary_goal,
        "avoid_ingredient_keys": profile.avoid_ingredient_keys,
        "daily_calorie_limit": profile.daily_calorie_limit,
    }


async def _load_recent_food_labels(user_id: str) -> List[str]:
    from app.db import meal_history as meal_history_service

    try:
        items = await meal_history_service.get_recent(user_id=user_id, limit=5)
    except Exception as exc:
        logger.warning("analyze_context_history_failed", user_id=user_id, error=str(exc))
        return []

    labels: List[str] = []
    seen: set[str] = set()
    for item in items:
        for food in item.foods:
            label = food.food_label_vi or food.food_label_key
            if label and label not in seen:
                seen.add(label)
                labels.append(label)
    return labels[:10]
