"""Build UserAnalysisContext from request-provided personalization fields."""
from __future__ import annotations

import json
from typing import List, Optional

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

    merged_prefs = list(prefs)
    merged_avoid_keys: List[str] = []
    from app.services.allergen_checker import resolve_avoid_keys_from_names

    for key in resolve_avoid_keys_from_names(avoid):
        if key not in merged_avoid_keys:
            merged_avoid_keys.append(key)

    return UserAnalysisContext(
        user_id=user_id,
        dietary_preferences=merged_prefs,
        avoid_foods=avoid,
        recent_dishes=recent,
        allergies=[],
        allergy_keys=[],
        health_conditions=[],
        health_condition_keys=[],
        dietary_goal=None,
        avoid_ingredient_keys=merged_avoid_keys,
        daily_calorie_limit=None,
    )
