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

    return UserAnalysisContext(
        user_id=user_id,
        dietary_preferences=prefs,
        avoid_foods=avoid,
        recent_dishes=recent,
    )


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
