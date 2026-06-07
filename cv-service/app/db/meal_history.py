"""
Database operations for meal_history table.

  store_meal(user_id, result)          -> None   (fire-and-forget)
  query_history(user_id, query, limit) -> List[MealHistoryItem]
  get_recent(user_id, limit)           -> List[MealHistoryItem]

Uses service client to bypass RLS for writes.
"""
from __future__ import annotations

import json
from typing import List

from app.core.config import settings
from app.core.logging import get_logger
from app.db.client import embed_text, run_rpc, service_client
from app.schemas.cv_schemas import (
    AnalysisResult,
    FoodNutrition,
    MacroNutrients,
    MealHistoryItem,
)

logger = get_logger(__name__)


# ── Serialisation helpers ────────────────────────────────────

def _build_meal_text(result: AnalysisResult) -> str:
    if result.nutrition_breakdown:
        foods_str = ", ".join(
            f"{f.food_label_key} {f.estimated_grams:.0f}g"
            for f in result.nutrition_breakdown
        )
    else:
        foods_str = "unknown food"
    m = result.total_macros
    return (
        f"meal: {foods_str}. "
        f"calories: {m.calories_kcal:.0f}kcal, "
        f"protein: {m.protein_g:.0f}g, "
        f"carbs: {m.carbs_g:.0f}g, "
        f"fat: {m.fat_g:.0f}g"
    )


def _foods_to_json(breakdown: List[FoodNutrition]) -> str:
    return json.dumps([
        {
            "food_label_key": f.food_label_key,
            "food_label_vi": f.food_label_vi,
            "estimated_grams": f.estimated_grams,
            "macros": {
                "calories_kcal": f.macros.calories_kcal,
                "protein_g": f.macros.protein_g,
                "carbs_g": f.macros.carbs_g,
                "fat_g": f.macros.fat_g,
                "fiber_g": f.macros.fiber_g,
            },
            "usda_fdc_id": f.usda_fdc_id,
        }
        for f in breakdown
    ])


def _macros_to_json(m: MacroNutrients) -> str:
    return json.dumps({
        "calories_kcal": m.calories_kcal,
        "protein_g": m.protein_g,
        "carbs_g": m.carbs_g,
        "fat_g": m.fat_g,
        "fiber_g": m.fiber_g,
    })


def _row_to_item(row: dict, similarity: float = 0.0) -> MealHistoryItem:
    foods_raw = row.get("foods_json") or []
    if isinstance(foods_raw, str):
        foods_raw = json.loads(foods_raw)

    macros_raw = row.get("total_macros") or {}
    if isinstance(macros_raw, str):
        macros_raw = json.loads(macros_raw)

    foods = [
        FoodNutrition(
            food_label_key=f["food_label_key"],
            food_label_vi=f["food_label_vi"],
            estimated_grams=f["estimated_grams"],
            macros=MacroNutrients(**f["macros"]),
            usda_fdc_id=f.get("usda_fdc_id"),
        )
        for f in foods_raw
    ]
    return MealHistoryItem(
        id=str(row["id"]),
        request_id=row["request_id"],
        analyzed_at=str(row["analyzed_at"]),
        foods=foods,
        total_macros=MacroNutrients(**macros_raw),
        similarity=round(similarity, 4),
    )


# ── Public API ───────────────────────────────────────────────

async def store_meal(user_id: str, result: AnalysisResult) -> None:
    """Embed and persist an AnalysisResult. Designed for fire-and-forget use."""
    client = service_client()
    if client is None:
        return

    try:
        meal_text = _build_meal_text(result)
        embedding = await embed_text(meal_text)
        await run_rpc(lambda: client.rpc(
            "store_meal",
            {
                "p_user_id": user_id,
                "p_request_id": result.request_id,
                "p_foods_json": _foods_to_json(result.nutrition_breakdown),
                "p_total_macros": _macros_to_json(result.total_macros),
                "p_meal_text": meal_text,
                "p_embedding": embedding,
            },
        ).json())
        logger.info("meal_stored", user_id=user_id, request_id=result.request_id)

    except Exception as exc:
        logger.warning("meal_store_failed", user_id=user_id, error=str(exc))


async def query_history(
    user_id: str,
    query_text: str,
    limit: int = 10,
) -> List[MealHistoryItem]:
    """Semantic search over a user's meal history."""
    client = service_client()
    if client is None:
        return []

    try:
        embedding = await embed_text(query_text)
        response = await run_rpc(lambda: client.rpc(
            "query_meal_history",
            {
                "p_user_id": user_id,
                "query_embedding": embedding,
                "similarity_threshold": settings.meal_history_similarity_threshold,
                "match_count": limit,
            },
        ).json())
        rows = response or []
        return [_row_to_item(r, r.get("similarity", 0.0)) for r in rows]

    except Exception as exc:
        logger.warning("meal_query_failed", user_id=user_id, error=str(exc))
        return []


async def get_recent(user_id: str, limit: int = 10) -> List[MealHistoryItem]:
    """Return the N most recent meals without vector search."""
    client = service_client()
    if client is None:
        return []

    try:
        response = await run_rpc(lambda: client.rpc(
            "get_recent_meals",
            {"p_user_id": user_id, "match_count": limit},
        ).json())
        rows = response or []
        return [_row_to_item(r, similarity=1.0) for r in rows]

    except Exception as exc:
        logger.warning("meal_recent_failed", user_id=user_id, error=str(exc))
        return []
