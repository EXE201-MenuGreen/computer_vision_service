"""
Meal history service — store and query meal analysis results with vector embeddings.

store_meal(user_id, result)          — embed + persist to Supabase (async, fire-and-forget safe)
query_history(user_id, query, limit) — semantic search over user's meal history
get_recent(user_id, limit)           — latest N meals without vector search
"""
from __future__ import annotations

import asyncio
import json
from typing import List, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.cv_schemas import (
    AnalysisResult,
    FoodNutrition,
    MacroNutrients,
    MealHistoryItem,
)

logger = get_logger(__name__)

_service_client = None


def _get_service_client():
    """Supabase client with service role key — bypasses RLS for server-side writes."""
    global _service_client
    if _service_client is not None:
        return _service_client

    if not settings.supabase_url or not settings.supabase_service_key:
        logger.warning(
            "meal_history_not_configured",
            hint="Set SUPABASE_URL and SUPABASE_SERVICE_KEY to enable meal history",
        )
        return None

    from supabase import create_client
    _service_client = create_client(settings.supabase_url, settings.supabase_service_key)
    logger.info("meal_history_service_client_ready")
    return _service_client


def _build_meal_text(result: AnalysisResult) -> str:
    """
    Create a descriptive text string from AnalysisResult for embedding.

    Format: "meal: pho 380g, pizza 200g. calories: 1200kcal, protein: 45g, carbs: 120g, fat: 30g"
    Enables natural language queries like "high protein meal" or "pho dinner".
    """
    if result.nutrition_breakdown:
        foods_str = ", ".join(
            f"{f.food_label} {f.estimated_grams:.0f}g"
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


def _foods_to_json(nutrition_breakdown: List[FoodNutrition]) -> str:
    return json.dumps([
        {
            "food_label": f.food_label,
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
        for f in nutrition_breakdown
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
    """Deserialise a Supabase meal_history row into MealHistoryItem."""
    foods_raw = row.get("foods_json") or []
    if isinstance(foods_raw, str):
        foods_raw = json.loads(foods_raw)

    macros_raw = row.get("total_macros") or {}
    if isinstance(macros_raw, str):
        macros_raw = json.loads(macros_raw)

    foods = [
        FoodNutrition(
            food_label=f["food_label"],
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


async def store_meal(user_id: str, result: AnalysisResult) -> None:
    """
    Embed and persist an AnalysisResult to meal_history.

    Designed to be called with asyncio.ensure_future() — never raises.
    Uses service role client to bypass RLS (server-side operation).
    """
    client = _get_service_client()
    if client is None:
        return

    try:
        from app.embeddings.text_embedder import get_embedder

        meal_text = _build_meal_text(result)
        loop = asyncio.get_event_loop()

        embedding: list = await loop.run_in_executor(
            None, get_embedder().encode, meal_text
        )

        foods_json = _foods_to_json(result.nutrition_breakdown)
        macros_json = _macros_to_json(result.total_macros)

        def _rpc():
            return client.rpc(
                "store_meal",
                {
                    "p_user_id": user_id,
                    "p_request_id": result.request_id,
                    "p_foods_json": foods_json,
                    "p_total_macros": macros_json,
                    "p_meal_text": meal_text,
                    "p_embedding": embedding,
                },
            ).execute()

        await loop.run_in_executor(None, _rpc)
        logger.info(
            "meal_stored",
            user_id=user_id,
            request_id=result.request_id,
            meal_text=meal_text[:80],
        )

    except Exception as exc:
        logger.warning(
            "meal_store_failed",
            user_id=user_id,
            request_id=result.request_id,
            error=str(exc),
        )


async def query_history(
    user_id: str,
    query_text: str,
    limit: int = 10,
) -> List[MealHistoryItem]:
    """
    Semantic search over a user's meal history.

    Embeds query_text and finds meals with cosine similarity
    above settings.meal_history_similarity_threshold.
    """
    client = _get_service_client()
    if client is None:
        return []

    try:
        from app.embeddings.text_embedder import get_embedder

        loop = asyncio.get_event_loop()
        embedding: list = await loop.run_in_executor(
            None, get_embedder().encode, query_text
        )

        def _rpc():
            return client.rpc(
                "query_meal_history",
                {
                    "p_user_id": user_id,
                    "query_embedding": embedding,
                    "similarity_threshold": settings.meal_history_similarity_threshold,
                    "match_count": limit,
                },
            ).execute()

        response = await loop.run_in_executor(None, _rpc)
        rows = response.data or []

        return [_row_to_item(row, row.get("similarity", 0.0)) for row in rows]

    except Exception as exc:
        logger.warning("meal_query_failed", user_id=user_id, error=str(exc))
        return []


async def get_recent(
    user_id: str,
    limit: int = 10,
) -> List[MealHistoryItem]:
    """Return the N most recent meals for a user (no vector search)."""
    client = _get_service_client()
    if client is None:
        return []

    try:
        loop = asyncio.get_event_loop()

        def _rpc():
            return client.rpc(
                "get_recent_meals",
                {"p_user_id": user_id, "match_count": limit},
            ).execute()

        response = await loop.run_in_executor(None, _rpc)
        rows = response.data or []

        return [_row_to_item(row, similarity=1.0) for row in rows]

    except Exception as exc:
        logger.warning("meal_recent_failed", user_id=user_id, error=str(exc))
        return []


# Singleton convenience functions (mirrors nutrition_service pattern)
store_meal_history = store_meal
query_meal_history = query_history
get_recent_meals = get_recent
