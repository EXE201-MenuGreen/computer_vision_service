"""
Database operations for food_nutrition and food_nutrition_verified tables.

  get_verified_food(label)         -> MacroNutrients | None   (Tier 0)
  match_food(label, search_text)   -> (MacroNutrients, fdc_id) | None  (Tier 2)
  upsert_food(...)                 -> None

All functions are no-ops when Supabase is not configured.
supabase-py v2 is synchronous — calls are wrapped in run_in_executor.
"""
from __future__ import annotations

from typing import Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.db.client import anon_client, embed_text, run_rpc
from app.schemas.cv_schemas import MacroNutrients

logger = get_logger(__name__)


async def get_verified_food(label: str) -> Optional[MacroNutrients]:
    """
    Tier 0: exact lookup in admin-curated food_nutrition_verified table.
    Returns MacroNutrients per 100g, or None if label not found.
    """
    client = anon_client()
    if client is None:
        return None

    try:
        rows = await run_rpc(lambda: client.rpc(
            "get_verified_food",
            {"p_label": label},
        ).execute())

        if not rows:
            return None

        row = rows[0]
        logger.debug("verified_hit", label=label)
        return MacroNutrients(
            calories_kcal=row["calories_kcal"],
            protein_g=row["protein_g"],
            carbs_g=row["carbs_g"],
            fat_g=row["fat_g"],
            fiber_g=row.get("fiber_g"),
        )

    except Exception as exc:
        logger.warning("verified_lookup_failed", label=label, error=str(exc))
        return None


async def match_food(
    label: str,
    search_text: str,
) -> Optional[tuple[MacroNutrients, str | None]]:
    """
    pgvector cosine-similarity search for the closest food entry.

    Returns (MacroNutrients per 100g, fdc_id) if similarity exceeds threshold,
    else None. Never raises.
    """
    client = anon_client()
    if client is None:
        return None

    try:
        embedding = await embed_text(search_text)
        rows = await run_rpc(lambda: client.rpc(
            "match_food",
            {
                "query_embedding": embedding,
                "similarity_threshold": settings.vector_similarity_threshold,
                "match_count": 1,
            },
        ).execute())

        if not rows:
            logger.debug("vector_search_no_match", label=label)
            return None

        row = rows[0]
        macros = MacroNutrients(
            calories_kcal=row["calories_kcal"],
            protein_g=row["protein_g"],
            carbs_g=row["carbs_g"],
            fat_g=row["fat_g"],
            fiber_g=row.get("fiber_g"),
        )
        logger.info(
            "vector_search_hit",
            label=label,
            matched=row["label"],
            similarity=round(row["similarity"], 4),
        )
        return macros, row.get("fdc_id")

    except Exception as exc:
        logger.warning("supabase_match_failed", label=label, error=str(exc))
        return None


async def upsert_food(
    label: str,
    display_name: str,
    search_text: str,
    macros: MacroNutrients,
    fdc_id: Optional[str] = None,
    source: str = "usda",
) -> None:
    """
    Insert or update a food entry after a successful USDA lookup.
    Fire-and-forget: errors are logged but never raised.
    """
    client = anon_client()
    if client is None:
        return

    try:
        embedding = await embed_text(search_text)
        await run_rpc(lambda: client.rpc(
            "upsert_food",
            {
                "p_label": label,
                "p_display_name": display_name,
                "p_search_text": search_text,
                "p_embedding": embedding,
                "p_calories_kcal": macros.calories_kcal,
                "p_protein_g": macros.protein_g,
                "p_carbs_g": macros.carbs_g,
                "p_fat_g": macros.fat_g,
                "p_fiber_g": macros.fiber_g,
                "p_fdc_id": fdc_id,
                "p_source": source,
            },
        ).execute())
        logger.debug("upsert_ok", label=label, source=source)

    except Exception as exc:
        logger.warning("supabase_upsert_failed", label=label, error=str(exc))
