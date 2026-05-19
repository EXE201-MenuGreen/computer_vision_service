"""
Supabase client for food_nutrition vector search and upsert.

Exposes two coroutines:
  - match_food(label, search_text) -> (MacroNutrients, fdc_id) | None
  - upsert_food(label, display_name, search_text, macros, fdc_id, source) -> None

Both are no-ops when Supabase is not configured, preserving backwards-compatibility.
supabase-py v2 is synchronous — all calls are wrapped in run_in_executor to avoid
blocking the FastAPI event loop.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.cv_schemas import MacroNutrients

logger = get_logger(__name__)

_client = None


def _get_client():
    """Lazily initialise the Supabase client. Returns None if not configured."""
    global _client
    if _client is not None:
        return _client

    if not settings.supabase_url or not settings.supabase_anon_key:
        logger.warning(
            "supabase_not_configured",
            hint="Set SUPABASE_URL and SUPABASE_ANON_KEY to enable vector search",
        )
        return None

    from supabase import create_client
    _client = create_client(settings.supabase_url, settings.supabase_anon_key)
    logger.info("supabase_client_initialised", url=settings.supabase_url)
    return _client


async def match_food(
    label: str,
    search_text: str,
) -> Optional[tuple[MacroNutrients, str | None]]:
    """
    Perform pgvector cosine-similarity search for the closest food entry.

    Returns (MacroNutrients per 100g, fdc_id) if similarity exceeds threshold,
    else None. Never raises — errors are logged and None is returned.
    """
    client = _get_client()
    if client is None:
        return None

    try:
        from app.embeddings.text_embedder import get_embedder

        loop = asyncio.get_event_loop()
        embedding: list = await loop.run_in_executor(
            None, get_embedder().encode, search_text
        )

        def _rpc():
            return client.rpc(
                "match_food",
                {
                    "query_embedding": embedding,
                    "similarity_threshold": settings.vector_similarity_threshold,
                    "match_count": 1,
                },
            ).execute()

        response = await loop.run_in_executor(None, _rpc)
        rows = response.data

        if not rows:
            logger.debug("vector_search_no_match", label=label, search_text=search_text)
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
            matched_label=row["label"],
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
    Insert or update a food entry in Supabase after a successful USDA lookup.

    Fire-and-forget: errors are logged but never raised.
    """
    client = _get_client()
    if client is None:
        return

    try:
        from app.embeddings.text_embedder import get_embedder

        loop = asyncio.get_event_loop()
        embedding: list = await loop.run_in_executor(
            None, get_embedder().encode, search_text
        )

        def _rpc():
            return client.rpc(
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
            ).execute()

        await loop.run_in_executor(None, _rpc)
        logger.debug("supabase_upsert_ok", label=label, source=source)

    except Exception as exc:
        logger.warning("supabase_upsert_failed", label=label, error=str(exc))
