"""
Database operations for food_visual_embeddings table.

  store_visual_embedding(...) -> None  (fire-and-forget)
"""
from __future__ import annotations

from typing import List

from app.core.logging import get_logger
from app.db.client import anon_client, run_rpc

logger = get_logger(__name__)


async def store_visual_embedding(
    request_id: str,
    food_label: str,
    confidence: float,
    embedding: List[float],
) -> None:
    """
    Insert a CLIP visual embedding row into Supabase.
    Never raises — errors are logged and swallowed.
    """
    client = anon_client()
    if client is None:
        return

    try:
        await run_rpc(lambda: client.table("food_visual_embeddings").insert({
            "request_id": request_id,
            "food_label": food_label,
            "confidence": confidence,
            "embedding": embedding,
        }).execute())
        logger.debug(
            "visual_embedding_stored",
            request_id=request_id,
            food_label=food_label,
        )

    except Exception as exc:
        logger.warning(
            "visual_store_failed",
            food_label=food_label,
            error=str(exc),
        )
