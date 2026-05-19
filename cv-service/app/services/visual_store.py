"""
Visual embedding store — persists CLIP image embeddings to Supabase.

Used by CLIPZeroShotClassifier (Option 2) to build a visual food database
that enables future visual similarity search (Option 3).

CLIP ViT-B/32 produces 512-dim embeddings, stored in the
food_visual_embeddings table defined in 002_visual_embeddings.sql.
"""
from __future__ import annotations

import asyncio
from typing import List, Optional

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


async def store_visual_embedding(
    request_id: str,
    food_label: str,
    confidence: float,
    embedding: List[float],
) -> None:
    """
    Insert a CLIP visual embedding row into Supabase.

    Never raises — any error is logged and swallowed so the pipeline
    response is never delayed by a write failure.
    """
    if not settings.supabase_url or not settings.supabase_anon_key:
        return

    try:
        from supabase import create_client

        loop = asyncio.get_event_loop()

        def _insert():
            client = create_client(settings.supabase_url, settings.supabase_anon_key)
            return (
                client.table("food_visual_embeddings")
                .insert({
                    "request_id": request_id,
                    "food_label": food_label,
                    "confidence": confidence,
                    "embedding": embedding,
                })
                .execute()
            )

        await loop.run_in_executor(None, _insert)
        logger.debug(
            "visual_embedding_stored",
            request_id=request_id,
            food_label=food_label,
        )

    except Exception as exc:
        logger.warning(
            "visual_store_failed",
            request_id=request_id,
            food_label=food_label,
            error=str(exc),
        )
