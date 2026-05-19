"""
Seed script: encode FALLBACK_NUTRITION entries and bulk-upsert to Supabase.

Usage:
    cd cv-service
    python scripts/seed_food_vectors.py

Idempotent — running twice does not create duplicates (ON CONFLICT DO UPDATE).
Requires SUPABASE_URL and SUPABASE_ANON_KEY in .env or environment.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.embeddings.text_embedder import TextEmbedder
from app.services.nutrition_service import FALLBACK_NUTRITION, VIET_TO_USDA_QUERY

setup_logging()
logger = get_logger(__name__)


def _search_text(label: str) -> str:
    return VIET_TO_USDA_QUERY.get(label, label.replace("_", " "))


def seed() -> None:
    if not settings.supabase_url or not settings.supabase_anon_key:
        logger.error("seed_aborted", reason="SUPABASE_URL or SUPABASE_ANON_KEY not set")
        sys.exit(1)

    from supabase import create_client

    client = create_client(settings.supabase_url, settings.supabase_anon_key)
    embedder = TextEmbedder(settings.embedding_model)

    labels = list(FALLBACK_NUTRITION.keys())
    search_texts = [_search_text(label) for label in labels]

    logger.info("encoding_fallback_entries", count=len(labels))
    embeddings = embedder.encode_batch(search_texts)

    upserted = 0
    errors = 0

    for label, search_text, embedding in zip(labels, search_texts, embeddings):
        macros = FALLBACK_NUTRITION[label]
        try:
            client.rpc(
                "upsert_food",
                {
                    "p_label": label,
                    "p_display_name": search_text.title(),
                    "p_search_text": search_text,
                    "p_embedding": embedding,
                    "p_calories_kcal": macros.calories_kcal,
                    "p_protein_g": macros.protein_g,
                    "p_carbs_g": macros.carbs_g,
                    "p_fat_g": macros.fat_g,
                    "p_fiber_g": macros.fiber_g,
                    "p_fdc_id": None,
                    "p_source": "fallback",
                },
            ).execute()
            upserted += 1
            logger.debug("seeded", label=label)
        except Exception as exc:
            logger.error("seed_row_failed", label=label, error=str(exc))
            errors += 1

    logger.info("seed_complete", upserted=upserted, errors=errors, total=len(labels))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    seed()
