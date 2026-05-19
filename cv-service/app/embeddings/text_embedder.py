"""
Singleton sentence-transformer wrapper for food label text embeddings.

- Lazy-loaded: model not imported until first encode() call
- Thread-safe via double-checked locking
- encode() returns list[float] — JSON-serialisable, ready for Supabase RPC
"""
from __future__ import annotations

import threading
from typing import List

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_instance: "TextEmbedder | None" = None
_lock = threading.Lock()


class TextEmbedder:
    DIMENSIONS = 384  # all-MiniLM-L6-v2 output size

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer
        logger.info("loading_embedding_model", model=model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("embedding_model_loaded", model=model_name)

    def encode(self, text: str) -> List[float]:
        """Encode a single string into a 384-dim normalised float list."""
        return self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple strings in one forward pass — for the seed script."""
        return [
            v.tolist()
            for v in self._model.encode(
                texts,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
        ]


def get_embedder() -> TextEmbedder:
    """Return the shared TextEmbedder singleton, creating it on first call."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = TextEmbedder(settings.embedding_model)
    return _instance
