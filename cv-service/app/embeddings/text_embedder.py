"""
Singleton sentence-transformer wrapper for food label text embeddings.

- Lazy-loaded: model is imported only on first use
- Thread-safe singleton with double-checked locking
- Uses normalized embeddings for cosine similarity search
- Caches the loaded model for the lifetime of the process
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
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "loading_embedding_model",
                model=model_name,
                device=settings.embedding_device,
            )
            self._model = SentenceTransformer(model_name, device=settings.embedding_device)
            logger.info("embedding_model_loaded", model=model_name)
        except Exception as exc:
            logger.warning("embedding_model_load_failed", model=model_name, error=str(exc))

    def _zero_vector(self) -> List[float]:
        return [0.0] * self.DIMENSIONS

    def encode(self, text: str) -> List[float]:
        text = (text or "").strip()
        if not text or self._model is None:
            return self._zero_vector()

        try:
            return self._model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
        except Exception as exc:
            logger.warning("embedding_encode_failed", error=str(exc))
            return self._zero_vector()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        texts = [(t or "").strip() for t in texts]
        if not texts:
            return []
        if self._model is None:
            return [self._zero_vector() for _ in texts]

        try:
            return [
                v.tolist()
                for v in self._model.encode(
                    texts,
                    batch_size=settings.embedding_batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            ]
        except Exception as exc:
            logger.warning("embedding_encode_batch_failed", error=str(exc))
            return [self._zero_vector() for _ in texts]


def get_embedder() -> TextEmbedder:
    """Return the shared TextEmbedder singleton, creating it on first call."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = TextEmbedder(settings.embedding_model)
    return _instance
