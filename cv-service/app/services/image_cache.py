"""
Image-hash cache for inference results.

Same image (byte-identical) → same SHA256 → return cached AI response.

Key schema:  image_result:<sha256>
Value:       JSON — full AI inference response (same shape as worker return)
TTL:         configurable via settings.image_cache_ttl_seconds (default 1h)
"""
from __future__ import annotations

import json
from typing import Any, Optional

import redis.asyncio as aioredis

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_client: Optional[aioredis.Redis] = None
_KEY_PREFIX = "image_result:"


def _get_client() -> Optional[aioredis.Redis]:
    """Lazy-init Redis client. Returns None if Redis URL not configured."""
    global _client
    if _client is not None:
        return _client
    url = settings.redis_url
    if not url:
        return None
    try:
        _client = aioredis.from_url(url, socket_connect_timeout=2)
        return _client
    except Exception as exc:
        logger.warning("image_cache_redis_init_failed", error=str(exc))
        return None


def hash_image(image_bytes: bytes) -> str:
    """SHA256 hex digest of image bytes."""
    import hashlib
    return hashlib.sha256(image_bytes).hexdigest()


def _key(image_hash: str, namespace: str | None = None) -> str:
    # No namespace intentionally preserves the deployed ingredient-scan key.
    return f"{_KEY_PREFIX}{namespace}:{image_hash}" if namespace else f"{_KEY_PREFIX}{image_hash}"


async def get_cached_result(image_hash: str, namespace: str | None = None) -> Optional[dict[str, Any]]:
    """Return cached result dict or None on miss / any error."""
    if settings.image_cache_ttl_seconds <= 0:
        return None
    client = _get_client()
    if client is None:
        return None
    try:
        raw = await client.get(_key(image_hash, namespace))
        if raw is None:
            logger.debug("image_cache_miss", image_hash_prefix=image_hash[:8], namespace=namespace or "ingredient_scan")
            return None
        logger.info("image_cache_hit", image_hash_prefix=image_hash[:8], namespace=namespace or "ingredient_scan")
        return json.loads(raw)
    except Exception as exc:
        logger.warning("image_cache_get_failed", error=str(exc))
        return None


async def set_cached_result(image_hash: str, result: dict[str, Any], namespace: str | None = None) -> None:
    """Store result for an image hash. Errors are swallowed."""
    if settings.image_cache_ttl_seconds <= 0:
        return
    client = _get_client()
    if client is None:
        return
    ttl = settings.image_cache_ttl_seconds
    try:
        await client.set(
            _key(image_hash, namespace),
            json.dumps(result, ensure_ascii=False),
            ex=ttl,
        )
        logger.debug("image_cache_set", image_hash_prefix=image_hash[:8], ttl=ttl, namespace=namespace or "ingredient_scan")
    except Exception as exc:
        logger.warning("image_cache_set_failed", error=str(exc))
