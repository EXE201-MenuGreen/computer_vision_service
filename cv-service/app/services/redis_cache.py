"""
Optional Redis-backed nutrition cache with TTL.

  get_nutrition(label)             -> dict | None
  set_nutrition(label, data, ttl)  -> None
  invalidate(label)                -> int   (keys deleted)
  invalidate_all()                 -> int   (keys deleted)

All operations are fire-and-forget safe: errors are caught and logged,
never propagated to callers. If Redis is unavailable the service degrades
gracefully — lookups just fall through to pgvector/USDA/fallback.

Key schema:  nutrition:<label>
Value:       JSON — {macros: {...}, fdc_id: str|null, data_source: str, confidence: float}
"""
from __future__ import annotations

import json
from typing import Any, Optional

import redis.asyncio as aioredis

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_client: Optional[aioredis.Redis] = None

_KEY_PREFIX = "nutrition:"


def _redis() -> aioredis.Redis:
    global _client
    if _client is None:
        _client = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _client


def _enabled() -> bool:
    return settings.nutrition_cache_enabled


async def get_nutrition(label: str) -> Optional[dict[str, Any]]:
    """Return cached nutrition dict or None on miss / error."""
    if not _enabled():
        return None

    try:
        raw = await _redis().get(f"{_KEY_PREFIX}{label}")
        if raw:
            return json.loads(raw)
    except Exception as exc:
        logger.debug("redis_cache_get_failed", label=label, error=str(exc))
    return None


async def set_nutrition(label: str, data: dict[str, Any], ttl: int) -> None:
    """Store nutrition dict with TTL (seconds). No-op on error."""
    if not _enabled():
        return

    try:
        await _redis().setex(f"{_KEY_PREFIX}{label}", ttl, json.dumps(data))
    except Exception as exc:
        logger.debug("redis_cache_set_failed", label=label, error=str(exc))


async def invalidate(label: str) -> int:
    """Delete one cache key. Returns number of keys deleted (0 or 1)."""
    if not _enabled():
        return 0

    try:
        return await _redis().delete(f"{_KEY_PREFIX}{label}")
    except Exception as exc:
        logger.warning("redis_invalidate_failed", label=label, error=str(exc))
        return 0


async def invalidate_all() -> int:
    """Delete all nutrition:* keys. Returns number of keys deleted."""
    if not _enabled():
        return 0

    try:
        r = _redis()
        keys = await r.keys(f"{_KEY_PREFIX}*")
        if not keys:
            return 0
        return await r.delete(*keys)
    except Exception as exc:
        logger.warning("redis_invalidate_all_failed", error=str(exc))
        return 0
