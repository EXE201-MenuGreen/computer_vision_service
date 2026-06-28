"""Tests for optional nutrition Redis cache."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.services import redis_cache


@pytest.mark.asyncio
async def test_cache_disabled_does_not_open_redis_connection():
    with patch("app.services.redis_cache.settings.nutrition_cache_enabled", False), \
         patch("app.services.redis_cache.aioredis.from_url") as from_url:
        assert await redis_cache.get_nutrition("uc_ga") is None
        await redis_cache.set_nutrition("uc_ga", {"macros": {}}, 60)
        assert await redis_cache.invalidate("uc_ga") == 0
        assert await redis_cache.invalidate_all() == 0

    from_url.assert_not_called()
