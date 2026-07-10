"""Tests for image-hash cache (SHA256 → inference result) in Redis."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.services import image_cache
from app.services.image_cache import hash_image


# --------------------------------------------------------------------------- #
# hash_image: pure function, no Redis
# --------------------------------------------------------------------------- #

def test_hash_image_is_sha256_hex_of_bytes():
    data = b"some-image-bytes"
    expected = __import__("hashlib").sha256(data).hexdigest()
    assert hash_image(data) == expected
    assert len(hash_image(data)) == 64
    assert all(c in "0123456789abcdef" for c in hash_image(data))


def test_hash_image_distinguishes_different_bytes():
    a = hash_image(b"image-A")
    b = hash_image(b"image-B")
    assert a != b


def test_hash_image_same_bytes_same_hash():
    data = b"abc123"
    assert hash_image(data) == hash_image(data)


# --------------------------------------------------------------------------- #
# Cache disabled (image_cache_ttl_seconds=0): should never touch Redis
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_get_cached_returns_none_when_disabled():
    with patch.object(image_cache.settings, "image_cache_ttl_seconds", 0), \
         patch("app.services.image_cache.aioredis.from_url") as from_url:
        result = await image_cache.get_cached_result("anyhash")
        assert result is None
        from_url.assert_not_called()


@pytest.mark.asyncio
async def test_set_cached_does_nothing_when_disabled():
    with patch.object(image_cache.settings, "image_cache_ttl_seconds", 0), \
         patch("app.services.image_cache.aioredis.from_url") as from_url:
        await image_cache.set_cached_result("anyhash", {"x": 1})  # no exception
        from_url.assert_not_called()


# --------------------------------------------------------------------------- #
# Cache miss (key not in Redis)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_get_cached_returns_none_on_miss():
    fake_client = _FakeRedis(get_return=None)
    with patch.object(image_cache, "_client", fake_client), \
         patch.object(image_cache.settings, "image_cache_ttl_seconds", 3600):
        result = await image_cache.get_cached_result("missinghash")
        assert result is None


# --------------------------------------------------------------------------- #
# Cache hit (key exists, returns parsed JSON)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_get_cached_returns_parsed_json_on_hit():
    stored = {"nguyen_lieu_tho_quet_duoc": [], "danh_sach_mon_an_goi_y": []}
    fake_client = _FakeRedis(get_return=b'{"nguyen_lieu_tho_quet_duoc":[],"danh_sach_mon_an_goi_y":[]}')
    with patch.object(image_cache, "_client", fake_client), \
         patch.object(image_cache.settings, "image_cache_ttl_seconds", 3600):
        result = await image_cache.get_cached_result("anyhash")
        assert result == stored


# --------------------------------------------------------------------------- #
# set_cached_result: writes JSON with configured TTL
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_set_cached_serializes_and_uses_ttl():
    stored_value = {}
    stored_ttl = {}

    class CaptureRedis:
        async def set(self, key, value, ex=None):
            stored_value["key"] = key
            stored_value["value"] = value
            stored_ttl["ex"] = ex

    fake = CaptureRedis()
    payload = {"a": 1, "b": [1, 2, 3]}
    with patch.object(image_cache, "_client", fake), \
         patch.object(image_cache.settings, "image_cache_ttl_seconds", 7200):
        await image_cache.set_cached_result("hash-xyz", payload)

    assert stored_value["key"] == "image_result:hash-xyz"
    assert stored_ttl["ex"] == 7200
    import json
    assert json.loads(stored_value["value"]) == payload


# --------------------------------------------------------------------------- #
# Failure modes: Redis errors must not propagate
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_get_cached_swallows_redis_errors():
    class BrokenRedis:
        async def get(self, *_a, **_k):
            raise RuntimeError("connection lost")

    with patch.object(image_cache, "_client", BrokenRedis()), \
         patch.object(image_cache.settings, "image_cache_ttl_seconds", 60):
        # Should not raise — returns None on failure
        result = await image_cache.get_cached_result("any")
        assert result is None


@pytest.mark.asyncio
async def test_set_cached_swallows_redis_errors():
    class BrokenRedis:
        async def set(self, *_a, **_k):
            raise RuntimeError("connection lost")

    with patch.object(image_cache, "_client", BrokenRedis()), \
         patch.object(image_cache.settings, "image_cache_ttl_seconds", 60):
        # Should not raise
        await image_cache.set_cached_result("any", {"x": 1})


# --------------------------------------------------------------------------- #
# End-to-end hook in analyze_image: cache hit short-circuits the AI call
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_analyze_image_returns_cache_hit_without_calling_gemini():
    from app.services import inference_client

    cached_response = {"nguyen_lieu_tho_quet_duoc": [], "danh_sach_mon_an_goi_y": [{"id": "cached"}]}
    fake_image_cache = FakeImageCache(get_return=cached_response)

    with patch.object(inference_client.settings, "image_cache_ttl_seconds", 3600), \
         patch.object(inference_client.settings, "ai_provider", "gemini"), \
         patch.object(inference_client, "get_cached_result", fake_image_cache.get_cached_result), \
         patch.object(inference_client, "set_cached_result", fake_image_cache.set_cached_result), \
         patch.object(inference_client, "analyze_image_via_gemini") as gemini:
        result = await inference_client.analyze_image(
            b"some-image", "img.jpg", "image/jpeg"
        )

    assert result == cached_response
    gemini.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_image_cache_miss_calls_gemini_and_stores_result():
    from app.services import inference_client

    fake_image_cache = FakeImageCache(get_return=None)
    gemini_response = {"nguyen_lieu_tho_quet_duoc": [{"ten_nguyen_lieu": "phở bò"}]}

    with patch.object(inference_client.settings, "image_cache_ttl_seconds", 3600), \
         patch.object(inference_client.settings, "ai_provider", "gemini"), \
         patch.object(inference_client, "get_cached_result", fake_image_cache.get_cached_result), \
         patch.object(inference_client, "set_cached_result", fake_image_cache.set_cached_result), \
         patch.object(inference_client, "analyze_image_via_gemini", AsyncReturn(gemini_response)):
        result = await inference_client.analyze_image(
            b"some-image", "img.jpg", "image/jpeg"
        )

    assert result == gemini_response
    assert fake_image_cache.last_set_hash is not None
    assert fake_image_cache.last_set_value == gemini_response


@pytest.mark.asyncio
async def test_analyze_image_disabled_never_uses_cache():
    from app.services import inference_client

    gemini_response = {"nguyen_lieu_tho_quet_duoc": []}

    with patch.object(inference_client.settings, "image_cache_ttl_seconds", 0), \
         patch.object(inference_client.settings, "ai_provider", "gemini"), \
         patch.object(inference_client, "get_cached_result") as get_mock, \
         patch.object(inference_client, "set_cached_result") as set_mock, \
         patch.object(inference_client, "analyze_image_via_gemini", AsyncReturn(gemini_response)):
        result = await inference_client.analyze_image(
            b"some-image", "img.jpg", "image/jpeg"
        )

    assert result == gemini_response
    get_mock.assert_not_called()
    set_mock.assert_not_called()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class _FakeRedis:
    def __init__(self, get_return):
        self._get_return = get_return
        self.set_calls = []

    async def get(self, key):
        return self._get_return

    async def set(self, key, value, ex=None):
        self.set_calls.append((key, value, ex))


class FakeImageCache:
    """Tracks last get/set for cache-miss → Gemini → set assertions."""

    def __init__(self, get_return):
        self._get_return = get_return
        self.last_get_hash = None
        self.last_set_hash = None
        self.last_set_value = None

    async def get_cached_result(self, image_hash):
        self.last_get_hash = image_hash
        return self._get_return

    async def set_cached_result(self, image_hash, value):
        self.last_set_hash = image_hash
        self.last_set_value = value


class AsyncReturn:
    """Async callable returning a fixed value (for mocking async functions)."""

    def __init__(self, value):
        self._value = value
        self.call_count = 0

    async def __call__(self, *args, **kwargs):
        self.call_count += 1
        return self._value
