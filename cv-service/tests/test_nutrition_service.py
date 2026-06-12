"""Tests for USDA lookup retry behaviour."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.services.nutrition_service import NutritionService, _usda_search_foods


def _mock_response(status_code: int, json_body: dict | None = None, **headers) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = headers
    resp.json.return_value = json_body or {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error",
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


@pytest.mark.asyncio
async def test_usda_search_retries_on_429_then_succeeds():
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.side_effect = [
        _mock_response(429, headers={"Retry-After": "0"}),
        _mock_response(200, {"foods": [{"description": "Chicken", "fdcId": 1}]}),
    ]

    with patch("app.services.nutrition_service.asyncio.sleep", new_callable=AsyncMock):
        foods = await _usda_search_foods(client, "chicken breast", label="uc_ga")

    assert foods is not None
    assert len(foods) == 1
    assert client.get.call_count == 2


@pytest.mark.asyncio
async def test_usda_search_returns_none_after_max_retries():
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = _mock_response(429)

    with patch("app.services.nutrition_service.settings.usda_max_retries", 2), \
         patch("app.services.nutrition_service.asyncio.sleep", new_callable=AsyncMock):
        foods = await _usda_search_foods(client, "chicken breast", label="uc_ga")

    assert foods is None
    assert client.get.call_count == 3


@pytest.mark.asyncio
async def test_fetch_nutrition_falls_back_when_usda_rate_limited():
    service = NutritionService()

    with patch("app.db.food_nutrition.get_verified_food", new_callable=AsyncMock, return_value=None), \
         patch("app.services.redis_cache.get_nutrition", new_callable=AsyncMock, return_value=None), \
         patch("app.db.food_nutrition.match_food", new_callable=AsyncMock, return_value=None), \
         patch(
             "app.services.nutrition_service._usda_search_foods",
             new_callable=AsyncMock,
             return_value=None,
         ), \
         patch("app.services.redis_cache.set_nutrition", new_callable=AsyncMock):
        macros, fdc_id, source, confidence = await service._fetch_nutrition("uc_ga")

    assert source == "fallback"
    assert fdc_id is None
    assert confidence == 0.3
    assert macros.calories_kcal > 0
