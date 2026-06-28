"""
NutritionService: stateless lookup chain. Database-owned nutrition tiers are disabled here.

  Tier 1  redis      confidence=—      Shared Redis cache (TTL=24h)
  Tier 2  usda       confidence=0.75  USDA FoodData Central API + name validation
  Tier 3  fallback   confidence=0.30  Hardcoded approximate dict

Rules:
  - Redis is checked first; a hit returns the source/confidence of original tier.
  - USDA results are accepted only when the returned food name is semantically close
    enough to the query (controlled by USDA_NAME_MATCH_THRESHOLD).
  - USDA and fallback results are stored in Redis with a TTL.
"""
from __future__ import annotations

import asyncio
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.cv_schemas import DetectedFood, FoodNutrition, MacroNutrients
from app.services import redis_cache
from app.services.food_labels import (
    get_fallback_macros,
    get_usda_query,
    get_vi_display,
    has_specific_fallback,
    resolve_canonical_key,
)

logger = get_logger(__name__)

# USDA FoodData Central nutrient IDs
_NUT_ENERGY  = 1008
_NUT_PROTEIN = 1003
_NUT_CARBS   = 1005
_NUT_FAT     = 1004
_NUT_FIBER   = 1079

# Confidence constants per tier
_CONF = {
    "usda":      0.75,
    "fallback":  0.30,
}

# ── Internal type alias ──────────────────────────────────────
_LookupResult = Tuple[MacroNutrients, Optional[str], str, float]
# (macros_per_100g, fdc_id, data_source, confidence)


def _scale(macro: MacroNutrients, grams: float) -> MacroNutrients:
    ratio = grams / 100.0
    return MacroNutrients(
        calories_kcal=round(macro.calories_kcal * ratio, 1),
        protein_g=round(macro.protein_g * ratio, 1),
        carbs_g=round(macro.carbs_g * ratio, 1),
        fat_g=round(macro.fat_g * ratio, 1),
        fiber_g=round((macro.fiber_g or 0) * ratio, 1) if macro.fiber_g is not None else None,
    )


def _extract_macros(food_json: dict) -> MacroNutrients:
    nutrients = {n["nutrientId"]: n.get("value", 0)
                 for n in food_json.get("foodNutrients", [])}
    return MacroNutrients(
        calories_kcal=nutrients.get(_NUT_ENERGY, 200),
        protein_g=nutrients.get(_NUT_PROTEIN, 0),
        carbs_g=nutrients.get(_NUT_CARBS, 0),
        fat_g=nutrients.get(_NUT_FAT, 0),
        fiber_g=nutrients.get(_NUT_FIBER),
    )


def _usda_name_matches(query_text: str, usda_description: str) -> bool:
    """
    Accept a USDA result only if the returned food description is semantically
    close enough to what we queried.  Two acceptance criteria (OR logic):
      1. SequenceMatcher ratio >= threshold
      2. At least one meaningful word (≥4 chars) from query_text appears in
         the USDA description
    This prevents cases like query="tom_su" (shrimp) matching "Tomato sauce".
    """
    threshold = settings.usda_name_match_threshold
    q = query_text.lower()
    d = usda_description.lower()
    ratio = SequenceMatcher(None, q, d).ratio()
    word_hit = any(w in d for w in q.split() if len(w) >= 4)
    return ratio >= threshold or word_hit


def _macros_to_cache_dict(
    macros: MacroNutrients,
    fdc_id: Optional[str],
    data_source: str,
    confidence: float,
) -> dict:
    return {
        "macros": {
            "calories_kcal": macros.calories_kcal,
            "protein_g": macros.protein_g,
            "carbs_g": macros.carbs_g,
            "fat_g": macros.fat_g,
            "fiber_g": macros.fiber_g,
        },
        "fdc_id": fdc_id,
        "data_source": data_source,
        "confidence": confidence,
    }


def _macros_from_cache_dict(data: dict) -> _LookupResult:
    return (
        MacroNutrients(**data["macros"]),
        data.get("fdc_id"),
        data.get("data_source", "unknown"),
        data.get("confidence", 1.0),
    )


def _usda_retry_delay_seconds(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(float(retry_after), 0.0)
        except ValueError:
            pass
    return settings.usda_retry_backoff_seconds * (2 ** attempt)


async def _usda_search_foods(
    client: httpx.AsyncClient,
    search_text: str,
    *,
    label: str,
) -> Optional[list[dict]]:
    """
    Call USDA /foods/search with retry on HTTP 429 (Too Many Requests).
    Returns the foods list on success, or None when rate-limited / exhausted retries.
    """
    url = f"{settings.usda_base_url}/foods/search"
    params = {
        "query": search_text,
        "api_key": settings.usda_api_key,
        "pageSize": 1,
        "dataType": "Foundation,SR Legacy",
    }
    max_retries = settings.usda_max_retries

    for attempt in range(max_retries + 1):
        resp = await client.get(url, params=params)

        if resp.status_code == 429:
            if attempt >= max_retries:
                logger.warning(
                    "nutrition_usda_rate_limited",
                    label=label,
                    attempts=attempt + 1,
                )
                return None

            delay = _usda_retry_delay_seconds(resp, attempt)
            logger.info(
                "nutrition_usda_retry",
                label=label,
                attempt=attempt + 1,
                wait_seconds=delay,
            )
            await asyncio.sleep(delay)
            continue

        resp.raise_for_status()
        return resp.json().get("foods", [])

    return None


class NutritionService:

    async def _fetch_nutrition(self, query: str) -> _LookupResult:
        """Lookup nutrition without touching any database."""
        canonical = resolve_canonical_key(query)
        search_text = get_usda_query(canonical)

        # Tier 1: Redis cache
        cached = await redis_cache.get_nutrition(canonical)
        if cached is not None:
            logger.debug("nutrition_tier1_cache", label=canonical,
                         source=cached.get("data_source"))
            return _macros_from_cache_dict(cached)

        # Tier 2: USDA FoodData Central
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                foods = await _usda_search_foods(client, search_text, label=canonical)

                if foods:
                    food = foods[0]
                    usda_desc = food.get("description", "")

                    if _usda_name_matches(search_text, usda_desc):
                        macros = _extract_macros(food)
                        fdc_id = str(food.get("fdcId"))
                        result = (macros, fdc_id, "usda", _CONF["usda"])

                        await redis_cache.set_nutrition(
                            canonical,
                            _macros_to_cache_dict(*result),
                            settings.nutrition_cache_ttl,
                        )
                        logger.info("nutrition_tier3_usda", label=canonical,
                                    usda_desc=usda_desc, fdc_id=fdc_id)
                        return result
                    else:
                        logger.warning(
                            "nutrition_usda_name_mismatch",
                            label=canonical,
                            search_text=search_text,
                            usda_returned=usda_desc,
                            action="rejected_falling_to_fallback",
                        )
                else:
                    logger.warning("nutrition_usda_no_results",
                                   label=canonical, search_text=search_text)

        except Exception as exc:
            logger.warning("nutrition_usda_failed", label=canonical, error=str(exc))

        # Tier 3: hardcoded fallback
        fallback = get_fallback_macros(canonical)
        logger.warning(
            "nutrition_tier4_fallback",
            label=canonical,
            search_text=search_text,
            has_specific_entry=has_specific_fallback(canonical),
            action="using_static_fallback",
        )
        result = (fallback, None, "fallback", _CONF["fallback"])
        await redis_cache.set_nutrition(
            canonical, _macros_to_cache_dict(*result), settings.nutrition_cache_ttl
        )
        return result

    async def lookup_batch(self, detected: List[DetectedFood]) -> List[FoodNutrition]:
        results = []
        for food in detected:
            canonical = resolve_canonical_key(food.ten_nguyen_lieu_ky_thuat)
            macros_per_100g, fdc_id, data_source, confidence = \
                await self._fetch_nutrition(canonical)
            scaled = _scale(macros_per_100g, food.estimated_grams)
            results.append(FoodNutrition(
                food_label_key=canonical,
                food_label_vi=get_vi_display(canonical),
                estimated_grams=food.estimated_grams,
                macros=scaled,
                usda_fdc_id=fdc_id,
                data_source=data_source,
                confidence=confidence,
            ))
        return results

    def sum_macros(self, breakdown: List[FoodNutrition]) -> MacroNutrients:
        return MacroNutrients(
            calories_kcal=round(sum(f.macros.calories_kcal for f in breakdown), 1),
            protein_g=round(sum(f.macros.protein_g for f in breakdown), 1),
            carbs_g=round(sum(f.macros.carbs_g for f in breakdown), 1),
            fat_g=round(sum(f.macros.fat_g for f in breakdown), 1),
            fiber_g=round(sum(f.macros.fiber_g or 0 for f in breakdown), 1),
        )


# Singleton
nutrition_service = NutritionService()
