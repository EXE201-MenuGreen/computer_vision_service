"""
NutritionService: 5-tier lookup chain with accuracy-first design.

  Tier 0  verified   confidence=1.00  Admin-curated Supabase table (ground truth)
  Tier 1  redis      confidence=—      Shared Redis cache (TTL=24h, serves prior tier result)
  Tier 2  pgvector   confidence=0.90  Semantic vector search, threshold ≥ 0.82
  Tier 3  usda       confidence=0.75  USDA FoodData Central API + name validation
  Tier 4  fallback   confidence=0.30  Hardcoded approximate dict

Rules:
  - Tier 0 always overrides everything and is NOT cached (admin can update any time).
  - Redis is checked after tier 0; a hit returns the source/confidence of original tier.
  - USDA results are accepted only when the returned food name is semantically close
    enough to the query (controlled by USDA_NAME_MATCH_THRESHOLD). Rejected USDA
    results fall through to tier 4 — they are NOT stored in pgvector.
  - Tier 3 & 4 hits are audit-logged so admins can review and promote to tier 0.
  - Every non-verified result is stored in Redis with a TTL so stale data expires
    and tier 2+ is re-checked on the next lookup.
"""
from __future__ import annotations

import asyncio
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.db.food_nutrition import get_verified_food, match_food, upsert_food
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
    "verified": 1.00,
    "pgvector":  0.90,
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


class NutritionService:

    async def _fetch_nutrition(self, query: str) -> _LookupResult:
        """
        5-tier lookup.  Returns (macros_per_100g, fdc_id, data_source, confidence).
        """
        canonical = resolve_canonical_key(query)
        search_text = get_usda_query(canonical)

        # ── Tier 0: admin-verified (always checked first, never cached) ───────
        verified = None
        try:
            verified = await get_verified_food(canonical)
        except Exception as exc:
            logger.warning("nutrition_verified_lookup_failed", label=canonical, error=str(exc))
        if verified is not None:
            logger.info("nutrition_tier0_verified", label=canonical)
            return verified, None, "verified", _CONF["verified"]

        # ── Tier 1: Redis cache (stores result from tier 2/3/4) ───────────────
        cached = await redis_cache.get_nutrition(canonical)
        if cached is not None:
            logger.debug("nutrition_tier1_cache", label=canonical,
                         source=cached.get("data_source"))
            return _macros_from_cache_dict(cached)

        # ── Tier 2: Supabase pgvector semantic search ─────────────────────────
        vector_result = await match_food(label=canonical, search_text=search_text)
        if vector_result is not None:
            macros, fdc_id = vector_result
            result: _LookupResult = (macros, fdc_id, "pgvector", _CONF["pgvector"])
            await redis_cache.set_nutrition(
                canonical, _macros_to_cache_dict(*result), settings.nutrition_cache_ttl
            )
            return result

        # ── Tier 3: USDA FoodData Central ────────────────────────────────────
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{settings.usda_base_url}/foods/search",
                    params={
                        "query": search_text,
                        "api_key": settings.usda_api_key,
                        "pageSize": 1,
                        "dataType": "Foundation,SR Legacy",
                    },
                )
                resp.raise_for_status()
                foods = resp.json().get("foods", [])

                if foods:
                    food = foods[0]
                    usda_desc = food.get("description", "")

                    if _usda_name_matches(search_text, usda_desc):
                        macros = _extract_macros(food)
                        fdc_id = str(food.get("fdcId"))
                        result = (macros, fdc_id, "usda", _CONF["usda"])

                        # Persist to pgvector for future tier-2 hits
                        asyncio.ensure_future(
                            upsert_food(
                                label=canonical,
                                display_name=search_text,
                                search_text=search_text,
                                macros=macros,
                                fdc_id=fdc_id,
                                source="usda",
                            )
                        )
                        await redis_cache.set_nutrition(
                            canonical,
                            _macros_to_cache_dict(*result),
                            settings.nutrition_cache_ttl,
                        )
                        logger.info("nutrition_tier3_usda", label=canonical,
                                    usda_desc=usda_desc, fdc_id=fdc_id)
                        return result
                    else:
                        # Step 4 audit: USDA returned unrelated food — log for admin review
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

        # ── Tier 4: hardcoded fallback ────────────────────────────────────────
        fallback = get_fallback_macros(canonical)
        # Step 4 audit: log every fallback hit for admin review
        logger.warning(
            "nutrition_tier4_fallback",
            label=canonical,
            search_text=search_text,
            has_specific_entry=has_specific_fallback(canonical),
            action="review_and_add_to_verified_table",
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
