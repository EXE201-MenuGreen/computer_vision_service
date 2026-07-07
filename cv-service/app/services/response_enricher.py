"""
Post-process AI inference JSON: normalize labels (EN→VN) and enrich nutrition.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.cv_schemas import BoundingBox, DetectedFood, MacroNutrients, UserAnalysisContext
from app.services.allergen_checker import annotate_dishes_safety, pick_random_safe_dish
from app.services.food_labels import normalize_ingredient

logger = get_logger(__name__)

_DUMMY_BBOX = BoundingBox(x1=0.0, y1=0.0, x2=0.0, y2=0.0)


def _to_detected_food(key: str, vi_display: str, grams: float) -> DetectedFood:
    return DetectedFood(
        id_nguyen_lieu=key,
        ten_nguyen_lieu_ky_thuat=key,
        ten_nguyen_lieu=vi_display,
        confidence=1.0,
        bbox=_DUMMY_BBOX,
        estimated_grams=max(0.0, grams),
    )


def _macros_to_nutrition_info(macros: MacroNutrients) -> Dict[str, float]:
    return {
        "tong_calories": macros.calories_kcal,
        "protein_g": macros.protein_g,
        "carbs_g": macros.carbs_g,
        "fat_g": macros.fat_g,
        "fiber_g": macros.fiber_g or 0.0,
    }


def _normalize_raw_ingredients(
    items: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[DetectedFood]]:
    normalized: List[Dict[str, Any]] = []
    detected: List[DetectedFood] = []

    for item in items:
        norm = normalize_ingredient(
            item.get("ten_nguyen_lieu_ky_thuat", ""),
            item.get("ten_nguyen_lieu", ""),
        )
        updated = {
            **item,
            "ten_nguyen_lieu_ky_thuat": norm.key,
            "ten_nguyen_lieu": norm.vi_display,
        }
        normalized.append(updated)
        detected.append(
            _to_detected_food(
                norm.key,
                norm.vi_display,
                float(item.get("khoi_luong_uoc_tinh_g") or 0),
            )
        )
        if not norm.matched:
            logger.info(
                "ingredient_label_unmatched",
                raw_key=item.get("ten_nguyen_lieu_ky_thuat"),
                resolved_key=norm.key,
                method=norm.match_method,
            )

    return normalized, detected


def _normalize_dishes(dishes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []

    for dish in dishes:
        dish_copy = dict(dish)
        recipe: List[Dict[str, Any]] = []
        for ing in dish.get("nguyen_lieu_su_dung") or []:
            norm = normalize_ingredient(
                ing.get("ten_ky_thuat", ""),
                ing.get("ten", ""),
            )
            recipe.append({
                **ing,
                "ten_ky_thuat": norm.key,
                "ten": norm.vi_display,
            })
        dish_copy["nguyen_lieu_su_dung"] = recipe

        if dish_copy.get("ten_mon_an_ky_thuat"):
            norm_dish = normalize_ingredient(
                dish_copy["ten_mon_an_ky_thuat"],
                dish_copy.get("ten_mon_an", ""),
            )
            dish_copy["ten_mon_an_ky_thuat"] = norm_dish.key

        result.append(dish_copy)

    return result


async def _enrich_dish_nutrition(dishes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from app.services.nutrition_service import nutrition_service

    enriched: List[Dict[str, Any]] = []

    for dish in dishes:
        dish_copy = dict(dish)
        recipe = dish_copy.get("nguyen_lieu_su_dung") or []
        if not recipe:
            enriched.append(dish_copy)
            continue

        detected = [
            _to_detected_food(
                ing["ten_ky_thuat"],
                ing["ten"],
                float(ing.get("khoi_luong_g") or 0),
            )
            for ing in recipe
        ]
        breakdown = await nutrition_service.lookup_batch(detected)
        total = nutrition_service.sum_macros(breakdown)
        dish_copy["nutrition_breakdown"] = [item.model_dump() for item in breakdown]
        dish_copy["total_macros"] = total.model_dump()
        dish_copy["thong_tin_dinh_duong_mon_an"] = _macros_to_nutrition_info(total)
        enriched.append(dish_copy)

    return enriched


async def enrich_ai_response(
    payload: Dict[str, Any],
    user_context: Optional[UserAnalysisContext] = None,
) -> Dict[str, Any]:
    """
    Normalize ingredient/dish labels and attach catalog-based nutrition data.

    Mutates a copy of *payload* and returns it. Safe to call multiple times.
    """
    if payload.get("status") != "done":
        return payload

    result = dict(payload)

    raw_items = list(result.get("nguyen_lieu_tho_quet_duoc") or [])
    if raw_items:
        normalized_items, detected = _normalize_raw_ingredients(raw_items)
        result["nguyen_lieu_tho_quet_duoc"] = normalized_items

        if settings.nutrition_enrichment_enabled and detected:
            from app.services.nutrition_service import nutrition_service

            breakdown = await nutrition_service.lookup_batch(detected)
            result["nutrition_breakdown"] = [item.model_dump() for item in breakdown]
            result["total_macros"] = nutrition_service.sum_macros(breakdown).model_dump()

    dishes = list(result.get("danh_sach_mon_an_goi_y") or [])
    if dishes:
        dishes = _normalize_dishes(dishes)
        if settings.nutrition_enrichment_enabled:
            dishes = await _enrich_dish_nutrition(dishes)
        if user_context is not None:
            dishes = annotate_dishes_safety(dishes, user_context)
        result["danh_sach_mon_an_goi_y"] = dishes
        chosen = pick_random_safe_dish(dishes) if user_context else random.choice(dishes) if dishes else None
        if chosen is not None:
            result["mon_an_goi_y_chon"] = chosen

    logger.info(
        "ai_response_enriched",
        ingredients=len(result.get("nguyen_lieu_tho_quet_duoc") or []),
        dishes=len(result.get("danh_sach_mon_an_goi_y") or []),
        chosen_dish=(result.get("mon_an_goi_y_chon") or {}).get("ten_mon_an"),
        nutrition=settings.nutrition_enrichment_enabled,
    )
    return result
