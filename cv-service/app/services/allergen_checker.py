"""
Evaluate dish safety against user allergies, avoid lists, and health conditions.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

from app.schemas.cv_schemas import UserAnalysisContext

# Ingredient technical key -> allergen category keys (align with Allergy.allergen_key in backend)
INGREDIENT_ALLERGEN_MAP: Dict[str, List[str]] = {
    "tom_su": ["hai_san"],
    "tom_dat": ["hai_san"],
    "tom_the": ["hai_san"],
    "cua_dong": ["hai_san"],
    "cua_bien": ["hai_san"],
    "ghe": ["hai_san"],
    "muc_la": ["hai_san"],
    "muc_ong": ["hai_san"],
    "muc_trung": ["hai_san"],
    "ngheu": ["hai_san"],
    "so_huyet": ["hai_san"],
    "hen": ["hai_san"],
    "ca_loc": ["hai_san"],
    "ca_tre": ["hai_san"],
    "trung_ga": ["trung"],
    "trung_vit": ["trung"],
    "trung_cut": ["trung"],
    "trung_vit_lon": ["trung"],
    "trung_muoi": ["trung"],
    "dau_phong": ["dau_phong"],
    "dau_hu_trang": ["dau_nanh"],
    "dau_hu_chien": ["dau_nanh"],
    "tau_hu_ky": ["dau_nanh"],
    "com": ["gluten"],
    "bun": ["gluten"],
    "banh_pho": ["gluten"],
    "banh_mi": ["gluten"],
    "mi_trung": ["gluten", "trung"],
    "bot_gao": ["gluten"],
    "duong_cat": ["duong"],
    "duong_phen": ["duong"],
    "duong_thot_not": ["duong"],
}

# Health condition key -> ingredient keys that should trigger a warning
CONDITION_AVOID_INGREDIENTS: Dict[str, List[str]] = {
    "tieu_duong": ["duong_cat", "duong_phen", "duong_thot_not", "bot_ngot", "gao_te", "com"],
    "tang_huyet_ap": ["muoi_hat", "muoi_ham", "nuoc_mam", "nuoc_mam_nhi", "mam_tom"],
    "gout": ["thit_ba_chi", "gan_bo", "mong_gio", "long_heo"],
    "suy_than": ["nuoc_mam", "muoi_hat", "thit_ba_chi"],
}


def _ingredient_keys_from_dish(dish: dict) -> Set[str]:
    keys: Set[str] = set()
    for ing in dish.get("nguyen_lieu_su_dung") or []:
        key = (ing.get("ten_ky_thuat") or "").strip()
        if key:
            keys.add(key)
    return keys


def _allergens_for_ingredient(ingredient_key: str) -> Set[str]:
    return set(INGREDIENT_ALLERGEN_MAP.get(ingredient_key, []))


def evaluate_dish_safety(
    dish: dict,
    context: UserAnalysisContext | None,
) -> Tuple[bool, List[str], List[str]]:
    """
    Returns (is_safe, matched_allergen_names, health_warnings).
    """
    if context is None:
        return True, [], []

    ingredient_keys = _ingredient_keys_from_dish(dish)
    user_allergen_keys = set(context.allergy_keys)
    avoid_keys = set(context.avoid_ingredient_keys) | set(
        resolve_avoid_keys_from_names(context.avoid_foods)
    )

    matched_allergens: List[str] = []
    allergen_key_to_name = dict(zip(context.allergy_keys, context.allergies))

    for ing_key in ingredient_keys:
        if ing_key in avoid_keys:
            matched_allergens.append(f"tránh: {ing_key}")

        for allergen_key in _allergens_for_ingredient(ing_key):
            if allergen_key in user_allergen_keys:
                display = allergen_key_to_name.get(allergen_key, allergen_key)
                if display not in matched_allergens:
                    matched_allergens.append(display)

    health_warnings: List[str] = []
    condition_keys = set(context.health_condition_keys)
    for condition_key in condition_keys:
        risky = set(CONDITION_AVOID_INGREDIENTS.get(condition_key, []))
        overlap = ingredient_keys & risky
        if overlap:
            name = next(
                (c for c, k in zip(context.health_conditions, context.health_condition_keys)
                 if k == condition_key),
                condition_key,
            )
            health_warnings.append(
                f"{name}: chứa {', '.join(sorted(overlap))}"
            )

    is_safe = len(matched_allergens) == 0 and len(health_warnings) == 0
    return is_safe, matched_allergens, health_warnings


def resolve_avoid_keys_from_names(avoid_foods: List[str]) -> List[str]:
    """Best-effort map free-text avoid list to catalog keys (already lowercase slug)."""
    from app.services.food_labels import resolve_canonical_key

    return [resolve_canonical_key(name) for name in avoid_foods if name.strip()]


def annotate_dishes_safety(
    dishes: List[dict],
    context: UserAnalysisContext | None,
) -> List[dict]:
    annotated: List[dict] = []
    for dish in dishes:
        is_safe, allergens, warnings = evaluate_dish_safety(dish, context)
        annotated.append({
            **dish,
            "an_toan_cho_user": is_safe,
            "dich_ung_trung": allergens,
            "canh_bao_suc_khoe": warnings,
        })
    return annotated


def pick_random_safe_dish(dishes: List[dict]) -> dict | None:
    """Prefer dishes marked safe; fall back to full list if none are safe."""
    import random

    if not dishes:
        return None
    safe = [d for d in dishes if d.get("an_toan_cho_user", True)]
    pool = safe if safe else dishes
    return random.choice(pool)
