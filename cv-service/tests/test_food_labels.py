"""Tests for unified food_labels catalog."""
from __future__ import annotations

from app.services.food_labels import (
    FOOD_CATALOG,
    get_fallback_macros,
    get_usda_query,
    get_vi_display,
    map_label,
    normalize_ingredient,
    resolve_canonical_key,
)


def test_catalog_includes_legacy_label_mapper_keys():
    assert "uc_ga" in FOOD_CATALOG
    assert FOOD_CATALOG["uc_ga"].vi_display == "Ức gà tươi sống"


def test_catalog_includes_nutrition_keys():
    assert "tom_su" in FOOD_CATALOG
    assert get_usda_query("tom_su") == "black tiger shrimp"


def test_resolve_alias_thit_heo_to_pork_belly():
    assert resolve_canonical_key("thit_heo") == "thit_ba_chi"


def test_normalize_english_chicken_breast():
    result = normalize_ingredient("chicken_breast", "Chicken breast")
    assert result.key == "uc_ga"
    assert result.vi_display == "Ức gà tươi sống"
    assert result.matched is True


def test_map_label_backward_compatible():
    key, vi = map_label("nuoc_mam")
    assert key == "nuoc_mam"
    assert vi == "Nước mắm"


def test_fallback_macros_for_unknown_uses_default():
    macros = get_fallback_macros("totally_unknown_food_xyz")
    default = get_fallback_macros("default")
    assert macros.calories_kcal == default.calories_kcal


def test_get_vi_display_for_known_key():
    assert get_vi_display("rau_muong") == "Rau muống"
