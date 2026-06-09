"""
Unified food label catalog — merges legacy label_mapper and nutrition_service dicts.

Responsibilities:
  - Canonical technical keys (snake_case Vietnamese)
  - Vietnamese display names
  - English USDA search terms and EN aliases for AI normalization
  - Per-100g fallback macros
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Optional, Tuple

from app.schemas.cv_schemas import MacroNutrients
from app.services.food_catalog_data import (
    EN_ALIASES_EXTRA,
    FALLBACK_NUTRITION,
    KEY_ALIASES,
    LABEL_VI_OVERRIDES,
    VIET_TO_USDA_QUERY,
)

_DEFAULT_FALLBACK = FALLBACK_NUTRITION["default"]
_UNKNOWN_VI = "Không xác định"


@dataclass(frozen=True)
class FoodEntry:
    key: str
    vi_display: str
    usda_query: str
    fallback_macros: Optional[MacroNutrients]


@dataclass(frozen=True)
class NormalizedLabel:
    """Result of normalizing raw AI or pipeline label text."""

    key: str
    vi_display: str
    matched: bool
    match_method: str  # exact_key | alias | en_alias | fuzzy | fallback


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s_-]", "", text)
    text = re.sub(r"[\s-]+", "_", text)
    return text.strip("_")


def _auto_vi_display(key: str) -> str:
    return key.replace("_", " ").title()


def _build_catalog() -> Dict[str, FoodEntry]:
    # Alias keys are resolved via KEY_ALIASES — not separate catalog entries.
    all_keys = (
        set(LABEL_VI_OVERRIDES)
        | set(VIET_TO_USDA_QUERY)
        | set(FALLBACK_NUTRITION)
        | set(KEY_ALIASES.values())
    )
    all_keys.discard("default")

    catalog: Dict[str, FoodEntry] = {}
    for key in sorted(all_keys):
        catalog[key] = FoodEntry(
            key=key,
            vi_display=LABEL_VI_OVERRIDES.get(key, _auto_vi_display(key)),
            usda_query=VIET_TO_USDA_QUERY.get(key, key.replace("_", " ")),
            fallback_macros=FALLBACK_NUTRITION.get(key),
        )
    return catalog


def _build_en_index(catalog: Dict[str, FoodEntry]) -> Dict[str, str]:
    """Map normalized English/slug tokens -> canonical key."""
    index: Dict[str, str] = {}

    def register(token: str, key: str) -> None:
        token = _slugify(token)
        if token and token not in index:
            index[token] = key

    for key, entry in catalog.items():
        register(key, key)
        register(entry.usda_query, key)
        register(entry.vi_display, key)
        for extra in EN_ALIASES_EXTRA.get(key, ()):
            register(extra, key)

    for alias, canonical in KEY_ALIASES.items():
        register(alias, canonical)
        if canonical in catalog:
            register(catalog[canonical].usda_query, canonical)

    return index


FOOD_CATALOG: Dict[str, FoodEntry] = _build_catalog()
_EN_INDEX: Dict[str, str] = _build_en_index(FOOD_CATALOG)


def resolve_canonical_key(raw_key: str) -> str:
    """Resolve a raw technical key or alias to a catalog key."""
    slug = _slugify(raw_key)
    if not slug:
        return raw_key
    if slug in KEY_ALIASES:
        return KEY_ALIASES[slug]
    if slug in FOOD_CATALOG:
        return slug
    if slug in _EN_INDEX:
        return _EN_INDEX[slug]
    return slug


def get_food_entry(key: str) -> Optional[FoodEntry]:
    canonical = resolve_canonical_key(key)
    return FOOD_CATALOG.get(canonical)


def get_vi_display(key: str) -> str:
    entry = get_food_entry(key)
    if entry is not None:
        return entry.vi_display
    canonical = resolve_canonical_key(key)
    return LABEL_VI_OVERRIDES.get(canonical, _auto_vi_display(canonical))


def get_usda_query(key: str) -> str:
    entry = get_food_entry(key)
    if entry is not None:
        return entry.usda_query
    canonical = resolve_canonical_key(key)
    return VIET_TO_USDA_QUERY.get(canonical, canonical.replace("_", " "))


def get_fallback_macros(key: str) -> MacroNutrients:
    entry = get_food_entry(key)
    if entry is not None and entry.fallback_macros is not None:
        return entry.fallback_macros
    canonical = resolve_canonical_key(key)
    return FALLBACK_NUTRITION.get(canonical, _DEFAULT_FALLBACK)


def has_specific_fallback(key: str) -> bool:
    canonical = resolve_canonical_key(key)
    return canonical in FALLBACK_NUTRITION and canonical != "default"


def map_label(label_key: str) -> Tuple[str, str]:
    """Backward-compatible API from legacy label_mapper."""
    canonical = resolve_canonical_key(label_key)
    return canonical, get_vi_display(canonical)


def normalize_ingredient(
    raw_key: str,
    raw_display: str = "",
    *,
    fuzzy_threshold: float = 0.82,
) -> NormalizedLabel:
    """
    Normalize AI output (English or Vietnamese) to a canonical key + Vietnamese display.

    Match order: exact key -> KEY_ALIASES -> EN index -> fuzzy on EN index -> fallback.
    """
    slug = _slugify(raw_key)
    if slug in KEY_ALIASES:
        canonical = KEY_ALIASES[slug]
        return NormalizedLabel(canonical, get_vi_display(canonical), True, "alias")

    if slug in FOOD_CATALOG:
        entry = FOOD_CATALOG[slug]
        return NormalizedLabel(slug, entry.vi_display, True, "exact_key")

    if slug in _EN_INDEX:
        canonical = _EN_INDEX[slug]
        return NormalizedLabel(canonical, get_vi_display(canonical), True, "en_alias")

    if raw_display.strip():
        display_slug = _slugify(raw_display)
        if display_slug in _EN_INDEX:
            canonical = _EN_INDEX[display_slug]
            return NormalizedLabel(canonical, get_vi_display(canonical), True, "en_alias")

    best_key = ""
    best_ratio = 0.0
    for token, canonical in _EN_INDEX.items():
        ratio = SequenceMatcher(None, slug, token).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_key = canonical
    if best_key and best_ratio >= fuzzy_threshold:
        return NormalizedLabel(best_key, get_vi_display(best_key), True, "fuzzy")

    vi = raw_display.strip() or _auto_vi_display(slug or raw_key)
    return NormalizedLabel(slug or raw_key, vi, False, "fallback")

