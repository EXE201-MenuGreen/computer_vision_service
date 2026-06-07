"""Canonical label mapping between technical keys and Vietnamese display labels."""
from __future__ import annotations

from typing import Dict, Tuple

LABEL_MAP: Dict[str, Tuple[str, str]] = {
    "uc_ga": ("uc_ga", "Ức gà tươi sống"),
    "bong_cai_xanh": ("bong_cai_xanh", "Bông cải xanh (Súp lơ)"),
    "thit_ba_chi": ("thit_ba_chi", "Thịt ba chỉ"),
    "ca_chua": ("ca_chua", "Cà chua"),
    "rau_muong": ("rau_muong", "Rau muống"),
    "dau_oliu": ("dau_oliu", "Dầu ô liu"),
    "unknown": ("unknown", "Không xác định"),
}


def map_label(label_key: str) -> tuple[str, str]:
    return LABEL_MAP.get(label_key, (label_key, label_key.replace("_", " ").title()))
