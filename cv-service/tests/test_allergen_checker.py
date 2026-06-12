"""Tests for dish safety evaluation against user allergies and health conditions."""
from __future__ import annotations

from app.schemas.cv_schemas import UserAnalysisContext
from app.services.allergen_checker import annotate_dishes_safety, evaluate_dish_safety, pick_random_safe_dish


def _seafood_dish() -> dict:
    return {
        "id_mon_an_goi_y": "rec_01",
        "ten_mon_an": "Tôm rang me",
        "nguyen_lieu_su_dung": [
            {"ten": "Tôm sú", "ten_ky_thuat": "tom_su", "khoi_luong_g": 200.0},
        ],
    }


def _chicken_dish() -> dict:
    return {
        "id_mon_an_goi_y": "rec_02",
        "ten_mon_an": "Ức gà nướng",
        "nguyen_lieu_su_dung": [
            {"ten": "Ức gà", "ten_ky_thuat": "uc_ga", "khoi_luong_g": 200.0},
        ],
    }


def test_seafood_allergy_marks_dish_unsafe():
    ctx = UserAnalysisContext(
        user_id="u1",
        allergies=["Hải sản"],
        allergy_keys=["hai_san"],
    )
    is_safe, matched, warnings = evaluate_dish_safety(_seafood_dish(), ctx)
    assert is_safe is False
    assert "Hải sản" in matched
    assert warnings == []


def test_safe_dish_for_user_without_allergy_match():
    ctx = UserAnalysisContext(
        user_id="u1",
        allergies=["Hải sản"],
        allergy_keys=["hai_san"],
    )
    is_safe, matched, _ = evaluate_dish_safety(_chicken_dish(), ctx)
    assert is_safe is True
    assert matched == []


def test_diabetes_warns_on_sugar_ingredients():
    ctx = UserAnalysisContext(
        user_id="u1",
        health_conditions=["Tiểu đường"],
        health_condition_keys=["tieu_duong"],
    )
    dish = {
        "nguyen_lieu_su_dung": [
            {"ten_ky_thuat": "duong_cat", "ten": "Đường cát", "khoi_luong_g": 10},
        ],
    }
    is_safe, _, warnings = evaluate_dish_safety(dish, ctx)
    assert is_safe is False
    assert any("Tiểu đường" in w for w in warnings)


def test_annotate_dishes_adds_safety_fields():
    ctx = UserAnalysisContext(allergies=["Hải sản"], allergy_keys=["hai_san"])
    dishes = annotate_dishes_safety([_seafood_dish(), _chicken_dish()], ctx)
    unsafe = next(d for d in dishes if d["ten_mon_an"] == "Tôm rang me")
    safe = next(d for d in dishes if d["ten_mon_an"] == "Ức gà nướng")
    assert unsafe["an_toan_cho_user"] is False
    assert safe["an_toan_cho_user"] is True


def test_pick_random_safe_dish_prefers_safe():
    dishes = [
        {"ten_mon_an": "Tôm", "an_toan_cho_user": False},
        {"ten_mon_an": "Gà", "an_toan_cho_user": True},
    ]
    chosen = pick_random_safe_dish(dishes)
    assert chosen["ten_mon_an"] == "Gà"
