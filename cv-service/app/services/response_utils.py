"""
Utilities for normalizing AI API responses and circuit breaker pattern.
"""
from __future__ import annotations

import time
from typing import Any, Optional
from dataclasses import dataclass, field
from threading import Lock

from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Response Normalization ────────────────────────────────────────────────

def normalize_ai_response(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure AI response has all required fields with sensible defaults.
    Prevents Pydantic validation errors from malformed AI responses.
    """
    normalized = raw.copy()

    # Ensure top-level required fields
    normalized.setdefault("api_version", "v1")
    if not normalized.get("job_id"):
        import uuid
        normalized["job_id"] = str(uuid.uuid4())
    if not normalized.get("request_id"):
        import uuid
        normalized["request_id"] = str(uuid.uuid4())
    normalized.setdefault("status", "done")

    # Normalize nguyen_lieu_tho_quet_duoc
    if normalized.get("nguyen_lieu_tho_quet_duoc"):
        normalized["nguyen_lieu_tho_quet_duoc"] = [
            _normalize_ingredient_item(item) for item in normalized["nguyen_lieu_tho_quet_duoc"]
        ]

    # Normalize danh_sach_mon_an_goi_y
    if normalized.get("danh_sach_mon_an_goi_y"):
        normalized["danh_sach_mon_an_goi_y"] = [
            _normalize_suggested_dish(item) for item in normalized["danh_sach_mon_an_goi_y"]
        ]

    # Ensure mon_an_goi_y_chon is normalized if present
    if normalized.get("mon_an_goi_y_chon"):
        normalized["mon_an_goi_y_chon"] = _normalize_suggested_dish(normalized["mon_an_goi_y_chon"])

    # Normalize nutrition_breakdown
    if normalized.get("nutrition_breakdown"):
        normalized["nutrition_breakdown"] = [
            _normalize_food_nutrition(item) for item in normalized["nutrition_breakdown"]
        ]

    return normalized


def _normalize_ingredient_item(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single ingredient item."""
    # Handle Pydantic v2: ensure None values are explicitly set to None, not missing
    do_chinh_xac = item.get("do_chinh_xac_uoc_tinh")
    if do_chinh_xac is None:
        do_chinh_xac = "unknown"
    
    normalized = {
        "id_nguyen_lieu": item.get("id_nguyen_lieu", "unknown"),
        "ten_nguyen_lieu": item.get("ten_nguyen_lieu", "Nguyên liệu không xác định"),
        "ten_nguyen_lieu_ky_thuat": item.get("ten_nguyen_lieu_ky_thuat", "unknown"),
        "khoi_luong_uoc_tinh_g": _safe_float(item.get("khoi_luong_uoc_tinh_g"), 0.0),
        "do_chinh_xac_uoc_tinh": do_chinh_xac,
    }
    return normalized


def _normalize_suggested_dish(dish: dict[str, Any]) -> dict[str, Any]:
    """Normalize a suggested dish item."""
    normalized = {
        "id_mon_an_goi_y": dish.get("id_mon_an_goi_y", "unknown"),
        "ten_mon_an": dish.get("ten_mon_an", "Món không xác định"),
        "ten_mon_an_ky_thuat": dish.get("ten_mon_an_ky_thuat"),
        "mo_ta_ngan": dish.get("mo_ta_ngan", ""),
        "do_kha_thi": dish.get("do_kha_thi", "unknown"),
        "confidence": _safe_float(dish.get("confidence"), 0.5),
        "nguyen_lieu_su_dung": [],
        "thong_tin_dinh_duong_mon_an": _normalize_nutrition_info(dish.get("thong_tin_dinh_duong_mon_an")),
    }

    # Normalize nguyen_lieu_su_dung
    if dish.get("nguyen_lieu_su_dung"):
        normalized["nguyen_lieu_su_dung"] = [
            {
                "ten": ing.get("ten", "Nguyên liệu"),
                "ten_ky_thuat": ing.get("ten_ky_thuat", "unknown"),
                "khoi_luong_g": _safe_float(ing.get("khoi_luong_g"), 0.0),
            }
            for ing in dish["nguyen_lieu_su_dung"]
        ]

    # Copy optional fields
    if dish.get("dich_ung_trung"):
        normalized["dich_ung_trung"] = dish["dich_ung_trung"]
    if dish.get("canh_bao_suc_khoe"):
        normalized["canh_bao_suc_khoe"] = dish["canh_bao_suc_khoe"]

    return normalized


def _normalize_nutrition_info(nutrition: Optional[dict[str, Any]]) -> dict[str, float]:
    """Normalize nutrition info."""
    if not nutrition:
        return {"tong_calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "fiber_g": 0.0}
    return {
        "tong_calories": _safe_float(nutrition.get("tong_calories"), 0.0),
        "protein_g": _safe_float(nutrition.get("protein_g"), 0.0),
        "carbs_g": _safe_float(nutrition.get("carbs_g"), 0.0),
        "fat_g": _safe_float(nutrition.get("fat_g"), 0.0),
        "fiber_g": _safe_float(nutrition.get("fiber_g"), 0.0),
    }


def _normalize_food_nutrition(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize a food nutrition item."""
    normalized = item.copy()
    normalized.setdefault("data_source", "unknown")
    normalized.setdefault("confidence", 1.0)

    # Ensure macros are normalized
    if normalized.get("macros"):
        normalized["macros"] = {
            "calories_kcal": _safe_float(normalized["macros"].get("calories_kcal"), 0.0),
            "protein_g": _safe_float(normalized["macros"].get("protein_g"), 0.0),
            "carbs_g": _safe_float(normalized["macros"].get("carbs_g"), 0.0),
            "fat_g": _safe_float(normalized["macros"].get("fat_g"), 0.0),
            "fiber_g": _safe_float(normalized["macros"].get("fiber_g"), 0.0),
        }

    return normalized


def _safe_float(value: Any, default: float) -> float:
    """Safely convert value to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ── Circuit Breaker ──────────────────────────────────────────────────────

@dataclass
class CircuitBreakerState:
    """State for circuit breaker pattern."""
    failures: int = 0
    last_failure_time: float = 0.0
    is_open: bool = False


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures when AI API is down.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Circuit is tripped, requests fail fast
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitBreakerState()
        self._lock = Lock()
        self._half_open_calls = 0

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        with self._lock:
            if not self._state.is_open:
                return False

            # Check if recovery timeout has passed
            if time.time() - self._state.last_failure_time >= self.recovery_timeout:
                self._state.is_open = False
                self._half_open_calls = 0
                logger.info("circuit_breaker_half_open", recovery_timeout=self.recovery_timeout)
                return False

            return True

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._state.failures = 0
            self._state.is_open = False
            self._half_open_calls = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._state.failures += 1
            self._state.last_failure_time = time.time()

            if self._state.failures >= self.failure_threshold:
                self._state.is_open = True
                logger.warning(
                    "circuit_breaker_opened",
                    failures=self._state.failures,
                    threshold=self.failure_threshold,
                )

    def can_execute(self) -> bool:
        """Check if a call can be executed."""
        if self.is_open:
            # Check if we can move to half-open
            with self._lock:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
        return True

    def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status."""
        with self._lock:
            state = "open" if self._state.is_open else "closed"
            return {
                "state": state,
                "failures": self._state.failures,
                "threshold": self.failure_threshold,
                "recovery_timeout_seconds": self.recovery_timeout,
            }


# Global circuit breaker instance for AI API
ai_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30.0,
    half_open_max_calls=3,
)
