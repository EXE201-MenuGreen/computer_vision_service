"""
Load user health profile, allergies, and dietary restrictions from PostgREST.

Expects a single RPC exposed by the MenuGreen backend:

  POST /rpc/get_user_cv_context
  Body: { "p_user_id": "<uuid>" }
  Returns JSON object:
  {
    "allergies": [
      {"allergen_key": "hai_san", "name": "Hải sản", "severity": "high"}
    ],
    "health_conditions": [
      {"condition_key": "tieu_duong", "name": "Tiểu đường", "notes": "Hạn chế đường"}
    ],
    "dietary_goal": "giam_can",
    "avoid_ingredient_keys": ["duong_cat", "bot_ngot"],
    "daily_calorie_limit": 2000,
    "daily_protein_limit": 80,
    "daily_carbs_limit": 200,
    "daily_fat_limit": 65
  }

No-op when PostgREST is not configured.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import get_logger
from app.db.client import run_rpc, service_client

logger = get_logger(__name__)


class UserAllergyRecord(BaseModel):
    allergen_key: str
    name: str
    severity: str = "medium"


class UserHealthCondition(BaseModel):
    condition_key: str
    name: str
    notes: str = ""


class UserCvProfile(BaseModel):
    """Health + allergy context loaded from the main MenuGreen database."""

    allergies: List[UserAllergyRecord] = Field(default_factory=list)
    health_conditions: List[UserHealthCondition] = Field(default_factory=list)
    dietary_goal: Optional[str] = None
    avoid_ingredient_keys: List[str] = Field(default_factory=list)
    daily_calorie_limit: Optional[float] = None
    daily_protein_limit: Optional[float] = None
    daily_carbs_limit: Optional[float] = None
    daily_fat_limit: Optional[float] = None

    @property
    def allergy_keys(self) -> List[str]:
        return [a.allergen_key for a in self.allergies if a.allergen_key]

    @property
    def allergy_names(self) -> List[str]:
        return [a.name for a in self.allergies if a.name]

    @property
    def health_condition_names(self) -> List[str]:
        return [c.name for c in self.health_conditions if c.name]


async def get_user_cv_profile(user_id: str) -> Optional[UserCvProfile]:
    """Fetch user CV context from PostgREST. Returns None if unavailable."""
    if not settings.user_profile_enabled or not user_id:
        return None

    client = service_client()
    if client is None:
        logger.debug("user_profile_skipped", reason="postgrest_not_configured")
        return None

    try:
        response = await run_rpc(lambda: client.rpc(
            "get_user_cv_context",
            {"p_user_id": user_id},
        ).json())

        if not response:
            return None

        data: dict[str, Any] = response if isinstance(response, dict) else response[0]
        profile = UserCvProfile(
            allergies=[UserAllergyRecord(**a) for a in data.get("allergies") or []],
            health_conditions=[
                UserHealthCondition(**c) for c in data.get("health_conditions") or []
            ],
            dietary_goal=data.get("dietary_goal"),
            avoid_ingredient_keys=list(data.get("avoid_ingredient_keys") or []),
            daily_calorie_limit=data.get("daily_calorie_limit"),
            daily_protein_limit=data.get("daily_protein_limit"),
            daily_carbs_limit=data.get("daily_carbs_limit"),
            daily_fat_limit=data.get("daily_fat_limit"),
        )
        logger.info(
            "user_profile_loaded",
            user_id=user_id,
            allergies=len(profile.allergies),
            conditions=len(profile.health_conditions),
        )
        return profile

    except Exception as exc:
        logger.warning("user_profile_load_failed", user_id=user_id, error=str(exc))
        return None
