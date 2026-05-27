"""
Admin endpoints for cache management and verified nutrition entries.

All routes require the X-Admin-Key header matching ADMIN_API_KEY env var.
Set ADMIN_API_KEY="" (empty) to disable admin routes entirely.

  POST /cv/admin/cache/clear           — flush all nutrition Redis keys
  POST /cv/admin/cache/clear/{label}   — flush one label
  POST /cv/admin/verified              — upsert an admin-verified food entry
  DELETE /cv/admin/verified/{label}    — remove a verified entry
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Path, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import get_logger
from app.db.client import run_rpc, service_client
from app.services import redis_cache

logger = get_logger(__name__)

router = APIRouter(prefix="/cv/admin", tags=["admin"])


# ── Auth dependency ──────────────────────────────────────────

def _require_admin(x_admin_key: str = Header(..., alias="X-Admin-Key")) -> None:
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API is disabled (ADMIN_API_KEY not configured).",
        )
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin key.",
        )


# ── Cache management ─────────────────────────────────────────

@router.post(
    "/cache/clear",
    summary="Flush all nutrition cache entries",
    dependencies=[Depends(_require_admin)],
)
async def clear_all_cache() -> dict:
    deleted = await redis_cache.invalidate_all()
    logger.info("admin_cache_cleared_all", deleted=deleted)
    return {"deleted": deleted, "message": f"Cleared {deleted} cache entries."}


@router.post(
    "/cache/clear/{label}",
    summary="Flush nutrition cache for a single food label",
    dependencies=[Depends(_require_admin)],
)
async def clear_cache_label(
    label: str = Path(..., description="Food label key, e.g. 'tom_su'"),
) -> dict:
    deleted = await redis_cache.invalidate(label)
    logger.info("admin_cache_cleared_label", label=label, deleted=deleted)
    return {
        "label": label,
        "deleted": deleted,
        "message": "Cache entry removed." if deleted else "Key not found in cache.",
    }


# ── Verified nutrition table ─────────────────────────────────

class VerifiedFoodRequest(BaseModel):
    food_label: str = Field(..., description="Exact label used by YOLO / pipeline")
    calories_kcal: float = Field(..., ge=0)
    protein_g: float = Field(..., ge=0)
    carbs_g: float = Field(..., ge=0)
    fat_g: float = Field(..., ge=0)
    fiber_g: Optional[float] = Field(None, ge=0)
    verified_by: Optional[str] = None
    notes: Optional[str] = None


@router.post(
    "/verified",
    status_code=status.HTTP_201_CREATED,
    summary="Upsert an admin-verified food entry (Tier 0)",
    dependencies=[Depends(_require_admin)],
)
async def upsert_verified(body: VerifiedFoodRequest) -> dict:
    client = service_client()
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PostgREST service client not configured.",
        )

    await run_rpc(lambda: client.rpc(
        "upsert_verified_food",
        {
            "p_label":         body.food_label,
            "p_calories_kcal": body.calories_kcal,
            "p_protein_g":     body.protein_g,
            "p_carbs_g":       body.carbs_g,
            "p_fat_g":         body.fat_g,
            "p_fiber_g":       body.fiber_g,
            "p_verified_by":   body.verified_by,
            "p_notes":         body.notes,
        },
    ).json())

    # Invalidate Redis so next lookup reads the fresh verified entry immediately
    await redis_cache.invalidate(body.food_label)

    logger.info("admin_verified_upserted",
                label=body.food_label, by=body.verified_by)
    return {
        "food_label": body.food_label,
        "message": "Verified entry saved. Redis cache invalidated.",
    }


@router.delete(
    "/verified/{label}",
    summary="Remove an admin-verified food entry",
    dependencies=[Depends(_require_admin)],
)
async def delete_verified(
    label: str = Path(..., description="Food label to remove from verified table"),
) -> dict:
    client = service_client()
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase service client not configured.",
        )

    await run_rpc(lambda: client.table("food_nutrition_verified")
                  .delete()
                  .eq("food_label", label)
                  .execute())

    await redis_cache.invalidate(label)

    logger.info("admin_verified_deleted", label=label)
    return {
        "food_label": label,
        "message": "Verified entry removed. Redis cache invalidated.",
    }
