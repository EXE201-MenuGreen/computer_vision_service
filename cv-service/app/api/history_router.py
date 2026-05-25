"""
Meal history endpoints — store and query meal analysis history.

POST /cv/history/query — semantic search via natural language
GET  /cv/history/me   — recent meals (no vector search needed)

Both endpoints require a valid Supabase Auth JWT in Authorization header.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_current_user
from app.schemas.cv_schemas import MealHistoryItem, MealQueryRequest, MealQueryResponse
from app.db import meal_history as meal_history_service
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/cv/history", tags=["Meal History"])


@router.post("/query", response_model=MealQueryResponse)
async def query_meal_history(
    body: MealQueryRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Semantic search over the authenticated user's meal history.

    Send a natural language query like:
    - "bữa ăn nhiều protein" (high protein meal)
    - "có pho hoặc bún bò" (meals with pho or bun bo)
    - "hôm qua ăn gì" (what I ate yesterday — works if yesterday's meals are stored)

    Returns meals ranked by semantic similarity.
    """
    logger.info("meal_history_query", user_id=user_id, query=body.query[:60])
    results = await meal_history_service.query_history(
        user_id=user_id,
        query_text=body.query,
        limit=body.limit,
    )
    return MealQueryResponse(
        query=body.query,
        results=results,
        total_found=len(results),
    )


@router.get("/me", response_model=MealQueryResponse)
async def get_my_recent_meals(
    limit: int = Query(default=10, ge=1, le=50, description="Number of recent meals to return"),
    user_id: str = Depends(get_current_user),
):
    """
    Return the N most recent meals for the authenticated user.
    No vector search — ordered by analyzed_at descending.
    """
    logger.info("meal_history_recent", user_id=user_id, limit=limit)
    results = await meal_history_service.get_recent(
        user_id=user_id,
        limit=limit,
    )
    return MealQueryResponse(
        query="recent",
        results=results,
        total_found=len(results),
    )
