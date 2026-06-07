"""API key authentication helpers for backend → AI service calls."""
from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.core.config import settings


async def require_api_key(authorization: str = Header(default="", alias="Authorization")) -> None:
    """
    Validate Authorization: Bearer <key> for requests coming from backend.
    """
    if not settings.api_secret_key:
        if settings.is_production:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI service API key auth is not configured.",
            )
        return

    prefix = "Bearer "
    if not authorization.startswith(prefix):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token.")

    token = authorization.removeprefix(prefix).strip()
    if token != settings.api_secret_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")
