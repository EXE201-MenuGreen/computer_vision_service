"""
FastAPI dependencies for authentication.

get_current_user() — verifies Supabase Auth JWT, returns user_id (UUID string).
_decode_user_id()  — non-raising helper used for fire-and-forget history store.
"""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_bearer = HTTPBearer()


def _decode_user_id(token: str) -> Optional[str]:
    """
    Decode a Supabase JWT and return the sub (user UUID).
    Returns None on any error — does NOT raise.
    """
    if not settings.supabase_jwt_secret or not token:
        return None
    try:
        from jose import JWTError, jwt

        payload = jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            options={"verify_aud": False},  # Supabase omits aud claim
        )
        return payload.get("sub")
    except Exception:
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    """
    FastAPI dependency — validates Supabase JWT and returns user_id.

    Raises HTTP 401 if token is missing, expired, or invalid.
    Raises HTTP 503 if SUPABASE_JWT_SECRET is not configured.
    """
    if not settings.supabase_jwt_secret:
        raise HTTPException(
            status_code=503,
            detail="Auth not configured — set SUPABASE_JWT_SECRET",
        )

    user_id = _decode_user_id(credentials.credentials)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return user_id
