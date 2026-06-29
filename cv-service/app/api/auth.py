"""API key authentication helpers for backend → AI service calls."""
from __future__ import annotations

from collections import defaultdict, deque
from hashlib import sha256
from threading import Lock
from time import monotonic

from fastapi import Header, HTTPException, Request, status

from app.core.config import settings

_RATE_WINDOW_SECONDS = 60.0
_rate_lock = Lock()
_rate_buckets: dict[str, deque[float]] = defaultdict(deque)


def _rate_limit_key(request: Request, token: str) -> str:
    if token:
        return f"token:{sha256(token.encode('utf-8')).hexdigest()}"
    client_host = request.client.host if request.client else "unknown"
    return f"ip:{client_host}"


def _check_rate_limit(request: Request, token: str) -> None:
    limit = settings.api_rate_limit_per_minute
    if limit <= 0:
        return

    now = monotonic()
    cutoff = now - _RATE_WINDOW_SECONDS
    key = _rate_limit_key(request, token)

    with _rate_lock:
        bucket = _rate_buckets[key]
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded.",
                headers={"Retry-After": str(int(_RATE_WINDOW_SECONDS))},
            )

        bucket.append(now)


async def require_api_key(
    request: Request,
    authorization: str = Header(default="", alias="Authorization"),
) -> None:
    """
    Validate Authorization: Bearer <key> for requests coming from backend.
    """
    if not settings.auth_enabled:
        return

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

    _check_rate_limit(request, token)
