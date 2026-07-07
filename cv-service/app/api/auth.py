"""API key authentication helpers for backend → AI service calls."""
from __future__ import annotations

from collections import defaultdict, deque
from hashlib import sha256
from threading import Lock
from time import monotonic

from fastapi import HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings

_RATE_WINDOW_SECONDS = 60.0
_rate_lock = Lock()
_rate_buckets: dict[str, deque[float]] = defaultdict(deque)
_bearer = HTTPBearer(auto_error=False)


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
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
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

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token.")

    token = credentials.credentials.strip()
    if token != settings.api_secret_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")

    _check_rate_limit(request, token)
