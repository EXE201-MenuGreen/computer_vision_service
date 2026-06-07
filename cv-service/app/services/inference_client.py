"""
Remote AI inference client.

The AI service can forward images to an external model API using a Bearer token.
This module is the thin integration layer and does not own any business logic.
"""
from __future__ import annotations

import io
from typing import Any

import httpx

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class InferenceClientError(RuntimeError):
    pass


async def analyze_image(image_bytes: bytes, filename: str, content_type: str) -> dict[str, Any]:
    """
    Send an image to the external AI inference API and return JSON.

    Expected contract:
      - POST {AI_API_BASE_URL}/analyze
      - Authorization: Bearer <AI_API_KEY>
      - multipart form field: image
    """
    if not settings.ai_api_base_url or not settings.ai_api_key:
        raise InferenceClientError("AI inference API is not configured")

    timeout = httpx.Timeout(settings.ai_api_timeout_seconds)
    headers = {"Authorization": f"Bearer {settings.ai_api_key}"}
    files = {"image": (filename or "image.jpg", io.BytesIO(image_bytes), content_type or "image/jpeg")}

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        resp = await client.post(f"{settings.ai_api_base_url.rstrip('/')}/analyze", files=files)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("ai_inference_http_error", status_code=resp.status_code, body=resp.text[:500])
            raise InferenceClientError(str(exc)) from exc
        return resp.json()
