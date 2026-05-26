"""
Supabase client factory + shared async helpers.

Clients:
  anon_client()     — anon key, reads + food_nutrition upsert
  service_client()  — service_role key, bypasses RLS for meal_history writes

Helpers:
  run_rpc(client, rpc_fn)    — execute a sync Supabase RPC in executor, return data
  embed_text(text)           — encode text to vector in executor
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, List, Optional

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_anon: Optional[object] = None
_service: Optional[object] = None


def anon_client():
    """Return Supabase anon client, or None if not configured."""
    global _anon
    if _anon is not None:
        return _anon

    if not settings.supabase_url or not settings.supabase_anon_key:
        logger.warning(
            "supabase_anon_not_configured",
            hint="Set SUPABASE_URL and SUPABASE_ANON_KEY",
        )
        return None

    try:
        from supabase import create_client
        _anon = create_client(settings.supabase_url, settings.supabase_anon_key)
        logger.info("supabase_anon_client_ready", url=settings.supabase_url)
        return _anon
    except Exception as exc:
        logger.warning("supabase_anon_client_failed", error=str(exc))
        return None


def service_client():
    """Return Supabase service-role client, or None if not configured."""
    global _service
    if _service is not None:
        return _service

    if not settings.supabase_url or not settings.supabase_service_key:
        logger.warning(
            "supabase_service_not_configured",
            hint="Set SUPABASE_URL and SUPABASE_SERVICE_KEY",
        )
        return None

    try:
        from supabase import create_client
        _service = create_client(settings.supabase_url, settings.supabase_service_key)
        logger.info("supabase_service_client_ready")
        return _service
    except Exception as exc:
        logger.warning("supabase_service_client_failed", error=str(exc))
        return None


async def run_rpc(rpc_fn: Callable[[], Any]) -> Any:
    """
    Execute a synchronous Supabase RPC call in a thread executor.
    Returns response.data, or None on error (caller decides how to handle).
    """
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, rpc_fn)
    return response.data


async def embed_text(text: str) -> List[float]:
    """Encode text to a float vector using the singleton text embedder."""
    from app.embeddings.text_embedder import get_embedder
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_embedder().encode, text)
