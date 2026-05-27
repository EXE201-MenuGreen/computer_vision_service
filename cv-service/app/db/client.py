"""
PostgREST client factory + shared async helpers.

Clients are represented by HTTP base configuration only.
Modules under app.db call PostgREST RPC/table endpoints directly.

Helpers:
  run_rpc(client, rpc_fn)    — execute a sync HTTP request in executor, return data
  embed_text(text)           — encode text to vector in executor
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, List, Optional

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class PostgRESTClient:
    def __init__(self, base_url: str, api_key: str = "", jwt_token: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.jwt_token = jwt_token

    @property
    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["apikey"] = self.api_key
        if self.jwt_token:
            headers["Authorization"] = f"Bearer {self.jwt_token}"
        elif self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def rpc(self, fn_name: str, payload: dict[str, Any]):
        import httpx

        return httpx.post(
            f"{self.base_url}/rpc/{fn_name}",
            json=payload,
            headers=self.headers,
            timeout=10.0,
        )

    def table(self, table_name: str):
        return _PostgRESTTable(self, table_name)


class _PostgRESTTable:
    def __init__(self, client: PostgRESTClient, table_name: str) -> None:
        self.client = client
        self.table_name = table_name
        self._payload: Any = None
        self._filters: list[tuple[str, str, Any]] = []
        self._method = "insert"

    def insert(self, payload: dict[str, Any]):
        self._method = "insert"
        self._payload = payload
        return self

    def delete(self):
        self._method = "delete"
        return self

    def eq(self, column: str, value: Any):
        self._filters.append((column, "eq", value))
        return self

    def execute(self):
        import httpx

        if self._method == "insert":
            response = httpx.post(
                f"{self.client.base_url}/{self.table_name}",
                json=self._payload,
                headers={**self.client.headers, "Prefer": "return=representation"},
                timeout=10.0,
            )
            response.raise_for_status()
            return type("Resp", (), {"data": response.json()})

        params = {}
        for column, op, value in self._filters:
            if op == "eq":
                params[column] = f"eq.{value}"
        response = httpx.delete(
            f"{self.client.base_url}/{self.table_name}",
            params=params,
            headers=self.client.headers,
            timeout=10.0,
        )
        response.raise_for_status()
        return type("Resp", (), {"data": response.json() if response.content else []})


_anon: Optional[PostgRESTClient] = None
_service: Optional[PostgRESTClient] = None


def anon_client():
    """Return PostgREST anon client, or None if not configured."""
    global _anon
    if _anon is not None:
        return _anon

    if not settings.postgrest_url:
        logger.warning("postgrest_anon_not_configured", hint="Set POSTGREST_URL")
        return None

    _anon = PostgRESTClient(settings.postgrest_url, settings.postgrest_api_key)
    logger.info("postgrest_anon_client_ready", url=settings.postgrest_url)
    return _anon


def service_client():
    """Return PostgREST service client, or None if not configured."""
    global _service
    if _service is not None:
        return _service

    if not settings.postgrest_url:
        logger.warning("postgrest_service_not_configured", hint="Set POSTGREST_URL")
        return None

    _service = PostgRESTClient(settings.postgrest_url, settings.postgrest_api_key, settings.postgrest_service_jwt)
    logger.info("postgrest_service_client_ready")
    return _service


async def run_rpc(rpc_fn: Callable[[], Any]) -> Any:
    """
    Execute a synchronous PostgREST RPC call in a thread executor.
    Returns JSON payload, or None on error (caller decides how to handle).
    """
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, rpc_fn)
    return response


async def embed_text(text: str) -> List[float]:
    """Encode text to a float vector using the singleton text embedder."""
    from app.embeddings.text_embedder import get_embedder
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_embedder().encode, text)
