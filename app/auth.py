"""Authentication middleware — single static API key."""
from __future__ import annotations

import logging
import os
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

current_user_id: ContextVar[int | None] = ContextVar("current_user_id", default=None)
_SINGLE_USER_ID = 1
_PUBLIC_PREFIXES = ("/health", "/ui")


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if any(request.url.path.startswith(p) for p in _PUBLIC_PREFIXES):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            raw_key = auth_header.removeprefix("Bearer ").strip()
        else:
            raw_key = request.query_params.get("api_key", "").strip()

        expected = os.getenv("COMPLIANCE_API_KEY", "")
        if not expected or not raw_key or raw_key != expected:
            logger.warning("auth failed (%s %s)", request.method, request.url.path)
            return Response(
                "Unauthorized",
                status_code=401,
                media_type="text/plain",
                headers={"WWW-Authenticate": 'Bearer realm="Compliance"'},
            )

        token = current_user_id.set(_SINGLE_USER_ID)
        try:
            return await call_next(request)
        finally:
            current_user_id.reset(token)
