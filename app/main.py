"""Compliance knowledge graph MCP server."""
from __future__ import annotations

import logging
import traceback

import uvicorn
from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI
from fastmcp import FastMCP
from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.auth import AuthMiddleware
from app.database import init_db
from app.ui import ui_app
from app.tools.compliance import register_compliance_tools
from app.tools.graph import register_graph_tools
from app.tools.ingest import register_ingest_tools
from app.tools.retrieve import register_retrieve_tools
from app.tools.search import register_search_tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_INSTRUCTIONS = """
This is a knowledge graph of AI regulations, standards, and compliance content.

## Tool groups

**Search**
- `search_regulations(query, jurisdiction?, doc_type?, status?, limit?)` — hybrid semantic + keyword search over documents. Returns ranked list with summaries.
- `search_requirements(query, jurisdiction?, obligation_type?, risk_level?, applies_to?, limit?)` — find atomic compliance obligations. obligation_type: MUST/SHOULD/MAY. applies_to: providers, deployers, users, etc.

**Retrieve**
- `list_documents(jurisdiction?, doc_type?, status?)` — see what's in the corpus.
- `get_document(doc_id, include_full_text?)` — fetch document metadata, summary, section/requirement counts.
- `get_sections(doc_id)` — list all sections of a document.
- `get_section(section_id)` — fetch section content + its requirements.
- `get_requirements(doc_id, obligation_type?)` — all requirements for a document, grouped by section.

**Knowledge graph**
- `get_related(doc_id, relationship?)` — graph neighbors. Relationships: CITES, AMENDS, IMPLEMENTS, SUPERSEDES, RELATED_TO, ANALYZED_BY.
- `link_documents(from_id, to_id, relationship, notes?)` — create a relationship edge.
- `get_topic_coverage(topic_name)` — which documents cover a topic.
- `list_topics()` — see the topic taxonomy.
- `tag_document_topic(doc_id, topic_name)` — associate a document with a topic.

**Compliance**
- `get_compliance_checklist(use_case, jurisdiction?, risk_level?, applies_to?)` — MUST requirements relevant to a specific use case, grouped by document.
- `get_timeline(jurisdiction?)` — regulatory milestones in chronological order.
- `compare_jurisdictions(topic, jurisdictions[])` — side-by-side view of how jurisdictions handle a topic.

**Ingestion (admin)**
- `ingest_document(title, doc_type, full_text, jurisdiction?, issuer?, status?, effective_date?, url?, auto_extract?)` — add a document. auto_extract=True uses Claude to extract summary, sections, and requirements automatically.
- `update_document_status(doc_id, status, notes?)` — mark as superseded, enacted, etc.
- `delete_document(doc_id, confirm?)` — remove a document and all its data.

## Jurisdiction codes
EU, US-Federal, US-CO (Colorado), US-TX (Texas), US-CA (California), UK, Global

## Workflow tip
Start with `list_documents()` to see the current corpus, then use `search_regulations()` or
`search_requirements()` to find relevant content. Use `get_related()` to traverse the knowledge
graph and find connected regulations and standards.
""".strip()

mcp = FastMCP("Compliance", instructions=_INSTRUCTIONS)
register_search_tools(mcp)
register_retrieve_tools(mcp)
register_graph_tools(mcp)
register_compliance_tools(mcp)
register_ingest_tools(mcp)

init_db()


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception on %s:\n%s", request.url.path, traceback.format_exc())
    return JSONResponse({"error": "Internal server error"}, status_code=500)


class _MCPCompatMiddleware:
    """Normalises Accept headers and disables response buffering for Cloudflare + Claude Desktop."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")

        # Allow /mcp/<key> as a convenience path for API key in URL
        if path.startswith("/mcp/") and len(path) > 5:
            api_key = path[5:]
            qs = scope.get("query_string", b"")
            qs = (qs + b"&api_key=" + api_key.encode()) if qs else (b"api_key=" + api_key.encode())
            scope = {**scope, "path": "/mcp", "raw_path": b"/mcp", "query_string": qs}

        if not scope.get("path", "").startswith("/mcp"):
            await self.app(scope, receive, send)
            return

        # Normalise Accept header
        raw_headers: list[tuple[bytes, bytes]] = list(scope.get("headers", []))
        accept = ", ".join(v.decode() for k, v in raw_headers if k == b"accept")
        missing = [t for t in ("application/json", "text/event-stream") if t not in accept]
        if missing:
            new_accept = (", ".join(missing) if not accept else f"{accept}, {', '.join(missing)}")
            raw_headers = [(k, v) for k, v in raw_headers if k != b"accept"]
            raw_headers.append((b"accept", new_accept.encode()))
            scope = {**scope, "headers": raw_headers}

        async def _send(message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Accel-Buffering"] = "no"
                headers["Cache-Control"] = "no-cache"
            await send(message)

        await self.app(scope, receive, _send)


app = mcp.http_app(json_response=True, stateless_http=True)
app.add_middleware(AuthMiddleware)
app.add_middleware(_MCPCompatMiddleware)
app.add_exception_handler(Exception, _unhandled_exception_handler)

# Health check (bypasses auth)
_health_api = FastAPI()


@_health_api.get("/")
def health():
    return {"status": "ok"}


app.mount("/health", _health_api)
app.mount("/ui", ui_app)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001)
