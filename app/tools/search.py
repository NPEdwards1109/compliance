"""Search tools: semantic + keyword search over regulations and requirements."""
from __future__ import annotations

import json
from typing import Optional

from fastmcp import FastMCP
from sqlalchemy import text

from app.database import get_session, engine
from app.embeddings import cosine_similarity, embed, from_json
from app.models import Document, Requirement


def register_search_tools(mcp: FastMCP) -> None:

    @mcp.tool()
    def search_regulations(
        query: str,
        jurisdiction: Optional[str] = None,
        doc_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search regulations and guidance documents using semantic + keyword matching.

        Returns documents ranked by relevance. Use filters to narrow by jurisdiction
        (e.g. 'EU', 'US-Federal', 'US-CO'), doc_type (regulation, standard, guidance,
        article, enforcement_action), or status (enacted, proposed, draft, superseded).
        """
        with get_session() as session:
            # FTS keyword search
            fts_ids: list[str] = []
            try:
                safe_query = query.replace('"', '""')
                rows = engine.connect().execute(
                    text(
                        "SELECT d.id FROM documents_fts f "
                        "JOIN documents d ON d.rowid = f.rowid "
                        "WHERE documents_fts MATCH :q ORDER BY rank LIMIT 50"
                    ),
                    {"q": safe_query},
                ).fetchall()
                fts_ids = [r[0] for r in rows]
            except Exception:
                pass

            # Vector semantic search
            query_vec = embed(query)
            all_docs = session.query(Document).filter(Document.embedding.isnot(None)).all()

            scored: list[tuple[float, Document]] = []
            seen_ids: set[str] = set()

            for doc in all_docs:
                vec = from_json(doc.embedding)
                if vec is None:
                    continue
                score = cosine_similarity(query_vec, vec)
                # Boost FTS matches
                if doc.id in fts_ids:
                    score = min(score + 0.15, 1.0)
                scored.append((score, doc))
                seen_ids.add(doc.id)

            # Include FTS-only matches (no embedding yet) with a fixed score
            for fts_id in fts_ids:
                if fts_id not in seen_ids:
                    doc = session.get(Document, fts_id)
                    if doc:
                        scored.append((0.5, doc))

            # Apply filters
            def _matches(doc: Document) -> bool:
                if jurisdiction and doc.jurisdiction != jurisdiction:
                    return False
                if doc_type and doc.doc_type != doc_type:
                    return False
                if status and doc.status != status:
                    return False
                return True

            scored = [(s, d) for s, d in scored if _matches(d)]
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:limit]

            return [
                {
                    "id": d.id,
                    "title": d.title,
                    "doc_type": d.doc_type,
                    "jurisdiction": d.jurisdiction,
                    "issuer": d.issuer,
                    "status": d.status,
                    "effective_date": d.effective_date,
                    "url": d.url,
                    "summary": d.summary,
                    "relevance_score": round(s, 3),
                }
                for s, d in top
            ]

    @mcp.tool()
    def search_requirements(
        query: str,
        jurisdiction: Optional[str] = None,
        obligation_type: Optional[str] = None,
        risk_level: Optional[str] = None,
        applies_to: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """Search atomic compliance requirements using semantic + keyword matching.

        Returns requirements ranked by relevance, with doc context.
        - obligation_type: MUST, SHOULD, or MAY
        - risk_level: unacceptable, high, limited, minimal, unspecified
        - applies_to: filter to requirements that apply to a specific entity
          (e.g. 'providers', 'deployers', 'importers', 'users')
        - jurisdiction: filter by the source document's jurisdiction
        """
        with get_session() as session:
            # FTS keyword search over requirement text
            fts_ids: list[str] = []
            try:
                safe_query = query.replace('"', '""')
                rows = engine.connect().execute(
                    text(
                        "SELECT r.id FROM requirements_fts f "
                        "JOIN requirements r ON r.rowid = f.rowid "
                        "WHERE requirements_fts MATCH :q ORDER BY rank LIMIT 100"
                    ),
                    {"q": safe_query},
                ).fetchall()
                fts_ids = [r[0] for r in rows]
            except Exception:
                pass

            # Vector semantic search
            query_vec = embed(query)
            all_reqs = session.query(Requirement).filter(Requirement.embedding.isnot(None)).all()

            scored: list[tuple[float, Requirement]] = []
            seen_ids: set[str] = set()

            for req in all_reqs:
                vec = from_json(req.embedding)
                if vec is None:
                    continue
                score = cosine_similarity(query_vec, vec)
                if req.id in fts_ids:
                    score = min(score + 0.15, 1.0)
                scored.append((score, req))
                seen_ids.add(req.id)

            for fts_id in fts_ids:
                if fts_id not in seen_ids:
                    req = session.get(Requirement, fts_id)
                    if req:
                        scored.append((0.5, req))

            # Apply filters
            def _matches(req: Requirement) -> bool:
                if obligation_type and req.obligation_type != obligation_type:
                    return False
                if risk_level and req.risk_level != risk_level:
                    return False
                if applies_to:
                    req_applies = json.loads(req.applies_to or "[]")
                    if applies_to not in req_applies:
                        return False
                if jurisdiction:
                    doc = session.get(Document, req.doc_id)
                    if not doc or doc.jurisdiction != jurisdiction:
                        return False
                return True

            scored = [(s, r) for s, r in scored if _matches(r)]
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:limit]

            results = []
            for s, req in top:
                doc = session.get(Document, req.doc_id)
                results.append({
                    "id": req.id,
                    "text": req.text,
                    "obligation_type": req.obligation_type,
                    "applies_to": json.loads(req.applies_to or "[]"),
                    "risk_level": req.risk_level,
                    "section_id": req.section_id,
                    "doc_id": req.doc_id,
                    "doc_title": doc.title if doc else None,
                    "doc_jurisdiction": doc.jurisdiction if doc else None,
                    "relevance_score": round(s, 3),
                })

            return results
