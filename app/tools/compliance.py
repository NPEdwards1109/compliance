"""Compliance tools: checklists, timelines, and cross-jurisdiction comparison."""
from __future__ import annotations

import json
from typing import Optional

from fastmcp import FastMCP

from app.database import get_session
from app.embeddings import cosine_similarity, embed, from_json
from app.models import Document, Requirement


def register_compliance_tools(mcp: FastMCP) -> None:

    @mcp.tool()
    def get_compliance_checklist(
        use_case: str,
        jurisdiction: Optional[str] = None,
        risk_level: Optional[str] = None,
        applies_to: Optional[str] = None,
    ) -> dict:
        """Generate a compliance checklist for a specific use case.

        Finds the most relevant MUST requirements for a described use case,
        optionally filtered by jurisdiction, risk level, and entity type.

        Example:
          use_case="facial recognition for employee monitoring"
          jurisdiction="EU"
          applies_to="providers"
        """
        with get_session() as session:
            query_vec = embed(use_case)

            q = session.query(Requirement).filter(Requirement.obligation_type == "MUST")
            if risk_level:
                q = q.filter(Requirement.risk_level == risk_level)

            all_reqs = q.all()

            # Filter by jurisdiction via doc join
            if jurisdiction:
                filtered = []
                for req in all_reqs:
                    doc = session.get(Document, req.doc_id)
                    if doc and doc.jurisdiction == jurisdiction:
                        filtered.append(req)
                all_reqs = filtered

            # Filter by applies_to
            if applies_to:
                filtered = []
                for req in all_reqs:
                    req_applies = json.loads(req.applies_to or "[]")
                    if applies_to.lower() in [a.lower() for a in req_applies]:
                        filtered.append(req)
                all_reqs = filtered

            # Score by semantic similarity to the use case
            scored: list[tuple[float, Requirement]] = []
            for req in all_reqs:
                vec = from_json(req.embedding)
                if vec is None:
                    scored.append((0.3, req))
                    continue
                score = cosine_similarity(query_vec, vec)
                scored.append((score, req))

            # Return top 30 most relevant requirements
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:30]

            # Group by jurisdiction/document
            by_doc: dict[str, list] = {}
            for score, req in top:
                doc = session.get(Document, req.doc_id)
                doc_key = f"{doc.jurisdiction} — {doc.title}" if doc else req.doc_id
                by_doc.setdefault(doc_key, []).append({
                    "id": req.id,
                    "text": req.text,
                    "applies_to": json.loads(req.applies_to or "[]"),
                    "risk_level": req.risk_level,
                    "relevance_score": round(score, 3),
                })

            return {
                "use_case": use_case,
                "jurisdiction_filter": jurisdiction,
                "applies_to_filter": applies_to,
                "total_requirements": len(top),
                "by_document": by_doc,
            }

    @mcp.tool()
    def get_timeline(jurisdiction: Optional[str] = None) -> list[dict]:
        """Get the regulatory timeline — key effective dates and enforcement milestones.

        Returns documents sorted chronologically by effective date.
        Optionally filter by jurisdiction.
        """
        with get_session() as session:
            q = session.query(Document).filter(Document.effective_date.isnot(None))
            if jurisdiction:
                q = q.filter(Document.jurisdiction == jurisdiction)

            docs = q.order_by(Document.effective_date).all()

            return [
                {
                    "effective_date": d.effective_date,
                    "title": d.title,
                    "jurisdiction": d.jurisdiction,
                    "doc_type": d.doc_type,
                    "status": d.status,
                    "id": d.id,
                    "url": d.url,
                }
                for d in docs
            ]

    @mcp.tool()
    def compare_jurisdictions(topic: str, jurisdictions: list[str]) -> dict:
        """Compare how different jurisdictions address a topic.

        Searches for the most relevant documents per jurisdiction and returns
        a side-by-side view of the top matching content.

        Example:
          topic="transparency and explainability requirements"
          jurisdictions=["EU", "US-Federal", "US-CO"]
        """
        from app.embeddings import cosine_similarity, embed, from_json

        with get_session() as session:
            topic_vec = embed(topic)

            result: dict[str, list] = {}

            for jur in jurisdictions:
                docs = session.query(Document).filter(Document.jurisdiction == jur).all()

                if not docs:
                    result[jur] = [{"message": "No documents in corpus for this jurisdiction."}]
                    continue

                # Score docs by semantic similarity
                scored = []
                for doc in docs:
                    vec = from_json(doc.embedding)
                    if vec is None:
                        scored.append((0.1, doc))
                        continue
                    score = cosine_similarity(topic_vec, vec)
                    scored.append((score, doc))

                scored.sort(key=lambda x: x[0], reverse=True)
                top = scored[:3]

                jur_results = []
                for score, doc in top:
                    # Find top requirements in this doc matching the topic
                    reqs = session.query(Requirement).filter_by(doc_id=doc.id).all()
                    req_scored = []
                    for req in reqs:
                        vec = from_json(req.embedding)
                        if vec:
                            s = cosine_similarity(topic_vec, vec)
                            req_scored.append((s, req))
                    req_scored.sort(key=lambda x: x[0], reverse=True)
                    top_reqs = [
                        {
                            "text": r.text,
                            "obligation_type": r.obligation_type,
                            "applies_to": json.loads(r.applies_to or "[]"),
                        }
                        for _, r in req_scored[:5]
                    ]

                    jur_results.append({
                        "doc_id": doc.id,
                        "title": doc.title,
                        "doc_type": doc.doc_type,
                        "status": doc.status,
                        "relevance_score": round(score, 3),
                        "summary": doc.summary,
                        "top_requirements": top_reqs,
                    })

                result[jur] = jur_results

            return {"topic": topic, "by_jurisdiction": result}
