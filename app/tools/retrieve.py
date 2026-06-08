"""Retrieval tools: fetch documents, sections, and requirements by ID."""
from __future__ import annotations

import json
from typing import Optional

from fastmcp import FastMCP

from app.database import get_session
from app.models import Document, Requirement, Section


def register_retrieve_tools(mcp: FastMCP) -> None:

    @mcp.tool()
    def get_document(doc_id: str, include_full_text: bool = False) -> dict:
        """Fetch a document by ID.

        Returns metadata and summary. Set include_full_text=True to include the
        complete regulatory text (can be large).
        """
        with get_session() as session:
            doc = session.get(Document, doc_id)
            if not doc:
                return {"error": f"Document {doc_id!r} not found"}

            result = {
                "id": doc.id,
                "title": doc.title,
                "doc_type": doc.doc_type,
                "jurisdiction": doc.jurisdiction,
                "issuer": doc.issuer,
                "status": doc.status,
                "effective_date": doc.effective_date,
                "url": doc.url,
                "summary": doc.summary,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
            }
            if include_full_text:
                result["full_text"] = doc.full_text

            section_count = session.query(Section).filter_by(doc_id=doc_id).count()
            req_count = session.query(Requirement).filter_by(doc_id=doc_id).count()
            result["section_count"] = section_count
            result["requirement_count"] = req_count

            return result

    @mcp.tool()
    def get_sections(doc_id: str) -> list[dict]:
        """List all sections for a document, in order."""
        with get_session() as session:
            doc = session.get(Document, doc_id)
            if not doc:
                return [{"error": f"Document {doc_id!r} not found"}]

            sections = session.query(Section).filter_by(doc_id=doc_id).all()
            return [
                {
                    "id": s.id,
                    "section_number": s.section_number,
                    "title": s.title,
                    "content": s.content,
                    "parent_section_id": s.parent_section_id,
                }
                for s in sections
            ]

    @mcp.tool()
    def get_section(section_id: str) -> dict:
        """Fetch a specific section by ID."""
        with get_session() as session:
            section = session.get(Section, section_id)
            if not section:
                return {"error": f"Section {section_id!r} not found"}

            doc = session.get(Document, section.doc_id)
            reqs = session.query(Requirement).filter_by(section_id=section_id).all()

            return {
                "id": section.id,
                "doc_id": section.doc_id,
                "doc_title": doc.title if doc else None,
                "section_number": section.section_number,
                "title": section.title,
                "content": section.content,
                "parent_section_id": section.parent_section_id,
                "requirements": [
                    {
                        "id": r.id,
                        "text": r.text,
                        "obligation_type": r.obligation_type,
                        "applies_to": json.loads(r.applies_to or "[]"),
                        "risk_level": r.risk_level,
                    }
                    for r in reqs
                ],
            }

    @mcp.tool()
    def get_requirements(doc_id: str, obligation_type: Optional[str] = None) -> list[dict]:
        """Get all requirements extracted from a document.

        Optionally filter by obligation_type: MUST, SHOULD, or MAY.
        """
        with get_session() as session:
            doc = session.get(Document, doc_id)
            if not doc:
                return [{"error": f"Document {doc_id!r} not found"}]

            q = session.query(Requirement).filter_by(doc_id=doc_id)
            if obligation_type:
                q = q.filter(Requirement.obligation_type == obligation_type)

            reqs = q.all()

            # Group by section for readability
            by_section: dict[str, list] = {}
            for req in reqs:
                key = req.section_id or "unsectioned"
                by_section.setdefault(key, []).append(req)

            result = []
            for section_id, section_reqs in by_section.items():
                section = session.get(Section, section_id) if section_id != "unsectioned" else None
                for req in section_reqs:
                    result.append({
                        "id": req.id,
                        "text": req.text,
                        "obligation_type": req.obligation_type,
                        "applies_to": json.loads(req.applies_to or "[]"),
                        "risk_level": req.risk_level,
                        "section_number": section.section_number if section else None,
                        "section_title": section.title if section else None,
                    })

            return result

    @mcp.tool()
    def list_documents(
        jurisdiction: Optional[str] = None,
        doc_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[dict]:
        """List all ingested documents with optional filters.

        Good starting point to see what's in the knowledge base.
        """
        with get_session() as session:
            q = session.query(Document)
            if jurisdiction:
                q = q.filter(Document.jurisdiction == jurisdiction)
            if doc_type:
                q = q.filter(Document.doc_type == doc_type)
            if status:
                q = q.filter(Document.status == status)

            docs = q.order_by(Document.jurisdiction, Document.effective_date).all()

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
                }
                for d in docs
            ]
