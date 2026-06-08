"""Ingestion tools: add documents to the knowledge graph with LLM-assisted extraction."""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Optional

from openai import OpenAI
from fastmcp import FastMCP

from app.database import get_session
from app.embeddings import embed, embed_batch, to_json
from app.models import Document, DocRelationship, Requirement, Section

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """\
You are extracting structured information from a regulatory/compliance document to populate a knowledge graph.

Document title: {title}
Document type: {doc_type}
Jurisdiction: {jurisdiction}

Full text:
---
{text}
---

Extract the following:
1. A 3-5 sentence summary covering the document's purpose, scope, and key obligations.
2. Major sections (article/section number, title, and complete content).
3. Atomic compliance requirements — specific obligations that regulated entities must/should/may fulfill.

For requirements:
- obligation_type: "MUST" (shall/must/required/prohibited), "SHOULD" (should/recommended/expected), "MAY" (may/can/permitted/optional)
- applies_to: list of entity types this applies to from: providers, deployers, importers, distributors, users, operators, notified-bodies, market-surveillance-authorities, high-risk-systems, general-purpose-ai, foundation-models
- risk_level: "unacceptable" (banned), "high" (high-risk systems), "limited" (limited risk), "minimal" (minimal risk), "unspecified" (not risk-classified)
- section_number: which article/section this comes from

Focus on substantive obligations, not procedural boilerplate. Extract the most important 20-50 requirements.
"""

_CRITIC_PROMPT = """\
You are a compliance expert and critical reviewer. An automated system has extracted requirements from a regulatory document. Your job is to improve the extraction quality.

Document title: {title}
Jurisdiction: {jurisdiction}

Original document text:
---
{text}
---

Extracted requirements (to review):
---
{requirements_json}
---

Review the extracted requirements for these specific failure modes:
1. MISSED: Obligations in the text that were not captured at all
2. SPLIT: Requirements that are compound (multiple obligations bundled into one text) and should be broken apart
3. OBLIGATION_TYPE: Incorrect classification — "shall/must/required" → MUST; "should/recommended" → SHOULD; "may/permitted" → MAY; prohibitions → MUST with negation
4. APPLIES_TO: Missing or wrong entity types (providers, deployers, importers, distributors, users, operators, notified-bodies, market-surveillance-authorities, high-risk-systems, general-purpose-ai, foundation-models)
5. RISK_LEVEL: Incorrect risk tier classification

Return the COMPLETE corrected requirements list — include both corrected existing requirements and any newly found ones. Do not return only the changes; return the full final set.
"""

_REQ_SCHEMA = {
    "type": "object",
    "required": ["requirements"],
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["text", "obligation_type"],
                "properties": {
                    "text": {"type": "string"},
                    "obligation_type": {"type": "string", "enum": ["MUST", "SHOULD", "MAY"]},
                    "applies_to": {"type": "array", "items": {"type": "string"}},
                    "risk_level": {"type": "string"},
                    "section_number": {"type": "string"},
                },
            },
        }
    },
}


def _get_llm_client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("PROXY_BASE_URL", "https://proxy.npedwards.com/v1"),
        api_key=os.getenv("PROXY_API_KEY"),
    )


def _call_structured(client, prompt: str, tool_name: str, schema: dict, max_tokens: int = 8192) -> dict:
    """Make a structured tool-call to the LLM and return the parsed result."""
    response = client.chat.completions.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        tools=[{"type": "function", "function": {"name": tool_name, "parameters": schema}}],
        tool_choice={"type": "function", "function": {"name": tool_name}},
        messages=[{"role": "user", "content": prompt}],
    )
    tool_calls = response.choices[0].message.tool_calls
    if tool_calls and tool_calls[0].function.name == tool_name:
        return json.loads(tool_calls[0].function.arguments)
    raise ValueError(f"Model did not return expected tool call '{tool_name}'")


def _extract_structure(title: str, doc_type: str, jurisdiction: str, text: str) -> dict:
    """Two-pass extraction: extractor followed by a critic that catches errors and omissions."""
    client = _get_llm_client()

    truncated = text[:150000]
    was_truncated = len(text) > 150000

    # --- Pass 1: Extractor ---
    extractor_schema = {
        "type": "object",
        "required": ["summary", "sections", "requirements"],
        "properties": {
            "summary": {"type": "string"},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["title", "content"],
                    "properties": {
                        "section_number": {"type": "string"},
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
            "requirements": _REQ_SCHEMA["properties"]["requirements"],
        },
    }

    logger.info("  [pass 1] Extracting sections and requirements...")
    initial = _call_structured(
        client,
        _EXTRACTION_PROMPT.format(
            title=title,
            doc_type=doc_type,
            jurisdiction=jurisdiction or "unspecified",
            text=truncated,
        ),
        tool_name="store_extraction",
        schema=extractor_schema,
    )

    initial_reqs = initial.get("requirements", [])
    logger.info("  [pass 1] → %d requirements, %d sections", len(initial_reqs), len(initial.get("sections", [])))

    # --- Pass 2: Critic ---
    logger.info("  [pass 2] Critic reviewing extraction...")
    corrected = _call_structured(
        client,
        _CRITIC_PROMPT.format(
            title=title,
            jurisdiction=jurisdiction or "unspecified",
            text=truncated,
            requirements_json=json.dumps(initial_reqs, indent=2),
        ),
        tool_name="apply_corrections",
        schema=_REQ_SCHEMA,
    )

    final_reqs = corrected.get("requirements", initial_reqs)
    logger.info("  [pass 2] → %d requirements (was %d)", len(final_reqs), len(initial_reqs))

    return {
        "summary": initial.get("summary", ""),
        "sections": initial.get("sections", []),
        "requirements": final_reqs,
        "truncated": was_truncated,
    }


def _find_section_id(sections: list[Section], section_number: str | None) -> str | None:
    if not section_number:
        return None
    for s in sections:
        if s.section_number and section_number.lower() in s.section_number.lower():
            return s.id
    return None


def register_ingest_tools(mcp: FastMCP) -> None:

    @mcp.tool()
    def ingest_document(
        title: str,
        doc_type: str,
        full_text: str,
        jurisdiction: Optional[str] = None,
        issuer: Optional[str] = None,
        status: str = "enacted",
        effective_date: Optional[str] = None,
        url: Optional[str] = None,
        auto_extract: bool = True,
    ) -> dict:
        """Add a regulation, standard, or guidance document to the knowledge graph.

        Accepts the full text of the document. With auto_extract=True (default),
        uses Claude to extract a summary, sections, and atomic requirements automatically.

        doc_type: regulation, standard, guidance, article, enforcement_action
        jurisdiction: EU, US-Federal, US-CO, US-TX, US-CA, UK, Global, etc.
        status: enacted, proposed, draft, superseded
        effective_date: YYYY-MM-DD format
        """
        doc_id = str(uuid.uuid4())
        sections_created: list[Section] = []
        reqs_created: list[Requirement] = []
        summary = None
        truncated = False

        if auto_extract:
            logger.info("Extracting structure from '%s' (%d chars)", title, len(full_text))
            extracted = _extract_structure(title, doc_type, jurisdiction or "unspecified", full_text)
            summary = extracted.get("summary", "")
            truncated = extracted.get("truncated", False)

            # Build section objects
            raw_sections = extracted.get("sections", [])
            for sec in raw_sections:
                section = Section(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    section_number=sec.get("section_number"),
                    title=sec.get("title"),
                    content=sec.get("content", ""),
                )
                sections_created.append(section)

            # Embed sections in batch
            if sections_created:
                section_texts = [
                    f"{s.section_number or ''} {s.title or ''}: {s.content}"
                    for s in sections_created
                ]
                section_vecs = embed_batch(section_texts)
                for s, vec in zip(sections_created, section_vecs):
                    s.embedding = to_json(vec)

            # Build requirement objects
            raw_reqs = extracted.get("requirements", [])
            for req_data in raw_reqs:
                section_id = _find_section_id(sections_created, req_data.get("section_number"))
                req = Requirement(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    section_id=section_id,
                    text=req_data.get("text", ""),
                    obligation_type=req_data.get("obligation_type", "MUST"),
                    applies_to=json.dumps(req_data.get("applies_to", [])),
                    risk_level=req_data.get("risk_level", "unspecified"),
                )
                reqs_created.append(req)

            # Embed requirements in batch
            if reqs_created:
                req_texts = [r.text for r in reqs_created]
                req_vecs = embed_batch(req_texts)
                for r, vec in zip(reqs_created, req_vecs):
                    r.embedding = to_json(vec)

        # Embed document (use summary if available, else first chunk of full_text)
        embed_text = summary or full_text[:4000]
        doc_vec = embed(f"{title}\n\n{embed_text}")

        with get_session() as session:
            doc = Document(
                id=doc_id,
                title=title,
                doc_type=doc_type,
                jurisdiction=jurisdiction,
                issuer=issuer,
                status=status,
                effective_date=effective_date,
                url=url,
                full_text=full_text,
                summary=summary,
                embedding=to_json(doc_vec),
            )
            session.add(doc)
            session.flush()

            for s in sections_created:
                session.add(s)
            for r in reqs_created:
                session.add(r)

        return {
            "status": "ingested",
            "doc_id": doc_id,
            "title": title,
            "sections_extracted": len(sections_created),
            "requirements_extracted": len(reqs_created),
            "auto_extracted": auto_extract,
            "text_truncated_for_extraction": truncated,
            "note": "Use link_documents to connect this to related documents." if auto_extract else None,
        }

    @mcp.tool()
    def update_document_status(doc_id: str, status: str, notes: Optional[str] = None) -> dict:
        """Update the status of a document (e.g. mark as superseded, enacted, etc.).

        status: enacted, proposed, draft, superseded
        """
        valid_statuses = {"enacted", "proposed", "draft", "superseded"}
        if status not in valid_statuses:
            return {"error": f"Invalid status. Must be one of: {sorted(valid_statuses)}"}

        with get_session() as session:
            doc = session.get(Document, doc_id)
            if not doc:
                return {"error": f"Document {doc_id!r} not found"}

            old_status = doc.status
            doc.status = status
            if notes:
                doc.summary = (doc.summary or "") + f"\n\n[Status update: {notes}]"

        return {
            "doc_id": doc_id,
            "title": doc.title,
            "old_status": old_status,
            "new_status": status,
        }

    @mcp.tool()
    def delete_document(doc_id: str, confirm: bool = False) -> dict:
        """Delete a document and all its sections, requirements, and relationships.

        Must pass confirm=True to actually delete.
        """
        if not confirm:
            return {"error": "Pass confirm=True to delete this document and all associated data."}

        with get_session() as session:
            doc = session.get(Document, doc_id)
            if not doc:
                return {"error": f"Document {doc_id!r} not found"}

            title = doc.title
            # Cascade: requirements, sections, relationships
            session.query(Requirement).filter_by(doc_id=doc_id).delete()
            session.query(Section).filter_by(doc_id=doc_id).delete()
            session.query(DocRelationship).filter(
                (DocRelationship.from_id == doc_id) | (DocRelationship.to_id == doc_id)
            ).delete()
            session.delete(doc)

        return {"status": "deleted", "doc_id": doc_id, "title": title}
