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
from app.embeddings import cosine_similarity, embed, embed_batch, from_json, to_json
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


_EDGE_SYSTEM = """\
You are building a knowledge graph of AI regulations and standards. A new document has been ingested and you must identify directed relationships FROM it TO existing documents in the corpus.

You have tools to explore the corpus — use them to find relevant documents before proposing edges. Search by topic, jurisdiction, or document name. Retrieve summaries for anything promising.

Relationship types (from new doc → existing doc):
- SUPERSEDES  — new doc replaces or revokes the existing doc
- AMENDS      — new doc modifies or updates the existing doc
- IMPLEMENTS  — new doc operationalizes a framework in the existing doc
- CITES       — new doc explicitly references or builds on the existing doc
- RELATED_TO  — new doc covers substantially overlapping regulatory territory
- ANALYZED_BY — new doc is an analysis/commentary on the existing doc

Rules:
- Only propose edges where the relationship is clearly justified — don't create noise
- SUPERSEDES/AMENDS require same jurisdiction and a clear versioning signal
- Prefer specific types over RELATED_TO
- You may propose 0 edges
- When done, call propose_edges with your final list
"""

_EDGE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": "Search existing corpus documents by keyword or topic. Returns doc IDs, titles, jurisdictions.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "jurisdiction": {"type": "string", "description": "Optional filter, e.g. EU, US-Federal, UK"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_doc_info",
            "description": "Get metadata and summary for a specific document by ID.",
            "parameters": {
                "type": "object",
                "required": ["doc_id"],
                "properties": {"doc_id": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_edges",
            "description": "Submit your final proposed edges. Call this when done exploring.",
            "parameters": {
                "type": "object",
                "required": ["edges"],
                "properties": {
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["to_doc_id", "relationship"],
                            "properties": {
                                "to_doc_id": {"type": "string"},
                                "relationship": {
                                    "type": "string",
                                    "enum": ["SUPERSEDES", "AMENDS", "IMPLEMENTS", "CITES", "RELATED_TO", "ANALYZED_BY"],
                                },
                                "notes": {"type": "string"},
                            },
                        },
                    }
                },
            },
        },
    },
]


def _run_edge_tool(name: str, args: dict, new_doc_id: str, new_vec: list) -> str:
    """Execute an agent tool call and return a JSON string result."""
    if name == "search_corpus":
        query = args.get("query", "")
        jurisdiction = args.get("jurisdiction")
        with get_session() as session:
            q = session.query(Document).filter(Document.id != new_doc_id)
            if jurisdiction:
                q = q.filter(Document.jurisdiction == jurisdiction)
            # FTS search
            try:
                from sqlalchemy import text as sa_text
                fts_q = f'"{query}"' if " " in query else query
                ids = [
                    r[0] for r in session.execute(
                        sa_text("SELECT doc_id FROM documents_fts WHERE documents_fts MATCH :q LIMIT 20"),
                        {"q": fts_q},
                    ).fetchall()
                ]
                if ids:
                    q = q.filter(Document.id.in_(ids))
                    docs = q.all()
                else:
                    docs = []
            except Exception:
                docs = q.limit(20).all()
            results = [
                {"id": d.id, "title": d.title, "jurisdiction": d.jurisdiction, "doc_type": d.doc_type}
                for d in docs
            ]
        return json.dumps(results)

    elif name == "get_doc_info":
        doc_id = args.get("doc_id")
        with get_session() as session:
            d = session.get(Document, doc_id)
            if not d or d.id == new_doc_id:
                return json.dumps({"error": "not found"})
            return json.dumps({
                "id": d.id, "title": d.title, "doc_type": d.doc_type,
                "jurisdiction": d.jurisdiction, "status": d.status,
                "effective_date": d.effective_date, "summary": (d.summary or "")[:600],
            })

    return json.dumps({"error": f"unknown tool {name}"})


def _suggest_edges(
    client: OpenAI,
    new_doc_id: str,
    title: str,
    doc_type: str,
    jurisdiction: str,
    summary: str,
    new_vec: list,
) -> list[dict]:
    """Agentic loop: let Claude search the corpus and propose graph edges."""
    messages = [
        {"role": "system", "content": _EDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"New document just ingested:\n"
                f"  Title: {title}\n"
                f"  Type: {doc_type}\n"
                f"  Jurisdiction: {jurisdiction}\n"
                f"  Summary: {summary[:600]}\n\n"
                "Search the corpus and propose edges. Call propose_edges when done."
            ),
        },
    ]

    max_iterations = 10
    for i in range(max_iterations):
        response = client.chat.completions.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            tools=_EDGE_TOOLS,
            messages=messages,
        )
        msg = response.choices[0].message
        # Append assistant turn — serialize tool_calls to dicts for the messages list
        assistant_turn = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_turn["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_turn)

        if not msg.tool_calls:
            logger.info("  [pass 3] Agent stopped without calling propose_edges (iteration %d)", i + 1)
            return []

        results = []
        done = False
        for tc in msg.tool_calls:
            fn = tc.function.name
            args = json.loads(tc.function.arguments)
            logger.info("  [pass 3] agent tool: %s(%s)", fn, list(args.keys()))

            if fn == "propose_edges":
                done = True
                results.append({"role": "tool", "tool_call_id": tc.id, "content": "edges recorded"})
                return args.get("edges", [])
            else:
                result = _run_edge_tool(fn, args, new_doc_id, new_vec)
                results.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        messages.extend(results)
        if done:
            break

    logger.warning("  [pass 3] Agent hit max iterations without proposing edges")
    return []


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

        # Pass 3: auto-suggest and create edges to existing documents
        edges_created = []
        if auto_extract and summary:
            logger.info("  [pass 3] Suggesting edges to existing corpus...")
            client = _get_llm_client()
            proposed = _suggest_edges(
                client, doc_id, title, doc_type, jurisdiction or "unspecified", summary, doc_vec
            )
            logger.info("  [pass 3] → %d edges proposed", len(proposed))
            valid_rels = {"SUPERSEDES", "AMENDS", "IMPLEMENTS", "CITES", "RELATED_TO", "ANALYZED_BY"}
            with get_session() as session:
                for edge in proposed:
                    to_id = edge.get("to_doc_id")
                    rel = edge.get("relationship", "").upper()
                    if not to_id or rel not in valid_rels:
                        continue
                    # verify target exists
                    if not session.get(Document, to_id):
                        continue
                    # skip duplicate
                    existing_edge = (
                        session.query(DocRelationship)
                        .filter_by(from_id=doc_id, to_id=to_id, relationship=rel)
                        .first()
                    )
                    if existing_edge:
                        continue
                    session.add(DocRelationship(
                        from_id=doc_id,
                        to_id=to_id,
                        relationship=rel,
                        notes=edge.get("notes", ""),
                    ))
                    edges_created.append({"to_doc_id": to_id, "relationship": rel, "notes": edge.get("notes", "")})

        return {
            "status": "ingested",
            "doc_id": doc_id,
            "title": title,
            "sections_extracted": len(sections_created),
            "requirements_extracted": len(reqs_created),
            "edges_created": len(edges_created),
            "edges": edges_created,
            "auto_extracted": auto_extract,
            "text_truncated_for_extraction": truncated,
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
