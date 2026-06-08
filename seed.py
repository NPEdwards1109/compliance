#!/usr/bin/env python3
"""Seed the compliance database with key AI regulation documents.

Fetches public regulatory texts, extracts structure via LLM, and stores
them in the knowledge graph with embeddings.

Usage:
    python seed.py [--doc eu_ai_act] [--doc nist_rmf] ...
    python seed.py  # seeds all documents in priority order
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from html.parser import HTMLParser

import io

import httpx
from dotenv import load_dotenv

load_dotenv()

# Must be after load_dotenv so DB/API env vars are set
from app.database import get_session, init_db
from app.embeddings import embed, embed_batch, to_json
from app.models import Document, DocRelationship, Requirement, Section
from app.tools.ingest import _extract_structure, _find_section_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTML → plain text
# ---------------------------------------------------------------------------

class _Stripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip_tags = {"script", "style", "nav", "header", "footer"}
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "article"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self.parts.append(data)

    def get_text(self) -> str:
        import re
        text = "".join(self.parts)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _html_to_text(html: str) -> str:
    s = _Stripper()
    s.feed(html)
    return s.get_text()


def _pdf_to_text(content: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(content))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _fetch_text(url: str, is_html: bool = True) -> str:
    logger.info("Fetching %s", url)
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        r = client.get(url, headers={"User-Agent": "Mozilla/5.0 compliance-mcp-seed/1.0"})
        r.raise_for_status()
        content_type = r.headers.get("content-type", "")
        if "pdf" in content_type or url.endswith(".pdf"):
            return _pdf_to_text(r.content)
        if is_html:
            return _html_to_text(r.text)
        return r.text


# ---------------------------------------------------------------------------
# Core ingest function
# ---------------------------------------------------------------------------

def ingest(
    title: str,
    doc_type: str,
    jurisdiction: str,
    issuer: str,
    status: str,
    effective_date: str | None,
    url: str | None,
    text: str,
    skip_if_exists: bool = True,
) -> str | None:
    """Ingest a document. Returns doc_id or None if skipped."""
    with get_session() as session:
        existing = session.query(Document).filter_by(title=title).first()
        if existing and skip_if_exists:
            logger.info("Already ingested: %s (%s)", title, existing.id)
            return existing.id

    doc_id = str(uuid.uuid4())
    logger.info("Extracting structure from '%s' (%d chars)...", title, len(text))

    extracted = _extract_structure(title, doc_type, jurisdiction, text)
    summary = extracted.get("summary", "")
    raw_sections = extracted.get("sections", [])
    raw_reqs = extracted.get("requirements", [])
    truncated = extracted.get("truncated", False)

    logger.info("  → %d sections, %d requirements%s", len(raw_sections), len(raw_reqs),
                " [text truncated]" if truncated else "")

    sections_created: list[Section] = []
    for sec in raw_sections:
        section = Section(
            id=str(uuid.uuid4()),
            doc_id=doc_id,
            section_number=sec.get("section_number"),
            title=sec.get("title"),
            content=sec.get("content", ""),
        )
        sections_created.append(section)

    if sections_created:
        section_texts = [f"{s.section_number or ''} {s.title or ''}: {s.content}" for s in sections_created]
        vecs = embed_batch(section_texts)
        for s, vec in zip(sections_created, vecs):
            s.embedding = to_json(vec)

    reqs_created: list[Requirement] = []
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

    if reqs_created:
        req_texts = [r.text for r in reqs_created]
        vecs = embed_batch(req_texts)
        for r, vec in zip(reqs_created, vecs):
            r.embedding = to_json(vec)

    embed_text = summary or text[:4000]
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
            full_text=text,
            summary=summary,
            embedding=to_json(doc_vec),
        )
        session.add(doc)
        session.flush()
        for s in sections_created:
            session.add(s)
        for r in reqs_created:
            session.add(r)

    logger.info("  ✓ Ingested '%s' → %s", title, doc_id)
    return doc_id


def link(from_id: str, to_id: str, relationship: str, notes: str | None = None) -> None:
    with get_session() as session:
        existing = session.query(DocRelationship).filter_by(
            from_id=from_id, to_id=to_id, relationship=relationship
        ).first()
        if not existing:
            session.add(DocRelationship(from_id=from_id, to_id=to_id, relationship=relationship, notes=notes))
            logger.info("  → Linked %s -[%s]-> %s", from_id[:8], relationship, to_id[:8])


# ---------------------------------------------------------------------------
# Document catalog
# ---------------------------------------------------------------------------

DOCUMENTS = {
    "eu_ai_act": dict(
        title="EU Artificial Intelligence Act",
        doc_type="regulation",
        jurisdiction="EU",
        issuer="European Parliament and Council",
        status="enacted",
        effective_date="2024-08-01",
        url="https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689",
        fetch_url="https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689",
        is_html=True,
    ),
    "nist_ai_rmf": dict(
        title="NIST AI Risk Management Framework (AI RMF 1.0)",
        doc_type="standard",
        jurisdiction="US-Federal",
        issuer="NIST",
        status="enacted",
        effective_date="2023-01-26",
        url="https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
        fetch_url="https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
        is_html=False,
    ),
    "iso_42001": dict(
        title="ISO/IEC 42001:2023 — AI Management Systems",
        doc_type="standard",
        jurisdiction="Global",
        issuer="ISO/IEC",
        status="enacted",
        effective_date="2023-12-18",
        url="https://www.iso.org/standard/81230.html",
        manual=True,
        note="ISO standards require purchase. Add manually via ingest_document.",
    ),
    "us_eo_14110": dict(
        title="Executive Order 14110 — Safe, Secure, and Trustworthy AI (Biden)",
        doc_type="regulation",
        jurisdiction="US-Federal",
        issuer="White House",
        status="superseded",
        effective_date="2023-10-30",
        url="https://www.federalregister.gov/documents/2023/11/01/2023-24283/safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence",
        fetch_url="https://www.federalregister.gov/documents/2023/11/01/2023-24283/safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence",
        is_html=True,
    ),
    "us_eo_14179": dict(
        title="Executive Order 14179 — Removing Barriers to American Leadership in AI (Trump)",
        doc_type="regulation",
        jurisdiction="US-Federal",
        issuer="White House",
        status="enacted",
        effective_date="2025-01-20",
        url="https://www.whitehouse.gov/presidential-actions/2025/01/removing-barriers-to-american-leadership-in-artificial-intelligence/",
        fetch_url="https://www.whitehouse.gov/presidential-actions/2025/01/removing-barriers-to-american-leadership-in-artificial-intelligence/",
        is_html=True,
    ),
    "colorado_ai_act": dict(
        title="Colorado AI Act (SB 24-205) — Consumer Protections for AI",
        doc_type="regulation",
        jurisdiction="US-CO",
        issuer="Colorado General Assembly",
        status="enacted",
        effective_date="2026-02-01",
        url="https://leg.colorado.gov/sites/default/files/2024a_205_signed.pdf",
        fetch_url="https://leg.colorado.gov/sites/default/files/2024a_205_signed.pdf",
        is_html=False,
    ),
    "ftc_ai_guidance": dict(
        title="FTC Policy Statement on AI and Consumer Protection",
        doc_type="guidance",
        jurisdiction="US-Federal",
        issuer="Federal Trade Commission",
        status="enacted",
        effective_date="2023-11-21",
        url="https://www.ftc.gov/system/files/ftc_gov/pdf/generative-ai-policy-statement.pdf",
        fetch_url="https://www.ftc.gov/system/files/ftc_gov/pdf/generative-ai-policy-statement.pdf",
        is_html=False,
    ),
    "nist_ai_600_1": dict(
        title="NIST AI 600-1 — Generative AI Profile",
        doc_type="standard",
        jurisdiction="US-Federal",
        issuer="NIST",
        status="enacted",
        effective_date="2024-07-26",
        url="https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf",
        fetch_url="https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf",
        is_html=False,
    ),
}


GRAPH_EDGES = [
    # EU AI Act is the anchor — other regs implement or relate to it
    ("eu_ai_act", "nist_ai_rmf", "RELATED_TO", "Both address AI risk management; NIST RMF aligns with EU AI Act requirements"),
    ("nist_ai_rmf", "eu_ai_act", "RELATED_TO", "NIST RMF provides a framework compatible with EU AI Act compliance"),
    ("colorado_ai_act", "eu_ai_act", "RELATED_TO", "Colorado AI Act draws on EU AI Act risk-tiering approach"),
    ("us_eo_14179", "us_eo_14110", "SUPERSEDES", "EO 14179 revoked EO 14110 and redirected US AI policy"),
    ("nist_ai_600_1", "nist_ai_rmf", "IMPLEMENTS", "AI 600-1 is a profile of the AI RMF for generative AI"),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _fetch_doc_text(key: str, meta: dict) -> str | None:
    if meta.get("manual"):
        logger.warning("Skipping %s — requires manual ingestion (%s)", key, meta.get("note", ""))
        return None

    fetch_url = meta.get("fetch_url") or meta.get("url")
    is_html = meta.get("is_html", True)

    try:
        return _fetch_text(fetch_url, is_html=is_html)
    except Exception as e:
        logger.error("Failed to fetch %s: %s", key, e)
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", action="append", help="Specific doc key(s) to ingest (default: all)")
    parser.add_argument("--no-links", action="store_true", help="Skip graph edge creation")
    args = parser.parse_args()

    init_db()

    target_keys = args.doc or list(DOCUMENTS.keys())
    doc_ids: dict[str, str] = {}

    for key in target_keys:
        if key not in DOCUMENTS:
            logger.error("Unknown document key: %s. Available: %s", key, list(DOCUMENTS.keys()))
            continue

        meta = DOCUMENTS[key]
        logger.info("=== %s ===", meta["title"])

        text = _fetch_doc_text(key, meta)
        if not text:
            continue

        if len(text.strip()) < 500:
            logger.warning("Fetched text too short (%d chars) — skipping %s", len(text), key)
            continue

        doc_id = ingest(
            title=meta["title"],
            doc_type=meta["doc_type"],
            jurisdiction=meta["jurisdiction"],
            issuer=meta["issuer"],
            status=meta["status"],
            effective_date=meta.get("effective_date"),
            url=meta.get("url"),
            text=text,
        )
        if doc_id:
            doc_ids[key] = doc_id

    if not args.no_links and len(doc_ids) > 1:
        logger.info("=== Creating graph edges ===")
        for from_key, to_key, relationship, notes in GRAPH_EDGES:
            if from_key in doc_ids and to_key in doc_ids:
                link(doc_ids[from_key], doc_ids[to_key], relationship, notes)

    logger.info("=== Done. Ingested %d documents. ===", len(doc_ids))
    for key, doc_id in doc_ids.items():
        logger.info("  %s → %s", key, doc_id)


if __name__ == "__main__":
    main()
