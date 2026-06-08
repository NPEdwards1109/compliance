#!/usr/bin/env python3
"""Daily compliance monitoring sweep.

Polls RSS feeds from key regulatory sources, scores new items for AI compliance
relevance, ingests relevant content, and sends an email digest.

Usage:
    python monitor.py          # run sweep
    python monitor.py --dry-run  # check feeds without ingesting or emailing
    python monitor.py --reset  # clear state (reprocess all items on next run)

Cron (2am UTC daily):
    0 2 * * * cd /app && python monitor.py >> /app/data/monitor.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from app.database import init_db
from app.tools.ingest import _extract_structure, _find_section_id
from jmap_send import JmapError, send_email
from app.embeddings import embed, embed_batch, to_json
from app.models import Document, Requirement, Section

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

STATE_FILE = Path("data/monitor_state.json")
LOG_FILE = Path("data/monitor.log")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Email goes out via Fastmail JMAP (port 443) — see jmap_send.py / JMAP_EMAIL.md.
# SMTP (465/587) is blocked outbound on the DigitalOcean droplet.
FASTMAIL_API_TOKEN = os.getenv("FASTMAIL_API_TOKEN", "")
FROM_EMAIL = os.getenv("COMPLIANCE_FROM_EMAIL", "compliance@npedwards.com")
TO_EMAIL = os.getenv("COMPLIANCE_TO_EMAIL", "")

RELEVANCE_THRESHOLD = 0.85  # default threshold; regulatory feeds use lower per-feed override

# ---------------------------------------------------------------------------
# RSS feed catalog
# ---------------------------------------------------------------------------

FEEDS = [
    # --- Primary regulatory sources (lower threshold — high signal, authoritative) ---
    {
        "key": "federal_register_ai",
        "name": "Federal Register — AI",
        "url": "https://www.federalregister.gov/api/v1/documents.rss?conditions[term]=artificial+intelligence",
        "jurisdiction": "US-Federal",
        "default_type": "regulation",
        "default_issuer": "Federal Register",
        "threshold": 0.70,
    },
    {
        "key": "ftc_news",
        "name": "Federal Register — FTC",
        "url": "https://www.federalregister.gov/api/v1/documents.rss?conditions[agencies][]=federal-trade-commission",
        "jurisdiction": "US-Federal",
        "default_type": "guidance",
        "default_issuer": "Federal Trade Commission",
        "threshold": 0.70,
    },
    {
        "key": "nist_news",
        "name": "NIST News",
        "url": "https://www.nist.gov/news-events/news/rss.xml",
        "jurisdiction": "US-Federal",
        "default_type": "guidance",
        "default_issuer": "NIST",
        "threshold": 0.70,
    },
    {
        "key": "cisa_news",
        "name": "CISA News",
        "url": "https://www.cisa.gov/news.xml",
        "jurisdiction": "US-Federal",
        "default_type": "guidance",
        "default_issuer": "CISA",
        "threshold": 0.75,
    },
    {
        "key": "eu_commission_digital",
        "name": "EU Commission — Digital Strategy",
        "url": "https://ec.europa.eu/commission/presscorner/api/rss?lang=en&topic=digitalization",
        "jurisdiction": "EU",
        "default_type": "guidance",
        "default_issuer": "European Commission",
        "threshold": 0.70,
    },
    {
        "key": "ai_safety_institute",
        "name": "UK AI Safety Institute",
        "url": "https://www.gov.uk/search/news-and-communications.atom?organisations%5B%5D=ai-safety-institute",
        "jurisdiction": "UK",
        "default_type": "guidance",
        "default_issuer": "UK AI Safety Institute",
        "threshold": 0.70,
    },
    # --- Commentary & analysis feeds (higher threshold — filter out advocacy noise) ---
    {
        "key": "eff_deeplinks",
        "name": "EFF Deeplinks",
        "url": "https://www.eff.org/rss/updates.xml",
        "jurisdiction": "US-Federal",
        "default_type": "article",
        "default_issuer": "Electronic Frontier Foundation",
        "threshold": 0.88,
    },
    {
        "key": "future_of_privacy_forum",
        "name": "Future of Privacy Forum",
        "url": "https://fpf.org/feed/",
        "jurisdiction": "Global",
        "default_type": "article",
        "default_issuer": "Future of Privacy Forum",
        "threshold": 0.85,
    },
    {
        "key": "ai_now_institute",
        "name": "AI Now Institute",
        "url": "https://ainowinstitute.org/feed",
        "jurisdiction": "Global",
        "default_type": "article",
        "default_issuer": "AI Now Institute",
        "threshold": 0.88,
    },
    {
        "key": "mit_tech_review_ai",
        "name": "MIT Technology Review — AI",
        "url": "https://www.technologyreview.com/topic/artificial-intelligence/feed",
        "jurisdiction": "Global",
        "default_type": "article",
        "default_issuer": "MIT Technology Review",
        "threshold": 0.88,
    },
    {
        "key": "future_of_life_institute",
        "name": "Future of Life Institute",
        "url": "https://futureoflife.org/feed/",
        "jurisdiction": "Global",
        "default_type": "article",
        "default_issuer": "Future of Life Institute",
        "threshold": 0.92,  # advocacy org — only substantive original research/assessments
    },
    {
        "key": "responsible_ai_institute",
        "name": "Responsible AI Institute",
        "url": "https://www.responsible.ai/feed",
        "jurisdiction": "Global",
        "default_type": "article",
        "default_issuer": "Responsible AI Institute",
        "threshold": 0.85,
    },
]

# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------

_RELEVANCE_PROMPT = """\
You are a compliance analyst maintaining a corpus of authoritative AI regulatory content.

Assess whether this item belongs in a compliance knowledge base. Score 0.0–1.0.

SCORE HIGH (0.85+) for:
- New or amended laws, regulations, rules, or executive orders with binding effect
- Official enforcement actions, consent orders, or fines with specific findings
- Government-issued standards, frameworks, or formal guidance documents
- Official regulatory consultations, proposed rulemakings, or public comment periods
- Primary research reports with specific, actionable compliance requirements

SCORE MEDIUM (0.70–0.84) for:
- Detailed analysis of specific regulatory provisions by authoritative bodies
- Official government announcements about upcoming regulatory timelines
- Substantive compliance frameworks or assessment tools (not press releases about them)

SCORE LOW (<0.70) — do NOT ingest — for:
- Staff statements, reactions, or op-eds from advocacy organizations
- Press releases promoting an organization's own campaigns or fundraising
- Journalism that summarizes existing regulations without new primary content
- Opinion pieces, blog posts, or thought leadership without regulatory specificity
- Announcements of partnerships, events, or organizational news

Item title: {title}
Source: {source}
Summary: {summary}

Return a relevance score and a one-sentence reason.
"""

_RELEVANCE_SCHEMA = {
    "type": "object",
    "required": ["score", "reason"],
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
}


def _score_relevance(title: str, source: str, summary: str) -> tuple[float, str]:
    client = OpenAI(
        base_url=os.getenv("PROXY_BASE_URL", "https://proxy.npedwards.com/v1"),
        api_key=os.getenv("PROXY_API_KEY"),
    )
    response = client.chat.completions.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        tools=[{"type": "function", "function": {"name": "score", "parameters": _RELEVANCE_SCHEMA}}],
        tool_choice={"type": "function", "function": {"name": "score"}},
        messages=[{"role": "user", "content": _RELEVANCE_PROMPT.format(
            title=title, source=source, summary=summary or "(no summary)",
        )}],
    )
    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        result = json.loads(tool_calls[0].function.arguments)
        return result["score"], result["reason"]
    return 0.0, "no response"


# ---------------------------------------------------------------------------
# Text fetching
# ---------------------------------------------------------------------------

def _fetch_full_text(url: str) -> str | None:
    try:
        with httpx.Client(timeout=30, follow_redirects=True,
                          headers={"User-Agent": "Mozilla/5.0 compliance-monitor/1.0"}) as client:
            r = client.get(url)
            r.raise_for_status()
            content_type = r.headers.get("content-type", "")
            if "pdf" in content_type or url.endswith(".pdf"):
                import io
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(r.content))
                return "\n\n".join(p.extract_text() for p in reader.pages if p.extract_text())
            # Strip HTML
            from html.parser import HTMLParser
            class _S(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.parts, self._skip = [], 0
                def handle_starttag(self, tag, attrs):
                    if tag in {"script", "style", "nav", "header", "footer"}:
                        self._skip += 1
                def handle_endtag(self, tag):
                    if tag in {"script", "style", "nav", "header", "footer"} and self._skip:
                        self._skip -= 1
                    if tag in {"p", "div", "li", "h1", "h2", "h3"}:
                        self.parts.append("\n")
                def handle_data(self, data):
                    if not self._skip:
                        self.parts.append(data)
            import re
            s = _S(); s.feed(r.text)
            return re.sub(r"\n{3,}", "\n\n", "".join(s.parts)).strip()
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return None


# ---------------------------------------------------------------------------
# Ingestion (shared logic with seed.py)
# ---------------------------------------------------------------------------

def _ingest_item(title: str, doc_type: str, jurisdiction: str, issuer: str,
                  url: str, text: str, published: str | None) -> str | None:
    import uuid
    from app.database import get_session

    with get_session() as session:
        if session.query(Document).filter_by(title=title).first():
            logger.info("  Already ingested: %s", title)
            return None

    doc_id = str(uuid.uuid4())
    extracted = _extract_structure(title, doc_type, jurisdiction, text)
    summary = extracted.get("summary", "")
    raw_sections = extracted.get("sections", [])
    raw_reqs = extracted.get("requirements", [])

    sections_created: list[Section] = []
    for sec in raw_sections:
        s = Section(
            id=str(uuid.uuid4()), doc_id=doc_id,
            section_number=sec.get("section_number"),
            title=sec.get("title"), content=sec.get("content", ""),
        )
        sections_created.append(s)

    if sections_created:
        vecs = embed_batch([f"{s.section_number or ''} {s.title or ''}: {s.content}" for s in sections_created])
        for s, v in zip(sections_created, vecs):
            s.embedding = to_json(v)

    reqs_created: list[Requirement] = []
    for rd in raw_reqs:
        r = Requirement(
            id=str(uuid.uuid4()), doc_id=doc_id,
            section_id=_find_section_id(sections_created, rd.get("section_number")),
            text=rd.get("text", ""),
            obligation_type=rd.get("obligation_type", "MUST"),
            applies_to=json.dumps(rd.get("applies_to", [])),
            risk_level=rd.get("risk_level", "unspecified"),
        )
        reqs_created.append(r)

    if reqs_created:
        vecs = embed_batch([r.text for r in reqs_created])
        for r, v in zip(reqs_created, vecs):
            r.embedding = to_json(v)

    doc_vec = embed(f"{title}\n\n{summary or text[:4000]}")

    from app.database import get_session
    with get_session() as session:
        doc = Document(
            id=doc_id, title=title, doc_type=doc_type, jurisdiction=jurisdiction,
            issuer=issuer, status="enacted", effective_date=published,
            url=url, full_text=text, summary=summary, embedding=to_json(doc_vec),
        )
        session.add(doc)
        session.flush()
        for s in sections_created: session.add(s)
        for r in reqs_created: session.add(r)

    logger.info("  ✓ Ingested '%s' → %s (%d reqs)", title, doc_id, len(reqs_created))
    return doc_id


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

def _send_email(subject: str, body: str) -> None:
    if not FASTMAIL_API_TOKEN or not TO_EMAIL:
        logger.warning("Email not configured — skipping notification")
        return
    try:
        send_email(
            subject,
            body,
            to=TO_EMAIL,
            from_addr=f"Compliance Monitor <{FROM_EMAIL}>",
            token=FASTMAIL_API_TOKEN,
        )
    except JmapError as exc:
        logger.error("email send failed: %s", exc)
        return
    logger.info("Email sent to %s", TO_EMAIL)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(dry_run: bool = False) -> None:
    init_db()
    state = _load_state()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    ingested: list[dict] = []
    skipped_irrelevant = 0
    errors = 0

    for feed_cfg in FEEDS:
        key = feed_cfg["key"]
        seen_ids: set[str] = set(state.get(key, []))
        new_seen: list[str] = list(seen_ids)

        logger.info("Checking feed: %s", feed_cfg["name"])
        try:
            parsed = feedparser.parse(feed_cfg["url"])
        except Exception as e:
            logger.error("Failed to parse feed %s: %s", feed_cfg["name"], e)
            errors += 1
            continue

        entries = parsed.entries or []
        logger.info("  %d entries, %d already seen", len(entries), len(seen_ids))

        for entry in entries:
            item_id = entry.get("id") or entry.get("link", "")
            if item_id in seen_ids:
                continue

            new_seen.append(item_id)
            title = entry.get("title", "Untitled")
            link = entry.get("link", "")
            summary = entry.get("summary", "") or entry.get("description", "")
            published = entry.get("published", "")[:10] if entry.get("published") else None

            # Strip HTML from summary
            import re
            summary_plain = re.sub(r"<[^>]+>", " ", summary).strip()

            # Relevance filter
            score, reason = _score_relevance(title, feed_cfg["name"], summary_plain)
            logger.info("  [%.2f] %s", score, title[:80])

            threshold = feed_cfg.get("threshold", RELEVANCE_THRESHOLD)
            if score < threshold:
                skipped_irrelevant += 1
                continue

            logger.info("  → Relevant (%s)", reason)

            if dry_run:
                ingested.append({"title": title, "source": feed_cfg["name"],
                                  "url": link, "score": score, "dry_run": True})
                continue

            # Fetch full text
            full_text = _fetch_full_text(link) if link else None
            if not full_text or len(full_text.strip()) < 300:
                logger.warning("  Skipping — could not fetch usable text from %s", link)
                continue

            # Ingest
            try:
                doc_id = _ingest_item(
                    title=title,
                    doc_type=feed_cfg["default_type"],
                    jurisdiction=feed_cfg["jurisdiction"],
                    issuer=feed_cfg["default_issuer"],
                    url=link,
                    text=full_text,
                    published=published,
                )
                if doc_id:
                    ingested.append({"title": title, "source": feed_cfg["name"],
                                      "url": link, "score": score, "doc_id": doc_id})
            except Exception as e:
                logger.error("  Failed to ingest '%s': %s", title, e)
                errors += 1

        # Update state with newly seen IDs (cap at 500 per feed to avoid unbounded growth)
        state[key] = new_seen[-500:]

    if not dry_run:
        _save_state(state)

    # Summary
    logger.info("Sweep complete: %d ingested, %d irrelevant, %d errors",
                len(ingested), skipped_irrelevant, errors)

    if not ingested:
        logger.info("Nothing new to report — no email sent")
        return

    # Email digest
    lines = [
        f"Compliance Monitor — {now}",
        f"{len(ingested)} new item{'s' if len(ingested) != 1 else ''} added to your knowledge base.",
        "",
    ]
    for i, item in enumerate(ingested, 1):
        tag = " [DRY RUN]" if item.get("dry_run") else ""
        lines.append(f"{i}. {item['title']}{tag}")
        lines.append(f"   Source: {item['source']} | Relevance: {item['score']:.0%}")
        lines.append(f"   {item['url']}")
        lines.append("")

    lines += [
        f"Stats: {skipped_irrelevant} items filtered as irrelevant, {errors} fetch errors.",
        "",
        "— Compliance Monitor",
    ]

    body = "\n".join(lines)

    if dry_run:
        print("\n" + body)
    else:
        subject = f"Compliance Monitor: {len(ingested)} new item{'s' if len(ingested) != 1 else ''} ({now})"
        _send_email(subject, body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Check feeds without ingesting or emailing")
    parser.add_argument("--reset", action="store_true",
                        help="Clear monitor state (reprocesses all items next run)")
    args = parser.parse_args()

    if args.reset:
        STATE_FILE.unlink(missing_ok=True)
        print("Monitor state cleared.")
        sys.exit(0)

    run_sweep(dry_run=args.dry_run)
