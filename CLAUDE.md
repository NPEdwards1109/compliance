# Compliance MCP Server

Knowledge graph of AI regulations, standards, and compliance content, exposed via MCP.

## Architecture

FastMCP 2.x server over HTTP. Single-user, bearer token auth. SQLite database with FTS5 and
JSON-serialized OpenAI embeddings for semantic search.

### Stack
- **Framework**: FastMCP 2.x + FastAPI
- **Database**: SQLite (`data/compliance.db`)
- **Search**: FTS5 (keyword) + cosine similarity over OpenAI embeddings (semantic)
- **LLM extraction**: Anthropic claude-sonnet-4-6 (ingest_document auto_extract)
- **Embeddings**: OpenAI text-embedding-3-small
- **Port**: 8001 (Thoth is on 8000)

### File layout

```
app/
  main.py          FastMCP assembly + middleware
  auth.py          Bearer token middleware (single API key via COMPLIANCE_API_KEY env var)
  database.py      SQLAlchemy engine, session factory, FTS5 init
  models.py        Document, Section, Requirement, DocRelationship, Topic, DocTopic
  embeddings.py    OpenAI embedding wrapper + cosine similarity
  tools/
    search.py      search_regulations, search_requirements
    retrieve.py    list_documents, get_document, get_sections, get_section, get_requirements
    graph.py       get_related, link_documents, get_topic_coverage, list_topics, tag_document_topic
    compliance.py  get_compliance_checklist, get_timeline, compare_jurisdictions
    ingest.py      ingest_document, update_document_status, delete_document
```

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in your keys
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

## Docker

```bash
docker compose up -d
```

## Claude Desktop config

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "compliance": {
      "url": "http://localhost:8001/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_COMPLIANCE_API_KEY"
      }
    }
  }
}
```

Or if deployed via Cloudflare Tunnel:
```json
{
  "mcpServers": {
    "compliance": {
      "url": "https://compliance.yourdomain.com/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_COMPLIANCE_API_KEY"
      }
    }
  }
}
```

## Schema

### Core tables
- **documents** — regulations, standards, guidance docs, articles, enforcement actions
- **sections** — document sections/articles with content and embeddings
- **requirements** — atomic obligations extracted from regulatory text
- **doc_relationships** — directed graph edges between documents
- **topics** — topic taxonomy
- **doc_topics** — document-topic associations

### FTS5 virtual tables (auto-synced via triggers)
- **documents_fts** — indexes title + summary + full_text
- **requirements_fts** — indexes requirement text

### Relationship types
`CITES`, `AMENDS`, `IMPLEMENTS`, `SUPERSEDES`, `RELATED_TO`, `ANALYZED_BY`

### Jurisdiction codes
`EU`, `US-Federal`, `US-CO`, `US-TX`, `US-CA`, `UK`, `Global`

## Seeding the corpus

Use `ingest_document` with `auto_extract=True` to add documents. Claude will automatically
extract a summary, sections, and requirements from the full text. Then use `link_documents`
to connect related documents in the graph.

Priority seed order:
1. EU AI Act (2024)
2. NIST AI RMF 1.0 + Playbook
3. ISO/IEC 42001:2023
4. US EO 14110 (Biden, 2023) + EO 14179 (Trump, 2025)
5. Colorado AI Act (SB 24-205)
6. Texas AI law
7. California AI bills
8. FTC AI guidance
9. UK AI Safety Institute frameworks
10. FDA AI/ML guidance, SEC AI guidance

## Migration strategy

Schema changes: add a `_migrate_vN()` function in database.py and call it from `init_db()`.
Always additive (ALTER TABLE ADD COLUMN or CREATE TABLE IF NOT EXISTS). Never destructive.

## Deployment notes

- Port 8001 (Thoth runs on 8000 — no conflict on the same droplet)
- COMPLIANCE_API_KEY must be set; server rejects all requests without it
- Database file: `data/compliance.db` (Docker volume mount at `/app/data`)
- Cloudflare Tunnel: same approach as Thoth — add a new tunnel route for compliance subdomain
- Email from address: add `compliance@ardalab.observer` routing rule in Cloudflare Email Routing

## Monitor setup

`monitor.py` sweeps RSS feeds daily, scores relevance via Claude, ingests new content, and
sends an email digest. Runs as a system crontab entry on the droplet (not a separate container).

**Required .env vars** (same Gmail App Password as Thoth):
```
GMAIL_FROM=jeff.edwards.11@gmail.com
GMAIL_APP_PASSWORD=...
COMPLIANCE_FROM_EMAIL=compliance@npedwards.com
COMPLIANCE_TO_EMAIL=nick.edwards1988@gmail.com
```

**Add to droplet system crontab** (`crontab -e`):
```
0 2 * * * cd /path/to/compliance && docker compose exec -T compliance python monitor.py >> /path/to/compliance/data/monitor.log 2>&1
```

**Manual runs:**
```bash
python monitor.py --dry-run   # check feeds, print digest, no ingest/email
python monitor.py             # full sweep
python monitor.py --reset     # clear state (reprocesses all items on next run)
```

**Feeds monitored:**
- Federal Register (AI rules, proposed rules, notices)
- FTC press releases
- NIST news
- EUR-Lex Official Journal
- UK ICO news
- UK AI Safety Institute

State tracked in `data/monitor_state.json` — last 500 seen item IDs per feed.
