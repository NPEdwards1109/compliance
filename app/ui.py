"""Graph visualization UI at /ui with session-cookie auth."""
from __future__ import annotations

import hashlib
import hmac
import os
import time

import httpx
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse, Response
from sqlalchemy import func, text as sa_text

from app.database import get_session, engine
from app.models import Document, DocRelationship, Requirement

router = APIRouter()

_COOKIE = "compliance_session"
_COOKIE_MAX_AGE = 30 * 86400  # 30 days


def _secret() -> bytes:
    return os.getenv("COMPLIANCE_API_KEY", "fallback").encode()


def _make_token() -> str:
    expires = int(time.time()) + _COOKIE_MAX_AGE
    msg = f"1.{expires}"
    sig = hmac.new(_secret(), msg.encode(), hashlib.sha256).hexdigest()
    return f"{msg}.{sig}"


def _verify_token(token: str) -> bool:
    parts = token.split(".")
    if len(parts) != 3:
        return False
    user_str, exp_str, sig = parts
    try:
        if time.time() > int(exp_str):
            return False
    except ValueError:
        return False
    expected = hmac.new(_secret(), f"{user_str}.{exp_str}".encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)


def _authed(request: Request) -> bool:
    return _verify_token(request.cookies.get(_COOKIE, ""))


# ---------------------------------------------------------------------------
# Login / logout
# ---------------------------------------------------------------------------

_LOGIN_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Compliance — Login</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@500;600;700&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: Inter, -apple-system, sans-serif; background: #F3EBDD; display: flex; align-items: center; justify-content: center; height: 100vh; }
.card { background: #fff; border: 1px solid #E3DDD2; border-radius: 4px; padding: 40px; width: 360px; }
h1 { font-family: Montserrat, sans-serif; font-size: 22px; font-weight: 700; color: #2F4057; letter-spacing: 0.03em; margin-bottom: 28px; }
label { display: block; font-size: 11px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.07em; color: #7A746A; margin-bottom: 6px; margin-top: 16px; }
label:first-of-type { margin-top: 0; }
input { width: 100%; padding: 9px 12px; border-radius: 4px; border: 1px solid #E3DDD2; background: #fff; color: #222427; font-size: 14px; font-family: Inter, sans-serif; }
input:focus { outline: none; border-color: #2F4057; }
button { margin-top: 24px; width: 100%; padding: 10px; border-radius: 4px; border: none; background: #2F4057; color: #ffffff; font-size: 13px; font-weight: 500; cursor: pointer; font-family: Inter, sans-serif; }
button:hover { background: #3d5270; }
.err { margin-top: 12px; font-size: 13px; color: #dc2626; text-align: center; }
</style>
</head>
<body>
<div class="card">
  <h1>Compliance</h1>
  <form method="post" action="/ui/login">
    <label for="username">Username</label>
    <input type="text" id="username" name="username" autocomplete="username" autofocus />
    <label for="password">Password</label>
    <input type="password" id="password" name="password" autocomplete="current-password" />
    <button type="submit">Sign in</button>
    {error}
  </form>
</div>
</body>
</html>
"""


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if _authed(request):
        return RedirectResponse("/ui/", status_code=302)
    return _LOGIN_HTML.replace("{error}", "")


@router.post("/login")
async def login_submit(request: Request):
    form = await request.form()
    username = (form.get("username") or "").strip()
    password = (form.get("password") or "").strip()
    expected_user = os.getenv("UI_USERNAME", "")
    expected_pass = os.getenv("UI_PASSWORD", "")
    user_ok = bool(expected_user) and hmac.compare_digest(username, expected_user)
    pass_ok = bool(expected_pass) and hmac.compare_digest(password, expected_pass)
    if not user_ok or not pass_ok:
        return HTMLResponse(_LOGIN_HTML.replace("{error}", '<div class="err">Incorrect credentials.</div>'), status_code=401)
    token = _make_token()
    response = RedirectResponse("/ui/", status_code=302)
    response.set_cookie(_COOKIE, token, max_age=_COOKIE_MAX_AGE, httponly=True, samesite="lax", secure=True, path="/")
    return response


@router.get("/logout")
async def logout():
    response = RedirectResponse("/ui/login", status_code=302)
    response.delete_cookie(_COOKIE, path="/")
    return response


# ---------------------------------------------------------------------------
# Graph data API
# ---------------------------------------------------------------------------

_JURISDICTION_COLORS = {
    "EU": "#2563eb", "US-Federal": "#dc2626", "US-CO": "#ea580c",
    "US-TX": "#16a34a", "US-CA": "#7c3aed", "UK": "#0891b2", "Global": "#6b7280",
}
_DOCTYPE_SHAPES = {
    "regulation": "rectangle", "standard": "hexagon",
    "guidance": "ellipse", "article": "diamond", "enforcement_action": "star",
}


@router.get("/graph")
def graph_data(request: Request):
    if not _authed(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    with get_session() as session:
        docs = session.query(Document).order_by(Document.effective_date).all()
        edges = session.query(DocRelationship).all()
        req_counts = dict(
            session.query(Requirement.doc_id, func.count(Requirement.id))
            .group_by(Requirement.doc_id).all()
        )
        nodes = [
            {
                "id": d.id, "title": d.title, "doc_type": d.doc_type,
                "jurisdiction": d.jurisdiction, "issuer": d.issuer,
                "status": d.status, "effective_date": d.effective_date,
                "summary": (d.summary or "")[:600],
                "requirement_count": req_counts.get(d.id, 0),
                "url": d.url or "",
                "color": _JURISDICTION_COLORS.get(d.jurisdiction, "#6b7280"),
                "shape": _DOCTYPE_SHAPES.get(d.doc_type, "ellipse"),
            }
            for d in docs
        ]
        edge_list = [
            {"id": str(e.id), "source": e.from_id, "target": e.to_id,
             "relationship": e.relationship, "notes": e.notes or ""}
            for e in edges
        ]
        return {"nodes": nodes, "edges": edge_list}


@router.get("/requirements")
def requirements_list(request: Request):
    if not _authed(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    with get_session() as session:
        rows = (
            session.query(Requirement, Document)
            .join(Document, Requirement.doc_id == Document.id)
            .order_by(Requirement.obligation_type, Document.title, Requirement.id)
            .all()
        )
        result = []
        for r, d in rows:
            applies = r.applies_to
            if isinstance(applies, str):
                import json as _json
                try:
                    applies = _json.loads(applies)
                except Exception:
                    applies = []
            result.append({
                "id": r.id,
                "text": (r.text or "")[:400],
                "obligation_type": r.obligation_type or "MUST",
                "applies_to": applies or [],
                "risk_level": r.risk_level or "",
                "doc_id": d.id,
                "doc_title": d.title,
                "doc_jurisdiction": d.jurisdiction,
                "doc_color": _JURISDICTION_COLORS.get(d.jurisdiction, "#6b7280"),
            })
        return result


# ---------------------------------------------------------------------------
# Concept browse API
# ---------------------------------------------------------------------------

@router.get("/concept")
def concept_browse(request: Request, name: str = ""):
    if not _authed(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not name.strip():
        return []
    import json as _json
    safe_q = '"' + name.replace('"', '').strip() + '"'
    rows = []
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                sa_text(
                    "SELECT r.id, r.text, r.obligation_type, r.applies_to, "
                    "d.id AS doc_id, d.title AS doc_title, d.jurisdiction, d.doc_type, d.status "
                    "FROM requirements_fts f "
                    "JOIN requirements r ON r.rowid = f.rowid "
                    "JOIN documents d ON d.id = r.doc_id "
                    "WHERE requirements_fts MATCH :q "
                    "ORDER BY rank, d.jurisdiction, d.title LIMIT 200"
                ),
                {"q": safe_q},
            ).fetchall()
    except Exception:
        pass
    if not rows:
        with engine.connect() as conn:
            rows = conn.execute(
                sa_text(
                    "SELECT r.id, r.text, r.obligation_type, r.applies_to, "
                    "d.id AS doc_id, d.title AS doc_title, d.jurisdiction, d.doc_type, d.status "
                    "FROM requirements r JOIN documents d ON d.id = r.doc_id "
                    "WHERE lower(r.text) LIKE :q "
                    "ORDER BY d.jurisdiction, d.title, r.obligation_type LIMIT 200"
                ),
                {"q": f"%{name.lower()}%"},
            ).fetchall()
    docs: dict = {}
    doc_order: list = []
    for row in rows:
        did = row.doc_id
        if did not in docs:
            docs[did] = {
                "doc_id": did, "doc_title": row.doc_title,
                "doc_jurisdiction": row.jurisdiction, "doc_type": row.doc_type,
                "doc_status": row.status or "",
                "doc_color": _JURISDICTION_COLORS.get(row.jurisdiction, "#6b7280"),
                "requirements": [],
            }
            doc_order.append(did)
        applies = row.applies_to
        if isinstance(applies, str):
            try: applies = _json.loads(applies)
            except: applies = []
        docs[did]["requirements"].append({
            "id": row.id,
            "text": (row.text or "")[:300],
            "obligation_type": row.obligation_type or "MUST",
        })
    return [docs[did] for did in doc_order]


# ---------------------------------------------------------------------------
# Standalone document page
# ---------------------------------------------------------------------------

_REL_OUT_LABELS = {
    "SUPERSEDES": "Supersedes", "IMPLEMENTS": "Implements",
    "CITES": "Cites", "AMENDS": "Amends",
    "RELATED_TO": "Related to", "ANALYZED_BY": "Analyzed by",
}
_REL_IN_LABELS = {
    "SUPERSEDES": "Superseded by", "IMPLEMENTS": "Implemented by",
    "CITES": "Cited by", "AMENDS": "Amended by",
    "RELATED_TO": "Related to", "ANALYZED_BY": "Analyzes",
}


def _conn_card_html(doc: dict, notes: str) -> str:
    from html import escape as he
    color = _JURISDICTION_COLORS.get(doc["jurisdiction"], "#6b7280")
    notes_h = f'<div class="conn-notes">{he(notes)}</div>' if notes else ""
    return (
        f'<a href="/ui/doc/{doc["id"]}" class="conn-card">'
        f'<span class="jbadge" style="background:{color}">{he(doc["jurisdiction"] or "")}</span>'
        f'<div class="conn-body"><div class="conn-title">{he(doc["title"] or "")}</div>{notes_h}</div>'
        f'</a>'
    )


def _doc_page_html(doc: dict, req_groups: dict, out_conns: list, in_conns: list) -> str:
    from html import escape as he

    color = _JURISDICTION_COLORS.get(doc["jurisdiction"], "#6b7280")
    status_cls = {
        "enacted": "chip-enacted", "proposed": "chip-proposed",
        "superseded": "chip-superseded", "draft": "chip-draft",
    }.get(doc["status"] or "", "chip-meta")

    chips = (
        f'<span class="chip chip-jur" style="background:{color}">{he(doc["jurisdiction"] or "")}</span>'
        f'<span class="chip chip-type">{he(doc["doc_type"] or "")}</span>'
        + (f'<span class="chip {status_cls}">{he(doc["status"])}</span>' if doc["status"] else "")
        + (f'<span class="chip chip-meta">{he(doc["effective_date"])}</span>' if doc["effective_date"] else "")
        + (f'<span class="chip chip-meta">{he(doc["issuer"])}</span>' if doc["issuer"] else "")
    )

    summary_h = f'<p class="doc-summary">{he(doc["summary"] or "")}</p>' if doc["summary"] else ""
    source_h = (
        f'<a class="source-link" href="{he(doc["url"])}" target="_blank" rel="noopener">View source &#8594;</a>'
        if doc["url"] else ""
    )

    # Connections
    all_rels = sorted(set(c["rel"] for c in out_conns + in_conns))
    conn_parts = []
    for rel in all_rels:
        outs = [c for c in out_conns if c["rel"] == rel]
        ins_ = [c for c in in_conns if c["rel"] == rel]
        if outs:
            cards = "".join(_conn_card_html(c["doc"], c["notes"]) for c in outs)
            lbl = he(_REL_OUT_LABELS.get(rel, rel))
            conn_parts.append(
                f'<div class="conn-group">'
                f'<div class="rel-hd"><span class="rel-arrow">&#8594;</span> {lbl} ({len(outs)})</div>'
                f'{cards}</div>'
            )
        if ins_:
            cards = "".join(_conn_card_html(c["doc"], c["notes"]) for c in ins_)
            lbl = he(_REL_IN_LABELS.get(rel, rel))
            conn_parts.append(
                f'<div class="conn-group">'
                f'<div class="rel-hd"><span class="rel-arrow">&#8592;</span> {lbl} ({len(ins_)})</div>'
                f'{cards}</div>'
            )
    total_conns = len(out_conns) + len(in_conns)
    conn_h = "".join(conn_parts) or '<p class="empty-note">No connections recorded.</p>'

    # Requirements
    OBL_ORDER = ["MUST", "SHOULD", "MAY"]
    req_parts = []
    for obl in OBL_ORDER:
        items = req_groups.get(obl, [])
        if not items:
            continue
        rows = ""
        for r in items:
            applies_h = (
                f'<div class="req-applies">{he(", ".join(r["applies_to"]))}</div>'
                if r.get("applies_to") else ""
            )
            rows += (
                f'<div class="req-item">'
                f'<span class="obl-badge obl-{obl}">{obl}</span>'
                f'<div class="req-body"><div class="req-text">{he(r["text"])}</div>{applies_h}</div>'
                f'</div>'
            )
        req_parts.append(
            f'<div class="obl-group">'
            f'<div class="obl-hd">{obl} <span class="obl-count">({len(items)})</span></div>'
            f'{rows}</div>'
        )
    total_reqs = sum(len(v) for v in req_groups.values())
    req_h = "".join(req_parts) or '<p class="empty-note">No requirements extracted.</p>'

    title_e = he(doc["title"] or "Document")
    chat_href = f"/ui/?doc={doc['id']}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title_e} &#8212; Compliance</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{{--navy:#2F4057;--navy-m:#3d5270;--stone:#A79F93;--stone-d:#8a8278;--stone-s:#EDE5D8;--line:#E3DDD2;--cream:#F3EBDD;--white:#ffffff;--charcoal:#222427;--muted:#7A746A}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:Inter,-apple-system,sans-serif;background:var(--cream);color:var(--charcoal)}}
a{{color:inherit;text-decoration:none}}
#topbar{{background:var(--navy);padding:0 32px;height:48px;display:flex;align-items:center;gap:12px;position:sticky;top:0;z-index:10;box-shadow:0 1px 0 rgba(0,0,0,.15)}}
.logo-mark{{width:24px;height:24px;border-radius:4px;display:flex;align-items:center;justify-content:center;flex-shrink:0}}
.logo-title{{font-size:13px;font-weight:600;color:rgba(255,255,255,.9)}}
.sep{{color:rgba(255,255,255,.25);font-size:12px}}
.back-link{{font-size:12px;font-weight:500;color:rgba(255,255,255,.55)}}
.back-link:hover{{color:#fff}}
.topbar-right{{margin-left:auto}}
.btn-outline{{padding:5px 14px;border-radius:4px;border:1px solid rgba(255,255,255,.25);color:rgba(255,255,255,.7);font-size:12px;font-weight:500;font-family:Inter,sans-serif;cursor:pointer;background:transparent;display:inline-block}}
.btn-outline:hover{{background:rgba(255,255,255,.08);color:#fff;border-color:rgba(255,255,255,.4)}}
main{{max-width:820px;margin:0 auto;padding:48px 32px 96px}}
.chips{{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:18px}}
.chip{{padding:3px 9px;border-radius:4px;font-size:11px;font-weight:500}}
.chip-jur{{color:#fff}}
.chip-type{{background:var(--cream);border:1px solid var(--line);color:var(--muted)}}
.chip-enacted{{background:#f0fdf4;color:#166534}}
.chip-proposed{{background:var(--stone-s);color:var(--muted)}}
.chip-superseded{{background:var(--cream);color:var(--muted);border:1px solid var(--line)}}
.chip-draft{{background:var(--stone-s);color:var(--muted)}}
.chip-meta{{background:var(--cream);border:1px solid var(--line);color:var(--muted)}}
.doc-title{{font-family:Montserrat,sans-serif;font-size:28px;font-weight:700;color:var(--navy);letter-spacing:.01em;line-height:1.3;margin-bottom:18px}}
.doc-summary{{font-size:14px;color:var(--muted);line-height:1.75;margin-bottom:18px}}
.source-link{{display:inline-flex;align-items:center;gap:4px;font-size:13px;font-weight:500;color:var(--navy);text-decoration:underline;text-underline-offset:3px;margin-bottom:44px}}
.source-link:hover{{color:var(--stone-d)}}
.section{{margin-bottom:52px}}
.section-hd{{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);padding-bottom:12px;border-bottom:1px solid var(--line);margin-bottom:20px}}
.conn-group{{margin-bottom:18px}}
.rel-hd{{font-size:12px;font-weight:600;color:var(--charcoal);margin-bottom:8px;display:flex;align-items:center;gap:6px}}
.rel-arrow{{font-size:14px;color:var(--stone)}}
.conn-card{{display:flex;align-items:flex-start;gap:10px;padding:11px 14px;border:1px solid var(--line);border-radius:4px;background:var(--white);margin-bottom:4px;cursor:pointer;transition:background .1s}}
.conn-card:hover{{background:var(--cream)}}
.jbadge{{padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600;color:#fff;flex-shrink:0;margin-top:2px}}
.conn-body{{flex:1;min-width:0}}
.conn-title{{font-size:14px;font-weight:500;color:var(--charcoal);line-height:1.4}}
.conn-notes{{font-size:12px;color:var(--muted);font-style:italic;margin-top:4px}}
.obl-group{{margin-bottom:28px}}
.obl-hd{{font-size:12px;font-weight:700;color:var(--charcoal);margin-bottom:10px;text-transform:uppercase;letter-spacing:.06em}}
.obl-count{{font-weight:400;letter-spacing:0;text-transform:none;color:var(--muted)}}
.req-item{{display:flex;align-items:flex-start;gap:10px;padding:10px 14px;border:1px solid var(--line);border-radius:4px;background:var(--white);margin-bottom:4px}}
.obl-badge{{font-size:10px;font-weight:700;letter-spacing:.05em;padding:2px 6px;border-radius:3px;flex-shrink:0;margin-top:2px}}
.obl-MUST{{background:var(--stone-s);color:var(--stone-d)}}
.obl-SHOULD{{background:var(--cream);color:var(--muted);border:1px solid var(--line)}}
.obl-MAY{{background:var(--cream);color:var(--muted);border:1px solid var(--line)}}
.req-body{{flex:1;min-width:0}}
.req-text{{font-size:13px;color:var(--charcoal);line-height:1.6}}
.req-applies{{font-size:11px;color:var(--muted);margin-top:5px;font-style:italic}}
.empty-note{{font-size:13px;color:var(--muted);font-style:italic;padding:8px 0}}
</style>
</head>
<body>
<div id="topbar">
  <div class="logo-mark"><svg xmlns="http://www.w3.org/2000/svg" width="34" height="26" viewBox="120 130 272 172"><g transform="translate(144,156) scale(1.25)"><rect x="0" y="0" width="12" height="120" fill="#FFFFFF"/><polygon points="14,0 30,0 96,120 80,120" fill="#FFFFFF"/><rect x="84" y="0" width="12" height="120" fill="#FFFFFF"/><rect x="112" y="0" width="68" height="12" fill="#A79F93"/><rect x="112" y="54" width="58" height="12" fill="#A79F93"/><rect x="112" y="108" width="68" height="12" fill="#A79F93"/></g></svg></div>
  <span class="logo-title">Compliance</span>
  <span class="sep">&#183;</span>
  <a href="/ui/" class="back-link">&#8592; Knowledge Graph</a>
  <div class="topbar-right">
    <a href="{chat_href}" class="btn-outline">Chat about this &#8594;</a>
  </div>
</div>
<main>
  <div class="chips">{chips}</div>
  <h1 class="doc-title">{title_e}</h1>
  {summary_h}
  {source_h}
  <div class="section">
    <div class="section-hd">Connections ({total_conns})</div>
    {conn_h}
  </div>
  <div class="section">
    <div class="section-hd">Requirements ({total_reqs})</div>
    {req_h}
  </div>
</main>
</body>
</html>"""


@router.get("/doc/{doc_id}", response_class=HTMLResponse)
async def doc_standalone(doc_id: str, request: Request):
    if not _authed(request):
        return RedirectResponse("/ui/login", status_code=302)
    import json as _json
    from collections import defaultdict

    def _doc_dict(d: Document) -> dict:
        return {
            "id": d.id, "title": d.title, "doc_type": d.doc_type,
            "jurisdiction": d.jurisdiction, "issuer": d.issuer,
            "status": d.status, "effective_date": d.effective_date,
            "url": d.url, "summary": d.summary,
        }

    with get_session() as session:
        doc_row = session.query(Document).filter_by(id=doc_id).first()
        if not doc_row:
            return HTMLResponse(
                "<html><body style='font-family:sans-serif;padding:40px'>"
                "<h2>Document not found</h2><p><a href='/ui/'>&#8592; Back</a></p></body></html>",
                status_code=404,
            )
        doc = _doc_dict(doc_row)
        raw_reqs = (
            session.query(Requirement)
            .filter_by(doc_id=doc_id)
            .order_by(Requirement.obligation_type, Requirement.id)
            .all()
        )
        reqs = [
            {"obligation_type": r.obligation_type, "text": r.text, "applies_to": r.applies_to}
            for r in raw_reqs
        ]
        raw_out = session.query(DocRelationship).filter_by(from_id=doc_id).all()
        raw_in = session.query(DocRelationship).filter_by(to_id=doc_id).all()
        out_edges = [{"rel": e.relationship, "to_id": e.to_id, "notes": e.notes or ""} for e in raw_out]
        in_edges = [{"rel": e.relationship, "from_id": e.from_id, "notes": e.notes or ""} for e in raw_in]
        related_ids = {e["to_id"] for e in out_edges} | {e["from_id"] for e in in_edges}
        related_ids.discard(doc_id)
        related = {
            d.id: _doc_dict(d)
            for d in session.query(Document).filter(Document.id.in_(list(related_ids))).all()
        } if related_ids else {}

    req_groups: dict = defaultdict(list)
    for r in reqs:
        obl = r["obligation_type"] or "MUST"
        applies = r["applies_to"]
        if isinstance(applies, str):
            try: applies = _json.loads(applies)
            except: applies = []
        req_groups[obl].append({"text": r["text"] or "", "applies_to": applies or []})
    out_conns = [
        {"rel": e["rel"], "doc": related[e["to_id"]], "notes": e["notes"]}
        for e in out_edges if e["to_id"] in related
    ]
    in_conns = [
        {"rel": e["rel"], "doc": related[e["from_id"]], "notes": e["notes"]}
        for e in in_edges if e["from_id"] in related
    ]
    return _doc_page_html(doc, dict(req_groups), out_conns, in_conns)


# ---------------------------------------------------------------------------
# Chat API
# ---------------------------------------------------------------------------

def _build_chat_context(doc_id: str | None) -> str:
    with get_session() as session:
        parts = []
        if doc_id:
            doc = session.query(Document).filter_by(id=doc_id).first()
            if doc:
                parts.append(
                    f"# {doc.title}\nJurisdiction: {doc.jurisdiction} | Status: {doc.status} | Effective: {doc.effective_date}\n\n{doc.summary or ''}"
                )
                reqs = session.query(Requirement).filter_by(doc_id=doc_id).limit(25).all()
                if reqs:
                    parts.append("Key requirements:\n" + "\n".join(f"• {r.text[:300]}" for r in reqs))
        core = (
            session.query(Document)
            .filter(Document.doc_type.in_(["regulation", "standard"]))
            .order_by(Document.effective_date)
            .all()
        )
        for doc in core:
            if doc_id and doc.id == doc_id:
                continue
            parts.append(f"## {doc.title} ({doc.jurisdiction})\n{(doc.summary or '')[:500]}")
        return "\n\n---\n\n".join(parts)


_CHAT_SYSTEM = (
    "You are a compliance analyst specializing in AI regulation. "
    "Answer questions concisely based on the regulatory context provided. "
    "Cite specific regulations and article numbers when relevant. "
    "If a question falls outside the provided context, say so briefly."
)


@router.post("/chat")
async def chat(request: Request):
    if not _authed(request):
        return Response("Unauthorized", status_code=401)
    body = await request.json()
    question = (body.get("question") or "").strip()
    doc_id = body.get("doc_id")
    if not question:
        return JSONResponse({"error": "Empty question"}, status_code=400)

    context = _build_chat_context(doc_id)
    proxy_url = os.getenv("PROXY_BASE_URL", "https://proxy.npedwards.com/v1")
    proxy_key = os.getenv("PROXY_API_KEY", "")

    payload = {
        "model": "claude-sonnet-4-6",
        "stream": True,
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": _CHAT_SYSTEM},
            {"role": "user", "content": f"Regulatory context:\n\n{context}\n\nQuestion: {question}"},
        ],
    }

    async def generate():
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{proxy_url}/chat/completions",
                headers={"Authorization": f"Bearer {proxy_key}", "Content-Type": "application/json"},
                json=payload,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and "[DONE]" not in line:
                        yield f"{line}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

_UI_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Compliance — AI Regulation Knowledge Graph</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --navy: #2F4057;
  --navy-mid: #3d5270;
  --stone: #A79F93;
  --stone-d: #8a8278;
  --stone-s: #EDE5D8;
  --white: #ffffff;
  --surface: #F3EBDD;
  --border: #E3DDD2;
  --text: #222427;
  --muted: #7A746A;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: Inter, -apple-system, sans-serif; display: flex; height: 100vh; background: var(--surface); color: var(--text); overflow: hidden; }

/* Nav */
#nav { width: 240px; min-width: 240px; background: var(--navy); display: flex; flex-direction: column; overflow: hidden; }
#nav-header { padding: 13px 14px; border-bottom: 1px solid rgba(255,255,255,0.08); display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
#nav-logo { display: flex; align-items: center; gap: 8px; }
#nav-mark { width: 38px; height: 30px; border-radius: 4px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
#nav-title { font-size: 13px; font-weight: 600; color: rgba(255,255,255,0.9); letter-spacing: 0.01em; }
#nav-signout { font-size: 11px; color: rgba(255,255,255,0.38); text-decoration: none; }
#nav-signout:hover { color: rgba(255,255,255,0.75); }
#nav-body { flex: 1; overflow-y: auto; padding-bottom: 10px; }
.group-label { padding: 14px 14px 4px; font-size: 10px; font-weight: 600; color: rgba(255,255,255,0.32); text-transform: uppercase; letter-spacing: 0.1em; }
.doc-item { padding: 7px 14px; cursor: pointer; border-left: 2px solid transparent; transition: background 0.1s; }
.doc-item:hover { background: rgba(255,255,255,0.07); }
.doc-item.selected { background: rgba(167,159,147,0.18); border-left-color: var(--stone); }
.doc-item.superseded { opacity: 0.42; }
.doc-item.superseded .di-title { text-decoration: line-through; }
.di-title { font-size: 12px; font-weight: 500; color: rgba(255,255,255,0.78); line-height: 1.4; }
.doc-item.selected .di-title { color: rgba(255,255,255,0.95); }
.di-meta { display: flex; gap: 5px; align-items: center; margin-top: 3px; flex-wrap: wrap; }
.jbadge { padding: 1px 5px; border-radius: 3px; font-size: 10px; font-weight: 600; color: #fff; flex-shrink: 0; }
.di-date { font-size: 10px; color: rgba(255,255,255,0.28); }
.di-reqs { font-size: 10px; color: rgba(255,255,255,0.28); }

/* Main */
#main { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-width: 0; background: var(--white); }
#topbar { padding: 0 20px; height: 44px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 6px; flex-shrink: 0; background: var(--white); }
.tab-btn { padding: 5px 14px; border-radius: 4px; border: 1px solid transparent; background: transparent; color: var(--muted); font-family: Inter, sans-serif; font-size: 13px; font-weight: 500; cursor: pointer; transition: all 0.13s; }
.tab-btn:hover:not(.active) { background: var(--surface); color: var(--text); }
.tab-btn.active { background: var(--navy); color: #fff; border-color: var(--navy); }
#topbar-stats { margin-left: auto; font-size: 12px; color: var(--muted); }

/* Content */
#content { flex: 1; overflow: hidden; position: relative; }
#browse-panel { position: absolute; inset: 0; overflow-y: auto; padding: 36px 44px; display: none; background: var(--white); }
#browse-panel.active { display: block; }
#graph-panel { position: absolute; inset: 0; display: none; background: var(--navy); }
#graph-panel.active { display: block; }
#sigma-container { width: 100%; height: 100%; }
#graph-controls { position: absolute; bottom: 16px; right: 16px; display: flex; gap: 8px; }
#graph-controls button { padding: 6px 13px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.15); background: rgba(255,255,255,0.07); color: rgba(255,255,255,0.65); font-family: Inter, sans-serif; font-size: 12px; cursor: pointer; transition: all 0.12s; }
#graph-controls button:hover { background: rgba(255,255,255,0.13); color: #fff; }
#graph-loading { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: rgba(255,255,255,0.35); font-size: 14px; background: var(--navy); }

/* Welcome */
.welcome { max-width: 540px; }
.welcome h2 { font-family: Montserrat, sans-serif; font-size: 28px; font-weight: 600; color: var(--navy); letter-spacing: -0.02em; margin-bottom: 10px; }
.welcome p { color: var(--muted); font-size: 14px; line-height: 1.65; margin-bottom: 24px; }
.welcome p:last-child { margin-bottom: 0; margin-top: -8px; font-size: 13px; }
.stat-row { display: flex; gap: 14px; margin-bottom: 24px; }
.stat-box { border: 1px solid var(--border); border-radius: 4px; padding: 22px 24px; flex: 1; background: var(--white); border-top: 3px solid transparent; transition: border-top-color 0.15s; }
.stat-box:hover { border-top-color: var(--stone); }
.stat-box .sv { font-family: Montserrat, sans-serif; font-size: 32px; font-weight: 600; color: var(--navy); letter-spacing: -0.02em; }
.stat-box .sl { font-size: 11px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted); margin-top: 4px; }

/* Doc detail */
.doc-detail { max-width: 680px; }
.dd-title { font-family: Montserrat, sans-serif; font-size: 22px; font-weight: 600; color: var(--navy); line-height: 1.35; letter-spacing: -0.02em; margin-bottom: 12px; }
.chips { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 16px; }
.chip { padding: 3px 9px; border-radius: 4px; font-size: 11px; font-weight: 500; }
.chip-jur { color: #fff; }
.chip-type { background: var(--surface); border: 1px solid var(--border); color: var(--muted); }
.chip-enacted { background: #f0fdf4; color: #166534; }
.chip-proposed { background: var(--stone-s); color: var(--stone-d); }
.chip-superseded { background: var(--surface); color: var(--muted); border: 1px solid var(--border); }
.chip-draft { background: var(--stone-s); color: var(--stone-d); }
.chip-meta { background: var(--surface); border: 1px solid var(--border); color: var(--muted); }
.dd-summary { font-size: 14px; color: var(--muted); line-height: 1.65; margin-bottom: 16px; }
.source-link { display: inline-flex; align-items: center; gap: 4px; font-size: 13px; font-weight: 500; color: var(--navy); text-decoration: underline; text-underline-offset: 3px; margin-bottom: 28px; }
.source-link:hover { color: var(--stone-d); }
.section-title { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
.conn-group { margin-bottom: 18px; }
.rel-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin-bottom: 6px; }
.conn-item { display: flex; align-items: flex-start; gap: 10px; padding: 10px 12px; border-radius: 4px; background: var(--white); border: 1px solid var(--border); margin-bottom: 4px; cursor: pointer; transition: background 0.1s; }
.conn-item:hover { background: var(--surface); }
.conn-jbadge { padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: 600; color: #fff; flex-shrink: 0; margin-top: 1px; }
.conn-body { flex: 1; min-width: 0; }
.conn-title { font-size: 13px; font-weight: 500; color: var(--text); line-height: 1.4; }
.conn-link { font-size: 11px; font-weight: 500; color: var(--navy); text-decoration: underline; text-underline-offset: 2px; display: inline-block; margin-top: 3px; }
.conn-link:hover { color: var(--stone-d); }
.conn-notes { font-size: 11px; color: var(--muted); margin-top: 3px; font-style: italic; }
.no-conn { font-size: 13px; color: var(--muted); font-style: italic; padding: 8px 0; }

/* List views */
.list-view { max-width: 680px; }
.list-back { display: inline-flex; align-items: center; gap: 4px; font-size: 12px; color: var(--muted); cursor: pointer; margin-bottom: 18px; }
.list-back:hover { color: var(--text); }
.list-heading { font-family: Montserrat, sans-serif; font-size: 22px; font-weight: 600; color: var(--navy); letter-spacing: -0.02em; margin-bottom: 20px; }
.list-count { font-family: Inter, sans-serif; font-size: 14px; font-weight: 400; color: var(--muted); letter-spacing: 0; }
.list-group-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin: 18px 0 7px; padding-bottom: 7px; border-bottom: 1px solid var(--border); }
.list-item { display: flex; align-items: flex-start; gap: 10px; padding: 10px 12px; border-radius: 4px; border: 1px solid var(--border); background: var(--white); margin-bottom: 4px; cursor: pointer; transition: background 0.1s; }
.list-item:hover { background: var(--surface); }
.list-meta { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 5px; }
.list-date { font-size: 11px; color: var(--muted); }
.list-reqs { font-size: 11px; color: var(--muted); }
.req-text { font-size: 13px; color: var(--text); line-height: 1.5; }
.req-doc { font-size: 11px; font-weight: 500; color: var(--muted); }
.req-applies { font-size: 11px; color: var(--muted); font-style: italic; }
.list-show-more { font-size: 12px; font-weight: 500; color: var(--navy); cursor: pointer; padding: 8px 12px; border: 1px dashed var(--border); border-radius: 4px; text-align: center; margin: 4px 0 12px; transition: background 0.1s; }
.list-show-more:hover { background: var(--surface); }
.stat-box { cursor: pointer; }

/* Concept view */
.concept-view { max-width: 720px; }
.concept-doc-block { margin-bottom: 22px; }
.concept-doc-header { display: flex; align-items: center; gap: 8px; padding-bottom: 8px; border-bottom: 2px solid var(--border); margin-bottom: 8px; cursor: pointer; }
.concept-doc-header:hover .concept-doc-title { color: var(--navy); }
.concept-doc-title { font-size: 14px; font-weight: 600; color: var(--text); flex: 1; min-width: 0; }
.concept-doc-meta { font-size: 11px; color: var(--muted); white-space: nowrap; }
.concept-req { display: flex; align-items: flex-start; gap: 8px; padding: 9px 12px; margin-bottom: 4px; border: 1px solid var(--border); border-radius: 4px; background: var(--white); }
.obl-badge { font-size: 10px; font-weight: 700; letter-spacing: 0.05em; padding: 2px 6px; border-radius: 3px; flex-shrink: 0; margin-top: 1px; }
.obl-MUST { background: var(--stone-s); color: var(--stone-d); }
.obl-SHOULD { background: var(--surface); color: var(--muted); border: 1px solid var(--border); }
.obl-MAY { background: var(--surface); color: var(--muted); border: 1px solid var(--border); }
.concept-req-text { font-size: 13px; color: var(--text); line-height: 1.5; }

/* Chat */
#chat { height: 400px; min-height: 240px; border-top: 1px solid var(--border); display: flex; flex-direction: column; flex-shrink: 0; transition: height 0.15s ease; }
#chat.collapsed { height: auto; min-height: 0; }
#chat.collapsed #chat-messages, #chat.collapsed #chat-footer { display: none; }
#chat-header { padding: 7px 16px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 6px; flex-shrink: 0; background: var(--white); cursor: pointer; }
#chat-header:hover { background: var(--surface); }
#chat.collapsed #chat-header { border-bottom: none; }
#chat-header-label { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
#chat-ctx { font-size: 11px; font-weight: 500; color: var(--stone-d); background: var(--stone-s); padding: 2px 8px; border-radius: 4px; max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#chat-toggle { margin-left: auto; background: none; border: none; cursor: pointer; color: var(--muted); font-size: 9px; padding: 0 2px; line-height: 1; font-family: Inter, sans-serif; }
#chat-clear { font-size: 11px; color: var(--muted); background: none; border: none; cursor: pointer; padding: 0; font-family: Inter, sans-serif; }
#chat-clear:hover, #chat-toggle:hover { color: var(--text); }
#chat-messages { flex: 1; overflow-y: auto; padding: 12px 16px; display: flex; flex-direction: column; gap: 12px; background: var(--surface); }
.cmsg { font-size: 13px; line-height: 1.65; }
.cmsg.user { align-self: flex-end; background: var(--navy); color: #fff; padding: 8px 12px; border-radius: 4px; max-width: 72%; }
.cmsg.assistant { color: var(--text); align-self: flex-start; max-width: 100%; }
.cmsg.assistant.streaming { white-space: pre-wrap; }
.cmsg.assistant.streaming::after { content: '|'; animation: blink 1s step-end infinite; }
@keyframes blink { 50% { opacity: 0; } }
.cmsg.assistant p { margin-bottom: 9px; }
.cmsg.assistant p:last-child { margin-bottom: 0; }
.cmsg.assistant ul, .cmsg.assistant ol { margin: 4px 0 9px 18px; }
.cmsg.assistant li { margin-bottom: 3px; }
.cmsg.assistant li:last-child { margin-bottom: 0; }
.cmsg.assistant code { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace; font-size: 12px; background: var(--white); border: 1px solid var(--border); border-radius: 3px; padding: 1px 5px; }
.cmsg.assistant pre { background: var(--white); border: 1px solid var(--border); border-radius: 4px; padding: 12px 14px; margin: 8px 0; overflow-x: auto; }
.cmsg.assistant pre code { background: none; border: none; padding: 0; font-size: 12px; line-height: 1.55; }
.cmsg.assistant strong { font-weight: 600; }
.cmsg.assistant em { font-style: italic; }
.cmsg.assistant h1, .cmsg.assistant h2, .cmsg.assistant h3 { font-family: Montserrat, sans-serif; font-weight: 700; color: var(--navy); margin: 14px 0 6px; }
.cmsg.assistant h1 { font-size: 16px; }
.cmsg.assistant h2 { font-size: 15px; }
.cmsg.assistant h3 { font-size: 14px; }
.cmsg.assistant blockquote { border-left: 3px solid var(--border); padding-left: 12px; color: var(--muted); margin: 8px 0; font-style: italic; }
.cmsg.assistant hr { border: none; border-top: 1px solid var(--border); margin: 12px 0; }
.cmsg.assistant a { color: var(--navy); text-underline-offset: 2px; }
#chat-footer { padding: 8px 16px; display: flex; gap: 8px; align-items: flex-end; flex-shrink: 0; background: var(--white); border-top: 1px solid var(--border); }
#chat-input { flex: 1; background: var(--white); border: 1px solid var(--border); border-radius: 4px; color: var(--text); font-size: 13px; padding: 8px 12px; resize: none; font-family: Inter, sans-serif; line-height: 1.4; min-height: 38px; max-height: 120px; overflow-y: auto; }
#chat-input:focus { outline: none; border-color: var(--navy); }
#chat-input::placeholder { color: var(--muted); }
#chat-send { padding: 0 18px; height: 38px; border-radius: 4px; border: none; background: var(--navy); color: #fff; font-size: 13px; font-weight: 500; cursor: pointer; flex-shrink: 0; font-family: Inter, sans-serif; transition: background 0.12s; }
#chat-send:hover:not(:disabled) { background: var(--navy-mid); }
#chat-send:disabled { background: var(--surface); color: var(--muted); cursor: not-allowed; border: 1px solid var(--border); }
</style>
</head>
<body>

<div id="nav">
  <div id="nav-header">
    <div id="nav-logo">
      <div id="nav-mark"><svg xmlns="http://www.w3.org/2000/svg" width="36" height="28" viewBox="120 130 272 172"><g transform="translate(144,156) scale(1.25)"><rect x="0" y="0" width="12" height="120" fill="#FFFFFF"/><polygon points="14,0 30,0 96,120 80,120" fill="#FFFFFF"/><rect x="84" y="0" width="12" height="120" fill="#FFFFFF"/><rect x="112" y="0" width="68" height="12" fill="#A79F93"/><rect x="112" y="54" width="58" height="12" fill="#A79F93"/><rect x="112" y="108" width="68" height="12" fill="#A79F93"/></g></svg></div>
      <span id="nav-title">Compliance</span>
    </div>
    <a id="nav-signout" href="/ui/logout">Sign out</a>
  </div>
  <div id="nav-body"></div>
</div>

<div id="main">
  <div id="topbar">
    <button class="tab-btn active" id="tab-browse" onclick="switchTab('browse')">Browse</button>
    <button class="tab-btn" id="tab-graph" onclick="switchTab('graph')">Graph</button>
    <div id="topbar-stats"></div>
  </div>
  <div id="content">
    <div id="browse-panel" class="active">
      <div id="browse-content"></div>
    </div>
    <div id="graph-panel">
      <div id="graph-loading">Loading&hellip;</div>
      <div id="sigma-container"></div>
      <div id="graph-controls">
        <button onclick="resetCamera()">Fit</button>
        <button onclick="relayout()">Relayout</button>
      </div>
    </div>
  </div>
  <div id="chat" class="collapsed">
    <div id="chat-header" onclick="toggleChat()">
      <span id="chat-header-label">Ask about</span>
      <span id="chat-ctx">all regulations</span>
      <button id="chat-toggle" onclick="event.stopPropagation();toggleChat()">▼</button>
      <button id="chat-clear" onclick="event.stopPropagation();clearChat()">Clear</button>
    </div>
    <div id="chat-messages"></div>
    <div id="chat-footer">
      <textarea id="chat-input" placeholder="Ask about AI regulations&hellip;" rows="1"></textarea>
      <button id="chat-send" onclick="sendChat()">Send</button>
    </div>
  </div>
</div>

<script src="https://unpkg.com/graphology@0.25.4/dist/graphology.umd.min.js"></script>
<script src="https://unpkg.com/sigma@2.4.0/build/sigma.min.js"></script>
<script>
// ── State ─────────────────────────────────────────────────────────────────────
let data = null, nodeById = {}, edgesFrom = {}, edgesTo = {};
let selectedDocId = null, selectedConcept = null, currentTab = 'browse';
let graph = null, renderer = null, graphInited = false;
let hoveredNode = null, neighborSet = null, simHandle = null, simPos = null, simSizes = null;
let chatStreaming = false;
let _expandData = {};

const JUR_COLORS = {
  "EU": "#2563eb", "US-Federal": "#dc2626", "US-CO": "#ea580c",
  "US-TX": "#16a34a", "US-CA": "#7c3aed", "UK": "#0891b2", "Global": "#6b7280"
};
const REL_COLORS_GRAPH = {
  SUPERSEDES: "#f59e0b", IMPLEMENTS: "#10b981", CITES: "#8b5cf6",
  RELATED_TO: "#94a3b8", AMENDS: "#f97316", ANALYZED_BY: "#475569"
};
const REL_OUT_LABELS = {
  CITES: "Cites", IMPLEMENTS: "Implements", SUPERSEDES: "Supersedes",
  AMENDS: "Amends", RELATED_TO: "Related to", ANALYZED_BY: "Analyzed by"
};
const REL_IN_LABELS = {
  CITES: "Cited by", IMPLEMENTS: "Implemented by", SUPERSEDES: "Superseded by",
  AMENDS: "Amended by", RELATED_TO: "Related to", ANALYZED_BY: "Analyzed by"
};
const ACRONYMS = {
  "EU Artificial Intelligence Act": "EU AI Act",
  "NIST AI Risk Management Framework (AI RMF 1.0)": "NIST AI RMF",
  "NIST AI 600-1 — Generative AI Profile": "NIST AI 600-1",
  "ISO/IEC 42001:2023 — AI Management Systems": "ISO 42001",
  "Executive Order 14110 — Safe, Secure, and Trustworthy AI (Biden)": "EO 14110",
  "Executive Order 14179 — Removing Barriers to American Leadership in AI (Trump)": "EO 14179",
  "Colorado AI Act (SB 24-205) — Consumer Protections for AI": "Colorado AI Act",
  "FTC Policy Statement on AI and Consumer Protection": "FTC AI Policy"
};

const CONCEPTS = [
  "Human Oversight",
  "Transparency",
  "Risk Assessment",
  "Data Governance",
  "Conformity Assessment",
  "Technical Documentation",
  "Algorithmic Discrimination",
  "Cybersecurity",
  "Serious Incident",
  "Enforcement",
];

function shortTitle(title) {
  return ACRONYMS[title] || (title.length > 30 ? title.slice(0, 28) + '…' : title);
}

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
  const res = await fetch("/ui/graph");
  if (res.status === 401) { location.href = "/ui/login"; return; }
  data = await res.json();
  for (const n of data.nodes) {
    nodeById[n.id] = n;
    edgesFrom[n.id] = [];
    edgesTo[n.id] = [];
  }
  for (const e of data.edges) {
    if (edgesFrom[e.source]) edgesFrom[e.source].push(e);
    if (edgesTo[e.target]) edgesTo[e.target].push(e);
  }
  buildNav();
  document.getElementById("topbar-stats").textContent =
    data.nodes.length + " docs · " + data.edges.length + " relationships";
  const params = new URLSearchParams(window.location.search);
  const docParam = params.get('doc');
  if (docParam) {
    selectedDocId = docParam;
    const n = nodeById[docParam];
    if (n) {
      document.getElementById("chat-ctx").textContent = shortTitle(n.title);
      const item = document.querySelector('.doc-item[data-id="' + docParam + '"]');
      if (item) { item.classList.add("selected"); item.scrollIntoView({ block: "nearest" }); }
    }
    renderBrowse(docParam);
  } else {
    showWelcome();
  }
}

// ── Nav ───────────────────────────────────────────────────────────────────────
const TYPE_ORDER = [
  ["Regulations", "regulation"],
  ["Standards", "standard"],
  ["Guidance", "guidance"],
  ["Articles", "article"],
  ["Enforcement", "enforcement_action"]
];

function buildNav() {
  const body = document.getElementById("nav-body");
  body.innerHTML = "";

  // Concepts section
  const cGrp = document.createElement("div");
  const cHdr = document.createElement("div");
  cHdr.className = "group-label";
  cHdr.textContent = "Concepts";
  cGrp.appendChild(cHdr);
  for (const name of CONCEPTS) {
    const el = document.createElement("div");
    el.className = "doc-item";
    el.dataset.concept = name;
    el.innerHTML = '<div class="di-title">' + name + '</div>';
    el.onclick = function() { selectConcept(this.dataset.concept); };
    cGrp.appendChild(el);
  }
  body.appendChild(cGrp);

  const knownTypes = TYPE_ORDER.map(function(g) { return g[1]; });
  const groups = TYPE_ORDER.slice();
  const otherNodes = data.nodes.filter(function(n) { return !knownTypes.includes(n.doc_type); });
  if (otherNodes.length) groups.push(["Other", "__other__"]);

  for (const pair of groups) {
    const label = pair[0], type = pair[1];
    const nodes = type === "__other__"
      ? otherNodes
      : data.nodes.filter(function(n) { return n.doc_type === type; });
    if (!nodes.length) continue;

    const grp = document.createElement("div");
    const hdr = document.createElement("div");
    hdr.className = "group-label";
    hdr.textContent = label;
    grp.appendChild(hdr);

    for (const n of nodes) {
      const el = document.createElement("div");
      el.className = "doc-item" + (n.status === "superseded" ? " superseded" : "");
      el.dataset.id = n.id;
      const title = shortTitle(n.title);
      el.innerHTML = '<div class="di-title">' + title + '</div>' +
        '<div class="di-meta">' +
        '<span class="jbadge" style="background:' + n.color + '">' + n.jurisdiction + '</span>' +
        '<span class="di-date">' + (n.effective_date || '—') + '</span>' +
        '<span class="di-reqs">' + n.requirement_count + ' reqs</span>' +
        '</div>';
      el.onclick = function(id) { return function() { selectDoc(id); }; }(n.id);
      grp.appendChild(el);
    }
    body.appendChild(grp);
  }
}

// ── Selection ─────────────────────────────────────────────────────────────────
function selectDoc(id) {
  window.location.href = '/ui/doc/' + id;
}

function clearSelection() {
  selectedDocId = null;
  selectedConcept = null;
  document.querySelectorAll(".doc-item").forEach(function(el) { el.classList.remove("selected"); });
  document.getElementById("chat-ctx").textContent = "all regulations";
  if (currentTab === 'browse') showWelcome();
}

// ── List views ────────────────────────────────────────────────────────────────
function showDocumentList() {
  clearSelection();
  const sorted = data.nodes.slice().sort(function(a, b) {
    return (a.doc_type + a.title).localeCompare(b.doc_type + b.title);
  });
  const typeOrder = ["regulation","standard","guidance","article","enforcement_action"];
  const grouped = {};
  for (const n of sorted) {
    if (!grouped[n.doc_type]) grouped[n.doc_type] = [];
    grouped[n.doc_type].push(n);
  }
  const typeLabel = { regulation: "Regulations", standard: "Standards", guidance: "Guidance",
    article: "Articles", enforcement_action: "Enforcement Actions" };
  let html = '<div class="list-view">' +
    '<div class="list-back" onclick="showWelcome()">&#8592; Overview</div>' +
    '<div class="list-heading">Documents <span class="list-count">' + sorted.length + '</span></div>';
  const orderedTypes = typeOrder.filter(function(t) { return grouped[t] && grouped[t].length; });
  const otherTypes = Object.keys(grouped).filter(function(t) { return !typeOrder.includes(t); });
  for (const type of orderedTypes.concat(otherTypes)) {
    const nodes = grouped[type];
    if (!nodes || !nodes.length) continue;
    html += '<div class="list-group-label">' + (typeLabel[type] || type) + ' (' + nodes.length + ')</div>';
    for (const n of nodes) {
      html += '<div class="list-item" data-id="' + n.id + '" onclick="selectDoc(this.dataset.id)">' +
        '<span class="conn-jbadge" style="background:' + n.color + '">' + n.jurisdiction + '</span>' +
        '<div class="conn-body">' +
        '<div class="conn-title">' + n.title + '</div>' +
        '<div class="list-meta">' +
        '<span class="chip ' + statusChipClass(n.status) + '">' + n.status + '</span>' +
        (n.effective_date ? '<span class="list-date">' + n.effective_date + '</span>' : '') +
        '<span class="list-reqs">' + n.requirement_count + ' requirements</span>' +
        (n.url ? '<a class="conn-link" href="' + n.url + '" target="_blank" onclick="event.stopPropagation()">Source &#8594;</a>' : '') +
        '</div></div></div>';
    }
  }
  html += '</div>';
  document.getElementById("browse-content").innerHTML = html;
}

async function showRequirementList() {
  clearSelection();
  const el = document.getElementById("browse-content");
  el.innerHTML = '<div class="list-view">' +
    '<div class="list-back" onclick="showWelcome()">&#8592; Overview</div>' +
    '<div class="list-heading">Requirements <span class="list-count">Loading&#8230;</span></div></div>';
  let reqs;
  try {
    const res = await fetch("/ui/requirements");
    if (!res.ok) { el.innerHTML = '<p style="color:var(--muted);padding:20px">Error loading requirements.</p>'; return; }
    reqs = await res.json();
  } catch (e) {
    el.innerHTML = '<p style="color:var(--muted);padding:20px">Error: ' + e.message + '</p>'; return;
  }
  const ORDER = ["MUST","SHOULD","MAY"];
  const grouped = {};
  for (const r of reqs) {
    const t = r.obligation_type || "MUST";
    if (!grouped[t]) grouped[t] = [];
    grouped[t].push(r);
  }
  _expandData = {};
  let html = '<div class="list-view">' +
    '<div class="list-back" onclick="showWelcome()">&#8592; Overview</div>' +
    '<div class="list-heading">Requirements <span class="list-count">' + reqs.length + '</span></div>';
  const types = ORDER.filter(function(t) { return grouped[t] && grouped[t].length; });
  const otherTypes = Object.keys(grouped).filter(function(t) { return !ORDER.includes(t); });
  for (const type of types.concat(otherTypes)) {
    const items = grouped[type] || [];
    if (!items.length) continue;
    html += '<div class="list-group-label">' + type + ' (' + items.length + ')</div>';
    const shown = items.slice(0, 50);
    for (const r of shown) html += reqItemHtml(r);
    if (items.length > 50) {
      const key = 'req_' + type;
      _expandData[key] = items.slice(50);
      html += '<div class="list-show-more" data-ekey="' + key + '" onclick="expandReqs(this)">Show ' + (items.length - 50) + ' more</div>';
    }
  }
  html += '</div>';
  el.innerHTML = html;
}

function reqItemHtml(r) {
  const applies = r.applies_to && r.applies_to.length ? r.applies_to.join(', ') : '';
  const text = r.text.length > 220 ? r.text.slice(0, 218) + '&#8230;' : r.text;
  return '<div class="list-item" data-id="' + r.doc_id + '" onclick="selectDoc(this.dataset.id)">' +
    '<span class="conn-jbadge" style="background:' + r.doc_color + '">' + r.doc_jurisdiction + '</span>' +
    '<div class="conn-body"><div class="req-text">' + text + '</div>' +
    '<div class="list-meta"><span class="req-doc">' + shortTitle(r.doc_title) + '</span>' +
    (applies ? '<span class="req-applies">' + applies + '</span>' : '') +
    '</div></div></div>';
}

function expandReqs(btn) {
  const key = btn.dataset.ekey;
  const items = _expandData[key] || [];
  let html = '';
  for (const r of items) html += reqItemHtml(r);
  btn.insertAdjacentHTML('beforebegin', html);
  btn.remove();
  delete _expandData[key];
}

function showRelationshipList() {
  clearSelection();
  const REL_LABELS = {
    SUPERSEDES: "Supersedes", IMPLEMENTS: "Implements", CITES: "Cites",
    AMENDS: "Amends", RELATED_TO: "Related to", ANALYZED_BY: "Analyzed by"
  };
  const grouped = {};
  for (const e of data.edges) {
    const t = e.relationship || "RELATED_TO";
    if (!grouped[t]) grouped[t] = [];
    grouped[t].push(e);
  }
  let html = '<div class="list-view">' +
    '<div class="list-back" onclick="showWelcome()">&#8592; Overview</div>' +
    '<div class="list-heading">Relationships <span class="list-count">' + data.edges.length + '</span></div>';
  for (const rel of Object.keys(grouped).sort()) {
    const items = grouped[rel];
    html += '<div class="list-group-label">' + (REL_LABELS[rel] || rel) + ' (' + items.length + ')</div>';
    for (const e of items) {
      const src = nodeById[e.source], tgt = nodeById[e.target];
      if (!src || !tgt) continue;
      html += '<div class="list-item" data-id="' + src.id + '" onclick="selectDoc(this.dataset.id)">' +
        '<div class="conn-body">' +
        '<div class="conn-title">' +
          '<span class="conn-jbadge" style="background:' + src.color + ';margin-right:5px">' + src.jurisdiction + '</span>' +
          shortTitle(src.title) +
          ' <span style="color:var(--muted);font-weight:400;font-size:11px;margin:0 4px">&#8594;</span> ' +
          '<span class="conn-jbadge" style="background:' + tgt.color + ';margin-right:5px">' + tgt.jurisdiction + '</span>' +
          shortTitle(tgt.title) +
        '</div>' +
        (e.notes ? '<div class="conn-notes">' + e.notes + '</div>' : '') +
        '</div></div>';
    }
  }
  html += '</div>';
  document.getElementById("browse-content").innerHTML = html;
}

// ── Concepts ──────────────────────────────────────────────────────────────────
function selectConcept(name) {
  selectedDocId = null;
  selectedConcept = name;
  document.querySelectorAll(".doc-item").forEach(function(el) { el.classList.remove("selected"); });
  const item = document.querySelector('.doc-item[data-concept="' + name + '"]');
  if (item) { item.classList.add("selected"); item.scrollIntoView({ block: "nearest" }); }
  document.getElementById("chat-ctx").textContent = name;
  if (currentTab === "browse") loadConceptView(name);
}

async function loadConceptView(name) {
  const el = document.getElementById("browse-content");
  el.innerHTML = '<div class="concept-view">' +
    '<div class="list-back" onclick="clearSelection()">&#8592; Overview</div>' +
    '<div class="list-heading">' + name + '</div>' +
    '<div style="font-size:13px;color:var(--muted);margin-top:4px">Loading&#8230;</div></div>';
  let docs;
  try {
    const res = await fetch("/ui/concept?name=" + encodeURIComponent(name));
    if (!res.ok) throw new Error("HTTP " + res.status);
    docs = await res.json();
  } catch (e) {
    el.innerHTML = '<p style="color:var(--muted);padding:20px">Error: ' + e.message + '</p>';
    return;
  }
  const totalReqs = docs.reduce(function(s, d) { return s + d.requirements.length; }, 0);
  _expandData = {};
  let html = '<div class="concept-view">' +
    '<div class="list-back" onclick="clearSelection()">&#8592; Overview</div>' +
    '<div class="list-heading">' + name + ' <span class="list-count">' + totalReqs + '</span></div>' +
    '<div style="font-size:12px;color:var(--muted);margin-bottom:24px">' + docs.length + ' document' + (docs.length !== 1 ? 's' : '') + '</div>';
  if (!docs.length) {
    html += '<div style="font-size:13px;color:var(--muted);font-style:italic">No requirements matched this concept.</div>';
  }
  for (const doc of docs) {
    const shown = doc.requirements.slice(0, 4);
    const rest = doc.requirements.slice(4);
    const key = 'con_' + doc.doc_id;
    html += '<div class="concept-doc-block">' +
      '<div class="concept-doc-header" data-id="' + doc.doc_id + '" onclick="selectDoc(this.dataset.id)">' +
        '<span class="conn-jbadge" style="background:' + doc.doc_color + '">' + doc.doc_jurisdiction + '</span>' +
        '<span class="concept-doc-title">' + shortTitle(doc.doc_title) + '</span>' +
        '<span class="concept-doc-meta">' + doc.requirements.length + ' req' + (doc.requirements.length !== 1 ? 's' : '') + '</span>' +
      '</div>';
    for (const r of shown) html += conceptReqHtml(r);
    if (rest.length) {
      _expandData[key] = rest;
      html += '<div class="list-show-more" data-ekey="' + key + '" onclick="expandConceptReqs(this)">+ ' + rest.length + ' more</div>';
    }
    html += '</div>';
  }
  html += '</div>';
  el.innerHTML = html;
}

function conceptReqHtml(r) {
  const obl = r.obligation_type || 'MUST';
  const text = r.text.length > 260 ? r.text.slice(0, 258) + '&#8230;' : r.text;
  return '<div class="concept-req">' +
    '<span class="obl-badge obl-' + obl + '">' + obl + '</span>' +
    '<span class="concept-req-text">' + text + '</span>' +
    '</div>';
}

function expandConceptReqs(btn) {
  const key = btn.dataset.ekey;
  const items = _expandData[key] || [];
  let html = '';
  for (const r of items) html += conceptReqHtml(r);
  btn.insertAdjacentHTML('beforebegin', html);
  btn.remove();
  delete _expandData[key];
}

// ── Browse ────────────────────────────────────────────────────────────────────
function showWelcome() {
  const totalReqs = data.nodes.reduce(function(s, n) { return s + n.requirement_count; }, 0);
  document.getElementById("browse-content").innerHTML =
    '<div class="welcome">' +
    '<h2>AI Regulation Knowledge Graph</h2>' +
    '<p>Browse regulations and standards, explore connections between documents, and ask questions about compliance requirements.</p>' +
    '<div class="stat-row">' +
    '<div class="stat-box" onclick="showDocumentList()"><div class="sv">' + data.nodes.length + '</div><div class="sl">Documents</div></div>' +
    '<div class="stat-box" onclick="showRequirementList()"><div class="sv">' + totalReqs + '</div><div class="sl">Requirements</div></div>' +
    '<div class="stat-box" onclick="showRelationshipList()"><div class="sv">' + data.edges.length + '</div><div class="sl">Relationships</div></div>' +
    '</div>' +
    '<p>Select a document from the left panel to view details, source links, and connections.</p>' +
    '</div>';
}

function statusChipClass(status) {
  var map = { enacted: "chip-enacted", proposed: "chip-proposed", superseded: "chip-superseded", draft: "chip-draft" };
  return map[status] || "chip-meta";
}

function renderBrowse(id) {
  const n = nodeById[id];
  if (!n) return;

  const out = edgesFrom[id] || [];
  const inc = edgesTo[id] || [];
  const outGroups = {}, inGroups = {};

  for (const e of out) {
    const t = nodeById[e.target];
    if (!t) continue;
    if (!outGroups[e.relationship]) outGroups[e.relationship] = [];
    outGroups[e.relationship].push({ node: t, notes: e.notes });
  }
  for (const e of inc) {
    const s = nodeById[e.source];
    if (!s) continue;
    if (!inGroups[e.relationship]) inGroups[e.relationship] = [];
    inGroups[e.relationship].push({ node: s, notes: e.notes });
  }

  const totalConns = out.length + inc.length;
  let connHtml = "";

  if (totalConns === 0) {
    connHtml = '<div class="no-conn">No connections recorded for this document.</div>';
  } else {
    for (const rel in outGroups) {
      connHtml += '<div class="conn-group"><div class="rel-label">' + (REL_OUT_LABELS[rel] || rel) + '</div>';
      for (const item of outGroups[rel]) connHtml += connItemHtml(item.node, item.notes);
      connHtml += '</div>';
    }
    for (const rel in inGroups) {
      connHtml += '<div class="conn-group"><div class="rel-label">' + (REL_IN_LABELS[rel] || rel) + '</div>';
      for (const item of inGroups[rel]) connHtml += connItemHtml(item.node, item.notes);
      connHtml += '</div>';
    }
  }

  document.getElementById("browse-content").innerHTML =
    '<div class="doc-detail">' +
    '<div class="dd-title">' + n.title + '</div>' +
    '<div class="chips">' +
    '<span class="chip chip-jur" style="background:' + n.color + '">' + n.jurisdiction + '</span>' +
    '<span class="chip chip-type">' + n.doc_type + '</span>' +
    '<span class="chip ' + statusChipClass(n.status) + '">' + n.status + '</span>' +
    (n.effective_date ? '<span class="chip chip-meta">' + n.effective_date + '</span>' : '') +
    '<span class="chip chip-meta">' + n.requirement_count + ' requirements</span>' +
    '</div>' +
    (n.summary ? '<div class="dd-summary">' + n.summary + '</div>' : '') +
    (n.url ? '<a class="source-link" href="' + n.url + '" target="_blank">View source →</a>' : '') +
    '<div class="section-title">Connections (' + totalConns + ')</div>' +
    connHtml +
    '</div>';
}

function connItemHtml(cn, notes) {
  const st = shortTitle(cn.title);
  const linkHtml = cn.url ? '<a class="conn-link" href="' + cn.url + '" target="_blank" onclick="event.stopPropagation()">Source →</a>' : '';
  const notesHtml = notes ? '<div class="conn-notes">' + notes + '</div>' : '';
  return '<div class="conn-item" data-id="' + cn.id + '" onclick="selectDoc(this.dataset.id)">' +
    '<span class="conn-jbadge" style="background:' + cn.color + '">' + cn.jurisdiction + '</span>' +
    '<div class="conn-body"><div class="conn-title">' + st + '</div>' + linkHtml + notesHtml + '</div>' +
    '</div>';
}

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(tab) {
  currentTab = tab;
  document.getElementById("tab-browse").classList.toggle("active", tab === "browse");
  document.getElementById("tab-graph").classList.toggle("active", tab === "graph");
  document.getElementById("browse-panel").classList.toggle("active", tab === "browse");
  document.getElementById("graph-panel").classList.toggle("active", tab === "graph");
  if (tab === "browse") {
    if (selectedConcept) loadConceptView(selectedConcept);
    else if (selectedDocId) renderBrowse(selectedDocId);
    else showWelcome();
  }
  if (tab === "graph" && !graphInited) initGraph();
}

// ── Force simulation ──────────────────────────────────────────────────────────
const SIM = { REPULSION: 320, ATTRACTION: 0.022, GRAVITY: 0.038, DAMPING: 0.84 };

function simStep() {
  const nids = Object.keys(simPos);
  for (const n of nids) {
    simPos[n].vx -= SIM.GRAVITY * simPos[n].x;
    simPos[n].vy -= SIM.GRAVITY * simPos[n].y;
  }
  for (let a = 0; a < nids.length; a++) {
    for (let b = a + 1; b < nids.length; b++) {
      const pa = simPos[nids[a]], pb = simPos[nids[b]];
      const dx = pa.x - pb.x, dy = pa.y - pb.y;
      const dist = Math.sqrt(dx * dx + dy * dy) + 0.1;
      const f = SIM.REPULSION * (simSizes[nids[a]] + simSizes[nids[b]]) / 14 / (dist * dist);
      const fx = f * dx / dist, fy = f * dy / dist;
      pa.vx += fx; pa.vy += fy; pb.vx -= fx; pb.vy -= fy;
    }
  }
  graph.forEachEdge(function(e, _a, src, tgt) {
    if (src === tgt || !simPos[src] || !simPos[tgt]) return;
    const pa = simPos[src], pb = simPos[tgt];
    const dx = pb.x - pa.x, dy = pb.y - pa.y;
    pa.vx += SIM.ATTRACTION * dx; pa.vy += SIM.ATTRACTION * dy;
    pb.vx -= SIM.ATTRACTION * dx; pb.vy -= SIM.ATTRACTION * dy;
  });
  let maxV = 0;
  for (const n of nids) {
    const p = simPos[n];
    p.x += p.vx; p.y += p.vy; p.vx *= SIM.DAMPING; p.vy *= SIM.DAMPING;
    if (Math.abs(p.vx) > maxV) maxV = Math.abs(p.vx);
    if (Math.abs(p.vy) > maxV) maxV = Math.abs(p.vy);
    graph.setNodeAttribute(n, "x", p.x);
    graph.setNodeAttribute(n, "y", p.y);
  }
  return maxV;
}

function startSim() {
  if (simHandle) cancelAnimationFrame(simHandle);
  simPos = {}; simSizes = {};
  graph.forEachNode(function(n) {
    simPos[n] = { x: graph.getNodeAttribute(n, "x"), y: graph.getNodeAttribute(n, "y"), vx: 0, vy: 0 };
    simSizes[n] = graph.getNodeAttribute(n, "size");
  });
  let frames = 0;
  function tick() {
    const STEPS = 3; let maxV = 0;
    for (let i = 0; i < STEPS; i++) maxV = Math.max(maxV, simStep());
    renderer.refresh(); frames++;
    if (maxV > 0.008 && frames < 300) simHandle = requestAnimationFrame(tick);
    else simHandle = null;
  }
  simHandle = requestAnimationFrame(tick);
}

function seedPositions() {
  const count = graph.order; let i = 0;
  graph.forEachNode(function(n) {
    const angle = (2 * Math.PI * i) / count + (Math.random() - 0.5) * 0.5;
    const r = 8 + Math.random() * 12;
    graph.setNodeAttribute(n, "x", r * Math.cos(angle));
    graph.setNodeAttribute(n, "y", r * Math.sin(angle));
    i++;
  });
}

// ── Graph (lazy init) ─────────────────────────────────────────────────────────
function initGraph() {
  graphInited = true;

  graph = new graphology.Graph({ multi: true, type: "directed" });
  for (const n of data.nodes) {
    const label = ACRONYMS[n.title] || (n.title.length > 26 ? n.title.slice(0, 24) + '…' : n.title);
    graph.addNode(String(n.id), {
      x: 0, y: 0, size: 1, label,
      color: n.status === "superseded" ? "rgba(255,255,255,0.2)" : (n.color || "#6b7280"),
      _id: n.id
    });
  }

  const seen = new Set();
  for (const e of data.edges) {
    const key = e.source + "-" + e.target + "-" + e.relationship;
    if (seen.has(key)) continue; seen.add(key);
    try {
      graph.addDirectedEdge(String(e.source), String(e.target), {
        color: REL_COLORS_GRAPH[e.relationship] || "#475569",
        size: ["SUPERSEDES","IMPLEMENTS","CITES","AMENDS"].includes(e.relationship) ? 2 : 1,
        type: "arrow",
        _rel: e.relationship
      });
    } catch (_) {}
  }

  graph.forEachNode(function(n) {
    const deg = graph.degree(n);
    graph.setNodeAttribute(n, "size", deg === 0 ? 3 : Math.max(7, Math.min(24, 5 + deg * 2.2)));
  });

  seedPositions();

  renderer = new Sigma(graph, document.getElementById("sigma-container"), {
    defaultEdgeType: "arrow",
    renderEdgeLabels: false,
    labelSize: 12,
    labelWeight: "500",
    labelColor: { color: "rgba(255,255,255,0.85)" },
    labelRenderedSizeThreshold: 7,
    defaultEdgeColor: "#334155",
    minCameraRatio: 0.03,
    maxCameraRatio: 10,
    nodeReducer: function(node, d) {
      if (!hoveredNode) return d;
      if (node === hoveredNode) return Object.assign({}, d, { highlighted: true, zIndex: 1 });
      if (neighborSet && neighborSet.has(node)) return Object.assign({}, d, { zIndex: 1 });
      return Object.assign({}, d, { color: "rgba(255,255,255,0.06)", label: "" });
    },
    edgeReducer: function(edge, d) {
      if (!hoveredNode) return d;
      const src = graph.source(edge), tgt = graph.target(edge);
      if (src === hoveredNode || tgt === hoveredNode) return Object.assign({}, d, { zIndex: 1 });
      return Object.assign({}, d, { color: "rgba(255,255,255,0.03)" });
    }
  });

  renderer.on("enterNode", function(e) {
    hoveredNode = e.node;
    neighborSet = new Set(graph.neighbors(e.node));
    document.body.style.cursor = "pointer";
    renderer.refresh();
  });
  renderer.on("leaveNode", function() {
    hoveredNode = null; neighborSet = null;
    document.body.style.cursor = "default";
    renderer.refresh();
  });
  renderer.on("clickNode", function(e) {
    const id = graph.getNodeAttribute(e.node, "_id");
    window.location.href = '/ui/doc/' + id;
  });
  renderer.on("clickStage", function() { clearSelection(); });

  document.getElementById("graph-loading").style.display = "none";
  startSim();
}

function resetCamera() { if (renderer) renderer.getCamera().animatedReset(); }
function relayout() { if (!graph || !renderer) return; seedPositions(); startSim(); renderer.getCamera().animatedReset(); }

// ── Markdown renderer ─────────────────────────────────────────────────────────
function renderMarkdown(raw) {
  const esc = raw
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  const lines = esc.split('\\n');
  let out = '', inCode = false, codeBuf = '', codeLang = '', inList = false;

  function flushList() {
    if (inList) { out += '</ul>'; inList = false; }
  }

  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];

    // fenced code block toggle
    if (l.startsWith('```')) {
      if (!inCode) {
        flushList();
        inCode = true; codeLang = l.slice(3).trim(); codeBuf = '';
      } else {
        out += '<pre><code>' + codeBuf + '</code></pre>';
        inCode = false;
      }
      continue;
    }
    if (inCode) { codeBuf += (codeBuf ? '\\n' : '') + l; continue; }

    // horizontal rule
    if (/^---+$/.test(l)) { flushList(); out += '<hr>'; continue; }

    // headings
    const h3 = l.match(/^### (.+)/); if (h3) { flushList(); out += '<h3>' + inlineRender(h3[1]) + '</h3>'; continue; }
    const h2 = l.match(/^## (.+)/);  if (h2) { flushList(); out += '<h2>' + inlineRender(h2[1]) + '</h2>'; continue; }
    const h1 = l.match(/^# (.+)/);   if (h1) { flushList(); out += '<h1>' + inlineRender(h1[1]) + '</h1>'; continue; }

    // bullet list items
    const li = l.match(/^[\-\*\+] (.+)/) || l.match(/^\d+\. (.+)/);
    if (li) {
      if (!inList) { out += '<ul>'; inList = true; }
      out += '<li>' + inlineRender(li[1]) + '</li>';
      continue;
    }

    // blank line → paragraph break
    if (l.trim() === '') { flushList(); out += '<br>'; continue; }

    flushList();
    out += '<p>' + inlineRender(l) + '</p>';
  }
  flushList();
  return out;
}

function inlineRender(s) {
  return s
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/__([^_]+)__/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    .replace(/_([^_]+)_/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');
}

// ── Chat ──────────────────────────────────────────────────────────────────────
function clearChat() {
  document.getElementById("chat-messages").innerHTML = "";
}

function toggleChat() {
  const chat = document.getElementById("chat");
  const btn = document.getElementById("chat-toggle");
  const collapsed = chat.classList.toggle("collapsed");
  btn.textContent = collapsed ? "▼" : "▲";
}

async function sendChat() {
  if (chatStreaming) return;
  const input = document.getElementById("chat-input");
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  input.style.height = "38px";

  const msgs = document.getElementById("chat-messages");
  const userEl = document.createElement("div");
  userEl.className = "cmsg user";
  userEl.textContent = question;
  msgs.appendChild(userEl);

  const assistEl = document.createElement("div");
  assistEl.className = "cmsg assistant streaming";
  msgs.appendChild(assistEl);
  msgs.scrollTop = msgs.scrollHeight;

  chatStreaming = true;
  document.getElementById("chat-send").disabled = true;
  let streamedText = "";

  try {
    const res = await fetch("/ui/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question, doc_id: selectedDocId })
    });
    if (!res.ok) { assistEl.textContent = "Error " + res.status; return; }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "", text = "";
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buf += decoder.decode(chunk.value, { stream: true });
      const lines = buf.split("\\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        try {
          const obj = JSON.parse(line.slice(6));
          const delta = obj.choices && obj.choices[0] && obj.choices[0].delta && obj.choices[0].delta.content;
          if (delta) { text += delta; streamedText = text; assistEl.textContent = text; msgs.scrollTop = msgs.scrollHeight; }
        } catch (_) {}
      }
    }
  } catch (err) {
    assistEl.textContent = "Error: " + err.message;
  } finally {
    assistEl.classList.remove("streaming");
    if (streamedText) {
      assistEl.innerHTML = renderMarkdown(streamedText);
    }
    chatStreaming = false;
    document.getElementById("chat-send").disabled = false;
    msgs.scrollTop = msgs.scrollHeight;
  }
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", function() {
  const chatInput = document.getElementById("chat-input");
  chatInput.addEventListener("keydown", function(e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
  });
  chatInput.addEventListener("input", function() {
    this.style.height = "38px";
    this.style.height = Math.min(this.scrollHeight, 120) + "px";
  });
  init();
});
</script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    if not _authed(request):
        return RedirectResponse("/ui/login", status_code=302)
    return _UI_HTML


ui_app = FastAPI()
ui_app.include_router(router)
