"""Send email via Fastmail JMAP over HTTPS (port 443).

WHY THIS EXISTS — the short version:
    DigitalOcean (and most cloud VMs) block outbound SMTP ports 25/465/587 on
    new droplets. `smtplib.SMTP_SSL("smtp.fastmail.com", 465)` will hang and
    time out on the server even though it works fine from your laptop. JMAP
    runs over plain HTTPS/443, which is never blocked, and Fastmail already
    handles DKIM/SPF/DMARC for your domain — so sent mail lands in your
    Fastmail Sent folder with no extra DNS records.

This module is a distilled, dependency-light port of ResearchAgent's
`src/email_io/outbound.py`. It uses `httpx` (already a compliance dependency)
instead of `requests`, and collapses the send to a single `send_email(...)`
call. Plain-text only; add an HTML part by extending `_build_methodcalls`.

AUTH — read this carefully, it's the usual tripwire:
    You need a Fastmail *API token*, NOT the IMAP/SMTP app password.
    Generate it at: Fastmail Settings -> Privacy & Security -> API tokens.
    JMAP uses Bearer-token auth; the app password will 401 here.

USAGE:
    from jmap_send import send_email, JmapError

    try:
        send_email(
            subject="Compliance digest 2026-06-08",
            body="3 new requirements ingested ...",
        )
    except JmapError as exc:
        logger.error("email send failed: %s", exc)

Environment variables (read lazily, so importing this module is side-effect free):
    FASTMAIL_API_TOKEN   - the Bearer API token (required)
    COMPLIANCE_FROM_EMAIL - the From: address; must be a verified Fastmail
                            identity on the token's account (required)
    COMPLIANCE_TO_EMAIL  - default recipient (optional if you pass `to=`)
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

import httpx

JMAP_SESSION_URL = "https://api.fastmail.com/jmap/session"
JMAP_TIMEOUT_SECONDS = 30
JMAP_USING = (
    "urn:ietf:params:jmap:core",
    "urn:ietf:params:jmap:mail",
    "urn:ietf:params:jmap:submission",
)


class JmapError(RuntimeError):
    """Any failure discovering the session, building, or submitting the email."""


@dataclass(frozen=True)
class _Session:
    api_url: str
    account_id: str
    drafts_id: str
    sent_id: str
    identity_id: str


# --------------------------------------------------------------- public API


def send_email(
    subject: str,
    body: str,
    *,
    to: str | None = None,
    from_addr: str | None = None,
    token: str | None = None,
    client: httpx.Client | None = None,
) -> None:
    """Send a plain-text email through Fastmail JMAP.

    Raises JmapError on missing config, auth failure, network error, or a
    per-method JMAP failure (JMAP returns HTTP 200 even when an individual
    method call fails, so we inspect the method responses explicitly).
    """
    token = token or os.getenv("FASTMAIL_API_TOKEN", "")
    from_addr = from_addr or os.getenv("COMPLIANCE_FROM_EMAIL", "")
    to = to or os.getenv("COMPLIANCE_TO_EMAIL", "")

    if not token:
        raise JmapError("FASTMAIL_API_TOKEN not set (generate at Fastmail -> API tokens)")
    if not from_addr:
        raise JmapError("COMPLIANCE_FROM_EMAIL not set")
    if not to:
        raise JmapError("no recipient: pass to= or set COMPLIANCE_TO_EMAIL")

    own_client = client is None
    cli = client or httpx.Client(timeout=JMAP_TIMEOUT_SECONDS)
    try:
        session = _discover_session(cli, token=token, identity_email=from_addr)
        payload = _build_methodcalls(
            session, from_addr=from_addr, to=to, subject=subject, body=body
        )
        response = _post(cli, session.api_url, payload, token=token)
        _raise_for_method_errors(response)
    finally:
        if own_client:
            cli.close()


# --------------------------------------------------------------- internals


def _discover_session(cli: httpx.Client, *, token: str, identity_email: str) -> _Session:
    """One GET against /jmap/session, then POSTs to the returned apiUrl to look
    up the Drafts/Sent mailbox IDs and the submission Identity ID. Fastmail
    refuses EmailSubmission/set without a valid identityId."""
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = cli.get(JMAP_SESSION_URL, headers=headers)
    except httpx.HTTPError as exc:
        raise JmapError(f"session discovery network error: {type(exc).__name__}: {exc}") from exc

    if resp.status_code == 401:
        raise JmapError("session: 401 Unauthorized — FASTMAIL_API_TOKEN invalid or revoked")
    if resp.status_code >= 400:
        raise JmapError(f"session: HTTP {resp.status_code}: {resp.text[:300]}")

    try:
        body = resp.json()
    except ValueError as exc:
        raise JmapError(f"session: unparseable JSON: {exc}") from exc

    api_url = body.get("apiUrl")
    accounts = body.get("primaryAccounts") or {}
    account_id = accounts.get("urn:ietf:params:jmap:mail")
    if not api_url or not account_id:
        raise JmapError(
            f"session: missing apiUrl or mail account (apiUrl={api_url!r}, accounts={list(accounts)})"
        )

    drafts_id, sent_id = _role_mailboxes(cli, api_url, account_id, token=token)
    identity_id = _identity(cli, api_url, account_id, token=token, match_email=identity_email)
    return _Session(
        api_url=api_url,
        account_id=account_id,
        drafts_id=drafts_id,
        sent_id=sent_id,
        identity_id=identity_id,
    )


def _role_mailboxes(
    cli: httpx.Client, api_url: str, account_id: str, *, token: str
) -> tuple[str, str]:
    payload = {
        "using": list(JMAP_USING),
        "methodCalls": [
            ["Mailbox/get", {"accountId": account_id, "ids": None,
                             "properties": ["id", "name", "role"]}, "m1"],
        ],
    }
    resp = _post(cli, api_url, payload, token=token)
    responses = resp.get("methodResponses") or []
    if not responses:
        raise JmapError(f"Mailbox/get returned no methodResponses: {resp}")
    name, mbody, _tag = responses[0]
    if name == "error":
        raise JmapError(f"Mailbox/get error: {mbody}")
    drafts_id = sent_id = ""
    for box in mbody.get("list") or []:
        role = (box.get("role") or "").lower()
        if role == "drafts":
            drafts_id = box["id"]
        elif role == "sent":
            sent_id = box["id"]
    if not drafts_id or not sent_id:
        raise JmapError(f"could not find Drafts/Sent mailbox (drafts={drafts_id!r}, sent={sent_id!r})")
    return drafts_id, sent_id


def _identity(
    cli: httpx.Client, api_url: str, account_id: str, *, token: str, match_email: str
) -> str:
    """Find the submission Identity whose email matches `match_email`
    (case-insensitive). Fall back to the sole identity if there's exactly one;
    otherwise raise so you can see which identities exist on the account."""
    payload = {
        "using": list(JMAP_USING),
        "methodCalls": [["Identity/get", {"accountId": account_id, "ids": None}, "i1"]],
    }
    resp = _post(cli, api_url, payload, token=token)
    responses = resp.get("methodResponses") or []
    if not responses:
        raise JmapError(f"Identity/get returned no methodResponses: {resp}")
    name, ibody, _tag = responses[0]
    if name == "error":
        raise JmapError(f"Identity/get error: {ibody}")
    identities = ibody.get("list") or []
    if not identities:
        raise JmapError("no submission identities exist on this account")

    target = _address_only(match_email).lower()
    for ident in identities:
        if (ident.get("email") or "").lower() == target:
            return ident["id"]
    if len(identities) == 1:
        return identities[0]["id"]
    available = ", ".join((i.get("email") or "?") for i in identities)
    raise JmapError(f"no identity matches {match_email!r} (available: {available})")


def _build_methodcalls(
    session: _Session, *, from_addr: str, to: str, subject: str, body: str
) -> dict:
    """Email/set creates a draft; EmailSubmission/set sends it and, on success,
    moves it Drafts -> Sent and marks it $seen — all in one round trip."""
    domain = from_addr.rsplit("@", 1)[-1] if "@" in from_addr else "localhost"
    message_id = f"compliance-{uuid.uuid4().hex[:12]}@{domain}"
    email_create = {
        "mailboxIds": {session.drafts_id: True},
        "keywords": {"$draft": True},
        "from": [{"email": _address_only(from_addr)}],
        "to": [{"email": _address_only(to)}],
        "subject": subject,
        "messageId": [message_id],
        "textBody": [{"partId": "text", "type": "text/plain"}],
        "bodyValues": {"text": {"value": body, "charset": "utf-8"}},
    }
    return {
        "using": list(JMAP_USING),
        "methodCalls": [
            ["Email/set", {"accountId": session.account_id,
                           "create": {"draft": email_create}}, "c1"],
            ["EmailSubmission/set", {
                "accountId": session.account_id,
                "create": {"sub": {
                    "identityId": session.identity_id,
                    "emailId": "#draft",
                    "envelope": {
                        "mailFrom": {"email": _address_only(from_addr)},
                        "rcptTo": [{"email": _address_only(to)}],
                    },
                }},
                "onSuccessUpdateEmail": {"#sub": {
                    f"mailboxIds/{session.drafts_id}": None,
                    f"mailboxIds/{session.sent_id}": True,
                    "keywords/$draft": None,
                    "keywords/$seen": True,
                }},
            }, "c2"],
        ],
    }


def _post(cli: httpx.Client, api_url: str, payload: dict, *, token: str) -> dict:
    try:
        resp = cli.post(
            api_url,
            json=payload,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
    except httpx.HTTPError as exc:
        raise JmapError(f"network error: {type(exc).__name__}: {exc}") from exc
    if resp.status_code == 401:
        raise JmapError("401 Unauthorized — FASTMAIL_API_TOKEN invalid or revoked")
    if resp.status_code >= 400:
        raise JmapError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    try:
        return resp.json()
    except ValueError as exc:
        raise JmapError(f"unparseable JSON response: {exc}") from exc


def _raise_for_method_errors(response: dict) -> None:
    """JMAP returns HTTP 200 even when an individual method call fails. Walk the
    methodResponses and raise on any error entry or notCreated payload."""
    for entry in response.get("methodResponses") or []:
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        name, mbody = entry[0], entry[1]
        if name == "error":
            raise JmapError(f"method-level error: {mbody}")
        not_created = mbody.get("notCreated") if isinstance(mbody, dict) else None
        if not_created:
            raise JmapError(f"{name} notCreated: {not_created}")


def _address_only(addr: str) -> str:
    """Strip a `Display Name <addr@x>` wrapper down to `addr@x`."""
    if "<" in addr and ">" in addr:
        return addr.split("<", 1)[1].rsplit(">", 1)[0].strip()
    return addr.strip()


if __name__ == "__main__":
    # Smoke test: `python jmap_send.py` sends one email using the env vars.
    import logging

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # env vars must already be exported

    logging.basicConfig(level=logging.INFO)
    send_email(
        subject="jmap_send.py smoke test",
        body="If you got this, JMAP-over-443 works from this host.",
    )
    print("sent OK")
