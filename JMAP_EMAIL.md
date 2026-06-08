# Sending email from the droplet (use JMAP, not SMTP)

## The problem you hit

`monitor.py` sends mail with `smtplib.SMTP_SSL("smtp.fastmail.com", 465)`. That
works from a laptop but **hangs and times out on the DigitalOcean droplet**:
DigitalOcean blocks outbound ports **25 / 465 / 587** on new droplets to stop
spam. There is no firewall rule you can add on your side to fix it — the block
is upstream at DO. (Hetzner, AWS, and most cloud VMs do the same.)

## The fix

Send through **Fastmail's JMAP API over HTTPS / port 443**, which is never
blocked. Same Fastmail account, same domain, same DKIM/SPF/DMARC — sent mail
even lands in your Fastmail Sent folder. This is exactly how the ResearchAgent
project (`~/Desktop/Projects/ResearchAgent`, `src/email_io/outbound.py`) sends
all of its mail from the same droplet. `jmap_send.py` here is a distilled,
single-function port of that module using `httpx` (already a dependency).

## Setup — 3 steps

### 1. Get a Fastmail API token (NOT the app password)

Fastmail Settings → **Privacy & Security → API tokens** → new token with
**read+write mail** scope. This is a different credential from the IMAP/SMTP
app password you're using now — the app password will return 401 against JMAP.

### 2. Update your env

Replace the SMTP block in `.env` / `.env.example` with:

```
FASTMAIL_API_TOKEN=fmu1-...          # the API token from step 1
COMPLIANCE_FROM_EMAIL=compliance@npedwards.com
COMPLIANCE_TO_EMAIL=nick.edwards1988@gmail.com
```

`COMPLIANCE_FROM_EMAIL` must be a **verified identity** on that Fastmail account
(Settings → Sending identities), or the submit step will fail with a clear
"no identity matches" error.

### 3. Swap the sender in `monitor.py`

The current `_send_email` (around line 375) builds an `EmailMessage` and calls
`smtplib`. Replace its body with a call to `jmap_send.send_email`:

```python
from jmap_send import send_email, JmapError

def _send_email(subject: str, body: str) -> None:
    try:
        send_email(subject, body)   # reads FASTMAIL_API_TOKEN / *_FROM_EMAIL / *_TO_EMAIL
    except JmapError as exc:
        logger.warning("email send failed: %s", exc)
```

You can delete the `smtplib` import and the `SMTP_*` env vars/constants once
nothing else references them.

## Verify

From the droplet (not your laptop — the whole point is testing where SMTP is
blocked):

```bash
cd /path/to/compliance && python jmap_send.py
```

That runs the `__main__` smoke test and sends one email using the env vars.
On success it prints `sent OK` and the mail appears in your inbox **and** the
Fastmail Sent folder. On failure `JmapError` carries a specific reason (401
token, missing identity, network, per-method `notCreated`).

## How it works (for the curious)

One `GET /jmap/session` discovers your account's `apiUrl`, mail `accountId`,
Drafts/Sent mailbox IDs, and submission identity. Then a single POST runs two
chained method calls: `Email/set` creates a draft, and `EmailSubmission/set`
sends it and (on success) moves it Drafts→Sent and marks it `$seen`. JMAP
returns HTTP 200 even when an individual call fails, so `jmap_send.py` inspects
the per-method responses and raises on any `error` / `notCreated`.

Reference: <https://jmap.io/spec-mail.html> (Email/set, EmailSubmission/set).
