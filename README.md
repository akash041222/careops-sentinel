# CareOps Sentinel - AI-Powered Healthcare Operations Assistant

Grounded answers with citations, guided workflows that ask for missing information, safe routing to human teams,
role-based consoles, and a tamper-evident audit trail. **Synthetic data only.**

## Run it (full-stack: API + website on ONE URL)
The built web app is included in `frontend/dist`, so Node is **not** needed to run it.
```bash

# Python 3.10+:
pip install -r requirements.txt && uvicorn app.main:app --port 8000
# or Docker:
echo "CAREOPS_SECRET_KEY=$(python3 -c 'import secrets;print(secrets.token_hex(32))')" > .env && docker compose up --build
```
Developing the UI? `cd frontend && npm install && npm run dev` (hot reload on :5173, talks to the API on :8000).
After UI changes run `npm run build` so the backend serves the new version. API docs: http://localhost:8000/docs

Demo logins (password `CareOps@123`): `EMP-1001` employee · `AG-2001` HR agent · `AG-2003` facilities agent ·
`SUP-3001` support agent · `OPS-2001` operations manager · `ADM-4001` admin. The login page has one-click buttons.

Tests: `python -m pytest tests -q` (79 backend tests) and `cd frontend && npm test` (chat-box tests).

## What was wrong, and what changed
| Problem in v1 | Fix |
|---|---|
| "hi" was treated as an *Unclear Request* workflow | Small-talk layer (greetings, thanks, help, identity, out-of-scope) answers naturally and offers starters; "hi, I need X" still processes X |
| Typed text stayed in the box and Enter did nothing | Real chat UI: Enter sends, Shift+Enter = newline, input clears instantly, optimistic bubble, typing indicator, retry on error, history |
| Keyword matching used substrings ("AC" matched "b**ac**k", so *"my back hurts"* became a Maintenance request) | Whole-word matching + hybrid classifier (nearest of 565 labelled examples + keywords). 94% held-out accuracy; confidence is calibrated (>=0.70 is ~99% correct) |
| Medical questions were not blocked reliably | Safety screen runs **first** on every message: clinical advice, emergencies, confidential data, credential requests, bypass/prompt-injection, irreversible actions |
| Login was cosmetic: tokens never checked, roles trusted from the request body | Signed expiring tokens, server-side RBAC on every endpoint, team-scoped agents, employees see only their own requests, brute-force lockout, rate limits |
| In-memory state lost on restart; fake audit | SQLite persistence; hash-chained audit log (`/api/audit/verify` detects tampering); CSV export |
| Missing fields were inferred or mis-named | Fields come from Workflows + Routing_Rules, validated per type (dates, times, IDs, quantities); identity comes from the login, so nobody can file as someone else |
| Duplicate/conflicting routes, dead nav pages, minified single-file UI | Clean module layout, every nav item works, 6 pages, accessible, responsive |

## RAG + LLM (optional, off by default)
Set `ANTHROPIC_API_KEY` (and optionally `CAREOPS_LLM_MODEL`) and restart. Without a key the app runs in extractive mode.

| Stage | What happens |
|---|---|
| 1. Safety screen | Runs **before** any model call. Medical, confidential, bypass and irreversible requests never reach the LLM |
| 2. Retrieval | Hybrid **BM25 + TF-IDF** fused with Reciprocal Rank Fusion over `approved` articles only (top-1 87%, top-3 92% on the workbook examples) |
| 3. Semantic routing | For vague questions the LLM sees only article *titles* and returns IDs; IDs are validated against the catalogue |
| 4. Grounded generation | LLM gets redacted text + retrieved articles, treated as data, and must return JSON `{answerable, answer, citations, confidence}` |
| 5. Validation | Citations must be real retrieved IDs; every number, URL and e-mail must appear in the sources; no new clinical content. Any failure -> extractive answer |
| 6. Intent tie-break | When local confidence is 0.35-0.70 the LLM may choose, but only among the local top-3 candidates and only if >= 0.80 sure |
| 7. Resilience | 20 s timeout, answer cache, circuit breaker (3 failures -> 60 s pause), full fallback, `LLM_ANSWER` / `LLM_FALLBACK` audit entries |

The UI marks LLM answers "AI-generated - validated"; the admin page shows model, calls, failures and token usage.
The LLM never approves, rejects or submits anything - those stay human-only.

## Responsible-AI design
* **Answers come only from `approved` articles** and always show source, ID, effective date and match score.
* **Uncertainty is shown**: confidence meter; 0.50-0.70 asks "is this what you meant?"; below that it declines to guess and offers a human.
* **Never medical advice.** Emergencies point to local emergency services (no invented numbers, per KA-046).
* **Passwords / OTPs / card numbers are masked before storage or logging**, and the user is told.
* **Human in the loop**: the assistant can only *submit*; take / resolve / reject / reassign are human-only API actions and the AI "agent assist" is advisory.
* **Explainability**: every reply has a "Why did the assistant do this?" trace (safety result, intent evidence, sources, routing rule).
* Sentiment and urgency detection raise priority; SLA timers and breach flags in the queue.

## Architecture
```
frontend (React/Vite)  ->  FastAPI  ->  engine.py (conversation state machine)
                                          |- nlu.py       safety screen, small-talk, tone, hybrid intent classifier
                                          |- retriever.py approved-knowledge TF-IDF search
                                          |- fields.py    extraction + validation
                                          |- ops.py       requests, RBAC scope, SLA, transitions, assist, metrics
                                          |- db.py        SQLite + hash-chained audit
                                          `- data_store.py  Excel workbook catalogue
```
Swap-in points for production: PostgreSQL in `db.py`, an LLM or embeddings in `retriever.py`/`engine.py` (keep the safety
screen and citation rules in front), SSO instead of `security.UserDirectory`.

## Configuration
See `.env.example`. **Set `CAREOPS_SECRET_KEY` outside demos.**

## Key endpoints
`POST /api/auth/login` · `POST /api/chat` · `GET /api/conversations[/{id}]` · `GET /api/catalog` · `GET/POST /api/requests...`
· `GET /api/requests/{id}/assist` · `GET /api/dashboard/summary` · `GET /api/audit[/verify|/export]` · `/api/admin/*` · `/health` · `/ready`

## Human escalation alarm
Support staff (support agent, operations manager, and admin) can enable the **Alert sound** control in the sidebar. While the app is open, it checks the pending-human queue every 8 seconds. Newly detected escalations show a prominent popup with request ID, issue type, priority, and requester, and a repeating browser-generated beep while alerts remain unacknowledged. **Acknowledge & Open Request** dismisses that alert and opens the request detail in the queue; it does not resolve the request. Dismiss only acknowledges the notification. Browser audio must be enabled by a user click, and the staff app must remain open. This is a prototype polling alert, not a guaranteed out-of-browser emergency notification service.
