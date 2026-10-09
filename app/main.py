"""CareOps Sentinel API.

Run:  uvicorn app.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import csv
import io
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from . import config, security
from .classifier_boot import build_components
from .db import Database
from .seed import seed_demo
from .text_utils import clean_input

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("careops")

C = build_components()                      # store, db, users, classifier, retriever, ops, engine
store, db, users, ops, engine = C["store"], C["db"], C["users"], C["ops"], C["engine"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.SECRET_KEY_IS_DEFAULT:
        log.warning("CAREOPS_SECRET_KEY not set - using a random per-process key (sessions reset on restart).")
    if config.SEED_DEMO_DATA:
        n = seed_demo(store, db, ops, users)
        if n:
            log.info("Seeded %s synthetic demo requests", n)
    yield


app = FastAPI(title="CareOps Sentinel API", version="1.0.0", lifespan=lifespan,
              description="AI-powered healthcare operations assistant: grounded answers, guided workflows, human-in-the-loop routing.")
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_credentials=False,
                   allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"], allow_headers=["Authorization", "Content-Type"])


@app.middleware("http")
async def hardening(request: Request, call_next):
    rid = uuid.uuid4().hex[:12]
    start = time.time()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("Unhandled error [%s] %s %s", rid, request.method, request.url.path)
        response = JSONResponse({"detail": "Internal server error", "request_id": rid}, status_code=500)
    response.headers.update({"X-Request-ID": rid, "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY",
                             "Referrer-Policy": "no-referrer", "Cache-Control": "no-store"})
    log.info("%s %s -> %s (%.0f ms) [%s]", request.method, request.url.path, response.status_code, (time.time() - start) * 1000, rid)
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    msg = f"{'.'.join(str(x) for x in first.get('loc', [])[1:])}: {first.get('msg', 'invalid input')}"
    return JSONResponse({"detail": msg}, status_code=422)


# ============================================================================ auth dependencies
bearer_user: Dict[str, Any] = {}


def current_user(request: Request) -> Dict[str, Any]:
    h = request.headers.get("Authorization", "")
    payload = security.decode_token(h[7:]) if h.startswith("Bearer ") else None
    user = users.get(payload["sub"]) if payload else None
    if not user:
        raise HTTPException(401, "Authentication required", headers={"WWW-Authenticate": "Bearer"})
    return user


def require(*roles: str):
    def dep(user: Dict[str, Any] = Depends(current_user)):
        if user["role"] not in roles:
            db.audit(user["user_id"], user["role"], "ACCESS_DENIED", f"Role '{user['role']}' attempted a restricted endpoint.", outcome="denied")
            raise HTTPException(403, "You do not have permission to perform this action")
        return user
    return dep


STAFF = ("support_agent", "operations_manager", "admin")
MANAGE = ("operations_manager", "admin")


# ============================================================================ schemas
class LoginBody(BaseModel):
    user_id: str = Field(min_length=3, max_length=40)
    password: str = Field(min_length=1, max_length=200)


class ChatBody(BaseModel):
    conversation_id: Optional[str] = Field(default=None, max_length=40)
    message: str = Field(default="", max_length=config.MAX_MESSAGE_CHARS * 2)
    fields: Optional[Dict[str, Any]] = None
    action: Optional[Literal["submit", "cancel", "decline", "handoff", "choose_intent", "start_workflow", "fields", "edit", "resume"]] = None
    payload: Optional[Dict[str, Any]] = None


class FeedbackBody(BaseModel):
    message_id: int
    rating: Literal["up", "down"]
    comment: str = Field(default="", max_length=500)


class ActionBody(BaseModel):
    action: Literal["take", "release", "request_info", "resolve", "reject", "reassign", "reopen", "note"]
    note: str = Field(default="", max_length=1000)
    team: Optional[str] = None


class ReplyBody(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


# ============================================================================ health
@app.get("/health", tags=["system"])
def health():
    return {"status": "healthy", "version": app.version, "time": time.time()}


@app.get("/ready", tags=["system"])
def ready():
    try:
        db.count_requests()
        return {"status": "ready", "articles": len(C["retriever"].documents), "request_types": len(C["classifier"].types),
                "workbook": config.WORKBOOK_PATH.name, "llm": C["rag"].status()}
    except Exception as e:
        raise HTTPException(503, f"not ready: {e}")


# ============================================================================ auth
@app.get("/api/auth/demo-accounts", tags=["auth"])
def demo_accounts():
    if not config.DEMO_MODE:
        return {"enabled": False, "accounts": []}
    pick = ["EMP-1001", "AG-2001", "SUP-3001", "OPS-2001", "ADM-4001"]
    return {"enabled": True, "password": config.DEMO_PASSWORD,
            "accounts": [{"user_id": u, "name": users.users[u]["name"], "role": users.users[u]["role"],
                          "role_label": security.ROLE_LABELS[users.users[u]["role"]], "department": users.users[u]["department"]}
                         for u in pick if u in users.users]}


@app.post("/api/auth/login", tags=["auth"])
def login(body: LoginBody, request: Request):
    key = f"login:{request.client.host if request.client else '?'}:{body.user_id.strip().upper()}"
    wait = security.limiter.check(key, config.LOGIN_ATTEMPTS, 300)
    if wait:
        db.audit(body.user_id.upper(), "unknown", "LOGIN_LOCKED", "Too many failed sign-in attempts.", outcome="denied")
        raise HTTPException(429, f"Too many attempts. Try again in {wait} seconds.", headers={"Retry-After": str(wait)})
    user = users.authenticate(body.user_id, body.password)
    if not user:
        db.audit(body.user_id.strip().upper()[:40], "unknown", "LOGIN_FAILED", "Invalid credentials.", outcome="denied")
        raise HTTPException(401, "Invalid user ID or password")
    security.limiter.reset(key)
    tok = security.create_token(user)
    db.audit(user["user_id"], user["role"], "LOGIN_SUCCESS", "User signed in.")
    return {"authenticated": True, "user": user, "token": tok["token"], "expires_at": tok["expires_at"]}


@app.get("/api/auth/me", tags=["auth"])
def me(user=Depends(current_user)):
    return {"user": user, "role_label": security.ROLE_LABELS[user["role"]]}


@app.post("/api/auth/logout", tags=["auth"])
def logout(user=Depends(current_user)):
    db.audit(user["user_id"], user["role"], "LOGOUT", "User signed out.")
    return {"ok": True}


# ============================================================================ assistant
@app.post("/api/chat", tags=["assistant"])
def chat(body: ChatBody, user=Depends(current_user)):
    wait = security.limiter.check(f"chat:{user['user_id']}", config.CHAT_PER_MINUTE, 60)
    if wait:
        raise HTTPException(429, f"You're sending messages very quickly. Please wait {wait}s.", headers={"Retry-After": str(wait)})
    return engine.handle(user, body.conversation_id, body.message, body.fields, body.action, body.payload)


@app.get("/api/conversations", tags=["assistant"])
def conversations(user=Depends(current_user)):
    return {"items": db.list_conversations(user["user_id"])}


@app.get("/api/conversations/{cid}", tags=["assistant"])
def conversation(cid: str, user=Depends(current_user)):
    conv = db.get_conversation(cid)
    if not conv or conv["user_id"] != user["user_id"]:
        raise HTTPException(404, "Conversation not found")
    msgs = db.get_messages(cid)
    state = conv.get("state") or {}
    wf = state.get("workflow")
    return {"id": cid, "title": conv["title"], "messages": [
        {"id": m["id"], "role": m["role"], "content": m["content"], "created_at": m["created_at"], "meta": m["meta"]} for m in msgs],
        "active_workflow": engine._wf_view(wf) if wf else None}


@app.post("/api/chat/feedback", tags=["assistant"])
def feedback(body: FeedbackBody, user=Depends(current_user)):
    db.add_feedback(body.message_id, None, user["user_id"], body.rating, clean_input(body.comment, 500))
    db.audit(user["user_id"], user["role"], "FEEDBACK", f"Rated an answer '{body.rating}'.", meta={"message_id": body.message_id})
    return {"ok": True}


@app.get("/api/catalog", tags=["assistant"])
def catalog(user=Depends(current_user)):
    """Service catalogue shown to employees: what the assistant can guide."""
    items = []
    for rt in store.records("Request_Types"):
        name = rt["request_type"]
        if name.lower() in ("unclear request", "low confidence"):
            continue
        rule = store.routing_rule(name)
        art = store.article_for(name)
        items.append({"request_type": name, "department": rt["department"], "risk_level": rt["risk_level"], "handling_mode": rt["handling_mode"],
                      "team": rule["target_team"] if rule else None, "summary": (art["content"].split(". ")[0] + ".") if art else "",
                      "fields": [store.field_def(f)["label"] for f in store.required_fields(name)],
                      "guided": bool(rule) and not str(rt["handling_mode"]).lower().startswith(("information only", "grounded"))})
    return {"items": items}


# ============================================================================ requests
@app.get("/api/requests", tags=["requests"])
def list_requests(status: Optional[str] = None, team: Optional[str] = None, priority: Optional[str] = None, q: Optional[str] = Query(None, max_length=80),
                  limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), user=Depends(current_user)):
    res = ops.list_for(user, status=status, team=team, priority=priority, q=q, limit=limit, offset=offset)
    return {"total": res["total"], "items": [ops.present(user, r) for r in res["items"]]}


@app.get("/api/requests/{rid}", tags=["requests"])
def get_request(rid: str, user=Depends(current_user)):
    req = ops.get_for(user, rid)
    out = ops.present(user, req, detail=True)
    if user["role"] != "employee" and req.get("conversation_id"):
        out["transcript"] = [{"role": m["role"], "content": m["content"], "created_at": m["created_at"]}
                             for m in db.get_messages(req["conversation_id"], 40)]
    return out


@app.post("/api/requests/{rid}/actions", tags=["requests"])
def request_action(rid: str, body: ActionBody, user=Depends(require(*STAFF))):
    return ops.present(user, ops.transition(user, rid, body.action, body.note, body.team), detail=True)


@app.post("/api/requests/{rid}/reply", tags=["requests"])
def request_reply(rid: str, body: ReplyBody, user=Depends(current_user)):
    return ops.present(user, ops.employee_reply(user, rid, body.text), detail=True)


@app.post("/api/requests/{rid}/cancel", tags=["requests"])
def request_cancel(rid: str, user=Depends(current_user)):
    return ops.present(user, ops.employee_cancel(user, rid), detail=True)


@app.get("/api/requests/{rid}/assist", tags=["requests"])
def request_assist(rid: str, user=Depends(require(*STAFF))):
    req = ops.get_for(user, rid)
    db.audit(user["user_id"], user["role"], "AI_ASSIST_VIEWED", f"Agent assist generated for {rid}.", request_id=req["request_id"])
    return ops.assist(req, C["retriever"])


@app.get("/api/teams", tags=["requests"])
def teams(user=Depends(require(*STAFF))):
    return {"items": store.team_names()}


# ============================================================================ analytics & audit
@app.get("/api/dashboard/summary", tags=["analytics"])
def dashboard(user=Depends(require(*STAFF))):
    return ops.metrics(user)


@app.get("/api/audit", tags=["audit"])
def audit_list(action: Optional[str] = None, actor: Optional[str] = None, request_id: Optional[str] = None, outcome: Optional[str] = None,
               limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0), user=Depends(require(*MANAGE))):
    return db.query_audit(action, actor, request_id, outcome, limit, offset)


@app.get("/api/audit/verify", tags=["audit"])
def audit_verify(user=Depends(require(*MANAGE))):
    return db.verify_audit_chain()


@app.get("/api/audit/export", tags=["audit"])
def audit_export(user=Depends(require(*MANAGE))):
    rows = db.query_audit(limit=5000)["items"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "timestamp", "actor_id", "actor_role", "action", "request_id", "outcome", "description", "hash"])
    for r in reversed(rows):
        w.writerow([r["id"], r["ts"], r["actor_id"], r["actor_role"], r["action"], r["request_id"] or "", r["outcome"],
                    str(r["description"]).replace("\n", " "), r["hash"]])
    db.audit(user["user_id"], user["role"], "AUDIT_EXPORTED", f"Exported {len(rows)} audit rows.")
    return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=careops_audit.csv"})


# ============================================================================ administration
@app.get("/api/admin/overview", tags=["admin"])
def admin_overview(user=Depends(require("admin"))):
    return {"users": len(users.users), "requests": db.count_requests(), "knowledge_articles": store.count("Knowledge_Articles"),
            "workflows": store.count("Workflows"), "request_types": store.count("Request_Types"), "teams": store.count("Teams"),
            "routing_rules": store.count("Routing_Rules"), "safety_rules": store.count("Safety_Rules"), "field_definitions": store.count("Field_Definitions"),
            "training_examples": store.count("Request_Examples"), "system_status": "operational", "ai_mode": ("LLM-grounded RAG (" + C["rag"].client.model + ")") if C["rag"].enabled else "extractive RAG (hybrid BM25 + TF-IDF); LLM not configured",
            "config": {"low_confidence_threshold": config.LOW_CONFIDENCE_THRESHOLD, "clarify_floor": config.CLARIFY_FLOOR,
                       "token_ttl_minutes": config.TOKEN_TTL_MINUTES, "sla_hours": config.SLA_HOURS, "demo_mode": config.DEMO_MODE,
                       "secret_key_configured": not config.SECRET_KEY_IS_DEFAULT},
            "llm": C["rag"].status(), "audit_chain": db.verify_audit_chain(), "workbook": config.WORKBOOK_PATH.name}


@app.get("/api/admin/knowledge", tags=["admin"])
def admin_knowledge(q: str = Query("", max_length=100), user=Depends(require(*MANAGE))):
    docs = C["retriever"].search(q, top_k=10) if q.strip() else [dict(d, score=None) for d in C["retriever"].documents]
    return {"items": docs}


@app.get("/api/admin/routing-rules", tags=["admin"])
def admin_routing(user=Depends(require(*MANAGE))):
    return {"items": store.records("Routing_Rules")}


@app.get("/api/admin/safety-rules", tags=["admin"])
def admin_safety(user=Depends(require(*MANAGE))):
    return {"items": store.records("Safety_Rules")}


@app.get("/api/admin/users", tags=["admin"])
def admin_users(user=Depends(require("admin"))):
    return {"items": [users.public(u) for u in users.users.values()]}


@app.post("/api/admin/reload", tags=["admin"])
def admin_reload(user=Depends(require("admin"))):
    store.reload(); C["classifier"].rebuild(); C["retriever"].rebuild(); users.rebuild()
    db.audit(user["user_id"], user["role"], "KNOWLEDGE_RELOADED", "Workbook reloaded and indexes rebuilt.")
    return {"status": "reloaded", "articles": len(C["retriever"].documents), "request_types": len(C["classifier"].types)}


# ============================================================================ serve the built web app (full-stack, single URL)
FRONTEND_DIST = (config.BASE_DIR / "frontend" / "dist").resolve()


@app.get("/{full_path:path}", include_in_schema=False)
def web_app(full_path: str):
    if full_path.startswith(("api/", "docs", "openapi", "redoc")) or full_path in ("health", "ready"):
        raise HTTPException(404, "Not found")
    index = FRONTEND_DIST / "index.html"
    if not index.exists():
        return JSONResponse({"detail": "Web app not built. Run `cd frontend && npm install && npm run build`, or use the dev server on :5173.",
                             "api_docs": "/docs"}, status_code=200)
    target = (FRONTEND_DIST / full_path).resolve()
    if full_path and FRONTEND_DIST in target.parents and target.is_file():     # blocks ../ traversal
        return FileResponse(target, headers={"Cache-Control": "public, max-age=31536000, immutable"} if "/assets/" in f"/{full_path}" else {})
    return FileResponse(index, headers={"Cache-Control": "no-cache"})            # SPA fallback
