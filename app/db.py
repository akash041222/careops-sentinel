"""SQLite persistence (stdlib only). Thread-safe via a single lock + WAL mode.

Swap for PostgreSQL in production: every query lives in this file.
"""
import hashlib
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations(
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT, state TEXT DEFAULT '{}',
  created_at TEXT, updated_at TEXT);
CREATE INDEX IF NOT EXISTS ix_conv_user ON conversations(user_id, updated_at);
CREATE TABLE IF NOT EXISTS messages(
  id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL, role TEXT NOT NULL,
  content TEXT NOT NULL, meta TEXT DEFAULT '{}', created_at TEXT);
CREATE INDEX IF NOT EXISTS ix_msg_conv ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS requests(
  request_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, request_type TEXT, department TEXT,
  request_text TEXT, fields TEXT DEFAULT '{}', priority TEXT, status TEXT, assigned_team TEXT,
  assignee TEXT, escalation_reason TEXT, confidence REAL, urgency TEXT, sentiment TEXT,
  risk_level TEXT, conversation_id TEXT, sources TEXT DEFAULT '[]', sla_due TEXT,
  resolution_note TEXT, synthetic INTEGER DEFAULT 0, created_at TEXT, updated_at TEXT);
CREATE INDEX IF NOT EXISTS ix_req_user ON requests(user_id);
CREATE INDEX IF NOT EXISTS ix_req_team ON requests(assigned_team, status);
CREATE TABLE IF NOT EXISTS request_events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL, ts TEXT, actor_id TEXT,
  actor_role TEXT, action TEXT, note TEXT, visible_to_employee INTEGER DEFAULT 1);
CREATE INDEX IF NOT EXISTS ix_evt_req ON request_events(request_id, id);
CREATE TABLE IF NOT EXISTS audit(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, request_id TEXT, conversation_id TEXT,
  actor_id TEXT, actor_role TEXT, action TEXT, description TEXT, outcome TEXT, meta TEXT,
  prev_hash TEXT, hash TEXT);
CREATE TABLE IF NOT EXISTS interactions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, user_id TEXT, conversation_id TEXT, intent TEXT,
  action TEXT, confidence REAL, escalated INTEGER, sentiment TEXT, urgency TEXT, flags TEXT);
CREATE TABLE IF NOT EXISTS feedback(
  id INTEGER PRIMARY KEY AUTOINCREMENT, message_id INTEGER, conversation_id TEXT, user_id TEXT,
  rating TEXT, comment TEXT, ts TEXT);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str, n: int = 8) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:n].upper()}"


class Database:
    def __init__(self, path: Path):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    # ------------------------------------------------------------------ generic
    def _all(self, sql: str, args=()) -> List[Dict[str, Any]]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def _one(self, sql: str, args=()) -> Optional[Dict[str, Any]]:
        rows = self._all(sql, args)
        return rows[0] if rows else None

    def _exec(self, sql: str, args=()) -> int:
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur.lastrowid

    @staticmethod
    def _decode(row: Optional[Dict[str, Any]], *json_cols: str):
        if row is None:
            return None
        for c in json_cols:
            if c in row and isinstance(row[c], str):
                try:
                    row[c] = json.loads(row[c])
                except json.JSONDecodeError:
                    row[c] = {} if c in ("fields", "state", "meta") else []
        return row

    # ------------------------------------------------------------------ conversations
    def create_conversation(self, user_id: str, title: str = "New conversation") -> Dict[str, Any]:
        cid = new_id("CNV", 10)
        ts = now_iso()
        self._exec("INSERT INTO conversations(id,user_id,title,state,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                   (cid, user_id, title, "{}", ts, ts))
        return self.get_conversation(cid)

    def get_conversation(self, cid: str) -> Optional[Dict[str, Any]]:
        return self._decode(self._one("SELECT * FROM conversations WHERE id=?", (cid,)), "state")

    def list_conversations(self, user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
        return self._all("SELECT id,title,created_at,updated_at FROM conversations WHERE user_id=? "
                         "ORDER BY updated_at DESC LIMIT ?", (user_id, limit))

    def save_conversation_state(self, cid: str, state: Dict[str, Any], title: Optional[str] = None):
        if title:
            self._exec("UPDATE conversations SET state=?, title=?, updated_at=? WHERE id=?",
                       (json.dumps(state), title, now_iso(), cid))
        else:
            self._exec("UPDATE conversations SET state=?, updated_at=? WHERE id=?",
                       (json.dumps(state), now_iso(), cid))

    def add_message(self, cid: str, role: str, content: str, meta: Optional[Dict[str, Any]] = None) -> int:
        return self._exec("INSERT INTO messages(conversation_id,role,content,meta,created_at) VALUES(?,?,?,?,?)",
                          (cid, role, content, json.dumps(meta or {}), now_iso()))

    def get_messages(self, cid: str, limit: int = 200) -> List[Dict[str, Any]]:
        rows = self._all("SELECT * FROM messages WHERE conversation_id=? ORDER BY id ASC LIMIT ?", (cid, limit))
        return [self._decode(r, "meta") for r in rows]

    # ------------------------------------------------------------------ requests
    def insert_request(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        cols = ["request_id", "user_id", "request_type", "department", "request_text", "fields", "priority",
                "status", "assigned_team", "assignee", "escalation_reason", "confidence", "urgency",
                "sentiment", "risk_level", "conversation_id", "sources", "sla_due", "resolution_note",
                "synthetic", "created_at", "updated_at"]
        vals = []
        for c in cols:
            v = rec.get(c)
            if c in ("fields", "sources"):
                v = json.dumps(v if v is not None else ({} if c == "fields" else []))
            vals.append(v)
        self._exec(f"INSERT INTO requests({','.join(cols)}) VALUES({','.join('?' * len(cols))})", vals)
        return self.get_request(rec["request_id"])

    def get_request(self, rid: str) -> Optional[Dict[str, Any]]:
        return self._decode(self._one("SELECT * FROM requests WHERE request_id=?", (rid,)), "fields", "sources")

    def update_request(self, rid: str, **changes):
        if not changes:
            return
        changes["updated_at"] = now_iso()
        for k in ("fields", "sources"):
            if k in changes and not isinstance(changes[k], str):
                changes[k] = json.dumps(changes[k])
        sets = ", ".join(f"{k}=?" for k in changes)
        self._exec(f"UPDATE requests SET {sets} WHERE request_id=?", (*changes.values(), rid))

    def query_requests(self, user_id: Optional[str] = None, teams: Optional[List[str]] = None,
                       status: Optional[str] = None, team: Optional[str] = None, priority: Optional[str] = None,
                       q: Optional[str] = None, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        where, args = [], []
        if user_id:
            where.append("user_id=?"); args.append(user_id)
        if teams is not None:
            if not teams:
                return {"total": 0, "items": []}
            where.append(f"assigned_team IN ({','.join('?' * len(teams))})"); args.extend(teams)
        if status:
            where.append("status=?"); args.append(status)
        if team:
            where.append("assigned_team=?"); args.append(team)
        if priority:
            where.append("priority=?"); args.append(priority)
        if q:
            where.append("(request_id LIKE ? OR request_type LIKE ? OR request_text LIKE ? OR user_id LIKE ?)")
            args.extend([f"%{q}%"] * 4)
        w = (" WHERE " + " AND ".join(where)) if where else ""
        total = self._one(f"SELECT COUNT(*) n FROM requests{w}", args)["n"]
        rows = self._all(
            f"SELECT * FROM requests{w} ORDER BY CASE priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1 "
            f"WHEN 'normal' THEN 2 ELSE 3 END, created_at DESC LIMIT ? OFFSET ?", (*args, limit, offset))
        return {"total": total, "items": [self._decode(r, "fields", "sources") for r in rows]}

    def all_requests(self, teams: Optional[List[str]] = None, user_id: Optional[str] = None):
        return self.query_requests(user_id=user_id, teams=teams, limit=100000)["items"]

    def count_requests(self) -> int:
        return self._one("SELECT COUNT(*) n FROM requests")["n"]

    # ------------------------------------------------------------------ request events
    def add_event(self, rid: str, actor_id: str, actor_role: str, action: str, note: str = "",
                  visible_to_employee: bool = True, ts: Optional[str] = None):
        self._exec("INSERT INTO request_events(request_id,ts,actor_id,actor_role,action,note,visible_to_employee) "
                   "VALUES(?,?,?,?,?,?,?)",
                   (rid, ts or now_iso(), actor_id, actor_role, action, note, 1 if visible_to_employee else 0))

    def get_events(self, rid: str, employee_view: bool = False) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM request_events WHERE request_id=?"
        if employee_view:
            sql += " AND visible_to_employee=1"
        return self._all(sql + " ORDER BY id ASC", (rid,))

    # ------------------------------------------------------------------ audit (hash-chained)
    def audit(self, actor_id: str, actor_role: str, action: str, description: str, outcome: str = "success",
              request_id: Optional[str] = None, conversation_id: Optional[str] = None,
              meta: Optional[Dict[str, Any]] = None, ts: Optional[str] = None) -> None:
        with self.lock:
            last = self._one("SELECT hash FROM audit ORDER BY id DESC LIMIT 1")
            prev = last["hash"] if last else "GENESIS"
            ts = ts or now_iso()
            meta_s = json.dumps(meta or {}, sort_keys=True)
            payload = "|".join([prev, ts, str(request_id), str(actor_id), str(actor_role), action, description, outcome, meta_s])
            h = hashlib.sha256(payload.encode()).hexdigest()
            self._exec("INSERT INTO audit(ts,request_id,conversation_id,actor_id,actor_role,action,description,"
                       "outcome,meta,prev_hash,hash) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (ts, request_id, conversation_id, actor_id, actor_role, action, description, outcome,
                        meta_s, prev, h))

    def query_audit(self, action: Optional[str] = None, actor: Optional[str] = None,
                    request_id: Optional[str] = None, outcome: Optional[str] = None,
                    limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        where, args = [], []
        for col, val in (("action", action), ("actor_id", actor), ("request_id", request_id), ("outcome", outcome)):
            if val:
                where.append(f"{col}=?"); args.append(val)
        w = (" WHERE " + " AND ".join(where)) if where else ""
        total = self._one(f"SELECT COUNT(*) n FROM audit{w}", args)["n"]
        rows = self._all(f"SELECT * FROM audit{w} ORDER BY id DESC LIMIT ? OFFSET ?", (*args, limit, offset))
        return {"total": total, "items": [self._decode(r, "meta") for r in rows]}

    def verify_audit_chain(self) -> Dict[str, Any]:
        rows = self._all("SELECT * FROM audit ORDER BY id ASC")
        prev = "GENESIS"
        for r in rows:
            payload = "|".join([prev, r["ts"], str(r["request_id"]), str(r["actor_id"]), str(r["actor_role"]),
                                r["action"], r["description"], r["outcome"], r["meta"]])
            if r["prev_hash"] != prev or hashlib.sha256(payload.encode()).hexdigest() != r["hash"]:
                return {"valid": False, "checked": len(rows), "broken_at": r["id"]}
            prev = r["hash"]
        return {"valid": True, "checked": len(rows), "broken_at": None}

    # ------------------------------------------------------------------ interactions / feedback
    def log_interaction(self, user_id, conversation_id, intent, action, confidence, escalated,
                        sentiment, urgency, flags, ts: Optional[str] = None):
        self._exec("INSERT INTO interactions(ts,user_id,conversation_id,intent,action,confidence,escalated,"
                   "sentiment,urgency,flags) VALUES(?,?,?,?,?,?,?,?,?,?)",
                   (ts or now_iso(), user_id, conversation_id, intent, action, confidence, 1 if escalated else 0,
                    sentiment, urgency, json.dumps(flags or [])))

    def interactions(self) -> List[Dict[str, Any]]:
        return self._all("SELECT * FROM interactions")

    def add_feedback(self, message_id, conversation_id, user_id, rating, comment):
        self._exec("INSERT INTO feedback(message_id,conversation_id,user_id,rating,comment,ts) VALUES(?,?,?,?,?,?)",
                   (message_id, conversation_id, user_id, rating, comment, now_iso()))

    def feedback_summary(self) -> Dict[str, int]:
        rows = self._all("SELECT rating, COUNT(*) n FROM feedback GROUP BY rating")
        return {r["rating"]: r["n"] for r in rows}
