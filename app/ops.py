"""Request lifecycle, access control, SLA, agent assist and metrics. All state changes here are human-initiated."""
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from . import config
from .db import Database, new_id, now_iso

OPEN_STATUSES = {"pending_human", "in_progress", "needs_information"}
TERMINAL = {"resolved", "rejected", "cancelled"}
STATUS_LABEL = {"pending_human": "Pending review", "in_progress": "In progress", "needs_information": "Needs information",
                "resolved": "Resolved", "rejected": "Rejected", "cancelled": "Cancelled"}
PRIORITY_RANK = {"low": 0, "normal": 1, "high": 2, "critical": 3}


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


class Operations:
    def __init__(self, store, db: Database, users):
        self.store, self.db, self.users = store, db, users

    # ------------------------------------------------------------------ priority / SLA
    def priority_for(self, request_type: str, tone: Dict[str, Any]) -> Tuple[str, List[str]]:
        rule = self.store.routing_rule(request_type) or {}
        rt = self.store.request_type(request_type) or {}
        base = {"critical": "critical", "high": "high", "sensitive": "high", "medium": "normal"}.get(
            str(rule.get("priority", "normal")).lower(), "normal")
        reasons: List[str] = []
        if str(rt.get("risk_level", "")).lower() == "critical":
            base = "critical"; reasons.append("critical-risk request type")
        if tone.get("urgency") == "high" and PRIORITY_RANK[base] < 2:
            base = "high"; reasons.append("urgency detected in message")
        if tone.get("sentiment") == "negative" and PRIORITY_RANK[base] < 2:
            base = "high"; reasons.append("negative sentiment detected")
        return base, reasons

    @staticmethod
    def sla_due(priority: str, start: Optional[datetime] = None) -> str:
        start = start or datetime.now(timezone.utc)
        return (start + timedelta(hours=config.SLA_HOURS.get(priority, 24))).isoformat(timespec="seconds")

    # ------------------------------------------------------------------ create
    def create_request(self, *, user: Dict[str, Any], request_type: str, fields: Dict[str, Any], request_text: str,
                       tone: Dict[str, Any], confidence: Optional[float], sources: List[Dict[str, Any]],
                       conversation_id: Optional[str], extra_reasons: Optional[List[str]] = None,
                       team_override: Optional[str] = None, risk_level: Optional[str] = None) -> Dict[str, Any]:
        rule = self.store.routing_rule(request_type) or {}
        rt = self.store.request_type(request_type) or {}
        team = team_override or rule.get("target_team") or "Operations Support"
        priority, why = self.priority_for(request_type, tone)
        reasons = list(extra_reasons or [])
        if rule.get("routing_action"):
            reasons.insert(0, str(rule["routing_action"]))
        reasons += why
        rid = new_id("REQ", 8)
        rec = self.db.insert_request({
            "request_id": rid, "user_id": user["user_id"], "request_type": request_type,
            "department": rt.get("department") or rule.get("source_department") or "",
            "request_text": request_text, "fields": fields, "priority": priority, "status": "pending_human",
            "assigned_team": team, "assignee": None, "escalation_reason": "; ".join(dict.fromkeys(reasons)) or "Requires human processing.",
            "confidence": confidence, "urgency": tone.get("urgency"), "sentiment": tone.get("sentiment"),
            "risk_level": risk_level or rt.get("risk_level"), "conversation_id": conversation_id, "sources": sources,
            "sla_due": self.sla_due(priority), "resolution_note": None, "synthetic": 0,
            "created_at": now_iso(), "updated_at": now_iso()})
        self.db.add_event(rid, user["user_id"], user["role"], "submitted", "Request submitted via CareOps Assistant.")
        self.db.audit(user["user_id"], user["role"], "REQUEST_SUBMITTED",
                      f"{request_type} routed to {team} (priority {priority}).", request_id=rid,
                      conversation_id=conversation_id, meta={"team": team, "priority": priority, "confidence": confidence})
        return rec

    # ------------------------------------------------------------------ access control
    def can_view(self, user: Dict[str, Any], req: Dict[str, Any]) -> bool:
        if user["role"] == "employee":
            return req["user_id"] == user["user_id"]
        teams = self.users.teams_for(user)
        return teams is None or req["assigned_team"] in teams

    def get_for(self, user: Dict[str, Any], rid: str) -> Dict[str, Any]:
        req = self.db.get_request(rid.upper())
        if not req or not self.can_view(user, req):
            if req:
                self.db.audit(user["user_id"], user["role"], "ACCESS_DENIED", f"Attempted to open {rid}.",
                              outcome="denied", request_id=rid)
            raise HTTPException(404, "Request not found")
        return req

    def list_for(self, user: Dict[str, Any], **filters) -> Dict[str, Any]:
        if user["role"] == "employee":
            return self.db.query_requests(user_id=user["user_id"], **filters)
        return self.db.query_requests(teams=self.users.teams_for(user), **filters)

    # ------------------------------------------------------------------ presentation
    def sla_state(self, req: Dict[str, Any]) -> str:
        if req["status"] not in OPEN_STATUSES or not req.get("sla_due"):
            return "n/a"
        remaining = (_dt(req["sla_due"]) - datetime.now(timezone.utc)).total_seconds()
        return "breached" if remaining < 0 else ("at_risk" if remaining < 2 * 3600 else "on_track")

    def available_actions(self, user: Dict[str, Any], req: Dict[str, Any]) -> List[str]:
        s, role = req["status"], user["role"]
        if role == "employee":
            acts = []
            if req["user_id"] == user["user_id"]:
                if s == "needs_information":
                    acts.append("reply")
                if s in ("pending_human", "needs_information"):
                    acts.append("cancel")
            return acts
        if not self.can_view(user, req):
            return []
        acts = []
        if s == "pending_human":
            acts += ["take", "request_info", "reject"]
        elif s == "in_progress":
            acts += ["resolve", "request_info", "reject", "release"]
        elif s == "needs_information":
            acts += ["take", "reject"]
        elif s in ("resolved", "rejected") and role in ("operations_manager", "admin"):
            acts += ["reopen"]
        if s in OPEN_STATUSES:
            acts += ["reassign"]
        return acts + ["note"]

    def present(self, user: Dict[str, Any], req: Dict[str, Any], detail: bool = False) -> Dict[str, Any]:
        out = dict(req)
        out["status_label"] = STATUS_LABEL.get(req["status"], req["status"])
        out["sla_state"] = self.sla_state(req)
        out["actions"] = self.available_actions(user, req)
        emp = self.users.get(req["user_id"])
        out["requester_name"] = emp["name"] if emp else req["user_id"]
        if user["role"] == "employee":     # employees never see internal AI/agent context
            for k in ("escalation_reason", "sources", "sentiment", "urgency", "confidence", "assignee", "risk_level"):
                out.pop(k, None)
            out["escalation_reason"] = None
        if detail:
            out["events"] = self.db.get_events(req["request_id"], employee_view=user["role"] == "employee")
        return out

    # ------------------------------------------------------------------ transitions (human only)
    def transition(self, user: Dict[str, Any], rid: str, action: str, note: str = "", team: Optional[str] = None):
        if user["role"] == "employee":
            raise HTTPException(403, "Human support role required")
        req = self.get_for(user, rid)
        if action not in self.available_actions(user, req):
            self.db.audit(user["user_id"], user["role"], "ACTION_REJECTED",
                          f"'{action}' not allowed on {rid} in status {req['status']}.", outcome="denied", request_id=rid)
            raise HTTPException(409, f"Action '{action}' is not allowed while the request is '{req['status']}'.")
        note = (note or "").strip()[:1000]
        if action in ("resolve", "reject", "request_info", "reopen", "reassign", "note") and not note:
            raise HTTPException(422, "A note is required for this action.")
        changes: Dict[str, Any] = {}
        visible = True
        if action == "take":
            changes = {"status": "in_progress", "assignee": user["user_id"]}
        elif action == "release":
            if req.get("assignee") not in (user["user_id"], None) and user["role"] == "support_agent":
                raise HTTPException(403, "Only the assignee or a manager can release this request.")
            changes = {"status": "pending_human", "assignee": None}
        elif action == "request_info":
            changes = {"status": "needs_information"}
        elif action == "resolve":
            if req.get("assignee") not in (user["user_id"], None) and user["role"] == "support_agent":
                raise HTTPException(403, "This request is assigned to another agent.")
            changes = {"status": "resolved", "resolution_note": note}
        elif action == "reject":
            changes = {"status": "rejected", "resolution_note": note}
        elif action == "reopen":
            changes = {"status": "in_progress", "assignee": user["user_id"], "resolution_note": None}
        elif action == "reassign":
            if team not in self.store.team_names():
                raise HTTPException(422, "Unknown team.")
            changes = {"assigned_team": team, "status": "pending_human", "assignee": None}
        elif action == "note":
            visible = False
        if changes:
            self.db.update_request(req["request_id"], **changes)
        self.db.add_event(req["request_id"], user["user_id"], user["role"], action, note or STATUS_LABEL.get(changes.get("status", ""), ""),
                          visible_to_employee=visible)
        self.db.audit(user["user_id"], user["role"], "REQUEST_" + action.upper(),
                      f"{action} on {rid}: {note[:120]}" if note else f"{action} on {rid}", request_id=req["request_id"],
                      meta={"from": req["status"], "to": changes.get("status", req["status"])})
        return self.db.get_request(req["request_id"])

    def employee_reply(self, user: Dict[str, Any], rid: str, text: str):
        req = self.get_for(user, rid)
        if "reply" not in self.available_actions(user, req):
            raise HTTPException(409, "This request is not waiting for your reply.")
        from .text_utils import clean_input, redact
        text, found = redact(clean_input(text, 1000))
        if not text:
            raise HTTPException(422, "Reply cannot be empty.")
        self.db.update_request(req["request_id"], status="pending_human")
        self.db.add_event(req["request_id"], user["user_id"], user["role"], "employee_reply", text)
        self.db.audit(user["user_id"], user["role"], "EMPLOYEE_REPLY", f"Employee replied on {rid}.", request_id=req["request_id"])
        return self.db.get_request(req["request_id"])

    def employee_cancel(self, user: Dict[str, Any], rid: str):
        req = self.get_for(user, rid)
        if "cancel" not in self.available_actions(user, req):
            raise HTTPException(409, "This request can no longer be cancelled.")
        self.db.update_request(req["request_id"], status="cancelled")
        self.db.add_event(req["request_id"], user["user_id"], user["role"], "cancelled", "Cancelled by requester.")
        self.db.audit(user["user_id"], user["role"], "REQUEST_CANCELLED", f"{rid} cancelled by requester.", request_id=req["request_id"])
        return self.db.get_request(req["request_id"])

    # ------------------------------------------------------------------ agent assist (advisory only)
    def assist(self, req: Dict[str, Any], retriever) -> Dict[str, Any]:
        rt = req["request_type"]
        rule = self.store.routing_rule(rt) or {}
        wf = self.store.workflow(rt) or {}
        article = self.store.article_for(rt)
        fields = req.get("fields") or {}
        lines = [f"{self.store.field_def(k)['label']}: {v}" for k, v in fields.items()]
        emp = self.users.get(req["user_id"]) or {"name": req["user_id"], "department": "?"}
        summary = (f"{emp['name']} ({req['user_id']}, {emp['department']}) submitted a **{rt}** request"
                   + (f" - {'; '.join(lines[:6])}." if lines else "."))
        checklist = ["Verify the requester's identity and employment status before acting."]
        if rule.get("routing_action"):
            checklist.append(f"Policy: {rule['routing_action']}.")
        if wf.get("completion_action"):
            checklist.append(f"Completion path: {str(wf['completion_action']).replace('_', ' ')}.")
        for k in fields:
            note = self.store.field_def(k).get("safety")
            if note:
                checklist.append(f"{self.store.field_def(k)['label']}: {note}.")
        if req.get("sentiment") == "negative":
            checklist.append("Requester sounded frustrated - acknowledge the delay and give a clear next step.")
        if req.get("urgency") == "high" or req["priority"] in ("high", "critical"):
            checklist.append(f"Priority is {req['priority']} - SLA due {req['sla_due']}.")
        draft = (f"Hi {emp['name'].split()[0]}, thanks for your {rt.lower()} request ({req['request_id']}). "
                 f"I'm reviewing it now"
                 + (f" in line with our policy: {article['content'].split('.')[0]}." if article else ".")
                 + " I'll update you as soon as there is progress.")
        return {"summary": summary, "checklist": checklist, "draft_reply": draft,
                "knowledge": ({"article_id": article["article_id"], "title": article["title"], "content": article["content"],
                               "source_name": article["source_name"]} if article else None),
                "note": "AI suggestions are advisory. You make the final decision."}

    # ------------------------------------------------------------------ metrics
    def metrics(self, user: Dict[str, Any]) -> Dict[str, Any]:
        teams = self.users.teams_for(user)
        reqs = self.db.all_requests(teams=teams)
        now = datetime.now(timezone.utc)
        status = Counter(r["status"] for r in reqs)
        by_team = Counter(r["assigned_team"] for r in reqs)
        by_prio = Counter(r["priority"] for r in reqs if r["status"] in OPEN_STATUSES)
        open_reqs = [r for r in reqs if r["status"] in OPEN_STATUSES]
        sla = Counter(self.sla_state(r) for r in open_reqs)
        confs = [r["confidence"] for r in reqs if r.get("confidence") is not None]
        res_hours = [(_dt(r["updated_at"]) - _dt(r["created_at"])).total_seconds() / 3600 for r in reqs if r["status"] == "resolved"]
        trend = []
        for i in range(6, -1, -1):
            d = (now - timedelta(days=i)).date().isoformat()
            trend.append({"date": d,
                          "created": sum(1 for r in reqs if r["created_at"][:10] == d),
                          "resolved": sum(1 for r in reqs if r["status"] == "resolved" and r["updated_at"][:10] == d)})
        out: Dict[str, Any] = {
            "scope": "all teams" if teams is None else ", ".join(teams),
            "total_requests": len(reqs), "open_requests": len(open_reqs),
            "status_counts": {k: status.get(k, 0) for k in ("pending_human", "in_progress", "needs_information", "resolved", "rejected", "cancelled")},
            "team_counts": dict(by_team.most_common()), "open_by_priority": {k: by_prio.get(k, 0) for k in ("critical", "high", "normal", "low")},
            "sla": {"breached": sla.get("breached", 0), "at_risk": sla.get("at_risk", 0), "on_track": sla.get("on_track", 0)},
            "avg_ai_confidence": round(sum(confs) / len(confs), 2) if confs else None,
            "avg_resolution_hours": round(sum(res_hours) / len(res_hours), 1) if res_hours else None,
            "trend": trend, "generated_at": now_iso(), "human_in_loop": True,
            "safety_boundary": "Clinical and confidential requests are never answered by the AI; they are routed to people.",
        }
        if user["role"] in ("operations_manager", "admin"):
            inter = self.db.interactions()
            flags = Counter()
            for i in inter:
                for f in json.loads(i["flags"] or "[]"):
                    flags[f] += 1
            n = len(inter)
            escalated = sum(i["escalated"] for i in inter)
            confs2 = [i["confidence"] for i in inter if i["confidence"] is not None]
            out["assistant"] = {
                "turns": n, "escalated_turns": escalated,
                "containment_rate": round(1 - escalated / n, 2) if n else None,
                "avg_confidence": round(sum(confs2) / len(confs2), 2) if confs2 else None,
                "sentiment": dict(Counter(i["sentiment"] for i in inter if i["sentiment"])),
                "urgency": dict(Counter(i["urgency"] for i in inter if i["urgency"])),
                "top_intents": Counter(i["intent"] for i in inter if i["intent"]).most_common(6),
                "safety_flags": dict(flags), "feedback": self.db.feedback_summary(),
            }
        return out
