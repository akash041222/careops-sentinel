"""Fills an EMPTY database with clearly-labelled synthetic requests so dashboards are not blank on first run."""
import json
import random
from datetime import datetime, timedelta, timezone

from . import config
from .db import new_id


def seed_demo(store, db, ops, users) -> int:
    if db.count_requests() > 0:
        return 0
    rnd = random.Random(42)
    examples = [e for e in store.records("Request_Examples")
                if store.routing_rule(e["request_type"]) and e["request_type"] not in ("Unclear Request",)]
    emps = [u for u in users.users.values() if u["role"] == "employee"]
    agents = [u for u in users.users.values() if u["role"] == "support_agent"]
    now = datetime.now(timezone.utc)
    statuses = ["pending_human"] * 8 + ["in_progress"] * 6 + ["resolved"] * 9 + ["needs_information"] + ["rejected"] * 2
    count = 0
    for i, ex in enumerate(rnd.sample(examples, 26)):
        rt = ex["request_type"]
        rule = store.routing_rule(rt)
        emp = rnd.choice(emps)
        status = rnd.choice(statuses)
        sla_h = None
        tone = {"urgency": rnd.choice(["normal"] * 4 + ["high"]), "sentiment": rnd.choice(["neutral"] * 5 + ["negative"])}
        priority, why = ops.priority_for(rt, tone)
        sla_h = config.SLA_HOURS[priority]
        if status in ("pending_human", "in_progress", "needs_information"):
            # mostly on-track, with a few deliberately breached examples
            created = now - timedelta(hours=(sla_h * rnd.uniform(1.05, 1.6)) if i % 7 == 0 else rnd.uniform(0.3, sla_h * 0.7))
        else:
            created = now - timedelta(hours=rnd.randint(6, 6 * 24))
        fields = {f: ("EMP-" + emp["user_id"].split("-")[1] if f in ("employee_id", "requester_id") else "(synthetic sample)")
                  for f in store.required_fields(rt)}
        fields = {f: (emp["user_id"] if f in ("employee_id", "requester_id") else v) for f, v in fields.items()}
        rid = new_id("REQ", 8)
        team = rule["target_team"]
        agent = next((a for a in agents if team in store.teams_for_department(a["department"])), None)
        assignee = agent["user_id"] if (agent and status in ("in_progress", "resolved")) else None
        updated = created + timedelta(hours=rnd.randint(1, 20)) if status != "pending_human" else created
        db.insert_request({
            "request_id": rid, "user_id": emp["user_id"], "request_type": rt, "department": ex["department"],
            "request_text": ex["request_text"], "fields": fields, "priority": priority, "status": status,
            "assigned_team": team, "assignee": assignee,
            "escalation_reason": "; ".join([str(rule["routing_action"])] + why), "confidence": round(float(ex["confidence_example"] or 0.9), 2),
            "urgency": tone["urgency"], "sentiment": tone["sentiment"], "risk_level": ex["risk_level"], "conversation_id": None,
            "sources": [], "sla_due": ops.sla_due(priority, created),
            "resolution_note": "Completed by team (synthetic sample)." if status == "resolved" else
                               ("Cannot be approved under current policy (synthetic sample)." if status == "rejected" else None),
            "synthetic": 1, "created_at": created.isoformat(timespec="seconds"), "updated_at": updated.isoformat(timespec="seconds")})
        db.add_event(rid, emp["user_id"], "employee", "submitted", "Request submitted (synthetic sample).", ts=created.isoformat(timespec="seconds"))
        if status != "pending_human" and agent:
            db.add_event(rid, agent["user_id"], "support_agent", "take", "Picked up by agent.", ts=(created + timedelta(minutes=30)).isoformat(timespec="seconds"))
        if status == "resolved":
            db.add_event(rid, agent["user_id"] if agent else "SUP-3001", "support_agent", "resolve", "Completed (synthetic sample).", ts=updated.isoformat(timespec="seconds"))
        db.audit(emp["user_id"], "employee", "REQUEST_SUBMITTED", f"{rt} routed to {team} (synthetic sample).", request_id=rid,
                 ts=created.isoformat(timespec="seconds"), meta={"synthetic": True})
        count += 1
    # synthetic assistant interactions for the analytics view
    kinds = ["answer"] * 14 + ["collect"] * 8 + ["smalltalk"] * 6 + ["submitted"] * 6 + ["refusal"] * 2 + ["no_answer"] * 2 + ["confirm_intent"] * 3
    intents = [e["request_type"] for e in rnd.sample(examples, 12)]
    for _ in range(60):
        k = rnd.choice(kinds)
        ts = (now - timedelta(hours=rnd.randint(1, 7 * 24))).isoformat(timespec="seconds")
        db.log_interaction(rnd.choice(emps)["user_id"], None, None if k == "smalltalk" else rnd.choice(intents), k,
                           None if k == "smalltalk" else round(rnd.uniform(0.55, 0.97), 2), k in ("submitted", "refusal", "no_answer"),
                           rnd.choice(["neutral"] * 6 + ["negative", "positive"]), rnd.choice(["normal"] * 5 + ["high"]),
                           ["clinical_advice"] if k == "refusal" else [], ts=ts)
    db.audit("SYSTEM", "system", "SEED_DATA", f"Loaded {count} synthetic sample requests for demonstration.", meta={"synthetic": True})
    return count
