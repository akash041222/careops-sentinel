import pytest


def test_auth_required(client):
    for path in ("/api/requests", "/api/dashboard/summary", "/api/audit", "/api/conversations", "/api/auth/me", "/api/catalog"):
        assert client.get(path).status_code == 401
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 401


def test_forged_and_expired_tokens_rejected(client):
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer abc.def"}).status_code == 401
    from app import security
    tok = security.create_token({"user_id": "EMP-1001", "role": "employee"}, ttl_minutes=-1)["token"]
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer " + tok}).status_code == 401
    good = security.create_token({"user_id": "EMP-1001", "role": "employee"})["token"]
    body, sig = good.split(".")
    forged = security.create_token({"user_id": "ADM-4001", "role": "admin"})["token"].split(".")[0] + "." + sig
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer " + forged}).status_code == 401


def test_login_failures_and_lockout(client):
    assert client.post("/api/auth/login", json={"user_id": "EMP-1004", "password": "wrong"}).status_code == 401
    codes = [client.post("/api/auth/login", json={"user_id": "EMP-1005", "password": "bad"}).status_code for _ in range(10)]
    assert 429 in codes


def test_role_restrictions(client, auth):
    emp = auth("EMP-1001")
    assert client.get("/api/dashboard/summary", headers=emp).status_code == 403
    assert client.get("/api/audit", headers=emp).status_code == 403
    assert client.get("/api/admin/overview", headers=auth("OPS-2001")).status_code == 403
    assert client.get("/api/admin/overview", headers=auth("ADM-4001")).status_code == 200
    assert client.get("/api/audit", headers=auth("AG-2001")).status_code == 403
    assert client.get("/api/dashboard/summary", headers=auth("AG-2001")).status_code == 200


def test_employees_only_see_own_requests_and_no_ai_internals(client, auth):
    items = client.get("/api/requests", headers=auth("EMP-1002")).json()["items"]
    assert items and all(i["user_id"] == "EMP-1002" for i in items)
    assert all("sources" not in i and "confidence" not in i and "sentiment" not in i for i in items)


def test_agents_are_scoped_to_their_teams(client, auth):
    hr = client.get("/api/requests", headers=auth("AG-2001")).json()["items"]
    assert hr and all(i["assigned_team"] in ("HR Employee Services", "HR Support", "Payroll Support", "HR Offboarding", "HR Documents",
                                              "Learning Operations", "HR Benefits Support") for i in hr)
    it = client.get("/api/requests", headers=auth("AG-2002")).json()["items"]
    foreign = next(i for i in it if True)
    assert client.get(f"/api/requests/{foreign['request_id']}", headers=auth("AG-2001")).status_code == 404
    assert client.post(f"/api/requests/{foreign['request_id']}/actions", headers=auth("AG-2001"), json={"action": "take"}).status_code == 404


def test_human_only_transitions(client, auth, chat):
    c = chat("EMP-1004")
    c.say("I need a replacement ID card because mine is damaged")
    c.say("department: Facilities, contact method: Email")
    rid = c.say(action="submit")["request"]["request_id"]
    emp, ag, hr = auth("EMP-1004"), auth("AG-2001"), auth("AG-2001")
    assert client.post(f"/api/requests/{rid}/actions", headers=emp, json={"action": "resolve", "note": "me"}).status_code == 403
    assert client.post(f"/api/requests/{rid}/actions", headers=ag, json={"action": "resolve", "note": "x"}).status_code == 409   # must take first
    assert client.post(f"/api/requests/{rid}/actions", headers=ag, json={"action": "take"}).json()["status"] == "in_progress"
    assert client.post(f"/api/requests/{rid}/actions", headers=ag, json={"action": "resolve"}).status_code == 422             # note required
    r = client.post(f"/api/requests/{rid}/actions", headers=ag, json={"action": "request_info", "note": "Which building?"}).json()
    assert r["status"] == "needs_information"
    assert client.post(f"/api/requests/{rid}/reply", headers=emp, json={"text": "Building B"}).json()["status"] == "pending_human"
    assert client.post(f"/api/requests/{rid}/actions", headers=ag, json={"action": "take"}).status_code == 200
    assert client.post(f"/api/requests/{rid}/actions", headers=ag, json={"action": "resolve", "note": "Card ordered"}).json()["status"] == "resolved"
    assert client.post(f"/api/requests/{rid}/actions", headers=ag, json={"action": "take"}).status_code == 409                 # terminal
    events = client.get(f"/api/requests/{rid}", headers=emp).json()["events"]
    assert [e["action"] for e in events][-1] == "resolve"
    assert client.post(f"/api/requests/{rid}/cancel", headers=emp).status_code == 409


def test_internal_notes_hidden_from_employee(client, auth, chat):
    c = chat("EMP-1002"); c.say("I need a replacement ID card because mine is damaged"); c.say("department: HR, contact method: Phone")
    rid = c.say(action="submit")["request"]["request_id"]
    client.post(f"/api/requests/{rid}/actions", headers=auth("AG-2001"), json={"action": "note", "note": "INTERNAL ONLY"})
    ev = client.get(f"/api/requests/{rid}", headers=auth("EMP-1002")).json()["events"]
    assert not any("INTERNAL" in (e["note"] or "") for e in ev)
    assert any("INTERNAL" in (e["note"] or "") for e in client.get(f"/api/requests/{rid}", headers=auth("AG-2001")).json()["events"])


def test_agent_assist_is_advisory(client, auth):
    item = client.get("/api/requests", headers=auth("AG-2001")).json()["items"][0]
    a = client.get(f"/api/requests/{item['request_id']}/assist", headers=auth("AG-2001")).json()
    assert a["summary"] and a["checklist"] and "advisory" in a["note"]
    assert client.get(f"/api/requests/{item['request_id']}/assist", headers=auth("EMP-1002")).status_code == 403


def test_audit_trail_is_complete_and_tamper_evident(client, auth):
    mg = auth("OPS-2001")
    actions = {a["action"] for a in client.get("/api/audit?limit=500", headers=mg).json()["items"]}
    assert {"LOGIN_SUCCESS", "LOGIN_FAILED", "CHAT_TURN", "SAFETY_BLOCK", "REQUEST_SUBMITTED", "ACCESS_DENIED"} <= actions
    assert client.get("/api/audit/verify", headers=mg).json()["valid"] is True
    assert client.get("/api/audit/export", headers=mg).headers["content-type"].startswith("text/csv")


def test_audit_tampering_is_detected():
    from app.db import Database
    from pathlib import Path
    d = Database(Path(":memory:"))
    for i in range(3):
        d.audit("u", "employee", "X", f"event {i}")
    assert d.verify_audit_chain()["valid"]
    d._exec("UPDATE audit SET description='tampered' WHERE id=2")
    assert d.verify_audit_chain() == {"valid": False, "checked": 3, "broken_at": 2}


def test_dashboard_numbers(client, auth):
    d = client.get("/api/dashboard/summary", headers=auth("OPS-2001")).json()
    assert d["total_requests"] == sum(d["status_counts"].values()) and "assistant" in d and len(d["trend"]) == 7
    agent = client.get("/api/dashboard/summary", headers=auth("AG-2001")).json()
    assert "assistant" not in agent and agent["total_requests"] <= d["total_requests"]


def test_feedback_and_catalog(client, auth):
    h = auth("EMP-1001")
    assert client.post("/api/chat/feedback", headers=h, json={"message_id": 1, "rating": "up"}).status_code == 200
    assert client.post("/api/chat/feedback", headers=h, json={"message_id": 1, "rating": "meh"}).status_code == 422
    cat = client.get("/api/catalog", headers=h).json()["items"]
    assert len(cat) >= 40 and all(c["request_type"] not in ("Unclear Request",) for c in cat)


def test_security_headers_and_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.headers["x-content-type-options"] == "nosniff" and "x-request-id" in r.headers
    assert client.get("/ready").json()["status"] == "ready"


def test_backend_serves_the_web_app_and_blocks_traversal(client):
    r = client.get("/")
    assert r.status_code == 200 and ("<div id=\"root\">" in r.text or "Web app not built" in r.text)
    assert client.get("/some/spa/route").status_code == 200            # SPA fallback
    assert client.get("/api/does-not-exist").status_code == 404        # API paths never fall back to HTML
    assert client.get("/..%2f..%2fapp%2fconfig.py").headers.get("content-type", "").startswith(("text/html", "application/json"))
    assert "SECRET" not in client.get("/%2e%2e/app/config.py").text
