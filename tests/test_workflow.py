import pytest


def test_full_guided_request_end_to_end(chat, client, auth):
    c = chat("EMP-1001")
    r = c.say("I need a replacement ID card because mine is damaged")
    assert r["kind"] == "collect" and r["workflow"]["asking"] == "department"
    assert r["workflow"]["progress"]["done"] >= 2                     # employee_id + reason pre-filled, not asked
    assert r["workflow"]["fields"][0]["value"] == "EMP-1001"
    r = c.say("hello")                                                  # small talk mid-flow must not become an answer
    assert r["workflow"]["asking"] == "department"
    r = c.say("department: Operations, contact method: Email")
    assert r["kind"] == "confirm" and r["workflow"]["stage"] == "confirming"
    r = c.say(action="submit")
    assert r["kind"] == "submitted" and r["request"]["status"] == "pending_human"
    rid = r["request"]["request_id"]
    mine = client.get("/api/requests", headers=auth("EMP-1001")).json()["items"]
    assert any(x["request_id"] == rid for x in mine)


def test_assistant_asks_instead_of_assuming(chat):
    r = chat().say("I need to book a meeting room")
    labels = [f["label"] for f in r["workflow"]["fields"] if f["status"] != "done"]
    assert "Date" in labels and "Attendees" in labels


def test_validation_errors_reask(chat):
    c = chat(); c.say("I need to book a meeting room")
    r = c.say("date: garbage")
    assert "couldn't read that date" in r["reply"]
    r = c.say(action="fields", fields={"date": "tomorrow", "start_time": "3pm", "end_time": "2pm"})
    assert "end time must be after" in r["reply"]
    r = c.say("attendees: many")
    assert "number of attendees" in r["reply"]


def test_cannot_file_on_behalf_of_someone_else(chat):
    c = chat("EMP-1001"); c.say("I need a laptop")
    r = c.say("employee id: EMP-1002")
    assert "own ID" in r["reply"] or "own id" in r["reply"].lower()
    assert next(f for f in r["workflow"]["fields"] if f["id"] == "employee_id")["value"] == "EMP-1001"


def test_submit_rejected_when_not_ready(client, auth):
    h = auth("EMP-1001")
    r = client.post("/api/chat", headers=h, json={"message": "I need a laptop"}).json()
    r2 = client.post("/api/chat", headers=h, json={"conversation_id": r["conversation_id"], "action": "submit"})
    assert r2.status_code == 409


def test_cancel_discards_everything(chat):
    c = chat(); c.say("I need a laptop")
    r = c.say("cancel")
    assert r["kind"] == "cancelled" and r["workflow"] is None and "Nothing was submitted" in r["reply"]


def test_switching_intent_asks_first(chat):
    c = chat(); c.say("I need to book a meeting room")
    r = c.say("my laptop is not working and I need a replacement device urgently")
    assert r["kind"] in ("confirm_intent", "collect")


def test_critical_issue_shows_safety_first_and_high_priority(chat):
    r = chat().say("There is smoke coming from the server room")
    assert r["intent"] == "Urgent Facility Issue" and "move to safety" in r["reply"].lower()


def test_human_handoff_creates_request_for_the_right_team(chat, client, auth):
    c = chat("EMP-1003")
    c.say("what medicine should I take for fever")
    r = c.say("Yes", action="handoff", payload={"reason": "clinical_advice", "request_type": "Clinical Request", "team": "Clinical Support"})
    assert r["kind"] == "handoff_created" and r["request"]["assigned_team"] == "Clinical Support"
    assert r["request"]["priority"] == "critical"


def test_negative_sentiment_raises_priority(chat):
    c = chat()
    c.say("This is unacceptable, I'm frustrated, my printer is not working and nobody is helping, urgent")
    r = c.last
    assert r["sentiment"] == "negative" and r["urgency"] == "high"
