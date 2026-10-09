"""Conversation engine: message -> safety -> small-talk -> intent -> grounded answer / guided workflow / human handoff.

Design rules (from the problem statement):
  * answers come only from approved knowledge and always carry citations
  * missing information is asked for, never assumed
  * clinical, confidential, low-confidence or unclear requests are routed to a human team
  * the assistant never approves, rejects or completes anything; it can only *submit* for human review
"""
import re
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from . import config, fields as F, nlu
from .db import Database
from .llm import RAGAssistant
from .ops import Operations
from .text_utils import clean_input, redact

KIND_ESCALATION = {"handoff_offer", "handoff_created", "submitted", "refusal"}


def S(label, message=None, action=None, payload=None, style="secondary"):
    s = {"label": label, "style": style}
    if message is not None:
        s["kind"] = "message"; s["message"] = message
    else:
        s["kind"] = "action"; s["action"] = action; s["payload"] = payload or {}
    return s


def conf_label(c: Optional[float]) -> Optional[str]:
    return None if c is None else ("High" if c >= 0.80 else "Medium" if c >= 0.60 else "Low")


class Ctx:
    def __init__(self, user, conv, raw, text, redactions, tone):
        self.user, self.conv, self.raw, self.text = user, conv, raw, text
        self.redactions, self.tone = redactions, tone
        self.state: Dict[str, Any] = conv.get("state") or {}
        self.trace: List[str] = []
        self.safety: Dict[str, Any] = {"flags": [], "primary": None, "blocked": False}
        self.sources: List[Dict[str, Any]] = []
        self.cls: Optional[Dict[str, Any]] = None
        self.generation: Dict[str, Any] = {"mode": "extractive"}


class Engine:
    def __init__(self, store, db: Database, users, classifier, retriever, ops: Operations, rag: Optional[RAGAssistant] = None):
        self.rag = rag
        self.store, self.db, self.users = store, db, users
        self.classifier, self.retriever, self.ops = classifier, retriever, ops

    # ================================================================== entry point
    def handle(self, user: Dict[str, Any], conversation_id: Optional[str], message: str,
               fields: Optional[Dict[str, Any]] = None, action: Optional[str] = None,
               payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        conv = self.db.get_conversation(conversation_id) if conversation_id else None
        if conversation_id and (not conv or conv["user_id"] != user["user_id"]):
            raise HTTPException(404, "Conversation not found")
        if not conv:
            conv = self.db.create_conversation(user["user_id"])
        raw = clean_input(message, config.MAX_MESSAGE_CHARS)
        payload = payload or {}
        if not raw and not fields and not action:
            raise HTTPException(422, "Please type a message.")
        text, found = redact(raw)
        ctx = Ctx(user, conv, raw, text, found, nlu.analyze_tone(text))
        shown = text or (payload.get("label") or action or "Submitted details")
        title = conv["title"] if conv["title"] != "New conversation" else None
        self.db.add_message(conv["id"], "user", shown[:config.MAX_MESSAGE_CHARS],
                            {"action": action, "redacted": found} if (action or found) else {})
        if title is None:
            self.db.save_conversation_state(conv["id"], ctx.state, title=shown[:48])
        try:
            if action:
                return self._handle_action(ctx, action, payload, fields)
            if fields and ctx.state.get("workflow"):
                return self._workflow_input(ctx, fields, "")
            return self._handle_text(ctx)
        except HTTPException:
            raise
        except Exception as exc:        # fail safely: never leave the user without a path to a human
            self.db.audit(user["user_id"], user["role"], "ENGINE_ERROR", f"{type(exc).__name__}: {exc}"[:300],
                          outcome="error", conversation_id=conv["id"])
            return self._reply(ctx, "error",
                               "Something went wrong on my side, and I don't want to guess. Nothing was submitted. "
                               "Please try again, or hand this to a human team.",
                               suggestions=[S("Send to Operations Support", action="handoff",
                                              payload={"reason": "system_error", "label": "Send to Operations Support"}, style="primary"),
                                            S("Try again", message=text or "help")])

    # ================================================================== text routing
    def _handle_text(self, ctx: Ctx) -> Dict[str, Any]:
        st = ctx.state
        wf = st.get("workflow")
        text = ctx.text
        had_greeting, remainder = nlu.split_greeting(text)
        core = remainder if had_greeting else text

        # 1) safety first - always. Inside a workflow, plain clinical words in a free-text answer are allowed to
        #    pass to the human team (they are not answered by AI); hard boundaries still fire.
        ctx.safety = nlu.screen_safety(text, ctx.redactions)
        prim = ctx.safety["primary"]
        if prim and wf and prim["id"] in ("clinical_advice",):
            prim = None
        if prim:
            return self._safety_response(ctx, prim)

        # 2) in-flight workflow
        if wf:
            return self._workflow_text(ctx, wf, core, had_greeting and not remainder)

        # 3) pending confirmations from earlier turns
        pend = st.get("pending")
        if pend:
            n = nlu
            if pend["type"] == "handoff":
                if n.is_yes(core):
                    return self._create_handoff(ctx, pend)
                if n.is_no(core) or n.is_cancel(core):
                    st["pending"] = None
                    return self._reply(ctx, "smalltalk", "No problem - I haven't passed anything on. Is there anything else I can help with?",
                                       suggestions=self._starter_chips(ctx))
                st["pending"] = None
            elif pend["type"] == "clarify":
                cands = pend["candidates"]
                if n.is_yes(core):
                    return self._start_intent(ctx, cands[0], pend.get("original") or core, forced=True)
                low = core.lower()
                for c in cands:
                    if c.lower() in low:
                        return self._start_intent(ctx, c, pend.get("original") or core, forced=True)
                if n.is_no(core):
                    return self._reply(ctx, "clarify", "Thanks - could you describe what you need in a few words? For example, "
                                       "\"my laptop screen is broken\" or \"I need access to a shared drive\".",
                                       suggestions=[S(c, action="choose_intent", payload={"request_type": c, "label": c}) for c in cands[1:]] +
                                                   [S("Talk to a human", action="handoff", payload={"reason": "low_confidence", "label": "Talk to a human"})])
                st["pending"] = None

        # 4) small talk
        if had_greeting and not remainder:
            return self._smalltalk(ctx, "greeting")
        cat = nlu.smalltalk(core)
        if cat:
            return self._smalltalk(ctx, cat)

        # 5) operational intent
        return self._route_intent(ctx, core or text)

    # ================================================================== actions (buttons)
    def _handle_action(self, ctx: Ctx, action: str, payload: Dict[str, Any], fields: Optional[Dict[str, Any]]):
        st = ctx.state
        wf = st.get("workflow")
        ctx.safety = {"flags": [], "primary": None, "blocked": False}
        if action == "cancel":
            return self._cancel(ctx)
        if action == "decline":
            st["pending"] = None
            return self._reply(ctx, "smalltalk", "No problem - I haven't passed anything on. Anything else I can help with?",
                               suggestions=self._starter_chips(ctx))
        if action == "handoff":
            pend = st.get("pending") or {}
            pend = {**pend, "type": "handoff", "reason": payload.get("reason") or pend.get("reason") or "user_requested"}
            pend.setdefault("summary", ctx.state.get("last_user_text") or "User asked to speak to a human.")
            return self._create_handoff(ctx, pend)
        if action in ("choose_intent", "start_workflow"):
            rt = payload.get("request_type", "")
            if not self.store.request_type(rt):
                raise HTTPException(422, "Unknown request type")
            pend = st.get("pending") or {}
            original = payload.get("original") or (pend.get("original") if pend.get("type") == "clarify" else "")
            st["pending"] = None
            return self._start_intent(ctx, rt, original or f"I need help with: {rt}", forced=True,
                                      force_workflow=(action == "start_workflow"))
        if action == "fields":
            if not wf:
                raise HTTPException(409, "No active request")
            return self._workflow_input(ctx, fields or payload.get("fields") or {}, "")
        if action == "resume":
            if not wf:
                raise HTTPException(409, "No active request")
            return self._ask(ctx, wf, prefix="Okay, let's carry on.")
        if action == "edit":
            if not wf:
                raise HTTPException(409, "No active request")
            fid = payload.get("field")
            if fid not in wf["required"]:
                raise HTTPException(422, "Unknown field")
            wf["stage"], wf["asking"] = "collecting", fid
            return self._ask(ctx, wf, prefix="Sure - let's update that.")
        if action == "submit":
            if not wf or wf["stage"] != "confirming":
                raise HTTPException(409, "Nothing is ready to submit")
            return self._submit(ctx, wf)
        raise HTTPException(422, "Unsupported action")

    # ================================================================== safety responses
    def _safety_response(self, ctx: Ctx, flag: Dict[str, Any]) -> Dict[str, Any]:
        fid, team = flag["id"], flag.get("route_to")
        ctx.trace.append(f"Safety screen: {flag['label']} (rule {flag['rule']}; matched {', '.join(map(str, flag['evidence'][:3]))}).")
        self.db.audit(ctx.user["user_id"], ctx.user["role"], "SAFETY_BLOCK", f"{flag['label']} ({fid}) - AI response withheld.",
                      outcome="blocked", conversation_id=ctx.conv["id"], meta={"flag": fid, "rule": flag["rule"]})
        ctx.state["last_user_text"] = ctx.text
        handoff_chip = lambda lbl, rt, tm: S(lbl, action="handoff", payload={"reason": fid, "request_type": rt, "team": tm, "label": lbl}, style="primary")
        decline = S("No thanks", action="decline", payload={"label": "No thanks"})
        sources = self._sources_for_title({"clinical_advice": "Clinical Request Boundary", "medical_emergency": "Clinical Request Boundary",
                                           "self_harm": "Clinical Request Boundary", "confidential_request": "Patient Information Boundary"}.get(fid))
        if fid == "self_harm":
            ctx.state["pending"] = {"type": "handoff", "request_type": "Clinical Request", "team": "Clinical Support", "reason": fid, "summary": "Employee may need urgent personal support (details withheld)."}
            return self._reply(ctx, "refusal",
                               "I'm really sorry you're going through this. You deserve support from a real person right now. "
                               "**Please contact your local emergency services or a crisis line in your region, or reach out to someone you trust.** "
                               "I'm not able to give medical or counselling guidance, but I can alert Clinical Support so a person follows up.",
                               confidence=0.95, sources=sources, escalation=self._esc(True, "Clinical Support", ["possible personal crisis"]),
                               suggestions=[handoff_chip("Alert Clinical Support", "Clinical Request", "Clinical Support"), decline])
        if fid == "medical_emergency":
            ctx.state["pending"] = {"type": "handoff", "request_type": "Clinical Request", "team": "Clinical Support", "reason": fid, "summary": "Possible medical emergency reported (details withheld)."}
            return self._reply(ctx, "refusal",
                               "**If this is an emergency, contact your local emergency services right away or ask someone nearby for help.** "
                               "I can't provide medical advice or assess symptoms. I can also notify Clinical Support so a person follows up.",
                               confidence=0.95, sources=sources, escalation=self._esc(True, "Clinical Support", ["possible medical emergency"]),
                               suggestions=[handoff_chip("Notify Clinical Support", "Clinical Request", "Clinical Support"), decline])
        if fid == "clinical_advice":
            ctx.state["pending"] = {"type": "handoff", "request_type": "Clinical Request", "team": "Clinical Support", "reason": fid, "summary": ctx.text}
            return self._reply(ctx, "refusal",
                               "I can't give medical advice, diagnosis or medication guidance - that needs a qualified clinician. "
                               "I'm here for operational questions. Would you like me to pass your message to **Clinical Support** so a person can follow up?",
                               confidence=0.95, sources=sources, escalation=self._esc(True, "Clinical Support", ["clinical advice boundary (SAFE-001)"]),
                               suggestions=[handoff_chip("Yes, hand over to Clinical Support", "Clinical Request", "Clinical Support"), decline])
        if fid == "confidential_request":
            ctx.state["pending"] = {"type": "handoff", "request_type": "Sensitive Information", "team": "Privacy Office", "reason": fid, "summary": ctx.text}
            return self._reply(ctx, "refusal",
                               "I can't share or look up confidential records, such as patient data or another person's personal or payroll information. "
                               "If you have an authorised need, the **Privacy Office** can review it. Shall I send them your request?",
                               confidence=0.95, sources=sources, escalation=self._esc(True, "Privacy Office", ["confidential information (SAFE-002)"]),
                               suggestions=[handoff_chip("Yes, send to Privacy Office", "Sensitive Information", "Privacy Office"), decline])
        if fid == "credential_request":
            return self._reply(ctx, "refusal",
                               "I can't give out or look up passwords, codes or login details - and nobody should ask you to share yours in chat. "
                               "I can explain the approved password-reset process or open a ticket with the IT Service Desk.",
                               confidence=0.95, escalation=self._esc(False, "IT Service Desk", []),
                               suggestions=[S("How do I reset my password?", message="How do I reset my password?", style="primary"),
                                            S("Open an IT ticket", action="start_workflow", payload={"request_type": "Account Unlock", "label": "Open an IT ticket"})])
        if fid == "bypass_attempt":
            return self._reply(ctx, "refusal",
                               "I can't bypass approvals, change permissions, or work outside my rules - every request needs human approval, "
                               "and I can only help with your own requests. If something is blocked, tell me what you need and I'll route it to the right team.",
                               confidence=0.95, escalation=self._esc(False, None, []), suggestions=self._starter_chips(ctx))
        if fid == "irreversible_action":
            ctx.state["pending"] = {"type": "handoff", "request_type": "Unclear Request", "team": "Operations Support", "reason": fid, "summary": ctx.text}
            return self._reply(ctx, "refusal",
                               "I can't carry out irreversible actions such as deletions or terminations. They need explicit approval from an authorised person. "
                               "Would you like me to send this to **Operations Support** for human review?",
                               confidence=0.9, escalation=self._esc(True, "Operations Support", ["irreversible action (SAFE-007)"]),
                               suggestions=[handoff_chip("Yes, send for human review", "Unclear Request", "Operations Support"), decline])
        raise ValueError(f"unhandled safety flag {fid}")

    # ================================================================== small talk
    def _smalltalk(self, ctx: Ctx, cat: str) -> Dict[str, Any]:
        first = ctx.user["name"].split()[0]
        ctx.trace.append(f"Recognised small-talk ({cat}); no knowledge lookup needed.")
        chips = self._starter_chips(ctx)
        if cat == "greeting":
            t = (f"Hello {first}! I'm the CareOps Assistant. I can answer operations questions from approved knowledge, "
                 "guide you through routine requests (HR, IT, facilities, procurement, finance, travel) and route anything that needs human judgment.\n\nWhat can I help you with today?")
        elif cat == "thanks":
            t, chips = "You're welcome! Is there anything else I can help with?", chips[:2]
        elif cat == "bye":
            t, chips = f"Goodbye, {first}! You can track your requests any time under **My Requests**.", []
        elif cat == "how_are_you":
            t = "I'm running smoothly, thank you! How can I help with your operations request?"
        elif cat == "identity":
            t = ("I'm the **CareOps Assistant**, an AI helper for healthcare operations staff. I only use approved, synthetic knowledge articles, "
                 "I always show my sources, and I never give medical advice or make final decisions - people review every request.")
        elif cat == "capabilities":
            t = ("Here's what I can do:\n- **Answer** operational questions from approved procedures, with sources\n"
                 "- **Guide** routine requests (ID cards, access, equipment, room booking, maintenance, procurement, travel and more) and ask only for missing details\n"
                 "- **Route** account-specific, clinical, sensitive or unclear requests to the right human team\n\n"
                 "I can't give medical advice, reveal confidential data, bypass approvals, or approve anything myself. Try one of these:")
        elif cat == "rude":
            t = ("I'm sorry this has been frustrating. I'm limited to operational help, but I can connect you with a person right away.")
            chips = [S("Talk to a human", action="handoff", payload={"reason": "frustration", "label": "Talk to a human"}, style="primary")] + chips[:2]
        else:   # out_of_scope
            t = ("That's outside what I can help with - I'm focused on healthcare operations support (HR, IT, facilities, procurement, finance, travel and security). "
                 "Try asking about one of those, for example:")
        return self._reply(ctx, "smalltalk", t, suggestions=chips)

    def _starter_chips(self, ctx: Ctx) -> List[Dict[str, Any]]:
        return [S("Replace my ID card", message="I need a replacement ID card because mine is damaged"),
                S("Book a meeting room", message="I need to book a meeting room"),
                S("Reset my password", message="How do I reset my password?"),
                S("Report a maintenance issue", message="The air conditioner is not working in Building A Floor 2")]

    # ================================================================== intent routing
    def _route_intent(self, ctx: Ctx, msg: str) -> Dict[str, Any]:
        cls = self.classifier.classify(msg)
        ctx.cls = cls
        ctx.state["last_user_text"] = ctx.text
        ev = cls["evidence"]
        ctx.trace.append(f"Intent: {cls['intent']} (confidence {cls['confidence']:.2f}) - keywords {ev['keywords'] or 'none'}; "
                         f"nearest approved example: \"{ev['nearest_example']}\".")
        results = self.retriever.search(msg, boost_title=cls["intent"])
        c = cls["confidence"]
        if c >= config.LOW_CONFIDENCE_THRESHOLD:
            return self._start_intent(ctx, cls["intent"], msg, cls=cls, results=results)
        if c >= 0.35 and self.rag and self.rag.enabled:
            pick = self.rag.choose_intent(msg, [x["request_type"] for x in cls["candidates"]])
            if pick and pick["confidence"] >= 0.80:
                ctx.trace.append(f"LLM tie-break among local candidates chose {pick['request_type']} ({pick['confidence']:.2f}); local confidence was {c:.2f}.")
                return self._start_intent(ctx, pick["request_type"], msg, results=results, via="llm")
        if c >= 0.50:
            cands = [x["request_type"] for x in cls["candidates"]]
            ctx.state["pending"] = {"type": "clarify", "candidates": cands, "original": msg}
            ctx.sources = self._cite(results[:2])
            ctx.trace.append("Confidence below threshold - asking the user to confirm instead of assuming.")
            chips = [S(f"Yes - {cands[0]}", action="choose_intent", payload={"request_type": cands[0], "label": f"Yes - {cands[0]}"}, style="primary")]
            chips += [S(x, action="choose_intent", payload={"request_type": x, "label": x}) for x in cands[1:]]
            chips.append(S("Talk to a human", action="handoff", payload={"reason": "low_confidence", "label": "Talk to a human"}))
            return self._reply(ctx, "confirm_intent", f"It sounds like you need help with **{cands[0]}** - is that right? If not, pick the closest option below or tell me more.",
                               confidence=c, cls=cls, suggestions=chips, escalation=self._esc(False, None, []))
        # low confidence: with an LLM, try a validated grounded answer from retrieved/routed articles
        gen = self._llm_answer(ctx, msg, results[:3], results)
        if gen:
            ctx.sources = self._cite(gen["articles"], primary=gen["articles"][0]["article_id"])
            ctx.state["pending"] = {"type": "handoff", "request_type": "Unclear Request", "team": "Operations Support", "reason": "low_confidence", "summary": ctx.text}
            support = min(1.0, gen["top_raw"] / 0.45) if gen["top_raw"] else 0.6
            ctx.cls = None                      # low local intent confidence: do not display a guessed intent badge
            return self._reply(ctx, "answer", gen["text"] + "\n\nIf this doesn't answer your question, I can pass it to a person.",
                               confidence=round(min(0.9, 0.5 * gen["confidence"] + 0.5 * support), 2),
                               escalation=self._esc(False, "Operations Support", []),
                               suggestions=[S("Send to a human", action="handoff", payload={"reason": "low_confidence", "label": "Send to a human"}, style="primary"),
                                            S("That helped", message="thanks")])
        if results and results[0]["score"] >= 0.30:
            art = results[0]
            ctx.sources = self._cite(results[:3], primary=art["article_id"])
            ctx.trace.append(f"Low intent confidence but strong source match {art['article_id']} ({art['score']:.2f}); answering with a caveat.")
            ctx.state["pending"] = {"type": "handoff", "request_type": "Unclear Request", "team": "Operations Support", "reason": "low_confidence", "summary": ctx.text}
            text = (f"I'm not fully certain this is what you're asking, but the closest approved guidance is:\n\n**{art['title']}**\n{art['content']}\n\n"
                    f"_Source: {art['source_name']} - {art['article_id']}_\n\nIf this doesn't answer your question, I can pass it to a person.")
            return self._reply(ctx, "answer", text, confidence=round(min(0.6, c + 0.1), 2), cls=cls,
                               escalation=self._esc(False, "Operations Support", ["low confidence"]),
                               suggestions=[S("Send to a human", action="handoff", payload={"reason": "low_confidence", "label": "Send to a human"}, style="primary"),
                                            S("That helped", message="thanks")])
        ctx.state["pending"] = {"type": "handoff", "request_type": "Unclear Request", "team": "Operations Support", "reason": "no_source", "summary": ctx.text}
        ctx.trace.append("No approved source or workflow matched (SAFE-006 / SAFE-008): declining to guess.")
        self.db.audit(ctx.user["user_id"], ctx.user["role"], "NO_SOURCE", "No approved source matched; AI declined to answer.",
                      outcome="escalated", conversation_id=ctx.conv["id"])
        return self._reply(ctx, "no_answer",
                           "I couldn't find approved information that answers this, and I don't want to guess. "
                           "You can rephrase it, browse what I can help with, or I can send it to **Operations Support** so a person replies.",
                           confidence=round(c, 2), cls=cls, escalation=self._esc(True, "Operations Support", ["no approved source found"]),
                           suggestions=[S("Send to Operations Support", action="handoff", payload={"reason": "no_source", "label": "Send to Operations Support"}, style="primary"),
                                        S("What can you do?", message="What can you do?")])

    def _start_intent(self, ctx: Ctx, rt: str, msg: str, cls: Optional[Dict[str, Any]] = None,
                      results: Optional[List[Dict[str, Any]]] = None, forced: bool = False, force_workflow: bool = False, via: str = "user"):
        row = self.store.request_type(rt)
        if cls is None:
            row_r = row
            cls = {"intent": rt, "department": row["department"], "workflow_id": row["workflow_id"], "risk_level": row["risk_level"],
                   "handling_mode": row["handling_mode"], "confidence": 0.95 if via == "user" else 0.75, "evidence": {"keywords": [], "nearest_example": "(confirmed by user)", "similarity": 1, "keyword_score": 1},
                   "candidates": [{"request_type": rt, "score": 1.0}]}
            if via == "user":
                ctx.trace.append(f"Intent confirmed by the user: {rt}.")
        ctx.cls = cls
        results = results if results is not None else self.retriever.search(msg, boost_title=rt)
        art = self.store.article_for(rt) or (results[0] if results else None)
        primary_id = art["article_id"] if art else None
        ctx.sources = self._cite(([art] if art else []) + [r for r in results if not art or r["article_id"] != art["article_id"]][:2], primary=primary_id)
        if art:
            ctx.trace.append(f"Grounded in {art['article_id']} - {art['source_name']}.")
        rule = self.store.routing_rule(rt) or {}
        team = rule.get("target_team") or "Operations Support"
        mode = str(row["handling_mode"]).lower()
        info_only = mode.startswith(("information", "grounded", "guidance"))
        info_q = nlu.is_information_question(msg)
        support = 1.0 if art else 0.4
        overall = round(min(0.98, 0.7 * cls["confidence"] + 0.3 * support), 2)
        body = (f"**{art['title']}**\n{art['content']}\n\n_Source: {art['source_name']} - {art['article_id']} (effective {str(art['effective_date'])[:10]})_"
                if art else "I don't have an approved article for this, so I'll route it to a person.")
        empathy = "I'm sorry for the trouble. " if ctx.tone["sentiment"] == "negative" else ""
        risk = str(row["risk_level"]).lower()
        urgent_note = ""
        if risk == "critical" and rt in ("Urgent Facility Issue", "Security Incident"):
            urgent_note = "**If anyone is in danger, move to safety and alert on-site emergency personnel first.**\n\n"

        if not force_workflow and (info_only or info_q):
            gen = self._llm_answer(ctx, msg, ([art] if art else []) + [r for r in results if not art or r["article_id"] != art["article_id"]][:2], results)
            if gen:
                body = gen["text"]
                ctx.sources = self._cite(gen["articles"], primary=gen["articles"][0]["article_id"])
                overall = round(min(0.98, 0.5 * cls["confidence"] + 0.5 * gen["confidence"]), 2)
            ask = ("I can open a request for you with **" + team + "**.") if rule else ""
            ctx.trace.append("Information request - answering from approved source; offering a request if they want one.")
            chips = [S("Start this request", action="start_workflow", payload={"request_type": rt, "label": f"Start a {rt} request"}, style="primary")] if rule else []
            if mode.startswith("information only") or mode.startswith("grounded"):
                body += "\n\nI can share the approved policy, but I can't look up personal records such as individual balances or live account data."
            return self._reply(ctx, "answer", f"{empathy}{body}\n\n{ask}".strip(), confidence=overall, cls=cls, intent_override=rt,
                               escalation=self._esc(False, team, []), suggestions=chips + [S("Ask something else", message="What can you do?")])
        # action request -> guided workflow
        ctx.trace.append("Action request - starting guided workflow; required fields come from Workflows/Routing_Rules.")
        required = self.store.required_fields(rt)
        wf = {"request_type": rt, "department": row["department"], "team": team, "required": required, "fields": {}, "prefilled": [],
              "asking": None, "stage": "collecting", "classification": {k: cls[k] for k in ("intent", "confidence", "risk_level", "handling_mode")},
              "original": msg[:config.MAX_FIELD_CHARS], "tone": {k: ctx.tone[k] for k in ("sentiment", "urgency")},
              "sources": ctx.sources, "confidence": overall}
        for fid in required:
            if fid in F.ID_FIELDS:
                wf["fields"][fid] = ctx.user["user_id"]; wf["prefilled"].append(fid)
        extract_text = "" if (force_workflow or msg.startswith("I need help with:")) else msg
        for fid, val in F.extract_from_message(extract_text, required, ctx.tone).items():
            if fid in wf["fields"]:
                continue
            ok, norm, err = F.validate_field(fid, val, ctx.user, self.store.field_def(fid))
            if ok:
                wf["fields"][fid] = norm; wf["prefilled"].append(fid)
        ctx.state["workflow"], ctx.state["pending"] = wf, None
        self.db.audit(ctx.user["user_id"], ctx.user["role"], "WORKFLOW_STARTED", f"{rt} workflow started.",
                      conversation_id=ctx.conv["id"], meta={"request_type": rt, "prefilled": wf["prefilled"]})
        intro = f"{empathy}{urgent_note}I can help with that - here's the approved guidance:\n\n{body}\n\nI'll only ask for what's missing."
        return self._advance(ctx, wf, prefix=intro, kind="collect")

    # ================================================================== workflow
    def _workflow_text(self, ctx: Ctx, wf: Dict[str, Any], core: str, pure_greeting: bool):
        if nlu.is_cancel(core):
            return self._cancel(ctx)
        cat = "greeting" if pure_greeting else (nlu.smalltalk(core) if len(core.split()) <= 6 else None)
        if cat:
            ctx.trace.append("Small-talk during a workflow - answering and repeating the pending question.")
            return self._ask(ctx, wf, prefix=self._smalltalk_line(ctx, cat))
        if wf["stage"] == "confirming":
            if nlu.is_yes(core):
                return self._submit(ctx, wf)
            labeled = F.extract_labeled(core, wf["required"], {f: self.store.field_def(f) for f in wf["required"]})
            if labeled:
                return self._workflow_input(ctx, labeled, "")
            if nlu.is_no(core) or re.search(r"\b(?:change|edit|modify|wrong|incorrect|update|fix)\b", core.lower()):
                return self._reply(ctx, "confirm", "No problem - which detail would you like to change? Pick one below, or type it like `reason: new text`.",
                                   workflow=self._wf_view(wf), suggestions=self._edit_chips(wf) + [S("Cancel request", action="cancel", payload={"label": "Cancel request"}, style="danger")])
            return self._reply(ctx, "confirm", "Please confirm with **Submit request**, or tell me what to change (for example `department: HR`).",
                               workflow=self._wf_view(wf), suggestions=self._confirm_chips(wf))
        # collecting
        labeled = F.extract_labeled(core, wf["required"], {f: self.store.field_def(f) for f in wf["required"]})
        if labeled:
            return self._workflow_input(ctx, labeled, "")
        if wf["asking"] and nlu.is_information_question(core) and len(core.split()) >= 4 and wf["asking"] not in F.FREE_TEXT_SUMMARY:
            results = self.retriever.search(core)
            if results and results[0]["score"] >= 0.25:
                a = results[0]
                ctx.sources = self._cite(results[:2], primary=a["article_id"])
                return self._ask(ctx, wf, prefix=f"Quick answer from approved knowledge:\n\n**{a['title']}**\n{a['content']}\n\n_Source: {a['source_name']} - {a['article_id']}_\n\nBack to your request:")
        if not wf["asking"]:
            return self._advance(ctx, wf)
        if wf["asking"] not in F.FREE_TEXT_SUMMARY and len(core.split()) >= 5:
            other = self.classifier.classify(core)
            if other["intent"] != wf["request_type"] and other["confidence"] >= 0.90:
                ctx.trace.append(f"Message looks like a different request ({other['intent']}, {other['confidence']:.2f}); asking before switching.")
                st_pend = {"type": "clarify", "candidates": [other["intent"]], "original": core}
                return self._reply(ctx, "confirm_intent", f"That sounds like a different request (**{other['intent']}**). Do you want to switch to it, or carry on with your **{wf['request_type']}** request?",
                                   workflow=self._wf_view(wf), confidence=other["confidence"],
                                   suggestions=[S(f"Switch to {other['intent']}", action="choose_intent", payload={"request_type": other["intent"], "label": f"Switch to {other['intent']}", "original": core}, style="primary"),
                                                S("Continue current request", action="resume", payload={"label": "Continue current request"})])
        return self._workflow_input(ctx, {wf["asking"]: core}, "")

    def _smalltalk_line(self, ctx: Ctx, cat: str) -> str:
        return {"greeting": "Hello! Let's finish your request.", "thanks": "You're welcome!", "bye": "Before you go - your request isn't submitted yet.",
                "rude": "I'm sorry this is frustrating.", "capabilities": "I can guide you step by step and a person reviews every request.",
                "identity": "I'm the CareOps Assistant.", "how_are_you": "Doing well, thanks!", "out_of_scope": "That's outside what I can help with."}.get(cat, "")

    def _workflow_input(self, ctx: Ctx, values: Dict[str, Any], prefix: str):
        wf = ctx.state.get("workflow")
        if not wf:
            raise HTTPException(409, "No active request")
        accepted, errors = [], []
        for fid, val in values.items():
            if fid not in wf["required"]:
                continue
            ok, norm, err = F.validate_field(fid, val, ctx.user, self.store.field_def(fid))
            if ok and fid in ("start_time", "end_time"):      # cross-field check
                other = values.get("start_time" if fid == "end_time" else "end_time") or wf["fields"].get("start_time" if fid == "end_time" else "end_time")
                o_ok, o_norm, _ = F.validate_field("start_time" if fid == "end_time" else "end_time", other, ctx.user, {"label": "time"}) if other else (False, None, None)
                if o_ok and ((fid == "end_time" and norm <= o_norm) or (fid == "start_time" and norm >= o_norm)):
                    ok, err = False, "The end time must be after the start time."
            if ok:
                wf["fields"][fid] = norm
                if fid in wf["prefilled"]:
                    wf["prefilled"].remove(fid)
                accepted.append(self.store.field_def(fid)["label"])
            else:
                errors.append((fid, err))
        if accepted:
            self.db.audit(ctx.user["user_id"], ctx.user["role"], "FIELD_COLLECTED", f"Collected: {', '.join(accepted)}.",
                          conversation_id=ctx.conv["id"], meta={"fields": list(values.keys())})
        pre = []
        if accepted:
            pre.append(f"Got it - saved **{', '.join(accepted)}**.")
        if errors:
            ctx.trace.append("Validation failed: " + "; ".join(f"{f}: {e}" for f, e in errors))
            wf["asking"] = errors[0][0]
            return self._reply(ctx, "collect", "\n".join(pre + [errors[0][1]]) + f"\n\n{self._question(wf['asking'])}",
                               workflow=self._wf_view(wf), suggestions=self._field_chips(wf["asking"]))
        if not accepted and not errors:
            return self._ask(ctx, wf, prefix="I didn't catch a value for the details I need.")
        was_confirming = wf["stage"] == "confirming"
        wf["stage"] = "collecting"
        return self._advance(ctx, wf, prefix=" ".join(pre), kind="collect", editing=was_confirming)

    def _advance(self, ctx: Ctx, wf: Dict[str, Any], prefix: str = "", kind: str = "collect", editing: bool = False):
        missing = [f for f in wf["required"] if f not in wf["fields"]]
        if missing:
            wf["asking"] = missing[0]
            return self._ask(ctx, wf, prefix=prefix, kind=kind)
        wf["asking"], wf["stage"] = None, "confirming"
        ctx.trace.append("All required fields present - asking for explicit confirmation before submitting to a human team.")
        pri, why = self.ops.priority_for(wf["request_type"], wf["tone"])
        rows = "\n".join(f"- **{self.store.field_def(f)['label']}:** {wf['fields'][f]}" for f in wf["required"])
        rule = self.store.routing_rule(wf["request_type"]) or {}
        text = (f"{prefix}\n\n" if prefix else "") + (
            f"Here's your **{wf['request_type']}** request:\n{rows}\n\n"
            f"It will go to **{wf['team']}** (priority **{pri}**, response target {config.SLA_HOURS[pri]}h). "
            f"{rule.get('routing_action', 'A person will review it')}. **Nothing is approved automatically.**\n\nShall I submit it?")
        return self._reply(ctx, "confirm", text.strip(), workflow=self._wf_view(wf), confidence=wf["confidence"],
                           escalation=self._esc(True, wf["team"], ([str(rule["routing_action"])] if rule.get("routing_action") else []) + why),
                           suggestions=self._confirm_chips(wf))

    def _ask(self, ctx: Ctx, wf: Dict[str, Any], prefix: str = "", kind: str = "collect"):
        fid = wf["asking"] or next((f for f in wf["required"] if f not in wf["fields"]), None)
        wf["asking"] = fid
        done = len(wf["fields"])
        lines = ([prefix] if prefix else []) + [self._question(fid)]
        if prefix and done == len(wf["prefilled"]) and "I'll only ask" in prefix:
            lines.append("_Tip: you can answer several at once, like `department: HR, manager: Ravi`, or use the form._")
        return self._reply(ctx, kind, "\n\n".join(lines), workflow=self._wf_view(wf), confidence=wf["confidence"],
                           suggestions=self._field_chips(fid) + [S("Cancel", action="cancel", payload={"label": "Cancel"}, style="danger")])

    def _question(self, fid: str) -> str:
        d = self.store.field_def(fid)
        q = f"What is the **{d['label']}**?" if d["label"] else "Please provide the next detail."
        if d["hint"]:
            q += f" _({d['hint']})_"
        if d["placeholder"]:
            q += "\nFor example: " + d["placeholder"].replace("e.g. ", "")
        return q

    def _field_chips(self, fid: Optional[str]) -> List[Dict[str, Any]]:
        return [S(o, message=o) for o in (self.store.field_def(fid)["options"] if fid else [])][:4]

    def _confirm_chips(self, wf):
        return [S("Submit request", action="submit", payload={"label": "Submit request"}, style="primary"),
                S("Change a detail", message="I want to change something"), S("Cancel", action="cancel", payload={"label": "Cancel"}, style="danger")]

    def _edit_chips(self, wf):
        return [S(self.store.field_def(f)["label"], action="edit", payload={"field": f, "label": f"Change {self.store.field_def(f)['label']}"})
                for f in wf["required"] if f not in wf["prefilled"] or f not in F.ID_FIELDS][:5]

    def _wf_view(self, wf: Dict[str, Any]) -> Dict[str, Any]:
        items = []
        for f in wf["required"]:
            d = self.store.field_def(f)
            has = f in wf["fields"]
            items.append({"id": f, "label": d["label"], "hint": d["hint"], "placeholder": d["placeholder"], "options": d["options"],
                          "value": wf["fields"].get(f), "status": "done" if has else ("asking" if f == wf["asking"] else "missing"),
                          "prefilled": f in wf["prefilled"], "locked": f in F.ID_FIELDS})
        return {"request_type": wf["request_type"], "team": wf["team"], "stage": wf["stage"], "asking": wf["asking"],
                "progress": {"done": len(wf["fields"]), "total": len(wf["required"])}, "fields": items}

    def _cancel(self, ctx: Ctx):
        wf = ctx.state.pop("workflow", None)
        ctx.state["pending"] = None
        if wf:
            self.db.audit(ctx.user["user_id"], ctx.user["role"], "WORKFLOW_CANCELLED", f"{wf['request_type']} workflow cancelled by user.",
                          conversation_id=ctx.conv["id"])
        return self._reply(ctx, "cancelled", "Okay - I've cancelled that. **Nothing was submitted.** What else can I help with?",
                           suggestions=self._starter_chips(ctx))

    def _submit(self, ctx: Ctx, wf: Dict[str, Any]):
        missing = [f for f in wf["required"] if f not in wf["fields"]]
        if missing:
            return self._advance(ctx, wf)
        rec = self.ops.create_request(user=ctx.user, request_type=wf["request_type"], fields=wf["fields"], request_text=wf["original"],
                                      tone=wf["tone"], confidence=wf["classification"]["confidence"], sources=wf["sources"],
                                      conversation_id=ctx.conv["id"],
                                      extra_reasons=[f"AI confidence {wf['classification']['confidence']:.2f}"] +
                                                    (["risk: " + str(wf["classification"]["risk_level"])] if str(wf["classification"]["risk_level"]).lower() in ("high", "critical", "sensitive") else []))
        ctx.state.pop("workflow", None)
        ctx.state["pending"] = None
        ctx.trace.append(f"Submitted as {rec['request_id']} to {rec['assigned_team']} - awaiting human review.")
        text = (f"Your request has been submitted. **Reference: {rec['request_id']}**\n\n"
                f"- **Team:** {rec['assigned_team']}\n- **Priority:** {rec['priority']}\n- **Response target:** {config.SLA_HOURS[rec['priority']]}h\n- **Status:** Pending human review\n\n"
                "A person will review it - I can't approve or complete it myself. You'll see updates under **My Requests**.")
        return self._reply(ctx, "submitted", text, request=self._req_brief(rec), confidence=wf["confidence"], sources=wf["sources"],
                           intent_override=wf["request_type"],
                           escalation=self._esc(True, rec["assigned_team"], [rec["escalation_reason"]]),
                           suggestions=[S("View my requests", action="navigate", payload={"page": "requests", "label": "View my requests"}, style="primary"),
                                        S("Start another request", message="help")])

    # ================================================================== handoff
    def _create_handoff(self, ctx: Ctx, pend: Dict[str, Any]):
        rt = pend.get("request_type") or "Unclear Request"
        reason = pend.get("reason") or "user_requested"
        team = pend.get("team") or (self.store.routing_rule(rt) or {}).get("target_team") or "Operations Support"
        summary = redact(str(pend.get("summary") or ctx.state.get("last_user_text") or "User asked to speak to a human."))[0][:config.MAX_FIELD_CHARS]
        fields = {"requester_id": ctx.user["user_id"], "request_summary": summary}
        label = {"clinical_advice": "clinical advice boundary (SAFE-001)", "medical_emergency": "possible emergency", "self_harm": "possible personal crisis",
                 "confidential_request": "confidential information (SAFE-002)", "irreversible_action": "irreversible action (SAFE-007)",
                 "low_confidence": "low AI confidence (SAFE-004)", "no_source": "no approved source (SAFE-006/008)", "frustration": "user frustration",
                 "system_error": "system error", "user_requested": "user asked for a human"}.get(reason, reason)
        tone = {**ctx.tone, "urgency": "high" if reason in ("medical_emergency", "self_harm") else ctx.tone["urgency"]}
        rec = self.ops.create_request(user=ctx.user, request_type=rt, fields=fields, request_text=summary, tone=tone, confidence=None,
                                      sources=[], conversation_id=ctx.conv["id"], extra_reasons=[f"AI handoff: {label}"], team_override=team,
                                      risk_level="critical" if reason in ("clinical_advice", "medical_emergency", "self_harm", "confidential_request") else None)
        ctx.state["pending"] = None
        ctx.trace.append(f"Human handoff created: {rec['request_id']} -> {team} ({label}).")
        text = (f"Done - I've passed this to **{team}**. **Reference: {rec['request_id']}**\n\nA person will follow up; I haven't made any decision on your behalf.")
        return self._reply(ctx, "handoff_created", text, request=self._req_brief(rec), escalation=self._esc(True, team, [label]),
                           suggestions=[S("View my requests", action="navigate", payload={"page": "requests", "label": "View my requests"}, style="primary")])

    # ================================================================== helpers
    def _cite(self, arts: List[Dict[str, Any]], primary: Optional[str] = None) -> List[Dict[str, Any]]:
        out, seen = [], set()
        for a in arts:
            if not a or a["article_id"] in seen:
                continue
            seen.add(a["article_id"])
            full = self.retriever.get(a["article_id"]) or a
            out.append({"article_id": a["article_id"], "title": full["title"], "source_name": full["source_name"],
                        "effective_date": str(full["effective_date"])[:10], "department": full["department"],
                        "score": a.get("score"), "snippet": full["content"][:220], "primary": a["article_id"] == primary})
        return out

    def _sources_for_title(self, title: Optional[str]):
        art = self.store.article_for(title) if title else None
        return self._cite([art], primary=art["article_id"]) if art else []

    @staticmethod
    def _esc(required: bool, team: Optional[str], reasons: List[str]) -> Dict[str, Any]:
        return {"required": required, "team": team, "reasons": [r for r in reasons if r]}

    @staticmethod
    def _req_brief(rec):
        return {k: rec[k] for k in ("request_id", "request_type", "status", "assigned_team", "priority", "sla_due")}

    # ------------------------------------------------------------------ RAG / LLM helpers
    def _history(self, ctx: Ctx, n: int = 4) -> List[str]:
        msgs = self.db.get_messages(ctx.conv["id"], 60)[-n - 1:-1]       # excludes the message just stored
        return [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}" for m in msgs]

    def _llm_answer(self, ctx: Ctx, msg: str, articles: List[Dict[str, Any]], results: Optional[List[Dict[str, Any]]] = None):
        """Try to produce a grounded, validated LLM answer. Returns dict or None (caller falls back to extractive)."""
        if not self.rag or not self.rag.enabled:
            return None
        pool = [a for a in articles if a]
        top_raw = max([r["score"] for r in (results or [])] or [0])
        if top_raw < 0.2:       # vague question: let the LLM pick candidate articles from the catalogue (IDs are validated)
            for aid in self.rag.select_articles(msg, self.retriever.catalog()):
                a = self.retriever.get(aid)
                if a and all(a["article_id"] != p["article_id"] for p in pool):
                    pool.append({**a, "score": 0.0})
            if pool:
                ctx.trace.append(f"Semantic routing selected {', '.join(p['article_id'] for p in pool[:3])} for a vague question.")
        pool = pool[:4]
        gen = self.rag.answer(msg, pool, self._history(ctx))
        model = self.rag.client.model
        if not gen:
            ctx.trace.append("LLM answer unavailable, not answerable from sources, or failed validation - using extractive answer.")
            self.db.audit(ctx.user["user_id"], ctx.user["role"], "LLM_FALLBACK", "LLM output rejected or unavailable; extractive path used.",
                          outcome="fallback", conversation_id=ctx.conv["id"])
            return None
        cited = [a for a in pool if a["article_id"] in gen["citations"]]
        ctx.generation = {"mode": "llm", "model": model, "validated": True}
        ctx.trace.append(f"Answer generated by {model} from {', '.join(gen['citations'])}; citations, numbers and safety re-validated before display.")
        self.db.audit(ctx.user["user_id"], ctx.user["role"], "LLM_ANSWER", f"Grounded answer generated from {', '.join(gen['citations'])}.",
                      conversation_id=ctx.conv["id"], meta={"model": model, "citations": gen["citations"]})
        text = gen["answer"] + "\n\n_Sources: " + "; ".join(f"{a['source_name']} - {a['article_id']}" for a in cited) + "_"
        return {"text": text, "confidence": gen["confidence"], "articles": cited, "top_raw": top_raw}

    # ------------------------------------------------------------------ final assembly
    def _reply(self, ctx: Ctx, kind: str, text: str, *, confidence: Optional[float] = None, cls: Optional[Dict[str, Any]] = None,
               sources: Optional[List[Dict[str, Any]]] = None, workflow=None, suggestions=None, escalation=None, request=None,
               intent_override: Optional[str] = None) -> Dict[str, Any]:
        cls = cls or ctx.cls
        if ctx.redactions:
            text = ("**Security notice:** I removed a password, code or sensitive identifier from your message. "
                    "Never share these in chat - if it was real, change it now.\n\n") + text
        wf_active = ctx.state.get("workflow")
        sources = sources if sources is not None else ctx.sources
        if not sources and wf_active:
            sources = wf_active.get("sources", [])
        if not cls and wf_active:
            cls = {**wf_active["classification"], "department": wf_active["department"], "candidates": []}
        intent = intent_override or (cls["intent"] if cls else None)
        if kind in ("no_answer", "confirm_intent") or (kind in ("smalltalk", "cancelled", "error") and not ctx.cls):
            intent = None
        flags = [f["id"] for f in ctx.safety["flags"]]
        out = {
            "conversation_id": ctx.conv["id"], "kind": kind, "reply": text, "intent": intent,
            "department": cls.get("department") if cls and intent else None, "risk_level": cls.get("risk_level") if cls and intent else None,
            "confidence": confidence, "confidence_label": conf_label(confidence), "sentiment": ctx.tone["sentiment"],
            "urgency": ctx.tone["urgency"], "sources": sources, "workflow": workflow, "request": request,
            "suggestions": suggestions or [], "escalation": escalation or self._esc(False, None, []),
            "safety": {"flags": ctx.safety["flags"], "blocked": ctx.safety["blocked"]}, "trace": ctx.trace,
            "candidates": cls["candidates"] if cls and kind in ("confirm_intent", "no_answer") else None,
            "generation": ctx.generation if sources else None,
        }
        mid = self.db.add_message(ctx.conv["id"], "assistant", text, {k: v for k, v in out.items() if k not in ("reply", "conversation_id")})
        out["message_id"] = mid
        self.db.save_conversation_state(ctx.conv["id"], ctx.state)
        escalated = kind in KIND_ESCALATION or bool(out["escalation"]["required"])
        self.db.log_interaction(ctx.user["user_id"], ctx.conv["id"], intent, kind, confidence, escalated,
                                ctx.tone["sentiment"], ctx.tone["urgency"], flags)
        self.db.audit(ctx.user["user_id"], ctx.user["role"], "CHAT_TURN",
                      f"{kind}" + (f" | intent={intent}" if intent else "") + (f" | confidence={confidence}" if confidence is not None else ""),
                      conversation_id=ctx.conv["id"], request_id=(request or {}).get("request_id"),
                      meta={"kind": kind, "intent": intent, "confidence": confidence, "sources": [s["article_id"] for s in sources],
                            "flags": flags, "sentiment": ctx.tone["sentiment"], "urgency": ctx.tone["urgency"]})
        return out

