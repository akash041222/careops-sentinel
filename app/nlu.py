"""Language understanding: small-talk, safety screening, sentiment/urgency and hybrid intent classification.

Everything here is deterministic and runs locally (no API keys). Each function returns *evidence*
(matched phrases, nearest example, scores) so decisions can be explained in the UI and audit trail.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text_utils import has_phrase, normalize

NON_INTENT_TYPES = {"unclear request", "low confidence"}

# =============================================================================== small talk
_GREETING = re.compile(r"^(?:hi+|hello+|hey+|heya|hiya|yo|howdy|namaste|vanakkam|greetings|good (?:morning|afternoon|evening|day)|sup)\b[\s,!.:-]*")
_THANKS = re.compile(r"^(?:thanks|thank you|thank u|thx|ty|cheers|much appreciated|appreciate it|great thanks|ok thanks|okay thanks)\b")
_BYE = re.compile(r"^(?:bye|goodbye|see you|see ya|cya|good night|that'?s all|that is all|nothing else|i'?m done|all done|no more)\b")
_HOW_ARE_YOU = re.compile(r"\bhow are (?:you|u)\b|\bhow'?s it going\b|\bwhat'?s up\b|\bwhats up\b|\bhow do you do\b")
_IDENTITY = re.compile(r"\bwho are you\b|\bwhat are you\b|\byour name\b|\bare you (?:a |an )?(?:bot|human|ai|real|person|robot)\b")
_CAPABILITY = re.compile(r"\bwhat can you do\b|\bhow can you help\b|\bhow do you work\b|\bwhat do you do\b|^help$|^help me$|^menu$|^options$|"
                         r"\bwhat (?:all )?can i ask\b|\bwhat (?:are|r) your (?:capabilities|features)\b|\bhow (?:to|do i) use (?:this|you)\b|\bcan you help me\b$")
_RUDE = re.compile(r"\b(?:stupid|idiot|useless|dumb|worthless|shut up)\b")
_OUT_OF_SCOPE = re.compile(r"\b(?:weather|forecast|joke|jokes|riddle|news|cricket|football|ipl|movie|movies|song|songs|recipe|"
                           r"stock price|bitcoin|crypto|horoscope|lottery|capital of|prime minister|president of|translate|poem|story|"
                           r"homework|essay|write (?:me )?(?:a )?(?:code|program|script))\b")
_YES = re.compile(r"^(?:yes|yep|yeah|yup|y|sure|ok|okay|confirm|confirmed|correct|right|go ahead|proceed|submit|do it|please do|sounds good|that'?s right|looks good)\b")
_NO = re.compile(r"^(?:no|nope|nah|n|not now|don'?t|do not|stop|negative|no thanks|no thank you)\b")
_CANCEL = re.compile(r"\b(?:cancel|abort|never ?mind|forget it|start over|restart|reset|discard|stop this|drop it)\b")


_GREETING_FILLER = {"there", "team", "assistant", "bot", "careops", "sentinel", "everyone", "all", "buddy", "friend", "sir", "madam",
                    "mam", "folks", "dear", "again", "ops", "support", "how", "are", "you", "u", "doing", "today", "good", "day"}


def split_greeting(text: str) -> Tuple[bool, str]:
    """Returns (had_greeting, remainder_text)."""
    norm = normalize(text)
    m = _GREETING.match(norm)
    if not m:
        return False, text
    rest = norm[m.end():].strip()
    if rest and all(w in _GREETING_FILLER for w in rest.split()):
        rest = ""
    if rest:
        # keep original casing of the remainder where possible
        idx = len(text) - len(text.lstrip())
        gm = re.match(r"(?i)^\s*(?:hi+|hello+|hey+|heya|hiya|yo|howdy|namaste|vanakkam|greetings|good (?:morning|afternoon|evening|day)|sup)\b[\s,!.:-]*", text)
        return True, (text[gm.end():].strip() if gm else rest)
    return True, ""


def smalltalk(text: str) -> Optional[str]:
    """Return a small-talk category or None. Only fires on short, non-operational messages."""
    norm = normalize(text)
    if not norm:
        return None
    words = norm.split()
    if _THANKS.match(norm) and len(words) <= 6:
        return "thanks"
    if _BYE.match(norm) and len(words) <= 6:
        return "bye"
    if _HOW_ARE_YOU.search(norm) and len(words) <= 7:
        return "how_are_you"
    if _IDENTITY.search(norm) and len(words) <= 8:
        return "identity"
    if _CAPABILITY.search(norm) and len(words) <= 9:
        return "capabilities"
    if _RUDE.search(norm):
        return "rude"
    if _OUT_OF_SCOPE.search(norm):
        return "out_of_scope"
    return None


def is_yes(text: str) -> bool:
    return bool(_YES.match(normalize(text)))


def is_no(text: str) -> bool:
    return bool(_NO.match(normalize(text)))


def is_cancel(text: str) -> bool:
    n = normalize(text)
    return bool(_CANCEL.search(n)) and len(n.split()) <= 6


# =============================================================================== sentiment / urgency
_NEG = ["frustrated", "frustrating", "angry", "annoyed", "annoying", "terrible", "awful", "worst", "unacceptable", "useless",
        "ridiculous", "disappointed", "disappointing", "fed up", "still waiting", "again and again", "nobody", "no one is",
        "horrible", "upset", "complaint", "complain", "escalate", "pathetic", "sick of",
        "unhappy", "ignored", "poor service", "hopeless", "stressed", "worried"]
_POS = ["thanks", "thank you", "great", "awesome", "perfect", "appreciate", "excellent", "helpful", "wonderful", "love", "good job"]
_URGENT_HIGH = ["urgent", "urgently", "asap", "immediately", "right now", "emergency", "critical", "as soon as possible",
                "deadline", "blocked", "cannot work", "can't work", "unable to work", "stuck", "production down", "today itself",
                "top priority", "very important", "time sensitive", "at the earliest"]
_URGENT_LOW = ["no rush", "whenever", "when you can", "not urgent", "no hurry", "low priority", "at your convenience", "sometime"]


def analyze_tone(text: str) -> Dict[str, Any]:
    n = normalize(text)
    neg = [w for w in _NEG if has_phrase(n, w)]
    pos = [w for w in _POS if has_phrase(n, w)]
    score = len(pos) * 0.4 - len(neg) * 0.45
    if text.count("!") >= 3 or (len(text) > 12 and text.isupper()):
        score -= 0.4
    score = max(-1.0, min(1.0, score))
    sentiment = "negative" if score <= -0.35 else ("positive" if score >= 0.35 else "neutral")
    hi = [w for w in _URGENT_HIGH if has_phrase(n, w)]
    lo = [w for w in _URGENT_LOW if has_phrase(n, w)]
    urgency = "high" if hi else ("low" if lo else "normal")
    return {"sentiment": sentiment, "sentiment_score": round(score, 2), "urgency": urgency,
            "signals": {"negative": neg, "positive": pos, "urgent": hi, "relaxed": lo}}


# =============================================================================== safety screening
def _any(norm: str, phrases: List[str]) -> List[str]:
    return [p for p in phrases if has_phrase(norm, p)]


_CLINICAL = ["medicine", "medication", "tablet", "pill", "dose", "dosage", "drug", "prescription", "prescribe", "diagnose",
             "diagnosis", "symptom", "treatment", "cure", "side effect", "fever", "cough", "headache", "migraine", "back pain",
             "chest pain", "stomach pain", "pain", "ache", "hurts", "hurt", "bleeding", "vomit", "vomiting", "nausea", "dizzy",
             "infection", "allergic", "allergy", "diabetes", "blood pressure", "lab result", "test result", "x-ray", "scan result",
             "pregnant", "pregnancy", "anxiety", "depression", "mental health", "feel sick", "unwell", "should i take",
             "what should i take", "injured", "injury", "swollen", "rash", "covid", "flu", "cancer", "surgery"]
_EMERGENCY_MED = ["chest pain", "cannot breathe", "can't breathe", "cant breathe", "difficulty breathing", "unconscious", "heart attack",
                  "stroke", "overdose", "severe bleeding", "seizure", "not breathing", "collapsed"]
_SELF_HARM = ["suicide", "suicidal", "kill myself", "want to die", "end my life", "self harm", "hurt myself", "harm myself"]
_CONFIDENTIAL_RE = re.compile(
    r"\b(?:patient(?:'s|s)? (?:record|records|data|information|info|file|files|name|names|list|details|history|chart|results?)|"
    r"medical records?|health records?|(?:his|her|their|someone'?s|colleague'?s|manager'?s|employee'?s|[a-z]+'s) "
    r"(?:salary|pay ?slip|payroll|record|records|address|phone number|personal (?:data|details|info)|performance review|appraisal)|"
    r"(?:list|names|details) of (?:all )?(?:the )?(?:employees|patients|users|members|staff)|all (?:employees|patients|users|staff)'? "
    r"(?:data|records|salaries|details)|vendor (?:contract|pricing|bank)|confidential (?:data|files?|documents?|records?))\b")
_BYPASS_RE = re.compile(
    r"\b(?:ignore (?:all |any |your |the |previous |above |prior )*(?:instructions?|rules?|guidelines?|polic(?:y|ies))|disregard (?:all |your |the |previous )*(?:instructions?|rules?)|"
    r"(?:reveal|show|print|repeat|tell me) (?:me )?(?:your |the )?(?:system )?(?:prompt|instructions)|you are now|developer mode|jailbreak|dan mode|"
    r"act as (?:an? |the )?(?:admin|administrator|root|manager|support agent|hr|dba)|pretend (?:to be|you are|you're)|"
    r"bypass (?:the )?(?:approval|verification|security|access|login|authentication|checks?|controls?)|"
    r"skip (?:the )?(?:approval|verification|manager|human|review|authentication|checks?)|override (?:the )?(?:approval|security|access|policy|controls?)|"
    r"without (?:any )?(?:approval|human review|verification|authorization)|(?:grant|make|give) (?:me|myself) (?:admin|root|full|superuser)|"
    r"approve (?:it |this |my own |my |the )?(?:request|myself|own)|auto[- ]?approve|sudo|(?:delete|erase|clear|wipe) (?:the |all )?(?:audit|logs?)|"
    r"(?:see|show|view|access|open) (?:all|everyone'?s|other (?:people'?s|users'?|employees'?)) (?:requests|tickets|conversations|chats))\b")
_IRREVERSIBLE_RE = re.compile(
    r"\b(?:delete (?:all|everything|my account|the account|all records|the records|the database)|wipe|permanently (?:delete|remove|close)|"
    r"terminate (?:the |an |my )?(?:employee|contract|account)|fire (?:him|her|them|the employee)|remove (?:all )?access (?:for|of) (?:everyone|all)|"
    r"shut ?down (?:the )?(?:server|system|network)|drop (?:the )?(?:table|database))\b")
_CRED_REQUEST_RE = re.compile(r"\b(?:give|tell|show|share|send|reveal|what(?:'s| is)) (?:me )?(?:\w+'s |the |his |her |their |someone'?s )?(?:password|passcode|otp|pin|login)\b")


def screen_safety(text: str, redaction_labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """Return {'flags': [...], 'primary': flag|None, 'blocked': bool}. Highest severity first."""
    n = normalize(text)
    flags: List[Dict[str, Any]] = []

    sh = _any(n, _SELF_HARM)
    if sh:
        flags.append({"id": "self_harm", "severity": "emergency", "rule": "SAFE-001", "route_to": "Clinical Support",
                      "evidence": sh, "label": "Possible personal crisis"})
    em = _any(n, _EMERGENCY_MED)
    if em:
        flags.append({"id": "medical_emergency", "severity": "emergency", "rule": "SAFE-001", "route_to": "Clinical Support",
                      "evidence": em, "label": "Possible medical emergency"})
    if _BYPASS_RE.search(n):
        flags.append({"id": "bypass_attempt", "severity": "high", "rule": "SAFE-003", "route_to": "Security Operations",
                      "evidence": [m.group(0) for m in [_BYPASS_RE.search(n)]], "label": "Access-control / instruction bypass attempt"})
    if _CRED_REQUEST_RE.search(n):
        flags.append({"id": "credential_request", "severity": "high", "rule": "SAFE-002", "route_to": "IT Service Desk",
                      "evidence": [_CRED_REQUEST_RE.search(n).group(0)], "label": "Request for credentials"})
    conf = _CONFIDENTIAL_RE.search(n)
    if conf:
        flags.append({"id": "confidential_request", "severity": "critical", "rule": "SAFE-002", "route_to": "Privacy Office",
                      "evidence": [conf.group(0)], "label": "Confidential information request"})
    cl = _any(n, _CLINICAL)
    if cl and not em and not sh:
        flags.append({"id": "clinical_advice", "severity": "critical", "rule": "SAFE-001", "route_to": "Clinical Support",
                      "evidence": cl[:4], "label": "Clinical / medical advice boundary"})
    irr = _IRREVERSIBLE_RE.search(n)
    if irr:
        flags.append({"id": "irreversible_action", "severity": "high", "rule": "SAFE-007", "route_to": "Operations Support",
                      "evidence": [irr.group(0)], "label": "Irreversible action requires human approval"})
    if redaction_labels:
        flags.append({"id": "secret_shared", "severity": "medium", "rule": "SAFE-002", "route_to": None,
                      "evidence": redaction_labels, "label": "Credential or identifier detected and masked"})

    order = {"emergency": 0, "critical": 1, "high": 2, "medium": 3}
    flags.sort(key=lambda f: order[f["severity"]])
    blocking = [f for f in flags if f["id"] != "secret_shared"]
    return {"flags": flags, "primary": blocking[0] if blocking else None, "blocked": bool(blocking),
            "secret_shared": bool(redaction_labels)}


# =============================================================================== intent classification
class IntentClassifier:
    """Hybrid classifier = nearest labelled example (TF-IDF word + char n-grams) + whole-word keyword evidence.

    Returns the top candidates with a calibrated confidence so the orchestrator can answer, clarify or escalate.
    """

    W_SIM, W_KW = 0.62, 0.38

    def __init__(self, store):
        self.store = store
        self.rebuild()

    def rebuild(self):
        rtypes = [r for r in self.store.records("Request_Types") if r["request_type"].lower() not in NON_INTENT_TYPES]
        self.types = {r["request_type"]: r for r in rtypes}
        self.keywords = {r["request_type"]: [k.strip() for k in str(r["keywords"]).split(",") if k.strip()] for r in rtypes}

        texts, labels = [], []
        seen = set()
        for r in self.store.records("Request_Examples"):
            t, rt = str(r["request_text"]).strip(), r["request_type"]
            if rt in self.types and (t.lower(), rt) not in seen:
                seen.add((t.lower(), rt)); texts.append(t); labels.append(rt)
        for rt, row in self.types.items():   # every type has at least its own name + keywords as pseudo-examples
            for t in (rt, f"{rt} {' '.join(self.keywords[rt])}"):
                texts.append(t); labels.append(rt)
        self.example_texts, self.example_labels = texts, np.array(labels)
        self.w_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.c_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.w_mat = self.w_vec.fit_transform(texts)
        self.c_mat = self.c_vec.fit_transform(texts)

    # --------------------------------------------------------------------------
    def _keyword_scores(self, norm: str) -> Dict[str, Tuple[float, List[str]]]:
        out = {}
        for rt, kws in self.keywords.items():
            hits, w = [], 0.0
            for k in kws:
                if has_phrase(norm, k):
                    hits.append(k)
                    w += 1.0 if " " in k.strip() else (0.55 if len(k) <= 3 else 0.7)
            if has_phrase(norm, rt):
                hits.append(rt); w += 0.9
            out[rt] = (min(1.0, w / 1.25), hits)
        return out

    def classify(self, text: str, exclude_exact: bool = False) -> Dict[str, Any]:
        norm = normalize(text)
        sims = 0.5 * cosine_similarity(self.w_vec.transform([text]), self.w_mat)[0] + \
               0.5 * cosine_similarity(self.c_vec.transform([text]), self.c_mat)[0]
        if exclude_exact:
            for i, t in enumerate(self.example_texts):
                if normalize(t) == norm:
                    sims[i] = 0.0
        kw = self._keyword_scores(norm)
        scored = []
        for rt in self.types:
            mask = self.example_labels == rt
            best_idx = int(np.argmax(np.where(mask, sims, -1)))
            best_sim = float(sims[best_idx])
            kw_s, hits = kw[rt]
            combined = self.W_SIM * min(1.0, best_sim * 1.25) + self.W_KW * kw_s
            scored.append({"request_type": rt, "score": combined, "similarity": round(best_sim, 3),
                           "keyword_score": round(kw_s, 3), "keywords": hits,
                           "nearest_example": self.example_texts[best_idx]})
        scored.sort(key=lambda s: s["score"], reverse=True)
        top, second = scored[0], scored[1]
        margin = top["score"] - second["score"]
        conf = top["score"] * 1.05 + min(0.12, margin * 0.6)
        if top["keyword_score"] == 0 and top["similarity"] < 0.45:
            conf *= 0.8                         # no lexical evidence and weak semantic match
        conf = float(max(0.05, min(0.98, conf)))
        row = self.types[top["request_type"]]
        return {
            "intent": top["request_type"], "department": row["department"], "workflow_id": row["workflow_id"],
            "risk_level": row["risk_level"], "handling_mode": row["handling_mode"],
            "confidence": round(conf, 2), "margin": round(margin, 3),
            "evidence": {k: top[k] for k in ("similarity", "keyword_score", "keywords", "nearest_example")},
            "candidates": [{"request_type": s["request_type"], "score": round(s["score"], 3)} for s in scored[:3]],
        }


_QUESTION_START = re.compile(r"^(?:how|what|where|when|who|which|why|can|could|do|does|is|are|will|should|may|am|whom)\b")


def is_information_question(text: str) -> bool:
    n = normalize(text)
    return text.strip().endswith("?") or bool(_QUESTION_START.match(n)) or has_phrase(n, "tell me about") or has_phrase(n, "explain")
