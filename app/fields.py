"""Workflow field extraction + validation. Never invents values: only what the user literally provided."""
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from dateutil import parser as dateparser

from . import config
from .text_utils import normalize, redact

DATE_FIELDS = {"date", "joining_date", "effective_date", "move_date", "event_date", "visit_date", "required_by"}
TIME_FIELDS = {"start_time", "end_time", "arrival_time", "issue_time", "incident_time", "approximate_time"}
ID_FIELDS = {"employee_id", "requester_id"}
FREE_TEXT_SUMMARY = {"request_summary", "issue_description", "query_summary", "description", "feedback_text",
                     "vendor_query", "query"}
BECAUSE_FIELDS = {"reason", "business_justification", "business_reason", "return_reason", "intended_use"}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_ID_RE = re.compile(r"^[A-Z]{2,4}-\d{3,6}$")


def _today() -> date:
    return datetime.now().date()


def parse_date(text: str) -> Optional[str]:
    iso = re.match(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$", str(text))
    if iso:
        try:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3))).isoformat()
        except ValueError:
            return None
    t = normalize(text).replace("'", "")
    t = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", t)
    today = _today()
    if t in ("today", "now"):
        return today.isoformat()
    if t == "tomorrow":
        return (today + timedelta(days=1)).isoformat()
    if t in ("day after tomorrow", "day after"):
        return (today + timedelta(days=2)).isoformat()
    m = re.match(r"^in (\d{1,3}) days?$", t)
    if m:
        return (today + timedelta(days=int(m.group(1)))).isoformat()
    m = re.match(r"^(?:next |this |on )?(" + "|".join(_WEEKDAYS) + r")$", t)
    if m:
        delta = (_WEEKDAYS.index(m.group(1)) - today.weekday()) % 7
        if delta == 0 or t.startswith("next "):
            delta = delta or 7
        return (today + timedelta(days=delta)).isoformat()
    try:
        has_year = bool(re.search(r"\b\d{4}\b", t))
        d = dateparser.parse(t, dayfirst=True, fuzzy=False, default=datetime(today.year, 1, 1))
        if not has_year and d.date() < today:
            d = d.replace(year=today.year + 1)
        if 2000 <= d.year <= 2100:
            return d.date().isoformat()
    except (ValueError, OverflowError):
        pass
    return None


def parse_time(text: str) -> Optional[str]:
    t = str(text).lower().strip()
    m = re.match(r"^(?:at )?(\d{1,2})(?::(\d{2}))?\s*(am|pm|a m|p m)?$", t)
    if not m:
        return None
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").replace(" ", "")
    if ap == "pm" and h < 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    if h > 23 or mi > 59 or (ap and not 1 <= int(m.group(1)) <= 12):
        return None
    return f"{h:02d}:{mi:02d}"


def validate_field(field_id: str, raw: Any, actor: Dict[str, Any], fdef: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
    """Returns (ok, normalised_value, error_message)."""
    text = re.sub(r"\s+", " ", str(raw if raw is not None else "")).strip()
    label = fdef.get("label", field_id)
    if not text:
        return False, None, f"{label} cannot be empty."
    if len(text) > config.MAX_FIELD_CHARS:
        return False, None, f"{label} is too long (max {config.MAX_FIELD_CHARS} characters)."
    clean, found = redact(text)
    if found:
        return False, None, "That looks like a password, code or sensitive identifier. I can't accept those in chat - please re-enter without it."
    if not re.search(r"[A-Za-z0-9]", text):
        return False, None, f"Please enter a valid {label.lower()}."

    if field_id in ID_FIELDS:
        v = text.upper().replace(" ", "")
        if not _ID_RE.match(v):
            return False, None, f"{label} should look like EMP-1001."
        if actor["role"] == "employee" and v != actor["user_id"]:
            return False, None, (f"For security I can only file requests under your own ID ({actor['user_id']}). "
                                 "I can't submit on behalf of another person.")
        return True, v, None
    if field_id in DATE_FIELDS:
        d = parse_date(text)
        return (True, d, None) if d else (False, None, f"I couldn't read that date. Try 2026-10-15, '15 Oct' or 'next Monday'.")
    if field_id in TIME_FIELDS:
        t = parse_time(text)
        return (True, t, None) if t else (False, None, "I couldn't read that time. Try 14:30 or 2:30pm.")
    if field_id == "end_time":
        pass
    if field_id in ("travel_dates", "requested_period"):
        if not re.search(r"\d", text) and not any(m in text.lower() for m in ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "week", "month", "day")):
            return False, None, f"Please give dates, e.g. 2026-11-03 to 2026-11-05."
        return True, text, None
    if field_id == "attendees" and not re.match(r"^\d+$", text):
        return False, None, "Please give the expected number of attendees, e.g. 8."
    if field_id in ("quantity", "attendees") and re.match(r"^\d+$", text):
        n = int(text)
        if not 1 <= n <= (10000 if field_id == "quantity" else 500):
            return False, None, f"{label} must be a sensible positive number."
        return True, str(n), None
    if field_id == "quantity":
        return False, None, "Quantity must be a whole number, e.g. 5."
    if field_id == "manager_approval":
        n = normalize(text)
        if re.match(r"^(yes|y|approved|obtained|got it|have it|yes approved)\b", n):
            return True, "Yes - approved", None
        if re.match(r"^(no|n|not yet|pending|waiting)\b", n):
            return True, "No - not yet", None
        return False, None, "Please answer Yes (approved) or No (not yet)."
    if field_id == "urgency":
        n = normalize(text)
        for opt in ("critical", "high", "normal", "low", "medium"):
            if opt in n:
                return True, "Normal" if opt == "medium" else opt.title(), None
        return False, None, "Urgency should be Low, Normal, High or Critical."
    if field_id == "purchase_order_id" and not re.match(r"^PO-?\d{3,}$", text.upper()):
        return False, None, "Purchase order IDs look like PO-12345."
    if field_id == "invoice_id" and not re.match(r"^INV-?\d{3,}$", text.upper()):
        return False, None, "Invoice IDs look like INV-12345."
    if field_id in ("purchase_order_id", "invoice_id"):
        return True, text.upper(), None
    if field_id == "request_id" and not re.match(r"^REQ-[A-Z0-9-]{4,}$", text.upper()):
        return False, None, "Request IDs look like REQ-1A2B3C4D."
    if field_id == "request_id":
        return True, text.upper(), None
    if len(text) < 2:
        return False, None, f"Please add a little more detail for {label.lower()}."
    return True, text, None


def _label_variants(fdef: Dict[str, Any]) -> List[str]:
    return sorted({fdef["label"].lower(), fdef["id"].replace("_", " "), fdef["id"]}, key=len, reverse=True)


def extract_labeled(message: str, field_ids: List[str], fdefs: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Parse 'department: HR, manager is Ravi' and 'change reason to damaged card'."""
    labels = {lv: fid for fid in field_ids for lv in _label_variants(fdefs[fid])}
    if not labels:
        return {}
    alt = "|".join(re.escape(l) for l in labels)
    splitter = re.compile(rf"[;\n]|,(?=\s*(?:my |the )?(?:{alt})\s*(?:is|are|=|:|-))|\band\b(?=\s+(?:my |the )?(?:{alt})\s*(?:is|are|=|:|-))", re.I)
    out: Dict[str, str] = {}
    for chunk in splitter.split(message):
        chunk = chunk.strip(" .")
        m = re.match(rf"^(?:my |the )?({alt})\s*(?:is|are|=|:|-|should be)\s*(.+)$", chunk, re.I) or \
            re.match(rf"^(?:change|set|update|edit|make|correct)\s+(?:my |the )?({alt})\s+(?:to|as|=)\s+(.+)$", chunk, re.I)
        if m:
            out[labels[m.group(1).lower()]] = m.group(2).strip()
    return out


def extract_from_message(message: str, required: List[str], tone: Dict[str, Any]) -> Dict[str, str]:
    """Pre-fill only what the user literally wrote (never assumptions)."""
    out: Dict[str, str] = {}
    low = message.lower()
    clean, _ = redact(message)
    for fid in required:
        if fid in FREE_TEXT_SUMMARY:
            out[fid] = clean[: config.MAX_FIELD_CHARS]
        if fid in BECAUSE_FIELDS:
            m = re.search(r"\b(?:because|since|due to|as)\b\s+(.{3,200})$", clean, re.I)
            if m:
                out[fid] = m.group(1).strip(" .")
        if fid == "device_id":
            m = re.search(r"\b([A-Z]{2,4}-\d{3,6})\b", message)
            if m:
                out[fid] = m.group(1)
        if fid == "purchase_order_id":
            m = re.search(r"\b(PO-?\d{3,})\b", message, re.I)
            if m:
                out[fid] = m.group(1).upper()
        if fid == "invoice_id":
            m = re.search(r"\b(INV-?\d{3,})\b", message, re.I)
            if m:
                out[fid] = m.group(1).upper()
        if fid == "request_id":
            m = re.search(r"\b(REQ-[A-Z0-9-]{4,})\b", message, re.I)
            if m:
                out[fid] = m.group(1).upper()
        if fid == "building":
            m = re.search(r"\bbuilding\s+([A-Za-z0-9]{1,3})\b", message, re.I)
            if m:
                out[fid] = f"Building {m.group(1).upper()}"
        if fid == "location":
            b = re.search(r"\bbuilding\s+([A-Za-z0-9]{1,3})\b", message, re.I)
            fl = re.search(r"\b(?:floor|level)\s+(\d{1,2}|ground|first|second|third)\b|\b(\d{1,2})(?:st|nd|rd|th)\s+floor\b", message, re.I)
            if b or fl:
                out[fid] = ", ".join(x for x in (f"Building {b.group(1).upper()}" if b else "",
                                                 f"Floor {fl.group(1) or fl.group(2)}" if fl else "") if x)
        if fid == "floor":
            m = re.search(r"\b(?:floor|level)\s+(\d{1,2}|ground|first|second|third)\b|\b(\d{1,2})(?:st|nd|rd|th)\s+floor\b", message, re.I)
            if m:
                out[fid] = f"Floor {m.group(1) or m.group(2)}"
        if fid == "urgency":
            if any(w in low for w in ("emergency", "critical", "smoke", "flood")):
                out[fid] = "Critical"
            elif tone["urgency"] == "high":
                out[fid] = "High"
    date_ids = [f for f in required if f in DATE_FIELDS]
    if len(date_ids) == 1:
        m = re.search(r"\b(today|tomorrow|day after tomorrow|next (?:" + "|".join(_WEEKDAYS) + r")|(?:this |on )?(?:" + "|".join(_WEEKDAYS) + r")|\d{4}-\d{2}-\d{2})\b", low)
        if m and parse_date(m.group(1)):
            out[date_ids[0]] = m.group(1)
    return out
