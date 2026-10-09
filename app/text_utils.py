"""Text helpers: normalisation, whole-word matching and secret/PII redaction."""
import re
import unicodedata
from typing import List, Tuple

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS = re.compile(r"\s+")


def clean_input(text: str, limit: int) -> str:
    """Strip control characters, normalise unicode, collapse runaway whitespace, cap length."""
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = _CONTROL.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:limit]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    text = re.sub(r"[^a-z0-9@/\-\s']", " ", text)
    return _WS.sub(" ", text).strip()


def has_phrase(text_norm: str, phrase: str) -> bool:
    """Whole-word / whole-phrase match (so 'ac' does not match 'back'). Allows simple plurals."""
    p = normalize(phrase)
    if not p:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(p) + r"(?:s|es|ed|ing)?(?![a-z0-9])", text_norm) is not None


def title_case_field(field_id: str) -> str:
    return field_id.replace("_", " ").strip().title()


# --------------------------------------------------------------------------- redaction
_SECRET_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("PASSWORD", re.compile(r"(?i)\b(?:password|passwd|pwd|passcode|pin)\s*(?:is|:|=|-)\s*\S+")),
    ("OTP", re.compile(r"(?i)\b(?:otp|one[- ]time (?:code|password)|verification code|2fa code|mfa code)\s*(?:is|:|=|-)?\s*\d{4,8}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("GOVT_ID", re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{4}\s\d{4}\s\d{4}\b")),
    ("API_KEY", re.compile(r"\b(?:sk|pk|api|key|token)[-_][A-Za-z0-9_\-]{16,}\b")),
]


def redact(text: str) -> Tuple[str, List[str]]:
    """Mask credentials and government/financial identifiers BEFORE anything is stored or logged."""
    found: List[str] = []
    out = text
    for label, pat in _SECRET_PATTERNS:
        if pat.search(out):
            found.append(label)
            out = pat.sub(f"[REDACTED {label}]", out)
    return out, found
