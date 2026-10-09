"""Authentication (signed, expiring bearer tokens), RBAC dependencies and rate limiting. stdlib only."""
import base64
import hashlib
import hmac
import json
import secrets
import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, List, Optional

from fastapi import Depends, HTTPException, Request

from . import config

# ------------------------------------------------------------------ passwords
_ITER = 120_000


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITER)
    return f"pbkdf2${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_hex, dk_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), _ITER)
        return hmac.compare_digest(dk.hex(), dk_hex)
    except Exception:
        return False


# ------------------------------------------------------------------ tokens (HS256-style)
def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def create_token(user: Dict[str, Any], ttl_minutes: Optional[int] = None) -> Dict[str, Any]:
    now = int(time.time())
    exp = now + 60 * (ttl_minutes or config.TOKEN_TTL_MINUTES)
    body = _b64(json.dumps({"sub": user["user_id"], "role": user["role"], "iat": now, "exp": exp,
                            "jti": secrets.token_hex(6)}).encode())
    sig = _b64(hmac.new(config.SECRET_KEY, body.encode(), hashlib.sha256).digest())
    return {"token": f"{body}.{sig}", "expires_at": exp}


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        body, sig = token.split(".")
        expected = _b64(hmac.new(config.SECRET_KEY, body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(_unb64(body))
        if payload["exp"] < time.time():
            return None
        return payload
    except Exception:
        return None


# ------------------------------------------------------------------ user directory
class UserDirectory:
    """Users come from the workbook (Users sheet) plus the documented demo accounts."""

    LEGACY = [
        {"user_id": "SUP-3001", "name": "Rahul Menon", "role": "support_agent", "department": "Support Operations", "scope": "all"},
        {"user_id": "OPS-2001", "name": "Priya Sharma", "role": "operations_manager", "department": "Operations", "scope": "all"},
        {"user_id": "ADM-4001", "name": "Admin User", "role": "admin", "department": "Platform Administration", "scope": "all"},
    ]

    def __init__(self, store):
        self.store = store
        self.users: Dict[str, Dict[str, Any]] = {}
        self._pw = hash_password(config.DEMO_PASSWORD)
        self.rebuild()

    def rebuild(self):
        users: Dict[str, Dict[str, Any]] = {}
        for r in self.store.records("Users"):
            if str(r.get("status", "active")).lower() != "active":
                continue
            users[r["user_id"].upper()] = {"user_id": r["user_id"].upper(), "name": r["name"], "role": r["role"],
                                           "department": r["department"], "scope": None}
        for u in self.LEGACY:
            users.setdefault(u["user_id"], dict(u))
        for u in users.values():
            u["password_hash"] = self._pw
        self.users = users

    def authenticate(self, user_id: str, password: str) -> Optional[Dict[str, Any]]:
        u = self.users.get(str(user_id).strip().upper())
        # Always run a hash so response time does not reveal whether the account exists.
        ok = verify_password(password, u["password_hash"] if u else self._pw)
        return self.public(u) if (u and ok) else None

    def get(self, user_id: str) -> Optional[Dict[str, Any]]:
        u = self.users.get(str(user_id).upper())
        return self.public(u) if u else None

    @staticmethod
    def public(u: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in u.items() if k != "password_hash"}

    def teams_for(self, user: Dict[str, Any]) -> Optional[List[str]]:
        """None = unrestricted. Agents only see teams belonging to their own department."""
        if user["role"] in ("operations_manager", "admin") or user.get("scope") == "all":
            return None
        return self.store.teams_for_department(user["department"])


# ------------------------------------------------------------------ rate limiting
class RateLimiter:
    def __init__(self):
        self.hits: Dict[str, Deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str, limit: int, window: int) -> Optional[int]:
        """Returns None if allowed, otherwise seconds until retry."""
        now = time.time()
        with self.lock:
            q = self.hits[key]
            while q and q[0] <= now - window:
                q.popleft()
            if len(q) >= limit:
                return max(1, int(q[0] + window - now))
            q.append(now)
            return None

    def reset(self, key: str):
        with self.lock:
            self.hits.pop(key, None)


limiter = RateLimiter()
ROLE_LABELS = {"employee": "Employee", "support_agent": "Support Agent",
               "operations_manager": "Operations Manager", "admin": "Administrator"}
