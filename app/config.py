"""Central configuration. Every value can be overridden with an environment variable."""
import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
WORKBOOK_PATH = Path(os.getenv("CAREOPS_WORKBOOK", DATA_DIR / "careops_phase1_expanded_synthetic_workbook.xlsx"))
DB_PATH = Path(os.getenv("CAREOPS_DB", DATA_DIR / "careops.db"))

# --- AI behaviour -----------------------------------------------------------
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("CAREOPS_LOW_CONFIDENCE", "0.70"))   # below: clarify / escalate
CLARIFY_FLOOR = float(os.getenv("CAREOPS_CLARIFY_FLOOR", "0.42"))               # below: "I don't know"
MIN_RETRIEVAL_SCORE = float(os.getenv("CAREOPS_MIN_RETRIEVAL", "0.12"))

# --- Security ---------------------------------------------------------------
_secret = os.getenv("CAREOPS_SECRET_KEY")
SECRET_KEY_IS_DEFAULT = _secret is None
# Without CAREOPS_SECRET_KEY a random key is generated per process (sessions end on restart).
SECRET_KEY = (_secret or secrets.token_hex(32)).encode()
TOKEN_TTL_MINUTES = int(os.getenv("CAREOPS_TOKEN_TTL_MIN", "480"))
DEMO_PASSWORD = os.getenv("CAREOPS_DEMO_PASSWORD", "CareOps@123")
DEMO_MODE = os.getenv("CAREOPS_DEMO_MODE", "1") == "1"      # exposes demo-account hints on the login page
SEED_DEMO_DATA = os.getenv("CAREOPS_SEED_DEMO", "1") == "1"  # fills an empty DB with synthetic requests
CORS_ORIGINS = [
    o.strip() for o in os.getenv(
        "CAREOPS_CORS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
    ).split(",") if o.strip()
]
LOGIN_ATTEMPTS = int(os.getenv("CAREOPS_LOGIN_ATTEMPTS", "8"))     # per 5 minutes per account+IP
CHAT_PER_MINUTE = int(os.getenv("CAREOPS_CHAT_PER_MIN", "40"))

# --- Operations -------------------------------------------------------------
SLA_HOURS = {"critical": 1, "high": 4, "normal": 24, "low": 72}
MAX_MESSAGE_CHARS = 2000
MAX_FIELD_CHARS = 500

# --- LLM (optional; everything works without it using extractive answers) -------------------------
LLM_PROVIDER = os.getenv("CAREOPS_LLM_PROVIDER", "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "none").lower()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("CAREOPS_LLM_MODEL", "claude-sonnet-5-5")
LLM_BASE_URL = os.getenv("CAREOPS_LLM_BASE_URL", "https://api.anthropic.com")
LLM_TIMEOUT_S = float(os.getenv("CAREOPS_LLM_TIMEOUT", "20"))
LLM_MAX_TOKENS = int(os.getenv("CAREOPS_LLM_MAX_TOKENS", "600"))
