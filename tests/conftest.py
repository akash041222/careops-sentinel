import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_tmp = tempfile.mkdtemp()
os.environ["CAREOPS_DB"] = str(Path(_tmp) / "test.db")
os.environ["CAREOPS_SEED_DEMO"] = "1"
os.environ["CAREOPS_LOGIN_ATTEMPTS"] = "8"
os.environ["CAREOPS_CHAT_PER_MIN"] = "1000"

from fastapi.testclient import TestClient  # noqa: E402

import app.main as main  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(main.app) as c:
        yield c


def _login(client, uid, pw="CareOps@123"):
    r = client.post("/api/auth/login", json={"user_id": uid, "password": pw})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


@pytest.fixture(scope="session")
def auth(client):
    return lambda uid: _login(client, uid)


class Chat:
    """Tiny helper that keeps a conversation going."""

    def __init__(self, client, headers):
        self.c, self.h, self.cid, self.last = client, headers, None, None

    def say(self, message="", **kw):
        r = self.c.post("/api/chat", headers=self.h, json={"conversation_id": self.cid, "message": message, **kw})
        assert r.status_code == 200, r.text
        self.last = r.json()
        self.cid = self.last["conversation_id"]
        return self.last


@pytest.fixture()
def chat(client, auth):
    def make(uid="EMP-1001"):
        return Chat(client, auth(uid))
    return make
