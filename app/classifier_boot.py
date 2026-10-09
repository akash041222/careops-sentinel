"""Builds the singleton components once (kept out of main.py so tests can build isolated instances)."""
from pathlib import Path
from typing import Any, Dict, Optional

from . import config
from .data_store import DataStore
from .db import Database
from .engine import Engine
from .llm import LLMClient, RAGAssistant
from .nlu import IntentClassifier
from .ops import Operations
from .retriever import KnowledgeRetriever
from .security import UserDirectory


def build_components(db_path: Optional[Path] = None) -> Dict[str, Any]:
    store = DataStore(config.WORKBOOK_PATH)
    db = Database(db_path or config.DB_PATH)
    users = UserDirectory(store)
    classifier = IntentClassifier(store)
    retriever = KnowledgeRetriever(store)
    ops = Operations(store, db, users)
    rag = RAGAssistant(LLMClient.from_env())
    engine = Engine(store, db, users, classifier, retriever, ops, rag)
    return {"rag": rag, "store": store, "db": db, "users": users, "classifier": classifier, "retriever": retriever, "ops": ops, "engine": engine}
