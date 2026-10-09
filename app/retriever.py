"""Approved-knowledge retrieval: hybrid BM25 + TF-IDF fused with Reciprocal Rank Fusion.

Only articles with status == 'approved' are ever retrievable. Reported `score` stays a true text-similarity
value (0-1) so thresholds mean the same thing in extractive and LLM mode.
"""
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS and len(t) > 1]


class BM25:
    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = float(self.len.mean()) if len(docs) else 1.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: List[str]) -> np.ndarray:
        out = np.zeros(len(self.tf))
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            for t in query:
                f = tf.get(t)
                if f:
                    out[i] += self.idf.get(t, 0) * f * (self.k1 + 1) / (f + norm)
        return out


class KnowledgeRetriever:
    def __init__(self, store, top_k: int = 4):
        self.store = store
        self.top_k = top_k
        self.rebuild()

    def rebuild(self):
        self.documents: List[Dict[str, Any]] = []
        for row in self.store.records("Knowledge_Articles"):
            if str(row.get("status", "")).lower() != "approved":
                continue
            self.documents.append({k: row[k] for k in ("article_id", "title", "department", "category", "content",
                                                       "source_name", "effective_date", "importance")})
        corpus = [f"{d['title']} {d['title']} {d['title']} {d['category']} {d['department']} {d['content']}"
                  for d in self.documents]
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(corpus) if corpus else None
        self.bm25 = BM25([_tokens(c) for c in corpus]) if corpus else None

    def search(self, query: str, top_k: Optional[int] = None, boost_title: Optional[str] = None) -> List[Dict[str, Any]]:
        if not query.strip() or not self.documents:
            return []
        cos = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        bm = self.bm25.scores(_tokens(query))
        # Reciprocal Rank Fusion of the two rankings (robust to different score scales)
        fused = np.zeros(len(self.documents))
        for scores in (cos, bm):
            order = np.argsort(scores)[::-1]
            for rank, idx in enumerate(order):
                if scores[idx] > 0:
                    fused[idx] += 1.0 / (60 + rank)
        if boost_title:     # boost only re-orders; reported score is the real text similarity
            for i, d in enumerate(self.documents):
                if d["title"].lower() == boost_title.lower():
                    fused[i] += 0.02
        out = []
        for idx in np.argsort(fused)[::-1][: top_k or self.top_k]:
            if fused[idx] <= 0:
                break
            d = dict(self.documents[int(idx)])
            # blend cosine with normalised BM25 so keyword-heavy queries are not under-scored
            bm_norm = float(bm[idx] / (bm.max() or 1.0)) * min(1.0, float(cos[idx]) * 2 + 0.2) if bm.max() > 0 else 0.0
            d["score"] = round(float(min(1.0, max(cos[idx], 0.6 * cos[idx] + 0.4 * bm_norm * float(cos.max() or 0)))), 4)
            d["effective_date"] = str(d["effective_date"])[:10]
            out.append(d)
        return out

    def get(self, article_id: str) -> Optional[Dict[str, Any]]:
        return next((d for d in self.documents if d["article_id"] == article_id), None)

    def catalog(self) -> List[Dict[str, str]]:
        """Compact id/title/category list (used to let an LLM route a vague question to the right articles)."""
        return [{"article_id": d["article_id"], "title": d["title"], "category": d["category"]} for d in self.documents]
