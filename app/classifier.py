from typing import Any, Dict, List
import re

class RuleClassifier:
    """
    Phase 2 classifier:
    - deterministic keyword matching against Request_Types.keywords
    - safe fallback to Unclear Request
    - no LLM calls yet
    """

    def __init__(self, store):
        self.store = store

    def classify(self, text: str) -> Dict[str, Any]:
        text_l = text.lower()
        candidates = []

        for row in self.store.records("Request_Types"):
            keywords = [k.strip().lower() for k in str(row.get("keywords", "")).split(",") if k.strip()]
            hits = sum(1 for k in keywords if k in text_l)
            if hits:
                candidates.append((hits, row))

        if not candidates:
            fallback = next(
                r for r in self.store.records("Request_Types")
                if r["request_type"] == "Unclear Request"
            )
            return self._result(fallback, 0.40)

        candidates.sort(key=lambda x: x[0], reverse=True)
        hits, best = candidates[0]
        keyword_count = max(1, len(str(best["keywords"]).split(",")))
        confidence = min(0.98, 0.60 + (hits / keyword_count) * 0.35)

        # Multiple equally strong candidates = uncertainty.
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            confidence = min(confidence, 0.68)

        return self._result(best, round(confidence, 2))

    def _result(self, row, confidence):
        return {
            "intent": row["request_type"],
            "department": row["department"],
            "workflow_id": row["workflow_id"],
            "risk_level": row["risk_level"],
            "handling_mode": row["handling_mode"],
            "confidence": confidence,
        }
