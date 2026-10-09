"""Workbook ingestion + a typed catalog layer on top of the raw sheets."""
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REQUIRED_SHEETS = [
    "Knowledge_Articles", "Workflows", "Request_Types", "Field_Definitions", "Routing_Rules",
    "Roles", "Teams", "Users", "Request_Examples", "Safety_Rules",
]

# Friendly option lists shown as quick-reply chips (free text is still accepted).
FIELD_OPTIONS: Dict[str, List[str]] = {
    "contact_method": ["Email", "Phone", "Teams chat", "Portal"],
    "urgency": ["Low", "Normal", "High", "Critical"],
    "access_type": ["Read only", "Read / Write", "Admin"],
    "device_type": ["Laptop", "Desktop", "Mobile", "Tablet"],
    "handover_method": ["Drop-off at IT desk", "Courier pickup"],
    "vehicle_category": ["Two-wheeler", "Four-wheeler"],
    "feedback_type": ["Incorrect answer", "Outdated answer", "Unclear answer", "Other"],
    "safety_status": ["Everyone is safe", "Not sure", "Someone needs help"],
    "manager_approval": ["Yes - approved", "No - not yet"],
    "leave_category": ["Annual leave", "Sick leave", "Parental leave", "Other"],
    "benefit_category": ["Insurance", "Allowances", "Retirement", "Other"],
    "expense_category": ["Travel", "Meals", "Equipment", "Other"],
    "query_category": ["Payslip", "Deductions", "Tax forms", "Other"],
    "letter_type": ["Employment certificate", "Experience letter", "Address proof", "Other"],
}

FIELD_PLACEHOLDERS: Dict[str, str] = {
    "employee_id": "e.g. EMP-1001", "requester_id": "e.g. EMP-1001",
    "date": "e.g. 2026-10-15 or 'next Monday'", "start_time": "e.g. 10:00 or 3pm", "end_time": "e.g. 11:30",
    "joining_date": "e.g. 2026-11-03", "effective_date": "e.g. 2026-12-01", "required_by": "e.g. 2026-10-30",
    "quantity": "e.g. 5", "attendees": "e.g. 8", "manager": "Manager's name",
}


class DataStore:
    def __init__(self, workbook_path: Path):
        self.workbook_path = Path(workbook_path)
        self.sheets: Dict[str, pd.DataFrame] = {}
        self._lock = threading.RLock()
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        self.reload()

    def reload(self):
        with self._lock:
            all_sheets = pd.read_excel(self.workbook_path, sheet_name=None)
            missing = [s for s in REQUIRED_SHEETS if s not in all_sheets]
            if missing:
                raise ValueError(f"Workbook is missing sheets: {missing}")
            clean = {}
            for name, df in all_sheets.items():
                df = df.fillna("")
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
                clean[name] = df
            self.sheets = clean
            self._cache = {}

    # ------------------------------------------------------------------ raw access
    def records(self, sheet: str) -> List[Dict[str, Any]]:
        with self._lock:
            if sheet not in self._cache:
                self._cache[sheet] = self.sheets[sheet].astype(object).to_dict(orient="records")
            return self._cache[sheet]

    def count(self, sheet: str) -> int:
        return len(self.sheets[sheet])

    # ------------------------------------------------------------------ typed helpers
    @staticmethod
    def split_list(raw: Any) -> List[str]:
        return [x.strip() for x in re.split(r"[,;|]", str(raw or "")) if x.strip()]

    def request_type(self, name: str) -> Optional[Dict[str, Any]]:
        key = str(name).strip().lower()
        return next((r for r in self.records("Request_Types") if r["request_type"].lower() == key), None)

    def routing_rule(self, request_type: str) -> Optional[Dict[str, Any]]:
        key = str(request_type).strip().lower()
        return next((r for r in self.records("Routing_Rules") if r["request_type"].lower() == key), None)

    def workflow(self, request_type: str) -> Optional[Dict[str, Any]]:
        key = str(request_type).strip().lower()
        return next((r for r in self.records("Workflows") if r["request_type"].lower() == key), None)

    def required_fields(self, request_type: str) -> List[str]:
        """Union of Routing_Rules + Workflows required fields (routing order first)."""
        out: List[str] = []
        for src in (self.routing_rule(request_type), self.workflow(request_type)):
            if src:
                for f in self.split_list(src.get("required_fields")):
                    if f not in out:
                        out.append(f)
        return out

    def field_def(self, field_id: str) -> Dict[str, Any]:
        row = next((r for r in self.records("Field_Definitions") if r["field_id"] == field_id), None)
        if row:
            return {
                "id": field_id, "label": row["display_name"], "type": row["data_type"],
                "hint": row["description"], "rule": row["validation_rule"], "safety": row["safety_note"],
                "options": FIELD_OPTIONS.get(field_id, []), "placeholder": FIELD_PLACEHOLDERS.get(field_id, ""),
            }
        return {"id": field_id, "label": field_id.replace("_", " ").title(), "type": "string", "hint": "",
                "rule": "", "safety": "", "options": FIELD_OPTIONS.get(field_id, []),
                "placeholder": FIELD_PLACEHOLDERS.get(field_id, "")}

    def team(self, name: str) -> Optional[Dict[str, Any]]:
        return next((t for t in self.records("Teams") if t["team_name"] == name), None)

    def team_names(self) -> List[str]:
        return [t["team_name"] for t in self.records("Teams")]

    def teams_for_department(self, department: str) -> List[str]:
        return [t["team_name"] for t in self.records("Teams") if t["department"].lower() == department.lower()]

    def article_for(self, request_type: str) -> Optional[Dict[str, Any]]:
        key = str(request_type).strip().lower()
        return next((a for a in self.records("Knowledge_Articles")
                     if a["title"].lower() == key and a["status"].lower() == "approved"), None)

    def safety_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        return next((r for r in self.records("Safety_Rules") if r["rule_id"] == rule_id), None)
