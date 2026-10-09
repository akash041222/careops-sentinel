import React, { useCallback, useEffect, useState } from "react";
import { Inbox, RefreshCw, Search } from "lucide-react";
import { api } from "../api";
import { Badge, Empty, ErrorNote, PageTitle, PRIORITY_TONE, SlaBadge, Spinner, StatusBadge, fmtDate } from "../components/ui";
import RequestDetail from "./RequestDetail";

const STATUSES = ["", "pending_human", "in_progress", "needs_information", "resolved", "rejected", "cancelled"];

export default function Requests({ user, mode, initialRequestId = null, initialRequestNonce = 0 }) {   // mode: "mine" (employee) | "queue" (staff)
  const staff = mode === "queue";
  const [items, setItems] = useState(null); const [total, setTotal] = useState(0); const [err, setErr] = useState("");
  const [status, setStatus] = useState(staff ? "" : ""); const [priority, setPriority] = useState(""); const [q, setQ] = useState(""); const [sel, setSel] = useState(null);
  const load = useCallback(() => {
    const p = new URLSearchParams({ limit: "100" }); if (status) p.set("status", status); if (priority) p.set("priority", priority); if (q.trim()) p.set("q", q.trim());
    api(`/api/requests?${p}`).then((d) => { setItems(d.items); setTotal(d.total); setErr(""); }).catch((e) => setErr(e.message));
  }, [status, priority, q]);
  useEffect(() => { const t = setTimeout(load, 250); return () => clearTimeout(t); }, [load]);
  useEffect(() => {
    if (staff && initialRequestId) setSel(initialRequestId);
  }, [staff, initialRequestId, initialRequestNonce]);
  useEffect(() => { const t = setInterval(load, 30000); return () => clearInterval(t); }, [load]);

  return (
    <div className={`page ${sel ? "with-drawer" : ""}`}>
      <div className="page-main">
        <PageTitle eyebrow={staff ? "Support console" : "My requests"} title={staff ? "Request queue" : "Track your requests"}
          sub={staff ? "AI-prepared requests waiting for a human decision. Highest priority first." : "Everything you've submitted through the assistant, with its current status."}
          right={<button className="secondary sm" onClick={load}><RefreshCw size={14} /> Refresh</button>} />
        <div className="toolbar">
          <div className="search"><Search size={15} /><input placeholder="Search by ID, type, requester…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search requests" /></div>
          <select value={status} onChange={(e) => setStatus(e.target.value)} aria-label="Status filter">{STATUSES.map((s) => <option key={s} value={s}>{s ? s.replace(/_/g, " ") : "All statuses"}</option>)}</select>
          {staff && <select value={priority} onChange={(e) => setPriority(e.target.value)} aria-label="Priority filter"><option value="">All priorities</option>{["critical", "high", "normal", "low"].map((p) => <option key={p}>{p}</option>)}</select>}
          <span className="muted">{total} result{total === 1 ? "" : "s"}</span>
        </div>
        <ErrorNote error={err} onRetry={load} />
        {!items ? <Spinner /> : items.length === 0 ? <Empty icon={Inbox} title="Nothing here yet">{staff ? "No requests match these filters." : "Ask the assistant to start a request and it will appear here."}</Empty> : (
          <div className="table-wrap"><table>
            <thead><tr><th>Request</th>{staff && <th>Requester</th>}<th>Team</th><th>Priority</th><th>Status</th><th>SLA</th><th>Created</th></tr></thead>
            <tbody>{items.map((r) => (
              <tr key={r.request_id} tabIndex={0} className={sel === r.request_id ? "selected" : ""} onClick={() => setSel(r.request_id)} onKeyDown={(e) => e.key === "Enter" && setSel(r.request_id)}>
                <td><b>{r.request_type}</b><small>{r.request_id}</small></td>{staff && <td>{r.requester_name}<small>{r.user_id}</small></td>}
                <td>{r.assigned_team}</td><td><Badge tone={PRIORITY_TONE[r.priority]}>{r.priority}</Badge></td><td><StatusBadge status={r.status} /></td><td><SlaBadge req={r} /></td><td>{fmtDate(r.created_at)}</td></tr>))}</tbody>
          </table></div>)}
      </div>
      {sel && <RequestDetail id={sel} user={user} onClose={() => setSel(null)} onChanged={load} />}
    </div>
  );
}
