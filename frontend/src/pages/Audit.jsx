import React, { useCallback, useEffect, useState } from "react";
import { Download, ShieldCheck, ShieldX } from "lucide-react";
import { api, download } from "../api";
import { Badge, ErrorNote, PageTitle, Spinner, fmtDate, useToast } from "../components/ui";

export default function Audit() {
  const toast = useToast();
  const [d, setD] = useState(null); const [err, setErr] = useState(""); const [chain, setChain] = useState(null);
  const [action, setAction] = useState(""); const [outcome, setOutcome] = useState(""); const [actor, setActor] = useState(""); const [offset, setOffset] = useState(0);
  const load = useCallback(() => {
    const p = new URLSearchParams({ limit: "50", offset: String(offset) }); if (action) p.set("action", action); if (outcome) p.set("outcome", outcome); if (actor.trim()) p.set("actor", actor.trim().toUpperCase());
    api(`/api/audit?${p}`).then((x) => { setD(x); setErr(""); }).catch((e) => setErr(e.message));
  }, [action, outcome, actor, offset]);
  useEffect(() => { load(); }, [load]);
  useEffect(() => { api("/api/audit/verify").then(setChain).catch(() => {}); }, []);
  return (
    <div className="page"><div className="page-main">
      <PageTitle eyebrow="Governance" title="Audit trail" sub="Every sign-in, AI decision, safety block and status change. Entries are hash-chained so tampering is detectable."
        right={<div className="actions">{chain && <Badge tone={chain.valid ? "green" : "red"}>{chain.valid ? <ShieldCheck size={12} /> : <ShieldX size={12} />} {chain.valid ? `Chain intact · ${chain.checked} entries` : `Broken at #${chain.broken_at}`}</Badge>}
          <button className="secondary sm" onClick={() => download("/api/audit/export", "careops_audit.csv").catch((e) => toast(e.message, "error"))}><Download size={14} /> Export CSV</button></div>} />
      <div className="toolbar">
        <select value={action} onChange={(e) => { setOffset(0); setAction(e.target.value); }} aria-label="Action filter"><option value="">All actions</option>
          {["LOGIN_SUCCESS", "LOGIN_FAILED", "CHAT_TURN", "SAFETY_BLOCK", "NO_SOURCE", "WORKFLOW_STARTED", "FIELD_COLLECTED", "REQUEST_SUBMITTED", "REQUEST_TAKE", "REQUEST_RESOLVE", "REQUEST_REJECT", "ACCESS_DENIED", "ACTION_REJECTED", "FEEDBACK"].map((a) => <option key={a}>{a}</option>)}</select>
        <select value={outcome} onChange={(e) => { setOffset(0); setOutcome(e.target.value); }} aria-label="Outcome filter"><option value="">Any outcome</option>{["success", "blocked", "denied", "escalated", "error"].map((o) => <option key={o}>{o}</option>)}</select>
        <input placeholder="Actor ID" value={actor} onChange={(e) => { setOffset(0); setActor(e.target.value); }} aria-label="Actor filter" />
        {d && <span className="muted">{d.total} entries</span>}
      </div>
      <ErrorNote error={err} onRetry={load} />
      {!d ? <Spinner /> : <>
        <div className="table-wrap"><table><thead><tr><th>Time</th><th>Actor</th><th>Action</th><th>Outcome</th><th>Details</th></tr></thead>
          <tbody>{d.items.map((r) => <tr key={r.id}><td>{fmtDate(r.ts)}</td><td><b>{r.actor_id}</b><small>{r.actor_role}</small></td><td><code>{r.action}</code></td>
            <td><Badge tone={{ success: "green", blocked: "red", denied: "red", escalated: "orange", error: "red" }[r.outcome] || "gray"}>{r.outcome}</Badge></td>
            <td>{r.description}{r.request_id && <small>{r.request_id}</small>}</td></tr>)}</tbody></table></div>
        <div className="pager"><button className="secondary sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 50))}>Previous</button>
          <span className="muted">{offset + 1}–{Math.min(offset + 50, d.total)} of {d.total}</span>
          <button className="secondary sm" disabled={offset + 50 >= d.total} onClick={() => setOffset(offset + 50)}>Next</button></div></>}
    </div></div>
  );
}
