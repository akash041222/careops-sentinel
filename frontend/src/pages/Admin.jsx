import React, { useEffect, useState } from "react";
import { Database, RefreshCw, ShieldCheck } from "lucide-react";
import { api } from "../api";
import { Badge, Card, ErrorNote, Metric, PageTitle, Spinner, useToast } from "../components/ui";

export default function Admin() {
  const toast = useToast();
  const [o, setO] = useState(null); const [err, setErr] = useState(""); const [tab, setTab] = useState("knowledge"); const [rows, setRows] = useState(null); const [q, setQ] = useState(""); const [busy, setBusy] = useState(false);
  const load = () => api("/api/admin/overview").then(setO).catch((e) => setErr(e.message));
  useEffect(() => { load(); }, []);
  useEffect(() => {
    setRows(null);
    const url = { knowledge: `/api/admin/knowledge?q=${encodeURIComponent(q)}`, routing: "/api/admin/routing-rules", safety: "/api/admin/safety-rules", users: "/api/admin/users" }[tab];
    const t = setTimeout(() => api(url).then((d) => setRows(d.items)).catch((e) => toast(e.message, "error")), tab === "knowledge" ? 250 : 0);
    return () => clearTimeout(t);
  }, [tab, q, toast]);
  const reload = async () => { setBusy(true); try { const r = await api("/api/admin/reload", { method: "POST" }); toast(`Reloaded: ${r.articles} articles, ${r.request_types} request types`, "success"); load(); } catch (e) { toast(e.message, "error"); } finally { setBusy(false); } };
  if (err) return <ErrorNote error={err} onRetry={load} />;
  if (!o) return <Spinner />;
  return (
    <div className="page"><div className="page-main">
      <PageTitle eyebrow="Administration" title="System & knowledge" sub={`Workbook: ${o.workbook} · ${o.ai_mode}`} right={<button className="secondary sm" disabled={busy} onClick={reload}><RefreshCw size={14} /> Reload workbook</button>} />
      <div className="metrics">{[["Knowledge articles", o.knowledge_articles], ["Request types", o.request_types], ["Workflows", o.workflows], ["Teams", o.teams], ["Training examples", o.training_examples], ["Users", o.users]].map(([l, v]) => <Metric key={l} icon={Database} label={l} value={v} />)}</div>
      <div className="grid-2">
        <Card title="Configuration" icon={ShieldCheck}><div className="kv"><span>Confidence threshold</span><b>{o.config.low_confidence_threshold}</b><span>Session length</span><b>{o.config.token_ttl_minutes} min</b>
          <span>SLA (h)</span><b>{Object.entries(o.config.sla_hours).map(([k, v]) => `${k} ${v}`).join(" · ")}</b><span>Demo mode</span><b>{String(o.config.demo_mode)}</b>
          <span>Secret key set</span><b>{o.config.secret_key_configured ? <Badge tone="green">yes</Badge> : <Badge tone="orange">no - set CAREOPS_SECRET_KEY</Badge>}</b></div></Card>
        <Card title="Language model (RAG)" icon={Database}><div className="kv"><span>Status</span><b>{o.llm.available ? <Badge tone="green">active</Badge> : o.llm.configured ? <Badge tone="orange">circuit open</Badge> : <Badge tone="gray">not configured - extractive mode</Badge>}</b>
          <span>Model</span><b>{o.llm.model || "—"}</b><span>Calls / failures</span><b>{o.llm.calls} / {o.llm.failures}</b><span>Cache hits</span><b>{o.llm.cache_hits}</b><span>Tokens in / out</span><b>{o.llm.input_tokens} / {o.llm.output_tokens}</b></div></Card>
        <Card title="Audit integrity" icon={ShieldCheck}><div className="kv"><span>Chain status</span><b>{o.audit_chain.valid ? <Badge tone="green">intact</Badge> : <Badge tone="red">broken</Badge>}</b><span>Entries verified</span><b>{o.audit_chain.checked}</b><span>Requests stored</span><b>{o.requests}</b></div></Card>
      </div>
      <div className="tabs">{[["knowledge", "Approved knowledge"], ["routing", "Routing rules"], ["safety", "Safety rules"], ["users", "Users"]].map(([k, l]) => <button key={k} className={tab === k ? "active" : ""} onClick={() => setTab(k)}>{l}</button>)}</div>
      {tab === "knowledge" && <div className="toolbar"><input placeholder="Search knowledge (semantic)…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search knowledge" /></div>}
      {!rows ? <Spinner /> : <div className="table-wrap"><table><thead><tr>{(tab === "knowledge" ? ["ID", "Title", "Dept", "Source", "Content"] : tab === "users" ? ["ID", "Name", "Role", "Department"] : tab === "routing" ? ["Request type", "Team", "Priority", "Action"] : ["Rule", "Name", "Trigger", "Action"]).map((h) => <th key={h}>{h}</th>)}</tr></thead>
        <tbody>{rows.slice(0, 120).map((r, i) => tab === "knowledge" ? <tr key={i}><td>{r.article_id}</td><td><b>{r.title}</b></td><td>{r.department}</td><td>{r.source_name}<small>{r.effective_date?.slice?.(0, 10)}</small></td><td>{r.content}</td></tr>
          : tab === "users" ? <tr key={i}><td>{r.user_id}</td><td>{r.name}</td><td>{r.role}</td><td>{r.department}</td></tr>
          : tab === "routing" ? <tr key={i}><td><b>{r.request_type}</b></td><td>{r.target_team}</td><td>{r.priority}</td><td>{r.routing_action}</td></tr>
          : <tr key={i}>{Object.values(r).slice(0, 4).map((v, j) => <td key={j}>{String(v)}</td>)}</tr>)}</tbody></table></div>}
    </div></div>
  );
}
