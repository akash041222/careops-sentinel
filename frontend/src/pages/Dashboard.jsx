import React, { useEffect, useState } from "react";
import { Activity, AlertTriangle, Bot, CheckCircle2, Clock, Inbox, ShieldAlert } from "lucide-react";
import { api } from "../api";
import { Card, ErrorNote, Metric, PageTitle, Spinner } from "../components/ui";

const Bars = ({ data, color = "var(--teal)" }) => {
  const max = Math.max(1, ...data.map((d) => d[1]));
  return <div className="bars">{data.map(([k, v]) => <div key={k} className="bar-row"><span title={k}>{k}</span><div><i style={{ width: `${(v / max) * 100}%`, background: color }} /></div><b>{v}</b></div>)}{data.length === 0 && <span className="muted">No data yet</span>}</div>;
};

function Trend({ trend }) {
  const max = Math.max(1, ...trend.flatMap((t) => [t.created, t.resolved]));
  return <div className="trend">{trend.map((t) => (
    <div key={t.date} className="trend-col"><div className="trend-bars"><i title={`${t.created} created`} style={{ height: `${(t.created / max) * 100}%` }} /><i className="alt" title={`${t.resolved} resolved`} style={{ height: `${(t.resolved / max) * 100}%` }} /></div><span>{t.date.slice(5)}</span></div>))}</div>;
}

export default function Dashboard({ user }) {
  const [d, setD] = useState(null); const [err, setErr] = useState("");
  const load = () => api("/api/dashboard/summary").then(setD).catch((e) => setErr(e.message));
  useEffect(() => { load(); const t = setInterval(load, 30000); return () => clearInterval(t); }, []);
  if (err) return <ErrorNote error={err} onRetry={load} />;
  if (!d) return <Spinner />;
  const a = d.assistant; const pct = (x) => (x == null ? "—" : `${Math.round(x * 100)}%`);
  return (
    <div className="page"><div className="page-main">
      <PageTitle eyebrow="Operations" title="Dashboard" sub={`Scope: ${d.scope}. Live from the request queue and the assistant's decisions.`} />
      <div className="metrics">
        <Metric icon={Inbox} label="Open requests" value={d.open_requests} />
        <Metric icon={AlertTriangle} label="SLA breached" value={d.sla.breached} tone={d.sla.breached ? "bad" : ""} />
        <Metric icon={Clock} label="At risk (<2h)" value={d.sla.at_risk} tone={d.sla.at_risk ? "warn" : ""} />
        <Metric icon={CheckCircle2} label="Resolved" value={d.status_counts.resolved} />
        <Metric icon={Activity} label="Avg resolution" value={d.avg_resolution_hours != null ? `${d.avg_resolution_hours}h` : "—"} />
        {a && <Metric icon={Bot} label="AI containment" value={pct(a.containment_rate)} />}
      </div>
      <div className="grid-2">
        <Card title="Requests, last 7 days" sub="Created vs resolved"><Trend trend={d.trend} /><div className="legend"><i /> Created <i className="alt" /> Resolved</div></Card>
        <Card title="Open by priority"><Bars data={Object.entries(d.open_by_priority)} color="var(--orange)" /></Card>
        <Card title="Requests by team"><Bars data={Object.entries(d.team_counts).slice(0, 8)} /></Card>
        <Card title="Status"><Bars data={Object.entries(d.status_counts).map(([k, v]) => [k.replace(/_/g, " "), v])} color="var(--blue)" /></Card>
        {a && <>
          <Card title="Assistant quality" icon={Bot} sub="Responsible-AI indicators">
            <div className="kv"><span>Turns handled</span><b>{a.turns}</b><span>Escalated to humans</span><b>{a.escalated_turns}</b><span>Average confidence</span><b>{pct(a.avg_confidence)}</b>
              <span>Helpful ratings</span><b>{a.feedback.up || 0} 👍 / {a.feedback.down || 0} 👎</b></div></Card>
          <Card title="Top intents"><Bars data={a.top_intents} /></Card>
          <Card title="Tone detected"><Bars data={Object.entries(a.sentiment)} color="var(--purple)" /></Card>
          <Card title="Safety boundary" icon={ShieldAlert} sub="Messages the AI declined to answer"><Bars data={Object.entries(a.safety_flags).map(([k, v]) => [k.replace(/_/g, " "), v])} color="var(--red)" /><p className="muted">{d.safety_boundary}</p></Card></>}
      </div>
    </div></div>
  );
}
