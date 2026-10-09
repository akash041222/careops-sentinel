import React, { useCallback, useEffect, useState } from "react";
import { Bot, ClipboardList, MessageSquare, Sparkles, X } from "lucide-react";
import { api } from "../api";
import { Badge, Card, ErrorNote, Markdown, PRIORITY_TONE, SlaBadge, Spinner, StatusBadge, fmtDate, useToast } from "../components/ui";

const ACTION_LABEL = { take: "Take ownership", release: "Release to queue", request_info: "Request information", resolve: "Mark resolved", reject: "Reject", reassign: "Reassign team", reopen: "Reopen", note: "Add internal note" };
const NEEDS_NOTE = ["resolve", "reject", "request_info", "reopen", "reassign", "note"];

export default function RequestDetail({ id, user, onClose, onChanged }) {
  const toast = useToast();
  const [r, setR] = useState(null); const [err, setErr] = useState(""); const [assist, setAssist] = useState(null);
  const [note, setNote] = useState(""); const [team, setTeam] = useState(""); const [teams, setTeams] = useState([]); const [busy, setBusy] = useState(false);
  const staff = user.role !== "employee";

  const load = useCallback(() => api(`/api/requests/${id}`).then((d) => { setR(d); setErr(""); }).catch((e) => setErr(e.message)), [id]);
  useEffect(() => { setR(null); setAssist(null); setNote(""); load(); if (staff) api("/api/teams").then((d) => setTeams(d.items)).catch(() => {}); }, [load, staff]);

  const act = async (action) => {
    if (NEEDS_NOTE.includes(action) && !note.trim()) return toast("Please add a note first.", "error");
    if (action === "reassign" && !team) return toast("Choose a team.", "error");
    setBusy(true);
    try {
      let d;
      if (action === "reply") d = await api(`/api/requests/${id}/reply`, { method: "POST", body: { text: note } });
      else if (action === "cancel") d = await api(`/api/requests/${id}/cancel`, { method: "POST" });
      else d = await api(`/api/requests/${id}/actions`, { method: "POST", body: { action, note, team: team || undefined } });
      setR(d); setNote(""); onChanged?.(); toast("Updated", "success");
    } catch (e) { toast(e.message, "error"); } finally { setBusy(false); }
  };
  const loadAssist = async () => { try { setAssist(await api(`/api/requests/${id}/assist`)); } catch (e) { toast(e.message, "error"); } };

  if (err) return <div className="drawer"><ErrorNote error={err} /><button className="secondary" onClick={onClose}>Close</button></div>;
  if (!r) return <div className="drawer"><Spinner /></div>;
  const fields = Object.entries(r.fields || {});
  return (
    <div className="drawer" role="dialog" aria-label={`Request ${r.request_id}`}>
      <div className="drawer-head">
        <div><span className="eyebrow">{r.request_id}</span><h2>{r.request_type}</h2>
          <div className="meta-row"><StatusBadge status={r.status} /><Badge tone={PRIORITY_TONE[r.priority]}>{r.priority}</Badge><SlaBadge req={r} /><Badge tone="gray">{r.assigned_team}</Badge></div></div>
        <button className="icon-btn" onClick={onClose} aria-label="Close"><X size={18} /></button>
      </div>
      <div className="drawer-body">
        <Card title="Request details" icon={ClipboardList}>
          <div className="kv"><span>Requester</span><b>{r.requester_name} ({r.user_id})</b><span>Submitted</span><b>{fmtDate(r.created_at)}</b>
            {fields.map(([k, v]) => <React.Fragment key={k}><span>{k.replace(/_/g, " ")}</span><b>{String(v)}</b></React.Fragment>)}</div>
          {r.resolution_note && <div className="note-box"><b>Outcome:</b> {r.resolution_note}</div>}
        </Card>

        {staff && (
          <Card title="AI context" sub="Why this reached a human" icon={Bot}>
            <div className="meta-row">{r.confidence != null && <Badge tone={r.confidence >= 0.8 ? "green" : r.confidence >= 0.6 ? "amber" : "red"}>Confidence {Math.round(r.confidence * 100)}%</Badge>}
              {r.sentiment && <Badge tone={r.sentiment === "negative" ? "red" : "gray"}>Sentiment: {r.sentiment}</Badge>}{r.urgency && <Badge tone={r.urgency === "high" ? "orange" : "gray"}>Urgency: {r.urgency}</Badge>}{r.risk_level && <Badge tone="gray">Risk: {r.risk_level}</Badge>}</div>
            <p className="muted"><b>Escalation reason:</b> {r.escalation_reason}</p>
            {r.sources?.length > 0 && <p className="muted"><b>Sources:</b> {r.sources.map((s) => `${s.article_id} ${s.title}`).join(" · ")}</p>}
            {!assist ? <button className="secondary sm" onClick={loadAssist}><Sparkles size={14} /> Agent assist (advisory)</button> : (
              <div className="assist"><Markdown text={assist.summary} />
                <b>Checklist</b><ul>{assist.checklist.map((c, i) => <li key={i}>{c}</li>)}</ul>
                <b>Suggested reply</b><div className="note-box">{assist.draft_reply}</div><small className="muted">{assist.note}</small></div>)}
          </Card>)}

        {staff && r.transcript?.length > 0 && (
          <Card title="Conversation" icon={MessageSquare}><div className="transcript">{r.transcript.map((m, i) => <div key={i} className={`t-${m.role}`}><b>{m.role === "user" ? "Requester" : "Assistant"}</b><Markdown text={m.content} /></div>)}</div></Card>)}

        <Card title="Timeline" icon={ClipboardList}>
          <ol className="timeline">{r.events.map((e) => <li key={e.id}><b>{e.action.replace(/_/g, " ")}</b> <small>{e.actor_id} · {fmtDate(e.ts)}</small>{e.note && <p>{e.note}</p>}</li>)}</ol>
        </Card>

        {r.actions.length > 0 && (
          <Card title={staff ? "Take action" : "Your options"} sub="Only people can change a request's status">
            <textarea className="note-input" rows={3} maxLength={1000} placeholder={staff ? "Add a note (required for most actions)…" : "Your reply…"} value={note} onChange={(e) => setNote(e.target.value)} />
            {r.actions.includes("reassign") && <select value={team} onChange={(e) => setTeam(e.target.value)} aria-label="Reassign to team"><option value="">Reassign to team…</option>{teams.filter((t) => t !== r.assigned_team).map((t) => <option key={t}>{t}</option>)}</select>}
            <div className="actions wrap">{r.actions.map((a) => (
              <button key={a} disabled={busy} className={a === "resolve" || a === "take" || a === "reply" ? "primary sm" : a === "reject" || a === "cancel" ? "danger sm" : "secondary sm"} onClick={() => act(a)}>
                {a === "reply" ? "Send reply" : a === "cancel" ? "Cancel request" : ACTION_LABEL[a]}</button>))}</div>
          </Card>)}
      </div>
    </div>
  );
}
