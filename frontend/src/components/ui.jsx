import React, { createContext, useCallback, useContext, useState } from "react";
import { AlertTriangle, CheckCircle2, Info, X } from "lucide-react";

export const ROLE_LABEL = { employee: "Employee", support_agent: "Support Agent", operations_manager: "Operations Manager", admin: "Administrator" };
export const STATUS = {
  pending_human: ["Pending review", "amber"], in_progress: ["In progress", "blue"], needs_information: ["Needs information", "purple"],
  resolved: ["Resolved", "green"], rejected: ["Rejected", "red"], cancelled: ["Cancelled", "gray"],
};
export const PRIORITY_TONE = { critical: "red", high: "orange", normal: "gray", low: "gray" };

export const Badge = ({ tone = "gray", children, title }) => <span className={`badge ${tone}`} title={title}>{children}</span>;
export const StatusBadge = ({ status }) => { const [l, t] = STATUS[status] || [status, "gray"]; return <Badge tone={t}>{l}</Badge>; };
export const Spinner = ({ label = "Loading" }) => <div className="spinner-wrap" role="status"><span className="spinner" /><span>{label}…</span></div>;

export function Empty({ icon: Icon, title, children }) {
  return <div className="empty-state">{Icon && <Icon size={30} />}<b>{title}</b>{children && <span>{children}</span>}</div>;
}
export function ErrorNote({ error, onRetry }) {
  if (!error) return null;
  return <div className="error-note" role="alert"><AlertTriangle size={16} /><span>{error}</span>{onRetry && <button className="link" onClick={onRetry}>Retry</button>}</div>;
}
export function PageTitle({ eyebrow, title, sub, right }) {
  return <div className="page-title"><div><span className="eyebrow">{eyebrow}</span><h1>{title}</h1>{sub && <p>{sub}</p>}</div>{right}</div>;
}
export function Card({ title, sub, icon: Icon, actions, children, className = "" }) {
  return (
    <section className={`panel ${className}`}>
      {(title || actions) && <div className="panel-head"><div className="panel-title">{Icon && <div className="panel-icon"><Icon size={18} /></div>}<div><h2>{title}</h2>{sub && <span>{sub}</span>}</div></div><div className="actions">{actions}</div></div>}
      {children}
    </section>
  );
}
export const Metric = ({ icon: Icon, label, value, tone = "" }) => (
  <div className={`metric ${tone}`}><div className="metric-icon"><Icon size={18} /></div><div><span>{label}</span><b>{value}</b></div></div>
);

export const fmtDate = (s) => (s ? new Date(s).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "—");
export function timeLeft(due) {
  if (!due) return "";
  const ms = new Date(due) - Date.now(); const h = Math.abs(ms) / 36e5;
  const t = h >= 24 ? `${Math.round(h / 24)}d` : h >= 1 ? `${Math.round(h)}h` : `${Math.max(1, Math.round(h * 60))}m`;
  return ms < 0 ? `${t} overdue` : `${t} left`;
}
export const SlaBadge = ({ req }) => {
  if (req.sla_state === "n/a" || !req.sla_state) return null;
  const tone = { breached: "red", at_risk: "orange", on_track: "green" }[req.sla_state];
  return <Badge tone={tone}>{timeLeft(req.sla_due)}</Badge>;
};

// ---------- tiny safe markdown (bold, italic, code, lists, paragraphs). Never uses dangerouslySetInnerHTML.
function inline(text, key) {
  const parts = []; const re = /(\*\*[^*]+\*\*|_[^_]+_|`[^`]+`)/g; let last = 0, m, i = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    const t = m[0];
    parts.push(t.startsWith("**") ? <strong key={`${key}-${i++}`}>{t.slice(2, -2)}</strong> : t.startsWith("`") ? <code key={`${key}-${i++}`}>{t.slice(1, -1)}</code> : <em key={`${key}-${i++}`}>{t.slice(1, -1)}</em>);
    last = m.index + t.length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}
export function Markdown({ text }) {
  const blocks = String(text || "").split(/\n{2,}/);
  return <div className="md">{blocks.map((b, bi) => {
    const lines = b.split("\n");
    if (lines.every((l) => /^\s*- /.test(l))) return <ul key={bi}>{lines.map((l, li) => <li key={li}>{inline(l.replace(/^\s*- /, ""), `${bi}-${li}`)}</li>)}</ul>;
    return <p key={bi}>{lines.map((l, li) => <React.Fragment key={li}>{li > 0 && <br />}{inline(l, `${bi}-${li}`)}</React.Fragment>)}</p>;
  })}</div>;
}

// ---------- toasts
const ToastCtx = createContext(() => {});
export const useToast = () => useContext(ToastCtx);
export function ToastProvider({ children }) {
  const [items, setItems] = useState([]);
  const push = useCallback((message, tone = "info") => {
    const id = Math.random(); setItems((x) => [...x, { id, message, tone }]);
    setTimeout(() => setItems((x) => x.filter((i) => i.id !== id)), 4500);
  }, []);
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="toasts" aria-live="polite">{items.map((t) => (
        <div key={t.id} className={`toast ${t.tone}`}>{t.tone === "success" ? <CheckCircle2 size={16} /> : t.tone === "error" ? <AlertTriangle size={16} /> : <Info size={16} />}<span>{t.message}</span>
          <button aria-label="Dismiss" onClick={() => setItems((x) => x.filter((i) => i.id !== t.id))}><X size={14} /></button></div>))}</div>
    </ToastCtx.Provider>
  );
}

export class ErrorBoundary extends React.Component {
  state = { error: null };
  static getDerivedStateFromError(error) { return { error }; }
  render() {
    if (!this.state.error) return this.props.children;
    return <div className="fatal"><AlertTriangle size={36} /><h2>Something went wrong</h2><p>The page hit an unexpected error. Your data is safe.</p><button className="primary" onClick={() => location.reload()}>Reload</button></div>;
  }
}
