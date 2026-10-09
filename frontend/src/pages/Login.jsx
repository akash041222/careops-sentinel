import React, { useEffect, useState } from "react";
import { HeartPulse, LogIn, ShieldCheck } from "lucide-react";
import { api } from "../api";
import { ErrorNote } from "../components/ui";

export default function Login({ onLogin }) {
  const [id, setId] = useState(""); const [pw, setPw] = useState(""); const [err, setErr] = useState(""); const [busy, setBusy] = useState(false); const [demo, setDemo] = useState(null);
  useEffect(() => { api("/api/auth/demo-accounts").then(setDemo).catch(() => {}); }, []);
  const submit = async (e) => {
    e.preventDefault(); if (busy) return; setErr(""); setBusy(true);
    try { onLogin(await api("/api/auth/login", { method: "POST", body: { user_id: id.trim(), password: pw } })); }
    catch (x) { setErr(x.message); } finally { setBusy(false); }
  };
  return (
    <div className="login-page">
      <div className="login-hero">
      <div className="logo big">
        <img src="LOGO.jpg" alt="CareOps Sentinel" />
      </div>
        <h1>CareOps Sentinel</h1><p>AI-powered healthcare operations assistant. Grounded answers, guided workflows, and a human always in the loop.</p>
        <ul><li><ShieldCheck size={16} /> No medical advice, no confidential data</li><li><ShieldCheck size={16} /> Every answer cites approved sources</li><li><ShieldCheck size={16} /> Full audit trail of every action</li></ul>
      </div>
      <form className="login-card" onSubmit={submit}>
        <h2>Sign in</h2><span className="muted">Synthetic data only. Do not enter real credentials.</span>
        <label>User ID<input value={id} onChange={(e) => setId(e.target.value)} autoComplete="username" autoFocus placeholder="e.g. EMP-1001" required /></label>
        <label>Password<input type="password" value={pw} onChange={(e) => setPw(e.target.value)} autoComplete="current-password" required /></label>
        <ErrorNote error={err} />
        <button className="primary wide" disabled={busy || !id || !pw}><LogIn size={16} /> {busy ? "Signing in…" : "Sign in"}</button>
        {demo?.enabled && (
          <div className="demo"><b>Demo accounts</b><small>Password for all: <code>{demo.password}</code></small>
            {demo.accounts.map((a) => <button type="button" key={a.user_id} onClick={() => { setId(a.user_id); setPw(demo.password); }}><span>{a.name}</span><small>{a.role_label} · {a.user_id}</small></button>)}</div>)}
      </form>
    </div>
  );
}
