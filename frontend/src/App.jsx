
import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  BellRing,
  BarChart3,
  Bot,
  ClipboardList,
  HeartPulse,
  Inbox,
  LogOut,
  ScrollText,
  Settings,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";

import { api, session, setUnauthorizedHandler } from "./api";
import { ROLE_LABEL, useToast } from "./components/ui";
import Login from "./pages/Login";
import Assistant from "./pages/Assistant";
import Requests from "./pages/Requests";
import Dashboard from "./pages/Dashboard";
import Audit from "./pages/Audit";
import Admin from "./pages/Admin";

const NAV = {
  employee: [["assistant", "Assistant", Bot], ["requests", "My requests", ClipboardList]],
  support_agent: [["queue", "Queue", Inbox], ["dashboard", "Dashboard", BarChart3], ["assistant", "Assistant", Bot]],
  operations_manager: [["dashboard", "Dashboard", BarChart3], ["queue", "Queue", Inbox], ["audit", "Audit trail", ScrollText], ["assistant", "Assistant", Bot]],
  admin: [["dashboard", "Dashboard", BarChart3], ["queue", "Queue", Inbox], ["audit", "Audit trail", ScrollText], ["admin", "Administration", Settings], ["assistant", "Assistant", Bot]],
};

export default function App() {
  const toast = useToast();
  const [auth, setAuth] = useState(() => session.get());
  const [page, setPage] = useState(null);
  
  // Human escalation alert state
  const [escalationAlerts, setEscalationAlerts] = useState([]);
  const [soundEnabled, setSoundEnabled] = useState(false);
  const [selectedEscalationId, setSelectedEscalationId] = useState(null);
  const [escalationOpenNonce, setEscalationOpenNonce] = useState(0);

  const seenPendingIds = useRef(null);
  const audioContextRef = useRef(null);

  // Poll the existing API for new human-review requests.
  useEffect(() => {
    const user = auth?.user;

    // Only support staff should receive queue-wide escalation alarms.
    if (
      !user ||
      !["support_agent", "operations_manager", "admin"].includes(user.role)
    ) {
      seenPendingIds.current = null;
      setEscalationAlerts([]);
      return;
    }

    let active = true;
    let polling = false;

    const checkEscalations = async () => {
      if (polling) return;
      polling = true;

      try {
        const data = await api(
          "/api/requests?status=pending_human&limit=200"
        );

        if (!active) return;

        const requests = data.items || [];
        const currentIds = new Set(
          requests.map((request) => request.request_id)
        );

        // First poll establishes a baseline. Do not alarm for
        // requests that were already waiting before this page opened.
        if (seenPendingIds.current === null) {
          seenPendingIds.current = currentIds;
          return;
        }

        const previousIds = seenPendingIds.current;

        const newlyEscalated = requests.filter(
          (request) => !previousIds.has(request.request_id)
        );

        if (newlyEscalated.length > 0) {
          setEscalationAlerts((previous) => {
            const alreadyAlerted = new Set(
              previous.map((request) => request.request_id)
            );

            const additions = newlyEscalated.filter(
              (request) => !alreadyAlerted.has(request.request_id)
            );

            return [...previous, ...additions];
          });
        }

        seenPendingIds.current = currentIds;
      } catch (error) {
        // A temporary network error should not crash the application.
        console.warn("Escalation polling failed:", error.message);
      } finally {
        polling = false;
      }
    };

    checkEscalations();

    const timer = window.setInterval(checkEscalations, 8000);

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [auth]);

  // Generate a short alarm tone using the browser's audio API.
  const playEscalationBeep = useCallback(() => {
    const context = audioContextRef.current;

    if (!context || context.state !== "running") return;

    const oscillator = context.createOscillator();
    const gain = context.createGain();

    oscillator.type = "sine";
    oscillator.frequency.value = 880;

    gain.gain.setValueAtTime(0.0001, context.currentTime);
    gain.gain.exponentialRampToValueAtTime(
      0.15,
      context.currentTime + 0.02
    );
    gain.gain.exponentialRampToValueAtTime(
      0.0001,
      context.currentTime + 0.3
    );

    oscillator.connect(gain);
    gain.connect(context.destination);

    oscillator.start();
    oscillator.stop(context.currentTime + 0.32);
  }, []);

  // Keep sounding the alarm until all active alert cards are acknowledged.
  useEffect(() => {
    if (!soundEnabled || escalationAlerts.length === 0) return;

    playEscalationBeep();

    const timer = window.setInterval(playEscalationBeep, 1000);

    return () => window.clearInterval(timer);
  }, [soundEnabled, escalationAlerts.length, playEscalationBeep]);

  // Audio must be initialized by a user gesture in most browsers.
  const enableAlertSound = async () => {
    try {
      const AudioContextClass =
        window.AudioContext || window.webkitAudioContext;

      if (!AudioContextClass) {
        toast("This browser does not support the alert sound.", "error");
        return;
      }

      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContextClass();
      }

      await audioContextRef.current.resume();
      setSoundEnabled(true);

      // Test the sound immediately so staff know it is enabled.
      const context = audioContextRef.current;
      const oscillator = context.createOscillator();
      const gain = context.createGain();

      oscillator.frequency.value = 880;
      gain.gain.setValueAtTime(0.12, context.currentTime);
      gain.gain.exponentialRampToValueAtTime(
        0.0001,
        context.currentTime + 0.25
      );

      oscillator.connect(gain);
      gain.connect(context.destination);
      oscillator.start();
      oscillator.stop(context.currentTime + 0.26);

      toast("Escalation alarm enabled.", "success");
    } catch {
      toast("Could not enable sound. Please try again.", "error");
    }
  };

  const acknowledgeEscalation = (requestId) => {
    setEscalationAlerts((previous) =>
      previous.filter((request) => request.request_id !== requestId)
    );
  };

  const logout = useCallback((expired = false) => {
    if (session.get()) api("/api/auth/logout", { method: "POST" }).catch(() => {});
    session.clear(); setAuth(null); setPage(null);
    if (expired === true) toast("Your session expired. Please sign in again.", "info");
  }, [toast]);
  useEffect(() => { setUnauthorizedHandler(() => logout(true)); }, [logout]);

  if (!auth) return <Login onLogin={(r) => { session.set({ token: r.token, user: r.user }); setAuth({ token: r.token, user: r.user }); setPage(NAV[r.user.role][0][0]); }} />;
  const user = auth.user; const nav = NAV[user.role]; const current = page || nav[0][0];
  const go = (p) => setPage(nav.some(([k]) => k === p) ? p : nav[0][0]);

  return (
    <div className="shell">
      <aside className="sidebar">
      <div className="brand">
        <div className="logo">
          <img src="LOGO.jpg" alt="CareOps Sentinel" />
        </div>
        <div>
          <b>CareOps</b>
          <span>Sentinel</span>
        </div>
      </div>
        <nav aria-label="Main">{nav.map(([k, label, Icon]) => <button key={k} className={current === k ? "active" : ""} onClick={() => setPage(k)} aria-current={current === k ? "page" : undefined}><Icon size={18} />{label}</button>)}</nav>
        
        {["support_agent", "operations_manager", "admin"].includes(user.role) && (
          <button
            className={`alert-sound-toggle ${soundEnabled ? "enabled" : ""}`}
            onClick={() => {
              if (soundEnabled) {
                setSoundEnabled(false);
                toast("Escalation alarm muted.", "info");
              } else {
                enableAlertSound();
              }
            }}
            type="button"
            aria-pressed={soundEnabled}
          >
            {soundEnabled ? <Volume2 size={17} /> : <VolumeX size={17} />}
            {soundEnabled ? "Mute alert sound" : "Enable alert sound"}
          </button>
        )}

        <div className="user-box"><div className="avatar user">{user.name[0]}</div><div><b>{user.name}</b><span>{ROLE_LABEL[user.role]}</span><small>{user.user_id}</small></div>
          <button className="icon-btn" onClick={() => logout()} aria-label="Sign out" title="Sign out"><LogOut size={16} /></button></div>
        <div className="sidebar-note">Synthetic demo data only</div>
      </aside>
      <main>
        {current === "assistant" && <Assistant user={user} navigate={go} />}
        {current === "requests" && <Requests user={user} mode="mine" />}
        {current === "queue" && (
          <Requests
            user={user}
            mode="queue"
            initialRequestId={selectedEscalationId}
            initialRequestNonce={escalationOpenNonce}
          />
        )}
        {current === "dashboard" && <Dashboard user={user} />}
        {current === "audit" && <Audit />}
        {current === "admin" && <Admin />}
      
        </main>

{escalationAlerts.length > 0 && (
  <div
    className="escalation-overlay"
    role="alertdialog"
    aria-modal="true"
    aria-labelledby="escalation-title"
    aria-describedby="escalation-description"
  >
    <section className="escalation-modal">
      <div className="escalation-pulse">
        <AlertTriangle size={42} />
      </div>

      <span className="escalation-eyebrow">
        HUMAN INTERVENTION REQUIRED
      </span>

      <h2 id="escalation-title">New Issue Escalated!</h2>

      <p id="escalation-description">
        A new request is waiting for review by the support team.
      </p>

      <div className="escalation-count">
        {escalationAlerts.length} unacknowledged alert
        {escalationAlerts.length === 1 ? "" : "s"}
      </div>

      <div className="escalation-request-list">
        {escalationAlerts.map((request) => (
          <article
            className="escalation-request"
            key={request.request_id}
          >
            <div>
              <span className="escalation-label">REQUEST ID</span>
              <strong>{request.request_id}</strong>
            </div>

            <div>
              <span className="escalation-label">ISSUE TYPE</span>
              <strong>{request.request_type}</strong>
            </div>

            <div>
              <span className="escalation-label">PRIORITY</span>
              <strong className={`escalation-priority ${request.priority}`}>
                {request.priority}
              </strong>
            </div>

            <div>
              <span className="escalation-label">REQUESTER</span>
              <strong>{request.requester_name || request.user_id}</strong>
            </div>

            <div className="escalation-actions">
              <button
                className="primary"
                onClick={() => {
                  acknowledgeEscalation(request.request_id);
                  setSelectedEscalationId(request.request_id);
                  setEscalationOpenNonce((value) => value + 1);
                  setPage("queue");
                }}
                type="button"
              >
                <BellRing size={16} />
Acknowledge & Open Request
              </button>

              <button
                className="secondary"
                onClick={() =>
                  acknowledgeEscalation(request.request_id)
                }
                type="button"
                aria-label={`Dismiss alert for ${request.request_id}`}
              >
                <X size={16} />
                Dismiss
              </button>
            </div>
          </article>
        ))}
      </div>


      {!soundEnabled && (
        <button
          className="secondary escalation-enable-sound"
          onClick={enableAlertSound}
          type="button"
        >
          <Volume2 size={16} />
          Enable alarm sound
        </button>
      )}

      <p className="escalation-footnote">
        Acknowledging an alert does not resolve the request.
      </p>
    </section>
  </div>
)}
</div>
);
}
