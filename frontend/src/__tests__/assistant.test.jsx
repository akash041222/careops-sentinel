import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import Assistant from "../pages/Assistant";
import { ToastProvider } from "../components/ui";

const user = { user_id: "EMP-1001", name: "Asha Kumar", role: "employee" };
const reply = (over = {}) => ({ conversation_id: "CNV-1", message_id: 7, kind: "smalltalk", reply: "Hello Asha! I'm the CareOps Assistant.", intent: null, confidence: null,
  sources: [], suggestions: [{ label: "Book a meeting room", kind: "message", message: "I need to book a meeting room", style: "secondary" }], escalation: { required: false, reasons: [] },
  safety: { flags: [] }, trace: [], workflow: null, ...over });

beforeEach(() => {
  globalThis.fetch = vi.fn(async (url, opts) => ({ ok: true, status: 200, json: async () => (String(url).includes("/api/chat") ? reply() : { items: [] }) }));
});
const setup = () => render(<ToastProvider><Assistant user={user} navigate={() => {}} /></ToastProvider>);

describe("chat box", () => {
  it("sends on Enter, clears the input and shows the reply to 'hi'", async () => {
    const u = userEvent.setup(); setup();
    const box = screen.getByLabelText("Message");
    await u.type(box, "hi{Enter}");
    expect(box).toHaveValue("");                                   // the original bug: text stayed in the box
    expect(await screen.findByText(/Hello Asha/)).toBeInTheDocument();
    expect(screen.getByText("hi")).toBeInTheDocument();            // user bubble is shown
    const call = fetch.mock.calls.find(([u2]) => String(u2).includes("/api/chat"));
    expect(JSON.parse(call[1].body).message).toBe("hi");
  });
  it("Shift+Enter inserts a newline instead of sending", async () => {
    const u = userEvent.setup(); setup();
    const box = screen.getByLabelText("Message");
    await u.type(box, "line1{Shift>}{Enter}{/Shift}line2");
    expect(box).toHaveValue("line1\nline2");
    expect(fetch.mock.calls.some(([x]) => String(x).includes("/api/chat"))).toBe(false);
  });
  it("does not send empty or whitespace messages", async () => {
    const u = userEvent.setup(); setup();
    await u.type(screen.getByLabelText("Message"), "   {Enter}");
    expect(fetch.mock.calls.some(([x]) => String(x).includes("/api/chat"))).toBe(false);
    expect(screen.getByLabelText("Send message")).toBeDisabled();
  });
  it("clicking a suggestion chip sends it", async () => {
    const u = userEvent.setup(); setup();
    await u.type(screen.getByLabelText("Message"), "hi{Enter}");
    await u.click(await screen.findByRole("button", { name: "Book a meeting room" }));
    await waitFor(() => expect(fetch.mock.calls.filter(([x]) => String(x).includes("/api/chat")).length).toBe(2));
  });
  it("shows a retryable error when the server is down", async () => {
    fetch.mockImplementation(async (url) => { if (String(url).includes("/api/chat")) throw new TypeError("fail"); return { ok: true, status: 200, json: async () => ({ items: [] }) }; });
    const u = userEvent.setup(); setup();
    await u.type(screen.getByLabelText("Message"), "hi{Enter}");
    expect(await screen.findByRole("alert")).toHaveTextContent(/Can't reach/);
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });
});
