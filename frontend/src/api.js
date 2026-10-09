// Thin fetch wrapper: bearer token from sessionStorage, uniform errors, automatic sign-out on 401.
// Dev (npm run dev): talk to the API on :8000. Production build: same origin (the backend serves this app).
const API = import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? "http://127.0.0.1:8000" : "");
const KEY = "careops_session";

export const session = {
  get() { try { return JSON.parse(sessionStorage.getItem(KEY)); } catch { return null; } },
  set(s) { sessionStorage.setItem(KEY, JSON.stringify(s)); },
  clear() { sessionStorage.removeItem(KEY); },
};

let onUnauthorized = () => {};
export const setUnauthorizedHandler = (fn) => { onUnauthorized = fn; };

export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

export async function api(path, { method = "GET", body, signal } = {}) {
  const s = session.get();
  let res;
  try {
    res = await fetch(API + path, {
      method, signal,
      headers: { "Content-Type": "application/json", ...(s?.token ? { Authorization: `Bearer ${s.token}` } : {}) },
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new ApiError("Can't reach the CareOps server. Check that the backend is running and try again.", 0);
  }
  const data = await res.json().catch(() => ({}));
  if (res.status === 401 && s?.token) { onUnauthorized(); }
  if (!res.ok) {
    const d = data.detail;
    throw new ApiError(typeof d === "string" ? d : d?.message || "Something went wrong.", res.status);
  }
  return data;
}

export const download = async (path, filename) => {
  const s = session.get();
  const res = await fetch(API + path, { headers: { Authorization: `Bearer ${s?.token}` } });
  if (!res.ok) throw new ApiError("Download failed", res.status);
  const url = URL.createObjectURL(await res.blob());
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  a.click(); URL.revokeObjectURL(url);
};
