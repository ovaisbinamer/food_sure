// Talks to the Food-Sure FastAPI backend (main.py / model_utils.py).
// Defaults to port 8000 on whichever machine served this page, so it works
// from the PC (localhost) and from a phone on the same Wi-Fi (the PC's IP).
export const API_BASE = import.meta.env.VITE_API_BASE || `http://${window.location.hostname}:8000`;

export async function predict(spice, file) {
  const form = new FormData();
  form.append("spice", spice);
  form.append("file", file);

  let res;
  try {
    res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: form,
      signal: AbortSignal.timeout(30000),
    });
  } catch (err) {
    if (err.name === "TimeoutError" || err.name === "AbortError") {
      throw new Error("The backend took too long to respond. Is it still running?");
    }
    throw new Error(
      "Can't reach the backend. Make sure uvicorn is running and VITE_API_BASE matches its address."
    );
  }

  if (!res.ok) {
    let detail = `Request failed (HTTP ${res.status}).`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }

  return res.json();
}
