const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function getCorpus() {
  const res = await fetch(`${BASE}/corpus`);
  if (!res.ok) return null;
  return res.json();
}

export async function chat({ message, history, useRerank }) {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history, use_rerank: useRerank }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `request failed (${res.status})`);
  }
  return res.json();
}
