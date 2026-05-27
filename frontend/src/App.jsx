import React, { useEffect, useRef, useState } from "react";
import { chat, getCorpus } from "./api.js";
import { AssistantMessage, UserMessage } from "./components.jsx";

const SAMPLES = [
  "How do I define a tool and its input schema in MCP?",
  "What is prompt caching and how do I use it?",
  "How do I handle tool calls Claude returns?",
];

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [useRerank, setUseRerank] = useState(true);
  const [corpus, setCorpus] = useState(null);
  const [error, setError] = useState("");
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme || "dark");
  const endRef = useRef(null);

  useEffect(() => { getCorpus().then(setCorpus); }, []);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, loading]);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("docs-agent-theme", theme); } catch { /* ignore */ }
  }, [theme]);

  async function send(text) {
    text = text.trim();
    if (!text || loading) return;
    const history = messages.map((m) => ({
      role: m.role,
      content: m.role === "assistant" ? m.result.answer : m.text,
    }));
    setMessages((p) => [...p, { role: "user", text }]);
    setInput("");
    setError("");
    setLoading(true);
    try {
      const res = await chat({ message: text, history, useRerank });
      setMessages((p) => [...p, { role: "assistant", result: res }]);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="head">
        <div>
          <h1>docs-agent</h1>
          <p className="sub">
            an agentic RAG assistant over{" "}
            {corpus ? `${corpus.corpus} (${corpus.total_chunks} chunks)` : "the docs"}
          </p>
        </div>
        <div className="head-controls">
          <button
            type="button"
            className="theme-toggle mono"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            title="Switch light / dark"
          >
            {theme === "dark" ? "☀ light" : "☾ dark"}
          </button>
          <label className="rerank mono" title="Compare the pipeline with the cross-encoder reranker on or off">
            <input type="checkbox" checked={useRerank} onChange={(e) => setUseRerank(e.target.checked)} />
            reranker {useRerank ? "on" : "off"}
          </label>
        </div>
      </header>

      <main className="thread">
        {messages.length === 0 && (
          <div className="empty">
            <p>Ask about the Model Context Protocol or the Claude API. Answers cite their sources, and you can open the agent trace to see each step.</p>
            <div className="samples">
              {SAMPLES.map((s) => (
                <button key={s} className="sample" onClick={() => send(s)}>{s}</button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) =>
          m.role === "user"
            ? <UserMessage key={i} text={m.text} />
            : <AssistantMessage key={i} result={m.result} />
        )}
        {loading && <div className="msg assistant loading mono">retrieving and reasoning…</div>}
        {error && <div className="error mono">{error}</div>}
        <div ref={endRef} />
      </main>

      <form className="composer" onSubmit={(e) => { e.preventDefault(); send(input); }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question…"
          autoFocus
        />
        <button type="submit" disabled={loading || !input.trim()}>ask</button>
      </form>
    </div>
  );
}
