import React, { useState } from "react";

// Render answer text, turning [n] markers into links to the cited source.
function CitedText({ text, sources }) {
  const byN = Object.fromEntries(sources.map((s) => [s.n, s]));
  const parts = text.split(/(\[\d+\])/g);
  return (
    <p className="answer">
      {parts.map((part, i) => {
        const m = part.match(/^\[(\d+)\]$/);
        if (m && byN[+m[1]]) {
          const s = byN[+m[1]];
          return (
            <a key={i} className="cite" href={s.url} target="_blank" rel="noreferrer" title={s.title}>
              {m[1]}
            </a>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </p>
  );
}

function Sources({ sources, citations }) {
  if (!sources.length) return null;
  const cited = new Set(citations);
  return (
    <div className="sources">
      <div className="label">sources</div>
      {sources.map((s) => (
        <a key={s.n} className={"source" + (cited.has(s.n) ? " is-cited" : "")}
           href={s.url} target="_blank" rel="noreferrer">
          <span className="src-n">[{s.n}]</span>
          <span className="src-body">
            <span className="src-title">{s.title}</span>
            <span className="src-meta mono">
              {s.source}
              {s.rerank_score != null && <> · ce {s.rerank_score.toFixed(2)}</>}
              {" · cos "}{s.retrieval_score.toFixed(3)}
            </span>
          </span>
        </a>
      ))}
    </div>
  );
}

function Trace({ trace, sources }) {
  const [open, setOpen] = useState(false);
  const byN = Object.fromEntries(sources.map((s) => [s.n, s]));
  return (
    <div className="trace">
      <button className="trace-toggle mono" onClick={() => setOpen(!open)}>
        {open ? "▾" : "▸"} agent trace · {trace.length} step{trace.length === 1 ? "" : "s"}
      </button>
      {open && (
        <ol className="trace-steps">
          {trace.map((st) => (
            <li key={st.index} className="trace-step">
              <div className="step-head mono">
                <span className={"kind kind-" + st.kind}>{st.kind}</span>
                {st.latency_ms > 0 && <span className="lat">{st.latency_ms}ms</span>}
              </div>
              {st.thought && <div className="thought">“{st.thought}”</div>}
              {st.query && <div className="query mono">search: {st.query}</div>}
              {st.retrieved_ns?.length > 0 && (
                <div className="step-sources mono">
                  {st.kind === "answer" ? "cited" : "got"}:{" "}
                  {st.retrieved_ns.map((n) => (
                    <span key={n} className="chip" title={byN[n]?.title}>{n}</span>
                  ))}
                </div>
              )}
              {st.note && <div className="note mono">{st.note}</div>}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

export function AssistantMessage({ result }) {
  return (
    <div className="msg assistant">
      <CitedText text={result.answer} sources={result.sources} />
      <Sources sources={result.sources} citations={result.citations} />
      <Trace trace={result.trace} sources={result.sources} />
      <div className="model mono">{result.model}</div>
    </div>
  );
}

export function UserMessage({ text }) {
  return <div className="msg user">{text}</div>;
}
