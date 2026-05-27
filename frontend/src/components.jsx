import React, { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";

// Collect the plain text under a hast node (used to grab raw code for "copy").
function hastText(node) {
  if (!node) return "";
  if (node.type === "text") return node.value;
  if (node.children) return node.children.map(hastText).join("");
  return "";
}

// A rehype plugin that turns [n] markers into <cite> nodes after the markdown
// is parsed, so it never touches text inside code/pre. Only numbers that match
// a real source become citations; anything else is left as plain text.
function makeRehypeCitations(validNs) {
  const CITE = /\[(\d+)\]/g;

  function splitText(value) {
    const out = [];
    let last = 0;
    CITE.lastIndex = 0;
    let m;
    let matched = false;
    while ((m = CITE.exec(value)) !== null) {
      const n = Number(m[1]);
      if (!validNs.has(n)) continue;
      matched = true;
      if (m.index > last) out.push({ type: "text", value: value.slice(last, m.index) });
      out.push({
        type: "element",
        tagName: "cite",
        properties: {},
        children: [{ type: "text", value: String(n) }],
      });
      last = m.index + m[0].length;
    }
    if (!matched) return null;
    if (last < value.length) out.push({ type: "text", value: value.slice(last) });
    return out;
  }

  function walk(node) {
    if (!node.children) return;
    const next = [];
    for (const child of node.children) {
      if (child.type === "element" && (child.tagName === "code" || child.tagName === "pre")) {
        next.push(child); // leave code blocks and inline code alone
      } else if (child.type === "text") {
        const split = splitText(child.value);
        next.push(...(split || [child]));
      } else {
        walk(child);
        next.push(child);
      }
    }
    node.children = next;
  }

  return () => (tree) => walk(tree);
}

// A [n] citation: superscript link to the source, same behavior as before.
function makeCite(byN) {
  return function Cite({ children }) {
    const n = Number(Array.isArray(children) ? children.join("") : children);
    const s = byN[n];
    if (!s) return <sup className="cite-plain">[{n}]</sup>;
    return (
      <a className="cite" href={s.url} target="_blank" rel="noreferrer" title={s.title}>
        {n}
      </a>
    );
  };
}

// A fenced code block: language label + copy button over the highlighted code.
function CodeBlock({ lang, code, children }) {
  const [copied, setCopied] = useState(false);
  function copy() {
    if (!navigator.clipboard) return;
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    });
  }
  return (
    <div className="code-block">
      <div className="code-bar mono">
        <span className="code-lang">{lang || "code"}</span>
        <button type="button" className="code-copy" onClick={copy}>
          {copied ? "copied" : "copy"}
        </button>
      </div>
      <pre className="code-pre">{children}</pre>
    </div>
  );
}

// react-markdown maps the <pre> wrapper; we read language + raw text off the
// hast node and let the highlighted <code> render as our children.
function Pre({ node, children }) {
  const codeNode = node?.children?.find((c) => c.type === "element" && c.tagName === "code");
  const className = (codeNode?.properties?.className || []).join(" ");
  const lang = (className.match(/language-([\w+#-]+)/) || [])[1] || "";
  return (
    <CodeBlock lang={lang} code={hastText(codeNode)}>
      {children}
    </CodeBlock>
  );
}

// Inline code gets the chip; highlighted block code (has hljs/language-) passes
// through untouched so the <pre> wrapper styles it.
function Code({ className, children, ...rest }) {
  if (className && /\b(hljs|language-)/.test(className)) {
    return (
      <code className={className} {...rest}>
        {children}
      </code>
    );
  }
  return <code className="inline-code">{children}</code>;
}

function AnswerMarkdown({ text, sources }) {
  const byN = useMemo(() => Object.fromEntries(sources.map((s) => [s.n, s])), [sources]);
  const Cite = useMemo(() => makeCite(byN), [byN]);
  const rehypeCitations = useMemo(
    () => makeRehypeCitations(new Set(sources.map((s) => s.n))),
    [sources]
  );
  return (
    <div className="answer">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[[rehypeHighlight, { detect: true, ignoreMissing: true }], rehypeCitations]}
        components={{ cite: Cite, pre: Pre, code: Code }}
      >
        {text}
      </ReactMarkdown>
    </div>
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
      <AnswerMarkdown text={result.answer} sources={result.sources} />
      <Sources sources={result.sources} citations={result.citations} />
      <Trace trace={result.trace} sources={result.sources} />
      <div className="model mono">{result.model}</div>
    </div>
  );
}

export function UserMessage({ text }) {
  return <div className="msg user">{text}</div>;
}
