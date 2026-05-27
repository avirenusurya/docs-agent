"""The agent.

A small loop: the model rewrites the question into a search query, reads what
comes back, optionally searches again, then answers with citations. Retrieval
goes through the rerank pipeline. Every step is recorded for the trace. Works
on any chat model because the control flow rides on a JSON protocol, not native
tool calling.
"""
from __future__ import annotations

import re
import time

from app.agent.prompts import parse_action, render_context, system_prompt
from app.agent.trace import AgentResult, Source, Step
from app.config import get_settings
from app.providers.registry import get_llm
from app.rag.retrieve import search as retrieve_search

SNIPPET_CHARS = 500
MAX_CONTEXT_SOURCES = 12
_CITE_RE = re.compile(r"\[(\d+)\]")


def _snippet(text: str, n: int = SNIPPET_CHARS) -> str:
    text = " ".join(text.split())
    return text[:n] + ("..." if len(text) > n else "")


def run_agent(message: str, history: list[dict] | None = None, *, k: int | None = None,
              top_n: int | None = None, use_rerank: bool = True,
              max_steps: int | None = None) -> AgentResult:
    settings = get_settings()
    llm = get_llm()
    max_steps = max_steps or settings.agent_max_steps
    max_searches = max_steps - 1  # leave the last step for answering

    sources: list[Source] = []
    by_chunk: dict[str, Source] = {}
    trace: list[Step] = []

    convo: list[dict] = [{"role": "system", "content": system_prompt(max_searches)}]
    for m in history or []:
        convo.append({"role": m["role"], "content": m["content"]})
    convo.append({"role": "user", "content": message})

    searches = 0
    for i in range(max_steps):
        context = sources[-MAX_CONTEXT_SOURCES:]
        turn = convo + [{"role": "user", "content": render_context(context)}]
        if searches >= max_searches:
            turn.append({"role": "user",
                         "content": "Search budget used up. Reply with the 'answer' action now."})

        t0 = time.time()
        raw = llm.chat(turn)
        latency = int((time.time() - t0) * 1000)
        action = parse_action(raw)
        convo.append({"role": "assistant", "content": raw})

        if not action or action.get("action") not in {"search", "answer"}:
            # The model skipped the JSON envelope and just wrote the answer.
            # That's still an answer: take the text and pull its inline [n] cites.
            text = raw.strip()
            cited = _collect_citations(None, text, sources)
            trace.append(Step(index=i, kind="answer" if text else "fallback",
                              retrieved_ns=cited, latency_ms=latency,
                              note="answered without the JSON envelope" if text else "empty reply"))
            return _finalize(text or "Sorry, I couldn't form an answer.",
                             cited, sources, trace, settings.llm_model)

        if action["action"] == "search":
            query = (action.get("query") or message).strip()
            hits = retrieve_search(query, k=k, top_n=top_n, use_rerank=use_rerank)
            ns = []
            for h in hits:
                existing = by_chunk.get(h.chunk.id)
                if existing is None:
                    src = Source(n=len(sources) + 1, chunk_id=h.chunk.id,
                                 source=h.chunk.source, title=h.chunk.title,
                                 url=h.chunk.url, retrieval_score=round(h.retrieval_score, 4),
                                 rerank_score=round(h.rerank_score, 4) if h.rerank_score is not None else None,
                                 snippet=_snippet(h.chunk.text))
                    sources.append(src)
                    by_chunk[h.chunk.id] = src
                    ns.append(src.n)
                else:
                    ns.append(existing.n)
            searches += 1
            trace.append(Step(index=i, kind="search", thought=action.get("thought", ""),
                              query=query, retrieved_ns=ns, latency_ms=latency))
            convo.append({"role": "user",
                          "content": f"Search '{query}' returned sources {ns}."})
            continue

        # answer
        answer = (action.get("answer") or "").strip()
        cited = _collect_citations(action.get("citations"), answer, sources)
        trace.append(Step(index=i, kind="answer", thought=action.get("thought", ""),
                          retrieved_ns=cited, latency_ms=latency))
        return _finalize(answer, cited, sources, trace, settings.llm_model)

    # Ran out of steps without an answer: make one final attempt.
    turn = convo + [{"role": "user", "content": render_context(sources[-MAX_CONTEXT_SOURCES:])},
                    {"role": "user", "content": "Give your best answer now as plain text with [n] citations."}]
    raw = llm.chat(turn)
    cited = _collect_citations(None, raw, sources)
    trace.append(Step(index=max_steps, kind="fallback", note="answered after step budget"))
    return _finalize(raw.strip(), cited, sources, trace, settings.llm_model)


def _collect_citations(cites, answer: str, sources: list[Source]) -> list[int]:
    valid = {s.n for s in sources}
    out: list[int] = []
    for c in (cites or []):
        try:
            n = int(c)
        except (TypeError, ValueError):
            continue
        if n in valid and n not in out:
            out.append(n)
    for m in _CITE_RE.findall(answer):  # also honor inline [n] markers
        n = int(m)
        if n in valid and n not in out:
            out.append(n)
    return out


def _finalize(answer, citations, sources, trace, model) -> AgentResult:
    return AgentResult(answer=answer, citations=citations, sources=sources,
                       trace=trace, model=model)
