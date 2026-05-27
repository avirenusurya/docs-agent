"""Trace types.

Every chat request returns the steps the agent took, so the UI (and a curious
engineer) can see why it answered the way it did: what it searched for, which
chunks came back, how the reranker scored them, and which sources it cited.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Source:
    n: int               # citation number shown to the model and the user
    chunk_id: str
    source: str          # corpus source, e.g. "mcp"
    title: str
    url: str
    retrieval_score: float
    rerank_score: float | None
    snippet: str


@dataclass
class Step:
    index: int
    kind: str            # "search" | "answer" | "fallback" | "error"
    thought: str = ""
    query: str | None = None       # the (possibly rewritten) search query
    retrieved_ns: list[int] = field(default_factory=list)  # sources from this step
    latency_ms: int = 0
    note: str = ""


@dataclass
class AgentResult:
    answer: str
    citations: list[int]           # source numbers the answer relies on
    sources: list[Source]          # everything retrieved across the turn
    trace: list[Step]
    model: str

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "sources": [asdict(s) for s in self.sources],
            "trace": [asdict(s) for s in self.trace],
            "model": self.model,
        }
