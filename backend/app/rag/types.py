"""Shared data types for the retrieval pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str          # stable hash of source + url + ordinal
    doc_id: str      # the document this chunk came from
    source: str      # corpus source name, e.g. "mcp"
    title: str       # nearest heading / document title
    url: str         # canonical url to cite
    text: str
    ordinal: int = 0  # position within the document


@dataclass
class Hit:
    chunk: Chunk
    retrieval_score: float          # cosine from the vector store
    rerank_score: float | None = None  # cross-encoder score, set after rerank

    @property
    def score(self) -> float:
        return self.retrieval_score if self.rerank_score is None else self.rerank_score


@dataclass
class Document:
    doc_id: str
    source: str
    title: str
    url: str
    text: str
    meta: dict = field(default_factory=dict)
