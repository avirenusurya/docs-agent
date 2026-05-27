"""Provider interfaces.

Three things vary by environment: the chat model, the embedder, and the
reranker. Each is a small Protocol so a local ONNX backend and a hosted API
backend are interchangeable. The registry decides which concrete class to
build from settings.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLM(Protocol):
    """A chat model. Messages are OpenAI-style dicts: {"role", "content"}."""

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. Query and document embeddings use the same
    model here, but the methods are split so an asymmetric model can override."""

    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder. Scores each (query, document) pair jointly, which is
    what makes it more accurate than the bi-encoder used for first-pass
    retrieval. Returns a score per document, aligned to the input order."""

    def score(self, query: str, documents: list[str]) -> list[float]: ...
