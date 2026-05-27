"""Vector store interface.

Two backends implement this: an in-process store for local dev (no services to
run) and pgvector on Postgres/Supabase for the deployed instance. Both return
cosine similarity as the retrieval score, higher is better.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.rag.types import Chunk, Hit


@runtime_checkable
class VectorStore(Protocol):
    def reset(self) -> None:
        """Drop everything. Called at the start of a fresh ingest."""
        ...

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    def search(self, query_embedding: list[float], k: int) -> list[Hit]: ...

    def count(self) -> int: ...
