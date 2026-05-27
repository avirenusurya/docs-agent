"""In-process vector store for local dev.

Chunks and their embeddings are kept in memory and mirrored to a jsonl file so
an ingest survives a restart. Search is exact (brute-force cosine over a numpy
matrix), which is plenty fast for a demo-sized corpus and keeps dev dependency
on a running database at zero.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from app.rag.store.base import VectorStore
from app.rag.types import Chunk, Hit


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class LocalVectorStore(VectorStore):
    def __init__(self, path: str):
        self.path = Path(path)
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None  # normalized, shape (n, dim)
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        vecs: list[list[float]] = []
        with self.path.open() as f:
            for line in f:
                rec = json.loads(line)
                vecs.append(rec.pop("embedding"))
                self._chunks.append(Chunk(**rec))
        if vecs:
            self._matrix = _normalize(np.array(vecs, dtype=np.float32))

    def reset(self) -> None:
        self._chunks = []
        self._matrix = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            for chunk, emb in zip(chunks, embeddings):
                rec = asdict(chunk)
                rec["embedding"] = emb
                f.write(json.dumps(rec) + "\n")
        self._chunks.extend(chunks)
        new = _normalize(np.array(embeddings, dtype=np.float32))
        self._matrix = new if self._matrix is None else np.vstack([self._matrix, new])

    def search(self, query_embedding: list[float], k: int) -> list[Hit]:
        if self._matrix is None or not self._chunks:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        sims = self._matrix @ q
        k = min(k, len(self._chunks))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [Hit(chunk=self._chunks[i], retrieval_score=float(sims[i])) for i in idx]

    def count(self) -> int:
        return len(self._chunks)
