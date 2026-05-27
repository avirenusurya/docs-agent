"""Retrieval pipeline: first-pass vector search, then cross-encoder rerank.

`retrieve` is the bi-encoder first pass. `rerank` rescores those candidates with
the cross-encoder and keeps the best. `search` runs both and is what the agent
calls; pass rerank=False to measure the pipeline without the reranker.
"""
from __future__ import annotations

from app.config import get_settings
from app.providers.registry import get_embedder, get_reranker, get_store
from app.rag.types import Hit


def retrieve(query: str, k: int | None = None) -> list[Hit]:
    settings = get_settings()
    k = k or settings.retrieve_k
    query_vec = get_embedder().embed_query(query)
    return get_store().search(query_vec, k)


def rerank(query: str, hits: list[Hit], top_n: int | None = None) -> list[Hit]:
    if not hits:
        return []
    top_n = top_n or get_settings().rerank_top_n
    scores = get_reranker().score(query, [h.chunk.text for h in hits])
    for hit, score in zip(hits, scores):
        hit.rerank_score = score
    ranked = sorted(hits, key=lambda h: h.rerank_score, reverse=True)
    return ranked[:top_n]


def search(query: str, k: int | None = None, top_n: int | None = None,
           use_rerank: bool = True) -> list[Hit]:
    hits = retrieve(query, k)
    if use_rerank:
        return rerank(query, hits, top_n)
    top_n = top_n or get_settings().rerank_top_n
    return hits[:top_n]
