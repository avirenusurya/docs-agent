"""Retrieval metrics.

A retrieved chunk counts as relevant if one of the question's gold paths is a
substring of the chunk url. Pages get split into many chunks, so we score on
the first time a relevant page appears in the ranking.
"""
from __future__ import annotations


def is_relevant(url: str, gold: list[str]) -> bool:
    return any(g in url for g in gold)


def first_relevant_rank(urls: list[str], gold: list[str]) -> int | None:
    """1-based rank of the first relevant url, or None if none are relevant."""
    for i, url in enumerate(urls, start=1):
        if is_relevant(url, gold):
            return i
    return None


def hit_at_k(urls: list[str], gold: list[str], k: int) -> float:
    rank = first_relevant_rank(urls[:k], gold)
    return 1.0 if rank is not None else 0.0


def reciprocal_rank(urls: list[str], gold: list[str]) -> float:
    rank = first_relevant_rank(urls, gold)
    return 1.0 / rank if rank else 0.0


def aggregate(per_query: list[dict], ks=(1, 3, 5, 10)) -> dict:
    n = len(per_query) or 1
    out = {f"hit@{k}": round(sum(q[f"hit@{k}"] for q in per_query) / n, 4) for k in ks}
    out["mrr"] = round(sum(q["rr"] for q in per_query) / n, 4)
    return out
