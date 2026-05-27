"""First-pass retrieval: embed the query, pull the nearest chunks."""
from __future__ import annotations

from app.config import get_settings
from app.providers.registry import get_embedder, get_store
from app.rag.types import Hit


def retrieve(query: str, k: int | None = None) -> list[Hit]:
    settings = get_settings()
    k = k or settings.retrieve_k
    query_vec = get_embedder().embed_query(query)
    return get_store().search(query_vec, k)
