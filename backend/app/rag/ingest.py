"""Ingest a corpus into the vector store.

    python -m app.rag.ingest                 # full ingest, resets the store
    python -m app.rag.ingest --no-reset      # add to what's there
    python -m app.rag.ingest --doc-limit 20  # quick smoke run

Steps: load sources -> chunk -> embed (batched) -> store. Writes a manifest
with counts and the embedding model so the API can report what's loaded.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import islice
from pathlib import Path

from app.config import DATA_DIR, get_settings
from app.providers.registry import get_embedder, get_store
from app.rag.chunk import chunk_document
from app.rag.corpus import documents_for, load_sources

MANIFEST_PATH = DATA_DIR / "manifest.json"
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / "corpus.yaml"


def _batched(seq, n):
    it = iter(seq)
    while batch := list(islice(it, n)):
        yield batch


def run_ingest(config_path=DEFAULT_CONFIG, reset=True, batch_size=128, doc_limit=None):
    settings = get_settings()
    store = get_store()
    embedder = get_embedder()

    if reset:
        store.reset()

    corpus_name, sources = load_sources(config_path)
    per_source = []
    for src in sources:
        docs = documents_for(src)
        if doc_limit:
            docs = docs[:doc_limit]
        chunks = []
        for d in docs:
            chunks.extend(chunk_document(d, settings.chunk_tokens, settings.chunk_overlap))
        for batch in _batched(chunks, batch_size):
            embeddings = embedder.embed_documents([c.text for c in batch])
            store.add(batch, embeddings)
        per_source.append({"name": src.name, "docs": len(docs), "chunks": len(chunks)})
        print(f"  {src.name}: {len(docs)} docs -> {len(chunks)} chunks")

    manifest = {
        "corpus": corpus_name,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedding_model": settings.embedding_model_local
        if settings.embedding_provider == "local"
        else settings.embedding_model_hosted,
        "embedding_dim": settings.embedding_dim,
        "vector_store": settings.vector_store,
        "total_chunks": store.count(),
        "sources": per_source,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"done: {store.count()} chunks from {corpus_name}")
    return manifest


def load_manifest() -> dict | None:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--no-reset", action="store_true")
    ap.add_argument("--doc-limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    run_ingest(args.config, reset=not args.no_reset,
               batch_size=args.batch_size, doc_limit=args.doc_limit)
