"""FastAPI entry point."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agent.loop import run_agent
from app.config import get_settings
from app.providers.base import LLMError
from app.providers.registry import describe_providers, get_store
from app.rag.corpus import load_sources
from app.rag.ingest import DEFAULT_CONFIG, load_manifest

settings = get_settings()

app = FastAPI(title="docs-agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    use_rerank: bool = True
    k: int | None = None
    top_n: int | None = None
    max_steps: int | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/config")
def config() -> dict:
    """Which providers this instance is wired to. No secrets returned."""
    return describe_providers(settings)


@app.get("/corpus")
def corpus() -> dict:
    """What's currently indexed. Prefers the manifest written by the local
    ingest; falls back to a live read of the store + corpus.yaml so the
    deployed instance (which never runs the ingest) still reports correctly."""
    manifest = load_manifest()
    if manifest:
        return manifest
    try:
        count = get_store().count()
    except Exception:
        raise HTTPException(503, "vector store unavailable")
    if count == 0:
        raise HTTPException(404, "no corpus ingested yet; run `python -m app.rag.ingest`")
    name, _ = load_sources(DEFAULT_CONFIG)
    return {
        "corpus": name,
        "embedding_model": settings.embedding_model_hosted
        if settings.embedding_provider == "jina"
        else settings.embedding_model_local,
        "embedding_dim": settings.embedding_dim,
        "vector_store": settings.vector_store,
        "total_chunks": count,
    }


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    if not req.message.strip():
        raise HTTPException(400, "message is empty")
    try:
        result = run_agent(
            req.message,
            history=[m.model_dump() for m in req.history],
            k=req.k,
            top_n=req.top_n,
            use_rerank=req.use_rerank,
            max_steps=req.max_steps,
        )
    except LLMError as e:  # upstream model error (rate limit, etc.); relay status
        raise HTTPException(e.status, str(e))
    except RuntimeError as e:  # e.g. missing API key
        raise HTTPException(503, str(e))
    return result.to_dict()
