"""FastAPI entry point.

For now this exposes health and a config summary. Retrieval, the agent /chat
endpoint, and the eval routes get wired in as the later phases land.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.providers.registry import describe_providers

settings = get_settings()

app = FastAPI(title="docs-agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/config")
def config() -> dict:
    """Which providers this instance is wired to. No secrets returned."""
    return describe_providers(settings)
