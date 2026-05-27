"""Corpus loading.

Sources are declared in corpus.yaml and come in three shapes:
  - llms-full : one big file of `Source:`-tagged pages (the MCP docs)
  - index     : an llms.txt index listing per-page .md urls (the Claude docs);
                we fetch each page, filtered by an allow/deny list
  - markdown  : a local file or directory of .md / .txt files
Downloads are cached under data/raw so re-ingesting doesn't refetch.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx
import yaml

from app.config import DATA_DIR
from app.rag.chunk import _chunk_id, parse_llms_full
from app.rag.types import Document

RAW_DIR = DATA_DIR / "raw"
_MD_URL_RE = re.compile(r"https://\S+?\.md")


@dataclass
class Source:
    name: str
    format: str
    url: str | None = None
    path: str | None = None
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    max_docs: int | None = None


def load_sources(config_path: str | Path) -> tuple[str, list[Source]]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    name = (cfg.get("corpus") or {}).get("name", "corpus")
    return name, [Source(**s) for s in cfg["sources"]]


def _cached(path: Path, fetch: Callable[[], str]) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    text = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def _keep(url: str, allow: list[str], deny: list[str]) -> bool:
    u = url.lower()
    if deny and any(d in u for d in deny):
        return False
    if allow and not any(a in u for a in allow):
        return False
    return True


def _markdown_doc(raw: str, source: str, url: str) -> Document:
    title = url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".md")
    for line in raw.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    return Document(doc_id=_chunk_id(source, url, 0), source=source,
                    title=title, url=url, text=raw.strip())


def documents_for(src: Source, cache_dir: Path = RAW_DIR) -> list[Document]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=60.0,
                      headers={"User-Agent": "docs-agent/0.1"}) as client:
        def fetch(u: str) -> str:
            r = client.get(u)
            r.raise_for_status()
            return r.text

        if src.format == "llms-full":
            raw = _cached(cache_dir / f"{src.name}.txt", lambda: fetch(src.url))
            docs = [d for d in parse_llms_full(raw, src.name)
                    if _keep(d.url, src.allow, src.deny)]

        elif src.format == "index":
            index = _cached(cache_dir / f"{src.name}-index.txt", lambda: fetch(src.url))
            urls = sorted({u for u in _MD_URL_RE.findall(index)
                           if _keep(u, src.allow, src.deny)})
            if src.max_docs:
                urls = urls[: src.max_docs]
            docs = []
            for u in urls:
                key = hashlib.sha1(u.encode()).hexdigest()[:16]
                raw = _cached(cache_dir / f"{src.name}-{key}.md", lambda u=u: fetch(u))
                docs.append(_markdown_doc(raw, src.name, u))

        elif src.format == "markdown":
            base = Path(src.path)
            files = sorted(base.rglob("*.md")) + sorted(base.rglob("*.txt")) \
                if base.is_dir() else [base]
            docs = [_markdown_doc(p.read_text(encoding="utf-8"), src.name, str(p))
                    for p in files]
        else:
            raise ValueError(f"unknown source format: {src.format}")

    if src.max_docs:
        docs = docs[: src.max_docs]
    return docs
