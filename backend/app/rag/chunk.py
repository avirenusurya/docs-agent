"""Parsing and chunking.

The default corpus comes as llms-full.txt files: a flat dump where each page
is a markdown header followed by a `Source: <url>` line and the page body. We
split that back into documents, then window each document into overlapping
chunks at paragraph boundaries so a chunk rarely cuts mid-thought.
"""
from __future__ import annotations

import hashlib
import re

from app.rag.types import Chunk, Document

_SOURCE_RE = re.compile(r"^\s*source:\s*(https?://\S+)\s*$", re.IGNORECASE)


def _is_header(line: str) -> bool:
    return line.lstrip().startswith("#")


def _chunk_id(source: str, url: str, ordinal: int) -> str:
    raw = f"{source}|{url}|{ordinal}".encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def parse_llms_full(raw: str, source: str) -> list[Document]:
    """Split an llms-full.txt dump into per-page documents using the
    `Source:` markers. Falls back to a single document if there are none."""
    lines = raw.splitlines()
    markers = [i for i, ln in enumerate(lines) if _SOURCE_RE.match(ln)]
    if not markers:
        return [Document(doc_id=_chunk_id(source, source, 0), source=source,
                         title=source, url="", text=raw.strip())]

    docs: list[Document] = []
    for j, i in enumerate(markers):
        url = _SOURCE_RE.match(lines[i]).group(1)
        # Title = nearest header just above the Source line.
        title = url.rstrip("/").rsplit("/", 1)[-1]
        for k in range(i - 1, max(-1, i - 6), -1):
            if _is_header(lines[k]):
                title = lines[k].lstrip("#").strip() or title
                break
        # Body runs until the header that opens the next page.
        if j + 1 < len(markers):
            end = markers[j + 1]
            for k in range(markers[j + 1] - 1, i, -1):
                if _is_header(lines[k]):
                    end = k
                    break
        else:
            end = len(lines)
        body = "\n".join(lines[i + 1:end]).strip()
        if body:
            docs.append(Document(doc_id=_chunk_id(source, url, 0), source=source,
                                 title=title, url=url, text=body))
    return docs


def _window(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Greedily pack paragraphs up to max_chars, carrying a tail for overlap.
    A single oversized paragraph is hard-split."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        while len(p) > max_chars:  # paragraph bigger than a whole window
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:max_chars])
            p = p[max_chars - overlap_chars:]
        if cur and len(cur) + len(p) + 2 > max_chars:
            chunks.append(cur)
            tail = cur[-overlap_chars:] if overlap_chars else ""
            cur = (tail + "\n\n" + p).strip()
        else:
            cur = (cur + "\n\n" + p) if cur else p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def chunk_document(doc: Document, max_tokens: int, overlap: int) -> list[Chunk]:
    # ~4 chars per token is a decent rule of thumb for English prose.
    max_chars = max_tokens * 4
    overlap_chars = overlap * 4
    out: list[Chunk] = []
    for ordinal, piece in enumerate(_window(doc.text, max_chars, overlap_chars)):
        # Prepend the page title so a standalone chunk keeps its context.
        text = f"{doc.title}\n\n{piece}" if doc.title else piece
        out.append(Chunk(
            id=_chunk_id(doc.source, doc.url or doc.doc_id, ordinal),
            doc_id=doc.doc_id,
            source=doc.source,
            title=doc.title,
            url=doc.url,
            text=text,
            ordinal=ordinal,
        ))
    return out
