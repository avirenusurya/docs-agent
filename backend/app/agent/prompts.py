"""Prompts and the JSON action protocol.

The agent talks to the model in a small JSON protocol instead of relying on
native function calling, so it works the same on any OpenRouter model including
the free ones. Each turn the model returns one action: search again, or answer.
"""
from __future__ import annotations

import json
import re

from app.agent.trace import Source

SYSTEM = """You are a documentation assistant for the Model Context Protocol \
and the Claude developer docs. You answer using only what you retrieve, and you \
cite your sources.

You work in steps. On each step reply with a single JSON object and nothing \
else. Two actions are available:

1. Search the docs:
   {"thought": "what you still need to find", "action": "search", "query": "a focused search query"}

2. Answer the user:
   {"thought": "why you can answer now", "action": "answer", "answer": "your answer with [n] citations", "citations": [n, ...]}

Rules:
- Start by searching. Rewrite the user's question into a precise query; do not \
just echo it. Search again with a different query if the first results are thin.
- Cite sources inline as [n] using the numbers shown in CONTEXT, and list those \
numbers in "citations".
- If the docs do not cover the question, say so plainly instead of guessing.
- Keep answers tight and concrete. Prefer the exact API or config detail over \
general description. Use short code snippets when they help.
- Do not search more than __MAX_SEARCHES__ times. After that you must answer."""


def system_prompt(max_searches: int) -> str:
    # SYSTEM contains literal JSON braces, so substitute by replace, not format.
    return SYSTEM.replace("__MAX_SEARCHES__", str(max_searches))


def render_context(sources: list[Source]) -> str:
    if not sources:
        return "CONTEXT: (nothing retrieved yet)"
    lines = ["CONTEXT (cite these by number):"]
    for s in sources:
        lines.append(f"\n[{s.n}] {s.title}  <{s.url}>\n{s.snippet}")
    return "\n".join(lines)


def _balanced_objects(text: str) -> list[str]:
    """Return the top-level {...} spans in text, honoring strings so braces
    inside quoted values don't throw off the depth count."""
    spans, depth, start = [], 0, None
    in_str = esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append(text[start : i + 1])
                start = None
    return spans


def extract_json(text: str, require_key: str | None = None) -> dict | None:
    """Return the first balanced JSON object in a model reply, tolerating prose
    and code fences around it. If require_key is given, skip objects that lack
    it (so an answer containing a JSON example isn't mistaken for the action)."""
    text = text.strip()
    for candidate in [text, *_balanced_objects(text)]:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and (require_key is None or require_key in obj):
            return obj
    return None


def parse_action(text: str) -> dict | None:
    """Find the JSON action ({"action": ...}) in a model reply."""
    return extract_json(text, require_key="action")
