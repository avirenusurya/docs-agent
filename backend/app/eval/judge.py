"""LLM-as-judge for answer quality.

Retrieval metrics tell you whether the right page was found. They don't tell you
whether the final answer was correct and grounded. This grades the agent's
answer against the docs it retrieved, on a 1-5 scale, and explains the score.
"""
from __future__ import annotations

from app.agent.prompts import extract_json
from app.providers.registry import get_llm

JUDGE_SYSTEM = """You grade a documentation assistant. You are given a question, \
the assistant's answer, and the documentation snippets it was allowed to use.

Score the answer from 1 to 5:
5 - correct, directly answers the question, fully supported by the snippets
4 - correct and supported, but missing a useful detail
3 - partly correct or partly supported
2 - mostly wrong or largely unsupported by the snippets
1 - wrong, empty, or contradicts the snippets

Reply with one JSON object only: {"score": <1-5>, "reason": "<one sentence>"}"""


def judge_answer(question: str, answer: str, context: list[str]) -> dict:
    ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context)) or "(no context)"
    user = f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nDOC SNIPPETS:\n{ctx}"
    raw = get_llm().chat(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.0,
    )
    parsed = extract_json(raw, require_key="score") or {}
    try:
        score = int(parsed.get("score", 0))
    except (TypeError, ValueError):
        score = 0
    return {"score": score, "reason": str(parsed.get("reason", "")).strip()}
