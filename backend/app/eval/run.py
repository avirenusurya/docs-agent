"""Run the eval and print a report.

    python -m app.eval.run                 # retrieval metrics, reranker on vs off
    python -m app.eval.run --judge         # also grade end-to-end answers (needs an LLM key)

The retrieval section is the point of the reranker: it shows recall@k and MRR
with the cross-encoder on versus off, so an improvement is a number, not a hunch.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from app.config import DATA_DIR, get_settings
from app.eval import metrics
from app.rag.retrieve import rerank, retrieve

TESTSET = Path(__file__).resolve().parent / "testset.yaml"
REPORT = DATA_DIR / "eval_report.json"


def _per_query(urls, gold, ks=(1, 3, 5, 10)) -> dict:
    row = {f"hit@{k}": metrics.hit_at_k(urls, gold, k) for k in ks}
    row["rr"] = metrics.reciprocal_rank(urls, gold)
    return row


def run_retrieval_eval(testset: list[dict]) -> dict:
    settings = get_settings()
    k = settings.retrieve_k
    vec_rows, rer_rows, details = [], [], []
    for item in testset:
        gold = item["gold"]
        hits = retrieve(item["question"], k=k)
        vec_urls = [h.chunk.url for h in hits]
        rer_urls = [h.chunk.url for h in rerank(item["question"], list(hits), top_n=k)]
        vrow, rrow = _per_query(vec_urls, gold), _per_query(rer_urls, gold)
        vec_rows.append(vrow)
        rer_rows.append(rrow)
        details.append({
            "id": item["id"],
            "vector_rank": metrics.first_relevant_rank(vec_urls, gold),
            "reranked_rank": metrics.first_relevant_rank(rer_urls, gold),
        })
    return {
        "candidates_per_query": k,
        "vector_only": metrics.aggregate(vec_rows),
        "with_reranker": metrics.aggregate(rer_rows),
        "per_query": details,
    }


def run_judge_eval(testset: list[dict]) -> dict:
    from app.agent.loop import run_agent
    from app.eval.judge import judge_answer

    rows = []
    for item in testset:
        res = run_agent(item["question"])
        verdict = judge_answer(item["question"], res.answer, [s.snippet for s in res.sources])
        rows.append({"id": item["id"], "score": verdict["score"],
                     "reason": verdict["reason"],
                     "n_sources": len(res.sources), "n_steps": len(res.trace)})
    scored = [r["score"] for r in rows if r["score"] > 0]
    return {"avg_score": round(sum(scored) / len(scored), 3) if scored else 0.0,
            "graded": len(scored), "per_query": rows}


def _fmt_compare(title: str, vec: dict, rer: dict) -> str:
    keys = ["hit@1", "hit@3", "hit@5", "hit@10", "mrr"]
    lines = [title, f"  {'metric':<10}{'vector':>10}{'+rerank':>10}{'delta':>10}"]
    for key in keys:
        d = rer[key] - vec[key]
        lines.append(f"  {key:<10}{vec[key]:>10.3f}{rer[key]:>10.3f}{d:>+10.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true", help="also run the LLM-as-judge answer eval")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    testset = yaml.safe_load(TESTSET.read_text())
    if args.limit:
        testset = testset[: args.limit]

    report = {"built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "n_questions": len(testset)}

    print(f"Retrieval eval over {len(testset)} questions")
    retr = run_retrieval_eval(testset)
    report["retrieval"] = retr
    print(_fmt_compare("", retr["vector_only"], retr["with_reranker"]))

    if args.judge:
        print("\nLLM-as-judge (running the agent on each question)...")
        report["judge"] = run_judge_eval(testset)
        print(f"  avg answer score: {report['judge']['avg_score']} / 5 "
              f"({report['judge']['graded']} graded)")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
