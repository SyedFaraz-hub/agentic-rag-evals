"""Evaluation harness for the agent.

Metrics (per golden case, then aggregated):
  * retrieval_hit  - an expected document appears in the retrieved results
  * mrr            - reciprocal rank of the first expected document
  * answer_match   - every expected substring appears in the final answer
  * grounded       - every citation points at a document the tools returned
  * tool_accuracy  - the expected tools were called
Exit code is non-zero when the overall pass rate is below --threshold, so CI can
block regressions.

Usage: python -m evals.run_evals [--threshold 0.9] [--json report.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.agent import Agent
from app.llm import MockLLM
from app.main import build_llm, load_index
from app.tools import ToolRegistry

GOLDEN = Path(__file__).with_name("golden.jsonl")


def retrieved_docs(result) -> list[str]:
    docs: list[str] = []
    for s in result.steps:
        if s.kind == "tool_result":
            docs += [r["doc_id"] for r in s.detail["result"].get("results", [])]
    return docs


def evaluate(agent: Agent, cases: list[dict]) -> dict:
    rows = []
    for case in cases:
        res = agent.run(case["question"])
        docs = retrieved_docs(res)
        expected = case["expected_docs"]
        hit = (not expected) or any(d in docs for d in expected)
        rr = 1.0 if not expected else next(
            (1.0 / (i + 1) for i, d in enumerate(dict.fromkeys(docs)) if d in expected), 0.0)
        match = all(s.lower() in res.answer.lower() for s in case["expected_substrings"])
        grounded = all(c in docs for c in res.citations)
        tools_ok = all(t in res.tools_used for t in case["expected_tools"])
        rows.append({
            "id": case["id"], "retrieval_hit": hit, "mrr": rr, "answer_match": match,
            "grounded": grounded, "tool_accuracy": tools_ok,
            "passed": hit and match and grounded and tools_ok,
            "latency_ms": res.latency_ms, "llm_calls": res.llm_calls,
        })
    n = len(rows)
    agg = {
        "cases": n,
        "pass_rate": sum(r["passed"] for r in rows) / n,
        "retrieval_hit_rate": sum(r["retrieval_hit"] for r in rows) / n,
        "mrr": sum(r["mrr"] for r in rows) / n,
        "answer_match_rate": sum(r["answer_match"] for r in rows) / n,
        "grounded_rate": sum(r["grounded"] for r in rows) / n,
        "tool_accuracy": sum(r["tool_accuracy"] for r in rows) / n,
        "avg_latency_ms": sum(r["latency_ms"] for r in rows) / n,
    }
    return {"summary": agg, "cases": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--live", action="store_true", help="use OpenAI if OPENAI_API_KEY is set")
    args = ap.parse_args()
    cases = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    llm = build_llm() if args.live else MockLLM()
    report = evaluate(Agent(llm, ToolRegistry(load_index())), cases)
    for r in report["cases"]:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['id']}")
    print(json.dumps(report["summary"], indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    return 0 if report["summary"]["pass_rate"] >= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
