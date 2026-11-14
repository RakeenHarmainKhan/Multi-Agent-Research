"""
Evaluation harness. Runs the real app (real LLM and search from backend/.env) and writes
evals/results/report.md and report.json.

    python -m evals.run                       # everything (slow: about 30-40 minutes)
    python -m evals.run --only guardrails     # guardrails | intent | research
    python -m evals.run --limit 3 --no-judge  # quick smoke test

Latency is only clean with --workers 1 (the default); more workers is faster but inflates it.
"""
import argparse
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

from app.agents.nodes import WRITER_EXCERPT_CHARS, input_guard_node, intent_classifier_node
from app.agents.workflow import research_graph
from app.core.usage import TokenUsageHandler
from evals import judge
from evals.metrics import (
    citation_coverage,
    cited_ids,
    claim_sentences,
    confusion,
    percentile,
    rate,
    score_band,
    source_stats,
    structure_ok,
    word_count,
)

HERE = Path(__file__).parent
MAX_CLAIMS_JUDGED = 8  # per report, to bound judge cost
MAX_SOURCES_JUDGED = 8
EXCERPT_CHARS = WRITER_EXCERPT_CHARS  # the judge sees exactly what the writer saw


def run_all(fn: Callable, items: List[Any], workers: int) -> List[Any]:
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(fn, items))


def sample(items: List[Any], limit: int) -> List[Any]:
    """Evenly spaced subset, so a small --limit still mixes the groups."""
    if not limit or limit >= len(items):
        return items
    return items[:: max(1, len(items) // limit)][:limit]


# --- Guardrails and intent -----------------------------------------------------------------------


def classify(query: str) -> str:
    """What the app decides for a query without searching: BLOCKED, CHAT, RESEARCH, REJECTED or ERROR."""
    try:
        state = {"query": query, "messages": [], "mode": "auto"}
        state.update(input_guard_node(state))
        if state["guardrails"]["blocked"]:
            return "BLOCKED"
        decision = intent_classifier_node(state)
        return "ERROR" if "error" in decision else decision["intent"]
    except Exception:
        return "ERROR"


def eval_guardrails(cases: List[Dict[str, Any]], workers: int) -> Dict[str, Any]:
    rows = [{**c, "decision": d} for c, d in zip(cases, run_all(lambda c: classify(c["query"]), cases, workers))]
    scored = [r for r in rows if r["decision"] != "ERROR"]
    harmful = [r for r in scored if r["harmful"] is True]
    benign = [r for r in scored if r["harmful"] is False]

    def blocked(r: Dict[str, Any]) -> bool:
        return r["decision"] == "BLOCKED"

    categories: Dict[str, List[bool]] = {}
    for r in harmful:
        categories.setdefault(r["category"], []).append(blocked(r))
    return {
        "harmful": len(harmful),
        "benign": len(benign),
        "block_rate": rate(sum(map(blocked, harmful)), len(harmful)),
        "false_positive_rate": rate(sum(map(blocked, benign)), len(benign)),
        "block_rate_by_category": {c: rate(sum(v), len(v)) for c, v in categories.items()},
        "missed": [r["query"] for r in harmful if not blocked(r)],
        "wrongly_blocked": [r["query"] for r in benign if blocked(r)],
        "borderline": {r["query"]: r["decision"] for r in scored if r["harmful"] is None},
        "errored": [r["query"] for r in rows if r["decision"] == "ERROR"],
    }


def eval_intent(cases: List[Dict[str, Any]], workers: int) -> Dict[str, Any]:
    predicted = run_all(lambda c: classify(c["query"]), cases, workers)
    pairs = [(c["intent"], p) for c, p in zip(cases, predicted) if p != "ERROR"]
    return {
        "scored": len(pairs),
        "accuracy": rate(sum(1 for e, p in pairs if e == p), len(pairs)),
        "confusion": confusion(pairs),
        "wrong": [{"query": c["query"], "expected": c["intent"], "got": p} for c, p in zip(cases, predicted) if p not in (c["intent"], "ERROR")],
        "errors": predicted.count("ERROR"),
    }


# --- Research quality ----------------------------------------------------------------------------


def research_one(question: str, use_judge: bool) -> Dict[str, Any]:
    usage = TokenUsageHandler()
    start = time.time()
    try:
        state = research_graph.invoke(
            {"query": question, "mode": "research", "messages": []},
            config={"configurable": {"thread_id": str(uuid.uuid4())}, "callbacks": [usage]},
        )
    except Exception as exc:
        return {"question": question, "error": str(exc)[:200]}

    report = state.get("final_report", "")
    sources, citations = state.get("all_sources", []), state.get("citations", [])
    cited, claims = citation_coverage(report)
    row: Dict[str, Any] = {
        "question": question,
        "plan": state.get("plan", []),
        "report": report,
        "latency": round(time.time() - start, 1),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "words": word_count(report),
        "structure_ok": structure_ok(report),
        "claims": claims,
        "cited_claims": cited,
        "sources": source_stats(sources, citations),
        "fallback": bool(state.get("error")),
    }
    if state.get("guardrails", {}).get("blocked") or report.startswith("Error"):
        return {**row, "error": report[:200]}

    if use_judge:
        by_id = {s["id"]: s for s in sources}
        cited_claims = [c for c in claim_sentences(report) if cited_ids(c)][:MAX_CLAIMS_JUDGED]
        row["claim_judgments"] = []
        for claim in cited_claims:
            excerpts = [by_id[i]["content"][:EXCERPT_CHARS] for i in cited_ids(claim) if i in by_id]
            # excerpts are stored so a human can check the verdict by hand
            row["claim_judgments"].append({"claim": claim, "excerpts": excerpts, **judge.claim_support(claim, excerpts)})
        row["sources_judged"] = [
            {
                "type": s["type"],
                "url": s.get("url", ""),
                "score": s.get("score"),
                "relevant": judge.source_relevant(question, s["content"][:EXCERPT_CHARS]),
            }
            for s in sources[:MAX_SOURCES_JUDGED]
        ]
        row["answer_relevance"] = judge.answer_relevance(question, report)
    return row


def _precision(items: List[Dict[str, Any]]) -> float:
    judged = [s["relevant"] for s in items if s["relevant"] is not None]
    return rate(sum(judged), len(judged))


def summarize_research(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    summary: Dict[str, Any] = {
        "runs": len(rows),
        "failed": len(rows) - len(ok),
        "success_rate": rate(len(ok), len(rows)),
        "latency_p50": percentile([r["latency"] for r in ok], 50),
        "latency_p95": percentile([r["latency"] for r in ok], 95),
        "avg_input_tokens": round(sum(r["input_tokens"] for r in ok) / len(ok)) if ok else 0,
        "avg_output_tokens": round(sum(r["output_tokens"] for r in ok) / len(ok)) if ok else 0,
        "citation_coverage": rate(sum(r["cited_claims"] for r in ok), sum(r["claims"] for r in ok)),
        "structure_rate": rate(sum(r["structure_ok"] for r in ok), len(ok)),
        "length_200_400_rate": rate(sum(200 <= r["words"] <= 400 for r in ok), len(ok)),
        "avg_sources": round(sum(r["sources"]["total"] for r in ok) / len(ok), 1) if ok else 0,
        "avg_domains": round(sum(r["sources"]["domains"] for r in ok) / len(ok), 1) if ok else 0,
        "runs_without_sources": sum(r["sources"]["total"] == 0 for r in ok),
        "planner_fallbacks": sum(r["fallback"] for r in ok),
    }
    claims = [j for r in ok for j in r.get("claim_judgments", [])]
    if claims:
        judged = [j for j in claims if j["verdict"] != "unparsed"]
        with_quote = [j for j in judged if j["verdict"] in ("yes", "partly")]
        sources = [s for r in ok for s in r.get("sources_judged", [])]
        scored_web = [s for s in sources if s["type"] == "web" and s["score"] is not None]
        answers = [r["answer_relevance"]["verdict"] for r in ok if "answer_relevance" in r]
        summary["judge"] = {
            "claims_judged": len(judged),
            "claims_unparsed": len(claims) - len(judged),
            "supported_yes": rate(sum(j["verdict"] == "yes" for j in judged), len(judged)),
            "supported_or_partly": rate(len(with_quote), len(judged)),
            "claims_adding_unsupported_details": rate(sum(j["adds_details"] is True for j in judged), len(judged)),
            "judge_quote_verified": rate(sum(j["quote_ok"] for j in with_quote), len(with_quote)),
            "source_precision": _precision(sources),
            "source_precision_by_type": {t: _precision([s for s in sources if s["type"] == t]) for t in ("web", "academic")},
            "web_source_precision_by_tavily_score": {
                band: _precision([s for s in scored_web if score_band(s["score"]) == band]) for band in ("<0.5", "0.5-0.7", ">=0.7")
            },
            "answer_relevance": {v: answers.count(v) for v in ("yes", "partly", "no", "unparsed")},
        }
    return summary


# --- Report --------------------------------------------------------------------------------------


def _table(rows: Dict[str, Any]) -> str:
    return "| Metric | Value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows.items()) + "\n"


def render_report(results: Dict[str, Any]) -> str:
    out = [f"# MARA evaluation report\n\nRun at {results['run_at']}. Labels are the author's judgement; the LLM judge is the app's own model, so it may favour its own style.\n"]
    if "guardrails" in results:
        g = results["guardrails"]
        summary = {k: g[k] for k in ("harmful", "benign", "block_rate", "false_positive_rate")}
        out += ["## Guardrails\n", _table({**summary, "errored": len(g["errored"])})]
        out.append("Block rate by category: " + ", ".join(f"{c} {v}" for c, v in g["block_rate_by_category"].items()) + "\n")
        out += [f"\nMissed ({len(g['missed'])}):\n"] + [f"- {q}" for q in g["missed"]]
        out += [f"\nWrongly blocked ({len(g['wrongly_blocked'])}):\n"] + [f"- {q}" for q in g["wrongly_blocked"]]
        out += [f"\nErrored, not scored ({len(g['errored'])}):\n"] + [f"- {q}" for q in g["errored"]]
        out += ["\nBorderline (not scored):\n"] + [f"- {q} -> {d}" for q, d in g["borderline"].items()]
    if "intent" in results:
        i = results["intent"]
        out += ["\n## Intent classification\n", _table({"scored": i["scored"], "accuracy": i["accuracy"], "errors": i["errors"]})]
        out += ["Confusion (expected -> predicted): " + json.dumps(i["confusion"]) + "\n", "\nWrong:\n"] + [f"- {w['query']}: expected {w['expected']}, got {w['got']}" for w in i["wrong"]]
    if "research" in results:
        s = results["research"]["summary"]
        rows = results["research"]["rows"]
        out += ["\n## Research quality\n", _table({k: v for k, v in s.items() if k != "judge"})]
        if "judge" in s:
            out += ["LLM-judge results:\n", _table(s["judge"])]
        details = [j for r in rows for j in r.get("claim_judgments", []) if j["adds_details"] is True]
        out += [f"\nClaims adding details their sources don't contain ({len(details)}), first 10:\n"] + [f"- [{j['verdict']}] {j['claim'][:160]}" for j in details[:10]]
        gaps = [
            (r["question"], [p for p in r["answer_relevance"]["parts"] if p["covered"] != "yes"])
            for r in rows
            if "answer_relevance" in r and r["answer_relevance"]["verdict"] != "yes"
        ]
        out += [f"\nQuestions not fully answered ({len(gaps)}) and the parts missed:\n"]
        out += [f"- {q}: " + ("; ".join(f"{p['part']} ({p['covered']})" for p in parts) or "no part detail") for q, parts in gaps]
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["guardrails", "intent", "research"])
    parser.add_argument("--limit", type=int, default=0, help="run an evenly spaced subset of each set")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM-judge metrics")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)

    data = json.loads((HERE / "dataset.json").read_text(encoding="utf-8"))
    results: Dict[str, Any] = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    if args.only in (None, "guardrails"):
        print("guardrails...", flush=True)
        results["guardrails"] = eval_guardrails(sample(data["guardrails"], args.limit), args.workers)
    if args.only in (None, "intent"):
        print("intent...", flush=True)
        results["intent"] = eval_intent(sample(data["intent"], args.limit), args.workers)
    if args.only in (None, "research"):
        print("research (slow)...", flush=True)
        rows = run_all(lambda q: research_one(q, not args.no_judge), sample(data["research"], args.limit), args.workers)
        results["research"] = {"summary": summarize_research(rows), "rows": rows}

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(results), encoding="utf-8")
    print(f"wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
