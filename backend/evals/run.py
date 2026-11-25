"""
Evaluation harness. Runs the real app (real LLM and search from backend/.env) and writes
evals/results/report.md and report.json.

    python -m evals.run                       # everything (slow: over an hour)
    python -m evals.run --only guardrails     # guardrails | intent | research | lookup
    python -m evals.run --limit 3 --no-judge  # quick smoke test

Latency is only clean with --workers 1 (the default); more workers is faster but inflates it.
"""
import argparse
import json
import time
import uuid
from collections import Counter
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
    """What the app decides for a query without searching: BLOCKED, CHAT, LOOKUP, RESEARCH or ERROR."""
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


ROUTE_DEPTH = {"CHAT": 0, "LOOKUP": 1, "RESEARCH": 2}


def domain_macro(groups: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Equal-weight mean of each per-domain metric, so no domain counts more because it has more rows or claims."""
    metrics = [k for k in dict.fromkeys(k for g in groups.values() for k in g) if k != "runs"]
    out: Dict[str, Any] = {"domains": len(groups)}
    for metric in metrics:
        values = [g[metric] for g in groups.values() if isinstance(g.get(metric), (int, float))]
        if values:
            out[metric] = round(sum(values) / len(values), 3)
    return out


def eval_intent(cases: List[Dict[str, Any]], workers: int) -> Dict[str, Any]:
    predicted = run_all(lambda c: classify(c["query"]), cases, workers)
    scored = [(c, p) for c, p in zip(cases, predicted) if p != "ERROR"]
    pairs = [(c["intent"], p) for c, p in scored]

    def depth_gap(expected: str, got: str) -> int:
        return ROUTE_DEPTH.get(got, ROUTE_DEPTH[expected]) - ROUTE_DEPTH[expected]

    domains: Dict[str, List[bool]] = {}
    for c, p in scored:
        domains.setdefault(c.get("domain", "unknown"), []).append(c["intent"] == p)
    by_domain = {d: rate(sum(v), len(v)) for d, v in sorted(domains.items())}
    return {
        "scored": len(pairs),
        "accuracy": rate(sum(1 for e, p in pairs if e == p), len(pairs)),
        "accuracy_domain_macro": round(sum(by_domain.values()) / len(by_domain), 3) if by_domain else 0.0,
        # under-routed: less work than needed (e.g. answered from memory when it needed a search);
        # over-routed: more work than needed (slower and costlier, but not less correct)
        "under_routed": rate(sum(depth_gap(e, p) < 0 for e, p in pairs), len(pairs)),
        "over_routed": rate(sum(depth_gap(e, p) > 0 for e, p in pairs), len(pairs)),
        "accuracy_by_domain": by_domain,
        "confusion": confusion(pairs),
        "wrong": [{"query": c["query"], "expected": c["intent"], "got": p} for c, p in zip(cases, predicted) if p not in (c["intent"], "ERROR")],
        "errors": predicted.count("ERROR"),
    }


# --- Research quality ----------------------------------------------------------------------------


def research_one(case: str | Dict[str, Any], use_judge: bool, mode: str = "research") -> Dict[str, Any]:
    """Run one question through the whole graph. `case` is a question or a dataset item with domain/type tags."""
    case = {"query": case} if isinstance(case, str) else case
    question = case["query"]
    tags = {k: case[k] for k in ("domain", "type") if k in case}
    usage = TokenUsageHandler()
    start = time.time()
    try:
        state = research_graph.invoke(
            {"query": question, "mode": mode, "messages": []},
            config={"configurable": {"thread_id": str(uuid.uuid4())}, "callbacks": [usage]},
        )
    except Exception as exc:
        return {"question": question, **tags, "error": str(exc)[:200]}

    report = state.get("final_report", "")
    sources, citations = state.get("all_sources", []), state.get("citations", [])
    cited, claims = citation_coverage(report)
    row: Dict[str, Any] = {
        "question": question,
        **tags,
        "route": state.get("intent", ""),
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
        "retrieved_sources": sources,
        "search_diagnostics": state.get("search_diagnostics", []),
        "extraction_diagnostics": state.get("extraction_diagnostics", []),
        "stage_timings": state.get("stage_timings", {}),
        "stage_retries": state.get("stage_retries", {}),
        "fallback": bool(state.get("error")),
    }
    if state.get("guardrails", {}).get("blocked") or report.startswith("Error"):
        return {**row, "error": report[:200]}
    if state.get("intent") in ("LOOKUP", "RESEARCH") and not sources:
        return {**row, "error": f"No sources retrieved for {state['intent'].lower()}"}

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


def _is_report(row: Dict[str, Any]) -> bool:
    """Rows answered with the full research report (rows saved before routes were recorded count too)."""
    return row.get("route", "") in ("RESEARCH", "")


def summarize_slice(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The few numbers worth comparing between domains or question types."""
    ok = [r for r in rows if "error" not in r]
    judged = [j for r in ok for j in r.get("claim_judgments", []) if j["verdict"] != "unparsed"]
    answers = [r["answer_relevance"]["verdict"] for r in ok if "answer_relevance" in r]
    summary: Dict[str, Any] = {
        "runs": len(rows),
        "success_rate": rate(len(ok), len(rows)),
        "citation_coverage": rate(sum(r["cited_claims"] for r in ok), sum(r["claims"] for r in ok)),
        "avg_sources": round(sum(r["sources"]["total"] for r in ok) / len(ok), 1) if ok else 0,
    }
    if judged:
        summary["supported_yes"] = rate(sum(j["verdict"] == "yes" for j in judged), len(judged))
    if answers:
        summary["answer_fully_relevant"] = rate(answers.count("yes"), len(answers))
    return summary


def slices(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r.get(key, "unknown"), []).append(r)
    return {name: summarize_slice(group) for name, group in sorted(groups.items())}


def summarize_research(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    reports = [r for r in ok if _is_report(r)]
    lookups = [r for r in ok if r.get("route") == "LOOKUP"]
    routes = dict(Counter(r.get("route") or "RESEARCH" for r in rows if r.get("route") or "error" not in r))
    summary: Dict[str, Any] = {
        "runs": len(rows),
        "failed": len(rows) - len(ok),
        "success_rate": rate(len(ok), len(rows)),
        "route_counts": routes,
        "latency_p50": percentile([r["latency"] for r in ok], 50),
        "latency_p95": percentile([r["latency"] for r in ok], 95),
        "avg_input_tokens": round(sum(r["input_tokens"] for r in ok) / len(ok)) if ok else 0,
        "avg_output_tokens": round(sum(r["output_tokens"] for r in ok) / len(ok)) if ok else 0,
        "citation_coverage": rate(sum(r["cited_claims"] for r in ok), sum(r["claims"] for r in ok)),
        # the report format is only asked of research answers; a lookup answer is short and has no headings
        "structure_rate": rate(sum(r["structure_ok"] for r in reports), len(reports)),
        "length_200_400_rate": rate(sum(200 <= r["words"] <= 400 for r in reports), len(reports)),
        "avg_sources": round(sum(r["sources"]["total"] for r in ok) / len(ok), 1) if ok else 0,
        "avg_domains": round(sum(r["sources"]["domains"] for r in ok) / len(ok), 1) if ok else 0,
        "runs_without_sources": sum(r["sources"]["total"] == 0 for r in rows if "sources" in r),
        "planner_fallbacks": sum(r["fallback"] for r in ok),
    }
    if lookups:
        summary["lookup_under_120_words_rate"] = rate(sum(r["words"] <= 120 for r in lookups), len(lookups))
    if any("domain" in r for r in rows):
        summary["by_domain"] = slices(rows, "domain")
        summary["domain_macro_avg"] = domain_macro(summary["by_domain"])
    if any("type" in r for r in rows):
        summary["by_type"] = slices(rows, "type")
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


def _slice_table(title: str, groups: Dict[str, Dict[str, Any]]) -> str:
    columns = list(dict.fromkeys(k for g in groups.values() for k in g))
    head = f"| {title} | " + " | ".join(columns) + " |\n|---|" + "---|" * len(columns) + "\n"
    return head + "\n".join(f"| {name} | " + " | ".join(str(g.get(c, "-")) for c in columns) + " |" for name, g in groups.items()) + "\n"


SLICE_KEYS = ("judge", "by_domain", "domain_macro_avg", "by_type")


def _render_answers(title: str, section: Dict[str, Any]) -> List[str]:
    """Report section for a set of full graph runs (the research or the lookup set)."""
    s, rows = section["summary"], section["rows"]
    out = [f"\n## {title}\n", _table({k: v for k, v in s.items() if k not in SLICE_KEYS})]
    if "judge" in s:
        out += ["LLM-judge results:\n", _table(s["judge"])]
    if "by_domain" in s:
        out += ["\nBy domain:\n", _slice_table("domain", s["by_domain"])]
        out += ["\nEqual-weight average across domains (each domain counts once):\n", _table(s["domain_macro_avg"])]
    if "by_type" in s:
        out += ["\nBy question type:\n", _slice_table("type", s["by_type"])]
    errors = [r for r in rows if "error" in r]
    if errors:
        out += [f"\nFailed runs ({len(errors)}):\n"] + [f"- {r['question']}: {r['error']}" for r in errors]
    search_issues = [(r["question"], d) for r in rows for d in r.get("search_diagnostics", []) if d["outcome"] != "accepted"]
    if search_issues:
        out.append("\nSearch issues:\n")
        for question, d in search_issues:
            counts = f"returned={d['returned']}, considered={d['considered']}, low_score={d['filtered_low_score']}, duplicate={d['filtered_duplicate']}, injection={d['filtered_injection']}, accepted={d['accepted']}"
            error = f", error={d['error_type']}" if "error_type" in d else ""
            status = f", HTTP={d['error_status']}" if "error_status" in d else ""
            out.append(f"- {question} ({d['type']}): {d['outcome']} ({counts}{error}{status})")
    if any(r.get("claim_judgments") or "answer_relevance" in r for r in rows):
        details = [j for r in rows for j in r.get("claim_judgments", []) if j["adds_details"] is True]
        out += [f"\nClaims adding details their sources don't contain ({len(details)}), first 10:\n"] + [f"- [{j['verdict']}] {j['claim'][:160]}" for j in details[:10]]
        gaps = [
            (r["question"], [p for p in r["answer_relevance"]["parts"] if p["covered"] != "yes"])
            for r in rows
            if "answer_relevance" in r and r["answer_relevance"]["verdict"] != "yes"
        ]
        out += [f"\nQuestions not fully answered ({len(gaps)}) and the parts missed:\n"]
        out += [f"- {q}: " + ("; ".join(f"{p['part']} ({p['covered']})" for p in parts) or "no part detail") for q, parts in gaps]
    else:
        out.append("\nLLM-judge results are unavailable; answer correctness and claim support were not assessed.\n")
    return out


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
        keys = ("scored", "accuracy", "accuracy_domain_macro", "under_routed", "over_routed", "errors")
        out += ["\n## Routing (intent classification)\n", _table({k: i[k] for k in keys if k in i})]
        if i.get("accuracy_by_domain"):
            out.append("Accuracy by domain: " + ", ".join(f"{d} {v}" for d, v in i["accuracy_by_domain"].items()) + "\n")
        out += ["Confusion (expected -> predicted): " + json.dumps(i["confusion"]) + "\n", "\nWrong:\n"] + [f"- {w['query']}: expected {w['expected']}, got {w['got']}" for w in i["wrong"]]
    if "research" in results:
        out += _render_answers("Research quality (research mode)", results["research"])
    if "lookup" in results:
        out += _render_answers("Lookup questions (auto mode)", results["lookup"])
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["guardrails", "intent", "research", "lookup"])
    parser.add_argument("--limit", type=int, default=0, help="run an evenly spaced subset of each set")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM-judge metrics")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

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
        rows = run_all(lambda c: research_one(c, not args.no_judge), sample(data["research"], args.limit), args.workers)
        results["research"] = {"summary": summarize_research(rows), "rows": rows}
    if args.only in (None, "lookup"):
        print("lookup...", flush=True)
        # auto mode, so the router's choice is part of what is measured
        rows = run_all(lambda c: research_one(c, not args.no_judge, mode="auto"), sample(data["lookup"], args.limit), args.workers)
        results["lookup"] = {"summary": summarize_research(rows), "rows": rows}

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(results), encoding="utf-8")
    print(f"wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
