# Evaluation

Measures MARA against the real model and search (settings from `backend/.env`).

```
cd backend
python -m evals.run                       # everything: about 30-40 minutes
python -m evals.run --only guardrails     # guardrails | intent | research
python -m evals.run --limit 3 --no-judge  # quick smoke test
```

Output: `evals/results/report.md` (read this) and `report.json` (every row).

## What is measured

| Set (in `dataset.json`) | Size | Metrics |
|---|---|---|
| `guardrails` | 20 harmful, 20 benign (17 are hard negatives like "AI kill switch"), 3 borderline | block rate (overall and by category), false-positive rate, the missed and wrongly blocked queries |
| `intent` | 30 (10 each: CHAT, RESEARCH, REJECTED) | accuracy, confusion table, the wrong ones |
| `research` | 25 questions, run in Web Research mode | success rate, latency p50/p95, tokens, citation coverage, structure, length, sources and domains per run, planner fallbacks |

LLM-judge metrics (skip with `--no-judge`): **claim support** (each cited sentence checked against the sources it cites, with a quote that is verified to appear in the source), **source precision** (share of retrieved sources that are relevant), **answer relevance**.

## Read this before trusting the numbers

- The labels in `dataset.json` are the author's judgement. Review them.
- The judge is the app's own model, so it may favour its own writing. Label about 10 claims by hand (see `claim_judgments` in `report.json`) and check how often you agree before relying on the judge scores.
- Latency is only meaningful with `--workers 1` (the default). More workers is faster but inflates it.
- Each full run makes a few hundred LLM and search calls.
