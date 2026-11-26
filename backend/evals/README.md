# Evaluation

Measures MARA against the real model and search (settings from `backend/.env`).

## Running from a fresh clone

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), a configured model deployment, and a Tavily API key. Run from the repository checkout, not just an installed backend package: the evaluation tools and dataset are kept in the repository.

From the repository root, using PowerShell:

```powershell
Set-Location backend
uv sync --locked --extra dev
Copy-Item .env.example .env
```

For a fresh checkout only, copy the example as shown; do not overwrite an existing `.env`. Configure the model endpoint, API key, deployment name and API version, plus `TAVILY_API_KEY`, using [`backend/.env.example`](../.env.example) as a guide. Never commit the populated `.env`.

Then run from `backend`:

```powershell
uv run --locked python -m pytest                     # mocked unit/integration tests; no live eval
uv run --locked python -m evals.run --only intent --limit 3  # small live routing check
uv run --locked python -m evals.run                  # all 127 cases, including answer judging
```

No backend server or frontend is required; the runner calls the application workflow directly. Live evaluations require internet access and incur model/search API costs. `--no-judge` skips answer judging, not the application's model or search calls. Results will vary with the deployment, current web content, and provider availability; saved reports do not guarantee the same scores on another run.

## Suite commands

With the backend environment active:

```
cd backend
python -m evals.run                       # everything: over an hour
python -m evals.run --only guardrails     # guardrails | intent | research | lookup
python -m evals.run --limit 3 --no-judge  # quick smoke test
```

Output: `evals/results/report.md` (read this) and `report.json` (every row, including retrieved source excerpts). Each run overwrites both files; save them first if needed.

The saved evaluation uses one JSON/report pair per suite: [guardrails](results/guardrails.md), [intent](results/intent.md), [lookup](results/lookup.md), and [research](results/research.md), with matching `.json` files. See the [consolidated results](results/baseline-20261125.md). New runs write `report.md` and `report.json` and do not update these saved results automatically.

Each search step records an outcome, returned/considered/accepted counts and filter counts in `search_diagnostics` in `report.json`; web steps also include the score threshold and rejected scores. Failed or empty steps are listed in `report.md`. Errors record the exception type and, when available, HTTP status, not the exception message. Normal warning and error logs are not suppressed. Transient search errors (connection errors, timeouts, HTTP 429/500/502/503/504) are retried up to 5 attempts with jittered exponential backoff (3–30s) before a step is recorded as an error. LLM 429s and connection errors are retried up to 5 attempts (2–30s), on top of the SDK's own retries.

For LOOKUP, an LLM reviews whether the draft contains the requested fact. If not, Tavily extracts one query-ranked passage from the first eligible accepted web page, with a 10-second request timeout. The passage is checked for injection, limited to 1,000 characters, and replaces that page's snippet for both the writer and judge. If the answer remains incomplete or extraction fails, the writer can run one follow-up web search with the same filters. Follow-up searches are marked `followup` in `search_diagnostics`; `extraction_diagnostics` records the extraction outcome, duration and error type when applicable. Incompleteness is disclosed. Extraction and additional model calls add latency and provider cost. Extraction, review and the judge are fallible and do not guarantee relevance, correctness or freshness.

`stage_timings` records workflow-node wall time in seconds per run, excluding judge calls. `stage_retries` records LLM outer-retry counts and scheduled waits per node. Each search diagnostic includes `duration_seconds`, `retry_count`, and `retry_wait_seconds` for search outer retries. Two searches can run concurrently per request, so search durations overlap and should not be added to calculate total latency. Retry events and stage durations are also logged; SDK-internal retries are not counted in these retry metrics. The model client is reused for the lifetime of the process; restart after changing its configuration.

## What is measured

| Set (in `dataset.json`) | Size | Metrics |
|---|---|---|
| `guardrails` | 15 harmful, 15 benign (13 hard negatives and 2 plain benign), no borderline cases | block rate (overall and by category), false-positive rate, missed and wrongly blocked queries |
| `intent` | 30 routing cases: one CHAT, one LOOKUP and one RESEARCH in each of 10 domains | accuracy, equal-weight accuracy across domains (`accuracy_domain_macro`), under-routed rate (less work than needed), over-routed rate (more work than needed), accuracy by domain, confusion table, the wrong ones |
| `research` | 47 `{query, domain, type}` items, 6 in each of 7 domains and 5 in history_society; run in Web Research mode | success rate, latency p50/p95, tokens, citation coverage, structure, length, sources and domains per run, planner fallbacks; the key numbers again per domain (`by_domain`), as an equal-weight average across domains (`domain_macro_avg`) and per question type (`by_type`) |
| `lookup` | 20 single-fact questions, 2 in each of 10 domains, run in **auto** mode | the route taken (`route_counts`), answers under 120 words, citation coverage and the judge metrics, per domain and averaged equally across domains |

The report format (structure, 200–400 words) is only scored on answers that went through the RESEARCH route; lookups are short by design. There is no headline "AI" number: compare slices. The overall numbers weight each run (or claim) equally; `domain_macro_avg` weights each domain equally, so a domain with more failures or more judged claims can't dominate. `--limit N` takes evenly spaced items; coverage depends on the limit and dataset order.

History/society has five research questions, with no multi-part case. The saved research results cover 47 retained questions; one failed case was excluded.

LLM-judge metrics (skip with `--no-judge`): **claim support** (each cited sentence checked against the sources it cites, with a quote that is verified to appear in the source), **source precision** (share of retrieved sources that are relevant), **answer relevance**.

## Read this before trusting the numbers

- The labels in `dataset.json` are the author's judgement. Review them.
- Success measures execution, not correctness. A LOOKUP or RESEARCH run with no retrieved sources is counted as failed; other unanswered or outdated responses may still count as successful. Check time-sensitive answers and their retrieved source excerpts manually.
- The classifier and writer receive the current UTC date to help with time-sensitive questions, but this does not guarantee that results or answers are current.
- Claim extraction uses approximate sentence splitting. The judge checks the supplied excerpts, not whether they are the latest available evidence, so a supported verdict does not prove a time-sensitive claim is current.
- The judge is the app's own model, so it may favour its own writing. Label about 10 claims by hand (see `claim_judgments` in `report.json`) and check how often you agree before relying on the judge scores.
- Latency is only meaningful with `--workers 1` (the default). More workers is faster but inflates it.
- Each full run makes a few hundred LLM and search calls.
