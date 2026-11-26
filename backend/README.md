# MARA Backend

The backend runs MARA's research workflow using FastAPI, LangGraph, OpenAI GPT-5.1 and Tavily.

## Workflow

- The input guard screens requests and masks personal information.
- In **Automatic** mode, the classifier routes questions to CHAT, LOOKUP or RESEARCH. **Web Research** mode uses the research workflow unless the request is blocked.
- The research workflow plans searches, retrieves web and academic sources, and writes a cited answer.
- The streaming endpoint reports agent progress to the frontend.

Run history and conversation state are stored in memory and reset when the backend restarts.

## Local setup

Requires Python 3.11 or newer. From the repository root in PowerShell:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

In `.env`, configure the model endpoint, API key, deployment name and API version, plus `TAVILY_API_KEY`. See [`.env.example`](.env.example) for the required variable names and optional settings. Never commit your populated `.env`.

Start the backend:

```powershell
python -m uvicorn main:app --reload
```

The API is available at `http://localhost:8000`, with interactive documentation at `http://localhost:8000/docs`. If the frontend uses a different origin from `http://localhost:5173`, add its exact origin to `BACKEND_CORS_ORIGINS` in `.env` and restart the backend.

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/research` | Run a question and return the completed answer |
| `POST /api/v1/research/stream` | Run the same workflow with server-sent `stage`, `result` or `error` events |
| `GET /api/v1/research/{thread_id}` | Get the latest result for a conversation |
| `GET /api/v1/runs/{run_id}` | Get a specific run |
| `GET /api/v1/admin/stats` and `/api/v1/admin/history` | Read current process statistics and history |
| `GET /api/v1/health` | Check backend and model configuration status |

Both research endpoints accept a JSON body such as `{"query": "What is RAG?", "mode": "auto"}`. Use `"research"` for Web Research mode; `thread_id` and conversation `messages` are optional.

## Observability

Langfuse tracing is optional and disabled by default. To enable it, add the following settings to `.env` and restart the backend:

```dotenv
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=<your-public-key>
LANGFUSE_SECRET_KEY=<your-secret-key>
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

Each request has a trace with agent spans and child LLM generations. These record latency, retries, error types and provider-reported input, output and total tokens. Classifier spans record `accepted`, `blocked` or `unchecked`.

Tracing excludes queries, conversation history, answers, source content, URLs, thread IDs and raw exception messages. Model deployment names and routing metadata are included. Missing Langfuse credentials prevent startup when tracing is enabled.

## Evaluation results

The dataset has **127 cases** across guardrails, routing, research and lookup. The saved suite runs from November 24–25, 2026 reported:

| Metric | Result |
|---|---|
| Answers judged fully relevant | **85.1% (57/67)**: research 80.9% (38/47), lookup 95.0% (19/20) |
| Sampled claims judged fully supported by supplied excerpts | **83.0% (333/401 parsed claims)**: research 82.1% (308/375), lookup 96.2% (25/26) |
| Returned sources judged relevant | **96.2% research**, **89.7% lookup** |
| Guardrails | **15/15 harmful blocked**, **0/15 benign wrongly blocked** |
| Routing | **30/30** expected routes |
| Median latency | **19.0 s research**, **9.1 s lookup** |
| p95 latency | **27.8 s research**, **13.0 s lookup** |

These are separate suite runs, not a single run on one frozen revision. The LLM judge checks relevance and support against supplied excerpts; it does not independently verify factual accuracy or freshness. Support covers up to eight sampled cited claims per answer; one unparsed claim was excluded. One failed research case was excluded from the 47 reported research answers. A fallback refusal after a classifier failure can also count as blocked in the guardrail results. Latencies are from evaluation runs, not load tests.

See the [consolidated evaluation report](evals/results/baseline-20261125.md) for the breakdown and limitations. Live evaluations use model and search APIs, may incur charges, and overwrite `evals/results/report.md` and `report.json`. For example, run `python -m evals.run --only research` from `backend` to evaluate research questions.

## Tests

With the backend environment active:

```powershell
python -m pytest
```

CI installs dependencies from `uv.lock` and runs the tests, including an offline guardrail regression gate with a fake LLM. The last recorded local run passed **449 backend tests**.