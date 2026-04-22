# news-mas

Multi-agent news intelligence system. Ingests topics, scores and filters articles,
generates summaries, and produces a curated digest — all via a LangGraph-orchestrated
pipeline of specialised FastAPI micro-agents.

Designed for healthcare contexts: PII/PHI scrubbing is mandatory at ingestion, content
is never logged, and every agent call carries data-sensitivity metadata through the
entire chain.

> For full design context see [`ARCHITECTURE.md`](ARCHITECTURE.md). For the planned
> extraction of `src/common` into standalone pip packages, see
> [`CHASSIS_DESIGN.md`](CHASSIS_DESIGN.md).

---

## Architecture

```
Phase 1 — Discovery & Scoring
  SearchWorker → HeatScorer → FilterAgent → Selector → Phase1Judge

Phase 2 — Summarisation & Quality Gate
  Summarizer ⇄ Reviewer (retry loop, max 3) → RelevanceGate → Digest
```

Each agent is an independent FastAPI service. Two LangGraph graphs in
`src/orchestrators/` coordinate them over HTTP. A central registry at port 8000
stores each agent's configuration; all agents call `bootstrap_agent()` on startup
to fetch their `AgentConfig`.

All agents share:

- **Pydantic v2 schemas** (`src/common/schemas.py`)
- **Structured JSON logging** — raw content is never emitted (`src/common/observability.py`)
- **PII/PHI scrubbing** via Presidio (`src/common/pii_scrubber.py`)
- **Input normalisation + injection detection** (`src/common/security.py`)
- **AAP token auth** + rate limiting (30 req/min) on every `/run` endpoint
- **Result confidence taxonomy** — `SearchResult` carries `ResultConfidence`
  (FULL/SNIPPET/PARTIAL/INJECTED/SCRUBBED); aggregated into `CandidateConfidence`
  (HIGH/MEDIUM/LOW) and forwarded to `Phase1Judge`

### Agent inventory

| Agent | Port | Model | Purpose |
|---|---|---|---|
| `registry` | 8000 | — | Stores and serves `AgentConfig`; seeds defaults on startup |
| `search_worker` | 8001 | Tavily | Fetches raw articles per topic |
| `heat_scorer` | 8002 | Gemma 4 (Ollama) | Scores articles 0.0–1.0: volume × velocity × novelty × significance |
| `filter_agent` | 8003 | — | Drops articles below `min_heat_score` |
| `selector` | 8004 | — | Picks top-K by heat score |
| `phase1_judge` | 8005 | Gemma 4 (Ollama) | Holistic approve/reject over the shortlist |
| `summarizer` | 8006 | Claude Sonnet | Generates summary + key points for one article |
| `reviewer` | 8007 | Claude Sonnet | Evaluates summary quality; approve or request revision |
| `relevance_gate` | 8008 | Gemma 4 (Ollama) | Scores digest relevance; gates final inclusion |

---

## Setup

### 1. Prerequisites

- Python 3.11+
- Docker + Docker Compose (observability stack)
- [Ollama](https://ollama.com/) with `gemma4:e4b` pulled (HeatScorer, Phase1Judge, RelevanceGate)

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -e .
```

### 4. Download spaCy model (required by Presidio)

```bash
python -m spacy download en_core_web_lg
```

### 5. Pull the Ollama model

```bash
ollama pull gemma4:e4b
```

### 6. Configure environment

```bash
copy .env.example .env           # Windows
# cp .env.example .env           # macOS / Linux
```

Required keys at minimum:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Summarizer + Reviewer (Claude Sonnet) |
| `TAVILY_API_KEY` | SearchWorker article fetching |
| `MAS_SECRET_KEY` | Shared secret for inter-agent auth (`X-MAS-Secret` header) |
| `FERNET_KEY` | AES encryption key for `FernetStorage` — generate with `python scripts/generate_key.py` |

Optional:

| Variable | Purpose |
|---|---|
| `LANGSMITH_API_KEY` | LLM call tracing via LangSmith (agents start without it) |
| `LANGSMITH_PROJECT` | LangSmith project name (default: `news-mas`) |
| `OLLAMA_BASE_URL` | Ollama endpoint (default: `http://localhost:11434`) |
| `MODEL_OVERRIDE` | Override the model for all LLM calls globally during development |
| `CONFIDENCE_HIGH_THRESHOLD` | Fraction of full-content results to qualify as HIGH confidence (default: `0.7`) |
| `CONFIDENCE_MEDIUM_THRESHOLD` | Fraction for MEDIUM confidence (default: `0.3`) |

---

## Observability stack

Runs OpenTelemetry Collector, Jaeger, Prometheus, Loki, and Grafana across four
isolated Docker networks (identity / orchestration / compute / observability).

```bash
docker compose up -d
```

| Service | URL | Purpose |
|---|---|---|
| Grafana | http://localhost:3000 | Dashboards: egress, injection rate, source confidence |
| Jaeger | http://localhost:16686 | Distributed traces |
| Prometheus | http://localhost:9090 | Metrics + alerting rules |
| Loki | (internal) | Log aggregation |
| OTel Collector | gRPC :4317 / HTTP :4318 | Span + metric ingestion |

Alerting rules in `prometheus/alerts.yml`:
- `EgressInjectionDetected` (high) — injection flag on any egress fetch
- `InjectionDetectedInRun` (high) — injection-flagged results in a pipeline run
- `LowSourceConfidence` (medium) — 30-min avg `confidence_ratio` < 0.3
- `TavilyAnomalousQueryLength` / `TavilyHighCallVolume` (medium)

```bash
docker compose down
```

---

## Running agents

Start the registry first, then agents in any order. Run from the project root so
`src.*` imports resolve correctly.

```bash
# Registry (port 8000) — start first
uvicorn src.registry.registry_server:app --port 8000 --reload

# Phase 1
uvicorn src.agents.search_worker.main:app --port 8001 --reload
uvicorn src.agents.heat_scorer.main:app --port 8002 --reload
uvicorn src.agents.filter_agent.main:app --port 8003 --reload
uvicorn src.agents.selector.main:app --port 8004 --reload
uvicorn src.agents.phase1_judge.main:app --port 8005 --reload

# Phase 2
uvicorn src.agents.summarizer.main:app --port 8006 --reload
uvicorn src.agents.reviewer.main:app --port 8007 --reload
uvicorn src.agents.relevance_gate.main:app --port 8008 --reload
```

All agents expose:
- `GET /health` — liveness check (no auth required)
- `POST /run` — main endpoint (requires `X-MAS-Secret` header)

---

## Running tests

```bash
pytest
```

Live integration tests (require Ollama + `INTEGRATION_TESTS=true`):

```bash
INTEGRATION_TESTS=true pytest tests/integration/ -v -m integration
```

Unit test coverage:

| Test file | Covers |
|---|---|
| `test_schemas.py` | All Pydantic models; schema validation |
| `test_result_confidence.py` | `ResultConfidence` enum, `CandidateConfidence` thresholds, injection override |
| `test_pii_scrubber.py` | PII/PHI detection and scrubbing via Presidio |
| `test_security.py` | Injection detection, NFKC normalisation |
| `test_pipeline_errors.py` | Error codes, severity, pipeline error protocol |
| `test_data_quality.py` | Data quality checks |
| `test_aap_tokens.py` | AAP token mint / exchange / validate; delegation depth; `[DPOP-TODO]` markers |
| `test_workload_identity.py` | Workload identity provider selection |
| `test_agent_bootstrap.py` | Agent registry bootstrap sequence |
| `test_storage.py` | `FernetStorage` encrypt/decrypt, repository CRUD |
| `test_langsmith_config.py` | LangSmith tag/metadata configuration |
| `tests/integration/test_heat_scorer_live.py` | Live Gemma 4 scoring, OTel spans, LangSmith config |
| `tests/integration/test_search_worker_live.py` | Live Tavily search |

---

## Project layout

```
news-mas/
├── configs/
│   ├── otel-collector-config.yaml
│   ├── prometheus.yml
│   └── grafana/provisioning/
│       └── dashboards/egress.json
├── prometheus/
│   └── alerts.yml
├── prompts/
│   ├── heat_scorer/v1.0.yaml
│   └── phase1_judge/v1.0.yaml
├── scripts/
│   └── generate_key.py
├── src/
│   ├── common/
│   │   ├── schemas.py              # All Pydantic v2 models (SearchResult, CandidateConfidence, …)
│   │   ├── pipeline_errors.py      # ResultConfidence enum, error codes, ErrorSeverity
│   │   ├── prompt_loader.py        # Cached YAML loader; make_run_config() for LangSmith tags
│   │   ├── pii_scrubber.py         # Presidio PII + PHI scrubbing
│   │   ├── observability.py        # OTel + LangSmith setup; _SafeJSONFormatter
│   │   ├── security.py             # NFKC normalisation, injection detection
│   │   ├── data_quality.py         # Data quality validation
│   │   ├── error_codes.py          # Structured error code registry
│   │   ├── storage.py              # FernetStorage + UserRepository / RunRepository / etc.
│   │   ├── agent_bootstrap.py      # bootstrap_agent(); reads AgentConfig from registry
│   │   └── agent_registry.py       # In-process agent capability registry
│   ├── auth/
│   │   ├── token_service.py        # AAP token mint / exchange / validate (HS256; DPoP deferred)
│   │   └── workload_identity.py    # WorkloadIdentityProvider; LocalDevIdentityProvider
│   ├── registry/
│   │   ├── registry_server.py      # FastAPI registry; seeds default AgentConfig on startup
│   │   └── card_store.py           # In-memory AgentConfig store (FernetStorage migration deferred)
│   ├── agents/
│   │   ├── search_worker/
│   │   ├── heat_scorer/
│   │   ├── filter_agent/
│   │   ├── selector/
│   │   ├── phase1_judge/
│   │   ├── summarizer/
│   │   ├── reviewer/
│   │   └── relevance_gate/
│   └── orchestrators/
│       ├── phase1/graph.py         # LangGraph Phase 1 pipeline
│       └── phase2/graph.py         # LangGraph Phase 2 pipeline (retry loop)
├── tests/
│   ├── integration/
│   │   ├── test_heat_scorer_live.py
│   │   └── test_search_worker_live.py
│   └── test_*.py
├── ARCHITECTURE.md                 # Full design doc; read this before modifying agents
├── CHASSIS_DESIGN.md               # Planned extraction of src/common into pip packages
├── DPOP_IMPLEMENTATION_GUIDE.md    # DPoP rollout plan; read before touching src/auth/
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

---

## Key design decisions

**Auth is layered.** `X-MAS-Secret` is the current inter-agent credential (dev only).
The target is Microsoft Entra ID + DPoP (RFC 9449). All insertion points are tagged
`[DPOP-TODO]` in `src/auth/token_service.py` and `src/auth/workload_identity.py`.
Full plan in `DPOP_IMPLEMENTATION_GUIDE.md`.

**Content never leaves the scrubber.** `scrub_text()` runs on all external content
before it enters any agent. The `_SafeJSONFormatter` blocks `content`, `text`, `body`,
`article`, `summary`, `raw`, `html`, and `snippet` keys from all log records.

**Model assignments live in the registry.** The model split (Gemma 4 for scoring,
Claude Sonnet for prose) is stored in `AgentConfig` records, not hardcoded. Set
`MODEL_OVERRIDE` in `.env` to redirect all LLM calls to a different model during
development.

**Prompts are versioned files.** Never edit a deployed prompt in place — create a new
`v<major>.<minor>.yaml` and update the `load_prompt()` call. Every LangSmith trace is
tagged with the exact prompt version that produced it.

**Confidence flows end-to-end.** Each `SearchResult` carries a `ResultConfidence`
value set by the search worker. `HeatScorer` aggregates these into a
`CandidateConfidence` object (HIGH/MEDIUM/LOW) which `Phase1Judge` receives in its
prompt to apply appropriate scepticism to low-quality source batches.
