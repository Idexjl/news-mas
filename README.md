# news-mas

Multi-agent news intelligence system. Ingests topics, scores and filters articles,
generates summaries, and produces a curated digest — all via a LangGraph-orchestrated
pipeline of specialised FastAPI micro-agents.

---

## Architecture

```
Phase 1 — Discovery & Scoring
  SearchWorker → HeatScorer → FilterAgent → Selector → Phase1Judge

Phase 2 — Summarisation & Quality Gate
  Summarizer ⇄ Reviewer (retry loop, max 3) → RelevanceGate → Digest
```

Each agent is an independent FastAPI service. The two LangGraph graphs in
`src/orchestrators/` coordinate them. All agents share:

- **Pydantic v2 schemas** (`src/common/schemas.py`)
- **Structured JSON logging** that never emits raw article content (`src/common/observability.py`)
- **PII/PHI scrubbing** via Presidio (`src/common/pii_scrubber.py`)
- **Input normalisation + injection detection** (`src/common/security.py`)
- **Shared-secret header auth** + rate limiting on every `/run` endpoint

---

## Setup

### 1. Prerequisites

- Python 3.11+
- Docker + Docker Compose (for observability stack)

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
pip install spacy
python -m spacy download en_core_web_lg
```

### 5. Configure environment

```bash
copy .env.example .env           # Windows
# cp .env.example .env           # macOS / Linux
# Edit .env — at minimum set ANTHROPIC_API_KEY and TAVILY_API_KEY
```

---

## Observability stack

The stack runs OpenTelemetry Collector, Jaeger, Prometheus, Loki, and Grafana.

```bash
docker compose up -d
```

| Service    | URL                      | Purpose                     |
|------------|--------------------------|-----------------------------|
| Grafana    | http://localhost:3000    | Dashboards (anon access)    |
| Jaeger     | http://localhost:16686   | Distributed traces          |
| Prometheus | http://localhost:9090    | Metrics                     |
| Loki       | (internal only)          | Log aggregation             |
| OTel       | grpc :4317 / http :4318  | Collector ingestion ports   |

Shut down:

```bash
docker compose down
```

---

## Running agents

Each agent is a standalone Uvicorn process. Run them from the project root so
`src.*` imports resolve correctly.

```bash
# Search Worker (port 8001)
uvicorn src.agents.search_worker.main:app --port 8001 --reload

# Heat Scorer (port 8002)
uvicorn src.agents.heat_scorer.main:app --port 8002 --reload

# Filter Agent (port 8003)
uvicorn src.agents.filter_agent.main:app --port 8003 --reload

# Selector (port 8004)
uvicorn src.agents.selector.main:app --port 8004 --reload

# Phase 1 Judge (port 8005)
uvicorn src.agents.phase1_judge.main:app --port 8005 --reload

# Summarizer (port 8006)
uvicorn src.agents.summarizer.main:app --port 8006 --reload

# Reviewer (port 8007)
uvicorn src.agents.reviewer.main:app --port 8007 --reload

# Relevance Gate (port 8008)
uvicorn src.agents.relevance_gate.main:app --port 8008 --reload
```

All agents expose:
- `GET /health` — liveness check (no auth required)
- `POST /run` — main endpoint (requires `X-MAS-Secret` header when `MAS_SECRET_KEY` is set)

---

## Running tests

```bash
pytest
```

Tests cover:
- All Pydantic schemas instantiate and validate correctly (`tests/test_schemas.py`)
- PII/PHI detection and scrubbing via Presidio (`tests/test_pii_scrubber.py`)
- Injection pattern detection and input normalisation (`tests/test_security.py`)

---

## Project layout

```
news-mas/
├── configs/                        # OTel collector, Prometheus configs
├── prompts/                        # Versioned YAML prompt files
│   └── heat_scorer/v1.0.yaml
├── src/
│   ├── common/
│   │   ├── schemas.py              # All Pydantic v2 models
│   │   ├── prompt_loader.py        # Cached YAML loader with version tagging
│   │   ├── pii_scrubber.py         # Presidio PII + PHI scrubbing
│   │   ├── observability.py        # OTel setup + safe JSON logger
│   │   └── security.py             # NFKC normalisation, injection detection
│   ├── agents/
│   │   ├── search_worker/          # FastAPI app + agent stub
│   │   ├── heat_scorer/
│   │   ├── filter_agent/
│   │   ├── selector/
│   │   ├── phase1_judge/
│   │   ├── summarizer/
│   │   ├── reviewer/
│   │   └── relevance_gate/
│   └── orchestrators/
│       ├── phase1/graph.py         # LangGraph Phase 1 pipeline
│       └── phase2/graph.py         # LangGraph Phase 2 pipeline with retry
├── tests/
│   ├── test_schemas.py
│   ├── test_pii_scrubber.py
│   └── test_security.py
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```
