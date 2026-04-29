# news-mas Architecture

> **For Claude Code sessions:** Read this file first in any new conversation to restore
> full project context. See [§13 Onboarding](#13-onboarding-note-for-claude-code) for
> the recommended warm-up sequence.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Two-Phase MAS Design](#2-two-phase-mas-design)
3. [Agent Inventory](#3-agent-inventory)
4. [A2A Protocol](#4-a2a-protocol)
5. [AAP Auth Layer](#5-aap-auth-layer)
6. [Model Split](#6-model-split)
7. [PII/PHI Handling](#7-piiphi-handling)
8. [Observability Stack and Network Segmentation](#8-observability-stack)
9. [Context Window Strategy](#9-context-window-strategy)
10. [Prompt Versioning](#10-prompt-versioning)
11. [Feedback Data Model](#11-feedback-data-model)
12. [Key Deferred Decisions](#12-key-deferred-decisions)
13. [Onboarding Note for Claude Code](#13-onboarding-note-for-claude-code)

---

## 1. System Overview

news-mas is a multi-agent news intelligence system. Given a list of topics, it discovers
recent articles, scores them for relevance and timeliness, filters to the best candidates,
generates summaries, quality-checks those summaries, and produces a curated digest — all
without human involvement in the loop.

The system is designed for a **healthcare context**: PII/PHI scrubbing is mandatory at
ingestion, content is never logged, and every agent runs behind an auth layer that
carries data-sensitivity metadata through the entire call chain.

Eight specialised agents are coordinated by two LangGraph orchestrator graphs. Each agent
is an independent FastAPI service with its own port, rate limit, and auth middleware.
The orchestrators call agents over HTTP; agents never call each other directly.

---

## 2. Two-Phase MAS Design

The pipeline is split into two sequential phases. They run as separate LangGraph graphs
(`src/orchestrators/phase1/graph.py` and `src/orchestrators/phase2/graph.py`) and share
state only through the list of judged articles that Phase 1 produces.

### Phase 1 — Discovery and Scoring

```
SearchWorker → HeatScorer → FilterAgent → Selector → Phase1Judge
```

**What it does:** Casts a wide net and narrows it to a shortlist.

- `SearchWorker` fetches raw articles for each topic via the Tavily search API. Each result is
  returned as a `SearchResult` (extends `RawArticle`) with a `ResultConfidence` value:
  `FULL` (fetched), `SNIPPET` (fetch failed), `PARTIAL` (truncated), `INJECTED` (discarded),
  or `SCRUBBED` (PHI removed). Injected articles are returned with empty content so the count
  flows to the heat scorer without forwarding payload.
- `HeatScorer` assigns each article a 0.0–1.0 heat score. It skips INJECTED articles and
  aggregates all confidence values into a `CandidateConfidence` object (attached to
  `HeatScorerOutput`) that summarises source quality: full/snippet/partial/injection/scrubbed
  counts, `confidence_ratio` (full / total), and an `overall_confidence` of HIGH/MEDIUM/LOW.
  Thresholds are configurable via `CONFIDENCE_HIGH_THRESHOLD` / `CONFIDENCE_MEDIUM_THRESHOLD`.
  Any injection forces `overall_confidence` to LOW regardless of ratio.
- `FilterAgent` enforces semantic constraints such as "no never-trumpers" or "no opinion pieces". It uses Gemma 4 to understand the *intent* of each constraint, not keyword matching. `INJECTED` articles are auto-removed without an LLM call. Results are batched (max 5 per call, configurable via `FILTER_BATCH_SIZE`). Constraint text is treated as user data and never appears in logs or OTel spans — violations are recorded by constraint index only. `FilterAgentOutput` carries `kept_results`, `removed_results`, and `removal_reasons` (with URL, constraint index, confidence, and an optional note for snippet-only removals).
- `Selector` makes a curatorial selection of the top MAX_CANDIDATES topics (default 5). It uses Gemma 4 to reason holistically over four signals: heat score, source confidence (HIGH/MEDIUM/LOW from `CandidateConfidence`), content coverage quality (kept vs. removed results ratio), and category diversity. When `ENFORCE_DIVERSITY=true` (default), it actively avoids selecting five topics from the same domain even if they rank highest by heat score alone. An optional `topic_memory_context` input carries the last 3 run outcomes per topic from `TopicMemoryRepository` when available. Input is `ScoredFilteredTopic` objects (one per user topic); output is `SelectorOutput` with `selected[]` (topic_id, rank, reasoning), `rejected[]` (topic_id, reject_reason), `selection_confidence`, and `selected_articles` (flattened kept results for Phase1Judge).
- `Phase1Judge` is an independent single-pass reviewer of the selector's topic choices. It receives the full picture: all candidates (with heat scores and confidence), the selector's selections with their reasoning, and the selector's rejections with their reasons. The judge either **ENDORSES** (selector's choices are sound) or **ADJUSTS** (makes targeted swaps within the existing candidate pool). Adjustments are one of three types: `confidence_upgrade` (swapping a LOW-confidence selection for a HIGH-confidence alternative), `diversity_improvement` (reducing category concentration), or `quality_concern` (extreme removal-ratio coverage issue). The judge is biased toward endorsement — it adjusts only when there is a clear quality failure, not to re-rank from scratch. Every adjustment is emitted as a `phase1_judge.adjustment` OTel span event (topic IDs + reason_code only, never reasoning text). `Phase1JudgeOutput` carries `verdict`, `final_selected[]` (with `source="selector"|"judge"` provenance), `adjustments[]`, `overall_confidence`, and `judged_articles` (flat approved-article list consumed by Phase 2).

**Why a judge at the end:** The selector applies holistic ranking; the judge provides independent quality assurance for confidence and coverage failures the selector may have missed. Single-pass design prevents runaway retry loops on a single topic.

### Phase 2 — Summarisation and Quality Gate

```
Summarizer ⇄ Reviewer (retry loop, max 3) → RelevanceGate → Digest
```

**What it does:** Produces a quality-checked summary for each approved article, then
gates the digest on relevance confidence.

- `Summarizer` generates a cited summary and key points for one article or a topic's source set. Accepts an optional `sources: list[SearchResult]` (multi-source) or falls back to the single `article` field. Uses a hybrid A/B context strategy: direct (single Claude Sonnet call) when the estimated token count is ≤ `TOKEN_BUDGET_SOURCES` (default 16 k), map-reduce (concurrent mini-summarize per source then synthesis) above the threshold. Every factual claim is grounded to a citation URL; hallucinated URLs are rejected with `CITATION_INVALID RETRYABLE`. Output is PHI-checked belt-and-suspenders and rejected with `PHI_DETECTED RETRYABLE` if entities are detected. Retry runs carry a `reviewer_feedback` field; their LangSmith trace is named `summarizer-retry`. Model is hardcoded to `claude-sonnet-4-6` — `MODEL_OVERRIDE` never applies.
- `Reviewer` evaluates quality. If the summary is rejected and retries remain
  (`MAX_SUMMARY_RETRIES = 3`), it routes back to `Summarizer` with feedback.
- `RelevanceGate` assigns a relevance confidence score. Low-confidence articles are
  excluded from the final digest.

**Why the retry loop is capped:** Uncapped retries would stall the pipeline on a single
bad article. Three attempts is enough to catch transient LLM variance; hard failures
after that are surfaced in the run log rather than silently dropped.

**Why two separate phases rather than one graph:** Phase 1 is batch-shaped (fan-out
across many articles, funnel down); Phase 2 is per-article (map over the shortlist with
a quality loop per item). They have different state shapes, different concurrency
patterns, and different failure modes. Splitting them makes each graph's routing logic
easier to reason about and test independently.

---

## 3. Agent Inventory

All agents expose `GET /health` (no auth) and `POST /run` (auth required). Rate limit:
30 requests/minute via `slowapi`.

| Agent | Port | Phase | Model provider | Model ID | Purpose |
|---|---|---|---|---|---|
| `registry` | 8000 | — | — | — | Stores and serves AgentConfig; seeds defaults on startup |
| `search_worker` | 8001 | 1 | tavily | — | Fetch raw articles from Tavily for each topic |
| `heat_scorer` | 8002 | 1 | ollama | `gemma4:e4b` | Score articles 0.0–1.0 on volume × velocity × novelty × significance |
| `filter_agent` | 8003 | 1 | ollama | `gemma4:e4b` | Semantic constraint filtering — drops articles that violate user-defined intent constraints |
| `selector` | 8004 | 1 | ollama† | `gemma4:e4b` | Holistic topic selection with diversity enforcement |
| `phase1_judge` | 8005 | 1 | ollama | `gemma4:e4b` | Independent single-pass review: ENDORSE or ADJUST selector choices |
| `summarizer` | 8006 | 2 | anthropic | `claude-sonnet-4-6` | Generate summary + key points for one article |
| `reviewer` | 8007 | 2 | anthropic | `claude-sonnet-4-6` | Evaluate summary quality; approve or request revision |
| `relevance_gate` | 8008 | 2 | ollama | `gemma4:e4b` | Score digest relevance confidence; gate final inclusion |

†`filter_agent` and `selector` use Gemma 4 via Ollama for their respective reasoning tasks — semantic constraint filtering and holistic topic selection — even though their registry entries still show `model_provider: none` (a registry metadata lag from when they were planned as deterministic; both agents load `gemma4:e4b` directly and respect `MODEL_OVERRIDE`).

**Model assignment is stored in the registry, not hardcoded.** The registry seeds default
`AgentConfig` records on startup (`src/registry/registry_server.py`). Agents fetch their
own config at startup via `bootstrap_agent()` (`src/common/agent_bootstrap.py`). The
`MODEL_OVERRIDE` env var remains available for global model switching during development.

### Agent bootstrap sequence

Every agent runs this sequence on startup (FastAPI `@on_event("startup")`):

```
1. Read REGISTRY_URL + AGENT_ID from environment
2. Call bootstrap_agent(registry_url, identity_provider, agent_id)
   → GET /agents/{agent_id}/config on the registry (auth: X-MAS-Secret)
   → Returns AgentConfig
3. Store AgentConfig on app.state.agent_config
4. Register with local in-memory agent_registry (capabilities from AgentConfig)
5. Emit OTel span "agent.bootstrap" with agent_id, model_provider, model_id
```

If the registry is unreachable at startup, the agent logs a warning and starts in
degraded mode (`app.state.agent_config = None`). This is acceptable for local dev
where the registry may not be running; it would not be acceptable in production.

**Shared across all agents:**
- Pydantic v2 schemas (`src/common/schemas.py`)
- Structured JSON logging that never emits raw content (`src/common/observability.py`)
- PII/PHI scrubbing via Presidio (`src/common/pii_scrubber.py`)
- Input normalisation + injection detection (`src/common/security.py`)
- AAP token validation middleware (current: shared-secret; target: DPoP — see §4 and §5)

---

## 4. A2A Protocol

### Current state

All agent-to-agent calls are HTTP/JSON. The orchestrators call agents directly over
`POST /run`. Authentication is handled by `get_identity_provider()` in
`src/auth/workload_identity.py`, which returns a `WorkloadIdentityProvider` for
the environment. Each agent's `SecurityHeaderMiddleware` calls
`provider.validate_incoming_token(X-MAS-Secret)` on every `/run` request.

The active provider is selected by the `WORKLOAD_IDENTITY_PROVIDER` env var
(default: `local`). Currently only `LocalDevIdentityProvider` is implemented —
it validates the `X-MAS-Secret` header against `MAS_SECRET_KEY` using constant-time
comparison. The three production providers (`azure-managed-identity`, `aws-sts`,
`gcp-workload`) raise `NotImplementedError` with implementation guides pointing to
`[WORKLOAD-IDENTITY-TODO]` comments in the source file.

### Why we chose HTTP + shared secret for now

The agents are already isolated FastAPI services, so the HTTP boundary was free. Shared
secret is trivially auditable and requires no external dependencies. The real auth layer
(Entra ID + DPoP) requires provisioning one Entra app registration per agent, which is
an infrastructure step gated on environment setup — not something to block development on.

### Target state — Microsoft Entra ID + DPoP

Each agent will have its own Entra ID app registration (client_id/client_secret per
agent, stored in the environment variables under `AGENT_{NAME}_CLIENT_ID` etc.). The
orchestrator acquires tokens via the OAuth 2.0 client-credentials flow and passes them
as `Authorization: DPoP <token>` headers with a per-request DPoP proof.

DPoP (RFC 9449) binds each access token to the requesting agent's EC P-256 key pair,
preventing token theft and replay. The receiving agent validates the token and proof via
`DPoPAuthMiddleware` (`src/common/auth/middleware.py`) before the request reaches the
handler.

### Token revocation and the introspection gap

Entra ID does not support RFC 7662 token introspection. Tokens are self-contained JWTs
validated locally by each resource server — there is no callback to Entra to confirm a
token is still valid. A revoked token remains usable until its `exp` claim passes.

DPoP partially closes this gap:
- The token is bound to the holder's private key, so intercepting it alone is not enough
- A fresh proof is required on every request; proofs expire within 60–120 seconds
- Each proof is bound to a specific HTTP method + URI, so it cannot be reused across calls

**Combined mitigation strategy for production:**

| Layer | Measure |
|---|---|
| Token lifetime | 15-minute TTL on all agent access tokens — limits the blast radius of a leaked token |
| DPoP binding | Stolen token is useless without the corresponding private key |
| CAE | Continuous Access Evaluation — resource servers subscribe to Entra revocation events and reject tokens inline (deferred; see §12) |
| Circuit breaker | Orchestrator marks the `run_id` as cancelled in `FernetStorage`; agents check run state before processing and refuse work even if their token is still technically valid |

The circuit breaker is the most immediately actionable layer. It requires no additional
infrastructure: the storage layer already exists, cancelling a run is a single
`storage.save("runs", run_id, {..., "status": "cancelled"})` call, and agents can check
that field before performing any work. This makes the work item — not the credential —
the revocable unit, which is effective regardless of token lifetime.

The full rollout plan is in `DPOP_IMPLEMENTATION_GUIDE.md` (§3.6 covers the revocation
strategy in detail). All stub functions in `src/common/auth/` are tagged `[DPOP-TODO]`
to mark the exact insertion points.

---

## 5. AAP Auth Layer

**File:** `src/auth/token_service.py`

The Agent Authorization Protocol (AAP) layer mints, exchanges, and validates JWTs that
carry authorization context through the agent call chain. It is separate from the Entra
ID layer: Entra authenticates *who is calling*; AAP carries *what they are allowed to do
and in what context*.

### RFC reference

The delegation model is inspired by OAuth 2.0 Token Exchange (RFC 8693), with custom
claims for agent-specific metadata. The `cnf` claim binding (RFC 7800) is deferred to
the DPoP phase — see below.

### Seven claim sections

Every AAP token must contain all seven sections or `validate_token()` rejects it:

| Section | Contents |
|---|---|
| `aap_agent` | `agent_name`, `model_provider`, `model_id` — identity of the agent currently holding the token |
| `aap_task` | `task_id`, `task_description`, `data_sensitivity` — the task the token authorises |
| `aap_capabilities` | List of allowed operations for the current agent |
| `aap_oversight` | `oversight_required` (bool), `oversight_level` — human-in-the-loop requirements |
| `aap_delegation` | `depth`, `chain` (ordered list of agents), `max_depth` — delegation history |
| `aap_context` | Arbitrary key-value context (run_id, topic, etc.) |
| `aap_audit` | `token_id` (UUID), `minted_at` (epoch) — immutable audit trail |

### Delegation mechanics

- `mint_token()` creates a token at depth 0 with a one-entry chain.
- `exchange_token()` increments depth and appends the new agent to the chain. It
  preserves `sub`, `aap_task`, and `aap_oversight` verbatim — those bind to the task,
  not the agent.
- When `depth >= max_depth`, `exchange_token()` raises `ValueError`. Default
  `MAX_DEPTH = 3`.
- `aap_capabilities` is replaced on exchange — the new agent gets only its own
  capabilities, not the caller's.

### Current signing algorithm

HS256 (HMAC-SHA256) with a shared secret (`MAS_SECRET_KEY`). Implemented using stdlib
`hmac` + `hashlib`; no external JWT library required.

### DPoP deferred — reason

Completing DPoP requires the `cryptography` package (EC P-256 key generation) and
`PyJWT>=2.8` (asymmetric signing), plus one Entra app registration per agent. Blocking
the AAP token layer on that infrastructure work would have delayed development. The
current HS256 implementation is fully functional for development and testing. Switching
to DPoP-bound tokens is a targeted swap at the three `[DPOP-TODO]` points in
`token_service.py`: cnf claim injection on mint, proof validation on exchange, and
cnf/jkt binding on validate.

### Entra ID target

Once DPoP is active, `mint_token()` will accept the minting agent's DPoP public key and
embed it as `cnf: {jwk: ...}`. `validate_token()` will verify the `cnf/jkt` thumbprint
against the DPoP proof on every call. This makes each token cryptographically bound to
the specific agent key that minted or received it.

---

## 6. Model Split

### Reasoning agents — Gemma 4 (local via Ollama)

`heat_scorer`, `phase1_judge`, and `relevance_gate` use Gemma 4 served locally by
Ollama (`OLLAMA_BASE_URL=http://localhost:11434`). These agents perform structured
scoring and classification tasks — they produce JSON with numeric scores and short
rationale strings. Local inference eliminates per-call API costs for the highest-volume
part of the pipeline (every candidate article passes through all three).

### User-facing agents — Claude Sonnet (Anthropic API)

`summarizer` and `reviewer` use Claude Sonnet via the Anthropic API
(`ANTHROPIC_API_KEY`). These agents produce the prose that end users read. Claude Sonnet
is used here because summary quality and editorial coherence matter more than throughput,
and these agents see far fewer calls than the Phase 1 scoring agents (only the approved
shortlist reaches Phase 2).

### Swapping models — `MODEL_OVERRIDE`

Set `MODEL_OVERRIDE` in `.env` to override the model for all LLM calls globally. The
override is resolved by `resolve_model()` in `src/common/agent_bootstrap.py`, which is
called inside each affected agent's `run_agent()` entry point.

**Exempt agents:** `summarizer` and `reviewer` always use their registry-configured model
regardless of `MODEL_OVERRIDE` — their output is user-facing prose where model quality
is fixed by product requirements, not development convenience.

**Affected agents:** `heat_scorer`, `filter_agent`, `selector`, `phase1_judge`, and
`relevance_gate` (when built).

**Provider routing:** `resolve_model()` returns a `(provider, model_id)` tuple. Each
affected agent dispatches LLM calls to either `_call_ollama()` or `_call_anthropic()`
based on the resolved provider. Both are LangSmith-traced with `@traceable`.

**Known override values and their mappings:**

| `MODEL_OVERRIDE` value | Provider | Resolved model ID |
|---|---|---|
| `claude-haiku` | `anthropic` | `claude-haiku-4-5-20251001` |
| `claude-haiku-4-5-20251001` | `anthropic` | `claude-haiku-4-5-20251001` |
| `claude-sonnet` | `anthropic` | `claude-sonnet-4-6` |
| `gemma4:e4b` | `ollama` | `gemma4:e4b` |
| Any other `claude-*` string | `anthropic` | the string as-is |
| Any other string | `ollama` | the string as-is |

Leave `MODEL_OVERRIDE` blank (or unset) to use per-agent registry defaults.

---

## 7. PII/PHI Handling

**File:** `src/common/pii_scrubber.py`

### Engine

Microsoft Presidio with spaCy NLP. The NLP backend loads `en_core_web_lg` at startup,
falling back through `en_core_web_md` → `en_core_web_sm` if larger models are not
installed.

### Detected entity types

Standard PII plus healthcare-specific PHI: `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`,
`US_SSN`, `UK_NHS`, `MEDICAL_LICENSE`, `LOCATION`, `DATE_TIME`, `NRP`, `IP_ADDRESS`,
`CREDIT_CARD`, `IBAN_CODE`.

### Scrub at ingestion

`scrub_text()` is called on article content before it enters any agent. Detected
entities are replaced with `<ENTITY_TYPE>` placeholders (e.g. `<PERSON>`, `<US_SSN>`).
The original content is never stored or forwarded downstream.

### Never log content

`_SafeJSONFormatter` in `src/common/observability.py` blocks the following keys from
all structured log records: `content`, `text`, `body`, `article`, `summary`, `raw`,
`html`, `snippet`. Log records that include these keys have their values stripped before
emission. Entity counts (how many of each type were detected) may be logged; the
content that triggered them may not.

---

## 8. Observability Stack

### Tracing, metrics, and logs

| Component | Purpose | Local URL |
|---|---|---|
| OTel Collector | Receives OTLP spans from all agents | gRPC :4317, HTTP :4318 |
| Jaeger | Distributed trace visualisation | http://localhost:16686 |
| Prometheus | Metrics scraping and storage | http://localhost:9090 |
| Loki | Structured log aggregation | internal |
| Grafana | Unified dashboards (traces + metrics + logs) | http://localhost:3000 |

Start the observability stack: `docker compose up -d`. Configs are in `configs/`.
Start the full agent stack: `docker compose --profile agents up -d` (requires agent images to be built first — Dockerfiles are not yet committed; see §12).

### Network segmentation

`docker-compose.yml` defines four Docker networks that reflect the trust tiers of the system. They map directly to Azure subnets + NSG rules for the production deployment.

```
mas-identity       (internal: true)  ← auth-server, registry
       │
mas-orchestration  (internal: true)  ← orchestrators (both phases)
       │
mas-compute        (internal: true)  ← all eight worker agents
       │
mas-observability  (external)        ← OTel, Jaeger, Prometheus, Grafana, Loki
```

**Trust rules enforced by network membership:**

| Source tier | Can reach | Cannot reach |
|---|---|---|
| Worker agents (mas-compute) | mas-observability (OTel push) | mas-identity, mas-orchestration |
| Orchestrators | mas-identity (token exchange), mas-compute (agent dispatch), mas-observability | — |
| auth-server / registry | mas-observability | mas-orchestration, mas-compute |
| Observability services | — (receive only) | mas-identity, mas-orchestration, mas-compute |

`internal: true` means Docker removes the default gateway from those networks — containers cannot reach the Docker host's external interface even if they try. In production on Azure this maps to:

- A dedicated subnet per tier with no internet route table entry
- NSG inbound rules that allow only the specific source subnet + port combinations listed in the `[NETWORK-SEGMENTATION-TODO]` comments on each service in `docker-compose.yml`
- No NSG rule permits worker agents (Compute subnet) to reach the Identity subnet under any condition

**Why workers are never on mas-identity:** Worker agents process external content (Tavily results, article bodies) and call external APIs (Anthropic, Ollama). They have the largest attack surface. Excluding them from `mas-identity` means that a prompt-injection attack or supply-chain compromise in a worker agent cannot reach the token service or agent registry — the network enforces the boundary regardless of application-layer controls.

Each agent calls `setup_telemetry(service_name)` on startup
(`src/common/observability.py`). This registers an `OTLPSpanExporter` pointed at
`OTEL_EXPORTER_OTLP_ENDPOINT` (default `http://localhost:4317`) and wires a
`BatchSpanProcessor`. The `service.name` resource attribute is read from the
`SERVICE_NAME` env var (with the agent module name as fallback) so traces in Jaeger
are grouped per agent. Each docker-compose agent service sets `SERVICE_NAME` explicitly.

`setup_metrics(service_name)` registers an `OTLPMetricExporter` on the same endpoint
with a 10-second `PeriodicExportingMetricReader`. Metrics are namespaced `mas.egress.*`
and exported through the OTel collector to Prometheus.

The OTel collector config (`configs/otel-collector-config.yaml`) includes a `resource`
processor that preserves `service.name` from the application and stamps
`deployment.environment=local-dev` on all telemetry. The resource processor runs before
the Jaeger and Prometheus exporters in all three pipelines (traces, metrics, logs).

Egress spans emitted by the search worker:
- `egress.tavily.search` — wraps each Tavily API call with query_length, response_time_ms, and results_returned. Never captures query content.
- `egress.http.fetch` — wraps each article fetch with hostname only (not full URL), pii_count, scrubbed, and injection_detected flags.

Egress metrics (Prometheus, via OTel collector):
- `news_mas_mas_egress_tavily_calls_total`
- `news_mas_mas_egress_tavily_errors_total`
- `news_mas_mas_egress_tavily_query_length` (histogram)
- `news_mas_mas_egress_tavily_response_time_ms` (histogram)
- `news_mas_mas_egress_fetch_calls_total`
- `news_mas_mas_egress_injection_detected_total`

Prometheus alerting rules are in `prometheus/alerts.yml` (mounted into the Prometheus
container). Alerts: `EgressInjectionDetected` (high), `TavilyAnomalousQueryLength` (medium),
`TavilyHighCallVolume` (medium), `InjectionDetectedInRun` (high — fires when any run has
injection-flagged results), `LowSourceConfidence` (medium — fires when the 30-minute average
`confidence_ratio` drops below 0.3, indicating a systematic source quality issue).

Grafana dashboard `configs/grafana/provisioning/dashboards/egress.json` shows Tavily
call rate, query length distribution, fetch response time, injection detection count,
and error rate.

### Observability split — LangSmith vs OTel/Jaeger/Grafana

The two stacks are complementary and run simultaneously:

| Stack | Scope | What it captures |
|---|---|---|
| OTel → Jaeger / Grafana | Infrastructure | HTTP latency, span trees across agents, error rates, rate-limit hits, system metrics |
| LangSmith | LLM calls | Prompt/completion pairs, token counts, per-call latency, prompt version, eval scores |

Neither replaces the other. Jaeger shows you why a request was slow across agent hops;
LangSmith shows you what the model received and produced and how quality changed between
prompt versions.

### LangSmith configuration

`configure_langsmith()` in `src/common/observability.py` maps `LANGSMITH_API_KEY` and
`LANGSMITH_PROJECT` to the `LANGCHAIN_*` env vars that the LangChain/LangSmith SDK reads
at call time. It is called from every LLM-agent `main.py` at startup alongside
`setup_telemetry()`. If `LANGSMITH_API_KEY` is absent the function logs a warning and
returns `False` — agents start normally with tracing disabled (offline dev).

### LLM run tagging

Every LLM call passes `config=make_run_config(...)` (from `src/common/prompt_loader.py`).
This attaches four tags to each LangSmith run for filtering:

| Tag / metadata key | Value |
|---|---|
| `agent:{name}` | Agent that made the call |
| `prompt:{version}` | Exact prompt version file (e.g. `v1.0`) |
| `run:{run_id}` | Pipeline run UUID — correlates with OTel traces and `RunRepository` |
| `topic:{topic_id}` | Topic being processed (when available) |

### LangSmith as the offline eval framework

LangSmith Datasets + Evaluators are the planned path for the offline eval framework
(currently deferred in §12). Labelled examples from real runs are added to a LangSmith
Dataset; evaluators run against new prompt versions on the same inputs. Because every
run is already tagged with `prompt_version` and `run_id`, building a labelled dataset
requires only selecting runs in the LangSmith UI and exporting them — no separate
ingestion job.

---

## 9. Context Window Strategy

### Current — Hybrid A/B

Phase 1 agents (scoring, filtering, judging) operate on structured metadata fields
(title, source, published_at, heat_score). Full article content is not passed into
context for these agents — they reason from the signals, not the text.

Phase 2 agents (summarizer, reviewer) receive the full article content in context,
because they are producing prose from it. The summarizer accepts a list of
`SearchResult` sources (one per approved article for a topic) or falls back to a
single `RawArticle`. The reviewer's input is the article + the summary under
evaluation.

The summarizer uses a two-tier context strategy controlled by `TOKEN_BUDGET_SOURCES`
(default 16 000 tokens):
- **Direct (Option A):** total estimated tokens ≤ budget → single Claude Sonnet call
  with all sources in context.
- **Map-reduce (Option B):** exceeds budget → each source is mini-summarized
  concurrently (max 150 words each, max 512 tokens), then a synthesis call combines
  the mini-summaries. Original source URLs are pinned through the mini-step so
  citations in the final output always reference the original articles.

### Option C — Deferred

Option C is a retrieval-augmented approach where Phase 2 agents receive additional
context chunks retrieved from a vector store (past digests, background knowledge) in
addition to the article content. This would improve summary quality for recurring topics
by grounding the agent in prior coverage.

Option C is deferred because it requires a vector store, an embedding pipeline, and an
ingestion job for historical digests — infrastructure that is not yet in scope. The
current hybrid A/B approach produces acceptable quality without retrieval. Option C will
be revisited once the core pipeline is stable and feedback data (§11) shows where
quality degrades most.

---

## 10. Prompt Versioning

**File:** `src/common/prompt_loader.py`  
**Directory:** `prompts/`

### Structure

Each agent's prompts live under `prompts/<agent_name>/`. A prompt version is a YAML
file named `v<major>.<minor>.yaml`:

```
prompts/
└── heat_scorer/
    └── v1.0.yaml
```

### YAML schema

```yaml
name: <agent_name>
version: "<semver string>"
description: "<one-line purpose>"

injection_defense_preamble: |
  # Hardened preamble inserted before all system turns.
  # Contains absolute instructions that cannot be overridden by user content.

system: |
  # Core system prompt for this agent.

user_template: |
  # Template for the user turn. Uses {variable} placeholders filled at runtime.
```

### Loading

`load_prompt(agent_name, version="v1.0")` returns a dict with keys
`injection_defense_preamble`, `system`, `user_template`, and `_meta`. The `_meta` dict
carries `agent`, `version`, and `path`. Results are cached via `@lru_cache(maxsize=64)`.

### Version tagging on LangSmith traces

Every LLM call attaches `_meta["version"]` as metadata on the LangSmith run. This means
every trace in LangSmith is tagged with the exact prompt version that produced it,
making prompt regression analysis straightforward: filter by `version` in the LangSmith
UI to compare outputs before and after a prompt change.

---

## 11. Feedback Data Model

**File:** `src/common/storage.py`

User feedback feeds back into the system in two ways: direct corrections to individual
summaries, and signals about topic relevance over time.

### Storage backend

All persistent data is stored as **Fernet-encrypted JSON files** on disk, not in a
relational database. Each record is a separate `.enc` file under `DATA_DIR` (default
`./data/`). There is no schema migration step — adding a field to a record is a code
change only.

`FernetStorage` is the low-level layer. Four repository classes wrap it with
collection-specific method names:

| Repository | Collection dir | Stores |
|---|---|---|
| `UserRepository` | `data/users/` | User profiles and topic preferences |
| `RunRepository` | `data/runs/` | Run state and history (`RunState` schema) |
| `DigestRepository` | `data/digests/` | Final digest output keyed by `run_id` |
| `FeedbackRepository` | `data/feedback/` | All feedback signal types (see below) |

### Feedback signal types

Stored as dicts in `FeedbackRepository` with a `type` discriminator field:

| `type` value | Purpose |
|---|---|
| `summary_feedback` | Thumbs up/down + free-text note, keyed on `run_id` + `article_url` |
| `topic_relevance` | Per-topic relevance score over time; weights future `SearchWorker` queries |
| `missed_topic` | Topic the user flagged as absent from a digest; injected into the next run |
| `candidate_override` | Article manually added or removed; used to tune `HeatScorer` and `RelevanceGate` thresholds |

### Key management

`FERNET_KEY` is a URL-safe base64-encoded 32-byte key (AES-128-CBC + HMAC-SHA256).
Generate one with `python scripts/generate_key.py` and add it to `.env`. If `FERNET_KEY`
is not set, `FernetStorage` raises `RuntimeError` at construction time.

### Why not PostgreSQL

PostgreSQL + Alembic was the original plan. It was replaced by `FernetStorage` to
eliminate the database infrastructure dependency for early development. The repository
interface (`save`, `load`, `list`, `delete`) is intentionally abstract: a
`PostgresStorage` class implementing the same four methods could replace `FernetStorage`
without changing any repository or caller code.

---

## 12. Key Deferred Decisions

| Decision | Blocked on | Notes |
|---|---|---|
| **DPoP / Entra ID rollout** | Infrastructure: 8 Entra app registrations, `cryptography` + `PyJWT` packages | Full plan in `DPOP_IMPLEMENTATION_GUIDE.md`. All insertion points tagged `[DPOP-TODO]` in source. Replace `LocalDevIdentityProvider` with `AzureManagedIdentityProvider` in Phase 2 — see `[WORKLOAD-IDENTITY-TODO]` in `src/auth/workload_identity.py`. |
| **Option C retrieval** | Vector store + embedding pipeline + historical digest ingestion job | Deferred until core pipeline is stable and feedback data shows where quality degrades. |
| **User feedback UI** | Product decision: web UI vs. email reply vs. API | Storage layer is built (§11); no frontend work started. |
| **Offline eval framework** | Feedback data accumulation | Planned as LangSmith Datasets + Evaluators (see §8). Every run is already tagged with `prompt_version` and `run_id`; building a labelled dataset requires only selecting and exporting runs from the LangSmith UI. Gated on accumulating enough real-run data. |
| **AAP → DPoP token binding** | Same as DPoP rollout | Three `[DPOP-TODO]` points in `src/auth/token_service.py` mark the exact swap. HS256 shared-secret is interim only. |
| **CAE (Continuous Access Evaluation)** | Entra CAE subscription + resource server event handler | Closes the introspection gap — tokens can be invalidated mid-lifetime. See §4 and `DPOP_IMPLEMENTATION_GUIDE.md §3.6`. Implement after DPoP Phase 5 is complete. |
| **Feedback-driven threshold tuning** | `candidate_override` and `topic_relevance` data | `HeatScorer` and `RelevanceGate` thresholds are currently static constants; they will become learned from feedback. |
| **Agent + orchestrator Dockerfiles** | Implementation of individual agents | Network segmentation is fully defined in `docker-compose.yml` with `profiles: [agents]`. Services start with `docker compose --profile agents up -d` once `Dockerfile.agent`, `Dockerfile.orchestrator`, `Dockerfile.auth-server`, and `Dockerfile.registry` are committed. Each service's `[NETWORK-SEGMENTATION-TODO]` comment documents its Azure NSG equivalent. |
| **Registry persistence** | FernetStorage migration | `src/registry/card_store.py` currently uses an in-memory dict. For production, swap for a `FernetStorage`-backed implementation using the same `save/get/update/list` interface. The interface is intentionally identical to `src/common/storage.py` repositories. |
| **Registry auth on startup degradation** | Product decision | Agents start in degraded mode if the registry is unreachable. In production this should be a hard failure. Implement a `BOOTSTRAP_REQUIRED=true` env flag that makes startup abort if `bootstrap_agent()` raises. |
| **src/common extraction into chassis libraries** | Phase 1 pipeline complete | `src/common` and `src/auth` will be extracted into five pip-installable packages: `mas-guard` (PHI scrubbing, injection detection, pipeline errors), `mas-observability` (OTel, safe logging, LangSmith), `mas-identity` (AAP token lifecycle, workload identity), `mas-memory` (encrypted topic memory + feedback storage adapters), `mas-learning` (contextual-bandit feedback-weighted scoring). Design in `CHASSIS_DESIGN.md`. Extraction order: mas-guard → mas-observability / mas-memory (parallel) → mas-identity → mas-learning (post-MVP). `mas-identity 1.0.0` gated on DPoP; `mas-learning 1.0.0` gated on ~500 feedback signals. |
| **TopicMemoryRepository + FeedbackProcessor** | React UI / feedback data accumulation | `TopicMemoryRepository` stores per-topic heat score history in Fernet-encrypted JSON (planned after Phase 1 complete). `FeedbackProcessor` applies incremental weight updates (learning rate 0.05, floor 0.1, ceiling 2.0) from `summary_feedback` / `topic_relevance` / `missed_topic` / `candidate_override` signals. `missed_topic` signals are flagged for manual review rather than auto-applied. Design in `CHASSIS_DESIGN.md §13`. |

---

## 13. Onboarding Note for Claude Code

When starting a new session on this project, read the following files in order to
restore full context before making any changes:

1. **This file** (`ARCHITECTURE.md`) — system design, decisions, and what is deferred.
2. **`DPOP_IMPLEMENTATION_GUIDE.md`** — detailed DPoP rollout plan; read before touching
   anything in `src/common/auth/` or `src/auth/`.
3. **`src/common/schemas.py`** — all Pydantic models; the shared contract between agents
   and orchestrators.
4. **`src/orchestrators/phase1/graph.py`** and **`src/orchestrators/phase2/graph.py`** —
   the LangGraph pipelines; understand the state shapes before modifying agent I/O.
5. **`.env.example`** — all environment variables with inline comments; check this
   before adding new config.

### Key conventions to respect

- **Never log raw content.** The `_SafeJSONFormatter` blocked-key list in
  `src/common/observability.py` is not exhaustive — when in doubt, log counts and IDs,
  not values.
- **Scrub before processing.** Call `scrub_text()` on any content that enters an agent
  from outside the system. Do not assume upstream agents already scrubbed it.
- **All `[DPOP-TODO]` comments are load-bearing.** They mark the exact points where
  the DPoP migration slots in. Do not remove them until the corresponding implementation
  is complete and the `test_token_service_contains_dpop_todo_comments` test in
  `tests/test_aap_tokens.py` is updated to reflect that.
- **Agent models are not yet wired in code.** The model split described in §6 is the
  intended design; the agent stubs currently raise `NotImplementedError`. When
  implementing an agent, check §6 for which model it should use and whether
  `MODEL_OVERRIDE` should take precedence.
- **Prompt files are versioned.** Never edit a deployed prompt file in place — create a
  new version file and update the `load_prompt()` call in the agent. This keeps
  LangSmith trace history coherent.

### Keeping this file current

**ARCHITECTURE.md is a living document. Update it whenever you make a change that
affects system design.**

Specifically, update this file when you:
- Add, remove, or rename an agent (§3)
- Change a model assignment or add `MODEL_OVERRIDE` behaviour (§6)
- Resolve a `[DPOP-TODO]` item or advance the auth layer (§4, §5, §12)
- Add a database table or change the feedback schema (§11)
- Change the context window strategy or implement Option C (§9)
- Add a new prompt file or change the versioning scheme (§10)
- Add or remove an observability component (§8)
- Promote a deferred decision to active (§12 — move it out of the table and into the
  relevant section with a note on what was decided)

Do not let implementation drift ahead of the document. A reader arriving cold should be
able to understand the current system from ARCHITECTURE.md alone without having to diff
the code. If you implement something that contradicts what is written here, fix the
document in the same commit.
