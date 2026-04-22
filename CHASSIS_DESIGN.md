# Chassis Design — Extracting src/common into Standalone Libraries

> **Status:** Design intent — no code changes. news-mas is the reference
> implementation. Extraction begins after the Phase 1 pipeline is complete
> and the library boundaries have been validated by real use inside news-mas.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Library: mas-guard](#2-library-mas-guard)
3. [Library: mas-observability](#3-library-mas-observability)
4. [Library: mas-identity](#4-library-mas-identity)
5. [Library: mas-memory](#5-library-mas-memory)
6. [Library: mas-learning](#6-library-mas-learning)
7. [Why Not Full RL?](#7-why-not-full-rl)
8. [Framework Adapter Pattern](#8-framework-adapter-pattern)
9. [Extraction Sequence and Prerequisites](#9-extraction-sequence-and-prerequisites)
10. [Versioning Strategy](#10-versioning-strategy)
11. [How news-mas Will Depend on Them](#11-how-news-mas-will-depend-on-them)
12. [Target User for Each Library](#12-target-user-for-each-library)
13. [news-mas Backlog: Memory and Learning](#13-news-mas-backlog-memory-and-learning)

---

## 1. Motivation

`src/common` currently does three unrelated jobs: content safety and pipeline
integrity (`security.py`, `pii_scrubber.py`, `data_quality.py`,
`pipeline_errors.py`); structured observability (`observability.py`,
`prompt_loader.py`); and agent identity and authorization (`auth/token_service.py`,
`auth/workload_identity.py`, `agent_bootstrap.py`). These concerns are bundled
together only because news-mas was the first system to need them all. Any other
FastAPI + LangGraph pipeline built for healthcare would need the same three layers
and would have to reimplement them from scratch.

Beyond the core three, two additional libraries address concerns that emerge once
a pipeline is running and accumulating real usage: cross-run memory and
feedback-driven learning. These are not extraction targets from existing code —
they are new packages that news-mas will build against once the Phase 1 pipeline
is stable and producing feedback signals.

The plan is to build or extract each concern into its own pip-installable package:

| Package | Source | Status | Concern |
|---|---|---|---|
| `mas-guard` | Extract from `security.py`, `pii_scrubber.py`, `data_quality.py`, `pipeline_errors.py`, `error_codes.py` | After Phase 1 complete | Content safety, PHI scrubbing, pipeline error protocol |
| `mas-observability` | Extract from `observability.py`, `prompt_loader.py` | After Phase 1 complete | OTel setup, safe structured logging, LangSmith integration |
| `mas-identity` | Extract from `auth/token_service.py`, `auth/workload_identity.py`, `agent_bootstrap.py` | After Phase 2 complete | AAP token lifecycle, workload identity, agent bootstrap |
| `mas-memory` | New — builds on `storage.py` patterns | v0.1.0 after Phase 1 complete | Cross-run topic memory, feedback persistence, storage adapters |
| `mas-learning` | New — post-MVP, requires feedback volume | Future | Feedback-weighted scoring, DSPy prompt optimisation, eval hooks |

The first three packages form a **chassis**: non-domain concerns any multi-agent
healthcare pipeline can plug in. The last two form a **learning layer**: optional
components that improve the pipeline from its own usage history. All five are
designed to be adopted independently — a team can use `mas-guard` alone without
touching `mas-memory`.

---

## 2. Library: mas-guard

### Problem

Multi-agent LLM pipelines that process external content have no standard way to
enforce four independent safety properties at once: prompt injection detection,
PHI removal before the content reaches any model, structured error contracts
that orchestrators can route on, and content-quality gates at each stage
boundary. Teams building these pipelines either skip one or more properties
(accepting the risk) or reimplement them in isolation without the healthcare
entity coverage or the tested error taxonomy that healthcare workloads require.

`mas-guard` solves this by providing all four as a cohesive library with
first-class adapters for FastAPI middleware and LangChain/LangGraph callbacks.

### Proposed Public API

```python
# ── Injection detection ──────────────────────────────────────────────────────

from mas_guard import detect_injection, normalize_input, is_safe_input

patterns: list[str] = detect_injection(text)       # matched pattern strings
clean: str           = normalize_input(text)        # NFKC normalisation only
safe: bool           = is_safe_input(text)          # True iff no patterns match

# ── PHI scrubbing ────────────────────────────────────────────────────────────

from mas_guard import scrub_text, detect_pii, ScrubResult

detections: list[dict] = detect_pii(text)          # [{type, start, end, score}]
scrubbed: str          = scrub_text(text)           # replaces spans with <TYPE>

# Richer return type for callers that need to count or log:
result: ScrubResult = scrub_text_with_metadata(text)
# result.text          — scrubbed string
# result.entity_counts — {entity_type: count}
# result.was_altered   — bool

# ── Pipeline error protocol ──────────────────────────────────────────────────

from mas_guard import (
    PipelineError,
    AgentResult,
    PipelineException,
    ErrorSeverity,
    ResultConfidence,
)

# PipelineError — structured error record (error_code, severity, agent_id,
#                 run_id, topic_id, message, retry_hint, context).
# AgentResult   — universal agent return envelope (success, data, error, warnings).
# PipelineException — wraps PipelineError for raise-style propagation.
# ErrorSeverity — RETRYABLE | SKIP | FATAL | DEGRADED
# ResultConfidence — FULL | SNIPPET | PARTIAL | INJECTED | SCRUBBED

# ── Data quality validators ───────────────────────────────────────────────────

from mas_guard import DataQualityValidator

validator = DataQualityValidator(agent_id="search-worker")

err = validator.validate_search_results(results, min_results=3, topic_id=t, run_id=r)
err = validator.validate_filtered_results(filtered_count=2, original_count=10, ...)
err = validator.validate_heat_score(score=0.0, topic_id=t, run_id=r)
err = validator.validate_candidates(candidates, min_candidates=1, run_id=r)
err = validator.validate_summary(summary, min_length=50, max_length=500, ...)
err = validator.validate_citations(citations, sources=sources, ...)
err = validator.validate_constraints_respected(content, constraints=[...], ...)

# All return PipelineError | None — None means the check passed.

# ── Run-level circuit breaker ─────────────────────────────────────────────────

from mas_guard import is_run_active

active: bool = is_run_active(run_id, storage=storage, aap_claims=claims, logger=log)
# Returns False and logs a WARNING when the run has been cancelled/killed.

# ── Error code constants ──────────────────────────────────────────────────────

from mas_guard.error_codes import (
    NO_RESULTS, INSUFFICIENT_RESULTS, ALL_FILTERED,
    HEAT_SCORE_TOO_LOW, NO_VIABLE_CANDIDATES,
    SUMMARY_TOO_SHORT, SUMMARY_TOO_LONG,
    CITATION_INVALID, CONSTRAINT_VIOLATED,
    INJECTION_DETECTED, AUTH_FAILED,
    # ... full set
)
```

### Framework Adapters

See [§8](#8-framework-adapter-pattern) for the general adapter pattern.
`mas-guard`-specific adapters:

```python
# FastAPI — middleware that scrubs request bodies before they reach handlers
from mas_guard.adapters.fastapi import (
    ScrubMiddleware,           # scrubs body fields in-place; logs entity counts
    InjectionGuardMiddleware,  # rejects requests with injection patterns; 422 response
)

app.add_middleware(ScrubMiddleware, fields=["content", "query", "text"])
app.add_middleware(InjectionGuardMiddleware)

# LangChain — callback that intercepts inputs/outputs at each LLM step
from mas_guard.adapters.langchain import GuardCallback

chain.invoke(input, config={"callbacks": [GuardCallback(scrub_inputs=True)]})

# LangGraph — node wrapper that scrubs state fields before each node runs
from mas_guard.adapters.langgraph import guarded_node

@guarded_node(scrub_fields=["content", "snippet"])
def my_node(state: GraphState) -> GraphState:
    ...
```

### Healthcare Positioning (README Lead)

The mas-guard README should open with this framing:

> LLM pipelines in healthcare process content that may contain PHI — patient
> names, SSNs, NHS numbers, medical licence identifiers — alongside adversarial
> content planted in external sources. `mas-guard` is the safety layer between
> your ingestion boundary and your models: it scrubs PHI before anything reaches
> a prompt, detects prompt-injection attempts before they reach an LLM, and
> gives your orchestrator a typed error vocabulary to route on rather than
> collapsing all failures into a generic exception. It is the library that makes
> HIPAA-adjacent multi-agent pipelines auditable.

---

## 3. Library: mas-observability

### Problem

Multi-agent pipelines need three distinct observability layers simultaneously:
distributed tracing across HTTP boundaries (OTel/Jaeger), LLM-call-level
tracing with prompt/completion pairs (LangSmith), and structured logs that are
safe for healthcare environments — meaning they must block PHI-adjacent field
names automatically, not by developer discipline alone. No single existing
library provides all three in a configuration that is ready for a multi-agent
pipeline out of the box. Teams wire each independently and end up with
inconsistent trace context propagation, logs that occasionally include content
fields, and LangSmith runs that have no `run_id` correlation with OTel traces.

`mas-observability` provides the full three-layer stack as a single
`setup_observability(service_name)` call, with safe structured logging and
LangSmith configuration built in.

### Proposed Public API

```python
# ── Setup ────────────────────────────────────────────────────────────────────

from mas_observability import setup_observability, teardown_observability

tracer, meter = setup_observability("news-mas-search-worker")
# Reads SERVICE_NAME, SERVICE_VERSION, ENVIRONMENT, OTEL_EXPORTER_OTLP_ENDPOINT
# from env with sensible defaults. Safe to call multiple times — idempotent.

# ── Individual setup (for callers that don't want the full stack) ─────────────

from mas_observability import setup_telemetry, setup_metrics, configure_langsmith

tracer = setup_telemetry("my-service")   # OTel trace + BatchSpanProcessor
meter  = setup_metrics("my-service")    # OTel metrics + PeriodicExportingMetricReader
ok     = configure_langsmith()          # maps LANGSMITH_* to LANGCHAIN_*; returns bool

# ── Tracer and meter accessors (after setup) ──────────────────────────────────

from mas_observability import get_tracer, get_meter

tracer = get_tracer("my-service")
meter  = get_meter("my-service")

# ── Safe structured logging ───────────────────────────────────────────────────

from mas_observability import get_logger, BLOCKED_LOG_KEYS

logger = get_logger(__name__)
# Returns a logger backed by _SafeJSONFormatter. Values whose keys appear in
# BLOCKED_LOG_KEYS (content, text, body, article, summary, raw, html, snippet)
# are stripped before emission — content never appears in logs regardless of
# what callers pass.

# Additional blocked keys can be registered at startup:
from mas_observability import register_blocked_keys
register_blocked_keys({"patient_id", "diagnosis"})

# ── Pipeline error logging helpers ────────────────────────────────────────────

from mas_observability import log_pipeline_error, record_span_error, log_with_run_id

log_pipeline_error(logger, error)          # structured log for PipelineError
record_span_error(span, error)             # set OTel span status + safe attributes
log_with_run_id(logger, level, msg, run_id, **kwargs)

# ── LangSmith run config helper ───────────────────────────────────────────────

from mas_observability import make_run_config

config = make_run_config("heat_scorer", "v1.0", run_id="run-001", topic_id="AI")
# Returns {"run_name": ..., "tags": [...], "metadata": {...}} for LangChain
# chain.invoke(input, config=config)
```

### Framework Adapters

```python
# FastAPI — middleware that opens an OTel span per request and injects run_id
# into the span context from the X-Run-ID header (or generates one).
from mas_observability.adapters.fastapi import (
    TelemetryMiddleware,      # span per /run request; sets run_id as span attribute
    StructuredLogMiddleware,  # attaches run_id + agent_id to every log record
)

app.add_middleware(TelemetryMiddleware, service_name="news-mas-heat-scorer")
app.add_middleware(StructuredLogMiddleware)

# LangChain — callback that attaches make_run_config metadata to every LLM run
from mas_observability.adapters.langchain import ObservabilityCallback

chain.invoke(input, config={
    "callbacks": [ObservabilityCallback(agent_name="heat_scorer", run_id="run-001")]
})

# LangGraph — graph decorator that adds tracing to every node transition
from mas_observability.adapters.langgraph import traced_graph

@traced_graph(service_name="news-mas-orchestrator-1")
def build_graph() -> StateGraph:
    ...
```

---

## 4. Library: mas-identity

### Problem

Agent-to-agent authentication in multi-agent pipelines is usually bolted on
as an afterthought — a shared API key or a hardcoded `Authorization` header —
because the correct solution (short-lived tokens that carry delegation history,
task context, capability assertions, and data-sensitivity metadata through the
entire call chain) requires significant infrastructure work before it is usable.
Teams either ship with no auth at all (for speed) or bind directly to a specific
cloud provider's identity service (creating vendor lock-in and making local
development painful). Neither option is acceptable for healthcare workloads where
the audit trail of which agent authorised what action on whose behalf must be
recoverable from logs.

`mas-identity` provides the AAP (Agent Authorization Protocol) token layer —
the mint / exchange / validate lifecycle — as a standalone package that works
with a shared secret in development and slots into any OIDC-compatible workload
identity system in production without changing calling code.

### Proposed Public API

```python
# ── AAP token lifecycle ───────────────────────────────────────────────────────

from mas_identity import mint_token, exchange_token, validate_token

token: str = mint_token(
    subject="orchestrator-1",
    agent_name="search-worker",
    model_provider="tavily",
    model_id="",
    capabilities=["search.web"],
    task_id="task-001",
    task_description="Fetch AI news articles",
    data_sensitivity="PHI",
    oversight_required=True,
    oversight_level="human-in-loop",
    context={"run_id": "run-001", "topic": "AI safety"},
    secret_key=os.getenv("MAS_SECRET_KEY"),
)

delegated: str = exchange_token(
    token=token,
    new_agent_name="heat-scorer",
    new_model_provider="ollama",
    new_model_id="gemma4:e4b",
    new_capabilities=["score.heat"],
    secret_key=os.getenv("MAS_SECRET_KEY"),
)

claims: dict = validate_token(token, secret_key=os.getenv("MAS_SECRET_KEY"))
# Raises ValueError for expired token, bad signature, or missing AAP sections.

# ── Workload identity providers ───────────────────────────────────────────────

from mas_identity import (
    WorkloadIdentityProvider,       # abstract base
    LocalDevIdentityProvider,       # shared-secret — dev/test only
    get_identity_provider,          # factory: reads WORKLOAD_IDENTITY_PROVIDER env var
)

provider = get_identity_provider()
identity_token: str   = provider.get_identity_token()
aap_token: str        = provider.exchange_for_aap_token(identity_token, capabilities)
claims: dict          = provider.get_workload_claims()
provider.validate_incoming_token(presented_token)   # raises ValueError on failure

# Production providers (post-DPoP — see §4.1 below):
from mas_identity import (
    AzureManagedIdentityProvider,   # IMDS → Entra federation → AAP
    AWSSTSFederationProvider,       # STS OIDC → Entra federation → AAP
    GCPWorkloadIdentityProvider,    # GCP metadata → Entra OIDC → AAP
)

# ── Agent bootstrap ───────────────────────────────────────────────────────────

from mas_identity import bootstrap_agent
from mas_identity import AgentConfig

config: AgentConfig = await bootstrap_agent(
    registry_url="http://registry:8000",
    identity_provider=provider,
    agent_id="heat-scorer",
)
# config.model_provider, config.model_id, config.capabilities,
# config.token_budget, config.prompt_version, config.network_tier

# ── Constants ─────────────────────────────────────────────────────────────────

from mas_identity import MAX_DEPTH         # default delegation chain depth limit
from mas_identity import REQUIRED_SECTIONS # the seven AAP claim sections
```

### Framework Adapters

```python
# FastAPI — middleware that validates the X-MAS-Secret header (shared secret,
# dev) or the Authorization: DPoP header (production) on every /run request.
from mas_identity.adapters.fastapi import (
    SecurityHeaderMiddleware,   # current: validates X-MAS-Secret
    DPoPAuthMiddleware,         # target: validates DPoP-bound Entra tokens
)

app.add_middleware(SecurityHeaderMiddleware)  # swap for DPoPAuthMiddleware in prod

# Context manager for token exchange in orchestrator nodes:
from mas_identity.adapters.langgraph import delegated_token

async with delegated_token(
    incoming_token=state.aap_token,
    new_agent="heat-scorer",
    capabilities=["score.heat"],
    secret_key=os.getenv("MAS_SECRET_KEY"),
) as token:
    response = await heat_scorer_client.run(inp, headers={"X-AAP-Token": token})

# LangChain callback that validates the token on the model's input and attaches
# the delegation chain to each LangSmith run as metadata:
from mas_identity.adapters.langchain import TokenAuditCallback

chain.invoke(input, config={
    "callbacks": [TokenAuditCallback(token=aap_token)]
})
```

### 4.1 DPoP Prerequisite Note for mas-identity 1.0.0

**`mas-identity` will not reach version `1.0.0` until DPoP is implemented.**

The current HS256 shared-secret signing in `token_service.py` is a development
convenience. It provides no platform attestation, no cryptographic binding to a
specific workload, and no forward secrecy. Any holder of `MAS_SECRET_KEY` can
mint or forge tokens.

`mas-identity 1.0.0` requires all of the following to be complete:

| Prerequisite | What it enables |
|---|---|
| Entra ID app registration per agent (8 registrations) | Each agent has a unique cryptographic identity, not a shared secret |
| `AzureManagedIdentityProvider` implemented | Agents prove identity via IMDS without credentials in env vars |
| `DPoPAuthMiddleware` implemented and active | Each token is bound to the requesting agent's EC P-256 key |
| `cnf` claim in `mint_token` and `validate_token` | Tokens are cryptographically bound from mint to receipt |
| All `[DPOP-TODO]` comments in `token_service.py` resolved | The three injection points (mint cnf, exchange proof, validate binding) are live |

The three `[DPOP-TODO]` comments in `src/auth/token_service.py` mark the exact
insertion points. `DPOP_IMPLEMENTATION_GUIDE.md` contains the full migration
checklist. Do not remove those comments until the corresponding test in
`tests/test_aap_tokens.py::test_token_service_contains_dpop_todo_comments` is
updated to reflect that the migration is complete.

Until `1.0.0`, the version line is `0.x.y` and the README must carry a visible
`> ⚠ Pre-1.0: HS256 shared-secret only. Not for production PHI workloads.`
warning.

---

## 5. Library: mas-memory

### Problem

Agent systems that run repeatedly — daily digests, scheduled monitors, recurring
evaluations — have no standard way to carry knowledge from one run to the next.
Each run starts cold: the heat scorer has no memory that a topic has been
consistently low-quality for two weeks, the summariser has no context that users
always flag a particular source as irrelevant, and the pipeline has no record of
which topics have generated the most positive feedback over time. This amnesia
is fine for one-off pipelines but is a structural limitation for systems intended
to improve with use.

`mas-memory` provides the `TopicMemory` pattern and `FeedbackProcessor` as a
standalone library that any recurring agent system can adopt without coupling to
news-mas's domain logic.

### Proposed Public API

```python
# ── Topic memory ──────────────────────────────────────────────────────────────

from mas_memory import TopicMemory, TopicMemoryEntry

memory = TopicMemory(storage=storage_adapter)

# Write
memory.record_run(
    topic_id="AI safety",
    run_id="run-001",
    heat_score=0.82,
    approved=True,
    feedback_score=1,       # +1 thumbs up, -1 thumbs down, 0 no signal
)

# Read — returns historical context for a topic
entry: TopicMemoryEntry = memory.get(topic_id="AI safety")
# entry.topic_id
# entry.run_count          — total times this topic has been processed
# entry.avg_heat_score     — rolling average heat score across all runs
# entry.approval_rate      — fraction of runs that produced approved articles
# entry.feedback_score     — cumulative feedback signal (sum of +1/-1)
# entry.last_run_at        — datetime of most recent run
# entry.preference_weight  — float in [0.0, 2.0]; 1.0 = neutral

# Query
entries: list[TopicMemoryEntry] = memory.list_topics()
low_quality: list[str] = memory.topics_below_threshold(avg_heat_score=0.3)

# ── Feedback processor ────────────────────────────────────────────────────────

from mas_memory import FeedbackProcessor, FeedbackSignal

processor = FeedbackProcessor(
    storage=storage_adapter,
    learning_rate=0.05,     # gradual drift; see §7 for rationale
)

# Process a single feedback signal
processor.apply(FeedbackSignal(
    signal_type="summary_feedback",  # or "topic_relevance" | "missed_topic" | "candidate_override"
    topic_id="AI safety",
    run_id="run-001",
    article_url="https://...",
    value=1,               # +1 thumbs up, -1 thumbs down
    note="Good coverage of regulatory angle",
))

# Missed-topic signals are flagged for manual review (not auto-applied):
processor.apply(FeedbackSignal(
    signal_type="missed_topic",
    topic_id="AI safety",
    run_id="run-001",
    value=0,
    note="Should have included the EU AI Act ruling",
))
# Raises FlaggedForReview — caller must route to a review queue.
# These signals are the richest in the system; they deserve human attention
# before any automated response. See §13 for the news-mas backlog item.

# ── Storage adapters ──────────────────────────────────────────────────────────

from mas_memory.storage import EncryptedJsonAdapter   # current: FernetStorage pattern
from mas_memory.storage import RedisAdapter           # future
from mas_memory.storage import PostgresAdapter        # future

storage = EncryptedJsonAdapter(
    data_dir="./data/memory/",
    fernet_key=os.getenv("FERNET_KEY"),
)
```

### Storage adapter contract

All adapters implement the same four-method interface so the core library has no
storage dependency:

```python
class MemoryStorageAdapter(Protocol):
    def save(self, collection: str, id: str, record: dict) -> None: ...
    def load(self, collection: str, id: str) -> dict: ...
    def list(self, collection: str) -> list[str]: ...
    def delete(self, collection: str, id: str) -> None: ...
```

This mirrors the `FernetStorage` interface in `src/common/storage.py`, so
`EncryptedJsonAdapter` is a thin wrapper around the existing implementation.
`RedisAdapter` and `PostgresAdapter` are future extensions — the interface
is stable first, implementations second.

### Framework Adapters

`mas-memory` is deliberately framework-agnostic at its core. The only adapters
are convenience injectors for common frameworks:

```python
# LangGraph — state field that carries topic memory into the graph
from mas_memory.adapters.langgraph import memory_state_field

class Phase1State(TypedDict):
    topic_memory: Annotated[TopicMemoryEntry | None, memory_state_field()]

# Makes topic memory available to any node in the graph without explicit plumbing.
# The orchestrator populates it before the graph runs; nodes read it as needed.

# CrewAI — tool that exposes topic memory to agents
from mas_memory.adapters.crewai import TopicMemoryTool
# Works with any CrewAI agent that has tool access.
```

---

## 6. Library: mas-learning

### Problem

Teams that have accumulated enough feedback signals from a running pipeline want
to improve it from that signal without retraining models, writing new prompts
manually, or building a custom experiment framework. The common path is to ignore
the feedback (easiest) or to hand-tune thresholds based on intuition (fragile).
Neither scales. What is needed is an incremental, observable update mechanism
that adjusts scoring weights and prompts from real usage patterns, degrades
gracefully when signal volume is low, and never makes a change that cannot be
observed and rolled back.

`mas-learning` provides `FeedbackWeightedScoring`, DSPy prompt optimisation
hooks, and evaluation framework integration as an optional layer on top of
`mas-memory`. It is intentionally separate from `mas-memory` because most teams
should run `mas-memory` for a while before enabling learning — the data volume
requirement is real (see §7).

> **Prerequisite:** Sufficient feedback data volume. Recommend 500+ signals
> before enabling any auto-tuning. See §7 for the full rationale and the
> signal-volume guidance.

### Proposed Public API

```python
# ── Feedback-weighted scoring ─────────────────────────────────────────────────

from mas_learning import FeedbackWeightedScoring

scorer = FeedbackWeightedScoring(
    memory=topic_memory,
    learning_rate=0.05,
    weight_floor=0.1,      # topic weight never drops below this
    weight_ceiling=2.0,    # topic weight never exceeds this
)

# Adjust a raw heat score by the topic's learned preference weight
adjusted: float = scorer.adjust(
    topic_id="AI safety",
    raw_heat_score=0.72,
)
# adjusted = raw_heat_score * entry.preference_weight (clamped to [0.0, 1.0])

# Update weights from a batch of new feedback signals
scorer.update_weights(signals=[...])   # emits OTel attributes for each change

# ── DSPy prompt optimisation integration ─────────────────────────────────────

from mas_learning import DSPyOptimiser

optimiser = DSPyOptimiser(
    prompt_dir="prompts/",
    dataset_name="news-mas-feedback",   # LangSmith dataset name
    metric="approval_rate",             # the signal to optimise for
)

# Run optimisation on a specific agent's prompt
result = await optimiser.optimise(
    agent_name="heat_scorer",
    current_version="v1.0",
    candidate_version="v1.1",
)
# result.new_prompt_path  — written to prompts/heat_scorer/v1.1.yaml
# result.delta_metric     — improvement in approval_rate on holdout set
# result.recommended      — bool; True if delta_metric > improvement_threshold

# ── Evaluation framework hooks ────────────────────────────────────────────────

from mas_learning import EvalHook, EvalResult

# Register a hook that runs offline evaluation after each pipeline run
hook = EvalHook(
    dataset=langsmith_dataset,
    evaluators=[approval_rate_evaluator, summary_quality_evaluator],
    run_threshold=10,      # only trigger after N new runs have accumulated
)

result: EvalResult = await hook.maybe_evaluate(run_id="run-001")
# result.triggered  — bool; False if threshold not yet met
# result.scores     — {evaluator_name: score} if triggered
# result.regression — bool; True if any score dropped below baseline
```

### Observability

Every weight change made by `FeedbackWeightedScoring.update_weights()` must be
logged via OTel and emitted as a structured log event so the change is
observable without inspecting the stored weights directly:

```
span: mas_learning.weight_update
  attributes:
    topic_id:       "AI safety"
    old_weight:     1.0
    new_weight:     1.05
    signal_count:   12
    learning_rate:  0.05
```

Weight changes are also versioned: each `update_weights()` call increments a
`weight_version` counter stored alongside the weights. Rolling back means
restoring the previous version record — no recomputation required.

---

## 7. Why Not Full RL?

This section explains the design choices behind `mas-learning` and provides the
honest framing that the library's README should use.

### What "RL for agents" usually means in practice

Most agent systems that describe themselves as using "reinforcement learning"
are actually implementing **weighted preference learning** — a form of online
learning that adjusts scoring weights incrementally from binary or ordinal
feedback signals (thumbs up/down, relevance ratings). This is not a criticism;
it is an appropriate choice for systems where:

- Feedback volume is modest (dozens to hundreds of signals per run cycle)
- The "reward" is a user preference, not an objective metric with a known optimum
- Model training infrastructure is not available or not warranted
- The system needs to explain its adjustments to non-ML stakeholders

True RL — specifically RLHF as applied to LLMs — requires a reward model trained
on human preference data, a fine-tuning step that updates model weights, and
training infrastructure (GPU cluster, distributed training framework). This is
not the situation for most production agent pipelines built on top of hosted LLM
APIs.

### What mas-learning actually implements

`mas-learning` implements the **contextual bandits** pattern:

- The "context" is the topic and its accumulated history (heat scores, feedback
  signals, approval rates from `mas-memory`)
- The "action" is the scoring weight adjustment applied to a topic's articles
- The "reward" is the feedback signal (+1 thumbs up, -1 thumbs down)
- Updates are incremental with a small learning rate — each signal shifts the
  weight slightly, not catastrophically
- No model training occurs; only scalar weights and prompt text are updated

Contextual bandits are well-studied in the academic literature (Langford &
Zhang 2007, Agarwal et al. 2014) and are the appropriate algorithm class for
this problem. Calling it "RL" would be technically inaccurate and would set
misleading expectations with engineering and compliance stakeholders.

### Signal volume guidance

Different update mechanisms require different minimum signal volumes before they
produce reliable improvements:

| Mechanism | Minimum signals | Notes |
|---|---|---|
| Weighted scoring (preference weights) | ~50 | Useful almost immediately; risk is over-fitting to early noise |
| Contextual bandits (mas-learning) | ~500 | Enough signals to distinguish genuine preference from random variance |
| True RLHF (fine-tuning) | ~10,000+ | Requires reward model training; out of scope for this system |

**Recommendation:** Run `mas-memory` and let weights accumulate passively until
500+ signals are available before enabling `FeedbackWeightedScoring` auto-tuning.
Use manual weight review (read the OTel spans) to catch obvious errors earlier.

### Why a small learning rate (0.05)

A learning rate of 0.05 means a single thumbs-up on a topic with a neutral
weight (1.0) moves the weight to 1.05 — a 5% increase. It would take 20
consecutive positive signals to double the weight to 2.0 (the ceiling). This
is intentionally slow:

- News topics are volatile — a topic that generates good articles for three
  weeks may go quiet without any signal from the user
- A large learning rate would allow a short burst of positive feedback to
  dominate the weight permanently, which is the wrong behaviour for a
  recurring pipeline
- Weights can be observed and audited via OTel; a slow drift is easier to
  explain to a compliance reviewer than a sudden jump

### Honest README framing

The mas-learning README should not claim "reinforcement learning" without
qualification. The recommended framing:

> `mas-learning` implements contextual bandits — incremental weight updates
> from user feedback signals, without model training or reward model inference.
> This is the appropriate algorithm for agent pipelines with modest feedback
> volumes where you want gradual, observable improvement rather than the
> infrastructure overhead of true RLHF.

---

## 8. Framework Adapter Pattern

Each library exposes its core functionality as plain Python — no framework
dependency — and provides optional framework adapters as extras. The adapter
pattern is consistent across all five packages so teams can adopt one without
the others and compose freely.

### Naming convention

```
mas_{library}.adapters.{framework}
```

Available framework targets: `fastapi`, `langchain`, `langgraph`, `crewai`.

### Adapter types

Every adapter falls into one of three categories:

| Category | Form | Purpose |
|---|---|---|
| **Middleware** | `class XxxMiddleware(BaseHTTPMiddleware)` | Request/response lifecycle hook in FastAPI. Runs before handlers for inbound validation (identity), before response for scrubbing (guard), or both for telemetry (observability). |
| **Context manager** | `async with xxx(...) as value:` | Wraps a single operation — typically a downstream agent call — with setup and teardown. Used for delegated token lifecycle (identity) and span scoping (observability). |
| **Callback / decorator** | `@xxx` or passed to `config={"callbacks": [...]}` | Hooks into LangChain's callback system or wraps a LangGraph node. Receives the same inputs/outputs as the underlying call but adds cross-cutting behaviour without modifying the chain or graph logic. |

### Dependency rule

Framework adapters are **extras**, not hard dependencies:

```toml
# pyproject.toml for each library
[project.optional-dependencies]
fastapi   = ["fastapi>=0.100", "starlette>=0.27"]
langchain = ["langchain-core>=0.1"]
langgraph = ["langgraph>=0.1"]
crewai    = ["crewai>=0.28"]
```

Install only what you need:

```bash
pip install mas-guard[fastapi]
pip install mas-observability[langchain,langgraph]
pip install mas-identity[fastapi]
pip install mas-memory[langgraph]
pip install mas-learning          # no framework adapters in v0.1
```

### Adapter contract

All adapters follow two rules:

1. **Never raise** on a missing dependency — raise `ImportError` with a helpful
   install message instead of crashing at import time. Each adapter file starts
   with a `try/except ImportError` guard.
2. **Pass through on error** — middleware and callbacks must not swallow
   exceptions from the underlying handler. They wrap, enrich, and re-raise.

---

## 9. Extraction Sequence and Prerequisites

The five libraries are built or extracted in this order. Each step has a hard
prerequisite that must be met before extraction or initial build begins.

### Step 1 — mas-guard (first)

**Prerequisite:** Phase 1 pipeline complete and running end-to-end in news-mas.

**Why first:** `mas-guard` has no dependency on `mas-observability` or
`mas-identity` at its core. The only cross-library call is
`log_pipeline_error(logger, ...)` and `record_span_error(span, ...)` — both of
which will be refactored to accept a duck-typed logger and span (no import from
`mas-observability` required). `DataQualityValidator` and `PipelineError` are
the most reusable pieces and carry no identity or telemetry concerns.

**Extraction steps:**
1. Create `packages/mas-guard/` with its own `pyproject.toml`
2. Copy `security.py`, `pii_scrubber.py`, `data_quality.py`, `pipeline_errors.py`,
   `error_codes.py` into `mas_guard/`
3. Remove all imports from `src.common.*` — the package is self-contained
4. Remove the news-mas-specific `schemas.py` import from `data_quality.py`
   (the validator currently imports `RawArticle`; parameterise with `Any`)
5. Add `packages/mas-guard` as an editable dependency in news-mas:
   `pip install -e ./packages/mas-guard`
6. Update `src/common` imports to `from mas_guard import ...`
7. Run `pytest tests/` — must be green before proceeding

### Step 2 — mas-observability (second)

**Prerequisite:** `mas-guard` is extracted and news-mas tests are green.

**Why second:** `observability.py` currently imports nothing from `security.py`
or the auth layer, making it straightforward to lift. LangSmith configuration
(`configure_langsmith`, `make_run_config`) travels with observability because
it is a tracing concern, not a pipeline logic concern. `prompt_loader.py`'s
`make_run_config` specifically is an observability helper — the prompt YAML
loading stays in news-mas (domain-specific versioning scheme).

**Extraction steps:**
1. Create `packages/mas-observability/` with its own `pyproject.toml`
2. Copy `observability.py` into `mas_observability/`
3. Copy only `make_run_config()` from `prompt_loader.py` — leave prompt loading
   logic in news-mas
4. Extend the blocked-key set with `register_blocked_keys()` for extensibility
5. Add editable dependency in news-mas; update imports; run tests

### Step 3 — mas-memory (parallel with mas-observability or after)

**Prerequisite:** Phase 1 pipeline complete and producing `FeedbackRepository`
records. `mas-memory` is a new library, not an extraction — it builds on the
`FernetStorage` interface pattern already established in `src/common/storage.py`.

**Why at this point:** Memory is most valuable once the pipeline has run enough
times to accumulate signals, but the library itself does not depend on
`mas-guard` or `mas-observability` being extracted first. It can be built
alongside Step 2. The `EncryptedJsonAdapter` is essentially a thin wrapper
around `FernetStorage`, so the core implementation is small.

**Build steps:**
1. Create `packages/mas-memory/` with its own `pyproject.toml`
2. Implement `TopicMemory`, `TopicMemoryEntry`, `FeedbackProcessor`, and
   `FeedbackSignal` from scratch (not copied from existing code)
3. Implement `EncryptedJsonAdapter` wrapping `FernetStorage`
4. Stub `RedisAdapter` and `PostgresAdapter` with `NotImplementedError` and
   implementation guides (same pattern as workload identity providers)
5. Wire into news-mas: `TopicMemoryRepository` added to `src/common/storage.py`
   repositories, heat scorer receives `TopicMemoryEntry` as optional input
6. Run tests

### Step 4 — mas-identity (fourth)

**Prerequisite:** `mas-observability` extracted AND Phase 2 pipeline is
complete. `mas-identity` is the most infrastructure-coupled library — it wraps
`MAS_SECRET_KEY` today and will wrap Entra ID after DPoP. Extracting it while
the auth layer is still in flux risks stabilising the wrong API surface.

**Why last among the chassis libraries:** `agent_bootstrap.py` depends on
`httpx` and the registry — it should be the last thing to stabilise. Also,
`bootstrap_agent()` currently imports from `src.registry.models` (news-mas
internal). Before extraction, `AgentConfig` must be moved to a shared location
(either into `mas-identity` itself or into a separate `mas-registry-client`
package — to be decided at extraction time).

**Extraction steps:**
1. Resolve `AgentConfig` location — move to `mas-identity` if no separate
   registry client is needed
2. Create `packages/mas-identity/` with `pyproject.toml`
3. Copy `auth/token_service.py` and `auth/workload_identity.py` into `mas_identity/`
4. Copy `agent_bootstrap.py` into `mas_identity/`
5. Preserve all `[DPOP-TODO]` comments verbatim — they are load-bearing markers
   for the 1.0.0 migration (see §4.1)
6. Add editable dependency in news-mas; update imports; run tests

### Step 5 — mas-learning (last, post-MVP)

**Prerequisite:** `mas-memory` is running in production and has accumulated
500+ feedback signals (see §7). DSPy integration requires `dspy-ai` as a
dependency — evaluate whether the added dependency weight is justified by the
signal volume available.

**Build steps:**
1. Create `packages/mas-learning/` — do not create it before the prerequisite
   is met; an empty package encourages premature adoption
2. Implement `FeedbackWeightedScoring` against the `mas-memory` interface
3. Implement DSPy prompt optimiser with LangSmith dataset as input
4. Implement `EvalHook` that consumes LangSmith runs
5. Wire weight update OTel spans before any production use

---

## 10. Versioning Strategy

All five packages follow the same versioning contract:

### 0.x — Reference implementation phase

All packages start at `0.1.0` when first built or extracted. The `0.x` range
signals that the API surface may change between minor versions. The `CHANGELOG.md`
for each package must document every breaking change with a migration note.

Semver rules in `0.x`:
- Patch (`0.1.x`): bug fixes and non-breaking additions
- Minor (`0.x.0`): breaking changes to the public API surface
- No 0.x release goes to PyPI without news-mas as a working consumer

### 1.0.0 — Stable API

- `mas-guard 1.0.0`: public API frozen; adapter contracts stable; full
  coverage of OWASP LLM Top 10 injection patterns documented in the README
- `mas-observability 1.0.0`: blocked-key registry stable; `register_blocked_keys`
  documented as the extension point; Sentinel integration guide complete
- `mas-memory 1.0.0`: `EncryptedJsonAdapter` stable; at least one of
  `RedisAdapter` or `PostgresAdapter` production-tested
- `mas-identity 1.0.0`: **gated on DPoP** (see §4.1); HS256 path removed from
  the default code path (still available as `LocalDevIdentityProvider` for tests)
- `mas-learning 1.0.0`: `FeedbackWeightedScoring` stable; weight versioning and
  rollback documented; DSPy integration validated on a real news-mas dataset

### Rationale for keeping mas-identity below 1.0.0 until DPoP

Publishing `mas-identity 1.0.0` before DPoP would imply the shared-secret auth
is a supported production API. Healthcare teams adopting the library would build
on it, and migrating them to DPoP later would require a 2.0.0 breaking change.
It is cheaper to hold at `0.x` and ship 1.0.0 with the correct production API
from the start.

### Distribution

During the reference implementation phase:
- Packages are editable local installs (`pip install -e ./packages/mas-*`)
- Not published to PyPI
- Versioned in the news-mas mono-repo under `packages/`

After stabilisation:
- Published to a private PyPI-compatible registry (Azure Artifacts or similar)
  or to PyPI if the packages are deemed safe to open-source
- news-mas pins exact versions; other consumers pin to compatible ranges

---

## 11. How news-mas Will Depend on Them

### During extraction (editable installs)

```toml
# news-mas pyproject.toml — during extraction phase
[tool.uv.sources]
mas-guard         = { path = "packages/mas-guard", editable = true }
mas-observability = { path = "packages/mas-observability", editable = true }
mas-identity      = { path = "packages/mas-identity", editable = true }
mas-memory        = { path = "packages/mas-memory", editable = true }
# mas-learning not added until 500+ feedback signals accumulated

[project.dependencies]
mas-guard           = ">=0.1.0"
mas-observability   = ">=0.1.0"
mas-identity        = ">=0.1.0"
mas-memory          = ">=0.1.0"
```

### After publication (pinned versions)

```toml
# news-mas pyproject.toml — after packages are published
[project.dependencies]
mas-guard           = "==0.3.1"
mas-observability   = "==0.2.0"
mas-identity        = "==0.4.2"
mas-memory          = "==0.2.0"
# mas-learning added here once the feedback volume prerequisite is met
```

news-mas pins exact versions (not compatible ranges) because it is the
reference implementation — it should be the first to adopt new versions, not
pulled along by loose constraints.

### Import surface in news-mas after extraction

All existing `from src.common.X import Y` calls become package imports.
The `src/common/` directory is removed once all imports are migrated.
`src/auth/` is removed once `mas-identity` extraction is complete.

Example migration:

```python
# Before extraction
from src.common.security import detect_injection, is_run_active
from src.common.pii_scrubber import scrub_text
from src.common.pipeline_errors import PipelineError, ErrorSeverity
from src.common.observability import get_logger, setup_telemetry
from src.auth.token_service import validate_token

# After extraction
from mas_guard import detect_injection, is_run_active, scrub_text
from mas_guard import PipelineError, ErrorSeverity
from mas_observability import get_logger, setup_telemetry
from mas_identity import validate_token
```

The only code that stays in `src/common/` after full extraction is news-mas
domain logic: `schemas.py` (pipeline Pydantic models), `prompt_loader.py`
(prompt YAML loading and versioning), `storage.py` (FernetStorage), and
`agent_registry.py` (in-memory registry). These are not general-purpose.
`mas-memory` builds on `storage.py` but does not absorb it — `FernetStorage`
stays in news-mas; `EncryptedJsonAdapter` wraps it without owning it.

---

## 12. Target User for Each Library

### mas-guard

**Primary target:** Python engineers building LLM pipelines that process
external or user-supplied content in regulated industries (healthcare, finance,
legal). The engineer may not be a security specialist — `mas-guard` gives them
a tested, opinionated safety layer they can drop in without needing deep
knowledge of OWASP LLM Top 10 or HIPAA Safe Harbor de-identification rules.

**Secondary target:** Healthcare data engineers adopting LangChain or LangGraph
for the first time. The `guarded_node` decorator and `ScrubMiddleware` are
designed to slot in with minimal friction — they should not require reading
the Presidio documentation to use.

**What it is not for:** General-purpose input validation (use Pydantic) or
network-level security (use an API gateway or WAF). `mas-guard` operates at the
application layer, after the request has been authenticated.

### mas-observability

**Primary target:** Engineers who want structured, PHI-safe logging and full
OTel + LangSmith observability wired together in one `setup_observability()`
call, without assembling the three stacks separately. They are building
FastAPI services that call LLMs and need trace correlation from the HTTP request
through to the LangSmith run.

**Secondary target:** Platform teams standardising observability across multiple
AI services. `register_blocked_keys()` and the consistent `_SafeJSONFormatter`
give them a single control point for what never appears in logs.

**What it is not for:** General application logging (use structlog or loguru) or
general OTel setup without the PHI-safety layer.

### mas-identity

**Primary target:** Engineers building multi-agent pipelines where agents need
to authorise each other with context-aware tokens, not raw credentials.
Specifically, engineers who need to carry task context (data sensitivity,
oversight requirements, delegation history) through an agent call chain so that
any agent can answer "what was I authorised to do, by whom, on whose behalf?"
from its token alone.

**Secondary target (post-1.0.0):** Security and compliance engineers at
healthcare organisations who need to demonstrate that every agent action in a
pipeline run can be traced back to an original authorization event. The AAP
token's seven claim sections and the mint/exchange audit trail are designed for
this use case.

**What it is not for:** General OAuth 2.0 client-credential flows (use `msal`
or `httpx-auth`), or fine-grained resource-level RBAC (use OPA or a policy
engine). `mas-identity` handles the agent-to-agent authorization layer, not
the user-to-system access layer.

### mas-memory

**Primary target:** Any team running an agent system on a recurring schedule
(daily, hourly, event-triggered) who wants the system to carry knowledge between
runs — which topics have been consistently productive, which sources have been
flagged as unreliable, which query patterns have generated the best results —
without building a custom data layer or coupling the pipeline to a specific
storage backend.

`mas-memory` is deliberately framework-agnostic. It should work equally well
whether the orchestrator is LangGraph, CrewAI, AutoGen, or plain Python. The
`MemoryStorageAdapter` protocol is the only integration point.

**What it is not for:** In-context memory within a single run (use LangGraph's
state or a LangChain memory buffer). `mas-memory` is cross-run persistence, not
within-run working memory.

### mas-learning

**Primary target:** Teams that have been running `mas-memory` for long enough
to have 500+ feedback signals and are ready to move beyond manual prompt and
threshold tuning. The DSPy integration and contextual bandits implementation
are for teams who understand that they are doing **weighted preference learning**,
not full RL — and who want an auditable, reversible path to improvement rather
than a black box.

**What it is not for:** Teams that have not yet accumulated sufficient feedback
data (enable `mas-memory` first and wait), teams that need model fine-tuning
(this requires training infrastructure outside the scope of this library), or
teams looking for a managed "auto-improve" feature (every weight change is
observable and must be reviewed before being considered correct).

---

## 13. news-mas Backlog: Memory and Learning

This section tracks the news-mas-specific work items that will be built against
`mas-memory` and eventually `mas-learning`. These items are sequenced after the
Phase 1 pipeline is complete and the feedback UI exists (see ARCHITECTURE.md §12).

### TopicMemoryRepository

**Build after:** Phase 1 pipeline complete.

**What:** Add `TopicMemoryRepository` to `src/common/storage.py` using the
existing `FernetStorage` backend (same pattern as `UserRepository`,
`RunRepository`, etc.). Wire heat scorer to receive a `TopicMemoryEntry` as
optional input so historical context can influence scoring.

**Design:**
- `TopicMemoryRepository.record_run(topic_id, run_id, heat_score, approved)`
  called at the end of every successful Phase 1 run
- `HeatScorerInput.topic_memory: dict[str, TopicMemoryEntry] | None = None`
  — optional; heat scorer logs a note when it has memory context but does not
  yet adjust scores (that is `mas-learning`'s job)
- Cross-run performance tracked per topic: avg heat score, approval rate,
  run count, last run timestamp

**Why this ordering:** Memory should accumulate passively for several weeks
before any learning update is applied. Starting to record early means the
dataset will be ready when `mas-learning` is.

### FeedbackProcessor

**Build after:** Feedback UI is in the React frontend (deferred — see
ARCHITECTURE.md §12).

**What:** A processor that reads `FeedbackRepository` records and translates
them into preference weight adjustments stored in `TopicMemoryRepository`.

**Design:**
- Learning rate `0.05` — single signal moves weight by ±5% (see §7 for
  rationale)
- Weight floor `0.1`, ceiling `2.0` — no topic is fully suppressed or
  unreasonably amplified by feedback alone
- Weights persisted in `data/topic_memory/` alongside other run state;
  each update increments a `weight_version` integer for rollback capability
- Every weight change logged as a structured OTel event with old weight,
  new weight, signal count, topic ID — no guessing what changed or why
- `missed_topic` signals are **flagged for manual review, not auto-applied**.
  These are the richest signals in the system — a user noting that an important
  story was absent from the digest contains more information than a thumbs
  up/down and deserves a human decision about whether to adjust the topic list,
  the search query, or the heat scorer configuration. They are queued in a
  separate `data/review_queue/` collection and surfaced in the admin UI.

**Why separate from TopicMemory:** `FeedbackProcessor` is the update mechanism;
`TopicMemoryRepository` is the state store. Keeping them separate means the
memory accumulates correctly even before the feedback UI exists, and the
processor can be tested against historical memory records independently.
