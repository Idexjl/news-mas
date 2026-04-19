# SIEM Integration Guide — Azure Sentinel

This document describes the production path for connecting news-mas telemetry
to Azure Sentinel. It covers what the local stack gives you, where it falls
short, how Sentinel fills the gap, and concrete configuration steps.

---

## What Grafana + Loki + Prometheus Gives You Locally

The local observability stack provides:

| Signal | Tool | What you can see |
|---|---|---|
| Distributed traces | Jaeger | Span trees across agents, latency per hop, error propagation |
| Metrics | Prometheus + Grafana | Tavily call rates, injection detection counts, response times |
| Logs | Loki + Grafana | Structured JSON events from every agent, correlated by run_id |

This is sufficient for **development debugging** and **operational monitoring**
of expected behaviour. You can answer: "Why was this run slow?" or "How many
injections were detected this week?"

---

## The Gap: Behavioural Analytics and Unknown Threat Detection

Local tooling answers operational questions. It does not answer:

- **Is this agent behaving unusually compared to its baseline?** Prometheus
  can alert when a counter crosses a threshold, but it has no memory of what
  "normal" looks like per entity over weeks of traffic.
- **Are multiple low-signal anomalies in different systems part of the same
  attack?** Grafana can show you that injection detections spiked and Tavily
  call volume rose in the same hour, but it cannot fuse those signals into a
  single incident with confidence scoring.
- **Did a compromise touch PHI?** HIPAA requires an audit trail that
  demonstrates access to sensitive data was authorised and monitored. Loki
  stores logs but does not produce the compliance reports HIPAA auditors expect.
- **Is a new kind of attack occurring?** Prometheus alerts only on patterns
  you pre-defined. Sentinel's ML-based anomaly detection (Fusion + UEBA)
  catches patterns you did not know to look for.

---

## Azure Sentinel as the Production SIEM

### Why Sentinel Specifically

**Entra ID integration.** Once DPoP is active (see DPOP_IMPLEMENTATION_GUIDE.md),
each agent has a Managed Identity. Sentinel natively ingests Entra ID sign-in
logs and risk events. A DPoP proof verification failure in the application layer
can be correlated with the Entra risk signal for the same identity in the same
time window — a correlation no local tool can make.

**HIPAA compliance reporting.** Sentinel's built-in compliance workbooks
(HIPAA/HITRUST) consume the structured log events this system emits. Because
`_SafeJSONFormatter` never logs PHI content — only counts, IDs, and metadata —
the log stream is safe to route to Sentinel without additional scrubbing. The
compliance reports show: which agents processed PHI-tagged runs, what data
sensitivity level was declared in the AAP token, and whether any anomalies were
detected during those runs.

**Fusion detection.** Sentinel's Fusion engine correlates signals across
multiple data connectors. An injection detection (application log), a Tavily
call volume spike (metric), and an unusual agent bootstrap time (trace) arriving
within minutes of each other would each be low-severity alone. Fusion combines
them into a high-confidence "multi-stage attack on AI pipeline" incident.

**UEBA for agent behavioral baselines.** UEBA (User and Entity Behavior
Analytics) builds a statistical model of each entity's normal behaviour. In this
system, the entities are agents (identified by `service.name`). UEBA learns:
typical Tavily call frequency per run, normal `egress.response_time_ms`
distribution, expected `aap_delegation.chain` patterns. Deviations from baseline
surface automatically — no alert rule to write.

---

## How AAP Token Claims Enrich Sentinel Events

Every security event in this system carries an AAP token context. The token's
seven claim sections map to Sentinel fields:

| AAP claim | Sentinel use |
|---|---|
| `aap_agent.agent_name` | Entity identifier for UEBA baseline |
| `aap_task.task_id` | Correlation key — join all events for one task |
| `aap_task.data_sensitivity` | Alert threshold selector (stricter rules for PHI) |
| `aap_delegation.chain` | Provenance — which orchestrator initiated the chain |
| `aap_context.run_id` | Top-level correlation across all agents in a pipeline run |
| `aap_audit.token_id` | Token lifecycle tracking — mint → use → expiry |
| `aap_oversight.oversight_required` | Escalation flag — human review required? |

When `mint_token()` emits a structured log event, `token_id` becomes the
primary correlation key. Every downstream event (injection detection, egress
anomaly, auth failure) that occurs within the same `run_id` can be joined back
to that mint event in Sentinel's Log Analytics workspace.

---

## OTel Connector Configuration

### Step 1 — Add the Azure Monitor exporter to the collector

In `configs/otel-collector-config.yaml`, add under `exporters`:

```yaml
azuremonitor:
  connection_string: "${AZURE_MONITOR_CONNECTION_STRING}"
  # Retrieved from Key Vault via the collector container's Managed Identity.
  # Never store this value in config files or environment files.
```

Update the pipelines to dual-write during the transition period:

```yaml
service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, resource, batch]
      exporters: [otlp/jaeger, azuremonitor]   # dual-write

    metrics:
      receivers: [otlp]
      processors: [memory_limiter, resource, batch]
      exporters: [prometheus, azuremonitor]     # dual-write

    logs:
      receivers: [otlp]
      processors: [memory_limiter, resource, batch]
      exporters: [loki, azuremonitor]           # dual-write
```

### Step 2 — Link the workspace to Sentinel

In the Azure portal:
1. Create a Log Analytics workspace in the same resource group as the
   Container Apps environment.
2. Enable Microsoft Sentinel on that workspace.
3. Copy the connection string from the Application Insights resource linked
   to the workspace into Key Vault as `AZURE-MONITOR-CONNECTION-STRING`.
4. Grant the OTel collector container's Managed Identity the
   `Monitoring Metrics Publisher` role on the workspace.

### Step 3 — Enable the Microsoft Sentinel Data Connector for OTel

In Sentinel → Data Connectors, find "OpenTelemetry (Preview)" and connect.
Map the `service.name` resource attribute to the Sentinel entity type `Host`.
Map `aap_audit.token_id` (from log events) to the Sentinel correlation field.

### Step 4 — Import Analytics Rules

Create the following Analytics Rules in Sentinel (KQL):

**Injection Detection:**
```kql
traces
| where name == "egress.http.fetch"
| where tostring(customDimensions["egress.injection_detected"]) == "True"
| summarize Count=count() by bin(timestamp, 5m), tostring(customDimensions["egress.host"])
| where Count > 0
```

**AAP Token Validation Failure Burst:**
```kql
traces
| where name == "aap_token_validation_failure"
| summarize FailureCount=count() by bin(timestamp, 5m), tostring(customDimensions["agent_id"])
| where FailureCount >= 3
```

**Anomalous Egress Host:**
```kql
traces
| where name == "egress.http.fetch"
| where tostring(customDimensions["egress.host"]) !in ("api.tavily.com", "api.anthropic.com")
| project timestamp, egress_host=customDimensions["egress.host"], run_id=customDimensions["run_id"]
```

---

## Incident Response Playbook Examples

### Injection Detection Event

**Trigger:** `EgressInjectionDetected` Prometheus alert → Sentinel incident.

**Playbook steps (Logic App):**
1. Extract `run_id` from the incident alert context.
2. Query Log Analytics for all spans with that `run_id`.
3. Identify `egress.host` and `pattern_count` from the injection span (never
   log matched content — HIPAA constraint).
4. Call FernetStorage API to set run status → `"killed"` (activates circuit
   breaker — see `is_run_active()` in `src/common/security.py`).
5. Create Sentinel Incident with severity `High`, assign to SOC queue.
6. Post summary to Teams SOC channel: run_id, agent, egress host, pattern count.

### AAP Token Validation Failure

**Trigger:** ≥3 `aap_token_validation_failure` log events in 5 minutes from
the same `agent_id`.

**Playbook steps:**
1. Query Entra ID sign-in logs for the agent's Managed Identity in the same
   window (cross-signal correlation — available once DPoP is active).
2. If Entra risk score > 0: escalate to `Critical`, page on-call.
3. If Entra risk score == 0: create `Medium` incident, request manual review.
4. Capture `token_id` values from failed validations for forensic audit trail.

### Anomalous Tavily Call Volume

**Trigger:** `TavilyHighCallVolume` Prometheus alert (>50 calls/hour).

**Playbook steps:**
1. Query the scheduler service logs for the past hour to determine whether a
   legitimate batch run is in progress.
2. If no scheduled run: pause the APScheduler job (call the scheduler's
   `/pause` endpoint), create `Medium` Sentinel incident.
3. Post to Teams: call count, time window, run_ids involved.

### Agent Registry Tampering Attempt

**Trigger:** `PUT /agents/{agent_id}/config` 403 response (capability check
failed) or request from an unexpected source subnet.

**Playbook steps:**
1. Extract source IP and `agent_id` from the 403 log event.
2. Cross-reference with UEBA: is this agent's source IP in its baseline?
3. If source IP is from the compute subnet (workers should never PUT): escalate
   to `High` — this is exactly the lateral movement the network segmentation
   is designed to prevent.
4. Capture the full request context for the audit trail (source IP, timestamp,
   attempted agent_id, auth header type — never the header value).

---

## Compliance: HIPAA Audit Trail Requirements

HIPAA's Security Rule (45 CFR §164.312(b)) requires audit controls that record
and examine activity in systems that contain PHI. This architecture satisfies
the requirement as follows:

| Requirement | How this architecture satisfies it |
|---|---|
| Activity in PHI systems is logged | Every agent emits structured JSON logs via `_SafeJSONFormatter`. `data_sensitivity` from the AAP token identifies PHI-tagged runs. |
| Logs are tamper-evident | Sentinel Log Analytics workspace has immutable log retention. Local Loki is not tamper-evident — production must route to Sentinel. |
| Access is authorised | AAP token's `aap_task.data_sensitivity` field documents declared sensitivity. `aap_delegation.chain` records which agents touched the run. |
| Anomalies are detected and investigated | Sentinel Analytics Rules + UEBA provide automated detection. Playbooks provide documented investigation procedures. |
| Audit trail is available for 6 years | Sentinel workspace retention can be set to 7 years to exceed the HIPAA minimum. |

The single most important HIPAA constraint in this codebase: **never log PHI
content**. `_SafeJSONFormatter` enforces this at the application layer. Routing
logs to Sentinel does not change this constraint — it only adds a compliant,
tamper-evident destination for the already-scrubbed log stream.

---

## Finding All Integration Points

```bash
grep -r "SIEM-TODO" src/
```

Expected output locations:
- `src/common/observability.py` — logger setup, tracer setup
- `src/common/security.py` — detect_injection, validate_shared_secret
- `src/auth/token_service.py` — mint_token, validate_token
- `src/agents/search_worker/agent.py` — Tavily egress span
- `src/registry/registry_server.py` — agent bootstrap endpoint
- `docker-compose.yml` — loki service, otel-collector service
