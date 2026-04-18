# DPoP + Microsoft Entra ID Implementation Guide

## 1. Overview

This guide covers replacing the development shared-secret (`X-MAS-Secret`) with
Demonstrating Proof-of-Possession (DPoP, RFC 9449) tokens issued by Microsoft
Entra ID for all agent-to-agent (A2A) calls in news-mas.

Every inter-agent call in production carries two headers:

```
Authorization: DPoP <access_token>
DPoP: <dpop_proof_jwt>
```

Key implementation files:

| File | Purpose |
|------|---------|
| `src/common/auth/dpop.py` | Proof generation + verification (RFC 9449) |
| `src/common/auth/token_validator.py` | Entra token validation + cnf/jkt binding |
| `src/common/auth/entra.py` | Token acquisition (client credentials + OBO) |
| `src/common/auth/agent_identity.py` | Per-agent Entra registration config |
| `src/common/auth/middleware.py` | FastAPI DPoP middleware (replaces SecurityHeaderMiddleware) |

Search for `[DPOP-TODO]` across the repo to locate every implementation anchor.

---

## 2. DPoP Primer (RFC 9449)

### 2.1 What DPoP Does

A standard Bearer token is reusable by anyone who holds it. DPoP binds a token
to a specific asymmetric key pair at issuance time. Even if an access token is
intercepted, it cannot be replayed without the corresponding private key.

### 2.2 Proof Structure

A DPoP proof is a short-lived, request-scoped JWT signed with the agent's private key.

**JOSE header:**
```json
{
  "typ": "dpop+jwt",
  "alg": "ES256",
  "jwk": { "kty": "EC", "crv": "P-256", "x": "...", "y": "..." }
}
```

**Payload:**
```json
{
  "htm": "POST",
  "htu": "https://agents.internal/heat-scorer/run",
  "iat": 1713441600,
  "jti": "550e8400-e29b-41d4-a716-446655440000",
  "ath": "base64url(sha256(access_token))"
}
```

- `htm` / `htu` — bind the proof to a specific method + URI  
- `iat` — issued-at; Entra validates within a ~60-second window  
- `jti` — unique per proof; stored server-side to prevent replay  
- `ath` — binds the proof to a specific access token (required when an AT is present)

### 2.3 Proof Lifecycle

```
Agent startup
  └── generate_dpop_keypair()          # EC P-256 key pair, once per instance

Token issuance (client_credentials or OBO)
  └── generate_dpop_proof(method="POST", uri=TOKEN_ENDPOINT, private_key=k)
  └── POST /token  body: token_type=DPoP  header: DPoP: <proof>
  └── Entra returns access_token with token_type="DPoP" and cnf/jkt claim
  └── Cache the access_token (expires ~1h)

Every outbound agent call
  └── generate_dpop_proof(method=..., uri=..., private_key=k, access_token=at)
  └── POST /run  Authorization: DPoP <at>  DPoP: <fresh_proof>
  ← Never reuse a proof — generate one per request
```

**Key rule:** cache the access token; generate proofs on demand. A cached proof
is a security vulnerability.

---

## 3. Microsoft Entra ID Integration

Reference: https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-auth-code-flow

### 3.1 App Registrations for Agent Client Credentials

Each agent requires its own Entra ID app registration. Shared credentials
break per-agent token revocation, audit trails, and conditional access.

| Agent | Suggested Registration Name | Exposes API |
|-------|-----------------------------|-------------|
| search_worker | `news-mas-search-worker` | No (calls external only) |
| heat_scorer | `news-mas-heat-scorer` | `api://news-mas-heat-scorer` |
| filter_agent | `news-mas-filter-agent` | `api://news-mas-filter-agent` |
| selector | `news-mas-selector` | `api://news-mas-selector` |
| phase1_judge | `news-mas-phase1-judge` | `api://news-mas-phase1-judge` |
| summarizer | `news-mas-summarizer` | `api://news-mas-summarizer` |
| reviewer | `news-mas-reviewer` | `api://news-mas-reviewer` |
| relevance_gate | `news-mas-relevance-gate` | `api://news-mas-relevance-gate` |

**Steps per registration (Azure Portal → App registrations → New registration):**

1. Set App ID URI: `api://{client_id}` under "Expose an API"
2. Add a scope named `Invoke` (or `run`), consent: Admins only for A2A flows
3. Under "App roles", add role `AgentCaller` (member type: Application)
4. In the **calling** agent's registration → "API permissions" → add the
   target agent's `api://{target_client_id}/Invoke` application permission
5. Generate a client secret (dev) or upload a certificate (production)
6. Run admin consent once per environment:

```bash
az ad app permission admin-consent --id <calling_agent_client_id>
```

**Why one registration per agent?**
- Enables per-agent token revocation without touching other agents
- Preserves audit trail (`appid` claim in tokens identifies exact caller)
- Supports per-agent conditional access policies (e.g. require compliant device,
  restrict to specific IP ranges for prod orchestrators)
- Required for the OBO delegation chain to carry the correct actor identity

### 3.2 Managed Identity and DPoP — Why Managed Identity Alone Is Not Sufficient

Azure Managed Identity (System or User Assigned) lets an Azure-hosted workload
authenticate to Entra-protected resources without storing credentials. However,
it has a hard limitation for this system:

**Managed Identity cannot participate in DPoP.**

Tokens acquired via the Azure Instance Metadata Service (IMDS) endpoint cannot
be bound to a DPoP key pair. The IMDS does not accept a `token_type=DPoP`
parameter and Entra will not embed a `cnf/jkt` claim in IMDS-issued tokens.

| Scenario | Managed Identity | DPoP + Client Credentials |
|----------|:---------------:|:-------------------------:|
| Agent → Azure Key Vault / Storage / Service Bus | ✅ Ideal | Not needed |
| Agent → Agent (A2A call within news-mas) | ❌ Not DPoP-bound | ✅ Required |
| Cross-cloud agent (AWS ECS, GCP Cloud Run) | ❌ MI unavailable | ✅ Works everywhere |
| Full AAP delegation chain with cnf/jkt | ❌ No cnf claim | ✅ Preserved end-to-end |
| Audit trail showing which agent called which | ⚠️ Shared MI identity | ✅ Per-agent appid claim |

**Recommended hybrid pattern:**

- Use Managed Identity for Azure resource access (Key Vault secret reads,
  Blob Storage, Service Bus) — it's the right tool for that job
- Store the agent's DPoP private key *in* Key Vault and *load* it via MI at startup
- Use DPoP + client credentials for all inter-agent HTTP calls

This way MI handles secrets management; DPoP handles A2A authentication.

### 3.3 RFC 8693 Token Exchange vs Entra On-Behalf-Of

When a user-initiated request flows through multiple agents, each hop must
propagate the user's identity for audit logging and policy enforcement.

**RFC 8693 (standard token exchange):**
```
POST /token
grant_type=urn:ietf:params:oauth:grant-type:token-exchange
subject_token={upstream_token}
subject_token_type=urn:ietf:params:oauth:token-type:jwt
requested_token_type=urn:ietf:params:oauth:token-type:jwt
actor_token={agent_assertion}        # optional in RFC 8693
scope={target_scope}
```

**Entra On-Behalf-Of (OBO):**
```
POST https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token
grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer   ← different from RFC 8693
assertion={upstream_access_token}
requested_token_use=on_behalf_of
client_id={this_agent_client_id}
client_secret={this_agent_client_secret}
scope={target_scope}
token_type=DPoP                                           ← DPoP binding
DPoP: {dpop_proof_jwt}                                    ← header
```

**Key divergences:**

| Aspect | RFC 8693 | Entra OBO |
|--------|----------|-----------|
| `grant_type` | `token-exchange` | `jwt-bearer` |
| `requested_token_type` | required | omit (Entra infers access_token) |
| `subject_token_type` | required | omit |
| `actor_token` | optional | not supported |
| DPoP binding | implementation-defined | `token_type=DPoP` + `DPoP:` header |
| Result token bound to | actor's key (if provided) | **this agent's** DPoP key |

**When to use OBO vs client credentials in news-mas:**

| Scenario | Flow |
|----------|------|
| Scheduled batch digest run (no user) | client_credentials |
| User-triggered run via API; need user claim in downstream audit logs | OBO |
| Orchestrator → all Phase 1 agents | client_credentials |
| Phase 2 summarizer needing user context | OBO (if user-triggered) |

The Phase 1 orchestrator will almost always use client credentials (scheduled
job). Phase 2 may use OBO if the run originates from a user API call.

### 3.4 Multi-Cloud Considerations

news-mas agents may run on AWS ECS, GCP Cloud Run, or on-premises Kubernetes
alongside Azure-hosted agents. In non-Azure environments:

- Managed Identity is unavailable (no IMDS endpoint)
- Azure Arc can extend MI to non-Azure hosts but adds significant operational
  overhead; not recommended unless Arc is already part of the platform
- **DPoP + client credentials is the consistent pattern that works everywhere**

**Cross-cloud deployment checklist:**

- [ ] Inject `ENTRA_TENANT_ID`, `AGENT_{NAME}_CLIENT_ID`, and
      `AGENT_{NAME}_CLIENT_SECRET` as secrets via the platform's secret manager
      (AWS Secrets Manager, GCP Secret Manager, or Kubernetes Secrets)
- [ ] Mount the DPoP private key (PEM) as a container secret; set
      `AGENT_{NAME}_DPOP_KEY_PATH` to the mount path
- [ ] Ensure outbound HTTPS to `login.microsoftonline.com` (port 443) is
      permitted from all agent host networks
- [ ] Ensure outbound HTTPS to the Entra JWKS endpoint is permitted:
      `https://login.microsoftonline.com/{tenant}/discovery/v2.0/keys`
- [ ] Configure token cache (in-memory `TokenCache` for single-instance;
      Redis for horizontally-scaled deployments)
- [ ] Verify clock sync (NTP) — DPoP proofs are rejected if `iat` drifts
      more than ~60 seconds from the server clock

**Network note:** In air-gapped or highly restricted environments, Entra Private
Endpoints (currently in preview for some regions) can route token traffic through
a private VNet. Evaluate if your security posture requires it.

### 3.5 Entra Scopes and App Registration Config for A2A Flows

**Scope pattern for A2A calls (application permissions, no user context):**
```
api://{target_agent_client_id}/.default
```

The `.default` scope requests all statically declared application permissions.
This is the correct pattern for client_credentials / machine-to-machine flows.

**Example — Phase 1 orchestrator acquiring a token to call heat_scorer:**
```python
token = await get_client_credentials_token(
    tenant_id=os.environ["ENTRA_TENANT_ID"],
    client_id=os.environ["AGENT_PHASE1_CLIENT_ID"],
    client_secret=os.environ["AGENT_PHASE1_CLIENT_SECRET"],
    scope=f"api://{os.environ['AGENT_HEAT_SCORER_CLIENT_ID']}/.default",
    dpop_private_key=phase1_identity.dpop_key,
)
```

**App registration config for the target agent (heat_scorer):**
1. "Expose an API" → App ID URI: `api://{heat_scorer_client_id}`
2. Add scope `Invoke` — application type, admin consent required
3. "Authorised client applications" → pre-authorise the orchestrator's
   `client_id` to suppress per-tenant admin consent prompts

**Optional: App Roles for defence-in-depth**

Define an `AgentCaller` app role on each target agent's registration:
```json
{
  "allowedMemberTypes": ["Application"],
  "displayName": "Agent Caller",
  "id": "<new-guid>",
  "isEnabled": true,
  "value": "AgentCaller"
}
```

Grant the calling agent's service principal that role:
```bash
az ad app permission add \
  --id <calling_client_id> \
  --api <target_client_id> \
  --api-permissions <role_id>=Role

az ad app permission admin-consent --id <calling_client_id>
```

`DPoPAuthMiddleware` can then verify `"AgentCaller" in claims.get("roles", [])`
as an additional layer of authorisation beyond token validity.

### 3.6 The Entra ID Introspection Gap and Compensating Controls

#### No RFC 7662 support

Entra ID does not implement RFC 7662 token introspection. Each resource server
(agent) validates tokens locally by verifying the JWT signature against the Entra
JWKS endpoint and checking standard claims (`iss`, `aud`, `exp`, `nbf`). There is
no runtime callback to Entra to confirm a token is still valid.

Consequence: a credential that has been revoked in Entra — for example, because a
client secret was rotated after a suspected compromise — produces tokens that remain
accepted by every resource server until their `exp` claim passes. For this system's
15-minute agent token lifetime, the worst-case enforcement window is 15 minutes.

This is a known limitation of the self-contained JWT model and applies to all
Entra-protected services, not just news-mas.

#### DPoP as blast radius limiter

DPoP does not close the revocation window, but it substantially limits what an
attacker can do with a stolen token.

| Threat | Without DPoP | With DPoP |
|--------|:-----------:|:---------:|
| Stolen Bearer token replayed from any host | ✅ Full access | — |
| Stolen DPoP token used without private key | — | ❌ `cnf/jkt` mismatch |
| Token replayed at a different URI or method | ✅ Exploitable | ❌ `htm`/`htu` mismatch |
| Proof reused after its 60–120 s window | ✅ Exploitable | ❌ `iat` out of window |
| Proof replayed within window (different `jti`) | ✅ Exploitable | ❌ `jti` replay rejected |

DPoP converts a stolen token from a universal pass into a narrow credential that
requires the matching private key, a fresh proof for each exact endpoint, and
continuous key possession. An attacker who controls the token but not the key gains
nothing. DPoP does not help if both the token and the private key are compromised;
short TTLs are the backstop for that scenario.

#### Short token lifetime strategy

Set `expires_in` to **900 seconds (15 minutes)** on all agent app registrations.

```
Azure Portal → App registrations → {agent} → Token configuration
  → Access token lifetime → 900
```

Or via the Entra token lifetime policy API:

```bash
az ad app update --id <client_id> \
  --set "tokenLifetimePolicies=[{\"definition\":[{\"TokenLifetimePolicy\":{\"Version\":1,\"AccessTokenLifetime\":\"00:15:00\"}}]}]"
```

Why 15 minutes specifically:
- Long enough that normal pipeline runs (typically 3–8 minutes end-to-end) complete
  without mid-run re-authentication
- Short enough that a revoked credential's window is operationally manageable
- Matches Entra's DPoP proof nonce window, simplifying clock-skew reasoning

This is the single most effective passive control: it works regardless of whether
DPoP, CAE, or the circuit breaker is in place.

#### Run-level circuit breaker — primary operational control

The circuit breaker is the **primary active control** for the introspection gap. It
does not wait for token expiry: the orchestrator marks a run as cancelled and agents
refuse work immediately, even if their token remains valid.

**Implementation — `src/common/security.py`:**

```python
from src.common.security import is_run_active

# In every agent's /run handler, before any work:
if not is_run_active(request.run_id, storage=storage, aap_claims=token_claims, logger=logger):
    raise HTTPException(status_code=409, detail="Run has been cancelled")
```

`is_run_active()` checks `FernetStorage` for the run's `status` field. Statuses
`cancelled`, `killed`, and `interrupted` all return `False`. A run that does not
yet exist in storage returns `True` (fail-open, to allow agents to start before the
orchestrator has persisted the initial state).

**Critically:** when the run is inactive but the auth token is still valid,
`is_run_active()` emits a `WARNING` log tagged `security_event: true` with the full
AAP claims dict. This is the forensic signal: a valid token was presented for a
cancelled run, which means either a slow/delayed agent or a replayed request.

**Orchestrator — cancelling a run:**

```python
# Immediate — no Entra interaction needed
storage.save("runs", run_id, {**current_state, "status": "cancelled"})
```

Why this is the primary control:
- **Immediate:** takes effect on the next agent request, not after token expiry
- **No infra:** uses `FernetStorage`, which is already deployed
- **Layered correctly:** the token controls *who may call*; the run state controls
  *whether work should proceed* — these are orthogonal concerns
- **Audit-friendly:** the WARNING log captures the exact AAP claims at the moment
  of the violation, giving forensic detail that token revocation alone cannot provide

#### CAE as partial mitigation (deferred)

Continuous Access Evaluation (RFC 9700 / Entra CAE) allows resource servers to
subscribe to near-real-time revocation notifications from Entra. When a credential
is revoked, participating resource servers are notified within seconds and reject
the token without waiting for `exp`.

CAE would close the introspection gap entirely for scenarios where the Entra
credential itself is revoked (not just the run). It is deferred because:

1. All eight agents must implement the `claims_challenge` round-trip simultaneously
2. `DPoPAuthMiddleware` must handle `401 insufficient_claims` and re-authenticate
3. This is an operational risk during active development

CAE requirements (for Phase 7):
- Add `xms_cc: {values: ["CP1"]}` to the `claims` parameter of every token request
  to signal CAE participation to Entra
- Handle `WWW-Authenticate: Bearer error="insufficient_claims"` in middleware —
  parse the `claims` challenge, re-acquire token with the challenge embedded, retry
- Integration test: force-revoke a credential mid-run; confirm rejection within the
  CAE notification window (target: < 30 seconds)

#### Defence-in-depth summary

| Control | Scope | Latency to enforce | Status |
|---|---|---|---|
| Short token TTL (15 min) | All token abuse | Up to 15 min | Implement now |
| DPoP binding | Replay / theft without key | Per-request (60–120 s proof window) | Phase 1–5 |
| `is_run_active()` circuit breaker | Work-level cancellation | Next agent request | **Implemented** |
| CAE | Credential revocation events | < 30 s | Phase 7 (deferred) |

#### Implementation order

```
Now
 ├── Set 15-min token TTL on all 8 app registrations
 └── Call is_run_active() at the top of every agent /run handler

Phase 1–5  (DPoP rollout — this guide)
 └── DPoP binding reduces blast radius of any leaked token

After Phase 5
 └── Phase 7 — CAE
      ├── Add xms_cc CP1 capability claim to all token requests
      ├── Implement claims_challenge handler in DPoPAuthMiddleware
      └── Red-team: force-revoke credential mid-run; confirm < 30 s enforcement
```

---

## 4. Implementation Checklist

### Phase 0 — Entra ID provisioning (platform / ops team)

- [ ] Create one app registration per agent in the target tenant
- [ ] Configure App ID URIs and `Invoke` scopes for each agent
- [ ] Pre-authorise all agent-to-agent API permissions
- [ ] Grant admin consent for all calling agents
- [ ] Add the `AgentCaller` app role and grant to calling agents (optional)
- [ ] Store `client_id` + secret/cert in the secrets manager for each environment
- [ ] Populate Entra env vars in deployment config (see `.env.example`)

### Phase 1 — DPoP key management (`src/common/auth/dpop.py`)

- [ ] Implement `generate_dpop_keypair()` (EC P-256 / ES256 recommended)
- [ ] Implement `private_key_to_jwk()`
- [ ] Add DPoP key loading to `agent_identity.load_agent_identity()`
      — Key Vault via Managed Identity for Azure-hosted agents
      — Mounted PEM secret for cross-cloud agents (see §3.4)

### Phase 2 — DPoP proof generation (`src/common/auth/dpop.py`)

- [ ] Implement `generate_dpop_proof()` with htm / htu / iat / jti / ath
- [ ] Implement `verify_dpop_proof()` with jti replay store
- [ ] Unit tests: correct proof verifies; wrong method fails; replayed jti fails
- [ ] Confirm Entra's ~60-second iat window matches `max_age_seconds`

### Phase 3 — Token acquisition (`src/common/auth/entra.py`)

- [ ] Implement `get_client_credentials_token()` with DPoP binding
- [ ] Implement `TokenCache.get()` / `set()` / `invalidate()`
- [ ] Implement `get_obo_token()` for user-delegated flows
- [ ] Integration test against a dev Entra tenant

### Phase 4 — Token validation (`src/common/auth/token_validator.py`)

- [ ] Implement `compute_jwk_thumbprint()` (RFC 7638, SHA-256)
- [ ] Implement `fetch_entra_jwks()` with caching + rotation-triggered refresh
- [ ] Implement `validate_entra_token()` — signature, claims, cnf/jkt, DPoP proof
- [ ] Integration test: valid DPoP token passes; tampered `cnf` fails; Bearer rejected

### Phase 5 — Middleware rollout (`src/common/auth/middleware.py`)

- [ ] Complete `DPoPAuthMiddleware.dispatch()` with real validation chain
- [ ] Replace `[DPOP-TODO]` stubs in each agent `main.py`
      — swap `SecurityHeaderMiddleware` for `DPoPAuthMiddleware`
      — pass `tenant_id` and `audience` from `load_agent_identity()`
- [ ] Update both orchestrator graphs to generate DPoP proofs on agent calls
- [ ] Remove (or feature-flag) `SecurityHeaderMiddleware` and `MAS_SECRET_KEY`

### Phase 6 — Hardening and validation

- [ ] Load test jti replay prevention (verify no false negatives under concurrency)
- [ ] Test JWKS key rotation: force-expire cached keys, confirm re-fetch works
- [ ] End-to-end test cross-cloud agent with mounted DPoP key (no Managed Identity)
- [ ] Red-team: confirm Bearer token (without DPoP proof) returns 401
- [ ] Red-team: confirm replayed DPoP proof returns 401
- [ ] Red-team: confirm token from agent A cannot be used by agent B (cnf/jkt mismatch)

### Phase 7 — Continuous Access Evaluation (after Phase 5 complete)

See §3.6 for the full rationale and implementation detail.

- [ ] Set 15-minute access token TTL on all 8 app registrations (Azure Portal or policy API)
- [ ] Add `run_id` status check at the top of every agent `/run` handler (circuit breaker)
- [ ] Add CP1 capability claim (`xms_cc`) to all `get_client_credentials_token()` calls
- [ ] Implement `claims_challenge` handler in `DPoPAuthMiddleware` — parse
      `WWW-Authenticate` header, embed challenge in re-authentication request, retry
- [ ] Integration test: force-revoke an agent credential mid-run; confirm the next
      call to that agent returns 401 within the CAE notification window (target: < 30 s)
- [ ] Update `validate_entra_token()` to check `xms_cc` claim presence in CAE-capable tokens

---

## 5. Quick-Reference: Testing DPoP

Once `dpop.py` is implemented, smoke-test with:

```python
# tests/test_dpop.py
from src.common.auth.dpop import (
    generate_dpop_keypair, generate_dpop_proof, verify_dpop_proof
)

def test_proof_roundtrip():
    private_key, _ = generate_dpop_keypair()
    proof = generate_dpop_proof(
        method="POST",
        uri="http://localhost:8002/run",
        private_key=private_key,
    )
    claims = verify_dpop_proof(
        proof_jwt=proof,
        method="POST",
        uri="http://localhost:8002/run",
    )
    assert claims["htm"] == "POST"

def test_wrong_method_rejected():
    private_key, _ = generate_dpop_keypair()
    proof = generate_dpop_proof(
        method="POST", uri="http://localhost:8002/run", private_key=private_key
    )
    with pytest.raises(ValueError):
        verify_dpop_proof(proof_jwt=proof, method="GET",
                          uri="http://localhost:8002/run")

def test_replay_rejected():
    private_key, _ = generate_dpop_keypair()
    proof = generate_dpop_proof(
        method="POST", uri="http://localhost:8002/run", private_key=private_key
    )
    verify_dpop_proof(proof_jwt=proof, method="POST",
                      uri="http://localhost:8002/run")
    with pytest.raises(ValueError, match="replay"):
        verify_dpop_proof(proof_jwt=proof, method="POST",
                          uri="http://localhost:8002/run")
```

For integration testing against Entra, use a free dev tenant from the
[Microsoft 365 Developer Program](https://developer.microsoft.com/en-us/microsoft-365/dev-program)
to avoid using production credentials in CI pipelines.
