---
name: Reviewer agent implementation
description: Reviewer agent fully implemented — model, schema extensions, prompt, OTel, LangSmith, error codes
type: project
---

Reviewer agent (`src/agents/reviewer/agent.py`) is implemented.

**Why:** Phase 2 quality gate — evaluates citation accuracy, constraint compliance, and content substance before summaries reach the user.

**How to apply:** When working on Phase 2 orchestrator (`graph.py`), reviewer_node can now be wired to call `run_agent(ReviewerInput(...))` directly. The routing key is `ReviewerOutput.approved` (bool).

Key design facts:
- Model: `claude-sonnet-4-6` hardcoded; `MODEL_OVERRIDE` never applies (same as summarizer)
- Prompt: `prompts/reviewer/v1.0.yaml`
- Capability: `review.content`
- Port: 8007
- Three verdict dimensions: citation_mismatch / constraint_violated / thin_content
- PHI auto-fail runs before LLM call; no-sources runs in DEGRADED (citation check skipped)
- Reviewer unavailability → DEGRADED pass with warning `reviewer_unavailable`
- Schema: `ReviewIssue` added to `schemas.py`; `ReviewerInput` extended with `citations`, `sources`, `constraints`, `topic_text`, `retry_count`, `aap_token`; `ReviewerOutput` extended with `verdict`, `confidence`, `endorsement_note`, `issues`
- Error code `REVIEWER_UNAVAILABLE` added to `error_codes.py`
- Integration tests: `tests/integration/test_reviewer_live.py`
