variable "env" {
  type        = string
  description = "Environment name used as a resource name suffix."
}

variable "location" {
  type        = string
  description = "Azure region."
}

variable "tenant_id" {
  type        = string
  description = "Azure Entra ID tenant ID injected as ENTRA_TENANT_ID env var."
}

# ── Agent client IDs (passed as env vars to the gateway container) ────────────

variable "gateway_client_id" {
  type = string
}

variable "search_worker_client_id" {
  type = string
}

variable "heat_scorer_client_id" {
  type = string
}

variable "filter_agent_client_id" {
  type = string
}

variable "selector_client_id" {
  type = string
}

variable "phase1_judge_client_id" {
  type = string
}

variable "summarizer_client_id" {
  type = string
}

variable "reviewer_client_id" {
  type = string
}

variable "relevance_gate_client_id" {
  type = string
}
