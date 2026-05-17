variable "env" {
  type    = string
  default = "dev"
}

variable "azure_location" {
  type    = string
  default = "eastus"
}

variable "aws_region" {
  type    = string
  default = "us-west-2"
}

variable "tenant_id" {
  type = string
}

variable "subscription_id" {
  type = string
}

# ── Agent app registration client IDs ─────────────────────────────────────────

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

# ── AWS / EC2 ─────────────────────────────────────────────────────────────────

variable "ami_id" {
  type    = string
  default = ""
}

variable "subnet_ids" {
  type    = list(string)
  default = []
}

variable "container_apps_cidr_blocks" {
  type    = list(string)
  default = ["0.0.0.0/0"]
}
