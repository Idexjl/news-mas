variable "env" {
  type        = string
  description = "Environment name (dev, staging, prod). Used as a suffix on all resource names."
}

variable "azure_location" {
  type        = string
  description = "Azure region for all resources."
  default     = "eastus"
}

variable "aws_region" {
  type        = string
  description = "AWS region for Bedrock IAM and Ollama EC2 resources."
  default     = "us-west-2"
}

variable "tenant_id" {
  type        = string
  description = "Azure Entra ID tenant ID."
}

variable "subscription_id" {
  type        = string
  description = "Azure subscription ID."
}

# ── Agent app registration client IDs ─────────────────────────────────────────

variable "gateway_client_id" {
  type        = string
  description = "Entra app registration client ID for news-mas-gateway."
}

variable "search_worker_client_id" {
  type        = string
  description = "Entra app registration client ID for news-mas-search-worker."
}

variable "heat_scorer_client_id" {
  type        = string
  description = "Entra app registration client ID for news-mas-heat-scorer."
}

variable "filter_agent_client_id" {
  type        = string
  description = "Entra app registration client ID for news-mas-filter-agent."
}

variable "selector_client_id" {
  type        = string
  description = "Entra app registration client ID for news-mas-selector."
}

variable "phase1_judge_client_id" {
  type        = string
  description = "Entra app registration client ID for news-mas-phase1-judge."
}

variable "summarizer_client_id" {
  type        = string
  description = "Entra app registration client ID for news-mas-summarizer."
}

variable "reviewer_client_id" {
  type        = string
  description = "Entra app registration client ID for news-mas-reviewer."
}

variable "relevance_gate_client_id" {
  type        = string
  description = "Entra app registration client ID for news-mas-relevance-gate."
}

# ── AWS / EC2 ─────────────────────────────────────────────────────────────────

variable "ami_id" {
  type        = string
  description = "AMI ID for the Ollama EC2 instance. Use a Deep Learning AMI (GPU) for the target region. Leave empty to use the module default data source."
  default     = ""
}

variable "subnet_ids" {
  type        = list(string)
  description = "Subnet IDs for the Ollama Auto Scaling Group. Defaults to default-VPC subnets when empty."
  default     = []
}

variable "container_apps_cidr_blocks" {
  type        = list(string)
  description = "CIDR blocks allowed to reach the Ollama API (port 11434). Tighten to the Container Apps outbound NAT IP in production."
  default     = ["0.0.0.0/0"]
}
