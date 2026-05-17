terraform {
  required_version = ">= 1.7.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.110"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.50"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
  tenant_id       = var.tenant_id
}

provider "aws" {
  region = var.aws_region
}

# ── Azure modules ─────────────────────────────────────────────────────────────

module "container_apps" {
  source = "../../modules/azure/container_apps"

  env      = var.env
  location = var.azure_location

  tenant_id                = var.tenant_id
  gateway_client_id        = var.gateway_client_id
  search_worker_client_id  = var.search_worker_client_id
  heat_scorer_client_id    = var.heat_scorer_client_id
  filter_agent_client_id   = var.filter_agent_client_id
  selector_client_id       = var.selector_client_id
  phase1_judge_client_id   = var.phase1_judge_client_id
  summarizer_client_id     = var.summarizer_client_id
  reviewer_client_id       = var.reviewer_client_id
  relevance_gate_client_id = var.relevance_gate_client_id
}

module "key_vault" {
  source = "../../modules/azure/key_vault"

  env                 = var.env
  location            = var.azure_location
  resource_group_name = module.container_apps.resource_group_name
  tenant_id           = var.tenant_id
}

module "static_web_app" {
  source = "../../modules/azure/static_web_app"

  env                 = var.env
  location            = var.azure_location
  resource_group_name = module.container_apps.resource_group_name
}

# ── AWS modules ───────────────────────────────────────────────────────────────

module "bedrock_iam" {
  source = "../../modules/aws/bedrock_iam"

  env = var.env
}

module "spot_ec2" {
  source = "../../modules/aws/spot_ec2"

  env                        = var.env
  region                     = var.aws_region
  ami_id                     = var.ami_id
  subnet_ids                 = var.subnet_ids
  container_apps_cidr_blocks = var.container_apps_cidr_blocks
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "gateway_url" {
  description = "Public FQDN of the news-mas-api Container App."
  value       = module.container_apps.gateway_url
}

output "key_vault_uri" {
  description = "Key Vault URI for secret references."
  value       = module.key_vault.vault_uri
}

output "static_web_app_url" {
  description = "Static Web App default hostname."
  value       = module.static_web_app.default_host_name
}

output "bedrock_role_arn" {
  description = "IAM role ARN for Bedrock access. Set as BEDROCK_ROLE_ARN in the gateway."
  value       = module.bedrock_iam.role_arn
}

output "ollama_asg_name" {
  description = "Ollama Auto Scaling Group name."
  value       = module.spot_ec2.asg_name
}
