# ── Azure outputs ─────────────────────────────────────────────────────────────

output "resource_group_name" {
  description = "Name of the Azure resource group."
  value       = module.container_apps.resource_group_name
}

output "container_app_environment_id" {
  description = "Resource ID of the Container Apps environment."
  value       = module.container_apps.container_app_environment_id
}

output "gateway_url" {
  description = "Public FQDN of the news-mas-api Container App."
  value       = module.container_apps.gateway_url
}

output "key_vault_uri" {
  description = "URI of the Key Vault (use for secret references in Container Apps)."
  value       = module.key_vault.vault_uri
}

output "key_vault_id" {
  description = "Resource ID of the Key Vault."
  value       = module.key_vault.vault_id
}

output "static_web_app_url" {
  description = "Default hostname of the Static Web App (prefix with https://)."
  value       = module.static_web_app.default_host_name
}

output "static_web_app_api_key" {
  description = "Deployment API key for the Static Web App (use in CI/CD)."
  value       = module.static_web_app.api_key
  sensitive   = true
}

# ── AWS outputs ───────────────────────────────────────────────────────────────

output "bedrock_role_arn" {
  description = "ARN of the IAM role that grants Bedrock InvokeModel. Provide to the gateway as BEDROCK_ROLE_ARN."
  value       = module.bedrock_iam.role_arn
}

output "bedrock_policy_arn" {
  description = "ARN of the Bedrock invoke IAM policy."
  value       = module.bedrock_iam.policy_arn
}

output "ollama_security_group_id" {
  description = "Security group ID governing Ollama EC2 ingress. Add the Container Apps NAT IP here to lock down access."
  value       = module.spot_ec2.security_group_id
}

output "ollama_asg_name" {
  description = "Name of the Ollama Auto Scaling Group. Scale desired to 1 before a pipeline run, 0 after."
  value       = module.spot_ec2.asg_name
}
