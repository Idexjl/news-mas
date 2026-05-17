output "resource_group_name" {
  description = "Name of the resource group created by this module."
  value       = azurerm_resource_group.main.name
}

output "resource_group_id" {
  description = "Resource ID of the resource group."
  value       = azurerm_resource_group.main.id
}

output "container_app_environment_id" {
  description = "Resource ID of the Container Apps environment."
  value       = azurerm_container_app_environment.main.id
}

output "gateway_url" {
  description = "Public FQDN of the gateway Container App (without scheme)."
  value       = azurerm_container_app.gateway.latest_revision_fqdn
}
