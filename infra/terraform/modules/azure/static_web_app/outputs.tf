output "default_host_name" {
  description = "Default hostname of the Static Web App (prefix with https:// to form the URL)."
  value       = azurerm_static_web_app.main.default_host_name
}

output "api_key" {
  description = "Deployment API key used by GitHub Actions / CI pipelines to publish the frontend."
  value       = azurerm_static_web_app.main.api_key
  sensitive   = true
}
