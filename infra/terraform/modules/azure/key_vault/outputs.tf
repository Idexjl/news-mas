output "vault_uri" {
  description = "URI of the Key Vault (e.g. https://news-mas-dev-kv.vault.azure.net/)."
  value       = azurerm_key_vault.main.vault_uri
}

output "vault_id" {
  description = "Resource ID of the Key Vault."
  value       = azurerm_key_vault.main.id
}
