data "azurerm_client_config" "current" {}

resource "azurerm_key_vault" "main" {
  name                = "news-mas-${var.env}-kv"
  location            = var.location
  resource_group_name = var.resource_group_name
  tenant_id           = var.tenant_id
  sku_name            = "standard"

  # 7-day soft-delete gives a recovery window without blocking re-deployment in dev.
  soft_delete_retention_days = 7

  # purge_protection disabled for dev so `terraform destroy` can fully clean up.
  # Enable in staging/prod to meet compliance requirements.
  purge_protection_enabled = false
}

# Grant the identity running Terraform full secret access so it can write
# ANTHROPIC_API_KEY and other runtime secrets after apply.
resource "azurerm_key_vault_access_policy" "current_user" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = data.azurerm_client_config.current.object_id

  secret_permissions = [
    "Get",
    "List",
    "Set",
    "Delete",
    "Purge",
    "Recover",
  ]
}
