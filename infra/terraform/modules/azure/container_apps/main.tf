resource "azurerm_resource_group" "main" {
  name     = "news-mas-${var.env}"
  location = var.location
}

resource "azurerm_log_analytics_workspace" "main" {
  name                = "news-mas-${var.env}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

resource "azurerm_container_app_environment" "main" {
  name                       = "news-mas-${var.env}"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
}

resource "azurerm_container_app" "gateway" {
  name                         = "news-mas-api"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"

  template {
    min_replicas = 0
    max_replicas = 3

    container {
      name   = "gateway"
      image  = "mcr.microsoft.com/azuredocs/containerapps-helloworld:latest"
      cpu    = 0.5
      memory = "1Gi"

      # Secrets are referenced via Key Vault once the vault module is applied.
      # Replace placeholder values by setting secret refs in the Azure portal or
      # via `az containerapp secret set` after the Key Vault is provisioned.
      env {
        name  = "ANTHROPIC_API_KEY"
        value = "placeholder-replace-with-key-vault-ref"
      }
      env {
        name  = "ENTRA_TENANT_ID"
        value = var.tenant_id
      }
      env {
        name  = "MAS_ENV"
        value = var.env
      }
      env {
        name  = "GATEWAY_CLIENT_ID"
        value = var.gateway_client_id
      }
      env {
        name  = "SEARCH_WORKER_CLIENT_ID"
        value = var.search_worker_client_id
      }
      env {
        name  = "HEAT_SCORER_CLIENT_ID"
        value = var.heat_scorer_client_id
      }
      env {
        name  = "FILTER_AGENT_CLIENT_ID"
        value = var.filter_agent_client_id
      }
      env {
        name  = "SELECTOR_CLIENT_ID"
        value = var.selector_client_id
      }
      env {
        name  = "PHASE1_JUDGE_CLIENT_ID"
        value = var.phase1_judge_client_id
      }
      env {
        name  = "SUMMARIZER_CLIENT_ID"
        value = var.summarizer_client_id
      }
      env {
        name  = "REVIEWER_CLIENT_ID"
        value = var.reviewer_client_id
      }
      env {
        name  = "RELEVANCE_GATE_CLIENT_ID"
        value = var.relevance_gate_client_id
      }
    }
  }

  ingress {
    external_enabled = true
    target_port      = 8000

    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }
}
