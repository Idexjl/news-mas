variable "env" {
  type        = string
  description = "Environment name used as a resource name suffix."
}

variable "location" {
  type        = string
  description = "Azure region."
}

variable "resource_group_name" {
  type        = string
  description = "Name of the resource group to deploy into (created by the container_apps module)."
}

variable "tenant_id" {
  type        = string
  description = "Azure Entra ID tenant ID for the Key Vault tenant configuration."
}
