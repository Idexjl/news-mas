variable "env" {
  type        = string
  description = "Environment name used as a resource name suffix."
}

variable "location" {
  type        = string
  description = "Azure region. Static Web Apps support a limited set of regions; eastus2 and westus2 are recommended."
}

variable "resource_group_name" {
  type        = string
  description = "Name of the resource group to deploy into."
}
