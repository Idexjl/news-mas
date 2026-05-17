terraform {
  backend "azurerm" {
    resource_group_name  = "news-mas-tfstate"
    # Replace with the storage account name you created in the bootstrap step.
    # See infra/README.md §Bootstrap for the az CLI commands.
    storage_account_name = "placeholder-replace-with-sa-name"
    container_name       = "tfstate"
    key                  = "news-mas-dev.tfstate"
  }
}
