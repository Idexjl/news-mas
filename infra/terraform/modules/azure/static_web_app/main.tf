resource "azurerm_static_web_app" "main" {
  name                = "news-mas-ui-${var.env}"
  resource_group_name = var.resource_group_name
  location            = var.location
  sku_tier            = "Free"
  sku_size            = "Free"
}
