variable "env" {
  type        = string
  description = "Environment name used as a resource name suffix."
}

variable "region" {
  type        = string
  description = "AWS region. Must have g5.xlarge spot capacity; us-west-2 and us-east-1 are reliable."
}

variable "ami_id" {
  type        = string
  description = "AMI ID for the Ollama instance. Leave empty to auto-select the latest Deep Learning AMI (GPU) for the region."
  default     = ""
}

variable "subnet_ids" {
  type        = list(string)
  description = "Subnet IDs for the Auto Scaling Group. Defaults to default-VPC subnets when empty."
  default     = []
}

variable "container_apps_cidr_blocks" {
  type        = list(string)
  description = "CIDR blocks allowed to call the Ollama API (port 11434). Lock down to the Container Apps outbound NAT IP in production."
  default     = ["0.0.0.0/0"]
}

variable "spot_max_price" {
  type        = string
  description = "Maximum spot bid price per hour (USD). Set to on-demand price as a ceiling."
  default     = "1.50"
}
