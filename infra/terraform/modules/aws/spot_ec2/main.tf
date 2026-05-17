data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Auto-select the latest Deep Learning AMI (GPU) when ami_id is not supplied.
data "aws_ami" "deep_learning_gpu" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04) *"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

locals {
  effective_ami_id    = var.ami_id != "" ? var.ami_id : data.aws_ami.deep_learning_gpu.id
  effective_subnet_ids = length(var.subnet_ids) > 0 ? var.subnet_ids : data.aws_subnets.default.ids

  ollama_user_data = <<-EOT
    #!/bin/bash
    set -euo pipefail
    export HOME=/root

    # Install Ollama
    curl -fsSL https://ollama.ai/install.sh | sh

    # Listen on all interfaces so Container Apps can reach port 11434.
    mkdir -p /etc/systemd/system/ollama.service.d
    cat > /etc/systemd/system/ollama.service.d/override.conf <<EOF
    [Service]
    Environment="OLLAMA_HOST=0.0.0.0"
    EOF

    systemctl daemon-reload
    systemctl enable --now ollama

    # Wait for the API to be ready.
    until curl -sf http://localhost:11434/api/tags > /dev/null; do
      sleep 3
    done

    # Pull the model used by HeatScorer and Selector agents.
    # gemma3:27b requires ~16 GB VRAM; g5.xlarge has 24 GB.
    ollama pull gemma3:27b
  EOT
}

resource "aws_security_group" "ollama" {
  name        = "news-mas-ollama-${var.env}"
  description = "Controls access to the Ollama API (port 11434) on spot EC2."
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "Ollama API — restrict to Container Apps outbound NAT IP in production."
    from_port   = 11434
    to_port     = 11434
    protocol    = "tcp"
    cidr_blocks = var.container_apps_cidr_blocks
  }

  egress {
    description = "Unrestricted outbound (model downloads, Ollama updates)."
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "news-mas-ollama-${var.env}"
    Project = "news-mas"
    Env     = var.env
  }
}

resource "aws_launch_template" "ollama" {
  name_prefix   = "news-mas-ollama-${var.env}-"
  image_id      = local.effective_ami_id
  instance_type = "g5.xlarge"

  instance_market_options {
    market_type = "spot"
    spot_options {
      max_price          = var.spot_max_price
      spot_instance_type = "one-time"
    }
  }

  network_interfaces {
    associate_public_ip_address = true
    security_groups             = [aws_security_group.ollama.id]
  }

  user_data = base64encode(local.ollama_user_data)

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size           = 100
      volume_type           = "gp3"
      delete_on_termination = true
    }
  }

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name    = "news-mas-ollama-${var.env}"
      Project = "news-mas"
      Env     = var.env
    }
  }

  tag_specifications {
    resource_type = "spot-instances-request"
    tags = {
      Name    = "news-mas-ollama-${var.env}"
      Project = "news-mas"
      Env     = var.env
    }
  }

  tags = {
    Project = "news-mas"
    Env     = var.env
  }
}

resource "aws_autoscaling_group" "ollama" {
  name                = "news-mas-ollama-${var.env}"
  vpc_zone_identifier = local.effective_subnet_ids

  # Start at 0. Scale to 1 before a pipeline run via:
  #   aws autoscaling set-desired-capacity \
  #     --auto-scaling-group-name news-mas-ollama-dev --desired-capacity 1
  min_size         = 0
  max_size         = 1
  desired_capacity = 0

  launch_template {
    id      = aws_launch_template.ollama.id
    version = "$Latest"
  }

  # Give the instance time to install Ollama and pull the model (~10 min).
  health_check_grace_period = 600
  health_check_type         = "EC2"

  tag {
    key                 = "Name"
    value               = "news-mas-ollama-${var.env}"
    propagate_at_launch = true
  }

  tag {
    key                 = "Project"
    value               = "news-mas"
    propagate_at_launch = true
  }

  tag {
    key                 = "Env"
    value               = var.env
    propagate_at_launch = true
  }
}
