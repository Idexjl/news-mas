output "ollama_endpoint" {
  description = <<-EOT
    Ollama API base URL. The instance IP is dynamic (spot ASG); retrieve it after scale-out:
      aws ec2 describe-instances \
        --filters "Name=tag:aws:autoscaling:groupName,Values=${asg_name}" \
        --query 'Reservations[*].Instances[*].PublicIpAddress' \
        --output text
    Then set OLLAMA_BASE_URL=http://<ip>:11434 in the Container Apps environment.
  EOT
  value       = "http://<dynamic-spot-ip>:11434"
}

output "asg_name" {
  description = "Name of the Ollama Auto Scaling Group. Scale desired_capacity to 1 before a pipeline run and back to 0 afterward to minimise spot cost."
  value       = aws_autoscaling_group.ollama.name
}

output "security_group_id" {
  description = "ID of the security group controlling Ollama ingress. Add the Container Apps outbound NAT IP as an additional ingress rule to lock down access."
  value       = aws_security_group.ollama.id
}

output "launch_template_id" {
  description = "ID of the Ollama launch template. Update the AMI here when a new Deep Learning AMI is released."
  value       = aws_launch_template.ollama.id
}
