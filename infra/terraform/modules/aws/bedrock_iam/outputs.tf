output "role_arn" {
  description = "ARN of the IAM role that grants bedrock:InvokeModel. Provide to the gateway as BEDROCK_ROLE_ARN."
  value       = aws_iam_role.bedrock.arn
}

output "policy_arn" {
  description = "ARN of the Bedrock invoke IAM policy."
  value       = aws_iam_policy.bedrock_invoke.arn
}
