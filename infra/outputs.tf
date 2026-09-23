output "artifacts_bucket" {
  description = "S3 bucket holding baselines and run artifacts."
  value       = aws_s3_bucket.eval_artifacts.bucket
}

output "run_registry_table" {
  description = "DynamoDB table indexing eval runs."
  value       = aws_dynamodb_table.run_registry.name
}

output "endpoint" {
  description = "Where the resources live (LocalStack or real AWS)."
  value       = var.use_localstack ? var.localstack_endpoint : "aws:${var.region}"
}
