# One provider block that works against either real AWS or LocalStack.
# With use_localstack = true it points S3/DynamoDB at the LocalStack edge port,
# uses dummy credentials, and skips the account/credential preflight checks that
# would otherwise fail with no real AWS account.
provider "aws" {
  region = var.region

  access_key                  = var.use_localstack ? "test" : null
  secret_key                  = var.use_localstack ? "test" : null
  skip_credentials_validation = var.use_localstack
  skip_requesting_account_id  = var.use_localstack
  skip_metadata_api_check     = var.use_localstack
  s3_use_path_style           = var.use_localstack

  dynamic "endpoints" {
    for_each = var.use_localstack ? [1] : []
    content {
      s3       = var.localstack_endpoint
      dynamodb = var.localstack_endpoint
    }
  }
}
