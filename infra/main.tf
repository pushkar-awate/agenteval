locals {
  name = "${var.project}-${var.environment}"
  tags = {
    Project     = var.project
    Environment = var.environment
    ManagedBy   = "terraform"
    Component   = "agenteval-eval-store"
  }
}

# Object store for eval artifacts: committed baselines and per-run scorecards +
# traces (the same envelopes evalcore.registry writes locally under runs/).
resource "aws_s3_bucket" "eval_artifacts" {
  bucket = "${local.name}-eval-artifacts"
  tags   = local.tags
}

# Versioning keeps the history of every baseline, so a promoted baseline can be
# rolled back - the storage-layer echo of the regression gate.
resource "aws_s3_bucket_versioning" "eval_artifacts" {
  bucket = aws_s3_bucket.eval_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Index of runs the dashboard / CI can query without scanning the bucket:
# one item per run (version -> task_success, saved_at, is_baseline).
resource "aws_dynamodb_table" "run_registry" {
  name         = "${local.name}-run-registry"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "version"

  attribute {
    name = "version"
    type = "S"
  }

  tags = local.tags
}
