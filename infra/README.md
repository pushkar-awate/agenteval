# infra - eval artifact store (Terraform + LocalStack)

A small Terraform module that provisions the cloud backing for agenteval's eval
history, and runs **free and offline against [LocalStack](https://localstack.cloud)**
so you can `terraform apply` with no AWS account.

## What it creates

- **S3 bucket** (`<project>-<env>-eval-artifacts`, versioned) - stores committed
  baselines and per-run scorecards + traces, the same envelopes
  `evalcore.registry` writes under `runs/` locally. Versioning keeps every
  baseline, so a promotion can be rolled back - the storage-layer echo of the
  regression gate.
- **DynamoDB table** (`<project>-<env>-run-registry`, pay-per-request) - one item
  per run (`version` -> task_success, saved_at, is_baseline) so the dashboard or
  CI can list runs without scanning the bucket.

## Run it locally (LocalStack, free)

```
docker compose up -d                 # LocalStack on :4566
terraform init
terraform apply -auto-approve        # use_localstack = true by default
terraform output                     # bucket + table names
```

Nothing here touches a real AWS account: the provider uses dummy credentials and
LocalStack endpoints, gated by `use_localstack`.

## Target real AWS

Set `use_localstack = false` in `terraform.tfvars`, provide AWS credentials the
usual way (env vars / shared config / assumed role - never in a file here), and
`terraform apply`. The same module provisions the real bucket and table.

## How it maps back to agenteval

The local `evalcore.registry.Registry` writes `baseline.json` and `runs/*.json`
to disk. This module is where those go in a hosted setup: the bucket is the
run/baseline store, the table is the queryable index. The regression gate logic
is unchanged - only where the artifacts live moves.
