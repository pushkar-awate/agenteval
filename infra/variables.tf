variable "project" {
  description = "Name prefix for all resources."
  type        = string
  default     = "agenteval"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)."
  type        = string
  default     = "dev"
}

variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "use_localstack" {
  description = "Target LocalStack (free, local) instead of real AWS. Flips the provider endpoints and credential handling so `terraform apply` runs with no AWS account."
  type        = bool
  default     = true
}

variable "localstack_endpoint" {
  description = "LocalStack edge endpoint."
  type        = string
  default     = "http://localhost:4566"
}
