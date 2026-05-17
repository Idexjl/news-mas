# news-mas Infrastructure

Terraform scaffold for deploying news-mas to Azure (primary) and AWS (Bedrock + Ollama spot GPU).

## Architecture overview

```
Azure (primary)                          AWS (secondary)
──────────────────────────────           ──────────────────────────────
Container Apps Environment               IAM Role (OIDC trust)
  └─ news-mas-api  (FastAPI gateway)       └─ bedrock:InvokeModel
       ├─ calls all 8 pipeline agents          └─ anthropic.claude-*
       └─ assumes Bedrock role via OIDC
                                         Auto Scaling Group (spot g5.xlarge)
Key Vault                                  └─ Ollama + gemma3:27b
  └─ ANTHROPIC_API_KEY                        (scaled 0→1 per pipeline run)
  └─ per-agent client secrets

Static Web App (Free tier)
  └─ future digest UI
```

Azure hosts all persistent resources and the pipeline runtime.  
AWS provides two optional capabilities:
- **Bedrock** — fallback LLM access if the primary Anthropic API is unavailable.
- **Ollama on spot EC2** — local model inference for HeatScorer and Selector, avoiding per-token costs.

## Prerequisites

| Tool | Min version | Notes |
|------|-------------|-------|
| Terraform | 1.7.0 | `terraform -version` |
| Azure CLI | latest | `az --version` |
| AWS CLI | v2 | `aws --version` |
| Entra permissions | Application Administrator | for app registrations |
| AWS permissions | IAMFullAccess + EC2 + BedrockFullAccess | |

## Bootstrap — create the Terraform state storage account

The remote state backend requires an Azure storage account that must exist before `terraform init`. Run this once per subscription:

```bash
LOCATION="eastus"
RG="news-mas-tfstate"
# Storage account names are globally unique — pick something distinct.
SA="newsmasstate$(openssl rand -hex 4)"

az group create --name "$RG" --location "$LOCATION"
az storage account create \
  --name "$SA" \
  --resource-group "$RG" \
  --sku Standard_LRS \
  --allow-blob-public-access false
az storage container create \
  --name tfstate \
  --account-name "$SA"

echo "storage_account_name = \"$SA\""
```

Copy the printed storage account name into `environments/dev/backend.tf`.

## Usage

All `terraform` commands are run from the environment directory, not the repo root.

```bash
cd infra/terraform/environments/dev
```

### 1 — Copy and populate the tfvars file

```bash
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars — the client IDs are already filled in from
# entra-app-ids.json. Add any AWS-specific overrides if needed.
```

`terraform.tfvars` is gitignored. Never commit it.

### 2 — Initialise

```bash
# Authenticate first if not already done.
az login
aws configure   # or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY

terraform init
```

### 3 — Plan

```bash
terraform plan -out=tfplan
```

Review the plan before applying. The first apply creates:
- 1 resource group, 1 Log Analytics workspace, 1 Container Apps environment, 1 Container App
- 1 Key Vault + access policy
- 1 Static Web App
- 1 IAM role + policy (AWS)
- 1 security group + launch template + Auto Scaling Group (AWS)

### 4 — Apply

```bash
terraform apply tfplan
```

Apply takes approximately 5–10 minutes. Outputs include `gateway_url`, `key_vault_uri`, and `bedrock_role_arn`.

### 5 — Post-apply: populate Key Vault secrets

```bash
KV_NAME="news-mas-dev-kv"
az keyvault secret set --vault-name "$KV_NAME" --name "ANTHROPIC-API-KEY" --value "<your-key>"
az keyvault secret set --vault-name "$KV_NAME" --name "TAVILY-API-KEY"    --value "<your-key>"
az keyvault secret set --vault-name "$KV_NAME" --name "MAS-SECRET-KEY"    --value "<fernet-key>"
```

Then wire Key Vault secret references into the Container App:

```bash
az containerapp secret set \
  --name news-mas-api \
  --resource-group news-mas-dev \
  --secrets "anthropic-api-key=keyvaultref:$(az keyvault secret show \
      --vault-name $KV_NAME --name ANTHROPIC-API-KEY --query id -o tsv),identityref:<managed-identity-id>"
```

### 6 — Scale the Ollama spot instance for a pipeline run

```bash
ASG=$(terraform output -raw ollama_asg_name)

# Before run — scale up (takes ~10 min to pull gemma3:27b on first boot)
aws autoscaling set-desired-capacity --auto-scaling-group-name "$ASG" --desired-capacity 1

# After run — scale back to zero to stop spot charges
aws autoscaling set-desired-capacity --auto-scaling-group-name "$ASG" --desired-capacity 0
```

### Destroy

```bash
terraform destroy
```

This removes all resources in the plan. The `news-mas-tfstate` resource group and storage account are **not** managed by this plan — delete them manually if needed.

## Module reference

| Module | Provider | Key resources |
|--------|----------|---------------|
| `azure/container_apps` | AzureRM | resource group, Log Analytics, Container Apps environment, gateway app |
| `azure/key_vault` | AzureRM | Key Vault (standard, soft-delete 7d) + access policy |
| `azure/static_web_app` | AzureRM | Static Web App (Free tier) |
| `aws/bedrock_iam` | AWS | IAM role + inline policy for `bedrock:InvokeModel` |
| `aws/spot_ec2` | AWS | security group, launch template (g5.xlarge spot), ASG (0–1) |

## Adding a new environment

```bash
cp -r environments/dev environments/staging
# Edit environments/staging/backend.tf  — change key to "news-mas-staging.tfstate"
# Edit environments/staging/terraform.tfvars.example — update env = "staging"
```
