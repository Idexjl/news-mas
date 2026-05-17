data "aws_caller_identity" "current" {}

# Trust policy — Azure Container Apps assumes this role via OIDC federation.
# Replace the placeholder OIDC provider ARN and subject claim once the Azure
# federated identity credential is set up on the gateway's managed identity.
data "aws_iam_policy_document" "bedrock_trust" {
  statement {
    sid     = "AllowAzureContainerAppsOIDC"
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type = "Federated"
      # TODO: replace with the actual OIDC provider ARN after running:
      #   aws iam create-open-id-connect-provider \
      #     --url https://login.microsoftonline.com/<tenant-id>/v2.0 \
      #     --client-id-list <gateway-client-id> \
      #     --thumbprint-list <thumbprint>
      identifiers = [
        "arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/placeholder-azure-oidc-provider"
      ]
    }

    condition {
      test     = "StringEquals"
      variable = "placeholder-azure-oidc-provider:sub"
      # Subject claim emitted by Azure for the gateway managed identity.
      values = ["system:serviceaccount:news-mas:gateway"]
    }
  }
}

resource "aws_iam_role" "bedrock" {
  name               = "news-mas-bedrock-${var.env}"
  assume_role_policy = data.aws_iam_policy_document.bedrock_trust.json

  tags = {
    Project = "news-mas"
    Env     = var.env
  }
}

data "aws_iam_policy_document" "bedrock_invoke" {
  statement {
    sid     = "AllowBedrockInvoke"
    effect  = "Allow"
    actions = ["bedrock:InvokeModel"]
    resources = [
      "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-*",
    ]
  }
}

resource "aws_iam_policy" "bedrock_invoke" {
  name        = "news-mas-bedrock-invoke-${var.env}"
  description = "Allow news-mas agents to invoke Anthropic Claude models via Bedrock."
  policy      = data.aws_iam_policy_document.bedrock_invoke.json

  tags = {
    Project = "news-mas"
    Env     = var.env
  }
}

resource "aws_iam_role_policy_attachment" "bedrock" {
  role       = aws_iam_role.bedrock.name
  policy_arn = aws_iam_policy.bedrock_invoke.arn
}
