import boto3

# Use the SSO profile and connect to Bedrock *admin* API
session = boto3.Session(profile_name="bedrock-sso")
bedrock = session.client("bedrock", region_name="us-east-1")  # <-- NOT bedrock-runtime

# Now list available models
models = bedrock.list_foundation_models()
for m in models["modelSummaries"]:
    print(m["modelId"])
