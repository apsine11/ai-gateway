from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
import boto3
import json

app = FastAPI()

# Claude 3.5 Sonnet model (supports image + text)
# MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
MODEL_ID = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

@app.post("/generate-narrative")
async def generate_narrative(prompt: str = Form(...), image: UploadFile = File(None)):
    try:
        # Build message content
        content = []

        # Add image if provided
        if image:
            image_bytes = await image.read()
            content.append({
                "image": {
                    "format": image.content_type.split("/")[-1],  # e.g., "png", "jpeg"
                    "source": {"bytes": image_bytes}
                }
            })

        # Add prompt text
        content.append({"text": prompt})

        messages = [{"role": "user", "content": content}]

        response = bedrock.converse(
            modelId=MODEL_ID,
            messages=messages
        )

        output_text = response["output"]["message"]["content"][0]["text"]
        return {"model": MODEL_ID, "result": output_text}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/grammar-check")
async def grammar_check(request: Request):
    try:
        data = await request.json()
        original_text = data.get("text", "")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            "Please review the following text and improve its grammar, punctuation, and clarity "
                            "without changing its tone or meaning. Only return the corrected version, without explanation.\n\n"
                            f"{original_text}"
                        )
                    }
                ]
            }
        ]

        response = bedrock.converse(
            modelId=MODEL_ID,
            messages=messages,
        )

        output_text = response["output"]["message"]["content"][0]["text"]
        return {"corrected": output_text.strip()}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})