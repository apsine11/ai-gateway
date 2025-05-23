from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
import boto3
from botocore.config import Config
import json
import uuid
import os

app = FastAPI()

# Claude 3.5 Sonnet model (supports image + text)
# MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
MODEL_ID = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

sigv4_config = Config(signature_version='s3v4')


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


s3 = boto3.client("s3", region_name="us-east-1")
BUCKET_NAME = "area-of-origin-images"


@app.post("/generate-upload-url")
async def generate_upload_url(request: Request):
    try:
        data = await request.json()
        content_type = data.get("content_type", "image/jpeg")

        # Generate a file key
        file_ext = "jpg" if "jpeg" in content_type else "png"
        file_key = f"uploads/{uuid.uuid4()}.{file_ext}"

        # Generate presigned POST URL + form fields
        post = s3.generate_presigned_post(
            Bucket=BUCKET_NAME,
            Key=file_key,
            Fields={"Content-Type": content_type},
            Conditions=[{"Content-Type": content_type}],
            ExpiresIn=900
        )

        return {
            "upload_url": post["url"],     # Always "https://<bucket>.s3.amazonaws.com/"
            "fields": post["fields"],      # Must be used in the POST body
            "file_url": f"https://{BUCKET_NAME}.s3.amazonaws.com/{file_key}"
        }

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