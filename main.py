from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
import boto3
from botocore.config import Config
import json
import uuid
import os

app = FastAPI()
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


sigv4_config = Config(signature_version='s3v4')
s3 = boto3.client("s3", region_name="us-east-2", config=sigv4_config)
BUCKET_NAME = "area-of-origin-images"


@app.post("/generate-upload-url")
async def generate_upload_url(request: Request):
    try:
        data = await request.json()
        content_type = data.get("content_type", "image/jpeg")

        file_ext = "jpg" if "jpeg" in content_type else "png"
        file_key = f"uploads/{uuid.uuid4()}.{file_ext}"

        # ✅ Generate presigned PUT URL
        presigned_url = s3.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": BUCKET_NAME,
                "Key": file_key,
                "ContentType": content_type
            },
            ExpiresIn=900
        )

        return {
            "upload_url": presigned_url,
            "file_url": f"https://{BUCKET_NAME}.s3.us-east-2.amazonaws.com/{file_key}"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import requests
import base64

@app.post("/generate-summary")
async def generate_summary(request: Request):
    try:
        data = await request.json()
        image_urls = data.get("image_urls", [])
        user_prompt = data.get("prompt", "Analyze these fire scene images and describe the area of origin.")

        if not image_urls:
            return JSONResponse(status_code=400, content={"error": "No image URLs provided."})

        content = []

        for url in image_urls:
            response = requests.get(url)
            if response.status_code != 200:
                return JSONResponse(status_code=400, content={"error": f"Failed to fetch image: {url}"})

            image_bytes = response.content
            content.append({
                "image": {
                    "format": "jpeg",  # or "png", based on your use case
                    "source": {
                        "bytes": base64.b64encode(image_bytes).decode("utf-8")
                    }
                }
            })

        # Add prompt as user input
        content.append({"text": user_prompt})
        messages = [{"role": "user", "content": content}]

        # Call Claude
        response = bedrock.converse(
            modelId=MODEL_ID,
            messages=messages
        )

        output_text = response["output"]["message"]["content"][0]["text"]
        return {"result": output_text.strip()}

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