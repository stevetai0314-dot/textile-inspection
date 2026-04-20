import os
import json
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

whisper_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

APPS_SCRIPT_URL = os.getenv("APPS_SCRIPT_URL")

PARSE_PROMPT = """
你是紡織工廠巡檢助手。以下是工人用語音記錄的量測數據原文，請解析成結構化資料。

語音原文：
{transcript}

請輸出 JSON 格式（只輸出 JSON，不要其他文字）：
{{
  "機台": "機台編號或名稱",
  "左右邊": "左 或 右 或 中 或 空白",
  "規格": "規格描述或空白",
  "量測數據": "數值含單位",
  "原始文字": "語音轉錄原文"
}}
"""

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/process")
async def process_audio(files: list[UploadFile] = File(...)):
    results = []

    for file in files:
        try:
            suffix = "." + file.filename.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            with open(tmp_path, "rb") as audio_file:
                transcript = whisper_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="zh"
                ).text

            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": PARSE_PROMPT.format(transcript=transcript)}],
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            data["音檔名稱"] = file.filename

            requests.post(APPS_SCRIPT_URL, json=data)
            os.unlink(tmp_path)

            results.append({"file": file.filename, "status": "ok", "data": data})

        except Exception as e:
            results.append({"file": file.filename, "status": "error", "message": str(e)})

    return {"results": results}
