import os
import json
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from openai import OpenAI
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

whisper_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
deepseek_client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)

APPS_SCRIPT_URL = os.environ["APPS_SCRIPT_URL"]

WHISPER_LANGUAGE = "zh"
WHISPER_PROMPT = ""  # 目前未設定自訂 Whisper prompt

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

HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>紡織巡檢語音上傳</title>
<style>
  body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
  h1 { color: #333; }
  .card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
  #drop-zone { border: 2px dashed #aaa; border-radius: 8px; padding: 40px; text-align: center; color: #666; cursor: pointer; transition: border-color 0.2s; }
  #drop-zone.dragover { border-color: #4a90e2; background: #eef4ff; }
  button { background: #4a90e2; color: white; border: none; padding: 10px 24px; border-radius: 6px; cursor: pointer; font-size: 15px; }
  button:disabled { background: #aaa; cursor: not-allowed; }
  #results { white-space: pre-wrap; font-family: monospace; font-size: 13px; background: #f0f0f0; padding: 12px; border-radius: 6px; max-height: 300px; overflow-y: auto; }
  details { margin-top: 0; }
  summary { cursor: pointer; font-weight: bold; color: #4a90e2; padding: 4px 0; user-select: none; }
  summary:hover { color: #2a6abf; }
  .prompt-block { margin-top: 12px; }
  .prompt-label { font-size: 12px; color: #888; margin-bottom: 4px; }
  .prompt-text { background: #f7f7f7; border: 1px solid #ddd; border-radius: 4px; padding: 10px; font-family: monospace; font-size: 13px; white-space: pre-wrap; word-break: break-all; }
  .tag { display: inline-block; background: #e8f0fe; color: #3367d6; border-radius: 4px; padding: 2px 8px; font-size: 12px; margin-left: 8px; }
</style>
</head>
<body>
<h1>紡織巡檢語音上傳</h1>

<div class="card">
  <details id="config-panel">
    <summary>目前 Prompt 設定 <span class="tag" id="config-status">載入中...</span></summary>
    <div class="prompt-block">
      <div class="prompt-label">Whisper 語言</div>
      <div class="prompt-text" id="whisper-language">—</div>
    </div>
    <div class="prompt-block">
      <div class="prompt-label">Whisper Prompt（語音辨識提示）</div>
      <div class="prompt-text" id="whisper-prompt">—</div>
    </div>
    <div class="prompt-block">
      <div class="prompt-label">Parse Prompt（DeepSeek 解析指令）</div>
      <div class="prompt-text" id="parse-prompt">—</div>
    </div>
  </details>
</div>

<div class="card">
  <div id="drop-zone">拖曳音檔到這裡，或點擊選擇檔案</div>
  <input type="file" id="file-input" multiple accept="audio/*" style="display:none">
  <div id="file-list" style="margin-top:12px; color:#555; font-size:14px;"></div>
  <div style="margin-top:16px;">
    <button id="upload-btn" disabled>上傳並處理</button>
  </div>
</div>

<div class="card" id="result-card" style="display:none">
  <strong>處理結果</strong>
  <div id="results" style="margin-top:10px;"></div>
</div>

<script>
  // 載入 config
  fetch('/config')
    .then(r => r.json())
    .then(cfg => {
      document.getElementById('whisper-language').textContent = cfg.whisper_language || '（未設定）';
      document.getElementById('whisper-prompt').textContent = cfg.whisper_prompt || '（未設定）';
      document.getElementById('parse-prompt').textContent = cfg.parse_prompt || '（未設定）';
      document.getElementById('config-status').textContent = '已載入';
      document.getElementById('config-status').style.background = '#e6f4ea';
      document.getElementById('config-status').style.color = '#2d7a3a';
    })
    .catch(() => {
      document.getElementById('config-status').textContent = '載入失敗';
      document.getElementById('config-status').style.background = '#fce8e6';
      document.getElementById('config-status').style.color = '#c5221f';
    });

  // 檔案選取
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const fileList = document.getElementById('file-list');
  const uploadBtn = document.getElementById('upload-btn');
  let selectedFiles = [];

  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    setFiles(Array.from(e.dataTransfer.files));
  });
  fileInput.addEventListener('change', () => setFiles(Array.from(fileInput.files)));

  function setFiles(files) {
    selectedFiles = files;
    fileList.textContent = files.length ? files.map(f => f.name).join('、') : '';
    uploadBtn.disabled = files.length === 0;
  }

  // 上傳
  uploadBtn.addEventListener('click', async () => {
    uploadBtn.disabled = true;
    uploadBtn.textContent = '處理中...';
    const form = new FormData();
    selectedFiles.forEach(f => form.append('files', f));
    try {
      const res = await fetch('/process', { method: 'POST', body: form });
      const data = await res.json();
      document.getElementById('result-card').style.display = 'block';
      document.getElementById('results').textContent = JSON.stringify(data, null, 2);
    } catch(e) {
      document.getElementById('result-card').style.display = 'block';
      document.getElementById('results').textContent = '錯誤：' + e.message;
    }
    uploadBtn.disabled = false;
    uploadBtn.textContent = '上傳並處理';
  });
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE

@app.get("/config")
def get_config():
    return {
        "whisper_language": WHISPER_LANGUAGE,
        "whisper_prompt": WHISPER_PROMPT,
        "parse_prompt": PARSE_PROMPT,
    }

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
                    language=WHISPER_LANGUAGE,
                    **({"prompt": WHISPER_PROMPT} if WHISPER_PROMPT else {})
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
