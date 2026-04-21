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

WHISPER_LANGUAGE = "vi"

WHISPER_PROMPT = (
    "Công nhân nhà máy dệt ghi lại số đo kiểm tra. "
    "Các từ thường gặp: máy số, bên trái, bên phải, chính giữa, "
    "quy cách, độ dày, độ rộng, mật độ, sợi, vải, mm, cm, g/m2."
)

PARSE_PROMPT = """
Bạn là trợ lý kiểm tra nhà máy dệt. Dưới đây là văn bản gốc được công nhân ghi âm bằng giọng nói. Hãy phân tích thành dữ liệu có cấu trúc.

Văn bản gốc:
{transcript}

Hãy xuất định dạng JSON (chỉ xuất JSON, không có văn bản khác):
{{
  "機台": "mã số hoặc tên máy",
  "左右邊": "trái hoặc phải hoặc giữa hoặc để trống",
  "規格": "mô tả quy cách hoặc để trống",
  "量測數據": "giá trị kèm đơn vị",
  "原始文字": "văn bản chuyển từ giọng nói"
}}
"""

HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>紡織巡檢上傳系統</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: "Noto Sans TC", sans-serif; background: #f0f2f5; min-height: 100vh; padding: 32px 16px; }
  h1 { text-align: center; color: #1a1a2e; margin-bottom: 8px; font-size: 1.6rem; }
  .subtitle { text-align: center; color: #666; margin-bottom: 32px; font-size: 0.9rem; }
  .card { background: white; border-radius: 12px; padding: 28px; max-width: 700px; margin: 0 auto 24px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }
  .drop-zone {
    border: 2px dashed #cbd5e1; border-radius: 10px; padding: 48px 24px;
    text-align: center; cursor: pointer; transition: all 0.2s; color: #94a3b8;
  }
  .drop-zone:hover, .drop-zone.drag-over { border-color: #6366f1; background: #f0f0ff; color: #6366f1; }
  .drop-zone svg { width: 48px; height: 48px; margin-bottom: 12px; }
  .drop-zone p { font-size: 1rem; margin-bottom: 4px; }
  .drop-zone small { font-size: 0.8rem; }
  #fileInput { display: none; }
  #fileList { margin-top: 16px; }
  .file-item {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px; background: #f8fafc; border-radius: 8px; margin-bottom: 6px; font-size: 0.88rem;
  }
  .file-item svg { color: #6366f1; flex-shrink: 0; }
  .btn {
    display: block; width: 100%; padding: 14px; margin-top: 20px;
    background: #6366f1; color: white; border: none; border-radius: 10px;
    font-size: 1rem; font-weight: 600; cursor: pointer; transition: background 0.2s;
  }
  .btn:hover { background: #4f46e5; }
  .btn:disabled { background: #a5b4fc; cursor: not-allowed; }
  #status { margin-top: 16px; text-align: center; font-size: 0.9rem; color: #6366f1; min-height: 20px; }
  .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid #c7d2fe; border-top-color: #6366f1; border-radius: 50%; animation: spin 0.7s linear infinite; vertical-align: middle; margin-right: 6px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  #results { max-width: 700px; margin: 0 auto; }
  .result-card { background: white; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }
  .result-header { display: flex; align-items: center; gap: 8px; margin-bottom: 14px; }
  .badge { padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
  .badge-ok { background: #dcfce7; color: #16a34a; }
  .badge-error { background: #fee2e2; color: #dc2626; }
  .result-filename { font-weight: 600; color: #1e293b; font-size: 0.95rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  th { text-align: left; color: #64748b; padding: 6px 10px; border-bottom: 1px solid #e2e8f0; width: 120px; }
  td { padding: 6px 10px; color: #1e293b; border-bottom: 1px solid #f1f5f9; }
  tr:last-child td, tr:last-child th { border-bottom: none; }
  .sheet-badge { display: inline-flex; align-items: center; gap: 5px; background: #f0fdf4; color: #15803d; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; margin-top: 12px; }
  /* Prompt panel */
  .config-card { background: white; border-radius: 12px; padding: 20px 28px; max-width: 700px; margin: 0 auto 24px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }
  details summary { cursor: pointer; font-weight: 600; color: #6366f1; font-size: 0.95rem; user-select: none; list-style: none; display: flex; align-items: center; gap: 8px; }
  details summary::-webkit-details-marker { display: none; }
  details summary::before { content: "▶"; font-size: 0.7rem; transition: transform 0.2s; }
  details[open] summary::before { transform: rotate(90deg); }
  .config-tag { padding: 2px 8px; border-radius: 12px; font-size: 0.72rem; font-weight: 600; background: #e0e7ff; color: #4338ca; margin-left: auto; }
  .config-section { margin-top: 16px; }
  .config-label { font-size: 0.75rem; color: #94a3b8; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
  .config-value { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; font-family: monospace; font-size: 0.82rem; white-space: pre-wrap; word-break: break-all; color: #334155; max-height: 200px; overflow-y: auto; }
</style>
</head>
<body>
<h1>紡織巡檢上傳系統</h1>
<p class="subtitle">上傳語音音檔，自動轉錄並寫入 Google Sheet</p>

<div class="config-card">
  <details id="configPanel">
    <summary>目前 Prompt 設定 <span class="config-tag" id="configTag">載入中</span></summary>
    <div class="config-section">
      <div class="config-label">Whisper 語言</div>
      <div class="config-value" id="cfgLang">—</div>
    </div>
    <div class="config-section">
      <div class="config-label">Whisper Prompt（語音辨識提示）</div>
      <div class="config-value" id="cfgWhisper">—</div>
    </div>
    <div class="config-section">
      <div class="config-label">Parse Prompt（DeepSeek 解析指令）</div>
      <div class="config-value" id="cfgParse">—</div>
    </div>
  </details>
</div>

<div class="card">
  <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
      <path d="M12 16V4m0 0L8 8m4-4l4 4M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2"/>
    </svg>
    <p>點擊或拖曳音檔至此</p>
    <small>支援 mp3、wav、m4a、ogg 等格式，可同時上傳多個</small>
  </div>
  <input type="file" id="fileInput" multiple accept="audio/*">
  <div id="fileList"></div>
  <button class="btn" id="uploadBtn" disabled onclick="uploadFiles()">上傳並送出</button>
  <div id="status"></div>
</div>

<div id="results"></div>

<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const uploadBtn = document.getElementById('uploadBtn');
let selectedFiles = [];

// 載入 config
fetch('/config')
  .then(r => r.json())
  .then(cfg => {
    document.getElementById('cfgLang').textContent = cfg.whisper_language || '（未設定）';
    document.getElementById('cfgWhisper').textContent = cfg.whisper_prompt || '（未設定）';
    document.getElementById('cfgParse').textContent = cfg.parse_prompt || '（未設定）';
    const tag = document.getElementById('configTag');
    tag.textContent = '已載入';
    tag.style.background = '#dcfce7';
    tag.style.color = '#16a34a';
  })
  .catch(() => {
    const tag = document.getElementById('configTag');
    tag.textContent = '載入失敗';
    tag.style.background = '#fee2e2';
    tag.style.color = '#dc2626';
  });

fileInput.addEventListener('change', () => handleFiles(fileInput.files));

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  handleFiles(e.dataTransfer.files);
});

function handleFiles(files) {
  selectedFiles = Array.from(files);
  fileList.innerHTML = selectedFiles.map(f => `
    <div class="file-item">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M9 19V6l12-3v13M9 19c0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2 2 .9 2 2zm12-3c0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2 2 .9 2 2z"/>
      </svg>
      ${f.name} <span style="color:#94a3b8;margin-left:auto">${(f.size/1024).toFixed(1)} KB</span>
    </div>
  `).join('');
  uploadBtn.disabled = selectedFiles.length === 0;
  document.getElementById('results').innerHTML = '';
}

async function uploadFiles() {
  if (!selectedFiles.length) return;

  const status = document.getElementById('status');
  const results = document.getElementById('results');
  uploadBtn.disabled = true;
  status.innerHTML = '<span class="spinner"></span>處理中，請稍候…';
  results.innerHTML = '';

  const formData = new FormData();
  selectedFiles.forEach(f => formData.append('files', f));

  try {
    const res = await fetch('/process', { method: 'POST', body: formData });
    const json = await res.json();

    status.innerHTML = `完成！共處理 ${json.results.length} 個檔案，資料已寫入 Google Sheet ✅`;

    results.innerHTML = json.results.map(r => `
      <div class="result-card">
        <div class="result-header">
          <span class="result-filename">${r.file}</span>
          <span class="badge ${r.status === 'ok' ? 'badge-ok' : 'badge-error'}">${r.status === 'ok' ? '成功' : '失敗'}</span>
        </div>
        ${r.status === 'ok' ? `
          <table>
            <tr><th>機台</th><td>${r.data.機台 || '—'}</td></tr>
            <tr><th>左右邊</th><td>${r.data.左右邊 || '—'}</td></tr>
            <tr><th>規格</th><td>${r.data.規格 || '—'}</td></tr>
            <tr><th>量測數據</th><td>${r.data.量測數據 || '—'}</td></tr>
            <tr><th>原始文字</th><td>${r.data.原始文字 || '—'}</td></tr>
          </table>
          <div class="sheet-badge">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
            已寫入 Google Sheet
          </div>
        ` : `<p style="color:#dc2626;font-size:0.88rem">${r.message}</p>`}
      </div>
    `).join('');
  } catch (err) {
    status.innerHTML = '上傳失敗，請稍後再試';
  }

  uploadBtn.disabled = false;
}
</script>
</body>
</html>"""


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
                    prompt=WHISPER_PROMPT
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
