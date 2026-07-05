"""Serve Coach — upload a tennis serve video, get an analysis report."""
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from analyzer.pipeline import analyze

BASE = Path(__file__).resolve().parent
UPLOAD_DIR = BASE / "uploads"
OUTPUT_DIR = BASE / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_MB = 200

app = FastAPI(title="Serve Coach")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

PAGE_CSS = """
body { font-family: system-ui, sans-serif; background: #f4f4f0; color: #222;
       display: flex; justify-content: center; padding: 48px 16px; }
.card { background: #fff; border-radius: 14px; padding: 32px; max-width: 520px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07); }
h1 { margin-top: 0; font-size: 1.5rem; }
p, li { font-size: 0.95rem; line-height: 1.45; }
ul { padding-left: 20px; color: #555; }
input[type=file] { margin: 14px 0; width: 100%; }
button { background: #0F6E56; color: #fff; border: 0; border-radius: 8px;
         padding: 12px 22px; font-size: 1rem; cursor: pointer; }
button:disabled { background: #999; cursor: wait; }
.err { background: #FCEBEB; border: 1px solid #E24B4A; border-radius: 8px;
       padding: 10px 14px; color: #791F1F; }
"""

INDEX = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Serve Coach</title><style>{PAGE_CSS}</style></head>
<body><div class="card">
<h1>🎾 Serve Coach</h1>
<p>Upload a video of one tennis serve and get an analysis report:
detected serve phases, joint-angle metrics, and an annotated video.</p>
<ul>
  <li>Film <b>side-on</b> from a tripod, full body in frame</li>
  <li>One serve per clip, a couple of seconds either side</li>
  <li>60fps if your phone supports it</li>
</ul>
<form id="f" action="/analyze" method="post" enctype="multipart/form-data">
  <input type="file" name="video" accept="video/*" required>
  <button id="b" type="submit">Analyse my serve</button>
</form>
<p id="status" style="display:none;color:#555;">Analysing&hellip; this takes
about 30&ndash;90 seconds. Don't close the tab.</p>
<script>
document.getElementById('f').addEventListener('submit', () => {{
  document.getElementById('b').disabled = true;
  document.getElementById('status').style.display = 'block';
}});
</script>
</div></body></html>"""


def error_page(message):
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Serve Coach</title>
<style>{PAGE_CSS}</style></head>
<body><div class="card"><h1>Couldn't analyse that video</h1>
<p class="err">{message}</p>
<p><a href="/">&larr; try another clip</a></p></div></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX


@app.post("/analyze")
def analyze_upload(video: UploadFile = File(...)):
    job = uuid.uuid4().hex[:12]
    suffix = Path(video.filename or "clip.mp4").suffix.lower() or ".mp4"
    dest = UPLOAD_DIR / f"{job}{suffix}"
    size = 0
    with dest.open("wb") as f:
        while chunk := video.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                f.close()
                dest.unlink(missing_ok=True)
                return HTMLResponse(
                    error_page(f"File is over {MAX_UPLOAD_MB}MB. Trim the clip "
                               "to just the serve and try again."),
                    status_code=413)
            f.write(chunk)
    try:
        analyze(dest, OUTPUT_DIR / job)
    except ValueError as e:
        return HTMLResponse(error_page(str(e)), status_code=422)
    return RedirectResponse(f"/outputs/{job}/report.html", status_code=303)
