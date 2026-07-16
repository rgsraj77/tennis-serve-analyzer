# Serve Coach

Prototype tennis serve analyser: upload a side-on video of one serve, get back
a report with detected serve phases, joint-angle metrics against guideline
ranges, an annotated skeleton-overlay video, and a motion timeline chart.

No training data required by design: pose estimation uses MediaPipe's
pretrained BlazePose model, serve phases are detected with rule-based
kinematics (inspectable, debuggable), and reference ranges come from
published serve biomechanics.

## Reference ranges

Metric thresholds are not invented — they are mean ±1 SD from published
kinematics, cited inline in every report. See `analyzer/references.py` for
sources, caveats, and the flexion→interior angle conversion.

| Metric | Reference (interior angle) | Source |
|---|---|---|
| Elbow at contact | 134–166° (elite mean 150°) | [Wang et al. 2024 meta-analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC11260724/) |
| Knee at trophy | 106–125° (elite mean 116°) | [Wang et al. 2024 meta-analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC11260724/) |
| Contact height | ≥1.30× nose-to-ankle | uncalibrated heuristic — no published equivalent |

Two caveats the reports state explicitly: these populations are elite
servers (a benchmark, not a target), and the source studies used 3D motion
capture while this tool measures 2D camera-plane projections.

## Pipeline

video → MediaPipe pose keypoints → interpolation + Savitzky-Golay smoothing →
phase detection (toss peak, trophy, contact, follow-through) → metrics
(elbow extension at contact, knee bend at trophy, contact height, tempo) →
report (annotated video, stills, chart, HTML)

## Run

Python **3.9–3.12** (see the deploy notes — 3.13+ has no mediapipe wheel).

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt -r requirements-dev.txt
```

Two frontends share the same analysis core (`analyzer/pipeline.py`):

```
.venv\Scripts\python -m streamlit run streamlit_app.py   # http://localhost:8501
.venv\Scripts\python -m uvicorn app:app --port 8123      # http://localhost:8123
```

Or headless:

```
.venv\Scripts\python -m analyzer.pipeline path\to\serve.mp4 outputs\test
```

## Deploying

Target: **Streamlit Community Cloud** (free, needs a public GitHub repo).

1. Push this repo to GitHub (public — the free tier requires it).
2. [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the repo.
3. Main file: `streamlit_app.py`.
4. **Advanced settings → Python version → 3.12.** ← the one that bites.
5. Deploy. First build takes ~5-10 min.

### Why 3.12 specifically

mediapipe 0.10.14 publishes Linux wheels for **cp39–cp312 only** and declares
no `requires_python`, so on 3.13+ pip does not refuse — it attempts a source
build and fails with an opaque error. Community Cloud defaults to 3.12 (fine)
but offers 3.13/3.14 in the dropdown, and the version **cannot be pinned from
the repo** — no `runtime.txt`/`.python-version` support. `streamlit_app.py`
therefore checks at runtime and shows a plain-English error instead of a
traceback.

### Other deployment facts

- Peak RAM is **~440MB** (measured — `measure_memory.py`, 1080x1920 @60fps
  clip, including the ffmpeg subprocess) against Community Cloud's 1GB cap.
  Render's 512MB free tier is too tight; Hugging Face Spaces requires PRO for
  CPU Basic/Docker as of July 2026.
- `packages.txt` installs `libgl1` + `libglib2.0-0`. MediaPipe pulls
  `opencv-contrib-python`, which needs these at runtime — without them the
  container dies at `import cv2`.
- Only **one** dependency file is read, and `requirements.txt` is it — hence
  the dev/deploy split. Do not add a `pyproject.toml` or `Pipfile` without
  checking Streamlit's precedence order.
- `.streamlit/config.toml` caps uploads at 60MB, matching `MAX_UPLOAD_MB`, so
  the browser enforces it before the upload rather than after.
- Free-tier CPU is slower than local: expect 1-3 min per clip, not 30-90s.
- Storage is ephemeral; reports vanish on restart. Uploaded videos are
  processed on Streamlit's servers.

## Recording protocol

- Side-on from a tripod, on the side of the hitting arm, full body in frame
- One serve per clip, ~2s padding either side
- 60fps or slow-mo if available (30fps works, contact frame is less precise)

## Known limitations (prototype)

- Single-camera 2D: angles are camera-plane projections; rotational metrics
  (hip-shoulder separation) are out of scope
- No ball or racket tracking — toss is inferred from the tossing wrist
- Reference ranges come from elite populations measured with 3D capture;
  the 2D-vs-3D gap is unquantified
- Contact height threshold is still an uncalibrated heuristic
- Not yet validated against a real serve video
- One serve per clip; analysis is synchronous in the request
