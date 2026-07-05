# Serve Coach

Prototype tennis serve analyser: upload a side-on video of one serve, get back
a report with detected serve phases, joint-angle metrics against guideline
ranges, an annotated skeleton-overlay video, and a motion timeline chart.

No training data required by design: pose estimation uses MediaPipe's
pretrained model, serve phases are detected with rule-based kinematics
(inspectable, debuggable), and guideline ranges are configurable constants.

## Pipeline

video → MediaPipe pose keypoints → interpolation + Savitzky-Golay smoothing →
phase detection (toss peak, trophy, contact, follow-through) → metrics
(elbow extension at contact, knee bend at trophy, contact height, tempo) →
report (annotated video, stills, chart, HTML)

## Run

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\uvicorn app:app --port 8123
```

Open http://localhost:8123 and upload a clip. Or run headless:

```
.venv\Scripts\python -m analyzer.pipeline path\to\serve.mp4 outputs\test
```

## Recording protocol

- Side-on from a tripod, on the side of the hitting arm, full body in frame
- One serve per clip, ~2s padding either side
- 60fps or slow-mo if available (30fps works, contact frame is less precise)

## Known limitations (prototype)

- Single-camera 2D: angles are camera-plane projections; rotational metrics
  (hip-shoulder separation) are out of scope
- No ball or racket tracking — toss is inferred from the tossing wrist
- Guideline thresholds are heuristics, not literature-cited ranges yet
- One serve per clip; analysis is synchronous in the request
