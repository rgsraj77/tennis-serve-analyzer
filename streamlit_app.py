"""Serve Coach — Streamlit frontend (deploy target: Streamlit Community Cloud).

Shares the analysis core with app.py (FastAPI); only the rendering differs.
Measured peak ~440MB against Community Cloud's 1GB cap, so uploads are held
to short clips to keep that headroom.
"""
import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st

# Guard BEFORE importing the analyzer: mediapipe 0.10.14 publishes Linux
# wheels only for cp39-cp312 and declares no requires_python, so on 3.13+ pip
# silently tries a source build and the analyzer import dies with an opaque
# error. Community Cloud offers 3.13/3.14 in its deploy dropdown and cannot
# pin the version from the repo, so this failure is one wrong click away.
MIN_PY, MAX_PY = (3, 9), (3, 12)
if not (MIN_PY <= sys.version_info[:2] <= MAX_PY):
    st.error(
        f"This app needs Python {MIN_PY[0]}.{MIN_PY[1]}–{MAX_PY[0]}.{MAX_PY[1]}, "
        f"but it is running on {sys.version_info.major}.{sys.version_info.minor}. "
        "MediaPipe publishes no wheel for this version. On Streamlit Community "
        "Cloud: Manage app → Settings → Python version → 3.12, then reboot."
    )
    st.stop()

from analyzer.pipeline import run_analysis  # noqa: E402
from analyzer.report import PHASE_LABELS  # noqa: E402

MAX_UPLOAD_MB = 60  # Community Cloud caps RAM; keep clips short

st.set_page_config(page_title="Serve Coach", page_icon="🎾", layout="centered")

STATUS_EMOJI = {"good": "✅", "info": "ℹ️", "warn": "⚠️"}


def sidebar():
    with st.sidebar:
        st.header("How to film")
        st.markdown(
            "- **Side-on** from a tripod, on the side of your hitting arm\n"
            "- Full body in frame, well lit\n"
            "- **One serve per clip**, ~2s either side\n"
            "- 60fps or slow-mo if your phone offers it"
        )
        st.header("What this is")
        st.markdown(
            "A pretrained pose estimator (MediaPipe BlazePose) plus a "
            "rule-based kinematics layer. No model was trained; serve phases "
            "are found from wrist and knee signals, and reference ranges come "
            "from published biomechanics."
        )
        st.caption(
            "Reference ranges are mean ±1 SD for **elite** servers measured "
            "with 3D motion capture. This tool measures 2D angles from one "
            "camera, so treat numbers as approximations and the ranges as a "
            "benchmark, not a target. Not coaching advice."
        )


def render_results(r):
    st.subheader("Metrics")
    cols = st.columns(2)
    for i, m in enumerate(r["metrics"]):
        with cols[i % 2]:
            with st.container(border=True):
                st.metric(m["label"], m["display"])
                st.markdown(f"{STATUS_EMOJI[m['status']]} {m['note']}")
                st.caption(m["guideline"])
                if m.get("source"):
                    st.caption(f"[source: {m['source']['cite']}]({m['source']['url']})")

    st.subheader("Key positions")
    stills = [n for n in ("toss_peak", "trophy", "contact") if n in r["assets"]["stills"]]
    if stills:
        for col, name in zip(st.columns(len(stills)), stills):
            with col:
                st.image(str(r["out_dir"] / r["assets"]["stills"][name]),
                         caption=f"{PHASE_LABELS[name].title()} — "
                                 f"{r['phases'][name] / r['fps']:.2f}s",
                         use_container_width=True)

    st.subheader("Annotated video")
    st.video(str(r["out_dir"] / r["assets"]["video"]))

    st.subheader("Motion timeline")
    st.image(str(r["out_dir"] / "chart.png"), use_container_width=True)


def main():
    sidebar()
    st.title("🎾 Serve Coach")
    st.write("Upload a video of one tennis serve to get a biomechanics report: "
             "detected serve phases, joint-angle metrics, and an annotated video.")

    upload = st.file_uploader("Serve video", type=["mp4", "mov", "avi", "mkv"],
                              label_visibility="collapsed")
    if not upload:
        st.info("Film side-on, one serve per clip. See the sidebar for tips.")
        return

    size_mb = upload.size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        st.error(f"That clip is {size_mb:.0f}MB — over the {MAX_UPLOAD_MB}MB limit "
                 "for the hosted demo. Trim it to just the serve and retry.")
        return

    if not st.button("Analyse my serve", type="primary"):
        return

    work = Path(tempfile.gettempdir()) / f"serve_{uuid.uuid4().hex[:10]}"
    work.mkdir(parents=True, exist_ok=True)
    clip = work / (upload.name or "clip.mp4")
    clip.write_bytes(upload.getbuffer())

    bar = st.progress(0.0, "Starting")
    try:
        with st.spinner("Analysing — 1-3 minutes on the free tier."):
            r = run_analysis(clip, work / "out",
                             progress=lambda f, label: bar.progress(f, label))
    except ValueError as e:
        bar.empty()
        st.error(str(e))
        return
    except Exception as e:
        bar.empty()
        st.error(f"Analysis failed unexpectedly: {type(e).__name__}: {e}")
        return
    bar.progress(1.0, "Done")
    bar.empty()

    st.success(f"Detected a **{r['handedness']}-handed** serve · "
               f"footage {r['fps']:.0f}fps · "
               f"contact at {r['phases']['contact'] / r['fps']:.2f}s")
    for w in r["warnings"]:
        st.warning(w)
    render_results(r)


if __name__ == "__main__":
    main()
