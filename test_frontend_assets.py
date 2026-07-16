"""run_analysis() must produce every file a frontend renders.

Regression test for a real bug: chart.png was generated inside write_report(),
but streamlit_app.py calls run_analysis() and never calls write_report -- so
the deployed app crashed with FileNotFoundError on chart.png after a
successful analysis. test_synthetic.py missed it because it calls
write_report() directly, which rendered the chart as a side effect.

This asserts the contract from the Streamlit frontend's point of view: every
path it touches in render_results() exists on disk after run_analysis alone.
"""
import sys
import tempfile
from pathlib import Path

from analyzer.pipeline import run_analysis
from analyzer import pipeline as pipeline_mod
import test_synthetic as ts


def main():
    tmp = Path(tempfile.mkdtemp())
    video = tmp / "synthetic.mp4"
    ts.blank_video(video)

    # MediaPipe finds no player in synthetic frames, so stub extraction with
    # scripted keypoints and let the rest of run_analysis run for real.
    kp = ts.synthetic_serve()
    kp[:, :, 0] *= 640 / 640.0
    kp[:, :, 1] *= 360 / 360.0
    pipeline_mod.extract_keypoints = lambda _p: (kp, float(ts.FPS), (640, 360))

    r = run_analysis(video, tmp / "out")

    failures = []
    # Exactly what streamlit_app.render_results() reads.
    needed = [r["out_dir"] / r["assets"]["video"],
              r["out_dir"] / r["assets"]["chart"]]
    needed += [r["out_dir"] / f for f in r["assets"]["stills"].values()]

    for p in needed:
        if not Path(p).exists():
            failures.append(f"missing: {p.name}")
        elif Path(p).stat().st_size == 0:
            failures.append(f"empty: {p.name}")
        else:
            print(f"  OK  {p.name} ({Path(p).stat().st_size:,} B)")

    for key in ("metrics", "series", "phases", "handedness", "fps", "assets"):
        if key not in r:
            failures.append(f"run_analysis result missing key: {key}")

    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print("\nPASS: run_analysis produces every asset the frontend renders")
    return 0


if __name__ == "__main__":
    sys.exit(main())
