"""End-to-end pipeline: serve video file -> analysis results / report.

run_analysis() returns structured results so any frontend can render them;
analyze() adds the standalone HTML report on top for the FastAPI path.
"""
from pathlib import Path

from .pose import extract_keypoints
from .phases import smooth_keypoints, detect_phases
from .metrics import compute_metrics
from .report import render_assets, write_report


def run_analysis(video_path, out_dir, progress=None):
    """Analyse a serve video; return results plus rendered asset filenames.

    progress: optional callable(fraction, label) for frontends that show a
    progress bar. Stage weights are rough -- pose extraction dominates.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def step(frac, label):
        if progress:
            progress(frac, label)

    step(0.05, "Reading video and finding the player")
    kp_raw, fps, _size = extract_keypoints(video_path)
    step(0.55, "Smoothing keypoint trajectories")
    kp = smooth_keypoints(kp_raw, fps)
    step(0.6, "Detecting serve phases")
    phases, handedness, warnings = detect_phases(kp, fps)
    step(0.65, "Measuring joint angles")
    metrics, series = compute_metrics(kp, phases, handedness, fps)
    step(0.7, "Rendering annotated video")
    assets = render_assets(video_path, kp, phases, handedness, out_dir, fps)
    step(0.95, "Building report")
    return {
        "out_dir": out_dir,
        "metrics": metrics,
        "series": series,
        "phases": phases,
        "handedness": handedness,
        "warnings": warnings,
        "fps": fps,
        "assets": assets,
    }


def analyze(video_path, out_dir, progress=None):
    """Analyse and write the standalone HTML report; return its path."""
    r = run_analysis(video_path, out_dir, progress)
    return write_report(r["out_dir"], r["metrics"], r["phases"], r["handedness"],
                        r["warnings"], r["fps"], r["assets"], r["series"])


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m analyzer.pipeline <video> [out_dir]")
    out = sys.argv[2] if len(sys.argv) > 2 else "outputs/cli"
    print(analyze(sys.argv[1], out))
