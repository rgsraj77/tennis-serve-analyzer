"""End-to-end pipeline: serve video file -> analysis report directory."""
from pathlib import Path

from .pose import extract_keypoints
from .phases import smooth_keypoints, detect_phases
from .metrics import compute_metrics
from .report import render_assets, write_report


def analyze(video_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kp_raw, fps, _size = extract_keypoints(video_path)
    kp = smooth_keypoints(kp_raw, fps)
    phases, handedness, warnings = detect_phases(kp, fps)
    metrics, series = compute_metrics(kp, phases, handedness, fps)
    assets = render_assets(video_path, kp, phases, handedness, out_dir, fps)
    return write_report(out_dir, metrics, phases, handedness, warnings, fps,
                        assets, series)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m analyzer.pipeline <video> [out_dir]")
    out = sys.argv[2] if len(sys.argv) > 2 else "outputs/cli"
    print(analyze(sys.argv[1], out))
