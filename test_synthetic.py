"""Exercise everything downstream of pose extraction with a synthetic serve.

Generates keypoint trajectories mimicking a right-handed serve (toss peak
~1.0s, trophy ~1.5s, contact ~2.0s) plus a matching blank video, then runs
phase detection, metrics, and full report rendering, asserting the detected
phases land near the scripted events.
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from analyzer.pose import LM
from analyzer.phases import smooth_keypoints, detect_phases
from analyzer.metrics import compute_metrics
from analyzer.report import render_assets, write_report

FPS = 30
N = 90  # 3 seconds
W, H = 640, 360


def bump(i, center, width, lo, hi):
    """Value moving from lo to hi and back, gaussian-shaped around center."""
    return lo + (hi - lo) * np.exp(-0.5 * ((i - center) / width) ** 2)


def synthetic_serve():
    kp = np.zeros((N, 33, 3))
    kp[:, :, 2] = 1.0  # full visibility
    for i in range(N):
        # Static body, side-on: ground y=340, nose y=100.
        kp[i, LM.NOSE] = [320, 100, 1]
        kp[i, LM.LEFT_SHOULDER] = [300, 160, 1]
        kp[i, LM.RIGHT_SHOULDER] = [330, 160, 1]
        kp[i, LM.LEFT_HIP] = [305, 230, 1]
        kp[i, LM.RIGHT_HIP] = [325, 230, 1]
        kp[i, LM.LEFT_ANKLE] = [305, 340, 1]
        kp[i, LM.RIGHT_ANKLE] = [325, 340, 1]
        # Knees bend (x pushes forward) peaking at the scripted trophy, i=45.
        knee_dx = bump(i, 45, 8, 0, 45)
        kp[i, LM.LEFT_KNEE] = [305 + knee_dx, 285, 1]
        kp[i, LM.RIGHT_KNEE] = [325 + knee_dx, 285, 1]
        # Tossing (left) wrist peaks at i=30.
        toss_y = bump(i, 30, 10, 250, 60)
        kp[i, LM.LEFT_WRIST] = [290, toss_y, 1]
        kp[i, LM.LEFT_ELBOW] = [295, (160 + toss_y) / 2, 1]
        # Hitting (right) wrist: racket drop at i=48, contact peak at i=60.
        hit_y = bump(i, 60, 6, 260, 20) + bump(i, 48, 5, 0, 40)
        kp[i, LM.RIGHT_WRIST] = [340, hit_y, 1]
        # Elbow halfway shoulder->wrist, slightly offset: ~straight arm.
        kp[i, LM.RIGHT_ELBOW] = [336, (160 + hit_y) / 2, 1]
    return kp


def blank_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for _ in range(N):
        writer.write(np.full((H, W, 3), 200, np.uint8))
    writer.release()


def main():
    tmp = Path(tempfile.mkdtemp())
    video = tmp / "synthetic.mp4"
    blank_video(video)

    kp = smooth_keypoints(synthetic_serve(), FPS)
    phases, handedness, warnings = detect_phases(kp, FPS)
    print(f"handedness={handedness} phases={phases} warnings={len(warnings)}")

    failures = []
    if handedness != "right":
        failures.append(f"handedness: expected right, got {handedness}")
    for name, expected in (("toss_peak", 30), ("trophy", 45), ("contact", 60)):
        if abs(phases[name] - expected) > 5:
            failures.append(f"{name}: expected ~{expected}, got {phases[name]}")

    metrics, series = compute_metrics(kp, phases, handedness, FPS)
    for m in metrics:
        print(f"  {m['label']}: {m['display']} [{m['status']}]")
        if m["display"] == "n/a":
            failures.append(f"metric came out n/a: {m['label']}")

    out = tmp / "report"
    out.mkdir()
    assets = render_assets(video, kp, phases, handedness, out, FPS)
    report = write_report(out, metrics, phases, handedness, warnings, FPS,
                          assets, series)
    for f in [report, out / assets["video"], out / "chart.png"]:
        if not Path(f).exists() or Path(f).stat().st_size == 0:
            failures.append(f"missing/empty asset: {f}")
    print(f"report assets: {sorted(p.name for p in out.iterdir())}")

    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print(f"\nPASS: full downstream pipeline OK — open {report} to inspect")
    return 0


if __name__ == "__main__":
    sys.exit(main())
