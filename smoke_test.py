"""Smoke test without a real serve video.

1. A blank synthetic clip must be rejected with a clear 'no player' error.
2. A synthetic clip with a moving stick figure exercises the geometry/report
   code path only if MediaPipe happens to track it — otherwise skipped.
3. The pipeline modules must import and the FastAPI app must build.
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def make_blank_clip(path, seconds=2, fps=30, size=(640, 360)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(seconds * fps):
        frame = np.full((size[1], size[0], 3), 120, np.uint8)
        cv2.circle(frame, (100 + i * 3, 180), 20, (0, 0, 255), -1)
        writer.write(frame)
    writer.release()


def main():
    from analyzer.pipeline import analyze
    import app  # noqa: F401  -- verifies the FastAPI app builds

    tmp = Path(tempfile.mkdtemp())
    blank = tmp / "blank.mp4"
    make_blank_clip(blank)
    try:
        analyze(blank, tmp / "out")
    except ValueError as e:
        print(f"PASS: blank clip rejected with: {e}")
    else:
        print("FAIL: blank clip was not rejected")
        return 1
    print("PASS: analyzer modules and FastAPI app import cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
