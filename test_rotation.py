"""Verify rotation handling normalises phone clips to upright.

The bug: cv2.VideoCapture reads a phone clip's raw landscape pixels and
ignores the display-matrix rotation flag, so the analysis runs sideways.

This tests the rotation fix's exact contract, deterministically and without
depending on MediaPipe: an iPhone-style clip (sideways pixels + display-matrix
flag), once read through open_video + apply_rotation, must yield frames that
match the upright original. A correct rotation gives a per-frame mean pixel
diff of ~2-3 (compression noise); a wrong one gives ~60+, so the threshold
cleanly separates them.

(The pipeline's *handedness* is separately, and known to be, sensitive to
sub-perceptual encoding noise -- that is the next fix, not this one, so it is
deliberately not asserted here.)
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import imageio_ffmpeg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyzer.pose import apply_rotation, read_rotation, open_video

UPRIGHT = Path(r"C:\Users\rajgu\Downloads\WhatsApp Video 2026-07-28 at 5.15.22 PM.mp4")
FF = imageio_ffmpeg.get_ffmpeg_exe()
MATCH_THRESHOLD = 10.0  # mean abs pixel diff; correct ~2-3, wrong ~60+


def synth_upright(path, n=100, fps=30, w=360, h=640):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        f = np.full((h, w, 3), 210, np.uint8)
        cv2.circle(f, (w // 2, 120), 26, (60, 60, 60), -1)
        cv2.line(f, (w // 2, 146), (w // 2, 380), (60, 60, 60), 12)
        cv2.line(f, (w // 2, 210), (w // 2 + 90, 200), (60, 60, 60), 10)
        cv2.rectangle(f, (30, 30), (90, 90), (0, 0, 200), -1)  # asymmetry marker
        writer.write(f)
    writer.release()


def frames_via_pipeline(path):
    """Frames exactly as the analyser sees them: open_video + apply_rotation."""
    cap, _, rotation = open_video(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(apply_rotation(f, rotation))
    cap.release()
    return out, rotation


def build_iphone_like(upright, dst, vf, display_rot):
    side = dst.with_name(dst.stem + "_side.mp4")
    subprocess.run([FF, "-y", "-i", str(upright), "-vf", vf, "-c:v", "libx264",
                    "-preset", "ultrafast", "-an", str(side)],
                   check=True, capture_output=True)
    subprocess.run([FF, "-y", "-display_rotation:v:0", str(display_rot),
                    "-i", str(side), "-c", "copy", str(dst)],
                   check=True, capture_output=True)


def main():
    tmp = Path(tempfile.mkdtemp())
    failures = []

    # 1. apply_rotation math.
    img = np.zeros((4, 6, 3), np.uint8)
    img[0, 0] = 255
    checks = [
        (apply_rotation(img, 90).shape == (6, 4, 3), "90 shape"),
        (apply_rotation(img, 90)[0, -1, 0] == 255, "90 corner"),
        (apply_rotation(img, 180)[-1, -1, 0] == 255, "180 corner"),
        (apply_rotation(img, 270)[-1, 0, 0] == 255, "270 corner"),
        (np.array_equal(apply_rotation(img, 0), img), "0 identity"),
    ]
    for ok, label in checks:
        if not ok:
            failures.append(f"apply_rotation {label} wrong")
    print("1. apply_rotation math:", "OK" if all(c[0] for c in checks) else "FAIL")

    upright = UPRIGHT if UPRIGHT.exists() else (tmp / "upright.mp4")
    if not UPRIGHT.exists():
        synth_upright(upright)
        print("   using synthetic upright clip")
    else:
        print(f"   using real clip: {UPRIGHT.name}")

    up_frames, up_rot = frames_via_pipeline(upright)
    print(f"2. upright: {len(up_frames)} frames, rotation={up_rot}")
    if up_rot != 0:
        failures.append(f"upright clip reported rotation {up_rot}, expected 0")

    # display_rotation needs ffmpeg >= 5.1; skip end-to-end gracefully if absent.
    try:
        build_iphone_like(upright, tmp / "probe.mp4", "transpose=1", 90)
    except subprocess.CalledProcessError:
        print("   (ffmpeg lacks -display_rotation; math-only)")
        return _done(failures)

    # 3. Each iPhone-style clip must be detected and restored to upright pixels.
    cases = [
        ("90cw", "transpose=1", 90),
        ("90ccw", "transpose=2", -90),
        ("180", "transpose=1,transpose=1", 180),
    ]
    for name, vf, disp in cases:
        clip = tmp / f"iphone_{name}.mp4"
        build_iphone_like(upright, clip, vf, disp)

        cap = cv2.VideoCapture(str(clip))
        detected = read_rotation(cap, clip)
        cap.release()

        rot_frames, applied = frames_via_pipeline(clip)
        n = min(len(up_frames), len(rot_frames))
        # sample frames across the clip
        idx = range(0, n, max(1, n // 10))
        diffs = [float(np.mean(np.abs(up_frames[i].astype(int) - rot_frames[i].astype(int))))
                 for i in idx]
        worst = max(diffs)
        same_shape = rot_frames[0].shape == up_frames[0].shape
        ok = detected != 0 and same_shape and worst < MATCH_THRESHOLD
        print(f"3.{name}: detected={detected} applied={applied} "
              f"shape={rot_frames[0].shape[:2]} worst_diff={worst:.1f} "
              f"[{'OK' if ok else 'FAIL'}]")
        if detected == 0:
            failures.append(f"{name}: flag not detected")
        if not same_shape:
            failures.append(f"{name}: restored shape {rot_frames[0].shape[:2]} != upright")
        if worst >= MATCH_THRESHOLD:
            failures.append(f"{name}: frames not restored to upright (diff {worst:.1f})")

    return _done(failures)


def _done(failures):
    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print("\nPASS: phone-rotation flags are detected and frames restored upright")
    return 0


if __name__ == "__main__":
    sys.exit(main())
