"""Video -> per-frame body keypoints via MediaPipe Pose."""
import re
import subprocess

import cv2
import numpy as np
import mediapipe as mp
import imageio_ffmpeg

mp_pose = mp.solutions.pose
LM = mp_pose.PoseLandmark
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

MAX_SIDE = 960     # frames are downscaled so the longest side is <= this
MAX_FRAMES = 1800  # ~60s at 30fps; keeps prototype memory and runtime bounded

# Phones (iPhones especially) store portrait clips as landscape pixels plus a
# rotation flag in the container. cv2.VideoCapture reads the raw landscape
# pixels and ignores the flag, so the whole analysis runs sideways -- wrong
# handedness, nonsense angles. We read the flag ourselves and rotate every
# frame upright, deterministically, so behaviour is identical on every
# platform (see open_video / apply_rotation).
_ROTATE_CODE = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _ffmpeg_rotation(video_path):
    """Rotation (deg clockwise to upright) parsed from container metadata.

    Fallback for when OpenCV does not surface CAP_PROP_ORIENTATION_META.
    """
    try:
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        err = subprocess.run([ff, "-i", str(video_path)],
                             capture_output=True, text=True).stderr
    except Exception:
        return 0
    # Newer ffmpeg: "displaymatrix: rotation of -90.00 degrees".
    m = re.search(r"rotation of ([-\d.]+) degrees", err)
    if m:
        return int(round(-float(m.group(1)) / 90.0)) % 4 * 90
    # Older ffmpeg: "rotate          : 90".
    m = re.search(r"rotate\s*:\s*([-\d]+)", err)
    if m:
        return int(m.group(1)) % 360
    return 0


def read_rotation(cap, video_path):
    """Degrees (0/90/180/270) to rotate raw frames to upright."""
    meta = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    if meta is not None:
        try:
            v = cap.get(meta)
            if v and not np.isnan(v):
                return int(round(v)) % 360
        except Exception:
            pass
    return _ffmpeg_rotation(video_path) % 360


def apply_rotation(frame, rotation):
    code = _ROTATE_CODE.get(rotation % 360)
    return cv2.rotate(frame, code) if code is not None else frame


def open_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    # Force auto-rotation OFF so raw frames are deterministic across OpenCV
    # builds; we then apply read_rotation ourselves. Without this, some
    # platforms auto-rotate and some don't, and we would double- or
    # zero-rotate depending on where the code runs.
    auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if auto is not None:
        try:
            cap.set(auto, 0)
        except Exception:
            pass
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 480:
        fps = 30.0
    rotation = read_rotation(cap, video_path)
    return cap, fps, rotation


def scale_frame(frame):
    h, w = frame.shape[:2]
    if max(h, w) <= MAX_SIDE:
        return frame
    s = MAX_SIDE / max(h, w)
    return cv2.resize(frame, (round(w * s), round(h * s)))


def extract_keypoints(video_path):
    """Return (keypoints, fps, (width, height)).

    keypoints: float array (n_frames, 33, 3) of x-pixels, y-pixels,
    visibility; NaN on frames where no person was detected.
    """
    cap, fps, rotation = open_video(video_path)
    rows = []
    size = None
    with mp_pose.Pose(model_complexity=1) as pose:
        while len(rows) < MAX_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            frame = apply_rotation(frame, rotation)
            frame = scale_frame(frame)
            h, w = frame.shape[:2]
            size = (w, h)
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks:
                rows.append([[p.x * w, p.y * h, p.visibility]
                             for p in result.pose_landmarks.landmark])
            else:
                rows.append(np.full((33, 3), np.nan).tolist())
    cap.release()
    if not rows:
        raise ValueError("Video contains no readable frames.")
    kp = np.asarray(rows, dtype=float)
    detected = ~np.isnan(kp[:, 0, 0])
    if detected.mean() < 0.3:
        raise ValueError(
            "Could not track a player in most of the video. Make sure the "
            "full body is visible and well lit, filmed side-on from a tripod."
        )
    return kp, fps, size
