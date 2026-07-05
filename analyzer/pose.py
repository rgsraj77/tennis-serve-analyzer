"""Video -> per-frame body keypoints via MediaPipe Pose."""
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
LM = mp_pose.PoseLandmark
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

MAX_SIDE = 960     # frames are downscaled so the longest side is <= this
MAX_FRAMES = 1800  # ~60s at 30fps; keeps prototype memory and runtime bounded


def open_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 480:
        fps = 30.0
    return cap, fps


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
    cap, fps = open_video(video_path)
    rows = []
    size = None
    with mp_pose.Pose(model_complexity=1) as pose:
        while len(rows) < MAX_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
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
