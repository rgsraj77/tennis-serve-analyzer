"""Serve phase detection from smoothed keypoint trajectories.

Rule-based on wrist and knee kinematics — no trained model, so every
detected event traces back to a specific, inspectable signal.
"""
import numpy as np
from scipy.signal import savgol_filter

from .pose import LM
from .metrics import knee_angle


def smooth_keypoints(kp, fps):
    """Interpolate missing frames, then Savitzky-Golay smooth x/y.

    Raw MediaPipe keypoints jitter frame to frame; joint angles computed
    on them are unusable without this pass.
    """
    kp = kp.copy()
    n = len(kp)
    t = np.arange(n)
    for j in range(kp.shape[1]):
        for c in range(3):
            v = kp[:, j, c]
            valid = ~np.isnan(v)
            if 0 < valid.sum() < n:
                kp[:, j, c] = np.interp(t, t[valid], v[valid])
    window = int(round(fps * 0.15))
    if window % 2 == 0:
        window += 1
    window = max(5, window)
    if n > window:
        kp[:, :, :2] = savgol_filter(kp[:, :, :2], window, 2, axis=0)
    return kp


def detect_phases(kp, fps):
    """Return (phase frame indices, handedness, warnings).

    Image y grows downward, so a wrist at its highest point is at min y.
    """
    warnings = []
    n = len(kp)
    right_y = kp[:, LM.RIGHT_WRIST, 1]
    left_y = kp[:, LM.LEFT_WRIST, 1]

    # The hitting wrist is the one that reaches highest anywhere in the clip.
    handedness = "right" if np.nanmin(right_y) < np.nanmin(left_y) else "left"
    hit_y = right_y if handedness == "right" else left_y
    toss_y = left_y if handedness == "right" else right_y

    contact = int(np.nanargmin(hit_y))
    if contact < fps * 0.3:
        warnings.append("Contact was detected almost immediately — start the "
                        "recording a second or two before the serve begins.")

    # Ball toss peak: tossing arm at its highest point before contact.
    toss_peak = int(np.nanargmin(toss_y[:contact])) if contact > 0 else 0

    # Trophy position: deepest knee bend between toss peak and contact.
    lo, hi = toss_peak, max(contact, toss_peak + 1)
    knees = np.array([knee_angle(kp, i) for i in range(lo, hi)])
    if len(knees) and not np.isnan(knees).all():
        trophy = lo + int(np.nanargmin(knees))
    else:
        trophy = toss_peak

    follow_through = min(n - 1, contact + int(0.3 * fps))

    if fps < 50:
        warnings.append(f"Footage is ~{fps:.0f}fps, so the exact contact frame "
                        "may be off by 1-2 frames. 60fps or slow-mo video "
                        "improves precision.")

    phases = {"toss_peak": toss_peak, "trophy": trophy,
              "contact": contact, "follow_through": follow_through}
    return phases, handedness, warnings
