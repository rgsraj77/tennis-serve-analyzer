"""Serve metrics measured at the detected phase frames.

All measurements come from a single side-on 2D view, so angles are
projections onto the camera plane. The guideline thresholds below are
prototype heuristics based on typical adult body proportions and common
coaching cues — v2 should replace them with literature-cited ranges.
"""
import numpy as np

from .pose import LM


def joint_angle(a, b, c):
    """Interior angle at b (degrees) for 2D points a-b-c."""
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    if np.any(np.isnan(ba)) or np.any(np.isnan(bc)):
        return float("nan")
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-9:
        return float("nan")
    cos = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def hitting_arm(handedness):
    if handedness == "right":
        return LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST
    return LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST


def knee_angle(kp, i):
    """Most-bent knee (smaller interior angle) at frame i."""
    left = joint_angle(kp[i, LM.LEFT_HIP, :2], kp[i, LM.LEFT_KNEE, :2],
                       kp[i, LM.LEFT_ANKLE, :2])
    right = joint_angle(kp[i, LM.RIGHT_HIP, :2], kp[i, LM.RIGHT_KNEE, :2],
                        kp[i, LM.RIGHT_ANKLE, :2])
    return np.nanmin([left, right])


def compute_metrics(kp, phases, handedness, fps):
    """Return (metrics list for the report, per-frame series for charts)."""
    n = len(kp)
    shoulder, elbow, wrist = hitting_arm(handedness)
    toss_wrist = LM.LEFT_WRIST if handedness == "right" else LM.RIGHT_WRIST
    contact, trophy = phases["contact"], phases["trophy"]

    elbow_at_contact = joint_angle(kp[contact, shoulder, :2],
                                   kp[contact, elbow, :2],
                                   kp[contact, wrist, :2])
    knee_at_trophy = knee_angle(kp, trophy)

    # Scale reference: no absolute size from one camera, so heights are
    # expressed relative to the player's nose-to-ankle length when standing.
    ankle_y = np.nanmean(kp[:, [LM.LEFT_ANKLE, LM.RIGHT_ANKLE], 1], axis=1)
    ground_y = np.nanpercentile(ankle_y, 95)
    body_len = np.nanpercentile(ankle_y - kp[:, LM.NOSE, 1], 90)
    if not body_len or body_len <= 0:
        body_len = float("nan")
    contact_height = (ground_y - kp[contact, wrist, 1]) / body_len
    tempo = (contact - phases["toss_peak"]) / fps

    def status(value, good, ok):
        if np.isnan(value):
            return "info"
        if good(value):
            return "good"
        return "info" if ok(value) else "warn"

    metrics = [
        {
            "label": "Elbow extension at contact",
            "display": "n/a" if np.isnan(elbow_at_contact) else f"{elbow_at_contact:.0f}°",
            "status": status(elbow_at_contact, lambda v: v >= 150, lambda v: v >= 125),
            "guideline": "≥150° — arm close to fully extended",
            "note": ("Good reach — the arm is nearly straight at contact."
                     if elbow_at_contact >= 150 else
                     "The arm looks bent at contact. Contacting with a straighter "
                     "arm raises your contact point, giving more net clearance "
                     "and a bigger service box window."),
        },
        {
            "label": "Knee bend at trophy position",
            "display": "n/a" if np.isnan(knee_at_trophy) else f"{knee_at_trophy:.0f}°",
            "status": status(knee_at_trophy, lambda v: v <= 125, lambda v: v <= 150),
            "guideline": "≤125° interior angle — real leg drive",
            "note": ("Solid knee bend — legs are loaded to drive upward."
                     if knee_at_trophy <= 125 else
                     "Knees stay fairly straight at the trophy position. More "
                     "knee bend stores energy for upward drive and easier power."),
        },
        {
            "label": "Contact height",
            "display": "n/a" if np.isnan(contact_height) else f"{contact_height:.2f}× body",
            "status": status(contact_height, lambda v: v >= 1.30, lambda v: v >= 1.15),
            "guideline": "≥1.30× nose-to-ankle length",
            "note": ("Contact point is nice and high — full upward extension."
                     if contact_height >= 1.30 else
                     "Contact happens relatively low. Reaching up at full stretch "
                     "(and tossing slightly higher/further in front) usually fixes this."),
        },
        {
            "label": "Tempo (toss peak → contact)",
            "display": "n/a" if np.isnan(tempo) else f"{tempo:.2f}s",
            "status": "info",
            "guideline": "no single ideal — consistency matters most",
            "note": "Track this across sessions: a steady tempo is a hallmark "
                    "of a repeatable serve.",
        },
    ]

    # Low pose confidence at the frames a metric was read from makes the
    # number unreliable — flag it rather than silently reporting it.
    key_vis = np.nanmean(kp[contact, [shoulder, elbow, wrist], 2])
    if key_vis < 0.5:
        metrics[0]["note"] += " (low tracking confidence on the hitting arm — treat as approximate)"

    series = {
        "t": (np.arange(n) / fps).tolist(),
        "hit_wrist": ((ground_y - kp[:, wrist, 1]) / body_len).tolist(),
        "toss_wrist": ((ground_y - kp[:, toss_wrist, 1]) / body_len).tolist(),
        "knee": [knee_angle(kp, i) for i in range(n)],
    }
    return metrics, series
