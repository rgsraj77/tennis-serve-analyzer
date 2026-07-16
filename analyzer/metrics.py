"""Serve metrics measured at the detected phase frames.

All measurements come from a single side-on 2D view, so angles are
projections onto the camera plane. Reference bands come from published serve
kinematics — see references.py for sources, the flexion/interior angle
conversion, and the caveats that apply to every comparison here.
"""
import numpy as np

from .pose import LM
from .references import (KNEE_TROPHY, ELBOW_CONTACT, CONTACT_HEIGHT_HEURISTIC,
                         SOURCES)


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

    def band_status(value, ref, direction):
        """Compare against the reference +/-1 SD band.

        'inside' = within the band; 'near' = outside by under 1 more SD;
        'outside' = further than that. direction says which side of the band
        counts as unremarkable rather than notable: for the knee, MORE bend
        than elite is not a fault, and likewise for a straighter elbow.
        """
        if np.isnan(value):
            return "info"
        lo, hi, sd = ref["interior_lo"], ref["interior_hi"], ref["flexion_sd"]
        if lo <= value <= hi:
            return "good"
        if direction == "lower_ok" and value < lo:
            return "good"
        if direction == "higher_ok" and value > hi:
            return "good"
        gap = (lo - value) if value < lo else (value - hi)
        return "info" if gap <= sd else "warn"

    elbow_ref, knee_ref = ELBOW_CONTACT, KNEE_TROPHY
    elbow_status = band_status(elbow_at_contact, elbow_ref, "higher_ok")
    knee_status = band_status(knee_at_trophy, knee_ref, "lower_ok")

    metrics = [
        {
            "label": "Elbow extension at contact",
            "display": "n/a" if np.isnan(elbow_at_contact) else f"{elbow_at_contact:.0f}°",
            "status": elbow_status,
            "guideline": f"reference {elbow_ref['interior_lo']:.0f}–"
                         f"{elbow_ref['interior_hi']:.0f}° (elite mean "
                         f"{elbow_ref['interior_mean']:.0f}°)",
            "source": SOURCES[elbow_ref["source"]],
            "note": ("Arm extension at contact sits in the reference range for "
                     "skilled servers."
                     if elbow_status == "good" else
                     "The arm looks more bent at contact than the reference range. "
                     "A straighter arm raises the contact point, giving more net "
                     "clearance and a bigger service box window."),
        },
        {
            "label": "Knee bend at trophy position",
            "display": "n/a" if np.isnan(knee_at_trophy) else f"{knee_at_trophy:.0f}°",
            "status": knee_status,
            "guideline": f"reference {knee_ref['interior_lo']:.0f}–"
                         f"{knee_ref['interior_hi']:.0f}° (elite mean "
                         f"{knee_ref['interior_mean']:.0f}°)",
            "source": SOURCES[knee_ref["source"]],
            "note": ("Knee bend is in the reference range — legs loaded to drive "
                     "upward."
                     if knee_status == "good" else
                     "Knees stay straighter at the trophy position than the "
                     "reference range. More knee bend stores energy for upward "
                     "drive and easier power."),
        },
        {
            "label": "Contact height",
            "display": "n/a" if np.isnan(contact_height) else f"{contact_height:.2f}× body",
            "status": "info",
            "guideline": f"≥{CONTACT_HEIGHT_HEURISTIC['good_min']:.2f}× nose-to-ankle "
                         "length (uncalibrated heuristic — no published equivalent)",
            "source": None,
            "note": ("Contact point is high — full upward extension."
                     if contact_height >= CONTACT_HEIGHT_HEURISTIC["good_min"] else
                     "Contact looks relatively low. Reaching up at full stretch "
                     "(and tossing slightly higher and further in front) usually "
                     "raises it. This threshold is a rough heuristic, so weigh it "
                     "lightly."),
        },
        {
            "label": "Tempo (toss peak → contact)",
            "display": "n/a" if np.isnan(tempo) else f"{tempo:.2f}s",
            "status": "info",
            "guideline": "no single ideal — consistency across serves matters most",
            "source": None,
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
