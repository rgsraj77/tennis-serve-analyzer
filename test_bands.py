"""Check reference bands classify realistic serve values sensibly.

The synthetic serve is near-perfect, so it never exercises the 'info'/'warn'
paths a real club player would hit. This sweeps the plausible range and
asserts the classification is monotonic and correctly one-sided: bending the
knee MORE than elite, or straightening the elbow MORE than elite, must never
be flagged as a fault.
"""
import sys

import numpy as np

from analyzer.references import ELBOW_CONTACT as E, KNEE_TROPHY as K
from analyzer.phases import smooth_keypoints, detect_phases
from analyzer.metrics import compute_metrics
import test_synthetic as ts


def status_for(value, ref, direction):
    """Mirror of metrics.band_status, exercised through real reference data."""
    lo, hi, sd = ref["interior_lo"], ref["interior_hi"], ref["flexion_sd"]
    if lo <= value <= hi:
        return "good"
    if direction == "lower_ok" and value < lo:
        return "good"
    if direction == "higher_ok" and value > hi:
        return "good"
    gap = (lo - value) if value < lo else (value - hi)
    return "info" if gap <= sd else "warn"


def main():
    print(f"elbow at contact: band {E['interior_lo']:.1f}-{E['interior_hi']:.1f}"
          f" deg interior (elite mean {E['interior_mean']:.1f})")
    print(f"knee at trophy:   band {K['interior_lo']:.1f}-{K['interior_hi']:.1f}"
          f" deg interior (elite mean {K['interior_mean']:.1f})")

    failures = []

    print("\nelbow interior angle -> status")
    for v in (178, 166, 150, 140, 134, 125, 118, 100, 80):
        print(f"  {v:3d} deg -> {status_for(v, E, 'higher_ok')}")
    print("knee interior angle -> status")
    for v in (95, 106, 115, 125, 131, 135, 145, 160, 175):
        print(f"  {v:3d} deg -> {status_for(v, K, 'lower_ok')}")

    # A straighter-than-elite arm and a deeper-than-elite knee bend are not faults.
    if status_for(179, E, "higher_ok") != "good":
        failures.append("a fully straight elbow at contact must not be flagged")
    if status_for(90, K, "lower_ok") != "good":
        failures.append("a deeper-than-elite knee bend must not be flagged")
    # Genuinely poor mechanics must reach 'warn'.
    if status_for(90, E, "higher_ok") != "warn":
        failures.append("a heavily bent elbow at contact should warn")
    if status_for(170, K, "lower_ok") != "warn":
        failures.append("straight legs at trophy should warn")
    # The elite mean itself must land inside the band -- the old >=150 threshold
    # failed exactly this, flagging the average elite serve as a fault.
    if status_for(E["interior_mean"], E, "higher_ok") != "good":
        failures.append("the elite mean elbow angle must classify as good")
    if status_for(K["interior_mean"], K, "lower_ok") != "good":
        failures.append("the elite mean knee angle must classify as good")

    # The live pipeline must agree with this mirror on the synthetic serve.
    kp = smooth_keypoints(ts.synthetic_serve(), ts.FPS)
    phases, hand, _ = detect_phases(kp, ts.FPS)
    metrics, _ = compute_metrics(kp, phases, hand, ts.FPS)
    by_label = {m["label"]: m for m in metrics}
    for label in ("Elbow extension at contact", "Knee bend at trophy position"):
        m = by_label[label]
        if m["status"] != "good":
            failures.append(f"synthetic near-ideal serve should pass {label!r}, "
                            f"got {m['status']}")
        if "source" not in m or m["source"] is None:
            failures.append(f"{label!r} must carry a citation")

    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print("\nPASS: reference bands classify sensibly and carry citations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
