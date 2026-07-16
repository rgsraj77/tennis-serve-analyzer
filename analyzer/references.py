"""Published serve kinematics used as the comparison baseline.

IMPORTANT UNIT CONVENTION
-------------------------
The literature reports *flexion* angles: 0 deg = fully straight limb, larger =
more bent. This codebase computes *interior* joint angles from keypoints:
180 deg = fully straight, smaller = more bent. So:

    interior = 180 - flexion

Every value below is stored in BOTH forms to keep the conversion auditable.

CAVEATS THE REPORT MUST NOT HIDE
--------------------------------
1. These are elite/world-class populations (Olympic competitors, skilled
   players). They are a *reference*, not a target for a club player, and
   deviation from them is not automatically a fault.
2. Source studies use 3D marker-based motion capture. We measure 2D angles
   projected onto one camera plane, which systematically under-reads any
   angle whose limb swings out of the camera plane. Treat our numbers as
   approximations of the 3D quantity, not equivalents.
3. Bands here are mean +/- 1 SD, i.e. roughly the middle ~68% of the
   reference population -- deliberately not "mean or better", which would
   fail half of the elite players the data came from.
"""

SOURCES = {
    "meta2024": {
        "cite": "Wang et al. (2024), Kinematics characteristics of key point of "
                "interest during tennis serve: a systematic review and "
                "meta-analysis, Front. Sports Act. Living",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11260724/",
    },
    "fleisig2003": {
        "cite": "Fleisig et al. (2003), Kinematics used by world class tennis "
                "players to produce high-velocity serves, Sports Biomech. 2(1)",
        "url": "https://pubmed.ncbi.nlm.nih.gov/14658245/",
    },
    "kovacs2011": {
        "cite": "Kovacs & Ellenbecker (2011), An 8-Stage Model for Evaluating "
                "the Tennis Serve, Sports Health 3(6)",
        "url": "https://journals.sagepub.com/doi/10.1177/1941738111414175",
    },
}


def _interior(flexion_mean, flexion_sd):
    """Convert a published flexion mean/SD into our interior-angle convention."""
    return {
        "flexion_mean": flexion_mean,
        "flexion_sd": flexion_sd,
        "interior_mean": 180.0 - flexion_mean,
        # +/-1 SD in flexion maps to the same +/-1 SD span in interior angle,
        # with the bounds swapped: more flexion = smaller interior angle.
        "interior_lo": 180.0 - flexion_mean - flexion_sd,
        "interior_hi": 180.0 - flexion_mean + flexion_sd,
    }


# Knee flexion at the trophy / loading position (front knee).
# Meta-analysis pooled mean 64.5 +/- 9.7 deg flexion -> 115.5 deg interior,
# 1 SD band 105.8-125.2 deg interior.
KNEE_TROPHY = {
    **_interior(64.5, 9.7),
    "source": "meta2024",
    "detail": "front knee at trophy position, pooled across studies",
}

# Elbow flexion at ball impact.
# Meta-analysis pooled mean 30.1 +/- 15.9 deg flexion -> 149.9 deg interior,
# 1 SD band 134.0-165.8 deg interior. Note Fleisig (2003) separately reports
# the elbow is only "slightly flexed" at impact, consistent with this.
ELBOW_CONTACT = {
    **_interior(30.1, 15.9),
    "source": "meta2024",
    "detail": "elbow at ball impact, pooled across nine studies",
}

# Contact height has no directly transferable published value: studies report
# it in absolute metres or as % of standing height from 3D capture, neither of
# which survives our single-camera, scale-free 2D setup. Kept explicitly as an
# uncalibrated heuristic rather than dressed up with a citation.
CONTACT_HEIGHT_HEURISTIC = {
    "good_min": 1.30,
    "ok_min": 1.15,
    "source": None,
    "detail": "uncalibrated heuristic in nose-to-ankle units; not literature-derived",
}
