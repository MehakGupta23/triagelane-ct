from .config import CONFIG


BOUNDARY_MARGIN = CONFIG["lanes"]["boundary_margin"]

THRESHOLDS = CONFIG["lanes"]["thresholds"]

CRITICAL_THRESHOLD = THRESHOLDS["critical"]
URGENT_THRESHOLD = THRESHOLDS["urgent"]
EXPEDITED_THRESHOLD = THRESHOLDS["expedited"]

LANE_BOUNDARIES = (
    CRITICAL_THRESHOLD,
    URGENT_THRESHOLD,
    EXPEDITED_THRESHOLD,
)

LANE_BOUNDARY_NAMES = {
    float(k): v
    for k, v in CONFIG["lanes"]["boundary_names"].items()
}

LOW_URGENCY_WEIGHT_THRESHOLD = (
    CONFIG["mismatch_check"]["low_urgency_weight_threshold"]
)

HIGH_CONFIDENCE_THRESHOLD = (
    CONFIG["mismatch_check"]["high_confidence_threshold"]
)

URGENCY_WEIGHTS = CONFIG["urgency_weights"]

LANE_CRITICAL = "critical"
LANE_URGENT = "urgent"
LANE_EXPEDITED = "expedited"
LANE_ROUTINE = "routine"


def assign_lane(
    urgency_study_score,
    raw_study_score,
    dominant_subtype,
    margin=BOUNDARY_MARGIN,
):
    if (
        URGENCY_WEIGHTS[dominant_subtype]
        <= LOW_URGENCY_WEIGHT_THRESHOLD
        and raw_study_score >= HIGH_CONFIDENCE_THRESHOLD
    ):
        return {
            "lane": None,
            "abstain": True,
            "in_abstention_tray": True,
            "sort_score": urgency_study_score,
            "abstain_reason": (
                f"High confidence ({raw_study_score:.3f}) "
                f"on a low-urgency finding type "
                f"({dominant_subtype}) -- shown immediately "
            ),
        }

    if any(
        abs(urgency_study_score - b) <= margin
        for b in LANE_BOUNDARIES
    ):
        nearest = min(
            LANE_BOUNDARIES,
            key=lambda b: abs(urgency_study_score - b),
        )

        return {
            "lane": None,
            "abstain": True,
            "in_abstention_tray": True,
            "sort_score": urgency_study_score,
            "abstain_reason": (
                f"Borderline {LANE_BOUNDARY_NAMES[nearest]} "
                f"(score={urgency_study_score:.3f}) "
            ),
        }

    if urgency_study_score >= CRITICAL_THRESHOLD:
        return {
            "lane": LANE_CRITICAL,
            "abstain": False,
            "in_abstention_tray": False,
            "sort_score": urgency_study_score,
            "abstain_reason": None,
        }

    if urgency_study_score >= URGENT_THRESHOLD:
        return {
            "lane": LANE_URGENT,
            "abstain": False,
            "in_abstention_tray": False,
            "sort_score": urgency_study_score,
            "abstain_reason": None,
        }

    if urgency_study_score >= EXPEDITED_THRESHOLD:
        return {
            "lane": LANE_EXPEDITED,
            "abstain": False,
            "in_abstention_tray": False,
            "sort_score": urgency_study_score,
            "abstain_reason": None,
        }

    return {
        "lane": LANE_ROUTINE,
        "abstain": False,
        "in_abstention_tray": False,
        "sort_score": urgency_study_score,
        "abstain_reason": None,
    }