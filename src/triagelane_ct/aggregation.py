from .config import CONFIG


K_FRACTION = CONFIG["aggregation"]["k_fraction"]


def aggregate_study_score(slice_results, k_fraction=K_FRACTION):
    urgency_scores = [
        r["urgency_weighted_score"]
        for r in slice_results
    ]

    raw_scores = [
        r["calibrated_likelihood"]
        for r in slice_results
    ]

    k = max(
        3,
        round(k_fraction * len(slice_results))
    )

    top_k_urgency = sorted(
        urgency_scores,
        reverse=True
    )[:k]

    urgency_study_score = (
        sum(p * p for p in top_k_urgency)
        / sum(top_k_urgency)
    )

    top_k_raw = sorted(
        raw_scores,
        reverse=True
    )[:k]

    raw_study_score = (
        sum(p * p for p in top_k_raw)
        / sum(top_k_raw)
    )

    return (
        urgency_study_score,
        raw_study_score,
        k,
    )