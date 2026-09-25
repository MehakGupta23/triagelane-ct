from src.triagelane_ct.aggregation import (
    aggregate_study_score,
)


def test_aggregate_study_score():
    slice_results = [
        {
            "urgency_weighted_score": 0.2,
            "calibrated_likelihood": 0.1,
        },
        {
            "urgency_weighted_score": 0.8,
            "calibrated_likelihood": 0.7,
        },
        {
            "urgency_weighted_score": 0.6,
            "calibrated_likelihood": 0.5,
        },
    ]

    urgency_score, raw_score, k = aggregate_study_score(
        slice_results
    )

    assert k == 3
    assert 0 <= urgency_score <= 1
    assert 0 <= raw_score <= 1