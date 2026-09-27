import numpy as np
from src.triagelane_ct import pipeline


def test_gradcam_localization_output(monkeypatch, tmp_path):
    slices = [
        (
            np.zeros((8, 8), dtype=np.int16),
            1.0,
            0.0,
        ),
        (
            np.ones((8, 8), dtype=np.int16),
            1.0,
            0.0,
        ),
        (
            np.full((8, 8), 2, dtype=np.int16),
            1.0,
            0.0,
        ),
    ]

    fake_slice_results = [
        {
            "urgency_weighted_score": 0.20,
            "dominant_subtype": "subarachnoid",
        },
        {
            "urgency_weighted_score": 0.60,
            "dominant_subtype": "subarachnoid",
        },
        {
            "urgency_weighted_score": 0.40,
            "dominant_subtype": "subarachnoid",
        },
    ]

    monkeypatch.setattr(
        pipeline,
        "predict_slice",
        lambda px, rs, ri: fake_slice_results.pop(0),
    )

    monkeypatch.setattr(
        pipeline,
        "aggregate_study_score",
        lambda results: (0.60, 0.55, 2),
    )

    monkeypatch.setattr(
        pipeline,
        "assign_lane",
        lambda urgency, raw, subtype: {
            "lane": "expedited",
            "abstain": False,
            "in_abstention_tray": False,
            "sort_score": urgency,
            "abstain_reason": None,
        },
    )

    monkeypatch.setattr(
        pipeline,
        "lane_decision_reason",
        lambda lane, score, subtype, k, n: "test decision reason",
    )

    monkeypatch.setattr(
        pipeline,
        "_dominant_subtype_class_index",
        lambda subtype: 4,
    )

    fake_heatmap = np.zeros((8, 8), dtype=np.float32)

    monkeypatch.setattr(
        pipeline,
        "compute_gradcam",
        lambda px, rs, ri, target_class_index: {
            "heatmap": fake_heatmap,
            "target_class_index": target_class_index,
            "target_class_name": "subarachnoid",
            "target_layer_name": "test_layer",
            "bounding_box": {
                "row_min": 1,
                "row_max": 6,
                "col_min": 2,
                "col_max": 7,
            },
            "centroid": {
                "row": 3.5,
                "col": 4.5,
            },
        },
    )

    visualization_path = tmp_path / "gradcam_slice_1.png"

    monkeypatch.setattr(
        pipeline,
        "save_gradcam_visualization",
        lambda *args: str(visualization_path),
    )

    result = pipeline.score_study(
        slices,
        slice_ids=["slice_0", "slice_1", "slice_2"],
        generate_localization=True,
        localization_output_dir=str(tmp_path),
    )

    assert result["decision_reason"] == "test decision reason"

    assert "localization" in result

    localization = result["localization"]

    assert localization["slice_index"] == 1
    assert localization["slice_id"] == "slice_1"
    assert localization["target_class_name"] == "subarachnoid"
    assert localization["target_layer"] == "test_layer"

    assert localization["bounding_box"] == {
        "row_min": 1,
        "row_max": 6,
        "col_min": 2,
        "col_max": 7,
    }

    assert localization["centroid"] == {
        "row": 3.5,
        "col": 4.5,
    }

    assert localization["visualization_path"] == str(
        visualization_path
    )