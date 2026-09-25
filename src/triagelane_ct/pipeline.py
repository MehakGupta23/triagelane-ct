import pydicom

from .aggregation import aggregate_study_score
from .dicom_io import (
    discover_study_dicom_paths,
    validate_dicom_slice,
)
from .lanes import assign_lane
from .model import predict_slice


def score_study(slices):
    slice_results = [
        predict_slice(px, rs, ri)
        for px, rs, ri in slices
    ]

    urgency_study_score, raw_study_score, k = (
        aggregate_study_score(slice_results)
    )

    top_slice = max(
        slice_results,
        key=lambda r: r["urgency_weighted_score"],
    )

    dominant_subtype = top_slice["dominant_subtype"]

    lane_result = assign_lane(
        urgency_study_score,
        raw_study_score,
        dominant_subtype,
    )

    return {
        "study_score": round(urgency_study_score, 3),
        "raw_score": round(raw_study_score, 3),
        "k_used": k,
        "n_slices": len(slices),
        "dominant_subtype": dominant_subtype,
        **lane_result,
    }


def score_study_from_dicom_paths(paths):
    slices = []

    for path in paths:
        dcm = pydicom.dcmread(path)

        pixel_array = validate_dicom_slice(dcm)

        slices.append(
            (
                pixel_array,
                dcm.RescaleSlope,
                dcm.RescaleIntercept,
            )
        )

    return score_study(slices)


def score_study_from_folder(folder_path):
    paths = discover_study_dicom_paths(folder_path)

    return score_study_from_dicom_paths(paths)