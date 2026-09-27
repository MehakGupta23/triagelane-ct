import os
import pydicom

from .aggregation import aggregate_study_score
from .dicom_io import (
    discover_study_dicom_paths,
    validate_dicom_slice,
)
from .lanes import assign_lane, lane_decision_reason
from .model import predict_slice

from .gradcam import (
    _dominant_subtype_class_index,
    compute_gradcam,
    save_gradcam_visualization,
)


def score_study(
    slices,
    slice_ids=None,
    generate_localization=False,
    localization_output_dir=None,
):
    if slice_ids is None:
        slice_ids = list(range(len(slices)))

    slice_results = [
        predict_slice(px, rs, ri)
        for px, rs, ri in slices
    ]

    urgency_study_score, raw_study_score, k = (
        aggregate_study_score(slice_results)
    )

    top_slice_index = max(
        range(len(slice_results)),
        key=lambda i: slice_results[i]["urgency_weighted_score"],
    )

    dominant_subtype = slice_results[top_slice_index]["dominant_subtype"]

    lane_result = assign_lane(
        urgency_study_score,
        raw_study_score,
        dominant_subtype,
    )

    if lane_result["abstain"]:
        decision_reason = lane_result["abstain_reason"]
    else:
        decision_reason = lane_decision_reason(
            lane_result["lane"],
            urgency_study_score,
            dominant_subtype,
            k,
            len(slices),
        )

    result = {
        "study_score": round(urgency_study_score, 3),
        "raw_score": round(raw_study_score, 3),
        "k_used": k,
        "n_slices": len(slices),
        "dominant_subtype": dominant_subtype,
        "decision_reason": decision_reason,
        **lane_result,
    }

    if generate_localization:
        top_pixel_array, top_rescale_slope, top_rescale_intercept = (
            slices[top_slice_index]
        )

        target_class_index = _dominant_subtype_class_index(
            dominant_subtype
        )

        gradcam_result = compute_gradcam(
            top_pixel_array,
            top_rescale_slope,
            top_rescale_intercept,
            target_class_index,
        )

        visualization_path = None

        if localization_output_dir is not None:
            file_name = f"gradcam_slice_{top_slice_index}.png"

            visualization_path = save_gradcam_visualization(
                top_pixel_array,
                top_rescale_slope,
                top_rescale_intercept,
                gradcam_result,
                os.path.join(
                    localization_output_dir,
                    file_name,
                ),
            )

        result["localization"] = {
            "slice_index": top_slice_index,
            "slice_id": slice_ids[top_slice_index],
            "target_class_name": gradcam_result["target_class_name"],
            "target_layer": gradcam_result["target_layer_name"],
            "bounding_box": gradcam_result["bounding_box"],
            "centroid": gradcam_result["centroid"],
            "visualization_path": visualization_path,
        }

    return result


def score_study_from_dicom_paths(
    paths,
    generate_localization=False,
    localization_output_dir=None,
):
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

    return score_study(
        slices,
        slice_ids=paths,
        generate_localization=generate_localization,
        localization_output_dir=localization_output_dir,
    )


def score_study_from_folder(
    folder_path,
    generate_localization=False,
    localization_output_dir=None,
):
    paths = discover_study_dicom_paths(folder_path)

    return score_study_from_dicom_paths(
        paths,
        generate_localization=generate_localization,
        localization_output_dir=localization_output_dir,
    )