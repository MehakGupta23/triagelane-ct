import glob
import os
from collections import defaultdict

import numpy as np
import pydicom

from .config import CONFIG


EXPECTED_MODALITY = CONFIG["input"]["expected_modality"]


class InvalidDicomSliceError(ValueError):
    pass


def validate_dicom_slice(dcm):
    modality = getattr(dcm, "Modality", None)

    if modality != EXPECTED_MODALITY:
        raise InvalidDicomSliceError(
            f"expected Modality={EXPECTED_MODALITY}, got {modality}"
        )

    if not hasattr(dcm, "RescaleSlope") or not hasattr(dcm, "RescaleIntercept"):
        raise InvalidDicomSliceError(
            "missing RescaleSlope/RescaleIntercept"
        )

    pixel_array = dcm.pixel_array

    if pixel_array is None or pixel_array.ndim != 2 or pixel_array.size == 0:
        raise InvalidDicomSliceError("invalid pixel_array shape")

    if np.all(pixel_array == pixel_array.flat[0]):
        raise InvalidDicomSliceError("pixel_array has no variation")

    return pixel_array


def discover_study_dicom_paths(folder_path):
    all_paths = sorted(
        glob.glob(os.path.join(folder_path, "*.dcm"))
    )

    if not all_paths:
        raise FileNotFoundError(folder_path)

    series_map = defaultdict(list)

    for path in all_paths:
        dcm = pydicom.dcmread(
            path,
            stop_before_pixels=True
        )

        if getattr(dcm, "Modality", None) != EXPECTED_MODALITY:
            continue

        series_uid = getattr(
            dcm,
            "SeriesInstanceUID",
            "UNKNOWN_SERIES"
        )

        instance_number = getattr(
            dcm,
            "InstanceNumber",
            None
        )

        series_map[series_uid].append(
            (instance_number, path)
        )

    if not series_map:
        raise InvalidDicomSliceError(
            f"no CT series found in {folder_path}"
        )

    chosen_series = max(
        series_map.values(),
        key=len
    )

    chosen_series.sort(
        key=lambda item: (
            item[0] is None,
            item[0]
        )
    )

    return [path for _, path in chosen_series]