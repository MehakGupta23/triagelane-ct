import numpy as np
import pytest

from src.triagelane_ct.dicom_io import (
    validate_dicom_slice,
    InvalidDicomSliceError,
)


class FakeDicom:
    Modality = "CT"
    RescaleSlope = 1
    RescaleIntercept = 0

    @property
    def pixel_array(self):
        return np.array(
            [
                [0, 1],
                [2, 3],
            ],
            dtype=np.int16,
        )


def test_validate_valid_ct_slice():
    dcm = FakeDicom()

    result = validate_dicom_slice(dcm)

    assert result.shape == (2, 2)


def test_validate_rejects_wrong_modality():
    dcm = FakeDicom()
    dcm.Modality = "MR"

    with pytest.raises(InvalidDicomSliceError):
        validate_dicom_slice(dcm)