import numpy as np

from src.triagelane_ct.preprocessing import (
    to_hounsfield,
    apply_window,
    three_window_stack,
)


def test_to_hounsfield():
    pixels = np.array([[0, 100]], dtype=np.int16)

    result = to_hounsfield(
        pixels,
        rescale_slope=1,
        rescale_intercept=-1000,
    )

    assert result[0, 0] == -1000
    assert result[0, 1] == -900


def test_apply_window():
    hu = np.array([[-1000, 40, 1000]], dtype=np.float32)

    result = apply_window(
        hu,
        center=40,
        width=80,
    )

    assert result.dtype == np.uint8
    assert result.shape == hu.shape


def test_three_window_stack():
    pixels = np.random.randint(
        -1000,
        2000,
        size=(64, 64),
        dtype=np.int16,
    )

    result = three_window_stack(
        pixels,
        1,
        0,
    )

    assert result.shape == (64, 64, 3)
    assert result.dtype == np.uint8