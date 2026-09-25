import numpy as np

from .config import CONFIG


BRAIN_WINDOW = (
    CONFIG["windows"]["brain"]["center"],
    CONFIG["windows"]["brain"]["width"],
)

SUBDURAL_WINDOW = (
    CONFIG["windows"]["subdural"]["center"],
    CONFIG["windows"]["subdural"]["width"],
)

BONE_WINDOW = (
    CONFIG["windows"]["bone"]["center"],
    CONFIG["windows"]["bone"]["width"],
)


def to_hounsfield(pixel_array, rescale_slope, rescale_intercept):
    return (
        pixel_array.astype(np.float32)
        * float(rescale_slope)
        + float(rescale_intercept)
    )


def apply_window(hu_array, center, width):
    lo = center - width // 2
    hi = center + width // 2

    return (
        (np.clip(hu_array, lo, hi) - lo)
        / (hi - lo)
        * 255
    ).astype(np.uint8)


def three_window_stack(
    pixel_array,
    rescale_slope,
    rescale_intercept,
):
    hu = to_hounsfield(
        pixel_array,
        rescale_slope,
        rescale_intercept,
    )

    return np.stack(
        [
            apply_window(hu, *BRAIN_WINDOW),
            apply_window(hu, *SUBDURAL_WINDOW),
            apply_window(hu, *BONE_WINDOW),
        ],
        axis=-1,
    )