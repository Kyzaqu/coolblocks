"""Landsat Collection 2 Level 2 helpers: QA_PIXEL cloud masking and scaling."""
import numpy as np

# QA_PIXEL bit positions (Landsat C2 L2)
FILL, DILATED_CLOUD, CIRRUS, CLOUD, CLOUD_SHADOW, SNOW = 0, 1, 2, 3, 4, 5


def clear_mask(qa_pixel: np.ndarray) -> np.ndarray:
    """True where a pixel is clear (no fill, dilated cloud, cirrus, cloud, shadow or snow)."""
    qa = qa_pixel.astype(np.uint16)
    bad = 0
    for bit in (FILL, DILATED_CLOUD, CIRRUS, CLOUD, CLOUD_SHADOW, SNOW):
        bad |= (qa >> bit) & 1
    return bad == 0


def st_to_celsius(st_b10: np.ndarray) -> np.ndarray:
    """Convert Landsat C2 L2 surface temperature DN to degrees Celsius."""
    kelvin = st_b10.astype("float32") * 0.00341802 + 149.0
    return kelvin - 273.15


def sr_to_reflectance(dn: np.ndarray) -> np.ndarray:
    """Convert Landsat C2 L2 surface reflectance DN to reflectance (0-1)."""
    return dn.astype("float32") * 0.0000275 - 0.2
