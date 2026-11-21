"""Spectral indices from Landsat C2 L2 surface reflectance (Landsat 8/9 band names)."""
import numpy as np


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den == 0, np.nan, num / den)


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Vegetation: Band 5 (NIR) and Band 4 (red)."""
    return _safe_ratio(nir - red, nir + red)


def ndbi(swir1: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Built-up: Band 6 (SWIR1) and Band 5 (NIR)."""
    return _safe_ratio(swir1 - nir, swir1 + nir)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Water: Band 3 (green) and Band 5 (NIR)."""
    return _safe_ratio(green - nir, green + nir)
