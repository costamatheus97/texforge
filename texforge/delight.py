"""De-lighting: remove large-scale brightness gradients while preserving the mean albedo."""
from __future__ import annotations

import cv2
import numpy as np

from .common import luminance, lin_to_srgb, srgb_to_lin


def _poly_field(lum: np.ndarray, degree: int = 3) -> np.ndarray:
    """Smooth 2-D polynomial fit of log-luminance (robust at the borders, unlike a reflect-padded blur)."""
    h, w = lum.shape
    sh, sw = 64, 64
    small = cv2.resize(lum, (sw, sh), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float64)
    xn, yn = xx / (sw - 1) * 2 - 1, yy / (sh - 1) * 2 - 1
    terms = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]
    A = np.stack([(xn ** i * yn ** j).ravel() for i, j in terms], 1)
    z = np.log(np.maximum(small, 1e-4)).ravel().astype(np.float64)
    # two reweighting passes so bright/dark features (joints, stones) do not drag the fit
    wgt = np.ones_like(z)
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(A * wgt[:, None], z * wgt, rcond=None)
        r = z - A @ coef
        wgt = 1.0 / (1.0 + (r / (1.5 * (np.median(np.abs(r)) + 1e-6))) ** 2)
    fit = (A @ coef).reshape(sh, sw)
    return cv2.resize(np.exp(fit).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)


def delight(srgb: np.ndarray, sigma_frac: float = 0.12, strength: float = 1.0, wrap: str | None = None) -> np.ndarray:
    """Remove large-scale lighting in linear light, keeping the mean albedo.

    1. divide by a cubic polynomial fit of the luminance (kills ramps / vignettes / one-sided light cleanly,
       including at the image borders);
    2. divide by a heavily blurred remainder (sigma = sigma_frac * short side) for blotchy low-frequency drift.
    sigma_frac must be much larger than the texture's own pattern period (bricks, cobbles) or the pattern
    itself gets flattened. strength 0..1 scales the correction; ratios are clamped to [0.25, 4].
    """
    lin = srgb_to_lin(srgb)
    lum = luminance(lin)
    h, w = lum.shape
    mean = float(lum.mean())
    poly = _poly_field(lum)
    lum2 = lum / np.maximum(poly, 1e-4)
    sigma = max(2.0, sigma_frac * min(h, w))
    small = cv2.resize(lum2, (max(8, w // 8), max(8, h // 8)), interpolation=cv2.INTER_AREA)
    p = int(3 * sigma / 8) + 2
    sm = cv2.GaussianBlur(np.pad(small, p, mode="reflect"), (0, 0), sigma / 8)[p:-p, p:-p]
    low = cv2.resize(sm, (w, h), interpolation=cv2.INTER_CUBIC)
    ratio = np.clip(mean / (np.maximum(poly, 1e-4) * np.maximum(low, 1e-4)), 0.25, 4.0)
    ratio = 1.0 + (ratio - 1.0) * strength
    out = lin * ratio[..., None]
    out *= mean / max(float(luminance(out).mean()), 1e-6)  # preserve mean albedo
    return lin_to_srgb(np.clip(out, 0, 1))
