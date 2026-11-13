"""Shared helpers: image IO, colour spaces, wrap-aware filtering."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def load_image(path: str | Path) -> np.ndarray:
    """-> float32 HxWx3 sRGB in [0,1] (alpha dropped)."""
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.float32) / 255.0


def save_png(path: str | Path, arr: np.ndarray) -> None:
    """Save float [0,1] array (HxW, HxWx3 or HxWx4) as 8-bit PNG."""
    a = np.clip(np.asarray(arr, np.float32), 0, 1)
    a = (a * 255.0 + 0.5).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)


def srgb_to_lin(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055).astype(np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luma of an RGB array in whatever space it is given."""
    return (rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722).astype(np.float32)


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def pad_wrap(a: np.ndarray, p: int, wrap: str = "xy") -> np.ndarray:
    """Pad p px: periodic on the axes named in `wrap` ('x','y','xy'), edge-replicated elsewhere."""
    pad = [(p, p), (p, p)] + [(0, 0)] * (a.ndim - 2)
    out = np.pad(a, [(pad[0] if "y" in wrap else (0, 0)), (pad[1] if "x" in wrap else (0, 0))] + pad[2:], mode="wrap")
    if "y" not in wrap:
        out = np.pad(out, [(p, p), (0, 0)] + [(0, 0)] * (a.ndim - 2), mode="edge")
    if "x" not in wrap:
        out = np.pad(out, [(0, 0), (p, p)] + [(0, 0)] * (a.ndim - 2), mode="edge")
    return out


def blur(a: np.ndarray, sigma: float, wrap: str = "xy") -> np.ndarray:
    """Gaussian blur honouring periodic boundaries (cv2 has no BORDER_WRAP for filters)."""
    if sigma <= 0:
        return a
    p = int(min(max(3 * sigma + 2, 4), min(a.shape[0], a.shape[1]) // 2 - 1))
    out = cv2.GaussianBlur(pad_wrap(a, p, wrap), (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    return out[p:-p, p:-p]


def resize_periodic(a: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Cubic resample that keeps the periodic phase exact (grid-wrap), so tiling survives the resize.
    (For x-only tiling the y borders blend over ~1px, which is harmless.)"""
    h, w = a.shape[:2]
    oh, ow = out_hw
    if (h, w) == (oh, ow):
        return a.astype(np.float32)
    zoom = [oh / h, ow / w] + [1] * (a.ndim - 2)
    return ndi.zoom(a.astype(np.float32), zoom, order=3, mode="grid-wrap", grid_mode=True).astype(np.float32)


def resize(a: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Non-periodic resize (Lanczos up, area down)."""
    h, w = a.shape[:2]
    oh, ow = out_hw
    interp = cv2.INTER_AREA if (oh * ow < h * w) else cv2.INTER_LANCZOS4
    return cv2.resize(a.astype(np.float32), (ow, oh), interpolation=interp)
