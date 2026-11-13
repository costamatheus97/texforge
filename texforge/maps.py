"""PBR map derivation from a (de-lit, seamless) albedo: height, normal, roughness, AO, ORM packing."""
from __future__ import annotations

import cv2
import numpy as np

from .common import blur, luminance, pad_wrap

# per-material defaults. normal: relief strength; rough: base roughness; joint_rough: roughness inside
# joints/gaps; ao: cavity amount; joint_w: how much of the height comes from the joint mask.
TYPE_DEFAULTS = {
    "brick":   dict(rough=0.80, joint_rough=0.92, normal=3.0, ao=0.70, joint_w=0.60, metallic=0.0),
    "tile":    dict(rough=0.40, joint_rough=0.85, normal=2.0, ao=0.60, joint_w=0.75, metallic=0.0),
    "cobble":  dict(rough=0.75, joint_rough=0.92, normal=3.5, ao=0.75, joint_w=0.55, metallic=0.0),
    "plaster": dict(rough=0.88, joint_rough=0.88, normal=1.6, ao=0.50, joint_w=0.0, metallic=0.0),
    "organic": dict(rough=0.92, joint_rough=0.92, normal=3.0, ao=0.70, joint_w=0.0, metallic=0.0),
}


def _norm(a: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    l, h = np.percentile(a, [lo, hi])
    return np.clip((a - l) / max(h - l, 1e-6), 0, 1).astype(np.float32)


def height_map(albedo: np.ndarray, joint: np.ndarray | None = None, joint_w: float = 0.0,
               wrap: str = "xy") -> np.ndarray:
    """Height in [0,1] (1 = high). Joints/gaps are low; the rest follows band-passed luminance
    (dark cavities low, bright high) so the surface keeps its micro and mid-scale relief."""
    lum = luminance(albedo)
    size = max(albedo.shape[:2])
    s_f = max(0.8, 0.0006 * size)
    fine = _norm(blur(lum, s_f, wrap) - blur(lum, 0.02 * size, wrap))
    mid = _norm(blur(lum, 0.008 * size, wrap) - blur(lum, 0.06 * size, wrap))
    h = 0.6 * fine + 0.4 * mid
    if joint is not None and joint_w > 0:
        j = blur(joint, max(1.0, 0.0025 * size), wrap)
        h = (1 - joint_w) * h * (1 - 0.5 * j) + joint_w * (1 - j) * (0.85 + 0.15 * h)
    return _norm(h, 0.5, 99.5)


def soft_unit(h: np.ndarray, lo_pct: float = 0.2, hi_pct: float = 99.8) -> np.ndarray:
    """Map a height to (0,1) WITHOUT a hard clip: robust linear rescale, then a tanh shoulder at both ends. A hard
    np.clip flattens every pixel below the floor to one value (flat plateaus with a sharp rim in the normal map);
    the shoulder keeps the slope non-zero, so micro-relief survives in the deepest gaps."""
    lo, hi = np.percentile(h, [lo_pct, hi_pct])
    t = (h - lo) / max(float(hi - lo), 1e-6)
    return (0.5 + 0.5 * np.tanh(1.6 * (t - 0.5))).astype(np.float32)      # p0.2 -> 0.17, p99.8 -> 0.83, never flat


def normal_map(height: np.ndarray, strength: float = 3.0, wrap: str = "xy", aspect_y: float = 1.0) -> np.ndarray:
    """Tangent-space normal in the DirectX/Unreal convention (+Y = green points DOWN the image).

    With image y pointing down: N = normalize(-k*dh/dx, -k*dh/dy, 1) -> encoded (N*0.5+0.5).
    Verified with a synthetic bump: the bump's bright-green edge is its bottom edge.
    aspect_y = world_cm_x / world_cm_y of the tile: a non-square tile stored as a square texture has taller
    (or shorter) texels along y, so the y slope is scaled to keep the relief physically consistent.
    """
    size = max(height.shape[:2])
    h = blur(height, 0.8, wrap)
    p = 2
    hp = pad_wrap(h, p, wrap)
    gx = cv2.Sobel(hp, cv2.CV_32F, 1, 0, ksize=3)[p:-p, p:-p] / 8.0
    gy = cv2.Sobel(hp, cv2.CV_32F, 0, 1, ksize=3)[p:-p, p:-p] / 8.0
    k = strength * size / 1024.0 * 8.0
    n = np.stack([-k * gx, -k * gy * aspect_y, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return (n * 0.5 + 0.5).astype(np.float32)


def ao_map(height: np.ndarray, amount: float = 0.7, wrap: str = "xy") -> np.ndarray:
    """Cavity AO: how far each pixel sits below its blurred neighbourhood at three scales."""
    size = max(height.shape[:2])
    cav = np.zeros_like(height)
    for s, w in ((0.004, 0.45), (0.016, 0.35), (0.05, 0.20)):
        cav += w * np.maximum(blur(height, s * size, wrap) - height, 0)
    cav = np.clip(cav / max(np.percentile(cav, 99.5), 1e-6), 0, 1)
    return (1 - amount * cav ** 1.6).astype(np.float32)


def roughness_map(albedo: np.ndarray, joint: np.ndarray | None, base: float, joint_rough: float,
                  wrap: str = "xy") -> np.ndarray:
    lum = luminance(albedo)
    size = max(albedo.shape[:2])
    ln = _norm(blur(lum, 0.003 * size, wrap) - blur(lum, 0.08 * size, wrap))
    r = base + 0.16 * (0.5 - ln)             # darker (grimy / cavity) = rougher, brighter = a bit smoother
    if joint is not None:
        j = blur(joint, max(1.0, 0.003 * size), wrap)
        r = r * (1 - j) + joint_rough * j
    return np.clip(r, 0.12, 1.0).astype(np.float32)


def pack_orm(ao: np.ndarray, rough: np.ndarray, metallic: np.ndarray | float = 0.0) -> np.ndarray:
    m = np.full_like(ao, metallic) if np.isscalar(metallic) else metallic
    return np.stack([ao, rough, m], -1).astype(np.float32)


def derive_all(albedo: np.ndarray, mtype: str, joint: np.ndarray | None = None, normal_strength: float | None = None,
               wrap: str = "xy", aspect_y: float = 1.0, height: np.ndarray | None = None,
               rough: float | None = None) -> dict:
    d = TYPE_DEFAULTS[mtype]
    h = height if height is not None else height_map(albedo, joint, d["joint_w"], wrap)
    n = normal_map(h, normal_strength if normal_strength is not None else d["normal"], wrap, aspect_y)
    ao = ao_map(h, d["ao"], wrap)
    r = roughness_map(albedo, joint, d["rough"] if rough is None else rough, d["joint_rough"], wrap)
    return {"height": h, "normal": n, "ao": ao, "rough": r, "orm": pack_orm(ao, r, d["metallic"])}
