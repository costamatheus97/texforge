"""Kit consistency helpers: albedo calibration to plausible PBR ranges and height normalisation, plus the numbers
the kit check reports for each layer."""
from __future__ import annotations

import numpy as np

from .common import lin_to_srgb, luminance, srgb_to_lin


def _smax(a, b, k):
    return 0.5 * (a + b + np.sqrt((a - b) ** 2 + k * k))


def albedo_stats(base_srgb: np.ndarray) -> dict:
    """Linear-light luminance statistics of an sRGB albedo (the numbers PBR albedo ranges are quoted in)."""
    lum = luminance(srgb_to_lin(base_srgb))
    p = np.percentile(lum, [0.5, 5, 50, 95, 99.5])
    return {"mean_lin": round(float(lum.mean()), 4), "p0.5_lin": round(float(p[0]), 4), "p5_lin": round(float(p[1]), 4),
            "p50_lin": round(float(p[2]), 4), "p95_lin": round(float(p[3]), 4), "p99.5_lin": round(float(p[4]), 4),
            "srgb_p0.5": int(round(float(lin_to_srgb(p[0])) * 255)), "srgb_p99.5": int(round(float(lin_to_srgb(p[4])) * 255)),
            "pct_below_0.02": round(float((lum < 0.02).mean() * 100), 2),
            "pct_below_srgb30": round(float((base_srgb.max(-1) * 255 < 30).mean() * 100), 3),
            "mean_srgb_rgb": [round(float(base_srgb[..., c].mean() * 255), 1) for c in range(3)], "pct_above_0.85": round(float((lum > 0.85).mean() * 100), 2)}


def calibrate_albedo(base_srgb: np.ndarray, mean_band: tuple[float, float], floor: float = 0.03, ceil: float = 0.80):
    """Gain (in linear light) so the mean luminance lands inside `mean_band`, a soft knee that lifts crushed blacks
    to `floor`, a soft shoulder that holds highlights under `ceil`. Hue is kept (RGB scaled by the luminance ratio).
    -> (sRGB albedo, dict of what was done)"""
    lin = srgb_to_lin(base_srgb)
    lum = luminance(lin)
    m = float(lum.mean())
    lo, hi = mean_band
    info = {"mean_in_lin": round(m, 4), "band": [lo, hi]}
    gain = 1.0
    if m < lo:
        gain = (lo + 0.1 * (hi - lo)) / m
    elif m > hi:
        gain = (hi - 0.1 * (hi - lo)) / m
    info["gain"] = round(gain, 3)
    lum1 = lum * gain
    lum2 = _smax(lum1, floor, 0.5 * floor)                         # lift blacks
    lum3 = ceil - _smax(ceil - lum2, 0.0, 0.12 * ceil)             # roll off highlights
    ratio = np.clip(lum3 / np.maximum(lum, 1e-4), 0, 12)[..., None]
    out = lin * ratio
    # pure black pixels have no hue to scale: add the missing luminance as neutral
    missing = np.maximum(lum3 - luminance(out), 0)[..., None]
    out = out + missing
    return lin_to_srgb(np.clip(out, 0, 1)).astype(np.float32), info


def height_stats(h: np.ndarray) -> dict:
    p = np.percentile(h, [5, 50, 95])
    return {"mean": round(float(h.mean()), 3), "std": round(float(h.std()), 3), "p5": round(float(p[0]), 3),
            "p50": round(float(p[1]), 3), "p95": round(float(p[2]), 3), "spread_p95_p5": round(float(p[2] - p[0]), 3)}


def normalise_height(h: np.ndarray, mean: float = 0.5, std: float = 0.18) -> np.ndarray:
    """Re-centre and re-scale so the stored height has this mean and standard deviation (after clipping to 0..1),
    which makes the layers comparable in a height blend."""
    g = std / max(float(h.std()), 1e-4)
    for _ in range(6):
        out = np.clip(mean + (h - h.mean()) * g, 0, 1)
        g *= std / max(float(out.std()), 1e-4)
    return np.clip(mean + (h - h.mean()) * g, 0, 1).astype(np.float32)


def correct_hue_chroma(base_srgb: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    """Per-material hue / chroma correction in CIE Lab (L is untouched, so luminance / albedo range is not changed).
    cfg keys (all optional): hue_range [lo, hi] in Lab hue degrees (0 = +a red, 90 = +b yellow; dry pine litter ~ 62-80:
    ochre / grey-brown, orange sits at ~45-55); hue_strength 0..1 (1 = pixels outside the range are moved onto the nearest
    bound, smoothly weighted by their chroma so neutral pixels keep their colour); chroma_scale (multiplier);
    chroma_max (soft cap: C' = cap * tanh(C / cap), so nothing exceeds it and the ordering of chroma is preserved).
    -> (sRGB, report with before / after mean sRGB + chroma + hue)."""
    import cv2

    def stats(x):
        lab = cv2.cvtColor(np.ascontiguousarray(x, np.float32), cv2.COLOR_RGB2Lab)
        c = np.hypot(lab[..., 1], lab[..., 2])
        return {"mean_srgb_rgb": [round(float(x[..., i].mean() * 255), 1) for i in range(3)], "mean_L": round(float(lab[..., 0].mean()), 2),
                "chroma_mean": round(float(c.mean()), 2), "chroma_p95": round(float(np.percentile(c, 95)), 2),
                "hue_mean_deg": round(float(np.degrees(np.arctan2(lab[..., 2].mean(), lab[..., 1].mean())) % 360), 1)}
    rep = {"cfg": cfg, "before": stats(base_srgb)}
    lab = cv2.cvtColor(np.ascontiguousarray(base_srgb, np.float32), cv2.COLOR_RGB2Lab)
    a, b = lab[..., 1].copy(), lab[..., 2].copy()
    c = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a))
    if cfg.get("hue_range"):
        lo, hi = cfg["hue_range"]
        mid = 0.5 * (lo + hi)
        rel = (hue - mid + 180) % 360 - 180                      # signed offset from the range centre
        half = 0.5 * (hi - lo)
        over = np.sign(rel) * np.maximum(np.abs(rel) - half, 0)  # how far outside the range (0 inside)
        w = np.clip(c / 6.0, 0, 1) * float(cfg.get("hue_strength", 1.0))
        hue = hue - w * over
    cs = float(cfg.get("chroma_scale", 1.0))
    c2 = c * cs
    if cfg.get("chroma_max"):
        cap = float(cfg["chroma_max"])
        c2 = cap * np.tanh(c2 / cap)
    lab[..., 1] = c2 * np.cos(np.radians(hue))
    lab[..., 2] = c2 * np.sin(np.radians(hue))
    out = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1).astype(np.float32)
    rep["after"] = stats(out)
    return out, rep
