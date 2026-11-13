"""Macro variation texture (procedural, no codex): large, low-frequency patches that break the repetition of the
ground layers at 20-50 m. Periodic by construction (spectral synthesis), so it tiles in both axes.
  R = damp / dark patches   G = leaf-drift density (stretched along a "wind" direction)   B = brightness / dryness drift
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import save_png, smoothstep


def fbm(n: int, rng, cycles: tuple[float, float], beta: float = 2.2, stretch: float = 1.0, angle: float = 0.0) -> np.ndarray:
    """Periodic noise in [0,1] (rank-normalised) with features between `cycles` cycles per tile, power spectrum ~ f^-beta.
    stretch > 1 elongates features along `angle` (radians)."""
    f = np.fft.fftfreq(n) * n
    fx, fy = np.meshgrid(f, f)
    c, s = np.cos(angle), np.sin(angle)
    u, v = fx * c + fy * s, -fx * s + fy * c
    r = np.hypot(u * stretch, v)                    # features are long along u when stretch > 1... (frequency along u penalised)
    amp = np.where((r >= cycles[0]) & (r <= cycles[1]), np.maximum(r, 1e-6) ** (-beta / 2), 0.0)
    spec = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * amp
    x = np.real(np.fft.ifft2(spec))
    order = x.ravel().argsort().argsort()
    return (order / (x.size - 1)).reshape(n, n).astype(np.float32)


def make_macro(out_root: Path, name: str = "macro", size: int = 2048, world_cm: float = 3000, seed: int = 7, log=print) -> dict:
    rng = np.random.default_rng(seed)
    # soft ramps, not thresholds: a damp patch fades over several metres; coverage ~25 %
    damp = smoothstep(0.40, 1.0, fbm(size, rng, (2, 12), 2.6))
    damp = np.clip(damp * (0.85 + 0.3 * fbm(size, rng, (12, 40), 2.0)), 0, 1)
    drift = fbm(size, rng, (3, 24), 2.4, stretch=3.0, angle=0.35)       # stretched along a wind direction
    drift = smoothstep(0.42, 1.0, drift) * (0.8 + 0.4 * (fbm(size, rng, (15, 60), 2.2) - 0.5))
    drift = np.clip(drift, 0, 1)
    bright = 0.5 + (fbm(size, rng, (1, 6), 2.4) - 0.5) * 0.7 + (fbm(size, rng, (6, 25), 2.2) - 0.5) * 0.2
    rgb = np.stack([damp, drift, np.clip(bright, 0, 1)], -1).astype(np.float32)
    d = Path(out_root) / name
    d.mkdir(parents=True, exist_ok=True)
    save_png(d / f"{name}_Macro.png", rgb)
    stats = {c: [round(float(rgb[..., i].mean()), 3), round(float(rgb[..., i].std()), 3)] for i, c in enumerate("RGB")}
    meta = {"name": name, "kind": "macro", "size": [size, size], "world_cm": [world_cm, world_cm], "tileable": True, "tile_axes": "xy",
            "channels": "R=damp/dark patches, G=leaf-drift density, B=brightness/dryness drift (0.5 = neutral)", "linear": True,
            "files": {"macro": f"{name}_Macro.png"}, "params": {"seed": seed, "mean_std": stats}}
    (d / f"{name}.texforge.json").write_text(json.dumps(meta, indent=2))
    # preview: the tile 2x2 (to judge repetition) next to the three channels
    from .qa import tile_sheet
    save_png(d / f"{name}_preview.png", tile_sheet(rgb, 2, 1024))
    save_png(d / f"{name}_channels.png", np.concatenate([np.repeat(rgb[::4, ::4, i:i + 1], 3, -1) for i in range(3)], 1))
    log(f"[macro] wrote {d}  channel mean/std {stats}")
    return meta
