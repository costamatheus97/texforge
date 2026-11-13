"""Height-blend preview of a kit's ground layers, in Python, so the blend can be judged without Unreal.

Same maths as the landscape material (unreal/create_landscape_material.py):
    a_i = h_i + w_i          (h_i layer height, normalised to mean 0.5 / std 0.18 by `kit`; w_i = painted weight, sum 1)
    m   = max_i(a_i) - depth
    w'_i = max(a_i - m, 0) / sum(...)
`depth` is the sharpness: small = hard, height-driven transitions; large -> approaches the plain linear blend.
The linear version just uses w_i. Weights come from procedural fbm masks (soft, like a hand-painted landscape).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .common import load_image, luminance, resize, save_png
from .macro import fbm
from .qa import render_lit

PALETTE = np.array([[.55, .38, .22], [.75, .45, .15], [.2, .65, .2], [.75, .8, .2], [.6, .6, .65], [.85, .3, .2], [.5, .3, .6]], np.float32)


def height_blend(h: np.ndarray, w: np.ndarray, depth: float) -> np.ndarray:
    """h, w: (N,H,W). -> new weights (N,H,W) summing to 1."""
    a = h + w
    m = a.max(0, keepdims=True) - depth
    wp = np.maximum(a - m, 0)
    return wp / np.maximum(wp.sum(0, keepdims=True), 1e-6)


def paint_weights(n_layers: int, px: int, size_cm: float, seed: int, beta: float = 3.2) -> np.ndarray:
    """Soft procedural weight maps (N,px,px), sum 1: independent fbm per layer, soft-max'd (transitions ~0.3-0.6 m)."""
    rng = np.random.default_rng(seed)
    fields = np.stack([fbm(px, rng, (2, 11), 2.0) + 0.35 * fbm(px, rng, (11, 45), 1.8) for _ in range(n_layers)])
    z = beta * (fields - fields.mean(0, keepdims=True)) / fields.std()
    z -= z.max(0, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(0, keepdims=True)).astype(np.float32)


def _load_tile(d: Path, name: str, kind: str, px_tile: int, offset: tuple[int, int]) -> np.ndarray:
    a = load_image(d / name / f"{name}_{kind}.png")
    if kind == "Height":
        a = luminance(a)[..., None]
    a = cv2.resize(a, (px_tile, px_tile), interpolation=cv2.INTER_AREA)
    if a.ndim == 2:
        a = a[..., None]
    a = np.roll(a, offset, (0, 1))                                  # each layer starts at a different phase
    return np.tile(a, (2, 2, 1)).astype(np.float32)


def _label(img, text):
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 8 * len(text) + 8, 16], fill=(0, 0, 0))
    dr.text((4, 2), text, fill=(255, 255, 0))
    return np.asarray(im, np.float32) / 255


def run(kit_path: str | Path, log=print) -> dict:
    from .kit import load_kit
    kit = load_kit(kit_path)
    cfg = kit["blend"]
    out = kit["out_dir"] / "blend"
    out.mkdir(parents=True, exist_ok=True)
    layers = [l["name"] for l in kit["layers"] if l.get("role", "ground") == "ground"]
    n = len(layers)
    px = cfg["px"]
    tile_cm = kit["world_cm"]
    px_tile = int(round(px * tile_cm / cfg["size_cm"]))
    rng = np.random.default_rng(cfg.get("seed", 3))
    offs = [tuple(int(v) for v in rng.integers(0, px_tile, 2)) for _ in layers]
    w = paint_weights(n, px, cfg["size_cm"], cfg.get("seed", 3))
    H = np.stack([_load_tile(kit["out_dir"], l, "Height", px_tile, offs[i])[..., 0] for i, l in enumerate(layers)])
    wh = height_blend(H, w, cfg["sharpness"])
    stats = {"layers": layers, "depth": cfg["sharpness"],
             "mean_max_weight_linear": round(float(w.max(0).mean()), 3), "mean_max_weight_heightblend": round(float(wh.max(0).mean()), 3),
             "pct_px_dominant_layer_changed": round(float((w.argmax(0) != wh.argmax(0)).mean() * 100), 2),
             "area_share_linear": [round(float((w.argmax(0) == i).mean()), 3) for i in range(n)],
             "area_share_heightblend": [round(float((wh.argmax(0) == i).mean()), 3) for i in range(n)],
             "px_per_cm": round(px / cfg["size_cm"], 3)}
    comp = {}
    for tag, ww in (("height", wh), ("linear", w)):
        comp[tag] = {"base": np.zeros((px, px, 3), np.float32), "nrm": np.zeros((px, px, 3), np.float32), "orm": np.zeros((px, px, 3), np.float32)}
    for i, l in enumerate(layers):
        base = _load_tile(kit["out_dir"], l, "BaseColor", px_tile, offs[i])
        nrm = _load_tile(kit["out_dir"], l, "Normal", px_tile, offs[i]) * 2 - 1
        orm = _load_tile(kit["out_dir"], l, "ORM", px_tile, offs[i])
        lin = np.where(base <= 0.04045, base / 12.92, ((base + 0.055) / 1.055) ** 2.4)      # blend albedo in linear light
        for tag, ww in (("height", wh), ("linear", w)):
            comp[tag]["base"] += lin * ww[i][..., None]
            comp[tag]["nrm"] += nrm * ww[i][..., None]
            comp[tag]["orm"] += orm * ww[i][..., None]
        del base, nrm, orm, lin
    for tag in comp:
        b = np.clip(comp[tag]["base"], 0, 1)
        comp[tag]["base"] = np.where(b <= 0.0031308, b * 12.92, 1.055 * np.power(b, 1 / 2.4) - 0.055).astype(np.float32)
        v = comp[tag]["nrm"]
        v /= np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-4)
        comp[tag]["nrm"] = (v * 0.5 + 0.5).astype(np.float32)
    # weight maps, colour-coded
    def idmap(ww):
        return np.clip((ww[..., None] * PALETTE[:n, None, None, :].transpose(0, 1, 2, 3).reshape(n, 1, 1, 3)).sum(0), 0, 1)
    wmap = np.concatenate([_label(idmap(w), "painted weights"), _label(idmap(wh), "after height blend")], 1)
    legend = " | ".join(f"{l}" for l in layers)
    save_png(out / "forest_weights.png", resize(wmap, (px // 2, px)))
    # unlit + lit composites
    unlit = np.concatenate([_label(comp["height"]["base"], "HEIGHT BLEND"), _label(comp["linear"]["base"], "LINEAR BLEND")], 1)
    save_png(out / "forest_blend_albedo.png", resize(unlit, (px // 2, px * 2 // 2 * 1)) if False else unlit)
    rows = []
    for L in cfg["lights"]:
        rows.append(np.concatenate([_label(render_lit(comp[t]["base"], comp[t]["nrm"], comp[t]["orm"], L, tiles=1, px=px), f"{t.upper()} BLEND  light {L}") for t in ("height", "linear")], 1))
    save_png(out / "forest_blend_lit.png", np.concatenate(rows, 0))
    # 1:1 crop of the transition-richest area
    ent = -(wh * np.log(np.maximum(wh, 1e-6))).sum(0)
    k = 384
    ent_c = cv2.boxFilter(ent.astype(np.float32), -1, (k, k))
    y, x = np.unravel_index(int(np.argmax(ent_c[k // 2: px - k // 2, k // 2: px - k // 2])), (px - k, px - k))
    crops = []
    for t in ("height", "linear"):
        crops.append(_label(render_lit(comp[t]["base"], comp[t]["nrm"], comp[t]["orm"], cfg["lights"][0], tiles=1, px=px)[y:y + k, x:x + k], f"{t.upper()} (crop, 1:1)"))
    save_png(out / "forest_blend_crop.png", np.concatenate(crops, 1))
    # per layer swatch strip with the kit height numbers
    stats["transition_crop_xy_px"] = [int(x), int(y)]
    (out / "forest_blend_stats.json").write_text(json.dumps(stats, indent=2))
    log(f"[blend] {out}  mean max weight linear {stats['mean_max_weight_linear']} -> height blend {stats['mean_max_weight_heightblend']}; "
        f"{stats['pct_px_dominant_layer_changed']}% of pixels change dominant layer")
    return stats
