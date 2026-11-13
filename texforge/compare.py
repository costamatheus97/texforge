"""`texforge compare OLD NEW`: old vs new material set, lit by a low grazing warm light from the side.

Rows: 3x3 tiled lit view (what a camera sees) | 1:1 crop of the lit 2048 px texture | flat albedo, sub-material mask,
roughness. `--old-tint r,g,b` multiplies the OLD albedo in linear light by the tint the game applies to it today, so the
left column is what the game shows now and the right column is the new set with a neutral tint.
Light: sun pitch -28 deg -> L = (0.88, 0.1, 0.47) in image axes (x right, y down, z toward the viewer), warm colour,
cool sky ambient."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import qa
from .common import load_image, lin_to_srgb, resize, save_png, srgb_to_lin

SUN = (-0.88, 0.10, 0.47)
SUN_COLOR = (1.0, 0.78, 0.55)
SKY = (0.62, 0.78, 1.0)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 7 * len(text) + 10, 18], fill=(0, 0, 0))
    d.text((5, 3), text, fill=(255, 255, 0))
    return np.asarray(im, np.float32) / 255


def _load_set(d: Path, name: str, tint=None):
    base = load_image(d / f"{name}_BaseColor.png")
    if tint is not None:
        base = lin_to_srgb(np.clip(srgb_to_lin(base) * np.array(tint, np.float32), 0, 1))
    mask_p = d / f"{name}_Mask.png"
    mask = np.asarray(Image.open(mask_p).convert("RGBA"), np.float32) / 255 if mask_p.exists() else None
    return base, load_image(d / f"{name}_Normal.png"), load_image(d / f"{name}_ORM.png"), mask


def mask_colour(mask: np.ndarray) -> np.ndarray:
    """False colour: living = green, dead = orange, soil = blue, twigs = white (stones / fines etc. reuse the first channels)."""
    pal = np.array([[0.15, 0.75, 0.2], [0.95, 0.55, 0.1], [0.15, 0.3, 0.95], [1, 1, 1]], np.float32)
    return np.clip((mask[..., None] * pal[None, None]).sum(2) if mask.ndim == 3 and mask.shape[2] == 4 else mask[..., :3], 0, 1)


def compare(old_dir: Path, old: str, new_dir: Path, new: str, out: Path, old_tint=None, new_tint=None,
            exposure: float = 1.6, crop: int = 640, tile_px: int = 1080) -> dict:
    sets = {"old": _load_set(old_dir, old, old_tint), "new": _load_set(new_dir, new, new_tint)}
    names = {"old": f"OLD {old}" + (f"  x game tint {tuple(round(t, 2) for t in old_tint)}" if old_tint else ""),
             "new": f"NEW {new}" + (f"  x tint {tuple(round(t, 2) for t in new_tint)}" if new_tint else "  (neutral tint)")}
    kw = dict(light_color=SUN_COLOR, ambient_color=SKY, exposure=exposure)
    cols = {}
    for k, (base, nrm, orm, mask) in sets.items():
        three = qa.render_lit(base, nrm, orm, SUN, tiles=3, px=tile_px, **kw)
        full = qa.render_lit(base, nrm, orm, SUN, tiles=1, px=base.shape[0], **kw)
        c = base.shape[0] // 2
        one = full[c - crop // 2:c + crop // 2, c - crop // 2:c + crop // 2]
        cols[k] = (three, one)
    gap = np.ones((tile_px, 12, 3), np.float32) * 0.1
    row1 = np.concatenate([_label(cols["old"][0], names["old"] + "  lit 3x3 tiles (2 m each)"), gap,
                           _label(cols["new"][0], names["new"] + "  lit 3x3 tiles (2 m each)")], 1)
    W = row1.shape[1]
    # row 2: 1:1 crops, centred in a dark band
    band = np.ones((crop + 12, W, 3), np.float32) * 0.1
    xo = (tile_px - crop) // 2
    band[6:6 + crop, xo:xo + crop] = _label(cols["old"][1], "OLD 1:1 crop (lit)")
    band[6:6 + crop, tile_px + 12 + xo:tile_px + 12 + xo + crop] = _label(cols["new"][1], "NEW 1:1 crop (lit)")
    # row 3: flat albedo old | new albedo | mask | roughness old | roughness new (each ~ W/5)
    cw = (W - 4 * 8) // 5
    cells = [_label(resize(sets["old"][0], (cw, cw)), "old albedo (tinted)"), _label(resize(sets["new"][0], (cw, cw)), "new albedo"),
             _label(resize(mask_colour(sets["new"][3]), (cw, cw)) if sets["new"][3] is not None else np.zeros((cw, cw, 3), np.float32),
                    "new mask: G living / O dead / B soil / W twigs"),
             _label(np.repeat(resize(sets["old"][2][..., 1], (cw, cw))[..., None], 3, -1), "old roughness"),
             _label(np.repeat(resize(sets["new"][2][..., 1], (cw, cw))[..., None], 3, -1), "new roughness")]
    sep = np.ones((cw, 8, 3), np.float32) * 0.1
    row3 = np.concatenate([x for c in cells for x in (c, sep)][:-1], 1)
    if row3.shape[1] < W:
        row3 = np.concatenate([row3, np.ones((cw, W - row3.shape[1], 3), np.float32) * 0.1], 1)
    sheet = np.concatenate([row1, band, row3], 0)
    save_png(out, sheet)
    return {"out": str(out), "size": list(sheet.shape[:2][::-1])}
