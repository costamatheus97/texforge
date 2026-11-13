"""`texforge decal`: decal-on-plain-background image -> BaseColor + opacity + Normal + ORM for UE deferred decals."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import gen as genmod
from . import maps, prompts
from .common import blur, luminance, load_image, resize, save_png, smoothstep

# surface presets: roughness inside the decal, normal strength, relief depth (cm)
SURFACES = {
    "rust":    dict(rough=0.85, normal=2.5, depth=0.2),
    "damp":    dict(rough=0.18, normal=0.5, depth=0.0),   # wet = glossy darkening, no relief
    "crack":   dict(rough=0.90, normal=5.0, depth=0.8),
    "dirt":    dict(rough=0.95, normal=1.5, depth=0.1),
    "generic": dict(rough=0.80, normal=2.0, depth=0.2),
}


def extract_alpha(img: np.ndarray, border_frac: float = 0.03) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (alpha, un-premultiplied colour, bg colour). Alpha comes from the distance to the plain background."""
    h, w = img.shape[:2]
    b = max(2, int(border_frac * min(h, w)))
    border = np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3), img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)])
    bg = np.median(border, 0).astype(np.float32)
    d = np.sqrt(((img - bg) ** 2).sum(-1))
    d = cv2.GaussianBlur(d, (0, 0), 0.8)
    floor = float(np.percentile(np.sqrt(((border - bg) ** 2).sum(-1)), 99)) * 1.5 + 0.015
    top = max(float(np.percentile(d, 99.5)), floor + 0.05)
    a = smoothstep(floor, top * 0.85, d)
    # no hard rectangle: fade to 0 over the outer 6% of the image
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    edge = np.minimum(np.minimum(xx, w - 1 - xx) / w, np.minimum(yy, h - 1 - yy) / h)
    a = a * smoothstep(0.015, 0.06, edge)
    a = np.clip(a, 0, 1).astype(np.float32)
    # un-premultiply against the background, then push colour into transparent texels (no halo in mips)
    col = (img - (1 - a[..., None]) * bg) / np.maximum(a[..., None], 1e-3)
    col = np.clip(col, 0, 1)
    solid = a > 0.25
    if solid.any():
        _, (iy, ix) = ndi.distance_transform_edt(~solid, return_indices=True)
        col = np.where(solid[..., None], col, col[iy, ix])
    return a, col.astype(np.float32), bg


def make_decal(prompt: str, name: str, out_root: Path, image: str | None = None, background: str = "white",
               size: int = 2048, world_cm: float = 150.0, surface: str = "generic", log=print) -> dict:
    out_dir = Path(out_root) / name
    if image is None:
        files = genmod.generate(prompts.decal_prompt(prompt, background), out_dir, name, n=1)
        image = str(files[0])
    img = load_image(image)
    a, col, bg = extract_alpha(img)
    # crop to the mark (+ margin), then scale so the longest side is `size`
    ys, xs = np.where(a > 0.02)
    if len(ys) == 0:
        raise RuntimeError("no decal content found: is the background plain white/black?")
    pad = int(0.05 * max(img.shape[:2]))
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
    a, col = a[y0:y1, x0:x1], col[y0:y1, x0:x1]
    h, w = a.shape
    s = size / max(h, w)
    oh, ow = max(8, int(round(h * s))), max(8, int(round(w * s)))
    a = np.clip(resize(a, (oh, ow)), 0, 1)
    col = np.clip(resize(col, (oh, ow)), 0, 1)
    a[:2] = a[-2:] = 0
    a[:, :2] = a[:, -2:] = 0
    pr = SURFACES[surface]
    lum = luminance(col)
    # height: raised where the mark is dense, plus its own luminance detail; flat (0) outside
    detail = np.clip((lum - blur(lum, 0.02 * max(oh, ow), "xy")) * 4 + 0.5, 0, 1)
    hgt = blur(a, 1.5, "xy") * (0.6 + 0.4 * detail) if pr["depth"] > 0 else blur(a, 1.5, "xy") * 0.5
    hgt = np.clip(hgt, 0, 1).astype(np.float32)
    normal = maps.normal_map(hgt, pr["normal"], wrap="xy")
    rough = np.clip(pr["rough"] + 0.12 * (0.5 - lum), 0.1, 1.0).astype(np.float32)
    ao = np.ones_like(rough)
    orm = maps.pack_orm(ao, rough, 0.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgba = np.concatenate([col, a[..., None]], -1)
    save_png(out_dir / f"{name}_BaseColor.png", rgba)
    save_png(out_dir / f"{name}_Opacity.png", a)
    save_png(out_dir / f"{name}_Normal.png", normal)
    save_png(out_dir / f"{name}_ORM.png", orm)
    save_png(out_dir / f"{name}_Height.png", hgt)
    sidecar = {
        "name": name, "kind": "decal", "type": surface, "size": [ow, oh],
        "world_cm": [float(world_cm), float(world_cm) * oh / ow],
        "normal_convention": "directx", "height_depth_cm": pr["depth"], "tileable": False, "tile_axes": "",
        "files": {"basecolor": f"{name}_BaseColor.png", "normal": f"{name}_Normal.png", "orm": f"{name}_ORM.png",
                  "height": f"{name}_Height.png", "opacity": f"{name}_Opacity.png"},
        "orm_channels": "R=AO(1), G=Roughness, B=Metallic(0)",
        "params": {"source": str(image), "background_rgb": [round(float(v), 3) for v in bg], "surface": surface,
                   "alpha": "smoothstep distance-from-background, 6% border fade, colour un-premultiplied and dilated"},
        "notes": [],
    }
    (out_dir / f"{name}.texforge.json").write_text(json.dumps(sidecar, indent=2))
    # preview: the decal over a mid-grey and a brick-ish checker so the soft edges can be judged
    prev = np.concatenate([
        a[..., None] * col + (1 - a[..., None]) * 0.5,
        a[..., None] * col + (1 - a[..., None]) * np.array([0.55, 0.3, 0.25], np.float32),
        np.repeat(a[..., None], 3, -1)], 1)
    save_png(out_dir / f"{name}_preview.png", resize(prev, (prev.shape[0] // 2, prev.shape[1] // 2)))
    return sidecar
