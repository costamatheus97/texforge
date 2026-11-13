"""QA: seam metric, 3x3 tiled preview sheets, 1:1 corner crops and a lit-plane preview (GIF)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .common import load_image, save_png, resize, srgb_to_lin, lin_to_srgb, luminance

# Failure thresholds on the seam ratio (seam discontinuity / 90th percentile of the same step measured at interior positions; <=~1 means invisible).
THRESH_COLOR = 1.15
THRESH_GRAD = 1.15


def _ratios(a: np.ndarray, axis: int) -> tuple[float, float, float]:
    """(colour, gradient, colour_p90) seam ratios for the wrap-around border across `axis` (1 = x, 0 = y).

    colour: for band widths of 1 px .. ~1.5% of the tile, |mean of s px left of the border - mean of s px
            right of it|, divided by the 99.5th percentile of that same step over all interior positions
            (worst band width wins). <= 1 means the seam is no bigger than the largest step the texture
            itself contains, i.e. it cannot be told apart from natural content. The percentile reference
            (not the mean) keeps structured patterns fair: a seam that lands exactly on a joint is
            compared with the other joints, not with flat brick/tile faces. colour_p90 is the stricter
            informational variant using the 90th percentile.
    gradient: same idea on the second difference (slope jump) at 1 px.
    """
    if axis == 0:
        a = a.transpose(1, 0, 2)
    a = a.astype(np.float64)
    W = a.shape[1]
    r995, r90 = [], []
    cs = np.concatenate([np.zeros_like(a[:, :1]), np.cumsum(a, axis=1)], axis=1)
    for s in sorted({1, max(2, W // 256), max(3, W // 64)}):
        mean_left = (cs[:, s:W - s + 1] - cs[:, :W - 2 * s + 1]) / s      # window [i, i+s)
        mean_right = (cs[:, 2 * s:W + 1] - cs[:, s:W - s + 1]) / s        # window [i+s, i+2s)
        prof = np.abs(mean_right - mean_left).mean((0, 2))                  # interior step size per position
        seam = np.abs(a[:, :s].mean(1) - a[:, W - s:].mean(1)).mean()
        r995.append(seam / max(float(np.percentile(prof, 99.5)), 1e-9))
        r90.append(seam / max(float(np.percentile(prof, 90)), 1e-9))
    d = np.diff(a, axis=1)
    prof2 = np.abs(np.diff(d, axis=1)).mean((0, 2))
    seam1 = a[:, 0] - a[:, -1]
    g = 0.5 * (np.abs(seam1 - d[:, -1]).mean() + np.abs(d[:, 0] - seam1).mean())
    return float(max(r995)), float(g / max(float(np.percentile(prof2, 99.5)), 1e-9)), float(max(r90))


def seam_metrics(a: np.ndarray) -> dict:
    """a: HxW or HxWxC float. See _ratios; 'color'/'grad' are the worst of the two axes."""
    if a.ndim == 2:
        a = a[..., None]
    cx, gx, px = _ratios(a, 1)
    cy, gy, py = _ratios(a, 0)
    return {"color_x": round(cx, 3), "color_y": round(cy, 3), "grad_x": round(gx, 3), "grad_y": round(gy, 3),
            "color": round(max(cx, cy), 3), "grad": round(max(gx, gy), 3), "color_p90": round(max(px, py), 3)}


def tile_sheet(a: np.ndarray, n: int = 3, tile_px: int = 512) -> np.ndarray:
    t = resize(a, (tile_px, tile_px)) if a.shape[0] != tile_px or a.shape[1] != tile_px else a
    if t.ndim == 2:
        t = t[..., None]
    return np.tile(t, (n, n, 1))


def corner_crop(a: np.ndarray, half: int = 256, mark: bool = True) -> np.ndarray:
    """1:1 crop centred on the point where four tiles meet (shows 1-3 px seams the downscaled sheet hides).
    Red ticks in a margin point at the seam lines so they can be located without drawing on the pixels."""
    h, w = a.shape[:2]
    half = min(half, h // 2, w // 2)
    t = np.tile(a if a.ndim == 3 else np.repeat(a[..., None], 3, -1), (2, 2, 1))
    c = t[h - half:h + half, w - half:w + half]
    if not mark:
        return c
    m = 12
    out = np.ones((2 * half + 2 * m, 2 * half + 2 * m, 3), np.float32)
    out[m:-m, m:-m] = c[..., :3]
    out[:m, half + m - 6:half + m + 6] = (1, 0, 0)
    out[-m:, half + m - 6:half + m + 6] = (1, 0, 0)
    out[half + m - 6:half + m + 6, :m] = (1, 0, 0)
    out[half + m - 6:half + m + 6, -m:] = (1, 0, 0)
    return out


def render_lit(albedo_srgb, normal, orm, light, tiles=2, px=384, ambient=0.18, spec_k=1.0,
              light_color=(1.0, 1.0, 1.0), ambient_color=(1.0, 1.0, 1.0), exposure=1.0):
    """Blinn-Phong plane preview. `normal` is DirectX/Unreal (G down): in image axes N = (x, y_down, z),
    so a light direction given in image axes (x right, y down, z toward viewer) is used as-is."""
    alb = srgb_to_lin(resize(albedo_srgb, (px // tiles, px // tiles)))
    n = resize(normal, (px // tiles, px // tiles)) * 2 - 1
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
    o = resize(orm, (px // tiles, px // tiles))
    ao, rough = o[..., 0:1], o[..., 1:2]
    metal = o[..., 2:3]
    L = np.array(light, np.float32); L /= np.linalg.norm(L)
    V = np.array([0, 0, 1], np.float32)
    H = L + V; H /= np.linalg.norm(H)
    ndl = np.clip((n * L).sum(-1, keepdims=True), 0, 1)
    ndh = np.clip((n * H).sum(-1, keepdims=True), 0, 1)
    shin = 2.0 / np.maximum(rough ** 4, 1e-3) - 2.0
    spec = (ndh ** np.clip(shin, 2, 400)) * (1 - rough) * 0.5 * spec_k
    lc = np.array(light_color, np.float32)
    ac = np.array(ambient_color, np.float32)
    col = (alb * (1 - metal) * (ndl * (1 - ambient) * lc + ambient * ac) * ao + (spec + 0.04 * metal) * ndl * lc
           + alb * metal * spec) * exposure
    img = lin_to_srgb(np.clip(col, 0, 1))
    return np.tile(img, (tiles, tiles, 1))


def lit_gif(base: np.ndarray, normal: np.ndarray, orm: np.ndarray, path: Path, frames: int = 24, px: int = 384,
            height: np.ndarray | None = None):
    ims = []
    for i in range(frames):
        ang = 2 * np.pi * i / frames
        ims.append(render_lit(base, normal, orm, (np.cos(ang) * 0.75, np.sin(ang) * 0.75, 0.55), px=px))
    pil = [Image.fromarray((np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)) for f in ims]
    pil[0].save(path, save_all=True, append_images=pil[1:], duration=80, loop=0)
    # contact sheet: three light directions on one tile at 512 px, then AO / roughness / height
    one = [render_lit(base, normal, orm, d, tiles=1, px=512) for d in ((-0.8, -0.3, 0.5), (0.0, -0.85, 0.5), (0.8, 0.3, 0.5))]
    g = lambda m: np.repeat(resize(m, (512, 512))[..., None], 3, -1)  # noqa: E731
    row2 = [g(orm[..., 0]), g(orm[..., 1]), g(height) if height is not None else g(orm[..., 2])]
    return np.concatenate([np.concatenate(one, 1), np.concatenate(row2, 1)], 0)


def _trim_check(out_dir: Path, name: str, meta: dict) -> dict:
    """Trim sheets tile along x only: per-strip seam ratios, a 3x horizontal repeat and 1:1 crops at the wrap."""
    cdir = out_dir / "check"
    base = load_image(out_dir / f"{name}_BaseColor.png")
    size = base.shape[0]
    report: dict = {"name": name, "kind": "trim", "strips": {}, "before": None}
    fails = []
    for st in meta["strips"]:
        y0, y1 = int(st["v0"] * size) + 2, int(st["v1"] * size) - 2
        band = base[y0:y1]
        r = _ratios(band, 1)
        report["strips"][st["name"]] = {"color_x": round(r[0], 3), "grad_x": round(r[1], 3), "color_p90": round(r[2], 3)}
        if r[0] > THRESH_COLOR:
            fails.append(f"strip '{st['name']}' colour seam {r[0]:.2f} > {THRESH_COLOR}")
        if r[1] > THRESH_GRAD:
            fails.append(f"strip '{st['name']}' gradient seam {r[1]:.2f} > {THRESH_GRAD}")
    three = np.tile(base, (1, 3, 1))
    save_png(cdir / f"{name}_after_3x1.png", resize(three, (size // 3, size)))
    half = 300
    crop = three[:, size - half:size + half]          # the wrap-around border, full height, 1:1
    save_png(cdir / f"{name}_after_seam_crop.png", np.concatenate([crop[:size // 2], crop[size // 2:]], 1))
    report["fails"] = fails
    report["passed"] = not fails
    (cdir / f"{name}_check.json").write_text(json.dumps(report, indent=2))
    return report


def check(out_dir: Path, name: str, wrap: str = "xy") -> dict:
    """Write check artefacts into <out_dir>/check/ and return the report (also saved as JSON)."""
    out_dir = Path(out_dir)
    cdir = out_dir / "check"
    cdir.mkdir(exist_ok=True)
    meta_p = out_dir / f"{name}.texforge.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    if meta.get("kind") == "trim":
        return _trim_check(out_dir, name, meta)
    maps = {k: out_dir / f"{name}_{k}.png" for k in ("BaseColor", "Normal", "ORM", "Height")}
    for k, p in maps.items():
        if not p.exists():
            raise FileNotFoundError(p)
    base = load_image(maps["BaseColor"])
    normal = load_image(maps["Normal"])
    orm = load_image(maps["ORM"])
    height = luminance(load_image(maps["Height"]))
    report: dict = {"name": name, "after": {}, "before": None}
    for k, arr in (("BaseColor", base), ("Normal", normal), ("ORM", orm), ("Height", height)):
        report["after"][k] = seam_metrics(arr)
    save_png(cdir / f"{name}_after_3x3.png", tile_sheet(base))
    save_png(cdir / f"{name}_after_corner.png", corner_crop(base))
    save_png(cdir / f"{name}_after_normal_3x3.png", tile_sheet(normal))
    mask_p = out_dir / f"{name}_Mask.png"
    if mask_p.exists():                       # sub-material mask (organic): seam metric on RGBA, shown as a sheet
        from PIL import Image
        m = np.asarray(Image.open(mask_p).convert("RGBA"), np.float32) / 255
        report["after"]["Mask"] = seam_metrics(m)
        save_png(cdir / f"{name}_after_mask_3x3.png", tile_sheet(m[..., :3]))
    src = out_dir / f"{name}_source.png"
    if src.exists():
        s = load_image(src)
        report["before"] = {"BaseColor": seam_metrics(s)}
        save_png(cdir / f"{name}_before_3x3.png", tile_sheet(s))
        save_png(cdir / f"{name}_before_corner.png", corner_crop(s))
        save_png(cdir / f"{name}_before_after.png",
                 np.concatenate([tile_sheet(s, 3, 384), np.ones((1152, 16, 3), np.float32), tile_sheet(base, 3, 384)], 1))
    contact = lit_gif(base, normal, orm, cdir / f"{name}_lit.gif", height=height)
    save_png(cdir / f"{name}_lit_sheet.png", contact)
    a = report["after"]
    fails = []
    if a["BaseColor"]["color"] > THRESH_COLOR:
        fails.append(f"BaseColor colour seam {a['BaseColor']['color']} > {THRESH_COLOR}")
    if a["BaseColor"]["grad"] > THRESH_GRAD:
        fails.append(f"BaseColor gradient seam {a['BaseColor']['grad']} > {THRESH_GRAD}")
    for k in ("Normal", "Height", "ORM", "Mask"):
        if k in a and a[k]["color"] > THRESH_COLOR + 0.15:
            fails.append(f"{k} seam {a[k]['color']} > {THRESH_COLOR + 0.15:.2f}")
    report["fails"] = fails
    report["passed"] = not fails
    (cdir / f"{name}_check.json").write_text(json.dumps(report, indent=2))
    return report
