"""Litter atlas: cutout leaves / twigs / pinecones / bark chips from one codex sprite sheet on a flat chroma-key
background -> clean alpha (key + despill + edge erosion + colour decontamination) -> packed atlas with
BaseColor+alpha, Normal and ORM, plus a JSON with every item's uv rectangle and real-world size.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import color as skcolor

from . import gen as genmod
from . import maps, prompts
from .common import blur, load_image, luminance, resize, save_png, smoothstep
from .qa import render_lit


def key_alpha(rgb: np.ndarray, lo: float = 14.0, hi: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    """alpha from the distance (Lab a/b, plus a little L) to the border-median background colour. -> (alpha, bg rgb)"""
    border = np.concatenate([rgb[:10].reshape(-1, 3), rgb[-10:].reshape(-1, 3), rgb[:, :10].reshape(-1, 3), rgb[:, -10:].reshape(-1, 3)])
    bg = np.median(border, 0)
    lab = skcolor.rgb2lab(rgb)
    lab_bg = skcolor.rgb2lab(bg[None, None, :])[0, 0]
    d = np.sqrt(((lab[..., 1:] - lab_bg[1:]) ** 2).sum(-1) + (0.35 * (lab[..., 0] - lab_bg[0])) ** 2)
    return smoothstep(lo, hi, d).astype(np.float32), bg.astype(np.float32)


def despill(rgb: np.ndarray) -> np.ndarray:
    """Magenta-key despill: remove the part of R and B that exceeds G together (min(R,B) - G)."""
    s = np.maximum(np.minimum(rgb[..., 0], rgb[..., 2]) - rgb[..., 1], 0)[..., None]
    return np.clip(rgb - s * np.array([1, 0, 1], np.float32), 0, 1)


def clean_item(rgb: np.ndarray, a: np.ndarray, erode: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """alpha eroded by `erode` px and smoothed; colour taken from the opaque interior near the edge (no key fringe)."""
    a = ndi.grey_erosion(a, size=(2 * erode + 1, 2 * erode + 1))
    a = cv2.GaussianBlur(a, (0, 0), 0.8)
    a = np.clip((a - 0.08) / 0.84, 0, 1)
    rgb = despill(rgb)
    solid = a > 0.97
    if solid.any():
        _, (iy, ix) = ndi.distance_transform_edt(~solid, return_indices=True)
        rgb = np.where(solid[..., None], rgb, rgb[iy, ix])
        # outside the object entirely, colour is only needed for mip bleeding: nearest solid colour (already done)
    return rgb.astype(np.float32), a.astype(np.float32)


def find_items(alpha: np.ndarray, min_frac: float = 0.0004) -> list[tuple[int, int, int, int]]:
    m = (alpha > 0.5).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    area_min = min_frac * m.size
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area >= area_min:
            boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def label_rows(boxes: list, rows: list) -> list[str]:
    """Cluster the boxes into len(rows) horizontal rows by their vertical centres (reading order) and name them."""
    if not boxes:
        return []
    cy = np.array([b[1] + b[3] / 2 for b in boxes])
    order = np.argsort(cy)
    gaps = np.diff(cy[order])
    kinds = ["item"] * len(boxes)
    if len(rows) > 1 and len(boxes) > 1:
        cuts = np.sort(np.argsort(gaps)[-(len(rows) - 1):])
        grp = np.zeros(len(boxes), int)
        g = 0
        for k, idx in enumerate(order):
            grp[idx] = g
            if k in cuts:
                g += 1
        if g + 1 == len(rows):
            kinds = [rows[grp[i]][0] for i in range(len(boxes))]
    return kinds


def shelf_pack(sizes: list[tuple[int, int]], atlas: int, pad: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][1])
    pos = [(0, 0)] * len(sizes)
    x = y = rowh = 0
    for i in order:
        w, h = sizes[i][0] + 2 * pad, sizes[i][1] + 2 * pad
        if x + w > atlas:
            x, y, rowh = 0, y + rowh, 0
        if y + h > atlas or w > atlas:
            return None
        pos[i] = (x + pad, y + pad)
        x += w
        rowh = max(rowh, h)
    return pos


def item_maps(rgb: np.ndarray, a: np.ndarray, strength: float = 2.2) -> tuple[np.ndarray, np.ndarray]:
    """(normal, orm) for one cutout: a thin raised sheet with rounded edges plus the luminance relief."""
    n = max(rgb.shape[:2])
    dist = ndi.distance_transform_edt(a > 0.5)
    edge = smoothstep(0, max(3.0, 0.02 * n), dist)
    lum = luminance(rgb)
    fine = maps._norm(blur(lum, 0.8, "") - blur(lum, 0.02 * n, ""))
    h = (0.55 * edge + 0.25 * fine * (a > 0.5) + 0.2 * a).astype(np.float32)
    nm = maps.normal_map(h, strength, "")
    nm = a[..., None] * nm + (1 - a[..., None]) * np.array([0.5, 0.5, 1.0], np.float32)
    ao = np.clip(1 - 0.6 * np.maximum(blur(h, 0.01 * n, "") - h, 0) / max(float(np.percentile(np.maximum(blur(h, 0.01 * n, "") - h, 0), 99)), 1e-4), 0.35, 1)
    rough = np.clip(0.72 + 0.2 * (1 - fine), 0.5, 0.95)
    orm = np.stack([np.where(a > 0.5, ao, 1.0), np.where(a > 0.5, rough, 0.8), np.zeros_like(a)], -1)
    return nm.astype(np.float32), orm.astype(np.float32)


def _checker(h: int, w: int, c: int = 24) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w]
    v = (((yy // c) + (xx // c)) % 2).astype(np.float32) * 0.18 + 0.42
    return np.repeat(v[..., None], 3, -1)


def _over(bg: np.ndarray, fg: np.ndarray, a: np.ndarray) -> np.ndarray:
    return bg * (1 - a[..., None]) + fg * a[..., None]


def build(kit_path: str, image: str | None = None, gen: bool = True, log=print) -> int:
    from .kit import load_kit
    kit = load_kit(kit_path)
    cfg = kit["atlas"]
    name = cfg["name"]
    d = kit["out_dir"] / name
    d.mkdir(parents=True, exist_ok=True)
    sheet_p = Path(image) if image else d / f"{name}_sheet_src_1.png"
    runs = 0
    if not sheet_p.exists():
        if not gen:
            log("[atlas] no sheet image and --no-gen")
            return 1
        prompt = prompts.atlas_prompt([tuple(r) for r in cfg["rows"]])
        (d / f"{name}_prompt.txt").write_text(prompt)
        try:
            genmod.generate(prompt, d, f"{name}_sheet", n=1, parallel=1)
            runs = 1
        except RuntimeError as e:
            print(f"[atlas] {e}", file=sys.stderr)
            genmod.print_manual(prompt)
            return 2
    rgb0 = load_image(sheet_p)
    alpha0, bg = key_alpha(rgb0)
    boxes = find_items(alpha0)
    kinds = label_rows(boxes, cfg["rows"])
    # row-major order
    idx = sorted(range(len(boxes)), key=lambda i: (round((boxes[i][1] + boxes[i][3] / 2) / (rgb0.shape[0] / max(len(cfg["rows"]), 1))), boxes[i][0]))
    px_per_cm_src = rgb0.shape[1] / cfg["sheet_cm"]
    items = []
    cnt: dict[str, int] = {}
    for i in idx:
        x, y, w, h = boxes[i]
        m = 10
        x0, y0, x1, y1 = max(x - m, 0), max(y - m, 0), min(x + w + m, rgb0.shape[1]), min(y + h + m, rgb0.shape[0])
        rgb, a = clean_item(rgb0[y0:y1, x0:x1], alpha0[y0:y1, x0:x1])
        # other objects that fall inside this crop must not leak in: keep the component of this box only
        comp = cv2.dilate((a > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8))
        _, lab = cv2.connectedComponents(comp)
        cx, cy = x + w // 2 - x0, y + h // 2 - y0
        sel = lab == lab[min(max(cy, 0), lab.shape[0] - 1), min(max(cx, 0), lab.shape[1] - 1)]
        if lab[min(max(cy, 0), lab.shape[0] - 1), min(max(cx, 0), lab.shape[1] - 1)] == 0:
            sel = lab > 0
        a = a * sel
        ys, xs = np.where(a > 0.02)
        rgb, a = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1], a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        k = kinds[i]
        cnt[k] = cnt.get(k, 0) + 1
        items.append({"name": f"{k}_{cnt[k]:02d}", "kind": k, "rgb": rgb, "a": a, "cm": [round(a.shape[1] / px_per_cm_src, 1), round(a.shape[0] / px_per_cm_src, 1)]})
    if not items:
        log("[atlas] no items found in the sheet")
        return 1
    # residual key spill along the edge, as a number
    spill = []
    for it in items:
        edge = (it["a"] > 0.05) & (ndi.binary_erosion(it["a"] > 0.5, iterations=3) == 0)
        s = np.minimum(it["rgb"][..., 0], it["rgb"][..., 2]) - it["rgb"][..., 1]
        spill.append(float((s[edge] > 0.10).mean()) if edge.any() else 0.0)
    # pack at one common scale (px per cm), as large as fits
    size, pad = cfg["size"], 6
    lo, hi = 0.2, 1.5
    for _ in range(18):
        f = (lo + hi) / 2
        sizes = [(int(it["a"].shape[1] * f) + 1, int(it["a"].shape[0] * f) + 1) for it in items]
        if shelf_pack(sizes, size, pad) is not None:
            lo = f
        else:
            hi = f
    f = lo
    sizes = [(int(it["a"].shape[1] * f) + 1, int(it["a"].shape[0] * f) + 1) for it in items]
    pos = shelf_pack(sizes, size, pad)
    atlas_rgb = np.zeros((size, size, 3), np.float32)
    atlas_a = np.zeros((size, size), np.float32)
    atlas_n = np.tile(np.array([0.5, 0.5, 1.0], np.float32), (size, size, 1))
    atlas_o = np.tile(np.array([1.0, 0.8, 0.0], np.float32), (size, size, 1))
    for it, (w, h), (px, py) in zip(items, sizes, pos):
        rgb = cv2.resize(it["rgb"] * it["a"][..., None], (w, h), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
        a = np.clip(cv2.resize(it["a"], (w, h), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC), 0, 1)
        rgb = rgb / np.maximum(a[..., None], 1e-3)                                  # un-premultiply
        nm, orm = item_maps(rgb, a)
        atlas_rgb[py:py + h, px:px + w] = rgb
        atlas_a[py:py + h, px:px + w] = a
        atlas_n[py:py + h, px:px + w] = nm
        atlas_o[py:py + h, px:px + w] = orm
        it["px"] = [px, py, w, h]
        it["uv"] = [round(px / size, 5), round(py / size, 5), round((px + w) / size, 5), round((py + h) / size, 5)]
        it["atlas_cm"] = [round(w / (f * px_per_cm_src), 1), round(h / (f * px_per_cm_src), 1)]
    # colour bleed into the gutter (no dark fringes in mips); alpha stays as is
    solid = atlas_a > 0.5
    _, (iy, ix) = ndi.distance_transform_edt(~solid, return_indices=True)
    bleed = ndi.distance_transform_edt(~solid) <= pad
    atlas_rgb = np.where(solid[..., None] | ~bleed[..., None], atlas_rgb, atlas_rgb[iy, ix]) if False else np.where(bleed[..., None] & ~solid[..., None], atlas_rgb[iy, ix], atlas_rgb)
    # crop to the used rows (next power of two): the rest of a square atlas would be empty
    used = max(py + h for (px, py), (w, h) in zip(pos, sizes)) + pad
    H = 1 << int(np.ceil(np.log2(max(used, 64))))
    H = min(H, size)
    atlas_rgb, atlas_a, atlas_n, atlas_o = atlas_rgb[:H], atlas_a[:H], atlas_n[:H], atlas_o[:H]
    for it in items:
        it["uv"] = [it["uv"][0], round(it["px"][1] / H, 5), it["uv"][2], round((it["px"][1] + it["px"][3]) / H, 5)]
    save_png(d / f"{name}_BaseColor.png", np.concatenate([atlas_rgb, atlas_a[..., None]], -1))
    save_png(d / f"{name}_Normal.png", atlas_n)
    save_png(d / f"{name}_ORM.png", atlas_o)
    meta = {"name": name, "kind": "atlas", "size": [size, H], "px_per_cm": round(f * px_per_cm_src, 3), "basecolor_alpha": "RGBA, alpha = cutout (straight alpha)",
            "normal_convention": "directx", "orm_channels": "R=AO, G=Roughness, B=Metallic",
            "files": {"basecolor": f"{name}_BaseColor.png", "normal": f"{name}_Normal.png", "orm": f"{name}_ORM.png"},
            "items": [{k: v for k, v in it.items() if k not in ("rgb", "a")} for it in items],
            "params": {"sheet": str(sheet_p), "key_bg_rgb": [round(float(c), 3) for c in bg], "residual_edge_spill_fraction_mean": round(float(np.mean(spill)), 4),
                       "residual_edge_spill_fraction_max": round(float(np.max(spill)), 4), "codex_runs": runs, "rows_expected": cfg["rows"]}}
    (d / f"{name}.texforge.json").write_text(json.dumps(meta, indent=2))
    # previews
    over_check = _over(_checker(H, size), atlas_rgb, atlas_a)
    dirt_p = kit["out_dir"] / "dirt" / "dirt_BaseColor.png"
    tiles = []
    if dirt_p.exists():
        dirt = load_image(dirt_p)
        px = cv2.resize(dirt, (size, size), interpolation=cv2.INTER_AREA)[:H]
        over_dirt = _over(px, atlas_rgb, atlas_a)
        tiles.append(over_dirt)
        scatter(kit, items, atlas_rgb, atlas_a, atlas_n, atlas_o, f * px_per_cm_src, d / f"{name}_scatter_lit.png")
    row = [cv2.resize(over_check, (1024, H // 2), interpolation=cv2.INTER_AREA)] + [cv2.resize(t, (1024, H // 2), interpolation=cv2.INTER_AREA) for t in tiles]
    save_png(d / f"{name}_preview.png", np.concatenate(row, 1))
    # individual cutouts over a checker, 1:1
    strip = []
    for it in items:
        x, y, w, h = it["px"]
        c = _over(_checker(h, w, 12), atlas_rgb[y:y + h, x:x + w], atlas_a[y:y + h, x:x + w])
        strip.append(cv2.resize(c, (160, int(160 * h / w))) if w >= h else cv2.resize(c, (int(160 * w / h), 160)))
    H = max(s.shape[0] for s in strip)
    save_png(d / f"{name}_items_over_checker.png", np.concatenate([np.pad(s, ((0, H - s.shape[0]), (4, 4), (0, 0))) for s in strip], 1))
    log(f"[atlas] {len(items)} items ({', '.join(f'{k} x{v}' for k, v in cnt.items())}); atlas scale {f * px_per_cm_src:.2f} px/cm; "
        f"edge spill mean {np.mean(spill):.3f} max {np.max(spill):.3f}; codex runs {runs}")
    return 0


def scatter(kit, items, rgb, a, nrm, orm, px_per_cm, out_path, n: int = 55, crop: int = 1536, seed: int = 5):
    """Drop the cutouts at real scale on the dirt layer (random rotation), light the result with dirt + item normals."""
    dd = kit["out_dir"] / "dirt"
    scale_dirt = 2048 / kit["world_cm"]                      # px per cm of the dirt tile
    base = load_image(dd / "dirt_BaseColor.png")
    nr = load_image(dd / "dirt_Normal.png")
    om = load_image(dd / "dirt_ORM.png")
    S = crop
    tile = lambda x: np.tile(x, (2, 2, 1))[:S, :S].copy()      # noqa: E731
    B, N, O = tile(base), tile(nr), tile(om)
    k = scale_dirt / px_per_cm
    rng = np.random.default_rng(seed)
    for _ in range(n):
        it = items[int(rng.integers(len(items)))]
        x, y, w, h = it["px"]
        sl = (slice(y, y + h), slice(x, x + w))
        ang = float(rng.uniform(0, 360))
        diag = int(np.hypot(w, h) * k) + 4
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, k)
        M[0, 2] += diag / 2 - w / 2
        M[1, 2] += diag / 2 - h / 2
        warp = lambda im: cv2.warpAffine(im, M, (diag, diag), flags=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)  # noqa: E731
        c, aa, nn, oo = warp(rgb[sl] * a[sl][..., None]), warp(a[sl]), warp(nrm[sl]), warp(orm[sl])
        c = c / np.maximum(aa[..., None], 1e-3)
        v = nn * 2 - 1
        r = np.deg2rad(ang)                                      # rotate the tangent-space xy with the sprite
        vx = v[..., 0] * np.cos(-r) - v[..., 1] * np.sin(-r)
        vy = v[..., 0] * np.sin(-r) + v[..., 1] * np.cos(-r)
        nn = np.stack([vx, vy, v[..., 2]], -1) * 0.5 + 0.5
        px0, py0 = int(rng.integers(0, S - diag)), int(rng.integers(0, S - diag))
        sl2 = (slice(py0, py0 + diag), slice(px0, px0 + diag))
        al = aa[..., None]
        B[sl2] = B[sl2] * (1 - al) + c * al
        N[sl2] = N[sl2] * (1 - al) + nn * al
        O[sl2] = O[sl2] * (1 - al) + oo * al
    imgs = [render_lit(B, N, O, L, tiles=1, px=S) for L in ((-0.8, -0.3, 0.5), (0.8, 0.3, 0.5))]
    save_png(out_path, np.concatenate([B] + imgs, 1)[::2, ::2] if False else np.concatenate([cv2.resize(B, (1024, 1024), interpolation=cv2.INTER_AREA)] + [cv2.resize(i, (1024, 1024), interpolation=cv2.INTER_AREA) for i in imgs], 1))
