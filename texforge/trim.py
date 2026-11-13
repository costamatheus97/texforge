"""`texforge trim`: one sheet of horizontal strips; every strip tiles left-to-right on its own.

Per strip: detect the strip band, remove horizontal lighting drift, make it periodic along x (either an exact
whole number of detected periods, e.g. brick band / corrugated zinc / dentils, or a min-cut seam blend), scale
to the sheet width, then stack with edge-extended gutters so mip levels never bleed between strips. The JSON
sidecar lists each strip's V range (v=0 is the top of the image) for modellers and the UE import script.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import gen as genmod
from . import maps, pattern, prompts, seamless
from .common import load_image, luminance, resize, resize_periodic, save_png, srgb_to_lin, lin_to_srgb

METAL = re.compile(r"zinc|metal|steel|iron|alumin|copper|tin\b|corrugat", re.I)
BRICK = re.compile(r"brick", re.I)
TILE = re.compile(r"tile|slate", re.I)


# ------------------------------------------------------------------ strip bounds

def detect_bounds(img: np.ndarray, n: int, min_frac: float = 0.04) -> list[int]:
    """n+1 row boundaries (0 .. H) from an optimal 1-D partition (dynamic programming) of per-row features
    (mean colour + horizontal edge energy) into n segments of minimum total variance."""
    h, w = img.shape[:2]
    lum = luminance(img)
    energy = np.abs(np.diff(lum, axis=1)).mean(1)
    f = np.concatenate([img.mean(1), energy[:, None]], 1)
    f = gaussian_filter1d(f, 2.0, axis=0)
    f = (f - f.mean(0)) / np.maximum(f.std(0), 1e-6)
    cs = np.vstack([np.zeros((1, f.shape[1])), np.cumsum(f, 0)])
    cs2 = np.concatenate([[0], np.cumsum((f ** 2).sum(1))])

    def sse(i, j):  # rows [i, j)
        s = cs[j] - cs[i]
        return cs2[j] - cs2[i] - (s ** 2).sum() / (j - i)

    m = max(3, int(min_frac * h))
    pos = np.array(sorted(set(range(0, h, 2)) | {h}))
    P = len(pos)
    INF = 1e18
    # cost[i, j] = SSE of rows [pos[i], pos[j])
    S = cs[pos]
    S2 = cs2[pos]
    cost = np.full((P, P), INF)
    for j in range(P):
        i = np.arange(j)
        seg = S[j] - S[i]
        c = S2[j] - S2[i] - (seg ** 2).sum(1) / np.maximum(pos[j] - pos[i], 1)
        c[(pos[j] - pos[i]) < m] = INF
        cost[:j, j] = c
    best = np.full(P, INF)
    best[0] = 0.0
    prev = []
    for k in range(n):
        tot = best[:, None] + cost
        arg = tot.argmin(0)
        best = tot[arg, np.arange(P)]
        prev.append(arg)
    j = P - 1
    out = [h]
    for k in range(n - 1, -1, -1):
        j = prev[k][j]
        out.append(int(pos[j]))
    return out[::-1]


# ------------------------------------------------------------------ one strip -> periodic

def _unlight_x(strip: np.ndarray) -> np.ndarray:
    """Remove horizontal brightness drift only (vertical profile is the moulding itself)."""
    lin = srgb_to_lin(strip)
    lum = luminance(lin).mean(0)
    w = len(lum)
    p = w // 3
    low = gaussian_filter1d(np.pad(lum, p, mode="reflect"), 0.15 * w)[p:-p]
    ratio = np.clip(lum.mean() / np.maximum(low, 1e-4), 0.6, 1.6)
    return lin_to_srgb(np.clip(lin * ratio[None, :, None], 0, 1))


def _x_period(strip: np.ndarray) -> tuple[float, float]:
    """(period px, strength) of a repeating pattern along x, summed over rows so each course/row counts."""
    lum = luminance(strip)
    lum = lum - lum.mean(1, keepdims=True)
    h, w = lum.shape
    f = np.fft.rfft(lum, 2 * w, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :w].sum(0)
    ac = ac / max(ac[0], 1e-12)
    lo = max(6, int(0.02 * w))
    from scipy.signal import find_peaks
    pk, _ = find_peaks(ac[: w // 2], height=0.2)
    pk = pk[pk >= lo]
    if len(pk) == 0:
        return 0.0, 0.0
    top = ac[pk].max()
    first = pk[np.argmax(ac[pk] >= 0.6 * top)]
    y0, y1, y2 = ac[first - 1], ac[first], ac[first + 1]
    den = y0 - 2 * y1 + y2
    off = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    return float(first + off), float(ac[first])


def periodic_strip(strip: np.ndarray, out_w: int) -> tuple[np.ndarray, dict]:
    """-> strip tiling along x, resampled to width out_w with uniform scale (height scales with it)."""
    strip = _unlight_x(strip)
    h, w = strip.shape[:2]
    P, strength = _x_period(strip)
    if strength > 0.35 and P > 6 and w / P >= 2:
        n = int(np.floor(w / P))
        span = n * P
        x0 = (w - span) / 2
        info = {"method": f"{n} x period {P:.1f}px", "period_strength": round(strength, 2)}
        scale = out_w / span
        out_h = int(round(h * scale))
        xs = (x0 + (np.arange(out_w) + 0.0) / out_w * span).astype(np.float32)
        ys = ((np.arange(out_h) + 0.5) / scale - 0.5).astype(np.float32)
        mx, my = np.meshgrid(xs, ys)
        res = cv2.remap(strip, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        # residual hairline mismatch at the wrap: min-cut over a very small overlap, then restore the width
        res = _heal_x(res)
        return np.clip(res, 0, 1), info
    sw = seamless.periodic_x(strip, 1 / 6)
    scale = out_w / sw.shape[1]
    out_h = int(round(h * scale))
    res = resize_periodic(sw, (out_h, out_w))   # exact periodic phase (a plain resize would reflect at the border)
    return np.clip(res, 0, 1), {"method": "min-cut", "period_strength": round(strength, 2)}


def _heal_x(a: np.ndarray, band: int = 10) -> np.ndarray:
    """Spread the per-row jump across the wrap-around border over `band` px each side (removes the hairline
    left by a sub-pixel period error without touching the interior)."""
    w = a.shape[1]
    out = a.copy()
    jump = a[:, 0] - a[:, -1]                       # what the border step would be
    wgt = (1 - np.arange(band) / band)[None, :, None]
    out[:, :band] -= 0.5 * jump[:, None, :] * wgt
    out[:, w - 1 - np.arange(band)] += 0.5 * jump[:, None, :] * wgt
    return out


# ------------------------------------------------------------------ the sheet

def _kind(name: str) -> str:
    if METAL.search(name):
        return "metal"
    if BRICK.search(name):
        return "brick"
    if TILE.search(name):
        return "tile"
    if re.search(r"stone|granite|limestone|cornice|sill|marble|concrete", name, re.I):
        return "organic"
    return "plaster"


def make_trim(strips_text: str, name: str, out_root: Path, image: str | None = None, size: int = 2048,
              world_cm: float = 200.0, bounds: str | None = None, log=print) -> dict:
    names = [s.strip() for s in strips_text.split(";") if s.strip()]
    out_dir = Path(out_root) / name
    if image is None:
        files = genmod.generate(prompts.trim_prompt(names), out_dir, name, n=1)
        image = str(files[0])
    img = load_image(image)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_png(out_dir / f"{name}_source.png", img)
    h = img.shape[0]
    n = len(names)
    if bounds:
        fr = [float(v) for v in bounds.split(",")]
        if len(fr) != n + 1:
            raise SystemExit(f"--bounds needs {n + 1} values")
        rows = [int(round(v * h)) for v in fr]
    else:
        rows = detect_bounds(img, n)
    log("strip rows:", rows)

    strips, infos = [], []
    for i, nm in enumerate(names):
        y0, y1 = rows[i], rows[i + 1]
        m = max(2, int(0.04 * (y1 - y0)))
        crop = img[y0 + m:y1 - m]
        a, info = periodic_strip(crop, size)
        strips.append(a)
        infos.append(info)
    total = sum(s.shape[0] for s in strips)
    gutter_total = max(0, size - total)
    if total > size - 8 * (n + 1):  # too tall: squash everything vertically to fit
        k = (size - 8 * (n + 1)) / total
        strips = [cv2.resize(s, (size, max(4, int(round(s.shape[0] * k)))), interpolation=cv2.INTER_AREA) for s in strips]
        total = sum(s.shape[0] for s in strips)
        gutter_total = size - total
    gap = gutter_total // (n + 1)

    base = np.zeros((size, size, 3), np.float32)
    normal = np.zeros((size, size, 3), np.float32)
    orm = np.zeros((size, size, 3), np.float32)
    height = np.zeros((size, size), np.float32)
    meta = []
    y = gap
    for i, (nm, s) in enumerate(zip(names, strips)):
        kind = _kind(nm)
        h_i = s.shape[0]
        joint = pattern.joint_mask(s) if kind in ("brick", "tile") else None
        mt = {"metal": "plaster", "brick": "brick", "tile": "tile", "organic": "organic", "plaster": "plaster"}[kind]
        d = maps.derive_all(s, mt, joint, wrap="x")
        if kind == "metal":
            d["orm"][..., 1] = np.clip(0.38 + 0.1 * (0.5 - luminance(s)), 0.2, 0.7)
            d["orm"][..., 2] = 0.9
        # paste strip, then edge-extend into the gutter above/below it (half each side)
        lo, hi = y - gap // 2 if i else 0, y + h_i + (gap - gap // 2)
        for arr, key, src in ((base, None, s), (normal, "normal", None), (orm, "orm", None), (height, "height", None)):
            val = src if key is None else d[key]
            arr[y:y + h_i] = val
            arr[lo:y] = val[0:1]
            arr[y + h_i:hi] = val[-1:]
        meta.append({"name": nm, "v0": round(y / size, 5), "v1": round((y + h_i) / size, 5),
                     "height_cm": round(h_i / size * world_cm, 2), "world_cm_width": float(world_cm),
                     "kind": kind, **infos[i]})
        y += h_i + gap
    # any rounding remainder at the bottom: extend the last strip
    if y - gap + 0 < size:
        yy = y - gap
        for arr in (base, normal, orm, height):
            arr[yy:] = arr[yy - 1:yy]
    files = {}
    for key, arr, suffix in (("basecolor", base, "BaseColor"), ("normal", normal, "Normal"), ("orm", orm, "ORM"), ("height", height, "Height")):
        fname = f"{name}_{suffix}.png"
        save_png(out_dir / fname, arr)
        files[key] = fname
    sidecar = {
        "name": name, "kind": "trim", "type": "trim", "size": [size, size], "world_cm": [float(world_cm), float(world_cm) * 1.0],
        "normal_convention": "directx", "height_depth_cm": 1.5, "tileable": True, "tile_axes": "x",
        "files": files, "orm_channels": "R=AO, G=Roughness, B=Metallic",
        "strips": meta,
        "params": {"source": str(image), "source_bounds_rows": rows,
                   "uv_hint": "u runs along the strip (1.0 = world_cm_width); map a piece to a strip by squeezing its V into [v0, v1]; v=0 is the TOP of the image"},
        "notes": ["strips tile along x only; never tile the sheet vertically"],
    }
    (out_dir / f"{name}.texforge.json").write_text(json.dumps(sidecar, indent=2))
    # preview: each strip repeated twice horizontally with its V range labelled by colour ticks
    prev = np.concatenate([base, base], 1)
    save_png(out_dir / f"{name}_preview.png", resize(prev, (size // 2, size)))
    return sidecar
