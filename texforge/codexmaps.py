"""Option 2: get the HEIGHT map from codex as an image-to-image edit of the finished albedo instead of deriving it
from luminance, measure how well it registers with the albedo, and make it tile.

Flow used by `texforge make --height-from codex|hybrid`:
  1. albedo (periodic, final) -> codex edit "grayscale height map of exactly this surface" (1 codex run)
  2. grey, resize to the albedo size, polarity check, global offset removed (phase correlation)
  3. quality gate: global offset, local flow, edge correlation against the albedo (see `registration`)
  4. codex output is NOT periodic: its border is faded into the periodic luminance height over a 3 % band
  5. smoothed (the generator adds grain), optionally 'hybrid' = + fine luminance micro-relief
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from . import maps
from .common import blur, load_image, luminance, resize, save_png, smoothstep

HEIGHT_PROMPT = (
    "Convert the attached image into a grayscale HEIGHT MAP of exactly this surface: same layout, same framing, "
    "same scale, every feature in exactly the same position as in the attached image. White = raised, black = "
    "recessed. Recessed joints, gaps, cracks and pits are dark; the faces of bricks / stones / plaster stay light "
    "to mid grey. Pure grayscale, no colour, no lighting, no shading, no cast shadows, no highlights, no texture "
    "from colour or stains (a stain or colour change is NOT a height change), no border, no vignette."
)
ROUGHNESS_PROMPT = (
    "Convert the attached image into a grayscale ROUGHNESS MAP of exactly this surface: same layout, same framing, "
    "same scale, every feature in exactly the same position as in the attached image. White = fully rough/matte, "
    "black = smooth/glossy. Pure grayscale, no colour, no lighting, no shading, no shadows, no border, no vignette."
)
NORMAL_PROMPT = (
    "Convert the attached image into a tangent-space NORMAL MAP of exactly this surface (DirectX / Unreal "
    "convention, green channel down): same layout, same framing, same scale, every feature in exactly the same "
    "position as in the attached image. Flat areas are exactly (128,128,255). Recessed joints, gaps and pits are "
    "bevelled inwards, brick / stone faces are raised. No lighting, no shading, no colour from the original, "
    "no border, no vignette."
)


def load_gray(path, size: int) -> np.ndarray:
    """Codex output (RGB, any size) -> float [0,1] grey at size x size (Lanczos)."""
    g = load_image(path).mean(-1)
    return np.clip(resize(g, (size, size)), 0, 1)


def _gradmag(a: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), sigma)
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-12))


def registration(albedo_srgb: np.ndarray, h: np.ndarray, joint: np.ndarray | None = None) -> dict:
    """How well does the codex map h register with the albedo? (all offsets in px of the albedo's resolution)

    global_offset : phase correlation of the gradient-magnitude images (dx, dy of h relative to the albedo)
    edge_corr     : Pearson correlation of those gradient-magnitude images, before and after removing the offset
    flow_p50/p95  : dense optical flow (Farneback) between the two gradient images = local non-rigid mismatch
    joint_*       : (only with a joint mask) contrast d' = (mean h outside joints - mean h inside joints)/std(h) at
                    zero shift, and the shift (within +-20 px) that maximises it
    """
    n = h.shape[0]
    lum = luminance(albedo_srgb)
    ga, gh = _gradmag(lum), _gradmag(h)
    win = cv2.createHanningWindow((n, n), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ga.astype(np.float32), gh.astype(np.float32), win)
    # phaseCorrelate returns the shift of the 2nd image relative to the 1st
    out = {"global_offset_px": [round(float(dx), 2), round(float(dy), 2)], "phase_response": round(float(resp), 3),
           "edge_corr": round(_pearson(ga, gh), 3)}
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    gh2 = cv2.warpAffine(gh, M, (n, n), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    out["edge_corr_after_offset"] = round(_pearson(ga, gh2), 3)
    s = 1024
    a8 = cv2.resize(np.clip(ga / max(np.percentile(ga, 99.5), 1e-6), 0, 1) * 255, (s, s), interpolation=cv2.INTER_AREA).astype(np.uint8)
    b8 = cv2.resize(np.clip(gh / max(np.percentile(gh, 99.5), 1e-6), 0, 1) * 255, (s, s), interpolation=cv2.INTER_AREA).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(a8, b8, None, 0.5, 5, 41, 5, 7, 1.5, 0)
    mag = np.sqrt((flow ** 2).sum(-1)) * (n / s)
    out["flow_p50_px"] = round(float(np.percentile(mag, 50)), 2)
    out["flow_p95_px"] = round(float(np.percentile(mag, 95)), 2)
    out["flow_mean_dxdy_px"] = [round(float(flow[..., 0].mean() * n / s), 2), round(float(flow[..., 1].mean() * n / s), 2)]
    if joint is not None:
        j = joint > 0.5
        sd = float(h.std()) + 1e-9

        def contrast(hh):
            return float((hh[~j].mean() - hh[j].mean()) / sd)
        out.update(edge_chamfer(h, joint))
        out["joint_contrast_dprime_at_0"] = round(contrast(h), 3)
        best, bs = -9.0, (0, 0)
        for sy in range(-20, 21, 2):
            for sx in range(-20, 21, 2):
                c = contrast(np.roll(h, (sy, sx), (0, 1)))
                if c > best:
                    best, bs = c, (sx, sy)
        out["joint_contrast_dprime_best"] = round(best, 3)
        out["joint_best_shift_px"] = list(bs)
        out["joint_mean_h_inside"] = round(float(h[j].mean()), 3)
        out["joint_mean_h_outside"] = round(float(h[~j].mean()), 3)
    return out


def edge_chamfer(h: np.ndarray, joint: np.ndarray) -> dict:
    """Distance (px) from the contour of the Otsu-thresholded height to the nearest boundary of the albedo's joint
    mask: median and 95th percentile over contour pixels. Measures whether brick / stone outlines line up."""
    hb = cv2.GaussianBlur(h.astype(np.float32), (0, 0), 3)
    h8 = np.clip(hb * 255, 0, 255).astype(np.uint8)
    _, m = cv2.threshold(h8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = np.ones((3, 3), np.uint8)
    contour = (m - cv2.erode(m, k)) > 0
    jb = (joint > 0.5).astype(np.uint8)
    jbound = (cv2.dilate(jb, k) - cv2.erode(jb, k)) > 0
    dist = cv2.distanceTransform((~jbound).astype(np.uint8), cv2.DIST_L2, 3)
    d = dist[contour]
    return {"edge_dist_median_px": round(float(np.median(d)), 2), "edge_dist_p95_px": round(float(np.percentile(d, 95)), 2)}


def match_hist(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Remap src's value distribution onto ref's (monotone)."""
    qs = np.linspace(0, 100, 257)
    return np.interp(src, np.percentile(src, qs), np.percentile(ref, qs)).astype(np.float32)


def make_periodic(h: np.ndarray, h_ref: np.ndarray, band_frac: float = 0.03) -> tuple[np.ndarray, float]:
    """Codex output is not periodic. Fade it into the (periodic) luminance height `h_ref` over a band at the tile
    border so the border pixels equal h_ref exactly. Returns (periodic height, fraction of the area altered)."""
    n = h.shape[0]
    band = max(4, int(band_frac * n))
    ref = match_hist(h_ref, h)
    d = np.minimum(np.arange(n), np.arange(n)[::-1]).astype(np.float32)
    w1 = smoothstep(0, band, d)
    w = np.minimum(w1[:, None], w1[None, :])
    out = w * h + (1 - w) * ref
    return out.astype(np.float32), float((w < 0.999).mean())


def build_variant(name: str, out_root: Path, h: np.ndarray, label: str, joint: np.ndarray | None, log=print) -> str:
    """Write a full set `<name>_<label>` whose normal/AO come from height `h` (BaseColor/roughness as the original)."""
    src_dir = Path(out_root) / name
    meta = json.loads((src_dir / f"{name}.texforge.json").read_text())
    base = load_image(src_dir / f"{name}_BaseColor.png")
    mtype = meta["type"]
    d = maps.TYPE_DEFAULTS[mtype]
    world = meta["world_cm"]
    new = f"{name}_{label}"
    out = Path(out_root) / new
    out.mkdir(exist_ok=True)
    normal = maps.normal_map(h, d["normal"], "xy", world[0] / world[1])
    ao = maps.ao_map(h, d["ao"], "xy")
    rough = maps.roughness_map(base, joint, d["rough"], d["joint_rough"], "xy")
    orm = maps.pack_orm(ao, rough, d["metallic"])
    save_png(out / f"{new}_BaseColor.png", base)
    save_png(out / f"{new}_Normal.png", normal)
    save_png(out / f"{new}_ORM.png", orm)
    save_png(out / f"{new}_Height.png", h)
    shutil.copyfile(src_dir / f"{name}_BaseColor.png", out / f"{new}_source.png")
    meta.update(name=new, files={k: v.replace(name, new, 1) for k, v in meta["files"].items()})
    meta.setdefault("notes", []).append(f"variant '{label}' of {name}: height supplied externally, normal/AO derived from it")
    (out / f"{new}.texforge.json").write_text(json.dumps(meta, indent=2))
    return new


# gate: a codex height is accepted only if it registers with the albedo this well (px at the albedo's resolution)
GATE = dict(max_offset_px=3.0, max_flow_p95_px=6.0, min_edge_corr_ratio=0.6,
            # no joint mask (organic / plaster): the generator smooths fine grain, so the edge correlation against the
            # luminance reference is lower even when the layout is right (offsets <= 1.7 px, flow p95 <= 2.6 px on the forest
            # kit, ratio 0.50-0.73); a layout from another image scores ~0.0 with offsets of 70-200 px
            min_edge_corr_ratio_organic=0.40)


def prepare_height(base: np.ndarray, h_raw: np.ndarray, h_lum: np.ndarray, joint: np.ndarray | None,
                   mode: str = "hybrid", sigma_px: float = 2.5, band_frac: float = 0.03,
                   soft: bool = False) -> tuple[np.ndarray, dict, bool]:
    """codex grey `h_raw` (same size as `base`) -> (height, report, accepted). `h_lum` is the luminance height
    (periodic) used as the quality reference and to close the tile border."""
    rep: dict = {}
    n = base.shape[0]
    if joint is not None:
        flip = h_raw[joint > 0.5].mean() > h_raw[joint <= 0.5].mean()      # joints must be dark
    else:
        flip = _pearson(h_raw, h_lum) < 0
    if flip:
        h_raw = 1 - h_raw
        rep["flipped_polarity"] = True
    ref = registration(base, h_lum, joint)
    reg = registration(base, h_raw, joint)
    dx, dy = reg["global_offset_px"]
    if abs(dx) > 0.7 or abs(dy) > 0.7:
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        h_raw = cv2.warpAffine(h_raw, M, (n, n), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        rep["offset_removed_px"] = [dx, dy]
        reg = registration(base, h_raw, joint)
    rep["registration_luminance_reference"] = ref
    rep["registration_codex"] = reg
    ok = (max(abs(dx), abs(dy)) <= GATE["max_offset_px"] and reg["flow_p95_px"] <= GATE["max_flow_p95_px"]
          and reg["edge_corr_after_offset"] >= (GATE["min_edge_corr_ratio"] if joint is not None else GATE["min_edge_corr_ratio_organic"]) * ref["edge_corr_after_offset"])
    rep["accepted"] = bool(ok)
    h, frac = make_periodic(h_raw, h_lum, band_frac)
    rep["border_blend_area_fraction"] = round(frac, 3)
    h = blur(h, sigma_px)
    if mode == "hybrid":
        lum = luminance(base)
        fine = maps._norm(blur(lum, max(0.8, 0.0006 * n)) - blur(lum, 0.02 * n))
        h = maps.soft_unit(h + 0.22 * (fine - 0.5), 0.05, 99.95) if soft else np.clip(h + 0.22 * (fine - 0.5), 0, 1)
    return h.astype(np.float32), rep, ok


# --- codex NORMAL (option 2b): image-to-image normal map, convention auto-corrected, made valid and periodic -------

NORMAL_GATE = dict(min_abs_corr=0.15)
Z_EPS = 0.05


SLOPE_SCALES = (1.5, 3.0, 6.0, 12.0)


def _slope_corr(v: np.ndarray, h: np.ndarray) -> tuple[float, float]:
    """Correlation of the smoothed normal x / y channels with -dH/dx, -dH/dy (DirectX: N = (-k dh/dx, -k dh/dy, 1),
    y down the image). Positive = this channel already follows the convention. Measured at several smoothing scales
    and the strongest one is kept per channel: fine-grained textures (needles, mulch, bark furrows) carry the sign
    in the coarse relief while the pixel-level grain is uncorrelated."""
    best = [0.0, 0.0]
    for sg in SLOPE_SCALES:
        hb = blur(h, sg)
        gx = cv2.Sobel(hb, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(hb, cv2.CV_32F, 0, 1, ksize=3)
        for ch, g in ((0, gx), (1, gy)):
            c = _pearson(blur(v[..., ch], sg), -g)
            if abs(c) > abs(best[ch]):
                best[ch] = c
    return best[0], best[1]


def _rnm(base: np.ndarray, detail: np.ndarray) -> np.ndarray:
    """Reoriented normal mapping: apply `detail` (unit vectors) on top of `base` (unit vectors)."""
    t = base + np.array([0, 0, 1], np.float32)
    u = detail * np.array([-1, -1, 1], np.float32)
    r = t * ((t * u).sum(-1, keepdims=True) / np.maximum(t[..., 2:3], 1e-4)) - u
    return r / np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-6)


def prepare_normal(base: np.ndarray, n_raw: np.ndarray, h_refs: list[np.ndarray], n_height: np.ndarray,
                   mtype: str, aspect_y: float = 1.0, detail: float = 0.0, band_frac: float = 0.03,
                   balance: bool = False):
    """codex RGB normal `n_raw` (albedo size, [0,1]) -> (normal [0,1] DirectX, report, accepted).

    1. channel convention from the SIGN of the correlation with the gradient of each reference height in `h_refs`
       (the set's codex height and the luminance height); a channel with |corr| < 0.15, or references that disagree
       on its sign, rejects the codex normal (caller keeps the height-derived `n_height`)
    2. flip channels, clamp z >= Z_EPS, renormalise
    3. periodic: fade the border into `n_height` (periodic) in vector space over a band_frac band, renormalise
    3b. optional `balance`: rescale the x / y slopes so their RMS ratio equals the height-derived normal's, keeping
       the geometric-mean strength (the generator can draw one axis much weaker than the other)
    4. optional `detail` (0 = off): RNM blend of the fine luminance relief on top
    """
    rep: dict = {}
    v = (n_raw * 2 - 1).astype(np.float32)
    corrs = [_slope_corr(v, h) for h in h_refs]
    rep["corr_x_y_per_reference"] = [[round(a, 3), round(b, 3)] for a, b in corrs]
    sx, sy = [], []
    ok = True
    for ch in (0, 1):
        vals = [c[ch] for c in corrs]
        if min(abs(x) for x in vals) < NORMAL_GATE["min_abs_corr"] or len({x > 0 for x in vals}) > 1:
            ok = False
        sx_ = 1.0 if np.mean(vals) > 0 else -1.0
        (sx if ch == 0 else sy).append(sx_)
    rep["flip_x"], rep["flip_y"] = bool(sx[0] < 0), bool(sy[0] < 0)
    rep["pct_z_negative_before"] = round(float((v[..., 2] < 0).mean() * 100), 2)
    rep["accepted"] = bool(ok)
    if not ok:
        return n_height, rep, False
    v[..., 0] *= sx[0]
    v[..., 1] *= sy[0]
    v[..., 2] = np.maximum(v[..., 2], Z_EPS)
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    vh = n_height * 2 - 1
    if balance:
        rms = lambda a: float(np.sqrt((blur(a, 1.0) ** 2).mean()))  # noqa: E731
        gx = float(np.sqrt((rms(vh[..., 0]) / rms(vh[..., 1])) / (rms(v[..., 0]) / rms(v[..., 1]))))
        sl = v[..., :2] / v[..., 2:3] * np.array([gx, 1 / gx], np.float32)
        v = np.concatenate([sl, np.ones_like(sl[..., :1])], -1)
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        v[..., 2] = np.maximum(v[..., 2], Z_EPS)
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        rep["balance_gain_x_y"] = [round(gx, 3), round(1 / gx, 3)]
    # periodic: border fades into the (periodic) height-derived normal, vector space, then renormalise
    n = v.shape[0]
    band = max(4, int(band_frac * n))
    d = np.minimum(np.arange(n), np.arange(n)[::-1]).astype(np.float32)
    w1 = smoothstep(0, band, d)
    w = np.minimum(w1[:, None], w1[None, :])[..., None]
    v = w * v + (1 - w) * vh
    v /= np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)
    rep["border_blend_area_fraction"] = round(float((w < 0.999).mean()), 3)
    if detail > 0:
        lum = luminance(base)
        fine = maps._norm(blur(lum, max(0.8, 0.0006 * n)) - blur(lum, 0.02 * n))
        dn = maps.normal_map(fine, maps.TYPE_DEFAULTS[mtype]["normal"] * detail, "xy", aspect_y) * 2 - 1
        v = _rnm(v, dn / np.maximum(np.linalg.norm(dn, axis=-1, keepdims=True), 1e-6))
        rep["detail_strength"] = detail
    v[..., 2] = np.maximum(v[..., 2], Z_EPS)
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    return (v * 0.5 + 0.5).astype(np.float32), rep, True
