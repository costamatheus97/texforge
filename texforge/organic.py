"""Organic / ground-material extras for `make` (default ON for type `organic`, optional for cobble / plaster):

1. despeckle_height  - isolated single/few-pixel spikes and pits in the height map are replaced by the local median
                       (gated by a local-outlier test, so real fine detail is untouched). Done BEFORE normal / AO.
2. material_masks    - the sheet is segmented into sub-materials (moss: living / dead / soil gaps / twigs; gravel: stones /
                       fines; generic: raised / mid / low) by k-means in Lab + height cue; every sub-material gets its own
                       roughness, a height offset (+ cushion doming for moss) and an AO tweak; the soft mask is written as
                       `<name>_Mask.png` (channels documented in the JSON).
3. lift_shadows      - removes mid/high-frequency cast / occlusion shadows that the image painted into the albedo:
                       albedo /= f(AO) (exponent fitted by regression of log-luminance on log-AO, gain capped, soil gaps
                       only partly lifted so they stay darker, mean luminance kept).

`derive_organic` is the entry point used by make.py. Nothing here is imported by trim / decal / atlas.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import maps
from .common import blur, lin_to_srgb, luminance, pad_wrap, srgb_to_lin

ORGANIC_TYPES = ("organic",)          # default ON
OPTIONAL_TYPES = ("cobble", "plaster")  # despeckle only by default (no masks, no shadow lift)


def defaults_for(mtype: str) -> dict:
    """Resolve the on/off defaults per material type (brick / tile keep the old behaviour)."""
    if mtype in ORGANIC_TYPES:
        return dict(despeckle=True, shadow_lift=True, materials=True)
    if mtype in OPTIONAL_TYPES:
        return dict(despeckle=True, shadow_lift=False, materials=False)
    return dict(despeckle=False, shadow_lift=False, materials=False)


# --------------------------------------------------------------------------------------------------------------------
# 1. height speck cleaning
# --------------------------------------------------------------------------------------------------------------------

def _median(h: np.ndarray, k: int = 5, wrap: str = "xy") -> np.ndarray:
    p = k
    return cv2.medianBlur(pad_wrap(h.astype(np.float32), p, wrap), k)[p:-p, p:-p]


def speck_mask(h: np.ndarray, thr: np.ndarray | None = None, k: float = 4.0, max_area: int = 12,
               wrap: str = "xy") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (speck mask, threshold map, residual). A pixel is an outlier when |h - median5(h)| exceeds `k` x a local robust
    sigma (blurred absolute residual, floored by half the global median-absolute-deviation sigma, so a perfectly flat
    area does not flag every tiny wiggle); an outlier is a SPECK when its 8-connected cluster has <= max_area pixels
    (bigger clusters are real features: a needle edge, a pebble). Pass the `thr` of a previous call to measure another
    map with exactly the same yardstick."""
    med = _median(h, 5, wrap)
    r = h - med
    if thr is None:
        sg = 1.4826 * float(np.median(np.abs(r))) + 1e-6
        loc = blur(np.minimum(np.abs(r), 3 * sg), 6.0, wrap) * 1.25
        thr = k * np.maximum(loc, 0.5 * sg)
    flag = np.abs(r) > thr
    lab, n = ndi.label(flag, structure=np.ones((3, 3), bool))
    if n:
        area = np.bincount(lab.ravel(), minlength=n + 1)
        small = (area <= max_area)
        small[0] = False
        spk = small[lab]
    else:
        spk = np.zeros_like(flag)
    return spk, thr.astype(np.float32), r


def despeckle_height(h: np.ndarray, wrap: str = "xy", k: float = 4.0, max_area: int = 12) -> tuple[np.ndarray, dict]:
    """Replace isolated spikes / pits by the local median. -> (cleaned height, stats). The stats count outlier pixels
    with the SAME threshold map before and after, and the RMS change on every non-flagged pixel (should be ~0)."""
    spk, thr, _ = speck_mask(h, None, k, max_area, wrap)
    out = h.copy()
    touched = spk.copy()
    cur = spk
    for _ in range(6):                      # a spike's halo can leave a stub: repeat at the ORIGINAL yardstick until clean
        if not cur.any():
            break
        med = _median(out, 5, wrap)
        out[cur] = med[cur]
        cur, _, _ = speck_mask(out, thr, k, max_area, wrap)
        touched |= cur
    after, _, _ = speck_mask(out, thr, k, max_area, wrap)
    changed = out != h
    keep = ~ndi.binary_dilation(touched, iterations=1)
    stats = {"outlier_px_before": int(spk.sum()), "outlier_px_after": int(after.sum()),
             "pct_before": round(float(spk.mean() * 100), 4), "pct_after": round(float(after.mean() * 100), 4),
             "px_changed": int(changed.sum()), "pct_px_changed": round(float(changed.mean() * 100), 4),
             "rms_change_outside_halo": round(float(np.sqrt(np.mean((out - h)[keep] ** 2))), 6),
             "rms_change_all": round(float(np.sqrt(np.mean((out - h) ** 2))), 6),
             "k": k, "max_area_px": max_area}
    return out.astype(np.float32), stats


# --------------------------------------------------------------------------------------------------------------------
# 2. material masks inside one texture
# --------------------------------------------------------------------------------------------------------------------

# roughness / height / AO per sub-material. `rough`: offset added to the base roughness. `dh`: height offset (0..1 scale).
# `ao`: extra occlusion (0..1) multiplied into the AO channel. `lum_slope`: roughness change with local brightness
# (+ = brighter is rougher, i.e. darker / damper parts are smoother; - = the old "darker = rougher").
PROFILES = {
    "moss": dict(
        k=5, channels=["living_moss", "dead_brown_moss", "soil_gaps", "twigs_needles"],
        rough=[0.00, +0.03, -0.14, -0.08], dh=[0.0, -0.03, -0.12, +0.06], ao=[0.0, 0.0, 0.40, 0.0],
        lum_slope=+0.10, dome=0.12, dead_hue_max=90.0,
        soil_feather=0.004, soil_relief=0.05, soil_detail=1.6, soil_keep=0.3,
        soil_relief_band=(2.5, 0.025),   # (low-pass px, high-pass fraction of the tile) of the band that drives the soil micro-relief
        doc="R living moss, G dead / dry brown moss, B soil + humus gaps between cushions, A twigs / needles on top"),
    "gravel": dict(
        k=3, channels=["stones", "fines", "-", "-"],
        rough=[-0.06, +0.08], dh=[+0.04, -0.07], ao=[0.0, 0.30], lum_slope=-0.10, dome=0.0,
        doc="R stones, G sandy fines between them"),
    "generic": dict(
        k=3, channels=["raised", "mid", "low"],
        rough=[-0.03, 0.0, +0.04], dh=[0.0, 0.0, 0.0], ao=[0.0, 0.0, 0.0], lum_slope=-0.16, dome=0.0,
        doc="R raised (high, brighter), G mid, B low / recessed (darker)"),
}


def pick_profile(name: str, requested: str | None) -> str:
    if requested and requested not in ("auto", None):
        return requested
    n = name.lower()
    if "moss" in n:
        return "moss"
    if "gravel" in n or "pebble" in n or "stone" in n:
        return "gravel"
    return "generic"


def _lab(base_srgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(base_srgb, dtype=np.float32), cv2.COLOR_RGB2Lab)


def _kmeans_soft(feat: np.ndarray, k: int, seed: int = 1, tau: float = 0.55):
    """k-means on a subsample, soft assignment of every pixel. feat: HxWxF z-scored. -> (prob HxWxK, centers)"""
    h, w, f = feat.shape
    flat = feat.reshape(-1, f)
    rng = np.random.default_rng(seed)
    sub = flat[rng.choice(len(flat), size=min(200_000, len(flat)), replace=False)]
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.05)
    _, _, centers = cv2.kmeans(sub.astype(np.float32), k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    d2 = np.empty((len(flat), k), np.float32)
    for i in range(k):
        d2[:, i] = ((flat - centers[i]) ** 2).sum(1)
    logit = -d2 / (2 * tau * tau)
    logit -= logit.max(1, keepdims=True)
    p = np.exp(logit)
    p /= p.sum(1, keepdims=True)
    return p.reshape(h, w, k), centers


def _twig_mask(dead_hard: np.ndarray) -> np.ndarray:
    """Thin elongated non-green components (twigs / needles lying on top) out of the hard dead-colour mask."""
    from skimage import measure
    lab = measure.label(dead_hard, connectivity=2)
    out = np.zeros(dead_hard.shape, bool)
    size = max(dead_hard.shape)
    for r in measure.regionprops(lab):
        if r.area < 40 or r.area > 0.004 * size * size:
            continue
        if r.major_axis_length > 0.02 * size and r.minor_axis_length < 0.006 * size and r.eccentricity > 0.96:
            out[lab == r.label] = True
    return out


def material_masks(base_srgb: np.ndarray, height: np.ndarray, profile: str, wrap: str = "xy", seed: int = 1) -> dict:
    """Soft sub-material masks. -> dict(mask=HxWx4 float (unused channels 0), prob=HxWxC soft class probabilities summing
    to 1 over the profile's real channels, channels=[...], info={...}). Periodic if the inputs are (per-pixel
    clustering + wrap-aware blur)."""
    prof = PROFILES[profile]
    k = prof["k"]
    lab = _lab(base_srgb)
    h, w = height.shape
    cue = blur(height, 1.2, wrap)
    feats = np.stack([lab[..., 0], lab[..., 1], lab[..., 2], cue * 100.0], -1)
    mu, sd = feats.reshape(-1, 4).mean(0), feats.reshape(-1, 4).std(0) + 1e-6
    z = (feats - mu) / sd * np.array([1.0, 0.9, 0.9, 1.0], np.float32)
    p, centers = _kmeans_soft(z.astype(np.float32), k, seed)
    # per-cluster physical stats (hard assignment) to name the clusters
    hard = p.argmax(-1)
    st = []
    for i in range(k):
        m = hard == i
        n = int(m.sum())
        st.append(dict(frac=n / hard.size, L=float(lab[..., 0][m].mean()) if n else 0.0, a=float(lab[..., 1][m].mean()) if n else 0.0,
                       b=float(lab[..., 2][m].mean()) if n else 0.0, h=float(height[m].mean()) if n else 0.0))
    L = np.array([s["L"] for s in st]); A = np.array([s["a"] for s in st]); Hh = np.array([s["h"] for s in st])
    zs = lambda v: (v - v.mean()) / (v.std() + 1e-6)  # noqa: E731
    role = {}                                            # cluster -> channel index
    notes = []
    if profile == "moss":
        soil = int(np.argmin(zs(L) + zs(Hh)))
        role[soil] = 2
        rest = [i for i in range(k) if i != soil]
        hue = np.degrees(np.arctan2(np.array([s_["b"] for s_ in st]), A))      # Lab hue: green-yellow ~100-130, brown/rust < 90
        dead_ids = [i for i in rest if hue[i] < prof["dead_hue_max"]]
        for i in rest:
            role[i] = 1 if i in dead_ids else 0
        if not dead_ids:
            notes.append(f"no brown/dead cluster (min hue {min(hue[i] for i in rest):.0f} deg >= {prof['dead_hue_max']}): dead-moss channel is empty")
        nch = 3
    elif profile == "gravel":
        order = np.argsort(zs(L) + zs(Hh))
        role = {int(order[0]): 1}
        for i in order[1:]:
            role[int(i)] = 0
        nch = 2
    else:
        order = np.argsort(-Hh)                          # highest first
        nch = 3
        for rank, i in enumerate(order):
            role[int(i)] = min(rank * nch // k, nch - 1)
    prob = np.zeros((h, w, nch), np.float32)
    for i, c in role.items():
        prob[..., c] += p[..., i]
    if profile == "moss":
        twig = _twig_mask(prob[..., 1] > 0.5)
        tw = blur(twig.astype(np.float32), 1.0, wrap)
        prob[..., 1] *= (1 - tw)
        prob[..., 0] *= (1 - tw)
        prob[..., 2] *= (1 - tw)
        prob = np.concatenate([prob, tw[..., None]], -1)
        nch = 4
    prob = np.stack([blur(prob[..., c], 1.0, wrap) for c in range(prob.shape[-1])], -1)
    prob /= np.maximum(prob.sum(-1, keepdims=True), 1e-6)
    mask = np.zeros((h, w, 4), np.float32)
    mask[..., :prob.shape[-1]] = prob
    cov = {prof["channels"][c]: round(float(prob[..., c].mean()), 4) for c in range(prob.shape[-1])}
    info = {"profile": profile, "k": k, "clusters": [{"role": prof["channels"][role[i]] if i in role else None, **{kk: round(vv, 3) for kk, vv in st[i].items()}} for i in range(k)],
            "coverage": cov, "notes": notes}
    return {"mask": mask, "prob": prob, "channels": prof["channels"][:prob.shape[-1]], "info": info}


def soil_feather(soil: np.ndarray, size: int, radius: float = 0.004, wrap: str = "xy") -> np.ndarray:
    """Soil-gap membership, feathered: the k-means membership is a near-binary cut-out (soft only over ~1 px), so any
    per-pixel offset keyed on it prints a vector-shape rim. Blur at `radius` x size (~8 px at 2k) and take the max with
    a half-weight wider halo, so every soil-dependent operation (height offset, albedo detail, roughness) ramps over ~10-20 px."""
    a = blur(soil.astype(np.float32), radius * size, wrap)
    b = blur(soil.astype(np.float32), 2.5 * radius * size, wrap)
    return np.clip(np.maximum(a, 0.8 * b), 0, 1).astype(np.float32)


def _hf(lum: np.ndarray, size: int, wrap: str = "xy", lo: float = 0.8, hi: float = 0.012) -> np.ndarray:
    """Band-passed log luminance (relative local contrast, `lo` px .. `hi` x tile, default ~1 px .. 1.2 %): the micro-detail
    that survives in the dark."""
    ll = np.log(np.maximum(lum, 1e-3))
    return (blur(ll, lo, wrap) - blur(ll, hi * size, wrap)).astype(np.float32)


def _dome(live: np.ndarray, size: int, wrap: str = "xy") -> np.ndarray:
    """0..1 'inside a cushion' field: distance from the nearest gap saturating at ~1.2 % of the tile, so cushions
    get rounded shoulders instead of a flat plateau."""
    pad = int(0.05 * size)
    hard = (pad_wrap((live > 0.5).astype(np.uint8), pad, wrap) > 0).astype(np.uint8)
    dist = cv2.distanceTransform(hard, cv2.DIST_L2, 5)[pad:-pad, pad:-pad]
    r0 = 0.012 * size
    return (1 - np.exp(-dist / r0)).astype(np.float32)


# --------------------------------------------------------------------------------------------------------------------
# 3. baked shadow removal
# --------------------------------------------------------------------------------------------------------------------

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def shadow_stats(base_srgb: np.ndarray, occ_ref: np.ndarray) -> dict:
    lum = luminance(srgb_to_lin(base_srgb))
    s = luminance(base_srgb)
    return {"corr_lum_vs_ao": round(_corr(np.log(np.maximum(lum, 1e-4)), occ_ref), 4),
            "pct_below_srgb30": round(float((np.max(base_srgb, -1) * 255 < 30).mean() * 100), 3),
            "pct_luma_below_srgb30": round(float((s * 255 < 30).mean() * 100), 3),
            "mean_srgb_rgb": [round(float(base_srgb[..., c].mean() * 255), 1) for c in range(3)]}


def lift_shadows(base_srgb: np.ndarray, height: np.ndarray, strength: float = 0.85, gain_max: float = 2.0,
                 soil: np.ndarray | None = None, soil_keep: float = 0.5, wrap: str = "xy",
                 occ_ref: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Lift occlusion shadows baked into the albedo.
    occ = multi-scale cavity AO of the height (1 = open). The luminance really follows the occlusion by log L ~ b * log occ
    (b fitted robustly on the band-passed log luminance, so the slow de-lit drift does not matter); the albedo is divided by
    occ^(b * strength), the gain soft-capped at `gain_max` and lightly smoothed. In `soil` (the gap mask) only
    (1 - soil_keep) of the lift is applied, so real dark material stays darker. The mean luminance is kept (as delight does).
    Hue is kept (RGB scaled by the luminance gain)."""
    size = max(base_srgb.shape[:2])
    lin = srgb_to_lin(base_srgb)
    lum = luminance(lin)
    occ = maps.ao_map(height, 1.0, wrap)
    ref = occ if occ_ref is None else occ_ref
    before = shadow_stats(base_srgb, ref)
    ll = np.log(np.maximum(lum, 1e-4))
    hp = ll - blur(ll, 0.06 * size, wrap)
    x = np.log(np.clip(occ, 0.25, 1.0))
    sel = x < -0.02                                   # only pixels that actually sit in a cavity inform the fit
    xs, ys = x[sel].astype(np.float64), hp[sel].astype(np.float64)
    b = 0.0
    if xs.size > 1000:
        w = np.ones_like(xs)
        for _ in range(4):
            b = float((w * xs * ys).sum() / max((w * xs * xs).sum(), 1e-9))
            r = ys - b * xs
            w = 1.0 / (1.0 + (r / (2.0 * np.median(np.abs(r)) + 1e-6)) ** 2)
    b = float(np.clip(b, 0.0, 1.2))
    s_eff = strength * (1.0 - soil_keep * (soil if soil is not None else 0.0))
    g = np.clip(occ, 0.25, 1.0) ** (-b * s_eff)
    g = 1.0 + (gain_max - 1.0) * np.tanh((g - 1.0) / (gain_max - 1.0))     # smooth cap
    g = blur(g, 1.0, wrap)
    out = lin * g[..., None]
    out *= float(lum.mean()) / max(float(luminance(out).mean()), 1e-6)       # keep mean albedo
    res = lin_to_srgb(np.clip(out, 0, 1)).astype(np.float32)
    after = shadow_stats(res, ref)
    info = {"fitted_exponent_b": round(b, 3), "strength": strength, "gain_max": gain_max, "gain_p99": round(float(np.percentile(g, 99)), 3),
            "before": before, "after": after}
    return res, info


# --------------------------------------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------------------------------------

def derive_organic(albedo: np.ndarray, mtype: str, joint, normal_strength, aspect_y: float, height: np.ndarray | None,
                   rough: float | None, name: str, opts: dict, wrap: str = "xy") -> dict:
    """Like maps.derive_all, plus despeckle / material masks / shadow lift. `opts`: despeckle, shadow_lift, materials (bool),
    profile (str|None), shadow_strength (float). Returns derive_all's dict + `albedo` (shadow-lifted), `mask` (HxWx4 or None),
    `organic` (the report that goes into the JSON)."""
    d0 = maps.TYPE_DEFAULTS[mtype]
    rep: dict = {}
    h = height if height is not None else maps.height_map(albedo, joint, d0["joint_w"], wrap)
    if opts.get("despeckle"):
        h, rep["despeckle"] = despeckle_height(h, wrap)
    size = max(h.shape[:2])
    if opts.get("height_blur"):
        # thin blades must be ~flat in height: low-pass the (codex / luminance) height, then give the living (green) areas a
        # gentle tuft-scale mound so the relief comes from clods and tufts, not from every blade outline
        h = blur(h, float(opts["height_blur"]), wrap)
        if opts.get("tuft_mound"):
            gr = albedo[..., 1] - np.maximum(albedo[..., 0], albedo[..., 2])
            gr = blur(gr, 0.010 * size, wrap)
            gr = (gr - gr.mean()) / (gr.std() + 1e-6)
            h = h + float(opts["tuft_mound"]) * gr
            rep["tuft_mound"] = float(opts["tuft_mound"])
        h = maps.soft_unit(h)
        rep["height_blur_px"] = float(opts["height_blur"])
    mk = None
    prof_name = None
    base_r = d0["rough"] if rough is None else rough
    if opts.get("materials"):
        prof_name = pick_profile(name, opts.get("profile"))
        prof = PROFILES[prof_name]
        mk = material_masks(albedo, h, prof_name, wrap)
        prob = mk["prob"]
        nc = prob.shape[-1]
        # height: per-class offsets on a cushion-scale (blurred) mask, plus doming inside the living moss
        soft = np.stack([blur(prob[..., c], 0.010 * size, wrap) for c in range(nc)], -1)   # cushion-scale, so no hard terraces in the normal
        h = h + sum(prof["dh"][c] * soft[..., c] for c in range(nc))
        if prof["dome"] > 0:
            live = prob[..., 0] + prob[..., 1] + (prob[..., 3] if nc > 3 else 0)
            h = h + prof["dome"] * _dome(live, size, wrap) * live
        if prof.get("soil_relief", 0) > 0 and nc > 2 and prof["channels"][2] == "soil_gaps":
            # the dark gaps carry ~no luminance signal in the height (crushed blacks) -> give them the relative micro
            # contrast of the albedo (bright crumbs / debris up, dark pits down), feathered into the moss
            sf = soil_feather(prob[..., 2], size, prof["soil_feather"], wrap)
            lo_, hi_ = prof.get("soil_relief_band", (0.8, 0.012))
            hf = _hf(luminance(albedo), size, wrap, lo_, hi_)
            hf = hf / (float(hf[sf > 0.6].std()) + 1e-6) if (sf > 0.6).any() else hf
            h = h + prof["soil_relief"] * sf * np.clip(hf, -3, 3)
        h = maps.soft_unit(h)      # NOT np.clip: a clip flattened the deepest gaps to one value (plateau + sharp rim)
    n = maps.normal_map(h, normal_strength if normal_strength is not None else d0["normal"], wrap, aspect_y)
    ao = maps.ao_map(h, d0["ao"], wrap)
    if mk is not None:
        prob, prof = mk["prob"], PROFILES[prof_name]
        nc = prob.shape[-1]
        pf = prob.copy()                                  # soil channel feathered for AO / roughness (no cut-out rim)
        if prof["channels"][2:3] == ["soil_gaps"]:
            pf[..., 2] = soil_feather(prob[..., 2], size, prof["soil_feather"], wrap)
        ao_extra = sum(prof["ao"][c] * pf[..., c] for c in range(nc))
        ao = ao * (1 - np.clip(ao_extra, 0, 0.9))
        # roughness: per-class base + local-brightness modulation (sign per profile)
        lum = luminance(albedo)
        ln = maps._norm(blur(lum, 0.003 * size, wrap) - blur(lum, 0.08 * size, wrap))
        r = base_r + sum(prof["rough"][c] * pf[..., c] for c in range(nc)) + prof["lum_slope"] * (ln - 0.5)
        r = np.clip(r, 0.12, 1.0).astype(np.float32)
        rep["materials"] = {**mk["info"], "channels": mk["channels"], "doc": prof["doc"],
                            "roughness_offsets": dict(zip(mk["channels"], prof["rough"])), "base_roughness": base_r,
                            "roughness_by_class": {mk["channels"][c]: round(float((r * prob[..., c]).sum() / max(prob[..., c].sum(), 1e-6)), 3) for c in range(nc)},
                            "roughness_std_before_after": None}
        r_old = maps.roughness_map(albedo, joint, base_r, d0["joint_rough"], wrap)
        rep["materials"]["roughness_std_before_after"] = [round(float(r_old.std()), 4), round(float(r.std()), 4)]
    else:
        r = maps.roughness_map(albedo, joint, base_r, d0["joint_rough"], wrap)
    alb = albedo
    if opts.get("shadow_lift"):
        soil = None
        if mk is not None and PROFILES[prof_name]["channels"][2:3] == ["soil_gaps"]:
            soil = mk["prob"][..., 2]
        elif mk is not None and prof_name == "gravel":
            soil = mk["prob"][..., 1]
        sk = PROFILES[prof_name].get("soil_keep", 0.5) if prof_name else 0.5
        if soil is not None:
            soil = soil_feather(soil, size, PROFILES[prof_name].get("soil_feather", 0.004), wrap)
        alb, rep["shadow_lift"] = lift_shadows(albedo, h, opts.get("shadow_strength", 0.85), soil=soil, soil_keep=sk, wrap=wrap)
        det = PROFILES[prof_name].get("soil_detail", 1.0) if prof_name else 1.0
        if soil is not None and det != 1.0:
            # restore / boost relative micro-contrast in the gaps (feathered): the albedo there is a few sRGB levels
            # above black, where 8-bit quantisation + the blacks floor in calibrate would otherwise flatten it
            lin = srgb_to_lin(alb)
            hf = _hf(luminance(lin), size, wrap)
            lin = lin * np.exp((det - 1.0) * soil * hf)[..., None]
            alb = lin_to_srgb(np.clip(lin, 0, 1)).astype(np.float32)
    out = {"height": h, "normal": n, "ao": ao, "rough": r, "orm": maps.pack_orm(ao, r, d0["metallic"]), "albedo": alb,
           "mask": None if mk is None else mk["mask"], "organic": rep}
    if mk is not None:
        out["organic"]["materials"]["file_channels"] = {"R": mk["channels"][0], "G": mk["channels"][1] if len(mk["channels"]) > 1 else None,
                                                         "B": mk["channels"][2] if len(mk["channels"]) > 2 else None,
                                                         "A": mk["channels"][3] if len(mk["channels"]) > 3 else None}
    return out
