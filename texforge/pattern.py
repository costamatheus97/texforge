"""Pattern analysis for patterned materials (brick, tile, cobble): joint mask, deskew, course/brick
detection, and the grid snap that makes courses repeat exactly.

Grid snap (what the video calls 'snap the joints to an exact grid, cut each row in the gaps between bricks
and stretch slightly so copies match at the edge'):
  1. bed joints (horizontal) are found from the row profile of the joint mask;
  2. every course (the band between two bed joints) is resampled to the same height;
  3. inside each course the head joints (vertical) are found; the same number of whole bricks is cut from
     head-joint centre to head-joint centre, each brick is stretched to the exact common pitch, and the
     course is rolled by its original half-brick phase so running bond survives;
  4. an even number of courses is kept so the stagger also matches across the vertical wrap.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# ------------------------------------------------------------------ joint mask

def joint_mask(srgb: np.ndarray, wrap: bool = False) -> np.ndarray:
    """Binary (float 0/1) mask of joints/gaps: 2-means in Lab; the minority cluster is the joint.
    wrap=True treats the image as a periodic tile so the mask itself has no seam."""
    if wrap:
        p = 32
        return joint_mask(np.pad(srgb, ((p, p), (p, p), (0, 0)), mode="wrap"))[p:-p, p:-p]
    h, w = srgb.shape[:2]
    small = cv2.resize(srgb, (max(64, w * 512 // max(h, w)), max(64, h * 512 // max(h, w))), interpolation=cv2.INTER_AREA)
    lab_s = cv2.cvtColor((np.clip(small, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    # soften single-pixel speckle before clustering
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, centers = cv2.kmeans(lab_s, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    lab = cv2.cvtColor((np.clip(cv2.GaussianBlur(srgb, (0, 0), max(1.0, 0.002 * max(h, w))), 0, 1) * 255).astype(np.uint8),
                       cv2.COLOR_RGB2LAB).astype(np.float32)
    d0 = ((lab - centers[0]) ** 2).sum(-1)
    d1 = ((lab - centers[1]) ** 2).sum(-1)
    lab0 = (d0 < d1)
    frac0 = lab0.mean()
    m = lab0 if frac0 < 0.5 else ~lab0
    m = m.astype(np.uint8)
    k = max(3, int(0.004 * max(h, w)) | 1)
    m = cv2.medianBlur(m * 255, k) // 255
    return m.astype(np.float32)


# ------------------------------------------------------------------ deskew

def estimate_skew(mask: np.ndarray, max_deg: float = 3.0) -> float:
    """Angle (deg) that makes horizontal joints most row-aligned (maximises variance of the row profile)."""
    h, w = mask.shape
    s = 400.0 / max(h, w)
    m = cv2.resize(mask, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    mh, mw = m.shape
    best, best_a = -1.0, 0.0
    scores = {}
    for a in np.arange(-max_deg, max_deg + 1e-6, 0.25):
        R = cv2.getRotationMatrix2D((mw / 2, mh / 2), a, 1.0)
        r = cv2.warpAffine(m, R, (mw, mh), flags=cv2.INTER_LINEAR, borderValue=0)
        c = r[int(mh * 0.15):int(mh * 0.85), int(mw * 0.15):int(mw * 0.85)]
        scores[a] = float(c.mean(1).var())
    best_a = max(scores, key=scores.get)
    if scores[best_a] < 1.08 * scores[0.0]:   # not clearly better than "no rotation"
        return 0.0
    return float(best_a)


def deskew(srgb: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.05:
        return srgb
    h, w = srgb.shape[:2]
    R = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    r = cv2.warpAffine(srgb, R, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    m = int(np.ceil(np.sin(np.radians(abs(angle))) * max(h, w) / 2)) + 4
    return r[m:h - m, m:w - m]


# ------------------------------------------------------------------ 1-D periodicity helpers

def autocorr_period(p: np.ndarray, min_lag: int, max_lag: int, rel: float = 0.5) -> tuple[float, float]:
    """(period, strength) of a 1-D signal: first autocorrelation peak that reaches `rel` of the strongest."""
    x = p - p.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / max(ac[0], 1e-12)
    max_lag = min(max_lag, n - 2)
    seg = ac[:max_lag + 1]
    pk, _ = find_peaks(seg, height=0.1)
    pk = pk[pk >= min_lag]
    if len(pk) == 0:
        return 0.0, 0.0
    top = seg[pk].max()
    first = pk[np.argmax(seg[pk] >= rel * top)]
    # parabolic sub-sample refinement
    y0, y1, y2 = seg[first - 1], seg[first], seg[min(first + 1, len(seg) - 1)]
    den = y0 - 2 * y1 + y2
    off = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    return float(first + off), float(seg[first])


def peak_centers(p: np.ndarray, period: float, thr: float = 0.5, pad: int | None = None) -> list[tuple[float, float]]:
    """Sub-pixel centres of the (joint) peaks of a 1-D profile, reflect-padded so half-visible edge joints count.
    Returns [(position, strength)] for positions inside [-0.15 period, n + 0.15 period)."""
    n = len(p)
    pad = pad if pad is not None else int(period)
    pp = np.pad(p, pad, mode="reflect")
    peaks, _ = find_peaks(pp, height=thr * float(p.max()), distance=max(2, int(0.6 * period)))
    out = []
    r = max(2, int(period / 4))
    for k in peaks:
        lo, hi = max(0, k - r), min(len(pp), k + r + 1)
        seg = pp[lo:hi]
        wts = np.maximum(seg - 0.5 * pp[k], 0)
        c = (np.arange(lo, hi) * wts).sum() / max(wts.sum(), 1e-9) - pad
        if -0.15 * period <= c < n + 0.15 * period - 1:
            out.append((float(c), float(pp[k])))
    return out


def _regularise(cs: list[float], period: float) -> list[float]:
    """Insert a virtual centre into over-long gaps, drop centres crowding a neighbour."""
    cs = sorted(cs)
    out = [cs[0]]
    for c in cs[1:]:
        gap = c - out[-1]
        if gap < 0.55 * period:
            continue
        k = int(round(gap / period))
        if k > 1:
            base = out[-1]
            out.extend(base + gap * i / k for i in range(1, k))
        out.append(c)
    return out


# ------------------------------------------------------------------ the grid snap

class PatternError(RuntimeError):
    pass


def analyse_courses(mask: np.ndarray) -> dict:
    h, w = mask.shape
    prof = gaussian_filter1d(mask.mean(1).astype(np.float32), 1.5)
    period, strength = autocorr_period(prof, int(0.02 * h), int(0.5 * h))
    if period < 4 or strength < 0.25:
        raise PatternError(f"no regular horizontal joints (period={period:.1f}, strength={strength:.2f})")
    pk = peak_centers(prof, period, thr=0.5)
    cs = [c for c, _ in pk if 0 <= c <= h - 1]
    if len(cs) < 3:
        raise PatternError("fewer than 3 bed joints found")
    cs = _regularise(cs, period)
    sp = np.diff(cs)
    if sp.std() / sp.mean() > 0.12:
        raise PatternError(f"bed joints irregular (spacing CV {sp.std() / sp.mean():.2f})")
    return {"centers": cs, "pitch": float(np.median(sp)), "strength": strength}


def analyse_bricks(mask: np.ndarray, cs: list[float], pitch_y: float) -> dict:
    h, w = mask.shape
    qs = []
    for k in range(len(cs) - 1):
        y0, y1 = int(cs[k] + 0.3 * pitch_y), int(cs[k + 1] - 0.3 * pitch_y)
        if y1 - y0 < 3:
            raise PatternError("courses too thin to analyse")
        qs.append(mask[y0:y1].mean(0).astype(np.float32))
    # period of head joints: average the per-course autocorrelations
    acs = []
    for q in qs:
        x = q - q.mean()
        f = np.fft.rfft(x, 2 * w)
        ac = np.fft.irfft(f * np.conj(f))[:w]
        acs.append(ac / max(ac[0], 1e-12))
    ac = np.mean(acs, 0)
    pk, _ = find_peaks(ac[: w // 2 + 1], height=0.1)
    pk = pk[pk >= int(0.04 * w)]
    if len(pk) == 0:
        raise PatternError("no head-joint periodicity")
    top = ac[pk].max()
    first = pk[np.argmax(ac[pk] >= 0.5 * top)]
    pitch_x = float(first)
    courses = []
    for q in qs:
        qq = gaussian_filter1d(q, 1.2)
        pks = peak_centers(qq, pitch_x, thr=0.45)
        xs = _regularise([c for c, _ in pks], pitch_x)
        courses.append(xs)
    return {"pitch": pitch_x, "courses": courses}


def snap_to_grid(srgb: np.ndarray, mask: np.ndarray, bond: str = "auto", target_aspect: float = 1.0) -> tuple[np.ndarray, dict]:
    """Return a tile whose courses and bricks sit on an exact grid and that tiles in both axes."""
    h, w = mask.shape
    cinfo = analyse_courses(mask)
    cs, py = cinfo["centers"], cinfo["pitch"]
    binfo = analyse_bricks(mask, cs, py)
    px = binfo["pitch"]
    nseg = [len(x) - 1 for x in binfo["courses"]]
    Nb = min(nseg)
    if Nb < 2:
        raise PatternError(f"fewer than 2 whole bricks per course (per-course counts {nseg})")
    Pc = int(round(py))
    bt = int(round(px))
    Wt = Nb * bt
    n_avail = len(cs) - 1
    # half-brick phase of each course relative to the first (snapped to {0, half a brick})
    x0 = binfo["courses"][0][0]
    raw = [((binfo["courses"][k][0] - x0) / px) % 1.0 for k in range(n_avail)]
    stagger = bond != "stack" and np.mean([0.25 <= p < 0.75 for p in raw]) > 0.25
    # running bond needs an even number of courses (the stagger must also match across the vertical wrap);
    # stack bond (grid tiles) can use any count. Prefer the most content whose aspect stays near the target.
    cand = list(range(2, n_avail + 1, 2)) if stagger else list(range(1, n_avail + 1))
    if not cand:
        raise PatternError("fewer than 2 courses")
    ok = [k for k in cand if 0.7 <= k * Pc / Wt / target_aspect <= 1.4]
    n = max(ok) if ok else min(cand, key=lambda k: abs(k * Pc / Wt - target_aspect))
    phases, starts = [], []
    for k in range(n):
        xs = binfo["courses"][k]
        js = (len(xs) - 1 - Nb) // 2
        starts.append(js)
        ph = ((xs[js] - x0) / px) % 1.0
        if not stagger:
            ph = 0.0
        else:  # snap the stagger to {0, half a brick}
            ph = 0.5 if 0.25 <= ph < 0.75 else 0.0
        phases.append(ph)
    Ht = n * Pc
    mapx = np.zeros((Ht, Wt), np.float32)
    mapy = np.zeros((Ht, Wt), np.float32)
    xo = np.arange(Wt, dtype=np.float32)
    for k in range(n):
        xs = binfo["courses"][k]
        js = starts[k]
        knots_out = np.arange(Nb + 1) * bt
        knots_src = np.array(xs[js:js + Nb + 1], np.float32)
        shift = int(round(phases[k] * bt))
        xi = (xo - shift) % Wt
        sx = np.interp(xi, knots_out, knots_src).astype(np.float32)
        r = np.arange(Pc, dtype=np.float32)
        sy = (cs[k] + r / Pc * (cs[k + 1] - cs[k])).astype(np.float32)
        mapx[k * Pc:(k + 1) * Pc] = sx[None, :]
        mapy[k * Pc:(k + 1) * Pc] = sy[:, None]
    tile = cv2.remap(srgb, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    info = {"courses": n, "bricks_per_course": Nb, "course_px": Pc, "brick_px": bt, "tile_px": [Wt, Ht],
            "running_bond": bool(stagger), "phases": phases, "bed_joint_strength": round(cinfo["strength"], 3), "bond": bond}
    return np.clip(tile, 0, 1), info


# ------------------------------------------------------------------ classification

def classify(srgb: np.ndarray, mask: np.ndarray) -> str:
    """Heuristic material type for --type auto."""
    from .common import luminance
    try:
        c = analyse_courses(mask)
        b = analyse_bricks(mask, c["centers"], c["pitch"])
        # aligned head joints between neighbouring courses -> stack bond (tile); alternating -> brick
        x0 = b["courses"][0][0]
        ph = [(((xs[0] - x0) / b["pitch"]) % 1.0) for xs in b["courses"]]
        stag = np.mean([0.25 <= p < 0.75 for p in ph])
        return "brick" if stag > 0.25 else "tile"
    except PatternError:
        pass
    frac = float(mask.mean())
    # cobble = a network of gaps splitting the image into many *large* cells; dirt/gravel splits into specks
    inv = cv2.erode((mask < 0.5).astype(np.uint8), np.ones((3, 3), np.uint8))
    ncomp, lab, stats, _ = cv2.connectedComponentsWithStats(inv)
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = areas[areas > 0.0015 * mask.size]
    if 0.05 < frac < 0.45 and len(big) >= 12 and big.sum() > 0.6 * max(areas.sum(), 1):
        return "cobble"
    lum = luminance(cv2.GaussianBlur(srgb, (0, 0), 3))
    return "plaster" if lum.std() < 0.03 else "organic"
