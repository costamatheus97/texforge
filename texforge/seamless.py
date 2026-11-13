"""Seam-optimal (min-cut) blending, as in image quilting / graph-cut texture synthesis.

Make an image periodic along x: overlap its right edge onto its left edge, find the vertical path through
the overlap where both sides look the same (dynamic programming on colour + gradient error) and switch from
one side to the other along that path. The path is forced to close on itself vertically so the next axis
pass cannot reintroduce a seam at the corner.
"""
from __future__ import annotations

import cv2
import numpy as np


def _grad(a: np.ndarray) -> np.ndarray:
    g = a.mean(-1) if a.ndim == 3 else a
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.stack([gx, gy], -1)


def cyclic_path(cost: np.ndarray, n_starts: int = 16, smooth: int = 1) -> np.ndarray:
    """Min-cost top-to-bottom path (|dx|<=smooth per row) whose last row lands next to its first (cyclic).

    cost: (H, O). Returns int array (H,) of column indices.
    """
    H, O = cost.shape
    starts = np.unique(np.linspace(0, O - 1, min(n_starts, O)).astype(int))
    S = len(starts)
    INF = 1e18
    acc = np.full((S, O), INF, np.float64)
    acc[np.arange(S), starts] = cost[0, starts]
    back = np.zeros((H, S, O), np.int8)
    for y in range(1, H):
        best = acc.copy()
        arg = np.zeros((S, O), np.int8)
        for d in range(1, smooth + 1):
            l = np.full_like(acc, INF); l[:, d:] = acc[:, :-d]
            r = np.full_like(acc, INF); r[:, :-d] = acc[:, d:]
            m = l < best; best = np.where(m, l, best); arg = np.where(m, d, arg)   # predecessor is column j-d
            m = r < best; best = np.where(m, r, best); arg = np.where(m, -d, arg)  # predecessor is column j+d
        acc = best + cost[y][None, :]
        back[y] = arg
    # the end must be within `smooth` of the start column so wrapping vertically is continuous
    cols = np.arange(O)[None, :]
    ok = np.abs(cols - starts[:, None]) <= smooth
    final = np.where(ok, acc, INF)
    s, e = np.unravel_index(np.argmin(final), final.shape)
    path = np.zeros(H, int)
    path[-1] = e
    for y in range(H - 1, 0, -1):
        path[y - 1] = path[y] - back[y, s, path[y]]
    return path


def periodic_x(img: np.ndarray, overlap_frac: float = 1 / 6, joint: np.ndarray | None = None,
               joint_bias: float = 0.0, feather: float = 1.5) -> np.ndarray:
    """Return an image of width W-overlap that tiles seamlessly along x.

    joint: optional 0..1 mask of joints/gaps; with joint_bias>0 the cut prefers pixels that are a gap on
    both sides, so for stones it runs through the gaps instead of halving a stone.
    """
    H, W = img.shape[:2]
    o = int(max(8, round(W * overlap_frac)))
    o = min(o, W // 2 - 1)
    Wp = W - o
    head = img[:, :o]
    tail = img[:, Wp:]
    e = ((head - tail) ** 2).sum(-1)
    gh, gt = _grad(head), _grad(tail)
    e = e + 0.5 * ((gh - gt) ** 2).sum(-1) * 4.0
    if joint is not None and joint_bias > 0:
        both = np.minimum(joint[:, :o], joint[:, Wp:])
        e = e * (1.0 - joint_bias * both)
    # mild preference for staying near the centre of the overlap keeps both sides' content around the cut
    cx = np.abs(np.arange(o) - o / 2) / (o / 2)
    e = e + 0.02 * float(e.mean()) * cx[None, :] ** 2
    path = cyclic_path(e)
    cols = np.arange(o)[None, :]
    m = (cols >= path[:, None]).astype(np.float32)  # 1 -> head, 0 -> tail
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=feather, sigmaY=0.01)
    m = m[..., None]
    blend = tail * (1 - m) + head * m
    return np.concatenate([blend, img[:, o:Wp]], axis=1).astype(np.float32)


def periodic_xy(img: np.ndarray, overlap_frac: float = 1 / 6, joint: np.ndarray | None = None,
                joint_bias: float = 0.0) -> np.ndarray:
    """Make an image periodic along both axes (x pass, then y pass on the transposed result)."""
    a = periodic_x(img, overlap_frac, joint, joint_bias)
    j = None
    if joint is not None:
        j = periodic_x(joint[..., None].repeat(3, -1), overlap_frac)[..., 0]
        # (joint mask is blended the same way only approximately; it is only used as a cost weight)
    a = periodic_x(np.ascontiguousarray(a.transpose(1, 0, 2)), overlap_frac,
                   None if j is None else np.ascontiguousarray(j.T), joint_bias)
    return np.ascontiguousarray(a.transpose(1, 0, 2))


def heal_seams(a: np.ndarray, band: int = 10) -> np.ndarray:
    """Spread the residual step across the wrap-around border of an already (nearly) periodic tile.

    Per row, the difference between the last and first pixel is distributed over `band` px on both sides
    with a linear falloff (a 1-D gradient-domain fix). It removes the hairline left by hard cuts through
    speckled mortar without moving any structure. Applied along x, then along y."""
    out = a.astype(np.float32).copy()
    for axis in (1, 0):
        t = out if axis == 1 else out.transpose(1, 0, 2)
        w = t.shape[1]
        b = min(band, w // 8)
        jump = t[:, 0] - t[:, -1]
        wgt = (1 - np.arange(b) / b)[None, :, None].astype(np.float32)
        t[:, :b] -= 0.5 * jump[:, None, :] * wgt
        t[:, w - 1 - np.arange(b)] += 0.5 * jump[:, None, :] * wgt
        out = t if axis == 1 else t.transpose(1, 0, 2)
    return np.ascontiguousarray(out)
