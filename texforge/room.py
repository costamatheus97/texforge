"""`texforge room`: one perspective room-interior image -> rectified 'room box' atlas for fake interiors.

The image is expected to be a one-point-perspective view from the middle of the open front wall: the back wall
is a frontal rectangle (default: the middle 50%), the floor / ceiling / side walls converge to it. Each of the
five surfaces is rectified with a homography into a square atlas cell so a ray-box intersection in the window
shader can sample them (see unreal/create_master_material.py, M_TexforgeWindowRoom).

Atlas: 3 columns x 2 rows of equal square cells (cell size = size/3 px), cell (col, row):
    (0,0) back wall    u = left->right,  v = top->bottom
    (1,0) left wall    u = front->back,  v = top->bottom
    (2,0) right wall   u = front->back,  v = top->bottom
    (0,1) floor        u = left->right,  v = front->back
    (1,1) ceiling      u = left->right,  v = front->back
    (2,1) unused (mid grey)
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import gen as genmod
from . import prompts
from .common import load_image, save_png

LAYOUT = {"back": [0, 0], "left": [1, 0], "right": [2, 0], "floor": [0, 1], "ceiling": [1, 1]}


def build_atlas(img: np.ndarray, rect: tuple[float, float, float, float], cell: int) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = rect[0] * w, rect[1] * h, rect[2] * w, rect[3] * h
    # source quads as TL, TR, BR, BL, ordered to match each cell's (u, v) axes
    quads = {
        "back": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        "left": [(0, 0), (x0, y0), (x0, y1), (0, h - 1)],          # u: front(image edge) -> back, v: top -> bottom
        "right": [(w - 1, 0), (x1, y0), (x1, y1), (w - 1, h - 1)],
        "floor": [(0, h - 1), (w - 1, h - 1), (x1, y1), (x0, y1)],  # u: left -> right, v: front -> back
        "ceiling": [(0, 0), (w - 1, 0), (x1, y0), (x0, y0)],
    }
    dst = np.float32([[0, 0], [cell - 1, 0], [cell - 1, cell - 1], [0, cell - 1]])
    atlas = np.full((2 * cell, 3 * cell, 3), 0.5, np.float32)
    for k, q in quads.items():
        M = cv2.getPerspectiveTransform(np.float32(q), dst)
        warped = cv2.warpPerspective(img, M, (cell, cell), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        c, r = LAYOUT[k]
        atlas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = warped
    return np.clip(atlas, 0, 1)


def make_room(prompt: str, name: str, out_root: Path, image: str | None = None, size: int = 2048,
              back_wall: str = "0.25,0.25,0.75,0.75", log=print) -> dict:
    out_dir = Path(out_root) / name
    if image is None:
        files = genmod.generate(prompts.room_prompt(prompt), out_dir, name, n=1)
        image = str(files[0])
    img = load_image(image)
    rect = tuple(float(v) for v in back_wall.split(","))
    cell = size // 3
    atlas = build_atlas(img, rect, cell)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_png(out_dir / f"{name}_Room.png", img)
    save_png(out_dir / f"{name}_RoomAtlas.png", atlas)
    sidecar = {
        "name": name, "kind": "room", "type": "room", "size": [3 * cell, 2 * cell], "world_cm": [300.0, 250.0],
        "normal_convention": "directx", "tileable": False, "tile_axes": "",
        "files": {"room": f"{name}_Room.png", "atlas": f"{name}_RoomAtlas.png"},
        "room": {"back_wall_rect": list(rect), "depth_ratio": 1.0, "atlas_cell_px": cell,
                 "atlas_layout": {"columns": 3, "rows": 2, "cells": LAYOUT,
                                  "axes": "back u=L->R v=T->B; left/right u=front->back v=T->B; floor/ceiling u=L->R v=front->back"}},
        "params": {"source": str(image)},
        "notes": ["depth_ratio (room depth / width) is a guess; tune it on the material instance (RoomDepthRatio)"],
    }
    (out_dir / f"{name}.texforge.json").write_text(json.dumps(sidecar, indent=2))
    return sidecar
