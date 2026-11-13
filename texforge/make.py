"""`texforge make`: source image -> seamless tiling PBR set (BaseColor / Normal / ORM / Height + JSON)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from . import maps, organic, pattern, seamless
from .common import load_image, resize, resize_periodic, save_png
from .delight import delight

TYPES = ("auto", "brick", "tile", "cobble", "plaster", "organic")
RELIEF_CM = {"brick": 1.5, "tile": 0.3, "cobble": 3.0, "plaster": 0.4, "organic": 2.0}


def write_set(out_dir: Path, name: str, base: np.ndarray, derived: dict, extra: dict) -> dict:
    """Write the four PNGs (BaseColor sRGB; Normal/ORM/Height linear) and return the files dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    items = [("basecolor", base), ("normal", derived["normal"]), ("orm", derived["orm"]), ("height", derived["height"])]
    if derived.get("mask") is not None:
        items.append(("mask", derived["mask"]))
    for key, arr in items:
        fname = f"{name}_{ {'basecolor': 'BaseColor', 'normal': 'Normal', 'orm': 'ORM', 'height': 'Height', 'mask': 'Mask'}[key] }.png"
        save_png(out_dir / fname, arr)
        files[key] = fname
    return files


def _codex_height(base, mtype, jt, name, out_dir, size, mode, height_image, info, notes, log):
    """Height from codex (image-to-image edit of the albedo); None -> caller falls back to luminance."""
    from . import codexmaps, gen
    h_lum = maps.height_map(base, jt, maps.TYPE_DEFAULTS[mtype]["joint_w"])
    cx = out_dir / "cxh"
    cx.mkdir(exist_ok=True)
    try:
        if height_image is None:
            inp = cx / f"{name}_albedo_in.png"
            save_png(inp, resize(base, (1254, 1254)))
            height_image = gen.generate_edit(codexmaps.HEIGHT_PROMPT, inp, cx, f"{name}_codex_height")
        h_raw = codexmaps.load_gray(height_image, size)
    except Exception as e:  # noqa: BLE001 - any codex failure must not lose the rest of the set
        notes.append(f"codex height failed ({e}); luminance height used")
        log(notes[-1])
        return None
    h, rep, ok = codexmaps.prepare_height(base, h_raw, h_lum, jt, mode, soft=(mtype == "organic"))
    info["height_from"] = mode
    info["codex_height"] = rep
    if not ok:
        notes.append("codex height did not register with the albedo well enough (see params.codex_height); luminance height used")
        log(notes[-1])
        return None
    return h


def _codex_normal(base, mtype, name, out_dir, size, normal_image, h_refs, n_height, aspect_y, detail, balance, info, notes, log):
    """Normal from codex (image-to-image edit of the albedo); None -> caller keeps the height-derived normal."""
    from PIL import Image

    from . import codexmaps, gen
    cx = out_dir / "cxh"
    cx.mkdir(exist_ok=True)
    try:
        if normal_image is None:
            inp = cx / f"{name}_albedo_in.png"
            save_png(inp, resize(base, (1254, 1254)))
            normal_image = gen.generate_edit(codexmaps.NORMAL_PROMPT, inp, cx, f"{name}_codex_normal")
        raw = np.asarray(Image.open(normal_image).convert("RGB"), np.float32) / 255
        raw = resize(raw, (size, size))
    except Exception as e:  # noqa: BLE001
        notes.append(f"codex normal failed ({e}); height-derived normal used")
        log(notes[-1])
        return None
    n, rep, ok = codexmaps.prepare_normal(base, raw, h_refs, n_height, mtype, aspect_y, detail, balance=balance)
    info["normal_from"] = "codex"
    info["codex_normal"] = rep
    if not ok:
        notes.append("codex normal convention is ambiguous (channel correlation < 0.15 or references disagree, see "
                     "params.codex_normal); height-derived normal used")
        log(notes[-1])
        return None
    return n


def make(image: Path, name: str, out_root: Path, mtype: str = "auto", size: int = 2048, world_cm: float = 200.0,
         normal_strength: float | None = None, overlap: float = 1 / 6, bond: str = "auto",
         height_from: str = "luminance", height_image: Path | None = None,
         normal_from: str = "height", normal_image: Path | None = None, normal_detail: float = 0.0,
         normal_balance: bool = False, rough: float | None = None,
         despeckle: bool | None = None, shadow_lift: bool | None = None, materials: bool | None = None,
         material_profile: str | None = None, shadow_strength: float = 0.85,
         height_blur: float = 0.0, tuft_mound: float = 0.0, log=print) -> dict:
    """despeckle / shadow_lift / materials: None = default for the type (on for organic, despeckle also for cobble / plaster,
    all off for brick / tile); True / False force it. material_profile: auto (from the name) | moss | gravel | generic."""
    src = load_image(image)
    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image, out_dir / f"{name}_source.png") if Path(image).suffix.lower() == ".png" else save_png(out_dir / f"{name}_source.png", src)
    notes: list[str] = []
    h0, w0 = src.shape[:2]
    if min(h0, w0) < size // 2:
        notes.append(f"source is {w0}x{h0}; output {size}px is a {size / min(h0, w0):.1f}x upscale (soft detail)")

    # 1. de-light (patterned materials need a wider kernel than the pattern period)
    dl = delight(src, 0.15)
    # 2. pattern detection
    mask = pattern.joint_mask(dl)
    if mtype == "auto":
        mtype = pattern.classify(dl, mask)
        log(f"auto type -> {mtype}")
    info: dict = {"method": None}
    tile, joint = None, None
    # 3/4. make it tile
    if mtype in ("brick", "tile"):
        try:
            angle = pattern.estimate_skew(mask)
            if angle:
                dl2 = pattern.deskew(dl, angle)
                mask = pattern.joint_mask(dl2)
                dl = dl2
                info["deskew_deg"] = angle
            tile, ginfo = pattern.snap_to_grid(dl, mask, bond=bond)
            tile = seamless.heal_seams(tile, 10)
            info.update(method="grid-snap + seam heal", **ginfo)
            joint = pattern.joint_mask(tile, wrap=True)
        except pattern.PatternError as e:
            notes.append(f"grid snap failed ({e}); fell back to joint-biased min-cut")
            log(notes[-1])
    if tile is None:
        if mtype in ("brick", "tile", "cobble"):
            tile = seamless.periodic_xy(dl, overlap if mtype == "cobble" else 1 / 5, mask, joint_bias=0.7)
            joint = pattern.joint_mask(tile, wrap=True)
            info["method"] = "min-cut (joint-biased)"
        else:
            tile = seamless.periodic_xy(delight(src, 0.12), overlap)
            info["method"] = "min-cut"
    th, tw = tile.shape[:2]
    # 5. albedo at final resolution; the tile may be non-square -> square texture + per-axis world size
    sq = (size, size)
    base = np.clip(resize_periodic(tile, sq), 0, 1)
    jt = None if joint is None else np.clip(resize_periodic(joint, sq), 0, 1)
    world = [float(world_cm), float(world_cm) * th / tw]
    # 6-9. height, normal, roughness, AO
    height = None
    if height_from != "luminance":
        height = _codex_height(base, mtype, jt, name, out_dir, size, height_from, height_image, info, notes, log)
    base_raw = base
    opts = organic.defaults_for(mtype)
    for k_, v_ in (("despeckle", despeckle), ("shadow_lift", shadow_lift), ("materials", materials)):
        if v_ is not None:
            opts[k_] = bool(v_)
    opts.update(profile=material_profile, shadow_strength=shadow_strength, height_blur=height_blur, tuft_mound=tuft_mound)
    if any(opts[k_] for k_ in ("despeckle", "shadow_lift", "materials")):
        d = organic.derive_organic(base, mtype, jt, normal_strength, world[0] / world[1], height, rough, name, opts)
        base = d["albedo"]
        if d["organic"]:
            info["organic"] = d["organic"]
    else:
        d = maps.derive_all(base, mtype, jt, normal_strength, aspect_y=world[0] / world[1], height=height, rough=rough)
    if normal_from == "codex":
        h_lum = maps.height_map(base_raw, jt, maps.TYPE_DEFAULTS[mtype]["joint_w"])
        refs = [d["height"]] + ([h_lum] if height is not None else [])
        cn = _codex_normal(base, mtype, name, out_dir, size, normal_image, refs, d["normal"], world[0] / world[1],
                           normal_detail, normal_balance, info, notes, log)
        if cn is not None:
            d["normal"] = cn
    files = write_set(out_dir, name, base, d, {})
    sidecar = {
        "name": name, "kind": "tile", "type": mtype, "size": [size, size], "world_cm": world,
        "normal_convention": "directx", "height_depth_cm": RELIEF_CM[mtype],
        "tileable": True, "tile_axes": "xy", "files": files,
        "orm_channels": "R=AO, G=Roughness, B=Metallic",
        **({"mask_channels": {"file": files["mask"], **d["organic"]["materials"]["file_channels"], "doc": d["organic"]["materials"]["doc"],
                              "encoding": "linear, soft memberships (sum to 1), sRGB off / TC_Masks"}} if "mask" in files else {}),
        "params": {"source": str(image), "source_size": [w0, h0], "native_tile_px": [tw, th],
                   "normal_strength": normal_strength if normal_strength is not None else maps.TYPE_DEFAULTS[mtype]["normal"],
                   "overlap": overlap, **{k: v for k, v in info.items()}},
        "notes": notes,
    }
    (out_dir / f"{name}.texforge.json").write_text(json.dumps(sidecar, indent=2, default=float))
    return sidecar
