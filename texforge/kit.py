"""`texforge kit`: a set of materials that share one capture style, one real-world scale, a calibrated albedo range and a
comparable height range, built from a kit.json (see kits/forest/kit.json)."""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import calibrate, prompts, qa
from . import gen as genmod
from .common import load_image, luminance, resize, save_png


def load_kit(path: str | Path) -> dict:
    kit = json.loads(Path(path).read_text())
    out = Path(kit.get("out", f"out/{kit['name']}"))
    kit["out_dir"] = out if out.is_absolute() else Path.cwd() / out
    return kit


def retry(fn, tries: int = 4, what: str = "step", log=print):
    """Run fn(); an out-of-memory error (a shared machine may be short of memory) is retried after a pause."""
    for i in range(tries):
        try:
            return fn()
        except (MemoryError, OSError) as e:
            if i == tries - 1 or (isinstance(e, OSError) and e.errno != 12):
                raise
            log(f"[kit] {what}: out of memory, retrying ({i + 1}/{tries - 1})")
            gc.collect()
            time.sleep(8)


def layer_prompt(kit: dict, layer: dict) -> str:
    return prompts.kit_prompt(layer["subject"], layer.get("scale", ""), kit.get("style", prompts.KIT_STYLE)
                              if "world_cm" not in layer else _style_for(kit, layer))


def _style_for(kit: dict, layer: dict) -> str:
    """The shared style, with the 2 m frame replaced when a layer has its own real-world size (trunk bark: 1 m)."""
    s = kit.get("style", prompts.KIT_STYLE)
    cm = layer["world_cm"]
    return s.replace("exactly 2 metres x 2 metres", f"exactly {cm / 100:g} metres x {cm / 100:g} metres")


def build_layer(kit: dict, layer: dict, gen: bool = True, log=print, sources_only: bool = False) -> dict | None:
    from .make import make
    name = layer["name"]
    out_root = kit["out_dir"]
    ldir = out_root / name
    ldir.mkdir(parents=True, exist_ok=True)
    src = ldir / f"{name}_src_1.png"
    runs = 0
    if not src.exists():
        if not gen:
            log(f"[kit] {name}: no source image and --no-gen")
            return None
        prompt = layer_prompt(kit, layer)
        (ldir / f"{name}_prompt.txt").write_text(prompt)
        try:
            genmod.generate(prompt, ldir, name, n=1, parallel=1)
            runs += 1
        except RuntimeError as e:
            print(f"[kit] {name}: {e}", file=sys.stderr)
            genmod.print_manual(prompt)
            return None
    if sources_only:
        return {"source": str(src), "codex_runs": runs}
    cx = ldir / "cxh"
    hi = cx / f"{name}_codex_height.png"
    ni = cx / f"{name}_codex_normal.png"
    hf = layer.get("height_from", "hybrid")
    nf = layer.get("normal_from", "height")
    before = {p.name for p in cx.glob("*codex*.png")} if cx.exists() else set()
    sc = retry(lambda: make(src, name, out_root, layer.get("type", "organic"), kit.get("size", 2048),
                            layer.get("world_cm", kit["world_cm"]), layer.get("normal_strength"),
                            height_from=hf, height_image=hi if hi.exists() else None,
                            normal_from=nf, normal_image=ni if ni.exists() else None,
                            normal_detail=layer.get("normal_detail", 0.0), rough=layer.get("rough"),
                            despeckle=layer.get("despeckle"), shadow_lift=layer.get("shadow_lift"), materials=layer.get("materials"),
                            material_profile=layer.get("material_profile"), shadow_strength=layer.get("shadow_strength", 0.85),
                            height_blur=layer.get("height_blur", 0.0), tuft_mound=layer.get("tuft_mound", 0.0), log=log),
               what=f"make {name}", log=log)
    after = {p.name for p in cx.glob("*codex*.png")} if cx.exists() else set()
    runs += len(after - before)
    gc.collect()
    # kit consistency: calibrate the albedo, normalise the height
    bp = ldir / f"{name}_BaseColor.png"
    hp = ldir / f"{name}_Height.png"
    base = load_image(bp)
    hc = layer.get("hue_chroma")                      # optional per-layer Lab hue / chroma correction (see calibrate.correct_hue_chroma)
    hc_rep = None
    if hc:
        base, hc_rep = calibrate.correct_hue_chroma(base, hc)
    bal = layer.get("albedo_balance_lin")             # optional per-layer white balance (linear RGB multipliers), before calibration
    if bal:
        from .common import lin_to_srgb, srgb_to_lin
        base = lin_to_srgb(np.clip(srgb_to_lin(base) * np.array(bal, np.float32), 0, 1))
    stats_in = calibrate.albedo_stats(base)
    floor = layer.get("floor_lin", kit["albedo"]["floor_lin"])     # per-layer: a layer with deliberate dark gaps may use a lower floor
    cal, cinfo = calibrate.calibrate_albedo(base, tuple(layer["albedo_mean_lin"]), floor, kit["albedo"]["ceil_lin"])
    save_png(bp, cal)
    h = luminance(load_image(hp))
    hstats_in = calibrate.height_stats(h)
    hn = calibrate.normalise_height(h, kit["height"]["mean"], kit["height"]["std"])
    save_png(hp, hn)
    sc["kit"] = {"layer": name, "hue_chroma": hc_rep, "albedo_balance_lin": bal, "role": layer.get("role", "ground"), "capture_style": "overcast top-down shadowless",
                 "albedo_before": stats_in, "albedo_calibration": cinfo, "height_before": hstats_in,
                 "height_after": calibrate.height_stats(hn), "codex_runs_this_build": runs,
                 "codex_runs_total": int(src.exists()) + int(hi.exists()) + int(nf == "codex" and ni.exists()),
                 "height_from": hf, "normal_from": nf, "normal_used": sc["params"].get("codex_normal", {}).get("accepted", False) if nf == "codex" else False}
    (ldir / f"{name}.texforge.json").write_text(json.dumps(sc, indent=2, default=float))
    log(f"[kit] {name}: done  mean lin {stats_in['mean_lin']} -> {calibrate.albedo_stats(cal)['mean_lin']}  codex runs {runs}")
    return sc


def build(kit_path: str, only: list[str] | None = None, gen: bool = True, log=print, sources_only: bool = False) -> int:
    kit = load_kit(kit_path)
    names = only or [l["name"] for l in kit["layers"]]
    ok = 0
    for layer in kit["layers"]:
        if layer["name"] in names:
            ok += build_layer(kit, layer, gen, log, sources_only) is not None
    return 0 if ok else 1


def check(kit_path: str, log=print) -> int:
    """Per layer: albedo range, height mean/spread, scale, seam metrics, which normal was used. Writes kit_report.json
    and a swatch sheet; exit code 1 when a layer is outside the kit rules."""
    kit = load_kit(kit_path)
    rows, bad = [], 0
    swatches = []
    for layer in kit["layers"]:
        name = layer["name"]
        d = kit["out_dir"] / name
        if not (d / f"{name}_BaseColor.png").exists():
            log(f"[kit] {name}: not built")
            continue
        meta = json.loads((d / f"{name}.texforge.json").read_text())
        base = load_image(d / f"{name}_BaseColor.png")
        a = calibrate.albedo_stats(base)
        h = calibrate.height_stats(luminance(load_image(d / f"{name}_Height.png")))
        rep = qa.check(d, name)
        lo, hi = layer["albedo_mean_lin"]
        fl, ce = layer.get("floor_lin", kit["albedo"]["floor_lin"]), kit["albedo"]["ceil_lin"]
        fails = []
        if not lo - 1e-3 <= a["mean_lin"] <= hi + 1e-3:
            fails.append(f"mean lum {a['mean_lin']} outside band [{lo}, {hi}]")
        if a["p0.5_lin"] < fl * 0.9:
            fails.append(f"crushed blacks (p0.5 {a['p0.5_lin']} < {fl})")
        if a["p99.5_lin"] > ce + 1e-3:
            fails.append(f"blown highlights (p99.5 {a['p99.5_lin']} > {ce})")
        if abs(h["mean"] - kit["height"]["mean"]) > 0.03:
            fails.append(f"height mean {h['mean']}")
        want_cm = layer.get("world_cm", kit["world_cm"])
        if abs(meta["world_cm"][0] - want_cm) > 0.5:
            fails.append(f"world_cm {meta['world_cm'][0]} != {want_cm}")
        fails += rep["fails"]
        bad += bool(fails)
        n = meta["params"].get("codex_normal", {})
        rows.append({"layer": name, "role": layer.get("role"), "world_cm": meta["world_cm"], "albedo": a, "height": h,
                     "seams": {k: rep["after"][k]["color"] for k in rep["after"]},
                     "height_from": meta["kit"]["height_from"], "height_accepted": meta["params"].get("codex_height", {}).get("accepted"),
                     "normal_from": layer.get("normal_from"), "normal_codex": {k: n.get(k) for k in ("accepted", "flip_x", "flip_y", "corr_x_y_per_reference")} if n else None,
                     "codex_runs": meta["kit"]["codex_runs_total"], "fails": fails, "notes": meta["notes"]})
        swatches.append(resize(base, (256, 256)))
    rep_name = kit.get("report", "kit")            # a kit file may write its own report (kit_v2.json -> kit_v2_report.json)
    (kit["out_dir"] / f"{rep_name}_report.json").write_text(json.dumps(rows, indent=2, default=float))
    if swatches:
        save_png(kit["out_dir"] / f"{rep_name}_swatches.png", np.concatenate(swatches, 1))
    hdr = f"{'layer':13s} {'cm':>4s} {'mean_lin':>8s} {'p0.5':>6s} {'p99.5':>6s} {'srgb p0.5/p99.5':>15s} {'h mean':>6s} {'h std':>6s} {'h p5-p95':>9s}  seams(B/N/O/H)  normal  height  fails"
    print(hdr)
    for r in rows:
        a, h, s = r["albedo"], r["height"], r["seams"]
        print(f"{r['layer']:13s} {r['world_cm'][0]:4.0f} {a['mean_lin']:8.3f} {a['p0.5_lin']:6.3f} {a['p99.5_lin']:6.3f} {a['srgb_p0.5']:7d}/{a['srgb_p99.5']:<7d} "
              f"{h['mean']:6.3f} {h['std']:6.3f} {h['p5']:.2f}-{h['p95']:.2f}  "
              f"{s['BaseColor']:.2f}/{s['Normal']:.2f}/{s['ORM']:.2f}/{s['Height']:.2f}  {('codex' if r['normal_from'] == 'codex' and r['normal_codex'] and r['normal_codex'].get('accepted') else 'height'):6s} {('codex' if r['height_accepted'] else 'lum'):6s}  {'; '.join(r['fails']) or 'ok'}")
    return 1 if bad else 0


def compare_normals(kit_path: str, names: list[str] | None = None, log=print) -> None:
    """Lit comparison (3 lights) of each codex-normal layer against the height-derived normal of the same layer.
    Rebuilds the layer into <out>/_cmp with --normal-from height from the existing codex height (no codex runs)."""
    from PIL import Image, ImageDraw

    from .make import make
    kit = load_kit(kit_path)
    cmp_root = kit["out_dir"] / "_cmp"
    lights = kit["blend"]["lights"]

    def label(img, text):
        im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        ImageDraw.Draw(im).rectangle([0, 0, 8 * len(text) + 8, 16], fill=(0, 0, 0))
        ImageDraw.Draw(im).text((4, 2), text, fill=(255, 255, 0))
        return np.asarray(im, np.float32) / 255

    for layer in kit["layers"]:
        n = layer["name"]
        if layer.get("normal_from") != "codex" or (names and n not in names):
            continue
        d = kit["out_dir"] / n
        hi = d / "cxh" / f"{n}_codex_height.png"
        retry(lambda: make(d / f"{n}_src_1.png", n, cmp_root, layer.get("type", "organic"), kit.get("size", 2048),
                           layer.get("world_cm", kit["world_cm"]), height_from=layer.get("height_from", "hybrid"),
                           height_image=hi, normal_from="height", rough=layer.get("rough"), log=lambda *_: None), what=f"compare {n}")
        base = load_image(d / f"{n}_BaseColor.png")
        orm = load_image(d / f"{n}_ORM.png")
        rows = []
        for tag, p in (("normal from height", cmp_root / n / f"{n}_Normal.png"), ("normal from codex (auto-corrected)", d / f"{n}_Normal.png")):
            nm = load_image(p)
            rows.append(np.concatenate([label(qa.render_lit(base, nm, orm, L, tiles=1, px=512), tag) for L in lights], 1))
        save_png(d / f"{n}_normal_compare.png", np.concatenate(rows, 0))
        log(f"[kit] {n}: {d / (n + '_normal_compare.png')}")
        gc.collect()
