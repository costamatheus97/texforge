"""texforge command line."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import gen as genmod
from . import prompts

DEFAULT_OUT = Path(os.environ.get("TEXFORGE_OUT") or "out")   # relative to the current directory unless TEXFORGE_OUT is set


def _cmd_gen(a) -> int:
    kinds = {"material": prompts.material_prompt, "decal": prompts.decal_prompt, "room": prompts.room_prompt}
    text = Path(a.prompt_file).read_text().strip() if a.prompt_file else a.prompt
    prompt = text if a.raw else kinds[a.kind](text)
    out = Path(a.out) / a.name
    try:
        files = genmod.generate(prompt, out, a.name, n=a.n, parallel=a.parallel)
    except RuntimeError as e:
        print(f"[gen] {e}", file=sys.stderr)
        genmod.print_manual(prompt)
        return 2
    for f in files:
        print(f)
    print(f"next: texforge make {files[0]} --name {a.name} --type auto")
    return 0


def _cmd_kit(a) -> int:
    from . import kit
    if a.action == "build":
        return kit.build(a.kit, a.only.split(",") if a.only else None, gen=not a.no_gen, sources_only=a.sources_only)
    if a.action == "compare":
        kit.compare_normals(a.kit, a.only.split(",") if a.only else None)
        return 0
    if a.action == "blend":
        from . import blend
        blend.run(a.kit)
        return 0
    if a.action == "macro":
        from . import macro
        k = kit.load_kit(a.kit)
        m = k["macro"]
        macro.make_macro(k["out_dir"], m["name"], m["size"], m["world_cm"], m.get("seed", 7))
        return 0
    if a.action == "atlas":
        from . import atlas
        return atlas.build(a.kit, image=a.image, gen=not a.no_gen)
    return kit.check(a.kit)


def _cmd_make(a) -> int:
    from .make import make
    sc = make(Path(a.image), a.name, Path(a.out), a.type, a.size, a.world_cm, a.normal_strength, bond=a.bond,
              height_from=a.height_from, height_image=Path(a.height_image) if a.height_image else None,
              normal_from=a.normal_from, normal_image=Path(a.normal_image) if a.normal_image else None,
              normal_detail=a.normal_detail, normal_balance=a.normal_balance, rough=a.rough,
              despeckle=False if a.no_despeckle else (True if a.despeckle else None),
              shadow_lift=False if a.no_shadow_lift else (True if a.shadow_lift else None),
              materials=False if a.no_materials else (True if a.materials else None),
              material_profile=a.material_profile, shadow_strength=a.shadow_strength)
    print(json.dumps({k: sc[k] for k in ("name", "type", "size", "world_cm", "files")}, indent=1))
    for n in sc["notes"]:
        print("note:", n)
    print(f"wrote {Path(a.out) / a.name}   next: texforge check {a.name}")
    return 0


def _cmd_check(a) -> int:
    from .qa import check
    rep = check(Path(a.out) / a.name, a.name)
    if rep.get("kind") == "trim":
        print("per-strip seam ratios (colour/gradient, 1.0 = invisible):")
        for k, v in rep["strips"].items():
            print(f"  {k:40s} {v['color_x']} / {v['grad_x']}")
        for f in rep["fails"]:
            print("FAIL:", f)
        print("PASS" if rep["passed"] else "FAILED")
        return 0 if rep["passed"] else 1
    b = rep["before"]
    print(f"seam ratios (1.0 = invisible)  after: " + "  ".join(f"{k} {v['color']}/{v['grad']}" for k, v in rep["after"].items()))
    if b:
        print(f"before (raw source tiled): colour {b['BaseColor']['color']} gradient {b['BaseColor']['grad']}")
    print(f"sheets: {Path(a.out) / a.name / 'check'}")
    for f in rep["fails"]:
        print("FAIL:", f)
    print("PASS" if rep["passed"] else "FAILED")
    return 0 if rep["passed"] else 1


def _cmd_compare(a) -> int:
    from .compare import compare
    tint = lambda t: tuple(float(x) for x in t.split(",")) if t else None  # noqa: E731
    root = Path(a.out)
    r = compare(root / a.old, a.old, root / a.new, a.new, Path(a.sheet), tint(a.old_tint), tint(a.new_tint), a.exposure)
    print(json.dumps(r))
    return 0


def _cmd_trim(a) -> int:
    from .trim import make_trim
    sc = make_trim(a.strips, a.name, Path(a.out), image=a.image, size=a.size, world_cm=a.world_cm, bounds=a.bounds)
    print(json.dumps({"strips": sc["strips"]}, indent=1))
    return 0


def _cmd_decal(a) -> int:
    from .decal import make_decal
    sc = make_decal(a.prompt, a.name, Path(a.out), image=a.image, background=a.background, size=a.size,
                    world_cm=a.world_cm, surface=a.surface)
    print(json.dumps({k: sc[k] for k in ("name", "size", "world_cm", "files")}, indent=1))
    return 0


def _cmd_room(a) -> int:
    from .room import make_room
    sc = make_room(a.prompt, a.name, Path(a.out), image=a.image, size=a.size, back_wall=a.back_wall)
    print(json.dumps({k: sc[k] for k in ("name", "files", "room")}, indent=1))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="texforge", description="AI image -> seamless game-ready PBR texture sets")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="output root (default: $TEXFORGE_OUT, else ./out)")
    p.add_argument("--generator", choices=genmod.BACKENDS,
                   help="image generator backend (default: $TEXFORGE_GENERATOR, else codex). none = never generate; "
                        "command = run $TEXFORGE_GENERATOR_COMMAND")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate source image(s) with the codex CLI")
    g.add_argument("prompt", nargs="?", default="", help="material description, e.g. 'weathered red brick wall' (or use --prompt-file)")
    g.add_argument("--name", required=True)
    g.add_argument("--n", type=int, default=1, help="number of variations (one codex run each)")
    g.add_argument("--kind", choices=["material", "decal", "room"], default="material")
    g.add_argument("--raw", action="store_true", help="send the prompt as-is (no template)")
    g.add_argument("--parallel", type=int, default=3, help="concurrent codex runs (use 1 when parallel runs time out)")
    g.add_argument("--prompt-file", help="read the prompt from this file instead of the positional argument")
    g.set_defaults(fn=_cmd_gen)

    m = sub.add_parser("make", help="source image -> seamless PBR set")
    m.add_argument("image")
    m.add_argument("--name", required=True)
    m.add_argument("--type", choices=["auto", "brick", "tile", "cobble", "plaster", "organic"], default="auto")
    m.add_argument("--size", type=int, default=2048)
    m.add_argument("--world-cm", type=float, default=200.0, help="world width of one tile in cm")
    m.add_argument("--normal-strength", type=float, default=None)
    m.add_argument("--bond", choices=["auto", "stack"], default="auto", help="brick/tile: auto-detect running bond or force stack bond")
    m.add_argument("--height-from", choices=["luminance", "codex", "hybrid"], default="luminance",
                   help="luminance (default, no codex) | codex (height drawn by codex from the finished albedo, 1 codex run) | "
                        "hybrid (codex layout + fine luminance micro-relief). Falls back to luminance if it does not register.")
    m.add_argument("--height-image", help="use this existing grey height image instead of asking codex (same layout as the albedo)")
    m.add_argument("--normal-from", choices=["height", "codex"], default="height",
                   help="height (default: derived from the height map) | codex (normal drawn by codex from the finished albedo, "
                        "1 codex run; red/green convention auto-corrected, made valid and periodic). Falls back to height if the "
                        "convention cannot be determined reliably.")
    m.add_argument("--normal-image", help="use this existing codex normal image instead of asking codex")
    m.add_argument("--normal-detail", type=float, default=0.0,
                   help="--normal-from codex: RNM-blend this much fine luminance relief on top (0 = off, 0.3-1 = visible)")
    m.add_argument("--rough", type=float, default=None, help="base roughness override (0..1), default per type")
    m.add_argument("--normal-balance", action="store_true",
                   help="--normal-from codex: rescale the x/y slopes to the x/y RMS ratio of the height-derived normal")
    # organic / ground extras: default ON for --type organic (despeckle also cobble / plaster), OFF for brick / tile
    for flag, what in (("despeckle", "clean isolated height spikes / pits before normal + AO"),
                       ("shadow-lift", "lift baked-in mid/high-frequency shadows in the albedo (albedo /= f(AO))"),
                       ("materials", "per-sub-material roughness / height / AO + write <name>_Mask.png")):
        m.add_argument(f"--no-{flag}", action="store_true", help=f"disable: {what}")
        m.add_argument(f"--{flag}", action="store_true", help=f"force on (also for non-organic types): {what}")
    m.add_argument("--material-profile", choices=["auto", "moss", "gravel", "generic"], default="auto",
                   help="sub-material profile for the masks (auto: from the name, else generic)")
    m.add_argument("--shadow-strength", type=float, default=0.85, help="shadow lift strength 0..1 (default 0.85)")
    m.set_defaults(fn=_cmd_make)

    c = sub.add_parser("check", help="seam metric + 3x3 tiled sheets + lit preview for a finished set")
    c.add_argument("name")
    c.set_defaults(fn=_cmd_check)

    cp = sub.add_parser("compare", help="old vs new set lit by a low grazing warm light (3x3 view, 1:1 crop, mask, roughness)")
    cp.add_argument("old", help="old set name (dir under --out)")
    cp.add_argument("new", help="new set name (dir under --out)")
    cp.add_argument("--sheet", required=True, help="output PNG")
    cp.add_argument("--old-tint", help="r,g,b linear multiplier applied to the OLD albedo (the tint the game uses today)")
    cp.add_argument("--new-tint", help="r,g,b linear multiplier applied to the NEW albedo")
    cp.add_argument("--exposure", type=float, default=1.6)
    cp.set_defaults(fn=_cmd_compare)

    t = sub.add_parser("trim", help="trim sheet of horizontal strips (generate with codex or use --image)")
    t.add_argument("strips", help="';'-separated strip descriptions, top to bottom")
    t.add_argument("--name", required=True)
    t.add_argument("--image", help="existing trim-sheet image instead of generating one")
    t.add_argument("--size", type=int, default=2048)
    t.add_argument("--world-cm", type=float, default=200.0, help="world length of the sheet width in cm")
    t.add_argument("--bounds", help="comma-separated strip boundaries as image fractions, overrides detection")
    t.set_defaults(fn=_cmd_trim)

    d = sub.add_parser("decal", help="decal (rain streaks, rust, damp, cracks) with alpha")
    d.add_argument("prompt", help="what to draw, e.g. 'rust streaks'")
    d.add_argument("--name", required=True)
    d.add_argument("--image", help="existing decal-on-plain-background image instead of generating one")
    d.add_argument("--background", choices=["white", "black"], default="white")
    d.add_argument("--surface", choices=["rust", "damp", "crack", "dirt", "generic"], default="generic")
    d.add_argument("--size", type=int, default=2048, help="longest side of the output")
    d.add_argument("--world-cm", type=float, default=150.0, help="world width of the decal in cm")
    d.set_defaults(fn=_cmd_decal)

    r = sub.add_parser("room", help="fake-interior room image + atlas for window glass")
    r.add_argument("prompt")
    r.add_argument("--name", required=True)
    r.add_argument("--image")
    r.add_argument("--size", type=int, default=2048)
    r.add_argument("--back-wall", default="0.25,0.25,0.75,0.75", help="x0,y0,x1,y1 of the back wall in the image")
    r.set_defaults(fn=_cmd_room)

    k = sub.add_parser("kit", help="build / check a kit of materials that share style, scale, albedo and height ranges")
    k.add_argument("action", choices=["build", "check", "blend", "macro", "atlas", "compare"])
    k.add_argument("kit", help="kit.json, e.g. kits/forest/kit.json")
    k.add_argument("--only", help="comma-separated layer names")
    k.add_argument("--sources-only", action="store_true", help="build: only generate the source images (one codex run per layer)")
    k.add_argument("--image", help="atlas: existing sprite-sheet image instead of asking codex")
    k.add_argument("--no-gen", action="store_true", help="never call codex; layers without a source image are skipped")
    k.set_defaults(fn=_cmd_kit)

    a = p.parse_args(argv)
    genmod.set_backend(a.generator)
    return a.fn(a)


def main_entry() -> None:
    """Console-script entry point (`texforge`)."""
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
