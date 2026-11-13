# texforge

AI-painted image -> game-ready, seamlessly tiling PBR texture set (BaseColor / Normal / ORM / Height + JSON),
plus trim sheets, decals, fake-interior rooms and an Unreal import step. Pure Python (numpy, OpenCV, SciPy,
scikit-image, Pillow), no frameworks.

## Install

Python 3.10 or newer.

```bash
python3 -m venv .venv && . .venv/bin/activate     # any virtualenv, kept outside or inside the checkout
pip install -e .                                  # installs the `texforge` command (minimum versions in pyproject.toml)
# pip install -r requirements.txt                 # optional: the exact versions this was tested with
texforge --help
```

Without installing, `bin/texforge` runs the checkout with the active virtualenv's Python (or `$TEXFORGE_PYTHON`,
or `python3` on PATH); the dependencies must already be installed there.

Everything lands in `./out/<name>/` (override with `$TEXFORGE_OUT` or the global `--out DIR`, placed before the
sub-command). Kit files resolve a relative `"out"` against the current directory.

## Image generators

Only `gen`, the `--height-from codex|hybrid` / `--normal-from codex` options of `make`, and the generating steps of
`trim`, `decal`, `room` and `kit build|atlas` create images; everything else works from an existing image. The
generator is a backend chosen by `TEXFORGE_GENERATOR` or the global `--generator` option (the option wins):

| backend | what it does |
|---|---|
| `codex` (default) | the `codex` CLI, see the codex sections below |
| `none` | never generates; every generation step stops with an error that says to pass an existing image (`make <image>`, `--image`, `--height-image`, `--normal-image`) |
| `command` | runs the command template in `TEXFORGE_GENERATOR_COMMAND` for every image |

```bash
texforge --generator none make my_photo.png --name wall          # no generator needed
export TEXFORGE_GENERATOR=command
export TEXFORGE_GENERATOR_COMMAND='my-image-tool --prompt-file {prompt_file} --init {input} --out {output}'
texforge gen "weathered red clay brick wall" --name brick_a
```

The template is split like a shell line and run without a shell. In every argument `{prompt}` (the text),
`{prompt_file}` (the same text in a file), `{input}` (the image to edit, image-to-image only; an argument that is
exactly `{input}` is dropped when there is none) and `{output}` (the PNG the command must write) are replaced; the
same values are in the environment as `TEXFORGE_PROMPT`, `TEXFORGE_PROMPT_FILE`, `TEXFORGE_INPUT` and
`TEXFORGE_OUTPUT`. The command must exit 0 and leave the file at `{output}`. The code is `texforge/gen.py`
(`_one` is the single entry point every generation request goes through).

## Examples

```bash
# 1. generate 3 variations of a material with codex, ~1 min each (writes out/brick_a/brick_a_src_1..3.png)
texforge gen "weathered red clay brick wall in running bond, light grey mortar" --name brick_a --n 3

# 2. turn one of them into a seamless PBR set, 2048 px, one tile = 2 m wide
texforge make out/brick_a/brick_a_src_2.png --name brick_a --type auto --size 2048 --world-cm 200

# 3. QA: seam metric (fails with exit code 1), before/after 3x3 sheets, 1:1 corner crops, lit GIF
texforge check brick_a

# 4. trim sheet: strips top->bottom, each tiles left-right on its own; V ranges go into the JSON
texforge trim "stone cornice with dentils; plain plaster band; red brick band; limestone sill; wooden frame; corrugated zinc" --name facade_trim

# 5. decal with soft alpha (rust / damp / crack / dirt presets), and a fake room for window glass
texforge decal "orange rust streaks running down" --name rust_a --surface rust
texforge room "cosy furnished living room" --name room_a --back-wall 0.283,0.314,0.716,0.725
```

`make` works on any image (a photo, a Poly Haven scan, a Midjourney render); `gen` is only a convenience (see Image generators).
`--type auto` classifies brick / tile / cobble / plaster / organic; force it when you know better.

## What each command does

### `make` (pipeline)
1. **De-light**: cubic polynomial fit of the log-luminance (kills ramps and one-sided light, correct at the
   borders) + a wide Gaussian for blotchy drift, applied in linear light, mean albedo preserved.
2. **Pattern analysis**: joint mask (2-means in Lab, the minority cluster is the joint), deskew (+-3 deg,
   row-profile variance), autocorrelation for the course pitch and the brick pitch, sub-pixel joint centres.
3. **brick / tile -> grid snap**: bed joints snapped to equal course height; in every course the same number of
   whole bricks is cut from head-joint centre to head-joint centre, each brick stretched to the exact common
   pitch, the course rolled by its original half-brick phase (running bond survives; an even course count is
   kept for the vertical wrap; stack bond / grid tiles may use any count). A 10 px 1-D step-healing pass
   removes the hairline left by cutting through speckled mortar.
4. **cobble / plaster / organic -> min-cut seams** (image quilting): overlap the right edge on the left, find the
   min-error vertical path (colour + gradient, cyclic so the second axis cannot reintroduce a corner seam),
   switch sides along it, repeat for y. For cobble the cost is lowered where both sides are a gap, so the cut
   runs through the gaps instead of halving a stone. A failed grid snap falls back to this and says so.
5. Resample (periodic cubic, phase-exact), then **height** (joints low + band-passed luminance), **normal**
   (Sobel, DirectX/Unreal convention: green points DOWN the image; verified with a synthetic bump),
   **roughness** (per-type base + luminance + joints), **AO** (multi-scale cavity of the height).
   All filters wrap, so the derived maps tile as well as the albedo.

Outputs in `out/<name>/`: `_BaseColor.png` (sRGB), `_Normal.png` (linear, DirectX), `_ORM.png` (linear,
R=AO G=Roughness B=Metallic), `_Height.png` (linear 8-bit), `_source.png`, `<name>.texforge.json`
(type, size, `world_cm` = [width, height] of one tile, relief depth, parameters, notes).
Non-square tiles (a brick tile of 3 bricks x 12 courses is 939x1260 px) are stored as a square texture;
`world_cm` carries the true per-axis size, so UV = world position / world_cm per axis in the material.

### `check`
Writes `out/<name>/check/`: `*_before_after.png` (raw source tiled 3x3 | result tiled 3x3), `*_after_3x3.png`,
`*_after_corner.png` (1:1 crop where four tiles meet; red ticks in the margin mark the seam lines, small seams
disappear in the downscaled 3x3 so always look at this one), `*_lit.gif` (Blinn-Phong plane, light circling,
uses Normal + ORM) and `*_lit_sheet.png` (three light directions + AO / roughness / height), `*_check.json`.

Seam metric: for several band widths (1 px .. ~1.5% of the tile), the step across the wrap-around border divided
by the 99.5th percentile of the same step measured at every interior position, worst axis. **<= 1 means the seam
is no bigger than the largest step the texture itself contains.** Also run on Normal, ORM and Height, plus a
slope-jump (second difference) metric. `check` exits non-zero if BaseColor colour/gradient > 1.15 or another map
> 1.30. Trim sheets are checked per strip, along x only.

### `trim`
Generates (or takes `--image`) a sheet of horizontal strips. Strip bounds come from a dynamic-programming
partition of the rows (or `--bounds 0,.17,.34,...`). Each strip: horizontal de-light, then either an exact
whole number of detected periods (dentils, corrugation) or a min-cut seam, scaled to the sheet width, stacked
with edge-extended gutters so mips do not bleed. `strips[]` in the JSON has `name, v0, v1, height_cm,
world_cm_width` (v = 0 is the top row of the image). Strips tile in U only. Metals (zinc/steel...) get
metallic 0.9.

### `decal`
Alpha from distance to the plain background, 6% border fade (no hard rectangle), colour un-premultiplied and
dilated into transparent texels (no halo in mips), crop to the mark. Outputs BaseColor (RGBA), `_Opacity`
(grey), Normal, ORM, Height, `_preview.png` (over grey, over a coloured wall, and the mask).
Presets: `rust`, `damp` (glossy, no relief), `crack`, `dirt`, `generic`.

### `room`
One one-point-perspective room image -> `_Room.png` plus `_RoomAtlas.png`: back wall, left, right, floor,
ceiling rectified by homography into a 3x2 atlas (layout in the sidecar and in `texforge/room.py`). Give the
back-wall rectangle with `--back-wall x0,y0,x1,y1` (fractions); check `_RoomAtlas.png` that all five cells look
flat.

## codex image generation: what works here (codex-cli 0.166, ChatGPT login)

* `codex exec --skip-git-repo-check --ephemeral -s workspace-write --json -C <dir> "<instruction>"` makes the
  model call its built-in image tool (feature `image_generation`, on by default). The PNG is written by codex
  to `~/.codex/generated_images/<thread-id>/*.png`; the thread id is in the first `thread.started` JSON event,
  and `gen` copies exactly that file (3 runs in parallel do not mix up).
* ~60 s per image. Size is chosen by the model from the prompt: 1254x1254 for square, 1536x1024 for wide
  prompts, 1774x887 for the room. You cannot request 2K/4K, so `make --size 2048` upscales ~1.6x.
* One image per run, so `--n K` = K runs. The result is not seedable or reproducible.
* The prompt template (`texforge/prompts.py`) asks for orthographic, flat-lit, no shadows, no vignette, filling
  the frame. It mostly obeys; it never produces a guaranteed-seamless image (that is what `make` is for) and
  it ignores exact strip heights in trim sheets (detected afterwards).
* If `codex` is missing, not logged in or returns no image, `gen` prints the exact prompt for you to paste into
  any image generator and exits with 2; everything else works on any input image.

## Height from codex instead of luminance (`make --height-from`)

`make ... --height-from {luminance,codex,hybrid}` (default `luminance`). With `codex`/`hybrid` the 2048 albedo is
sent to `codex exec -i` as an image EDIT ("grayscale height map, same layout, white = raised"); one extra codex
run (~60 s). `--height-image PATH` reuses an existing codex height instead of calling codex. The result is
registered against the albedo (phase correlation + optical flow), checked against a gate (offset <= 3 px, flow
p95 <= 6 px, edge correlation >= 0.6 x the luminance reference; otherwise it falls back to luminance and writes a
note), made periodic by fading its 3 % border into the luminance height, smoothed (sigma 2.5 px, the codex height
carries generator grain) and, for `hybrid`, topped up with 0.22 x fine luminance micro-detail.
Findings (brick / cobble / dirt): offsets <= 1.3 px, flow p95 <= 3.8 px, so image-to-image keeps the layout.
hybrid > codex-only > luminance on brick and cobble (no stain-driven pits, clean rounded edges); marginal on dirt.
Not adopted from codex: ROUGHNESS (median 0.5-0.6 and joints smoother than faces, i.e. too glossy and inverted
against any sane roughness). Use it only as inspiration.

## Normal from codex (`make --normal-from codex`)

Default stays `height` (derived from the height map). `--normal-from codex` asks codex for a tangent-space normal
(image edit of the finished albedo, 1 codex run; `--normal-image PATH` reuses an existing one). The raw codex
normal is not usable as delivered: its red/green convention changes from run to run (brick: both channels
inverted, cobble: only green), 4.5 % of cobble vectors point backwards and it does not tile. `make` therefore:
1. decides each channel's flip from the SIGN of its correlation with the gradient of the set's height (the codex
   height with `--height-from codex|hybrid`) and of the luminance height; a channel with |corr| < 0.15, or two
   references that disagree, falls back to the height-derived normal and writes a note (brick x is the weakest
   case: -0.20 / -0.17, still consistent);
2. flips, clamps z >= 0.05, renormalises, writes DirectX/Unreal (green down);
3. fades its 3 % border into the height-derived normal in vector space and renormalises, so it tiles (seam ratio
   brick 1.20, cobble 1.04, limit 1.30);
4. optional `--normal-detail D` (default 0): RNM-blend fine luminance relief of strength D on top;
   optional `--normal-balance`: rescale x/y slopes to the x/y RMS ratio of the height-derived normal (brick: no-op,
   gains 1.05 / 0.95; cobble: x 0.71, y 1.41, changes the look, off by default).

Recommended per material: **`--normal-from codex --height-from hybrid` for masonry and stone (brick, cobble,
tile with stone-like faces)**; both were preferred over the height-derived normal in lit renders (rounded bevels,
less grain than the luminance-based relief). Plaster, dirt and organic: keep `height` (not tested, the luminance
relief has no layout to improve). Example:

    texforge make samples/brick/brick_src_1.png --name brick --type brick --height-from hybrid --normal-from codex

## Unreal (run later, inside the editor; NOT run yet)

Editor-Python scripts in `integrations/unreal/` (see their docstrings; configure the `CONFIG` dict or `TEXFORGE_*` env
vars, editor Python has no argv):

1. `create_master_material.py` once: builds `/Game/Texforge/M_TexforgeMaster`, `M_TexforgeMaster_Masked`,
   `M_TexforgeDecal`, `M_TexforgeWindowRoom`, plus tiny default textures. Output Log: `py "/path/to/texforge/integrations/unreal/create_master_material.py"`.
2. `unreal_import.py` per set: imports the PNGs into a `/Game/...` folder with the right settings and creates
   `MI_<name>`:
   BaseColor sRGB + default compression | Normal TC_Normalmap (no green flip, it is already DirectX) |
   ORM TC_Masks sRGB off | Height TC_Grayscale sRGB off | Opacity TC_Masks sRGB off.
   Tile sets use world-aligned (cm) projection with `WorldSizeCm` from the JSON; trims use mesh UVs (and
   `TRIM_STRIP` sets the V window); decals use the decal master; rooms the window-room master.

Master material options (all static switches, so a disabled option costs nothing; costs in the script header):

| option | what | cost / note |
|---|---|---|
| `UseMacroVariation` | large-scale noise brightness variation | cheap, first thing to try (anti-tiling 1) |
| `UseTwoScaleBlend` | second sample at 0.37x, rotated, blended by a noise mask | 2x texture samples (anti-tiling 2) |
| `UseHexTiling` | hex / texture-bombing, 3 blended samples | 3x samples; non-patterned materials only, **breaks brick and tile joints** (anti-tiling 3) |
| `UsePOM` | parallax occlusion from Height | 8..32 samples/pixel; close-up only |
| `UseSilhouetteClip` | "Crimson Desert trick": in the Masked variant, discard pixels whose parallax-offset UV leaves the surface at grazing angles, so outlines get bumpy | masked blend (no early-Z), needs POM, mesh UVs |
| `UseDirt` | world-Z gradient dirt near the ground (+ noise) | ~50 ALU, no samples |
| `UseEdgeWear` | edge mask (vertex colour channel or `EdgeMaskTex`, broken up by Height) blends to an "under" material set (`Under_BaseColor/Normal/ORM`, e.g. plaster -> brick), whiteout normal blend | 3-4 extra samples |

Window room: `M_TexforgeWindowRoom` does a ray-vs-box intersection against the room atlas (an alternate switch
uses the engine's `InteriorCubemap` function with a cube texture). The scripts were only dry-run against a stub
of the `unreal` module: expect to fix a pin or property name and to read the Output Log / shader compile errors
on the first run (`# VERIFY` comments mark every guess).

Anti-repetition beyond that (a second detail scale, hex tiling, decals, vertex-colour variation) is shader
and level-art work, not part of the texture generation.

## Forest-floor kit (`texforge kit ...`, `kits/forest/kit.json`)

A kit is one JSON file: shared real-world scale (`world_cm`, 200 cm per tile), shared capture style (overcast, top-down,
orthographic, shadowless, a 2 m x 2 m frame, put into every prompt by `prompts.kit_prompt`), the albedo rules, the height
normalisation, and a list of layers (subject, scale hint, type, `normal_from`, roughness, per-layer albedo mean band).

    texforge kit build kits/forest/kit.json [--only dirt,moss] [--sources-only] [--no-gen]   # codex source + make + calibrate
    texforge kit check kits/forest/kit.json      # per-layer albedo range, height mean/spread, seams, which normal/height was used
    texforge kit blend kits/forest/kit.json      # height-blend vs linear-blend preview of the 6 ground layers (out/forest/blend/)
    texforge kit compare kits/forest/kit.json    # height-derived vs codex normal, lit, for every layer that uses codex normals
    texforge kit macro kits/forest/kit.json      # procedural macro-variation texture (no codex)
    texforge kit atlas kits/forest/kit.json      # litter atlas from one codex sprite sheet (--image to reuse one)

* Codex calls run one at a time (parallel runs time out); a rerun reuses every codex image already on disk, so
  rebuilding/recalibrating costs no codex run. `kit build` retries on out-of-memory.
* **Albedo calibration** (`calibrate.py`): in linear light, a gain puts the layer mean inside its band, a soft knee lifts blacks to
  0.03 and a soft shoulder holds highlights under 0.80. The check fails a layer on mean outside band, p0.5 < 0.03 (crushed) or
  p99.5 > 0.80 (blown).
* **Height normalisation**: every layer's Height PNG is re-centred to mean 0.5, std 0.18 (after clipping) so layers compare in a height
  blend. Normal and AO are derived before this, from the un-normalised height.
* **Height blend** (`blend.py`, identical formula in `integrations/unreal/create_landscape_material.py`): `a_i = h_i + w_i; m = max(a) - depth;
  w'_i = max(a_i - m, 0)/sum`, depth = 0.18 (`blend.sharpness`). Larger depth tends to the linear blend.
* **Per-layer normal choice**: `normal_from: codex` for chunky materials (gravel, mulch, pine needles, bark), `height` for fine/soft
  ones (dirt, moss, grassy dirt). The codex height gate was relaxed for organic layers (no joint mask): edge-correlation ratio >= 0.40
  instead of 0.60, because the generator smooths grain (right layouts score 0.50-0.73, a layout from another layer ~0.0 with offsets of 70-200 px).
  The normal sign test now takes, per channel, the strongest correlation over smoothing scales 1.5-12 px (needle/mulch y only shows at coarse scale).
* **Atlas** (`atlas.py`): codex draws leaves / twigs / pinecones / bark chips on flat magenta; Lab-distance key, magenta despill,
  1 px erosion, colour taken from the opaque interior (no fringe); items found as connected components, named by row, packed
  with a gutter and colour bleed; Normal from a rounded-sheet height + luminance relief; ORM. `litter.texforge.json` has every item's
  uv rectangle and size in cm (nominal: the sheet is assumed to be `sheet_cm` = 90 cm wide, codex does not control absolute scale).
* **Macro texture**: periodic spectral noise, R = damp patches, G = leaf drift (stretched along a wind direction), B = brightness; tiles every 30 m.

* **Unreal side (written, NOT run):** `integrations/unreal/create_landscape_material.py` imports the kit (6 ground layers, bark, macro, atlas) and builds
  `M_Forest_Landscape` (6-layer weight blend with the height blend above, anti-tile / macro / Nanite-displacement / RVT switches), `M_Forest_Bark`,
  `M_Forest_Litter` and one layer-info asset per layer. It has 25 `# VERIFY` marks; the HLSL has never been compiled.

## Known limits

* **The repeat is small.** A brick tile is 3 bricks wide; the grid snap keeps only whole bricks per course. Use
  the macro / two-scale options or decals to hide it.
* Grid snap needs axis-aligned (or <3 deg skewed), regular, straight joints. Wavy or perspective-distorted
  input fails the regularity checks and falls back to min-cut (noted in the JSON `notes`).
* Cobble / stone: min-cut, not a grid; stones near the cut can be slightly deformed or a gap can look smeared
  along the cut. Large landmarks (a big rock, a stain) get cut in half or repeated.
* Delight divides by a smooth luminance estimate: genuine large-scale colour variation is flattened too, and
  micro-shadows inside joints stay baked into the albedo.
* Normals are scaled for the non-square-tile aspect, but AO and the cavity blur are not (they run on the squashed square).
* Height is inferred from luminance and joints, not measured: dark spots become pits, so relief on brick faces is
  plausible but not physical. AO is a cavity approximation, not ray traced.
* The seam metric only sees steps at the wrap border: it cannot see a half-brick phase error or a repeated
  landmark; the sheets are the real test. It also cannot see the internal cut of a min-cut texture.
* Sources are ~1.25k px, outputs 2k: detail is upscaled (cubic).
* Trim: strip bounds are detected, not given by the model; brick bands inside trims are min-cut, not
  grid-snapped, so a seam may shear a joint (check `check/..._after_seam_crop.png`).
* The Unreal scripts have never run.

## Organic / ground extras (`make`, default ON for `--type organic`; `texforge/organic.py`)

Three steps, resolved per type in `organic.defaults_for` (organic: all three; cobble / plaster: despeckle only; brick / tile: none, old behaviour).
CLI: `--no-despeckle`, `--no-shadow-lift`, `--no-materials` (and `--despeckle` / `--shadow-lift` / `--materials` to force on), `--material-profile {auto,moss,gravel,generic}`, `--shadow-strength 0.85`.
Kit layers take the same keys (`despeckle`, `shadow_lift`, `materials`, `material_profile`, `shadow_strength`), plus `floor_lin` (per-layer albedo floor) and `albedo_balance_lin` (linear RGB white balance before calibration).
Every number below is written to `params.organic` in `<name>.texforge.json`.

1. **Height despeckle**: a pixel is an outlier when |h - median5| > 4 x local robust sigma; only clusters <= 12 px (isolated spikes / pits) are replaced by the median, repeated until stable. Real features (clusters > 12 px) are untouched.
2. **Material masks**: k-means (Lab + height cue) with soft assignment; moss = living / dead brown (Lab hue < 90) / soil gaps / twigs (thin elongated brown components), gravel = stones / fines, generic = raised / mid / low. Per class: roughness offset, cushion-scale height offset (moss also gets doming inside the cushions), AO deepening in gaps. Written as `<name>_Mask.png` (RGBA, linear, soft memberships summing to 1; channels in the JSON `mask_channels`).
3. **Shadow lift**: albedo / occ^(b x strength) with occ = cavity AO of the height, b fitted by regression of band-passed log-luminance on log-occ, gain capped (2.0), gaps only half lifted, mean luminance kept. Hue is not touched (it will not remove an orange cast).

`texforge compare OLD NEW --sheet out.png [--old-tint r,g,b]` renders old vs new under a low warm grazing sun (3x3, 1:1 crop, mask, roughness).
`texforge gen --prompt-file F --parallel 1` reads a long prompt from a file and runs codex one at a time.
`kits/forest/kit_v2.json` (own report: `kit_v2_report.json`) builds `moss_v2` and `pine_needles_v2` next to the originals.
Tests: `python -m unittest discover -s tests`.

## License

MIT, see `LICENSE`.
