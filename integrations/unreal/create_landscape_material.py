"""
texforge forest kit -> Unreal Engine 5.8 : texture import + LANDSCAPE material builder  (editor Python, UNTESTED)
================================================================================================================

Creates / rebuilds, in ONE run inside the editor:
    /Game/Forest/Textures/            T_<set>_<BaseColor|Normal|ORM|Height>  for the 6 ground layers, bark, the macro
                                      texture (T_macro_Macro) and the litter atlas (T_litter_*), with correct
                                      compression / sRGB / address mode. Generated 4x4 fallbacks in Textures/Defaults.
    /Game/Forest/Materials/M_Forest_Landscape   layer-blended + HEIGHT-BLENDED landscape material (the main product)
    /Game/Forest/Materials/M_Forest_Bark        tiny lit material (mesh UV * Tiling) for trunks
    /Game/Forest/Materials/M_Forest_Litter      tiny masked material (atlas alpha = opacity mask) for foliage / cards
    /Game/Forest/Layers/LI_<layer>              one LandscapeLayerInfoObject per ground layer (weight-blended)

HOW TO RUN (inside the editor, never from a shell; the editor has no argv)
    Output Log -> switch the console box to "Python" and run:
        py "/path/to/texforge/integrations/unreal/create_landscape_material.py"
    or  Tools > Execute Python Script...  and pick this file.
    Options: edit CONFIG below, or set environment variables TEXFORGE_<KEY> before starting the editor.
    Idempotent: textures are re-imported in place (replace_existing), materials are cleared in place
    (delete_all_material_expressions) and rebuilt, so assets / landscapes that reference them keep working.
    Fail soft: a missing layer folder or texture is a WARNING; that texture is replaced by a neutral generated default
    so the material still compiles and the script can be tried before all sets exist.

HEIGHT BLEND  (identical to texforge/blend.py, a_i = h_i + w_i, NO zero-weight special casing)
        a_i  = h_i + w_i          h_i = layer Height texture (kit-normalised: mean 0.5, std 0.18)
                                  w_i = landscape layer weight 0..1 (LandscapeLayerSample, ParameterName = layer name)
        m    = max_i(a_i) - Depth                      Depth = scalar param HeightBlendDepth (default kit blend.sharpness 0.18)
        w'_i = max(a_i - m, 0);  w'_i /= sum(w'_i)     (if the sum is ~0, falls back to the plain linear weights)
    Implemented in Custom HLSL nodes. A Custom node has ONE output, so the new weights are packed 3 per node
    (float3 each: 6 layers -> 2 nodes, both evaluate the same cheap ALU expression). A second pair of nodes yields the plain
    normalised linear weights; static switch UseHeightBlend picks between the pairs (off = linear, for comparison).
    One "combine" Custom node per output (BaseColor, Normal, ORM, Height) then sums layer_i * w'_i.
    Normals are blended in slope space (n.xy / n.z weighted by w', then normalize(float3(slope, 1))).

TILING
    UV_i = AbsoluteWorldPosition.xy / WorldSizeCm_<layer> * UVScale_<layer>   -> 1 tile per world_cm centimetres
    (= 100 / world_cm tiles per metre = 1 / world_cm per cm). WorldSizeCm_<layer> defaults to the sidecar/kit world_cm,
    UVScale_<layer> (default 1) is the per-layer tweak.

STATIC SWITCHES (per material instance; each combination is a shader permutation)
    UseHeightBlend  default ON   see above
    UseAntiTile     default OFF  2nd sample of every layer texture at a rotated (+17 deg per layer) 0.37x-scaled UV, blended
                                 by a low-frequency noise mask (same idea as UseTwoScaleBlend / option A in
                                 create_master_material.py; normal xy of the 2nd sample is counter-rotated).
    UseMacro        default ON   macro texture variation (R damp, G leaf drift, B brightness), see MACRO below
    UseDisplacement default OFF  blended height -> material Displacement output (Nanite tessellation), see below
    UseRVT          default OFF  RuntimeVirtualTextureOutput node gets the real BaseColor/Roughness/Normal

MACRO (UseMacro): MacroUV = WorldPosition.xy / MacroWorldSizeCm (3000). R = damp: albedo darkened by MacroDampDarken and
    roughness raised by MacroDampRoughAdd; G = leaf drift: albedo multiplied towards MacroLeafTint by MacroLeafAmount
    (textures are NOT re-blended, it is a tint); B = brightness: albedo * (1 + (B-0.5)*MacroBrightnessAmount).

COST (rough; per pixel, textures are always ALL sampled - there is no weight-based branching)
    layer textures : 6 layers x 4 maps (BaseColor, Normal, ORM, Height) = 24 samples      (+ 2-3 landscape weightmap fetches)
    + UseAntiTile  : 48 samples (x2) + 1 noise (~40 ALU) + 6 UV rotations + 6 normal rotations
    + UseMacro     : +1 sample, ~20 ALU
    height blend / combine : roughly 150 ALU. Total, defaults (blend + macro, no anti-tile): 25 samples;  with anti-tile: 49.
    Samplers: every sample node uses Sampler Source "Shared: Wrap" (SSM_WRAP_WORLD_GROUP_SETTINGS), so the whole material
    uses 1 shared wrap sampler instead of 25 (the hardware limit is 16 sampler states per material).
    This is a HIGH-END (PC / console) landscape material; for lower targets drop layers or merge Height into ORM.A.

WHAT IS NOT DONE
    * No landscape actor is created or modified, no layers painted, no RVT asset/volume created, no Nanite flags toggled.
    * No per-layer branching / weight culling (skip a layer whose weight is 0), no distance fade to a cheaper shader.
    * No triplanar projection for steep slopes (world XY planar projection only: cliffs will stretch).
    * Litter/Bark materials are intentionally minimal (no wind, no fade, no per-instance variation, no uv-rect picking).
    * Material instances are not created (parameters are all on the material itself).
    * The generated Custom HLSL is only compiled by the shader compiler: open the material and read the Output Log.

VERIFY LIST (things that could not be checked without the editor; search for "# VERIFY" in the code)
  1. unreal.MaterialExpressionLandscapeLayerSample exists with properties parameter_name / preview_weight; fallback is
     MaterialExpressionLandscapeLayerWeight (Base = 0, Layer = 1 -> output = weight), pins "Base"/"Layer".
  2. Custom node: unreal.CustomInput constructible, set_editor_property('inputs') rebuilds pins, output_type enum
     CustomMaterialOutputType.CMOT_FLOAT1..4; a WorldPosition (LWC in 5.x) input arrives as plain float3 in the HLSL.
  3. TextureSampleParameter2D properties texture / sampler_type / sampler_source and enum SamplerSourceMode.
     SSM_WRAP_WORLD_GROUP_SETTINGS; UV pin name "UVs" (older: "Coordinates"); duplicate parameter names (the anti-tile
     second sample re-uses the same parameter_name + texture) are accepted without a duplicate-parameter error.
  4. A TextureSample with SAMPLERTYPE_NORMAL outputs the UNPACKED normal (-1..1, z reconstructed) on RGB.
  5. MaterialProperty.MP_DISPLACEMENT exists in 5.8 and accepts a connection; Material property `displacement_scaling`
     (struct DisplacementScaling: magnitude, center) is writable from python; units of magnitude are centimetres.
  6. unreal.MaterialExpressionRuntimeVirtualTextureOutput and its pin names (BaseColor, Specular, Roughness, Normal,
     WorldHeight, Opacity, Mask).
  7. Creating LandscapeLayerInfoObject assets from python: class unreal.LandscapeLayerInfoObject, a factory (script tries
     LandscapeLayerInfoObjectFactory), properties layer_name (FName) and no_weight_blend (bool).
  8. Static switch pin names ("True"/"False"/"Value") - the link() helper introspects and prints the real ones on mismatch.
  9. Landscape layer weights from LandscapeLayerSample: with weight-blended layer infos they sum to 1 per pixel.
 10. Texture properties: virtual_texture_streaming (must be False for shared-sampler landscape textures), flip_green_channel,
     address_x/address_y, compression_settings / srgb. Macro texture is imported as TC_MASKS (linear, sampler type MASKS).
 11. Material properties: used_with_instanced_static_meshes (Litter), two_sided, tangent_space_normal, shading_model.
 12. Absolute world position: MaterialExpressionWorldPosition default type (WPT_DEFAULT) is absolute world position.
"""

import os
import json
import struct
import zlib

import unreal

# --------------------------------------------------------------------------------------------------
# CONFIG  (override with environment variables TEXFORGE_<KEY>)
# --------------------------------------------------------------------------------------------------
CONFIG = {
    "SOURCE_DIR": "",                  # REQUIRED: absolute path of the kit output folder (<out>/forest/); or set TEXFORGE_SOURCE_DIR
    "KIT_JSON": "",                    # absolute path of kit.json (optional: built-in layer list is used if empty/missing); or TEXFORGE_KIT_JSON
    "TEXTURES_DIR": "/Game/Forest/Textures",
    "MATERIALS_DIR": "/Game/Forest/Materials",
    "LAYERS_DIR": "/Game/Forest/Layers",
    "BUILD": "textures,bark,litter,layer_infos,landscape",   # comma list of stages
    "SAVE": "1",
    "RVT_NODE": "1",                  # 1 = always create the RuntimeVirtualTextureOutput node (inputs gated by UseRVT); 0 = never
    "DEFAULT_DEPTH": "0.18",          # HeightBlendDepth when kit.json has no blend.sharpness
    "DISPLACEMENT_MAGNITUDE_CM": "3.0",
    "DISPLACEMENT_CENTER": "0.5",
}

# fallback when kit.json is missing: (name, world_cm)
BUILTIN_LAYERS = [("dirt", 200.0), ("mulch", 200.0), ("moss", 200.0), ("grassy_dirt", 200.0), ("gravel", 200.0),
                  ("pine_needles", 200.0)]
MACRO_WORLD_CM = 3000.0


def _cfg(key):
    return os.environ.get("TEXFORGE_" + key, CONFIG[key])


# suffix -> (compression, srgb, sampler type name, default texture name)
TEX_KINDS = {
    "BaseColor": ("TC_DEFAULT", True, "SAMPLERTYPE_COLOR"),
    "Normal": ("TC_NORMALMAP", False, "SAMPLERTYPE_NORMAL"),
    "ORM": ("TC_MASKS", False, "SAMPLERTYPE_MASKS"),
    "Height": ("TC_GRAYSCALE", False, "SAMPLERTYPE_LINEAR_GRAYSCALE"),
    "Macro": ("TC_MASKS", False, "SAMPLERTYPE_MASKS"),
}
DEFAULT_TEXTURES = {   # suffix -> rgb of the generated fallback
    "BaseColor": (128, 128, 128),
    "Normal": (128, 128, 255),
    "ORM": (255, 128, 0),          # AO 1, rough 0.5, metal 0
    "Height": (128, 128, 128),
    "Macro": (0, 128, 128),        # damp 0, leaf 0.5, brightness 0.5 (neutral)
}

# --------------------------------------------------------------------------------------------------
# defensive layer (same helpers as create_master_material.py)
# --------------------------------------------------------------------------------------------------
FAILS = []
WARNS = []
CREATED = []
PARAMS = {"scalar": [], "vector": [], "texture": [], "static": []}   # (material, name, default)
CURRENT_MATERIAL = ["?"]
CURRENT_MAT = [None]

MEL = getattr(unreal, "MaterialEditingLibrary", None)
EAL = getattr(unreal, "EditorAssetLibrary", None)


def _info(msg):
    unreal.log("[forest] " + str(msg))


def _warn(msg):
    WARNS.append(str(msg))
    unreal.log_warning("[forest] WARN " + str(msg))


def _fail(msg):
    FAILS.append("[%s] %s" % (CURRENT_MATERIAL[0], msg))
    unreal.log_error("[forest] FAIL " + str(msg))


def _desc(expr):
    try:
        return "%s(%s)" % (type(expr).__name__, expr.get_editor_property("desc") or expr.get_name())
    except Exception:
        return type(expr).__name__


def _set(obj, prop, value):
    """set_editor_property that logs instead of raising. Returns True on success."""
    try:
        obj.set_editor_property(prop, value)
        return True
    except Exception as exc:  # noqa
        _fail("set_editor_property(%s.%s) failed: %s" % (type(obj).__name__, prop, exc))
        return False


def _soft(obj, prop, value):
    """Like _set but a failure is only a warning."""
    try:
        obj.set_editor_property(prop, value)
        return True
    except Exception as exc:  # noqa
        _warn("could not set %s.%s (%s); check it in the details panel" % (type(obj).__name__, prop, exc))
        return False


def _enum(cls_name, member, soft=False):
    """Return unreal.<cls_name>.<member> trying the usual python spellings; None (+FAIL/WARN) if absent."""
    cls = getattr(unreal, cls_name, None)
    if cls is not None:
        for cand in (member, member.upper(), member.replace("_", "").upper()):
            if hasattr(cls, cand):
                return getattr(cls, cand)
    (_warn if soft else _fail)("enum unreal.%s.%s not found" % (cls_name, member))
    return None


def _names(fn, expr):
    try:
        return [str(n) for n in fn(expr)]
    except Exception:
        return None


def _in_names(expr):
    return _names(MEL.get_material_expression_input_names, expr)


def _out_names(expr):
    return _names(MEL.get_material_expression_output_names, expr)


def _sig(x):
    return x if isinstance(x, tuple) else (x, "")


def link(src, dst, pins, what=""):
    """Connect src (expr or (expr, output_name)) to input `pin` of dst. `pins` may be a list of candidate names
    (first one that exists on the node wins). Checks pin names via introspection AND the bool result."""
    se, so = _sig(src)
    cands = [pins] if isinstance(pins, str) else list(pins)
    avail = _in_names(dst)
    use = None
    if avail is None:
        use = cands[0]
    else:
        low = dict((a.lower(), a) for a in avail)
        for c in cands:
            if c.lower() in low:
                use = low[c.lower()]
                break
    if use is None:
        _fail("%s: no input %s on %s; available inputs: %s" % (what or "link", cands, _desc(dst), avail))
        return False
    if so:
        outs = _out_names(se)
        if outs is not None and so.lower() not in [o.lower() for o in outs]:
            _fail("%s: output '%s' not on %s; available outputs: %s" % (what or "link", so, _desc(se), outs))
            return False
    try:
        ok = MEL.connect_material_expressions(se, so, dst, use)
    except Exception as exc:
        _fail("%s: connect_material_expressions raised %s (%s -> %s.%s)" % (what or "link", exc, _desc(se), _desc(dst), use))
        return False
    if not ok:
        _fail("%s: connect_material_expressions returned False (%s[%s] -> %s.%s)" % (what or "link", _desc(se), so, _desc(dst), use))
    return bool(ok)


def connect_prop(src, prop_name, what="", soft=False):
    """Connect to a material output pin (MP_BASE_COLOR ...) and read it back to prove it stuck."""
    se, so = _sig(src)
    prop = _enum("MaterialProperty", prop_name, soft=soft)
    if prop is None:
        return False
    report = _warn if soft else _fail
    try:
        ok = MEL.connect_material_property(se, so, prop)
    except Exception as exc:
        report("connect_material_property(%s) raised %s" % (prop_name, exc))
        return False
    if not ok:
        report("connect_material_property(%s) returned False (%s)" % (prop_name, _desc(se)))
        return False
    try:
        if MEL.get_material_property_input_node(CURRENT_MAT[0], prop) is None:
            report("material pin %s is still empty after connect (%s)" % (prop_name, _desc(se)))
            return False
    except Exception as exc:
        _warn("could not read back pin %s: %s" % (prop_name, exc))
    return True


# --------------------------------------------------------------------------------------------------
# HLSL snippets (Custom node bodies). Custom bodies cannot define functions; everything is inline.
# --------------------------------------------------------------------------------------------------
def _tpl(code, **kw):
    for k, v in kw.items():
        code = code.replace("@%s@" % k, v)
    return code


def hash22(out, expr):
    """Hoskins hash22: float2 `out` from float2 expression `expr`."""
    return _tpl("""float3 @O@_p = frac(float3((@E@).xyx) * float3(0.1031, 0.1030, 0.0973));
@O@_p += dot(@O@_p, @O@_p.yzx + 33.33);
float2 @O@ = frac((@O@_p.xx + @O@_p.yz) * @O@_p.zy);
""", O=out, E=expr)


def noise2(out, expr, octaves=2):
    """2-octave value noise in [0,1] -> float `out`, from float2 expression `expr`."""
    lines = []
    acc = []
    amp = 0.65
    for o in range(octaves):
        p = "%s_o%d" % (out, o)
        scale = "%.4f" % (2.13 ** o)
        lines.append(_tpl("""float2 @P@_x = (@E@) * @S@ + float2(@OFF@, @OFF2@);
float2 @P@_i = floor(@P@_x);
float2 @P@_f = frac(@P@_x);
float2 @P@_u = @P@_f * @P@_f * (3.0 - 2.0 * @P@_f);
""", P=p, E=expr, S=scale, OFF="%.2f" % (17.3 * o), OFF2="%.2f" % (5.7 * o)))
        for ci, (dx, dy) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
            lines.append(hash22("%s_h%d" % (p, ci), "%s_i + float2(%d.0, %d.0)" % (p, dx, dy)))
        lines.append(_tpl("""float @P@ = lerp(lerp(@P@_h0.x, @P@_h1.x, @P@_u.x), lerp(@P@_h2.x, @P@_h3.x, @P@_u.x), @P@_u.y);
""", P=p))
        acc.append("%s * %.3f" % (p, amp))
        amp = 1.0 - amp
    lines.append("float %s = %s;\n" % (out, " + ".join(acc)))
    return "".join(lines)


HLSL_LAYER_UV = "return WP.xy / max(WorldCm, 1.0) * UVScale;\n"   # VERIFY: WP arrives as float3 (LWC auto-converted)
HLSL_MACRO_UV = "return WP.xy / max(MacroWorldCm, 1.0);\n"
HLSL_AT_MASK = noise2("nm", "WP.xy / max(MaskScaleCm, 1.0)") + "return smoothstep(0.30, 0.70, nm) * saturate(Strength);\n"


def hlsl_at_uv(layer_index):
    """Second, rotated + rescaled UV for anti-tiling. Rotation differs per layer (+17 deg each)."""
    return _tpl("""float ang = (RotDeg + @OFF@) * 0.017453292;
float ca = cos(ang);
float sa = sin(ang);
return float2(ca * UV.x - sa * UV.y, sa * UV.x + ca * UV.y) * Ratio + float2(0.37, 0.71);
""", OFF="%.1f" % (17.0 * layer_index))


def hlsl_at_normal(layer_index):
    """Blend n1 (UV) and n2 (rotated UV) tangent normals (both already unpacked -1..1): counter-rotate n2.xy, lerp, rebuild z."""
    return _tpl("""float ang = (RotDeg + @OFF@) * 0.017453292;
float ca = cos(ang);
float sa = sin(ang);
float2 n2r = float2(ca * N2.x + sa * N2.y, -sa * N2.x + ca * N2.y);
float2 n = lerp(N1.xy, n2r, M);
return float3(n, sqrt(saturate(1.0 - dot(n, n))));
""", OFF="%.1f" % (17.0 * layer_index))


def _wsel(i):
    return "WG%d.%s" % (i // 3, "xyz"[i % 3])


def hlsl_weights(n, height_blend, group):
    """New layer weights packed 3 per node: returns float3 of layers [3*group, 3*group+2] (0 when beyond n).
    Inputs: H0..H(n-1), W0..W(n-1), Depth (the linear variant only needs W0..W(n-1))."""
    lines = []
    if height_blend:
        for i in range(n):
            lines.append("float a%d = H%d + W%d;" % (i, i, i))
        mx = "a0"
        for i in range(1, n):
            mx = "max(%s, a%d)" % (mx, i)
        lines.append("float m = %s - Depth;" % mx)
        for i in range(n):
            lines.append("float v%d = max(a%d - m, 0.0);" % (i, i))
    else:
        for i in range(n):
            lines.append("float v%d = W%d;" % (i, i))
    lines.append("float s = %s;" % " + ".join("v%d" % i for i in range(n)))
    if height_blend:   # degenerate (Depth <= 0): fall back to the plain landscape weights
        lines.append("if (s <= 0.00001) { %s s = %s; }" % (
            " ".join("v%d = W%d;" % (i, i) for i in range(n)), " + ".join("W%d" % i for i in range(n))))
    else:
        pass
    lines.append("if (s <= 0.00001) { v0 = 1.0; s = 1.0; }")   # nothing painted at all -> first layer
    lines.append("float inv = 1.0 / s;")
    comps = []
    for k in range(3):
        i = 3 * group + k
        comps.append("v%d * inv" % i if i < n else "0.0")
    lines.append("return float3(%s);" % ", ".join(comps))
    return "\n".join(lines) + "\n"


def hlsl_combine(kind, n):
    """kind: color3 (float3 layers) | scalar (float layers) | normal. Inputs C0..C(n-1), WG0..WG(ceil(n/3)-1)."""
    if kind in ("color3", "scalar"):
        return "return " + " + ".join("C%d * %s" % (i, _wsel(i)) for i in range(n)) + ";\n"
    # normal: slope-space blend, z reconstructed by normalising float3(slope, 1)
    lines = ["float2 sl = float2(0.0, 0.0);"]
    for i in range(n):
        lines.append("sl += (C%d.xy / max(C%d.z, 0.05)) * %s;" % (i, i, _wsel(i)))
    lines.append("return normalize(float3(sl, 1.0));")
    return "\n".join(lines) + "\n"


HLSL_MACRO_COLOR = """float damp = saturate(M.r);
float leaf = saturate(M.g);
float3 c = C * lerp(1.0, 1.0 - saturate(DampDarken), damp);
c = lerp(c, c * LeafTint, leaf * saturate(LeafAmount));
c *= max(1.0 + (M.b - 0.5) * BrightAmount, 0.0);
return c;
"""
HLSL_MACRO_ROUGH = "return saturate(R + DampRoughAdd * saturate(M.r));\n"


# --------------------------------------------------------------------------------------------------
# kit descriptor / sidecars
# --------------------------------------------------------------------------------------------------
def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        _warn("could not read %s (%s)" % (path, exc))
        return None


def load_kit():
    """Return (ground_layers [(name, world_cm)], bark_world_cm, depth)."""
    kit = _read_json(_cfg("KIT_JSON")) if (_cfg("KIT_JSON") and os.path.exists(_cfg("KIT_JSON"))) else None
    layers = []
    bark_cm = 100.0
    depth = float(_cfg("DEFAULT_DEPTH"))
    if kit:
        kit_cm = float(kit.get("world_cm", 200.0))
        for ly in kit.get("layers", []):
            cm = float(ly.get("world_cm", kit_cm))
            if ly.get("role", "ground") == "ground":
                layers.append((ly["name"], cm))
            elif ly["name"] == "bark":
                bark_cm = cm
        depth = float((kit.get("blend") or {}).get("sharpness", depth))
    else:
        _warn("kit.json not found: using the built-in 6 layer list")
    if not layers:
        layers = list(BUILTIN_LAYERS)
    return layers, bark_cm, depth


def sidecar_world_cm(layer, fallback):
    p = os.path.join(_cfg("SOURCE_DIR"), layer, layer + ".texforge.json")
    if os.path.exists(p):
        meta = _read_json(p) or {}
        wc = meta.get("world_cm")
        if isinstance(wc, (list, tuple)) and wc:
            return float(wc[0])
        if isinstance(wc, (int, float)):
            return float(wc)
    return fallback


# --------------------------------------------------------------------------------------------------
# texture import
# --------------------------------------------------------------------------------------------------
def _write_png(path, rgb, size=4):
    row = b"\x00" + bytes(rgb) * size

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)) +
           chunk(b"IDAT", zlib.compress(row * size)) + chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(png)


def _project_saved_dir():
    for fn in (lambda: unreal.Paths.project_saved_dir(), lambda: unreal.SystemLibrary.get_project_saved_directory()):  # VERIFY: one exists
        try:
            return fn()
        except Exception:
            continue
    import tempfile
    return tempfile.gettempdir()


def _apply_tex_settings(tex, suffix, address, vt_off=True):
    comp, srgb, _ = TEX_KINDS[suffix]
    c = _enum("TextureCompressionSettings", comp)
    if c is not None:
        _set(tex, "compression_settings", c)
    _set(tex, "srgb", srgb)
    if suffix == "Normal":
        _set(tex, "flip_green_channel", False)           # texforge writes DirectX (green-down): no flip
    a = _enum("TextureAddress", "TA_CLAMP" if address == "clamp" else "TA_WRAP")
    if a is not None:
        _set(tex, "address_x", a)
        _set(tex, "address_y", a)
    if vt_off:
        _soft(tex, "virtual_texture_streaming", False)   # VERIFY: VT textures cannot use the shared wrap sampler path


def import_pngs(items, dest, address):
    """items: list of (png_path, asset_name, suffix). Imports + applies settings. Returns {asset_name: texture}."""
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    EAL.make_directory(dest)
    tasks = []
    for png, name, _ in items:
        task = unreal.AssetImportTask()
        for k, v in (("filename", png), ("destination_path", dest), ("destination_name", name),
                     ("automated", True), ("save", True), ("replace_existing", True)):
            _set(task, k, v)
        tasks.append(task)
    if tasks:
        tools.import_asset_tasks(tasks)
    out = {}
    for png, name, suffix in items:
        path = "%s/%s" % (dest, name)
        tex = EAL.load_asset(path)
        if tex is None:
            _fail("import failed or asset not loadable: %s (source %s)" % (path, png))
            continue
        _apply_tex_settings(tex, suffix, address)
        try:
            EAL.save_loaded_asset(tex)
        except Exception as exc:
            _warn("could not save %s: %s" % (path, exc))
        out[name] = tex
        comp, srgb, _s = TEX_KINDS[suffix]
        CREATED.append("%s  (%s, sRGB=%s, %s)" % (path, comp, srgb, address))
    return out


def ensure_default_textures():
    """Generated 4x4 neutral fallbacks {suffix: texture}, used for any texture that is missing on disk."""
    ddir = _cfg("TEXTURES_DIR") + "/Defaults"
    EAL.make_directory(ddir)
    tmp = os.path.join(_project_saved_dir(), "Texforge", "ForestDefaults")
    os.makedirs(tmp, exist_ok=True)
    items = []
    for suffix, rgb in DEFAULT_TEXTURES.items():
        name = "T_Forest_Default_" + suffix
        if not EAL.does_asset_exist("%s/%s" % (ddir, name)):
            png = os.path.join(tmp, name + ".png")
            _write_png(png, rgb)
            items.append((png, name, suffix))
    if items:
        import_pngs(items, ddir, "wrap")
    res = {}
    for suffix in DEFAULT_TEXTURES:
        tex = EAL.load_asset("%s/T_Forest_Default_%s" % (ddir, suffix))
        if tex is None:
            _fail("default texture T_Forest_Default_%s missing" % suffix)
        else:
            _apply_tex_settings(tex, suffix, "wrap")
            res[suffix] = tex
    return res


def import_set(prefix, folder, suffixes, address, defaults, label):
    """Import <folder>/<prefix>_<suffix>.png for each suffix; missing ones -> neutral default (warning).
    Returns {suffix: (texture, is_default)}."""
    items = []
    present = {}
    for suffix in suffixes:
        png = os.path.join(_cfg("SOURCE_DIR"), folder, "%s_%s.png" % (prefix, suffix)).replace("\\", "/")
        if os.path.exists(png):
            items.append((png, "T_%s_%s" % (prefix, suffix), suffix))
        else:
            _warn("%s: %s not found -> neutral default used for %s" % (label, png, suffix))
    imported = import_pngs(items, _cfg("TEXTURES_DIR"), address) if items else {}
    for png, name, suffix in items:
        if name in imported:
            present[suffix] = imported[name]
    result = {}
    for suffix in suffixes:
        if suffix in present:
            result[suffix] = (present[suffix], False)
        else:
            result[suffix] = (defaults.get(suffix), True)
    return result


# --------------------------------------------------------------------------------------------------
# graph builder (compact version of create_master_material.Graph)
# --------------------------------------------------------------------------------------------------
class Graph(object):
    COLW = 420
    BASE_X = -5200

    def __init__(self, mat):
        self.mat = mat
        self.nodes = []
        self.col_of = {}
        self.col_y = {}
        self.band_y = 0
        self.section = "Misc"
        self.statics = {}

    def sec(self, name):
        maxy = max(self.col_y.values()) if self.col_y else None
        if maxy is not None:
            self.band_y = maxy + 140
        self.col_y = {}
        self.section = name

    def _col(self, *sigs):
        cols = [self.col_of.get(id(_sig(s)[0]), 0) for s in sigs if s is not None and not isinstance(s, (int, float))]
        return (max(cols) + 1) if cols else 0

    def add(self, cls_name, col, h=130, **props):
        cls = getattr(unreal, "MaterialExpression" + cls_name, None)
        if cls is None:
            raise RuntimeError("unreal.MaterialExpression%s does not exist in this engine build" % cls_name)
        x = self.BASE_X + col * self.COLW
        y = self.col_y.get(col, self.band_y)
        self.col_y[col] = y + h
        expr = MEL.create_material_expression(self.mat, cls, x, y)
        if expr is None:
            raise RuntimeError("create_material_expression(%s) returned None" % cls_name)
        desc = props.pop("desc", None)
        if desc:
            _set(expr, "desc", desc)
        for k, v in props.items():
            _set(expr, k, v)
        self.nodes.append((expr, x, y, h, self.section))
        self.col_of[id(expr)] = col
        return expr

    def finish_comments(self):
        by_sec = {}
        for expr, x, y, h, sec in self.nodes:
            b = by_sec.setdefault(sec, [x, y, x, y + h])
            b[0] = min(b[0], x)
            b[1] = min(b[1], y)
            b[2] = max(b[2], x)
            b[3] = max(b[3], y + h)
        for sec, (x0, y0, x1, y1) in by_sec.items():
            try:
                c = MEL.create_material_expression(self.mat, unreal.MaterialExpressionComment, x0 - 40, y0 - 60)
                _set(c, "text", sec)
                _set(c, "size_x", int(x1 - x0 + self.COLW))
                _set(c, "size_y", int(y1 - y0 + 90))
                _set(c, "font_size", 18)
                _set(c, "comment_color", unreal.LinearColor(0.15, 0.2, 0.3, 1.0))
            except Exception as exc:
                _warn("comment node for section '%s' failed: %s" % (sec, exc))

    def note(self, text, col=0):
        """Free-standing comment box with explanatory text (instructions that live in the graph)."""
        try:
            c = MEL.create_material_expression(self.mat, unreal.MaterialExpressionComment, self.BASE_X + col * self.COLW,
                                               self.band_y - 220)
            _set(c, "text", text)
            _set(c, "size_x", 900)
            _set(c, "size_y", 180)
            _set(c, "font_size", 12)
        except Exception as exc:
            _warn("note comment failed: %s" % exc)

    # parameters
    def scalar(self, name, default, group, smin=None, smax=None, desc=None):
        e = self.add("ScalarParameter", 0, h=110, parameter_name=name, default_value=float(default), group=group, desc=desc or name)
        if smin is not None:
            _set(e, "slider_min", float(smin))
        if smax is not None:
            _set(e, "slider_max", float(smax))
        PARAMS["scalar"].append((CURRENT_MATERIAL[0], name, default))
        return e

    def vector(self, name, rgba, group, desc=None):
        rgba = tuple(rgba) + (1.0,) * (4 - len(rgba))
        e = self.add("VectorParameter", 0, h=130, parameter_name=name, default_value=unreal.LinearColor(*rgba), group=group,
                     desc=desc or name)
        PARAMS["vector"].append((CURRENT_MATERIAL[0], name, rgba))
        return e

    def static(self, name, default, group, desc=None):
        e = self.add("StaticBoolParameter", 0, h=110, parameter_name=name, default_value=bool(default), group=group, desc=desc or name)
        self.statics[name] = e
        PARAMS["static"].append((CURRENT_MATERIAL[0], name, bool(default)))
        return e

    def c1(self, v):
        return self.add("Constant", 0, h=90, r=float(v))

    def c3(self, r, g, b):
        return self.add("Constant3Vector", 0, h=110, constant=unreal.LinearColor(r, g, b, 1.0))

    def texcoord(self, idx=0):
        return self.add("TextureCoordinate", 0, h=110, coordinate_index=idx)

    def world_pos(self):
        # VERIFY: default world_position_shader_offset / type is ABSOLUTE world position (WPT_DEFAULT) and LWC-compatible
        return self.add("WorldPosition", 0, h=100, desc="Absolute World Position")

    # math
    def _bin(self, cls, a, b, desc=None):
        n = self.add(cls, self._col(a, b), h=100, desc=desc)
        for key, v, pin in (("a", a, "A"), ("b", b, "B")):
            if isinstance(v, (int, float)):
                _set(n, "const_" + key, float(v))
            else:
                link(v, n, pin, "%s.%s" % (cls, pin))
        return n

    def mul(self, a, b, desc=None):
        return self._bin("Multiply", a, b, desc)

    def add_(self, a, b, desc=None):
        return self._bin("Add", a, b, desc)

    def sub(self, a, b, desc=None):
        return self._bin("Subtract", a, b, desc)

    def lerp(self, a, b, alpha, desc=None):
        n = self.add("LinearInterpolate", self._col(a, b, alpha), h=110, desc=desc)
        for key, v, pin in (("a", a, "A"), ("b", b, "B"), ("alpha", alpha, "Alpha")):
            if isinstance(v, (int, float)):
                _set(n, "const_" + key, float(v))
            else:
                link(v, n, pin, "Lerp.%s" % pin)
        return n

    def mask(self, a, chans):
        n = self.add("ComponentMask", self._col(a), h=90, r="r" in chans, g="g" in chans, b="b" in chans, a="a" in chans)
        link(a, n, "Input", "ComponentMask")
        return n

    def sw(self, static_name, when_true, when_false, desc=None):
        """StaticSwitch driven by the named StaticBoolParameter. The branch not selected is not compiled."""
        n = self.add("StaticSwitch", self._col(when_true, when_false, self.statics[static_name]), h=110,
                     desc=desc or ("switch " + static_name))
        # VERIFY: StaticSwitch (non-parameter) pins are named True/False/Value (fallbacks A/B); link() introspects.
        link(when_true, n, ["True", "A"], "StaticSwitch(%s).True" % static_name)
        link(when_false, n, ["False", "B"], "StaticSwitch(%s).False" % static_name)
        link(self.statics[static_name], n, ["Value"], "StaticSwitch(%s).Value" % static_name)
        return n

    def custom(self, label, code, inputs, out="f3"):
        """inputs: list of (name, signal). out: f1|f2|f3|f4."""
        col = self._col(*[s for _, s in inputs])
        n = self.add("Custom", col, h=70 + 24 * len(inputs), desc=label)
        ot = _enum("CustomMaterialOutputType", {"f1": "CMOT_FLOAT1", "f2": "CMOT_FLOAT2", "f3": "CMOT_FLOAT3", "f4": "CMOT_FLOAT4"}[out])
        cis = []
        for nm, _ in inputs:
            ci = unreal.CustomInput()   # VERIFY: constructible
            _set(ci, "input_name", nm)
            cis.append(ci)
        _set(n, "code", code)
        _set(n, "description", label)
        if ot is not None:
            _set(n, "output_type", ot)
        _set(n, "inputs", cis)
        avail = _in_names(n)
        if avail is not None and len(avail) != len(inputs):
            _fail("Custom '%s': expected %d input pins, node reports %s" % (label, len(inputs), avail))
        for nm, sig in inputs:
            if isinstance(sig, (int, float)):
                sig = self.c1(sig)
            link(sig, n, nm, "Custom '%s'.%s" % (label, nm))
        return n

    def tex(self, pname, tex_asset, suffix, group, uv, desc=None, sampler_source="wrap"):
        """TextureSampleParameter2D with Shared sampler. Duplicate calls with the same pname re-use the parameter (anti-tile 2nd sample)."""
        comp, srgb, sampler = TEX_KINDS[suffix]
        e = self.add("TextureSampleParameter2D", self._col(uv), h=150, parameter_name=pname, group=group, desc=desc or pname)
        if tex_asset is not None:
            _set(e, "texture", tex_asset)
        else:
            _fail("no texture (not even a default) for parameter %s" % pname)
        st = _enum("MaterialSamplerType", sampler)
        if st is not None:
            _set(e, "sampler_type", st)         # set AFTER texture (assigning a texture auto-picks the sampler type)
        if sampler_source == "wrap":
            ss = _enum("SamplerSourceMode", "SSM_WRAP_WORLD_GROUP_SETTINGS")   # VERIFY: python enum class name SamplerSourceMode
            if ss is not None:
                _set(e, "sampler_source", ss)
        # VERIFY: UV pin is "UVs" in 5.x ("Coordinates" in older builds)
        link(uv, e, ["UVs", "Coordinates"], "%s.UVs" % pname)
        return e


def _new_material(name, directory):
    path = "%s/%s" % (directory, name)
    EAL.make_directory(directory)
    if EAL.does_asset_exist(path):
        mat = EAL.load_asset(path)
        if mat is None or not isinstance(mat, unreal.Material):
            raise RuntimeError("%s exists but is not a Material" % path)
        MEL.delete_all_material_expressions(mat)
        CURRENT_MAT[0] = mat
        return mat
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    mat = tools.create_asset(name, directory, unreal.Material, unreal.MaterialFactoryNew())
    if mat is None:
        raise RuntimeError("create_asset failed for %s" % path)
    CURRENT_MAT[0] = mat
    return mat


def _finish_material(g, mat):
    g.finish_comments()
    try:
        MEL.recompile_material(mat)
    except Exception as exc:
        _fail("recompile_material raised %s" % exc)
    if _cfg("SAVE") == "1":
        try:
            EAL.save_loaded_asset(mat)
        except Exception as exc:
            _warn("save failed for %s: %s" % (mat.get_name(), exc))
    CREATED.append("%s/%s  (material)" % (_cfg("MATERIALS_DIR"), mat.get_name()))


def _setup_surface(mat, masked=False):
    _set(mat, "material_domain", _enum("MaterialDomain", "MD_SURFACE"))
    _set(mat, "blend_mode", _enum("BlendMode", "BLEND_MASKED" if masked else "BLEND_OPAQUE"))
    _soft(mat, "shading_model", _enum("MaterialShadingModel", "MSM_DEFAULT_LIT"))   # VERIFY: may be read-only (already default)
    _set(mat, "tangent_space_normal", True)


# --------------------------------------------------------------------------------------------------
# M_Forest_Bark / M_Forest_Litter  (deliberately tiny)
# --------------------------------------------------------------------------------------------------
def build_bark(tex):
    name = "M_Forest_Bark"
    CURRENT_MATERIAL[0] = name
    mat = _new_material(name, _cfg("MATERIALS_DIR"))
    _setup_surface(mat)
    g = Graph(mat)
    g.sec("Bark: mesh UV0 * Tiling (kit bark tile = 100 cm: author UVs so 0..1 = 1 m, or raise Tiling)")
    tiling = g.scalar("Tiling", 1.0, "00 Bark", 0.01, 64)
    uv = g.mul(g.texcoord(0), tiling, desc="UV0 * Tiling")
    bc = g.tex("BaseColor", tex["BaseColor"][0], "BaseColor", "00 Bark", uv, sampler_source="own")
    nm = g.tex("Normal", tex["Normal"][0], "Normal", "00 Bark", uv, sampler_source="own")
    orm = g.tex("ORM", tex["ORM"][0], "ORM", "00 Bark", uv, sampler_source="own")
    PARAMS["texture"].extend([(name, "BaseColor", ""), (name, "Normal", ""), (name, "ORM", "")])
    connect_prop(bc, "MP_BASE_COLOR")
    connect_prop(nm, "MP_NORMAL")
    connect_prop(g.mask(orm, "r"), "MP_AMBIENT_OCCLUSION")
    connect_prop(g.mask(orm, "g"), "MP_ROUGHNESS")
    connect_prop(g.mask(orm, "b"), "MP_METALLIC")
    _finish_material(g, mat)
    return mat


def build_litter(tex):
    name = "M_Forest_Litter"
    CURRENT_MATERIAL[0] = name
    mat = _new_material(name, _cfg("MATERIALS_DIR"))
    _setup_surface(mat, masked=True)
    _soft(mat, "two_sided", True)
    _soft(mat, "used_with_instanced_static_meshes", True)   # VERIFY: property name (foliage / HISM instancing flag)
    g = Graph(mat)
    g.sec("Litter atlas (clamp addressing, alpha = cutout). Mesh/card UVs pick the atlas cell (rects in litter.texforge.json)")
    uv = g.texcoord(0)
    bc = g.tex("BaseColor", tex["BaseColor"][0], "BaseColor", "00 Litter", uv, sampler_source="own")
    nm = g.tex("Normal", tex["Normal"][0], "Normal", "00 Litter", uv, sampler_source="own")
    orm = g.tex("ORM", tex["ORM"][0], "ORM", "00 Litter", uv, sampler_source="own")
    PARAMS["texture"].extend([(name, "BaseColor", ""), (name, "Normal", ""), (name, "ORM", "")])
    connect_prop(bc, "MP_BASE_COLOR")
    connect_prop((bc, "A"), "MP_OPACITY_MASK")          # VERIFY: output "A" on a TextureSample
    connect_prop(nm, "MP_NORMAL")
    connect_prop(g.mask(orm, "r"), "MP_AMBIENT_OCCLUSION")
    connect_prop(g.mask(orm, "g"), "MP_ROUGHNESS")
    connect_prop(g.mask(orm, "b"), "MP_METALLIC")
    _finish_material(g, mat)
    return mat


# --------------------------------------------------------------------------------------------------
# M_Forest_Landscape
# --------------------------------------------------------------------------------------------------
LANDSCAPE_DISPLACEMENT_NOTES = """NANITE DISPLACEMENT (UseDisplacement, default OFF) - UE 5.8, VERIFY every step in your build:
 1. Project Settings > Engine > Rendering > Nanite: enable Nanite Tessellation / displacement (r.Nanite.AllowTessellation=1).
 2. Landscape actor > Details > Nanite: Enable Nanite Landscape (rebuild the Nanite data), assign this material.
 3. Material > Details > Nanite/Tessellation: displacement is read from the Displacement pin: D = 0.5 + (Height - DisplacementCenter)
    * DisplacementMagnitudeCm with the material's Displacement Scaling = (magnitude 1 cm, center 0.5) set by the script.
    Kit height is mean 0.5 / std 0.18, so +-1 sigma = +-0.18 * magnitude (3 cm default: +-0.5 cm; raise to 8-15 for relief).
 4. Turn UseDisplacement ON in the material (or an instance). Needs a Nanite-enabled landscape; plain landscapes ignore it."""

RVT_NOTES = """RUNTIME VIRTUAL TEXTURE (UseRVT, default OFF) - VERIFY:
 1. Content Browser > Add > Materials > Runtime Virtual Texture; type BaseColor, Normal, Roughness (+ Specular) as needed.
 2. Level: add a RuntimeVirtualTextureVolume covering the landscape, assign the RVT asset, press 'Set Bounds' / 'Build'.
 3. Landscape actor > Virtual Texture: add the RVT to 'Draw in Virtual Textures'.
 4. Set UseRVT ON. The node below then receives the final BaseColor / Roughness / Normal (otherwise neutral constants)."""


def make_layer_weight(g, layer, preview):
    """Expression producing the landscape weight (float) of `layer`."""
    cls = getattr(unreal, "MaterialExpressionLandscapeLayerSample", None)   # VERIFY: exists in 5.8
    if cls is not None:
        e = g.add("LandscapeLayerSample", 0, h=110, parameter_name=layer, preview_weight=float(preview), desc="weight " + layer)
        return e
    _warn("LandscapeLayerSample missing: falling back to LandscapeLayerWeight(Base=0, Layer=1) for %s" % layer)
    e = g.add("LandscapeLayerWeight", 0, h=130, parameter_name=layer, preview_weight=float(preview), const_base=0.0, desc="weight " + layer)
    link(g.c3(1.0, 1.0, 1.0), e, "Layer", "LayerWeight.Layer")   # VERIFY: pin "Layer"; output = Base + Layer * weight
    return e


def build_landscape(layers, tex_by_layer, depth):
    name = "M_Forest_Landscape"
    CURRENT_MATERIAL[0] = name
    mat = _new_material(name, _cfg("MATERIALS_DIR"))
    _setup_surface(mat)
    n = len(layers)
    ngroups = (n + 2) // 3
    G_L, G_B, G_A, G_M, G_D = "00 Layers", "10 Height blend", "20 Anti-tile", "30 Macro", "40 Displacement / RVT"

    # displacement scaling on the material: magnitude 1 cm / center 0.5, the graph does the real scaling (so it is parametric)
    try:
        ds = unreal.DisplacementScaling()          # VERIFY: struct exists, fields magnitude/center
        ds.set_editor_property("magnitude", 1.0)
        ds.set_editor_property("center", 0.5)
        mat.set_editor_property("displacement_scaling", ds)
    except Exception as exc:
        _warn("could not set material displacement_scaling (%s); set Magnitude=1, Center=0.5 by hand if you use UseDisplacement" % exc)

    g = Graph(mat)

    g.sec("Static switches")
    g.static("UseHeightBlend", True, G_B, "off = plain linear (normalised) landscape weights")
    g.static("UseAntiTile", False, G_A, "2nd rotated/scaled sample of every layer texture, blended by low-freq noise (2x texture cost)")
    g.static("UseMacro", True, G_M, "macro texture variation")
    g.static("UseDisplacement", False, G_D, "blended height -> Nanite displacement output")
    g.static("UseRVT", False, G_D, "feed the RuntimeVirtualTextureOutput node")

    g.sec("Global parameters")
    p_depth = g.scalar("HeightBlendDepth", depth, G_B, 0.0, 1.0, "m = max(h+w) - Depth. kit blend.sharpness")
    p_ratio = g.scalar("AntiTileRatio", 0.37, G_A, 0.1, 1.0)
    p_rot = g.scalar("AntiTileRotDeg", 31.0, G_A, 0.0, 90.0)
    p_mscale = g.scalar("AntiTileMaskScaleCm", 900.0, G_A, 100.0, 5000.0)
    p_mstr = g.scalar("AntiTileStrength", 1.0, G_A, 0.0, 1.0)
    p_macro_cm = g.scalar("MacroWorldSizeCm", MACRO_WORLD_CM, G_M, 100.0, 20000.0)
    p_damp_dark = g.scalar("MacroDampDarken", 0.45, G_M, 0.0, 1.0)
    p_damp_rough = g.scalar("MacroDampRoughAdd", 0.12, G_M, 0.0, 0.5)
    p_leaf_amt = g.scalar("MacroLeafAmount", 0.35, G_M, 0.0, 1.0)
    p_leaf_tint = g.vector("MacroLeafTint", (1.25, 0.90, 0.60), G_M, "multiplier towards the mulch / pine-needle brown")
    p_bright = g.scalar("MacroBrightnessAmount", 0.5, G_M, 0.0, 2.0)
    p_dmag = g.scalar("DisplacementMagnitudeCm", float(_cfg("DISPLACEMENT_MAGNITUDE_CM")), G_D, 0.0, 50.0)
    p_dcen = g.scalar("DisplacementCenter", float(_cfg("DISPLACEMENT_CENTER")), G_D, 0.0, 1.0, "kit height mean = 0.5, std = 0.18")

    wp = g.world_pos()

    # ---- per-layer UV, weights, textures -------------------------------------------------------
    cols, nrms, orms, hts, ws = [], [], [], [], []
    at_mask = None
    for i, (layer, world_cm) in enumerate(layers):
        g.sec("Layer %d: %s  (%g cm per tile)" % (i, layer, world_cm))
        p_cm = g.scalar("WorldSizeCm_" + layer, world_cm, G_L, 10.0, 2000.0, "real-world size of one %s tile (cm)" % layer)
        p_sc = g.scalar("UVScale_" + layer, 1.0, G_L, 0.1, 10.0, "extra tiling multiplier for %s" % layer)
        uv = g.custom("UV " + layer, HLSL_LAYER_UV, [("WP", wp), ("WorldCm", p_cm), ("UVScale", p_sc)], out="f2")
        ws.append(make_layer_weight(g, layer, 1.0 if i == 0 else 0.0))
        texs = tex_by_layer[layer]
        samples = {}
        for suffix in ("BaseColor", "Normal", "ORM", "Height"):
            samples[suffix] = g.tex("%s_%s" % (layer, suffix), texs[suffix][0], suffix, G_L, uv)
        PARAMS["texture"].extend([(name, "%s_%s" % (layer, s), "") for s in samples])
        # anti-tile second sample (shares parameter names with the first -> VERIFY duplicate names are accepted)
        if at_mask is None:
            at_mask = g.custom("AntiTile mask (low-freq noise)", HLSL_AT_MASK,
                               [("WP", wp), ("MaskScaleCm", p_mscale), ("Strength", p_mstr)], out="f1")
        uv2 = g.custom("UV2 " + layer, hlsl_at_uv(i), [("UV", uv), ("RotDeg", p_rot), ("Ratio", p_ratio)], out="f2")
        s2 = {}
        for suffix in ("BaseColor", "Normal", "ORM", "Height"):
            s2[suffix] = g.tex("%s_%s" % (layer, suffix), texs[suffix][0], suffix, G_L, uv2, desc="%s_%s (2nd)" % (layer, suffix))
        nrm_at = g.custom("Normal anti-tile " + layer, hlsl_at_normal(i),
                          [("N1", samples["Normal"]), ("N2", s2["Normal"]), ("M", at_mask), ("RotDeg", p_rot)], out="f3")
        c_at = g.lerp(samples["BaseColor"], s2["BaseColor"], at_mask, desc="anti-tile color")
        o_at = g.lerp(samples["ORM"], s2["ORM"], at_mask, desc="anti-tile ORM")
        h_at = g.lerp(g.mask(samples["Height"], "r"), g.mask(s2["Height"], "r"), at_mask, desc="anti-tile height")
        cols.append(g.sw("UseAntiTile", c_at, samples["BaseColor"], desc="BaseColor " + layer))
        nrms.append(g.sw("UseAntiTile", nrm_at, samples["Normal"], desc="Normal " + layer))
        orms.append(g.sw("UseAntiTile", o_at, samples["ORM"], desc="ORM " + layer))
        hts.append(g.sw("UseAntiTile", h_at, g.mask(samples["Height"], "r"), desc="Height " + layer))

    # ---- height blend ---------------------------------------------------------------------------
    g.sec("Height blend: a_i = h_i + w_i ; m = max(a) - Depth ; w' = max(a - m, 0) / sum  (matches texforge/blend.py)")
    new_w = []
    for grp in range(ngroups):
        hb_inputs = [("H%d" % i, hts[i]) for i in range(n)] + [("W%d" % i, ws[i]) for i in range(n)] + [("Depth", p_depth)]
        hb = g.custom("HeightBlend weights g%d" % grp, hlsl_weights(n, True, grp), hb_inputs, out="f3")
        lin = g.custom("Linear weights g%d" % grp, hlsl_weights(n, False, grp), [("W%d" % i, ws[i]) for i in range(n)], out="f3")
        new_w.append(g.sw("UseHeightBlend", hb, lin, desc="weights g%d" % grp))

    def combine(label, kind, items, out):
        inputs = [("C%d" % i, items[i]) for i in range(n)] + [("WG%d" % k, new_w[k]) for k in range(ngroups)]
        return g.custom(label, hlsl_combine(kind, n), inputs, out=out)

    g.sec("Combine layers")
    color = combine("Blend BaseColor", "color3", cols, "f3")
    normal = combine("Blend Normal (slope space)", "normal", nrms, "f3")
    orm = combine("Blend ORM", "color3", orms, "f3")
    height = combine("Blend Height", "scalar", hts, "f1")

    # ---- macro ----------------------------------------------------------------------------------
    g.sec("Macro variation (R damp, G leaf drift, B brightness)")
    mtex = tex_by_layer["__macro__"]["Macro"][0]
    macro_uv = g.custom("Macro UV", HLSL_MACRO_UV, [("WP", wp), ("MacroWorldCm", p_macro_cm)], out="f2")
    macro = g.tex("MacroTex", mtex, "Macro", G_M, macro_uv, desc="MacroTex (linear RGB)")
    PARAMS["texture"].append((name, "MacroTex", ""))
    rough0 = g.mask(orm, "g")
    color_m = g.custom("Macro color", HLSL_MACRO_COLOR, [("C", color), ("M", macro), ("DampDarken", p_damp_dark),
                                                          ("LeafTint", g.mask(p_leaf_tint, "rgb")),
                                                          ("LeafAmount", p_leaf_amt), ("BrightAmount", p_bright)], out="f3")
    rough_m = g.custom("Macro roughness", HLSL_MACRO_ROUGH, [("R", rough0), ("M", macro), ("DampRoughAdd", p_damp_rough)], out="f1")
    final_color = g.sw("UseMacro", color_m, color, desc="final BaseColor")
    final_rough = g.sw("UseMacro", rough_m, rough0, desc="final Roughness")

    # ---- outputs --------------------------------------------------------------------------------
    g.sec("Outputs")
    connect_prop(final_color, "MP_BASE_COLOR")
    connect_prop(normal, "MP_NORMAL")
    connect_prop(final_rough, "MP_ROUGHNESS")
    connect_prop(g.mask(orm, "r"), "MP_AMBIENT_OCCLUSION")
    connect_prop(g.mask(orm, "b"), "MP_METALLIC")

    g.sec("Nanite displacement hook (blended Height)")
    g.note(LANDSCAPE_DISPLACEMENT_NOTES)
    try:
        disp = g.add_(g.mul(g.sub(height, p_dcen), p_dmag), 0.5, desc="D = 0.5 + (H - center) * magnitude")
        disp_sw = g.sw("UseDisplacement", disp, g.c1(0.5), desc="Displacement (off = flat 0.5)")
        if not connect_prop(disp_sw, "MP_DISPLACEMENT", soft=True):   # VERIFY: MaterialProperty.MP_DISPLACEMENT in 5.8
            _warn("MP_DISPLACEMENT not connectable: the rest of the material is complete; wire 'Height' into Displacement by hand")
    except Exception as exc:
        _warn("displacement hook skipped: %s" % exc)

    g.sec("Runtime virtual texture output (optional)")
    g.note(RVT_NOTES)
    if _cfg("RVT_NODE") == "1":
        try:
            if getattr(unreal, "MaterialExpressionRuntimeVirtualTextureOutput", None) is None:
                raise RuntimeError("unreal.MaterialExpressionRuntimeVirtualTextureOutput missing")   # VERIFY class name
            rvt = g.add("RuntimeVirtualTextureOutput", 6, h=200, desc="RVT output (UseRVT)")
            half = g.c1(0.5)
            link(g.sw("UseRVT", final_color, g.c3(0.5, 0.5, 0.5), desc="RVT BaseColor"), rvt, ["BaseColor"], "RVT.BaseColor")
            link(g.sw("UseRVT", final_rough, half, desc="RVT Roughness"), rvt, ["Roughness"], "RVT.Roughness")
            link(g.sw("UseRVT", normal, g.c3(0.0, 0.0, 1.0), desc="RVT Normal"), rvt, ["Normal"], "RVT.Normal")
            link(half, rvt, ["Specular"], "RVT.Specular")   # VERIFY pin names (Specular, WorldHeight, Opacity, Mask also exist)
        except Exception as exc:
            _warn("RVT output skipped: %s" % exc)
    _finish_material(g, mat)
    return mat


# --------------------------------------------------------------------------------------------------
# layer info objects
# --------------------------------------------------------------------------------------------------
def make_layer_info(layer):
    d = _cfg("LAYERS_DIR")
    name = "LI_" + layer
    path = "%s/%s" % (d, name)
    EAL.make_directory(d)
    cls = getattr(unreal, "LandscapeLayerInfoObject", None)   # VERIFY: exposed to python
    if cls is None:
        _warn("unreal.LandscapeLayerInfoObject missing: create layer infos in the Landscape Paint tab ('+' > Weight-Blended Layer)")
        return None
    obj = EAL.load_asset(path) if EAL.does_asset_exist(path) else None
    if obj is None:
        tools = unreal.AssetToolsHelpers.get_asset_tools()
        factory = None
        for fname in ("LandscapeLayerInfoObjectFactory", "LandscapeLayerInfoFactory"):   # VERIFY: any of these exists?
            fcls = getattr(unreal, fname, None)
            if fcls is not None:
                factory = fcls()
                break
        try:
            obj = tools.create_asset(name, d, cls, factory)
        except Exception as exc:
            _warn("create_asset(%s) raised %s" % (name, exc))
            obj = None
    if obj is None:
        _warn("could not create %s from python (no factory?): create it in the Landscape Paint tab instead (layer name '%s', Weight-Blended)" % (name, layer))
        return None
    _set(obj, "layer_name", unreal.Name(layer))        # VERIFY: property layer_name (FName); must equal the material parameter name
    _soft(obj, "no_weight_blend", False)               # VERIFY: weight-blended (required by the height-blend formula)
    try:
        EAL.save_loaded_asset(obj)
    except Exception as exc:
        _warn("could not save %s: %s" % (path, exc))
    CREATED.append("%s  (LandscapeLayerInfoObject, layer_name=%s, weight-blended)" % (path, layer))
    return obj


MANUAL_STEPS = """MANUAL STEPS (the script does not touch levels or landscape actors):
  1. Open M_Forest_Landscape once; fix any red node / HLSL error shown in the Output Log (read this log first).
  2. Select the Landscape actor > Details > Landscape > Landscape Material = /Game/Forest/Materials/M_Forest_Landscape.
  3. Landscape Mode > Paint tab: for each layer (%(names)s) click '+' next to the layer and either pick the existing
     /Game/Forest/Layers/LI_<layer> (Layer Info) or create a Weight-Blended one; the layer name must equal the layer above.
  4. Paint the layers (the first layer 'dirt' is the base; give every vertex some weight - weights are normalised).
  5. Compare Height Blend: toggle UseHeightBlend / UseAntiTile / UseMacro in the material (or an MI) - recompiles shaders.
  6. Nanite displacement / RVT: follow the comment boxes inside M_Forest_Landscape (and the module docstring); both are OFF.
  7. Bark: assign M_Forest_Bark to trunk meshes; litter: M_Forest_Litter on foliage / instanced meshes or cards (UV into the atlas cell)."""


# --------------------------------------------------------------------------------------------------
def main():
    if MEL is None or EAL is None:
        raise RuntimeError("This script must run inside the Unreal Editor (unreal.MaterialEditingLibrary missing).")
    todo = [s.strip() for s in _cfg("BUILD").split(",") if s.strip()]
    if not _cfg("SOURCE_DIR"):
        raise RuntimeError("SOURCE_DIR is empty: edit CONFIG['SOURCE_DIR'] or set TEXFORGE_SOURCE_DIR to the kit output folder")
    layers, bark_cm, depth = load_kit()
    layers = [(nm, sidecar_world_cm(nm, cm)) for nm, cm in layers]
    _info("building %s ; layers=%s ; HeightBlendDepth=%.3f ; SOURCE_DIR=%s" % (todo, layers, depth, _cfg("SOURCE_DIR")))

    defaults = ensure_default_textures()
    suffixes4 = ("BaseColor", "Normal", "ORM", "Height")
    tex_by_layer = {}
    for nm, cm in layers:
        folder = os.path.join(_cfg("SOURCE_DIR"), nm)
        if not os.path.isdir(folder):
            _warn("layer folder missing: %s (layer '%s' will use neutral default textures)" % (folder, nm))
        tex_by_layer[nm] = import_set(nm, nm, suffixes4, "wrap", defaults, "layer " + nm) if "textures" in todo else \
            dict((s, (EAL.load_asset("%s/T_%s_%s" % (_cfg("TEXTURES_DIR"), nm, s)) or defaults.get(s), False)) for s in suffixes4)
    tex_by_layer["__macro__"] = import_set("macro", "macro", ("Macro",), "wrap", defaults, "macro")
    bark_tex = import_set("bark", "bark", ("BaseColor", "Normal", "ORM", "Height"), "wrap", defaults, "bark") if "bark" in todo else None
    litter_tex = import_set("litter", "litter", ("BaseColor", "Normal", "ORM"), "clamp", defaults, "litter") if "litter" in todo else None

    if litter_tex is not None:
        meta = _read_json(os.path.join(_cfg("SOURCE_DIR"), "litter", "litter.texforge.json").replace("\\", "/")) \
            if os.path.exists(os.path.join(_cfg("SOURCE_DIR"), "litter", "litter.texforge.json")) else None
        if meta:
            items = meta.get("items") or []
            _info("litter atlas: %d items in sidecar (uv rects stay in litter.texforge.json, not baked into the material)" % len(items))

    built = []
    for what, fn in (("bark", lambda: build_bark(bark_tex)), ("litter", lambda: build_litter(litter_tex)),
                     ("landscape", lambda: build_landscape(layers, tex_by_layer, depth))):
        if what not in todo:
            continue
        if what == "bark" and bark_tex is None or what == "litter" and litter_tex is None:
            continue
        try:
            built.append(fn().get_path_name())
        except Exception as exc:
            import traceback
            _fail("building '%s' aborted: %s\n%s" % (what, exc, traceback.format_exc()))

    infos = []
    if "layer_infos" in todo:
        for nm, _cm in layers:
            try:
                if make_layer_info(nm) is not None:
                    infos.append(nm)
            except Exception as exc:
                _warn("layer info %s failed: %s" % (nm, exc))

    _info("==== SUMMARY ====")
    for p in built:
        _info("built " + p)
    for c in CREATED:
        _info("asset " + c)
    if infos:
        _info("layer infos: %s" % ", ".join(infos))
    for kind in ("static", "scalar", "vector", "texture"):
        for mat_name, pname, default in PARAMS[kind]:
            _info("param %-8s %-20s %-30s default=%s" % (kind, mat_name, pname, default))
    _info("%d failure(s), %d warning(s)" % (len(FAILS), len(WARNS)))
    for f in FAILS:
        unreal.log_error("[forest] FAIL " + f)
    for w in WARNS:
        unreal.log_warning("[forest] WARN " + w)
    for line in (MANUAL_STEPS % {"names": ", ".join(nm for nm, _ in layers)}).splitlines():
        _info(line)
    if not FAILS:
        _info("No wiring failures reported. STILL open the materials and check the Output Log for HLSL compile errors "
              "(Custom node bodies are only compiled by the shader compiler, not by this script).")


main()
