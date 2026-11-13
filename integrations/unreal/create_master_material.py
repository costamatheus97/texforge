"""
texforge -> Unreal Engine 5.8 : master material builder  (editor Python, UNTESTED - see VERIFY notes)
==================================================================================================

Creates / rebuilds, under /Game/Texforge/ :
    M_TexforgeMaster          opaque surface master (tiles + trims), feature-switched
    M_TexforgeMaster_Masked   same graph, BLEND_MASKED, adds the "silhouette clip" opacity mask
    M_TexforgeDecal           deferred decal (BaseColor/Normal/Roughness/Metallic/Opacity)
    M_TexforgeWindowRoom      window glass with interior mapping (atlas box projection or engine cubemap)
    /Game/Texforge/Defaults/  tiny generated default textures (needed so the masters compile with no MI)

HOW TO RUN (inside the editor, never from a shell)
    Output Log -> switch the console box to "Python" and run:
        py "/path/to/texforge/integrations/unreal/create_master_material.py"
    or  Tools > Execute Python Script...  and pick this file.
    Options: edit CONFIG below, or set environment variables TEXFORGE_<KEY> before starting the editor
    (e.g. TEXFORGE_BUILD="master,decal", TEXFORGE_POM_IMPL=engine). Editor Python has no argv.
    Re-running is safe: existing master assets are cleared in place (delete_all_material_expressions) and
    rebuilt, so material instances that point at them keep working.

AFTER RUNNING: open each material once. Any red node / "[Compiler]" error in the Output Log means a pin
name or HLSL line was wrong - the script prints every connection it could not make ("[texforge] FAIL").
The script ends with a list of all failures and warnings.

DESIGN NOTES
  * All textures are declared once as MaterialExpressionTextureObjectParameter and sampled inside tiny
    Custom (HLSL) nodes (Tex.Sample / Tex.SampleGrad). This is required for POM and hex tiling (they need the
    texture object and explicit gradients) and gives ONE code path for plain / two-scale / hex sampling.
    Normal maps are decoded manually in HLSL (xy*2-1, z reconstructed) because a Custom node receives the raw
    texels, not the engine-unpacked value.
  * Every optional feature sits behind a StaticBoolParameter feeding StaticSwitch nodes, so a disabled
    feature is not compiled at all (zero cost). Switches are per-instance (they create shader permutations:
    do not enable every combination on hundreds of instances).
  * UseWorldAlignedUV = True : planar projection of absolute world position onto the dominant axis of the
    VERTEX normal (ProjectionAxis 0 = auto, 1 = top/Z, 2 = front/Y, 3 = side/X). Hard seams where the
    dominant axis flips (no triplanar blend - texforge tiles are meant for flat/box-ish architecture).
    The sampled normal is rotated from projection space into world space and then transformed to tangent
    space, so lighting is correct even though the mesh tangents do not follow the projection.
    UseWorldAlignedUV = False : mesh UV0 (* UVScale), tangent-space normals used as-is (trims, props).
  * UV pipeline:  UV_proj -> * UVTransform.rg + UVTransform.ba  (strip select / non-square tiles)
                  -> [POM offset] -> every texture sample. Edge-mask texture uses raw UV0.
  * "CmPerUV" = WorldSizeCm / UVScale = physical cm covered by one texture repeat. It drives POM depth,
    macro/dirt noise scale and the under-layer scale.

FEATURE TABLE
  feature (static switch)      what it does                                   cost                          when to use
  ---------------------------  ---------------------------------------------  ----------------------------  ------------------------------------------
  UseMacroVariation            multiplies BaseColor by 2-octave value noise   ~40 ALU, 0 extra tex samples  any large surface; first thing to try
  UseTwoScaleBlend             2nd sample of BaseColor/Normal/ORM at 0.37x,   2x texture samples (x3 maps),  patterned or organic; keeps joints mostly
                               rotated, blended with a low-freq noise mask    ~40 ALU                       readable; some contrast loss in blended zones
  UseHexTiling                 Mikkelsen hex tiling, 3 jittered samples       3x texture samples (x3 maps)  NON-PATTERNED only (plaster, dirt, rock,
                               blended with variance-preserving weights       + ~60 ALU                     grass). BREAKS brick/tile joints and any
                                                                                                            oriented pattern. Overrides TwoScale if both on.
  UsePOM                       parallax occlusion mapping from Height         min..max Height samples/pixel  close-up flat surfaces; scale steps down
                               (custom HLSL ray march or engine function)     (default 8..32) + cheap       with distance via MinSteps; avoid on huge
                                                                              view-dependent step count     overdraw
  UseSilhouetteClip            (Masked variant ONLY) opacity mask = 0 where   masked blend = no early-Z,    flat UV-island plates/trims with POM where
                               the POM-offset mesh UV leaves [0,1] (+margin)  needs TAA to hide edges;      you want bumpy outlines. Mesh-UV mode only.
                                                                              1 extra small custom node
  UseDirt                      world-Z gradient dirt (+ noise breakup):       ~50 ALU, 0 tex samples        ground-contact grime on walls/props
                               darkens color, raises roughness near ground
  UseEdgeWear                  mask (vertex colour OR EdgeMaskTex, broken up  3 extra texture samples +     plaster-over-brick, painted-over-metal etc.
                               by the Height texture) blends to an "under"    1 for breakup + ~40 ALU
                               material set; normals use whiteout blend
  (always on) tint/brightness/roughness/metallic/AO/normal-strength multipliers - ~10 ALU

MASTER PARAMETER REFERENCE is printed at the end of a run and kept in PARAMS below.

VERIFY LIST (things that could not be checked without the editor; search for "# VERIFY" in the code)
  1. Custom-node texture inputs: a TextureObject input named `Tex` must expose `TexSampler` (documented UE behaviour).
  2. WorldPosition is LWC in 5.x; it is fed straight into Custom inputs (expected: auto-converted to float3).
  3. Static switch pin names ("True"/"False"/"Value"); the script introspects and prints the real ones on mismatch.
  4. unreal.CustomInput() constructible + set_editor_property('inputs') rebuilds pins (the script counts pins afterwards).
  5. MaterialExpressionTextureObjectParameter / StaticBoolParameter / VertexColor R,G,B,A output names.
  6. Transform World->Tangent on camera vector / world normal inside the pixel shader (used by the engine POM function too).
  7. Engine function pin names (POM, InteriorCubemap) come from `strings` on the .uasset, not from a live graph.
  8. shading_model / decal_blend_mode property writes (soft: warning only).
  9. Default textures: generated PNGs must import with the intended compression, else sampler-type compile errors.
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
    "MASTER_DIR": "/Game/Texforge",                    # where the master materials live
    "DEFAULTS_DIR": "/Game/Texforge/Defaults",         # generated default textures
    "BUILD": "master,master_masked,decal,window_room", # comma list of what to (re)build
    "POM_IMPL": "custom",                              # "custom" (own HLSL, works in both UV modes) | "engine" (Texturing/ParallaxOcclusionMapping, mesh-UV semantics)
    "SAVE": "1",                                       # save assets at the end
    "AUTO_LAYOUT": "0",                                # 1 = call MaterialEditingLibrary.layout_material_expressions after the manual layout (overwrites positions, comments lose their frames)
    "ENGINE_POM_FUNCTION": "/Engine/Functions/Engine_MaterialFunctions01/Texturing/ParallaxOcclusionMapping",
    "ENGINE_INTERIOR_CUBEMAP_FUNCTION": "/Engine/Functions/Engine_MaterialFunctions01/Cubemaps/InteriorCubemap",
    "DEFAULT_CUBE_CANDIDATES": "/Engine/EngineResources/GrayTextureCube,/Engine/EngineResources/DefaultTextureCube",
}


def _cfg(key):
    return os.environ.get("TEXFORGE_" + key, CONFIG[key])


# room atlas layout (FINAL, from texforge): 3 columns x 2 rows of equal square cells, rectified by homography, row 0 = TOP of the image.
# (col,row) per face. Per-face (u,v) conventions (implemented in hlsl_room):
#   back    u left->right  v top->bottom      left/right wall  u front->back  v top->bottom
#   floor/ceiling  u left->right  v front->back      cell (2,1) is unused (mid grey)
ATLAS_COLS = 3
ATLAS_ROWS = 2
ATLAS_CELLS = {"back": (0, 0), "left": (1, 0), "right": (2, 0), "floor": (0, 1), "ceiling": (1, 1)}

# --------------------------------------------------------------------------------------------------
# defensive layer
# --------------------------------------------------------------------------------------------------
FAILS = []
WARNS = []
PARAMS = {"scalar": [], "vector": [], "texture": [], "static": []}   # (material, name, default)
CURRENT_MATERIAL = ["?"]
CURRENT_MAT = [None]

MEL = getattr(unreal, "MaterialEditingLibrary", None)
EAL = getattr(unreal, "EditorAssetLibrary", None)


def _info(msg):
    unreal.log("[texforge] " + str(msg))


def _warn(msg):
    WARNS.append(str(msg))
    unreal.log_warning("[texforge] WARN " + str(msg))


def _fail(msg):
    FAILS.append("[%s] %s" % (CURRENT_MATERIAL[0], msg))
    unreal.log_error("[texforge] FAIL " + str(msg))


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
    """Like _set but a failure is only a warning (properties that already default to the wanted value)."""
    try:
        obj.set_editor_property(prop, value)
        return True
    except Exception as exc:  # noqa
        _warn("could not set %s.%s (%s); check it in the material details panel" % (type(obj).__name__, prop, exc))
        return False


def _enum(cls_name, member):
    """Return unreal.<cls_name>.<member> trying the usual python spellings; None (+FAIL) if absent."""
    cls = getattr(unreal, cls_name, None)
    if cls is not None:
        for cand in (member, member.upper(), member.replace("_", "").upper()):
            if hasattr(cls, cand):
                return getattr(cls, cand)
    _fail("enum unreal.%s.%s not found" % (cls_name, member))
    return None


def _names(fn, expr):
    try:
        res = fn(expr)
        return [str(n) for n in res]
    except Exception:
        return None


def _in_names(expr):
    return _names(MEL.get_material_expression_input_names, expr)


def _out_names(expr):
    return _names(MEL.get_material_expression_output_names, expr)


def _sig(x):
    return x if isinstance(x, tuple) else (x, "")


def link(src, dst, pins, what=""):
    """Connect src (expr or (expr, output_name)) to input `pin` of dst. `pins` may be a list of candidate
    names (first one that exists on the node wins). Checks pin names via introspection AND the bool result of
    connect_material_expressions, which fails silently on a wrong name."""
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
    if use is None and avail is not None and len(avail) == 1 and avail[0] in ("None", ""):
        use = ""      # 5.8: the single unnamed input of ComponentMask / Saturate / Transform etc. is reported as 'None' and connected with an empty name
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


def connect_prop(src, prop_name, what=""):
    """Connect to a material output pin (MP_BASE_COLOR ...) and read it back to prove it stuck."""
    se, so = _sig(src)
    prop = _enum("MaterialProperty", prop_name)
    if prop is None:
        return False
    try:
        ok = MEL.connect_material_property(se, so, prop)
    except Exception as exc:
        _fail("connect_material_property(%s) raised %s" % (prop_name, exc))
        return False
    if not ok:
        _fail("connect_material_property(%s) returned False (%s)" % (prop_name, _desc(se)))
        return False
    try:  # read back: proves the pin really holds the node (connect can report True on a stale pin)
        if MEL.get_material_property_input_node(CURRENT_MAT[0], prop) is None:
            _fail("material pin %s is still empty after connect (%s)" % (prop_name, _desc(se)))
            return False
    except Exception as exc:
        _warn("could not read back pin %s: %s" % (prop_name, exc))
    return True


# --------------------------------------------------------------------------------------------------
# HLSL snippets (Custom node bodies). Placeholders @NAME@ are substituted by _tpl().
# NOTE: Custom bodies cannot define functions, everything is inline; the helpers below generate repeated blocks.
# --------------------------------------------------------------------------------------------------
def _tpl(code, **kw):
    for k, v in kw.items():
        code = code.replace("@%s@" % k, v)
    return code


def hash22(out, expr):
    """Hoskins hash22: float2 `out` from float2 expression `expr` (no sin(), stable for large inputs)."""
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


# projection basis for world-aligned UVs. Inputs required: N (float3 vertex normal WS), Axis (float).
# Produces Ud (direction of increasing U), Vd (direction of increasing V = "down"), Fd (face normal).
# The (U,V,F) frame is LEFT-handed on purpose: it matches DirectX/Unreal tangent space (green = +V = down).
BASIS = """float3 an = abs(N);
int ax = (an.z >= an.x && an.z >= an.y) ? 2 : ((an.x >= an.y) ? 0 : 1);
int fa = (int)(Axis + 0.5);
if (fa == 1) ax = 2; else if (fa == 2) ax = 1; else if (fa == 3) ax = 0;
float3 Ud; float3 Vd; float3 Fd;
if (ax == 2) { float sg = (N.z >= 0.0) ? 1.0 : -1.0; Ud = float3(sg, 0.0, 0.0); Vd = float3(0.0, -1.0, 0.0); Fd = float3(0.0, 0.0, sg); }
else if (ax == 1) { float sg = (N.y >= 0.0) ? 1.0 : -1.0; Ud = float3(-sg, 0.0, 0.0); Vd = float3(0.0, 0.0, -1.0); Fd = float3(0.0, sg, 0.0); }
else { float sg = (N.x >= 0.0) ? 1.0 : -1.0; Ud = float3(0.0, sg, 0.0); Vd = float3(0.0, 0.0, -1.0); Fd = float3(sg, 0.0, 0.0); }
"""

HLSL_WORLD_UV = BASIS + "return float2(dot(WP, Ud), dot(WP, Vd)) / max(CmPerUV, 0.0001);\n"
HLSL_VIEW_PROJ = BASIS + "return float3(dot(Vw, Ud), dot(Vw, Vd), dot(Vw, Fd));\n"
HLSL_NORMAL_TO_WORLD = BASIS + "return normalize(Nt.x * Ud + Nt.y * Vd + Nt.z * Fd);\n"
HLSL_NORMAL_STRENGTH = "return normalize(float3(Nt.xy * Strength, max(Nt.z, 0.0001)));\n"

# Custom POM. Ray-marches the height field in the projection/tangent frame.
#   UV   : texture UV (already transformed), Vt : view vector (surface -> camera) in the frame where x = +U, y = +V, z = out
#   DepthUV : relief depth in texture-UV units (= POMHeightCm / CmPerUV)
# Height = 1 is "high" (surface), depth = 1 - height. Linear search + one secant refinement. Uses SampleGrad
# with the gradients of the *input* UV so there are no mip seams inside the loop.
HLSL_POM = """float3 v = normalize(Vt);
float vz = max(v.z, 0.08);
float2 dUV = -v.xy / vz * DepthUV;
float nS = lerp(MaxSteps, MinSteps, saturate(v.z));
nS = clamp(floor(nS + 0.5), 1.0, 128.0);
float dt = 1.0 / nS;
float2 gx = ddx(UV);
float2 gy = ddy(UV);
float h0 = HeightTex.SampleGrad(HeightTexSampler, UV, gx, gy).r;
float t = 0.0;
float tPrev = 0.0;
float dPrev = h0 - 1.0;
float tHit = 1.0;
[loop] for (int i = 0; i < 128; i++)
{
    if ((float)i >= nS) break;
    t += dt;
    float h = HeightTex.SampleGrad(HeightTexSampler, UV + dUV * t, gx, gy).r;
    float d = t - (1.0 - h);
    if (d >= 0.0)
    {
        float w = saturate(dPrev / (dPrev - d - 0.00001));
        tHit = lerp(tPrev, t, w);
        break;
    }
    tPrev = t;
    dPrev = d;
}
return UV + dUV * tHit;
"""

# Silhouette clip: 1 if the parallax-offset mesh UV0 is still inside the UV island [0,1] (+margin), else 0.
HLSL_CLIP = """float2 sc = max(float2(UVScale, UVScale) * max(abs(UVTscale), float2(0.0001, 0.0001)), float2(0.0001, 0.0001));
float2 m = UV0 + (UVHit - UVBase) / sc;
float inside = (m.x >= -Margin && m.x <= 1.0 + Margin && m.y >= -Margin && m.y <= 1.0 + Margin) ? 1.0 : 0.0;
return inside;
"""

HLSL_MACRO = noise2("nm", "UV * CmPerUV / max(MacroScaleCm, 1.0)") + \
    "return C * max(1.0 + (nm - 0.5) * 2.0 * MacroStrength, 0.0);\n"

HLSL_DIRT = noise2("nd", "float2(WP.x + WP.y, WP.z) / max(DirtNoiseScaleCm, 1.0)") + """float hgt = (WP.z - GroundZ) / max(DirtHeightCm, 0.001);
float baseM = pow(saturate(1.0 - hgt), max(DirtFalloff, 0.01));
float m = saturate(baseM * (1.0 + (nd - 0.5) * 2.0 * DirtNoiseAmount)) * saturate(DirtStrength);
float3 oc = lerp(C, C * DirtColor, m);
float orr = saturate(Rough + DirtRoughnessAdd * m);
return float4(oc, orr);
"""

HLSL_VC_PICK = """int ch = (int)(Channel + 0.5);
return (ch == 0) ? VCR : ((ch == 1) ? VCG : ((ch == 2) ? VCB : VCA));
"""

HLSL_EDGE_MASK = """float x = Wear + (Breakup01 - 0.5) * 2.0 * BreakupAmount;
return saturate((x - Threshold) * Contrast + 0.5);
"""

# whiteout blend of base and under normals, each flattened by its own mask first (proper mask-aware blend).
HLSL_EDGE_NORMAL = """float2 b = Nb.xy * (1.0 - M);
float2 u = Nu.xy * M;
float bz = lerp(1.0, Nb.z, 1.0 - M);
float uz = lerp(1.0, Nu.z, M);
return normalize(float3(b + u, max(bz * uz, 0.0001)));
"""


_TFS_PRELUDE = """#if RAYHITGROUPSHADER
#define TFS(T, S, U) T.SampleLevel(S, U, 0)
#else
#define TFS(T, S, U) T.Sample(S, U)
#endif
"""


def hlsl_sample(kind, mode):
    """Body of a texture-sampling Custom node. kind: color|orm|normal|scalar ; mode: plain|two|hex.
    Always has inputs Tex (texture object) and UV (float2); extra inputs per mode:
      two : CmPerUV, MaskScaleCm, Ratio, RotDeg, Strength        hex : HexScale, Sharp
    Returns (code, extra_input_names, output_type 'f1'|'f3')."""
    out = "f1" if kind == "scalar" else "f3"
    is_n = kind == "normal"

    def finish_n(var):
        return "return float3(%s, sqrt(saturate(1.0 - dot(%s, %s))));\n" % (var, var, var)

    if mode == "plain":
        if kind == "scalar":
            return "return TFS(Tex, TexSampler, UV).r;\n", [], out
        if is_n:
            return "float4 s1 = TFS(Tex, TexSampler, UV);\nfloat2 n = s1.xy * 2.0 - 1.0;\n" + finish_n("n"), [], out
        return "return TFS(Tex, TexSampler, UV).rgb;\n", [], out

    if mode == "two":
        code = """float4 s1 = TFS(Tex, TexSampler, UV);
float ang = RotDeg * 0.017453292;
float ca = cos(ang);
float sa = sin(ang);
float2 uv2 = float2(ca * UV.x - sa * UV.y, sa * UV.x + ca * UV.y) * Ratio + float2(0.37, 0.71);
float4 s2 = TFS(Tex, TexSampler, uv2);
""" + noise2("nmask", "UV * CmPerUV / max(MaskScaleCm, 1.0)") + """float m = smoothstep(0.30, 0.70, nmask) * saturate(Strength);
"""
        if is_n:
            code += """float2 n1 = s1.xy * 2.0 - 1.0;
float2 n2 = s2.xy * 2.0 - 1.0;
float2 n2r = float2(ca * n2.x + sa * n2.y, -sa * n2.x + ca * n2.y);
float2 n = lerp(n1, n2r, m);
""" + finish_n("n")
        else:
            code += "return lerp(s1.rgb, s2.rgb, m);\n"
        return code, ["CmPerUV", "MaskScaleCm", "Ratio", "RotDeg", "Strength"], out

    # hex tiling (Mikkelsen, offset-only variant: no per-tile rotation => normals need no rotation)
    code = """float2 hx_st = UV * HexScale * 3.4641016;
float2 hx_sk = float2(hx_st.x, -0.57735027 * hx_st.x + 1.15470054 * hx_st.y);
float2 hx_id = floor(hx_sk);
float2 hx_fr = frac(hx_sk);
float hx_tz = 1.0 - hx_fr.x - hx_fr.y;
float hx_s = step(0.0, -hx_tz);
float hx_s2 = 2.0 * hx_s - 1.0;
float3 hx_w = saturate(float3(-hx_tz * hx_s2, hx_s - hx_fr.y * hx_s2, hx_s - hx_fr.x * hx_s2));
float2 hx_v1 = hx_id + float2(hx_s, hx_s);
float2 hx_v2 = hx_id + float2(hx_s, 1.0 - hx_s);
float2 hx_v3 = hx_id + float2(1.0 - hx_s, hx_s);
""" + hash22("hx_o1", "hx_v1") + hash22("hx_o2", "hx_v2") + hash22("hx_o3", "hx_v3") + """float2 hx_gx = ddx(UV);
float2 hx_gy = ddy(UV);
float4 hx_s1 = Tex.SampleGrad(TexSampler, UV + hx_o1, hx_gx, hx_gy);
float4 hx_s2s = Tex.SampleGrad(TexSampler, UV + hx_o2, hx_gx, hx_gy);
float4 hx_s3 = Tex.SampleGrad(TexSampler, UV + hx_o3, hx_gx, hx_gy);
float hx_p = max(Sharp, 1.0);
hx_w = pow(hx_w, float3(hx_p, hx_p, hx_p));
hx_w = hx_w / max(hx_w.x + hx_w.y + hx_w.z, 0.00001);
float hx_k = rsqrt(max(dot(hx_w, hx_w), 0.00001));
"""
    if is_n:
        code += """float2 hx_n = (hx_w.x * (hx_s1.xy * 2.0 - 1.0) + hx_w.y * (hx_s2s.xy * 2.0 - 1.0) + hx_w.z * (hx_s3.xy * 2.0 - 1.0)) * hx_k;
float hx_l = length(hx_n);
if (hx_l > 0.99) hx_n = hx_n * (0.99 / hx_l);
""" + finish_n("hx_n")
    else:
        code += """float3 hx_mean = Tex.SampleLevel(TexSampler, float2(0.5, 0.5), 12.0).rgb;
float3 hx_c = hx_w.x * hx_s1.rgb + hx_w.y * hx_s2s.rgb + hx_w.z * hx_s3.rgb;
return saturate(hx_mean + (hx_c - hx_mean) * hx_k);
"""
    return code, ["HexScale", "Sharp"], out


def hlsl_room():
    """Window-room box projection from the texforge room atlas (final layout, see ATLAS_CELLS)."""
    def cell(face):
        c, r = ATLAS_CELLS[face]
        return "float2(%d.0, %d.0)" % (c, r)

    return _tpl("""// ---- face -> atlas cell mapping comes from ATLAS_CELLS in create_master_material.py ----
// Box in "half window width" units: x in [-1,1] (U), y in [-1/a,1/a] (V, down), z in [-D,0] (0 = glass plane, -D = back wall).
float2 uvt = UV * Tiles;
float2 cellId = floor(uvt);
float2 fcell = frac(uvt);
float2 gx = ddx(uvt) * float2(1.0 / @COLS@.0, 1.0 / @ROWS@.0);
float2 gy = ddy(uvt) * float2(1.0 / @COLS@.0, 1.0 / @ROWS@.0);
float a = max(Aspect, 0.01);
float D = max(DepthRatio, 0.01) * 2.0;
float3 p0 = float3(fcell.x * 2.0 - 1.0, (fcell.y * 2.0 - 1.0) / a, 0.0);
float3 d = -normalize(Vt);
d.z = min(d.z, -0.001);
d.x = (abs(d.x) < 0.00001) ? 0.00001 : d.x;
d.y = (abs(d.y) < 0.00001) ? 0.00001 : d.y;
float3 inv = 1.0 / d;
float3 tHi = (float3(1.0, 1.0 / a, 0.0) - p0) * inv;
float3 tLo = (float3(-1.0, -1.0 / a, -D) - p0) * inv;
float3 tExit = max(tHi, tLo);
float t = min(min(tExit.x, tExit.y), tExit.z);
float3 h = p0 + d * t;
float depth01 = saturate(-h.z / D);
float2 uvf;
float2 cellIdx;
if (t >= tExit.z - 0.00001)               // back wall
{
    uvf = float2(h.x * 0.5 + 0.5, h.y * a * 0.5 + 0.5);
    cellIdx = @BACK@;
}
else if (t >= tExit.x - 0.00001)          // side walls
{
    if (d.x > 0.0) { uvf = float2(depth01, h.y * a * 0.5 + 0.5); cellIdx = @RIGHT@; }
    else           { uvf = float2(depth01,       h.y * a * 0.5 + 0.5); cellIdx = @LEFT@;  }
}
else                                       // floor / ceiling (V is down, so +y = floor)
{
    if (d.y > 0.0) { uvf = float2(h.x * 0.5 + 0.5, depth01); cellIdx = @FLOOR@; }
    else           { uvf = float2(h.x * 0.5 + 0.5, depth01);       cellIdx = @CEIL@;  }
}
uvf = lerp(float2(Pad, Pad), float2(1.0 - Pad, 1.0 - Pad), saturate(uvf));
float2 auv = (cellIdx + uvf) / float2(@COLS@.0, @ROWS@.0);
float3 col = Atlas.SampleGrad(AtlasSampler, auv, gx, gy).rgb;
""" + hash22("rr", "cellId") + """col *= 1.0 - Variation * rr.x;
return col;
""", COLS=str(ATLAS_COLS), ROWS=str(ATLAS_ROWS), BACK=cell("back"), LEFT=cell("left"), RIGHT=cell("right"),
        FLOOR=cell("floor"), CEIL=cell("ceiling"))


# --------------------------------------------------------------------------------------------------
# generated default textures
# --------------------------------------------------------------------------------------------------
DEFAULT_TEXTURES = {
    # name: (rgb, compression, srgb, sampler type)
    "T_TF_Default_BaseColor": ((128, 128, 128), "TC_DEFAULT", True, "SAMPLERTYPE_COLOR"),
    "T_TF_Default_Normal": ((128, 128, 255), "TC_NORMALMAP", False, "SAMPLERTYPE_NORMAL"),
    "T_TF_Default_ORM": ((255, 128, 0), "TC_MASKS", False, "SAMPLERTYPE_MASKS"),
    "T_TF_Default_Height": ((128, 128, 128), "TC_GRAYSCALE", False, "SAMPLERTYPE_LINEAR_GRAYSCALE"),
    "T_TF_Default_Mask": ((0, 0, 0), "TC_MASKS", False, "SAMPLERTYPE_MASKS"),
    "T_TF_Default_Opacity": ((255, 255, 255), "TC_MASKS", False, "SAMPLERTYPE_MASKS"),
}
EXPECT_SAMPLER = {  # (compression, srgb) -> sampler type name; used for the compression/sampler consistency check
    ("TC_DEFAULT", True): "SAMPLERTYPE_COLOR",
    ("TC_NORMALMAP", False): "SAMPLERTYPE_NORMAL",
    ("TC_MASKS", False): "SAMPLERTYPE_MASKS",
    ("TC_GRAYSCALE", False): "SAMPLERTYPE_LINEAR_GRAYSCALE",
}


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
    for fn in (lambda: unreal.Paths.project_saved_dir(), lambda: unreal.SystemLibrary.get_project_saved_directory()):  # VERIFY: one of these exists
        try:
            return fn()
        except Exception:
            continue
    import tempfile
    return tempfile.gettempdir()


def ensure_default_textures():
    """Import the generated default textures once and return {name: texture}. Re-imports only if missing."""
    ddir = _cfg("DEFAULTS_DIR")
    EAL.make_directory(ddir)
    tmp = os.path.join(_project_saved_dir(), "Texforge", "Defaults")
    os.makedirs(tmp, exist_ok=True)
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    result = {}
    for name, (rgb, comp, srgb, sampler) in DEFAULT_TEXTURES.items():
        path = "%s/%s" % (ddir, name)
        if not EAL.does_asset_exist(path):
            png = os.path.join(tmp, name + ".png")
            _write_png(png, rgb)
            task = unreal.AssetImportTask()
            for k, v in (("filename", png), ("destination_path", ddir), ("destination_name", name),
                         ("automated", True), ("save", True), ("replace_existing", True)):
                _set(task, k, v)
            tools.import_asset_tasks([task])
        tex = EAL.load_asset(path)
        if tex is None:
            _fail("default texture %s could not be imported/loaded (png at %s)" % (path, tmp))
            continue
        c = _enum("TextureCompressionSettings", comp)
        if c is not None:
            _set(tex, "compression_settings", c)
        _set(tex, "srgb", srgb)
        try:
            EAL.save_loaded_asset(tex)
        except Exception as exc:
            _warn("save %s failed: %s" % (path, exc))
        # consistency check (advisor): sampler type of params using this default must match compression/sRGB
        try:
            got = (str(tex.get_editor_property("compression_settings")).split(".")[-1].replace("TextureCompressionSettings.", ""),
                   bool(tex.get_editor_property("srgb")))
            _info("default texture %s : compression=%s srgb=%s -> sampler %s" % (name, got[0], got[1], sampler))
        except Exception:
            pass
        result[name] = tex
    return result


# --------------------------------------------------------------------------------------------------
# graph builder
# --------------------------------------------------------------------------------------------------
class Graph(object):
    COLW = 420
    BASE_X = -4400

    def __init__(self, mat):
        self.mat = mat
        self.nodes = []          # (expr, x, y, h, section)
        self.col_of = {}
        self.col_y = {}
        self.band_y = 0
        self.section = "Misc"
        self.sections = []
        self.defaults = {}       # default textures
        self.statics = {}        # name -> StaticBoolParameter expr

    # ---- layout ----------------------------------------------------------------
    def sec(self, name):
        maxy = max(self.col_y.values()) if self.col_y else None
        if maxy is not None:
            self.band_y = maxy + 140
        self.col_y = {}
        self.section = name
        self.sections.append(name)

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
                _soft(c, "size_x", int(x1 - x0 + self.COLW))     # 5.8: property name differs; comment frames are cosmetic
                _soft(c, "size_y", int(y1 - y0 + 90))
                _set(c, "font_size", 18)
                _set(c, "comment_color", unreal.LinearColor(0.15, 0.2, 0.3, 1.0))
            except Exception as exc:
                _warn("comment node for section '%s' failed: %s" % (sec, exc))

    # ---- parameters ------------------------------------------------------------
    def scalar(self, name, default, group, smin=None, smax=None, desc=None):
        e = self.add("ScalarParameter", 0, h=110, parameter_name=name, default_value=float(default), group=group,
                     desc=desc or name)
        if smin is not None:
            _set(e, "slider_min", float(smin))
        if smax is not None:
            _set(e, "slider_max", float(smax))
        PARAMS["scalar"].append((CURRENT_MATERIAL[0], name, default))
        return e

    def vector(self, name, rgba, group, desc=None):
        rgba = tuple(rgba) + (1.0,) * (4 - len(rgba))
        e = self.add("VectorParameter", 0, h=130, parameter_name=name, default_value=unreal.LinearColor(*rgba),
                     group=group, desc=desc or name)
        PARAMS["vector"].append((CURRENT_MATERIAL[0], name, rgba))
        return e

    def static(self, name, default, group, desc=None):
        """Static bool parameter; returns the node (feed it into StaticSwitch 'Value')."""
        e = self.add("StaticBoolParameter", 0, h=110, parameter_name=name, default_value=bool(default), group=group,
                     desc=desc or name)
        self.statics[name] = e
        PARAMS["static"].append((CURRENT_MATERIAL[0], name, bool(default)))
        return e

    def texobj(self, name, default_tex_name, sampler_member, group, desc=None):
        # VERIFY: unreal.MaterialExpressionTextureObjectParameter exists (it does in UE5); add() raises with a clear message otherwise.
        e = self.add("TextureObjectParameter", 0, h=130, parameter_name=name, group=group, desc=desc or name)
        tex = self.defaults.get(default_tex_name)
        if tex is not None:
            _set(e, "texture", tex)
        else:
            _fail("no default texture for parameter %s (%s): the material will not compile" % (name, default_tex_name))
        st = _enum("MaterialSamplerType", sampler_member)
        if st is not None:
            _set(e, "sampler_type", st)   # set AFTER texture (assigning a texture auto-picks the sampler type)
        exp = DEFAULT_TEXTURES.get(default_tex_name)
        if exp and exp[3] != sampler_member:
            _warn("param %s: default texture %s implies sampler %s but %s was requested" % (name, default_tex_name, exp[3], sampler_member))
        PARAMS["texture"].append((CURRENT_MATERIAL[0], name, default_tex_name))
        return e

    # ---- constants / inputs ----------------------------------------------------
    def c1(self, v):
        return self.add("Constant", 0, h=90, r=float(v))

    def c3(self, r, g, b):
        return self.add("Constant3Vector", 0, h=110, constant=unreal.LinearColor(r, g, b, 1.0))

    def c4(self, r, g, b, a):
        return self.add("Constant4Vector", 0, h=110, constant=unreal.LinearColor(r, g, b, a))

    def texcoord(self, idx=0):
        return self.add("TextureCoordinate", 0, h=110, coordinate_index=idx)

    def world_pos(self):
        return self.add("WorldPosition", 0, h=100)

    def vertex_normal(self):
        return self.add("VertexNormalWS", 0, h=90)

    def camera_vec(self):
        return self.add("CameraVectorWS", 0, h=90)

    def vertex_color(self):
        return self.add("VertexColor", 0, h=130)

    # ---- math ------------------------------------------------------------------
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

    def div(self, a, b, desc=None):
        return self._bin("Divide", a, b, desc)

    def lerp(self, a, b, alpha, desc=None):
        n = self.add("LinearInterpolate", self._col(a, b, alpha), h=110, desc=desc)
        for key, v, pin in (("a", a, "A"), ("b", b, "B"), ("alpha", alpha, "Alpha")):
            if isinstance(v, (int, float)):
                _set(n, "const_" + key, float(v))
            else:
                link(v, n, pin, "Lerp.%s" % pin)
        return n

    def unary(self, cls, a, pin="Input", desc=None):
        n = self.add(cls, self._col(a), h=90, desc=desc)
        link(a, n, pin, cls)
        return n

    def sat(self, a):
        return self.unary("Saturate", a)

    def mask(self, a, chans):
        n = self.add("ComponentMask", self._col(a), h=90, r="r" in chans, g="g" in chans, b="b" in chans, a="a" in chans)
        link(a, n, "Input", "ComponentMask")
        return n

    def append(self, a, b):
        n = self.add("AppendVector", self._col(a, b), h=90)
        link(a, n, "A", "Append.A")
        link(b, n, "B", "Append.B")
        return n

    def to_tangent(self, a, desc=None):
        n = self.add("Transform", self._col(a), h=110, desc=desc or "World -> Tangent")
        src = _enum("MaterialVectorCoordTransformSource", "TRANSFORMSOURCE_WORLD")
        dst = _enum("MaterialVectorCoordTransform", "TRANSFORM_TANGENT")
        if src is not None:
            _set(n, "transform_source_type", src)
        if dst is not None:
            _set(n, "transform_type", dst)
        link(a, n, "Input", "Transform")
        return n

    # ---- switches --------------------------------------------------------------
    def sw(self, static_name, when_true, when_false, desc=None):
        """StaticSwitch driven by the named StaticBoolParameter. Branch not selected is not compiled."""
        n = self.add("StaticSwitch", self._col(when_true, when_false, self.statics[static_name]), h=110,
                     desc=desc or ("switch " + static_name))
        # VERIFY: StaticSwitch (non-parameter) pins are named True/False/Value (fallbacks A/B); link() introspects.
        link(when_true, n, ["True", "A"], "StaticSwitch(%s).True" % static_name)
        link(when_false, n, ["False", "B"], "StaticSwitch(%s).False" % static_name)
        link(self.statics[static_name], n, ["Value"], "StaticSwitch(%s).Value" % static_name)
        return n

    # ---- custom ----------------------------------------------------------------
    def custom(self, label, code, inputs, out="f3", h=None):
        """inputs: list of (name, signal). out: f1|f2|f3|f4."""
        col = self._col(*[s for _, s in inputs])
        n = self.add("Custom", col, h=h or (70 + 24 * len(inputs)), desc=label)
        # VERIFY: Custom inputs fed with WorldPosition (LWC in 5.x) are expected to be auto-converted to float3.
        # VERIFY: a TextureObject input named "Tex" gives HLSL access to `Tex` and `TexSampler`.
        ot = _enum("CustomMaterialOutputType", {"f1": "CMOT_FLOAT1", "f2": "CMOT_FLOAT2", "f3": "CMOT_FLOAT3", "f4": "CMOT_FLOAT4"}[out])
        cis = []
        for nm, _ in inputs:
            ci = unreal.CustomInput()
            _set(ci, "input_name", nm)
            cis.append(ci)
        # Implicit-derivative Sample is illegal in ray-tracing closest-hit shaders (UE 5.8 compile fails with
        # "Opcode Sample not valid in lib_6_6"); TFS() falls back to SampleLevel(.,.,0) there.
        code = _TFS_PRELUDE + code
        _set(n, "code", code)
        _set(n, "description", label)
        if ot is not None:
            _set(n, "output_type", ot)
        _set(n, "inputs", cis)
        # VERIFY: after set_editor_property('inputs') the node should expose one pin per CustomInput (checked below)
        avail = _in_names(n)
        if avail is not None and len(avail) != len(inputs):
            _fail("Custom '%s': expected %d input pins, node reports %s" % (label, len(inputs), avail))
        for nm, sig in inputs:
            if isinstance(sig, (int, float)):
                sig = self.c1(sig)
            link(sig, n, nm, "Custom '%s'.%s" % (label, nm))
        return n

    def sample(self, label, kind, mode, tex, uv, extra=None):
        code, extra_names, out = hlsl_sample(kind, mode)
        inputs = [("Tex", tex), ("UV", uv)]
        extra = extra or {}
        for nm in extra_names:
            inputs.append((nm, extra[nm]))
        return self.custom("%s [%s]" % (label, mode), code, inputs, out=out)

    def fcall(self, path, label):
        fn = EAL.load_asset(path) if EAL.does_asset_exist(path) else None
        if fn is None:
            _fail("material function %s not found" % path)
            return None
        n = self.add("MaterialFunctionCall", 3, h=260, desc=label)
        _set(n, "material_function", fn)   # VERIFY: triggers UpdateFromFunctionResource (pins appear); introspected below
        _info("%s pins: inputs=%s outputs=%s" % (path.split("/")[-1], _in_names(n), _out_names(n)))
        return n


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
    if _cfg("AUTO_LAYOUT") == "1":
        try:
            MEL.layout_material_expressions(mat)
        except Exception as exc:
            _warn("layout_material_expressions failed: %s" % exc)
    try:
        MEL.recompile_material(mat)
    except Exception as exc:
        _fail("recompile_material raised %s" % exc)
    if _cfg("SAVE") == "1":
        try:
            EAL.save_loaded_asset(mat)
        except Exception as exc:
            _warn("save failed for %s: %s" % (mat.get_name(), exc))


# --------------------------------------------------------------------------------------------------
# M_TexforgeMaster / M_TexforgeMaster_Masked
# --------------------------------------------------------------------------------------------------
def build_surface_master(name, masked, defaults):
    CURRENT_MATERIAL[0] = name
    mat = _new_material(name, _cfg("MASTER_DIR"))
    _set(mat, "material_domain", _enum("MaterialDomain", "MD_SURFACE"))
    _set(mat, "blend_mode", _enum("BlendMode", "BLEND_MASKED" if masked else "BLEND_OPAQUE"))
    _soft(mat, "shading_model", _enum("MaterialShadingModel", "MSM_DEFAULT_LIT"))  # VERIFY: may be read-only in 5.8 (already the default)
    _set(mat, "tangent_space_normal", True)
    _soft(mat, "two_sided", False)
    if masked:
        _set(mat, "opacity_mask_clip_value", 0.5)

    g = Graph(mat)
    g.defaults = defaults
    pom_impl = _cfg("POM_IMPL")

    # ================= parameters =================
    g.sec("Parameters")
    G_BASE, G_UV, G_AT, G_POM, G_SIL, G_DIRT, G_EDGE = ("00 Base", "10 UV", "20 AntiTiling", "30 POM", "40 SilhouetteClip",
                                                         "50 Dirt", "60 EdgeWear")
    t_base = g.texobj("BaseColor", "T_TF_Default_BaseColor", "SAMPLERTYPE_COLOR", G_BASE)
    t_norm = g.texobj("Normal", "T_TF_Default_Normal", "SAMPLERTYPE_NORMAL", G_BASE)
    t_orm = g.texobj("ORM", "T_TF_Default_ORM", "SAMPLERTYPE_MASKS", G_BASE, desc="ORM: R=AO G=Roughness B=Metallic")
    t_hgt = g.texobj("Height", "T_TF_Default_Height", "SAMPLERTYPE_LINEAR_GRAYSCALE", G_BASE)

    p_world = g.scalar("WorldSizeCm", 100.0, G_UV, 1, 1000, "cm covered by one texture repeat (U direction)")
    p_uvscale = g.scalar("UVScale", 1.0, G_UV, 0.01, 16, "multiplier on mesh UV0 (and on world UV)")
    p_uvt = g.vector("UVTransform", (1.0, 1.0, 0.0, 0.0), G_UV, "UV = UV*(r,g)+(b,a). trims: strip select; non-square tiles: g = w/h")
    b_world = g.static("UseWorldAlignedUV", True, G_UV)
    p_axis = g.scalar("ProjectionAxis", 0, G_UV, 0, 3, "0 auto (dominant normal axis), 1 top Z, 2 front Y, 3 side X")
    p_tint = g.vector("BaseColorTint", (1.0, 1.0, 1.0), G_BASE)
    p_bright = g.scalar("BrightnessMul", 1.0, G_BASE, 0, 4)
    p_rough = g.scalar("RoughnessMul", 1.0, G_BASE, 0, 4)
    p_metal = g.scalar("MetallicMul", 1.0, G_BASE, 0, 1)
    p_ao = g.scalar("AOStrength", 1.0, G_BASE, 0, 1)
    p_nstr = g.scalar("NormalStrength", 1.0, G_BASE, 0, 4)

    b_macro = g.static("UseMacroVariation", False, G_AT)
    p_macro_scale = g.scalar("MacroScaleCm", 600.0, G_AT, 10, 5000)
    p_macro_str = g.scalar("MacroStrength", 0.35, G_AT, 0, 1)
    b_two = g.static("UseTwoScaleBlend", False, G_AT)
    p_two_ratio = g.scalar("TwoScaleRatio", 0.37, G_AT, 0.05, 3)
    p_two_rot = g.scalar("TwoScaleRotationDeg", 23.0, G_AT, 0, 90)
    p_two_str = g.scalar("TwoScaleStrength", 0.6, G_AT, 0, 1)
    p_two_mask = g.scalar("TwoScaleMaskScaleCm", 300.0, G_AT, 10, 5000)
    b_hex = g.static("UseHexTiling", False, G_AT)
    p_hex_scale = g.scalar("HexScale", 1.0, G_AT, 0.1, 8)
    p_hex_sharp = g.scalar("HexSharpness", 6.0, G_AT, 1, 16)

    b_pom = g.static("UsePOM", False, G_POM)
    p_pom_h = g.scalar("POMHeightCm", 1.0, G_POM, 0, 20)
    p_pom_min = g.scalar("POMMinSteps", 8, G_POM, 1, 64)
    p_pom_max = g.scalar("POMMaxSteps", 32, G_POM, 1, 128)
    if masked:
        b_clip = g.static("UseSilhouetteClip", False, G_SIL)
        p_clip_margin = g.scalar("SilhouetteClipMargin", 0.0, G_SIL, -0.1, 0.5)

    b_dirt = g.static("UseDirt", False, G_DIRT)
    p_ground = g.scalar("GroundZCm", 0.0, G_DIRT, -10000, 10000)
    p_dirt_h = g.scalar("DirtHeightCm", 100.0, G_DIRT, 1, 1000)
    p_dirt_fall = g.scalar("DirtFalloff", 1.5, G_DIRT, 0.1, 8)
    p_dirt_col = g.vector("DirtColor", (0.35, 0.28, 0.20), G_DIRT)
    p_dirt_str = g.scalar("DirtStrength", 0.8, G_DIRT, 0, 1)
    p_dirt_rough = g.scalar("DirtRoughnessAdd", 0.25, G_DIRT, 0, 1)
    p_dirt_nscale = g.scalar("DirtNoiseScaleCm", 120.0, G_DIRT, 5, 2000)
    p_dirt_namt = g.scalar("DirtNoiseAmount", 0.6, G_DIRT, 0, 1)

    b_edge = g.static("UseEdgeWear", False, G_EDGE)
    b_edge_vc = g.static("EdgeMaskFromVertexColor", False, G_EDGE)
    p_edge_ch = g.scalar("EdgeVertexChannel", 0, G_EDGE, 0, 3, "0=R 1=G 2=B 3=A")
    t_edge = g.texobj("EdgeMaskTex", "T_TF_Default_Mask", "SAMPLERTYPE_MASKS", G_EDGE, desc="baked wear mask, UV0, R channel")
    p_edge_thr = g.scalar("EdgeThreshold", 0.4, G_EDGE, 0, 1)
    p_edge_con = g.scalar("EdgeContrast", 6.0, G_EDGE, 0.1, 30)
    p_edge_brk = g.scalar("EdgeBreakup", 0.35, G_EDGE, 0, 1)
    p_edge_brk_t = g.scalar("EdgeBreakupTiling", 2.0, G_EDGE, 0.1, 16)
    t_ubase = g.texobj("Under_BaseColor", "T_TF_Default_BaseColor", "SAMPLERTYPE_COLOR", G_EDGE)
    t_unorm = g.texobj("Under_Normal", "T_TF_Default_Normal", "SAMPLERTYPE_NORMAL", G_EDGE)
    t_uorm = g.texobj("Under_ORM", "T_TF_Default_ORM", "SAMPLERTYPE_MASKS", G_EDGE)
    p_under_size = g.scalar("UnderWorldSizeCm", 100.0, G_EDGE, 1, 1000)

    # ================= UV =================
    g.sec("UV generation")
    cm_per_uv = g.div(p_world, p_uvscale, desc="CmPerUV = WorldSizeCm / UVScale")
    uv0 = g.texcoord(0)
    uv_mesh = g.mul(uv0, p_uvscale, desc="mesh UV0 * UVScale")
    wp = g.world_pos()
    vn = g.vertex_normal()
    uv_world = g.custom("World projected UV", HLSL_WORLD_UV,
                        [("WP", wp), ("N", vn), ("Axis", p_axis), ("CmPerUV", cm_per_uv)], out="f2")
    uv_sel = g.sw("UseWorldAlignedUV", uv_world, uv_mesh, desc="world-aligned or mesh UV")
    uvt_scale = g.mask(p_uvt, "rg")
    uvt_off = g.append((p_uvt, "B"), (p_uvt, "A"))      # 5.8: the VectorParameter default output is float3 (rgb): a "ba" ComponentMask on it failed to compile
    uv_base = g.add_(g.mul(uv_sel, uvt_scale), uvt_off, desc="UV_base = UV*UVTransform.rg + .ba")

    # ================= POM =================
    g.sec("Parallax occlusion mapping (UsePOM)")
    depth_uv = g.div(p_pom_h, cm_per_uv, desc="relief depth in UV units")
    cam = g.camera_vec()
    vt_mesh = g.to_tangent(cam, desc="camera vector -> tangent")
    vt_world = g.custom("View vector in projection frame", HLSL_VIEW_PROJ, [("Vw", cam), ("N", vn), ("Axis", p_axis)], out="f3")
    vt = g.sw("UseWorldAlignedUV", vt_world, vt_mesh, desc="view vector frame")
    pom_uv = None
    if pom_impl == "engine":
        fn = g.fcall(_cfg("ENGINE_POM_FUNCTION"), "Engine POM function")
        if fn is not None:
            # VERIFY: pin names taken from `strings` on ParallaxOcclusionMapping.uasset; types/semantics of
            # 'Height Ratio' and 'Heightmap Channel' unverified. Engine POM works in MESH tangent space only,
            # so combine with UseWorldAlignedUV = False.
            link(uv_base, fn, "UVs", "enginePOM.UVs")
            link(t_hgt, fn, ["Heightmap Texture"], "enginePOM.Heightmap Texture")
            link(g.c4(1, 0, 0, 0), fn, ["Heightmap Channel"], "enginePOM.Heightmap Channel")
            link(depth_uv, fn, ["Height Ratio"], "enginePOM.Height Ratio")
            link(p_pom_min, fn, ["Min Steps"], "enginePOM.Min Steps")
            link(p_pom_max, fn, ["Max Steps"], "enginePOM.Max Steps")
            pom_uv = (fn, "Parallax UVs")
    if pom_uv is None:
        if pom_impl == "engine":
            _warn("falling back to custom POM because the engine function could not be set up")
        pom_uv = g.custom("POM ray march (custom)", HLSL_POM,
                          [("UV", uv_base), ("Vt", vt), ("HeightTex", t_hgt), ("DepthUV", depth_uv),
                           ("MinSteps", p_pom_min), ("MaxSteps", p_pom_max)], out="f2")
    uv_main = g.sw("UsePOM", pom_uv, uv_base, desc="UV used by ALL texture samples")

    # ================= base samples =================
    g.sec("Albedo / Normal / ORM sampling (plain | two-scale | hex)")
    extra = {"CmPerUV": cm_per_uv, "MaskScaleCm": p_two_mask, "Ratio": p_two_ratio, "RotDeg": p_two_rot,
             "Strength": p_two_str, "HexScale": p_hex_scale, "Sharp": p_hex_sharp}

    def sampled(label, kind, tex):
        plain = g.sample(label, kind, "plain", tex, uv_main)
        two = g.sample(label, kind, "two", tex, uv_main, extra)
        hexn = g.sample(label, kind, "hex", tex, uv_main, extra)
        return g.sw("UseHexTiling", hexn, g.sw("UseTwoScaleBlend", two, plain, desc="two-scale?"), desc=label + " mode")

    c0 = sampled("BaseColor", "color", t_base)
    n0 = sampled("Normal", "normal", t_norm)
    o0 = sampled("ORM", "orm", t_orm)

    # ================= macro variation =================
    g.sec("Anti-tiling A: macro variation (UseMacroVariation)")
    c_macro = g.custom("Macro brightness noise", HLSL_MACRO,
                       [("C", c0), ("UV", uv_main), ("CmPerUV", cm_per_uv), ("MacroScaleCm", p_macro_scale),
                        ("MacroStrength", p_macro_str)], out="f3")
    c1 = g.sw("UseMacroVariation", c_macro, c0, desc="macro on/off")

    # ================= edge wear =================
    g.sec("Edge wear -> under material (UseEdgeWear)")
    uv_under = g.mul(uv_main, g.div(cm_per_uv, p_under_size), desc="UV for Under_* (own world size)")
    vc = g.vertex_color()
    vc_names = _out_names(vc) or []
    chan = {}
    for ch in ("R", "G", "B", "A"):
        cand = [n for n in vc_names if n.upper() == ch]
        chan[ch] = (vc, cand[0]) if cand else None
    if any(v is None for v in chan.values()):
        _fail("VertexColor node lacks R/G/B/A named outputs (outputs: %s); EdgeMaskFromVertexColor will not work" % vc_names)
        zero = g.c1(0.0)
        chan = dict((k, v if v is not None else zero) for k, v in chan.items())
    vc_pick = g.custom("Vertex colour channel", HLSL_VC_PICK,
                       [("VCR", chan["R"]), ("VCG", chan["G"]), ("VCB", chan["B"]), ("VCA", chan["A"]), ("Channel", p_edge_ch)], out="f1")
    edge_tex_r = g.sample("EdgeMaskTex", "scalar", "plain", t_edge, uv0)
    wear = g.sw("EdgeMaskFromVertexColor", vc_pick, edge_tex_r, desc="wear source")
    hgt_brk = g.sample("Height breakup", "scalar", "plain", t_hgt, g.mul(uv_main, p_edge_brk_t))
    m_edge = g.custom("Edge mask", HLSL_EDGE_MASK,
                      [("Wear", wear), ("Breakup01", hgt_brk), ("BreakupAmount", p_edge_brk), ("Threshold", p_edge_thr),
                       ("Contrast", p_edge_con)], out="f1")
    uc = g.sample("Under_BaseColor", "color", "plain", t_ubase, uv_under)
    un = g.sample("Under_Normal", "normal", "plain", t_unorm, uv_under)
    uo = g.sample("Under_ORM", "orm", "plain", t_uorm, uv_under)
    c_edge = g.lerp(c1, uc, m_edge, desc="BaseColor -> under")
    o_edge = g.lerp(o0, uo, m_edge, desc="ORM -> under")
    n_edge = g.custom("Whiteout normal blend (base/under)", HLSL_EDGE_NORMAL, [("Nb", n0), ("Nu", un), ("M", m_edge)], out="f3")
    c2 = g.sw("UseEdgeWear", c_edge, c1, desc="edge wear colour")
    n2 = g.sw("UseEdgeWear", n_edge, n0, desc="edge wear normal")
    o2 = g.sw("UseEdgeWear", o_edge, o0, desc="edge wear ORM")

    # ================= dirt =================
    g.sec("Dirt gradient (UseDirt)")
    rough2 = g.mask(o2, "g")
    dirt = g.custom("Dirt", HLSL_DIRT,
                    [("C", c2), ("Rough", rough2), ("WP", wp), ("GroundZ", p_ground), ("DirtHeightCm", p_dirt_h),
                     ("DirtFalloff", p_dirt_fall), ("DirtColor", g.mask(p_dirt_col, "rgb")), ("DirtStrength", p_dirt_str),
                     ("DirtRoughnessAdd", p_dirt_rough), ("DirtNoiseScaleCm", p_dirt_nscale), ("DirtNoiseAmount", p_dirt_namt)],
                    out="f4")
    c3_ = g.sw("UseDirt", g.mask(dirt, "rgb"), c2, desc="dirt colour")
    r3 = g.sw("UseDirt", g.mask(dirt, "a"), rough2, desc="dirt roughness")

    # ================= outputs =================
    g.sec("Outputs")
    base_color = g.mul(g.mul(c3_, g.mask(p_tint, "rgb")), p_bright, desc="BaseColor out")
    roughness = g.sat(g.mul(r3, p_rough))
    metallic = g.sat(g.mul(g.mask(o2, "b"), p_metal))
    ao = g.lerp(1.0, g.mask(o2, "r"), p_ao, desc="AO out")
    n_scaled = g.custom("Normal strength", HLSL_NORMAL_STRENGTH, [("Nt", n2), ("Strength", p_nstr)], out="f3")
    n_world = g.custom("Projection normal -> world", HLSL_NORMAL_TO_WORLD, [("Nt", n_scaled), ("N", vn), ("Axis", p_axis)], out="f3")
    n_world_tan = g.to_tangent(n_world, desc="world normal -> tangent")
    normal_out = g.sw("UseWorldAlignedUV", n_world_tan, n_scaled, desc="final tangent-space normal")

    connect_prop(base_color, "MP_BASE_COLOR")
    connect_prop(normal_out, "MP_NORMAL")
    connect_prop(roughness, "MP_ROUGHNESS")
    connect_prop(metallic, "MP_METALLIC")
    connect_prop(ao, "MP_AMBIENT_OCCLUSION")

    if masked:
        one = g.c1(1.0)
        clip = g.custom("Silhouette clip (UV island test)", HLSL_CLIP,
                        [("UVBase", uv_base), ("UVHit", uv_main), ("UV0", uv0), ("UVScale", p_uvscale),
                         ("UVTscale", uvt_scale), ("Margin", p_clip_margin)], out="f1")
        # clip only meaningful when POM is on AND UV is mesh UV0 (world-aligned UVs have no [0,1] island)
        m1 = g.sw("UseWorldAlignedUV", one, clip, desc="world UV -> no clip")
        m2 = g.sw("UseSilhouetteClip", m1, one, desc="clip on/off")
        m3 = g.sw("UsePOM", m2, one, desc="needs POM")
        connect_prop(m3, "MP_OPACITY_MASK")

    _finish_material(g, mat)
    return mat


# --------------------------------------------------------------------------------------------------
# M_TexforgeDecal
# --------------------------------------------------------------------------------------------------
def build_decal(defaults):
    name = "M_TexforgeDecal"
    CURRENT_MATERIAL[0] = name
    mat = _new_material(name, _cfg("MASTER_DIR"))
    _set(mat, "material_domain", _enum("MaterialDomain", "MD_DEFERRED_DECAL"))
    _set(mat, "blend_mode", _enum("BlendMode", "BLEND_TRANSLUCENT"))
    _soft(mat, "shading_model", _enum("MaterialShadingModel", "MSM_DEFAULT_LIT"))  # VERIFY: may be read-only in 5.8 (already the default)
    # VERIFY: decal_blend_mode exists on UMaterial in 5.8 (DBM_TRANSLUCENT = write base color/normal/roughness/metallic with opacity)
    dbm = _enum("DecalBlendMode", "DBM_TRANSLUCENT")
    if dbm is not None:
        _soft(mat, "decal_blend_mode", dbm)
    _set(mat, "tangent_space_normal", True)

    g = Graph(mat)
    g.defaults = defaults
    g.sec("Parameters")
    t_base = g.texobj("BaseColor", "T_TF_Default_BaseColor", "SAMPLERTYPE_COLOR", "00 Decal")
    t_norm = g.texobj("Normal", "T_TF_Default_Normal", "SAMPLERTYPE_NORMAL", "00 Decal")
    t_orm = g.texobj("ORM", "T_TF_Default_ORM", "SAMPLERTYPE_MASKS", "00 Decal")
    t_op = g.texobj("Opacity", "T_TF_Default_Opacity", "SAMPLERTYPE_MASKS", "00 Decal", desc="linear opacity mask (R)")
    p_tint = g.vector("BaseColorTint", (1.0, 1.0, 1.0), "00 Decal")
    p_op = g.scalar("OpacityMul", 1.0, "00 Decal", 0, 1)
    p_rough = g.scalar("RoughnessMul", 1.0, "00 Decal", 0, 4)
    p_nstr = g.scalar("NormalStrength", 1.0, "00 Decal", 0, 4)
    p_metal = g.scalar("MetallicMul", 1.0, "00 Decal", 0, 1)

    g.sec("Sampling")
    uv = g.texcoord(0)   # decal projection UV (0..1 across the decal box face)
    c = g.sample("BaseColor", "color", "plain", t_base, uv)
    n = g.sample("Normal", "normal", "plain", t_norm, uv)
    o = g.sample("ORM", "orm", "plain", t_orm, uv)
    op = g.sample("Opacity", "scalar", "plain", t_op, uv)

    g.sec("Outputs")
    base = g.mul(c, g.mask(p_tint, "rgb"))
    rough = g.sat(g.mul(g.mask(o, "g"), p_rough))
    metal = g.sat(g.mul(g.mask(o, "b"), p_metal))
    nrm = g.custom("Normal strength", HLSL_NORMAL_STRENGTH, [("Nt", n), ("Strength", p_nstr)], out="f3")
    opacity = g.sat(g.mul(op, p_op))
    connect_prop(base, "MP_BASE_COLOR")
    connect_prop(nrm, "MP_NORMAL")
    connect_prop(rough, "MP_ROUGHNESS")
    connect_prop(metal, "MP_METALLIC")
    connect_prop(opacity, "MP_OPACITY")
    _finish_material(g, mat)
    return mat


# --------------------------------------------------------------------------------------------------
# M_TexforgeWindowRoom   (lower priority - first implementation)
# --------------------------------------------------------------------------------------------------
def build_window_room(defaults):
    name = "M_TexforgeWindowRoom"
    CURRENT_MATERIAL[0] = name
    mat = _new_material(name, _cfg("MASTER_DIR"))
    _set(mat, "material_domain", _enum("MaterialDomain", "MD_SURFACE"))
    _set(mat, "blend_mode", _enum("BlendMode", "BLEND_OPAQUE"))
    _soft(mat, "shading_model", _enum("MaterialShadingModel", "MSM_DEFAULT_LIT"))  # VERIFY: may be read-only in 5.8 (already the default)
    _set(mat, "tangent_space_normal", True)

    g = Graph(mat)
    g.defaults = defaults
    G = "00 Room"
    g.sec("Parameters")
    t_atlas = g.texobj("RoomAtlas", "T_TF_Default_BaseColor", "SAMPLERTYPE_COLOR", G, desc="3x2 atlas, layout = ATLAS_CELLS")
    b_cube = g.static("UseCubemap", False, G)
    p_tx = g.scalar("RoomTilesX", 1.0, G, 1, 32)
    p_ty = g.scalar("RoomTilesY", 1.0, G, 1, 32)
    p_depth = g.scalar("RoomDepthRatio", 1.0, G, 0.1, 5, "room depth / room width")
    p_aspect = g.scalar("WindowAspect", 1.0, G, 0.2, 5, "window width / height")
    p_pad = g.scalar("AtlasPadUV", 0.003, G, 0, 0.05)
    p_var = g.scalar("RoomLightVariation", 0.3, G, 0, 1)
    p_bright = g.scalar("RoomBrightness", 1.0, G, 0, 8)
    p_tint = g.vector("RoomTint", (1.0, 1.0, 1.0), G)
    p_glass = g.vector("GlassTint", (0.02, 0.03, 0.035), G)
    p_rough = g.scalar("GlassRoughness", 0.05, G, 0, 1)
    p_spec = g.scalar("GlassSpecular", 0.5, G, 0, 1)

    g.sec("Interior mapping - atlas box projection (custom HLSL)")
    uv0 = g.texcoord(0)
    tiles = g.append(p_tx, p_ty)
    vt = g.to_tangent(g.camera_vec(), desc="camera -> tangent")
    room_atlas = g.custom("Room box projection (atlas)", hlsl_room(),
                          [("UV", uv0), ("Tiles", tiles), ("Vt", vt), ("Atlas", t_atlas), ("DepthRatio", p_depth),
                           ("Aspect", p_aspect), ("Pad", p_pad), ("Variation", p_var)], out="f3")

    g.sec("Interior mapping - engine InteriorCubemap + TextureCube")
    room_cube = None
    cube = None
    for cand in _cfg("DEFAULT_CUBE_CANDIDATES").split(","):
        if EAL.does_asset_exist(cand.strip()):
            cube = EAL.load_asset(cand.strip())
            break
    fn = g.fcall(_cfg("ENGINE_INTERIOR_CUBEMAP_FUNCTION"), "Engine InteriorCubemap")
    cls_cube = getattr(unreal, "MaterialExpressionTextureSampleParameterCube", None)
    if fn is not None and cube is not None and cls_cube is not None:
        # VERIFY: input names "UVs"/"Tiling" and output "Result" come from `strings` on InteriorCubemap.uasset
        link(uv0, fn, "UVs", "InteriorCubemap.UVs")
        link(tiles, fn, "Tiling", "InteriorCubemap.Tiling")
        sample = g.add("TextureSampleParameterCube", 4, h=140, parameter_name="RoomCube", texture=cube, group=G, desc="RoomCube (TextureCube)")
        PARAMS["texture"].append((name, "RoomCube", getattr(cube, "get_name", lambda: "?")()))
        link((fn, "Result"), sample, ["Coordinates"], "RoomCube.Coordinates")
        room_cube = (sample, "")
    else:
        _warn("cubemap path disabled (function=%s default cube=%s class=%s)" % (fn is not None, cube is not None, cls_cube is not None))
    if room_cube is None:
        room_cube = room_atlas   # keep the switch valid; UseCubemap just selects the same thing

    g.sec("Outputs")
    room = g.sw("UseCubemap", room_cube, room_atlas, desc="room source")
    emissive = g.mul(g.mul(room, g.mask(p_tint, "rgb")), p_bright, desc="Emissive = room image")
    connect_prop(emissive, "MP_EMISSIVE_COLOR")
    connect_prop(g.mask(p_glass, "rgb"), "MP_BASE_COLOR")
    connect_prop(p_rough, "MP_ROUGHNESS")
    connect_prop(p_spec, "MP_SPECULAR")
    connect_prop(g.c1(0.0), "MP_METALLIC")
    _finish_material(g, mat)
    return mat


# --------------------------------------------------------------------------------------------------
def main():
    if MEL is None or EAL is None:
        raise RuntimeError("This script must run inside the Unreal Editor (unreal.MaterialEditingLibrary missing).")
    todo = [s.strip() for s in _cfg("BUILD").split(",") if s.strip()]
    _info("building: %s   (POM_IMPL=%s)" % (todo, _cfg("POM_IMPL")))
    defaults = ensure_default_textures()
    built = []
    for what, fn in (("master", lambda: build_surface_master("M_TexforgeMaster", False, defaults)),
                     ("master_masked", lambda: build_surface_master("M_TexforgeMaster_Masked", True, defaults)),
                     ("decal", lambda: build_decal(defaults)),
                     ("window_room", lambda: build_window_room(defaults))):
        if what not in todo:
            continue
        try:
            m = fn()
            built.append(m.get_path_name())
        except Exception as exc:  # keep going with the next material
            import traceback
            _fail("building '%s' aborted: %s\n%s" % (what, exc, traceback.format_exc()))

    _info("==== SUMMARY ====")
    for p in built:
        _info("built " + p)
    for kind in ("scalar", "vector", "static", "texture"):
        for mat_name, pname, default in PARAMS[kind]:
            _info("param %-8s %-26s %-28s default=%s" % (kind, mat_name, pname, default))
    _info("%d failure(s), %d warning(s)" % (len(FAILS), len(WARNS)))
    for f in FAILS:
        unreal.log_error("[texforge] FAIL " + f)
    for w in WARNS:
        unreal.log_warning("[texforge] WARN " + w)
    if not FAILS:
        _info("No wiring failures reported. STILL open each material and check the Output Log for HLSL compile errors "
              "(Custom node bodies are only compiled by the shader compiler, not by this script).")


main()
