"""
texforge -> Unreal Engine 5.8 : texture set importer + material instance builder  (editor Python, UNTESTED)
==========================================================================================================

Imports one texforge texture set (PNGs + <name>.texforge.json) into the project, applies the correct texture
settings, and creates/updates a Material Instance Constant `MI_<name>` parented to the right texforge master.
Run create_master_material.py FIRST (once) so /Game/Texforge/M_Texforge* exist.

HOW TO RUN (inside the editor):
    1. edit CONFIG below (SOURCE_DIR, GAME_PATH, ...)  -- or set environment variables TEXFORGE_<KEY>
       before launching the editor (TEXFORGE_SOURCE_DIR, TEXFORGE_GAME_PATH, TEXFORGE_FEATURES, ...)
    2. Output Log -> Python console:
           py "/path/to/texforge/integrations/unreal/unreal_import.py"
       (or Tools > Execute Python Script...). Editor Python has no argv, hence CONFIG/env.
    Re-running replaces the textures (replace_existing) and updates the SAME MI in place (references survive).

WHAT IT DOES
    kind "tile"  -> M_TexforgeMaster, UseWorldAlignedUV=True,  WorldSizeCm=world_cm[0], wrap/wrap
    kind "trim"  -> M_TexforgeMaster, UseWorldAlignedUV=False (mesh UVs), WorldSizeCm=world_cm[0],
                    wrap U / clamp V, strips table printed and stored as asset metadata tags
    kind "decal" -> M_TexforgeDecal (textures clamp)
    kind "room"  -> M_TexforgeWindowRoom (textures clamp)
    Texture settings:  BaseColor TC_DEFAULT sRGB | Normal TC_NORMALMAP linear, flip_green_channel False
                       ORM TC_MASKS linear | Height TC_GRAYSCALE linear | Opacity TC_MASKS linear
                       Room / RoomAtlas TC_DEFAULT sRGB.   (the master's sampler types / default textures match these)
    Feature switches (static) default to OFF; enable via CONFIG["FEATURES"] (comma list of switch names).
"""

import os
import glob
import json

import unreal

# --------------------------------------------------------------------------------------------------
CONFIG = {
    "SOURCE_DIR": "",                      # folder holding <name>.texforge.json + PNGs, a parent of such a folder, or the .json itself
    "GAME_PATH": "",                       # e.g. "/Game/Textures/Brick_A"; empty -> /Game/Textures/<name>
    "MASTER_MATERIAL": "/Game/Texforge/M_TexforgeMaster",
    "MASKED_MASTER_MATERIAL": "/Game/Texforge/M_TexforgeMaster_Masked",   # used when UseSilhouetteClip is enabled
    "DECAL_MATERIAL": "/Game/Texforge/M_TexforgeDecal",
    "ROOM_MATERIAL": "/Game/Texforge/M_TexforgeWindowRoom",
    # comma list of static switches to turn ON. Valid: UseMacroVariation, UseTwoScaleBlend, UseHexTiling, UsePOM,
    # UseSilhouetteClip (needs UsePOM + trim/mesh-UV), UseDirt, UseEdgeWear, EdgeMaskFromVertexColor, UseCubemap
    "FEATURES": "",
    # for trims: name of a strip from the sidecar to map onto the whole 0..1 mesh V range (sets UVTransform), "" = whole sheet
    "TRIM_STRIP": "",
    # extra parameter overrides applied last. scalars: {"GroundZCm": 0.0}; vectors: {"BaseColorTint": (1, 1, 1, 1)}
    "SCALAR_OVERRIDES": {},
    "VECTOR_OVERRIDES": {},
    "SAVE": "1",
}


def _cfg(key):
    v = os.environ.get("TEXFORGE_" + key)
    return CONFIG[key] if v is None else v


# key -> (texture suffix, compression, srgb, material parameter name or None)
TEX_SPECS = {
    "basecolor": ("BaseColor", "TC_DEFAULT", True, "BaseColor"),
    "normal": ("Normal", "TC_NORMALMAP", False, "Normal"),
    "orm": ("ORM", "TC_MASKS", False, "ORM"),
    "height": ("Height", "TC_GRAYSCALE", False, "Height"),
    "opacity": ("Opacity", "TC_MASKS", False, "Opacity"),
    "room": ("Room", "TC_DEFAULT", True, None),          # single room image: imported, not bound (atlas is what the shader uses)
    "roomatlas": ("RoomAtlas", "TC_DEFAULT", True, "RoomAtlas"),
}
KIND_KEYS = {
    "tile": ["basecolor", "normal", "orm", "height"],
    "trim": ["basecolor", "normal", "orm", "height"],
    "decal": ["basecolor", "normal", "orm", "height", "opacity"],
    "room": ["room", "roomatlas"],
}
PATTERNED_TYPES = ("brick", "tile", "cobble", "stone", "block", "plank", "wood", "shingle")

PROBLEMS = []
CREATED = []

MEL = getattr(unreal, "MaterialEditingLibrary", None)
EAL = getattr(unreal, "EditorAssetLibrary", None)


def _info(m):
    unreal.log("[texforge-import] " + str(m))


def _warn(m):
    PROBLEMS.append("WARN " + str(m))
    unreal.log_warning("[texforge-import] WARN " + str(m))


def _fail(m):
    PROBLEMS.append("FAIL " + str(m))
    unreal.log_error("[texforge-import] FAIL " + str(m))


def _set(obj, prop, value):
    try:
        obj.set_editor_property(prop, value)
        return True
    except Exception as exc:  # noqa
        _fail("set_editor_property(%s.%s) failed: %s" % (type(obj).__name__, prop, exc))
        return False


def _enum(cls_name, member):
    cls = getattr(unreal, cls_name, None)
    if cls is not None:
        for cand in (member, member.upper()):
            if hasattr(cls, cand):
                return getattr(cls, cand)
    _fail("enum unreal.%s.%s not found" % (cls_name, member))
    return None


# --------------------------------------------------------------------------------------------------
# sidecar discovery
# --------------------------------------------------------------------------------------------------
def find_sidecar(src):
    src = src.replace("\\", "/").rstrip("/")
    if not src:
        raise RuntimeError("SOURCE_DIR is empty: edit CONFIG['SOURCE_DIR'] or set TEXFORGE_SOURCE_DIR")
    if src.lower().endswith(".json"):
        return src
    found = sorted(glob.glob(src + "/*.texforge.json"))
    if not found:
        found = sorted(glob.glob(src + "/*/*.texforge.json"))
    if not found:
        raise RuntimeError("no *.texforge.json found in %s (or its sub-folders)" % src)
    if len(found) > 1:
        raise RuntimeError("several texture sets under %s; point SOURCE_DIR at one of: %s" % (src, found))
    return found[0].replace("\\", "/")


def resolve_files(meta, folder, name, kind):
    """Return {key: absolute png path} for the keys this kind needs (missing optional ones skipped with a warning)."""
    files = dict((k.lower().replace("_", ""), v) for k, v in (meta.get("files") or {}).items())
    out = {}
    for key in KIND_KEYS[kind]:
        suffix = TEX_SPECS[key][0]
        cands = []
        if key == "roomatlas":
            for alt in ("roomatlas", "atlas"):
                if alt in files:
                    cands.append(files[alt])
        elif key in files:
            cands.append(files[key])
        cands.append("%s_%s.png" % (name, suffix))
        path = None
        for c in cands:
            p = c if os.path.isabs(c) else os.path.join(folder, c)
            if os.path.exists(p):
                path = p.replace("\\", "/")
                break
        if path:
            out[key] = path
        else:
            required = key in ("basecolor", "normal", "orm") or (kind == "room" and key == "roomatlas")
            (_fail if required else _warn)("texture '%s' for set %s not found (looked for %s)" % (key, name, cands))
    return out


# --------------------------------------------------------------------------------------------------
# texture import + settings
# --------------------------------------------------------------------------------------------------
def import_textures(paths, game_path, name, kind, tile_axes):
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    EAL.make_directory(game_path)
    tasks = []
    for key, png in paths.items():
        task = unreal.AssetImportTask()
        tex_name = "T_%s_%s" % (name, TEX_SPECS[key][0])
        for k, v in (("filename", png), ("destination_path", game_path), ("destination_name", tex_name),
                     ("automated", True), ("save", True), ("replace_existing", True)):
            _set(task, k, v)
        tasks.append(task)
    tools.import_asset_tasks(tasks)

    wrap = _enum("TextureAddress", "TA_WRAP")
    clamp = _enum("TextureAddress", "TA_CLAMP")
    if kind in ("decal", "room"):
        ax, ay = clamp, clamp
    elif kind == "trim" or tile_axes == "x":
        ax, ay = wrap, clamp          # trims tile horizontally only
    else:
        ax, ay = wrap, wrap

    textures = {}
    for key in paths:
        suffix, comp, srgb, _ = TEX_SPECS[key]
        asset_path = "%s/T_%s_%s" % (game_path, name, suffix)
        tex = EAL.load_asset(asset_path)
        if tex is None:
            _fail("import failed or asset not loadable: %s (source %s)" % (asset_path, paths[key]))
            continue
        c = _enum("TextureCompressionSettings", comp)
        if c is not None:
            _set(tex, "compression_settings", c)
        _set(tex, "srgb", srgb)
        if key == "normal":
            _set(tex, "flip_green_channel", False)   # texforge writes DirectX/Unreal green-down: no flip
        if ax is not None:
            _set(tex, "address_x", ax)
        if ay is not None:
            _set(tex, "address_y", ay)
        # mip_gen_settings left at the engine default (FromTextureGroup)
        try:
            EAL.save_loaded_asset(tex)
        except Exception as exc:
            _warn("could not save %s: %s" % (asset_path, exc))
        textures[key] = tex
        CREATED.append("%s  (%s, sRGB=%s)" % (asset_path, comp, srgb))
    return textures


# --------------------------------------------------------------------------------------------------
# material instance
# --------------------------------------------------------------------------------------------------
def _pnames(fn, mat):
    try:
        return set(str(n) for n in fn(mat))
    except Exception as exc:
        _warn("%s failed (%s); parameter validation skipped" % (getattr(fn, "__name__", "param-names"), exc))
        return None


class MIWriter(object):
    def __init__(self, mi, master):
        self.mi = mi
        self.names = {
            "scalar": _pnames(MEL.get_scalar_parameter_names, master),
            "vector": _pnames(MEL.get_vector_parameter_names, master),
            "texture": _pnames(MEL.get_texture_parameter_names, master),
            "static": _pnames(MEL.get_static_switch_parameter_names, master),
        }
        self.log = []

    def _known(self, kind, pname):
        names = self.names[kind]
        if names is not None and pname not in names:
            _warn("parent has no %s parameter '%s' (rebuild the master?)" % (kind, pname))
            return False
        return True

    def scalar(self, pname, value):
        if self._known("scalar", pname):
            ok = MEL.set_material_instance_scalar_parameter_value(self.mi, pname, float(value))
            (self.log.append("scalar  %-22s = %s" % (pname, value))) if ok is not False else _warn("set scalar %s returned False" % pname)

    def vector(self, pname, rgba):
        if self._known("vector", pname):
            rgba = tuple(rgba) + (1.0,) * (4 - len(rgba))
            ok = MEL.set_material_instance_vector_parameter_value(self.mi, pname, unreal.LinearColor(*rgba))
            (self.log.append("vector  %-22s = %s" % (pname, rgba))) if ok is not False else _warn("set vector %s returned False" % pname)

    def texture(self, pname, tex):
        if self._known("texture", pname):
            ok = MEL.set_material_instance_texture_parameter_value(self.mi, pname, tex)
            (self.log.append("texture %-22s = %s" % (pname, tex.get_name()))) if ok is not False else _warn("set texture %s returned False" % pname)

    def static(self, pname, value):
        # symbol SetMaterialInstanceStaticSwitchParameterValue is present in the 5.8 MaterialEditor binary (checked with `strings`);
        # VERIFY at runtime: it should return True and the switch should show checked in the MI details panel.
        if self._known("static", pname):
            try:
                ok = MEL.set_material_instance_static_switch_parameter_value(self.mi, pname, bool(value))
            except Exception as exc:
                _fail("set_material_instance_static_switch_parameter_value(%s) raised %s -> toggle it by hand in the MI "
                      "(Static Switch Parameters section)" % (pname, exc))
                return
            (self.log.append("static  %-22s = %s" % (pname, bool(value)))) if ok is not False else _warn("set static switch %s returned False" % pname)


def get_or_create_mi(game_path, mi_name, master):
    path = "%s/%s" % (game_path, mi_name)
    mi = None
    if EAL.does_asset_exist(path):
        mi = EAL.load_asset(path)
        if mi is not None and not isinstance(mi, unreal.MaterialInstanceConstant):
            EAL.delete_asset(path)
            mi = None
    if mi is None:
        tools = unreal.AssetToolsHelpers.get_asset_tools()
        mi = tools.create_asset(mi_name, game_path, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
        if mi is None:
            raise RuntimeError("create_asset failed for " + path)
    else:
        for prop in ("scalar_parameter_values", "vector_parameter_values", "texture_parameter_values"):
            try:   # clear stale overrides from a previous run; every parameter we want is re-set below
                mi.set_editor_property(prop, [])
            except Exception as exc:
                _warn("could not clear %s on existing MI (%s); old overrides may linger" % (prop, exc))
    try:
        MEL.set_material_instance_parent(mi, master)
    except Exception:
        _set(mi, "parent", master)
    return mi


def strips_report(meta, name):
    strips = meta.get("strips") or []
    if not strips:
        return ""
    size = meta.get("size") or [0, 0]
    lines = ["Trim strips of '%s' (v=0 is the TOP of the sheet; Unreal V also starts at the top)" % name,
             "%-14s %-8s %-8s %-8s %-10s %-10s %s" % ("strip", "v0", "v1", "px rows", "height_cm", "width_cm", "UVTransform for a 0..1 mesh island")]
    for s in strips:
        v0, v1 = float(s.get("v0", 0)), float(s.get("v1", 1))
        rows = int(round((v1 - v0) * size[1])) if size[1] else 0
        lines.append("%-14s %-8.4f %-8.4f %-8d %-10s %-10s (1, %.4f, 0, %.4f)" % (
            s.get("name", "?"), v0, v1, rows, s.get("height_cm", "?"), s.get("world_cm_width", "?"), v1 - v0, v0))
    lines.append("UV remap: texture_v = v0 + mesh_v * (v1 - v0); u unchanged. In the MI set vector param UVTransform = (1, v1-v0, 0, v0)")
    lines.append("   (set CONFIG['TRIM_STRIP'] to a strip name to have this importer do it), or remap the mesh UVs in the DCC.")
    lines.append("   Same thing for a mesh island [a,b] in V: texture_v = v0 + (mesh_v - a)/(b - a)*(v1 - v0).")
    return "\n".join(lines)


def main():
    if MEL is None or EAL is None:
        raise RuntimeError("must run inside the Unreal Editor")
    sidecar = find_sidecar(_cfg("SOURCE_DIR"))
    folder = os.path.dirname(sidecar)
    with open(sidecar, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    name = meta["name"]
    kind = meta.get("kind", "tile")
    if kind not in KIND_KEYS:
        raise RuntimeError("unsupported kind '%s'" % kind)
    wcm = meta.get("world_cm") or [100.0, 100.0]
    game_path = (_cfg("GAME_PATH") or "/Game/Textures/%s" % name).rstrip("/")
    if not game_path.startswith("/Game"):
        raise RuntimeError("GAME_PATH must start with /Game (got %s)" % game_path)
    features = set(s.strip() for s in _cfg("FEATURES").split(",") if s.strip())
    _info("set '%s' kind=%s type=%s -> %s  features=%s" % (name, kind, meta.get("type"), game_path, sorted(features) or "none"))

    # ---- choose master ----
    if kind == "decal":
        master_path = _cfg("DECAL_MATERIAL")
    elif kind == "room":
        master_path = _cfg("ROOM_MATERIAL")
    elif "UseSilhouetteClip" in features:
        master_path = _cfg("MASKED_MASTER_MATERIAL")
    else:
        master_path = _cfg("MASTER_MATERIAL")
    if not EAL.does_asset_exist(master_path):
        unreal.log_error("[texforge-import] master material %s does not exist. Run it FIRST in the editor:\n"
                         '    py "/path/to/texforge/integrations/unreal/create_master_material.py"\n'
                         "and then re-run this script." % master_path)
        return
    master = EAL.load_asset(master_path)

    # ---- textures ----
    paths = resolve_files(meta, folder, name, kind)
    textures = import_textures(paths, game_path, name, kind, meta.get("tile_axes", "xy"))

    # ---- MI ----
    mi_name = "MI_" + name
    mi = get_or_create_mi(game_path, mi_name, master)
    w = MIWriter(mi, master)
    for key, tex in textures.items():
        pname = TEX_SPECS[key][3]
        if pname is None or (kind == "decal" and pname == "Height"):
            _info("note: %s imported but not bound (M_TexforgeDecal / the room shader have no such parameter)" % tex.get_name())
            continue
        w.texture(pname, tex)

    aspect = (float(wcm[0]) / float(wcm[1])) if wcm and wcm[1] else 1.0
    if kind in ("tile", "trim"):
        w.scalar("WorldSizeCm", wcm[0])
        w.scalar("UVScale", 1.0)
        w.scalar("POMHeightCm", meta.get("height_depth_cm", 1.0))
        w.static("UseWorldAlignedUV", kind == "tile")
        uvt = (1.0, aspect if kind == "tile" and abs(aspect - 1.0) > 1e-3 else 1.0, 0.0, 0.0)
        if kind == "trim" and _cfg("TRIM_STRIP"):
            hit = [s for s in (meta.get("strips") or []) if s.get("name") == _cfg("TRIM_STRIP")]
            if hit:
                v0, v1 = float(hit[0]["v0"]), float(hit[0]["v1"])
                uvt = (1.0, v1 - v0, 0.0, v0)
                w.scalar("WorldSizeCm", float(hit[0].get("world_cm_width", wcm[0])))
            else:
                _warn("TRIM_STRIP '%s' not in sidecar strips" % _cfg("TRIM_STRIP"))
        w.vector("UVTransform", uvt)
        for sw in ("UseMacroVariation", "UseTwoScaleBlend", "UseHexTiling", "UsePOM", "UseDirt", "UseEdgeWear",
                   "EdgeMaskFromVertexColor") + (("UseSilhouetteClip",) if "UseSilhouetteClip" in features else ()):
            w.static(sw, sw in features)
        if "UseHexTiling" in features and (meta.get("type") or "").lower() in PATTERNED_TYPES:
            _warn("UseHexTiling on a patterned type ('%s') BREAKS joints/pattern; prefer UseTwoScaleBlend or UseMacroVariation" % meta.get("type"))
        if "UsePOM" in features and kind == "tile":
            _warn("UsePOM with world-aligned UVs works only if the master was built with POM_IMPL=custom (default); "
                  "with POM_IMPL=engine switch UseWorldAlignedUV off")
        if "UseSilhouetteClip" in features and (kind != "trim" or "UsePOM" not in features):
            _warn("UseSilhouetteClip only has an effect with UsePOM and mesh UVs (UseWorldAlignedUV off, island = 0..1)")
        if "UseEdgeWear" in features:
            _info("UseEdgeWear: also assign EdgeMaskTex, Under_BaseColor, Under_Normal, Under_ORM (and UnderWorldSizeCm) in the MI")
    elif kind == "decal":
        if "opacity" not in textures:
            _warn("decal has no Opacity texture; the default (fully opaque) will be used - BaseColor alpha is NOT read by M_TexforgeDecal")
    elif kind == "room":
        w.scalar("WindowAspect", aspect)
        room = meta.get("room") or {}
        w.scalar("RoomDepthRatio", room.get("depth_ratio", 1.0))
        w.static("UseCubemap", "UseCubemap" in features)
        cell_px = room.get("atlas_cell_px")
        if cell_px:   # inset one source texel per cell edge so bilinear/mips never bleed into the neighbouring cell
            w.scalar("AtlasPadUV", 1.0 / float(cell_px[0] if isinstance(cell_px, (list, tuple)) else cell_px))

    for k, v in (CONFIG.get("SCALAR_OVERRIDES") or {}).items():
        w.scalar(k, v)
    for k, v in (CONFIG.get("VECTOR_OVERRIDES") or {}).items():
        w.vector(k, v)

    # ---- metadata (a MaterialInstanceConstant has no description field -> asset registry metadata tags) ----
    report = strips_report(meta, name) if kind == "trim" else ""
    try:
        EAL.set_metadata_tag(mi, "TexforgeKind", kind)
        EAL.set_metadata_tag(mi, "TexforgeSidecar", json.dumps(meta, separators=(",", ":")))
        if meta.get("strips"):
            EAL.set_metadata_tag(mi, "TexforgeStrips", json.dumps(meta["strips"], separators=(",", ":")))
    except Exception as exc:
        _warn("set_metadata_tag failed (%s); strips only printed below" % exc)  # VERIFY: EditorAssetLibrary.set_metadata_tag

    try:
        MEL.update_material_instance(mi)
    except Exception as exc:
        _warn("update_material_instance failed: %s" % exc)
    if _cfg("SAVE") == "1":
        EAL.save_loaded_asset(mi)
    CREATED.append("%s/%s  (parent %s)" % (game_path, mi_name, master_path))

    # ---- summary ----
    _info("======== SUMMARY for '%s' ========" % name)
    for line in CREATED:
        _info("asset  " + line)
    for line in w.log:
        _info("  MI  " + line)
    if report:
        for line in report.splitlines():
            _info(line)
    if PROBLEMS:
        _info("%d problem(s):" % len(PROBLEMS))
        for p in PROBLEMS:
            _info("  " + p)
    else:
        _info("no problems reported")


main()
