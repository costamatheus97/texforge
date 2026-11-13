"""Prompt templates for AI image generation (swatches, trim sheets, decals, rooms)."""

_COMMON = (
    "Orthographic, perfectly straight-on view with absolutely no perspective and no camera tilt. "
    "Flat, even, diffuse lighting from everywhere: NO cast shadows, NO directional light, NO highlights, "
    "NO vignette, NO lens effects, NO depth of field, NO watermark, NO text, no borders or frame. "
    "The material fills the entire frame edge to edge. Physically plausible, photoreal PBR albedo-style "
    "photograph scan of the material with fine natural detail, sharp focus everywhere, high resolution."
)


def material_prompt(subject: str) -> str:
    return (
        f"A square, flat, top-down / straight-on texture swatch of {subject}. {_COMMON} "
        "Keep colour variation natural but avoid any single dominant landmark, large stain or gradient "
        "(it will be tiled). Keep the pattern axis-aligned (horizontal courses / rows, vertical joints) if it has one."
    )


def trim_prompt(strips: list[str]) -> str:
    n = len(strips)
    listing = "; ".join(f"{i + 1}. {s}" for i, s in enumerate(strips))
    return (
        f"A square architectural TRIM SHEET texture atlas: exactly {n} horizontal strips stacked vertically, "
        f"each strip occupying an equal share ({100 // n}%) of the image height and the full image width, "
        "separated only by a thin neutral grey gap of about 3% of the strip height. Each strip is a long "
        "straight band that continues the same profile/pattern uniformly left to right so it can repeat "
        f"horizontally, from top to bottom: {listing}. {_COMMON} "
        "Strips are viewed face-on (elevation view), as if a flat orthographic drawing of the moulding or "
        "material profile; no strip should have a different lighting direction."
    )


def decal_prompt(subject: str, background: str = "white") -> str:
    return (
        f"A single {subject} drawn as an isolated surface decal on a perfectly plain, uniform, flat pure "
        f"{background} background (no gradient, no texture, no shadows, no floor). Straight-on orthographic "
        "view, flat lighting, no perspective, no text, no border. The mark is centred and fully contained with "
        "generous empty margin all around it (nothing touches the image edge), its edges fade naturally and "
        "softly into the background. Photoreal, high resolution, physically plausible."
    )


def room_prompt(subject: str) -> str:
    return (
        f"A single image of the interior of {subject}, seen from exactly the centre of the open front wall, "
        "looking straight at the back wall with strict one-point perspective: the back wall is a perfectly "
        "frontal rectangle centred in the image occupying the middle 50% of width and height, with the floor, "
        "ceiling and two side walls converging toward it. Wide angle of view, no people, no window glass, "
        "no visible camera, no text. Evenly lit, soft interior lighting, high resolution, photoreal."
    )


# --- forest-floor kit: one shared capture style so every layer sits at the same scale / exposure -------------------

KIT_STYLE = (
    "Photographed from directly above (orthographic top-down, camera perfectly perpendicular to the ground), "
    "under flat overcast daylight: soft, even, shadowless, neutral white balance, no sun patches, no cast shadows, "
    "no dark crevice shadows baked in, no vignette, no depth of field, sharp everywhere. The frame shows a patch of "
    "ground exactly 2 metres x 2 metres, filling the whole frame edge to edge. Natural PBR-albedo exposure: not too "
    "dark, no crushed blacks, no blown highlights. No landmark, no large single object, no gradient or brightness "
    "falloff across the frame (it will be tiled). No text, no watermark, no border."
)


def kit_prompt(subject: str, scale: str = "", style: str = KIT_STYLE) -> str:
    return f"A square photoreal texture scan of {subject}. {scale} {style}".replace("  ", " ")


def atlas_prompt(rows: list[tuple[str, int, str]], key: str = "magenta") -> str:
    """rows: [(kind, count, description)] -> a sheet of isolated objects on a flat chroma-key background."""
    listing = " ".join(f"Row {i + 1}: {n} different {d}." for i, (_k, n, d) in enumerate(rows))
    return (
        f"A flat sprite sheet of isolated forest-floor litter objects laid out in {len(rows)} horizontal rows on a "
        f"perfectly uniform, flat, pure saturated {key} background (RGB 255, 0, 255), with no gradient, no texture, "
        f"no floor, no cast shadows and no ground contact shadows. {listing} Every object is a separate, fully "
        "visible item, photographed from directly above, evenly lit by flat overcast light, drawn at the SAME real-world "
        "scale (leaves about 8 to 12 cm, twigs 10 to 20 cm, pinecones 8 to 10 cm, bark chips 4 to 8 cm), with "
        "wide empty background margin between all objects; nothing overlaps, nothing touches another object or the "
        "image edge. No object contains any magenta or purple. Photoreal, high resolution, sharp, no text."
    )
