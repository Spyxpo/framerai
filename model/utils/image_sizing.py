"""Aspect-ratio buckets and dimension snapping for image generation.

Image generation used to be square-only at a hardcoded 256 pixels. The latent
diffusion transformer computes its positional grid per call, so any resolution
and aspect ratio works; this module decides which ones a request may ask for.

Buckets are generated, not hand-listed. For a tier area ``A = tier ** 2`` and a
ratio ``r``, the bucket is ``round_to(sqrt(A * r))`` by ``round_to(sqrt(A / r))``,
so every aspect ratio costs roughly the same compute and the same batch shape.
"""

import math
import re

from ..configs.model_config import ASPECT_RATIOS as ASPECT_RATIO_NAMES

# Both dimensions must be a multiple of vae_downsample * dit_patch_size, which is
# 8 * 2 at the defaults. 16 keeps the latent grid divisible by the patch size.
DEFAULT_MULTIPLE = 16

# Every ratio the API accepts by name, as (width, height). The name list lives in
# model_config so validate() can check a configured default without importing
# torch; this is the geometry behind those names.
ASPECT_RATIOS = {
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "3:2": (3, 2),
    "2:3": (2, 3),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "21:9": (21, 9),
}

assert set(ASPECT_RATIOS) == set(ASPECT_RATIO_NAMES), (
    "model_config.ASPECT_RATIOS and image_sizing.ASPECT_RATIOS have drifted apart"
)

# Size tiers, in "square-equivalent" pixels per side.
SIZE_TIERS = (256, 512, 768, 1024)

DEFAULT_TIER = 512
DEFAULT_ASPECT = "1:1"


def round_to(value: float, multiple: int = DEFAULT_MULTIPLE) -> int:
    """Round to the nearest positive multiple."""
    return max(multiple, int(round(value / multiple)) * multiple)


def bucket_for(aspect: str, tier: int = DEFAULT_TIER, multiple: int = DEFAULT_MULTIPLE) -> tuple:
    """Resolve a named aspect ratio at a size tier to ``(width, height)``."""
    if aspect not in ASPECT_RATIOS:
        raise ValueError(
            f"Unknown aspect ratio '{aspect}'. Available: {', '.join(ASPECT_RATIOS)}"
        )
    ratio_w, ratio_h = ASPECT_RATIOS[aspect]
    ratio = ratio_w / ratio_h
    area = float(tier) ** 2
    return round_to(math.sqrt(area * ratio), multiple), round_to(math.sqrt(area / ratio), multiple)


def buckets(tier: int = DEFAULT_TIER, multiple: int = DEFAULT_MULTIPLE) -> dict:
    """Every named ratio resolved at one tier."""
    return {name: bucket_for(name, tier, multiple) for name in ASPECT_RATIOS}


def nearest_aspect(width: int, height: int) -> str:
    """The named ratio closest to an arbitrary size, compared in log space.

    Log space so 2:1 and 1:2 are equidistant from 1:1, which they are not under
    a plain difference of ratios.
    """
    target = math.log(width / height)
    return min(
        ASPECT_RATIOS,
        key=lambda name: abs(math.log(ASPECT_RATIOS[name][0] / ASPECT_RATIOS[name][1]) - target),
    )


def snap(width: int, height: int, multiple: int = DEFAULT_MULTIPLE, max_pixels: int = None) -> tuple:
    """Round a requested size to legal dimensions, preserving aspect ratio.

    Scales down first if the request exceeds ``max_pixels``, so the returned
    dimensions always satisfy both constraints.
    """
    width, height = max(1, int(width)), max(1, int(height))
    if max_pixels and width * height > max_pixels:
        scale = math.sqrt(max_pixels / (width * height))
        width, height = width * scale, height * scale

    snapped = (round_to(width, multiple), round_to(height, multiple))
    # Rounding up can push a borderline request back over the cap; step the
    # longer side down until it fits rather than returning something illegal.
    while max_pixels and snapped[0] * snapped[1] > max_pixels:
        if snapped[0] >= snapped[1] and snapped[0] > multiple:
            snapped = (snapped[0] - multiple, snapped[1])
        elif snapped[1] > multiple:
            snapped = (snapped[0], snapped[1] - multiple)
        else:
            break
    return snapped


# --------------------------------------------------------------------------
# Prompt parsing
# --------------------------------------------------------------------------

# "1024x1024", "1024 x 768", "1024 by 768"
_EXPLICIT_SIZE = re.compile(r"\b(\d{2,5})\s*(?:x|×|by)\s*(\d{2,5})\b", re.IGNORECASE)
# "16:9", "16 / 9". "16x9" is ambiguous with a size and is not matched here.
_EXPLICIT_RATIO = re.compile(r"\b(\d{1,2})\s*[:/]\s*(\d{1,2})\b")

# Phrasings that name a shape rather than a measurement. Longest first, so
# "phone wallpaper" wins over "wallpaper".
_NAMED_SHAPES = (
    ("phone wallpaper", "9:16"),
    ("desktop wallpaper", "16:9"),
    ("ultra wide", "21:9"),
    ("ultrawide", "21:9"),
    ("cinematic", "21:9"),
    ("widescreen", "16:9"),
    ("wide screen", "16:9"),
    ("landscape", "3:2"),
    ("portrait", "2:3"),
    ("vertical", "9:16"),
    ("horizontal", "16:9"),
    ("square", "1:1"),
    ("banner", "21:9"),
    ("wallpaper", "16:9"),
    ("poster", "2:3"),
    ("thumbnail", "1:1"),
)

# Shape words that are just as likely to be describing the subject. These still
# set the aspect ratio, but are left in the prompt unless used as a directive.
_SUBJECT_WORDS = frozenset(
    {"portrait", "landscape", "poster", "banner", "wallpaper", "thumbnail", "cinematic"}
)

# Tier words, mapped to a square-equivalent side length.
_NAMED_TIERS = (
    ("4k", 1024),
    ("2k", 1024),
    ("ultra hd", 1024),
    ("full hd", 1024),
    ("high resolution", 1024),
    ("high-resolution", 1024),
    ("hi res", 1024),
    ("hd", 768),
    ("low resolution", 256),
    ("thumbnail size", 256),
)

# Words that mark a shape phrase as an instruction rather than subject matter.
# "make it portrait" is a directive; "a portrait of a woman" is a painting.
_DIRECTIVE_BEFORE = r"(?:in|at|as|make it|render it|format|orientation|aspect|ratio|size)\s+"
_DIRECTIVE_AFTER = r"\s+(?:orientation|format|aspect|ratio|mode)"


def _is_directive(prompt: str, phrase: str) -> bool:
    """Whether a shape word is being used as an instruction."""
    escaped = re.escape(phrase)
    return bool(
        re.search(rf"\b{_DIRECTIVE_BEFORE}{escaped}\b", prompt, re.IGNORECASE)
        or re.search(rf"\b{escaped}{_DIRECTIVE_AFTER}\b", prompt, re.IGNORECASE)
    )


def parse_size_intent(prompt: str) -> dict:
    """Extract sizing intent from a natural-language prompt.

    Returns ``{"width", "height", "aspect", "tier", "matched"}`` with ``None``
    for anything not found.

    ``matched`` holds only the spans that are safe to remove from the
    conditioning text: explicit sizes and ratios, and shape words used as
    instructions. A bare shape word still sets the aspect ratio but stays in the
    prompt, because it is usually describing the subject. "a portrait of a
    woman" is a painting, not an orientation, and stripping the word would leave
    "a of a woman".
    """
    found = {"width": None, "height": None, "aspect": None, "tier": None, "matched": []}
    if not prompt:
        return found

    lowered = prompt.lower()

    match = _EXPLICIT_SIZE.search(prompt)
    if match:
        found["width"], found["height"] = int(match.group(1)), int(match.group(2))
        found["matched"].append(match.group(0))
    else:
        ratio_match = _EXPLICIT_RATIO.search(prompt)
        if ratio_match:
            w, h = int(ratio_match.group(1)), int(ratio_match.group(2))
            if w > 0 and h > 0:
                name = f"{w}:{h}"
                found["aspect"] = name if name in ASPECT_RATIOS else nearest_aspect(w, h)
                found["matched"].append(ratio_match.group(0))

    if found["aspect"] is None and found["width"] is None:
        for phrase, aspect in _NAMED_SHAPES:
            if re.search(rf"\b{re.escape(phrase)}\b", lowered):
                found["aspect"] = aspect
                if phrase not in _SUBJECT_WORDS or _is_directive(prompt, phrase):
                    found["matched"].append(phrase)
                break

    for phrase, tier in _NAMED_TIERS:
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            found["tier"] = tier
            found["matched"].append(phrase)
            break

    return found


def strip_size_intent(prompt: str, matched: list) -> str:
    """Remove recognised sizing directives from the conditioning text.

    Only the matched spans and the words that mark them as instructions go; the
    rest of the prompt is left alone. Removing the whole clause would drop
    subject matter along with the instruction.
    """
    cleaned = prompt
    for phrase in matched:
        cleaned = re.sub(
            rf"(?:\b{_DIRECTIVE_BEFORE})?{re.escape(phrase)}\b(?:{_DIRECTIVE_AFTER})?",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
    cleaned = re.sub(r"\s*,\s*,", ",", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" ,.;")
