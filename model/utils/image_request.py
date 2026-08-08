"""Resolve an image request's dimensions from parameters and prompt text.

Precedence is explicit over implicit, always:

1. Explicit ``width``/``height`` from the caller.
2. An explicit ``aspect`` (and ``tier``) from the caller.
3. Sizing intent parsed out of the prompt text.
4. The configured default.

A resolved request reports which tier it came from, so a caller can tell the
difference between "you asked for this" and "we guessed", and the prompt is
returned with any recognised sizing directive stripped so it does not also act
as subject matter.
"""

from dataclasses import dataclass, field

from .image_sizing import (
    ASPECT_RATIOS,
    DEFAULT_MULTIPLE,
    bucket_for,
    nearest_aspect,
    parse_size_intent,
    snap,
    strip_size_intent,
)


@dataclass
class ImageRequest:
    """A fully resolved image generation request."""

    prompt: str
    width: int
    height: int
    num_images: int = 1
    aspect: str = None
    tier: int = None
    source: str = "default"  # "explicit" | "prompt" | "default"
    snapped: bool = False
    seed: int = None
    matched: list = field(default_factory=list)

    @property
    def shape(self) -> tuple:
        """Pixel-space shape for the generator, ``(B, C, H, W)``."""
        return (self.num_images, 3, self.height, self.width)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "aspect": self.aspect,
            "tier": self.tier,
            "source": self.source,
            "snapped": self.snapped,
            "num_images": self.num_images,
            "seed": self.seed,
        }


def resolve_image_request(
    prompt: str = "",
    *,
    width: int = None,
    height: int = None,
    aspect: str = None,
    tier: int = None,
    num_images: int = 1,
    seed: int = None,
    config=None,
    multiple: int = DEFAULT_MULTIPLE,
) -> ImageRequest:
    """Resolve a request to concrete, legal dimensions.

    ``config`` supplies the defaults (``image_size_tier``, ``image_default_aspect``,
    ``image_max_pixels``, ``image_allow_custom_size``) and may be omitted, in
    which case the module defaults apply.
    """
    default_tier = getattr(config, "image_size_tier", 512) if config else 512
    default_aspect = getattr(config, "image_default_aspect", "1:1") if config else "1:1"
    max_pixels = getattr(config, "image_max_pixels", 1024 * 1024) if config else 1024 * 1024
    allow_prompt_parsing = getattr(config, "image_allow_custom_size", True) if config else True

    # An explicit parameter means the prompt is not consulted at all, so a
    # sizing word in the text cannot quietly override what the caller asked for.
    parse = allow_prompt_parsing and width is None and height is None and aspect is None
    intent = (
        parse_size_intent(prompt)
        if parse
        else {"width": None, "height": None, "aspect": None, "tier": None, "matched": []}
    )

    cleaned_prompt = strip_size_intent(prompt, intent["matched"]) if intent["matched"] else prompt
    resolved_tier = tier or intent["tier"] or default_tier

    if width is not None and height is not None:
        source = "explicit"
        raw = (int(width), int(height))
        resolved_aspect = aspect if aspect in ASPECT_RATIOS else nearest_aspect(*raw)
    elif aspect is not None:
        source = "explicit"
        resolved_aspect = aspect
        raw = bucket_for(aspect, resolved_tier, multiple)
    elif intent["width"] is not None:
        source = "prompt"
        raw = (intent["width"], intent["height"])
        resolved_aspect = nearest_aspect(*raw)
    elif intent["aspect"] is not None:
        source = "prompt"
        resolved_aspect = intent["aspect"]
        raw = bucket_for(resolved_aspect, resolved_tier, multiple)
    elif intent["tier"] is not None:
        source = "prompt"
        resolved_aspect = default_aspect
        raw = bucket_for(resolved_aspect, resolved_tier, multiple)
    else:
        source = "default"
        resolved_aspect = default_aspect
        raw = bucket_for(resolved_aspect, resolved_tier, multiple)

    final = snap(raw[0], raw[1], multiple, max_pixels)

    return ImageRequest(
        prompt=cleaned_prompt,
        width=final[0],
        height=final[1],
        num_images=max(1, int(num_images)),
        aspect=resolved_aspect,
        tier=resolved_tier,
        source=source,
        snapped=final != tuple(int(v) for v in raw),
        seed=seed,
        matched=intent["matched"],
    )
