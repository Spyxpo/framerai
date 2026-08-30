"""Dense text recognition: is the text in an image read, or only described.

The audio side has had a character error rate since the harness was written.
The image side has Frechet distance and contrastive alignment, which say
whether a picture looks right and whether it matches a caption, and say nothing
at all about whether the words inside it were read. So there was no number to
move, and no way to tell "a page of text about quarterly results" from an
answer about the figure on that page.

Every document use depends on this, and so does any screenshot or photographed
page. The suite renders known strings at a range of sizes and densities,
reads them back through the model's own vision path, and scores the result
against what was rendered.

The material is generated rather than downloaded: the strings are fixed, the
rendering is deterministic, and the whole thing runs on CPU at tiny scale, so
it has no data dependency and no licence attached.

This is the measurement. Training recognition is the work it measures, and is
deliberately separate: a metric that only exists after the thing it scores is
a metric nobody can use to decide whether the training helped.
"""

import torch

from .audio import character_error_rate, word_error_rate

# Short, unambiguous strings. Digits and letters that are easy to confuse are
# present on purpose, because a page of real text contains them.
SAMPLE_LINES = (
    "the quarterly total was 4820",
    "invoice 10592 dated 14 March",
    "section 3.2 page 47",
    "balance carried forward 918.30",
    "reference AB1029 approved",
    "temperature held at 21 degrees",
)

# Rendered sizes, from comfortable down to the point where a fixed-resolution
# encoder loses the glyphs. The sweep is the interesting part: one size says
# almost nothing.
DEFAULT_FONT_SIZES = (24, 16, 11)

DEFAULT_PROMPT = "Read the text in this image exactly:"


def render_text_image(text: str, width: int = 384, height: int = 96, font_size: int = 16):
    """Render a line of text to a PIL image, black on white.

    Uses the bundled bitmap font when no scalable font is available, so the
    suite renders the same way on any machine rather than depending on whatever
    fonts happen to be installed.
    """
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    font = None
    for candidate in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            font = ImageFont.truetype(candidate, font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    draw.text((8, max(0, (height - font_size) // 2)), text, fill=(0, 0, 0), font=font)
    return image


def image_to_tensor(image, config=None) -> torch.Tensor:
    """Convert a rendered page to the tensor the vision tower expects."""
    import numpy as np

    if config is not None and not getattr(config, "vision_tiling", False):
        size = config.image_size
        image = image.resize((size, size))

    array = torch.from_numpy(np.asarray(image, dtype="float32"))
    return array.permute(2, 0, 1) / 127.5 - 1.0


def build_samples(lines=SAMPLE_LINES, font_sizes=DEFAULT_FONT_SIZES, config=None) -> list:
    """Rendered pages paired with the text that was rendered on them."""
    samples = []
    for index, line in enumerate(lines):
        font_size = font_sizes[index % len(font_sizes)]
        image = render_text_image(line, font_size=font_size)
        samples.append({
            "text": line,
            "font_size": font_size,
            "image": image_to_tensor(image, config),
        })
    return samples


def read_back(generator, sample, prompt: str = DEFAULT_PROMPT, max_new_tokens: int = 48) -> str:
    """What the model says is written on a rendered page."""
    answer = generator.generate_text(
        prompt, max_new_tokens=max_new_tokens, image=sample["image"], temperature=0.1
    )
    # The prompt is echoed by the decoder, so only the continuation is scored.
    return answer[len(prompt):].strip() if answer.startswith(prompt) else answer.strip()


def dense_text_accuracy(generator, samples=None, prompt: str = DEFAULT_PROMPT,
                        max_new_tokens: int = 48) -> dict:
    """Character and word error rates over rendered text, and per font size.

    A rate of 1.0 is the floor an untrained model sits at, not a failure of the
    suite: reading is trained, and this is what says whether the training took.
    """
    config = getattr(getattr(generator, "model", None), "config", None)
    samples = samples if samples is not None else build_samples(config=config)
    if not samples:
        raise ValueError("dense text eval needs at least one rendered sample")

    per_size, cers, wers = {}, [], []
    for sample in samples:
        hypothesis = read_back(generator, sample, prompt, max_new_tokens)
        cer = character_error_rate(sample["text"], hypothesis)
        wer = word_error_rate(sample["text"], hypothesis)
        cers.append(cer)
        wers.append(wer)
        per_size.setdefault(sample["font_size"], []).append(cer)

    values = {
        "cer": sum(cers) / len(cers),
        "wer": sum(wers) / len(wers),
        "samples": len(samples),
    }
    # Size is reported separately because losing small text is a different
    # failure from reading nothing, and an average hides which one happened.
    for font_size, results in sorted(per_size.items(), reverse=True):
        values[f"cer_{font_size}px"] = sum(results) / len(results)
    return values
