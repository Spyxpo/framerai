"""Evaluation harness: register metrics, run suites, report results.

Parameter count is a ceiling, not a measurement. This is the piece that turns
"the architecture is now the family that can reach frontier output" from a claim
into something with a number attached, and it is deliberately small: a registry,
a runner, and a report that says which suites ran and which were skipped.

Skipping is reported rather than silent. A suite that could not run because its
inputs were missing is not the same as one that scored well, and a harness that
conflates them is worse than no harness.
"""

import json
from dataclasses import dataclass, field


@dataclass
class EvalReport:
    """Results, plus what did not run and why."""

    metrics: dict = field(default_factory=dict)
    skipped: dict = field(default_factory=dict)

    def add(self, suite: str, values: dict):
        self.metrics[suite] = values

    def skip(self, suite: str, reason: str):
        self.skipped[suite] = reason

    def to_dict(self) -> dict:
        return {"metrics": self.metrics, "skipped": self.skipped}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=float)

    def summary(self) -> str:
        lines = []
        for suite, values in self.metrics.items():
            lines.append(f"{suite}:")
            for name, value in values.items():
                lines.append(f"  {name:<24s} {value:.6g}" if isinstance(value, (int, float))
                             else f"  {name:<24s} {value}")
        for suite, reason in self.skipped.items():
            lines.append(f"{suite}: skipped ({reason})")
        return "\n".join(lines) or "no suites ran"


class EvalHarness:
    """A registry of named suites, each a callable returning a dict of metrics."""

    def __init__(self, model=None, device: str = "cpu"):
        self.model = model
        self.device = device
        self._suites = {}

    def register(self, name: str, fn):
        """Register a suite. ``fn(model, device, **inputs) -> dict``."""
        if name in self._suites:
            raise ValueError(f"suite '{name}' is already registered")
        self._suites[name] = fn
        return fn

    def suite(self, name: str):
        """Decorator form of :meth:`register`."""
        def decorator(fn):
            self.register(name, fn)
            return fn
        return decorator

    @property
    def names(self) -> list:
        return sorted(self._suites)

    def run(self, suites: list = None, inputs: dict = None, strict: bool = False) -> EvalReport:
        """Run the named suites (or all of them) and collect a report.

        A suite that raises is recorded as skipped with its message, so one
        missing input does not abandon the rest of the evaluation. Pass
        ``strict=True`` to re-raise instead, which is what tests want.
        """
        inputs = inputs or {}
        report = EvalReport()

        for name in suites or self.names:
            if name not in self._suites:
                report.skip(name, "not registered")
                continue
            try:
                report.add(name, self._suites[name](self.model, self.device, **inputs.get(name, {})))
            except Exception as error:  # noqa: BLE001 - reported, not swallowed
                if strict:
                    raise
                report.skip(name, f"{type(error).__name__}: {error}")

        return report


def default_harness(model, device: str = "cpu") -> EvalHarness:
    """A harness with one suite per modality, using the model's own extractors."""
    from . import audio as audio_metrics
    from . import dense_text, longcontext
    from . import image as image_metrics
    from . import text as text_metrics
    from . import video as video_metrics

    harness = EvalHarness(model, device)

    @harness.suite("text")
    def _text(model, device, batches=None, **_):
        if not batches:
            raise ValueError("text eval needs `batches` of (input_ids, labels)")
        return {
            "perplexity": text_metrics.perplexity(model, batches, device),
            "token_accuracy": text_metrics.token_accuracy(model, batches, device),
        }

    @harness.suite("image")
    def _image(model, device, real=None, fake=None, captions=None, **_):
        if real is None or fake is None:
            raise ValueError("image eval needs `real` and `fake` image batches")
        values = {"fid": image_metrics.fid(model, real, fake, device)}
        if captions is not None:
            values["alignment"] = image_metrics.alignment_score(model, fake, captions, device)
        return values

    @harness.suite("audio")
    def _audio(model, device, reference=None, estimate=None, transcript=None, hypothesis=None, **_):
        values = {}
        if reference is not None and estimate is not None:
            values["si_sdr"] = audio_metrics.si_sdr(estimate, reference)
            values["mel_distance"] = audio_metrics.mel_distance(estimate, reference)
        if transcript is not None and hypothesis is not None:
            values["wer"] = audio_metrics.word_error_rate(transcript, hypothesis)
            values["cer"] = audio_metrics.character_error_rate(transcript, hypothesis)
        if not values:
            raise ValueError("audio eval needs waveforms or a transcript pair")
        return values

    @harness.suite("dense_text")
    def _dense_text(model, device, generator=None, samples=None, **_):
        # Needs a generator rather than the model alone: reading a page is a
        # generation, and the audio side's character error rate has no image
        # counterpart without one.
        if generator is None:
            raise ValueError("dense text eval needs a `generator`")
        return dense_text.dense_text_accuracy(generator, samples=samples)

    @harness.suite("long_context")
    def _long_context(model, device, tokenizer=None, lengths=None, seed=0, chunk=4096, **_):
        # A declared window is a number in a config until something retrieves
        # from it. Needs the tokenizer because the material is text, not tensors.
        if tokenizer is None:
            raise ValueError("long-context eval needs a `tokenizer`")
        return {
            "single_fact": longcontext.single_fact_accuracy(
                model, tokenizer, device, lengths=lengths, seed=seed, chunk=chunk
            ),
            "multi_hop": longcontext.multi_hop_accuracy(
                model, tokenizer, device, lengths=lengths, seed=seed, chunk=chunk
            ),
            "aggregation": longcontext.aggregation_accuracy(
                model, tokenizer, device, lengths=lengths, seed=seed, chunk=chunk
            ),
        }

    @harness.suite("video")
    def _video(model, device, real=None, fake=None, **_):
        if real is None or fake is None:
            raise ValueError("video eval needs `real` and `fake` clip batches")
        return {
            "fvd": video_metrics.fvd(model, real, fake, device),
            "temporal_consistency": video_metrics.temporal_consistency(fake),
        }

    return harness
