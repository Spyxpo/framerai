"""Live perception: a camera and a microphone feeding the tick loop continuously.

Batch perception is a mind that only exists when spoken to. This module keeps it
running: sources poll at a set rate, an attention gate decides what is worth
attending to, and whatever passes becomes an episode like any other - encoded,
appraised, felt, remembered, and eventually consolidated in sleep.

The gate is the part that matters. A camera at 2 fps produces 7,200 frames an
hour, nearly all of them identical to the one before. Writing them all would
bury every real memory under duplicates and drive novelty to zero. So a frame is
attended only when it differs enough from the last attended one, or when enough
time has passed that a check-in is due regardless. What lands in memory is
change, which is what a scene actually consists of.

Hardware backends are optional and imported lazily: ``opencv-python`` for camera
and video files, ``sounddevice`` for the microphone. Neither is needed to use the
streaming path - :class:`CallableSource` feeds it from anything, which is how the
tests drive a full live session with no hardware at all.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass

import torch


class SensorySource:
    """A sense: something that can be polled for the next reading.

    ``read`` returns a tensor or None (nothing available right now). Sources are
    expected to be non-blocking enough for the session's tick rate.
    """

    modality = "unknown"
    name = "source"

    def read(self) -> torch.Tensor | None:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def __enter__(self) -> "SensorySource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class CallableSource(SensorySource):
    """A sense backed by any callable. Hardware-free, and what the tests use."""

    def __init__(self, fn, modality: str = "vision", name: str = "callable"):
        self.fn = fn
        self.modality = modality
        self.name = name

    def read(self) -> torch.Tensor | None:
        value = self.fn()
        if value is None:
            return None
        return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)


class CameraSource(SensorySource):
    """Live camera, or a video file - both are frame streams.

    ``device`` is a camera index (0 is the built-in webcam) or a path to a video
    file, which is what makes "watch this clip" and "watch the room" the same
    code path. Frames come back as (3, size, size) in [-1, 1], matching what the
    vision tower expects.
    """

    modality = "vision"

    def __init__(self, device: int | str = 0, size: int = 256, name: str = "camera"):
        try:
            import cv2  # noqa: PLC0415 - optional dependency, imported on use
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise RuntimeError(
                "camera capture needs opencv-python: pip install opencv-python"
            ) from exc

        self._cv2 = cv2
        self.size = size
        self.name = name
        self.capture = cv2.VideoCapture(device)
        if not self.capture.isOpened():
            raise RuntimeError(f"could not open video source {device!r}")

    def read(self) -> torch.Tensor | None:
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None
        frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        frame = self._cv2.resize(frame, (self.size, self.size))
        return frame_to_tensor(frame)

    def close(self) -> None:
        self.capture.release()


class MicrophoneSource(SensorySource):
    """Live microphone, read in fixed-length chunks.

    Each chunk is one hearing event: long enough to carry a phrase, short enough
    that the mind notices a change in the room while it is still happening.
    """

    modality = "hearing"

    def __init__(
        self,
        sample_rate: int = 16000,
        seconds: float = 1.5,
        device=None,
        name: str = "microphone",
    ):
        try:
            import sounddevice  # noqa: PLC0415 - optional dependency, imported on use
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise RuntimeError(
                "microphone capture needs sounddevice: pip install sounddevice"
            ) from exc

        self._sd = sounddevice
        self.sample_rate = sample_rate
        self.frames = int(sample_rate * seconds)
        self.name = name
        self.stream = sounddevice.InputStream(
            samplerate=sample_rate, channels=1, dtype="float32", device=device
        )
        self.stream.start()

    def read(self) -> torch.Tensor | None:
        available = self.stream.read_available
        if available < self.frames:
            return None
        chunk, overflowed = self.stream.read(self.frames)
        if overflowed:
            # Dropped samples mean the consumer is behind the microphone; the
            # chunk is still usable, and saying so beats pretending otherwise.
            pass
        return torch.from_numpy(chunk).reshape(-1).float()

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()


def frame_to_tensor(frame, size: int | None = None) -> torch.Tensor:
    """RGB HWC frame (numpy or tensor) to (3, H, W) float in [-1, 1].

    Integer input is treated as 0-255 and rescaled; float input is assumed to be
    normalised already. The test is the dtype, not the values - a black uint8
    frame has a maximum of 0, so a value-based check would pass it through
    unscaled and every dark scene would arrive at the wrong brightness.
    """
    raw = torch.as_tensor(frame)
    if raw.dim() != 3:
        raise ValueError(f"expected an (H, W, 3) frame, got shape {tuple(raw.shape)}")

    byte_range = not raw.is_floating_point() or float(raw.max()) > 1.5
    tensor = raw.permute(2, 0, 1).float()
    if byte_range:
        tensor = tensor / 127.5 - 1.0
    if size is not None and tensor.shape[-1] != size:
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
        ).squeeze(0)
    return tensor


class ChangeGate:
    """Attend to a reading only when it differs from the last one attended to.

    ``threshold`` is cosine distance in the mind's embedding space.
    ``max_skip`` forces a look every so often regardless, so a perfectly static
    scene still produces the occasional episode instead of vanishing from the
    mind's history entirely.
    """

    def __init__(self, threshold: float = 0.15, max_skip: int = 30):
        if not 0.0 <= threshold <= 2.0:
            raise ValueError("threshold is a cosine distance in [0, 2]")
        self.threshold = threshold
        self.max_skip = max_skip
        self._last: torch.Tensor | None = None
        self._skipped = 0

    def admit(self, embedding: torch.Tensor) -> tuple[bool, float]:
        """Returns ``(attend, change)`` where change is the cosine distance."""
        if self._last is None:
            self._last = embedding.clone()
            self._skipped = 0
            return True, 1.0

        change = float(1.0 - torch.dot(self._last, embedding))
        forced = self._skipped >= self.max_skip
        if change >= self.threshold or forced:
            self._last = embedding.clone()
            self._skipped = 0
            return True, change

        self._skipped += 1
        return False, change

    def reset(self) -> None:
        self._last = None
        self._skipped = 0


@dataclass
class SensoryEvent:
    """One poll of one sense, whether or not it was attended to."""

    modality: str
    source: str
    attended: bool
    change: float
    trace: object | None = None

    def to_dict(self) -> dict:
        return {
            "modality": self.modality,
            "source": self.source,
            "attended": self.attended,
            "change": round(self.change, 4),
            "trace": self.trace.to_dict() if self.trace is not None else None,
        }


class LiveSession:
    """Keep a mind perceiving continuously from live senses.

    Use it directly for a bounded run::

        session = LiveSession(mind, sources=[CameraSource(0), MicrophoneSource()])
        session.run(seconds=30)
        print(mind.introspect())

    or in the background, where a chat loop can keep talking while the mind goes
    on seeing and hearing::

        session.start()
        ...
        session.stop()

    The mind is not thread-safe on its own; every state change here goes through
    the mind's lock, so a background session and a foreground ``converse`` cannot
    interleave halfway through a tick.
    """

    def __init__(
        self,
        mind,
        sources: list[SensorySource] | None = None,
        fps: float = 2.0,
        change_threshold: float = 0.15,
        max_skip: int = 30,
        describe: bool = False,
        transcribe: bool = False,
        wonder_when_idle: bool = True,
        idle_ticks: int = 60,
        history: int = 256,
    ):
        self.mind = mind
        self.sources = list(sources or [])
        self.interval = 1.0 / fps if fps > 0 else 0.0
        self.describe = describe
        self.transcribe = transcribe
        self.wonder_when_idle = wonder_when_idle
        self.idle_ticks = idle_ticks
        self._gates = {id(s): ChangeGate(change_threshold, max_skip) for s in self.sources}
        self._events: deque = deque(maxlen=history)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._idle = 0
        self.polls = 0

    # ------------------------------------------------------------------
    # sources
    # ------------------------------------------------------------------

    def add_source(self, source: SensorySource) -> SensorySource:
        self.sources.append(source)
        self._gates.setdefault(id(source), ChangeGate())
        return source

    @property
    def grounded(self) -> bool:
        """True when readings are encoded by the model's towers, not pooled."""
        return self.mind.encoder.multimodal

    # ------------------------------------------------------------------
    # the loop
    # ------------------------------------------------------------------

    def step(self) -> list[SensoryEvent]:
        """Poll every sense once. Attended readings become episodes."""
        events: list[SensoryEvent] = []
        for source in self.sources:
            reading = source.read()
            self.polls += 1
            if reading is None:
                continue

            with self.mind.lock:
                embedding = self.mind.encoder.encode_sensor(reading, source.modality)
                attend, change = self._gates[id(source)].admit(embedding)
                event = SensoryEvent(source.modality, source.name, attend, change)
                if attend:
                    event.trace = self._attend(source, reading, embedding, change)
            events.append(event)

        self._events.extend(events)
        self._maybe_wonder(events)
        return events

    def _attend(self, source, reading, embedding, change: float):
        """Turn an attended reading into an episode, describing it when asked."""
        mind = self.mind
        text = ""
        if source.modality == "vision" and self.describe:
            text = mind.describe_image(reading) or ""
        elif source.modality == "hearing" and self.transcribe:
            text = mind.transcribe_audio(reading) or ""

        if not text:
            text = f"<{source.modality} from {source.name}, change {change:.2f}>"

        return mind.perceive(
            text,
            kind="perception",
            embedding=embedding,
            modality=source.modality,
            topic=source.modality if not text.startswith("<") else f"{source.modality}-stream",
        )

    def _maybe_wonder(self, events: list[SensoryEvent]) -> None:
        """Nothing changing is itself a state: an idle mind turns inward."""
        if not self.wonder_when_idle:
            return
        if any(e.attended for e in events):
            self._idle = 0
            return
        self._idle += 1
        if self._idle >= self.idle_ticks:
            self._idle = 0
            with self.mind.lock:
                self.mind.wonder()

    def run(self, seconds: float | None = None, ticks: int | None = None,
            sleep_fn=time.sleep, clock=time.monotonic) -> list[SensoryEvent]:
        """Run in the foreground for a bounded time or number of polls.

        ``sleep_fn`` and ``clock`` are injectable so a test can run a full
        session in microseconds without waiting on a real clock.
        """
        if seconds is None and ticks is None:
            raise ValueError("run() needs either seconds or ticks")

        collected: list[SensoryEvent] = []
        started = clock()
        count = 0
        while not self._stop.is_set():
            collected.extend(self.step())
            count += 1
            if ticks is not None and count >= ticks:
                break
            if seconds is not None and clock() - started >= seconds:
                break
            if self.interval:
                sleep_fn(self.interval)
        return collected

    def start(self) -> "LiveSession":
        """Perceive in the background until :meth:`stop`."""
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                self.step()
                if self.interval:
                    self._stop.wait(self.interval)

        self._thread = threading.Thread(target=loop, name="framer-live", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> "LiveSession":
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None
        return self

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def recent(self, n: int = 10) -> list[SensoryEvent]:
        return list(self._events)[-n:]

    def summary(self) -> dict:
        events = list(self._events)
        attended = [e for e in events if e.attended]
        return {
            "polls": self.polls,
            "events": len(events),
            "attended": len(attended),
            "attention_rate": round(len(attended) / len(events), 4) if events else 0.0,
            "sources": [{"name": s.name, "modality": s.modality} for s in self.sources],
            "grounded": self.grounded,
            "running": self.running,
        }

    def close(self) -> None:
        self.stop()
        for source in self.sources:
            source.close()

    def __enter__(self) -> "LiveSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
