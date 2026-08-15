#!/usr/bin/env python3
"""FramerAI inference worker.

A line-oriented JSON worker driven over stdin/stdout, used by the backend to run
real inference. It loads an exported checkpoint once, then answers requests:

    request:  {"id": 1, "op": "chat", "params": {"prompt": "hi"}}
    response: {"id": 1, "ok": true, "result": {"content": "..."}}

Ops: chat, code, image, video, audio, transcribe, understand.

With --mind PATH (or MIND_PATH) the worker also carries a persistent cognitive
layer - memory, curiosity, affect, self-model - across requests and restarts.
Chat then runs through it, and these ops become available: see, hear, watch,
live, wonder, reflect, feedback, introspect. Without the flag nothing changes.

Generated media (image/video/audio) is written into the directory passed as
params.out_dir and the response returns the file name so the backend can serve it.

Usage:
    python -m model.serve --model PATH --tokenizer PATH [--device auto] [--mind PATH]
"""

import argparse
import json
import os
import sys
import uuid


def _print(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _save_image(images, out_dir):
    name = f"{uuid.uuid4()}.png"
    images[0].save(os.path.join(out_dir, name))
    return name


def _save_video(frames, out_dir):
    name = f"{uuid.uuid4()}.gif"
    path = os.path.join(out_dir, name)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    return name


def _save_audio(waveform, sample_rate, out_dir):
    import numpy as np

    name = f"{uuid.uuid4()}.wav"
    path = os.path.join(out_dir, name)
    data = np.clip(waveform, -1.0, 1.0)
    try:
        import soundfile as sf

        sf.write(path, data, sample_rate)
    except ImportError:
        import struct
        import wave

        pcm = (data * 32767).astype("<i2")
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f"<{len(pcm)}h", *pcm.tolist()))
    return name


def _sampling(params):
    """Sampling controls the caller set, leaving the rest to the generator."""
    keys = ("temperature", "top_k", "top_p")
    return {k: params[k] for k in keys if params.get(k) is not None}


def _load_image(path, size):
    """Load an image file as a (3, size, size) tensor in the model's range."""
    import numpy as np
    import torch
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((size, size))
    return torch.from_numpy(np.asarray(img, dtype="float32")).permute(2, 0, 1) / 127.5 - 1.0


def _load_frames(path, size, limit=32):
    """Load a video file as a (T, 3, size, size) tensor. Needs opencv-python."""
    import torch

    from .cognition.perception import frame_to_tensor

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("video input needs opencv-python: pip install opencv-python") from exc

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"could not open video {path!r}")
    frames = []
    try:
        while len(frames) < limit:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (size, size))
            frames.append(frame_to_tensor(rgb))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path!r}")
    return torch.stack(frames)


def _require_mind(mind, op):
    if mind is None:
        raise ValueError(f"op '{op}' needs the cognition layer; start the worker with --mind PATH")
    return mind


def handle(gen, op, params, mind=None):

    out_dir = params.get("out_dir", ".")
    os.makedirs(out_dir, exist_ok=True)
    prompt = params.get("prompt", "")

    if op in ("chat", "text"):
        if mind is not None:
            reply, trace = mind.converse(
                prompt,
                max_new_tokens=params.get("max_new_tokens", 256),
                **_sampling(params),
            )
            # The trace travels with the reply so a client can show what was
            # recalled and how the model felt, rather than guessing.
            return {"content": reply, "trace": trace.to_dict()}
        return {
            "content": gen.generate_text(
                prompt,
                max_new_tokens=params.get("max_new_tokens", 256),
                **_sampling(params),
            )
        }

    if op in ("see", "hear", "watch", "wonder", "reflect", "feedback", "introspect", "live"):
        return _handle_mind(gen, op, params, _require_mind(mind, op))

    if op == "code":
        # generate_code fixes top_k / top_p itself; only temperature is exposed.
        code_kwargs = {"language": params.get("language", "python")}
        if params.get("max_new_tokens") is not None:
            code_kwargs["max_new_tokens"] = params["max_new_tokens"]
        if params.get("temperature") is not None:
            code_kwargs["temperature"] = params["temperature"]
        return {"content": gen.generate_code(prompt, **code_kwargs)}

    if op == "image":
        images, request = gen.generate_image(
            prompt,
            num_images=params.get("num_images", 1),
            width=params.get("width"),
            height=params.get("height"),
            aspect=params.get("aspect"),
            tier=params.get("tier"),
            seed=params.get("seed"),
            resolution=params.get("resolution"),
            return_request=True,
        )
        # The resolved size comes back so the caller can tell what was
        # understood, especially when it was read out of the prompt.
        return {"file": _save_image(images, out_dir), **request.to_dict()}

    if op == "video":
        frames, request = gen.generate_video(
            prompt,
            num_frames=params.get("num_frames"),
            width=params.get("width"),
            height=params.get("height"),
            aspect=params.get("aspect"),
            tier=params.get("tier"),
            fps=params.get("fps"),
            seed=params.get("seed"),
            return_request=True,
        )
        return {
            "file": _save_video(frames, out_dir),
            "frames": len(frames),
            "fps": params.get("fps"),
            **request.to_dict(),
        }

    if op == "audio":
        waveform, sample_rate = gen.generate_audio(prompt)
        return {"file": _save_audio(waveform, sample_rate, out_dir)}

    if op == "transcribe":
        from model.data import load_waveform

        wav = load_waveform(params["audio_path"], gen.model.config.audio_sample_rate)
        return {"content": gen.transcribe(wav)}

    if op == "understand":
        tensor = _load_image(params["image_path"], gen.model.config.image_size)
        return {"content": gen.generate_text(params.get("prompt", "Describe this:"), image=tensor)}

    raise ValueError(f"Unknown op: {op}")


def _handle_mind(gen, op, params, mind):
    """Ops that only exist when the cognition layer is running."""
    from .data import load_waveform

    config = gen.model.config

    if op == "see":
        image = _load_image(params["image_path"], config.image_size)
        trace = mind.perceive_image(
            image, caption=params.get("caption", ""), describe=params.get("describe", True)
        )
        return trace.to_dict()

    if op == "hear":
        waveform = load_waveform(params["audio_path"], config.audio_sample_rate)
        trace = mind.perceive_audio(
            waveform, caption=params.get("caption", ""),
            transcribe=params.get("transcribe", True),
        )
        return trace.to_dict()

    if op == "watch":
        frames = _load_frames(
            params["video_path"], config.image_size, limit=params.get("max_frames", 32)
        )
        trace = mind.perceive_video(
            frames, caption=params.get("caption", ""),
            describe=params.get("describe", True), keyframes=params.get("keyframes", 4),
        )
        return trace.to_dict()

    if op == "live":
        return _handle_live(params, mind)

    if op == "wonder":
        return {"content": mind.wonder()}

    if op == "reflect":
        return mind.rest()

    if op == "feedback":
        trace = mind.reward(float(params.get("value", 0.0)), note=params.get("note", ""))
        return trace.to_dict()

    if op == "introspect":
        return mind.introspect()

    raise ValueError(f"Unknown op: {op}")


def _handle_live(params, mind):
    """Watch and listen to real hardware for a bounded stretch of time."""
    from .cognition.perception import CameraSource, LiveSession, MicrophoneSource

    sources = []
    if params.get("camera", True):
        sources.append(CameraSource(params.get("camera_device", 0), size=params.get("size", 256)))
    if params.get("microphone", False):
        sources.append(MicrophoneSource(seconds=params.get("chunk_seconds", 1.5)))
    if not sources:
        raise ValueError("live needs at least one of camera or microphone enabled")

    session = LiveSession(
        mind, sources,
        fps=params.get("fps", 2.0),
        change_threshold=params.get("change_threshold", 0.15),
        describe=params.get("describe", True),
        transcribe=params.get("transcribe", True),
    )
    try:
        events = session.run(seconds=float(params.get("seconds", 10.0)))
    finally:
        session.close()

    return {
        **session.summary(),
        "attended": [e.to_dict() for e in events if e.attended],
    }


def load_mind(generator, path):
    """Attach the cognition layer, resuming the saved mind when there is one."""
    from .cognition import Mind

    if path and os.path.exists(path):
        return Mind.load(path, generator=generator)
    return Mind.from_generator(generator)


def main():
    parser = argparse.ArgumentParser(description="FramerAI inference worker")
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--tokenizer", default=os.environ.get("TOKENIZER_PATH"))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "auto"))
    parser.add_argument(
        "--mind", default=os.environ.get("MIND_PATH"),
        help="path to a saved mind; enables the cognition layer and persists it there",
    )
    args = parser.parse_args()

    if not args.model or not os.path.exists(args.model):
        _print({"ready": False, "error": f"Model not found: {args.model}"})
        sys.exit(1)

    try:
        from model.generate import FramerGenerator

        gen = FramerGenerator.from_checkpoint(args.model, args.tokenizer, args.device)
    except Exception as exc:  # noqa: BLE001 - report any load failure to the caller
        _print({"ready": False, "error": str(exc)})
        sys.exit(1)

    mind, mind_error = None, None
    if args.mind:
        try:
            mind = load_mind(gen, args.mind)
        except Exception as exc:  # noqa: BLE001 - a broken mind must not block serving
            mind_error = str(exc)

    # One ready line, always: the bridge waits for exactly one.
    ready = {"ready": True, "mind": mind is not None}
    if mind_error:
        ready["mind_error"] = mind_error
    _print(ready)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        req_id = req.get("id")
        try:
            result = handle(gen, req.get("op", "chat"), req.get("params", {}), mind=mind)
            if mind is not None and args.mind:
                # Persist after every request: a mind that only survives a clean
                # shutdown is a mind that loses its day whenever the worker dies.
                mind.save(args.mind)
            _print({"id": req_id, "ok": True, "result": result})
        except Exception as exc:  # noqa: BLE001 - never let one request kill the worker
            _print({"id": req_id, "ok": False, "error": str(exc)})


if __name__ == "__main__":
    main()
