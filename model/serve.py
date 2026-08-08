#!/usr/bin/env python3
"""FramerAI inference worker.

A line-oriented JSON worker driven over stdin/stdout, used by the backend to run
real inference. It loads an exported checkpoint once, then answers requests:

    request:  {"id": 1, "op": "chat", "params": {"prompt": "hi"}}
    response: {"id": 1, "ok": true, "result": {"content": "..."}}

Ops: chat, code, image, video, audio, transcribe, understand.

Generated media (image/video/audio) is written into the directory passed as
params.out_dir and the response returns the file name so the backend can serve it.

Usage:
    python -m model.serve --model PATH --tokenizer PATH [--device auto]
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


def handle(gen, op, params):
    import torch

    out_dir = params.get("out_dir", ".")
    os.makedirs(out_dir, exist_ok=True)
    prompt = params.get("prompt", "")

    if op in ("chat", "text"):
        return {
            "content": gen.generate_text(
                prompt,
                max_new_tokens=params.get("max_new_tokens", 256),
                **_sampling(params),
            )
        }

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
        frames = gen.generate_video(prompt, num_frames=params.get("num_frames", 16))
        return {"file": _save_video(frames, out_dir)}

    if op == "audio":
        waveform, sample_rate = gen.generate_audio(prompt)
        return {"file": _save_audio(waveform, sample_rate, out_dir)}

    if op == "transcribe":
        from model.data import load_waveform

        wav = load_waveform(params["audio_path"], gen.model.config.audio_sample_rate)
        return {"content": gen.transcribe(wav)}

    if op == "understand":
        import numpy as np
        from PIL import Image

        img = Image.open(params["image_path"]).convert("RGB").resize(
            (gen.model.config.image_size, gen.model.config.image_size)
        )
        tensor = torch.from_numpy(np.asarray(img, dtype="float32")).permute(2, 0, 1) / 127.5 - 1.0
        return {"content": gen.generate_text(params.get("prompt", "Describe this:"), image=tensor)}

    raise ValueError(f"Unknown op: {op}")


def main():
    parser = argparse.ArgumentParser(description="FramerAI inference worker")
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--tokenizer", default=os.environ.get("TOKENIZER_PATH"))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "auto"))
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

    _print({"ready": True})

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
            result = handle(gen, req.get("op", "chat"), req.get("params", {}))
            _print({"id": req_id, "ok": True, "result": result})
        except Exception as exc:  # noqa: BLE001 - never let one request kill the worker
            _print({"id": req_id, "ok": False, "error": str(exc)})


if __name__ == "__main__":
    main()
