#!/usr/bin/env python3
"""FramerAI inference worker.

A line-oriented JSON worker driven over stdin/stdout, used by the backend to run
real inference. It loads an exported checkpoint once, then answers requests:

    request:  {"id": 1, "op": "chat", "params": {"prompt": "hi"}}
    response: {"id": 1, "ok": true, "result": {"content": "..."}}

Ops: chat, code, image, video, audio, transcribe, understand.

With --tools web the worker can reach the internet: the
`search` and `fetch` ops become available, and a chat request carrying
params.tools runs a bounded tool-calling loop before answering. With --tools cli
it can also run commands on the host, inside a sandbox root, under the mode set
by --cli-mode. Without the flags no tool is registered and nothing changes.

With --mind PATH (or MIND_PATH) the worker also carries a persistent cognitive
layer - memory, curiosity, affect, self-model - across requests and restarts.
Chat then runs through it, and these ops become available: see, hear, watch,
live, wonder, reflect, feedback, introspect. Without the flag nothing changes.

Generated media (image/video/audio) is written into the directory passed as
params.out_dir and the response returns the file name so the backend can serve it.

Usage:
    python -m model.serve --model PATH --tokenizer PATH [--device auto] [--mind PATH]
                          [--tools web,cli] [--cli-mode allow] [--cli-root .]
"""

import argparse
import io
import json
import os
import select
import sys
import uuid


def _print(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def make_stdio_approver(root: str, timeout_sec: float = 30.0):
    """An approver callback for ShellPolicy that requests approval over stdio."""
    def approve(command: str, argv: list[str]) -> bool:
        approval_id = str(uuid.uuid4())
        _print({
            "type": "approval_request",
            "approval_id": approval_id,
            "command": command,
            "argv": argv,
            "root": root,
        })
        try:
            has_fd = True
            try:
                sys.stdin.fileno()
            except (AttributeError, io.UnsupportedOperation, ValueError, OSError):
                has_fd = False

            if has_fd:
                rlist, _, _ = select.select([sys.stdin], [], [], float(timeout_sec))
                if not rlist:
                    return False

            line = sys.stdin.readline()
            if not line:
                return False
            data = json.loads(line.strip())
            if not isinstance(data, dict):
                return False
            if data.get("type") != "approval_response":
                return False
            if data.get("approval_id") != approval_id:
                return False
            return bool(data.get("approved", False))
        except Exception:
            return False

    return approve


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


def _select_tools(registry, requested):
    """The tools this request may use: all of them, a named subset, or none.

    A client asking for a capability the worker was not started with gets a
    worker without that capability, not an error - the same posture the
    cognition layer takes when it is absent.
    """
    if registry is None or not len(registry) or not requested:
        return None
    if requested is True or requested == "all":
        return registry
    from .tools import expand_toolsets

    names = [requested] if isinstance(requested, str) else list(requested)
    subset = registry.subset(expand_toolsets(names))
    return subset if len(subset) else None


TOOL_FLAGS = {"web_search": "web", "web_fetch": "web", "shell": "cli"}


def _require_tool(registry, name, op):
    tool = registry.get(name) if registry is not None else None
    if tool is None:
        toolset = TOOL_FLAGS.get(name, "web")
        raise ValueError(
            f"op '{op}' needs the {name} tool; start the worker with --tools {toolset}"
        )
    return tool


def _require_mind(mind, op):
    if mind is None:
        raise ValueError(f"op '{op}' needs the cognition layer; start the worker with --mind PATH")
    return mind


def handle(gen, op, params, mind=None, tools=None):

    out_dir = params.get("out_dir", ".")
    os.makedirs(out_dir, exist_ok=True)
    prompt = params.get("prompt", "")

    if op in ("chat", "text"):
        active = _select_tools(tools, params.get("tools"))
        max_new_tokens = params.get("max_new_tokens", 256)

        messages = params.get("messages")
        tool_trace = None
        if active is not None:
            from .tools import run_tool_loop

            def generate(text):
                return gen.generate_text(text, max_new_tokens=max_new_tokens, **_sampling(params))

            tool_input = messages if messages else prompt
            reply, tool_trace = run_tool_loop(
                generate, active, tool_input, max_steps=params.get("max_tool_steps", 4)
            )
            if mind is None:
                # The trace carries every query and page, so an answer sourced
                # from the web can be checked rather than taken on faith.
                return {"content": reply, "tools": tool_trace.to_dict()}
            # With a mind attached the tools gather; the mind still answers, so
            # the exchange lands in memory as one episode rather than four.
            prompt = f"{tool_trace.context()}\n\n{prompt}" if tool_trace.context() else prompt

        if messages and not prompt:
            from .tokenizer.chat_template import ChatTemplate

            prompt = ChatTemplate(version="v1").format_messages(
                messages, add_generation_prompt=True
            )
        elif prompt and not prompt.startswith("<"):
            from .tokenizer.chat_template import ChatTemplate

            prompt = ChatTemplate(version="v1").format_messages(
                [{"role": "user", "content": prompt}], add_generation_prompt=True
            )

        if mind is not None:
            reply, trace = mind.converse(
                prompt,
                max_new_tokens=max_new_tokens,
                **_sampling(params),
            )
            # The trace travels with the reply so a client can show what was
            # recalled and how the model felt, rather than guessing.
            result = {"content": reply, "trace": trace.to_dict()}
            if tool_trace is not None:
                result["tools"] = tool_trace.to_dict()
            return result
        return {
            "content": gen.generate_text(
                prompt,
                max_new_tokens=max_new_tokens,
                **_sampling(params),
            )
        }

    if op == "search":
        tool = _require_tool(tools, "web_search", op)
        result = tool.run(
            query=params.get("query", prompt), max_results=params.get("max_results", 5)
        )
        if not result.ok:
            raise ValueError(result.content)
        return {"content": result.content, **result.data}

    if op == "shell":
        tool = _require_tool(tools, "shell", op)
        result = tool.run(command=params["command"], timeout=params.get("timeout"))
        if not result.ok and not result.data.get("command"):
            raise ValueError(result.content)
        # A non-zero exit is an answer, not a worker failure: the model asked
        # what happens when this runs, and this is what happened.
        return {"content": result.content, "ok": result.ok, **result.data}

    if op == "fetch":
        tool = _require_tool(tools, "web_fetch", op)
        result = tool.run(url=params["url"], max_chars=params.get("max_chars"))
        if not result.ok:
            raise ValueError(result.content)
        return {"content": result.content, **result.data}

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

    if op == "document":
        from .document import DocumentError, read_document

        try:
            document = read_document(
                params["document_path"], max_pages=params.get("max_pages")
            )
        except DocumentError as exc:
            # A missing optional reader is a deployment fact, not a crash: the
            # caller gets the reason and can install it or send something else.
            return {"error": str(exc), "code": "DOCUMENT_UNREADABLE"}

        text = document.to_text(max_pages=params.get("max_pages"))
        scanned = [page.number for page in document.scanned_pages]
        result = {
            "pages": len(document),
            "title": document.title,
            "scanned_pages": scanned,
            "text": text,
        }
        if prompt:
            result["content"] = gen.generate_text(
                f"{text}\n\n{prompt}",
                max_new_tokens=params.get("max_new_tokens", 256),
                **_sampling(params),
            )
        return result

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
    parser.add_argument(
        "--tools", default=os.environ.get("MODEL_TOOLS", ""),
        help="comma-separated toolsets the model may call, for example 'web,cli'",
    )
    parser.add_argument(
        "--cli-mode", default=os.environ.get("MODEL_CLI_MODE", "off"), choices=("off", "ask", "allow"),
        help="how the cli toolset decides: off refuses everything, allow runs allowlisted "
             "programs unattended, ask needs an approver and so refuses in this worker",
    )
    parser.add_argument(
        "--cli-root", default=os.environ.get("MODEL_CLI_ROOT", os.getcwd()),
        help="sandbox root for the cli toolset; no argument may resolve outside it",
    )
    parser.add_argument(
        "--cli-timeout", type=float, default=float(os.environ.get("MODEL_CLI_TIMEOUT", 30.0)),
        help="wall-clock seconds a command may run before its process group is killed",
    )
    parser.add_argument(
        "--cli-approval-timeout", type=float, default=float(os.environ.get("MODEL_CLI_APPROVAL_TIMEOUT", 30.0)),
        help="wall-clock seconds to wait for command approval in ask mode before failing closed",
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

    tools, tools_error = None, None
    if args.tools:
        try:
            from model.tools import ShellPolicy, build_registry

            approver = None
            if args.cli_mode == "ask":
                approver = make_stdio_approver(args.cli_root, args.cli_approval_timeout)

            policy = ShellPolicy(
                mode=args.cli_mode,
                root=args.cli_root,
                timeout=args.cli_timeout,
                approve=approver,
            )
            tools = build_registry(args.tools, cli_policy=policy)
        except Exception as exc:  # noqa: BLE001 - a bad toolset must not block serving
            tools_error = str(exc)

    mind, mind_error = None, None
    if args.mind:
        try:
            mind = load_mind(gen, args.mind)
        except Exception as exc:  # noqa: BLE001 - a broken mind must not block serving
            mind_error = str(exc)

    # One ready line, always: the bridge waits for exactly one.
    ready = {"ready": True, "mind": mind is not None, "tools": tools.names() if tools else []}
    if tools and "shell" in tools:
        ready["cli_mode"] = args.cli_mode
    if mind_error:
        ready["mind_error"] = mind_error
    if tools_error:
        ready["tools_error"] = tools_error
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
            result = handle(
                gen, req.get("op", "chat"), req.get("params", {}), mind=mind, tools=tools
            )
            if mind is not None and args.mind:
                # Persist after every request: a mind that only survives a clean
                # shutdown is a mind that loses its day whenever the worker dies.
                mind.save(args.mind)
            _print({"id": req_id, "ok": True, "result": result})
        except Exception as exc:  # noqa: BLE001 - never let one request kill the worker
            _print({"id": req_id, "ok": False, "error": str(exc)})


if __name__ == "__main__":
    main()
