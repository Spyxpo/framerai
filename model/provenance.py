"""Where a checkpoint's weights came from.

FramerAI is trained from scratch: no outside model is distilled into it, used as
a teacher, or used to initialise it. That is a property of the training history,
and a history nobody wrote down cannot be checked. This module writes it down.

Every checkpoint this codebase saves carries a ``provenance`` record. A run that
loaded nothing is a root, trained from random initialisation. A run that loaded
weights, to resume, to distil from a teacher, or as a DPO reference, lists each
one as a parent, with a fingerprint of those weights and the parent's own
record, so the whole lineage travels inside the file and can be audited without
the parent files at hand.

Loading a checkpoint whose record names another origin is refused, which is what
stops an outside teacher entering a run by accident. :func:`verify` walks a
record back to its roots; ``scripts/verify_provenance.py`` runs it on a file.

This is an audit trail, not a signature: a record can be edited by anyone who
can write the file. It answers "what did this codebase do", which is the
question a training history has to answer.
"""

import hashlib

import torch

ORIGIN = "framerai"
VERSION = 1

_parents: list = []


class ProvenanceError(RuntimeError):
    """A checkpoint came from somewhere this codebase may not train from."""


def reset() -> None:
    """Forget the parents of the current run, as at the start of a new one."""
    _parents.clear()


def fingerprint(state_dict: dict) -> str:
    """SHA-256 over every tensor's name, dtype, shape and bytes, in name order."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        digest.update(name.encode())
        if not hasattr(tensor, "detach"):
            digest.update(repr(tensor).encode())
            continue
        flat = tensor.detach().cpu().contiguous().reshape(-1)
        digest.update(str(flat.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        if flat.numel():
            digest.update(flat.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def check(payload: dict, path: str = "") -> None:
    """Refuse a checkpoint whose record names another origin."""
    record = payload.get("provenance") if isinstance(payload, dict) else None
    if record is not None and record.get("origin") != ORIGIN:
        raise ProvenanceError(
            f"{path or 'checkpoint'} records origin {record.get('origin')!r}; FramerAI trains "
            f"only from its own checkpoints, so weights from another model cannot be loaded"
        )


def note_parent(payload: dict, path: str = "", role: str = "loaded") -> dict:
    """Record that the current run loaded ``payload``'s weights."""
    check(payload, path)
    entry = {
        "role": role,
        "fingerprint": fingerprint(payload["model_state_dict"]),
        "provenance": payload.get("provenance"),
    }
    if all(p["fingerprint"] != entry["fingerprint"] or p["role"] != role for p in _parents):
        _parents.append(entry)
    return entry


def record() -> dict:
    """The provenance to write into a checkpoint saved now."""
    return {"origin": ORIGIN, "version": VERSION, "parents": [dict(p) for p in _parents]}


def verify(record_: dict | None) -> dict:
    """Walk a record to its roots.

    Returns ``status``: ``"scratch"`` when every root is a FramerAI run that
    loaded nothing, ``"unrecorded"`` when some ancestor was saved before
    provenance existed, ``"foreign"`` when any ancestor names another origin.
    """
    counts = {"roots": 0, "unrecorded": 0, "foreign": 0, "depth": 0, "teachers": 0}

    def walk(rec, depth):
        counts["depth"] = max(counts["depth"], depth)
        if rec is None:
            counts["unrecorded"] += 1
            return
        if rec.get("origin") != ORIGIN:
            counts["foreign"] += 1
            return
        parents = rec.get("parents") or []
        if not parents:
            counts["roots"] += 1
        for parent in parents:
            if parent.get("role") == "teacher":
                counts["teachers"] += 1
            walk(parent.get("provenance"), depth + 1)

    walk(record_, 0)
    if counts["foreign"]:
        status = "foreign"
    elif counts["unrecorded"]:
        status = "unrecorded"
    else:
        status = "scratch"
    return {"status": status, **counts}
