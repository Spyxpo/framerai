"""Tests for the inference worker's heartbeat liveness signal (Issue #278).

`_heartbeat_loop` runs on its own daemon thread, independent of whatever the
main thread is doing inside `handle()`, so the bridge can tell a worker that
is busy but healthy from one whose interpreter has gone silent. `_print` is
shared by that thread and the main request/response path, so it needs a lock
to keep two concurrent writers from interleaving partial JSON lines on
stdout - these tests exercise both the loop itself and that lock.
"""

import json
import threading

from model.serve import _heartbeat_loop, _print


def test_heartbeat_loop_emits_heartbeats_until_stopped(monkeypatch):
    calls = []
    monkeypatch.setattr("model.serve._print", lambda obj: calls.append(obj))

    stop_event = threading.Event()
    thread = threading.Thread(target=_heartbeat_loop, args=(0.01, stop_event), daemon=True)
    thread.start()

    # Let several intervals elapse so more than one heartbeat is emitted.
    stopped_in_time = stop_event.wait(0.2)
    assert not stopped_in_time  # nothing else sets this event during the sleep

    stop_event.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive(), "heartbeat thread should exit promptly once stop_event is set"
    assert len(calls) >= 2, f"expected multiple heartbeats over 0.2s at a 0.01s interval, got {calls}"
    assert all(c == {"type": "heartbeat"} for c in calls)

    calls_at_stop = len(calls)
    # Give a would-be extra tick a chance to fire; none should, since the
    # loop already observed the stop event and returned.
    threading.Event().wait(0.05)
    assert len(calls) == calls_at_stop, "no further heartbeats should be emitted after the thread has stopped"


def test_heartbeat_loop_never_prints_when_stop_event_is_already_set(monkeypatch):
    calls = []
    monkeypatch.setattr("model.serve._print", lambda obj: calls.append(obj))

    stop_event = threading.Event()
    stop_event.set()

    # Runs synchronously to completion immediately: wait() on an already-set
    # event returns True right away, so the loop body never executes.
    _heartbeat_loop(0.01, stop_event)

    assert calls == []


def test_print_is_safe_across_concurrent_threads(monkeypatch):
    """Guards against the exact failure this lock exists to prevent: two
    threads (the heartbeat thread and the main thread) writing to stdout at
    the same time and interleaving into a line neither side can parse."""
    import io

    buffer = io.StringIO()
    monkeypatch.setattr("sys.stdout", buffer)

    def writer(n):
        for i in range(25):
            _print({"type": "heartbeat", "thread": n, "seq": i})

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    lines = [line for line in buffer.getvalue().split("\n") if line]
    assert len(lines) == 8 * 25

    for line in lines:
        # A corrupted/interleaved write would fail to parse as JSON at all,
        # or would parse into something other than one clean object.
        parsed = json.loads(line)
        assert parsed["type"] == "heartbeat"
        assert isinstance(parsed["thread"], int)
        assert isinstance(parsed["seq"], int)
