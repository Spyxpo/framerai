/**
 * Simulated-streaming output amplification (Issue #364).
 *
 * Root cause: when the worker streams no deltas, the chat handler simulated
 * streaming by splitting the reply on spaces and sending one frame per word.
 * Every frame carries the whole accumulated prefix, so the bytes on the wire
 * grew with the square of the word count:
 *
 *   frame 1 = "a", frame 2 = "a a", frame 3 = "a a a", … frame N = whole reply
 *
 * In mock mode the reply echoes the user's prompt, so the client controlled N.
 * One ~8 KB frame produced ~4,000 frames / ~16 MB over ~174 s, and the
 * generation rate limit bounds how often a turn starts, not how long it runs.
 *
 * Fix: group words so a reply arrives in at most MAX_SIMULATED_FRAMES frames.
 * Frames stay cumulative prefixes (the wire protocol is unchanged — the client
 * replaces its message content with each frame's content, and
 * websocket.test.js asserts each frame extends the previous one), the last
 * frame is still the complete reply, and nothing is truncated.
 *
 * These tests drive the fallback by stubbing processMessage to return a long
 * reply with no deltas, which is the same code path mock mode reaches through
 * mockChat but with an exact, deterministic word count.
 *
 * These tests verify that:
 *   - the audit case is bounded in frames, bytes and wall clock,
 *   - the terminal frame still carries the complete reply,
 *   - frames remain cumulative prefixes,
 *   - short replies are unchanged, one frame per word,
 *   - output grows linearly with reply length, not quadratically,
 *   - the worker-streaming path is untouched.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const WebSocket = require("ws");

const { mockModel, startServer } = require("./helpers");

mockModel();

const CONV_ID = "3fa85f64-5717-4562-b3fc-2c963f66afa6";

// Mirrors MAX_SIMULATED_FRAMES in src/services/websocket.js.
const MAX_SIMULATED_FRAMES = 64;

/** Reply with `content` and stream no deltas: the fallback path. */
function nonStreamingModel(content) {
  return {
    processMessage: async () => ({ type: "text", content, metadata: { model: "test-model" } }),
  };
}

/** Reply that streams deltas first: the worker path, which must be untouched. */
function streamingModel(content, deltas) {
  return {
    processMessage: async (messages, type, settings, options) => {
      for (const delta of deltas) options?.onStream?.({ delta });
      return { type: "text", content, metadata: { model: "test-model" } };
    },
  };
}

/** Run one turn, recording every frame plus its raw size on the wire. */
function turn(wsUrl, { timeoutMs = 30000 } = {}) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    const frames = [];
    let bytes = 0;
    const startedAt = Date.now();

    const finish = (reason) => {
      clearTimeout(timer);
      ws.close();
      resolve({ frames, bytes, ms: Date.now() - startedAt, reason });
    };
    const timer = setTimeout(() => finish("TIMEOUT"), timeoutMs);

    ws.on("open", () =>
      ws.send(JSON.stringify({ type: "chat", content: "hi", conversationId: CONV_ID }))
    );
    ws.on("error", reject);
    ws.on("message", (data) => {
      bytes += data.length;
      const message = JSON.parse(data);
      frames.push(message);
      if (message.type === "error" || (message.type === "stream" && message.done)) finish("done");
    });
  });
}

const streamsOf = (frames) => frames.filter((m) => m.type === "stream");
const words = (n) => Array(n).fill("a").join(" ");

// ---------------------------------------------------------------------------
// The audit case
// ---------------------------------------------------------------------------

// DISCRIMINATING TEST. Before the fix this emitted ~4,000 frames and ~16 MB
// over ~174 s, so it would fail on frames, on bytes and on the timeout.
test("a 4000-word reply is bounded in frames, bytes and time", async (t) => {
  const reply = words(4000);
  mockModel(nonStreamingModel(reply));
  const server = await startServer();
  t.after(() => server.stop());

  const { frames, bytes, ms, reason } = await turn(server.wsUrl);
  const streams = streamsOf(frames);

  assert.equal(reason, "done", "the turn must finish rather than hit the timeout");
  assert.ok(
    streams.length <= MAX_SIMULATED_FRAMES,
    `expected at most ${MAX_SIMULATED_FRAMES} frames, got ${streams.length}`
  );
  assert.ok(
    bytes < 1024 * 1024,
    `expected well under 1 MB on the wire, got ${(bytes / 1048576).toFixed(2)} MB`
  );
  assert.ok(ms < 20000, `expected the turn to finish quickly, took ${ms} ms`);
});

test("the terminal frame still carries the complete reply", async (t) => {
  const reply = words(4000);
  mockModel(nonStreamingModel(reply));
  const server = await startServer();
  t.after(() => server.stop());

  const streams = streamsOf((await turn(server.wsUrl)).frames);
  const final = streams[streams.length - 1];

  assert.equal(final.done, true, "the last frame must be the terminal one");
  assert.equal(final.content, reply, "the reply must arrive complete, not truncated");
  assert.equal(final.content.split(" ").length, 4000, "every word must be present");
  assert.deepEqual(final.metadata, { model: "test-model" }, "terminal metadata unchanged");
});

test("frames remain cumulative prefixes", async (t) => {
  mockModel(nonStreamingModel(words(500)));
  const server = await startServer();
  t.after(() => server.stop());

  const streams = streamsOf((await turn(server.wsUrl)).frames);

  assert.ok(streams.length > 1, "a long reply should still stream in several frames");
  for (let i = 1; i < streams.length; i++) {
    assert.ok(
      streams[i].content.startsWith(streams[i - 1].content),
      `frame ${i} must extend frame ${i - 1}, which the client relies on`
    );
  }
  // Intermediate frames carry no metadata; only the terminal one does.
  for (const frame of streams.slice(0, -1)) {
    assert.equal(frame.done, false);
    assert.equal(frame.metadata, undefined);
  }
});

// ---------------------------------------------------------------------------
// Growth is linear, not quadratic
// ---------------------------------------------------------------------------

test("output grows with reply length, not with its square", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  mockModel(nonStreamingModel(words(500)));
  const small = await turn(server.wsUrl);

  mockModel(nonStreamingModel(words(4000)));
  const large = await turn(server.wsUrl);

  // Both must actually finish, otherwise a turn cut short by the timeout would
  // under-report its bytes and make quadratic growth look linear.
  assert.equal(small.reason, "done", "the 500-word turn must complete");
  assert.equal(large.reason, "done", "the 4000-word turn must complete");

  // The reply is 8x longer. Linear growth keeps the ratio near 8; the old
  // quadratic behaviour made it ~64x.
  const ratio = large.bytes / small.bytes;
  assert.ok(
    ratio < 16,
    `8x the reply length should not cost ${ratio.toFixed(0)}x the bytes (quadratic): ` +
      `${small.bytes} -> ${large.bytes}`
  );
});

// ---------------------------------------------------------------------------
// Normal behaviour is unchanged
// ---------------------------------------------------------------------------

test("a short reply still streams one frame per word", async (t) => {
  const reply = "reply to: hello there friend";
  mockModel(nonStreamingModel(reply));
  const server = await startServer();
  t.after(() => server.stop());

  const streams = streamsOf((await turn(server.wsUrl)).frames);

  assert.equal(streams.length, reply.split(" ").length, "unchanged for short replies");
  assert.deepEqual(
    streams.map((m) => m.content),
    ["reply", "reply to:", "reply to: hello", "reply to: hello there", "reply to: hello there friend"]
  );
  assert.equal(streams[streams.length - 1].done, true);
});

test("a reply at exactly the frame cap is still one frame per word", async (t) => {
  mockModel(nonStreamingModel(words(MAX_SIMULATED_FRAMES)));
  const server = await startServer();
  t.after(() => server.stop());

  const streams = streamsOf((await turn(server.wsUrl)).frames);

  assert.equal(streams.length, MAX_SIMULATED_FRAMES, "the cap itself must not group words");
  assert.equal(streams[0].content, "a", "first frame is still a single word");
});

test("the worker streaming path is unaffected", async (t) => {
  const deltas = ["Hello", " world", " again"];
  mockModel(streamingModel("Hello world again", deltas));
  const server = await startServer();
  t.after(() => server.stop());

  const streams = streamsOf((await turn(server.wsUrl)).frames);

  // One frame per delta from onStream, then the terminal frame — the fallback
  // never runs, so the cap plays no part here.
  assert.equal(streams.length, deltas.length + 1);
  assert.equal(streams[0].content, "Hello");
  assert.equal(streams[0].done, false);
  const final = streams[streams.length - 1];
  assert.equal(final.done, true);
  assert.equal(final.content, "Hello world again");
  assert.deepEqual(final.metadata, { model: "test-model" });
});
