/**
 * WAV fmt value validation and the audio chunk-size guard (Issue #362).
 *
 * Root cause: parseWavFile read channels, sampleRate and bitsPerSample out of
 * the fmt chunk without checking their values. The earlier hardening (#312
 * chunk walking, #347 async reads) validated structure only, so a zero in any
 * of those fields was accepted. durationSec then divided by their product, and
 * streamAudio derived its chunk size from them:
 *
 *   samplesPerChunk = Math.floor(sampleRate * AUDIO_CHUNK_DURATION_SEC)
 *   bytesPerChunk   = samplesPerChunk * channels * (bitsPerSample / 8)
 *   totalChunks     = Math.ceil(pcmData.length / bytesPerChunk)
 *
 * With bytesPerChunk === 0 that made totalChunks Infinity for non-empty audio
 * (isLast never true, so empty frames forever and done never sent) or NaN for
 * empty audio (loop skipped, so done never sent either). Both left the client
 * with no terminal frame.
 *
 * Fix, in two layers as the maintainer asked:
 *   1. parseWavFile rejects channels / sampleRate / bitsPerSample <= 0.
 *   2. streamAudio bails to its existing fallback frame when bytesPerChunk is
 *      not > 0, which still happens for a sample rate under
 *      1 / AUDIO_CHUNK_DURATION_SEC even though the parser allows it.
 *
 * Note on the boundary: AUDIO_CHUNK_DURATION_SEC is 0.5, so samplesPerChunk
 * floors to zero only for sampleRate 0 and 1. A rate of 9 yields 4 samples per
 * chunk and streams normally; the second layer exists for rates below 2, not
 * below 10.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const { mockModel, startServer, createTestWav } = require("./helpers");

// Drives the audio branch; each test points this at the file it wrote.
let audioUrl = null;
mockModel({
  processMessage: (messages, type = "text") => {
    if (type === "audio") {
      return Promise.resolve({
        type: "audio",
        content: "Here is your audio",
        metadata: { model: "test-model", url: audioUrl },
      });
    }
    return Promise.resolve({ type, content: "reply", metadata: { model: "test-model" } });
  },
});

const { parseWavFile } = require("../src/services/websocket");

// Same canonical layout as parseWavFile.test.js's makeCanonicalWav, but with
// every fmt field open so the degenerate values can be built.
function makeWav({ sampleRate = 24000, channels = 1, bitsPerSample = 16, numSamples = 100 } = {}) {
  const bytesPerFrame = Math.max(channels, 1) * 2;
  const dataSize = numSamples * bytesPerFrame;
  const buf = Buffer.alloc(44 + dataSize);
  buf.write("RIFF", 0);
  buf.writeUInt32LE(36 + dataSize, 4);
  buf.write("WAVE", 8);
  buf.write("fmt ", 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);                   // PCM
  buf.writeUInt16LE(channels, 22);
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * bytesPerFrame, 28);
  buf.writeUInt16LE(bytesPerFrame, 32);
  buf.writeUInt16LE(bitsPerSample, 34);
  buf.write("data", 36);
  buf.writeUInt32LE(dataSize, 40);
  return buf;
}

const tmpFiles = [];
function writeTmp(buf) {
  const p = path.join(os.tmpdir(), `framerai-wav362-${Math.random().toString(36).slice(2)}.wav`);
  fs.writeFileSync(p, buf);
  tmpFiles.push(p);
  return p;
}

// Files the streaming path resolves from /uploads/generated.
const generatedDir = path.join(__dirname, "..", "uploads", "generated");
const generated = [];
function writeGenerated(name, buf) {
  fs.mkdirSync(generatedDir, { recursive: true });
  const full = path.join(generatedDir, name);
  fs.writeFileSync(full, buf);
  generated.push(full);
  return `/uploads/generated/${name}`;
}

test.after(() => {
  for (const p of tmpFiles) fs.rmSync(p, { force: true });
  for (const p of generated) fs.rmSync(p, { force: true });
});

const WebSocket = require("ws");

/** Drive one audio turn and collect every frame until done or timeout. */
function audioTurn(wsUrl, { timeoutMs = 3000 } = {}) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    const received = [];
    const finish = () => {
      clearTimeout(timer);
      ws.close();
      resolve(received);
    };
    // A non-progressing loop would emit frames until this fires; the
    // assertions below would then see no done frame.
    const timer = setTimeout(finish, timeoutMs);
    ws.on("open", () =>
      ws.send(JSON.stringify({ type: "chat", content: "make audio", messageType: "audio" }))
    );
    ws.on("error", reject);
    ws.on("message", (data) => {
      const message = JSON.parse(data);
      received.push(message);
      if (message.type === "error" || (message.type === "stream" && message.done)) finish();
    });
  });
}

// ---------------------------------------------------------------------------
// Layer 1: parseWavFile rejects unusable fmt values
// ---------------------------------------------------------------------------

test("sampleRate = 0 with non-empty audio data is rejected", async () => {
  const p = writeTmp(makeWav({ sampleRate: 0, numSamples: 100 }));
  // Previously accepted, giving durationSec === Infinity.
  await assert.rejects(() => parseWavFile(p), /unusable values/i);
});

test("sampleRate = 0 with empty audio data is rejected", async () => {
  const p = writeTmp(makeWav({ sampleRate: 0, numSamples: 0 }));
  // Previously accepted, giving durationSec === NaN.
  await assert.rejects(() => parseWavFile(p), /unusable values/i);
});

test("channels = 0 is rejected", async () => {
  const p = writeTmp(makeWav({ channels: 0 }));
  await assert.rejects(() => parseWavFile(p), /unusable values/i);
});

test("bitsPerSample = 0 is rejected", async () => {
  const p = writeTmp(makeWav({ bitsPerSample: 0 }));
  await assert.rejects(() => parseWavFile(p), /unusable values/i);
});

test("all three zero at once is rejected", async () => {
  const p = writeTmp(makeWav({ sampleRate: 0, channels: 0, bitsPerSample: 0, numSamples: 0 }));
  await assert.rejects(() => parseWavFile(p), /unusable values/i);
});

test("the rejection names the offending values", async () => {
  const p = writeTmp(makeWav({ sampleRate: 0 }));
  await assert.rejects(
    () => parseWavFile(p),
    /channels=1, sampleRate=0, bitsPerSample=16/,
    "the error should say which field was unusable"
  );
});

test("an accepted file always has a finite positive durationSec", async () => {
  for (const sampleRate of [2, 9, 8000, 24000, 48000]) {
    const p = writeTmp(makeWav({ sampleRate, numSamples: 100 }));
    const result = await parseWavFile(p);
    assert.ok(
      Number.isFinite(result.durationSec) && result.durationSec > 0,
      `sampleRate=${sampleRate} gave durationSec=${result.durationSec}`
    );
  }
});

// ---------------------------------------------------------------------------
// Layer 2: the chunk-size guard, for rates the parser allows
// ---------------------------------------------------------------------------

// AUDIO_CHUNK_DURATION_SEC is 0.5, so this is the real boundary: 1 floors to
// zero samples per chunk, 2 does not. The prompt's "< 10" would only hold for
// a 0.1 second chunk.
test("sampleRate = 1 parses but cannot produce a zero-progress stream", async (t) => {
  const p = writeTmp(makeWav({ sampleRate: 1, numSamples: 100 }));
  const result = await parseWavFile(p);

  // The parser allows it: 1 is a positive rate.
  assert.equal(result.sampleRate, 1);
  // But it floors to a zero chunk size, which is what the guard is for.
  const samplesPerChunk = Math.floor(result.sampleRate * 0.5);
  assert.equal(samplesPerChunk, 0, "this is the condition the guard must catch");

  audioUrl = writeGenerated("wav362-rate-1.wav", makeWav({ sampleRate: 1, numSamples: 100 }));
  const server = await startServer();
  t.after(() => server.stop());

  const frames = await audioTurn(server.wsUrl);
  const streams = frames.filter((m) => m.type === "stream");

  assert.equal(streams.length, 1, "the guard must bail rather than emit chunk frames");
  assert.equal(streams[0].done, true, "the terminal done:true frame must still be reachable");
  assert.equal(
    streams[0].metadata?.chunkData,
    undefined,
    "the loop must not have been entered, so no chunk payload"
  );
});

test("sampleRate = 2 is the smallest rate that still streams chunks", async (t) => {
  // Few samples on purpose: at this rate a chunk is 2 bytes, and the streaming
  // loop sleeps 50 ms between chunks, so 100 samples would take ~5 s.
  audioUrl = writeGenerated("wav362-rate-2.wav", makeWav({ sampleRate: 2, numSamples: 4 }));
  const server = await startServer();
  t.after(() => server.stop());

  const frames = await audioTurn(server.wsUrl);
  const streams = frames.filter((m) => m.type === "stream");

  assert.ok(streams.length > 0, "should stream");
  assert.equal(streams[streams.length - 1].done, true, "must terminate");
  assert.ok(
    streams.some((m) => typeof m.metadata?.chunkData === "string"),
    "a positive chunk size must still produce chunk payloads"
  );
});

test("a WAV whose fmt values are unusable falls back with done:true, not silence", async (t) => {
  // The file reaches streamAudio, parseWavFile throws, and the existing catch
  // sends the fallback frame. Proves the rejection cannot hang the turn.
  audioUrl = writeGenerated("wav362-zero.wav", makeWav({ sampleRate: 0, numSamples: 100 }));
  const server = await startServer();
  t.after(() => server.stop());

  const frames = await audioTurn(server.wsUrl);
  const streams = frames.filter((m) => m.type === "stream");

  assert.equal(streams.length, 1, "exactly the fallback frame");
  assert.equal(streams[0].done, true, "the client must always get a terminal frame");
  assert.equal(streams[0].responseType, "audio");
  assert.equal(streams[0].metadata?.chunkData, undefined, "no chunk payload on the fallback");
});

test("an empty-data WAV with zero fmt values also falls back with done:true", async (t) => {
  // Previously this produced totalChunks === NaN, so the loop was skipped and
  // no frame was ever sent.
  audioUrl = writeGenerated(
    "wav362-zero-empty.wav",
    makeWav({ sampleRate: 0, numSamples: 0 })
  );
  const server = await startServer();
  t.after(() => server.stop());

  const frames = await audioTurn(server.wsUrl);
  const streams = frames.filter((m) => m.type === "stream");

  assert.equal(streams.length, 1);
  assert.equal(streams[0].done, true, "done must be reachable for empty data too");
});

// ---------------------------------------------------------------------------
// Valid behaviour is unchanged
// ---------------------------------------------------------------------------

test("a normal WAV still parses with the same values", async () => {
  const p = writeTmp(makeWav({ sampleRate: 24000, channels: 1, bitsPerSample: 16, numSamples: 240 }));
  const result = await parseWavFile(p);

  assert.equal(result.channels, 1);
  assert.equal(result.sampleRate, 24000);
  assert.equal(result.bitsPerSample, 16);
  assert.equal(result.pcmData.length, 240 * 2);
  assert.ok(Math.abs(result.durationSec - 240 / 24000) < 1e-6);
});

test("a normal WAV still streams multiple chunks and terminates", async (t) => {
  // The existing helper fixture, same as websocket-audio.test.js uses.
  audioUrl = writeGenerated("wav362-valid.wav", createTestWav(1.5, 24000));
  const server = await startServer();
  t.after(() => server.stop());

  const frames = await audioTurn(server.wsUrl, { timeoutMs: 5000 });
  const streams = frames.filter((m) => m.type === "stream");

  assert.ok(streams.length >= 3, `1.5s at 0.5s chunks should be ~3 frames, got ${streams.length}`);
  for (const [i, msg] of streams.entries()) {
    assert.equal(msg.metadata.chunk, i, "chunks in order");
    assert.equal(typeof msg.metadata.chunkData, "string", "chunk payload present");
    assert.equal(msg.metadata.sampleRate, 24000);
    assert.equal(msg.metadata.channels, 1);
    assert.equal(msg.metadata.bitsPerSample, 16);
  }
  const final = streams[streams.length - 1];
  assert.equal(final.done, true);
  assert.equal(typeof final.metadata.durationSec, "number");
  assert.ok(Number.isFinite(final.metadata.durationSec), "durationSec must be finite");
});
