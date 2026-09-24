/**
 * Regression tests for parseWavFile() in websocket.js.
 *
 * Covers two fixes, and guards that neither regresses the other.
 *
 * 1. Dynamic RIFF chunk walking.
 *    Root cause: the original implementation hardcoded that the "data" chunk
 *    starts at byte 36, which is only true when the fmt chunk is exactly 16
 *    bytes and no other chunks appear before "data". A WAV file with any
 *    metadata chunk (JUNK, LIST, …) between fmt and data would throw
 *    "WAV data chunk not found".
 *    Fix: replace the hardcoded offsets with a proper RIFF chunk iterator.
 *
 *    The "WAV with extra chunk" test is the discriminating test:
 *      old code: throws "WAV data chunk not found" (reads "JUNK" at byte 36)
 *      new code: iterates past JUNK and finds the real data chunk.
 *
 * 2. Asynchronous file reads (Issue #347).
 *    Root cause: parseWavFile() read the file with fs.readFileSync(), which
 *    blocks the single Node event loop for the whole duration of the read, so
 *    one large or slow WAV delayed every other request the process served.
 *    Fix: read via fs/promises and make parseWavFile() async.
 *
 * These tests verify that:
 *   - a canonical 44-byte WAV still parses correctly (regression guard),
 *   - a WAV with an extra chunk before "data" is handled,
 *   - malformed/truncated inputs still reject with appropriate errors,
 *   - a WAV that has no "data" chunk still rejects,
 *   - parseWavFile() is async and rejects rather than throwing synchronously,
 *   - the parsing path performs no synchronous file I/O.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

// parseWavFile must be exported for these unit tests.
// On unfixed code it is not exported, so this will be undefined
// and every test will fail with TypeError, proving the tests run.
const { parseWavFile } = require("../src/services/websocket");

// ---------------------------------------------------------------------------
// In-memory WAV builders — no helper dependency
// ---------------------------------------------------------------------------

/**
 * Build a canonical PCM WAV buffer (same layout as Python's wave module and
 * helpers.js createTestWav): RIFF(12) + fmt(24) + data header(8) + PCM.
 */
function makeCanonicalWav({ sampleRate = 24000, channels = 1, numSamples = 200 } = {}) {
  const dataSize = numSamples * channels * 2; // 16-bit
  const buf = Buffer.alloc(44 + dataSize);
  buf.write("RIFF", 0);
  buf.writeUInt32LE(36 + dataSize, 4);
  buf.write("WAVE", 8);
  buf.write("fmt ", 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);                       // PCM
  buf.writeUInt16LE(channels, 22);
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * channels * 2, 28);
  buf.writeUInt16LE(channels * 2, 32);
  buf.writeUInt16LE(16, 34);
  buf.write("data", 36);
  buf.writeUInt32LE(dataSize, 40);
  // PCM bytes: zeros are fine for a shape test
  return buf;
}

/**
 * Build a WAV with an extra metadata chunk between fmt and data.
 * extraChunkId must be exactly 4 ASCII characters (e.g. "JUNK", "LIST").
 *
 * Layout:  RIFF(12) + fmt(24) + extra(8+extraPayload.length) + data(8+dataSize)
 *
 * The "data" chunk starts at byte 36 + 8 + extraPayload.length, NOT at 36.
 * The old code reads bytes 36-40 as the "data" marker and finds the extra
 * chunk ID there instead — exactly the bug.
 */
function makeWavWithExtraChunk({
  extraChunkId = "JUNK",
  extraPayload = Buffer.alloc(8),
  sampleRate = 24000,
  numSamples = 200,
} = {}) {
  const dataSize = numSamples * 2; // mono 16-bit
  const extraSize = extraPayload.length;
  // Per the RIFF spec, odd-sized chunks are followed by a silent padding byte so
  // that the next chunk starts on a 2-byte boundary.
  const extraPad = extraSize % 2;
  // Total: RIFF header(12) + fmt(24) + extra(8+extraSize+extraPad) + data(8+dataSize)
  const total = 12 + 24 + 8 + extraSize + extraPad + 8 + dataSize;
  const buf = Buffer.alloc(total);

  // RIFF container
  buf.write("RIFF", 0);
  buf.writeUInt32LE(total - 8, 4);
  buf.write("WAVE", 8);

  // fmt chunk at offset 12
  buf.write("fmt ", 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);           // PCM
  buf.writeUInt16LE(1, 22);           // mono
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * 2, 28);
  buf.writeUInt16LE(2, 32);
  buf.writeUInt16LE(16, 34);

  // extra chunk at offset 36 — this is where the old code looks for "data"
  buf.write(extraChunkId.slice(0, 4).padEnd(4, " "), 36);
  buf.writeUInt32LE(extraSize, 40);
  extraPayload.copy(buf, 44);
  // padding byte (if odd) is already zero from Buffer.alloc

  // data chunk after extra payload + any padding byte
  const dataOffset = 44 + extraSize + extraPad;
  buf.write("data", dataOffset);
  buf.writeUInt32LE(dataSize, dataOffset + 4);

  return buf;
}

// Write a buffer to a unique temp file; return its path.
function writeTmp(buf) {
  const p = path.join(os.tmpdir(), `framerai-wav-test-${Date.now()}-${Math.random().toString(36).slice(2)}.wav`);
  fs.writeFileSync(p, buf);
  return p;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test("canonical WAV (44-byte header) parses correctly", async (t) => {
  const p = writeTmp(makeCanonicalWav({ sampleRate: 24000, numSamples: 240 }));
  t.after(() => fs.rmSync(p, { force: true }));

  const result = await parseWavFile(p);

  assert.equal(result.channels, 1, "channels should be 1");
  assert.equal(result.sampleRate, 24000, "sampleRate should be 24000");
  assert.equal(result.bitsPerSample, 16, "bitsPerSample should be 16");
  assert.equal(result.pcmData.length, 240 * 2, "pcmData length should match numSamples * bytesPerSample");
  assert.ok(result.durationSec > 0, "durationSec should be positive");
  assert.ok(Math.abs(result.durationSec - 240 / 24000) < 1e-6, "durationSec should match samples/sampleRate");
});

// DISCRIMINATING REGRESSION TEST:
// Old code: buffer.toString("ascii", 36, 40) == "JUNK" → throws "WAV data chunk not found"
// New code: iterates past JUNK, finds "data" → parses successfully
test("WAV with extra JUNK chunk before data parses correctly", async (t) => {
  // 9-byte payload is odd, so the chunk walker must add a padding byte:
  // advance = 8 + 9 + 1 = 18. Without the (chunkSize % 2) term it would land
  // one byte inside the next chunk and misread the "data" marker.
  const p = writeTmp(makeWavWithExtraChunk({ extraChunkId: "JUNK", extraPayload: Buffer.alloc(9) }));
  t.after(() => fs.rmSync(p, { force: true }));

  const result = await parseWavFile(p);

  assert.equal(result.channels, 1, "channels should be 1");
  assert.equal(result.sampleRate, 24000, "sampleRate should be 24000");
  assert.equal(result.bitsPerSample, 16, "bitsPerSample should be 16");
  assert.equal(result.pcmData.length, 200 * 2, "pcmData length should be correct");
  assert.ok(result.durationSec > 0, "durationSec should be positive");
});

test("WAV with extra LIST chunk before data parses correctly", async (t) => {
  // LIST is a real-world metadata chunk used by many DAWs and encoders.
  const listPayload = Buffer.from("INFOtest metadata padding\x00", "ascii");
  const p = writeTmp(makeWavWithExtraChunk({ extraChunkId: "LIST", extraPayload: listPayload }));
  t.after(() => fs.rmSync(p, { force: true }));

  const result = await parseWavFile(p);

  assert.equal(result.channels, 1);
  assert.equal(result.sampleRate, 24000);
  assert.ok(result.pcmData.length > 0, "pcmData should be non-empty");
});

test("truncated/malformed WAV rejects with an error", async (t) => {
  // 8 bytes is not enough for a valid RIFF container
  const p = writeTmp(Buffer.from("RIFF\x00\x00\x00\x00", "ascii"));
  t.after(() => fs.rmSync(p, { force: true }));

  await assert.rejects(() => parseWavFile(p), /WAV file too small|not a valid WAV/i);
});

test("WAV with truncated chunk (declared size exceeds buffer) rejects", async (t) => {
  const buf = makeCanonicalWav({ numSamples: 100 });
  // Overwrite the data chunk size to a huge value
  buf.writeUInt32LE(0x7fffffff, 40);
  const p = writeTmp(buf);
  t.after(() => fs.rmSync(p, { force: true }));

  await assert.rejects(() => parseWavFile(p), /exceeds file size|WAV/i);
});

test("WAV with no data chunk rejects with 'WAV data chunk not found'", async (t) => {
  // Build a buffer with RIFF/WAVE/fmt but no data chunk. The buffer is padded
  // to 44 bytes so the old code's "< 44" guard does not fire first, and we can
  // confirm both old and new code throw for the right reason.
  const sampleRate = 24000;
  const buf = Buffer.alloc(44); // RIFF(12) + fmt(24) + 8 zero bytes (not "data")
  buf.write("RIFF", 0);
  buf.writeUInt32LE(36, 4);
  buf.write("WAVE", 8);
  buf.write("fmt ", 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * 2, 28);
  buf.writeUInt16LE(2, 32);
  buf.writeUInt16LE(16, 34);
  // bytes 36-43: zeros — old code reads "\x00\x00\x00\x00" != "data"
  //               new code: unknown chunk ID with size 0, no "data" found
  const p = writeTmp(buf);
  t.after(() => fs.rmSync(p, { force: true }));

  await assert.rejects(() => parseWavFile(p), /WAV data chunk not found/);
});

test("non-WAV file rejects with 'Not a valid WAV file'", async (t) => {
  // Buffer must be >= 44 bytes so the old code's "< 44" guard does not
  // fire before the RIFF signature check.
  const buf = Buffer.alloc(44, 0xff); // all 0xff — first 4 bytes != "RIFF"
  const p = writeTmp(buf);
  t.after(() => fs.rmSync(p, { force: true }));

  await assert.rejects(() => parseWavFile(p), /Not a valid WAV file/i);
});

test("file with valid RIFF but wrong WAVE identifier rejects", async (t) => {
  // The old implementation never checked bytes 8-11 (the WAVE identifier).
  // It read channels/sampleRate/bitsPerSample from the canonical absolute offsets
  // and checked for "data" at byte 36 — all of which land in valid positions here —
  // so it returned successfully despite the file not being a WAV.
  // The new code's WAVE check catches this and throws.
  const buf = makeCanonicalWav({ numSamples: 10 });
  buf.write("AIFF", 8); // overwrite "WAVE" with a different RIFF form type
  const p = writeTmp(buf);
  t.after(() => fs.rmSync(p, { force: true }));

  await assert.rejects(() => parseWavFile(p), /not a valid WAV file/i);
});

// ---------------------------------------------------------------------------
// Asynchronous reads (Issue #347)
// ---------------------------------------------------------------------------

test("parseWavFile returns a Promise", async (t) => {
  const p = writeTmp(makeCanonicalWav({ numSamples: 64 }));
  t.after(() => fs.rmSync(p, { force: true }));

  const pending = parseWavFile(p);
  assert.equal(typeof pending?.then, "function", "parseWavFile must return a Promise");
  await pending;
});

// DISCRIMINATING TEST for Issue #347: trips on any sync read primitive a
// blocking implementation would reach for. On the pre-fix code readFileSync
// fires and `calls` is non-empty.
test("parsing performs no synchronous file I/O", async (t) => {
  const p = writeTmp(makeCanonicalWav({ numSamples: 120 }));
  t.after(() => fs.rmSync(p, { force: true }));

  // writeFileSync is deliberately not watched — the fixture helpers need it.
  const watched = ["readFileSync", "existsSync", "openSync", "readSync"];
  const calls = [];
  const originals = {};
  for (const name of watched) {
    originals[name] = fs[name];
    fs[name] = (...args) => {
      calls.push(name);
      return originals[name](...args);
    };
  }
  t.after(() => {
    for (const name of watched) fs[name] = originals[name];
  });

  const result = await parseWavFile(p);

  assert.equal(result.pcmData.length, 120 * 2, "parse should still succeed");
  assert.deepEqual(calls, [], `parseWavFile must not use sync fs APIs, saw: ${calls.join(", ")}`);
});

test("validation failures reject rather than throwing synchronously", async (t) => {
  const p = writeTmp(Buffer.alloc(44, 0xff)); // first 4 bytes != "RIFF"
  t.after(() => fs.rmSync(p, { force: true }));

  // An async function must never throw synchronously; callers attach .catch()
  // to the returned promise and would miss a synchronous throw entirely.
  let pending;
  assert.doesNotThrow(() => {
    pending = parseWavFile(p);
  }, "must not throw synchronously");

  await assert.rejects(pending, /Not a valid WAV file/i);
});

test("a missing file rejects instead of throwing synchronously", async () => {
  const missing = path.join(os.tmpdir(), `framerai-wav-absent-${Math.random().toString(36).slice(2)}.wav`);

  await assert.rejects(() => parseWavFile(missing), /ENOENT|no such file/i);
});

// ---------------------------------------------------------------------------
// Hardening preserved across the async change
// ---------------------------------------------------------------------------

test("data chunk before fmt chunk rejects", async (t) => {
  // RIFF(12) + data(8+4) + fmt(8+16) — the walker reaches "data" with no
  // format read yet, so channels is still null.
  const dataSize = 4;
  const buf = Buffer.alloc(12 + 8 + dataSize + 8 + 16);
  buf.write("RIFF", 0);
  buf.writeUInt32LE(buf.length - 8, 4);
  buf.write("WAVE", 8);
  buf.write("data", 12);
  buf.writeUInt32LE(dataSize, 16);
  buf.write("fmt ", 20 + dataSize);
  buf.writeUInt32LE(16, 24 + dataSize);
  const p = writeTmp(buf);
  t.after(() => fs.rmSync(p, { force: true }));

  await assert.rejects(() => parseWavFile(p), /data chunk found before fmt chunk/i);
});

test("fmt chunk smaller than 16 bytes rejects", async (t) => {
  // RIFF(12) + fmt(8+14) + data(8+4). 16 is the PCM spec minimum; a shorter
  // fmt would make the channels/sampleRate reads run past the chunk.
  const buf = Buffer.alloc(12 + 8 + 14 + 8 + 4);
  buf.write("RIFF", 0);
  buf.writeUInt32LE(buf.length - 8, 4);
  buf.write("WAVE", 8);
  buf.write("fmt ", 12);
  buf.writeUInt32LE(14, 16);
  buf.write("data", 34);
  buf.writeUInt32LE(4, 38);
  const p = writeTmp(buf);
  t.after(() => fs.rmSync(p, { force: true }));

  await assert.rejects(() => parseWavFile(p), /fmt chunk too small/i);
});
