/**
 * Regression tests for parseWavFile() in websocket.js.
 *
 * Root cause: the original implementation hardcoded that the "data" chunk
 * starts at byte 36, which is only true when the fmt chunk is exactly 16 bytes
 * and no other chunks appear before "data". A WAV file with any metadata chunk
 * (JUNK, LIST, …) between fmt and data would throw "WAV data chunk not found".
 *
 * Fix: replace the hardcoded offsets with a proper RIFF chunk iterator.
 *
 * These tests verify that:
 *   - a canonical 44-byte WAV still parses correctly (regression guard),
 *   - a WAV with an extra chunk before "data" is now handled,
 *   - malformed/truncated inputs still throw appropriate errors,
 *   - a WAV that has no "data" chunk still throws.
 *
 * The "WAV with extra chunk" test (test 2) is the discriminating test:
 *   old code: throws "WAV data chunk not found" (reads "JUNK" at byte 36)
 *   new code: iterates past JUNK and finds the real data chunk.
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

test("canonical WAV (44-byte header) parses correctly", (t) => {
  const p = writeTmp(makeCanonicalWav({ sampleRate: 24000, numSamples: 240 }));
  t.after(() => fs.rmSync(p, { force: true }));

  const result = parseWavFile(p);

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
test("WAV with extra JUNK chunk before data parses correctly", (t) => {
  // 9-byte payload is odd, so the chunk walker must add a padding byte:
  // advance = 8 + 9 + 1 = 18. Without the (chunkSize % 2) term it would land
  // one byte inside the next chunk and misread the "data" marker.
  const p = writeTmp(makeWavWithExtraChunk({ extraChunkId: "JUNK", extraPayload: Buffer.alloc(9) }));
  t.after(() => fs.rmSync(p, { force: true }));

  const result = parseWavFile(p);

  assert.equal(result.channels, 1, "channels should be 1");
  assert.equal(result.sampleRate, 24000, "sampleRate should be 24000");
  assert.equal(result.bitsPerSample, 16, "bitsPerSample should be 16");
  assert.equal(result.pcmData.length, 200 * 2, "pcmData length should be correct");
  assert.ok(result.durationSec > 0, "durationSec should be positive");
});

test("WAV with extra LIST chunk before data parses correctly", (t) => {
  // LIST is a real-world metadata chunk used by many DAWs and encoders.
  const listPayload = Buffer.from("INFOtest metadata padding\x00", "ascii");
  const p = writeTmp(makeWavWithExtraChunk({ extraChunkId: "LIST", extraPayload: listPayload }));
  t.after(() => fs.rmSync(p, { force: true }));

  const result = parseWavFile(p);

  assert.equal(result.channels, 1);
  assert.equal(result.sampleRate, 24000);
  assert.ok(result.pcmData.length > 0, "pcmData should be non-empty");
});

test("truncated/malformed WAV throws an error", (t) => {
  // 8 bytes is not enough for a valid RIFF container
  const p = writeTmp(Buffer.from("RIFF\x00\x00\x00\x00", "ascii"));
  t.after(() => fs.rmSync(p, { force: true }));

  assert.throws(() => parseWavFile(p), /WAV file too small|not a valid WAV/i);
});

test("WAV with truncated chunk (declared size exceeds buffer) throws", (t) => {
  const buf = makeCanonicalWav({ numSamples: 100 });
  // Overwrite the data chunk size to a huge value
  buf.writeUInt32LE(0x7fffffff, 40);
  const p = writeTmp(buf);
  t.after(() => fs.rmSync(p, { force: true }));

  assert.throws(() => parseWavFile(p), /exceeds file size|WAV/i);
});

test("WAV with no data chunk throws 'WAV data chunk not found'", (t) => {
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

  assert.throws(() => parseWavFile(p), /WAV data chunk not found/);
});

test("non-WAV file throws 'Not a valid WAV file'", (t) => {
  // Buffer must be >= 44 bytes so the old code's "< 44" guard does not
  // fire before the RIFF signature check.
  const buf = Buffer.alloc(44, 0xff); // all 0xff — first 4 bytes != "RIFF"
  const p = writeTmp(buf);
  t.after(() => fs.rmSync(p, { force: true }));

  assert.throws(() => parseWavFile(p), /Not a valid WAV file/i);
});

test("file with valid RIFF but wrong WAVE identifier throws", (t) => {
  // The old implementation never checked bytes 8-11 (the WAVE identifier).
  // It read channels/sampleRate/bitsPerSample from the canonical absolute offsets
  // and checked for "data" at byte 36 — all of which land in valid positions here —
  // so it returned successfully despite the file not being a WAV.
  // The new code's WAVE check catches this and throws.
  const buf = makeCanonicalWav({ numSamples: 10 });
  buf.write("AIFF", 8); // overwrite "WAVE" with a different RIFF form type
  const p = writeTmp(buf);
  t.after(() => fs.rmSync(p, { force: true }));

  assert.throws(() => parseWavFile(p), /not a valid WAV file/i);
});
