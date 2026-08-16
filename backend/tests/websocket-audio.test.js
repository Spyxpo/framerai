/**
 * WebSocket audio streaming tests.
 * 
 * Separate file because mockModel must be set before startServer is called,
 * and each test file runs in its own process with its own module cache.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const WebSocket = require("ws");
const fs = require("node:fs");
const path = require("node:path");

const { mockModel, startServer, createTestWav } = require("./helpers");

// Mock processMessage to return audio responses
mockModel({
  processMessage: (messages, type = "text") => {
    if (type === "audio") {
      return Promise.resolve({
        type: "audio",
        content: "Here is your audio",
        metadata: {
          model: "test-model",
          url: "/uploads/generated/test-audio-stream.wav",
        },
      });
    }
    return Promise.resolve({
      type,
      content: `reply to: ${messages[messages.length - 1].content}`,
      metadata: { model: "test-model" },
    });
  },
});

/**
 * Open a connection, send one frame, and collect every reply until the stream
 * finishes, an error arrives, or the wait times out.
 */
function exchange(wsUrl, frame, { timeoutMs = 4000 } = {}) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    const received = [];

    const finish = () => {
      clearTimeout(timer);
      ws.close();
      resolve(received);
    };
    const timer = setTimeout(finish, timeoutMs);

    ws.on("open", () => ws.send(JSON.stringify(frame)));
    ws.on("error", reject);
    ws.on("message", (data) => {
      const message = JSON.parse(data);
      received.push(message);
      if (message.type === "error" || (message.type === "stream" && message.done)) finish();
      if (message.type === "pong") finish();
    });
  });
}

test("audio streams PCM chunks with metadata over existing WebSocket protocol", async (t) => {
  // Create a test WAV file for audio streaming
  const testWavPath = path.join(__dirname, "..", "uploads", "generated", "test-audio-stream.wav");
  const testWavDir = path.dirname(testWavPath);
  fs.mkdirSync(testWavDir, { recursive: true });
  fs.writeFileSync(testWavPath, createTestWav(1.5, 24000)); // 1.5 second audio at 24kHz
  t.after(() => fs.rmSync(testWavPath, { force: true }));

  const server = await startServer();
  t.after(() => server.stop());

  const messages = await exchange(server.wsUrl, {
    type: "chat",
    content: "generate audio",
    messageType: "audio",
    conversationId: "audio-test",
  });

  // Verify stream messages
  const streams = messages.filter((m) => m.type === "stream");
  assert.ok(streams.length > 0, "should emit stream messages");

  // All stream messages should have responseType: "audio"
  for (const msg of streams) {
    assert.equal(msg.responseType, "audio", "each stream should have responseType: audio");
  }

  // Verify multiple chunks for 1.5s audio (chunk duration is 0.5s, expect 3 chunks)
  assert.ok(streams.length >= 3, `should emit multiple chunks, got ${streams.length}`);

  // Verify chunk structure
  for (let i = 0; i < streams.length; i++) {
    const msg = streams[i];
    assert.equal(msg.conversationId, "audio-test");
    assert.equal(typeof msg.metadata.chunk, "number", "should include chunk index");
    assert.equal(typeof msg.metadata.totalChunks, "number", "should include total chunks");
    assert.equal(typeof msg.metadata.chunkData, "string", "should include base64 PCM data");
    assert.equal(msg.metadata.sampleRate, 24000, "should include sample rate");
    assert.equal(msg.metadata.channels, 1, "should include channel count");
    assert.equal(msg.metadata.bitsPerSample, 16, "should include bits per sample");
    
    // Verify chunks arrive in order
    assert.equal(msg.metadata.chunk, i, "chunks should be in order");
  }

  // Verify first chunk has content
  assert.ok(streams[0].content, "first chunk should include content message");

  // Verify last chunk
  const lastChunk = streams[streams.length - 1];
  assert.equal(lastChunk.done, true, "last chunk should have done: true");
  assert.equal(lastChunk.metadata.url, "/uploads/generated/test-audio-stream.wav", "last chunk should include file URL");
  assert.equal(lastChunk.metadata.model, "test-model", "last chunk should include model name");
  assert.equal(typeof lastChunk.metadata.durationSec, "number", "last chunk should include duration");

  // Verify intermediate chunks don't have done flag
  for (let i = 0; i < streams.length - 1; i++) {
    assert.equal(streams[i].done, false, "intermediate chunks should not be done");
  }
});

test("audio streaming falls back gracefully when file is missing", async (t) => {
  // For this test, the mock still returns a URL but the file doesn't exist
  const server = await startServer();
  t.after(() => server.stop());

  const messages = await exchange(server.wsUrl, {
    type: "chat",
    content: "generate audio",
    messageType: "audio",
  });

  // Should receive a single non-streaming response as fallback
  const streams = messages.filter((m) => m.type === "stream");
  assert.equal(streams.length, 1, "should send single fallback message");
  assert.equal(streams[0].done, true);
  assert.equal(streams[0].responseType, "audio");
});
