/**
 * Test helpers.
 *
 * The model service is replaced before the app is loaded, so tests never spawn
 * the Python worker, never sleep on the placeholder delay, and can assert on
 * exactly what the routes pass down.
 */

const request = require("supertest");
const fs = require("node:fs");
const path = require("node:path");

const MODEL_MODULE = require.resolve("../src/services/model");

/**
 * Create a minimal valid WAV file for testing audio streaming.
 * Returns a Buffer containing a mono 16-bit PCM WAV file with the specified duration.
 */
function createTestWav(durationSec = 1.0, sampleRate = 24000) {
  const numSamples = Math.floor(sampleRate * durationSec);
  const dataSize = numSamples * 2; // 16-bit = 2 bytes per sample
  const fileSize = 36 + dataSize;

  const buffer = Buffer.alloc(44 + dataSize);

  // RIFF header
  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(fileSize, 4);
  buffer.write("WAVE", 8);

  // fmt chunk
  buffer.write("fmt ", 12);
  buffer.writeUInt32LE(16, 16); // fmt chunk size
  buffer.writeUInt16LE(1, 20);  // audio format (1 = PCM)
  buffer.writeUInt16LE(1, 22);  // channels (mono)
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * 2, 28); // byte rate
  buffer.writeUInt16LE(2, 32);  // block align
  buffer.writeUInt16LE(16, 34); // bits per sample

  // data chunk
  buffer.write("data", 36);
  buffer.writeUInt32LE(dataSize, 40);

  // PCM data: simple sine wave at 440 Hz
  for (let i = 0; i < numSamples; i++) {
    const sample = Math.sin(2 * Math.PI * 440 * i / sampleRate);
    const value = Math.floor(sample * 16384); // Scale to 16-bit range
    buffer.writeInt16LE(value, 44 + i * 2);
  }

  return buffer;
}

/**
 * Install a stub in place of src/services/model. Returns the call log so a
 * test can assert on what a route forwarded.
 *
 * Must run before the app is required, which is why every test file calls it
 * at the top. `node --test` gives each file its own process, so the stub and
 * any environment variables stay local to that file.
 */
function mockModel(overrides = {}) {
  const calls = [];

  const record = (name, result) => (...args) => {
    calls.push({ name, args });
    return Promise.resolve(typeof result === "function" ? result(...args) : result);
  };

  const targetExports = {
    processMessage: record("processMessage", (messages, type = "text", settings = {}, options = {}) => ({
      type,
      content: `reply to: ${messages[messages.length - 1].content}`,
      metadata: { model: "test-model" },
      options,
    })),
    generateImage: record("generateImage", (prompt, numImages, resolution) => ({
      id: "img-1",
      prompt,
      images: [{ id: "img-1", url: "/uploads/generated/img.png", placeholder: false }],
      metadata: { resolution, numImages, model: "test-model" },
    })),
    generateVideo: record("generateVideo", (prompt, numFramesOrOptions, size) => {
      // Handle both call forms:
      //   upstream route: generateVideo(prompt, numFrames, sizeObj, requestId)
      //   legacy/options: generateVideo(prompt, optionsObj)
      let numFrames = 16, fps = 24, width = 256, height = 256;
      if (typeof numFramesOrOptions === "number") {
        numFrames = numFramesOrOptions;
        const s = size || {};
        fps = s.fps ?? 24;
        width = s.width ?? 256;
        height = s.height ?? 256;
      } else {
        const opts = numFramesOrOptions || {};
        numFrames = opts.numFrames ?? 16;
        fps = opts.fps ?? 24;
        width = opts.width ?? 256;
        height = opts.height ?? 256;
      }
      return {
        id: "vid-1",
        prompt,
        video: { url: "/uploads/generated/vid.gif", frames: numFrames, fps, placeholder: false },
        metadata: { frames: numFrames, fps, width, height, model: "test-model" },
      };
    }),
    generateAudio: record("generateAudio", (prompt) => ({
      id: "aud-1",
      prompt,
      audio: { url: "/uploads/generated/aud.wav", placeholder: false },
      metadata: { model: "test-model" },
    })),
    generateCode: record("generateCode", (prompt, language) => ({
      id: "code-1",
      prompt,
      code: "print('hi')",
      language,
      metadata: { model: "test-model" },
    })),
    transcribeAudio: record("transcribeAudio", { text: "transcribed", metadata: { model: "test-model" } }),
    understandImage: record("understandImage", { description: "a description" }),
    // The routes bind their handlers when the app module loads, so a test
    // cannot restub an export afterwards. The sentinel prompt is how a route
    // test drives the unreadable-document branch.
    readDocument: record("readDocument", (documentPath, prompt = "") =>
      prompt === "force-unreadable"
        ? { error: "Reading PDFs needs the 'pypdf' package.", code: "DOCUMENT_UNREADABLE" }
        : {
            text: "<doc><page>1\nthe page text\n<doc_end>",
            pages: 1,
            title: "A Title",
            scannedPages: [],
            metadata: { model: "test-model" },
          }
    ),
    // The route layer re-validates and re-gates the trace as defence in depth
    // (Issue #164), so the stub has to carry the real helpers through.
    validateTrace: require(MODEL_MODULE).validateTrace,
    traceAllowed: require(MODEL_MODULE).traceAllowed,
    ...overrides,
  };

  if (require.cache[MODEL_MODULE] && require.cache[MODEL_MODULE].exports) {
    Object.assign(require.cache[MODEL_MODULE].exports, targetExports);
  } else {
    require.cache[MODEL_MODULE] = {
      id: MODEL_MODULE,
      filename: MODEL_MODULE,
      loaded: true,
      exports: targetExports,
    };
  }

  return calls;
}

/**
 * The Express app, loaded after the stub is in place.
 */
function loadApp() {
  const { createApp } = require("../src/app");
  return createApp();
}

/**
 * Start the whole server, including the WebSocket endpoint, on a free port.
 */
async function startServer() {
  const { createServer } = require("../src/app");
  const { app, server, wss } = createServer();

  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));

  return {
    app,
    server,
    wss,
    wsUrl: `ws://127.0.0.1:${server.address().port}/ws`,
    stop: () =>
      new Promise((resolve) => {
        if (wss) {
          try {
            if (wss.clients) {
              for (const client of wss.clients) {
                try {
                  client.terminate();
                } catch (_) {}
              }
            }
            wss.close();
          } catch (_) {}
        }
        if (typeof server.closeAllConnections === "function") {
          server.closeAllConnections();
        }
        server.close(resolve);
      }),
  };
}

/**
 * Create a conversation and return its id.
 */
async function newConversation(app) {
  const res = await request(app).post("/api/chat/conversations");
  return res.body.id;
}

module.exports = { mockModel, loadApp, startServer, newConversation, createTestWav };
