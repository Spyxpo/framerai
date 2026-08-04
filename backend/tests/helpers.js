/**
 * Test helpers.
 *
 * The model service is replaced before the app is loaded, so tests never spawn
 * the Python worker, never sleep on the placeholder delay, and can assert on
 * exactly what the routes pass down.
 */

const request = require("supertest");

const MODEL_MODULE = require.resolve("../src/services/model");

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

  const exports = {
    processMessage: record("processMessage", (messages, type = "text") => ({
      type,
      content: `reply to: ${messages[messages.length - 1].content}`,
      metadata: { model: "test-model" },
    })),
    generateImage: record("generateImage", (prompt, numImages, resolution) => ({
      id: "img-1",
      prompt,
      images: [{ id: "img-1", url: "/uploads/generated/img.png", placeholder: false }],
      metadata: { resolution, numImages, model: "test-model" },
    })),
    generateVideo: record("generateVideo", (prompt, numFrames) => ({
      id: "vid-1",
      prompt,
      video: { url: "/uploads/generated/vid.gif", frames: numFrames, placeholder: false },
      metadata: { frames: numFrames, model: "test-model" },
    })),
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
    ...overrides,
  };

  require.cache[MODEL_MODULE] = {
    id: MODEL_MODULE,
    filename: MODEL_MODULE,
    loaded: true,
    exports,
  };

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
  const { app, server } = createServer();

  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));

  return {
    app,
    server,
    wsUrl: `ws://127.0.0.1:${server.address().port}/ws`,
    stop: () => new Promise((resolve) => server.close(resolve)),
  };
}

/**
 * Create a conversation and return its id.
 */
async function newConversation(app) {
  const res = await request(app).post("/api/chat/conversations");
  return res.body.id;
}

module.exports = { mockModel, loadApp, startServer, newConversation };
