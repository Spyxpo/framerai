/**
 * Generation route behaviour: validation, forwarding, and uploads.
 *
 * Rate limiting is covered by limits.test.js, so it is raised out of the way
 * here. The limits are read from the environment when the app module loads, and
 * `node --test` runs each file in its own process, so this does not leak.
 */

process.env.GENERATE_RATE_LIMIT_MAX = "500";
process.env.RATE_LIMIT_MAX = "1000";

const test = require("node:test");
const { after } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const request = require("supertest");

const { mockModel, loadApp } = require("./helpers");

const calls = mockModel();
const app = loadApp();

// Uploads land in the real uploads directory, so remove what the run created.
const uploaded = [];
after(() => {
  for (const relative of uploaded) {
    fs.rmSync(path.join(__dirname, "..", relative), { force: true });
  }
});

function lastCall(name) {
  return [...calls].reverse().find((c) => c.name === name);
}

// Only the size fields the caller actually set are forwarded, so the worker can
// fall through to prompt intent and then to the model's own default.
function sizeArg(call) {
  return Object.fromEntries(Object.entries(call.args[2]).filter(([, v]) => v !== undefined));
}

test("image generation forwards the prompt and no size at all", async () => {
  const res = await request(app).post("/api/generate/image").send({ prompt: "a cat" });

  assert.equal(res.status, 200);
  assert.equal(res.body.prompt, "a cat");
  const call = lastCall("generateImage");
  assert.deepEqual([call.args[0], call.args[1]], ["a cat", 1]);
  assert.deepEqual(sizeArg(call), {});
});

test("image generation forwards an explicit width and height", async () => {
  const res = await request(app)
    .post("/api/generate/image")
    .send({ prompt: "a cat", num_images: 3, width: 1024, height: 768 });

  assert.equal(res.status, 200);
  assert.deepEqual(sizeArg(lastCall("generateImage")), { width: 1024, height: 768 });
});

test("image generation forwards an aspect ratio and size tier", async () => {
  const res = await request(app)
    .post("/api/generate/image")
    .send({ prompt: "a cat", aspect: "16:9", tier: 1024, seed: 7 });

  assert.equal(res.status, 200);
  assert.deepEqual(sizeArg(lastCall("generateImage")), { aspect: "16:9", tier: 1024, seed: 7 });
});

test("the deprecated square resolution field still works", async () => {
  const res = await request(app)
    .post("/api/generate/image")
    .send({ prompt: "a cat", resolution: 512 });

  assert.equal(res.status, 200);
  assert.deepEqual(sizeArg(lastCall("generateImage")), { resolution: 512 });
});

test("width without height is rejected", async () => {
  const res = await request(app).post("/api/generate/image").send({ prompt: "a cat", width: 512 });

  assert.equal(res.status, 400);
  assert.equal(res.body.code, "VALIDATION_ERROR");
  assert.deepEqual(
    res.body.details.map((d) => d.field),
    ["height"]
  );
});

test("out-of-range dimensions and unknown ratios are rejected", async () => {
  const res = await request(app)
    .post("/api/generate/image")
    .send({ prompt: "a cat", width: 99999, height: 32, aspect: "5:1" });

  assert.equal(res.status, 400);
  assert.deepEqual(
    res.body.details.map((d) => d.field),
    ["width", "height", "aspect"]
  );
});

test("every bad field is reported in one response", async () => {
  const res = await request(app)
    .post("/api/generate/image")
    .send({ prompt: "a cat", num_images: 99, resolution: 999 });

  assert.equal(res.status, 400);
  assert.equal(res.body.code, "VALIDATION_ERROR");
  assert.deepEqual(
    res.body.details.map((d) => d.field),
    ["num_images", "resolution"]
  );
});

test("a missing prompt is rejected on every generation route", async () => {
  for (const route of ["image", "video", "audio", "code"]) {
    const res = await request(app).post(`/api/generate/${route}`).send({});
    assert.equal(res.status, 400, `${route} should reject an empty body`);
    assert.deepEqual(res.body.details, [{ field: "prompt", message: "is required" }]);
  }
});

test("a whitespace-only prompt counts as missing", async () => {
  const res = await request(app).post("/api/generate/code").send({ prompt: "   " });

  assert.equal(res.status, 400);
  assert.deepEqual(res.body.details, [{ field: "prompt", message: "is required" }]);
});

test("video frames are bounded", async () => {
  const ok = await request(app).post("/api/generate/video").send({ prompt: "x", num_frames: 32 });
  assert.equal(ok.status, 200);
  const videoArgs = lastCall("generateVideo").args;
  assert.deepEqual(videoArgs.slice(0, 2), ["x", 32]);

  const tooMany = await request(app)
    .post("/api/generate/video")
    .send({ prompt: "x", num_frames: 1000 });
  assert.equal(tooMany.status, 400);
  assert.equal(tooMany.body.details[0].field, "num_frames");
});

test("code generation defaults to python and rejects unknown languages", async () => {
  const ok = await request(app).post("/api/generate/code").send({ prompt: "sort a list" });
  assert.equal(ok.status, 200);
  const codeArgs = lastCall("generateCode").args;
  assert.deepEqual(codeArgs.slice(0, 3), ["sort a list", "python", {}]);

  const bad = await request(app)
    .post("/api/generate/code")
    .send({ prompt: "sort a list", language: "brainfuck" });
  assert.equal(bad.status, 400);
  assert.equal(bad.body.details[0].field, "language");
});

test("generation settings are forwarded, and only the ones that were set", async () => {
  const res = await request(app)
    .post("/api/generate/code")
    .send({ prompt: "sort a list", settings: { temperature: 1.25, top_k: 20 } });

  assert.equal(res.status, 200);
  assert.deepEqual(lastCall("generateCode").args[2], { temperature: 1.25, top_k: 20 });
});

test("out of range settings are reported under their own field names", async () => {
  const res = await request(app)
    .post("/api/generate/code")
    .send({ prompt: "x", settings: { temperature: 9, top_p: 5, top_k: "lots" } });

  assert.equal(res.status, 400);
  assert.deepEqual(res.body.details, [
    { field: "settings.temperature", message: "must be at most 2" },
    { field: "settings.top_p", message: "must be at most 1" },
    { field: "settings.top_k", message: "must be an integer" },
  ]);
});

test("settings that are not an object are rejected", async () => {
  const res = await request(app)
    .post("/api/generate/code")
    .send({ prompt: "x", settings: "hot" });

  assert.equal(res.status, 400);
  assert.deepEqual(res.body.details, [{ field: "settings", message: "must be an object" }]);
});

test("transcribe accepts an audio upload and rejects a missing one", async () => {
  const ok = await request(app)
    .post("/api/generate/transcribe")
    .attach("audio", Buffer.from("fake audio"), { filename: "clip.wav", contentType: "audio/wav" });

  assert.equal(ok.status, 200);
  assert.equal(ok.body.text, "transcribed");
  assert.match(ok.body.audioPath, /^\/uploads\/audio\//);
  uploaded.push(ok.body.audioPath);

  const missing = await request(app).post("/api/generate/transcribe").field("prompt", "hi");
  assert.equal(missing.status, 400);
  assert.deepEqual(missing.body.details, [{ field: "audio", message: "is required" }]);
});

test("an upload of the wrong media type is refused", async () => {
  const res = await request(app)
    .post("/api/generate/transcribe")
    .attach("audio", Buffer.from("not audio"), { filename: "notes.txt", contentType: "text/plain" });

  assert.equal(res.status, 400);
  assert.match(res.body.error, /Expected audio upload/);
});

test("understand accepts an image upload", async () => {
  const res = await request(app)
    .post("/api/generate/understand")
    .attach("image", Buffer.from("fake image"), { filename: "pic.png", contentType: "image/png" })
    .field("prompt", "what is this");

  assert.equal(res.status, 200);
  assert.equal(res.body.description, "a description");
  assert.equal(lastCall("understandImage").args[1], "what is this");
  uploaded.push(res.body.imagePath);
});
