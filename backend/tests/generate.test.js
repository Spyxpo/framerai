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

test("image generation forwards the prompt and defaults", async () => {
  const res = await request(app).post("/api/generate/image").send({ prompt: "a cat" });

  assert.equal(res.status, 200);
  assert.equal(res.body.prompt, "a cat");
  assert.deepEqual(lastCall("generateImage").args, ["a cat", 1, 256]);
});

test("image generation forwards explicit count and resolution", async () => {
  const res = await request(app)
    .post("/api/generate/image")
    .send({ prompt: "a cat", num_images: 3, resolution: 512 });

  assert.equal(res.status, 200);
  assert.deepEqual(lastCall("generateImage").args, ["a cat", 3, 512]);
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
  assert.deepEqual(lastCall("generateVideo").args, ["x", 32]);

  const tooMany = await request(app)
    .post("/api/generate/video")
    .send({ prompt: "x", num_frames: 1000 });
  assert.equal(tooMany.status, 400);
  assert.equal(tooMany.body.details[0].field, "num_frames");
});

test("code generation defaults to python and rejects unknown languages", async () => {
  const ok = await request(app).post("/api/generate/code").send({ prompt: "sort a list" });
  assert.equal(ok.status, 200);
  assert.deepEqual(lastCall("generateCode").args, ["sort a list", "python"]);

  const bad = await request(app)
    .post("/api/generate/code")
    .send({ prompt: "sort a list", language: "brainfuck" });
  assert.equal(bad.status, 400);
  assert.equal(bad.body.details[0].field, "language");
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
