const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const request = require("supertest");

const { mockModel, loadApp, createTestWav } = require("./helpers");

const calls = mockModel();
const app = loadApp();

const AUDIO_DIR = path.join(__dirname, "..", "uploads", "audio");
const before = new Set(fs.existsSync(AUDIO_DIR) ? fs.readdirSync(AUDIO_DIR) : []);

// The routes store each reference clip; remove the ones these tests made.
test.after(() => {
  if (!fs.existsSync(AUDIO_DIR)) return;
  for (const name of fs.readdirSync(AUDIO_DIR)) {
    if (!before.has(name)) fs.rmSync(path.join(AUDIO_DIR, name), { force: true });
  }
});

const wav = createTestWav(2.0);

function clone(fields) {
  let req = request(app).post("/api/generate/voice");
  for (const [k, v] of Object.entries(fields)) req = req.field(k, v);
  return req.attach("reference", wav, { filename: "me.wav", contentType: "audio/wav" });
}

test("a consented clone returns watermarked speech", async () => {
  const res = await clone({ prompt: "hello there", consent: "true", reference_text: "this is me" });
  assert.equal(res.status, 200);
  assert.equal(res.body.metadata.cloned, true);
  assert.equal(res.body.metadata.watermarked, true);
  const call = [...calls].reverse().find((c) => c.name === "cloneVoice");
  assert.match(call.args[0], /uploads[\\/]audio[\\/]/);
  assert.equal(call.args[1], "hello there");
  assert.equal(call.args[2].referenceText, "this is me");
});

test("cloning without consent is refused", async () => {
  const denied = await clone({ prompt: "hello", consent: "false" });
  assert.equal(denied.status, 400);
  assert.match(denied.body.error, /consent/);

  const missing = await clone({ prompt: "hello" });
  assert.equal(missing.status, 400);
  assert.deepEqual(missing.body.details.map((d) => d.field), ["consent"]);
});

test("cloning needs a reference clip and a prompt", async () => {
  const noFile = await request(app).post("/api/generate/voice").field("prompt", "hi").field("consent", "true");
  assert.equal(noFile.status, 400);
  assert.deepEqual(noFile.body.details, [{ field: "reference", message: "is required" }]);

  const noPrompt = await clone({ consent: "true" });
  assert.equal(noPrompt.status, 400);
  assert.deepEqual(noPrompt.body.details.map((d) => d.field), ["prompt"]);
});

test("a model that cannot clone says so", async () => {
  const res = await clone({ prompt: "force-refusal", consent: "true" });
  assert.equal(res.status, 400);
  assert.match(res.body.error, /rvq_lm/);
});

test("a non-audio reference is refused", async () => {
  const res = await request(app)
    .post("/api/generate/voice")
    .field("prompt", "hi")
    .field("consent", "true")
    .attach("reference", Buffer.from("text"), { filename: "a.txt", contentType: "text/plain" });
  assert.equal(res.status, 400);
  assert.match(res.body.error, /Expected audio upload/);
});

test("verify-audio reports the watermark check", async () => {
  const res = await request(app)
    .post("/api/generate/verify-audio")
    .attach("audio", wav, { filename: "clip.wav", contentType: "audio/wav" });
  assert.equal(res.status, 200);
  assert.deepEqual(res.body, { watermarked: true, score: 14.2, threshold: 4 });

  const missing = await request(app).post("/api/generate/verify-audio");
  assert.equal(missing.status, 400);
});
