const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const request = require("supertest");

const { mockModel, loadApp } = require("./helpers");

const calls = mockModel();
const app = loadApp();

const UPLOADS = path.join(__dirname, "..", "uploads");
const BUCKETS = ["images", "clips"];
const before = Object.fromEntries(
  BUCKETS.map((b) => [b, new Set(fs.existsSync(path.join(UPLOADS, b)) ? fs.readdirSync(path.join(UPLOADS, b)) : [])])
);

test.after(() => {
  for (const bucket of BUCKETS) {
    const dir = path.join(UPLOADS, bucket);
    if (!fs.existsSync(dir)) continue;
    for (const name of fs.readdirSync(dir)) {
      if (!before[bucket].has(name)) fs.rmSync(path.join(dir, name), { force: true });
    }
  }
});

const png = Buffer.from("89504e470d0a1a0a", "hex");
const last = (name) => [...calls].reverse().find((c) => c.name === name);

test("image-edit passes the image, mask and strength to the generator", async () => {
  const res = await request(app)
    .post("/api/generate/image-edit")
    .field("prompt", "make it night")
    .field("strength", "0.4")
    .attach("image", png, { filename: "a.png", contentType: "image/png" })
    .attach("mask", png, { filename: "m.png", contentType: "image/png" });
  assert.equal(res.status, 200);
  const [prompt, count, edit] = last("generateImage").args;
  assert.equal(prompt, "make it night");
  assert.equal(count, 1);
  assert.equal(edit.strength, 0.4);
  assert.match(edit.init_image_path, /uploads[\\/]images[\\/]/);
  assert.match(edit.mask_path, /uploads[\\/]images[\\/]/);
});

test("image-edit validates its inputs", async () => {
  const noImage = await request(app).post("/api/generate/image-edit").field("prompt", "x");
  assert.equal(noImage.status, 400);
  assert.deepEqual(noImage.body.details, [{ field: "image", message: "is required" }]);

  const badStrength = await request(app)
    .post("/api/generate/image-edit")
    .field("prompt", "x")
    .field("strength", "2")
    .attach("image", png, { filename: "a.png", contentType: "image/png" });
  assert.equal(badStrength.status, 400);
  assert.deepEqual(badStrength.body.details.map((d) => d.field), ["strength"]);
});

test("animate sends the first frame to the video generator", async () => {
  const res = await request(app)
    .post("/api/generate/animate")
    .field("prompt", "the waves roll in")
    .field("num_frames", "8")
    .attach("image", png, { filename: "a.png", contentType: "image/png" });
  assert.equal(res.status, 200);
  const [prompt, frames, size] = last("generateVideo").args;
  assert.equal(prompt, "the waves roll in");
  assert.equal(frames, 8);
  assert.match(size.first_frame_path, /uploads[\\/]images[\\/]/);
});

test("understand-video stores the clip apart from generated videos", async () => {
  const res = await request(app)
    .post("/api/generate/understand-video")
    .field("prompt", "what happens")
    .attach("video", Buffer.from("not really mp4"), { filename: "c.mp4", contentType: "video/mp4" });
  assert.equal(res.status, 200);
  assert.equal(res.body.description, "a video description");
  assert.match(res.body.videoPath, /^\/uploads\/clips\/[0-9a-f-]+\.mp4$/);
  assert.match(last("understandVideo").args[0], /uploads[\\/]clips[\\/]/);

  const wrong = await request(app)
    .post("/api/generate/understand-video")
    .attach("video", png, { filename: "a.png", contentType: "image/png" });
  assert.equal(wrong.status, 400);
});

test("image accepts sampler controls and checks the guidance interval", async () => {
  const ok = await request(app).post("/api/generate/image").send({ prompt: "x", shift: 3, guidance_interval: [0.1, 0.8] });
  assert.equal(ok.status, 200);
  const size = last("generateImage").args[2];
  assert.equal(size.shift, 3);
  assert.deepEqual(size.guidance_interval, [0.1, 0.8]);

  const bad = await request(app).post("/api/generate/image").send({ prompt: "x", guidance_interval: [0.9, 0.1] });
  assert.equal(bad.status, 400);
  assert.deepEqual(bad.body.details.map((d) => d.field), ["guidance_interval"]);
});
