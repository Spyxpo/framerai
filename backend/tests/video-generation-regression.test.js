/**
 * Video route: the size, frame rate and seed the caller asked for.
 *
 * The worker accepted all of these all along and the route exposed none of
 * them, so the tests assert both halves: that the route forwards each one to
 * the service, and that the response reports back what was resolved.
 *
 * Rate limiting is covered by limits.test.js, so it is raised out of the way
 * here. The limits are read from the environment when the app module loads, and
 * `node --test` runs each file in its own process, so this does not leak.
 */

process.env.GENERATE_RATE_LIMIT_MAX = "500";
process.env.RATE_LIMIT_MAX = "1000";

const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");

const { mockModel, loadApp } = require("./helpers");

const calls = mockModel();
const app = loadApp();
// The bounds are the route's own, so they are read from it rather than
// restated here, where they would drift the moment a limit moved.
const { MAX_FRAMES, MIN_FPS, MAX_FPS, MIN_DIMENSION } = require("../src/routes/generate");

function lastCall(name) {
  return [...calls].reverse().find((c) => c.name === name);
}

// The route calls generateVideo(prompt, numFrames, size, requestId), so the
// size arguments all arrive together in args[2].
function lastVideoCall() {
  const { args } = lastCall("generateVideo");
  return { prompt: args[0], numFrames: args[1], size: args[2] };
}

test("a requested frame rate is forwarded and reported back", async () => {
  const res = await request(app)
    .post("/api/generate/video")
    .send({ prompt: "a spinning cube", fps: 30, num_frames: 16 });

  assert.equal(res.status, 200);
  assert.equal(res.body.video.fps, 30);
  assert.equal(res.body.metadata.fps, 30);

  const { prompt, numFrames, size } = lastVideoCall();
  assert.equal(prompt, "a spinning cube");
  assert.equal(numFrames, 16);
  assert.equal(size.fps, 30);
});

test("a requested size is forwarded and reported back", async () => {
  const res = await request(app)
    .post("/api/generate/video")
    .send({ prompt: "a dancing robot", width: 512, height: 256, num_frames: 8 });

  assert.equal(res.status, 200);
  assert.equal(res.body.metadata.width, 512);
  assert.equal(res.body.metadata.height, 256);

  const { numFrames, size } = lastVideoCall();
  assert.equal(numFrames, 8);
  assert.equal(size.width, 512);
  assert.equal(size.height, 256);
});

test("an aspect ratio and a size tier are forwarded", async () => {
  const res = await request(app)
    .post("/api/generate/video")
    .send({ prompt: "a landscape view", aspect: "16:9", tier: 512, num_frames: 24 });

  assert.equal(res.status, 200);

  const { size } = lastVideoCall();
  assert.equal(size.aspect, "16:9");
  assert.equal(size.tier, 512);
});

test("a seed is forwarded so a clip can be reproduced", async () => {
  const res = await request(app)
    .post("/api/generate/video")
    .send({ prompt: "a random pattern", seed: 42, num_frames: 16 });

  assert.equal(res.status, 200);
  assert.equal(lastVideoCall().size.seed, 42);
});

test("a clip longer than one denoising window is accepted", async () => {
  const res = await request(app)
    .post("/api/generate/video")
    .send({ prompt: "a long journey", num_frames: 128 });

  assert.equal(res.status, 200);
  assert.equal(res.body.video.frames, 128);
  assert.equal(res.body.metadata.frames, 128);
  assert.equal(lastVideoCall().numFrames, 128);

  const past = await request(app)
    .post("/api/generate/video")
    .send({ prompt: "x", num_frames: MAX_FRAMES + 1 });
  assert.equal(past.status, 400);
  assert.equal(past.body.details[0].field, "num_frames");
});

test("a request that asks for nothing in particular still gets a rate", async () => {
  const res = await request(app).post("/api/generate/video").send({ prompt: "a simple test" });

  assert.equal(res.status, 200);
  assert.equal(res.body.video.frames, 16);
  // The placeholder path used to omit fps entirely, though the response schema
  // declares it, so a caller could not tell what rate it was going to get.
  assert.equal(res.body.video.fps, 24);
  assert.equal(res.body.metadata.fps, 24);
});

test("a frame rate outside the supported range is refused", async () => {
  for (const fps of [MIN_FPS - 1, MAX_FPS + 1]) {
    const res = await request(app).post("/api/generate/video").send({ prompt: "test", fps });
    assert.equal(res.status, 400, `fps ${fps} should be rejected`);
    assert.equal(res.body.details[0].field, "fps");
  }

  for (const fps of [MIN_FPS, MAX_FPS]) {
    const res = await request(app).post("/api/generate/video").send({ prompt: "test", fps });
    assert.equal(res.status, 200, `fps ${fps} should be accepted`);
  }
});

test("a size that is out of range, or given by half, is refused", async () => {
  for (const field of ["width", "height"]) {
    const res = await request(app)
      .post("/api/generate/video")
      .send({ prompt: "test", [field]: MIN_DIMENSION - 1 });
    assert.equal(res.status, 400, `${field} below the minimum should be rejected`);
  }

  // Width and height mean nothing apart, so one without the other is an error
  // rather than a half-applied size.
  const widthOnly = await request(app)
    .post("/api/generate/video")
    .send({ prompt: "test", width: 512 });
  assert.equal(widthOnly.status, 400);
  assert.equal(widthOnly.body.details[0].field, "height");
});
