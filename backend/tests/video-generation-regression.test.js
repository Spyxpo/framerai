const { describe, it } = require("node:test");
const assert = require("node:assert");
const request = require("supertest");
const { mockModel, loadApp } = require("./helpers");

process.env.GENERATE_RATE_LIMIT_MAX = "500";
process.env.RATE_LIMIT_MAX = "1000";

const calls = mockModel();
const app = loadApp();

function lastCall(name) {
  return [...calls].reverse().find((c) => c.name === name);
}

// Route calls: generateVideo(prompt, numFrames, { width, height, aspect, tier, fps, seed }, requestId)
// args[0] = prompt, args[1] = numFrames, args[2] = size object

describe("Video Generation Regression Tests - Issue #207", () => {

  it("should honour requested FPS in video generation", async () => {
    const response = await request(app)
      .post("/api/generate/video")
      .send({
        prompt: "a spinning cube",
        fps: 30,
        num_frames: 16
      });

    assert.equal(response.status, 200);
    assert.equal(response.body.video.fps, 30);
    assert.equal(response.body.metadata.fps, 30);

    const videoArgs = lastCall("generateVideo").args;
    assert.equal(videoArgs[0], "a spinning cube"); // prompt
    assert.equal(videoArgs[1], 16);                // numFrames
    assert.equal(videoArgs[2].fps, 30);            // size.fps
  });

  it("should honour requested width and height", async () => {
    const response = await request(app)
      .post("/api/generate/video")
      .send({
        prompt: "a dancing robot",
        width: 512,
        height: 256,
        num_frames: 8
      });

    assert.equal(response.status, 200);
    assert.equal(response.body.metadata.width, 512);
    assert.equal(response.body.metadata.height, 256);

    const videoArgs = lastCall("generateVideo").args;
    assert.equal(videoArgs[0], "a dancing robot"); // prompt
    assert.equal(videoArgs[1], 8);                 // numFrames
    assert.equal(videoArgs[2].width, 512);         // size.width
    assert.equal(videoArgs[2].height, 256);        // size.height
  });

  it("should support extended video duration up to 128 frames", async () => {
    const response = await request(app)
      .post("/api/generate/video")
      .send({
        prompt: "a long journey",
        num_frames: 128
      });

    assert.equal(response.status, 200);
    assert.equal(response.body.video.frames, 128);
    assert.equal(response.body.metadata.frames, 128);

    const videoArgs = lastCall("generateVideo").args;
    assert.equal(videoArgs[1], 128); // numFrames
  });

  it("should support aspect ratio and tier parameters", async () => {
    const response = await request(app)
      .post("/api/generate/video")
      .send({
        prompt: "a landscape view",
        aspect: "16:9",
        tier: 512,
        num_frames: 24
      });

    assert.equal(response.status, 200);

    const videoArgs = lastCall("generateVideo").args;
    assert.equal(videoArgs[1], 24);              // numFrames
    assert.equal(videoArgs[2].aspect, "16:9");   // size.aspect
    assert.equal(videoArgs[2].tier, 512);        // size.tier
  });

  it("should support seed parameter for reproducible generation", async () => {
    const response = await request(app)
      .post("/api/generate/video")
      .send({
        prompt: "a random pattern",
        seed: 42,
        num_frames: 16
      });

    assert.equal(response.status, 200);

    const videoArgs = lastCall("generateVideo").args;
    assert.equal(videoArgs[2].seed, 42); // size.seed
  });

  it("should use default values when optional parameters are not provided", async () => {
    const response = await request(app)
      .post("/api/generate/video")
      .send({
        prompt: "a simple test"
      });

    assert.equal(response.status, 200);
    assert.equal(response.body.video.frames, 16); // default num_frames
    assert.equal(response.body.video.fps, 24);    // default fps

    const videoArgs = lastCall("generateVideo").args;
    assert.equal(videoArgs[1], 16); // default numFrames
  });

  it("should validate FPS is within acceptable range", async () => {
    // Test lower bound
    const tooLow = await request(app)
      .post("/api/generate/video")
      .send({ prompt: "test", fps: 0 });

    assert.equal(tooLow.status, 400);
    assert.equal(tooLow.body.details[0].field, "fps");

    // Test upper bound
    const tooHigh = await request(app)
      .post("/api/generate/video")
      .send({ prompt: "test", fps: 100 });

    assert.equal(tooHigh.status, 400);
    assert.equal(tooHigh.body.details[0].field, "fps");

    // Test valid bounds
    const validLow = await request(app)
      .post("/api/generate/video")
      .send({ prompt: "test", fps: 1 });
    assert.equal(validLow.status, 200);

    const validHigh = await request(app)
      .post("/api/generate/video")
      .send({ prompt: "test", fps: 60 });
    assert.equal(validHigh.status, 200);
  });

  it("should validate num_frames is within extended range", async () => {
    // Test lower bound
    const tooLow = await request(app)
      .post("/api/generate/video")
      .send({ prompt: "test", num_frames: 0 });

    assert.equal(tooLow.status, 400);
    assert.equal(tooLow.body.details[0].field, "num_frames");

    // Test upper bound (MAX_FRAMES is 512 on upstream/dev)
    const tooHigh = await request(app)
      .post("/api/generate/video")
      .send({ prompt: "test", num_frames: 1000 });

    assert.equal(tooHigh.status, 400);
    assert.equal(tooHigh.body.details[0].field, "num_frames");

    // Test valid extended range
    const valid = await request(app)
      .post("/api/generate/video")
      .send({ prompt: "test", num_frames: 128 });
    assert.equal(valid.status, 200);
  });

  it("should validate width and height dimensions", async () => {
    const invalidWidth = await request(app)
      .post("/api/generate/video")
      .send({ prompt: "test", width: -1 });
    assert.equal(invalidWidth.status, 400);

    const invalidHeight = await request(app)
      .post("/api/generate/video")
      .send({ prompt: "test", height: -1 });
    assert.equal(invalidHeight.status, 400);

    // width without height (or vice versa) must also be rejected
    const widthOnly = await request(app)
      .post("/api/generate/video")
      .send({ prompt: "test", width: 512 });
    assert.equal(widthOnly.status, 400);
  });

  it("should maintain backwards compatibility with legacy function calls", async () => {
    const response = await request(app)
      .post("/api/generate/video")
      .send({
        prompt: "backwards compatibility test",
        num_frames: 24
      });

    assert.equal(response.status, 200);
    assert.equal(response.body.video.frames, 24);

    const videoArgs = lastCall("generateVideo").args;
    assert.equal(videoArgs[0], "backwards compatibility test");
    assert.equal(videoArgs[1], 24); // numFrames as positional arg
  });
});
