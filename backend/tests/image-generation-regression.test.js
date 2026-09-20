/**
 * Focused regression tests for multi-image generation (num_images > 1).
 */

const test = require("node:test");
const assert = require("node:assert/strict");

const bridge = require("../src/services/pythonBridge");
const { generateImage } = require("../src/services/model");

test("generateImage maps multiple returned files to images array", async () => {
  const origAvailable = bridge.available;
  const origRequest = bridge.request;

  bridge.available = () => true;
  bridge.request = async (op, payload) => {
    assert.equal(op, "image");
    assert.equal(payload.num_images, 3);
    return {
      file: "img1.png",
      files: ["img1.png", "img2.png", "img3.png"],
      width: 512,
      height: 512,
      aspect: "1:1",
      source: "preset",
      snapped: false,
      seed: 42,
    };
  };

  try {
    const res = await generateImage("three cats", 3);
    assert.equal(res.prompt, "three cats");
    assert.equal(res.images.length, 3);
    assert.equal(res.images[0].url, "/uploads/generated/img1.png");
    assert.equal(res.images[1].url, "/uploads/generated/img2.png");
    assert.equal(res.images[2].url, "/uploads/generated/img3.png");
    assert.equal(res.images[0].placeholder, false);
    assert.equal(res.images[1].placeholder, false);
    assert.equal(res.images[2].placeholder, false);
  } finally {
    bridge.available = origAvailable;
    bridge.request = origRequest;
  }
});

test("generateImage falls back to single file when files is absent", async () => {
  const origAvailable = bridge.available;
  const origRequest = bridge.request;

  bridge.available = () => true;
  bridge.request = async () => ({
    file: "legacy.png",
    width: 256,
    height: 256,
    aspect: "1:1",
    source: "default",
    snapped: false,
    seed: 0,
  });

  try {
    const res = await generateImage("one cat", 1);
    assert.equal(res.images.length, 1);
    assert.equal(res.images[0].url, "/uploads/generated/legacy.png");
    assert.equal(res.images[0].placeholder, false);
  } finally {
    bridge.available = origAvailable;
    bridge.request = origRequest;
  }
});
