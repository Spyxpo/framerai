/**
 * Regression tests for Issue #359:
 * Reject malformed generation request payloads with consistent API errors.
 */

process.env.GENERATE_RATE_LIMIT_MAX = "500";
process.env.RATE_LIMIT_MAX = "1000";

const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");

const { mockModel, loadApp } = require("./helpers");

const calls = mockModel();
const app = loadApp();

function assertValidationError(res, expectedField, expectedMessagePattern) {
  assert.equal(res.status, 400, `Expected 400 status, got ${res.status}`);
  assert.equal(res.body.code, "VALIDATION_ERROR");
  assert.equal(res.body.error, "Request validation failed");
  assert.ok(Array.isArray(res.body.details), "details must be an array");
  assert.ok(res.body.details.length > 0, "details must have at least one error");
  assert.ok(typeof res.body.requestId === "string", "requestId should be present");

  if (expectedField) {
    const detail = res.body.details.find((d) => d.field === expectedField);
    assert.ok(detail, `Expected validation error for field '${expectedField}', got: ${JSON.stringify(res.body.details)}`);
    if (expectedMessagePattern) {
      assert.match(detail.message, expectedMessagePattern);
    }
  }
}

// ---------------------------------------------------------------------------
// 1. Missing required field -> 4xx
// ---------------------------------------------------------------------------

test("Issue #359: missing required field prompt on all generation endpoints returns 400 VALIDATION_ERROR", async () => {
  const routes = ["image", "video", "audio", "code"];
  for (const route of routes) {
    const res = await request(app).post(`/api/generate/${route}`).send({});
    assertValidationError(res, "prompt", /is required/);
  }
});

// ---------------------------------------------------------------------------
// 2. Invalid null value -> 4xx where appropriate
// ---------------------------------------------------------------------------

test("Issue #359: null prompt is rejected at API boundary", async () => {
  const res = await request(app).post("/api/generate/image").send({ prompt: null });
  assertValidationError(res, "prompt", /must be a string/);
});

test("Issue #359: null num_images is rejected at API boundary", async () => {
  const res = await request(app).post("/api/generate/image").send({ prompt: "cat", num_images: null });
  assertValidationError(res, "num_images", /must be an integer/);
});

test("Issue #359: null width and height are rejected at API boundary", async () => {
  const res = await request(app).post("/api/generate/image").send({ prompt: "cat", width: null, height: null });
  assertValidationError(res, "width", /must be an integer/);
  assertValidationError(res, "height", /must be an integer/);
});

test("Issue #359: null aspect ratio is rejected at API boundary", async () => {
  const res = await request(app).post("/api/generate/image").send({ prompt: "cat", aspect: null });
  assertValidationError(res, "aspect", /must be one of/);
});

test("Issue #359: null size tier is rejected at API boundary", async () => {
  const res = await request(app).post("/api/generate/image").send({ prompt: "cat", tier: null });
  assertValidationError(res, "tier", /must be one of/);
});

test("Issue #359: null seed is rejected at API boundary", async () => {
  const res = await request(app).post("/api/generate/image").send({ prompt: "cat", seed: null });
  assertValidationError(res, "seed", /must be an integer/);
});

test("Issue #359: null video num_frames and fps are rejected at API boundary", async () => {
  const res = await request(app).post("/api/generate/video").send({ prompt: "cat", num_frames: null, fps: null });
  assertValidationError(res, "num_frames", /must be an integer/);
  assertValidationError(res, "fps", /must be an integer/);
});

test("Issue #359: null code language is rejected at API boundary", async () => {
  const res = await request(app).post("/api/generate/code").send({ prompt: "cat", language: null });
  assertValidationError(res, "language", /must be one of/);
});

test("Issue #359: null settings object is rejected at API boundary", async () => {
  const res = await request(app).post("/api/generate/code").send({ prompt: "cat", settings: null });
  assertValidationError(res, "settings", /must be an object/);
});

test("Issue #359: null nested settings values are rejected at API boundary", async () => {
  const testFields = [
    { field: "temperature", expectedPattern: /must be a number/ },
    { field: "top_p", expectedPattern: /must be a number/ },
    { field: "top_k", expectedPattern: /must be an integer/ },
    { field: "max_new_tokens", expectedPattern: /must be an integer/ },
    { field: "stop", expectedPattern: /must be an array/ },
    { field: "tools", expectedPattern: /must be an array/ },
    { field: "repetition_penalty", expectedPattern: /must be a number/ },
    { field: "seed", expectedPattern: /must be an integer/ },
  ];

  for (const { field, expectedPattern } of testFields) {
    const res = await request(app)
      .post("/api/generate/code")
      .send({ prompt: "cat", settings: { [field]: null } });
    assertValidationError(res, `settings.${field}`, expectedPattern);
  }
});

// ---------------------------------------------------------------------------
// 3. Incorrect data type -> 4xx
// ---------------------------------------------------------------------------

test("Issue #359: incorrect primitive data types are rejected at API boundary", async () => {
  const cases = [
    { endpoint: "image", payload: { prompt: 12345 }, field: "prompt", pattern: /must be a string/ },
    { endpoint: "image", payload: { prompt: "cat", num_images: "two" }, field: "num_images", pattern: /must be an integer/ },
    { endpoint: "image", payload: { prompt: "cat", num_images: true }, field: "num_images", pattern: /must be an integer/ },
    { endpoint: "image", payload: { prompt: "cat", width: "wide", height: 512 }, field: "width", pattern: /must be an integer/ },
    { endpoint: "image", payload: { prompt: "cat", aspect: 16 }, field: "aspect", pattern: /must be one of/ },
    { endpoint: "video", payload: { prompt: "cat", fps: "fast" }, field: "fps", pattern: /must be an integer/ },
    { endpoint: "code", payload: { prompt: "cat", language: 123 }, field: "language", pattern: /must be one of/ },
  ];

  for (const c of cases) {
    const res = await request(app).post(`/api/generate/${c.endpoint}`).send(c.payload);
    assertValidationError(res, c.field, c.pattern);
  }
});

// ---------------------------------------------------------------------------
// 4. Malformed nested request data -> 4xx
// ---------------------------------------------------------------------------

test("Issue #359: malformed nested request data is rejected at API boundary", async () => {
  const cases = [
    { payload: { prompt: "cat", settings: "not-an-object" }, field: "settings", pattern: /must be an object/ },
    { payload: { prompt: "cat", settings: [1, 2, 3] }, field: "settings", pattern: /must be an object/ },
    { payload: { prompt: "cat", settings: true }, field: "settings", pattern: /must be an object/ },
    { payload: { prompt: "cat", settings: { temperature: "warm" } }, field: "settings.temperature", pattern: /must be a number/ },
    { payload: { prompt: "cat", settings: { stop: "STOP" } }, field: "settings.stop", pattern: /must be an array/ },
    { payload: { prompt: "cat", settings: { stop: [123] } }, field: "settings.stop[0]", pattern: /must be a string/ },
    { payload: { prompt: "cat", settings: { tools: "web" } }, field: "settings.tools", pattern: /must be an array/ },
  ];

  for (const c of cases) {
    const res = await request(app).post("/api/generate/code").send(c.payload);
    assertValidationError(res, c.field, c.pattern);
  }
});

// ---------------------------------------------------------------------------
// 5. Verify invalid input does NOT reach the generation/model logic
// ---------------------------------------------------------------------------

test("Issue #359: invalid requests never reach the generation/model logic", async () => {
  const beforeCount = calls.length;

  const invalidPayloads = [
    { route: "image", payload: {} },
    { route: "image", payload: { prompt: null } },
    { route: "image", payload: { prompt: "cat", num_images: null } },
    { route: "image", payload: { prompt: "cat", width: null, height: null } },
    { route: "image", payload: { prompt: "cat", aspect: null } },
    { route: "video", payload: { prompt: "cat", num_frames: null } },
    { route: "code", payload: { prompt: "cat", settings: null } },
    { route: "code", payload: { prompt: "cat", settings: { temperature: null } } },
    { route: "code", payload: { prompt: "cat", settings: { stop: null } } },
    { route: "code", payload: { prompt: "cat", settings: "invalid-type" } },
  ];

  for (const { route, payload } of invalidPayloads) {
    const res = await request(app).post(`/api/generate/${route}`).send(payload);
    assert.equal(res.status, 400);
  }

  assert.equal(calls.length, beforeCount, "Model service functions must not be invoked for invalid payloads");
});

// ---------------------------------------------------------------------------
// 6. Verify valid /api/generate requests still work
// ---------------------------------------------------------------------------

test("Issue #359: valid /api/generate requests continue to work normally", async () => {
  const imageRes = await request(app).post("/api/generate/image").send({ prompt: "a friendly cat", num_images: 2 });
  assert.equal(imageRes.status, 200);
  assert.equal(imageRes.body.prompt, "a friendly cat");
  assert.equal(imageRes.body.images.length, 2);

  const videoRes = await request(app).post("/api/generate/video").send({ prompt: "flying bird", num_frames: 24 });
  assert.equal(videoRes.status, 200);
  assert.equal(videoRes.body.prompt, "flying bird");

  const audioRes = await request(app).post("/api/generate/audio").send({ prompt: "piano chords" });
  assert.equal(audioRes.status, 200);
  assert.equal(audioRes.body.prompt, "piano chords");

  const codeRes = await request(app).post("/api/generate/code").send({
    prompt: "quicksort in python",
    language: "python",
    settings: { temperature: 0.5, top_p: 0.9 },
  });
  assert.equal(codeRes.status, 200);
  assert.equal(codeRes.body.language, "python");
});

// ---------------------------------------------------------------------------
// 7. Verify error responses follow existing API conventions
// ---------------------------------------------------------------------------

test("Issue #359: error responses follow repository API error conventions", async () => {
  const res = await request(app).post("/api/generate/image").send({
    prompt: 999,
    num_images: "three",
    aspect: "unknown",
  });

  assert.equal(res.status, 400);
  assert.equal(res.body.code, "VALIDATION_ERROR");
  assert.equal(res.body.error, "Request validation failed");
  assert.ok(Array.isArray(res.body.details));
  assert.equal(res.body.details.length, 3);
  assert.deepEqual(
    res.body.details.map((d) => d.field).sort(),
    ["aspect", "num_images", "prompt"]
  );
  for (const d of res.body.details) {
    assert.ok(typeof d.field === "string");
    assert.ok(typeof d.message === "string");
  }
  assert.match(res.body.requestId, /^[0-9a-f-]{36}$/);
});
