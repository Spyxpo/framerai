const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");

const { mockModel, loadApp, newConversation } = require("./helpers");
const { readSettings } = require("../src/generationSettings");
const { Validator } = require("../src/middleware/validate");

const calls = mockModel();
const app = loadApp();

function parse(settings) {
  const v = new Validator({ settings });
  const result = readSettings(v);
  return { result, errors: v.errors };
}

test("readSettings accepts temperature 0 without dropping it", () => {
  const { result, errors } = parse({ temperature: 0 });
  assert.equal(errors.length, 0);
  assert.equal(result.temperature, 0);
});

test("readSettings validates repetition_penalty", () => {
  const valid = parse({ repetition_penalty: 1.2 });
  assert.equal(valid.errors.length, 0);
  assert.equal(valid.result.repetition_penalty, 1.2);

  const minBoundary = parse({ repetition_penalty: 0.0001 });
  assert.equal(minBoundary.errors.length, 0);
  assert.equal(minBoundary.result.repetition_penalty, 0.0001);

  const maxBoundary = parse({ repetition_penalty: 10 });
  assert.equal(maxBoundary.errors.length, 0);
  assert.equal(maxBoundary.result.repetition_penalty, 10);

  const belowMin = parse({ repetition_penalty: 0.00005 });
  assert.ok(belowMin.errors.some((e) => e.field === "settings.repetition_penalty"));

  const aboveMax = parse({ repetition_penalty: 10.1 });
  assert.ok(aboveMax.errors.some((e) => e.field === "settings.repetition_penalty"));

  const zero = parse({ repetition_penalty: 0 });
  assert.ok(zero.errors.some((e) => e.field === "settings.repetition_penalty"));

  const negative = parse({ repetition_penalty: -1.0 });
  assert.ok(negative.errors.some((e) => e.field === "settings.repetition_penalty"));
});

test("readSettings validates stop sequences", () => {
  const valid = parse({ stop: ["STOP", "\n\n"] });
  assert.equal(valid.errors.length, 0);
  assert.deepEqual(valid.result.stop, ["STOP", "\n\n"]);

  const notArray = parse({ stop: "STOP" });
  assert.ok(notArray.errors.some((e) => e.field === "settings.stop" && /array/.test(e.message)));

  const notStrings = parse({ stop: [123] });
  assert.ok(notStrings.errors.some((e) => e.field === "settings.stop[0]" && /string/.test(e.message)));

  const tooMany = parse({ stop: Array(17).fill("a") });
  assert.ok(tooMany.errors.some((e) => e.field === "settings.stop" && /16 items/.test(e.message)));
});

test("readSettings enforces per-token length limit on stop strings", () => {
  const limit = 256;

  // Exactly at the limit — accepted
  const atLimit = parse({ stop: ["A".repeat(limit)] });
  assert.equal(atLimit.errors.length, 0);
  assert.deepEqual(atLimit.result.stop, ["A".repeat(limit)]);

  // One character over the limit — rejected
  const overLimit = parse({ stop: ["A".repeat(limit + 1)] });
  assert.ok(
    overLimit.errors.some(
      (e) => e.field === "settings.stop[0]" && /256 characters/.test(e.message)
    )
  );
  assert.equal(overLimit.result.stop, undefined);

  // Mixed array: one valid, one too long — whole setting is rejected
  const mixed = parse({ stop: ["END", "A".repeat(limit + 1)] });
  assert.ok(
    mixed.errors.some(
      (e) => e.field === "settings.stop[1]" && /256 characters/.test(e.message)
    )
  );
  assert.equal(mixed.result.stop, undefined);

  // Multiple short strings all within the limit — accepted
  const multipleValid = parse({ stop: ["END", "\n\n", "###"] });
  assert.equal(multipleValid.errors.length, 0);
  assert.deepEqual(multipleValid.result.stop, ["END", "\n\n", "###"]);

  // 16 strings each at the limit — maximum valid input accepted
  const maxValid = parse({ stop: Array(16).fill("A".repeat(limit)) });
  assert.equal(maxValid.errors.length, 0);
  assert.equal(maxValid.result.stop.length, 16);

  // Existing 16-item cap still enforced
  const tooManyLong = parse({ stop: Array(17).fill("A".repeat(limit)) });
  assert.ok(
    tooManyLong.errors.some((e) => e.field === "settings.stop" && /16 items/.test(e.message))
  );

  // Non-string values still rejected (type check takes precedence)
  const nonString = parse({ stop: [42] });
  assert.ok(
    nonString.errors.some((e) => e.field === "settings.stop[0]" && /string/.test(e.message))
  );
});

test("readSettings validates per-request seed", () => {
  const valid = parse({ seed: 42 });
  assert.equal(valid.errors.length, 0);
  assert.equal(valid.result.seed, 42);

  const negative = parse({ seed: -1 });
  assert.ok(negative.errors.some((e) => e.field === "settings.seed"));

  const float = parse({ seed: 3.14 });
  assert.ok(float.errors.some((e) => e.field === "settings.seed"));
});

test("generation settings with issue-243 controls reach model via chat endpoint", async () => {
  const id = await newConversation(app);
  const before = calls.length;

  const settings = {
    temperature: 0,
    repetition_penalty: 1.15,
    stop: ["END", "\n\n"],
    seed: 42,
  };

  const res = await request(app)
    .post(`/api/chat/conversations/${id}/messages`)
    .send({ content: "test issue 243 controls", settings });

  assert.equal(res.status, 200);
  const [call] = calls.slice(before);
  assert.deepEqual(call.args[2], settings);
});
