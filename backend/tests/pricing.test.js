const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const request = require("supertest");

const { mockModel, loadApp } = require("./helpers");
const pricing = require("../src/pricing");

mockModel();
const app = loadApp();

function withPricingFile(config, fn) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pricing-"));
  const file = path.join(dir, "pricing.json");
  fs.writeFileSync(file, JSON.stringify(config));
  const previous = process.env.PRICING_FILE;
  process.env.PRICING_FILE = file;
  pricing.reset();
  try {
    return fn();
  } finally {
    if (previous === undefined) delete process.env.PRICING_FILE;
    else process.env.PRICING_FILE = previous;
    pricing.reset();
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

const SIMPLE = {
  currency: "USD",
  // $3.60/hr with no margin is exactly $0.001 per GPU-second.
  basis: { gpu_hour_usd: 3.6, margin: 0 },
  throughput: {
    "framerai-text": {
      prefill_tokens_per_second: 1000,
      decode_tokens_per_second: 100,
      cached_input_discount: 0.5,
    },
    "framerai-image": { gpu_seconds_per_megapixel: 2 },
  },
};

test("rates are GPU cost divided by throughput", () => {
  withPricingFile(SIMPLE, () => {
    const card = pricing.rates();
    assert.equal(card.models["framerai-text"].input_per_million, 1);
    assert.equal(card.models["framerai-text"].cached_input_per_million, 0.5);
    assert.equal(card.models["framerai-text"].output_per_million, 10);
    assert.equal(card.models["framerai-image"].per_megapixel, 0.002);
  });
});

test("margin raises every rate by the same factor", () => {
  withPricingFile({ ...SIMPLE, basis: { gpu_hour_usd: 3.6, margin: 0.5 } }, () => {
    assert.equal(pricing.rates().models["framerai-text"].output_per_million, 15);
  });
});

test("meter prices input, cached input and output separately", () => {
  withPricingFile(SIMPLE, () => {
    const usage = pricing.meter("framerai-text", {
      input_tokens: 1_000_000,
      cached_input_tokens: 400_000,
      output_tokens: 100_000,
    });
    // 600k fresh at $1 + 400k cached at $0.5 + 100k out at $10.
    assert.deepEqual(usage, {
      input_tokens: 1_000_000,
      output_tokens: 100_000,
      cached_input_tokens: 400_000,
      cost_usd: 1.8,
    });
  });
});

test("meter returns null with no usage or no token rates", () => {
  withPricingFile(SIMPLE, () => {
    assert.equal(pricing.meter("framerai-text", undefined), null);
    assert.equal(pricing.meter("framerai-image", { input_tokens: 5 }), null);
    assert.equal(pricing.meter("unknown", { input_tokens: 5 }), null);
  });
});

test("meter ignores negative and non-numeric counts", () => {
  withPricingFile(SIMPLE, () => {
    const usage = pricing.meter("framerai-text", { input_tokens: -4, output_tokens: "x" });
    assert.deepEqual(usage, { input_tokens: 0, output_tokens: 0, cost_usd: 0 });
  });
});

test("GET /api/pricing serves the derived card", async () => {
  const res = await request(app).get("/api/pricing");
  assert.equal(res.status, 200);
  assert.equal(res.body.currency, "USD");
  const text = res.body.models["framerai-text"];
  assert.ok(text.input_per_million > 0);
  assert.ok(text.output_per_million > text.input_per_million);
  assert.ok(text.cached_input_per_million < text.input_per_million);
});
