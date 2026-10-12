/**
 * Pricing and metering.
 *
 * Every rate is derived from what a request costs to serve: GPU cost per second
 * divided by measured throughput, plus a margin. Nothing here is a hand-picked
 * price, so a faster deployment is a cheaper one by construction, and the only
 * inputs an operator edits are the hardware cost and the measured throughput in
 * backend/config/pricing.json (or the file PRICING_FILE names).
 */

const fs = require("node:fs");
const path = require("node:path");

const DEFAULT_FILE = path.join(__dirname, "..", "config", "pricing.json");
const PER_MILLION = 1_000_000;

let cached = null;

function loadConfig() {
  if (cached) return cached;
  const file = process.env.PRICING_FILE || DEFAULT_FILE;
  cached = JSON.parse(fs.readFileSync(file, "utf8"));
  return cached;
}

/** Forget the loaded file, so a test or a reload reads it again. */
function reset() {
  cached = null;
}

// Prices are fractions of a cent; eight places keeps a single token visible
// without carrying float noise into the response.
function round(value) {
  return Math.round(value * 1e8) / 1e8;
}

function costPerGpuSecond(basis) {
  return (basis.gpu_hour_usd / 3600) * (1 + basis.margin);
}

/**
 * The derived rate card: one entry per model, each in the unit that model is
 * billed by.
 */
function rates() {
  const config = loadConfig();
  const perSecond = costPerGpuSecond(config.basis);
  const card = {};

  for (const [model, t] of Object.entries(config.throughput)) {
    const entry = {};
    if (t.prefill_tokens_per_second) {
      entry.input_per_million = round((perSecond / t.prefill_tokens_per_second) * PER_MILLION);
      const discount = t.cached_input_discount || 0;
      entry.cached_input_per_million = round(entry.input_per_million * (1 - discount));
    }
    if (t.decode_tokens_per_second) {
      entry.output_per_million = round((perSecond / t.decode_tokens_per_second) * PER_MILLION);
    }
    if (t.gpu_seconds_per_megapixel) {
      entry.per_megapixel = round(perSecond * t.gpu_seconds_per_megapixel);
    }
    if (t.gpu_seconds_per_output_second) {
      entry.per_output_second = round(perSecond * t.gpu_seconds_per_output_second);
    }
    if (t.gpu_seconds_per_input_second) {
      entry.per_input_second = round(perSecond * t.gpu_seconds_per_input_second);
    }
    card[model] = entry;
  }

  return { currency: config.currency, basis: config.basis, models: card };
}

function count(value) {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : 0;
}

/**
 * Usage as the worker reported it, with its cost attached.
 *
 * Returns null when the worker sent no usage or the model has no token rates,
 * so a reply is never shown a cost that was guessed.
 */
function meter(model, usage) {
  if (!usage || typeof usage !== "object") return null;
  const entry = rates().models[model];
  if (!entry || entry.input_per_million === undefined) return null;

  const input = count(usage.input_tokens);
  const output = count(usage.output_tokens);
  const cached = Math.min(count(usage.cached_input_tokens), input);

  const cost =
    ((input - cached) * entry.input_per_million +
      cached * entry.cached_input_per_million +
      output * entry.output_per_million) /
    PER_MILLION;

  const metered = { input_tokens: input, output_tokens: output, cost_usd: round(cost) };
  if (cached) metered.cached_input_tokens = cached;
  return metered;
}

module.exports = { rates, meter, reset };
