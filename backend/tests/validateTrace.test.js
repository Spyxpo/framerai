/**
 * Regression tests for validateTrace() — trace.tools validation (#337).
 *
 * Root cause: the original implementation forwarded trace.tools as-is.
 * A tool entry with a non-string name or a non-JSON-safe input/output could
 * reach the rendering layer and cause a crash (JSON.stringify on circular
 * reference, or .name rendered as undefined).
 *
 * Fix: map each tool entry through a normalizer that requires a non-empty
 * string name (entries without one are dropped entirely), round-trips
 * input/output through JSON to verify serializability, and drops any
 * non-JSON-safe input/output value.
 *
 * Discriminating tests (fail on old code):
 *   - "validateTrace filters tool entry with non-string name"
 *   - "validateTrace drops non-JSON-safe tool input"
 *   - "validateTrace preserves valid tool input and output unchanged"
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const { validateTrace } = require("../src/services/model");

// ---------------------------------------------------------------------------
// Valid tools — must pass through intact
// ---------------------------------------------------------------------------

test("validateTrace preserves valid tool entries", () => {
  const trace = {
    tools: [
      { name: "web_search", input: { query: "test" }, output: { results: ["a", "b"] } },
      { name: "web_fetch", input: { url: "https://example.com" } },
      { name: "minimal_tool" },
    ],
  };

  const result = validateTrace(trace);

  assert.ok(result, "trace should not be null");
  assert.ok(Array.isArray(result.tools), "tools should be an array");
  assert.equal(result.tools.length, 3, "all three valid entries should pass through");

  assert.equal(result.tools[0].name, "web_search");
  assert.deepEqual(result.tools[0].input, { query: "test" });
  assert.deepEqual(result.tools[0].output, { results: ["a", "b"] });

  assert.equal(result.tools[1].name, "web_fetch");
  assert.deepEqual(result.tools[1].input, { url: "https://example.com" });
  assert.equal(result.tools[1].output, undefined, "output should be absent when not provided");

  assert.equal(result.tools[2].name, "minimal_tool");
  assert.equal(result.tools[2].input, undefined);
  assert.equal(result.tools[2].output, undefined);
});

// ---------------------------------------------------------------------------
// DISCRIMINATING TEST 1: non-string name
// Old code: entry passes through with name: 42
// New code: entry filtered out (name normalises to "" then dropped)
// ---------------------------------------------------------------------------

test("validateTrace filters tool entry with non-string name", () => {
  const trace = {
    tools: [
      { name: 42, input: { x: 1 } },           // numeric name — should be dropped
      { name: null },                            // null name — should be dropped
      { input: { x: 1 } },                      // missing name — should be dropped
      { name: "valid_tool", input: { x: 1 } },  // valid — should be kept
    ],
  };

  const result = validateTrace(trace);

  assert.ok(result, "trace should not be null (has at least one valid tool)");
  assert.ok(Array.isArray(result.tools), "tools should be an array");
  assert.equal(result.tools.length, 1, "only the valid entry should survive");
  assert.equal(result.tools[0].name, "valid_tool");
  assert.deepEqual(result.tools[0].input, { x: 1 });

  // Confirm no entry with a non-string name slipped through
  for (const tool of result.tools) {
    assert.equal(typeof tool.name, "string", `tool name must be a string, got: ${typeof tool.name}`);
    assert.ok(tool.name.length > 0, "tool name must not be empty");
  }
});

// ---------------------------------------------------------------------------
// DISCRIMINATING TEST 2: non-JSON-safe (circular) input/output
// Old code: entry passes through with circular reference still attached
// New code: the input/output property is dropped; name is preserved
// ---------------------------------------------------------------------------

test("validateTrace drops non-JSON-safe tool input", () => {
  const circular = {};
  circular.self = circular; // circular reference — JSON.stringify throws

  const trace = {
    tools: [
      { name: "circular_tool", input: circular, output: { result: "ok" } },
      { name: "safe_tool", input: { x: 1 }, output: { y: 2 } },
    ],
  };

  const result = validateTrace(trace);

  assert.ok(result, "trace should not be null");
  assert.equal(result.tools.length, 2, "both tool names should survive");

  const circularTool = result.tools.find((t) => t.name === "circular_tool");
  assert.ok(circularTool, "entry with valid name but circular input should be kept (name preserved)");
  assert.equal(circularTool.input, undefined, "circular input must be dropped");
  // The output was JSON-safe, so it should be preserved
  assert.deepEqual(circularTool.output, { result: "ok" });

  const safeTool = result.tools.find((t) => t.name === "safe_tool");
  assert.ok(safeTool, "safe tool should be present");
  assert.deepEqual(safeTool.input, { x: 1 });
  assert.deepEqual(safeTool.output, { y: 2 });
});

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

test("validateTrace returns null when all tools are invalid", () => {
  const trace = {
    tools: [
      { name: 99 },
      { name: null },
    ],
  };

  const result = validateTrace(trace);
  // No valid tools, no other fields — result should be null
  assert.equal(result, null);
});

test("validateTrace handles tools alongside other valid fields", () => {
  const trace = {
    memories: [{ text: "mem", score: 0.8 }],
    tools: [
      { name: "tool_a" },
      { name: 123 }, // should be filtered
    ],
  };

  const result = validateTrace(trace);

  assert.ok(result, "trace should not be null");
  assert.equal(result.memories.length, 1);
  assert.equal(result.tools.length, 1, "only valid tool should remain");
  assert.equal(result.tools[0].name, "tool_a");
});

// ---------------------------------------------------------------------------
// #337: non-finite numeric values must not reach the rendering layer
// ---------------------------------------------------------------------------

test("#337: validateTrace clamps non-finite memories.score to 0", () => {
  const t = (score) => validateTrace({ memories: [{ text: "m", score }] });

  // Non-finite — must be clamped to 0
  assert.equal(t(Infinity).memories[0].score, 0,  "Infinity → 0");
  assert.equal(t(-Infinity).memories[0].score, 0, "-Infinity → 0");
  assert.equal(t(NaN).memories[0].score, 0,       "NaN → 0");

  // Valid finite values — must pass through unchanged
  assert.equal(t(0.85).memories[0].score, 0.85,   "0.85 preserved");
  assert.equal(t(0).memories[0].score, 0,          "0 preserved");
  assert.equal(t(-1).memories[0].score, -1,        "-1 preserved");

  // Numeric string — converted to number, then checked
  assert.equal(t("0.7").memories[0].score, 0.7,   "numeric string '0.7' converted");
  assert.equal(t("Infinity").memories[0].score, 0,"string 'Infinity' clamped to 0");
});

test("#337: validateTrace clamps non-finite affect elements to 0", () => {
  const result = validateTrace({ affect: [0.5, Infinity, -Infinity, NaN, -0.3] });

  assert.ok(result, "trace should not be null");
  assert.deepEqual(result.affect, [0.5, 0, 0, 0, -0.3]);
});

test("#337: validateTrace excludes affect_adj when non-finite", () => {
  // Infinity
  const r1 = validateTrace({ affect_adj: Infinity, memories: [{ text: "m", score: 0 }] });
  assert.ok(r1, "trace not null — memories still present");
  assert.equal(r1.affect_adj, undefined, "Infinity affect_adj must be excluded");

  // -Infinity
  const r2 = validateTrace({ affect_adj: -Infinity, memories: [{ text: "m", score: 0 }] });
  assert.equal(r2.affect_adj, undefined, "-Infinity affect_adj must be excluded");

  // NaN
  const r3 = validateTrace({ affect_adj: NaN, memories: [{ text: "m", score: 0 }] });
  assert.equal(r3.affect_adj, undefined, "NaN affect_adj must be excluded");

  // Valid finite values — must be included
  const r4 = validateTrace({ affect_adj: 1.5 });
  assert.ok(r4, "trace with valid affect_adj not null");
  assert.equal(r4.affect_adj, 1.5, "finite affect_adj preserved");

  // 0 is a valid finite value
  const r5 = validateTrace({ affect_adj: 0 });
  assert.ok(r5);
  assert.equal(r5.affect_adj, 0, "zero affect_adj preserved");
});

test("#337: validateTrace clamps non-finite sampling values to 0", () => {
  const result = validateTrace({
    sampling: { top_k: 40, top_p: Infinity, temperature: -Infinity, seed: NaN, penalty: 1.1 },
  });

  assert.ok(result, "trace should not be null");
  assert.equal(result.sampling.top_k, 40,  "top_k preserved");
  assert.equal(result.sampling.top_p, 0,   "Infinity top_p → 0");
  assert.equal(result.sampling.temperature, 0, "-Infinity temperature → 0");
  assert.equal(result.sampling.seed, 0,    "NaN seed → 0");
  assert.equal(result.sampling.penalty, 1.1, "penalty preserved");
});

test("validateTrace does not mutate the original trace object", () => {
  const original = {
    tools: [
      { name: "tool", input: { x: 1 } },
    ],
  };
  const originalRef = original.tools[0];

  validateTrace(original);

  // Original object must be unmodified
  assert.strictEqual(original.tools[0], originalRef, "original tool entry reference must not change");
  assert.deepEqual(original.tools[0], { name: "tool", input: { x: 1 } });
});
