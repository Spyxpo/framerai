/**
 * Limits sized to the model, and the history that reaches it.
 *
 * The API capped a message at 8000 characters, a prompt at 4000, and
 * max_new_tokens at 2048 whatever it was serving, so a preset declaring a
 * 1,048,576-token window was unreachable through it. And history was recorded
 * for display only: every turn arrived at the model as a single line.
 */

const test = require("node:test");
const assert = require("node:assert/strict");

const bridge = require("../src/services/pythonBridge");
const modelLimits = require("../src/modelLimits");
const { conversationHistory } = require("../src/services/model");
const conversations = require("../src/conversationStore");

function withModelInfo(info, fn) {
  const original = bridge.modelInfo;
  bridge.modelInfo = () => info;
  try {
    fn();
  } finally {
    bridge.modelInfo = original;
  }
}

// ── Limits ────────────────────────────────────────────────────────────────

test("with no model running the previous constants are what apply", () => {
  withModelInfo(null, () => {
    assert.equal(modelLimits.messageChars(), modelLimits.BASE_MESSAGE_CHARS);
    assert.equal(modelLimits.promptChars(), modelLimits.BASE_PROMPT_CHARS);
    assert.equal(modelLimits.maxNewTokens(), modelLimits.BASE_MAX_NEW_TOKENS);
  });
});

test("a large window raises every limit", () => {
  withModelInfo({ max_seq_len: 1048576 }, () => {
    assert.ok(modelLimits.messageChars() > 1_000_000);
    assert.ok(modelLimits.promptChars() > 1_000_000);
    assert.ok(modelLimits.maxNewTokens() > 200_000);
  });
});

test("a small model does not shrink the API below what clients relied on", () => {
  withModelInfo({ max_seq_len: 1024 }, () => {
    assert.equal(modelLimits.messageChars(), modelLimits.BASE_MESSAGE_CHARS);
    assert.equal(modelLimits.maxNewTokens(), modelLimits.BASE_MAX_NEW_TOKENS);
  });
});

test("a window that makes no sense is ignored rather than trusted", () => {
  for (const info of [{}, { max_seq_len: 0 }, { max_seq_len: -5 }, { max_seq_len: "big" }]) {
    withModelInfo(info, () => {
      assert.equal(modelLimits.messageChars(), modelLimits.BASE_MESSAGE_CHARS);
      assert.equal(modelLimits.limits().windowTokens, null);
    });
  }
});

test("the reported window is the model's own", () => {
  withModelInfo({ max_seq_len: 4096 }, () => {
    assert.equal(modelLimits.limits().windowTokens, 4096);
  });
});

// ── History ───────────────────────────────────────────────────────────────

test("a single turn is sent as it always was", () => {
  assert.equal(conversationHistory([{ role: "user", content: "hi" }]), null);
  assert.equal(conversationHistory([]), null);
  assert.equal(conversationHistory(undefined), null);
});

test("a real exchange is sent as the turns it is made of", () => {
  const history = conversationHistory([
    { role: "user", content: "what is a tensor", type: "text" },
    { role: "assistant", content: "an array with axes", metadata: {} },
    { role: "user", content: "and a rank" },
  ]);
  assert.deepEqual(history, [
    { role: "user", content: "what is a tensor" },
    { role: "assistant", content: "an array with axes" },
    { role: "user", content: "and a rank" },
  ]);
});

test("turns with nothing in them are left out", () => {
  const history = conversationHistory([
    { role: "user", content: "first" },
    { role: "assistant" },
    null,
    { content: "no role" },
    { role: "user", content: "second" },
  ]);
  assert.deepEqual(history.map((m) => m.content), ["first", "second"]);
});

// ── The shared store ──────────────────────────────────────────────────────

test("both transports read and write one conversation", () => {
  conversations.clear();
  conversations.create({ id: "conv-1", title: "t", messages: [], createdAt: "now" });

  assert.ok(conversations.append("conv-1", { role: "user", content: "one" }));
  assert.ok(conversations.append("conv-1", { role: "assistant", content: "two" }));
  assert.deepEqual(conversations.messages("conv-1").map((m) => m.content), ["one", "two"]);

  conversations.clear();
});

test("an unknown conversation is not invented", () => {
  conversations.clear();
  assert.equal(conversations.append("absent", { role: "user", content: "x" }), false);
  assert.deepEqual(conversations.messages("absent"), []);
  assert.equal(conversations.get("absent"), null);
});
