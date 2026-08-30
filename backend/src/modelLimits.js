/**
 * Input limits sized to the model that is actually running.
 *
 * The API used to cap a message at 8000 characters, a prompt at 4000, and
 * max_new_tokens at 2048, whatever it was serving. The largest presets declare
 * a 1,048,576-token context, so those constants put the window several orders
 * of magnitude out of reach: a client could not send enough input to use it,
 * and no setting would let them.
 *
 * The limits below are derived from the window the worker reports when it
 * starts. With no model running the previous constants are the fallback, so a
 * placeholder deployment behaves exactly as it did.
 */

const bridge = require("./services/pythonBridge");

// Fallbacks, and the floor: a small model does not shrink the API below what
// clients already relied on.
const BASE_MESSAGE_CHARS = 8000;
const BASE_PROMPT_CHARS = 4000;
const BASE_MAX_NEW_TOKENS = 2048;

// Byte-level BPE averages a little over three characters per token on prose.
// Three is the conservative direction: it under-promises characters, so the
// model's own bound is what finally trims rather than a surprise at the edge.
const CHARS_PER_TOKEN = 3;

// Generation is a share of the window, matching how the worker divides it.
const GENERATION_SHARE = 4;

function windowTokens() {
  const info = typeof bridge.modelInfo === "function" ? bridge.modelInfo() : null;
  const tokens = info && Number(info.max_seq_len);
  return Number.isFinite(tokens) && tokens > 0 ? tokens : null;
}

/** Characters accepted in one chat message. */
function messageChars() {
  const tokens = windowTokens();
  if (!tokens) return BASE_MESSAGE_CHARS;
  return Math.max(BASE_MESSAGE_CHARS, tokens * CHARS_PER_TOKEN);
}

/** Characters accepted in a generation prompt. */
function promptChars() {
  const tokens = windowTokens();
  if (!tokens) return BASE_PROMPT_CHARS;
  return Math.max(BASE_PROMPT_CHARS, tokens * CHARS_PER_TOKEN);
}

/** Largest max_new_tokens a client may ask for. */
function maxNewTokens() {
  const tokens = windowTokens();
  if (!tokens) return BASE_MAX_NEW_TOKENS;
  return Math.max(BASE_MAX_NEW_TOKENS, Math.floor(tokens / GENERATION_SHARE));
}

/** Everything at once, for the routes that report their own bounds. */
function limits() {
  return {
    messageChars: messageChars(),
    promptChars: promptChars(),
    maxNewTokens: maxNewTokens(),
    windowTokens: windowTokens(),
  };
}

module.exports = {
  messageChars,
  promptChars,
  maxNewTokens,
  limits,
  BASE_MESSAGE_CHARS,
  BASE_PROMPT_CHARS,
  BASE_MAX_NEW_TOKENS,
};
