/**
 * Per-request generation settings.
 *
 * The website sends these with a chat message or a generation request. Only
 * what the caller sets is forwarded; anything omitted is left to the model's
 * own defaults rather than being pinned here.
 */

const LIMITS = {
  temperature: { min: 0.1, max: 2 },
  top_p: { min: 0.1, max: 1 },
  top_k: { min: 0, max: 200 },
  max_new_tokens: { min: 16, max: 2048 },
  resolution: [64, 128, 256, 512],
  num_frames: { min: 1, max: 64 },
  tools: ["web", "cli"],
};

/**
 * Read a `settings` object off a request body. Takes the parent validator so
 * bad values are reported together with the rest of the request, as
 * `settings.temperature` and so on.
 */
function readSettings(parent) {
  const v = parent.nested("settings");

  return compact({
    temperature: v.number("temperature", LIMITS.temperature),
    top_p: v.number("top_p", LIMITS.top_p),
    top_k: v.integer("top_k", LIMITS.top_k),
    max_new_tokens: v.integer("max_new_tokens", LIMITS.max_new_tokens),
    resolution: v.oneOf("resolution", LIMITS.resolution),
    num_frames: v.integer("num_frames", LIMITS.num_frames),
    tools: readTools(v),
  });
}

/**
 * The toolsets a chat turn may use. Unknown names are dropped rather than
 * rejected, because the worker decides what it actually has registered; an
 * empty list is returned as undefined so the default path stays untouched.
 */
function readTools(v) {
  const raw = v.array("tools", { max: LIMITS.tools.length });
  const names = raw.filter((name) => LIMITS.tools.includes(name));
  return names.length ? names : undefined;
}

function compact(obj) {
  return Object.fromEntries(Object.entries(obj).filter(([, value]) => value !== undefined));
}

module.exports = { readSettings, LIMITS };
