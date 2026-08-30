/**
 * Per-request generation settings.
 *
 * The website sends these with a chat message or a generation request. Only
 * what the caller sets is forwarded; anything omitted is left to the model's
 * own defaults rather than being pinned here.
 */

const modelLimits = require("./modelLimits");

// The documented bounds. max_new_tokens is the floor of what is accepted: the
// runtime ceiling rises with the window of the model actually loaded, so the
// published schema stays stable while a large preset is still usable.
// The canonical size vocabulary, kept here rather than in the routes so the
// chat path and the REST routes cannot drift apart. The settings panel collects
// an aspect ratio and a size tier and the chat path used to drop both, which is
// the only path the interface actually uses.
const ASPECT_RATIOS = ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "21:9"];
const SIZE_TIERS = [256, 512, 768, 1024];

const LIMITS = {
  temperature: { min: 0.1, max: 2 },
  top_p: { min: 0.1, max: 1 },
  top_k: { min: 0, max: 200 },
  max_new_tokens: { min: 16, max: 2048 },
  resolution: [64, 128, 256, 512],
  // Duration is bounded by the overlapped-window path now, not by one window.
  num_frames: { min: 1, max: 512 },
  fps: { min: 1, max: 60 },
  aspect: ASPECT_RATIOS,
  tier: SIZE_TIERS,
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
    max_new_tokens: v.integer("max_new_tokens", {
      ...LIMITS.max_new_tokens,
      max: modelLimits.maxNewTokens(),
    }),
    resolution: v.oneOf("resolution", LIMITS.resolution),
    num_frames: v.integer("num_frames", LIMITS.num_frames),
    fps: v.integer("fps", LIMITS.fps),
    aspect: v.oneOf("aspect", LIMITS.aspect),
    tier: v.oneOf("tier", LIMITS.tier),
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

module.exports = { readSettings, LIMITS, ASPECT_RATIOS, SIZE_TIERS };
