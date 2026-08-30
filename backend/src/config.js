/**
 * Runtime limits.
 *
 * Everything here is overridable through the environment so a deployment can
 * tighten or loosen limits without a code change. See backend/.env.example.
 */

function int(name, fallback) {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  const value = Number(raw);
  return Number.isFinite(value) ? value : fallback;
}

/**
 * Express `trust proxy`. Behind a reverse proxy this has to be set, otherwise
 * every request looks like it comes from the proxy and shares one rate limit
 * bucket. Accepts a hop count, `true`/`false`, or a named value such as
 * `loopback`.
 */
function trustProxy() {
  const raw = (process.env.TRUST_PROXY || "").trim();
  if (!raw || raw === "false") return false;
  if (raw === "true") return true;
  const hops = Number(raw);
  return Number.isFinite(hops) ? hops : raw;
}

module.exports = {
  // Largest JSON body express will parse. A prompt filling a million-token
  // window is a few megabytes of text, so 1mb made the largest presets
  // unreachable through the API whatever their own limits said.
  jsonBodyLimit: process.env.JSON_BODY_LIMIT || "8mb",

  // Largest single upload accepted by the image and audio routes.
  maxFileSize: int("MAX_FILE_SIZE", 50 * 1024 * 1024),

  // Largest WebSocket frame accepted before the connection is closed. Matches
  // the JSON body limit, so the streaming path is not the narrower one.
  maxWsPayload: int("MAX_WS_PAYLOAD", 8 * 1024 * 1024),

  trustProxy: trustProxy(),

  rateLimit: {
    windowMs: int("RATE_LIMIT_WINDOW_MS", 60 * 1000),
    // All /api traffic from one client.
    api: int("RATE_LIMIT_MAX", 300),
    // Generation specifically, which is what costs real work. Also applied to
    // sending a chat message and to WebSocket chat frames.
    generate: int("GENERATE_RATE_LIMIT_MAX", 20),
  },
};
