/**
 * Minimal structured logger.
 *
 * Emits newline-delimited JSON in production and a compact human-readable line
 * in development.  No dependencies beyond the Node.js standard library.
 *
 * Environment variables:
 *   LOG_LEVEL   - error | warn | info | debug  (default: info)
 *   LOG_FORMAT  - json | pretty                 (default: json;
 *                                                pretty when NODE_ENV=development)
 */

const LEVELS = { error: 0, warn: 1, info: 2, debug: 3 };

const configuredLevel = (process.env.LOG_LEVEL || "info").toLowerCase();
const maxLevel = LEVELS[configuredLevel] ?? LEVELS.info;

const defaultFormat =
  process.env.NODE_ENV === "development" ? "pretty" : "json";
const format = (process.env.LOG_FORMAT || defaultFormat).toLowerCase();

function write(level, context, message, extra) {
  if ((LEVELS[level] ?? 0) > maxLevel) return;

  const entry = {
    level,
    timestamp: new Date().toISOString(),
    ...(context.requestId ? { requestId: context.requestId } : {}),
    ...(context.route ? { route: context.route } : {}),
    ...(context.connectionId ? { connectionId: context.connectionId } : {}),
    message,
    ...extra,
  };

  if (format === "pretty") {
    const id = entry.requestId ? ` [${entry.requestId}]` : "";
    const conn = entry.connectionId ? ` [ws:${entry.connectionId}]` : "";
    const route = entry.route ? ` ${entry.route}` : "";
    // eslint-disable-next-line no-console
    console.log(`${entry.timestamp} ${level.toUpperCase()}${id}${conn}${route} ${message}`);
  } else {
    // eslint-disable-next-line no-console
    console.log(JSON.stringify(entry));
  }
}

/**
 * Create a child logger bound to a fixed context object.
 * Additional context keys can be passed on each call as the last argument.
 *
 * @param {object} context - e.g. { requestId, route } or { connectionId }
 */
function createLogger(context = {}) {
  return {
    error: (message, extra = {}) => write("error", context, message, extra),
    warn:  (message, extra = {}) => write("warn",  context, message, extra),
    info:  (message, extra = {}) => write("info",  context, message, extra),
    debug: (message, extra = {}) => write("debug", context, message, extra),
    /** Return a new child logger that merges extra context */
    child: (extra) => createLogger({ ...context, ...extra }),
  };
}

/** Root logger (no context) */
const logger = createLogger();

module.exports = { logger, createLogger, LEVELS, _maxLevel: () => maxLevel };
