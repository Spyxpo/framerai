/**
 * Rate limiting.
 *
 * Fixed-window counters held in memory and keyed by client address. That is
 * enough for the single backend process this project ships; limiting across
 * replicas would need a shared store such as Redis.
 */

const proxyaddr = require("proxy-addr");
const { ApiError } = require("./errors");

class FixedWindowCounter {
  constructor({ windowMs, max }) {
    this.windowMs = windowMs;
    this.max = max;
    this.buckets = new Map(); // key -> { count, resetAt }
    this.lastSweep = 0;
  }

  get enabled() {
    return this.max > 0;
  }

  /**
   * Record a hit and report where the caller stands in the current window.
   */
  hit(key, now = Date.now()) {
    this.sweep(now);

    let bucket = this.buckets.get(key);
    if (!bucket || bucket.resetAt <= now) {
      bucket = { count: 0, resetAt: now + this.windowMs };
      this.buckets.set(key, bucket);
    }
    bucket.count += 1;

    const retryAfter = Math.max(1, Math.ceil((bucket.resetAt - now) / 1000));
    return {
      allowed: bucket.count <= this.max,
      remaining: Math.max(0, this.max - bucket.count),
      retryAfter,
    };
  }

  /**
   * Drop expired buckets so the map does not grow with every unique client.
   * Runs at most once per window.
   */
  sweep(now) {
    if (now - this.lastSweep < this.windowMs) return;
    this.lastSweep = now;
    for (const [key, bucket] of this.buckets) {
      if (bucket.resetAt <= now) this.buckets.delete(key);
    }
  }
}

/**
 * Express middleware around a counter. A max of 0 or less turns the limiter
 * off, which is what tests and local development usually want.
 *
 * The counter is passed in rather than created here so the same buckets can be
 * shared, for example between the REST generation routes and WebSocket chat.
 */
function rateLimit(counter, message = "Too many requests") {
  return (req, res, next) => {
    if (!counter.enabled) return next();

    const key = req.ip || req.socket.remoteAddress || "unknown";
    const { allowed, remaining, retryAfter } = counter.hit(key);

    res.setHeader("RateLimit-Limit", counter.max);
    res.setHeader("RateLimit-Remaining", remaining);
    res.setHeader("RateLimit-Reset", retryAfter);

    if (!allowed) {
      res.setHeader("Retry-After", retryAfter);
      return next(new ApiError(429, `${message}. Try again in ${retryAfter}s.`, "RATE_LIMITED"));
    }
    next();
  };
}

/**
 * Resolve the real client IP from an HTTP(S)/WebSocket upgrade request using
 * the same proxy-trust semantics that Express uses for req.ip.
 *
 * Mirrors Express compileTrust (lib/utils.js) so that WebSocket upgrade
 * requests and REST requests from the same physical client always resolve to
 * the same key and share one rate-limit bucket.
 *
 * Trust values follow Express semantics exactly:
 *   false / falsy  → trust nobody  (socket address always returned)
 *   true           → trust all hops
 *   number n       → trust n hops  (Express uses `(addr, i) => i < n`;
 *                                   proxyaddr.compile does not accept numbers)
 *   string/array   → named subnets / CIDRs passed to proxyaddr.compile
 *
 * @param {import("http").IncomingMessage} req - The raw HTTP upgrade request.
 * @param {boolean|number|string} trustProxy    - The value of config.trustProxy.
 * @returns {string} The resolved client IP address.
 */
function resolveClientIp(req, trustProxy) {
  try {
    // Mirror Express compileTrust (lib/utils.js) exactly:
    //   false / falsy  → trust nobody
    //   true           → trust all hops
    //   number n       → trust n hops  (proxyaddr.compile does NOT accept numbers)
    //   string/array   → named subnets / CIDRs via proxyaddr.compile
    let trust;
    if (trustProxy === true) {
      trust = () => true;
    } else if (typeof trustProxy === "number") {
      trust = (addr, i) => i < trustProxy;
    } else {
      trust = proxyaddr.compile(trustProxy || []);
    }
    return proxyaddr(req, trust);
  } catch {
    // Fallback to socket address if proxyaddr fails (e.g. malformed header)
    return req?.socket?.remoteAddress || "unknown";
  }
}

module.exports = { rateLimit, FixedWindowCounter, resolveClientIp };
