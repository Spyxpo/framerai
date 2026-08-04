/**
 * The limiters the app shares.
 *
 * Generation has one set of buckets covering the REST routes, chat messages,
 * and WebSocket frames, so a client cannot sidestep the limit by switching
 * transport.
 */

const config = require("../config");
const { rateLimit, FixedWindowCounter } = require("./rateLimit");

const apiCounter = new FixedWindowCounter({
  windowMs: config.rateLimit.windowMs,
  max: config.rateLimit.api,
});

const generationCounter = new FixedWindowCounter({
  windowMs: config.rateLimit.windowMs,
  max: config.rateLimit.generate,
});

module.exports = {
  apiLimiter: rateLimit(apiCounter, "Too many requests"),
  generationLimiter: rateLimit(generationCounter, "Too many generation requests"),
  generationCounter,
};
