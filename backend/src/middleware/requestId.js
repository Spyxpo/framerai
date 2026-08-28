/**
 * Request ID middleware.
 *
 * Accepts an inbound X-Request-Id header when present; generates a new UUID
 * when absent.  The ID is stored on req.requestId and echoed in the response
 * via X-Request-Id so callers can correlate logs.
 */

const { randomUUID } = require("node:crypto");

function requestIdMiddleware(req, res, next) {
  const id = req.headers["x-request-id"] || randomUUID();
  req.requestId = id;
  res.setHeader("X-Request-Id", id);
  next();
}

module.exports = { requestIdMiddleware };
