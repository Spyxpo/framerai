/**
 * Error handling.
 *
 * Every failure leaves the API in the same shape so clients can rely on it:
 *
 *   { "error": "human readable message", "code": "MACHINE_CODE", "details": [...] }
 *
 * `details` is only present for validation failures. `error` stays a plain
 * string because that is what the website already reads.
 */

const { MulterError } = require("multer");
const { createLogger } = require("../services/logger");

class ApiError extends Error {
  constructor(status, message, code = "ERROR", details) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    if (details) this.details = details;
  }

  static badRequest(message, details) {
    return new ApiError(400, message, "VALIDATION_ERROR", details);
  }

  static notFound(message) {
    return new ApiError(404, message, "NOT_FOUND");
  }

  static payloadTooLarge(message) {
    return new ApiError(413, message, "PAYLOAD_TOO_LARGE");
  }
}

/**
 * Wrap an async route handler so a rejected promise reaches the error handler.
 */
function asyncHandler(fn) {
  return (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);
}

/**
 * Anything that did not match a route.
 */
function notFoundHandler(req, res, next) {
  next(ApiError.notFound(`Cannot ${req.method} ${req.originalUrl}`));
}

/**
 * Translate a thrown error into the standard response body. The fourth
 * argument is what marks this as an error handler to Express, so it stays even
 * though it is unused.
 */
function errorHandler(err, req, res, next) {
  const normalized = normalize(err);

  if (normalized.status >= 500) {
    const log = createLogger({ requestId: req.requestId, route: `${req.method} ${req.originalUrl}` });
    log.error("unhandled error", { status: normalized.status, error: err.message });
  }

  const body = { error: normalized.message, code: normalized.code };
  if (normalized.details) body.details = normalized.details;
  if (req.requestId) body.requestId = req.requestId;

  res.status(normalized.status).json(body);
}

function normalize(err) {
  if (err instanceof ApiError) {
    return { status: err.status, message: err.message, code: err.code, details: err.details };
  }

  if (err instanceof MulterError) {
    if (err.code === "LIMIT_FILE_SIZE") {
      return { status: 413, message: "Uploaded file is too large", code: "PAYLOAD_TOO_LARGE" };
    }
    return {
      status: 400,
      message: err.message,
      code: "UPLOAD_ERROR",
      details: err.field ? [{ field: err.field, message: err.message }] : undefined,
    };
  }

  // Raised by express.json() for malformed or oversized bodies.
  if (err.type === "entity.parse.failed") {
    return { status: 400, message: "Request body is not valid JSON", code: "INVALID_JSON" };
  }
  if (err.type === "entity.too.large") {
    return { status: 413, message: "Request body is too large", code: "PAYLOAD_TOO_LARGE" };
  }

  return {
    status: err.status || err.statusCode || 500,
    message: err.message || "Internal server error",
    code: "INTERNAL_ERROR",
  };
}

module.exports = { ApiError, asyncHandler, notFoundHandler, errorHandler };
