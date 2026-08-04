/**
 * Request validation.
 *
 * A small dependency-free validator. Each reader pulls one field out of a
 * source object (`req.body`, `req.params`, ...), returns a normalised value,
 * and collects a field error instead of throwing, so a single response can
 * report every problem at once. Call `done()` to raise a 400 if anything failed.
 */

const { ApiError } = require("./errors");

class Validator {
  constructor(source, { prefix = "", errors = [] } = {}) {
    this.source = source || {};
    this.prefix = prefix;
    this.errors = errors;
  }

  fail(field, message) {
    this.errors.push({ field: this.prefix + field, message });
  }

  /**
   * A validator over a nested object. It shares this one's error list, so
   * `done()` on the parent still reports everything, with dotted field names.
   */
  nested(field) {
    const raw = this.source[field];
    const options = { prefix: `${this.prefix}${field}.`, errors: this.errors };

    if (raw === undefined || raw === null) return new Validator({}, options);
    if (typeof raw !== "object" || Array.isArray(raw)) {
      this.fail(field, "must be an object");
      return new Validator({}, options);
    }
    return new Validator(raw, options);
  }

  /**
   * A trimmed string. Empty strings count as missing.
   */
  string(field, { required = false, min = 1, max = 8000, fallback = undefined } = {}) {
    const raw = this.source[field];

    if (raw === undefined || raw === null || raw === "") {
      if (required) this.fail(field, "is required");
      return fallback;
    }
    if (typeof raw !== "string") {
      this.fail(field, "must be a string");
      return fallback;
    }

    const value = raw.trim();
    if (value.length < min) {
      this.fail(field, required && !value.length ? "is required" : `must be at least ${min} characters`);
      return fallback;
    }
    if (value.length > max) {
      this.fail(field, `must be at most ${max} characters`);
      return fallback;
    }
    return value;
  }

  /**
   * An integer, optionally bounded. Numeric strings are accepted so the same
   * validator works for JSON bodies and multipart form fields.
   */
  integer(field, { required = false, min, max, fallback = undefined } = {}) {
    const raw = this.source[field];

    if (raw === undefined || raw === null || raw === "") {
      if (required) this.fail(field, "is required");
      return fallback;
    }

    const value = typeof raw === "string" ? Number(raw) : raw;
    if (typeof value !== "number" || !Number.isFinite(value) || !Number.isInteger(value)) {
      this.fail(field, "must be an integer");
      return fallback;
    }
    if (min !== undefined && value < min) {
      this.fail(field, `must be at least ${min}`);
      return fallback;
    }
    if (max !== undefined && value > max) {
      this.fail(field, `must be at most ${max}`);
      return fallback;
    }
    return value;
  }

  /**
   * A finite number, optionally bounded. Used for the sampling controls, which
   * are fractional.
   */
  number(field, { required = false, min, max, fallback = undefined } = {}) {
    const raw = this.source[field];

    if (raw === undefined || raw === null || raw === "") {
      if (required) this.fail(field, "is required");
      return fallback;
    }

    const value = typeof raw === "string" ? Number(raw) : raw;
    if (typeof value !== "number" || !Number.isFinite(value)) {
      this.fail(field, "must be a number");
      return fallback;
    }
    if (min !== undefined && value < min) {
      this.fail(field, `must be at least ${min}`);
      return fallback;
    }
    if (max !== undefined && value > max) {
      this.fail(field, `must be at most ${max}`);
      return fallback;
    }
    return value;
  }

  /**
   * A value restricted to a known set.
   */
  oneOf(field, values, { required = false, fallback = undefined } = {}) {
    const raw = this.source[field];

    if (raw === undefined || raw === null || raw === "") {
      if (required) this.fail(field, "is required");
      return fallback;
    }
    if (!values.includes(raw)) {
      this.fail(field, `must be one of: ${values.join(", ")}`);
      return fallback;
    }
    return raw;
  }

  /**
   * An array, length capped so a single request cannot carry unbounded items.
   */
  array(field, { max = 20, fallback = [] } = {}) {
    const raw = this.source[field];

    if (raw === undefined || raw === null) return fallback;
    if (!Array.isArray(raw)) {
      this.fail(field, "must be an array");
      return fallback;
    }
    if (raw.length > max) {
      this.fail(field, `must contain at most ${max} items`);
      return fallback;
    }
    return raw;
  }

  /**
   * A UUID, used for the conversation identifiers in the chat routes.
   */
  uuid(field) {
    const raw = this.source[field];
    const pattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

    if (typeof raw !== "string" || !pattern.test(raw)) {
      this.fail(field, "must be a valid id");
      return undefined;
    }
    return raw;
  }

  /**
   * Raise a single 400 listing every field that failed.
   */
  done() {
    if (this.errors.length) {
      throw ApiError.badRequest("Request validation failed", this.errors);
    }
  }
}

function validator(source) {
  return new Validator(source);
}

module.exports = { validator, Validator };
