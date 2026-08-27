/**
 * Tests for request ID middleware and structured logging (Issue #161).
 */

const test = require("node:test");
const assert = require("node:assert");
const request = require("supertest");
const { EventEmitter } = require("events");
const { mockModel, loadApp } = require("./helpers");

// ── Setup shared mock ────────────────────────────────────────────────────────
// All HTTP/middleware tests share one app instance, per the existing pattern.
const calls = mockModel();
const app = loadApp();

// ── Request-ID middleware ────────────────────────────────────────────────────

test("request ID middleware: generates a UUID when X-Request-Id is absent", async () => {
  const res = await request(app).get("/api/chat/conversations");
  assert.strictEqual(res.status, 200);
  assert.ok(res.headers["x-request-id"], "X-Request-Id header must be present");
  assert.match(
    res.headers["x-request-id"],
    /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/,
    "generated ID must be a valid UUID"
  );
});

test("request ID middleware: preserves supplied X-Request-Id", async () => {
  const suppliedId = "test-request-123";
  const res = await request(app)
    .get("/api/chat/conversations")
    .set("X-Request-Id", suppliedId);
  assert.strictEqual(res.status, 200);
  assert.strictEqual(res.headers["x-request-id"], suppliedId);
});

test("request ID middleware: echoes ID in error response body", async () => {
  const res = await request(app).get("/api/chat/conversations/not-a-uuid");
  assert.strictEqual(res.status, 400);
  assert.ok(res.body.requestId, "error body must include requestId");
  assert.strictEqual(res.headers["x-request-id"], res.body.requestId);
});

test("request ID middleware: requestId flows through to model service", async () => {
  // Clear both app and chat route caches so the fresh mock is wired in
  delete require.cache[require.resolve("../src/app")];
  delete require.cache[require.resolve("../src/routes/chat")];
  const freshCalls = mockModel();
  const freshApp = loadApp();
  const suppliedId = "flow-test-456";

  const convRes = await request(freshApp).post("/api/chat/conversations");
  const convId = convRes.body.id;

  await request(freshApp)
    .post(`/api/chat/conversations/${convId}/messages`)
    .set("X-Request-Id", suppliedId)
    .send({ content: "hello", type: "text" });

  const call = freshCalls.find((c) => c.name === "processMessage");
  assert.ok(call, "processMessage must be called");
  assert.strictEqual(
    call.args[3],
    suppliedId,
    "requestId must be the 4th argument to processMessage"
  );
});

// ── Structured logger ────────────────────────────────────────────────────────

test("logger: outputs valid JSON with required fields", () => {
  const { createLogger } = require("../src/services/logger");
  const log = createLogger({ route: "test-route" });

  const lines = [];
  const originalWrite = process.stdout.write.bind(process.stdout);
  process.stdout.write = (chunk) => { lines.push(chunk); return true; };

  try {
    log.info("test message", { key: "value" });
  } finally {
    process.stdout.write = originalWrite;
  }

  const parsed = JSON.parse(lines.join("").trim().split("\n").pop());
  assert.strictEqual(parsed.level, "info");
  assert.strictEqual(parsed.message, "test message");
  assert.strictEqual(parsed.route, "test-route");
  assert.strictEqual(parsed.key, "value");
  assert.ok(parsed.timestamp, "must include timestamp");
});

test("logger: respects LOG_LEVEL env var", () => {
  const originalEnv = process.env.LOG_LEVEL;

  try {
    process.env.LOG_LEVEL = "warn";
    // Reload so the module re-reads LOG_LEVEL at parse time
    delete require.cache[require.resolve("../src/services/logger")];
    const { createLogger } = require("../src/services/logger");
    const log = createLogger();

    const lines = [];
    const originalWrite = process.stdout.write.bind(process.stdout);
    process.stdout.write = (chunk) => { lines.push(chunk); return true; };

    try {
      log.info("filtered");
      log.debug("also filtered");
      log.warn("appears");
      log.error("also appears");
    } finally {
      process.stdout.write = originalWrite;
    }

    const parsed = lines.join("").trim().split("\n").filter(Boolean).map((l) => JSON.parse(l));
    assert.strictEqual(parsed.length, 2, "only warn and error must be emitted");
    assert.deepStrictEqual(parsed.map((e) => e.level), ["warn", "error"]);
  } finally {
    process.env.LOG_LEVEL = originalEnv;
    delete require.cache[require.resolve("../src/services/logger")];
  }
});

test("logger: context (requestId, route) appears in every emitted line", () => {
  const { createLogger } = require("../src/services/logger");
  const log = createLogger({ requestId: "ctx-123", route: "test" });

  const lines = [];
  const originalWrite = process.stdout.write.bind(process.stdout);
  process.stdout.write = (chunk) => { lines.push(chunk); return true; };

  try {
    log.info("msg1");
    log.warn("msg2", { extra: "data" });
  } finally {
    process.stdout.write = originalWrite;
  }

  const parsed = lines.join("").trim().split("\n").filter(Boolean).map((l) => JSON.parse(l));
  assert.ok(parsed.length >= 2, "must have at least 2 entries");
  for (const entry of parsed) {
    assert.strictEqual(entry.requestId, "ctx-123");
    assert.strictEqual(entry.route, "test");
  }
});

// ── requestId propagation through bridge ────────────────────────────────────
// Uses the same MockChildProcess + _setTimerImpl approach as pythonBridge.test.js
// so the 60 s startup timeout is never engaged.

{
  // Module-level mock (same file, same process) — must be set before bridge loads
  let mockSpawn = null;
  const originalSpawn = require("child_process").spawn;
  require("child_process").spawn = (...a) => (mockSpawn ? mockSpawn(...a) : originalSpawn(...a));

  class MockChild extends EventEmitter {
    constructor() {
      super();
      this.lastPayload = null;
      this.stdin = {
        write: (data) => {
          this.lastPayload = JSON.parse(data.toString().trim());
        },
      };
      this.stdout = new EventEmitter();
      this.stderr = new EventEmitter();
    }
    kill() { this.emit("exit", 0); }
    removeAllListeners() {
      super.removeAllListeners();
      this.stdout.removeAllListeners();
      this.stderr.removeAllListeners();
    }
    simulateReady() {
      this.stdout.emit("data", Buffer.from(JSON.stringify({ ready: true }) + "\n"));
    }
    simulateResponse(id, result) {
      this.stdout.emit("data", Buffer.from(JSON.stringify({ id, ok: true, result }) + "\n"));
    }
  }

  function makeBridge() {
    delete require.cache[require.resolve("../src/services/pythonBridge")];
    const fs = require("fs");
    const origExists = fs.existsSync;
    fs.existsSync = (p) => (p === "/fake/model.pt" ? true : origExists(p));

    process.env.MODEL_PATH = "/fake/model.pt";
    process.env.MODEL_ENABLED = "true";
    process.env.MODEL_WORKERS = "1";

    const bridge = require("../src/services/pythonBridge");

    // Replace timers so the 60 s startup timeout never fires
    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        // Startup / stability timers (>=10 000 ms) are held pending; backoff
        // timers (<10 000 ms) are fired immediately.
        if (ms < 10000) { fn(); return null; }
        return { _held: true };
      },
      (_id) => {} // clearTimeout no-op for held timers
    );

    return { bridge, prev, restoreExists: () => { fs.existsSync = origExists; } };
  }

  test("bridge: requestId is written into worker payload", async () => {
    let worker = null;
    mockSpawn = () => {
      worker = new MockChild();
      return worker;
    };

    const { bridge, prev, restoreExists } = makeBridge();

    try {
      const startP = bridge.start();
      await new Promise((r) => setImmediate(r));
      worker.simulateReady();
      await startP;

      const testId = "bridge-req-abc";
      const reqP = bridge.request("chat", { prompt: "hi" }, testId);
      await new Promise((r) => setImmediate(r));

      assert.ok(worker.lastPayload, "stdin.write must have been called");
      assert.strictEqual(worker.lastPayload.requestId, testId,
        "requestId MUST be in the worker payload");

      worker.simulateResponse(worker.lastPayload.id, { content: "ok" });
      await reqP;
    } finally {
      bridge._pool()?.shutdown();
      bridge._setTimerImpl(prev.set, prev.clear);
      restoreExists();
      delete require.cache[require.resolve("../src/services/pythonBridge")];
    }
  });

  test("bridge: requestId is absent from payload when not provided", async () => {
    let worker = null;
    mockSpawn = () => {
      worker = new MockChild();
      return worker;
    };

    const { bridge, prev, restoreExists } = makeBridge();

    try {
      const startP = bridge.start();
      await new Promise((r) => setImmediate(r));
      worker.simulateReady();
      await startP;

      const reqP = bridge.request("chat", { prompt: "hi" }); // no requestId
      await new Promise((r) => setImmediate(r));

      assert.ok(worker.lastPayload, "stdin.write must have been called");
      assert.ok(
        !Object.prototype.hasOwnProperty.call(worker.lastPayload, "requestId"),
        "requestId key must NOT exist when not provided"
      );

      worker.simulateResponse(worker.lastPayload.id, { content: "ok" });
      await reqP;
    } finally {
      bridge._pool()?.shutdown();
      bridge._setTimerImpl(prev.set, prev.clear);
      restoreExists();
      delete require.cache[require.resolve("../src/services/pythonBridge")];
      mockSpawn = null;
    }
  });
}

// ── Regression: X-Request-Id header missing ──────────────────────────────────
// Proves that removing res.setHeader("X-Request-Id") breaks header echoing.

test("regression: missing X-Request-Id header is detectable", async () => {
  const middlewarePath = require.resolve("../src/middleware/requestId");
  const appPath = require.resolve("../src/app");
  const original = require.cache[middlewarePath];

  // Install broken middleware — no header set
  require.cache[middlewarePath] = {
    id: middlewarePath, filename: middlewarePath, loaded: true,
    exports: {
      requestIdMiddleware: (req, res, next) => {
        req.requestId = "broken-id";
        // deliberately no res.setHeader
        next();
      },
    },
  };
  delete require.cache[appPath];

  try {
    const { createApp } = require("../src/app");
    const brokenApp = createApp();
    const res = await request(brokenApp).get("/api/chat/conversations");
    // With broken middleware the header is absent
    assert.ok(!res.headers["x-request-id"], "header must NOT be present with broken middleware");
  } finally {
    if (original) require.cache[middlewarePath] = original;
    delete require.cache[appPath];
  }
});

// ── Regression: requestId absent from error body ─────────────────────────────

test("regression: requestId absent from error body is detectable", async () => {
  const errPath = require.resolve("../src/middleware/errors");
  const appPath = require.resolve("../src/app");
  const originalModule = require.cache[errPath];

  // Reload errors module fresh, then patch errorHandler to strip requestId
  delete require.cache[errPath];
  const errModule = require(errPath);
  const realHandler = errModule.errorHandler;
  errModule.errorHandler = (err, req, res, next) => {
    const origJson = res.json.bind(res);
    res.json = (body) => { delete body.requestId; return origJson(body); };
    return realHandler(err, req, res, next);
  };

  delete require.cache[appPath];

  try {
    const { createApp } = require("../src/app");
    const brokenApp = createApp();
    const res = await request(brokenApp).get("/api/chat/conversations/invalid-uuid");
    assert.strictEqual(res.status, 400);
    assert.ok(!res.body.requestId, "requestId must NOT be in body when handler is broken");
  } finally {
    if (originalModule) require.cache[errPath] = originalModule;
    else delete require.cache[errPath];
    delete require.cache[appPath];
  }
});

// ── Regression: LOG_LEVEL filtering ──────────────────────────────────────────
// Proves that the level guard actually silences sub-threshold logs.
// We can't "break" production code here, so instead we verify the positive and
// negative cases: at LOG_LEVEL=error, info is silent; at LOG_LEVEL=info it appears.

test("regression: LOG_LEVEL=error silences info, LOG_LEVEL=info does not", () => {
  const loggerPath = require.resolve("../src/services/logger");
  const originalEnv = process.env.LOG_LEVEL;

  function captureInfoLine(level) {
    process.env.LOG_LEVEL = level;
    delete require.cache[loggerPath];
    const { createLogger } = require(loggerPath);
    const log = createLogger();
    const lines = [];
    const origWrite = process.stdout.write.bind(process.stdout);
    process.stdout.write = (c) => { lines.push(c); return true; };
    try {
      log.info("probe");
    } finally {
      process.stdout.write = origWrite;
    }
    return lines.filter(Boolean);
  }

  try {
    const atError = captureInfoLine("error");
    assert.strictEqual(atError.length, 0,
      "info must be filtered when LOG_LEVEL=error (this IS the regression check: without the guard this would fail)");

    const atInfo = captureInfoLine("info");
    assert.ok(atInfo.length > 0,
      "info must appear when LOG_LEVEL=info");

    const parsed = JSON.parse(atInfo[0]);
    assert.strictEqual(parsed.level, "info");
    assert.strictEqual(parsed.message, "probe");
  } finally {
    process.env.LOG_LEVEL = originalEnv;
    delete require.cache[loggerPath];
  }
});
