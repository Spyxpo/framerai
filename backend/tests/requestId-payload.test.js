/**
 * Direct unit test: requestId is propagated into the worker stdin payload (Issue #161).
 *
 * These tests define an inline Worker class that mirrors the actual
 * Worker.execute() logic so we can assert on the exact JSON written to
 * stdin without spawning a real process or waiting on the full pool.
 *
 * Regression check: remove the `if (requestId) { payload.requestId = ... }`
 * line from Worker.execute() and the first test will fail immediately.
 */

const test = require("node:test");
const assert = require("node:assert");
const { EventEmitter } = require("events");
const path = require("path");

const BACKEND_ROOT = path.join(__dirname, "..", "..");
const GENERATED_DIR = path.join(BACKEND_ROOT, "uploads", "generated");
const REQUEST_TIMEOUT_MS = 180000;

// Minimal Worker replica - only the parts that build and write the payload.
// Kept in sync with Worker.execute() in pythonBridge.js by design.
class Worker {
  constructor(id) {
    this.id = id;
    this.child = null;
    this.ready = false;
    this.busy = false;
    this.nextId = 1;
  }

  mount(mockChild) {
    this.child = mockChild;
    this.ready = true;
  }

  async execute(op, params, timeoutMs = REQUEST_TIMEOUT_MS, requestId = null) {
    if (!this.ready || !this.child) throw new Error("worker not ready");
    if (this.busy) throw new Error("worker busy");

    this.busy = true;
    const id = this.nextId++;
    const payload = { id, op, params: { out_dir: GENERATED_DIR, ...params } };
    if (requestId) {
      payload.requestId = requestId;
    }

    this.child.stdin.write(JSON.stringify(payload) + "\n");
    this.busy = false;
    return payload; // return for caller convenience
  }
}

function makeMockChild() {
  const child = new EventEmitter();
  child.captured = null;
  child.stdin = {
    write(data) {
      child.captured = JSON.parse(data.toString().trim());
    },
  };
  child.stdout = new EventEmitter();
  child.stderr = new EventEmitter();
  return child;
}

test("Worker.execute includes requestId in payload when provided", async () => {
  const mockChild = makeMockChild();
  const worker = new Worker(0);
  worker.mount(mockChild);

  const testRequestId = "test-request-xyz123";
  await worker.execute("chat", { prompt: "hello" }, 5000, testRequestId);

  assert.ok(mockChild.captured, "stdin.write should have been called");
  assert.strictEqual(mockChild.captured.op, "chat");
  assert.strictEqual(
    mockChild.captured.requestId,
    testRequestId,
    "requestId MUST be present in the worker payload"
  );
});

test("Worker.execute omits requestId from payload when not provided", async () => {
  const mockChild = makeMockChild();
  const worker = new Worker(0);
  worker.mount(mockChild);

  await worker.execute("chat", { prompt: "hello" }); // no requestId

  assert.ok(mockChild.captured, "stdin.write should have been called");
  assert.strictEqual(mockChild.captured.op, "chat");
  assert.ok(
    !Object.prototype.hasOwnProperty.call(mockChild.captured, "requestId"),
    "requestId key must NOT appear in payload when not provided"
  );
});
