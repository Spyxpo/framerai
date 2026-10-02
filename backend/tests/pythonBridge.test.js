/**
 * Tests for the Python inference bridge worker pool.
 *
 * These tests verify that the bridge correctly manages multiple workers,
 * handles concurrency without head-of-line blocking, queues requests when
 * all workers are busy, and automatically restarts workers that exit.
 */

const { describe, it, before, after, beforeEach, afterEach } = require("node:test");
const assert = require("node:assert");
const { EventEmitter } = require("events");

// Mock child_process at the module level before requiring pythonBridge
let mockSpawn = null;
let spawnedProcesses = [];

// Create a mock ChildProcess class
class MockChildProcess extends EventEmitter {
  constructor(command, args, options) {
    super();
    this.command = command;
    this.args = args;
    this.options = options;
    this.killed = false;
    // Mirrors real ChildProcess: set by the runtime when the process actually
    // terminates, independent of any 'exit' listeners attached to it. Tests
    // that check escalation logic against a listener-stripped child rely on
    // this pair rather than an emitted event reaching a (possibly removed)
    // listener.
    this.exitCode = null;
    this.signalCode = null;
    // A real child's stdin is a stream, so it can emit 'error' (EPIPE once the worker
    // is gone), and the bridge listens for that. Like the real one, it is a separate
    // emitter from the ChildProcess: removeAllListeners() below leaves it alone.
    this.stdin = Object.assign(new EventEmitter(), {
      write: (data) => {
        this.lastWrite = data;
        if (this.onStdinWrite) this.onStdinWrite(data);
      },
    });
    this.stdout = new EventEmitter();
    this.stderr = new EventEmitter();
    spawnedProcesses.push(this);
  }

  kill(signal) {
    this.killed = true;
    // A cooperative process terminated by a signal reports a null exit code
    // and the signal that killed it - matches real ChildProcess semantics.
    this.signalCode = signal || "SIGTERM";
    this.emit("exit", 0);
  }

  removeAllListeners() {
    super.removeAllListeners();
    this.stdout.removeAllListeners();
    this.stderr.removeAllListeners();
  }

  simulateReady(success = true) {
    const msg = success ? { ready: true } : { ready: false, error: "mock error" };
    this.stdout.emit("data", Buffer.from(JSON.stringify(msg) + "\n"));
  }

  simulateResponse(id, ok, result) {
    const msg = ok ? { id, ok: true, result } : { id, ok: false, error: result };
    this.stdout.emit("data", Buffer.from(JSON.stringify(msg) + "\n"));
  }

  simulateExit(code = 0) {
    this.exitCode = code;
    this.emit("exit", code);
  }
}

// Replace spawn globally
const originalSpawn = require("child_process").spawn;
require("child_process").spawn = function (...args) {
  if (mockSpawn) {
    return mockSpawn(...args);
  }
  return originalSpawn(...args);
};

describe("pythonBridge worker pool", () => {
  let bridge = null;
  let originalExistsSync = null;

  beforeEach(() => {
    // Reset module state
    delete require.cache[require.resolve("../src/services/pythonBridge")];
    spawnedProcesses.length = 0; // Clear array without reassigning

    // Setup environment
    process.env.MODEL_ENABLED = "true";
    process.env.MODEL_PATH = "/fake/model.pt";
    process.env.TOKENIZER_PATH = "/fake/tokenizer";
    process.env.MODEL_WORKERS = "2";
    delete process.env.MODEL_TOOLS;
    delete process.env.MODEL_CLI_MODE;
    delete process.env.MODEL_CLI_ROOT;
    delete process.env.MODEL_TIMEOUT_MS;
    delete process.env.MODEL_STARTUP_TIMEOUT_MS;

    // Mock fs.existsSync to return true for model path
    const fs = require("fs");
    originalExistsSync = fs.existsSync;
    fs.existsSync = (path) => {
      if (path.includes("model.pt")) return true;
      return originalExistsSync(path);
    };

    // Mock spawn to return our mock processes
    mockSpawn = (command, args, options) => {
      return new MockChildProcess(command, args, options);
    };
  });

  afterEach(() => {
    // Restore fs.existsSync
    if (originalExistsSync) {
      const fs = require("fs");
      fs.existsSync = originalExistsSync;
      originalExistsSync = null;
    }

    // Cleanup
    mockSpawn = null;
    spawnedProcesses.length = 0; // Clear array without reassigning
    if (bridge && bridge._pool && bridge._pool()) {
      try {
        bridge._pool().shutdown();
      } catch (e) {
        // ignore
      }
    }
  });

  it("should spawn multiple workers on start", async () => {
    bridge = require("../src/services/pythonBridge");

    const startPromise = bridge.start();

    // Workers should be spawned
    assert.strictEqual(spawnedProcesses.length, 2, "should spawn 2 workers");

    // Simulate both workers becoming ready
    setImmediate(() => {
      spawnedProcesses[0].simulateReady(true);
      spawnedProcesses[1].simulateReady(true);
    });

    const result = await startPromise;
    assert.strictEqual(result, true, "start should return true");
  });

  it("should reuse workers for multiple requests", async () => {
    bridge = require("../src/services/pythonBridge");

    const startPromise = bridge.start();

    setImmediate(() => {
      spawnedProcesses[0].simulateReady(true);
      spawnedProcesses[1].simulateReady(true);
    });

    await startPromise;

    // Send first request
    const req1Promise = bridge.request("chat", { prompt: "test1" });
    await new Promise((r) => setImmediate(r));

    // Extract request ID and simulate response
    const write1 = spawnedProcesses[0].lastWrite || spawnedProcesses[1].lastWrite;
    const msg1 = JSON.parse(write1);
    const worker1 = spawnedProcesses[0].lastWrite ? spawnedProcesses[0] : spawnedProcesses[1];
    worker1.simulateResponse(msg1.id, true, { content: "response1" });

    const result1 = await req1Promise;
    assert.strictEqual(result1.content, "response1");

    // Send second request - should reuse a worker, not spawn a new one
    const req2Promise = bridge.request("chat", { prompt: "test2" });
    await new Promise((r) => setImmediate(r));

    assert.strictEqual(spawnedProcesses.length, 2, "should still have only 2 workers");

    const write2 = spawnedProcesses[0].lastWrite || spawnedProcesses[1].lastWrite;
    const msg2 = JSON.parse(write2);
    const worker2 = spawnedProcesses[0].lastWrite === write2 ? spawnedProcesses[0] : spawnedProcesses[1];
    worker2.simulateResponse(msg2.id, true, { content: "response2" });

    const result2 = await req2Promise;
    assert.strictEqual(result2.content, "response2");
  });

  it("should execute concurrent requests without head-of-line blocking", async () => {
    bridge = require("../src/services/pythonBridge");

    const startPromise = bridge.start();
    setImmediate(() => {
      spawnedProcesses[0].simulateReady(true);
      spawnedProcesses[1].simulateReady(true);
    });
    await startPromise;

    // Send two requests concurrently
    const req1Promise = bridge.request("chat", { prompt: "fast" });
    const req2Promise = bridge.request("chat", { prompt: "slow" });

    await new Promise((r) => setImmediate(r));

    // Both workers should have received requests
    assert.strictEqual(spawnedProcesses[0].lastWrite !== undefined, true);
    assert.strictEqual(spawnedProcesses[1].lastWrite !== undefined, true);

    const msg1 = JSON.parse(spawnedProcesses[0].lastWrite);
    const msg2 = JSON.parse(spawnedProcesses[1].lastWrite);

    // Simulate worker 2 (slow) responding later, worker 1 (fast) responding first
    spawnedProcesses[0].simulateResponse(msg1.id, true, { content: "fast-result" });

    const result1 = await req1Promise;
    assert.strictEqual(result1.content, "fast-result", "fast request should complete first");

    // Slow request completes later
    spawnedProcesses[1].simulateResponse(msg2.id, true, { content: "slow-result" });
    const result2 = await req2Promise;
    assert.strictEqual(result2.content, "slow-result", "slow request should eventually complete");
  });

  it("should queue requests when all workers are busy", async () => {
    bridge = require("../src/services/pythonBridge");

    const startPromise = bridge.start();
    setImmediate(() => {
      spawnedProcesses[0].simulateReady(true);
      spawnedProcesses[1].simulateReady(true);
    });
    await startPromise;

    // Send 3 requests (more than worker count)
    const req1Promise = bridge.request("chat", { prompt: "req1" });
    const req2Promise = bridge.request("chat", { prompt: "req2" });
    const req3Promise = bridge.request("chat", { prompt: "req3" });

    await new Promise((r) => setImmediate(r));

    // First two should be dispatched to workers
    assert.strictEqual(spawnedProcesses[0].lastWrite !== undefined, true);
    assert.strictEqual(spawnedProcesses[1].lastWrite !== undefined, true);

    const msg1 = JSON.parse(spawnedProcesses[0].lastWrite);
    const msg2 = JSON.parse(spawnedProcesses[1].lastWrite);

    // Complete first request - this should dispatch the queued third request
    spawnedProcesses[0].simulateResponse(msg1.id, true, { content: "result1" });
    const result1 = await req1Promise;
    assert.strictEqual(result1.content, "result1");

    // Third request should now be dispatched to worker 0
    await new Promise((r) => setImmediate(r));
    const msg3 = JSON.parse(spawnedProcesses[0].lastWrite);

    // Complete remaining requests
    spawnedProcesses[1].simulateResponse(msg2.id, true, { content: "result2" });
    spawnedProcesses[0].simulateResponse(msg3.id, true, { content: "result3" });

    const result2 = await req2Promise;
    const result3 = await req3Promise;

    assert.strictEqual(result2.content, "result2");
    assert.strictEqual(result3.content, "result3");
  });

  it("should route responses to correct requests even when out of order", async () => {
    bridge = require("../src/services/pythonBridge");

    const startPromise = bridge.start();
    setImmediate(() => {
      spawnedProcesses[0].simulateReady(true);
      spawnedProcesses[1].simulateReady(true);
    });
    await startPromise;

    const req1Promise = bridge.request("chat", { prompt: "req1" });
    const req2Promise = bridge.request("chat", { prompt: "req2" });

    await new Promise((r) => setImmediate(r));

    const msg1 = JSON.parse(spawnedProcesses[0].lastWrite);
    const msg2 = JSON.parse(spawnedProcesses[1].lastWrite);

    // Respond in reverse order
    spawnedProcesses[1].simulateResponse(msg2.id, true, { content: "result2" });
    spawnedProcesses[0].simulateResponse(msg1.id, true, { content: "result1" });

    const [result1, result2] = await Promise.all([req1Promise, req2Promise]);

    assert.strictEqual(result1.content, "result1", "request 1 should get its own result");
    assert.strictEqual(result2.content, "result2", "request 2 should get its own result");
  });

  it("should reject in-flight request when worker exits", async () => {
    bridge = require("../src/services/pythonBridge");

    const startPromise = bridge.start();
    setImmediate(() => {
      spawnedProcesses[0].simulateReady(true);
      spawnedProcesses[1].simulateReady(true);
    });
    await startPromise;

    const reqPromise = bridge.request("chat", { prompt: "test" });
    await new Promise((r) => setImmediate(r));

    // Worker exits before responding
    const workerWithRequest = spawnedProcesses[0].lastWrite ? spawnedProcesses[0] : spawnedProcesses[1];
    workerWithRequest.simulateExit(1);

    await assert.rejects(
      reqPromise,
      /worker exited/,
      "should reject request when worker exits"
    );
  });

  it("should spawn replacement worker after exit and use it for subsequent requests", async () => {
    bridge = require("../src/services/pythonBridge");

    // Zero-delay backoff, but leave the startup timeout pending so it does not
    // fire and tear down the worker mid-test.
    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        // Startup timeouts use 60000ms (or the configured value); backoff is 500-8000ms.
        if (ms >= 10000) return { startupTimeout: true };
        fn();
        return null;
      },
      () => {}
    );

    const startPromise = bridge.start();
    setImmediate(() => {
      spawnedProcesses[0].simulateReady(true);
      spawnedProcesses[1].simulateReady(true);
    });
    await startPromise;

    assert.strictEqual(spawnedProcesses.length, 2, "should start with 2 workers");

    // Kill first worker
    spawnedProcesses[0].simulateExit(1);

    // Give time for replacement to spawn (backoff is zero)
    await new Promise((r) => setTimeout(r, 10));

    assert.strictEqual(spawnedProcesses.length, 3, "should spawn replacement worker");

    // Make replacement ready
    spawnedProcesses[2].simulateReady(true);
    await new Promise((r) => setTimeout(r, 10));

    // Restore real timers before using the pool further
    bridge._setTimerImpl(prev.set, prev.clear);

    // Send new request - should use a ready worker (either worker 1 or the replacement)
    const reqPromise = bridge.request("chat", { prompt: "after-restart" });
    await new Promise((r) => setImmediate(r));

    // Should be able to send request to an available worker
    const workerWithWrite = spawnedProcesses.find((p) => p.lastWrite && !p.killed);
    assert.ok(workerWithWrite, "should find a worker that received the request");

    const msg = JSON.parse(workerWithWrite.lastWrite);
    workerWithWrite.simulateResponse(msg.id, true, { content: "restart-result" });

    const result = await reqPromise;
    assert.strictEqual(result.content, "restart-result");
  });

  it("should respect MODEL_TIMEOUT_MS", async () => {
    process.env.MODEL_TIMEOUT_MS = "100";

    bridge = require("../src/services/pythonBridge");

    const startPromise = bridge.start();
    setImmediate(() => {
      spawnedProcesses[0].simulateReady(true);
      spawnedProcesses[1].simulateReady(true);
    });
    await startPromise;

    const reqPromise = bridge.request("chat", { prompt: "timeout-test" });
    await new Promise((r) => setImmediate(r));

    // Don't respond - let it timeout
    await assert.rejects(
      reqPromise,
      /timed out/,
      "should timeout after MODEL_TIMEOUT_MS"
    );
  });

  it("should return false from start() when no workers can initialize", async () => {
    bridge = require("../src/services/pythonBridge");

    const startPromise = bridge.start();

    setImmediate(() => {
      // Both workers fail to initialize
      spawnedProcesses[0].simulateReady(false);
      spawnedProcesses[1].simulateReady(false);
    });

    const result = await startPromise;
    assert.strictEqual(result, false, "start should return false when all workers fail");
  });

  it("should dispatch queued requests after startup completes", async () => {
    bridge = require("../src/services/pythonBridge");

    // Start pool initialization but don't await - send requests during startup
    const startPromise = bridge.start();

    // Both requests arrive DURING startup before any worker is ready
    const req1Promise = bridge.request("chat", { prompt: "during-startup-1" });
    const req2Promise = bridge.request("chat", { prompt: "during-startup-2" });

    // Now make workers ready
    setImmediate(() => {
      spawnedProcesses[0].simulateReady(true);
      spawnedProcesses[1].simulateReady(true);
    });

    await startPromise;
    // Give dispatch a tick to run
    await new Promise((r) => setImmediate(r));

    // Both workers should have received requests - not just one
    assert.ok(spawnedProcesses[0].lastWrite !== undefined, "worker 0 should receive a request");
    assert.ok(spawnedProcesses[1].lastWrite !== undefined, "worker 1 should receive a request");

    const msg1 = JSON.parse(spawnedProcesses[0].lastWrite);
    const msg2 = JSON.parse(spawnedProcesses[1].lastWrite);

    spawnedProcesses[0].simulateResponse(msg1.id, true, { content: "result1" });
    spawnedProcesses[1].simulateResponse(msg2.id, true, { content: "result2" });

    const [result1, result2] = await Promise.all([req1Promise, req2Promise]);
    assert.strictEqual(result1.content, "result1");
    assert.strictEqual(result2.content, "result2");
  });

  it("should timeout queued requests that cannot be dispatched", async () => {
    process.env.MODEL_TIMEOUT_MS = "100";
    process.env.MODEL_WORKERS = "1";

    bridge = require("../src/services/pythonBridge");

    const startPromise = bridge.start();
    setImmediate(() => {
      spawnedProcesses[0].simulateReady(true);
    });
    await startPromise;

    // Occupy the single worker; capture the request ID before sending the queued one
    const blockerMsg = JSON.parse(await new Promise((resolve) => {
      spawnedProcesses[0].onStdinWrite = resolve;
      bridge.request("chat", { prompt: "blocker" }).catch(() => {});
    }));

    // This request must queue because the only worker is busy
    const queuedPromise = bridge.request("chat", { prompt: "queued" });

    // Queued request should timeout (blocker never responds)
    await assert.rejects(queuedPromise, /timed out/, "queued request should timeout");

    // Resolve the blocker so its inference timer is cleared before the test ends
    spawnedProcesses[0].simulateResponse(blockerMsg.id, true, { content: "done" });
    await new Promise((r) => setImmediate(r));
  });

  it("should reject queued requests with 'pool shutdown' when shutdown() is called", async () => {
    process.env.MODEL_WORKERS = "1";

    bridge = require("../src/services/pythonBridge");

    const startPromise = bridge.start();
    setImmediate(() => {
      spawnedProcesses[0].simulateReady(true);
    });
    await startPromise;

    // Occupy the single worker so the next request must queue
    const blockerMsg = JSON.parse(await new Promise((resolve) => {
      spawnedProcesses[0].onStdinWrite = resolve;
      bridge.request("chat", { prompt: "blocker" }).catch(() => {});
    }));

    // Queue a second request
    const queuedPromise = bridge.request("chat", { prompt: "queued" });

    // Yield a tick so the queued request lands in pool.queue before we shut down
    await new Promise((r) => setImmediate(r));

    // Shut down - queued request must reject, not hang
    bridge._pool().shutdown();

    await assert.rejects(queuedPromise, /pool shutdown/, "queued request should reject on shutdown");

    // Resolve the blocker so its timer is cleared before the test ends
    spawnedProcesses[0].simulateResponse(blockerMsg.id, true, { content: "done" });
    await new Promise((r) => setImmediate(r));
  });

  it("should clear safety timeout when worker initialization completes", async () => {
    bridge = require("../src/services/pythonBridge");

    const startPromise = bridge.start();
    setImmediate(() => {
      spawnedProcesses[0].simulateReady(true);
      spawnedProcesses[1].simulateReady(true);
    });
    await startPromise;

    // Workers are functional - safety timer must have been cleared or it would
    // keep the process alive for 60s and the test runner would flag it
    const reqPromise = bridge.request("chat", { prompt: "test-after-init" });
    await new Promise((r) => setImmediate(r));

    const proc = spawnedProcesses[0].lastWrite ? spawnedProcesses[0] : spawnedProcesses[1];
    const msg = JSON.parse(proc.lastWrite);
    proc.simulateResponse(msg.id, true, { content: "success" });

    const result = await reqPromise;
    assert.strictEqual(result.content, "success", "worker should be functional after initialization");
  });

  it("should wire --tools and --cli-mode/--cli-root args when MODEL_TOOLS includes cli", async () => {
    process.env.MODEL_TOOLS = "web,cli";
    process.env.MODEL_CLI_MODE = "allowlist";
    process.env.MODEL_CLI_ROOT = "/sandbox";

    bridge = require("../src/services/pythonBridge");

    const startPromise = bridge.start();
    setImmediate(() => {
      spawnedProcesses[0].simulateReady(true);
      spawnedProcesses[1].simulateReady(true);
    });
    await startPromise;

    const spawnedArgs = spawnedProcesses[0].args;
    assert.ok(spawnedArgs.includes("--tools"), "should pass --tools");
    assert.ok(spawnedArgs.includes("web,cli"), "should pass MODEL_TOOLS value");
    assert.ok(spawnedArgs.includes("--cli-mode"), "should pass --cli-mode");
    assert.ok(spawnedArgs.includes("allowlist"), "should pass MODEL_CLI_MODE value");
    assert.ok(spawnedArgs.includes("--cli-root"), "should pass --cli-root");
  });

  it("should apply backoff before respawning a crashed worker", async () => {
    process.env.MODEL_WORKERS = "1";
    bridge = require("../src/services/pythonBridge");

    // Capture backoff delay calls without actually waiting
    const delays = [];
    let fireFn = null;
    const prev = bridge._setTimerImpl(
      (fn, ms) => { delays.push(ms); fireFn = fn; return {}; },
      () => {}
    );

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    // Clear delays from initial startup (includes startup timeout)
    delays.length = 0;

    // Kill the worker - triggers backoff
    spawnedProcesses[0].simulateExit(1);
    await new Promise((r) => setImmediate(r));

    // Backoff timer should have been requested, not yet fired
    // Note: delays may include startup timeout for replacement worker + backoff timer
    const backoffDelays = delays.filter(d => d >= 500 && d <= 8000); // Backoff range
    assert.strictEqual(backoffDelays.length, 1, "should request exactly one backoff timer");
    assert.ok(backoffDelays[0] > 0, "backoff delay should be positive");
    assert.strictEqual(spawnedProcesses.length, 1, "replacement should NOT spawn before backoff fires");

    // Fire the backoff - replacement should now spawn
    fireFn();
    await new Promise((r) => setImmediate(r));

    assert.strictEqual(spawnedProcesses.length, 2, "replacement should spawn after backoff fires");

    bridge._setTimerImpl(prev.set, prev.clear);
    spawnedProcesses[1].simulateReady(true);
    await new Promise((r) => setImmediate(r));
  });

  it("should stop restarting after MAX_RESTART_ATTEMPTS and disable the pool", async () => {
    process.env.MODEL_WORKERS = "1";
    bridge = require("../src/services/pythonBridge");

    // Zero-delay backoff so attempts run synchronously
    const prev = bridge._setTimerImpl(
      (fn) => { fn(); return null; },
      () => {}
    );

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    // Kill and immediately fail each replacement, 5 times (the cap)
    for (let i = 0; i < 5; i++) {
      const current = spawnedProcesses[spawnedProcesses.length - 1];
      current.simulateExit(1);
      await new Promise((r) => setImmediate(r));
      // Fail the replacement spawn
      const replacement = spawnedProcesses[spawnedProcesses.length - 1];
      if (replacement !== current) {
        replacement.simulateReady(false);
        await new Promise((r) => setImmediate(r));
      }
    }

    // One more exit to exhaust the cap
    const last = spawnedProcesses[spawnedProcesses.length - 1];
    last.simulateExit(1);
    await new Promise((r) => setImmediate(r));

    bridge._setTimerImpl(prev.set, prev.clear);

    // Pool should be disabled - no more spawns, bridge.available() returns false
    const countBefore = spawnedProcesses.length;
    await new Promise((r) => setImmediate(r));
    assert.strictEqual(spawnedProcesses.length, countBefore, "no further spawns after cap");
    assert.strictEqual(bridge.available(), false, "pool should be disabled after cap");
  });

  it("should reset restart counter after a worker recovers successfully", async () => {
    process.env.MODEL_WORKERS = "1";
    bridge = require("../src/services/pythonBridge");

    // Zero-delay backoff, but leave the startup timeout pending so it does not
    // fire and tear down the worker mid-test.
    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        // Startup timeouts / stability timers use ms >= 10000; backoff is 500-8000ms.
        if (ms >= 10000) return { longTimer: true };
        fn();
        return null;
      },
      () => {}
    );

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    // First exit + successful recovery via request completion
    spawnedProcesses[0].simulateExit(1);
    await new Promise((r) => setImmediate(r));
    spawnedProcesses[1].simulateReady(true);
    await new Promise((r) => setImmediate(r));

    // Send a request to worker 1 to complete recovery and reset restart count
    const reqPromise = bridge.request("chat", { prompt: "recover" });
    await new Promise((r) => setImmediate(r));
    const msg = JSON.parse(spawnedProcesses[1].lastWrite);
    spawnedProcesses[1].simulateResponse(msg.id, true, { content: "recovered" });
    await reqPromise;

    // Second exit - counter should have reset, so this is attempt 1 again
    spawnedProcesses[1].simulateExit(1);
    await new Promise((r) => setImmediate(r));

    // A third worker should spawn (not hit the cap)
    assert.ok(spawnedProcesses.length >= 3, "should spawn again after counter reset");

    // Make the new worker ready if it exists
    if (spawnedProcesses.length >= 3) {
      spawnedProcesses[2].simulateReady(true);
    }

    bridge._setTimerImpl(prev.set, prev.clear);
    await new Promise((r) => setImmediate(r));
  });

  it("should respect absolute deadline for queued requests (Issue #155)", async () => {
    // This test verifies the fix for GitHub Issue #155:
    // Queued requests must NOT receive a fresh full timeout after dispatch.
    // The timeout must bound the TOTAL time from acceptance to response.

    process.env.MODEL_WORKERS = "1";
    process.env.MODEL_TIMEOUT_MS = "100"; // Short timeout
    bridge = require("../src/services/pythonBridge");

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    // Submit first request that will block the worker
    let firstRequestMsg = null;
    spawnedProcesses[0].onStdinWrite = (data) => {
      firstRequestMsg = JSON.parse(data);
    };

    const firstReqPromise = bridge.request("chat", { prompt: "blocker" });
    await new Promise((r) => setImmediate(r));

    // Clear the stdin write handler to capture the second request separately
    spawnedProcesses[0].onStdinWrite = null;

    // Submit second request - should be queued
    const secondReqStart = Date.now();
    const secondReqPromise = bridge.request("chat", { prompt: "queued" });
    await new Promise((r) => setImmediate(r));

    // Wait 80ms (most of the 100ms timeout)
    await new Promise((r) => setTimeout(r, 80));

    // Complete the first request to free up the worker
    spawnedProcesses[0].simulateResponse(firstRequestMsg.id, true, { content: "done" });
    await firstReqPromise;

    // Now the queued request should be dispatched with very little time remaining
    // It should timeout quickly (within the remaining ~20ms + processing time)

    const timeoutStart = Date.now();
    try {
      await secondReqPromise;
      assert.fail("Expected queued request to timeout due to absolute deadline");
    } catch (err) {
      const timeoutDuration = Date.now() - timeoutStart;
      assert.ok(err.message.includes("timed out"), `Expected timeout, got: ${err.message}`);

      // The timeout should happen quickly since we consumed most of the 100ms in queue
      // Allow some extra time for processing but it should be much less than 100ms
      assert.ok(timeoutDuration < 80,
        `Timeout took ${timeoutDuration}ms, expected < 80ms (indicating reduced remaining time)`);
    }
  });

  it("should reach MAX_RESTART_ATTEMPTS for workers that crash after becoming ready (Issue #154)", async () => {
    process.env.MODEL_WORKERS = "1";
    bridge = require("../src/services/pythonBridge");

    // Capture backoff delays without waiting, but keep startup/stability timers pending
    const delays = [];
    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        if (ms >= 10000) return { longTimer: true };
        delays.push(ms);
        fn();
        return null;
      },
      () => {}
    );

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    delays.length = 0;

    // Simulate crash after ready without completing any request 5 times
    for (let i = 0; i < 5; i++) {
      const current = spawnedProcesses[spawnedProcesses.length - 1];
      current.simulateExit(1);
      await new Promise((r) => setImmediate(r));
      const replacement = spawnedProcesses[spawnedProcesses.length - 1];
      assert.notStrictEqual(replacement, current, `replacement #${i + 1} should spawn`);
      replacement.simulateReady(true);
      await new Promise((r) => setImmediate(r));
    }

    // 6th exit: attempts becomes 6 > MAX_RESTART_ATTEMPTS (5)
    const current = spawnedProcesses[spawnedProcesses.length - 1];
    current.simulateExit(1);
    await new Promise((r) => setImmediate(r));

    bridge._setTimerImpl(prev.set, prev.clear);

    assert.deepStrictEqual(delays, [500, 1000, 2000, 4000, 8000]);
    assert.strictEqual(bridge.available(), false, "pool should be disabled after crash-after-ready cap is reached");
  });

  it("should reset restart counter after worker remains ready for stability duration", async () => {
    process.env.MODEL_WORKERS = "1";
    process.env.MODEL_WORKER_STABILITY_MS = "30000";
    bridge = require("../src/services/pythonBridge");

    let stabilityTimerFn = null;
    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        if (ms === 30000) {
          stabilityTimerFn = fn;
          return { stabilityTimer: true };
        }
        if (ms >= 10000) return { startupTimeout: true };
        fn();
        return null;
      },
      () => {}
    );

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    spawnedProcesses[0].simulateExit(1);
    await new Promise((r) => setImmediate(r));
    spawnedProcesses[1].simulateReady(true);
    await new Promise((r) => setImmediate(r));

    assert.ok(stabilityTimerFn, "should have registered a 30s stability timer");
    stabilityTimerFn();
    await new Promise((r) => setImmediate(r));

    spawnedProcesses[1].simulateExit(1);
    await new Promise((r) => setImmediate(r));

    assert.strictEqual(spawnedProcesses.length, 3, "should spawn replacement worker 2 after stability reset");

    bridge._setTimerImpl(prev.set, prev.clear);
  });

  it("should kill the worker child process when startup times out", async () => {
    // Regression test for #153: a worker that never reports ready must be torn
    // down, otherwise the Python process is orphaned and keeps holding memory.
    process.env.MODEL_STARTUP_TIMEOUT_MS = "25";
    bridge = require("../src/services/pythonBridge");

    const startPromise = bridge.start();
    assert.strictEqual(spawnedProcesses.length, 2, "both workers should spawn");

    // Never send a ready message, so the startup timeout is the only exit path.
    const result = await startPromise;

    assert.strictEqual(result, false, "start should return false after the timeout");
    assert.strictEqual(spawnedProcesses[0].killed, true, "worker 0 should be killed by timeout cleanup");
    assert.strictEqual(spawnedProcesses[1].killed, true, "worker 1 should be killed by timeout cleanup");
  });

  it("should use MODEL_STARTUP_TIMEOUT_MS for the startup timeout", async () => {
    process.env.MODEL_STARTUP_TIMEOUT_MS = "1234";
    bridge = require("../src/services/pythonBridge");

    const delays = [];
    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        delays.push(ms);
        return null;
      },
      () => {}
    );

    const startPromise = bridge.start();
    setImmediate(() => {
      spawnedProcesses[0].simulateReady(true);
      spawnedProcesses[1].simulateReady(true);
    });
    await startPromise;
    bridge._setTimerImpl(prev.set, prev.clear);

    assert.ok(
      delays.includes(1234),
      `startup timeout should use the configured value, saw ${JSON.stringify(delays)}`
    );
  });

  // A failed spawn is not thrown. child_process.spawn() returns a ChildProcess
  // and reports ENOENT, EACCES, EAGAIN and friends afterwards as an 'error'
  // event; 'exit' never follows. These tests deliver the failure that way,
  // because a spawn() that simply throws would only exercise the try/catch.
  const enoent = (command) =>
    Object.assign(new Error(`spawn ${command} ENOENT`), {
      code: "ENOENT",
      errno: -2,
      syscall: `spawn ${command}`,
      path: command,
    });

  // Mimics Node: the child is returned now and its 'error' is emitted on the next tick.
  const failSpawnAsync = () => {
    mockSpawn = (command, args, options) => {
      const child = new MockChildProcess(command, args, options);
      process.nextTick(() => child.emit("error", enoent(command)));
      return child;
    };
  };

  // Resolves to "STILL PENDING" instead of hanging the run when startup never settles.
  const settlesWithin = (promise, ms) => {
    let timer;
    const timeout = new Promise((resolve) => {
      timer = setTimeout(() => resolve("STILL PENDING"), ms);
    });
    return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
  };

  it("should resolve start() false and clean up when every worker fails to spawn", async () => {
    failSpawnAsync();
    bridge = require("../src/services/pythonBridge");

    const armed = [];
    const cleared = [];
    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        const timer = { ms };
        armed.push(timer);
        return timer;
      },
      (timer) => {
        cleared.push(timer);
      }
    );

    let result;
    try {
      result = await settlesWithin(bridge.start(), 500);
    } finally {
      bridge._setTimerImpl(prev.set, prev.clear);
    }

    assert.strictEqual(result, false, "start should return false when no worker can spawn");
    assert.strictEqual(bridge.available(), false, "the bridge should fall back instead of staying enabled");
    assert.strictEqual(spawnedProcesses.length, 2, "both workers should have tried to spawn");
    for (const worker of bridge._pool().workers) {
      assert.strictEqual(worker.child, null, `worker ${worker.id} should not keep the dead child`);
      assert.strictEqual(worker.ready, false, `worker ${worker.id} should not be ready`);
    }
    assert.ok(armed.length > 0, "startup should have armed its safety timers");
    assert.ok(
      armed.every((timer) => cleared.includes(timer)),
      "every startup timer must be cleared once the spawn has failed"
    );
  });

  it("should not run the exit path for a worker that already failed to spawn", async () => {
    // Node documents that 'exit' may or may not follow 'error'. If it does, it must
    // not reach handleWorkerExit(): that would clean the worker up a second time
    // and schedule a restart for a worker that was already reported as failed.
    process.env.MODEL_WORKERS = "1";
    failSpawnAsync();
    bridge = require("../src/services/pythonBridge");

    const delays = [];
    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        delays.push(ms);
        return {};
      },
      () => {}
    );

    try {
      const result = await settlesWithin(bridge.start(), 500);
      assert.strictEqual(result, false, "start should return false when the worker cannot spawn");

      delays.length = 0;
      spawnedProcesses[0].simulateExit(1);
      await new Promise((r) => setImmediate(r));

      assert.deepStrictEqual(delays, [], "a late 'exit' must not schedule a restart");
      assert.strictEqual(spawnedProcesses.length, 1, "no replacement worker should be spawned");
      assert.strictEqual(
        bridge._pool().workers.length,
        1,
        "the failed worker must not go through handleWorkerExit() a second time"
      );
    } finally {
      bridge._setTimerImpl(prev.set, prev.clear);
    }
  });

  it("should fail startup cleanly when PYTHON_BIN does not exist (real spawn)", async () => {
    // No mock child: the real child_process.spawn reports the missing interpreter
    // exactly as it does in production. Before the fix the 'error' event had no
    // listener, and the uncaught exception failed this test (and would kill the backend).
    mockSpawn = null;
    const previous = process.env.PYTHON_BIN;
    process.env.PYTHON_BIN = "/nonexistent/py";
    try {
      bridge = require("../src/services/pythonBridge");
    } finally {
      if (previous === undefined) delete process.env.PYTHON_BIN;
      else process.env.PYTHON_BIN = previous;
    }

    const result = await settlesWithin(bridge.start(), 5000);

    assert.strictEqual(result, false, "start should return false when the interpreter cannot be spawned");
    assert.strictEqual(spawnedProcesses.length, 0, "the real spawn should have been used, not the mock");
    assert.strictEqual(bridge.available(), false, "the bridge should fall back instead of staying enabled");
    for (const worker of bridge._pool().workers) {
      assert.strictEqual(worker.child, null, `worker ${worker.id} should not keep the dead child`);
      assert.strictEqual(worker.ready, false, `worker ${worker.id} should not be ready`);
    }
  });

  it("should not crash the process when PYTHON_BIN cannot be spawned (real process)", () => {
    // The real bridge and the real child_process, in a separate Node process so a
    // crash shows up as an exit status. Before the fix the failed spawn was an
    // uncaught exception and this process exited with status 1.
    const { spawnSync } = require("child_process");
    const script = `
      process.env.MODEL_ENABLED = "true";
      process.env.MODEL_PATH = ${JSON.stringify(__filename)};
      process.env.PYTHON_BIN = "/nonexistent/py";
      const bridge = require(${JSON.stringify(require.resolve("../src/services/pythonBridge"))});
      bridge.request("chat", { prompt: "hi" }).then(
        () => console.log("settled: resolved"),
        (err) => console.log("settled: rejected " + err.message)
      );
    `;
    const res = spawnSync(process.execPath, ["-e", script], {
      encoding: "utf8",
      timeout: 15000,
      env: { ...process.env, LOG_LEVEL: "error" },
    });

    assert.notStrictEqual(
      res.error && res.error.code,
      "ETIMEDOUT",
      "the request never settled, or a startup timer kept the process alive"
    );
    const reason = String(res.stderr).split("\n").find((line) => /Error/.test(line));
    assert.strictEqual(res.status, 0, `the process crashed: ${reason}`);
    assert.match(res.stdout, /settled: rejected no workers available/);
  });

  it("should start with the workers that did spawn when another one fails to spawn", async () => {
    mockSpawn = (command, args, options) => {
      const child = new MockChildProcess(command, args, options);
      if (spawnedProcesses.length === 1) {
        // The first worker cannot spawn; the second one is healthy.
        process.nextTick(() => child.emit("error", enoent(command)));
      }
      return child;
    };
    bridge = require("../src/services/pythonBridge");

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[1].simulateReady(true));

    assert.strictEqual(await settlesWithin(startPromise, 1000), true, "start should succeed on the surviving worker");
    assert.strictEqual(bridge.available(), true);

    const reqPromise = bridge.request("chat", { prompt: "served by the healthy worker" });
    await new Promise((r) => setImmediate(r));
    const msg = JSON.parse(spawnedProcesses[1].lastWrite);
    spawnedProcesses[1].simulateResponse(msg.id, true, { content: "healthy" });

    assert.strictEqual((await reqPromise).content, "healthy");
  });

  it("should leave a ready worker running when an 'error' event arrives after startup", async () => {
    // Node also emits 'error' when a kill or a send fails. Once startup has settled
    // the process is alive and its exit handler owns it, so the event is logged and
    // must not be treated as a failed spawn.
    bridge = require("../src/services/pythonBridge");

    const startPromise = bridge.start();
    setImmediate(() => {
      spawnedProcesses[0].simulateReady(true);
      spawnedProcesses[1].simulateReady(true);
    });
    assert.strictEqual(await startPromise, true);

    spawnedProcesses[0].emit("error", Object.assign(new Error("kill EPERM"), { code: "EPERM", syscall: "kill" }));

    const worker = bridge._pool().workers[0];
    assert.strictEqual(worker.ready, true, "the healthy worker must stay ready");
    assert.strictEqual(worker.child, spawnedProcesses[0], "the healthy worker must keep its child");
    assert.strictEqual(bridge.available(), true);

    const reqPromise = bridge.request("chat", { prompt: "after the stray error" });
    await new Promise((r) => setImmediate(r));
    const msg = JSON.parse(spawnedProcesses[0].lastWrite);
    spawnedProcesses[0].simulateResponse(msg.id, true, { content: "still serving" });

    assert.strictEqual((await reqPromise).content, "still serving");
  });

  it("should re-enable bridge.available() when worker pool recovers after transient failure (Issue #152)", async () => {
    process.env.MODEL_WORKERS = "1";
    bridge = require("../src/services/pythonBridge");

    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        if (ms >= 10000) return { longTimer: true };
        fn();
        return null;
      },
      () => {}
    );

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    assert.strictEqual(bridge.available(), true, "bridge should be available when worker is ready");

    // Worker exits
    spawnedProcesses[0].simulateExit(1);
    await new Promise((r) => setImmediate(r));

    // First replacement startup fails -> pool becomes disabled
    const replacement1 = spawnedProcesses[spawnedProcesses.length - 1];
    assert.notStrictEqual(replacement1, spawnedProcesses[0], "replacement 1 should spawn");
    replacement1.simulateReady(false);
    await new Promise((r) => setImmediate(r));

    assert.strictEqual(bridge.available(), false, "bridge should be disabled after replacement spawn failure");

    // Exit replacement1 to trigger next restart attempt
    replacement1.simulateExit(1);
    await new Promise((r) => setImmediate(r));

    // Replacement 2 spawns on retry
    const replacement2 = spawnedProcesses[spawnedProcesses.length - 1];
    assert.notStrictEqual(replacement2, replacement1, "replacement 2 should spawn after retry");

    // Replacement 2 becomes ready
    replacement2.simulateReady(true);
    await new Promise((r) => setImmediate(r));

    // Confirm bridge.available() is true again
    assert.strictEqual(bridge.available(), true, "bridge should become available again after worker pool recovers");

    // Verify a request can actually be dispatched/completed by the recovered worker
    const reqPromise = bridge.request("chat", { prompt: "after-recovery" });
    await new Promise((r) => setImmediate(r));

    const msg = JSON.parse(replacement2.lastWrite);
    replacement2.simulateResponse(msg.id, true, { content: "recovered-success" });

    const result = await reqPromise;
    assert.strictEqual(result.content, "recovered-success", "recovered worker should handle request successfully");

    // Verify that exhausting MAX_RESTART_ATTEMPTS leaves the bridge unavailable
    for (let i = 0; i < 4; i++) {
      const current = spawnedProcesses[spawnedProcesses.length - 1];
      current.simulateExit(1);
      await new Promise((r) => setImmediate(r));
      const rep = spawnedProcesses[spawnedProcesses.length - 1];
      if (rep !== current) {
        rep.simulateReady(false);
        await new Promise((r) => setImmediate(r));
      }
    }

    const last = spawnedProcesses[spawnedProcesses.length - 1];
    last.simulateExit(1);
    await new Promise((r) => setImmediate(r));

    bridge._setTimerImpl(prev.set, prev.clear);

    assert.strictEqual(bridge.available(), false, "bridge should remain disabled after MAX_RESTART_ATTEMPTS cap");
  });

  it("should handle approval_request and route response to worker stdin", async () => {
    bridge = require("../src/services/pythonBridge");
    const startPromise = bridge.start();

    setImmediate(() => {
      spawnedProcesses[0].simulateReady(true);
      spawnedProcesses[1].simulateReady(true);
    });

    await startPromise;

    let receivedApproval = null;
    const reqPromise = bridge.request(
      "chat",
      { prompt: "run ls" },
      {
        onApprovalRequest: (info) => {
          receivedApproval = info;
          info.respond(true);
        },
      }
    );

    await new Promise((r) => setImmediate(r));
    const worker = spawnedProcesses[0];

    // Worker emits approval request over stdout
    const approvalReq = {
      type: "approval_request",
      approval_id: "app-uuid-1",
      command: "ls -la",
      argv: ["ls", "-la"],
      root: "/sandbox",
    };
    worker.stdout.emit("data", Buffer.from(JSON.stringify(approvalReq) + "\n"));

    assert.ok(receivedApproval, "onApprovalRequest should be called");
    assert.strictEqual(receivedApproval.approvalId, "app-uuid-1");
    assert.strictEqual(receivedApproval.command, "ls -la");
    assert.deepStrictEqual(receivedApproval.argv, ["ls", "-la"]);

    // Verify response was written to worker stdin
    const responseObj = JSON.parse(worker.lastWrite);
    assert.strictEqual(responseObj.type, "approval_response");
    assert.strictEqual(responseObj.approval_id, "app-uuid-1");
    assert.strictEqual(responseObj.approved, true);

    // Complete request
    worker.simulateResponse(1, true, { content: "ls result" });
    const res = await reqPromise;
    assert.strictEqual(res.content, "ls result");
  });

  it("should fail closed when onApprovalRequest callback is absent or fails", async () => {
    bridge = require("../src/services/pythonBridge");
    const startPromise = bridge.start();

    setImmediate(() => {
      spawnedProcesses[0].simulateReady(true);
      spawnedProcesses[1].simulateReady(true);
    });

    await startPromise;

    const reqPromise = bridge.request("chat", { prompt: "run command" });
    await new Promise((r) => setImmediate(r));

    const worker = spawnedProcesses[0];
    const approvalReq = {
      type: "approval_request",
      approval_id: "app-uuid-2",
      command: "rm -rf /",
      argv: ["rm", "-rf", "/"],
      root: "/sandbox",
    };
    worker.stdout.emit("data", Buffer.from(JSON.stringify(approvalReq) + "\n"));

    const responseObj = JSON.parse(worker.lastWrite);
    assert.strictEqual(responseObj.type, "approval_response");
    assert.strictEqual(responseObj.approval_id, "app-uuid-2");
    assert.strictEqual(responseObj.approved, false);

    worker.simulateResponse(1, false, "refused");
    await reqPromise.catch(() => {});
  });

  // Regression tests for timeout race condition - these must run sequentially
  // because they use long waits for worker restart lifecycle
  describe("timeout race condition regression tests", { concurrency: 1 }, () => {
    it("REGRESSION TEST: timeout should NOT allow worker reuse before Python request actually completes", async () => {
    // This test verifies the fix for the race condition:
    // When execute() times out, the worker must be terminated/restarted
    // rather than marked available, because the underlying Python process
    // may still be executing the timed-out request.

    process.env.MODEL_WORKERS = "1"; // Single worker to force reuse scenario
    process.env.MODEL_TIMEOUT_MS = "50"; // Very short timeout
    bridge = require("../src/services/pythonBridge");

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    const worker = spawnedProcesses[0];

    // Track all messages sent to worker stdin
    const sentMessages = [];
    worker.onStdinWrite = (data) => {
      sentMessages.push(JSON.parse(data));
    };

    // Request A with short timeout - keep Python response pending
    const reqAPromise = bridge.request("chat", { prompt: "request-A" });
    await new Promise((r) => setImmediate(r));

    assert.strictEqual(sentMessages.length, 1, "request A should be sent");
    const msgA = sentMessages[0];
    assert.strictEqual(msgA.params.prompt, "request-A");

    // Attach rejection handler immediately to avoid unhandled rejection warnings
    const timeoutCheck = reqAPromise.catch((err) => {
      if (!/timed out/.test(err.message)) {
        throw err; // Re-throw if it's not the expected timeout error
      }
    });

    // Wait for request A to timeout (50ms timeout + margin)
    await new Promise((r) => setTimeout(r, 100));

    // Verify request A timed out
    await timeoutCheck;

    // FIX VERIFICATION: Give time for worker exit handler and replacement to spawn
    // The timeout calls child.kill() → exit event → onExit → handleWorkerExit (async!) → spawn replacement
    // handleWorkerExit is async and schedules the replacement with setTimeout(..., 500ms backoff)
    await new Promise((r) => setTimeout(r, 700)); // 500ms backoff + 200ms margin

    // A replacement worker should have spawned after timeout kill
    assert.ok(spawnedProcesses.length >= 2, `replacement worker should spawn after timeout kill (spawned: ${spawnedProcesses.length})`);
    const replacement = spawnedProcesses[spawnedProcesses.length - 1];
    assert.notStrictEqual(replacement, worker, "replacement should be a different worker");

    // Make replacement ready
    replacement.simulateReady(true);
    await new Promise((r) => setImmediate(r));

    // Now send request B - it should go to the replacement worker, not the killed one
    const reqBPromise = bridge.request("chat", { prompt: "request-B" });
    await new Promise((r) => setImmediate(r));

    // Request B should be sent to the replacement worker
    assert.ok(replacement.lastWrite, "replacement worker should receive request B");
    const msgB = JSON.parse(replacement.lastWrite);
    assert.strictEqual(msgB.params.prompt, "request-B");

    // Complete request B normally
    replacement.simulateResponse(msgB.id, true, { content: "response-B" });
    const resultB = await reqBPromise;
    assert.strictEqual(resultB.content, "response-B", "request B should complete with correct response");
  });

  // -------------------------------------------------------------------------
  // Regression tests for Issue #238:
  // "A timed-out worker is never force-killed and the pool can drain to zero"
  //
  // Root cause: child.kill() sends SIGTERM. A Python worker blocked inside a
  // native GPU kernel may ignore SIGTERM, so the exit event never fires,
  // handleWorkerExit() is never called, no replacement is spawned, and the
  // pool permanently loses that slot.
  //
  // Fix: escalate to SIGKILL after a grace period if the process has not
  // exited. _setTimeout/_clearTimeout are used for the escalation timer so
  // tests can capture and control it.
  // -------------------------------------------------------------------------

  describe("Issue #238 regression: timed-out worker force-kill and pool replenishment", { concurrency: 1 }, () => {
    // A MockChildProcess subclass that ignores SIGTERM (does not emit 'exit')
    // but dies immediately when kill('SIGKILL') is called. This accurately
    // models a Python process stuck inside a native extension.
    class SigTermImmuneMockChildProcess extends MockChildProcess {
      constructor(...args) {
        super(...args);
        this.killSignals = [];
        this.exited = false;
      }

      kill(signal) {
        this.killSignals.push(signal || "SIGTERM");
        // Real ChildProcess sets `killed` as soon as a signal is DELIVERED, not
        // when the process exits. Mirror that here, otherwise the mock would let
        // a `!child.killed` liveness check pass in tests while never firing in
        // production. Aliveness is tracked separately via `exited`.
        this.killed = true;
        if (signal === "SIGKILL") {
          // Respond to SIGKILL by actually dying
          this.exited = true;
          this.signalCode = "SIGKILL";
          this.emit("exit", 137); // 128 + 9 (SIGKILL)
        }
        // SIGTERM is silently ignored — the process does not exit, so
        // exitCode/signalCode both stay null, same as a genuinely live process.
      }
    }

    it("REGRESSION #238: timed-out worker that ignores SIGTERM is force-killed via SIGKILL escalation", async () => {
      // Demonstrates the bug: with only SIGTERM the exit event never fires,
      // onExit is never called, and no replacement is spawned.
      // The fix: _setTimeout fires an escalation that calls kill('SIGKILL').
      //
      // UPDATE for Issue #276: The timeout path now calls handleWorkerExit directly
      // so the replacement spawns immediately without waiting for the exit event.
      // The SIGKILL escalation still fires to clean up the stuck process.

      process.env.MODEL_WORKERS = "1";
      process.env.MODEL_TIMEOUT_MS = "50";
      bridge = require("../src/services/pythonBridge");

      // Replace the spawn mock to produce SIGTERM-immune workers
      mockSpawn = () => new SigTermImmuneMockChildProcess();

      // Capture escalation timer so we can fire it deterministically
      let escalationFn = null;
      const prev = bridge._setTimerImpl(
        (fn, ms) => {
          if (ms >= 60000) return { startupTimeout: true }; // startup timeout - don't fire
          if (ms >= 5000) {
            // This is the SIGKILL escalation timer (5000ms) — capture it
            escalationFn = fn;
            return { escalationTimer: true };
          }
          if (ms >= 500 && ms <= 8000) {
            // Backoff timer — fire immediately for test speed
            fn();
            return null;
          }
          // Short timers (request timeout 50ms) — run natively
          return setTimeout(fn, ms);
        },
        (id) => {
          if (id && (id.escalationTimer || id.startupTimeout)) return;
          clearTimeout(id);
        }
      );

      const startPromise = bridge.start();
      setImmediate(() => spawnedProcesses[0].simulateReady(true));
      await startPromise;

      const worker = spawnedProcesses[0];
      assert.ok(worker instanceof SigTermImmuneMockChildProcess, "should use SIGTERM-immune worker");

      // Send a request that will time out - attach error handler immediately
      const reqPromise = bridge.request("chat", { prompt: "slow-gpu-request" }).catch((err) => {
        // Expected timeout error
        if (!/timed out/.test(err.message)) throw err;
      });
      await new Promise((r) => setImmediate(r));

      // Wait for the 50ms request timeout to fire
      await new Promise((r) => setTimeout(r, 100));

      // Drain the promise
      await reqPromise;

      // Worker received SIGTERM but is still alive (ignores it)
      assert.ok(worker.killSignals.includes("SIGTERM"), "SIGTERM should have been sent");
      assert.strictEqual(worker.exited, false, "worker should NOT have exited yet (ignores SIGTERM)");
      assert.ok(escalationFn, "SIGKILL escalation timer should have been registered");

      // Issue #276 fix: handleWorkerExit is called directly from timeout path,
      // so replacement spawns immediately (before SIGKILL escalation fires).
      await new Promise((r) => setImmediate(r));
      assert.ok(spawnedProcesses.length >= 2, `replacement should spawn immediately via Issue #276 fix (found ${spawnedProcesses.length})`);

      // Fire the escalation — this should send SIGKILL, which triggers exit
      escalationFn();
      await new Promise((r) => setImmediate(r));

      // Worker should now be dead
      assert.ok(worker.killSignals.includes("SIGKILL"), "SIGKILL should have been sent after escalation");
      assert.strictEqual(worker.exited, true, "worker should be dead after SIGKILL");

      bridge._setTimerImpl(prev.set, prev.clear);
    });

    it("REGRESSION #238: pool worker count does not drain to zero when worker ignores SIGTERM", async () => {
      // Verifies the pool replenishment path works end-to-end after SIGKILL escalation.
      //
      // UPDATE for Issue #276: The timeout path now calls handleWorkerExit directly
      // so the replacement spawns immediately without waiting for the exit event.

      process.env.MODEL_WORKERS = "1";
      process.env.MODEL_TIMEOUT_MS = "50";
      bridge = require("../src/services/pythonBridge");

      mockSpawn = () => new SigTermImmuneMockChildProcess();

      // Use zero-delay backoff but capture the escalation timer
      let escalationFn = null;
      const prev = bridge._setTimerImpl(
        (fn, ms) => {
          if (ms >= 60000) return { startupTimeout: true }; // startup timeout — don't fire
          if (ms >= 5000) {
            // SIGKILL escalation — capture it
            escalationFn = fn;
            return { escalationTimer: true };
          }
          if (ms >= 500 && ms <= 8000) {
            // Backoff — fire immediately
            fn();
            return null;
          }
          // Short timers (request timeout 50ms) — run natively
          return setTimeout(fn, ms);
        },
        (id) => {
          if (id && (id.escalationTimer || id.startupTimeout)) return;
          clearTimeout(id);
        }
      );

      const startPromise = bridge.start();
      setImmediate(() => spawnedProcesses[0].simulateReady(true));
      await startPromise;

      assert.strictEqual(spawnedProcesses.length, 1, "start: 1 worker");
      assert.strictEqual(bridge.available(), true, "pool should be available");

      // Fire a request that will time out - attach error handler immediately
      const reqPromise = bridge.request("chat", { prompt: "will-timeout" }).catch((err) => {
        if (!/timed out/.test(err.message)) throw err;
      });
      await new Promise((r) => setImmediate(r));

      // Let the 50ms timeout fire
      await new Promise((r) => setTimeout(r, 100));
      await reqPromise;

      // Escalation timer should be registered
      assert.ok(escalationFn, "escalation timer should be registered after SIGTERM");

      // Issue #276 fix: handleWorkerExit is called directly, so replacement spawns immediately
      await new Promise((r) => setImmediate(r));
      assert.ok(spawnedProcesses.length >= 2, `replacement should spawn immediately via Issue #276 fix (got ${spawnedProcesses.length})`);

      // Fire escalation → worker dies (cleanup stuck process)
      escalationFn();
      await new Promise((r) => setImmediate(r));

      // Make the replacement ready
      const replacement = spawnedProcesses[spawnedProcesses.length - 1];
      replacement.simulateReady(true);
      await new Promise((r) => setImmediate(r));

      bridge._setTimerImpl(prev.set, prev.clear);

      // Pool must be usable again
      assert.strictEqual(bridge.available(), true, "pool should be available after replacement");

      // A new request should complete successfully on the replacement
      const req2Promise = bridge.request("chat", { prompt: "after-recovery" });
      await new Promise((r) => setImmediate(r));
      assert.ok(replacement.lastWrite, "replacement should receive new request");
      const msg = JSON.parse(replacement.lastWrite);
      replacement.simulateResponse(msg.id, true, { content: "ok" });
      const result = await req2Promise;
      assert.strictEqual(result.content, "ok", "pool should serve requests after SIGKILL recovery");
    });

    it("REGRESSION #238: escalation timer is cleared when worker exits on SIGTERM (normal case)", async () => {
      // Confirms the fix does NOT break the normal SIGTERM path:
      // when the process cooperatively exits, the SIGKILL escalation timer
      // is cancelled and SIGKILL is never sent.

      process.env.MODEL_WORKERS = "1";
      process.env.MODEL_TIMEOUT_MS = "50";
      bridge = require("../src/services/pythonBridge");

      // Normal mock: kill() emits exit immediately (cooperative SIGTERM response)
      // This is the default MockChildProcess behavior.

      let escalationTimerCleared = false;
      let escalationTimerRegistered = false;
      const prev = bridge._setTimerImpl(
        (fn, ms) => {
          if (ms >= 60000) return { startupTimeout: true };
          if (ms >= 5000) {
            escalationTimerRegistered = true;
            // Return a real timer but track clearing
            const id = setTimeout(fn, ms);
            id._isEscalation = true;
            return id;
          }
          if (ms >= 500 && ms <= 8000) { fn(); return null; }
          return setTimeout(fn, ms);
        },
        (id) => {
          if (id && id.startupTimeout) return;
          if (id && id._isEscalation) { escalationTimerCleared = true; }
          clearTimeout(id);
        }
      );

      const startPromise = bridge.start();
      setImmediate(() => spawnedProcesses[0].simulateReady(true));
      await startPromise;

      // Fire a request that will time out — default mock cooperative kill - attach error handler immediately
      const reqPromise = bridge.request("chat", { prompt: "cooperative-timeout" }).catch((err) => {
        if (!/timed out/.test(err.message)) throw err;
      });
      await new Promise((r) => setImmediate(r));

      await new Promise((r) => setTimeout(r, 100)); // let 50ms timeout fire
      await reqPromise;

      bridge._setTimerImpl(prev.set, prev.clear);

      // The escalation timer should have been registered AND then cleared
      // because the mock process exits immediately on kill() (SIGTERM response)
      assert.ok(escalationTimerRegistered, "escalation timer should be registered");
      assert.ok(escalationTimerCleared, "escalation timer should be cleared when process exits on SIGTERM");
    });

    it("REGRESSION #238: timed-out worker cannot be reused for a later request", async () => {
      // Verifies that after a timeout the killed worker is never dispatched
      // a new request. Only the replacement worker handles subsequent requests.

      process.env.MODEL_WORKERS = "1";
      process.env.MODEL_TIMEOUT_MS = "50";
      bridge = require("../src/services/pythonBridge");

      mockSpawn = () => new SigTermImmuneMockChildProcess();

      let escalationFn = null;
      const prev = bridge._setTimerImpl(
        (fn, ms) => {
          if (ms >= 60000) return { startupTimeout: true };
          if (ms >= 5000) { escalationFn = fn; return { escalationTimer: true }; }
          if (ms >= 500 && ms <= 8000) { fn(); return null; }
          return setTimeout(fn, ms);
        },
        (id) => {
          if (id && (id.escalationTimer || id.startupTimeout)) return;
          clearTimeout(id);
        }
      );

      const startPromise = bridge.start();
      setImmediate(() => spawnedProcesses[0].simulateReady(true));
      await startPromise;

      const originalWorker = spawnedProcesses[0];

      // Time out a request - attach error handler immediately
      const reqPromise = bridge.request("chat", { prompt: "timeout-me" }).catch((err) => {
        if (!/timed out/.test(err.message)) throw err;
      });
      await new Promise((r) => setImmediate(r));
      await new Promise((r) => setTimeout(r, 100));
      await reqPromise;

      // Fire SIGKILL escalation → exit → replacement spawns
      assert.ok(escalationFn, "escalation timer must be registered");
      escalationFn();
      await new Promise((r) => setImmediate(r));
      await new Promise((r) => setImmediate(r));

      assert.ok(spawnedProcesses.length >= 2, "replacement must spawn");
      const replacement = spawnedProcesses[spawnedProcesses.length - 1];
      replacement.simulateReady(true);
      await new Promise((r) => setImmediate(r));

      bridge._setTimerImpl(prev.set, prev.clear);

      // Send a new request — it MUST go to the replacement, not the dead original
      const req2Promise = bridge.request("chat", { prompt: "new-request" });
      await new Promise((r) => setImmediate(r));

      assert.ok(replacement.lastWrite, "new request should go to replacement worker");
      assert.ok(
        !originalWorker.lastWrite || JSON.parse(originalWorker.lastWrite).params.prompt !== "new-request",
        "dead original worker must NOT receive new requests"
      );

      const msg = JSON.parse(replacement.lastWrite);
      replacement.simulateResponse(msg.id, true, { content: "new-ok" });
      const result = await req2Promise;
      assert.strictEqual(result.content, "new-ok");
    });

    it("REGRESSION #238: normal worker reuse still works after successful requests", async () => {
      // Confirms the fix does not break the happy path: successful requests
      // continue to be served by the same worker (no unnecessary replacement).

      process.env.MODEL_WORKERS = "1";
      bridge = require("../src/services/pythonBridge");

      const startPromise = bridge.start();
      setImmediate(() => spawnedProcesses[0].simulateReady(true));
      await startPromise;

      const worker = spawnedProcesses[0];

      // First successful request
      const req1 = bridge.request("chat", { prompt: "req1" });
      await new Promise((r) => setImmediate(r));
      const msg1 = JSON.parse(worker.lastWrite);
      worker.simulateResponse(msg1.id, true, { content: "r1" });
      const res1 = await req1;
      assert.strictEqual(res1.content, "r1");

      // Second successful request — same worker, no replacement
      const req2 = bridge.request("chat", { prompt: "req2" });
      await new Promise((r) => setImmediate(r));
      const msg2 = JSON.parse(worker.lastWrite);
      worker.simulateResponse(msg2.id, true, { content: "r2" });
      const res2 = await req2;
      assert.strictEqual(res2.content, "r2");

      assert.strictEqual(spawnedProcesses.length, 1, "no new workers spawned for successful requests");
    });
  });

  it("REGRESSION TEST: approval_request after timeout should not route to wrong currentRequest", async () => {
    // This test verifies the fix for approval routing confusion:
    // When request A times out, the worker is killed. Late approval_request
    // messages from A should not be routed to a subsequent request B on a
    // replacement worker.

    process.env.MODEL_WORKERS = "1";
    process.env.MODEL_TIMEOUT_MS = "50";
    bridge = require("../src/services/pythonBridge");

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    const worker = spawnedProcesses[0];

    let approvalAReceived = null;
    let approvalBReceived = null;

    // Request A with approval callback
    const reqAPromise = bridge.request(
      "chat",
      { prompt: "request-A needs approval" },
      {
        onApprovalRequest: (info) => {
          approvalAReceived = info;
          info.respond(true);
        },
      }
    );
    await new Promise((r) => setImmediate(r));

    const msgA = JSON.parse(worker.lastWrite);

    // Attach rejection handler immediately to avoid unhandled rejection warnings
    const timeoutCheck = reqAPromise.catch((err) => {
      if (!/timed out/.test(err.message)) {
        throw err; // Re-throw if it's not the expected timeout error
      }
    });

    // Wait for request A to timeout
    await new Promise((r) => setTimeout(r, 100));
    await timeoutCheck;

    // FIX VERIFICATION: Give time for worker exit and replacement to spawn
    // The timeout calls child.kill() → exit event → onExit → handleWorkerExit (async!) → spawn replacement
    // handleWorkerExit is async and schedules the replacement with setTimeout(..., 500ms backoff)
    // Wait for backoff delay + margin. Use a single setTimeout to block other tests from starting.
    await new Promise((r) => setTimeout(r, 700)); // 500ms backoff + 200ms margin

    // A replacement worker should have spawned (the original worker was killed on timeout)
    assert.ok(spawnedProcesses.length >= 2, `replacement worker should spawn after timeout kill (spawned: ${spawnedProcesses.length})`);
    const replacement = spawnedProcesses[spawnedProcesses.length - 1];
    assert.notStrictEqual(replacement, worker, "replacement should be a different worker");

    // Make replacement ready
    replacement.simulateReady(true);
    await new Promise((r) => setImmediate(r));

    // Request B with different approval callback on replacement worker
    const reqBPromise = bridge.request(
      "chat",
      { prompt: "request-B different approval" },
      {
        onApprovalRequest: (info) => {
          approvalBReceived = info;
          info.respond(false);
        },
      }
    );
    await new Promise((r) => setImmediate(r));

    const msgB = JSON.parse(replacement.lastWrite);

    // FIX VERIFICATION: Even if we simulate a late approval from the dead worker,
    // it won't reach request B because the old worker is dead and replacement
    // is a fresh instance with separate state
    const approvalFromDeadWorker = {
      type: "approval_request",
      approval_id: "approval-from-A",
      command: "command-from-A",
      argv: ["cmd-A"],
      root: "/sandbox",
    };

    // Simulate approval from dead worker (should be ignored)
    worker.stdout.emit("data", Buffer.from(JSON.stringify(approvalFromDeadWorker) + "\n"));
    await new Promise((r) => setImmediate(r));

    // B's callback should NOT have received A's approval
    assert.strictEqual(approvalBReceived, null, "Request B should not receive approval from dead worker's request A");

    // Complete request B normally
    replacement.simulateResponse(msgB.id, true, { content: "response-B" });
    await reqBPromise;
  });

  // -------------------------------------------------------------------------
  // Regression tests for Issue #278:
  // "WorkerPool availability must reflect dispatchable workers"
  //
  // Root cause of the first attempt at this fix: available() was changed to
  // check the actual worker pool (ready && !busy), but available() is the
  // "real model vs placeholder" gate that model.js checks at all 8 of its
  // call sites before calling bridge.request(). bridge.request() already
  // queues a request when every worker is busy (see "should queue requests
  // when all workers are busy" above), so making available() track busy
  // state made model.js skip the bridge - and its queueing - entirely once
  // every worker was busy, silently falling back to placeholder output for
  // requests that should have queued and completed for real.
  //
  // The fix: available() stays exactly isConfigured() && !disabled (a config
  // gate, unaffected by worker load). Dispatchability is exposed separately
  // via hasAvailableWorker(), mirroring WorkerPool.getAvailableWorker()'s
  // "ready && !busy" semantics for callers that specifically need to know
  // whether a request would dispatch immediately or queue.
  // -------------------------------------------------------------------------

  describe("Issue #278 regression: hasAvailableWorker() reflects dispatchable workers", () => {
    it("REGRESSION #278: hasAvailableWorker() is false when the only worker is busy, while available() stays true", async () => {
      process.env.MODEL_WORKERS = "1";
      bridge = require("../src/services/pythonBridge");

      const startPromise = bridge.start();
      setImmediate(() => spawnedProcesses[0].simulateReady(true));
      await startPromise;

      assert.strictEqual(bridge.available(), true, "available before any request is in flight");
      assert.strictEqual(bridge.hasAvailableWorker(), true, "a dispatchable worker exists before any request");

      // Occupy the only worker with an in-flight request
      const reqPromise = bridge.request("chat", { prompt: "occupy" });
      await new Promise((r) => setImmediate(r));

      assert.strictEqual(
        bridge.hasAvailableWorker(),
        false,
        "hasAvailableWorker() must be false while the only worker in the pool is busy"
      );
      assert.strictEqual(
        bridge.available(),
        true,
        "available() must stay true while busy - it is a config gate, not a capacity check"
      );

      // Resolve so pending state/timers are cleaned up before the test ends
      const msg = JSON.parse(spawnedProcesses[0].lastWrite);
      spawnedProcesses[0].simulateResponse(msg.id, true, { content: "done" });
      await reqPromise;

      assert.strictEqual(bridge.hasAvailableWorker(), true, "hasAvailableWorker() returns true once the worker frees up");
    });

    it("hasAvailableWorker() is true when a ready, idle worker exists", async () => {
      bridge = require("../src/services/pythonBridge");
      const startPromise = bridge.start();
      setImmediate(() => {
        spawnedProcesses[0].simulateReady(true);
        spawnedProcesses[1].simulateReady(true);
      });
      await startPromise;

      assert.strictEqual(bridge.hasAvailableWorker(), true, "should have a dispatchable worker with idle ready workers");
    });

    it("hasAvailableWorker() reflects mixed workers: true while any worker is dispatchable, false once all are busy", async () => {
      bridge = require("../src/services/pythonBridge"); // MODEL_WORKERS defaults to 2 in beforeEach
      const startPromise = bridge.start();
      setImmediate(() => {
        spawnedProcesses[0].simulateReady(true);
        spawnedProcesses[1].simulateReady(true);
      });
      await startPromise;

      // Occupy worker 0 only - worker 1 is still idle
      const req1Promise = bridge.request("chat", { prompt: "occupy-1" });
      await new Promise((r) => setImmediate(r));

      assert.strictEqual(bridge.hasAvailableWorker(), true, "should have a dispatchable worker while worker 1 is still idle");
      assert.strictEqual(bridge.available(), true, "available() is unaffected by worker 0 being busy");

      // Occupy worker 1 too - now both workers are busy
      const req2Promise = bridge.request("chat", { prompt: "occupy-2" });
      await new Promise((r) => setImmediate(r));

      assert.strictEqual(bridge.hasAvailableWorker(), false, "no worker is dispatchable once all workers are busy");
      assert.strictEqual(bridge.available(), true, "available() is unaffected even when every worker is busy");

      // Clean up
      const msg1 = JSON.parse(spawnedProcesses[0].lastWrite);
      const msg2 = JSON.parse(spawnedProcesses[1].lastWrite);
      spawnedProcesses[0].simulateResponse(msg1.id, true, { content: "d1" });
      spawnedProcesses[1].simulateResponse(msg2.id, true, { content: "d2" });
      await Promise.all([req1Promise, req2Promise]);
    });

    it("REGRESSION #278/#279: a third concurrent request queues through the bridge instead of falling back to placeholder output", async () => {
      // Reproduces the exact shape of model.js's gating pattern at all 8 of
      // its call sites: `if (bridge.available()) { try bridge.request() }
      // else { use placeholder }`. With MODEL_WORKERS=2, a third concurrent
      // request must still be routed through bridge.request() (which queues
      // it internally) rather than skipping the bridge because a capacity
      // check reported "unavailable". This mirrors the maintainer's repro
      // for the PR #279 review.
      bridge = require("../src/services/pythonBridge"); // MODEL_WORKERS defaults to 2 in beforeEach

      const startPromise = bridge.start();
      setImmediate(() => {
        spawnedProcesses[0].simulateReady(true);
        spawnedProcesses[1].simulateReady(true);
      });
      await startPromise;

      // Mirrors model.js: gate on bridge.available(), fall back to a
      // placeholder marker when it reports false.
      async function callLikeModelJs(prompt) {
        if (bridge.available()) {
          return bridge.request("chat", { prompt });
        }
        return { content: "PLACEHOLDER", placeholder: true };
      }

      const req1Promise = callLikeModelJs("req1");
      const req2Promise = callLikeModelJs("req2");
      await new Promise((r) => setImmediate(r));

      // Both workers are now busy; available() must still gate "true" here -
      // this is the exact moment the old (buggy) available() implementation
      // would have flipped to false and made the third call below skip the
      // bridge entirely.
      assert.strictEqual(bridge.available(), true, "available() must remain true while both workers are busy");
      assert.strictEqual(bridge.hasAvailableWorker(), false, "no worker is currently dispatchable");

      const req3Promise = callLikeModelJs("req3");
      await new Promise((r) => setImmediate(r));

      // The third request must have been queued inside the pool, not
      // resolved as a placeholder.
      const pool = bridge._pool();
      assert.strictEqual(pool.queue.length, 1, "the third request should be queued by the pool, not skipped");

      // Complete the two in-flight requests; the queued third should then
      // dispatch and complete for real.
      const msg1 = JSON.parse(spawnedProcesses[0].lastWrite);
      const msg2 = JSON.parse(spawnedProcesses[1].lastWrite);
      spawnedProcesses[0].simulateResponse(msg1.id, true, { content: "result1" });
      spawnedProcesses[1].simulateResponse(msg2.id, true, { content: "result2" });

      const result1 = await req1Promise;
      const result2 = await req2Promise;
      assert.strictEqual(result1.content, "result1");
      assert.strictEqual(result2.content, "result2");

      await new Promise((r) => setImmediate(r));
      const workerWithReq3 = spawnedProcesses.find((p) => p.lastWrite && JSON.parse(p.lastWrite).params.prompt === "req3");
      assert.ok(workerWithReq3, "queued third request should eventually dispatch to a freed worker");
      const msg3 = JSON.parse(workerWithReq3.lastWrite);
      workerWithReq3.simulateResponse(msg3.id, true, { content: "result3" });

      const result3 = await req3Promise;
      assert.strictEqual(result3.content, "result3", "third request must complete via the bridge, not a placeholder");
      assert.strictEqual(result3.placeholder, undefined, "third request must not have fallen back to placeholder output");
    });
  });
});

});


// ============================================================================
// Issue #276 Regression Tests: Timeout path directly calls handleWorkerExit
// ============================================================================

describe("Issue #276: timeout path calls handleWorkerExit directly", () => {
  let bridge = null;
  let originalExistsSync = null;

  beforeEach(() => {
    // Reset module state
    delete require.cache[require.resolve("../src/services/pythonBridge")];
    spawnedProcesses.length = 0;

    // Setup environment
    process.env.MODEL_ENABLED = "true";
    process.env.MODEL_PATH = "/fake/model.pt";
    process.env.TOKENIZER_PATH = "/fake/tokenizer";
    process.env.MODEL_WORKERS = "1";
    process.env.MODEL_TIMEOUT_MS = "50";

    // Mock fs.existsSync
    const fs = require("fs");
    originalExistsSync = fs.existsSync;
    fs.existsSync = (path) => {
      if (path.includes("model.pt")) return true;
      return originalExistsSync(path);
    };

    // Mock spawn
    mockSpawn = (command, args, options) => {
      return new MockChildProcess(command, args, options);
    };
  });

  afterEach(() => {
    // Restore fs.existsSync
    if (originalExistsSync) {
      const fs = require("fs");
      fs.existsSync = originalExistsSync;
      originalExistsSync = null;
    }

    // Cleanup
    mockSpawn = null;
    spawnedProcesses.length = 0;
    if (bridge && bridge._pool && bridge._pool()) {
      try {
        bridge._pool().shutdown();
      } catch (e) {
        // ignore
      }
    }
  });

  it("REGRESSION #276: timeout handler calls handleWorkerExit without waiting for exit event", async () => {
    // Issue #276: A process stuck in uninterruptible state (e.g., CUDA ioctl)
    // may not be reaped even after SIGKILL. The timeout path must call
    // handleWorkerExit directly to ensure immediate pool cleanup/replenishment.

    bridge = require("../src/services/pythonBridge");

    // Mock child that NEVER emits exit (stuck in uninterruptible state)
    class StuckMockChildProcess extends MockChildProcess {
      kill(signal) {
        this.killed = true;
        // DO NOT emit exit - simulates process stuck even after SIGKILL
      }
    }

    mockSpawn = () => new StuckMockChildProcess();

    // Use immediate backoff but keep escalation timer pending
    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        if (ms >= 60000) return { startupTimeout: true }; // startup timeout
        if (ms >= 5000) return { escalationTimer: true }; // SIGKILL escalation
        if (ms >= 500 && ms <= 8000) {
          // Backoff - fire immediately
          fn();
          return null;
        }
        // Request timeout - run natively
        return setTimeout(fn, ms);
      },
      (id) => {
        if (id && (id.startupTimeout || id.escalationTimer)) return;
        clearTimeout(id);
      }
    );

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    assert.strictEqual(spawnedProcesses.length, 1, "should start with 1 worker");

    // Make a request that will timeout
    const reqPromise = bridge.request("generate", { prompt: "test" }).catch((err) => {
      if (!/timed out/.test(err.message)) throw err;
    });
    await new Promise((r) => setImmediate(r));

    // Wait for timeout to fire
    await new Promise((r) => setTimeout(r, 100));
    await reqPromise;

    // The stuck worker never emits exit, but handleWorkerExit should be called
    // from the timeout path (Issue #276 fix), triggering replacement spawn.
    await new Promise((r) => setImmediate(r));

    bridge._setTimerImpl(prev.set, prev.clear);

    // Verify replacement was spawned despite no exit event
    assert.ok(
      spawnedProcesses.length >= 2,
      `replacement should spawn via timeout path calling handleWorkerExit (got ${spawnedProcesses.length})`
    );
  });

  it("REGRESSION #276: handleWorkerExit is idempotent when called from both paths", async () => {
    // When timeout path calls handleWorkerExit AND the child later emits exit,
    // handleWorkerExit must be idempotent to avoid duplicate cleanup.

    bridge = require("../src/services/pythonBridge");

    // Use immediate backoff/timers
    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        if (ms >= 60000) return { startupTimeout: true };
        fn();
        return null;
      },
      () => {}
    );

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    const pool = bridge._pool();
    const worker = pool.workers[0];
    const initialCount = pool.workers.length;

    // Call handleWorkerExit twice rapidly (simulating timeout path + exit event)
    pool.handleWorkerExit(worker);
    pool.handleWorkerExit(worker);

    // Wait a tick for the removal to complete
    await new Promise((r) => setImmediate(r));

    bridge._setTimerImpl(prev.set, prev.clear);

    // The key test: calling handleWorkerExit twice should not crash
    // The second call should be a no-op because the worker is already removed
    assert.ok(true, "duplicate handleWorkerExit calls should not crash");
  });
});


// ============================================================================
// Regression tests: stale SIGKILL escalation after worker exits
//
// Root cause: the timeout path's escalation timer tracks whether the worker
// exited via a `once("exit", ...)` listener attached to the child process.
// Issue #276 made the timeout path call handleWorkerExit() directly and
// synchronously, right after sending SIGTERM - before the real (async) exit
// event can ever arrive. handleWorkerExit() -> cleanup() then calls
// child.removeAllListeners(), which strips that listener before it has a
// chance to observe the real exit. When the worker later does honour SIGTERM
// and exit, nothing is left to notice, so the escalation timer fires anyway,
// logs the misleading "worker did not exit after SIGTERM, sending SIGKILL",
// and sends SIGKILL to an already-dead process.
// ============================================================================

describe("Regression: stale SIGKILL escalation after worker exits", () => {
  let bridge = null;
  let originalExistsSync = null;

  // Models a real ChildProcess: kill() does NOT synchronously emit 'exit'
  // (real process termination is always asynchronous, reported later by the
  // OS/libuv). finishExit() simulates that later, real termination.
  class AsyncExitMockChildProcess extends MockChildProcess {
    constructor(...args) {
      super(...args);
      this.killSignals = [];
    }

    kill(signal) {
      this.killSignals.push(signal || "SIGTERM");
      this.killed = true;
      // Do not emit 'exit' here - a real process dies asynchronously.
    }

    finishExit(code = 0, signal = null) {
      if (signal) {
        this.signalCode = signal;
        this.exitCode = null;
      } else {
        this.exitCode = code;
        this.signalCode = null;
      }
      this.emit("exit", code, signal);
    }
  }

  beforeEach(() => {
    delete require.cache[require.resolve("../src/services/pythonBridge")];
    spawnedProcesses.length = 0;

    process.env.MODEL_ENABLED = "true";
    process.env.MODEL_PATH = "/fake/model.pt";
    process.env.TOKENIZER_PATH = "/fake/tokenizer";
    process.env.MODEL_WORKERS = "1";
    process.env.MODEL_TIMEOUT_MS = "50";

    const fs = require("fs");
    originalExistsSync = fs.existsSync;
    fs.existsSync = (path) => {
      if (path.includes("model.pt")) return true;
      return originalExistsSync(path);
    };

    mockSpawn = () => new AsyncExitMockChildProcess();
  });

  afterEach(() => {
    if (originalExistsSync) {
      const fs = require("fs");
      fs.existsSync = originalExistsSync;
      originalExistsSync = null;
    }

    mockSpawn = null;
    spawnedProcesses.length = 0;
    if (bridge && bridge._pool && bridge._pool()) {
      try {
        bridge._pool().shutdown();
      } catch (e) {
        // ignore
      }
    }
  });

  // Intercepts _setTimeout/_clearTimeout the same way the Issue #238/#276
  // tests do: startup timeouts and stability timers become inert sentinels,
  // backoff fires immediately, and the escalation timer is captured so it
  // can be fired deterministically instead of waiting out the real grace
  // period.
  function interceptTimers(bridge) {
    let escalationFn = null;
    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        if (ms >= 10000) return { longTimer: true }; // startup timeout / stability timer
        if (ms >= 5000) {
          escalationFn = fn;
          return { escalationTimer: true };
        }
        if (ms >= 500 && ms <= 8000) {
          fn();
          return null;
        }
        return setTimeout(fn, ms);
      },
      (id) => {
        if (id && (id.longTimer || id.escalationTimer)) return;
        clearTimeout(id);
      }
    );
    return { prev, getEscalationFn: () => escalationFn };
  }

  it("TEST A - REGRESSION: does not send a stale SIGKILL once the worker has actually exited after SIGTERM", async () => {
    bridge = require("../src/services/pythonBridge");
    const { prev, getEscalationFn } = interceptTimers(bridge);

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    const worker = spawnedProcesses[0];

    const reqPromise = bridge.request("chat", { prompt: "will-timeout" }).catch((err) => {
      if (!/timed out/.test(err.message)) throw err;
    });
    await new Promise((r) => setImmediate(r));

    // Let the 50ms request timeout fire: SIGTERM is sent, and the #276 fix
    // calls handleWorkerExit()/cleanup() synchronously right afterwards.
    await new Promise((r) => setTimeout(r, 100));
    await reqPromise;

    assert.ok(worker.killSignals.includes("SIGTERM"), "SIGTERM should have been sent");
    const escalationFn = getEscalationFn();
    assert.ok(escalationFn, "escalation timer should have been registered");

    // The worker cooperatively honours SIGTERM and exits shortly after -
    // well before the escalation grace period would fire. This happens
    // after cleanup() has already run and stripped the exit listener.
    worker.finishExit(0);

    // Fire the escalation timer (simulating the grace period elapsing).
    escalationFn();

    bridge._setTimerImpl(prev.set, prev.clear);

    assert.ok(
      !worker.killSignals.includes("SIGKILL"),
      `escalation must not send SIGKILL to an already-exited worker (signals sent: ${JSON.stringify(worker.killSignals)})`
    );
  });

  it("TEST B - worker that never exits after SIGTERM still gets escalated to SIGKILL", async () => {
    bridge = require("../src/services/pythonBridge");
    const { prev, getEscalationFn } = interceptTimers(bridge);

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    const worker = spawnedProcesses[0];

    const reqPromise = bridge.request("chat", { prompt: "will-timeout" }).catch((err) => {
      if (!/timed out/.test(err.message)) throw err;
    });
    await new Promise((r) => setImmediate(r));

    await new Promise((r) => setTimeout(r, 100));
    await reqPromise;

    assert.ok(worker.killSignals.includes("SIGTERM"), "SIGTERM should have been sent");
    const escalationFn = getEscalationFn();
    assert.ok(escalationFn, "escalation timer should have been registered");

    // The worker never exits (genuinely stuck) - do NOT call finishExit().
    escalationFn();

    bridge._setTimerImpl(prev.set, prev.clear);

    assert.ok(
      worker.killSignals.includes("SIGKILL"),
      "escalation must still send SIGKILL to a worker that never exited"
    );
  });
});


// ============================================================================
// Issue #278 Heartbeat/Liveness Tests
// ============================================================================

describe("Issue #278: Heartbeat liveness mechanism", () => {
  let bridge = null;
  let originalExistsSync = null;

  beforeEach(() => {
    delete require.cache[require.resolve("../src/services/pythonBridge")];
    spawnedProcesses.length = 0;

    process.env.MODEL_ENABLED = "true";
    process.env.MODEL_PATH = "/fake/model.pt";
    process.env.MODEL_TOKENIZER_PATH = "/fake/tokenizer";
    process.env.MODEL_WORKERS = "1";
    process.env.MODEL_HEARTBEAT_TIMEOUT_MS = "100"; // Short for testing
    process.env.MODEL_HEARTBEAT_STARTUP_GRACE_MS = "50"; // Short for testing

    const fs = require("fs");
    originalExistsSync = fs.existsSync;
    fs.existsSync = (path) => {
      if (path.includes("model.pt")) return true;
      return originalExistsSync(path);
    };

    mockSpawn = () => new MockChildProcess();
  });

  afterEach(() => {
    if (originalExistsSync) {
      const fs = require("fs");
      fs.existsSync = originalExistsSync;
      originalExistsSync = null;
    }

    delete process.env.MODEL_HEARTBEAT_TIMEOUT_MS;
    delete process.env.MODEL_HEARTBEAT_STARTUP_GRACE_MS;
    delete process.env.MODEL_TIMEOUT_MS;

    mockSpawn = null;
    spawnedProcesses.length = 0;
    if (bridge && bridge._pool && bridge._pool()) {
      try {
        bridge._pool().shutdown();
      } catch (e) {
        // ignore
      }
    }
  });

  it("TEST A: heartbeats are received and update worker lastHeartbeat timestamp", async () => {
    bridge = require("../src/services/pythonBridge");

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    const pool = bridge._pool();
    const worker = pool.workers[0];
    const initialHeartbeat = worker.lastHeartbeat;

    assert.ok(initialHeartbeat > 0, "lastHeartbeat should be set after worker becomes ready");

    // Simulate heartbeat
    await new Promise((r) => setTimeout(r, 10));
    spawnedProcesses[0].stdout.emit("data", Buffer.from('{"type":"heartbeat"}\n'));
    await new Promise((r) => setImmediate(r));

    assert.ok(worker.lastHeartbeat > initialHeartbeat, "lastHeartbeat should update when heartbeat is received");
  });

  it("TEST B: healthy busy worker continues sending heartbeats and is NOT treated as wedged", async () => {
    process.env.MODEL_TIMEOUT_MS = "500"; // Long enough for the test
    bridge = require("../src/services/pythonBridge");

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    const pool = bridge._pool();
    const worker = pool.workers[0];

    // Start a long-running request
    const reqPromise = bridge.request("chat", { prompt: "long-running" });
    await new Promise((r) => setImmediate(r));

    assert.strictEqual(worker.busy, true, "worker should be busy");

    // Simulate periodic heartbeats during the long request
    const heartbeatInterval = setInterval(() => {
      if (spawnedProcesses[0] && !spawnedProcesses[0].killed) {
        spawnedProcesses[0].stdout.emit("data", Buffer.from('{"type":"heartbeat"}\n'));
      }
    }, 20);

    // Wait longer than HEARTBEAT_TIMEOUT_MS but keep sending heartbeats
    await new Promise((r) => setTimeout(r, 150));

    clearInterval(heartbeatInterval);

    // Worker should still be alive (not killed)
    assert.strictEqual(spawnedProcesses[0].killed, false, "worker should NOT be killed while heartbeats continue");
    assert.strictEqual(worker.ready, true, "worker should still be ready");

    // Complete the request normally
    const msg = JSON.parse(spawnedProcesses[0].lastWrite);
    spawnedProcesses[0].simulateResponse(msg.id, true, { content: "done" });
    await reqPromise;
  });

  it("TEST C: worker that stops sending heartbeats is detected as wedged and terminated", async () => {
    bridge = require("../src/services/pythonBridge");

    // Use immediate backoff for faster test
    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        if (ms >= 60000) return { startupTimeout: true };
        if (ms >= 5000) return { escalationTimer: true }; // Don't auto-fire escalation
        if (ms >= 500 && ms <= 8000) {
          fn();
          return null;
        }
        return setTimeout(fn, ms);
      },
      (id) => {
        if (id && (id.startupTimeout || id.escalationTimer)) return;
        clearTimeout(id);
      }
    );

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    const worker = spawnedProcesses[0];

    // Start a request (worker becomes busy)
    const reqPromise = bridge.request("chat", { prompt: "will-wedge" }).catch(() => {});
    await new Promise((r) => setImmediate(r));

    // DO NOT send any heartbeats - simulates wedged worker
    // Wait for heartbeat timeout + startup grace + margin
    await new Promise((r) => setTimeout(r, 200));

    bridge._setTimerImpl(prev.set, prev.clear);

    // Worker should have been killed due to heartbeat timeout
    assert.ok(worker.killed, "worker should be killed when heartbeats stop");

    // Cleanup
    await reqPromise;
  });

  it("TEST D: wedged worker follows safe termination/replacement path (SIGTERM → SIGKILL)", async () => {
    bridge = require("../src/services/pythonBridge");

    // SIGTERM-immune worker that needs SIGKILL
    class SigTermImmuneMockChildProcess extends MockChildProcess {
      constructor(...args) {
        super(...args);
        this.killSignals = [];
        this.exited = false;
      }

      kill(signal) {
        this.killSignals.push(signal || "SIGTERM");
        this.killed = true;
        if (signal === "SIGKILL") {
          this.exited = true;
          this.signalCode = "SIGKILL";
          this.emit("exit", 137);
        }
      }
    }

    mockSpawn = () => new SigTermImmuneMockChildProcess();

    let escalationFn = null;
    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        if (ms >= 60000) return { startupTimeout: true };
        if (ms >= 5000) {
          escalationFn = fn;
          return { escalationTimer: true };
        }
        if (ms >= 500 && ms <= 8000) {
          fn();
          return null;
        }
        return setTimeout(fn, ms);
      },
      (id) => {
        if (id && (id.startupTimeout || id.escalationTimer)) return;
        clearTimeout(id);
      }
    );

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    const worker = spawnedProcesses[0];

    // Start request then stop heartbeats
    const reqPromise = bridge.request("chat", { prompt: "wedge" }).catch(() => {});
    await new Promise((r) => setImmediate(r));

    // Wait for heartbeat timeout
    await new Promise((r) => setTimeout(r, 200));

    // SIGTERM should be sent
    assert.ok(worker.killSignals.includes("SIGTERM"), "SIGTERM should be sent to wedged worker");
    assert.ok(escalationFn, "escalation timer should be registered");

    // Fire escalation
    escalationFn();
    await new Promise((r) => setImmediate(r));

    bridge._setTimerImpl(prev.set, prev.clear);

    // SIGKILL should be sent
    assert.ok(worker.killSignals.includes("SIGKILL"), "SIGKILL should be sent after escalation");

    // Replacement should spawn
    await new Promise((r) => setImmediate(r));
    assert.ok(spawnedProcesses.length >= 2, "replacement worker should spawn");

    await reqPromise;
  });

  it("TEST E: multiple workers - one wedged does not affect healthy workers", async () => {
    process.env.MODEL_WORKERS = "2";
    process.env.MODEL_TIMEOUT_MS = "500"; // Long enough for the test
    bridge = require("../src/services/pythonBridge");

    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        if (ms >= 60000) return { startupTimeout: true };
        if (ms >= 5000) return { escalationTimer: true };
        if (ms >= 500 && ms <= 8000) {
          fn();
          return null;
        }
        return setTimeout(fn, ms);
      },
      (id) => {
        if (id && (id.startupTimeout || id.escalationTimer)) return;
        clearTimeout(id);
      }
    );

    const startPromise = bridge.start();
    setImmediate(() => {
      spawnedProcesses[0].simulateReady(true);
      spawnedProcesses[1].simulateReady(true);
    });
    await startPromise;

    const worker0 = spawnedProcesses[0];
    const worker1 = spawnedProcesses[1];

    // Worker 0: start request and send heartbeats (healthy busy)
    const req0Promise = bridge.request("chat", { prompt: "req0" });
    await new Promise((r) => setImmediate(r));

    const heartbeat0 = setInterval(() => {
      if (!worker0.killed) {
        worker0.stdout.emit("data", Buffer.from('{"type":"heartbeat"}\n'));
      }
    }, 20);

    // Worker 1: start request but stop heartbeats (wedged)
    const req1Promise = bridge.request("chat", { prompt: "req1" }).catch(() => {});
    await new Promise((r) => setImmediate(r));

    // Wait for worker 1 to be detected as wedged
    await new Promise((r) => setTimeout(r, 200));

    clearInterval(heartbeat0);

    bridge._setTimerImpl(prev.set, prev.clear);

    // Worker 1 should be killed
    assert.ok(worker1.killed, "wedged worker 1 should be killed");

    // Worker 0 should still be alive
    assert.strictEqual(worker0.killed, false, "healthy worker 0 should NOT be killed");

    // Complete worker 0 request
    const msg0 = JSON.parse(worker0.lastWrite);
    worker0.simulateResponse(msg0.id, true, { content: "done0" });
    await req0Promise;

    await req1Promise;
  });

  it("TEST F: startup grace period prevents premature wedge detection", async () => {
    process.env.MODEL_HEARTBEAT_STARTUP_GRACE_MS = "200";
    bridge = require("../src/services/pythonBridge");

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    const worker = spawnedProcesses[0];

    // During grace period: no heartbeats yet, but worker should not be killed
    await new Promise((r) => setTimeout(r, 100));

    assert.strictEqual(worker.killed, false, "worker should NOT be killed during startup grace period");

    // Send heartbeats periodically after grace period
    const heartbeatInterval = setInterval(() => {
      if (!worker.killed) {
        worker.stdout.emit("data", Buffer.from('{"type":"heartbeat"}\n'));
      }
    }, 20);

    // After grace period with heartbeats, worker should remain alive
    await new Promise((r) => setTimeout(r, 150));

    clearInterval(heartbeatInterval);

    assert.strictEqual(worker.killed, false, "worker should remain alive after grace period when heartbeats are sent");
  });

  it("TEST G: existing #277 idempotency test still passes with heartbeat mechanism", async () => {
    // Verify that handleWorkerExit idempotency (Issue #277) is preserved
    bridge = require("../src/services/pythonBridge");

    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        if (ms >= 60000) return { startupTimeout: true };
        if (ms >= 50 && ms <= 10000) return { livenessTimer: true }; // Don't fire liveness checks (covers 50ms-5000ms range)
        fn();
        return null;
      },
      (id) => {
        if (id && (id.startupTimeout || id.livenessTimer)) return;
      }
    );

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    const pool = bridge._pool();
    const worker = pool.workers[0];

    // Call handleWorkerExit twice (simulating timeout + exit event)
    pool.handleWorkerExit(worker);
    pool.handleWorkerExit(worker);

    await new Promise((r) => setImmediate(r));

    bridge._setTimerImpl(prev.set, prev.clear);

    // Should not crash - idempotency is preserved
    assert.ok(true, "handleWorkerExit is still idempotent with heartbeat mechanism");
  });

  it("TEST H: existing #281 stale SIGKILL test still passes with heartbeat mechanism", async () => {
    // Verify that stale SIGKILL escalation fix (Issue #281) is preserved
    bridge = require("../src/services/pythonBridge");

    class AsyncExitMockChildProcess extends MockChildProcess {
      constructor(...args) {
        super(...args);
        this.killSignals = [];
      }

      kill(signal) {
        this.killSignals.push(signal || "SIGTERM");
        this.killed = true;
      }

      finishExit(code = 0, signal = null) {
        if (signal) {
          this.signalCode = signal;
          this.exitCode = null;
        } else {
          this.exitCode = code;
          this.signalCode = null;
        }
        this.emit("exit", code, signal);
      }
    }

    mockSpawn = () => new AsyncExitMockChildProcess();

    let escalationFn = null;
    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        if (ms >= 10000) return { longTimer: true };
        if (ms >= 5000) {
          escalationFn = fn;
          return { escalationTimer: true };
        }
        if (ms >= 500 && ms <= 8000) {
          fn();
          return null;
        }
        return setTimeout(fn, ms);
      },
      (id) => {
        if (id && (id.longTimer || id.escalationTimer)) return;
        clearTimeout(id);
      }
    );

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    const worker = spawnedProcesses[0];

    // Stop heartbeats to trigger wedge detection
    await new Promise((r) => setTimeout(r, 200));

    assert.ok(worker.killSignals.includes("SIGTERM"), "SIGTERM should be sent");
    assert.ok(escalationFn, "escalation timer should be registered");

    // Worker exits cooperatively after SIGTERM
    worker.finishExit(0);

    // Fire escalation
    escalationFn();

    bridge._setTimerImpl(prev.set, prev.clear);

    // SIGKILL should NOT be sent to already-exited worker (Issue #281 fix)
    assert.ok(
      !worker.killSignals.includes("SIGKILL"),
      "stale SIGKILL escalation fix is preserved - no SIGKILL to exited worker"
    );
  });
});

// ============================================================================
// Issue #239 regression tests:
// [Bug]: Timed-out worker is never force-killed and the pool can drain to zero
//
// Verifies that:
// A. Timed-out worker is actually terminated/removed and force-killed via SIGKILL.
// B. The pool replaces the worker so configured capacity remains usable.
// C. Repeated worker timeouts do not drain the pool to zero (exceeding MAX_RESTART_ATTEMPTS).
// D. After repeated timeouts, a subsequent normal task successfully completes.
// E. Process crash failures (non-timeout) still enforce MAX_RESTART_ATTEMPTS.
// F. Race conditions between completion and timeout are handled consistently.
// ============================================================================

describe("Issue #239 regression: timed-out worker force-kill and pool capacity preservation", { concurrency: 1 }, () => {
  let bridge = null;
  let originalExistsSync = null;

  class SigTermImmuneMockChildProcess extends MockChildProcess {
    constructor(...args) {
      super(...args);
      this.killSignals = [];
      this.exited = false;
    }

    kill(signal) {
      this.killSignals.push(signal || "SIGTERM");
      this.killed = true;
      if (signal === "SIGKILL") {
        this.exited = true;
        this.signalCode = "SIGKILL";
        this.emit("exit", 137);
      }
    }
  }

  beforeEach(() => {
    delete require.cache[require.resolve("../src/services/pythonBridge")];
    spawnedProcesses.length = 0;

    process.env.MODEL_ENABLED = "true";
    process.env.MODEL_PATH = "/fake/model.pt";
    process.env.TOKENIZER_PATH = "/fake/tokenizer";
    process.env.MODEL_WORKERS = "1";
    process.env.MODEL_TIMEOUT_MS = "50";
    delete process.env.MODEL_HEARTBEAT_TIMEOUT_MS;
    delete process.env.MODEL_HEARTBEAT_STARTUP_GRACE_MS;

    const fs = require("fs");
    originalExistsSync = fs.existsSync;
    fs.existsSync = (p) => {
      if (p && p.includes("model.pt")) return true;
      return originalExistsSync ? originalExistsSync(p) : false;
    };

    mockSpawn = (...args) => new MockChildProcess(...args);
  });

  afterEach(() => {
    if (originalExistsSync) {
      const fs = require("fs");
      fs.existsSync = originalExistsSync;
      originalExistsSync = null;
    }

    delete process.env.MODEL_TIMEOUT_MS;

    mockSpawn = null;
    spawnedProcesses.length = 0;
    if (bridge && bridge._pool && bridge._pool()) {
      try {
        bridge._pool().shutdown();
      } catch (e) {
        // ignore
      }
    }
  });

  it("TEST A: timed-out worker is forcefully terminated via SIGKILL and removed from pool bookkeeping", async () => {
    bridge = require("../src/services/pythonBridge");
    mockSpawn = () => new SigTermImmuneMockChildProcess();

    let escalationFn = null;
    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        if (ms >= 60000) return { startupTimeout: true };
        if (ms >= 5000) {
          escalationFn = fn;
          return { escalationTimer: true };
        }
        if (ms >= 500 && ms <= 8000) {
          fn();
          return null;
        }
        return setTimeout(fn, ms);
      },
      (id) => {
        if (id && (id.startupTimeout || id.escalationTimer)) return;
        clearTimeout(id);
      }
    );

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    const worker = spawnedProcesses[0];
    const pool = bridge._pool();

    const reqPromise = bridge.request("chat", { prompt: "will-timeout" }).catch((err) => {
      if (!/timed out/.test(err.message)) throw err;
    });
    await new Promise((r) => setImmediate(r));

    // Wait for 50ms request timeout
    await new Promise((r) => setTimeout(r, 100));
    await reqPromise;

    // Worker received SIGTERM
    assert.ok(worker.killSignals.includes("SIGTERM"), "SIGTERM should have been sent on timeout");
    assert.strictEqual(worker.exited, false, "SIGTERM-immune worker is still alive before SIGKILL");

    // Timed-out worker should be removed from active pool immediately (bookkeeping)
    assert.ok(!pool.workers.includes(worker), "timed-out worker must be removed from pool.workers");

    // Fire SIGKILL escalation
    assert.ok(escalationFn, "escalation timer must be registered");
    escalationFn();
    await new Promise((r) => setImmediate(r));

    assert.ok(worker.killSignals.includes("SIGKILL"), "SIGKILL must be sent on escalation");
    assert.strictEqual(worker.exited, true, "worker must be dead after SIGKILL");

    bridge._setTimerImpl(prev.set, prev.clear);
  });

  it("TEST B: pool replaces timed-out worker so configured pool capacity remains usable", async () => {
    process.env.MODEL_WORKERS = "2";
    bridge = require("../src/services/pythonBridge");

    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        if (ms >= 60000) return { startupTimeout: true };
        if (ms >= 5000) return { escalationTimer: true };
        if (ms >= 500 && ms <= 8000) {
          fn();
          return null;
        }
        return setTimeout(fn, ms);
      },
      () => {}
    );

    const startPromise = bridge.start();
    setImmediate(() => {
      spawnedProcesses[0].simulateReady(true);
      spawnedProcesses[1].simulateReady(true);
    });
    await startPromise;

    const pool = bridge._pool();
    assert.strictEqual(pool.workers.length, 2, "initial pool size should be 2");

    // Time out worker 0
    const reqPromise = bridge.request("chat", { prompt: "timeout-worker-0" }).catch((err) => {
      if (!/timed out/.test(err.message)) throw err;
    });
    await new Promise((r) => setImmediate(r));
    await new Promise((r) => setTimeout(r, 100));
    await reqPromise;

    // Give a tick for replacement to spawn
    await new Promise((r) => setImmediate(r));
    assert.ok(spawnedProcesses.length >= 3, "replacement worker should be spawned");

    const replacement = spawnedProcesses[spawnedProcesses.length - 1];
    replacement.simulateReady(true);
    await new Promise((r) => setImmediate(r));

    // Configured pool capacity must remain at 2
    const readyWorkers = pool.workers.filter((w) => w.ready);
    assert.strictEqual(readyWorkers.length, 2, "pool must maintain configured capacity of 2 ready workers");
    assert.strictEqual(bridge.available(), true, "bridge must be available");
    assert.strictEqual(bridge.hasAvailableWorker(), true, "bridge must have available dispatchable worker");

    bridge._setTimerImpl(prev.set, prev.clear);
  });

  it("TEST C: repeated worker timeouts do not drain the pool to zero (exceeding MAX_RESTART_ATTEMPTS)", async () => {
    process.env.MODEL_WORKERS = "1";
    bridge = require("../src/services/pythonBridge");

    // Fast backoff timers so 7 timeouts execute quickly
    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        if (ms >= 60000) return { startupTimeout: true };
        if (ms >= 5000) return { escalationTimer: true };
        if (ms >= 500 && ms <= 8000) {
          fn();
          return null;
        }
        return setTimeout(fn, ms);
      },
      () => {}
    );

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    const pool = bridge._pool();

    // Trigger 7 consecutive timeouts (MAX_RESTART_ATTEMPTS is 5)
    for (let i = 0; i < 7; i++) {
      const currentWorker = spawnedProcesses[spawnedProcesses.length - 1];

      const reqPromise = bridge.request("chat", { prompt: `timeout-run-${i}` }).catch((err) => {
        if (!/timed out/.test(err.message)) throw err;
      });
      await new Promise((r) => setImmediate(r));

      // Wait for 50ms request timeout
      await new Promise((r) => setTimeout(r, 80));
      await reqPromise;

      // Replacement should have spawned
      await new Promise((r) => setImmediate(r));
      const replacement = spawnedProcesses[spawnedProcesses.length - 1];
      assert.notStrictEqual(replacement, currentWorker, `replacement should spawn for timeout ${i + 1}`);

      // Make replacement ready
      replacement.simulateReady(true);
      await new Promise((r) => setImmediate(r));

      // Capacity check after each timeout
      assert.strictEqual(pool.workers.length, 1, `pool should retain 1 worker after timeout ${i + 1}`);
      assert.strictEqual(bridge.available(), true, `pool should remain available after timeout ${i + 1}`);
      assert.strictEqual(bridge.hasAvailableWorker(), true, `pool should have dispatchable worker after timeout ${i + 1}`);
    }

    // After 7 timeouts (exceeding 5-attempt cap), pool must NOT be drained
    assert.strictEqual(pool.workers.length, 1, "pool must not drain to zero after 7 consecutive timeouts");
    assert.strictEqual(bridge.available(), true, "pool must remain available after 7 timeouts");
    assert.strictEqual(bridge.hasAvailableWorker(), true, "pool must have an available worker");

    bridge._setTimerImpl(prev.set, prev.clear);
  });

  it("TEST D: after repeated worker timeouts, a subsequent normal task successfully completes", async () => {
    process.env.MODEL_WORKERS = "1";
    bridge = require("../src/services/pythonBridge");

    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        if (ms >= 60000) return { startupTimeout: true };
        if (ms >= 5000) return { escalationTimer: true };
        if (ms >= 500 && ms <= 8000) {
          fn();
          return null;
        }
        return setTimeout(fn, ms);
      },
      () => {}
    );

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    // Trigger 6 consecutive timeouts
    for (let i = 0; i < 6; i++) {
      const req = bridge.request("chat", { prompt: `timeout-${i}` }).catch(() => {});
      await new Promise((r) => setImmediate(r));
      await new Promise((r) => setTimeout(r, 80));
      await req;

      await new Promise((r) => setImmediate(r));
      const replacement = spawnedProcesses[spawnedProcesses.length - 1];
      replacement.simulateReady(true);
      await new Promise((r) => setImmediate(r));
    }

    // Now send a normal task
    const normalPromise = bridge.request("chat", { prompt: "normal-task-after-recovery" });
    await new Promise((r) => setImmediate(r));

    const activeWorker = spawnedProcesses[spawnedProcesses.length - 1];
    assert.ok(activeWorker.lastWrite, "active worker should receive normal request");
    const msg = JSON.parse(activeWorker.lastWrite);
    assert.strictEqual(msg.params.prompt, "normal-task-after-recovery");

    // Normal task succeeds
    activeWorker.simulateResponse(msg.id, true, { content: "normal-task-success" });
    const result = await normalPromise;

    assert.strictEqual(result.content, "normal-task-success", "normal task must complete successfully after timeouts");

    bridge._setTimerImpl(prev.set, prev.clear);
  });

  it("TEST E: non-timeout process crashes still respect MAX_RESTART_ATTEMPTS and disable the pool", async () => {
    process.env.MODEL_WORKERS = "1";
    bridge = require("../src/services/pythonBridge");

    const prev = bridge._setTimerImpl(
      (fn) => {
        fn();
        return null;
      },
      () => {}
    );

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    // Crash the worker repeatedly on startup/exit (not a request timeout)
    for (let i = 0; i < 5; i++) {
      const current = spawnedProcesses[spawnedProcesses.length - 1];
      current.simulateExit(1);
      await new Promise((r) => setImmediate(r));
      const replacement = spawnedProcesses[spawnedProcesses.length - 1];
      if (replacement !== current) {
        replacement.simulateReady(false);
        await new Promise((r) => setImmediate(r));
      }
    }

    // One more crash to exhaust the cap
    const last = spawnedProcesses[spawnedProcesses.length - 1];
    last.simulateExit(1);
    await new Promise((r) => setImmediate(r));

    bridge._setTimerImpl(prev.set, prev.clear);

    // Non-timeout crashes must still disable the pool after reaching the cap
    assert.strictEqual(bridge.available(), false, "pool must be disabled after crash restart limit exceeded");
  });

  it("TEST F: race condition between worker completion and timeout detection is handled safely", async () => {
    process.env.MODEL_WORKERS = "1";
    process.env.MODEL_TIMEOUT_MS = "50";
    bridge = require("../src/services/pythonBridge");

    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        if (ms >= 60000) return { startupTimeout: true };
        if (ms >= 5000) return { escalationTimer: true };
        if (ms >= 500 && ms <= 8000) {
          fn();
          return null;
        }
        return setTimeout(fn, ms);
      },
      () => {}
    );

    const startPromise = bridge.start();
    setImmediate(() => spawnedProcesses[0].simulateReady(true));
    await startPromise;

    const worker = spawnedProcesses[0];

    // Case 1: Response arrives before timeout -> timer is cleared, request resolves
    const req1Promise = bridge.request("chat", { prompt: "fast-response" });
    await new Promise((r) => setImmediate(r));
    const msg1 = JSON.parse(worker.lastWrite);
    worker.simulateResponse(msg1.id, true, { content: "fast-success" });
    const result1 = await req1Promise;
    assert.strictEqual(result1.content, "fast-success", "fast response completes normally");

    // Case 2: Response arrives AFTER timeout -> rejected on timeout, late response is ignored without error
    const req2Promise = bridge.request("chat", { prompt: "slow-then-late" }).catch((err) => {
      if (!/timed out/.test(err.message)) throw err;
      return "timed-out-ok";
    });
    await new Promise((r) => setImmediate(r));
    const msg2 = JSON.parse(worker.lastWrite);

    // Wait for timeout to fire
    await new Promise((r) => setTimeout(r, 100));
    const res2 = await req2Promise;
    assert.strictEqual(res2, "timed-out-ok", "request timed out");

    // Simulate late response from dead worker - should be ignored without throwing
    assert.doesNotThrow(() => {
      worker.simulateResponse(msg2.id, true, { content: "late-response-ignored" });
    }, "late response for timed-out request must be ignored safely");

    bridge._setTimerImpl(prev.set, prev.clear);
  });
});


// ============================================================================
// Issue #386 Regression Tests: worker that exits before reporting ready
// ============================================================================

// A worker can die before it reports ready: a broken virtualenv, an OOM kill while
// the model loads, PYTHON_BIN=/usr/bin/false. The exit handler cancelled the startup
// safety timer and nothing else settled the startup promise, so start() and the
// first request waited forever.
describe("Issue #386: worker that exits before reporting ready", () => {
  let bridge = null;
  let originalExistsSync = null;
  let savedEnv = {};

  // Everything the bridge reads when it loads. Earlier suites in this file leave their
  // own values behind, so pin each one here and put the previous values back afterwards.
  // undefined means unset, which leaves the bridge on its defaults.
  const MANAGED_ENV = {
    MODEL_ENABLED: "true",
    MODEL_PATH: "/fake/model.pt",
    TOKENIZER_PATH: "/fake/tokenizer",
    MODEL_WORKERS: "2",
    PYTHON_BIN: undefined,
    MODEL_TOOLS: undefined,
    MODEL_CLI_MODE: undefined,
    MODEL_CLI_ROOT: undefined,
    MODEL_TIMEOUT_MS: undefined,
    MODEL_STARTUP_TIMEOUT_MS: undefined,
    MODEL_WORKER_STABILITY_MS: undefined,
    MODEL_KILL_GRACE_MS: undefined,
    MODEL_HEARTBEAT_TIMEOUT_MS: undefined,
    MODEL_HEARTBEAT_STARTUP_GRACE_MS: undefined,
  };

  beforeEach(() => {
    // Reset module state
    delete require.cache[require.resolve("../src/services/pythonBridge")];
    spawnedProcesses.length = 0;

    savedEnv = {};
    for (const [key, value] of Object.entries(MANAGED_ENV)) {
      savedEnv[key] = process.env[key];
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }

    // Mock fs.existsSync
    const fs = require("fs");
    originalExistsSync = fs.existsSync;
    fs.existsSync = (path) => {
      if (path.includes("model.pt")) return true;
      return originalExistsSync(path);
    };

    // Mock spawn
    mockSpawn = (command, args, options) => {
      return new MockChildProcess(command, args, options);
    };
  });

  afterEach(() => {
    for (const [key, value] of Object.entries(savedEnv)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }

    // Restore fs.existsSync
    if (originalExistsSync) {
      const fs = require("fs");
      fs.existsSync = originalExistsSync;
      originalExistsSync = null;
    }

    // Cleanup
    mockSpawn = null;
    spawnedProcesses.length = 0;
    if (bridge && bridge._pool && bridge._pool()) {
      try {
        bridge._pool().shutdown();
      } catch (e) {
        // ignore
      }
    }
  });

  // Resolves to "STILL PENDING" instead of hanging the run when the original bug is present.
  const settlesWithin = (promise, ms) => {
    let timer;
    const timeout = new Promise((resolve) => {
      timer = setTimeout(() => resolve("STILL PENDING"), ms);
    });
    return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
  };

  const tick = () => new Promise((r) => setImmediate(r));

  // With the timers below unset, restart backoffs (500 ms doubling to 8 s) are the
  // only timers the bridge arms in this range.
  const isBackoff = (timer) => timer.ms >= 500 && timer.ms <= 8000;

  // Records every timer the bridge arms instead of running it, so a test decides which
  // restart backoff fires and when. Startup, liveness and stability timers never fire.
  const recordTimers = () => {
    const armed = [];
    const cleared = [];
    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        const timer = { fn, ms };
        armed.push(timer);
        return timer;
      },
      (timer) => {
        cleared.push(timer);
      }
    );
    return { armed, cleared, restore: () => bridge._setTimerImpl(prev.set, prev.clear) };
  };

  it("REGRESSION #386: start() resolves false when every worker exits before reporting ready", async () => {
    bridge = require("../src/services/pythonBridge");

    const startPromise = bridge.start();
    setImmediate(() => {
      spawnedProcesses[0].simulateExit(1);
      spawnedProcesses[1].simulateExit(1);
    });

    assert.strictEqual(await settlesWithin(startPromise, 1000), false, "start should fail, not hang");
    assert.strictEqual(bridge.available(), false, "the bridge should fall back instead of staying enabled");
  });

  it("REGRESSION #386: the first request is rejected, not left pending, when every worker exits before ready", async () => {
    bridge = require("../src/services/pythonBridge");

    const first = bridge.request("chat", { prompt: "hi" }).then(
      () => "resolved",
      (err) => `rejected: ${err.message}`
    );
    setImmediate(() => {
      spawnedProcesses[0].simulateExit(1);
      spawnedProcesses[1].simulateExit(1);
    });

    assert.strictEqual(await settlesWithin(first, 1000), "rejected: no workers available");
  });

  it("REGRESSION #386: a worker that exits before ready does not stall the first request for a healthy one", async () => {
    bridge = require("../src/services/pythonBridge");

    const first = bridge.request("chat", { prompt: "hi" });
    spawnedProcesses[1].onStdinWrite = (data) => {
      spawnedProcesses[1].simulateResponse(JSON.parse(data).id, true, { content: "from the healthy worker" });
    };
    setImmediate(() => {
      spawnedProcesses[0].simulateExit(1);
      spawnedProcesses[1].simulateReady(true);
    });

    assert.deepStrictEqual(await settlesWithin(first, 1000), { content: "from the healthy worker" });
  });

  it("REGRESSION #386: a ready line that arrives after the worker exited does not revive it", async () => {
    // 'exit' can be delivered ahead of stdout the child wrote before it died. The startup
    // promise is already settled by then, so the late line must not count as a start.
    process.env.MODEL_WORKERS = "1";
    bridge = require("../src/services/pythonBridge");

    const startPromise = bridge.start();
    const worker = bridge._pool().workers[0]; // the pool drops a dead worker, so keep a handle
    try {
      setImmediate(() => {
        spawnedProcesses[0].simulateExit(1);
        spawnedProcesses[0].simulateReady(true);
      });

      assert.strictEqual(
        await settlesWithin(startPromise, 1000),
        false,
        "a worker that already exited is not a started worker"
      );
      assert.strictEqual(worker.ready, false, "the dead worker must not be marked ready");
      assert.strictEqual(worker._livenessTimer, null, "a dead worker must not get a liveness watchdog");
    } finally {
      worker.cleanup(); // the pool no longer owns it, so afterEach will not
    }
  });

  it("REGRESSION #386: a worker that exits before ready is still replaced after its backoff, and the pool recovers", async () => {
    process.env.MODEL_WORKERS = "1";
    bridge = require("../src/services/pythonBridge");
    const timers = recordTimers();
    try {
      const startPromise = bridge.start();
      const startupTimer = timers.armed.find((timer) => timer.ms === 60000);
      setImmediate(() => spawnedProcesses[0].simulateExit(1));

      assert.strictEqual(await settlesWithin(startPromise, 1000), false, "start should fail, not hang");
      assert.ok(timers.cleared.includes(startupTimer), "the exit must still cancel the startup safety timer");
      assert.strictEqual(timers.armed.filter(isBackoff).length, 1, "exactly one restart should be scheduled");
      assert.strictEqual(spawnedProcesses.length, 1, "the replacement must wait for its backoff");

      timers.armed.find(isBackoff).fn();
      await tick();
      assert.strictEqual(spawnedProcesses.length, 2, "the replacement should spawn once the backoff fires");

      spawnedProcesses[1].onStdinWrite = (data) => {
        spawnedProcesses[1].simulateResponse(JSON.parse(data).id, true, { content: "recovered" });
      };
      spawnedProcesses[1].simulateReady(true);
      await tick();
      assert.strictEqual(bridge.available(), true, "the pool should recover once the replacement is ready");

      const reply = await settlesWithin(bridge.request("chat", { prompt: "after recovery" }), 1000);
      assert.deepStrictEqual(reply, { content: "recovered" });
    } finally {
      timers.restore();
    }
  });

  it("REGRESSION #386: a replacement that exits before ready is reported failed, like one that reports ready:false", async () => {
    // The Issue #152 recovery test, except that the failing replacement dies before it
    // reports anything. Its startup promise used to stay unsettled, so the restart logic
    // never recorded the failed attempt.
    process.env.MODEL_WORKERS = "1";
    bridge = require("../src/services/pythonBridge");
    const timers = recordTimers();
    try {
      const startPromise = bridge.start();
      setImmediate(() => spawnedProcesses[0].simulateReady(true));
      assert.strictEqual(await settlesWithin(startPromise, 1000), true);
      assert.strictEqual(bridge.available(), true);

      // The ready worker crashes: the existing restart path, which must not change.
      spawnedProcesses[0].simulateExit(1);
      await tick();
      timers.armed.filter(isBackoff)[0].fn();
      await tick();
      assert.strictEqual(spawnedProcesses.length, 2, "the replacement should spawn after the backoff");

      // The replacement dies before it ever reports ready.
      spawnedProcesses[1].simulateExit(1);
      await tick();
      assert.strictEqual(bridge.available(), false, "a replacement that never became ready must disable the pool");
      const backoffs = timers.armed.filter(isBackoff);
      assert.strictEqual(backoffs.length, 2, "the restart loop must keep going");

      backoffs[1].fn();
      await tick();
      spawnedProcesses[2].simulateReady(true);
      await tick();
      assert.strictEqual(bridge.available(), true, "the pool should recover once a later replacement is ready");
    } finally {
      timers.restore();
    }
  });

  it("REGRESSION #386: workers that keep exiting before ready are replaced MAX_RESTART_ATTEMPTS times, then given up on", async () => {
    process.env.MODEL_WORKERS = "1";
    bridge = require("../src/services/pythonBridge");
    const timers = recordTimers();
    try {
      const startPromise = bridge.start();

      // Kill each worker as soon as it spawns, before it can report ready, and run each
      // backoff, until the pool stops scheduling restarts.
      let killed = 0;
      while (killed < 20) {
        spawnedProcesses[killed].simulateExit(1);
        killed += 1;
        await tick();
        const next = timers.armed.filter(isBackoff)[killed - 1];
        if (!next) break;
        next.fn();
        await tick();
      }

      assert.strictEqual(await settlesWithin(startPromise, 1000), false, "start should fail, not hang");
      assert.deepStrictEqual(
        timers.armed.filter(isBackoff).map((timer) => timer.ms),
        [500, 1000, 2000, 4000, 8000],
        "each dead worker should be counted once, with the usual backoff"
      );
      assert.strictEqual(spawnedProcesses.length, 6, "the first worker plus MAX_RESTART_ATTEMPTS replacements");
      assert.strictEqual(bridge.available(), false, "the pool should be given up on after the cap");
    } finally {
      timers.restore();
    }
  });

  // The issue reproduces with PYTHON_BIN=/usr/bin/false. Windows has no such file, so
  // there node itself is the interpreter that dies at once: it rejects -m and exits 9.
  const EXITS_AT_ONCE =
    process.platform !== "win32" && require("fs").existsSync("/usr/bin/false") ? "/usr/bin/false" : process.execPath;

  it("REGRESSION #386 (real process): the first request is rejected when the interpreter exits before ready", async () => {
    mockSpawn = null; // fall through to the real child_process.spawn
    process.env.PYTHON_BIN = EXITS_AT_ONCE;
    bridge = require("../src/services/pythonBridge");
    // The restart loop is not under test, and its real timers would outlive the test.
    const timers = recordTimers();
    try {
      const first = bridge.request("chat", { prompt: "hi" }).then(
        () => "resolved",
        (err) => `rejected: ${err.message}`
      );

      assert.strictEqual(await settlesWithin(first, 5000), "rejected: no workers available");
      assert.strictEqual(spawnedProcesses.length, 0, "real child processes should have been used, not the mock");
      assert.strictEqual(bridge.available(), false, "the bridge should fall back instead of staying enabled");
    } finally {
      timers.restore();
    }
  });
});

describe("Issue #390 regression: worker exit before ready propagates startup failure", { concurrency: 1 }, () => {
  let bridge = null;
  let savedEnv = null;
  let originalExistsSync = null;

  const MANAGED_ENV = {
    MODEL_ENABLED: "true",
    MODEL_PATH: "/fake/model.pt",
    TOKENIZER_PATH: "/fake/tokenizer",
    MODEL_WORKERS: "1",
    PYTHON_BIN: undefined,
    MODEL_TOOLS: undefined,
    MODEL_CLI_MODE: undefined,
    MODEL_CLI_ROOT: undefined,
    MODEL_TIMEOUT_MS: undefined,
    MODEL_STARTUP_TIMEOUT_MS: undefined,
    MODEL_WORKER_STABILITY_MS: undefined,
    MODEL_KILL_GRACE_MS: undefined,
    MODEL_HEARTBEAT_TIMEOUT_MS: undefined,
    MODEL_HEARTBEAT_STARTUP_GRACE_MS: undefined,
  };

  beforeEach(() => {
    delete require.cache[require.resolve("../src/services/pythonBridge")];
    spawnedProcesses.length = 0;

    savedEnv = {};
    for (const [key, value] of Object.entries(MANAGED_ENV)) {
      savedEnv[key] = process.env[key];
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }

    const fs = require("fs");
    originalExistsSync = fs.existsSync;
    fs.existsSync = (path) => {
      if (path && path.includes("model.pt")) return true;
      return originalExistsSync ? originalExistsSync(path) : false;
    };

    mockSpawn = (command, args, options) => {
      return new MockChildProcess(command, args, options);
    };
  });

  afterEach(() => {
    for (const [key, value] of Object.entries(savedEnv)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }

    if (originalExistsSync) {
      const fs = require("fs");
      fs.existsSync = originalExistsSync;
      originalExistsSync = null;
    }

    mockSpawn = null;
    spawnedProcesses.length = 0;
    if (bridge && bridge._pool && bridge._pool()) {
      try {
        bridge._pool().shutdown();
      } catch (e) {
        // ignore
      }
    }
  });

  const settlesWithin = (promise, ms) => {
    let timer;
    const timeout = new Promise((resolve) => {
      timer = setTimeout(() => resolve("STILL PENDING"), ms);
    });
    return Promise.race([
      promise.then(
        (val) => (val === undefined ? "resolved: undefined" : val),
        (err) => `rejected: ${err.message}`
      ),
      timeout,
    ]).finally(() => clearTimeout(timer));
  };

  const tick = () => new Promise((r) => setImmediate(r));

  it("1. Worker exits before ready: Worker.prototype.start() rejects and state is cleaned up", async () => {
    bridge = require("../src/services/pythonBridge");
    const worker = new bridge.Worker(0);
    assert.strictEqual(worker.starting, false);

    const startPromise = worker.start();
    assert.strictEqual(worker.starting, true);
    await tick();

    // Exit before reporting ready
    spawnedProcesses[0].simulateExit(1);

    const result = await settlesWithin(startPromise, 1000);
    assert.match(result, /rejected: worker 0 startup failed: process exited before ready/);
    assert.strictEqual(worker.starting, false, "worker must not remain starting");
    assert.strictEqual(worker.ready, false, "worker must not be ready");
    assert.strictEqual(worker.child, null, "child process reference must be cleared");
    assert.strictEqual(worker._safetyTimer, null, "safety timer must be cleared");
  });

  it("2. Multiple concurrent worker.start() callers all settle correctly on exit before ready", async () => {
    bridge = require("../src/services/pythonBridge");
    const worker = new bridge.Worker(0);

    const p1 = worker.start();
    const p2 = worker.start();
    const p3 = worker.start();

    await tick();
    spawnedProcesses[0].simulateExit(1);

    const [r1, r2, r3] = await Promise.all([
      settlesWithin(p1, 1000),
      settlesWithin(p2, 1000),
      settlesWithin(p3, 1000),
    ]);

    assert.match(r1, /rejected: worker 0 startup failed: process exited before ready/);
    assert.match(r2, /rejected: worker 0 startup failed: process exited before ready/);
    assert.match(r3, /rejected: worker 0 startup failed: process exited before ready/);
    assert.strictEqual(worker.starting, false);
  });

  it("3. Worker error + exit sequence: startup rejects cleanly with no unhandled rejections", async () => {
    bridge = require("../src/services/pythonBridge");
    const worker = new bridge.Worker(0);

    const startPromise = worker.start();
    await tick();

    // Error event followed by exit event
    spawnedProcesses[0].emit("error", new Error("spawn ENOENT"));
    spawnedProcesses[0].simulateExit(1);

    const result = await settlesWithin(startPromise, 1000);
    assert.match(result, /rejected: worker 0 startup failed: process exited before ready/);
    assert.strictEqual(worker.starting, false);
    assert.strictEqual(worker.ready, false);
  });

  it("4. Worker exit due to signal rejects startup with deterministic error", async () => {
    bridge = require("../src/services/pythonBridge");
    const worker = new bridge.Worker(0);

    const startPromise = worker.start();
    await tick();

    // Child terminated by SIGKILL
    spawnedProcesses[0].emit("exit", null, "SIGKILL");

    const result = await settlesWithin(startPromise, 1000);
    assert.match(result, /rejected: worker 0 startup failed: process exited before ready/);
    assert.strictEqual(worker.starting, false);
    assert.strictEqual(worker.ready, false);
  });

  it("5. Late ready event after worker exit cannot revive the worker or resolve failed startup", async () => {
    bridge = require("../src/services/pythonBridge");
    const worker = new bridge.Worker(0);

    const startPromise = worker.start();
    await tick();

    spawnedProcesses[0].simulateExit(1);
    const result = await settlesWithin(startPromise, 1000);
    assert.match(result, /rejected: worker 0 startup failed: process exited before ready/);

    // Late ready arrives after exit
    spawnedProcesses[0].simulateReady(true);

    assert.strictEqual(worker.ready, false, "dead worker must not become ready from late event");
    assert.strictEqual(worker.starting, false);
    assert.strictEqual(worker.child, null);
    assert.strictEqual(worker._livenessTimer, null);
  });

  it("6. Subsequent worker startup can succeed after an earlier startup failure", async () => {
    bridge = require("../src/services/pythonBridge");
    const worker = new bridge.Worker(0);

    // First attempt fails
    const failPromise = worker.start();
    await tick();
    spawnedProcesses[0].simulateExit(1);
    await settlesWithin(failPromise, 1000);
    assert.strictEqual(worker.ready, false);

    // Second attempt on new worker instance succeeds
    const healthyWorker = new bridge.Worker(1);
    const successPromise = healthyWorker.start();
    await tick();
    spawnedProcesses[1].simulateReady(true);

    const res = await settlesWithin(successPromise, 1000);
    assert.strictEqual(res, healthyWorker);
    assert.strictEqual(healthyWorker.ready, true);
    assert.strictEqual(healthyWorker.starting, false);
    healthyWorker.cleanup();
  });

  it("7. WorkerPool.prototype.start() rejects multiple concurrent startup callers when workers exit before ready", async () => {
    bridge = require("../src/services/pythonBridge");
    const pool = new bridge.WorkerPool(1);

    const s1 = pool.start();
    const s2 = pool.start();
    const s3 = pool.start();

    await tick();
    spawnedProcesses[0].simulateExit(1);

    const [r1, r2, r3] = await Promise.all([
      settlesWithin(s1, 1000),
      settlesWithin(s2, 1000),
      settlesWithin(s3, 1000),
    ]);

    assert.strictEqual(r1, "rejected: no workers available");
    assert.strictEqual(r2, "rejected: no workers available");
    assert.strictEqual(r3, "rejected: no workers available");
    assert.strictEqual(pool.starting, false, "pool must not remain starting");
    pool.shutdown();
  });

  it("8. First request and multiple pending requests on WorkerPool are all rejected when worker exits before ready", async () => {
    bridge = require("../src/services/pythonBridge");
    const pool = new bridge.WorkerPool(1);

    const startPromise = pool.start();
    const req1 = pool.execute("chat", { prompt: "first" });
    const req2 = pool.execute("chat", { prompt: "second" });
    const req3 = pool.execute("chat", { prompt: "third" });

    await tick();
    spawnedProcesses[0].simulateExit(1);

    const [startRes, res1, res2, res3] = await Promise.all([
      settlesWithin(startPromise, 1000),
      settlesWithin(req1, 1000),
      settlesWithin(req2, 1000),
      settlesWithin(req3, 1000),
    ]);

    assert.strictEqual(startRes, "rejected: no workers available");
    assert.strictEqual(res1, "rejected: no workers available");
    assert.strictEqual(res2, "rejected: no workers available");
    assert.strictEqual(res3, "rejected: no workers available");
    assert.strictEqual(pool.queue.length, 0, "queue must be emptied");
    assert.strictEqual(pool.starting, false);
    pool.shutdown();
  });

  it("9. Bridge-level concurrent start() and request() callers all settle promptly and do not hang", async () => {
    bridge = require("../src/services/pythonBridge");

    const start1 = bridge.start();
    const start2 = bridge.start();
    const req1 = bridge.request("chat", { prompt: "req1" });
    const req2 = bridge.request("chat", { prompt: "req2" });

    await tick();
    spawnedProcesses[0].simulateExit(1);

    const [s1, s2, r1, r2] = await Promise.all([
      settlesWithin(start1, 1000),
      settlesWithin(start2, 1000),
      settlesWithin(req1, 1000),
      settlesWithin(req2, 1000),
    ]);

    assert.strictEqual(s1, false, "start1 must resolve false");
    assert.strictEqual(s2, false, "start2 must resolve false, not true");
    assert.strictEqual(r1, "rejected: no workers available");
    assert.strictEqual(r2, "rejected: no workers available");

    // Subsequent request after startup failure must reject immediately rather than hanging
    const req3 = bridge.request("chat", { prompt: "req3" });
    const r3 = await settlesWithin(req3, 1000);
    assert.strictEqual(r3, "rejected: model disabled");
  });

  it("10. Existing restart/recovery behavior remains functional and replaces dead worker", async () => {
    bridge = require("../src/services/pythonBridge");
    const timers = [];
    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        const t = { fn, ms };
        timers.push(t);
        return t;
      },
      () => {}
    );

    try {
      const startPromise = bridge.start();
      await tick();
      spawnedProcesses[0].simulateReady(true);

      const started = await settlesWithin(startPromise, 1000);
      assert.strictEqual(started, true);
      assert.strictEqual(bridge.available(), true);

      // Healthy worker crashes
      spawnedProcesses[0].simulateExit(1);
      await tick();

      // Find restart backoff timer
      const backoff = timers.find((t) => t.ms >= 500 && t.ms <= 8000);
      assert.ok(backoff, "restart backoff should be scheduled");
      backoff.fn();
      await tick();

      assert.strictEqual(spawnedProcesses.length, 2, "replacement worker should spawn");
      spawnedProcesses[1].onStdinWrite = (data) => {
        spawnedProcesses[1].simulateResponse(JSON.parse(data).id, true, { content: "recovered response" });
      };
      spawnedProcesses[1].simulateReady(true);
      await tick();

      assert.strictEqual(bridge.available(), true, "bridge should be available after recovery");
      const reply = await settlesWithin(bridge.request("chat", { prompt: "hello" }), 1000);
      assert.deepStrictEqual(reply, { content: "recovered response" });
    } finally {
      bridge._setTimerImpl(prev.set, prev.clear);
    }
  });
});


// ============================================================================
// Issue #399 Regression Tests: stdin error after the worker is running
// ============================================================================

// A worker that dies after reporting ready (an OOM kill, a crash) or closes its stdin
// leaves the pipe without a reader. The next write to it does not throw: Node reports
// EPIPE afterwards, as an 'error' event on child.stdin, once the try/catch around the
// write has returned. The ChildProcess 'error' listener added for spawn failures (#384)
// is on a different emitter and never sees it, so with nothing listening on the stream
// it was an uncaught exception that took the whole backend down, and every conversation
// held in memory with it.
describe("Issue #399: stdin error after the worker is running", { concurrency: 1 }, () => {
  let bridge = null;
  let savedEnv = null;
  let originalExistsSync = null;

  const MANAGED_ENV = {
    MODEL_ENABLED: "true",
    MODEL_PATH: "/fake/model.pt",
    TOKENIZER_PATH: "/fake/tokenizer",
    MODEL_WORKERS: "1",
    PYTHON_BIN: undefined,
    MODEL_TOOLS: undefined,
    MODEL_CLI_MODE: undefined,
    MODEL_CLI_ROOT: undefined,
    MODEL_TIMEOUT_MS: undefined,
    MODEL_STARTUP_TIMEOUT_MS: undefined,
    MODEL_WORKER_STABILITY_MS: undefined,
    MODEL_KILL_GRACE_MS: undefined,
    MODEL_HEARTBEAT_TIMEOUT_MS: undefined,
    MODEL_HEARTBEAT_STARTUP_GRACE_MS: undefined,
  };

  beforeEach(() => {
    delete require.cache[require.resolve("../src/services/pythonBridge")];
    spawnedProcesses.length = 0;

    savedEnv = {};
    for (const [key, value] of Object.entries(MANAGED_ENV)) {
      savedEnv[key] = process.env[key];
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }

    const fs = require("fs");
    originalExistsSync = fs.existsSync;
    fs.existsSync = (path) => {
      if (path && path.includes("model.pt")) return true;
      return originalExistsSync ? originalExistsSync(path) : false;
    };

    mockSpawn = (command, args, options) => {
      return new MockChildProcess(command, args, options);
    };
  });

  afterEach(() => {
    for (const [key, value] of Object.entries(savedEnv)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }

    if (originalExistsSync) {
      const fs = require("fs");
      fs.existsSync = originalExistsSync;
      originalExistsSync = null;
    }

    mockSpawn = null;
    spawnedProcesses.length = 0;
    if (bridge && bridge._pool && bridge._pool()) {
      try {
        bridge._pool().shutdown();
      } catch (e) {
        // ignore
      }
    }
  });

  // Resolves to "STILL PENDING" instead of hanging the run when a request is never settled.
  const settlesWithin = (promise, ms) => {
    let timer;
    const timeout = new Promise((resolve) => {
      timer = setTimeout(() => resolve("STILL PENDING"), ms);
    });
    return Promise.race([
      promise.then(
        (val) => (val === undefined ? "resolved: undefined" : val),
        (err) => `rejected: ${err.message}`
      ),
      timeout,
    ]).finally(() => clearTimeout(timer));
  };

  const tick = () => new Promise((r) => setImmediate(r));

  // With the timers below unset, restart backoffs (500 ms doubling to 8 s) are the
  // only timers the bridge arms in this range.
  const isBackoff = (timer) => timer.ms >= 500 && timer.ms <= 8000;

  // Records every timer the bridge arms instead of running it, so a test decides which
  // restart backoff fires and when. Startup, liveness and stability timers never fire.
  const recordTimers = () => {
    const armed = [];
    const prev = bridge._setTimerImpl(
      (fn, ms) => {
        const timer = { fn, ms };
        armed.push(timer);
        return timer;
      },
      () => {}
    );
    return { armed, restore: () => bridge._setTimerImpl(prev.set, prev.clear) };
  };

  const epipe = () => Object.assign(new Error("write EPIPE"), { code: "EPIPE", errno: -32, syscall: "write" });

  // Node reports a failed write to a broken pipe afterwards, as an 'error' event on the
  // stdin stream, and not from write() itself. This does the same: the write goes through
  // and the failure follows on the next tick. With no listener emit() rethrows, which in
  // production is an uncaught exception. It is recorded in `escaped` instead, so a missing
  // listener fails the test with an assertion and does not take the test run down.
  const breakPipeOnWrite = (child, escaped, shouldBreak = () => true) => {
    child.onStdinWrite = (data) => {
      if (!shouldBreak(String(data))) return;
      process.nextTick(() => {
        try {
          child.stdin.emit("error", epipe());
        } catch (err) {
          escaped.push(err);
        }
      });
    };
  };

  const startReady = async () => {
    bridge = require("../src/services/pythonBridge");
    const started = bridge.start();
    await tick();
    for (const child of spawnedProcesses) child.simulateReady(true);
    assert.strictEqual(await settlesWithin(started, 1000), true, "the pool should start");
  };

  it("1. every started worker has an 'error' listener on child.stdin", async () => {
    process.env.MODEL_WORKERS = "2";
    await startReady();

    assert.strictEqual(spawnedProcesses.length, 2);
    for (const child of spawnedProcesses) {
      assert.ok(
        child.stdin.listenerCount("error") >= 1,
        "child.stdin has no 'error' listener, so an EPIPE would be an uncaught exception"
      );
    }
  });

  it("2. an EPIPE emitted on stdin is handled instead of thrown", async () => {
    await startReady();

    // With no listener EventEmitter#emit rethrows the error, which in Node is an uncaught exception.
    assert.doesNotThrow(() => spawnedProcesses[0].stdin.emit("error", epipe()));
  });

  it("3. Worker.execute: a failed write settles the request in flight at once, not at the model timeout", async () => {
    await startReady();
    const escaped = [];
    breakPipeOnWrite(spawnedProcesses[0], escaped);

    // The model timeout is three minutes, so a request left to it shows as STILL PENDING here.
    const request = settlesWithin(bridge.request("chat", { prompt: "hello" }), 2000);
    await tick();

    assert.ok(spawnedProcesses[0].lastWrite, "the request should have been written to the worker");
    assert.deepStrictEqual(escaped.map((err) => err.message), [], "the EPIPE escaped as an uncaught exception");
    assert.strictEqual(await request, "rejected: worker exited");
  });

  it("4. the failed worker is marked not ready, terminated and taken out of the pool", async () => {
    await startReady();
    const failed = bridge._pool().workers[0];
    const escaped = [];
    breakPipeOnWrite(spawnedProcesses[0], escaped);

    const request = settlesWithin(bridge.request("chat", { prompt: "hello" }), 2000);
    await tick();
    await request;

    assert.deepStrictEqual(escaped.map((err) => err.message), [], "the EPIPE escaped as an uncaught exception");
    assert.strictEqual(failed.ready, false, "a worker whose pipe broke must not stay ready");
    assert.strictEqual(spawnedProcesses[0].killed, true, "a worker that can no longer be written to should be terminated");
    assert.strictEqual(bridge._pool().workers.includes(failed), false, "the exit lifecycle should take it out of the pool");
    assert.strictEqual(bridge.hasAvailableWorker(), false, "nothing is left to dispatch to");
  });

  it("5. no further request is written to the failed worker, and the replacement serves it through the existing restart", async () => {
    bridge = require("../src/services/pythonBridge");
    const timers = recordTimers();
    try {
      await startReady();
      const failedChild = spawnedProcesses[0];
      const escaped = [];
      breakPipeOnWrite(failedChild, escaped);

      const first = settlesWithin(bridge.request("chat", { prompt: "first" }), 2000);
      await tick();
      assert.strictEqual(await first, "rejected: worker exited");
      const firstPayload = failedChild.lastWrite;

      // A request that arrives while the slot restarts finds nothing available and waits.
      const second = settlesWithin(bridge.request("chat", { prompt: "second" }), 5000);
      await tick();
      assert.strictEqual(failedChild.lastWrite, firstPayload, "the failed worker must not be handed another request");

      const backoff = timers.armed.find(isBackoff);
      assert.ok(backoff, "the exit lifecycle should schedule a restart");
      backoff.fn();
      await tick();

      assert.strictEqual(spawnedProcesses.length, 2, "a replacement worker should be spawned");
      assert.ok(spawnedProcesses[1].stdin.listenerCount("error") >= 1, "the replacement must be protected too");
      spawnedProcesses[1].onStdinWrite = (data) => {
        spawnedProcesses[1].simulateResponse(JSON.parse(data).id, true, { content: "from the replacement" });
      };
      spawnedProcesses[1].simulateReady(true);
      await tick();

      assert.deepStrictEqual(await second, { content: "from the replacement" });
      assert.deepStrictEqual(escaped.map((err) => err.message), [], "the EPIPE escaped as an uncaught exception");
    } finally {
      timers.restore();
    }
  });

  it("6. Worker.sendApprovalResponse: a failed write settles the request in flight at once", async () => {
    await startReady();
    const child = spawnedProcesses[0];
    const escaped = [];
    // Only the answer to the approval breaks the pipe; the request itself is written normally.
    breakPipeOnWrite(child, escaped, (data) => data.includes("approval_response"));

    const request = settlesWithin(
      bridge.request("chat", { prompt: "run ls" }, { onApprovalRequest: (info) => info.respond(true) }),
      2000
    );
    await tick();
    child.stdout.emit(
      "data",
      Buffer.from(
        JSON.stringify({
          type: "approval_request",
          approval_id: "app-uuid-1",
          command: "ls -la",
          argv: ["ls", "-la"],
          root: "/sandbox",
        }) + "\n"
      )
    );
    await tick();

    assert.strictEqual(
      JSON.parse(child.lastWrite).type,
      "approval_response",
      "the answer to the approval should be the write that failed"
    );
    assert.deepStrictEqual(escaped.map((err) => err.message), [], "the EPIPE escaped as an uncaught exception");
    assert.strictEqual(await request, "rejected: worker exited");
  });

  it("7. a stdin error that arrives after cleanup() is still handled", async () => {
    // cleanup() runs on the timeout and restart paths. It strips the ChildProcess's
    // listeners, not the stdin stream's, and an EPIPE from a write made just before
    // the kill can still arrive afterwards.
    await startReady();
    const child = spawnedProcesses[0];
    bridge._pool().workers[0].cleanup();

    assert.doesNotThrow(() => child.stdin.emit("error", epipe()));
  });

  it("8. a stdin error that arrives after the worker already exited is still handled", async () => {
    await startReady();
    const child = spawnedProcesses[0];
    child.simulateExit(1);
    await tick();

    assert.doesNotThrow(() => child.stdin.emit("error", epipe()));
  });

  it("9. a stdin error before the worker reports ready fails the startup through the exit handling from #386", async () => {
    bridge = require("../src/services/pythonBridge");
    const timers = recordTimers();
    try {
      const started = bridge.start();
      await tick();

      assert.doesNotThrow(() => spawnedProcesses[0].stdin.emit("error", epipe()));
      assert.strictEqual(await settlesWithin(started, 1000), false, "start() should settle as a failed startup");
    } finally {
      timers.restore();
    }
  });

  it("10. until its exit is processed, the failed worker is not handed another request", async () => {
    // A real child takes a moment to die after kill(), and its 'exit' event arrives later
    // still. The mock exits at once, which would hide that window, so the exit is held back.
    bridge = require("../src/services/pythonBridge");
    const timers = recordTimers();
    try {
      await startReady();
      const failed = bridge._pool().workers[0];
      const child = spawnedProcesses[0];
      child.kill = () => {
        child.killed = true;
      };

      assert.doesNotThrow(() => child.stdin.emit("error", epipe()));
      assert.strictEqual(child.killed, true, "a worker that can no longer be written to should be terminated");
      assert.strictEqual(failed.ready, false, "it must stop being ready when its pipe breaks, not when its exit arrives");
      assert.strictEqual(bridge.hasAvailableWorker(), false, "nothing is left to dispatch to");

      const waiting = settlesWithin(bridge.request("chat", { prompt: "while it is dying" }), 2000);
      await tick();
      assert.strictEqual(child.lastWrite, undefined, "the dying worker must not be handed a request");
      void waiting; // queued; the pool shutdown after the test rejects it
    } finally {
      timers.restore();
    }
  });

  it("11. the stdin listener never throws itself, even when ending the worker fails", async () => {
    // An exception from the listener would be the uncaught exception it exists to prevent.
    await startReady();
    spawnedProcesses[0].kill = () => {
      throw new Error("kill failed");
    };

    assert.doesNotThrow(() => spawnedProcesses[0].stdin.emit("error", epipe()));
    assert.strictEqual(bridge._pool().workers[0].ready, false, "the worker should be out of service all the same");
  });

  it("12. the failure is logged as a warning for that worker, with the error", async () => {
    // The logger reads its level and format when it loads, so load it fresh with known ones.
    const saved = { LOG_LEVEL: process.env.LOG_LEVEL, LOG_FORMAT: process.env.LOG_FORMAT };
    process.env.LOG_LEVEL = "info";
    process.env.LOG_FORMAT = "json";
    delete require.cache[require.resolve("../src/services/logger")];
    delete require.cache[require.resolve("../src/services/pythonBridge")];

    const lines = [];
    const originalWrite = process.stdout.write.bind(process.stdout);
    try {
      await startReady();
      // Nothing is awaited while stdout is replaced, so nothing else can write to it meanwhile.
      process.stdout.write = (chunk) => {
        lines.push(String(chunk));
        return true;
      };
      spawnedProcesses[0].stdin.emit("error", epipe());
    } finally {
      process.stdout.write = originalWrite;
      for (const [key, value] of Object.entries(saved)) {
        if (value === undefined) delete process.env[key];
        else process.env[key] = value;
      }
      delete require.cache[require.resolve("../src/services/logger")];
    }

    const entries = lines
      .join("")
      .split("\n")
      .filter(Boolean)
      .map((line) => {
        try {
          return JSON.parse(line);
        } catch {
          return null;
        }
      })
      .filter(Boolean);
    const entry = entries.find((e) => e.message === "worker stdin error");
    assert.ok(entry, "the failure should be logged");
    assert.strictEqual(entry.level, "warn");
    assert.strictEqual(entry.route, "worker-0");
    assert.match(entry.error, /write EPIPE/);
  });

  // The real thing: a real child, so the pipe and the EPIPE are the operating system's.
  // The child is a Node script that speaks the worker protocol, so no Python or model is
  // needed. A killed worker leaves its stdin pipe without a reader, but the bridge learns
  // of the death only when the child's 'exit' event is processed. A busy event loop (a
  // large JSON.parse, a long GC pause) keeps that event from running first, so the next
  // request is written to a dead pipe.
  const WORKER_SCRIPT = `
    const readline = require('node:readline');
    const out = (obj) => process.stdout.write(JSON.stringify(obj) + '\\n');
    out({ ready: true });
    readline.createInterface({ input: process.stdin }).on('line', (line) => {
      out({ id: JSON.parse(line).id, ok: true, result: { content: 'reply' } });
    });
  `;

  it("13. REAL process: a request written to a killed worker's pipe fails cleanly instead of crashing the backend", async () => {
    mockSpawn = (_command, _args, options) => originalSpawn(process.execPath, ["-e", WORKER_SCRIPT], options);
    bridge = require("../src/services/pythonBridge");

    const warm = await settlesWithin(bridge.request("chat", { prompt: "warm" }), 10000);
    assert.deepStrictEqual(warm, { content: "reply" }, "the real worker should have answered");

    bridge._pool().workers[0].child.kill("SIGKILL");
    // Keep the event loop busy until the child is certainly dead and its exit not yet processed.
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 100);

    const outcome = await settlesWithin(bridge.request("chat", { prompt: "after the kill" }), 5000);
    assert.strictEqual(outcome, "rejected: worker exited");
  });
});
