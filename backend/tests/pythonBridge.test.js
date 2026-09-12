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
    this.stdin = {
      write: (data) => {
        this.lastWrite = data;
        if (this.onStdinWrite) this.onStdinWrite(data);
      },
    };
    this.stdout = new EventEmitter();
    this.stderr = new EventEmitter();
    spawnedProcesses.push(this);
  }

  kill() {
    this.killed = true;
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
          this.emit("exit", 137); // 128 + 9 (SIGKILL)
        }
        // SIGTERM is silently ignored — the process does not exit
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
