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
    spawnedProcesses = [];

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
    spawnedProcesses = [];
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
});
