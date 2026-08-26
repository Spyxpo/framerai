/**
 * Python inference bridge with worker pooling.
 *
 * Spawns a pool of FramerAI inference workers (model/serve.py) and distributes
 * requests across them to avoid head-of-line blocking. Each worker loads the
 * model once and reuses it for multiple requests. When workers exit, they are
 * automatically replaced.
 */

const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");

const REPO_ROOT = path.join(__dirname, "..", "..", "..");
const BACKEND_ROOT = path.join(__dirname, "..", "..");

function resolvePath(p, base) {
  if (!p) return null;
  return path.isAbsolute(p) ? p : path.resolve(base, p);
}

const MODEL_PATH = resolvePath(process.env.MODEL_PATH, BACKEND_ROOT);
const TOKENIZER_PATH = resolvePath(process.env.TOKENIZER_PATH, BACKEND_ROOT);
const PYTHON_BIN = process.env.PYTHON_BIN || "python3";
const MODEL_ENABLED = (process.env.MODEL_ENABLED || "true").toLowerCase() !== "false";
const REQUEST_TIMEOUT_MS = Number(process.env.MODEL_TIMEOUT_MS || 180000);
const STARTUP_TIMEOUT_MS = Number(process.env.MODEL_STARTUP_TIMEOUT_MS || 60000);
const WORKER_COUNT = Number(process.env.MODEL_WORKERS || 2);
// Toolsets the worker may register, for example MODEL_TOOLS=web. Empty means
// the worker starts with no tools at all, which is the default.
const MODEL_TOOLS = (process.env.MODEL_TOOLS || "").trim();
// How the cli toolset decides, when it is registered at all. Left at "off" the
// worker refuses every command, which is the default the model ships with.
const MODEL_CLI_MODE = (process.env.MODEL_CLI_MODE || "off").trim();
const MODEL_CLI_ROOT = resolvePath(process.env.MODEL_CLI_ROOT, BACKEND_ROOT) || REPO_ROOT;

const GENERATED_DIR = path.join(BACKEND_ROOT, "uploads", "generated");

// Restart policy
const MAX_RESTART_ATTEMPTS = 5;
// Backoff delays in ms: 500, 1000, 2000, 4000, 8000 (capped at 8s)
const RESTART_BACKOFF_BASE_MS = 500;
const RESTART_BACKOFF_MAX_MS = 8000;
const WORKER_STABILITY_MS = Number(process.env.MODEL_WORKER_STABILITY_MS || 30000);

// Replaceable timer for testing
let _setTimeout = (fn, ms) => setTimeout(fn, ms);
let _clearTimeout = (id) => clearTimeout(id);

// Swap both timer functions at once; returns the previous impl for teardown.
function _setTimerImpl(setFn, clearFn) {
  const prev = { set: _setTimeout, clear: _clearTimeout };
  _setTimeout = setFn;
  _clearTimeout = clearFn;
  return prev;
}

// Worker state
class Worker {
  constructor(id) {
    this.id = id;
    this.child = null;
    this.ready = false;
    this.busy = false;
    this.currentRequest = null;
    this.buffer = "";
    this.pending = new Map();
    this.nextId = 1;
    this._safetyTimer = null;
    this.onSuccess = null;
  }

  spawn() {
    return new Promise((resolve) => {
      const argv = ["-m", "model.serve", "--model", MODEL_PATH, "--tokenizer", TOKENIZER_PATH];
      if (MODEL_TOOLS) argv.push("--tools", MODEL_TOOLS);
      if (MODEL_TOOLS.includes("cli")) {
        argv.push("--cli-mode", MODEL_CLI_MODE, "--cli-root", MODEL_CLI_ROOT);
      }

      try {
        this.child = spawn(PYTHON_BIN, argv, { cwd: REPO_ROOT });
      } catch (err) {
        console.warn(`[model:worker-${this.id}] spawn failed: ${err.message}`);
        resolve(false);
        return;
      }

      let resolved = false;

      this.child.stdout.on("data", (data) => {
        this.buffer += data.toString();
        let idx;
        while ((idx = this.buffer.indexOf("\n")) >= 0) {
          const line = this.buffer.slice(0, idx).trim();
          this.buffer = this.buffer.slice(idx + 1);
          if (!line) continue;
          let msg;
          try {
            msg = JSON.parse(line);
          } catch {
            continue;
          }
          if (!resolved && Object.prototype.hasOwnProperty.call(msg, "ready")) {
            resolved = true;
            // Clear safety timeout on success or failure
            if (this._safetyTimer) {
              _clearTimeout(this._safetyTimer);
              this._safetyTimer = null;
            }
            if (msg.ready) {
              console.log(`[model:worker-${this.id}] ready`);
              this.ready = true;
              resolve(true);
            } else {
              console.warn(`[model:worker-${this.id}] failed to load: ${msg.error}`);
              resolve(false);
            }
            continue;
          }
          if (msg.id != null && this.pending.has(msg.id)) {
            const { resolve: res, reject, timer } = this.pending.get(msg.id);
            clearTimeout(timer);
            this.pending.delete(msg.id);
            this.busy = false;
            this.currentRequest = null;
            if (msg.ok) {
              if (this.onSuccess) this.onSuccess(this);
              res(msg.result);
            } else {
              reject(new Error(msg.error || "inference failed"));
            }
            // Notify pool that worker is available
            if (this.onAvailable) this.onAvailable(this);
          }
        }
      });

      this.child.stderr.on("data", (d) => process.stderr.write(`[model:worker-${this.id}] ${d}`));

      this.child.on("exit", (code) => {
        console.warn(`[model:worker-${this.id}] exited (code ${code})`);
        this.ready = false;
        this.child = null;
        // Clear safety timeout if still pending
        if (this._safetyTimer) {
          _clearTimeout(this._safetyTimer);
          this._safetyTimer = null;
        }
        // Reject current request if any
        if (this.currentRequest) {
          const { reject, timer } = this.currentRequest;
          clearTimeout(timer);
          reject(new Error("worker exited"));
          this.currentRequest = null;
        }
        // Reject all pending
        for (const { reject, timer } of this.pending.values()) {
          clearTimeout(timer);
          reject(new Error("worker exited"));
        }
        this.pending.clear();
        this.busy = false;
        // Notify pool of worker death
        if (this.onExit) this.onExit(this);
      });

      // Safety timeout - stored on instance so cleanup() can cancel it
      this._safetyTimer = _setTimeout(() => {
        if (!resolved) {
          resolved = true;
          this._safetyTimer = null;
          console.warn(`[model:worker-${this.id}] startup timeout after ${STARTUP_TIMEOUT_MS}ms, cleaning up`);
          this.cleanup();
          resolve(false);
        }
      }, STARTUP_TIMEOUT_MS);
    });
  }

  async execute(op, params, timeoutMs = REQUEST_TIMEOUT_MS) {
    if (!this.ready || !this.child) {
      throw new Error("worker not ready");
    }
    if (this.busy) {
      throw new Error("worker busy");
    }

    this.busy = true;
    const id = this.nextId++;
    const payload = { id, op, params: { out_dir: GENERATED_DIR, ...params } };

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        this.busy = false;
        this.currentRequest = null;
        reject(new Error("inference timed out"));
        // Notify pool
        if (this.onAvailable) this.onAvailable(this);
      }, timeoutMs);

      const requestState = { resolve, reject, timer };
      this.pending.set(id, requestState);
      this.currentRequest = requestState;

      try {
        this.child.stdin.write(JSON.stringify(payload) + "\n");
      } catch (err) {
        clearTimeout(timer);
        this.pending.delete(id);
        this.busy = false;
        this.currentRequest = null;
        reject(err);
        if (this.onAvailable) this.onAvailable(this);
      }
    });
  }

  cleanup() {
    if (this._safetyTimer) {
      _clearTimeout(this._safetyTimer);
      this._safetyTimer = null;
    }
    if (this.child) {
      this.child.removeAllListeners();
      this.child.kill();
      this.child = null;
    }
    for (const { reject, timer } of this.pending.values()) {
      clearTimeout(timer);
      reject(new Error("worker cleanup"));
    }
    this.pending.clear();
    this.ready = false;
    this.busy = false;
    this.currentRequest = null;
  }
}

// Pool state
let pool = null;
let disabled = false;
let poolInitialized = false;

class WorkerPool {
  constructor(size) {
    this.workers = [];
    this.queue = [];
    this.size = size;
    this.starting = false;
    this.stopped = false;
    // Track consecutive restart attempts per worker slot (by id)
    this._restartCounts = new Map();
    this._restartTimers = new Map();
    this._stabilityTimers = new Map();
  }

  markWorkerStable(workerId) {
    this._restartCounts.delete(workerId);
    if (this._stabilityTimers.has(workerId)) {
      _clearTimeout(this._stabilityTimers.get(workerId));
      this._stabilityTimers.delete(workerId);
    }
  }

  scheduleStabilityTimer(workerId) {
    if (this._stabilityTimers.has(workerId)) {
      _clearTimeout(this._stabilityTimers.get(workerId));
    }
    const timer = _setTimeout(() => {
      this._stabilityTimers.delete(workerId);
      this.markWorkerStable(workerId);
      console.log(`[model] worker ${workerId} considered stable after ${WORKER_STABILITY_MS}ms`);
    }, WORKER_STABILITY_MS);
    this._stabilityTimers.set(workerId, timer);
  }

  async start() {
    if (this.starting) return;
    this.starting = true;

    fs.mkdirSync(GENERATED_DIR, { recursive: true });

    const startPromises = [];
    for (let i = 0; i < this.size; i++) {
      const worker = new Worker(i);
      worker.onAvailable = () => this.dispatch();
      worker.onExit = (w) => this.handleWorkerExit(w);
      worker.onSuccess = (w) => this.markWorkerStable(w.id);
      this.workers.push(worker);
      startPromises.push(worker.spawn());
    }

    const results = await Promise.all(startPromises);
    const successCount = results.filter((r) => r).length;

    if (successCount === 0) {
      console.warn("[model] no workers started successfully");
      disabled = true;
      throw new Error("no workers available");
    }

    for (let i = 0; i < this.size; i++) {
      if (results[i]) {
        this.scheduleStabilityTimer(i);
      }
    }

    console.log(`[model] started ${successCount}/${this.size} workers`);
    this.dispatch();
  }

  async handleWorkerExit(deadWorker) {
    if (this.stopped) return;
    const workerId = deadWorker.id;
    if (this._stabilityTimers.has(workerId)) {
      _clearTimeout(this._stabilityTimers.get(workerId));
      this._stabilityTimers.delete(workerId);
    }

    // Remove dead worker from the live set
    const idx = this.workers.indexOf(deadWorker);
    if (idx >= 0) {
      this.workers.splice(idx, 1);
    }
    deadWorker.cleanup();

    const attempts = (this._restartCounts.get(workerId) || 0) + 1;

    // Hard cap: abandon this worker slot after too many consecutive failures
    if (attempts > MAX_RESTART_ATTEMPTS) {
      console.warn(
        `[model] worker ${workerId} has failed ${MAX_RESTART_ATTEMPTS} restart attempts, giving up`
      );
      if (this.workers.filter((w) => w.ready).length === 0) {
        disabled = true;
      }
      return;
    }

    this._restartCounts.set(workerId, attempts);

    // Exponential backoff, capped at RESTART_BACKOFF_MAX_MS
    const delay = Math.min(
      RESTART_BACKOFF_BASE_MS * Math.pow(2, attempts - 1),
      RESTART_BACKOFF_MAX_MS
    );
    console.log(
      `[model] worker ${workerId} restart attempt ${attempts}/${MAX_RESTART_ATTEMPTS} in ${delay}ms`
    );

    // Wait out the backoff (interruptible by shutdown)
    await new Promise((resolve) => {
      const timer = _setTimeout(resolve, delay);
      this._restartTimers.set(workerId, timer);
    });
    this._restartTimers.delete(workerId);

    if (this.stopped) return;

    // Spawn replacement
    console.log(`[model] spawning replacement worker ${workerId}`);
    const replacement = new Worker(workerId);
    replacement.onAvailable = () => this.dispatch();
    replacement.onExit = (w) => this.handleWorkerExit(w);
    replacement.onSuccess = (w) => this.markWorkerStable(w.id);
    this.workers.push(replacement);

    const success = await replacement.spawn();
    if (this.stopped) {
      replacement.cleanup();
      return;
    }
    if (success) {
      disabled = false;
      this.scheduleStabilityTimer(workerId);
      console.log(`[model] replacement worker ${workerId} ready`);
      this.dispatch();
    } else {
      console.warn(`[model] replacement worker ${workerId} failed to start`);
      if (this.workers.filter((w) => w.ready).length === 0) {
        disabled = true;
      }
    }
  }

  getAvailableWorker() {
    return this.workers.find((w) => w.ready && !w.busy);
  }

  dispatch() {
    while (this.queue.length > 0) {
      const worker = this.getAvailableWorker();
      if (!worker) break;

      const { op, params, resolve, reject, timer, deadline } = this.queue.shift();
      clearTimeout(timer);

      // Calculate remaining time from original deadline
      const remainingMs = deadline - Date.now();
      if (remainingMs <= 0) {
        // Already past deadline while in queue
        reject(new Error("queued request timed out"));
        continue;
      }

      worker
        .execute(op, params, remainingMs)
        .then(resolve)
        .catch(reject);
    }
  }

  async execute(op, params) {
    const worker = this.getAvailableWorker();
    if (worker) {
      return worker.execute(op, params, REQUEST_TIMEOUT_MS);
    }

    // All workers busy, queue the request with absolute deadline
    return new Promise((resolve, reject) => {
      const deadline = Date.now() + REQUEST_TIMEOUT_MS;

      const timer = setTimeout(() => {
        // Remove from queue
        const idx = this.queue.findIndex((item) => item.timer === timer);
        if (idx >= 0) {
          this.queue.splice(idx, 1);
        }
        reject(new Error("queued request timed out"));
      }, REQUEST_TIMEOUT_MS);

      this.queue.push({ op, params, resolve, reject, timer, deadline });
    });
  }

  shutdown() {
    this.stopped = true;
    // Cancel any pending backoff timers so restart loops stop immediately
    for (const timer of this._restartTimers.values()) {
      _clearTimeout(timer);
    }
    this._restartTimers.clear();
    for (const timer of this._stabilityTimers.values()) {
      _clearTimeout(timer);
    }
    this._stabilityTimers.clear();
    for (const worker of this.workers) {
      worker.cleanup();
    }
    this.workers = [];
    // Clear all queued request timeouts and explicitly reject them
    for (const { reject, timer } of this.queue) {
      clearTimeout(timer);
      reject(new Error("pool shutdown"));
    }
    this.queue = [];
  }
}

function isConfigured() {
  return MODEL_ENABLED && MODEL_PATH && fs.existsSync(MODEL_PATH);
}

async function ensurePool() {
  if (poolInitialized) return pool;
  if (disabled) throw new Error("model disabled");

  if (!isConfigured()) {
    disabled = true;
    throw new Error("model not configured");
  }

  pool = new WorkerPool(WORKER_COUNT);
  poolInitialized = true;

  try {
    await pool.start();
  } catch (err) {
    disabled = true;
    throw err;
  }

  return pool;
}

async function request(op, params = {}) {
  const p = await ensurePool();
  return p.execute(op, params);
}

function available() {
  return isConfigured() && !disabled;
}

async function start() {
  if (poolInitialized) return !disabled;
  try {
    await ensurePool();
    return true;
  } catch {
    return false;
  }
}

module.exports = { request, available, start, GENERATED_DIR, _pool: () => pool, _setTimerImpl };
