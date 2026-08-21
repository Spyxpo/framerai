/**
 * Python inference bridge.
 *
 * Lazily spawns the FramerAI inference worker (model/serve.py) and talks to it
 * over newline-delimited JSON on stdin/stdout. When the model is disabled, the
 * checkpoint is missing, or the worker fails, callers fall back to placeholder
 * responses - so the backend always runs, with or without trained weights.
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
// Toolsets the worker may register, for example MODEL_TOOLS=web. Empty means
// the worker starts with no tools at all, which is the default.
const MODEL_TOOLS = (process.env.MODEL_TOOLS || "").trim();

const GENERATED_DIR = path.join(BACKEND_ROOT, "uploads", "generated");

let child = null;
let ready = null; // Promise<boolean>
let disabled = false;
let buffer = "";
let nextId = 1;
const pending = new Map();

function isConfigured() {
  return MODEL_ENABLED && MODEL_PATH && fs.existsSync(MODEL_PATH);
}

function start() {
  if (ready) return ready;

  ready = new Promise((resolve) => {
    if (!isConfigured()) {
      disabled = true;
      resolve(false);
      return;
    }

    fs.mkdirSync(GENERATED_DIR, { recursive: true });

    const argv = ["-m", "model.serve", "--model", MODEL_PATH, "--tokenizer", TOKENIZER_PATH];
    if (MODEL_TOOLS) argv.push("--tools", MODEL_TOOLS);

    try {
      child = spawn(PYTHON_BIN, argv, { cwd: REPO_ROOT });
    } catch (err) {
      console.warn(`[model] worker spawn failed: ${err.message}. Using placeholder responses.`);
      disabled = true;
      resolve(false);
      return;
    }

    let resolved = false;

    child.stdout.on("data", (data) => {
      buffer += data.toString();
      let idx;
      while ((idx = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line) continue;
        let msg;
        try {
          msg = JSON.parse(line);
        } catch {
          continue;
        }
        if (!resolved && Object.prototype.hasOwnProperty.call(msg, "ready")) {
          resolved = true;
          if (msg.ready) {
            console.log("[model] inference worker ready");
            resolve(true);
          } else {
            console.warn(`[model] worker failed to load: ${msg.error}. Using placeholder responses.`);
            disabled = true;
            resolve(false);
          }
          continue;
        }
        if (msg.id != null && pending.has(msg.id)) {
          const { resolve: res, reject, timer } = pending.get(msg.id);
          clearTimeout(timer);
          pending.delete(msg.id);
          if (msg.ok) res(msg.result);
          else reject(new Error(msg.error || "inference failed"));
        }
      }
    });

    child.stderr.on("data", (d) => process.stderr.write(`[model:worker] ${d}`));

    child.on("exit", (code) => {
      console.warn(`[model] worker exited (code ${code}). Falling back to placeholder responses.`);
      disabled = true;
      child = null;
      for (const { reject, timer } of pending.values()) {
        clearTimeout(timer);
        reject(new Error("worker exited"));
      }
      pending.clear();
    });

    if (!resolved) {
      // Safety net: if the worker never reports readiness, give up.
      setTimeout(() => {
        if (!resolved) {
          resolved = true;
          disabled = true;
          resolve(false);
        }
      }, 60000);
    }
  });

  return ready;
}

async function request(op, params = {}) {
  if (disabled) throw new Error("model disabled");
  const ok = await start();
  if (!ok || !child) throw new Error("model unavailable");

  const id = nextId++;
  const payload = { id, op, params: { out_dir: GENERATED_DIR, ...params } };

  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error("inference timed out"));
    }, REQUEST_TIMEOUT_MS);
    pending.set(id, { resolve, reject, timer });
    child.stdin.write(JSON.stringify(payload) + "\n");
  });
}

function available() {
  return isConfigured() && !disabled;
}

module.exports = { request, available, start, GENERATED_DIR };
