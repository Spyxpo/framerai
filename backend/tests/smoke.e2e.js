/**
 * End-to-end smoke test suite for the FramerAI backend server process.
 *
 * Boots the actual Node backend process as a child process on an ephemeral port
 * and exercises HTTP endpoints, validation/error paths, rate limiting, and real
 * WebSocket connections.
 */

const { describe, it } = require("node:test");
const assert = require("node:assert");
const { spawn } = require("node:child_process");
const http = require("node:http");
const path = require("node:path");
const fs = require("node:fs");
const WebSocket = require("ws");

const BACKEND_DIR = path.join(__dirname, "..");
const INDEX_PATH = path.join(BACKEND_DIR, "src", "index.js");

/**
 * Spawn the backend Node process on an ephemeral port.
 */
function spawnBackend(envOverrides = {}) {
  return new Promise((resolve, reject) => {
    const env = {
      ...process.env,
      PORT: "0",
      MODEL_ENABLED: "false",
      RATE_LIMIT_WINDOW_MS: "60000",
      GENERATE_RATE_LIMIT_MAX: "10",
      RATE_LIMIT_MAX: "300",
      ...envOverrides,
    };

    const child = spawn(process.execPath, [INDEX_PATH], {
      cwd: BACKEND_DIR,
      env,
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    let resolved = false;
    let boundPort = null;

    const onData = (data) => {
      const str = data.toString();
      stdout += str;

      const match = stdout.match(/http:\/\/localhost:(\d+)/);
      if (match && !resolved) {
        resolved = true;
        boundPort = Number(match[1]);
        resolve({
          child,
          port: boundPort,
          baseUrl: `http://127.0.0.1:${boundPort}`,
          wsUrl: `ws://127.0.0.1:${boundPort}/ws`,
          getStdout: () => stdout,
          getStderr: () => stderr,
          stop: () =>
            new Promise((res) => {
              if (child.killed || child.exitCode !== null) {
                res({ code: child.exitCode, signal: child.signalCode });
                return;
              }
              child.once("exit", (code, signal) => {
                res({ code, signal });
              });
              child.kill("SIGTERM");
            }),
        });
      }
    };

    const onErrorData = (data) => {
      stderr += data.toString();
    };

    child.stdout.on("data", onData);
    child.stderr.on("data", onErrorData);

    child.on("error", (err) => {
      if (!resolved) {
        resolved = true;
        reject(err);
      }
    });

    child.on("exit", (code, signal) => {
      if (!resolved) {
        resolved = true;
        reject(new Error(`Server exited prematurely (code: ${code}, signal: ${signal})\nStderr: ${stderr}`));
      }
    });
  });
}

/**
 * Helper to make HTTP requests against the running server process.
 */
function requestJson(url, options = {}, body = null) {
  return new Promise((resolve, reject) => {
    const parsedUrl = new URL(url);
    const reqOptions = {
      hostname: parsedUrl.hostname,
      port: parsedUrl.port,
      path: parsedUrl.pathname + parsedUrl.search,
      method: options.method || "GET",
      headers: {
        "Content-Type": "application/json",
        Connection: "close",
        ...(options.headers || {}),
      },
    };

    const req = http.request(reqOptions, (res) => {
      let data = "";
      res.on("data", (chunk) => (data += chunk));
      res.on("end", () => {
        let json = null;
        try {
          json = JSON.parse(data);
        } catch (_) {
          json = data;
        }
        resolve({ status: res.statusCode, headers: res.headers, body: json });
      });
    });

    req.on("error", reject);

    if (body) {
      req.write(typeof body === "string" ? body : JSON.stringify(body));
    }
    req.end();
  });
}

describe("E2E Smoke Tests (Process-Level)", () => {
  describe("Tier 1: Fast Smoke Suite (MODEL_ENABLED=false)", () => {
    it("boots backend process, exercises HTTP/WS endpoints, rate limits, error handling, and exits cleanly on SIGTERM", async () => {
      const serverInstance = await spawnBackend({
        MODEL_ENABLED: "false",
        GENERATE_RATE_LIMIT_MAX: "6",
      });

      const { baseUrl, wsUrl, stop, getStdout, getStderr } = serverInstance;

      try {
        // 1. Health check endpoint
        const healthRes = await requestJson(`${baseUrl}/api/health`);
        assert.strictEqual(healthRes.status, 200, "health endpoint should return 200");
        assert.strictEqual(healthRes.body.status, "ok", "health status should be ok");
        assert.strictEqual(healthRes.body.model, "FramerAI", "health model should be FramerAI");

        // 2. OpenAPI spec endpoint
        const openApiRes = await requestJson(`${baseUrl}/api/openapi.json`);
        assert.strictEqual(openApiRes.status, 200, "openapi endpoint should return 200");
        assert.ok(openApiRes.body.openapi, "openapi spec should contain openapi version");

        // 3. Validation & Error Handling
        // Bad request / empty prompt validation failure
        const badReqRes = await requestJson(
          `${baseUrl}/api/generate/image`,
          { method: "POST" },
          { prompt: "" }
        );
        assert.strictEqual(badReqRes.status, 400, "empty prompt should return 400 Bad Request");
        assert.ok(badReqRes.body.error, "error response should contain error message");

        // 404 Not Found
        const notFoundRes = await requestJson(`${baseUrl}/api/nonexistent`);
        assert.strictEqual(notFoundRes.status, 404, "unknown route should return 404");

        // 4. Chat workflow
        const createConvRes = await requestJson(`${baseUrl}/api/chat/conversations`, { method: "POST" });
        assert.strictEqual(createConvRes.status, 200, "create conversation should return 200");
        const convId = createConvRes.body.id;
        assert.ok(convId, "conversation ID should be defined");

        const msgRes = await requestJson(
          `${baseUrl}/api/chat/conversations/${convId}/messages`,
          { method: "POST" },
          { content: "Hello FramerAI", type: "text" }
        );
        assert.strictEqual(msgRes.status, 200, "chat message endpoint should return 200");
        assert.strictEqual(msgRes.body.role, "assistant", "response role should be assistant");

        const getConvRes = await requestJson(`${baseUrl}/api/chat/conversations/${convId}`);
        assert.strictEqual(getConvRes.status, 200, "get conversation should return 200");

        // 5. Generation endpoints
        const imgRes = await requestJson(
          `${baseUrl}/api/generate/image`,
          { method: "POST" },
          { prompt: "a cute cat" }
        );
        assert.strictEqual(imgRes.status, 200, "generate image endpoint should return 200");
        assert.ok(imgRes.body.id, "image response should have ID");

        const codeRes = await requestJson(
          `${baseUrl}/api/generate/code`,
          { method: "POST" },
          { prompt: "fibonacci", language: "python" }
        );
        assert.strictEqual(codeRes.status, 200, "generate code endpoint should return 200");

        // Cleanup conversation
        const delConvRes = await requestJson(`${baseUrl}/api/chat/conversations/${convId}`, { method: "DELETE" });
        assert.strictEqual(delConvRes.status, 200, "delete conversation should return 200");

        // 6. WebSocket session over real TCP socket BEFORE rate limit is exhausted
        const wsMessages = [];
        const ws = new WebSocket(wsUrl);

        await new Promise((res, rej) => {
          ws.on("open", res);
          ws.on("error", rej);
        });

        ws.on("message", (data) => {
          wsMessages.push(JSON.parse(data.toString()));
        });

        // Test ping/pong
        ws.send(JSON.stringify({ type: "ping" }));
        await new Promise((r) => setTimeout(r, 100));

        const pong = wsMessages.find((m) => m.type === "pong");
        assert.ok(pong, "WebSocket should respond to ping with pong");

        // Test WS chat streaming (send prompt "hello" for short response)
        const wsChatId = "ws-smoke-conv-1";
        ws.send(JSON.stringify({ type: "chat", content: "hello", conversationId: wsChatId }));

        let streamDone = false;
        for (let attempt = 0; attempt < 60; attempt++) {
          await new Promise((r) => setTimeout(r, 100));
          if (wsMessages.some((m) => m.type === "stream" && m.conversationId === wsChatId && m.done)) {
            streamDone = true;
            break;
          }
        }
        assert.ok(streamDone, "WebSocket chat should complete stream with done: true");

        ws.close();
        await new Promise((r) => setTimeout(r, 50));

        // 7. Rate Limiting path
        // Exceed the remaining generation limit hits (limit set to 6)
        let rateLimitedRes = null;
        for (let i = 0; i < 7; i++) {
          const res = await requestJson(
            `${baseUrl}/api/generate/image`,
            { method: "POST" },
            { prompt: "test rate limit" }
          );
          if (res.status === 429) {
            rateLimitedRes = res;
            break;
          }
        }
        assert.ok(rateLimitedRes, "should hit 429 rate limit after exceeding max allowed requests");
        assert.strictEqual(rateLimitedRes.status, 429, "rate-limited response status should be 429");

      } finally {
        // 8. Process-level clean shutdown on SIGTERM
        const exitResult = await stop();

        const stderrLog = getStderr();

        assert.strictEqual(
          stderrLog.includes("UnhandledPromiseRejection") || stderrLog.includes("UncaughtException"),
          false,
          `Child process stderr contained unhandled rejection or exception:\n${stderrLog}`
        );

        assert.ok(
          exitResult.code === 0 || exitResult.signal === "SIGTERM",
          `Child process should exit cleanly on SIGTERM (got code ${exitResult.code}, signal ${exitResult.signal})`
        );
      }
    });
  });

  describe("Tier 2: Optional Real Checkpoint Suite (BACKEND_E2E_REAL_MODEL=1)", () => {
    it("runs smoke suite against real model checkpoint if enabled and available", async (t) => {
      const realModelEnabled =
        process.env.BACKEND_E2E_REAL_MODEL === "1" || process.env.BACKEND_E2E_REAL_MODEL === "true";

      if (!realModelEnabled) {
        t.skip("Real model smoke tier is disabled (set BACKEND_E2E_REAL_MODEL=1 to enable)");
        return;
      }

      const modelPath = process.env.MODEL_PATH || path.join(BACKEND_DIR, "..", "checkpoints", "model_final.pt");
      if (!fs.existsSync(modelPath)) {
        t.skip(`Real model checkpoint not found at ${modelPath}`);
        return;
      }

      const serverInstance = await spawnBackend({
        MODEL_ENABLED: "true",
        MODEL_PATH: modelPath,
      });

      const { baseUrl, stop, getStderr } = serverInstance;

      try {
        const healthRes = await requestJson(`${baseUrl}/api/health`);
        assert.strictEqual(healthRes.status, 200);

        const imgRes = await requestJson(
          `${baseUrl}/api/generate/image`,
          { method: "POST" },
          { prompt: "a sunset over mountains" }
        );
        assert.strictEqual(imgRes.status, 200);
        assert.ok(imgRes.body.images);
      } finally {
        const exitResult = await stop();
        assert.ok(exitResult.code === 0 || exitResult.signal === "SIGTERM");
        assert.strictEqual(
          getStderr().includes("UnhandledPromiseRejection"),
          false
        );
      }
    });
  });
});
