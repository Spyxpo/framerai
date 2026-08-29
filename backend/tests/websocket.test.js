const test = require("node:test");
const assert = require("node:assert/strict");
const WebSocket = require("ws");

const { mockModel, startServer } = require("./helpers");

// Default mock for most tests
mockModel();

/**
 * Open a connection, send one frame, and collect every reply until the stream
 * finishes, an error arrives, or the wait times out.
 */
function exchange(wsUrl, frame, { timeoutMs = 4000 } = {}) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    const received = [];

    const finish = () => {
      clearTimeout(timer);
      ws.close();
      resolve(received);
    };
    const timer = setTimeout(finish, timeoutMs);

    ws.on("open", () => ws.send(JSON.stringify(frame)));
    ws.on("error", reject);
    ws.on("message", (data) => {
      const message = JSON.parse(data);
      received.push(message);
      if (message.type === "error" || (message.type === "stream" && message.done)) finish();
      if (message.type === "pong") finish();
    });
  });
}

test("a chat frame is acknowledged and streamed to completion", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  const messages = await exchange(server.wsUrl, {
    type: "chat",
    content: "hello there",
    conversationId: "abc",
  });

  const types = messages.map((m) => m.type);
  assert.ok(types.includes("ack"), "should acknowledge the frame");
  assert.ok(types.includes("typing"), "should send a typing indicator");

  const streamed = messages.filter((m) => m.type === "stream");
  assert.ok(streamed.length > 0, "should stream at least one chunk");

  const final = streamed[streamed.length - 1];
  assert.equal(final.done, true);
  assert.equal(final.content, "reply to: hello there");
  assert.equal(final.responseType, "text");
  assert.deepEqual(final.metadata, { model: "test-model" });

  // Chunks accumulate rather than replacing each other.
  assert.ok(final.content.startsWith(streamed[0].content));
});

test("chunks arrive in order and each extends the previous one", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  const messages = await exchange(server.wsUrl, {
    type: "chat",
    content: "one two three four",
    conversationId: "abc",
  });

  const streamed = messages.filter((m) => m.type === "stream");
  for (let i = 1; i < streamed.length; i++) {
    assert.ok(
      streamed[i].content.startsWith(streamed[i - 1].content),
      "each chunk should extend the previous one"
    );
  }
});

test("a frame with no content is answered with an error, not a stream", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  const [message] = await exchange(server.wsUrl, { type: "chat", conversationId: "abc" });

  assert.equal(message.type, "error");
  assert.match(message.message, /content is required/);
});

test("an unknown message type is rejected", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  const [message] = await exchange(server.wsUrl, {
    type: "chat",
    content: "hi",
    messageType: "hologram",
  });

  assert.equal(message.type, "error");
  assert.match(message.message, /messageType must be one of/);
});

test("an over-long message is rejected", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  const [message] = await exchange(server.wsUrl, {
    type: "chat",
    content: "x".repeat(8001),
  });

  assert.equal(message.type, "error");
  assert.match(message.message, /at most 8000 characters/);
});

test("ping is answered with pong", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  const [message] = await exchange(server.wsUrl, { type: "ping" });

  assert.equal(message.type, "pong");
});

test("a frame that is not JSON does not take the connection down", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  const ws = new WebSocket(server.wsUrl);
  const first = await new Promise((resolve, reject) => {
    ws.on("error", reject);
    ws.on("open", () => ws.send("definitely not json"));
    ws.on("message", (data) => resolve(JSON.parse(data)));
  });

  assert.equal(first.type, "error");

  // Still usable afterwards.
  const pong = await new Promise((resolve, reject) => {
    ws.on("error", reject);
    ws.on("message", (data) => {
      const message = JSON.parse(data);
      if (message.type === "pong") resolve(message);
    });
    ws.send(JSON.stringify({ type: "ping" }));
  });

  assert.equal(pong.type, "pong");
  ws.close();
});

test("existing text streaming still works after audio changes", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  const messages = await exchange(server.wsUrl, {
    type: "chat",
    content: "hello there",
    conversationId: "text-test",
  });

  const types = messages.map((m) => m.type);
  assert.ok(types.includes("ack"), "should acknowledge");
  assert.ok(types.includes("typing"), "should show typing");

  const streamed = messages.filter((m) => m.type === "stream");
  assert.ok(streamed.length > 0, "should stream text");

  const final = streamed[streamed.length - 1];
  assert.equal(final.done, true);
  assert.equal(final.responseType, "text");
});

test("WebSocket approval request round trip and session isolation", async (t) => {
  let triggerApproval = null;
  mockModel({
    processMessage: (messages, type, settings, options) => {
      if (options && typeof options.onApprovalRequest === "function") {
        triggerApproval = options.onApprovalRequest;
      }
      return new Promise((resolve) => {
        setImmediate(() => {
          resolve({
            type: "text",
            content: "approved response",
            metadata: { model: "test-model" },
          });
        });
      });
    },
  });

  const server = await startServer();
  t.after(() => server.stop());

  const clientA = new WebSocket(server.wsUrl);
  const clientB = new WebSocket(server.wsUrl);

  const messagesA = [];
  const messagesB = [];

  await Promise.all([
    new Promise((r) => clientA.on("open", r)),
    new Promise((r) => clientB.on("open", r)),
  ]);

  clientA.on("message", (data) => messagesA.push(JSON.parse(data)));
  clientB.on("message", (data) => messagesB.push(JSON.parse(data)));

  clientA.send(JSON.stringify({ type: "chat", content: "run ls", conversationId: "conv-A" }));

  await new Promise((r) => setTimeout(r, 50));
  assert.ok(triggerApproval, "triggerApproval callback should be passed");

  let approvalResult = null;
  triggerApproval({
    approvalId: "approval-101",
    command: "ls -la",
    argv: ["ls", "-la"],
    root: "/sandbox",
    respond: (approved) => {
      approvalResult = approved;
    },
  });

  await new Promise((r) => setTimeout(r, 50));

  // Verify approval request frame was delivered to Client A only
  const reqFrameA = messagesA.find((m) => m.type === "approval_request");
  const reqFrameB = messagesB.find((m) => m.type === "approval_request");

  assert.ok(reqFrameA, "Client A should receive approval_request frame");
  assert.equal(reqFrameA.approvalId, "approval-101");
  assert.equal(reqFrameA.command, "ls -la");
  assert.equal(reqFrameB, undefined, "Client B must NOT receive Client A's approval request");

  // Test session isolation: Client B trying to respond to Client A's approval ID must be ignored
  clientB.send(JSON.stringify({ type: "approval_response", approvalId: "approval-101", approved: true }));
  await new Promise((r) => setTimeout(r, 50));
  assert.equal(approvalResult, null, "Client B approval response must be ignored due to session isolation");

  // Client A responds with approval
  clientA.send(JSON.stringify({ type: "approval_response", approvalId: "approval-101", approved: true }));
  await new Promise((r) => setTimeout(r, 50));
  assert.equal(approvalResult, true, "Client A approval response should succeed");

  clientA.close();
  clientB.close();
});

test("WebSocket disconnect fails closed pending approvals", async (t) => {
  let triggerApproval = null;
  mockModel({
    processMessage: (messages, type, settings, options) => {
      if (options && typeof options.onApprovalRequest === "function") {
        triggerApproval = options.onApprovalRequest;
      }
      return Promise.resolve({ type: "text", content: "done", metadata: {} });
    },
  });

  const server = await startServer();
  t.after(() => server.stop());

  const ws = new WebSocket(server.wsUrl);
  await new Promise((r) => ws.on("open", r));

  ws.send(JSON.stringify({ type: "chat", content: "run task", conversationId: "conv-disc" }));
  await new Promise((r) => setTimeout(r, 50));

  let approvalResult = null;
  triggerApproval({
    approvalId: "approval-disc",
    command: "pwd",
    argv: ["pwd"],
    root: "/sandbox",
    respond: (approved) => {
      approvalResult = approved;
    },
  });

  await new Promise((r) => setTimeout(r, 50));
  assert.equal(approvalResult, null);

  ws.close();
  await new Promise((r) => setTimeout(r, 50));
  assert.equal(approvalResult, false, "Disconnect must fail closed returning false");
});

test("WebSocket denyEverything mode automatically denies future requests", async (t) => {
  let triggerApproval = null;
  mockModel({
    processMessage: (messages, type, settings, options) => {
      if (options && typeof options.onApprovalRequest === "function") {
        triggerApproval = options.onApprovalRequest;
      }
      return Promise.resolve({ type: "text", content: "done", metadata: {} });
    },
  });

  const server = await startServer();
  t.after(() => server.stop());

  const ws = new WebSocket(server.wsUrl);
  await new Promise((r) => ws.on("open", r));

  ws.send(JSON.stringify({ type: "chat", content: "cmd 1", conversationId: "c1" }));
  await new Promise((r) => setTimeout(r, 50));

  let firstApproved = null;
  triggerApproval({
    approvalId: "app-1",
    command: "first",
    argv: ["first"],
    root: "/root",
    respond: (app) => {
      firstApproved = app;
    },
  });
  await new Promise((r) => setTimeout(r, 50));

  // User denies and sets denyEverything: true
  ws.send(JSON.stringify({ type: "approval_response", approvalId: "app-1", approved: false, denyEverything: true }));
  await new Promise((r) => setTimeout(r, 50));
  assert.equal(firstApproved, false);

  // Subsequent approval request arrives on same session
  let secondApproved = null;
  triggerApproval({
    approvalId: "app-2",
    command: "second",
    argv: ["second"],
    root: "/root",
    respond: (app) => {
      secondApproved = app;
    },
  });

  await new Promise((r) => setTimeout(r, 50));
  assert.equal(secondApproved, false, "Subsequent request should be auto-denied when denyEverything is active");

  ws.close();
});
