const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");
const { randomUUID } = require("node:crypto");
const WebSocket = require("ws");

let resolveMsgA;
let resolveMsgB;
let resolveMsgC;

const { mockModel, loadApp, startServer } = require("./helpers");

mockModel({
  processMessage: (messages) => {
    const last = messages[messages.length - 1]?.content;
    if (last === "msgA") {
      return new Promise((resolve) => {
        resolveMsgA = () => resolve({ type: "text", content: "reply to msgA", metadata: {} });
      });
    }
    if (last === "msgB") {
      return new Promise((resolve) => {
        resolveMsgB = () => resolve({ type: "text", content: "reply to msgB", metadata: {} });
      });
    }
    if (last === "msgC") {
      return new Promise((resolve) => {
        resolveMsgC = () => resolve({ type: "text", content: "reply to msgC", metadata: {} });
      });
    }
    return Promise.resolve({
      type: "text",
      content: `reply to: ${last}`,
      metadata: { model: "test-model" },
    });
  },
});

const app = loadApp();
const conversations = require("../src/conversationStore");
const { generationCounter, apiCounter } = require("../src/middleware/limiters");

test.beforeEach(() => {
  generationCounter.buckets.clear();
  apiCounter.buckets.clear();
  conversations.clear();
});

function connectWebSocket(wsUrl) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    ws.on("open", () => resolve(ws));
    ws.on("error", reject);
  });
}

function collectFramesUntilDone(ws, expectedCount = 1, timeoutMs = 4000) {
  return new Promise((resolve) => {
    const frames = [];
    let doneCount = 0;
    const onMessage = (data) => {
      const msg = JSON.parse(data);
      frames.push(msg);
      if (msg.type === "stream" && msg.done) {
        doneCount++;
        if (doneCount >= expectedCount) {
          cleanup();
          resolve(frames);
        }
      }
    };
    const timer = setTimeout(() => {
      cleanup();
      resolve(frames);
    }, timeoutMs);

    function cleanup() {
      clearTimeout(timer);
      ws.off("message", onMessage);
    }

    ws.on("message", onMessage);
  });
}

function startPostMessage(app, convId, content) {
  return request(app)
    .post(`/api/chat/conversations/${convId}/messages`)
    .send({ content })
    .then((res) => res);
}

test("Test 1 & 7 — Out-of-order completion: Operation A starts first, B starts second; B completes first, A later. Message order is deterministic across in-memory and reload", async () => {
  const created = await request(app).post("/api/chat/conversations");
  const convId = created.body.id;

  // Operation A starts first
  const reqAPromise = startPostMessage(app, convId, "msgA");

  while ((conversations.get(convId)?.messages.length || 0) < 1) {
    await new Promise((r) => setTimeout(r, 5));
  }

  // Operation B starts second
  const reqBPromise = startPostMessage(app, convId, "msgB");

  while ((conversations.get(convId)?.messages.length || 0) < 2) {
    await new Promise((r) => setTimeout(r, 5));
  }

  // Operation B completes FIRST
  assert.ok(resolveMsgB, "resolveMsgB should be ready");
  resolveMsgB();
  const resB = await reqBPromise;
  assert.equal(resB.status, 200);

  // Operation A completes LATER
  assert.ok(resolveMsgA, "resolveMsgA should be ready");
  resolveMsgA();
  const resA = await reqAPromise;
  assert.equal(resA.status, 200);

  // In-memory conversation inspection
  const inMemoryMessages = conversations.get(convId).messages;
  assert.equal(inMemoryMessages.length, 4);

  // Expected causal and chronological sequence:
  // [user msgA, assistant reply to msgA, user msgB, assistant reply to msgB]
  assert.equal(inMemoryMessages[0].role, "user");
  assert.equal(inMemoryMessages[0].content, "msgA");

  assert.equal(inMemoryMessages[1].role, "assistant");
  assert.equal(inMemoryMessages[1].content, "reply to msgA");

  assert.equal(inMemoryMessages[2].role, "user");
  assert.equal(inMemoryMessages[2].content, "msgB");

  assert.equal(inMemoryMessages[3].role, "assistant");
  assert.equal(inMemoryMessages[3].content, "reply to msgB");

  // Reload via GET /api/chat/conversations/:id
  const fetched = await request(app).get(`/api/chat/conversations/${convId}`);
  assert.equal(fetched.status, 200);
  const reloadedMessages = fetched.body.messages;

  assert.equal(reloadedMessages.length, 4);
  assert.equal(reloadedMessages[0].content, "msgA");
  assert.equal(reloadedMessages[1].content, "reply to msgA");
  assert.equal(reloadedMessages[2].content, "msgB");
  assert.equal(reloadedMessages[3].content, "reply to msgB");

  // Timestamps must be monotonic
  const t0 = new Date(reloadedMessages[0].timestamp).getTime();
  const t1 = new Date(reloadedMessages[1].timestamp).getTime();
  const t2 = new Date(reloadedMessages[2].timestamp).getTime();
  const t3 = new Date(reloadedMessages[3].timestamp).getTime();
  assert.ok(t1 >= t0, "Reply A timestamp should be >= prompt A timestamp");
  assert.ok(t2 >= t1, "Prompt B timestamp should be >= reply A timestamp");
  assert.ok(t3 >= t2, "Reply B timestamp should be >= prompt B timestamp");
});

test("Test 2 — Multiple concurrent messages with controlled resolution order (C, then A, then B)", async () => {
  const created = await request(app).post("/api/chat/conversations");
  const convId = created.body.id;

  const reqA = startPostMessage(app, convId, "msgA");
  while ((conversations.get(convId)?.messages.length || 0) < 1) {
    await new Promise((r) => setTimeout(r, 5));
  }

  const reqB = startPostMessage(app, convId, "msgB");
  while ((conversations.get(convId)?.messages.length || 0) < 2) {
    await new Promise((r) => setTimeout(r, 5));
  }

  const reqC = startPostMessage(app, convId, "msgC");
  while ((conversations.get(convId)?.messages.length || 0) < 3) {
    await new Promise((r) => setTimeout(r, 5));
  }

  // Resolve order: C first, A second, B last
  resolveMsgC();
  await reqC;

  resolveMsgA();
  await reqA;

  resolveMsgB();
  await reqB;

  const msgs = conversations.get(convId).messages;
  assert.equal(msgs.length, 6);
  assert.equal(msgs[0].content, "msgA");
  assert.equal(msgs[1].content, "reply to msgA");
  assert.equal(msgs[2].content, "msgB");
  assert.equal(msgs[3].content, "reply to msgB");
  assert.equal(msgs[4].content, "msgC");
  assert.equal(msgs[5].content, "reply to msgC");
});

test("Test 3 — Persistence race: older conversation snapshot cannot overwrite newer state in store", () => {
  const convId = randomUUID();
  conversations.create({
    id: convId,
    title: "New Chat",
    messages: [
      { id: "u1", role: "user", content: "hello" },
      { id: "a1", role: "assistant", content: "world" },
    ],
  });

  // Newer message added to store
  conversations.append(convId, { id: "u2", role: "user", content: "next question" });
  conversations.append(convId, { id: "a2", role: "assistant", content: "next answer" });

  assert.equal(conversations.get(convId).messages.length, 4);

  // Stale snapshot arriving late
  const staleSnapshot = {
    id: convId,
    title: "New Chat",
    messages: [
      { id: "u1", role: "user", content: "hello" },
      { id: "a1", role: "assistant", content: "world" },
    ],
  };

  conversations.create(staleSnapshot);

  // The store must keep the 4 messages, not be regressed to 2 messages
  assert.equal(
    conversations.get(convId).messages.length,
    4,
    "Stale snapshot must not overwrite newer conversation state"
  );
  assert.equal(conversations.get(convId).messages[2].id, "u2");
  assert.equal(conversations.get(convId).messages[3].id, "a2");
});

test("Test 4 — WebSocket / REST interaction: interleaved requests maintain correct causal order", async () => {
  const server = await startServer();
  const convId = randomUUID();
  conversations.create({ id: convId, title: "Interleaved Chat", messages: [] });

  let ws;
  try {
    ws = await connectWebSocket(server.wsUrl);

    // 1. Send msgA via REST
    const reqAPromise = startPostMessage(server.app, convId, "msgA");
    while ((conversations.get(convId)?.messages.length || 0) < 1) {
      await new Promise((r) => setTimeout(r, 5));
    }

    // 2. Send msgB via WebSocket
    ws.send(JSON.stringify({ type: "chat", conversationId: convId, content: "msgB" }));
    while ((conversations.get(convId)?.messages.length || 0) < 2) {
      await new Promise((r) => setTimeout(r, 5));
    }

    // 3. Resolve WebSocket turn (msgB) FIRST
    assert.ok(resolveMsgB, "resolveMsgB should be ready");
    resolveMsgB();

    // 4. Resolve REST turn (msgA) LATER
    assert.ok(resolveMsgA, "resolveMsgA should be ready");
    resolveMsgA();

    await reqAPromise;
    while ((conversations.get(convId)?.messages.length || 0) < 4) {
      await new Promise((r) => setTimeout(r, 5));
    }

    const conv = conversations.get(convId);
    assert.equal(conv.messages.length, 4);
    assert.equal(conv.messages[0].content, "msgA");
    assert.equal(conv.messages[1].content, "reply to msgA");
    assert.equal(conv.messages[2].content, "msgB");
    assert.equal(conv.messages[3].content, "reply to msgB");
  } finally {
    if (ws) {
      try { ws.close(); } catch (_) {}
    }
    await server.stop();
  }
});

test("Test 6 — Branching during pending message operations retains correct ordering and isolation", async () => {
  const created = await request(app).post("/api/chat/conversations");
  const parentId = created.body.id;

  // Add an initial turn
  await request(app)
    .post(`/api/chat/conversations/${parentId}/messages`)
    .send({ content: "initial prompt" });

  const parentInitial = conversations.get(parentId);
  assert.equal(parentInitial.messages.length, 2);
  const initialUserMsgId = parentInitial.messages[0].id;

  // Start pending operation in parent
  const reqAPromise = startPostMessage(app, parentId, "msgA");
  while ((conversations.get(parentId)?.messages.length || 0) < 3) {
    await new Promise((r) => setTimeout(r, 5));
  }

  // Branch from the initial message while msgA is pending
  const branchRes = await request(app)
    .post(`/api/chat/conversations/${parentId}/branch`)
    .send({ messageId: initialUserMsgId });

  assert.equal(branchRes.status, 200);
  const branchId = branchRes.body.id;

  // Resolve pending operation in parent
  resolveMsgA();
  await reqAPromise;

  // Parent must have [initial, reply, msgA, reply to msgA]
  const parentMessages = conversations.get(parentId).messages;
  assert.equal(parentMessages.length, 4);
  assert.equal(parentMessages[2].content, "msgA");
  assert.equal(parentMessages[3].content, "reply to msgA");

  // Branch must only have [initial] and not be polluted by msgA
  const branchMessages = conversations.get(branchId).messages;
  assert.equal(branchMessages.length, 1);
  assert.equal(branchMessages[0].id, initialUserMsgId);
});
