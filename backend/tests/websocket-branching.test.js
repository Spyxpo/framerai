const test = require("node:test");
const assert = require("node:assert/strict");
const WebSocket = require("ws");
const { randomUUID } = require("node:crypto");

const { mockModel, startServer } = require("./helpers");

mockModel();
const conversations = require("../src/conversationStore");
const { generationCounter, apiCounter } = require("../src/middleware/limiters");

test.beforeEach(() => {
  generationCounter.buckets.clear();
  apiCounter.buckets.clear();
});

function connectWebSocket(wsUrl) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    ws.on("open", () => resolve(ws));
    ws.on("error", reject);
  });
}

function collectFrames(ws, count, condition, timeoutMs = 4000) {
  return new Promise((resolve) => {
    const frames = [];
    const onMessage = (data) => {
      const msg = JSON.parse(data);
      frames.push(msg);
      if (condition && condition(msg, frames)) {
        cleanup();
        resolve(frames);
      } else if (count && frames.length >= count) {
        cleanup();
        resolve(frames);
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

test("1 & 4. Parent and branch streaming isolation and completion targeting", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  // Create parent conversation in store
  const parentId = randomUUID();
  conversations.create({
    id: parentId,
    title: "Parent Conversation",
    messages: [
      { id: randomUUID(), role: "user", content: "root 1", timestamp: new Date().toISOString() },
      { id: randomUUID(), role: "assistant", content: "reply 1", timestamp: new Date().toISOString() },
    ],
    createdAt: new Date().toISOString(),
  });

  // Create branch conversation in store
  const branchId = randomUUID();
  conversations.create({
    id: branchId,
    title: "Parent Conversation (Branch)",
    parentConversationId: parentId,
    branchedFromMessageId: conversations.get(parentId).messages[0].id,
    messages: [conversations.get(parentId).messages[0]],
    createdAt: new Date().toISOString(),
  });

  const ws = await connectWebSocket(server.wsUrl);
  t.after(() => ws.close());

  // Stream in branch
  const branchFramesPromise = collectFrames(
    ws,
    null,
    (msg) => msg.type === "stream" && msg.done && msg.conversationId === branchId
  );

  ws.send(
    JSON.stringify({
      type: "chat",
      content: "branch prompt",
      conversationId: branchId,
    })
  );

  const branchFrames = await branchFramesPromise;

  // Verify all streaming frames have conversationId === branchId
  const streamFrames = branchFrames.filter((f) => f.type === "stream");
  assert.ok(streamFrames.length > 0, "must receive stream frames for branch");
  for (const f of streamFrames) {
    assert.equal(f.conversationId, branchId, "stream frame must carry branch conversation ID");
  }

  const finalBranchFrame = streamFrames[streamFrames.length - 1];
  assert.equal(finalBranchFrame.done, true);
  assert.equal(finalBranchFrame.conversationId, branchId);

  // 7. Verify parent conversation in conversationStore was NOT mutated by branch generation
  const parentInStore = conversations.get(parentId);
  assert.equal(parentInStore.messages.length, 2, "parent must still have exactly 2 messages");
  assert.ok(!parentInStore.messages.some((m) => m.content === "branch prompt"));

  // Verify branch in store has its messages
  const branchInStore = conversations.get(branchId);
  assert.equal(branchInStore.messages.length, 3, "branch has root 1 + branch prompt + assistant reply");
});

test("2 & 3. Simultaneous generation in parent and branch on same connection", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  const parentId = randomUUID();
  const branchId = randomUUID();

  conversations.create({
    id: parentId,
    title: "Parent",
    messages: [],
    createdAt: new Date().toISOString(),
  });

  conversations.create({
    id: branchId,
    title: "Branch",
    parentConversationId: parentId,
    messages: [],
    createdAt: new Date().toISOString(),
  });

  const ws = await connectWebSocket(server.wsUrl);
  t.after(() => ws.close());

  let parentDone = false;
  let branchDone = false;

  const allFramesPromise = collectFrames(
    ws,
    null,
    (msg) => {
      if (msg.type === "stream" && msg.done) {
        if (msg.conversationId === parentId) parentDone = true;
        if (msg.conversationId === branchId) branchDone = true;
      }
      return parentDone && branchDone;
    },
    6000
  );

  // Send frames for both parent and branch simultaneously
  ws.send(JSON.stringify({ type: "chat", content: "simultaneous parent", conversationId: parentId }));
  ws.send(JSON.stringify({ type: "chat", content: "simultaneous branch", conversationId: branchId }));

  const frames = await allFramesPromise;

  const parentStreamFrames = frames.filter((f) => f.type === "stream" && f.conversationId === parentId);
  const branchStreamFrames = frames.filter((f) => f.type === "stream" && f.conversationId === branchId);

  assert.ok(parentStreamFrames.length > 0, "parent must receive stream frames");
  assert.ok(branchStreamFrames.length > 0, "branch must receive stream frames");

  const lastParent = parentStreamFrames[parentStreamFrames.length - 1];
  const lastBranch = branchStreamFrames[branchStreamFrames.length - 1];

  assert.equal(lastParent.done, true);
  assert.equal(lastParent.content, "reply to: simultaneous parent");

  assert.equal(lastBranch.done, true);
  assert.equal(lastBranch.content, "reply to: simultaneous branch");
});

test("5. Error events target the correct conversation ID", async (t) => {
  const originalMax = generationCounter.max;
  generationCounter.max = 1;
  generationCounter.buckets.clear();
  t.after(() => {
    generationCounter.max = originalMax;
    generationCounter.buckets.clear();
  });

  const server = await startServer();
  t.after(() => server.stop());

  const branchId = randomUUID();
  const ws = await connectWebSocket(server.wsUrl);
  t.after(() => ws.close());

  // First request consumes quota
  const firstReq = collectFrames(
    ws,
    null,
    (msg) => msg.type === "stream" && msg.done
  );
  ws.send(JSON.stringify({ type: "chat", content: "allowed prompt", conversationId: branchId }));
  await firstReq;

  // Second request triggers rate limit error
  const errorFramesPromise = collectFrames(
    ws,
    null,
    (msg) => msg.type === "error"
  );
  ws.send(JSON.stringify({ type: "chat", content: "rate limited prompt", conversationId: branchId }));

  const frames = await errorFramesPromise;
  const errorFrame = frames.find((f) => f.type === "error");
  assert.ok(errorFrame, "must receive error frame");
  assert.equal(errorFrame.code, "RATE_LIMITED");
  assert.equal(errorFrame.conversationId, branchId, "error frame must echo branch conversation ID");
});
