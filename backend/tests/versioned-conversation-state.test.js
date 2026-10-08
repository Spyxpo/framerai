const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");
const WebSocket = require("ws");
const conversationStore = require("../src/conversationStore");
const { mockModel, loadApp, newConversation, startServer } = require("./helpers");

mockModel();
const app = loadApp();

test("basic versioning: conversation starts at version 1 and increments on mutations", async () => {
  const created = await request(app).post("/api/chat/conversations");
  assert.equal(created.status, 200);
  assert.equal(created.body.version, 1);
  assert.ok(created.body.updatedAt);
  const convId = created.body.id;

  // Append a message -> user turn (v2) + assistant reply (v3)
  const msgRes = await request(app)
    .post(`/api/chat/conversations/${convId}/messages`)
    .send({ content: "hello world", expectedVersion: 1 });
  assert.equal(msgRes.status, 200);
  assert.equal(msgRes.body.version, 3);

  const fetched = await request(app).get(`/api/chat/conversations/${convId}`);
  assert.equal(fetched.status, 200);
  assert.equal(fetched.body.version, 3);

  // Update title via PATCH -> increments version to 4
  const patchRes = await request(app)
    .patch(`/api/chat/conversations/${convId}`)
    .send({ title: "Custom Title", expectedVersion: 3 });
  assert.equal(patchRes.status, 200);
  assert.equal(patchRes.body.version, 4);
  assert.equal(patchRes.body.title, "Custom Title");

  const afterPatch = await request(app).get(`/api/chat/conversations/${convId}`);
  assert.equal(afterPatch.body.version, 4);
  assert.equal(afterPatch.body.title, "Custom Title");
});

test("concurrent mutations: stale expectedVersion is rejected with 409 VERSION_CONFLICT", async () => {
  const id = await newConversation(app);
  const initial = await request(app).get(`/api/chat/conversations/${id}`);
  const baseVersion = initial.body.version || 1;

  // Client A mutates at baseVersion (user msg + assistant reply => baseVersion + 2)
  const resA = await request(app)
    .post(`/api/chat/conversations/${id}/messages`)
    .send({ content: "Client A message", expectedVersion: baseVersion });
  assert.equal(resA.status, 200);
  assert.equal(resA.body.version, baseVersion + 2);

  // Client B attempts mutation using stale baseVersion
  const resB = await request(app)
    .post(`/api/chat/conversations/${id}/messages`)
    .send({ content: "Client B message", expectedVersion: baseVersion });
  assert.equal(resB.status, 409);
  assert.equal(resB.body.code, "VERSION_CONFLICT");
  assert.equal(resB.body.currentVersion, baseVersion + 2);
  assert.equal(resB.body.expectedVersion, baseVersion);

  // Newer state is preserved
  const current = await request(app).get(`/api/chat/conversations/${id}`);
  assert.equal(current.body.version, baseVersion + 2);
  assert.equal(current.body.messages.length, 2);
  assert.equal(current.body.messages[0].content, "Client A message");
});

test("rename concurrency: stale expectedVersion cannot overwrite newer state", async () => {
  const id = await newConversation(app);
  const conv = await request(app).get(`/api/chat/conversations/${id}`);
  const baseVersion = conv.body.version || 1;

  // First operation succeeds
  const patch1 = await request(app)
    .patch(`/api/chat/conversations/${id}`)
    .send({ title: "First Title", expectedVersion: baseVersion });
  assert.equal(patch1.status, 200);
  assert.equal(patch1.body.version, baseVersion + 1);

  // Second operation with stale baseVersion fails
  const patch2 = await request(app)
    .patch(`/api/chat/conversations/${id}`)
    .send({ title: "Stale Title", expectedVersion: baseVersion });
  assert.equal(patch2.status, 409);
  assert.equal(patch2.body.code, "VERSION_CONFLICT");
  assert.equal(patch2.body.currentVersion, baseVersion + 1);

  // Verify state wasn't overwritten
  const check = await request(app).get(`/api/chat/conversations/${id}`);
  assert.equal(check.body.title, "First Title");
  assert.equal(check.body.version, baseVersion + 1);
});

test("branching: tracks parentVersion, initializes branch at version 1, rejects stale expectedVersion", async () => {
  const id = await newConversation(app);
  const sendRes = await request(app)
    .post(`/api/chat/conversations/${id}/messages`)
    .send({ content: "Message before branch" });
  assert.equal(sendRes.status, 200);

  const parentConv = await request(app).get(`/api/chat/conversations/${id}`);
  const parentVersion = parentConv.body.version;
  const msgId = parentConv.body.messages[0].id;

  // Stale expectedVersion on branch rejection
  const staleBranch = await request(app)
    .post(`/api/chat/conversations/${id}/branch`)
    .send({ messageId: msgId, expectedVersion: 999 });
  assert.equal(staleBranch.status, 409);
  assert.equal(staleBranch.body.code, "VERSION_CONFLICT");

  // Valid branch creation
  const branchRes = await request(app)
    .post(`/api/chat/conversations/${id}/branch`)
    .send({ messageId: msgId, expectedVersion: parentVersion });
  assert.equal(branchRes.status, 200);
  assert.equal(branchRes.body.version, 1);
  assert.equal(branchRes.body.parentVersion, parentVersion);
  assert.equal(branchRes.body.parentConversationId, id);

  const branchId = branchRes.body.id;

  // Mutating the branch increments branch version (user msg + assistant reply => 3) and does NOT mutate parent
  await request(app)
    .post(`/api/chat/conversations/${branchId}/messages`)
    .send({ content: "Branch turn" });

  const updatedBranch = await request(app).get(`/api/chat/conversations/${branchId}`);
  const parentCheck = await request(app).get(`/api/chat/conversations/${id}`);

  assert.equal(updatedBranch.body.version, 3);
  assert.equal(parentCheck.body.version, parentVersion);
});

test("persistence and backwards compatibility: legacy unversioned conversations default to version 1", () => {
  const legacyId = "legacy-" + Date.now();
  conversationStore.create({
    id: legacyId,
    title: "Legacy Chat",
    messages: [],
    createdAt: new Date().toISOString(),
  });

  const legacy = conversationStore.get(legacyId);
  assert.ok(legacy);
  assert.equal(legacy.version, 1);

  // Appending increments legacy conversation safely
  conversationStore.append(legacyId, { role: "user", content: "hi" });
  const updated = conversationStore.get(legacyId);
  assert.equal(updated.version, 2);
});

test("conversationStore.create protects against stale snapshot overwriting newer version", () => {
  const id = "snapshot-" + Date.now();
  conversationStore.create({
    id,
    title: "V1",
    version: 1,
    messages: [{ id: "m1", content: "first" }],
  });

  // Mutate to version 2
  conversationStore.append(id, { id: "m2", content: "second" });
  const current = conversationStore.get(id);
  assert.equal(current.version, 2);

  // Attempt to overwrite with stale snapshot version 1
  conversationStore.create({
    id,
    title: "Stale Overwrite",
    version: 1,
    messages: [{ id: "m1", content: "first" }],
  });

  const afterStale = conversationStore.get(id);
  assert.equal(afterStale.version, 2);
  assert.equal(afterStale.messages.length, 2);
});

const { randomUUID } = require("node:crypto");

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

test("WebSocket: stale expectedVersion emits VERSION_CONFLICT error frame", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  const convId = randomUUID();
  conversationStore.create({
    id: convId,
    title: "WS Test",
    version: 2,
    messages: [],
  });

  const ws = new WebSocket(server.wsUrl);
  await new Promise((resolve, reject) => {
    ws.on("open", resolve);
    ws.on("error", reject);
  });
  t.after(() => ws.close());

  const framesPromise = collectFrames(ws, null, (msg) => msg.type === "error");

  ws.send(JSON.stringify({
    type: "chat",
    conversationId: convId,
    content: "hello",
    expectedVersion: 1,
  }));

  const frames = await framesPromise;
  const frame = frames.find((f) => f.type === "error");
  assert.ok(frame);
  assert.equal(frame.code, "VERSION_CONFLICT");
  assert.equal(frame.currentVersion, 2);
  assert.equal(frame.expectedVersion, 1);

  // Conversation remains at version 2 and untouched
  const current = conversationStore.get(convId);
  assert.equal(current.version, 2);
  assert.equal(current.messages.length, 0);
});

test("WebSocket: matching expectedVersion emits ack and stream with updated version", async (t) => {
  const server = await startServer();
  t.after(() => server.stop());

  const convId = randomUUID();
  conversationStore.create({
    id: convId,
    title: "WS Success Test",
    version: 1,
    messages: [],
  });

  const ws = new WebSocket(server.wsUrl);
  await new Promise((resolve, reject) => {
    ws.on("open", resolve);
    ws.on("error", reject);
  });
  t.after(() => ws.close());

  const framesPromise = collectFrames(
    ws,
    null,
    (msg) => msg.type === "stream" && msg.done && msg.conversationId === convId
  );

  ws.send(JSON.stringify({
    type: "chat",
    conversationId: convId,
    content: "hello",
    expectedVersion: 1,
  }));

  const frames = await framesPromise;

  const ack = frames.find((f) => f.type === "ack");
  assert.ok(ack);
  assert.equal(ack.version, 2);

  const doneFrame = frames.find((f) => f.done);
  assert.ok(doneFrame);
  assert.equal(doneFrame.version, 3);

  const current = conversationStore.get(convId);
  assert.equal(current.version, 3);
});
