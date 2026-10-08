const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");
const { randomUUID } = require("node:crypto");
const conversationStore = require("../src/conversationStore");
const conversationSync = require("../src/conversationSync");
const { mockModel, loadApp, newConversation } = require("./helpers");

mockModel();
const app = loadApp();

test("Scenario 1: No differences -> no unnecessary changes or version bumps", async () => {
  const id = await newConversation(app);
  const convRes = await request(app).get(`/api/chat/conversations/${id}`);
  const initialConv = convRes.body;

  const syncRes = await request(app)
    .post(`/api/chat/conversations/${id}/sync`)
    .send({
      clientVersion: initialConv.version,
      messages: initialConv.messages,
      title: initialConv.title,
    });

  assert.equal(syncRes.status, 200);
  assert.equal(syncRes.body.status, "up_to_date");
  assert.equal(syncRes.body.version, initialConv.version);
  assert.equal(syncRes.body.diff.hasDifferences, false);
});

test("Scenario 2: Client-only message -> correctly reconciled and appended", async () => {
  const id = await newConversation(app);
  const convRes = await request(app).get(`/api/chat/conversations/${id}`);
  const baseVersion = convRes.body.version;

  const clientMsg = {
    id: randomUUID(),
    role: "user",
    content: "Client offline message",
    type: "text",
    timestamp: new Date().toISOString(),
  };

  const syncRes = await request(app)
    .post(`/api/chat/conversations/${id}/sync`)
    .send({
      clientVersion: baseVersion,
      messages: [clientMsg],
    });

  assert.equal(syncRes.status, 200);
  assert.equal(syncRes.body.status, "client_ahead");
  assert.equal(syncRes.body.version, baseVersion + 1);
  assert.equal(syncRes.body.conversation.messages.length, 1);
  assert.equal(syncRes.body.conversation.messages[0].id, clientMsg.id);
  assert.equal(syncRes.body.conversation.messages[0].content, "Client offline message");

  // Verify in store
  const stored = conversationStore.get(id);
  assert.equal(stored.version, baseVersion + 1);
  assert.equal(stored.messages[0].id, clientMsg.id);
});

test("Scenario 3: Server-only message -> correctly reported and preserved", async () => {
  const id = await newConversation(app);
  // Add a message on server via REST
  const msgRes = await request(app)
    .post(`/api/chat/conversations/${id}/messages`)
    .send({ content: "Server question" });
  assert.equal(msgRes.status, 200);

  const serverConv = (await request(app).get(`/api/chat/conversations/${id}`)).body;
  assert.equal(serverConv.messages.length, 2);

  // Client syncs with empty messages and clientVersion 1
  const syncRes = await request(app)
    .post(`/api/chat/conversations/${id}/sync`)
    .send({
      clientVersion: 1,
      messages: [],
    });

  assert.equal(syncRes.status, 200);
  assert.equal(syncRes.body.status, "server_ahead");
  assert.equal(syncRes.body.version, serverConv.version);
  assert.equal(syncRes.body.conversation.messages.length, 2);
  assert.equal(syncRes.body.diff.messages.serverOnly.length, 2);
});

test("Scenario 4: Both sides add different messages -> deterministic merge without concatenation duplicates", async () => {
  const id = await newConversation(app);
  // Initial message: Turn 1
  await request(app)
    .post(`/api/chat/conversations/${id}/messages`)
    .send({ content: "Turn 1" });

  const afterTurn1 = (await request(app).get(`/api/chat/conversations/${id}`)).body;
  const commonMessages = afterTurn1.messages;

  // Server adds Turn 2
  await request(app)
    .post(`/api/chat/conversations/${id}/messages`)
    .send({ content: "Turn 2 from Server" });

  const clientNewMsg = {
    id: randomUUID(),
    role: "user",
    content: "Turn 2 from Client",
    type: "text",
    timestamp: new Date(Date.now() + 1000).toISOString(),
  };

  // Client syncs having commonMessages + clientNewMsg
  const syncRes = await request(app)
    .post(`/api/chat/conversations/${id}/sync`)
    .send({
      clientVersion: afterTurn1.version,
      messages: [...commonMessages, clientNewMsg],
    });

  assert.equal(syncRes.status, 200);
  const finalMessages = syncRes.body.conversation.messages;

  // Final messages must contain common messages + server turn + client turn
  assert.ok(finalMessages.some((m) => m.id === clientNewMsg.id));
  assert.ok(finalMessages.some((m) => m.content === "Turn 2 from Server"));
  assert.ok(finalMessages.some((m) => m.content === "Turn 1"));

  // Check no duplicate IDs exist
  const ids = finalMessages.map((m) => m.id);
  assert.equal(new Set(ids).size, ids.length);
});

test("Scenario 5: Same message ID on both sides -> correct deduplication", async () => {
  const id = await newConversation(app);
  const msgId = randomUUID();
  const msg = {
    id: msgId,
    role: "user",
    content: "Identical message",
    type: "text",
    timestamp: "2026-10-01T12:00:00.000Z",
  };

  conversationStore.append(id, msg);
  const convBefore = conversationStore.get(id);

  const syncRes = await request(app)
    .post(`/api/chat/conversations/${id}/sync`)
    .send({
      clientVersion: convBefore.version,
      messages: [{ ...msg }],
    });

  assert.equal(syncRes.status, 200);
  assert.equal(syncRes.body.conversation.messages.length, 1);
  assert.equal(syncRes.body.conversation.messages[0].id, msgId);
  assert.equal(syncRes.body.diff.messages.identicalCount, 1);
});

test("Scenario 6: Message update conflict -> completed assistant message resolves conflict", async () => {
  const id = await newConversation(app);
  const asstId = randomUUID();
  const serverMsg = {
    id: asstId,
    role: "assistant",
    content: "Full completed response",
    type: "text",
    completed: true,
    timestamp: "2026-10-01T12:00:01.000Z",
  };
  conversationStore.append(id, serverMsg);

  const clientMsg = {
    id: asstId,
    role: "assistant",
    content: "Partial streaming...",
    type: "text",
    completed: false,
    timestamp: "2026-10-01T12:00:00.000Z",
  };

  const syncRes = await request(app)
    .post(`/api/chat/conversations/${id}/sync`)
    .send({
      clientVersion: 1,
      messages: [clientMsg],
    });

  assert.equal(syncRes.status, 200);
  const resolved = syncRes.body.conversation.messages.find((m) => m.id === asstId);
  assert.equal(resolved.content, "Full completed response");
  assert.equal(resolved.completed, true);
});

test("Scenario 7: Delete vs update -> deleted message is never silently resurrected", async () => {
  const id = await newConversation(app);
  const msgToDeleteId = randomUUID();
  const msgToKeepId = randomUUID();

  conversationStore.append(id, { id: msgToDeleteId, role: "user", content: "To delete", type: "text" });
  conversationStore.append(id, { id: msgToKeepId, role: "user", content: "To keep", type: "text" });

  // Delete message via DELETE endpoint
  const delRes = await request(app).delete(`/api/chat/conversations/${id}/messages/${msgToDeleteId}`);
  assert.equal(delRes.status, 200);
  assert.equal(delRes.body.success, true);

  // Stale client still carries the deleted message
  const syncRes = await request(app)
    .post(`/api/chat/conversations/${id}/sync`)
    .send({
      clientVersion: 2,
      messages: [
        { id: msgToDeleteId, role: "user", content: "Resurrect me?", type: "text" },
        { id: msgToKeepId, role: "user", content: "To keep", type: "text" },
      ],
      deletedMessageIds: [msgToDeleteId],
    });

  assert.equal(syncRes.status, 200);
  const finalMessages = syncRes.body.conversation.messages;
  assert.equal(finalMessages.some((m) => m.id === msgToDeleteId), false);
  assert.equal(finalMessages.some((m) => m.id === msgToKeepId), true);
});

test("Scenario 8: Rename + message update -> preserves both independent valid changes", async () => {
  const id = await newConversation(app);
  // Server appends a message
  await request(app)
    .post(`/api/chat/conversations/${id}/messages`)
    .send({ content: "Server message" });

  const serverConv = (await request(app).get(`/api/chat/conversations/${id}`)).body;

  // Client concurrently renamed with titleUpdatedAt
  const syncRes = await request(app)
    .post(`/api/chat/conversations/${id}/sync`)
    .send({
      clientVersion: 1,
      title: "Client Renamed Title",
      titleUpdatedAt: Date.now() + 5000,
      messages: [],
    });

  assert.equal(syncRes.status, 200);
  assert.equal(syncRes.body.conversation.title, "Client Renamed Title");
  // Server message is preserved
  assert.equal(syncRes.body.conversation.messages.length, 2);
  assert.equal(syncRes.body.conversation.messages[0].content, "Server message");
});

test("Scenario 9: Concurrent message creation -> monotonic ordering and version increment", async () => {
  const id = await newConversation(app);
  const msg1 = { id: randomUUID(), role: "user", content: "Msg 1", type: "text", timestamp: "2026-10-01T10:00:00.000Z" };
  const msg2 = { id: randomUUID(), role: "user", content: "Msg 2", type: "text", timestamp: "2026-10-01T10:00:05.000Z" };

  conversationStore.append(id, msg1);

  const syncRes = await request(app)
    .post(`/api/chat/conversations/${id}/sync`)
    .send({
      clientVersion: 2,
      messages: [msg1, msg2],
    });

  assert.equal(syncRes.status, 200);
  assert.equal(syncRes.body.conversation.messages.length, 2);
  assert.equal(syncRes.body.conversation.messages[0].id, msg1.id);
  assert.equal(syncRes.body.conversation.messages[1].id, msg2.id);
});

test("Scenario 10: Repeated synchronization is idempotent", async () => {
  const id = await newConversation(app);
  const clientMsg = { id: randomUUID(), role: "user", content: "Repeated sync", type: "text" };

  const firstSync = await request(app)
    .post(`/api/chat/conversations/${id}/sync`)
    .send({
      clientVersion: 1,
      messages: [clientMsg],
    });
  assert.equal(firstSync.status, 200);
  const firstVersion = firstSync.body.version;

  const secondSync = await request(app)
    .post(`/api/chat/conversations/${id}/sync`)
    .send({
      clientVersion: firstVersion,
      messages: firstSync.body.conversation.messages,
    });
  assert.equal(secondSync.status, 200);
  assert.equal(secondSync.body.status, "up_to_date");
  assert.equal(secondSync.body.version, firstVersion);
  assert.equal(secondSync.body.conversation.messages.length, 1);
});

test("Scenario 11: Stale version cannot overwrite newer state", async () => {
  const id = await newConversation(app);
  // Advance server to version 4
  await request(app).post(`/api/chat/conversations/${id}/messages`).send({ content: "T1" });
  await request(app).post(`/api/chat/conversations/${id}/messages`).send({ content: "T2" });

  const current = (await request(app).get(`/api/chat/conversations/${id}`)).body;
  assert.ok(current.version >= 4);

  // Client attempts sync with stale clientVersion 1 and empty messages
  const syncRes = await request(app)
    .post(`/api/chat/conversations/${id}/sync`)
    .send({
      clientVersion: 1,
      messages: [],
    });

  assert.equal(syncRes.status, 200);
  // Server state was not wiped out
  assert.equal(syncRes.body.conversation.messages.length, 4);
  assert.equal(syncRes.body.version, current.version);
});

test("Scenario 12: Branch divergence -> branch metadata preserved and isolated from parent", async () => {
  const parentId = await newConversation(app);
  const m1Res = await request(app)
    .post(`/api/chat/conversations/${parentId}/messages`)
    .send({ content: "Parent message" });

  const parentConv = (await request(app).get(`/api/chat/conversations/${parentId}`)).body;
  const userMsgId = parentConv.messages[0].id;

  const branchRes = await request(app)
    .post(`/api/chat/conversations/${parentId}/branch`)
    .send({ messageId: userMsgId });
  assert.equal(branchRes.status, 200);
  const branchId = branchRes.body.id;

  // Sync on branch
  const branchClientMsg = {
    id: randomUUID(),
    role: "user",
    content: "Branch specific turn",
    type: "text",
  };

  const syncRes = await request(app)
    .post(`/api/chat/conversations/${branchId}/sync`)
    .send({
      clientVersion: 1,
      messages: [...branchRes.body.messages, branchClientMsg],
    });

  assert.equal(syncRes.status, 200);
  assert.equal(syncRes.body.conversation.parentConversationId, parentId);
  assert.equal(syncRes.body.conversation.branchedFromMessageId, userMsgId);

  // Verify parent was unaffected
  const checkParent = (await request(app).get(`/api/chat/conversations/${parentId}`)).body;
  assert.equal(checkParent.messages.some((m) => m.content === "Branch specific turn"), false);
});

test("Scenario 13: Large conversation with small diff -> linear time O(N) diff", async () => {
  const id = await newConversation(app);
  const baseMessages = [];
  for (let i = 0; i < 200; i++) {
    baseMessages.push({
      id: randomUUID(),
      role: i % 2 === 0 ? "user" : "assistant",
      content: `Message ${i}`,
      type: "text",
      timestamp: new Date(1700000000000 + i * 1000).toISOString(),
    });
  }

  // Populate server store
  const conv = conversationStore.get(id);
  conv.messages = [...baseMessages];
  conv.version = 100;

  // Client has all 200 plus 1 new message
  const newMsg = {
    id: randomUUID(),
    role: "user",
    content: "New 201st message",
    type: "text",
    timestamp: new Date().toISOString(),
  };

  const startTime = Date.now();
  const syncRes = await request(app)
    .post(`/api/chat/conversations/${id}/sync`)
    .send({
      clientVersion: 100,
      messages: [...baseMessages, newMsg],
    });
  const elapsed = Date.now() - startTime;

  assert.equal(syncRes.status, 200);
  assert.equal(syncRes.body.conversation.messages.length, 201);
  assert.equal(syncRes.body.version, 101);
  assert.ok(elapsed < 1000, `Large conversation sync took ${elapsed}ms, expected < 1000ms`);
});
