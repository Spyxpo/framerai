/**
 * Regression tests for Issue #475:
 * Deleted and Truncated Messages with Client IDs Must Not Resurrect During Reconnect Sync.
 *
 * When a message is deleted (deleteMessage) or truncated (truncateAfter, truncateFrom),
 * both the server-assigned id and the client-assigned clientId must be tracked in
 * conv.deletedMessageIds so that reconnect synchronization does not treat the removed
 * turn as an offline message and resurrect it.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");
const { randomUUID } = require("node:crypto");

const { mockModel, startServer } = require("./helpers");
const conversationStore = require("../src/conversationStore");

mockModel();

async function newConversation(app) {
  const res = await request(app).post("/api/chat/conversations");
  assert.equal(res.status, 200);
  return res.body.id;
}

async function stored(app, conversationId) {
  const res = await request(app).get(`/api/chat/conversations/${conversationId}`);
  assert.equal(res.status, 200);
  return res.body;
}

async function sendRest(app, conversationId, content, clientMessageId) {
  return request(app)
    .post(`/api/chat/conversations/${conversationId}/messages`)
    .send({ content, ...(clientMessageId ? { clientMessageId } : {}) });
}

async function sync(app, conversationId, messages, clientVersion = 1, deletedMessageIds = []) {
  return request(app)
    .post(`/api/chat/conversations/${conversationId}/sync`)
    .send({ clientVersion, messages, deletedMessageIds });
}

const local = (id, content) => ({
  id,
  role: "user",
  content,
  type: "text",
  attachments: [],
  timestamp: new Date().toISOString(),
});

test("#475: a message deleted by server ID does not resurrect when synced with client ID", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const clientMsgId = randomUUID();
    const sendRes = await sendRest(server.app, conversationId, "message to delete", clientMsgId);
    assert.equal(sendRes.status, 200);

    const beforeDelete = await stored(server.app, conversationId);
    assert.equal(beforeDelete.messages.length, 2); // user + assistant
    const serverUserMsg = beforeDelete.messages.find((m) => m.role === "user");
    assert.equal(serverUserMsg.clientId, clientMsgId);

    // Delete message using server ID
    const delRes = await request(server.app).delete(`/api/chat/conversations/${conversationId}/messages/${serverUserMsg.id}`);
    assert.equal(delRes.status, 200);

    const afterDelete = await stored(server.app, conversationId);
    assert.equal(afterDelete.messages.some((m) => m.id === serverUserMsg.id), false);

    // Verify both server ID and client ID are recorded in deletedMessageIds
    const rawConv = conversationStore.get(conversationId);
    assert.ok(rawConv.deletedMessageIds.includes(serverUserMsg.id), "server ID must be tracked");
    assert.ok(rawConv.deletedMessageIds.includes(clientMsgId), "client ID must be tracked");

    // Client reconnects and syncs holding its local copy under clientMsgId
    const syncRes = await sync(server.app, conversationId, [local(clientMsgId, "message to delete")], afterDelete.version);
    assert.equal(syncRes.status, 200);
    assert.equal(syncRes.body.diff.messages.clientOnly.length, 0, "deleted message must not be treated as client-only");

    // Confirm message did not resurrect
    const afterSync = await stored(server.app, conversationId);
    assert.equal(afterSync.messages.some((m) => m.content === "message to delete"), false);
  } finally {
    await server.stop();
  }
});

test("#475: a message can be deleted by its client ID and tracks both IDs", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const clientMsgId = randomUUID();
    const sendRes = await sendRest(server.app, conversationId, "delete by client id", clientMsgId);
    assert.equal(sendRes.status, 200);

    const beforeDelete = await stored(server.app, conversationId);
    const serverUserMsg = beforeDelete.messages.find((m) => m.role === "user");
    assert.equal(serverUserMsg.clientId, clientMsgId);

    // Delete message using client ID
    const delRes = await request(server.app).delete(`/api/chat/conversations/${conversationId}/messages/${clientMsgId}`);
    assert.equal(delRes.status, 200);
    assert.equal(delRes.body.success, true);

    const afterDelete = await stored(server.app, conversationId);
    assert.equal(afterDelete.messages.some((m) => m.id === serverUserMsg.id), false);

    // Both IDs are recorded
    const rawConv = conversationStore.get(conversationId);
    assert.ok(rawConv.deletedMessageIds.includes(clientMsgId));
    assert.ok(rawConv.deletedMessageIds.includes(serverUserMsg.id));

    // Sync under client ID does not resurrect
    const syncRes = await sync(server.app, conversationId, [local(clientMsgId, "delete by client id")], afterDelete.version);
    assert.equal(syncRes.status, 200);
    assert.equal(syncRes.body.diff.messages.clientOnly.length, 0);

    const afterSync = await stored(server.app, conversationId);
    assert.equal(afterSync.messages.some((m) => m.content === "delete by client id"), false);
  } finally {
    await server.stop();
  }
});

test("#475: truncateAfter tracks client IDs of truncated messages and prevents sync resurrection", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const client1 = randomUUID();
    const client2 = randomUUID();

    await sendRest(server.app, conversationId, "first prompt", client1);
    await sendRest(server.app, conversationId, "second prompt", client2);

    const convBefore = await stored(server.app, conversationId);
    assert.equal(convBefore.messages.length, 4); // 2 turns: u1, a1, u2, a2
    const u1 = convBefore.messages.find((m) => m.content === "first prompt");
    const u2 = convBefore.messages.find((m) => m.content === "second prompt");
    assert.equal(u2.clientId, client2);

    // User edits message 1, truncating after u1
    const editRes = await request(server.app)
      .post(`/api/chat/conversations/${conversationId}/messages`)
      .send({ content: "edited first prompt", editMessageId: u1.id });
    assert.equal(editRes.status, 200);

    const rawConv = conversationStore.get(conversationId);
    assert.ok(rawConv.deletedMessageIds.includes(u2.id), "u2 server ID must be tracked as deleted");
    assert.ok(rawConv.deletedMessageIds.includes(client2), "u2 client ID must be tracked as deleted");

    // Client reconnects and syncs with stale copy of turn 2 under client2
    const syncRes = await sync(server.app, conversationId, [
      local(u1.id, "edited first prompt"),
      local(client2, "second prompt"),
    ], rawConv.version);
    assert.equal(syncRes.status, 200);
    assert.equal(syncRes.body.diff.messages.clientOnly.length, 0);

    const afterSync = await stored(server.app, conversationId);
    assert.equal(afterSync.messages.some((m) => m.content === "second prompt"), false, "truncated turn 2 must not resurrect");
  } finally {
    await server.stop();
  }
});

test("#475: truncateFrom tracks client IDs of truncated messages and prevents sync resurrection", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const client1 = randomUUID();
    const client2 = randomUUID();

    await sendRest(server.app, conversationId, "first turn", client1);
    await sendRest(server.app, conversationId, "second turn", client2);

    const convBefore = await stored(server.app, conversationId);
    const u2 = convBefore.messages.find((m) => m.content === "second turn");
    const a2 = convBefore.messages[3];

    // User regenerates turn 2 (regenerateMessageId: a2.id truncates from a2)
    const regenRes = await request(server.app)
      .post(`/api/chat/conversations/${conversationId}/messages`)
      .send({ regenerateMessageId: a2.id });
    assert.equal(regenRes.status, 200);

    const rawConv = conversationStore.get(conversationId);
    assert.ok(rawConv.deletedMessageIds.includes(a2.id), "a2 must be tracked in deletedMessageIds");

    // Direct truncateFrom on u2 tracks u2.id and u2.clientId
    const client3 = randomUUID();
    await sendRest(server.app, conversationId, "third turn", client3);
    const convWith3 = await stored(server.app, conversationId);
    const u3 = convWith3.messages.find((m) => m.content === "third turn");

    conversationStore.truncateFrom(conversationId, u3.id);
    assert.ok(rawConv.deletedMessageIds.includes(u3.id));
    assert.ok(rawConv.deletedMessageIds.includes(client3));

    // Stale sync carrying client3 must not resurrect turn 3
    const syncRes = await sync(server.app, conversationId, [local(client3, "third turn")], rawConv.version);
    assert.equal(syncRes.status, 200);
    assert.equal(syncRes.body.diff.messages.clientOnly.length, 0);

    const afterSync = await stored(server.app, conversationId);
    assert.equal(afterSync.messages.some((m) => m.content === "third turn"), false);
  } finally {
    await server.stop();
  }
});

test("#475: deleted message tracking remains isolated between conversations", async () => {
  const server = await startServer();
  try {
    const convA = await newConversation(server.app);
    const convB = await newConversation(server.app);

    const clientMsgId = randomUUID();
    await sendRest(server.app, convA, "message in A", clientMsgId);
    const storedA = await stored(server.app, convA);
    const msgA = storedA.messages.find((m) => m.role === "user");

    // Delete in conversation A
    await request(server.app).delete(`/api/chat/conversations/${convA}/messages/${msgA.id}`);

    const rawA = conversationStore.get(convA);
    const rawB = conversationStore.get(convB);
    assert.ok(rawA.deletedMessageIds.includes(clientMsgId));
    assert.equal(rawB.deletedMessageIds || [].length, 0, "conversation B must have no deleted IDs");

    // Syncing an offline message in conversation B with that ID is NOT blocked
    const syncB = await sync(server.app, convB, [local(clientMsgId, "message in B")]);
    assert.equal(syncB.status, 200);
    assert.equal(syncB.body.diff.messages.clientOnly.length, 1);

    const storedB = await stored(server.app, convB);
    assert.equal(storedB.messages.some((m) => m.content === "message in B"), true);
  } finally {
    await server.stop();
  }
});

test("#475: sync with both active and deleted turns preserves active and ignores deleted", async () => {
  const server = await startServer();
  try {
    const conversationId = await newConversation(server.app);
    const deletedClientId = randomUUID();
    const activeClientId = randomUUID();

    // Send and delete first message
    const sendRes = await sendRest(server.app, conversationId, "deleted content", deletedClientId);
    const serverUserMsg = (await stored(server.app, conversationId)).messages.find((m) => m.role === "user");
    await request(server.app).delete(`/api/chat/conversations/${conversationId}/messages/${serverUserMsg.id}`);

    // Client syncs sending both the deleted turn and a new active offline turn
    const syncRes = await sync(server.app, conversationId, [
      local(deletedClientId, "deleted content"),
      local(activeClientId, "active offline content"),
    ]);

    assert.equal(syncRes.status, 200);
    assert.deepEqual(syncRes.body.diff.messages.clientOnly.map((m) => m.id), [activeClientId]);

    const afterSync = await stored(server.app, conversationId);
    assert.equal(afterSync.messages.some((m) => m.content === "deleted content"), false);
    assert.equal(afterSync.messages.some((m) => m.content === "active offline content"), true);
  } finally {
    await server.stop();
  }
});
