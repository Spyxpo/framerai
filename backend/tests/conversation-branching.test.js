const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");
const { randomUUID } = require("node:crypto");

const { mockModel, loadApp } = require("./helpers");

const calls = mockModel();
const app = loadApp();
const conversations = require("../src/conversationStore");
const { generationCounter, apiCounter } = require("../src/middleware/limiters");

test.beforeEach(() => {
  generationCounter.buckets.clear();
  apiCounter.buckets.clear();
});

async function createConversationWithMessages(messages) {
  const created = await request(app).post("/api/chat/conversations");
  const convId = created.body.id;

  for (const msg of messages) {
    if (msg.role === "user") {
      await request(app)
        .post(`/api/chat/conversations/${convId}/messages`)
        .send({ content: msg.content, type: msg.type || "text" });
    }
  }

  const fetched = await request(app).get(`/api/chat/conversations/${convId}`);
  return fetched.body;
}

test("1. Branch from first message", async () => {
  const conv = await createConversationWithMessages([
    { role: "user", content: "first user question" },
    { role: "user", content: "second user question" },
  ]);

  const firstMsg = conv.messages[0];
  const res = await request(app)
    .post(`/api/chat/conversations/${conv.id}/branch`)
    .send({ messageId: firstMsg.id });

  assert.equal(res.status, 200);
  assert.equal(res.body.parentConversationId, conv.id);
  assert.equal(res.body.branchedFromMessageId, firstMsg.id);
  assert.equal(res.body.messages.length, 1);
  assert.equal(res.body.messages[0].id, firstMsg.id);
  assert.equal(res.body.messages[0].content, firstMsg.content);
});

test("2. Branch from middle message", async () => {
  const conv = await createConversationWithMessages([
    { role: "user", content: "A" },
    { role: "user", content: "B" },
    { role: "user", content: "C" },
  ]);
  // conv.messages: [user A, assistant reply A, user B, assistant reply B, user C, assistant reply C]
  assert.equal(conv.messages.length, 6);
  const middleMsg = conv.messages[2]; // user B

  const res = await request(app)
    .post(`/api/chat/conversations/${conv.id}/branch`)
    .send({ messageId: middleMsg.id });

  assert.equal(res.status, 200);
  assert.equal(res.body.messages.length, 3); // user A, assistant A, user B
  assert.equal(res.body.messages[2].id, middleMsg.id);
  assert.equal(res.body.messages[2].content, "B");
});

test("3. Branch from latest message", async () => {
  const conv = await createConversationWithMessages([
    { role: "user", content: "msg1" },
    { role: "user", content: "msg2" },
  ]);
  const latestMsg = conv.messages[conv.messages.length - 1];

  const res = await request(app)
    .post(`/api/chat/conversations/${conv.id}/branch`)
    .send({ messageId: latestMsg.id });

  assert.equal(res.status, 200);
  assert.equal(res.body.messages.length, conv.messages.length);
  assert.equal(res.body.messages[res.body.messages.length - 1].id, latestMsg.id);
});

test("4. Branch from user message", async () => {
  const conv = await createConversationWithMessages([
    { role: "user", content: "user question" },
  ]);
  const userMsg = conv.messages[0];
  assert.equal(userMsg.role, "user");

  const res = await request(app)
    .post(`/api/chat/conversations/${conv.id}/branch`)
    .send({ messageId: userMsg.id });

  assert.equal(res.status, 200);
  assert.equal(res.body.messages.length, 1);
  assert.equal(res.body.messages[0].role, "user");
  assert.equal(res.body.messages[0].id, userMsg.id);
});

test("5. Branch from assistant message", async () => {
  const conv = await createConversationWithMessages([
    { role: "user", content: "user question" },
  ]);
  const assistantMsg = conv.messages[1];
  assert.equal(assistantMsg.role, "assistant");

  const res = await request(app)
    .post(`/api/chat/conversations/${conv.id}/branch`)
    .send({ messageId: assistantMsg.id });

  assert.equal(res.status, 200);
  assert.equal(res.body.messages.length, 2);
  assert.equal(res.body.messages[1].role, "assistant");
  assert.equal(res.body.messages[1].id, assistantMsg.id);
});

test("6. Selected message is included", async () => {
  const conv = await createConversationWithMessages([
    { role: "user", content: "m1" },
    { role: "user", content: "m2" },
  ]);
  const target = conv.messages[2]; // user m2

  const res = await request(app)
    .post(`/api/chat/conversations/${conv.id}/branch`)
    .send({ messageId: target.id });

  assert.equal(res.status, 200);
  assert.ok(res.body.messages.some((m) => m.id === target.id));
  assert.equal(res.body.messages[res.body.messages.length - 1].id, target.id);
});

test("7. Later messages are excluded", async () => {
  const conv = await createConversationWithMessages([
    { role: "user", content: "keep 1" },
    { role: "user", content: "keep 2" },
    { role: "user", content: "exclude 3" },
  ]);
  const branchPoint = conv.messages[1]; // assistant reply to keep 1
  const excludedUserMsg = conv.messages[2];
  const excludedAssistantMsg = conv.messages[3];

  const res = await request(app)
    .post(`/api/chat/conversations/${conv.id}/branch`)
    .send({ messageId: branchPoint.id });

  assert.equal(res.status, 200);
  assert.equal(res.body.messages.length, 2);
  assert.ok(!res.body.messages.some((m) => m.id === excludedUserMsg.id));
  assert.ok(!res.body.messages.some((m) => m.id === excludedAssistantMsg.id));
});

test("8. Message ordering is preserved", async () => {
  const conv = await createConversationWithMessages([
    { role: "user", content: "order 1" },
    { role: "user", content: "order 2" },
  ]);

  const res = await request(app)
    .post(`/api/chat/conversations/${conv.id}/branch`)
    .send({ messageId: conv.messages[3].id });

  assert.equal(res.status, 200);
  assert.deepEqual(
    res.body.messages.map((m) => m.id),
    conv.messages.map((m) => m.id)
  );
});

test("9. New unique conversation ID generated", async () => {
  const conv = await createConversationWithMessages([
    { role: "user", content: "hello" },
  ]);

  const res = await request(app)
    .post(`/api/chat/conversations/${conv.id}/branch`)
    .send({ messageId: conv.messages[0].id });

  assert.equal(res.status, 200);
  assert.notEqual(res.body.id, conv.id);
  assert.match(
    res.body.id,
    /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
  );
});

test("10. Parent remains unchanged after branch creation", async () => {
  const conv = await createConversationWithMessages([
    { role: "user", content: "parent msg 1" },
    { role: "user", content: "parent msg 2" },
  ]);
  const originalMsgCount = conv.messages.length;

  const res = await request(app)
    .post(`/api/chat/conversations/${conv.id}/branch`)
    .send({ messageId: conv.messages[0].id });

  assert.equal(res.status, 200);

  const parentAfter = await request(app).get(`/api/chat/conversations/${conv.id}`);
  assert.equal(parentAfter.body.messages.length, originalMsgCount);
  assert.deepEqual(
    parentAfter.body.messages.map((m) => m.id),
    conv.messages.map((m) => m.id)
  );
});

test("11. Multiple branches from same parent", async () => {
  const conv = await createConversationWithMessages([
    { role: "user", content: "common root" },
    { role: "user", content: "follow up" },
  ]);

  const b1 = await request(app)
    .post(`/api/chat/conversations/${conv.id}/branch`)
    .send({ messageId: conv.messages[0].id });

  const b2 = await request(app)
    .post(`/api/chat/conversations/${conv.id}/branch`)
    .send({ messageId: conv.messages[1].id });

  const b3 = await request(app)
    .post(`/api/chat/conversations/${conv.id}/branch`)
    .send({ messageId: conv.messages[2].id });

  assert.equal(b1.status, 200);
  assert.equal(b2.status, 200);
  assert.equal(b3.status, 200);

  assert.notEqual(b1.body.id, b2.body.id);
  assert.notEqual(b2.body.id, b3.body.id);
  assert.notEqual(b1.body.id, b3.body.id);

  assert.equal(b1.body.parentConversationId, conv.id);
  assert.equal(b2.body.parentConversationId, conv.id);
  assert.equal(b3.body.parentConversationId, conv.id);

  assert.equal(b1.body.messages.length, 1);
  assert.equal(b2.body.messages.length, 2);
  assert.equal(b3.body.messages.length, 3);
});

test("12. Nested branch (branch of a branch)", async () => {
  const conv = await createConversationWithMessages([
    { role: "user", content: "root turn" },
  ]);

  // Branch A from root
  const branchA = await request(app)
    .post(`/api/chat/conversations/${conv.id}/branch`)
    .send({ messageId: conv.messages[0].id });
  assert.equal(branchA.status, 200);

  // Send a message in Branch A
  const msgA = await request(app)
    .post(`/api/chat/conversations/${branchA.body.id}/messages`)
    .send({ content: "branch A addition" });
  assert.equal(msgA.status, 200);

  // Branch A.1 from Branch A
  const branchA1 = await request(app)
    .post(`/api/chat/conversations/${branchA.body.id}/branch`)
    .send({ messageId: msgA.body.id });
  assert.equal(branchA1.status, 200);
  assert.equal(branchA1.body.parentConversationId, branchA.body.id);
  assert.equal(branchA1.body.branchedFromMessageId, msgA.body.id);
  assert.equal(branchA1.body.messages.length, 3); // root turn, branch A user, branch A assistant
});

test("13. Invalid conversation ID", async () => {
  // Malformed UUID
  const resBadId = await request(app)
    .post("/api/chat/conversations/not-a-uuid/branch")
    .send({ messageId: randomUUID() });
  assert.equal(resBadId.status, 400);

  // Non-existent UUID
  const nonExistentId = randomUUID();
  const resNotFound = await request(app)
    .post(`/api/chat/conversations/${nonExistentId}/branch`)
    .send({ messageId: randomUUID() });
  assert.equal(resNotFound.status, 404);
});

test("14. Invalid message ID", async () => {
  const conv = await createConversationWithMessages([
    { role: "user", content: "hello" },
  ]);

  // Missing messageId
  const resMissing = await request(app)
    .post(`/api/chat/conversations/${conv.id}/branch`)
    .send({});
  assert.equal(resMissing.status, 400);

  // Malformed messageId
  const resMalformed = await request(app)
    .post(`/api/chat/conversations/${conv.id}/branch`)
    .send({ messageId: "123-abc" });
  assert.equal(resMalformed.status, 400);
});

test("15. Message belongs to another conversation", async () => {
  const conv1 = await createConversationWithMessages([
    { role: "user", content: "conv 1" },
  ]);
  const conv2 = await createConversationWithMessages([
    { role: "user", content: "conv 2" },
  ]);

  const foreignMessageId = conv2.messages[0].id;
  const res = await request(app)
    .post(`/api/chat/conversations/${conv1.id}/branch`)
    .send({ messageId: foreignMessageId });

  assert.equal(res.status, 400);
  assert.match(res.body.error, /message not found in conversation/i);
});

test("16. Branch persistence and retrieval", async () => {
  const conv = await createConversationWithMessages([
    { role: "user", content: "persisted" },
  ]);

  const branch = await request(app)
    .post(`/api/chat/conversations/${conv.id}/branch`)
    .send({ messageId: conv.messages[0].id });

  assert.equal(branch.status, 200);

  // Retrieve via GET /chat/conversations/:id
  const fetched = await request(app).get(`/api/chat/conversations/${branch.body.id}`);
  assert.equal(fetched.status, 200);
  assert.equal(fetched.body.id, branch.body.id);
  assert.equal(fetched.body.parentConversationId, conv.id);
  assert.equal(fetched.body.branchedFromMessageId, conv.messages[0].id);

  // Listed in GET /chat/conversations
  const list = await request(app).get("/api/chat/conversations");
  const foundInList = list.body.find((c) => c.id === branch.body.id);
  assert.ok(foundInList);
  assert.equal(foundInList.parentConversationId, conv.id);
  assert.equal(foundInList.branchedFromMessageId, conv.messages[0].id);
});

test("17. Branch generation context contains only branch history", async () => {
  const conv = await createConversationWithMessages([
    { role: "user", content: "turn A" },
    { role: "user", content: "turn B" },
    { role: "user", content: "turn C (should be excluded)" },
  ]);
  // Branch from turn A (messages[1] is assistant reply to turn A)
  const branch = await request(app)
    .post(`/api/chat/conversations/${conv.id}/branch`)
    .send({ messageId: conv.messages[1].id });
  assert.equal(branch.status, 200);

  const beforeCalls = calls.length;

  // Send turn D in branch
  await request(app)
    .post(`/api/chat/conversations/${branch.body.id}/messages`)
    .send({ content: "turn D in branch" });

  const [lastCall] = calls.slice(beforeCalls);
  assert.equal(lastCall.name, "processMessage");
  const modelMessages = lastCall.args[0];

  // Context must contain: turn A (user), turn A (assistant), turn D (user)
  // It MUST NOT contain turn B or turn C
  const contents = modelMessages.map((m) => m.content);
  assert.ok(contents.includes("turn A"));
  assert.ok(contents.includes("reply to: turn A"));
  assert.ok(contents.includes("turn D in branch"));
  assert.ok(!contents.includes("turn B"));
  assert.ok(!contents.includes("turn C (should be excluded)"));
});

test("18. REST generation in branch is completely isolated from parent", async () => {
  const conv = await createConversationWithMessages([
    { role: "user", content: "root" },
  ]);

  const branch = await request(app)
    .post(`/api/chat/conversations/${conv.id}/branch`)
    .send({ messageId: conv.messages[0].id });

  // Generate in branch
  const reply = await request(app)
    .post(`/api/chat/conversations/${branch.body.id}/messages`)
    .send({ content: "only for branch" });

  assert.equal(reply.status, 200);

  // Check branch has new turns
  const branchAfter = await request(app).get(`/api/chat/conversations/${branch.body.id}`);
  assert.equal(branchAfter.body.messages.length, 3); // root user + branch user + branch assistant

  // Check parent does NOT have the branch turns
  const parentAfter = await request(app).get(`/api/chat/conversations/${conv.id}`);
  assert.equal(parentAfter.body.messages.length, 2); // root user + root assistant
  assert.ok(!parentAfter.body.messages.some((m) => m.content === "only for branch"));
});

test("19. Concurrent branch creation produces isolated branches", async () => {
  const conv = await createConversationWithMessages([
    { role: "user", content: "concurrent root" },
  ]);
  const msgId = conv.messages[0].id;

  const [b1, b2, b3] = await Promise.all([
    request(app).post(`/api/chat/conversations/${conv.id}/branch`).send({ messageId: msgId }),
    request(app).post(`/api/chat/conversations/${conv.id}/branch`).send({ messageId: msgId }),
    request(app).post(`/api/chat/conversations/${conv.id}/branch`).send({ messageId: msgId }),
  ]);

  assert.equal(b1.status, 200);
  assert.equal(b2.status, 200);
  assert.equal(b3.status, 200);

  const ids = new Set([b1.body.id, b2.body.id, b3.body.id]);
  assert.equal(ids.size, 3, "Each concurrent branch must have a unique ID");

  // Appending to b1 does not affect b2 or b3
  await request(app)
    .post(`/api/chat/conversations/${b1.body.id}/messages`)
    .send({ content: "b1 unique message" });

  const b1After = await request(app).get(`/api/chat/conversations/${b1.body.id}`);
  const b2After = await request(app).get(`/api/chat/conversations/${b2.body.id}`);
  const b3After = await request(app).get(`/api/chat/conversations/${b3.body.id}`);

  assert.equal(b1After.body.messages.length, 3);
  assert.equal(b2After.body.messages.length, 1);
  assert.equal(b3After.body.messages.length, 1);
});

test("20. Branch deletion behavior", async () => {
  const conv = await createConversationWithMessages([
    { role: "user", content: "to delete test" },
  ]);

  const b1 = await request(app)
    .post(`/api/chat/conversations/${conv.id}/branch`)
    .send({ messageId: conv.messages[0].id });

  const b2 = await request(app)
    .post(`/api/chat/conversations/${conv.id}/branch`)
    .send({ messageId: conv.messages[0].id });

  // 1. Deleting b1 must not delete parent conv or sibling b2
  const delB1 = await request(app).delete(`/api/chat/conversations/${b1.body.id}`);
  assert.equal(delB1.status, 200);

  const getB1 = await request(app).get(`/api/chat/conversations/${b1.body.id}`);
  assert.equal(getB1.status, 404);

  const getParent = await request(app).get(`/api/chat/conversations/${conv.id}`);
  assert.equal(getParent.status, 200);

  const getB2 = await request(app).get(`/api/chat/conversations/${b2.body.id}`);
  assert.equal(getB2.status, 200);

  // 2. Deleting parent conv does not delete b2 (safe non-cascading policy)
  const delParent = await request(app).delete(`/api/chat/conversations/${conv.id}`);
  assert.equal(delParent.status, 200);

  const getB2AfterParentDel = await request(app).get(`/api/chat/conversations/${b2.body.id}`);
  assert.equal(getB2AfterParentDel.status, 200);
});
