const test = require("node:test");
const assert = require("node:assert/strict");
const request = require("supertest");

const { mockModel, loadApp, newConversation } = require("./helpers");

const calls = mockModel();
const app = loadApp();

test("a conversation can be created, listed, read, and deleted", async () => {
  const created = await request(app).post("/api/chat/conversations");
  assert.equal(created.status, 200);
  assert.equal(created.body.title, "New Chat");
  assert.deepEqual(created.body.messages, []);

  const list = await request(app).get("/api/chat/conversations");
  assert.equal(list.status, 200);
  assert.ok(list.body.some((c) => c.id === created.body.id));

  const read = await request(app).get(`/api/chat/conversations/${created.body.id}`);
  assert.equal(read.status, 200);
  assert.equal(read.body.id, created.body.id);

  const deleted = await request(app).delete(`/api/chat/conversations/${created.body.id}`);
  assert.equal(deleted.status, 200);

  const gone = await request(app).get(`/api/chat/conversations/${created.body.id}`);
  assert.equal(gone.status, 404);
  assert.equal(gone.body.code, "NOT_FOUND");
});

test("a message is answered and both turns are stored", async () => {
  const id = await newConversation(app);
  const before = calls.length;

  const res = await request(app)
    .post(`/api/chat/conversations/${id}/messages`)
    .send({ content: "hello there" });

  assert.equal(res.status, 200);
  assert.equal(res.body.role, "assistant");
  assert.equal(res.body.content, "reply to: hello there");

  const [call] = calls.slice(before);
  assert.equal(call.name, "processMessage");
  assert.equal(call.args[1], "text");
  assert.deepEqual(call.args[2], {}, "no settings sent means none forwarded");

  const conv = await request(app).get(`/api/chat/conversations/${id}`);
  assert.equal(conv.body.messages.length, 2);
  assert.equal(conv.body.messages[0].role, "user");
  assert.equal(conv.body.messages[1].role, "assistant");
});

test("the conversation title comes from the first message", async () => {
  const id = await newConversation(app);
  const content = "x".repeat(60);

  await request(app).post(`/api/chat/conversations/${id}/messages`).send({ content });

  const list = await request(app).get("/api/chat/conversations");
  const conv = list.body.find((c) => c.id === id);
  assert.equal(conv.title, `${"x".repeat(50)}...`);
});

test("generation settings sent with a message reach the model", async () => {
  const id = await newConversation(app);
  const before = calls.length;

  const res = await request(app)
    .post(`/api/chat/conversations/${id}/messages`)
    .send({ content: "hi", settings: { temperature: 1.4, max_new_tokens: 128 } });

  assert.equal(res.status, 200);
  const [call] = calls.slice(before);
  assert.deepEqual(call.args[2], { temperature: 1.4, max_new_tokens: 128 });
});

test("a message with no content is rejected, not treated as a server error", async () => {
  const id = await newConversation(app);

  const res = await request(app).post(`/api/chat/conversations/${id}/messages`).send({});

  assert.equal(res.status, 400);
  assert.equal(res.body.code, "VALIDATION_ERROR");
  assert.deepEqual(res.body.details, [{ field: "content", message: "is required" }]);
});

test("an unknown message type is rejected with the allowed values", async () => {
  const id = await newConversation(app);

  const res = await request(app)
    .post(`/api/chat/conversations/${id}/messages`)
    .send({ content: "hi", type: "hologram" });

  assert.equal(res.status, 400);
  assert.equal(res.body.details[0].field, "type");
  assert.match(res.body.details[0].message, /text, code, image, video, audio/);
});

test("a malformed conversation id is a validation error, not a 404", async () => {
  const res = await request(app).get("/api/chat/conversations/not-a-uuid");

  assert.equal(res.status, 400);
  assert.deepEqual(res.body.details, [{ field: "id", message: "must be a valid id" }]);
});

test("messages to a missing conversation are 404", async () => {
  const res = await request(app)
    .post("/api/chat/conversations/11111111-1111-1111-1111-111111111111/messages")
    .send({ content: "hi" });

  assert.equal(res.status, 404);
  assert.equal(res.body.code, "NOT_FOUND");
});
