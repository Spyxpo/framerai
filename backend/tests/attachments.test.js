/**
 * Attachment resolution: what a client may point the worker at.
 *
 * Attachments arrive as client-supplied strings and are turned into filesystem
 * paths the worker will open, so the containment check is the whole point of
 * this file. Only files this server stored are addressable, and a reference
 * that fails any check costs the caller its attachment rather than the request.
 */

const test = require("node:test");
const { after } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const bridge = require("../src/services/pythonBridge");
const { resolveAttachments, processMessage } = require("../src/services/model");

const uploadsRoot = path.join(__dirname, "..", "uploads");
const imageName = "attachment-test.png";
const documentName = "attachment-test.pdf";
const imagePath = path.join(uploadsRoot, "images", imageName);
const documentPath = path.join(uploadsRoot, "documents", documentName);

fs.mkdirSync(path.dirname(imagePath), { recursive: true });
fs.mkdirSync(path.dirname(documentPath), { recursive: true });
fs.writeFileSync(imagePath, "not really a png");
fs.writeFileSync(documentPath, "%PDF-1.4");

after(() => {
  fs.rmSync(imagePath, { force: true });
  fs.rmSync(documentPath, { force: true });
});

test("a stored upload resolves to an absolute path and its kind", () => {
  const resolved = resolveAttachments([`/uploads/images/${imageName}`]);
  assert.equal(resolved.length, 1);
  assert.equal(resolved[0].kind, "image");
  assert.equal(resolved[0].path, imagePath);
});

test("the object form is accepted alongside the string form", () => {
  const resolved = resolveAttachments([{ path: `/uploads/documents/${documentName}` }]);
  assert.deepEqual(resolved, [{ path: documentPath, kind: "document" }]);
});

test("a traversal out of the uploads root is refused", () => {
  const escapes = [
    "/uploads/../../../etc/passwd",
    "/uploads/images/../../../../etc/passwd",
    "/uploads/images/../../package.json",
  ];
  for (const reference of escapes) {
    assert.deepEqual(resolveAttachments([reference]), [], `should refuse ${reference}`);
  }
});

test("a path that is not an upload is refused", () => {
  assert.deepEqual(resolveAttachments(["/etc/passwd"]), []);
  assert.deepEqual(resolveAttachments(["uploads/images/x.png"]), []);
  assert.deepEqual(resolveAttachments(["https://example.com/x.png"]), []);
});

test("an unknown bucket is refused even when the file exists", () => {
  assert.deepEqual(resolveAttachments(["/uploads/attachment-test.png"]), []);
});

test("a reference to a file that is not there is dropped", () => {
  assert.deepEqual(resolveAttachments(["/uploads/images/absent.png"]), []);
});

test("one bad reference does not lose the good ones", () => {
  const resolved = resolveAttachments([
    "/uploads/images/absent.png",
    `/uploads/images/${imageName}`,
    "/uploads/../secrets",
    `/uploads/documents/${documentName}`,
  ]);
  assert.deepEqual(
    resolved.map((a) => a.kind),
    ["image", "document"]
  );
});

test("no attachments is not an error", () => {
  assert.deepEqual(resolveAttachments(undefined), []);
  assert.deepEqual(resolveAttachments([]), []);
  assert.deepEqual(resolveAttachments("not an array"), []);
  assert.deepEqual(resolveAttachments([null, 42, {}]), []);
});

// One file, however many times it is listed (Issue #392).
//
// Every reference the worker receives is a full read of the file and another
// copy of its text in the prompt, so a repeat costs as much as a distinct file
// and carries nothing. A message may list ten, which is ten reads of one upload.

test("a file listed twice is resolved once", () => {
  const resolved = resolveAttachments([`/uploads/images/${imageName}`, `/uploads/images/${imageName}`]);
  assert.deepEqual(resolved, [{ path: imagePath, kind: "image" }]);
});

test("different spellings of one path are one file", () => {
  const resolved = resolveAttachments([
    `/uploads/documents/${documentName}`,
    { path: `/uploads/documents/${documentName}` },
    `/uploads/documents/./${documentName}`,
    `/uploads/documents//${documentName}`,
    `/uploads/images/../documents/${documentName}`,
  ]);
  assert.deepEqual(resolved, [{ path: documentPath, kind: "document" }]);
});

test("repeats keep the order of first appearance and displace nothing else", () => {
  const image = `/uploads/images/${imageName}`;
  const document = `/uploads/documents/${documentName}`;
  const resolved = resolveAttachments([document, image, document, image, document]);
  assert.deepEqual(
    resolved.map((a) => a.kind),
    ["document", "image"]
  );
});

test("a refused reference is refused every time it is listed", () => {
  const escape = "/uploads/../../../etc/passwd";
  assert.deepEqual(resolveAttachments([escape, escape]), []);

  const absent = "/uploads/images/absent.png";
  const resolved = resolveAttachments([absent, absent, `/uploads/images/${imageName}`]);
  assert.deepEqual(resolved, [{ path: imagePath, kind: "image" }]);
});

test("ten references to one upload reach the worker as one attachment", async () => {
  const origAvailable = bridge.available;
  const origRequest = bridge.request;
  const requests = [];
  bridge.available = () => true;
  bridge.request = async (op, params) => {
    requests.push({ op, params });
    return { content: "a reply", finish_reason: "stop" };
  };

  try {
    const reference = `/uploads/documents/${documentName}`;
    const reply = await processMessage(
      [{ role: "user", content: "Summarise the attached file.", attachments: Array(10).fill(reference) }],
      "text",
      {},
      "req-392"
    );

    assert.equal(requests.length, 1);
    assert.equal(requests[0].op, "chat");
    assert.deepEqual(requests[0].params.attachments, [{ path: documentPath, kind: "document" }]);
    // The reply reports what reached the model, so it reports one, not ten.
    assert.deepEqual(reply.metadata.attachments, ["document"]);
  } finally {
    bridge.available = origAvailable;
    bridge.request = origRequest;
  }
});
