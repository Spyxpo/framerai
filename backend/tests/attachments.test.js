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

const { resolveAttachments } = require("../src/services/model");

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
