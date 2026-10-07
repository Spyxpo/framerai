/**
 * Multipart form-field limits on the four upload routes (Issue #440).
 *
 * express.json's jsonBodyLimit never applies to these routes: Express picks a
 * body parser by Content-Type, and a multipart request goes to multer/busboy
 * instead. The four multer configurations in generate.js bounded the upload's
 * own size and count (fileSize, files) but not the request's other fields, so
 * a request with no file at all but many large text fields could still make
 * multer buffer an unbounded amount of memory — bounded only by Busboy's own
 * unconfigured defaults, which cap one field's size but not how many of them a
 * request may carry.
 *
 * Each route now gets `fields` set to the number of non-file fields its own
 * handler actually reads (0 for /upload, 1 for /understand and /transcribe, 2
 * for /document — see openapi.js, which documents the same set), and a shared
 * `fieldSize`, so a request that exceeds either is rejected rather than
 * buffered. The field-size override below is small so these tests do not need
 * to send megabytes of data; the field-count limits are the real ones from
 * generate.js, unchanged by it.
 */

process.env.MAX_MULTIPART_FIELD_SIZE = "200";
process.env.GENERATE_RATE_LIMIT_MAX = "500";
process.env.RATE_LIMIT_MAX = "1000";

const test = require("node:test");
const { after } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const request = require("supertest");

const { mockModel, loadApp } = require("./helpers");

mockModel();
const app = loadApp();

// Uploads land in the real uploads directory, same cleanup as generate.test.js.
const uploaded = [];
after(() => {
  for (const relative of uploaded) {
    fs.rmSync(path.join(__dirname, "..", relative), { force: true });
  }
});

function attachImage(req) {
  return req.attach("image", Buffer.from("fake image"), { filename: "pic.png", contentType: "image/png" });
}
function attachAudio(req) {
  return req.attach("audio", Buffer.from("fake audio"), { filename: "clip.wav", contentType: "audio/wav" });
}
function attachDocument(req) {
  return req.attach("document", Buffer.from("%PDF-1.4 fake"), { filename: "report.pdf", contentType: "application/pdf" });
}
function attachFile(req) {
  return req.attach("file", Buffer.from("image content"), { filename: "sample.png", contentType: "image/png" });
}

// ── field value size ────────────────────────────────────────────────────────
// MAX_MULTIPART_FIELD_SIZE is 200 bytes above. Busboy can only tell a field is
// truncated once it has read exactly fieldSize bytes of it and there turns out
// to be more, so a field of exactly that many bytes is indistinguishable from
// one cut short at it and is rejected the same way; the largest value actually
// accepted is one byte under the limit.

test("/understand rejects a prompt over the field-size limit", async () => {
  const res = await attachImage(request(app).post("/api/generate/understand")).field("prompt", "x".repeat(201));

  assert.equal(res.status, 400);
  assert.equal(res.body.code, "UPLOAD_ERROR");
  assert.equal(res.body.error, "Field value too long");
  assert.deepEqual(res.body.details, [{ field: "prompt", message: "Field value too long" }]);
});

test("/understand rejects a prompt of exactly the field-size limit", async () => {
  const res = await attachImage(request(app).post("/api/generate/understand")).field("prompt", "x".repeat(200));

  assert.equal(res.status, 400);
  assert.equal(res.body.code, "UPLOAD_ERROR");
  assert.equal(res.body.error, "Field value too long");
});

test("/understand accepts a prompt one byte under the field-size limit", async () => {
  const res = await attachImage(request(app).post("/api/generate/understand")).field("prompt", "x".repeat(199));

  assert.equal(res.status, 200);
  uploaded.push(res.body.imagePath);
});

test("/transcribe rejects a prompt over the field-size limit", async () => {
  const res = await attachAudio(request(app).post("/api/generate/transcribe")).field("prompt", "x".repeat(201));

  assert.equal(res.status, 400);
  assert.equal(res.body.code, "UPLOAD_ERROR");
  assert.equal(res.body.error, "Field value too long");
});

test("/document rejects a prompt over the field-size limit", async () => {
  const res = await attachDocument(request(app).post("/api/generate/document")).field("prompt", "x".repeat(201));

  assert.equal(res.status, 400);
  assert.equal(res.body.code, "UPLOAD_ERROR");
  assert.equal(res.body.error, "Field value too long");
});

// ── field count ──────────────────────────────────────────────────────────────
// /upload's handler reads no fields at all; /understand and /transcribe each
// read one (prompt); /document reads two (prompt, max_pages) — generate.js and
// openapi.js agree on this set, and the website's only two multipart callers
// (uploadAttachment: no fields; transcribe: prompt only) never need more.

test("/upload rejects a request that carries any non-file field", async () => {
  const res = await attachFile(request(app).post("/api/generate/upload")).field("extra", "x");

  assert.equal(res.status, 400);
  assert.equal(res.body.code, "UPLOAD_ERROR");
  assert.equal(res.body.error, "Too many fields");
});

test("/upload still accepts a file with no fields at all", async () => {
  const res = await attachFile(request(app).post("/api/generate/upload"));

  assert.equal(res.status, 201);
  uploaded.push(res.body.path);
});

test("/understand rejects a second field alongside prompt", async () => {
  const res = await attachImage(request(app).post("/api/generate/understand"))
    .field("prompt", "describe this")
    .field("extra", "x");

  assert.equal(res.status, 400);
  assert.equal(res.body.code, "UPLOAD_ERROR");
  assert.equal(res.body.error, "Too many fields");
});

test("/understand still accepts exactly its one documented field", async () => {
  const res = await attachImage(request(app).post("/api/generate/understand")).field("prompt", "describe this");

  assert.equal(res.status, 200);
  uploaded.push(res.body.imagePath);
});

test("/transcribe rejects a second field alongside prompt", async () => {
  const res = await attachAudio(request(app).post("/api/generate/transcribe"))
    .field("prompt", "transcribe this")
    .field("extra", "x");

  assert.equal(res.status, 400);
  assert.equal(res.body.code, "UPLOAD_ERROR");
  assert.equal(res.body.error, "Too many fields");
});

test("/transcribe still accepts exactly its one documented field", async () => {
  const res = await attachAudio(request(app).post("/api/generate/transcribe")).field("prompt", "transcribe this");

  assert.equal(res.status, 200);
  uploaded.push(res.body.audioPath);
});

test("/transcribe still accepts no fields at all (prompt is optional)", async () => {
  const res = await attachAudio(request(app).post("/api/generate/transcribe"));

  assert.equal(res.status, 200);
  uploaded.push(res.body.audioPath);
});

test("/document rejects a third field alongside prompt and max_pages", async () => {
  const res = await attachDocument(request(app).post("/api/generate/document"))
    .field("prompt", "summarize")
    .field("max_pages", "5")
    .field("extra", "x");

  assert.equal(res.status, 400);
  assert.equal(res.body.code, "UPLOAD_ERROR");
  assert.equal(res.body.error, "Too many fields");
});

test("/document still accepts exactly its two documented fields", async () => {
  const res = await attachDocument(request(app).post("/api/generate/document"))
    .field("prompt", "summarize")
    .field("max_pages", "5");

  assert.equal(res.status, 200);
  uploaded.push(res.body.documentPath);
});

test("/document still accepts just one of its two documented fields", async () => {
  const res = await attachDocument(request(app).post("/api/generate/document")).field("prompt", "summarize");

  assert.equal(res.status, 200);
  uploaded.push(res.body.documentPath);
});

// ── the limits are independent per route ────────────────────────────────────
// Each multer instance is its own object; raising what one route's handler
// happens to read must not loosen another's.

test("/document's extra field allowance does not carry over to /understand", async () => {
  const res = await attachImage(request(app).post("/api/generate/understand"))
    .field("prompt", "describe this")
    .field("max_pages", "5"); // a field /document reads, /understand does not

  assert.equal(res.status, 400);
  assert.equal(res.body.code, "UPLOAD_ERROR");
  assert.equal(res.body.error, "Too many fields");
});
