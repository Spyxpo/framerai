/**
 * Stored XSS through uploaded SVG files (Issue #350).
 *
 * Root cause: uploads are served from the application's own origin by
 * express.static, which derives the Content-Type from the stored extension. An
 * accepted image/svg+xml upload is stored as .svg and served as image/svg+xml,
 * and an SVG may carry a <script> element, so navigating to an uploaded URL ran
 * attacker JavaScript in the app's origin — with access to same-origin storage
 * and API routes.
 *
 * #342 is a separate defect and stays fixed: it stopped the client-supplied
 * filename from choosing the stored extension. It did not stop image/svg+xml
 * from being stored, legitimately, as .svg.
 *
 * Fix: the /uploads static handler sends Content-Disposition: attachment and
 * X-Content-Type-Options: nosniff on every response, so a navigation downloads
 * the file rather than rendering it and no response is sniffed into something
 * active. Content-Disposition binds only top-level navigations, so <img>,
 * <audio> and <video> subresource loads still render inline — which is why no
 * upload type has to be dropped to close this.
 *
 * These tests verify that:
 *   - an uploaded .svg is served with both headers,
 *   - both headers are sent whatever the extension or content type,
 *   - the file's bytes are served unchanged,
 *   - a real upload through /api/generate/upload is served the same way,
 *   - upload acceptance and the #342 MIME→extension mapping are unchanged,
 *   - resolveAttachments still reads uploaded files server-side,
 *   - the headers are scoped to /uploads and a missing file still 404s.
 */

const test = require("node:test");
const { after } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const request = require("supertest");

const { mockModel, loadApp } = require("./helpers");

mockModel();
const app = loadApp();

// mockModel assigns over the model module's exports in place, so the real
// resolveAttachments survives and this is the same function the routes call.
const { resolveAttachments } = require("../src/services/model");

const uploadsRoot = path.join(__dirname, "..", "uploads");

// The payload from the issue: an SVG that executes script on navigation.
const SVG_XSS_PAYLOAD =
  '<?xml version="1.0" encoding="UTF-8"?>\n' +
  '<svg xmlns="http://www.w3.org/2000/svg">\n' +
  '  <script>document.body.innerHTML = "XSS";</script>\n' +
  "</svg>\n";

// Files seeded straight into the uploads tree, and paths returned by a real
// upload. Both are removed at the end of the run.
const seeded = [];
const uploaded = [];

function seed(bucket, name, contents) {
  const dir = path.join(uploadsRoot, bucket);
  fs.mkdirSync(dir, { recursive: true });
  const full = path.join(dir, name);
  fs.writeFileSync(full, contents);
  seeded.push(full);
  return `/uploads/${bucket}/${name}`;
}

after(() => {
  for (const full of seeded) fs.rmSync(full, { force: true });
  for (const relative of uploaded) {
    fs.rmSync(path.join(__dirname, "..", relative), { force: true });
  }
});

// ---------------------------------------------------------------------------
// The vulnerability
// ---------------------------------------------------------------------------

test("an uploaded SVG is served as a download, not as renderable content", async () => {
  const url = seed("images", "xss-350.svg", SVG_XSS_PAYLOAD);

  const res = await request(app).get(url);

  assert.equal(res.status, 200);
  assert.equal(
    res.headers["content-disposition"],
    "attachment",
    "an SVG must download rather than render, or its <script> runs in our origin"
  );
  assert.equal(res.headers["x-content-type-options"], "nosniff");
});

// ---------------------------------------------------------------------------
// Applied uniformly, not per type
// ---------------------------------------------------------------------------

// Guessing which types are "active content" is how this class of bug comes
// back, so the headers go on every response regardless of extension.
const CASES = [
  ["images", "case-350.svg", SVG_XSS_PAYLOAD],
  ["images", "case-350.png", Buffer.from([0x89, 0x50, 0x4e, 0x47])],
  ["images", "case-350.gif", Buffer.from("GIF89a")],
  ["images", "case-350.webp", Buffer.from("RIFF____WEBP")],
  ["images", "case-350.bin", Buffer.from("opaque bytes")],
  ["documents", "case-350.pdf", "%PDF-1.4"],
  ["documents", "case-350.txt", "plain text"],
  ["documents", "case-350.md", "# heading"],
  ["audio", "case-350.wav", Buffer.from("RIFF____WAVE")],
];

test("both headers are sent whatever the extension or content type", async () => {
  for (const [bucket, name, contents] of CASES) {
    const url = seed(bucket, name, contents);

    const res = await request(app).get(url);

    assert.equal(res.status, 200, `${name} should still be served`);
    assert.equal(res.headers["content-disposition"], "attachment", `${name} must be a download`);
    assert.equal(res.headers["x-content-type-options"], "nosniff", `${name} must not be sniffed`);
  }
});

test("the file's bytes are served unchanged — only how a browser treats them changes", async () => {
  const url = seed("images", "bytes-350.svg", SVG_XSS_PAYLOAD);

  // buffer(true) because superagent does not populate res.text for
  // image/svg+xml; the payload arrives in res.body as a Buffer.
  const res = await request(app).get(url).buffer(true);

  assert.equal(res.status, 200);
  assert.ok(Buffer.isBuffer(res.body), "the response body should arrive as raw bytes");
  assert.equal(
    res.body.toString("utf8"),
    SVG_XSS_PAYLOAD,
    "the fix must not rewrite or strip file contents"
  );
});

// ---------------------------------------------------------------------------
// End to end, through the real upload route
// ---------------------------------------------------------------------------

test("a real SVG upload is served back with both headers", async () => {
  const upload = await request(app)
    .post("/api/generate/upload")
    .attach("file", Buffer.from(SVG_XSS_PAYLOAD), {
      filename: "innocent.png",
      contentType: "image/svg+xml",
    });

  assert.equal(upload.status, 201, "SVG uploads must still be accepted");
  // Also pins #342: the extension comes from the MIME type, not "innocent.png".
  assert.match(upload.body.path, /\.svg$/, "stored extension must come from the MIME type");
  uploaded.push(upload.body.path);

  const res = await request(app).get(upload.body.path);

  assert.equal(res.status, 200);
  assert.equal(res.headers["content-disposition"], "attachment");
  assert.equal(res.headers["x-content-type-options"], "nosniff");
});

test("server-side attachment processing still reads uploaded files", async () => {
  const upload = await request(app)
    .post("/api/generate/upload")
    .attach("file", Buffer.from(SVG_XSS_PAYLOAD), {
      filename: "diagram.svg",
      contentType: "image/svg+xml",
    });

  assert.equal(upload.status, 201);
  uploaded.push(upload.body.path);

  // resolveAttachments opens the stored file from disk and never goes through
  // express.static, so response headers cannot reach it. This is the guard that
  // the mitigation is delivery-only.
  const resolved = resolveAttachments([upload.body.path]);

  assert.equal(resolved.length, 1, "an uploaded SVG must still resolve for the worker");
  assert.equal(resolved[0].kind, "image");
  assert.ok(fs.existsSync(resolved[0].path), "the resolved path must point at the stored file");
  assert.equal(
    fs.readFileSync(resolved[0].path, "utf8"),
    SVG_XSS_PAYLOAD,
    "the worker must still see the exact stored bytes"
  );
});

// ---------------------------------------------------------------------------
// Scope and existing behaviour
// ---------------------------------------------------------------------------

test("the headers are scoped to /uploads and do not leak onto API responses", async () => {
  const res = await request(app).get("/api/health");

  assert.equal(res.status, 200);
  assert.equal(
    res.headers["content-disposition"],
    undefined,
    "API JSON must not be turned into a download"
  );
});

test("a missing upload still 404s", async () => {
  const res = await request(app).get("/uploads/images/absent-350.svg");

  assert.equal(res.status, 404);
});
