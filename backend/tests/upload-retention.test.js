/**
 * Upload retention (Issue #367).
 *
 * Root cause: the upload routes wrote files under uploads/ and nothing ever
 * removed them. There was no TTL, no size ceiling and no reaper, so disk grew
 * without bound — at the generation rate limit roughly a gigabyte a minute,
 * retained forever — and a stored file outlived the conversation that
 * referenced it.
 *
 * Retention model under test: age, with a total-size ceiling. References are
 * not tracked, because they cannot be determined reliably (attachment paths are
 * resolved on demand from client strings, the server store is in-memory and
 * LRU/TTL evicted, and the website keeps its own copy in localStorage). The
 * invariant these tests pin is therefore:
 *
 *   a managed upload older than the TTL is reclaimed, a younger one is kept,
 *   and when the buckets exceed the ceiling the oldest go first.
 *
 * Determinism: sweep() takes an injected `now` and the tests set mtimes with
 * utimesSync, so nothing waits on real time. Limits come from _resetLimits().
 *
 * Every file these tests create is removed again in the after() hook, including
 * on failure, so no upload artefacts are left behind.
 */

const test = require("node:test");
const { after } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { randomUUID } = require("node:crypto");
const request = require("supertest");

const { mockModel, loadApp } = require("./helpers");

mockModel();
const app = loadApp();

const retention = require("../src/services/uploadRetention");
const { resolveAttachments } = require("../src/services/model");
const conversations = require("../src/conversationStore");

const UPLOADS_ROOT = path.join(__dirname, "..", "uploads");
const HOUR = 60 * 60 * 1000;

// Everything this file puts on disk, removed in after().
const created = [];

/** Write a file with a managed <uuid><ext> name into a bucket. */
function makeManaged(bucket, { ext = ".png", bytes = 1024, ageMs = 0, now = Date.now() } = {}) {
  const dir = path.join(UPLOADS_ROOT, bucket);
  fs.mkdirSync(dir, { recursive: true });
  const full = path.join(dir, `${randomUUID()}${ext}`);
  fs.writeFileSync(full, Buffer.alloc(bytes, 0x41));
  const when = new Date(now - ageMs);
  fs.utimesSync(full, when, when);
  created.push(full);
  return full;
}

/** Write a file whose name is NOT one the upload routes would generate. */
function makeUnmanaged(bucket, name, { ageMs = 0, now = Date.now() } = {}) {
  const dir = path.join(UPLOADS_ROOT, bucket);
  fs.mkdirSync(dir, { recursive: true });
  const full = path.join(dir, name);
  fs.writeFileSync(full, Buffer.alloc(16, 0x42));
  const when = new Date(now - ageMs);
  fs.utimesSync(full, when, when);
  created.push(full);
  return full;
}

after(() => {
  for (const full of created) {
    try {
      if (fs.existsSync(full)) fs.rmSync(full, { force: true, recursive: true });
    } catch {
      // best effort; never fail the run on cleanup
    }
  }
  retention._resetLimits();
});

// ---------------------------------------------------------------------------
// 1 & 2. Stale files go, fresh files stay
// ---------------------------------------------------------------------------

test("a stale managed upload is reclaimed and a fresh one is kept", async (t) => {
  const now = Date.now();
  retention._resetLimits({ ttlMs: 6 * HOUR, maxBytes: Infinity });
  t.after(() => retention._resetLimits());

  const stale = makeManaged("images", { ageMs: 7 * HOUR, now });
  const fresh = makeManaged("images", { ageMs: 1 * HOUR, now });

  const stats = await retention.sweep({ now });

  assert.equal(fs.existsSync(stale), false, "a file past the TTL must be reclaimed");
  assert.equal(fs.existsSync(fresh), true, "a file inside the TTL must be kept");
  assert.ok(stats.removedStale >= 1, `expected a stale removal, got ${JSON.stringify(stats)}`);
});

test("the configured TTL is what decides, not a hard-coded period", async (t) => {
  const now = Date.now();
  const file = () => makeManaged("images", { ageMs: 3 * HOUR, now });

  // With a 2 hour TTL the same age is stale.
  retention._resetLimits({ ttlMs: 2 * HOUR, maxBytes: Infinity });
  t.after(() => retention._resetLimits());
  const a = file();
  await retention.sweep({ now });
  assert.equal(fs.existsSync(a), false, "3h old must be reclaimed under a 2h TTL");

  // With a 10 hour TTL it is not.
  retention._resetLimits({ ttlMs: 10 * HOUR, maxBytes: Infinity });
  const b = file();
  await retention.sweep({ now });
  assert.equal(fs.existsSync(b), true, "3h old must survive a 10h TTL");
});

test("every managed bucket is swept, not just images", async (t) => {
  const now = Date.now();
  retention._resetLimits({ ttlMs: 1 * HOUR, maxBytes: Infinity });
  t.after(() => retention._resetLimits());

  const files = {
    images: makeManaged("images", { ext: ".png", ageMs: 5 * HOUR, now }),
    audio: makeManaged("audio", { ext: ".wav", ageMs: 5 * HOUR, now }),
    documents: makeManaged("documents", { ext: ".pdf", ageMs: 5 * HOUR, now }),
  };

  await retention.sweep({ now });

  for (const [bucket, full] of Object.entries(files)) {
    assert.equal(fs.existsSync(full), false, `${bucket} must be swept`);
  }
});

// ---------------------------------------------------------------------------
// 3. Orphaned uploads — no conversation involved at all
// ---------------------------------------------------------------------------

test("an orphaned upload with no conversation is reclaimed once stale", async (t) => {
  retention._resetLimits({ ttlMs: 6 * HOUR, maxBytes: Infinity });
  t.after(() => retention._resetLimits());

  // A real upload through the route, never attached to anything. This is the
  // /understand, /document and /transcribe shape: a file with no conversation.
  const res = await request(app)
    .post("/api/generate/upload")
    .attach("file", Buffer.alloc(2048, 0x43), { filename: "orphan.png", contentType: "image/png" });

  assert.equal(res.status, 201, "upload behaviour must be unchanged");
  const full = path.join(__dirname, "..", res.body.path);
  created.push(full);
  assert.equal(fs.existsSync(full), true, "the upload landed on disk");

  // Age it past the TTL, then sweep.
  const now = Date.now();
  const old = new Date(now - 7 * HOUR);
  fs.utimesSync(full, old, old);

  await retention.sweep({ now });

  assert.equal(fs.existsSync(full), false, "an orphan past the TTL must be reclaimed");
});

// ---------------------------------------------------------------------------
// 4. A file a live conversation still points at
// ---------------------------------------------------------------------------

test("a fresh upload referenced by a live conversation is not reclaimed", async (t) => {
  const now = Date.now();
  retention._resetLimits({ ttlMs: 6 * HOUR, maxBytes: Infinity });
  t.after(() => retention._resetLimits());

  const res = await request(app)
    .post("/api/generate/upload")
    .attach("file", Buffer.alloc(1024, 0x44), { filename: "kept.png", contentType: "image/png" });
  const full = path.join(__dirname, "..", res.body.path);
  created.push(full);

  const conv = await request(app).post("/api/chat/conversations");
  await request(app)
    .post(`/api/chat/conversations/${conv.body.id}/messages`)
    .send({ content: "see this", attachments: [res.body.path] });

  await retention.sweep({ now });

  assert.equal(fs.existsSync(full), true, "a fresh referenced attachment must survive");
  // And it still resolves for the worker, so the sweep did not break the path.
  const resolved = resolveAttachments([res.body.path]);
  assert.equal(resolved.length, 1, "resolveAttachments must still resolve it");
  assert.equal(resolved[0].kind, "image");
});

// Documents the deliberate limit of an age-based model: retention is by age, so
// an attachment older than the TTL is reclaimed even while its conversation
// lives. The contract is that this degrades rather than errors.
test("a stale attachment is reclaimed and resolveAttachments degrades cleanly", async (t) => {
  const now = Date.now();
  retention._resetLimits({ ttlMs: 1 * HOUR, maxBytes: Infinity });
  t.after(() => retention._resetLimits());

  const res = await request(app)
    .post("/api/generate/upload")
    .attach("file", Buffer.alloc(512, 0x45), { filename: "aged.png", contentType: "image/png" });
  const full = path.join(__dirname, "..", res.body.path);
  created.push(full);

  const conv = await request(app).post("/api/chat/conversations");
  await request(app)
    .post(`/api/chat/conversations/${conv.body.id}/messages`)
    .send({ content: "see this", attachments: [res.body.path] });

  const old = new Date(now - 5 * HOUR);
  fs.utimesSync(full, old, old);
  await retention.sweep({ now });

  assert.equal(fs.existsSync(full), false, "past the TTL it goes, by design");
  assert.ok(conversations.get(conv.body.id), "the conversation itself is untouched");
  // The documented degradation: dropped, not thrown.
  assert.deepEqual(
    resolveAttachments([res.body.path]),
    [],
    "a missing attachment is dropped, not an error"
  );
});

// ---------------------------------------------------------------------------
// 5. A file that disappears mid-sweep
// ---------------------------------------------------------------------------

test("a file removed between scan and unlink does not fail the sweep", async (t) => {
  const now = Date.now();
  retention._resetLimits({ ttlMs: 1 * HOUR, maxBytes: Infinity });
  t.after(() => retention._resetLimits());

  const a = makeManaged("images", { ageMs: 5 * HOUR, now });
  const b = makeManaged("images", { ageMs: 5 * HOUR, now });

  // Make the first unlink behave as if the file had already gone.
  const fsp = require("node:fs/promises");
  const realUnlink = fsp.unlink;
  let first = true;
  fsp.unlink = async (p) => {
    if (first) {
      first = false;
      await realUnlink(p); // actually remove it
      const err = new Error("ENOENT: no such file or directory");
      err.code = "ENOENT";
      throw err; // then report it as already gone
    }
    return realUnlink(p);
  };
  t.after(() => {
    fsp.unlink = realUnlink;
  });

  const stats = await retention.sweep({ now });

  assert.equal(fs.existsSync(a), false);
  assert.equal(fs.existsSync(b), false, "the sweep must continue past a vanished file");
  assert.equal(stats.removedStale, 2, "ENOENT counts as reclaimed, not as a failure");
});

test("an unlink that fails for another reason is reported, not thrown", async (t) => {
  const now = Date.now();
  retention._resetLimits({ ttlMs: 1 * HOUR, maxBytes: Infinity });
  t.after(() => retention._resetLimits());

  const stuck = makeManaged("images", { ageMs: 5 * HOUR, now });

  const fsp = require("node:fs/promises");
  const realUnlink = fsp.unlink;
  fsp.unlink = async () => {
    const err = new Error("EPERM: operation not permitted");
    err.code = "EPERM";
    throw err;
  };
  t.after(() => {
    fsp.unlink = realUnlink;
  });

  const stats = await retention.sweep({ now }); // must resolve, not reject

  assert.equal(stats.removedStale, 0, "a real failure is not counted as reclaimed");
  assert.equal(fs.existsSync(stuck), true, "and the file is still there");
});

// ---------------------------------------------------------------------------
// 6. Path safety
// ---------------------------------------------------------------------------

test("only managed names in managed buckets are touched", async (t) => {
  const now = Date.now();
  retention._resetLimits({ ttlMs: 1 * HOUR, maxBytes: Infinity });
  t.after(() => retention._resetLimits());

  // All of these are old enough to be stale if they were candidates.
  const gitkeep = path.join(UPLOADS_ROOT, "images", ".gitkeep");
  const hadGitkeep = fs.existsSync(gitkeep);
  const notUuid = makeUnmanaged("images", "operator-notes.png", { ageMs: 9 * HOUR, now });
  const noExt = makeUnmanaged("images", randomUUID(), { ageMs: 9 * HOUR, now });
  // uploads/generated holds model output that message metadata points at.
  const generated = makeManaged("generated", { ageMs: 9 * HOUR, now });
  const videos = makeManaged("videos", { ext: ".gif", ageMs: 9 * HOUR, now });
  // A directory inside a managed bucket must never be removed.
  const subdir = path.join(UPLOADS_ROOT, "images", `dir-${randomUUID()}`);
  fs.mkdirSync(subdir, { recursive: true });
  created.push(subdir);

  await retention.sweep({ now });

  assert.equal(fs.existsSync(notUuid), true, "a non-uuid name is not ours to remove");
  assert.equal(fs.existsSync(noExt), true, "a uuid with no extension is not our shape");
  assert.equal(fs.existsSync(generated), true, "uploads/generated is a different lifecycle");
  assert.equal(fs.existsSync(videos), true, "uploads/videos is a different lifecycle");
  assert.equal(fs.existsSync(subdir), true, "directories are never removed");
  if (hadGitkeep) {
    assert.equal(fs.existsSync(gitkeep), true, ".gitkeep must survive every sweep");
  }
});

test("the managed buckets and name shape are constrained", () => {
  assert.deepEqual(retention.MANAGED_BUCKETS, ["images", "audio", "documents"]);
  assert.ok(retention.UPLOADS_ROOT.endsWith(`${path.sep}uploads`));
  // The shape the upload routes actually produce.
  assert.ok(retention.MANAGED_NAME.test(`${randomUUID()}.png`));
  assert.ok(retention.MANAGED_NAME.test(`${randomUUID()}.bin`));
  // Anything else is out of scope.
  assert.equal(retention.MANAGED_NAME.test(".gitkeep"), false);
  assert.equal(retention.MANAGED_NAME.test("../escape.png"), false);
  assert.equal(retention.MANAGED_NAME.test("notes.png"), false);
  assert.equal(retention.MANAGED_NAME.test(`${randomUUID()}`), false);
});

// ---------------------------------------------------------------------------
// 7. The size ceiling — what actually bounds a burst
// ---------------------------------------------------------------------------

test("the size ceiling reclaims oldest-first until the buckets fit", async (t) => {
  const now = Date.now();
  const DAY = 24 * HOUR;

  // The buckets may already hold files from other tests or earlier runs, and
  // those are newer than anything backdated here. So: measure what is already
  // there, make this test's files unambiguously the oldest, disable the TTL so
  // only the ceiling can act, and set the ceiling to leave room for exactly two
  // of the three.
  const baseBytes = (await retention._listManaged()).reduce((sum, f) => sum + f.size, 0);
  retention._resetLimits({ ttlMs: Infinity, maxBytes: baseBytes + 8000 });
  t.after(() => retention._resetLimits());

  const oldest = makeManaged("images", { bytes: 4000, ageMs: 30 * DAY, now });
  const middle = makeManaged("images", { bytes: 4000, ageMs: 29 * DAY, now });
  const newest = makeManaged("images", { bytes: 4000, ageMs: 28 * DAY, now });

  const stats = await retention.sweep({ now });

  assert.equal(stats.removedStale, 0, "the TTL was disabled, so nothing was stale");
  assert.equal(stats.removedForCeiling, 1, "freeing 4000 bytes needs exactly one removal");
  assert.ok(
    stats.bytesAfter <= baseBytes + 8000,
    `expected <= ${baseBytes + 8000} bytes retained, got ${stats.bytesAfter}`
  );
  assert.equal(fs.existsSync(oldest), false, "the oldest is reclaimed first");
  assert.equal(fs.existsSync(middle), true, "the middle one survives");
  assert.equal(fs.existsSync(newest), true, "the newest survives");
});

test("a generous ceiling leaves fresh files alone", async (t) => {
  const now = Date.now();
  // TTL disabled and the ceiling set well above whatever is already on disk, so
  // neither rule can fire and the buckets' existing contents cannot skew this.
  const baseBytes = (await retention._listManaged()).reduce((sum, f) => sum + f.size, 0);
  retention._resetLimits({ ttlMs: Infinity, maxBytes: baseBytes + 10 * 1024 * 1024 });
  t.after(() => retention._resetLimits());

  const a = makeManaged("images", { bytes: 2048, ageMs: 1 * HOUR, now });
  const b = makeManaged("images", { bytes: 2048, ageMs: 2 * HOUR, now });

  const stats = await retention.sweep({ now });

  assert.equal(stats.removedStale, 0, "the TTL was disabled");
  assert.equal(stats.removedForCeiling, 0, "the ceiling had room");
  assert.equal(fs.existsSync(a), true);
  assert.equal(fs.existsSync(b), true);
});

// ---------------------------------------------------------------------------
// Throttling / no overlap
// ---------------------------------------------------------------------------

const settle = () => new Promise((r) => setTimeout(r, 60));

test("maybeSweep does not start a second sweep while one is running", async (t) => {
  retention._resetLimits({ ttlMs: 1 * HOUR, maxBytes: Infinity, minSweepIntervalMs: 0 });
  t.after(() => retention._resetLimits());

  const now = Date.now();
  // The interval is 0, so only the in-flight guard can refuse the second call.
  assert.equal(retention.maybeSweep({ now }), true, "the first call runs");
  assert.equal(retention.maybeSweep({ now }), false, "no overlapping sweep");

  await settle();
});

test("maybeSweep throttles to at most one sweep per interval", async (t) => {
  retention._resetLimits({ ttlMs: 1 * HOUR, maxBytes: Infinity, minSweepIntervalMs: 60_000 });
  t.after(() => retention._resetLimits());

  const now = Date.now();
  assert.equal(retention.maybeSweep({ now }), true, "the first call runs");
  await settle(); // let it finish so the in-flight guard is not what refuses

  assert.equal(retention.maybeSweep({ now: now + 1000 }), false, "still inside the interval");
  await settle();

  assert.equal(retention.maybeSweep({ now: now + 61_000 }), true, "past the interval it runs again");
  await settle();
});

test("an upload still succeeds while retention is active", async () => {
  retention._resetLimits(); // production defaults
  const res = await request(app)
    .post("/api/generate/upload")
    .attach("file", Buffer.alloc(256, 0x46), { filename: "still-works.png", contentType: "image/png" });

  assert.equal(res.status, 201);
  assert.match(res.body.path, /^\/uploads\/images\/[0-9a-f-]+\.png$/, "response shape unchanged");
  assert.equal(res.body.kind, "image");
  assert.equal(res.body.name, "still-works.png");
  created.push(path.join(__dirname, "..", res.body.path));
});
