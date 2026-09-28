/**
 * Upload retention.
 *
 * Files accepted by the upload routes were written under uploads/ and never
 * removed, so disk grew without bound: at the generation rate limit a client
 * could retain roughly a gigabyte a minute indefinitely, and a stored file
 * outlived the conversation that referenced it (Issue #367).
 *
 * Retention model
 * ---------------
 * Age, with a total-size ceiling. References cannot be determined reliably:
 * attachment paths are resolved on demand from client-supplied strings, the
 * server-side conversation store is in memory and LRU/TTL evicted, and the
 * website keeps its own copy of a conversation in localStorage. A reference
 * count built on those would be wrong in both directions, so this reclaims by
 * age and keeps one invariant:
 *
 *   a managed upload older than _ttlMs is reclaimed, a younger one is kept —
 *   unless the managed buckets together exceed _maxBytes, in which case the
 *   oldest are reclaimed until they fit.
 *
 * The ceiling is what bounds the adversarial case. A TTL on its own still
 * admits rate × TTL, which is terabytes over a day.
 *
 * A file reclaimed while its conversation is still alive degrades the way a
 * missing attachment already does: resolveAttachments drops it with a log line,
 * so the caller loses that attachment rather than the request.
 *
 * Only names matching the <uuid><ext> form the upload routes generate are
 * considered, so the .gitkeep files that hold the buckets in git — and anything
 * an operator put there by hand — are never touched. uploads/generated and
 * uploads/videos hold model output that message metadata still points at; they
 * are a separate lifecycle and are left alone.
 *
 * Sweeps are lazy (triggered by an upload arriving, never by a timer) and
 * throttled, mirroring conversationStore's eviction so behaviour is
 * deterministic and tests can drive it without fake timers.
 */

const fsp = require("node:fs/promises");
const path = require("node:path");

const config = require("../config");
const { logger } = require("./logger");

const UPLOADS_ROOT = path.join(__dirname, "..", "..", "uploads");

// Exactly the buckets the upload routes write to (UPLOAD_SUBDIRS plus the
// bucketForMime fallback in routes/generate.js).
const MANAGED_BUCKETS = ["images", "audio", "documents"];

// The shape multer's filename callback produces: a UUID plus one of the
// extensions MIME_EXTENSIONS maps to, or .bin.
const MANAGED_NAME =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.[a-z0-9]{1,5}$/i;

let _ttlMs = config.uploadTtlMs;
let _maxBytes = config.maxUploadBytes;
let _minSweepIntervalMs = config.uploadSweepIntervalMs;

let _lastSweepAtMs = 0;
let _sweeping = false;

/** Absolute path of a managed bucket, or null if the name is not one of ours. */
function bucketDir(bucket) {
  if (!MANAGED_BUCKETS.includes(bucket)) return null;
  const dir = path.join(UPLOADS_ROOT, bucket);
  // Belt and braces: the bucket list is a constant, but keep the containment
  // check so this can never be pointed outside the uploads root.
  return dir.startsWith(UPLOADS_ROOT + path.sep) ? dir : null;
}

/**
 * Every managed file in the managed buckets, with its size and mtime.
 *
 * Anything that is not a plain file, or whose name is not one we generated, is
 * skipped. A file that vanishes between readdir and stat is skipped too.
 */
async function listManaged() {
  const found = [];

  for (const bucket of MANAGED_BUCKETS) {
    const dir = bucketDir(bucket);
    if (!dir) continue;

    let entries;
    try {
      entries = await fsp.readdir(dir, { withFileTypes: true });
    } catch {
      continue; // bucket not created yet, or unreadable
    }

    for (const entry of entries) {
      // isFile() is false for directories and for symlinks, so neither a
      // nested directory nor a link out of the tree is ever a candidate.
      if (!entry.isFile()) continue;
      if (!MANAGED_NAME.test(entry.name)) continue;

      const full = path.join(dir, entry.name);
      if (!full.startsWith(dir + path.sep)) continue;

      try {
        const stat = await fsp.lstat(full);
        if (!stat.isFile()) continue;
        found.push({ full, size: stat.size, mtimeMs: stat.mtimeMs });
      } catch {
        // Removed between readdir and lstat; nothing to reclaim.
      }
    }
  }

  return found;
}

/** Unlink one file, treating an already-gone file as success. */
async function remove(file) {
  try {
    await fsp.unlink(file.full);
    return true;
  } catch (err) {
    if (err && err.code === "ENOENT") return true; // lost a race, same outcome
    logger.warn("upload retention could not remove a file", { error: err && err.message });
    return false;
  }
}

/**
 * Reclaim stale uploads, then trim to the size ceiling.
 *
 * Returns what it did, so a caller or a test can assert on it rather than on
 * the filesystem alone. Never throws: a sweep failing must not fail an upload.
 */
async function sweep({ now = Date.now() } = {}) {
  const stats = { scanned: 0, removedStale: 0, removedForCeiling: 0, bytesBefore: 0, bytesAfter: 0 };

  let managed;
  try {
    managed = await listManaged();
  } catch (err) {
    logger.warn("upload retention scan failed", { error: err && err.message });
    return stats;
  }

  stats.scanned = managed.length;
  stats.bytesBefore = managed.reduce((sum, f) => sum + f.size, 0);

  const kept = [];
  for (const file of managed) {
    if (Number.isFinite(_ttlMs) && now - file.mtimeMs > _ttlMs) {
      if (await remove(file)) stats.removedStale += 1;
      else kept.push(file);
    } else {
      kept.push(file);
    }
  }

  // Oldest first, so a burst of fresh uploads cannot evict an older file that
  // is still inside its TTL while the burst itself survives.
  kept.sort((a, b) => a.mtimeMs - b.mtimeMs);

  let total = kept.reduce((sum, f) => sum + f.size, 0);
  for (const file of kept) {
    if (total <= _maxBytes) break;
    if (await remove(file)) {
      total -= file.size;
      stats.removedForCeiling += 1;
    }
  }

  stats.bytesAfter = total;

  if (stats.removedStale || stats.removedForCeiling) {
    logger.info("upload retention reclaimed files", {
      removedStale: stats.removedStale,
      removedForCeiling: stats.removedForCeiling,
      bytesBefore: stats.bytesBefore,
      bytesAfter: stats.bytesAfter,
    });
  }

  return stats;
}

/**
 * Sweep if one has not run recently, without making the caller wait.
 *
 * Called when an upload arrives. Throttled so a burst causes one scan rather
 * than one per request, and guarded so two sweeps never overlap.
 */
function maybeSweep({ now = Date.now() } = {}) {
  if (_sweeping) return false;
  if (now - _lastSweepAtMs < _minSweepIntervalMs) return false;

  _lastSweepAtMs = now;
  _sweeping = true;

  // Deliberately not awaited: the upload must not wait on a directory scan.
  // sweep() never rejects, and the catch is here so a future change cannot
  // turn this into an unhandled rejection.
  Promise.resolve()
    .then(() => sweep({ now }))
    .catch((err) => logger.warn("upload retention sweep failed", { error: err && err.message }))
    .finally(() => {
      _sweeping = false;
    });

  return true;
}

/**
 * Override retention limits. Pass no arguments to restore the configured
 * values. For tests only — not part of the public API.
 */
function _resetLimits({ ttlMs, maxBytes, minSweepIntervalMs } = {}) {
  _ttlMs = ttlMs !== undefined ? ttlMs : config.uploadTtlMs;
  _maxBytes = maxBytes !== undefined ? maxBytes : config.maxUploadBytes;
  _minSweepIntervalMs =
    minSweepIntervalMs !== undefined ? minSweepIntervalMs : config.uploadSweepIntervalMs;
  _lastSweepAtMs = 0;
  _sweeping = false;
}

module.exports = {
  sweep,
  maybeSweep,
  UPLOADS_ROOT,
  MANAGED_BUCKETS,
  MANAGED_NAME,
  _listManaged: listManaged,
  _resetLimits,
  get _ttlMs() {
    return _ttlMs;
  },
  get _maxBytes() {
    return _maxBytes;
  },
};
