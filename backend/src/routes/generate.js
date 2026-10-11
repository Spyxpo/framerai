const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { randomUUID } = require("node:crypto");
const {
  generateImage,
  generateVideo,
  generateAudio,
  generateCode,
  cloneVoice,
  verifyAudio,
  transcribeAudio,
  understandImage,
  understandVideo,
  readDocument,
} = require("../services/model");
const { ApiError, asyncHandler } = require("../middleware/errors");
const { validator } = require("../middleware/validate");
const uploadRetention = require("../services/uploadRetention");
const config = require("../config");
const { readSettings, ASPECT_RATIOS, SIZE_TIERS } = require("../generationSettings");
const modelLimits = require("../modelLimits");

// The documented floor; the accepted length rises with the loaded model's window.
const MAX_PROMPT_LENGTH = modelLimits.BASE_PROMPT_CHARS;
// Square-only sizes, kept so existing clients keep working. Prefer
// width/height or aspect + tier.
const RESOLUTIONS = [64, 128, 256, 512];
const MIN_DIMENSION = 64;
const MAX_DIMENSION = 2048;
// Document types the ingestion path understands. Kept as an explicit list
// rather than a prefix, because "application/" covers far more than this.
// Each maps to the extension the stored file gets, which is also how the worker
// picks a reader. Code is listed under the types browsers and tools actually
// send for it; the website maps an unlabelled file to one of these first.
const DOCUMENT_TYPES = {
  "application/pdf": ".pdf",
  "text/plain": ".txt",
  "text/markdown": ".md",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
  "text/csv": ".csv",
  "text/html": ".html",
  "application/json": ".json",
  "text/x-python": ".py",
  "text/x-script.python": ".py",
  "text/javascript": ".js",
  "application/javascript": ".js",
  "application/typescript": ".ts",
  "text/x-typescript": ".ts",
  "text/x-java-source": ".java",
  "text/x-java": ".java",
  "text/x-go": ".go",
  "text/x-rust": ".rs",
  "text/x-c": ".c",
  "text/x-csrc": ".c",
  "text/x-c++src": ".cpp",
  "text/x-csharp": ".cs",
  "text/x-ruby": ".rb",
  "text/x-php": ".php",
  "application/x-sh": ".sh",
  "text/x-sh": ".sh",
  "application/sql": ".sql",
  "text/css": ".css",
  "application/yaml": ".yaml",
  "text/yaml": ".yaml",
  "application/toml": ".toml",
};
const DOCUMENT_MIME_TYPES = Object.keys(DOCUMENT_TYPES);
const MAX_DOCUMENT_PAGES = 2000;
// Duration is bounded by the overlapped-window path rather than by a single
// denoising window, so the frame ceiling is no longer one window's worth.
const MAX_FRAMES = 512;
const MIN_FPS = 1;
const MAX_FPS = 60;
const LANGUAGES = [
  "python",
  "javascript",
  "typescript",
  "java",
  "go",
  "rust",
  "c",
  "cpp",
  "csharp",
  "ruby",
  "php",
  "shell",
  "sql",
  "html",
  "css",
];

// Route uploads to a subfolder chosen by the form field name.
const UPLOAD_SUBDIRS = { audio: "audio", document: "documents", image: "images", mask: "images", video: "clips" };

// A generic upload field says nothing about the file, so fall back to its type.
function bucketForMime(mimetype = "") {
  if (mimetype.startsWith("audio/")) return "audio";
  // Not "videos": that directory holds model output with its own lifecycle.
  if (mimetype.startsWith("video/")) return "clips";
  if (DOCUMENT_MIME_TYPES.includes(mimetype)) return "documents";
  return "images";
}

// Server-side MIME → extension map. The stored extension is derived exclusively
// from the validated MIME type, never from the client-supplied filename, so an
// upload that passes the MIME filter cannot be stored with an executable
// extension such as .html or .js (Issue #342).
const MIME_EXTENSIONS = {
  // images
  "image/jpeg":    ".jpg",
  "image/jpg":     ".jpg",
  "image/png":     ".png",
  "image/gif":     ".gif",
  "image/webp":    ".webp",
  "image/svg+xml": ".svg",
  "image/bmp":     ".bmp",
  "image/tiff":    ".tiff",
  "image/avif":    ".avif",
  "image/heic":    ".heic",
  "image/heif":    ".heif",
  "image/x-icon":  ".ico",
  // audio
  "audio/wav":     ".wav",
  "audio/wave":    ".wav",
  "audio/x-wav":   ".wav",
  "audio/mpeg":    ".mp3",
  "audio/mp3":     ".mp3",
  "audio/ogg":     ".ogg",
  "audio/flac":    ".flac",
  "audio/aac":     ".aac",
  "audio/mp4":     ".m4a",
  "audio/x-m4a":   ".m4a",
  "audio/webm":    ".webm",
  // video
  "video/mp4":       ".mp4",
  "video/webm":      ".webm",
  "video/quicktime": ".mov",
  // documents — exact allowlist from DOCUMENT_TYPES
  ...DOCUMENT_TYPES,
};

function safeExtension(mimetype) {
  return MIME_EXTENSIONS[mimetype] || ".bin";
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const sub = UPLOAD_SUBDIRS[file.fieldname] || bucketForMime(file.mimetype);
    const dir = path.join(__dirname, "..", "..", "uploads", sub);
    fs.mkdirSync(dir, { recursive: true });
    // Every accepted upload passes through here, so this is the one hook that
    // covers all four upload routes. It is throttled and not awaited, and it
    // runs before this file exists, so it can neither delay the upload nor
    // reclaim the upload that triggered it (Issue #367).
    uploadRetention.maybeSweep();
    cb(null, dir);
  },
  filename: (req, file, cb) => {
    cb(null, `${randomUUID()}${safeExtension(file.mimetype)}`);
  },
});

// Only accept the media type the route actually understands, so a mislabelled
// upload fails before it is written to disk.
function mimeFilter(prefix) {
  return (req, file, cb) => {
    if (file.mimetype.startsWith(prefix)) return cb(null, true);
    cb(ApiError.badRequest(`Expected ${prefix.replace("/", "")} upload, got ${file.mimetype}`));
  };
}

// `files`/`fileSize` bound the upload itself; without `fields`/`fieldSize`
// alongside them, the rest of the request — every non-file field — was
// bounded only by Busboy's own unconfigured defaults (1 MiB per field, and no
// limit at all on how many of them a request may carry). A request with no
// file but hundreds of large text fields could still make multer buffer an
// unbounded amount of memory (Issue #440). Each route below gets exactly the
// field count its own handler reads, both here and in generate.js's own
// `req.body` reads, and is otherwise mentioned nowhere else in this file; the
// shared `fieldSize` only makes explicit the bound Busboy already enforced.
//
// `/understand`'s handler reads one field: `prompt`.
const uploadImage = multer({
  storage,
  limits: { fileSize: config.maxFileSize, files: 1, fields: 1, fieldSize: config.maxMultipartFieldSize },
  fileFilter: mimeFilter("image/"),
});
// `/transcribe`'s handler reads one field: `prompt`.
const uploadAudio = multer({
  storage,
  limits: { fileSize: config.maxFileSize, files: 1, fields: 1, fieldSize: config.maxMultipartFieldSize },
  fileFilter: mimeFilter("audio/"),
});

// Documents are matched against an allowlist rather than a prefix: the useful
// types share no prefix, and "application/" would admit anything at all.
function mimeAllowlist(types) {
  return (req, file, cb) => {
    if (types.includes(file.mimetype)) return cb(null, true);
    cb(ApiError.badRequest(`Expected one of ${types.join(", ")}, got ${file.mimetype}`));
  };
}

// `/document`'s handler reads two fields: `prompt` and `max_pages`.
const uploadDocument = multer({
  storage,
  limits: { fileSize: config.maxFileSize, files: 1, fields: 2, fieldSize: config.maxMultipartFieldSize },
  fileFilter: mimeAllowlist(DOCUMENT_MIME_TYPES),
});

// Attachments are stored and then referenced by later chat turns, so this
// uploader takes anything the chat path can carry and runs no model at all.
// `/upload`'s handler reads no fields of its own — only the file.
const uploadAttachment = multer({
  storage,
  limits: { fileSize: config.maxFileSize, files: 1, fields: 0, fieldSize: config.maxMultipartFieldSize },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith("image/") || DOCUMENT_MIME_TYPES.includes(file.mimetype)) {
      return cb(null, true);
    }
    cb(ApiError.badRequest(`Cannot attach ${file.mimetype}`));
  },
});

// Sampler controls shared by generation and editing. `shift` spends more steps
// at high noise; `guidance_interval` applies guidance only between two times.
function readSampler(v) {
  const out = {};
  const shift = v.number("shift", { min: 0.1, max: 10 });
  if (shift !== undefined) out.shift = shift;
  const interval = v.array("guidance_interval", { max: 2 });
  if (interval && interval.length) {
    const ok = interval.length === 2 && interval.every((n) => typeof n === "number" && n >= 0 && n <= 1)
      && interval[0] <= interval[1];
    if (!ok) v.fail("guidance_interval", "must be two numbers in [0, 1], low then high");
    else out.guidance_interval = interval;
  }
  return out;
}

// Generate image from text
router.post(
  "/image",
  asyncHandler(async (req, res) => {
    const v = validator(req.body);
    const prompt = v.string("prompt", { required: true, max: modelLimits.promptChars() });
    const numImages = v.integer("num_images", { min: 1, max: 4, fallback: 1 });
    // Size is optional at every level. Width and height must be given together;
    // otherwise an aspect ratio at a tier, otherwise whatever the prompt asks
    // for, otherwise the model's default.
    const width = v.integer("width", { min: MIN_DIMENSION, max: MAX_DIMENSION });
    const height = v.integer("height", { min: MIN_DIMENSION, max: MAX_DIMENSION });
    const aspect = v.oneOf("aspect", ASPECT_RATIOS);
    const tier = v.oneOf("tier", SIZE_TIERS);
    const seed = v.integer("seed", { min: 0, max: 2 ** 31 - 1 });
    const resolution = v.oneOf("resolution", RESOLUTIONS);
    const sampler = readSampler(v);
    v.done();

    if ((width === undefined) !== (height === undefined)) {
      throw ApiError.badRequest("Request validation failed", [
        { field: width === undefined ? "width" : "height", message: "is required alongside the other" },
      ]);
    }

    res.json(
      await generateImage(prompt, numImages, { width, height, aspect, tier, seed, resolution, ...sampler }, req.requestId)
    );
  })
);

// Generate video from text
router.post(
  "/video",
  asyncHandler(async (req, res) => {
    const v = validator(req.body);
    const prompt = v.string("prompt", { required: true, max: modelLimits.promptChars() });
    const numFrames = v.integer("num_frames", { min: 1, max: MAX_FRAMES, fallback: 16 });
    // The worker has accepted every one of these all along; the route exposed
    // none of them, so a client could ask for neither a size nor a frame rate
    // nor a reproducible seed.
    const width = v.integer("width", { min: MIN_DIMENSION, max: MAX_DIMENSION });
    const height = v.integer("height", { min: MIN_DIMENSION, max: MAX_DIMENSION });
    const aspect = v.oneOf("aspect", ASPECT_RATIOS);
    const tier = v.oneOf("tier", SIZE_TIERS);
    const fps = v.integer("fps", { min: MIN_FPS, max: MAX_FPS });
    const seed = v.integer("seed", { min: 0, max: 2 ** 31 - 1 });
    v.done();

    if ((width === undefined) !== (height === undefined)) {
      throw ApiError.badRequest("Request validation failed", [
        { field: width === undefined ? "width" : "height", message: "is required alongside the other" },
      ]);
    }

    res.json(
      await generateVideo(prompt, numFrames, { width, height, aspect, tier, fps, seed }, req.requestId)
    );
  })
);

// Generate audio / speech from text
router.post(
  "/audio",
  asyncHandler(async (req, res) => {
    const v = validator(req.body);
    const prompt = v.string("prompt", { required: true, max: modelLimits.promptChars() });
    v.done();

    res.json(await generateAudio(prompt, {}, req.requestId));
  })
);

// `/voice`'s handler reads three fields: `prompt`, `consent` and `reference_text`.
const uploadVoice = multer({
  storage,
  limits: { fileSize: config.maxFileSize, files: 1, fields: 3, fieldSize: config.maxMultipartFieldSize },
  fileFilter: mimeFilter("audio/"),
});

// Speak a prompt in the voice of an uploaded reference clip. Cloning someone's
// voice needs their permission, so the caller has to say they have it, and the
// output is always watermarked so it can be identified as generated.
router.post(
  "/voice",
  uploadVoice.single("reference"),
  asyncHandler(async (req, res) => {
    if (!req.file) throw ApiError.badRequest("Request validation failed", [{ field: "reference", message: "is required" }]);

    const v = validator(req.body);
    const prompt = v.string("prompt", { required: true, max: modelLimits.promptChars() });
    const consent = v.oneOf("consent", ["true", "false"], { required: true });
    const referenceText = v.string("reference_text", { max: modelLimits.promptChars() });
    v.done();

    if (consent !== "true") {
      throw ApiError.badRequest("Voice cloning requires the speaker's consent", [
        { field: "consent", message: "must be true: confirm you have the speaker's permission to clone their voice" },
      ]);
    }

    const result = await cloneVoice(req.file.path, prompt, { referenceText, requestId: req.requestId });
    if (result.error) throw ApiError.badRequest(result.error);
    res.json(result);
  })
);

// `/verify-audio`'s handler reads no fields of its own, only the file.
const uploadVerify = multer({
  storage,
  limits: { fileSize: config.maxFileSize, files: 1, fields: 0, fieldSize: config.maxMultipartFieldSize },
  fileFilter: mimeFilter("audio/"),
});

// Check a recording for the watermark generated speech carries.
router.post(
  "/verify-audio",
  uploadVerify.single("audio"),
  asyncHandler(async (req, res) => {
    if (!req.file) throw ApiError.badRequest("Request validation failed", [{ field: "audio", message: "is required" }]);
    const result = await verifyAudio(req.file.path, req.requestId);
    if (result.unavailable) {
      throw new ApiError(503, "The model worker is not running, so audio cannot be checked", "MODEL_UNAVAILABLE");
    }
    res.json(result);
  })
);

const mimeImages = mimeFilter("image/");

// `/image-edit`'s handler reads four fields: `prompt`, `strength`, `seed` and `shift`.
const uploadEdit = multer({
  storage,
  limits: { fileSize: config.maxFileSize, files: 2, fields: 4, fieldSize: config.maxMultipartFieldSize },
  fileFilter: mimeImages,
});

// Redraw an uploaded image to a prompt, or repaint only where a mask is white.
router.post(
  "/image-edit",
  uploadEdit.fields([{ name: "image", maxCount: 1 }, { name: "mask", maxCount: 1 }]),
  asyncHandler(async (req, res) => {
    const image = req.files && req.files.image && req.files.image[0];
    if (!image) throw ApiError.badRequest("Request validation failed", [{ field: "image", message: "is required" }]);
    const mask = req.files.mask && req.files.mask[0];

    const v = validator(req.body);
    const prompt = v.string("prompt", { required: true, max: modelLimits.promptChars() });
    const strength = v.number("strength", { min: 0, max: 1, fallback: 0.6 });
    const seed = v.integer("seed", { min: 0, max: 2 ** 31 - 1 });
    const shift = v.number("shift", { min: 0.1, max: 10 });
    v.done();

    const edit = { init_image_path: image.path, strength, seed, shift };
    if (mask) edit.mask_path = mask.path;
    res.json(await generateImage(prompt, 1, edit, req.requestId));
  })
);

// `/animate`'s handler reads four fields: `prompt`, `num_frames`, `fps` and `seed`.
const uploadAnimate = multer({
  storage,
  limits: { fileSize: config.maxFileSize, files: 1, fields: 4, fieldSize: config.maxMultipartFieldSize },
  fileFilter: mimeImages,
});

// Make a clip that opens on an uploaded image and continues from it.
router.post(
  "/animate",
  uploadAnimate.single("image"),
  asyncHandler(async (req, res) => {
    if (!req.file) throw ApiError.badRequest("Request validation failed", [{ field: "image", message: "is required" }]);

    const v = validator(req.body);
    const prompt = v.string("prompt", { required: true, max: modelLimits.promptChars() });
    const numFrames = v.integer("num_frames", { min: 1, max: MAX_FRAMES, fallback: 16 });
    const fps = v.integer("fps", { min: MIN_FPS, max: MAX_FPS });
    const seed = v.integer("seed", { min: 0, max: 2 ** 31 - 1 });
    v.done();

    res.json(await generateVideo(prompt, numFrames, { fps, seed, first_frame_path: req.file.path }, req.requestId));
  })
);

// `/understand-video`'s handler reads one field: `prompt`.
const uploadVideo = multer({
  storage,
  limits: { fileSize: config.maxFileSize, files: 1, fields: 1, fieldSize: config.maxMultipartFieldSize },
  fileFilter: mimeFilter("video/"),
});

// Answer a question about an uploaded video, from frames spread across it.
router.post(
  "/understand-video",
  uploadVideo.single("video"),
  asyncHandler(async (req, res) => {
    if (!req.file) throw ApiError.badRequest("Request validation failed", [{ field: "video", message: "is required" }]);

    const v = validator(req.body);
    const prompt = v.string("prompt", { max: modelLimits.promptChars(), fallback: "Describe this video:" });
    v.done();

    const result = await understandVideo(req.file.path, prompt, req.requestId);
    res.json({ description: result.description, frames: result.frames, videoPath: `/uploads/clips/${req.file.filename}` });
  })
);

// Generate code
router.post(
  "/code",
  asyncHandler(async (req, res) => {
    const v = validator(req.body);
    const prompt = v.string("prompt", { required: true, max: modelLimits.promptChars() });
    const language = v.oneOf("language", LANGUAGES, { fallback: "python" });
    const settings = readSettings(v);
    v.done();

    res.json(await generateCode(prompt, language, settings, req.requestId));
  })
);

// Upload an image for vision understanding
router.post(
  "/understand",
  uploadImage.single("image"),
  asyncHandler(async (req, res) => {
    if (!req.file) throw ApiError.badRequest("Request validation failed", [{ field: "image", message: "is required" }]);

    const v = validator(req.body);
    const prompt = v.string("prompt", { max: modelLimits.promptChars(), fallback: "Describe this image" });
    v.done();

    const imagePath = `/uploads/images/${req.file.filename}`;
    const result = await understandImage(req.file.path, prompt, req.requestId);

    res.json({ description: result.description, imagePath });
  })
);

// Store a file for a later chat turn to reference. Deliberately runs no model:
// attaching is not asking, and paying for inference to learn a path is waste.
router.post(
  "/upload",
  uploadAttachment.single("file"),
  asyncHandler(async (req, res) => {
    if (!req.file) throw ApiError.badRequest("Request validation failed", [{ field: "file", message: "is required" }]);

    const bucket = path.basename(req.file.destination);
    res.status(201).json({
      path: `/uploads/${bucket}/${req.file.filename}`,
      kind: bucket === "documents" ? "document" : bucket === "audio" ? "audio" : "image",
      name: req.file.originalname,
      size: req.file.size,
      mimetype: req.file.mimetype,
    });
  })
);

// Upload a document for reading. Returns the extracted text in reading order,
// and answers a prompt about it when one is given.
router.post(
  "/document",
  uploadDocument.single("document"),
  asyncHandler(async (req, res) => {
    if (!req.file) throw ApiError.badRequest("Request validation failed", [{ field: "document", message: "is required" }]);

    const v = validator(req.body);
    const prompt = v.string("prompt", { max: modelLimits.promptChars(), fallback: "" });
    const maxPages = v.integer("max_pages", { min: 1, max: MAX_DOCUMENT_PAGES });
    v.done();

    const documentPath = `/uploads/documents/${req.file.filename}`;
    const result = await readDocument(req.file.path, prompt, { maxPages, requestId: req.requestId });
    if (result.error) throw ApiError.badRequest(result.error);

    res.json({ ...result, documentPath });
  })
);

// Upload audio for transcription / understanding
router.post(
  "/transcribe",
  uploadAudio.single("audio"),
  asyncHandler(async (req, res) => {
    if (!req.file) throw ApiError.badRequest("Request validation failed", [{ field: "audio", message: "is required" }]);

    const v = validator(req.body);
    const prompt = v.string("prompt", { max: modelLimits.promptChars(), fallback: "Transcribe the audio:" });
    v.done();

    const audioPath = `/uploads/audio/${req.file.filename}`;
    const result = await transcribeAudio(req.file.path, prompt, req.requestId);

    res.json({ text: result.text, audioPath, metadata: result.metadata });
  })
);

router.MAX_PROMPT_LENGTH = MAX_PROMPT_LENGTH;
router.RESOLUTIONS = RESOLUTIONS;
router.ASPECT_RATIOS = ASPECT_RATIOS;
router.SIZE_TIERS = SIZE_TIERS;
router.MIN_DIMENSION = MIN_DIMENSION;
router.MAX_DIMENSION = MAX_DIMENSION;
router.LANGUAGES = LANGUAGES;
router.DOCUMENT_MIME_TYPES = DOCUMENT_MIME_TYPES;
router.DOCUMENT_TYPES = DOCUMENT_TYPES;
router.MAX_DOCUMENT_PAGES = MAX_DOCUMENT_PAGES;
router.MAX_FRAMES = MAX_FRAMES;
router.MIN_FPS = MIN_FPS;
router.MAX_FPS = MAX_FPS;

module.exports = router;
