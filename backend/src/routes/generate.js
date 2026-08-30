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
  transcribeAudio,
  understandImage,
  readDocument,
} = require("../services/model");
const { ApiError, asyncHandler } = require("../middleware/errors");
const { validator } = require("../middleware/validate");
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
const DOCUMENT_MIME_TYPES = ["application/pdf", "text/plain", "text/markdown"];
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
const UPLOAD_SUBDIRS = { audio: "audio", document: "documents", image: "images" };

// A generic upload field says nothing about the file, so fall back to its type.
function bucketForMime(mimetype = "") {
  if (mimetype.startsWith("audio/")) return "audio";
  if (DOCUMENT_MIME_TYPES.includes(mimetype)) return "documents";
  return "images";
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const sub = UPLOAD_SUBDIRS[file.fieldname] || bucketForMime(file.mimetype);
    const dir = path.join(__dirname, "..", "..", "uploads", sub);
    fs.mkdirSync(dir, { recursive: true });
    cb(null, dir);
  },
  filename: (req, file, cb) => {
    cb(null, `${randomUUID()}${path.extname(file.originalname)}`);
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

const uploadImage = multer({
  storage,
  limits: { fileSize: config.maxFileSize, files: 1 },
  fileFilter: mimeFilter("image/"),
});
const uploadAudio = multer({
  storage,
  limits: { fileSize: config.maxFileSize, files: 1 },
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

const uploadDocument = multer({
  storage,
  limits: { fileSize: config.maxFileSize, files: 1 },
  fileFilter: mimeAllowlist(DOCUMENT_MIME_TYPES),
});

// Attachments are stored and then referenced by later chat turns, so this
// uploader takes anything the chat path can carry and runs no model at all.
const uploadAttachment = multer({
  storage,
  limits: { fileSize: config.maxFileSize, files: 1 },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith("image/") || DOCUMENT_MIME_TYPES.includes(file.mimetype)) {
      return cb(null, true);
    }
    cb(ApiError.badRequest(`Cannot attach ${file.mimetype}`));
  },
});

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
    v.done();

    if ((width === undefined) !== (height === undefined)) {
      throw ApiError.badRequest("Request validation failed", [
        { field: width === undefined ? "width" : "height", message: "is required alongside the other" },
      ]);
    }

    res.json(
      await generateImage(prompt, numImages, { width, height, aspect, tier, seed, resolution }, req.requestId)
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
router.MAX_DOCUMENT_PAGES = MAX_DOCUMENT_PAGES;
router.MAX_FRAMES = MAX_FRAMES;
router.MIN_FPS = MIN_FPS;
router.MAX_FPS = MAX_FPS;

module.exports = router;
