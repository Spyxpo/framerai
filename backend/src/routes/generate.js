const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { v4: uuidv4 } = require("uuid");
const {
  generateImage,
  generateVideo,
  generateAudio,
  generateCode,
  transcribeAudio,
  understandImage,
} = require("../services/model");
const { ApiError, asyncHandler } = require("../middleware/errors");
const { validator } = require("../middleware/validate");

const MAX_PROMPT_LENGTH = 4000;
const RESOLUTIONS = [64, 128, 256, 512];
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

// Route uploads to a subfolder chosen by the form field name (image / audio).
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const sub = file.fieldname === "audio" ? "audio" : "images";
    const dir = path.join(__dirname, "..", "..", "uploads", sub);
    fs.mkdirSync(dir, { recursive: true });
    cb(null, dir);
  },
  filename: (req, file, cb) => {
    cb(null, `${uuidv4()}${path.extname(file.originalname)}`);
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
  limits: { fileSize: 50 * 1024 * 1024, files: 1 },
  fileFilter: mimeFilter("image/"),
});
const uploadAudio = multer({
  storage,
  limits: { fileSize: 50 * 1024 * 1024, files: 1 },
  fileFilter: mimeFilter("audio/"),
});

// Generate image from text
router.post(
  "/image",
  asyncHandler(async (req, res) => {
    const v = validator(req.body);
    const prompt = v.string("prompt", { required: true, max: MAX_PROMPT_LENGTH });
    const numImages = v.integer("num_images", { min: 1, max: 4, fallback: 1 });
    const resolution = v.oneOf("resolution", RESOLUTIONS, { fallback: 256 });
    v.done();

    res.json(await generateImage(prompt, numImages, resolution));
  })
);

// Generate video from text
router.post(
  "/video",
  asyncHandler(async (req, res) => {
    const v = validator(req.body);
    const prompt = v.string("prompt", { required: true, max: MAX_PROMPT_LENGTH });
    const numFrames = v.integer("num_frames", { min: 1, max: 64, fallback: 16 });
    v.done();

    res.json(await generateVideo(prompt, numFrames));
  })
);

// Generate audio / speech from text
router.post(
  "/audio",
  asyncHandler(async (req, res) => {
    const v = validator(req.body);
    const prompt = v.string("prompt", { required: true, max: MAX_PROMPT_LENGTH });
    v.done();

    res.json(await generateAudio(prompt));
  })
);

// Generate code
router.post(
  "/code",
  asyncHandler(async (req, res) => {
    const v = validator(req.body);
    const prompt = v.string("prompt", { required: true, max: MAX_PROMPT_LENGTH });
    const language = v.oneOf("language", LANGUAGES, { fallback: "python" });
    v.done();

    res.json(await generateCode(prompt, language));
  })
);

// Upload an image for vision understanding
router.post(
  "/understand",
  uploadImage.single("image"),
  asyncHandler(async (req, res) => {
    if (!req.file) throw ApiError.badRequest("Request validation failed", [{ field: "image", message: "is required" }]);

    const v = validator(req.body);
    const prompt = v.string("prompt", { max: MAX_PROMPT_LENGTH, fallback: "Describe this image" });
    v.done();

    const imagePath = `/uploads/images/${req.file.filename}`;
    const result = await understandImage(req.file.path, prompt);

    res.json({ description: result.description, imagePath });
  })
);

// Upload audio for transcription / understanding
router.post(
  "/transcribe",
  uploadAudio.single("audio"),
  asyncHandler(async (req, res) => {
    if (!req.file) throw ApiError.badRequest("Request validation failed", [{ field: "audio", message: "is required" }]);

    const v = validator(req.body);
    const prompt = v.string("prompt", { max: MAX_PROMPT_LENGTH, fallback: "Transcribe the audio:" });
    v.done();

    const audioPath = `/uploads/audio/${req.file.filename}`;
    const result = await transcribeAudio(req.file.path, prompt);

    res.json({ text: result.text, audioPath, metadata: result.metadata });
  })
);

module.exports = router;
