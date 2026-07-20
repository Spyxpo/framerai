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
const upload = multer({ storage, limits: { fileSize: 50 * 1024 * 1024 } });

// Generate image from text
router.post("/image", async (req, res) => {
  try {
    const { prompt, num_images = 1, resolution = 256 } = req.body;
    if (!prompt) return res.status(400).json({ error: "Prompt is required" });

    const result = await generateImage(prompt, num_images, resolution);
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Generate video from text
router.post("/video", async (req, res) => {
  try {
    const { prompt, num_frames = 16 } = req.body;
    if (!prompt) return res.status(400).json({ error: "Prompt is required" });

    const result = await generateVideo(prompt, num_frames);
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Generate audio / speech from text
router.post("/audio", async (req, res) => {
  try {
    const { prompt } = req.body;
    if (!prompt) return res.status(400).json({ error: "Prompt is required" });

    const result = await generateAudio(prompt);
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Generate code
router.post("/code", async (req, res) => {
  try {
    const { prompt, language = "python" } = req.body;
    if (!prompt) return res.status(400).json({ error: "Prompt is required" });

    const result = await generateCode(prompt, language);
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Upload an image for vision understanding
router.post("/understand", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "Image file is required" });

    const { prompt = "Describe this image" } = req.body;
    const imagePath = `/uploads/images/${req.file.filename}`;
    const result = await understandImage(req.file.path, prompt);

    res.json({ description: result.description, imagePath });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Upload audio for transcription / understanding
router.post("/transcribe", upload.single("audio"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "Audio file is required" });

    const { prompt = "Transcribe the audio:" } = req.body;
    const audioPath = `/uploads/audio/${req.file.filename}`;
    const result = await transcribeAudio(req.file.path, prompt);

    res.json({ text: result.text, audioPath, metadata: result.metadata });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
