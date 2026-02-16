const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");
const { v4: uuidv4 } = require("uuid");
const { generateImage, generateVideo, generateCode } = require("../services/model");

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, path.join(__dirname, "..", "..", "uploads", "images"));
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

// Upload image for vision understanding
router.post("/understand", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "Image file is required" });

    const { prompt = "Describe this image" } = req.body;
    const imagePath = `/uploads/images/${req.file.filename}`;

    res.json({
      description: `[FramerAI Vision Analysis]\nImage: ${req.file.originalname}\nPrompt: ${prompt}\n\nThis is a placeholder response. Connect a trained FramerAI model for actual vision analysis.`,
      imagePath,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
