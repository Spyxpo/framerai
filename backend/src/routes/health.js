const express = require("express");
const router = express.Router();

router.get("/", (req, res) => {
  res.json({
    status: "ok",
    model: "FramerAI",
    version: "1.0.0",
    capabilities: ["text", "code", "image", "video", "audio"],
    timestamp: new Date().toISOString(),
  });
});

module.exports = router;
