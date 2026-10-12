const express = require("express");
const router = express.Router();
const pricing = require("../pricing");

// The rate card, derived from the serving cost and measured throughput.
router.get("/", (req, res) => {
  res.json(pricing.rates());
});

module.exports = router;
