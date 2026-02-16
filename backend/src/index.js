const express = require("express");
const cors = require("cors");
const http = require("http");
const { WebSocketServer } = require("ws");
const path = require("path");

require("dotenv").config({ path: path.join(__dirname, "..", ".env") });

const chatRoutes = require("./routes/chat");
const generateRoutes = require("./routes/generate");
const healthRoutes = require("./routes/health");
const { setupWebSocket } = require("./services/websocket");

const app = express();
const server = http.createServer(app);

const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors({ origin: process.env.CORS_ORIGIN || "http://localhost:5173" }));
app.use(express.json({ limit: "50mb" }));
app.use("/uploads", express.static(path.join(__dirname, "..", "uploads")));

// Routes
app.use("/api/health", healthRoutes);
app.use("/api/chat", chatRoutes);
app.use("/api/generate", generateRoutes);

// WebSocket for streaming
const wss = new WebSocketServer({ server, path: "/ws" });
setupWebSocket(wss);

server.listen(PORT, () => {
  console.log(`FramerAI backend running on http://localhost:${PORT}`);
  console.log(`WebSocket available at ws://localhost:${PORT}/ws`);
});
