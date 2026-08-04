const express = require("express");
const cors = require("cors");
const http = require("http");
const { WebSocketServer } = require("ws");
const path = require("path");

require("dotenv").config({ path: path.join(__dirname, "..", ".env") });

const config = require("./config");
const chatRoutes = require("./routes/chat");
const generateRoutes = require("./routes/generate");
const healthRoutes = require("./routes/health");
const { setupWebSocket } = require("./services/websocket");
const { notFoundHandler, errorHandler } = require("./middleware/errors");
const { apiLimiter, generationLimiter } = require("./middleware/limiters");

const app = express();
const server = http.createServer(app);

const PORT = process.env.PORT || 3001;

// Rate limits are keyed by client address, so the proxy setting has to be
// right for them to mean anything behind nginx.
app.set("trust proxy", config.trustProxy);

// Middleware
app.use(cors({ origin: process.env.CORS_ORIGIN || "http://localhost:5173" }));
app.use(express.json({ limit: config.jsonBodyLimit }));
app.use("/uploads", express.static(path.join(__dirname, "..", "uploads")));

// A broad ceiling for the whole API, then a much tighter one for the routes
// that actually run the model.
app.use("/api", apiLimiter);

// Routes
app.use("/api/health", healthRoutes);
app.use("/api/chat", chatRoutes);
app.use("/api/generate", generationLimiter, generateRoutes);

// Unmatched routes and every thrown error share one response shape
app.use(notFoundHandler);
app.use(errorHandler);

// WebSocket for streaming
const wss = new WebSocketServer({ server, path: "/ws", maxPayload: config.maxWsPayload });
setupWebSocket(wss);

server.listen(PORT, () => {
  console.log(`FramerAI backend running on http://localhost:${PORT}`);
  console.log(`WebSocket available at ws://localhost:${PORT}/ws`);
});
