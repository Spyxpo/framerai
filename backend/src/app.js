/**
 * The Express app and the WebSocket wiring, without starting a server.
 *
 * src/index.js starts it for real; tests mount the same app on an ephemeral
 * port, so what is exercised is what runs in production.
 */

const express = require("express");
const cors = require("cors");
const http = require("http");
const { WebSocketServer } = require("ws");
const path = require("path");

const config = require("./config");
const chatRoutes = require("./routes/chat");
const generateRoutes = require("./routes/generate");
const healthRoutes = require("./routes/health");
const { setupWebSocket } = require("./services/websocket");
const { getOpenApiSpecJson } = require("./openapi");
const { notFoundHandler, errorHandler } = require("./middleware/errors");
const { apiLimiter, generationLimiter } = require("./middleware/limiters");
const { requestIdMiddleware } = require("./middleware/requestId");

function createApp() {
  const app = express();

  // Rate limits are keyed by client address, so the proxy setting has to be
  // right for them to mean anything behind nginx.
  app.set("trust proxy", config.trustProxy);

  // Middleware
  app.use(cors({ origin: process.env.CORS_ORIGIN || "http://localhost:5173" }));
  app.use(express.json({ limit: config.jsonBodyLimit }));
  app.use(requestIdMiddleware);
  app.use("/uploads", express.static(path.join(__dirname, "..", "uploads")));

  // A broad ceiling for the whole API, then a much tighter one for the routes
  // that actually run the model.
  app.use("/api", apiLimiter);

  // OpenAPI 3.1 specification endpoint
  app.get("/api/openapi.json", (req, res) => {
    res.setHeader("Content-Type", "application/json");
    res.send(getOpenApiSpecJson());
  });

  // Routes
  app.use("/api/health", healthRoutes);
  app.use("/api/chat", chatRoutes);
  app.use("/api/generate", generationLimiter, generateRoutes);

  // Unmatched routes and every thrown error share one response shape
  app.use(notFoundHandler);
  app.use(errorHandler);

  return app;
}

/**
 * An HTTP server with the app mounted and the WebSocket endpoint attached.
 */
function createServer() {
  const app = createApp();
  const server = http.createServer(app);
  const wss = new WebSocketServer({ server, path: "/ws", maxPayload: config.maxWsPayload });
  setupWebSocket(wss);
  return { app, server, wss };
}

module.exports = { createApp, createServer };
