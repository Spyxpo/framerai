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
const pricingRoutes = require("./routes/pricing");
const { setupWebSocket } = require("./services/websocket");
const { getOpenApiSpecJson } = require("./openapi");
const { ApiError, notFoundHandler, errorHandler } = require("./middleware/errors");
const { apiLimiter, generationLimiter } = require("./middleware/limiters");
const { requestIdMiddleware } = require("./middleware/requestId");

/**
 * The one browser origin the API is for: CORS_ORIGIN, or the Vite dev server when
 * it is not set. The CORS middleware and the WebSocket handshake both read it
 * here, so the two cannot disagree about who is allowed in.
 */
function allowedOrigin() {
  return process.env.CORS_ORIGIN || "http://localhost:5173";
}

/**
 * Whether a request from `origin` may act, for `allowed` (CORS_ORIGIN). The
 * comparison is the one a browser makes against Access-Control-Allow-Origin:
 * exact, with "*" meaning every origin, so another spelling of the allowed origin
 * is another origin.
 *
 * No Origin header means the client is not a browser. A browser sends one on
 * every cross-origin request, and on every state-changing request it makes,
 * so a page on another origin cannot leave it out. CORS lets those clients
 * through and so does this. A header that is present but empty, or "null"
 * (an opaque origin: a sandboxed frame, a file: page), is not absent, and is
 * refused. This decides which browser pages may drive the API. It is not
 * authentication: any client that can reach the server and sends no Origin
 * is still served.
 */
function originAllowed(origin, allowed) {
  return origin === undefined || allowed === "*" || origin === allowed;
}

// Methods that only read. Everything else changes state or runs the model.
const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);

/**
 * Refuse a state-changing API request from a page on another origin (Issue #513).
 *
 * CORS only stops a page reading the answer. A request a browser sends without
 * a preflight (a multipart upload, a text/plain or empty POST) still reached
 * the routes: any page a user visited could run the model, store files in
 * uploads/ and create conversations on their backend. The WebSocket has
 * refused other origins since #425; this applies the same rule to REST. It runs
 * before the body is parsed or an upload written, so a refused request does
 * nothing.
 */
function refuseForeignOrigins(allowed) {
  return (req, res, next) => {
    if (SAFE_METHODS.has(req.method) || originAllowed(req.headers.origin, allowed)) return next();
    next(new ApiError(403, "Origin not allowed", "ORIGIN_NOT_ALLOWED"));
  };
}

function createApp() {
  const app = express();

  // Rate limits are keyed by client address, so the proxy setting has to be
  // right for them to mean anything behind nginx.
  app.set("trust proxy", config.trustProxy);

  // Middleware
  app.use(cors({ origin: allowedOrigin() }));
  // Ahead of the body parser, the limiters and every route, so a refused
  // request is neither read, counted nor acted on.
  app.use("/api", refuseForeignOrigins(allowedOrigin()));
  app.use(express.json({ limit: config.jsonBodyLimit }));
  app.use(requestIdMiddleware);
  // Uploads are user-controlled bytes served from the app's own origin, and an
  // SVG can carry a <script> that runs on direct navigation — #342 stopped the
  // client choosing the stored extension, but image/svg+xml is still legitimately
  // stored as .svg. Serving every upload as a download with sniffing off removes
  // the whole class rather than guessing which types are active content, and it
  // costs nothing: Content-Disposition only binds top-level navigations, so
  // <img>/<audio>/<video> subresource loads still render inline (Issue #350).
  // Server-side readers open these files from disk and never come through here.
  app.use(
    "/uploads",
    express.static(path.join(__dirname, "..", "uploads"), {
      setHeaders(res) {
        res.setHeader("Content-Disposition", "attachment");
        res.setHeader("X-Content-Type-Options", "nosniff");
      },
    })
  );

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
  app.use("/api/pricing", pricingRoutes);
  app.use("/api/chat", chatRoutes);
  // Support direct /api/conversations routes as well as /api/chat/conversations
  app.use("/api/conversations", (req, res, next) => {
    req.url = "/conversations" + req.url;
    chatRoutes(req, res, next);
  });
  app.use("/api/generate", generationLimiter, generateRoutes);

  // Unmatched routes and every thrown error share one response shape
  app.use(notFoundHandler);
  app.use(errorHandler);

  return app;
}

/**
 * Hold a WebSocket handshake to the origin CORS allows (Issue #425).
 *
 * CORS only tells a browser what it may read, and a browser never applies it to
 * a WebSocket, so without this any web page could open /ws on a locally running
 * backend and drive it. The comparison is the one a browser makes against
 * Access-Control-Allow-Origin: exact, with "*" meaning every origin, so another
 * spelling of the allowed origin is another origin.
 *
 * No Origin header means the client is not a browser, which always sends one on
 * a WebSocket handshake. CORS lets those clients through and so does this. A
 * header that is present but empty is not absent, and is refused. The rule is
 * originAllowed's, the one REST requests are held to.
 */
function verifyWebSocketOrigin(allowed) {
  return ({ origin }, done) => {
    if (originAllowed(origin, allowed)) return done(true);
    done(false, 403, "Origin not allowed");
  };
}

/**
 * An HTTP server with the app mounted and the WebSocket endpoint attached.
 */
function createServer() {
  const app = createApp();
  const server = http.createServer(app);
  const wss = new WebSocketServer({
    server,
    path: "/ws",
    maxPayload: config.maxWsPayload,
    verifyClient: verifyWebSocketOrigin(allowedOrigin()),
  });
  setupWebSocket(wss);
  return { app, server, wss };
}

module.exports = { createApp, createServer };
