const path = require("path");

require("dotenv").config({ path: path.join(__dirname, "..", ".env") });

const { createServer } = require("./app");

const PORT = process.env.PORT || 3001;
const { server, wss } = createServer();

server.listen(PORT, () => {
  const actualPort = server.address() ? server.address().port : PORT;
  console.log(`FramerAI backend running on http://localhost:${actualPort}`);
  console.log(`WebSocket available at ws://localhost:${actualPort}/ws`);
});

function shutdown() {
  if (wss) {
    try {
      if (wss.clients) {
        for (const client of wss.clients) {
          client.terminate();
        }
      }
      wss.close();
    } catch (_) {}
  }
  if (typeof server.closeAllConnections === "function") {
    server.closeAllConnections();
  }
  server.close(() => {
    process.exit(0);
  });
  setTimeout(() => process.exit(0), 1000).unref();
}

process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
