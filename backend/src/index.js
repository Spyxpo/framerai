const path = require("path");

require("dotenv").config({ path: path.join(__dirname, "..", ".env") });

const { createServer } = require("./app");
const { logger } = require("./services/logger");

const PORT = process.env.PORT || 3001;
const { server } = createServer();

server.listen(PORT, () => {
  logger.info(`FramerAI backend running on http://localhost:${PORT}`);
  logger.info(`WebSocket available at ws://localhost:${PORT}/ws`);
});
