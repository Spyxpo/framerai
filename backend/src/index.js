const path = require("path");

require("dotenv").config({ path: path.join(__dirname, "..", ".env") });

const { createServer } = require("./app");

const PORT = process.env.PORT || 3001;
const { server } = createServer();

server.listen(PORT, () => {
  console.log(`FramerAI backend running on http://localhost:${PORT}`);
  console.log(`WebSocket available at ws://localhost:${PORT}/ws`);
});
