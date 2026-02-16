/**
 * WebSocket service for real-time streaming responses.
 */

const { v4: uuidv4 } = require("uuid");
const { processMessage } = require("./model");

function setupWebSocket(wss) {
  wss.on("connection", (ws) => {
    const clientId = uuidv4();
    console.log(`WebSocket client connected: ${clientId}`);

    ws.on("message", async (data) => {
      try {
        const message = JSON.parse(data);

        if (message.type === "chat") {
          const { content, conversationId, messageType = "text" } = message;

          // Send acknowledgment
          ws.send(
            JSON.stringify({
              type: "ack",
              messageId: uuidv4(),
              conversationId,
            })
          );

          // Send typing indicator
          ws.send(JSON.stringify({ type: "typing", conversationId }));

          // Process and stream response
          const messages = [{ role: "user", content }];
          const response = await processMessage(messages, messageType);

          // Simulate streaming by sending chunks
          const words = response.content.split(" ");
          let accumulated = "";

          for (let i = 0; i < words.length; i++) {
            accumulated += (i > 0 ? " " : "") + words[i];
            ws.send(
              JSON.stringify({
                type: "stream",
                conversationId,
                content: accumulated,
                done: i === words.length - 1,
                responseType: response.type,
                metadata: i === words.length - 1 ? response.metadata : undefined,
              })
            );
            // Simulate token generation delay
            await new Promise((r) => setTimeout(r, 20 + Math.random() * 30));
          }
        }

        if (message.type === "ping") {
          ws.send(JSON.stringify({ type: "pong" }));
        }
      } catch (err) {
        ws.send(JSON.stringify({ type: "error", message: err.message }));
      }
    });

    ws.on("close", () => {
      console.log(`WebSocket client disconnected: ${clientId}`);
    });
  });
}

module.exports = { setupWebSocket };
