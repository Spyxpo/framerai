/**
 * WebSocket service for real-time streaming responses.
 */

const { v4: uuidv4 } = require("uuid");
const { processMessage } = require("./model");

const MESSAGE_TYPES = ["text", "code", "image", "video", "audio"];
const MAX_MESSAGE_LENGTH = 8000;

/**
 * Validate an incoming chat frame the same way the REST route does, so a bad
 * frame gets a clear error instead of failing somewhere in the model service.
 */
function parseChatFrame(message) {
  const content = typeof message.content === "string" ? message.content.trim() : "";
  if (!content) throw new Error("content is required");
  if (content.length > MAX_MESSAGE_LENGTH) {
    throw new Error(`content must be at most ${MAX_MESSAGE_LENGTH} characters`);
  }

  const messageType = message.messageType || "text";
  if (!MESSAGE_TYPES.includes(messageType)) {
    throw new Error(`messageType must be one of: ${MESSAGE_TYPES.join(", ")}`);
  }

  return { content, messageType, conversationId: message.conversationId };
}

function setupWebSocket(wss) {
  wss.on("connection", (ws) => {
    const clientId = uuidv4();
    console.log(`WebSocket client connected: ${clientId}`);

    ws.on("message", async (data) => {
      try {
        const message = JSON.parse(data);

        if (message.type === "chat") {
          const { content, conversationId, messageType } = parseChatFrame(message);

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
