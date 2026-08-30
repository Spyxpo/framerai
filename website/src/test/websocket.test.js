import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { WebSocketClient } from "../services/websocket.js";

describe("WebSocketClient", () => {
  let mockWebSocket;
  let client;

  beforeEach(() => {
    // Mock WebSocket constructor
    mockWebSocket = {
      readyState: 0, // CONNECTING
      onopen: null,
      onclose: null,
      onmessage: null,
      onerror: null,
      close: vi.fn(),
      send: vi.fn(),
    };

    global.WebSocket = class MockWebSocket {
      constructor() {
        Object.assign(this, mockWebSocket);
      }
    };
    global.WebSocket.CONNECTING = 0;
    global.WebSocket.OPEN = 1;
    global.WebSocket.CLOSING = 2;
    global.WebSocket.CLOSED = 3;

    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  describe("disconnect() reconnect bug", () => {
    it("REGRESSION TEST: disconnect() should NOT trigger automatic reconnect", async () => {
      // Create client
      client = new WebSocketClient("ws://localhost:8080/ws");

      // Start connection
      const connectPromise = client.connect();

      // Simulate successful connection
      client.ws.readyState = WebSocket.OPEN;
      client.ws.onopen();
      await connectPromise;

      // Capture the onclose handler before disconnect
      const oncloseHandler = client.ws.onclose;

      // Spy on connect method to detect reconnect attempts
      const connectSpy = vi.spyOn(client, "connect");

      // User intentionally disconnects
      client.disconnect();

      // Verify disconnect was called
      expect(client.ws).toBeNull();

      // Simulate the close event firing (which triggers onclose handler)
      oncloseHandler();

      // Fast-forward past the reconnect delay (1 second)
      vi.advanceTimersByTime(1500);

      // BUG VERIFICATION: connect() should NOT have been called after intentional disconnect
      // This test should FAIL with current implementation because onclose always reconnects
      expect(connectSpy).not.toHaveBeenCalled();
    });

    it("should allow reconnect after network-initiated close", async () => {
      // Create client
      client = new WebSocketClient("ws://localhost:8080/ws");

      // Start connection
      const connectPromise = client.connect();
      client.ws.readyState = WebSocket.OPEN;
      client.ws.onopen();
      await connectPromise;

      const connectSpy = vi.spyOn(client, "connect");

      // Server/network closes the connection (NOT user-initiated)
      // Do NOT call client.disconnect() - just trigger onclose
      client.ws.onclose();

      // Fast-forward past reconnect delay
      vi.advanceTimersByTime(1500);

      // This SHOULD reconnect automatically
      expect(connectSpy).toHaveBeenCalledTimes(1);
    });

    it("should cancel pending reconnect timer on disconnect", async () => {
      // Create client
      client = new WebSocketClient("ws://localhost:8080/ws");

      // Start connection
      const connectPromise = client.connect();
      client.ws.readyState = WebSocket.OPEN;
      client.ws.onopen();
      await connectPromise;

      // Simulate network close (which schedules a reconnect)
      client.ws.onclose();

      // Verify reconnect timer is scheduled
      expect(client.reconnectTimer).not.toBeNull();

      // User calls disconnect before the timer fires
      client.disconnect();

      // Verify timer is cancelled
      expect(client.reconnectTimer).toBeNull();

      // Fast-forward past reconnect delay
      const connectSpy = vi.spyOn(client, "connect");
      vi.advanceTimersByTime(1500);

      // Verify no reconnect happened
      expect(connectSpy).not.toHaveBeenCalled();
    });

    it("should allow explicit connect after intentional disconnect", async () => {
      // Create client and connect
      client = new WebSocketClient("ws://localhost:8080/ws");
      let connectPromise = client.connect();
      client.ws.readyState = WebSocket.OPEN;
      client.ws.onopen();
      await connectPromise;

      // Intentionally disconnect
      const oncloseHandler = client.ws.onclose;
      client.disconnect();
      oncloseHandler();

      // Fast-forward to ensure no automatic reconnect
      vi.advanceTimersByTime(2000);
      expect(client.ws).toBeNull();

      // User explicitly calls connect again
      connectPromise = client.connect();
      client.ws.readyState = WebSocket.OPEN;
      client.ws.onopen();
      await connectPromise;

      // Should be connected now
      expect(client.ws).not.toBeNull();
      expect(client.ws.readyState).toBe(WebSocket.OPEN);
    });
  });

  describe("basic functionality", () => {
    it("connects successfully", async () => {
      client = new WebSocketClient("ws://localhost:8080/ws");
      const connectPromise = client.connect();

      client.ws.readyState = WebSocket.OPEN;
      client.ws.onopen();

      await expect(connectPromise).resolves.toBeUndefined();
      expect(client.ws).toBeDefined();
    });

    it("sends messages when connected", () => {
      client = new WebSocketClient("ws://localhost:8080/ws");
      client.ws = {
        readyState: WebSocket.OPEN,
        send: vi.fn(),
      };

      client.send({ type: "test", data: "hello" });

      expect(client.ws.send).toHaveBeenCalledWith(
        JSON.stringify({ type: "test", data: "hello" })
      );
    });

    it("registers and triggers message listeners", async () => {
      client = new WebSocketClient("ws://localhost:8080/ws");
      const connectPromise = client.connect();
      client.ws.readyState = WebSocket.OPEN;
      client.ws.onopen();
      await connectPromise;

      const handler = vi.fn();
      client.on("test_message", handler);

      // Simulate incoming message
      const messageEvent = {
        data: JSON.stringify({ type: "test_message", payload: "data" }),
      };
      client.ws.onmessage(messageEvent);

      expect(handler).toHaveBeenCalledWith({
        type: "test_message",
        payload: "data",
      });
    });
  });
});
