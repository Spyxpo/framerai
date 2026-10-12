/**
 * A new message goes out with the id the website gave it (Issue #471).
 *
 * The server stores the message under an id of its own and keeps this one as
 * the message's clientId. When the ack or the REST response is lost, the sync
 * sent on reconnect still carries the website's id, and the server can only
 * recognise the message it already stored if that is the id it was sent with.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { useChat } from "../hooks/useChat";
import { api } from "../services/api";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

let mockConnected = true;
let mockSent = [];
let mockHandlers = new Map();

vi.mock("../services/api", () => ({
  api: {
    listConversations: vi.fn(() => Promise.resolve([])),
    createConversation: vi.fn(() => Promise.resolve({ id: "conv-1", title: "New Chat", version: 1, messages: [] })),
    getConversation: vi.fn((id) => Promise.resolve({ id, title: "New Chat", version: 1, messages: [] })),
    sendMessage: vi.fn(() =>
      Promise.resolve({ id: "server-reply", userMessageId: "server-user", content: "Echo", type: "text", version: 3 })
    ),
    syncConversation: vi.fn((id) => Promise.resolve({ conversation: { id, title: "New Chat", version: 2, messages: [] } })),
  },
}));

vi.mock("../services/websocket", () => ({
  WebSocketClient: class {
    constructor() {
      this.ws = { readyState: mockConnected ? 1 : 3 };
      mockSent = [];
      mockHandlers = new Map();
    }
    connect() {
      return Promise.resolve();
    }
    isConnected() {
      return mockConnected;
    }
    on(event, handler) {
      if (!mockHandlers.has(event)) mockHandlers.set(event, []);
      mockHandlers.get(event).push(handler);
      return () => {};
    }
    send(data) {
      mockSent.push(data);
    }
    sendApprovalResponse() {}
    disconnect() {}
  },
}));

async function startChat() {
  const { result } = renderHook(() => useChat({}));
  await act(async () => {
    await result.current.createConversation();
  });
  return result;
}

const userMessage = (result) => result.current.messages.find((m) => m.role === "user");

describe("Issue #471: the id a new message is sent with", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  it("goes out on the WebSocket frame as the id the message has on screen", async () => {
    mockConnected = true;
    const result = await startChat();

    await act(async () => {
      await result.current.sendMessage("hello", "text", []);
    });

    const frame = mockSent.find((f) => f.type === "chat");
    expect(frame.clientMessageId).toMatch(UUID);
    expect(frame.clientMessageId).toBe(userMessage(result).id);
  });

  it("goes out with a REST send, and stays on the message as its clientId once the server's id arrives", async () => {
    mockConnected = false;
    const result = await startChat();

    await act(async () => {
      await result.current.sendMessage("hello", "text", []);
    });

    expect(api.sendMessage).toHaveBeenCalledTimes(1);
    const options = api.sendMessage.mock.calls[0][5];
    expect(options).toEqual({ expectedVersion: 1, clientMessageId: expect.stringMatching(UUID) });
    await waitFor(() => expect(userMessage(result).id).toBe("server-user"));
    expect(userMessage(result).clientId).toBe(options.clientMessageId);
  });

  it("is the id a reconnect sync carries for a message whose response has not arrived", async () => {
    mockConnected = false;
    let answer;
    api.sendMessage.mockImplementationOnce(
      () => new Promise((resolve) => {
        answer = resolve;
      })
    );
    const result = await startChat();

    let sending;
    act(() => {
      sending = result.current.sendMessage("hello", "text", []);
    });
    await waitFor(() => expect(api.sendMessage).toHaveBeenCalledTimes(1));
    const sentId = api.sendMessage.mock.calls[0][5].clientMessageId;

    // The socket comes back while the REST turn is still in flight.
    await act(async () => {
      for (const handler of mockHandlers.get("reconnect") || []) handler();
    });
    await waitFor(() => expect(api.syncConversation).toHaveBeenCalled());
    const payload = api.syncConversation.mock.calls[0][1];
    const synced = payload.messages.find((m) => m.role === "user");
    expect(synced.id).toBe(sentId);

    await act(async () => {
      answer({ id: "server-reply", userMessageId: "server-user", content: "Echo", type: "text", version: 3 });
      await sending;
    });
  });
});
