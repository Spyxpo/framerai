/**
 * A conversation the server lost, still held here (Issue #503).
 *
 * The server's store is in memory: a restart empties it and the idle sweep
 * drops a conversation past its TTL, while this browser keeps every
 * conversation. A turn sent for one the server lost is refused, over the
 * WebSocket with a CONVERSATION_NOT_FOUND frame naming the message, over REST
 * with 404. The website then restores the conversation from its own copy
 * (POST /chat/conversations/:id/restore) and, only once that has succeeded,
 * sends the refused message again, once. If the restore fails, sending in that
 * conversation stops and the user is offered a new chat: a reply that knows
 * nothing of the conversation above it is never shown as if it did.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, renderHook, act, waitFor } from "@testing-library/react";
import { useChat } from "../hooks/useChat";
import { api } from "../services/api";
import Chat from "../components/Chat/Chat";

const CONV = "0c0c0c0c-0000-4000-8000-000000000001";
const NEW_CONV = "0c0c0c0c-0000-4000-8000-000000000002";

let mockConnected = true;
let mockSent = [];
let mockHandlers = new Map();

vi.mock("../services/api", () => ({
  api: {
    listConversations: vi.fn(() => Promise.resolve([])),
    createConversation: vi.fn(() => Promise.resolve({ id: "0c0c0c0c-0000-4000-8000-000000000001", title: "New Chat", version: 1, messages: [] })),
    getConversation: vi.fn(() => Promise.reject(Object.assign(new Error("Conversation not found"), { status: 404 }))),
    sendMessage: vi.fn(() => Promise.resolve({ id: "rest-reply", userMessageId: "rest-user", content: "Echo", type: "text", version: 3 })),
    syncConversation: vi.fn(() => Promise.reject(Object.assign(new Error("Conversation not found"), { status: 404 }))),
    restoreConversation: vi.fn((id, payload) => Promise.resolve({ id, title: payload.title || "New Chat", version: 1, messages: payload.messages })),
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

async function fire(event, data) {
  await act(async () => {
    for (const handler of mockHandlers.get(event) || []) handler(data);
  });
}

const chatFrames = () => mockSent.filter((f) => f.type === "chat");
const lastFrame = () => chatFrames()[chatFrames().length - 1];

/** A conversation with one finished exchange, carried over the WebSocket as the server sends it. */
async function conversationWithHistory() {
  const { result } = renderHook(() => useChat({}));
  await act(async () => {
    await result.current.createConversation();
  });
  await act(async () => {
    await result.current.sendMessage("My name is Ada.", "text", []);
  });
  await fire("ack", { type: "ack", conversationId: CONV, messageId: "0c0c0c0c-0000-4000-8000-0000000000a1", assistantMessageId: "0c0c0c0c-0000-4000-8000-0000000000b1", version: 2 });
  await fire("stream", { type: "stream", conversationId: CONV, messageId: "0c0c0c0c-0000-4000-8000-0000000000b1", content: "Nice to meet you, Ada.", done: true, responseType: "text", version: 3 });
  await waitFor(() => expect(result.current.messages.map((m) => m.content)).toEqual(["My name is Ada.", "Nice to meet you, Ada."]));
  return result;
}

/** Send a message the server will refuse; returns the id it went out with. */
async function sendRefused(result, content = "What is my name?") {
  await act(async () => {
    await result.current.sendMessage(content, "text", []);
  });
  return lastFrame().clientMessageId;
}

const refusal = (clientMessageId) => ({
  type: "error",
  code: "CONVERSATION_NOT_FOUND",
  conversationId: CONV,
  clientMessageId,
  message: "Conversation not found",
});

const contents = (result) => result.current.messages.map((m) => `${m.role}: ${m.type === "error" ? "[error] " : ""}${m.content}`);

describe("Issue #503: a conversation the server lost", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    mockConnected = true;
  });

  it("is restored from the whole local history before the refused message is sent again, once", async () => {
    const result = await conversationWithHistory();
    let finish;
    api.restoreConversation.mockImplementationOnce(
      (id, payload) => new Promise((resolve) => {
        finish = () => resolve({ id, title: "New Chat", version: 1, messages: payload.messages });
      })
    );
    const clientMessageId = await sendRefused(result);
    expect(chatFrames()).toHaveLength(2);

    await fire("error", refusal(clientMessageId));
    await waitFor(() => expect(api.restoreConversation).toHaveBeenCalledTimes(1));

    const [id, payload] = api.restoreConversation.mock.calls[0];
    expect(id).toBe(CONV);
    expect(payload.messages.map((m) => [m.id, m.role, m.content, m.completed])).toEqual([
      ["0c0c0c0c-0000-4000-8000-0000000000a1", "user", "My name is Ada.", undefined],
      ["0c0c0c0c-0000-4000-8000-0000000000b1", "assistant", "Nice to meet you, Ada.", true],
    ]);
    // Nothing goes out again until the restore has answered.
    expect(chatFrames()).toHaveLength(2);

    await act(async () => {
      finish();
    });
    await waitFor(() => expect(chatFrames()).toHaveLength(3));
    const resent = lastFrame();
    expect(resent).toMatchObject({ conversationId: CONV, content: "What is my name?", clientMessageId, expectedVersion: 1 });
    // On screen: the history, and the message once, waiting for its reply.
    expect(contents(result)).toEqual([
      "user: My name is Ada.",
      "assistant: Nice to meet you, Ada.",
      "user: What is my name?",
      "assistant: ",
    ]);
    expect(result.current.recoveryBlocked).toBe(false);

    // And it is never sent a third time.
    await new Promise((r) => setTimeout(r, 50));
    expect(chatFrames()).toHaveLength(3);
  });

  it("starts one restore and sends the message again once, however often the refusal arrives", async () => {
    const result = await conversationWithHistory();
    let finish;
    api.restoreConversation.mockImplementationOnce(
      (id, payload) => new Promise((resolve) => {
        finish = () => resolve({ id, version: 1, messages: payload.messages });
      })
    );
    const clientMessageId = await sendRefused(result);
    await fire("error", refusal(clientMessageId));
    await fire("error", refusal(clientMessageId));
    // A reconnect while the restore is out syncs as it always does, and restores nothing more.
    await fire("reconnect");
    await fire("error", refusal(clientMessageId));

    await act(async () => {
      finish();
    });
    await waitFor(() => expect(chatFrames()).toHaveLength(3));
    await new Promise((r) => setTimeout(r, 50));
    expect(api.restoreConversation).toHaveBeenCalledTimes(1);
    expect(chatFrames()).toHaveLength(3);
    expect(contents(result).filter((c) => c === "user: What is my name?")).toHaveLength(1);
  });

  it("does not send a message a third time when the server refuses it again after the restore", async () => {
    const result = await conversationWithHistory();
    const clientMessageId = await sendRefused(result);
    await fire("error", refusal(clientMessageId));
    await waitFor(() => expect(chatFrames()).toHaveLength(3));

    await fire("error", refusal(clientMessageId));
    await waitFor(() => expect(result.current.recoveryBlocked).toBe(true));
    expect(api.restoreConversation).toHaveBeenCalledTimes(1);
    expect(chatFrames()).toHaveLength(3);
  });

  it("blocks sending and keeps the history when the restore fails, and a new chat still works", async () => {
    const result = await conversationWithHistory();
    api.restoreConversation.mockImplementationOnce(() =>
      Promise.reject(Object.assign(new Error("Request failed"), { status: 500 }))
    );
    const clientMessageId = await sendRefused(result);
    await fire("error", refusal(clientMessageId));

    await waitFor(() => expect(result.current.recoveryBlocked).toBe(true));
    // The history stays, the message stays, and its reply says it was not sent.
    expect(contents(result).slice(0, 3)).toEqual([
      "user: My name is Ada.",
      "assistant: Nice to meet you, Ada.",
      "user: What is my name?",
    ]);
    expect(result.current.messages[3].type).toBe("error");
    expect(result.current.streaming).toBe(false);

    // Nothing more is sent in this conversation.
    await act(async () => {
      await result.current.sendMessage("Hello?", "text", []);
    });
    expect(chatFrames()).toHaveLength(2);
    expect(api.sendMessage).not.toHaveBeenCalled();

    // A new chat is not blocked.
    api.createConversation.mockImplementationOnce(() => Promise.resolve({ id: NEW_CONV, title: "New Chat", version: 1, messages: [] }));
    await act(async () => {
      await result.current.createConversation();
    });
    expect(result.current.recoveryBlocked).toBe(false);
    await act(async () => {
      await result.current.sendMessage("A fresh start", "text", []);
    });
    expect(lastFrame()).toMatchObject({ conversationId: NEW_CONV, content: "A fresh start" });
  });

  it("restores the same way when a REST send is answered 404, and sends it again once", async () => {
    mockConnected = false;
    const { result } = renderHook(() => useChat({}));
    await act(async () => {
      await result.current.createConversation();
    });
    api.sendMessage
      .mockImplementationOnce(() => Promise.resolve({ id: "0c0c0c0c-0000-4000-8000-0000000000b1", userMessageId: "0c0c0c0c-0000-4000-8000-0000000000a1", content: "Nice to meet you, Ada.", type: "text", version: 3 }))
      .mockImplementationOnce(() => Promise.reject(Object.assign(new Error("Conversation not found"), { status: 404, code: "NOT_FOUND" })))
      .mockImplementationOnce(() => Promise.resolve({ id: "0c0c0c0c-0000-4000-8000-0000000000b2", userMessageId: "0c0c0c0c-0000-4000-8000-0000000000a2", content: "Your name is Ada.", type: "text", version: 3 }));

    await act(async () => {
      await result.current.sendMessage("My name is Ada.", "text", []);
    });
    await act(async () => {
      await result.current.sendMessage("What is my name?", "text", []);
    });

    await waitFor(() => expect(api.sendMessage).toHaveBeenCalledTimes(3));
    const [id, payload] = api.restoreConversation.mock.calls[0];
    expect(id).toBe(CONV);
    expect(payload.messages.map((m) => [m.role, m.content])).toEqual([
      ["user", "My name is Ada."],
      ["assistant", "Nice to meet you, Ada."],
    ]);
    const refusedId = api.sendMessage.mock.calls[1][5].clientMessageId;
    expect(api.sendMessage.mock.calls[2][5]).toEqual({ expectedVersion: 1, clientMessageId: refusedId });
    expect(api.restoreConversation.mock.invocationCallOrder[0]).toBeLessThan(api.sendMessage.mock.invocationCallOrder[2]);
    await waitFor(() =>
      expect(contents(result)).toEqual([
        "user: My name is Ada.",
        "assistant: Nice to meet you, Ada.",
        "user: What is my name?",
        "assistant: Your name is Ada.",
      ])
    );
  });

  it("sends again with the server's version when another tab restored the conversation first (409)", async () => {
    const result = await conversationWithHistory();
    api.restoreConversation.mockImplementationOnce(() =>
      Promise.reject(Object.assign(new Error("Conversation already exists"), { status: 409, code: "CONVERSATION_EXISTS" }))
    );
    api.getConversation.mockImplementationOnce((id) => Promise.resolve({ id, title: "New Chat", version: 4, messages: [] }));
    const clientMessageId = await sendRefused(result);
    await fire("error", refusal(clientMessageId));

    await waitFor(() => expect(chatFrames()).toHaveLength(3));
    expect(lastFrame()).toMatchObject({ clientMessageId, expectedVersion: 4 });
    expect(result.current.recoveryBlocked).toBe(false);
  });

  it("leaves ordinary turns alone: no restore without a refusal", async () => {
    const result = await conversationWithHistory();
    await act(async () => {
      await result.current.sendMessage("Thanks!", "text", []);
    });
    await fire("reconnect");
    expect(api.restoreConversation).not.toHaveBeenCalled();
    expect(result.current.recoveryBlocked).toBe(false);
  });
});

describe("Issue #503: the notice for a conversation that could not be restored", () => {
  const props = (overrides = {}) => ({
    messages: [],
    loading: false,
    streaming: false,
    loadingMessages: false,
    error: null,
    sidebarOpen: true,
    onSend: vi.fn(),
    onToggleSidebar: vi.fn(),
    onDismissError: vi.fn(),
    onOpenSettings: vi.fn(),
    onFocusSidebar: vi.fn(),
    onFocusSidebarSettings: vi.fn(),
    focusRef: { current: null },
    textareaFocusRef: { current: null },
    chatSettingsFocusRef: { current: null },
    ...overrides,
  });

  it("says why sending stopped, blocks the composer, and offers a new chat", () => {
    const onStartNewChat = vi.fn();
    render(<Chat {...props({ recoveryBlocked: true, onStartNewChat })} />);

    expect(screen.getByText(/could not be restored/i)).toBeInTheDocument();
    expect(screen.getByLabelText("Message input")).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: /start a new chat/i }));
    expect(onStartNewChat).toHaveBeenCalledTimes(1);
  });

  it("is not shown for a conversation that can be used", () => {
    render(<Chat {...props()} />);
    expect(screen.queryByText(/could not be restored/i)).not.toBeInTheDocument();
    expect(screen.getByLabelText("Message input")).not.toBeDisabled();
  });
});
