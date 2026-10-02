/**
 * Regression tests for Issue #394: client and server generate different message IDs.
 *
 * The client mints a temporary id for every message it shows so the interface can
 * render and stream into it at once. The server stores the same message under an
 * id of its own, and the client never adopted it, so "Branch from here" asked the
 * server for an id it had never seen: "Message not found in conversation", until
 * a reload replaced the client's copy with the server's.
 *
 * The server's id is the authoritative one. The temporary id is kept for the
 * optimistic bubble and handed over when the server reports the persisted
 * message: the user message at the ack (WebSocket) or in the response (REST), the
 * reply when its last frame or response arrives.
 *
 * Requirements verified:
 *  1. A newly created message ends up with the server's id in client state.
 *  2. No message keeps a temporary id that conflicts with the server's.
 *  3. "Branch from here" sends the authoritative id.
 *  4. The server finds the message by that id.
 *  5. Messages loaded from the server keep their ids and can be branched.
 *  6. Streaming and reconciliation never leave a duplicate message.
 *  7. Ordering, errors, retry and concurrent conversations behave as before.
 *  8. A message keeps its place in the rendered list when its id changes.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, render, screen, act, waitFor } from "@testing-library/react";
import Chat from "../components/Chat/Chat";
import { useChat } from "../hooks/useChat";
import { api } from "../services/api";
import { STORAGE_KEY } from "../utils/storage";

const doubles = vi.hoisted(() => ({ sockets: [] }));

vi.mock("../services/api", () => ({
  api: {
    listConversations: vi.fn(),
    createConversation: vi.fn(),
    getConversation: vi.fn(),
    deleteConversation: vi.fn(),
    sendMessage: vi.fn(),
    branchConversation: vi.fn(),
  },
}));

vi.mock("../services/websocket", () => ({
  WebSocketClient: class {
    constructor() {
      this.ws = { readyState: 1 };
      this.handlers = new Map();
      this.sent = [];
      doubles.sockets.push(this);
    }
    connect() {
      return Promise.resolve();
    }
    on(type, handler) {
      if (!this.handlers.has(type)) this.handlers.set(type, []);
      this.handlers.get(type).push(handler);
      return () => {};
    }
    emit(type, data) {
      (this.handlers.get(type) || []).forEach((handler) => handler(data));
    }
    send(payload) {
      this.sent.push(payload);
    }
    disconnect() {}
  },
}));

const clone = (value) => JSON.parse(JSON.stringify(value));

/**
 * A server that behaves like backend/src/routes/chat.js and
 * backend/src/services/websocket.js where ids are concerned: it stores every
 * turn under ids it mints itself, tells the client what they are, and looks a
 * message up by id when asked to branch, refusing one it has never seen.
 */
function makeServer() {
  const store = new Map();
  const now = () => new Date().toISOString();

  return {
    conversation: (id) => store.get(id),

    create(messages = []) {
      const conversation = { id: crypto.randomUUID(), title: "New Chat", messages };
      store.set(conversation.id, conversation);
      return conversation;
    },

    // REST: the reply, plus the id the user's message was stored under.
    send(conversationId, content, type = "text") {
      const conversation = store.get(conversationId);
      const user = { id: crypto.randomUUID(), role: "user", content, type, timestamp: now() };
      const reply = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: `reply to: ${content}`,
        type: "text",
        metadata: { model: "test-model" },
        timestamp: now(),
      };
      conversation.messages.push(user, reply);
      return { ...reply, userMessageId: user.id };
    },

    // WebSocket: the user's turn is stored when it is acknowledged, the reply
    // when its last frame is sent. Each step is driven by the test.
    beginTurn(frame, socket) {
      const conversation = store.get(frame.conversationId);
      const user = { id: crypto.randomUUID(), role: "user", content: frame.content, type: "text", timestamp: now() };
      const replyId = crypto.randomUUID();
      const base = { type: "stream", conversationId: conversation.id, responseType: "text" };
      conversation.messages.push(user);

      return {
        userId: user.id,
        assistantId: replyId,
        ack: () => socket.emit("ack", { type: "ack", messageId: user.id, conversationId: conversation.id }),
        partial: (content) => socket.emit("stream", { ...base, content, done: false }),
        done: (content) => {
          conversation.messages.push({
            id: replyId,
            role: "assistant",
            content,
            type: "text",
            timestamp: now(),
          });
          socket.emit("stream", { ...base, content, done: true, metadata: { model: "test-model" }, messageId: replyId });
        },
        // An audio reply's last chunk is its completion frame.
        doneAudio: (content, chunkData = "AAAA") => {
          conversation.messages.push({ id: replyId, role: "assistant", content, type: "audio", timestamp: now() });
          socket.emit("stream", {
            ...base,
            responseType: "audio",
            content,
            done: true,
            messageId: replyId,
            metadata: { chunk: 0, totalChunks: 1, chunkData, sampleRate: 24000, channels: 1, bitsPerSample: 16, url: "/uploads/generated/x.wav" },
          });
        },
        fail: (message) => socket.emit("error", { type: "error", conversationId: conversation.id, message }),
      };
    },

    branch(conversationId, messageId) {
      const conversation = store.get(conversationId);
      const index = conversation.messages.findIndex((m) => m.id === messageId);
      if (index === -1) throw new Error("Message not found in conversation");
      const branch = {
        id: crypto.randomUUID(),
        title: `${conversation.title} (Branch)`,
        parentConversationId: conversationId,
        branchedFromMessageId: messageId,
        messages: clone(conversation.messages.slice(0, index + 1)),
        createdAt: now(),
      };
      store.set(branch.id, branch);
      return branch;
    },
  };
}

describe("Issue #394: one id per message, shared by client and server", () => {
  let server;
  let socket;

  beforeEach(() => {
    localStorage.clear();
    doubles.sockets.length = 0;
    server = makeServer();
    api.listConversations.mockReset().mockResolvedValue([]);
    api.createConversation.mockReset().mockImplementation(async () => clone(server.create()));
    api.getConversation.mockReset().mockImplementation(async (id) => clone(server.conversation(id)));
    api.deleteConversation.mockReset().mockResolvedValue({ success: true });
    api.sendMessage
      .mockReset()
      .mockImplementation(async (conversationId, content, type) => server.send(conversationId, content, type));
    api.branchConversation
      .mockReset()
      .mockImplementation(async (conversationId, messageId) => server.branch(conversationId, messageId));
  });

  async function startChat() {
    const hook = renderHook(() => useChat({}));
    await waitFor(() => expect(hook.result.current.loadingConversations).toBe(false));
    socket = doubles.sockets[doubles.sockets.length - 1];
    await act(async () => {
      await hook.result.current.createConversation();
    });
    return hook;
  }

  /**
   * Reload a conversation the way a click does: the synchronous part renders the
   * local messages first, and the server's copy replaces them when it arrives.
   * Awaiting selectConversation inside one act() batches that order away and
   * leaves the stale local copy on screen.
   */
  async function reloadFromServer(result, conversationId) {
    let pending;
    act(() => {
      pending = result.current.selectConversation(conversationId);
    });
    await act(async () => {
      await pending;
    });
  }

  /** Send over the open socket; the server has seen the frame but answered nothing yet. */
  async function sendOverSocket(result, content) {
    await act(async () => {
      await result.current.sendMessage(content);
    });
    return server.beginTurn(socket.sent[socket.sent.length - 1], socket);
  }

  // ─── 1 & 2. The server's id replaces the temporary one ──────────────────────
  describe("over WebSocket", () => {
    it("the user message takes the id the server stored it under when the ack arrives", async () => {
      const { result } = await startChat();
      await act(async () => {
        await result.current.sendMessage("hello");
      });
      const [user, reply] = result.current.messages;
      const turn = server.beginTurn(socket.sent[0], socket);

      expect(user.id).not.toBe(turn.userId); // the client minted its own

      await act(async () => {
        turn.ack();
      });

      expect(result.current.messages).toHaveLength(2);
      expect(result.current.messages[0].id).toBe(turn.userId);
      expect(result.current.messages[1].id).toBe(reply.id); // the reply is still in flight
    });

    it("the reply takes the server's id when it completes, and no temporary id survives", async () => {
      const { result } = await startChat();
      const turn = await sendOverSocket(result, "hello");
      const temporary = result.current.messages.map((m) => m.id);

      await act(async () => {
        turn.ack();
        turn.partial("hello th");
      });
      expect(result.current.messages[1].content).toBe("hello th");
      expect(result.current.messages[1].id).toBe(temporary[1]); // still streaming under its own id

      await act(async () => {
        turn.done("hello there");
      });

      const [user, reply] = result.current.messages;
      expect([user.id, reply.id]).toEqual([turn.userId, turn.assistantId]);
      for (const message of result.current.messages) {
        expect(temporary).not.toContain(message.id);
      }
      expect(reply.content).toBe("hello there");
      expect(reply.completed).toBe(true);
      expect(result.current.streaming).toBe(false);
      // The temporary id lives on only as the key the list renders the bubble under.
      expect([user.clientId, reply.clientId]).toEqual(temporary);
    });

    it("every message of several turns ends up under the id the server stored it under", async () => {
      const { result } = await startChat();
      const conversationId = result.current.activeConversation;

      for (const text of ["one", "two", "three"]) {
        const turn = await sendOverSocket(result, text);
        await act(async () => {
          turn.ack();
          turn.done(`reply to: ${text}`);
        });
      }

      const stored = server.conversation(conversationId).messages;
      expect(stored).toHaveLength(6);
      expect(result.current.messages.map((m) => m.id)).toEqual(stored.map((m) => m.id));
      expect(result.current.messages.map((m) => m.content)).toEqual(stored.map((m) => m.content));
      expect(new Set(result.current.messages.map((m) => m.id)).size).toBe(6);
    });
  });

  describe("over REST", () => {
    it("the response's ids replace both temporary ids", async () => {
      const { result } = await startChat();
      const conversationId = result.current.activeConversation;
      socket.ws.readyState = 3; // closed, so the hook falls back to REST

      let respond;
      api.sendMessage.mockImplementationOnce(
        (id, content, type) =>
          new Promise((resolve) => {
            respond = () => resolve(server.send(id, content, type));
          })
      );
      let sending;
      await act(async () => {
        sending = result.current.sendMessage("hello");
      });
      const temporary = result.current.messages.map((m) => m.id);
      expect(temporary).toHaveLength(2);

      await act(async () => {
        respond();
        await sending;
      });

      const stored = server.conversation(conversationId).messages;
      expect(result.current.messages).toHaveLength(2);
      expect(result.current.messages.map((m) => m.id)).toEqual(stored.map((m) => m.id));
      for (const message of result.current.messages) {
        expect(temporary).not.toContain(message.id);
      }
      expect(result.current.messages[1].content).toBe("reply to: hello");
      expect(result.current.messages[1].completed).toBe(true);
    });

    it("an older server that names no ids leaves the temporary ones in place", async () => {
      const { result } = await startChat();
      socket.ws.readyState = 3;
      api.sendMessage.mockResolvedValueOnce({ content: "a reply", type: "text" });

      await act(async () => {
        await result.current.sendMessage("hello");
      });

      expect(result.current.messages).toHaveLength(2);
      expect(result.current.messages.every((m) => typeof m.id === "string" && m.id)).toBe(true);
      expect(result.current.messages[1].content).toBe("a reply");
      expect(result.current.messages[1].completed).toBe(true);
      expect(result.current.error).toBeNull();
    });
  });

  // ─── 3 & 4. Branch from here ────────────────────────────────────────────────
  describe("Branch from here", () => {
    it("sends the authoritative id of a reply, and the server finds it (WebSocket)", async () => {
      const { result } = await startChat();
      const conversationId = result.current.activeConversation;
      const turn = await sendOverSocket(result, "hello");
      await act(async () => {
        turn.ack();
        turn.done("hello there");
      });

      let branch;
      await act(async () => {
        branch = await result.current.branchConversation(result.current.messages[1].id);
      });

      expect(api.branchConversation).toHaveBeenCalledWith(conversationId, turn.assistantId);
      expect(result.current.error).toBeNull();
      expect(branch.parentConversationId).toBe(conversationId);
      expect(result.current.activeConversation).toBe(branch.id);
      expect(result.current.messages.map((m) => m.id)).toEqual([turn.userId, turn.assistantId]);
    });

    it("sends the authoritative id of the user's message, and the server finds it (WebSocket)", async () => {
      const { result } = await startChat();
      const conversationId = result.current.activeConversation;
      const turn = await sendOverSocket(result, "hello");
      await act(async () => {
        turn.ack();
        turn.done("hello there");
      });

      await act(async () => {
        await result.current.branchConversation(result.current.messages[0].id, conversationId);
      });

      expect(api.branchConversation).toHaveBeenCalledWith(conversationId, turn.userId);
      expect(result.current.error).toBeNull();
      expect(result.current.messages.map((m) => m.id)).toEqual([turn.userId]);
    });

    it("works straight after a REST reply, without a reload", async () => {
      const { result } = await startChat();
      const conversationId = result.current.activeConversation;
      socket.ws.readyState = 3;
      await act(async () => {
        await result.current.sendMessage("hello");
      });
      const stored = server.conversation(conversationId).messages;

      await act(async () => {
        await result.current.branchConversation(result.current.messages[1].id, conversationId);
      });

      expect(api.branchConversation).toHaveBeenCalledWith(conversationId, stored[1].id);
      expect(result.current.error).toBeNull();
      expect(result.current.messages.map((m) => m.id)).toEqual([stored[0].id, stored[1].id]);
    });

    it("is still refused for an id the server never stored", async () => {
      const { result } = await startChat();
      const conversationId = result.current.activeConversation;
      // A message the client holds but the server does not: the failure the
      // reconciliation removes for real messages must still be reported for this one.
      await act(async () => {
        await result.current.sendMessage("hello");
      });
      const unreconciled = result.current.messages[1].id;

      await act(async () => {
        await result.current.branchConversation(unreconciled, conversationId);
      });

      expect(result.current.error).toBe("Message not found in conversation");
    });
  });

  // ─── 5. Messages loaded from the server ─────────────────────────────────────
  describe("messages loaded from the server", () => {
    it("keep their ids and can be branched", async () => {
      const conversation = server.create([
        { id: crypto.randomUUID(), role: "user", content: "an earlier question", type: "text", timestamp: "2026-10-01T10:00:00.000Z" },
        { id: crypto.randomUUID(), role: "assistant", content: "an earlier answer", type: "text", timestamp: "2026-10-01T10:00:01.000Z" },
      ]);
      const serverIds = conversation.messages.map((m) => m.id);
      const { result } = await startChat();

      await act(async () => {
        await result.current.selectConversation(conversation.id);
      });

      expect(result.current.messages.map((m) => m.id)).toEqual(serverIds);
      expect(result.current.messages.every((m) => m.clientId === undefined)).toBe(true);

      await act(async () => {
        await result.current.branchConversation(serverIds[1]);
      });

      expect(api.branchConversation).toHaveBeenCalledWith(conversation.id, serverIds[1]);
      expect(result.current.error).toBeNull();
      expect(result.current.messages.map((m) => m.id)).toEqual(serverIds);
    });

    it("are not renamed when a later turn is reconciled", async () => {
      const conversation = server.create([
        { id: crypto.randomUUID(), role: "user", content: "earlier", type: "text", timestamp: "2026-10-01T10:00:00.000Z" },
        { id: crypto.randomUUID(), role: "assistant", content: "earlier answer", type: "text", timestamp: "2026-10-01T10:00:01.000Z" },
      ]);
      const earlierIds = conversation.messages.map((m) => m.id);
      const { result } = await startChat();
      await act(async () => {
        await result.current.selectConversation(conversation.id);
      });

      const turn = await sendOverSocket(result, "a new question");
      await act(async () => {
        turn.ack();
        turn.done("a new answer");
      });

      expect(result.current.messages.map((m) => m.id)).toEqual([...earlierIds, turn.userId, turn.assistantId]);
    });
  });

  // ─── 6. No duplicates while streaming or reconciling ────────────────────────
  describe("streaming", () => {
    it("partial frames, the completion and a repeated completion leave one reply under one id", async () => {
      const { result } = await startChat();
      const turn = await sendOverSocket(result, "hello");

      await act(async () => {
        turn.ack();
        turn.partial("hel");
        turn.partial("hello th");
        turn.done("hello there");
      });
      const completed = result.current.messages[1];

      // The same completion again, one that names the reply by id and altered
      // content, and a stale partial: none may add a message or touch the reply.
      await act(async () => {
        turn.done("hello there");
        socket.emit("stream", {
          type: "stream",
          conversationId: result.current.activeConversation,
          content: "overwritten",
          done: true,
          messageId: turn.assistantId,
        });
        socket.emit("stream", {
          type: "stream",
          conversationId: result.current.activeConversation,
          content: "late partial",
          done: false,
          messageId: turn.assistantId,
        });
      });

      expect(result.current.messages).toHaveLength(2);
      expect(result.current.messages.filter((m) => m.role === "assistant")).toHaveLength(1);
      expect(result.current.messages[1].id).toBe(turn.assistantId);
      expect(result.current.messages[1].content).toBe("hello there");
      expect(result.current.messages[1]).toEqual(completed);
      expect(result.current.streaming).toBe(false);
    });

    it("a repeated completion is still ignored after the conversation is reloaded from the server", async () => {
      const { result } = await startChat();
      const conversationId = result.current.activeConversation;
      const turn = await sendOverSocket(result, "hello");
      await act(async () => {
        turn.ack();
        turn.done("hello there");
      });

      // The reload brings back the server's copy. Its messages now match the id
      // the frames name, and carry no completed flag of their own.
      await reloadFromServer(result, conversationId);
      expect(result.current.messages.map((m) => m.id)).toEqual([turn.userId, turn.assistantId]);
      expect(result.current.messages.every((m) => m.completed === undefined)).toBe(true);

      await act(async () => {
        socket.emit("stream", {
          type: "stream",
          conversationId,
          content: "overwritten",
          done: true,
          messageId: turn.assistantId,
        });
      });

      expect(result.current.messages).toHaveLength(2);
      expect(result.current.messages[1].content).toBe("hello there");
    });

    it("an audio reply takes the server's id with its last frame, once", async () => {
      const { result } = await startChat();
      await act(async () => {
        await result.current.sendMessage("make some audio", "audio");
      });
      const turn = server.beginTurn(socket.sent[0], socket);
      const temporary = result.current.messages[1].id;

      await act(async () => {
        turn.ack();
        turn.doneAudio("Here is your audio");
      });

      expect(result.current.messages).toHaveLength(2);
      expect(result.current.messages[1].id).toBe(turn.assistantId);
      expect(result.current.messages[1].clientId).toBe(temporary);
      expect(result.current.messages[1].audioComplete).toBe(true);
      expect(result.current.messages[1].audioChunks).toEqual(["AAAA"]);

      await act(async () => {
        turn.doneAudio("Here is your audio");
      });
      expect(result.current.messages).toHaveLength(2);
      expect(result.current.messages[1].audioChunks).toEqual(["AAAA"]);
    });

    it("an audio completion repeated after a reload is ignored too", async () => {
      const { result } = await startChat();
      const conversationId = result.current.activeConversation;
      await act(async () => {
        await result.current.sendMessage("make some audio", "audio");
      });
      const turn = server.beginTurn(socket.sent[0], socket);
      await act(async () => {
        turn.ack();
        turn.doneAudio("Here is your audio");
      });
      await reloadFromServer(result, conversationId);
      expect(result.current.messages[1].audioComplete).toBeUndefined(); // the server's copy, not the local one
      const reloaded = result.current.messages[1];

      await act(async () => {
        turn.doneAudio("Here is your audio");
      });

      expect(result.current.messages).toHaveLength(2);
      expect(result.current.messages[1]).toEqual(reloaded);
    });

    it("a repeated ack changes nothing", async () => {
      const { result } = await startChat();
      const turn = await sendOverSocket(result, "hello");

      await act(async () => {
        turn.ack();
      });
      const afterFirst = result.current.messages;
      await act(async () => {
        turn.ack();
      });

      expect(result.current.messages).toHaveLength(2);
      expect(result.current.messages.map((m) => m.id)).toEqual(afterFirst.map((m) => m.id));
    });

    it("an ack that arrives with nothing in flight is ignored", async () => {
      const { result } = await startChat();
      await act(async () => {
        socket.emit("ack", { type: "ack", messageId: crypto.randomUUID(), conversationId: result.current.activeConversation });
        socket.emit("ack", { type: "ack", messageId: crypto.randomUUID() });
      });

      expect(result.current.messages).toEqual([]);
    });
  });

  // ─── 7. Existing behaviour is preserved ─────────────────────────────────────
  describe("existing behaviour", () => {
    it("an error after the ack keeps the user's server id and shows the error in place", async () => {
      const { result } = await startChat();
      const turn = await sendOverSocket(result, "faulty");
      const temporaryReply = result.current.messages[1].id;

      await act(async () => {
        turn.ack();
        turn.fail("Model execution failed");
      });

      expect(result.current.messages).toHaveLength(2);
      expect(result.current.messages[0].id).toBe(turn.userId);
      expect(result.current.messages[1].type).toBe("error");
      expect(result.current.messages[1].content).toBe("Model execution failed");
      // Nothing was stored for the reply, so it keeps the id it was created with.
      expect(result.current.messages[1].id).toBe(temporaryReply);
      expect(result.current.streaming).toBe(false);
    });

    it("retrying after an error reconciles the new turn and leaves the old pair alone", async () => {
      const { result } = await startChat();
      const failed = await sendOverSocket(result, "faulty");
      await act(async () => {
        failed.ack();
        failed.fail("Model execution failed");
      });
      const errorBubble = result.current.messages[1];

      const retry = await sendOverSocket(result, "faulty");
      expect(result.current.messages).toHaveLength(4);
      await act(async () => {
        retry.ack();
        retry.done("worked this time");
      });

      expect(result.current.messages.map((m) => m.id)).toEqual([
        failed.userId,
        errorBubble.id,
        retry.userId,
        retry.assistantId,
      ]);
      expect(result.current.messages.map((m) => m.role)).toEqual(["user", "assistant", "user", "assistant"]);
      expect(result.current.messages[1].content).toBe("Model execution failed");
      expect(result.current.messages[3].content).toBe("worked this time");
    });

    it("a REST error leaves the temporary ids and the error message as before", async () => {
      const { result } = await startChat();
      socket.ws.readyState = 3;
      api.sendMessage.mockRejectedValueOnce(new Error("Server exploded"));

      await act(async () => {
        await result.current.sendMessage("hello");
      });

      expect(result.current.messages).toHaveLength(2);
      expect(result.current.messages[1].type).toBe("error");
      expect(result.current.messages[1].content).toBe("Server exploded");
    });

    it("turns streaming in two conversations at once are reconciled in their own conversations", async () => {
      const { result } = await startChat();
      const firstId = result.current.activeConversation;
      const first = await sendOverSocket(result, "in the first");

      await act(async () => {
        await result.current.createConversation();
      });
      const secondId = result.current.activeConversation;
      expect(secondId).not.toBe(firstId);
      const second = await sendOverSocket(result, "in the second");

      // The first conversation is not the one on screen.
      await act(async () => {
        first.ack();
        first.done("first reply");
      });
      const firstOnScreen = result.current.conversations.find((c) => c.id === firstId);
      expect(firstOnScreen.messages.map((m) => m.id)).toEqual([first.userId, first.assistantId]);
      expect(result.current.messages).toHaveLength(2);
      expect(result.current.messages.map((m) => m.id)).not.toContain(first.userId);

      await act(async () => {
        second.ack();
        second.done("second reply");
      });
      expect(result.current.messages.map((m) => m.id)).toEqual([second.userId, second.assistantId]);
      expect(result.current.conversations.find((c) => c.id === firstId).messages.map((m) => m.id)).toEqual([
        first.userId,
        first.assistantId,
      ]);
    });

    it("persists the server's ids, not the temporary ones", async () => {
      const { result } = await startChat();
      const turn = await sendOverSocket(result, "hello");
      await act(async () => {
        turn.ack();
        turn.done("hello there");
      });

      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
      const persisted = saved.conversations.find((c) => c.id === result.current.activeConversation);
      expect(persisted.messages.map((m) => m.id)).toEqual([turn.userId, turn.assistantId]);
    });
  });
});

// ─── 8. The rendered list ─────────────────────────────────────────────────────
describe("Issue #394: reconciling an id does not remount the message", () => {
  function chatProps(overrides = {}) {
    return {
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
    };
  }

  it("a bubble keeps its DOM node when its id is replaced by the server's", () => {
    const stamp = "2026-10-01T10:00:00.000Z";
    const optimistic = [
      { id: "client-user", role: "user", type: "text", content: "hello", timestamp: stamp },
      { id: "client-reply", role: "assistant", type: "text", content: "hi there", completed: true, timestamp: stamp },
    ];
    const { rerender } = render(<Chat {...chatProps({ messages: optimistic })} />);
    const before = screen.getByText("hi there").closest("article");

    // What the hook does when the server reports the persisted ids.
    const reconciled = [
      { ...optimistic[0], id: "server-user", clientId: "client-user" },
      { ...optimistic[1], id: "server-reply", clientId: "client-reply" },
    ];
    rerender(<Chat {...chatProps({ messages: reconciled })} />);

    // A remount would replay the entry animation and, for audio, tear down the player.
    expect(screen.getByText("hi there").closest("article")).toBe(before);
  });
});
