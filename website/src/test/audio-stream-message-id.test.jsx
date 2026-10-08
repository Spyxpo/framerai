/**
 * Audio chunks of an abandoned reply must not land in its replacement.
 *
 * The server streams an audio reply in chunks, and only the last one used to
 * name the reply it belonged to. A chunk without that id goes to whichever
 * reply the conversation is currently waiting for. Edit and regenerate abandon
 * the reply in flight and start a new one, while the server keeps sending the
 * old reply's chunks, so those chunks were written into the replacement: an
 * edited question answered in text showed the abandoned reply's audio player,
 * and a regenerated audio reply played the old reply's audio mixed into its own.
 *
 * Every chunk now names its reply (backend/src/services/websocket.js,
 * streamAudio). Edit and regenerate already mark the abandoned reply as done,
 * so its chunks are dropped like any other frame for a finished reply. The
 * frames below are built the way streamAudio builds them.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { useChat } from "../hooks/useChat";
import { api } from "../services/api";

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

const CONVERSATION_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
const AUDIO_CONTENT = "Here is your audio";

const chunkData = (label, index) => `${label}${index}`;

/** Chunk `index` of `total` for one audio reply, as streamAudio() sends it. */
function audioFrame(messageId, label, index, total) {
  const isLast = index === total - 1;
  return {
    type: "stream",
    conversationId: CONVERSATION_ID,
    content: index === 0 ? AUDIO_CONTENT : "",
    done: isLast,
    messageId,
    responseType: "audio",
    metadata: {
      chunk: index,
      totalChunks: total,
      chunkData: chunkData(label, index),
      sampleRate: 24000,
      channels: 1,
      bitsPerSample: 16,
      ...(isLast ? { url: `/uploads/generated/${label}.wav`, model: "test-model", durationSec: 1.5 } : {}),
    },
  };
}

/** The completion frame of a text reply, as the WebSocket service sends it. */
function textDoneFrame(messageId, content) {
  return {
    type: "stream",
    conversationId: CONVERSATION_ID,
    content,
    done: true,
    messageId,
    responseType: "text",
    metadata: { model: "test-model" },
  };
}

async function mountChat() {
  const { result } = renderHook(() => useChat({}));
  await waitFor(() => expect(result.current.loadingConversations).toBe(false));
  await act(async () => {
    await result.current.createConversation();
  });
  expect(result.current.activeConversation).toBe(CONVERSATION_ID);
  return { result, socket: doubles.sockets[doubles.sockets.length - 1] };
}

/** The server acknowledges a turn, naming the user message and the reply. */
function ack(socket, userId, replyId) {
  act(() => {
    socket.emit("ack", {
      type: "ack",
      messageId: userId,
      assistantMessageId: replyId,
      conversationId: CONVERSATION_ID,
    });
    socket.emit("typing", { type: "typing", conversationId: CONVERSATION_ID });
  });
}

function stream(socket, frame) {
  act(() => {
    socket.emit("stream", frame);
  });
}

async function sendAndAck(result, socket, content, type) {
  await act(async () => {
    await result.current.sendMessage(content, type, []);
  });
  const userId = crypto.randomUUID();
  const replyId = crypto.randomUUID();
  ack(socket, userId, replyId);
  return { userId, replyId };
}

describe("audio stream chunks name the reply they belong to", () => {
  beforeEach(() => {
    localStorage.clear();
    doubles.sockets.length = 0;
    api.listConversations.mockReset().mockResolvedValue([]);
    api.createConversation.mockReset().mockResolvedValue({ id: CONVERSATION_ID, title: "New Chat", messages: [] });
    api.getConversation.mockReset().mockResolvedValue({ id: CONVERSATION_ID, title: "New Chat", messages: [] });
    api.deleteConversation.mockReset().mockResolvedValue({ success: true });
    api.sendMessage.mockReset();
    api.branchConversation.mockReset();
  });

  it("an abandoned audio reply's later chunks cannot turn the edited reply into audio", async () => {
    const { result, socket } = await mountChat();

    // Text mode, but "say ... out loud" is answered with audio.
    const turnA = await sendAndAck(result, socket, "say this out loud", "text");
    stream(socket, audioFrame(turnA.replyId, "A", 0, 3));
    expect(result.current.messages[1]).toMatchObject({ id: turnA.replyId, type: "audio" });
    expect(result.current.messages[1].audioChunks).toEqual([chunkData("A", 0)]);

    // The user edits the question before the audio has finished streaming.
    await act(async () => {
      await result.current.editMessage(turnA.userId, "just summarise it in text");
    });
    expect(socket.sent[socket.sent.length - 1]).toMatchObject({ type: "chat", editMessageId: turnA.userId });
    const replyB = crypto.randomUUID();
    ack(socket, turnA.userId, replyB);

    // The server is still streaming the abandoned reply.
    stream(socket, audioFrame(turnA.replyId, "A", 1, 3));
    const midStream = result.current.messages[1];
    expect(midStream).toMatchObject({ id: replyB, type: "text", content: "" });
    expect(midStream.audioChunks).toBeUndefined();
    expect(midStream.audioMetadata).toBeUndefined();

    stream(socket, audioFrame(turnA.replyId, "A", 2, 3));

    // The replacement's own reply arrives.
    stream(socket, textDoneFrame(replyB, "Here is a short summary."));

    const final = result.current.messages;
    expect(final).toHaveLength(2);
    expect(final[0]).toMatchObject({ id: turnA.userId, role: "user", content: "just summarise it in text" });
    expect(final[1]).toMatchObject({
      id: replyB,
      role: "assistant",
      type: "text",
      content: "Here is a short summary.",
      completed: true,
    });
    expect(final[1].audioChunks).toBeUndefined();
    expect(final[1].audioMetadata).toBeUndefined();
    expect(final.some((m) => m.id === turnA.replyId)).toBe(false);
  });

  it("an abandoned audio reply's later chunks cannot mix into the regenerated audio reply", async () => {
    const { result, socket } = await mountChat();

    const turnA = await sendAndAck(result, socket, "say this out loud", "audio");
    stream(socket, audioFrame(turnA.replyId, "A", 0, 3));

    await act(async () => {
      await result.current.regenerateResponse(turnA.replyId);
    });
    expect(socket.sent[socket.sent.length - 1]).toMatchObject({
      type: "chat",
      regenerateMessageId: turnA.replyId,
      messageType: "audio",
    });
    const replyB = crypto.randomUUID();
    ack(socket, turnA.userId, replyB);

    // Both replies stream at once, interleaved.
    stream(socket, audioFrame(replyB, "B", 0, 3));
    stream(socket, audioFrame(turnA.replyId, "A", 1, 3));
    stream(socket, audioFrame(replyB, "B", 1, 3));
    stream(socket, audioFrame(turnA.replyId, "A", 2, 3));
    stream(socket, audioFrame(replyB, "B", 2, 3));

    const final = result.current.messages;
    expect(final).toHaveLength(2);
    expect(final[1]).toMatchObject({
      id: replyB,
      role: "assistant",
      type: "audio",
      content: AUDIO_CONTENT,
      completed: true,
      audioComplete: true,
    });
    expect(final[1].audioChunks).toEqual([chunkData("B", 0), chunkData("B", 1), chunkData("B", 2)]);
    expect(final[1].metadata.url).toBe("/uploads/generated/B.wav");
    expect(final.some((m) => m.id === turnA.replyId)).toBe(false);
  });

  it("an uninterrupted audio reply still collects every chunk and completes as audio", async () => {
    const { result, socket } = await mountChat();

    const turn = await sendAndAck(result, socket, "say this out loud", "audio");

    stream(socket, audioFrame(turn.replyId, "A", 0, 3));
    expect(result.current.messages[1].audioChunks).toEqual([chunkData("A", 0)]);
    expect(result.current.messages[1].completed).toBeFalsy();

    stream(socket, audioFrame(turn.replyId, "A", 1, 3));
    expect(result.current.messages[1].audioChunks).toEqual([chunkData("A", 0), chunkData("A", 1)]);
    expect(result.current.messages[1].completed).toBeFalsy();

    stream(socket, audioFrame(turn.replyId, "A", 2, 3));

    const reply = result.current.messages[1];
    expect(reply).toMatchObject({
      id: turn.replyId,
      role: "assistant",
      type: "audio",
      content: AUDIO_CONTENT,
      completed: true,
      audioComplete: true,
    });
    expect(reply.audioChunks).toEqual([chunkData("A", 0), chunkData("A", 1), chunkData("A", 2)]);
    expect(reply.audioMetadata).toEqual({ sampleRate: 24000, channels: 1, bitsPerSample: 16, totalChunks: 3 });
    expect(reply.metadata).toMatchObject({ url: "/uploads/generated/A.wav", model: "test-model", durationSec: 1.5 });
    expect(result.current.streaming).toBe(false);
  });

  it("a frame naming a reply that already finished is still dropped", async () => {
    const { result, socket } = await mountChat();

    const turnA = await sendAndAck(result, socket, "say this out loud", "audio");
    stream(socket, audioFrame(turnA.replyId, "A", 0, 2));
    stream(socket, audioFrame(turnA.replyId, "A", 1, 2));
    const finishedA = result.current.messages[1];
    expect(finishedA).toMatchObject({ id: turnA.replyId, completed: true });

    // A second turn is in flight in the same conversation.
    const turnB = await sendAndAck(result, socket, "and now in words", "text");

    // Repeats of A's frames, each naming A.
    stream(socket, audioFrame(turnA.replyId, "A", 0, 2));
    stream(socket, audioFrame(turnA.replyId, "A", 1, 2));

    const messages = result.current.messages;
    expect(messages).toHaveLength(4);
    expect(messages[1]).toEqual(finishedA);
    expect(messages[3]).toMatchObject({ id: turnB.replyId, type: "text", content: "" });
    expect(messages[3].audioChunks).toBeUndefined();
    expect(messages[3].completed).toBeFalsy();
  });
});
