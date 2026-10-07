import { describe, it, expect } from "vitest";
import { dedupeMessages, mergeMessages, reconcileConversation } from "../utils/dedupe";

describe("dedupeMessages utility", () => {
  it("returns empty array for non-array inputs", () => {
    expect(dedupeMessages(null)).toEqual([]);
    expect(dedupeMessages(undefined)).toEqual([]);
    expect(dedupeMessages("not an array")).toEqual([]);
    expect(dedupeMessages(123)).toEqual([]);
    expect(dedupeMessages({})).toEqual([]);
  });

  it("returns empty array for empty input array", () => {
    expect(dedupeMessages([])).toEqual([]);
  });

  it("preserves arrays with all unique message IDs", () => {
    const input = [
      { id: "msg-1", role: "user", content: "Hello" },
      { id: "msg-2", role: "assistant", content: "Hi there" },
    ];
    expect(dedupeMessages(input)).toEqual(input);
  });

  it("removes duplicate message IDs keeping the first occurrence", () => {
    const input = [
      { id: "dup-1", role: "user", content: "Original" },
      { id: "unique-1", role: "assistant", content: "Response" },
      { id: "dup-1", role: "user", content: "Duplicate content" },
      { id: "dup-1", role: "user", content: "Another duplicate" },
    ];
    const result = dedupeMessages(input);
    expect(result).toHaveLength(2);
    expect(result[0]).toEqual({ id: "dup-1", role: "user", content: "Original" });
    expect(result[1]).toEqual({ id: "unique-1", role: "assistant", content: "Response" });
  });

  it("preserves messages without an ID", () => {
    const input = [
      { role: "user", content: "No id 1" },
      { id: "has-id", role: "assistant", content: "With id" },
      { role: "user", content: "No id 2" },
    ];
    const result = dedupeMessages(input);
    expect(result).toHaveLength(3);
    expect(result[0].content).toBe("No id 1");
    expect(result[1].id).toBe("has-id");
    expect(result[2].content).toBe("No id 2");
  });

  it("handles empty string and falsy IDs by preserving them", () => {
    const input = [
      { id: "", role: "user", content: "Empty string id 1" },
      { id: "", role: "user", content: "Empty string id 2" },
      { id: null, role: "user", content: "Null id" },
      { id: undefined, role: "user", content: "Undefined id" },
    ];
    const result = dedupeMessages(input);
    expect(result).toHaveLength(4);
  });

  it("does not mutate the original message objects", () => {
    const msg1 = { id: "m1", role: "user", content: "Text", attachments: ["doc.pdf"] };
    const msg2 = { id: "m1", role: "user", content: "Text clone" };
    const input = [msg1, msg2];
    const result = dedupeMessages(input);
    expect(result).toHaveLength(1);
    expect(result[0]).toBe(msg1);
    expect(result[0].attachments).toEqual(["doc.pdf"]);
  });
});

describe("mergeMessages utility", () => {
  it("returns remote messages when local messages array is empty", () => {
    const remote = [{ id: "m1", role: "user", content: "Remote msg" }];
    expect(mergeMessages([], remote)).toEqual(remote);
    expect(mergeMessages(null, remote)).toEqual(remote);
  });

  it("returns local messages when remote messages array is empty", () => {
    const local = [{ id: "m1", role: "user", content: "Local msg" }];
    expect(mergeMessages(local, [])).toEqual(local);
    expect(mergeMessages(local, null)).toEqual(local);
  });

  it("preserves in-flight local messages absent from remote snapshot", () => {
    const local = [
      { id: "m1", role: "user", content: "Hello" },
      { id: "m2", role: "assistant", content: "Hi", completed: true },
      { id: "m3", role: "user", content: "In-flight question" },
    ];
    const remote = [
      { id: "m1", role: "user", content: "Hello" },
      { id: "m2", role: "assistant", content: "Hi", completed: true },
    ];
    const merged = mergeMessages(local, remote);
    expect(merged).toHaveLength(3);
    expect(merged[2].id).toBe("m3");
    expect(merged[2].content).toBe("In-flight question");
  });

  it("updates placeholder assistant message with completed remote content", () => {
    const local = [
      { id: "m1", role: "user", content: "Generate code" },
      { id: "m2", role: "assistant", content: "", completed: false },
    ];
    const remote = [
      { id: "m1", role: "user", content: "Generate code" },
      { id: "m2", role: "assistant", content: "def hello(): pass", completed: true, metadata: { latency: 120 } },
    ];
    const merged = mergeMessages(local, remote);
    expect(merged).toHaveLength(2);
    expect(merged[1].id).toBe("m2");
    expect(merged[1].content).toBe("def hello(): pass");
    expect(merged[1].completed).toBe(true);
    expect(merged[1].metadata.latency).toBe(120);
  });

  it("appends new remote messages not present in local state", () => {
    const local = [{ id: "m1", role: "user", content: "Old message" }];
    const remote = [
      { id: "m1", role: "user", content: "Old message" },
      { id: "m2", role: "assistant", content: "Server reply" },
    ];
    const merged = mergeMessages(local, remote);
    expect(merged).toHaveLength(2);
    expect(merged[1].id).toBe("m2");
  });
});

describe("reconcileConversation utility", () => {
  it("handles null / undefined / single-sided inputs", () => {
    expect(reconcileConversation(null, null)).toBeNull();
    const conv = { id: "c1", title: "Chat 1", messages: [{ id: "m1" }] };
    expect(reconcileConversation(conv, null)).toBe(conv);
    expect(reconcileConversation(null, conv)).toEqual({
      id: "c1",
      title: "Chat 1",
      messages: [{ id: "m1" }],
    });
  });

  it("preserves newer local title when localTitleUpdatedAt > incoming.titleUpdatedAt", () => {
    const existing = { id: "c1", title: "Renamed Title", messages: [] };
    const incoming = { id: "c1", title: "Stale Backend Title", messages: [] };
    const reconciled = reconcileConversation(existing, incoming, 100);
    expect(reconciled.title).toBe("Renamed Title");
    expect(reconciled.titleUpdatedAt).toBe(100);
  });

  it("preserves newer local title when existingConv.titleUpdatedAt > incoming", () => {
    const existing = { id: "c1", title: "Local Custom Title", titleUpdatedAt: 200, messages: [] };
    const incoming = { id: "c1", title: "Stale Remote", messages: [] };
    const reconciled = reconcileConversation(existing, incoming, 0);
    expect(reconciled.title).toBe("Local Custom Title");
    expect(reconciled.titleUpdatedAt).toBe(200);
  });

  it("adopts incoming title when incoming title is newer", () => {
    const existing = { id: "c1", title: "Older Title", titleUpdatedAt: 50, messages: [] };
    const incoming = { id: "c1", title: "Newer Remote Title", titleUpdatedAt: 150, messages: [] };
    const reconciled = reconcileConversation(existing, incoming, 0);
    expect(reconciled.title).toBe("Newer Remote Title");
    expect(reconciled.titleUpdatedAt).toBe(150);
  });

  it("adopts incoming title when local conversation has never been updated", () => {
    const existing = { id: "c1", title: "New Chat", messages: [] };
    const incoming = { id: "c1", title: "First Message Generated Title", messages: [] };
    const reconciled = reconcileConversation(existing, incoming, 0);
    expect(reconciled.title).toBe("First Message Generated Title");
  });

  it("deduplicates messages and preserves combined properties", () => {
    const existing = {
      id: "c1",
      title: "Chat",
      messages: [{ id: "m1", role: "user", content: "Hi" }],
      parentConversationId: "p1",
    };
    const incoming = {
      id: "c1",
      title: "Chat",
      messages: [
        { id: "m1", role: "user", content: "Hi" },
        { id: "m2", role: "assistant", content: "Hello" },
      ],
      createdAt: "2026-10-01T00:00:00.000Z",
    };
    const reconciled = reconcileConversation(existing, incoming, 0);
    expect(reconciled.messages).toHaveLength(2);
    expect(reconciled.parentConversationId).toBe("p1");
    expect(reconciled.createdAt).toBe("2026-10-01T00:00:00.000Z");
  });
});
