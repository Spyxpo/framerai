import { describe, it, expect } from "vitest";
import { dedupeMessages } from "../utils/dedupe";

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
      { id: "msg-2", role: "assistant", content: "Hi" },
    ];
    const result = dedupeMessages(input);
    expect(result).toHaveLength(2);
    expect(result).toEqual(input);
  });

  it("removes duplicate message IDs keeping the first occurrence", () => {
    const input = [
      { id: "dup-1", role: "user", content: "Original message" },
      { id: "unique-1", role: "assistant", content: "Middle message" },
      { id: "dup-1", role: "user", content: "Duplicate message to drop" },
    ];
    const result = dedupeMessages(input);
    expect(result).toHaveLength(2);
    expect(result[0]).toEqual({ id: "dup-1", role: "user", content: "Original message" });
    expect(result[1]).toEqual({ id: "unique-1", role: "assistant", content: "Middle message" });
  });

  it("preserves messages without an ID", () => {
    const input = [
      { role: "user", content: "No id 1" },
      { role: "assistant", content: "No id 2" },
      { id: "has-id", role: "user", content: "Has id" },
      { role: "user", content: "No id 3" },
    ];
    const result = dedupeMessages(input);
    expect(result).toHaveLength(4);
    expect(result[0].content).toBe("No id 1");
    expect(result[1].content).toBe("No id 2");
    expect(result[2].content).toBe("Has id");
    expect(result[3].content).toBe("No id 3");
  });

  it("handles empty string and falsy IDs by preserving them", () => {
    const input = [
      { id: "", role: "user", content: "Empty string id 1" },
      { id: "", role: "assistant", content: "Empty string id 2" },
      { id: null, role: "user", content: "Null id" },
    ];
    const result = dedupeMessages(input);
    expect(result).toHaveLength(3);
  });

  it("does not mutate the original message objects", () => {
    const msg1 = { id: "m1", role: "user", content: "Hello", attachments: ["doc.pdf"] };
    const msg2 = { id: "m1", role: "user", content: "Hello duplicate" };
    const input = [msg1, msg2];
    const result = dedupeMessages(input);
    expect(result).toHaveLength(1);
    expect(result[0]).toBe(msg1);
    expect(result[0].attachments).toEqual(["doc.pdf"]);
  });
});
