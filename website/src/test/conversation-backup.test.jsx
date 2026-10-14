import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  BACKUP_VERSION,
  MAX_IMPORT_FILE_SIZE,
  createBackup,
  exportConversationToMarkdown,
  exportConversationToPlainText,
  parseMarkdownConversation,
  validateBackup,
  restoreBackup,
} from "../utils/backup";
import {
  STORAGE_KEY,
  loadConversationsFromStorage,
} from "../utils/storage";
import { useChat } from "../hooks/useChat";
import Sidebar from "../components/Sidebar/Sidebar";
import Chat from "../components/Chat/Chat";
import BackupModal from "../components/Backup/BackupModal";

// Mock external services for hook testing
vi.mock("../services/api", () => ({
  api: {
    listConversations: vi.fn(() => Promise.resolve([])),
    createConversation: vi.fn(() =>
      Promise.resolve({ id: "api-conv-1", title: "API Chat", messages: [] })
    ),
    getConversation: vi.fn((id) => Promise.resolve({ id, messages: [] })),
    deleteConversation: vi.fn(() => Promise.resolve()),
    sendMessage: vi.fn(() => Promise.resolve({ content: "API reply", type: "text" })),
    health: vi.fn(() => Promise.resolve({ status: "ok", model: "framerai-test" })),
  },
}));

vi.mock("../services/websocket", () => ({
  WebSocketClient: class {
    connect() {
      return Promise.resolve();
    }
    on() {
      return () => {};
    }
    send() {}
    disconnect() {}
  },
}));

describe("Conversation Backup System (Issue #411)", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  afterEach(() => {
    localStorage.clear();
  });

  // =========================================================================
  // 1. EXPORT TESTS
  // =========================================================================
  describe("Export Functionality", () => {
    const sampleConv = {
      id: "conv-101",
      title: "Architecture Discussion",
      createdAt: "2026-08-01T12:00:00.000Z",
      updatedAt: "2026-08-01T12:05:00.000Z",
      messages: [
        {
          id: "msg-1",
          role: "user",
          content: "Let's plan the new database schema.",
          type: "text",
          timestamp: "2026-08-01T12:00:01.000Z",
        },
        {
          id: "msg-2",
          role: "assistant",
          content: "Here is the proposed schema with tables...",
          type: "text",
          timestamp: "2026-08-01T12:00:05.000Z",
          metadata: {
            model: "framerai-v1",
            trace: {
              memories: [{ text: "schema-design", score: 0.95 }],
            },
          },
        },
        {
          id: "msg-3",
          role: "user",
          content: "Looks great, please add indexes.",
          type: "text",
          timestamp: "2026-08-01T12:01:00.000Z",
        },
      ],
    };

    const branchConv = {
      id: "conv-102",
      title: "Architecture Discussion (Branch)",
      createdAt: "2026-08-01T12:10:00.000Z",
      updatedAt: "2026-08-01T12:15:00.000Z",
      parentConversationId: "conv-101",
      branchedFromMessageId: "msg-2",
      messages: [
        {
          id: "msg-1",
          role: "user",
          content: "Let's plan the new database schema.",
          type: "text",
          timestamp: "2026-08-01T12:00:01.000Z",
        },
        {
          id: "msg-2",
          role: "assistant",
          content: "Here is the proposed schema with tables...",
          type: "text",
          timestamp: "2026-08-01T12:00:05.000Z",
          metadata: {
            model: "framerai-v1",
          },
        },
        {
          id: "msg-4",
          role: "user",
          content: "Alternative approach: what if we use document store?",
          type: "text",
          timestamp: "2026-08-01T12:10:05.000Z",
        },
      ],
    };

    it("exports a single conversation into versioned JSON backup format", () => {
      const backup = createBackup(sampleConv);

      expect(backup.version).toBe(BACKUP_VERSION);
      expect(backup.framerai_backup).toBe(true);
      expect(typeof backup.exportedAt).toBe("string");
      expect(backup.conversationCount).toBe(1);
      expect(backup.conversations).toHaveLength(1);

      const exportedConv = backup.conversations[0];
      expect(exportedConv.id).toBe("conv-101");
      expect(exportedConv.title).toBe("Architecture Discussion");
      expect(exportedConv.createdAt).toBe("2026-08-01T12:00:00.000Z");
      expect(exportedConv.messages).toHaveLength(3);
    });

    it("exports multiple and all conversations into unified backup", () => {
      const backup = createBackup([sampleConv, branchConv]);

      expect(backup.version).toBe(BACKUP_VERSION);
      expect(backup.conversationCount).toBe(2);
      expect(backup.conversations).toHaveLength(2);
      expect(backup.conversations[0].id).toBe("conv-101");
      expect(backup.conversations[1].id).toBe("conv-102");
    });

    it("preserves strict message ordering across export", () => {
      const backup = createBackup(sampleConv);
      const msgs = backup.conversations[0].messages;

      expect(msgs[0].id).toBe("msg-1");
      expect(msgs[1].id).toBe("msg-2");
      expect(msgs[2].id).toBe("msg-3");
      expect(msgs[0].content).toBe("Let's plan the new database schema.");
      expect(msgs[2].content).toBe("Looks great, please add indexes.");
    });

    it("preserves metadata including cognition trace and model info", () => {
      const backup = createBackup(sampleConv);
      const assistantMsg = backup.conversations[0].messages[1];

      expect(assistantMsg.metadata).toBeDefined();
      expect(assistantMsg.metadata.model).toBe("framerai-v1");
      expect(assistantMsg.metadata.trace).toBeDefined();
      expect(assistantMsg.metadata.trace.memories[0].text).toBe("schema-design");
      expect(assistantMsg.metadata.trace.memories[0].score).toBe(0.95);
    });

    it("preserves branch relationships: parentConversationId and branchedFromMessageId", () => {
      const backup = createBackup([sampleConv, branchConv]);
      const exportedBranch = backup.conversations[1];

      expect(exportedBranch.parentConversationId).toBe("conv-101");
      expect(exportedBranch.branchedFromMessageId).toBe("msg-2");
    });

    it("exports conversation to readable Markdown format", () => {
      const md = exportConversationToMarkdown(sampleConv);

      expect(md).toContain("# Architecture Discussion");
      expect(md).toContain("### User");
      expect(md).toContain("Let's plan the new database schema.");
      expect(md).toContain("### Assistant");
      expect(md).toContain("Here is the proposed schema with tables...");
    });

    it("exports conversation to plain text format", () => {
      const txt = exportConversationToPlainText(sampleConv);

      expect(txt).toContain("TITLE: Architecture Discussion");
      expect(txt).toContain("USER");
      expect(txt).toContain("ASSISTANT");
    });
  });

  // =========================================================================
  // 2. VALIDATION & SAFETY (REJECTION) TESTS
  // =========================================================================
  describe("Backup Validation and Rejection", () => {
    it("rejects malformed JSON strings safely", () => {
      const result = validateBackup("{ not valid json at all ... }");
      expect(result.valid).toBe(false);
      expect(result.error).toMatch(/malformed json/i);
    });

    it("rejects null, undefined, or empty string backup data", () => {
      expect(validateBackup(null).valid).toBe(false);
      expect(validateBackup("").valid).toBe(false);
      expect(validateBackup("   ").valid).toBe(false);
    });

    it("rejects non-object root or array root", () => {
      expect(validateBackup("123").valid).toBe(false);
      expect(validateBackup("true").valid).toBe(false);
      expect(validateBackup("[]").valid).toBe(false);
    });

    it("rejects missing version field", () => {
      const payload = {
        conversations: [{ id: "c1", messages: [] }],
      };
      const result = validateBackup(payload);
      expect(result.valid).toBe(false);
      expect(result.error).toMatch(/version/i);
    });

    it("rejects unsupported backup versions", () => {
      const payload = {
        version: 999,
        conversations: [{ id: "c1", messages: [] }],
      };
      const result = validateBackup(payload);
      expect(result.valid).toBe(false);
      expect(result.error).toMatch(/unsupported backup version 999/i);
    });

    it("rejects missing or non-array conversations property", () => {
      expect(validateBackup({ version: BACKUP_VERSION }).valid).toBe(false);
      expect(validateBackup({ version: BACKUP_VERSION, conversations: "not-an-array" }).valid).toBe(false);
    });

    it("rejects empty conversations array", () => {
      const result = validateBackup({ version: BACKUP_VERSION, conversations: [] });
      expect(result.valid).toBe(false);
      expect(result.error).toMatch(/contains no conversations/i);
    });

    it("rejects conversation without an id", () => {
      const payload = {
        version: BACKUP_VERSION,
        conversations: [{ title: "No ID", messages: [] }],
      };
      const result = validateBackup(payload);
      expect(result.valid).toBe(false);
      expect(result.error).toMatch(/missing or invalid 'id'/i);
    });

    it("rejects conversation with non-array messages", () => {
      const payload = {
        version: BACKUP_VERSION,
        conversations: [{ id: "c1", messages: "invalid-messages" }],
      };
      const result = validateBackup(payload);
      expect(result.valid).toBe(false);
      expect(result.error).toMatch(/'messages' must be an array/i);
    });

    it("rejects messages missing id or role", () => {
      const payloadMissingId = {
        version: BACKUP_VERSION,
        conversations: [
          {
            id: "c1",
            messages: [{ role: "user", content: "hello" }],
          },
        ],
      };
      expect(validateBackup(payloadMissingId).valid).toBe(false);

      const payloadMissingRole = {
        version: BACKUP_VERSION,
        conversations: [
          {
            id: "c1",
            messages: [{ id: "m1", content: "hello" }],
          },
        ],
      };
      expect(validateBackup(payloadMissingRole).valid).toBe(false);
    });

    it("safety: invalid backup rejected before state modification (atomic)", () => {
      const existing = [
        {
          id: "existing-1",
          title: "Untouched Chat",
          messages: [{ id: "m1", role: "user", content: "Stay untouched" }],
        },
      ];

      const malformedPayload = "{ bad json }";
      const result = restoreBackup(malformedPayload, existing);

      expect(result.success).toBe(false);
      expect(result.conversations).toBeUndefined();
      // Ensure existing is completely unaffected
      expect(existing).toHaveLength(1);
      expect(existing[0].title).toBe("Untouched Chat");
    });
  });

  // =========================================================================
  // 3. RESTORATION, COLLISION, & ROUND TRIP TESTS
  // =========================================================================
  describe("Restoration & Collision Handling", () => {
    it("successfully restores a valid backup without collisions", () => {
      const conv = {
        id: "c-10",
        title: "Clean Chat",
        createdAt: "2026-08-10T10:00:00.000Z",
        messages: [
          { id: "m-1", role: "user", content: "Clean prompt", timestamp: "2026-08-10T10:00:01.000Z" },
          { id: "m-2", role: "assistant", content: "Clean answer", timestamp: "2026-08-10T10:00:02.000Z" },
        ],
      };
      const backup = createBackup([conv]);

      const result = restoreBackup(backup, []);

      expect(result.success).toBe(true);
      expect(result.count).toBe(1);
      expect(result.conversations[0].id).toBe("c-10");
      expect(result.conversations[0].title).toBe("Clean Chat");
      expect(result.conversations[0].messages).toHaveLength(2);
      expect(result.conversations[0].messages[0].content).toBe("Clean prompt");
    });

    it("export → import round trip preserves exact content, timestamps, and metadata", () => {
      const original = {
        id: "roundtrip-conv",
        title: "Roundtrip Test",
        createdAt: "2026-08-15T08:00:00.000Z",
        updatedAt: "2026-08-15T08:05:00.000Z",
        messages: [
          {
            id: "rt-m1",
            role: "user",
            content: "Roundtrip question",
            type: "text",
            timestamp: "2026-08-15T08:00:01.000Z",
          },
          {
            id: "rt-m2",
            role: "assistant",
            content: "Roundtrip answer",
            type: "text",
            timestamp: "2026-08-15T08:00:05.000Z",
            metadata: {
              model: "framerai-text",
              trace: {
                memories: [{ text: "rt-test", score: 0.88 }],
                affect: [0.1, 0.2],
              },
            },
          },
        ],
      };

      const exportedJson = JSON.stringify(createBackup([original]));
      const restoredResult = restoreBackup(exportedJson, []);

      expect(restoredResult.success).toBe(true);
      const restored = restoredResult.conversations[0];

      expect(restored.id).toBe(original.id);
      expect(restored.title).toBe(original.title);
      expect(restored.createdAt).toBe(original.createdAt);
      expect(restored.messages).toHaveLength(2);
      expect(restored.messages[0].content).toBe("Roundtrip question");
      expect(restored.messages[1].content).toBe("Roundtrip answer");
      expect(restored.messages[1].metadata.model).toBe("framerai-text");
      expect(restored.messages[1].metadata.trace.memories[0].text).toBe("rt-test");
      expect(restored.messages[1].metadata.trace.memories[0].score).toBe(0.88);
    });

    it("prevents accidental overwrite when conversation IDs collide with existing conversations", () => {
      const existingConv = {
        id: "collide-c1",
        title: "Existing Original Chat",
        messages: [{ id: "m-existing", role: "user", content: "Do not overwrite me!" }],
      };

      const incomingConv = {
        id: "collide-c1",
        title: "Imported Conflicting Chat",
        messages: [{ id: "m-incoming", role: "user", content: "I am from backup" }],
      };

      const backup = createBackup([incomingConv]);
      const result = restoreBackup(backup, [existingConv]);

      expect(result.success).toBe(true);
      const restored = result.conversations[0];

      // Must NOT overwrite: must be given a fresh ID
      expect(restored.id).not.toBe(existingConv.id);
      expect(restored.title).toBe("Imported Conflicting Chat");
      expect(restored.messages[0].content).toBe("I am from backup");

      // Existing conversation remains intact
      expect(existingConv.id).toBe("collide-c1");
      expect(existingConv.title).toBe("Existing Original Chat");
    });

    it("preserves nested branch relationships when parent and child conversation IDs collide and are remapped", () => {
      const existingConvs = [
        { id: "parent-1", title: "Existing Parent", messages: [] },
      ];

      const backupParent = {
        id: "parent-1",
        title: "Imported Parent",
        messages: [
          { id: "msg-p1", role: "user", content: "Parent prompt" },
          { id: "msg-p2", role: "assistant", content: "Parent reply" },
        ],
      };

      const backupChild = {
        id: "child-branch-1",
        title: "Imported Child Branch",
        parentConversationId: "parent-1",
        branchedFromMessageId: "msg-p2",
        messages: [
          { id: "msg-p1", role: "user", content: "Parent prompt" },
          { id: "msg-p2", role: "assistant", content: "Parent reply" },
          { id: "msg-c1", role: "user", content: "Branched turn" },
        ],
      };

      const backup = createBackup([backupParent, backupChild]);
      const result = restoreBackup(backup, existingConvs);

      expect(result.success).toBe(true);
      expect(result.conversations).toHaveLength(2);

      const restoredParent = result.conversations[0];
      const restoredChild = result.conversations[1];

      // Parent ID was remapped to avoid collision
      expect(restoredParent.id).not.toBe("parent-1");

      // Crucial: Child's parentConversationId MUST point to the remapped parent ID!
      expect(restoredChild.parentConversationId).toBe(restoredParent.id);

      // Child's branchedFromMessageId points to the parent message
      const parentMsgP2Id = restoredParent.messages[1].id;
      expect(restoredChild.branchedFromMessageId).toBe(parentMsgP2Id);
    });

    it("deduplicates message IDs within an imported conversation", () => {
      const convWithDuplicateMsgs = {
        id: "dup-msg-conv",
        title: "Duplicate Messages",
        messages: [
          { id: "same-id", role: "user", content: "First instance" },
          { id: "same-id", role: "user", content: "Duplicate dropped" },
          { id: "other-id", role: "assistant", content: "Second msg" },
        ],
      };

      const backup = createBackup(convWithDuplicateMsgs);
      const result = restoreBackup(backup, []);

      expect(result.success).toBe(true);
      const msgs = result.conversations[0].messages;
      expect(msgs).toHaveLength(2);
      expect(msgs[0].content).toBe("First instance");
      expect(msgs[1].content).toBe("Second msg");
    });

    it("handles large conversation backups with hundreds of messages efficiently", () => {
      const largeMessages = [];
      for (let i = 0; i < 500; i++) {
        largeMessages.push({
          id: `large-m-${i}`,
          role: i % 2 === 0 ? "user" : "assistant",
          content: `Message ${i} in large thread`,
          type: "text",
          timestamp: new Date(Date.now() + i * 1000).toISOString(),
        });
      }

      const largeConv = {
        id: "large-conv-1",
        title: "500-Message Thread",
        messages: largeMessages,
      };

      const backup = createBackup(largeConv);
      const t0 = performance.now();
      const result = restoreBackup(backup, []);
      const t1 = performance.now();

      expect(result.success).toBe(true);
      expect(result.conversations[0].messages).toHaveLength(500);
      expect(t1 - t0).toBeLessThan(1000); // Must be fast (< 1s)
      expect(result.conversations[0].messages[0].id).toBe("large-m-0");
      expect(result.conversations[0].messages[499].id).toBe("large-m-499");
    });
  });

  // =========================================================================
  // 4. HOOK INTEGRATION & PERSISTENCE TESTS (useChat)
  // =========================================================================
  describe("useChat Integration & Persistence", () => {
    it("importBackup updates state immediately without full page reload", async () => {
      const { result } = renderHook(() => useChat({}));

      await act(async () => {
        await result.current.createConversation();
      });
      const initialCount = result.current.conversations.length;

      const importPayload = createBackup({
        id: "imported-direct-1",
        title: "Imported via Hook",
        messages: [{ id: "m1", role: "user", content: "Imported greeting" }],
      });

      let importRes;
      await act(async () => {
        importRes = await result.current.importBackup(importPayload);
      });

      expect(importRes.success).toBe(true);
      expect(result.current.conversations.length).toBe(initialCount + 1);

      // Imported conversation is immediately active and messages visible
      expect(result.current.activeConversation).toBe("imported-direct-1");
      expect(result.current.messages).toHaveLength(1);
      expect(result.current.messages[0].content).toBe("Imported greeting");
    });

    it("imported conversations persist to localStorage and reload on refresh", async () => {
      const { result } = renderHook(() => useChat({}));

      const backup = createBackup({
        id: "persisted-import-1",
        title: "Persisted Chat",
        messages: [{ id: "m-p1", role: "user", content: "Persist me!" }],
      });

      await act(async () => {
        await result.current.importBackup(backup);
      });

      // Verify localStorage was updated
      const rawStored = localStorage.getItem(STORAGE_KEY);
      expect(rawStored).toBeTruthy();
      const parsed = JSON.parse(rawStored);
      expect(parsed.conversations.some((c) => c.id === "persisted-import-1")).toBe(true);

      // Simulate page refresh by loading from storage directly
      const refreshed = loadConversationsFromStorage(localStorage);
      expect(refreshed.conversations.some((c) => c.id === "persisted-import-1")).toBe(true);
      expect(refreshed.activeConversationId).toBe("persisted-import-1");
      expect(refreshed.messages[0].content).toBe("Persist me!");
    });

    it("failed import does not modify existing conversations in state or storage", async () => {
      const { result } = renderHook(() => useChat({}));

      await act(async () => {
        await result.current.createConversation();
      });
      const originalCount = result.current.conversations.length;
      const originalActive = result.current.activeConversation;

      // Malformed import payload
      let importRes;
      await act(async () => {
        importRes = await result.current.importBackup("{ invalid json ");
      });

      expect(importRes.success).toBe(false);
      expect(result.current.error).toMatch(/import failed/i);

      // State is completely unchanged
      expect(result.current.conversations.length).toBe(originalCount);
      expect(result.current.activeConversation).toBe(originalActive);
    });

    it("exportConversation and exportAllConversations return valid backup structures", async () => {
      const { result } = renderHook(() => useChat({}));

      await act(async () => {
        await result.current.createConversation();
      });

      const singleExport = result.current.exportConversation();
      expect(singleExport.version).toBe(BACKUP_VERSION);
      expect(singleExport.conversationCount).toBe(1);

      const allExport = result.current.exportAllConversations();
      expect(allExport.version).toBe(BACKUP_VERSION);
      expect(allExport.conversationCount).toBe(result.current.conversations.length);
    });
  });

  // =========================================================================
  // 5. UI CONTROLS & COMPONENT TESTS
  // =========================================================================
  describe("UI Component Controls", () => {
    it("Sidebar renders Export All and Import Backup buttons and hidden file input", async () => {
      const onExportAll = vi.fn();
      const onImportBackup = vi.fn();

      render(
        <Sidebar
          open={true}
          conversations={[{ id: "c1", title: "Chat 1" }]}
          activeId="c1"
          onExportAll={onExportAll}
          onImportBackup={onImportBackup}
        />
      );

      const exportBtn = screen.getByRole("button", { name: /export all conversations/i });
      const importBtn = screen.getByRole("button", { name: /import conversations backup/i });
      const fileInput = screen.getByTestId("sidebar-backup-file-input");

      expect(exportBtn).toBeInTheDocument();
      expect(importBtn).toBeInTheDocument();
      expect(fileInput).toBeInTheDocument();

      await userEvent.click(exportBtn);
      expect(onExportAll).toHaveBeenCalledTimes(1);
    });

    it("Chat header renders Export Conversation button", async () => {
      const onExportConversation = vi.fn();

      render(
        <Chat
          messages={[]}
          onExportConversation={onExportConversation}
        />
      );

      const exportBtn = screen.getByRole("button", { name: /export conversation/i });
      expect(exportBtn).toBeInTheDocument();

      await userEvent.click(exportBtn);
      expect(onExportConversation).toHaveBeenCalledTimes(1);
    });

    it("BackupModal renders Export and Import tabs, triggers actions, and displays feedback", async () => {
      const onExportConversation = vi.fn();
      const onExportAll = vi.fn();
      const onImportBackup = vi.fn(() => Promise.resolve({ success: true, count: 2 }));
      const onClose = vi.fn();

      render(
        <BackupModal
          open={true}
          conversations={[{ id: "c1", title: "Active Test Chat", messages: [] }]}
          activeConversationId="c1"
          onClose={onClose}
          onExportConversation={onExportConversation}
          onExportAll={onExportAll}
          onImportBackup={onImportBackup}
        />
      );

      expect(screen.getByRole("dialog", { name: /backup & restore/i })).toBeInTheDocument();

      // Export section test
      const jsonExportBtn = screen.getByRole("button", { name: /json backup/i });
      await userEvent.click(jsonExportBtn);
      expect(onExportConversation).toHaveBeenCalledWith("c1", "json");

      const allExportBtn = screen.getByRole("button", { name: /download full backup/i });
      await userEvent.click(allExportBtn);
      expect(onExportAll).toHaveBeenCalledTimes(1);

      // Switch to Import tab
      const importTab = screen.getByRole("tab", { name: /import/i });
      await userEvent.click(importTab);

      // Dropzone and file input exist
      const dropzone = screen.getByRole("button", { name: /click or drag and drop a backup json file/i });
      expect(dropzone).toBeInTheDocument();

      const fileInput = screen.getByTestId("backup-modal-file-input");
      const testFile = new File(['{"valid":true}'], "backup.json", { type: "application/json" });

      await userEvent.upload(fileInput, testFile);
      expect(onImportBackup).toHaveBeenCalledWith(testFile);
    });
  });

  // =========================================================================
  // 6. ISSUE #486: ADVANCED PORTABILITY, MARKDOWN IMPORT, AND ROBUSTNESS
  // =========================================================================
  describe("Issue #486 Portability & Validation Enhancements", () => {
    it("exports Markdown with multiple roles, code blocks, Unicode, attachments, and timestamps", () => {
      const conv = {
        id: "conv-complex",
        title: "Unicode & Code Chat 🚀",
        parentConversationId: "parent-001",
        branchedFromMessageId: "branch-msg-002",
        messages: [
          {
            id: "m-user",
            role: "user",
            content: "Please write a Python script with # comments and emojis ✨:\n```python\n# This is a comment\ndef greet():\n    return 'Hello, 世界'\n```",
            timestamp: "2026-08-01T10:00:00.000Z",
            attachments: ["data.csv", "chart.png"],
          },
          {
            id: "m-assistant",
            role: "assistant",
            content: "Here is your script:\n```python\n# Another comment\nprint('🚀 Done')\n```",
            timestamp: "2026-08-01T10:00:05.000Z",
          },
          {
            id: "m-system",
            role: "system",
            content: "System notice: memory updated.",
          },
        ],
      };

      const md = exportConversationToMarkdown(conv);
      expect(md).toContain("# Unicode & Code Chat 🚀");
      expect(md).toContain("*Parent Conversation: `parent-001`*");
      expect(md).toContain("*Branched from Message: `branch-msg-002`*");
      expect(md).toContain("### User (2026-08-01T10:00:00.000Z)");
      expect(md).toContain("*Attachments: data.csv, chart.png*");
      expect(md).toContain("def greet():");
      expect(md).toContain("### Assistant (2026-08-01T10:00:05.000Z)");
      expect(md).toContain("### System");
    });

    it("parses exported Markdown back into structured conversation without breaking on inner code block comments", () => {
      const sampleMarkdown = [
        "# Python Architecture",
        "",
        "*Parent Conversation: `parent-abc`*",
        "*Branched from Message: `msg-xyz`*",
        "",
        "---",
        "",
        "### User (2026-08-01T10:00:00.000Z)",
        "",
        "Can you fix this snippet?",
        "```python",
        "# This is a python comment",
        "### This is NOT a header",
        "def foo():",
        "    return 42",
        "```",
        "",
        "*Attachments: test.py, notes.md*",
        "",
        "---",
        "",
        "### Assistant (2026-08-01T10:00:05.000Z)",
        "",
        "Sure! The snippet looks solid.",
        "",
        "---",
      ].join("\n");

      const result = parseMarkdownConversation(sampleMarkdown);
      expect(result.success).toBe(true);
      expect(result.conversation).toBeDefined();

      const conv = result.conversation;
      expect(conv.title).toBe("Python Architecture");
      expect(conv.parentConversationId).toBe("parent-abc");
      expect(conv.branchedFromMessageId).toBe("msg-xyz");
      expect(conv.messages).toHaveLength(2);

      // Verify user message
      expect(conv.messages[0].role).toBe("user");
      expect(conv.messages[0].timestamp).toBe("2026-08-01T10:00:00.000Z");
      expect(conv.messages[0].attachments).toEqual(["test.py", "notes.md"]);
      expect(conv.messages[0].content).toContain("### This is NOT a header");
      expect(conv.messages[0].content).toContain("# This is a python comment");

      // Verify assistant message
      expect(conv.messages[1].role).toBe("assistant");
      expect(conv.messages[1].content).toBe("Sure! The snippet looks solid.");
    });

    it("parses empty Markdown conversation title with no messages", () => {
      const md = "# Empty Conversation\n\n*Exported on 2026-08-01T10:00:00.000Z*\n\n---\n";
      const result = parseMarkdownConversation(md);
      expect(result.success).toBe(true);
      expect(result.conversation.title).toBe("Empty Conversation");
      expect(result.conversation.messages).toHaveLength(0);
    });

    it("rejects empty or whitespace-only Markdown files", () => {
      expect(parseMarkdownConversation("").success).toBe(false);
      expect(parseMarkdownConversation("   \n\t  ").success).toBe(false);
    });

    it("supports single-conversation JSON import format without top-level conversations array", () => {
      const singleJson = {
        id: "single-conv-001",
        title: "Single Conversation",
        messages: [
          { id: "m1", role: "user", content: "Hello single!" },
          { id: "m2", role: "assistant", content: "Hi there!" },
        ],
      };

      const result = validateBackup(singleJson);
      expect(result.valid).toBe(true);
      expect(result.backup.conversations).toHaveLength(1);
      expect(result.backup.conversations[0].id).toBe("single-conv-001");
      expect(result.backup.conversations[0].messages).toHaveLength(2);
    });

    it("validates and restores Markdown conversation via validateBackup and restoreBackup", () => {
      const mdContent = [
        "# Restored Chat",
        "",
        "### User (2026-08-01T12:00:00.000Z)",
        "Hello from markdown!",
        "",
        "---",
        "",
        "### Assistant (2026-08-01T12:00:01.000Z)",
        "Hello! Restored successfully.",
      ].join("\n");

      const validation = validateBackup(mdContent, { filename: "chat.md" });
      expect(validation.valid).toBe(true);
      expect(validation.format).toBe("markdown");

      const restore = restoreBackup(mdContent, [], { filename: "chat.md" });
      expect(restore.success).toBe(true);
      expect(restore.count).toBe(1);
      expect(restore.conversations[0].title).toBe("Restored Chat");
      expect(restore.conversations[0].messages).toHaveLength(2);
    });

    it("enforces MAX_IMPORT_FILE_SIZE limit (10MB) for File objects and raw strings", () => {
      expect(MAX_IMPORT_FILE_SIZE).toBe(10 * 1024 * 1024);

      // Oversized string payload
      const hugeString = "a".repeat(10 * 1024 * 1024 + 10);
      const stringResult = validateBackup(hugeString);
      expect(stringResult.valid).toBe(false);
      expect(stringResult.error).toMatch(/exceeds maximum allowed size of 10MB/i);

      // Oversized File object
      const hugeFile = {
        name: "huge_backup.json",
        size: 11 * 1024 * 1024,
      };
      const fileResult = validateBackup(hugeFile);
      expect(fileResult.valid).toBe(false);
      expect(fileResult.error).toMatch(/exceeds maximum allowed size of 10MB/i);
    });

    it("resolves duplicate message IDs safely and maintains unique IDs during restoration", () => {
      const backupWithDuplicateMsgIds = {
        version: BACKUP_VERSION,
        conversations: [
          {
            id: "conv-dup-msg",
            title: "Duplicate Msg ID Chat",
            messages: [
              { id: "duplicate-id", role: "user", content: "First turn" },
              { id: "duplicate-id", role: "assistant", content: "Second turn" },
            ],
          },
        ],
      };

      const restored = restoreBackup(backupWithDuplicateMsgIds, []);
      expect(restored.success).toBe(true);
      const messages = restored.conversations[0].messages;
      expect(messages).toHaveLength(2);
      expect(messages[0].id).not.toBe(messages[1].id);
      expect(messages[0].content).toBe("First turn");
      expect(messages[1].content).toBe("Second turn");
    });

    it("verifies full round-trip: export to Markdown -> restore from Markdown preserves data", () => {
      const originalConv = {
        id: "original-001",
        title: "Full Round Trip",
        messages: [
          { id: "m1", role: "user", content: "Step 1: Test export" },
          { id: "m2", role: "assistant", content: "Step 2: Output received" },
        ],
      };

      const exportedMd = exportConversationToMarkdown(originalConv);
      const restored = restoreBackup(exportedMd, [], { filename: "export.md" });

      expect(restored.success).toBe(true);
      expect(restored.conversations[0].title).toBe("Full Round Trip");
      expect(restored.conversations[0].messages[0].role).toBe("user");
      expect(restored.conversations[0].messages[0].content).toBe("Step 1: Test export");
      expect(restored.conversations[0].messages[1].role).toBe("assistant");
      expect(restored.conversations[0].messages[1].content).toBe("Step 2: Output received");
    });

    it("useChat importBackup rejects oversized file without altering state or storage", async () => {
      const { result } = renderHook(() => useChat({}));

      await act(async () => {
        result.current.createConversation();
      });
      const initialCount = result.current.conversations.length;

      const oversizedFile = new File(["some content"], "oversized.json", { type: "application/json" });
      Object.defineProperty(oversizedFile, "size", { value: 12 * 1024 * 1024 });

      let importRes;
      await act(async () => {
        importRes = await result.current.importBackup(oversizedFile);
      });

      expect(importRes.success).toBe(false);
      expect(importRes.error).toMatch(/exceeds maximum allowed size of 10MB/i);
      expect(result.current.error).toMatch(/exceeds maximum allowed size of 10MB/i);
      expect(result.current.conversations).toHaveLength(initialCount);
    });

    it("useChat importBackup successfully restores a Markdown File object", async () => {
      const { result } = renderHook(() => useChat({}));

      const mdContent = "# Uploaded Markdown\n\n### User\nHello Vitest\n\n---\n\n### Assistant\nHello Agent";
      const mdFile = new File([mdContent], "test-chat.md", { type: "text/markdown" });

      let importRes;
      await act(async () => {
        importRes = await result.current.importBackup(mdFile);
      });

      expect(importRes.success).toBe(true);
      expect(importRes.count).toBe(1);
      expect(result.current.activeConversation).toBe(importRes.activeId);
      expect(result.current.messages).toHaveLength(2);
      expect(result.current.messages[0].content).toBe("Hello Vitest");
    });
  });
});

