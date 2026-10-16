const API_BASE = "/api";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    const error = new Error(err.error || err.message || "Request failed");
    error.status = res.status;
    error.code = err.code;
    error.details = err.details;
    throw error;
  }
  return res.json();
}

async function uploadRequest(path, formData) {
  const res = await fetch(`${API_BASE}${path}`, { method: "POST", body: formData });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || "Upload failed");
  }
  return res.json();
}


// Types the backend accepts for a file a browser leaves unlabelled. Source code
// and office files often arrive as "" or application/octet-stream, which the
// server refuses, so they are labelled by extension before upload.
const TYPES_BY_EXTENSION = {
  docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  pptx: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  xlsx: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  csv: "text/csv", html: "text/html", htm: "text/html", json: "application/json",
  md: "text/markdown", txt: "text/plain", py: "text/x-python", js: "text/javascript",
  mjs: "text/javascript", ts: "application/typescript", tsx: "application/typescript",
  jsx: "text/javascript", java: "text/x-java-source", go: "text/x-go", rs: "text/x-rust",
  c: "text/x-c", h: "text/x-c", cpp: "text/x-c++src", cc: "text/x-c++src", hpp: "text/x-c++src",
  cs: "text/x-csharp", rb: "text/x-ruby", php: "text/x-php", sh: "application/x-sh",
  sql: "application/sql", css: "text/css", yaml: "application/yaml", yml: "application/yaml",
  toml: "application/toml",
};

export function labelled(file) {
  const ext = (file.name || "").split(".").pop().toLowerCase();
  const type = TYPES_BY_EXTENSION[ext];
  // A browser's own label wins unless it is missing or generic, or names a
  // type the server would refuse for this extension (.ts as video/mp2t).
  const generic = !file.type || file.type === "application/octet-stream" || file.type === "video/mp2t";
  if (!type || !generic) return file;
  return new File([file], file.name, { type, lastModified: file.lastModified });
}

export const api = {
  // Health
  health: () => request("/health"),

  // Conversations
  createConversation: () => request("/chat/conversations", { method: "POST" }),
  listConversations: () => request("/chat/conversations"),
  getConversation: (id) => request(`/chat/conversations/${id}`),
  updateConversation: (id, updates = {}) =>
    request(`/chat/conversations/${id}`, {
      method: "PATCH",
      body: JSON.stringify(updates),
    }),
  deleteConversation: (id) =>
    request(`/chat/conversations/${id}`, { method: "DELETE" }),
  syncConversation: (id, syncPayload = {}) =>
    request(`/chat/conversations/${id}/sync`, {
      method: "POST",
      body: JSON.stringify(syncPayload),
    }),
  // Put back a conversation the server lost, under its own id, from the copy kept
  // here (Issue #503). 409 when the server holds one by that id.
  restoreConversation: (id, payload = {}) =>
    request(`/chat/conversations/${id}/restore`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  branchConversation: (conversationId, messageId, expectedVersion) =>
    request(`/chat/conversations/${conversationId}/branch`, {
      method: "POST",
      body: JSON.stringify({
        messageId,
        ...(expectedVersion !== undefined ? { expectedVersion } : {}),
      }),
    }),

  // Messages
  sendMessage: (conversationId, content, type = "text", attachments = [], settings, optionsOrExpectedVersion = {}) => {
    const options = typeof optionsOrExpectedVersion === "number"
      ? { expectedVersion: optionsOrExpectedVersion }
      : (optionsOrExpectedVersion || {});
    return request(`/chat/conversations/${conversationId}/messages`, {
      method: "POST",
      body: JSON.stringify({
        content,
        type,
        attachments,
        settings,
        ...options,
      }),
    });
  },
  editMessage: (conversationId, messageId, content, type = "text", attachments = [], settings, expectedVersion) =>
    request(`/chat/conversations/${conversationId}/messages`, {
      method: "POST",
      body: JSON.stringify({
        content,
        type,
        attachments,
        settings,
        editMessageId: messageId,
        ...(expectedVersion !== undefined ? { expectedVersion } : {}),
      }),
    }),
  regenerateResponse: (conversationId, messageId, settings, content = "", expectedVersion) =>
    request(`/chat/conversations/${conversationId}/messages`, {
      method: "POST",
      body: JSON.stringify({
        content,
        regenerateMessageId: messageId,
        settings,
        ...(expectedVersion !== undefined ? { expectedVersion } : {}),
      }),
    }),
  deleteMessage: (conversationId, messageId) =>
    request(`/chat/conversations/${conversationId}/messages/${messageId}`, {
      method: "DELETE",
    }),

  // Generation
  generateImage: (prompt, numImages = 1, size = {}) =>
    request("/generate/image", {
      method: "POST",
      // Only send the size fields that were actually set, so the backend can
      // fall through to prompt intent and then to its own default.
      body: JSON.stringify({ prompt, num_images: numImages, ...size }),
    }),

  generateVideo: (prompt, numFrames = 16) =>
    request("/generate/video", {
      method: "POST",
      body: JSON.stringify({ prompt, num_frames: numFrames }),
    }),

  generateCode: (prompt, language = "python") =>
    request("/generate/code", {
      method: "POST",
      body: JSON.stringify({ prompt, language }),
    }),

  generateAudio: (prompt) =>
    request("/generate/audio", {
      method: "POST",
      body: JSON.stringify({ prompt }),
    }),

  // Audio understanding (upload -> transcription)
  // Store a file and get back the path a chat message can attach. No model runs.
  uploadAttachment: (file) => {
    const form = new FormData();
    form.append("file", labelled(file));
    return uploadRequest("/generate/upload", form);
  },

  transcribe: (file, prompt = "Transcribe the audio:") => {
    const form = new FormData();
    form.append("audio", file);
    form.append("prompt", prompt);
    return uploadRequest("/generate/transcribe", form);
  },
};
