const API_BASE = "/api";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || "Request failed");
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

export const api = {
  // Health
  health: () => request("/health"),

  // Conversations
  createConversation: () => request("/chat/conversations", { method: "POST" }),
  listConversations: () => request("/chat/conversations"),
  getConversation: (id) => request(`/chat/conversations/${id}`),
  deleteConversation: (id) =>
    request(`/chat/conversations/${id}`, { method: "DELETE" }),

  // Messages
  sendMessage: (conversationId, content, type = "text", attachments = [], settings) =>
    request(`/chat/conversations/${conversationId}/messages`, {
      method: "POST",
      body: JSON.stringify({ content, type, attachments, settings }),
    }),

  // Generation
  generateImage: (prompt, numImages = 1, resolution = 256) =>
    request("/generate/image", {
      method: "POST",
      body: JSON.stringify({ prompt, num_images: numImages, resolution }),
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
  transcribe: (file, prompt = "Transcribe the audio:") => {
    const form = new FormData();
    form.append("audio", file);
    form.append("prompt", prompt);
    return uploadRequest("/generate/transcribe", form);
  },
};
