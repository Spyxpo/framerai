import React, { useState, useCallback } from "react";
import Sidebar from "./components/Sidebar/Sidebar";
import Chat from "./components/Chat/Chat";
import { useChat } from "./hooks/useChat";

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const {
    conversations,
    activeConversation,
    messages,
    loading,
    streaming,
    createConversation,
    selectConversation,
    deleteConversation,
    sendMessage,
  } = useChat();

  return (
    <div className="app">
      <Sidebar
        open={sidebarOpen}
        conversations={conversations}
        activeId={activeConversation}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        onNew={createConversation}
        onSelect={selectConversation}
        onDelete={deleteConversation}
      />
      <Chat
        messages={messages}
        loading={loading}
        streaming={streaming}
        sidebarOpen={sidebarOpen}
        onSend={sendMessage}
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
      />
    </div>
  );
}
