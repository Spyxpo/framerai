import React, { useState } from "react";
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
    loadingConversations,
    loadingMessages,
    error,
    createConversation,
    selectConversation,
    deleteConversation,
    sendMessage,
    dismissError,
  } = useChat();

  return (
    <div className="app">
      <Sidebar
        open={sidebarOpen}
        conversations={conversations}
        activeId={activeConversation}
        loadingConversations={loadingConversations}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        onNew={createConversation}
        onSelect={selectConversation}
        onDelete={deleteConversation}
      />
      <Chat
        messages={messages}
        loading={loading}
        streaming={streaming}
        loadingMessages={loadingMessages}
        error={error}
        sidebarOpen={sidebarOpen}
        onSend={sendMessage}
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
        onDismissError={dismissError}
      />
    </div>
  );
}
