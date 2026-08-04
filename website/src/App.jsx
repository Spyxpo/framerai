import React, { useEffect, useState } from "react";
import Sidebar from "./components/Sidebar/Sidebar";
import Chat from "./components/Chat/Chat";
import SettingsPanel from "./components/Settings/SettingsPanel";
import { useChat } from "./hooks/useChat";
import { useSettings } from "./hooks/useSettings";
import { api } from "./services/api";

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [model, setModel] = useState(null);
  const { settings, updateSetting, resetSettings } = useSettings();
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
  } = useChat(settings);

  // The settings panel shows which checkpoint the backend is serving.
  useEffect(() => {
    api
      .health()
      .then((info) => setModel(info.model))
      .catch(() => setModel(null));
  }, []);

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
        onOpenSettings={() => setSettingsOpen(true)}
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
        onOpenSettings={() => setSettingsOpen(true)}
      />
      <SettingsPanel
        open={settingsOpen}
        settings={settings}
        model={model}
        onChange={updateSetting}
        onReset={resetSettings}
        onClose={() => setSettingsOpen(false)}
      />
    </div>
  );
}
