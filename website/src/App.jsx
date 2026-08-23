import { useState, useRef, useCallback, useEffect } from "react";
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
  const chatFocusRef = useRef(null);          // Chat: focus first suggestion or textarea
  const textareaFocusRef = useRef(null);      // Chat: focus textarea directly
  const chatSettingsFocusRef = useRef(null);  // Chat: focus header settings button
  const sidebarFocusRef = useRef(null);       // Sidebar: focus New Chat button
  const sidebarSettingsFocusRef = useRef(null); // Sidebar: focus footer settings button

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
    clearAllConversations,
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

  // Select conversation → focus textarea when done
  const handleSelectConversation = useCallback(async (id) => {
    await selectConversation(id);
    textareaFocusRef.current?.();
  }, [selectConversation]);

  // Sidebar → → Chat area
  const focusChatArea = useCallback(() => {
    chatFocusRef.current?.();
  }, []);

  // Sidebar → → Chat header settings button
  const focusChatSettings = useCallback(() => {
    chatSettingsFocusRef.current?.();
  }, []);

  // Chat ← → Sidebar New Chat button
  const focusSidebar = useCallback(() => {
    sidebarFocusRef.current?.();
  }, []);

  // Chat settings ← → Sidebar footer settings button
  const focusSidebarSettings = useCallback(() => {
    sidebarSettingsFocusRef.current?.();
  }, []);

  // Delete conversation → focus New Chat button after React re-renders
  const handleDeleteConversation = useCallback(async (id) => {
    await deleteConversation(id);
    // Two frames: first lets React flush state, second lets DOM settle
    setTimeout(() => requestAnimationFrame(() => sidebarFocusRef.current?.()), 0);
  }, [deleteConversation]);

  // Dismiss error and return focus to textarea (improvement a)
  const handleDismissError = useCallback(() => {
    dismissError();
    // setTimeout lets React re-render (remove the banner) before shifting focus
    setTimeout(() => textareaFocusRef.current?.(), 0);
  }, [dismissError]);

  return (
    <div className="app">
      <a href="#chat-input" className="skip-link">Skip to chat input</a>
      <Sidebar
        open={sidebarOpen}
        conversations={conversations}
        activeId={activeConversation}
        loadingConversations={loadingConversations}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        onNew={createConversation}
        onSelect={handleSelectConversation}
        onDelete={handleDeleteConversation}
        onClearAll={clearAllConversations}
        onOpenSettings={() => setSettingsOpen(true)}
        onFocusChat={focusChatArea}
        onFocusChatSettings={focusChatSettings}
        focusRef={sidebarFocusRef}
        footerSettingsFocusRef={sidebarSettingsFocusRef}
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
        onDismissError={handleDismissError}
        onOpenSettings={() => setSettingsOpen(true)}
        focusRef={chatFocusRef}
        textareaFocusRef={textareaFocusRef}
        chatSettingsFocusRef={chatSettingsFocusRef}
        onFocusSidebar={focusSidebar}
        onFocusSidebarSettings={focusSidebarSettings}
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
