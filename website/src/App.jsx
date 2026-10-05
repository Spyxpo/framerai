import { useState, useRef, useCallback, useEffect } from "react";
import Sidebar from "./components/Sidebar/Sidebar";
import Chat from "./components/Chat/Chat";
import SettingsPanel from "./components/Settings/SettingsPanel";
import SearchModal from "./components/Search/SearchModal";
import { useChat } from "./hooks/useChat";
import { useSettings } from "./hooks/useSettings";
import { api } from "./services/api";

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [highlightedMessageId, setHighlightedMessageId] = useState(null);
  const highlightTimeoutRef = useRef(null);
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
    pendingApproval,
    branching,
    createConversation,
    selectConversation,
    deleteConversation,
    clearAllConversations,
    sendMessage,
    branchConversation,
    dismissError,
    approveCommand,
    denyCommand,
  } = useChat(settings);

  // The settings panel shows which checkpoint the backend is serving.
  useEffect(() => {
    api
      .health()
      .then((info) => setModel(info.model))
      .catch(() => setModel(null));
  }, []);

  // Global keyboard shortcut: Cmd+K / Ctrl+K toggles global search
  useEffect(() => {
    const handleGlobalKeyDown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setSearchOpen((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleGlobalKeyDown);
    return () => window.removeEventListener("keydown", handleGlobalKeyDown);
  }, []);

  // Clean up highlight timer on unmount
  useEffect(() => {
    return () => {
      if (highlightTimeoutRef.current) {
        clearTimeout(highlightTimeoutRef.current);
      }
    };
  }, []);

  // Select conversation → focus textarea when done
  const handleSelectConversation = useCallback(async (id) => {
    await selectConversation(id);
    textareaFocusRef.current?.();
  }, [selectConversation]);

  // Navigate to search result (conversation + message)
  const handleNavigateToResult = useCallback(async (result) => {
    if (!result) return;
    if (result.conversationId) {
      await selectConversation(result.conversationId);
    }
    if (result.messageId) {
      if (highlightTimeoutRef.current) {
        clearTimeout(highlightTimeoutRef.current);
      }
      setHighlightedMessageId(result.messageId);
      highlightTimeoutRef.current = setTimeout(() => {
        setHighlightedMessageId(null);
      }, 3000);
    } else {
      textareaFocusRef.current?.();
    }
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
        onOpenSearch={() => setSearchOpen(true)}
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
        pendingApproval={pendingApproval}
        onApproveCommand={approveCommand}
        onDenyCommand={denyCommand}
        branching={branching}
        onBranch={branchConversation}
        sidebarOpen={sidebarOpen}
        onSend={sendMessage}
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
        onDismissError={handleDismissError}
        onOpenSettings={() => setSettingsOpen(true)}
        onOpenSearch={() => setSearchOpen(true)}
        highlightedMessageId={highlightedMessageId}
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
      <SearchModal
        open={searchOpen}
        conversations={conversations}
        activeConversationId={activeConversation}
        onClose={() => setSearchOpen(false)}
        onNavigate={handleNavigateToResult}
      />
    </div>
  );
}
