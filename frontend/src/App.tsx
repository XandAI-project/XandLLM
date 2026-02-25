import { useState, useCallback, useEffect } from "react";
import Sidebar from "./components/Sidebar";
import ChatPane from "./components/ChatPane";
import SettingsModal from "./components/SettingsModal";
import { useConversations } from "./hooks/useConversations";
import { useServerInfo } from "./hooks/useServerInfo";
import { useChat } from "./hooks/useChat";
import type { Settings } from "./types";
import { DEFAULT_SETTINGS } from "./types";

const SETTINGS_KEY = "xandllm_settings";

function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    return raw ? { ...DEFAULT_SETTINGS, ...JSON.parse(raw) } : DEFAULT_SETTINGS;
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export default function App() {
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [showSettings, setShowSettings] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const serverStatus = useServerInfo(settings.baseUrl);

  const {
    conversations,
    active,
    activeId,
    newConversation,
    selectConversation,
    deleteConversation,
    appendMessage,
    updateLastAssistantMessage,
    getApiMessages,
  } = useConversations();

  const { send, abort, isStreaming, error } = useChat({
    settings,
    model: serverStatus.model,
    conversationId: activeId,
    getApiMessages,
    appendMessage,
    updateLastAssistantMessage,
  });

  // Auto-create first conversation on load if none exist
  useEffect(() => {
    if (conversations.length === 0) {
      newConversation();
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSaveSettings = useCallback((next: Settings) => {
    setSettings(next);
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(next));
    serverStatus.refresh();
  }, [serverStatus]);

  const handleSettingChange = useCallback(<K extends keyof Settings>(key: K, value: Settings[K]) => {
    setSettings((prev) => {
      const next = { ...prev, [key]: value };
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  const handleSend = useCallback(
    (text: string) => {
      // Ensure we have an active conversation
      const id = activeId ?? newConversation();
      if (!activeId) {
        // newConversation sets activeId asynchronously, so just call send
        // on next tick
        setTimeout(() => send(text), 0);
      } else {
        void id;
        send(text);
      }
    },
    [activeId, newConversation, send]
  );

  return (
    <div className="flex h-full bg-gray-950 overflow-hidden">
      {/* Sidebar */}
      {sidebarOpen && (
        <Sidebar
          conversations={conversations}
          activeId={activeId}
          onNew={newConversation}
          onSelect={selectConversation}
          onDelete={deleteConversation}
          onOpenSettings={() => setShowSettings(true)}
        />
      )}

      {/* Sidebar toggle */}
      <button
        onClick={() => setSidebarOpen((v) => !v)}
        className="absolute top-3 left-3 z-10 w-8 h-8 flex items-center justify-center text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
        title={sidebarOpen ? "Close sidebar" : "Open sidebar"}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <line x1="3" y1="6" x2="21" y2="6" />
          <line x1="3" y1="12" x2="21" y2="12" />
          <line x1="3" y1="18" x2="21" y2="18" />
        </svg>
      </button>

      {/* Main chat area */}
      <main className="flex-1 flex overflow-hidden">
        <ChatPane
          conversation={active}
          serverStatus={serverStatus}
          settings={settings}
          isStreaming={isStreaming}
          error={error}
          onSend={handleSend}
          onAbort={abort}
          onNew={newConversation}
          onSettingChange={handleSettingChange}
        />
      </main>

      {/* Settings modal */}
      {showSettings && (
        <SettingsModal
          settings={settings}
          onSave={handleSaveSettings}
          onClose={() => setShowSettings(false)}
        />
      )}
    </div>
  );
}
