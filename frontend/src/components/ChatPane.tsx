import { useEffect, useRef, useState, useCallback } from "react";
import { Send, Square, AlertCircle, Brain, Thermometer, Hash } from "lucide-react";
import clsx from "clsx";
import Message from "./Message";
import ModelBadge from "./ModelBadge";
import type { Conversation, ServerStatus, Settings } from "../types";

interface Props {
  conversation: Conversation | null;
  serverStatus: ServerStatus;
  settings: Settings;
  isStreaming: boolean;
  error: string | null;
  onSend: (text: string) => void;
  onAbort: () => void;
  onNew: () => void;
  onSettingChange: <K extends keyof Settings>(key: K, value: Settings[K]) => void;
}

// ── Inline editable chip ──────────────────────────────────────────────────────

function ParamChip({
  icon,
  label,
  value,
  min,
  max,
  step: _step,
  integer,
  onChange,
}: {
  icon: React.ReactNode;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  integer?: boolean;
  onChange: (v: number) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(String(value));
  const inputRef = useRef<HTMLInputElement>(null);

  const commit = () => {
    const n = integer ? parseInt(draft, 10) : parseFloat(draft);
    if (!isNaN(n)) onChange(Math.min(max, Math.max(min, n)));
    setEditing(false);
  };

  useEffect(() => {
    if (editing) {
      setDraft(integer ? String(value) : value.toFixed(2));
      inputRef.current?.select();
    }
  }, [editing, value, integer]);

  return (
    <button
      type="button"
      title={label}
      onClick={() => setEditing(true)}
      className="flex items-center gap-1 px-2 py-1 rounded-md bg-gray-700 hover:bg-gray-600 text-gray-300 text-xs transition-colors"
    >
      {icon}
      {editing ? (
        <input
          ref={inputRef}
          className="w-14 bg-transparent text-gray-100 focus:outline-none"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") commit();
            if (e.key === "Escape") setEditing(false);
          }}
          autoFocus
          onClick={(e) => e.stopPropagation()}
        />
      ) : (
        <span>{integer ? value : value.toFixed(2)}</span>
      )}
    </button>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function ChatPane({
  conversation,
  serverStatus,
  settings,
  isStreaming,
  error,
  onSend,
  onAbort,
  onNew,
  onSettingChange,
}: Props) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversation?.messages]);

  // Auto-resize textarea
  const handleInput = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px";
  }, []);

  const canSend = serverStatus.online && serverStatus.model !== null;

  const submit = useCallback(() => {
    const text = input.trim();
    if (!text || isStreaming || !canSend) return;
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    onSend(text);
  }, [input, isStreaming, canSend, onSend]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        submit();
      }
    },
    [submit]
  );

  const messages = conversation?.messages ?? [];

  // Warning when server is online but no model is loaded
  const noModelWarning = serverStatus.online && serverStatus.model === null;

  return (
    <div className="flex flex-col flex-1 h-full min-w-0">
      {/* Top bar */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-gray-800 bg-gray-950">
        <h1 className="text-sm font-medium text-gray-200 truncate">
          {conversation?.title ?? "XandLLM Chat"}
        </h1>
        <ModelBadge status={serverStatus} />
      </div>

      {/* No-model warning */}
      {noModelWarning && (
        <div className="mx-4 mt-3 flex items-center gap-2 px-3 py-2 bg-yellow-950 border border-yellow-700 rounded-lg text-yellow-300 text-sm">
          <AlertCircle size={14} className="flex-shrink-0" />
          Server is online but no model is loaded. Start the server with{" "}
          <code className="text-xs bg-yellow-900/50 px-1 py-0.5 rounded">
            xandllm serve --model &lt;id&gt;
          </code>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-8 gap-4">
            <div className="w-16 h-16 bg-blue-600/20 rounded-2xl flex items-center justify-center">
              <span className="text-3xl">🤖</span>
            </div>
            <div>
              <h2 className="text-xl font-semibold text-white mb-1">
                How can I help you today?
              </h2>
              <p className="text-gray-400 text-sm">
                {serverStatus.online
                  ? serverStatus.model
                    ? `Model: ${serverStatus.model}`
                    : "No model loaded"
                  : "Start the server with: xandllm serve --model <id>"}
              </p>
            </div>

            {/* Quick-start suggestions */}
            {canSend && (
              <div className="flex flex-wrap gap-2 justify-center mt-2">
                {[
                  "Explain quantum computing",
                  "Write a Python script",
                  "Help me debug code",
                ].map((s) => (
                  <button
                    key={s}
                    onClick={() => { onNew(); onSend(s); }}
                    className="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-full text-xs text-gray-300 transition-colors"
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="max-w-3xl mx-auto py-4">
            {messages.map((msg) => (
              <Message key={msg.id} message={msg} showThink={settings.showThink} />
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Error banner */}
      {error && (
        <div className="mx-6 mb-2 flex items-center gap-2 px-3 py-2 bg-red-950 border border-red-800 rounded-lg text-red-300 text-sm">
          <AlertCircle size={14} className="flex-shrink-0" />
          {error}
        </div>
      )}

      {/* Input area */}
      <div className="px-4 pb-4 pt-2 bg-gray-950">
        <div className="max-w-3xl mx-auto">
          {/* ── Inline parameter bar ── */}
          <div className="flex items-center gap-2 mb-2 flex-wrap">
            <ParamChip
              icon={<Thermometer size={11} />}
              label="Temperature"
              value={settings.temperature}
              min={0}
              max={2}
              step={0.05}
              onChange={(v) => onSettingChange("temperature", v)}
            />
            <ParamChip
              icon={<Hash size={11} />}
              label="Max tokens"
              value={settings.maxTokens}
              min={64}
              max={32768}
              step={64}
              integer
              onChange={(v) => onSettingChange("maxTokens", v)}
            />
            {/* Show-think toggle */}
            <button
              type="button"
              title={settings.showThink ? "Hide thinking" : "Show thinking"}
              onClick={() => onSettingChange("showThink", !settings.showThink)}
              className={clsx(
                "flex items-center gap-1 px-2 py-1 rounded-md text-xs transition-colors",
                settings.showThink
                  ? "bg-amber-600/30 text-amber-300 hover:bg-amber-600/50"
                  : "bg-gray-700 text-gray-400 hover:bg-gray-600"
              )}
            >
              <Brain size={11} />
              {settings.showThink ? "Think: on" : "Think: off"}
            </button>
          </div>

          {/* ── Textarea + send button ── */}
          <div className="flex items-end gap-2 bg-gray-800 border border-gray-700 rounded-2xl px-4 py-3 focus-within:border-blue-500 transition-colors">
            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={handleInput}
              onKeyDown={handleKeyDown}
              placeholder={
                !serverStatus.online
                  ? "Server is offline"
                  : noModelWarning
                  ? "No model loaded"
                  : "Message XandLLM… (Shift+Enter for newline)"
              }
              disabled={!canSend && !isStreaming}
              className="flex-1 bg-transparent text-sm text-gray-100 placeholder-gray-500 resize-none focus:outline-none max-h-48 leading-relaxed disabled:opacity-40"
            />
            <button
              onClick={isStreaming ? onAbort : submit}
              disabled={!isStreaming && (!input.trim() || !canSend)}
              className={clsx(
                "flex-shrink-0 w-8 h-8 rounded-xl flex items-center justify-center transition-all",
                isStreaming
                  ? "bg-red-600 hover:bg-red-500 text-white"
                  : input.trim() && canSend
                  ? "bg-blue-600 hover:bg-blue-500 text-white"
                  : "bg-gray-700 text-gray-500 cursor-not-allowed"
              )}
            >
              {isStreaming ? <Square size={14} /> : <Send size={14} />}
            </button>
          </div>
          <p className="text-xs text-gray-600 text-center mt-2">
            XandLLM can make mistakes. Verify important information.
          </p>
        </div>
      </div>
    </div>
  );
}
