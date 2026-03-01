import { useState } from "react";
import { X } from "lucide-react";
import type { Settings } from "../types";

interface Props {
  settings: Settings;
  onSave: (next: Settings) => void;
  onClose: () => void;
}

export default function SettingsModal({ settings, onSave, onClose }: Props) {
  const [draft, setDraft] = useState<Settings>({ ...settings });

  function field<K extends keyof Settings>(key: K) {
    return (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      const raw = e.target.value;
      const value =
        key === "temperature" || key === "topP"
          ? parseFloat(raw)
          : key === "maxTokens"
          ? parseInt(raw, 10)
          : raw;
      setDraft((d) => ({ ...d, [key]: value }));
    };
  }

  function toggle<K extends keyof Settings>(key: K) {
    setDraft((d) => ({ ...d, [key]: !d[key] }));
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-gray-900 border border-gray-700 rounded-xl w-full max-w-lg mx-4 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-700">
          <h2 className="text-lg font-semibold text-white">Settings</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Body */}
        <div className="px-6 py-4 space-y-5">
          {/* API URL */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              API base URL
            </label>
            <input
              type="url"
              value={draft.baseUrl}
              onChange={field("baseUrl")}
              className="w-full bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="http://localhost:11435"
            />
            <p className="text-xs text-gray-500 mt-1">
              Change to connect to a different running XandLLM server.
            </p>
          </div>

          {/* System prompt */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              System prompt
            </label>
            <textarea
              rows={3}
              value={draft.systemPrompt}
              onChange={field("systemPrompt")}
              className="w-full bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
            />
          </div>

          {/* Temperature */}
          <div>
            <label className="flex items-center justify-between text-sm font-medium text-gray-300 mb-1">
              <span>Temperature</span>
              <span className="text-gray-400 font-mono">{draft.temperature.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={0}
              max={2}
              step={0.05}
              value={draft.temperature}
              onChange={field("temperature")}
              className="w-full accent-blue-500"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-0.5">
              <span>Deterministic</span>
              <span>Creative</span>
            </div>
          </div>

          {/* Top-p */}
          <div>
            <label className="flex items-center justify-between text-sm font-medium text-gray-300 mb-1">
              <span>Top-p</span>
              <span className="text-gray-400 font-mono">{draft.topP.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={draft.topP}
              onChange={field("topP")}
              className="w-full accent-blue-500"
            />
          </div>

          {/* Max tokens */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Max tokens
            </label>
            <input
              type="number"
              min={64}
              max={32768}
              value={draft.maxTokens}
              onChange={field("maxTokens")}
              className="w-32 bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Show thinking */}
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-300">Show thinking by default</p>
              <p className="text-xs text-gray-500 mt-0.5">
                Display thinking blocks in assistant responses.
              </p>
            </div>
            <button
              type="button"
              onClick={() => toggle("showThink")}
              className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors focus:outline-none ${
                draft.showThink ? "bg-blue-600" : "bg-gray-600"
              }`}
            >
              <span
                className={`inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform ${
                  draft.showThink ? "translate-x-5" : "translate-x-0"
                }`}
              />
            </button>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 px-6 py-4 border-t border-gray-700">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-gray-300 hover:text-white transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={() => { onSave(draft); onClose(); }}
            className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-colors"
          >
            Save changes
          </button>
        </div>
      </div>
    </div>
  );
}
