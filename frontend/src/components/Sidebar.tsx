import { Plus, Trash2, MessageSquare, Settings } from "lucide-react";
import clsx from "clsx";
import type { Conversation } from "../types";

interface Props {
  conversations: Conversation[];
  activeId: string | null;
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onOpenSettings: () => void;
}

function formatDate(ts: number): string {
  const d = new Date(ts);
  const now = new Date();
  if (d.toDateString() === now.toDateString()) {
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

export default function Sidebar({
  conversations,
  activeId,
  onNew,
  onSelect,
  onDelete,
  onOpenSettings,
}: Props) {
  return (
    <aside className="w-64 flex-shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-7 h-7 bg-blue-600 rounded-lg flex items-center justify-center">
            <span className="text-white text-xs font-bold">X</span>
          </div>
          <span className="font-semibold text-white">XandLLM</span>
        </div>
        <button
          onClick={onNew}
          className="w-full flex items-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-sm font-medium transition-colors"
        >
          <Plus size={16} />
          New chat
        </button>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto py-2">
        {conversations.length === 0 ? (
          <p className="text-gray-500 text-xs text-center mt-8 px-4">
            No conversations yet. Start a new chat!
          </p>
        ) : (
          conversations.map((c) => (
            <div
              key={c.id}
              className={clsx(
                "group flex items-center gap-2 px-3 py-2.5 mx-2 rounded-lg cursor-pointer transition-colors",
                c.id === activeId
                  ? "bg-gray-700 text-white"
                  : "text-gray-300 hover:bg-gray-800"
              )}
              onClick={() => onSelect(c.id)}
            >
              <MessageSquare size={14} className="flex-shrink-0 text-gray-400" />
              <div className="flex-1 min-w-0">
                <p className="text-sm truncate">{c.title}</p>
                <p className="text-xs text-gray-500">{formatDate(c.updatedAt)}</p>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(c.id);
                }}
                className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-red-400 transition-all flex-shrink-0"
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))
        )}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-gray-800">
        <button
          onClick={onOpenSettings}
          className="w-full flex items-center gap-2 px-3 py-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg text-sm transition-colors"
        >
          <Settings size={15} />
          Settings
        </button>
      </div>
    </aside>
  );
}
