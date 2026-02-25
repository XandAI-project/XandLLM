import { useState, useCallback } from "react";
import type { Conversation, Message, ChatMessage } from "../types";

const STORAGE_KEY = "xandllm_conversations";

/** Remove known model end-of-turn tokens that should never appear in stored content. */
function stripStopTags(text: string): string {
  return text
    .replace(/<\|im_end\|>/g, "")
    .replace(/<\|endoftext\|>/g, "")
    .replace(/<\|eot_id\|>/g, "")
    .replace(/<\|end_of_text\|>/g, "")
    .replace(/<\/s>/g, "")
    .trimEnd();
}

function load(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Conversation[]) : [];
  } catch {
    return [];
  }
}

function save(convos: Conversation[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(convos));
}

function makeId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function titleFromMessage(content: string): string {
  return content.length > 50 ? content.slice(0, 47) + "…" : content;
}

export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>(load);
  const [activeId, setActiveId] = useState<string | null>(
    () => load()[0]?.id ?? null
  );

  const active = conversations.find((c) => c.id === activeId) ?? null;

  const persist = useCallback((next: Conversation[]) => {
    setConversations(next);
    save(next);
  }, []);

  const newConversation = useCallback((): string => {
    const id = makeId();
    const convo: Conversation = {
      id,
      title: "New conversation",
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    const next = [convo, ...conversations];
    persist(next);
    setActiveId(id);
    return id;
  }, [conversations, persist]);

  const selectConversation = useCallback((id: string) => {
    setActiveId(id);
  }, []);

  const deleteConversation = useCallback(
    (id: string) => {
      const next = conversations.filter((c) => c.id !== id);
      persist(next);
      if (activeId === id) {
        setActiveId(next[0]?.id ?? null);
      }
    },
    [conversations, persist, activeId]
  );

  const clearAll = useCallback(() => {
    persist([]);
    setActiveId(null);
  }, [persist]);

  const appendMessage = useCallback(
    (conversationId: string, message: Message) => {
      setConversations((prev) => {
        const next = prev.map((c) => {
          if (c.id !== conversationId) return c;
          const messages = [...c.messages, message];
          const title =
            c.messages.length === 0 && message.role === "user"
              ? titleFromMessage(message.content)
              : c.title;
          return { ...c, messages, title, updatedAt: Date.now() };
        });
        save(next);
        return next;
      });
    },
    []
  );

  const updateLastAssistantMessage = useCallback(
    (conversationId: string, content: string, streaming: boolean) => {
      setConversations((prev) => {
        const next = prev.map((c) => {
          if (c.id !== conversationId) return c;
          const messages = [...c.messages];
          let idx = -1;
          for (let i = messages.length - 1; i >= 0; i--) {
            if (messages[i].role === "assistant") { idx = i; break; }
          }
          if (idx === -1) return c;
          messages[idx] = { ...messages[idx], content, streaming };
          return { ...c, messages, updatedAt: Date.now() };
        });
        if (!streaming) save(next);
        return next;
      });
    },
    []
  );

  const getApiMessages = useCallback(
    (conversationId: string, systemPrompt: string): ChatMessage[] => {
      const convo = conversations.find((c) => c.id === conversationId);
      if (!convo) return [];
      const history: ChatMessage[] = convo.messages
        .filter((m) => !m.streaming)
        .map((m) => ({
          role: m.role,
          // Strip any stop tokens that may have leaked into stored content.
          // This prevents stale <|im_end|> etc. from corrupting future prompts.
          content: stripStopTags(m.content),
        }))
        // Discard empty assistant messages (e.g. aborted turns)
        .filter((m) => m.role !== "assistant" || m.content.trim().length > 0);
      if (systemPrompt) {
        return [{ role: "system", content: systemPrompt }, ...history];
      }
      return history;
    },
    [conversations]
  );

  return {
    conversations,
    active,
    activeId,
    newConversation,
    selectConversation,
    deleteConversation,
    clearAll,
    appendMessage,
    updateLastAssistantMessage,
    getApiMessages,
  };
}
