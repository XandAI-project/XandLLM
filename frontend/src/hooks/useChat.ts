import { useState, useRef, useCallback } from "react";
import { streamChat } from "../api";
import type { Settings } from "../types";

interface UseChatOptions {
  settings: Settings;
  model: string | null;
  conversationId: string | null;
  getApiMessages: (
    conversationId: string,
    systemPrompt: string
  ) => { role: string; content: string }[];
  appendMessage: (
    conversationId: string,
    msg: {
      id: string;
      role: "user" | "assistant";
      content: string;
      createdAt: number;
      streaming?: boolean;
    }
  ) => void;
  updateLastAssistantMessage: (
    conversationId: string,
    content: string,
    streaming: boolean
  ) => void;
}

function makeId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export function useChat({
  settings,
  model,
  conversationId,
  getApiMessages,
  appendMessage,
  updateLastAssistantMessage,
}: UseChatOptions) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const abort = useCallback(() => {
    abortRef.current?.abort();
    setIsStreaming(false);
  }, []);

  const send = useCallback(
    async (userText: string) => {
      if (!conversationId || !model || isStreaming) return;

      setError(null);

      // Read history BEFORE appending new messages so we get a clean,
      // up-to-date snapshot of prior turns (no race with setState).
      const history = getApiMessages(conversationId, settings.systemPrompt);
      const fullMessages: { role: "system" | "user" | "assistant"; content: string }[] = [
        ...history.map((m) => ({ role: m.role as "system" | "user" | "assistant", content: m.content })),
        { role: "user", content: userText },
      ];

      // Append user message to UI after history is captured
      appendMessage(conversationId, {
        id: makeId(),
        role: "user",
        content: userText,
        createdAt: Date.now(),
      });

      // Placeholder assistant message (streaming)
      const assistantId = makeId();
      appendMessage(conversationId, {
        id: assistantId,
        role: "assistant",
        content: "",
        createdAt: Date.now(),
        streaming: true,
      });

      const controller = new AbortController();
      abortRef.current = controller;
      setIsStreaming(true);

      let accumulated = "";

      try {
        for await (const delta of streamChat(
          fullMessages,
          settings,
          model,
          controller.signal
        )) {
          accumulated += delta;
          updateLastAssistantMessage(conversationId, accumulated, true);
        }
        updateLastAssistantMessage(conversationId, accumulated, false);
      } catch (err) {
        if ((err as Error).name === "AbortError") {
          updateLastAssistantMessage(conversationId, accumulated, false);
        } else {
          const msg = err instanceof Error ? err.message : String(err);
          setError(msg);
          updateLastAssistantMessage(
            conversationId,
            accumulated || `Error: ${msg}`,
            false
          );
        }
      } finally {
        setIsStreaming(false);
        abortRef.current = null;
      }
    },
    [
      conversationId,
      model,
      isStreaming,
      settings,
      appendMessage,
      updateLastAssistantMessage,
      getApiMessages,
    ]
  );

  return { send, abort, isStreaming, error };
}
