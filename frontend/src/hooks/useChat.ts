import { useState, useRef, useCallback } from "react";
import { streamChat } from "../api";
import type { GenerationStats, Message, Settings, ChatMessage } from "../types";

interface UseChatOptions {
  settings: Settings;
  model: string | null;
  conversationId: string | null;
  getApiMessages: (
    conversationId: string,
    systemPrompt: string
  ) => ChatMessage[];
  appendMessage: (conversationId: string, msg: Message) => void;
  updateLastAssistantMessage: (
    conversationId: string,
    content: string,
    streaming: boolean,
    stats?: GenerationStats
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
      const fullMessages: ChatMessage[] = [
        ...history,
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
      let tokenCount = 0;
      let startMs = 0;

      try {
        for await (const delta of streamChat(
          fullMessages,
          settings,
          model,
          controller.signal
        )) {
          if (tokenCount === 0) startMs = Date.now();
          tokenCount++;
          accumulated += delta;
          updateLastAssistantMessage(conversationId, accumulated, true);
        }

        const elapsedSec = tokenCount > 0 ? (Date.now() - startMs) / 1000 : 0;
        const stats: GenerationStats = {
          tokens: tokenCount,
          tokensPerSec: elapsedSec > 0 ? tokenCount / elapsedSec : 0,
        };

        updateLastAssistantMessage(conversationId, accumulated, false, stats);
      } catch (err) {
        if ((err as Error).name === "AbortError") {
          // Show partial stats even on abort
          const elapsedSec = tokenCount > 0 ? (Date.now() - startMs) / 1000 : 0;
          const stats: GenerationStats | undefined = tokenCount > 0
            ? { tokens: tokenCount, tokensPerSec: elapsedSec > 0 ? tokenCount / elapsedSec : 0 }
            : undefined;
          updateLastAssistantMessage(conversationId, accumulated, false, stats);
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
