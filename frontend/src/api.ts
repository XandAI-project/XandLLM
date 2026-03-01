import type {
  ChatMessage,
  ChatCompletionChunk,
  ModelList,
  Settings,
} from "./types";

// ── Health ────────────────────────────────────────────────────────────────────

export async function fetchHealth(baseUrl: string): Promise<boolean> {
  try {
    const res = await fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}

// ── Models ────────────────────────────────────────────────────────────────────

export async function fetchModels(baseUrl: string): Promise<ModelList | null> {
  try {
    const res = await fetch(`${baseUrl}/v1/models`, { signal: AbortSignal.timeout(3000) });
    if (!res.ok) return null;
    return res.json() as Promise<ModelList>;
  } catch {
    return null;
  }
}

// ── Streaming chat completions ────────────────────────────────────────────────

/**
 * Sends a chat request with `stream: true` and yields text delta strings as
 * they arrive via SSE.  Stops when `[DONE]` is received or the signal fires.
 */
export async function* streamChat(
  messages: ChatMessage[],
  settings: Settings,
  model: string,
  signal: AbortSignal
): AsyncGenerator<string> {
  const body = JSON.stringify({
    model,
    messages,
    stream: true,
    max_tokens: settings.maxTokens,
    temperature: settings.temperature,
    top_p: settings.topP,
  });

  const res = await fetch(`${settings.baseUrl}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    signal,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API error ${res.status}: ${text}`);
  }

  if (!res.body) throw new Error("No response body");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

        for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const data = trimmed.slice(5).trim();
        if (data === "[DONE]") return;

        try {
          const chunk = JSON.parse(data) as ChatCompletionChunk;
          const choice = chunk.choices[0];
          if (!choice) continue;
          // Yield content BEFORE checking finish_reason — the final chunk may
          // contain the last token(s) of the response along with stop signal.
          const raw = choice.delta?.content;
          if (raw) {
            const delta = raw
              .replace(/<\|im_end\|>/g, "")
              .replace(/<\|endoftext\|>/g, "")
              .replace(/<\|eot_id\|>/g, "")
              .replace(/<\|end_of_text\|>/g, "")
              .replace(/<\/s>/g, "")
              .replace(/<\|end\|>/g, "")
              .replace(/<end_of_turn>/g, "");
            if (delta) yield delta;
          }
          if (choice.finish_reason === "stop") return;
        } catch {
          // malformed chunk — skip
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
