// ── Chat messages ─────────────────────────────────────────────────────────────

export type Role = "system" | "user" | "assistant";

export interface ChatMessage {
  role: Role;
  content: string;
}

export interface Message extends ChatMessage {
  id: string;
  createdAt: number;
  /** true while the assistant is still streaming */
  streaming?: boolean;
}

// ── Conversations ─────────────────────────────────────────────────────────────

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

// ── Settings ──────────────────────────────────────────────────────────────────

export interface Settings {
  baseUrl: string;
  systemPrompt: string;
  temperature: number;
  topP: number;
  maxTokens: number;
  showThink: boolean;
}

export const DEFAULT_SETTINGS: Settings = {
  baseUrl: "http://localhost:11435",
  systemPrompt: "You are a helpful assistant.",
  temperature: 0.7,
  topP: 0.9,
  maxTokens: 4096,
  showThink: false,
};

// ── API response types ────────────────────────────────────────────────────────

export interface ModelObject {
  id: string;
  object: string;
  created: number;
  owned_by: string;
}

export interface ModelList {
  object: string;
  data: ModelObject[];
}

export interface ChatCompletionChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: {
    index: number;
    delta: { role?: string; content?: string };
    finish_reason: string | null;
  }[];
}

export interface ServerStatus {
  online: boolean;
  model: string | null;
}
