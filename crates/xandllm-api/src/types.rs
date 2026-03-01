use serde::{Deserialize, Serialize};

// ─── Request types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub seed: Option<u64>,
    pub stop: Option<serde_json::Value>,
    pub n: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub stream: bool,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub seed: Option<u64>,
    pub stop: Option<serde_json::Value>,
    pub n: Option<u32>,
}

// ─── Response types ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

// ── Chat completions ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatCompletionMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

// ── Streaming chat completions (SSE delta) ─────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamingChatChoice {
    pub index: u32,
    pub delta: ChatDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<StreamingChatChoice>,
}

// ── Text completions ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

// ── Streaming text completions ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct StreamingCompletionChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<StreamingCompletionChoice>,
}

// ── Models ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelObject>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{from_str, to_value};

    // ── ChatCompletionRequest ─────────────────────────────────────────────────

    #[test]
    fn test_chat_request_minimal_fields() {
        let json = r#"{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}"#;
        let req: ChatCompletionRequest = from_str(json).unwrap();
        assert_eq!(req.model, "gpt-4");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.messages[0].content, "hi");
        assert!(!req.stream, "stream must default to false");
        assert!(req.max_tokens.is_none());
        assert!(req.temperature.is_none());
    }

    #[test]
    fn test_chat_request_stream_true() {
        let json = r#"{"model":"m","messages":[],"stream":true}"#;
        let req: ChatCompletionRequest = from_str(json).unwrap();
        assert!(req.stream);
    }

    #[test]
    fn test_chat_request_optional_params() {
        let json = r#"{"model":"m","messages":[],"max_tokens":100,"temperature":0.5,"top_p":0.8}"#;
        let req: ChatCompletionRequest = from_str(json).unwrap();
        assert_eq!(req.max_tokens, Some(100));
        assert!((req.temperature.unwrap() - 0.5).abs() < f64::EPSILON);
        assert!((req.top_p.unwrap() - 0.8).abs() < f64::EPSILON);
    }

    // ── CompletionRequest ─────────────────────────────────────────────────────

    #[test]
    fn test_completion_request_stream_defaults_false() {
        let json = r#"{"model":"test","prompt":"hello world"}"#;
        let req: CompletionRequest = from_str(json).unwrap();
        assert_eq!(req.prompt, "hello world");
        assert!(!req.stream);
    }

    #[test]
    fn test_completion_request_with_stop() {
        let json = r#"{"model":"m","prompt":"p","stop":"\n"}"#;
        let req: CompletionRequest = from_str(json).unwrap();
        assert!(req.stop.is_some());
    }

    // ── Response serialization ────────────────────────────────────────────────

    #[test]
    fn test_chat_completion_response_serializes_correctly() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1_700_000_000,
            model: "llama3".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: "Hello!".to_string(),
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 3,
                total_tokens: 8,
            },
        };
        let v = to_value(&resp).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["total_tokens"], 8);
    }

    #[test]
    fn test_streaming_delta_skips_none_fields() {
        let delta = ChatDelta { role: None, content: Some("hi".to_string()) };
        let v = to_value(&delta).unwrap();
        assert!(v.get("role").is_none(), "role=None must be omitted from JSON");
        assert_eq!(v["content"], "hi");
    }

    #[test]
    fn test_model_list_serialization() {
        let list = ModelList {
            object: "list".to_string(),
            data: vec![ModelObject {
                id: "llama3".to_string(),
                object: "model".to_string(),
                created: 0,
                owned_by: "xandllm".to_string(),
            }],
        };
        let v = to_value(&list).unwrap();
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"][0]["id"], "llama3");
    }
}
