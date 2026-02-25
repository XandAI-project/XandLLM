use axum::{
    Extension, Json,
    response::{IntoResponse, Sse, sse::Event},
};
use chrono::Utc;
use futures::stream::{self, StreamExt};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::instrument;
use uuid::Uuid;

use xandllm_core::{GenerateInput, Model, SamplingParams};

use crate::{
    error::{ApiError, ApiResult},
    streaming::{sse_done, sse_event},
    types::{
        ChatChoice, ChatCompletionChunk, ChatCompletionMessage, ChatCompletionRequest,
        ChatCompletionResponse, ChatDelta, StreamingChatChoice, Usage,
    },
};

/// Build a prompt string from OpenAI-format messages using the model's
/// actual chat template (`chatml`, `llama2`, `llama3`, etc.).
///
/// Extracts system/history/user from the messages array and delegates to
/// `chat_template::build_chat_prompt`, which ensures correct formatting
/// regardless of architecture.
fn messages_to_prompt(fmt: &str, messages: &[crate::types::ChatMessage]) -> String {
    let system = messages
        .iter()
        .find(|m| m.role == "system")
        .map(|m| m.content.as_str())
        .unwrap_or("You are a helpful assistant.");

    let non_system: Vec<&crate::types::ChatMessage> =
        messages.iter().filter(|m| m.role != "system").collect();

    // Build alternating (user, assistant) history, excluding the last user turn.
    let mut history: Vec<(String, String)> = Vec::new();
    let mut i = 0;
    while i + 1 < non_system.len() {
        if non_system[i].role == "user" && non_system[i + 1].role == "assistant" {
            history.push((
                non_system[i].content.clone(),
                non_system[i + 1].content.clone(),
            ));
            i += 2;
        } else {
            i += 1;
        }
    }

    let user_msg = non_system
        .last()
        .filter(|m| m.role == "user")
        .map(|m| m.content.as_str())
        .unwrap_or("");

    xandllm_core::chat_template::build_chat_prompt(fmt, system, &history, user_msg)
}

/// Parse the `stop` field from the API request and convert stop strings to token IDs.
///
/// Handles three cases:
/// - `null` or missing → empty vec
/// - String → single stop sequence
/// - Array of strings → multiple stop sequences
fn parse_stop_sequences(
    stop: &Option<serde_json::Value>,
    tokenizer: &xandllm_core::Tokenizer,
) -> ApiResult<Vec<u32>> {
    let Some(stop_val) = stop else {
        return Ok(vec![]);
    };

    let stop_strings: Vec<String> = match stop_val {
        serde_json::Value::String(s) => vec![s.clone()],
        serde_json::Value::Array(arr) => {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        }
        _ => {
            return Err(ApiError::BadRequest(
                "stop must be a string or array of strings".to_string(),
            ))
        }
    };

    let mut token_ids = Vec::new();
    for stop_str in stop_strings {
        let encoded = tokenizer.encode(&stop_str, false).map_err(ApiError::Model)?;
        if encoded.len() == 1 {
            token_ids.push(encoded[0]);
        }
    }

    Ok(token_ids)
}

/// `POST /v1/chat/completions`
#[instrument(skip_all, fields(model = %req.model, stream = req.stream))]
pub async fn create_chat_completion(
    Extension(model): Extension<Arc<Mutex<dyn Model + Send>>>,
    Extension(model_id): Extension<Arc<String>>,
    Extension(tokenizer): Extension<Arc<xandllm_core::Tokenizer>>,
    Extension(chat_format): Extension<Arc<String>>,
    Json(req): Json<ChatCompletionRequest>,
) -> ApiResult<impl IntoResponse> {
    if req.messages.is_empty() {
        return Err(ApiError::BadRequest(
            "messages array must not be empty".to_string(),
        ));
    }

    // Build prompt using the model's actual chat template (not hardcoded ChatML)
    let prompt = messages_to_prompt(&chat_format, &req.messages);

    // Collect format-specific stop tokens (e.g. <|im_end|> for ChatML, <|eot_id|> for LLaMA-3)
    let mut stop_token_ids: Vec<u32> = xandllm_core::chat_template::stop_token_strings_for_format(&chat_format)
        .iter()
        .filter_map(|s| tokenizer.token_id(s))
        .collect();

    // Also add the tokenizer's native EOS token as a stop condition
    if let Some(eos) = tokenizer.eos_token_id() {
        if !stop_token_ids.contains(&eos) {
            stop_token_ids.push(eos);
        }
    }

    // Merge user-provided stop sequences from the API request
    let user_stop_ids = parse_stop_sequences(&req.stop, &tokenizer)?;
    stop_token_ids.extend(user_stop_ids);
    
    let params = SamplingParams {
        max_new_tokens: req.max_tokens.unwrap_or(512),
        temperature: req.temperature.unwrap_or(0.7),
        top_p: req.top_p.unwrap_or(0.9),
        top_k: req.top_k,
        repetition_penalty: req.repetition_penalty.unwrap_or(1.0),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        seed: req.seed,
        greedy: false,
        stop_token_ids,
        repeat_last_n: Some(64),
    };

    let token_ids = tokenizer
        .encode(&prompt, true)
        .map_err(ApiError::Model)?;
    let prompt_tokens = token_ids.len();
    let input = GenerateInput { token_ids };

    if req.stream {
        let id = format!("chatcmpl-{}", Uuid::new_v4());
        let model_name = req.model.clone();

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let model_clone = model.clone();
        // Fire inference in a blocking thread — do NOT await.
        // Tokens flow into `rx` as they are produced, enabling true per-token streaming.
        tokio::task::spawn_blocking(move || {
            let mut guard = model_clone.blocking_lock();
            if let Err(e) = guard.generate_stream(input, params, tx) {
                tracing::error!(error = %e, "Generation failed");
            }
            // tx is dropped here → channel closes → SSE stream ends
        });

        let receiver_stream = UnboundedReceiverStream::new(rx);
        let id_clone = id.clone();
        let model_name_clone = model_name.clone();
        let mut first = true;

        let event_stream = receiver_stream.filter_map(move |result| {
            let event = match result {
                Ok(token) => {
                    // Never emit stop-tag text — the core should never send EOS tokens,
                    // but guard defensively here as well.
                    if token.is_eos || token.text.is_empty() {
                        return std::future::ready(None);
                    }
                    let delta = if first {
                        first = false;
                        ChatDelta {
                            role: Some("assistant".to_string()),
                            content: Some(token.text),
                        }
                    } else {
                        ChatDelta {
                            role: None,
                            content: Some(token.text),
                        }
                    };
                    let chunk = ChatCompletionChunk {
                        id: id_clone.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created: Utc::now().timestamp(),
                        model: model_name_clone.clone(),
                        choices: vec![StreamingChatChoice {
                            index: 0,
                            delta,
                            finish_reason: None,
                        }],
                    };
                    sse_event(&chunk).map_err(|e| {
                        tracing::error!(error = %e, "SSE serialization error");
                        e
                    })
                }
                Err(e) => {
                    tracing::error!(error = %e, "Generation error during streaming");
                    Err(serde_json::from_str::<serde_json::Value>("").unwrap_err())
                }
            };
            std::future::ready(Some(event))
        });

        let done_stream = stream::once(async { Ok::<Event, serde_json::Error>(sse_done()) });
        let full_stream = event_stream.chain(done_stream);

        Ok(Sse::new(full_stream).into_response())
    } else {
        let rx = {
            let model = model.clone();
            tokio::task::spawn_blocking(move || {
                let mut guard = model.blocking_lock();
                guard.generate(input, params)
            })
            .await
            .map_err(|e| ApiError::Internal(e.to_string()))?
            .map_err(ApiError::Model)?
        };
        let mut tokens = Vec::new();
        let mut rx = rx;
        while let Some(result) = rx.recv().await {
            let token = result.map_err(ApiError::Model)?;
            // Never include stop-tag text in the final content
            if token.is_eos {
                break;
            }
            if !token.text.is_empty() {
                tokens.push(token.text);
            }
        }
        let content: String = tokens.concat();
        let completion_tokens = content.split_whitespace().count();

        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: Utc::now().timestamp(),
            model: model_id.as_ref().clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        };
        Ok(Json(response).into_response())
    }
}
