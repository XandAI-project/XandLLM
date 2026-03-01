use axum::{
    Extension, Json,
    response::{IntoResponse, Sse, sse::Event},
};
use chrono::Utc;
use futures::stream::{self, StreamExt};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{info, instrument};
use uuid::Uuid;

use xandllm_core::{GenerateInput, Model, SamplingParams};

use crate::{
    error::{ApiError, ApiResult},
    server::{ChatFormat, ModelId},
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
    let base_system = messages
        .iter()
        .find(|m| m.role == "system")
        .map(|m| m.content.as_str())
        .unwrap_or("You are a helpful assistant.");

    let non_system: Vec<&crate::types::ChatMessage> =
        messages.iter().filter(|m| m.role != "system").collect();

    // Build alternating (user, assistant) history pairs.
    let mut history: Vec<(String, String)> = Vec::new();
    let mut i = 0;

    while i < non_system.len() {
        let msg = non_system[i];

        if msg.role == "user" {
            if i + 1 < non_system.len() && non_system[i + 1].role == "assistant" {
                let asst = non_system[i + 1];
                history.push((msg.content.clone(), asst.content.clone()));
                i += 2;
            } else if i + 1 < non_system.len() && non_system[i + 1].role == "user" {
                // Consecutive user messages (assistant response is missing).
                // Skip the first user message and continue to process the second.
                i += 1;
                continue;
            } else {
                // This is the final, unanswered user turn — stop here.
                break;
            }
        } else {
            i += 1;
        }
    }

    let user_msg = non_system
        .get(i)
        .filter(|m| m.role == "user")
        .map(|m| m.content.as_str())
        .unwrap_or("");

    xandllm_core::chat_template::build_chat_prompt(fmt, base_system, &history, user_msg)
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
    Extension(ModelId(model_id)): Extension<ModelId>,
    Extension(tokenizer): Extension<Arc<xandllm_core::Tokenizer>>,
    Extension(ChatFormat(chat_format)): Extension<ChatFormat>,
    Json(req): Json<ChatCompletionRequest>,
) -> ApiResult<impl IntoResponse> {
    if req.messages.is_empty() {
        return Err(ApiError::BadRequest(
            "messages array must not be empty".to_string(),
        ));
    }

    // Build prompt using the model's actual chat template (not hardcoded ChatML)
    let prompt = messages_to_prompt(&chat_format, &req.messages);

    info!(
        chat_format = %chat_format,
        prompt_len = prompt.len(),
        prompt = %prompt,
        "Formatted prompt sent to LLM"
    );

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
    
    // Collect format-specific multi-token text stop strings.
    // These catch degenerate role-reversal loops (e.g. "\nUser:") that cannot
    // be expressed as a single stop token ID.
    let stop_strings: Vec<String> = xandllm_core::chat_template::stop_text_strings_for_format(&chat_format)
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Thinking models need their text stop strings suppressed until </think>
    // is seen — otherwise the reasoning content (which often contains phrases
    // like "\nAssistant:") triggers a false-positive early stop.
    let thinking_mode = chat_format.as_str() == "chatml-thinking";

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
        stop_strings,
        thinking_mode,
    };

    // add_special_tokens=false: the chat template already embeds all special
    // tokens (BOS, role markers, etc.) as literal text that the tokenizer
    // recognises as single token IDs.  Using true would cause a double-BOS
    // for any model whose tokenizer.json post-processor also prepends BOS
    // (e.g. Gemma models loaded with an external tokenizer.json).
    let token_ids = tokenizer
        .encode(&prompt, false)
        .map_err(ApiError::Model)?;
    let prompt_tokens = token_ids.len();

    info!(
        prompt_tokens,
        stop_token_ids = ?params.stop_token_ids,
        stop_strings = ?params.stop_strings,
        "Prompt encoded"
    );

    let input = GenerateInput { token_ids };

    if req.stream {
        let id = format!("chatcmpl-{}", Uuid::new_v4());
        let model_name = req.model.clone();

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let model_clone = model.clone();
        // Fire inference in a blocking thread — do NOT await.
        // Tokens flow into `rx` as they are produced, enabling true per-token streaming.
        tokio::task::spawn_blocking(move || {
            // Catch any panics to prevent the server from crashing
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut guard = model_clone.blocking_lock();
                if let Err(e) = guard.generate_stream(input, params, tx) {
                    tracing::error!(error = %e, "Generation failed");
                }
            }));
            if let Err(e) = result {
                tracing::error!(error = ?e, "Generation thread panicked");
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
                    if token.is_eos || token.text.is_empty() {
                        return std::future::ready(None);
                    }
                    let delta = if first {
                        first = false;
                        let content = if thinking_mode {
                            format!("<think>{}", token.text)
                        } else {
                            token.text
                        };
                        ChatDelta {
                            role: Some("assistant".to_string()),
                            content: Some(content),
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
        let model_clone = model.clone();
        let mut rx = tokio::task::spawn_blocking(move || {
            // Catch any panics to prevent server crashes
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut guard = model_clone.blocking_lock();
                guard.generate(input, params)
            }));
            match result {
                Ok(generation_result) => generation_result.map_err(ApiError::Model),
                Err(e) => {
                    tracing::error!(error = ?e, "Non-streaming generation panicked");
                    Err(ApiError::Internal("Generation thread panicked".to_string()))
                }
            }
        })
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))
        .and_then(|inner| inner)?;
        let mut tokens = Vec::new();
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
        // For thinking models, prepend <think> so clients can detect the reasoning block.
        let content: String = if thinking_mode {
            format!("<think>{}", tokens.concat())
        } else {
            tokens.concat()
        };

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
