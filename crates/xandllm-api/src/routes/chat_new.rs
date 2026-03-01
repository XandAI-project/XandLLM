        let receiver_stream = UnboundedReceiverStream::new(rx);
        let id_clone = id.clone();
        let model_name_clone = model_name.clone();
        let mut first = true;
        let detector = std::sync::Arc::new(std::sync::Mutex::new(StreamingToolDetector::new()));
        let detector_flush = detector.clone();

        let event_stream = receiver_stream.filter_map(move |result| {
            let mut det = detector.lock().unwrap();
            let event = match result {
                Ok(token) => {
                    if token.is_eos || token.text.is_empty() {
                        return std::future::ready(None);
                    }
                    let detection = det.push_token(&token.text);
                    let (delta_content, tool_calls, finish_reason) = match detection {
                        DetectionState::PassThrough(text) => (Some(text), None, None),
                        DetectionState::Buffering => (None, None, None),
                        DetectionState::ToolCallDetected(calls, cleaned) => {
                            (Some(cleaned), Some(calls), Some("tool_calls".to_string()))
                        }
                    };
                    if delta_content.is_none() && tool_calls.is_none() {
                        return std::future::ready(None);
                    }
                    let delta = if first {
                        first = false;
                        let content = if thinking_mode {
                            format!("{}", delta_content.unwrap_or_default())
                        } else {
                            delta_content.unwrap_or_default()
                        };
                        ChatDelta {
                            role: Some("assistant".to_string()),
                            content: Some(content),
                            tool_calls,
                        }
                    } else {
                        ChatDelta {
                            role: None,
                            content: delta_content,
                            tool_calls,
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
                            finish_reason,
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

        // Flush any remaining buffered text from the detector when stream ends
        let flush_stream = stream::once(async move {
            let mut det = detector_flush.lock().unwrap();
            if let Some(flushed) = det.flush() {
                if !flushed.is_empty() {
                    let delta = ChatDelta {
                        role: None,
                        content: Some(flushed),
                        tool_calls: None,
                    };
                    let chunk = ChatCompletionChunk {
                        id: id_clone.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created: Utc::now().timestamp(),
                        model: model_name_clone.clone(),
                        choices: vec![StreamingChatChoice {
                            index: 0,
                            delta,
                            finish_reason: Some("stop".to_string()),
                        }],
                    };
                    return sse_event(&chunk).ok();
                }
            }
            None
        }).filter_map(|x| async move { x });

        let done_stream = stream::once(async { Ok::<Event, serde_json::Error>(sse_done()) });
        let full_stream = event_stream.chain(flush_stream).chain(done_stream);

        Ok(Sse::new(full_stream).into_response())
