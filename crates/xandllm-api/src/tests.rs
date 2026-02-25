/// Integration tests for xandllm-api routes.
///
/// Uses a [`MockModel`] so no actual model weights are required.
/// All tests exercise the full axum router stack: parsing → handler → response.
#[cfg(test)]
pub(crate) mod tests {
    use std::{path::Path, sync::Arc};

    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
        response::Response,
    };
    use candle_core::Device;
    use http_body_util::BodyExt;
    use serde_json::Value;
    use tokio::sync::{mpsc, Mutex};
    use tower::ServiceExt;

    use xandllm_core::{CoreResult, GenerateInput, Model, ModelConfig, SamplingParams, Token, Tokenizer};

    use crate::server::build_router;

    // ── MockModel ─────────────────────────────────────────────────────────────

    /// A minimal model that emits two tokens then closes the channel.
    ///
    /// Mirrors the fixed core behaviour: stop tokens are NOT sent into the
    /// channel at all (the core breaks before calling `tx.send`).  The `is_eos`
    /// field is kept for the defensive API-layer checks.
    struct MockModel;

    impl Model for MockModel {
        fn load(_: &ModelConfig, _: &Device) -> CoreResult<Self>
        where
            Self: Sized,
        {
            Ok(Self)
        }

        fn generate(
            &mut self,
            _input: GenerateInput,
            _params: SamplingParams,
        ) -> CoreResult<mpsc::UnboundedReceiver<CoreResult<Token>>> {
            let (tx, rx) = mpsc::unbounded_channel();
            // Core never sends stop-token text — only content tokens with is_eos: false
            let _ = tx.send(Ok(Token { id: 2, text: "hello".to_string(), is_eos: false }));
            // Simulate the defensive case: is_eos token whose text must be suppressed
            let _ = tx.send(Ok(Token { id: 1, text: "<|im_end|>".to_string(), is_eos: true }));
            Ok(rx)
        }

        fn generate_stream(
            &mut self,
            _input: GenerateInput,
            _params: SamplingParams,
            tx: mpsc::UnboundedSender<CoreResult<Token>>,
        ) -> CoreResult<()> {
            let _ = tx.send(Ok(Token { id: 2, text: "hello".to_string(), is_eos: false }));
            // Same: is_eos stop token must not reach client
            let _ = tx.send(Ok(Token { id: 1, text: "<|im_end|>".to_string(), is_eos: true }));
            Ok(())
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn stub_tokenizer() -> Arc<Tokenizer> {
        // Path relative to workspace root for the stub tokenizer used in tests
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test-data/stub_tokenizer.json");
        Arc::new(
            Tokenizer::from_file(&path)
                .expect("test-data/stub_tokenizer.json must exist"),
        )
    }

    fn test_router() -> axum::Router<()> {
        let model: Arc<Mutex<dyn Model + Send>> =
            Arc::new(Mutex::new(MockModel));
        let model_id = Arc::new("mock-llama-3".to_string());
        build_router(model, stub_tokenizer(), model_id, "chatml".to_string(), 30)
    }

    async fn call(router: axum::Router<()>, req: Request<Body>) -> Response {
        router.oneshot(req).await.unwrap()
    }

    async fn body_json(resp: Response) -> Value {
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    }

    fn get(uri: &str) -> Request<Body> {
        Request::builder().uri(uri).body(Body::empty()).unwrap()
    }

    fn post_json(uri: &str, payload: &str) -> Request<Body> {
        Request::builder()
            .method(Method::POST)
            .uri(uri)
            .header("Content-Type", "application/json")
            .body(Body::from(payload.to_owned()))
            .unwrap()
    }

    // ── GET /health ───────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_health_returns_200() {
        let resp = call(test_router(), get("/health")).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_body_status_ok() {
        let resp = call(test_router(), get("/health")).await;
        let v = body_json(resp).await;
        assert_eq!(v["status"], "ok");
    }

    // ── GET /v1/models ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_models_returns_200() {
        let resp = call(test_router(), get("/v1/models")).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_models_object_is_list() {
        let resp = call(test_router(), get("/v1/models")).await;
        let v = body_json(resp).await;
        assert_eq!(v["object"], "list");
        assert!(v["data"].is_array());
    }

    #[tokio::test]
    async fn test_models_contains_model_id() {
        let resp = call(test_router(), get("/v1/models")).await;
        let v = body_json(resp).await;
        assert_eq!(v["data"][0]["id"], "mock-llama-3");
    }

    // ── Unknown route ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_unknown_route_is_404() {
        let resp = call(test_router(), get("/no/such/route")).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ── POST /v1/chat/completions ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_chat_bad_json_is_client_error() {
        let resp = call(test_router(), post_json("/v1/chat/completions", "{not json}")).await;
        assert!(resp.status().is_client_error(), "got {}", resp.status());
    }

    #[tokio::test]
    async fn test_chat_empty_messages_is_400() {
        let payload = r#"{"model":"m","messages":[]}"#;
        let resp = call(test_router(), post_json("/v1/chat/completions", payload)).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_chat_non_streaming_returns_completion() {
        let payload = r#"{
            "model":"mock-llama-3",
            "messages":[{"role":"user","content":"hello"}],
            "stream":false,
            "max_tokens":5
        }"#;
        let resp = call(test_router(), post_json("/v1/chat/completions", payload)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["object"], "chat.completion");
        assert!(v["choices"].is_array() && !v["choices"].as_array().unwrap().is_empty());
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    }

    #[tokio::test]
    async fn test_chat_response_has_usage() {
        let payload = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":false}"#;
        let resp = call(test_router(), post_json("/v1/chat/completions", payload)).await;
        let v = body_json(resp).await;
        assert!(v["usage"]["prompt_tokens"].is_number());
        assert!(v["usage"]["completion_tokens"].is_number());
        assert!(v["usage"]["total_tokens"].is_number());
    }

    // ── POST /v1/completions ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_completion_non_streaming() {
        let payload = r#"{"model":"m","prompt":"hello world","stream":false,"max_tokens":5}"#;
        let resp = call(test_router(), post_json("/v1/completions", payload)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["object"], "text_completion");
        assert!(v["choices"].is_array());
    }

    #[tokio::test]
    async fn test_completion_has_usage() {
        let payload = r#"{"model":"m","prompt":"test","stream":false}"#;
        let resp = call(test_router(), post_json("/v1/completions", payload)).await;
        let v = body_json(resp).await;
        assert!(v["usage"]["total_tokens"].is_number());
    }

    // ── Stop-tag absence regression tests ─────────────────────────────────────

    /// Non-streaming chat must not include stop-tag text even when the mock
    /// model emits an `is_eos` token with stop-tag content.
    #[tokio::test]
    async fn test_chat_non_streaming_no_stop_tags() {
        let payload = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":false}"#;
        let resp = call(test_router(), post_json("/v1/chat/completions", payload)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        let content = v["choices"][0]["message"]["content"].as_str().unwrap_or("");
        assert!(
            !content.contains("<|im_end|>"),
            "Response must not contain <|im_end|>, got: {content}"
        );
        assert!(
            !content.contains("</s>"),
            "Response must not contain </s>, got: {content}"
        );
        assert!(
            !content.contains("<|eot_id|>"),
            "Response must not contain <|eot_id|>, got: {content}"
        );
        // Content should have the actual model output
        assert_eq!(content, "hello");
    }

    /// Non-streaming completion must not include stop-tag text.
    #[tokio::test]
    async fn test_completion_non_streaming_no_stop_tags() {
        let payload = r#"{"model":"m","prompt":"test","stream":false}"#;
        let resp = call(test_router(), post_json("/v1/completions", payload)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        let text = v["choices"][0]["text"].as_str().unwrap_or("");
        assert!(
            !text.contains("<|im_end|>"),
            "Completion must not contain <|im_end|>, got: {text}"
        );
        assert_eq!(text, "hello");
    }

    /// Multi-turn payload must be accepted without error (memory smoke test).
    #[tokio::test]
    async fn test_chat_multi_turn_accepted() {
        let payload = r#"{
            "model": "m",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "what did I say before?"}
            ],
            "stream": false
        }"#;
        let resp = call(test_router(), post_json("/v1/chat/completions", payload)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["object"], "chat.completion");
    }
}
