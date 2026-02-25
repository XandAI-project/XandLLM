use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

use xandllm_core::CoreError;

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Model error: {0}")]
    Model(#[from] CoreError),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Request timeout")]
    Timeout,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_type, message) = match &self {
            ApiError::BadRequest(msg) => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                msg.clone(),
            ),
            ApiError::Model(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "model_error",
                e.to_string(),
            ),
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, "not_found_error", msg.clone()),
            ApiError::Internal(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_server_error",
                msg.clone(),
            ),
            ApiError::Timeout => (
                StatusCode::REQUEST_TIMEOUT,
                "timeout_error",
                "Request timed out".to_string(),
            ),
        };

        let body = json!({
            "error": {
                "message": message,
                "type": error_type,
                "code": status.as_u16()
            }
        });

        (status, Json(body)).into_response()
    }
}

pub type ApiResult<T> = Result<T, ApiError>;

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use http_body_util::BodyExt;

    async fn body_string(resp: axum::response::Response) -> String {
        let bytes = resp
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn test_bad_request_yields_400() {
        let err = ApiError::BadRequest("missing field".to_string());
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_string(resp).await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["message"], "missing field");
    }

    #[tokio::test]
    async fn test_not_found_yields_404() {
        let err = ApiError::NotFound("model xyz not found".to_string());
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = body_string(resp).await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["error"]["code"], 404);
    }

    #[tokio::test]
    async fn test_internal_yields_500() {
        let err = ApiError::Internal("something exploded".to_string());
        assert_eq!(err.into_response().status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn test_timeout_yields_408() {
        let err = ApiError::Timeout;
        assert_eq!(err.into_response().status(), StatusCode::REQUEST_TIMEOUT);
    }

    #[tokio::test]
    async fn test_error_body_has_required_keys() {
        let err = ApiError::BadRequest("x".to_string());
        let resp = err.into_response();
        let body = body_string(resp).await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        // OpenAI schema requires error.message, error.type, error.code
        assert!(v["error"]["message"].is_string());
        assert!(v["error"]["type"].is_string());
        assert!(v["error"]["code"].is_number());
    }
}
