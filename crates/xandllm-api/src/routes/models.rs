use axum::{Extension, Json, response::IntoResponse};
use chrono::Utc;
use serde_json::json;

use crate::{server::ModelId, types::{ModelList, ModelObject}};

/// `GET /v1/models` — list models currently loaded in the server.
pub async fn list_models(
    Extension(ModelId(model_id)): Extension<ModelId>,
) -> Json<ModelList> {
    let now = Utc::now().timestamp();
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: model_id.as_ref().clone(),
            object: "model".to_string(),
            created: now,
            owned_by: "xandllm".to_string(),
        }],
    })
}

/// `GET /health` — liveness probe.
pub async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}
