//! `POST /v1/sessions/{id}/responses/{request_id}`: client responses to mid-turn
//! `permission_required` SSE events. The HTTP API models only permission approvals; MCP
//! elicitation auto-declines server-side without reaching the wire (service-to-service
//! callers can't render interactive prompts). With only one outcome category the body has no
//! `kind` discriminator: just `{"outcome": "..."}`.

use axum::{
    Extension,
    body::Bytes,
    extract::{Path, State},
    http::StatusCode,
};
use serde::Deserialize;
use utoipa::ToSchema;
use uuid::Uuid;

use crate::server::{
    auth::Principal,
    errors::{ErrorKind, ProblemDetail},
    http_frontend::PermissionResolution,
    reattach::ensure_session_loaded,
    scope,
    state::ServerState,
};

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ResponseBody {
    pub outcome: PermissionDecision,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum PermissionDecision {
    Allow,
    Deny,
    AllowAlways,
    DenyAlways,
}

#[utoipa::path(
    post,
    path = "/v1/sessions/{id}/responses/{request_id}",
    tag = "turn",
    params(
        ("id" = Uuid, Path, description = "Session UUID"),
        ("request_id" = String, Path, description = "Pending request id emitted by the matching permission_required SSE event."),
    ),
    request_body = ResponseBody,
    responses(
        (status = 204, description = "Resolved; the parked turn continues"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session or request not found / already resolved", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid body, or the id names a sub-agent's conversation (`/errors/session-not-drivable`), which no payload makes acceptable", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn respond(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    Path((session_id, request_id)): Path<(Uuid, String)>,
    raw_body: Bytes,
) -> Result<StatusCode, ProblemDetail> {
    scope::require(&principal, "sessions:w")?;
    // Parse via `serde_json` directly so `deny_unknown_fields` rejections produce a
    // `application/problem+json` 422 instead of axum's default text/plain response.
    let body: ResponseBody = serde_json::from_slice(&raw_body).map_err(|error| {
        ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            format!("invalid response body: {}", error),
        )
    })?;
    // A re-attached session has an empty pending parking lot. Any `permission_required` request
    // emitted by the previous in-memory incarnation was already lost when the original SSE stream
    // disconnected. `resolve_permission` below will return `false` for unknown request ids in
    // that case, which surfaces to the client as 404 RequestNotFound.
    let entry = ensure_session_loaded(&state, session_id).await?;

    let resolution = match body.outcome {
        PermissionDecision::Allow => PermissionResolution::Allow,
        PermissionDecision::AllowAlways => PermissionResolution::AllowAlways,
        PermissionDecision::Deny => PermissionResolution::Deny,
        PermissionDecision::DenyAlways => PermissionResolution::DenyAlways,
    };

    if entry.frontend.resolve_permission(&request_id, resolution) {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(ProblemDetail::new(
            ErrorKind::RequestNotFound,
            StatusCode::NOT_FOUND,
            format!(
                "pending request '{}' for session '{}' is unknown or already resolved",
                request_id, session_id
            ),
        )
        .with("session_id", session_id.to_string())
        .with("request_id", request_id))
    }
}
