//! Server-level discovery endpoints: `/v1/info`, `/v1/skills`, `/v1/mcp`.

use axum::{Extension, Json, extract::State, http::StatusCode};
use serde::Serialize;
use utoipa::ToSchema;

use crate::server::{
    auth::Principal,
    errors::{ErrorKind, ProblemDetail},
    state::ServerState,
};

/// Any read scope is sufficient for discovery endpoints.
fn has_any_read_scope(principal: &Principal) -> bool {
    principal.has_scope("sessions:r")
        || principal.has_scope("mcp:r")
        || principal.has_scope("skills:r")
}

// `ProblemDetail` is ~128 bytes and only constructed on the rejection path of an auth check.
// Same trade-off as `extract_bearer` in auth.rs. See the rationale there.
#[allow(clippy::result_large_err)]
fn require_any_read_scope(principal: &Principal) -> Result<(), ProblemDetail> {
    if has_any_read_scope(principal) {
        Ok(())
    } else {
        Err(ProblemDetail::new(
            ErrorKind::AuthScope,
            StatusCode::FORBIDDEN,
            "scope `sessions:r`, `mcp:r`, or `skills:r` is required",
        ))
    }
}

#[derive(Serialize, ToSchema)]
pub struct InfoResponse {
    pub version: String,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub default_permission: String,
    pub enabled_permissions: Vec<String>,
    /// Whether the active profile accepts image attachments on `POST /turn`. The HTTP analogue of
    /// ACP's `promptCapabilities.image`, so a client can tell whether attaching one is worth the
    /// base64 payload instead of discovering it from a 422.
    pub vision: bool,
}

/// `GET /v1/info`: server identity + model surface. Authenticated; admits any token holding
/// at least one of `sessions:r`, `mcp:r`, or `skills:r`. Tokens with only write scopes get 403.
/// The broad-read fallback is intentional: a token configured for `sessions:r` to surface
/// session listings can also see the server's own version/model identity without operators
/// having to also grant `mcp:r` / `skills:r`.
#[utoipa::path(
    get,
    path = "/v1/info",
    tag = "discovery",
    responses(
        (status = 200, description = "Server identity and capability flags", body = InfoResponse),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn info(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
) -> Result<Json<InfoResponse>, ProblemDetail> {
    require_any_read_scope(&principal)?;
    let config = &state.shared.config;
    Ok(Json(InfoResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
        provider: config.provider_name.clone(),
        model: config.model.clone(),
        default_permission: config.permission.to_string(),
        enabled_permissions: config
            .enabled_permissions
            .iter()
            .map(|p| p.to_string())
            .collect(),
        vision: config.vision,
    }))
}

#[derive(Serialize, ToSchema)]
pub struct SkillView {
    pub name: String,
    pub description: String,
}

/// `GET /v1/skills`: installed skill palette. Mirrors what the REPL `/skill` command and
/// the ACP `available_commands_update` notification surface.
#[utoipa::path(
    get,
    path = "/v1/skills",
    tag = "discovery",
    responses(
        (status = 200, description = "Installed skill palette", body = [SkillView]),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn skills(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
) -> Result<Json<Vec<SkillView>>, ProblemDetail> {
    require_any_read_scope(&principal)?;
    let snapshot = state.shared.skills.current().await;
    let skills = snapshot
        .iter()
        .map(|skill| SkillView {
            name: skill.name.clone(),
            description: skill.description.clone(),
        })
        .collect();
    Ok(Json(skills))
}

#[derive(Serialize, ToSchema)]
pub struct McpServerView {
    pub name: String,
    pub state: String,
}

/// `GET /v1/mcp`: configured MCP servers and their current connection state.
#[utoipa::path(
    get,
    path = "/v1/mcp",
    tag = "discovery",
    responses(
        (status = 200, description = "Per-server connection state", body = [McpServerView]),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn mcp(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
) -> Result<Json<Vec<McpServerView>>, ProblemDetail> {
    require_any_read_scope(&principal)?;
    let Some(manager) = state.shared.mcp_manager.as_ref() else {
        return Ok(Json(Vec::new()));
    };
    let names = manager.server_names();
    let mut servers = Vec::with_capacity(names.len());
    for name in names {
        let server_state = match manager.server_entry(&name) {
            Some(entry) => entry.state().await.label().to_string(),
            None => "unknown".to_string(),
        };
        servers.push(McpServerView {
            name,
            state: server_state,
        });
    }
    Ok(Json(servers))
}
