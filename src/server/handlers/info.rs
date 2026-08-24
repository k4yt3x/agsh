//! Server-level discovery endpoints: `/v1/info`, `/v1/skills`, `/v1/mcp`.

use axum::{
    Extension, Json,
    extract::{Path, State},
    http::StatusCode,
};
use serde::Serialize;
use utoipa::ToSchema;

use crate::server::{
    auth::Principal,
    errors::{ErrorKind, ProblemDetail},
    scope::{self, ANY_READ_SCOPES},
    state::ServerState,
};

// `ProblemDetail` is ~128 bytes and only constructed on the rejection path of an auth check.
// Same trade-off as `extract_bearer` in auth.rs. See the rationale there.
fn require_any_read_scope(principal: &Principal) -> Result<(), ProblemDetail> {
    scope::require_any(principal, ANY_READ_SCOPES)
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

/// `GET /v1/info`: server identity + model surface. Authenticated; admits any token holding at
/// least one read scope (see [`crate::server::scope::ANY_READ_SCOPES`]). Tokens with only write
/// scopes get 403. The broad-read fallback is intentional: a token configured for `sessions:r` to
/// surface session listings can also see the server's own version/model identity without operators
/// having to grant it anything else.
#[utoipa::path(
    get,
    path = "/v1/info",
    tag = "discovery",
    responses(
        (status = 200, description = "Server identity and capability flags", body = InfoResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
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
    /// Listing rank, 0..=9, lower first.
    pub priority: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    /// What the skill needs from its environment, from the Agent Skills `compatibility` field.
    ///
    /// The one optional spec field that changes how the skill's instructions should be carried
    /// out, so a client rendering this palette has a reason to show it. `license` and
    /// `allowed-tools` are per-skill detail and stay on `GET /v1/skills/{name}`, which is where
    /// the body is too.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compatibility: Option<String>,
}

/// `GET /v1/skills`: installed skill palette. Mirrors what the REPL `/skill` command and
/// the ACP `available_commands_update` notification surface.
#[utoipa::path(
    get,
    path = "/v1/skills",
    tag = "discovery",
    responses(
        (status = 200, description = "Installed skill palette", body = [SkillView]),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
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
        .skills
        .iter()
        .map(|skill| SkillView {
            name: skill.name.clone(),
            // Sanitised: this is an index for a client to render, not a backup door. The body is
            // returned verbatim by `GET /v1/skills/{name}` for that purpose; a one-line label the
            // caller will draw in a list is a render, and the store now hands back file bytes.
            description: crate::memory::render_description_for_model(&skill.description),
            priority: skill.priority,
            version: skill.version(),
            author: skill.author(),
            compatibility: skill.compatibility.clone(),
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
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
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

#[derive(Serialize, ToSchema)]
pub struct McpToolView {
    /// Raw name as the server advertises it. This is the value to put in `allowed_tools` /
    /// `disabled_tools` / `tool_permissions`, which is why it is reported unmangled.
    pub raw_name: String,
    pub description: String,
    /// Output of the 5-step permission resolution chain.
    pub required_permission: String,
    /// Which step of that chain decided it, so a misclassified tool can be traced to the rule that
    /// classified it rather than guessed at.
    pub permission_source: String,
    /// `false` when `allowed_tools` / `disabled_tools` filters this tool out, i.e. the agent never
    /// sees it. Listed anyway: "configured away" and "not advertised" are different problems.
    pub allowed: bool,
}

#[derive(Serialize, ToSchema)]
pub struct McpToolsResponse {
    pub server: String,
    pub tools: Vec<McpToolView>,
}

/// `GET /v1/mcp/{name}/tools`: what one MCP server advertises, with resolved permissions.
///
/// Queries the server live rather than reporting the registered set, matching `meka mcp tools`:
/// the point is to see everything it offers, including tools config is currently filtering out.
#[utoipa::path(
    get,
    path = "/v1/mcp/{name}/tools",
    tag = "discovery",
    params(("name" = String, Path, description = "MCP server name")),
    responses(
        (status = 200, description = "Advertised tools", body = McpToolsResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such server", body = ProblemDetail),
        (status = 502, description = "Server unreachable or list_tools failed", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn mcp_tools(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    Path(name): Path<String>,
) -> Result<Json<McpToolsResponse>, ProblemDetail> {
    scope::require(&principal, "mcp:r")?;
    let manager = state
        .shared
        .mcp_manager
        .as_ref()
        .ok_or_else(|| no_such_server(&name))?;
    if manager.server_entry(&name).is_none() {
        return Err(no_such_server(&name));
    }
    let tools = manager
        .list_advertised_tools(&name)
        .await
        // A connection or `list_tools` failure is upstream, not the caller's: 502, the same
        // classification `MekaError::Provider` gets.
        .map_err(|error| {
            ProblemDetail::new(ErrorKind::Provider, StatusCode::BAD_GATEWAY, error.to_string())
                .with("server", name.clone())
        })?;
    Ok(Json(McpToolsResponse {
        server: name,
        tools: tools
            .into_iter()
            .map(|tool| McpToolView {
                raw_name: tool.raw_name,
                description: tool.description,
                required_permission: tool.resolved_permission.to_string(),
                permission_source: tool.permission_source.as_str().to_string(),
                allowed: tool.allowed,
            })
            .collect(),
    }))
}

fn no_such_server(name: &str) -> ProblemDetail {
    ProblemDetail::new(
        ErrorKind::NotFound,
        StatusCode::NOT_FOUND,
        format!("no MCP server named '{}'", name),
    )
    .with("server", name.to_string())
}

#[derive(Serialize, ToSchema)]
pub struct McpReconnectResponse {
    pub server: String,
    /// Where the server stands now: `connected`, `failed`, or `pending`.
    ///
    /// A 200 says meka acted on the request, not that the server came back, so read this to find
    /// out which. `pending` means no attempt was made because one was already under way. A
    /// `disabled` server is a 422 and never appears here.
    pub state: String,
}

/// `POST /v1/mcp/{name}/reconnect`: heal one server now.
///
/// An impatience button rather than the only route back: a server that failed its initial connect
/// is already being retried in the background with exponential backoff. This collapses the wait for
/// an operator who has just fixed whatever was wrong.
#[utoipa::path(
    post,
    path = "/v1/mcp/{name}/reconnect",
    tag = "discovery",
    params(("name" = String, Path, description = "MCP server name")),
    responses(
        (status = 200, description = "Read `state` for where the server now stands", body = McpReconnectResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such server", body = ProblemDetail),
        (status = 422, description = "Server is disabled in config", body = ProblemDetail),
        (status = 502, description = "A connected server's transport could not be re-established in time", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn mcp_reconnect(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    Path(name): Path<String>,
) -> Result<Json<McpReconnectResponse>, ProblemDetail> {
    scope::require(&principal, "mcp:w")?;
    let manager = state
        .shared
        .mcp_manager
        .as_ref()
        .ok_or_else(|| no_such_server(&name))?;
    let entry = manager
        .server_entry(&name)
        .ok_or_else(|| no_such_server(&name))?;

    // Read once, before the attempt, and classify the outcome against it rather than against the
    // wording of the error. `reconnect_server` refuses two states outright, and telling them apart
    // by sniffing the message for "disabled" makes a status code hostage to a string an upstream
    // server could also produce.
    let state_before = entry.state().await;
    if matches!(state_before, crate::mcp::ServerState::Pending) {
        // Not an error. The startup connector owns every `Pending` entry and is already connecting
        // it, so nothing is wrong and nothing needs doing; reporting the refusal as 502 would tell
        // a dashboard the server had failed when it is merely still starting, which is exactly
        // when a dashboard polling `GET /v1/mcp` reaches for this button.
        return Ok(Json(McpReconnectResponse {
            server: name,
            state: state_before.label().to_string(),
        }));
    }

    let timeout = state.shared.config.mcp_connect_timeout;
    let resolved = manager
        .reconnect_server(&name, timeout)
        .await
        .map_err(|error| {
            // A disabled server is the caller asking for something the config forbids (422); a
            // transport failure is upstream (502), the same classification `mcp_tools` gives it.
            // Collapsing both into 422 would tell a client to fix its request when the fix is to
            // start the server.
            let (kind, status) = if matches!(state_before, crate::mcp::ServerState::Disabled) {
                (ErrorKind::InvalidBody, StatusCode::UNPROCESSABLE_ENTITY)
            } else {
                (ErrorKind::Provider, StatusCode::BAD_GATEWAY)
            };
            ProblemDetail::new(kind, status, error.to_string()).with("server", name.clone())
        })?;
    tracing::info!(
        "reconnected MCP server '{}' via HTTP: {}",
        name,
        resolved.label()
    );
    Ok(Json(McpReconnectResponse {
        server: name,
        state: resolved.label().to_string(),
    }))
}
