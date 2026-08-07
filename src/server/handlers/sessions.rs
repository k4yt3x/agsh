//! Session CRUD: create, list, get, delete.
//!
//! Mirrors the ACP session lifecycle (`session/new` / `session/list` / etc.) but over HTTP+JSON
//! and with `Authorization: Bearer` gating per scope.

use std::sync::{Arc, RwLock};

use axum::{
    Extension, Json,
    body::Bytes,
    extract::{Path, Query, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

use crate::{
    agent::SharedCwd,
    conversation::Conversation,
    permission::{EnabledPermissions, Permission, SharedPermission},
    server::{
        auth::Principal,
        errors::{ErrorKind, ProblemDetail},
        http_frontend::{HttpFrontend, SessionCapabilities},
        reattach::ensure_session_loaded,
        state::{ServerState, SessionEntry, SessionRuntime},
    },
    session::SessionManager,
};

/// RAII guard that deletes a freshly-created session DB row when an in-flight create handler
/// returns an error after the row has been written. Without this, a failure between
/// `create_session_with_metadata` and the final success response leaves an orphan row.
///
/// `Drop` can't `.await` the async `delete_session` call directly, so we spawn it on the runtime.
/// The cleanup task runs after the response has flushed; that's fine because nothing else can
/// observe the orphaned row until the next `GET /v1/sessions` scan.
struct SessionRollback {
    uuid: Uuid,
    manager: SessionManager,
    armed: bool,
}

impl SessionRollback {
    fn new(uuid: Uuid, manager: SessionManager) -> Self {
        Self {
            uuid,
            manager,
            armed: true,
        }
    }

    fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for SessionRollback {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // Guard `tokio::spawn` with `Handle::try_current`: during graceful shutdown the
        // runtime may already be tearing down and an unguarded spawn would panic.
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::debug!(
                "session rollback for {} during shutdown; skipping orphan-row delete",
                self.uuid
            );
            return;
        };
        let uuid = self.uuid;
        let manager = self.manager.clone();
        handle.spawn(async move {
            if let Err(error) = manager.delete_session(uuid).await {
                tracing::warn!(
                    "session rollback: failed to delete orphan row {}: {}",
                    uuid,
                    error,
                );
            } else {
                tracing::info!("session rollback: deleted orphan row {}", uuid);
            }
        });
    }
}

/// `deny_unknown_fields` rejects typos like `permision: "read"` with 422 instead of silently
/// falling back to defaults.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct CreateSessionRequest {
    /// Absolute path. Defaults to the server process's `current_dir` if omitted.
    #[schema(value_type = Option<String>)]
    pub cwd: Option<std::path::PathBuf>,
    /// Permission level the session starts in. Defaults to the server's configured default
    /// from `[permissions].default` (typically `read`). Must be in the enabled set.
    pub permission: Option<String>,
    /// Per-session capability flags. See the HTTP API docs § "Capabilities".
    #[serde(default)]
    pub capabilities: CapabilitiesBody,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct CapabilitiesBody {
    /// When `true`, the SSE stream includes `thinking.delta` events for extended-thinking
    /// content. Default `false`: chat-transcript clients (Telegram bridges) don't want
    /// reasoning inline.
    #[serde(default)]
    pub supports_reasoning_stream: bool,
    /// When `false`, mid-turn permission requests are denied immediately with a notice instead of
    /// parking on the SSE channel. Streaming clients with no approval interface set this. Default
    /// `true`, which is the historical behaviour.
    #[serde(default = "default_true")]
    pub supports_permission_prompts: bool,
}

/// `#[serde(default)]` on a `bool` yields `false`; this is for the fields that default to `true`.
fn default_true() -> bool {
    true
}

impl Default for CapabilitiesBody {
    fn default() -> Self {
        Self {
            supports_reasoning_stream: false,
            supports_permission_prompts: true,
        }
    }
}

impl From<CapabilitiesBody> for SessionCapabilities {
    fn from(body: CapabilitiesBody) -> Self {
        Self {
            supports_reasoning_stream: body.supports_reasoning_stream,
            supports_permission_prompts: body.supports_permission_prompts,
        }
    }
}

/// Decode the persisted `capabilities_json` column back into a `SessionCapabilities`. NULL or
/// invalid JSON yields the defaults. Used on the DB-fallback path for evicted sessions.
fn capabilities_from_row(json: Option<&str>) -> SessionCapabilities {
    json.and_then(|raw| serde_json::from_str::<SessionCapabilities>(raw).ok())
        .unwrap_or_default()
}

#[derive(Debug, Serialize, ToSchema)]
pub struct SessionResponse {
    pub id: Uuid,
    pub created_at: String,
    pub updated_at: String,
    /// Wall-clock timestamp (RFC 3339) of the last successful turn on this session. `None`
    /// when the session has never run a turn (just-created or just-re-attached). Distinct
    /// from `updated_at`, which advances on any session-level mutation (PATCH included).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_turn_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<String>)]
    pub cwd: Option<std::path::PathBuf>,
    pub permission: String,
    pub title: String,
    /// Per-session capability flags declared at create time (or re-attach), echoed back so clients
    /// can confirm the settings their session actually ended up with.
    pub capabilities: SessionCapabilities,
    /// Whether a turn is running on this session right now.
    ///
    /// Exists so a client whose SSE stream dropped mid-turn can tell "my turn is still running"
    /// from "my turn died" without submitting a speculative turn and reading the 409. A dropped
    /// stream does not cancel the turn: the spawned task keeps the runtime lock and dropping the
    /// `JoinHandle` detaches rather than aborts, so the work completes and resubmitting would
    /// duplicate a reply the user is about to receive anyway.
    ///
    /// Always `false` for a GC-evicted session, since eviction requires an idle session.
    pub turn_in_flight: bool,
}

#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct ListSessionsQuery {
    #[serde(default)]
    pub limit: Option<u32>,
    #[serde(default)]
    pub cursor: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ListSessionsResponse {
    pub sessions: Vec<SessionResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

/// Validate a caller-supplied `cwd` path. Rejects:
/// - relative paths
/// - paths containing null bytes (the kernel truncates at `\0`, creating a mismatch between the
///   path the caller intended and the path the OS actually resolves)
/// - paths that don't exist on the filesystem
/// - paths that exist but aren't directories
///
/// This is *input validation*, not a security sandbox. A valid absolute directory still lets the
/// agent operate anywhere the OS permissions allow. The check prevents obviously-wrong inputs
/// (like `/dev/null` or `/proc/self`) from producing confusing downstream tool errors.
#[allow(clippy::result_large_err)]
fn validate_cwd(path: &std::path::Path) -> Result<(), ProblemDetail> {
    if !path.is_absolute() {
        return Err(ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            "`cwd` must be an absolute path",
        ));
    }
    // Null bytes in a path are always a bug: Unix syscalls treat \0 as the terminator, so
    // `/tmp\0/etc/shadow` would resolve to `/tmp` at the kernel level while the application
    // layer thinks it's something else.
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        if path.as_os_str().as_bytes().contains(&0) {
            return Err(ProblemDetail::new(
                ErrorKind::InvalidBody,
                StatusCode::UNPROCESSABLE_ENTITY,
                "`cwd` must not contain null bytes",
            ));
        }
    }
    match std::fs::metadata(path) {
        Ok(meta) => {
            if !meta.is_dir() {
                return Err(ProblemDetail::new(
                    ErrorKind::InvalidBody,
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!("`cwd` exists but is not a directory: {}", path.display()),
                ));
            }
        }
        Err(_) => {
            return Err(ProblemDetail::new(
                ErrorKind::InvalidBody,
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("`cwd` does not exist: {}", path.display()),
            ));
        }
    }
    Ok(())
}

/// POST /v1/sessions: create a session.
///
/// Requires scope `sessions:w`. The created session's runtime (Agent, ToolRegistry,
/// HttpFrontend) is constructed eagerly so subsequent `POST /turn` doesn't pay the build cost.
#[utoipa::path(
    post,
    path = "/v1/sessions",
    tag = "sessions",
    request_body = CreateSessionRequest,
    responses(
        (status = 201, description = "Session created", body = SessionResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 422, description = "Invalid body", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn create_session(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    raw_body: Bytes,
) -> Result<(StatusCode, Json<SessionResponse>), ProblemDetail> {
    if !principal.has_scope("sessions:w") {
        return Err(ProblemDetail::new(
            ErrorKind::AuthScope,
            StatusCode::FORBIDDEN,
            "scope `sessions:w` is required to create sessions",
        ));
    }

    let body: CreateSessionRequest = serde_json::from_slice(&raw_body).map_err(|_| {
        ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid session creation request body",
        )
    })?;

    let cwd_path = match body.cwd {
        Some(path) => {
            validate_cwd(&path)?;
            path
        }
        // Propagate `current_dir()` failure as 500 rather than falling back to a relative
        // path, which would surprise tools that resolve paths absolutely.
        None => std::env::current_dir().map_err(|error| {
            ProblemDetail::new(
                ErrorKind::Internal,
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(
                    "server cannot resolve a default working directory: {}",
                    error
                ),
            )
        })?,
    };
    let cwd: SharedCwd = Arc::new(RwLock::new(cwd_path.clone()));

    let permission: Permission = match body.permission.as_deref() {
        Some(value) => value.parse().map_err(|error| {
            ProblemDetail::new(
                ErrorKind::InvalidBody,
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("invalid `permission` value: {}", error),
            )
        })?,
        None => state.shared.config.permission,
    };
    let enabled: EnabledPermissions = state.shared.config.enabled_permissions;
    if !enabled.is_enabled(permission) {
        return Err(ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            format!(
                "permission `{}` is not in the server's enabled set",
                permission
            ),
        ));
    }
    let shared_permission = SharedPermission::new(permission, enabled);

    let capabilities: SessionCapabilities = body.capabilities.into();
    let http_frontend = Arc::new(HttpFrontend::with_capabilities(capabilities));
    let frontend_dyn: Arc<dyn crate::frontend::Frontend> = http_frontend.clone();

    // Persist `permission` and `capabilities` so a GC-evicted session re-attaches with the
    // same shape the client created it with.
    let capabilities_json = serde_json::to_string(&capabilities).ok();
    let created = state
        .shared
        .session_manager
        .create_session_with_metadata(
            Some(cwd_path.clone()),
            Some(permission.to_string()),
            capabilities_json,
            Some(principal.token_id.clone()),
        )
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to create session", error))?;
    let session_uuid = created.id;
    // Parse the canonical `created_at` returned by the DB so the in-memory entry's timestamp
    // matches the persisted row exactly.
    let created_at_wall = chrono::DateTime::parse_from_rfc3339(&created.created_at)
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .unwrap_or_else(|_| chrono::Utc::now());
    // Arm the rollback guard: every `?` below will clean up the orphan DB row on failure.
    let rollback = SessionRollback::new(session_uuid, state.shared.session_manager.clone());
    let session_lock = state
        .shared
        .session_manager
        .lock_session(session_uuid)
        .map_err(|error| ProblemDetail::internal_sanitized("failed to lock session", error))?;

    // Build the per-session Agent + ToolRegistry.
    let (agent, tool_registry) = crate::build_session_agent(
        &state.shared,
        shared_permission.clone(),
        frontend_dyn,
        cwd.clone(),
        // The HTTP API is single-root: additional workspace roots are an ACP-only surface.
        Arc::new(std::sync::RwLock::new(Vec::new())),
    )
    .await
    .map_err(|error| ProblemDetail::internal_sanitized("failed to build session agent", error))?;

    let runtime = SessionRuntime {
        session_uuid,
        messages: Conversation::new(),
        agent,
        tool_registry,
    };

    let entry = SessionEntry {
        session_uuid,
        token_id: Some(principal.token_id.clone()),
        runtime: Arc::new(tokio::sync::Mutex::new(runtime)),
        permission: shared_permission,
        cwd: cwd.clone(),
        created_at: created_at_wall,
        updated_at: Arc::new(RwLock::new(created_at_wall)),
        last_turn_at: Arc::new(RwLock::new(std::time::Instant::now())),
        last_turn_at_wall: Arc::new(RwLock::new(None)),
        capabilities,
        frontend: http_frontend,
        cancellation: Arc::new(RwLock::new(tokio_util::sync::CancellationToken::new())),
        session_lock: Arc::new(session_lock),
        in_flight: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    };

    state.sessions.write().await.insert(session_uuid, entry);

    // Just-created session has zero messages, so title is always empty. Skip the DB round-trip.
    let title = String::new();

    tracing::info!(
        "session created: id={} cwd={:?} permission={} token={}",
        session_uuid,
        cwd_path,
        permission,
        principal.token_id,
    );

    // Past the point of no return: disarm so the rollback Drop doesn't fire.
    rollback.disarm();
    // Use the canonical `created_at` from the DB insert so all three surfaces agree.
    let timestamp = created.created_at;
    Ok((
        StatusCode::CREATED,
        Json(SessionResponse {
            id: session_uuid,
            created_at: timestamp.clone(),
            updated_at: timestamp,
            last_turn_at: None,
            cwd: Some(cwd_path),
            permission: permission.to_string(),
            title,
            capabilities,
            turn_in_flight: false,
        }),
    ))
}

/// GET /v1/sessions: paginated list. Returns persisted sessions from the DB (not just
/// in-memory entries) so audit consumers can see everything regardless of GC state.
#[utoipa::path(
    get,
    path = "/v1/sessions",
    tag = "sessions",
    params(ListSessionsQuery),
    responses(
        (status = 200, description = "Page of sessions", body = ListSessionsResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn list_sessions(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    Query(query): Query<ListSessionsQuery>,
) -> Result<Json<ListSessionsResponse>, ProblemDetail> {
    if !principal.has_scope("sessions:r") {
        return Err(ProblemDetail::new(
            ErrorKind::AuthScope,
            StatusCode::FORBIDDEN,
            "scope `sessions:r` is required",
        ));
    }
    let limit = query.limit.unwrap_or(50).min(200);
    let (rows, next_cursor) = state
        .shared
        .session_manager
        .list_sessions(limit, false, None, query.cursor.as_deref())
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to list sessions", error))?;

    // Enrich DB rows with live in-memory metadata where available; GC-evicted sessions fall
    // back to persisted columns (which may be empty for legacy rows).
    let in_memory = state.sessions.read().await;
    let sessions = rows
        .into_iter()
        .map(|row| {
            let live = in_memory.get(&row.id);
            // Fall back to the persisted `permission` column for GC-evicted entries.
            let permission = match live {
                Some(entry) => entry.permission.get().to_string(),
                None => row.permission.clone().unwrap_or_default(),
            };
            // Use `row.created_at` (not `updated_at`) for evicted sessions so the creation
            // timestamp isn't incorrectly aged forward by subsequent turns.
            let created_at = live
                .map(|entry| entry.created_at.to_rfc3339())
                .unwrap_or_else(|| row.created_at.clone());
            let updated_at = live
                .and_then(|entry| entry.updated_at.read().ok().map(|guard| guard.to_rfc3339()))
                .unwrap_or_else(|| row.updated_at.clone());
            let last_turn_at = live.and_then(|entry| {
                entry
                    .last_turn_at_wall
                    .read()
                    .ok()
                    .and_then(|guard| guard.map(|ts| ts.to_rfc3339()))
            });
            // Recover capabilities from the persisted JSON column for evicted rows.
            let capabilities = live
                .map(|entry| entry.capabilities)
                .unwrap_or_else(|| capabilities_from_row(row.capabilities_json.as_deref()));
            let turn_in_flight = live.is_some_and(|entry| {
                entry.in_flight.load(std::sync::atomic::Ordering::Acquire) > 0
            });
            SessionResponse {
                id: row.id,
                created_at,
                updated_at,
                last_turn_at,
                cwd: row.cwd,
                permission,
                title: row.preview,
                capabilities,
                turn_in_flight,
            }
        })
        .collect();
    drop(in_memory);

    Ok(Json(ListSessionsResponse {
        sessions,
        next_cursor,
    }))
}

/// GET /v1/sessions/{id}: single session metadata.
#[utoipa::path(
    get,
    path = "/v1/sessions/{id}",
    tag = "sessions",
    params(("id" = Uuid, Path, description = "Session UUID")),
    responses(
        (status = 200, description = "Session record", body = SessionResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn get_session(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    Path(id): Path<Uuid>,
) -> Result<Json<SessionResponse>, ProblemDetail> {
    if !principal.has_scope("sessions:r") {
        return Err(ProblemDetail::new(
            ErrorKind::AuthScope,
            StatusCode::FORBIDDEN,
            "scope `sessions:r` is required",
        ));
    }
    if let Some(entry) = state.sessions.read().await.get(&id).cloned() {
        let updated_at = entry
            .updated_at
            .read()
            .ok()
            .map(|guard| guard.to_rfc3339())
            .unwrap_or_default();
        // Title isn't cached in-memory; a DB error falls back to empty rather than 500
        // because title is descriptive, not load-bearing.
        let title = state
            .shared
            .session_manager
            .session_info(id)
            .await
            .ok()
            .flatten()
            .map(|info| info.preview)
            .unwrap_or_default();
        let last_turn_at = entry
            .last_turn_at_wall
            .read()
            .ok()
            .and_then(|guard| guard.map(|ts| ts.to_rfc3339()));
        return Ok(Json(SessionResponse {
            id: entry.session_uuid,
            created_at: entry.created_at.to_rfc3339(),
            updated_at,
            last_turn_at,
            cwd: Some(crate::agent::cwd_snapshot(&entry.cwd)),
            permission: entry.permission.get().to_string(),
            title,
            capabilities: entry.capabilities,
            turn_in_flight: entry.in_flight.load(std::sync::atomic::Ordering::Acquire) > 0,
        }));
    }
    let summary = state
        .shared
        .session_manager
        .session_info(id)
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to look up session", error))?
        .ok_or_else(|| {
            ProblemDetail::new(
                ErrorKind::SessionNotFound,
                StatusCode::NOT_FOUND,
                format!("session '{}' does not exist", id),
            )
            .with("session_id", id.to_string())
        })?;
    // Evicted-but-persisted row: fall back to DB columns for permission/capabilities.
    let capabilities = capabilities_from_row(summary.capabilities_json.as_deref());
    Ok(Json(SessionResponse {
        id: summary.id,
        created_at: summary.created_at,
        updated_at: summary.updated_at,
        last_turn_at: None,
        cwd: summary.cwd,
        permission: summary.permission.unwrap_or_default(),
        title: summary.preview,
        capabilities,
        turn_in_flight: false,
    }))
}

/// PATCH /v1/sessions/{id}: update mutable session knobs (permission, cwd) on a live session
/// without re-creating it. Returns the updated metadata.
///
/// Permission and cwd are hoisted on [`SessionEntry`] outside the runtime mutex precisely so the
/// PATCH handler can apply them without contending with a long-running turn; the change is
/// visible to the next agent operation that reads the cells.
#[utoipa::path(
    patch,
    path = "/v1/sessions/{id}",
    tag = "sessions",
    params(("id" = Uuid, Path, description = "Session UUID")),
    request_body = PatchSessionRequest,
    responses(
        (status = 200, description = "Updated session record", body = SessionResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
        (status = 409, description = "Turn in flight; cancel first", body = ProblemDetail),
        (status = 422, description = "Invalid body", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn patch_session(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    Path(id): Path<Uuid>,
    raw_body: Bytes,
) -> Result<Json<SessionResponse>, ProblemDetail> {
    if !principal.has_scope("sessions:w") {
        return Err(ProblemDetail::new(
            ErrorKind::AuthScope,
            StatusCode::FORBIDDEN,
            "scope `sessions:w` is required",
        ));
    }

    let body: PatchSessionRequest = serde_json::from_slice(&raw_body).map_err(|_| {
        ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid session patch request body",
        )
    })?;

    let entry = ensure_session_loaded(&state, id).await?;

    // Reject PATCH while a turn is in-flight: the agent snapshots cwd/permission at turn
    // start, but tools read them live, creating a split-brain within one iteration.
    if entry.in_flight.load(std::sync::atomic::Ordering::Acquire) > 0 {
        return Err(turn_in_flight_conflict(
            id,
            "session has an in-flight turn; cancel it first via POST /v1/sessions/{id}/cancel \
             before patching session metadata",
        ));
    }

    // Validate all fields up-front before any DB write so a mixed valid/invalid request
    // (e.g. valid permission + invalid cwd) doesn't leave a half-applied state.
    let new_permission = match body.permission.as_deref() {
        Some(level) => {
            let parsed: Permission = level.parse().map_err(|error| {
                ProblemDetail::new(
                    ErrorKind::InvalidBody,
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!("invalid `permission` value: {}", error),
                )
            })?;
            if !state.shared.config.enabled_permissions.is_enabled(parsed) {
                return Err(ProblemDetail::new(
                    ErrorKind::InvalidBody,
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!("permission `{}` is not in the server's enabled set", parsed),
                ));
            }
            Some(parsed)
        }
        None => None,
    };
    let new_cwd = match body.cwd.clone() {
        Some(path) => {
            validate_cwd(&path)?;
            Some(path)
        }
        None => None,
    };

    // Filter out no-op fields so a PATCH that doesn't change anything skips the DB write
    // and doesn't advance `updated_at` (used by clients for change detection).
    let permission_change: Option<Permission> =
        new_permission.filter(|parsed| entry.permission.get() != *parsed);
    let cwd_change: Option<std::path::PathBuf> =
        new_cwd.filter(|path| crate::agent::cwd_snapshot(&entry.cwd) != *path);
    let mutated = permission_change.is_some() || cwd_change.is_some();
    if mutated {
        state
            .shared
            .session_manager
            .update_session_metadata_atomic(
                id,
                permission_change.map(|perm| perm.to_string()),
                cwd_change.clone(),
            )
            .await
            .map_err(|error| {
                ProblemDetail::internal_sanitized(
                    "failed to persist session metadata atomically",
                    error,
                )
            })?;
        // DB write succeeded; apply the in-memory mirror. `try_set` re-validates against
        // the enabled set as belt-and-braces; a failure here would indicate a config
        // reload race (not currently supported) and is treated as a 500.
        if let Some(parsed) = permission_change {
            entry.permission.try_set(parsed).map_err(|error| {
                ProblemDetail::new(
                    ErrorKind::Internal,
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(
                        "permission `{}` failed in-memory validation after DB persist",
                        error.0
                    ),
                )
            })?;
        }
        if let Some(path) = cwd_change {
            *crate::server::poisoned::write(&entry.cwd, "patch_session::cwd") = path;
        }
    }

    // Bump `updated_at` only on actual changes; leave `last_turn_at` alone so the GC
    // scanner's idle timer tracks provider activity, not metadata edits.
    if mutated && let Ok(mut guard) = entry.updated_at.write() {
        *guard = chrono::Utc::now();
    }
    let cwd_snapshot = crate::agent::cwd_snapshot(&entry.cwd);
    let updated_at = entry
        .updated_at
        .read()
        .ok()
        .map(|guard| guard.to_rfc3339())
        .unwrap_or_default();
    let title = state
        .shared
        .session_manager
        .session_info(id)
        .await
        .ok()
        .flatten()
        .map(|info| info.preview)
        .unwrap_or_default();
    let last_turn_at = entry
        .last_turn_at_wall
        .read()
        .ok()
        .and_then(|guard| guard.map(|ts| ts.to_rfc3339()));
    Ok(Json(SessionResponse {
        id: entry.session_uuid,
        created_at: entry.created_at.to_rfc3339(),
        updated_at,
        last_turn_at,
        cwd: Some(cwd_snapshot),
        permission: entry.permission.get().to_string(),
        title,
        capabilities: entry.capabilities,
        turn_in_flight: entry.in_flight.load(std::sync::atomic::Ordering::Acquire) > 0,
    }))
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct PatchSessionRequest {
    /// New permission level (`read` / `write` / `ask`). Must be in the server's enabled set.
    /// Absent → keep current.
    #[serde(default)]
    pub permission: Option<String>,
    /// New working directory. Must be absolute. Absent → keep current.
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub cwd: Option<std::path::PathBuf>,
}

/// Build the 409 returned when a mutating session operation races an in-flight turn. The
/// `detail` message varies per call site (delete vs patch); the type, status, and `session_id`
/// extension are fixed.
fn turn_in_flight_conflict(id: Uuid, detail: impl Into<String>) -> ProblemDetail {
    ProblemDetail::new(ErrorKind::TurnInFlight, StatusCode::CONFLICT, detail)
        .with("session_id", id.to_string())
}

/// DELETE /v1/sessions/{id}: drop the in-memory entry and (optionally) the DB row.
#[utoipa::path(
    delete,
    path = "/v1/sessions/{id}",
    tag = "sessions",
    params(("id" = Uuid, Path, description = "Session UUID")),
    responses(
        (status = 204, description = "Session deleted (idempotent, also returned for unknown ids)"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 409, description = "Turn in flight; cancel first", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn delete_session(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, ProblemDetail> {
    if !principal.has_scope("sessions:w") {
        return Err(ProblemDetail::new(
            ErrorKind::AuthScope,
            StatusCode::FORBIDDEN,
            "scope `sessions:w` is required",
        ));
    }
    // Refuse DELETE while a turn is in flight: silently destroying agent work would surprise
    // callers. DB-delete runs BEFORE the in-memory remove so a transient DB failure leaves
    // the session usable (client can retry).
    {
        let map = state.sessions.read().await;
        if let Some(entry) = map.get(&id)
            && entry.in_flight.load(std::sync::atomic::Ordering::Acquire) > 0
        {
            return Err(turn_in_flight_conflict(
                id,
                "session has an in-flight turn; cancel it first via POST /v1/sessions/{id}/cancel",
            ));
        }
        let present_in_memory = map.contains_key(&id);
        drop(map);
        if !present_in_memory {
            let exists = state
                .shared
                .session_manager
                .session_exists(id)
                .await
                .map_err(|error| {
                    ProblemDetail::internal_sanitized(
                        "failed to check session existence during delete",
                        error,
                    )
                })?;
            // Truly idempotent: return 204 even when the id is unknown.
            if !exists {
                return Ok(StatusCode::NO_CONTENT);
            }
        }
    }

    let mut map = state.sessions.write().await;
    if let Some(entry) = map.get(&id)
        && entry.in_flight.load(std::sync::atomic::Ordering::Acquire) > 0
    {
        return Err(turn_in_flight_conflict(
            id,
            "session has an in-flight turn; cancel it first via POST /v1/sessions/{id}/cancel",
        ));
    }
    // DB-delete first: on failure the in-memory entry stays so the session keeps working.
    state
        .shared
        .session_manager
        .delete_session(id)
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to delete session", error))?;
    let removed = map.remove(&id);
    drop(map);

    // Detach the tool registry from MCP so `tools/list_changed` callbacks stop targeting it.
    // `try_lock` is safe: the in-flight check passed and the entry is removed from the map.
    if let Some(entry) = removed.as_ref()
        && let Some(manager) = state.shared.mcp_manager.as_ref()
        && let Ok(runtime) = entry.runtime.try_lock()
    {
        let registry = runtime.tool_registry.clone();
        drop(runtime);
        manager.detach_registry(&registry).await;
    }
    Ok(StatusCode::NO_CONTENT)
}
