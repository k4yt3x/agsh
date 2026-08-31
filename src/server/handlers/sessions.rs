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
    conversation::Conversation,
    permission::{EnabledPermissions, Permission, SharedPermission},
    server::{
        auth::Principal,
        errors::{ErrorKind, ProblemDetail},
        http_frontend::{HttpFrontend, SessionCapabilities},
        reattach::{agent_build_problem, ensure_session_loaded, require_session_exists},
        scope,
        state::{ServerState, SessionEntry, SessionRuntime},
    },
    session::SessionManager,
    workspace::SharedCwd,
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
                "session rollback: skipping orphan-row delete for {} during shutdown",
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
    /// Provider profile the session runs on, and keeps for the rest of its life unless a later
    /// PATCH moves it. Defaults to the server's own default profile. Must name a profile in
    /// `config.toml`; `GET /v1/providers` lists them.
    pub provider: Option<String>,
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
    /// `true`, so a client that says nothing keeps the chance to approve.
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
    /// The provider profile this session runs on.
    pub provider: String,
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
    /// The session this one was spawned from, for a sub-agent; absent for a top-level session.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<Uuid>,
}

#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct ListSessionsQuery {
    #[serde(default)]
    pub limit: Option<u32>,
    #[serde(default)]
    pub cursor: Option<String>,
    /// Include sub-agent sessions. Default `false`, which lists only top-level conversations.
    ///
    /// Off by default because a dispatcher that spawns freely would otherwise bury its own
    /// sessions under the workers it started, and a client paging through the list wants the
    /// conversations it created. Turn it on to audit what was spawned; `parent_id` on each row is
    /// what reconnects a worker to the session that dispatched it.
    #[serde(default)]
    pub include_children: Option<bool>,
    /// Only list sessions whose working directory matches this path exactly.
    #[serde(default)]
    pub cwd: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ListSessionsResponse {
    pub sessions: Vec<SessionResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

/// Reject a provider profile the server does not have configured, naming the ones it does.
///
/// The 422 is what keeps a typo from being recorded on the row: a name that resolves to nothing
/// would fail every subsequent turn on that session, and the write is what makes it stick.
fn require_configured_profile(state: &ServerState, name: &str) -> Result<(), ProblemDetail> {
    if state.shared.config.providers.contains_key(name) {
        return Ok(());
    }
    let configured: Vec<&str> = state
        .shared
        .config
        .providers
        .keys()
        .map(String::as_str)
        .collect();
    Err(ProblemDetail::new(
        ErrorKind::InvalidBody,
        StatusCode::UNPROCESSABLE_ENTITY,
        format!(
            "provider profile `{}` is not configured; configured profiles: {}",
            name,
            if configured.is_empty() {
                "none".to_string()
            } else {
                configured.join(", ")
            }
        ),
    ))
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
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid body", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn create_session(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    raw_body: Bytes,
) -> Result<(StatusCode, Json<SessionResponse>), ProblemDetail> {
    scope::require(&principal, "sessions:w")?;

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
    // The profile this session will run with for the rest of its life, unless a PATCH moves it.
    let profile = match body.provider {
        Some(name) => {
            require_configured_profile(&state, &name)?;
            name
        }
        None => state.shared.default_profile.clone(),
    };
    // Created and locked in one step, the lock taken *before* the row exists. A row committed
    // ahead of its lock is visible to `meka session delete --all`, which enumerates at delete
    // time, takes the lock nobody holds yet, and cascades the session away underneath this
    // handler. See `SessionManager::create_session_locked`.
    let (created, created_lock) = state
        .shared
        .session_manager
        .create_session_locked(
            Some(cwd_path.clone()),
            Some(permission.to_string()),
            capabilities_json,
            Some(principal.token_id.clone()),
            profile.clone(),
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
    // `None` means the claim could not be made at all -- an unwritable lock directory, descriptors
    // exhausted -- never that somebody else holds it, since no other process can know this id yet.
    // A served session that cannot be held alone is one this server must not admit.
    let session_lock = created_lock
        .map_err(|error| ProblemDetail::internal_sanitized("failed to lock session", error))?;

    // Build the per-session Agent + ToolRegistry.
    // Retained so `GET /context` and `GET /tools` can read them without the runtime mutex; see
    // the note on `SessionEntry::context_used`.
    let context_used = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let context_overhead = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let (agent, tool_registry) = crate::build_session_agent(
        &state.shared,
        Some(session_uuid),
        shared_permission.clone(),
        frontend_dyn,
        cwd.clone(),
        // The HTTP API is single-root: additional workspace roots are an ACP-only surface.
        Arc::new(std::sync::RwLock::new(Vec::new())),
        Arc::clone(&context_used),
        Arc::clone(&context_overhead),
        // Nothing here holds a window gauge built before the agent, the way the REPL prompt and
        // ACP's frontend do, so this handle is supplied only to be adopted; the entry reads it
        // back through `published_binding` below.
        Arc::new(std::sync::atomic::AtomicU64::new(0)),
    )
    .await
    .map_err(|error| agent_build_problem(session_uuid, "failed to build session agent", error))?;

    let background_tasks = agent.background_tasks();
    // Taken out of the assembled agent rather than rebuilt here, so the entry watches the cell
    // `set_provider` writes instead of a snapshot something has to remember to refresh.
    let binding = agent.published_binding();
    let runtime = SessionRuntime {
        session_uuid,
        messages: Conversation::new(),
        agent,
    };

    let entry = SessionEntry {
        session_uuid,
        token_id: Some(principal.token_id.clone()),
        runtime: Arc::new(tokio::sync::Mutex::new(runtime)),
        permission: shared_permission,
        binding,
        cwd: cwd.clone(),
        background_tasks,
        tool_registry,
        context_used,
        context_overhead,
        created_at: created_at_wall,
        updated_at: Arc::new(RwLock::new(created_at_wall)),
        last_turn_at: Arc::new(RwLock::new(std::time::Instant::now())),
        last_turn_at_wall: Arc::new(RwLock::new(None)),
        capabilities,
        frontend: http_frontend,
        cancellation: Arc::new(RwLock::new(tokio_util::sync::CancellationToken::new())),
        cancel_epoch: Arc::new(std::sync::atomic::AtomicU64::new(0)),
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
            provider: profile,
            title,
            capabilities,
            turn_in_flight: false,
            // Top-level by construction: `POST /v1/sessions` has no way to name a parent, and a
            // sub-agent session is only ever minted by `agent_spawn` inside a turn.
            parent_id: None,
        }),
    ))
}

/// Body for `POST /v1/sessions/{id}/fork`. Every field is optional; omitted means "inherit from
/// the session being forked".
///
/// Only `cwd` is offered, mirroring ACP's `session/fork` request, which likewise carries a
/// workspace but no permission or capability fields. Both of those are inherited and remain
/// changeable afterwards via `PATCH /v1/sessions/{id}`.
#[derive(Debug, Default, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ForkSessionBody {
    /// Absolute path for the forked session. Absent → inherit the source's.
    #[schema(value_type = Option<String>)]
    pub cwd: Option<std::path::PathBuf>,
}

/// POST /v1/sessions/{id}/fork: copy a session's conversation into a new one.
///
/// Requires scope `sessions:w`. The copy starts with the source's full conversation and is
/// immediately usable; the source is left untouched and is not required to be in memory, so an
/// evicted session forks just as well as a live one.
#[utoipa::path(
    post,
    path = "/v1/sessions/{id}/fork",
    tag = "sessions",
    params(("id" = Uuid, Path, description = "Session UUID to fork")),
    request_body = Option<ForkSessionBody>,
    responses(
        (status = 201, description = "Forked session", body = SessionResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid body, or the source is a sub-agent's conversation, whose copy is another sub-agent and so has no live session to hand back (`/errors/session-not-drivable`)", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn fork_session(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    Path(id): Path<Uuid>,
    raw_body: Bytes,
) -> Result<(StatusCode, Json<SessionResponse>), ProblemDetail> {
    scope::require(&principal, "sessions:w")?;

    // An empty body is the common case (`inherit everything`), and axum hands it to us as zero
    // bytes, which is not valid JSON.
    let body: ForkSessionBody = if raw_body.is_empty() {
        ForkSessionBody::default()
    } else {
        serde_json::from_slice(&raw_body).map_err(|_| {
            ProblemDetail::new(
                ErrorKind::InvalidBody,
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid session fork request body",
            )
        })?
    };
    if let Some(path) = body.cwd.as_deref() {
        validate_cwd(path)?;
    }

    // Before the copy, not after it. A fork of a sub-agent is a sibling under the same parent
    // (`SessionManager::fork_session`), so the copy is a worker too and `ensure_session_loaded`
    // below refuses to build it -- correctly, but far too late to say anything useful: the caller
    // would get a refusal naming an id it has never seen, for a row this handler had already
    // rolled back. Answering here names the id the caller actually sent.
    //
    // The store still copies the columns; what this door declines is the part of its own contract
    // it cannot honour, which is handing back a live session. `meka session fork` takes no runtime
    // and so still makes the copy.
    if let Some(terms) = state
        .shared
        .session_manager
        .spawn_terms(id)
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to read session", error))?
    {
        let detail = match terms.parent {
            Some(parent) => format!(
                "session '{id}' was spawned by session '{parent}', and a copy of it is a \
                 sub-agent under that same parent rather than a session of its own, so this \
                 endpoint has no live session to return. Read it with \
                 `GET /v1/sessions/{id}/messages`, or continue the conversation with \
                 `agent_followup` from '{parent}'."
            ),
            // `spawn_terms` rather than the parent link, so this names the id the caller sent even
            // for an imported worker. Keyed on the link alone, that case fell through, the copy
            // was made, `ensure_session_loaded` refused *it*, and the rollback ran -- so the
            // caller got a 422 naming an id it had never seen, for a row that no longer existed.
            // That is verbatim the failure this up-front guard exists to prevent.
            None => format!(
                "session '{id}' carries the terms another session spawned it under, so it is a \
                 sub-agent's conversation and a copy of it is another one. Its parent is not in \
                 this store, so there is nothing here that can drive either. Read it with \
                 `GET /v1/sessions/{id}/messages`."
            ),
        };
        return Err(ProblemDetail::new(
            ErrorKind::SessionNotDrivable,
            StatusCode::UNPROCESSABLE_ENTITY,
            detail,
        )
        .with("session_id", id.to_string()));
    }

    // Deliberately the *unlocked* fork, unlike the REPL's and ACP's. `ensure_session_loaded` below
    // takes the lock itself, and `flock` is per open file description, so a lock held here would
    // make this handler refuse its own copy. That leaves the same commit-then-claim window the
    // other two doors closed -- but this one fails safe: `ensure_session_loaded` checks
    // `session_info` before locking, so a copy swept before that check surfaces as a 404 rather
    // than a session whose next turn dies on a foreign-key violation.
    //
    // Not "cannot happen": a sweep landing between re-attach's `session_info` and its
    // `lock_session` still builds a runtime on a vanished row, answers 201, and leaves the first
    // turn to hit the foreign key. There is no `.await` between those two, so the window is far
    // narrower than the one this door still has -- but it is the same outcome, and closing either
    // properly means teaching re-attach to adopt a lock it did not take.
    let forked = state
        .shared
        .session_manager
        .fork_session(id, crate::session::ForkOverrides {
            cwd: body.cwd,
            // The HTTP API is single-root, but the copy keeps whatever the source recorded so a
            // later ACP `session/load` still sees the workspace shape. HTTP runtimes ignore the
            // column either way, exactly as re-attach already does for ACP-created sessions.
            additional_roots: None,
            // Never inherited: this fingerprints the token that created a session, and the token
            // doing the forking is the only correct answer.
            token_id: Some(principal.token_id.clone()),
        })
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to fork session", error))?
        .ok_or_else(|| {
            ProblemDetail::new(
                ErrorKind::SessionNotFound,
                StatusCode::NOT_FOUND,
                format!("session '{}' does not exist", id),
            )
            .with("session_id", id.to_string())
        })?;

    // Build the copy's runtime through the re-attach path rather than duplicating it: the row was
    // just written, so re-attach resolves permission, capabilities, cwd, and `token_id` straight
    // from it, hydrates the conversation, takes the lock, and registers the entry.
    let entry = match crate::server::reattach::ensure_session_loaded(&state, forked.id).await {
        Ok(entry) => entry,
        Err(problem) => {
            // The row exists but is unusable; drop it rather than leaving an orphan the caller
            // was never told about.
            if let Err(error) = state.shared.session_manager.delete_session(forked.id).await {
                tracing::warn!(
                    "failed to roll back fork {} after runtime build failed: {}",
                    forked.id,
                    error
                );
            }
            return Err(problem);
        }
    };

    tracing::info!(
        "session forked: source={} id={} token={}",
        id,
        forked.id,
        principal.token_id,
    );

    let forked_info = state
        .shared
        .session_manager
        .session_info(forked.id)
        .await
        .ok()
        .flatten();
    let title = forked_info
        .as_ref()
        .map(|info| info.preview.clone())
        .unwrap_or_default();
    // Read back rather than copied off the source entry for the same reason `parent_id` is: the
    // fork's row is what the next turn on it will resolve, so the response should quote that.
    let forked_provider = forked_info
        .as_ref()
        .map(|info| info.provider.clone())
        .unwrap_or_default();
    Ok((
        StatusCode::CREATED,
        Json(SessionResponse {
            id: forked.id,
            created_at: forked.created_at.clone(),
            updated_at: forked.created_at,
            last_turn_at: None,
            cwd: Some(crate::workspace::cwd_snapshot(&entry.cwd)),
            permission: entry.permission.get().to_string(),
            provider: forked_provider,
            title,
            capabilities: entry.capabilities,
            turn_in_flight: false,
            // Read back rather than assumed: forking a sub-agent session keeps it under the same
            // parent, so the copy is a sibling of the original, not a new root. That sentence was
            // here before `fork_session` did it -- the copy took a NULL parent, so this reported
            // one thing and the store held another, and the difference was a `sessions:w` holder's
            // way around `crate::refuse_a_spawned_session`.
            parent_id: forked_info.and_then(|info| info.parent_id),
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
    scope::require(&principal, "sessions:r")?;
    let limit = query.limit.unwrap_or(50).min(200);
    let cwd_filter = query.cwd.as_deref().map(std::path::Path::new);
    let (rows, next_cursor) = state
        .shared
        .session_manager
        .list_sessions(
            limit,
            query.include_children.unwrap_or(false),
            cwd_filter,
            query.cursor.as_deref(),
        )
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to list sessions", error))?;

    // Enrich DB rows with live in-memory metadata where available; sessions with no in-memory entry
    // fall back to persisted columns, which are NULL for rows the HTTP server did not create (REPL,
    // ACP, sub-agent, imported).
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
                provider: row.provider,
                title: row.preview,
                capabilities,
                turn_in_flight,
                parent_id: row.parent_id,
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
    scope::require(&principal, "sessions:r")?;
    if let Some(entry) = state.sessions.read().await.get(&id).cloned() {
        let updated_at = entry
            .updated_at
            .read()
            .ok()
            .map(|guard| guard.to_rfc3339())
            .unwrap_or_default();
        // Title isn't cached in-memory; a DB error falls back to empty rather than 500
        // because title is descriptive, not load-bearing.
        let info = state
            .shared
            .session_manager
            .session_info(id)
            .await
            .ok()
            .flatten();
        let title = info
            .as_ref()
            .map(|info| info.preview.clone())
            .unwrap_or_default();
        let provider = info
            .as_ref()
            .map(|info| info.provider.clone())
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
            cwd: Some(crate::workspace::cwd_snapshot(&entry.cwd)),
            permission: entry.permission.get().to_string(),
            provider,
            title,
            capabilities: entry.capabilities,
            turn_in_flight: entry.in_flight.load(std::sync::atomic::Ordering::Acquire) > 0,
            parent_id: info.and_then(|info| info.parent_id),
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
        provider: summary.provider,
        title: summary.preview,
        capabilities,
        turn_in_flight: false,
        parent_id: summary.parent_id,
    }))
}

/// Move a session that is not resident onto another provider profile, without building an agent
/// for it.
///
/// The rescue path for a session whose recorded profile is no longer configured. Nothing here needs
/// the agent: the row is the thing being changed, there is no in-flight turn to conflict with and
/// no in-memory mirror to update, and the response comes off the row the way `GET` already answers
/// for an evicted session. The profile is still resolved first, so a body naming one that cannot
/// produce a provider is refused before the row moves, exactly as on the resident path.
///
/// `Ok(None)` means the session became resident after all, and the caller must take the resident
/// path so the agent moves with the row. Everything below rests on "not resident", and the check
/// that established it is several awaits behind: resolving a profile can build a provider and load
/// a credential, and any turn, scheduler fire, compaction or rewind arriving meanwhile rebuilds the
/// agent from the *old* profile and inserts it. Writing the row then left the session running,
/// billing and gauging the profile it had left, with `GET /v1/sessions` reporting the new one, for
/// as long as it stayed resident. Holding the reconstruction lock for the whole body is what makes
/// the re-check below conclusive rather than another sample.
///
/// The reconstruction lock answers for *this* process only, so the session lock is taken too. Every
/// other writer of `sessions.provider` already holds it -- a CLI resume, `/provider`, ACP's
/// `session/set_config_option`, and this handler's own resident path, which is resident here and so
/// holds it through the entry -- which makes "you may move a session's binding only while you own
/// the session" an invariant rather than a coincidence. Without it, a second `meka serve` on the
/// same store answered `200` for a session the first one was running: the row moved, `GET
/// /v1/sessions` on *both* reported the new profile, and the host actually holding the session went
/// on building requests for the old one until eviction. Reproduced over HTTP against two servers.
async fn repin_dormant_session(
    state: &ServerState,
    id: Uuid,
    profile: &str,
) -> Result<Option<Json<SessionResponse>>, ProblemDetail> {
    let _reconstruction = crate::server::reattach::lock_session_reconstruction(id).await;
    if state.sessions.read().await.contains_key(&id) {
        return Ok(None);
    }
    require_session_exists(state, id).await?;
    // `session-locked`, the same code `ensure_session_loaded` answers a cross-process conflict
    // with, and for the same reason: this is another process owning the session, not an in-process
    // turn. Held for the rest of the body, so nothing can attach between here and the write.
    let _session = state
        .shared
        .session_manager
        .lock_session(id)
        .map_err(|error| {
            ProblemDetail::new(
                ErrorKind::SessionLocked,
                StatusCode::CONFLICT,
                format!(
                    "another meka process is running this session, so its provider cannot be \
                     changed from here; move it where it is running, or stop that process: {}",
                    error
                ),
            )
            .with("session_id", id.to_string())
        })?;
    require_configured_profile(state, profile)?;
    let resolved = crate::provider::resolved_binding(
        &state.shared.providers,
        profile.to_string(),
    )
    .await
    // Discriminated, not a blanket 422; see the sibling site in `patch_session`.
    .map_err(|error| {
        agent_build_problem(
            id,
            &format!("failed to resolve provider profile `{}`", profile),
            error,
        )
    })?;

    let current = state
        .shared
        .session_manager
        .recorded_provider(id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to read session provider", error)
        })?;
    // Skipped when nothing changes, so a no-op PATCH does not advance `updated_at` that clients
    // watch for changes. Nothing to reconcile beyond the name: a dormant session has no live agent
    // to put back in step, which is what the resident path has to do unconditionally.
    if current.as_ref() != Some(&resolved.binding) {
        state
            .shared
            .session_manager
            .update_session_metadata_atomic(id, None, None, Some(resolved.binding))
            .await
            .map_err(|error| {
                ProblemDetail::internal_sanitized(
                    "failed to persist session metadata atomically",
                    error,
                )
            })?;
    }

    // Re-read rather than patching the pre-write copy, so `updated_at` and `provider` are what the
    // store now holds.
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
    Ok(Some(Json(SessionResponse {
        id: summary.id,
        created_at: summary.created_at,
        updated_at: summary.updated_at,
        last_turn_at: None,
        cwd: summary.cwd,
        permission: summary.permission.unwrap_or_default(),
        provider: summary.provider,
        title: summary.preview,
        capabilities: capabilities_from_row(summary.capabilities_json.as_deref()),
        // Not resident, and nothing could have made it so while the reconstruction lock was held.
        turn_in_flight: false,
        parent_id: summary.parent_id,
    })))
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
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid body, or the id names a sub-agent's conversation, whose permission, cwd and profile come from the terms its parent spawned it with (`/errors/session-not-drivable`)", body = ProblemDetail),
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
    scope::require(&principal, "sessions:w")?;

    let body: PatchSessionRequest = serde_json::from_slice(&raw_body).map_err(|_| {
        ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid session patch request body",
        )
    })?;

    // Ahead of the dormant fast path below, which is the one branch of this handler that writes a
    // session row without ever building an agent -- so it is the one branch
    // `crate::refuse_a_spawned_session` cannot answer for from inside the builders. A worker is
    // almost never resident (`build_subagent` runs it under its parent's runtime rather than
    // registering it here), so that branch was where every `PATCH` aimed at a worker actually
    // landed: it wrote `sessions.provider` on a sub-agent's row and answered `200`, while the same
    // request against a *resident* session was refused. The write did not even survive -- the next
    // `agent_followup` records the profile it really ran on, which is the parent's -- so what a
    // `sessions:w` holder got was a row that disagreed with the worker until something quietly put
    // it back.
    crate::refuse_a_spawned_session(&state.shared.session_manager, Some(id))
        .await
        .map_err(|error| agent_build_problem(id, "failed to read session", error))?;

    // A body that names only a provider is the documented way to move a session whose recorded
    // profile has left `config.toml`, and reviving it to apply the change is the one thing that
    // cannot work in that state: `ensure_session_loaded` rebuilds the agent, which resolves the
    // very profile that is gone, so the rescue was refused by the failure it was meant to
    // repair. A session already resident falls through to the path below, because
    // `ensure_session_loaded` returns it without rebuilding and its live agent has to move with
    // the row.
    if body.permission.is_none()
        && body.cwd.is_none()
        && let Some(name) = body.provider.as_deref()
        && !state.sessions.read().await.contains_key(&id)
        && let Some(response) = repin_dormant_session(&state, id, name).await?
    {
        return Ok(response);
    }

    let entry = ensure_session_loaded(&state, id).await?;

    // Reject PATCH while a turn is in-flight: the agent snapshots cwd/permission at turn
    // start, but tools read them live, creating a split-brain within one iteration.
    //
    //
    // A read and not an `InFlightGuard`, deliberately. That guard means "this session is busy with
    // turn-like work", and claiming it here would make two concurrent PATCHes conflict when they
    // should simply serialise -- metadata edits are not turns, and a client that sends two is not
    // doing anything wrong. The cost is that a turn admitted between this load and the agent swap
    // below makes that swap wait for it, which is slow rather than wrong: the row has already
    // moved, and the swap lands correctly afterwards.
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
    // Built before the DB write for the same reason the other two are validated first: a profile
    // that cannot produce a provider must not reach the row, and building it here means the 422
    // covers a missing credential as well as a missing profile.
    let new_provider = match body.provider.as_deref() {
        Some(name) => {
            require_configured_profile(&state, name)?;
            // The profile as configured. A `PATCH` naming a provider moves the session to that
            // bundle entire, which is the only thing naming a profile can mean.
            Some(
                crate::provider::resolved_binding(
                    &state.shared.providers,
                    name.to_string(),
                )
                .await
                // Not a blanket 422: `resolved_binding` reaches `provider_credential_version` and
                // `load_provider_credential`, so a locked or unreadable store arrives here as
                // `MekaError::Database`, and formatting it into the body answered "your request is
                // invalid" with an internal message attached. `agent_build_problem` is the
                // discriminator the rest of this surface already uses -- `Config` verbatim, because
                // naming the profile is the whole point, everything else sanitised into a 500.
                .map_err(|error| {
                    agent_build_problem(
                        id,
                        &format!("failed to resolve provider profile `{}`", name),
                        error,
                    )
                })?,
            )
        }
        None => None,
    };

    // Filter out no-op fields so a PATCH that doesn't change anything skips the DB write
    // and doesn't advance `updated_at` (used by clients for change detection).
    let permission_change: Option<Permission> =
        new_permission.filter(|parsed| entry.permission.get() != *parsed);
    let cwd_change: Option<std::path::PathBuf> =
        new_cwd.filter(|path| crate::workspace::cwd_snapshot(&entry.cwd) != *path);
    // Only whether the *row* already says this, because that is all this decides. The live agent
    // is moved further down regardless, so a `PATCH` naming the profile the row already records is
    // how a session whose agent and row have come apart is put back together. Gating the swap on
    // this too made that the one state no request could repair: the retry after a failed `PATCH`
    // found the row already correct, wrote nothing, swapped nothing, and left every turn running
    // on the profile the client had just been told it was off.
    let provider_row_write = match new_provider.as_ref() {
        Some(resolved) => {
            let current = state
                .shared
                .session_manager
                .recorded_provider(id)
                .await
                .map_err(|error| {
                    ProblemDetail::internal_sanitized("failed to read session provider", error)
                })?;
            current.as_ref() != Some(&resolved.binding)
        }
        None => false,
    };
    let mutated = permission_change.is_some() || cwd_change.is_some() || provider_row_write;
    if mutated {
        state
            .shared
            .session_manager
            .update_session_metadata_atomic(
                id,
                permission_change.map(|perm| perm.to_string()),
                cwd_change.clone(),
                provider_row_write
                    .then(|| {
                        new_provider
                            .as_ref()
                            .map(|resolved| resolved.binding.clone())
                    })
                    .flatten(),
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

    // The live agent moves last, and only after the row it must agree with. An `await` and not a
    // `try_lock`: the row has already moved, and a resident session's next turn runs on the agent
    // rather than re-reading the row, so refusing to swap would leave the two disagreeing until
    // eviction. See the in-flight note at the top of the handler.
    //
    // Outside the `mutated` block, because "the row already says this" is exactly when a repair is
    // wanted and never a reason to skip one. Re-publishing the profile the agent already runs on
    // costs one lock and changes nothing.
    if let Some(resolved) = new_provider {
        // One call, not three. The window and the vision flag the entry reports both come off the
        // cell `set_provider` publishes into, so moving the agent moves them.
        entry.runtime.lock().await.agent.set_provider(resolved);
    }

    // Bump `updated_at` only on actual changes; leave `last_turn_at` alone so the GC
    // scanner's idle timer tracks provider activity, not metadata edits.
    if mutated && let Ok(mut guard) = entry.updated_at.write() {
        *guard = chrono::Utc::now();
    }
    let cwd_snapshot = crate::workspace::cwd_snapshot(&entry.cwd);
    let updated_at = entry
        .updated_at
        .read()
        .ok()
        .map(|guard| guard.to_rfc3339())
        .unwrap_or_default();
    let info = state
        .shared
        .session_manager
        .session_info(id)
        .await
        .ok()
        .flatten();
    let title = info
        .as_ref()
        .map(|info| info.preview.clone())
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
        provider: info
            .as_ref()
            .map(|info| info.provider.clone())
            .unwrap_or_default(),
        title,
        capabilities: entry.capabilities,
        turn_in_flight: entry.in_flight.load(std::sync::atomic::Ordering::Acquire) > 0,
        parent_id: info.and_then(|info| info.parent_id),
    }))
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct PatchSessionRequest {
    /// New permission level (`none` / `read` / `workspace` / `ask` / `unrestricted`). Must be in
    /// the server's enabled set. Absent → keep current.
    #[serde(default)]
    pub permission: Option<String>,
    /// New working directory. Must be absolute. Absent → keep current.
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub cwd: Option<std::path::PathBuf>,
    /// Provider profile to move the session onto. Must name a profile in `config.toml`. Absent →
    /// keep current.
    ///
    /// Switching mid-conversation is allowed and is the client's call. Thinking blocks are tagged
    /// with the provider that produced them and are not replayed to a different one, so the
    /// reasoning recorded so far stops being visible to the model from the next turn onward.
    #[serde(default)]
    pub provider: Option<String>,
}

/// Build the 409 returned when a mutating session operation races an in-flight turn. The
/// `detail` message varies per call site (delete vs patch); the type, status, and `session_id`
/// extension are fixed.
pub(crate) fn turn_in_flight_conflict(id: Uuid, detail: impl Into<String>) -> ProblemDetail {
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
    scope::require(&principal, "sessions:w")?;
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

    // The write lock covers the in-flight re-check and the map removal, and nothing else. It used
    // to span the DB delete as well, which is a cascading `DELETE` plus a lock-directory sweep that
    // does blocking `read_dir` / `remove_file` on the connection thread -- and that thread is
    // shared, so the call can queue behind a large `GET /export` or `POST /import`. Held across
    // that, and with tokio's `RwLock` being write-preferring, a single DELETE stalls every reader
    // in the process: `POST /cancel`, `GET /stream`, the GC scan, the background poller.
    //
    // Shortening it is safe because the map lock is not what serialises this against a concurrent
    // re-attach: the removed entry still owns the session's cross-process `FileLock`, and holds
    // it until the end of this function. A request arriving in the gap finds no map entry, loads
    // the row, fails to take the file lock, and gets `session-locked` -- never a second live entry
    // for a session being deleted.
    //
    // The one case that argument does not cover is a session that was already evicted, where there
    // is no entry and so no lock to hold. A re-attach that has taken the file lock and passed its
    // own existence re-check could then insert an entry for a row this delete is about to remove.
    // The window is the few instructions between the two, and both sides funnel through the single
    // database connection, which orders the delete ahead of the re-check in practice.
    let removed = {
        let mut map = state.sessions.write().await;
        if let Some(entry) = map.get(&id)
            && entry.in_flight.load(std::sync::atomic::Ordering::Acquire) > 0
        {
            return Err(turn_in_flight_conflict(
                id,
                "session has an in-flight turn; cancel it first via POST /v1/sessions/{id}/cancel",
            ));
        }
        map.remove(&id)
    };

    // A failure here leaves the row in place with the entry already evicted, so the session is
    // still usable: the next request re-attaches it from the row it just failed to delete.
    state
        .shared
        .session_manager
        .delete_session(id)
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to delete session", error))?;

    if let Some(entry) = removed.as_ref() {
        // Signalled after the row is gone, not before: a failed DB delete leaves the session
        // usable, and killing its work first would make that rollback a lie.
        //
        // The in-flight check above only covers *turns*. A detached background task never sets
        // `in_flight`, so DELETE sails past one, and the cascade has just taken the
        // `background_tasks` rows with the session -- which is what makes this the last chance to
        // stop it. Without this the task and its whole process group run on with no row to find
        // them by, so `DELETE /v1/sessions/{id}/tasks/{task_id}` now 404s and only restarting the
        // server (which still would not reap the process group) ends it. The REPL does the same
        // thing on its way out.
        let signalled = entry.background_tasks.cancel_session(id).await;
        if signalled > 0 {
            tracing::info!(
                "deleting session {} signalled {} running background task(s) to stop",
                id,
                signalled
            );
        }

        // Detach the tool registry from MCP so `tools/list_changed` callbacks stop targeting it.
        // Read off the hoisted handle rather than through `runtime`: a `try_lock` here fails
        // against anything holding the mutex, and skipping the detach would leak the registry into
        // the manager for the life of the process.
        if let Some(manager) = state.shared.mcp_manager.as_ref() {
            manager.detach_registry(&entry.tool_registry).await;
        }
    }
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    /// The dormant repin must hold both locks; see
    /// [`crate::server::reattach::assert_dormant_fast_path_is_serialised`] for why this is asserted
    /// against the source.
    ///
    /// What it defends here: the function decides what to write on the strength of the session
    /// *not* being resident, then awaits five times before writing. Without the reconstruction
    /// lock, any turn, scheduler fire, compaction or rewind arriving meanwhile rebuilds the agent
    /// from the *old* profile; the row then moves and the session runs, bills and gauges the
    /// profile it had left until it is evicted -- hours, at the default idle timeout.
    ///
    /// `tests/serve.rs`'s `a_dormant_repin_and_the_agent_rebuilt_after_it_agree` covers what is
    /// reachable -- that the PATCH takes the dormant path at all, and that the row and the agent
    /// rebuilt from it agree afterwards -- but not the serialisation, which only shows up under a
    /// race it cannot create.
    #[test]
    fn the_dormant_repin_serialises_against_reconstruction() {
        crate::server::reattach::assert_dormant_fast_path_is_serialised(
            include_str!("sessions.rs"),
            "async fn repin_dormant_session(",
            "session_info(id)",
            "update_session_metadata_atomic",
        );
    }
}
