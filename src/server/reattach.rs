//! GC-evicted-session re-attach. When a session's in-memory `SessionEntry` is dropped (because
//! the GC scanner exceeded `idle_timeout`) but its DB row is retained (the default `delete_on_idle
//! = false`), a later request with the same session UUID rebuilds the runtime from disk and
//! continues the conversation. Mirrors ACP's `session/load` semantics
//! (`src/acp.rs::handle_load_session`).
//!
//! Inserted between every mutating handler and its session-map lookup: instead of
//! `state.sessions.read().await.get(&id).cloned()` returning `None` → 404, the handler calls
//! [`ensure_session_loaded`] which falls through to reconstruction.
//!
//! Permission and per-session capabilities are persisted on the `sessions` row
//! (`src/session.rs::SessionSummary`), so a session created with `permission = "read"` re-attaches
//! with `permission = "read"`, not the process default.

use std::sync::{Arc, RwLock};

use axum::http::StatusCode;
use uuid::Uuid;

use super::{
    errors::{ErrorKind, ProblemDetail},
    http_frontend::{HttpFrontend, SessionCapabilities},
    state::{ServerState, SessionEntry, SessionRuntime},
};
use crate::{
    agent::SharedCwd,
    conversation::Conversation,
    permission::{Permission, SharedPermission},
};

/// Look up a session, reconstructing it from the persisted DB row if the in-memory entry has been
/// evicted. Returns the (now in-memory) `SessionEntry` on success, a 404 problem detail when the
/// session id is unknown to both the map and the DB, or a 500-class problem detail when
/// reconstruction fails.
///
/// On reconstruction, emits a `tracing::info!` so operators can see re-attach events in their
/// observability pipeline.
pub async fn ensure_session_loaded(
    state: &ServerState,
    id: Uuid,
) -> Result<SessionEntry, ProblemDetail> {
    // Fast path: in-memory entry.
    if let Some(entry) = state.sessions.read().await.get(&id).cloned() {
        return Ok(entry);
    }

    let started = std::time::Instant::now();

    // Cold path: query the DB to see whether the session exists at all.
    let summary = state
        .shared
        .session_manager
        .session_info(id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to look up session during re-attach", error)
                .with("session_id", id.to_string())
        })?
        .ok_or_else(|| {
            ProblemDetail::new(
                ErrorKind::SessionNotFound,
                StatusCode::NOT_FOUND,
                format!("session '{}' does not exist", id),
            )
            .with("session_id", id.to_string())
        })?;

    // Resolve persisted permission. NULL on legacy rows (REPL / ACP / pre-0.27 HTTP): fall back
    // to the process default. The HTTP `create_session` handler validates against the enabled set
    // at insert time, but a stored permission could in principle become disabled by an operator
    // editing config; defensively re-check.
    let enabled = state.shared.config.enabled_permissions;
    let permission: Permission = match summary.permission.as_deref() {
        Some(value) => value.parse().unwrap_or(state.shared.config.permission),
        None => state.shared.config.permission,
    };
    let permission = if enabled.is_enabled(permission) {
        permission
    } else {
        state.shared.config.permission
    };
    let shared_permission = SharedPermission::new(permission, enabled);

    // Resolve persisted capabilities. NULL → defaults. Parse failures are surfaced via
    // `warn!` rather than silently falling back, so schema mismatches are operator-visible.
    let capabilities = match summary.capabilities_json.as_deref() {
        Some(json) => match serde_json::from_str::<SessionCapabilities>(json) {
            Ok(parsed) => parsed,
            Err(error) => {
                tracing::warn!(
                    session_id = %id,
                    error = %error,
                    "capabilities_json failed to parse; falling back to default capabilities"
                );
                SessionCapabilities::default()
            }
        },
        None => SessionCapabilities::default(),
    };

    // When no persisted `cwd` exists (legacy rows), default to the server's process working
    // directory. Propagate `current_dir()` failure as 500: the operator can fix it.
    let cwd_path = match summary.cwd.clone() {
        Some(path) => path,
        None => std::env::current_dir().map_err(|error| {
            ProblemDetail::new(
                ErrorKind::Internal,
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(
                    "server cannot resolve a default working directory for session re-attach: {}",
                    error
                ),
            )
            .with("session_id", id.to_string())
        })?,
    };
    let cwd: SharedCwd = Arc::new(RwLock::new(cwd_path.clone()));

    let session_lock = state
        .shared
        .session_manager
        .lock_session(id)
        .map_err(|error| {
            // `session-locked` (not `turn-in-flight`): this is a cross-process file-lock
            // conflict, not an in-process turn concurrency issue.
            ProblemDetail::new(
                ErrorKind::SessionLocked,
                StatusCode::CONFLICT,
                format!("failed to re-attach session: {}", error),
            )
            .with("session_id", id.to_string())
        })?;

    // Whatever the last owner left running is ours to retire now; see
    // `crate::background::claim_session`.
    crate::background::claim_session(&state.shared.session_manager, id).await;

    let events = state
        .shared
        .session_manager
        .load_events(id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to load session events", error)
                .with("session_id", id.to_string())
        })?;
    let mut conversation = Conversation::from_events(events);
    // Drop an orphaned `tool_use` (no following `tool_result`) before adopting the session; the
    // provider rejects orphans on the next request. Matches the ACP load/resume/fork paths and REPL
    // resume, so every route back into a persisted session sanitises the same way: without this an
    // evicted session whose log stopped mid-tool-call came back only to fail its next turn.
    let dropped = conversation.sanitize_orphans();
    if !dropped.is_empty() {
        tracing::warn!(
            "dropped {} orphaned assistant message(s) with unmatched tool calls while re-attaching session {}",
            dropped.len(),
            id,
        );
    }

    let http_frontend = Arc::new(HttpFrontend::with_capabilities(capabilities));
    let frontend_dyn: Arc<dyn crate::frontend::Frontend> = http_frontend.clone();

    let (agent, tool_registry) = crate::build_session_agent(
        &state.shared,
        shared_permission.clone(),
        frontend_dyn,
        cwd.clone(),
        // The HTTP API is single-root: additional workspace roots are an ACP-only surface.
        Arc::new(std::sync::RwLock::new(Vec::new())),
    )
    .await
    .map_err(|error| {
        ProblemDetail::internal_sanitized("failed to rebuild session agent", error)
            .with("session_id", id.to_string())
    })?;

    let runtime = SessionRuntime {
        session_uuid: id,
        messages: conversation,
        agent,
        tool_registry,
    };

    // Use DB-persisted timestamps so GC + re-attach doesn't reset creation time.
    // Fall back to `Utc::now()` on parse failure (shouldn't happen; we wrote the RFC 3339).
    let now = chrono::Utc::now();
    let parsed_created_at = chrono::DateTime::parse_from_rfc3339(&summary.created_at)
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .unwrap_or(now);
    let parsed_updated_at = chrono::DateTime::parse_from_rfc3339(&summary.updated_at)
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .unwrap_or(now);
    let new_entry = SessionEntry {
        session_uuid: id,
        // Restore persisted `token_id`. `None` only for legacy rows that predate the column.
        token_id: summary.token_id.clone(),
        runtime: Arc::new(tokio::sync::Mutex::new(runtime)),
        permission: shared_permission,
        cwd,
        created_at: parsed_created_at,
        updated_at: Arc::new(RwLock::new(parsed_updated_at)),
        // `last_turn_at` is monotonic and used by GC; reset to `now` so a re-attached session
        // isn't immediately eligible for eviction. The wall-clock `last_turn_at_wall` reflects
        // "last turn time", which is unknown after re-attach (the DB stores `updated_at`, but
        // PATCH mutations bump that too), so leave it `None` until the next successful turn.
        last_turn_at: Arc::new(RwLock::new(std::time::Instant::now())),
        last_turn_at_wall: Arc::new(RwLock::new(None)),
        capabilities,
        frontend: http_frontend,
        cancellation: Arc::new(RwLock::new(tokio_util::sync::CancellationToken::new())),
        session_lock: Arc::new(session_lock),
        in_flight: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    };

    // Acquire the write lock to insert. Re-check the map under the write lock: a concurrent
    // request may have reconstructed the same session between our read and write acquisitions; if
    // so, drop ours and return theirs so the DB-row lock + agent isn't duplicated. The
    // `session_lock` and `agent` we built are dropped here, releasing the OS file lock cleanly.
    let mut sessions = state.sessions.write().await;
    if let Some(existing) = sessions.get(&id).cloned() {
        return Ok(existing);
    }
    // Re-check DB existence under the write lock to close the reconstruction-vs-delete
    // race: a DELETE between the initial load and this point would leave a dangling entry.
    let still_exists = state
        .shared
        .session_manager
        .session_exists(id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized(
                "failed to verify session existence during re-attach",
                error,
            )
            .with("session_id", id.to_string())
        })?;
    if !still_exists {
        return Err(ProblemDetail::new(
            ErrorKind::SessionNotFound,
            StatusCode::NOT_FOUND,
            format!("session '{}' was deleted during re-attach", id),
        )
        .with("session_id", id.to_string()));
    }
    sessions.insert(id, new_entry.clone());
    drop(sessions);

    tracing::info!(
        "session re-attached: id={} elapsed_ms={} permission={} cwd={:?}",
        id,
        started.elapsed().as_millis(),
        permission,
        cwd_path,
    );

    Ok(new_entry)
}
