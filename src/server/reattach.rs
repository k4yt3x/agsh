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
    conversation::Conversation,
    permission::{Permission, SharedPermission},
    workspace::SharedCwd,
};

/// Assert a session exists, without building anything.
///
/// The counterpart to [`ensure_session_loaded`] for read-only handlers. Reconstruction builds an
/// `Agent`, a `ToolRegistry` and an MCP-attached registry, then pins the result in the session map
/// until the GC scanner evicts it again; a handler that only reads rows out of SQLite (messages,
/// export, background tasks, scheduled jobs) has no use for any of that and should not resurrect a
/// session as a side effect of being asked about it. Sub-agent transcripts make the difference
/// concrete: listing a spawn tree would otherwise revive every worker in it.
pub async fn require_session_exists(state: &ServerState, id: Uuid) -> Result<(), ProblemDetail> {
    let exists = state
        .shared
        .session_manager
        .session_exists(id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to look up session", error)
                .with("session_id", id.to_string())
        })?;
    if exists {
        return Ok(());
    }
    Err(ProblemDetail::new(
        ErrorKind::SessionNotFound,
        StatusCode::NOT_FOUND,
        format!("session '{}' does not exist", id),
    )
    .with("session_id", id.to_string()))
}

/// One reconstruction at a time per session id.
///
/// Reconstruction takes the session's `fd_lock`, so two requests arriving together for a session
/// this process has not loaded raced: the loser got a `session-locked` 409 whose documented remedy
/// ("another process holds it") was wrong, because the holder was this process, a millisecond
/// ahead. Serialising makes the loser wait and then find the winner's entry on the re-check below.
static RECONSTRUCTION_LOCKS: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<Uuid, Arc<tokio::sync::Mutex<()>>>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Acquire the reconstruction lock for `id`, creating it on first use. Entries whose only remaining
/// owner is the map are dropped on the way past, so it stays bounded by concurrent reconstructions
/// rather than by every session the process has ever loaded.
async fn lock_session_reconstruction(id: Uuid) -> tokio::sync::OwnedMutexGuard<()> {
    let lock = {
        let mut registry = match RECONSTRUCTION_LOCKS.lock() {
            Ok(guard) => guard,
            // A panic while holding this map leaves only `Arc` clones behind; the data cannot be
            // torn, so recovering beats propagating the poison into every later re-attach.
            Err(poisoned) => poisoned.into_inner(),
        };
        registry.retain(|_, held| Arc::strong_count(held) > 1);
        Arc::clone(
            registry
                .entry(id)
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
        )
    };
    lock.lock_owned().await
}

/// Look up a session, reconstructing it from the persisted DB row if the in-memory entry has been
/// evicted. Returns the (now in-memory) `SessionEntry` on success, a 404 problem detail when the
/// session id is unknown to both the map and the DB, or a 500-class problem detail when
/// reconstruction fails.
///
/// On reconstruction, emits a `tracing::info!` so operators can see re-attach events in their
/// observability pipeline.
///
/// **Takes `state.sessions.write()` internally**, so a caller must not already hold a guard on that
/// map. Three call sites `drop(map)` immediately before reaching here for exactly this reason, and
/// nothing in the type system enforces it.
pub async fn ensure_session_loaded(
    state: &ServerState,
    id: Uuid,
) -> Result<SessionEntry, ProblemDetail> {
    // Fast path: in-memory entry.
    if let Some(entry) = state.sessions.read().await.get(&id).cloned() {
        return Ok(entry);
    }

    let _reconstruction = lock_session_reconstruction(id).await;

    // Re-check now that this task owns the reconstruction: a concurrent caller may have finished
    // while this one waited, and its entry is the one to use.
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

    // Resolve persisted permission. NULL for REPL, ACP and sub-agent rows, which carry no
    // per-session level and derive permission from process config instead, and for an imported
    // session whose archive omitted one: fall back to the process default. The HTTP
    // `create_session` handler validates against the enabled set at insert time, but a stored
    // permission could in principle become disabled by an operator editing config; defensively
    // re-check.
    let enabled = state.shared.config.enabled_permissions;
    let permission: Permission = crate::permission::parse_recorded_permission(
        summary.permission.as_deref(),
        &format_args!("session {id}"),
    )
    .unwrap_or(state.shared.config.permission);
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

    // A row can carry no `cwd`: `meka session import` stores the archive's value verbatim, and an
    // archive may omit it. Default to the server's process working directory, and propagate
    // `current_dir()` failure as 500 so the operator can fix it.
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

    let context_used = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let context_overhead = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let (agent, tool_registry) = crate::build_session_agent(
        &state.shared,
        shared_permission.clone(),
        frontend_dyn,
        cwd.clone(),
        // The HTTP API is single-root: additional workspace roots are an ACP-only surface.
        Arc::new(std::sync::RwLock::new(Vec::new())),
        Arc::clone(&context_used),
        Arc::clone(&context_overhead),
    )
    .await
    .map_err(|error| {
        ProblemDetail::internal_sanitized("failed to rebuild session agent", error)
            .with("session_id", id.to_string())
    })?;

    let background_tasks = agent.background_tasks();
    let runtime = SessionRuntime {
        session_uuid: id,
        messages: conversation,
        agent,
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
        // Restore persisted `token_id`. `None` for every session not created through an
        // authenticated HTTP request: REPL, ACP, sub-agent and imported rows.
        token_id: summary.token_id.clone(),
        runtime: Arc::new(tokio::sync::Mutex::new(runtime)),
        permission: shared_permission,
        cwd,
        background_tasks,
        tool_registry,
        context_used,
        context_overhead,
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
        cancel_epoch: Arc::new(std::sync::atomic::AtomicU64::new(0)),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Two requests for the same unloaded session must not reconstruct it concurrently.
    ///
    /// Both would open the session's file lock; the loser got a 409 whose documented remedy ("retry
    /// against the process that holds it") did not apply, because the winner was this very process.
    /// The fix serialises reconstruction per session id so the loser waits and then finds the
    /// winner's entry. Nothing tested it: making `lock_session_reconstruction` hand out a fresh
    /// mutex every call -- i.e. no serialisation at all -- left every suite green.
    #[tokio::test]
    async fn reconstructing_one_session_twice_is_serialised() {
        let id = Uuid::from_u128(0x5eed);
        let inside = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let overlapped = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let mut handles = Vec::new();
        for _ in 0..8 {
            let inside = Arc::clone(&inside);
            let overlapped = Arc::clone(&overlapped);
            handles.push(tokio::spawn(async move {
                let _guard = lock_session_reconstruction(id).await;
                if inside.fetch_add(1, std::sync::atomic::Ordering::SeqCst) != 0 {
                    overlapped.store(true, std::sync::atomic::Ordering::SeqCst);
                }
                // Long enough that an unserialised run overlaps essentially every time.
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                inside.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            }));
        }
        for handle in handles {
            handle.await.expect("task");
        }

        assert!(
            !overlapped.load(std::sync::atomic::Ordering::SeqCst),
            "two tasks reconstructed the same session id at once; both would race its file lock"
        );
    }

    /// The per-id locks must not accumulate forever: one entry per session ever re-attached would
    /// be an unbounded map keyed by attacker-suppliable ids. `retain` drops the ones nobody holds.
    #[tokio::test]
    async fn the_reconstruction_lock_registry_does_not_grow_without_bound() {
        for index in 0..64u128 {
            let guard = lock_session_reconstruction(Uuid::from_u128(index)).await;
            drop(guard);
        }
        // The next acquisition sweeps the released entries, so the map holds only the live one.
        let _guard = lock_session_reconstruction(Uuid::from_u128(9999)).await;
        let held = match RECONSTRUCTION_LOCKS.lock() {
            Ok(guard) => guard.len(),
            Err(poisoned) => poisoned.into_inner().len(),
        };
        assert!(
            held <= 2,
            "released reconstruction locks are never swept; the map grows per session id: {held}"
        );
    }
}
