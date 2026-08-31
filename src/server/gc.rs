//! Background session GC. Periodically scans the in-memory session map and evicts entries
//! whose `last_turn_at` is older than the configured `idle_timeout`. Eviction drops the
//! `SessionEntry` (which in turn drops the `FileLock`, releasing the OS file lock) but
//! leaves the SQLite row in place by default; a later request with the same session ID can
//! re-attach (mirroring ACP's `session/load` semantics).
//!
//! Set `[serve].delete_on_idle = true` to also delete the DB row.

use std::time::Duration;

use crate::server::state::ServerState;

/// Spawn the GC scanner task. Returns the `JoinHandle` so the caller can cancel-on-drop or
/// wait for it during shutdown. The task loops forever; cancel by aborting the handle or by
/// the parent runtime shutting down.
pub fn spawn(state: ServerState) -> tokio::task::JoinHandle<()> {
    let scan_interval = state.config.gc_scan_interval;
    let idle_timeout = state.config.idle_timeout;
    let delete_on_idle = state.config.delete_on_idle;
    if idle_timeout.is_zero() {
        tracing::info!("session GC disabled ([serve] idle_timeout = 0)");
        return tokio::spawn(async {});
    }
    tracing::info!(
        "session GC enabled: idle_timeout={:?}, gc_scan_interval={:?}, delete_on_idle={}",
        idle_timeout,
        scan_interval,
        delete_on_idle
    );
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(scan_interval);
        // Skip the immediate first tick: give the server a moment to settle before we start
        // scanning.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            // Unguarded, one panicking scan ends eviction for the life of the process, silently:
            // the task dies, nothing joins it, and sessions accumulate until the operator notices
            // the memory. Log it and take the next tick instead.
            let scan =
                std::panic::AssertUnwindSafe(evict_idle(&state, idle_timeout, delete_on_idle));
            if let Err(panic) = futures::FutureExt::catch_unwind(scan).await {
                tracing::error!(
                    "session GC scan panicked ({}); continuing",
                    crate::error::panic_message(&*panic)
                );
            }
        }
    })
}

async fn evict_idle(state: &ServerState, idle_timeout: Duration, delete_on_idle: bool) {
    // Collect candidates under a brief read lock: don't hold the write lock across the
    // per-row deletion logic.
    let candidates: Vec<uuid::Uuid> = {
        let sessions = state.sessions.read().await;
        // A loop rather than `filter`, because `is_idle` has to await the background-task registry
        // to answer whether this session still owns detached work.
        let mut candidates = Vec::new();
        for (id, entry) in sessions.iter() {
            if entry.is_idle(idle_timeout).await {
                candidates.push(*id);
            }
        }
        candidates
    };
    if candidates.is_empty() {
        return;
    }
    // Truly evicted under the write lock. A turn may have started between read and write
    // acquisition, which would have refreshed `last_turn_at` (or bumped `in_flight`). Only
    // these IDs are eligible for the optional DB-row delete; iterating the original candidate
    // list there would silently destroy an entry whose recheck just decided to keep it.
    let mut evicted: Vec<(uuid::Uuid, crate::server::state::SessionEntry)> =
        Vec::with_capacity(candidates.len());
    {
        let mut sessions = state.sessions.write().await;
        for id in &candidates {
            let still_idle = match sessions.get(id) {
                Some(entry) => entry.is_idle(idle_timeout).await,
                None => false,
            };
            if still_idle && let Some(entry) = sessions.remove(id) {
                evicted.push((*id, entry));
            }
        }
    }
    if evicted.is_empty() {
        return;
    }

    let evicted_ids: Vec<String> = evicted.iter().map(|(id, _)| id.to_string()).collect();
    tracing::info!(
        count = evicted.len(),
        session_ids = %evicted_ids.join(","),
        "session GC: evicted idle session(s)"
    );

    // Detach each evicted session's tool registry from the MCP manager so its
    // `tools/list_changed` callbacks stop targeting a registry that's about to drop. Mirrors
    // `handle_close_session` in `acp.rs`.
    //
    // Read off the hoisted handle, not through `runtime`. Out-of-band turns now mark themselves
    // busy, so `is_idle` keeps one from being evicted mid-run, but this loop still must not wait
    // on a mutex it does not need: anything blocking here stalls the rest of the batch, including
    // the `FileLock`s of sessions already removed from the map but still held by `evicted`,
    // whose owners would get `session-locked` for the duration.
    // `DELETE /v1/sessions/{id}` reads the same handle.
    if let Some(manager) = state.shared.mcp_manager.as_ref() {
        for (_id, entry) in &evicted {
            manager.detach_registry(&entry.tool_registry).await;
        }
    }

    if delete_on_idle {
        for (id, _entry) in &evicted {
            if let Err(error) = state.shared.session_manager.delete_session(*id).await {
                tracing::warn!("session GC: failed to delete row for {}: {}", id, error);
            }
        }
    }
}
