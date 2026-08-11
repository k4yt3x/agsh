//! Scheduled-job execution for `meka serve`.
//!
//! This is the durable host for [`crate::schedule`]. The server can revive any session on demand
//! (`reattach::ensure_session_loaded`), so it owns every job in the database rather than only those
//! belonging to a conversation it happens to have open, which is the REPL's limit.
//!
//! There is no human on the far end of a scheduled turn. Two consequences run through everything
//! here: `Ask`-mode approvals resolve to deny (`SilentFrontend`), and the turn's only durable
//! output is the `messages` rows `Agent::run_turn` writes. A client sees the result by reading the
//! session back. When a push API lands, this is where it hooks in.

use std::sync::Arc;

use crate::{
    conversation::Conversation,
    frontend::SilentFrontend,
    schedule::{FireOutcome, SchedulerScope, Wakeup},
    server::{reattach, state::ServerState},
};

/// Start the scheduler for a running server. Returns the handle so `run_serve` can abort it during
/// shutdown, exactly as it does for the GC scanner.
pub fn spawn(state: ServerState) -> tokio::task::JoinHandle<()> {
    let config = state.shared.config.schedule.clone();
    if !config.enabled {
        tracing::info!("scheduler disabled ([schedule] enabled = false)");
        return tokio::spawn(async {});
    }
    tracing::info!(
        "scheduler enabled: poll_interval={:?}, missed_grace={:?}",
        config.poll_interval,
        config.missed_grace
    );
    let session_manager = Arc::new(state.shared.session_manager.clone());
    crate::schedule::spawn(
        session_manager,
        config,
        SchedulerScope::AllSessions,
        move |wakeup| {
            let state = state.clone();
            async move { run_wakeup(state, wakeup).await }
        },
    )
}

/// Run one fired job. Errors are logged rather than propagated: the scheduler must survive a job
/// whose session has gone missing or whose provider call failed.
async fn run_wakeup(state: ServerState, wakeup: Wakeup) -> FireOutcome {
    let job_id = wakeup.job.short_id().to_string();
    let outcome = if wakeup.job.isolated {
        run_isolated(&state, &wakeup).await
    } else {
        run_in_session(&state, &wakeup).await
    };
    match outcome {
        Ok(()) => {
            tracing::info!("scheduled job {} completed", job_id);
            FireOutcome::Ran
        }
        Err(RunError::SessionBusyElsewhere) => {
            // Another process holds this session -- in practice a REPL the operator has open. It
            // runs its own watcher and will take the job, so the occurrence goes back rather than
            // being spent on a turn that never happened.
            tracing::debug!(
                "scheduled job {} belongs to a session held by another process; deferring",
                job_id
            );
            FireOutcome::Deferred
        }
        Err(RunError::Failed(error)) => {
            tracing::warn!("scheduled job {} failed: {}", job_id, error);
            FireOutcome::Ran
        }
    }
}

/// Why a scheduled run did not complete. The distinction exists solely to separate "this host
/// cannot take the job" (retry elsewhere) from "the job ran and went wrong" (do not retry).
enum RunError {
    SessionBusyElsewhere,
    Failed(anyhow::Error),
}

impl From<anyhow::Error> for RunError {
    fn from(error: anyhow::Error) -> Self {
        Self::Failed(error)
    }
}

impl From<crate::error::MekaError> for RunError {
    fn from(error: crate::error::MekaError) -> Self {
        Self::Failed(error.into())
    }
}

/// Deliver the prompt into the conversation that created the job.
async fn run_in_session(state: &ServerState, wakeup: &Wakeup) -> Result<(), RunError> {
    let entry = reattach::ensure_session_loaded(state, wakeup.job.session_id)
        .await
        .map_err(|problem| {
            if problem.type_uri == crate::server::errors::ErrorKind::SessionLocked.type_uri() {
                RunError::SessionBusyElsewhere
            } else {
                RunError::Failed(anyhow::anyhow!("session unavailable: {}", problem.title))
            }
        })?;

    // Wait for the session rather than skipping, unlike `POST /turn`, which rejects a busy session
    // with 409. A caller can retry; a fired job cannot -- the scheduler has already advanced its
    // next-fire time, so a skip here loses the occurrence outright. The scheduler awaits each fire
    // in turn, so this only ever delays the tick.
    let mut runtime = entry.runtime.lock().await;

    // Publish the token so `POST /v1/sessions/{id}/cancel` reaches a scheduled turn. It reads
    // `entry.cancellation` (`handlers::turn::cancel_turn`), so a turn that only held a shutdown
    // child token could be stopped by killing the process and no other way.
    let cancellation = state.shutdown.child_token();
    {
        let mut slot =
            crate::server::poisoned::write(&entry.cancellation, "schedule::publish_cancellation");
        *slot = cancellation.clone();
    }

    let mut session_uuid = Some(runtime.session_uuid);
    let runtime_inner = &mut *runtime;
    let outcome = runtime_inner
        .agent
        .run_turn(
            &mut session_uuid,
            &mut runtime_inner.messages,
            wakeup.render_prompt(),
            Vec::new(),
            cancellation,
        )
        .await;
    entry.touch();
    // Whatever the frontend recorded belongs to nobody: the events accumulate in the session's
    // `HttpFrontend` and would otherwise be handed to the next client turn as if they were its own.
    // `run_blocking_turn` drains defensively at its start, but clearing here keeps the invariant
    // local to the code that broke it.
    let _scheduled_turn_events = entry.frontend.drain();
    outcome?;
    Ok(())
}

/// Run the prompt in a fresh session, leaving the creating conversation untouched.
///
/// A new top-level session rather than a sub-agent: the run is then a first-class thing
/// `meka session list` can show and `meka session export` can dump, where a sub-agent's report
/// would have nowhere to go once the tool call that would have carried it does not exist.
async fn run_isolated(state: &ServerState, wakeup: &Wakeup) -> Result<(), RunError> {
    // The creating session's permission, not the process default: an isolated run is the same job
    // with a cheaper context, and should not quietly gain authority by moving house.
    //
    // Read from the row rather than through `ensure_session_loaded`, which would rebuild the whole
    // runtime -- agent, tool registry, MCP attachment -- and leave it resident. For a job firing on
    // a short interval that would pin the parent session in memory permanently, which is the exact
    // cost `isolated` exists to avoid.
    let summary = state
        .shared
        .session_manager
        .session_info(wakeup.job.session_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("session {} no longer exists", wakeup.job.session_id))?;
    let enabled = state.shared.config.enabled_permissions;
    let permission = summary
        .permission
        .as_deref()
        .and_then(|value| value.parse().ok())
        .filter(|parsed| enabled.is_enabled(*parsed))
        .unwrap_or(state.shared.config.permission);
    let permission = crate::permission::SharedPermission::new(permission, enabled);

    // The creating session's directory, for the same reason as its permission: an isolated run is
    // the same job with a cheaper context. Falling through to the process cwd would put the shell
    // and file tools wherever the `meka serve` unit was launched from, which under systemd is `/`.
    let cwd: crate::agent::SharedCwd = Arc::new(std::sync::RwLock::new(
        summary
            .cwd
            .clone()
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| std::path::PathBuf::from(".")),
    ));
    let roots: crate::agent::SharedRoots = Arc::new(std::sync::RwLock::new(Vec::new()));
    let (agent, _registry) = crate::build_session_agent(
        &state.shared,
        permission,
        Arc::new(SilentFrontend),
        cwd,
        roots,
    )
    .await?;

    // `None` makes `run_turn` create the session row, so the isolated run gets its own id.
    let mut session_uuid = None;
    let mut messages = Conversation::new();
    agent
        .run_turn(
            &mut session_uuid,
            &mut messages,
            wakeup.render_prompt(),
            Vec::new(),
            state.shutdown.child_token(),
        )
        .await?;
    if let Some(id) = session_uuid {
        tracing::info!(
            "scheduled job {} ran in isolated session {}",
            wakeup.job.short_id(),
            id
        );
    }
    Ok(())
}
