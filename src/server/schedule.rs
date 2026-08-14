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

/// Start the background-outcome poller. Separate task from the scheduler because the two are
/// independent switches: an installation can want timers without detached work, or the reverse.
///
/// Only sessions already resident are polled. Reviving an evicted session to deliver an outcome
/// would rebuild its whole runtime and pin it in memory, and the outcome is not going anywhere: it
/// keeps its `delivered_at IS NULL` until something opens that session again. The session-load
/// sweep is what guarantees the row exists to be found.
pub fn spawn_background_poller(state: ServerState) -> tokio::task::JoinHandle<()> {
    let config = state.shared.config.background.clone();
    if !config.enabled {
        return tokio::spawn(async {});
    }
    let poll_interval = state.shared.config.schedule.poll_interval;
    tracing::info!(
        "background tasks enabled: max_tasks={}, poll_interval={:?}",
        config.max_tasks,
        poll_interval
    );
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(poll_interval);
        ticker.tick().await;
        loop {
            tokio::select! {
                _ = state.shutdown.cancelled() => return,
                _ = ticker.tick() => {}
            }
            // Snapshot the ids and drop the lock before any await: holding the sessions map
            // across a turn would block every request that needs to look a session up.
            let resident: Vec<uuid::Uuid> = {
                let sessions = state.sessions.read().await;
                sessions.keys().copied().collect()
            };
            for session_id in resident {
                // Re-checked per session, not just per tick. Each iteration stamps its outcomes
                // delivered *before* awaiting a turn that can take minutes, which is right for a
                // turn that wedges but fatal across a shutdown: the sessions later in this loop
                // would be stamped and then never processed, and
                // `list_undelivered_background_tasks` never returns a stamped row again, so the
                // model would never learn its task finished.
                if state.shutdown.is_cancelled() {
                    return;
                }
                // Skipped rather than queued behind. This sweep is serial, so waiting on one busy
                // session delays every later session's outcomes in the same tick. Checked *before
                // anything is stamped*, so the next tick picks this session up unchanged -- which
                // is why the delivery below can afford to wait if a turn starts in the gap.
                // A session that has left the map since the snapshot counts as un-takeable too,
                // not as idle: falling through would stamp its outcomes and then have
                // `ensure_session_loaded` rebuild the whole runtime and pin the file lock, which
                // is exactly the revival this poller documents itself as never doing.
                if state
                    .sessions
                    .read()
                    .await
                    .get(&session_id)
                    .is_none_or(|entry| entry.runtime.try_lock().is_err())
                {
                    continue;
                }
                let ready = match state
                    .shared
                    .session_manager
                    .list_undelivered_background_tasks(session_id)
                    .await
                {
                    Ok(ready) if !ready.is_empty() => ready,
                    Ok(_) => continue,
                    Err(error) => {
                        tracing::warn!("background poller failed for {}: {}", session_id, error);
                        continue;
                    }
                };
                let ids: Vec<String> = ready.iter().map(|task| task.id.clone()).collect();
                // Stamped before the turn, like `stamp_scheduled_job_fired`: an outcome that wedges
                // the process must not be redelivered on every restart.
                if let Err(error) = state
                    .shared
                    .session_manager
                    .mark_background_tasks_delivered(&ids)
                    .await
                {
                    tracing::warn!("failed to stamp background outcomes delivered: {}", error);
                    continue;
                }
                // Fired before the delivery turn rather than after it: the fact a task finished
                // is the news, and it should not wait on a model call that may itself fail.
                for task in &ready {
                    state.webhooks.send(
                        crate::server::webhook::WebhookEvent::TaskFinished,
                        // No `label`. It is the tool's primary argument, which for
                        // `execute_command` is the shell command line -- the highest-entropy
                        // field in the system and the one most likely to carry a credential
                        // someone pasted into a `curl`. A subscriber that wants it reads
                        // `GET /v1/sessions/{id}/tasks` with its own token, which is the whole
                        // reason deliveries carry identifiers rather than content.
                        serde_json::json!({
                            "task_id": task.id,
                            "session_id": task.session_id,
                            "tool_name": task.tool_name,
                            "status": task.status.as_str(),
                        }),
                    );
                }
                if let Err(error) = run_prompt_in_session(
                    &state,
                    session_id,
                    crate::background::render_outcomes(&ready),
                    true,
                )
                .await
                {
                    tracing::warn!(
                        "background outcome turn for {} failed: {:?}",
                        session_id,
                        error
                    );
                }
            }
        }
    })
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
    // Announced whatever the outcome, because "the 3am job ran and failed" and "the 3am job never
    // fired" need very different responses from whoever is watching, and without this both look
    // like silence. Deferral is excluded deliberately: the occurrence was handed back, not spent,
    // and another host is about to run it.
    let notify = |status: &str| {
        state.webhooks.send(
            crate::server::webhook::WebhookEvent::ScheduleFired,
            serde_json::json!({
                "job_id": wakeup.job.id,
                "session_id": wakeup.job.session_id,
                "isolated": wakeup.job.isolated,
                "status": status,
            }),
        );
    };
    // Checked before the outcome is classified, because a drain makes every outcome a lie. The
    // scheduler keeps ticking until it is aborted, which happens *after* the drain window, and by
    // then `state.shutdown` has already been fired -- so the turn's token starts cancelled,
    // `run_turn` returns `Interrupted` immediately, and this would score a job that never ran as
    // `Ran`. `prepare` has already deleted a one-shot's row or advanced a recurring job's
    // `next_fire_at`, so scoring it `Ran` spends the occurrence outright: the 3am job is gone
    // because someone deployed at 3am. Deferring hands it back, which is exactly what the
    // `SessionBusyElsewhere` arm below exists to do for the other "this host cannot take it now".
    if state.shutdown.is_cancelled() {
        tracing::info!(
            "scheduled job {} fired during shutdown; deferring the occurrence",
            job_id
        );
        return FireOutcome::Deferred;
    }
    match outcome {
        Ok(()) => {
            tracing::info!("scheduled job {} completed", job_id);
            notify("completed");
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
            notify("failed");
            FireOutcome::Ran
        }
    }
}

/// Why a scheduled run did not complete. The distinction exists solely to separate "this host
/// cannot take the job" (retry elsewhere) from "the job ran and went wrong" (do not retry).
#[derive(Debug)]
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
///
/// An ungated job does not wait: the sweep fires jobs one at a time, so queueing behind a session
/// that is mid-turn holds up every *other* session's jobs in the same tick, and a recurring job
/// that misses its slot has its occurrences coalesced rather than run. Deferring hands the
/// occurrence back for the next tick instead.
///
/// A *gated* job does wait, because deferral is not free for it. `prepare` evaluates the gate
/// before this host gets a say, and restoring the job puts its fire time back in the past, so it
/// comes due again on the very next tick: a gated hourly job on a session busy for ten minutes
/// would run its shell command every `poll_interval` instead of once. Blocking the sweep is the
/// lesser cost, and it is what this did before deferral existed.
async fn run_in_session(state: &ServerState, wakeup: &Wakeup) -> Result<(), RunError> {
    let wait = wakeup.job.gate.is_some();
    run_prompt_in_session(state, wakeup.job.session_id, wakeup.render_prompt(), wait).await
}

/// Run one out-of-band prompt inside a live session, publishing a cancellation token so
/// `POST /cancel` can reach the turn.
///
/// `wait` decides what a busy session means. A scheduled job defers and is retried whole on the
/// next tick, so it does not wait. Background-outcome delivery has already stamped its rows
/// `delivered` by the time it gets here and cannot be retried, so it does wait -- its caller
/// pre-checks instead, which keeps that from becoming a queue in practice.
async fn run_prompt_in_session(
    state: &ServerState,
    session_id: uuid::Uuid,
    prompt: String,
    wait: bool,
) -> Result<(), RunError> {
    let entry = reattach::ensure_session_loaded(state, session_id)
        .await
        .map_err(|problem| {
            if problem.type_uri == crate::server::errors::ErrorKind::SessionLocked.type_uri() {
                RunError::SessionBusyElsewhere
            } else {
                RunError::Failed(anyhow::anyhow!("session unavailable: {}", problem.title))
            }
        })?;

    let mut runtime = match wait {
        true => entry.runtime.lock().await,
        false => match entry.runtime.try_lock() {
            Ok(runtime) => runtime,
            // Reported as busy rather than waited out. `prepare` has already advanced the job's
            // next-fire time, so the caller turns this into a deferral, which hands the occurrence
            // back intact -- the same treatment a session held by another process gets.
            Err(_) => return Err(RunError::SessionBusyElsewhere),
        },
    };

    // Marked busy for the length of the turn. An out-of-band turn is still a turn: without this
    // the counter reads zero while the agent is mid-tool, so `GET /v1/sessions/{id}` reports
    // `turn_in_flight: false`, `PATCH` slips a permission change into a running unattended turn,
    // and `DELETE` deletes the row out from under it -- all three past guards that exist to stop
    // exactly that. Taken after the lock, so it never contends with the turn it is describing.
    let _busy = crate::server::state::InFlightGuard::mark_busy(&entry);

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
            prompt,
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
    let (agent, registry) = crate::build_session_agent(
        &state.shared,
        permission,
        Arc::new(SilentFrontend),
        cwd,
        roots,
        // The HTTP surface has no context gauge of its own; the session owns the counter.
        Arc::new(std::sync::atomic::AtomicU64::new(0)),
        Arc::new(std::sync::atomic::AtomicU64::new(0)),
    )
    .await?;

    // `None` makes `run_turn` create the session row, so the isolated run gets its own id.
    let mut session_uuid = None;
    let mut messages = Conversation::new();
    let outcome = agent
        .run_turn(
            &mut session_uuid,
            &mut messages,
            wakeup.render_prompt(),
            Vec::new(),
            state.shutdown.child_token(),
        )
        .await;

    // Before the `?`, so a failed run cleans up too. `build_session_agent` hands the registry to
    // the MCP manager, which holds a *strong* clone; an isolated agent is discarded at the end of
    // this function, so without a detach every fire leaves one behind. An hourly isolated job adds
    // 24 a day, indefinitely, each pinning a whole dead tool set and lengthening every
    // `tools/list_changed` fan-out. Sessions get this from `handle_close_session`, the GC and
    // `DELETE /v1/sessions/{id}`; an isolated fire has no such moment but the end of its own turn.
    if let Some(manager) = state.shared.mcp_manager.as_ref() {
        manager.detach_registry(&registry).await;
    }
    outcome?;

    if let Some(id) = session_uuid {
        tracing::info!(
            "scheduled job {} ran in isolated session {}",
            wakeup.job.short_id(),
            id
        );
    }
    Ok(())
}
