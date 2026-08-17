//! Scheduled-job execution for `meka serve`.
//!
//! This is the durable host for [`crate::schedule`]. The server can revive any session on demand
//! (`reattach::ensure_session_loaded`), so it reaches every job in the database rather than only
//! those belonging to a conversation it happens to have open, which is the REPL's limit. What it
//! leaves alone is a job on a session another process has locked; see [`runnable_here`].
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
    let scope = SchedulerScope::Jobs(runnable_here(&state));
    crate::schedule::spawn(session_manager, config, scope, move |wakeup| {
        let state = state.clone();
        async move { run_wakeup(state, wakeup).await }
    })
}

/// "Could this process run a turn for that session right now?", asked once per due job per sweep.
///
/// The server can revive any session, so taking every job looks right and is not. `prepare`
/// evaluates a job's *gate* before the host is offered the wakeup, and a job whose session another
/// process holds comes straight back as a deferral -- which restores its original fire time,
/// already in the past, so it is due again on the very next tick. A gated hourly job on a session
/// an operator has open in a REPL would therefore run its shell command every `poll_interval` for
/// as long as that REPL stayed open. Declining here instead is precisely what [`SchedulerScope`]
/// documents its predicate variant for.
///
/// An `isolated` job is exempt, because the question does not apply to it: [`run_isolated`] reads
/// the creating session's row and then runs in a conversation of its own, never taking that
/// session's lock. Declining one here would leave it to whichever host holds the lock, and neither
/// the REPL nor ACP honours `isolated` -- both run it in the conversation they have open, with a
/// warning. The exemption keeps serve eligible so the flag can be honoured at all; it does not make
/// serve the winner. Isolated jobs have no cross-host arbitration, so with a REPL open on the same
/// session which host takes a given occurrence is a race between their two tickers. That gap
/// predates this and is not closed here.
///
/// Otherwise there are two ways to be runnable, and the order matters. A resident session is
/// runnable by definition, and has to be checked first because *this* process is the one holding
/// its file lock -- probing would report our own sessions as busy. Everything else is a lock probe:
/// `lock_session` already uses a non-blocking `try_write`, so taking the lock and dropping it is a
/// cheap, synchronous "is anyone else on this".
///
/// The window between this and `ensure_session_loaded` is left open deliberately. A lock taken
/// inside it still produces a deferral, which is correct and costs one gate evaluation; what this
/// removes is paying that on every tick, forever.
fn runnable_here(
    state: &ServerState,
) -> Arc<dyn Fn(&crate::schedule::ScheduledJob) -> bool + Send + Sync> {
    let sessions = Arc::clone(&state.sessions);
    let session_manager = state.shared.session_manager.clone();
    Arc::new(move |job: &crate::schedule::ScheduledJob| {
        runnable(
            job,
            || {
                // `try_read` rather than `read`: the map is briefly write-locked while a session
                // loads, and blocking a sweep behind that is worse than skipping a tick.
                match sessions.try_read() {
                    Ok(open) if open.contains_key(&job.session_id) => Residency::Resident,
                    Ok(_) => Residency::NotResident,
                    Err(_) => Residency::Unknown,
                }
            },
            || match session_manager.lock_session(job.session_id) {
                Ok(_lock) => true,
                Err(crate::error::MekaError::SessionLocked(_)) => false,
                // Not "someone else has it" but "we could not ask": a lock file owned by another
                // user, an unwritable or swept lock directory, file descriptors exhausted.
                // Declining is still the right answer -- a host that cannot take the lock cannot
                // run the turn either -- but this must be loud. The symptom is a
                // *partial* outage: resident sessions and isolated jobs keep firing
                // while everything else silently stops, which looks like nothing
                // being scheduled rather than like a fault. A persistent cause
                // repeating every sweep is noisy on purpose; it is the same
                // reasoning as the `held_over` line below, that a bound on coverage nobody is told
                // about reads as "everything ran".
                Err(error) => {
                    tracing::warn!(
                        "cannot take the session lock for job {}: {}; it will not fire here",
                        job.short_id(),
                        error
                    );
                    false
                }
            },
        )
    })
}

/// Whether a session is one this process currently has loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Residency {
    Resident,
    NotResident,
    /// The session map could not be read without blocking, so we do not know.
    Unknown,
}

/// The rule [`runnable_here`] applies, separated from the state it reads so its short-circuits are
/// under test.
///
/// Both inputs are deferred rather than passed by value, because *not consulting them* is the
/// substance. An isolated job must not reach either: it has no relationship with its creating
/// session beyond one row read. And `lock_is_free` must not be consulted unless the session is
/// already known not to be ours, since a resident session's lock is held by this very process and
/// probing it would report every session we serve as busy.
fn runnable(
    job: &crate::schedule::ScheduledJob,
    residency: impl FnOnce() -> Residency,
    lock_is_free: impl FnOnce() -> bool,
) -> bool {
    if job.isolated {
        return true;
    }
    match residency() {
        Residency::Resident => true,
        Residency::NotResident => lock_is_free(),
        // Declined rather than probed. We may be the holder, and answering "busy" for one sweep
        // costs a tick, where answering "free" would hand the job to a `run_wakeup` that then has
        // to defer it anyway.
        Residency::Unknown => false,
    }
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
            // Supervised for the same reason the scheduler is: this sweep runs a whole agent turn,
            // so anything in the tool loop can panic, and losing the task would stop every
            // background outcome from ever being delivered -- silently, since nothing joins this
            // handle. A task that finished would then sit `delivered_at`-stamped and unreported
            // forever, which is exactly the promise `background.rs` opens by making.
            let sweep = std::panic::AssertUnwindSafe(deliver_ready_outcomes(&state));
            match futures::FutureExt::catch_unwind(sweep).await {
                Ok(std::ops::ControlFlow::Break(())) => return,
                Ok(std::ops::ControlFlow::Continue(())) => {}
                Err(panic) => tracing::error!(
                    "background outcome sweep panicked ({}); continuing",
                    crate::error::panic_message(&*panic)
                ),
            }
        }
    })
}

/// One pass over the resident sessions, delivering whatever outcomes are ready.
///
/// Split out of the loop so it can be caught: a `return` for shutdown inside the loop body cannot
/// survive being wrapped, so the two exits became [`std::ops::ControlFlow`].
async fn deliver_ready_outcomes(state: &ServerState) -> std::ops::ControlFlow<()> {
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
            return std::ops::ControlFlow::Break(());
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
            .background_store()
            .list_undelivered_background_tasks(session_id)
            .await
        {
            Ok(ready) if !ready.is_empty() => ready,
            Ok(_) => continue,
            Err(error) => {
                tracing::warn!(
                    "failed to list undelivered background tasks for session {}: {}",
                    session_id,
                    error
                );
                continue;
            }
        };
        let ids: Vec<String> = ready.iter().map(|task| task.id.clone()).collect();
        // Stamped before the turn, like `stamp_scheduled_job_fired`: an outcome that wedges
        // the process must not be redelivered on every restart.
        if let Err(error) = state
            .shared
            .session_manager
            .background_store()
            .mark_background_tasks_delivered(&ids)
            .await
        {
            tracing::warn!(
                "failed to stamp background outcomes as delivered: {}",
                error
            );
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
            state,
            session_id,
            crate::background::render_outcomes(&ready),
            OutOfBand::BackgroundOutcome,
        )
        .await
        {
            tracing::warn!(
                "background outcome turn for session {} failed: {}",
                session_id,
                error
            );
        }
    }
    std::ops::ControlFlow::Continue(())
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

impl std::fmt::Display for RunError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // Named as a condition rather than a failure: the job is intact and will run wherever
            // the session is resident, which is what an operator reading this needs to know.
            Self::SessionBusyElsewhere => {
                write!(formatter, "its session is held by another process")
            }
            Self::Failed(error) => write!(formatter, "{}", error),
        }
    }
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
    run_prompt_in_session(
        state,
        wakeup.job.session_id,
        wakeup.render_prompt(),
        OutOfBand::ScheduledJob {
            wait,
            retention: wakeup.job.prompt_retention(),
        },
    )
    .await
}

/// Which kind of out-of-band work a turn is delivering.
///
/// One discriminator rather than loose flags at each call site, so what a failure leaves behind is
/// decided in one place. It comes down to whether the prompt can be produced again: a recurring
/// job's fire can, a one-shot's cannot (its row is already deleted), and a background outcome is
/// handed out exactly once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutOfBand {
    /// A scheduled job's fire.
    ///
    /// Neither field follows from the kind. `wait` is about a busy session: see [`run_in_session`]
    /// for why a gated job waits and an ungated one defers. `retention` is about the schedule, and
    /// is the job's own answer -- see [`crate::schedule::ScheduledJob::prompt_retention`].
    ScheduledJob {
        wait: bool,
        retention: crate::agent::PromptRetention,
    },
    /// A finished background task's outcome. Its rows are stamped `delivered` before the turn
    /// starts and are never handed out again, so it waits for the session and keeps its prompt
    /// whatever happens. Its caller pre-checks for a busy session, which keeps that wait from
    /// becoming a queue in practice.
    BackgroundOutcome,
}

/// Run one out-of-band prompt inside a live session, publishing a cancellation token so
/// `POST /cancel` can reach the turn.
async fn run_prompt_in_session(
    state: &ServerState,
    session_id: uuid::Uuid,
    prompt: String,
    kind: OutOfBand,
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

    let mut runtime = match kind {
        OutOfBand::BackgroundOutcome | OutOfBand::ScheduledJob { wait: true, .. } => {
            entry.runtime.lock().await
        }
        OutOfBand::ScheduledJob { wait: false, .. } => match entry.runtime.try_lock() {
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
    let outcome = match kind {
        OutOfBand::ScheduledJob { retention, .. } => {
            runtime_inner
                .agent
                .run_turn_retaining(
                    &mut session_uuid,
                    &mut runtime_inner.messages,
                    prompt,
                    Vec::new(),
                    cancellation,
                    retention,
                )
                .await
        }
        OutOfBand::BackgroundOutcome => {
            runtime_inner
                .agent
                .run_turn(
                    &mut session_uuid,
                    &mut runtime_inner.messages,
                    prompt,
                    Vec::new(),
                    cancellation,
                )
                .await
        }
    };
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
    let cwd: crate::workspace::SharedCwd = Arc::new(std::sync::RwLock::new(
        summary
            .cwd
            .clone()
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| std::path::PathBuf::from(".")),
    ));
    let roots: crate::workspace::SharedRoots = Arc::new(std::sync::RwLock::new(Vec::new()));
    // Counted as an in-flight turn for the whole of the fire.
    //
    // `wait_for_turns_to_unwind` polls `state.concurrent_turns` and each session entry's
    // `in_flight`; an isolated fire creates a session that is never in `state.sessions` and took no
    // guard, so the drain saw an idle process and aborted this turn mid-flight -- undoing the
    // shutdown fix for precisely the unattended work it was written to protect.
    let _drain_guard =
        crate::server::state::TurnGuard::mark_process_busy(Arc::clone(&state.concurrent_turns));

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

    // `run_turn` rather than `run_scheduled_turn`, though this is a scheduled fire. Withdrawing the
    // prompt here would leave an *empty* session row, which is worse noise in `meka session list`
    // than a session holding one unanswered message -- and unlike an in-session fire, a failure
    // costs a row rather than a share of a conversation the user has to keep reading. Retention is
    // the lever for these, not withdrawal.
    //
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

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::{Residency, runnable};

    fn job(isolated: bool) -> crate::schedule::ScheduledJob {
        crate::schedule::ScheduledJob {
            id: "7f3a1b2c".to_string(),
            session_id: uuid::Uuid::nil(),
            schedule: crate::schedule::Schedule::parse_every("1h").expect("parses"),
            prompt: "check the news".to_string(),
            gate: None,
            isolated,
            created_at: chrono::Utc::now(),
            last_fired_at: None,
            next_fire_at: chrono::Utc::now(),
        }
    }

    /// The ordering that makes the probe safe. Every session `meka serve` has loaded is one whose
    /// file lock *this process* holds, so probing a resident session would report it busy and the
    /// server would stop running its own jobs.
    #[test]
    fn test_a_resident_session_is_never_probed() {
        let probed = Cell::new(false);
        let takeable = runnable(
            &job(false),
            || Residency::Resident,
            || {
                probed.set(true);
                false
            },
        );
        assert!(takeable, "a session we already hold is ours to run");
        assert!(
            !probed.get(),
            "and asking the lock would have said otherwise"
        );
    }

    /// The case this exists for: a session an operator has open in a REPL. Declining here is what
    /// keeps `prepare` from evaluating the job's gate, which is the cost the deferral path could
    /// not avoid -- it runs the command first and finds out afterwards.
    #[test]
    fn test_a_session_another_process_holds_is_declined() {
        assert!(!runnable(&job(false), || Residency::NotResident, || false));
        assert!(
            runnable(&job(false), || Residency::NotResident, || true),
            "a free lock is ours to take"
        );
    }

    /// An isolated job runs in a conversation of its own and never touches the session that created
    /// it, so that session's lock says nothing about whether this host can run the job. Declining
    /// would hand it to whatever holds the lock -- a REPL, which does not honour `isolated` and
    /// would fire it into the operator's conversation instead.
    #[test]
    fn test_an_isolated_job_is_not_gated_on_its_creating_session() {
        let consulted = Cell::new(false);
        let takeable = runnable(
            &job(true),
            || {
                consulted.set(true);
                Residency::NotResident
            },
            || {
                consulted.set(true);
                false
            },
        );
        assert!(takeable, "the creating session's state is beside the point");
        assert!(!consulted.get(), "so neither input is even read");
        // And the rule it bypasses really would have refused an ordinary job in the same state.
        assert!(!runnable(&job(false), || Residency::NotResident, || false));
    }

    /// An unreadable session map means we cannot rule out being the holder, so the probe would be
    /// unsound. Skipping the tick is free: the job keeps its occurrence and comes back.
    #[test]
    fn test_an_unreadable_session_map_declines_rather_than_probing() {
        let probed = Cell::new(false);
        let takeable = runnable(
            &job(false),
            || Residency::Unknown,
            || {
                probed.set(true);
                true
            },
        );
        assert!(!takeable);
        assert!(
            !probed.get(),
            "probing would be unsound when we may be the holder"
        );
    }
}
