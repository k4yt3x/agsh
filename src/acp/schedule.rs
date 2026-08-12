//! Scheduled-job execution for `meka acp`.
//!
//! An editor is a live client, so unlike `meka serve` this host has a human on the far side. Two
//! things follow. `ask`-mode approvals genuinely round-trip (`session/request_permission` reaches
//! the editor), rather than resolving to deny the way an unattended turn must. And the job's prompt
//! is pushed as a `UserMessageChunk` before the turn runs, so the transcript shows what triggered
//! the reply instead of an answer to a question nobody asked.
//!
//! ACP fires only jobs whose session the editor currently has open. It cannot revive an arbitrary
//! session the way the server can: a session's cwd, permission and capabilities come from the
//! client's `session/new`, and inventing them here would run a job against a workspace the editor
//! never opened. Jobs for other sessions are simply not this host's to run, and the scope predicate
//! filters them out before their gates are ever evaluated.

use std::sync::Arc;

use agent_client_protocol::schema::v1::{ContentBlock, ContentChunk, SessionUpdate, TextContent};

use crate::schedule::{FireOutcome, SchedulerScope, Wakeup};

/// Start the scheduler for a running ACP process. Returns the handle so the caller can abort it on
/// shutdown, matching how `meka serve` treats its own.
pub(super) fn spawn(state: Arc<super::ServerState>) -> tokio::task::JoinHandle<()> {
    let config = state.shared.config.schedule.clone();
    if !config.enabled {
        tracing::info!("scheduler disabled ([schedule] enabled = false)");
        return tokio::spawn(async {});
    }

    // Re-asked every sweep, so a session opened or closed since the last tick is accounted for
    // without the scheduler holding a stale snapshot. The map is keyed by the session's uuid in
    // string form (every insertion site derives the key from `session_uuid.to_string()`), so this
    // needs no lock on any session's runtime.
    let sessions = Arc::clone(&state.sessions);
    let scope = SchedulerScope::Sessions(Arc::new(move |session_uuid| {
        // `try_read` rather than `read`: the map is briefly write-locked on every `session/new` and
        // `session/close`, and blocking the sweep behind session setup would be worse than skipping
        // a tick. The job stays due either way.
        sessions
            .try_read()
            .map(|open| open.contains_key(&session_uuid.to_string()))
            .unwrap_or(false)
    }));

    tracing::info!(
        "scheduler enabled for ACP: poll_interval={:?}, open sessions only",
        config.poll_interval
    );
    let session_manager = Arc::new(state.shared.session_manager.clone());
    crate::schedule::spawn(session_manager, config, scope, move |wakeup| {
        let state = Arc::clone(&state);
        async move { run_wakeup(state, wakeup).await }
    })
}

/// Start the background-outcome poller for a running ACP process.
///
/// Same limit as the scheduler above: only sessions the editor currently has open. A session it
/// closed is not this host's to revive, and the outcome keeps its `delivered_at IS NULL` until
/// something opens it again.
pub(super) fn spawn_background_poller(
    state: Arc<super::ServerState>,
) -> tokio::task::JoinHandle<()> {
    if !state.shared.config.background.enabled {
        return tokio::spawn(async {});
    }
    let poll_interval = state.shared.config.schedule.poll_interval;
    tracing::info!("background tasks enabled for ACP: open sessions only");
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(poll_interval);
        ticker.tick().await;
        loop {
            ticker.tick().await;
            let open: Vec<(String, super::SessionEntry)> = {
                let sessions = state.sessions.read().await;
                sessions
                    .iter()
                    .map(|(key, entry)| (key.clone(), entry.clone()))
                    .collect()
            };
            for (key, entry) in open {
                let Ok(session_uuid) = uuid::Uuid::parse_str(&key) else {
                    continue;
                };
                let ready = match state
                    .shared
                    .session_manager
                    .list_undelivered_background_tasks(session_uuid)
                    .await
                {
                    Ok(ready) if !ready.is_empty() => ready,
                    Ok(_) => continue,
                    Err(error) => {
                        tracing::warn!("background poller failed for {}: {}", session_uuid, error);
                        continue;
                    }
                };
                let ids: Vec<String> = ready.iter().map(|task| task.id.clone()).collect();
                if let Err(error) = state
                    .shared
                    .session_manager
                    .mark_background_tasks_delivered(&ids)
                    .await
                {
                    tracing::warn!("failed to stamp background outcomes delivered: {}", error);
                    continue;
                }
                deliver_outcomes(&entry, crate::background::render_outcomes(&ready)).await;
            }
        }
    })
}

/// Run one outcome report inside an open session, mirroring `run_wakeup`'s ordering: take the lock
/// first, then show the prompt, so the transcript reads trigger-then-reply and the report never
/// lands in the middle of a turn the user typed.
async fn deliver_outcomes(entry: &super::SessionEntry, prompt: String) {
    let mut runtime = entry.runtime.lock().await;
    entry.frontend.push_out_of_band_prompt(&prompt);

    let cancellation = tokio_util::sync::CancellationToken::new();
    entry.publish_cancellation(cancellation.clone());

    let mut session_uuid = Some(runtime.session_uuid);
    let runtime_inner = &mut *runtime;
    if let Err(error) = runtime_inner
        .agent
        .run_turn(
            &mut session_uuid,
            &mut runtime_inner.messages,
            prompt,
            Vec::new(),
            cancellation,
        )
        .await
    {
        tracing::warn!("background outcome turn failed: {}", error);
    }
}

/// Run one fired job in the session the editor has open for it.
async fn run_wakeup(state: Arc<super::ServerState>, wakeup: Wakeup) -> FireOutcome {
    let job_id = wakeup.job.short_id().to_string();

    // The scope predicate said this session was open, but that was a separate lock acquisition and
    // the editor may have closed it since. Deferring rather than failing puts the occurrence back
    // for whichever host picks the session up next.
    let Some(entry) = state
        .sessions
        .read()
        .await
        .get(&wakeup.job.session_id.to_string())
        .cloned()
    else {
        tracing::debug!(
            "session for job {} closed before it could run; deferring",
            job_id
        );
        return FireOutcome::Deferred;
    };

    if wakeup.job.isolated {
        // Same limit as the REPL: this agent belongs to a session the editor created, with its own
        // cwd and capabilities, and an isolated run would need a session the client never asked for
        // and has no window for. Said out loud rather than silently downgraded.
        tracing::warn!(
            "job {} asked for an isolated session; ACP runs it in the open conversation instead. \
             Run `meka serve` for isolated jobs.",
            job_id
        );
    }

    // Taken before the prompt is shown, not after. The editor may be part-way through a
    // `session/prompt` the user typed, in which case this waits; announcing first would drop the
    // job's prompt into the middle of that turn's output and then leave it unanswered until the
    // lock came free.
    let mut runtime = entry.runtime.lock().await;

    // Now that the session is ours, the transcript reads in order: the trigger, then the reply.
    entry.frontend.push_scheduled_prompt(&wakeup);

    // Publish the token the way the prompt handler does, so `session/cancel` from the editor stops
    // a scheduled turn. Under the lock, so a turn already running cannot have its token replaced.
    let cancellation = tokio_util::sync::CancellationToken::new();
    entry.publish_cancellation(cancellation.clone());

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

    match outcome {
        Ok(_) => tracing::info!("scheduled job {} completed", job_id),
        Err(error) => tracing::warn!("scheduled job {} failed: {}", job_id, error),
    }
    // Either way the job ran: a failed turn is not something a different host would do better, and
    // retrying a prompt that errored would just spend the failure again.
    FireOutcome::Ran
}

/// Build the `session/update` that shows a job's prompt as the user turn that triggered the reply.
///
/// Zed skips an echoed `UserMessageChunk` only when it matches a message it optimistically added
/// itself before calling `session/prompt`. A scheduled turn has no such message, so this renders --
/// and without it the editor would show an answer with nothing above it explaining the question.
pub(super) fn out_of_band_prompt_update(prompt: &str) -> SessionUpdate {
    SessionUpdate::UserMessageChunk(ContentChunk::new(ContentBlock::Text(TextContent::new(
        prompt.to_string(),
    ))))
}
