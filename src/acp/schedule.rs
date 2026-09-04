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
    let scope = SchedulerScope::Jobs(Arc::new(move |job: &crate::schedule::ScheduledJob| {
        // `try_read` rather than `read`: the map is briefly write-locked on every `session/new` and
        // `session/close`, and blocking the sweep behind session setup would be worse than skipping
        // a tick. The job stays due either way.
        sessions
            .try_read()
            .map(|open| open.contains_key(&job.session_id.to_string()))
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
                    .background_store()
                    .list_undelivered_background_tasks(session_uuid)
                    .await
                {
                    // A cancellation never causes a turn, but joins one that is happening anyway:
                    // see `TaskStatus::wakes_a_host`. Judged here rather than inside
                    // `deliver_outcomes`, so a batch that will wait costs no `touch`, no runtime
                    // lock and no binding resolve.
                    Ok(ready) if ready.iter().any(|task| task.status.wakes_a_host()) => ready,
                    Ok(_) => continue,
                    Err(error) => {
                        tracing::warn!(
                            "failed to list undelivered background tasks for session {}: {}",
                            session_uuid,
                            error
                        );
                        continue;
                    }
                };
                // Supervised for the reason `meka serve`'s `spawn_background_poller` documents:
                // this runs a whole agent turn, so anything in the tool loop can panic, and
                // nothing joins this handle -- losing it would strand the batch that was just
                // stamped `delivered` and every outcome after it, with no error anywhere.
                let sweep = std::panic::AssertUnwindSafe(deliver_outcomes(&state, &entry, &ready));
                if let Err(panic) = futures::FutureExt::catch_unwind(sweep).await {
                    tracing::error!(
                        "background outcome delivery panicked ({}); continuing",
                        crate::error::panic_message(&*panic)
                    );
                }
            }
        }
    })
}

/// Run one outcome report inside an open session, mirroring `run_wakeup`'s ordering: take the lock
/// first, then show the prompt, so the transcript reads trigger-then-reply and the report never
/// lands in the middle of a turn the user typed.
///
/// Stamps the batch `delivered` itself, and only once the turn is certain to run on the right
/// profile. Stamping in the caller instead meant a binding failure discarded outcomes the user was
/// waiting on: `list_undelivered_background_tasks` filters on `delivered_at IS NULL`, so there is
/// no re-delivery path, and the failure is not always permanent -- `recorded_provider` and
/// `credential_for` are both database reads that can come back `SQLITE_BUSY` when another meka
/// process is mid-write. Stamped here, that tick drops the batch and the next one picks it up.
///
/// Still stamped *before* the turn, which is the deliberate part and unchanged: an outcome that
/// reliably wedges the process must not be redelivered on every restart.
async fn deliver_outcomes(
    state: &Arc<super::ServerState>,
    entry: &super::SessionEntry,
    ready: &[crate::background::BackgroundTask],
) {
    // A turn is activity whoever started it. Without this the idle sweep sees a session whose only
    // traffic is out-of-band as untouched since the user last typed, and evicts it after 24h --
    // taking its schedule out of scope permanently, since `run_due` skips jobs whose session is not
    // in the live map. `meka serve` stamps on this same path, in `run_prompt_in_session`.
    entry.touch();
    // Waits, where `meka serve`'s twin skips a busy session and takes it next tick. An editor
    // serves few sessions and a report the user is waiting on should not be deferred a whole
    // interval; the cost is that one session mid-prompt stalls the rest of this sweep.
    let mut runtime = entry.runtime.lock().await;

    // This turn is a turn like any other, so it runs on the profile the row names -- not on
    // whichever one the agent happened to be assembled with. Returned before the batch is stamped,
    // so the outcomes survive to the next tick instead of being destroyed by a failure that may be
    // a transient `SQLITE_BUSY`.
    if let Err(error) = super::apply_recorded_binding(state, &mut runtime).await {
        tracing::warn!(
            "holding a background outcome report for session {}: its recorded provider could not \
             be resolved. It is retried on the next sweep: {}",
            runtime.session_uuid,
            error
        );
        return;
    }

    // Asked before the stamp, for the reason the serve twin gives: `run_turn` gates on MCP
    // readiness as its first statement, so a required server that is down would leave every sweep
    // with a stamped batch and no turn to report it.
    if !crate::background::a_turn_can_carry_them(&runtime.agent).await {
        return;
    }
    let ids: Vec<String> = ready.iter().map(|task| task.id.clone()).collect();
    // The stamp is the claim, and `ready` was listed before the lock above was taken: a
    // `session/prompt` that won the mutex in that gap has already claimed and reported some of
    // these. Reporting what this call actually won is what stops the model hearing it twice.
    let claimed = match state
        .shared
        .session_manager
        .background_store()
        .mark_background_tasks_delivered(&ids)
        .await
    {
        Ok(claimed) => claimed,
        Err(error) => {
            tracing::warn!(
                "failed to stamp background outcomes as delivered: {}",
                error
            );
            return;
        }
    };
    let ready = crate::background::only_what_was_won(ready.to_vec(), &claimed);
    if ready.is_empty() {
        return;
    }

    let prompt = crate::background::render_outcomes(&ready);
    entry.frontend.push_out_of_band_prompt(&prompt);

    let cancellation = tokio_util::sync::CancellationToken::new();
    let _turn_cancellation = entry.publish_cancellation(cancellation.clone());

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

    // A fire is activity on this session; see `deliver_outcomes`. A schedule whose session is only
    // ever driven by the scheduler would otherwise be evicted as idle and never fire again.
    entry.touch();

    // Taken before the prompt is shown, not after. The editor may be part-way through a
    // `session/prompt` the user typed, in which case this waits; announcing first would drop the
    // job's prompt into the middle of that turn's output and then leave it unanswered until the
    // lock came free.
    let mut runtime = entry.runtime.lock().await;

    // The row names the profile this session runs on, and a fire is a turn on that session. Without
    // this the job ran on whichever profile the agent was assembled with: a switch made through
    // `session/set_config_option` while a turn was in flight reached `session/prompt` and nothing
    // else, so the scheduler kept billing the account the user had left.
    //
    // `Unrunnable`, which is neither of the other two on purpose. `Deferred` means "another host
    // should take this" and releases the lease, putting `next_fire_at` back in the past so the
    // occurrence is due on the very next sweep: right for the closed-session arm above, wrong here,
    // because ACP is the only host for its own session and `prepare` evaluates the *gate* before
    // offering the wakeup, so a job gated on a shell command would re-run that probe every
    // `poll_interval`. `Ran` is worse: it spends the occurrence, which for a one-shot deletes the
    // row, so a `SQLITE_BUSY` from another meka process mid-write would lose the job outright.
    // Leaving the claim to expire retries at `claim_lease` cadence, which a blip survives and a
    // real misconfiguration does not.
    if let Err(error) = super::apply_recorded_binding(&state, &mut runtime).await {
        tracing::warn!(
            "job {} did not run: its session's recorded provider could not be resolved. Fix the \
             profile, or move the session with `meka -r <id> --provider <name>`: {}",
            job_id,
            error
        );
        return FireOutcome::Unrunnable;
    }

    // Publish the token the way the prompt handler does, so `session/cancel` from the editor stops
    // a scheduled turn. Under the lock, so a turn already running cannot have its token replaced.
    let cancellation = tokio_util::sync::CancellationToken::new();
    let _turn_cancellation = entry.publish_cancellation(cancellation.clone());

    // A scheduled fire is a turn that is happening anyway, so an outcome that did not warrant one
    // of its own rides it, as it does on a `session/prompt` the editor sends. Without this a
    // session whose only traffic is scheduled jobs never learns its task was cancelled. Claimed
    // under the lock, so a claim cannot outlive the turn that was going to carry it.
    let riding = if state.shared.config.background.enabled {
        crate::background::claim_undelivered_outcomes(
            &runtime.agent,
            &state.shared.session_manager,
            runtime.session_uuid,
        )
        .await
    } else {
        Vec::new()
    };
    let retention = crate::background::retention_carrying(&riding, wakeup.job.prompt_retention());
    let prompt = if riding.is_empty() {
        wakeup.render_prompt()
    } else {
        crate::background::render_outcomes_before(&riding, &wakeup.render_prompt())
    };

    // Now that the session is ours and the prompt is final, the transcript reads in order: the
    // trigger, then the reply. Shown *after* the fold is decided rather than before it, so the
    // editor sees the text the model actually received -- and in the same order, the outcome above
    // the job's prompt, because that is how `render_outcomes_before` joins them.
    if !riding.is_empty() {
        entry
            .frontend
            .push_out_of_band_prompt(&crate::background::render_outcomes(&riding));
    }
    entry.frontend.push_scheduled_prompt(&wakeup);

    let mut session_uuid = Some(runtime.session_uuid);
    let runtime_inner = &mut *runtime;
    let outcome = runtime_inner
        .agent
        .run_turn_retaining(
            &mut session_uuid,
            &mut runtime_inner.messages,
            prompt,
            Vec::new(),
            cancellation,
            retention,
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

#[cfg(test)]
mod tests {
    /// The binding check must come *before* the batch is stamped `delivered`.
    ///
    /// Asserted against the source, and honestly weaker than a behavioural test: nothing in the
    /// suite drives ACP background-outcome delivery at all, which a mutation sweep confirmed by
    /// replacing `deliver_outcomes` with `()` and staying green. Until that coverage exists this is
    /// what stands between a future edit and a silent regression, so it checks *order* rather than
    /// mere presence -- the same lesson `the_dormant_repin_serialises_against_reconstruction`
    /// learned when a one-character edit defeated a contains-only assertion.
    ///
    /// What it defends: `list_undelivered_background_tasks` filters on `delivered_at IS NULL`, so a
    /// stamped batch has no re-delivery path. Stamping first meant a provider lookup that came back
    /// `SQLITE_BUSY` -- an ordinary occurrence with a second meka process on the store -- destroyed
    /// a report the user was waiting on, with one `warn!` and nothing else.
    #[test]
    fn a_background_outcome_is_stamped_only_once_its_turn_can_run() {
        let source = include_str!("schedule.rs");
        let body = source
            .split("async fn deliver_outcomes(")
            .nth(1)
            .expect("the function this test is about")
            .split("\n/// ")
            .next()
            .expect("splitting always yields a first part");

        let binding = body
            .find("apply_recorded_binding")
            .expect("the turn must run on the profile the row names");
        let stamp = body
            .find("mark_background_tasks_delivered")
            .expect("the batch is stamped here, or this test is watching the wrong function");
        // The call, not the word: prose above it mentions `run_turn` by name, and matching that
        // put the "turn" earlier in the body than the stamp it is supposed to follow.
        let turn = body
            .find(".run_turn(")
            .expect("the turn this whole ordering is about is no longer in this function");
        assert!(
            binding < stamp,
            "the binding must be resolved before the batch is stamped, so a failure leaves the \
             outcomes for the next sweep instead of destroying them; found binding@{binding} \
             stamp@{stamp}"
        );
        // Order alone is not the invariant, and asserting only order made this test blind: deleting
        // the `return;` leaves the sequence intact while the failure falls through to stamp the
        // batch and run the turn on the wrong profile. Verified green with that one line removed.
        let arm = body.get(binding..stamp).expect("ordered above");
        assert!(
            arm.contains("return;"),
            "a binding failure must return before the stamp, not merely be logged before it"
        );
        assert!(
            stamp < turn,
            "but the stamp must still precede the turn: an outcome that reliably wedges the \
             process must not be redelivered on every restart; found stamp@{stamp} turn@{turn}"
        );
    }
}
