//! Background tool calls: work the agent starts and does not wait for.
//!
//! An ordinary tool call blocks the turn until it returns. That is right for a file read and wrong
//! for a twenty-minute build, where the agent's only alternatives today are to hold the turn open
//! or to approximate detachment with `nohup … &` and then poll for a result nothing will announce.
//!
//! A backgrounded call returns a handle immediately and delivers its outcome later, as a turn, over
//! the same path [`crate::schedule`] already uses to inject one from outside the conversation.
//!
//! The invariant that shapes everything here: **a task always ends in a delivered outcome.**
//! [`crate::agent::Agent::run_turn`] is not resumable mid-tool-loop, so work in flight when the
//! process dies cannot be recovered. Silence would be worse than never having offered, because the
//! agent has usually already told someone it would report back. So a task that cannot finish is
//! [`TaskStatus::Interrupted`] and is delivered as such, and the lease that decides *when* a task
//! counts as dead is the session lock itself ([`crate::session::SessionManager::lock_session`]):
//! holding it means nothing else can still be running that session's tasks.

pub mod cli;

use chrono::{DateTime, Local, Utc};
// Reached through `humantime_serde`'s re-export rather than a direct dependency, matching
// `crate::schedule`.
use humantime_serde::re::humantime;
use uuid::Uuid;

/// How a task ended, or that it hasn't.
///
/// The four terminal states deliver identically and differ only in their rendered header. Keeping
/// them distinct is what lets the agent tell "your build failed" from "your build never ran".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    Running,
    Completed,
    /// The tool returned an error, or panicked.
    Failed,
    /// Stopped on request, via `task_cancel` or a second Ctrl+C.
    Cancelled,
    /// The process holding it went away. Reconstructed by the session-load sweep, never written by
    /// the task itself, which by definition is not around to write it.
    Interrupted,
}

impl TaskStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Interrupted => "interrupted",
        }
    }

    /// `None` for an unrecognised string, which `decode` turns into a skipped row rather than a
    /// hard error, matching how the session store treats a forward-compatible event it can't read.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "running" => Some(Self::Running),
            "completed" => Some(Self::Completed),
            "failed" => Some(Self::Failed),
            "cancelled" => Some(Self::Cancelled),
            "interrupted" => Some(Self::Interrupted),
            _ => None,
        }
    }

    pub fn is_terminal(self) -> bool {
        !matches!(self, Self::Running)
    }

    /// The header word the delivered turn leads with.
    fn headline(self) -> &'static str {
        match self {
            Self::Running => "is still running",
            Self::Completed => "finished",
            Self::Failed => "failed",
            Self::Cancelled => "was cancelled",
            Self::Interrupted => "was interrupted",
        }
    }
}

/// A persisted background task.
#[derive(Debug, Clone)]
pub struct BackgroundTask {
    pub id: String,
    pub session_id: Uuid,
    /// The tool that was backgrounded, e.g. `execute_command`.
    pub tool_name: String,
    /// Human-readable summary of what was started, from
    /// [`crate::render::resolve_primary_param`]. Carried so `task_list` and the delivered turn can
    /// name the work without re-deriving it from arguments that are no longer around.
    pub label: String,
    pub status: TaskStatus,
    /// The tool's own output, for a terminal task. Truncated to [`OUTCOME_INLINE_LIMIT`] when it
    /// is also spilled to the scratchpad.
    pub outcome: Option<String>,
    /// Scratchpad entry holding the full output, when it was too large to carry inline.
    pub scratchpad_name: Option<String>,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub delivered_at: Option<DateTime<Utc>>,
}

impl BackgroundTask {
    /// Short id for display, matching the width `task_cancel` accepts. Same convention as
    /// [`crate::schedule::ScheduledJob::short_id`].
    pub fn short_id(&self) -> &str {
        self.id.get(..8).unwrap_or(&self.id)
    }

    /// How long the task ran, or has been running.
    pub fn elapsed(&self) -> chrono::Duration {
        self.finished_at.unwrap_or_else(Utc::now) - self.started_at
    }
}

/// Retire whatever the previous owner left running, for a process that has just taken this session.
///
/// Every path that hydrates a conversation has to call this, not just the CLI resume that first
/// needed it: `meka serve` reattaching an evicted session, and ACP's `session/load`,
/// `session/resume` and `session/fork`, all take the same lease and all inherit the same wreckage.
/// Missing one does not merely skip a report, it strands the row:
/// `list_undelivered_background_tasks` ignores `running`, so the outcome is never delivered, while
/// `list_running_background_tasks` keeps injecting the dead task into `[Background]` on every later
/// turn, telling the model not to restart work that died days ago.
///
/// Non-fatal: a session that cannot sweep must still open.
pub async fn claim_session(session_manager: &crate::session::SessionManager, session_id: Uuid) {
    match session_manager
        .sweep_interrupted_background_tasks(session_id)
        .await
    {
        Ok(0) => {}
        Ok(swept) => tracing::info!(
            "{} background task(s) did not survive the last run; reporting them as interrupted",
            swept
        ),
        Err(error) => tracing::warn!("failed to retire interrupted background tasks: {}", error),
    }
}

/// Live handles for the tasks *this process* started.
///
/// The database row is the durable record; this is the control surface. Deliberately not persisted
/// and deliberately not global: a `CancellationToken` cannot outlive the process holding it, and a
/// task started elsewhere is not this process's to stop. That asymmetry is exactly why the
/// session-load sweep exists, since a row with no handle behind it is a task nobody can finish.
#[derive(Clone, Default)]
pub struct BackgroundTasks {
    inner: std::sync::Arc<tokio::sync::Mutex<std::collections::HashMap<String, TaskHandle>>>,
}

struct TaskHandle {
    session_id: Uuid,
    cancellation: tokio_util::sync::CancellationToken,
    /// `None` between reserving the slot and spawning the work. Cancelling in that window still
    /// works, because the token exists from the start; only joining has to wait.
    join: Option<tokio::task::JoinHandle<()>>,
}

impl BackgroundTasks {
    /// Claim a slot for `session_id`, or refuse because the ceiling is full.
    ///
    /// Check and insert happen under one lock, which is the whole point. Several background calls
    /// in a single assistant message are dispatched concurrently by `execute_tool_calls`, so a
    /// separate count-then-register would let every one of them read the same pre-registration
    /// count and sail past a ceiling of one.
    pub async fn try_reserve(
        &self,
        id: String,
        session_id: Uuid,
        cancellation: tokio_util::sync::CancellationToken,
        max_tasks: usize,
    ) -> bool {
        let mut guard = self.inner.lock().await;
        let running = guard
            .values()
            .filter(|handle| handle.session_id == session_id)
            .count();
        if running >= max_tasks {
            return false;
        }
        guard.insert(id, TaskHandle {
            session_id,
            cancellation,
            join: None,
        });
        true
    }

    /// Attach the spawned work to a slot already reserved by [`Self::try_reserve`].
    pub async fn attach(&self, id: &str, join: tokio::task::JoinHandle<()>) {
        if let Some(handle) = self.inner.lock().await.get_mut(id) {
            handle.join = Some(join);
        }
    }

    /// Drop a task's handle: on the way out of the work itself, and on the failure path between
    /// reserving a slot and spawning into it, so a refused start cannot leak the reservation and
    /// shrink the ceiling for the rest of the session.
    pub async fn forget(&self, id: &str) {
        self.inner.lock().await.remove(id);
    }

    /// How many tasks this process is running, across sessions. For the REPL's survivor line, which
    /// cannot name a session for the same reason [`Self::cancel_all`] cannot.
    pub async fn running_count_all(&self) -> usize {
        self.inner.lock().await.len()
    }

    /// How many of `session_id`'s tasks are running here. Backs the `max_tasks` ceiling.
    pub async fn running_count(&self, session_id: Uuid) -> usize {
        self.inner
            .lock()
            .await
            .values()
            .filter(|handle| handle.session_id == session_id)
            .count()
    }

    /// Signal one task to stop. `false` when it isn't ours, which is the honest answer for a task
    /// whose process is gone: the row will be swept, not cancelled.
    pub async fn cancel(&self, id: &str) -> bool {
        match self.inner.lock().await.get(id) {
            Some(handle) => {
                handle.cancellation.cancel();
                true
            }
            None => false,
        }
    }

    /// Every task id this process is running, without touching them.
    ///
    /// Callers cancelling in bulk need this *before* signalling: a `cancelled` row has to be
    /// written first, because `finish_background_task` only overwrites a `running` row and the
    /// work reacting to its token would otherwise land `failed` there first. "Your build
    /// failed" and "you stopped your build" are exactly the distinction the four terminal
    /// states exist to keep.
    pub async fn task_ids(&self) -> Vec<String> {
        self.inner.lock().await.keys().cloned().collect()
    }

    /// Every task id this process is running for one session, without touching them.
    pub async fn session_task_ids(&self, session_id: Uuid) -> Vec<String> {
        self.inner
            .lock()
            .await
            .iter()
            .filter(|(_, handle)| handle.session_id == session_id)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Signal every task this process is running, returning how many were signalled.
    ///
    /// For the REPL's second Ctrl+C, which cannot name a session: on the first turn the id does not
    /// exist yet (`run_turn` creates it), and a REPL process has exactly one conversation anyway,
    /// since sub-agents never get background calls.
    pub async fn cancel_all(&self) -> usize {
        let guard = self.inner.lock().await;
        for handle in guard.values() {
            handle.cancellation.cancel();
        }
        guard.len()
    }

    /// Signal every one of this session's tasks to stop, returning how many were signalled. Backs
    /// `task_cancel --all`.
    pub async fn cancel_session(&self, session_id: Uuid) -> usize {
        let guard = self.inner.lock().await;
        let mut signalled = 0;
        for handle in guard.values().filter(|h| h.session_id == session_id) {
            handle.cancellation.cancel();
            signalled += 1;
        }
        signalled
    }

    /// Wait for this session's tasks to finish, for a host with nowhere to deliver an outcome
    /// later. `--oneshot` is the case: the process exits with the turn, so a background call there
    /// has to degrade into a slow synchronous one rather than a promise nothing will keep.
    pub async fn wait_for_session(&self, session_id: Uuid) {
        let joins: Vec<tokio::task::JoinHandle<()>> = {
            let mut guard = self.inner.lock().await;
            let ids: Vec<String> = guard
                .iter()
                .filter(|(_, handle)| handle.session_id == session_id)
                .map(|(id, _)| id.clone())
                .collect();
            ids.into_iter()
                .filter_map(|id| guard.remove(&id).and_then(|handle| handle.join))
                .collect()
        };
        for join in joins {
            // A panicking task has already written its own `failed` outcome via the dispatch
            // wrapper, so there is nothing to report here beyond not hanging on it.
            if let Err(error) = join.await {
                tracing::warn!("background task ended abnormally: {}", error);
            }
        }
    }
}

/// Ceiling on outcome text carried inline in the delivered turn. Past this the full output goes to
/// a scratchpad entry and the turn carries the head plus the entry name: a twenty-minute build log
/// would otherwise land in the conversation permanently, for a result that mattered once.
pub const OUTCOME_INLINE_LIMIT: usize = 4 * 1024;

/// Longest task label shown in the `[Background]` index and in delivered headers.
pub const LABEL_MAX_CHARS: usize = 80;

/// Render one or more finished tasks as the user-turn text that delivers them.
///
/// The header is not decoration, for the same reason [`crate::schedule::Wakeup::render_prompt`]
/// carries one: without it the model reads a bare result as though a human had just typed it, and
/// answers conversationally to nobody. It also has to be unambiguous about *who* is speaking,
/// because a backgrounded `spawn_agent` reports in a sub-agent's words, and a sub-agent is
/// permission-clamped by `resolve_subagent_permission` while its words are not.
///
/// Several outcomes ready at once coalesce into one turn rather than one turn each, as the
/// scheduler already does for a backlog.
pub fn render_outcomes(tasks: &[BackgroundTask]) -> String {
    let mut rendered = format!(
        "[Background {} reporting at {}]",
        if tasks.len() == 1 {
            "task".to_string()
        } else {
            format!("tasks ({})", tasks.len())
        },
        Utc::now().with_timezone(&Local).format("%Y-%m-%d %H:%M %Z"),
    );
    rendered.push_str(
        "\nYou started these earlier and did not wait for them. Pick the work back up from here; \
         do not restate this header.\n",
    );

    for task in tasks {
        rendered.push_str(&format!(
            "\n---\n\n**{}** ({}) {} after {}.\n",
            task.short_id(),
            elide(&task.label, LABEL_MAX_CHARS),
            task.status.headline(),
            format_elapsed(task.elapsed()),
        ));
        if let Some(name) = &task.scratchpad_name {
            rendered.push_str(&format!(
                "\nFull output is in scratchpad entry `{}`; the beginning follows.\n",
                name
            ));
        }
        if let Some(outcome) = &task.outcome
            && !outcome.trim().is_empty()
        {
            rendered.push('\n');
            // Sanitised for the same reason MCP text is: this is content from a shell command or a
            // sub-agent, and a terminal-control or bidi-override sequence in it would be rendered
            // to the user and fed to the model verbatim.
            rendered.push_str(&crate::mcp::sanitize::sanitize_text(outcome));
            rendered.push('\n');
        }
    }
    rendered
}

/// Split a tool's output into what the turn carries inline and what, if anything, needs a
/// scratchpad entry. `None` in the second slot means it fit.
///
/// Measured in bytes throughout, including the cut. Deciding in bytes and then cutting in
/// characters would make the head of a non-ASCII log as much as four times the budget, and could
/// spill an output whose "head" is then the whole of it: a delivered turn announcing a scratchpad
/// entry and quoting the entire text it was supposed to spare the conversation.
pub fn split_outcome(output: &str) -> (String, Option<String>) {
    if output.len() <= OUTCOME_INLINE_LIMIT {
        return (output.to_string(), None);
    }
    let mut end = OUTCOME_INLINE_LIMIT;
    while end > 0 && !output.is_char_boundary(end) {
        end -= 1;
    }
    (output[..end].to_string(), Some(output.to_string()))
}

/// Scratchpad entry name for a spilled outcome. Namespaced by the short id so two tasks of the same
/// tool cannot collide, and recognisable so the agent can find it without being told twice.
pub fn spill_entry_name(task_id: &str, tool_name: &str) -> String {
    let short = task_id.get(..8).unwrap_or(task_id);
    format!("task_{}_{}", short, tool_name)
}

/// Human-readable duration, coarse on purpose: nobody needs milliseconds on a twenty-minute build.
fn format_elapsed(elapsed: chrono::Duration) -> String {
    let seconds = elapsed.num_seconds().max(0) as u64;
    if seconds == 0 {
        return "less than a second".to_string();
    }
    humantime::format_duration(std::time::Duration::from_secs(seconds)).to_string()
}

/// A single-line excerpt of `text`, whitespace collapsed and clipped to `limit`. For listing a
/// finished task's result without reprinting it.
pub fn excerpt(text: &str, limit: usize) -> String {
    elide(text, limit)
}

/// Shorten `text` to `limit` characters on a whitespace boundary.
fn elide(text: &str, limit: usize) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= limit {
        return collapsed;
    }
    let clipped: String = collapsed.chars().take(limit).collect();
    let trimmed = match clipped.rfind(char::is_whitespace) {
        Some(space) => &clipped[..space],
        None => clipped.as_str(),
    };
    format!("{}…", trimmed.trim_end())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(status: TaskStatus, outcome: Option<&str>) -> BackgroundTask {
        BackgroundTask {
            id: "7f3a1c22-0000-0000-0000-000000000000".to_string(),
            session_id: Uuid::nil(),
            tool_name: "execute_command".to_string(),
            label: "cargo test --all".to_string(),
            status,
            outcome: outcome.map(str::to_string),
            scratchpad_name: None,
            started_at: Utc::now() - chrono::Duration::seconds(90),
            finished_at: Some(Utc::now()),
            delivered_at: None,
        }
    }

    /// The ceiling has to hold against the sibling calls in one assistant message, which
    /// `execute_tool_calls` dispatches concurrently. Counting and then registering separately let
    /// four calls each read "zero running" and all four start.
    #[tokio::test]
    async fn test_the_ceiling_holds_against_concurrent_reservations() {
        let tasks = BackgroundTasks::default();
        let session_id = Uuid::new_v4();

        let attempts: Vec<_> = (0..8)
            .map(|index| {
                let tasks = tasks.clone();
                tokio::spawn(async move {
                    tasks
                        .try_reserve(
                            format!("task-{index}"),
                            session_id,
                            tokio_util::sync::CancellationToken::new(),
                            3,
                        )
                        .await
                })
            })
            .collect();

        let mut granted = 0;
        for attempt in attempts {
            if attempt.await.expect("join") {
                granted += 1;
            }
        }
        assert_eq!(
            granted, 3,
            "exactly the ceiling, no matter the interleaving"
        );
        assert_eq!(tasks.running_count(session_id).await, 3);
    }

    /// A start that fails after reserving must hand the slot back, or the ceiling shrinks for the
    /// rest of the session.
    #[tokio::test]
    async fn test_a_released_reservation_frees_its_slot() {
        let tasks = BackgroundTasks::default();
        let session_id = Uuid::new_v4();
        let token = tokio_util::sync::CancellationToken::new;

        assert!(
            tasks
                .try_reserve("a".to_string(), session_id, token(), 1)
                .await
        );
        assert!(
            !tasks
                .try_reserve("b".to_string(), session_id, token(), 1)
                .await
        );

        tasks.forget("a").await;
        assert!(
            tasks
                .try_reserve("b".to_string(), session_id, token(), 1)
                .await
        );
    }

    /// The ceiling is per session, so one session filling it must not starve another.
    #[tokio::test]
    async fn test_the_ceiling_is_per_session() {
        let tasks = BackgroundTasks::default();
        let first = Uuid::new_v4();
        let second = Uuid::new_v4();
        let token = tokio_util::sync::CancellationToken::new;

        assert!(tasks.try_reserve("a".to_string(), first, token(), 1).await);
        assert!(!tasks.try_reserve("b".to_string(), first, token(), 1).await);
        assert!(tasks.try_reserve("c".to_string(), second, token(), 1).await);
    }

    #[test]
    fn test_status_round_trips_through_its_string() {
        for status in [
            TaskStatus::Running,
            TaskStatus::Completed,
            TaskStatus::Failed,
            TaskStatus::Cancelled,
            TaskStatus::Interrupted,
        ] {
            assert_eq!(TaskStatus::parse(status.as_str()), Some(status));
        }
        assert_eq!(TaskStatus::parse("wedged"), None);
    }

    #[test]
    fn test_render_names_the_task_and_what_happened() {
        let rendered = render_outcomes(&[task(TaskStatus::Completed, Some("42 passed"))]);
        assert!(rendered.contains("7f3a1c22"), "{rendered}");
        assert!(rendered.contains("cargo test --all"), "{rendered}");
        assert!(rendered.contains("finished"), "{rendered}");
        assert!(rendered.contains("42 passed"), "{rendered}");
    }

    /// Each terminal state has to read differently: "your build failed" and "your build never ran"
    /// call for different next moves.
    #[test]
    fn test_each_terminal_status_reads_differently() {
        let headlines: Vec<String> = [
            TaskStatus::Completed,
            TaskStatus::Failed,
            TaskStatus::Cancelled,
            TaskStatus::Interrupted,
        ]
        .into_iter()
        .map(|status| render_outcomes(&[task(status, None)]))
        .collect();
        for (index, first) in headlines.iter().enumerate() {
            for second in headlines.iter().skip(index + 1) {
                assert_ne!(first, second);
            }
        }
    }

    #[test]
    fn test_several_outcomes_coalesce_into_one_turn() {
        let rendered = render_outcomes(&[
            task(TaskStatus::Completed, Some("one")),
            task(TaskStatus::Failed, Some("two")),
        ]);
        assert!(rendered.contains("tasks (2)"), "{rendered}");
        assert!(
            rendered.contains("one") && rendered.contains("two"),
            "{rendered}"
        );
    }

    /// Outcome text is a shell command's stdout or a sub-agent's prose. Either can carry a bidi
    /// override or a terminal escape, and both are rendered to the user on the way past.
    #[test]
    fn test_outcome_text_is_sanitised() {
        let rendered = render_outcomes(&[task(
            TaskStatus::Completed,
            Some("done\u{202E}gnihtemos rehto"),
        )]);
        assert!(!rendered.contains('\u{202E}'), "{rendered}");
    }

    #[test]
    fn test_split_outcome_spills_only_when_oversized() {
        let (inline, spilled) = split_outcome("short");
        assert_eq!(inline, "short");
        assert!(spilled.is_none());

        let long = "x".repeat(OUTCOME_INLINE_LIMIT + 100);
        let (inline, spilled) = split_outcome(&long);
        assert_eq!(inline.len(), OUTCOME_INLINE_LIMIT);
        assert_eq!(spilled.as_deref(), Some(long.as_str()));
    }

    /// The limit bounds what lands in the conversation, which is measured in bytes. Cutting in
    /// characters instead let a multi-byte log carry several times the budget inline, and at the
    /// sizes just past the threshold the "head" was the whole output: a turn that announced a
    /// scratchpad entry and then quoted everything it was meant to spare.
    #[test]
    fn test_split_outcome_bounds_multibyte_output_by_bytes() {
        // Over the limit in bytes (three each), comfortably under it in characters.
        let log = "√".repeat(OUTCOME_INLINE_LIMIT / 2);
        assert!(log.len() > OUTCOME_INLINE_LIMIT);
        assert!(log.chars().count() < OUTCOME_INLINE_LIMIT);

        let (inline, spilled) = split_outcome(&log);
        assert!(inline.len() <= OUTCOME_INLINE_LIMIT, "{}", inline.len());
        assert!(
            inline.len() < log.len(),
            "a spilled outcome must not also carry the whole log inline",
        );
        assert_eq!(spilled.as_deref(), Some(log.as_str()));
    }

    #[test]
    fn test_spill_entry_name_is_unique_per_task() {
        assert_ne!(
            spill_entry_name("7f3a1c22-aaaa", "execute_command"),
            spill_entry_name("91bd0e44-bbbb", "execute_command"),
        );
        assert!(spill_entry_name("7f3a1c22-aaaa", "execute_command").contains("7f3a1c22"));
    }

    #[test]
    fn test_spilled_outcome_names_its_entry() {
        let mut spilled = task(TaskStatus::Completed, Some("head of the log"));
        spilled.scratchpad_name = Some("task_7f3a1c22_execute_command".to_string());
        let rendered = render_outcomes(&[spilled]);
        assert!(
            rendered.contains("task_7f3a1c22_execute_command"),
            "{rendered}"
        );
    }
}
