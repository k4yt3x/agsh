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

use crate::error::MekaError;

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
        .background_store()
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

    /// Wait for every task this process is running, for a host that is about to exit.
    ///
    /// [`Self::cancel_all`] only fires the tokens, and firing a token is not the same as the task
    /// acting on it. A task parked at an await is dropped without ever being polled again when the
    /// runtime goes, so it reaches neither `kill_child_tree` -- leaving the `setsid()`-ed child
    /// running with no meka process tracking it -- nor `finish_background_task`, leaving the row
    /// `running` for the next session open to sweep to `interrupted`. Those two are precisely what
    /// cancelling it was for, so the wait is what makes the cancel mean anything.
    ///
    /// Callers should bound this: cancelling asks, and a task that does not answer must not hold
    /// the terminal.
    pub async fn wait_all(&self) {
        let joins: Vec<tokio::task::JoinHandle<()>> = {
            let mut guard = self.inner.lock().await;
            let ids: Vec<String> = guard.keys().cloned().collect();
            ids.into_iter()
                .filter_map(|id| guard.remove(&id).and_then(|handle| handle.join))
                .collect()
        };
        for join in joins {
            if let Err(error) = join.await {
                tracing::warn!("background task ended abnormally: {}", error);
            }
        }
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
///
/// Interacts with `tools::shell`'s `OUTPUT_WINDOW_BYTES`, which is eight times larger and
/// keeps **both ends** of an overflowing stream. `split_outcome` keeps only the head, so a
/// *backgrounded* `execute_command` that overflowed loses the tail the shell tool went to trouble
/// to preserve. That is deliberate -- an outcome arrives unbidden, mid-conversation, so it should
/// cost less window than a result the model asked for -- but the two numbers are coupled, and
/// raising this one without reading that one produces a delivered turn wider than the tool's own
/// result. Nothing is lost either way: the entry name reaches the model and the scratchpad holds
/// all of it.
pub const OUTCOME_INLINE_LIMIT: usize = 4 * 1024;

/// Longest task label shown in the `[Background]` index and in delivered headers.
pub const LABEL_MAX_CHARS: usize = 80;

/// Render one or more finished tasks as the user-turn text that delivers them.
///
/// The header is not decoration, for the same reason [`crate::schedule::Wakeup::render_prompt`]
/// carries one: without it the model reads a bare result as though a human had just typed it, and
/// answers conversationally to nobody. It also has to be unambiguous about *who* is speaking,
/// because a backgrounded `agent_spawn` reports in a sub-agent's words, and a sub-agent is
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
            // Sanitised like the outcome below. The label is derived from the tool's primary
            // argument, which for `execute_command` is a shell command line the *model* wrote, so
            // it is no more trusted than the output it names.
            elide(
                &crate::mcp::sanitize::sanitize_text(&task.label),
                LABEL_MAX_CHARS
            ),
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

/// Background work's slice of the session database, handed out by
/// [`crate::session::SessionManager::background_store`].
#[derive(Clone)]
pub struct BackgroundStore {
    connection: std::sync::Arc<tokio_rusqlite::Connection>,
}

impl BackgroundStore {
    pub(crate) fn new(connection: std::sync::Arc<tokio_rusqlite::Connection>) -> Self {
        Self { connection }
    }

    /// Record a task as started. Written before the work is spawned, so a process that dies between
    /// the two leaves a `running` row the sweep will retire rather than a task nobody knows about.
    pub async fn start_background_task(&self, task: &BackgroundTask) -> crate::error::Result<()> {
        let id = task.id.clone();
        let session_id = task.session_id.to_string();
        let tool_name = task.tool_name.clone();
        let label = task.label.clone();
        let status = task.status.as_str().to_string();
        let started_at = task.started_at.to_rfc3339();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO background_tasks \
                     (id, session_id, tool_name, label, status, started_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    rusqlite::params![id, session_id, tool_name, label, status, started_at],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to start background task: {}", error))
            })
    }

    /// Record a terminal outcome.
    ///
    /// Guarded on `status = 'running'` so a task that was cancelled, or swept to `interrupted`,
    /// cannot be overwritten by its own work finishing a moment later. The first terminal write
    /// wins, which is what keeps a cancelled task from reporting success.
    pub async fn finish_background_task(
        &self,
        id: &str,
        status: TaskStatus,
        outcome: Option<String>,
        scratchpad_name: Option<String>,
    ) -> crate::error::Result<()> {
        let id = id.to_string();
        let status = status.as_str().to_string();
        let finished_at = chrono::Utc::now().to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE background_tasks \
                     SET status = ?2, outcome = ?3, scratchpad_name = ?4, finished_at = ?5 \
                     WHERE id = ?1 AND status = 'running'",
                    rusqlite::params![id, status, outcome, scratchpad_name, finished_at],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to finish background task: {}", error))
            })
    }

    /// Every task belonging to one session, newest first.
    pub async fn list_background_tasks(
        &self,
        session_id: Uuid,
    ) -> crate::error::Result<Vec<BackgroundTask>> {
        self.query_background_tasks(
            "SELECT id, session_id, tool_name, label, status, outcome, scratchpad_name, \
             started_at, finished_at, delivered_at FROM background_tasks \
             WHERE session_id = ?1 ORDER BY started_at DESC",
            vec![session_id.to_string()],
        )
        .await
    }

    /// A session's tasks still running, oldest first. Backs the `[Background]` index and the Ctrl+C
    /// survivor line.
    pub async fn list_running_background_tasks(
        &self,
        session_id: Uuid,
    ) -> crate::error::Result<Vec<BackgroundTask>> {
        self.query_background_tasks(
            "SELECT id, session_id, tool_name, label, status, outcome, scratchpad_name, \
             started_at, finished_at, delivered_at FROM background_tasks \
             WHERE session_id = ?1 AND status = 'running' ORDER BY started_at ASC",
            vec![session_id.to_string()],
        )
        .await
    }

    /// A session's finished-but-unreported tasks, oldest first. The delivery poll's query; served
    /// by `idx_background_tasks_session_status`.
    pub async fn list_undelivered_background_tasks(
        &self,
        session_id: Uuid,
    ) -> crate::error::Result<Vec<BackgroundTask>> {
        self.query_background_tasks(
            "SELECT id, session_id, tool_name, label, status, outcome, scratchpad_name, \
             started_at, finished_at, delivered_at FROM background_tasks \
             WHERE session_id = ?1 AND status != 'running' AND delivered_at IS NULL \
             ORDER BY finished_at ASC",
            vec![session_id.to_string()],
        )
        .await
    }

    /// Stamp outcomes as delivered.
    ///
    /// Called *before* the turn runs, matching
    /// [`crate::schedule::ScheduleStore::stamp_scheduled_job_fired`] and for the same reason: an
    /// outcome that reliably wedges the process would otherwise be redelivered on every restart,
    /// turning one bad result into a boot loop. Losing one report is the cheaper failure.
    pub async fn mark_background_tasks_delivered(
        &self,
        ids: &[String],
    ) -> crate::error::Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let ids = ids.to_vec();
        let delivered_at = chrono::Utc::now().to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                // One transaction, so a failure part-way cannot leave half a batch stamped. The
                // caller renders every one of these into a single turn, and stamping only some of
                // them would repeat the rest alongside it on the next tick.
                let txn = connection.transaction()?;
                {
                    let mut statement =
                        txn.prepare("UPDATE background_tasks SET delivered_at = ?2 WHERE id = ?1")?;
                    for id in &ids {
                        statement.execute(rusqlite::params![id, delivered_at])?;
                    }
                }
                txn.commit()?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to mark tasks delivered: {}", error))
            })
    }

    /// Retire every `running` task in this session as [`TaskStatus::Interrupted`], returning how
    /// many were swept.
    ///
    /// Called when a process takes ownership of a session. The session lock
    /// ([`crate::session::SessionManager::lock_session`]) is the lease: holding it means no other
    /// process can still be running this session's tasks, so any row that still says `running`
    /// belongs to a process that is gone. Without this a task in flight at shutdown would leave the
    /// agent waiting on a report that can never arrive, having very likely already promised one.
    pub async fn sweep_interrupted_background_tasks(
        &self,
        session_id: Uuid,
    ) -> crate::error::Result<usize> {
        let session_id = session_id.to_string();
        let status = TaskStatus::Interrupted.as_str().to_string();
        let finished_at = chrono::Utc::now().to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let swept = connection.execute(
                    "UPDATE background_tasks SET status = ?2, finished_at = ?3 \
                     WHERE session_id = ?1 AND status = 'running'",
                    rusqlite::params![session_id, status, finished_at],
                )?;
                Ok(swept)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to sweep background tasks: {}", error))
            })
    }

    /// Resolve a full or unique-prefix task id within a session. An ambiguous prefix is an error
    /// rather than an arbitrary pick, matching
    /// [`crate::schedule::ScheduleStore::cancel_scheduled_job`] in behaviour and in error variant.
    /// That one returned `Database` for the identical condition, so the two `serve` endpoints
    /// answered different HTTP statuses for the same mistake; both now use `Config`, which reaches
    /// the caller as a 422.
    pub async fn resolve_background_task(
        &self,
        session_id: Uuid,
        id_prefix: &str,
    ) -> crate::error::Result<Option<BackgroundTask>> {
        let tasks = self.list_background_tasks(session_id).await?;
        let matches: Vec<BackgroundTask> = tasks
            .into_iter()
            .filter(|task| task.id.starts_with(id_prefix))
            .collect();
        match matches.len() {
            0 => Ok(None),
            1 => Ok(matches.into_iter().next()),
            _ => Err(MekaError::Config(format!(
                "task id '{}' is ambiguous; it matches {} tasks",
                id_prefix,
                matches.len()
            ))),
        }
    }

    /// Shared row decoder, mirroring `ScheduleStore::query_scheduled_jobs`: one unreadable row is
    /// skipped with a warning rather than failing every other task in the query.
    async fn query_background_tasks(
        &self,
        sql: &'static str,
        params: Vec<String>,
    ) -> crate::error::Result<Vec<BackgroundTask>> {
        let rows: Vec<BackgroundTaskRow> = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(sql)?;
                let rows = statement
                    .query_map(rusqlite::params_from_iter(params.iter()), |row| {
                        Ok(BackgroundTaskRow {
                            id: row.get(0)?,
                            session_id: row.get(1)?,
                            tool_name: row.get(2)?,
                            label: row.get(3)?,
                            status: row.get(4)?,
                            outcome: row.get(5)?,
                            scratchpad_name: row.get(6)?,
                            started_at: row.get(7)?,
                            finished_at: row.get(8)?,
                            delivered_at: row.get(9)?,
                        })
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to load background tasks: {}", error))
            })?;

        Ok(rows
            .into_iter()
            .filter_map(|row| {
                let id = row.id.clone();
                row.decode()
                    .inspect_err(|error| {
                        tracing::warn!("skipping unreadable background task {}: {}", id, error);
                    })
                    .ok()
            })
            .collect())
    }
}

/// Raw `background_tasks` row, decoded outside the database closure so a parse failure can be
/// logged and skipped individually.
struct BackgroundTaskRow {
    id: String,
    session_id: String,
    tool_name: String,
    label: String,
    status: String,
    outcome: Option<String>,
    scratchpad_name: Option<String>,
    started_at: String,
    finished_at: Option<String>,
    delivered_at: Option<String>,
}

impl BackgroundTaskRow {
    fn decode(self) -> std::result::Result<BackgroundTask, String> {
        let parse_time =
            |text: &str| -> std::result::Result<chrono::DateTime<chrono::Utc>, String> {
                chrono::DateTime::parse_from_rfc3339(text)
                    .map(|at| at.with_timezone(&chrono::Utc))
                    .map_err(|error| format!("bad timestamp '{}': {}", text, error))
            };
        let parse_optional = |text: Option<String>| -> std::result::Result<_, String> {
            text.as_deref().map(parse_time).transpose()
        };

        Ok(BackgroundTask {
            id: self.id,
            session_id: Uuid::parse_str(&self.session_id)
                .map_err(|error| format!("bad session id: {}", error))?,
            tool_name: self.tool_name,
            label: self.label,
            status: TaskStatus::parse(&self.status)
                .ok_or_else(|| format!("unknown status '{}'", self.status))?,
            outcome: self.outcome,
            scratchpad_name: self.scratchpad_name,
            started_at: parse_time(&self.started_at)?,
            finished_at: parse_optional(self.finished_at)?,
            delivered_at: parse_optional(self.delivered_at)?,
        })
    }
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

    /// Cancelling and leaving is not enough: the task has to be given the chance to act on it.
    ///
    /// `/exit` returns straight into `Runtime::shutdown_background`, which drops every task where
    /// it stands. A task parked at an await is then never polled again, so the cleanup that follows
    /// its cancellation check -- killing its process group, writing its terminal row -- simply
    /// never happens, which is everything the cancel was for. The task here records the same way:
    /// it observes the token, then does one more await before setting the flag.
    #[tokio::test]
    async fn cancelling_every_task_and_waiting_lets_them_run_their_cleanup() {
        let tasks = BackgroundTasks::default();
        let session_id = Uuid::new_v4();
        let token = tokio_util::sync::CancellationToken::new();
        let cleaned_up = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

        assert!(
            tasks
                .try_reserve("a".to_string(), session_id, token.clone(), 4)
                .await
        );
        let join = tokio::spawn({
            let cleaned_up = std::sync::Arc::clone(&cleaned_up);
            async move {
                token.cancelled().await;
                tokio::task::yield_now().await;
                cleaned_up.store(true, std::sync::atomic::Ordering::Release);
            }
        });
        tasks.attach("a", join).await;

        assert_eq!(tasks.cancel_all().await, 1);
        tasks.wait_all().await;

        assert!(
            cleaned_up.load(std::sync::atomic::Ordering::Acquire),
            "the task must have reached its cleanup before the wait returned",
        );
        assert_eq!(
            tasks.running_count_all().await,
            0,
            "and the registry must be empty afterwards",
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
