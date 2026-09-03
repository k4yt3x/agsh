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
    /// Whether a finished task should wake a host, or wait for the next turn to carry it.
    ///
    /// Every terminal outcome is delivered; this decides only whether delivering it is worth a turn
    /// nobody asked for. A cancellation is always somebody's deliberate act -- `/tasks cancel`, the
    /// `task_cancel` tool, a second Ctrl+C, `POST .../cancel` -- so the one party who would learn
    /// something from the turn already knows, and a command whose whole purpose is to stop work
    /// would be starting some. The outcome still reaches the model, on the next turn there is.
    ///
    /// The others are nobody's decision: a build finished, a tool failed, or a host died holding
    /// the task. The agent asked to be told about the first two and cannot infer the third, and
    /// there may be no human about to type.
    pub fn wakes_a_host(self) -> bool {
        !matches!(self, Self::Cancelled)
    }

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
    /// When subscribers were told, which is the poller's job and happens whether or not a session
    /// is ever live again. Separate from `delivered_at` because the two stopped coinciding once an
    /// outcome could wait for a turn instead of causing one.
    pub announced_at: Option<DateTime<Utc>>,
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

/// What a wake should do with the outcomes waiting on it.
///
/// A REPL wake has two doors -- the outcome watcher and `wake_would_produce_work` for a due job --
/// so the arm cannot infer from *being woken* what woke it. Asked twice per wake, first of what is
/// pending to decide whether to claim at all, then of what was actually claimed. One definition, so
/// the two answers cannot disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutcomeDelivery {
    /// Leave them undelivered. Claiming would stamp an outcome that nothing goes on to deliver.
    Wait,
    /// Something in the batch is nobody's decision, so it earns a turn of its own.
    OwnTurn,
    /// Nothing in the batch warrants a turn, but a job fired: it rides that job's prompt.
    RideAFiredJob,
}

impl OutcomeDelivery {
    /// Whether this answer justifies stamping the batch delivered.
    pub fn claims(self) -> bool {
        !matches!(self, Self::Wait)
    }
}

/// Decide [`OutcomeDelivery`] for a batch, given whether a scheduled job fired on the same wake.
pub fn wake_outcome_delivery(ready: &[BackgroundTask], a_job_fired: bool) -> OutcomeDelivery {
    if ready.iter().any(|task| task.status.wakes_a_host()) {
        OutcomeDelivery::OwnTurn
    } else if a_job_fired && !ready.is_empty() {
        OutcomeDelivery::RideAFiredJob
    } else {
        // Includes the empty batch, whichever door woke this: there is nothing to deliver, and a
        // fired job's prompt goes out unchanged.
        OutcomeDelivery::Wait
    }
}

/// Take this session's undelivered outcomes and stamp them, ready to ride on a turn.
///
/// Empty on a database error rather than propagating, because failing to *report* a finished task
/// must not also fail the turn the caller was about to run. Not delivering is better than
/// delivering forever: without the stamp the next drain would repeat these, and every drain after
/// it.
pub async fn claim_undelivered_outcomes(
    agent: &crate::agent::Agent,
    session_manager: &crate::session::SessionManager,
    session_id: uuid::Uuid,
) -> Vec<BackgroundTask> {
    if !a_turn_can_carry_them(agent).await {
        return Vec::new();
    }
    claim_outcomes_now(session_manager, session_id).await
}

/// [`claim_undelivered_outcomes`] without the readiness gate.
///
/// For a caller that is not about to run a turn -- the one-shot's post-turn report, which prints to
/// stderr on the way out -- and for tests that drive the claim concurrently with a store but no
/// agent. A caller that *is* about to run a turn must use the gated form, or a turn refused before
/// it touches the conversation leaves the batch stamped and unreadable.
pub(crate) async fn claim_outcomes_now(
    session_manager: &crate::session::SessionManager,
    session_id: uuid::Uuid,
) -> Vec<BackgroundTask> {
    let store = session_manager.background_store();
    let ready = match store.list_undelivered_background_tasks(session_id).await {
        Ok(ready) => ready,
        Err(error) => {
            tracing::warn!("failed to load background task outcomes: {}", error);
            return Vec::new();
        }
    };
    if ready.is_empty() {
        return ready;
    }
    let ids: Vec<String> = ready.iter().map(|task| task.id.clone()).collect();
    let claimed = match store.mark_background_tasks_delivered(&ids).await {
        Ok(claimed) => claimed,
        Err(error) => {
            tracing::warn!(
                "failed to stamp background outcomes as delivered: {}",
                error
            );
            return Vec::new();
        }
    };
    // Only what this caller won. Listing and stamping are two statements, and a poller can claim
    // the same row in the gap: whoever the `WHERE delivered_at IS NULL` refuses reports nothing,
    // rather than both of them reporting it.
    only_what_was_won(ready, &claimed)
}

/// Keep only the outcomes a stamp actually won.
///
/// The stamps are compare-and-swaps that return the rows they took, and every caller must report
/// against that rather than against the snapshot it chose from: two claimers reading the same row
/// as unclaimed is the ordinary case, and acting on the read is how the same outcome reaches the
/// model twice. Four call sites asked this and each wrote the filter out; naming it is what stops
/// the fifth forgetting.
pub fn only_what_was_won(ready: Vec<BackgroundTask>, claimed: &[String]) -> Vec<BackgroundTask> {
    ready
        .into_iter()
        .filter(|task| claimed.contains(&task.id))
        .collect()
}

/// Whether the turn that would carry these outcomes can start at all.
///
/// `Agent::run_turn` gates on MCP readiness as its first statement, before it touches the
/// conversation, so a required server that is down refuses every turn -- and a batch stamped ahead
/// of one is a batch nobody is ever told about, because the stamp is one-way. Every claimer asks
/// this first: [`claim_undelivered_outcomes`] for the hosts that fold a batch into somebody's
/// prompt, and the two pollers directly, since they stamp with
/// [`BackgroundStore::mark_background_tasks_delivered`] to render the batch as a turn of its own.
pub async fn a_turn_can_carry_them(agent: &crate::agent::Agent) -> bool {
    match agent.ensure_ready_for_turn().await {
        Ok(()) => true,
        Err(error) => {
            tracing::warn!(
                "holding background task outcomes: a turn cannot start right now: {}",
                error
            );
            false
        }
    }
}

/// The retention a carrier prompt deserves once an outcome may be riding on it.
///
/// A recurring job asks for [`crate::agent::PromptRetention::WithdrawOnFailure`] because its next
/// occurrence regenerates the prompt, so a failed copy carries nothing worth keeping. That stops
/// being true the moment an outcome joins it: the row is stamped delivered before the turn starts
/// and is never handed out again, so withdrawing the message loses the only copy. Three hosts fold
/// an outcome into a fired job's prompt and all three ask this, rather than each remembering.
pub fn retention_carrying(
    riding: &[BackgroundTask],
    job: crate::agent::PromptRetention,
) -> crate::agent::PromptRetention {
    if riding.is_empty() {
        job
    } else {
        crate::agent::PromptRetention::Keep
    }
}

/// [`render_outcomes`] ahead of a prompt somebody else wrote.
///
/// The standalone form ends "Pick the work back up from here", which is right when it *is* the
/// prompt and wrong above an unrelated question. Joining rather than appending a message of its own
/// is what keeps this off the conversation as a turn boundary: a lone user message opens a turn
/// (`crate::conversation::opens_turn`), and one nobody answers would be rewound in place of the
/// user's last exchange and sent as a second consecutive user turn.
pub fn render_outcomes_before(tasks: &[BackgroundTask], prompt: &str) -> String {
    format!(
        "{}\n---\n\n{}",
        render_outcomes_with_trailer(tasks, PREAMBLE_TRAILER),
        prompt
    )
}

/// What the standalone form tells the model to do next.
const STANDALONE_TRAILER: &str = "\nYou started these earlier and did not wait for them. Pick the \
    work back up from here; do not restate this header.\n";

/// The same, for a report that precedes a prompt of its own.
const PREAMBLE_TRAILER: &str = "\nYou started these earlier and did not wait for them. Read them, \
    then answer what follows; do not restate this header.\n";

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
    render_outcomes_with_trailer(tasks, STANDALONE_TRAILER)
}

fn render_outcomes_with_trailer(tasks: &[BackgroundTask], trailer: &str) -> String {
    let mut rendered = format!(
        "[Background {} reporting at {}]",
        if tasks.len() == 1 {
            "task".to_string()
        } else {
            format!("tasks ({})", tasks.len())
        },
        Utc::now().with_timezone(&Local).format("%Y-%m-%d %H:%M %Z"),
    );
    rendered.push_str(trailer);

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
             started_at, finished_at, announced_at, delivered_at FROM background_tasks \
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
             started_at, finished_at, announced_at, delivered_at FROM background_tasks \
             WHERE session_id = ?1 AND status = 'running' ORDER BY started_at ASC",
            vec![session_id.to_string()],
        )
        .await
    }

    /// A session's finished-but-unreported tasks, oldest first. The delivery poll's query; served
    /// by `idx_background_tasks_session_status`.
    ///
    /// This and [`Self::mark_background_tasks_delivered`] are two statements, not one transaction,
    /// so two processes that both list before either marks would each render the same outcome. That
    /// is currently unreachable -- delivery only happens inside a session, and a session is held by
    /// one process at a time from the moment its row exists -- so the pair rests on the session
    /// lock rather than on its own atomicity. Anything that ever lets two hosts open one
    /// session at once has to make this a transaction first.
    pub async fn list_undelivered_background_tasks(
        &self,
        session_id: Uuid,
    ) -> crate::error::Result<Vec<BackgroundTask>> {
        self.query_background_tasks(
            "SELECT id, session_id, tool_name, label, status, outcome, scratchpad_name, \
             started_at, finished_at, announced_at, delivered_at FROM background_tasks \
             WHERE session_id = ?1 AND status != 'running' AND delivered_at IS NULL \
             ORDER BY finished_at ASC",
            vec![session_id.to_string()],
        )
        .await
    }

    /// Stamp outcomes as delivered.
    ///
    /// Called *before* the turn runs: an outcome that reliably wedges the process would otherwise
    /// be redelivered on every restart, turning one bad result into a boot loop. Losing one report
    /// is the cheaper failure.
    ///
    ///
    /// [`crate::schedule::ScheduleStore::complete_claim`] writes *after* the turn, because a lease
    /// plus an attempt ceiling gives it the same boot-loop protection without paying an occurrence
    /// for every crash. A background outcome has no lease to hold, so it keeps the cruder rule.
    pub async fn mark_background_tasks_delivered(
        &self,
        ids: &[String],
    ) -> crate::error::Result<Vec<String>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let ids = ids.to_vec();
        let delivered_at = chrono::Utc::now().to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                // One transaction, so a failure part-way cannot leave half a batch stamped. The
                // caller renders every one of these into a single turn, and stamping only some of
                // them would repeat the rest alongside it on the next tick.
                let txn = connection.transaction()?;
                let mut claimed = Vec::new();
                {
                    let mut statement = txn.prepare(
                        "UPDATE background_tasks SET delivered_at = ?2 \
                         WHERE id = ?1 AND delivered_at IS NULL",
                    )?;
                    for id in &ids {
                        if statement.execute(rusqlite::params![id, delivered_at])? == 1 {
                            claimed.push(id.clone());
                        }
                    }
                }
                txn.commit()?;
                Ok(claimed)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to mark tasks delivered: {}", error))
            })
    }

    /// A session's finished-but-unannounced tasks, oldest first.
    ///
    /// The poller's own question, and deliberately not [`Self::list_undelivered_background_tasks`]:
    /// telling subscribers a task finished needs nothing from a live session, while telling the
    /// model needs a turn. An outcome that waits for one must stay undelivered without also being
    /// re-announced on every poll.
    ///
    /// Bounded to the undelivered pool, which is what keeps it news. `announced_at` is written only
    /// by `meka serve`, so every task a REPL or ACP session ever ran is unannounced forever in a
    /// shared store; without this clause, opening one of those sessions in `meka serve` would fire
    /// `task.finished` for work that finished weeks ago and was reported at the time. A task still
    /// in the pool has been reported to nobody, which is the case worth pushing.
    pub async fn list_unannounced_background_tasks(
        &self,
        session_id: Uuid,
    ) -> crate::error::Result<Vec<BackgroundTask>> {
        self.query_background_tasks(
            "SELECT id, session_id, tool_name, label, status, outcome, scratchpad_name, \
             started_at, finished_at, announced_at, delivered_at FROM background_tasks \
             WHERE session_id = ?1 AND status != 'running' AND announced_at IS NULL \
             AND delivered_at IS NULL ORDER BY finished_at ASC",
            vec![session_id.to_string()],
        )
        .await
    }

    /// Stamp a batch announced. One transaction, for the reason
    /// [`Self::mark_background_tasks_delivered`] gives.
    pub async fn mark_background_tasks_announced(
        &self,
        ids: &[String],
    ) -> crate::error::Result<Vec<String>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let ids = ids.to_vec();
        let announced_at = chrono::Utc::now().to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let txn = connection.transaction()?;
                let mut claimed = Vec::new();
                {
                    let mut statement = txn.prepare(
                        "UPDATE background_tasks SET announced_at = ?2 \
                         WHERE id = ?1 AND announced_at IS NULL",
                    )?;
                    for id in &ids {
                        if statement.execute(rusqlite::params![id, announced_at])? == 1 {
                            claimed.push(id.clone());
                        }
                    }
                }
                txn.commit()?;
                Ok(claimed)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to mark tasks announced: {}", error))
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
        if !crate::render::is_usable_id_prefix(id_prefix) {
            return Ok(None);
        }
        let wanted = crate::render::id_prefix_for_matching(id_prefix);
        let tasks = self.list_background_tasks(session_id).await?;
        let matches: Vec<BackgroundTask> = tasks
            .into_iter()
            .filter(|task| task.id.starts_with(&wanted))
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
                            announced_at: row.get(9)?,
                            delivered_at: row.get(10)?,
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
    announced_at: Option<String>,
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
            announced_at: parse_optional(self.announced_at)?,
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
            announced_at: None,
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

    /// Only a cancellation is somebody's deliberate act, so only a cancellation waits.
    ///
    /// Every terminal outcome still reaches the model; this decides whether reaching it is worth a
    /// turn nobody asked for. Written over the whole enum rather than the two cases that motivated
    /// it, so a status added later has to answer the question rather than inherit an answer.
    #[test]
    fn only_a_cancellation_waits_for_a_turn_that_was_happening_anyway() {
        for status in [
            TaskStatus::Running,
            TaskStatus::Completed,
            TaskStatus::Failed,
            TaskStatus::Interrupted,
        ] {
            assert!(
                status.wakes_a_host(),
                "{} is nobody's decision, so the agent has to be told when it happens",
                status.as_str()
            );
        }
        assert!(
            !TaskStatus::Cancelled.wakes_a_host(),
            "whoever cancelled it already knows, and a stop command must not start a turn"
        );
    }

    /// A caller reports what the stamp gave it, not what it read a moment earlier.
    ///
    /// Three call sites used to write this filter out and one test covered one of them. The rule is
    /// the whole of the double-delivery fix: two claimers reading the same row as unclaimed is
    /// ordinary, the compare-and-swap picks one, and a loser that reports its snapshot anyway
    /// delivers the outcome a second time.
    #[test]
    fn a_report_carries_only_the_outcomes_the_stamp_won() {
        // Distinct ids: the shared fixture hands out a fixed one, and a filter keyed on id cannot
        // be tested with two rows that carry the same one.
        let mut won = task(TaskStatus::Completed, Some("built"));
        won.id = "11111111-0000-0000-0000-000000000000".to_string();
        let mut lost = task(TaskStatus::Cancelled, None);
        lost.id = "22222222-0000-0000-0000-000000000000".to_string();
        let read = vec![won.clone(), lost.clone()];

        let reported = only_what_was_won(read.clone(), &[won.id.clone()]);
        assert_eq!(
            reported.iter().map(|task| &task.id).collect::<Vec<_>>(),
            vec![&won.id],
            "the row this caller lost must not be reported by it"
        );
        assert!(
            only_what_was_won(read.clone(), &[]).is_empty(),
            "a caller that won nothing reports nothing"
        );
        assert_eq!(
            only_what_was_won(read.clone(), &[won.id.clone(), lost.id.clone()]).len(),
            2,
            "and winning everything reports everything"
        );
    }

    /// A prompt carrying an outcome is never withdrawn, whatever the job asked for.
    ///
    /// The branch only matters when a turn fails, so nothing that drives a *successful* turn can
    /// see it -- which was every test of this path. Withdrawing a carrier prompt destroys the only
    /// copy of the outcome: the row was stamped delivered before the turn began and
    /// `list_undelivered_background_tasks` never returns it again.
    #[test]
    fn a_prompt_carrying_an_outcome_is_never_withdrawn() {
        let carried = [task(TaskStatus::Cancelled, None)];
        for asked in [
            crate::agent::PromptRetention::Keep,
            crate::agent::PromptRetention::WithdrawOnFailure,
        ] {
            assert_eq!(
                retention_carrying(&carried, asked),
                crate::agent::PromptRetention::Keep,
                "an outcome rides on this prompt, so a failure must not take it with it"
            );
        }
        assert_eq!(
            retention_carrying(&[], crate::agent::PromptRetention::WithdrawOnFailure),
            crate::agent::PromptRetention::WithdrawOnFailure,
            "and a job carrying only its own prompt keeps the job's answer: the next occurrence \
             regenerates it"
        );
        assert_eq!(
            retention_carrying(&[], crate::agent::PromptRetention::Keep),
            crate::agent::PromptRetention::Keep
        );
    }

    /// The REPL wake arm, over its whole matrix. Two doors raise the same flag, so every
    /// combination of "what is waiting" and "did a job fire" has to have an answer.
    ///
    /// The two that matter: a quiet batch with no fired job must NOT be claimed, because claiming
    /// stamps it delivered and there is no turn to put it in -- reachable whenever the scheduler
    /// door woke this and the gate then declined the job. And a quiet batch WITH a fired job must
    /// be claimed, or the cancellation waits for a user who may never type.
    #[test]
    fn a_wake_claims_an_outcome_only_when_a_turn_will_carry_it() {
        let quiet = [task(TaskStatus::Cancelled, None)];
        let loud = [task(TaskStatus::Completed, Some("done"))];
        let mixed = [
            task(TaskStatus::Cancelled, None),
            task(TaskStatus::Failed, None),
        ];

        assert_eq!(
            wake_outcome_delivery(&[], false),
            OutcomeDelivery::Wait,
            "nothing waiting, nothing fired"
        );
        assert_eq!(
            wake_outcome_delivery(&[], true),
            OutcomeDelivery::Wait,
            "a job fired with nothing waiting: its prompt goes out unchanged"
        );
        assert_eq!(
            wake_outcome_delivery(&quiet, false),
            OutcomeDelivery::Wait,
            "a cancellation alone must not be claimed: there is no turn to carry it, and the stamp \
             is one-way"
        );
        assert_eq!(
            wake_outcome_delivery(&quiet, true),
            OutcomeDelivery::RideAFiredJob,
            "a fired job is a turn that is happening anyway, so the cancellation joins it"
        );
        assert_eq!(
            wake_outcome_delivery(&loud, false),
            OutcomeDelivery::OwnTurn,
            "a finished build is nobody's decision, so it earns the turn"
        );
        assert_eq!(
            wake_outcome_delivery(&loud, true),
            OutcomeDelivery::OwnTurn,
            "and still does when a job also fired"
        );
        assert_eq!(
            wake_outcome_delivery(&mixed, false),
            OutcomeDelivery::OwnTurn,
            "a batch with anything turn-worthy in it goes as a turn entire"
        );

        assert!(!OutcomeDelivery::Wait.claims());
        assert!(OutcomeDelivery::OwnTurn.claims());
        assert!(
            OutcomeDelivery::RideAFiredJob.claims(),
            "riding is a delivery, so it stamps"
        );
    }

    /// The notice rides on a prompt rather than standing as a message of its own.
    ///
    /// A lone user message opens a turn (`crate::conversation::opens_turn`), so one nobody answers
    /// is a boundary with no turn behind it: `/rewind 1` cuts there instead of at the user's last
    /// exchange, compaction reads the conversation as ending on an unanswered prompt, and the next
    /// real prompt goes out as a second consecutive user turn.
    #[test]
    fn an_outcome_joins_a_prompt_rather_than_becoming_one() {
        let task = BackgroundTask {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: uuid::Uuid::new_v4(),
            tool_name: "execute_command".to_string(),
            label: "sleep 900".to_string(),
            status: TaskStatus::Cancelled,
            outcome: None,
            scratchpad_name: None,
            announced_at: None,
            started_at: chrono::Utc::now(),
            finished_at: Some(chrono::Utc::now()),
            delivered_at: None,
        };
        let joined = render_outcomes_before(std::slice::from_ref(&task), "what is in this CSV?");

        assert!(
            joined.ends_with("what is in this CSV?"),
            "the user's prompt has to be the thing the model answers: {joined}"
        );
        assert!(
            joined.contains("was cancelled"),
            "and the outcome has to be in it: {joined}"
        );
        assert!(
            !joined.contains("Pick the work back up from here"),
            "the standalone trailer instructs the model to resume the cancelled work, which is \
             wrong above somebody else's question: {joined}"
        );
        assert!(
            render_outcomes(std::slice::from_ref(&task))
                .contains("Pick the work back up from here"),
            "while the standalone form, which is the whole prompt, still says it"
        );
    }
}
