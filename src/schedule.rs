//! Scheduled wakeups: the agent arranging to be prompted again later.
//!
//! Every turn meka runs today originates outside the agent -- a human typing, an editor sending
//! `session/prompt`, a client calling `POST /v1/sessions/{id}/turn`. This module supplies the one
//! trigger nothing else can: a timer. External events already have a door (that HTTP endpoint takes
//! anything that can make a request), so what was missing was the agent's ability to say "wake me
//! at 09:00" and have that survive the process it was said in.
//!
//! A job pairs a [`Schedule`] with the prompt to deliver, and optionally a *gate*: a cheap shell
//! command run first, whose result decides whether the expensive model turn happens at all. Without
//! one, "watch X every 30s" costs a model turn every 30 seconds; with one it costs a process spawn,
//! and a turn only when something actually changed.
//!
//! Jobs live in the session database rather than `config.toml` because they are runtime data the
//! agent creates, not settings a human writes, and they are keyed to a session so a job dies with
//! the conversation that asked for it.

pub mod cli;

use std::{str::FromStr, time::Duration};

use chrono::{DateTime, Local, Utc};
use croner::Cron;
// Reached through `humantime_serde`'s re-export rather than a direct dependency, which is also
// how `crate::config` gets at it. One duration syntax, one copy of the parser.
use humantime_serde::re::humantime;

/// Smallest interval a recurring job may use. Not a policy limit -- a zero or sub-second interval
/// makes `next_after` return an instant that is already in the past by the time it is stored, so
/// the job fires every poll tick forever.
const MIN_EVERY: Duration = Duration::from_secs(1);

/// How far ahead [`Schedule::next_after`] will search for a cron match before giving up. A pattern
/// like `0 0 30 2 *` (February 30th) matches no calendar date, and without a bound the search walks
/// forward indefinitely.
const CRON_SEARCH_HORIZON_DAYS: i64 = 366;

/// When a job fires.
///
/// `Cron` is boxed because [`croner::Cron`] carries a parsed component table an order of magnitude
/// larger than the other two variants, and a `Schedule` is cloned per poll tick.
#[derive(Debug, Clone)]
pub enum Schedule {
    /// Fire once at an instant, then delete. Always absolute: see [`Schedule::parse_at`] for why a
    /// relative input is resolved at creation rather than stored as written.
    At(DateTime<Utc>),
    /// Fire repeatedly, this far apart.
    Every(Duration),
    /// Fire on a 5-field cron pattern, evaluated in the host's local time.
    Cron(Box<Cron>),
}

impl Schedule {
    /// Parse a one-shot time: either an RFC 3339 timestamp or a humantime duration relative to
    /// `now` (`"20m"`, `"2h"`, `"1h 30m"`).
    ///
    /// A relative input is resolved to an absolute instant here and stored that way. Keeping it
    /// relative would be a job that never fires: every process restart would re-parse `"20m"` and
    /// push the target twenty minutes further into the future.
    pub fn parse_at(input: &str, now: DateTime<Utc>) -> Result<Self, String> {
        let input = input.trim();
        if let Ok(absolute) = DateTime::parse_from_rfc3339(input) {
            return Ok(Self::At(absolute.with_timezone(&Utc)));
        }
        let offset = parse_duration(input).map_err(|error| {
            format!(
                "'{}' is neither an RFC 3339 timestamp nor a duration: {}",
                input, error
            )
        })?;
        let offset = chrono::Duration::from_std(offset)
            .map_err(|_| format!("'{}' is too far in the future to schedule", input))?;
        now.checked_add_signed(offset)
            .map(Self::At)
            .ok_or_else(|| format!("'{}' is too far in the future to schedule", input))
    }

    /// Parse a recurring interval (`"30m"`, `"1h"`).
    ///
    /// Note that an interval shorter than the scheduler's poll tick fires once per tick, not once
    /// per interval; the tick is the real resolution floor.
    pub fn parse_every(input: &str) -> Result<Self, String> {
        let input = input.trim();
        let interval = parse_duration(input)
            .map_err(|error| format!("'{}' is not a valid duration: {}", input, error))?;
        if interval < MIN_EVERY {
            return Err(format!(
                "interval '{}' is below the {}s minimum",
                input,
                MIN_EVERY.as_secs()
            ));
        }
        Ok(Self::Every(interval))
    }

    /// Parse a 5-field cron pattern, evaluated in the host's local time.
    ///
    /// Rejects patterns that match no date within [`CRON_SEARCH_HORIZON_DAYS`], which is the only
    /// way to catch a well-formed but unsatisfiable pattern like `0 0 30 2 *` at creation instead
    /// of leaving a job that silently never fires.
    pub fn parse_cron(input: &str) -> Result<Self, String> {
        let input = input.trim();
        let cron = Cron::from_str(input)
            .map_err(|error| format!("'{}' is not a valid cron expression: {}", input, error))?;
        let schedule = Self::Cron(Box::new(cron));
        if schedule.next_after(Utc::now()).is_none() {
            return Err(format!(
                "cron expression '{}' matches no date within the next {} days",
                input, CRON_SEARCH_HORIZON_DAYS
            ));
        }
        Ok(schedule)
    }

    /// Rebuild a schedule from its two persisted columns. The inverse of
    /// [`Schedule::kind_str`] + [`Schedule::spec`].
    pub fn from_stored(kind: &str, spec: &str) -> Result<Self, String> {
        match kind {
            "at" => DateTime::parse_from_rfc3339(spec)
                .map(|absolute| Self::At(absolute.with_timezone(&Utc)))
                .map_err(|error| format!("stored 'at' spec '{}' is not RFC 3339: {}", spec, error)),
            "every" => Self::parse_every(spec),
            "cron" => Cron::from_str(spec)
                .map(|cron| Self::Cron(Box::new(cron)))
                .map_err(|error| format!("stored cron spec '{}' is invalid: {}", spec, error)),
            other => Err(format!("unknown schedule kind '{}'", other)),
        }
    }

    /// Discriminant as persisted in `scheduled_jobs.kind`.
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::At(_) => "at",
            Self::Every(_) => "every",
            Self::Cron(_) => "cron",
        }
    }

    /// Round-trippable form as persisted in `scheduled_jobs.spec`.
    pub fn spec(&self) -> String {
        match self {
            Self::At(instant) => instant.to_rfc3339(),
            Self::Every(interval) => humantime::format_duration(*interval).to_string(),
            Self::Cron(cron) => cron.pattern.to_string(),
        }
    }

    /// Whether firing this schedule leaves a job to reschedule. One-shots are deleted on fire.
    pub fn is_recurring(&self) -> bool {
        !matches!(self, Self::At(_))
    }

    /// The first occurrence strictly after `anchor`, or `None` when there is no next occurrence
    /// (a one-shot whose instant has passed, or a cron pattern matching no upcoming date).
    ///
    /// Callers must pass the job's own anchor -- `last_fired_at` if it has ever fired, otherwise
    /// `created_at` -- and never `Utc::now()`. Anchoring on the current time makes a pinned pattern
    /// such as `30 14 27 2 *` skip to next year whenever the process happens to restart after its
    /// window; anchoring permanently on `created_at` makes a long-lived job replay every occurrence
    /// since it was created.
    pub fn next_after(&self, anchor: DateTime<Utc>) -> Option<DateTime<Utc>> {
        match self {
            Self::At(instant) => (*instant > anchor).then_some(*instant),
            Self::Every(interval) => {
                let interval = chrono::Duration::from_std(*interval).ok()?;
                anchor.checked_add_signed(interval)
            }
            Self::Cron(cron) => {
                // Cron patterns are wall-clock expressions: "0 9 * * *" means 09:00 where the user
                // lives, so the search runs in local time and the result converts back to the UTC
                // meka stores.
                let local_anchor = anchor.with_timezone(&Local);
                let next = cron.find_next_occurrence(&local_anchor, false).ok()?;
                let next = next.with_timezone(&Utc);
                let horizon =
                    anchor.checked_add_signed(chrono::Duration::days(CRON_SEARCH_HORIZON_DAYS))?;
                (next <= horizon).then_some(next)
            }
        }
    }

    /// One-line human description, for the confirmation a tool hands back and for `schedule list`.
    /// A scheduling mistake is otherwise invisible until it fires, which may be days later.
    pub fn describe(&self) -> String {
        match self {
            Self::At(instant) => format!(
                "once at {}",
                instant.with_timezone(&Local).format("%Y-%m-%d %H:%M %Z")
            ),
            Self::Every(interval) => {
                format!("every {}", humantime::format_duration(*interval))
            }
            Self::Cron(cron) => cron.pattern.to_string(),
        }
    }
}

/// What a gate's result means. Chosen per job because the two answer different questions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateFire {
    /// Fire when the command's stdout differs from the previous run's. Edge-triggered: "tell me
    /// when the build *finishes*", not "tell me every 30s while it is running".
    OnChange,
    /// Fire while the command exits 0. Level-triggered, for "is this true yet".
    OnSuccess,
}

impl GateFire {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OnChange => "on-change",
            Self::OnSuccess => "on-success",
        }
    }

    pub fn parse(input: &str) -> Result<Self, String> {
        match input {
            "on-change" => Ok(Self::OnChange),
            "on-success" => Ok(Self::OnSuccess),
            other => Err(format!(
                "unknown gate mode '{}'; expected 'on-change' or 'on-success'",
                other
            )),
        }
    }
}

/// The cheap check that decides whether a due job spends a model turn.
///
/// This is the whole reason a 30-second cadence is affordable: without it, watching something costs
/// one model turn per interval whether or not anything happened.
#[derive(Debug, Clone)]
pub struct Gate {
    pub command: String,
    pub fire: GateFire,
    /// stdout from the last evaluation, for [`GateFire::OnChange`]. `None` until the first run, at
    /// which point the job fires: with nothing to compare against, "changed" is the honest answer,
    /// and it also proves the gate works rather than leaving it silently untested.
    pub last_output: Option<String>,
}

/// A persisted wakeup.
#[derive(Debug, Clone)]
pub struct ScheduledJob {
    pub id: String,
    pub session_id: uuid::Uuid,
    pub schedule: Schedule,
    pub prompt: String,
    pub gate: Option<Gate>,
    /// Run in a fresh sub-agent session rather than the conversation that created the job. Cheaper
    /// for anything recurring, since the parent's history is not replayed every fire.
    pub isolated: bool,
    pub created_at: DateTime<Utc>,
    pub last_fired_at: Option<DateTime<Utc>>,
    pub next_fire_at: DateTime<Utc>,
}

impl ScheduledJob {
    /// The instant [`Schedule::next_after`] must be measured from for this job.
    ///
    /// Exists so no call site has to remember the rule, because both ways of getting it wrong are
    /// silent: anchoring on `now` skips a pinned pattern to next year after an ill-timed restart,
    /// and anchoring permanently on `created_at` replays every occurrence since creation.
    pub fn anchor(&self) -> DateTime<Utc> {
        self.last_fired_at.unwrap_or(self.created_at)
    }

    /// Short id for display, matching the width `schedule_cancel` accepts.
    pub fn short_id(&self) -> &str {
        self.id.get(..8).unwrap_or(&self.id)
    }
}

/// Upper bound on the missed-occurrence count reported to the model. Counting is a courtesy ("you
/// missed 12 checks"), and walking a per-minute cron across a month-long outage to get an exact
/// figure is not worth the tick it would spend.
const MAX_COALESCED_REPORTED: u32 = 1000;

/// A due job, with everything the turn needs to know about why it is running now.
pub struct Wakeup {
    pub job: ScheduledJob,
    /// The gate's stdout, when the job has one. Carried so the model does not re-run the check the
    /// gate just ran.
    pub gate_output: Option<String>,
    /// How far past its due time this fire is. Near zero in normal operation; large after
    /// downtime.
    pub late_by: chrono::Duration,
    /// Occurrences this fire stands in for, beyond the one being delivered. Non-zero only after
    /// downtime, since the scheduler collapses a backlog into a single turn.
    pub coalesced: u32,
}

impl Wakeup {
    /// Render the user-turn text delivered to the model.
    ///
    /// The header is not decoration. Without it the model reads a bare instruction as if a human
    /// had just typed it and answers conversationally -- into an empty terminal at 03:00, to
    /// nobody.
    pub fn render_prompt(&self) -> String {
        let mut rendered = format!(
            "[Scheduled job {} fired {}]",
            self.job.short_id(),
            Utc::now().with_timezone(&Local).format("%Y-%m-%d %H:%M %Z")
        );
        // Only mention lateness when it is material. A tick's worth of delay is normal and saying
        // so every time would train the model to ignore the line that matters after an outage.
        if self.late_by > chrono::Duration::minutes(1) {
            rendered.push_str(&format!(
                "\n[Late by {}; this fire replaces {} missed occurrence(s)]",
                format_late(self.late_by),
                self.coalesced + 1
            ));
        }
        rendered.push_str("\n\n");
        rendered.push_str(&self.job.prompt);
        if let Some(output) = &self.gate_output {
            rendered.push_str("\n\n[Gate output]\n");
            rendered.push_str(output);
        }
        rendered
    }
}

/// Human-readable lateness, for the header above.
fn format_late(late_by: chrono::Duration) -> String {
    late_by
        .to_std()
        .map(|std| humantime::format_duration(Duration::from_secs(std.as_secs())).to_string())
        .unwrap_or_else(|_| "an unknown interval".to_string())
}

/// Ceiling on gate stdout carried into a turn. A gate is meant to yield a status line, not a
/// payload; anything past this is truncated so a runaway command cannot push the prompt over the
/// context window.
const GATE_OUTPUT_LIMIT: usize = 8 * 1024;

/// What a gate evaluation decided.
#[derive(Debug, Clone)]
pub struct GateOutcome {
    /// Whether to spend a model turn.
    pub fired: bool,
    /// stdout, trimmed and truncated. Handed to the turn as context when `fired`, and persisted as
    /// the comparison baseline for the next [`GateFire::OnChange`] evaluation.
    pub output: String,
}

/// Run a gate and decide whether the job it guards should fire.
///
/// Errors are for the gate itself failing (spawn failure, timeout), never for the condition being
/// false. The distinction matters: a watcher that goes quiet because its command broke looks
/// exactly like a healthy watcher with nothing to report, so the caller must surface an `Err`
/// rather than treating it as "no change".
///
/// The command runs unsandboxed. Authoring a gate already requires `write` permission, which is the
/// same level at which `execute_command` runs arbitrary unsandboxed commands, so a sandbox here
/// would block the ordinary cases (`gh`, `curl`) without raising the bar the agent must clear.
pub async fn evaluate_gate(gate: &Gate, timeout: Duration) -> Result<GateOutcome, String> {
    let mut builder = gate_command_builder(&gate.command);
    builder
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        // Dropping the future on timeout must not leave the command running until the next tick
        // spawns another. Note this reaps the direct child only: a gate whose shell backgrounds
        // something of its own can still orphan it, which is a reason to keep gates to simple
        // checks.
        .kill_on_drop(true);

    let child = builder
        .spawn()
        .map_err(|error| format!("failed to start gate: {}", error))?;

    let output = match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => output,
        Ok(Err(error)) => return Err(format!("gate failed to run: {}", error)),
        Err(_) => {
            return Err(format!(
                "gate exceeded its {} budget",
                humantime::format_duration(timeout)
            ));
        }
    };

    let stdout = truncate_gate_output(&String::from_utf8_lossy(&output.stdout));
    let fired = match gate.fire {
        // A first evaluation has no baseline, so "changed" is the honest answer. It also means a
        // freshly created watcher proves itself immediately instead of staying silent until
        // something happens, which is when a typo in the command would otherwise surface.
        GateFire::OnChange => gate.last_output.as_deref() != Some(stdout.as_str()),
        GateFire::OnSuccess => output.status.success(),
    };

    Ok(GateOutcome {
        fired,
        output: stdout,
    })
}

/// Build the platform's shell invocation for a gate, mirroring what `execute_command` does on its
/// unsandboxed path (`crate::tools::shell`).
fn gate_command_builder(command: &str) -> tokio::process::Command {
    #[cfg(windows)]
    {
        // Same UTF-8 prelude the shell tool uses: PowerShell 5.1 otherwise emits the legacy console
        // code page and non-ASCII output comes back as `?`, which would make an `on-change` gate
        // flap between encodings rather than on the thing it watches.
        let wrapped = crate::sandbox::wrap_command_with_utf8_output(command);
        let mut builder = tokio::process::Command::new("powershell.exe");
        builder
            .arg("-NoProfile")
            .arg("-NonInteractive")
            .arg("-Command")
            .arg(&wrapped);
        builder
    }
    #[cfg(not(windows))]
    {
        let mut builder = tokio::process::Command::new("sh");
        builder.arg("-c").arg(command);
        builder
    }
}

/// Trim and cap gate stdout. Trimming matters for correctness, not tidiness: most commands emit a
/// trailing newline, and comparing untrimmed output would be fine, but a command whose trailing
/// whitespace varies run to run would fire an `on-change` gate forever.
fn truncate_gate_output(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.len() <= GATE_OUTPUT_LIMIT {
        return trimmed.to_string();
    }
    // Cut on a character boundary so the result is still valid UTF-8.
    let mut end = GATE_OUTPUT_LIMIT;
    while end > 0 && !trimmed.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n[gate output truncated]", &trimmed[..end])
}

/// How many occurrences of `schedule` fall between `from` and `to`, capped at
/// [`MAX_COALESCED_REPORTED`].
fn occurrences_between(schedule: &Schedule, from: DateTime<Utc>, to: DateTime<Utc>) -> u32 {
    match schedule {
        // A one-shot has exactly the one occurrence, which is the fire being delivered.
        Schedule::At(_) => 0,
        Schedule::Every(interval) => {
            let interval = interval.as_secs();
            if interval == 0 {
                return 0;
            }
            let elapsed = (to - from).num_seconds().max(0) as u64;
            u32::try_from(elapsed / interval)
                .unwrap_or(MAX_COALESCED_REPORTED)
                .min(MAX_COALESCED_REPORTED)
        }
        Schedule::Cron(_) => {
            // Counts occurrences in `(from, to]`, i.e. everything after the one being delivered --
            // matching the `Every` arm above, which divides the same open interval.
            let mut cursor = from;
            let mut count = 0;
            while count < MAX_COALESCED_REPORTED {
                match schedule.next_after(cursor) {
                    Some(next) if next <= to => {
                        cursor = next;
                        count += 1;
                    }
                    _ => break,
                }
            }
            count
        }
    }
}

/// What a host did with a job handed to it.
///
/// Exists because deciding to fire and being *able* to fire are separate questions, and the gap
/// between them is where an occurrence can be lost. `prepare` consumes a job -- stamps it, advances
/// its schedule -- before the host is asked to run it, which is deliberate (a crashing prompt must
/// not re-fire forever). But a host that then cannot run it would silently eat the occurrence, so
/// it says so and the schedule is put back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FireOutcome {
    /// The turn ran, or failed in a way that re-running would not fix.
    Ran,
    /// This host could not take the job and another one should. The concrete case is `meka serve`
    /// finding the session's file lock held by a REPL: that REPL has its own watcher and will run
    /// the job itself, so the occurrence is restored rather than burnt.
    Deferred,
}

/// Which jobs a scheduler instance is responsible for.
///
/// The distinction is what makes `meka serve` the durable host and the REPL a best-effort one: the
/// server can revive any session on demand and so owns every job, while a REPL can only run turns
/// against the conversation it has open.
#[derive(Debug, Clone, Copy)]
pub enum SchedulerScope {
    /// Every job in the database. `meka serve`.
    AllSessions,
    /// Only jobs belonging to this session. The REPL.
    OneSession(uuid::Uuid),
}

impl SchedulerScope {
    fn covers(&self, job: &ScheduledJob) -> bool {
        match self {
            Self::AllSessions => true,
            Self::OneSession(id) => job.session_id == *id,
        }
    }
}

/// Start the scheduler loop. Returns the handle so the caller can abort it on shutdown; the task
/// runs until then.
///
/// Modelled on [`crate::server::gc::spawn`]: a tokio interval that wakes, queries, and hands work
/// to a host-supplied callback. Fires are awaited one at a time rather than spawned, so a process
/// with several due jobs runs one turn at a time. That bounds concurrent model spend, which matters
/// more here than latency: nobody is waiting on these.
pub fn spawn<Callback, Fired>(
    session_manager: std::sync::Arc<crate::session::SessionManager>,
    config: crate::config::ResolvedScheduleConfig,
    scope: SchedulerScope,
    fire: Callback,
) -> tokio::task::JoinHandle<()>
where
    Callback: Fn(Wakeup) -> Fired + Send + Sync + 'static,
    Fired: std::future::Future<Output = FireOutcome> + Send,
{
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(config.poll_interval);
        // The first tick resolves immediately; skip it so startup is not competing with provider
        // and MCP connection setup.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            if let Err(error) = run_due(&session_manager, &config, scope, &fire).await {
                // A failed sweep must not end the loop: a transient database error would otherwise
                // silently disable every scheduled job for the life of the process.
                tracing::warn!("scheduler tick failed: {}", error);
            }
        }
    })
}

/// One sweep: evaluate every due job in scope and fire what survives.
///
/// Public because the REPL drives it directly rather than from a timer. There, the agent loop owns
/// the conversation and must be the one to run the turn, so a watcher only nudges reedline awake
/// and this runs on the agent side. `meka serve` reaches it through [`spawn`] instead.
pub async fn run_due<Callback, Fired>(
    session_manager: &crate::session::SessionManager,
    config: &crate::config::ResolvedScheduleConfig,
    scope: SchedulerScope,
    fire: &Callback,
) -> crate::error::Result<()>
where
    Callback: Fn(Wakeup) -> Fired,
    Fired: std::future::Future<Output = FireOutcome>,
{
    let now = Utc::now();
    let due = session_manager.list_due_scheduled_jobs(now).await?;
    for job in due {
        if !scope.covers(&job) {
            continue;
        }
        // Cloned whole, before `prepare` claims it. Claiming a job rewrites its schedule, advances
        // its gate baseline, and for a one-shot deletes the row outright, so nothing short of the
        // original can put it back.
        let original = job.clone();
        if let Some(wakeup) = prepare(session_manager, config, job, now).await?
            && fire(wakeup).await == FireOutcome::Deferred
        {
            tracing::debug!("job {} deferred; restoring it", original.short_id());
            session_manager.restore_scheduled_job(&original).await?;
        }
    }
    Ok(())
}

/// Decide what to do with one due job: retire it, reschedule it quietly, or produce the [`Wakeup`]
/// that spends a turn.
async fn prepare(
    session_manager: &crate::session::SessionManager,
    config: &crate::config::ResolvedScheduleConfig,
    job: ScheduledJob,
    now: DateTime<Utc>,
) -> crate::error::Result<Option<Wakeup>> {
    let late_by = now - job.next_fire_at;
    let recurring = job.schedule.is_recurring();

    // A one-shot far past its moment is noise rather than a reminder: "join the standup" delivered
    // five days late helps nobody. Recurring jobs need no equivalent rule -- their occurrences are
    // one period apart, so the most recent missed one is always less than a period old.
    if !recurring
        && chrono::Duration::from_std(config.missed_grace)
            .map(|grace| late_by > grace)
            .unwrap_or(false)
    {
        tracing::warn!(
            "dropping one-shot job {}: due {} ago, past the missed-job grace period",
            job.short_id(),
            format_late(late_by)
        );
        session_manager.delete_scheduled_job(&job.id).await?;
        return Ok(None);
    }

    // Advance the schedule before doing anything that can fail or hang. A prompt that reliably
    // crashes the process would otherwise be re-selected on every restart, turning one bad job into
    // a boot loop in the daemon that is supposed to stay up. Paying for it with one missed
    // occurrence is the cheaper failure.
    let coalesced = occurrences_between(&job.schedule, job.next_fire_at, now);
    // `Some` only for a job that lives on. A one-shot's moment is spent, and a cron pattern with
    // nothing left in range has no future to be scheduled for; both are retired here and every
    // write below then has nothing to update.
    let next_fire_at = job.schedule.next_after(now).filter(|_| recurring);
    match next_fire_at {
        Some(next) => {
            session_manager
                .reschedule_scheduled_job(&job.id, next)
                .await?
        }
        None => session_manager.delete_scheduled_job(&job.id).await?,
    }

    let gate_output = match &job.gate {
        None => None,
        Some(gate) => match evaluate_gate(gate, config.gate_timeout).await {
            Ok(outcome) => {
                // Persist the new baseline even when it did not fire; that is exactly how an
                // `on-change` gate stops firing once it has seen the new value. A retired job has
                // no row left to write to, and needs none -- it will not be evaluated again.
                if next_fire_at.is_some()
                    && let Err(error) = session_manager
                        .update_scheduled_job_gate_output(&job.id, &outcome.output)
                        .await
                {
                    tracing::warn!(
                        "failed to record gate output for {}: {}",
                        job.short_id(),
                        error
                    );
                }
                if !outcome.fired {
                    tracing::debug!("gate for job {} declined to fire", job.short_id());
                    return Ok(None);
                }
                Some(outcome.output)
            }
            Err(error) => {
                // Loud on purpose. A watcher whose command breaks produces the same silence as a
                // watcher with nothing to report, and that is the failure most likely to go
                // unnoticed for weeks.
                tracing::warn!("gate for job {} failed: {}", job.short_id(), error);
                return Ok(None);
            }
        },
    };

    // Only a surviving job has an anchor worth recording. Recomputing the next fire here rather
    // than reusing `next_fire_at` would be the same value today, but it is the kind of duplicated
    // derivation that drifts: the reschedule above is the single writer of that column.
    if let Some(next) = next_fire_at {
        session_manager
            .stamp_scheduled_job_fired(&job.id, now, next)
            .await?;
    }

    tracing::info!(
        "firing scheduled job {} ({})",
        job.short_id(),
        job.schedule.describe()
    );
    Ok(Some(Wakeup {
        job,
        gate_output,
        late_by,
        coalesced,
    }))
}

/// Parse a humantime duration, via the same re-export `crate::config` uses so `every = "30m"` in a
/// tool call and `idle_timeout = "30m"` in `config.toml` mean the same thing.
///
/// Worth spelling out in any tool description built on this: humantime reads `m` as minutes and
/// `M` as months, so `1m` and `1M` differ by a factor of 43,800. Decimals and compound forms both
/// work (`1.5h` and `1h 30m` are the same duration).
fn parse_duration(input: &str) -> Result<Duration, humantime::DurationError> {
    humantime::parse_duration(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(text: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(text)
            .expect("test timestamp parses")
            .with_timezone(&Utc)
    }

    #[test]
    fn test_parse_at_accepts_rfc3339() {
        let now = at("2026-08-11T12:00:00Z");
        let schedule = Schedule::parse_at("2026-08-12T09:30:00Z", now).expect("parses");
        assert!(matches!(schedule, Schedule::At(instant) if instant == at("2026-08-12T09:30:00Z")));
    }

    /// A relative `at` must be resolved against `now` at parse time. Storing "20m" verbatim would
    /// produce a job that re-bases itself on every restart and therefore never fires.
    #[test]
    fn test_parse_at_resolves_relative_input_to_an_absolute_instant() {
        let now = at("2026-08-11T12:00:00Z");
        let schedule = Schedule::parse_at("20m", now).expect("parses");
        assert!(matches!(schedule, Schedule::At(instant) if instant == at("2026-08-11T12:20:00Z")));
        assert_eq!(schedule.spec(), at("2026-08-11T12:20:00Z").to_rfc3339());
    }

    #[test]
    fn test_parse_at_rejects_nonsense() {
        let now = at("2026-08-11T12:00:00Z");
        assert!(Schedule::parse_at("next tuesday", now).is_err());
        assert!(Schedule::parse_at("", now).is_err());
        assert!(
            Schedule::parse_at("2026-08-12", now).is_err(),
            "a bare date is not RFC 3339 and is not a duration either"
        );
    }

    /// `m` is minutes and `M` is months, a factor of roughly 43,800 apart. Pinned because a model
    /// writing `30M` for "half an hour" would schedule two and a half years out, and nothing else
    /// in the system would notice.
    #[test]
    fn test_duration_units_distinguish_minutes_from_months() {
        let minutes = Schedule::parse_every("30m").expect("parses");
        let months = Schedule::parse_every("30M").expect("parses");
        assert!(
            matches!(minutes, Schedule::Every(interval) if interval == Duration::from_secs(1800))
        );
        assert!(
            matches!(months, Schedule::Every(interval) if interval > Duration::from_secs(60 * 60 * 24 * 365))
        );
    }

    #[test]
    fn test_parse_every_rejects_sub_minimum_intervals() {
        // A zero interval yields a next-fire that is already in the past, so the job would fire on
        // every poll tick forever.
        assert!(Schedule::parse_every("0s").is_err());
        assert!(Schedule::parse_every("1s").is_ok());
    }

    #[test]
    fn test_parse_every_reads_m_as_minutes() {
        let schedule = Schedule::parse_every("30m").expect("parses");
        assert!(
            matches!(schedule, Schedule::Every(interval) if interval == Duration::from_secs(1800))
        );
    }

    #[test]
    fn test_parse_cron_accepts_five_fields() {
        // The reason croner is used over the `cron` crate: `cron` demands a seconds field, so the
        // five-field expressions users and models actually write would fail to parse.
        assert!(Schedule::parse_cron("0 9 * * 1-5").is_ok());
        assert!(Schedule::parse_cron("*/5 * * * *").is_ok());
    }

    #[test]
    fn test_parse_cron_rejects_unsatisfiable_pattern() {
        // Well-formed but matches no calendar date; caught at creation rather than leaving a job
        // that silently never fires.
        assert!(Schedule::parse_cron("0 0 30 2 *").is_err());
    }

    #[test]
    fn test_parse_cron_rejects_malformed_pattern() {
        assert!(Schedule::parse_cron("not a cron").is_err());
        assert!(Schedule::parse_cron("99 * * * *").is_err());
    }

    #[test]
    fn test_next_after_at_fires_once_then_never() {
        let instant = at("2026-08-12T09:00:00Z");
        let schedule = Schedule::At(instant);
        assert_eq!(
            schedule.next_after(at("2026-08-11T12:00:00Z")),
            Some(instant)
        );
        // Anchored at or past its own instant, a one-shot has no next occurrence.
        assert_eq!(schedule.next_after(instant), None);
        assert_eq!(schedule.next_after(at("2026-08-13T00:00:00Z")), None);
    }

    #[test]
    fn test_next_after_every_advances_one_interval_from_the_anchor() {
        let schedule = Schedule::parse_every("30m").expect("parses");
        assert_eq!(
            schedule.next_after(at("2026-08-11T12:00:00Z")),
            Some(at("2026-08-11T12:30:00Z"))
        );
    }

    /// The anchoring rule, stated as a test: a job that last fired ten days ago yields exactly one
    /// next fire, not ten. Coalescing missed occurrences is the scheduler's job, but it depends on
    /// `next_after` returning a single instant rather than a backlog.
    #[test]
    fn test_next_after_every_yields_one_occurrence_from_a_stale_anchor() {
        let schedule = Schedule::parse_every("1h").expect("parses");
        let stale = at("2026-08-01T12:00:00Z");
        assert_eq!(schedule.next_after(stale), Some(at("2026-08-01T13:00:00Z")));
    }

    #[test]
    fn test_schedule_round_trips_through_stored_columns() {
        for original in [
            Schedule::parse_at("2026-08-12T09:30:00Z", at("2026-08-11T12:00:00Z")).expect("parses"),
            Schedule::parse_every("45m").expect("parses"),
            Schedule::parse_cron("0 9 * * 1-5").expect("parses"),
        ] {
            let restored = Schedule::from_stored(original.kind_str(), &original.spec())
                .expect("stored form parses back");
            assert_eq!(restored.kind_str(), original.kind_str());
            assert_eq!(restored.spec(), original.spec());
        }
    }

    #[test]
    fn test_from_stored_rejects_unknown_kind() {
        assert!(Schedule::from_stored("on-exit", "make build").is_err());
    }

    fn gate(command: &str, fire: GateFire, last_output: Option<&str>) -> Gate {
        Gate {
            command: command.to_string(),
            fire,
            last_output: last_output.map(str::to_string),
        }
    }

    const GATE_BUDGET: Duration = Duration::from_secs(10);

    #[tokio::test]
    async fn test_on_success_gate_follows_the_exit_code() {
        let passing = evaluate_gate(&gate("exit 0", GateFire::OnSuccess, None), GATE_BUDGET)
            .await
            .expect("gate ran");
        assert!(passing.fired);

        let failing = evaluate_gate(&gate("exit 1", GateFire::OnSuccess, None), GATE_BUDGET)
            .await
            .expect("gate ran");
        assert!(
            !failing.fired,
            "a false condition is not an error, it is just no fire"
        );
    }

    #[tokio::test]
    async fn test_on_change_gate_fires_on_its_first_evaluation() {
        // No baseline means the watcher has never run. Firing proves the command works instead of
        // leaving a typo undiscovered until the thing being watched finally changes.
        let outcome = evaluate_gate(&gate("echo ready", GateFire::OnChange, None), GATE_BUDGET)
            .await
            .expect("gate ran");
        assert!(outcome.fired);
        assert_eq!(outcome.output, "ready");
    }

    #[tokio::test]
    async fn test_on_change_gate_is_quiet_until_the_output_differs() {
        let unchanged = evaluate_gate(
            &gate("echo steady", GateFire::OnChange, Some("steady")),
            GATE_BUDGET,
        )
        .await
        .expect("gate ran");
        assert!(!unchanged.fired, "same output must not spend a turn");

        let changed = evaluate_gate(
            &gate("echo moved", GateFire::OnChange, Some("steady")),
            GATE_BUDGET,
        )
        .await
        .expect("gate ran");
        assert!(changed.fired);
        assert_eq!(changed.output, "moved");
    }

    /// The failure mode this guards is the nastiest one in the feature: a broken watcher that
    /// reports nothing looks identical to a healthy watcher with nothing to report. A gate that
    /// overruns must surface as `Err`, never as a quiet `fired: false`.
    #[tokio::test]
    async fn test_a_gate_that_overruns_its_budget_is_an_error_not_a_silent_skip() {
        #[cfg(unix)]
        let command = "sleep 30";
        #[cfg(windows)]
        let command = "Start-Sleep -Seconds 30";

        let error = evaluate_gate(
            &gate(command, GateFire::OnChange, None),
            Duration::from_millis(150),
        )
        .await
        .expect_err("an overrunning gate must not report success");
        assert!(error.contains("budget"), "{error}");
    }

    #[tokio::test]
    async fn test_gate_output_is_trimmed_so_trailing_newlines_do_not_flap() {
        // `echo` appends a newline. Comparing untrimmed, a gate whose command varied its trailing
        // whitespace would fire forever.
        let outcome = evaluate_gate(&gate("echo spaced", GateFire::OnChange, None), GATE_BUDGET)
            .await
            .expect("gate ran");
        assert_eq!(outcome.output, "spaced");
    }

    // --- scheduler ---

    /// What a fire delivered, in the shape the assertions care about.
    #[derive(Debug, Clone)]
    struct FiredRecord {
        job_id: String,
        coalesced: u32,
        gate_output: Option<String>,
        late_by: chrono::Duration,
    }

    struct SchedulerHarness {
        manager: std::sync::Arc<crate::session::SessionManager>,
        session_id: uuid::Uuid,
        config: crate::config::ResolvedScheduleConfig,
        fired: std::sync::Arc<std::sync::Mutex<Vec<FiredRecord>>>,
    }

    impl SchedulerHarness {
        async fn new() -> Self {
            let manager = std::sync::Arc::new(
                crate::session::SessionManager::open(Some(std::path::Path::new(":memory:")))
                    .await
                    .expect("open in-memory database"),
            );
            let session_id = manager.create_session(None).await.expect("create session");
            Self {
                manager,
                session_id,
                config: crate::config::ResolvedScheduleConfig::default(),
                fired: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            }
        }

        /// Insert a job already overdue by `overdue`.
        async fn overdue_job(
            &self,
            schedule: Schedule,
            gate: Option<Gate>,
            overdue: chrono::Duration,
        ) -> ScheduledJob {
            let now = Utc::now();
            let job = ScheduledJob {
                id: uuid::Uuid::new_v4().to_string(),
                session_id: self.session_id,
                schedule,
                prompt: "do the thing".to_string(),
                gate,
                isolated: false,
                created_at: now - overdue - chrono::Duration::seconds(1),
                last_fired_at: None,
                next_fire_at: now - overdue,
            };
            self.manager
                .create_scheduled_job(&job)
                .await
                .expect("create job");
            job
        }

        async fn tick(&self) {
            let fired = self.fired.clone();
            run_due(
                &self.manager,
                &self.config,
                SchedulerScope::AllSessions,
                &move |wakeup: Wakeup| {
                    let fired = fired.clone();
                    async move {
                        if let Ok(mut guard) = fired.lock() {
                            guard.push(FiredRecord {
                                job_id: wakeup.job.id.clone(),
                                coalesced: wakeup.coalesced,
                                gate_output: wakeup.gate_output.clone(),
                                late_by: wakeup.late_by,
                            });
                        }
                        FireOutcome::Ran
                    }
                },
            )
            .await
            .expect("tick runs");
        }

        fn fired(&self) -> Vec<FiredRecord> {
            self.fired
                .lock()
                .map(|guard| guard.clone())
                .unwrap_or_default()
        }

        async fn jobs(&self) -> Vec<ScheduledJob> {
            self.manager
                .list_scheduled_jobs(self.session_id)
                .await
                .expect("list jobs")
        }
    }

    /// The headline missed-job rule: an outage does not become a burst. A 30-second job that was
    /// due six hours ago has 720 missed occurrences, and must produce exactly one turn.
    #[tokio::test]
    async fn test_a_long_outage_coalesces_into_a_single_fire() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::parse_every("30s").expect("parses"),
                None,
                chrono::Duration::hours(6),
            )
            .await;

        harness.tick().await;

        let fired = harness.fired();
        assert_eq!(fired.len(), 1, "one turn, not 720");
        assert_eq!(fired[0].job_id, job.id);
        // Six hours of 30-second occurrences: the one due at the stored time, plus 720 after it.
        assert_eq!(
            fired[0].coalesced, 720,
            "the skipped occurrences are reported, not replayed"
        );
    }

    /// The `Every` and `Cron` arms must agree on what they count, or the same outage reports a
    /// different backlog depending on how the schedule happened to be written.
    #[test]
    fn test_occurrence_counting_agrees_between_interval_and_cron() {
        let from = at("2026-08-11T12:00:00Z");
        let to = at("2026-08-11T12:05:00Z");
        let every = occurrences_between(&Schedule::parse_every("1m").expect("parses"), from, to);
        let cron = occurrences_between(
            &Schedule::parse_cron("* * * * *").expect("parses"),
            from,
            to,
        );
        assert_eq!(every, 5);
        assert_eq!(every, cron);
    }

    /// After firing late, the next due time is measured from now rather than from the slot that was
    /// missed. Rescheduling from the missed slot would leave the job still overdue and fire it
    /// again on the very next tick, which is the burst this avoids.
    #[tokio::test]
    async fn test_a_late_recurring_job_reschedules_from_now() {
        let harness = SchedulerHarness::new().await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::hours(5),
            )
            .await;

        harness.tick().await;
        assert_eq!(harness.fired().len(), 1);

        let job = harness.jobs();
        let job = job.await;
        let job = job.first().expect("recurring job survives");
        assert!(
            job.next_fire_at > Utc::now(),
            "next fire must be in the future, not still overdue"
        );

        // A second sweep must find nothing.
        harness.tick().await;
        assert_eq!(
            harness.fired().len(),
            1,
            "no second fire on the same tick cycle"
        );
    }

    #[tokio::test]
    async fn test_a_one_shot_past_the_grace_period_is_dropped_not_delivered() {
        let harness = SchedulerHarness::new().await;
        let stale = Utc::now() - chrono::Duration::days(5);
        harness
            .overdue_job(Schedule::At(stale), None, chrono::Duration::days(5))
            .await;

        harness.tick().await;

        assert!(
            harness.fired().is_empty(),
            "a five-day-old reminder is noise, not a reminder"
        );
        assert!(harness.jobs().await.is_empty(), "and it is retired");
    }

    #[tokio::test]
    async fn test_a_one_shot_inside_the_grace_period_fires_and_reports_its_lateness() {
        let harness = SchedulerHarness::new().await;
        let due = Utc::now() - chrono::Duration::hours(3);
        harness
            .overdue_job(Schedule::At(due), None, chrono::Duration::hours(3))
            .await;

        harness.tick().await;

        let fired = harness.fired();
        assert_eq!(fired.len(), 1);
        assert!(fired[0].late_by >= chrono::Duration::hours(3) - chrono::Duration::seconds(5));
        assert!(harness.jobs().await.is_empty(), "one-shots retire on fire");
    }

    /// A gate that declines must still advance the schedule, or the job stays overdue and the gate
    /// runs on every single tick instead of on its interval.
    #[tokio::test]
    async fn test_a_declining_gate_reschedules_without_spending_a_turn() {
        let harness = SchedulerHarness::new().await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate("exit 1", GateFire::OnSuccess, None)),
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        assert!(harness.fired().is_empty(), "the condition was false");
        let jobs = harness.jobs().await;
        let job = jobs.first().expect("job survives");
        assert!(job.next_fire_at > Utc::now(), "but the schedule moved on");
        assert!(
            job.last_fired_at.is_none(),
            "evaluating is not firing; recording it as fired would misreport the job"
        );
    }

    /// The nastiest failure in the feature: a broken gate must not look like a quiet one. It does
    /// not fire, but it also must not be recorded as having fired.
    #[tokio::test]
    async fn test_a_broken_gate_does_not_fire_and_does_not_claim_to_have() {
        let harness = SchedulerHarness::new().await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate(
                    "definitely-not-a-real-command-xyzzy",
                    GateFire::OnSuccess,
                    None,
                )),
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        assert!(harness.fired().is_empty());
        let jobs = harness.jobs().await;
        assert!(jobs.first().expect("job survives").last_fired_at.is_none());
    }

    /// A one-shot is retired the moment it comes due, before its gate is consulted: its moment has
    /// passed either way. The writes that follow a fire must therefore tolerate the row being gone,
    /// which is what this pins -- an earlier version issued them unconditionally and relied on the
    /// updates happening to match nothing.
    #[tokio::test]
    async fn test_a_one_shot_with_a_declining_gate_is_retired_without_firing() {
        let harness = SchedulerHarness::new().await;
        let due = Utc::now() - chrono::Duration::minutes(1);
        harness
            .overdue_job(
                Schedule::At(due),
                Some(gate("exit 1", GateFire::OnSuccess, None)),
                chrono::Duration::minutes(1),
            )
            .await;

        harness.tick().await;

        assert!(harness.fired().is_empty(), "the condition was false");
        assert!(
            harness.jobs().await.is_empty(),
            "and the one-shot is gone rather than left to retry forever"
        );
    }

    #[tokio::test]
    async fn test_gate_output_rides_along_to_the_turn() {
        let harness = SchedulerHarness::new().await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate("echo ci-red", GateFire::OnChange, None)),
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        let fired = harness.fired();
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].gate_output.as_deref(), Some("ci-red"));
        // The baseline is persisted, so an unchanged second evaluation stays quiet.
        let jobs = harness.jobs().await;
        assert_eq!(
            jobs.first()
                .and_then(|job| job.gate.as_ref())
                .and_then(|gate| gate.last_output.as_deref()),
            Some("ci-red")
        );
    }

    /// The REPL only owns the conversation it has open; a job belonging to another session must be
    /// left for whichever host can actually run it.
    #[tokio::test]
    async fn test_session_scope_ignores_jobs_from_other_sessions() {
        let harness = SchedulerHarness::new().await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;

        let fired = harness.fired.clone();
        run_due(
            &harness.manager,
            &harness.config,
            SchedulerScope::OneSession(uuid::Uuid::new_v4()),
            &move |wakeup: Wakeup| {
                let fired = fired.clone();
                async move {
                    if let Ok(mut guard) = fired.lock() {
                        guard.push(FiredRecord {
                            job_id: wakeup.job.id.clone(),
                            coalesced: wakeup.coalesced,
                            gate_output: None,
                            late_by: wakeup.late_by,
                        });
                    }
                    FireOutcome::Ran
                }
            },
        )
        .await
        .expect("tick runs");

        assert!(harness.fired().is_empty());
        assert_eq!(
            harness.jobs().await.len(),
            1,
            "and the job is untouched, not consumed"
        );
    }

    /// A host that cannot take a job must not consume its occurrence. `meka serve` hits this
    /// whenever a REPL holds the session's file lock: `prepare` has already stamped and advanced
    /// the job by the time the lock is attempted, so without the restore the job would be
    /// silently skipped on every tick for as long as the REPL stayed open.
    #[tokio::test]
    async fn test_a_deferred_job_keeps_its_occurrence() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;
        let due_before = job.next_fire_at;

        run_due(
            &harness.manager,
            &harness.config,
            SchedulerScope::AllSessions,
            &|_wakeup: Wakeup| std::future::ready(FireOutcome::Deferred),
        )
        .await
        .expect("tick runs");

        let jobs = harness.jobs().await;
        let job = jobs.first().expect("job survives a deferral");
        assert_eq!(
            job.next_fire_at, due_before,
            "the occurrence must be put back, not spent"
        );
        assert!(
            job.last_fired_at.is_none(),
            "and it must not be recorded as having fired"
        );

        // Still due, so the host that *can* run it sees it on its next sweep.
        assert_eq!(
            harness
                .manager
                .list_due_scheduled_jobs(Utc::now())
                .await
                .expect("list due")
                .len(),
            1
        );
    }

    /// The one-shot case of a deferral, which the recurring test above does not reach. Claiming a
    /// one-shot *deletes* its row, so a restore that only updated columns matched nothing, reported
    /// success, and lost the reminder for good. That is the concrete "remind me in 20 minutes"
    /// failure when `meka serve` and a REPL race for the same session.
    #[tokio::test]
    async fn test_a_deferred_one_shot_is_not_lost() {
        let harness = SchedulerHarness::new().await;
        let due = Utc::now() - chrono::Duration::minutes(1);
        let created = harness
            .overdue_job(Schedule::At(due), None, chrono::Duration::minutes(1))
            .await;

        run_due(
            &harness.manager,
            &harness.config,
            SchedulerScope::AllSessions,
            &|_wakeup: Wakeup| std::future::ready(FireOutcome::Deferred),
        )
        .await
        .expect("tick runs");

        let jobs = harness.jobs().await;
        assert_eq!(jobs.len(), 1, "the reminder must survive a deferral");
        let job = jobs.first().expect("job present");
        assert_eq!(job.id, created.id);
        assert_eq!(
            job.next_fire_at, created.next_fire_at,
            "still due, for the host that can run it"
        );
        assert!(job.last_fired_at.is_none());
    }

    /// A deferral must also put back the gate's baseline. `prepare` advances it before the host is
    /// asked to run the job, so restoring only the schedule would leave the watcher having already
    /// absorbed the change it exists to report: it would go quiet on the next evaluation and the
    /// event would never surface.
    #[tokio::test]
    async fn test_a_deferred_gated_job_keeps_its_baseline() {
        let harness = SchedulerHarness::new().await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate("echo changed", GateFire::OnChange, Some("original"))),
                chrono::Duration::minutes(1),
            )
            .await;

        run_due(
            &harness.manager,
            &harness.config,
            SchedulerScope::AllSessions,
            &|_wakeup: Wakeup| std::future::ready(FireOutcome::Deferred),
        )
        .await
        .expect("tick runs");

        let jobs = harness.jobs().await;
        assert_eq!(
            jobs.first()
                .and_then(|job| job.gate.as_ref())
                .and_then(|gate| gate.last_output.as_deref()),
            Some("original"),
            "the baseline must be as it was, so the change still fires for the next host"
        );
    }

    #[tokio::test]
    async fn test_rendered_prompt_marks_the_turn_as_scheduled() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::seconds(1),
            )
            .await;
        let wakeup = Wakeup {
            job,
            gate_output: Some("ci-red".to_string()),
            late_by: chrono::Duration::seconds(1),
            coalesced: 0,
        };
        let rendered = wakeup.render_prompt();
        assert!(rendered.starts_with("[Scheduled job "));
        assert!(rendered.contains("do the thing"));
        assert!(rendered.contains("[Gate output]\nci-red"));
        assert!(
            !rendered.contains("Late by"),
            "a second of tick latency is not worth reporting"
        );
    }

    #[test]
    fn test_truncate_gate_output_caps_and_marks() {
        let short = truncate_gate_output("  brief  ");
        assert_eq!(short, "brief");

        let long = truncate_gate_output(&"x".repeat(GATE_OUTPUT_LIMIT * 2));
        assert!(long.len() < GATE_OUTPUT_LIMIT * 2);
        assert!(long.ends_with("[gate output truncated]"));
    }

    /// Truncation cuts by byte offset, so a multi-byte character straddling the limit would panic a
    /// naive slice.
    #[test]
    fn test_truncate_gate_output_cuts_on_a_character_boundary() {
        // Three bytes wide, and the limit is not a multiple of three, so the cut lands
        // mid-character and the walk-back actually runs. A two-byte character would divide
        // the even limit exactly and never exercise it.
        assert_ne!(GATE_OUTPUT_LIMIT % 3, 0, "fixture relies on a ragged cut");
        let multibyte = "☃".repeat(GATE_OUTPUT_LIMIT);
        let truncated = truncate_gate_output(&multibyte);
        assert!(truncated.ends_with("[gate output truncated]"));
    }

    #[test]
    fn test_is_recurring_separates_one_shots() {
        assert!(!Schedule::At(at("2026-08-12T09:00:00Z")).is_recurring());
        assert!(Schedule::parse_every("1h").expect("parses").is_recurring());
        assert!(
            Schedule::parse_cron("0 9 * * *")
                .expect("parses")
                .is_recurring()
        );
    }
}
