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

use std::time::Duration;

use chrono::{DateTime, Local, Utc};
use croner::Cron;
// Reached through `humantime_serde`'s re-export rather than a direct dependency, which is also
// how `crate::config` gets at it. One duration syntax, one copy of the parser.
use humantime_serde::re::humantime;

use crate::error::MekaError;

/// Smallest interval a recurring job may use. Not a policy limit -- a zero or sub-second interval
/// makes `next_after` return an instant that is already in the past by the time it is stored, so
/// the job fires every poll tick forever.
const MIN_EVERY: Duration = Duration::from_secs(1);

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
    /// Rejects a well-formed but unsatisfiable pattern like `0 0 30 2 *` (February 30th) at
    /// creation, which is the only way to catch it instead of leaving a job that silently never
    /// fires. croner reports its own search-limit error for those, so this only has to ask it for
    /// one occurrence.
    pub fn parse_cron(input: &str) -> Result<Self, String> {
        let input = input.trim();
        // Five fields, explicitly. `Cron::from_str` defaults to `Seconds::Optional`, so a six-field
        // pattern parsed with the *first* field as seconds -- and `*/10 * * * * *`, written by a
        // model meaning "every 10 minutes" in the Quartz shape, became every 10 seconds instead.
        // The `MIN_EVERY` floor that stops `every` firing on each poll tick does not apply to
        // `cron`, so nothing else caught it, and the confirmation echoed the pattern back verbatim.
        let cron = croner::parser::CronParser::builder()
            .seconds(croner::parser::Seconds::Disallowed)
            .build()
            .parse(input)
            .map_err(|error| format!("'{}' is not a valid cron expression: {}", input, error))?;
        let schedule = Self::Cron(Box::new(cron));
        if schedule.next_after(Utc::now()).is_none() {
            return Err(format!(
                "cron expression '{}' matches no calendar date",
                input
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
            // Rehydrates through `parse_cron`, not `Cron::from_str`, so a stored spec is read back
            // under the same five-field grammar that accepted it. The permissive parser would read
            // a six-field pattern's first field as seconds, giving a stored job a different meaning
            // on reload than it had at creation. A spec that does not parse under those rules is
            // surfaced as an error rather than silently reinterpreted.
            "cron" => Self::parse_cron(spec)
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

    /// The occurrence to schedule after delivering the one at `delivered`, given the clock is now
    /// `now`.
    ///
    /// This is what a firing job needs, and it is **not** `next_after(now)`. For `Every`, anchoring
    /// on the current time adds the pickup latency to every interval, permanently: the scheduler
    /// polls, notices a job is due some milliseconds late, and schedules the next one a full
    /// interval from *that* moment rather than from the occurrence it just spent. Measured at
    /// `poll_interval = 1s`: `every = "5s"` ran at a mean of 5.83s, and `every = "1s"` at a clean
    /// 2.0s -- half the requested rate, because the tick one interval later lands microseconds
    /// early and the job loses a whole poll.
    ///
    /// Advancing by whole intervals from `delivered` preserves the phase, so a job fires on the
    /// grid it was created on however late any single pickup is. A backlog is skipped in one
    /// multiplication rather than a loop, which matters after an outage: `occurrences_between`
    /// separately reports how many were coalesced.
    ///
    /// `Cron` is unaffected -- its occurrences are absolute wall-clock instants, so the next one
    /// after `now` is the next one, and `At` has no successor at all.
    pub fn next_after_delivering(
        &self,
        delivered: DateTime<Utc>,
        now: DateTime<Utc>,
    ) -> Option<DateTime<Utc>> {
        match self {
            Self::At(_) => None,
            Self::Every(interval) => {
                let step = chrono::Duration::from_std(*interval).ok()?;
                // Milliseconds, not seconds. `num_seconds` truncates, and `parse_every` accepts
                // anything from 1s up while humantime parses `"1500ms"` and `"1s 500ms"`, which
                // round-trip through `spec()`. With the interval truncated to 1s, a `1500ms` job
                // advanced by whole seconds and walked off its own grid a little further on every
                // fire -- never looping, because the result stayed strictly after `now`, just
                // drifting. That is exactly the property this function exists to hold.
                let step_millis = step.num_milliseconds();
                // A zero or negative interval has no grid to stay on; fall back rather than divide
                // by zero.
                if step_millis <= 0 {
                    return self.next_after(now);
                }
                let mut next = delivered.checked_add_signed(step)?;
                if next <= now {
                    let behind = (now - next).num_milliseconds();
                    let skips = behind / step_millis + 1;
                    next = next.checked_add_signed(chrono::Duration::milliseconds(
                        skips.checked_mul(step_millis)?,
                    ))?;
                }
                Some(next)
            }
            Self::Cron(_) => self.next_after(now),
        }
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
                // croner bounds its own forward search and reports a search-limit error for a
                // pattern that matches no calendar date, so its verdict is the whole answer. An
                // extra horizon here would be indistinguishable from that verdict at the call site,
                // and `prepare` retires a job whose schedule has no next occurrence: a 366-day one
                // deleted `0 0 29 2 *` the first time it fired, because the next February 29th is
                // up to four years out.
                let next = cron.find_next_occurrence(&local_anchor, false).ok()?;
                Some(next.with_timezone(&Utc))
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
    /// The permission level the creating session held when this gate was authorised.
    ///
    /// A gate is a shell command that runs unattended, unsandboxed, on a timer, in whatever
    /// process happens to pick the job up. Creation requires
    /// [`crate::permission::Permission::Unrestricted`], but creation is a moment and the row
    /// outlives it: the session drops to `read`, or `meka serve --permission read` restarts
    /// and inherits the job, and without this field nothing downstream can tell that the
    /// authority behind the command is gone. Carrying the level on the row is what lets
    /// [`prepare`] re-check it at fire time instead of trusting a decision made days ago.
    pub permission: crate::permission::Permission,
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

    /// What should become of this job's prompt if the turn it triggers fails before the model ever
    /// sees it.
    ///
    /// Defined once here rather than at each host, because every host has to answer it and the rule
    /// is not obvious enough to restate three times. A recurring job produces the prompt again on
    /// its next occurrence, so a failure withdrawing it costs nothing and spares the conversation
    /// one unanswered message per fire through an outage. A one-shot does not: [`prepare`] deletes
    /// its row *before* the host runs the turn, so the unanswered message is the last trace that
    /// the reminder ever fired, and withdrawing it would be the deletion the feature is
    /// supposed to prevent.
    pub fn prompt_retention(&self) -> crate::agent::PromptRetention {
        match self.schedule.is_recurring() {
            true => crate::agent::PromptRetention::WithdrawOnFailure,
            false => crate::agent::PromptRetention::Keep,
        }
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
/// The command runs unsandboxed. Authoring a gate already requires `unrestricted` permission, which
/// is the same level at which `execute_command` runs arbitrary unsandboxed commands, so a sandbox
/// here would block the ordinary cases (`gh`, `curl`) without raising the bar the agent must clear.
pub async fn evaluate_gate(
    gate: &Gate,
    timeout: Duration,
    cwd: Option<&std::path::Path>,
) -> Result<GateOutcome, String> {
    let mut builder = gate_command_builder(&gate.command);
    // The creating session's directory, not the host process's. A gate is almost always written by
    // the model right after verifying the same command through `execute_command`, which runs in the
    // session's cwd -- so a gate that runs anywhere else silently stops matching the command the
    // model tested. Under a `meka serve` systemd unit the process cwd is `/`, where a repo-relative
    // `gh pr checks` exits non-zero with empty stdout, and an `on-change` gate then latches onto
    // that empty baseline and never fires again.
    if let Some(directory) = cwd {
        if directory.is_dir() {
            builder.current_dir(directory);
        } else {
            tracing::warn!(
                "gate's session directory '{}' no longer exists; running it in the current \
                 directory instead",
                directory.display(),
            );
        }
    }
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

    // A non-zero exit from an `on-change` gate is reported, not refused.
    //
    // The failure this exists for is a watcher that breaks silently: an expired token has `gh` exit
    // non-zero with empty stdout, the first evaluation stores `""` as the baseline, and every
    // evaluation after compares `"" == ""` and stays quiet forever. A log line makes that visible.
    //
    // Refusing to fire would not: for a large class of perfectly good gates, a non-zero exit *is*
    // the signal. `diff -q a b` and `git diff --exit-code` exit 1 exactly when there is a
    // difference; `grep ERROR log` exits 1 through the whole quiet period it is watching; `curl -f`
    // exits non-zero until the endpoint comes back. Treating any of those as broken would silence
    // the gate permanently, which is the bug this was meant to fix, pointed the other way.
    if matches!(gate.fire, GateFire::OnChange) && !output.status.success() {
        let stderr = truncate_gate_output(&String::from_utf8_lossy(&output.stderr));
        tracing::warn!(
            "on-change gate exited with {}{}; comparing its output anyway, since a non-zero exit \
             is how several common gates signal a change",
            output.status,
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {}", stderr)
            }
        );
    }

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
            // Milliseconds, matching `next_after_delivering`: `as_secs` truncated, so a `1500ms`
            // job counted its coalesced occurrences against a 1s grid it does not run on.
            let interval = interval.as_millis();
            if interval == 0 {
                return 0;
            }
            let elapsed = (to - from).num_milliseconds().max(0) as u128;
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
/// Every host answers a predicate rather than claiming a static set, because none of them can
/// actually take every job: `meka serve` can revive any session but not one another process has
/// locked, the REPL owns exactly the conversation it has open, and ACP owns whatever the editor
/// currently has open.
///
/// Asked here rather than letting a host decline afterwards, and that placement is the whole point:
/// `prepare` evaluates a job's *gate* before the host is offered the wakeup, so a scope that
/// admitted everything would run every gated job's shell command on every tick for sessions it
/// could never serve.
#[derive(Clone)]
pub enum SchedulerScope {
    /// Only jobs belonging to this session. The REPL, which has exactly one conversation open.
    OneSession(uuid::Uuid),
    /// Jobs the predicate accepts, re-asked every sweep, so a host whose set of runnable jobs moves
    /// under it (ACP's open editors, serve's session locks) is never working from a snapshot.
    ///
    /// Takes the whole job rather than its session id because "can this host run it" is not always
    /// a question about the session: an `isolated` job runs in a fresh conversation and needs
    /// nothing from the one that created it, including its lock.
    Jobs(std::sync::Arc<dyn Fn(&ScheduledJob) -> bool + Send + Sync>),
}

impl std::fmt::Debug for SchedulerScope {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OneSession(id) => write!(formatter, "OneSession({})", id),
            Self::Jobs(_) => formatter.write_str("Jobs(<predicate>)"),
        }
    }
}

impl SchedulerScope {
    /// Every job in the database, whoever it belongs to. Only the tests want this: a real host
    /// always has some job it cannot take.
    #[cfg(test)]
    fn every_job() -> Self {
        Self::Jobs(std::sync::Arc::new(|_| true))
    }

    fn covers(&self, job: &ScheduledJob) -> bool {
        match self {
            Self::OneSession(id) => job.session_id == *id,
            Self::Jobs(predicate) => predicate(job),
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
        // A sweep contains its turns, so it routinely overruns `poll_interval` by minutes. Tokio's
        // default `Burst` then resolves *every* tick missed during it, so a sweep that ran twelve
        // periods long is followed by twelve immediate sweeps, eleven of which find nothing due.
        // `Delay` collapses that to one.
        //
        // It does not create a gap, and nothing here does: `Delay` schedules the next tick a period
        // after the miss is *recognised*, so the first `tick()` following a long sweep still
        // returns at once. Batches are therefore adjacent, and `max_consecutive_fires`
        // splits a backlog without spacing it out.
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // The first tick resolves immediately; skip it so startup is not competing with provider
        // and MCP connection setup.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            // A panic must not end the loop either, and here that matters more than for the GC
            // scanner this pattern comes from: under `meka serve` the callback runs a whole agent
            // turn, so the surface that can panic is the entire tool loop. Losing the task would
            // stop every scheduled job for the life of the process, and stop it silently -- nothing
            // joins this handle, so the only symptom is jobs that quietly never fire again.
            let sweep =
                std::panic::AssertUnwindSafe(run_due(&session_manager, &config, &scope, &fire));
            match futures::FutureExt::catch_unwind(sweep).await {
                // A failed sweep must not end the loop: a transient database error would otherwise
                // silently disable every scheduled job for the life of the process.
                Ok(Err(error)) => tracing::warn!("scheduler tick failed: {}", error),
                Err(panic) => tracing::error!(
                    "scheduler tick panicked ({}); continuing",
                    crate::error::panic_message(&*panic)
                ),
                Ok(Ok(())) => {}
            }
        }
    })
}

/// One sweep: evaluate every due job in scope and fire what survives, up to
/// [`crate::config::ResolvedScheduleConfig::max_consecutive_fires`] turns per session.
///
/// Public because the REPL drives it directly rather than from a timer. There, the agent loop owns
/// the conversation and must be the one to run the turn, so a watcher only nudges reedline awake
/// and this runs on the agent side. `meka serve` reaches it through [`spawn`] instead.
///
/// What the budget buys is a seam, not a ceiling. A sweep contains the turns it fires -- they are
/// awaited here -- so forty due jobs still cost forty turns however small the budget is. What
/// changes is that they arrive in groups, so another session's due job is reached after five of the
/// first session's rather than after all forty. The groups are adjacent rather than spaced -- a
/// sweep that overran `poll_interval` leaves its successor already due -- so this splits a backlog
/// without slowing it. Bounding how much one conversation absorbs in total would mean holding jobs
/// across sweeps, which this deliberately does not do.
pub async fn run_due<Callback, Fired>(
    session_manager: &crate::session::SessionManager,
    config: &crate::config::ResolvedScheduleConfig,
    scope: &SchedulerScope,
    fire: &Callback,
) -> crate::error::Result<()>
where
    Callback: Fn(Wakeup) -> Fired,
    Fired: std::future::Future<Output = FireOutcome>,
{
    let now = Utc::now();
    let store = session_manager.schedule_store();
    let due = store.list_due_scheduled_jobs(now).await?;
    let mut fired: std::collections::HashMap<uuid::Uuid, usize> = std::collections::HashMap::new();
    let mut held_over = 0usize;
    for job in due {
        if !scope.covers(&job) {
            continue;
        }
        // Checked *before* `prepare`, which is what makes holding a job over free: `prepare` is
        // where a gate runs and where the schedule is advanced, so a job skipped here has done
        // neither and is still due, unchanged, on the next sweep. Reaching `prepare` and then
        // declining would pay a gate evaluation and a `restore_scheduled_job` round trip to arrive
        // at the same place.
        //
        // `list_due_scheduled_jobs` orders by `next_fire_at`, so a held-over job is still the most
        // overdue one next time and goes first. Nothing starves.
        if fired.get(&job.session_id).copied().unwrap_or(0) >= config.max_consecutive_fires {
            held_over += 1;
            continue;
        }
        // Cloned whole, before `prepare` claims it. Claiming a job rewrites its schedule, advances
        // its gate baseline, and for a one-shot deletes the row outright, so nothing short of the
        // original can put it back.
        let original = job.clone();
        // `warn!`, not `?`. The same treatment `stamp_scheduled_job_fired` was given one level
        // down, and for the reason recorded there: propagating aborted the whole sweep, so one
        // transient `SQLITE_BUSY` skipped every *other* job due in the same tick. A job that was
        // never claimed comes back next tick, so the cost of continuing is nothing.
        let prepared = match prepare(session_manager, config, job, now).await {
            Ok(prepared) => prepared,
            Err(error) => {
                tracing::warn!(
                    "could not prepare job {}: {}. It keeps its occurrence and will be \
                     reconsidered on the next tick",
                    original.short_id(),
                    error
                );
                continue;
            }
        };
        if let Some((claim, wakeup)) = prepared {
            if fire(wakeup).await == FireOutcome::Deferred {
                tracing::debug!("job {} deferred; restoring it", original.short_id());
                // Also `warn!`: this arm has *already* claimed the occurrence, so propagating both
                // lost it and took the rest of the sweep with it -- the one combination worth
                // avoiding. Losing a deferral's occurrence is bad; losing every later job's too is
                // worse.
                if let Err(error) = store.restore_scheduled_job(&original, claim).await {
                    tracing::warn!(
                        "job {} was deferred but could not be restored: {}. Its occurrence is \
                         spent",
                        original.short_id(),
                        error
                    );
                }
            } else {
                // Counted only once a turn has actually been spent. A job `prepare` retired (a
                // declining gate, a one-shot past its grace period) and a job the host handed back
                // both cost the conversation nothing, so neither may consume a session's budget --
                // otherwise five quiet watchers would starve the sixth job that had something to
                // say.
                *fired.entry(original.session_id).or_default() += 1;
            }
        }
    }
    // Said out loud: a cap that bounds coverage silently reads as "everything ran". `info!` rather
    // than `warn!` because holding a job over is the budget working, not a fallback -- the jobs are
    // intact and the next sweep takes them.
    if held_over > 0 {
        tracing::info!(
            "held over {} due job(s) past [schedule].max_consecutive_fires ({}); they keep their \
             occurrence and run on the next sweep",
            held_over,
            config.max_consecutive_fires
        );
    }
    Ok(())
}

/// Jobs currently held back because their gate's authority was withdrawn.
///
/// Exists only to keep the explanation to once per episode. The check runs on every sweep, and the
/// state it reports does not change between them, so warning per evaluation turns one fact into a
/// line a minute for as long as the session stays below write.
static PERMISSION_DECLINED: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashSet<String>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashSet::new()));

/// True the first time a job is held back for permission, false while it stays held back.
///
/// Entries are dropped when the job is authorised again *or* when it stops being a job meka can
/// see, so a cancelled-while-declined job does not sit here for the life of the process. The set is
/// one string per held-back job, which is small, but a long-lived `meka serve` has no other bound
/// on it.
fn declined_for_permission_first_time(job_id: &str) -> bool {
    match PERMISSION_DECLINED.lock() {
        Ok(mut held) => held.insert(job_id.to_string()),
        Err(poisoned) => poisoned.into_inner().insert(job_id.to_string()),
    }
}

/// Forget a job's held-back state, so the next withdrawal is announced again.
fn clear_permission_decline(job_id: &str) {
    match PERMISSION_DECLINED.lock() {
        Ok(mut held) => held.remove(job_id),
        Err(poisoned) => poisoned.into_inner().remove(job_id),
    };
}

/// What claiming an occurrence wrote to the row.
///
/// Carried out of [`prepare`] so a host that turns out to be unable to run the job puts back
/// exactly what it took, and so every write that follows the claim can be scoped to the claim *this
/// process* won rather than to the job id alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Claim {
    /// The row survives, advanced to this next fire time.
    Advanced(DateTime<Utc>),
    /// The row was deleted: a one-shot's moment is spent, and a cron pattern with nothing left in
    /// range has no future to be scheduled for.
    Retired,
}

impl Claim {
    /// The fire time this claim wrote, or `None` for a claim that retired the row.
    ///
    /// Every write after the claim needs it twice over: as the value to guard on, and as the answer
    /// to "is there still a row to write to at all".
    fn advanced_to(self) -> Option<DateTime<Utc>> {
        match self {
            Self::Advanced(next) => Some(next),
            Self::Retired => None,
        }
    }
}

/// Decide what to do with one due job: retire it, reschedule it quietly, or produce the [`Wakeup`]
/// that spends a turn.
async fn prepare(
    session_manager: &crate::session::SessionManager,
    config: &crate::config::ResolvedScheduleConfig,
    job: ScheduledJob,
    now: DateTime<Utc>,
) -> crate::error::Result<Option<(Claim, Wakeup)>> {
    let store = session_manager.schedule_store();
    let late_by = now - job.next_fire_at;
    let recurring = job.schedule.is_recurring();
    // Kept because the claim below consumes the row: for a one-shot `claim_by_retiring` is a
    // `DELETE`, and every path that decides *not* to fire has to be able to put it back.
    let original = job.clone();

    // A one-shot far past its moment is noise rather than a reminder: "join the standup" delivered
    // five days late helps nobody. Recurring jobs need no equivalent rule -- their occurrences are
    // one period apart, so the most recent missed one is always less than a period old.
    if !recurring
        && chrono::Duration::from_std(config.missed_grace)
            .map(|grace| late_by > grace)
            .unwrap_or(false)
    {
        // Claimed rather than deleted outright, so the announcement is made once by whichever host
        // actually removed the row rather than once per host that had it in its due list.
        if store.claim_by_retiring(&job.id, job.next_fire_at).await? {
            tracing::warn!(
                "dropping one-shot job {}: due {} ago, past the missed-job grace period",
                job.short_id(),
                format_late(late_by)
            );
        }
        return Ok(None);
    }

    // Advance the schedule before doing anything that can fail or hang. A prompt that reliably
    // crashes the process would otherwise be re-selected on every restart, turning one bad job into
    // a boot loop in the daemon that is supposed to stay up. Paying for it with one missed
    // occurrence is the cheaper failure.
    //
    // This write is also what arbitrates between hosts, which is why it is conditional. Every
    // `meka serve`, REPL and ACP session polls the same table, so one occurrence is in several
    // hosts' due lists at once; whoever moves the row off the value they all read owns it, and the
    // rest return here having neither evaluated the gate nor spent the occurrence.
    let coalesced = occurrences_between(&job.schedule, job.next_fire_at, now);
    // `Some` only for a job that lives on. A one-shot's moment is spent, and a cron pattern with
    // nothing left in range has no future to be scheduled for; both are retired here and every
    // write below then has nothing to update.
    let next_fire_at = job
        .schedule
        .next_after_delivering(job.next_fire_at, now)
        .filter(|_| recurring);
    let claim = match next_fire_at {
        Some(next) => match store
            .claim_by_advancing(&job.id, job.next_fire_at, next)
            .await?
        {
            true => Claim::Advanced(next),
            false => {
                tracing::debug!(
                    "job {} was claimed for this occurrence by another host",
                    job.short_id()
                );
                return Ok(None);
            }
        },
        None => match store.claim_by_retiring(&job.id, job.next_fire_at).await? {
            true => {
                // The row is gone, so its held-back state is dead weight. Nothing else removes an
                // entry, and a long-lived `meka serve` retires a one-shot on every `at` job it
                // ever runs.
                clear_permission_decline(&job.id);
                Claim::Retired
            }
            false => {
                tracing::debug!(
                    "job {} was claimed for this occurrence by another host",
                    job.short_id()
                );
                return Ok(None);
            }
        },
    };

    // A gate that *errored* answered nothing, so a one-shot must keep its row.
    //
    // The claim is taken before the gate runs, and for a one-shot that claim is a `DELETE`. The
    // error arm then returned without restoring, so a 30s `gate_timeout` overrun or a transient
    // `sh` spawn failure destroyed the job outright -- `schedule_list` showed nothing and the
    // user's reminder had silently ceased to exist, with `gate for job X failed` in the log as the
    // only trace, a line that reads as "this evaluation failed" rather than "the job is gone".
    // Losing data because the condition could not be evaluated is different in kind from losing it
    // because the condition said no.
    //
    // Deliberately *not* applied to the declining arm: a one-shot is retired the moment it comes
    // due whether or not the gate fires, because its moment has passed either way. That is the
    // documented rule and `test_a_one_shot_with_a_declining_gate_is_retired_without_firing` pins
    // it. Recurring jobs are left alone in both arms -- their occurrence was spent considering the
    // condition, and restoring one would re-fire it immediately.
    async fn keep_a_one_shot(
        store: &ScheduleStore,
        original: &ScheduledJob,
        claim: Claim,
    ) -> crate::error::Result<()> {
        if matches!(claim, Claim::Retired) {
            store.restore_scheduled_job(original, claim).await?;
        }
        Ok(())
    }

    // Looked up only when there is a gate to run, so an ungated job costs no query. One lookup
    // serves both the working directory and the live permission the re-check below needs.
    let gate_session = match &job.gate {
        None => None,
        Some(_) => match session_manager.session_info(job.session_id).await {
            Ok(info) => info,
            Err(error) => {
                // Not silently "no session". A failed lookup means the level cannot be confirmed,
                // and the re-check below has to fail closed on that rather than fall back to the
                // recorded value.
                tracing::warn!(
                    "could not read session {} while preparing job {}: {}",
                    job.session_id,
                    job.short_id(),
                    error
                );
                None
            }
        },
    };
    let gate_cwd = gate_session.as_ref().and_then(|info| info.cwd.clone());

    // What the session's permission is *now*, as opposed to what it was when the gate was authored.
    //
    // `None` means the row carries no per-session level, so the host's own level is the live
    // answer. That is now only an ACP session that has never been through `session/set_mode`
    // (`session/new` writes no level) or an imported archive that carried none: `POST /v1/sessions`
    // records it at insert, and `run_turn` records it for the REPL, one-shot and sub-agent rows it
    // creates. A session row that exists but cannot be read also leaves this `None` and the host
    // level decides, which is why the lookup failure above warns rather than passing silently.
    //
    // Filtered by the enabled set, like the other four readers of this column. A row records what a
    // session was set to, not what this installation still permits, and the two diverge the moment
    // an operator narrows `[permissions].enabled` and restarts: the session re-attaches clamped,
    // while this read saw the unclamped row and kept firing the gate. That was verified end to end
    // against a live `meka serve` -- and the creation door two files over returns 403 for the same
    // authority, so the two doors disagreed about one job.
    let live_permission = crate::permission::parse_recorded_permission(
        gate_session
            .as_ref()
            .and_then(|info| info.permission.as_deref()),
        &format_args!("session {}", job.session_id),
    )
    .filter(|level| config.enabled_permissions.is_enabled(*level))
    .unwrap_or(config.host_permission);

    let gate_output = match &job.gate {
        None => None,
        // Both the recorded level and the live one must still say `Unrestricted`.
        //
        // Checking only the recorded value was a tautology: `schedule_create` and the HTTP handler
        // each demand `Unrestricted` before writing the row, and nothing ever updates the column,
        // so the recorded value is `unrestricted` for every job that exists. The comparison
        // could not fail,
        // and the case it was written for -- the session cycles down to `read`, or a
        // `meka serve --permission read` restarts and inherits the row -- went unnoticed. The live
        // level is what makes the withdrawal real; the recorded one still matters because a
        // hand-edited or unparseable `gate_permission` decodes as `Permission::None` and must stay
        // refused.
        //
        // The occurrence is declined, exactly as a gate that ran and said no is declined. A gate
        // is the condition on the job, so a gate that could not be evaluated has not passed, and
        // firing anyway converts a conditional job into an unconditional one. The shape that makes
        // this concrete is `every = "1m"` with an `on-change` gate: firing it unconditionally turns
        // a near-silent job into a turn a minute, which is the opposite of what the row asks for
        // and expensive besides.
        Some(gate)
            if !gate.permission.allows_unattended_shell()
                || !live_permission.allows_unattended_shell() =>
        {
            // Said once per downgrade, not once per evaluation. The condition is a standing state
            // rather than an event: a session left below the bar with an `every = "1m"` job wrote
            // this line every minute for as long as it stayed there, which buries the log it is
            // supposed to be the signal in. The id is cleared the moment the gate is authorised
            // again, so a later downgrade is announced afresh.
            if declined_for_permission_first_time(&job.id) {
                tracing::warn!(
                    "job {} not fired: its gate was authorised at {} and the session is currently \
                     {}, and an unattended shell command needs `unrestricted` at \
                     both. Raise the session back to restore it",
                    job.short_id(),
                    gate.permission,
                    live_permission,
                );
            } else {
                tracing::debug!(
                    "job {} still not fired: the session remains at {}",
                    job.short_id(),
                    live_permission,
                );
            }
            return Ok(None);
        }
        Some(gate) => {
            // Authorised again, so the next withdrawal is announced rather than swallowed.
            clear_permission_decline(&job.id);
            match evaluate_gate(gate, config.gate_timeout, gate_cwd.as_deref()).await {
                Ok(outcome) => {
                    // Persist the new baseline even when it did not fire; that is exactly how an
                    // `on-change` gate stops firing once it has seen the new value. A retired job
                    // has no row left to write to, and needs none -- it will
                    // not be evaluated again.
                    if let Some(claimed) = claim.advanced_to()
                        && let Err(error) = store
                            .update_scheduled_job_gate_output(&job.id, claimed, &outcome.output)
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
                    // Loud on purpose. A watcher whose command breaks produces the same silence as
                    // a watcher with nothing to report, and that is the failure
                    // most likely to go unnoticed for weeks.
                    tracing::warn!("gate for job {} failed: {}", job.short_id(), error);
                    keep_a_one_shot(&store, &original, claim).await?;
                    return Ok(None);
                }
            }
        }
    };

    // Only a surviving job has an anchor worth recording. Recomputing the next fire here rather
    // than reusing the claim would be the same value today, but it is the kind of duplicated
    // derivation that drifts: the claim above is the single writer of that column.
    if let Some(claimed) = claim.advanced_to() {
        // Logged rather than propagated, like the two writes either side of it.
        //
        // `?` here spent the occurrence and then produced no `Wakeup`, so the turn never ran -- and
        // because `run_due` returns the error, every *other* job due in the same sweep was skipped
        // too. One transient database hiccup therefore silently dropped a whole tick's worth of
        // reminders, with `scheduler tick failed` at `warn!` as the only trace. The claim has
        // already succeeded at this point; the anchor is bookkeeping, and a stale one costs at most
        // a slightly wrong "late by" on the next fire.
        if let Err(error) = store.stamp_scheduled_job_fired(&job.id, now, claimed).await {
            tracing::warn!(
                "could not record the fire time for job {}: {}. The job has already been claimed \
                 for this occurrence and will still run",
                job.short_id(),
                error
            );
        }
    }

    tracing::info!(
        "firing scheduled job {} ({})",
        job.short_id(),
        job.schedule.describe()
    );
    Ok(Some((claim, Wakeup {
        job,
        gate_output,
        late_by,
        coalesced,
    })))
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

/// Scheduling's slice of the session database, handed out by
/// [`crate::session::SessionManager::schedule_store`].
#[derive(Clone)]
pub struct ScheduleStore {
    connection: std::sync::Arc<tokio_rusqlite::Connection>,
}

impl ScheduleStore {
    pub(crate) fn new(connection: std::sync::Arc<tokio_rusqlite::Connection>) -> Self {
        Self { connection }
    }

    /// Persist a new scheduled job. The caller owns computing `next_fire_at` from the job's anchor
    /// (see [`ScheduledJob::anchor`]).
    pub async fn create_scheduled_job(&self, job: &ScheduledJob) -> crate::error::Result<()> {
        let id = job.id.clone();
        let session_id = job.session_id.to_string();
        let kind = job.schedule.kind_str().to_string();
        let spec = job.schedule.spec();
        let prompt = job.prompt.clone();
        let gate_command = job.gate.as_ref().map(|gate| gate.command.clone());
        let gate_fire = job.gate.as_ref().map(|gate| gate.fire.as_str().to_string());
        let gate_last_output = job.gate.as_ref().and_then(|gate| gate.last_output.clone());
        let gate_permission = job.gate.as_ref().map(|gate| gate.permission.to_string());
        let isolated = i64::from(job.isolated);
        let created_at = job.created_at.to_rfc3339();
        let last_fired_at = job.last_fired_at.map(|at| at.to_rfc3339());
        let next_fire_at = job.next_fire_at.to_rfc3339();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO scheduled_jobs (id, session_id, kind, spec, prompt, gate_command, \
                     gate_fire, gate_last_output, gate_permission, isolated, created_at, \
                     last_fired_at, next_fire_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
                    rusqlite::params![
                        id,
                        session_id,
                        kind,
                        spec,
                        prompt,
                        gate_command,
                        gate_fire,
                        gate_last_output,
                        gate_permission,
                        isolated,
                        created_at,
                        last_fired_at,
                        next_fire_at
                    ],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to create scheduled job: {}", error))
            })
    }

    /// Every job belonging to one session, soonest first.
    pub async fn list_scheduled_jobs(
        &self,
        session_id: uuid::Uuid,
    ) -> crate::error::Result<Vec<ScheduledJob>> {
        self.query_scheduled_jobs(
            "SELECT id, session_id, kind, spec, prompt, gate_command, gate_fire, \
             gate_last_output, gate_permission, isolated, created_at, last_fired_at, next_fire_at \
             FROM scheduled_jobs WHERE session_id = ?1 ORDER BY next_fire_at ASC",
            vec![session_id.to_string()],
        )
        .await
    }

    /// Every job in the database, soonest first. Backs `meka schedule list` and `meka schedule
    /// cancel`, which work from a job id and so cannot ask the caller which session to look in.
    pub async fn list_all_scheduled_jobs(&self) -> crate::error::Result<Vec<ScheduledJob>> {
        self.query_scheduled_jobs(
            "SELECT id, session_id, kind, spec, prompt, gate_command, gate_fire, \
             gate_last_output, gate_permission, isolated, created_at, last_fired_at, next_fire_at \
             FROM scheduled_jobs ORDER BY next_fire_at ASC",
            Vec::new(),
        )
        .await
    }

    /// Every job across all sessions whose `next_fire_at` has passed, soonest first. The
    /// scheduler's per-tick query; served by `idx_scheduled_jobs_next_fire`.
    pub async fn list_due_scheduled_jobs(
        &self,
        now: chrono::DateTime<chrono::Utc>,
    ) -> crate::error::Result<Vec<ScheduledJob>> {
        self.query_scheduled_jobs(
            "SELECT id, session_id, kind, spec, prompt, gate_command, gate_fire, \
             gate_last_output, gate_permission, isolated, created_at, last_fired_at, next_fire_at \
             FROM scheduled_jobs WHERE next_fire_at <= ?1 ORDER BY next_fire_at ASC",
            vec![now.to_rfc3339()],
        )
        .await
    }

    /// Shared row decoder. A row that fails to decode (hand-edited spec, a `kind` from a future
    /// version) is skipped with a warning rather than failing the whole query: one bad row must not
    /// stop every other job in the database from firing.
    async fn query_scheduled_jobs(
        &self,
        sql: &'static str,
        params: Vec<String>,
    ) -> crate::error::Result<Vec<ScheduledJob>> {
        let rows: Vec<ScheduledJobRow> = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(sql)?;
                let rows = statement
                    .query_map(rusqlite::params_from_iter(params.iter()), |row| {
                        Ok(ScheduledJobRow {
                            id: row.get(0)?,
                            session_id: row.get(1)?,
                            kind: row.get(2)?,
                            spec: row.get(3)?,
                            prompt: row.get(4)?,
                            gate_command: row.get(5)?,
                            gate_fire: row.get(6)?,
                            gate_last_output: row.get(7)?,
                            gate_permission: row.get(8)?,
                            isolated: row.get::<_, i64>(9)? != 0,
                            created_at: row.get(10)?,
                            last_fired_at: row.get(11)?,
                            next_fire_at: row.get(12)?,
                        })
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to load scheduled jobs: {}", error))
            })?;

        Ok(rows
            .into_iter()
            .filter_map(|row| {
                let id = row.id.clone();
                row.decode()
                    .inspect_err(|error| {
                        tracing::warn!("skipping unreadable scheduled job {}: {}", id, error);
                    })
                    .ok()
            })
            .collect())
    }

    /// Delete a job by full or unique-prefix id. Returns the id actually removed, or `None` when
    /// nothing matched.
    ///
    /// An ambiguous prefix is an error rather than an arbitrary pick, and a `Config` rather than a
    /// `Database` one. Nothing went wrong with the database; the caller's prefix is
    /// under-specified, and `Config` is the variant that carries that to HTTP as a 422 rather than
    /// a 500. `BackgroundStore::resolve_background_task` says the same thing about the same
    /// condition and already used `Config`, so the two `serve` endpoints answered different
    /// statuses for one mistake. The variant name fits neither of them well; its mapping does, and
    /// a new variant for one condition is not worth the churn through every match.
    pub async fn cancel_scheduled_job(
        &self,
        session_id: uuid::Uuid,
        id_prefix: &str,
    ) -> crate::error::Result<Option<String>> {
        let jobs = self.list_scheduled_jobs(session_id).await?;
        let matches: Vec<&ScheduledJob> = jobs
            .iter()
            .filter(|job| job.id.starts_with(id_prefix))
            .collect();
        let id = match matches.as_slice() {
            [] => return Ok(None),
            [job] => job.id.clone(),
            several => {
                return Err(MekaError::Config(format!(
                    "'{}' matches {} jobs; use a longer id",
                    id_prefix,
                    several.len()
                )));
            }
        };

        let id_for_db = id.clone();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "DELETE FROM scheduled_jobs WHERE id = ?1",
                    rusqlite::params![id_for_db],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to cancel scheduled job: {}", error))
            })?;
        // The set is keyed by job id and nothing else removes an entry, so a job cancelled through
        // this door -- which is the one `schedule_cancel` uses, i.e. the one the agent reaches --
        // left its id behind for the life of the process. `delete_scheduled_job` clears here too;
        // its doc used to claim every removal went through it, which this function disproves.
        clear_permission_decline(&id);
        Ok(Some(id))
    }

    /// Delete a job by exact id, without the prefix resolution [`Self::cancel_scheduled_job`] does.
    /// Used by the scheduler to retire a fired one-shot.
    pub async fn delete_scheduled_job(&self, id: &str) -> crate::error::Result<()> {
        // A job that no longer exists cannot be held back for permission, and the process-global
        // set is otherwise only cleared when a job is *authorised* again. A job cancelled while
        // declined therefore left its id there for the life of a `meka serve`. Clearing at the one
        // door every removal goes through -- the scheduler retiring a one-shot, and
        // `cancel_scheduled_job`, which delegates here -- keeps the set bounded by the jobs that
        // exist rather than by every job that ever did.
        clear_permission_decline(id);
        let id = id.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "DELETE FROM scheduled_jobs WHERE id = ?1",
                    rusqlite::params![id],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to delete scheduled job: {}", error))
            })
    }

    /// Take an occurrence for this process by advancing the job to its next fire time. Returns
    /// whether the claim was won.
    ///
    /// This is the write that arbitrates between hosts, which is why it is a compare-and-swap on
    /// `next_fire_at` rather than the plain update it used to be. Every `meka serve`, REPL and ACP
    /// session polls the same table, so one occurrence sits in several hosts' due lists at once;
    /// before this, each of them evaluated the job's gate and fired it. Two servers sharing a
    /// database ran one hourly job's gate command three times and its agent turn twice, for a
    /// single occurrence, with nothing said on either side.
    ///
    /// `occurrence` is the value that host read into its due list, so exactly one of them can move
    /// the row off it. The losers change no rows, and [`prepare`] evaluates the gate only after
    /// this has returned `true`, so a lost claim costs a process spawn nobody asked for.
    ///
    /// `last_fired_at` is deliberately not written here. A gated job that evaluates to "no change"
    /// has been *considered*, not fired, and recording it as fired would both mislead
    /// `meka schedule list` and re-anchor an interval schedule on evaluations rather than on fires.
    /// [`Self::stamp_scheduled_job_fired`] adds it once the turn is really going to happen.
    pub(crate) async fn claim_by_advancing(
        &self,
        id: &str,
        occurrence: chrono::DateTime<chrono::Utc>,
        next_fire_at: chrono::DateTime<chrono::Utc>,
    ) -> crate::error::Result<bool> {
        let id = id.to_string();
        let occurrence = occurrence.to_rfc3339();
        let next_fire_at = next_fire_at.to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    &format!(
                        "UPDATE scheduled_jobs SET next_fire_at = ?3 WHERE id = ?1 AND {}",
                        SAME_OCCURRENCE
                    ),
                    rusqlite::params![id, occurrence, next_fire_at],
                )
            })
            .await
            .map(|changed| changed == 1)
            .map_err(|error| {
                MekaError::Database(format!("failed to claim scheduled job: {}", error))
            })
    }

    /// Take an occurrence for this process by deleting the job outright, for a one-shot whose
    /// moment has come and for a cron pattern with nothing left in range. Returns whether the claim
    /// was won.
    ///
    /// The delete half of [`Self::claim_by_advancing`]. What arbitrates here is the affected-row
    /// count: a `DELETE ... WHERE id = ?` that discards it reports success to every host that
    /// issues it, so all of them go on to deliver a reminder exactly one of them removed. The
    /// occurrence is in the `WHERE` to keep the delete scoped to the row this host read, which
    /// matters if the row is ever re-pointed underneath a sweep rather than merely removed.
    pub(crate) async fn claim_by_retiring(
        &self,
        id: &str,
        occurrence: chrono::DateTime<chrono::Utc>,
    ) -> crate::error::Result<bool> {
        let id = id.to_string();
        let occurrence = occurrence.to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    &format!(
                        "DELETE FROM scheduled_jobs WHERE id = ?1 AND {}",
                        SAME_OCCURRENCE
                    ),
                    rusqlite::params![id, occurrence],
                )
            })
            .await
            .map(|changed| changed == 1)
            .map_err(|error| {
                MekaError::Database(format!("failed to retire scheduled job: {}", error))
            })
    }

    /// Record that a job fired.
    ///
    /// Written *before* the turn runs, not after: a prompt that reliably crashes or hangs the
    /// process would otherwise re-fire on every restart, turning one bad job into a boot loop in
    /// the daemon that is supposed to stay up. Stamping first costs one missed occurrence instead.
    ///
    /// `claimed` is the fire time [`Self::claim_by_advancing`] wrote, and is the guard rather than
    /// a value to write: the column already holds it. For a short interval with a slow gate the
    /// claimed time can itself be in the past by the time the gate returns, so another host may
    /// have legitimately claimed the *following* occurrence in between -- and an unguarded stamp
    /// would drag `next_fire_at` back onto an occurrence that host is already running.
    pub async fn stamp_scheduled_job_fired(
        &self,
        id: &str,
        fired_at: chrono::DateTime<chrono::Utc>,
        claimed: chrono::DateTime<chrono::Utc>,
    ) -> crate::error::Result<()> {
        let id = id.to_string();
        let fired_at = fired_at.to_rfc3339();
        let claimed = claimed.to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    &format!(
                        "UPDATE scheduled_jobs SET last_fired_at = ?3 WHERE id = ?1 AND {}",
                        SAME_OCCURRENCE
                    ),
                    rusqlite::params![id, claimed, fired_at],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to stamp scheduled job: {}", error))
            })
    }

    /// Put a job back exactly as it was, for a host that turned out to be unable to run it after
    /// [`prepare`] had already claimed the occurrence.
    ///
    /// Scoped to `claim`, which is the whole substance. This used to be an `INSERT OR REPLACE` of
    /// the row as it stood *before* the claim, applied by id alone, and that made a lost race
    /// permanent: two hosts both claimed an occurrence, the loser was refused the session lock, and
    /// its restore overwrote the winner's `next_fire_at`, `last_fired_at` and `gate_last_output` --
    /// putting the job back in the past so it came due again on the very next tick, while the
    /// winner was still running the turn. The blind upsert also resurrected a job the user had
    /// cancelled during the turn; an id-and-claim scoped update leaves that deletion alone.
    ///
    /// A retired job is re-inserted rather than updated, because claiming one *deletes* the row: a
    /// one-shot has no next occurrence, so an `UPDATE` would match nothing while still reporting
    /// success, losing the reminder outright. `OR IGNORE` because a row back under that id is one
    /// this process no longer has any claim on.
    pub(crate) async fn restore_scheduled_job(
        &self,
        job: &ScheduledJob,
        claim: Claim,
    ) -> crate::error::Result<()> {
        let id = job.id.clone();
        let short_id = job.short_id().to_string();
        let gate_last_output = job.gate.as_ref().and_then(|gate| gate.last_output.clone());
        let last_fired_at = job.last_fired_at.map(|at| at.to_rfc3339());
        let next_fire_at = job.next_fire_at.to_rfc3339();

        let restored = match claim {
            Claim::Advanced(claimed) => {
                let claimed = claimed.to_rfc3339();
                self.connection
                    .call(move |connection| -> rusqlite::Result<_> {
                        connection.execute(
                            &format!(
                                "UPDATE scheduled_jobs SET next_fire_at = ?3, last_fired_at = ?4, \
                                 gate_last_output = ?5 WHERE id = ?1 AND {}",
                                SAME_OCCURRENCE
                            ),
                            rusqlite::params![
                                id,
                                claimed,
                                next_fire_at,
                                last_fired_at,
                                gate_last_output
                            ],
                        )
                    })
                    .await
            }
            Claim::Retired => {
                let session_id = job.session_id.to_string();
                let kind = job.schedule.kind_str().to_string();
                let spec = job.schedule.spec();
                let prompt = job.prompt.clone();
                let gate_command = job.gate.as_ref().map(|gate| gate.command.clone());
                let gate_fire = job.gate.as_ref().map(|gate| gate.fire.as_str().to_string());
                let gate_permission = job.gate.as_ref().map(|gate| gate.permission.to_string());
                let isolated = i64::from(job.isolated);
                let created_at = job.created_at.to_rfc3339();
                self.connection
                    .call(move |connection| -> rusqlite::Result<_> {
                        connection.execute(
                            "INSERT OR IGNORE INTO scheduled_jobs (id, session_id, kind, spec, \
                             prompt, gate_command, gate_fire, gate_last_output, gate_permission, \
                             isolated, created_at, last_fired_at, next_fire_at) \
                             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
                            rusqlite::params![
                                id,
                                session_id,
                                kind,
                                spec,
                                prompt,
                                gate_command,
                                gate_fire,
                                gate_last_output,
                                gate_permission,
                                isolated,
                                created_at,
                                last_fired_at,
                                next_fire_at
                            ],
                        )
                    })
                    .await
            }
        }
        .map_err(|error| {
            MekaError::Database(format!("failed to restore scheduled job: {}", error))
        })?;

        // Not an error. The row can have moved on for reasons that are all fine -- the job was
        // cancelled while the turn was being declined, or a later occurrence has since been claimed
        // -- and in every one of them the right answer is to leave what is there alone.
        if restored == 0 {
            tracing::debug!(
                "job {} was not restored: its row has moved on since the claim",
                short_id
            );
        }
        Ok(())
    }

    /// Write `next_fire_at` verbatim, bypassing the `to_rfc3339` rendering every real writer goes
    /// through.
    ///
    /// Exists so a test can plant the shape [`SAME_OCCURRENCE`]'s second arm is for. Nothing in
    /// meka can produce a timestamp in any other form, so without this the fallback would be
    /// unreachable from the test suite and its guarantee would rest on the comment alone.
    #[cfg(test)]
    pub(crate) async fn set_next_fire_at_verbatim_for_test(
        &self,
        id: &str,
        raw: &str,
    ) -> crate::error::Result<()> {
        let id = id.to_string();
        let raw = raw.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE scheduled_jobs SET next_fire_at = ?2 WHERE id = ?1",
                    rusqlite::params![id, raw],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to plant a timestamp: {}", error)))
    }

    /// Persist the gate's latest stdout so the next `on-change` evaluation has something to compare
    /// against.
    ///
    /// Guarded on the claim for the same reason [`Self::stamp_scheduled_job_fired`] is: a late
    /// write here would overwrite the baseline a host running the *following* occurrence has
    /// already recorded, and an `on-change` gate whose baseline goes backwards reports a change
    /// that has already been reported.
    pub(crate) async fn update_scheduled_job_gate_output(
        &self,
        id: &str,
        claimed: chrono::DateTime<chrono::Utc>,
        output: &str,
    ) -> crate::error::Result<()> {
        let id = id.to_string();
        let claimed = claimed.to_rfc3339();
        let output = output.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    &format!(
                        "UPDATE scheduled_jobs SET gate_last_output = ?3 WHERE id = ?1 AND {}",
                        SAME_OCCURRENCE
                    ),
                    rusqlite::params![id, claimed, output],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to record gate output: {}", error))
            })
    }
}

/// The `next_fire_at` half of every claim-scoped `WHERE`, comparing `?2` against the stored column.
///
/// Textual equality is the fast path and is what matches in practice: every writer renders the
/// column with `DateTime::<Utc>::to_rfc3339`, and re-rendering a value parsed back out of it
/// reproduces the same bytes. The `julianday` arm is there so a row that reached the database any
/// other way -- a hand-edited timestamp, a `Z` suffix, a non-UTC offset -- is still claimable
/// rather than silently unclaimable forever, which is the shape this failure would take: the
/// compare-and-swap would match nothing, on every sweep, and the job would simply never fire again
/// with nothing logged. `julianday` returns `NULL` for anything it cannot parse and `NULL = NULL`
/// is not true, so an unreadable timestamp fails closed. It compares instants at millisecond
/// resolution, which cannot conflate two occurrences: [`MIN_EVERY`] is a second.
const SAME_OCCURRENCE: &str = "(next_fire_at = ?2 OR julianday(next_fire_at) = julianday(?2))";

/// Raw `scheduled_jobs` row, decoded into a [`ScheduledJob`] outside the database closure so parse
/// failures can be logged and skipped individually.
struct ScheduledJobRow {
    id: String,
    session_id: String,
    kind: String,
    spec: String,
    prompt: String,
    gate_command: Option<String>,
    gate_fire: Option<String>,
    gate_last_output: Option<String>,
    gate_permission: Option<String>,
    isolated: bool,
    created_at: String,
    last_fired_at: Option<String>,
    next_fire_at: String,
}

impl ScheduledJobRow {
    fn decode(self) -> std::result::Result<ScheduledJob, String> {
        let parse_time =
            |text: &str| -> std::result::Result<chrono::DateTime<chrono::Utc>, String> {
                chrono::DateTime::parse_from_rfc3339(text)
                    .map(|at| at.with_timezone(&chrono::Utc))
                    .map_err(|error| format!("bad timestamp '{}': {}", text, error))
            };

        let gate = match (self.gate_command, self.gate_fire) {
            (Some(command), Some(fire)) => Some(Gate {
                command,
                fire: GateFire::parse(&fire)?,
                last_output: self.gate_last_output,
                // Every write path stores a level alongside the gate, so an absent or unparseable
                // one means a hand-edited or damaged row. Reading that as `Unrestricted` would hand
                // an arbitrary shell command the authority the column exists to
                // record, so it resolves to `None`: the gate is refused at fire
                // time and the user is told to recreate the job. Failing closed
                // costs one re-creation; failing open costs the guarantee.
                //
                // Through `parse_recorded_permission` like the five session-row readers, so the
                // *unreadable* case is heard rather than folded into the absent one. Without the
                // warning the only clue is a later message saying the gate was authorised at
                // `none` -- naming a level the job was never created at.
                permission: crate::permission::parse_recorded_permission(
                    self.gate_permission.as_deref(),
                    &format_args!("the gate on job {}", self.id),
                )
                .unwrap_or(crate::permission::Permission::None),
            }),
            // A half-written gate is a corrupt row, not a job without a gate: silently dropping the
            // condition would turn a watcher into an unconditional timer.
            (Some(_), None) | (None, Some(_)) => {
                return Err("gate_command and gate_fire must both be set or both be null".into());
            }
            (None, None) => None,
        };

        Ok(ScheduledJob {
            id: self.id,
            session_id: uuid::Uuid::parse_str(&self.session_id)
                .map_err(|error| format!("bad session id '{}': {}", self.session_id, error))?,
            schedule: Schedule::from_stored(&self.kind, &self.spec)?,
            prompt: self.prompt,
            gate,
            isolated: self.isolated,
            created_at: parse_time(&self.created_at)?,
            last_fired_at: self.last_fired_at.as_deref().map(parse_time).transpose()?,
            next_fire_at: parse_time(&self.next_fire_at)?,
        })
    }
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

    /// The fix has to reach the rows already on disk, which is the only population it matters for.
    /// Creation was closed first and rehydration was left on the permissive parser, so a six-field
    /// row kept its every-ten-seconds reading forever.
    #[test]
    fn a_stored_cron_spec_is_read_with_the_same_grammar_it_was_created_under() {
        assert!(
            Schedule::parse_cron("*/10 * * * * *").is_err(),
            "six fields are refused at creation"
        );
        assert!(
            Schedule::from_stored("cron", "*/10 * * * * *").is_err(),
            "and refused on the way back out of the database"
        );

        // A legitimate five-field row still round-trips.
        let stored = Schedule::from_stored("cron", "0 9 * * 1-5").expect("five fields still load");
        assert_eq!(stored.spec(), "0 9 * * 1-5");
    }

    #[test]
    fn test_parse_cron_rejects_unsatisfiable_pattern() {
        // Well-formed but matches no calendar date; caught at creation rather than leaving a job
        // that silently never fires.
        assert!(Schedule::parse_cron("0 0 30 2 *").is_err());
    }

    /// A pattern whose next occurrence is years away is satisfiable, and `prepare` retires a job
    /// whose schedule has no next occurrence, so `next_after` must not confuse "far off" with
    /// "never". February 29th is the shortest such case at up to four years.
    #[test]
    fn a_schedule_whose_next_occurrence_is_years_away_still_has_one() {
        use chrono::TimeZone;

        let schedule = Schedule::parse_cron("0 0 29 2 *").expect("Feb 29 is a real date");
        let anchor = Utc
            .with_ymd_and_hms(2026, 8, 16, 12, 0, 0)
            .single()
            .expect("anchor");
        let next = schedule
            .next_after(anchor)
            .expect("a leap day must not read as an unschedulable pattern");
        assert!(next > anchor + chrono::Duration::days(366), "{next}");
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

    /// A gate command that creates `path`, in whichever shell the host will run it under.
    ///
    /// `evaluate_gate` spawns the platform shell, so a `touch` hardcoded in a fixture is a Unix
    /// command handed to PowerShell on Windows: the gate reports a non-zero exit, the probe never
    /// appears, and the test reads that as the scheduler having declined to fire.
    fn create_file_command(path: &std::path::Path) -> String {
        if cfg!(windows) {
            format!(
                "New-Item -ItemType File -Force -Path '{}' | Out-Null",
                path.display()
            )
        } else {
            format!("touch '{}'", path.display())
        }
    }

    fn gate(command: &str, fire: GateFire, last_output: Option<&str>) -> Gate {
        Gate {
            command: command.to_string(),
            fire,
            last_output: last_output.map(str::to_string),
            // The level every gate is created at. Tests that exercise a gate *running* need it; the
            // one that exercises a withdrawn authority overrides it explicitly.
            permission: crate::permission::Permission::Unrestricted,
        }
    }

    const GATE_BUDGET: Duration = Duration::from_secs(10);

    #[tokio::test]
    async fn test_on_success_gate_follows_the_exit_code() {
        let passing = evaluate_gate(
            &gate("exit 0", GateFire::OnSuccess, None),
            GATE_BUDGET,
            None,
        )
        .await
        .expect("gate ran");
        assert!(passing.fired);

        let failing = evaluate_gate(
            &gate("exit 1", GateFire::OnSuccess, None),
            GATE_BUDGET,
            None,
        )
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
        let outcome = evaluate_gate(
            &gate("echo ready", GateFire::OnChange, None),
            GATE_BUDGET,
            None,
        )
        .await
        .expect("gate ran");
        assert!(outcome.fired);
        assert_eq!(outcome.output, "ready");
    }

    /// A gate runs in its session's directory, not the host process's.
    ///
    /// The model almost always authors a gate right after verifying the same command through
    /// `execute_command`, which runs in the session cwd. Under a `meka serve` unit the host process
    /// sits somewhere else entirely (`/`, or wherever systemd put it), so a gate that ignores the
    /// session cwd silently stops matching the command the user watched succeed. Nothing caught
    /// this: the `cwd` argument threads all the way through `prepare` with no assertion on it.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_gate_runs_in_its_sessions_directory() {
        let temp = tempfile::tempdir().expect("tempdir");
        // Resolved because macOS hands out `/var/...` symlinked to `/private/var/...`, and `pwd`
        // in the child reports the resolved form.
        let directory = temp.path().canonicalize().expect("canonicalize");

        let outcome = evaluate_gate(
            &gate("pwd", GateFire::OnChange, None),
            GATE_BUDGET,
            Some(&directory),
        )
        .await
        .expect("gate ran");

        assert_eq!(
            outcome.output,
            directory.to_string_lossy(),
            "the gate ran in the host's directory instead of the session's"
        );
    }

    /// A non-zero exit is how several perfectly good on-change gates signal a change: `diff -q`
    /// and `git diff --exit-code` exit 1 exactly when there is a difference. Refusing to fire on a
    /// non-zero exit silenced those permanently.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_on_change_gate_that_signals_through_its_exit_code_still_fires() {
        let gate = Gate {
            command: "echo 'Files a and b differ'; exit 1".to_string(),
            fire: GateFire::OnChange,
            last_output: Some("".to_string()),
            permission: crate::permission::Permission::Unrestricted,
        };
        let outcome = evaluate_gate(&gate, GATE_BUDGET, None)
            .await
            .expect("a non-zero exit is a signal, not a broken gate");
        assert!(
            outcome.fired,
            "output differs from the baseline, so the gate must fire"
        );
        assert_eq!(outcome.output, "Files a and b differ");
    }

    /// The other half: a watcher in its quiet period exits non-zero with nothing on stdout, every
    /// time, and must stay quiet rather than erroring. `grep PATTERN log` is the canonical shape.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_on_change_gate_quiet_period_is_not_an_error() {
        let gate = Gate {
            command: "exit 1".to_string(),
            fire: GateFire::OnChange,
            last_output: Some("".to_string()),
            permission: crate::permission::Permission::Unrestricted,
        };
        let outcome = evaluate_gate(&gate, GATE_BUDGET, None)
            .await
            .expect("a quiet watcher is not a broken one");
        assert!(!outcome.fired, "nothing changed, so nothing fires");
    }

    #[tokio::test]
    async fn test_on_change_gate_is_quiet_until_the_output_differs() {
        let unchanged = evaluate_gate(
            &gate("echo steady", GateFire::OnChange, Some("steady")),
            GATE_BUDGET,
            None,
        )
        .await
        .expect("gate ran");
        assert!(!unchanged.fired, "same output must not spend a turn");

        let changed = evaluate_gate(
            &gate("echo moved", GateFire::OnChange, Some("steady")),
            GATE_BUDGET,
            None,
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
            None,
        )
        .await
        .expect_err("an overrunning gate must not report success");
        assert!(error.contains("budget"), "{error}");
    }

    #[tokio::test]
    async fn test_gate_output_is_trimmed_so_trailing_newlines_do_not_flap() {
        // `echo` appends a newline. Comparing untrimmed, a gate whose command varied its trailing
        // whitespace would fire forever.
        let outcome = evaluate_gate(
            &gate("echo spaced", GateFire::OnChange, None),
            GATE_BUDGET,
            None,
        )
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
            Self::at_host_permission(crate::permission::Permission::Unrestricted).await
        }

        /// A harness whose *host* runs at `host_permission`, which is what a
        /// `meka serve --permission read` inheriting someone else's job looks like. `new` uses
        /// `Unrestricted` because that is the ordinary case every other test needs; `Default`
        /// deliberately gives `None`, so the level has to be stated here rather than
        /// inherited by accident.
        async fn at_host_permission(host_permission: crate::permission::Permission) -> Self {
            let manager = std::sync::Arc::new(
                crate::session::SessionManager::open(Some(std::path::Path::new(":memory:")))
                    .await
                    .expect("open in-memory database"),
            );
            let session_id = manager.create_session(None).await.expect("create session");
            Self {
                manager,
                session_id,
                config: crate::config::ResolvedScheduleConfig {
                    host_permission,
                    ..Default::default()
                },
                fired: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            }
        }

        /// A second session in the same database, for the per-session budget tests.
        async fn another_session(&self) -> uuid::Uuid {
            self.manager
                .create_session(None)
                .await
                .expect("create session")
        }

        /// Insert a job already overdue by `overdue`.
        async fn overdue_job(
            &self,
            schedule: Schedule,
            gate: Option<Gate>,
            overdue: chrono::Duration,
        ) -> ScheduledJob {
            self.overdue_job_in(self.session_id, schedule, gate, overdue)
                .await
        }

        /// Same, against an explicit session.
        async fn overdue_job_in(
            &self,
            session_id: uuid::Uuid,
            schedule: Schedule,
            gate: Option<Gate>,
            overdue: chrono::Duration,
        ) -> ScheduledJob {
            let now = Utc::now();
            let job = ScheduledJob {
                id: uuid::Uuid::new_v4().to_string(),
                session_id,
                schedule,
                prompt: "do the thing".to_string(),
                gate,
                isolated: false,
                created_at: now - overdue - chrono::Duration::seconds(1),
                last_fired_at: None,
                next_fire_at: now - overdue,
            };
            self.manager
                .schedule_store()
                .create_scheduled_job(&job)
                .await
                .expect("create job");
            job
        }

        async fn tick(&self) {
            self.tick_with(self.config.clone()).await;
        }

        /// One sweep under a config other than the harness's own, for the tests that model an
        /// operator changing `config.toml` and restarting while the rows stay as they were.
        async fn tick_with(&self, config: crate::config::ResolvedScheduleConfig) {
            let fired = self.fired.clone();
            run_due(
                &self.manager,
                &config,
                &SchedulerScope::every_job(),
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
                .schedule_store()
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

    /// Coalescing bounds one job's backlog; this bounds the whole sweep's. A session that
    /// accumulated more watchers than the budget must not wake to a turn per job, and must not lose
    /// any of them either.
    #[tokio::test]
    async fn test_the_fire_budget_holds_jobs_over_without_losing_them() {
        let mut harness = SchedulerHarness::new().await;
        harness.config.max_consecutive_fires = 5;
        for _ in 0..8 {
            harness
                .overdue_job(
                    Schedule::parse_every("1h").expect("parses"),
                    None,
                    chrono::Duration::hours(6),
                )
                .await;
        }

        harness.tick().await;
        assert_eq!(harness.fired().len(), 5, "the budget bounds the burst");

        harness.tick().await;
        let fired = harness.fired();
        assert_eq!(fired.len(), 8, "and the next sweep takes the rest");
        let distinct: std::collections::HashSet<&str> =
            fired.iter().map(|record| record.job_id.as_str()).collect();
        assert_eq!(distinct.len(), 8, "every job fired exactly once");
    }

    /// Per session, not per sweep. A budget shared across sessions would let one conversation's
    /// backlog delay another's due job, which under `meka serve` is somebody else's job entirely.
    #[tokio::test]
    async fn test_the_fire_budget_is_per_session_rather_than_global() {
        let mut harness = SchedulerHarness::new().await;
        harness.config.max_consecutive_fires = 5;
        let other = harness.another_session().await;
        // The backlog is older, so it sorts first and would exhaust a global budget before the
        // quiet session's single job was ever reached.
        for _ in 0..6 {
            harness
                .overdue_job(
                    Schedule::parse_every("1h").expect("parses"),
                    None,
                    chrono::Duration::hours(6),
                )
                .await;
        }
        let lonely = harness
            .overdue_job_in(
                other,
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        let fired = harness.fired();
        assert_eq!(
            fired.len(),
            6,
            "five from the busy session, one from the other"
        );
        assert!(
            fired.iter().any(|record| record.job_id == lonely.id),
            "the quiet session's job was not held behind the busy one's backlog"
        );
    }

    /// A job the gate retires spends no turn, so it must not spend budget either. Otherwise a
    /// handful of quiet watchers would starve the one job that had something to report.
    #[tokio::test]
    async fn test_a_declining_gate_does_not_consume_the_fire_budget() {
        let mut harness = SchedulerHarness::new().await;
        harness.config.max_consecutive_fires = 2;
        // More overdue, so both are evaluated before the ungated job below.
        for _ in 0..2 {
            harness
                .overdue_job(
                    Schedule::parse_every("1h").expect("parses"),
                    Some(gate("exit 1", GateFire::OnSuccess, None)),
                    chrono::Duration::hours(6),
                )
                .await;
        }
        let speaks = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        let fired = harness.fired();
        assert_eq!(fired.len(), 1, "the two silent gates cost nothing");
        assert_eq!(fired[0].job_id, speaks.id);
    }

    /// A host handing an occurrence back has not spent a turn on it, so the deferral must not spend
    /// budget either. Otherwise a `meka serve` whose session is held by a REPL would burn its whole
    /// per-sweep budget on jobs it never ran.
    #[tokio::test]
    async fn test_a_deferral_does_not_consume_the_fire_budget() {
        let mut harness = SchedulerHarness::new().await;
        harness.config.max_consecutive_fires = 1;
        for _ in 0..2 {
            harness
                .overdue_job(
                    Schedule::parse_every("1h").expect("parses"),
                    None,
                    chrono::Duration::hours(6),
                )
                .await;
        }

        let offered = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        run_due(
            &harness.manager,
            &harness.config,
            &SchedulerScope::every_job(),
            &move |_wakeup: Wakeup| {
                let offered = offered.clone();
                async move {
                    // The first is handed back; only the second spends a turn.
                    match offered.fetch_add(1, std::sync::atomic::Ordering::Relaxed) {
                        0 => FireOutcome::Deferred,
                        _ => FireOutcome::Ran,
                    }
                }
            },
        )
        .await
        .expect("sweep runs");

        // Both jobs reached the host despite a budget of one, because the deferred one cost
        // nothing. The second is the only one whose schedule advanced.
        let still_due: Vec<_> = harness
            .jobs()
            .await
            .into_iter()
            .filter(|job| job.next_fire_at <= Utc::now())
            .collect();
        assert_eq!(
            still_due.len(),
            1,
            "the deferred job kept its occurrence; the other one spent its turn"
        );
    }

    /// The budget is checked before `prepare`, which is where a gate runs, so holding a job over
    /// must cost nothing -- not even the shell command whose expense is half the reason gates
    /// exist.
    ///
    /// Observed through a side effect on the filesystem rather than through the job's stored gate
    /// baseline. Enforcing the budget *after* `prepare` and then restoring the job would run the
    /// command and put the baseline back, leaving every column identical to a job that was never
    /// touched. Only the command's own footprint tells the two apart.
    #[tokio::test]
    async fn test_a_held_over_job_does_not_run_its_gate() {
        let mut harness = SchedulerHarness::new().await;
        harness.config.max_consecutive_fires = 1;
        // Removed on drop rather than at the end of the test, so a failing assertion does not leak
        // it into the temp directory of whoever ran the suite.
        struct Probe(std::path::PathBuf);
        impl Drop for Probe {
            fn drop(&mut self) {
                std::fs::remove_file(&self.0).ok();
            }
        }
        let guard =
            Probe(std::env::temp_dir().join(format!("meka-gate-probe-{}", uuid::Uuid::new_v4())));
        let probe = guard.0.clone();
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::hours(6),
            )
            .await;
        let watcher = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate(&create_file_command(&probe), GateFire::OnChange, None)),
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        assert!(
            !probe.exists(),
            "the held-over job's gate command never ran"
        );
        let held = harness
            .jobs()
            .await
            .into_iter()
            .find(|job| job.id == watcher.id)
            .expect("the held-over job is still there");
        assert_eq!(
            held.next_fire_at, watcher.next_fire_at,
            "and its schedule was not advanced, so it is still due"
        );

        // The probe is only evidence if it can fire at all: the next sweep has budget for it.
        harness.tick().await;
        assert!(probe.exists(), "and it runs once the budget has room");
        std::fs::remove_file(&probe).expect("clean up the probe");
    }

    /// The rule every host defers to. Withdrawing a failed fire's prompt is only safe when the job
    /// will produce it again; `prepare` deletes a one-shot's row *before* the turn runs, so for
    /// those the unanswered message is the last trace the reminder ever fired.
    #[test]
    fn test_only_a_recurring_job_lets_a_failed_fire_withdraw_its_prompt() {
        let recurring = |schedule: Schedule| ScheduledJob {
            id: "7f3a1b2c".to_string(),
            session_id: uuid::Uuid::nil(),
            schedule,
            prompt: "check the news".to_string(),
            gate: None,
            isolated: false,
            created_at: at("2026-08-11T12:00:00Z"),
            last_fired_at: None,
            next_fire_at: at("2026-08-11T12:00:00Z"),
        };

        for schedule in [
            Schedule::parse_every("1h").expect("parses"),
            Schedule::parse_cron("0 9 * * 1-5").expect("parses"),
        ] {
            assert_eq!(
                recurring(schedule).prompt_retention(),
                crate::agent::PromptRetention::WithdrawOnFailure,
                "a recurring job regenerates its prompt"
            );
        }
        assert_eq!(
            recurring(Schedule::At(at("2026-08-11T12:00:00Z"))).prompt_retention(),
            crate::agent::PromptRetention::Keep,
            "a one-shot's row is already gone; its prompt is all that is left"
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

    /// A gate is authorised once, at `unrestricted`, and then persists as a row that any process
    /// executes on a timer. Nothing about the creating session's later downgrade -- Shift+Tab
    /// to `read`, or a `meka serve --permission read` restart inheriting the job -- can reach
    /// back to withdraw it, so the level travels on the row and is re-checked here. Asserted
    /// through a real filesystem side effect rather than through the returned outcome, because
    /// "did not fire" and "did not *run*" are different claims and only the second one is the
    /// security property.
    /// The scenario the feature exists for, driven the way production produces it: the gate carries
    /// the `Unrestricted` it was legitimately created with, and the *host* has since dropped to
    /// `read`.
    ///
    /// The sibling below hand-sets `gate.permission` to `Read`, which no creation path can produce
    /// -- both `schedule_create` and the HTTP handler demand `Unrestricted` before writing the row,
    /// and nothing updates the column afterwards. So that test proved the mechanism worked on
    /// an input reality never supplies, and the check it guarded compared `unrestricted` with
    /// itself for every real job.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_gate_is_not_executed_once_the_host_drops_below_unrestricted() {
        let marker = std::env::temp_dir().join(format!("meka-gate-{}", uuid::Uuid::new_v4()));
        let _ = std::fs::remove_file(&marker);

        let harness =
            SchedulerHarness::at_host_permission(crate::permission::Permission::Read).await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                // Recorded `Unrestricted`, exactly as `schedule_create` would have written it.
                Some(gate(
                    &create_file_command(&marker),
                    GateFire::OnSuccess,
                    None,
                )),
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        assert!(
            !marker.exists(),
            "the gate command must not run at all once the host is below write"
        );
        let _ = std::fs::remove_file(&marker);
    }

    /// A recurring job stays on the grid it was created on, however late a pickup is.
    ///
    /// The old call passed `now` to `next_after`, so each fire added that sweep's pickup latency to
    /// the interval and kept it. At `poll_interval = 1s` a measured `every = "1s"` ran at 2.0s --
    /// half the requested rate -- because the tick one interval later lands microseconds early and
    /// the job waits a whole extra poll. This asserts the property directly rather than by timing a
    /// live scheduler, which would be flaky.
    #[test]
    fn a_recurring_schedule_advances_from_the_occurrence_it_delivered_not_from_now() {
        let every = Schedule::parse_every("5s").expect("parses");
        let delivered = chrono::Utc::now() - chrono::Duration::seconds(12);
        // Pickup is late, and not on a multiple of the interval.
        let now = delivered + chrono::Duration::milliseconds(12_345);

        let next = every
            .next_after_delivering(delivered, now)
            .expect("a recurring schedule always has a next occurrence");
        assert!(next > now, "the next occurrence must be in the future");
        let offset = (next - delivered).num_milliseconds();
        assert_eq!(
            offset % 5_000,
            0,
            "the next fire must sit on a whole multiple of the interval from the delivered \
             occurrence, got {offset}ms"
        );
        assert_eq!(
            offset, 15_000,
            "and it must be the *first* such multiple after now"
        );

        // The shape the old code produced, kept here so the difference is legible: anchoring on
        // `now` yields a time that is not on the grid at all.
        let drifted = every.next_after(now).expect("every always has a next");
        assert_ne!(
            (drifted - delivered).num_milliseconds() % 5_000,
            0,
            "anchoring on `now` should drift off the grid, or this test proves nothing"
        );

        // A long outage skips whole intervals rather than replaying them, and still lands on the
        // grid.
        let after_outage = delivered + chrono::Duration::seconds(3_601);
        let resumed = every
            .next_after_delivering(delivered, after_outage)
            .expect("still recurring");
        assert!(resumed > after_outage);
        assert_eq!((resumed - delivered).num_milliseconds() % 5_000, 0);

        // Cron is absolute wall-clock and must be untouched by any of this.
        let cron = Schedule::parse_cron("*/5 * * * *").expect("parses");
        assert_eq!(
            cron.next_after_delivering(delivered, now),
            cron.next_after(now),
            "a cron pattern's next occurrence does not depend on which one was just delivered"
        );
    }

    /// And the wiring: the row `prepare` writes back is on the grid, not one interval from now.
    ///
    /// The unit test above covers `next_after_delivering`; reverting the *call site* to
    /// `next_after(now)` left every schedule test passing, which is the same shape as the four
    /// dead wirings this round started with. What the user feels is this value, persisted.
    #[tokio::test]
    async fn a_fired_job_is_rescheduled_onto_its_own_grid() {
        let harness =
            SchedulerHarness::at_host_permission(crate::permission::Permission::Unrestricted).await;
        let job = harness
            .overdue_job(
                Schedule::parse_every("5s").expect("parses"),
                None,
                chrono::Duration::seconds(12),
            )
            .await;
        let delivered = job.next_fire_at;

        harness.tick().await;

        let rescheduled = harness
            .jobs()
            .await
            .into_iter()
            .find(|candidate| candidate.id == job.id)
            .expect("a recurring job lives on");
        let offset = (rescheduled.next_fire_at - delivered).num_milliseconds();
        assert_eq!(
            offset % 5_000,
            0,
            "the persisted next fire must be a whole number of intervals from the occurrence just \
             delivered, got {offset}ms -- anchoring on `now` instead makes every job drift"
        );
        assert!(
            rescheduled.next_fire_at > delivered,
            "and must be in the future relative to what was delivered"
        );
    }

    /// Narrowing `[permissions].enabled` disarms a gate whose session row still records a level
    /// the installation no longer permits.
    ///
    /// A row outlives the configuration that produced it. Every session `meka serve` creates
    /// persists its own level, so `--permission` on the host is a *default* and not a ceiling, and
    /// the only thing an operator can narrow that a row cannot exceed is the enabled set. Without
    /// the filter this test guards, that operator restarts, watches the session re-attach at
    /// `read` in the log, and the gate keeps firing at `unrestricted` -- while the creation door
    /// two files over returns 403 for the very same authority.
    #[cfg(unix)]
    #[tokio::test]
    async fn narrowing_the_enabled_set_disarms_a_gate_the_row_still_authorises() {
        let marker = std::env::temp_dir().join(format!("meka-gate-{}", uuid::Uuid::new_v4()));
        let _ = std::fs::remove_file(&marker);

        // The host is `read`, which is what a restart under the narrowed config produces:
        // `resolve_permission` clamps the process default into the enabled set. What makes this
        // test different from its sibling above is the *row*: that one has no per-session level at
        // all, so the host answer is reached trivially, while here the row still says
        // `unrestricted` and something has to refuse to believe it.
        let harness =
            SchedulerHarness::at_host_permission(crate::permission::Permission::Read).await;
        harness
            .manager
            .update_session_permission(harness.session_id, "unrestricted")
            .await
            .expect("record the level the session was set to");
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate(
                    &create_file_command(&marker),
                    GateFire::OnSuccess,
                    None,
                )),
                chrono::Duration::minutes(5),
            )
            .await;

        // What the operator did: narrowed the installation to `read` and restarted.
        let narrowed = crate::config::ResolvedScheduleConfig {
            enabled_permissions: crate::permission::EnabledPermissions::from_modes([
                crate::permission::Permission::Read,
            ])
            .expect("a single mode is a valid set"),
            ..harness.config.clone()
        };
        harness.tick_with(narrowed).await;
        assert!(
            !marker.exists(),
            "a level the installation no longer enables must not authorise a gate"
        );

        // And the control: with `unrestricted` still enabled, the identical row does authorise the
        // gate, so the refusal above is the enabled set rather than some unrelated part of the
        // fixture. A *second* job, because the sweep above claimed the first one's occurrence by
        // advancing it -- a declined gate still spends the occurrence, which is what stops a
        // refused watcher from re-running every poll.
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate(
                    &create_file_command(&marker),
                    GateFire::OnSuccess,
                    None,
                )),
                chrono::Duration::minutes(5),
            )
            .await;
        let permitted = crate::config::ResolvedScheduleConfig {
            enabled_permissions: crate::permission::EnabledPermissions::DEFAULT,
            ..harness.config.clone()
        };
        harness.tick_with(permitted).await;
        assert!(
            marker.exists(),
            "the same job must still fire while its recorded level is enabled"
        );
        let _ = std::fs::remove_file(&marker);
    }

    /// A panic in one fire must not stop the scheduler.
    ///
    /// Under `meka serve` the callback runs a whole agent turn, so everything the tool loop can do
    /// is inside the surface that can panic. Nothing joins this task, so losing it produced no
    /// error anywhere: scheduled jobs simply stopped firing, for the life of the process, and the
    /// first sign was a reminder that never arrived.
    #[tokio::test]
    async fn a_panicking_fire_does_not_stop_the_scheduler() {
        let harness =
            SchedulerHarness::at_host_permission(crate::permission::Permission::Unrestricted).await;
        harness
            .overdue_job(
                Schedule::parse_every("1s").expect("parses"),
                None,
                chrono::Duration::seconds(30),
            )
            .await;

        let fires = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let config = crate::config::ResolvedScheduleConfig {
            poll_interval: std::time::Duration::from_millis(20),
            ..harness.config.clone()
        };
        let handle = spawn(
            std::sync::Arc::clone(&harness.manager),
            config,
            SchedulerScope::every_job(),
            {
                let fires = std::sync::Arc::clone(&fires);
                move |_wakeup| {
                    let fires = std::sync::Arc::clone(&fires);
                    async move {
                        fires.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        panic!("the turn blew up");
                    }
                }
            },
        );

        // Two fires means the loop survived the first panic, which is the whole claim.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while fires.load(std::sync::atomic::Ordering::SeqCst) < 2 {
            assert!(
                std::time::Instant::now() < deadline,
                "the scheduler stopped after {} fire(s)",
                fires.load(std::sync::atomic::Ordering::SeqCst),
            );
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        handle.abort();
    }

    /// The other half of a withdrawn gate: the job does not fire either.
    ///
    /// "Did not run" and "did not fire" are separate claims and both matter. A gate is the
    /// condition on the job, so a gate that cannot be evaluated has not passed, and delivering the
    /// prompt regardless turns a conditional job into an unconditional one. Delivering it was the
    /// first shape of this fix, and on an `every = "1m"` watcher it meant a turn a minute for as
    /// long as the session stayed below `unrestricted`.
    #[tokio::test]
    async fn a_job_whose_gate_cannot_be_run_does_not_fire_regardless() {
        let harness =
            SchedulerHarness::at_host_permission(crate::permission::Permission::Read).await;
        harness
            .overdue_job(
                Schedule::parse_every("1m").expect("parses"),
                Some(gate("true", GateFire::OnSuccess, None)),
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        assert!(
            harness.fired().is_empty(),
            "an unevaluated gate is not a passed gate",
        );
        let jobs = harness.jobs().await;
        let job = jobs.first().expect("the job survives for the next sweep");
        assert!(
            job.last_fired_at.is_none(),
            "and it must not be recorded as having fired",
        );
        assert!(
            job.next_fire_at > Utc::now(),
            "the occurrence is spent, so a restored session does not get a backlog",
        );
    }

    /// The companion: the same job, same recorded level, on a host still at `unrestricted`, runs.
    /// Without this the test above would pass just as well if gates never ran at all.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_gate_is_executed_while_the_host_still_holds_write() {
        let marker = std::env::temp_dir().join(format!("meka-gate-{}", uuid::Uuid::new_v4()));
        let _ = std::fs::remove_file(&marker);

        let harness = SchedulerHarness::new().await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate(
                    &create_file_command(&marker),
                    GateFire::OnSuccess,
                    None,
                )),
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        assert!(marker.exists(), "a fully authorised gate must still run");
        let _ = std::fs::remove_file(&marker);
    }

    /// The held-back explanation is a fact about a standing state, so it is said once.
    ///
    /// The sweep re-evaluates every due job, and a session parked below write does not change
    /// between sweeps. An `every = "1m"` job wrote the full explanation every minute for as long as
    /// it stayed there, which turns the one line an operator needs to see into the noise they stop
    /// reading. Restoring the authority arms it again, so the next withdrawal is not swallowed.
    #[test]
    fn a_job_held_back_for_permission_explains_itself_once_per_downgrade() {
        let job = format!("job-{}", uuid::Uuid::new_v4());

        assert!(
            declined_for_permission_first_time(&job),
            "the first sweep of a downgrade has to say why"
        );
        assert!(
            !declined_for_permission_first_time(&job),
            "and the ones after it must not repeat"
        );
        assert!(
            !declined_for_permission_first_time(&job),
            "however many there are"
        );

        clear_permission_decline(&job);
        assert!(
            declined_for_permission_first_time(&job),
            "a later withdrawal is a new fact and is announced again"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_gate_whose_authority_was_withdrawn_is_not_executed() {
        let marker = std::env::temp_dir().join(format!("meka-gate-{}", uuid::Uuid::new_v4()));
        let _ = std::fs::remove_file(&marker);

        let harness = SchedulerHarness::new().await;
        let mut withdrawn = gate(&create_file_command(&marker), GateFire::OnSuccess, None);
        withdrawn.permission = crate::permission::Permission::Read;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(withdrawn),
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        assert!(
            !marker.exists(),
            "a gate authorised at read must never reach the shell"
        );
        // And the occurrence is declined rather than delivered ungated; see
        // `a_job_whose_gate_cannot_be_run_does_not_fire_regardless` for why.
        assert!(harness.fired().is_empty());

        let _ = std::fs::remove_file(&marker);
    }

    /// An unreadable `gate_permission` decodes to the level that authorises nothing.
    ///
    /// This is the arm a row holding a value this build does not resolve hits.
    /// It fails *closed* -- `Permission::None` authorises no gate -- and nothing asserted it, so
    /// changing the fallback to `Unrestricted` left the suite green while every upgraded database
    /// silently regained unattended arbitrary shell. The fixtures all plant valid values, which is
    /// why the one value that matters on upgrade was the one never exercised.
    #[test]
    fn an_unreadable_gate_permission_authorises_nothing() {
        let row = |permission: Option<&str>| ScheduledJobRow {
            id: "7f3a1b2c-0000-0000-0000-000000000000".to_string(),
            session_id: uuid::Uuid::nil().to_string(),
            kind: "every".to_string(),
            spec: "1h".to_string(),
            prompt: "do the thing".to_string(),
            gate_command: Some("true".to_string()),
            gate_fire: Some("on-success".to_string()),
            gate_last_output: None,
            gate_permission: permission.map(str::to_string),
            isolated: false,
            created_at: Utc::now().to_rfc3339(),
            last_fired_at: None,
            next_fire_at: Utc::now().to_rfc3339(),
        };

        // A value no build of meka resolves.
        let unreadable = row(Some("elevated"))
            .decode()
            .expect("the row still decodes");
        let gate = unreadable
            .gate
            .expect("the gate survives; only its level is unreadable");
        assert_eq!(gate.permission, crate::permission::Permission::None);
        assert!(
            !gate.permission.allows_unattended_shell(),
            "an unreadable level must authorise no gate, or an upgrade re-arms every one of them"
        );

        // Absent is the same answer, for the same reason.
        let absent = row(None).decode().expect("decodes");
        assert_eq!(
            absent.gate.expect("gate").permission,
            crate::permission::Permission::None
        );

        // The control: a level meka still reads survives intact.
        let current = row(Some("unrestricted")).decode().expect("decodes");
        assert_eq!(
            current.gate.expect("gate").permission,
            crate::permission::Permission::Unrestricted
        );
    }

    /// The session's own recorded level decides, not the polling process's startup flag.
    ///
    /// This is the cross-process half of the withdrawal, and it was broken. A REPL session never
    /// wrote its level to the row, so `live_permission` fell through to
    /// `ResolvedScheduleConfig::host_permission` -- which for a `meka serve` sharing the data
    /// directory is that daemon's `--permission`, not anything the user touched. Shift+Tab-ing the
    /// REPL down therefore stopped the gate in the REPL and left serve firing it, unattended and
    /// unsandboxed, which is the opposite of what `scheduling.md` promises.
    ///
    /// The row is now written at session creation and on every level change, so this asserts the
    /// property that makes those writes matter: a host at `unrestricted` must still refuse a gate
    /// whose *session* has been withdrawn to `read`.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_withdrawn_session_beats_a_permissive_host() {
        let marker = std::env::temp_dir().join(format!("meka-gate-{}", uuid::Uuid::new_v4()));
        let _ = std::fs::remove_file(&marker);

        let harness = SchedulerHarness::new().await;
        // Authorised when it was created, exactly as `schedule_create` would have written it.
        let authorised = gate(&create_file_command(&marker), GateFire::OnSuccess, None);
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(authorised),
                chrono::Duration::minutes(5),
            )
            .await;

        // The session has since been withdrawn to `read` -- the row the REPL now keeps current.
        harness
            .manager
            .update_session_permission(harness.session_id, "read")
            .await
            .expect("record the withdrawal");

        // The polling host is at `unrestricted`, which is the whole point: before the row carried a
        // level, this flag is what answered and the gate ran.
        harness
            .tick_with(crate::config::ResolvedScheduleConfig {
                host_permission: crate::permission::Permission::Unrestricted,
                ..crate::config::ResolvedScheduleConfig::default()
            })
            .await;

        assert!(
            !marker.exists(),
            "the session's recorded `read` must beat the host's `unrestricted`, or Shift+Tab \
             withdraws nothing while a daemon is up"
        );
        assert!(harness.fired().is_empty());

        let _ = std::fs::remove_file(&marker);
    }

    /// The companion to the above: at `unrestricted` the same gate does run, so the refusal is
    /// about the recorded authority and not about gates having quietly stopped working.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_gate_that_still_holds_unrestricted_is_executed() {
        let marker = std::env::temp_dir().join(format!("meka-gate-{}", uuid::Uuid::new_v4()));
        let _ = std::fs::remove_file(&marker);

        let harness = SchedulerHarness::new().await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate(
                    &create_file_command(&marker),
                    GateFire::OnSuccess,
                    None,
                )),
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        assert!(
            marker.exists(),
            "a gate at unrestricted permission must run"
        );
        let _ = std::fs::remove_file(&marker);
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
            &SchedulerScope::OneSession(uuid::Uuid::new_v4()),
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
            &SchedulerScope::every_job(),
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
                .schedule_store()
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
            &SchedulerScope::every_job(),
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
            &SchedulerScope::every_job(),
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

    /// The arbitration between hosts, at the primitive. Two `meka serve` instances sharing a
    /// database both read the same occurrence into their due lists; exactly one of them may move
    /// the row off it. Before this the write was unconditional, so both advanced the row and both
    /// went on to fire.
    #[tokio::test]
    async fn test_only_one_host_can_claim_an_occurrence() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;
        let store = harness.manager.schedule_store();
        let next = job.next_fire_at + chrono::Duration::hours(1);

        assert!(
            store
                .claim_by_advancing(&job.id, job.next_fire_at, next)
                .await
                .expect("claim"),
            "the first host to reach the row takes the occurrence"
        );
        assert!(
            !store
                .claim_by_advancing(&job.id, job.next_fire_at, next)
                .await
                .expect("claim"),
            "and the second, still holding the copy it listed, is refused"
        );
        assert_eq!(
            harness
                .jobs()
                .await
                .first()
                .map(|job| job.next_fire_at)
                .expect("job survives"),
            next,
            "one advance, not two"
        );
    }

    /// The one-shot half. Claiming "remind me in 20 minutes" is a delete, and an unconditional
    /// `DELETE ... WHERE id = ?` reports success to every host that issues it -- so all of them
    /// deliver the reminder one of them removed.
    #[tokio::test]
    async fn test_only_one_host_can_retire_a_one_shot() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::At(Utc::now() - chrono::Duration::minutes(1)),
                None,
                chrono::Duration::minutes(1),
            )
            .await;
        let store = harness.manager.schedule_store();

        assert!(
            store
                .claim_by_retiring(&job.id, job.next_fire_at)
                .await
                .expect("claim")
        );
        assert!(
            !store
                .claim_by_retiring(&job.id, job.next_fire_at)
                .await
                .expect("claim"),
            "the row is gone, and the host that did not remove it must know"
        );
    }

    /// What a lost claim must cost: nothing. `prepare` evaluates the gate only after the claim is
    /// won, so a host that arrives second neither spawns the command nor produces a wakeup -- and
    /// leaves the winner's schedule exactly as the winner wrote it.
    ///
    /// Observed through a side effect on the filesystem for the same reason
    /// [`test_a_held_over_job_does_not_run_its_gate`] is: a gate that ran and was then discarded
    /// leaves every column identical to one that never ran.
    #[tokio::test]
    async fn test_a_lost_claim_runs_no_gate_and_produces_no_wakeup() {
        let harness = SchedulerHarness::new().await;
        struct Probe(std::path::PathBuf);
        impl Drop for Probe {
            fn drop(&mut self) {
                std::fs::remove_file(&self.0).ok();
            }
        }
        let guard =
            Probe(std::env::temp_dir().join(format!("meka-claim-probe-{}", uuid::Uuid::new_v4())));
        let probe = guard.0.clone();
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate(
                    &create_file_command(&probe),
                    GateFire::OnSuccess,
                    None,
                )),
                chrono::Duration::minutes(5),
            )
            .await;

        // The other host gets there first, while this one is still holding the copy it listed.
        let theirs = job.next_fire_at + chrono::Duration::hours(1);
        assert!(
            harness
                .manager
                .schedule_store()
                .claim_by_advancing(&job.id, job.next_fire_at, theirs)
                .await
                .expect("the other host claims")
        );

        let wakeup = prepare(&harness.manager, &harness.config, job, Utc::now())
            .await
            .expect("prepare runs");

        assert!(wakeup.is_none(), "a host that lost the claim does not fire");
        assert!(!probe.exists(), "and never ran the gate command");
        assert_eq!(
            harness
                .jobs()
                .await
                .first()
                .map(|job| job.next_fire_at)
                .expect("job survives"),
            theirs,
            "the winner's schedule is untouched"
        );
    }

    /// The one-shot half of a lost claim. Claiming a reminder is a delete, so the host that did not
    /// perform it holds a row that no longer exists -- and firing from that copy delivers "remind
    /// me in 20 minutes" twice.
    #[tokio::test]
    async fn test_a_lost_one_shot_claim_produces_no_wakeup() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::At(Utc::now() - chrono::Duration::minutes(1)),
                None,
                chrono::Duration::minutes(1),
            )
            .await;
        assert!(
            harness
                .manager
                .schedule_store()
                .claim_by_retiring(&job.id, job.next_fire_at)
                .await
                .expect("the other host claims")
        );

        let wakeup = prepare(&harness.manager, &harness.config, job, Utc::now())
            .await
            .expect("prepare runs");

        assert!(
            wakeup.is_none(),
            "the reminder belongs to the host that removed it"
        );
    }

    /// A deferral hands back the occurrence *this* host took, and nothing else. The restore used to
    /// be a whole-row upsert applied by id, so a host that lost the claim and was then refused the
    /// session lock overwrote the winner's `next_fire_at` with a time already in the past -- and
    /// the job came due again on the very next tick while the winner was still running the
    /// turn. One hourly occurrence produced three gate runs and two agent turns that way.
    #[tokio::test]
    async fn test_a_deferral_does_not_reach_past_its_own_claim() {
        let harness = SchedulerHarness::new().await;
        let planted = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;
        let manager = harness.manager.clone();
        // Where the row ends up if the restore respects the claim: a later occurrence, taken by
        // another host while this one was deciding it could not run the turn.
        let theirs = planted.next_fire_at + chrono::Duration::hours(9);

        run_due(
            &harness.manager,
            &harness.config,
            &SchedulerScope::every_job(),
            &move |wakeup: Wakeup| {
                let manager = manager.clone();
                async move {
                    let store = manager.schedule_store();
                    let ours = store
                        .list_all_scheduled_jobs()
                        .await
                        .expect("list jobs")
                        .first()
                        .map(|job| job.next_fire_at)
                        .expect("the claim advanced the row");
                    assert!(
                        store
                            .claim_by_advancing(&wakeup.job.id, ours, theirs)
                            .await
                            .expect("the other host claims the following occurrence")
                    );
                    FireOutcome::Deferred
                }
            },
        )
        .await
        .expect("tick runs");

        assert_eq!(
            harness
                .jobs()
                .await
                .first()
                .map(|job| job.next_fire_at)
                .expect("job survives"),
            theirs,
            "the deferral must not drag the row back onto an occurrence another host owns"
        );
    }

    /// The two writes that come *after* a claim carry the claimed time as a guard, and this is what
    /// that guard buys. A short interval with a slow gate leaves the claimed time already in the
    /// past by the time the gate returns, so another host can legitimately be running the following
    /// occurrence -- and an unguarded write would stamp this host's fire onto that host's row and
    /// drag the `on-change` baseline back to a value it has already reported on.
    #[tokio::test]
    async fn test_a_late_write_does_not_land_on_another_hosts_occurrence() {
        let harness = SchedulerHarness::new().await;
        let planted = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate("echo state", GateFire::OnChange, None)),
                chrono::Duration::minutes(5),
            )
            .await;
        let store = harness.manager.schedule_store();

        let ours = planted.next_fire_at + chrono::Duration::hours(1);
        assert!(
            store
                .claim_by_advancing(&planted.id, planted.next_fire_at, ours)
                .await
                .expect("claim")
        );
        // The following occurrence, taken by another host while this one's gate is still running.
        let theirs = ours + chrono::Duration::hours(1);
        assert!(
            store
                .claim_by_advancing(&planted.id, ours, theirs)
                .await
                .expect("the other host claims")
        );
        store
            .update_scheduled_job_gate_output(&planted.id, theirs, "theirs")
            .await
            .expect("the other host records its baseline");

        // This host finally finishes, and writes against the occurrence it claimed.
        store
            .stamp_scheduled_job_fired(&planted.id, Utc::now(), ours)
            .await
            .expect("stamp");
        store
            .update_scheduled_job_gate_output(&planted.id, ours, "ours")
            .await
            .expect("record baseline");

        let jobs = harness.jobs().await;
        let job = jobs.first().expect("job survives");
        assert_eq!(
            job.gate
                .as_ref()
                .and_then(|gate| gate.last_output.as_deref()),
            Some("theirs"),
            "a late baseline must not overwrite the one the current occurrence's host recorded"
        );
        assert!(
            job.last_fired_at.is_none(),
            "and a late stamp must not land on a row another host owns"
        );
    }

    /// The fallback arm of [`SAME_OCCURRENCE`]. Every writer in meka renders the column with
    /// `to_rfc3339`, so the textual comparison matches in practice -- but a row that reached the
    /// database any other way must still be claimable. The failure this guards against is the
    /// quietest one available: a compare-and-swap that matches nothing on every sweep, forever,
    /// with the job simply never firing again and not a line said about it.
    #[tokio::test]
    async fn test_a_timestamp_stored_in_another_shape_is_still_claimable() {
        let harness = SchedulerHarness::new().await;
        let planted = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;
        let store = harness.manager.schedule_store();
        // The same instant, written the way something that is not meka would write it.
        let raw = planted
            .next_fire_at
            .format("%Y-%m-%dT%H:%M:%S%.3fZ")
            .to_string();
        store
            .set_next_fire_at_verbatim_for_test(&planted.id, &raw)
            .await
            .expect("plant the timestamp");

        let due = store
            .list_due_scheduled_jobs(Utc::now())
            .await
            .expect("list due");
        let job = due.first().expect("still due");
        assert!(
            store
                .claim_by_advancing(
                    &job.id,
                    job.next_fire_at,
                    job.next_fire_at + chrono::Duration::hours(1)
                )
                .await
                .expect("claim"),
            "a job whose timestamp is not in meka's own shape must still be claimable"
        );
    }

    /// A job that really fires records that it fired, and a recurring one past the grace period is
    /// rescheduled rather than deleted.
    ///
    /// Two mutations survived the whole suite, including the cross-process tests. Emptying the
    /// `stamp_scheduled_job_fired` call at the end of `prepare` changed nothing any test could see
    /// -- the store method has its own test, and nothing checked that `prepare` calls it -- so
    /// every job would have read as never-fired in `meka schedule list` and an interval schedule
    /// would re-anchor on `created_at` after a restart and replay everything since.
    ///
    /// And `if !recurring && past_grace` still passed with the `!recurring` term forced true. The
    /// comment says "Recurring jobs need no equivalent rule"; `DEFAULT_MISSED_GRACE` is 24 hours
    /// and the latest fixture in the suite is 6 hours overdue, so the term was never the deciding
    /// factor. A laptop shut for a weekend would have had every recurring job silently retired.
    #[tokio::test]
    async fn a_fire_is_recorded_and_a_long_outage_does_not_retire_a_recurring_job() {
        let harness = SchedulerHarness::new().await;
        // Well past `DEFAULT_MISSED_GRACE`, which is what makes the `!recurring` term the only
        // thing standing between this job and deletion.
        let planted = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::days(3),
            )
            .await;

        let wakeup = prepare(&harness.manager, &harness.config, planted, Utc::now())
            .await
            .expect("prepare runs");

        assert!(
            wakeup.is_some(),
            "a recurring job is never past a grace period: its occurrences are one interval apart"
        );
        let job = harness
            .jobs()
            .await
            .first()
            .cloned()
            .expect("and the row survives rather than being retired");
        assert!(
            job.last_fired_at.is_some(),
            "a job that fires has to record it, or a restart re-anchors on `created_at` and \
             replays every occurrence since"
        );
        assert!(
            job.next_fire_at > Utc::now(),
            "and be scheduled forward rather than left due"
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
