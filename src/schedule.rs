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

/// How a gate obtains the value it judges.
///
/// Split from [`GatePredicate`] because the two answer different questions and only one of them is
/// shell-shaped. Welding them together is what made the old `on-success` (an exit code) meaningless
/// for anything but a command, and it is why a tool result containing a timestamp could only ever
/// be described as "changed".
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GateProbe {
    /// A shell command, run unsandboxed. Requires `unrestricted`; see
    /// [`crate::permission::Permission::allows_unattended_shell`].
    Shell { command: String },
    /// A tool call, by the same name the model would use (`mcp__server__tool`, or a built-in).
    ///
    /// Deliberately *not* held to `unrestricted`: a structured call to a server the operator
    /// configured is not a bare `sh -c` with meka's environment, so the bar is the tool's own
    /// level. See [`gate_probe_is_authorised`].
    Tool {
        name: String,
        #[serde(default)]
        arguments: serde_json::Value,
    },
}

impl GateProbe {
    /// Discriminant stored in `scheduled_jobs.gate_kind`, mirroring [`Schedule::kind_str`] so a row
    /// can be read for its shape without parsing the spec.
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::Shell { .. } => "shell",
            Self::Tool { .. } => "tool",
        }
    }

    /// How the probe reads in a listing, short enough for a one-line summary.
    ///
    /// A tool's arguments are deliberately absent. They can be long and can carry a token the
    /// caller pasted into a gate, and this feeds a `Check` column and an HTTP field. Where the
    /// arguments matter, use [`Self::detail`].
    pub fn summary(&self) -> String {
        match self {
            Self::Shell { command } => command.clone(),
            Self::Tool { name, .. } => name.clone(),
        }
    }

    /// The probe with its kind named and a tool's arguments attached, for the one reader that needs
    /// them.
    ///
    /// `schedule_list` is that reader: the model wrote those arguments and cannot otherwise read
    /// back what it created, so a gate it built with the wrong `since` looks identical to a correct
    /// one. Every other surface stays on [`Self::summary`], because the operator's listing and the
    /// HTTP view are read by parties who did not author the job.
    ///
    /// The kind is named because the two are otherwise indistinguishable where a tool's name would
    /// also be a valid command: `fetch_url` as a shell gate and `fetch_url` as a tool gate rendered
    /// identically, and they are an unsandboxed `sh -c` and a structured call. The model needs the
    /// difference to recreate the job it is reading back.
    pub fn detail(&self) -> String {
        match self {
            Self::Shell { command } => format!("shell {}", command),
            Self::Tool { name, arguments } => match arguments {
                // An omitted or empty argument object is the common case and adds nothing.
                serde_json::Value::Null => format!("tool {}", name),
                serde_json::Value::Object(fields) if fields.is_empty() => format!("tool {}", name),
                other => format!("tool {} {}", name, truncate_gate_output(&other.to_string())),
            },
        }
    }
}

/// Which value a [`GatePredicate::At`] test is applied to, and how.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PointerTest {
    /// A non-empty array, object or string, or any non-null scalar.
    NotEmpty,
    /// The inverse, including a pointer that resolves to nothing.
    Empty,
    /// The pointed-at value differs from the previous evaluation's.
    Changed,
}

/// What the probe's result has to look like for the job to fire.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GatePredicate {
    /// The whole output differs from the previous evaluation's. Edge-triggered: "tell me when the
    /// build *finishes*", not "tell me every 30s while it is running".
    Changed,
    /// The probe reported success: a shell command exiting 0, or a tool call that did not come back
    /// as an error. Level-triggered, for "is this true yet".
    Succeeded,
    /// The output matches a regular expression.
    Matches { pattern: String },
    /// A JSON pointer into the result satisfies `is`.
    ///
    /// The reason this exists. A structured result carrying anything self-moving (a `checked_at`, a
    /// request id) is different on every single call, so [`Self::Changed`] over the whole of it
    /// fires every interval and costs exactly the turns a gate is supposed to save. Pointing at the
    /// part that matters is the only honest way to watch one.
    At { pointer: String, is: PointerTest },
}

impl GateProbe {
    /// Parse the `check` half of a gate request.
    ///
    /// Hand-written rather than derived because the request shape and the stored shape answer to
    /// different readers. Storage is meka talking to itself and uses serde's tagging; a request is
    /// authored by a model or typed into a `curl`, so it reads `{"command": ...}` rather than
    /// `{"shell": {"command": ...}}`, and a wrong one has to say what was wrong.
    pub fn parse_request(value: Option<&serde_json::Value>) -> std::result::Result<Self, String> {
        let Some(object) = value.and_then(|value| value.as_object()) else {
            return Err("`check` must be an object naming either `command` or `tool`".to_string());
        };
        let command = object.get("command").filter(|value| !value.is_null());
        let tool = object.get("tool").filter(|value| !value.is_null());
        match (command, tool) {
            (Some(command), None) => {
                let command = command
                    .as_str()
                    .ok_or_else(|| "`check.command` must be a string".to_string())?;
                if command.trim().is_empty() {
                    return Err("`check.command` cannot be empty".to_string());
                }
                Ok(Self::Shell {
                    command: command.to_string(),
                })
            }
            (None, Some(tool)) => {
                let name = tool
                    .as_str()
                    .ok_or_else(|| "`check.tool` must be a tool name".to_string())?;
                if name.trim().is_empty() {
                    return Err("`check.tool` cannot be empty".to_string());
                }
                // Checked against the shape the tool schema declares. Anything else reaches the
                // tool as null arguments, so the gate errors on every interval instead of being
                // refused once, here, by the door that could have said which field was wrong.
                let arguments = match object.get("arguments") {
                    None | Some(serde_json::Value::Null) => serde_json::json!({}),
                    Some(value) if value.is_object() => value.clone(),
                    Some(_) => {
                        return Err("`check.arguments` must be an object".to_string());
                    }
                };
                Ok(Self::Tool {
                    name: name.to_string(),
                    arguments,
                })
            }
            // Naming both is refused rather than resolved by precedence: the two run entirely
            // different things, and guessing which was meant is how a gate ends up watching
            // something nobody asked it to watch.
            (Some(_), Some(_)) => {
                Err("`check` names both `command` and `tool`; use one".to_string())
            }
            (None, None) => Err("`check` must name either `command` or `tool`".to_string()),
        }
    }
}

impl GatePredicate {
    /// Parse the `when` half of a gate request. Absent means [`Self::Changed`], which is the
    /// predicate most watchers want and the one the old `fire` field defaulted to.
    pub fn parse_request(value: Option<&serde_json::Value>) -> std::result::Result<Self, String> {
        const EXPECTED: &str = "expected \"changed\", \"succeeded\", {\"matches\": \"<regex>\"} or \
                                {\"at\": \"<json pointer>\", \"is\": \"not-empty\"|\"empty\"|\"changed\"}";

        let Some(value) = value.filter(|value| !value.is_null()) else {
            return Ok(Self::Changed);
        };
        if let Some(word) = value.as_str() {
            return match word {
                "changed" => Ok(Self::Changed),
                "succeeded" => Ok(Self::Succeeded),
                other => Err(format!("unknown gate condition '{}'; {}", other, EXPECTED)),
            };
        }
        let Some(object) = value.as_object() else {
            return Err(format!("`when` is not a condition; {}", EXPECTED));
        };
        // Refused rather than resolved, exactly as `check` refuses naming both `command` and
        // `tool`. Taking `matches` and ignoring `at` gave the model a gate watching something it
        // did not ask for, silently, at both creation doors -- and the two halves of a `when` that
        // names both are usually meant as *different* conditions, so neither reading is safe.
        if object.contains_key("matches") && object.contains_key("at") {
            return Err(format!(
                "`when` names both `matches` and `at`; give exactly one. {}",
                EXPECTED
            ));
        }
        if let Some(pattern) = object.get("matches") {
            let pattern = pattern
                .as_str()
                .ok_or_else(|| "`when.matches` must be a regular expression".to_string())?;
            // Compiled here so a bad pattern is refused by the door that accepted it, rather than
            // becoming a gate that silently never fires.
            regex::Regex::new(pattern)
                .map_err(|error| format!("`when.matches` does not compile: {}", error))?;
            return Ok(Self::Matches {
                pattern: pattern.to_string(),
            });
        }
        if let Some(pointer) = object.get("at") {
            let pointer = pointer
                .as_str()
                .ok_or_else(|| "`when.at` must be a JSON pointer such as \"/chats\"".to_string())?;
            if !pointer.is_empty() && !pointer.starts_with('/') {
                return Err(format!(
                    "`when.at` must be a JSON pointer starting with '/', got '{}'",
                    pointer
                ));
            }
            let is = match object.get("is").and_then(|value| value.as_str()) {
                Some("not-empty") | None => PointerTest::NotEmpty,
                Some("empty") => PointerTest::Empty,
                Some("changed") => PointerTest::Changed,
                Some(other) => {
                    return Err(format!(
                        "unknown `when.is` '{}'; expected 'not-empty', 'empty' or 'changed'",
                        other
                    ));
                }
            };
            return Ok(Self::At {
                pointer: pointer.to_string(),
                is,
            });
        }
        Err(format!("`when` is not a condition; {}", EXPECTED))
    }

    /// How the predicate reads in a listing.
    pub fn summary(&self) -> String {
        match self {
            Self::Changed => "changed".to_string(),
            Self::Succeeded => "succeeded".to_string(),
            Self::Matches { pattern } => format!("matches /{}/", pattern),
            Self::At { pointer, is } => format!("{} {}", pointer, match is {
                PointerTest::NotEmpty => "not-empty",
                PointerTest::Empty => "empty",
                PointerTest::Changed => "changed",
            }),
        }
    }
}

/// The cheap check that decides whether a due job spends a model turn.
///
/// This is the whole reason a 30-second cadence is affordable: without it, watching something costs
/// one model turn per interval whether or not anything happened.
#[derive(Debug, Clone)]
pub struct Gate {
    pub probe: GateProbe,
    pub predicate: GatePredicate,
    /// The comparison baseline from the last evaluation, for the predicates that need one. `None`
    /// until the first run, at which point the job fires: with nothing to compare against,
    /// "changed" is the honest answer, and it also proves the gate works rather than leaving
    /// it silently untested.
    ///
    /// Not always the same bytes the turn saw. [`GatePredicate::At`] with
    /// [`PointerTest::Changed`] stores the *pointed-at* value, because storing the whole result
    /// would re-admit the moving field the pointer was chosen to exclude.
    pub last_output: Option<String>,
    /// The permission level the creating session held when this gate was authorised.
    ///
    /// Creation checks the level, but creation is a moment and the row outlives it: the session
    /// drops to `read`, or `meka serve --permission read` restarts and inherits the job, and
    /// without this field nothing downstream can tell that the authority behind the gate is
    /// gone. Carrying the level on the row is what lets [`prepare`] re-check it at fire time
    /// instead of trusting a decision made days ago.
    pub permission: crate::permission::Permission,
}

/// The parts of a gate that round-trip through `scheduled_jobs.gate_spec` as one JSON value.
///
/// `last_output` and `permission` stay in their own columns: the first is rewritten on every
/// evaluation and the second is read by the fire-time authority check, and neither wants a
/// parse-and-reserialise to touch it.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct GateSpec {
    #[serde(flatten)]
    probe: GateProbe,
    when: GatePredicate,
}

impl Gate {
    /// The `gate_spec` column: probe and predicate as one JSON value.
    pub fn spec(&self) -> String {
        // Both halves are meka's own types, so the only way this fails is a serde bug. An empty
        // spec would decode as a corrupt row and refuse the gate, which is the safe direction.
        serde_json::to_string(&GateSpec {
            probe: self.probe.clone(),
            when: self.predicate.clone(),
        })
        .unwrap_or_default()
    }

    /// Rebuild a gate from its columns.
    ///
    /// `kind` is validated against the spec rather than trusted: the two are written together, so
    /// disagreement means a hand-edited or damaged row, and a gate whose stored shape cannot be
    /// read must not resolve to some other shape that happens to parse.
    pub fn from_stored(
        kind: &str,
        spec: &str,
        last_output: Option<String>,
        permission: crate::permission::Permission,
    ) -> std::result::Result<Self, String> {
        let parsed: GateSpec = serde_json::from_str(spec)
            .map_err(|error| format!("unreadable gate spec: {}", error))?;
        if parsed.probe.kind_str() != kind {
            return Err(format!(
                "gate_kind '{}' does not match its spec, which describes a '{}' gate",
                kind,
                parsed.probe.kind_str()
            ));
        }
        Ok(Self {
            probe: parsed.probe,
            predicate: parsed.when,
            last_output,
            permission,
        })
    }
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
    /// Claims that ended without delivering the turn and without the *host* declining the job.
    ///
    /// A claim raises it, and it is only cleared by an ending that says the job is fine. Two
    /// endings do not: a host that dies or panics mid-delivery, and a job with no next occurrence
    /// whose gate probe could not be evaluated, which keeps its lease so the retry waits out
    /// `claim_lease` rather than coming round on the next tick. Both leave an occurrence that
    /// nothing has spent, so something has to bound how often it is retried;
    /// [`MAX_CLAIM_ATTEMPTS`] is where a job that keeps doing it stops being retried, which is
    /// what stands in for the old protection of spending the occurrence before the turn ran.
    ///
    /// A deferral does *not* raise it. That is a host saying "not me", which is a fact about the
    /// host rather than the job, so [`ScheduleStore::release_claim`] resets the count.
    pub attempts: u32,
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
    /// one unanswered message per fire through an outage. A one-shot does not: its row is retired
    /// the moment the turn is delivered, so once a failed fire has been through `complete_claim`
    /// the unanswered message is the last trace that the reminder ever existed, and withdrawing it
    /// would be the deletion the feature is supposed to prevent.
    ///
    /// The row now survives *during* the turn rather than being deleted before it, which is what
    /// leasing changed. That is why the reasoning is about the completion rather than the claim:
    /// the outcome is the same, and a reader tracing the old sentence would look for a delete that
    /// no longer happens there.
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

/// Ceiling on a probe result meka will parse as JSON, which is a different question from how much
/// of it the turn is shown.
///
/// Separate from [`GATE_OUTPUT_LIMIT`] because the two bound different costs. That one bounds the
/// prompt; this one bounds the work an evaluation does, which is the larger number: a `serde_json`
/// `Value` runs several times the size of its input, and a pointer predicate re-serialises the
/// part it judges. A megabyte is far past any status a gate should be reading and far short of
/// what a runaway command can emit.
const GATE_PARSE_LIMIT: usize = 1024 * 1024;

/// What a gate evaluation decided.
#[derive(Debug, Clone)]
pub struct GateOutcome {
    /// Whether to spend a model turn.
    pub fired: bool,
    /// The probe's result, trimmed and truncated. Handed to the turn as context when `fired`.
    pub output: String,
    /// What to persist as the next evaluation's comparison baseline.
    ///
    /// Usually the same as `output`, and separate from it for one predicate:
    /// [`GatePredicate::At`] with [`PointerTest::Changed`] compares the pointed-at value, so
    /// storing the whole result would re-admit the moving field the pointer exists to exclude and
    /// the gate would fire every interval.
    pub baseline: String,
}

/// What a probe produced, before any predicate is applied to it.
#[derive(Debug, Clone)]
pub struct ProbeOutcome {
    /// The result as text, trimmed and capped at [`GATE_OUTPUT_LIMIT`]. What the turn is shown.
    pub text: String,
    /// The machine-readable result, when there is one: a tool's `structuredContent`, or whatever
    /// the untruncated text parsed as.
    pub structured: Option<serde_json::Value>,
    /// Whether the probe itself reported success: exit 0, or a tool call that was not an error.
    pub succeeded: bool,
}

impl ProbeOutcome {
    /// Assemble a result, parsing before truncating.
    ///
    /// The order is the point. `text` is capped at [`GATE_OUTPUT_LIMIT`] and gains a truncation
    /// marker, and [`pointed_at`] falls back to parsing that text whenever there is no structured
    /// value -- which is the path every shell probe takes, and every MCP server that returns its
    /// JSON as text content, which is most of them. A document over the cap therefore never parsed
    /// again, so an `at` gate over it failed permanently with "the probe did not return JSON". It
    /// did; meka truncated it.
    ///
    /// Parsing `raw` and keeping the result means the cap goes on being what it is for -- bounding
    /// what a runaway probe can push into the turn's context -- without deciding what the gate is
    /// allowed to judge.
    pub(crate) fn new(raw: &str, structured: Option<serde_json::Value>, succeeded: bool) -> Self {
        // Applied to a value the caller already parsed, not only to the fallback below. An MCP
        // server's `structuredContent` arrives as a `Value` and took the `or_else` branch's cap
        // with it -- which is to say the cap covered shell probes and text-only servers, and
        // missed the path the feature was built for.
        //
        // Serialising to measure looks circular and is not: it happens once, here, against a
        // predicate that would otherwise re-serialise the same value on every evaluation
        // (`canonical_json(...).to_string()` in the `At` arm). What this cannot do is un-receive
        // the value: the MCP layer parsed it before meka saw it, so the peak allocation has
        // already been paid. The bound is on what meka keeps and keeps re-doing.
        let structured = structured.filter(|value| {
            serde_json::to_string(value).is_ok_and(|rendered| rendered.len() <= GATE_PARSE_LIMIT)
        });
        let structured = structured.or_else(|| {
            // Bounded separately from the display cap, because relaxing that cap quietly removed
            // the only bound on this. `text` is capped so a runaway probe cannot push the prompt
            // over the context window; parsing what the cap had already trimmed *also* meant every
            // allocation downstream was bounded by 8 KiB. Parsing `raw` instead is what makes a
            // large result readable, and it hands a probe that returns hundreds of megabytes a
            // `Value` several times that size -- built, and for a pointer predicate re-serialised
            // whole, on the scheduler's own task, on every evaluation.
            //
            // A megabyte covers any result a gate has business judging while keeping the cost of
            // a hostile or runaway one flat. Past it there is no structured value, so a pointer
            // predicate declines and says the probe did not return JSON, which is the same answer
            // it gives for a result it genuinely cannot read.
            (raw.len() <= GATE_PARSE_LIMIT)
                .then(|| serde_json::from_str::<serde_json::Value>(raw.trim()).ok())
                .flatten()
        });
        Self {
            text: truncate_gate_output(raw),
            structured,
            succeeded,
        }
    }
}

/// Why a gate may not run at a given level.
///
/// One type so the doors that ask -- `schedule_create`, `POST /v1/sessions/{id}/schedule`,
/// and the fire-time re-check in [`prepare`] -- give the same answer for the same state. They used
/// to phrase the single shell rule three ways, which is how one of them came to name a mode that no
/// longer existed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateRefusal {
    /// A shell gate is a bare `sh -c` on a timer with nobody watching.
    ShellNeedsUnrestricted,
    /// The tool is not registered, or its server is not connected, right now.
    ToolUnavailable,
    /// The tool resolves above `read`. A gate asks a question; a tool that can act is not one.
    ToolNotReadOnly(crate::permission::Permission),
    /// The tool is read-only, but the session is not even at `read`.
    SessionBelowTool,
}

impl GateRefusal {
    /// The user- and model-facing reason, naming the level actually held.
    pub fn explain(&self, probe: &GateProbe, level: crate::permission::Permission) -> String {
        match self {
            // `ask` gets its own sentence. At every other level below the bar the reason is the
            // missing sandbox, but at `ask` there is no sandbox for anything and the objection is
            // different in kind: the approval prompt that is the level's entire safety has nobody
            // to answer it on a timer. Telling an `ask` user about a sandbox sends them looking for
            // a setting that would not help.
            Self::ShellNeedsUnrestricted if level == crate::permission::Permission::Ask => {
                "a gate command runs unattended, with nobody present to approve it, so `ask` is \
                 not enough; it needs `unrestricted`"
                    .to_string()
            }
            Self::ShellNeedsUnrestricted => format!(
                "a gate command runs unattended with no sandbox, so it needs `unrestricted` \
                 (currently {})",
                level
            ),
            // Deliberately not "right now". That reads as transient, and the common cause is not:
            // a name that does not exist, or a session-scoped tool a gate could never reach, is
            // permanent, and a model told "right now" will keep the job and wait. The
            // genuinely-transient case is a server still connecting, which the reporting surfaces
            // decline to mention at all until it settles.
            Self::ToolUnavailable => format!(
                "no gate tool named `{}`. A gate can call a read-only tool that does not depend on \
                 the session -- an MCP tool, or one of `read_file`, `find_files`, \
                 `search_contents`, `fetch_url`, `search_web`, and `execute_command` where a \
                 sandbox is available -- or the server providing it is not connected",
                probe.summary()
            ),
            Self::ToolNotReadOnly(required) => format!(
                "gate tool `{}` requires `{}`; a gate may only call a tool that resolves to `read`",
                probe.summary(),
                required
            ),
            Self::SessionBelowTool => format!(
                "gate tool `{}` needs `read` (currently {})",
                probe.summary(),
                level
            ),
        }
    }
}

/// Whether `level` may author or fire this probe, re-resolving the tool every time it is asked.
///
/// Both halves are checked, and both are checked *now* rather than trusted from creation. A job
/// authored at `unrestricted` must stop firing its command once the session drops, or a daemon runs
/// an unsandboxed command after the user lowered the mode and is entitled to be surprised. And a
/// tool that resolved to `read` when the job was written but resolves higher today must stop being
/// a gate, because the operator retuned `tool_permissions` and meant it.
pub fn gate_probe_is_authorised(
    probe: &GateProbe,
    level: crate::permission::Permission,
    tools: Option<&dyn GateTools>,
) -> std::result::Result<(), GateRefusal> {
    match probe {
        GateProbe::Shell { .. } => {
            if level.allows_unattended_shell() {
                Ok(())
            } else {
                Err(GateRefusal::ShellNeedsUnrestricted)
            }
        }
        GateProbe::Tool { name, .. } => {
            // No dispatcher means this process cannot resolve the name, which is the same answer as
            // a disconnected server: not right now.
            let Some(required) = tools.and_then(|tools| tools.resolve(name)) else {
                return Err(GateRefusal::ToolUnavailable);
            };
            if required != crate::permission::Permission::Read {
                return Err(GateRefusal::ToolNotReadOnly(required));
            }
            if !level.allows(crate::permission::Permission::Read) {
                return Err(GateRefusal::SessionBelowTool);
            }
            Ok(())
        }
    }
}

/// Why `gate` will not fire right now, and the level that answer was reached at.
///
/// One function for three readers: the fire door in [`prepare`], the `[Scheduled]` index the model
/// sees every turn, and `schedule_list`. Before this the fire door was the only one that asked, so
/// a held-back job was reported to the operator's log and to nobody else: the model saw a job that
/// looked healthy, could not tell a gate that had said "no" from one that was never consulted, and
/// had nothing to act on. It can cancel a job it cannot fire, so the asymmetry was worth closing.
///
/// The live level is tried first because it is the one that can be put back. A refusal that only
/// the *recorded* level produces means a row nothing can currently restore, which is a different
/// thing to say.
pub fn gate_withheld_reason(
    gate: &Gate,
    live: crate::permission::Permission,
    tools: Option<&dyn GateTools>,
) -> Option<(GateRefusal, crate::permission::Permission)> {
    if let Err(refusal) = gate_probe_is_authorised(&gate.probe, live, tools) {
        return Some((refusal, live));
    }
    if let Err(refusal) = gate_probe_is_authorised(&gate.probe, gate.permission, tools) {
        return Some((refusal, gate.permission));
    }
    None
}

/// Why this job will not fire right now, phrased for the model, or `None` if it will.
///
/// The whole answer, not just the gate's half: a session at `none` withholds every job, gated or
/// not, and an *ungated* job is exactly the case a gate-shaped question misses. Without this an
/// ungated reminder on such a session read as perfectly healthy on every surface while never
/// firing -- the same "held and healthy look identical" problem the gate marker exists to solve,
/// one level up, and a disagreement between the creation door (which accepts) and the fire door
/// (which refuses).
pub fn job_withheld_reason(
    job: &ScheduledJob,
    live: crate::permission::Permission,
    tools: Option<&dyn GateTools>,
) -> Option<String> {
    match job_withheld(job, Some(live), tools) {
        Withheld::Yes(reason) => Some(reason),
        Withheld::No | Withheld::Undetermined => None,
    }
}

/// What a reader is entitled to say about whether a job will fire.
///
/// Three answers rather than two, because a reader without a dispatcher cannot resolve a tool gate
/// and "I cannot tell" is not "it is fine". Collapsing them is right for the surfaces that render a
/// *sentence* -- there is nothing to say -- and wrong for one that renders a *column*, where the
/// empty cell beside a populated one reads as a verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Withheld {
    /// It will fire, as far as this reader can establish.
    No,
    /// It will not, for this reason.
    Yes(String),
    /// A tool gate this reader cannot resolve: no dispatcher at all, or a server still connecting.
    Undetermined,
}

/// The three-way form of [`job_withheld_reason`], for a reader that can express "I cannot tell".
pub fn job_withheld(
    job: &ScheduledJob,
    live: Option<crate::permission::Permission>,
    tools: Option<&dyn GateTools>,
) -> Withheld {
    // First, because it is the most specific, the one nothing else will explain, and the only
    // verdict here that needs no permission level. A parked job has a healthy gate and an adequate
    // session, so every other question answers "it will fire" while the fire door refuses it on
    // every sweep -- and asking it before the level means a reader that could not establish one
    // still reports the job it can see is dead.
    if job.attempts >= MAX_CLAIM_ATTEMPTS {
        return Withheld::Yes(parked_reason(job));
    }
    let Some(live) = live else {
        return Withheld::Undetermined;
    };
    if !live.allows_unattended_work() {
        return Withheld::Yes(format!(
            "the session is at {}, where nothing is executable, so a scheduled turn could neither \
             act on this nor cancel it",
            live
        ));
    }
    let Some(gate) = job.gate.as_ref() else {
        return Withheld::No;
    };
    let Some((refusal, level)) = gate_withheld_reason(gate, live, tools) else {
        return match standing_probe_failure(job) {
            Some(reason) => Withheld::Yes(reason),
            None => Withheld::No,
        };
    };
    // `ToolUnavailable` is the one refusal this function will not report on a guess, because it is
    // the one that can mean "I cannot tell" rather than "it is broken".
    //
    // Two readers hit that. A caller with no dispatcher at all -- `meka schedule list`, which has
    // no MCP manager -- would otherwise report *every* tool gate as dead, libelling healthy jobs to
    // the one audience that cannot check. And a server still completing its first handshake is not
    // a verdict yet: reporting one marks a healthy job dead for the second it takes and announces
    // it alive again a turn later, on every start and every reconnect.
    //
    // The fire door is unaffected in both cases: it still declines the occurrence, because a gate
    // whose tool cannot be resolved cannot be evaluated. Silence here is about what we are entitled
    // to *say*, not about what runs.
    if matches!(refusal, GateRefusal::ToolUnavailable)
        && let GateProbe::Tool { name, .. } = &gate.probe
        && tools.is_none_or(|tools| tools.is_still_connecting(name))
    {
        return Withheld::Undetermined;
    }
    Withheld::Yes(refusal.explain(&gate.probe, level))
}

/// Why a parked job stopped, said only as far as the row can support.
///
/// Two things fill `attempts`, and they have opposite remedies: a prompt that takes the host down,
/// and a gate probe that can never answer. The probe's error is the discriminator when this process
/// has one, but [`PROBE_FAILURES`] is per-process, so a restart -- which is exactly what an
/// operator does after noticing a job has gone inert -- loses it, and `meka schedule list` never
/// had it. Asserting the commoner cause from that absence produced the worst outcome available:
/// telling someone whose MCP server was misconfigured that their prompt crashes meka, with a
/// remedy aimed at the wrong artefact, on the model's own `[Scheduled]` block.
///
/// So absence is treated as absence. The row does still settle it in one direction: a job with no
/// gate has no probe that could have failed, so a crash is the only thing left and can be named
/// outright.
fn parked_reason(job: &ScheduledJob) -> String {
    let opening = format!("{} claims ended without delivering", job.attempts);
    match (probe_failure(&job.id), job.gate.is_some()) {
        (Some((_, error)), _) => format!(
            "{}, because its gate could not be evaluated: {}. It is no longer retried; fix the \
             check by recreating the job, or cancel it",
            opening,
            elide_for_message(&error)
        ),
        (None, false) => format!(
            "{} or handing back, and it has no gate that could have failed, so the host died each \
             time. It is no longer retried. Cancel it, or recreate it with a prompt that does not \
             take the process down",
            opening
        ),
        (None, true) => format!(
            "{} or handing back, so either its gate cannot be evaluated or the turn takes the host \
             down; this process no longer has the record that would say which. It is no longer \
             retried. Run the gate's check by hand, and cancel or recreate the job",
            opening
        ),
    }
}

/// Why an *authorised* gate is still not firing: its probe keeps breaking.
///
/// Authority is not the only way a watcher dies, and it is not the commonest. A server that changed
/// its schema, a command that was uninstalled, a pointer into a result that stopped being JSON:
/// each produces a gate that errors on every evaluation, and each looks from the model's side
/// exactly like a healthy watcher with nothing to report. The marker existed for that
/// indistinguishability and covered only half of it.
fn standing_probe_failure(job: &ScheduledJob) -> Option<String> {
    let (failures, error, witness) = {
        let held = match PROBE_FAILURES.lock() {
            Ok(held) => held,
            Err(poisoned) => poisoned.into_inner(),
        };
        held.get(&job.id).cloned()?
    };
    // The verdict is this process's, but the job is not. Only the host that wins `claim_occurrence`
    // evaluates, and `isolated` jobs are exempt from the session-residency filter, so a second
    // `meka serve` on the same store can take over every occurrence and heal the gate without this
    // process ever hearing. Nothing here re-enters the counting path in that case, so without this
    // check the marker stood forever: the model was told, every turn, that a job firing hourly was
    // dead.
    //
    // Neither half of the witness is written by the failing path -- it persists no `fired_at` and
    // no baseline, deliberately -- so either of them having moved is proof of a successful
    // evaluation since. The gap it cannot close is a gate that keeps evaluating, keeps declining,
    // and keeps producing the identical output: an unchanged row cannot testify to anything.
    if witness != probe_witness(job) {
        clear_probe_failure(&job.id);
        return None;
    }
    if failures < PROBE_FAILURES_BEFORE_REPORTING {
        return None;
    }
    // No count in the sentence, deliberately.
    //
    // Every reader of this compares it by equality. `render_world_state_diff` announces a job to
    // the model when its withheld reason *changes*, so a running total made the reason change on
    // every failed evaluation and the model was told "can no longer fire: … 7 evaluations", then
    // 8, then 9, for as long as the gate stayed broken. `context.rs` already leaves next-fire
    // times out of the snapshot for exactly this reason; a counter is the same mistake wearing a
    // different hat.
    //
    // The number is not lost: it is in the `warn!` at each failure, where an event belongs, and
    // `-v` shows it. What the model needs is the standing fact and what to do about it, and that
    // does not change between the second failure and the two-hundredth.
    Some(format!(
        "its gate keeps failing and cannot say whether to fire: {}. Fix the check by recreating \
         the job, or cancel it",
        elide_for_message(&error)
    ))
}

/// Where a gate's [`GateProbe::Tool`] call is dispatched.
///
/// A trait rather than a concrete handle so `schedule` does not take a dependency on the tool and
/// MCP stacks, which would be circular. `src/tools.rs` supplies the implementation.
#[async_trait::async_trait]
pub trait GateTools: Send + Sync + std::fmt::Debug {
    /// Look up a tool by the name the model would use, and report the permission it currently
    /// resolves to.
    ///
    /// `None` when the name is unknown *or* its server is not connected. Both are the same answer
    /// for a gate: it cannot be evaluated right now, so it has not passed.
    fn resolve(&self, name: &str) -> Option<crate::permission::Permission>;

    /// Whether this name might still resolve once its server finishes connecting.
    ///
    /// Only the *reporting* surfaces ask. Authority does not: a gate whose server is mid-handshake
    /// genuinely cannot run, and [`Self::resolve`] returning `None` is the right answer there. But
    /// saying "not available right now" in the model's `[Scheduled]` block during startup marks a
    /// healthy job as dead and then announces it alive again a turn later, which is worse than
    /// saying nothing for the second it takes.
    ///
    /// Defaulted to `false` so a dispatcher with no notion of connecting -- every test stub, and
    /// any future non-MCP one -- keeps the plain behaviour.
    fn is_still_connecting(&self, _name: &str) -> bool {
        false
    }

    /// Call it, in the creating session's directory. Only reached once [`Self::resolve`] has
    /// answered and the authority check has passed.
    ///
    /// `cwd` is here and not on `resolve` because only the call needs it: what a tool *requires* is
    /// a property of the tool, while where it runs is a property of the job.
    async fn call(
        &self,
        name: &str,
        arguments: &serde_json::Value,
        timeout: Duration,
        cwd: Option<&std::path::Path>,
    ) -> Result<ProbeOutcome, String>;
}

/// Run a gate and decide whether the job it guards should fire.
///
/// Errors are for the gate itself failing (spawn failure, timeout, a tool that cannot be reached),
/// never for the condition being false. The distinction matters: a watcher that goes quiet because
/// its probe broke looks exactly like a healthy watcher with nothing to report, so the caller must
/// surface an `Err` rather than treating it as "no change".
pub async fn evaluate_gate(
    gate: &Gate,
    timeout: Duration,
    cwd: Option<&std::path::Path>,
    tools: Option<&dyn GateTools>,
) -> Result<GateOutcome, String> {
    let probe = run_probe(&gate.probe, timeout, cwd, tools).await?;
    apply_predicate(&gate.predicate, &probe, gate.last_output.as_deref())
}

/// Obtain the value a gate judges, without judging it.
async fn run_probe(
    probe: &GateProbe,
    timeout: Duration,
    cwd: Option<&std::path::Path>,
    tools: Option<&dyn GateTools>,
) -> Result<ProbeOutcome, String> {
    match probe {
        GateProbe::Shell { command } => run_shell_probe(command, timeout, cwd).await,
        GateProbe::Tool { name, arguments } => {
            let Some(tools) = tools else {
                // Not a misconfiguration to report at creation: the host that authored the job can
                // dispatch tools, and this is a *different* host picking the row up. Declining is
                // the same answer a disconnected server gets, for the same reason.
                return Err(format!(
                    "gate calls `{}`, which this process cannot dispatch",
                    name
                ));
            };
            tools.call(name, arguments, timeout, cwd).await
        }
    }
}

/// The shell probe: unsandboxed, in the session's directory, bounded by `timeout`.
///
/// Authoring one requires `unrestricted`, which is the same level at which `execute_command` runs
/// arbitrary unsandboxed commands, so a sandbox here would block the ordinary cases (`gh`, `curl`)
/// without raising the bar the agent must clear.
async fn run_shell_probe(
    command: &str,
    timeout: Duration,
    cwd: Option<&std::path::Path>,
) -> Result<ProbeOutcome, String> {
    let mut builder = gate_command_builder(command);
    // The creating session's directory, not the host process's. A gate is almost always written by
    // the model right after verifying the same command through `execute_command`, which runs in the
    // session's cwd -- so a gate that runs anywhere else silently stops matching the command the
    // model tested. Under a `meka serve` systemd unit the process cwd is `/`, where a repo-relative
    // `gh pr checks` exits non-zero with empty stdout, and a `changed` gate then latches onto
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

    let stdout = String::from_utf8_lossy(&output.stdout);

    // A non-zero exit is reported, not refused.
    //
    // The failure this exists for is a watcher that breaks silently: an expired token has `gh` exit
    // non-zero with empty stdout, the first evaluation stores `""` as the baseline, and every
    // evaluation after compares `"" == ""` and stays quiet forever. The line is `debug!`, so `-vv`
    // is what surfaces it; it cannot be `warn!` for the reason immediately below, which is that a
    // non-zero exit is the *normal* state of a large class of correct gates.
    //
    // Refusing to produce output would not: for a large class of perfectly good gates, a non-zero
    // exit *is* the signal. `diff -q a b` and `git diff --exit-code` exit 1 exactly when there is a
    // difference; `grep ERROR log` exits 1 through the whole quiet period it is watching; `curl -f`
    // exits non-zero until the endpoint comes back. Treating any of those as broken would silence
    // the gate permanently, which is the bug this was meant to fix, pointed the other way. The
    // `succeeded` flag carries the exit status to whichever predicate asked for it instead.
    if !output.status.success() {
        let stderr = truncate_gate_output(&String::from_utf8_lossy(&output.stderr));
        tracing::debug!(
            "gate command exited with {}{}; its output still stands, since a non-zero exit is how \
             several common gates signal a change",
            output.status,
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {}", stderr)
            }
        );
    }

    Ok(ProbeOutcome::new(&stdout, None, output.status.success()))
}

/// Decide whether a probe's result means "fire".
///
/// Pure, so every predicate is testable without spawning anything.
fn apply_predicate(
    predicate: &GatePredicate,
    probe: &ProbeOutcome,
    last_output: Option<&str>,
) -> Result<GateOutcome, String> {
    // A first evaluation has no baseline, so "changed" is the honest answer. It also means a
    // freshly created watcher proves itself immediately instead of staying silent until
    // something happens, which is when a typo in the probe would otherwise surface.
    let changed_from = |current: &str| last_output != Some(current);

    match predicate {
        GatePredicate::Changed => Ok(GateOutcome {
            fired: changed_from(&probe.text),
            output: probe.text.clone(),
            baseline: probe.text.clone(),
        }),
        GatePredicate::Succeeded => Ok(GateOutcome {
            fired: probe.succeeded,
            output: probe.text.clone(),
            baseline: probe.text.clone(),
        }),
        GatePredicate::Matches { pattern } => {
            // A pattern that no longer compiles cannot be an error here: `evaluate_gate` reserves
            // `Err` for a probe that broke, and this one ran fine. It is refused at creation, so
            // reaching this means a hand-edited row; declining is the safe direction and the
            // warning says which job to fix.
            let fired = match regex::Regex::new(pattern) {
                Ok(regex) => regex.is_match(&probe.text),
                Err(error) => {
                    tracing::warn!("gate pattern /{}/ does not compile: {}", pattern, error);
                    false
                }
            };
            Ok(GateOutcome {
                fired,
                output: probe.text.clone(),
                baseline: probe.text.clone(),
            })
        }
        GatePredicate::At { pointer, is } => {
            let value = match pointed_at(probe, pointer) {
                Pointed::Found(value) => Some(value),
                Pointed::Absent => None,
                // An `Err`, like a probe that could not be spawned: the pointer describes a shape
                // this result does not have, so no predicate over it has an honest answer. The
                // caller declines the occurrence and warns, naming the job.
                Pointed::NotADocument => {
                    return Err(format!(
                        "gate points at `{}` but the probe did not return JSON: {}",
                        pointer,
                        elide_for_message(&probe.text)
                    ));
                }
            };
            // Serialised rather than compared as a `Value` because the baseline has to survive a
            // round trip through a TEXT column, and canonically because `serde_json` is built with
            // `preserve_order`: a `Value`'s object keys come back in the order the *input* had
            // them, so a server that emits the same object with its keys in a different order
            // renders as a different string. That is precisely the flap `at` exists to prevent,
            // arriving through the door left open for it. Arrays keep their order, which is part of
            // the value rather than an artefact of how it was written.
            let rendered = value
                .as_ref()
                .map(|value| truncate_gate_output(&canonical_json(value).to_string()))
                .unwrap_or_default();
            let fired = match is {
                PointerTest::NotEmpty => value.as_ref().is_some_and(json_is_non_empty),
                PointerTest::Empty => !value.as_ref().is_some_and(json_is_non_empty),
                PointerTest::Changed => changed_from(&rendered),
            };
            Ok(GateOutcome {
                fired,
                // The turn still sees the whole result: the pointer narrows what is *judged*, not
                // what the model is told, and the surrounding fields are usually the context that
                // makes the fire worth reading.
                output: probe.text.clone(),
                baseline: rendered,
            })
        }
    }
}

/// One line of a probe's result, short enough to sit inside a warning.
fn elide_for_message(text: &str) -> String {
    const LIMIT: usize = 120;
    let first = text.trim().lines().next().unwrap_or_default();
    match first.char_indices().nth(LIMIT) {
        Some((cut, _)) => format!("{}…", &first[..cut]),
        None => first.to_string(),
    }
}

/// What resolving a JSON pointer against a probe's result found.
///
/// Three outcomes, not two, because the two failures mean opposite things. A document that parsed
/// and simply lacks the field is an *answer*: an API that omits `chats` when there are none is
/// saying there are none. A result that is not a JSON document at all is a *broken probe*: nothing
/// was measured, so there is nothing to conclude.
///
/// Collapsing them cost real turns. `{"at": "/chats", "is": "empty"}` reads a missing value as
/// empty and fires, so a server that started returning an error string or prose fired the job on
/// every interval, indefinitely -- the exact expense the pointer predicate exists to avoid, aimed
/// the other way. `not-empty` and `changed` fail toward silence, which is survivable; `empty` was
/// alone in failing toward spending.
enum Pointed {
    /// The document parsed and the pointer resolved to this.
    Found(serde_json::Value),
    /// The document parsed; the pointer names nothing in it.
    Absent,
    /// The probe's result is not a JSON document, so the pointer means nothing here.
    NotADocument,
}

/// Resolve a JSON pointer against a probe's result.
///
/// Prefers the structured value, and falls back to parsing the text. The fallback is what makes a
/// pointer usable against the many MCP servers that return JSON as their text content and set no
/// `structuredContent`; there is no fence in that case, so the parse is unambiguous.
fn pointed_at(probe: &ProbeOutcome, pointer: &str) -> Pointed {
    let document = match &probe.structured {
        Some(structured) => std::borrow::Cow::Borrowed(structured),
        None => match serde_json::from_str::<serde_json::Value>(probe.text.trim()) {
            Ok(parsed) => std::borrow::Cow::Owned(parsed),
            Err(_) => return Pointed::NotADocument,
        },
    };
    match document.pointer(pointer) {
        Some(value) => Pointed::Found(value.clone()),
        None => Pointed::Absent,
    }
}

/// The same value with every object's keys in a fixed order, so equal documents render alike.
///
/// See the call site: `preserve_order` makes a `Value` remember its input's key order, which makes
/// a string comparison sensitive to something no watcher means to watch.
fn canonical_json(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(fields) => {
            let mut sorted: Vec<(&String, &serde_json::Value)> = fields.iter().collect();
            sorted.sort_by(|left, right| left.0.cmp(right.0));
            serde_json::Value::Object(
                sorted
                    .into_iter()
                    .map(|(key, value)| (key.clone(), canonical_json(value)))
                    .collect(),
            )
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(canonical_json).collect())
        }
        scalar => scalar.clone(),
    }
}

/// Whether a pointed-at value counts as "there is something here".
///
/// Containers go by length and scalars by presence, because those are the two ways a watched thing
/// reads as absent: an empty `chats` array, or a `null` field.
fn json_is_non_empty(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::Array(items) => !items.is_empty(),
        serde_json::Value::Object(entries) => !entries.is_empty(),
        serde_json::Value::String(text) => !text.is_empty(),
        serde_json::Value::Bool(_) | serde_json::Value::Number(_) => true,
    }
}

/// Build the platform's shell invocation for a gate, mirroring what `execute_command` does on its
/// unsandboxed path (`crate::tools::shell`).
fn gate_command_builder(command: &str) -> tokio::process::Command {
    #[cfg(windows)]
    {
        // Same UTF-8 prelude the shell tool uses: PowerShell 5.1 otherwise emits the legacy console
        // code page and non-ASCII output comes back as `?`, which would make a `changed` gate
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

/// Trim and cap a probe's result. Trimming matters for correctness, not tidiness: most commands
/// emit a trailing newline, and comparing untrimmed output would be fine, but a command whose
/// trailing whitespace varies run to run would fire a `changed` gate forever.
pub(crate) fn truncate_gate_output(raw: &str) -> String {
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
/// between them is where an occurrence can be lost. `prepare` leases the occurrence before handing
/// the wakeup over, so the row is untouched while the host works; a host that turns out to be
/// unable to run the job says so, and the lease is released rather than completed.
///
/// It used to be the other way round: `prepare` *consumed* the job -- stamped it and advanced its
/// schedule, or for a one-shot deleted the row -- before the host was asked, so that a prompt which
/// crashed the process could not re-fire forever. The handback then had to put something back,
/// which for a one-shot meant an `INSERT` that could not tell a cancellation from its own delete.
/// [`MAX_CLAIM_ATTEMPTS`] took over the crash protection, and this variant went back to meaning
/// only what it says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FireOutcome {
    /// The turn ran, or failed in a way that re-running would not fix.
    Ran,
    /// This host could not take the job and another one should. The concrete case is `meka serve`
    /// finding the session's file lock held by a REPL: that REPL has its own watcher and will run
    /// the job itself, so the occurrence is restored rather than burnt.
    Deferred,
    /// This host owns the job and could not run it, and trying again immediately would not help.
    ///
    /// The claim is left to expire rather than released or completed, which is the treatment a
    /// panicking turn already gets and for the same reason: releasing puts `next_fire_at` back in
    /// the past, so the occurrence comes due on the very next sweep and a gated job re-runs its
    /// probe every `poll_interval`; completing spends the occurrence, which for a one-shot deletes
    /// the row and loses the job outright. Waiting out `claim_lease` retries at a cadence a blip
    /// survives and a persistent fault does not, and `MAX_CLAIM_ATTEMPTS` still parks it in the
    /// end.
    ///
    /// The concrete case is a session whose recorded provider profile no longer resolves: running
    /// the turn would bill an account the row does not name, and the cause may be a configuration
    /// error the user has yet to fix or a transient `SQLITE_BUSY` from another meka process.
    Unrunnable,
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
/// admitted everything would run every gated job's probe -- a shell command, or a call to someone
/// else's server -- on every tick for sessions it could never serve.
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
        // declining would pay a gate evaluation and a lease round trip to arrive at the same
        // place.
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
        // `warn!`, not `?`. The same treatment `complete_claim` was given one level
        // down, and for the reason recorded there: propagating aborted the whole sweep, so one
        // transient `SQLITE_BUSY` skipped every *other* job due in the same tick. A job that was
        // never claimed comes back next tick, so the cost of continuing is nothing.
        let prepared = match prepare(session_manager, config, job, now).await {
            Ok(prepared) => prepared,
            Err(error) => {
                // Deliberately not promising the occurrence is intact. Everything before the claim
                // leaves the job untouched, and that is the common case -- but the one error that
                // can arrive *after* it is a one-shot's restore failing, and there the row is
                // already gone. Saying "will be reconsidered" would then be the opposite of what
                // happened, in the one case a reader most needs to know about.
                tracing::warn!(
                    "could not prepare job {}: {}. If it was claimed first, that occurrence is \
                     spent; otherwise it is untouched and the next tick reconsiders it",
                    original.short_id(),
                    error
                );
                continue;
            }
        };
        if let Some((claim, wakeup)) = prepared {
            // The callback is host code and can panic: a turn that blows up must not take the lease
            // with it for a whole `claim_lease`, nor end the sweep before the jobs behind it.
            // Caught here rather than only at the sweep boundary so the occurrence goes
            // back immediately, and released *without* forgiving the attempt, so a
            // prompt that does this every time climbs to `MAX_CLAIM_ATTEMPTS` and parks
            // instead of looping.
            let outcome =
                futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(fire(wakeup))).await;
            let outcome = match outcome {
                Ok(outcome) => outcome,
                // The lease is kept and left to expire, which is the retry.
                //
                // Releasing it here was the obvious thing and produced the failure the ceiling
                // exists to prevent, by a shorter route than the one it was written for. The row
                // stays due, so the next sweep re-delivers, and since every claim raises
                // `attempts` a prompt that panics reliably is parked after three of them: half a
                // minute at the default `poll_interval`. A recurring job is never retired by
                // `missed_grace`, so it was then dead until a person noticed -- and a panic from a
                // transient condition would kill a healthy job just as fast.
                //
                // Waiting out `claim_lease` gives the same three attempts an hour apart, which a
                // blip survives and a genuinely broken prompt does not. This is the same rule the
                // gate-probe path follows, and there is now only one of it.
                Err(_) => {
                    tracing::warn!(
                        "the turn for job {} panicked; its claim is left to expire, so the retry \
                         waits out [schedule].claim_lease",
                        original.short_id()
                    );
                    continue;
                }
            };
            if outcome == FireOutcome::Unrunnable {
                // Neither released nor completed; see the variant. Same handling as the panicking
                // turn above, and for the same reason: the retry is paced by `claim_lease` rather
                // than by `poll_interval`, so a transient cause survives and a persistent one is
                // parked by `MAX_CLAIM_ATTEMPTS` instead of spinning.
                tracing::warn!(
                    "job {} could not be run by this host; its claim is left to expire, so the \
                     retry waits out [schedule].claim_lease",
                    original.short_id()
                );
                continue;
            }
            if outcome == FireOutcome::Deferred {
                tracing::debug!("job {} deferred; releasing the lease", original.short_id());
                // Also `warn!`: propagating would take the rest of the sweep with it, and a lease
                // that is not released expires on its own, so the cost of continuing is a delay
                // rather than a loss. This is the whole gain of leasing over consuming: the
                // failure mode of the handback is now "later" instead of "never".
                if let Err(error) = store.release_claim(&original.id, &claim.owner).await {
                    tracing::warn!(
                        "job {} was deferred but its lease could not be released: {}. It will be \
                         reconsidered once the lease expires",
                        original.short_id(),
                        error
                    );
                }
                // A `false` here needs nothing said: the lease was already gone, and the
                // occurrence this host declined to run is open for whoever holds it now, which is
                // the outcome a deferral wants anyway.
            } else {
                // Delivered, so the occurrence is spent: advance a job that lives on, retire one
                // whose moment has passed. Written after the turn rather than before it, which is
                // what the attempt counter buys.
                match store
                    .complete_claim(
                        &original.id,
                        &claim.owner,
                        claim.next_fire_at,
                        Some(now),
                        claim.gate_baseline.as_deref(),
                    )
                    .await
                {
                    Ok(ClaimClosed::Yes) => {}
                    // Not a problem, and not silent either: a job that fires and then cancels
                    // itself is an ordinary shape, and the turn that did it should not look like
                    // a fault in the log.
                    Ok(ClaimClosed::RowGone) => tracing::debug!(
                        "job {} ran and its row was removed during the turn, so there was no \
                         occurrence left to close",
                        original.short_id()
                    ),
                    // The same outcome as the `Err` below, reached silently: the turn ran, and the
                    // occurrence it belonged to is still open because the lease expired under it.
                    // Worth its own sentence because the remedy differs -- an error is a database
                    // problem, this is a `claim_lease` shorter than a turn takes.
                    Ok(ClaimClosed::LeaseLost) => tracing::warn!(
                        "job {} ran, but its lease had already expired, so the occurrence stayed \
                         open and may be delivered again. Raise [schedule].claim_lease past how \
                         long this job's turn takes",
                        original.short_id()
                    ),
                    Err(error) => tracing::warn!(
                        "job {} ran but its occurrence could not be closed: {}. The lease expires \
                         on its own and the job is retried, which may deliver it twice",
                        original.short_id(),
                        error
                    ),
                }
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
    std::sync::Mutex<std::collections::HashMap<String, String>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// True the first time a job is held back for *this* reason, false while that reason persists.
///
/// Keyed by job and by reason, not by job alone. The two refusals are different facts with
/// different remedies -- "the session is at `none`" and "this gate needs `unrestricted`" -- and a
/// job that moves between them has changed in a way the operator acts on. Keyed by job alone the
/// second condition arrived silently, because the entry was already there: dropping a session from
/// `read` to `none` said nothing at all, and raising it back to `read` said nothing either.
///
/// Entries are dropped when the job is authorised again *or* when it stops being a job meka can
/// see, so a cancelled-while-declined job does not sit here for the life of the process. It is one
/// short string per held-back job, which is small, but a long-lived `meka serve` has no other bound
/// on it.
fn declined_for_permission_first_time(job_id: &str, reason: &str) -> bool {
    let mut held = match PERMISSION_DECLINED.lock() {
        Ok(held) => held,
        Err(poisoned) => poisoned.into_inner(),
    };
    match held.get(job_id) {
        Some(previous) if previous == reason => false,
        _ => {
            held.insert(job_id.to_string(), reason.to_string());
            true
        }
    }
}

/// What a job's row said when a probe failure was counted against it.
///
/// Neither half is written by the failing path -- it persists no `fired_at` and no baseline,
/// deliberately -- so either of them moving is proof that *something else* evaluated this gate
/// afterwards and got an answer. See [`standing_probe_failure`].
type ProbeWitness = (Option<DateTime<Utc>>, Option<String>);

/// The witness for `job` as its row stands now.
fn probe_witness(job: &ScheduledJob) -> ProbeWitness {
    (
        job.last_fired_at,
        job.gate.as_ref().and_then(|gate| gate.last_output.clone()),
    )
}

/// Consecutive failed probe evaluations per job, with the last reason and the row as it stood.
///
/// In memory rather than on the row, which is a real limitation and the right trade. Persisting it
/// would mean a schema change and a write on every failed evaluation, to report a condition that a
/// restart re-establishes within one poll interval. The cost is that the reporting surface and the
/// scheduler have to be the same process to agree: they are for the REPL, ACP and `meka serve`,
/// which are the three that both run jobs and render `[Scheduled]`. `meka schedule list` is a
/// separate process and sees nothing here, which is the same thing it already does with tool gates
/// it cannot resolve.
///
/// The [`ProbeWitness`] is what keeps that from becoming a *wrong* answer rather than a missing
/// one. Only the host that wins `claim_occurrence` evaluates, and which host that is, is a race
/// between their tickers; a host that recorded two failures and then stopped winning would go on
/// telling its resident session's model that a job firing every hour is dead, forever, because
/// nothing else in this process ever re-enters the counting path.
static PROBE_FAILURES: std::sync::LazyLock<std::sync::Mutex<ProbeFailures>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// How many consecutive failures, the last one's reason, and the row they were counted against.
type ProbeFailures = std::collections::HashMap<String, (u32, String, ProbeWitness)>;

/// How many consecutive failures before a broken probe is reported as a standing condition.
///
/// Two, not one. A single failure is as often a blip as a break -- a server restarting, a network
/// blip, a command losing a race -- and the marker says "this job is not firing", which is a
/// statement about a state rather than an event. A watcher that recovers on its next evaluation
/// never earns one.
///
/// Two *evaluations*, not two ticks, and the difference grew teeth when a failed probe stopped
/// being retried every `poll_interval`. Evaluations are now one occurrence apart for a recurring
/// job and one `claim_lease` apart for a job with no next occurrence, so the marker arrives after
/// two periods rather than twenty seconds: twelve hours for a `6h` job, two hours for a one-shot at
/// the default lease. That is the cost of not re-probing a broken gate at tick cadence, and it is
/// the right way round -- the alternative spent real work to reach the same conclusion sooner --
/// but it does mean this constant no longer implies anything about wall-clock latency.
const PROBE_FAILURES_BEFORE_REPORTING: u32 = 2;

/// Count one failed evaluation against a job and return the running total.
fn record_probe_failure(job: &ScheduledJob, error: &str) -> u32 {
    let mut held = match PROBE_FAILURES.lock() {
        Ok(held) => held,
        Err(poisoned) => poisoned.into_inner(),
    };
    let entry = held
        .entry(job.id.clone())
        .or_insert_with(|| (0, String::new(), (None, None)));
    entry.0 = entry.0.saturating_add(1);
    entry.1 = error.to_string();
    entry.2 = probe_witness(job);
    entry.0
}

/// What this process knows about a job's recent probe failures: how many, and why the last one
/// failed.
fn probe_failure(job_id: &str) -> Option<(u32, String)> {
    let held = match PROBE_FAILURES.lock() {
        Ok(held) => held,
        Err(poisoned) => poisoned.into_inner(),
    };
    held.get(job_id)
        .map(|(failures, error, _)| (*failures, error.clone()))
}

/// Forget a job's probe failures, on an evaluation that worked or on the job going away.
fn clear_probe_failure(job_id: &str) {
    match PROBE_FAILURES.lock() {
        Ok(mut held) => held.remove(job_id),
        Err(poisoned) => poisoned.into_inner().remove(job_id),
    };
}

/// Forget a job's held-back state, so the next withdrawal is announced again.
fn clear_permission_decline(job_id: &str) {
    match PERMISSION_DECLINED.lock() {
        Ok(mut held) => held.remove(job_id),
        Err(poisoned) => poisoned.into_inner().remove(job_id),
    };
}

/// The lease this host holds on one occurrence.
///
/// Carried out of [`prepare`] so the host that turns out to be unable to run the job hands back
/// exactly what it took, and so every write that follows is scoped to the claim *this process* won
/// rather than to the job id alone. A stale writer whose lease has since expired and been taken by
/// someone else changes nothing.
#[derive(Debug, Clone)]
pub(crate) struct Claim {
    /// Proof of ownership, matched against `scheduled_jobs.claimed_by`.
    owner: String,
    /// Where the schedule goes once the turn is delivered: `Some` advances a job that lives on,
    /// `None` retires one whose moment is spent.
    next_fire_at: Option<DateTime<Utc>>,
    /// What the gate saw, to be recorded when the occurrence is disposed of.
    ///
    /// Carried rather than written as soon as the gate returns, because a baseline is only true
    /// once the occurrence it belongs to is finished with. A host that evaluates, decides to fire
    /// and then cannot run the turn hands the occurrence back, and if it had already advanced the
    /// baseline the next host would compare the new value against itself, see no change, and never
    /// fire: the change would be swallowed by the handback that was supposed to preserve it.
    gate_baseline: Option<String>,
}

/// How many undelivered claims a job may accumulate before it stops being retried.
///
/// A lease brings back a hazard that spending the occurrence up front used to prevent: a prompt
/// that reliably kills the process is claimed again every time its lease expires, forever. Counting
/// the claims that ended in neither a delivery nor a handback identifies exactly that job, and this
/// is where it is parked: still listed, still cancellable, reported as held on every surface, and
/// no longer able to take the daemon down with it. Three, because two is within the range of
/// ordinary bad luck -- a deploy during a fire, then a machine restart -- and a fourth attempt on
/// something that has failed three times is not going to be the one that works.
///
/// It also bounds the one shape the "advance on a failed probe" rule cannot reach: a job with no
/// next occurrence, which in practice means a one-shot. There the lease is held rather than
/// released, so the retry waits out `claim_lease` instead of coming round on the next tick -- and
/// this is what stops that from going on until the grace period closes. Three attempts an hour
/// apart is a budget a transient outage survives and a broken gate does not; three ten-second ones,
/// which is what releasing the lease would have given, is neither.
pub(crate) const MAX_CLAIM_ATTEMPTS: u32 = 3;

/// A token identifying this process to the claim column, for the life of the process.
///
/// Per process rather than per sweep: the point is to tell *my* lease from someone else's, and a
/// value that changed between the claim and the write scoped to it would defeat both. Random rather
/// than derived from the pid, because a pid is reused and a reused pid would let a fresh process
/// finish a dead one's claim.
static SCHEDULER_OWNER: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| uuid::Uuid::new_v4().to_string());

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
        if store
            .retire_unclaimed(&job.id, job.next_fire_at, now)
            .await?
        {
            // One of the two doors out of the table that do not go through
            // `delete_scheduled_job`; the other is `complete_claim` retiring a job whose moment is
            // spent, which is reached only after the clears below have already run. Without these
            // the ledgers keyed by job id outlive the job, which a held one-shot reaches
            // routinely: it survives every sweep until its grace window closes, and this is where
            // it ends.
            clear_permission_decline(&job.id);
            clear_probe_failure(&job.id);
            tracing::warn!(
                "dropping one-shot job {}: due {} ago, past the missed-job grace period",
                job.short_id(),
                format_late(late_by)
            );
        }
        return Ok(None);
    }

    // One lookup, for every job rather than only gated ones, serving the working directory a gate
    // runs in and the live level both the checks below need.
    //
    // It used to be skipped for an ungated job, on the grounds that such a job had no authority to
    // re-check. That was wrong about `none`: an ungated job kept firing there, waking a model that
    // could read nothing and act on nothing. The query costs one row read per job that is actually
    // due, against a model turn, so paying it unconditionally is not a trade worth thinking about.
    let session = match session_manager.session_info(job.session_id).await {
        Ok(info) => info,
        Err(error) => {
            // Not silently "no session". A failed lookup means the level cannot be confirmed, and
            // the checks below have to fail closed on that rather than fall back to the recorded
            // value.
            tracing::warn!(
                "could not read session {} while preparing job {}: {}",
                job.session_id,
                job.short_id(),
                error
            );
            None
        }
    };
    let gate_cwd = session.as_ref().and_then(|info| info.cwd.clone());

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
        session.as_ref().and_then(|info| info.permission.as_deref()),
        &format_args!("session {}", job.session_id),
    )
    .filter(|level| config.enabled_permissions.is_enabled(*level))
    .unwrap_or(config.host_permission);

    let coalesced = occurrences_between(&job.schedule, job.next_fire_at, now);
    // `Some` only for a job that lives on. A one-shot's moment is spent, and a cron pattern with
    // nothing left in range has no future to be scheduled for; both are retired by
    // `complete_claim` once the turn is delivered, which is the only place the scheduler deletes a
    // job.
    //
    // Derived before the claim because a refusal needs it too: it is what spends a recurring job's
    // occurrence without evaluating anything.
    let next_fire_at = job
        .schedule
        .next_after_delivering(job.next_fire_at, now)
        .filter(|_| recurring);

    // Spend the occurrence of a job refused before its claim was taken.
    //
    // A recurring job advances, which is the documented rule and the same thing a gate that ran and
    // said no does: without it a held job sits permanently due and then reports a month-long
    // backlog the moment it is authorised again.
    //
    // A one-shot is left completely alone, and that asymmetry is the reason this exists rather than
    // the claim being taken first. It is cheaper to refuse before leasing than to lease and hand
    // back, and under the design this replaced it was not merely cheaper but necessary: claiming a
    // one-shot was a `DELETE`, so a refusal that followed one had to re-`INSERT`, and an `INSERT`
    // cannot tell "I deleted this a moment ago" from "the user cancelled it in between". Leasing
    // removed that hazard; every refusal that needs nothing but a row read is still made here,
    // because there is no reason to pay for a lease to reach the same answer.
    async fn decline_before_claiming(
        store: &ScheduleStore,
        job: &ScheduledJob,
        now: DateTime<Utc>,
        next_fire_at: Option<DateTime<Utc>>,
    ) -> crate::error::Result<()> {
        // Whether *this* host won the advance does not matter: either way the occurrence has moved
        // off the value every host read, and no host will deliver it.
        if let Some(next) = next_fire_at {
            store
                .advance_unclaimed(&job.id, job.next_fire_at, now, next)
                .await?;
        }
        Ok(())
    }

    // A job that has crashed its host repeatedly is parked rather than retried again.
    //
    // Claiming is a lease now, so nothing else stops a prompt that reliably kills the process from
    // being picked up on every expiry, forever. The row stays exactly where it is: listed,
    // cancellable, and reported as held on every surface, because destroying a user's job over a
    // failure meka cannot diagnose would be worse than leaving it visible and inert.
    if job.attempts >= MAX_CLAIM_ATTEMPTS {
        if declined_for_permission_first_time(&job.id, "crashed") {
            tracing::warn!(
                "job {} not fired: {}",
                job.short_id(),
                job_withheld_reason(&job, live_permission, config.gate_tools.as_deref())
                    .unwrap_or_else(|| "it has been parked".to_string()),
            );
        }
        return Ok(None);
    }

    // Before the gate, and regardless of whether there is one. At `none` nothing the turn could
    // reach is dispatchable: it reads nothing, changes nothing, and `schedule_cancel` is refused
    // too, so it cannot stop itself being woken again. Registration does not depend on the level --
    // the model is shown the job and offered the tool, and only refused when it reaches for one --
    // so the turn is left able to describe its predicament and unable to act on it. Firing anyway
    // spends tokens on that, every interval, until an operator notices.
    if !live_permission.allows_unattended_work() {
        if declined_for_permission_first_time(
            &job.id,
            &format!("unattended-work:{}", live_permission),
        ) {
            tracing::warn!(
                "job {} not fired: the session is at {}, where no tool is executable, so the turn \
                 could neither act nor cancel the job. Raise the session to restore it",
                job.short_id(),
                live_permission,
            );
        } else {
            tracing::debug!(
                "job {} still not fired: the session remains at {}",
                job.short_id(),
                live_permission,
            );
        }
        decline_before_claiming(&store, &job, now, next_fire_at).await?;
        return Ok(None);
    }

    // The same predicate the two creation doors use, asked again here against both the recorded
    // level and the live one.
    //
    // Checking only the recorded value was a tautology: the row is written by a door that already
    // demanded the level, and nothing ever updates the column, so the recorded value always
    // satisfies whatever admitted it. The comparison could not fail, and the case it was written
    // for -- the session cycles down to `read`, or a `meka serve --permission read` restarts and
    // inherits the row -- went unnoticed. The live level is what makes the withdrawal real; the
    // recorded one still matters because a hand-edited or unparseable `gate_permission` decodes as
    // `Permission::None` and must stay refused.
    //
    // Going through `gate_probe_is_authorised` rather than re-deriving the rule is what keeps the
    // doors in agreement. Asking `allows_unattended_shell` here regardless of probe kind accepted
    // every tool gate at creation and then declined it forever at fire time, with a message about a
    // shell command the job did not have: the headline case (`mcp__…__unseen` at `read`) never
    // called its probe once.
    //
    // The occurrence is declined, exactly as a gate that ran and said no is declined. A gate is the
    // condition on the job, so a gate that could not be evaluated has not passed, and firing anyway
    // converts a conditional job into an unconditional one. The shape that makes this concrete is
    // `every = "1m"` with a `changed` gate: firing it unconditionally turns a near-silent job into
    // a turn a minute, which is the opposite of what the row asks for and expensive besides.
    if let Some(gate) = &job.gate {
        let tools = config.gate_tools.as_deref();
        if let Some((refusal, level)) = gate_withheld_reason(gate, live_permission, tools) {
            // Said once per decline, not once per evaluation. The condition is a standing state
            // rather than an event: a session left below the bar with an `every = "1m"` job wrote
            // this line every minute for as long as it stayed there, which buries the log it is
            // supposed to be the signal in. The id is cleared the moment the gate is authorised
            // again, so a later withdrawal is announced afresh.
            let explained = refusal.explain(&gate.probe, level);
            if declined_for_permission_first_time(&job.id, &explained) {
                // `gate_withheld_reason` reports the live level first because it is the one an
                // operator can act on, so a refusal carrying the *recorded* level instead means the
                // live level was fine: a hand-edited or damaged row, which no amount of cycling the
                // session will fix. Saying which of the two it was is the only way to tell those
                // apart from the log.
                if level == live_permission {
                    // Naming the level it was authorised at only helps when that level would still
                    // pass; otherwise it reads as a promise that restoring it is enough.
                    match gate_probe_is_authorised(&gate.probe, gate.permission, tools) {
                        Ok(()) => tracing::warn!(
                            "job {} not fired: {}. It was authorised at {}; raise the session back \
                             to restore it",
                            job.short_id(),
                            explained,
                            gate.permission,
                        ),
                        Err(_) => {
                            tracing::warn!("job {} not fired: {}", job.short_id(), explained)
                        }
                    }
                } else {
                    tracing::warn!(
                        "job {} not fired: {}, which is the level recorded when the gate was \
                         authorised",
                        job.short_id(),
                        explained,
                    );
                }
            } else {
                tracing::debug!(
                    "job {} still not fired: its gate is still unauthorised",
                    job.short_id(),
                );
            }
            decline_before_claiming(&store, &job, now, next_fire_at).await?;
            return Ok(None);
        }
    }
    // Nothing is holding this job back, so a later withdrawal is announced afresh rather than
    // swallowed by the once-per-decline suppression above. An ungated job passes through here too:
    // it is held by nothing, and the `none` floor above is the only thing that could have stopped
    // it.
    clear_permission_decline(&job.id);

    // Lease the occurrence before doing anything that can fail or hang.
    //
    // This is what arbitrates between hosts, which is why it is conditional. Every `meka serve`,
    // REPL and ACP session polls the same table, so one occurrence is in several hosts' due lists
    // at once; whoever takes the lease owns it, and the rest return here having neither
    // evaluated the gate nor spent the occurrence.
    //
    // The lease is taken *before* the gate runs and released or completed after, so the row is
    // never absent while this host is working. Claiming used to consume the row instead --
    // advancing a recurring job's `next_fire_at`, deleting a one-shot outright -- which is why
    // a refusal had to put something back, and why a crash between the claim and the turn lost
    // the occurrence, or for a one-shot the entire job.
    let mut claim = Claim {
        owner: SCHEDULER_OWNER.clone(),
        next_fire_at,
        gate_baseline: None,
    };
    if !store
        .claim_occurrence(
            &job.id,
            job.next_fire_at,
            &claim.owner,
            now,
            now + chrono::Duration::from_std(config.claim_lease).unwrap_or_else(|_| {
                // Only an out-of-range configured duration reaches this, and a lease that cannot be
                // expressed must not become one that never expires.
                chrono::Duration::hours(1)
            }),
        )
        .await?
    {
        tracing::debug!(
            "job {} was claimed for this occurrence by another host",
            job.short_id()
        );
        return Ok(None);
    }

    let gate_output = match &job.gate {
        None => None,
        Some(gate) => {
            match evaluate_gate(
                gate,
                config.gate_timeout,
                gate_cwd.as_deref(),
                config.gate_tools.as_deref(),
            )
            .await
            {
                Ok(outcome) => {
                    // An evaluation that produced an answer, whichever answer it was, ends any
                    // standing failure: the probe works.
                    clear_probe_failure(&job.id);
                    // Persist the new baseline even when it did not fire; that is exactly how a
                    // `changed` gate stops firing once it has seen the new value. A retired job has
                    // no row left to write to, and needs none -- it will not be
                    // evaluated again.
                    //
                    // `baseline`, not `output`: for a pointer predicate the two differ, and storing
                    // the whole result would put the moving field the pointer excludes back into
                    // the comparison, firing the gate every interval.
                    claim.gate_baseline = Some(outcome.baseline);
                    if !outcome.fired {
                        // The occurrence is spent: the condition was asked and said no. A recurring
                        // job moves to its next occurrence and a one-shot's moment has passed,
                        // which is the documented rule and the same thing
                        // `complete_claim` does after a turn. The only
                        // difference is that no turn ran.
                        tracing::debug!("gate for job {} declined to fire", job.short_id());
                        match store
                            .complete_claim(
                                &job.id,
                                &claim.owner,
                                claim.next_fire_at,
                                None,
                                claim.gate_baseline.as_deref(),
                            )
                            .await
                        {
                            Ok(ClaimClosed::Yes) => {}
                            Ok(ClaimClosed::RowGone) => tracing::debug!(
                                "job {}'s gate declined and its row was removed while the probe \
                                 ran, so there was no occurrence left to close",
                                job.short_id()
                            ),
                            // The lease was taken from under this host while the probe ran, so the
                            // baseline it just measured was not recorded. Said out loud because
                            // the visible symptom is a `changed` gate firing twice for one change,
                            // which reads as a flapping probe rather than a lost write.
                            Ok(ClaimClosed::LeaseLost) => tracing::warn!(
                                "job {}'s gate declined, but this host no longer held the lease, \
                                 so the occurrence stayed open. It may be evaluated again",
                                job.short_id()
                            ),
                            Err(error) => tracing::warn!(
                                "job {} declined but its occurrence could not be closed: {}. The \
                                 lease expires on its own",
                                job.short_id(),
                                error
                            ),
                        }
                        return Ok(None);
                    }
                    Some(outcome.output)
                }
                Err(error) => {
                    // Loud on purpose. A watcher whose probe breaks produces the same silence as
                    // a watcher with nothing to report, and that is the failure
                    // most likely to go unnoticed for weeks.
                    //
                    // Counted as well as logged, so the *model* hears about it too once the
                    // condition is standing rather than momentary. The log alone reaches only
                    // whoever is reading it, and a scheduled job exists precisely because nobody
                    // is.
                    // The row as it stands *now* -- before this arm's own disposal, which writes
                    // neither half of the witness. See [`standing_probe_failure`].
                    let failures = record_probe_failure(&job, &error);
                    tracing::warn!(
                        "gate for job {} failed: {} (failure {})",
                        job.short_id(),
                        error,
                        failures
                    );
                    // The same disposal a refusal gets, and for the same reason: the condition was
                    // not answered, so this occurrence is over. A recurring job advances to its
                    // next one; a one-shot keeps its row, because its moment has not been spent on
                    // anything.
                    //
                    // Simply releasing the lease was the obvious thing and was wrong. It leaves
                    // `next_fire_at` where it was, so the row is due again on the very next sweep:
                    // a six-hour job whose server is down was re-probed every `poll_interval`
                    // rather than every six hours, and a probe that *hangs* burned the whole
                    // `gate_timeout` out of each sweep, delaying every job behind it. Under the
                    // old design the schedule had already been advanced to claim the job, so this
                    // was structurally impossible and nothing here had to think about it.
                    //
                    // `fired_at` and `gate_baseline` are both `None`. Nothing fired, and nothing
                    // was measured -- leaving the baseline alone is what makes the recovery
                    // correct, because the next successful evaluation then compares against the
                    // last value actually observed and reports the change that happened while the
                    // probe was broken.
                    match claim.next_fire_at {
                        Some(next) => match store
                            .complete_claim(&job.id, &claim.owner, Some(next), None, None)
                            .await
                        {
                            Ok(ClaimClosed::Yes) => {}
                            Ok(ClaimClosed::RowGone) => tracing::debug!(
                                "job {}'s gate failed and its row was removed while the probe \
                                 ran, so there was no occurrence left to close",
                                job.short_id()
                            ),
                            // The occurrence stayed open, so the next sweep probes again. Said out
                            // loud because it is the state this whole arm exists to avoid, and it
                            // is otherwise indistinguishable from a probe that is simply failing
                            // often.
                            Ok(ClaimClosed::LeaseLost) => tracing::warn!(
                                "job {}'s gate failed and this host no longer held the lease, so \
                                 the occurrence stayed open and will be probed again",
                                job.short_id()
                            ),
                            Err(error) => tracing::warn!(
                                "job {}'s gate failed and its occurrence could not be closed: {}. \
                                 The lease expires on its own",
                                job.short_id(),
                                error
                            ),
                        },
                        // Nothing at all: the lease is *kept*, and left to expire.
                        //
                        // This is the one case the advance above cannot reach -- a schedule with no
                        // next occurrence to move to, which in practice means a one-shot. Releasing
                        // the lease here makes the row due again on the very next sweep, so a probe
                        // that is down gets re-run every `poll_interval` and, since each claim
                        // raises `attempts`, the job is parked after three of them: half a minute
                        // at the default. An MCP server restarting anywhere near a one-shot's due
                        // time would silently destroy the reminder, which is a worse failure than
                        // the one the advance is here to prevent.
                        //
                        // A lease already means "not available until then", so holding it *is* the
                        // backoff, and it needs no new state to express. The job is retried once
                        // per `claim_lease` rather than once per tick, and because each of those
                        // retries is a fresh claim the attempt ceiling still applies -- three
                        // hourly attempts before parking rather than three ten-second ones, which
                        // is a budget a transient outage survives and a broken gate does not.
                        None => tracing::debug!(
                            "job {}'s gate failed and it has no next occurrence; holding the lease \
                             so the retry waits out [schedule].claim_lease",
                            job.short_id()
                        ),
                    }
                    return Ok(None);
                }
            }
        }
    };

    // The schedule is advanced by `run_due` once the turn has actually been delivered, not here.
    // Until then this host holds a lease and the row is untouched, so a crash costs a retry rather
    // than the occurrence.
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
        let gate_kind = job
            .gate
            .as_ref()
            .map(|gate| gate.probe.kind_str().to_string());
        let gate_spec = job.gate.as_ref().map(|gate| gate.spec());
        let gate_last_output = job.gate.as_ref().and_then(|gate| gate.last_output.clone());
        let gate_permission = job.gate.as_ref().map(|gate| gate.permission.to_string());
        let isolated = i64::from(job.isolated);
        let created_at = job.created_at.to_rfc3339();
        let last_fired_at = job.last_fired_at.map(|at| at.to_rfc3339());
        let next_fire_at = job.next_fire_at.to_rfc3339();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO scheduled_jobs (id, session_id, kind, spec, prompt, gate_kind, \
                     gate_spec, gate_last_output, gate_permission, isolated, created_at, \
                     last_fired_at, next_fire_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
                    rusqlite::params![
                        id,
                        session_id,
                        kind,
                        spec,
                        prompt,
                        gate_kind,
                        gate_spec,
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
            "SELECT id, session_id, kind, spec, prompt, gate_kind, gate_spec, \
             gate_last_output, gate_permission, isolated, created_at, last_fired_at, next_fire_at, \
             attempts FROM scheduled_jobs WHERE session_id = ?1 ORDER BY next_fire_at ASC"
                .to_string(),
            vec![session_id.to_string()],
        )
        .await
    }

    /// Every job in the database, soonest first. Backs `meka schedule list` and `meka schedule
    /// cancel`, which work from a job id and so cannot ask the caller which session to look in.
    pub async fn list_all_scheduled_jobs(&self) -> crate::error::Result<Vec<ScheduledJob>> {
        self.query_scheduled_jobs(
            "SELECT id, session_id, kind, spec, prompt, gate_kind, gate_spec, \
             gate_last_output, gate_permission, isolated, created_at, last_fired_at, next_fire_at, \
             attempts FROM scheduled_jobs ORDER BY next_fire_at ASC"
                .to_string(),
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
            format!(
                "SELECT id, session_id, kind, spec, prompt, gate_kind, gate_spec, \
                 gate_last_output, gate_permission, isolated, created_at, last_fired_at, \
                 next_fire_at, attempts FROM scheduled_jobs \
                 WHERE next_fire_at <= ?1 AND {} \
                 ORDER BY next_fire_at ASC",
                no_live_claim("?1")
            ),
            vec![now.to_rfc3339()],
        )
        .await
    }

    /// Shared row decoder. A row that fails to decode (hand-edited spec, a `kind` from a future
    /// version) is skipped with a warning rather than failing the whole query: one bad row must not
    /// stop every other job in the database from firing.
    async fn query_scheduled_jobs(
        &self,
        sql: String,
        params: Vec<String>,
    ) -> crate::error::Result<Vec<ScheduledJob>> {
        let rows: Vec<ScheduledJobRow> = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(&sql)?;
                let rows = statement
                    .query_map(rusqlite::params_from_iter(params.iter()), |row| {
                        Ok(ScheduledJobRow {
                            id: row.get(0)?,
                            session_id: row.get(1)?,
                            kind: row.get(2)?,
                            spec: row.get(3)?,
                            prompt: row.get(4)?,
                            gate_kind: row.get(5)?,
                            gate_spec: row.get(6)?,
                            gate_last_output: row.get(7)?,
                            gate_permission: row.get(8)?,
                            isolated: row.get::<_, i64>(9)? != 0,
                            created_at: row.get(10)?,
                            last_fired_at: row.get(11)?,
                            next_fire_at: row.get(12)?,
                            attempts: row.get::<_, i64>(13)?.max(0) as u32,
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
    /// nothing matched *or* when the row was gone by the time the delete ran.
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

        // `None` when the row was already gone, not `Some(id)`.
        //
        // The listing above and the `DELETE` below are two statements, and a scheduler sweep can
        // retire the row between them: a one-shot's occurrence retires it, and a session deleted
        // elsewhere takes its jobs with it through the foreign key. Reporting the id regardless
        // told the agent "Cancelled job abc12345" about a job this call did not cancel, which is
        // the same sentence it gets when it did -- and there is no way to tell them apart
        // afterwards, because both end with no such row.
        match self.delete_scheduled_job(&id).await? {
            true => Ok(Some(id)),
            false => Ok(None),
        }
    }

    /// Delete a job by exact id, without the prefix resolution [`Self::cancel_scheduled_job`] does.
    ///
    /// `true` when a row was actually removed. Callers that report an outcome to a person or to the
    /// model must not treat `false` as success: it means something else removed the job first, and
    /// saying "cancelled" then is a claim about work this call did not do.
    pub async fn delete_scheduled_job(&self, id: &str) -> crate::error::Result<bool> {
        // A job that no longer exists cannot be held back for permission, and the process-global
        // set is otherwise only cleared when a job is *authorised* again. A job cancelled while
        // declined therefore left its id there for the life of a `meka serve`. Clearing here keeps
        // the set bounded by the jobs that exist rather than by every job that ever did. This is
        // not the only `DELETE` -- `retire_unclaimed` and `complete_claim` have their own -- but
        // both of those clear the ledgers themselves, on the path that reaches them.
        clear_permission_decline(id);
        clear_probe_failure(id);
        let id = id.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "DELETE FROM scheduled_jobs WHERE id = ?1",
                    rusqlite::params![id],
                )
            })
            .await
            .map(|changed| changed == 1)
            .map_err(|error| {
                MekaError::Database(format!("failed to delete scheduled job: {}", error))
            })
    }

    /// Take this occurrence for this process, by leasing the row rather than consuming it.
    ///
    /// One shape for both kinds of schedule, and the reason the two used to differ is worth
    /// recording. A recurring job was claimed by advancing `next_fire_at`, which is a
    /// compare-and-swap: whoever moves the row off the value every host read owns the occurrence.
    /// A one-shot has no next occurrence to move to, so it was claimed by *deleting* the row, which
    /// makes the same arbitration work and is where every problem came from. Deletion overloads one
    /// piece of state with two facts -- "the user still wants this job" and "this occurrence is
    /// available" -- so a host that later had to hand the occurrence back could only re-`INSERT`,
    /// and an `INSERT` cannot tell "I deleted this a moment ago" from "the user cancelled it in
    /// between". A cancellation issued inside that window was silently undone, and a crash inside
    /// it destroyed the job outright, since nothing ever put the row back.
    ///
    /// A lease separates the facts. `claimed_by` says who is delivering this occurrence and
    /// `claimed_until` says how long that claim is good for; the row itself stays put, so a
    /// cancellation is an unconditional `DELETE` that always wins and a crash expires rather than
    /// erasing. Every write after this one is scoped to `claimed_by`, so a late writer whose lease
    /// has since been taken changes nothing.
    ///
    /// `occurrence` is the value this host read into its due list, so exactly one host can take a
    /// given occurrence. `attempts` counts claims that neither completed nor were handed back --
    /// a host that died, or a probe that could not answer and left its lease to expire -- and is
    /// what replaces the old protection of spending the occurrence up front; see
    /// [`ScheduledJob::attempts`] and [`MAX_CLAIM_ATTEMPTS`].
    pub(crate) async fn claim_occurrence(
        &self,
        id: &str,
        occurrence: chrono::DateTime<chrono::Utc>,
        owner: &str,
        now: chrono::DateTime<chrono::Utc>,
        until: chrono::DateTime<chrono::Utc>,
    ) -> crate::error::Result<bool> {
        let id = id.to_string();
        let occurrence = occurrence.to_rfc3339();
        let owner = owner.to_string();
        let now = now.to_rfc3339();
        let until = until.to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    &format!(
                        "UPDATE scheduled_jobs \
                         SET claimed_by = ?3, claimed_until = ?5, attempts = attempts + 1 \
                         WHERE id = ?1 AND {} AND {}",
                        SAME_OCCURRENCE,
                        no_live_claim("?4")
                    ),
                    rusqlite::params![id, occurrence, owner, now, until],
                )
            })
            .await
            .map(|changed| changed == 1)
            .map_err(|error| {
                MekaError::Database(format!("failed to claim scheduled job: {}", error))
            })
    }

    /// Hand the occurrence back: this host took it and will not deliver it after all.
    ///
    /// Scoped to `owner`, which is what makes it safe. The refusals that reach here -- a gate whose
    /// probe broke, a host that cannot run the turn -- all used to put a deleted row back, and
    /// could not distinguish that from undoing a cancellation. Releasing a lease can only
    /// affect a row this host still holds, so a job cancelled in the meantime stays cancelled
    /// and a job another host has since claimed is left alone.
    ///
    /// `attempts` is reset, because a claim the *host* handed back is not the failure
    /// [`MAX_CLAIM_ATTEMPTS`] is counting. That ceiling is about jobs that cannot be delivered;
    /// this is about a host that could not take one, which says nothing about the job.
    ///
    /// This is the only way a lease is given up early, and that is the whole disposal rule: a
    /// claim is either handed back by a host that declined the work, or left to expire. There used
    /// to be a second, non-forgiving release for a claim that ended without delivering -- a
    /// panicking turn, an unevaluable probe -- and it was a mistake in a way that is worth
    /// recording, because it looked like the careful option. Giving the occurrence back at once
    /// leaves the row due on the next sweep, so those retries came one `poll_interval` apart and
    /// the ceiling was reached in half a minute, killing jobs whose only problem was a blip.
    /// Expiry is already the mechanism for "try again later", so nothing needed to be added; a
    /// call needed to be removed.
    ///
    /// Matching nothing is ordinary and is not reported: the lease had already expired and someone
    /// else holds the occurrence, which for a host that was declining it anyway is the outcome it
    /// wanted. Hence `()` rather than a `bool` no caller reads -- the same reason
    /// [`Self::advance_unclaimed`] returns nothing, and a second instance of the same oversight,
    /// found by a mutation sweep rather than by reading.
    pub(crate) async fn release_claim(&self, id: &str, owner: &str) -> crate::error::Result<()> {
        let id = id.to_string();
        let owner = owner.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE scheduled_jobs SET claimed_by = NULL, claimed_until = NULL, \
                     attempts = 0 WHERE id = ?1 AND claimed_by = ?2",
                    rusqlite::params![id, owner],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to release a scheduled job: {}", error))
            })
    }

    /// Finish the occurrence: the turn was delivered, so advance the schedule or retire the job.
    ///
    /// `next_fire_at` is `Some` for a job that lives on and `None` for one whose moment is spent --
    /// a one-shot, or a cron pattern with nothing left in range -- which is the only place a
    /// scheduled job is deleted by the scheduler rather than by a person.
    ///
    /// Written *after* delivery, where the fire stamp used to be written before it. That
    /// ordering existed so a prompt which reliably crashed the process could not be re-selected on
    /// every restart, at the price of losing the occurrence to any crash at all. The lease's
    /// `attempts` counter takes over that job and does it better: a crash now costs a retry rather
    /// than the occurrence, and a job that crashes repeatedly is parked instead of looping.
    ///
    /// `fired_at` is `None` for an occurrence that was *considered* rather than delivered, which is
    /// what a gate saying no amounts to. `last_fired_at` means a turn happened: recording one for
    /// an evaluation would misreport the job in every listing and re-anchor an interval
    /// schedule on evaluations rather than on fires.
    ///
    /// See [`ClaimClosed`] for what the outcomes mean; only one of them is a problem.
    pub(crate) async fn complete_claim(
        &self,
        id: &str,
        owner: &str,
        next_fire_at: Option<chrono::DateTime<chrono::Utc>>,
        fired_at: Option<chrono::DateTime<chrono::Utc>>,
        gate_baseline: Option<&str>,
    ) -> crate::error::Result<ClaimClosed> {
        let id = id.to_string();
        let owner = owner.to_string();
        let fired_at = fired_at.map(|at| at.to_rfc3339());
        let next_fire_at = next_fire_at.map(|at| at.to_rfc3339());
        let gate_baseline = gate_baseline.map(str::to_string);
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                if match next_fire_at {
                    // `COALESCE` twice, so an evaluation leaves the previous fire time alone rather
                    // than clearing it, and an ungated job leaves the baseline alone rather than
                    // erasing a gate's memory.
                    Some(next) => connection.execute(
                        "UPDATE scheduled_jobs SET next_fire_at = ?3, \
                         last_fired_at = COALESCE(?4, last_fired_at), \
                         gate_last_output = COALESCE(?5, gate_last_output), \
                         claimed_by = NULL, claimed_until = NULL, attempts = 0 \
                         WHERE id = ?1 AND claimed_by = ?2",
                        rusqlite::params![id, owner, next, fired_at, gate_baseline],
                    ),
                    None => connection.execute(
                        "DELETE FROM scheduled_jobs WHERE id = ?1 AND claimed_by = ?2",
                        rusqlite::params![id, owner],
                    ),
                }? == 1
                {
                    return Ok(ClaimClosed::Yes);
                }
                // Both statements are scoped to this owner's lease, so a zero count says only "no
                // row carrying my claim" -- which has two causes that mean opposite things. Asked
                // here, on the failure path only, because the answer decides whether the caller
                // has a problem or has merely been cancelled.
                let still_there: bool = connection.query_row(
                    "SELECT EXISTS(SELECT 1 FROM scheduled_jobs WHERE id = ?1)",
                    rusqlite::params![id],
                    |row| row.get(0),
                )?;
                Ok(match still_there {
                    true => ClaimClosed::LeaseLost,
                    false => ClaimClosed::RowGone,
                })
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to complete a scheduled job: {}", error))
            })
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

    /// Drop a job nobody is delivering, for the one-shot whose moment passed while meka was not
    /// running. Returns whether this call is the one that removed it.
    ///
    /// Scoped to the occurrence so that several hosts noticing the same expired job produce one
    /// announcement rather than one each.
    pub(crate) async fn retire_unclaimed(
        &self,
        id: &str,
        occurrence: chrono::DateTime<chrono::Utc>,
        now: chrono::DateTime<chrono::Utc>,
    ) -> crate::error::Result<bool> {
        let id = id.to_string();
        let occurrence = occurrence.to_rfc3339();
        let now = now.to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    &format!(
                        "DELETE FROM scheduled_jobs WHERE id = ?1 AND {} AND {}",
                        SAME_OCCURRENCE,
                        no_live_claim("?3")
                    ),
                    rusqlite::params![id, occurrence, now],
                )
            })
            .await
            .map(|changed| changed == 1)
            .map_err(|error| {
                MekaError::Database(format!("failed to retire scheduled job: {}", error))
            })
    }

    /// Move a job to its next occurrence without claiming or delivering it, for a refusal made
    /// before any lease was taken.
    ///
    /// The occurrence is spent because the job was *considered*: that is the documented rule for a
    /// gate that says no, and a refusal is the same shape. `last_fired_at` is deliberately not
    /// written, because nothing fired.
    ///
    /// Matching nothing is ordinary here and is not reported: another host advanced the same
    /// occurrence first, or took it while this one was deciding, and either way the occurrence has
    /// moved off the value this host read. That is why this returns `()` where
    /// [`Self::retire_unclaimed`] returns `bool` -- there, whether *this* call was the one that
    /// removed the row decides who announces it.
    pub(crate) async fn advance_unclaimed(
        &self,
        id: &str,
        occurrence: chrono::DateTime<chrono::Utc>,
        now: chrono::DateTime<chrono::Utc>,
        next_fire_at: chrono::DateTime<chrono::Utc>,
    ) -> crate::error::Result<()> {
        let id = id.to_string();
        let occurrence = occurrence.to_rfc3339();
        let now = now.to_rfc3339();
        let next_fire_at = next_fire_at.to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    &format!(
                        "UPDATE scheduled_jobs SET next_fire_at = ?4 \
                         WHERE id = ?1 AND {} AND {}",
                        SAME_OCCURRENCE,
                        no_live_claim("?3")
                    ),
                    rusqlite::params![id, occurrence, now, next_fire_at],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to advance a scheduled job: {}", error))
            })
    }
}

/// What became of an attempt to close out an occurrence.
///
/// Three answers rather than two, because "the write matched nothing" has two causes that mean
/// opposite things and prescribe opposite remedies. Reporting the commoner one is how a healthy
/// job came to be told, on every fire, to raise a setting that had nothing to do with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClaimClosed {
    /// Written. The occurrence is spent.
    Yes,
    /// The row is gone: the job was cancelled, or its session deleted, while the turn ran. Nothing
    /// is wrong -- there is no occurrence left to close, and a cancellation is meant to win.
    /// `schedule_cancel` is offered to the model in the same breath as `schedule_create`, so a job
    /// that fires, decides it is finished and cancels itself lands here every time.
    RowGone,
    /// The row is there, under someone else's lease. This one is a problem: the occurrence is
    /// still open, so the turn that just ran may be delivered again by whoever holds it now.
    LeaseLost,
}

/// The lease half of every availability test: a row is takeable when nothing holds it, or the hold
/// has run out. `now` names the bound parameter carrying the current instant, because the four
/// queries that ask this number their parameters differently.
///
/// One definition because two are what a stale lease exploits. This clause and `claimed_by IS NULL`
/// look interchangeable and are not: nothing clears `claimed_by` except a release, a completion or
/// a fresh claim, so a host that dies holding a lease leaves it set for good. The due query and
/// [`ScheduleStore::claim_occurrence`] used the expiry; [`ScheduleStore::retire_unclaimed`] and
/// [`ScheduleStore::advance_unclaimed`] used the column, which are the two paths that move an
/// occurrence *without* taking a lease. A row with an expired lease was therefore handed out on
/// every sweep and was invisible to both, so a one-shot past its grace period was never retired,
/// never fired and never logged, and a recurring job refused before its claim never advanced.
fn no_live_claim(now: &str) -> String {
    format!(
        "(claimed_by IS NULL OR claimed_until IS NULL OR claimed_until < {})",
        now
    )
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
    gate_kind: Option<String>,
    gate_spec: Option<String>,
    gate_last_output: Option<String>,
    gate_permission: Option<String>,
    isolated: bool,
    created_at: String,
    last_fired_at: Option<String>,
    next_fire_at: String,
    attempts: u32,
}

impl ScheduledJobRow {
    fn decode(self) -> std::result::Result<ScheduledJob, String> {
        let parse_time =
            |text: &str| -> std::result::Result<chrono::DateTime<chrono::Utc>, String> {
                chrono::DateTime::parse_from_rfc3339(text)
                    .map(|at| at.with_timezone(&chrono::Utc))
                    .map_err(|error| format!("bad timestamp '{}': {}", text, error))
            };

        let gate = match (self.gate_kind, self.gate_spec) {
            (Some(kind), Some(spec)) => Some(Gate::from_stored(
                &kind,
                &spec,
                self.gate_last_output,
                // Every write path stores a level alongside the gate, so an absent or unparseable
                // one means a hand-edited or damaged row. Reading that as `Unrestricted` would
                // hand an arbitrary shell command the authority the column exists
                // to record, so it resolves to `None`: the gate is refused at fire
                // time and the user is told to recreate the job. Failing closed
                // costs one re-creation; failing open costs the guarantee.
                //
                // Through `parse_recorded_permission` like the five session-row readers, so the
                // *unreadable* case is heard rather than folded into the absent one. Without the
                // warning the only clue is a later message saying the gate was authorised at
                // `none` -- naming a level the job was never created at.
                crate::permission::parse_recorded_permission(
                    self.gate_permission.as_deref(),
                    &format_args!("the gate on job {}", self.id),
                )
                .unwrap_or(crate::permission::Permission::None),
            )?),
            // A half-written gate is a corrupt row, not a job without a gate: silently dropping the
            // condition would turn a watcher into an unconditional timer.
            (Some(_), None) | (None, Some(_)) => {
                return Err("gate_kind and gate_spec must both be set or both be null".into());
            }
            (None, None) => None,
        };

        Ok(ScheduledJob {
            attempts: self.attempts,
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

    fn gate(command: &str, predicate: GatePredicate, last_output: Option<&str>) -> Gate {
        Gate {
            probe: GateProbe::Shell {
                command: command.to_string(),
            },
            predicate,
            last_output: last_output.map(str::to_string),
            // The level every gate is created at. Tests that exercise a gate *running* need it; the
            // one that exercises a withdrawn authority overrides it explicitly.
            permission: crate::permission::Permission::Unrestricted,
        }
    }

    const GATE_BUDGET: Duration = Duration::from_secs(10);

    /// A probe result, without running anything. `apply_predicate` is pure, so every predicate can
    /// be exercised directly rather than through a command that has to produce the shape.
    /// `apply_predicate` for tests that expect an answer rather than a broken probe. The `Err`
    /// arm is exercised on its own, by the tests that hand it something that is not a document.
    fn judged(
        predicate: &GatePredicate,
        probe: &ProbeOutcome,
        last_output: Option<&str>,
    ) -> GateOutcome {
        apply_predicate(predicate, probe, last_output).expect("the predicate has an answer")
    }

    fn probed(text: &str, structured: Option<serde_json::Value>, succeeded: bool) -> ProbeOutcome {
        ProbeOutcome {
            text: text.to_string(),
            structured,
            succeeded,
        }
    }

    /// The regression this whole design exists to prevent.
    ///
    /// A structured result carrying anything self-moving is different on every call, so `changed`
    /// over the whole of it fires every single interval and spends exactly the turns a gate is
    /// meant to save. It looks like a working watcher right up until the bill arrives. Pointing at
    /// the field that matters is the only honest way to watch one, so this asserts both halves: the
    /// pointer stays quiet while only the timestamp moves, and the naive predicate over the same
    /// two results does not.
    #[test]
    fn a_pointer_ignores_a_sibling_that_moves_on_its_own() {
        let first = serde_json::json!({"chats": [], "checked_at": "2026-08-25T03:00:00Z"});
        let second = serde_json::json!({"chats": [], "checked_at": "2026-08-25T03:00:30Z"});
        let pointer = GatePredicate::At {
            pointer: "/chats".to_string(),
            is: PointerTest::Changed,
        };

        let baseline = judged(&pointer, &probed("", Some(first), true), None).baseline;
        let outcome = judged(
            &pointer,
            &probed("", Some(second.clone()), true),
            Some(&baseline),
        );
        assert!(
            !outcome.fired,
            "only `checked_at` moved, so the watched list did not change"
        );

        // The same two results under `changed`, to show the pointer is doing the work rather than
        // the values happening to be equal.
        let naive = judged(
            &GatePredicate::Changed,
            &probed(&second.to_string(), None, true),
            Some(
                &serde_json::json!({"chats": [], "checked_at": "2026-08-25T03:00:00Z"}).to_string(),
            ),
        );
        assert!(
            naive.fired,
            "whole-result `changed` sees the timestamp and fires, which is the trap"
        );
    }

    /// The user's case: fire when there is something to read, stay quiet when there is not.
    #[test]
    fn not_empty_follows_the_pointed_at_collection() {
        let predicate = GatePredicate::At {
            pointer: "/chats".to_string(),
            is: PointerTest::NotEmpty,
        };
        let empty = serde_json::json!({"chats": []});
        let full = serde_json::json!({"chats": [{"id": "a"}]});

        assert!(!judged(&predicate, &probed("", Some(empty), true), None).fired);
        assert!(judged(&predicate, &probed("", Some(full), true), None).fired);
    }

    /// Plenty of MCP servers send JSON as their text content and set no `structuredContent`, so a
    /// pointer has to reach that too or it would work against half the servers people run.
    #[test]
    fn a_pointer_falls_back_to_parsing_the_text() {
        let predicate = GatePredicate::At {
            pointer: "/chats".to_string(),
            is: PointerTest::NotEmpty,
        };
        let outcome = judged(
            &predicate,
            &probed(r#"{"chats": [{"id": "a"}]}"#, None, true),
            None,
        );
        assert!(outcome.fired, "the text parsed and the list is non-empty");
    }

    /// And it still parses when the document is larger than the turn is allowed to see.
    ///
    /// The two limits are unrelated and were entangled: `text` is capped so a runaway probe cannot
    /// push the prompt over the context window, and the cap appends a marker, so the result no
    /// longer parsed. Every shell probe and every MCP server that returns JSON as text content
    /// took that path, so an `at` gate over a large result reported "the probe did not return
    /// JSON" -- about a probe that did.
    #[test]
    fn a_pointer_reads_a_document_larger_than_the_turn_is_shown() {
        let filler = "x".repeat(GATE_OUTPUT_LIMIT);
        let raw = format!(r#"{{"filler": "{}", "chats": [{{"id": "a"}}]}}"#, filler);
        assert!(
            raw.len() > GATE_OUTPUT_LIMIT,
            "the document exceeds the cap"
        );

        let probe = ProbeOutcome::new(&raw, None, true);
        assert!(
            probe.text.ends_with("[gate output truncated]"),
            "the turn is still shown a bounded result"
        );

        let outcome = judged(
            &GatePredicate::At {
                pointer: "/chats".to_string(),
                is: PointerTest::NotEmpty,
            },
            &probe,
            None,
        );
        assert!(
            outcome.fired,
            "the pointer judges the whole document, which is what was measured"
        );
    }

    /// A pointer into something that is not a JSON document is an error, not an answer.
    ///
    /// The process ran fine, so this is not a spawn failure -- but the predicate describes a shape
    /// the result does not have, so nothing was measured. It used to decline silently, which is
    /// survivable for `not-empty` and ruinous for `empty`: a missing value reads as empty, so a
    /// server that began returning prose or an error string fired the job every interval,
    /// indefinitely. Both directions are covered here because the asymmetry is the point.
    #[test]
    fn a_pointer_into_something_that_is_not_json_is_an_error() {
        for is in [
            PointerTest::NotEmpty,
            PointerTest::Empty,
            PointerTest::Changed,
        ] {
            let predicate = GatePredicate::At {
                pointer: "/chats".to_string(),
                is,
            };
            let error = apply_predicate(&predicate, &probed("upstream is down", None, true), None)
                .expect_err("prose is not a document to point into");
            assert!(error.contains("/chats"), "{error}");
            assert!(
                error.contains("upstream is down"),
                "the message has to carry what it actually got: {error}"
            );
        }
    }

    /// The other half of the split: a document that parsed and simply lacks the field is an
    /// answer, and `empty` still fires on it. An API that omits `chats` when there are none is
    /// saying there are none, which is exactly what the predicate was asked.
    #[test]
    fn a_pointer_at_an_absent_field_in_real_json_still_answers() {
        let predicate = GatePredicate::At {
            pointer: "/chats".to_string(),
            is: PointerTest::Empty,
        };
        assert!(
            judged(&predicate, &probed(r#"{"other": 1}"#, None, true), None).fired,
            "the document parsed, so an absent field is a genuine `empty`"
        );

        let predicate = GatePredicate::At {
            pointer: "/chats".to_string(),
            is: PointerTest::NotEmpty,
        };
        assert!(!judged(&predicate, &probed(r#"{"other": 1}"#, None, true), None).fired);
    }

    #[test]
    fn matches_judges_the_text() {
        let predicate = GatePredicate::Matches {
            pattern: r"ERROR \d+".to_string(),
        };
        assert!(judged(&predicate, &probed("saw ERROR 500", None, true), None).fired);
        assert!(!judged(&predicate, &probed("all quiet", None, true), None).fired);
    }

    /// `succeeded` is the one predicate that reads the probe's own status rather than its output,
    /// which is what lets it mean the same thing for a command's exit code and a tool's error flag.
    #[test]
    fn succeeded_follows_the_probe_status_not_its_output() {
        let predicate = GatePredicate::Succeeded;
        assert!(judged(&predicate, &probed("", None, true), None).fired);
        assert!(!judged(&predicate, &probed("lots of output", None, false), None).fired);
    }

    /// Every shape survives the round trip through the two columns, including the arguments a tool
    /// probe carries. A gate that stored but did not reload would fire on whatever the decode
    /// happened to produce.
    #[test]
    fn every_gate_shape_round_trips_through_its_columns() {
        let shapes = [
            (
                GateProbe::Shell {
                    command: "gh pr checks".to_string(),
                },
                GatePredicate::Changed,
            ),
            (
                GateProbe::Shell {
                    command: "curl -f https://example.test".to_string(),
                },
                GatePredicate::Succeeded,
            ),
            (
                GateProbe::Tool {
                    name: "mcp__bridge__unseen".to_string(),
                    arguments: serde_json::json!({"folder": "inbox"}),
                },
                GatePredicate::At {
                    pointer: "/chats".to_string(),
                    is: PointerTest::NotEmpty,
                },
            ),
            (
                GateProbe::Tool {
                    name: "fetch_url".to_string(),
                    arguments: serde_json::json!({"url": "https://example.test/health"}),
                },
                GatePredicate::Matches {
                    pattern: "ok".to_string(),
                },
            ),
        ];

        for (probe, predicate) in shapes {
            let gate = Gate {
                probe: probe.clone(),
                predicate: predicate.clone(),
                last_output: Some("baseline".to_string()),
                permission: crate::permission::Permission::Read,
            };
            let restored = Gate::from_stored(
                probe.kind_str(),
                &gate.spec(),
                gate.last_output.clone(),
                gate.permission,
            )
            .expect("a gate meka wrote must be one meka can read");
            assert_eq!(restored.probe, probe);
            assert_eq!(restored.predicate, predicate);
            assert_eq!(restored.last_output.as_deref(), Some("baseline"));
        }
    }

    /// The three request-parser refusals that used to be silent resolutions or late failures.
    #[test]
    fn the_request_parsers_refuse_what_they_cannot_answer() {
        // Naming both halves of a `when` is an ambiguity, not a precedence question. Resolving it
        // to `matches` gave the model a gate watching something it did not ask for.
        let both = serde_json::json!({"matches": "x", "at": "/y", "is": "changed"});
        let error = GatePredicate::parse_request(Some(&both))
            .expect_err("`when` naming both is refused, as `check` naming both is");
        assert!(error.contains("both"), "{error}");

        // `arguments` reaches a tool, so it has to be the shape the tool schema declares.
        let scalar = serde_json::json!({"tool": "t", "arguments": "oops"});
        let error = GateProbe::parse_request(Some(&scalar))
            .expect_err("a string is not an argument object");
        assert!(error.contains("`check.arguments`"), "{error}");

        // The shapes that are fine stay fine, so the guards above are not just refusing everything.
        assert!(GatePredicate::parse_request(Some(&serde_json::json!({"matches": "x"}))).is_ok());
        assert!(
            GateProbe::parse_request(Some(&serde_json::json!({"tool": "t"})))
                .is_ok_and(|probe| matches!(probe, GateProbe::Tool { .. }))
        );
    }

    /// `ask` is refused for its own reason, and the message says which.
    ///
    /// Every other level below the bar fails on the missing sandbox. At `ask` nothing is sandboxed
    /// anyway; the objection is that the approval prompt the level exists for has nobody to answer
    /// it on a timer. Naming the sandbox there sends the reader after a setting that cannot help.
    #[test]
    fn the_ask_refusal_names_the_missing_approver_not_the_missing_sandbox() {
        let probe = GateProbe::Shell {
            command: "true".to_string(),
        };
        let refusal = gate_probe_is_authorised(&probe, crate::permission::Permission::Ask, None)
            .expect_err("`ask` cannot authorise a gate");
        let explained = refusal.explain(&probe, crate::permission::Permission::Ask);
        assert!(explained.contains("approve"), "{explained}");
        assert!(
            !explained.contains("sandbox"),
            "the sandbox is not why `ask` fails: {explained}"
        );

        // The ordinary case still gives the ordinary reason.
        let explained = refusal.explain(&probe, crate::permission::Permission::Read);
        assert!(explained.contains("no sandbox"), "{explained}");
    }

    /// Count the `warn!` lines a block emits, so "said once, not once per tick" is testable.
    ///
    /// The suppression this guards is about log volume, and log volume has no other observable:
    /// the job's state after a sweep is identical whether the line was written or not. Capturing
    /// the output is the only way to tell a fix from a no-op here.
    ///
    /// Capture goes through [`crate::render::log_capture`], which explains why the subscriber
    /// behind it is global. This test used `tracing::subscriber::set_default` instead and lost
    /// roughly two runs in ten to the callsite-interest race described there.
    ///
    /// `#[tokio::test]` is single-threaded, so `body` is polled on the thread that owns the
    /// capture buffer throughout.
    async fn warnings_from<F>(body: F) -> usize
    where
        F: std::future::Future<Output = ()>,
    {
        crate::render::log_capture::start();
        body.await;
        crate::render::log_capture::warnings()
            .matches("not fired:")
            .count()
    }

    /// A held job says so once, not once per poll interval -- including a one-shot.
    ///
    /// Retiring a one-shot used to clear its held-back state, on the reasoning that the row was
    /// gone. Once an authority refusal started putting the row back, that clear ran *before* the
    /// refusal on every sweep, so the "first time" check was true every time: a held one-shot
    /// warned every 10 seconds for up to the whole `missed_grace` window. Recurring jobs were
    /// unaffected, which is why it went unnoticed.
    #[tokio::test]
    async fn a_held_job_warns_once_not_once_per_sweep() {
        for (label, schedule) in [
            (
                "one-shot",
                Schedule::At(Utc::now() - chrono::Duration::minutes(1)),
            ),
            ("recurring", Schedule::parse_every("1h").expect("parses")),
        ] {
            let harness = SchedulerHarness::new().await;
            harness
                .manager
                .update_session_permission(harness.session_id, "read")
                .await
                .expect("record the level the session was set to");
            harness
                .overdue_job(
                    schedule,
                    Some(gate("true", GatePredicate::Succeeded, None)),
                    chrono::Duration::minutes(5),
                )
                .await;

            let warnings = warnings_from(async {
                for _ in 0..4 {
                    harness.tick().await;
                }
            })
            .await;
            assert_eq!(
                warnings, 1,
                "{label}: a standing condition is announced once, not on every sweep"
            );
        }
    }

    /// A dispatcher that knows nothing yet because its server is mid-handshake.
    #[derive(Debug)]
    struct StillConnecting;

    #[async_trait::async_trait]
    impl GateTools for StillConnecting {
        fn resolve(&self, _name: &str) -> Option<crate::permission::Permission> {
            None
        }

        fn is_still_connecting(&self, _name: &str) -> bool {
            true
        }

        async fn call(
            &self,
            _name: &str,
            _arguments: &serde_json::Value,
            _timeout: Duration,
            _cwd: Option<&std::path::Path>,
        ) -> std::result::Result<ProbeOutcome, String> {
            Err("not connected".to_string())
        }
    }

    /// A job carrying `gate` and nothing else of interest, for the predicate tests that never touch
    /// a store. A fresh id every call, so the process-global ledgers keyed by job id cannot carry
    /// one test's state into another's.
    fn job_carrying(gate: Option<Gate>) -> ScheduledJob {
        ScheduledJob {
            attempts: 0,
            id: uuid::Uuid::new_v4().to_string(),
            session_id: uuid::Uuid::nil(),
            schedule: Schedule::parse_every("1h").expect("parses"),
            prompt: "watch the thing".to_string(),
            gate,
            isolated: false,
            created_at: Utc::now(),
            last_fired_at: None,
            next_fire_at: Utc::now(),
        }
    }

    /// A gate whose server has not finished connecting is not reported as dead.
    ///
    /// Between process start and `Connected`, and again on every reconnect, the tool is absent from
    /// the snapshot. Marking that in the model's `[Scheduled]` block says a healthy job is dead and
    /// then announces it alive a turn later -- churn the model may act on. Authority is unchanged:
    /// the fire door still declines, because the probe genuinely cannot run.
    #[test]
    fn a_gate_whose_server_is_still_connecting_is_not_reported_as_dead() {
        let gate = Gate {
            probe: tool_probe(),
            predicate: GatePredicate::Succeeded,
            last_output: None,
            permission: crate::permission::Permission::Read,
        };
        let level = crate::permission::Permission::Read;

        let job = job_carrying(Some(gate.clone()));

        assert_eq!(
            job_withheld_reason(&job, level, Some(&StillConnecting)),
            None,
            "a handshake in progress is not a verdict"
        );
        assert!(
            job_withheld_reason(&job, level, Some(&FixedTools(None))).is_some(),
            "but a server that is simply not there still is"
        );
        assert!(
            gate_probe_is_authorised(&gate.probe, level, Some(&StillConnecting)).is_err(),
            "and the authority check refuses either way, since the probe cannot run"
        );
        assert_eq!(
            job_withheld(&job, Some(level), None),
            Withheld::Undetermined,
            "a reader with no dispatcher has not established that the job is fine; it has \
             established nothing, and `meka schedule list` renders that as `?` rather than as the \
             blank cell that means healthy"
        );
    }

    /// Cancelling a job that is already gone reports a miss, not a cancellation.
    ///
    /// Both cancel doors resolve an id from a listing and then delete it, and a scheduler sweep can
    /// retire the row in between: a one-shot's occurrence retires it, and deleting a session takes
    /// its jobs through the foreign key. Reporting success regardless said "Cancelled job abc12345"
    /// about a job this call did not cancel, in the same words it uses when it did.
    #[tokio::test]
    async fn deleting_a_job_that_is_already_gone_reports_that_it_removed_nothing() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;
        let store = harness.manager.schedule_store();

        assert!(
            store
                .delete_scheduled_job(&job.id)
                .await
                .expect("the delete runs"),
            "the row was there, so this call is the one that removed it"
        );
        assert!(
            !store
                .delete_scheduled_job(&job.id)
                .await
                .expect("the delete runs"),
            "nothing was removed the second time, and saying otherwise is a claim about work that \
             did not happen"
        );
        assert_eq!(
            harness
                .manager
                .schedule_store()
                .cancel_scheduled_job(harness.session_id, job.short_id())
                .await
                .expect("the cancel runs"),
            None,
            "and the prefix door agrees, since it delegates to the same delete"
        );
    }

    /// A dispatcher whose answers are fixed by the test, so the authority rule can be exercised
    /// without an MCP server.
    #[derive(Debug)]
    struct FixedTools(Option<crate::permission::Permission>);

    #[async_trait::async_trait]
    impl GateTools for FixedTools {
        fn resolve(&self, _name: &str) -> Option<crate::permission::Permission> {
            self.0
        }

        async fn call(
            &self,
            _name: &str,
            _arguments: &serde_json::Value,
            _timeout: Duration,
            _cwd: Option<&std::path::Path>,
        ) -> std::result::Result<ProbeOutcome, String> {
            Ok(probed("{}", None, true))
        }
    }

    fn tool_probe() -> GateProbe {
        GateProbe::Tool {
            name: "mcp__bridge__unseen".to_string(),
            arguments: serde_json::json!({}),
        }
    }

    /// The point of the whole permission split. A read-only tool call is not a bare `sh -c`, so
    /// holding it to `unrestricted` would leave gating unavailable to everyone below it -- which,
    /// with `workspace` now the default rung, is most people.
    #[test]
    fn a_read_only_tool_gate_is_allowed_at_read() {
        let tools = FixedTools(Some(crate::permission::Permission::Read));
        assert!(
            gate_probe_is_authorised(
                &tool_probe(),
                crate::permission::Permission::Read,
                Some(&tools)
            )
            .is_ok()
        );
    }

    /// The user's second scenario: a tool that resolved to `read` when the job was written but
    /// resolves higher today. Re-resolving at fire time is what catches it; trusting the level
    /// recorded at creation would keep calling it.
    #[test]
    fn a_tool_that_now_resolves_above_read_stops_being_a_gate() {
        let tools = FixedTools(Some(crate::permission::Permission::Unrestricted));
        let refusal = gate_probe_is_authorised(
            &tool_probe(),
            crate::permission::Permission::Unrestricted,
            Some(&tools),
        )
        .expect_err("a tool that can act is not a question");
        assert_eq!(
            refusal,
            GateRefusal::ToolNotReadOnly(crate::permission::Permission::Unrestricted)
        );
    }

    /// The user's first scenario, for the tool half: the session dropped below what the tool needs.
    #[test]
    fn a_tool_gate_stops_once_the_session_falls_below_read() {
        let tools = FixedTools(Some(crate::permission::Permission::Read));
        let refusal = gate_probe_is_authorised(
            &tool_probe(),
            crate::permission::Permission::None,
            Some(&tools),
        )
        .expect_err("`none` cannot call even a read-only tool");
        assert_eq!(refusal, GateRefusal::SessionBelowTool);
    }

    /// An unknown name and a disconnected server are the same answer, and both decline rather than
    /// fire. A gate that could not be evaluated has not passed.
    #[test]
    fn an_unresolvable_tool_declines_rather_than_fires() {
        for tools in [FixedTools(None), FixedTools(None)] {
            let refusal = gate_probe_is_authorised(
                &tool_probe(),
                crate::permission::Permission::Unrestricted,
                Some(&tools),
            )
            .expect_err("nothing to resolve against");
            assert_eq!(refusal, GateRefusal::ToolUnavailable);
        }
        // And a process with no dispatcher at all reads the same way.
        let refusal = gate_probe_is_authorised(
            &tool_probe(),
            crate::permission::Permission::Unrestricted,
            None,
        )
        .expect_err("this process cannot dispatch tools");
        assert_eq!(refusal, GateRefusal::ToolUnavailable);
    }

    /// The shell bar is unchanged, and unchanged for its own reason: a bare `sh -c` on a timer with
    /// meka's environment is not made safer by the probe split.
    #[test]
    fn a_shell_gate_still_needs_unrestricted() {
        let probe = GateProbe::Shell {
            command: "true".to_string(),
        };
        for level in [
            crate::permission::Permission::None,
            crate::permission::Permission::Read,
            crate::permission::Permission::Workspace,
            crate::permission::Permission::Ask,
        ] {
            assert_eq!(
                gate_probe_is_authorised(&probe, level, None),
                Err(GateRefusal::ShellNeedsUnrestricted),
                "a shell gate must not run at {level}"
            );
        }
        assert!(
            gate_probe_is_authorised(&probe, crate::permission::Permission::Unrestricted, None)
                .is_ok()
        );
    }

    /// `gate_kind` and `gate_spec` are written together, so disagreement means a hand-edited or
    /// damaged row. Resolving it to whichever half happens to parse would run a gate the operator
    /// did not write.
    #[test]
    fn a_gate_kind_that_contradicts_its_spec_is_refused() {
        let spec = r#"{"shell":{"command":"true"},"when":"changed"}"#;
        let error = Gate::from_stored("tool", spec, None, crate::permission::Permission::Read)
            .expect_err("the two columns disagree");
        assert!(error.contains("does not match its spec"), "{error}");
    }

    #[tokio::test]
    async fn test_on_success_gate_follows_the_exit_code() {
        let passing = evaluate_gate(
            &gate("exit 0", GatePredicate::Succeeded, None),
            GATE_BUDGET,
            None,
            None,
        )
        .await
        .expect("gate ran");
        assert!(passing.fired);

        let failing = evaluate_gate(
            &gate("exit 1", GatePredicate::Succeeded, None),
            GATE_BUDGET,
            None,
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
            &gate("echo ready", GatePredicate::Changed, None),
            GATE_BUDGET,
            None,
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
            &gate("pwd", GatePredicate::Changed, None),
            GATE_BUDGET,
            Some(&directory),
            None,
        )
        .await
        .expect("gate ran");

        assert_eq!(
            outcome.output,
            directory.to_string_lossy(),
            "the gate ran in the host's directory instead of the session's"
        );
    }

    /// A non-zero exit is how several perfectly good `changed` gates signal a change: `diff -q`
    /// and `git diff --exit-code` exit 1 exactly when there is a difference. Refusing to fire on a
    /// non-zero exit silenced those permanently.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_on_change_gate_that_signals_through_its_exit_code_still_fires() {
        let gate = Gate {
            probe: GateProbe::Shell {
                command: "echo 'Files a and b differ'; exit 1".to_string(),
            },
            predicate: GatePredicate::Changed,
            last_output: Some("".to_string()),
            permission: crate::permission::Permission::Unrestricted,
        };
        let outcome = evaluate_gate(&gate, GATE_BUDGET, None, None)
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
            probe: GateProbe::Shell {
                command: "exit 1".to_string(),
            },
            predicate: GatePredicate::Changed,
            last_output: Some("".to_string()),
            permission: crate::permission::Permission::Unrestricted,
        };
        let outcome = evaluate_gate(&gate, GATE_BUDGET, None, None)
            .await
            .expect("a quiet watcher is not a broken one");
        assert!(!outcome.fired, "nothing changed, so nothing fires");
    }

    #[tokio::test]
    async fn test_on_change_gate_is_quiet_until_the_output_differs() {
        let unchanged = evaluate_gate(
            &gate("echo steady", GatePredicate::Changed, Some("steady")),
            GATE_BUDGET,
            None,
            None,
        )
        .await
        .expect("gate ran");
        assert!(!unchanged.fired, "same output must not spend a turn");

        let changed = evaluate_gate(
            &gate("echo moved", GatePredicate::Changed, Some("steady")),
            GATE_BUDGET,
            None,
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
            &gate(command, GatePredicate::Changed, None),
            Duration::from_millis(150),
            None,
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
            &gate("echo spaced", GatePredicate::Changed, None),
            GATE_BUDGET,
            None,
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
                crate::session::SessionManager::open(
                    Some(std::path::Path::new(":memory:")),
                    &Default::default(),
                )
                .await
                .expect("open in-memory database"),
            );
            let session_id = manager
                .create_session(None, "test-profile".to_string())
                .await
                .expect("create session");
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
                .create_session(None, "test-profile".to_string())
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
                attempts: 0,
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

        /// Drag a job back to a moment ago, so the next sweep considers it again without waiting
        /// out its interval.
        async fn overdue_now(&self, id: &str) {
            let id = id.to_string();
            let due = (Utc::now() - chrono::Duration::minutes(1)).to_rfc3339();
            self.manager
                .schedule_store()
                .connection
                .call(move |connection| {
                    connection.execute(
                        "UPDATE scheduled_jobs SET next_fire_at = ?2 WHERE id = ?1",
                        rusqlite::params![id, due],
                    )
                })
                .await
                .expect("the row is there");
        }

        /// Swap a shell gate's command, for the tests that watch a probe break and then recover.
        async fn rewrite_gate(&self, id: &str, command: &str) {
            let id = id.to_string();
            let spec = serde_json::json!({
                "shell": { "command": command },
                "when": { "at": { "pointer": "/chats", "is": "not-empty" } },
            })
            .to_string();
            self.manager
                .schedule_store()
                .connection
                .call(move |connection| {
                    connection.execute(
                        "UPDATE scheduled_jobs SET gate_spec = ?2 WHERE id = ?1",
                        rusqlite::params![id, spec],
                    )
                })
                .await
                .expect("the row is there");
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
                    Some(gate("exit 1", GatePredicate::Succeeded, None)),
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
                Some(gate(
                    &create_file_command(&probe),
                    GatePredicate::Changed,
                    None,
                )),
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
            attempts: 0,
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
                Some(gate("exit 1", GatePredicate::Succeeded, None)),
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
                    GatePredicate::Succeeded,
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
                    GatePredicate::Succeeded,
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
                    GatePredicate::Succeeded,
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
                    GatePredicate::Succeeded,
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
            // A panicking turn leaves its claim to expire, so without a lease shorter than the
            // test's patience the same job would not come round twice. What is being asserted is
            // that the *loop* survives a panic, not how long the retry waits.
            claim_lease: std::time::Duration::from_millis(1),
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
                Some(gate("true", GatePredicate::Succeeded, None)),
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
                    GatePredicate::Succeeded,
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
        let gate_bar = "a gate command runs unattended with no sandbox";

        assert!(
            declined_for_permission_first_time(&job, gate_bar),
            "the first sweep of a downgrade has to say why"
        );
        assert!(
            !declined_for_permission_first_time(&job, gate_bar),
            "and the ones after it must not repeat"
        );
        assert!(
            !declined_for_permission_first_time(&job, gate_bar),
            "however many there are"
        );

        // A different reason for the same job. Keyed by job alone this was silent, so dropping a
        // session from `read` to `none` -- which stops *every* job, not just the gated one --
        // arrived with no line at all, and raising it back to `read` said nothing either.
        assert!(
            declined_for_permission_first_time(&job, "unattended-work:none"),
            "a job held for a different reason is a different fact, and the remedy differs too"
        );
        assert!(
            !declined_for_permission_first_time(&job, "unattended-work:none"),
            "and that one settles into silence in its turn"
        );
        assert!(
            declined_for_permission_first_time(&job, gate_bar),
            "including on the way back"
        );

        clear_permission_decline(&job);
        assert!(
            declined_for_permission_first_time(&job, gate_bar),
            "a later withdrawal is a new fact and is announced again"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_gate_whose_authority_was_withdrawn_is_not_executed() {
        let marker = std::env::temp_dir().join(format!("meka-gate-{}", uuid::Uuid::new_v4()));
        let _ = std::fs::remove_file(&marker);

        let harness = SchedulerHarness::new().await;
        let mut withdrawn = gate(
            &create_file_command(&marker),
            GatePredicate::Succeeded,
            None,
        );
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
            attempts: 0,
            id: "7f3a1b2c-0000-0000-0000-000000000000".to_string(),
            session_id: uuid::Uuid::nil().to_string(),
            kind: "every".to_string(),
            spec: "1h".to_string(),
            prompt: "do the thing".to_string(),
            gate_kind: Some("shell".to_string()),
            gate_spec: Some(r#"{"shell":{"command":"true"},"when":"succeeded"}"#.to_string()),
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
        let authorised = gate(
            &create_file_command(&marker),
            GatePredicate::Succeeded,
            None,
        );
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
                    GatePredicate::Succeeded,
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

    /// A job on a session at `none` does not fire, gate or no gate.
    ///
    /// The turn it would spend can do nothing: every tool is refused at dispatch, so it reads
    /// nothing, changes nothing, and its `schedule_cancel` is refused too, leaving it unable to
    /// stop itself being woken again. Registration does not depend on the level, so it *sees* the
    /// job in `[Scheduled]` and is offered the tool; the refusal is at the point of use, which is
    /// the worst of both. An ungated `every = "5s"` job was a turn every five seconds for as long
    /// as the session sat there, stoppable only by an operator. Ungated is the case that matters
    /// here: a gated job was already refused by the authority check, which is why this went
    /// unnoticed.
    #[tokio::test]
    async fn an_ungated_job_does_not_fire_on_a_session_at_none() {
        let harness = SchedulerHarness::new().await;
        harness
            .manager
            .update_session_permission(harness.session_id, "none")
            .await
            .expect("record the level the session was set to");
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        assert!(
            harness.fired().is_empty(),
            "waking a model that cannot act, and cannot cancel the job that woke it, is a turn \
             spent to no purpose"
        );
    }

    /// A one-shot held back for authority survives; it is not spent.
    ///
    /// A one-shot is retired the instant it comes due, *before* the gate is consulted, which is
    /// right when the gate ran and said no: its moment has passed either way. It is wrong when the
    /// gate was never consulted at all. Lowering a session for one minute destroyed every one-shot
    /// that happened to come due in that minute, and the log line said "not fired", which reads as
    /// held rather than deleted. The gate-error path already made this distinction for a gate that
    /// *errored*; the two authority refusals belong on the same side of it.
    #[tokio::test]
    async fn a_one_shot_held_for_authority_is_kept_rather_than_retired() {
        for (label, level, gate_on_it) in [
            ("session at none, ungated", "none", false),
            ("session below the gate's bar", "read", true),
        ] {
            let harness = SchedulerHarness::new().await;
            harness
                .manager
                .update_session_permission(harness.session_id, level)
                .await
                .expect("record the level the session was set to");
            harness
                .overdue_job(
                    Schedule::At(Utc::now() - chrono::Duration::minutes(1)),
                    gate_on_it.then(|| gate("true", GatePredicate::Succeeded, None)),
                    chrono::Duration::minutes(1),
                )
                .await;

            harness.tick().await;

            assert!(harness.fired().is_empty(), "{label}: must not fire");
            assert_eq!(
                harness.jobs().await.len(),
                1,
                "{label}: the reminder was never evaluated, so it must still be there once the \
                 level is restored"
            );
        }
    }

    /// A refused job's row is never deleted, not even for the instant it took to put it back.
    ///
    /// Kept-and-restored and never-touched leave identical rows, which is why the resurrection this
    /// prevents was invisible: claiming used to delete a one-shot before the refusal was even
    /// consulted, and the restore is an `INSERT` that cannot tell "I deleted this a moment ago"
    /// from "the user cancelled it in between" -- so a `schedule_cancel` landing in that window was
    /// silently undone while both cancel doors reported success.
    ///
    /// SQLite's `rowid` is the observable: a re-`INSERT` takes `max(rowid) + 1`, so a row that kept
    /// its rowid was never deleted. Restoring the delete-then-restore shape fails this with two
    /// different values.
    ///
    /// The second job is not decoration. With one row in the table the deleted rowid is also
    /// `max + 1`, so SQLite hands the same value straight back and the assertion holds under both
    /// orderings -- which is what this test did until the mutation check caught it.
    #[tokio::test]
    async fn a_refused_one_shot_keeps_its_row_rather_than_being_deleted_and_restored() {
        let harness = SchedulerHarness::new().await;
        harness
            .manager
            .update_session_permission(harness.session_id, "read")
            .await
            .expect("record the level the session was set to");
        let job = harness
            .overdue_job(
                Schedule::At(Utc::now() - chrono::Duration::minutes(1)),
                Some(gate("true", GatePredicate::Succeeded, None)),
                chrono::Duration::minutes(1),
            )
            .await;
        // Behind it in the table and not due, so it is never considered and only ever raises
        // `max(rowid)`.
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                -chrono::Duration::hours(1),
            )
            .await;

        let rowid = |id: String| {
            let store = harness.manager.schedule_store();
            async move {
                store
                    .connection
                    .call(move |connection| {
                        connection.query_row(
                            "SELECT rowid FROM scheduled_jobs WHERE id = ?1",
                            rusqlite::params![id],
                            |row| row.get::<_, i64>(0),
                        )
                    })
                    .await
                    .expect("the row is there")
            }
        };

        let before = rowid(job.id.clone()).await;
        harness.tick().await;
        let after = rowid(job.id.clone()).await;

        assert_eq!(
            before, after,
            "a held one-shot must keep its row: deleting it to put it back is what loses a cancel \
             issued in between"
        );
    }

    /// A gate whose probe keeps breaking is reported to the model, once the breakage is standing.
    ///
    /// Authority is not the commonest way a watcher dies. A server that changed its schema, a
    /// command that was uninstalled, a pointer into a result that stopped being JSON: each errors
    /// on every evaluation and each is indistinguishable, from the model's side, from a healthy
    /// watcher with nothing to report -- which is the whole reason the marker exists. The first
    /// failure is deliberately silent, because one failure is as often a blip as a break.
    #[tokio::test]
    async fn a_gate_whose_probe_keeps_failing_is_reported_after_the_second_failure() {
        let harness = SchedulerHarness::new().await;
        harness
            .manager
            .update_session_permission(harness.session_id, "unrestricted")
            .await
            .expect("record the level the session was set to");
        // A pointer into output that is not JSON: an error, not an answer, on every evaluation.
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate(
                    "echo not-json-at-all",
                    GatePredicate::At {
                        pointer: "/chats".to_string(),
                        is: PointerTest::NotEmpty,
                    },
                    None,
                )),
                chrono::Duration::minutes(5),
            )
            .await;
        let level = crate::permission::Permission::Unrestricted;

        harness.tick().await;
        assert_eq!(
            job_withheld_reason(&job, level, None),
            None,
            "one failure is a blip, and marking a job dead for it would flap"
        );

        harness.overdue_now(&job.id).await;
        harness.tick().await;
        let reported = job_withheld_reason(&job, level, None).unwrap_or_default();
        assert!(
            reported.contains("keeps failing"),
            "a standing breakage is the model's to act on: {reported}"
        );

        // And the sentence does not move once it has been said. Every reader compares it by
        // equality, so a running total in it made `render_world_state_diff` re-announce the job to
        // the model on every single failed evaluation.
        harness.overdue_now(&job.id).await;
        harness.tick().await;
        assert_eq!(
            job_withheld_reason(&job, level, None).unwrap_or_default(),
            reported,
            "a third failure says exactly what the second did"
        );
        assert!(
            reported.contains("not JSON") || reported.contains("did not return JSON"),
            "and it has to say what broke: {reported}"
        );

        // A working evaluation ends it, so the marker tracks the gate rather than accumulating.
        harness.overdue_now(&job.id).await;
        // Single-quoted, or `sh` eats the quotes and the probe emits `{chats: []}`, which is not
        // JSON either.
        harness
            .rewrite_gate(&job.id, "echo '{\"chats\": []}'")
            .await;
        harness.tick().await;
        assert_eq!(
            job_withheld_reason(&job, level, None),
            None,
            "the probe answered, so there is nothing left to report"
        );
    }

    /// A crash between the claim and the delivery costs a retry, not the job.
    ///
    /// This is what the lease is for. Claiming used to consume the row -- advancing a recurring
    /// job's `next_fire_at`, deleting a one-shot's row outright -- so a host that died before the
    /// turn ran left the occurrence spent, and for a one-shot that meant the reminder was gone with
    /// nothing anywhere to recover it from. Nothing swept for it either, unlike `background_tasks`,
    /// which is marked `interrupted` when a process takes the session lock.
    #[tokio::test]
    async fn a_crash_between_the_claim_and_the_turn_does_not_lose_the_job() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::At(Utc::now() - chrono::Duration::minutes(1)),
                None,
                chrono::Duration::minutes(1),
            )
            .await;
        let store = harness.manager.schedule_store();

        // A host takes the occurrence and is killed before it can deliver: no completion, no
        // release, just a lease nobody will ever come back for.
        let died_at = Utc::now() - chrono::Duration::hours(2);
        assert!(
            store
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "host-that-died",
                    died_at,
                    died_at + chrono::Duration::hours(1),
                )
                .await
                .expect("claim")
        );

        harness.tick().await;

        assert_eq!(
            harness.fired().len(),
            1,
            "the reminder is delivered by the next host once the dead one's lease expires"
        );
        assert!(
            harness.jobs().await.is_empty(),
            "and retired properly afterwards, rather than left leased forever"
        );
    }

    /// A job whose turn keeps taking the host down is parked rather than retried forever.
    ///
    /// The counterpart the lease requires. Spending the occurrence up front used to make a
    /// crash-on-every-fire job cost one occurrence per crash and move on; a lease hands the same
    /// occurrence back every time, so without a ceiling a prompt that kills the process is claimed
    /// again on every sweep, forever. The row is kept rather than deleted: it stays listed,
    /// cancellable, and reported as held, because meka cannot tell a poisonous prompt from an
    /// unlucky one and destroying a user's job on that guess would be worse.
    #[tokio::test]
    async fn a_job_that_keeps_crashing_its_host_is_parked_rather_than_retried_forever() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;
        let store = harness.manager.schedule_store();

        // Every claim ends in a crash: taken, never completed, never voluntarily handed back.
        for attempt in 1..=MAX_CLAIM_ATTEMPTS {
            let now = Utc::now();
            assert!(
                store
                    .claim_occurrence(
                        &job.id,
                        job.next_fire_at,
                        &format!("host-{attempt}"),
                        now,
                        now - chrono::Duration::seconds(1),
                    )
                    .await
                    .expect("claim"),
                "attempt {attempt}: an expired lease is takeable"
            );
        }

        harness.tick().await;

        assert!(
            harness.fired().is_empty(),
            "the ceiling is reached, so the job is not handed to a fourth host to kill"
        );
        let parked = harness.jobs().await;
        let parked = parked.first().expect("and the job is kept, not destroyed");
        assert_eq!(parked.attempts, MAX_CLAIM_ATTEMPTS);
        assert_eq!(
            job_withheld_reason(parked, crate::permission::Permission::Unrestricted, None)
                .as_deref()
                .map(|reason| reason.contains("claims ended without delivering")),
            Some(true),
            "and every surface says so, because a parked job that looks healthy is the thing this \
             whole marker exists to prevent"
        );
    }

    /// A stale *completion* is refused too, not just a stale release.
    ///
    /// The completion of a one-shot is the one `DELETE` the scheduler issues, so an unscoped one
    /// would let a host whose lease expired retire a job the current holder is still delivering --
    /// the reminder vanishing mid-turn, from a write issued by a host that no longer owns it.
    #[tokio::test]
    async fn a_stale_completion_does_not_retire_another_hosts_job() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::At(Utc::now() - chrono::Duration::minutes(1)),
                None,
                chrono::Duration::minutes(1),
            )
            .await;
        let store = harness.manager.schedule_store();

        let long_ago = Utc::now() - chrono::Duration::hours(2);
        assert!(
            store
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "host-a",
                    long_ago,
                    long_ago + chrono::Duration::minutes(1),
                )
                .await
                .expect("claim")
        );
        let now = Utc::now();
        assert!(
            store
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "host-b",
                    now,
                    now + chrono::Duration::hours(1)
                )
                .await
                .expect("host B takes the expired lease")
        );

        // Host A finally finishes and retires the one-shot it thinks it delivered.
        store
            .complete_claim(&job.id, "host-a", None, Some(Utc::now()), None)
            .await
            .expect("complete");

        assert_eq!(
            harness.jobs().await.len(),
            1,
            "host B is still delivering it: a stale completion may not retire the row"
        );
    }

    /// A turn that panics leaves its claim to expire, and is not forgiven for it.
    ///
    /// Two properties in one shape, because they are the same trade. Leaving the lease is what
    /// spaces the retries: giving the occurrence back at once leaves the row due on the next
    /// sweep, so three panics arrive within `3 * poll_interval` -- half a minute at the default --
    /// and park a recurring job that `missed_grace` will never retire. *Not* resetting the attempt
    /// count is what stops the same panic being retried forever once the spacing is in place.
    #[tokio::test]
    async fn a_panicking_turn_leaves_its_claim_to_expire_and_still_counts_against_the_ceiling() {
        let mut harness = SchedulerHarness::new().await;
        // Expired by the time the next sweep looks, so one sweep stands in for one `claim_lease`.
        // Set on the resolved config directly because `validate` refuses anything this short.
        harness.config.claim_lease = Duration::from_millis(1);
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;

        let attempts_after = |attempt: u32| {
            let manager = harness.manager.clone();
            let config = harness.config.clone();
            let id = job.id.clone();
            async move {
                run_due(
                    &manager,
                    &config,
                    &SchedulerScope::every_job(),
                    &move |_wakeup: Wakeup| async move {
                        panic!("the turn blew up");
                    },
                )
                .await
                .expect("the sweep survives the panic");
                tokio::time::sleep(Duration::from_millis(3)).await;
                manager
                    .schedule_store()
                    .list_all_scheduled_jobs()
                    .await
                    .expect("list")
                    .into_iter()
                    .find(|job| job.id == id)
                    .unwrap_or_else(|| panic!("attempt {attempt}: the job survives the panic"))
                    .attempts
            }
        };

        assert_eq!(
            attempts_after(1).await,
            1,
            "the crash is counted rather than forgiven"
        );
        assert_eq!(
            attempts_after(2).await,
            2,
            "a second panic counts again: forgiving it would retry this prompt forever"
        );
        assert_eq!(attempts_after(3).await, MAX_CLAIM_ATTEMPTS);

        // The ceiling is reached, so the fourth sweep does not hand it to another turn to kill.
        harness.tick().await;
        assert!(harness.fired().is_empty(), "and the job is parked");
    }

    /// The other half of the reorder: a *recurring* job refused for authority still spends the
    /// occurrence it came due for.
    ///
    /// Moving the checks above the claim must not turn a held job into one that sits permanently
    /// due, accumulating a backlog it would report the moment it was authorised again. Only the
    /// one-shot is spared, because only the one-shot's claim destroys anything.
    #[tokio::test]
    async fn a_refused_recurring_job_still_spends_its_occurrence() {
        let harness = SchedulerHarness::new().await;
        harness
            .manager
            .update_session_permission(harness.session_id, "read")
            .await
            .expect("record the level the session was set to");
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate("true", GatePredicate::Succeeded, None)),
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        let after = harness.jobs().await;
        let [refused] = after.as_slice() else {
            panic!("the job survives a refusal: {after:?}");
        };
        assert!(harness.fired().is_empty(), "it must not have fired");
        assert!(
            refused.next_fire_at > job.next_fire_at,
            "the occurrence is spent, exactly as it is when a gate runs and says no"
        );
    }

    /// The companion, so the refusal above is about `none` and not about ungated jobs having
    /// quietly stopped working: one rung up, the same job fires.
    #[tokio::test]
    async fn an_ungated_job_fires_on_a_session_at_read() {
        let harness = SchedulerHarness::new().await;
        harness
            .manager
            .update_session_permission(harness.session_id, "read")
            .await
            .expect("record the level the session was set to");
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        assert_eq!(
            harness.fired().len(),
            1,
            "`read` can still read, report and cancel, so the reminder is worth waking for"
        );
    }

    /// The fire door has to ask the same question the two creation doors ask.
    ///
    /// Both creation doors call `gate_probe_is_authorised`, but `prepare` kept 0.42's
    /// `allows_unattended_shell` check, which demands `unrestricted` whatever the probe is. A tool
    /// gate was therefore *accepted* at `read` and then declined on every tick forever, warning
    /// about an unattended shell command the job did not have. The headline case,
    /// `mcp__…__unseen` at `read`, never called its probe once: the feature was inert wherever it
    /// was reachable.
    ///
    /// Nothing in the suite caught it because every other fire-time test here uses a shell gate at
    /// `unrestricted`, where the old check and the new one agree. It took running the thing.
    #[tokio::test]
    async fn a_read_only_tool_gate_fires_at_read() {
        let harness =
            SchedulerHarness::at_host_permission(crate::permission::Permission::Read).await;
        harness
            .manager
            .update_session_permission(harness.session_id, "read")
            .await
            .expect("record the level the session was set to");
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(Gate {
                    probe: tool_probe(),
                    predicate: GatePredicate::Succeeded,
                    last_output: None,
                    permission: crate::permission::Permission::Read,
                }),
                chrono::Duration::minutes(5),
            )
            .await;

        harness
            .tick_with(crate::config::ResolvedScheduleConfig {
                gate_tools: Some(std::sync::Arc::new(FixedTools(Some(
                    crate::permission::Permission::Read,
                )))),
                ..harness.config.clone()
            })
            .await;

        assert_eq!(
            harness.fired().len(),
            1,
            "a read-only tool gate must fire at `read`, which is the entire point of not holding \
             every probe to the shell bar"
        );
    }

    /// The companion, and the user's second scenario: the same job, after the operator retuned the
    /// tool above `read`. Withdrawn at fire time, without the row changing.
    #[tokio::test]
    async fn a_tool_gate_stops_firing_once_the_tool_resolves_above_read() {
        let harness =
            SchedulerHarness::at_host_permission(crate::permission::Permission::Read).await;
        harness
            .manager
            .update_session_permission(harness.session_id, "read")
            .await
            .expect("record the level the session was set to");
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(Gate {
                    probe: tool_probe(),
                    predicate: GatePredicate::Succeeded,
                    last_output: None,
                    permission: crate::permission::Permission::Read,
                }),
                chrono::Duration::minutes(5),
            )
            .await;

        harness
            .tick_with(crate::config::ResolvedScheduleConfig {
                gate_tools: Some(std::sync::Arc::new(FixedTools(Some(
                    crate::permission::Permission::Unrestricted,
                )))),
                ..harness.config.clone()
            })
            .await;

        assert!(
            harness.fired().is_empty(),
            "a tool that no longer resolves to `read` must stop being a gate, or retuning \
             `tool_permissions` means nothing to a job already on the timer"
        );
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
                Some(gate("exit 1", GatePredicate::Succeeded, None)),
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
                Some(gate("echo ci-red", GatePredicate::Changed, None)),
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

    /// A deferral must also leave the gate's baseline alone.
    ///
    /// The baseline is measured inside `prepare` and rides on the [`Claim`] until the occurrence is
    /// disposed of, which is what makes this work: a host that evaluates, decides to fire and then
    /// cannot run the turn writes nothing. Persisting it at the moment the gate returned would
    /// leave the watcher having already absorbed the change it exists to report, so the next host
    /// would compare the new value against itself, see nothing, and stay quiet forever.
    #[tokio::test]
    async fn test_a_deferred_gated_job_keeps_its_baseline() {
        let harness = SchedulerHarness::new().await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate(
                    "echo changed",
                    GatePredicate::Changed,
                    Some("original"),
                )),
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
    /// database both read the same occurrence into their due lists; exactly one of them may take
    /// it. Before this the write was unconditional, so both advanced the row and both went on to
    /// fire.
    ///
    /// One shape for both kinds of schedule now, which is the point of leasing rather than
    /// consuming: a one-shot used to be claimed by deleting its row, so the same test had to be
    /// written twice and the delete had nothing to hand back.
    #[tokio::test]
    async fn test_only_one_host_can_claim_an_occurrence() {
        for (label, schedule) in [
            ("recurring", Schedule::parse_every("1h").expect("parses")),
            (
                "one-shot",
                Schedule::At(Utc::now() - chrono::Duration::minutes(1)),
            ),
        ] {
            let harness = SchedulerHarness::new().await;
            let job = harness
                .overdue_job(schedule, None, chrono::Duration::minutes(5))
                .await;
            let store = harness.manager.schedule_store();
            let now = Utc::now();
            let until = now + chrono::Duration::hours(1);

            assert!(
                store
                    .claim_occurrence(&job.id, job.next_fire_at, "host-a", now, until)
                    .await
                    .expect("claim"),
                "{label}: the first host to reach the row takes the occurrence"
            );
            assert!(
                !store
                    .claim_occurrence(&job.id, job.next_fire_at, "host-b", now, until)
                    .await
                    .expect("claim"),
                "{label}: and the second, still holding the copy it listed, is refused"
            );
            assert_eq!(
                harness.jobs().await.first().map(|job| job.next_fire_at),
                Some(job.next_fire_at),
                "{label}: and the row itself has not moved, because a claim no longer consumes it"
            );
        }
    }

    /// A lease that has run out is takeable, and that is what a crash costs: a delay, not the job.
    ///
    /// The host that dies mid-delivery never releases. Under the old design its claim had already
    /// consumed the row -- advanced past the occurrence, or for a one-shot deleted outright -- so
    /// the occurrence was gone with nothing to recover it from and the reminder was simply never
    /// delivered.
    #[tokio::test]
    async fn an_expired_lease_is_taken_by_the_next_host() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::At(Utc::now() - chrono::Duration::minutes(1)),
                None,
                chrono::Duration::minutes(1),
            )
            .await;
        let store = harness.manager.schedule_store();
        let died_at = Utc::now() - chrono::Duration::hours(2);

        assert!(
            store
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "host-that-died",
                    died_at,
                    died_at + chrono::Duration::hours(1),
                )
                .await
                .expect("claim"),
            "a host takes the occurrence and then never comes back"
        );
        let now = Utc::now();
        assert!(
            store
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "host-b",
                    now,
                    now + chrono::Duration::hours(1)
                )
                .await
                .expect("claim"),
            "an hour later the lease has expired and the job is deliverable again"
        );
        assert_eq!(
            harness.jobs().await.len(),
            1,
            "and it was there to be taken, which a consumed row would not have been"
        );
    }

    /// A gate that cannot be evaluated spends the occurrence, exactly as one that ran and said no
    /// does.
    ///
    /// The obvious handling was to release the lease, since nothing was measured and the job has
    /// not had its turn. That leaves `next_fire_at` where it was, so the row is due again on the
    /// very next sweep: a six-hour job whose server is down was re-probed every `poll_interval`
    /// rather than every six hours, and a probe that hangs burned the whole `gate_timeout` out of
    /// each sweep. Under consume-to-claim the schedule had already advanced before the gate ran, so
    /// this could not happen and nothing had to decide it.
    #[tokio::test]
    async fn a_gate_that_cannot_be_evaluated_spends_a_recurring_occurrence() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::parse_every("6h").expect("parses"),
                // A pointer into output that is not JSON: an error on every evaluation.
                Some(gate(
                    "echo not-json-at-all",
                    GatePredicate::At {
                        pointer: "/chats".to_string(),
                        is: PointerTest::NotEmpty,
                    },
                    Some("the value last actually observed"),
                )),
                chrono::Duration::minutes(1),
            )
            .await;

        for _ in 0..4 {
            harness.tick().await;
        }

        assert!(harness.fired().is_empty(), "a broken gate never fires");
        let after = harness.jobs().await;
        let after = after.first().expect("the job survives");
        assert!(
            after.next_fire_at > job.next_fire_at,
            "the occurrence is spent, so the next probe is a period away and not a tick away: \
             {:?} vs {:?}",
            after.next_fire_at,
            job.next_fire_at
        );
        assert_eq!(
            probe_failure(&job.id).map(|(count, _)| count),
            Some(1),
            "and four sweeps cost one probe, because only one occurrence came due"
        );
        assert_eq!(
            after
                .gate
                .as_ref()
                .and_then(|gate| gate.last_output.as_deref()),
            Some("the value last actually observed"),
            "nothing was measured, so the baseline must survive: the next working evaluation is \
             what reports the change that happened while the probe was broken"
        );
        assert_eq!(
            after.attempts, 0,
            "and the job is not on its way to being parked, because its occurrences are spent \
             rather than accumulating"
        );
    }

    /// The same for a one-shot, which has no next occurrence to spend: the lease is what waits.
    ///
    /// Advancing is not available, so the row stays due, and releasing the lease would make it due
    /// *now* -- re-probed on every sweep, and parked by the attempt ceiling after three of them,
    /// which at the default `poll_interval` is half a minute. A server restarting anywhere near a
    /// one-shot's due time would silently destroy the reminder, which is a worse failure than the
    /// cost the advance exists to avoid. Keeping the lease spaces the retry by `claim_lease`.
    #[tokio::test]
    async fn a_gate_that_cannot_be_evaluated_holds_a_one_shots_lease_rather_than_reprobing() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::At(Utc::now() - chrono::Duration::minutes(1)),
                Some(gate(
                    "echo not-json-at-all",
                    GatePredicate::At {
                        pointer: "/chats".to_string(),
                        is: PointerTest::NotEmpty,
                    },
                    None,
                )),
                chrono::Duration::minutes(1),
            )
            .await;

        for _ in 0..6 {
            harness.tick().await;
        }

        assert_eq!(
            probe_failure(&job.id).map(|(count, _)| count),
            Some(1),
            "six sweeps inside one lease cost one probe, not six"
        );
        let after = harness.jobs().await;
        let after = after.first().expect("the job is kept, not deleted");
        assert_eq!(
            after.attempts, 1,
            "and it is nowhere near the ceiling, so a server that comes back inside the lease \
             still delivers the reminder"
        );
    }

    /// It does still park, once those spaced-out retries are spent.
    ///
    /// The ceiling has to survive the backoff above, or a one-shot whose gate is permanently broken
    /// is probed once per lease until its grace period closes. Each expiry is a fresh claim, so the
    /// count rises on the retries rather than on the sweeps.
    #[tokio::test]
    async fn a_one_shot_whose_gate_never_works_is_parked_once_its_retries_are_spent() {
        let mut harness = SchedulerHarness::new().await;
        // A lease that has run out by the next sweep, so one tick stands in for one `claim_lease`
        // without the test waiting one out. Set on the resolved config directly because
        // `validate` refuses anything this short: what is being exercised is the expiry, not a
        // setting anyone can configure.
        harness.config.claim_lease = Duration::from_millis(1);
        harness
            .overdue_job(
                Schedule::At(Utc::now() - chrono::Duration::minutes(1)),
                Some(gate(
                    "echo not-json-at-all",
                    GatePredicate::At {
                        pointer: "/chats".to_string(),
                        is: PointerTest::NotEmpty,
                    },
                    None,
                )),
                chrono::Duration::minutes(1),
            )
            .await;

        for _ in 0..6 {
            tokio::time::sleep(Duration::from_millis(3)).await;
            harness.tick().await;
        }

        let after = harness.jobs().await;
        let after = after.first().expect("a parked job is kept, not deleted");
        assert_eq!(
            after.attempts, MAX_CLAIM_ATTEMPTS,
            "the ceiling still bites: past it the job is refused before its gate runs"
        );
        let reported =
            job_withheld_reason(after, crate::permission::Permission::Unrestricted, None)
                .unwrap_or_default();
        assert!(
            reported.contains("gate could not be evaluated"),
            "and it is reported for what happened, not as a crash: {reported}"
        );
        assert!(
            reported.contains("JSON"),
            "naming the probe's own error, which is the actionable half: {reported}"
        );
    }

    /// An expired lease is "unclaimed" to the paths that hand out work, so it must be unclaimed to
    /// the paths that retire and advance without taking one.
    ///
    /// Nothing clears `claimed_by` but a release, a completion or a fresh claim, so a host that
    /// dies holding a lease leaves it set for good. While those two paths tested the column rather
    /// than the expiry, such a row was handed to `prepare` on every sweep and was invisible to
    /// both: a one-shot past its grace period was never retired, never fired and never logged, and
    /// a refused recurring job never advanced.
    #[tokio::test]
    async fn a_lease_left_by_a_dead_host_does_not_wedge_the_occurrence() {
        for (label, schedule, host, overdue) in [
            (
                "one-shot past its grace period is retired",
                Schedule::At(Utc::now() - chrono::Duration::hours(25)),
                crate::permission::Permission::Unrestricted,
                chrono::Duration::hours(25),
            ),
            (
                "recurring job refused at `none` still spends its occurrence",
                Schedule::parse_every("1h").expect("parses"),
                crate::permission::Permission::None,
                chrono::Duration::hours(6),
            ),
        ] {
            let recurring = schedule.is_recurring();
            let harness = SchedulerHarness::at_host_permission(host).await;
            let job = harness.overdue_job(schedule, None, overdue).await;
            let store = harness.manager.schedule_store();
            let died_at = Utc::now() - overdue;
            assert!(
                store
                    .claim_occurrence(
                        &job.id,
                        job.next_fire_at,
                        "host-that-died",
                        died_at,
                        died_at + chrono::Duration::hours(1),
                    )
                    .await
                    .expect("claim"),
                "{label}: a host takes the occurrence and is killed before it can release"
            );

            harness.tick().await;

            assert!(harness.fired().is_empty(), "{label}: nothing is delivered");
            let after = harness.jobs().await;
            match recurring {
                false => assert!(
                    after.is_empty(),
                    "{label}: it must be retired, as it is when no crashed host ever touched it"
                ),
                true => assert!(
                    after
                        .first()
                        .is_some_and(|after| after.next_fire_at > job.next_fire_at),
                    "{label}: {:?} vs {:?}",
                    after.first().map(|after| after.next_fire_at),
                    job.next_fire_at
                ),
            }
        }
    }

    /// A sweep that bounded its own coverage says so.
    ///
    /// The budget holds jobs over, and the next sweep takes them, so nothing is lost -- which is
    /// exactly why the line matters: without it a capped run is indistinguishable from a complete
    /// one in the log, and an operator watching a backlog has no way to tell that the cap is what
    /// they are looking at. Found by a mutation sweep: `held_over += 1` could be neutered and
    /// `held_over > 0` inverted with every test still green, because the count had no reader but
    /// this line and the line had no reader at all.
    #[tokio::test]
    async fn a_sweep_that_holds_jobs_over_reports_that_it_did() {
        let mut harness = SchedulerHarness::new().await;
        harness.config.max_consecutive_fires = 2;
        for _ in 0..5 {
            harness
                .overdue_job(
                    Schedule::parse_every("1h").expect("parses"),
                    None,
                    chrono::Duration::hours(6),
                )
                .await;
        }

        crate::render::log_capture::start();
        harness.tick().await;
        let reported = crate::render::log_capture::infos();
        assert!(
            reported.contains("held over 3 due job(s)"),
            "five due jobs against a budget of two leaves three, and the count has to be right or \
             the line is worse than nothing: {reported}"
        );

        // And it stays quiet when the budget did not engage, or it would train the reader to skip
        // it.
        let harness = SchedulerHarness::new().await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::hours(6),
            )
            .await;
        crate::render::log_capture::start();
        harness.tick().await;
        assert!(
            !crate::render::log_capture::infos().contains("held over"),
            "one job and a budget of five is not a bounded sweep"
        );
    }

    /// Several hosts noticing the same expired one-shot produce one announcement, not one each.
    ///
    /// The delete is scoped to the occurrence, so whoever wins removes the row and everyone else
    /// changes nothing -- and the return value is how the winner knows to be the one that speaks.
    /// Every assertion about the row itself passes whichever way that value goes, which is why a
    /// mutation of it survived.
    #[tokio::test]
    async fn only_the_host_that_removed_an_expired_one_shot_announces_it() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::At(Utc::now() - chrono::Duration::hours(30)),
                None,
                chrono::Duration::hours(30),
            )
            .await;
        let store = harness.manager.schedule_store();
        let now = Utc::now();

        assert!(
            store
                .retire_unclaimed(&job.id, job.next_fire_at, now)
                .await
                .expect("retire"),
            "the host whose delete removed the row is the one that announces it"
        );
        assert!(
            !store
                .retire_unclaimed(&job.id, job.next_fire_at, now)
                .await
                .expect("retire"),
            "and a host arriving afterwards stays quiet rather than repeating it"
        );
    }

    /// A parked job is not accused of crashing meka when nothing knows that it did.
    ///
    /// `attempts` is on the row; the reason it rose is in a process-global map. A restart is
    /// exactly what an operator does once a job has gone inert, and `meka schedule list` is a
    /// separate process that never had the map at all, so the commonest way to read this message
    /// is with the cause missing. Asserting the likelier cause from that absence told someone whose
    /// MCP server was misconfigured that their prompt takes meka down, with a remedy aimed at the
    /// wrong thing, in the model's own `[Scheduled]` block.
    ///
    /// The row still settles it one way: no gate means no probe that could have failed.
    #[tokio::test]
    async fn a_parked_job_is_only_called_a_crash_when_the_row_can_prove_it() {
        let harness = SchedulerHarness::new().await;
        let mut gated = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate("true", GatePredicate::Succeeded, None)),
                chrono::Duration::minutes(1),
            )
            .await;
        gated.attempts = MAX_CLAIM_ATTEMPTS;
        // A fresh id, so this process holds no record of why the claims failed -- which is the
        // state every reader is in after a restart.
        clear_probe_failure(&gated.id);

        let reported =
            job_withheld_reason(&gated, crate::permission::Permission::Unrestricted, None)
                .unwrap_or_default();
        assert!(
            !reported.contains("the host died"),
            "a gated job could have been parked by either cause, so neither may be asserted: \
             {reported}"
        );
        assert!(
            reported.contains("gate cannot be evaluated") && reported.contains("takes the host"),
            "both possibilities are named, so the operator knows what to check: {reported}"
        );

        let mut ungated = gated.clone();
        ungated.gate = None;
        let reported =
            job_withheld_reason(&ungated, crate::permission::Permission::Unrestricted, None)
                .unwrap_or_default();
        assert!(
            reported.contains("the host died"),
            "with no gate there is no probe that could have failed, so the crash can be named: \
             {reported}"
        );
    }

    /// A standing "this gate is broken" verdict retires itself once the row shows otherwise.
    ///
    /// The counter is per process and only the host that wins `claim_occurrence` ever touches it.
    /// `isolated` jobs are exempt from the session-residency filter and which host wins is a race
    /// between their tickers, so a second `meka serve` on the same store can take over every
    /// occurrence and heal the gate while this process's count stays where it stopped. The marker
    /// then stood forever: the model was told, every turn, that a job firing hourly was dead.
    ///
    /// Driven in one process rather than two. Advancing `last_fired_at` here is exactly what the
    /// other host's `complete_claim` writes, and the counter it has to convince lives in this
    /// process either way, so a second one would add wall-clock and no signal. It would add
    /// *coverage*, though: `GET /v1/schedule` does put this verdict on a wire
    /// (`server::handlers::jobs` renders it as `withheld`), so a two-host test is constructible
    /// and would exercise the convergence end to end rather than at the predicate.
    #[tokio::test]
    async fn a_probe_verdict_stands_down_when_another_host_has_evaluated_the_gate() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate("true", GatePredicate::Succeeded, None)),
                chrono::Duration::minutes(1),
            )
            .await;
        clear_probe_failure(&job.id);

        for _ in 0..PROBE_FAILURES_BEFORE_REPORTING {
            record_probe_failure(&job, "server said no such tool");
        }
        assert!(
            standing_probe_failure(&job).is_some(),
            "two failures with nothing to contradict them is a standing condition"
        );

        // What the other host's `complete_claim` writes when it fires the job. The failing path
        // writes neither this nor the baseline, so it cannot be this process's own doing.
        let mut fired = job.clone();
        fired.last_fired_at = Some(Utc::now());
        assert!(
            standing_probe_failure(&fired).is_none(),
            "the job has fired since the last failure was counted, so somebody evaluated this \
             gate and got an answer"
        );
        assert!(
            standing_probe_failure(&job).is_none(),
            "and the verdict is dropped rather than merely suppressed, so it does not come back \
             the next time this process reads the pre-fire row"
        );
    }

    /// A changed baseline is the other half: a gate that evaluates and declines never advances
    /// `last_fired_at`, but a successful evaluation still records what it saw.
    #[tokio::test]
    async fn a_probe_verdict_also_stands_down_on_a_baseline_another_host_recorded() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate("true", GatePredicate::Changed, Some("old"))),
                chrono::Duration::minutes(1),
            )
            .await;
        clear_probe_failure(&job.id);

        for _ in 0..PROBE_FAILURES_BEFORE_REPORTING {
            record_probe_failure(&job, "server said no such tool");
        }
        assert!(standing_probe_failure(&job).is_some());

        let mut evaluated = job.clone();
        if let Some(gate) = evaluated.gate.as_mut() {
            gate.last_output = Some("new".to_string());
        }
        assert!(
            standing_probe_failure(&evaluated).is_none(),
            "a baseline this process did not write means the gate answered somewhere else"
        );
    }

    /// Closing an occurrence that is not there any more is not the same as losing the lease.
    ///
    /// Both make the scoped write match nothing, and they mean opposite things. A job that fires
    /// and then cancels itself is an ordinary shape -- `schedule_create`'s own reply tells the
    /// model how -- and it was being told, on every such fire, that a duplicate delivery was
    /// possible and that an unrelated setting should be raised.
    #[tokio::test]
    async fn closing_an_occurrence_tells_a_cancelled_job_from_a_lost_lease() {
        let harness = SchedulerHarness::new().await;
        let store = harness.manager.schedule_store();
        let now = Utc::now();

        let cancelled = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(1),
            )
            .await;
        assert!(
            store
                .claim_occurrence(
                    &cancelled.id,
                    cancelled.next_fire_at,
                    "host-a",
                    now,
                    now + chrono::Duration::hours(1)
                )
                .await
                .expect("claim")
        );
        store
            .delete_scheduled_job(&cancelled.id)
            .await
            .expect("the model cancels the job during its own turn");
        assert_eq!(
            store
                .complete_claim(&cancelled.id, "host-a", Some(now), Some(now), None)
                .await
                .expect("complete"),
            ClaimClosed::RowGone,
            "there is no occurrence left to close, and nothing has gone wrong"
        );

        let taken = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(1),
            )
            .await;
        assert!(
            store
                .claim_occurrence(
                    &taken.id,
                    taken.next_fire_at,
                    "host-b",
                    now,
                    now + chrono::Duration::hours(1)
                )
                .await
                .expect("claim")
        );
        assert_eq!(
            store
                .complete_claim(&taken.id, "host-a", Some(now), Some(now), None)
                .await
                .expect("complete"),
            ClaimClosed::LeaseLost,
            "the row is there under someone else's claim, so this turn may be delivered again"
        );
    }

    /// A cancellation issued while a host holds the lease is not undone by the handback.
    ///
    /// This is the failure the lease exists for. Claiming a one-shot used to delete its row, so the
    /// handback was an `INSERT` that could not tell "I deleted this a moment ago" from "the user
    /// cancelled it in between" -- and put the job back either way, silently discarding the
    /// cancellation. A release is scoped to the claim, so it cannot recreate anything.
    #[tokio::test]
    async fn a_cancellation_during_a_claim_is_not_undone_by_the_handback() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::At(Utc::now() - chrono::Duration::minutes(1)),
                None,
                chrono::Duration::minutes(1),
            )
            .await;
        let store = harness.manager.schedule_store();
        let now = Utc::now();

        assert!(
            store
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "host-a",
                    now,
                    now + chrono::Duration::hours(1)
                )
                .await
                .expect("claim")
        );
        assert!(
            store
                .delete_scheduled_job(&job.id)
                .await
                .expect("the operator cancels it while the host works"),
            "the cancel removes a row that is still there, and says so"
        );

        store
            .release_claim(&job.id, "host-a")
            .await
            .expect("the host hands the occurrence back");

        assert!(
            harness.jobs().await.is_empty(),
            "a cancelled job stays cancelled: the handback may not resurrect it"
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
                    GatePredicate::Succeeded,
                    None,
                )),
                chrono::Duration::minutes(5),
            )
            .await;
        let job_next_fire_at = job.next_fire_at;

        // The other host gets there first, while this one is still holding the copy it listed.
        let now = Utc::now();
        assert!(
            harness
                .manager
                .schedule_store()
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "the-other-host",
                    now,
                    now + chrono::Duration::hours(1)
                )
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
            job_next_fire_at,
            "and the winner's row is where the winner left it"
        );
    }

    /// The one-shot half of a lost claim: the host that did not win must not deliver "remind me in
    /// 20 minutes" a second time.
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
        let now = Utc::now();
        assert!(
            harness
                .manager
                .schedule_store()
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "the-other-host",
                    now,
                    now + chrono::Duration::hours(1)
                )
                .await
                .expect("the other host claims")
        );

        let wakeup = prepare(&harness.manager, &harness.config, job, Utc::now())
            .await
            .expect("prepare runs");

        assert!(
            wakeup.is_none(),
            "the reminder belongs to the host holding the lease"
        );
    }

    /// A handback releases the occurrence *this* host holds, and nothing else.
    ///
    /// The restore used to be a whole-row upsert applied by id, so a host that lost the claim and
    /// was then refused the session lock overwrote the winner's `next_fire_at` with a time already
    /// in the past, and the job came due again on the very next tick while the winner was still
    /// running the turn. One hourly occurrence produced three gate runs and two agent turns that
    /// way.
    ///
    /// Scoping to the lease makes that structural rather than careful: the shape below is a host
    /// whose lease expired and was taken over while it was still working, which is the only way two
    /// hosts can now hold opinions about one job at once.
    #[tokio::test]
    async fn a_handback_does_not_reach_past_its_own_lease() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;
        let store = harness.manager.schedule_store();
        let long_ago = Utc::now() - chrono::Duration::hours(2);

        assert!(
            store
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "host-a",
                    long_ago,
                    long_ago + chrono::Duration::minutes(1),
                )
                .await
                .expect("claim"),
            "host A takes the occurrence, and then takes far too long"
        );
        let now = Utc::now();
        assert!(
            store
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "host-b",
                    now,
                    now + chrono::Duration::hours(1)
                )
                .await
                .expect("claim"),
            "host B finds the lease expired and takes it over"
        );

        // Host A finally gives up and hands back what it thinks it holds.
        store
            .release_claim(&job.id, "host-a")
            .await
            .expect("release");

        assert!(
            !store
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "host-c",
                    now,
                    now + chrono::Duration::hours(1)
                )
                .await
                .expect("claim"),
            "host B still holds it: a stale release must not free an occurrence someone else owns"
        );
    }

    /// The writes that come after a claim are scoped to the lease, and this is what that buys. A
    /// host whose lease expires while its gate is still running has been taken over; an unscoped
    /// write would then stamp this host's fire onto the new holder's row and drag the `changed`
    /// baseline back to a value already reported on.
    #[tokio::test]
    async fn test_a_late_write_does_not_land_on_another_hosts_occurrence() {
        let harness = SchedulerHarness::new().await;
        let planted = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate("echo state", GatePredicate::Changed, None)),
                chrono::Duration::minutes(5),
            )
            .await;
        let store = harness.manager.schedule_store();

        let long_ago = Utc::now() - chrono::Duration::hours(2);
        assert!(
            store
                .claim_occurrence(
                    &planted.id,
                    planted.next_fire_at,
                    "host-a",
                    long_ago,
                    long_ago + chrono::Duration::minutes(1),
                )
                .await
                .expect("claim")
        );
        // Taken over by another host while this one's gate is still running.
        let now = Utc::now();
        assert!(
            store
                .claim_occurrence(
                    &planted.id,
                    planted.next_fire_at,
                    "host-b",
                    now,
                    now + chrono::Duration::hours(1)
                )
                .await
                .expect("the other host takes the expired lease")
        );
        store
            .complete_claim(
                &planted.id,
                "host-b",
                Some(planted.next_fire_at + chrono::Duration::hours(1)),
                None,
                Some("theirs"),
            )
            .await
            .expect("the other host finishes and records its baseline");

        // Host A finally finishes, and writes against the lease it thinks it holds.
        store
            .complete_claim(
                &planted.id,
                "host-a",
                Some(planted.next_fire_at + chrono::Duration::hours(2)),
                Some(Utc::now()),
                Some("ours"),
            )
            .await
            .expect("complete");

        let jobs = harness.jobs().await;
        let job = jobs.first().expect("job survives");
        assert_eq!(
            job.gate
                .as_ref()
                .and_then(|gate| gate.last_output.as_deref()),
            Some("theirs"),
            "a late baseline must not overwrite the one the lease holder recorded"
        );
        assert!(
            job.last_fired_at.is_none(),
            "and a late completion must not land on a row another host owns"
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
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "host-a",
                    Utc::now(),
                    Utc::now() + chrono::Duration::hours(1)
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
    /// completion at the end of the sweep changed nothing any test could see
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

        // Through a whole sweep rather than `prepare` alone, because the schedule is advanced and
        // the fire recorded once the turn has actually been delivered. That ordering is the point
        // of the lease: a crash before delivery costs a retry rather than the occurrence.
        harness.tick().await;

        assert_eq!(
            harness.fired().len(),
            1,
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
        assert_eq!(
            (job.attempts, planted.attempts),
            (0, 0),
            "and a delivered occurrence clears the crash count rather than accumulating one"
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
