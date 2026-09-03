//! `meka schedule`: inspecting and cancelling the wakeups the agent scheduled for itself.
//!
//! Read-and-cancel only. Creating a job needs a session for the resulting turn to run in, and
//! picking one from the command line would mean guessing which conversation the user meant; the
//! agent creates jobs through `schedule_create`, where the session is unambiguous.

use crate::{
    cli::ScheduleAction,
    error::{MekaError, Result},
    session::SessionManager,
};

/// Ceiling on the rendered schedule column. `format_columns` widens a column to its longest cell,
/// so an unbounded value here would push every following column off the terminal.
const SCHEDULE_TRUNCATE: usize = 24;
/// Same, for the trailing prompt, and the column that spends whatever the others leave.
///
/// The table is a width budget rather than a set of independent ceilings, which is how the previous
/// seven columns reached 186 on a real store against the 120 meka targets. Worst case is `8 + 2`
/// id, `8 + 2` session, `24 + 2` schedule, `8 + 2` next (`999d 23h`, which is what
/// [`NEVER_SOON_DAYS`] exists to keep it under), `5 + 2` gate, and this: 120 exactly. An id widened
/// by [`crate::render::unique_prefix_len_within`] is taken back off this, so a collision costs
/// prompt rather than the budget.
const PROMPT_TRUNCATE: usize = 57;

/// Floor on the prompt, so two widened ids cannot squeeze it to nothing.
///
/// Both ids widening to a full UUID leaves `57 - 28 - 28 = 1`, and a one-column prompt says less
/// than no column would. Overrunning the budget in that case is the better trade: it takes a
/// deliberate pair of colliding ids to reach, and the same floor is why the other two tables have
/// one.
const PROMPT_MINIMUM: usize = 16;

/// What a row is allowed to occupy, which the constants above add up to exactly.
///
/// Only the test asserts it: the widths are chosen to sum to this rather than derived from it, so
/// naming it in the production path would imply a division of it that does not happen.
#[cfg(test)]
const TABLE_BUDGET: usize = 120;

/// Where the `Next` column stops counting and starts saying "not soon".
///
/// Chosen so the cell fits the `8 + 2` the budget above reserves for it. `format_duration_short`
/// renders `{days}d {hours}h`, which is nine columns once the day count reaches four digits and the
/// hour two -- so the clamp has to land before four digits, not at them. `999d 23h` is eight and
/// `>999d` is five.
const NEVER_SOON_DAYS: i64 = 999;
const NEVER_SOON_SECONDS: i64 = NEVER_SOON_DAYS * 24 * 60 * 60;

pub async fn run(
    session_manager: &SessionManager,
    action: &ScheduleAction,
    cli_args: &crate::cli::Cli,
) -> anyhow::Result<()> {
    match action {
        ScheduleAction::List { session } => {
            // Read for its refusal, not its contents: this listing answers about jobs that a
            // differently-configured host will run, and a table rendered over an unreadable config
            // is a listing of what may already be wrong.
            crate::config::ResolvedConfig::from_cli(cli_args).require_readable_config()?;
            list(session_manager, session.as_deref()).await
        }
        ScheduleAction::Show { id } => {
            // Contents as well as refusal here, since `withheld` is decided against
            // `[permissions].enabled` and answering off defaults would report a job as able to fire
            // that the running host declines on every sweep.
            let config = crate::config::ResolvedConfig::from_cli(cli_args);
            config.require_readable_config()?;
            show(session_manager, id, &config.schedule, None).await
        }
        ScheduleAction::Cancel { id } => cancel(session_manager, id).await,
    }
    .map_err(Into::into)
}

/// `/schedule` in the REPL: the same table, scoped to the conversation the user is in.
///
/// Trades the `Session` column, which would carry one id on every row here, for `Held`. This
/// process *is* a host, so it hands the renderer its gate dispatcher and can answer a tool gate;
/// `meka schedule list` cannot, and is the surface where a spent column is worst.
pub async fn run_list_for_session(
    session_manager: &SessionManager,
    session: uuid::Uuid,
    config: &crate::config::ResolvedScheduleConfig,
) -> Result<()> {
    let jobs = session_manager
        .schedule_store()
        .list_scheduled_jobs(session)
        .await?;
    let jobs = with_levels(session_manager, config, jobs).await;
    render(
        &jobs,
        Layout::SessionScoped,
        config.gate_tools.as_deref(),
        &Resolvable::of(&jobs),
    )
}

/// Pair each job with the level its session holds now, which is what decides whether it can fire.
///
/// One row read per job, and only for a command the user typed.
///
/// Clamped by the enabled set, like the five other readers of that column and like [`prepare`]. A
/// row records what a session was *set* to, not what this installation still permits, and the two
/// diverge the moment an operator narrows `[permissions].enabled`: the fire door reads the clamped
/// level and refuses, while this read saw the unclamped row and rendered the job as healthy.
///
/// A level that cannot be established at all leaves this `None`, which [`withheld_summary`] reports
/// as unknown. The host level is deliberately *not* used as a fallback here the way the fire door
/// uses it: the host that eventually runs this job is some other process, and its `--permission` is
/// not something this command can see. "I cannot tell" is the honest answer.
///
/// [`prepare`]: crate::schedule
async fn with_levels(
    session_manager: &SessionManager,
    config: &crate::config::ResolvedScheduleConfig,
    jobs: Vec<crate::schedule::ScheduledJob>,
) -> Vec<(
    crate::schedule::ScheduledJob,
    Option<crate::permission::Permission>,
)> {
    let mut out = Vec::with_capacity(jobs.len());
    for job in jobs {
        let level = match session_manager.session_info(job.session_id).await {
            Ok(info) => info
                .and_then(|info| {
                    crate::permission::parse_recorded_permission(
                        info.permission.as_deref(),
                        &format_args!("session {}", job.session_id),
                    )
                })
                .filter(|level| config.enabled_permissions.is_enabled(*level)),
            Err(error) => {
                tracing::debug!("could not read session {}: {}", job.session_id, error);
                None
            }
        };
        out.push((job, level));
    }
    out
}

async fn list(session_manager: &SessionManager, session: Option<&str>) -> Result<()> {
    let jobs = match session {
        // A prefix, as `--session 0b5c...` in this command's own help has always advertised and as
        // `meka -c` accepts. It parsed a whole UUID and rejected everything shorter, which is also
        // the form the `Session` column prints.
        Some(raw) => {
            let id = crate::resolve_session_id(session_manager, raw)
                .await
                .map_err(|error| MekaError::Config(error.to_string()))?;
            session_manager
                .schedule_store()
                .list_scheduled_jobs(id)
                .await?
        }
        None => {
            session_manager
                .schedule_store()
                .list_all_scheduled_jobs()
                .await?
        }
    };

    // No levels read: the unscoped table has no `Held` column to spend them on, and reading one
    // session row per job across every session is not free.
    let jobs: Vec<_> = jobs.into_iter().map(|job| (job, None)).collect();
    // Every job and every session, not the rows above. `--session` narrows this listing while
    // `show` and `cancel` still scan the whole store, so an id sized to the filtered rows is one
    // they would refuse. Read only when the listing is filtered, since otherwise it is the rows.
    let resolvable = match session {
        Some(_) => Resolvable {
            jobs: session_manager
                .schedule_store()
                .list_all_scheduled_jobs()
                .await?
                .into_iter()
                .map(|job| job.id)
                .collect(),
            sessions: session_manager.all_session_ids().await?,
        },
        None => Resolvable::of(&jobs),
    };
    render(&jobs, Layout::Unscoped, None, &resolvable)
}

/// The table's header, beside the row builder so the two cannot drift apart in width or order.
///
/// `Session` only where it can differ. `meka schedule list` spans every session, so a row without
/// it names no owner; `/schedule` in the REPL is already scoped to one conversation, where the
/// column would repeat the same id on every line.
const COLUMNS: [&str; 6] = ["ID", "Session", "Schedule", "Next", "Gate", "Prompt"];

/// [`COLUMNS`] with `Held` in place of `Session`: one is unanswerable on the surface that has room
/// for the other. It sits last-but-one rather than second, because dropping a column shifts the row
/// rather than blanking a cell, and `Held` reads beside the gate it is a verdict about. Narrower
/// than the column it replaces, so this table ends inside the budget the other one exactly fills.
const SESSION_SCOPED_COLUMNS: [&str; 6] = ["ID", "Schedule", "Next", "Gate", "Held", "Prompt"];

/// Which of the two tables is being built.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Layout {
    /// `meka schedule list`: every session, so each row must name its own.
    Unscoped,
    /// `/schedule`: one session, and a host that can resolve a gate.
    SessionScoped,
}

/// What a printed prefix will be resolved against: every job, and every session that owns one.
///
/// Not the rows being drawn. `meka schedule list --session <id>` shows one conversation's jobs
/// while `show` and `cancel` scan all of them, so a width sized to the rows printed an id those
/// commands then refused as ambiguous.
struct Resolvable {
    jobs: Vec<String>,
    sessions: Vec<String>,
}

impl Resolvable {
    /// The rows as their own universe, which is the truth for an unfiltered listing and what a
    /// test about layout wants.
    fn of(
        jobs: &[(
            crate::schedule::ScheduledJob,
            Option<crate::permission::Permission>,
        )],
    ) -> Self {
        Self {
            jobs: jobs.iter().map(|(job, _)| job.id.clone()).collect(),
            sessions: jobs
                .iter()
                .map(|(job, _)| job.session_id.to_string())
                .collect(),
        }
    }
}

/// One row per job, separated from printing so it can be tested.
///
/// The `Prompt` cell carries text meka did not write, possibly composed from a page the agent
/// fetched, so it goes through [`crate::render::sanitize_to_line`]. The separation from `render`
/// matters for exactly that: `render` prints, so sanitising done there has no way to be asserted.
///
/// `Schedule` goes through the same helper without ever having been a way in: everything
/// `Schedule::describe` can emit is a formatted timestamp, a `humantime` duration, or a cron
/// pattern, and croner's five-field grammar admits no control characters at either the creation or
/// the rehydration door. It is sanitised so the guarantee this function makes does not quietly
/// depend on a parser two modules away staying that strict. The rest of the row is an id, a
/// duration, and a word from an enum.
fn rows_for(
    jobs: &[(
        crate::schedule::ScheduledJob,
        Option<crate::permission::Permission>,
    )],
    layout: Layout,
    tools: Option<&dyn crate::schedule::GateTools>,
    resolvable: &Resolvable,
) -> Vec<Vec<String>> {
    let now = chrono::Utc::now();
    // Widened only as far as resolving requires, and paid for out of the prompt so the table holds
    // its budget either way. Sized against `resolvable` rather than the rows, because
    // `--session` narrows the listing while `show` and `cancel` still scan every job.
    let job_width = crate::render::unique_prefix_len_within(
        jobs.iter().map(|(job, _)| job.id.as_str()),
        resolvable.jobs.iter().map(String::as_str),
    )
    .max(crate::render::ID_PREFIX);
    let sessions: Vec<String> = jobs
        .iter()
        .map(|(job, _)| job.session_id.to_string())
        .collect();
    let session_width = crate::render::unique_prefix_len_within(
        sessions.iter().map(String::as_str),
        resolvable.sessions.iter().map(String::as_str),
    )
    .max(crate::render::ID_PREFIX);
    let prompt_width = PROMPT_TRUNCATE
        .saturating_sub(job_width.saturating_sub(crate::render::ID_PREFIX))
        .saturating_sub(match layout {
            Layout::Unscoped => session_width.saturating_sub(crate::render::ID_PREFIX),
            Layout::SessionScoped => 0,
        })
        .max(PROMPT_MINIMUM);

    jobs.iter()
        .zip(sessions.iter())
        .map(|((job, level), session)| {
            let mut row = vec![job.id.get(..job_width).unwrap_or(&job.id).to_string()];
            if layout == Layout::Unscoped {
                row.push(session.get(..session_width).unwrap_or(session).to_string());
            }
            row.push(crate::render::sanitize_to_line(
                &job.schedule.describe(),
                SCHEDULE_TRUNCATE,
            ));
            // Relative, because the absolute instant cost 23 columns to answer a question a reader
            // asks in the relative form. An occurrence already due reads as `due` rather than as
            // `0s`: a host that was down through it has not missed it, and the two are different
            // states. `show` carries the instant.
            row.push(match (job.next_fire_at - now).num_seconds() {
                seconds if seconds <= 0 => "due".to_string(),
                // Bounded, unlike a duration rendered straight. `Schedule::parse_at` takes any RFC
                // 3339 instant, so a job dated 3121 reads `399999d 23h` and spends 11 of the 8
                // columns this table budgeted for it, pushing the row past 120. Nobody reads a
                // four-digit day count as anything but "not soon", and `show` carries the instant.
                seconds if seconds >= NEVER_SOON_SECONDS => format!(">{}d", NEVER_SOON_DAYS),
                seconds => crate::render::format_duration_short(seconds),
            });
            // The kind alone. What the gate *runs* is model-authored text that no column can hold
            // legibly -- a real one truncated to `shell bash -c 'S=/home/mica/backup-st...`, which
            // does not answer the question that column existed to answer. `shell` against `tool` is
            // the distinction worth a column: an unsandboxed `sh -c` against a structured call. A
            // fixed word from an enum, so unlike the summary it replaces it needs no sanitising.
            row.push(match &job.gate {
                Some(gate) => gate.probe.kind_str().to_string(),
                None => "-".to_string(),
            });
            // Three answers, and blank is one of them: a job that will fire says nothing, so the
            // column is quiet on the common case and speaks on the two that need attention.
            if layout == Layout::SessionScoped {
                row.push(match crate::schedule::job_withheld(job, *level, tools) {
                    crate::schedule::Withheld::Yes(_) => "yes".to_string(),
                    crate::schedule::Withheld::Undetermined => "?".to_string(),
                    crate::schedule::Withheld::No => String::new(),
                });
            }
            // Whitespace collapsed first, then sanitised. The collapse is for legibility -- a
            // prompt is prose and wraps -- and it is not a safety measure: `\u{1b}` is not
            // whitespace, so it survived a `split_whitespace` that looked like it was cleaning
            // the cell.
            row.push(crate::render::sanitize_to_line(
                &job.prompt.split_whitespace().collect::<Vec<_>>().join(" "),
                prompt_width,
            ));
            row
        })
        .collect()
}

fn render(
    jobs: &[(
        crate::schedule::ScheduledJob,
        Option<crate::permission::Permission>,
    )],
    layout: Layout,
    tools: Option<&dyn crate::schedule::GateTools>,
    resolvable: &Resolvable,
) -> Result<()> {
    if jobs.is_empty() {
        // stderr: an empty list is a status note, not the data a script asked for. A caller piping
        // this still gets a clean empty stdout.
        crate::render::write_stderr_line("No scheduled jobs.");
        return Ok(());
    }

    let headers: &[&str] = match layout {
        Layout::Unscoped => &COLUMNS,
        Layout::SessionScoped => &SESSION_SCOPED_COLUMNS,
    };
    crate::render::write_stdout(crate::render::format_columns(
        headers,
        &rows_for(jobs, layout, tools, resolvable),
    ))?;
    Ok(())
}

/// `meka schedule show <id>`: one job, in full.
///
/// The table exists to be scanned, so every cell in it is bounded. This is the surface that answers
/// what a job actually does -- the whole prompt, the whole command a gate runs, the session it
/// wakes -- none of which survives a column. It is also the only place `Held` is explained rather
/// than encoded, which matters because a withheld job is one that will never fire.
pub(crate) async fn show(
    session_manager: &SessionManager,
    id_prefix: &str,
    config: &crate::config::ResolvedScheduleConfig,
    scope: Option<uuid::Uuid>,
) -> Result<()> {
    let job = resolve_job(session_manager, id_prefix, scope).await?;
    let level = with_levels(session_manager, config, vec![job]).await;
    let Some((job, level)) = level.into_iter().next() else {
        return Err(MekaError::Config(format!(
            "scheduled job '{}' was already gone",
            id_prefix
        )));
    };

    let mut out = String::new();
    // A free function rather than a closure: the untruncated prompt below appends to `out`
    // directly, and a closure holding it borrowed would rule that out.
    fn field(out: &mut String, name: &str, value: &str) {
        use std::fmt::Write as _;
        // The width is the longest label plus one, so the values line up without a table.
        if value.is_empty() {
            // A heading rather than a field, so padding it to the column would leave a run of
            // trailing spaces on the row. Same rule as `/tasks show`'s `result:`.
            writeln!(out, "{}:", name).ok();
        } else {
            writeln!(out, "{:<12} {}", format!("{}:", name), value).ok();
        }
    }

    field(&mut out, "id", &job.id);
    field(&mut out, "session", &job.session_id.to_string());
    field(
        &mut out,
        "schedule",
        &sanitize_untruncated(&job.schedule.describe()),
    );
    field(
        &mut out,
        "next fire",
        &job.next_fire_at
            .with_timezone(&chrono::Local)
            .format("%Y-%m-%d %H:%M %Z")
            .to_string(),
    );
    field(&mut out, "last fired", &match job.last_fired_at {
        Some(at) => at
            .with_timezone(&chrono::Local)
            .format("%Y-%m-%d %H:%M %Z")
            .to_string(),
        None => "never".to_string(),
    });
    field(
        &mut out,
        "created",
        &job.created_at
            .with_timezone(&chrono::Local)
            .format("%Y-%m-%d %H:%M %Z")
            .to_string(),
    );
    // Every value below is model-authored or server-chosen, so all of it is sanitised. Untruncated
    // is the point of this command; `sanitize_to_line` still collapses anything that could forge a
    // line, which is the part that was never about width.
    field(
        &mut out,
        "withheld",
        withheld_summary(&job, level, config.gate_tools.as_deref()),
    );
    match &job.gate {
        Some(gate) => {
            field(
                &mut out,
                "gate when",
                &sanitize_untruncated(&gate.predicate.summary()),
            );
            field(
                &mut out,
                "gate check",
                &format!(
                    "{} {}",
                    gate.probe.kind_str(),
                    sanitize_untruncated(&gate.probe.summary())
                ),
            );
        }
        None => field(&mut out, "gate", "none -- fires on schedule"),
    }
    // Line structure preserved and indented, as `/tasks show` does for a task's output: a job's
    // prompt is prose the model wrote for its own future self and is routinely multi-line, and the
    // indent is what stops a line in it reading as another field.
    field(&mut out, "prompt", "");
    for line in job.prompt.lines() {
        out.push_str("  ");
        out.push_str(&sanitize_untruncated(line));
        out.push('\n');
    }

    crate::render::write_stdout(&out)?;
    Ok(())
}

/// Whether a job can fire, spelled out.
///
/// Three answers, because this command has no MCP manager and so cannot resolve a tool gate at all.
/// The table this replaced encoded them as `yes` / `?` / blank, where blank was a verdict that read
/// as missing data. With a line to itself each one can say which it is.
fn withheld_summary(
    job: &crate::schedule::ScheduledJob,
    level: Option<crate::permission::Permission>,
    tools: Option<&dyn crate::schedule::GateTools>,
) -> &'static str {
    match crate::schedule::job_withheld(job, level, tools) {
        crate::schedule::Withheld::Yes(_) => "yes -- this job cannot fire",
        // `Undetermined` has two causes and they send an operator to different places, so the one
        // this reader is in is named rather than the commoner one guessed at. A missing level is
        // its own answer: the row records no permission, or the one it records is not in
        // `[permissions] enabled` any more, which is the case that makes a job stop firing after a
        // config change and has nothing to do with gates.
        crate::schedule::Withheld::Undetermined if level.is_none() => {
            "unknown -- this session's permission level could not be established"
        }
        crate::schedule::Withheld::Undetermined => {
            "unknown -- this process cannot resolve a tool gate"
        }
        crate::schedule::Withheld::No => "no",
    }
}

/// Sanitise without a ceiling, for [`show`], whose whole job is to not truncate.
fn sanitize_untruncated(text: &str) -> String {
    crate::render::sanitize_to_line(text, usize::MAX)
}

/// Find the one job whose id starts with `id_prefix`, or say why not.
///
/// `scope` is `None` for the CLI, which scans every session because the operator has an id, not a
/// session, and requiring them to find the session first would make the id useless on its own. It
/// is `Some` for `/schedule show`, which answers inside one conversation like everything else on
/// that surface. `show` and `cancel` share this rather than each filtering the listing themselves:
/// a resolver that only one of them guards is a destructive command resolving ids the other
/// refuses.
async fn resolve_job(
    session_manager: &SessionManager,
    id_prefix: &str,
    scope: Option<uuid::Uuid>,
) -> Result<crate::schedule::ScheduledJob> {
    let no_match = || {
        Err(MekaError::Config(format!(
            "no scheduled job matching '{}'",
            id_prefix
        )))
    };
    if !crate::render::is_usable_id_prefix(id_prefix) {
        return no_match();
    }
    let jobs = match scope {
        // `/schedule` in the REPL is one conversation's view, and its `cancel` resolves against
        // that session alone. A `show` that reached outside it would answer about a job the
        // listing above it never mentioned and the `cancel` beside it would refuse.
        Some(session) => {
            session_manager
                .schedule_store()
                .list_scheduled_jobs(session)
                .await?
        }
        None => {
            session_manager
                .schedule_store()
                .list_all_scheduled_jobs()
                .await?
        }
    };
    let wanted = crate::render::id_prefix_for_matching(id_prefix);
    let mut matches = jobs.into_iter().filter(|job| job.id.starts_with(&wanted));
    match (matches.next(), matches.next()) {
        (None, _) => no_match(),
        (Some(job), None) => Ok(job),
        // Full ids, as `resolve_session_id` reports: every match shares the prefix, so echoing it
        // back twice names nothing the caller could retype.
        (Some(first), Some(second)) => Err(MekaError::Config(format!(
            "ambiguous job id '{}' matches at least: {}, {}",
            id_prefix, first.id, second.id
        ))),
    }
}

async fn cancel(session_manager: &SessionManager, id_prefix: &str) -> Result<()> {
    let job = resolve_job(session_manager, id_prefix, None).await?;
    // Reported from the delete's own row count, not from the listing that found the job: a
    // scheduler sweep can retire it in between, and `ok: cancelled` about a job this command did
    // not cancel is indistinguishable from the real thing.
    match session_manager
        .schedule_store()
        .delete_scheduled_job(&job.id)
        .await?
    {
        true => {
            tracing::info!("cancelled scheduled job {}", job.id);
            Ok(())
        }
        false => Err(MekaError::Config(format!(
            "scheduled job '{}' was already gone",
            job.short_id()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        permission::Permission,
        schedule::{Gate, GatePredicate, GateProbe, Schedule, ScheduledJob},
    };

    /// `/schedule show` answers inside the conversation, like everything else on that surface.
    ///
    /// `/schedule` lists one session's jobs and `/schedule cancel` resolves against that session
    /// alone. A `show` that scanned every session would print a job the listing above it never
    /// mentioned and the `cancel` beside it would refuse -- and would leak another conversation's
    /// prompt and gate command into this one. The CLI passes `None` and does scan everything,
    /// because there the operator has an id and no session.
    #[tokio::test]
    async fn the_repl_resolves_a_job_only_inside_its_own_session() {
        let manager = crate::session::SessionManager::open(
            Some(std::path::Path::new(":memory:")),
            &crate::session::migrations::Context::adopting(None),
        )
        .await
        .expect("open");
        let mine = manager
            .create_session(None, "p".to_string())
            .await
            .expect("session");
        let theirs = manager
            .create_session(None, "p".to_string())
            .await
            .expect("session");

        let mut job = job_with(None);
        job.session_id = theirs;
        manager
            .schedule_store()
            .create_scheduled_job(&job)
            .await
            .expect("create");

        assert!(
            resolve_job(&manager, &job.id[..8], Some(mine))
                .await
                .is_err(),
            "another session's job must not resolve here"
        );
        assert_eq!(
            resolve_job(&manager, &job.id[..8], Some(theirs))
                .await
                .expect("its own session finds it")
                .id,
            job.id
        );
        assert_eq!(
            resolve_job(&manager, &job.id[..8], None)
                .await
                .expect("the CLI scans every session")
                .id,
            job.id
        );
    }

    fn job_with(gate: Option<Gate>) -> ScheduledJob {
        ScheduledJob {
            id: "7f3a1b2c-0000-0000-0000-000000000000".to_string(),
            session_id: uuid::Uuid::nil(),
            schedule: Schedule::parse_every("1h").expect("parses"),
            prompt: "watch the thing".to_string(),
            gate,
            created_at: chrono::Utc::now(),
            last_fired_at: None,
            next_fire_at: chrono::Utc::now(),
            attempts: 0,
        }
    }

    fn shell_gate(command: &str, predicate: GatePredicate) -> Gate {
        Gate {
            probe: GateProbe::Shell {
                command: command.to_string(),
            },
            predicate,
            last_output: None,
            permission: Permission::Unrestricted,
        }
    }

    /// Nothing the model wrote reaches the terminal as control characters or as extra rows.
    ///
    /// Four cells carry model-authored text: a shell command, a tool name an MCP server chose, a
    /// regular expression or JSON pointer the predicate gained with `matches` and `at`, and the
    /// prompt itself. The `When` column was a fixed enum word until then and was left unsanitised
    /// in the change that widened it, which is exactly the kind of regression a listing that only
    /// ever printed could not be asked about.
    ///
    /// An escape goes into every one of them. Looping over `row` while planting in only two is how
    /// the `Prompt` cell stayed unsanitised through a test that reads as though it covered
    /// everything: the loop was honest and the corpus was not.
    #[test]
    fn no_cell_can_carry_an_escape_or_forge_a_row() {
        let forged = "/chats\u{1b}[31m\nffffffff  every 1m  now  changed  -  -  harmless";
        let mut job = job_with(Some(shell_gate(
            "gh pr checks\u{1b}[31m\nsecond line",
            GatePredicate::At {
                pointer: forged.to_string(),
                is: crate::schedule::PointerTest::NotEmpty,
            },
        )));
        // A prompt is prose, and an agent that composed one from a page it fetched is the ordinary
        // way something hostile arrives in this column.
        job.prompt =
            "watch the thing\u{1b}[31m\nffffffff  every 1m  now  -  -  -  harmless".to_string();
        let rows = rows_for(&[(job, None)], Layout::Unscoped, None, &Resolvable::of(&[]));

        let [row] = rows.as_slice() else {
            panic!("one job, one row: {rows:?}");
        };
        for cell in row {
            assert!(
                !cell.contains('\u{1b}') && !cell.contains('\n'),
                "a cell reaches a terminal verbatim, so neither may survive: {cell:?}"
            );
        }
        assert!(
            row[2].chars().count() <= SCHEDULE_TRUNCATE,
            "and the schedule is bounded, or one long spec pushes every later column off the \
             screen: {:?}",
            row[2]
        );
        assert!(
            row[5].chars().count() <= PROMPT_TRUNCATE,
            "the prompt too, since a paragraph would wrap the row into unreadability: {:?}",
            row[5]
        );
    }

    /// The recorded level is clamped by the enabled set before anything is concluded from it.
    ///
    /// Where the fire door reads a session's recorded level it also applies `[permissions].enabled`
    /// and refuses what falls outside it. This listing did not, so narrowing the enabled set and
    /// restarting left the operator's own table reporting a job as healthy that the running host
    /// declines on every sweep. Exercised through `with_levels` rather than `rows_for`, because
    /// `rows_for` is handed a level that has already been through this and would pass either way:
    /// deleting the filter left all 2452 tests green.
    #[tokio::test]
    async fn a_level_the_enabled_set_excludes_is_not_treated_as_the_session_level() {
        let manager = crate::session::SessionManager::open(
            Some(std::path::Path::new(":memory:")),
            &Default::default(),
        )
        .await
        .expect("open in-memory database");
        let session = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        manager
            .update_session_permission(session, "unrestricted")
            .await
            .expect("record the level the session was set to");

        let mut job = job_with(None);
        job.session_id = session;

        let permissive = crate::config::ResolvedScheduleConfig::default();
        assert_eq!(
            with_levels(&manager, &permissive, vec![job.clone()]).await[0].1,
            Some(Permission::Unrestricted),
            "with the default enabled set the recorded level is the answer"
        );

        let narrowed = crate::config::ResolvedScheduleConfig {
            enabled_permissions: crate::permission::EnabledPermissions::from_modes([
                Permission::Read,
            ])
            .expect("a single mode is a non-empty set"),
            ..crate::config::ResolvedScheduleConfig::default()
        };
        assert_eq!(
            with_levels(&manager, &narrowed, vec![job]).await[0].1,
            None,
            "once the installation stops permitting that level the row no longer establishes one, \
             so the column says so instead of reporting the job as healthy"
        );
    }

    /// A verdict that needs no permission level is still reached without one.
    ///
    /// Parking is decided by the attempt count alone. Matching on the level first and answering
    /// `?` when it was absent made the one surface an operator opens to find out why a job stopped
    /// decline to say, about the one job it could have answered for outright.
    #[test]
    fn a_parked_job_is_reported_withheld_even_with_no_session_level() {
        let mut job = job_with(None);
        job.attempts = crate::schedule::MAX_CLAIM_ATTEMPTS;
        assert!(
            withheld_summary(&job, None, None).starts_with("yes"),
            "a parked job is provably dead whatever the session is set to"
        );
    }

    /// A level this command could not establish must not read as one that permits the job.
    ///
    /// Two ways to get there, and the second is why the enabled set is applied here at all: the row
    /// carries no level, or it carries one this installation no longer permits. The fire door
    /// clamps by `[permissions].enabled` and refuses; this listing did not, and reported the job as
    /// healthy to the one audience that came here to check.
    #[test]
    fn an_unestablished_level_is_not_reported_as_healthy() {
        let unknown = withheld_summary(
            &job_with(Some(shell_gate("gh pr checks", GatePredicate::Changed))),
            None,
            None,
        );
        assert!(
            unknown.starts_with("unknown"),
            "a session whose level could not be read is unestablished, not healthy: {unknown}"
        );
    }

    /// The verdict answers three things, and each says which it is.
    ///
    /// This listing has no MCP manager, so it cannot resolve a tool gate at all. Rendering that as
    /// blank put "I cannot tell" in the same cell as "it will fire", under a header that reads as
    /// the latter.
    #[test]
    fn the_verdict_distinguishes_cannot_fire_from_cannot_tell() {
        let held = withheld_summary(
            &job_with(Some(shell_gate("gh pr checks", GatePredicate::Changed))),
            Some(Permission::Read),
            None,
        );
        assert!(
            held.starts_with("yes"),
            "a shell gate below `unrestricted` cannot fire: {held}"
        );

        let undetermined = withheld_summary(
            &job_with(Some(Gate {
                probe: GateProbe::Tool {
                    name: "mcp__bridge__unseen".to_string(),
                    arguments: serde_json::json!({}),
                },
                predicate: GatePredicate::Changed,
                last_output: None,
                permission: Permission::Read,
            })),
            Some(Permission::Read),
            None,
        );
        assert!(
            undetermined.starts_with("unknown"),
            "a tool gate this process cannot resolve is unestablished, not healthy: {undetermined}"
        );

        let healthy = withheld_summary(
            &job_with(Some(shell_gate("gh pr checks", GatePredicate::Changed))),
            Some(Permission::Unrestricted),
            None,
        );
        assert_eq!(healthy, "no", "and an authorised gate is not withheld");
    }

    /// A tool name is also a valid command, so the kind has to be on the row.
    ///
    /// The column carries the kind and nothing else now, which is what makes it 5 columns wide
    /// instead of 40. The distinction it exists for survives that: an unsandboxed `sh -c` and a
    /// structured call still cannot read alike. What each one *runs* is [`show`]'s job.
    /// The `Held` cell distinguishes a verdict from a shrug.
    ///
    /// `/schedule` is the only surface that renders this cell, and its three values mean different
    /// things: `yes` is "this job cannot fire", blank is "it can", and `?` is "this process cannot
    /// establish which". Blank and `?` are the pair worth pinning, because swapping them turns "I
    /// did not check" into "it will fire" on the surface an operator uses to find out why a job is
    /// silent. The tests that covered this moved to `withheld_summary`, which only `show` calls.
    #[test]
    fn the_held_cell_separates_a_verdict_from_an_unanswerable_question() {
        let held_cell = |gate: Option<Gate>, level: Option<Permission>| {
            let jobs = [(job_with(gate), level)];
            rows_for(&jobs, Layout::SessionScoped, None, &Resolvable::of(&jobs))[0][4].clone()
        };

        assert_eq!(
            held_cell(
                Some(shell_gate("gh pr checks", GatePredicate::Changed)),
                Some(Permission::Read)
            ),
            "yes",
            "a shell gate below `unrestricted` cannot fire, and the column says so"
        );
        assert_eq!(
            held_cell(
                Some(shell_gate("gh pr checks", GatePredicate::Changed)),
                Some(Permission::Unrestricted)
            ),
            "",
            "an authorised gate will fire, and blank means that rather than `not checked`"
        );
        assert_eq!(
            held_cell(
                Some(shell_gate("gh pr checks", GatePredicate::Changed)),
                None
            ),
            "?",
            "a level that could not be established is unanswerable, not healthy"
        );
    }

    /// The table holds 120 columns against the widest cell every column can produce.
    ///
    /// The other two listings got this test and this one did not, which is where the `Next` overrun
    /// hid: `format_duration_short` renders `{days}d {hours}h`, nine columns once the days reach
    /// four digits, against the eight the budget reserves. Sizing each cell to its own ceiling and
    /// never measuring the row is how a table that fits on every real store overruns on one job.
    #[test]
    fn the_table_holds_its_budget_against_the_widest_row_it_can_draw() {
        let mut job = job_with(Some(shell_gate("gh pr checks", GatePredicate::Changed)));
        job.prompt = "x".repeat(400);
        job.schedule = Schedule::parse_every("31536000s").expect("parses");
        // A horizon past the clamp, and one just under it: the second is the wider cell.
        // The middle one is the case that overran: four-digit days with a two-digit hour renders
        // `9990d 14h`, nine columns against the eight the budget reserves. It is only reachable
        // below a clamp set at four digits, which is why the clamp sits at three.
        for offset in [
            chrono::Duration::days(400_000),
            chrono::Duration::days(9_990) + chrono::Duration::hours(14),
            chrono::Duration::days(999) + chrono::Duration::hours(23),
        ] {
            let mut job = job.clone();
            job.next_fire_at = chrono::Utc::now() + offset;
            for layout in [Layout::Unscoped, Layout::SessionScoped] {
                let jobs = [(job.clone(), Some(Permission::Read))];
                let rows = rows_for(&jobs, layout, None, &Resolvable::of(&jobs));
                let width: usize = rows[0]
                    .iter()
                    .map(|cell| crate::render::display_width(cell))
                    .sum::<usize>()
                    + 2 * (rows[0].len() - 1);
                assert!(
                    width <= TABLE_BUDGET,
                    "{layout:?} row is {width} columns against a budget of {TABLE_BUDGET}: {rows:?}"
                );
            }
        }
    }

    #[test]
    fn the_gate_column_names_the_kind() {
        let shell = &rows_for(
            &[(
                job_with(Some(shell_gate("fetch_url", GatePredicate::Changed))),
                None,
            )],
            Layout::Unscoped,
            None,
            &Resolvable::of(&[(
                job_with(Some(shell_gate("fetch_url", GatePredicate::Changed))),
                None,
            )]),
        )[0][4];
        let tool = &rows_for(
            &[(
                job_with(Some(Gate {
                    probe: GateProbe::Tool {
                        name: "fetch_url".to_string(),
                        arguments: serde_json::json!({}),
                    },
                    predicate: GatePredicate::Changed,
                    last_output: None,
                    permission: Permission::Read,
                })),
                None,
            )],
            Layout::Unscoped,
            None,
            &Resolvable::of(&[(
                job_with(Some(Gate {
                    probe: GateProbe::Tool {
                        name: "fetch_url".to_string(),
                        arguments: serde_json::json!({}),
                    },
                    predicate: GatePredicate::Changed,
                    last_output: None,
                    permission: Permission::Read,
                })),
                None,
            )]),
        )[0][4];

        assert_eq!(shell, "shell");
        assert_eq!(tool, "tool");
        assert_ne!(
            shell, tool,
            "an unsandboxed `sh -c` and a structured call must not read alike"
        );
        assert_eq!(
            rows_for(
                &[(job_with(None), None)],
                Layout::Unscoped,
                None,
                &Resolvable::of(&[(job_with(None), None)])
            )[0][4],
            "-",
            "and an ungated job says so, rather than leaving the cell ambiguous"
        );
    }

    /// `meka schedule list` spans every session, so a row that does not name one names nothing.
    ///
    /// The column is absent from the REPL's listing, which is already scoped to one conversation.
    /// Both header sets are asserted against the row width, since a header and a row that disagree
    /// shift every cell under the wrong title without failing anything.
    #[test]
    fn the_session_column_appears_only_where_it_can_differ() {
        let job = job_with(None);
        let unscoped = &rows_for(
            &[(job.clone(), None)],
            Layout::Unscoped,
            None,
            &Resolvable::of(&[(job.clone(), None)]),
        )[0];
        let scoped = &rows_for(
            &[(job.clone(), None)],
            Layout::SessionScoped,
            None,
            &Resolvable::of(&[(job.clone(), None)]),
        )[0];

        assert_eq!(unscoped.len(), COLUMNS.len());
        assert_eq!(scoped.len(), SESSION_SCOPED_COLUMNS.len());
        assert_eq!(
            unscoped[1],
            job.session_id.to_string()[..8],
            "the session cell is the id's prefix, which is what `--session` now accepts"
        );
        assert_eq!(
            scoped[1], unscoped[2],
            "and dropping it must shift the row, not blank one cell"
        );
    }

    /// An id column shows a first segment until that stops identifying a row.
    ///
    /// A prefix is what the reader retypes into `show`, `cancel` or `--session`, so printing one
    /// that matches two rows hands them a string those commands refuse. Widening only on collision
    /// is what keeps two UUID columns from spending 76 of the 120 available on every other listing.
    #[test]
    fn an_id_widens_only_far_enough_to_stay_unique() {
        let distinct = [
            "4d71eeca-9f21-4c3a-b8e7-1a2b3c4d5e6f",
            "1fb28dc6-0000-0000-0000-000000000000",
        ];
        assert_eq!(
            crate::render::unique_prefix_len(distinct.iter().copied()),
            crate::render::ID_PREFIX,
            "ids that differ in the first segment need no more than the first segment"
        );

        let colliding = [
            "4d71eeca-9f21-4c3a-b8e7-1a2b3c4d5e6f",
            "4d71eeca-0000-0000-0000-000000000000",
        ];
        let width = crate::render::unique_prefix_len(colliding.iter().copied());
        assert!(
            width > crate::render::ID_PREFIX,
            "a shared first segment has to widen"
        );
        assert_ne!(
            colliding[0].as_bytes()[width - 1],
            b'-',
            "and must not stop on the hyphen: {:?}",
            &colliding[0][..width]
        );
        assert_ne!(
            &colliding[0][..width],
            &colliding[1][..width],
            "the widened prefixes still have to differ"
        );

        assert_eq!(
            crate::render::unique_prefix_len(
                ["4d71eeca-9f21-4c3a-b8e7-1a2b3c4d5e6f"].iter().copied()
            ),
            crate::render::ID_PREFIX,
            "one row collides with nothing"
        );
    }

    /// Repeating an id is ordinary, and must not be mistaken for a collision.
    ///
    /// One session with several jobs fills the whole `Session` column with itself. Counting rows
    /// rather than distinct ids made that unsatisfiable, so the column widened to a full UUID to
    /// distinguish an id from itself -- and charged the prompt 28 columns for it.
    #[test]
    fn one_session_with_several_jobs_does_not_widen_the_session_column() {
        let session = uuid::Uuid::new_v4();
        let job = |id: &str| {
            let mut job = job_with(None);
            job.id = id.to_string();
            job.session_id = session;
            job.prompt = "x".repeat(200);
            job
        };
        let rows = rows_for(
            &[
                (job("7f3a1b2c-0000-0000-0000-000000000000"), None),
                (job("9e5d4c3b-0000-0000-0000-000000000000"), None),
            ],
            Layout::Unscoped,
            None,
            &Resolvable::of(&[
                (job("7f3a1b2c-0000-0000-0000-000000000000"), None),
                (job("9e5d4c3b-0000-0000-0000-000000000000"), None),
            ]),
        );

        assert_eq!(
            rows[0][1].chars().count(),
            crate::render::ID_PREFIX,
            "two jobs in one session are not two sessions: {:?}",
            rows[0][1]
        );
        assert_eq!(
            rows[0][5].chars().count(),
            PROMPT_TRUNCATE,
            "and the prompt keeps its full width, having paid for nothing"
        );
    }

    /// Widening an id is paid for out of the prompt, so the table holds its budget either way.
    #[test]
    fn a_widened_id_is_taken_out_of_the_prompt() {
        let long = "x".repeat(200);
        let mut first = job_with(None);
        let mut second = job_with(None);
        first.id = "4d71eeca-9f21-4c3a-b8e7-1a2b3c4d5e6f".to_string();
        second.id = "4d71eeca-0000-0000-0000-000000000000".to_string();
        first.prompt = long.clone();
        second.prompt = long;

        let jobs = [(first, None), (second, None)];
        let rows = rows_for(&jobs, Layout::SessionScoped, None, &Resolvable::of(&jobs));
        let widened = rows[0][0].chars().count();
        assert!(widened > crate::render::ID_PREFIX);
        assert_eq!(
            rows[0][5].chars().count(),
            PROMPT_TRUNCATE - (widened - crate::render::ID_PREFIX),
            "the prompt gives back exactly what the id took"
        );
    }

    /// An occurrence that is already due is a different state from one due in no time at all.
    ///
    /// A host that was down through a fire time leaves the job overdue, and `format_duration_short`
    /// clamps a negative duration to `0s` -- which reads as "about to fire" for a job that should
    /// already have.
    #[test]
    fn an_overdue_occurrence_reads_as_due() {
        let mut job = job_with(None);
        job.next_fire_at = chrono::Utc::now() - chrono::Duration::hours(3);
        assert_eq!(
            rows_for(&[(job, None)], Layout::Unscoped, None, &Resolvable::of(&[]))[0][3],
            "due"
        );

        let mut soon = job_with(None);
        soon.next_fire_at = chrono::Utc::now() + chrono::Duration::minutes(90);
        assert_eq!(
            rows_for(
                &[(soon, None)],
                Layout::Unscoped,
                None,
                &Resolvable::of(&[])
            )[0][3],
            "1h 29m"
        );
    }
}
