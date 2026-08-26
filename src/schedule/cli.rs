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
/// Same, for what the gate runs: a shell command, or a tool name. Wide enough that the interesting
/// part of a real check (`gh pr checks 123 --json state`) is legible, since the point of showing it
/// is that an operator can recognise what runs unattended.
const GATE_CHECK_TRUNCATE: usize = 40;
/// Same, for the fire condition. It used to be one of two fixed words and needed no ceiling; it now
/// renders a regular expression or a JSON pointer, both written by the model and neither bounded or
/// validated beyond compiling.
const GATE_WHEN_TRUNCATE: usize = 28;
/// Same, for the trailing prompt. Last column, so it is never padded, but a paragraph-long prompt
/// would still wrap the row into unreadability.
const PROMPT_TRUNCATE: usize = 60;

pub async fn run(
    session_manager: &SessionManager,
    action: &ScheduleAction,
    cli_args: &crate::cli::Cli,
) -> anyhow::Result<()> {
    match action {
        ScheduleAction::List { session } => {
            // Resolved rather than defaulted, exactly as `meka tools list` does and for the same
            // reason: the `Held` column is computed against `[permissions].enabled`, so a table
            // rendered off defaults would report a job as firing that the running host refuses.
            let config = crate::config::ResolvedConfig::from_cli(cli_args);
            config.require_readable_config()?;
            list(session_manager, session.as_deref(), &config.schedule).await
        }
        ScheduleAction::Cancel { id } => cancel(session_manager, id).await,
    }
    .map_err(Into::into)
}

/// `/schedule` in the REPL: the same table, scoped to the conversation the user is in.
///
/// Unlike the standalone command this process *is* a host, so it hands the renderer its gate
/// dispatcher. Without it every tool gate rendered `?`, and the docs explained that as "the CLI has
/// no MCP manager" -- true of `meka schedule list`, and not of the surface the same sentence named.
pub async fn run_list_for_session(
    session_manager: &SessionManager,
    session: uuid::Uuid,
    config: &crate::config::ResolvedScheduleConfig,
) -> Result<()> {
    let jobs = session_manager
        .schedule_store()
        .list_scheduled_jobs(session)
        .await?;
    render(
        with_levels(session_manager, config, jobs).await,
        config.gate_tools.as_deref(),
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
/// A level that cannot be established at all leaves this `None`, which the `Held` column renders as
/// `?`. The host level is deliberately *not* used as a fallback here the way the fire door uses it:
/// the host that eventually runs this job is some other process, and its `--permission` is not
/// something this command can see. "I cannot tell" is the honest answer, and the column has a way
/// to say it.
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

async fn list(
    session_manager: &SessionManager,
    session: Option<&str>,
    config: &crate::config::ResolvedScheduleConfig,
) -> Result<()> {
    let jobs = match session {
        Some(raw) => {
            let id = uuid::Uuid::parse_str(raw)
                .map_err(|error| MekaError::Config(format!("invalid session id: {}", error)))?;
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

    // No dispatcher: a separate process from any host, so it cannot resolve a tool gate and says
    // so rather than guessing. `/schedule` in the REPL passes the one its process holds.
    render(with_levels(session_manager, config, jobs).await, None)
}

/// The table's header, beside the row builder so the two cannot drift apart in width or order.
const COLUMNS: [&str; 7] = [
    "ID",
    "Schedule",
    "Next fire",
    "When",
    "Check",
    "Held",
    "Prompt",
];

/// One row per job, separated from printing so it can be tested.
///
/// Every cell here except the id and the timestamp carries text meka did not write: a command or a
/// pointer the model authored, a tool name an MCP server chose, a prompt that may have been
/// composed from a page the agent fetched. That is why they all go through
/// [`crate::render::sanitize_to_line`], and why the separation matters: `render` prints, so the
/// sanitising it does had no way to be asserted.
///
/// "All" is meant literally, and did not used to be: the `Prompt` cell was truncated and not
/// sanitised. `split_whitespace` had been doing half the job by dropping newlines, so a forged row
/// was impossible and an injected colour sequence was not -- and that gap was invisible in a test
/// that loops over every cell while planting escapes in only two of them.
///
/// The `Schedule` cell goes through the same helper without ever having been a way in: everything
/// `Schedule::describe` can emit is a formatted timestamp, a `humantime` duration, or a cron
/// pattern, and croner's five-field grammar admits no control characters at either the creation or
/// the rehydration door. It is sanitised so the guarantee this function makes does not quietly
/// depend on a parser two modules away staying that strict.
fn rows_for(
    jobs: &[(
        crate::schedule::ScheduledJob,
        Option<crate::permission::Permission>,
    )],
    tools: Option<&dyn crate::schedule::GateTools>,
) -> Vec<Vec<String>> {
    jobs.iter()
        .map(|(job, level)| {
            vec![
                job.short_id().to_string(),
                crate::render::sanitize_to_line(&job.schedule.describe(), SCHEDULE_TRUNCATE),
                job.next_fire_at
                    .with_timezone(&chrono::Local)
                    .format("%Y-%m-%d %H:%M %Z")
                    .to_string(),
                // Sanitised like the `Check` beside it, and for the same reason. This cell was a
                // fixed enum word until the predicate gained `matches` and `at`; both now carry
                // model-authored text into a terminal, and a pointer is checked only for its
                // leading `/`, so an escape sequence or a newline in one would forge a row.
                match &job.gate {
                    Some(gate) => crate::render::sanitize_to_line(
                        &gate.predicate.summary(),
                        GATE_WHEN_TRUNCATE,
                    ),
                    None => "-".to_string(),
                },
                // The probe itself, not just the predicate. This is the only listing meka ships
                // for "what runs unattended on this machine", and rendering only
                // `changed` left an operator auditing a box unable to see what was
                // executing without opening SQLite.
                //
                // `summary`, so a tool gate shows its name and not its arguments: an operator
                // auditing the box did not write them, and they are where a pasted credential
                // would sit. `schedule_list` shows them to the agent that did
                // write them. Prefixed with the kind, because a tool name is also
                // a valid command: a shell gate running `fetch_url` and a tool
                // gate calling `fetch_url` rendered identically here, and the
                // difference between them is an unsandboxed `sh -c` and a structured call.
                match &job.gate {
                    Some(gate) => crate::render::sanitize_to_line(
                        &format!("{} {}", gate.probe.kind_str(), gate.probe.summary()),
                        GATE_CHECK_TRUNCATE,
                    ),
                    None => "-".to_string(),
                },
                // Three answers, because this listing has no MCP manager and so cannot resolve a
                // tool gate at all. Rendering that as blank put "I cannot tell" in the same cell
                // as "it will fire", under a header that reads as a verdict; `?`
                // says which it is. A level that could not be read leaves the cell
                // blank, since then even the question is unanswerable and there is
                // nothing to point at.
                // The level is handed over as an `Option` rather than being matched on here, so
                // that the questions needing no level are still asked. A parked job is the one
                // that matters: it is provably dead, and answering `?` because this process could
                // not read the session's level would have the one surface an operator opens to
                // find out why a job stopped decline to say.
                //
                // `?`, not blank, when the answer genuinely cannot be reached: a level that could
                // not be established is the same kind of non-answer as a tool this reader cannot
                // resolve, and blank is documented as a verdict.
                match crate::schedule::job_withheld(job, *level, tools) {
                    crate::schedule::Withheld::Yes(_) => "yes".to_string(),
                    crate::schedule::Withheld::Undetermined => "?".to_string(),
                    crate::schedule::Withheld::No => String::new(),
                },
                // Whitespace collapsed first, then sanitised. The collapse is for legibility -- a
                // prompt is prose and wraps -- and it is not a safety measure: `\u{1b}` is not
                // whitespace, so it survived a `split_whitespace` that looked like it was cleaning
                // the cell.
                crate::render::sanitize_to_line(
                    &job.prompt.split_whitespace().collect::<Vec<_>>().join(" "),
                    PROMPT_TRUNCATE,
                ),
            ]
        })
        .collect()
}

fn render(
    jobs: Vec<(
        crate::schedule::ScheduledJob,
        Option<crate::permission::Permission>,
    )>,
    tools: Option<&dyn crate::schedule::GateTools>,
) -> Result<()> {
    if jobs.is_empty() {
        // stderr: an empty list is a status note, not the data a script asked for. A caller piping
        // this still gets a clean empty stdout.
        eprintln!("No scheduled jobs.");
        return Ok(());
    }

    print!(
        "{}",
        crate::render::format_columns(&COLUMNS, &rows_for(&jobs, tools))
    );
    Ok(())
}

async fn cancel(session_manager: &SessionManager, id_prefix: &str) -> Result<()> {
    // Scan every session: the operator has an id, not a session, and requiring them to find the
    // session first would make the id useless on its own.
    let jobs = session_manager
        .schedule_store()
        .list_all_scheduled_jobs()
        .await?;
    let matches: Vec<_> = jobs
        .iter()
        .filter(|job| job.id.starts_with(id_prefix))
        .collect();

    match matches.as_slice() {
        [] => Err(MekaError::Config(format!(
            "no scheduled job matching '{}'",
            id_prefix
        ))),
        [job] => {
            // Reported from the delete's own row count, not from the listing that found the job: a
            // scheduler sweep can retire it in between, and `ok: cancelled` about a job this
            // command did not cancel is indistinguishable from the real thing.
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
        several => Err(MekaError::Config(format!(
            "'{}' matches {} jobs; use a longer id",
            id_prefix,
            several.len()
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

    fn job_with(gate: Option<Gate>) -> ScheduledJob {
        ScheduledJob {
            id: "7f3a1b2c-0000-0000-0000-000000000000".to_string(),
            session_id: uuid::Uuid::nil(),
            schedule: Schedule::parse_every("1h").expect("parses"),
            prompt: "watch the thing".to_string(),
            gate,
            isolated: false,
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
        let rows = rows_for(&[(job, Some(Permission::Unrestricted))], None);

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
            row[3].chars().count() <= GATE_WHEN_TRUNCATE,
            "and the condition is bounded, or one long pointer pushes every later column off the \
             screen: {:?}",
            row[3]
        );
        assert!(
            row[6].chars().count() <= PROMPT_TRUNCATE,
            "the prompt too, since a paragraph would wrap the row into unreadability: {:?}",
            row[6]
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
        let manager = crate::session::SessionManager::open(Some(std::path::Path::new(":memory:")))
            .await
            .expect("open in-memory database");
        let session = manager.create_session(None).await.expect("create session");
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
    fn the_held_column_reports_a_parked_job_even_with_no_session_level() {
        let mut job = job_with(None);
        job.attempts = crate::schedule::MAX_CLAIM_ATTEMPTS;
        assert_eq!(
            &rows_for(&[(job, None)], None)[0][5],
            "yes",
            "a parked job is provably dead whatever the session is set to"
        );
    }

    /// Blank means "it will fire", so a level this command could not establish must not render as
    /// one.
    ///
    /// Two ways to get there, and the second is why the enabled set is applied here at all: the row
    /// carries no level, or it carries one this installation no longer permits. The fire door
    /// clamps by `[permissions].enabled` and refuses; this listing did not, and reported the job as
    /// healthy to the one audience that came here to check.
    #[test]
    fn the_held_column_does_not_report_an_unestablished_level_as_healthy() {
        let unknown = &rows_for(
            &[(
                job_with(Some(shell_gate("gh pr checks", GatePredicate::Changed))),
                None,
            )],
            None,
        )[0][5];
        assert_eq!(
            unknown, "?",
            "a session whose level could not be read is unestablished, not healthy"
        );
    }

    /// The `Held` column answers three things, and blank is a verdict rather than a shrug.
    ///
    /// This listing has no MCP manager, so it cannot resolve a tool gate at all. Rendering that as
    /// blank put "I cannot tell" in the same cell as "it will fire", under a header that reads as
    /// the latter.
    #[test]
    fn the_held_column_distinguishes_cannot_fire_from_cannot_tell() {
        let held = &rows_for(
            &[(
                job_with(Some(shell_gate("gh pr checks", GatePredicate::Changed))),
                Some(Permission::Read),
            )],
            None,
        )[0][5];
        assert_eq!(held, "yes", "a shell gate below `unrestricted` cannot fire");

        let undetermined = &rows_for(
            &[(
                job_with(Some(Gate {
                    probe: GateProbe::Tool {
                        name: "mcp__bridge__unseen".to_string(),
                        arguments: serde_json::json!({}),
                    },
                    predicate: GatePredicate::Changed,
                    last_output: None,
                    permission: Permission::Read,
                })),
                Some(Permission::Read),
            )],
            None,
        )[0][5];
        assert_eq!(
            undetermined, "?",
            "a tool gate this process cannot resolve is unestablished, not healthy"
        );

        let healthy = &rows_for(
            &[(
                job_with(Some(shell_gate("gh pr checks", GatePredicate::Changed))),
                Some(Permission::Unrestricted),
            )],
            None,
        )[0][5];
        assert_eq!(healthy, "", "and an authorised gate says nothing at all");
    }

    /// A tool name is also a valid command, so the kind has to be on the row.
    #[test]
    fn the_check_column_names_the_kind() {
        let shell = &rows_for(
            &[(
                job_with(Some(shell_gate("fetch_url", GatePredicate::Changed))),
                Some(Permission::Unrestricted),
            )],
            None,
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
                Some(Permission::Unrestricted),
            )],
            None,
        )[0][4];

        assert_eq!(shell, "shell fetch_url");
        assert_eq!(tool, "tool fetch_url");
        assert_ne!(
            shell, tool,
            "an unsandboxed `sh -c` and a structured call must not read alike"
        );
    }
}
