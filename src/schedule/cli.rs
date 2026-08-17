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
/// Same, for the gate's shell command. Wide enough that the interesting part of a real check
/// (`gh pr checks 123 --json state`) is legible, since the point of showing it is that an operator
/// can recognise what runs unattended.
const GATE_COMMAND_TRUNCATE: usize = 40;
/// Same, for the trailing prompt. Last column, so it is never padded, but a paragraph-long prompt
/// would still wrap the row into unreadability.
const PROMPT_TRUNCATE: usize = 60;

pub async fn run(session_manager: &SessionManager, action: &ScheduleAction) -> anyhow::Result<()> {
    match action {
        ScheduleAction::List { session } => list(session_manager, session.as_deref()).await,
        ScheduleAction::Cancel { id } => cancel(session_manager, id).await,
    }
    .map_err(Into::into)
}

/// `/schedule` in the REPL: the same table, scoped to the conversation the user is in.
pub async fn run_list_for_session(
    session_manager: &SessionManager,
    session: uuid::Uuid,
) -> Result<()> {
    render(
        session_manager
            .schedule_store()
            .list_scheduled_jobs(session)
            .await?,
    )
}

async fn list(session_manager: &SessionManager, session: Option<&str>) -> Result<()> {
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

    render(jobs)
}

fn render(jobs: Vec<crate::schedule::ScheduledJob>) -> Result<()> {
    if jobs.is_empty() {
        // stderr: an empty list is a status note, not the data a script asked for. A caller piping
        // this still gets a clean empty stdout.
        eprintln!("No scheduled jobs.");
        return Ok(());
    }

    let rows: Vec<Vec<String>> = jobs
        .iter()
        .map(|job| {
            vec![
                job.short_id().to_string(),
                truncate(&job.schedule.describe(), SCHEDULE_TRUNCATE),
                job.next_fire_at
                    .with_timezone(&chrono::Local)
                    .format("%Y-%m-%d %H:%M %Z")
                    .to_string(),
                match &job.gate {
                    Some(gate) => gate.fire.as_str().to_string(),
                    None => "-".to_string(),
                },
                // The command itself, not just the fire mode. This is the only listing meka ships
                // for "what runs unattended on this machine", and rendering only `on-change` left
                // an operator auditing a box unable to see what was executing
                // without opening SQLite. Sanitised because it reaches a terminal
                // and its author is the model.
                match &job.gate {
                    Some(gate) => {
                        crate::render::sanitize_to_line(&gate.command, GATE_COMMAND_TRUNCATE)
                    }
                    None => "-".to_string(),
                },
                truncate(
                    &job.prompt.split_whitespace().collect::<Vec<_>>().join(" "),
                    PROMPT_TRUNCATE,
                ),
            ]
        })
        .collect();

    print!(
        "{}",
        crate::render::format_columns(
            &["ID", "Schedule", "Next fire", "Gate", "Command", "Prompt"],
            &rows
        )
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
            session_manager
                .schedule_store()
                .delete_scheduled_job(&job.id)
                .await?;
            tracing::info!("cancelled scheduled job {}", job.id);
            Ok(())
        }
        several => Err(MekaError::Config(format!(
            "'{}' matches {} jobs; use a longer id",
            id_prefix,
            several.len()
        ))),
    }
}

fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let kept: String = text.chars().take(limit.saturating_sub(1)).collect();
    format!("{}…", kept)
}
