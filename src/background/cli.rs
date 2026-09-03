//! Listing and cancelling background tasks from outside the agent.
//!
//! Read-and-cancel only, for the same reason [`crate::schedule::cli`] is: *starting* one needs a
//! session for the result to be reported into, and the decision to detach belongs to the agent
//! making the call.

use crate::{
    background::TaskStatus,
    error::{MekaError, Result},
    session::SessionManager,
};

/// What [`render`] aims to fit in.
///
/// Six columns of independent ceilings reached about 138 with an ordinary tool name, and nothing
/// capped the tool at all: an MCP name is chosen by the server, so a long one pushed every column
/// after it off the screen. The id takes whatever distinguishes it, the fixed columns take theirs,
/// and the two authored cells split what is left.
const TABLE_WIDTH: usize = 120;

/// Ceiling on the tool name. Server-chosen, so nothing else bounds it.
const TOOL_TRUNCATE: usize = 24;

/// Floor on each of the two authored columns, so a long tool name cannot squeeze them to nothing.
const AUTHORED_MINIMUM: usize = 16;

/// `/tasks` in the REPL: one row per task in this conversation.
pub async fn run_list_for_session(
    session_manager: &SessionManager,
    session: uuid::Uuid,
) -> Result<()> {
    render(
        session_manager
            .background_store()
            .list_background_tasks(session)
            .await?,
    )
}

fn render(tasks: Vec<crate::background::BackgroundTask>) -> Result<()> {
    if tasks.is_empty() {
        // stderr: an empty list is a status note, not the data a script asked for.
        crate::render::write_stderr_line("No background tasks in this session.");
        return Ok(());
    }

    let rows = task_rows(&tasks);

    crate::render::write_stdout(crate::render::format_columns(
        &["ID", "Status", "Tool", "What", "Elapsed", "Result"],
        &rows,
    ))?;
    Ok(())
}

/// One row per task, separated from printing so the widths and the sanitising can be asserted.
///
/// `What` and `Result` both carry text meka did not write -- a command line the model composed, and
/// an excerpt of whatever that command printed -- so both go through
/// [`crate::render::sanitize_to_line`], which caps in terminal columns rather than characters.
fn task_rows(tasks: &[crate::background::BackgroundTask]) -> Vec<Vec<String>> {
    let ids: Vec<&str> = tasks.iter().map(|task| task.id.as_str()).collect();
    let id_width = crate::render::unique_prefix_len(ids.iter().copied()).max("ID".len());
    let width_of = |cells: Vec<String>, header: usize| {
        cells
            .iter()
            .map(|cell| unicode_width::UnicodeWidthStr::width(cell.as_str()))
            .chain(std::iter::once(header))
            .max()
            .unwrap_or(header)
    };
    let tools: Vec<String> = tasks
        .iter()
        .map(|task| crate::render::sanitize_to_line(&task.tool_name, TOOL_TRUNCATE))
        .collect();
    let tool_width = width_of(tools.clone(), "Tool".len());
    let elapsed: Vec<String> = tasks.iter().map(format_elapsed).collect();
    let elapsed_width = width_of(elapsed.clone(), "Elapsed".len());
    let status_width = width_of(
        tasks
            .iter()
            .map(|task| task.status.as_str().to_string())
            .collect(),
        "Status".len(),
    );

    // What is left after the fixed columns, split between the two authored ones.
    let remaining = TABLE_WIDTH
        .saturating_sub(id_width + 2)
        .saturating_sub(status_width + 2)
        .saturating_sub(tool_width + 2)
        .saturating_sub(elapsed_width + 2)
        .saturating_sub(2);
    let label_width = (remaining / 2).max(AUTHORED_MINIMUM);
    let result_width = remaining.saturating_sub(label_width).max(AUTHORED_MINIMUM);

    tasks
        .iter()
        .zip(tools)
        .zip(elapsed)
        .map(|((task, tool), elapsed)| {
            vec![
                task.id.get(..id_width).unwrap_or(&task.id).to_string(),
                task.status.as_str().to_string(),
                tool,
                crate::render::sanitize_to_line(&collapse(&task.label), label_width),
                elapsed,
                match &task.outcome {
                    Some(outcome) if task.status.is_terminal() => crate::render::sanitize_to_line(
                        &collapse(&crate::background::excerpt(outcome, result_width)),
                        result_width,
                    ),
                    _ => "-".to_string(),
                },
            ]
        })
        .collect()
}

/// `/tasks show <id>`: one task, with the id in full.
///
/// The listing shortens an id to whatever distinguishes it, which is only safe because this prints
/// the whole thing. It also carries the two cells a column cannot hold: the command line as
/// written, and the outcome rather than a 40-column excerpt of it.
pub async fn show(
    session_manager: &SessionManager,
    session: uuid::Uuid,
    id_prefix: &str,
) -> Result<()> {
    let Some(task) = session_manager
        .background_store()
        .resolve_background_task(session, id_prefix)
        .await?
    else {
        return Err(MekaError::Config(format!(
            "no background task matching '{}'",
            id_prefix
        )));
    };

    crate::render::write_stdout(show_lines(&task))?;
    Ok(())
}

/// One `name: value` line per field, separated from printing so the alignment and the sanitising
/// can be asserted. Five of the values are model-authored or server-chosen, and this is the one
/// surface that prints them untruncated.
fn show_lines(task: &crate::background::BackgroundTask) -> String {
    let mut out = String::new();
    // A free function rather than a closure: the two untruncated blocks below append to `out`
    // directly, and a closure holding it borrowed would rule that out.
    fn field(out: &mut String, name: &str, value: &str) {
        use std::fmt::Write as _;
        // The longest label plus one, as in `schedule show` and `session show`, so the values line
        // up without a table. `full output` is both the longest and the only conditional one, so a
        // width measured against the fields that always print is two columns short of it.
        if value.is_empty() {
            // `result:` heads the outcome rather than carrying one, so padding it to the column
            // would leave a run of trailing spaces on the row.
            writeln!(out, "{}:", name).ok();
        } else {
            writeln!(out, "{:<13} {}", format!("{}:", name), value).ok();
        }
    }
    field(&mut out, "id", &task.id);
    field(&mut out, "session", &task.session_id.to_string());
    field(&mut out, "status", task.status.as_str());
    field(
        &mut out,
        "tool",
        &crate::render::sanitize_to_line(&task.tool_name, usize::MAX),
    );
    field(&mut out, "elapsed", &format_elapsed(task));
    field(&mut out, "started", &task.started_at.to_rfc3339());
    field(&mut out, "finished", &match task.finished_at {
        Some(at) => at.to_rfc3339(),
        None => "-".to_string(),
    });
    field(
        &mut out,
        "what",
        &crate::render::sanitize_to_line(&task.label, usize::MAX),
    );
    // Named before the result is printed, because the result below is then only the part that fit.
    // `render_outcomes` and `task_list` both tell the model where the rest went; this is the
    // human-facing surface that claims to print the outcome in full, so it is the one place the
    // pointer cannot be missing.
    if let Some(scratchpad) = &task.scratchpad_name {
        field(
            &mut out,
            "full output",
            &format!(
                "scratchpad entry {}",
                crate::render::sanitize_to_line(scratchpad, usize::MAX)
            ),
        );
    }
    if let Some(outcome) = &task.outcome {
        field(&mut out, "result", "");
        // Sanitised per line rather than collapsed to one: an outcome is program output and its
        // line structure is most of what makes it readable. Indented, because it is not: a program
        // whose stdout contains `status:    completed` would otherwise render exactly like the
        // fields above it, and this command exists to be believed about what a task did.
        for line in outcome.lines() {
            out.push_str("  ");
            out.push_str(&crate::render::sanitize_to_line(line, usize::MAX));
            out.push('\n');
        }
    }
    out
}

/// Cancel one task in `session` by full or unique-prefix id, or every running one.
///
/// Records the terminal outcome; signalling the live handle is the caller's job, since only the
/// process that started a task holds its token.
pub async fn cancel(
    session_manager: &SessionManager,
    session: uuid::Uuid,
    id_prefix: Option<&str>,
) -> Result<Vec<String>> {
    let Some(id_prefix) = id_prefix else {
        let store = session_manager.background_store();
        let running: Vec<String> = store
            .list_running_background_tasks(session)
            .await?
            .into_iter()
            .map(|task| task.id)
            .collect();
        for id in &running {
            store
                .finish_background_task(id, TaskStatus::Cancelled, None, None)
                .await?;
        }
        return Ok(running);
    };

    let Some(task) = session_manager
        .background_store()
        .resolve_background_task(session, id_prefix)
        .await?
    else {
        return Err(MekaError::Config(format!(
            "no background task matching '{}'",
            id_prefix
        )));
    };
    if task.status.is_terminal() {
        return Err(MekaError::Config(format!(
            "task {} already {}",
            task.short_id(),
            task.status.as_str()
        )));
    }
    session_manager
        .background_store()
        .finish_background_task(&task.id, TaskStatus::Cancelled, None, None)
        .await?;
    Ok(vec![task.id])
}

/// How long a task ran, or has been running.
/// How long a task ran, in a cell the table can budget for.
///
/// `humantime` spells a duration in full -- `3years 4months 16days 17h 28m 57s` is 33 columns --
/// and nothing bounds how long a task has been running: an `interrupted` row from a session
/// resumed months later renders exactly that, and the two authored columns have floors, so the row
/// simply overran. The two coarsest units are what a reader takes from this cell anyway.
fn format_elapsed(task: &crate::background::BackgroundTask) -> String {
    let seconds = task.elapsed().num_seconds().max(0) as u64;
    let spelled =
        humantime_serde::re::humantime::format_duration(std::time::Duration::from_secs(seconds))
            .to_string();
    spelled
        .split_whitespace()
        .take(2)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Collapse runs of whitespace, for legibility rather than safety: a command line and a build-log
/// excerpt both wrap. `sanitize_to_line` is what makes the cell safe, and it is not a substitute --
/// `\u{1b}` is not whitespace, so this alone leaves an escape intact.
fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::background::BackgroundTask;

    /// Every field starts its value in the same column, including the one that appears only when
    /// the output overflowed into a scratchpad.
    ///
    /// `full output` is the longest label and the only conditional one, so a width measured against
    /// the fields that always print reads as correct until a task is big enough to need it.
    #[test]
    fn every_field_of_a_task_starts_in_the_same_column() {
        let task = BackgroundTask {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: uuid::Uuid::new_v4(),
            tool_name: "execute_command".to_string(),
            label: "cargo test --all".to_string(),
            status: TaskStatus::Completed,
            outcome: Some("test result: ok".to_string()),
            scratchpad_name: Some("build-log".to_string()),
            started_at: chrono::Utc::now(),
            finished_at: Some(chrono::Utc::now()),
            announced_at: None,
            delivered_at: None,
        };

        let rendered = show_lines(&task);
        // The outcome is indented under `result:` rather than being a field of its own, and
        // `result:` carries no value to line anything up against. A value may hold colons of its
        // own -- both timestamps do -- so the label's is the first one.
        let columns: Vec<(usize, &str)> = rendered
            .lines()
            .filter(|line| !line.starts_with(' '))
            .filter_map(|line| {
                let label = line.find(':')?;
                let value = line[label + 1..].find(|c: char| c != ' ')? + label + 1;
                Some((value, line))
            })
            .collect();

        assert!(
            columns
                .iter()
                .any(|(_, line)| line.starts_with("full output:")),
            "the longest label has to be in the sample, or this proves nothing: {rendered}"
        );
        let (first_column, first_line) = *columns.first().expect("the fields rendered");
        for (column, line) in &columns {
            assert_eq!(
                *column, first_column,
                "`{line}` starts its value at {column}, `{first_line}` at {first_column}"
            );
        }
    }

    /// The values a model or a server chose reach this surface sanitised.
    ///
    /// `show` is the one command that prints them untruncated, so it is also the one where a forged
    /// `status:` row or a cleared screen would be most convincing. `sanitize_to_line` is asked for
    /// each of them; this checks it was actually asked.
    #[test]
    fn a_task_show_sanitises_every_value_it_did_not_write() {
        let forged = "done\x1b[2J\rstatus:      completed";
        let task = BackgroundTask {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: uuid::Uuid::new_v4(),
            tool_name: format!("mcp__evil__{}", forged),
            label: forged.to_string(),
            status: TaskStatus::Completed,
            outcome: Some(forged.to_string()),
            scratchpad_name: Some(forged.to_string()),
            started_at: chrono::Utc::now(),
            finished_at: Some(chrono::Utc::now()),
            announced_at: None,
            delivered_at: None,
        };

        let rendered = show_lines(&task);
        assert!(
            !rendered.contains('\x1b'),
            "an escape reached the terminal: {rendered:?}"
        );
        assert!(
            !rendered.contains('\r'),
            "a carriage return can repaint the row above it: {rendered:?}"
        );
        assert!(
            rendered.contains("done"),
            "and the text either side of it still shows: {rendered:?}"
        );
    }

    async fn manager_with_session() -> (SessionManager, uuid::Uuid) {
        let manager =
            SessionManager::open(Some(std::path::Path::new(":memory:")), &Default::default())
                .await
                .expect("in-memory db");
        let session = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        (manager, session)
    }

    async fn seed(manager: &SessionManager, session: uuid::Uuid, label: &str) -> BackgroundTask {
        let task = BackgroundTask {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: session,
            tool_name: "execute_command".to_string(),
            label: label.to_string(),
            status: TaskStatus::Running,
            outcome: None,
            scratchpad_name: None,
            started_at: chrono::Utc::now(),
            finished_at: None,
            announced_at: None,
            delivered_at: None,
        };
        manager
            .background_store()
            .start_background_task(&task)
            .await
            .expect("start task");
        task
    }

    #[tokio::test]
    async fn test_cancel_by_prefix_records_the_outcome() {
        let (manager, session) = manager_with_session().await;
        let task = seed(&manager, session, "sleep 600").await;

        let cancelled = cancel(&manager, session, Some(&task.id[..8]))
            .await
            .expect("cancel");
        assert_eq!(cancelled, vec![task.id.clone()]);

        let undelivered = manager
            .background_store()
            .list_undelivered_background_tasks(session)
            .await
            .expect("list");
        assert_eq!(undelivered.len(), 1);
        assert_eq!(undelivered[0].status, TaskStatus::Cancelled);
    }

    /// `/tasks cancel ""` must not stop the only running task.
    ///
    /// `id.starts_with("")` is true of every id, so an unset variable resolved to whichever task
    /// happened to be alone and cancelled it -- while reading as correct, because the ambiguity
    /// error only appears once a second task exists. `--all` is the way to mean all of them, and it
    /// is spelled.
    #[tokio::test]
    async fn cancelling_an_empty_prefix_stops_nothing() {
        let (manager, session) = manager_with_session().await;
        let task = seed(&manager, session, "sleep 600").await;

        let error = cancel(&manager, session, Some(""))
            .await
            .expect_err("an empty prefix names no task");
        assert!(
            error.to_string().contains("no background task"),
            "it must read as a miss, not as an ambiguity: {error}"
        );
        assert_eq!(
            manager
                .background_store()
                .list_running_background_tasks(session)
                .await
                .expect("list")
                .first()
                .map(|running| running.id.clone()),
            Some(task.id),
            "and the task must still be running"
        );
    }

    #[tokio::test]
    async fn test_cancel_all_records_every_running_task() {
        let (manager, session) = manager_with_session().await;
        seed(&manager, session, "sleep 1").await;
        seed(&manager, session, "sleep 2").await;

        let cancelled = cancel(&manager, session, None).await.expect("cancel all");
        assert_eq!(cancelled.len(), 2);
        assert!(
            manager
                .background_store()
                .list_running_background_tasks(session)
                .await
                .expect("list")
                .is_empty()
        );
    }

    #[tokio::test]
    async fn test_cancel_rejects_an_unknown_id() {
        let (manager, session) = manager_with_session().await;
        assert!(cancel(&manager, session, Some("deadbeef")).await.is_err());
    }

    #[tokio::test]
    async fn test_cancel_rejects_an_already_finished_task() {
        let (manager, session) = manager_with_session().await;
        let task = seed(&manager, session, "make").await;
        manager
            .background_store()
            .finish_background_task(&task.id, TaskStatus::Completed, None, None)
            .await
            .expect("finish");

        assert!(
            cancel(&manager, session, Some(&task.id[..8]))
                .await
                .is_err()
        );
    }

    #[test]
    fn a_collapsed_cell_is_legible_but_not_yet_safe() {
        assert_eq!(collapse("cargo   test\n--all"), "cargo test --all");
        // The half `collapse` deliberately does not do, and why every cell it feeds goes through
        // `sanitize_to_line` after it.
        assert!(collapse("cargo\u{1b}[31m test").contains('\u{1b}'));
    }

    /// The table has a budget, and both cells that carry authored text respect it.
    ///
    /// Six independent ceilings reached about 138, and `tool_name` had none at all -- an MCP server
    /// chooses that name, so a long one pushed every later column off the screen.
    #[test]
    fn the_task_table_fits_its_budget_and_sanitises_what_it_did_not_write() {
        let task = BackgroundTask {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: uuid::Uuid::new_v4(),
            tool_name: "mcp__a_very_long_server_name__an_even_longer_tool".to_string(),
            label: "cargo test\u{1b}[31m\n1234abcd  running  x  y  z  forged".to_string(),
            status: TaskStatus::Completed,
            outcome: Some("out\u{1b}[31m\nput ".to_string() + &"x".repeat(400)),
            scratchpad_name: None,
            started_at: chrono::Utc::now(),
            finished_at: Some(chrono::Utc::now()),
            announced_at: None,
            delivered_at: None,
        };

        let rows = task_rows(&[task]);
        let [row] = rows.as_slice() else {
            panic!("one task, one row: {rows:?}");
        };
        for cell in row {
            assert!(
                !cell.contains('\u{1b}') && !cell.contains('\n'),
                "a cell reaches a terminal verbatim, so neither may survive: {cell:?}"
            );
        }
        let width: usize = row
            .iter()
            .map(|cell| unicode_width::UnicodeWidthStr::width(cell.as_str()))
            .sum::<usize>()
            + 2 * (row.len() - 1);
        assert!(
            width <= TABLE_WIDTH,
            "the row spends {width} of a {TABLE_WIDTH} budget: {row:?}"
        );
    }
}
