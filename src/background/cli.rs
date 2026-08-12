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

/// Ceiling on the rendered label. `format_columns` widens a column to its longest cell, so an
/// unbounded command line here would push the columns after it off the terminal.
const LABEL_TRUNCATE: usize = 48;
/// Same, for the trailing result excerpt. Last column, so never padded, but a build log would still
/// wrap the row into unreadability.
const RESULT_TRUNCATE: usize = 40;

/// `/tasks` in the REPL: one row per task in this conversation.
pub async fn run_list_for_session(
    session_manager: &SessionManager,
    session: uuid::Uuid,
) -> Result<()> {
    render(session_manager.list_background_tasks(session).await?)
}

fn render(tasks: Vec<crate::background::BackgroundTask>) -> Result<()> {
    if tasks.is_empty() {
        // stderr: an empty list is a status note, not the data a script asked for.
        eprintln!("No background tasks in this session.");
        return Ok(());
    }

    let rows: Vec<Vec<String>> = tasks
        .iter()
        .map(|task| {
            vec![
                task.short_id().to_string(),
                task.status.as_str().to_string(),
                task.tool_name.clone(),
                truncate(&task.label, LABEL_TRUNCATE),
                format_elapsed(task),
                match &task.outcome {
                    Some(outcome) if task.status.is_terminal() => truncate(
                        &crate::background::excerpt(outcome, RESULT_TRUNCATE),
                        RESULT_TRUNCATE,
                    ),
                    _ => "-".to_string(),
                },
            ]
        })
        .collect();

    print!(
        "{}",
        crate::render::format_columns(
            &["ID", "Status", "Tool", "What", "Elapsed", "Result"],
            &rows
        )
    );
    Ok(())
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
        let running: Vec<String> = session_manager
            .list_running_background_tasks(session)
            .await?
            .into_iter()
            .map(|task| task.id)
            .collect();
        for id in &running {
            session_manager
                .finish_background_task(id, TaskStatus::Cancelled, None, None)
                .await?;
        }
        return Ok(running);
    };

    let Some(task) = session_manager
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
        .finish_background_task(&task.id, TaskStatus::Cancelled, None, None)
        .await?;
    Ok(vec![task.id])
}

/// How long a task ran, or has been running.
fn format_elapsed(task: &crate::background::BackgroundTask) -> String {
    let seconds = task.elapsed().num_seconds().max(0) as u64;
    humantime_serde::re::humantime::format_duration(std::time::Duration::from_secs(seconds))
        .to_string()
}

fn truncate(text: &str, limit: usize) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= limit {
        return collapsed;
    }
    let kept: String = collapsed.chars().take(limit.saturating_sub(1)).collect();
    format!("{}…", kept)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::background::BackgroundTask;

    async fn manager_with_session() -> (SessionManager, uuid::Uuid) {
        let manager = SessionManager::open(Some(std::path::Path::new(":memory:")))
            .await
            .expect("in-memory db");
        let session = manager.create_session(None).await.expect("create session");
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
            delivered_at: None,
        };
        manager
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
            .list_undelivered_background_tasks(session)
            .await
            .expect("list");
        assert_eq!(undelivered.len(), 1);
        assert_eq!(undelivered[0].status, TaskStatus::Cancelled);
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
    fn test_truncate_collapses_whitespace_and_marks_the_cut() {
        assert_eq!(truncate("cargo   test\n--all", 40), "cargo test --all");
        assert_eq!(truncate("abcdefghij", 5), "abcd…");
    }
}
