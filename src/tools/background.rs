//! The `task_*` tools: the agent watching and stopping its own background work
//! ([`crate::background`]).
//!
//! Both gate at [`Permission::Read`]. Neither runs anything: listing reads meka's own store, and
//! cancelling only signals a token belonging to work whose permission was already checked when it
//! was dispatched. Requiring `write` to stop something the agent itself started would leave it
//! unable to clean up after a call it had every right to make.
//!
//! Sub-agents deliberately get neither, for the same reason they get no `schedule_*`: a sub-agent's
//! session ends with the single turn that spawned it, so it can neither start a task that outlives
//! that turn nor be around to hear about one.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::{
    Tool, ToolOutput,
    util::{require_str, resolve_session_id},
};
use crate::{
    background::{BackgroundTasks, TaskStatus},
    error::Result,
    permission::Permission,
    provider::ToolDefinition,
    session::SessionManager,
};

/// Name of the tool the `[Background]` index exists to drive. Without it the index would be a menu
/// with nothing to order from.
pub const TASK_INDEX_TOOL: &str = "task_list";

/// How much of a finished task's output `task_list` shows. Enough to recognise what happened, not
/// enough to make listing tasks a way to re-read every result.
const OUTCOME_EXCERPT_CHARS: usize = 200;

/// Ceiling on the rendered label. `format_columns` widens a column to its longest cell, so an
/// unbounded command line would push everything after it far off to the right.
const LABEL_EXCERPT_CHARS: usize = 48;

struct TaskContext {
    session_manager: SessionManager,
    session_id: Arc<RwLock<Option<Uuid>>>,
    tasks: BackgroundTasks,
}

pub(super) struct TaskListTool {
    context: TaskContext,
}

#[async_trait]
impl Tool for TaskListTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: TASK_INDEX_TOOL.to_string(),
            description: "List this session's background tasks: what is still running, and what \
                has already reported. Your per-turn context carries a short index of the running \
                ones, so reach for this when you want the full picture including finished tasks."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        _input: serde_json::Value,
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let session_id = resolve_session_id(&self.context.session_id, TASK_INDEX_TOOL).await?;
        let tasks = self
            .context
            .session_manager
            .background_store()
            .list_background_tasks(session_id)
            .await?;
        if tasks.is_empty() {
            return Ok(ToolOutput::text(
                "No background tasks in this session.".to_string(),
                false,
            ));
        }

        // The same column layout every other listing in meka uses (`scratchpad_list`, `meka session
        // list`, `meka mcp list`), so a reader who has seen one has seen them all.
        let rows: Vec<Vec<String>> = tasks
            .iter()
            .map(|task| {
                vec![
                    task.short_id().to_string(),
                    task.status.as_str().to_string(),
                    task.tool_name.clone(),
                    crate::background::excerpt(&task.label, LABEL_EXCERPT_CHARS),
                    humantime_serde::re::humantime::format_duration(
                        std::time::Duration::from_secs(task.elapsed().num_seconds().max(0) as u64),
                    )
                    .to_string(),
                    // An excerpt for anything already finished. Outcomes are delivered as their
                    // own turn, but that delivery is stamped before the turn
                    // runs, so a turn that fails (a provider error, an
                    // interrupt) consumes the report. Without this the result
                    // would be reachable only by reading the database by hand, which for the agent
                    // means not at all.
                    match (&task.outcome, &task.scratchpad_name) {
                        (_, Some(name)) => format!("in scratchpad '{}'", name),
                        (Some(outcome), None) if task.status.is_terminal() => {
                            crate::background::excerpt(outcome, OUTCOME_EXCERPT_CHARS)
                        }
                        _ => "-".to_string(),
                    },
                ]
            })
            .collect();

        let mut rendered = crate::render::format_columns(
            &["ID", "Status", "Tool", "What", "Elapsed", "Result"],
            &rows,
        );
        rendered.push_str(
            "\nA finished task's full result is delivered to you on its own; this listing is for \
             checking what is still running.",
        );
        Ok(ToolOutput::text(rendered, false))
    }
}

pub(super) struct TaskCancelTool {
    context: TaskContext,
}

#[async_trait]
impl Tool for TaskCancelTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "task_cancel".to_string(),
            description:
                "Stop a running background task by id (the short form from `task_list` is \
                enough), or every one of them with all=true. A cancelled task still reports back, \
                so you will be told when it has actually stopped."
                    .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Task id, full or the short prefix task_list shows",
                    },
                    "all": {
                        "type": "boolean",
                        "default": false,
                        "description": "Cancel every running task in this session instead",
                    }
                }
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let session_id = resolve_session_id(&self.context.session_id, "task_cancel").await?;

        if input
            .get("all")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            // Recorded before signalling, exactly as the single-id path below does and for the same
            // reason: the work reacting to its token reports an interruption, which would otherwise
            // land as `failed` and tell the agent its build broke rather than that it was stopped.
            let ids = self.context.tasks.session_task_ids(session_id).await;
            let store = self.context.session_manager.background_store();
            for id in &ids {
                store
                    .finish_background_task(id, TaskStatus::Cancelled, None, None)
                    .await?;
            }
            let signalled = self.context.tasks.cancel_session(session_id).await;
            return Ok(ToolOutput::text(
                if signalled == 0 {
                    "No running background tasks to cancel.".to_string()
                } else {
                    format!(
                        "Asked {} background task(s) to stop. Each will report back once it has.",
                        signalled
                    )
                },
                false,
            ));
        }

        let id_prefix = require_str(&input, "id", "task_cancel")?;
        let Some(task) = self
            .context
            .session_manager
            .background_store()
            .resolve_background_task(session_id, &id_prefix)
            .await?
        else {
            return Ok(ToolOutput::text(
                format!(
                    "Error: no background task in this session matches '{}'. Call `task_list` for \
                     the current ids.",
                    id_prefix
                ),
                true,
            ));
        };

        if task.status.is_terminal() {
            return Ok(ToolOutput::text(
                format!(
                    "Task {} already {} and is not running.",
                    task.short_id(),
                    task.status.as_str()
                ),
                false,
            ));
        }

        // Record the cancellation before signalling. `finish_background_task` only writes over a
        // `running` row, so whichever of the two lands first wins, and doing it in this order means
        // a task that happens to finish in the same instant cannot report success after the agent
        // was told it was stopped.
        self.context
            .session_manager
            .background_store()
            .finish_background_task(&task.id, TaskStatus::Cancelled, None, None)
            .await?;
        let signalled = self.context.tasks.cancel(&task.id).await;
        if !signalled {
            // The row was ours to retire but the handle was not: the task belonged to a process
            // that is gone. Recording it is still the right move, and is what stops the agent
            // waiting forever.
            tracing::debug!(
                "task {} had no live handle in this process; recorded as cancelled",
                task.short_id()
            );
        }
        Ok(ToolOutput::text(
            format!(
                "Asked task {} ({}) to stop. It will report back once it has.",
                task.short_id(),
                task.label
            ),
            false,
        ))
    }
}

pub(super) fn build(
    session_manager: SessionManager,
    session_id: Arc<RwLock<Option<Uuid>>>,
    tasks: BackgroundTasks,
) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(TaskListTool {
            context: TaskContext {
                session_manager: session_manager.clone(),
                session_id: session_id.clone(),
                tasks: tasks.clone(),
            },
        }),
        Arc::new(TaskCancelTool {
            context: TaskContext {
                session_manager,
                session_id,
                tasks,
            },
        }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{background::BackgroundTask, provider::ContentBlock};

    async fn context() -> (TaskContext, Uuid) {
        let session_manager = SessionManager::open(Some(std::path::Path::new(":memory:")))
            .await
            .expect("in-memory db");
        let session_id = session_manager
            .create_session(None)
            .await
            .expect("create session");
        (
            TaskContext {
                session_manager,
                session_id: Arc::new(RwLock::new(Some(session_id))),
                tasks: BackgroundTasks::default(),
            },
            session_id,
        )
    }

    async fn seed(context: &TaskContext, session_id: Uuid, label: &str) -> BackgroundTask {
        let task = BackgroundTask {
            id: Uuid::new_v4().to_string(),
            session_id,
            tool_name: "execute_command".to_string(),
            label: label.to_string(),
            status: TaskStatus::Running,
            outcome: None,
            scratchpad_name: None,
            started_at: chrono::Utc::now(),
            finished_at: None,
            delivered_at: None,
        };
        context
            .session_manager
            .background_store()
            .start_background_task(&task)
            .await
            .expect("start task");
        task
    }

    #[tokio::test]
    async fn test_task_list_reports_nothing_when_there_is_nothing() {
        let (context, _) = context().await;
        let tool = TaskListTool { context };
        let result = tool
            .execute(serde_json::json!({}), CancellationToken::new())
            .await
            .expect("ok");
        assert!(!result.is_error);
        assert!(
            ContentBlock::tool_result_text_content(&result.content).contains("No background tasks")
        );
    }

    #[tokio::test]
    async fn test_task_list_names_each_task_and_its_state() {
        let (context, session_id) = context().await;
        seed(&context, session_id, "cargo test --all").await;
        let tool = TaskListTool { context };
        let result = tool
            .execute(serde_json::json!({}), CancellationToken::new())
            .await
            .expect("ok");
        let text = ContentBlock::tool_result_text_content(&result.content);
        assert!(text.contains("ID"), "column headers: {text}");
        assert!(text.contains("cargo test --all"), "{text}");
        assert!(text.contains("running"), "{text}");
    }

    /// `[failed] … running for 30s` reads as a contradiction and invites the agent to keep waiting
    /// on work that already stopped.
    #[tokio::test]
    async fn test_task_list_does_not_say_a_finished_task_is_running() {
        let (context, session_id) = context().await;
        let task = seed(&context, session_id, "make").await;
        context
            .session_manager
            .background_store()
            .finish_background_task(&task.id, TaskStatus::Failed, Some("boom".to_string()), None)
            .await
            .expect("finish");

        let tool = TaskListTool { context };
        let result = tool
            .execute(serde_json::json!({}), CancellationToken::new())
            .await
            .expect("ok");
        let text = ContentBlock::tool_result_text_content(&result.content);
        // Scoped to the task's own row: the trailing note legitimately mentions what is still
        // running, and asserting over the whole output would be testing the footer.
        let row = text
            .lines()
            .find(|line| line.contains(task.short_id()))
            .unwrap_or_else(|| panic!("no row for the task: {text}"));
        assert!(row.contains("failed"), "{row}");
        // The `Status` column is the single place a task's state is stated. An elapsed time
        // labelled "running for" beside a `failed` badge read as a contradiction and invited the
        // agent to keep waiting on work that had already stopped.
        assert!(!row.contains("running"), "{row}");
    }

    /// The cancellation has to be recorded even when the handle is gone, or the agent waits forever
    /// on a task it was told it had stopped.
    #[tokio::test]
    async fn test_task_cancel_records_a_terminal_outcome() {
        let (context, session_id) = context().await;
        let task = seed(&context, session_id, "sleep 600").await;
        let session_manager = context.session_manager.clone();
        let tool = TaskCancelTool { context };

        let result = tool
            .execute(
                serde_json::json!({"id": &task.id[..8]}),
                CancellationToken::new(),
            )
            .await
            .expect("ok");
        assert!(!result.is_error, "{:?}", result.content);

        let undelivered = session_manager
            .background_store()
            .list_undelivered_background_tasks(session_id)
            .await
            .expect("list");
        assert_eq!(undelivered.len(), 1);
        assert_eq!(undelivered[0].status, TaskStatus::Cancelled);
    }

    #[tokio::test]
    async fn test_task_cancel_rejects_an_unknown_id() {
        let (context, _) = context().await;
        let tool = TaskCancelTool { context };
        let result = tool
            .execute(
                serde_json::json!({"id": "deadbeef"}),
                CancellationToken::new(),
            )
            .await
            .expect("ok");
        assert!(result.is_error);
        assert!(
            ContentBlock::tool_result_text_content(&result.content).contains("no background task")
        );
    }

    /// Cancelling in bulk must record `cancelled`, not leave the task's own interruption to land as
    /// `failed`. "Your build failed" and "you stopped your build" call for different next moves.
    /// Outcome delivery is stamped before its turn runs, so a turn that fails consumes the report.
    /// Listing has to be able to recover it, or the result is reachable only by reading the
    /// database by hand.
    #[tokio::test]
    async fn test_task_list_shows_a_finished_task_s_result() {
        let (context, session_id) = context().await;
        let task = seed(&context, session_id, "cargo test").await;
        context
            .session_manager
            .background_store()
            .finish_background_task(
                &task.id,
                TaskStatus::Completed,
                Some("42 passed; 0 failed".to_string()),
                None,
            )
            .await
            .expect("finish");

        let tool = TaskListTool { context };
        let result = tool
            .execute(serde_json::json!({}), CancellationToken::new())
            .await
            .expect("ok");
        let text = ContentBlock::tool_result_text_content(&result.content);
        assert!(text.contains("42 passed"), "{text}");
    }

    /// A running task has no result yet, so listing must not invent one.
    #[tokio::test]
    async fn test_task_list_shows_no_result_for_a_running_task() {
        let (context, session_id) = context().await;
        seed(&context, session_id, "sleep 600").await;
        let tool = TaskListTool { context };
        let result = tool
            .execute(serde_json::json!({}), CancellationToken::new())
            .await
            .expect("ok");
        let text = ContentBlock::tool_result_text_content(&result.content);
        assert!(!text.contains("result:"), "{text}");
    }

    #[tokio::test]
    async fn test_task_cancel_all_records_cancelled_not_failed() {
        let (context, session_id) = context().await;
        let task = seed(&context, session_id, "sleep 600").await;
        // A live handle, so the bulk path has something to enumerate.
        context
            .tasks
            .try_reserve(task.id.clone(), session_id, CancellationToken::new(), 10)
            .await;
        let session_manager = context.session_manager.clone();
        let tool = TaskCancelTool { context };

        let result = tool
            .execute(serde_json::json!({"all": true}), CancellationToken::new())
            .await
            .expect("ok");
        assert!(!result.is_error);

        let undelivered = session_manager
            .background_store()
            .list_undelivered_background_tasks(session_id)
            .await
            .expect("list");
        assert_eq!(undelivered.len(), 1);
        assert_eq!(undelivered[0].status, TaskStatus::Cancelled);
    }

    #[tokio::test]
    async fn test_task_cancel_all_is_a_no_op_when_nothing_runs() {
        let (context, _) = context().await;
        let tool = TaskCancelTool { context };
        let result = tool
            .execute(serde_json::json!({"all": true}), CancellationToken::new())
            .await
            .expect("ok");
        assert!(!result.is_error);
        assert!(
            ContentBlock::tool_result_text_content(&result.content)
                .contains("No running background")
        );
    }
}
