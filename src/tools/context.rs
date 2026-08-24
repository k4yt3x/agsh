//! The `context_*` tools: the agent managing its own context window.
//!
//! Deliberately a separate family from `conversation_*`, which reads the *archive* - the full
//! on-disk log, including turns compaction removed from the window entirely. These three act on the
//! live window instead. The two families sort adjacently (`cont` precedes `conv`), so the split
//! costs nothing in the catalogue while keeping each name honest about what it touches.
//!
//! `context_check` exists because the pushed `[Context budget]` block
//! ([`crate::context::ContextBudget`]) is rendered once per turn, into the user message at turn
//! start. The counter behind it moves on every provider response including mid-tool-loop, but the
//! rendered text does not, so the gauge is stalest exactly when a tool loop is ingesting large
//! results. Refreshing the block in place would rewrite a message the cached prefix already covers
//! and invalidate it on every iteration; a tool result appends at the tail and is cache-safe by
//! construction.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use async_trait::async_trait;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::{Tool, ToolOutput, util::resolve_session_id};
use crate::{
    agent::{CompactOrigin, CompactRequest, compaction_tail_budget},
    error::Result,
    permission::Permission,
    provider::ToolDefinition,
    session::SessionManager,
};

/// A compaction the agent asked for, parked until the turn it was asked in finishes.
///
/// Tools hold no `&mut Conversation` - the agent loop owns it for the duration of the turn - so
/// `context_compact` cannot compact where it stands. It records the request here and `run_turn`
/// drains it once the tool loop is done, which also preserves the invariant that compaction happens
/// between turns and never mid-loop.
pub type PendingCompaction = Arc<std::sync::Mutex<Option<CompactRequest>>>;

/// The summary a checkpoint turn submitted, and what it decided about the tail.
pub struct Submission {
    pub summary: String,
    pub keep_recent: Option<bool>,
}

/// Slot `context_replace` writes into. Owned by the checkpoint turn that registered the tool, so a
/// fresh one is created per compaction and never outlives it.
pub type SubmissionSlot = Arc<std::sync::Mutex<Option<Submission>>>;

/// Live numbers `context_check` reports, kept current by the agent.
#[derive(Clone)]
pub struct ContextGauge {
    /// Total tokens behind the most recent provider round: the same handle
    /// `Agent::last_context_tokens` writes after every response, so this moves within a turn
    /// rather than only between turns.
    pub used: Arc<AtomicU64>,
    /// Estimated system prompt + tool schemas, re-stamped by the agent each turn. Separate from
    /// `used` because it is the part compaction *cannot* reclaim, which is what makes it worth
    /// reporting.
    pub overhead: Arc<AtomicU64>,
    /// The model's window, or zero when meka has no metadata for it.
    pub window: u64,
    /// Occupancy at which auto-compaction fires, or `None` when it is off.
    pub compact_at_percent: Option<u64>,
}

pub(super) struct ContextCheckTool {
    pub gauge: ContextGauge,
    pub session_manager: SessionManager,
    pub session_id: Arc<RwLock<Option<Uuid>>>,
}

#[async_trait]
impl Tool for ContextCheckTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "context_check".to_string(),
            description: "Measure your context window now. Unlike the `[Context budget]` line, \
                          which is fixed at the start of the turn, this is live. Call it before \
                          reading a large file, starting a long stretch of tool calls, or \
                          deciding whether a task fits."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
            }),
            title: None,
            annotations: None,
            meta: None,
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
        let used = self.gauge.used.load(Ordering::Relaxed);
        let overhead = self.gauge.overhead.load(Ordering::Relaxed);
        let window = self.gauge.window;

        let mut report = String::new();

        // `checked_div` is the zero-window test: without a denominator there is no occupancy to
        // report, and a percentage of an unknown total is worse than silence. Same call
        // `ContextBudget::render` makes.
        match used.saturating_mul(100).checked_div(window) {
            None => report.push_str(
                "Context window: unknown for this model, so occupancy cannot be reported.\n",
            ),
            Some(percent) => {
                report.push_str(&format!(
                    "Using {} of {} tokens ({}%).\n",
                    used, window, percent
                ));
                match self.gauge.compact_at_percent {
                    Some(threshold) => {
                        let limit = window.saturating_mul(threshold) / 100;
                        report.push_str(&format!(
                            "Headroom: {} tokens before auto-compaction fires at {}%.\n",
                            limit.saturating_sub(used),
                            threshold
                        ));
                    }
                    None => report.push_str(&format!(
                        "Headroom: {} tokens before the window is full. Auto-compaction is off, so \
                         a request past it fails the turn.\n",
                        window.saturating_sub(used)
                    )),
                }
                report.push_str(&format!(
                    "Kept verbatim on compaction: about {} tokens of the most recent turns; \
                     everything older is replaced by a summary.\n",
                    compaction_tail_budget(window)
                ));
            }
        }

        if overhead > 0 {
            report.push_str(&format!(
                "Fixed overhead: about {} tokens of system prompt and tool schemas (estimated). \
                 Compaction does not reclaim this.\n",
                overhead
            ));
            if used > overhead {
                report.push_str(&format!(
                    "Conversation: about {} tokens, which is the part compaction acts on.\n",
                    used.saturating_sub(overhead)
                ));
            }
        }

        // Best-effort: a session that has not been created yet, or a read that fails, should not
        // fail the whole call over a line that is context rather than the answer.
        if let Ok(session_id) = resolve_session_id(&self.session_id, "context_check").await
            && let Ok(generation) = self.session_manager.count_compactions(session_id).await
        {
            report.push_str(&match generation {
                0 => "Compactions so far: none, so nothing has been summarized away yet.\n".into(),
                1 => "Compactions so far: 1. Detail from before it survives only as a summary; \
                      `conversation_search` reaches the original turns.\n"
                    .to_string(),
                count => format!(
                    "Compactions so far: {}. Each one summarizes the previous summary, so early \
                     detail is now several removes from the original; write anything that must \
                     last to memory rather than trusting it to survive another pass.\n",
                    count
                ),
            });
        }

        Ok(ToolOutput::text(report, false))
    }
}

pub(super) struct ContextCompactTool {
    pub pending: PendingCompaction,
    /// Whether a checkpoint turn will actually run (`[session].compact_checkpoint`).
    ///
    /// Carried so this tool can tell the truth about what happens next. The difference is not
    /// cosmetic: with a checkpoint the agent gets a chance to save durable notes *after* asking to
    /// compact, so it can reasonably defer that work; without one the summary is written by a
    /// separate call with no tools, and anything not already saved is simply gone. An agent told
    /// it would get a checkpoint that never comes would skip the one action that mattered.
    pub checkpoint_enabled: bool,
}

#[async_trait]
impl Tool for ContextCompactTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "context_compact".to_string(),
            description: "Compact this conversation when the turn ends. Earlier turns become a \
                          summary you write, after a checkpoint for saving anything that must \
                          outlive them. Use it when a stretch of work is done rather than \
                          waiting for auto-compaction mid-task. `conversation_search` still \
                          reaches the full history."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "instructions": {
                        "type": "string",
                        "description": "What to preserve or drop, e.g. \"keep the design decisions, drop the debugging\". Takes precedence over the default summary sections."
                    },
                    "keep_recent": {
                        "type": "boolean",
                        "description": "Whether to keep the most recent turns verbatim after the summary. Default: true. Set false to start clean, only when the summary and what you have saved cover everything - e.g. closing out a day's work."
                    }
                },
            }),
            title: None,
            annotations: None,
            meta: None,
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
        let request = CompactRequest {
            origin: CompactOrigin::Requested,
            instructions: input["instructions"]
                .as_str()
                .map(str::trim)
                .filter(|instructions| !instructions.is_empty())
                .map(str::to_string),
            keep_recent: input["keep_recent"].as_bool(),
        };
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Last call wins rather than first: a turn that asks twice most likely refined what it
        // wanted, and refusing the second would silently apply the stale instructions.
        *pending = Some(request);
        Ok(ToolOutput::text(
            if self.checkpoint_enabled {
                "Compaction will run when this turn ends. Finish what you are doing; you will get \
                 a checkpoint to save anything durable before the summary replaces your context."
                    .to_string()
            } else {
                "Compaction will run when this turn ends. There is no checkpoint on this \
                 installation, so the summary will be written without you: save anything that \
                 must outlive this conversation to memory now, before you finish this turn."
                    .to_string()
            },
            false,
        ))
    }
}

/// The checkpoint turn's terminal call. Registered only for that turn, so it never appears in the
/// ordinary catalogue and is deliberately absent from `BUILTIN_TOOL_NAMES`: listing it there would
/// let a `disabled_tools` entry silently downgrade every compaction to the fallback summariser.
pub(super) struct ContextReplaceTool {
    pub slot: SubmissionSlot,
}

#[async_trait]
impl Tool for ContextReplaceTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "context_replace".to_string(),
            description: "Replace your context with the summary you have written. Call this once, \
                          last, to end the checkpoint. Everything before the kept tail is \
                          discarded from your window, so the summary has to carry whatever the \
                          work still depends on."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "The text that will replace the earlier turns. Write it for yourself, in your own voice, as the record you would want to pick the work back up from."
                    },
                    "keep_recent": {
                        "type": "boolean",
                        "description": "Whether to keep the most recent turns verbatim after the summary. Default: true. Set false only when your summary and what you have saved fully cover them."
                    }
                },
                "required": ["summary"]
            }),
            title: None,
            annotations: None,
            meta: None,
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
        let summary = input["summary"].as_str().unwrap_or_default().trim();
        if summary.is_empty() {
            // A tool error rather than a hard failure: the loop gets another iteration to try
            // again, and if it never does, the fallback ladder catches it.
            return Ok(ToolOutput::text(
                "The 'summary' parameter is required and cannot be empty. It becomes your entire \
                 context, so an empty one would erase the conversation."
                    .to_string(),
                true,
            ));
        }
        let mut slot = self
            .slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *slot = Some(Submission {
            summary: summary.to_string(),
            keep_recent: input["keep_recent"].as_bool(),
        });
        Ok(ToolOutput::text("Checkpoint accepted.".to_string(), false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::tests::text_content;

    fn gauge(used: u64, overhead: u64, window: u64, compact_at: Option<u64>) -> ContextGauge {
        ContextGauge {
            used: Arc::new(AtomicU64::new(used)),
            overhead: Arc::new(AtomicU64::new(overhead)),
            window,
            compact_at_percent: compact_at,
        }
    }

    async fn check(gauge: ContextGauge) -> String {
        let tool = ContextCheckTool {
            gauge,
            session_manager: SessionManager::open(Some(std::path::Path::new(":memory:")))
                .await
                .expect("in-memory db"),
            session_id: Arc::new(RwLock::new(None)),
        };
        let output = tool
            .execute(serde_json::json!({}), CancellationToken::new())
            .await
            .expect("context_check");
        text_content(&output)
    }

    #[tokio::test]
    async fn reports_headroom_to_the_compaction_threshold() {
        let report = check(gauge(40_000, 5_000, 200_000, Some(80))).await;
        assert!(report.contains("40000 of 200000 tokens (20%)"), "{report}");
        // 80% of 200k is 160k, so 120k of headroom is left.
        assert!(report.contains("Headroom: 120000 tokens"), "{report}");
        assert!(
            report.contains("about 5000 tokens of system prompt"),
            "{report}"
        );
        assert!(
            report.contains("Conversation: about 35000 tokens"),
            "{report}"
        );
    }

    /// The threshold is what matters when auto-compaction is on, but with it off the window itself
    /// is the wall, and reporting headroom to a threshold that will never fire would be a lie.
    #[tokio::test]
    async fn reports_headroom_to_the_window_when_auto_compaction_is_off() {
        let report = check(gauge(40_000, 0, 200_000, None)).await;
        assert!(report.contains("Headroom: 160000 tokens"), "{report}");
        assert!(report.contains("Auto-compaction is off"), "{report}");
    }

    /// An unknown window must not divide by zero, and must not invent a percentage either.
    #[tokio::test]
    async fn suppresses_occupancy_when_the_window_is_unknown() {
        let report = check(gauge(40_000, 0, 0, Some(80))).await;
        assert!(report.contains("unknown for this model"), "{report}");
        assert!(!report.contains('%'), "{report}");
    }

    #[tokio::test]
    async fn compact_requests_carry_instructions_and_the_tail_decision() {
        let pending: PendingCompaction = Arc::new(std::sync::Mutex::new(None));
        let tool = ContextCompactTool {
            pending: Arc::clone(&pending),
            checkpoint_enabled: true,
        };
        tool.execute(
            serde_json::json!({"instructions": "keep the decisions", "keep_recent": false}),
            CancellationToken::new(),
        )
        .await
        .expect("context_compact");

        let request = pending.lock().expect("lock").take().expect("recorded");
        assert_eq!(request.origin, CompactOrigin::Requested);
        assert_eq!(request.instructions.as_deref(), Some("keep the decisions"));
        assert_eq!(request.keep_recent, Some(false));
    }

    /// An empty summary would erase the conversation outright, so it has to be refused in a way the
    /// model can recover from rather than accepted.
    #[tokio::test]
    async fn replace_refuses_an_empty_summary() {
        let slot: SubmissionSlot = Arc::new(std::sync::Mutex::new(None));
        let tool = ContextReplaceTool {
            slot: Arc::clone(&slot),
        };
        let output = tool
            .execute(
                serde_json::json!({"summary": "   "}),
                CancellationToken::new(),
            )
            .await
            .expect("context_replace");
        assert!(output.content.iter().any(|_| true));
        assert!(slot.lock().expect("lock").is_none());
    }

    #[tokio::test]
    async fn replace_records_the_summary_and_tail_decision() {
        let slot: SubmissionSlot = Arc::new(std::sync::Mutex::new(None));
        let tool = ContextReplaceTool {
            slot: Arc::clone(&slot),
        };
        tool.execute(
            serde_json::json!({"summary": "what happened", "keep_recent": false}),
            CancellationToken::new(),
        )
        .await
        .expect("context_replace");

        let submission = slot.lock().expect("lock").take().expect("recorded");
        assert_eq!(submission.summary, "what happened");
        assert_eq!(submission.keep_recent, Some(false));
    }
}
