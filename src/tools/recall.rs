//! `recall` / `recall_read`: read-only search and retrieval over the current session's full
//! conversation log, including turns that compaction summarized away and removed from the model's
//! context. Compaction never deletes (it appends a boundary, [`crate::conversation`]), so every
//! turn is still on disk in the `messages` table; these tools give the model a way back to detail
//! the summary may have dropped.
//!
//! A rewind is the one thing that does remove turns from this view. That is deliberate: `/rewind`
//! exists to take a turn back, and a search that could hand it straight back to the model would
//! defeat it. Those turns remain on disk and in `meka session export`.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::{
    Tool, ToolOutput,
    util::{MAX_SEARCH_MATCHES, compile_user_regex, require_str, resolve_session_id},
};
use crate::{
    conversation::Event,
    error::{MekaError, Result},
    permission::Permission,
    provider::{ContentBlock, Message, Role, ToolDefinition},
    session::SessionManager,
};

/// Default number of matches `recall` returns when the caller doesn't set `limit`.
const DEFAULT_RECALL_LIMIT: usize = 20;
/// Hard cap on how many messages `recall_read` returns in one call.
const MAX_RECALL_READ_MESSAGES: usize = 20;
/// Each `recall` match line is truncated to this many characters so the result stays compact.
const SNIPPET_CHARS: usize = 200;

fn role_label(role: &Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

/// The `Append` messages in log order. `CompactBoundary` events are skipped: their synthetic
/// summary is already in the model's context, and skipping them keeps message indices stable across
/// compactions (the log only ever grows).
fn append_messages(events: &[Event]) -> Vec<&Message> {
    let mut messages: Vec<&Message> = Vec::new();
    for event in events {
        match event {
            Event::Append(message) => messages.push(message),
            Event::CompactBoundary { .. } => {}
            // Applied rather than skipped: a repaired message is what the model actually saw, so
            // searching the superseded original would surface content that is no longer in the
            // conversation. Unlike a boundary this can shift indices, but only across a repair,
            // which is rare and confined to a single run of trailing messages.
            Event::Repair {
                replaced_count,
                messages: replacements,
            } => {
                let truncate_to = messages.len().saturating_sub(*replaced_count);
                messages.truncate(truncate_to);
                messages.extend(replacements.iter());
            }
        }
    }
    messages
}

/// Flatten a message into searchable lines. Non-text blocks are prefixed so a match is
/// identifiable; images carry no text and are skipped.
fn searchable_text(message: &Message) -> String {
    let mut segments: Vec<String> = Vec::new();
    for block in &message.content {
        match block {
            ContentBlock::Text { text } => segments.push(text.clone()),
            ContentBlock::Thinking { thinking, .. } => {
                segments.push(format!("[thinking] {thinking}"))
            }
            ContentBlock::RedactedThinking { .. } => {
                segments.push("[redacted thinking]".to_string())
            }
            ContentBlock::ToolUse { name, input, .. } => {
                segments.push(format!("[tool call: {name}] {input}"));
            }
            ContentBlock::ToolResult {
                content, is_error, ..
            } => {
                let label = if *is_error {
                    "[tool result (error)]"
                } else {
                    "[tool result]"
                };
                segments.push(format!(
                    "{label} {}",
                    ContentBlock::tool_result_text_content(content)
                ));
            }
            ContentBlock::Image { .. } => {}
        }
    }
    segments.join("\n")
}

/// Full, untruncated rendering of a message for `recall_read`.
fn render_message_full(message: &Message) -> String {
    let mut segments: Vec<String> = Vec::new();
    for block in &message.content {
        match block {
            ContentBlock::Text { text } => segments.push(text.clone()),
            ContentBlock::Thinking { thinking, .. } => {
                segments.push(format!("[thinking]\n{thinking}"))
            }
            ContentBlock::RedactedThinking { .. } => {
                segments.push("[redacted thinking]".to_string())
            }
            ContentBlock::ToolUse { name, input, .. } => {
                let pretty =
                    serde_json::to_string_pretty(input).unwrap_or_else(|_| input.to_string());
                segments.push(format!("[tool call: {name}]\n{pretty}"));
            }
            ContentBlock::ToolResult {
                content, is_error, ..
            } => {
                let label = if *is_error {
                    "[tool result (error)]"
                } else {
                    "[tool result]"
                };
                segments.push(format!(
                    "{label}\n{}",
                    ContentBlock::tool_result_text_content(content)
                ));
            }
            ContentBlock::Image { .. } => segments.push("[image]".to_string()),
        }
    }
    segments.join("\n")
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() > max_chars {
        let head: String = text.chars().take(max_chars).collect();
        format!("{head}…")
    } else {
        text.to_string()
    }
}

/// Literal-substring (case-insensitive) or regex matcher for one query.
enum Matcher {
    Substring(String),
    Regex(regex::Regex),
}

impl Matcher {
    fn build(query: &str, use_regex: bool) -> Result<Self> {
        if use_regex {
            Ok(Matcher::Regex(compile_user_regex(query, "recall")?))
        } else {
            Ok(Matcher::Substring(query.to_lowercase()))
        }
    }

    fn is_match(&self, line: &str) -> bool {
        match self {
            Matcher::Substring(needle) => line.to_lowercase().contains(needle.as_str()),
            Matcher::Regex(regex) => regex.is_match(line),
        }
    }
}

fn search_events(events: &[Event], query: &str, use_regex: bool, limit: usize) -> Result<String> {
    use std::fmt::Write;

    let messages = append_messages(events);
    let total_messages = messages.len();
    let matcher = Matcher::build(query, use_regex)?;
    let cap = limit.clamp(1, MAX_SEARCH_MATCHES);

    let mut shown: Vec<String> = Vec::new();
    let mut total_found = 0usize;
    let mut first_index: Option<usize> = None;
    for (offset, message) in messages.iter().enumerate() {
        let index = offset + 1;
        let label = role_label(&message.role);
        let text = searchable_text(message);
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || !matcher.is_match(line) {
                continue;
            }
            total_found += 1;
            first_index.get_or_insert(index);
            if shown.len() < cap {
                shown.push(format!(
                    "#{index} [{label}] {}",
                    truncate_chars(line, SNIPPET_CHARS)
                ));
            }
        }
    }

    if total_found == 0 {
        return Ok(format!(
            "No matches for {query:?} in {total_messages} message(s)."
        ));
    }

    let mut output = String::new();
    if total_found > shown.len() {
        writeln!(
            output,
            "{total_found} matches for {query:?} across {total_messages} message(s) (showing first {}):\n",
            shown.len()
        )
        .ok();
    } else {
        writeln!(
            output,
            "{} match(es) for {query:?} across {total_messages} message(s):\n",
            shown.len()
        )
        .ok();
    }
    for line in &shown {
        writeln!(output, "{line}").ok();
    }
    if let Some(index) = first_index {
        writeln!(
            output,
            "\nRead a full turn with `recall_read` (e.g. {{\"start\": {index}}})."
        )
        .ok();
    }
    Ok(output)
}

fn read_messages(events: &[Event], start: usize, count: usize) -> Result<String> {
    use std::fmt::Write;

    let messages = append_messages(events);
    let total = messages.len();
    if start == 0 || start > total {
        return Err(MekaError::ToolExecution {
            tool_name: "recall_read".to_string(),
            message: format!("start {start} is out of range (valid: 1..={total})"),
        });
    }
    let count = count.clamp(1, MAX_RECALL_READ_MESSAGES);
    let end = (start - 1 + count).min(total);

    let mut output = String::new();
    writeln!(output, "Messages #{start}..#{end} of {total}:\n").ok();
    for index in start..=end {
        let message = messages[index - 1];
        writeln!(output, "#{index} [{}]", role_label(&message.role)).ok();
        output.push_str(&render_message_full(message));
        output.push_str("\n\n");
    }
    Ok(output)
}

pub(super) struct RecallTool {
    pub session_manager: SessionManager,
    pub session_id: Arc<RwLock<Option<Uuid>>>,
}

#[async_trait]
impl Tool for RecallTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "recall".to_string(),
            description: format!(
                "Search this session's full conversation history, including earlier turns that \
                 compaction summarized away and removed from your context. Returns matching lines, \
                 each tagged with its message index (#N) and role; follow up with `recall_read` to \
                 read a full turn. Use this to recover a detail the compaction summary may have \
                 omitted. Substring matching is case-insensitive; set `regex: true` to match \
                 `query` as a case-sensitive regular expression. Large tool outputs appear as \
                 <large-output> references here; read their full content with `scratchpad_read`. \
                 Returns at most {MAX_SEARCH_MATCHES} matches."
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for (a literal substring unless `regex` is true)."
                    },
                    "regex": {
                        "type": "boolean",
                        "description": "Treat `query` as a regular expression instead of a literal substring. Default: false."
                    },
                    "limit": {
                        "type": "integer",
                        "description": format!("Maximum matches to return (max {MAX_SEARCH_MATCHES}). Default: {DEFAULT_RECALL_LIMIT}.")
                    }
                },
                "required": ["query"]
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
        let query = require_str(&input, "query", "recall")?;
        let use_regex = input
            .get("regex")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let limit = input
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(DEFAULT_RECALL_LIMIT);

        let session_id = resolve_session_id(&self.session_id, "recall").await?;
        let events = self.session_manager.load_events(session_id).await?;
        Ok(ToolOutput::text(
            search_events(&events, &query, use_regex, limit)?,
            false,
        ))
    }
}

pub(super) struct RecallReadTool {
    pub session_manager: SessionManager,
    pub session_id: Arc<RwLock<Option<Uuid>>>,
}

#[async_trait]
impl Tool for RecallReadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "recall_read".to_string(),
            description: format!(
                "Read the full content of conversation turns by message index, including turns \
                 compaction removed from your context. Use the #N indices reported by `recall`. \
                 Reads up to {MAX_RECALL_READ_MESSAGES} messages starting at `start`."
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "start": {
                        "type": "integer",
                        "description": "1-based message index to start reading from (the #N from `recall`)."
                    },
                    "count": {
                        "type": "integer",
                        "description": format!("Number of consecutive messages to read (max {MAX_RECALL_READ_MESSAGES}). Default: 1.")
                    },
                    "scratchpad": {
                        "type": "string",
                        "description": "If provided, save the output to the scratchpad under this name instead of returning it inline."
                    }
                },
                "required": ["start"]
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
        let start = input
            .get("start")
            .and_then(serde_json::Value::as_u64)
            .map(|value| value as usize)
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "recall_read".to_string(),
                message: "missing or invalid 'start' parameter".to_string(),
            })?;
        let count = input
            .get("count")
            .and_then(serde_json::Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(1);

        let session_id = resolve_session_id(&self.session_id, "recall_read").await?;
        let events = self.session_manager.load_events(session_id).await?;
        Ok(ToolOutput::text(
            read_messages(&events, start, count)?,
            false,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    fn boundary(replaced: usize) -> Event {
        Event::CompactBoundary {
            summary: Message::user("[summary]"),
            replaced_count: replaced,
            loaded_tools_snapshot: HashSet::new(),
        }
    }

    /// A log where the first two turns were compacted away (a boundary sits after them) and one
    /// turn follows. `recall` must still see the pre-boundary turns.
    fn compacted_log() -> Vec<Event> {
        vec![
            Event::Append(Message::user("the auth token expired unexpectedly")),
            Event::Append(Message::assistant_text(
                "I'll investigate the token refresh path",
            )),
            boundary(2),
            Event::Append(Message::user("now let's talk about something unrelated")),
        ]
    }

    #[test]
    fn search_finds_pre_compaction_turn_case_insensitively() {
        let events = compacted_log();
        let out = search_events(&events, "AUTH Token", false, 20).expect("search");
        // Message #1 is before the boundary (hidden from the model) but still searchable.
        assert!(out.contains("#1 [user]"), "expected a hit on #1: {out}");
        assert!(out.contains("auth token expired"), "{out}");
    }

    #[test]
    fn search_regex_mode_matches() {
        let events = compacted_log();
        let out = search_events(&events, "tok.n refresh", true, 20).expect("search");
        assert!(out.contains("#2 [assistant]"), "{out}");
    }

    #[test]
    fn search_no_match_reports_zero() {
        let events = compacted_log();
        let out = search_events(&events, "nonexistent phrase", false, 20).expect("search");
        assert!(out.starts_with("No matches"), "{out}");
    }

    #[test]
    fn boundary_summary_is_not_indexed() {
        // The synthetic summary message must not get an index; #3 is the post-boundary user turn.
        let events = compacted_log();
        let out = search_events(&events, "unrelated", false, 20).expect("search");
        assert!(out.contains("#3 [user]"), "{out}");
    }

    #[test]
    fn read_messages_returns_full_turns_by_index() {
        let events = compacted_log();
        let out = read_messages(&events, 1, 2).expect("read");
        assert!(
            out.contains("#1 [user]") && out.contains("auth token expired"),
            "{out}"
        );
        assert!(
            out.contains("#2 [assistant]") && out.contains("token refresh path"),
            "{out}"
        );
    }

    #[test]
    fn read_messages_out_of_range_errors() {
        let events = compacted_log();
        assert!(read_messages(&events, 0, 1).is_err());
        assert!(read_messages(&events, 99, 1).is_err());
    }
}
