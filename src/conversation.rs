//! [`Conversation`]: append-only-by-default newtype for the agent's conversation.
//!
//! Built on an event log: each mutation pushes one or more [`Event`]s, and the materialized
//! `&[Message]` view consumed by providers and the scanner is derived from those events. The three
//! legitimate destructive operations ([`Conversation::pop_unsaved`],
//! [`Conversation::replace_for_compaction`], [`Conversation::sanitize_orphans`]) remain explicit,
//! named methods; the compiler refuses casual mutation.
//!
//! On disk, events are stored row-per-event in the existing `messages` table (no schema migration);
//! the encoding lives in `session.rs`'s `encode_event_for_db` / `decode_event_from_row` helpers,
//! behind the [`crate::session::SessionManager::save_event`] /
//! [`crate::session::SessionManager::load_events`] API.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::provider::{ContentBlock, Message, Role};

/// One entry in the underlying event log of a [`Conversation`]. Persisted as a single row in the
/// `messages` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Event {
    /// Adds a message to the materialized view.
    Append(Message),
    /// Marks a compaction boundary: when materializing, drop the last `replaced_count` materialized
    /// messages and push `summary` instead. Subsequent `Append` events extend the new tail. Carries
    /// the set of deferred tools that were active at compaction time so `extract_loaded_tool_names`
    /// can recover them after the boundary (otherwise compaction would silently un-load them).
    CompactBoundary {
        summary: Message,
        replaced_count: usize,
        loaded_tools_snapshot: HashSet<String>,
    },
}

/// Append-only conversation. Public API matches PR 1's `Vec<Message>`-backed implementation
/// byte-for-byte; PR 2 swaps the internals to an event log.
#[derive(Debug, Default, Clone)]
pub struct Conversation {
    events: Vec<Event>,
    /// Materialized view kept in lockstep with `events`. Rebuilt by `materialize_into` after every
    /// mutation; reads are zero-cost.
    materialized: Vec<Message>,
}

impl Conversation {
    pub fn new() -> Self {
        Self::default()
    }

    /// Hydrate from a sequence of events (typically loaded from the session DB on resume). The
    /// materialized view is computed once and cached.
    pub fn from_events(events: Vec<Event>) -> Self {
        let mut log = Self {
            events,
            materialized: Vec::new(),
        };
        log.rebuild_materialized();
        log
    }

    /// Hydrate from a flat `Vec<Message>`; every entry becomes an `Event::Append`. Used by the
    /// resume path until the persistence layer is fully event-aware.
    pub fn from_vec(entries: Vec<Message>) -> Self {
        let events = entries.into_iter().map(Event::Append).collect();
        Self::from_events(events)
    }

    /// Read the underlying event log (e.g. for persistence or scanning).
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    /// The only canonical mutation. Push a fully-formed message onto the log as a new
    /// `Event::Append`.
    pub fn append(&mut self, message: Message) {
        self.materialized.push(message.clone());
        self.events.push(Event::Append(message));
    }

    /// Read-only borrow of the materialized view. Providers and the scanner
    /// ([`crate::tools::extract_loaded_tool_names`]) consume this.
    pub fn as_slice(&self) -> &[Message] {
        &self.materialized
    }

    pub fn len(&self) -> usize {
        self.materialized.len()
    }

    pub fn is_empty(&self) -> bool {
        self.materialized.is_empty()
    }

    pub fn last(&self) -> Option<&Message> {
        self.materialized.last()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Message> {
        self.materialized.iter()
    }

    /// Text content of the most recent `Role::Assistant` message, or `None` when no assistant
    /// message exists. Walks backward, which is necessary because a turn that ended via tool-use
    /// leaves a `Role::User` tool-result trailer in the conversation, hiding the assistant's
    /// final text from the plain [`Self::last`].
    pub fn last_assistant_text(&self) -> Option<String> {
        self.materialized
            .iter()
            .rev()
            .find(|message| matches!(message.role, crate::provider::Role::Assistant))
            .map(|message| message.text_content())
    }

    /// Roll back an [`Conversation::append`] that did not reach the persistence layer. Used by
    /// `Agent::run_turn`'s error path when `save_message(user)` fails before any consumer could
    /// observe the message. Returns the popped message for diagnostics.
    ///
    /// Removes only a trailing `Event::Append`. If the last event is a `Event::CompactBoundary`
    /// (which can only be true after a successful compaction round-trip), this is a programmer
    /// error and the call returns `None` without mutating the log.
    pub fn pop_unsaved(&mut self) -> Option<Message> {
        match self.events.last() {
            Some(Event::Append(_)) => {}
            _ => return None,
        }
        let popped = match self.events.pop() {
            Some(Event::Append(message)) => message,
            _ => unreachable!("checked Append above"),
        };
        // Mirror the in-memory removal in the materialized view.
        self.materialized.pop();
        Some(popped)
    }

    /// Replace the visible window with `summary` followed by `tail`. Used by `compact_session`:
    /// appends one [`Event::CompactBoundary`] (which tells the materializer to truncate the prior
    /// tail and push the summary), then appends each kept tail message as an [`Event::Append`]. The
    /// events log itself is *only ever appended to*; pre-compaction events stay untouched in the
    /// log and on disk.
    ///
    /// `loaded_tools_snapshot` is the active deferred-tool set captured from the conversation
    /// *before* the boundary is appended. Carried so `extract_loaded_tool_names_from_events` can
    /// recover deferred tools after the boundary; otherwise a session that loaded a tool, then
    /// compacted, would fall back to the deferred state.
    pub fn replace_for_compaction(
        &mut self,
        summary: Message,
        tail: Vec<Message>,
        loaded_tools_snapshot: HashSet<String>,
    ) {
        let replaced_count = self.materialized.len();
        self.events.push(Event::CompactBoundary {
            summary: summary.clone(),
            replaced_count,
            loaded_tools_snapshot,
        });
        for message in tail {
            self.events.push(Event::Append(message));
        }
        self.rebuild_materialized();
        // Make sure `summary` is referenced even if `tail` is empty; the boundary's summary alone
        // is the visible head after the truncate. (Materialization handles this; the let-binding
        // above only exists to consume `summary`.)
        let _ = summary;
    }

    /// Drop every event preceding the most recent `CompactBoundary`.
    ///
    /// Those events are fully superseded: a `CompactBoundary` truncates all materialized messages
    /// before it and replaces them with its summary, and [`extract_loaded_tool_names_from_events`]
    /// reads the boundary's `loaded_tools_snapshot` rather than the events preceding it. So the
    /// materialized view and the recovered tool set are byte-identical before and after this call;
    /// it only stops the in-memory log from growing unbounded across a long-lived,
    /// repeatedly-compacted session.
    ///
    /// Persistence is unaffected: every event was already written to its own row by `save_event`,
    /// so the on-disk log stays complete.
    pub fn prune_compacted_events(&mut self) {
        let last_boundary = self
            .events
            .iter()
            .rposition(|event| matches!(event, Event::CompactBoundary { .. }));
        if let Some(index) = last_boundary
            && index > 0
        {
            self.events.drain(..index);
            self.rebuild_materialized();
        }
    }

    /// Drop assistant messages whose `tool_use` blocks lack matching `tool_result`s in the
    /// immediately-following user message. Returns the dropped messages so callers can log them.
    /// Used at session resume to repair the log after a crash mid-tool-call (the Anthropic API
    /// rejects orphaned `tool_use` blocks).
    ///
    /// Removes the corresponding `Event::Append` entries from the event log so future
    /// re-materializations stay clean. `Event::CompactBoundary` events are never touched (their
    /// synthetic summary is a plain user message that can't be orphaned).
    pub fn sanitize_orphans(&mut self) -> Vec<Message> {
        let dropped_indices = orphan_event_indices(&self.events);
        if dropped_indices.is_empty() {
            return Vec::new();
        }

        let mut dropped = Vec::with_capacity(dropped_indices.len());
        // Walk indices in reverse so each `swap_remove`-style remove doesn't invalidate the rest.
        // Use `remove` (linear) to preserve ordering; the dropped vector is filled in original
        // order via a post-sort.
        let mut to_remove = dropped_indices.clone();
        to_remove.sort_unstable_by(|a, b| b.cmp(a));
        for idx in to_remove {
            if let Event::Append(message) = self.events.remove(idx) {
                dropped.push(message);
            }
        }
        dropped.reverse();
        self.rebuild_materialized();
        dropped
    }

    fn rebuild_materialized(&mut self) {
        self.materialized.clear();
        for event in &self.events {
            match event {
                Event::Append(message) => self.materialized.push(message.clone()),
                Event::CompactBoundary {
                    summary,
                    replaced_count,
                    ..
                } => {
                    let truncate_to = self.materialized.len().saturating_sub(*replaced_count);
                    self.materialized.truncate(truncate_to);
                    self.materialized.push(summary.clone());
                }
            }
        }
    }
}

impl<'a> IntoIterator for &'a Conversation {
    type IntoIter = std::slice::Iter<'a, Message>;
    type Item = &'a Message;

    fn into_iter(self) -> Self::IntoIter {
        self.materialized.iter()
    }
}

/// Walk the event log and return the indices of `Event::Append` entries that carry orphaned
/// assistant `tool_use` blocks (i.e. no matching `tool_result` in the next materialized message).
/// The check uses the *materialized* view so a `CompactBoundary` between an orphan and its would-be
/// result correctly counts as orphaned.
fn orphan_event_indices(events: &[Event]) -> Vec<usize> {
    // Build (event_idx, &Message) pairs in materialization order so we can scan adjacency and
    // report orphan event indices, not just materialized indices. Skip the "previous Append is
    // gone" case (the event was truncated by a CompactBoundary) since the materialized view never
    // sees that orphan.
    let mut pairs: Vec<(usize, &Message)> = Vec::new();
    for (idx, event) in events.iter().enumerate() {
        match event {
            Event::Append(message) => pairs.push((idx, message)),
            Event::CompactBoundary { replaced_count, .. } => {
                let truncate_to = pairs.len().saturating_sub(*replaced_count);
                pairs.truncate(truncate_to);
                // The synthetic summary is not a real Append event, so we don't push a (idx, …)
                // pair for it; sanitization only removes Append events anyway.
            }
        }
    }

    let mut orphan = Vec::new();
    for window_idx in 0..pairs.len() {
        let (event_idx, message) = pairs[window_idx];
        if message.role != Role::Assistant {
            continue;
        }
        let tool_use_ids: Vec<&str> = message
            .content
            .iter()
            .filter_map(|block| {
                if let ContentBlock::ToolUse { id, .. } = block {
                    Some(id.as_str())
                } else {
                    None
                }
            })
            .collect();
        if tool_use_ids.is_empty() {
            continue;
        }

        let next = pairs.get(window_idx + 1).map(|(_, m)| *m);
        let has_results = next.is_some_and(|next_msg| {
            next_msg.role == Role::User
                && tool_use_ids.iter().all(|id| {
                    next_msg.content.iter().any(|block| {
                        matches!(block, ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == *id)
                    })
                })
        });

        if !has_results {
            orphan.push(event_idx);
        }
    }
    orphan
}

/// Walk events and collect the names of tools loaded via successful `load_tool` calls. Same
/// contract as [`crate::tools::extract_loaded_tool_names`] but events-aware so it can absorb
/// [`Event::CompactBoundary::loaded_tools_snapshot`] when it crosses a boundary. Pending uses
/// inside the summarized window are cleared at the boundary (the actual tool_use/tool_result rows
/// for those uses are still in the log on disk, but they're below the materialized view's "logical
/// start" so the model can't act on them).
/// Returns names in **load order**, de-duplicated. The order is what makes the tools array a stable
/// cache prefix: `load_tool` calls only ever append to the conversation, so appending each newly
/// loaded tool to the tail means the array can only grow at the end. Returning an unordered set and
/// letting the registry impose its own order would reinsert an earlier-registered tool ahead of a
/// later-registered one that was loaded first, which is a mid-array edit and re-caches the whole
/// conversation behind it.
pub fn extract_loaded_tool_names_from_events(events: &[Event]) -> Vec<String> {
    use std::collections::HashMap;
    let mut loaded: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut pending: HashMap<String, String> = HashMap::new();

    for event in events {
        match event {
            Event::Append(message) => {
                for block in &message.content {
                    match block {
                        ContentBlock::ToolUse { id, name, input }
                            if name == crate::tools::LOAD_TOOL_NAME =>
                        {
                            if let Some(loaded_name) = input.get("name").and_then(|v| v.as_str()) {
                                pending.insert(id.clone(), loaded_name.to_string());
                            }
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            is_error,
                            ..
                        } => {
                            if let Some(loaded_name) = pending.remove(tool_use_id)
                                && !is_error
                                && seen.insert(loaded_name.clone())
                            {
                                loaded.push(loaded_name);
                            }
                        }
                        _ => {}
                    }
                }
            }
            Event::CompactBoundary {
                loaded_tools_snapshot,
                ..
            } => {
                // Pending uses inside the summarized window are gone from the model's view; their
                // would-be results are also gone. Drop them and absorb the snapshot.
                pending.clear();
                // The snapshot is an unordered set, so sort it for a deterministic tail. Continuity
                // with the pre-boundary order isn't needed: compaction rewrites the head of the
                // conversation and re-caches everything anyway. What matters is that every turn
                // *after* the boundary agrees on the order.
                let mut absorbed: Vec<&String> = loaded_tools_snapshot.iter().collect();
                absorbed.sort();
                for name in absorbed {
                    if seen.insert(name.clone()) {
                        loaded.push(name.clone());
                    }
                }
            }
        }
    }

    loaded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assistant_with_tool_use(use_id: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: use_id.to_string(),
                name: "read_file".to_string(),
                input: serde_json::json!({"path": "/tmp/x"}),
            }],
        }
    }

    fn user_with_tool_result(use_id: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: use_id.to_string(),
                content: vec![crate::provider::ToolResultContent::Text {
                    text: "ok".to_string(),
                }],
                is_error: false,
            }],
        }
    }

    fn load_tool_use(id: &str, target: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: id.to_string(),
                name: crate::tools::LOAD_TOOL_NAME.to_string(),
                input: serde_json::json!({"name": target}),
            }],
        }
    }

    fn load_tool_result(use_id: &str, is_error: bool) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: use_id.to_string(),
                content: vec![crate::provider::ToolResultContent::Text {
                    text: "ok".to_string(),
                }],
                is_error,
            }],
        }
    }

    #[test]
    fn test_message_log_append_and_read() {
        let mut log = Conversation::new();
        log.append(Message::user("first"));
        log.append(Message::assistant_text("second"));
        log.append(Message::user("third"));

        assert_eq!(log.len(), 3);
        assert!(!log.is_empty());
        assert_eq!(log.as_slice().len(), 3);
        assert_eq!(log.as_slice()[0].text_content(), "first");
        assert_eq!(log.last().unwrap().text_content(), "third");
        let collected: Vec<&Message> = log.iter().collect();
        assert_eq!(collected.len(), 3);
    }

    #[test]
    fn test_last_assistant_text_walks_past_tool_results() {
        // Sub-agent turn shape after a tool-use round: assistant emits a tool_use, then the loop
        // appends the matching tool_result as a Role::User trailer. `last()` would return that
        // trailer, not the assistant's text; the helper has to walk backward.
        let mut log = Conversation::new();
        log.append(Message::user("kick off"));
        log.append(Message::assistant_text("final assistant answer"));
        log.append(user_with_tool_result("call_id"));

        assert_eq!(
            log.last_assistant_text().as_deref(),
            Some("final assistant answer")
        );
    }

    #[test]
    fn test_last_assistant_text_none_on_empty() {
        let log = Conversation::new();
        assert_eq!(log.last_assistant_text(), None);
    }

    #[test]
    fn test_last_assistant_text_none_when_no_assistant_message() {
        let mut log = Conversation::new();
        log.append(Message::user("only user message"));
        assert_eq!(log.last_assistant_text(), None);
    }

    #[test]
    fn test_message_log_replace_for_compaction_replaces_all() {
        let mut log = Conversation::new();
        log.append(Message::user("m1"));
        log.append(Message::assistant_text("m2"));
        log.append(Message::user("m3"));

        let summary = Message::user("[summary]");
        let tail = vec![Message::assistant_text("kept-1"), Message::user("kept-2")];
        log.replace_for_compaction(summary, tail, HashSet::new());

        let view = log.as_slice();
        assert_eq!(view.len(), 3);
        assert_eq!(view[0].text_content(), "[summary]");
        assert_eq!(view[1].text_content(), "kept-1");
        assert_eq!(view[2].text_content(), "kept-2");
    }

    #[test]
    fn test_message_log_replace_for_compaction_empty_tail() {
        let mut log = Conversation::new();
        log.append(Message::user("m1"));
        log.replace_for_compaction(Message::user("[summary]"), Vec::new(), HashSet::new());
        assert_eq!(log.len(), 1);
        assert_eq!(log.as_slice()[0].text_content(), "[summary]");
    }

    #[test]
    fn test_message_log_pop_unsaved() {
        let mut log = Conversation::new();
        log.append(Message::user("staying"));
        log.append(Message::user("rolling-back"));

        let popped = log.pop_unsaved();
        assert!(popped.is_some());
        assert_eq!(popped.unwrap().text_content(), "rolling-back");
        assert_eq!(log.len(), 1);
        assert_eq!(log.as_slice()[0].text_content(), "staying");
    }

    #[test]
    fn test_message_log_pop_unsaved_on_empty() {
        let mut log = Conversation::new();
        assert!(log.pop_unsaved().is_none());
    }

    #[test]
    fn test_message_log_sanitize_orphans_drops_unmatched_tool_use() {
        let mut log = Conversation::new();
        log.append(Message::user("hello"));
        log.append(assistant_with_tool_use("u1"));
        // No matching tool_result follows; the assistant message is orphaned.

        let dropped = log.sanitize_orphans();
        assert_eq!(dropped.len(), 1);
        assert_eq!(log.len(), 1);
        assert_eq!(log.as_slice()[0].text_content(), "hello");
    }

    #[test]
    fn test_message_log_sanitize_orphans_drops_truncated_multi_tool_use_tail() {
        // Reproduces the real corruption shape: a model response truncated at `max_tokens` while
        // emitting tools, persisted as a trailing assistant message with leading text plus several
        // `tool_use` blocks and no following `tool_result`. Anthropic rejects this on the next turn
        // ("tool_use ids were found without tool_result blocks"); sanitize must drop the whole
        // message regardless of the leading text or the number of tool_use blocks.
        let mut log = Conversation::new();
        log.append(Message::user("read the diff and explain it"));
        log.append(Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "I'll check the uncommitted changes.".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "u1".to_string(),
                    name: "execute_command".to_string(),
                    input: serde_json::json!({"command": "git diff"}),
                },
                ContentBlock::ToolUse {
                    id: "u2".to_string(),
                    name: "scratchpad_read".to_string(),
                    input: serde_json::json!({"name": "tool_1_output"}),
                },
            ],
        });

        let dropped = log.sanitize_orphans();
        assert_eq!(dropped.len(), 1);
        assert_eq!(log.len(), 1);
        assert_eq!(
            log.as_slice()[0].text_content(),
            "read the diff and explain it"
        );
        // Idempotent: a second pass on the now-clean log is a no-op.
        assert!(log.sanitize_orphans().is_empty());
    }

    #[test]
    fn test_message_log_sanitize_orphans_preserves_matched_tool_use() {
        let mut log = Conversation::new();
        log.append(Message::user("ask"));
        log.append(assistant_with_tool_use("u1"));
        log.append(user_with_tool_result("u1"));

        let dropped = log.sanitize_orphans();
        assert!(dropped.is_empty());
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn test_message_log_clone_independent() {
        let mut log = Conversation::new();
        log.append(Message::user("original"));
        let mut cloned = log.clone();
        cloned.append(Message::user("only-in-clone"));

        assert_eq!(log.len(), 1);
        assert_eq!(cloned.len(), 2);
    }

    #[test]
    fn test_message_log_into_iter_for_ref() {
        let mut log = Conversation::new();
        log.append(Message::user("a"));
        log.append(Message::user("b"));
        let texts: Vec<String> = (&log).into_iter().map(|m| m.text_content()).collect();
        assert_eq!(texts, vec!["a", "b"]);
    }

    #[test]
    fn test_events_are_append_only_after_compaction() {
        // After replace_for_compaction, the prior Append events MUST still be present in the events
        // log, even though the materialized view has truncated them. This is the structural
        // invariant: events in the log only ever grow.
        let mut log = Conversation::new();
        log.append(Message::user("m1"));
        log.append(Message::assistant_text("m2"));
        log.append(Message::user("m3"));
        let pre_event_count = log.events().len();

        log.replace_for_compaction(
            Message::user("[summary]"),
            vec![Message::user("tail")],
            HashSet::new(),
        );

        let post_event_count = log.events().len();
        // pre + 1 boundary + 1 tail Append = pre + 2.
        assert_eq!(post_event_count, pre_event_count + 2);
        // The original three Append events are still there.
        let append_count = log
            .events()
            .iter()
            .filter(|e| matches!(e, Event::Append(_)))
            .count();
        assert_eq!(append_count, pre_event_count + 1); // 3 + 1 tail
    }

    #[test]
    fn test_materialize_with_compact_boundary() {
        let mut log = Conversation::new();
        for i in 1..=5 {
            log.append(Message::user(format!("m{}", i)));
        }
        log.replace_for_compaction(
            Message::user("[summary]"),
            vec![Message::assistant_text("kept-1"), Message::user("kept-2")],
            HashSet::new(),
        );

        let view = log.as_slice();
        assert_eq!(view.len(), 3);
        assert_eq!(view[0].text_content(), "[summary]");
        assert_eq!(view[1].text_content(), "kept-1");
        assert_eq!(view[2].text_content(), "kept-2");
    }

    #[test]
    fn test_extract_loaded_tool_names_pure_appends() {
        let log = Conversation::from_vec(vec![
            load_tool_use("u1", "scratchpad_read"),
            load_tool_result("u1", false),
        ]);
        let loaded = extract_loaded_tool_names_from_events(log.events());
        assert!(loaded.iter().any(|name| name == "scratchpad_read"));
    }

    /// Load order, not registry order, is what makes the tools array append-only. A failed load
    /// contributes nothing, and a repeat load must not move a name to the back.
    #[test]
    fn test_extract_loaded_tool_names_preserves_load_order() {
        let log = Conversation::from_vec(vec![
            load_tool_use("u1", "omega"),
            load_tool_result("u1", false),
            load_tool_use("u2", "alpha"),
            load_tool_result("u2", false),
            load_tool_use("u3", "broken"),
            load_tool_result("u3", true),
            load_tool_use("u4", "omega"),
            load_tool_result("u4", false),
        ]);
        assert_eq!(extract_loaded_tool_names_from_events(log.events()), vec![
            "omega".to_string(),
            "alpha".to_string()
        ]);
    }

    #[test]
    fn test_extract_loaded_tool_names_recovers_snapshot_across_boundary() {
        // Pre-boundary: load_tool(scratchpad_read) succeeds. After the boundary swallows it, the
        // snapshot must restore scratchpad_read in the active set.
        let mut log = Conversation::new();
        log.append(load_tool_use("u1", "scratchpad_read"));
        log.append(load_tool_result("u1", false));

        let snapshot: HashSet<String> = ["scratchpad_read".to_string()].into_iter().collect();
        log.replace_for_compaction(Message::user("[summary]"), Vec::new(), snapshot);

        let loaded = extract_loaded_tool_names_from_events(log.events());
        assert!(loaded.iter().any(|name| name == "scratchpad_read"));
    }

    #[test]
    fn test_prune_compacted_events_drops_pre_boundary_log() {
        let mut log = Conversation::new();
        log.append(load_tool_use("u1", "scratchpad_read"));
        log.append(load_tool_result("u1", false));
        log.append(Message::user("m1"));

        let snapshot: HashSet<String> = ["scratchpad_read".to_string()].into_iter().collect();
        log.replace_for_compaction(
            Message::user("[summary-1]"),
            vec![Message::user("tail-1")],
            snapshot.clone(),
        );
        log.append(Message::assistant_text("m2"));
        log.replace_for_compaction(
            Message::user("[summary-2]"),
            vec![Message::user("tail-2")],
            snapshot,
        );

        let view_before: Vec<String> = log.as_slice().iter().map(|m| m.text_content()).collect();
        let loaded_before = extract_loaded_tool_names_from_events(log.events());

        log.prune_compacted_events();

        // Materialized view and recovered tool set are unchanged.
        let view_after: Vec<String> = log.as_slice().iter().map(|m| m.text_content()).collect();
        assert_eq!(view_before, view_after);
        assert_eq!(
            loaded_before,
            extract_loaded_tool_names_from_events(log.events())
        );
        assert!(
            extract_loaded_tool_names_from_events(log.events())
                .iter()
                .any(|name| name == "scratchpad_read"),
            "deferred tool must survive the prune"
        );

        // The log now starts at the last boundary; nothing precedes it.
        assert!(matches!(
            log.events().first(),
            Some(Event::CompactBoundary { .. })
        ));
        let boundary_count = log
            .events()
            .iter()
            .filter(|e| matches!(e, Event::CompactBoundary { .. }))
            .count();
        assert_eq!(boundary_count, 1, "only the last boundary should remain");
    }

    #[test]
    fn test_extract_loaded_tool_names_pending_use_wiped_at_boundary() {
        // load_tool tool_use lives on one side of the boundary, its tool_result on the other; both
        // vanish from the materialized view, so the scanner must NOT count the pending pair across
        // the boundary.
        let mut log = Conversation::new();
        log.append(load_tool_use("u1", "scratchpad_read"));
        // No tool_result yet.
        log.replace_for_compaction(
            Message::user("[summary]"),
            vec![load_tool_result("u1", false)],
            HashSet::new(),
        );

        let loaded = extract_loaded_tool_names_from_events(log.events());
        assert!(!loaded.iter().any(|name| name == "scratchpad_read"));
    }

    #[test]
    fn test_pop_unsaved_only_removes_trailing_append() {
        // After a CompactBoundary, the next legal call is `append`. A failed-save rollback after
        // that should remove the failed append, not the boundary.
        let mut log = Conversation::new();
        log.append(Message::user("pre"));
        log.replace_for_compaction(Message::user("[summary]"), Vec::new(), HashSet::new());
        log.append(Message::user("post-comp"));

        let popped = log.pop_unsaved();
        assert!(popped.is_some());
        assert_eq!(popped.unwrap().text_content(), "post-comp");

        // Boundary's summary survives.
        assert_eq!(log.len(), 1);
        assert_eq!(log.as_slice()[0].text_content(), "[summary]");

        // Calling pop_unsaved again must NOT eat the boundary.
        assert!(log.pop_unsaved().is_none());
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_from_vec_produces_append_events() {
        let log = Conversation::from_vec(vec![Message::user("a"), Message::assistant_text("b")]);
        assert_eq!(log.events().len(), 2);
        assert!(log.events().iter().all(|e| matches!(e, Event::Append(_))));
    }

    #[test]
    fn test_event_serializes_round_trip() {
        // Serialize one of each event variant and round-trip through JSON.
        let append = Event::Append(Message::user("hi"));
        let json = serde_json::to_string(&append).expect("serialize append");
        let back: Event = serde_json::from_str(&json).expect("deserialize append");
        match back {
            Event::Append(m) => assert_eq!(m.text_content(), "hi"),
            _ => panic!("wrong variant"),
        }

        let snapshot: HashSet<String> = ["mcp__notion__fetch".to_string()].into_iter().collect();
        let boundary = Event::CompactBoundary {
            summary: Message::user("[summary]"),
            replaced_count: 5,
            loaded_tools_snapshot: snapshot,
        };
        let json = serde_json::to_string(&boundary).expect("serialize boundary");
        let back: Event = serde_json::from_str(&json).expect("deserialize boundary");
        match back {
            Event::CompactBoundary {
                replaced_count,
                loaded_tools_snapshot,
                ..
            } => {
                assert_eq!(replaced_count, 5);
                assert!(loaded_tools_snapshot.contains("mcp__notion__fetch"));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_sanitize_orphans_does_not_touch_compact_boundary() {
        let mut log = Conversation::new();
        log.append(Message::user("u1"));
        log.append(Message::assistant_text("a1"));
        log.replace_for_compaction(Message::user("[summary]"), Vec::new(), HashSet::new());
        // Synthetic summary is a plain user message; sanitize must leave it.
        log.sanitize_orphans();
        assert_eq!(log.len(), 1);
        assert_eq!(log.as_slice()[0].text_content(), "[summary]");
    }
}
