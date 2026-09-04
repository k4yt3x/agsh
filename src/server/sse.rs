//! SSE event serialisation and the live broadcast channel that streaming turn responses read
//! from.
//!
//! Flat event taxonomy: each `FrontendEvent` variant maps to one named SSE event
//! (`turn.started`, `assistant_text.delta`, `tool_call.executing`, etc.).

use std::sync::atomic::{AtomicU64, Ordering};

use axum::response::sse::Event;
use serde::Serialize;

use crate::{
    frontend::FrontendEvent, provider::Notice, server::http_frontend::SessionCapabilities,
};

/// One SSE event emitted on the wire. Monotonic `id` per turn, which is what makes
/// `Last-Event-ID` resumption work: a re-attaching client names the last id it saw and the replay
/// ring hands back everything after it. See [`crate::server::http_frontend::TurnStream`].
#[derive(Debug, Clone)]
pub struct SseEvent {
    pub id: u64,
    pub event_type: SseEventType,
    pub data: serde_json::Value,
}

/// Stable event-name strings shipped on the wire. Keep these in lockstep with the HTTP API docs.
///
/// Lifecycle events (`turn.started`, `turn.finished`, `turn.failed`, `turn.cancelled`) do not pass
/// through [`translate`]: they carry one-off envelopes the turn handler assembles rather than
/// anything a `FrontendEvent` describes. They are named here, and ride on [`SseEvent`], because a
/// re-attaching client has to be able to receive a *terminal* event, and that means the terminal
/// has to be storable in the replay ring like everything else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SseEventType {
    AssistantTextDelta,
    ThinkingDelta,
    ToolCallComposing,
    ToolCallExecuting,
    ToolCallCompleted,
    Notice,
    PermissionRequired,
    ContextCompacted,
    TurnStarted,
    TurnFinished,
    TurnFailed,
    TurnCancelled,
}

impl SseEventType {
    /// Whether this event ends the stream. A client receiving one should expect the connection to
    /// close and must not wait for more.
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::TurnFinished | Self::TurnFailed | Self::TurnCancelled
        )
    }
}

impl SseEventType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AssistantTextDelta => "assistant_text.delta",
            Self::ThinkingDelta => "thinking.delta",
            Self::ToolCallComposing => "tool_call.composing",
            Self::ToolCallExecuting => "tool_call.executing",
            Self::ToolCallCompleted => "tool_call.completed",
            Self::Notice => "notice",
            Self::PermissionRequired => "permission_required",
            Self::ContextCompacted => "context.compacted",
            Self::TurnStarted => "turn.started",
            Self::TurnFinished => "turn.finished",
            Self::TurnFailed => "turn.failed",
            Self::TurnCancelled => "turn.cancelled",
        }
    }
}

impl SseEvent {
    /// Convert to an `axum::response::sse::Event` ready for the SSE response stream.
    pub fn into_axum(self) -> Event {
        Event::default()
            .id(self.id.to_string())
            .event(self.event_type.as_str())
            .json_data(self.data)
            .unwrap_or_else(|error| {
                // `json_data` only fails on serializer errors, which the variants below
                // never produce (all `Serialize` impls are for owned-data structs). Fall back
                // to a comment-line event so the stream stays alive if the impossible
                // happens.
                tracing::error!("SSE serialize failed: {}", error);
                Event::default().comment("serialize-failed")
            })
    }
}

/// Per-*session* event ID counter, owned by [`crate::server::http_frontend::HttpFrontend`].
///
/// Session-scoped rather than per-turn precisely so `Last-Event-ID` works: an id from a finished
/// turn sorts strictly below everything the current one emits, which is what makes the plain
/// `event.id > last` replay filter correct without knowing which turn an id belongs to.
#[derive(Debug, Default)]
pub struct EventIdGenerator {
    next: AtomicU64,
}

impl EventIdGenerator {
    /// Returns a monotonic 0-based id. The spec's example stream shows `id: 0` on the first
    /// event (`turn.started`) of a session, so the first turn matches that convention exactly;
    /// later turns continue the sequence rather than restarting.
    pub fn next(&self) -> u64 {
        self.next.fetch_add(1, Ordering::Relaxed)
    }

    /// The id the next event would get, i.e. one past the highest issued.
    ///
    /// A `Last-Event-ID` at or above this was never issued by this session at all: a fabricated
    /// value, or one carried over from a different session. Resumption discards it rather than
    /// filtering against it, which would hand back nothing.
    pub fn peek(&self) -> u64 {
        self.next.load(Ordering::Relaxed)
    }
}

/// Translate a `FrontendEvent` into the SSE wire shape. Unsupported variants return `None`.
/// `capabilities` is a defense-in-depth gate: even if a `ThinkingBlock` bypasses the
/// upstream filter, it won't leak onto the wire.
///
/// Returns the pair without an id so callers can defer `ids.next()` until after the filter
/// resolves, keeping the on-wire id sequence dense.
pub fn translate(
    event: FrontendEvent,
    capabilities: SessionCapabilities,
) -> Option<(SseEventType, serde_json::Value)> {
    let pair = match event {
        FrontendEvent::TurnStarted => {
            // The streaming handler emits a richer `turn.started` with extra fields;
            // suppress this bare form to avoid duplicate events.
            return None;
        }
        FrontendEvent::TurnFinished => {
            // stop_reason is unknown here; the turn handler emits the real `turn.finished`
            // after run_turn returns. This event is used as an internal end-of-stream marker.
            return None;
        }
        FrontendEvent::AssistantTextDelta(text) => (
            SseEventType::AssistantTextDelta,
            serde_json::json!({ "text": text }),
        ),
        FrontendEvent::ThinkingProgress { .. } => {
            // Not forwarded. Every event on this stream is content a client appends and keeps,
            // whereas this one is meant to be drawn over and erased; delivering it as a
            // `thinking.delta` would leave a trail of stale counters in the transcript. Giving
            // remote clients a live indicator means a distinct event type they can treat as
            // transient, which is an addition to the HTTP contract rather than a rendering change.
            return None;
        }
        FrontendEvent::ThinkingEnded => {
            // Closes a transient indicator this stream never carried; see `ThinkingProgress`.
            return None;
        }
        FrontendEvent::ThinkingDelta(text) => {
            if !capabilities.supports_reasoning_stream {
                return None;
            }
            (
                SseEventType::ThinkingDelta,
                serde_json::json!({ "text": text }),
            )
        }
        FrontendEvent::ThinkingBlock { .. } => {
            // Not forwarded. The deltas above already carried this text, and the agent emits them
            // for every block that has any -- including the ones a non-streaming provider hands
            // back whole -- so forwarding the block as well would deliver the reasoning twice. The
            // blocking response still reads it from the recorder, which sees both.
            return None;
        }
        // The only event that separates "the model is writing a message" from any other work in a
        // turn: assistant text is usually narration around a call, and `tool_call.executing`
        // arrives once the arguments are already written. A client rendering a typing indicator
        // holds it between this and the matching `tool_call.executing`.
        FrontendEvent::ToolCallComposing { id, name } => (
            SseEventType::ToolCallComposing,
            serde_json::json!({ "id": id, "name": name }),
        ),
        FrontendEvent::ToolCallStarted {
            id,
            name,
            input,
            display_summary,
        } => (
            SseEventType::ToolCallExecuting,
            serde_json::json!({
                "id": id,
                "name": name,
                "input": input,
                "display_summary": display_summary,
            }),
        ),
        FrontendEvent::ToolCallCompleted {
            id,
            is_error,
            content,
            ..
        } => (
            SseEventType::ToolCallCompleted,
            serde_json::json!({
                "id": id,
                "is_error": is_error,
                "content": tool_result_content_view(&content),
            }),
        ),
        // Metadata-only events: the recorder captures them for the blocking JSON / terminal
        // SSE payload but they don't get their own wire events.
        // `SubAgentActivity` is a progressive rewrite of one tool call's display, which only makes
        // sense against a stateful view like ACP's; the SSE stream already carries the sub-agent's
        // `agent_spawn` tool result when it lands.
        // `ToolCallOutputDelta` is left out for the same reason as `SubAgentActivity`: both are
        // progressive rewrites of one tool call's display, and the SSE stream carries the tool
        // result once, when it lands. Surfacing partial output here would add a wire event whose
        // consumers have to reassemble it, for a stream that already delivers the whole thing.
        FrontendEvent::TodoListUpdated { .. }
        | FrontendEvent::TokenUsage(_)
        | FrontendEvent::SubAgentActivity { .. }
        | FrontendEvent::ToolCallOutputDelta { .. }
        | FrontendEvent::McpProgress(_) => return None,
        FrontendEvent::Notice(notice) => (SseEventType::Notice, notice_view(notice)),
        // The one event a streaming client cannot infer. Everything else on this stream is
        // additive, so a client that misses one still holds a prefix of the truth; a compaction
        // *removes* messages it has already rendered, and without this it would only find out by
        // noticing `total` went down on the next `GET /messages`.
        FrontendEvent::Compacted {
            source,
            replaced_count,
            generation,
        } => (
            SseEventType::ContextCompacted,
            serde_json::json!({
                "source": source,
                "replaced_count": replaced_count,
                "generation": generation,
            }),
        ),
        FrontendEvent::SessionStarted { .. } => return None,
    };
    Some(pair)
}

fn tool_result_content_view(content: &[crate::provider::ToolResultContent]) -> Vec<TextOrImage> {
    content
        .iter()
        .map(|item| match item {
            crate::provider::ToolResultContent::Text { text } => {
                TextOrImage::Text { text: text.clone() }
            }
            crate::provider::ToolResultContent::Image { source } => TextOrImage::Image {
                media_type: source.media_type.clone(),
            },
        })
        .collect()
}

fn notice_view(notice: Notice) -> serde_json::Value {
    serde_json::json!({
        "level": match notice.level {
            crate::provider::NoticeLevel::Info => "info",
            crate::provider::NoticeLevel::Warn => "warn",
        },
        "text": notice.text,
    })
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TextOrImage {
    Text { text: String },
    Image { media_type: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translate_text_delta() {
        let event = FrontendEvent::AssistantTextDelta("hello".into());
        let (event_type, data) =
            translate(event, SessionCapabilities::default()).expect("translates");
        assert_eq!(event_type, SseEventType::AssistantTextDelta);
        assert_eq!(data["text"], "hello");
    }

    /// The composing event carries the name and nothing else, because nothing else has streamed
    /// yet. A client pairs it with the `tool_call.executing` on the same id.
    #[test]
    fn translate_tool_call_composing() {
        let event = FrontendEvent::ToolCallComposing {
            id: "tu_1".into(),
            name: "read_file".into(),
        };
        let (event_type, data) =
            translate(event, SessionCapabilities::default()).expect("translates");
        assert_eq!(event_type, SseEventType::ToolCallComposing);
        assert_eq!(event_type.as_str(), "tool_call.composing");
        assert_eq!(data["id"], "tu_1");
        assert_eq!(data["name"], "read_file");
    }

    #[test]
    fn translate_tool_call_started() {
        let event = FrontendEvent::ToolCallStarted {
            id: "tu_1".into(),
            name: "read_file".into(),
            input: serde_json::json!({"path": "/etc/hosts"}),
            display_summary: Some("/etc/hosts".into()),
        };
        let (event_type, data) =
            translate(event, SessionCapabilities::default()).expect("translates");
        assert_eq!(event_type, SseEventType::ToolCallExecuting);
        assert_eq!(data["id"], "tu_1");
        assert_eq!(data["display_summary"], "/etc/hosts");
    }

    #[test]
    fn translate_session_started_drops() {
        let event = FrontendEvent::SessionStarted {
            id: uuid::Uuid::nil(),
        };
        assert!(translate(event, SessionCapabilities::default()).is_none());
    }

    #[test]
    fn translate_thinking_block_drops_when_capability_disabled() {
        let event = FrontendEvent::ThinkingBlock {
            content: "musing".into(),
        };
        let capabilities = SessionCapabilities {
            supports_reasoning_stream: false,
            ..Default::default()
        };
        assert!(translate(event, capabilities).is_none());
    }

    #[test]
    fn translate_thinking_delta_emits_when_capability_enabled() {
        let capabilities = SessionCapabilities {
            supports_reasoning_stream: true,
            ..Default::default()
        };
        let (event_type, data) =
            translate(FrontendEvent::ThinkingDelta("musing".into()), capabilities)
                .expect("translates");
        assert_eq!(event_type, SseEventType::ThinkingDelta);
        assert_eq!(data["text"], "musing");
    }

    /// The deltas above already carried the block's text, and the agent emits them for every block
    /// that has any, so forwarding the block as well would put the reasoning on the wire twice.
    #[test]
    fn translate_drops_the_whole_thinking_block_the_deltas_already_carried() {
        let capabilities = SessionCapabilities {
            supports_reasoning_stream: true,
            ..Default::default()
        };
        assert!(
            translate(
                FrontendEvent::ThinkingBlock {
                    content: "musing".into(),
                },
                capabilities,
            )
            .is_none()
        );
    }

    #[test]
    fn event_id_generator_is_monotonic_and_zero_based() {
        let generator = EventIdGenerator::default();
        assert_eq!(generator.next(), 0, "first id must be 0 per spec example");
        assert_eq!(generator.next(), 1);
        assert_eq!(generator.next(), 2);
    }
}
