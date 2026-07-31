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

/// One SSE event emitted on the wire. Monotonic `id` per turn, reserved for future
/// `Last-Event-ID` resumption (out of current spec scope).
#[derive(Debug, Clone)]
pub struct SseEvent {
    pub id: u64,
    pub event_type: SseEventType,
    pub data: serde_json::Value,
}

/// Stable event-name strings shipped on the wire. Keep these in lockstep with the HTTP API docs.
///
/// Lifecycle events (`turn.started`, `turn.finished`, `turn.failed`, `turn.cancelled`) are
/// emitted directly by the streaming-turn handler via `Event::default().event(...)` rather
/// than through this enum because they carry one-off envelopes (e.g. usage on
/// `turn.finished`); they don't pass through `translate`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SseEventType {
    AssistantTextDelta,
    ThinkingDelta,
    ToolCallExecuting,
    ToolCallCompleted,
    Notice,
    PermissionRequired,
}

impl SseEventType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AssistantTextDelta => "assistant_text.delta",
            Self::ThinkingDelta => "thinking.delta",
            Self::ToolCallExecuting => "tool_call.executing",
            Self::ToolCallCompleted => "tool_call.completed",
            Self::Notice => "notice",
            Self::PermissionRequired => "permission_required",
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

/// Per-turn event ID counter. Provides monotonic `id` values; reserved for future
/// `Last-Event-ID` resumption.
#[derive(Debug, Default)]
pub struct EventIdGenerator {
    next: AtomicU64,
}

impl EventIdGenerator {
    /// Returns a monotonic 0-based id. The spec's example stream shows `id: 0` on the first
    /// event (`turn.started`), so we match that convention exactly.
    pub fn next(&self) -> u64 {
        self.next.fetch_add(1, Ordering::Relaxed)
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
        FrontendEvent::ThinkingBlock { content, .. } => {
            if !capabilities.supports_reasoning_stream {
                return None;
            }
            (
                SseEventType::ThinkingDelta,
                serde_json::json!({ "text": content }),
            )
        }
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
        // `spawn_agent` tool result when it lands.
        FrontendEvent::TodoListUpdated { .. }
        | FrontendEvent::TokenUsage(_)
        | FrontendEvent::SubAgentActivity { .. }
        | FrontendEvent::McpProgress(_) => return None,
        FrontendEvent::Notice(notice) => (SseEventType::Notice, notice_view(notice)),
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
            signature: None,
        };
        let capabilities = SessionCapabilities {
            supports_reasoning_stream: false,
        };
        assert!(translate(event, capabilities).is_none());
    }

    #[test]
    fn translate_thinking_block_emits_when_capability_enabled() {
        let event = FrontendEvent::ThinkingBlock {
            content: "musing".into(),
            signature: None,
        };
        let capabilities = SessionCapabilities {
            supports_reasoning_stream: true,
        };
        let (event_type, _data) = translate(event, capabilities).expect("translates");
        assert_eq!(event_type, SseEventType::ThinkingDelta);
    }

    #[test]
    fn event_id_generator_is_monotonic_and_zero_based() {
        let generator = EventIdGenerator::default();
        assert_eq!(generator.next(), 0, "first id must be 0 per spec example");
        assert_eq!(generator.next(), 1);
        assert_eq!(generator.next(), 2);
    }
}
