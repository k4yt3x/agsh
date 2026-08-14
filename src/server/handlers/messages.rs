//! Conversation history: `GET /v1/sessions/{id}/messages`.
//!
//! Returns the materialized `Conversation` view (post-compaction-aware) for clients that want
//! to read past turns. Pagination via `?limit=` and `?offset=` is intentionally simple; the
//! source of truth is the SQLite event log, which holds the full history.

use axum::{
    Extension, Json,
    extract::{Path, Query, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

use crate::{
    conversation::Event,
    provider::{ContentBlock, Message, Role, ToolResultContent},
    server::{
        auth::Principal,
        errors::{ErrorKind, ProblemDetail},
        scope,
        state::ServerState,
    },
};

#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct MessagesQuery {
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub offset: Option<usize>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct MessagesResponse {
    pub session_id: Uuid,
    pub messages: Vec<MessageView>,
    pub total: usize,
    /// How many times this conversation has been rewritten rather than appended to.
    ///
    /// Increments on every compaction, rewind, and mid-turn repair. A polling client that sees it
    /// change knows its copy is no longer a prefix of the server's and must re-fetch rather than
    /// diff; `total` alone cannot tell it apart from data loss.
    ///
    /// The per-message `compaction` marker explains one of those three. This covers the other two:
    /// a rewind removes messages with nothing left behind to attach a marker to, which would
    /// otherwise reproduce exactly the silent-rewrite failure the marker was added to prevent.
    pub revision: u64,
}

/// Lightweight wire view of a `Message`. Strips provider-internal blocks like `Thinking`
/// signatures down to their textual content, since callers consuming this endpoint are
/// typically just rendering a transcript.
#[derive(Debug, Serialize, ToSchema)]
pub struct MessageView {
    pub role: String,
    pub content: Vec<ContentBlockView>,
    /// RFC 3339 timestamp at which this message was persisted. `None` for messages produced by
    /// `assemble_response` in the same turn (no DB round-trip yet). Once the conversation has
    /// been read back via `GET /v1/sessions/{id}/messages`, every row has a `created_at`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    /// Virtual per-conversation turn correlator (`t_001`, `t_002`, …). Derived at query time
    /// by grouping every user-role message into a new turn that includes the assistant +
    /// tool-result messages that follow it. `None` on messages from the assembled-response
    /// path (no turn boundary known yet).
    ///
    /// Note: these are dense sequential indexes (`t_0001`, not UUIDs). The UUID-shaped
    /// `turn_id` on `POST /v1/sessions/{id}/turn` is a different identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    /// Present only on a message that *is* a compaction summary.
    ///
    /// Without this a client polling `/messages` watches history rewrite itself: a compaction
    /// truncates the materialised tail and pushes a summary in its place, so `total` shrinks and
    /// messages the client already rendered stop coming back. The marker is what lets it tell
    /// "the window was summarised" from "the server lost my conversation".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compaction: Option<CompactionMarker>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct CompactionMarker {
    /// How many materialised messages the boundary removed.
    ///
    /// The whole pre-compaction window, including the recent tail that compaction then re-appends
    /// verbatim, so it over-counts what the summary itself stands for. Use it to detect *that* the
    /// view was rewritten, not to compute how much was lost.
    pub replaced_count: usize,
    /// Which compaction produced it, counting from 1. Fidelity degrades with each pass, so a
    /// client rendering a transcript can say how far from the original it is.
    pub generation: u64,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockView {
    Text {
        text: String,
    },
    Image {
        // Signal an input image was present without the base64 payload, mirroring
        // `ToolResultContentView::Image`, so history responses stay tractable.
        media_type: String,
    },
    Thinking {
        thinking: String,
    },
    /// Encrypted reasoning whose contents the API withholds; surfaced as a presence marker so
    /// history responses note it occurred without exposing the opaque payload.
    RedactedThinking {},
    ToolUse {
        id: String,
        name: String,
        #[schema(value_type = Object)]
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        is_error: bool,
        content: Vec<ToolResultContentView>,
    },
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContentView {
    Text {
        text: String,
    },
    Image {
        // Just signal that an image was present; clients fetching the JSON history don't get
        // the full base64 payload to keep responses tractable.
        media_type: String,
    },
}

#[utoipa::path(
    get,
    path = "/v1/sessions/{id}/messages",
    tag = "sessions",
    params(
        ("id" = Uuid, Path, description = "Session UUID"),
        MessagesQuery,
    ),
    responses(
        (status = 200, description = "Page of conversation messages", body = MessagesResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn list_messages(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    Path(id): Path<Uuid>,
    Query(query): Query<MessagesQuery>,
) -> Result<Json<MessagesResponse>, ProblemDetail> {
    scope::require(&principal, "sessions:r")?;
    let events_with_ts = state
        .shared
        .session_manager
        .load_events_with_timestamps(id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to load session events", error)
        })?;
    if events_with_ts.is_empty() {
        // No events → either empty session or unknown. Distinguish via `session_exists`.
        // Propagate DB failure as 500 so the client retries instead of assuming 404.
        let exists = state
            .shared
            .session_manager
            .session_exists(id)
            .await
            .map_err(|error| {
                ProblemDetail::internal_sanitized("failed to verify session existence", error)
            })?;
        if !exists {
            return Err(ProblemDetail::new(
                ErrorKind::SessionNotFound,
                StatusCode::NOT_FOUND,
                format!("session '{}' does not exist", id),
            )
            .with("session_id", id.to_string()));
        }
    }
    // Materialize messages while keeping per-message timestamps aligned. Mirrors
    // `Conversation::rebuild_materialized` but with a parallel timestamp
    // vec: Event::Append pushes (timestamp, message); Event::CompactBoundary truncates the
    // tail and pushes (boundary_timestamp, summary). Result: `messages.len() == timestamps.len()`.
    let Materialized {
        messages: materialized,
        timestamps,
        markers,
        revision,
    } = materialize_with_timestamps(&events_with_ts);
    let total = materialized.len();
    let offset = query.offset.unwrap_or(0).min(total);
    let limit = query.limit.unwrap_or(200).min(1000);
    let end = (offset + limit).min(total);
    // Derive virtual turn indexes: every user-role message opens a new turn. Computed
    // across the *full* materialized view (not just the page) so paging doesn't shift the
    // correlator.
    let turn_indexes = derive_turn_indexes(&materialized);
    let messages = materialized[offset..end]
        .iter()
        .zip(timestamps[offset..end].iter())
        .zip(turn_indexes[offset..end].iter())
        .zip(markers[offset..end].iter())
        .map(|(((message, timestamp), turn_index), marker)| MessageView {
            role: match message.role {
                Role::User => "user".to_string(),
                Role::Assistant => "assistant".to_string(),
            },
            content: message.content.iter().map(view_for_block).collect(),
            created_at: Some(timestamp.clone()),
            turn_id: Some(turn_index.clone()),
            compaction: marker.clone(),
        })
        .collect();
    Ok(Json(MessagesResponse {
        session_id: id,
        messages,
        total,
        revision,
    }))
}

/// Replays the event log into a `(messages, timestamps)` pair where the two vectors are aligned
/// 1:1. `Event::Append` pushes the message + its created_at; `Event::CompactBoundary` truncates
/// the tail by `replaced_count` and pushes the boundary's summary + the boundary row's
/// created_at. Mirrors `Conversation::rebuild_materialized` byte-for-byte on the messages side.
fn materialize_with_timestamps(events: &[(String, Event)]) -> Materialized {
    let mut messages: Vec<Message> = Vec::with_capacity(events.len());
    let mut timestamps: Vec<String> = Vec::with_capacity(events.len());
    let mut markers: Vec<Option<CompactionMarker>> = Vec::with_capacity(events.len());
    let mut generation: u64 = 0;
    let mut revision: u64 = 0;
    for (ts, event) in events {
        match event {
            Event::Append(message) => {
                messages.push(message.clone());
                timestamps.push(ts.clone());
                markers.push(None);
            }
            Event::CompactBoundary {
                summary,
                replaced_count,
                ..
            } => {
                revision = revision.saturating_add(1);
                let truncate_to = messages.len().saturating_sub(*replaced_count);
                messages.truncate(truncate_to);
                timestamps.truncate(truncate_to);
                markers.truncate(truncate_to);
                messages.push(summary.clone());
                timestamps.push(ts.clone());
                generation = generation.saturating_add(1);
                // `replaced_count` as the event recorded it, not `messages.len() - truncate_to`:
                // a boundary can legitimately claim more messages than are still materialised
                // (a prior boundary already collapsed some), and the event's own figure is what
                // actually happened.
                markers.push(Some(CompactionMarker {
                    replaced_count: *replaced_count,
                    generation,
                }));
            }
            Event::Repair {
                replaced_count,
                messages: replacements,
            } => {
                revision = revision.saturating_add(1);
                let truncate_to = messages.len().saturating_sub(*replaced_count);
                messages.truncate(truncate_to);
                timestamps.truncate(truncate_to);
                markers.truncate(truncate_to);
                // The repair row's own timestamp, not the replaced messages', because that is when
                // this content came into being. Keeping the originals' would put the replacement
                // ahead of messages that predate it in a client sorting by time.
                for replacement in replacements {
                    messages.push(replacement.clone());
                    timestamps.push(ts.clone());
                    markers.push(None);
                }
            }
        }
    }
    Materialized {
        messages,
        timestamps,
        markers,
        revision,
    }
}

/// The replayed event log, as three vectors the caller indexes in lockstep.
///
/// A struct rather than a tuple because the invariant is that all three have the same length, and
/// every arm of the replay has to maintain it; a fourth parallel `Vec` returned loose is exactly
/// how that gets silently broken.
struct Materialized {
    messages: Vec<Message>,
    timestamps: Vec<String>,
    markers: Vec<Option<CompactionMarker>>,
    /// Count of non-`Append` events: every point at which the log was rewritten.
    revision: u64,
}

/// Group materialized messages into virtual turns. Every user-role message opens a new turn;
/// subsequent assistant + tool-result messages belong to that turn. Returns a parallel `Vec`
/// the same length as `messages` with `t_NNNN` indexes.
fn derive_turn_indexes(messages: &[Message]) -> Vec<String> {
    let mut counter: u32 = 0;
    let mut current = String::from("t_0001"); // placeholder for messages before the first user msg
    let mut ids = Vec::with_capacity(messages.len());
    for message in messages {
        if matches!(message.role, Role::User) {
            counter = counter.saturating_add(1);
            current = format!("t_{:04}", counter);
        }
        ids.push(current.clone());
    }
    ids
}

fn view_for_block(block: &ContentBlock) -> ContentBlockView {
    match block {
        ContentBlock::Text { text } => ContentBlockView::Text { text: text.clone() },
        ContentBlock::Image { source } => ContentBlockView::Image {
            media_type: source.media_type.clone(),
        },
        ContentBlock::Thinking { thinking, .. } => ContentBlockView::Thinking {
            thinking: thinking.clone(),
        },
        ContentBlock::RedactedThinking { .. } => ContentBlockView::RedactedThinking {},
        ContentBlock::ToolUse { id, name, input } => ContentBlockView::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: input.clone(),
        },
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => ContentBlockView::ToolResult {
            tool_use_id: tool_use_id.clone(),
            is_error: *is_error,
            content: content
                .iter()
                .map(|item| match item {
                    ToolResultContent::Text { text } => {
                        ToolResultContentView::Text { text: text.clone() }
                    }
                    ToolResultContent::Image { source } => ToolResultContentView::Image {
                        media_type: source.media_type.clone(),
                    },
                })
                .collect(),
        },
    }
}
