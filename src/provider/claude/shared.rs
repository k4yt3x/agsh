//! Helpers shared by [`super::api::ClaudeApiProvider`] and [`super::oauth::ClaudeOAuthProvider`].
//! Everything in this module is independent of the authentication scheme: message/tool conversion
//! to the Claude wire format, SSE streaming, response parsing, per-model capability detection, and
//! the thinking-override helper.

use std::{borrow::Cow, sync::atomic::AtomicI8};

use eventsource_stream::Eventsource;
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    error::{MekaError, Result},
    provider::{
        ContentBlock, Message, ModelInfo, Role, StopReason, StreamEvent, TokenUsage,
        ToolDefinition, ToolResultContent,
    },
};

/// Subset of the Anthropic `GET /v1/models/{id}` body carrying the model's context window.
/// Anthropic exposes no reasoning-effort levels here, so [`ModelInfo::effort_levels`] is always
/// `None` for Claude and effort falls to the name-based predicates.
#[derive(serde::Deserialize)]
struct ClaudeModelResponse {
    #[serde(default)]
    max_input_tokens: Option<u64>,
}

/// Parse an Anthropic model object into [`ModelInfo`]. Shared by both Claude providers (same wire
/// shape). `Err` on malformed JSON. The API reports `0` for an unset limit, so treat `0` as unknown
/// (`None`); returns `Ok(None)` when the context window is unknown.
pub(super) fn model_info_from_claude_model(body: &str) -> Result<Option<ModelInfo>> {
    let parsed: ClaudeModelResponse = serde_json::from_str(body)
        .map_err(|error| MekaError::Provider(format!("invalid Claude model JSON: {}", error)))?;
    let context_window = parsed.max_input_tokens.filter(|value| *value > 0);
    if context_window.is_none() {
        return Ok(None);
    }
    Ok(Some(ModelInfo {
        context_window,
        effort_levels: None,
    }))
}

/// Anthropic's hard request-body cap is 32 MiB; we reserve ~2 MiB headroom for headers, URL,
/// attestation patches, and serialization slack. Bodies above this threshold are reactively shrunk
/// by [`redact_oldest_images`] before they're posted.
pub(super) const MAX_REQUEST_BYTES: usize = 30 * 1024 * 1024;

/// When redaction fires, drop the body to roughly this size, which leaves a ~6 MiB buffer below
/// [`MAX_REQUEST_BYTES`] so the next several turns don't re-trigger redaction. Mirrors Claude
/// Code's `apiMicrocompact` watermark (180k → 140k = ~78% of trigger). Stable cache prefix between
/// redactions matters more than minimum-impact redaction per event.
pub(super) const REDACTION_TARGET_BYTES: usize = 24 * 1024 * 1024;

/// Placeholder text that replaces a `ToolResultContent::Image` payload when the request body would
/// otherwise exceed [`MAX_REQUEST_BYTES`].
pub(super) const IMAGE_REDACTION_PLACEHOLDER: &str = "[image redacted to fit request size budget]";

/// Anthropic accepts up to 8000 px per axis on a *single*-image request, but rejects anything over
/// 2000 px on either axis once the request contains more than one image. We always downscale to fit
/// so a session can freely accumulate images without tripping the multi-image cap. This is enforced
/// at the Claude provider layer only; non-Claude providers don't need it (and shouldn't pay the
/// resize cost).
pub(super) const MAX_IMAGE_DIMENSION_PX: u32 = 2000;

/// Extract a `TokenUsage` from an Anthropic `usage` object. Used by both the non-streaming response
/// parser and the SSE driver. Anthropic emits the same shape (`input_tokens`, `output_tokens`,
/// `cache_creation_input_tokens`, `cache_read_input_tokens`) in both places. Missing fields default
/// to 0 (older API responses, or providers that don't surface cache stats).
pub(super) fn parse_usage_object(usage: &serde_json::Value) -> TokenUsage {
    let field = |key: &str| usage.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
    TokenUsage {
        input_tokens: field("input_tokens"),
        output_tokens: field("output_tokens"),
        cache_creation_input_tokens: field("cache_creation_input_tokens"),
        cache_read_input_tokens: field("cache_read_input_tokens"),
    }
}

/// Resolves the effective thinking state given the override atomic's raw value (`-1` = unset, `0` =
/// forced off, `1` = forced on) and the configured default. Kept separate from the atomic itself so
/// the providers can own their own `AtomicI8` without duplicating the branching logic.
pub(super) fn is_thinking_enabled(override_raw: i8, default: bool) -> bool {
    match override_raw {
        0 => false,
        1 => true,
        _ => default,
    }
}

/// Convenience wrapper that loads the atomic with relaxed ordering and applies
/// [`is_thinking_enabled`]. Most callers want this form.
pub(super) fn resolve_thinking_enabled(override_atomic: &AtomicI8, default: bool) -> bool {
    is_thinking_enabled(
        override_atomic.load(std::sync::atomic::Ordering::Relaxed),
        default,
    )
}

/// Parse a `(major, minor)` version out of a Claude model name. The version is written as
/// hyphen-separated digit groups, which sit after the family on the 4.x line (`claude-opus-4-8`)
/// but before it on the 3.x line (`claude-3-5-sonnet`); in both layouts it is the first one or two
/// *short* numeric segments. The trailing date stamp (`-20250514`) is skipped because it has more
/// than two digits. The first short number becomes the major and the next the minor (defaulting to
/// 0); returns `None` when no version-like segment is present.
fn parse_model_version(model: &str) -> Option<(u32, u32)> {
    let mut numbers = model
        .split('-')
        .filter(|segment| {
            !segment.is_empty()
                && segment.len() <= 2
                && segment.bytes().all(|byte| byte.is_ascii_digit())
        })
        .filter_map(|segment| segment.parse::<u32>().ok());
    let major = numbers.next()?;
    let minor = numbers.next().unwrap_or(0);
    Some((major, minor))
}

/// Whether `model` is a known *pre-4.6* Claude model, i.e. one that predates adaptive thinking and
/// the `output_config.effort` knob (both shipped with Claude 4.6). Rather than allowlist the models
/// that have these features (which silently denies every newly released model, like
/// `claude-fable-5`, until the list is updated), we denylist the ones that don't: only the families
/// that existed before 4.6 - `opus`, `sonnet`, `haiku`, and the Claude 3.x line - can predate them,
/// and then only when their parsed version (see [`parse_model_version`]) is below 4.6. Any other
/// model - a new family such as `fable`, or anything without a recognised pre-4.6 family - is
/// assumed to support these features regardless of its version number.
fn model_predates_adaptive_thinking(model: &str) -> bool {
    const ADAPTIVE_MIN_VERSION: (u32, u32) = (4, 6);
    let lower = model.to_ascii_lowercase();
    let pre_adaptive_family =
        lower.contains("opus") || lower.contains("sonnet") || lower.contains("haiku");
    pre_adaptive_family
        && parse_model_version(&lower).is_some_and(|version| version < ADAPTIVE_MIN_VERSION)
}

/// Whether a Claude model supports adaptive thinking (`thinking: {type: "adaptive"}`, with no
/// explicit `budget_tokens`) rather than the older budgeted form. Enabled by default; only known
/// pre-4.6 models are excluded (see [`model_predates_adaptive_thinking`]).
pub(crate) fn model_supports_adaptive_thinking(model: &str) -> bool {
    !model_predates_adaptive_thinking(model)
}

pub(super) fn model_is_haiku(model: &str) -> bool {
    model.to_ascii_lowercase().contains("haiku")
}

/// Insert the `max_tokens` + `thinking` fields shared by both Claude providers' request bodies.
/// Adaptive-thinking models get a fixed 64k ceiling; others get `max(budget*2, 32k)` with an
/// explicit budget; thinking-off uses a flat 32k. A `max_output_tokens` override (the profile knob)
/// replaces whichever default would otherwise apply; on the budgeted path it is clamped to stay
/// above `budget_tokens` (the API rejects `max_tokens <= thinking.budget_tokens`).
///
/// No `display` field is set: real Claude Code 2.1.219 sends `{type:"adaptive"}` with no `display`
/// (verified by wire capture), so the model default applies.
pub(super) fn insert_thinking_fields(
    body: &mut serde_json::Map<String, serde_json::Value>,
    thinking_enabled: bool,
    model: &str,
    budget_tokens: u64,
    max_output_tokens: Option<u64>,
) {
    if thinking_enabled {
        if model_supports_adaptive_thinking(model) {
            let max_tokens = max_output_tokens.unwrap_or(64_000);
            body.insert("max_tokens".to_string(), serde_json::json!(max_tokens));
            body.insert(
                "thinking".to_string(),
                serde_json::json!({ "type": "adaptive" }),
            );
        } else {
            let default_max = std::cmp::max(budget_tokens * 2, 32_000);
            // Clamp above the budget so an override that's too small can't produce a 400.
            let max_tokens = max_output_tokens
                .unwrap_or(default_max)
                .max(budget_tokens.saturating_add(1));
            body.insert("max_tokens".to_string(), serde_json::json!(max_tokens));
            body.insert(
                "thinking".to_string(),
                serde_json::json!({
                    "type": "enabled",
                    "budget_tokens": budget_tokens
                }),
            );
        }
    } else {
        let max_tokens = max_output_tokens.unwrap_or(32_000);
        body.insert("max_tokens".to_string(), serde_json::json!(max_tokens));
    }
}

/// Mirrors Claude Code's `modelSupportsThinking` (and the equivalent
/// `modelSupportsISP` / `modelSupportsContextManagement`) on the 1P API:
/// any Claude 4+ model. Claude-3.x is excluded.
pub(super) fn model_supports_modern_features(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("claude") && !lower.contains("claude-3-")
}

/// Whether a Claude model supports the `output_config.effort` knob (the `effort-2025-11-24` beta).
/// Effort shipped with Claude 4.6, so adaptive-thinking-era models are enabled by default. The one
/// exception is Opus 4.5, which predates adaptive thinking (manual budgeted thinking, 200K window)
/// yet still supports effort (`low`/`medium`/`high`, no `xhigh`); it's carved out here so only the
/// other pre-4.6 gates keep treating it as old (see [`model_predates_adaptive_thinking`]).
///
/// Haiku is excluded by family rather than by version. The tier has never shipped an effort knob,
/// and the version gate below would wave through a future Haiku whose version clears 4.6, sending a
/// tier the model rejects. The costs are asymmetric: omitting effort is benign (`high` is defined
/// as identical to omitting the field) while sending an unsupported one is a 400, so the durable
/// family rule wins over the optimistic default that serves the frontier models.
pub(super) fn model_supports_effort(model: &str) -> bool {
    if model_is_haiku(model) {
        return false;
    }
    if !model_predates_adaptive_thinking(model) {
        return true;
    }
    let lower = model.to_ascii_lowercase();
    lower.contains("opus") && parse_model_version(&lower) == Some((4, 5))
}

/// Whether a Claude model supports the `xhigh` effort tier. Effort-capable, but excluding the
/// models that shipped effort without `xhigh`: Opus 4.5 (caps at `high`), the 4.6 line (Opus 4.6,
/// Sonnet 4.6), and Mythos Preview. The last three have `max` but not `xhigh`, which is why this
/// can't be derived from effort support alone: "`xhigh` is a newer level; some models that support
/// `max` don't support `xhigh`." Everything newer (Opus 4.7+, Sonnet 5, Fable/Mythos 5) and any
/// unrecognized future model supports it, mirroring the optimistic default in
/// [`model_supports_effort`].
pub(super) fn model_supports_xhigh(model: &str) -> bool {
    if !model_supports_effort(model) {
        return false;
    }
    let lower = model.to_ascii_lowercase();
    // Mythos Preview is versionless, so the generation check below can't catch it: it supports
    // effort (and `max`) but not `xhigh`, and would otherwise default to a tier the API rejects.
    if lower.contains("mythos-preview") {
        return false;
    }
    let no_xhigh = (lower.contains("opus") || lower.contains("sonnet"))
        && matches!(parse_model_version(&lower), Some((4, 5)) | Some((4, 6)));
    !no_xhigh
}

/// Resolve the `output_config.effort` value for `model` given the profile's explicit override
/// (`None` = unset). Returns `None` when the model has no effort knob (the caller then omits the
/// field). See [`crate::provider::resolve_effort_level`] for the shared policy.
pub(super) fn resolve_effort(configured: Option<&str>, model: &str) -> Option<String> {
    crate::provider::resolve_effort_level(
        configured,
        model_supports_effort(model),
        model_supports_xhigh(model),
    )
}

/// Whether a Claude model accepts the `temperature` sampling parameter. Mirrors Claude Code
/// 2.1.219's `mro`, which is an **allowlist** of the older models that still accept sampling
/// params: the Claude 3.x line, Opus 4.0/4.1/4.5/4.6, Sonnet 4.0/4.5/4.6, and Haiku 4.5. Everything
/// newer (Opus 4.7/4.8/5, Sonnet 5, Fable/Mythos 5) rejects `temperature` with a 400.
///
/// The allowlist direction is the point: an unrecognised model, which in practice means one newer
/// than this list, resolves to `false` and the parameter is omitted. A denylist would instead send
/// `temperature` to every future model and earn a 400 until the list was updated. Matching is by
/// family + [`parse_model_version`] rather than Claude Code's exact string equality because meka
/// accepts date-stamped ids (`claude-haiku-4-5-20251001`) as well as canonical ones.
pub(super) fn model_supports_temperature(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    if lower.contains("claude-3-") {
        return true;
    }
    let Some(version) = parse_model_version(&lower) else {
        return false;
    };
    if lower.contains("opus") {
        matches!(version, (4, 0) | (4, 1) | (4, 5) | (4, 6))
    } else if lower.contains("sonnet") {
        matches!(version, (4, 0) | (4, 5) | (4, 6))
    } else if lower.contains("haiku") {
        version == (4, 5)
    } else {
        false
    }
}

/// Whether a Claude model supports mid-conversation system messages (the
/// `mid-conversation-system-2026-04-07` beta). Claude Code sends this for Opus 4.8, Opus 5, Sonnet
/// 5, and the Fable/Mythos 5 family, but not for Opus 4.7, Sonnet 4.6, or Haiku 4.5 (verified
/// against Claude Code 2.1.219's `_er` gate plus the per-model `mid_conv_system` capability).
pub(super) fn model_supports_mid_conversation_system(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("opus-4-8")
        || lower.contains("opus-5")
        || lower.contains("sonnet-5")
        || lower.contains("fable-5")
        || lower.contains("mythos-5")
}

pub(super) fn convert_messages_to_claude_content(messages: &[Message]) -> Vec<serde_json::Value> {
    let message_count = messages.len();
    let mut claude_messages: Vec<serde_json::Value> = messages
        .iter()
        .enumerate()
        .map(|(message_index, message)| {
            let role = match message.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };

            let is_last_message = message_index + 1 == message_count;
            let block_count = message.content.len();

            let content: Vec<serde_json::Value> = message
                .content
                .iter()
                .enumerate()
                .map(|(block_index, block)| {
                    let mut value = match block {
                        ContentBlock::Text { text } => {
                            serde_json::json!({
                                "type": "text",
                                "text": text,
                            })
                        }
                        // `ImageSource` serializes to `{type:"base64", media_type, data}`, which is
                        // exactly Anthropic's image `source` object.
                        ContentBlock::Image { source } => {
                            serde_json::json!({
                                "type": "image",
                                "source": source,
                            })
                        }
                        ContentBlock::ToolUse { id, name, input } => {
                            serde_json::json!({
                                "type": "tool_use",
                                "id": id,
                                "name": name,
                                "input": input,
                            })
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => {
                            serde_json::json!({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": content,
                                "is_error": is_error,
                            })
                        }
                        ContentBlock::Thinking {
                            thinking,
                            signature,
                        } => {
                            let mut obj = serde_json::json!({
                                "type": "thinking",
                                "thinking": thinking
                            });
                            if let Some(sig) = signature {
                                obj["signature"] = serde_json::json!(sig);
                            }
                            obj
                        }
                        // Replayed verbatim: the API needs the opaque `data` unchanged to continue
                        // the redacted reasoning chain.
                        ContentBlock::RedactedThinking { data } => {
                            serde_json::json!({
                                "type": "redacted_thinking",
                                "data": data,
                            })
                        }
                    };

                    if is_last_message
                        && block_index + 1 == block_count
                        && let Some(obj) = value.as_object_mut()
                    {
                        obj.insert(
                            "cache_control".to_string(),
                            serde_json::json!({"type": "ephemeral", "ttl": "1h"}),
                        );
                    }

                    value
                })
                .collect();

            serde_json::json!({
                "role": role,
                "content": content,
            })
        })
        .collect();

    // Strip trailing thinking blocks from the last assistant message (Claude API requirement).
    if let Some(last_assistant) = claude_messages
        .iter_mut()
        .rev()
        .find(|msg| msg.get("role").and_then(|r| r.as_str()) == Some("assistant"))
        && let Some(content) = last_assistant
            .get_mut("content")
            .and_then(|c| c.as_array_mut())
    {
        while content
            .last()
            .and_then(|b| b.get("type"))
            .and_then(|t| t.as_str())
            == Some("thinking")
        {
            content.pop();
        }
        if content.is_empty() {
            content.push(serde_json::json!({
                "type": "text",
                "text": "[No message content]"
            }));
        }
    }

    claude_messages
}

pub(super) fn convert_tools_to_claude_tools(tools: &[ToolDefinition]) -> Vec<serde_json::Value> {
    // No per-tool `cache_control`: Claude Code's captured CLI wire leaves tools unmarked, and the
    // rolling last-message breakpoint already caches the tools+system prefix cumulatively, so the
    // extra tool breakpoint is redundant.
    tools
        .iter()
        .map(|tool| {
            serde_json::json!({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            })
        })
        .collect()
}

pub(super) fn parse_claude_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" => StopReason::EndTurn,
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        // Claude does not include the refusal text alongside the streaming `stop_reason` delta; the
        // model's text content is what the user sees as the refusal. Surface an empty refusal
        // payload; the assistant message blocks carry the human-readable explanation already.
        "refusal" => StopReason::Refusal(String::new()),
        // Log unrecognized reasons (e.g. `pause_turn`) so a recurrence is diagnosable; the raw
        // string is otherwise discarded once mapped to `Unknown`.
        other => {
            tracing::warn!(
                "claude returned unrecognized stop_reason {other:?}; mapping to Unknown"
            );
            StopReason::Unknown(other.to_string())
        }
    }
}

pub(super) fn parse_non_streaming_response(
    response: &serde_json::Value,
) -> Result<(Message, StopReason, TokenUsage)> {
    let stop_reason_str = response
        .get("stop_reason")
        .and_then(|reason| reason.as_str())
        .unwrap_or("end_turn");

    let stop_reason = parse_claude_stop_reason(stop_reason_str);

    let token_usage = response
        .get("usage")
        .map(parse_usage_object)
        .unwrap_or_default();

    let content_array = response
        .get("content")
        .and_then(|content| content.as_array())
        .ok_or_else(|| MekaError::Provider("no content array in response".to_string()))?;

    let mut content_blocks = Vec::new();

    for block in content_array {
        let block_type = block
            .get("type")
            .and_then(|block_type| block_type.as_str())
            .unwrap_or("");

        match block_type {
            "text" => {
                if let Some(text) = block.get("text").and_then(|text| text.as_str()) {
                    content_blocks.push(ContentBlock::Text {
                        text: text.to_string(),
                    });
                }
            }
            "tool_use" => {
                let id = block
                    .get("id")
                    .and_then(|id| id.as_str())
                    .ok_or_else(|| {
                        MekaError::Provider("tool_use block missing 'id' field".to_string())
                    })?
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(|name| name.as_str())
                    .ok_or_else(|| {
                        MekaError::Provider("tool_use block missing 'name' field".to_string())
                    })?
                    .to_string();
                let input = block.get("input").cloned().unwrap_or_else(|| {
                    tracing::warn!("missing 'input' in tool_use block");
                    serde_json::json!({})
                });

                content_blocks.push(ContentBlock::ToolUse { id, name, input });
            }
            "thinking" => {
                if let Some(thinking) = block.get("thinking").and_then(|t| t.as_str()) {
                    let signature = block
                        .get("signature")
                        .and_then(|s| s.as_str())
                        .map(|s| s.to_string());
                    content_blocks.push(ContentBlock::Thinking {
                        thinking: thinking.to_string(),
                        signature,
                    });
                }
            }
            "redacted_thinking" => {
                if let Some(data) = block.get("data").and_then(|d| d.as_str()) {
                    content_blocks.push(ContentBlock::RedactedThinking {
                        data: data.to_string(),
                    });
                } else {
                    tracing::warn!("redacted_thinking block missing 'data' field");
                }
            }
            _ => {
                tracing::warn!("unknown Claude content block type: {}", block_type);
            }
        }
    }

    Ok((
        Message {
            role: Role::Assistant,
            content: content_blocks,
        },
        stop_reason,
        token_usage,
    ))
}

/// Whether a mid-stream Anthropic `event: error`'s `error.type` indicates a transient condition
/// safe to retry. `overloaded_error` (529 capacity), `rate_limit_error`, and `api_error` are
/// documented as retryable; everything else (invalid_request_error, authentication_error,
/// permission_error, not_found_error, and any unrecognized type) defaults to *not* retryable —
/// conservative on purpose, since retrying a permanent failure would just burn the retry budget
/// before surfacing the real problem.
fn is_retryable_claude_error_type(error_type: &str) -> bool {
    matches!(
        error_type,
        "overloaded_error" | "rate_limit_error" | "api_error"
    )
}

pub(super) async fn drive_claude_sse_stream(
    response: reqwest::Response,
    event_sender: mpsc::Sender<StreamEvent>,
    cancellation: CancellationToken,
) -> Result<()> {
    let status = response.status();
    if !status.is_success() {
        let retry_after = crate::error::parse_retry_after(response.headers());
        let response_text = response.text().await.unwrap_or_else(|error| {
            tracing::warn!("failed to read Claude error response body: {}", error);
            String::new()
        });
        return Err(crate::error::provider_http_error(
            status,
            &response_text,
            retry_after,
        ));
    }

    let mut event_stream = response.bytes_stream().eventsource();

    let mut current_tool_input = String::new();
    let mut in_tool_use = false;
    let mut in_thinking = false;
    let mut current_thinking_signature: Option<String> = None;

    loop {
        tokio::select! {
            _ = cancellation.cancelled() => {
                return Err(MekaError::Interrupted);
            }
            event = event_stream.next() => {
                let Some(event) = event else {
                    break;
                };

                match event {
                    Ok(event) => {
                        let data: serde_json::Value = match serde_json::from_str(&event.data) {
                            Ok(data) => data,
                            Err(error) => {
                                tracing::warn!("failed to parse SSE data: {}", error);
                                continue;
                            }
                        };

                        match event.event.as_str() {
                            "content_block_start" => {
                                let Some(content_block) = data.get("content_block") else {
                                    continue;
                                };
                                let block_type = content_block
                                    .get("type")
                                    .and_then(|block_type| block_type.as_str())
                                    .unwrap_or("");

                                if block_type == "thinking" {
                                    in_thinking = true;
                                    // Announce the block itself, before any estimate: this is the
                                    // earliest point the pause becomes explainable, and on a
                                    // redacted block it is otherwise the only thing that happens
                                    // for seconds at a time.
                                    if event_sender
                                        .send(StreamEvent::ThinkingProgress {
                                            estimated_tokens: None,
                                        })
                                        .await
                                        .is_err()
                                    {
                                        tracing::trace!("stream event receiver dropped");
                                        break;
                                    }
                                } else if block_type == "redacted_thinking" {
                                    // The opaque `data` arrives whole in the start event; forward it
                                    // so the agent can replay it verbatim on later turns.
                                    if let Some(data) =
                                        content_block.get("data").and_then(|d| d.as_str())
                                        && event_sender
                                            .send(StreamEvent::RedactedThinking {
                                                data: data.to_string(),
                                            })
                                            .await
                                            .is_err()
                                    {
                                        tracing::trace!("stream event receiver dropped");
                                        break;
                                    }
                                } else if block_type == "tool_use" {
                                    let id = content_block
                                        .get("id")
                                        .and_then(|id| id.as_str())
                                        .ok_or_else(|| {
                                            MekaError::Provider(
                                                "tool_use block missing 'id' field".to_string(),
                                            )
                                        })?
                                        .to_string();
                                    let name = content_block
                                        .get("name")
                                        .and_then(|name| name.as_str())
                                        .ok_or_else(|| {
                                            MekaError::Provider(
                                                "tool_use block missing 'name' field"
                                                    .to_string(),
                                            )
                                        })?
                                        .to_string();

                                    current_tool_input.clear();
                                    in_tool_use = true;
                                    if event_sender
                                        .send(StreamEvent::ToolUseStart { id, name })
                                        .await
                                        .is_err()
                                    {
                                        tracing::trace!("stream event receiver dropped");
                                        break;
                                    }
                                }
                            }
                            "content_block_delta" => {
                                let Some(delta) = data.get("delta") else {
                                    continue;
                                };
                                let delta_type = delta
                                    .get("type")
                                    .and_then(|delta_type| delta_type.as_str())
                                    .unwrap_or("");

                                match delta_type {
                                    "thinking_delta" => {
                                        // `estimated_tokens` is the server's running count of
                                        // thinking spent so far (the `thinking-token-count` beta).
                                        // It is the only progress signal on a redacted block,
                                        // where `thinking` is `""` on every delta. The final delta
                                        // of a block carries `null`; skipping it leaves the last
                                        // real figure on screen rather than blanking the display
                                        // just before the block ends.
                                        if let Some(estimated) =
                                            delta.get("estimated_tokens").and_then(|t| t.as_u64())
                                            && event_sender.send(
                                                StreamEvent::ThinkingProgress {
                                                    estimated_tokens: Some(estimated),
                                                },
                                            ).await.is_err() {
                                                tracing::trace!("stream event receiver dropped");
                                                break;
                                            }
                                        if let Some(thinking) = delta.get("thinking").and_then(|t| t.as_str())
                                            && !thinking.is_empty()
                                                && event_sender.send(
                                                    StreamEvent::ThinkingDelta(thinking.to_string()),
                                                ).await.is_err() {
                                                    tracing::trace!("stream event receiver dropped");
                                                    break;
                                                }
                                    }
                                    "text_delta" => {
                                        if let Some(text) = delta.get("text").and_then(|text| text.as_str())
                                            && !text.is_empty()
                                                && event_sender.send(
                                                    StreamEvent::TextDelta(text.to_string()),
                                                ).await.is_err() {
                                                    tracing::trace!("stream event receiver dropped");
                                                    break;
                                                }
                                    }
                                    "signature_delta" => {
                                        if let Some(sig) = delta.get("signature").and_then(|s| s.as_str()) {
                                            current_thinking_signature = Some(
                                                current_thinking_signature
                                                    .map_or_else(|| sig.to_string(), |existing| existing + sig),
                                            );
                                        }
                                    }
                                    "input_json_delta" => {
                                        if let Some(partial_json) =
                                            delta.get("partial_json").and_then(|partial_json| partial_json.as_str())
                                        {
                                            current_tool_input.push_str(partial_json);
                                            if event_sender.send(
                                                StreamEvent::ToolInputDelta(
                                                    partial_json.to_string(),
                                                ),
                                            ).await.is_err() {
                                                tracing::trace!("stream event receiver dropped");
                                                break;
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            "content_block_stop" => {
                                if in_thinking {
                                    in_thinking = false;
                                    let signature = current_thinking_signature.take();
                                    if event_sender
                                        .send(StreamEvent::ThinkingComplete { signature })
                                        .await
                                        .is_err()
                                    {
                                        tracing::trace!("stream event receiver dropped");
                                        break;
                                    }
                                } else if in_tool_use {
                                    let input = if current_tool_input.is_empty() {
                                        serde_json::json!({})
                                    } else {
                                        match serde_json::from_str(&current_tool_input) {
                                            Ok(value) => value,
                                            Err(error) => {
                                                tracing::warn!("failed to parse tool input JSON: {}", error);
                                                serde_json::json!({})
                                            }
                                        }
                                    };
                                    if event_sender
                                        .send(StreamEvent::ToolUseEnd { input })
                                        .await
                                        .is_err()
                                    {
                                        tracing::trace!("stream event receiver dropped");
                                        break;
                                    }
                                    current_tool_input.clear();
                                    in_tool_use = false;
                                }
                            }
                            "message_delta" => {
                                let Some(delta) = data.get("delta") else {
                                    continue;
                                };
                                if let Some(usage) = data.get("usage") {
                                    let token_usage = parse_usage_object(usage);
                                    if event_sender
                                        .send(StreamEvent::Usage(token_usage))
                                        .await
                                        .is_err()
                                    {
                                        tracing::trace!("stream event receiver dropped");
                                        break;
                                    }
                                }
                                if let Some(stop_reason_str) =
                                    delta.get("stop_reason").and_then(|reason| reason.as_str())
                                {
                                    let stop_reason = parse_claude_stop_reason(stop_reason_str);
                                    if event_sender
                                        .send(StreamEvent::MessageEnd { stop_reason })
                                        .await
                                        .is_err()
                                    {
                                        tracing::trace!("stream event receiver dropped");
                                        break;
                                    }
                                }
                            }
                            "message_stop" => {
                                break;
                            }
                            "message_start" => {
                                if let Some(usage) =
                                    data.get("message").and_then(|m| m.get("usage"))
                                {
                                    let token_usage = parse_usage_object(usage);
                                    if event_sender
                                        .send(StreamEvent::Usage(token_usage))
                                        .await
                                        .is_err()
                                    {
                                        tracing::trace!("stream event receiver dropped");
                                        break;
                                    }
                                }
                            }
                            "ping" => {}
                            // Anthropic can send this *after* the 200 response has already started
                            // streaming (typically right after `message_start`, before any visible
                            // content) — e.g. `overloaded_error` during a capacity spike. Previously
                            // fell into the `other` catch-all below and was silently dropped,
                            // making an overloaded turn look like it succeeded with truncated/empty
                            // content. Forward it on the channel for visibility, then return the
                            // classified error directly so the caller can decide whether to retry.
                            "error" => {
                                let error_type = data
                                    .get("error")
                                    .and_then(|error| error.get("type"))
                                    .and_then(|kind| kind.as_str())
                                    .unwrap_or("unknown")
                                    .to_string();
                                let message = data
                                    .get("error")
                                    .and_then(|error| error.get("message"))
                                    .and_then(|message| message.as_str())
                                    .unwrap_or("stream error event")
                                    .to_string();
                                if event_sender
                                    .send(StreamEvent::Error(message.clone()))
                                    .await
                                    .is_err()
                                {
                                    tracing::trace!("stream event receiver dropped");
                                }
                                return Err(if is_retryable_claude_error_type(&error_type) {
                                    MekaError::RetryableProvider {
                                        message,
                                        retry_after: None,
                                    }
                                } else {
                                    MekaError::Provider(message)
                                });
                            }
                            other => {
                                tracing::debug!("unknown Claude SSE event: {}", other);
                            }
                        }
                    }
                    Err(error) => {
                        if event_sender
                            .send(StreamEvent::Error(error.to_string()))
                            .await
                            .is_err()
                        {
                            tracing::trace!("stream event receiver dropped");
                        }
                        return Err(MekaError::StreamError(error.to_string()));
                    }
                }
            }
        }
    }

    Ok(())
}

/// Stats from a single [`redact_oldest_images`] invocation. Returned to callers so they can surface
/// a user-visible advisory and increment a per-session redaction counter.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct RedactionStats {
    pub images_redacted: usize,
    pub bytes_freed: usize,
}

/// Walk `messages` oldest-first and replace `ToolResultContent::Image` payloads with
/// [`IMAGE_REDACTION_PLACEHOLDER`] until at least `bytes_to_drop` base64 bytes have been removed.
/// The LAST message is never touched; it carries the moving `cache_control` breakpoint set in
/// [`convert_messages_to_claude_content`] and disturbing it would invalidate the cache for the new
/// turn unnecessarily.
///
/// Returns `Cow::Borrowed` if no work was needed (`bytes_to_drop == 0`). Otherwise returns
/// `Cow::Owned` with whatever redaction was possible. Even when the budget couldn't be met, the
/// cloned messages are still returned so the caller can re-serialize and decide whether the body
/// fits.
pub(super) fn redact_oldest_images(
    messages: &[Message],
    bytes_to_drop: usize,
) -> (Cow<'_, [Message]>, RedactionStats) {
    if bytes_to_drop == 0 || messages.len() <= 1 {
        return (Cow::Borrowed(messages), RedactionStats::default());
    }

    let mut redacted: Vec<Message> = messages.to_vec();
    let last = redacted.len() - 1;
    let mut stats = RedactionStats::default();

    'outer: for message in &mut redacted[..last] {
        for block in &mut message.content {
            match block {
                ContentBlock::ToolResult { content, .. } => {
                    for item in content.iter_mut() {
                        if let ToolResultContent::Image { source } = item {
                            stats.bytes_freed = stats.bytes_freed.saturating_add(source.data.len());
                            stats.images_redacted = stats.images_redacted.saturating_add(1);
                            *item = ToolResultContent::Text {
                                text: IMAGE_REDACTION_PLACEHOLDER.to_string(),
                            };
                            if stats.bytes_freed >= bytes_to_drop {
                                break 'outer;
                            }
                        }
                    }
                }
                // Input images (ACP @-mentions) count toward the same 32 MiB cap; collapse them to
                // the placeholder text just like tool-result images.
                ContentBlock::Image { source } => {
                    let freed = source.data.len();
                    stats.bytes_freed = stats.bytes_freed.saturating_add(freed);
                    stats.images_redacted = stats.images_redacted.saturating_add(1);
                    *block = ContentBlock::Text {
                        text: IMAGE_REDACTION_PLACEHOLDER.to_string(),
                    };
                    if stats.bytes_freed >= bytes_to_drop {
                        break 'outer;
                    }
                }
                _ => {}
            }
        }
    }

    (Cow::Owned(redacted), stats)
}

/// Walk `messages` and downscale any `ToolResultContent::Image` whose pixel dimensions exceed
/// [`MAX_IMAGE_DIMENSION_PX`] on either axis. The body bytes (base64) for those images are replaced
/// with a re-encoded PNG that fits within the cap; smaller images are left alone. Returns
/// `Cow::Borrowed` when no work was needed.
///
/// Anthropic-specific: the 2000 px cap only matters for Anthropic's multi-image requests; this
/// helper is intentionally not applied to non-Claude providers. Decode/resize cost is incurred per
/// turn for each oversized image, but typical sessions have few oversized images, and the cheap
/// [`crate::image::read_image_dimensions`] header read short-circuits the common case.
pub(super) fn downscale_oversized_images(messages: &[Message]) -> Cow<'_, [Message]> {
    use base64::Engine;
    use image::ImageFormat;

    fn parse_format(media_type: &str) -> Option<ImageFormat> {
        ImageFormat::from_mime_type(media_type)
    }

    // True when this image decodes and exceeds the per-axis pixel cap.
    fn oversized(source: &crate::provider::ImageSource) -> bool {
        let Some(format) = parse_format(&source.media_type) else {
            return false;
        };
        let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(&source.data) else {
            return false;
        };
        crate::image::read_image_dimensions(&bytes, format)
            .map(|(w, h)| w > MAX_IMAGE_DIMENSION_PX || h > MAX_IMAGE_DIMENSION_PX)
            .unwrap_or(false)
    }

    // Re-encode `source` to a within-cap PNG in place; no-op if it can't be decoded or already
    // fits.
    fn downscale_in_place(source: &mut crate::provider::ImageSource) {
        let Some(format) = parse_format(&source.media_type) else {
            return;
        };
        let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(&source.data) else {
            return;
        };
        let Ok((w, h)) = crate::image::read_image_dimensions(&bytes, format) else {
            return;
        };
        if w <= MAX_IMAGE_DIMENSION_PX && h <= MAX_IMAGE_DIMENSION_PX {
            return;
        }
        match crate::image::downscale_to_dim_cap(&bytes, format, MAX_IMAGE_DIMENSION_PX) {
            Ok(png) => {
                source.media_type = "image/png".to_string();
                source.data = base64::engine::general_purpose::STANDARD.encode(&png);
            }
            Err(error) => {
                tracing::warn!(
                    "failed to downscale {}x{} {} image: {}",
                    w,
                    h,
                    source.media_type,
                    error,
                );
            }
        }
    }

    // First pass: detect whether any image - tool-result OR input (`ContentBlock::Image`) - needs
    // downscaling. Cheap header read; if nothing's oversized, skip the clone+rewrite and return
    // Cow::Borrowed.
    let needs_work = messages.iter().any(|message| {
        message.content.iter().any(|block| match block {
            ContentBlock::ToolResult { content, .. } => content.iter().any(
                |item| matches!(item, ToolResultContent::Image { source } if oversized(source)),
            ),
            ContentBlock::Image { source } => oversized(source),
            _ => false,
        })
    });
    if !needs_work {
        return Cow::Borrowed(messages);
    }

    let mut owned: Vec<Message> = messages.to_vec();
    for message in owned.iter_mut() {
        for block in message.content.iter_mut() {
            match block {
                ContentBlock::ToolResult { content, .. } => {
                    for item in content.iter_mut() {
                        if let ToolResultContent::Image { source } = item {
                            downscale_in_place(source);
                        }
                    }
                }
                ContentBlock::Image { source } => downscale_in_place(source),
                _ => {}
            }
        }
    }
    Cow::Owned(owned)
}

/// Serialize a Claude request body, downscaling oversized images first and reactively redacting old
/// tool-result image blocks if the serialized JSON still exceeds [`MAX_REQUEST_BYTES`]. Both Claude
/// providers run this same redact-and-retry loop; the caller supplies the body builder via `build`
/// so each provider's thinking / metadata wiring stays in its own file.
///
/// `build` takes a `messages` slice (the downscaled-then-maybe-redacted view) and returns the
/// serialized JSON. It's called once on the original messages and, if oversized, a second time on
/// the redacted set.
///
/// Returns the serialized body plus an optional [`crate::provider::Notice`]: on a successful
/// redaction pass, the notice describes what was dropped so the caller can forward it to the active
/// frontend (REPL renders via `render_hint`; ACP surfaces in the session/update stream). On the
/// happy path (no redaction needed), the notice is `None`. The function also records
/// [`RedactionStats`] on `session_stats` when one is provided.
pub(super) fn build_body_within_budget<F>(
    messages: &[Message],
    session_stats: Option<&std::sync::Arc<crate::stats::SessionStats>>,
    mut build: F,
) -> Result<(String, Option<crate::provider::Notice>)>
where
    F: FnMut(&[Message]) -> Result<String>,
{
    let prepared = downscale_oversized_images(messages);
    let body_json = build(prepared.as_ref())?;

    if body_json.len() <= MAX_REQUEST_BYTES {
        return Ok((body_json, None));
    }

    let bytes_to_drop = body_json.len() - REDACTION_TARGET_BYTES;
    let (redacted, stats) = redact_oldest_images(prepared.as_ref(), bytes_to_drop);
    let body_json = build(redacted.as_ref())?;

    if body_json.len() > MAX_REQUEST_BYTES {
        return Err(MekaError::Provider(format!(
            "request body is {} MiB after redacting old tool-result images; \
             Anthropic's limit is 32 MiB. Run /compact, remove large attachments \
             from the most recent turn, or split the work across smaller turns.",
            body_json.len() / 1_048_576,
        )));
    }

    if let Some(session_stats) = session_stats {
        session_stats.record_redaction(stats.images_redacted as u64, stats.bytes_freed as u64);
    }
    let notice_text = format!(
        "Redacted {} old image{} (~{} MiB freed). Cache prefix invalidated for those messages.",
        stats.images_redacted,
        if stats.images_redacted == 1 { "" } else { "s" },
        stats.bytes_freed / 1_048_576,
    );
    tracing::warn!(
        "redacted {} old tool-result image(s); body now {} MiB",
        stats.images_redacted,
        body_json.len() / 1_048_576,
    );
    Ok((body_json, Some(crate::provider::Notice::info(notice_text))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ImageSource;

    #[test]
    fn test_model_info_from_claude_model() {
        let info = model_info_from_claude_model(
            r#"{"id":"claude-opus-4-6","max_input_tokens":200000,"max_tokens":64000}"#,
        )
        .unwrap()
        .expect("known context window");
        assert_eq!(info.context_window, Some(200_000));
        // Anthropic exposes no effort levels; effort falls to the name predicates.
        assert_eq!(info.effort_levels, None);
        // The API reports 0 for an unset limit → treated as unknown.
        assert!(
            model_info_from_claude_model(r#"{"max_input_tokens":0}"#)
                .unwrap()
                .is_none()
        );
        // Malformed JSON → Err.
        assert!(model_info_from_claude_model("{nope").is_err());
    }

    #[test]
    fn test_is_retryable_claude_error_type() {
        for retryable in ["overloaded_error", "rate_limit_error", "api_error"] {
            assert!(
                is_retryable_claude_error_type(retryable),
                "{retryable} should be retryable"
            );
        }
        for permanent in [
            "invalid_request_error",
            "authentication_error",
            "permission_error",
            "not_found_error",
            "unknown",
            "",
        ] {
            assert!(
                !is_retryable_claude_error_type(permanent),
                "{permanent} should not be retryable"
            );
        }
    }

    #[test]
    fn test_convert_messages_serializes_input_image() {
        let message =
            crate::provider::Message::user_with_images("look at this", vec![ImageSource {
                source_type: "base64".to_string(),
                media_type: "image/png".to_string(),
                data: "QUJD".to_string(),
            }]);
        let converted = convert_messages_to_claude_content(&[message]);
        let blocks = converted[0]["content"].as_array().expect("content array");
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "image");
        assert_eq!(blocks[1]["source"]["type"], "base64");
        assert_eq!(blocks[1]["source"]["media_type"], "image/png");
        assert_eq!(blocks[1]["source"]["data"], "QUJD");
    }

    #[test]
    fn test_parse_redacted_thinking_block() {
        let response = serde_json::json!({
            "content": [
                { "type": "redacted_thinking", "data": "ENCRYPTED_OPAQUE_BLOB" },
                { "type": "text", "text": "done" },
            ],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 1, "output_tokens": 1 },
        });
        let (message, ..) = parse_non_streaming_response(&response).unwrap();
        assert!(matches!(
            &message.content[0],
            ContentBlock::RedactedThinking { data } if data == "ENCRYPTED_OPAQUE_BLOB"
        ));
    }

    #[test]
    fn test_redacted_thinking_round_trips_verbatim() {
        let message = crate::provider::Message {
            role: crate::provider::Role::Assistant,
            content: vec![
                ContentBlock::RedactedThinking {
                    data: "ENCRYPTED_OPAQUE_BLOB".to_string(),
                },
                ContentBlock::Text {
                    text: "hi".to_string(),
                },
            ],
        };
        let converted = convert_messages_to_claude_content(&[message]);
        let block = &converted[0]["content"].as_array().unwrap()[0];
        assert_eq!(block["type"], "redacted_thinking");
        assert_eq!(block["data"], "ENCRYPTED_OPAQUE_BLOB");
        // The opaque data must never be wrapped in `thinking`/`signature` fields.
        assert!(block.get("thinking").is_none());
        assert!(block.get("signature").is_none());
    }

    #[test]
    fn test_empty_thinking_block_with_signature_serializes_signature() {
        let message = crate::provider::Message {
            role: crate::provider::Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    thinking: String::new(),
                    signature: Some("SIG620".to_string()),
                },
                ContentBlock::Text {
                    text: "answer".to_string(),
                },
            ],
        };
        let converted = convert_messages_to_claude_content(&[message]);
        let block = &converted[0]["content"].as_array().unwrap()[0];
        assert_eq!(block["type"], "thinking");
        assert_eq!(block["thinking"], "");
        assert_eq!(block["signature"], "SIG620");
    }

    #[test]
    fn test_redact_oldest_images_redacts_input_image() {
        let big = "x".repeat(2_000);
        let messages = vec![
            crate::provider::Message::user_with_images("first", vec![ImageSource {
                source_type: "base64".to_string(),
                media_type: "image/png".to_string(),
                data: big.clone(),
            }]),
            // A trailing message: the last message is never redacted.
            crate::provider::Message::assistant_text("ok"),
        ];
        let (redacted, stats) = redact_oldest_images(&messages, 1_000);
        assert_eq!(stats.images_redacted, 1);
        assert!(stats.bytes_freed >= 2_000);
        // The input image block became a text placeholder.
        assert!(matches!(&redacted[0].content[1], ContentBlock::Text { .. }));
    }

    #[test]
    fn test_insert_thinking_fields_max_output_default_thinking_off() {
        let mut body = serde_json::Map::new();
        insert_thinking_fields(&mut body, false, "claude-opus-4-8", 10_000, None);
        assert_eq!(body["max_tokens"], 32_000);
        // Thinking off: no `thinking` object is emitted.
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn test_insert_thinking_fields_max_output_override_thinking_off() {
        let mut body = serde_json::Map::new();
        insert_thinking_fields(&mut body, false, "claude-opus-4-8", 10_000, Some(50_000));
        assert_eq!(body["max_tokens"], 50_000);
    }

    #[test]
    fn test_insert_thinking_fields_max_output_override_adaptive() {
        // `opus-4-6` is an adaptive-thinking model; the override replaces the 64k default.
        let mut body = serde_json::Map::new();
        insert_thinking_fields(&mut body, true, "claude-opus-4-6", 10_000, Some(80_000));
        assert_eq!(body["max_tokens"], 80_000);
        assert_eq!(body["thinking"]["type"], "adaptive");
        // No `display` field: real Claude Code sends `{type:"adaptive"}` only.
        assert!(body["thinking"].get("display").is_none());
    }

    #[test]
    fn test_insert_thinking_fields_max_output_clamped_above_budget() {
        // Non-adaptive thinking sends an explicit budget; an override below it is clamped so the
        // API's `max_tokens > budget_tokens` invariant holds.
        let mut body = serde_json::Map::new();
        insert_thinking_fields(&mut body, true, "claude-3-5-sonnet", 20_000, Some(5_000));
        assert_eq!(body["max_tokens"], 20_001);
        assert_eq!(body["thinking"]["budget_tokens"], 20_000);
    }

    #[test]
    fn test_is_thinking_enabled_override_off() {
        assert!(!is_thinking_enabled(0, true));
        assert!(!is_thinking_enabled(0, false));
    }

    #[test]
    fn test_is_thinking_enabled_override_on() {
        assert!(is_thinking_enabled(1, true));
        assert!(is_thinking_enabled(1, false));
    }

    #[test]
    fn test_is_thinking_enabled_unset_uses_default() {
        assert!(is_thinking_enabled(-1, true));
        assert!(!is_thinking_enabled(-1, false));
        // Any non-0/1 value should be treated as "unset" and fall through to the configured
        // default.
        assert!(is_thinking_enabled(42, true));
        assert!(!is_thinking_enabled(-99, false));
    }

    #[test]
    fn test_model_supports_modern_features() {
        assert!(model_supports_modern_features("claude-opus-4-6-20250514"));
        assert!(model_supports_modern_features("claude-sonnet-4-20250514"));
        assert!(model_supports_modern_features("claude-haiku-4-5-20251001"));
        assert!(!model_supports_modern_features(
            "claude-3-5-sonnet-20241022"
        ));
        assert!(!model_supports_modern_features("claude-3-opus-20240229"));
        assert!(!model_supports_modern_features("gpt-4o"));
    }

    #[test]
    fn test_model_supports_effort() {
        assert!(model_supports_effort("claude-opus-4-6-20250514"));
        assert!(model_supports_effort("claude-sonnet-4-6"));
        // Opus 4.5 predates adaptive thinking but still supports effort (the one exception).
        assert!(model_supports_effort("claude-opus-4-5-20250929"));
        // Older / non-4-6 sonnet/opus/haiku are explicitly denied (Sonnet 4.5 is NOT
        // effort-capable).
        assert!(!model_supports_effort("claude-sonnet-4-5"));
        assert!(!model_supports_effort("claude-sonnet-4-20250514"));
        assert!(!model_supports_effort("claude-opus-4-1"));
        assert!(!model_supports_effort("claude-haiku-4-5-20251001"));
        // Haiku is denied by family, not version: a future Haiku that clears the 4.6 boundary must
        // still omit effort rather than send a tier the tier has never had.
        for model in [
            "claude-haiku-5",
            "claude-haiku-5-1",
            "claude-haiku-6-20270101",
        ] {
            assert!(!model_supports_effort(model), "{model}");
            assert!(!model_supports_xhigh(model), "{model}");
            assert_eq!(resolve_effort(None, model), None, "{model}");
        }
        // Unknown 1P model defaults to true.
        assert!(model_supports_effort("claude-future-experimental-7"));
        // New families are effort-capable by default, even with a low version number.
        assert!(model_supports_effort("claude-fable-5"));
        assert!(model_supports_effort("claude-fable-2"));
    }

    #[test]
    fn test_model_supports_xhigh() {
        // Opus 4.5 and the 4.6 generation are effort-capable but have no xhigh.
        assert!(!model_supports_xhigh("claude-opus-4-5-20250929"));
        assert!(!model_supports_xhigh("claude-opus-4-6-20250514"));
        assert!(!model_supports_xhigh("claude-sonnet-4-6"));
        // 4.7+ / Opus 5 / Sonnet 5 / new families support xhigh.
        assert!(model_supports_xhigh("claude-opus-4-8"));
        assert!(model_supports_xhigh("claude-opus-4-7"));
        assert!(model_supports_xhigh("claude-opus-5"));
        assert!(model_supports_xhigh("claude-sonnet-5"));
        assert!(model_supports_xhigh("claude-fable-5"));
        assert!(model_supports_xhigh("claude-future-experimental-7"));
        // Non-effort models never reach xhigh.
        assert!(!model_supports_xhigh("claude-haiku-4-5-20251001"));
        assert!(!model_supports_xhigh("claude-opus-4-1"));
        // Mythos Preview takes effort and `max` but NOT `xhigh`, and carries no version to sort it
        // into the 4.5/4.6 carve-out; defaulting it to xhigh would be a 400.
        assert!(model_supports_effort("claude-mythos-preview"));
        assert!(!model_supports_xhigh("claude-mythos-preview"));
        assert_eq!(
            resolve_effort(None, "claude-mythos-preview").as_deref(),
            Some("high")
        );
    }

    #[test]
    fn test_resolve_effort() {
        // Unset: best tier the model supports.
        assert_eq!(
            resolve_effort(None, "claude-opus-4-8").as_deref(),
            Some("xhigh")
        );
        assert_eq!(
            resolve_effort(None, "claude-opus-5").as_deref(),
            Some("xhigh")
        );
        assert_eq!(
            resolve_effort(None, "claude-sonnet-5").as_deref(),
            Some("xhigh")
        );
        assert_eq!(
            resolve_effort(None, "claude-opus-4-6").as_deref(),
            Some("high")
        );
        // Explicit values are absolute: passed through verbatim, never clamped.
        assert_eq!(
            resolve_effort(Some("xhigh"), "claude-opus-4-6").as_deref(),
            Some("xhigh")
        );
        assert_eq!(
            resolve_effort(Some("low"), "claude-opus-4-8").as_deref(),
            Some("low")
        );
        assert_eq!(
            resolve_effort(Some("max"), "claude-opus-4-6").as_deref(),
            Some("max")
        );
        // ...even on a model the default would omit effort for.
        assert_eq!(
            resolve_effort(Some("high"), "claude-opus-4-1").as_deref(),
            Some("high")
        );
        // Unset on a legacy / non-effort model omits the field entirely.
        assert_eq!(resolve_effort(None, "claude-haiku-4-5-20251001"), None);
        // A blank override (empty / whitespace) is treated as unset → model-aware default, never an
        // empty `output_config.effort` the API would reject.
        assert_eq!(
            resolve_effort(Some("  "), "claude-opus-4-8").as_deref(),
            Some("xhigh")
        );
        assert_eq!(
            resolve_effort(Some(""), "claude-opus-4-6").as_deref(),
            Some("high")
        );
        assert_eq!(
            resolve_effort(Some("   "), "claude-haiku-4-5-20251001"),
            None
        );
    }

    #[test]
    fn test_parse_model_version() {
        assert_eq!(parse_model_version("claude-opus-4-8"), Some((4, 8)));
        // Trailing date stamp is ignored (too many digits).
        assert_eq!(
            parse_model_version("claude-opus-4-6-20250514"),
            Some((4, 6))
        );
        assert_eq!(parse_model_version("claude-sonnet-4-5"), Some((4, 5)));
        // 3.x line carries the version before the family.
        assert_eq!(
            parse_model_version("claude-3-5-sonnet-20241022"),
            Some((3, 5))
        );
        // A single version segment -> minor defaults to 0.
        assert_eq!(parse_model_version("claude-3-opus-20240229"), Some((3, 0)));
        assert_eq!(
            parse_model_version("claude-sonnet-4-20250514"),
            Some((4, 0))
        );
        // No version-like segment at all.
        assert_eq!(parse_model_version("claude-custom"), None);
    }

    #[test]
    fn test_model_supports_adaptive_thinking_by_version() {
        // 4.6 is the cutoff: 4.6 and newer use adaptive thinking.
        assert!(model_supports_adaptive_thinking("claude-opus-4-6"));
        assert!(model_supports_adaptive_thinking("claude-sonnet-4-6"));
        // Regression: opus 4.8 was wrongly denied by the old hardcoded allowlist.
        assert!(model_supports_adaptive_thinking("claude-opus-4-8"));
        assert!(model_supports_adaptive_thinking("claude-opus-5-0"));
        assert!(model_supports_adaptive_thinking("claude-opus-5"));
        assert!(model_supports_adaptive_thinking("claude-sonnet-5"));
        assert!(model_supports_adaptive_thinking("claude-opus-4-6-20250514"));
        // Older than 4.6 -> budgeted thinking.
        assert!(!model_supports_adaptive_thinking("claude-sonnet-4-5"));
        assert!(!model_supports_adaptive_thinking("claude-opus-4-1"));
        assert!(!model_supports_adaptive_thinking(
            "claude-haiku-4-5-20251001"
        ));
        assert!(!model_supports_adaptive_thinking(
            "claude-3-5-sonnet-20241022"
        ));
        assert!(!model_supports_adaptive_thinking("claude-3-opus-20240229"));
        // Unparseable version -> default to adaptive (matches Claude Code's 1P default).
        assert!(model_supports_adaptive_thinking(
            "claude-future-experimental"
        ));
        // New model families are adaptive by default; the denylist only excludes known pre-4.6
        // families.
        assert!(model_supports_adaptive_thinking("claude-fable-5"));
        // A new family is adaptive even with a low version number (the old `>= 4.6` rule would
        // wrongly deny this).
        assert!(model_supports_adaptive_thinking("claude-fable-2"));
    }

    #[test]
    fn test_model_is_haiku() {
        assert!(model_is_haiku("claude-haiku-4-5-20251001"));
        assert!(model_is_haiku("claude-haiku-4-5"));
        assert!(!model_is_haiku("claude-opus-4-6-20250514"));
        assert!(!model_is_haiku("claude-sonnet-4-20250514"));
    }

    #[test]
    fn test_model_supports_temperature() {
        // The allowlist: the 3.x line, Opus 4.0/4.1/4.5/4.6, Sonnet 4.0/4.5/4.6, Haiku 4.5. Dated
        // and canonical spellings both resolve (`claude-sonnet-4-20250514` parses as 4.0).
        for model in [
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20241022",
            "claude-opus-4-0",
            "claude-opus-4-1",
            "claude-opus-4-5-20251101",
            "claude-opus-4-6-20250514",
            "claude-sonnet-4-20250514",
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            "claude-haiku-4-5-20251001",
        ] {
            assert!(model_supports_temperature(model), "{model}");
        }
        // Sampling-params-removed models (400 on `temperature`).
        for model in [
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-fable-5",
            "claude-mythos-5",
        ] {
            assert!(!model_supports_temperature(model), "{model}");
        }
        // The allowlist fails safe: anything it doesn't recognise (a model newer than this list)
        // omits `temperature` rather than earning a 400. This is what a denylist got wrong.
        for model in [
            "claude-opus-6",
            "claude-sonnet-6-0",
            "claude-future-experimental-7",
            "claude-custom",
        ] {
            assert!(!model_supports_temperature(model), "{model}");
        }
    }

    #[test]
    fn test_model_supports_mid_conversation_system() {
        // Opus 4.8, Opus 5, Sonnet 5, and the Fable/Mythos 5 family send the beta.
        for model in [
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-opus-5-0",
            "claude-sonnet-5",
            "claude-fable-5",
            "claude-mythos-5",
        ] {
            assert!(model_supports_mid_conversation_system(model), "{model}");
        }
        // Opus 4.7, Sonnet 4.6, Haiku 4.5, and older do not.
        for model in [
            "claude-opus-4-7",
            "claude-opus-4-6-20250514",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ] {
            assert!(!model_supports_mid_conversation_system(model), "{model}");
        }
    }

    #[test]
    fn test_parse_claude_stop_reason_all_variants() {
        assert_eq!(parse_claude_stop_reason("end_turn"), StopReason::EndTurn);
        assert_eq!(parse_claude_stop_reason("tool_use"), StopReason::ToolUse);
        assert_eq!(
            parse_claude_stop_reason("max_tokens"),
            StopReason::MaxTokens
        );
        assert_eq!(
            parse_claude_stop_reason("refusal"),
            StopReason::Refusal(String::new())
        );
        // `pause_turn` is unrecognized and maps to `Unknown` carrying the raw string, so the
        // literal reaches the warn log and the empty-turn stand-in intact.
        assert_eq!(
            parse_claude_stop_reason("pause_turn"),
            StopReason::Unknown("pause_turn".to_string())
        );
        assert_eq!(
            parse_claude_stop_reason("something_else"),
            StopReason::Unknown("something_else".to_string())
        );
    }

    fn image_block(tool_use_id: &str, payload: &str) -> ContentBlock {
        ContentBlock::ToolResult {
            tool_use_id: tool_use_id.to_string(),
            content: vec![ToolResultContent::Image {
                source: ImageSource {
                    source_type: "base64".to_string(),
                    media_type: "image/png".to_string(),
                    data: payload.to_string(),
                },
            }],
            is_error: false,
        }
    }

    fn user_with_block(block: ContentBlock) -> Message {
        Message {
            role: Role::User,
            content: vec![block],
        }
    }

    fn assistant_text(text: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
        }
    }

    #[test]
    fn test_redact_no_op_when_under_threshold() {
        let messages = vec![
            user_with_block(image_block("call_a", "AAAA")),
            assistant_text("ack"),
        ];
        let (result, stats) = redact_oldest_images(&messages, 0);
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(stats.images_redacted, 0);
        assert_eq!(stats.bytes_freed, 0);
    }

    #[test]
    fn test_redact_drops_oldest_image_first() {
        // Two images: one in msg[0] (older), one in msg[1] (last). The helper must only touch the
        // older one; the last message carries the moving cache_control marker.
        let payload_a = "A".repeat(1024);
        let payload_b = "B".repeat(1024);
        let messages = vec![
            user_with_block(image_block("call_a", &payload_a)),
            user_with_block(image_block("call_b", &payload_b)),
        ];
        let (result, stats) = redact_oldest_images(&messages, 1);
        assert_eq!(stats.images_redacted, 1);
        assert_eq!(stats.bytes_freed, 1024);
        let owned = match result {
            Cow::Owned(v) => v,
            Cow::Borrowed(_) => panic!("expected owned redacted vec"),
        };
        // msg[0] image redacted to placeholder text.
        match &owned[0].content[0] {
            ContentBlock::ToolResult { content, .. } => match &content[0] {
                ToolResultContent::Text { text } => {
                    assert_eq!(text, IMAGE_REDACTION_PLACEHOLDER);
                }
                other => panic!("expected text placeholder, got {:?}", other),
            },
            other => panic!("expected ToolResult, got {:?}", other),
        }
        // msg[1] (last) image untouched.
        match &owned[1].content[0] {
            ContentBlock::ToolResult { content, .. } => match &content[0] {
                ToolResultContent::Image { source } => {
                    assert_eq!(source.data, payload_b);
                }
                other => panic!("expected untouched image, got {:?}", other),
            },
            other => panic!("expected ToolResult, got {:?}", other),
        }
    }

    #[test]
    fn test_redact_stops_when_target_reached() {
        // Three images each 1 KiB. Target = 1500 bytes. Only the FIRST image should be redacted;
        // the second remains because we hit the budget after one (1024 >= 1500 is false, but
        // saturating_add gets us past after the first redaction since we then loop-check before the
        // second image is considered? No: the check is `bytes_dropped >= bytes_to_drop`, so 1024
        // < 1500 means we redact the second too). Clarify by setting target = 1024.
        let payload = "X".repeat(1024);
        let messages = vec![
            user_with_block(image_block("call_a", &payload)),
            user_with_block(image_block("call_b", &payload)),
            assistant_text("end"),
        ];
        let (result, stats) = redact_oldest_images(&messages, 1024);
        assert_eq!(stats.images_redacted, 1);
        assert_eq!(stats.bytes_freed, 1024);
        let owned = match result {
            Cow::Owned(v) => v,
            Cow::Borrowed(_) => panic!("expected owned"),
        };
        // First image redacted.
        match &owned[0].content[0] {
            ContentBlock::ToolResult { content, .. } => match &content[0] {
                ToolResultContent::Text { text } => {
                    assert_eq!(text, IMAGE_REDACTION_PLACEHOLDER);
                }
                _ => panic!("first should be redacted"),
            },
            _ => unreachable!(),
        }
        // Second image preserved (budget already met).
        match &owned[1].content[0] {
            ContentBlock::ToolResult { content, .. } => match &content[0] {
                ToolResultContent::Image { .. } => {}
                _ => panic!("second image should still be intact"),
            },
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_redact_preserves_last_message() {
        // Single image, in the LAST message. Helper must not touch it even when the budget is huge.
        let payload = "P".repeat(8 * 1024);
        let messages = vec![
            assistant_text("setup"),
            user_with_block(image_block("call_only", &payload)),
        ];
        let (result, stats) = redact_oldest_images(&messages, usize::MAX);
        // No redactable images outside the last message → 0 redactions.
        assert_eq!(stats.images_redacted, 0);
        assert_eq!(stats.bytes_freed, 0);
        let owned = match result {
            Cow::Owned(v) => v,
            Cow::Borrowed(_) => panic!("expected owned (cloned even when no redaction)"),
        };
        match &owned[1].content[0] {
            ContentBlock::ToolResult { content, .. } => match &content[0] {
                ToolResultContent::Image { source } => assert_eq!(source.data, payload),
                _ => panic!("last-message image must survive"),
            },
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_redact_handles_no_images() {
        let messages = vec![
            assistant_text("hello"),
            assistant_text("world"),
            assistant_text("end"),
        ];
        let (result, stats) = redact_oldest_images(&messages, 1024);
        assert_eq!(stats.images_redacted, 0);
        assert_eq!(stats.bytes_freed, 0);
        let owned = match result {
            Cow::Owned(v) => v,
            Cow::Borrowed(_) => panic!("expected owned (cloned even when no images)"),
        };
        assert_eq!(owned.len(), 3);
        for (orig, copy) in messages.iter().zip(owned.iter()) {
            assert_eq!(orig.content.len(), copy.content.len());
        }
    }

    fn synthesize_png_base64(width: u32, height: u32) -> String {
        use std::io::Cursor;

        use base64::Engine;
        use image::{ImageFormat, RgbaImage};
        let img = RgbaImage::from_pixel(width, height, image::Rgba([100, 150, 200, 255]));
        let mut bytes = Vec::new();
        img.write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
            .expect("encode png");
        base64::engine::general_purpose::STANDARD.encode(&bytes)
    }

    fn user_with_image_block(tool_use_id: &str, base64_payload: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: tool_use_id.to_string(),
                content: vec![ToolResultContent::Image {
                    source: crate::provider::ImageSource {
                        source_type: "base64".to_string(),
                        media_type: "image/png".to_string(),
                        data: base64_payload.to_string(),
                    },
                }],
                is_error: false,
            }],
        }
    }

    #[test]
    fn test_downscale_no_op_when_all_within_cap() {
        let small = synthesize_png_base64(800, 600);
        let messages = vec![
            user_with_image_block("call_a", &small),
            assistant_text("ack"),
        ];
        assert!(matches!(
            downscale_oversized_images(&messages),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn test_downscale_resizes_oversized_input_image() {
        use base64::Engine;
        use image::ImageFormat;
        // A user message with a top-level input image (ACP @-mention / pasted screenshot) must be
        // downscaled the same as a tool-result image, or Anthropic rejects the multi-image request.
        let big = synthesize_png_base64(2400, 1200);
        let messages = vec![crate::provider::Message::user_with_images("look", vec![
            crate::provider::ImageSource {
                source_type: "base64".to_string(),
                media_type: "image/png".to_string(),
                data: big,
            },
        ])];
        let owned = match downscale_oversized_images(&messages) {
            Cow::Owned(v) => v,
            Cow::Borrowed(_) => panic!("expected owned (input-image resize triggered)"),
        };
        // content[0] = Text, content[1] = the downscaled input image.
        match &owned[0].content[1] {
            ContentBlock::Image { source } => {
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(&source.data)
                    .expect("decode");
                let decoded =
                    image::load_from_memory_with_format(&bytes, ImageFormat::Png).expect("png");
                assert!(decoded.width() <= MAX_IMAGE_DIMENSION_PX);
                assert!(decoded.height() <= MAX_IMAGE_DIMENSION_PX);
            }
            other => panic!("expected downscaled input image; got {:?}", other),
        }
    }

    #[test]
    fn test_downscale_resizes_oversized_image() {
        use base64::Engine;
        use image::ImageFormat;
        let big = synthesize_png_base64(2400, 1200);
        let small = synthesize_png_base64(800, 600);
        let messages = vec![
            user_with_image_block("call_big", &big),
            user_with_image_block("call_small", &small),
        ];
        let result = downscale_oversized_images(&messages);
        let owned = match result {
            Cow::Owned(v) => v,
            Cow::Borrowed(_) => panic!("expected owned (resize triggered)"),
        };
        // First image was downscaled to fit 2000 px on each axis.
        match &owned[0].content[0] {
            ContentBlock::ToolResult { content, .. } => match &content[0] {
                ToolResultContent::Image { source } => {
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(&source.data)
                        .expect("decode");
                    let decoded =
                        image::load_from_memory_with_format(&bytes, ImageFormat::Png).expect("png");
                    assert!(decoded.width() <= MAX_IMAGE_DIMENSION_PX);
                    assert!(decoded.height() <= MAX_IMAGE_DIMENSION_PX);
                    // 2:1 aspect ratio preserved.
                    assert_eq!(decoded.width() / decoded.height(), 2);
                }
                _ => panic!("expected resized image"),
            },
            _ => unreachable!(),
        }
        // Second image was within cap → unchanged.
        match &owned[1].content[0] {
            ContentBlock::ToolResult { content, .. } => match &content[0] {
                ToolResultContent::Image { source } => assert_eq!(source.data, small),
                _ => panic!("small image should be untouched"),
            },
            _ => unreachable!(),
        }
    }

    /// Locks in the contract that `build_body_within_budget` returns a user-visible
    /// [`crate::provider::Notice`] (rather than printing to stderr directly) when redaction kicks
    /// in. The agent loop then forwards it through `Frontend::emit`, which is how ACP clients see
    /// the redaction signal that used to silently bypass them.
    #[test]
    fn test_build_body_within_budget_returns_notice_on_redaction() {
        use std::cell::Cell;

        // Two messages, the first containing an oversized image and the second a small one. The
        // redactor only touches non-last messages, so the older image is the one that gets
        // dropped.
        let big_payload = "X".repeat(2 * 1024 * 1024);
        let messages = vec![
            user_with_block(image_block("call_a", &big_payload)),
            user_with_block(image_block("call_b", "BBB")),
            assistant_text("ack"),
        ];

        let call_count: Cell<usize> = Cell::new(0);
        let build = |_msgs: &[Message]| -> Result<String> {
            let n = call_count.get();
            call_count.set(n + 1);
            if n == 0 {
                // First serialization: oversize. Use a slim payload so the test stays cheap; the
                // function only cares about `.len() > MAX_REQUEST_BYTES`.
                Ok("X".repeat(MAX_REQUEST_BYTES + 1024))
            } else {
                Ok("{}".to_string())
            }
        };

        let (body, notice) =
            build_body_within_budget(&messages, None, build).expect("redaction should succeed");
        assert_eq!(body, "{}");
        let notice = notice.expect("redaction must surface a Notice");
        assert_eq!(notice.level, crate::provider::NoticeLevel::Info);
        assert!(
            notice.text.starts_with("Redacted "),
            "notice text should describe the redaction: {:?}",
            notice.text,
        );
        assert_eq!(call_count.get(), 2, "build closure called twice");
    }

    /// On the happy path (no redaction needed), the function returns `None` for the notice. Locks
    /// the contract: frontends never see a no-op advisory.
    #[test]
    fn test_build_body_within_budget_no_notice_when_within_budget() {
        let messages = vec![assistant_text("hi")];
        let build = |_msgs: &[Message]| -> Result<String> { Ok("{}".to_string()) };
        let (body, notice) = build_body_within_budget(&messages, None, build).expect("happy path");
        assert_eq!(body, "{}");
        assert!(notice.is_none());
    }
}
