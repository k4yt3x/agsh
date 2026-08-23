//! OpenAI Responses API encoder + SSE decoder.
//!
//! The protocol, not a backend: both [`super::responses`] (API key, any endpoint serving
//! `/v1/responses`) and [`super::subscription`] (ChatGPT's `/backend-api/codex/responses`) drive
//! this. It is the OpenAI counterpart of [`crate::provider::anthropic`]'s `shared`, and the reason
//! Chat Completions is a separate module rather than a flag: the two are different wire formats,
//! not two dialects of one.
//!
//! Nothing here may assume which endpoint is on the other end. `store: false` is set
//! unconditionally because meka replays the whole conversation every turn and never uses
//! server-side state, which also happens to be all the stateless implementations (Ollama, vLLM)
//! support. Anything that *is* endpoint-specific -- the encrypted-reasoning `include`, auth
//! headers, the URL -- belongs to the backend, not here.
//!
//! The on-the-wire request shape is documented at
//! <https://platform.openai.com/docs/guides/function-calling?api-mode=responses>. Verified against
//! the first-party Codex client:
//! - request shape: `temp/codex/codex-rs/codex-api/src/common.rs:163`
//! - input items:   `temp/codex/codex-rs/protocol/src/models.rs:686`
//! - SSE events:    `temp/codex/codex-rs/codex-api/src/sse/responses.rs:283`

use eventsource_stream::Eventsource;
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Abort the SSE read if no event arrives within this window. A Responses endpoint can
/// silently stall a stream; without a ceiling the turn would hang forever. Shared with the
/// Anthropic and `openai-chat-completions` drivers, and matches the first-party Codex client's
/// `stream_idle_timeout` default (`stream_idle_timeout_ms = 300000`). A timeout surfaces as a
/// [`MekaError::StreamError`], which the agent retries when no output has been forwarded yet.
use crate::provider::STREAM_IDLE_TIMEOUT as RESPONSES_STREAM_IDLE_TIMEOUT;
use crate::{
    error::{MekaError, Result},
    provider::{
        ContentBlock, Message, Notice, OpaqueReasoning, Role, StopReason, StreamEvent, TokenUsage,
        ToolDefinition, ToolResultContent,
    },
};

/// Build the JSON body POSTed to `/responses`. Translates the meka internal `Message` /
/// `ContentBlock` shape into Responses API `input` items (`message`, `function_call`,
/// `function_call_output`).
///
/// The result carries only what every Responses implementation understands. A backend that knows
/// more about its own endpoint adds to it afterwards -- see
/// [`include_encrypted_reasoning`], which only `chatgpt-subscription` applies.
pub(super) fn build_request_body(
    model: &str,
    system_prompt: &str,
    messages: &[Message],
    tools: &[ToolDefinition],
    reasoning_effort: Option<&str>,
    max_output_tokens: Option<u64>,
    stream: bool,
) -> serde_json::Value {
    let mut input = Vec::with_capacity(messages.len());

    for message in messages {
        match message.role {
            Role::User => encode_user_message(message, &mut input),
            Role::Assistant => encode_assistant_message(message, &mut input),
        }
    }

    let mut body = serde_json::json!({
        "model": model,
        "input": input,
        "tool_choice": "auto",
        "parallel_tool_calls": false,
        "store": false,
        "stream": stream,
    });

    if !system_prompt.is_empty() {
        body["instructions"] = serde_json::Value::String(system_prompt.to_string());
    }

    if !tools.is_empty() {
        body["tools"] = serde_json::Value::Array(encode_tools(tools));
    }

    if let Some(effort) = reasoning_effort {
        body["reasoning"] = serde_json::json!({"effort": effort});
    }

    if let Some(max_output) = max_output_tokens {
        body["max_output_tokens"] = serde_json::json!(max_output);
    }

    body
}

/// Fold a [`StreamEvent`] stream into the tuple [`crate::provider::Provider::complete`] returns.
/// Mirrors the accumulation in `Agent::run_streaming_attempt` but without any frontend emission, so
/// a streaming-only provider can satisfy the non-streaming completion contract by silently
/// consuming its own SSE. Text deltas concatenate into a trailing `Text` block; tool-call and
/// thinking events fold into their blocks; usage tiers merge; notices collect; `MessageEnd` sets
/// the stop reason. `StreamEvent::Error` is logged, not returned: the typed error surfaces from the
/// concurrently-awaited `stream` future in each backend's `complete`.
pub(super) async fn aggregate_stream(
    mut receiver: mpsc::Receiver<StreamEvent>,
) -> (Message, StopReason, TokenUsage, Vec<Notice>) {
    let mut content_blocks: Vec<ContentBlock> = Vec::new();
    let mut current_text = String::new();
    let mut current_thinking = String::new();
    let mut current_tool_id = String::new();
    let mut current_tool_name = String::new();
    let mut stop_reason = StopReason::EndTurn;
    let mut token_usage = TokenUsage::default();
    let mut notices: Vec<Notice> = Vec::new();

    while let Some(event) = receiver.recv().await {
        match event {
            StreamEvent::TextDelta(text) => current_text.push_str(&text),
            StreamEvent::ThinkingDelta(text) => current_thinking.push_str(&text),
            StreamEvent::ThinkingComplete { opaque } => {
                let thinking = std::mem::take(&mut current_thinking);
                if !thinking.is_empty() || opaque.is_some() {
                    content_blocks.push(ContentBlock::Thinking { thinking, opaque });
                }
            }
            StreamEvent::RedactedThinking { data } => {
                content_blocks.push(ContentBlock::RedactedThinking { data });
            }
            // A display-only liveness signal with nothing to accumulate. Only the Claude providers
            // emit it today; the arm exists so adding it there cannot silently change what this
            // provider persists.
            StreamEvent::ThinkingProgress { .. } => {}
            StreamEvent::ToolUseStart { id, name } => {
                if !current_text.is_empty() {
                    content_blocks.push(ContentBlock::Text {
                        text: std::mem::take(&mut current_text),
                    });
                }
                current_tool_id = id;
                current_tool_name = name;
            }
            // The full arguments object arrives whole in `ToolUseEnd`; the incremental JSON only
            // feeds the live renderer, which this silent path has none of.
            StreamEvent::ToolInputDelta(_) => {}
            StreamEvent::ToolUseEnd { input } => {
                content_blocks.push(ContentBlock::ToolUse {
                    id: std::mem::take(&mut current_tool_id),
                    name: std::mem::take(&mut current_tool_name),
                    input,
                });
            }
            StreamEvent::ToolCallRejected { id, name, reason } => {
                let input = serde_json::json!({
                    crate::provider::INVALID_TOOL_ARGS_MARKER: reason,
                });
                content_blocks.push(ContentBlock::ToolUse { id, name, input });
            }
            StreamEvent::MessageEnd {
                stop_reason: reason,
            } => stop_reason = reason,
            StreamEvent::Usage(usage) => token_usage.merge_stream(&usage),
            StreamEvent::Notice(notice) => notices.push(notice),
            StreamEvent::Error(error) => {
                tracing::error!("responses: stream error: {}", error);
            }
        }
    }

    if !current_text.is_empty() {
        content_blocks.push(ContentBlock::Text { text: current_text });
    }

    (
        Message {
            role: Role::Assistant,
            content: content_blocks,
        },
        stop_reason,
        token_usage,
        notices,
    )
}

/// Ask the server to round-trip its reasoning as `reasoning.encrypted_content`.
///
/// A no-op unless the request already asks for reasoning, since there would be nothing to encrypt.
/// The one caller settles that with [`request_reasoning_summary`] first, so the guard is really a
/// refusal to be used out of order rather than a live branch; the ordering is asserted by
/// `an_unconfigured_profile_still_asks_for_a_summary_and_encrypted_reasoning`.
///
/// This is *not* in [`build_request_body`] because it is a fact about one endpoint rather than
/// about the protocol. `include` is an OpenAI extension; `chatgpt-subscription` sends it because
/// its endpoint is always ChatGPT and the first-party Codex client sends it, while
/// `openai-responses` reaches Ollama, vLLM, LM Studio and OpenRouter, where meka has no basis for
/// assuming it is understood and a rejected request is the cost of guessing wrong.
pub(super) fn include_encrypted_reasoning(body: &mut serde_json::Value) {
    if body.get("reasoning").is_some() {
        body["include"] = serde_json::json!(["reasoning.encrypted_content"]);
    }
}

/// Drop every replayed `reasoning` item from a built body.
///
/// The encoder is protocol-level and replays sealed reasoning for any backend, which is right for
/// the protocol and wrong for one endpoint: `openai-responses` never asks for encrypted reasoning,
/// so the only way its history holds any is a session recorded under `chatgpt-subscription` and
/// resumed against a `base_url` -- Ollama, vLLM, LM Studio, OpenRouter. Replaying it there ships
/// ChatGPT's sealed blob and `rs_...` id to a third party that cannot decrypt it and may reject the
/// item shape outright.
///
/// The same refusal the Claude encoder makes by matching only [`OpaqueReasoning::Signed`], one
/// level up: an endpoint drops what it could not have produced.
pub(super) fn drop_replayed_reasoning(body: &mut serde_json::Value) {
    if let Some(input) = body.get_mut("input").and_then(|input| input.as_array_mut()) {
        input.retain(|item| item.get("type").and_then(|kind| kind.as_str()) != Some("reasoning"));
    }
}

/// Ask the server for a reasoning summary, the only part of its reasoning it will show a person.
///
/// Without this the model still reasons, but `summary` comes back empty and there is nothing for
/// the REPL to render, so a long think looks like a hang. Codex asks for `auto` by default
/// (`ReasoningSummary::Auto`, and its `supports_reasoning_summary_parameter` defaults to true), so
/// this is what the first-party client's own users see.
///
/// Establishes the `reasoning` object when the profile configured no effort, which is deliberate:
/// it is also what makes [`include_encrypted_reasoning`] apply to a default profile. Codex likewise
/// sends `reasoning` on every request and omits only the fields it has no value for.
///
/// Endpoint-specific for the same reason as [`include_encrypted_reasoning`]: `summary` is an
/// OpenAI parameter, and only `chatgpt-subscription` knows it is talking to OpenAI.
///
/// Takes the body [`build_request_body`] returns, where `reasoning` is either an object or absent;
/// indexing promotes an absent key through a null to an object, which is how the no-effort case
/// gets its `reasoning` block.
pub(super) fn request_reasoning_summary(body: &mut serde_json::Value) {
    body["reasoning"]["summary"] = serde_json::json!("auto");
}

/// Build the `output` field of a `function_call_output` item from a slice of `ToolResultContent`.
/// The Responses API accepts either a plain string OR an array of `input_text` / `input_image` /
/// `input_file` content items (per OpenAI's docs: "For functions that return images or files, you
/// can pass an array of image or file objects instead of a string."). We emit the array form when
/// at least one image is present to preserve image data; otherwise we collapse to a string for the
/// simpler wire shape.
///
/// Sent unconditionally. Non-vision models will return a clear API error rather than us trying to
/// detect model capabilities client-side. Mirrors our Claude path, which also sends images without
/// a model gate.
fn build_tool_result_output(content: &[ToolResultContent]) -> serde_json::Value {
    let has_image = content
        .iter()
        .any(|block| matches!(block, ToolResultContent::Image { .. }));

    if !has_image {
        return serde_json::Value::String(ContentBlock::tool_result_text_content(content));
    }

    let parts: Vec<serde_json::Value> = content
        .iter()
        .map(|block| match block {
            ToolResultContent::Text { text } => serde_json::json!({
                "type": "input_text",
                "text": text,
            }),
            ToolResultContent::Image { source } => input_image_part(source),
        })
        .collect();
    serde_json::Value::Array(parts)
}

/// Build a Responses API `input_image` content part from an image source. Shared by the tool-result
/// and user-message encoders.
fn input_image_part(source: &crate::provider::ImageSource) -> serde_json::Value {
    serde_json::json!({
        "type": "input_image",
        "image_url": super::data_url(source),
        "detail": "auto",
    })
}

fn encode_user_message(message: &Message, input: &mut Vec<serde_json::Value>) {
    let mut text_parts: Vec<&str> = Vec::new();
    let mut image_parts: Vec<serde_json::Value> = Vec::new();

    for block in &message.content {
        match block {
            ContentBlock::Text { text } => text_parts.push(text),
            // Responses takes `input_image` content parts on the user message. No model gate;
            // non-vision models return a clear error.
            ContentBlock::Image { source } => image_parts.push(input_image_part(source)),
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                input.push(serde_json::json!({
                    "type": "function_call_output",
                    "call_id": tool_use_id,
                    "output": build_tool_result_output(content),
                }));
            }
            // ToolUse / Thinking on a user message would be malformed; ignore defensively to match
            // the Chat Completions encoder's behaviour.
            _ => {}
        }
    }

    let mut content_parts: Vec<serde_json::Value> = Vec::new();
    if !text_parts.is_empty() {
        content_parts.push(serde_json::json!({
            "type": "input_text",
            "text": text_parts.join("\n"),
        }));
    }
    content_parts.extend(image_parts);
    if !content_parts.is_empty() {
        input.push(serde_json::json!({
            "type": "message",
            "role": "user",
            "content": content_parts,
        }));
    }
}

/// Encode one assistant turn as `input` items, in the order its blocks arrived.
///
/// Order is load-bearing here in a way it is not for the user encoder. A `reasoning` item must be
/// followed by the output that reasoning produced, or the API rejects the whole request: `Item
/// '<id>' of type 'reasoning' was provided without its required following item`. Emitting all the
/// text first and then all the calls, as this did before reasoning was replayed at all, breaks that
/// pairing as soon as a turn thinks twice.
fn encode_assistant_message(message: &Message, input: &mut Vec<serde_json::Value>) {
    let mut pending_text = String::new();

    for (index, block) in message.content.iter().enumerate() {
        match block {
            ContentBlock::Text { text } => pending_text.push_str(text),
            // Replayed only when it carries `encrypted_content`, which is the reasoning itself and
            // the only part the model can resume from. A summary without it is a digest written
            // for a human, so sending one back buys nothing and would put an item shape on the wire
            // for endpoints that were never asked for reasoning in the first place.
            ContentBlock::Thinking {
                thinking,
                opaque:
                    Some(OpaqueReasoning::Sealed {
                        encrypted_content,
                        id,
                    }),
            } if an_emitted_item_follows(&message.content, index) => {
                flush_assistant_text(&mut pending_text, input);
                input.push(reasoning_input_item(
                    id.as_deref(),
                    thinking,
                    encrypted_content,
                ));
            }
            ContentBlock::ToolUse {
                id,
                name,
                input: arguments,
            } => {
                flush_assistant_text(&mut pending_text, input);
                input.push(serde_json::json!({
                    "type": "function_call",
                    "name": name,
                    "call_id": id,
                    "arguments": arguments.to_string(),
                }));
            }
            _ => {}
        }
    }

    flush_assistant_text(&mut pending_text, input);
}

/// Emit the assistant text accumulated so far as one `message` item, and clear it.
fn flush_assistant_text(pending: &mut String, input: &mut Vec<serde_json::Value>) {
    if pending.is_empty() {
        return;
    }
    input.push(serde_json::json!({
        "type": "message",
        "role": "assistant",
        "content": [{
            "type": "output_text",
            "text": std::mem::take(pending),
        }],
    }));
}

/// Whether anything this encoder will actually emit follows `index` in the same turn.
///
/// A reasoning item left dangling is a rejected request, and meka's own history reaches that shape
/// without any help from the model: [`Message::without_tool_use`] drops an interrupted turn's calls
/// but keeps its thinking, and a rewind can cut anywhere.
///
/// Only output counts, so a run of reasoning items is carried by whatever follows the run rather
/// than by each other -- which is the order the server produced them in, and so the order they may
/// be replayed in. What this refuses is a turn that ends in reasoning with no output at all.
fn an_emitted_item_follows(content: &[ContentBlock], index: usize) -> bool {
    content.iter().skip(index + 1).any(|block| match block {
        ContentBlock::Text { text } => !text.is_empty(),
        ContentBlock::ToolUse { .. } => true,
        _ => false,
    })
}

/// A `reasoning` item, shaped as the server sent it so it can be replayed verbatim.
///
/// meka flattens a multi-part summary into one string as it streams (the parts are display
/// sections, separated by a blank line), and does not try to reconstruct the original parts here:
/// splitting back on the separator would invent boundaries wherever a single part contained a
/// paragraph break. The whole text goes back as one part, which preserves what was said.
fn reasoning_input_item(
    id: Option<&str>,
    summary: &str,
    encrypted_content: &str,
) -> serde_json::Value {
    let summary = if summary.is_empty() {
        serde_json::json!([])
    } else {
        serde_json::json!([{ "type": "summary_text", "text": summary }])
    };
    let mut item = serde_json::json!({
        "type": "reasoning",
        "summary": summary,
        "encrypted_content": encrypted_content,
    });
    if let Some(id) = id {
        item["id"] = serde_json::Value::String(id.to_string());
    }
    item
}

fn encode_tools(tools: &[ToolDefinition]) -> Vec<serde_json::Value> {
    tools
        .iter()
        .map(|tool| {
            serde_json::json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "strict": false,
                "parameters": tool.parameters,
            })
        })
        .collect()
}

/// Mutable state threaded through SSE event processing. Tracks the in-flight tool call's
/// accumulated arguments so we can return a parsed `ToolUseEnd` even if the server elides the final
/// `arguments` field.
#[derive(Default)]
pub(super) struct SseState {
    active_tool_call: Option<ActiveToolCall>,
    in_reasoning: bool,
    /// Whether this response emitted any `function_call` output item. The Responses API reports
    /// `status: "completed"` even for tool-call turns, so the stop reason has to be inferred from
    /// the presence of function calls rather than the status (mirrors the Chat Completions
    /// `finish_reason: "tool_calls"` mapping).
    emitted_tool_call: bool,
    /// Once `response.completed` (or `response.failed` / `response.incomplete`) has been
    /// processed, the driver should stop pulling new events.
    pub(super) finished: bool,
}

struct ActiveToolCall {
    arguments_buffer: String,
}

/// Pure SSE-event handler. Inspects the named event + parsed JSON payload, updates `state`, and
/// returns the meka-level [`StreamEvent`]s to forward to the agent. Returns `Err` when the server
/// reports a fatal stream error; the driver propagates this back to the caller.
/// Whether a `response.failed` event's error `code`/`type` indicates a transient, retryable
/// condition. Conservative on purpose (matches the Claude driver's equivalent): only the codes
/// OpenAI documents as transient server-side conditions are retryable; anything else (including
/// unrecognized codes) is treated as permanent so a real problem surfaces immediately instead of
/// being masked by retries.
fn is_retryable_responses_error_code(code: &str) -> bool {
    matches!(code, "server_error" | "rate_limit_exceeded" | "overloaded")
}

/// Which Responses frame this is: the payload's `type`, falling back to the SSE `event:` line.
///
/// The payload field is the spec's discriminator and is always present; the `event:` line is an
/// optional convenience only some servers set. Reading the event name alone worked while ChatGPT
/// and OpenAI -- which send both -- were the only endpoints meka reached, and broke the moment an
/// API-key backend could point anywhere: OpenRouter streams bare `data:` frames, so every frame
/// looked unhandled and a perfectly good turn died with "stream ended before a terminal response
/// event". The fallback keeps working for any server that names the event but omits `type`.
fn frame_name<'a>(data: &'a serde_json::Value, event_name: &'a str) -> &'a str {
    data.get("type")
        .and_then(|value| value.as_str())
        .unwrap_or(event_name)
}

pub(super) fn process_event(
    event_name: &str,
    data: &serde_json::Value,
    state: &mut SseState,
) -> Result<Vec<StreamEvent>> {
    let mut out = Vec::new();
    match event_name {
        "response.created" | "response.in_progress" => {}

        "response.output_text.delta" => {
            if let Some(delta) = data.get("delta").and_then(|v| v.as_str())
                && !delta.is_empty()
            {
                out.push(StreamEvent::TextDelta(delta.to_string()));
            }
        }

        "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" => {
            if let Some(delta) = data.get("delta").and_then(|v| v.as_str())
                && !delta.is_empty()
            {
                state.in_reasoning = true;
                out.push(StreamEvent::ThinkingDelta(delta.to_string()));
            }
        }

        // A reasoning summary arrives as several parts, each its own section with its own heading,
        // and the deltas carry no separator of their own. Codex renders a break here
        // (`on_reasoning_section_break`); meka's thinking is flat text, so the break is a blank
        // line. The first part opens the block rather than separating anything, which is what
        // `in_reasoning` distinguishes.
        "response.reasoning_summary_part.added" => {
            if state.in_reasoning {
                out.push(StreamEvent::ThinkingDelta("\n\n".to_string()));
            }
        }

        "response.output_item.added" => {
            let Some(item) = data.get("item") else {
                return Ok(out);
            };
            if item.get("type").and_then(|v| v.as_str()) == Some("function_call") {
                let id = item
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let name = item
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                state.active_tool_call = Some(ActiveToolCall {
                    arguments_buffer: String::new(),
                });
                state.emitted_tool_call = true;
                out.push(StreamEvent::ToolUseStart { id, name });
            }
        }

        "response.function_call_arguments.delta" | "response.custom_tool_call_input.delta" => {
            if let Some(delta) = data.get("delta").and_then(|v| v.as_str())
                && !delta.is_empty()
            {
                if let Some(tool) = state.active_tool_call.as_mut() {
                    tool.arguments_buffer.push_str(delta);
                }
                out.push(StreamEvent::ToolInputDelta(delta.to_string()));
            }
        }

        "response.output_item.done" => {
            let Some(item) = data.get("item") else {
                return Ok(out);
            };
            let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if item_type == "function_call" {
                let buffered = state.active_tool_call.take();
                // Read off the completed item: `ActiveToolCall` carries only the argument buffer,
                // and the item is the authoritative copy of the identity anyway.
                let call_id = item
                    .get("call_id")
                    .or_else(|| item.get("id"))
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_string();
                let call_name = item
                    .get("name")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_string();
                // Prefer the final `arguments` string from the item over our accumulated buffer;
                // the server may normalise it.
                let arguments_str = item
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| buffered.map(|tool| tool.arguments_buffer))
                    .unwrap_or_default();
                // Empty arguments are a legitimate zero-parameter call; arguments that arrived and
                // do not parse are rejected rather than replaced with `{}`. Substituting an empty
                // object runs the tool on whatever defaults it tolerates -- a valid call the model
                // never made -- with nothing told to anyone. Matches the Chat Completions path,
                // which has a regression test named for this exact bug.
                let parsed = if arguments_str.is_empty() {
                    Ok(serde_json::json!({}))
                } else {
                    serde_json::from_str(&arguments_str)
                };
                match parsed {
                    Ok(input) => out.push(StreamEvent::ToolUseEnd { input }),
                    Err(error) => {
                        tracing::warn!(
                            tool = %call_name,
                            "rejecting tool call with unparseable JSON arguments: {}",
                            error
                        );
                        out.push(StreamEvent::ToolCallRejected {
                            id: call_id,
                            name: call_name,
                            reason: format!("invalid JSON arguments: {}", error),
                        });
                    }
                }
            } else if item_type == "reasoning" {
                // True when this item streamed anything readable, by either spelling: a requested
                // summary, or the raw reasoning text a local server emits unprompted.
                let showed_its_reasoning = state.in_reasoning;
                state.in_reasoning = false;
                let encrypted_content = item
                    .get("encrypted_content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                // Deliberately not gated on that. `encrypted_content` is what the next request
                // replays, and it arrives whether or not summaries were requested -- so gating its
                // capture on visible text silently threw the reasoning chain away for exactly the
                // configuration that asked for it. An item with neither is worth nothing to either
                // the reader or the next turn, so it makes no block.
                if showed_its_reasoning || encrypted_content.is_some() {
                    let id = item
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    out.push(StreamEvent::ThinkingComplete {
                        opaque: encrypted_content.map(|encrypted_content| {
                            OpaqueReasoning::Sealed {
                                encrypted_content,
                                id,
                            }
                        }),
                    });
                }
            }
        }

        "response.completed" => {
            state.finished = true;
            if let Some(response) = data.get("response") {
                if let Some(usage) = response.get("usage") {
                    out.push(StreamEvent::Usage(TokenUsage {
                        input_tokens: usage
                            .get("input_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0),
                        output_tokens: usage
                            .get("output_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0),
                        ..TokenUsage::default()
                    }));
                }
                // The Responses API reports `status: "completed"` even when the output is function
                // calls, so a tool-call turn must be surfaced as `ToolUse` regardless of status;
                // otherwise the agent warns and mislabels the turn as `EndTurn`.
                let stop_reason = if state.emitted_tool_call {
                    StopReason::ToolUse
                } else {
                    response
                        .get("status")
                        .and_then(|v| v.as_str())
                        .map(parse_response_status)
                        .unwrap_or(StopReason::EndTurn)
                };
                out.push(StreamEvent::MessageEnd { stop_reason });
            } else if state.emitted_tool_call {
                out.push(StreamEvent::MessageEnd {
                    stop_reason: StopReason::ToolUse,
                });
            } else {
                out.push(StreamEvent::MessageEnd {
                    stop_reason: StopReason::EndTurn,
                });
            }
        }

        "response.failed" => {
            state.finished = true;
            let error_object = data
                .get("response")
                .and_then(|response| response.get("error"));
            let message = error_object
                .and_then(|error| error.get("message"))
                .and_then(|v| v.as_str())
                .unwrap_or("response.failed event")
                .to_string();
            // OpenAI's error objects carry `code` (occasionally `type`); either indicates a
            // transient server-side condition worth retrying. Sending `StreamEvent::Error` is
            // handled by the caller (`drive_responses_sse_stream`), which has channel access.
            let error_code = error_object
                .and_then(|error| error.get("code").or_else(|| error.get("type")))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            return Err(if is_retryable_responses_error_code(error_code) {
                MekaError::RetryableProvider {
                    message,
                    retry_after: None,
                }
            } else {
                MekaError::Provider(message)
            });
        }

        "response.incomplete" => {
            state.finished = true;
            // `incomplete_details.reason` (e.g. "max_output_tokens", "content_filter") is a
            // deterministic outcome, not a transient failure — never retryable.
            let reason = data
                .get("response")
                .and_then(|response| response.get("incomplete_details"))
                .and_then(|details| details.get("reason"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let message = format!("response.incomplete: {}", reason);
            return Err(MekaError::Provider(message));
        }

        other => {
            tracing::debug!("unhandled Responses SSE event: {}", other);
        }
    }
    Ok(out)
}

fn parse_response_status(status: &str) -> StopReason {
    match status {
        "completed" => StopReason::EndTurn,
        "incomplete" => StopReason::MaxTokens,
        other => {
            tracing::warn!(
                "the Responses endpoint returned an unrecognized response status {other:?}; mapping to Unknown"
            );
            StopReason::Unknown(other.to_string())
        }
    }
}

/// Drive the SSE stream for a Responses API call. Pulls events off the transport, runs them through
/// [`process_event`], and forwards the resulting [`StreamEvent`]s to the agent.
pub(super) async fn drive_responses_sse_stream(
    response: reqwest::Response,
    event_sender: mpsc::Sender<StreamEvent>,
    cancellation: CancellationToken,
) -> Result<()> {
    let status = response.status();
    if !status.is_success() {
        let retry_after = crate::error::parse_retry_after(response.headers());
        let response_text = response.text().await.unwrap_or_else(|error| {
            tracing::warn!("failed to read the Responses error body: {}", error);
            String::new()
        });
        return Err(crate::error::provider_http_error(
            status,
            &response_text,
            retry_after,
        ));
    }

    let mut event_stream = response.bytes_stream().eventsource();
    let mut state = SseState::default();

    loop {
        tokio::select! {
            _ = cancellation.cancelled() => {
                return Err(MekaError::Interrupted);
            }
            event = tokio::time::timeout(RESPONSES_STREAM_IDLE_TIMEOUT, event_stream.next()) => {
                let event = match event {
                    Ok(event) => event,
                    Err(_elapsed) => {
                        // No event for the idle window: treat a stalled stream as a transport
                        // error so the agent can retry rather than hang forever.
                        let message = format!(
                            "idle timeout waiting for a Responses SSE event after {}s",
                            RESPONSES_STREAM_IDLE_TIMEOUT.as_secs()
                        );
                        if event_sender
                            .send(StreamEvent::Error(message.clone()))
                            .await
                            .is_err()
                        {
                            tracing::trace!("stream event receiver dropped");
                        }
                        return Err(MekaError::StreamError(message));
                    }
                };
                let Some(event) = event else { break };
                let event = match event {
                    Ok(event) => event,
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
                };

                let data: serde_json::Value = match serde_json::from_str(&event.data) {
                    Ok(data) => data,
                    Err(error) => {
                        tracing::warn!("failed to parse Responses SSE data: {}", error);
                        continue;
                    }
                };

                let outcomes = process_event(frame_name(&data, &event.event), &data, &mut state);
                let events = match outcomes {
                    Ok(events) => events,
                    Err(error) => {
                        // `process_event` doesn't have channel access, so forward the error here
                        // (mirrors the Claude driver's pattern) rather than relying on the caller
                        // to notice — best-effort: a dropped receiver just means no one's
                        // listening anymore, not a reason to fail differently.
                        if event_sender
                            .send(StreamEvent::Error(error.to_string()))
                            .await
                            .is_err()
                        {
                            tracing::trace!("stream event receiver dropped");
                        }
                        return Err(error);
                    }
                };

                for emit in events {
                    if event_sender.send(emit).await.is_err() {
                        tracing::trace!("stream event receiver dropped");
                        return Ok(());
                    }
                }

                if state.finished {
                    return Ok(());
                }
            }
        }
    }

    // The stream ended without `response.completed`, `response.failed` or `response.incomplete`.
    // Falling through here committed a truncated turn as a complete one: the agent saw whatever
    // text had arrived, wrote it to the conversation, and moved on, with the retry path never
    // consulted. A connection cut mid-response is exactly what that path exists for.
    let message = "the Responses stream ended before a terminal response event".to_string();
    if event_sender
        .send(StreamEvent::Error(message.clone()))
        .await
        .is_err()
    {
        tracing::trace!("stream event receiver dropped");
    }
    Err(MekaError::StreamError(message))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ToolResultContent;

    #[test]
    fn test_request_body_minimal() {
        let body = build_request_body("gpt-5", "", &[Message::user("hi")], &[], None, None, true);
        assert_eq!(body["model"], "gpt-5");
        assert_eq!(body["stream"], true);
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["parallel_tool_calls"], false);
        assert_eq!(body["store"], false);
        assert!(body.get("instructions").is_none());
        assert!(body.get("tools").is_none());
        assert!(body.get("reasoning").is_none());
    }

    #[test]
    fn test_request_body_includes_instructions_when_system_prompt_set() {
        let body = build_request_body(
            "gpt-5",
            "be helpful",
            &[Message::user("hi")],
            &[],
            None,
            None,
            true,
        );
        assert_eq!(body["instructions"], "be helpful");
    }

    #[test]
    fn test_request_body_user_message_uses_input_text() {
        let body = build_request_body(
            "gpt-5",
            "",
            &[Message::user("hello")],
            &[],
            None,
            None,
            true,
        );
        let input = body["input"].as_array().expect("input array");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[0]["content"][0]["text"], "hello");
    }

    #[test]
    fn test_request_body_assistant_text_uses_output_text() {
        let messages = vec![
            Message::user("a"),
            Message::assistant_text("b"),
            Message::user("c"),
        ];
        let body = build_request_body("gpt-5", "", &messages, &[], None, None, true);
        let input = body["input"].as_array().expect("input array");
        assert_eq!(input.len(), 3);
        assert_eq!(input[1]["role"], "assistant");
        assert_eq!(input[1]["content"][0]["type"], "output_text");
        assert_eq!(input[1]["content"][0]["text"], "b");
    }

    #[test]
    fn test_request_body_tool_use_emits_function_call_item() {
        let messages = vec![
            Message::user("read /tmp/x"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_abc".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/x"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_abc".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "contents".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ];

        let body = build_request_body("gpt-5", "", &messages, &[], None, None, true);
        let input = body["input"].as_array().expect("input array");

        // [0] user message, [1] function_call, [2] function_call_output
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["name"], "read_file");
        assert_eq!(input[1]["call_id"], "call_abc");
        // arguments must be a JSON string, not a parsed object
        assert!(input[1]["arguments"].is_string());

        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[2]["call_id"], "call_abc");
        assert_eq!(input[2]["output"], "contents");
    }

    #[test]
    fn test_request_body_tools_use_responses_api_flat_shape() {
        let tools = vec![ToolDefinition::new(
            "demo",
            "A demo tool",
            serde_json::json!({"type": "object", "properties": {}}),
        )];
        let body = build_request_body("gpt-5", "", &[], &tools, None, None, true);
        let tools_arr = body["tools"].as_array().expect("tools");
        assert_eq!(tools_arr[0]["type"], "function");
        // Top-level `name` / `description` / `parameters` (NOT wrapped under a `function` object
        // like Chat Completions). This is the Responses API shape.
        assert_eq!(tools_arr[0]["name"], "demo");
        assert_eq!(tools_arr[0]["description"], "A demo tool");
        assert!(tools_arr[0].get("parameters").is_some());
        assert!(tools_arr[0].get("function").is_none());
    }

    /// The shared body carries only what every Responses implementation understands.
    ///
    /// `include` is an OpenAI extension, so it is not the protocol's to add: it was moved out of
    /// here when `openai-responses` arrived, because that backend reaches servers where an
    /// unrecognised field is a rejected request. The subscription backend opts in explicitly, and
    /// its half of this split is asserted alongside.
    #[test]
    fn the_shared_body_asks_for_reasoning_but_never_for_an_openai_extension() {
        let mut body = build_request_body(
            "gpt-5",
            "",
            &[Message::user("think hard")],
            &[],
            Some("high"),
            None,
            true,
        );
        assert_eq!(body["reasoning"]["effort"], "high");
        assert!(body.get("include").is_none(), "{body}");

        // The subscription backend adds it, so reasoning survives a stateless round trip against
        // ChatGPT, whose support for it is a fact rather than a guess.
        include_encrypted_reasoning(&mut body);
        let include = body["include"].as_array().expect("include");
        assert!(
            include
                .iter()
                .any(|value| value == "reasoning.encrypted_content")
        );
    }

    /// Nothing to encrypt means nothing to ask for.
    #[test]
    fn the_encrypted_reasoning_include_is_a_no_op_without_reasoning() {
        let mut body =
            build_request_body("gpt-5", "", &[Message::user("hi")], &[], None, None, true);
        include_encrypted_reasoning(&mut body);
        assert!(body.get("include").is_none(), "{body}");
    }

    #[test]
    fn test_request_body_user_image_emits_input_image() {
        let message = Message::user_with_images("describe", vec![crate::provider::ImageSource {
            source_type: "base64".to_string(),
            media_type: "image/png".to_string(),
            data: "QUJD".to_string(),
        }]);
        let body = build_request_body("gpt-5", "", &[message], &[], None, None, true);
        let input = body["input"].as_array().expect("input array");
        let content = input[0]["content"].as_array().expect("content array");
        assert_eq!(content[0]["type"], "input_text");
        assert_eq!(content[0]["text"], "describe");
        assert_eq!(content[1]["type"], "input_image");
        assert_eq!(content[1]["image_url"], "data:image/png;base64,QUJD");
    }

    #[test]
    fn test_request_body_sets_max_output_tokens_when_overridden() {
        let body = build_request_body(
            "gpt-5",
            "",
            &[Message::user("hi")],
            &[],
            None,
            Some(40_000),
            true,
        );
        assert_eq!(body["max_output_tokens"], 40_000);
    }

    #[test]
    fn test_request_body_omits_max_output_tokens_when_unset() {
        let body = build_request_body("gpt-5", "", &[Message::user("hi")], &[], None, None, true);
        assert!(body.get("max_output_tokens").is_none());
    }

    #[test]
    fn test_request_body_omits_reasoning_when_effort_unset() {
        let body = build_request_body("gpt-5", "", &[Message::user("hi")], &[], None, None, true);
        assert!(body.get("reasoning").is_none());
        assert!(body.get("include").is_none());
    }

    #[test]
    fn test_request_body_user_message_with_tool_result_only_no_text_block() {
        // A user turn that's *only* a tool_result must produce only a function_call_output input
        // item, no empty user message.
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: vec![ToolResultContent::Text {
                    text: "result".to_string(),
                }],
                is_error: false,
            }],
        }];
        let body = build_request_body("gpt-5", "", &messages, &[], None, None, true);
        let input = body["input"].as_array().expect("input array");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "function_call_output");
    }

    fn run_events(
        events: &[(&str, serde_json::Value)],
    ) -> (Vec<StreamEvent>, std::result::Result<(), MekaError>) {
        let mut state = SseState::default();
        let mut emitted = Vec::new();
        let mut outcome = Ok(());
        for (name, data) in events {
            match process_event(name, data, &mut state) {
                Ok(events) => emitted.extend(events),
                Err(error) => {
                    // process_event still yields a StreamEvent::Error before
                    // returning. Draining it via re-running with a fresh state
                    // would be wrong; instead we capture both outcomes by
                    // running once and then preserving the events that were
                    // emitted before the error. For the error path the caller
                    // populates `out` *and* returns Err, so the events the
                    // caller would forward are already in `out` for the call
                    // that errored. Re-process with a side-channel:
                    if let Some(message) = error.to_string().strip_prefix("Provider error: ") {
                        emitted.push(StreamEvent::Error(message.to_string()));
                    }
                    outcome = Err(error);
                    break;
                }
            }
            if state.finished {
                break;
            }
        }
        (emitted, outcome)
    }

    /// A frame is named by its payload, not by the SSE `event:` line.
    ///
    /// Found against OpenRouter, which sends bare `data:` frames. That is valid SSE and the
    /// Responses spec's own discriminator is the payload's `type`; the `event:` line is an optional
    /// convenience. Keying off the event name alone worked only while ChatGPT and OpenAI -- which
    /// send both -- were the only endpoints meka reached, and failed the moment an API-key backend
    /// could point anywhere. The symptom was total: every frame looked unhandled, so a turn that
    /// streamed perfectly well died with "stream ended before a terminal response event".
    ///
    /// Both orderings are asserted, because a fix that read only the payload would break the
    /// endpoints that name the event and send a payload without one.
    #[test]
    fn a_frame_is_identified_by_its_payload_type_not_the_sse_event_name() {
        // How OpenRouter streams: bare `data:`, no event name, type in the payload.
        let bare = serde_json::json!({"type": "response.output_text.delta", "delta": "hi"});
        assert_eq!(frame_name(&bare, ""), "response.output_text.delta");

        // How ChatGPT streams: the event named and the payload agreeing.
        assert_eq!(
            frame_name(&bare, "response.output_text.delta"),
            "response.output_text.delta"
        );

        // A payload with no `type` still falls back to the event name, so an endpoint that names
        // the event and omits the field keeps working.
        let untyped = serde_json::json!({"delta": "hi"});
        assert_eq!(
            frame_name(&untyped, "response.output_text.delta"),
            "response.output_text.delta"
        );

        // And the name that comes out actually drives the decoder.
        let (emitted, _) =
            run_events(&[(frame_name(&bare, ""), serde_json::json!({"delta": "hi"}))]);
        assert!(
            matches!(emitted.as_slice(), [StreamEvent::TextDelta(text)] if text == "hi"),
            "{emitted:?}"
        );
    }

    #[test]
    fn test_process_event_text_delta() {
        let mut state = SseState::default();
        let events = process_event(
            "response.output_text.delta",
            &serde_json::json!({"delta": "hello"}),
            &mut state,
        )
        .expect("ok");
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::TextDelta(ref t) if t == "hello"));
    }

    #[test]
    fn test_process_event_text_delta_empty_emits_nothing() {
        let mut state = SseState::default();
        let events = process_event(
            "response.output_text.delta",
            &serde_json::json!({"delta": ""}),
            &mut state,
        )
        .expect("ok");
        assert!(events.is_empty());
    }

    #[test]
    fn test_process_event_reasoning_delta_emits_thinking() {
        let mut state = SseState::default();
        let events = process_event(
            "response.reasoning_text.delta",
            &serde_json::json!({"delta": "hmm"}),
            &mut state,
        )
        .expect("ok");
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::ThinkingDelta(ref t) if t == "hmm"));
        assert!(state.in_reasoning);
    }

    #[test]
    fn test_process_event_tool_call_full_lifecycle() {
        let (events, outcome) = run_events(&[
            (
                "response.output_item.added",
                serde_json::json!({
                    "item": {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "read_file"
                    }
                }),
            ),
            (
                "response.function_call_arguments.delta",
                serde_json::json!({"delta": "{\"path\":"}),
            ),
            (
                "response.function_call_arguments.delta",
                serde_json::json!({"delta": "\"/tmp/x\"}"}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({
                    "item": {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "read_file",
                        "arguments": "{\"path\":\"/tmp/x\"}"
                    }
                }),
            ),
            (
                "response.completed",
                serde_json::json!({
                    "response": {"id": "r1", "status": "completed"}
                }),
            ),
        ]);
        outcome.expect("clean stream");
        assert!(matches!(
            events[0],
            StreamEvent::ToolUseStart { ref id, ref name } if id == "c1" && name == "read_file"
        ));
        assert!(matches!(events[1], StreamEvent::ToolInputDelta(_)));
        assert!(matches!(events[2], StreamEvent::ToolInputDelta(_)));
        match &events[3] {
            StreamEvent::ToolUseEnd { input } => assert_eq!(input["path"], "/tmp/x"),
            other => panic!("expected ToolUseEnd, got {:?}", other),
        }
        // The Responses API reports `status: "completed"` even for tool-call turns; the presence of
        // the function call must still surface as `ToolUse`, not `EndTurn`.
        assert!(matches!(events[4], StreamEvent::MessageEnd {
            stop_reason: StopReason::ToolUse
        }));
    }

    /// `run_events` drives `process_event` directly, so it can never reach the fall-through at the
    /// bottom of `drive_responses_sse_stream` -- the code that decides what a stream ending with
    /// no terminal event means. Drive the real loop instead, the way the Claude decoder's
    /// counterpart test does.
    async fn decode_sse(body: &str) -> (Vec<StreamEvent>, Result<()>) {
        let response: reqwest::Response = axum::http::Response::builder()
            .status(200)
            .header("content-type", "text/event-stream")
            .body(body.to_string())
            .expect("build response")
            .into();
        let (sender, mut receiver) = mpsc::channel(64);
        let outcome = drive_responses_sse_stream(response, sender, CancellationToken::new()).await;
        let mut events = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            events.push(event);
        }
        (events, outcome)
    }

    /// A stream of bare `data:` frames decodes, end to end, through the real driver.
    ///
    /// This is how OpenRouter streams: valid SSE with no `event:` line, the frame named only by the
    /// payload's `type`. It has to be asserted here rather than against [`frame_name`] alone,
    /// because the bug was the *call site* -- the helper can be perfect while the driver still
    /// passes it the wrong string. Every other case in this harness carries an `event:` line and a
    /// payload without a `type`, so they exercise only the fallback branch and all stayed green
    /// while every OpenRouter turn died with "stream ended before a terminal response event".
    #[tokio::test]
    async fn a_stream_of_bare_data_frames_decodes_end_to_end() {
        let (events, outcome) = decode_sse(concat!(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"PO\"}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"NG\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\
             \"usage\":{\"input_tokens\":1,\"output_tokens\":2}}}\n\n",
        ))
        .await;
        assert!(outcome.is_ok(), "{outcome:?}");
        let text: String = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::TextDelta(delta) => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "PONG", "{events:?}");
        assert!(
            events
                .iter()
                .any(|event| matches!(event, StreamEvent::MessageEnd { .. })),
            "the terminal frame must be recognised too: {events:?}"
        );
    }

    /// A connection cut partway through the answer never sends `response.completed`. Committing
    /// that as a finished turn hands the agent half a response and never consults the retry path,
    /// which is the one case that path exists for.
    #[tokio::test]
    async fn a_stream_cut_before_its_terminal_event_is_an_error() {
        let (events, outcome) = decode_sse(concat!(
            "event: response.output_text.delta\n",
            "data: {\"delta\":\"half an ans\"}\n\n",
        ))
        .await;

        assert!(
            matches!(outcome, Err(MekaError::StreamError(_))),
            "a truncated stream must reach the retry path, got {outcome:?}",
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, StreamEvent::Error(_))),
            "and say so on the stream: {events:?}",
        );
    }

    /// The other half of the same boundary: a stream that did reach `response.completed` is
    /// finished, and must not be turned into a spurious retry by the check above.
    #[tokio::test]
    async fn a_stream_that_reaches_its_terminal_event_is_complete() {
        let (_events, outcome) = decode_sse(concat!(
            "event: response.output_text.delta\n",
            "data: {\"delta\":\"a whole answer\"}\n\n",
            "event: response.completed\n",
            "data: {\"response\": {\"id\": \"r1\", \"status\": \"completed\"}}\n\n",
        ))
        .await;

        outcome.expect("a stream with a terminal event is complete");
    }

    /// The Responses half of the same boundary the Anthropic decoder has: arguments that do not
    /// parse are the model's intent, mangled, and running the tool with `{}` executes something
    /// it never asked for while reporting success.
    #[test]
    fn a_tool_call_with_unparseable_arguments_is_rejected_not_run_empty() {
        let (events, outcome) = run_events(&[
            (
                "response.output_item.added",
                serde_json::json!({
                    "item": {"type": "function_call", "call_id": "c1", "name": "write_file"}
                }),
            ),
            (
                "response.function_call_arguments.delta",
                serde_json::json!({"delta": "{\"path\": "}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({
                    "item": {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "write_file",
                        "arguments": "{\"path\": "
                    }
                }),
            ),
        ]);
        outcome.expect("a rejected tool call is not a stream failure");

        assert!(
            events.iter().any(|event| matches!(
                event,
                StreamEvent::ToolCallRejected { name, .. } if name == "write_file"
            )),
            "the call must be rejected: {events:?}",
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, StreamEvent::ToolUseEnd { .. })),
            "and must not also be dispatched: {events:?}",
        );
    }

    #[test]
    fn test_process_event_completed_without_tool_call_is_end_turn() {
        let (events, outcome) = run_events(&[
            (
                "response.output_text.delta",
                serde_json::json!({"delta": "final answer"}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"id": "r1", "status": "completed"}}),
            ),
        ]);
        outcome.expect("clean stream");
        assert!(matches!(
            events.last(),
            Some(StreamEvent::MessageEnd {
                stop_reason: StopReason::EndTurn
            })
        ));
    }

    #[test]
    fn test_process_event_tool_call_recovers_arguments_from_done_only() {
        // Server elides per-delta events and sends arguments only on `done`.
        let (events, outcome) = run_events(&[
            (
                "response.output_item.added",
                serde_json::json!({
                    "item": {"type": "function_call", "call_id": "c1", "name": "x"}
                }),
            ),
            (
                "response.output_item.done",
                serde_json::json!({
                    "item": {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "x",
                        "arguments": "{\"k\":1}"
                    }
                }),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"id": "r1", "status": "completed"}}),
            ),
        ]);
        outcome.expect("clean stream");
        let input = events
            .iter()
            .find_map(|event| match event {
                StreamEvent::ToolUseEnd { input } => Some(input),
                _ => None,
            })
            .expect("ToolUseEnd present");
        assert_eq!(input["k"], 1);
    }

    #[test]
    fn test_process_event_completed_emits_token_usage() {
        let mut state = SseState::default();
        let events = process_event(
            "response.completed",
            &serde_json::json!({
                "response": {
                    "id": "r1",
                    "status": "completed",
                    "usage": {"input_tokens": 42, "output_tokens": 7}
                }
            }),
            &mut state,
        )
        .expect("ok");
        let usage = events
            .iter()
            .find_map(|event| match event {
                StreamEvent::Usage(usage) => Some(usage),
                _ => None,
            })
            .expect("Usage event");
        assert_eq!(usage.input_tokens, 42);
        assert_eq!(usage.output_tokens, 7);
        assert!(state.finished);
    }

    #[test]
    fn test_process_event_failed_yields_error_and_propagates() {
        let mut state = SseState::default();
        let result = process_event(
            "response.failed",
            &serde_json::json!({
                "response": {"error": {"message": "too many tokens"}}
            }),
            &mut state,
        );
        assert!(state.finished);
        assert!(
            matches!(result, Err(MekaError::Provider(ref message)) if message.contains("too many tokens"))
        );
    }

    #[test]
    fn test_process_event_failed_with_server_error_code_is_retryable() {
        let mut state = SseState::default();
        let result = process_event(
            "response.failed",
            &serde_json::json!({
                "response": {"error": {"code": "server_error", "message": "internal error"}}
            }),
            &mut state,
        );
        assert!(matches!(result, Err(MekaError::RetryableProvider { .. })));
    }

    #[test]
    fn test_process_event_failed_without_code_stays_permanent() {
        // No `code`/`type` field at all — default is not-retryable, matching today's behavior.
        let mut state = SseState::default();
        let result = process_event(
            "response.failed",
            &serde_json::json!({
                "response": {"error": {"message": "bad request"}}
            }),
            &mut state,
        );
        assert!(matches!(result, Err(MekaError::Provider(_))));
    }

    #[test]
    fn test_is_retryable_responses_error_code() {
        for retryable in ["server_error", "rate_limit_exceeded", "overloaded"] {
            assert!(is_retryable_responses_error_code(retryable));
        }
        for permanent in ["invalid_request_error", "unknown", ""] {
            assert!(!is_retryable_responses_error_code(permanent));
        }
    }

    #[test]
    fn test_process_event_incomplete_yields_error() {
        let mut state = SseState::default();
        let result = process_event(
            "response.incomplete",
            &serde_json::json!({
                "response": {"incomplete_details": {"reason": "max_output_tokens"}}
            }),
            &mut state,
        );
        assert!(state.finished);
        assert!(matches!(
            result,
            Err(MekaError::Provider(ref message)) if message.contains("max_output_tokens")
        ));
    }

    #[test]
    fn test_process_event_status_incomplete_maps_to_max_tokens() {
        let mut state = SseState::default();
        let events = process_event(
            "response.completed",
            &serde_json::json!({
                "response": {"id": "r1", "status": "incomplete"}
            }),
            &mut state,
        )
        .expect("ok");
        assert!(
            events
                .iter()
                .any(|event| matches!(event, StreamEvent::MessageEnd {
                    stop_reason: StopReason::MaxTokens
                }))
        );
    }

    #[test]
    fn test_process_event_unknown_event_silently_skipped() {
        let mut state = SseState::default();
        let events = process_event(
            "response.output_audio_transcript.delta",
            &serde_json::json!({"delta": "audio"}),
            &mut state,
        )
        .expect("ok");
        assert!(events.is_empty());
        assert!(!state.finished);
    }

    #[test]
    fn test_process_event_reasoning_done_emits_thinking_complete_with_signature() {
        let mut state = SseState {
            in_reasoning: true,
            ..SseState::default()
        };
        let events = process_event(
            "response.output_item.done",
            &serde_json::json!({
                "item": {
                    "type": "reasoning",
                    "id": "rs_123",
                    "summary": [],
                    "encrypted_content": "OPAQUE"
                }
            }),
            &mut state,
        )
        .expect("ok");
        assert_eq!(events.len(), 1);
        match &events[0] {
            StreamEvent::ThinkingComplete {
                opaque:
                    Some(OpaqueReasoning::Sealed {
                        encrypted_content,
                        id,
                    }),
            } => {
                assert_eq!(encrypted_content, "OPAQUE");
                assert_eq!(id.as_deref(), Some("rs_123"));
            }
            other => panic!("expected ThinkingComplete, got {:?}", other),
        }
        assert!(!state.in_reasoning);
    }

    /// The defect this whole path had: `encrypted_content` arrives whether or not a summary was
    /// asked for, and gating its capture on having seen a summary delta threw the reasoning chain
    /// away for every request that did not also ask for summaries.
    #[test]
    fn encrypted_reasoning_is_captured_without_a_summary_to_show_for_it() {
        let mut state = SseState::default();
        let events = process_event(
            "response.output_item.done",
            &serde_json::json!({
                "item": {
                    "type": "reasoning",
                    "id": "rs_silent",
                    "summary": [],
                    "encrypted_content": "OPAQUE"
                }
            }),
            &mut state,
        )
        .expect("ok");
        assert!(
            matches!(
                events.as_slice(),
                [StreamEvent::ThinkingComplete {
                    opaque: Some(OpaqueReasoning::Sealed {
                        encrypted_content,
                        id,
                    }),
                }] if id.as_deref() == Some("rs_silent") && encrypted_content == "OPAQUE"
            ),
            "silent reasoning must still be captured, got {:?}",
            events
        );
    }

    /// A reasoning item with neither summary nor encrypted content has nothing to render and
    /// nothing to replay, so it should not manufacture a block.
    #[test]
    fn an_empty_reasoning_item_produces_no_event() {
        let mut state = SseState::default();
        let events = process_event(
            "response.output_item.done",
            &serde_json::json!({ "item": { "type": "reasoning", "summary": [] } }),
            &mut state,
        )
        .expect("ok");
        assert!(events.is_empty(), "got {:?}", events);
    }

    /// Each summary part is its own section. Without a break between them the parts run together
    /// into one paragraph, and the first part must not be preceded by one.
    #[test]
    fn summary_parts_are_separated_but_the_first_one_is_not() {
        let mut state = SseState::default();
        let opening = process_event(
            "response.reasoning_summary_part.added",
            &serde_json::json!({ "summary_index": 0 }),
            &mut state,
        )
        .expect("ok");
        assert!(opening.is_empty(), "got {:?}", opening);

        process_event(
            "response.reasoning_summary_text.delta",
            &serde_json::json!({ "delta": "**First**\nthinking" }),
            &mut state,
        )
        .expect("ok");

        let between = process_event(
            "response.reasoning_summary_part.added",
            &serde_json::json!({ "summary_index": 1 }),
            &mut state,
        )
        .expect("ok");
        assert!(
            matches!(between.as_slice(), [StreamEvent::ThinkingDelta(text)] if text == "\n\n"),
            "got {:?}",
            between
        );
    }

    fn image_content(media_type: &str, data: &str) -> ToolResultContent {
        ToolResultContent::Image {
            source: crate::provider::ImageSource {
                source_type: "base64".to_string(),
                media_type: media_type.to_string(),
                data: data.to_string(),
            },
        }
    }

    #[test]
    fn test_build_tool_result_output_text_only_returns_string() {
        let content = vec![ToolResultContent::Text {
            text: "result".to_string(),
        }];
        let out = build_tool_result_output(&content);
        assert_eq!(out, serde_json::Value::String("result".to_string()));
    }

    #[test]
    fn test_build_tool_result_output_with_image_returns_array() {
        let content = vec![
            ToolResultContent::Text {
                text: "before".to_string(),
            },
            image_content("image/png", "AAAA"),
            ToolResultContent::Text {
                text: "after".to_string(),
            },
        ];
        let out = build_tool_result_output(&content);
        let array = out.as_array().expect("should be array when image present");
        assert_eq!(array.len(), 3);
        assert_eq!(array[0]["type"], "input_text");
        assert_eq!(array[0]["text"], "before");
        assert_eq!(array[1]["type"], "input_image");
        assert_eq!(array[1]["image_url"], "data:image/png;base64,AAAA");
        assert_eq!(array[1]["detail"], "auto");
        assert_eq!(array[2]["type"], "input_text");
        assert_eq!(array[2]["text"], "after");
    }

    #[test]
    fn test_build_tool_result_output_image_only_returns_array() {
        let content = vec![image_content("image/jpeg", "DEAD")];
        let out = build_tool_result_output(&content);
        let array = out.as_array().expect("should be array");
        assert_eq!(array.len(), 1);
        assert_eq!(array[0]["type"], "input_image");
        assert_eq!(array[0]["image_url"], "data:image/jpeg;base64,DEAD");
    }

    #[test]
    fn test_function_call_output_carries_image_array_in_request_body() {
        // End-to-end: build_request_body wires build_tool_result_output via encode_user_message;
        // confirm the function_call_output's `output` field is the array form when an image is
        // present.
        let mut messages = vec![Message::user("look at this"), Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "screenshot".to_string(),
                input: serde_json::json!({}),
            }],
        }];
        messages.push(Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: vec![image_content("image/png", "QkFTRTY0")],
                is_error: false,
            }],
        });
        let body = build_request_body("gpt-5", "", &messages, &[], None, None, true);
        let input = body["input"].as_array().expect("input array");
        let output_item = input
            .iter()
            .find(|item| item["type"] == "function_call_output")
            .expect("function_call_output present");
        let output = output_item["output"]
            .as_array()
            .expect("output should be array (image present)");
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["type"], "input_image");
        assert_eq!(output[0]["image_url"], "data:image/png;base64,QkFTRTY0");
    }

    fn thinking(summary: &str, encrypted: Option<&str>, id: Option<&str>) -> ContentBlock {
        ContentBlock::Thinking {
            thinking: summary.to_string(),
            opaque: encrypted.map(|encrypted_content| OpaqueReasoning::Sealed {
                encrypted_content: encrypted_content.to_string(),
                id: id.map(str::to_string),
            }),
        }
    }

    fn tool_call(id: &str) -> ContentBlock {
        ContentBlock::ToolUse {
            id: id.to_string(),
            name: "read_file".to_string(),
            input: serde_json::json!({}),
        }
    }

    fn assistant(content: Vec<ContentBlock>) -> Message {
        Message {
            role: Role::Assistant,
            content,
        }
    }

    fn input_of(messages: &[Message]) -> Vec<serde_json::Value> {
        build_request_body("gpt-5", "", messages, &[], Some("high"), None, true)["input"]
            .as_array()
            .expect("input array")
            .clone()
    }

    /// The whole point of the `include`: reasoning meka asked the server to encrypt has to go back
    /// out on the next request, or the model restarts its chain of thought at every tool call.
    #[test]
    fn encrypted_reasoning_is_replayed_on_the_next_request() {
        let input = input_of(&[
            Message::user("hi"),
            assistant(vec![
                thinking("weighing it up", Some("OPAQUE"), Some("rs_1")),
                tool_call("call_1"),
            ]),
        ]);

        let reasoning = input
            .iter()
            .find(|item| item["type"] == "reasoning")
            .expect("the reasoning item must survive the round trip");
        assert_eq!(reasoning["id"], "rs_1");
        assert_eq!(reasoning["encrypted_content"], "OPAQUE");
        assert_eq!(reasoning["summary"][0]["type"], "summary_text");
        assert_eq!(reasoning["summary"][0]["text"], "weighing it up");
    }

    /// The API pairs a reasoning item with the output it produced and rejects one left dangling
    /// ("was provided without its required following item"). `without_tool_use` reaches that shape
    /// on any interrupted turn, so the encoder has to refuse it rather than the server.
    #[test]
    fn reasoning_with_nothing_after_it_is_not_sent() {
        let interrupted = assistant(vec![
            thinking("weighing it up", Some("OPAQUE"), Some("rs_1")),
            tool_call("call_1"),
        ])
        .without_tool_use();
        let input = input_of(&[Message::user("hi"), interrupted]);

        assert!(
            !input.iter().any(|item| item["type"] == "reasoning"),
            "a dangling reasoning item would be rejected: {:?}",
            input
        );
    }

    /// What the API wants after a reasoning item is the output that reasoning produced. Another
    /// reasoning item is not that, so a turn cut down to nothing but thinking sends none of it --
    /// otherwise the last one still dangles and takes the whole request down with it.
    #[test]
    fn reasoning_does_not_count_as_another_reasoning_item_s_follower() {
        let input = input_of(&[
            Message::user("hi"),
            assistant(vec![
                thinking("first", Some("ONE"), Some("rs_1")),
                thinking("second", Some("TWO"), Some("rs_2")),
            ]),
        ]);

        assert!(
            !input.iter().any(|item| item["type"] == "reasoning"),
            "got {:?}",
            input
        );
    }

    /// Reasoning has to precede the output it produced, which the old encoder could not express:
    /// it emitted every text block first and every call after, so a turn that thought twice came
    /// back in an order the API rejects.
    #[test]
    fn reasoning_precedes_the_output_it_produced_across_two_thinking_rounds() {
        let input = input_of(&[
            Message::user("hi"),
            assistant(vec![
                ContentBlock::Text {
                    text: "let me look".to_string(),
                },
                thinking("first", Some("ONE"), Some("rs_1")),
                tool_call("call_1"),
                thinking("second", Some("TWO"), Some("rs_2")),
                ContentBlock::Text {
                    text: "done".to_string(),
                },
            ]),
        ]);

        let shape: Vec<&str> = input
            .iter()
            .filter_map(|item| item["type"].as_str())
            .collect();
        assert_eq!(shape, [
            // The user turn, then text the model wrote before it thought, then each round of
            // reasoning immediately ahead of what that reasoning produced.
            "message",
            "message",
            "reasoning",
            "function_call",
            "reasoning",
            "message"
        ]);
        assert_eq!(input[2]["encrypted_content"], "ONE");
        assert_eq!(input[4]["encrypted_content"], "TWO");
    }

    /// A session is not bound to the provider that recorded it: `meka -c -p other` resumes one
    /// under anything. A Claude thinking block reaching this encoder must not be replayed, because
    /// its signature is a MAC from another cryptosystem and its `thinking` is the reasoning itself
    /// -- sending them as `encrypted_content` and a `summary` hands OpenAI a blob it cannot
    /// decrypt and leaks the whole Claude reasoning beside it. This was live before the two shapes
    /// were named apart.
    #[test]
    fn a_claude_signature_is_never_replayed_to_openai() {
        let claude_turn = assistant(vec![
            ContentBlock::Thinking {
                thinking: "the full Claude reasoning".to_string(),
                opaque: Some(OpaqueReasoning::Signed {
                    signature: "CLAUDE_MAC".to_string(),
                }),
            },
            ContentBlock::Text {
                text: "answer".to_string(),
            },
        ]);
        let serialized = serde_json::to_string(&serde_json::Value::Array(input_of(&[
            Message::user("hi"),
            claude_turn,
        ])))
        .expect("serialize");

        assert!(!serialized.contains("CLAUDE_MAC"), "{serialized}");
        assert!(!serialized.contains("\"reasoning\""), "{serialized}");
    }

    /// A summary is a digest written for a person; without `encrypted_content` there is no
    /// reasoning to resume, so replaying it would only put an unexpected item shape on the wire of
    /// an endpoint that was never asked for reasoning.
    #[test]
    fn a_summary_without_encrypted_content_is_not_replayed() {
        let input = input_of(&[
            Message::user("hi"),
            assistant(vec![thinking("visible only", None, None), tool_call("c1")]),
        ]);

        assert!(
            !input.iter().any(|item| item["type"] == "reasoning"),
            "got {:?}",
            input
        );
    }

    /// A reasoning item that never carried an id is still replayable; the id is optional upstream
    /// too (`Option<ResponseItemId>`, skipped when absent).
    #[test]
    fn a_reasoning_item_without_an_id_is_still_replayed() {
        let input = input_of(&[
            Message::user("hi"),
            assistant(vec![thinking("", Some("OPAQUE"), None), tool_call("c1")]),
        ]);

        let reasoning = input
            .iter()
            .find(|item| item["type"] == "reasoning")
            .expect("present");
        assert!(reasoning.get("id").is_none(), "got {:?}", reasoning);
        assert_eq!(reasoning["summary"], serde_json::json!([]));
    }

    /// Without this the model still reasons and the user sees nothing, which reads as a hang.
    /// Codex asks for `auto`; so does meka, on the one backend whose endpoint is always OpenAI.
    #[test]
    fn the_two_asks_compose_into_a_reasoning_block_even_with_no_effort_set() {
        let mut body = build_request_body("gpt-5", "", &[], &[], None, None, true);
        assert!(body.get("reasoning").is_none(), "no effort, no reasoning");

        request_reasoning_summary(&mut body);
        assert_eq!(body["reasoning"]["summary"], "auto");

        // And settling `reasoning` is what lets the `include` apply to a default profile, which is
        // the configuration that previously asked for no encrypted reasoning at all.
        include_encrypted_reasoning(&mut body);
        assert_eq!(
            body["include"],
            serde_json::json!(["reasoning.encrypted_content"])
        );
    }

    /// The two halves composed: a real ChatGPT reasoning turn off the wire, folded into a message,
    /// and handed straight back to the request builder.
    ///
    /// Each half is pinned on its own above, but the thing that actually has to hold is that what
    /// the parser produces is what the encoder can replay. A field renamed on one side and not the
    /// other passes every unit test here and still loses the reasoning chain in production.
    #[tokio::test]
    async fn reasoning_survives_the_wire_and_goes_straight_back_out() {
        let (events, outcome) = decode_sse(concat!(
            "data: {\"type\":\"response.reasoning_summary_part.added\",\"summary_index\":0}\n\n",
            "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"**Plan**\"}\n\n",
            "data: {\"type\":\"response.reasoning_summary_part.added\",\"summary_index\":1}\n\n",
            "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"**Act**\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\
             \"id\":\"rs_live\",\"summary\":[],\"encrypted_content\":\"OPAQUE\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"here you go\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\
             \"usage\":{\"input_tokens\":1,\"output_tokens\":2}}}\n\n",
        ))
        .await;
        assert!(outcome.is_ok(), "{outcome:?}");

        let (sender, receiver) = tokio::sync::mpsc::channel(64);
        for event in events {
            sender.send(event).await.expect("buffered");
        }
        drop(sender);
        let (message, _stop, _usage, _notices) = aggregate_stream(receiver).await;

        let input = input_of(&[Message::user("hi"), message]);
        let reasoning = input
            .iter()
            .find(|item| item["type"] == "reasoning")
            .expect("the turn's reasoning must reach the next request");
        assert_eq!(reasoning["id"], "rs_live");
        assert_eq!(reasoning["encrypted_content"], "OPAQUE");
        assert_eq!(reasoning["summary"][0]["text"], "**Plan**\n\n**Act**");
        // And ahead of the answer it produced, which is what the API pairs it with.
        let shape: Vec<&str> = input
            .iter()
            .filter_map(|item| item["type"].as_str())
            .collect();
        assert_eq!(shape, ["message", "reasoning", "message"]);
    }

    /// The summary request must stay out of the shared body for the same reason the `include`
    /// does: `openai-responses` reaches endpoints that never agreed to an OpenAI parameter.
    #[test]
    fn the_shared_body_never_asks_for_a_reasoning_summary() {
        let body = build_request_body("gpt-5", "", &[], &[], Some("high"), None, true);
        assert_eq!(body["reasoning"]["effort"], "high");
        assert!(
            body["reasoning"].get("summary").is_none(),
            "got {:?}",
            body["reasoning"]
        );
    }
}
