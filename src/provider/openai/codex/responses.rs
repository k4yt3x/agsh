//! OpenAI Responses API encoder + SSE decoder.
//!
//! Codex's subscription endpoint (`chatgpt.com/backend-api/codex/responses`) speaks the Responses
//! API, not Chat Completions. The on-the-wire request shape is documented at
//! <https://platform.openai.com/docs/guides/function-calling?api-mode=responses>.
//!
//! Reference Codex source:
//! - request shape: `temp/codex/codex-rs/codex-api/src/common.rs:163`
//! - input items:   `temp/codex/codex-rs/protocol/src/models.rs:686`
//! - SSE events:    `temp/codex/codex-rs/codex-api/src/sse/responses.rs:283`

use std::time::Duration;

use eventsource_stream::Eventsource;
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    error::{MekaError, Result},
    provider::{
        ContentBlock, Message, Role, StopReason, StreamEvent, TokenUsage, ToolDefinition,
        ToolResultContent,
    },
};

/// Abort the SSE read if no event arrives within this window. The ChatGPT backend can silently
/// stall a stream; without a ceiling the turn would hang forever. Matches codex's own
/// `stream_idle_timeout` default (`stream_idle_timeout_ms = 300000`). A timeout surfaces as a
/// [`MekaError::StreamError`], which the agent retries when no output has been forwarded yet.
const CODEX_STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(300);

/// Build the JSON body POSTed to `/responses`. Translates the meka internal `Message` /
/// `ContentBlock` shape into Responses API `input` items (`message`, `function_call`,
/// `function_call_output`).
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
        body["include"] = serde_json::json!(["reasoning.encrypted_content"]);
    }

    if let Some(max_output) = max_output_tokens {
        body["max_output_tokens"] = serde_json::json!(max_output);
    }

    body
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
        "image_url": super::super::data_url(source),
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

fn encode_assistant_message(message: &Message, input: &mut Vec<serde_json::Value>) {
    let text = message.text_content();
    if !text.is_empty() {
        input.push(serde_json::json!({
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": text,
            }],
        }));
    }

    for block in &message.content {
        if let ContentBlock::ToolUse {
            id,
            name,
            input: arguments,
        } = block
        {
            input.push(serde_json::json!({
                "type": "function_call",
                "name": name,
                "call_id": id,
                "arguments": arguments.to_string(),
            }));
        }
    }
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
fn is_retryable_codex_error_code(code: &str) -> bool {
    matches!(code, "server_error" | "rate_limit_exceeded" | "overloaded")
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
                // Prefer the final `arguments` string from the item over our accumulated buffer;
                // the server may normalise it.
                let arguments_str = item
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| buffered.map(|tool| tool.arguments_buffer))
                    .unwrap_or_default();
                let input = if arguments_str.is_empty() {
                    serde_json::json!({})
                } else {
                    match serde_json::from_str(&arguments_str) {
                        Ok(value) => value,
                        Err(error) => {
                            tracing::warn!(
                                "failed to parse tool arguments JSON; running the tool with empty arguments: {}",
                                error
                            );
                            serde_json::json!({})
                        }
                    }
                };
                out.push(StreamEvent::ToolUseEnd { input });
            } else if item_type == "reasoning" && state.in_reasoning {
                state.in_reasoning = false;
                let signature = item
                    .get("encrypted_content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                out.push(StreamEvent::ThinkingComplete { signature });
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
            return Err(if is_retryable_codex_error_code(error_code) {
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
            tracing::debug!("unhandled Codex SSE event: {}", other);
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
                "Codex returned unrecognized response status {other:?}; mapping to Unknown"
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
            tracing::warn!("failed to read Codex error response body: {}", error);
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
            event = tokio::time::timeout(CODEX_STREAM_IDLE_TIMEOUT, event_stream.next()) => {
                let event = match event {
                    Ok(event) => event,
                    Err(_elapsed) => {
                        // No event for the idle window: treat a stalled stream as a transport
                        // error so the agent can retry rather than hang forever.
                        let message = format!(
                            "idle timeout waiting for Codex SSE event after {}s",
                            CODEX_STREAM_IDLE_TIMEOUT.as_secs()
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
                        tracing::warn!("failed to parse Codex SSE data: {}", error);
                        continue;
                    }
                };

                let outcomes = process_event(&event.event, &data, &mut state);
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
                    break;
                }
            }
        }
    }

    Ok(())
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

    #[test]
    fn test_request_body_reasoning_effort_attaches_include_field() {
        let body = build_request_body(
            "gpt-5",
            "",
            &[Message::user("think hard")],
            &[],
            Some("high"),
            None,
            true,
        );
        assert_eq!(body["reasoning"]["effort"], "high");
        // Codex always asks for encrypted reasoning content so the server round-trips reasoning
        // blocks across multi-turn conversations.
        let include = body["include"].as_array().expect("include");
        assert!(include.iter().any(|v| v == "reasoning.encrypted_content"));
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
    fn test_is_retryable_codex_error_code() {
        for retryable in ["server_error", "rate_limit_exceeded", "overloaded"] {
            assert!(is_retryable_codex_error_code(retryable));
        }
        for permanent in ["invalid_request_error", "unknown", ""] {
            assert!(!is_retryable_codex_error_code(permanent));
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
                    "summary": [],
                    "encrypted_content": "OPAQUE"
                }
            }),
            &mut state,
        )
        .expect("ok");
        assert_eq!(events.len(), 1);
        match &events[0] {
            StreamEvent::ThinkingComplete { signature } => {
                assert_eq!(signature.as_deref(), Some("OPAQUE"));
            }
            other => panic!("expected ThinkingComplete, got {:?}", other),
        }
        assert!(!state.in_reasoning);
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
}
