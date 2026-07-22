//! OpenAI-compatible provider. Targets the Chat Completions API and works with any compatible
//! endpoint (vLLM, Together, Groq, local proxies, etc.) by way of the `--base-url` flag and
//! `OPENAI_API_KEY`.

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    error::{MekaError, Result},
    provider::{
        ContentBlock, Message, Provider, Role, StopReason, StreamEvent, TokenUsage,
        ToolCallAccumulator, ToolDefinition, finalize_tool_call_accumulators,
    },
};

pub struct OpenAiProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    /// The settled `reasoning_effort` for the request body, resolved once at construction from the
    /// profile's override and the model's name predicates. The public OpenAI API exposes no models
    /// catalog, so there is nothing to refine from post-build.
    resolved_effort: Option<String>,
    max_output_tokens: Option<u64>,
}

impl OpenAiProvider {
    pub fn new(
        api_key: String,
        model: String,
        base_url: Option<String>,
        reasoning_effort: Option<String>,
        max_output_tokens: Option<u64>,
    ) -> Self {
        let resolved_effort = super::resolve_reasoning_effort(reasoning_effort.as_deref(), &model);
        Self {
            client: reqwest::Client::new(),
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
            model,
            resolved_effort,
            max_output_tokens,
        }
    }

    /// The settled reasoning-effort to send as `reasoning_effort` (see [`Self::resolved_effort`]).
    fn wire_effort(&self) -> Option<String> {
        self.resolved_effort.clone()
    }

    pub(super) fn build_request_body(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        stream: bool,
    ) -> serde_json::Value {
        let mut openai_messages = Vec::new();

        if !system_prompt.is_empty() {
            openai_messages.push(serde_json::json!({
                "role": "system",
                "content": system_prompt,
            }));
        }

        for message in messages {
            match message.role {
                Role::User => {
                    let has_tool_results = message
                        .content
                        .iter()
                        .any(|block| matches!(block, ContentBlock::ToolResult { .. }));

                    if has_tool_results {
                        for block in &message.content {
                            if let ContentBlock::ToolResult {
                                tool_use_id,
                                content,
                                is_error,
                            } = block
                            {
                                // Chat Completions deliberately restricts the `tool` role's content
                                // to text-only. The Chat reference defines
                                // `ChatCompletionToolMessageParam.content` as `string | array of
                                // ChatCompletionContentPartText` and notes "for tool messages, only
                                // type `text` is supported." Vision is on `user`-role messages
                                // only. So we collapse any image blocks to the literal "[Image]"
                                // via `tool_result_text_content` here. The Responses API (used by
                                // `openai-codex`) does accept `input_image` content blocks in
                                // `function_call_output.output`, and we emit those there.
                                let text = ContentBlock::tool_result_text_content(content);
                                let mut tool_msg = serde_json::json!({
                                    "role": "tool",
                                    "tool_call_id": tool_use_id,
                                    "content": text,
                                });
                                if *is_error {
                                    tool_msg["is_error"] = serde_json::json!(true);
                                }
                                openai_messages.push(tool_msg);
                            }
                        }
                    } else {
                        // No `match` on `ContentBlock` here means the compiler can't force this
                        // path to handle `Image`; it must be done by hand. When the user message
                        // carries images, Chat Completions wants a `content` array of `text` +
                        // `image_url` parts (vision is user-role only); otherwise a plain string.
                        let has_images = message
                            .content
                            .iter()
                            .any(|block| matches!(block, ContentBlock::Image { .. }));
                        if has_images {
                            let mut parts: Vec<serde_json::Value> = Vec::new();
                            let text = message.text_content();
                            if !text.is_empty() {
                                parts.push(serde_json::json!({"type": "text", "text": text}));
                            }
                            for block in &message.content {
                                if let ContentBlock::Image { source } = block {
                                    parts.push(serde_json::json!({
                                        "type": "image_url",
                                        "image_url": { "url": super::data_url(source) },
                                    }));
                                }
                            }
                            openai_messages.push(serde_json::json!({
                                "role": "user",
                                "content": parts,
                            }));
                        } else {
                            openai_messages.push(serde_json::json!({
                                "role": "user",
                                "content": message.text_content(),
                            }));
                        }
                    }
                }
                Role::Assistant => {
                    let tool_calls: Vec<_> = message
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Thinking { .. } => None,
                            ContentBlock::ToolUse { id, name, input } => Some(serde_json::json!({
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": input.to_string(),
                                }
                            })),
                            _ => None,
                        })
                        .collect();

                    if tool_calls.is_empty() {
                        openai_messages.push(serde_json::json!({
                            "role": "assistant",
                            "content": message.text_content(),
                        }));
                    } else {
                        let text = message.text_content();
                        let mut msg = serde_json::json!({
                            "role": "assistant",
                            "tool_calls": tool_calls,
                        });
                        if !text.is_empty() {
                            msg["content"] = serde_json::json!(text);
                        }
                        openai_messages.push(msg);
                    }
                }
            }
        }

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": openai_messages,
            "stream": stream,
        });

        // OpenAI omits `usage` from streamed responses unless explicitly asked; without this the
        // final usage-only chunk never arrives and token accounting (the `/status` context gauge,
        // auto-compact) silently reads zero for streaming turns.
        if stream {
            body["stream_options"] = serde_json::json!({ "include_usage": true });
        }

        let reasoning_effort = self.wire_effort();
        if let Some(effort) = &reasoning_effort {
            body["reasoning_effort"] = serde_json::json!(effort);
            // A reasoning model needs a generous completion cap or it truncates mid-thought. The
            // profile override wins; otherwise default to 32k.
            body["max_completion_tokens"] =
                serde_json::json!(self.max_output_tokens.unwrap_or(32_000));
        } else if let Some(max_output) = self.max_output_tokens {
            body["max_completion_tokens"] = serde_json::json!(max_output);
        }

        if !tools.is_empty() {
            let openai_tools: Vec<_> = tools
                .iter()
                .map(|tool| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(openai_tools);
        }

        body
    }

    pub(super) fn parse_non_streaming_response(
        &self,
        response: &serde_json::Value,
    ) -> Result<(Message, StopReason, TokenUsage)> {
        let choice = response
            .get("choices")
            .and_then(|choices| choices.get(0))
            .ok_or_else(|| MekaError::Provider("no choices in response".to_string()))?;

        let finish_reason = choice
            .get("finish_reason")
            .and_then(|reason| reason.as_str())
            .unwrap_or("stop");

        let stop_reason = parse_openai_stop_reason(finish_reason);

        let assistant_message = choice
            .get("message")
            .ok_or_else(|| MekaError::Provider("no 'message' in choice".to_string()))?;
        let mut content_blocks = Vec::new();

        if let Some(text) = assistant_message
            .get("content")
            .and_then(|content| content.as_str())
            && !text.is_empty()
        {
            content_blocks.push(ContentBlock::Text {
                text: text.to_string(),
            });
        }

        if let Some(tool_calls) = assistant_message
            .get("tool_calls")
            .and_then(|tool_calls| tool_calls.as_array())
        {
            for tool_call in tool_calls {
                let id = tool_call
                    .get("id")
                    .and_then(|id| id.as_str())
                    .ok_or_else(|| MekaError::Provider("tool call missing 'id' field".to_string()))?
                    .to_string();
                let name = tool_call
                    .get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(|name| name.as_str())
                    .or_else(|| tool_call.get("name").and_then(|name| name.as_str()))
                    .ok_or_else(|| {
                        MekaError::Provider("tool call missing 'function.name' field".to_string())
                    })?
                    .to_string();
                let arguments_str = tool_call
                    .get("function")
                    .and_then(|function| function.get("arguments"))
                    .and_then(|arguments| arguments.as_str())
                    .or_else(|| {
                        tool_call
                            .get("arguments")
                            .and_then(|arguments| arguments.as_str())
                    })
                    .unwrap_or("{}");
                let input: serde_json::Value = match serde_json::from_str(arguments_str) {
                    Ok(value) => value,
                    Err(error) => {
                        // Mirror the streaming path: surface the parse failure via the sentinel so
                        // the dispatch loop rejects the call instead of silently running the tool
                        // with empty arguments.
                        tracing::warn!(
                            "rejecting tool call with unparseable JSON arguments: {}",
                            error
                        );
                        serde_json::json!({
                            crate::provider::INVALID_TOOL_ARGS_MARKER:
                                format!("invalid JSON arguments: {}", error),
                        })
                    }
                };

                content_blocks.push(ContentBlock::ToolUse { id, name, input });
            }
        }

        let token_usage = TokenUsage {
            input_tokens: response
                .get("usage")
                .and_then(|u| u.get("prompt_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            output_tokens: response
                .get("usage")
                .and_then(|u| u.get("completion_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            ..TokenUsage::default()
        };

        Ok((
            Message {
                role: Role::Assistant,
                content: content_blocks,
            },
            stop_reason,
            token_usage,
        ))
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    async fn complete(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<(
        Message,
        StopReason,
        TokenUsage,
        Vec<crate::provider::Notice>,
    )> {
        let body = self.build_request_body(system_prompt, messages, tools, false);

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|error| {
                MekaError::Provider(format!(
                    "HTTP request failed: {}",
                    crate::error::format_reqwest_error(&error)
                ))
            })?;

        let status = response.status();
        let retry_after = crate::error::parse_retry_after(response.headers());
        let response_text = response
            .text()
            .await
            .map_err(|error| MekaError::Provider(format!("failed to read response: {}", error)))?;

        if !status.is_success() {
            return Err(crate::error::provider_http_error(
                status,
                &response_text,
                retry_after,
            ));
        }

        let response_json: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|error| MekaError::Provider(format!("invalid JSON response: {}", error)))?;

        let (message, stop_reason, usage) = self.parse_non_streaming_response(&response_json)?;
        Ok((message, stop_reason, usage, Vec::new()))
    }

    async fn stream(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        event_sender: mpsc::Sender<StreamEvent>,
        cancellation: CancellationToken,
    ) -> Result<()> {
        use eventsource_stream::Eventsource;
        use futures::StreamExt;

        let body = self.build_request_body(system_prompt, messages, tools, true);

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|error| {
                MekaError::Provider(format!(
                    "HTTP request failed: {}",
                    crate::error::format_reqwest_error(&error)
                ))
            })?;

        let status = response.status();
        if !status.is_success() {
            let retry_after = crate::error::parse_retry_after(response.headers());
            let response_text = response.text().await.unwrap_or_default();
            return Err(crate::error::provider_http_error(
                status,
                &response_text,
                retry_after,
            ));
        }

        let mut event_stream = response.bytes_stream().eventsource();

        let mut tool_call_accumulators: std::collections::HashMap<i64, ToolCallAccumulator> =
            std::collections::HashMap::new();

        // Set when a `finish_reason` chunk arrives. The loop keeps running afterward so the
        // trailing usage chunk (emitted by `stream_options.include_usage`) is captured;
        // finalisation and the single MessageEnd run once the stream ends, below the loop.
        let mut final_stop: Option<StopReason> = None;

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
                            if event.data == "[DONE]" {
                                break;
                            }

                            let data: serde_json::Value = match serde_json::from_str(&event.data) {
                                Ok(data) => data,
                                Err(error) => {
                                    tracing::warn!("failed to parse SSE data: {}", error);
                                    continue;
                                }
                            };

                            if let Some(usage) = data.get("usage") {
                                let token_usage = TokenUsage {
                                    input_tokens: usage
                                        .get("prompt_tokens")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0),
                                    output_tokens: usage
                                        .get("completion_tokens")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0),
                                    ..TokenUsage::default()
                                };
                                if event_sender
                                    .send(StreamEvent::Usage(token_usage))
                                    .await
                                    .is_err()
                                {
                                    tracing::trace!("stream event receiver dropped");
                                    break;
                                }
                            }

                            let Some(choice) = data.get("choices").and_then(|choices| choices.get(0)) else {
                                continue;
                            };

                            if let Some(finish_reason) = choice.get("finish_reason").and_then(|reason| reason.as_str()) {
                                // Record the stop reason but keep reading: with
                                // `stream_options.include_usage` the usage arrives in a trailing
                                // chunk AFTER this one (and before `[DONE]`). Finalisation and the
                                // single MessageEnd happen once the stream ends, below the loop.
                                final_stop = Some(parse_openai_stop_reason(finish_reason));
                                continue;
                            }

                            let Some(delta) = choice.get("delta") else {
                                continue;
                            };

                            if let Some(text) = delta.get("content").and_then(|content| content.as_str())
                                && !text.is_empty()
                                    && event_sender.send(StreamEvent::TextDelta(text.to_string())).await.is_err() {
                                        tracing::trace!("stream event receiver dropped");
                                        break;
                                    }

                            if let Some(tool_calls) = delta.get("tool_calls").and_then(|tool_calls| tool_calls.as_array()) {
                                for tool_call in tool_calls {
                                    let index = tool_call.get("index").and_then(|index| index.as_i64()).unwrap_or(0);

                                    let name = tool_call
                                        .get("function")
                                        .and_then(|function| function.get("name"))
                                        .and_then(|name| name.as_str())
                                        .or_else(|| tool_call.get("name").and_then(|name| name.as_str()));

                                    if let Some(id) = tool_call.get("id").and_then(|id| id.as_str()) {
                                        let accumulator = tool_call_accumulators
                                            .entry(index)
                                            .or_insert_with(|| ToolCallAccumulator {
                                                id: id.to_string(),
                                                name: String::new(),
                                                arguments: String::new(),
                                            });
                                        if let Some(name) = name
                                            && accumulator.name.is_empty()
                                        {
                                            accumulator.name = name.to_string();
                                        }
                                    } else if let Some(name) = name
                                        && let Some(accumulator) = tool_call_accumulators.get_mut(&index)
                                        && accumulator.name.is_empty()
                                    {
                                        accumulator.name = name.to_string();
                                    }

                                    if let Some(args) = tool_call
                                        .get("function")
                                        .and_then(|function| function.get("arguments"))
                                        .and_then(|arguments| arguments.as_str())
                                        .or_else(|| tool_call.get("arguments").and_then(|arguments| arguments.as_str()))
                                        && !args.is_empty() {
                                            if let Some(accumulator) = tool_call_accumulators.get_mut(&index) {
                                                accumulator.arguments.push_str(args);
                                            }
                                            if event_sender.send(StreamEvent::ToolInputDelta(args.to_string())).await.is_err() {
                                                tracing::trace!("stream event receiver dropped");
                                                break;
                                            }
                                        }
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

        // The stream ended (`[DONE]` or the connection closed). Finalise any pending tool calls and
        // emit the single MessageEnd, preferring the finish_reason we recorded; fall back to
        // tool-presence when no finish_reason arrived.
        let has_tools =
            finalize_tool_call_accumulators(&mut tool_call_accumulators, &event_sender).await;
        let stop_reason = final_stop.unwrap_or(if has_tools {
            StopReason::ToolUse
        } else {
            StopReason::EndTurn
        });
        if event_sender
            .send(StreamEvent::MessageEnd { stop_reason })
            .await
            .is_err()
        {
            tracing::trace!("stream event receiver dropped");
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "openai-api"
    }

    fn resolved_effort(&self) -> Option<String> {
        self.wire_effort()
    }
}

fn parse_openai_stop_reason(reason: &str) -> StopReason {
    match reason {
        "stop" => StopReason::EndTurn,
        "tool_calls" => StopReason::ToolUse,
        "length" => StopReason::MaxTokens,
        other => {
            tracing::warn!(
                "openai returned unrecognized finish_reason {other:?}; mapping to Unknown"
            );
            StopReason::Unknown(other.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ToolResultContent;

    #[test]
    fn test_openai_request_body_simple() {
        let provider = OpenAiProvider::new(
            "test-key".to_string(),
            "gpt-4o".to_string(),
            None,
            None,
            None,
        );

        let messages = vec![Message::user("hello")];
        let body = provider.build_request_body("system prompt", &messages, &[], false);

        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["stream"], false);

        let openai_messages = body["messages"]
            .as_array()
            .expect("messages should be array");
        assert_eq!(openai_messages.len(), 2);
        assert_eq!(openai_messages[0]["role"], "system");
        assert_eq!(openai_messages[0]["content"], "system prompt");
        assert_eq!(openai_messages[1]["role"], "user");
        assert_eq!(openai_messages[1]["content"], "hello");
    }

    #[test]
    fn test_openai_request_body_user_image_uses_content_array() {
        let provider = OpenAiProvider::new(
            "test-key".to_string(),
            "gpt-4o".to_string(),
            None,
            None,
            None,
        );
        let message =
            Message::user_with_images("what is this", vec![crate::provider::ImageSource {
                source_type: "base64".to_string(),
                media_type: "image/png".to_string(),
                data: "QUJD".to_string(),
            }]);
        let body = provider.build_request_body("", &[message], &[], false);
        let user = &body["messages"].as_array().expect("messages")[0];
        assert_eq!(user["role"], "user");
        let parts = user["content"].as_array().expect("content array");
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "what is this");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[1]["image_url"]["url"], "data:image/png;base64,QUJD");
    }

    #[test]
    fn test_openai_request_body_max_output_tokens_override_without_effort() {
        // No reasoning_effort, but an explicit cap: `max_completion_tokens` is set.
        let provider = OpenAiProvider::new(
            "test-key".to_string(),
            "gpt-4o".to_string(),
            None,
            None,
            Some(8_000),
        );
        let body = provider.build_request_body("", &[Message::user("hi")], &[], false);
        assert_eq!(body["max_completion_tokens"], 8_000);
    }

    #[test]
    fn test_openai_request_body_max_output_tokens_override_wins_over_effort_default() {
        // With effort the default cap is 32k; the profile override replaces it.
        let provider = OpenAiProvider::new(
            "test-key".to_string(),
            "gpt-5".to_string(),
            None,
            Some("high".to_string()),
            Some(120_000),
        );
        let body = provider.build_request_body("", &[Message::user("hi")], &[], false);
        assert_eq!(body["max_completion_tokens"], 120_000);
        assert_eq!(body["reasoning_effort"], "high");
    }

    #[test]
    fn test_openai_request_body_no_cap_without_effort_or_override() {
        let provider = OpenAiProvider::new(
            "test-key".to_string(),
            "gpt-4o".to_string(),
            None,
            None,
            None,
        );
        let body = provider.build_request_body("", &[Message::user("hi")], &[], false);
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn test_openai_request_body_reasoning_effort_defaults_by_model() {
        // Unset effort on a recognized reasoning model resolves to its strongest tier.
        let provider = OpenAiProvider::new(
            "test-key".to_string(),
            "gpt-5.6-sol".to_string(),
            None,
            None,
            None,
        );
        let body = provider.build_request_body("", &[Message::user("hi")], &[], false);
        assert_eq!(body["reasoning_effort"], "xhigh");
        // An unrecognized (local) model omits the field even with effort unset.
        let local = OpenAiProvider::new(
            "test-key".to_string(),
            "llama3.1".to_string(),
            None,
            None,
            None,
        );
        let body = local.build_request_body("", &[Message::user("hi")], &[], false);
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn test_openai_request_body_with_tools() {
        let provider = OpenAiProvider::new(
            "test-key".to_string(),
            "gpt-4o".to_string(),
            None,
            None,
            None,
        );

        let tools = vec![ToolDefinition::new(
            "read_file".to_string(),
            "Read a file".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }),
        )];

        let body = provider.build_request_body("", &[], &tools, false);
        let openai_tools = body["tools"].as_array().expect("tools should be array");
        assert_eq!(openai_tools.len(), 1);
        assert_eq!(openai_tools[0]["type"], "function");
        assert_eq!(openai_tools[0]["function"]["name"], "read_file");
    }

    #[test]
    fn test_openai_request_body_with_tool_calls() {
        let provider = OpenAiProvider::new(
            "test-key".to_string(),
            "gpt-4o".to_string(),
            None,
            None,
            None,
        );

        let messages = vec![
            Message::user("read /tmp/test.txt"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/test.txt"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "file contents here".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ];

        let body = provider.build_request_body("", &messages, &[], false);
        let openai_messages = body["messages"]
            .as_array()
            .expect("messages should be array");

        assert_eq!(openai_messages[0]["role"], "user");
        assert_eq!(openai_messages[1]["role"], "assistant");
        assert!(openai_messages[1].get("tool_calls").is_some());
        assert_eq!(openai_messages[2]["role"], "tool");
        assert_eq!(openai_messages[2]["tool_call_id"], "call_1");
    }

    #[test]
    fn test_openai_parse_non_streaming_text() {
        let provider = OpenAiProvider::new(
            "test-key".to_string(),
            "gpt-4o".to_string(),
            None,
            None,
            None,
        );

        let response = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello there!"
                },
                "finish_reason": "stop"
            }]
        });

        let (message, stop_reason, _) = provider
            .parse_non_streaming_response(&response)
            .expect("should parse");

        assert_eq!(message.text_content(), "Hello there!");
        assert_eq!(stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn test_openai_parse_non_streaming_tool_call() {
        let provider = OpenAiProvider::new(
            "test-key".to_string(),
            "gpt-4o".to_string(),
            None,
            None,
            None,
        );

        let response = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{\"path\":\"/tmp/test.txt\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let (message, stop_reason, _) = provider
            .parse_non_streaming_response(&response)
            .expect("should parse");

        assert_eq!(stop_reason, StopReason::ToolUse);
        let tool_uses = message.tool_uses();
        assert_eq!(tool_uses.len(), 1);

        if let ContentBlock::ToolUse { id, name, input } = &tool_uses[0] {
            assert_eq!(id, "call_abc");
            assert_eq!(name, "read_file");
            assert_eq!(input["path"], "/tmp/test.txt");
        } else {
            panic!("expected ToolUse block");
        }
    }

    #[test]
    fn test_openai_parse_non_streaming_malformed_tool_args() {
        let provider = OpenAiProvider::new(
            "test-key".to_string(),
            "gpt-4o".to_string(),
            None,
            None,
            None,
        );

        let response = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_bad",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{not valid json"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let (message, ..) = provider
            .parse_non_streaming_response(&response)
            .expect("envelope should parse even with bad tool args");

        let tool_uses = message.tool_uses();
        assert_eq!(tool_uses.len(), 1);
        if let ContentBlock::ToolUse { input, .. } = &tool_uses[0] {
            assert!(
                input
                    .get(crate::provider::INVALID_TOOL_ARGS_MARKER)
                    .and_then(|reason| reason.as_str())
                    .is_some(),
                "malformed args must surface the invalid-args sentinel, got: {}",
                input
            );
        } else {
            panic!("expected ToolUse block");
        }
    }

    #[test]
    fn test_openai_parse_missing_message_in_choice() {
        let provider = OpenAiProvider::new(
            "test-key".to_string(),
            "gpt-4o".to_string(),
            None,
            None,
            None,
        );

        let response = serde_json::json!({
            "choices": [{
                "finish_reason": "stop"
            }]
        });

        let result = provider.parse_non_streaming_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn test_openai_parse_missing_tool_call_id() {
        let provider = OpenAiProvider::new(
            "test-key".to_string(),
            "gpt-4o".to_string(),
            None,
            None,
            None,
        );

        let response = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let result = provider.parse_non_streaming_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn test_openai_parse_missing_tool_call_function_name() {
        let provider = OpenAiProvider::new(
            "test-key".to_string(),
            "gpt-4o".to_string(),
            None,
            None,
            None,
        );

        let response = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "arguments": "{}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let result = provider.parse_non_streaming_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn test_openai_parse_non_streaming_flattened_tool_call() {
        let provider = OpenAiProvider::new(
            "test-key".to_string(),
            "gpt-4o".to_string(),
            None,
            None,
            None,
        );

        let response = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "name": "read_file",
                        "arguments": "{\"path\":\"/tmp/test.txt\"}"
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let (message, stop_reason, _) = provider
            .parse_non_streaming_response(&response)
            .expect("should parse flattened tool call");

        assert_eq!(stop_reason, StopReason::ToolUse);
        let tool_uses = message.tool_uses();
        assert_eq!(tool_uses.len(), 1);

        if let ContentBlock::ToolUse { id, name, input } = &tool_uses[0] {
            assert_eq!(id, "call_abc");
            assert_eq!(name, "read_file");
            assert_eq!(input["path"], "/tmp/test.txt");
        } else {
            panic!("expected ToolUse block");
        }
    }

    #[test]
    fn test_openai_parse_non_streaming_flattened_missing_name_still_errors() {
        let provider = OpenAiProvider::new(
            "test-key".to_string(),
            "gpt-4o".to_string(),
            None,
            None,
            None,
        );

        let response = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "arguments": "{}"
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let result = provider.parse_non_streaming_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn test_openai_tool_definitions_use_standard_chat_completions_format() {
        let provider = OpenAiProvider::new(
            "test-key".to_string(),
            "gpt-4o".to_string(),
            None,
            None,
            None,
        );

        let tools = vec![ToolDefinition::new(
            "write_file".to_string(),
            "Create or overwrite a file".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" }
                },
                "required": ["path", "content"]
            }),
        )];

        let body = provider.build_request_body("", &[], &tools, false);
        let openai_tools = body["tools"].as_array().expect("tools should be array");

        assert_eq!(openai_tools[0]["type"], "function");
        assert_eq!(openai_tools[0]["function"]["name"], "write_file");
        assert_eq!(
            openai_tools[0]["function"]["description"],
            "Create or overwrite a file"
        );
        assert!(openai_tools[0]["function"].get("parameters").is_some());

        // Top-level name/description/parameters must NOT be present to avoid triggering Responses
        // API strict validation on OpenAI/OpenRouter
        assert!(openai_tools[0].get("name").is_none());
        assert!(openai_tools[0].get("description").is_none());
        assert!(openai_tools[0].get("parameters").is_none());
    }
}
