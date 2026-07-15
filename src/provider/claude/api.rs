//! Direct Claude Messages API provider. Uses `x-api-key` auth without the Claude Code
//! fingerprinting / attestation machinery that `claude-oauth` requires. Intended for users bringing
//! their own `CLAUDE_API_KEY`.

use std::sync::{
    Arc,
    atomic::{AtomicI8, Ordering},
};

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::shared::{
    self, convert_messages_to_claude_content, convert_tools_to_claude_tools,
    drive_claude_sse_stream, parse_non_streaming_response,
};
use crate::{
    error::{MekaError, Result},
    provider::{
        Message, ModelInfo, Notice, Provider, StopReason, StreamEvent, TokenUsage, ToolDefinition,
    },
};

pub struct ClaudeApiProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    thinking_enabled: bool,
    thinking_budget_tokens: u64,
    thinking_override: AtomicI8,
    /// Per-request output token cap from the profile; `None` keeps the built-in default.
    max_output_tokens: Option<u64>,
    /// Per-session counters incremented when image-redaction events fire.
    session_stats: Option<Arc<crate::stats::SessionStats>>,
}

impl ClaudeApiProvider {
    pub fn new(
        api_key: String,
        model: String,
        base_url: Option<String>,
        thinking_enabled: bool,
        thinking_budget_tokens: u64,
        max_output_tokens: Option<u64>,
        session_stats: Option<Arc<crate::stats::SessionStats>>,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://api.anthropic.com".to_string()),
            model,
            thinking_enabled,
            thinking_budget_tokens,
            thinking_override: AtomicI8::new(-1),
            max_output_tokens,
            session_stats,
        }
    }

    fn is_thinking_enabled(&self) -> bool {
        shared::resolve_thinking_enabled(&self.thinking_override, self.thinking_enabled)
    }

    fn compute_betas(&self) -> Option<String> {
        if self.is_thinking_enabled() {
            Some("interleaved-thinking-2025-05-14".to_string())
        } else {
            None
        }
    }

    pub(super) fn build_request_body(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        stream: bool,
    ) -> serde_json::Value {
        let claude_messages = convert_messages_to_claude_content(messages);

        let mut body = serde_json::Map::new();
        body.insert("model".to_string(), serde_json::json!(self.model));
        if !system_prompt.is_empty() {
            body.insert("system".to_string(), serde_json::json!(system_prompt));
        }
        body.insert("messages".to_string(), serde_json::json!(claude_messages));

        shared::insert_thinking_fields(
            &mut body,
            self.is_thinking_enabled(),
            &self.model,
            self.thinking_budget_tokens,
            self.max_output_tokens,
        );

        body.insert("stream".to_string(), serde_json::json!(stream));

        if !tools.is_empty() {
            body.insert(
                "tools".to_string(),
                serde_json::json!(convert_tools_to_claude_tools(tools)),
            );
        }

        serde_json::Value::Object(body)
    }

    fn apply_headers(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut request = request
            .header("accept", "application/json")
            .header("content-type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .header("x-api-key", &self.api_key);

        if let Some(betas) = self.compute_betas() {
            request = request.header("anthropic-beta", betas);
        }

        request
    }
}

#[async_trait]
impl Provider for ClaudeApiProvider {
    async fn complete(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<(Message, StopReason, TokenUsage, Vec<Notice>)> {
        let (body_json, redaction_notice) =
            shared::build_body_within_budget(messages, self.session_stats.as_ref(), |msgs| {
                serde_json::to_string(&self.build_request_body(system_prompt, msgs, tools, false))
                    .map_err(|error| {
                        MekaError::Provider(format!("failed to serialize body: {}", error))
                    })
            })?;
        let body_size_mib = body_json.len() / 1_048_576;
        let request = self
            .apply_headers(self.client.post(format!("{}/v1/messages", self.base_url)))
            .body(body_json);

        let response = request.send().await.map_err(|error| {
            MekaError::Provider(format!(
                "HTTP request failed (body {} MiB): {}",
                body_size_mib,
                crate::error::format_reqwest_error(&error),
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

        let (message, stop_reason, usage) = parse_non_streaming_response(&response_json)?;
        let notices = redaction_notice.into_iter().collect();
        Ok((message, stop_reason, usage, notices))
    }

    async fn stream(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        event_sender: mpsc::Sender<StreamEvent>,
        cancellation: CancellationToken,
    ) -> Result<()> {
        let (body_json, redaction_notice) =
            shared::build_body_within_budget(messages, self.session_stats.as_ref(), |msgs| {
                serde_json::to_string(&self.build_request_body(system_prompt, msgs, tools, true))
                    .map_err(|error| {
                        MekaError::Provider(format!("failed to serialize body: {}", error))
                    })
            })?;
        // Surface the redaction notice as the first stream event so the frontend renders it before
        // any provider text appears. The agent's `run_streaming` translates it to
        // `FrontendEvent::Notice`. Send-error here means the consumer hung up between this call
        // and now; `drive_claude_sse_stream` will surface that on its own.
        if let Some(notice) = redaction_notice
            && let Err(error) = event_sender.send(StreamEvent::Notice(notice)).await
        {
            tracing::debug!("failed to forward redaction notice into stream: {}", error);
        }
        let body_size_mib = body_json.len() / 1_048_576;
        let request = self
            .apply_headers(
                self.client
                    .post(format!("{}/v1/messages", self.base_url))
                    .header("accept-encoding", "identity"),
            )
            .body(body_json);

        let response = request.send().await.map_err(|error| {
            MekaError::Provider(format!(
                "HTTP request failed (body {} MiB): {}",
                body_size_mib,
                crate::error::format_reqwest_error(&error),
            ))
        })?;

        drive_claude_sse_stream(response, event_sender, cancellation).await
    }

    fn name(&self) -> &str {
        "claude-api"
    }

    fn set_thinking_override(&self, enabled: Option<bool>) {
        let value = match enabled {
            None => -1,
            Some(false) => 0,
            Some(true) => 1,
        };
        self.thinking_override.store(value, Ordering::Relaxed);
    }

    async fn fetch_model_info(&self) -> Result<Option<ModelInfo>> {
        let request = self.apply_headers(
            self.client
                .get(format!("{}/v1/models/{}", self.base_url, self.model)),
        );
        let response = request.send().await.map_err(|error| {
            MekaError::Provider(format!(
                "model info request failed: {}",
                crate::error::format_reqwest_error(&error),
            ))
        })?;
        let status = response.status();
        let retry_after = crate::error::parse_retry_after(response.headers());
        let text = response.text().await.map_err(|error| {
            MekaError::Provider(format!("failed to read model info response: {}", error))
        })?;
        if !status.is_success() {
            return Err(crate::error::provider_http_error(
                status,
                &text,
                retry_after,
            ));
        }
        super::shared::model_info_from_claude_model(&text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_provider() -> ClaudeApiProvider {
        ClaudeApiProvider::new(
            "test-key".to_string(),
            "claude-sonnet-4-20250514".to_string(),
            None,
            false,
            10000,
            None,
            None,
        )
    }

    #[test]
    fn test_api_body_has_no_billing_header() {
        let provider = test_provider();
        let messages = vec![Message::user("hello")];
        let body = provider.build_request_body("be nice", &messages, &[], false);

        let serialized = serde_json::to_string(&body).unwrap();
        assert!(
            !serialized.contains("cc_version"),
            "claude-api body must not contain Claude Code billing header: {}",
            serialized
        );
        assert!(
            !serialized.contains("cc_entrypoint"),
            "claude-api body must not contain Claude Code entrypoint tag: {}",
            serialized
        );
        assert!(
            !serialized.contains("cch="),
            "claude-api body must not contain cch attestation placeholder: {}",
            serialized
        );
    }

    #[test]
    fn test_api_body_has_no_metadata() {
        let provider = test_provider();
        let body = provider.build_request_body("", &[Message::user("hi")], &[], false);
        assert!(
            body.get("metadata").is_none(),
            "claude-api body must not include metadata.user_id"
        );
    }

    #[test]
    fn test_api_body_plain_string_system_prompt() {
        let provider = test_provider();
        let body = provider.build_request_body("my system", &[Message::user("hi")], &[], false);
        let system = body.get("system").unwrap();
        assert_eq!(
            system.as_str(),
            Some("my system"),
            "claude-api should serialize `system` as a plain string"
        );
    }

    #[test]
    fn test_api_body_omits_system_when_empty() {
        let provider = test_provider();
        let body = provider.build_request_body("", &[Message::user("hi")], &[], false);
        assert!(
            body.get("system").is_none(),
            "claude-api should omit `system` when the prompt is empty"
        );
    }
}
