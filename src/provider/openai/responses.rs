//! `openai-responses`: the Responses API against any endpoint that serves it, with an API key.
//!
//! The protocol sibling of [`super::chat_completions`] and the auth sibling of
//! [`super::subscription`]. It posts to `{base_url}/responses` with a bearer token, which reaches
//! OpenAI itself and equally reaches Ollama (v0.13.3+), vLLM, LM Studio and OpenRouter -- all of
//! which implement the same endpoint. The wire format lives in [`super::responses_wire`].
//!
//! What this backend deliberately does *not* send is
//! [`super::responses_wire::include_encrypted_reasoning`]. That `include` is an OpenAI extension,
//! and `chatgpt-subscription` may assume it because its endpoint is always ChatGPT. Here the
//! endpoint is whatever `base_url` names, meka has no way to know whether it is understood, and
//! guessing wrong costs a rejected request rather than a degraded one. meka only replays whole
//! conversations anyway, so nothing depends on the server round-tripping reasoning.

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::responses_wire::{aggregate_stream, build_request_body, drive_responses_sse_stream};
use crate::{
    error::{MekaError, Result},
    provider::{Message, Notice, Provider, StopReason, StreamEvent, TokenUsage, ToolDefinition},
};

pub struct OpenAiResponsesProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    /// The settled `reasoning.effort` for the request body, resolved once at construction from the
    /// profile's override. `None` - the unconfigured case - omits the `reasoning` block so the
    /// endpoint applies its own default, which matters most for the local servers this backend
    /// also reaches.
    resolved_effort: Option<String>,
    /// Per-request output token cap from the profile; `None` leaves the endpoint's default.
    max_output_tokens: Option<u64>,
}

impl OpenAiResponsesProvider {
    pub fn new(
        api_key: String,
        model: String,
        base_url: Option<String>,
        reasoning_effort: Option<String>,
        max_output_tokens: Option<u64>,
    ) -> Result<Self> {
        let resolved_effort = crate::provider::resolve_effort_level(reasoning_effort.as_deref());
        Ok(Self {
            client: crate::provider::build_http_client("openai-responses", |builder| builder)?,
            api_key,
            // The same normalizer Chat Completions uses, and for the same reason:
            // `{base}/responses` composes exactly as `{base}/chat/completions` does, so `https://api.openai.com/v1`,
            // `http://localhost:11434/v1` and `https://openrouter.ai/api/v1` all work verbatim.
            base_url: crate::provider::normalize_base_url(
                base_url
                    .as_deref()
                    .unwrap_or(crate::provider::DEFAULT_OPENAI_BASE_URL),
            ),
            model,
            resolved_effort,
            max_output_tokens,
        })
    }

    /// The settled reasoning-effort to send as `reasoning.effort` (see [`Self::resolved_effort`]).
    fn wire_effort(&self) -> Option<String> {
        self.resolved_effort.clone()
    }

    fn responses_url(&self) -> String {
        format!("{}/responses", self.base_url)
    }

    /// The request body, exactly as `stream` sends it.
    ///
    /// A named method rather than inline in `stream`, and the *only* body builder this backend has,
    /// so a test asserting what is on the wire is asserting the shipping path. A `#[cfg(test)]`
    /// parallel copy would not: adding `include_encrypted_reasoning` here is the one regression the
    /// protocol/endpoint split exists to prevent, and against a duplicate builder that change is
    /// invisible to every test in the suite.
    fn build_body(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> serde_json::Value {
        build_request_body(
            &self.model,
            system_prompt,
            messages,
            tools,
            self.wire_effort().as_deref(),
            self.max_output_tokens,
            true,
        )
    }
}

#[async_trait]
impl Provider for OpenAiResponsesProvider {
    async fn complete(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<(Message, StopReason, TokenUsage, Vec<Notice>)> {
        // Streaming-only, like the subscription backend: the Responses API's non-streaming shape is
        // a second decoder to keep correct for no gain, so `complete` folds its own SSE. Both
        // futures are awaited together so the channel drains while the stream fills it; awaiting
        // them in sequence would deadlock on a full buffer.
        let (event_sender, event_receiver) = mpsc::channel::<StreamEvent>(1024);
        let (stream_result, aggregated) = tokio::join!(
            self.stream(
                system_prompt,
                messages,
                tools,
                event_sender,
                CancellationToken::new(),
            ),
            aggregate_stream(event_receiver),
        );
        stream_result?;
        Ok(aggregated)
    }

    async fn stream(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        event_sender: mpsc::Sender<StreamEvent>,
        cancellation: CancellationToken,
    ) -> Result<()> {
        let body = self.build_body(system_prompt, messages, tools);

        let response = self
            .client
            .post(self.responses_url())
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Accept", "text/event-stream")
            .json(&body)
            .send()
            .await
            .map_err(|error| {
                MekaError::Provider(format!(
                    "Responses HTTP request failed: {}",
                    crate::error::format_reqwest_error(&error)
                ))
            })?;

        drive_responses_sse_stream(response, event_sender, cancellation).await
    }

    fn name(&self) -> &str {
        "openai-responses"
    }

    fn resolved_effort(&self) -> Option<String> {
        self.wire_effort()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(effort: Option<&str>, base_url: Option<&str>) -> OpenAiResponsesProvider {
        OpenAiResponsesProvider::new(
            "test-key".to_string(),
            "gpt-5.6-sol".to_string(),
            base_url.map(str::to_string),
            effort.map(str::to_string),
            None,
        )
        .expect("build test provider")
    }

    /// The endpoint is composed the same way Chat Completions composes its own.
    ///
    /// `{base}/responses` has to work verbatim for the published base URL of every server that
    /// serves this protocol, which is why it reuses `normalize_base_url` rather than inventing a
    /// second rule.
    #[test]
    fn the_url_composes_against_every_published_base() {
        for (base, expected) in [
            (None, "https://api.openai.com/v1/responses"),
            (
                Some("https://api.openai.com/v1"),
                "https://api.openai.com/v1/responses",
            ),
            (
                Some("http://127.0.0.1:11434/v1"),
                "http://127.0.0.1:11434/v1/responses",
            ),
            (
                Some("https://openrouter.ai/api/v1"),
                "https://openrouter.ai/api/v1/responses",
            ),
            // A trailing slash must not double up.
            (
                Some("http://127.0.0.1:11434/v1/"),
                "http://127.0.0.1:11434/v1/responses",
            ),
        ] {
            assert_eq!(provider(None, base).responses_url(), expected, "{base:?}");
        }
    }

    /// This backend must never send `include: ["reasoning.encrypted_content"]`.
    ///
    /// It is an OpenAI extension, and this backend reaches Ollama, vLLM, LM Studio and OpenRouter,
    /// where an unrecognised field is a rejected request. `chatgpt-subscription` sends it because
    /// its endpoint is always ChatGPT; the split is the whole reason the `include` was lifted out
    /// of the shared body builder, so it is asserted on both sides.
    #[test]
    fn an_openai_extension_is_never_sent_to_an_endpoint_that_may_not_know_it() {
        // Even with effort set, which is the condition that used to pull `include` in.
        let body = provider(Some("high"), None).build_body("s", &[Message::user("hi")], &[]);
        assert_eq!(body["reasoning"]["effort"], "high");
        assert!(body.get("include").is_none(), "{body}");

        // And the protocol-level fields every implementation understands are still there.
        assert_eq!(body["store"], false);
        assert_eq!(body["stream"], true);
        assert_eq!(body["instructions"], "s");
    }

    /// Effort follows the same rule as every other backend: sent only when the profile asks.
    #[test]
    fn an_unconfigured_effort_omits_the_reasoning_block() {
        let body = provider(None, None).build_body("", &[Message::user("hi")], &[]);
        assert!(body.get("reasoning").is_none(), "{body}");
        assert!(body.get("include").is_none(), "{body}");
        // An empty system prompt sends no `instructions` rather than an empty one.
        assert!(body.get("instructions").is_none(), "{body}");
    }
}
