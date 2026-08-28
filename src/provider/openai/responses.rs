//! `openai-responses`: the Responses API against any endpoint that serves it, with an API key.
//!
//! The protocol sibling of [`super::chat_completions`] and the auth sibling of
//! [`super::subscription`]. It posts to `{base_url}/responses` with a bearer token, which reaches
//! OpenAI itself and equally reaches Ollama (v0.13.3+), vLLM, LM Studio and OpenRouter -- all of
//! which implement the same endpoint. The wire format lives in [`super::responses_wire`].
//!
//! What this backend deliberately does *not* send is
//! [`super::responses_wire::include_encrypted_reasoning`] or
//! [`super::responses_wire::request_reasoning_summary`]. Both are OpenAI extensions, and
//! `chatgpt-subscription` may assume them because its endpoint is always ChatGPT. Here the endpoint
//! is whatever `base_url` names, meka has no way to know whether either is understood, and guessing
//! wrong costs a rejected request rather than a degraded one.
//!
//! The cost of that choice is real and worth naming: without the `include` there is no encrypted
//! reasoning to replay, so a turn here carries no reasoning chain between its own tool calls, and
//! without the summary the reasoning stays invisible. An endpoint that serves reasoning of its own
//! accord -- vLLM and Ollama emit `response.reasoning_text.delta` unprompted -- still renders.

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::responses_wire::{
    aggregate_stream, build_request_body, drive_responses_sse_stream, drop_replayed_reasoning,
};
use crate::{
    error::Result,
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
    /// parallel copy would not: adding `include_encrypted_reasoning` or `request_reasoning_summary`
    /// here is the regression the protocol/endpoint split exists to prevent, and against a
    /// duplicate builder that change is invisible to every test in the suite.
    fn build_body(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> serde_json::Value {
        let mut body = build_request_body(
            &self.model,
            system_prompt,
            messages,
            tools,
            self.wire_effort().as_deref(),
            self.max_output_tokens,
            true,
        );
        drop_replayed_reasoning(&mut body);
        body
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
                crate::error::provider_transport_error(
                    "Responses HTTP request failed",
                    &error,
                    None,
                )
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

    /// A stream that never started is retryable here too.
    ///
    /// The sibling of the pair in `provider::anthropic::messages`, and here for the same reason:
    /// this backend has its own `.send()` site, so it is its own wiring into
    /// [`crate::error::provider_transport_error`] and its own chance to go back to a bare
    /// `MekaError::Provider` that the agent loop discards.
    #[tokio::test]
    async fn a_stream_that_could_not_start_reports_a_retryable_failure() {
        // Bound and dropped, so the port is refused rather than answered or hung.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind an ephemeral port");
        let port = listener.local_addr().expect("the bound address").port();
        drop(listener);

        let provider = provider(None, Some(&format!("http://127.0.0.1:{port}/v1")));
        let (sender, _receiver) = mpsc::channel(8);
        let error = provider
            .stream(
                "",
                &[Message::user("hello")],
                &[],
                sender,
                CancellationToken::new(),
            )
            .await
            .expect_err("nothing is listening there");

        assert!(
            matches!(error, crate::error::MekaError::RetryableProvider { .. }),
            "a stream that never started must be retryable, got: {error}"
        );
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

    /// This backend must never send `include: ["reasoning.encrypted_content"]`, nor ask for a
    /// `reasoning.summary`.
    ///
    /// Both are OpenAI extensions, and this backend reaches Ollama, vLLM, LM Studio and OpenRouter,
    /// where an unrecognised field is a rejected request. `chatgpt-subscription` sends them because
    /// its endpoint is always ChatGPT; the split is the whole reason they were lifted out of the
    /// shared body builder, so it is asserted on both sides.
    #[test]
    fn an_openai_extension_is_never_sent_to_an_endpoint_that_may_not_know_it() {
        // Even with effort set, which is the condition that used to pull `include` in.
        let body = provider(Some("high"), None).build_body("s", &[Message::user("hi")], &[]);
        assert_eq!(body["reasoning"]["effort"], "high");
        assert!(body.get("include").is_none(), "{body}");
        assert!(body["reasoning"].get("summary").is_none(), "{body}");

        // And the protocol-level fields every implementation understands are still there.
        assert_eq!(body["store"], false);
        assert_eq!(body["stream"], true);
        assert_eq!(body["instructions"], "s");
    }

    /// Sealed reasoning recorded elsewhere must not be replayed to whatever `base_url` names.
    ///
    /// A session is not bound to the provider that recorded it, so `meka -c -p local` after a
    /// `chatgpt-subscription` turn hands this backend a history full of ChatGPT's sealed blobs.
    /// Shipping those to Ollama or OpenRouter leaks them to a third party that cannot read them,
    /// and puts an item shape on the wire that the endpoint never agreed to -- the same argument
    /// that keeps the `include` and the `summary` off it.
    #[test]
    fn sealed_reasoning_from_another_endpoint_is_never_replayed() {
        let history = vec![
            Message::user("hi"),
            Message {
                role: crate::provider::Role::Assistant,
                content: vec![
                    crate::provider::ContentBlock::Thinking {
                        thinking: "summary".to_string(),
                        opaque: Some(crate::provider::OpaqueReasoning::Sealed {
                            encrypted_content: "CHATGPT_SEALED".to_string(),
                            id: Some("rs_1".to_string()),
                        }),
                    },
                    crate::provider::ContentBlock::Text {
                        text: "answer".to_string(),
                    },
                ],
            },
            Message::user("go on"),
        ];
        let body = provider(Some("high"), None).build_body("s", &history, &[]);
        let serialized = serde_json::to_string(&body).expect("serialize");

        assert!(!serialized.contains("CHATGPT_SEALED"), "{serialized}");
        assert!(!serialized.contains("rs_1"), "{serialized}");
        assert!(
            !body["input"]
                .as_array()
                .expect("input")
                .iter()
                .any(|item| item["type"] == "reasoning"),
            "{serialized}"
        );
        // And the turn it belonged to is still there, so nothing but the item was dropped.
        assert!(serialized.contains("answer"), "{serialized}");
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
