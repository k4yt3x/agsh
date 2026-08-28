//! `anthropic-messages`: the Anthropic Messages API against any endpoint serving it, with an API
//! key.
//!
//! `POST {base_url}/v1/messages` with `x-api-key`, and none of the Claude Code fingerprinting or
//! attestation machinery [`super::subscription`] needs -- the two speak the same protocol and
//! differ only in how they authenticate and whose client they look like. The wire format is shared
//! through [`super::shared`].
//!
//! The key comes from the profile's stored credential, never from the environment: meka reads no
//! provider env vars at all, so an ambient key cannot silently rebind which account a profile
//! bills.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
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
        Message, Notice, Provider, StopReason, StreamEvent, ThinkingMode, TokenUsage,
        ToolDefinition,
    },
};

pub struct AnthropicMessagesProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    thinking: ThinkingMode,
    thinking_budget_tokens: u64,
    /// Set while an internal turn (compaction) runs, so its summary doesn't pay for reasoning.
    /// Only ever suppresses; it cannot turn thinking on for a profile that asked for none.
    thinking_suppressed: AtomicBool,
    /// The settled `output_config.effort` for the request body, resolved once at construction from
    /// the profile's override. `None` - the unconfigured case - omits the field so Anthropic (or
    /// whatever endpoint `base_url` names) applies its own default. The direct Messages API takes
    /// effort with no beta header.
    resolved_effort: Option<String>,
    /// Per-request output token cap from the profile; `None` keeps the built-in default.
    max_output_tokens: Option<u64>,
    /// Per-session counters incremented when image-redaction events fire.
    session_stats: Option<Arc<crate::stats::SessionStats>>,
}

impl AnthropicMessagesProvider {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        api_key: String,
        model: String,
        base_url: Option<String>,
        thinking: ThinkingMode,
        thinking_budget_tokens: u64,
        effort: Option<String>,
        max_output_tokens: Option<u64>,
        session_stats: Option<Arc<crate::stats::SessionStats>>,
    ) -> Result<Self> {
        let resolved_effort = crate::provider::resolve_effort_level(effort.as_deref());
        Ok(Self {
            client: crate::provider::build_http_client("anthropic-messages", |builder| builder)?,
            api_key,
            base_url: shared::normalize_claude_base_url(
                base_url
                    .as_deref()
                    .unwrap_or(crate::provider::DEFAULT_ANTHROPIC_BASE_URL),
            ),
            model,
            thinking,
            thinking_budget_tokens,
            thinking_suppressed: AtomicBool::new(false),
            resolved_effort,
            max_output_tokens,
            session_stats,
        })
    }

    fn effective_thinking(&self) -> ThinkingMode {
        shared::effective_thinking(&self.thinking_suppressed, self.thinking)
    }

    /// The settled effort to send as `output_config.effort` (see [`Self::resolved_effort`]).
    fn wire_effort(&self) -> Option<String> {
        self.resolved_effort.clone()
    }

    fn compute_betas(&self) -> Option<String> {
        let mut parts: Vec<&str> = Vec::new();
        if self.effective_thinking().is_on() {
            parts.push("interleaved-thinking-2025-05-14");
        }
        // No `context-1m-2025-08-07`: on the direct Messages API, 1M context is the *default* for
        // the current large-context models (Opus 4.6+, Sonnet 4.6, Fable 5) with no beta header,
        // so the 1M window is already what the request gets. See
        // <https://platform.claude.com/docs/en/build-with-claude/context-windows>. The
        // claude-subscription path does not send it either; see `compute_betas` there.
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(","))
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
            self.effective_thinking(),
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

        // The direct Messages API takes `output_config.effort` in the body with no beta header
        // (unlike claude-subscription, which mirrors Claude Code's `effort-2025-11-24` beta). See
        // <https://platform.claude.com/docs/en/build-with-claude/effort>.
        if let Some(effort) = self.wire_effort() {
            body.insert(
                "output_config".to_string(),
                serde_json::json!({ "effort": effort }),
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
impl Provider for AnthropicMessagesProvider {
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
        let body_length = body_json.len();
        let request = self
            .apply_headers(self.client.post(format!("{}/v1/messages", self.base_url)))
            .body(body_json);

        let response = request.send().await.map_err(|error| {
            crate::error::provider_transport_error(
                &format!(
                    "HTTP request failed (body {} MiB)",
                    shared::body_size_mib(body_length)
                ),
                &error,
                None,
            )
        })?;

        let status = response.status();
        let retry_after = crate::error::parse_retry_after(response.headers());
        let response_text = response.text().await.map_err(|error| {
            crate::error::provider_transport_error("failed to read response", &error, retry_after)
        })?;

        if !status.is_success() {
            return Err(crate::error::provider_http_error(
                status,
                &response_text,
                retry_after,
                crate::error::ProviderRequest::Completion,
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
        let body_length = body_json.len();
        let request = self
            .apply_headers(
                self.client
                    .post(format!("{}/v1/messages", self.base_url))
                    .header("accept-encoding", "identity"),
            )
            .body(body_json);

        let response = request.send().await.map_err(|error| {
            crate::error::provider_transport_error(
                &format!(
                    "HTTP request failed (body {} MiB)",
                    shared::body_size_mib(body_length)
                ),
                &error,
                None,
            )
        })?;

        drive_claude_sse_stream(response, event_sender, cancellation).await
    }

    fn name(&self) -> &str {
        "anthropic-messages"
    }

    fn resolved_effort(&self) -> Option<String> {
        self.wire_effort()
    }

    fn suppress_thinking(&self, suppressed: bool) {
        self.thinking_suppressed
            .store(suppressed, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_provider() -> AnthropicMessagesProvider {
        provider("claude-sonnet-4-20250514", None)
    }

    fn provider(model: &str, effort: Option<&str>) -> AnthropicMessagesProvider {
        provider_with_base(model, effort, None)
    }

    fn provider_with_base(
        model: &str,
        effort: Option<&str>,
        base_url: Option<&str>,
    ) -> AnthropicMessagesProvider {
        AnthropicMessagesProvider::new(
            "test-key".to_string(),
            model.to_string(),
            base_url.map(str::to_string),
            ThinkingMode::Off,
            10000,
            effort.map(str::to_string),
            None,
            None,
        )
        .expect("build test provider")
    }

    /// A backend that could not reach its endpoint reports a failure the agent loop will retry.
    ///
    /// This is the wiring rather than the rule: [`crate::error::provider_transport_error`] has its
    /// own tests, and what this one asserts is that a real `.send()` site actually calls it. Every
    /// such site in every backend was hand-rolling a bare `MekaError::Provider`, which the retry
    /// loop discards, so a site that quietly goes back to doing that is the regression worth
    /// catching. One backend stands for the pattern; the classifier they all share is what makes
    /// that enough.
    #[tokio::test]
    async fn a_backend_that_could_not_reach_its_endpoint_reports_a_retryable_failure() {
        // Bound and dropped, so the port is refused rather than answered or hung.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind an ephemeral port");
        let port = listener.local_addr().expect("the bound address").port();
        drop(listener);

        let provider = provider_with_base(
            "claude-sonnet-4-20250514",
            None,
            Some(&format!("http://127.0.0.1:{port}")),
        );
        let error = provider
            .complete("", &[Message::user("hello")], &[])
            .await
            .expect_err("nothing is listening there");

        assert!(
            matches!(error, MekaError::RetryableProvider { .. }),
            "a call that never reached the provider must be retryable, got: {error}"
        );
    }

    /// The same, for the streaming path, which is the one an interactive turn actually takes.
    ///
    /// Worth its own test rather than trusting the sibling above: `complete` and `stream` build and
    /// send their requests separately, so they are two wirings, and it was the streaming one that
    /// ended a real session by reporting a connection reset as terminal.
    #[tokio::test]
    async fn a_backend_whose_stream_could_not_start_reports_a_retryable_failure() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind an ephemeral port");
        let port = listener.local_addr().expect("the bound address").port();
        drop(listener);

        let provider = provider_with_base(
            "claude-sonnet-4-20250514",
            None,
            Some(&format!("http://127.0.0.1:{port}")),
        );
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
            matches!(error, MekaError::RetryableProvider { .. }),
            "a stream that never started must be retryable, got: {error}"
        );
    }

    /// A read that fails after the headers arrived keeps the hint those headers carried.
    ///
    /// Parsing `Retry-After` and passing it on are separate acts, and an earlier version did only
    /// the first: every read site parsed the header into a local and then handed
    /// [`crate::error::provider_transport_error`] a `None`, so a provider that had just said "wait
    /// 60 seconds" was retried on plain 1s/2s backoff. The classifier's own tests cannot catch
    /// that, because they supply the argument themselves; this one makes a real site parse a real
    /// header. Asserting the message as well as the hint is what pins it to the read site: a 429
    /// whose body *did* arrive is also retryable and also carries the hint, but says so in
    /// `provider_http_error`'s words.
    #[tokio::test]
    async fn a_truncated_body_keeps_the_rate_limit_hint() {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind an ephemeral port");
        let port = listener.local_addr().expect("the bound address").port();
        tokio::spawn(async move {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            // Read the request out in full before answering. A response written while the client is
            // still uploading can be lost to the reset that closing an unread socket sends.
            let mut request = Vec::new();
            let mut chunk = [0u8; 4096];
            loop {
                let read = match socket.read(&mut chunk).await {
                    Ok(0) => break,
                    Ok(read) => read,
                    Err(_) => return,
                };
                request.extend_from_slice(&chunk[..read]);
                let Some(head_end) = request.windows(4).position(|window| window == b"\r\n\r\n")
                else {
                    continue;
                };
                let length: usize = String::from_utf8_lossy(&request[..head_end])
                    .split("\r\n")
                    .filter_map(|line| line.split_once(':'))
                    .find(|(name, _)| name.trim().eq_ignore_ascii_case("content-length"))
                    .and_then(|(_, value)| value.trim().parse().ok())
                    .unwrap_or(0);
                if request.len() >= head_end + 4 + length {
                    break;
                }
            }

            // A `Content-Length` the body then stops short of, so the read ends at EOF with the
            // status and headers already delivered.
            let head = "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 60\r\n\
                        Content-Type: application/json\r\nContent-Length: 4096\r\n\r\n";
            if socket.write_all(head.as_bytes()).await.is_err() {
                return;
            }
            if socket.write_all(b"{\"error\":").await.is_err() {
                return;
            }
            if socket.shutdown().await.is_err() {
                tracing::debug!("mock endpoint could not shut its socket down cleanly");
            }
        });

        let provider = provider_with_base(
            "claude-sonnet-4-20250514",
            None,
            Some(&format!("http://127.0.0.1:{port}")),
        );
        let error = provider
            .complete("", &[Message::user("hello")], &[])
            .await
            .expect_err("the body stops short of its declared length");

        match error {
            MekaError::RetryableProvider {
                message,
                retry_after,
            } => {
                assert!(
                    message.starts_with("failed to read response"),
                    "expected the read site rather than the status classifier: {message}"
                );
                assert_eq!(retry_after, Some(std::time::Duration::from_secs(60)));
            }
            other => panic!("expected a retryable failure carrying the hint, got: {other}"),
        }
    }

    /// The size in a failed send's message is reported to one decimal.
    ///
    /// Nothing else reads this string, so nothing else would notice it going wrong, and it went
    /// wrong once already: integer division reported every body from 2.0 to just under 3.0 MiB as
    /// "2 MiB". It is the figure a user quotes when asking why a request was refused, so being out
    /// by up to a megabyte sends the answer in the wrong direction.
    #[test]
    fn a_body_size_is_reported_to_one_decimal() {
        assert_eq!(shared::body_size_mib(0), "0.0");
        assert_eq!(shared::body_size_mib(1_048_576), "1.0");
        // The case truncation got wrong: two and a half megabytes is not "2 MiB".
        assert_eq!(shared::body_size_mib(2_621_440), "2.5");
        assert_eq!(shared::body_size_mib(3_145_727), "3.0");
    }

    #[test]
    fn test_a_claude_base_url_is_normalized_at_construction() {
        // The shape a gateway publishes for its Anthropic endpoint; meka appends `/v1/messages`
        // itself, so leaving this would request `/v1/v1/messages`.
        let versioned = provider_with_base(
            "claude-sonnet-4-20250514",
            None,
            Some("https://api.synthetic.new/anthropic/v1"),
        );
        assert_eq!(versioned.base_url, "https://api.synthetic.new/anthropic");

        let trailing = provider_with_base(
            "claude-sonnet-4-20250514",
            None,
            Some("https://api.anthropic.com/"),
        );
        assert_eq!(trailing.base_url, "https://api.anthropic.com");

        assert_eq!(test_provider().base_url, "https://api.anthropic.com");
    }

    #[test]
    fn an_unconfigured_profile_sends_no_output_config_whatever_the_model() {
        // No model name earns an effort tier. `anthropic-messages` reaches any Anthropic-compatible
        // endpoint, so meka cannot know which tiers the far side implements; omitting the field is
        // how it asks for whatever that endpoint's default is.
        for model in [
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-sonnet-4-20250514",
            "hf.co/bartowski/Qwen3.8-27B-GGUF:Q8_0",
        ] {
            let body =
                provider(model, None).build_request_body("s", &[Message::user("hi")], &[], false);
            assert!(body.get("output_config").is_none(), "{model}");
        }
        // A configured value is absolute: sent verbatim on any model, including one meka has never
        // heard of, because the user knows their endpoint and meka does not.
        for model in ["claude-opus-4-8", "hf.co/bartowski/Qwen3.8-27B-GGUF:Q8_0"] {
            let body = provider(model, Some("medium")).build_request_body(
                "s",
                &[Message::user("hi")],
                &[],
                false,
            );
            assert_eq!(body["output_config"]["effort"], "medium", "{model}");
        }
    }

    #[test]
    fn test_betas_omit_context_1m() {
        // The direct Messages API serves 1M context by default for 1M-capable models with no beta
        // header, so anthropic-messages never sends `context-1m-2025-08-07` (unlike
        // claude-subscription, which mirrors Claude Code's captured wire). A
        // thinking-enabled request still sends only the interleaved beta.
        let thinking_on = AnthropicMessagesProvider::new(
            "test-key".to_string(),
            "claude-opus-4-8".to_string(),
            None,
            ThinkingMode::Adaptive,
            10000,
            None,
            None,
            None,
        )
        .expect("build test provider");
        let betas = thinking_on.compute_betas().unwrap_or_default();
        assert!(betas.contains("interleaved-thinking-2025-05-14"));
        assert!(
            !betas.contains("context-1m"),
            "1M is the API default; no beta expected: {betas}"
        );
        // Thinking off on a 1M-capable model → no betas at all.
        assert!(provider("claude-opus-4-8", None).compute_betas().is_none());
    }

    #[test]
    fn test_api_body_has_no_billing_header() {
        let provider = test_provider();
        let messages = vec![Message::user("hello")];
        let body = provider.build_request_body("be nice", &messages, &[], false);

        let serialized = serde_json::to_string(&body).unwrap();
        assert!(
            !serialized.contains("cc_version"),
            "anthropic-messages body must not contain Claude Code billing header: {}",
            serialized
        );
        assert!(
            !serialized.contains("cc_entrypoint"),
            "anthropic-messages body must not contain Claude Code entrypoint tag: {}",
            serialized
        );
        assert!(
            !serialized.contains("cch="),
            "anthropic-messages body must not contain cch attestation placeholder: {}",
            serialized
        );
    }

    #[test]
    fn test_api_body_has_no_metadata() {
        let provider = test_provider();
        let body = provider.build_request_body("", &[Message::user("hi")], &[], false);
        assert!(
            body.get("metadata").is_none(),
            "anthropic-messages body must not include metadata.user_id"
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
            "anthropic-messages should serialize `system` as a plain string"
        );
    }

    #[test]
    fn test_api_body_omits_system_when_empty() {
        let provider = test_provider();
        let body = provider.build_request_body("", &[Message::user("hi")], &[], false);
        assert!(
            body.get("system").is_none(),
            "anthropic-messages should omit `system` when the prompt is empty"
        );
    }
}
