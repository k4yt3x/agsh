//! Crate-wide [`MekaError`] enum and [`Result`] alias. All non-binary code paths return `Result<T,
//! MekaError>`; the `main` binary wraps these in `anyhow::Result` for top-level reporting.

use std::time::Duration;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum MekaError {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("database error: {0}")]
    Database(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("provider error: {0}")]
    Provider(String),

    /// The provider rejected the request because the prompt exceeded the model's context window
    /// (e.g. HTTP 400 "prompt is too long" / 413 / `context_length_exceeded`). Distinct from
    /// [`Self::Provider`] so the agent loop can catch it by type and compact-and-retry once instead
    /// of matching error strings at the call site.
    #[error("context window exceeded: {0}")]
    ContextOverflow(String),

    /// The provider rejected the request as malformed (HTTP 400 / 422 that isn't an overflow).
    /// Deterministic on the request body, so retrying it unchanged is pointless; distinct from
    /// [`Self::Provider`] so the agent loop can instead degrade the content it most recently
    /// appended and retry once (see `Agent::run_turn`). Without that path a single rejected block
    /// is permanent: it is already committed to the session, so every later request carries it and
    /// fails the same way.
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// A transient provider failure (HTTP 429, any 5xx including Anthropic's 529 "overloaded", or a
    /// mid-stream `overloaded_error`/`rate_limit_error`/`api_error` SSE event) that is safe to
    /// retry with backoff. Distinct from [`Self::Provider`] so the agent loop can retry by type
    /// instead of matching error strings. Carries the provider's `Retry-After` hint when
    /// present.
    #[error("provider temporarily unavailable: {message}")]
    RetryableProvider {
        message: String,
        retry_after: Option<Duration>,
    },

    #[error("tool execution error: {tool_name}: {message}")]
    ToolExecution { tool_name: String, message: String },

    #[error("tool registration error: {message}")]
    ToolRegistration { message: String },

    #[error("session already attached by another process: {0}")]
    SessionLocked(uuid::Uuid),

    #[error("agent interrupted by user")]
    Interrupted,

    /// A logic invariant in meka itself was violated. Used in place of `.expect()` for cases where
    /// a bug in our own code (not user input or I/O) is the only path to the error.
    #[error("internal error: {0}")]
    Internal(String),

    #[error("SSE stream error: {0}")]
    StreamError(String),

    #[error("MCP connection error: {server_name}: {message}")]
    McpConnection {
        server_name: String,
        message: String,
    },

    #[error("MCP tool error: {server_name}: {tool_name}: {message}")]
    McpToolExecution {
        server_name: String,
        tool_name: String,
        message: String,
    },

    #[error("MCP authentication error: {server_name}: {message}")]
    McpAuth {
        server_name: String,
        message: String,
    },

    /// MCP readiness gate rejected the turn: at least one server marked
    /// [`crate::config::McpServerConfig::required`] wasn't `Connected` within the configured grace
    /// period. Servers that aren't required never appear here. Turn contents haven't been sent to
    /// the provider. The REPL catches this and loops back to the prompt; one-shot mode propagates
    /// to a non-zero process exit.
    #[error("mcp: {} server(s) not ready: {}", .servers.len(), .servers.iter().map(|(n, s)| format!("{} ({})", n, s)).collect::<Vec<_>>().join(", "))]
    McpTurnGated { servers: Vec<(String, String)> },
}

pub type Result<T> = std::result::Result<T, MekaError>;

/// Classify a provider HTTP failure response: map context-window overflows to
/// [`MekaError::ContextOverflow`] (so the agent loop can compact-and-retry once), transient
/// failures (429, any 5xx including Anthropic's 529 "overloaded") to
/// [`MekaError::RetryableProvider`] (so the agent loop can retry with backoff), malformed requests
/// (400 / 422) to [`MekaError::InvalidRequest`] (so the agent loop can degrade-and-retry once), and
/// everything else to [`MekaError::Provider`]. Anthropic returns HTTP 400 `invalid_request_error`
/// with "prompt is too long"; OpenAI returns 400 `context_length_exceeded` (or 413). The overflow
/// check is matched on the body (a bare 400 is shared with many unrelated errors) and takes
/// priority over the status code so it can't be shadowed by an unrelated retryable status.
///
/// The 400 / 422 bucket deliberately makes no attempt to tell a content problem from a parameter
/// problem: a `max_tokens` above the model's ceiling, an unknown beta header and a mislabelled
/// image all arrive in the same shape. The agent loop restores what it degraded when the retry also
/// fails, so classifying too broadly here costs one round trip and destroys nothing.
pub(crate) fn provider_http_error(
    status: reqwest::StatusCode,
    body: &str,
    retry_after: Option<Duration>,
) -> MekaError {
    let lower = body.to_ascii_lowercase();
    let overflow = status == reqwest::StatusCode::PAYLOAD_TOO_LARGE
        || lower.contains("prompt is too long")
        || lower.contains("context_length_exceeded")
        || lower.contains("context length exceeded")
        || lower.contains("maximum context length")
        || lower.contains("exceeds the maximum context");
    let message = format!("API returned status {status}: {body}");
    if overflow {
        MekaError::ContextOverflow(message)
    } else if status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
        MekaError::RetryableProvider {
            message,
            retry_after,
        }
    } else if status == reqwest::StatusCode::BAD_REQUEST
        || status == reqwest::StatusCode::UNPROCESSABLE_ENTITY
    {
        MekaError::InvalidRequest(message)
    } else {
        MekaError::Provider(message)
    }
}

/// Parse the `Retry-After` response header as a whole number of seconds. Only the delta-seconds
/// form is handled (what every provider we talk to actually sends); the less common HTTP-date form
/// is ignored (returns `None`, falling back to computed backoff) rather than pulling in a date
/// parser for a form we've never observed in practice.
pub(crate) fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}

/// Read a caught panic's payload as text.
///
/// `catch_unwind` hands back a `Box<dyn Any>` whose only useful shapes are the two the standard
/// panic machinery boxes: a `&'static str` for a literal message and a `String` for a formatted
/// one. Everything else is a payload from a hand-rolled `panic_any`, which nothing here does.
///
/// Shared because every supervised loop needs the same three lines, and a loop that catches a panic
/// and then cannot say what it was is barely better than one that dies.
pub(crate) fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|text| (*text).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string())
}

/// Format a [`reqwest::Error`] together with its full source chain.
///
/// reqwest's outer Display string ("error sending request for url …") usually hides the actual
/// cause (TCP reset, HTTP/2 GOAWAY, TLS handshake failure, connect timeout, DNS resolution failure,
/// …). Walking [`std::error::Error::source`] surfaces the underlying reason inline, so users (and
/// bug reports) see what actually broke instead of reqwest's generic wrapper.
///
/// Used at every site that wraps a `reqwest::Error` in an `MekaError` via Display formatting.
pub(crate) fn format_reqwest_error(error: &reqwest::Error) -> String {
    use std::error::Error as _;
    let mut out = error.to_string();
    let mut source: Option<&dyn std::error::Error> = error.source();
    while let Some(cause) = source {
        out.push_str(": ");
        out.push_str(&cause.to_string());
        source = cause.source();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_http_error_maps_overflow() {
        // Anthropic: 400 invalid_request_error / "prompt is too long".
        assert!(matches!(
            provider_http_error(
                reqwest::StatusCode::BAD_REQUEST,
                r#"{"error":{"type":"invalid_request_error","message":"prompt is too long: 250000 tokens > 200000 maximum"}}"#,
                None,
            ),
            MekaError::ContextOverflow(_)
        ));
        // OpenAI: context_length_exceeded.
        assert!(matches!(
            provider_http_error(
                reqwest::StatusCode::BAD_REQUEST,
                r#"{"error":{"code":"context_length_exceeded"}}"#,
                None,
            ),
            MekaError::ContextOverflow(_)
        ));
        // 413 Payload Too Large is an overflow regardless of body.
        assert!(matches!(
            provider_http_error(
                reqwest::StatusCode::PAYLOAD_TOO_LARGE,
                "Request Entity Too Large",
                None,
            ),
            MekaError::ContextOverflow(_)
        ));
    }

    #[test]
    fn test_provider_http_error_maps_other_as_provider() {
        // Nothing the agent loop can repair: the credentials or the endpoint are wrong.
        assert!(matches!(
            provider_http_error(
                reqwest::StatusCode::UNAUTHORIZED,
                r#"{"error":{"type":"authentication_error"}}"#,
                None,
            ),
            MekaError::Provider(_)
        ));
        assert!(matches!(
            provider_http_error(reqwest::StatusCode::NOT_FOUND, "not found", None),
            MekaError::Provider(_)
        ));
    }

    #[test]
    fn test_provider_http_error_maps_malformed_request() {
        assert!(matches!(
            provider_http_error(
                reqwest::StatusCode::BAD_REQUEST,
                r#"{"error":{"type":"invalid_request_error","message":"messages.34.content.0.tool_result.content.1.image.source.base64: The image was specified using the image/png media type, but the image appears to be a image/jpeg image"}}"#,
                None,
            ),
            MekaError::InvalidRequest(_)
        ));
        assert!(matches!(
            provider_http_error(reqwest::StatusCode::UNPROCESSABLE_ENTITY, "invalid", None),
            MekaError::InvalidRequest(_)
        ));
    }

    /// An overflow also arrives as a 400 `invalid_request_error`, and compacting is the right
    /// response to it rather than degrading content.
    #[test]
    fn test_provider_http_error_overflow_takes_priority_over_invalid_request() {
        assert!(matches!(
            provider_http_error(
                reqwest::StatusCode::BAD_REQUEST,
                r#"{"error":{"type":"invalid_request_error","message":"prompt is too long"}}"#,
                None,
            ),
            MekaError::ContextOverflow(_)
        ));
    }

    #[test]
    fn test_provider_http_error_maps_retryable() {
        // 429 rate limit.
        assert!(matches!(
            provider_http_error(
                reqwest::StatusCode::TOO_MANY_REQUESTS,
                "rate limit exceeded",
                None,
            ),
            MekaError::RetryableProvider { .. }
        ));
        // 5xx server errors, including Anthropic's non-standard 529 "overloaded".
        for status in [500u16, 502, 503, 504, 529] {
            let status = reqwest::StatusCode::from_u16(status).expect("valid status");
            assert!(
                matches!(
                    provider_http_error(status, "Overloaded", None),
                    MekaError::RetryableProvider { .. }
                ),
                "status {status} should be retryable"
            );
        }
    }

    #[test]
    fn test_provider_http_error_retryable_carries_retry_after() {
        let delay = Duration::from_secs(7);
        match provider_http_error(
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            "rate limited",
            Some(delay),
        ) {
            MekaError::RetryableProvider { retry_after, .. } => {
                assert_eq!(retry_after, Some(delay));
            }
            other => panic!("expected RetryableProvider, got {other:?}"),
        }
    }

    #[test]
    fn test_provider_http_error_overflow_takes_priority_over_retryable_status() {
        // A 500 whose body happens to mention the context window should still classify as
        // overflow, not retryable — the body check is checked first regardless of status.
        assert!(matches!(
            provider_http_error(
                reqwest::StatusCode::INTERNAL_SERVER_ERROR,
                "context_length_exceeded",
                None,
            ),
            MekaError::ContextOverflow(_)
        ));
    }

    #[test]
    fn test_parse_retry_after_present_integer_seconds() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "30".parse().unwrap());
        assert_eq!(parse_retry_after(&headers), Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_parse_retry_after_absent() {
        let headers = reqwest::header::HeaderMap::new();
        assert_eq!(parse_retry_after(&headers), None);
    }

    #[test]
    fn test_parse_retry_after_ignores_http_date_form() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::RETRY_AFTER,
            "Wed, 21 Oct 2026 07:28:00 GMT".parse().unwrap(),
        );
        assert_eq!(parse_retry_after(&headers), None);
    }

    #[test]
    fn test_parse_retry_after_ignores_malformed_value() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::RETRY_AFTER,
            "not-a-number".parse().unwrap(),
        );
        assert_eq!(parse_retry_after(&headers), None);
    }
}
