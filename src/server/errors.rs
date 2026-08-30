//! RFC 9457 Problem Details for HTTP APIs. Every error response from `meka serve` uses this
//! shape, with content type `application/problem+json`. Stable `type` URIs under
//! `https://meka.so/errors/` act as machine-readable error codes that survive HTTP-status
//! collisions (multiple 404 meanings, multiple 409 meanings); see the HTTP API docs for the full
//! catalogue.
//!
//! Mid-stream failures (after the SSE response has started) are emitted as an in-band
//! `turn.failed` SSE event carrying the same JSON shape; this module owns the wire type and
//! `axum` integration for the HTTP-level path.

use std::collections::BTreeMap;

use axum::{
    Json,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::Serialize;
use serde_json::Value;
use utoipa::ToSchema;

use crate::error::MekaError;

/// RFC 9457 Problem Details body. The five core members (`type`, `title`, `status`, `detail`,
/// `instance`) are first-class; meka-specific extension members ride in `extensions` and get
/// flattened into the top-level JSON object on serialization.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ProblemDetail {
    /// Stable URI identifying the error class. Always set; opaque to clients beyond exact
    /// comparison. URIs are documented (not dereferenced); clients should never fetch them.
    #[serde(rename = "type")]
    pub type_uri: String,
    /// Short, human-readable summary. Stable for a given `type_uri`.
    pub title: String,
    /// HTTP status code that accompanied this response, mirrored into the body for clients that
    /// only see the body.
    pub status: u16,
    /// Instance-specific message. May vary between occurrences of the same `type_uri`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// URI (typically a request path) identifying the specific occurrence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance: Option<String>,
    /// Extension members (e.g. `session_id`, `request_id`, `retry_after`). Serialized as
    /// top-level JSON fields via `#[serde(flatten)]`.
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub extensions: BTreeMap<String, Value>,
    /// When present, surfaced as an HTTP `Retry-After: <n>` response header (seconds). Not
    /// serialized into the body; call sites also pass the same value into the `retry_after`
    /// body extension via `.with(...)` for clients that only parse JSON.
    #[serde(skip)]
    #[schema(value_type = Option<u32>)]
    pub retry_after_seconds: Option<u32>,
}

impl ProblemDetail {
    pub fn new(error: ErrorKind, status: StatusCode, detail: impl Into<String>) -> Self {
        Self {
            type_uri: error.type_uri().to_string(),
            title: error.title().to_string(),
            status: status.as_u16(),
            detail: Some(detail.into()),
            instance: None,
            extensions: BTreeMap::new(),
            retry_after_seconds: None,
        }
    }

    /// Attach the request path as `instance` (RFC 9457's "URI reference that identifies the
    /// specific occurrence").
    #[must_use]
    pub fn instance(mut self, instance: impl Into<String>) -> Self {
        self.instance = Some(instance.into());
        self
    }

    /// Attach an extension member. Common keys: `session_id`, `turn_id`, `request_id`,
    /// `retry_after`. Caller is responsible for the value's JSON shape.
    #[must_use]
    pub fn with(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.extensions.insert(key.into(), value.into());
        self
    }

    /// Attach a `Retry-After: <seconds>` HTTP response header. The spec requires this on every
    /// 429 (concurrency-limit, idempotency-key cache cap). The same value is typically also
    /// added as the `retry_after` extension via `.with(...)` so clients reading just the JSON
    /// body can see it; most callers should use [`Self::with_retry_after`] which sets both.
    #[must_use]
    pub fn retry_after(mut self, seconds: u32) -> Self {
        self.retry_after_seconds = Some(seconds);
        self
    }

    /// Convenience: set both the `Retry-After` HTTP header *and* the `retry_after` JSON body
    /// extension. Always use this on 429 responses: calling only one of the two halves is
    /// a wire-shape bug clients can hit silently.
    #[must_use]
    pub fn with_retry_after(self, seconds: u32) -> Self {
        self.with("retry_after", Value::from(seconds))
            .retry_after(seconds)
    }

    /// Build a 500 Problem Detail whose body carries a generic message while the full error
    /// detail is logged server-side, so the wire response doesn't leak internal details.
    ///
    /// `context` is a short operator-readable description logged alongside the error.
    pub fn internal_sanitized(context: &str, error: impl std::fmt::Display) -> Self {
        tracing::error!("{}: {}", context, error);
        Self::new(
            ErrorKind::Internal,
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal server error; consult server logs",
        )
    }
}

/// The longest `Retry-After` meka will relay from an upstream, in seconds (one hour).
///
/// Not a judgement about how long a client should wait, which is the upstream's to make, but a
/// bound on what meka will repeat: `parse_retry_after` returns whatever the header said, and a
/// header saying a year is a broken or hostile upstream steering every client of this server into a
/// stall. An hour is far past any real rate-limit window and far short of that.
const RELAYED_RETRY_AFTER_CAP: u32 = 3_600;

/// Catalogue of stable error types, matching the HTTP API docs table. Each variant maps to a
/// `type` URI plus a fixed `title`. New variants land alongside new endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    Auth,
    AuthScope,
    SessionNotFound,
    /// A named resource that is not a session: a skill, a memory, an MCP server, a background
    /// task, a turn stream. Distinct from [`Self::SessionNotFound`] because `type` is the
    /// machine-readable code a client switches on, and "your session is gone" and "that skill does
    /// not exist" call for very different responses.
    NotFound,
    /// The token's scopes are sufficient, but the *session* sits at too low a permission for what
    /// was asked.
    ///
    /// Distinct from [`Self::AuthScope`], which is the same 403 with the opposite remedy. A client
    /// routing on `type` reads `auth-scope` as "get a better token" and will re-provision forever;
    /// the fix here is `PATCH /v1/sessions/{id}` with a higher `permission`.
    SessionPermission,
    SessionLocked,
    /// The session exists but is not resident in memory, and the endpoint needs live state.
    ///
    /// Distinct from [`Self::TurnInFlight`], which reads as "something is running, cancel it" and
    /// sends a client into a `POST /cancel` loop that returns 204 forever because there is no turn.
    /// The remedy here is the opposite: submit a turn, which loads the session.
    SessionNotLoaded,
    TurnInFlight,
    TurnCancelled,
    /// A re-attached stream ended with no recorded outcome, because the task that would have
    /// recorded one died. Only ever carried inside a terminal `turn.failed` SSE event, never as an
    /// HTTP response body, but catalogued here so the type URI has one definition rather than a
    /// string literal at the emitting site.
    StreamDetached,
    RequestNotFound,
    Idempotency,
    /// The named skill lives under a read-only root from `[skills] extra_paths`, so writing it
    /// here would create a shadowing copy in meka's own store rather than change the file.
    ///
    /// Distinct from [`Self::InvalidBody`]: the request is well-formed and the remedy is to pick
    /// another name or edit that file directly, not to fix the payload.
    StoreReadOnly,
    InvalidBody,
    PayloadTooLarge,
    ConcurrencyLimit,
    Provider,
    /// The conversation no longer fits the model's context window, and meka could not compact it
    /// down far enough (or `auto_compact` is off).
    ///
    /// A 502 like [`Self::Provider`], because the upstream is what refused the turn, but a distinct
    /// `type` so a client can tell the two apart. They call for opposite responses: `provider` is
    /// "the upstream is unwell, try the same request again", while this one will refuse the same
    /// request forever. A client that cannot distinguish them retries an oversized conversation
    /// until it gives up on wall-clock, which is the failure this variant exists to prevent. The
    /// remedy is to shorten the conversation: `POST /v1/sessions/{id}/compact`, or `/rewind`, or a
    /// smaller message.
    ///
    /// Not [`Self::PayloadTooLarge`], which is meka's own `max_body_bytes` on the HTTP request and
    /// has nothing to do with the model's window; a request well under one limit routinely exceeds
    /// the other.
    ContextOverflow,
    /// A server marked `[mcp.servers.<name>] required` was not connected when a turn asked for it,
    /// so the turn was refused before anything reached the provider.
    ///
    /// A 503 rather than a 502: the dependency that is unwell is one meka manages rather than the
    /// model provider, and the caller's request was never forwarded to anything. Distinct from
    /// [`Self::Internal`], which is where this used to land, and which reads as "meka broke"
    /// and sends an operator to the wrong log; meka classified this one correctly and the fault
    /// is in a subprocess or a remote MCP server.
    McpUnavailable,
    Internal,
}

impl ErrorKind {
    pub const fn type_uri(self) -> &'static str {
        match self {
            Self::Auth => "https://meka.so/errors/auth",
            Self::AuthScope => "https://meka.so/errors/auth-scope",
            Self::SessionNotFound => "https://meka.so/errors/session-not-found",
            Self::NotFound => "https://meka.so/errors/not-found",
            Self::SessionPermission => "https://meka.so/errors/session-permission",
            Self::SessionLocked => "https://meka.so/errors/session-locked",
            Self::SessionNotLoaded => "https://meka.so/errors/session-not-loaded",
            Self::TurnInFlight => "https://meka.so/errors/turn-in-flight",
            Self::TurnCancelled => "https://meka.so/errors/turn-cancelled",
            Self::StreamDetached => "https://meka.so/errors/stream-detached",
            Self::RequestNotFound => "https://meka.so/errors/request-not-found",
            Self::Idempotency => "https://meka.so/errors/idempotency",
            Self::StoreReadOnly => "https://meka.so/errors/store-read-only",
            Self::InvalidBody => "https://meka.so/errors/invalid-body",
            Self::PayloadTooLarge => "https://meka.so/errors/payload-too-large",
            Self::ConcurrencyLimit => "https://meka.so/errors/concurrency-limit",
            Self::Provider => "https://meka.so/errors/provider",
            Self::ContextOverflow => "https://meka.so/errors/context-overflow",
            Self::McpUnavailable => "https://meka.so/errors/mcp-unavailable",
            Self::Internal => "https://meka.so/errors/internal",
        }
    }

    pub const fn title(self) -> &'static str {
        match self {
            Self::Auth => "Authentication failed",
            Self::AuthScope => "Insufficient scope",
            Self::SessionNotFound => "Session not found",
            Self::NotFound => "Resource not found",
            Self::SessionPermission => "Session permission too low",
            Self::SessionLocked => "Session is locked by another process",
            Self::SessionNotLoaded => "Session is not loaded",
            Self::TurnInFlight => "Turn already in flight",
            Self::TurnCancelled => "Turn cancelled",
            Self::StreamDetached => "Turn outcome unavailable",
            Self::RequestNotFound => "Pending request not found",
            Self::Idempotency => "Idempotency-Key conflict",
            Self::StoreReadOnly => "Skill is in a read-only root",
            Self::InvalidBody => "Invalid request body",
            Self::PayloadTooLarge => "Request body exceeds configured limit",
            Self::ConcurrencyLimit => "Process-wide concurrency limit reached",
            Self::Provider => "Provider call failed",
            Self::ContextOverflow => "Conversation exceeds the model's context window",
            Self::McpUnavailable => "Required MCP server is not ready",
            Self::Internal => "Internal server error",
        }
    }
}

impl IntoResponse for ProblemDetail {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let retry_after = self.retry_after_seconds;
        let mut response = (status, Json(&self)).into_response();
        // RFC 9457 mandates `application/problem+json` instead of axum's default
        // `application/json`.
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/problem+json"),
        );
        if let Some(seconds) = retry_after
            && let Ok(value) = header::HeaderValue::from_str(&seconds.to_string())
        {
            response.headers_mut().insert(header::RETRY_AFTER, value);
        }
        // RFC 9110 §15.5.2: 401 responses MUST include WWW-Authenticate.
        if status == StatusCode::UNAUTHORIZED {
            response.headers_mut().insert(
                header::WWW_AUTHENTICATE,
                header::HeaderValue::from_static(r#"Bearer realm="meka""#),
            );
        }
        response
    }
}

/// Best-effort mapping from internal `MekaError` to a Problem Detail. Used by handlers that
/// propagate agent-layer errors back to the client. Variants without a dedicated HTTP shape
/// land on `internal` (500). Refine on demand as new error paths surface.
impl From<&MekaError> for ProblemDetail {
    fn from(error: &MekaError) -> Self {
        match error {
            MekaError::Config(message) => ProblemDetail::new(
                ErrorKind::InvalidBody,
                StatusCode::UNPROCESSABLE_ENTITY,
                message.clone(),
            ),
            // Every one of these is an upstream failure rather than anything the HTTP caller got
            // wrong, and what is left to say is that the upstream would not serve this turn, which
            // is what 502 means.
            //
            // Not *because* the agent loop exhausted its retries first, which is only sometimes
            // true: `should_retry_provider_error` gives up immediately once any output has reached
            // the frontend, and `compact_session`'s summariser calls `complete` with no retry loop
            // at all. So one of these can arrive after four attempts or after one. The mapping does
            // not depend on which, and saying it did would be inventing a guarantee.
            //
            // `RetryableProvider`, `StreamError` and `ContextOverflow` belong here rather than
            // merely filling the match. All three used to fall through to the `other` arm, which
            // answers 500 and logs at `error!` as an unhandled internal fault, sending an operator
            // to look in the wrong process for a failure meka had classified correctly.
            //
            // Logged rather than relayed. `MekaError::Provider` can carry the upstream's verbatim
            // response body, which is not meka's to publish to an HTTP caller: it has held an
            // account identifier, a rate-limit posture, and on one backend a fragment of the
            // request that triggered it.
            //
            // A length bound was the first attempt and did not work, for a reason worth recording:
            // it keeps the *start* of the body, and every one of those lives at the start of a JSON
            // error object. Most provider errors are also shorter than any sensible bound, so the
            // common case was relayed whole and the cut fired only on the long ones. The policy
            // here is the one `webhook.rs` already states for outbound deliveries -- identifiers
            // and status travel, content does not -- and the log is where an operator reads the
            // rest.
            MekaError::Provider(message)
            | MekaError::InvalidRequest(message)
            | MekaError::StreamError(message) => {
                tracing::warn!("provider error: {}", message);
                ProblemDetail::new(
                    ErrorKind::Provider,
                    StatusCode::BAD_GATEWAY,
                    "the provider rejected or failed this turn; its response is in the server log",
                )
            }
            // Its own arm only to keep the upstream's `Retry-After`, which the neighbours have
            // nothing equivalent to. Same status and same type: a client switching on those cannot
            // tell this from the arm above, and should not need to.
            //
            // The hint is worth relaying because it is fresh rather than spent. It is read from the
            // headers of the *final* attempt, and the agent loop gives up rather than sleeping
            // again, so nothing has elapsed against it by the time it arrives here. Dropping it
            // leaves a client backing off blind against a server that was told the number.
            //
            // Clamped, because `parse_retry_after` relays whatever the header said and a broken or
            // hostile upstream can say a year. `u32` seconds is also what `ProblemDetail` carries,
            // so an unclamped `u64` would wrap rather than saturate.
            MekaError::RetryableProvider {
                message,
                retry_after,
                ..
            } => {
                tracing::warn!("provider error: {}", message);
                let problem = ProblemDetail::new(
                    ErrorKind::Provider,
                    StatusCode::BAD_GATEWAY,
                    "the provider rejected or failed this turn; its response is in the server log",
                );
                match retry_after {
                    Some(delay) => problem.with_retry_after(
                        delay.as_secs().min(RELAYED_RETRY_AFTER_CAP as u64) as u32,
                    ),
                    None => problem,
                }
            }
            // The same 502 as its neighbours, and for the same reason: the upstream refused the
            // turn. Its own `type` because the remedy is the opposite one. `provider` invites the
            // client to send the same request again, which is right for a 529 and wrong here: the
            // conversation is too long and will be too long next time. Sharing the type left a
            // correct client retrying forever, so the split is the point rather than tidiness.
            //
            // Not a 413. That status is spoken for by `max_body_bytes`, which is meka's own limit
            // on the HTTP request rather than the model's window, so reusing it would make one
            // status mean two unrelated things. Sharing 502 with the arm above is the opposite
            // arrangement and the one this module is built on: distinct types over a shared status,
            // exactly as the module docs describe for the several 404s and 409s.
            //
            // Reaching here at all means the agent loop could not compact its way out:
            // `auto_compact` is off, or its retries are spent, or there was only one message and
            // nothing to drop.
            MekaError::ContextOverflow(message) => {
                tracing::warn!("context overflow: {}", message);
                ProblemDetail::new(
                    ErrorKind::ContextOverflow,
                    StatusCode::BAD_GATEWAY,
                    "the conversation exceeds the model's context window and could not be \
                     compacted further; shorten it before retrying",
                )
            }
            // The last variant that was falling into the `other` arm with a fault of its own, and
            // the one where 500 read worst: a required MCP server being down is a clean, fully
            // classified pre-flight refusal, and answering "internal server error; consult server
            // logs" sends an operator looking for a bug in meka instead of at the subprocess that
            // did not start.
            //
            // The names travel and the reasons do not, which is the policy the provider arms above
            // state. A reason here is the connector's own text and has carried a spawn failure
            // complete with the command line and its path; the names are the operator's own
            // configuration and the only part a caller can act on. `servers` rides as an extension
            // so a client can branch on which one rather than parse the sentence.
            MekaError::McpTurnGated { servers } => {
                tracing::warn!("mcp gate refused a turn: {}", error);
                let names: Vec<&str> = servers.iter().map(|(name, _)| name.as_str()).collect();
                ProblemDetail::new(
                    ErrorKind::McpUnavailable,
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!(
                        "required MCP server(s) not ready: {}; each server's reason is in the \
                         server log",
                        names.join(", ")
                    ),
                )
                .with("servers", Value::from(names))
            }
            MekaError::Interrupted => ProblemDetail::new(
                ErrorKind::TurnCancelled,
                StatusCode::CONFLICT,
                "turn was cancelled (client cancel, shutdown, or disconnect)",
            ),
            MekaError::SessionLocked(id) => ProblemDetail::new(
                ErrorKind::SessionLocked,
                StatusCode::CONFLICT,
                format!("session {} is locked by another process", id),
            )
            .with("session_id", id.to_string()),
            other => {
                tracing::error!("unhandled agent error mapped to 500: {}", other);
                ProblemDetail::new(
                    ErrorKind::Internal,
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal server error; consult server logs",
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn problem_detail_serializes_extensions_at_top_level() {
        let problem = ProblemDetail::new(
            ErrorKind::SessionNotFound,
            StatusCode::NOT_FOUND,
            "session 's_abc' does not exist",
        )
        .instance("/v1/sessions/s_abc/turn")
        .with("session_id", "s_abc");
        let body = serde_json::to_value(&problem).expect("serializable");
        assert_eq!(
            body["type"], "https://meka.so/errors/session-not-found",
            "type URI must match the catalogue entry exactly",
        );
        assert_eq!(body["status"], 404);
        assert_eq!(body["instance"], "/v1/sessions/s_abc/turn");
        assert_eq!(
            body["session_id"], "s_abc",
            "extension members flatten to the top level"
        );
    }

    #[test]
    fn meka_error_provider_maps_to_502() {
        let error = MekaError::Provider("upstream 529".into());
        let problem = ProblemDetail::from(&error);
        assert_eq!(problem.status, 502);
        assert_eq!(problem.type_uri, "https://meka.so/errors/provider");
    }

    /// A 502 must not carry the provider's own response text.
    ///
    /// That body has held an account identifier, a rate-limit posture and a fragment of the
    /// request; whoever holds a `sessions:w` token is not necessarily whoever holds the provider
    /// account. Truncating it was the first attempt and kept the start, which is where all three
    /// live in a JSON error object.
    #[test]
    fn a_provider_failure_does_not_relay_the_upstream_body() {
        let leaky = "{\"error\":{\"account_uuid\":\"acct-0f3c\",\"type\":\"rate_limit_error\",\
                     \"message\":\"organization has exceeded its quota\"}}";
        for error in [
            MekaError::Provider(leaky.to_string()),
            MekaError::InvalidRequest(leaky.to_string()),
            MekaError::RetryableProvider {
                message: leaky.to_string(),
                retry_after: None,
                server_error_on_completion: false,
            },
            MekaError::StreamError(leaky.to_string()),
        ] {
            let problem = ProblemDetail::from(&error);
            // Load-bearing, and not decoration around the redaction checks. Without it the variants
            // added later pass this test even with their arm deleted: the catch-all's detail is
            // "internal server error; consult server logs", which contains neither secret and does
            // contain "server log" as a substring of "server logs".
            assert_eq!(problem.status, 502, "{error}");
            let detail = problem.detail.unwrap_or_default();
            assert!(
                !detail.contains("acct-0f3c"),
                "the upstream body reached the caller: {detail}",
            );
            assert!(
                !detail.contains("exceeded its quota"),
                "the upstream body reached the caller: {detail}",
            );
            assert!(
                detail.contains("server log"),
                "and the caller must be told where the detail went: {detail}",
            );
        }
    }

    /// A rejection the *upstream* issued is an upstream failure (502), not a complaint about the
    /// caller's own body (4xx). The name says "invalid request" and the status deliberately does
    /// not agree with it: what was invalid is the conversation meka assembled, which the caller
    /// neither sent nor can fix by correcting its own payload. Whether the agent loop tried to
    /// repair it first is not part of the mapping: `/compact`'s summariser has no repair path at
    /// all, so an `InvalidRequest` from there arrives having been tried exactly once.
    #[test]
    fn meka_error_invalid_request_maps_to_502() {
        let error = MekaError::InvalidRequest("400 invalid_request_error".into());
        let problem = ProblemDetail::from(&error);
        assert_eq!(problem.status, 502);
        assert_eq!(problem.type_uri, "https://meka.so/errors/provider");
    }

    /// An upstream failure reaches the caller as one rather than as an internal fault.
    ///
    /// Both used to fall through to the catch-all, so exhausting the retries on a 429 answered 500
    /// and logged itself as an unhandled internal fault, sending an operator to look in the wrong
    /// process for a failure meka had classified correctly.
    ///
    /// The type URI is asserted alongside the status because the two travel together: one error
    /// type answering with two different statuses is what a client keying on the type cannot
    /// handle.
    #[test]
    fn an_exhausted_upstream_failure_maps_to_502() {
        for error in [
            MekaError::RetryableProvider {
                message: "529 overloaded, four attempts".into(),
                retry_after: None,
                server_error_on_completion: false,
            },
            MekaError::StreamError("connection closed mid-stream".into()),
        ] {
            let problem = ProblemDetail::from(&error);
            assert_eq!(problem.status, 502, "{error}");
            assert_eq!(
                problem.type_uri, "https://meka.so/errors/provider",
                "{error}"
            );
        }
    }

    /// A required MCP server that never came up answers 503 under its own type, and its reasons
    /// stay in the log.
    ///
    /// Both halves matter. This variant used to fall through to the `other` arm, so a subprocess
    /// that failed to start was reported as `/errors/internal` 500 and logged as an unhandled
    /// internal fault, which is the one classification that sends an operator looking in the wrong
    /// process. And a reason string is the connector's own text: it has carried a spawn failure
    /// complete with the command line and its path, which is the same argument that keeps a
    /// provider's response body out of the arm above.
    #[test]
    fn a_required_mcp_server_that_is_down_is_503_under_its_own_type() {
        let error = MekaError::McpTurnGated {
            servers: vec![
                (
                    "ida".to_string(),
                    "failed to spawn process: /opt/private/ida-mcp: No such file".to_string(),
                ),
                ("exa".to_string(), "handshake timed out".to_string()),
            ],
        };
        let problem = ProblemDetail::from(&error);

        assert_eq!(problem.status, 503);
        assert_eq!(problem.type_uri, "https://meka.so/errors/mcp-unavailable");
        let detail = problem.detail.clone().expect("a detail naming the servers");
        assert!(detail.contains("ida") && detail.contains("exa"), "{detail}");
        assert_eq!(
            problem.extensions.get("servers"),
            Some(&serde_json::json!(["ida", "exa"])),
            "a client should branch on the names without parsing the sentence"
        );

        let body = serde_json::to_string(&problem).expect("serialize the problem");
        assert!(!body.contains("/opt/private"), "{body}");
        assert!(!body.contains("handshake timed out"), "{body}");
    }

    /// The upstream's own `Retry-After` reaches the caller, clamped.
    ///
    /// Both halves, because `ProblemDetail` carries the header and the body extension separately
    /// and setting one without the other is a wire-shape bug a client hits silently. The clamp is
    /// not cosmetic: `parse_retry_after` relays whatever the header said, `Duration::as_secs` is a
    /// `u64`, and the field is a `u32`, so an upstream saying a year would wrap to a small number
    /// rather than saturate if the cast were unguarded.
    #[test]
    fn a_retryable_failure_relays_the_upstream_retry_after() {
        let problem = ProblemDetail::from(&MekaError::RetryableProvider {
            message: "429 rate limited".into(),
            retry_after: Some(std::time::Duration::from_secs(30)),
            server_error_on_completion: false,
        });
        assert_eq!(problem.retry_after_seconds, Some(30));
        assert_eq!(
            problem.extensions.get("retry_after"),
            Some(&Value::from(30))
        );

        let absurd = ProblemDetail::from(&MekaError::RetryableProvider {
            message: "529 overloaded".into(),
            retry_after: Some(std::time::Duration::from_secs(31_536_000)),
            server_error_on_completion: false,
        });
        assert_eq!(
            absurd.retry_after_seconds,
            Some(RELAYED_RETRY_AFTER_CAP),
            "a header saying a year must be clamped, not relayed and not wrapped"
        );

        let silent = ProblemDetail::from(&MekaError::RetryableProvider {
            message: "connection reset".into(),
            retry_after: None,
            server_error_on_completion: false,
        });
        assert_eq!(
            silent.retry_after_seconds, None,
            "a transport failure has no header to relay, so meka must not invent one"
        );
    }

    /// A conversation that will not fit is 502 like its neighbours but says so under its own type.
    ///
    /// The status is shared because the upstream is what refused the turn. The type is not, because
    /// the remedies are opposites: `/errors/provider` means "the upstream is unwell, send it
    /// again", and a client reading this as that retries an oversized conversation until it gives
    /// up on wall-clock. Nothing else on the wire distinguishes them, so if this assertion is ever
    /// relaxed to make a match arm simpler, that loop comes back.
    #[test]
    fn a_context_overflow_is_502_under_its_own_type() {
        let problem = ProblemDetail::from(&MekaError::ContextOverflow(
            "API returned status 400: {\"error\":{\"account_uuid\":\"acct-0f3c\",\"message\":\
             \"prompt is too long: 250000 tokens > 200000 maximum\"}}"
                .into(),
        ));
        assert_eq!(problem.status, 502);
        assert_eq!(
            problem.type_uri, "https://meka.so/errors/context-overflow",
            "sharing `/errors/provider` sends a correct client into a retry loop"
        );
        // The title is a wire field too, and RFC 9457 asks it to be stable for a given type, so a
        // client may render it verbatim. Asserting one is also what stops `ErrorKind::title` being
        // gutted wholesale without a test noticing.
        assert_eq!(
            problem.title,
            "Conversation exceeds the model's context window"
        );
        let detail = problem.detail.unwrap_or_default();
        assert!(
            detail.contains("shorten it before retrying"),
            "the detail has to name the remedy, since the type alone is opaque: {detail}"
        );
        // Built from a provider response like every other arm, so it carries that body and must not
        // relay it. Unlike the 502 arm it does not point at the log, because what the caller needs
        // to know is stated outright rather than withheld -- which is why this lives here rather
        // than in `a_provider_failure_does_not_relay_the_upstream_body`, whose loop asserts that
        // pointer for every entry.
        assert!(
            !detail.contains("250000"),
            "the upstream body reached the caller: {detail}"
        );
    }

    #[test]
    fn meka_error_session_locked_carries_session_id() {
        let id = uuid::Uuid::nil();
        let problem = ProblemDetail::from(&MekaError::SessionLocked(id));
        assert_eq!(problem.status, 409);
        assert_eq!(problem.type_uri, "https://meka.so/errors/session-locked");
        assert_eq!(
            problem.extensions.get("session_id"),
            Some(&Value::String(id.to_string()))
        );
    }
}
