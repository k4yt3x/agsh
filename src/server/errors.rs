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
    #[schema(value_type = Option<u64>)]
    pub retry_after_seconds: Option<u64>,
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
    pub fn retry_after(mut self, seconds: u64) -> Self {
        self.retry_after_seconds = Some(seconds);
        self
    }

    /// Convenience: set both the `Retry-After` HTTP header *and* the `retry_after` JSON body
    /// extension. Always use this on 429 responses: calling only one of the two halves is
    /// a wire-shape bug clients can hit silently.
    #[must_use]
    pub fn with_retry_after(self, seconds: u64) -> Self {
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
    InvalidBody,
    PayloadTooLarge,
    ConcurrencyLimit,
    Provider,
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
            Self::InvalidBody => "https://meka.so/errors/invalid-body",
            Self::PayloadTooLarge => "https://meka.so/errors/payload-too-large",
            Self::ConcurrencyLimit => "https://meka.so/errors/concurrency-limit",
            Self::Provider => "https://meka.so/errors/provider",
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
            Self::InvalidBody => "Invalid request body",
            Self::PayloadTooLarge => "Request body exceeds configured limit",
            Self::ConcurrencyLimit => "Process-wide concurrency limit reached",
            Self::Provider => "Provider call failed",
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
            // Both are upstream refusals rather than anything the HTTP caller got wrong: by the
            // time a request is malformed enough for the provider to reject it, the agent loop has
            // already tried to repair it.
            MekaError::Provider(message) | MekaError::InvalidRequest(message) => {
                ProblemDetail::new(
                    ErrorKind::Provider,
                    StatusCode::BAD_GATEWAY,
                    message.clone(),
                )
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

    /// A malformed-request rejection reaches the HTTP caller only after the agent loop has already
    /// tried to repair it, so it is an upstream failure (502) rather than a complaint about the
    /// caller's own body (4xx).
    #[test]
    fn meka_error_invalid_request_maps_to_502() {
        let error = MekaError::InvalidRequest("400 invalid_request_error".into());
        let problem = ProblemDetail::from(&error);
        assert_eq!(problem.status, 502);
        assert_eq!(problem.type_uri, "https://meka.so/errors/provider");
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
