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

/// Shared by the two [`ErrorKind::ProviderUnavailable`] arms, which differ only in whether there is
/// a `Retry-After` to relay. Named rather than written twice because a caller matching on `type`
/// gets one answer and a human reading `detail` must not get two.
const PROVIDER_UNAVAILABLE_DETAIL: &str =
    "the provider did not complete this turn; its response is in the server log";

/// The most of an upstream's response `provider_response` will repeat, in bytes.
///
/// Not a redaction measure. The arm below records why a length bound failed as one: it keeps the
/// start, which is where an identifier sits. Keeping the start is right for *this* purpose, since
/// that is also where the error type sits.
///
/// Bounded at all because the text is attacker-influenced and repeated to more than one reader. It
/// arrives from `response.text()` with no cap of its own, it is copied into every 502 body, and the
/// terminal `turn.failed` event carrying it is retained in the per-turn replay ring for reconnects.
/// A `base_url` is user-supplied, so a misconfigured endpoint answering a multi-megabyte error page
/// should cost a truncated string rather than a copy per reader. Same reasoning as
/// [`RELAYED_RETRY_AFTER_CAP`] above.
///
/// Far above any real provider error, which run to a few hundred bytes of JSON.
const RELAYED_BODY_CAP: usize = 4 * 1024;

/// Truncation marker, charged against [`RELAYED_BODY_CAP`] rather than added on top of it.
const TRUNCATION_MARKER: &str = "… (truncated; full text in the server log)";

/// The two constants above are subtracted from each other, so their relationship is load-bearing
/// rather than incidental. Asserted at compile time because the failure is otherwise a runtime
/// panic inside a handler, on attacker-influenced input: the subtraction underflows, and a wrapped
/// `usize` then indexes far past the end of the string.
const _: () = assert!(
    RELAYED_BODY_CAP > TRUNCATION_MARKER.len() + 4,
    "RELAYED_BODY_CAP must exceed the marker by at least one UTF-8 character, or \
     `bounded_upstream_body` underflows its budget"
);

/// Repeat at most [`RELAYED_BODY_CAP`] bytes of `message`, marking the cut when one happens.
///
/// The result never exceeds the cap and is never longer than the input, because the marker comes
/// out of the budget instead of being appended to it. Added on top, a body a few bytes over the cap
/// came back larger than it went in, while telling the reader it had been shortened.
///
/// Cuts on a character boundary. Multi-byte text is ordinary here -- a provider error in Japanese,
/// an emoji in a proxy's HTML page -- and slicing mid-codepoint panics.
fn bounded_upstream_body(message: &str) -> String {
    if message.len() <= RELAYED_BODY_CAP {
        return message.to_string();
    }
    let mut end = RELAYED_BODY_CAP - TRUNCATION_MARKER.len();
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}{}", &message[..end], TRUNCATION_MARKER)
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
    /// The named skill lives under a read-only root from `[skills] extra_paths`, so writing it
    /// here would create a shadowing copy in meka's own store rather than change the file.
    ///
    /// Distinct from [`Self::InvalidBody`]: the request is well-formed and the remedy is to pick
    /// another name or edit that file directly, not to fix the payload.
    StoreReadOnly,
    /// The id names a sub-agent's conversation, which only its parent can continue.
    ///
    /// Distinct from [`Self::InvalidBody`] for the reason [`Self::StoreReadOnly`] is: the request
    /// parsed and validated, and no edit to it will ever be accepted. A client routing on `type`
    /// reads `invalid-body` as "my payload is malformed" and rewrites the payload forever, when the
    /// remedy is to stop addressing this id and call `agent_followup` from the parent instead.
    ///
    /// Not [`Self::SessionPermission`], which is about the level a session runs at and is raised
    /// with `PATCH /v1/sessions/{id}`; no permission makes a worker drivable from outside its
    /// parent. Not [`Self::SessionNotFound`] either: the session exists and every read endpoint
    /// still serves it.
    SessionNotDrivable,
    InvalidBody,
    PayloadTooLarge,
    ConcurrencyLimit,
    /// An upstream call failed for a reason meka could not place in one of the classes below.
    ///
    /// **A catch-all, not a "permanent" bucket, and the difference is worth stating because the
    /// obvious reading of the name is the wrong one.** It is the `else` arm of
    /// [`crate::error::provider_http_error`] plus every site that builds a bare
    /// [`crate::error::MekaError::Provider`], so a revoked credential lands here, and so do a 408,
    /// a 425, a 200 whose body was replaced by a proxy's HTML, and any mid-stream error type
    /// outside the retryable allowlists. Several of those are transient; meka simply did not
    /// recognise them as such.
    ///
    /// So it means "not classified as transient", which is weaker than "will fail again", and the
    /// docs must not promise the stronger thing. [`Self::ProviderUnavailable`] is the positive
    /// signal and this is its absence.
    Provider,
    /// The upstream failed in a way meka's own classifier had already labelled transient.
    ///
    /// A 502 like [`Self::Provider`], and a distinct `type` for the reason
    /// [`Self::ContextOverflow`] is one: the remedies differ. This one is worth sending again after
    /// a pause. Its absence is not the opposite claim, only the lack of this one; see
    /// [`Self::Provider`].
    ///
    /// **A relayed `Retry-After` cannot be what separates this from [`Self::Provider`].** That
    /// header is absent from most of this variant's own instances: a transport failure has no
    /// response to carry one, a mid-stream `overloaded_error` has no headers to read, and
    /// [`crate::error::parse_retry_after`] reads only the delta-seconds form, so an upstream
    /// answering with an HTTP date sends none meka can use. An overload therefore arrived
    /// byte-identical to a dead credential, and a client had to choose between retrying that
    /// credential forever and dropping turns a second attempt would have completed.
    ///
    /// Claims a *class*, never an outcome, and says nothing about how many attempts meka made.
    /// `should_retry_provider_error` declines to retry at all once any output has reached the
    /// stream or the retry budget is spent, so this can arrive after three attempts or after none.
    ProviderUnavailable,
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
    /// [`Self::Internal`], which reads as "meka broke" and sends an operator to the wrong log: meka
    /// classified this one correctly, and the fault is in a subprocess or a remote MCP server.
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
            Self::SessionNotDrivable => "https://meka.so/errors/session-not-drivable",
            Self::InvalidBody => "https://meka.so/errors/invalid-body",
            Self::PayloadTooLarge => "https://meka.so/errors/payload-too-large",
            Self::ConcurrencyLimit => "https://meka.so/errors/concurrency-limit",
            Self::Provider => "https://meka.so/errors/provider",
            Self::ProviderUnavailable => "https://meka.so/errors/provider-unavailable",
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
            Self::SessionNotDrivable => "Session is a sub-agent's conversation",
            Self::InvalidBody => "Invalid request body",
            Self::PayloadTooLarge => "Request body exceeds configured limit",
            Self::ConcurrencyLimit => "Process-wide concurrency limit reached",
            Self::Provider => "Provider call failed",
            Self::ProviderUnavailable => "Provider temporarily unavailable",
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

impl ProblemDetail {
    /// Best-effort mapping from internal `MekaError` to a Problem Detail. Used by handlers that
    /// propagate agent-layer errors back to the client. Variants without a dedicated HTTP shape
    /// land on `internal` (500). Refine on demand as new error paths surface.
    ///
    /// `relay_provider_errors` is `[serve] relay_provider_errors`, and it decides whether the four
    /// upstream arms attach the provider's own response as a `provider_response` extension member.
    /// It does not touch `detail`, which is meka's own sentence and identical in both states. A
    /// parameter rather than a `From` impl precisely so it cannot be forgotten: a `.into()` that
    /// silently picked one policy is how a deployment configured to withhold would have gone on
    /// publishing from whichever call site missed the memo.
    ///
    /// Deliberately not extended to [`MekaError::McpTurnGated`], whose reasons are meka's own
    /// subprocess text rather than a provider's response; see the key's own documentation.
    pub fn for_error(error: &MekaError, relay_provider_errors: bool) -> Self {
        // An extension member rather than a replacement for `detail`, because the two carry
        // different things and neither substitutes for the other. `detail` is meka's own sentence
        // and for a context overflow it is the entire remedy ("shorten it before retrying"), which
        // relaying by overwrite would have deleted in exchange for a JSON blob. A client wanting to
        // branch on the upstream's error type also wants one well-known field, not prose to parse.
        //
        // Closured rather than repeated per arm so the four cannot drift into disagreeing about
        // what relaying means. Where `detail` does point at the server log -- three of the four
        // arms; the overflow arm gives a remedy instead -- that stays true either way, since every
        // arm logs the body unconditionally and saying so is not wrong just because the payload now
        // carries it too.
        let attach = |problem: ProblemDetail, message: &String| -> ProblemDetail {
            if relay_provider_errors {
                problem.with(
                    "provider_response",
                    Value::from(bounded_upstream_body(message)),
                )
            } else {
                problem
            }
        };
        match error {
            // Adjacent, and to the same shape [`crate::server::reattach::agent_build_problem`]
            // gives them, because the two arrive the same way: a builder refusing something the
            // caller can act on, in its own words. Every door that can raise `SessionNotDrivable`
            // today goes through that function, so these arms are the belt to its braces. Falling
            // through to `other` would answer "internal server error; consult server logs" to a
            // client whose only problem is that it posted at a sub-agent's id, which is the failure
            // that function exists to prevent.
            //
            // Same status, different `type`, because the remedies do not overlap: a `Config`
            // refusal is about what the caller sent or how the installation is set up, while
            // [`ErrorKind::SessionNotDrivable`] is a permanent property of the id itself.
            MekaError::Config(message) => ProblemDetail::new(
                ErrorKind::InvalidBody,
                StatusCode::UNPROCESSABLE_ENTITY,
                message.clone(),
            ),
            MekaError::SessionNotDrivable(message) => ProblemDetail::new(
                ErrorKind::SessionNotDrivable,
                StatusCode::UNPROCESSABLE_ENTITY,
                message.clone(),
            ),
            // Everything upstream that the arm below did not claim. Deliberately *not* described as
            // permanent: this is `provider_http_error`'s `else`, so a revoked credential lands here
            // and so does a 408, a 425, and any mid-stream error type outside the retryable
            // allowlists. Some of those are transient and meka merely failed to recognise them.
            // "Not classified as transient" is the whole claim, and `ErrorKind::Provider` carries
            // the longer version of why the stronger one would be false.
            //
            // `InvalidRequest` belongs with it rather than with the arm below despite naming a 400.
            // What the upstream called invalid is the conversation meka assembled, which the caller
            // neither sent nor can fix by correcting its own payload, and sending it again
            // unchanged gets the same 400 back.
            //
            // Logged always, and relayed when `[serve] relay_provider_errors` says so. That body
            // can hold an account identifier, a rate-limit posture, and on one backend a fragment
            // of the request that triggered it, which is why publishing it is the operator's call
            // rather than this module's; `attach` above is where the answer is applied and
            // `ServeConfig::relay_provider_errors` is where it is argued.
            //
            // Withholding was the default until 0.44 and `webhook.rs` still states that policy for
            // outbound deliveries -- identifiers and status travel, content does not -- which is
            // right there, because a webhook URL is a string in a config file rather than a caller
            // holding a token. The difference is who is on the other end.
            //
            // A length bound was tried as a *redaction* technique and did not work, for a reason
            // worth keeping: it keeps the *start* of the body, and every one of those identifiers
            // lives at the start of a JSON error object. `RELAYED_BODY_CAP` is a size bound rather
            // than a redaction one and does not revisit that.
            MekaError::Provider(message) | MekaError::InvalidRequest(message) => {
                tracing::warn!("provider error: {}", message);
                let problem = ProblemDetail::new(
                    ErrorKind::Provider,
                    StatusCode::BAD_GATEWAY,
                    "the provider rejected or failed this turn; its response is in the server log",
                );
                attach(problem, message)
            }
            // The transient half, under its own type: failures meka's own classifier labelled
            // retryable, so the answer is recovered from the branch it took rather than guessed at
            // here. It says nothing about how many attempts followed. `should_retry_provider_error`
            // declines to retry at all once any output has reached the frontend or the retry budget
            // is spent, so one of these can arrive after three attempts or after none, and reading
            // a count into the type would be inventing a guarantee.
            //
            // Split from the arm above because a client's sensible responses differ, and sharing
            // one type left it unable to choose. A relayed `Retry-After` was the only thing
            // separating them, and it is missing from most instances of this very arm: a transport
            // failure never received a response to carry one, a mid-stream `overloaded` event has
            // no headers, and `parse_retry_after` reads only delta-seconds. The 529 a bridge sees
            // most often therefore arrived byte-identical to a dead credential.
            //
            // `StreamError` is here rather than above because every producer is transport-shaped:
            // an idle timeout, an `Err` from the SSE stream itself, and a stream that ended before
            // its terminal event. A malformed SSE payload is *not* one of them, being skipped with
            // a `warn!` and a `continue`, so nothing in this arm resends into a body the provider
            // will reject identically forever.
            //
            // Falling through to the `other` arm answers 500 and logs at `error!` as an unhandled
            // internal fault, sending an operator to look in the wrong process for a failure meka
            // classified correctly.
            MekaError::StreamError(message) => {
                tracing::warn!("provider error: {}", message);
                let problem = ProblemDetail::new(
                    ErrorKind::ProviderUnavailable,
                    StatusCode::BAD_GATEWAY,
                    PROVIDER_UNAVAILABLE_DETAIL,
                );
                attach(problem, message)
            }
            // The `Retry-After`, when there is one, is worth relaying because it is fresh rather
            // than spent. It is read from the headers of the *final* attempt, and the agent loop
            // gives up rather than sleeping again, so nothing has elapsed against it by the time it
            // arrives here. Dropping it leaves a client backing off blind against a server that was
            // told the number. `StreamError` above has no equivalent, which is the only reason
            // these two are separate arms rather than one.
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
                let problem = attach(
                    ProblemDetail::new(
                        ErrorKind::ProviderUnavailable,
                        StatusCode::BAD_GATEWAY,
                        PROVIDER_UNAVAILABLE_DETAIL,
                    ),
                    message,
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
            // correct client retrying forever, so the split is deliberate.
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
                let problem = ProblemDetail::new(
                    ErrorKind::ContextOverflow,
                    StatusCode::BAD_GATEWAY,
                    "the conversation exceeds the model's context window and could not be \
                     compacted further; shorten it before retrying",
                );
                attach(problem, message)
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
        let problem = ProblemDetail::for_error(&error, false);
        assert_eq!(problem.status, 502);
        assert_eq!(problem.type_uri, "https://meka.so/errors/provider");
    }

    /// The size bound never exceeds the cap, never grows the input, and cuts on a character
    /// boundary.
    ///
    /// The four-byte case carries a one-byte prefix on purpose: `RELAYED_BODY_CAP` is 4096, which
    /// is itself divisible by four, so an unprefixed run of four-byte characters lands the cut on a
    /// boundary already and a naive slice would pass. Only a misaligned input exercises the walk.
    #[test]
    fn the_relayed_body_is_bounded_and_cut_on_a_character_boundary() {
        for (label, body) in [
            ("three-byte", "\u{3042}".repeat(RELAYED_BODY_CAP)),
            (
                "four-byte misaligned",
                format!("a{}", "\u{1F642}".repeat(RELAYED_BODY_CAP)),
            ),
            (
                "mixed",
                format!(
                    "{}{}",
                    "a".repeat(RELAYED_BODY_CAP - 1),
                    "\u{3042}".repeat(16)
                ),
            ),
            // Just over the cap, which is the one size a bound must not grow: adding a marker on
            // top would return more bytes than it was given.
            ("one byte over", "a".repeat(RELAYED_BODY_CAP + 1)),
        ] {
            let out = bounded_upstream_body(&body);
            assert!(
                out.len() <= RELAYED_BODY_CAP,
                "{label}: a bound that can be exceeded is not one; got {} bytes",
                out.len()
            );
            assert!(
                out.len() <= body.len(),
                "{label}: truncation must not grow the input; {} -> {}",
                body.len(),
                out.len()
            );
            assert!(out.contains("truncated"), "{label} must mark the cut");
            assert!(
                body.starts_with(&out[..out.len() - TRUNCATION_MARKER.len()]),
                "{label}: the kept part must be the input's own prefix, which is where a \
                 provider's error type sits"
            );
        }
        let exact = "x".repeat(RELAYED_BODY_CAP);
        assert_eq!(
            bounded_upstream_body(&exact),
            exact,
            "a body exactly at the cap is relayed whole, not marked truncated"
        );
    }

    /// The operator's switch decides whether a 502 carries the provider's own response text.
    ///
    /// That body has held an account identifier, a rate-limit posture and a fragment of the
    /// request; whoever holds a `sessions:w` token is not necessarily whoever holds the provider
    /// account. Truncating it was the first attempt and kept the start, which is where all three
    /// live in a JSON error object.
    #[test]
    fn the_upstream_body_travels_only_when_the_operator_asked_for_it() {
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
            // In the loop, not merely documented. `http-api.md` promises this arm relays like the
            // others, and nothing else in the suite calls it with relaying on: deleting its
            // `attach` left a documented member silently absent with everything green.
            MekaError::ContextOverflow(leaky.to_string()),
        ] {
            let withheld = ProblemDetail::for_error(&error, false);
            // Kept because a deleted arm falls to the catch-all, whose detail is "internal server
            // error; consult server logs" -- it contains neither secret and does contain "server
            // log" as a substring of "server logs", so the redaction checks alone would pass it.
            assert_eq!(withheld.status, 502, "{error}");
            let body = serde_json::to_string(&withheld).expect("serialize");
            assert!(
                !body.contains("acct-0f3c") && !body.contains("exceeded its quota"),
                "the upstream body reached the caller: {body}",
            );
            // The three provider arms point at the log; the overflow arm spends its `detail` on the
            // remedy instead ("shorten it before retrying") and names no log, which is right and is
            // why this is not asserted across the loop.
            let detail = withheld.detail.as_deref().unwrap_or_default().to_string();
            if !matches!(error, MekaError::ContextOverflow(_)) {
                assert!(
                    detail.contains("server log"),
                    "and the caller must be told where the detail went: {error}",
                );
            }
            assert!(
                !detail.is_empty(),
                "every arm owes the caller a sentence of meka's own: {error}",
            );

            // The whole body, not just `detail`: an arm that moved the text into an extension
            // would satisfy a `detail`-only assertion while publishing exactly what the operator
            // turned off. `provider_response` is flattened to the top level like every other
            // extension, so one serialisation covers both places it could hide.
            let relayed = ProblemDetail::for_error(&error, true);
            assert_eq!(
                relayed.extensions.get("provider_response"),
                Some(&Value::from(leaky)),
                "with relaying on, the upstream's own words must reach the caller: {error}",
            );
            assert_eq!(
                relayed.detail, withheld.detail,
                "relaying adds a member; it must not rewrite meka's own sentence, which for a \
                 context overflow is the entire remedy: {error}",
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
        let problem = ProblemDetail::for_error(&error, false);
        assert_eq!(problem.status, 502);
        assert_eq!(problem.type_uri, "https://meka.so/errors/provider");
    }

    /// An upstream failure reaches the caller as one rather than as an internal fault, and says it
    /// was the transient kind.
    ///
    /// Falling through to the catch-all makes an exhausted 429 answer 500 and log itself as an
    /// unhandled internal fault, sending an operator to look in the wrong process for a failure
    /// meka classified correctly.
    ///
    /// The type URI is asserted alongside the status because the two travel together: one error
    /// type answering with two different statuses is what a client keying on the type cannot
    /// handle.
    ///
    /// `StreamError` is in here rather than with `Provider` because every producer is
    /// transport-shaped -- an idle timeout, an `Err` from the SSE stream, a stream that ended
    /// before its terminal event -- and a malformed payload is skipped rather than raised, so
    /// nothing here invites a resend into a body the provider rejects identically every time.
    #[test]
    fn an_exhausted_upstream_failure_maps_to_502_as_unavailable() {
        for error in [
            MekaError::RetryableProvider {
                message: "529 overloaded, four attempts".into(),
                retry_after: None,
                server_error_on_completion: false,
            },
            MekaError::StreamError("connection closed mid-stream".into()),
        ] {
            let problem = ProblemDetail::for_error(&error, false);
            assert_eq!(problem.status, 502, "{error}");
            assert_eq!(
                problem.type_uri, "https://meka.so/errors/provider-unavailable",
                "{error}"
            );
        }
    }

    /// A transient failure carrying no `Retry-After` is still distinguishable from a permanent one.
    ///
    /// This is the whole complaint, as a test. The two shared `/errors/provider`, so a relayed
    /// `Retry-After` was the only thing separating them, and it is absent from most transient
    /// failures: a transport error has no response to read a header from, a mid-stream `overloaded`
    /// event has no headers at all, and `parse_retry_after` understands only delta-seconds. An
    /// overload therefore reached a bridge byte-identical to a revoked credential, leaving it to
    /// choose between retrying a dead token forever and discarding turns a second attempt would
    /// have completed.
    ///
    /// `retry_after: None` is the load-bearing part of the setup. With a header present the two
    /// were already distinguishable, so a version of this test that supplied one would pass against
    /// the shape it exists to reject.
    #[test]
    fn a_transient_failure_without_a_retry_after_is_still_distinguishable() {
        let transient = ProblemDetail::for_error(
            &MekaError::RetryableProvider {
                message: "529 overloaded".into(),
                retry_after: None,
                server_error_on_completion: false,
            },
            false,
        );
        let permanent =
            ProblemDetail::for_error(&MekaError::Provider("401 invalid x-api-key".into()), false);

        assert_eq!(
            transient.retry_after_seconds, None,
            "the setup must be the case with no header, or this proves nothing"
        );
        assert_eq!(transient.status, permanent.status, "both are 502");
        assert_ne!(
            transient.type_uri, permanent.type_uri,
            "with no `Retry-After` to tell them apart, `type` is all a client has left"
        );
        assert_eq!(
            transient.type_uri,
            "https://meka.so/errors/provider-unavailable"
        );
        assert_eq!(permanent.type_uri, "https://meka.so/errors/provider");

        // The titles too, for the reason `a_context_overflow_is_502_under_its_own_type` asserts
        // its own: `type` is what a client switches on, but `title` is what a UI renders, and
        // nothing else in the suite would notice the two variants being given one string.
        assert_eq!(transient.title, "Provider temporarily unavailable");
        assert_eq!(permanent.title, "Provider call failed");
        assert_ne!(transient.title, permanent.title);
    }

    /// A required MCP server that never came up answers 503 under its own type, and its reasons
    /// stay in the log.
    ///
    /// Both halves matter. Falling through to the `other` arm reports a subprocess that failed to
    /// start as `/errors/internal` 500, logged as an unhandled internal fault, which is the one
    /// classification that sends an operator looking in the wrong process. And a reason string is
    /// the connector's own text: it has carried a spawn failure complete with the command line and
    /// its path, which is the same argument that keeps a provider's response body out of the arm
    /// above.
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
        // `true`, which is the case that can actually fail. `[serve] relay_provider_errors` governs
        // the *provider's* response and is documented in three places as not reaching this arm, but
        // asserting it with relaying off proved only that the off switch works. Adding `attach()`
        // here would have published a spawn failure's command line and path to every caller with
        // nothing going red.
        let problem = ProblemDetail::for_error(&error, true);

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
        let problem = ProblemDetail::for_error(
            &MekaError::RetryableProvider {
                message: "429 rate limited".into(),
                retry_after: Some(std::time::Duration::from_secs(30)),
                server_error_on_completion: false,
            },
            false,
        );
        assert_eq!(problem.retry_after_seconds, Some(30));
        assert_eq!(
            problem.extensions.get("retry_after"),
            Some(&Value::from(30))
        );

        let absurd = ProblemDetail::for_error(
            &MekaError::RetryableProvider {
                message: "529 overloaded".into(),
                retry_after: Some(std::time::Duration::from_secs(31_536_000)),
                server_error_on_completion: false,
            },
            false,
        );
        assert_eq!(
            absurd.retry_after_seconds,
            Some(RELAYED_RETRY_AFTER_CAP),
            "a header saying a year must be clamped, not relayed and not wrapped"
        );

        let silent = ProblemDetail::for_error(
            &MekaError::RetryableProvider {
                message: "connection reset".into(),
                retry_after: None,
                server_error_on_completion: false,
            },
            false,
        );
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
        let problem = ProblemDetail::for_error(
            &MekaError::ContextOverflow(
                "API returned status 400: {\"error\":{\"account_uuid\":\"acct-0f3c\",\"message\":\
             \"prompt is too long: 250000 tokens > 200000 maximum\"}}"
                    .into(),
            ),
            false,
        );
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
        let problem = ProblemDetail::for_error(&MekaError::SessionLocked(id), false);
        assert_eq!(problem.status, 409);
        assert_eq!(problem.type_uri, "https://meka.so/errors/session-locked");
        assert_eq!(
            problem.extensions.get("session_id"),
            Some(&Value::String(id.to_string()))
        );
    }
}
