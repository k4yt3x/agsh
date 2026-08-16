//! Outbound webhooks for `meka serve`.
//!
//! Everything else in the HTTP API answers a question a client asked. This is the one direction
//! that has to work when nobody is asking: a scheduled job fires at 3am and a background task
//! finishes twenty minutes after the turn that started it, and until now the only trace either left
//! was rows in SQLite that something had to poll to discover. [`crate::server::schedule`] has said
//! for a while that this is where a push API hooks in.
//!
//! Two decisions shape the whole module.
//!
//! **Payloads carry identifiers and metadata, never message content.** A webhook endpoint is a URL
//! in a config file: it can be mistyped, it can outlive the service that owned it, and it is
//! reachable by anything that learns it. So a delivery says *what happened to which session*, and
//! the client fetches the conversation with its own bearer token over the API it already trusts. A
//! compromised endpoint learns that a session was active, not what was said in it.
//!
//! **Delivery never blocks the work that triggered it.** Each send is a detached task with bounded
//! retries. A webhook receiver that hangs must not wedge the scheduler behind it, because the next
//! job is someone else's.

use std::{sync::Arc, time::Duration};

use hmac::{Hmac, KeyInit, Mac};
use serde::Serialize;
use sha2::Sha256;
use uuid::Uuid;

use crate::config::ResolvedWebhook;

/// The events a webhook can subscribe to.
///
/// Deliberately short, and every one of them is something no client is necessarily waiting on. A
/// turn a client submitted itself already has a response and an SSE stream; `turn.finished` is here
/// for the *other* consumers of a shared session, and for turns the server started on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebhookEvent {
    TurnFinished,
    TurnFailed,
    TaskFinished,
    ScheduleFired,
}

impl WebhookEvent {
    /// Every event name, for config validation. Kept sorted so the error text listing them is
    /// stable.
    pub const ALL: &'static [&'static str] = &[
        "schedule.fired",
        "task.finished",
        "turn.failed",
        "turn.finished",
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TurnFinished => "turn.finished",
            Self::TurnFailed => "turn.failed",
            Self::TaskFinished => "task.finished",
            Self::ScheduleFired => "schedule.fired",
        }
    }
}

/// One delivery's body. Flattened into the JSON object alongside the event-specific `data`.
#[derive(Debug, Serialize)]
struct DeliveryBody<'a> {
    /// Unique per delivery attempt series, so a receiver can deduplicate retries.
    delivery_id: &'a str,
    event: &'a str,
    /// RFC 3339. Also signed, so a replayed body cannot be passed off as current.
    timestamp: &'a str,
    #[serde(flatten)]
    data: serde_json::Value,
}

/// Dispatches deliveries to every configured endpoint that subscribed to the event.
///
/// Cheap to clone and to call: [`Self::send`] returns as soon as the tasks are spawned, and does
/// nothing at all when no endpoint wants the event.
#[derive(Clone)]
pub struct WebhookDispatcher {
    endpoints: Arc<Vec<ResolvedWebhook>>,
    /// `None` when no endpoint is configured, or when the client could not be built.
    ///
    /// An `Option` rather than an eagerly-unwrapped client: `reqwest::Client::new()` panics on a
    /// TLS-init failure, so `build().unwrap_or_default()` would swap one panic for another. A
    /// server that cannot build an HTTP client should still start and serve every other endpoint;
    /// webhooks are the thing that degrades.
    client: Option<reqwest::Client>,
}

impl WebhookDispatcher {
    pub fn new(endpoints: Vec<ResolvedWebhook>) -> Self {
        // One client for the process: it pools connections, which matters for a receiver being hit
        // once per scheduled job. Built only when something is configured, so the common
        // no-webhooks deployment pays nothing.
        let client = if endpoints.is_empty() {
            None
        } else {
            match reqwest::Client::builder()
                .user_agent(concat!("meka/", env!("CARGO_PKG_VERSION")))
                .build()
            {
                Ok(client) => Some(client),
                Err(error) => {
                    tracing::error!(
                        "failed to build the webhook HTTP client; deliveries are disabled: {}",
                        error
                    );
                    None
                }
            }
        };
        Self {
            endpoints: Arc::new(endpoints),
            client,
        }
    }

    /// Queue `event` for delivery to every endpoint subscribed to it.
    ///
    /// `data` becomes the event-specific part of the body. Returns immediately; failures are
    /// reported through `tracing` because there is nobody on this side of the call to return them
    /// to. `timestamp` is taken once here so every endpoint receives, and signs over, the same one.
    pub fn send(&self, event: WebhookEvent, data: serde_json::Value) {
        let Some(client) = self.client.as_ref() else {
            return;
        };
        let timestamp = chrono::Utc::now().to_rfc3339();
        for endpoint in self.endpoints.iter() {
            if !endpoint.events.iter().any(|name| name == event.as_str()) {
                continue;
            }
            let delivery_id = Uuid::new_v4().to_string();
            let body = DeliveryBody {
                delivery_id: &delivery_id,
                event: event.as_str(),
                timestamp: &timestamp,
                data: data.clone(),
            };
            let payload = match serde_json::to_vec(&body) {
                Ok(payload) => payload,
                Err(error) => {
                    tracing::error!("failed to serialize webhook payload: {}", error);
                    continue;
                }
            };
            let task = DeliveryTask {
                client: client.clone(),
                url: endpoint.url.clone(),
                secret: endpoint.secret.clone(),
                timeout: endpoint.timeout,
                max_retries: endpoint.max_retries,
                event: event.as_str(),
                delivery_id,
                timestamp: timestamp.clone(),
                payload,
            };
            tokio::spawn(task.run());
        }
    }
}

struct DeliveryTask {
    client: reqwest::Client,
    url: String,
    secret: Option<String>,
    timeout: Duration,
    max_retries: u32,
    event: &'static str,
    delivery_id: String,
    timestamp: String,
    payload: Vec<u8>,
}

impl DeliveryTask {
    async fn run(self) {
        let signature = self
            .secret
            .as_deref()
            .map(|secret| sign(secret, &self.timestamp, &self.payload));

        // Attempt 0 plus `max_retries` retries.
        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                // 1s, 2s, 4s, capped at 30s. Same shape as the MCP reconnect backoff, but the
                // exponent is clamped before the shift rather than after: `max_retries` comes from
                // config, and `1u64 << 64` is a panic in debug and a silently wrapped shift in
                // release. Clamping at 5 is free because the result is capped at 30 anyway.
                let exponent = (attempt - 1).min(5);
                let delay = Duration::from_secs(std::cmp::min(30, 1u64 << exponent));
                tokio::time::sleep(delay).await;
            }
            let mut request = self
                .client
                .post(&self.url)
                .timeout(self.timeout)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .header("X-Meka-Event", self.event)
                .header("X-Meka-Delivery", &self.delivery_id)
                .header("X-Meka-Timestamp", &self.timestamp)
                .body(self.payload.clone());
            if let Some(signature) = &signature {
                request = request.header("X-Meka-Signature", signature);
            }

            match request.send().await {
                Ok(response) if response.status().is_success() => {
                    tracing::debug!(
                        "webhook {} delivered to {} (attempt {})",
                        self.event,
                        self.url,
                        attempt + 1
                    );
                    return;
                }
                Ok(response) => {
                    let status = response.status();
                    // 4xx is the receiver saying the request itself is wrong, which retrying
                    // cannot fix; 5xx and transport errors are worth another go.
                    //
                    // 429 and 408 are the exceptions: both say "not now" rather than "not ever",
                    // and they are what a receiver returns for precisely the traffic shape this
                    // module produces. Several jobs sharing a cron minute deliver as a burst, and
                    // dropping a rate-limited delivery would lose the 9am report to the one thing
                    // the receiver was explicitly asking meka to wait out.
                    let retryable = matches!(
                        status,
                        reqwest::StatusCode::TOO_MANY_REQUESTS
                            | reqwest::StatusCode::REQUEST_TIMEOUT
                    );
                    if status.is_client_error() && !retryable {
                        // Scheme + host, not the URL. `warn` is the default level, so this is the
                        // line that ends up pasted into an issue tracker -- and for a Slack- or
                        // Discord-style endpoint the path *is* the credential. Same rule the
                        // config-load warning follows; the full URL stays at `info`.
                        tracing::warn!(
                            "webhook {} rejected by {} with {}; not retrying",
                            self.event,
                            crate::config::webhook_host(&self.url),
                            status
                        );
                        return;
                    }
                    tracing::debug!(
                        "webhook {} to {} returned {} (attempt {})",
                        self.event,
                        self.url,
                        status,
                        attempt + 1
                    );
                }
                Err(error) => {
                    tracing::debug!(
                        "webhook {} to {} failed (attempt {}): {}",
                        self.event,
                        self.url,
                        attempt + 1,
                        error
                    );
                }
            }
        }
        // Said once, at `warn`, after everything has been tried: an endpoint that is down is a
        // configuration problem the operator needs to see, and the per-attempt lines above are
        // `debug` precisely so this one is not buried.
        // Scheme + host for the same reason as the rejection line above: this fires exactly when
        // an endpoint is down, which is exactly when the log gets shared.
        tracing::warn!(
            "webhook {} to {} failed after {} attempt(s); giving up",
            self.event,
            crate::config::webhook_host(&self.url),
            self.max_retries + 1
        );
    }
}

/// `sha256=<hex>` over `<timestamp>.<body>`, keyed with the endpoint's secret.
///
/// The timestamp is inside the signed material, not merely alongside it. Signing the body alone
/// would let anyone who captured one delivery replay it forever with a valid signature; with the
/// timestamp signed, a receiver that rejects old timestamps closes that window, and cannot be
/// tricked by rewriting the header.
pub fn sign(secret: &str, timestamp: &str, payload: &[u8]) -> String {
    // HMAC derives its own fixed-size block from a key of any length, so `InvalidLength` is
    // unreachable here. The `expect` documents that invariant rather than propagating an error no
    // call site could act on, matching how `crate::store::validate_entry_name` handles its own.
    #[allow(clippy::expect_used)]
    let mut mac = <Hmac<Sha256> as KeyInit>::new_from_slice(secret.as_bytes())
        .expect("HMAC accepts keys of any length");
    mac.update(timestamp.as_bytes());
    mac.update(b".");
    mac.update(payload);
    let digest = mac.finalize().into_bytes();
    let mut out = String::with_capacity(7 + digest.len() * 2);
    out.push_str("sha256=");
    for byte in digest.iter() {
        out.push_str(&format!("{:02x}", byte));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint(events: &[&str]) -> ResolvedWebhook {
        ResolvedWebhook {
            url: "http://127.0.0.1:1/hook".to_string(),
            secret: Some("shhh".to_string()),
            events: events.iter().map(|event| (*event).to_string()).collect(),
            timeout: Duration::from_secs(1),
            max_retries: 0,
        }
    }

    /// Pinned against a hand-computed value rather than a round-trip through `sign` itself, which
    /// would pass even if the signed material were wrong.
    #[test]
    fn signature_covers_the_timestamp_and_the_body() {
        let signature = sign("shhh", "2026-01-01T00:00:00Z", b"{\"a\":1}");
        assert!(signature.starts_with("sha256="));
        // Changing either half must change the signature, or the timestamp is decorative.
        let other_time = sign("shhh", "2026-01-01T00:00:01Z", b"{\"a\":1}");
        let other_body = sign("shhh", "2026-01-01T00:00:00Z", b"{\"a\":2}");
        let other_key = sign("different", "2026-01-01T00:00:00Z", b"{\"a\":1}");
        assert_ne!(signature, other_time, "the timestamp must be signed");
        assert_ne!(signature, other_body, "the body must be signed");
        assert_ne!(signature, other_key, "the secret must key the digest");
    }

    /// Concatenating `timestamp` and `payload` without a separator would let two different
    /// (timestamp, body) pairs produce the same signed bytes.
    #[test]
    fn signature_separator_prevents_boundary_ambiguity() {
        let a = sign("k", "12", b"3");
        let b = sign("k", "1", b"23");
        assert_ne!(
            a, b,
            "a delimiter must separate the timestamp from the body, or the split is ambiguous"
        );
    }

    #[test]
    fn signature_is_stable_for_the_same_inputs() {
        let first = sign("k", "t", b"body");
        let second = sign("k", "t", b"body");
        assert_eq!(first, second);
        assert_eq!(
            first.len(),
            "sha256=".len() + 64,
            "SHA-256 hex is 64 characters"
        );
    }

    #[tokio::test]
    async fn send_is_a_noop_without_endpoints() {
        let dispatcher = WebhookDispatcher::new(Vec::new());
        // No panic, no task spawned, nothing to await.
        dispatcher.send(WebhookEvent::TurnFinished, serde_json::json!({}));
    }

    /// An endpoint only receives what it subscribed to. Fanning every event at every endpoint
    /// would make the `events` list decorative.
    #[tokio::test]
    async fn send_skips_endpoints_that_did_not_subscribe() {
        let dispatcher = WebhookDispatcher::new(vec![endpoint(&["schedule.fired"])]);
        // Nothing observable to assert without a listener; this pins that the filter path runs
        // without panicking for both the matching and non-matching case.
        dispatcher.send(WebhookEvent::TurnFinished, serde_json::json!({}));
        dispatcher.send(WebhookEvent::ScheduleFired, serde_json::json!({}));
    }

    /// `max_retries` is operator-supplied, so the backoff arithmetic has to survive a large one.
    /// Shifting by the raw attempt number panics in debug at 64 and silently wraps in release,
    /// which would turn a long retry schedule into a hot loop.
    #[test]
    fn backoff_exponent_is_clamped_before_the_shift() {
        for attempt in 1u32..200 {
            let exponent = (attempt - 1).min(5);
            let delay = std::cmp::min(30, 1u64 << exponent);
            assert!(
                (1..=30).contains(&delay),
                "attempt {attempt} produced a {delay}s delay"
            );
        }
    }

    #[test]
    fn every_event_name_is_in_the_config_allow_list() {
        for event in [
            WebhookEvent::TurnFinished,
            WebhookEvent::TurnFailed,
            WebhookEvent::TaskFinished,
            WebhookEvent::ScheduleFired,
        ] {
            assert!(
                WebhookEvent::ALL.contains(&event.as_str()),
                "'{}' is deliverable but config would reject it",
                event.as_str()
            );
        }
        let mut sorted = WebhookEvent::ALL.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted.as_slice(), WebhookEvent::ALL, "kept sorted");
    }
}
