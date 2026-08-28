//! LLM provider abstraction. Defines the [`Provider`] trait, the shared message/content/tool types,
//! and the [`ProviderBuilder`] that returns a concrete backend.
//!
//! A backend is named for the wire protocol it speaks, not for a vendor or an auth method, because
//! neither of those identifies the request shape: one vendor serves several protocols (OpenAI has
//! both Chat Completions and Responses) and one protocol is served by many vendors (`/v1/messages`
//! by Anthropic, LiteLLM, Databricks, Ollama, …). The two subscription backends carry a vendor name
//! instead, because what they select is a billing relationship whose endpoint and client shape come
//! with it.

mod anthropic;
/// `meka provider` subcommand suite (add/list/use/remove/login) and the provider OAuth login flows.
pub mod cli;
/// Scripted provider used by the integration tests. Available in debug builds only; release builds
/// don't pay the binary-size cost. Activated by the `MEKA_MOCK_PROVIDER` env var inside
/// `acp::run_acp`, `server::run_serve` and `create_agent_from_config`; never reachable from
/// production paths otherwise.
#[cfg(debug_assertions)]
pub(crate) mod mock;
pub(crate) mod openai;
/// Backoff policy for retrying [`crate::error::MekaError::RetryableProvider`] failures.
pub(crate) mod retry;

use std::{sync::Arc, time::Duration};

pub use anthropic::{AnthropicMessagesProvider, ClaudeSubscriptionProvider};
use async_trait::async_trait;
pub use openai::{
    ChatGptSubscriptionProvider, OpenAiChatCompletionsProvider, OpenAiResponsesProvider,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    error::{MekaError, Result},
    session::TokenStore,
};

pub(crate) const DEFAULT_CLAUDE_SUBSCRIPTION_CLIENT_ID: &str =
    "9d1c250a-e61b-44d9-88ed-5944d1962f5e";

/// Codex's hardcoded OpenAI OAuth client ID. Mirrors the value used by the first-party CLI at
/// `codex-rs/login/src/auth/manager.rs`.
pub(crate) const DEFAULT_CHATGPT_SUBSCRIPTION_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

pub const SUPPORTED_PROVIDERS: &[&str] = &[
    "anthropic-messages",
    "chatgpt-subscription",
    "claude-subscription",
    "openai-chat-completions",
    "openai-responses",
];

/// The endpoint each backend talks to when a profile sets no `base_url`.
///
/// Named rather than written inline at each constructor so `meka provider add` can *show* the
/// default it is about to apply. A prompt carrying its own copy of the string would eventually
/// offer one URL while the request went to another.
pub(crate) const DEFAULT_ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com";
pub(crate) const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
pub(crate) const DEFAULT_CHATGPT_BASE_URL: &str = "https://chatgpt.com";

/// The default endpoint for `backend`, or `None` for a name that isn't a backend. Every supported
/// backend has one, so the `None` arm exists only to keep the match total.
pub(crate) fn default_base_url(backend: &str) -> Option<&'static str> {
    match backend {
        "anthropic-messages" | "claude-subscription" => Some(DEFAULT_ANTHROPIC_BASE_URL),
        "openai-chat-completions" | "openai-responses" => Some(DEFAULT_OPENAI_BASE_URL),
        "chatgpt-subscription" => Some(DEFAULT_CHATGPT_BASE_URL),
        _ => None,
    }
}

/// Whether `backend`'s requests carry a `thinking` field at all, i.e. whether it speaks Anthropic's
/// Messages API. `/status` and `meka provider add` both key off this, so they agree about which
/// profiles the [`ThinkingMode`] setting is even meaningful for.
///
/// A single function rather than a test at each site. Both used to ask
/// `backend.starts_with("claude")`, which was true of every Anthropic backend only while they were
/// *named* for Claude; the protocol name `anthropic-messages` silently falls out of a prefix test,
/// and nothing about the failure is visible except a missing line.
pub(crate) fn backend_takes_thinking(backend: &str) -> bool {
    matches!(backend, "anthropic-messages" | "claude-subscription")
}

/// How long a provider stream may go without producing anything before it is treated as dead.
///
/// This bounds *silence*, never a turn. A model thinking hard, or a request queued behind a busy
/// endpoint, keeps sending: reasoning deltas, text deltas, ping events. Nothing here caps how long
/// a turn may legitimately run, how many tool calls it may make, or how many tokens it may spend.
///
/// Measured in decodable SSE *events*, not bytes. All three drivers wrap `event_stream.next()`, and
/// `eventsource-stream` discards comment lines and refuses to dispatch a data-empty event, so a
/// keepalive that is only `: ping` does not reset this clock -- an endpoint sending nothing else
/// for five minutes is treated as silent, which is the intended reading of it but not what "without
/// a byte" would mean. Bounding actual bytes would mean timing the response body underneath the SSE
/// decoder in all three drivers, for a shape no provider meka targets produces; the wording is
/// corrected instead. Every provider in use sends `ping` / `keep-alive` events with a data field,
/// which do reset it.
pub(crate) const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(300);

/// How long to wait for the TCP + TLS handshake before giving up on a provider endpoint.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// The context window meka assumes when a profile doesn't state one.
///
/// meka does not look this up, probe for it, or cache it. The window is a local budgeting number -
/// it drives compaction timing, the keep-budget and the `/status` gauge, and is never sent on the
/// wire - so a wrong value cannot fail a request, and the user can state the real one via
/// `[providers.<name>].context_window`.
///
/// 1M is right for the current flagships on both vendors and too generous for the smaller and older
/// models. That direction is deliberate but not free: planned compaction never fires when the real
/// window is smaller, so those turns hit the provider's own limit and recover through the
/// `ContextOverflow` compact-and-retry path instead, paying one rejected round trip each time. The
/// opposite default would make the common case compact at a fraction of its real window, which is
/// worse every day rather than occasionally.
pub(crate) const DEFAULT_CONTEXT_WINDOW: u64 = 1_000_000;

/// The HTTP client every provider backend uses.
///
/// Deliberately sets `connect_timeout` and `read_timeout` and *not* `timeout`. A whole-request
/// deadline would kill a legitimate long turn, which is the one thing the harness must not do;
/// `read_timeout` resets on every successful read, so it fires only when the connection has gone
/// quiet. Without either, a dropped route left the turn waiting on a socket that would never
/// produce another byte, with no error and no retry.
pub(crate) fn build_http_client(
    backend: &str,
    configure: impl FnOnce(reqwest::ClientBuilder) -> reqwest::ClientBuilder,
) -> Result<reqwest::Client> {
    configure(
        reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .read_timeout(STREAM_IDLE_TIMEOUT),
    )
    .build()
    .map_err(|error| MekaError::Provider(format!("failed to build {backend} HTTP client: {error}")))
}

/// How long before an access token's stated expiry to refresh it, so a token cannot expire between
/// the header being built and the request arriving.
const OAUTH_REFRESH_SKEW_MILLIS: i64 = 300_000;

/// How long a refreshed access token is assumed to last when its issuer states no expiry.
///
/// An *unlabelled stored* credential being due is right: refreshing it is how it becomes labelled.
/// An unlabelled credential coming back *from that refresh* is a different thing -- the issuer
/// answered and still said nothing -- and treating it the same way put every subsequent request
/// back on the slow path: the credential write lock, a database re-read, and a full OAuth round
/// trip that rotates the refresh token, all serialised behind one another. Assuming a short life
/// bounds the staleness without the storm, and a token that dies sooner is corrected by its 401.
const OAUTH_ASSUMED_LIFETIME_MILLIS: i64 = 3_600_000;

/// Expiry to record for a refreshed token whose issuer named none.
pub(crate) fn oauth_assumed_expiry(now_millis: i64) -> i64 {
    now_millis.saturating_add(OAUTH_ASSUMED_LIFETIME_MILLIS)
}

/// Whether an OAuth access token should be refreshed before the next request.
///
/// `expires_at: None` means the issuer did not say when the token expires, which is not a promise
/// that it never will. Reading it that way turned an unlabelled token into a 401 on every request
/// with no refresh ever attempted. It is treated as due, but only when there is a refresh token to
/// act on: without one, the only thing left is to send it and let the 401 speak.
///
/// The refresh paths stamp [`oauth_assumed_expiry`] rather than handing `None` straight back, so
/// "due" here stays a one-shot rather than a per-request loop.
pub(crate) fn oauth_needs_refresh(
    expires_at: Option<i64>,
    has_refresh_token: bool,
    now_millis: i64,
) -> bool {
    match expires_at {
        Some(expiry) => now_millis.saturating_add(OAUTH_REFRESH_SKEW_MILLIS) >= expiry,
        None => has_refresh_token,
    }
}

/// Milliseconds since the Unix epoch, the unit [`oauth_needs_refresh`] and every stored
/// `expires_at` are in.
///
/// Clamped at zero for a clock before 1970 rather than propagating: every caller is asking "is
/// this token due", and a machine that far out of step has already answered yes.
pub(crate) fn now_epoch_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

/// Whether a refresh that lost its swap should switch to what the row holds instead.
///
/// Two conditions, and both are about the request in hand rather than about write order.
///
/// It has to be the same *kind* of credential. A row that held an OAuth token and now holds an API
/// key is a profile somebody repurposed with `meka provider add --api-key`, not a token this
/// refresh has been overtaken by, and a subscription backend cannot authenticate with it at all.
///
/// And it has to be *live*. The winner may have stored a token and then sat idle past its
/// lifetime, so adopting on write order alone would authenticate this very request with a token
/// that is already dead, having thrown away the good one this refresh just minted. Neither
/// outcome writes to the row -- what it holds is not this process's to change -- so the only
/// question is which token to spend the turn on.
fn is_worth_adopting(derived_from: &AuthCredential, current: &AuthCredential) -> bool {
    match (derived_from, current) {
        (AuthCredential::ApiKey(_), AuthCredential::ApiKey(_)) => true,
        (
            AuthCredential::OAuthToken { .. },
            AuthCredential::OAuthToken {
                expires_at,
                refresh_token,
                ..
            },
        ) => !oauth_needs_refresh(*expires_at, refresh_token.is_some(), now_epoch_millis()),
        _ => false,
    }
}

/// How long to wait for another process to finish rotating a profile's credential before going
/// ahead anyway.
///
/// Bounded rather than blocking, because a wedged holder must not wedge every other meka on the
/// machine. Going ahead is safe: [`store_refreshed_credential`]'s compare-and-swap is what makes
/// the outcome correct, and this only spares the wasted refresh in the common case.
const CREDENTIAL_LOCK_WAIT: Duration = Duration::from_secs(10);

/// Wait, briefly, for exclusive use of a profile's credential across processes.
///
/// `None` means the wait ran out or the lock could not be asked for at all. The caller proceeds
/// either way: this reduces contention, it does not establish correctness.
pub(crate) async fn await_credential_lock(
    store: &TokenStore,
    profile: &str,
) -> Option<crate::session::FileLock> {
    let deadline = tokio::time::Instant::now() + CREDENTIAL_LOCK_WAIT;
    loop {
        match store.try_lock_provider_credential(profile) {
            Ok(Some(lock)) => return Some(lock),
            Ok(None) => {}
            // Not "someone has it" but "we could not ask": an unwritable lock directory, or
            // descriptors exhausted. Retrying would not help and refusing to refresh would be
            // worse than refreshing unserialised, so stop waiting and let the swap arbitrate.
            Err(error) => {
                tracing::debug!(
                    "could not take the credential lock for '{}': {}",
                    profile,
                    error
                );
                return None;
            }
        }
        if tokio::time::Instant::now() >= deadline {
            tracing::debug!(
                "another process has been refreshing '{}' for {:?}; going ahead",
                profile,
                CREDENTIAL_LOCK_WAIT
            );
            return None;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// How long an OAuth refresh round trip may take before it is abandoned.
///
/// A whole-request deadline is right here in a way it never is for a turn: this is a small request
/// to a well-known endpoint and nothing legitimate about it takes minutes. It also runs behind the
/// gate that serialises a profile's refreshes, and *inside* `complete`/`stream` via
/// `ensure_valid_credential`, so a token endpoint that accepts the connection and then goes quiet
/// would otherwise hold both the next refresher and the user's turn in progress.
///
/// It cannot usefully go below [`CONNECT_TIMEOUT`], which is the trap in tightening it: reqwest's
/// per-request `timeout` runs from the start of connecting, not from the request being sent, so a
/// value under 30 seconds would abandon a connection this same client is still willing to spend 30
/// seconds establishing. The two numbers being equal means a slow connect can consume the whole
/// refresh budget, which is the correct precedence -- the total is what bounds the user's wait.
const REFRESH_TIMEOUT: Duration = Duration::from_secs(30);

/// Everything an OAuth refresh-token exchange needs, named rather than positional.
///
/// A struct because the alternative is six parameters of which five are `&str`, where transposing
/// two of them still compiles and produces a request that fails at the issuer with a message about
/// the wrong thing. Named fields make the two call sites say which is which.
pub(crate) struct RefreshExchange<'a> {
    pub(crate) client: &'a reqwest::Client,
    /// The issuer's token endpoint.
    pub(crate) token_url: &'a str,
    /// The OAuth client this profile authenticated as.
    pub(crate) client_id: &'a str,
    /// The grant being spent. Single-use under rotation; see
    /// [`crate::error::oauth_refresh_error`].
    pub(crate) refresh_token: &'a str,
    /// The profile whose credential this is, named in the error so a user knows which one to log
    /// back in to.
    pub(crate) profile: &'a str,
    /// The call site's own description of the request, in the voice that site has always used
    /// ("OAuth token refresh", "Codex OAuth token refresh"), so its messages do not all read
    /// alike.
    pub(crate) context: &'a str,
}

/// Spend a refresh token at an issuer's token endpoint and hand back its decoded response.
///
/// One copy of a branch both subscription backends used to keep their own version of. The shape was
/// identical -- POST the grant, classify the answer, decode the payload -- and only the payload
/// differs, so the generic parameter is exactly the part that was ever really different: Claude's
/// issuer states `expires_in` and an account uuid, ChatGPT's states neither and has to be read out
/// of the JWTs.
///
/// The duplication was not theoretical. A mutation sweep deleted the `!` from the Codex copy's
/// `if !status.is_success()` and no test noticed, because the Claude copy was the one under test;
/// two hand-written copies of a classification means two chances to get it wrong and two places to
/// remember when it changes. Both call sites keep their own test even so, since each still has to
/// prove it reaches this function at all.
///
/// Errors are classified where they belong rather than here: a call that got no answer through
/// [`crate::error::provider_transport_error`], and an answered one through
/// [`crate::error::oauth_refresh_error`], which is the function that knows a spent grant from an
/// unwell server.
pub(crate) async fn exchange_refresh_token<T: serde::de::DeserializeOwned>(
    exchange: RefreshExchange<'_>,
) -> Result<T> {
    let response = exchange
        .client
        .post(exchange.token_url)
        .timeout(REFRESH_TIMEOUT)
        // One order for both backends, where they used to list the same three fields in two
        // different orders. `serde_json` is built with `preserve_order`, so this does change the
        // bytes the Claude endpoint receives; it cannot change what they mean, because a JSON
        // object is an unordered collection (RFC 8259 §1) and nothing here is signed over. Worth
        // saying only
        // because this is the backend whose request shape is otherwise reproduced from a capture.
        .json(&serde_json::json!({
            "client_id": exchange.client_id,
            "grant_type": "refresh_token",
            "refresh_token": exchange.refresh_token,
        }))
        .send()
        .await
        .map_err(|error| {
            // Retryable like any other call that got no answer. The grant's fate is the one
            // wrinkle, and `oauth_refresh_error` records why a possible replay is still the better
            // trade than ending a turn on a blip.
            crate::error::provider_transport_error(
                &format!("{} request failed", exchange.context),
                &error,
                None,
            )
        })?;

    let status = response.status();
    let retry_after = crate::error::parse_retry_after(response.headers());
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|error| {
            tracing::warn!(
                "failed to read the OAuth refresh error body for '{}': {}",
                exchange.profile,
                error
            );
            String::new()
        });
        return Err(crate::error::oauth_refresh_error(
            &format!("{} failed", exchange.context),
            status,
            &body,
            retry_after,
            exchange.profile,
        ));
    }

    response.json::<T>().await.map_err(|error| {
        crate::error::oauth_refresh_error(
            &format!(
                "{} returned a response that could not be read",
                exchange.context
            ),
            status,
            &crate::error::format_reqwest_error(&error),
            None,
            exchange.profile,
        )
    })
}

/// Persist a refreshed credential, and answer with the one that should actually be used.
///
/// A refresh is derived from the credential it read, so it may only replace *that* credential.
/// Where the row has moved on -- another process refreshed first, or a `meka provider login`
/// completed while this round trip was in flight -- the stored value is newer than what this
/// refresh produced, and adopting it is both correct and what keeps the issuer's live token and the
/// database in agreement. The blind upsert this replaces left them disagreeing silently, and the
/// symptom arrived at the *next* launch as `invalid_grant` with nothing naming the cause.
///
/// A write that cannot be made at all is a warning rather than an error: the session in hand still
/// has a working token, and failing the turn over a persistence problem would be the worse trade.
pub(crate) async fn store_refreshed_credential(
    store: &TokenStore,
    profile: &str,
    derived_from: &AuthCredential,
    refreshed: AuthCredential,
) -> AuthCredential {
    match store
        .replace_provider_credential(profile, derived_from, &refreshed)
        .await
    {
        Ok(crate::session::CredentialWrite::Stored) => refreshed,
        // Adopted only when it is worth adopting: "newer" here means newer in *write order*, which
        // is neither "unexpired" nor "the same kind of credential". See [`is_worth_adopting`].
        Ok(crate::session::CredentialWrite::Superseded(current))
            if is_worth_adopting(derived_from, &current) =>
        {
            tracing::info!(
                "'{}' was re-authenticated elsewhere while this refresh was in flight; adopting \
                 the stored credential",
                profile
            );
            *current
        }
        Ok(crate::session::CredentialWrite::Superseded(_)) => {
            tracing::warn!(
                "'{}' was written by something else while this refresh was in flight, and what it \
                 holds now cannot authenticate this session; continuing on the token just minted \
                 and leaving the stored credential alone",
                profile
            );
            refreshed
        }
        // The profile's credential was removed mid-refresh, which is `meka provider remove` or a
        // `logout`. Writing this token back would resurrect an account the user just disconnected;
        // this process finishes its turn on what it has and the next launch asks for a login.
        Ok(crate::session::CredentialWrite::Gone) => {
            tracing::warn!(
                "'{}' has no stored credential any more, so the refreshed token was not persisted",
                profile
            );
            refreshed
        }
        Err(error) => {
            tracing::warn!(
                "failed to persist the refreshed token for '{}' ({}); this session continues, but \
                 the stored token is now stale and the next launch will need \
                 `meka provider login`",
                profile,
                error
            );
            refreshed
        }
    }
}

/// The slot a conversation keeps its most recent provider request id in, so the next request can
/// name it. Held by the [`crate::agent::Agent`] and handed to each turn via [`scope_turn`].
pub(crate) type PreviousRequestSlot = Arc<std::sync::Mutex<Option<String>>>;

/// Who a request is for, as the billing header has to describe it.
///
/// None of this can live on the provider: one `Arc<dyn Provider>` serves the main agent and every
/// sub-agent at once, so the answer differs between two requests it is handling concurrently. It
/// rides a task-local instead, mirroring the `AsyncLocalStorage` Claude Code threads the same three
/// facts through.
///
/// Every field is optional in the same way the wire segment is: absent means the request genuinely
/// has no such attribution, which is how meka's own side queries come out bare and how a
/// conversation's first request omits `cc_prev_req`.
#[derive(Clone, Default)]
pub(crate) struct Attribution {
    /// Whether this is a sub-agent's request (`cc_is_subagent`).
    subagent: bool,
    /// The prompt this request serves (`cc_prompt_id`).
    prompt_id: Option<Uuid>,
    /// Where the last response's id is kept, for `cc_prev_req`. Shared with the conversation that
    /// owns it, which is why it is a handle and not a value: turn two has to be able to name turn
    /// one's last response. Claude Code reads that off the last assistant message in its history
    /// (`t0E`); meka's conversation doesn't carry request ids, so the provider deposits them here.
    previous_request: Option<PreviousRequestSlot>,
}

tokio::task_local! {
    static ATTRIBUTION: Attribution;
}

fn attribution() -> Attribution {
    ATTRIBUTION.try_with(Attribution::clone).unwrap_or_default()
}

/// Whether the current task is executing a sub-agent's turn. `false` outside any
/// [`scope_subagent`] (the main agent, tests, etc.).
pub(crate) fn is_subagent() -> bool {
    ATTRIBUTION
        .try_with(|current| current.subagent)
        .unwrap_or(false)
}

/// The current prompt's id, or `None` outside any [`scope_turn`].
pub(crate) fn current_prompt_id() -> Option<Uuid> {
    ATTRIBUTION
        .try_with(|current| current.prompt_id)
        .ok()
        .flatten()
}

/// The id of the most recent response in this conversation, if one has arrived and a turn is in
/// scope.
pub(crate) fn previous_request_id() -> Option<String> {
    ATTRIBUTION
        .try_with(|current| {
            current
                .previous_request
                .as_ref()
                .map(|slot| match slot.lock() {
                    Ok(guard) => guard.clone(),
                    Err(poisoned) => poisoned.into_inner().clone(),
                })
        })
        .ok()
        .flatten()
        .flatten()
}

/// Record the id a provider response arrived with, for the next request in this conversation.
///
/// Anything that isn't shaped like an Anthropic request id is dropped rather than forwarded:
/// Claude Code gates the segment on `^req_[A-Za-z0-9_-]{1,36}$` and a value that fails it would be
/// a claim about a response that doesn't exist.
pub(crate) fn record_request_id(request_id: &str) {
    if !is_anthropic_request_id(request_id) {
        tracing::debug!("ignoring malformed provider request id '{}'", request_id);
        return;
    }
    let _ = ATTRIBUTION.try_with(|current| {
        if let Some(slot) = &current.previous_request {
            match slot.lock() {
                Ok(mut guard) => *guard = Some(request_id.to_string()),
                Err(poisoned) => *poisoned.into_inner() = Some(request_id.to_string()),
            }
        }
    });
}

fn is_anthropic_request_id(value: &str) -> bool {
    let Some(rest) = value.strip_prefix("req_") else {
        return false;
    };
    !rest.is_empty()
        && rest.len() <= 36
        && rest
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

/// Run `future` with the sub-agent attribution flag set, keeping whatever else is in scope.
pub(crate) async fn scope_subagent<F: std::future::Future>(future: F) -> F::Output {
    ATTRIBUTION
        .scope(
            Attribution {
                subagent: true,
                ..attribution()
            },
            future,
        )
        .await
}

/// Run one turn's work with its prompt id and its conversation's request-id slot in scope.
///
/// An already-scoped prompt id is kept rather than replaced, which is what makes a sub-agent
/// inherit its spawner's: the sub-agent's turn runs nested inside the parent's, and Claude Code
/// falls back to `agentContext.parentPromptId` for exactly this case. The request-id slot is not
/// inherited -- every agent passes its own -- because a sub-agent's requests belong to its own
/// conversation.
pub(crate) async fn scope_turn<F: std::future::Future>(
    previous_request: PreviousRequestSlot,
    future: F,
) -> F::Output {
    let current = attribution();
    ATTRIBUTION
        .scope(
            Attribution {
                prompt_id: Some(current.prompt_id.unwrap_or_else(Uuid::new_v4)),
                previous_request: Some(previous_request),
                ..current
            },
            future,
        )
        .await
}

/// Carry the current attribution into a `tokio::spawn`.
///
/// A task-local does not cross a spawn, and the streaming path spawns the provider call. Without
/// this pair every streamed request goes out with no prompt id and no previous response, which is
/// indistinguishable from a side query.
///
/// `subagent` rides along for a reason that is about the future rather than today: sub-agents run
/// with `streaming: false` and reach the provider through the awaited `complete`, so their flag has
/// never crossed a spawn. Carrying all three together means the day one of them does stream, it
/// does not quietly bill as the main agent.
///
/// Capture on the calling side, re-establish as the first thing the spawned future does.
pub(crate) fn capture_attribution() -> Attribution {
    attribution()
}

pub(crate) async fn scope_attribution<F: std::future::Future>(
    captured: Attribution,
    future: F,
) -> F::Output {
    ATTRIBUTION.scope(captured, future).await
}

/// Strip trailing slashes so a provider can append its own path with a leading `/`.
///
/// Every backend builds request URLs as `format!("{base}/some/path")`, so a base the user pasted
/// with a trailing slash would otherwise produce a doubled separator (`https://host//v1/messages`).
/// Servers are not obliged to treat that as the same route, and the ones that don't return a 404
/// that names nothing.
pub(crate) fn normalize_base_url(url: &str) -> String {
    let normalized = url.trim().trim_end_matches('/');
    if normalized != url {
        tracing::debug!("normalized provider base URL '{}' to '{}'", url, normalized);
    }
    normalized.to_string()
}

#[derive(Clone, Serialize, Deserialize)]
pub enum AuthCredential {
    ApiKey(String),
    OAuthToken {
        access_token: String,
        refresh_token: Option<String>,
        expires_at: Option<i64>,
        /// Provider-flavoured identity carried alongside the bearer token. Currently only
        /// `chatgpt-subscription` populates this, the `chatgpt_account_id` extracted from the
        /// id_token JWT, sent on every request as `ChatGPT-Account-ID`. Claude OAuth
        /// leaves it `None`.
        account_id: Option<String>,
    },
}

/// Hand-written so a credential cannot reach a log through a `{:?}` on any struct that holds one.
///
/// The derived impl printed the bearer token verbatim, and a provider struct is exactly the kind of
/// thing that ends up inside a `tracing::debug!` or an error's `{:?}` during a bad afternoon.
/// Lengths are kept because they are what a "wrong key pasted" diagnosis actually needs.
impl std::fmt::Debug for AuthCredential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiKey(key) => f
                .debug_tuple("ApiKey")
                .field(&format_args!("[REDACTED len={}]", key.len()))
                .finish(),
            Self::OAuthToken {
                access_token,
                refresh_token,
                expires_at,
                account_id,
            } => f
                .debug_struct("OAuthToken")
                .field(
                    "access_token",
                    &format_args!("[REDACTED len={}]", access_token.len()),
                )
                .field(
                    "refresh_token",
                    &refresh_token
                        .as_ref()
                        .map(|token| format_args!("[REDACTED len={}]", token.len()).to_string()),
                )
                .field("expires_at", expires_at)
                .field("account_id", account_id)
                .finish(),
        }
    }
}

impl AuthCredential {
    pub fn auth_header(&self) -> (&'static str, String) {
        match self {
            AuthCredential::ApiKey(key) => ("x-api-key", key.clone()),
            AuthCredential::OAuthToken { access_token, .. } => {
                ("Authorization", format!("Bearer {}", access_token))
            }
        }
    }
}

/// Normalized account rate-limit usage, as returned by a subscription provider's usage endpoint
/// (Claude OAuth's `/api/oauth/usage`, Codex's `/wham/usage`). Provider-agnostic so one renderer
/// serves every backend; providers map their native shapes into [`UsageWindow`]s.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AccountUsage {
    /// Rolling rate-limit windows (e.g. the 5-hour session window and the weekly window), in the
    /// order the provider reported them.
    pub windows: Vec<UsageWindow>,
    /// Pay-as-you-go / extra-usage (overage credit) state, when the provider reports it.
    pub extra_usage: Option<ExtraUsage>,
    /// Optional one-line addendum (e.g. the plan name) shown beneath the windows. `None` when the
    /// provider offered nothing extra.
    pub note: Option<String>,
}

/// Whether a Claude request asks for extended thinking, and which wire encoding it uses.
///
/// One knob rather than two. Thinking used to be a global on/off plus a shape meka inferred from
/// the model name, and that inference is what this replaces: `anthropic-messages` reaches any
/// Anthropic-compatible endpoint, so meka cannot tell which encoding the far side implements. The
/// profile states it, and is the user's to keep correct if they later change `model`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingMode {
    /// The model sets its own budget. Claude 4.6 and newer.
    ///
    /// Sends the adaptive thinking block, and is the default because it is what the current models
    /// want. `--thinking` hides its possible-values list to keep `meka -h` to one line, so these
    /// variant docs no longer reach a terminal and that flag's own help summarises the three modes;
    /// the wire shapes each mode produces live in `anthropic::shared::insert_thinking_fields`.
    #[default]
    Adaptive,
    /// A fixed budget, from the thinking budget_tokens setting. Required by pre-4.6 Claude.
    ///
    /// The older encoding, and the one most third-party Anthropic-compatible servers implement.
    Budgeted,
    /// No thinking requested.
    Off,
}

impl ThinkingMode {
    /// Whether the request asks for thinking in any encoding. The betas and the `temperature` gate
    /// key off this rather than off a specific encoding.
    pub fn is_on(self) -> bool {
        !matches!(self, ThinkingMode::Off)
    }

    /// The one spelling of each mode, for anything that writes or displays it: the TOML `meka
    /// provider add` emits, and the `/status` block. Both used to carry their own copy of this
    /// match, which is two hand-written mappings that have to agree with each other, with serde's
    /// `rename_all` on the way back in, and with the values clap accepts on the way in from the CLI
    /// - four derivations of the same three strings, with nothing checking them against each other.
    pub fn as_str(self) -> &'static str {
        match self {
            ThinkingMode::Adaptive => "adaptive",
            ThinkingMode::Budgeted => "budgeted",
            ThinkingMode::Off => "off",
        }
    }
}

/// The shared reasoning-effort policy, applied identically by every provider that exposes an effort
/// knob (Claude `output_config.effort`, OpenAI `reasoning.effort`).
///
/// Effort is a request parameter the *provider* owns: leaving the field off is not a degraded
/// setting, it is how you ask for the provider's own default. meka therefore sends it only when the
/// profile asks for one. A configured value is passed through verbatim (trimmed + lowercased) and
/// is **absolute** - never clamped, never dropped, whatever model it is aimed at; the user owns
/// correctness for their model and endpoint. A blank value (empty or whitespace-only) reads as
/// unset.
///
/// meka deliberately picks no default of its own. It cannot know what tiers a given endpoint
/// implements - `anthropic-messages` and `openai-chat-completions` reach any compatible server,
/// including local ones serving weights that never had an effort knob - and a tier the backend does
/// not implement is a rejected request rather than a graceful ignore.
pub(crate) fn resolve_effort_level(configured: Option<&str>) -> Option<String> {
    configured
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
}

/// Pay-as-you-go / extra-usage (overage credits) state. Normalized from Anthropic's `extra_usage` +
/// `spend` blocks and Codex's `credits` + `spend_control` blocks; every numeric field is optional
/// because the two providers report different subsets.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExtraUsage {
    /// Whether extra usage / pay-as-you-go is enabled on the account.
    pub enabled: bool,
    /// Percent of the extra-usage / spend limit consumed (`0.0..=100.0`), if reported.
    pub utilization: Option<f64>,
    /// Amount spent this period, in `currency`, if reported.
    pub used: Option<f64>,
    /// Remaining credit balance, in `currency`, if reported.
    pub balance: Option<f64>,
    /// Currency code (e.g. `"USD"`); `None` when the provider didn't say.
    pub currency: Option<String>,
}

/// A single rolling usage window.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UsageWindow {
    /// Human label, e.g. `"5-hour (session)"` or `"Weekly"`.
    pub label: String,
    /// Percentage of the window consumed, `0.0..=100.0`.
    pub used_percent: f64,
    /// When the window resets, as a Unix timestamp in seconds. `None` if the provider didn't say.
    pub resets_at: Option<i64>,
}

/// Normalized account identity, from a subscription provider's profile endpoint (Claude OAuth's
/// `/api/oauth/profile` + `/api/oauth/claude_cli/roles`, Codex's `plan_type`). Every field is
/// optional so each backend fills what it can. Serialized as the `identity` block of `meka account
/// whoami --format json`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AccountIdentity {
    pub display_name: Option<String>,
    pub email: Option<String>,
    /// Plan label, e.g. `"claude_max"`, `"pro"`, `"plus"`.
    pub plan: Option<String>,
    /// Rate-limit tier, e.g. `"default_claude_max_20x"`.
    pub tier: Option<String>,
    pub subscription_status: Option<String>,
    pub organization: Option<String>,
    /// Organization role, e.g. `"admin"`.
    pub role: Option<String>,
}

/// Normalized historical usage, from a provider's stats endpoint (Codex's `/wham/profiles/me`,
/// Claude's `/api/organization/claude_code_first_token_date`). Fields are optional because the
/// providers report very different amounts: Codex is rich (lifetime/daily/streaks), Claude offers
/// only a first-used date.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UsageHistory {
    pub lifetime_tokens: Option<i64>,
    pub peak_daily_tokens: Option<i64>,
    pub current_streak_days: Option<i64>,
    pub longest_streak_days: Option<i64>,
    /// When the account first used the tool (RFC 3339 or `YYYY-MM-DD`), if known.
    pub first_used: Option<String>,
    /// Per-day token counts, in the order the provider returned them.
    pub daily: Vec<DailyUsage>,
}

/// One day's token count in [`UsageHistory::daily`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DailyUsage {
    pub date: String,
    pub tokens: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContent {
    Text { text: String },
    Image { source: ImageSource },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

/// The half of a thinking block meka cannot read, and which provider it belongs to.
///
/// The two backends that emit reasoning hand back different things, and the difference decides both
/// what may be replayed and what the readable half is worth. Holding them as one nullable
/// `signature` made wrong states representable, and one of them shipped: `chatgpt-subscription`
/// stored OpenAI's `encrypted_content` in that field, and resuming such a session under Claude
/// replayed it verbatim as Claude's `signature`, a blob from the wrong cryptosystem presented as
/// authentication for text it does not authenticate.
///
/// The mirror of that was only ever unreachable by omission: the Responses encoder dropped every
/// thinking block, so nothing Claude wrote could reach OpenAI. This release starts replaying
/// reasoning there, which is exactly what would have opened the other direction. Naming the two
/// shapes makes both a type error instead of a thing to remember.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OpaqueReasoning {
    /// Anthropic. The reasoning is in the block's `thinking` text; this authenticates it, and the
    /// API wants both back together.
    Signed { signature: String },
    /// The Responses API. This *is* the reasoning, sealed, and the block's `thinking` holds only
    /// the summary the server chose to show. Replayed under `id` when the server issued one.
    Sealed {
        encrypted_content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    /// Image supplied as *input* (e.g. an ACP client's @-mention or pasted screenshot). Distinct
    /// from a tool result's image, which travels inside [`ContentBlock::ToolResult`] as a
    /// [`ToolResultContent::Image`].
    Image {
        source: ImageSource,
    },
    Thinking {
        /// The readable half, and only the readable half. How much of the reasoning it is depends
        /// on `opaque`: under [`OpaqueReasoning::Signed`] this is the reasoning, under
        /// [`OpaqueReasoning::Sealed`] it is a summary of reasoning kept elsewhere.
        thinking: String,
        /// What the provider wants handed back to carry this reasoning into the next request.
        /// `None` when it gave nothing to carry, which makes the block display-only.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        opaque: Option<OpaqueReasoning>,
    },
    /// Encrypted reasoning the API declines to return in the clear (the `redact-thinking` beta).
    /// `data` is opaque: it cannot be read, only replayed verbatim on later turns so the model can
    /// continue its prior reasoning chain. Distinct from a [`ContentBlock::Thinking`] with empty
    /// text, which carries a `signature` instead of `data`.
    RedactedThinking {
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: Vec<ToolResultContent>,
        is_error: bool,
    },
}

impl ContentBlock {
    /// Extract the text content of a ToolResult (for display/logging).
    pub fn tool_result_text_content(content: &[ToolResultContent]) -> String {
        content
            .iter()
            .map(|block| match block {
                ToolResultContent::Text { text } => text.as_str(),
                ToolResultContent::Image { .. } => "[Image]",
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    /// User message carrying a text block followed by zero or more input images. Used by the ACP
    /// prompt path when the client attaches images; `images` empty yields the same shape as
    /// [`Message::user`].
    pub fn user_with_images(text: impl Into<String>, images: Vec<ImageSource>) -> Self {
        let mut content = vec![ContentBlock::Text { text: text.into() }];
        content.extend(
            images
                .into_iter()
                .map(|source| ContentBlock::Image { source }),
        );
        Self {
            role: Role::User,
            content,
        }
    }

    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// A copy of this message with every [`ContentBlock::ToolUse`] removed. Used when persisting a
    /// turn that was interrupted before its tools ran: keeping the `tool_use` blocks would orphan
    /// them (no matching `tool_result`) and the provider would reject the next request.
    pub fn without_tool_use(&self) -> Message {
        Message {
            role: self.role.clone(),
            content: self
                .content
                .iter()
                .filter(|block| !matches!(block, ContentBlock::ToolUse { .. }))
                .cloned()
                .collect(),
        }
    }

    #[cfg(test)]
    pub fn tool_uses(&self) -> Vec<&ContentBlock> {
        self.content
            .iter()
            .filter(|block| matches!(block, ContentBlock::ToolUse { .. }))
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    /// Human-readable title for the tool, optionally set by MCP servers. Providers may render this
    /// in UIs instead of the machine name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// MCP `tool.annotations`: hints such as `readOnlyHint`, `destructiveHint`, `openWorldHint`.
    /// Passed through verbatim as JSON; providers that don't recognise the field ignore it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<serde_json::Value>,
    /// MCP `tool.meta` payload, forwarded verbatim so permission heuristics and audit logs can
    /// access it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<serde_json::Value>,
}

#[cfg(test)]
impl ToolDefinition {
    /// Test-only convenience constructor. Production code builds `ToolDefinition` as a struct
    /// literal and explicitly sets the MCP-specific `title`/`annotations`/`meta` fields; this
    /// helper just keeps test fixtures terse.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            title: None,
            annotations: None,
            meta: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    TextDelta(String),
    ThinkingDelta(String),
    /// The model has entered a thinking block, carrying the server's running estimate of the
    /// thinking tokens spent so far when it offers one (`None` at the start of the block, before
    /// any estimate has arrived).
    ///
    /// Separate from [`Self::ThinkingDelta`] because thinking can be *silent*: under Claude's
    /// `redact-thinking` beta every delta carries an empty string, so a UI driven by deltas alone
    /// shows nothing at all for the whole reasoning phase. This event is the liveness signal, and
    /// the estimate is what distinguishes "still working" from "wedged".
    ThinkingProgress {
        estimated_tokens: Option<u64>,
    },
    ThinkingComplete {
        /// Whatever the provider gave to carry this reasoning forward, in its own shape. Passed
        /// through to [`ContentBlock::Thinking`] untouched.
        opaque: Option<OpaqueReasoning>,
    },
    /// A complete `redacted_thinking` block (the `redact-thinking` beta). `data` is opaque and
    /// arrives whole in the `content_block_start` event, so there is no delta/complete pair.
    RedactedThinking {
        data: String,
    },
    ToolUseStart {
        id: String,
        name: String,
    },
    ToolInputDelta(String),
    ToolUseEnd {
        input: serde_json::Value,
    },
    /// Emitted in lieu of `ToolUseEnd` when the accumulated tool-call arguments fail to parse as
    /// JSON. The agent layer must not execute the tool; it should surface the parse error back to
    /// the model as a `ToolResult { is_error: true }` instead.
    ToolCallRejected {
        id: String,
        name: String,
        reason: String,
    },
    MessageEnd {
        stop_reason: StopReason,
    },
    Usage(TokenUsage),
    /// User-visible advisory from the provider layer (e.g. "redacted N old images to fit the
    /// 32 MiB request limit"). The agent translates this into
    /// [`crate::frontend::FrontendEvent::Notice`] so every frontend renders it consistently.
    /// Distinct from `Error`: the request itself is proceeding successfully; the notice
    /// describes a side-effect the user should know about.
    Notice(Notice),
    Error(String),
}

/// Severity hint for a provider-emitted [`Notice`]. Frontends can map these to per-level styling
/// (a dim hint for `Info`, a warn-colored line for `Warn`). `Info` carries the image-redaction
/// notice; `Warn` carries recoverable conditions the user should see.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoticeLevel {
    Info,
    Warn,
}

/// User-visible advisory surfaced by a provider during a request. Carries no structured data
/// beyond the level and the message; frontends format it themselves.
#[derive(Debug, Clone)]
pub struct Notice {
    pub level: NoticeLevel,
    pub text: String,
}

impl Notice {
    pub fn info(text: impl Into<String>) -> Self {
        Self {
            level: NoticeLevel::Info,
            text: text.into(),
        }
    }

    pub fn warn(text: impl Into<String>) -> Self {
        Self {
            level: NoticeLevel::Warn,
            text: text.into(),
        }
    }
}

/// Sentinel key inserted into `ToolUse::input` when the upstream tool-call arguments failed to
/// parse. `resolve_and_execute_tool` checks for this and short-circuits to an error result instead
/// of invoking the tool with a potentially surprising default-filled object.
pub(crate) const INVALID_TOOL_ARGS_MARKER: &str = "_meka_invalid_arguments";

#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Tokens billed at the cache-write tier (content newly cached this turn). Anthropic-only;
    /// OpenAI providers leave this at 0.
    pub cache_creation_input_tokens: u64,
    /// Tokens served from the prompt cache (cache-read tier). Anthropic returns this in
    /// `usage.cache_read_input_tokens`; OpenAI providers leave it at 0 today.
    pub cache_read_input_tokens: u64,
}

impl TokenUsage {
    /// Fold a streamed usage update into the running per-round total, taking each field from
    /// `update` only when it is non-zero. Providers split usage across events: Anthropic reports
    /// the input/cache tiers on `message_start` and the final `output_tokens` on
    /// `message_delta` (the other fields absent, i.e. parsed as 0), while OpenAI/Codex send a
    /// single usage event. The non-zero rule keeps the `message_start` input/cache values
    /// instead of letting a later event that omits them clobber the count back to 0.
    pub fn merge_stream(&mut self, update: &TokenUsage) {
        if update.input_tokens > 0 {
            self.input_tokens = update.input_tokens;
        }
        if update.output_tokens > 0 {
            self.output_tokens = update.output_tokens;
        }
        if update.cache_creation_input_tokens > 0 {
            self.cache_creation_input_tokens = update.cache_creation_input_tokens;
        }
        if update.cache_read_input_tokens > 0 {
            self.cache_read_input_tokens = update.cache_read_input_tokens;
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    /// The model declined to comply with the request. Claude's API surfaces this as `stop_reason:
    /// "refusal"`; OpenAI's responses API has the equivalent. The string carries the model's
    /// refusal text when the provider includes one, empty otherwise.
    Refusal(String),
    Unknown(String),
}

/// Abstraction over an LLM provider backend, each named for the wire protocol it speaks or the
/// account it bills (see the module header). Implementors are held behind
/// `Arc<dyn Provider>` and shared across concurrent tool dispatch; calls must be safe to make in
/// parallel from multiple sub-agents in one turn.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Single round-trip request. Returns the assistant message, stop-reason, token-usage metadata,
    /// and any user-visible notices that arose during the request (e.g. the redaction hint from
    /// `anthropic::shared::build_body_within_budget`). The caller is expected to forward each
    /// notice to the active frontend; an empty `Vec` means nothing to surface. No streaming;
    /// the agent awaits the full response.
    async fn complete(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<(Message, StopReason, TokenUsage, Vec<Notice>)>;

    /// Streaming variant. The provider pushes `StreamEvent`s onto `event_sender` as they arrive.
    /// Cancellation is observed via `cancellation`; implementors must check the token and abort
    /// in-flight HTTP work when it fires.
    async fn stream(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        event_sender: mpsc::Sender<StreamEvent>,
        cancellation: CancellationToken,
    ) -> Result<()>;

    #[allow(dead_code)]
    fn name(&self) -> &str;

    /// Suppress thinking for calls made while the flag is set, then clear it. Used for meka's own
    /// internal turns (compaction), which gain nothing from reasoning. It can only turn thinking
    /// off - a profile that configured [`ThinkingMode::Off`] stays off - so the two settings cannot
    /// disagree about whether a request asks for thinking. Default impl is a silent no-op; the
    /// Claude providers override.
    fn suppress_thinking(&self, _suppressed: bool) {}

    /// Fetch the account's current rate-limit usage (session / weekly windows and reset times).
    /// Returns `Ok(None)` for providers that have no per-account usage endpoint (API-key backends,
    /// Ollama, the local CLI); OAuth subscription providers override this. Errors propagate so the
    /// caller can surface a refresh/auth failure rather than silently showing nothing.
    async fn fetch_usage(&self) -> Result<Option<AccountUsage>> {
        Ok(None)
    }

    /// Fetch the account's identity (display name, plan/tier, organization, role). Same `Ok(None)`
    /// contract as [`Self::fetch_usage`]: only OAuth subscription providers override it.
    async fn fetch_identity(&self) -> Result<Option<AccountIdentity>> {
        Ok(None)
    }

    /// Fetch the account's historical usage (lifetime tokens, streaks, per-day counts, first-used
    /// date). Same `Ok(None)` contract as [`Self::fetch_usage`].
    async fn fetch_history(&self) -> Result<Option<UsageHistory>> {
        Ok(None)
    }

    /// The reasoning-effort value this provider will send on the wire (its settled
    /// `output_config.effort` / `reasoning.effort`), or `None` when it sends none. For display only
    /// (the `/status` model block); the request path reads the same settled slot. Default: `None`
    /// (providers with no effort knob).
    fn resolved_effort(&self) -> Option<String> {
        None
    }
}

struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

async fn finalize_tool_call_accumulators(
    accumulators: &mut std::collections::HashMap<i64, ToolCallAccumulator>,
    event_sender: &mpsc::Sender<StreamEvent>,
) -> bool {
    let has_tools = !accumulators.is_empty();
    let mut indices: Vec<i64> = accumulators.keys().copied().collect();
    indices.sort();
    for index in indices {
        if let Some(accumulator) = accumulators.remove(&index) {
            if event_sender
                .send(StreamEvent::ToolUseStart {
                    id: accumulator.id.clone(),
                    name: accumulator.name.clone(),
                })
                .await
                .is_err()
            {
                tracing::trace!("stream event receiver dropped");
                return has_tools;
            }
            // A zero-argument tool call legitimately streams `"arguments": ""`, and
            // `handle_stream_chunk` only appends non-empty argument fragments, so the accumulator
            // for such a call is an empty string rather than `"{}"`. Parsing that as JSON fails,
            // and rejecting it would refuse a call the model made correctly -- exactly the
            // "agent's intent discarded" failure the rejection path exists to prevent, inverted.
            // The Claude driver has carried this carve-out since the rejection was introduced;
            // this is the sibling that did not get it.
            let arguments = if accumulator.arguments.trim().is_empty() {
                "{}"
            } else {
                accumulator.arguments.as_str()
            };
            match serde_json::from_str::<serde_json::Value>(arguments) {
                Ok(value) => {
                    if event_sender
                        .send(StreamEvent::ToolUseEnd { input: value })
                        .await
                        .is_err()
                    {
                        tracing::trace!("stream event receiver dropped");
                        return has_tools;
                    }
                }
                Err(error) => {
                    tracing::warn!(
                        tool = %accumulator.name,
                        "rejecting tool call with unparseable JSON arguments: {}",
                        error
                    );
                    if event_sender
                        .send(StreamEvent::ToolCallRejected {
                            id: accumulator.id.clone(),
                            name: accumulator.name.clone(),
                            reason: format!("invalid JSON arguments: {}", error),
                        })
                        .await
                        .is_err()
                    {
                        tracing::trace!("stream event receiver dropped");
                        return has_tools;
                    }
                }
            }
        }
    }
    has_tools
}

/// Constructs a concrete [`Provider`] for any name in [`SUPPORTED_PROVIDERS`] from a bag
/// of provider-specific settings. Each setter documents which provider(s) consume it; unused
/// settings are silently ignored by providers that don't need them. The only required inputs are
/// the provider name, the credential, and the model; everything else has a sensible default.
pub struct ProviderBuilder {
    provider_name: String,
    credential: AuthCredential,
    model: String,
    base_url: Option<String>,
    client_id: Option<String>,
    oauth_token_url: Option<String>,
    token_store: Option<Arc<TokenStore>>,
    /// Profile name the credential is stored under; OAuth providers use it to write refreshed
    /// tokens back to the right `provider_credentials` row. Defaults to the backend name.
    credential_key: Option<String>,
    thinking: ThinkingMode,
    thinking_budget_tokens: u64,
    device_id: String,
    effort: Option<String>,
    redact_thinking: bool,
    max_output_tokens: Option<u64>,
    session_stats: Option<Arc<crate::stats::SessionStats>>,
}

impl ProviderBuilder {
    pub fn new(
        provider_name: impl Into<String>,
        credential: AuthCredential,
        model: impl Into<String>,
    ) -> Self {
        Self {
            provider_name: provider_name.into(),
            credential,
            model: model.into(),
            base_url: None,
            client_id: None,
            oauth_token_url: None,
            token_store: None,
            credential_key: None,
            thinking: ThinkingMode::Off,
            thinking_budget_tokens: 0,
            device_id: String::new(),
            effort: None,
            redact_thinking: true,
            max_output_tokens: None,
            session_stats: None,
        }
    }

    /// Override the HTTP endpoint. Applies to every provider variant; defaults to the Claude or
    /// OpenAI production URL.
    pub fn base_url(mut self, value: Option<String>) -> Self {
        self.base_url = value;
        self
    }

    /// OAuth client ID. Only consumed by `claude-subscription`.
    pub fn client_id(mut self, value: Option<String>) -> Self {
        self.client_id = value;
        self
    }

    /// OAuth token endpoint. Only consumed by `claude-subscription`.
    pub fn oauth_token_url(mut self, value: Option<String>) -> Self {
        self.oauth_token_url = value;
        self
    }

    /// Sink for refreshed OAuth tokens. Only consumed by `claude-subscription`; when `None`,
    /// refreshed tokens are held in memory only.
    pub fn token_store(mut self, value: Option<Arc<TokenStore>>) -> Self {
        self.token_store = value;
        self
    }

    /// Profile name the credential is stored under (OAuth refresh write-back key). Defaults to the
    /// backend name when unset.
    pub fn credential_key(mut self, value: Option<String>) -> Self {
        self.credential_key = value;
        self
    }

    /// Claude-only: the profile's thinking mode, plus the budget cap [`ThinkingMode::Budgeted`]
    /// uses. Ignored by the OpenAI backends.
    pub fn thinking(mut self, mode: ThinkingMode, budget_tokens: u64) -> Self {
        self.thinking = mode;
        self.thinking_budget_tokens = budget_tokens;
        self
    }

    /// Stable device identity embedded in `metadata.user_id`. Only consumed by
    /// `claude-subscription`.
    pub fn device_id(mut self, value: String) -> Self {
        self.device_id = value;
        self
    }

    /// The user's explicit reasoning-effort override (`low` / `medium` / `high` / `xhigh` / `max`),
    /// or `None` to leave the field off, so the provider applies its own. Consumed by every
    /// backend: Claude maps it to `output_config.effort`, OpenAI to `reasoning.effort`.
    pub fn effort(mut self, value: Option<String>) -> Self {
        self.effort = value;
        self
    }

    /// Request `redacted_thinking` blocks. Only consumed by `claude-subscription`.
    pub fn redact_thinking(mut self, value: bool) -> Self {
        self.redact_thinking = value;
        self
    }

    /// Per-request output (completion) token cap. When `None`, each backend keeps its built-in
    /// default. Consumed by every backend.
    pub fn max_output_tokens(mut self, value: Option<u64>) -> Self {
        self.max_output_tokens = value;
        self
    }

    /// Per-session counters incremented when image-redaction events fire. Currently consumed only
    /// by `claude-subscription` and `anthropic-messages`.
    pub fn session_stats(mut self, value: Option<Arc<crate::stats::SessionStats>>) -> Self {
        self.session_stats = value;
        self
    }

    pub fn build(self) -> Result<Arc<dyn Provider>> {
        match self.provider_name.as_str() {
            "openai-responses" => {
                // Same credential handling as `openai-chat-completions`: these two differ by
                // protocol, not by how they authenticate.
                let api_key = match self.credential {
                    AuthCredential::ApiKey(key) => key,
                    AuthCredential::OAuthToken { access_token, .. } => access_token,
                };
                Ok(Arc::new(OpenAiResponsesProvider::new(
                    api_key,
                    self.model,
                    self.base_url,
                    self.effort,
                    self.max_output_tokens,
                )?))
            }
            "openai-chat-completions" => {
                let api_key = match self.credential {
                    AuthCredential::ApiKey(key) => key,
                    AuthCredential::OAuthToken { access_token, .. } => access_token,
                };
                Ok(Arc::new(OpenAiChatCompletionsProvider::new(
                    api_key,
                    self.model,
                    self.base_url,
                    self.effort,
                    self.max_output_tokens,
                )?))
            }
            "anthropic-messages" => {
                let api_key = match self.credential {
                    AuthCredential::ApiKey(key) => key,
                    AuthCredential::OAuthToken { .. } => {
                        return Err(MekaError::Config(
                            "provider 'anthropic-messages' requires an API key, not an OAuth \
                             token. Use 'claude-subscription' to bill a Claude subscription."
                                .to_string(),
                        ));
                    }
                };
                Ok(Arc::new(AnthropicMessagesProvider::new(
                    api_key,
                    self.model,
                    self.base_url,
                    self.thinking,
                    self.thinking_budget_tokens,
                    self.effort,
                    self.max_output_tokens,
                    self.session_stats,
                )?))
            }
            "claude-subscription" => {
                if matches!(self.credential, AuthCredential::ApiKey(_)) {
                    return Err(MekaError::Config(
                        "provider 'claude-subscription' requires an OAuth token, not an API key. \
                         Use 'anthropic-messages' to bill an Anthropic API key."
                            .to_string(),
                    ));
                }
                Ok(Arc::new(ClaudeSubscriptionProvider::new(
                    self.credential,
                    self.model,
                    self.base_url,
                    self.client_id,
                    self.oauth_token_url,
                    self.token_store,
                    self.credential_key
                        .unwrap_or_else(|| self.provider_name.clone()),
                    self.thinking,
                    self.thinking_budget_tokens,
                    self.device_id,
                    self.effort,
                    self.redact_thinking,
                    self.max_output_tokens,
                    self.session_stats,
                )?))
            }
            "chatgpt-subscription" => {
                if matches!(self.credential, AuthCredential::ApiKey(_)) {
                    return Err(MekaError::Config(
                        "provider 'chatgpt-subscription' requires an OAuth token, not an API key. \
                         Use 'openai-responses' for the same protocol with an API key, or \
                         'openai-chat-completions' for an endpoint that serves only that."
                            .to_string(),
                    ));
                }
                Ok(Arc::new(ChatGptSubscriptionProvider::new(
                    self.credential,
                    self.model,
                    self.base_url,
                    self.client_id,
                    self.oauth_token_url,
                    self.token_store,
                    self.credential_key
                        .unwrap_or_else(|| self.provider_name.clone()),
                    self.effort,
                    self.max_output_tokens,
                )?))
            }
            other => Err(MekaError::Config(format!(
                "unknown provider: '{}'. Supported providers: {}",
                other,
                SUPPORTED_PROVIDERS.join(", ")
            ))),
        }
    }
}

/// Find a profile by the name a session recorded, or say why it cannot be found.
///
/// **Refuses rather than falling back to the default.** A session names the profile it runs with,
/// and quietly running the conversation somewhere else is exactly the failure this arrangement
/// exists to prevent: the reasoning would not replay and a different account would bill. The
/// configured names are listed because the recorded one is usually a rename or a typo away from a
/// real one, and a refusal that does not say what *is* available makes the user go and look.
fn look_up_profile<'a>(
    profiles: &'a std::collections::BTreeMap<String, crate::config::ProviderProfile>,
    profile: &str,
) -> Result<&'a crate::config::ProviderProfile> {
    profiles.get(profile).ok_or_else(|| {
        // "Move the session to one of those" rather than a flag. This refusal reaches the CLI, the
        // HTTP API and ACP verbatim, and each has its own way to do it -- `--provider` on a resume,
        // `PATCH /v1/sessions/{id}`, the provider picker. Naming only the CLI's told two of the
        // three callers to reach for something they do not have.
        //
        // Unconditional on the recorded name, including the empty one a store carried forward from
        // before this column can hold. That row is the migration's to explain, and it does, at the
        // moment it creates it; a branch here would be this reader knowing that an older meka
        // wrote the store, which is the one thing every module but the ledger is meant not to.
        // Every door that mints a session resolves a profile first and refuses without one, so
        // there is no such thing as a session this meka created with none.
        MekaError::Config(if profiles.is_empty() {
            format!(
                "this session runs on provider profile '{}', and no profiles are configured. Run \
                 `meka provider add <name>` to set one up",
                profile
            )
        } else {
            format!(
                "this session runs on provider profile '{}', which is not configured. Configured \
                 profiles: {}. Add it back to config.toml, or move the session to one of those",
                profile,
                profiles.keys().cloned().collect::<Vec<_>>().join(", ")
            )
        })
    })
}

/// What distinguishes one built provider from another.
///
/// The profile name alone is not enough: a session may override the model or the endpoint, and two
/// sessions on one profile with different overrides are talking to different things. Everything
/// else a profile states is fixed for that name, so it needs no place here.
///
/// The credential is deliberately *not* part of this, even though the provider is built from one.
/// See [`CachedProvider`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ProviderKey {
    profile: String,
    model: Option<String>,
    base_url: Option<String>,
    thinking: ThinkingMode,
}

/// A provider kept for reuse, beside the credential it was built from.
///
/// The credential is a *tag* rather than part of [`ProviderKey`] because it is the one input to a
/// build that supersedes its predecessor rather than distinguishing a sibling. Keyed, every
/// rotation would mint an entry and none would ever be dropped, so a long-lived `meka serve`
/// against an OAuth profile -- which rewrites the credential on every token refresh -- would
/// accumulate one `reqwest` client and its connection pool per hour, forever. Tagged, there is
/// exactly one entry per key and a rotation replaces it.
///
/// [`TokenStore::provider_credential_version`] rather than the credential itself: the comparison
/// wants to know *whether* the credential moved, and nothing here needs a secret in memory to
/// answer that.
///
/// **What this reaches, and what it does not.** It reaches every later `build()`, which is every
/// session assembled or re-attached after the rotation. It does not reach a session already
/// resident: that one holds its own `Arc<dyn Provider>` on [`crate::agent::Agent`] until
/// `set_provider` replaces it, so it keeps presenting the old credential until it is evicted. That
/// is a property of resolving the binding once per session rather than once per turn, which is a
/// separate decision from this memo; it is documented for users at
/// `docs/book/src/configuration/overview.md` § "When edits take effect".
struct CachedProvider {
    /// What [`TokenStore::provider_credential_version`] said *before* this provider's credential
    /// was loaded. That order matters and is the fail-safe direction: read after, a rotation
    /// landing between the two reads would tag a provider built from the old credential with
    /// the new version and pin it forever. Read before, the same race tags a provider built
    /// from the new credential with the old version, and costs one needless rebuild that then
    /// converges.
    credential_version: Option<String>,
    provider: Arc<dyn Provider>,
}

/// Providers, built on demand for whichever profile is asked for and kept for reuse.
///
/// Replaces the single `Arc<dyn Provider>` that used to be built once at startup and shared by
/// every session. That was right while the profile was a property of the process; it is wrong now
/// that a session records the one it runs with, because two sessions in one `meka serve` may name
/// different profiles and both have to work.
///
/// Reuse is the reason this caches rather than building per turn: an `Arc<dyn Provider>` owns a
/// `reqwest::Client` and therefore a connection pool, and rebuilding one per turn would throw away
/// every kept-alive connection. Under the old single-provider design that sharing was automatic.
///
/// **A profile's credential is now checked at first use, not at startup.** Only the process default
/// could be validated up front, and validating every configured profile would make an unused one
/// with a stale token fail a launch it has nothing to do with. The cost is that a typo in a profile
/// a session names shows up when that session runs.
///
/// Reuse does not extend to a credential that has moved: see [`CachedProvider`].
pub struct ProviderRegistry {
    /// `config.toml`'s `[providers.*]`, as they stood when this process started.
    ///
    /// A snapshot on purpose, and the one place in this file where that is the right answer.
    /// `config.toml` is hand-edited and has no change feed; re-reading it per request would mean a
    /// half-written file could take a running server's sessions down, and would make which account
    /// a turn bills depend on when the turn happened to run. So a `meka provider add` while
    /// `meka serve` is up is invisible to it until restart, and `POST /v1/sessions` answers "not
    /// configured" about a profile `meka provider list` shows. Documented for users at
    /// `docs/book/src/configuration/overview.md` § "When edits take effect"; the credential behind
    /// a profile is *not* snapshotted (see [`CachedProvider`]), because that one has a writer meka
    /// owns.
    profiles: std::collections::BTreeMap<String, crate::config::ProviderProfile>,
    /// `[session].context_window`, which a profile's own value takes precedence over.
    session_context_window: Option<u64>,
    /// `[thinking].budget_tokens`. A profile states the thinking *mode*; the budget is one setting
    /// for the process, so it is held here rather than resolved per profile.
    thinking_budget_tokens: u64,
    /// `--thinking` for this run, which outranks whatever profile a session resolves to. Held
    /// beside the budget because both are properties of the run rather than of a profile, and it
    /// has to be applied here: a session's own overrides carry a model and an endpoint, and the
    /// flag is neither.
    requested_thinking: Option<ThinkingMode>,
    /// Device ids, resolved at most once per profile.
    ///
    /// `claude-subscription` mints one and writes it into `config.toml` when the profile states
    /// none, and `profiles` above is a snapshot taken when this was built, so a resolver reached
    /// per request never saw the value it had just persisted. It minted another, and rewrote the
    /// config, on every session create, context poll and status query -- for an identifier whose
    /// entire purpose is to stay the same. Lazy rather than filled in `new` so a profile this
    /// process never uses is never seeded.
    device_ids: std::sync::Mutex<std::collections::HashMap<String, String>>,
    token_store: Arc<TokenStore>,
    session_stats: Arc<crate::stats::SessionStats>,
    built: std::sync::Mutex<std::collections::HashMap<ProviderKey, CachedProvider>>,
    /// Debug-only: a scripted provider that stands in for every profile.
    ///
    /// Set by the three hosts when `MEKA_MOCK_PROVIDER=1`, replacing the older arrangement where
    /// each rebuilt its shared state around a mock. One override here covers every profile, which
    /// is what a harness driving a session on any profile actually wants.
    #[cfg(debug_assertions)]
    scripted: std::sync::Mutex<Option<Arc<dyn Provider>>>,
}

impl ProviderRegistry {
    pub fn new(
        config: &crate::config::ResolvedConfig,
        token_store: TokenStore,
        session_stats: Arc<crate::stats::SessionStats>,
    ) -> Self {
        Self {
            profiles: config.providers.clone(),
            session_context_window: config.session_context_window,
            thinking_budget_tokens: config.thinking_budget_tokens,
            requested_thinking: config.requested_thinking,
            device_ids: std::sync::Mutex::new(std::collections::HashMap::new()),
            token_store: Arc::new(token_store),
            session_stats,
            built: std::sync::Mutex::new(std::collections::HashMap::new()),
            #[cfg(debug_assertions)]
            scripted: std::sync::Mutex::new(None),
        }
    }

    /// Resolve a profile by name without building anything.
    ///
    /// Refuses by name rather than falling back to the default. A recorded profile that is no
    /// longer configured is the user's to fix, and quietly running the conversation somewhere else
    /// is the failure this whole arrangement exists to prevent.
    pub fn settings(
        &self,
        profile: &str,
        overrides: &crate::config::ProfileOverrides,
    ) -> Result<crate::config::ProfileSettings> {
        let configured = look_up_profile(&self.profiles, profile)?;
        Ok(crate::config::resolve_profile(
            configured,
            // `--thinking` folded in here rather than carried on the session's own overrides: it
            // describes this run, and applying it where the profile's value is read is what keeps
            // `ProviderKey.thinking` the mode the built provider actually uses.
            &crate::config::ProfileOverrides {
                model: overrides.model.clone(),
                base_url: overrides.base_url.clone(),
                thinking: self.requested_thinking.or(overrides.thinking),
            },
            self.session_context_window,
            self.device_id_for(profile, configured),
        ))
    }

    /// This profile's `claude-subscription` device id, resolved on the first ask and remembered.
    fn device_id_for(&self, name: &str, profile: &crate::config::ProviderProfile) -> String {
        let mut cache = match self.device_ids.lock() {
            Ok(cache) => cache,
            // A poisoned memo costs the memo, not correctness: resolving again returns the same
            // configured value, and a seeded one has been persisted by then, so the second ask
            // reads it back rather than minting another.
            Err(poisoned) => poisoned.into_inner(),
        };
        cache
            .entry(name.to_string())
            .or_insert_with(|| {
                crate::config::resolve_device_id(
                    &profile.backend,
                    name,
                    profile.device_id.as_deref(),
                )
            })
            .clone()
    }

    /// The provider for one profile, built on first ask and reused after, with the settings it was
    /// built from.
    ///
    /// Both, because the caller wants both and resolving is not free: it reads the profile, applies
    /// the run's overrides and looks up a device id. Returning only the provider meant
    /// [`crate::resolved_binding`] resolved a second time to learn the window and the vision flag
    /// that this call had just computed and thrown away.
    pub async fn build(
        &self,
        profile: &str,
        overrides: &crate::config::ProfileOverrides,
    ) -> Result<(Arc<dyn Provider>, crate::config::ProfileSettings)> {
        let settings = self.settings(profile, overrides)?;

        #[cfg(debug_assertions)]
        if let Ok(scripted) = self.scripted.lock()
            && let Some(scripted) = scripted.as_ref()
        {
            return Ok((Arc::clone(scripted), settings));
        }

        // Asked of the profile this session actually names, not of the process default. A pairing
        // that cannot produce a valid request is worth catching before the request, and only this
        // profile's own values can say whether it can.
        crate::config::validate_max_output_tokens(
            profile,
            Some(&settings.backend),
            settings.max_output_tokens,
            settings.thinking,
            self.thinking_budget_tokens,
        )?;
        let key = ProviderKey {
            profile: profile.to_string(),
            model: settings.model.clone(),
            base_url: settings.base_url.clone(),
            thinking: settings.thinking,
        };
        // Pulled and compared rather than pushed, because the writer that supersedes a credential
        // is usually a *different process* -- `meka provider login work` run against a
        // store a `meka serve` is already using -- and no invalidation hook can reach
        // across that. Without it, a rotation was invisible for the life of the process:
        // every later build served the provider holding the revoked key.
        let credential_version = self
            .token_store
            .provider_credential_version(profile)
            .await?;
        if let Ok(built) = self.built.lock()
            && let Some(existing) = built.get(&key)
            && existing.credential_version == credential_version
        {
            return Ok((Arc::clone(&existing.provider), settings));
        }

        let credential = self.credential_for(profile).await?;
        let model = settings.model.clone().ok_or_else(|| {
            MekaError::Config(format!(
                "provider profile '{}' names no model. Set `model` on [providers.{}]",
                profile, profile
            ))
        })?;
        let needs_token_store = matches!(credential, AuthCredential::OAuthToken { .. });
        let provider = ProviderBuilder::new(&settings.backend, credential, model)
            .base_url(settings.base_url.clone())
            .client_id(settings.client_id.clone())
            .credential_key(Some(profile.to_string()))
            .oauth_token_url(settings.oauth_token_url.clone())
            .token_store(needs_token_store.then(|| Arc::clone(&self.token_store)))
            .thinking(settings.thinking, self.thinking_budget_tokens)
            .device_id(settings.device_id.clone())
            .effort(settings.effort.clone())
            .redact_thinking(settings.redact_thinking)
            .max_output_tokens(settings.max_output_tokens)
            .session_stats(Some(Arc::clone(&self.session_stats)))
            .build()?;

        match self.built.lock() {
            Ok(mut built) => {
                let cached = built.entry(key).or_insert_with(|| CachedProvider {
                    credential_version: credential_version.clone(),
                    provider: Arc::clone(&provider),
                });
                // Whoever got here first wins, and a loser drops its own build rather than
                // replacing a provider another turn may already be using -- unless
                // what is there was built from a credential that has since been
                // superseded, which is the case this call exists to serve.
                //
                // Two builds spanning two rotations can land out of order and leave the older one
                // cached. That converges rather than sticking: the tag records which credential the
                // entry was built from, so the next ask compares it against the row and rebuilds.
                if cached.credential_version != credential_version {
                    *cached = CachedProvider {
                        credential_version,
                        provider: Arc::clone(&provider),
                    };
                }
                Ok((Arc::clone(&cached.provider), settings))
            }
            // A poisoned cache costs reuse, not correctness: the provider just built is complete
            // and usable, and the next ask builds another.
            Err(_) => Ok((provider, settings)),
        }
    }

    async fn credential_for(&self, profile: &str) -> Result<AuthCredential> {
        // Debug-only: the scripted provider replaces whatever this returns, so a harness need not
        // seed a credential it will never use. Reached only when `ProviderRegistry::build` was
        // called before a host installed the script.
        #[cfg(debug_assertions)]
        if std::env::var("MEKA_MOCK_PROVIDER").as_deref() == Ok("1") {
            return Ok(AuthCredential::ApiKey("mock-provider".to_string()));
        }
        match self.token_store.load_provider_credential(profile).await? {
            Some(credential) => Ok(credential),
            None => Err(MekaError::Config(format!(
                "provider profile '{}' has no stored credential. Run `meka provider login {}` to \
                 authenticate.",
                profile, profile
            ))),
        }
    }

    /// Install a scripted provider in place of every profile's. Debug builds only.
    #[cfg(debug_assertions)]
    pub fn install_scripted(&self, provider: Arc<dyn Provider>) {
        if let Ok(mut scripted) = self.scripted.lock() {
            *scripted = Some(provider);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What a host reports for a session it has not loaded, pinned to three distinct answers.
    ///
    /// The whole function could be replaced with `None`, `Some(0)` or `Some(1)` with the suite
    /// green. Its one non-test caller is `GET /v1/sessions/{id}/context`, which answers with it for
    /// an evicted session, so a wrong number here is a percentage a client divides by, and `None`
    /// specifically is the "meka cannot say" signal that stops a client inventing one.
    #[tokio::test]
    async fn a_bindings_window_is_its_profiles_or_the_documented_default_or_nothing() {
        let mut profiles = profiles(&["stated", "silent"]);
        profiles
            .get_mut("stated")
            .expect("seeded above")
            .context_window = Some(32_000);
        let registry = test_registry(profiles, None).await;

        assert_eq!(
            crate::binding_context_window(
                &registry,
                &crate::session::SessionProvider::from("stated".to_string())
            ),
            Some(32_000),
            "a profile that states a window reports it"
        );
        assert_eq!(
            crate::binding_context_window(
                &registry,
                &crate::session::SessionProvider::from("silent".to_string())
            ),
            Some(DEFAULT_CONTEXT_WINDOW),
            "one that states none reports the documented default, not its neighbour's value"
        );
        assert_eq!(
            crate::binding_context_window(
                &registry,
                &crate::session::SessionProvider::from("departed".to_string())
            ),
            None,
            "a profile that has left config.toml reports nothing, so no client divides by a guess"
        );
    }

    /// A session's recorded model and endpoint must survive the trip into a provider build.
    ///
    /// `overrides()` could return `Default::default()` -- dropping both -- with nothing failing,
    /// which would silently run a session recorded against one model on its profile's default
    /// instead. It feeds `ProviderKey`, so it also decides whether two sessions share a provider.
    #[test]
    fn a_recorded_binding_carries_its_overrides_into_the_registry() {
        let binding = crate::session::SessionProvider {
            profile: "work".to_string(),
            model_override: Some("claude-opus-5".to_string()),
            base_url_override: Some("https://example.invalid/v1".to_string()),
        };
        let overrides = binding.overrides();
        assert_eq!(overrides.model.as_deref(), Some("claude-opus-5"));
        assert_eq!(
            overrides.base_url.as_deref(),
            Some("https://example.invalid/v1")
        );
        assert_eq!(
            overrides.thinking, None,
            "thinking is a per-profile encoding the user states, never something a session pins"
        );

        let bare = crate::session::SessionProvider::from("work".to_string());
        assert_eq!(bare.overrides().model, None);
        assert_eq!(bare.overrides().base_url, None);
    }

    fn profiles(
        names: &[&str],
    ) -> std::collections::BTreeMap<String, crate::config::ProviderProfile> {
        names
            .iter()
            .map(|name| {
                (name.to_string(), crate::config::ProviderProfile {
                    backend: "anthropic-messages".to_string(),
                    ..Default::default()
                })
            })
            .collect()
    }

    /// A registry over `profiles`, with everything a resolution needs and nothing it does not.
    async fn test_registry(
        profiles: std::collections::BTreeMap<String, crate::config::ProviderProfile>,
        requested_thinking: Option<ThinkingMode>,
    ) -> ProviderRegistry {
        let manager = crate::session::SessionManager::open(
            Some(std::path::Path::new(":memory:")),
            &Default::default(),
        )
        .await
        .expect("memory store");
        ProviderRegistry {
            profiles,
            session_context_window: None,
            thinking_budget_tokens: 4_096,
            requested_thinking,
            device_ids: std::sync::Mutex::new(std::collections::HashMap::new()),
            token_store: Arc::new(manager.token_store()),
            session_stats: Arc::new(crate::stats::SessionStats::default()),
            built: std::sync::Mutex::new(std::collections::HashMap::new()),
            #[cfg(debug_assertions)]
            scripted: std::sync::Mutex::new(None),
        }
    }

    /// `--thinking` is a property of the run, and a session's own overrides carry a model and an
    /// endpoint, neither of which is a thinking mode. Once a session resolved its own profile the
    /// flag reached nothing: it parsed, it was documented as an override, and it was discarded.
    #[tokio::test]
    async fn the_runs_thinking_flag_outranks_the_profile_a_session_resolves_to() {
        let mut profiles = profiles(&["work"]);
        if let Some(profile) = profiles.get_mut("work") {
            profile.thinking = Some(ThinkingMode::Adaptive);
        }
        let overrides = crate::config::ProfileOverrides::default();

        let without = test_registry(profiles.clone(), None).await;
        assert_eq!(
            without
                .settings("work", &overrides)
                .expect("resolve")
                .thinking,
            ThinkingMode::Adaptive,
            "with no flag the profile's own mode stands"
        );

        let with = test_registry(profiles, Some(ThinkingMode::Off)).await;
        assert_eq!(
            with.settings("work", &overrides).expect("resolve").thinking,
            ThinkingMode::Off,
            "the flag must reach the settings the provider is built from"
        );
    }

    /// Resolving a `claude-subscription` device id mints one and writes `config.toml`, and
    /// `profiles` here is a snapshot, so a resolver called per request never saw what it had just
    /// persisted. It minted another every time, for an identifier whose whole purpose is to stay
    /// the same, and took the config lock to write it on every context poll and status query.
    ///
    /// Isolated, because the thing under test *writes a config file and takes a cross-process file
    /// lock*. Unisolated it read the developer's real `~/.claude.json`, then `create_dir_all`ed and
    /// `flock`ed `~/.config/meka/.config.toml.lock` with no timeout, so a `meka` running in another
    /// terminal would hang `cargo test` outright. It escaped rewriting the real `config.toml` only
    /// because `persist` bails when no profile of that name is there to write into, which is luck
    /// rather than design: a developer with a profile named `sub` had their device id rewritten.
    #[tokio::test]
    async fn a_device_id_is_resolved_once_per_profile_however_often_it_is_asked_for() {
        let mut profiles = profiles(&["sub"]);
        if let Some(profile) = profiles.get_mut("sub") {
            profile.backend = "claude-subscription".to_string();
        }
        let registry = test_registry(profiles, None).await;
        let overrides = crate::config::ProfileOverrides::default();

        let config_dir = tempfile::tempdir().expect("tempdir");
        // Seeded with the profile, so `persist` takes its write path rather than the bail it used
        // to take by accident against the real file.
        std::fs::write(
            config_dir.path().join("config.toml"),
            "[providers.sub]\ntype = \"claude-subscription\"\n",
        )
        .expect("write config.toml");

        // SAFETY: `MEKA_CONFIG_DIR` is process-global; `CONFIG_DIR_ENV_LOCK` serialises every test
        // that touches it, and the guard is held across the whole set -> resolve -> clear cycle.
        let _env = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", config_dir.path()) };

        let first = registry
            .settings("sub", &overrides)
            .expect("resolve")
            .device_id;
        let second = registry
            .settings("sub", &overrides)
            .expect("resolve")
            .device_id;

        let persisted = std::fs::read_to_string(config_dir.path().join("config.toml"))
            .expect("read config.toml back");
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        assert!(!first.is_empty(), "a subscription profile gets one");
        assert_eq!(
            first, second,
            "asking twice must not mint a second identifier"
        );
        assert!(
            persisted.contains(&first),
            "the minted id must be written back to the profile, or the next process mints another"
        );
    }

    /// The memo used to be keyed on the profile and its overrides alone, so a built provider
    /// outlived the credential it was built from: `meka provider login work` run against the store
    /// a `meka serve` was already using rotated the key, and every later build in the running
    /// process still handed out the provider holding the revoked one. The writer is another
    /// process, so this has to be a pull rather than an invalidation hook.
    #[tokio::test]
    async fn rotating_a_credential_retires_the_provider_built_from_it() {
        let mut profiles = profiles(&["work"]);
        if let Some(profile) = profiles.get_mut("work") {
            profile.model = Some("claude-opus-5".to_string());
        }
        let registry = test_registry(profiles, None).await;
        let overrides = crate::config::ProfileOverrides::default();
        registry
            .token_store
            .save_provider_credential("work", &AuthCredential::ApiKey("first".to_string()))
            .await
            .expect("store a credential");

        let (first, _) = registry.build("work", &overrides).await.expect("build");
        let (again, _) = registry
            .build("work", &overrides)
            .await
            .expect("build again");
        assert!(
            Arc::ptr_eq(&first, &again),
            "an unmoved credential must still be served from the memo -- keeping the connection \
             pool is the whole reason this caches"
        );

        registry
            .token_store
            .save_provider_credential("work", &AuthCredential::ApiKey("second".to_string()))
            .await
            .expect("rotate the credential");

        let (after, _) = registry
            .build("work", &overrides)
            .await
            .expect("build after the rotation");
        assert!(
            !Arc::ptr_eq(&first, &after),
            "a rotated credential must retire the provider built from its predecessor"
        );
    }

    /// The refusal every turn-running site depends on. Falling back to the default here would put
    /// the conversation on another account without saying so, which is the bug being fixed.
    #[test]
    fn an_unconfigured_profile_is_refused_by_name_and_lists_what_exists() {
        let configured = profiles(&["work", "personal"]);
        let error = look_up_profile(&configured, "retired")
            .expect_err("a profile that is not configured cannot resolve")
            .to_string();

        assert!(
            error.contains("'retired'"),
            "names the missing one: {error}"
        );
        assert!(error.contains("work"), "lists what is available: {error}");
        assert!(error.contains("personal"), "lists all of them: {error}");
    }

    /// The empty case says something different, because "configured profiles: " followed by nothing
    /// reads as a bug rather than as an instruction. This is also the state a store carried forward
    /// with no resolvable provider lands in.
    #[test]
    fn no_profiles_at_all_points_at_provider_add() {
        let error = look_up_profile(&profiles(&[]), "anything")
            .expect_err("nothing can resolve")
            .to_string();

        assert!(error.contains("meka provider add"), "{error}");
    }

    #[test]
    fn a_configured_profile_resolves() {
        let configured = profiles(&["work"]);
        assert_eq!(
            look_up_profile(&configured, "work")
                .expect("configured")
                .backend,
            "anthropic-messages"
        );
    }

    #[tokio::test]
    async fn a_nested_turn_keeps_the_prompt_it_was_spawned_from() {
        // Claude Code gives a sub-agent's requests its spawner's prompt id
        // (`agentContext.parentPromptId`), and a sub-agent's turn runs nested inside the turn that
        // spawned it, so keeping an already-scoped id is the whole mechanism. Minting a fresh one
        // per scope would look identical in every single-turn test.
        let outer_slot = PreviousRequestSlot::default();
        let (outer, inner) = scope_turn(outer_slot, async {
            let outer = current_prompt_id();
            let inner = scope_turn(PreviousRequestSlot::default(), async {
                current_prompt_id()
            })
            .await;
            (outer, inner)
        })
        .await;
        assert!(outer.is_some());
        assert_eq!(outer, inner);
    }

    #[tokio::test]
    async fn attribution_survives_the_spawn_the_streaming_path_makes() {
        // The streaming path hands `provider.stream(...)` to `tokio::spawn`, and a task-local does
        // not cross that. Every assertion about the billing header that runs inline passes whether
        // or not the carry-over exists, so this is the one that fails when it is dropped. All three
        // fields are checked together because they travel together: `subagent` has no streaming
        // caller today, and this is what keeps it correct when it gets one.
        let slot = PreviousRequestSlot::default();
        let spawned = scope_subagent(scope_turn(Arc::clone(&slot), async {
            let expected = current_prompt_id();
            record_request_id("req_before_the_spawn");
            let captured = capture_attribution();
            tokio::spawn(scope_attribution(captured, async move {
                (is_subagent(), current_prompt_id(), previous_request_id())
            }))
            .await
            .expect("spawned task")
                == (true, expected, Some("req_before_the_spawn".to_string()))
        }))
        .await;
        assert!(spawned, "the spawned request lost its attribution");
    }

    #[tokio::test]
    async fn a_nested_turn_records_its_own_responses() {
        // The other half: the request-id slot is *not* inherited, because a sub-agent's responses
        // belong to its own conversation. A sub-agent recording into its spawner's slot would make
        // the parent's next request name a response it never received.
        let outer_slot = PreviousRequestSlot::default();
        let inner_seen = scope_turn(Arc::clone(&outer_slot), async {
            record_request_id("req_outer");
            scope_turn(PreviousRequestSlot::default(), async {
                let before = previous_request_id();
                record_request_id("req_inner");
                before
            })
            .await
        })
        .await;
        assert_eq!(inner_seen, None, "the sub-agent starts with a clean slot");
        assert_eq!(
            outer_slot.lock().expect("slot").as_deref(),
            Some("req_outer"),
            "the sub-agent's response must not land in its spawner's slot"
        );
    }

    #[tokio::test]
    async fn only_a_well_formed_request_id_is_carried_forward() {
        // A value that fails Claude Code's `^req_[A-Za-z0-9_-]{1,36}$` would be a claim about a
        // response that doesn't exist, so it is dropped rather than forwarded.
        for rejected in [
            "011CeJwF1NDYzXkUu6cyFJp2",
            "req_",
            "req_has spaces",
            "req_has;semicolon",
            "req_0123456789012345678901234567890123456789",
        ] {
            let slot = PreviousRequestSlot::default();
            scope_turn(Arc::clone(&slot), async { record_request_id(rejected) }).await;
            assert_eq!(
                slot.lock().expect("slot").as_deref(),
                None,
                "'{rejected}' must not reach the billing header"
            );
        }
        let slot = PreviousRequestSlot::default();
        scope_turn(Arc::clone(&slot), async {
            record_request_id("req_011CeJwF1NDYzXkUu6cyFJp2")
        })
        .await;
        assert_eq!(
            slot.lock().expect("slot").as_deref(),
            Some("req_011CeJwF1NDYzXkUu6cyFJp2")
        );
    }

    /// A refresh that loses its swap adopts what the row holds only when that can authenticate the
    /// request in hand.
    ///
    /// "Superseded" is a statement about write order and nothing else, and adopting on write order
    /// alone throws away a token the issuer has just minted in favour of one that may be dead or
    /// may not even be the same kind of credential. The failure is quiet in both directions: an
    /// expired adoption 401s the very request it was fetched for, and an API key adopted into a
    /// subscription profile is sent as a bearer token that endpoint has never accepted.
    #[tokio::test]
    async fn a_superseding_credential_is_adopted_only_when_it_can_authenticate() {
        let manager = crate::session::SessionManager::open(
            Some(std::path::Path::new(":memory:")),
            &Default::default(),
        )
        .await
        .expect("memory store");
        let store = manager.token_store();
        let oauth = |access: &str, expires_at: Option<i64>| AuthCredential::OAuthToken {
            access_token: access.to_string(),
            refresh_token: Some("refresh".to_string()),
            expires_at,
            account_id: None,
        };
        let hour = 3_600_000;
        let derived_from = oauth("original", Some(now_epoch_millis() - hour));
        let minted = || oauth("just-minted", Some(now_epoch_millis() + hour));
        // Every case below is a lost swap, which is what the row holding something other than
        // `derived_from` produces.
        let plant = async |credential: AuthCredential| {
            store
                .save_provider_credential("work", &credential)
                .await
                .expect("plant what the winner left");
        };
        let access_token = |credential: &AuthCredential| match credential {
            AuthCredential::OAuthToken { access_token, .. } => access_token.clone(),
            AuthCredential::ApiKey(key) => key.clone(),
        };

        plant(oauth("theirs", Some(now_epoch_millis() + hour))).await;
        assert_eq!(
            access_token(
                &store_refreshed_credential(&store, "work", &derived_from, minted()).await
            ),
            "theirs",
            "the premise: a live credential from the winner is the one to use"
        );

        plant(oauth("stale", Some(now_epoch_millis() - hour))).await;
        assert_eq!(
            access_token(
                &store_refreshed_credential(&store, "work", &derived_from, minted()).await
            ),
            "just-minted",
            "a token that won the write and then expired cannot authenticate this request"
        );

        plant(AuthCredential::ApiKey("repurposed".to_string())).await;
        assert_eq!(
            access_token(
                &store_refreshed_credential(&store, "work", &derived_from, minted()).await
            ),
            "just-minted",
            "and a profile somebody repurposed to an API key is not a newer version of this token"
        );
    }

    /// `AuthCredential` derived `Debug` over its plaintext, and a provider struct holding one is
    /// exactly the kind of thing that lands in a `{:?}` during a bad afternoon.
    #[test]
    fn a_credential_never_reaches_a_debug_rendering() {
        let rendered = format!(
            "{:?}",
            AuthCredential::ApiKey("sk-APIKEYSECRET".to_string())
        );
        assert!(!rendered.contains("APIKEYSECRET"), "{rendered}");
        assert!(rendered.contains("REDACTED"), "{rendered}");

        let rendered = format!("{:?}", AuthCredential::OAuthToken {
            access_token: "ACCESSSECRET".to_string(),
            refresh_token: Some("REFRESHSECRET".to_string()),
            expires_at: Some(42),
            account_id: Some("acct-visible".to_string()),
        });
        assert!(!rendered.contains("ACCESSSECRET"), "{rendered}");
        assert!(!rendered.contains("REFRESHSECRET"), "{rendered}");
        // The non-secret identity stays: it is what a wrong-account diagnosis needs.
        assert!(rendered.contains("acct-visible"), "{rendered}");
    }

    /// An issuer that omits `expires_in` is not promising the token is eternal. Reading `None` as
    /// "valid forever" meant the refresh path was never entered, so the credential 401'd on every
    /// request for the life of the process with nothing naming why.
    #[test]
    fn an_unlabelled_expiry_is_refreshed_rather_than_trusted_forever() {
        let now = 1_700_000_000_000;
        assert!(oauth_needs_refresh(None, true, now));

        // With no refresh token there is nothing to act on, so send it and let the 401 speak.
        assert!(!oauth_needs_refresh(None, false, now));

        // A stated expiry is still honoured, skew included, and still refuses to refresh early.
        assert!(oauth_needs_refresh(Some(now + 60_000), true, now));
        assert!(!oauth_needs_refresh(Some(now + 600_000), true, now));

        // A `now` near the top of the range must not wrap into "not yet due".
        assert!(oauth_needs_refresh(Some(i64::MAX), true, i64::MAX));
    }

    /// And "due" has to stay a one-shot.
    ///
    /// Handing `None` straight back out of a refresh whose issuer stated no expiry made the token
    /// due again the instant it arrived, so every request re-entered the slow path -- credential
    /// write lock, database re-read, full OAuth round trip -- serialised, rotating the refresh
    /// token on each pass. The refresh paths stamp an assumed expiry instead.
    #[test]
    fn a_refresh_that_states_no_expiry_is_not_due_again_immediately() {
        let now = 1_700_000_000_000;
        assert!(
            !oauth_needs_refresh(Some(oauth_assumed_expiry(now)), true, now),
            "a token that just arrived must not already be due",
        );
        assert!(
            oauth_needs_refresh(
                Some(oauth_assumed_expiry(now)),
                true,
                now + OAUTH_ASSUMED_LIFETIME_MILLIS
            ),
            "but the assumption expires: it bounds staleness, it does not trust the token forever",
        );
    }

    #[test]
    fn test_a_base_url_keeps_its_path_and_loses_only_trailing_slashes() {
        assert_eq!(
            normalize_base_url("https://openrouter.ai/api/v1/"),
            "https://openrouter.ai/api/v1"
        );
        assert_eq!(
            normalize_base_url("https://openrouter.ai/api/v1///"),
            "https://openrouter.ai/api/v1"
        );
        assert_eq!(
            normalize_base_url("  https://openrouter.ai/api/v1  "),
            "https://openrouter.ai/api/v1"
        );
        // Already clean: byte-identical, so the common path rewrites nothing.
        assert_eq!(
            normalize_base_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1"
        );
    }

    #[test]
    fn test_the_generic_normalizer_leaves_the_version_segment_alone() {
        // The OpenAI family carries `/v1` in the base by convention, so stripping it here would
        // break every profile pasted from a provider's own documentation.
        assert_eq!(
            normalize_base_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1"
        );
        assert_eq!(
            normalize_base_url("https://api.synthetic.new/openai/v1/"),
            "https://api.synthetic.new/openai/v1"
        );
    }

    /// The four ways a thinking mode is spelled must agree.
    ///
    /// A mode is written to `config.toml` by `meka provider add`, read back by serde, accepted on
    /// the CLI by clap, and printed by `/status`. Each of those has its own derivation of the
    /// string, and today they coincide only because every variant happens to be one lowercase word:
    /// clap kebab-cases the variant name, serde lowercases it, and the other two are hand-written.
    /// Adding a two-word variant would split them silently - clap taking `some-mode` and serde
    /// `somemode` - so the agreement is pinned here rather than left to that coincidence.
    #[test]
    fn every_spelling_of_a_thinking_mode_agrees() {
        use clap::ValueEnum;

        #[derive(Deserialize)]
        struct Profile {
            thinking: ThinkingMode,
        }

        for mode in ThinkingMode::value_variants() {
            let written = mode.as_str();
            let parsed: Profile = toml::from_str(&format!("thinking = \"{written}\""))
                .unwrap_or_else(|error| panic!("serde must read back `{written}`: {error}"));
            assert_eq!(parsed.thinking, *mode);
            assert_eq!(
                mode.to_possible_value()
                    .map(|value| value.get_name().to_string()),
                Some(written.to_string()),
                "clap must accept on the CLI what the profile is written with: {written}"
            );
        }
    }

    /// The prompt's default endpoint has to be the constructor's default endpoint.
    ///
    /// `meka provider add` shows this URL as the value an empty answer accepts, and each provider
    /// reads the constant directly, so the backend-name mapping is the one joint where they can
    /// disagree. A wrong arm here compiles, passes every other test, and quietly tells every user
    /// of that backend their requests go somewhere they do not.
    #[test]
    fn the_advertised_default_endpoint_is_the_one_the_backend_uses() {
        assert_eq!(
            default_base_url("anthropic-messages"),
            Some(DEFAULT_ANTHROPIC_BASE_URL)
        );
        assert_eq!(
            default_base_url("claude-subscription"),
            Some(DEFAULT_ANTHROPIC_BASE_URL)
        );
        assert_eq!(
            default_base_url("openai-chat-completions"),
            Some(DEFAULT_OPENAI_BASE_URL)
        );
        // The two OpenAI protocols share a default host: `openai-responses` is a different wire
        // format against the same API, not a different service. Aiming it at the subscription's
        // `chatgpt.com` would compile and pass a mere is-some check, so it is named explicitly.
        assert_eq!(
            default_base_url("openai-responses"),
            Some(DEFAULT_OPENAI_BASE_URL)
        );
        assert_eq!(
            default_base_url("chatgpt-subscription"),
            Some(DEFAULT_CHATGPT_BASE_URL)
        );
        // Every supported backend answers, so the prompt never has to fall back to naming no URL.
        // A weaker guard than the assertions above and deliberately kept alongside them: it is what
        // catches a *newly added* backend that nobody remembered to give a default.
        for backend in SUPPORTED_PROVIDERS {
            assert!(default_base_url(backend).is_some(), "{backend}");
        }
        assert_eq!(default_base_url("not-a-backend"), None);
    }

    #[test]
    fn an_unconfigured_effort_is_omitted_so_the_provider_applies_its_own() {
        // The whole policy: meka names a tier only when the profile did. It cannot know which tiers
        // a given endpoint implements - `anthropic-messages` and `openai-chat-completions` reach
        // any compatible server - and an unimplemented tier is a rejected request, not a
        // graceful ignore.
        assert_eq!(resolve_effort_level(None), None);
        // A blank override reads as unset rather than as an empty wire value the API would reject.
        assert_eq!(resolve_effort_level(Some("")), None);
        assert_eq!(resolve_effort_level(Some("   ")), None);
        // A configured value is absolute: verbatim, trimmed and lowercased, never clamped, and
        // never rejected for the model it is aimed at.
        assert_eq!(
            resolve_effort_level(Some("medium")).as_deref(),
            Some("medium")
        );
        assert_eq!(
            resolve_effort_level(Some("  XHigh ")).as_deref(),
            Some("xhigh")
        );
        assert_eq!(resolve_effort_level(Some("max")).as_deref(), Some("max"));
    }

    #[test]
    fn test_user_with_images_appends_image_blocks_after_text() {
        let images = vec![
            ImageSource {
                source_type: "base64".to_string(),
                media_type: "image/png".to_string(),
                data: "AAAA".to_string(),
            },
            ImageSource {
                source_type: "base64".to_string(),
                media_type: "image/jpeg".to_string(),
                data: "BBBB".to_string(),
            },
        ];
        let message = Message::user_with_images("look at these", images);
        assert_eq!(message.role, Role::User);
        assert_eq!(message.content.len(), 3);
        assert!(
            matches!(&message.content[0], ContentBlock::Text { text } if text == "look at these")
        );
        assert!(
            matches!(&message.content[1], ContentBlock::Image { source } if source.media_type == "image/png")
        );
        assert!(
            matches!(&message.content[2], ContentBlock::Image { source } if source.media_type == "image/jpeg")
        );
        // No images yields the same shape as `Message::user`.
        assert_eq!(Message::user_with_images("hi", vec![]).content.len(), 1);
    }

    /// A `tool_result` whose `content` is a bare string is refused, not silently wrapped.
    ///
    /// `content` is a list, and it used to accept a string too, which was the shape meka persisted
    /// before the field could hold images. Nothing pinned that, so this pins its removal: the
    /// release script rewrites those rows, and if this ever passes again the script has quietly
    /// become optional while still being the only thing that converts them.
    #[test]
    fn a_tool_result_content_written_as_a_string_no_longer_parses() {
        let stored =
            r#"{"type":"tool_result","tool_use_id":"toolu_1","content":"bare","is_error":false}"#;
        assert!(
            serde_json::from_str::<ContentBlock>(stored).is_err(),
            "the string form must not resurrect as a silent wrap"
        );

        let converted = r#"{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"bare"}],"is_error":false}"#;
        let block: ContentBlock = serde_json::from_str(converted).expect("the converted shape");
        assert!(matches!(
            block,
            ContentBlock::ToolResult { ref content, .. } if content.len() == 1
        ));
    }

    /// Both shapes survive a round trip, and `Sealed` omits an id it does not have.
    ///
    /// Named for what it can actually check. There is no backward compatibility to test: 0.42 reads
    /// only this shape, and a store written by an earlier release is brought forward by the
    /// migration script, not by serde. The `skip_serializing_if` is the part worth pinning -- drop
    /// it and every id-less sealed block starts writing `"id":null` into the log.
    #[test]
    fn both_opaque_shapes_round_trip_and_an_absent_id_is_not_written() {
        for (stored, expected) in [
            (
                r#"{"type":"thinking","thinking":"hmm","opaque":{"kind":"signed","signature":"SIG"}}"#,
                OpaqueReasoning::Signed {
                    signature: "SIG".to_string(),
                },
            ),
            (
                r#"{"type":"thinking","thinking":"s","opaque":{"kind":"sealed","encrypted_content":"E"}}"#,
                OpaqueReasoning::Sealed {
                    encrypted_content: "E".to_string(),
                    id: None,
                },
            ),
        ] {
            let block: ContentBlock = serde_json::from_str(stored).expect("must load");
            assert!(
                matches!(&block, ContentBlock::Thinking { opaque: Some(opaque), .. }
                    if *opaque == expected),
                "got {block:?}"
            );
            assert_eq!(
                serde_json::to_string(&block).expect("serialize"),
                stored,
                "an absent id must not be written back as null"
            );
        }
    }

    #[test]
    fn test_without_tool_use_keeps_text_and_thinking_drops_tool_use() {
        let message = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    thinking: "hmm".to_string(),
                    opaque: None,
                },
                ContentBlock::Text {
                    text: "let me check".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({}),
                },
            ],
        };
        let stripped = message.without_tool_use();
        assert_eq!(stripped.role, Role::Assistant);
        assert_eq!(stripped.content.len(), 2);
        assert!(matches!(
            &stripped.content[0],
            ContentBlock::Thinking { .. }
        ));
        assert!(
            matches!(&stripped.content[1], ContentBlock::Text { text } if text == "let me check")
        );
        // A tool-use-only message strips to empty content.
        let only_tool = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call_2".to_string(),
                name: "x".to_string(),
                input: serde_json::json!({}),
            }],
        };
        assert!(only_tool.without_tool_use().content.is_empty());
    }

    #[test]
    fn test_token_usage_merge_stream_keeps_input_from_start_output_from_delta() {
        // Anthropic streaming: `message_start` carries the input/cache tiers (output a
        // placeholder), `message_delta` carries the final output with the input/cache
        // fields absent (parsed as 0).
        let mut usage = TokenUsage::default();
        usage.merge_stream(&TokenUsage {
            input_tokens: 1000,
            output_tokens: 1,
            cache_creation_input_tokens: 200,
            cache_read_input_tokens: 5000,
        });
        usage.merge_stream(&TokenUsage {
            input_tokens: 0,
            output_tokens: 250,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        });
        assert_eq!(
            usage.input_tokens, 1000,
            "input retained from message_start"
        );
        assert_eq!(usage.cache_creation_input_tokens, 200);
        assert_eq!(usage.cache_read_input_tokens, 5000);
        assert_eq!(usage.output_tokens, 250, "output taken from message_delta");
    }

    #[test]
    fn test_token_usage_merge_stream_single_event_is_verbatim() {
        // OpenAI/Codex emit a single usage event; merging from default keeps it as-is.
        let mut usage = TokenUsage::default();
        usage.merge_stream(&TokenUsage {
            input_tokens: 800,
            output_tokens: 120,
            ..Default::default()
        });
        assert_eq!(usage.input_tokens, 800);
        assert_eq!(usage.output_tokens, 120);
        assert_eq!(usage.cache_read_input_tokens, 0);
    }

    #[test]
    fn test_auth_credential_json_round_trip() {
        // `AuthCredential` is serialized to JSON for storage in `provider_credentials`; both
        // variants must survive a round-trip intact.
        let api_key = AuthCredential::ApiKey("sk-test".to_string());
        let json = serde_json::to_string(&api_key).expect("serialize ApiKey");
        match serde_json::from_str::<AuthCredential>(&json).expect("deserialize ApiKey") {
            AuthCredential::ApiKey(key) => assert_eq!(key, "sk-test"),
            other => panic!("expected ApiKey, got {:?}", other),
        }

        let oauth = AuthCredential::OAuthToken {
            access_token: "access".to_string(),
            refresh_token: Some("refresh".to_string()),
            expires_at: Some(1_700_000_000_000),
            account_id: Some("acct".to_string()),
        };
        let json = serde_json::to_string(&oauth).expect("serialize OAuthToken");
        match serde_json::from_str::<AuthCredential>(&json).expect("deserialize OAuthToken") {
            AuthCredential::OAuthToken {
                access_token,
                refresh_token,
                expires_at,
                account_id,
            } => {
                assert_eq!(access_token, "access");
                assert_eq!(refresh_token.as_deref(), Some("refresh"));
                assert_eq!(expires_at, Some(1_700_000_000_000));
                assert_eq!(account_id.as_deref(), Some("acct"));
            }
            other => panic!("expected OAuthToken, got {:?}", other),
        }
    }

    /// Regression test for the "silent `{}` fallback" bug: a tool call with unparseable JSON
    /// arguments must be rejected via [`StreamEvent::ToolCallRejected`] rather than replayed with
    /// an empty input object (which would run the tool on whatever defaults it happens to
    /// tolerate).
    #[tokio::test]
    async fn test_finalize_tool_call_accumulators_rejects_invalid_json() {
        let mut accumulators = std::collections::HashMap::new();
        accumulators.insert(0, ToolCallAccumulator {
            id: "call-1".to_string(),
            name: "read_file".to_string(),
            arguments: "{not json".to_string(),
        });

        let (sender, mut receiver) = mpsc::channel::<StreamEvent>(16);
        let has_tools = finalize_tool_call_accumulators(&mut accumulators, &sender).await;
        assert!(has_tools, "accumulator was non-empty");

        let first = receiver.try_recv().expect("ToolUseStart emitted first");
        assert!(
            matches!(first, StreamEvent::ToolUseStart { .. }),
            "expected ToolUseStart, got {:?}",
            first
        );

        let second = receiver.try_recv().expect("follow-up event");
        match second {
            StreamEvent::ToolCallRejected { id, name, reason } => {
                assert_eq!(id, "call-1");
                assert_eq!(name, "read_file");
                assert!(reason.starts_with("invalid JSON arguments"));
            }
            other => panic!("expected ToolCallRejected, got {:?}", other),
        }

        assert!(
            receiver.try_recv().is_err(),
            "no further events after rejection"
        );
    }

    #[tokio::test]
    async fn test_finalize_tool_call_accumulators_passes_valid_json() {
        let mut accumulators = std::collections::HashMap::new();
        accumulators.insert(0, ToolCallAccumulator {
            id: "call-2".to_string(),
            name: "read_file".to_string(),
            arguments: r#"{"path": "/tmp/x"}"#.to_string(),
        });

        let (sender, mut receiver) = mpsc::channel::<StreamEvent>(16);
        finalize_tool_call_accumulators(&mut accumulators, &sender).await;

        let first = receiver.try_recv().expect("ToolUseStart");
        assert!(matches!(first, StreamEvent::ToolUseStart { .. }));

        match receiver.try_recv().expect("ToolUseEnd") {
            StreamEvent::ToolUseEnd { input } => {
                assert_eq!(input["path"], "/tmp/x");
            }
            other => panic!("expected ToolUseEnd, got {:?}", other),
        }
    }

    #[test]
    fn test_message_user() {
        let message = Message::user("hello");
        assert_eq!(message.role, Role::User);
        assert_eq!(message.text_content(), "hello");
    }

    #[test]
    fn test_message_assistant_text() {
        let message = Message::assistant_text("response");
        assert_eq!(message.role, Role::Assistant);
        assert_eq!(message.text_content(), "response");
    }

    #[test]
    fn test_message_tool_uses() {
        let message = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "I'll read that file.".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/test.txt"}),
                },
            ],
        };
        assert_eq!(message.tool_uses().len(), 1);
    }

    #[test]
    fn test_content_block_serialization() {
        let block = ContentBlock::Text {
            text: "hello".to_string(),
        };
        let json = serde_json::to_string(&block).expect("should serialize");
        let deserialized: ContentBlock = serde_json::from_str(&json).expect("should deserialize");

        if let ContentBlock::Text { text } = deserialized {
            assert_eq!(text, "hello");
        } else {
            panic!("expected Text block");
        }
    }

    #[test]
    fn test_message_serialization_roundtrip() {
        let message = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "Let me read that.".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/test"}),
                },
            ],
        };

        let json = serde_json::to_string(&message).expect("should serialize");
        let deserialized: Message = serde_json::from_str(&json).expect("should deserialize");

        assert_eq!(deserialized.role, Role::Assistant);
        assert_eq!(deserialized.content.len(), 2);
        assert_eq!(deserialized.text_content(), "Let me read that.");
    }

    /// Every supported backend must actually build.
    ///
    /// The last hand-written backend list with no loop behind it. `SUPPORTED_PROVIDERS`,
    /// `validate_backend` and `ResolvedConfig::validate` all accept a name before `build` is
    /// reached, so a backend added to the list and forgotten in the dispatch falls to the catch-all
    /// and dies at runtime with "unknown provider" — the same failure `acquire_credential`'s
    /// `unreachable!()` produced, which shipped as far as a live `provider add` before being
    /// caught.
    ///
    /// Iterating the list rather than naming five backends is the point: a sixth is covered the day
    /// it is added.
    #[test]
    fn every_supported_backend_builds() {
        for backend in SUPPORTED_PROVIDERS {
            // Hand each backend the credential shape it accepts; the mismatch cases are asserted
            // separately by the `_rejects_` tests.
            let credential = if backend.ends_with("-subscription") {
                AuthCredential::OAuthToken {
                    access_token: "token".to_string(),
                    refresh_token: None,
                    expires_at: None,
                    account_id: None,
                }
            } else {
                AuthCredential::ApiKey("key".to_string())
            };
            let built = ProviderBuilder::new(*backend, credential, "some-model")
                .device_id("a".repeat(64))
                .build();
            assert!(
                built.is_ok(),
                "{backend} is supported but does not build: {:?}",
                built.err()
            );
        }
    }

    #[test]
    fn a_chat_completions_profile_builds_from_an_api_key() {
        let result = ProviderBuilder::new(
            "openai-chat-completions",
            AuthCredential::ApiKey("key".to_string()),
            "gpt-4o",
        )
        .device_id("a".repeat(64))
        .build();
        assert!(result.is_ok());
    }

    #[test]
    fn an_anthropic_messages_profile_builds_from_an_api_key() {
        let result = ProviderBuilder::new(
            "anthropic-messages",
            AuthCredential::ApiKey("key".to_string()),
            "claude-sonnet-4-20250514",
        )
        .thinking(ThinkingMode::Off, 10000)
        .build();
        assert!(result.is_ok());
    }

    #[test]
    fn a_claude_subscription_profile_builds_from_an_oauth_token() {
        let result = ProviderBuilder::new(
            "claude-subscription",
            AuthCredential::OAuthToken {
                access_token: "sk-ant-oat01-test".to_string(),
                refresh_token: None,
                expires_at: None,
                account_id: None,
            },
            "claude-sonnet-4-20250514",
        )
        .device_id("a".repeat(64))
        .build();
        assert!(result.is_ok());
    }

    #[test]
    fn anthropic_messages_refuses_an_oauth_token() {
        let result = ProviderBuilder::new(
            "anthropic-messages",
            AuthCredential::OAuthToken {
                access_token: "sk-ant-oat01-test".to_string(),
                refresh_token: None,
                expires_at: None,
                account_id: None,
            },
            "claude-sonnet-4-20250514",
        )
        .build();
        assert!(result.is_err());
    }

    #[test]
    fn claude_subscription_refuses_an_api_key() {
        let result = ProviderBuilder::new(
            "claude-subscription",
            AuthCredential::ApiKey("sk-ant-api03-test".to_string()),
            "claude-sonnet-4-20250514",
        )
        .build();
        assert!(result.is_err());
    }

    #[test]
    fn a_chatgpt_subscription_profile_builds_from_an_oauth_token() {
        let result = ProviderBuilder::new(
            "chatgpt-subscription",
            AuthCredential::OAuthToken {
                access_token: "codex-access".to_string(),
                refresh_token: Some("codex-refresh".to_string()),
                expires_at: Some(now_ms_in_far_future()),
                account_id: Some("workspace-1".to_string()),
            },
            "gpt-5",
        )
        .effort(Some("high".to_string()))
        .build();
        assert!(result.is_ok());
    }

    #[test]
    fn chatgpt_subscription_refuses_an_api_key() {
        let result = ProviderBuilder::new(
            "chatgpt-subscription",
            AuthCredential::ApiKey("sk-...".to_string()),
            "gpt-5",
        )
        .build();
        assert!(result.is_err());
    }

    fn now_ms_in_far_future() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_millis() as i64 + 86_400_000)
            .unwrap_or(0)
    }

    #[test]
    fn test_create_provider_unknown() {
        let result = ProviderBuilder::new(
            "unknown",
            AuthCredential::ApiKey("key".to_string()),
            "model",
        )
        .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_auth_credential_api_key_header() {
        let credential = AuthCredential::ApiKey("my-key".to_string());
        let (name, value) = credential.auth_header();
        assert_eq!(name, "x-api-key");
        assert_eq!(value, "my-key");
    }

    #[test]
    fn test_auth_credential_oauth_header() {
        let credential = AuthCredential::OAuthToken {
            access_token: "my-token".to_string(),
            refresh_token: None,
            expires_at: None,
            account_id: None,
        };
        let (name, value) = credential.auth_header();
        assert_eq!(name, "Authorization");
        assert_eq!(value, "Bearer my-token");
    }
}
