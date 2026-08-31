//! `chatgpt-subscription`: the Responses API billed to a ChatGPT subscription.
//!
//! Talks the Responses API to `chatgpt.com/backend-api/codex/responses`, authenticated by the
//! bearer token + `ChatGPT-Account-ID` header issued by the Codex OAuth flow. Mirrors how OpenAI's
//! own first-party Codex CLI authenticates so the wire shape matches.
//!
//! The protocol itself lives in [`super::responses_wire`], shared with the API-key
//! [`super::responses`] backend. What is particular to this one is the endpoint, the OAuth
//! credential, the Codex client headers, and the two reasoning parameters Codex sends -- the
//! `include` of encrypted reasoning content and the `reasoning.summary` that makes the reasoning
//! visible. Those two stay here because this backend's endpoint is always ChatGPT.

mod auth;

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use self::auth::{extract_account_id, extract_expiration_seconds};
use super::responses_wire::{
    aggregate_stream, build_request_body, drive_responses_sse_stream, include_encrypted_reasoning,
    request_reasoning_summary,
};
use crate::{
    error::{MekaError, Result},
    provider::{
        AccountIdentity, AccountUsage, AuthCredential, DEFAULT_CHATGPT_SUBSCRIPTION_CLIENT_ID,
        DailyUsage, ExtraUsage, Message, Notice, Provider, StopReason, StreamEvent, TokenUsage,
        ToolDefinition, UsageHistory, UsageWindow,
    },
    session::TokenStore,
};

/// Default OAuth token endpoint. Refresh requests POST here as JSON.
const DEFAULT_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";

/// `originator` request header value. Mirrors Codex's `codex_cli_rs` slot, flagged as the calling
/// tool so OpenAI can attribute traffic.
const ORIGINATOR: &str = "meka_cli";

pub struct ChatGptSubscriptionProvider {
    client: reqwest::Client,
    credential: tokio::sync::RwLock<AuthCredential>,
    /// Serialises refreshes without blocking readers. Held across the database and network awaits
    /// a refresh performs; `credential` is not.
    refresh_gate: tokio::sync::Mutex<()>,
    base_url: String,
    model: String,
    client_id: String,
    oauth_token_url: String,
    token_store: Option<Arc<TokenStore>>,
    /// Profile name this provider's credential is stored under, so refreshed tokens are written
    /// back to the correct `provider_credentials` row.
    credential_key: String,
    /// The settled `reasoning.effort` for the request body, resolved once at construction from the
    /// profile's override. `None` - the unconfigured case - skips the reasoning block entirely, so
    /// the Responses API applies its own default.
    resolved_effort: Option<String>,
    /// Per-request output token cap from the profile; `None` leaves the Responses API default.
    max_output_tokens: Option<u64>,
    user_agent: String,
}

impl ChatGptSubscriptionProvider {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        credential: AuthCredential,
        model: String,
        base_url: Option<String>,
        client_id: Option<String>,
        oauth_token_url: Option<String>,
        token_store: Option<Arc<TokenStore>>,
        credential_key: String,
        reasoning_effort: Option<String>,
        max_output_tokens: Option<u64>,
    ) -> Result<Self> {
        // chatgpt.com is fronted by Cloudflare; enabling the cookie jar lets bot-clearance cookies
        // (e.g. `__cf_bm`) persist across requests.
        let client = crate::provider::build_http_client("chatgpt-subscription", |builder| {
            builder.cookie_store(true)
        })?;

        let resolved_effort = crate::provider::resolve_effort_level(reasoning_effort.as_deref());
        Ok(Self {
            client,
            credential: tokio::sync::RwLock::new(credential),
            refresh_gate: tokio::sync::Mutex::new(()),
            base_url: crate::provider::normalize_base_url(
                base_url
                    .as_deref()
                    .unwrap_or(crate::provider::DEFAULT_CHATGPT_BASE_URL),
            ),
            model,
            client_id: client_id
                .unwrap_or_else(|| DEFAULT_CHATGPT_SUBSCRIPTION_CLIENT_ID.to_string()),
            oauth_token_url: oauth_token_url.unwrap_or_else(|| DEFAULT_TOKEN_URL.to_string()),
            token_store,
            credential_key,
            resolved_effort,
            max_output_tokens,
            user_agent: format!(
                "meka/{} ({}; {})",
                env!("CARGO_PKG_VERSION"),
                std::env::consts::OS,
                std::env::consts::ARCH
            ),
        })
    }

    /// The settled reasoning-effort to send as `reasoning.effort` (see [`Self::resolved_effort`]).
    fn wire_effort(&self) -> Option<String> {
        self.resolved_effort.clone()
    }

    /// The request body: the shared Responses encoding, plus the one thing this backend may add
    /// that its API-key sibling may not.
    ///
    /// A named method rather than inline in `stream` so the `include` can be asserted without a
    /// live endpoint. It is the half of the split that has to keep *sending*, and a test that only
    /// covered the other half would let it fall away silently.
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
        // Safe here and only here: this backend's endpoint is always ChatGPT, and the first-party
        // Codex client asks for the same two things -- a summary so the reasoning is visible, and
        // the encrypted content so it survives a stateless round trip. Summary first: it settles
        // the `reasoning` object the `include` keys off.
        request_reasoning_summary(&mut body);
        include_encrypted_reasoning(&mut body);
        body
    }

    /// Returns the URL the request POSTs to. Codex's own client appends `/backend-api`
    /// automatically when the base URL is one of the chatgpt.com domains, but we keep the path
    /// explicit so a profile whose `base_url` names a custom proxy doesn't need its author to know
    /// the rewrite rule.
    fn responses_url(&self) -> String {
        let base = &self.base_url;
        if base.contains("/backend-api") || base.contains("/codex") {
            format!("{}/responses", base)
        } else {
            format!("{}/backend-api/codex/responses", base)
        }
    }

    /// URL of the ChatGPT-backend usage endpoint (`/wham/usage`), which lives under `/backend-api`
    /// alongside the responses endpoint.
    fn usage_url(&self) -> String {
        let base = &self.base_url;
        if base.contains("/backend-api") {
            format!("{}/wham/usage", base)
        } else {
            format!("{}/backend-api/wham/usage", base)
        }
    }

    /// URL of the ChatGPT-backend token-usage-profile endpoint (`/wham/profiles/me`).
    fn profiles_url(&self) -> String {
        let base = &self.base_url;
        if base.contains("/backend-api") {
            format!("{}/wham/profiles/me", base)
        } else {
            format!("{}/backend-api/wham/profiles/me", base)
        }
    }

    /// GET `/wham/usage` and parse it. Shared by `fetch_usage` (rate-limit windows) and
    /// `fetch_identity` (the `plan_type` field), which both read this one payload.
    async fn fetch_wham_usage(&self) -> Result<CodexUsageResponse> {
        let (bearer, account_id) = self.ensure_valid_credential().await?;
        // Not `apply_headers`: that sets `Accept: text/event-stream` for the SSE responses call,
        // but the usage endpoint returns plain JSON.
        let mut request = self
            .client
            .get(self.usage_url())
            .header("Authorization", format!("Bearer {}", bearer))
            .header("originator", ORIGINATOR)
            .header("User-Agent", &self.user_agent)
            .header("Accept", "application/json");
        if let Some(account_id) = account_id.as_deref() {
            request = request.header("ChatGPT-Account-ID", account_id);
        }
        let response = request.send().await.map_err(|error| {
            crate::error::provider_transport_error("Codex usage request failed", &error, None)
        })?;
        let status = response.status();
        let retry_after = crate::error::parse_retry_after(response.headers());
        let response_text = response.text().await.map_err(|error| {
            crate::error::provider_transport_error(
                "failed to read Codex usage response",
                &error,
                retry_after,
            )
        })?;
        if !status.is_success() {
            return Err(crate::error::provider_http_error(
                status,
                &response_text,
                retry_after,
                crate::error::ProviderRequest::Auxiliary,
            ));
        }
        serde_json::from_str(&response_text)
            .map_err(|error| MekaError::Provider(format!("invalid Codex usage JSON: {}", error)))
    }

    /// Returns `(bearer_token, account_id)`, refreshing the access token first if it's within 5
    /// minutes of expiry. The account_id is `Option<String>` because free-tier accounts may not
    /// have one (Codex's auth/manager.rs treats the missing claim as non-fatal).
    async fn ensure_valid_credential(&self) -> Result<(String, Option<String>)> {
        {
            let credential = self.credential.read().await;
            let AuthCredential::OAuthToken {
                access_token,
                expires_at,
                account_id,
                refresh_token,
            } = &*credential
            else {
                return Err(MekaError::Provider(
                    "chatgpt-subscription requires an OAuth token, not an API key".to_string(),
                ));
            };

            if !crate::provider::oauth_needs_refresh(
                *expires_at,
                refresh_token.is_some(),
                crate::provider::now_epoch_millis(),
            ) {
                return Ok((access_token.clone(), account_id.clone()));
            }
        }

        // Only refreshers queue here. `credential` is taken for the reads and writes themselves and
        // never held across the database or network awaits below: using its write lock as the
        // refresh gate meant a provider endpoint that went silent wedged every reader in the
        // process, not just the task refreshing. See the Claude provider for the same contract.
        let _refreshing = self.refresh_gate.lock().await;

        // And the same thing one layer out. `refresh_gate` is a `tokio::sync::Mutex`, so it
        // serialises the tasks in *this* process and says nothing about the meka in the next
        // terminal, which is holding the same refresh token and is just as due. Bounded, and
        // advisory: the compare-and-swap on the write is what makes the outcome correct whether or
        // not this is held.
        let _across_processes = match &self.token_store {
            Some(store) => {
                crate::provider::await_credential_lock(store, &self.credential_key).await
            }
            None => None,
        };

        // Re-read the latest credential from the DB. Refresh tokens rotate on each successful
        // refresh, and a sibling meka process may have rotated ours since startup. Without this
        // re-read we'd POST a stale refresh_token and the OAuth provider would reject it with
        // `invalid_grant`.
        if let Some(store) = &self.token_store {
            match store.load_provider_credential(&self.credential_key).await {
                Ok(Some(latest)) => *self.credential.write().await = latest,
                Ok(None) => {}
                Err(error) => {
                    tracing::warn!(
                        "failed to re-read Codex OAuth token before refresh: {}",
                        error
                    );
                }
            }
        }

        // Double-check after the DB re-read: another caller (in this process or a sibling meka) may
        // already have rotated to a still-valid access token.
        // Cloned whole, not just its refresh token: the swap below has to name the exact credential
        // this refresh is derived from, and anything less cannot tell "the row still holds what I
        // read" from "the row holds something equivalent".
        let derived_from = {
            let credential = self.credential.read().await;
            if let AuthCredential::OAuthToken {
                access_token,
                expires_at,
                account_id,
                refresh_token,
            } = &*credential
                && !crate::provider::oauth_needs_refresh(
                    *expires_at,
                    refresh_token.is_some(),
                    crate::provider::now_epoch_millis(),
                )
            {
                return Ok((access_token.clone(), account_id.clone()));
            }
            credential.clone()
        };
        let refresh_token = match &derived_from {
            AuthCredential::OAuthToken { refresh_token, .. } => refresh_token.clone(),
            _ => None,
        };
        let Some(refresh_token) = refresh_token else {
            return Err(MekaError::Provider(
                "OAuth access token expired and no refresh token available".to_string(),
            ));
        };

        let refreshed = self.refresh_oauth_token(&refresh_token).await?;

        // A refresh rotates the refresh token, so the one in the database is now dead -- but only
        // if the database still holds the one this was derived from. Where it does not, what comes
        // back is the newer credential to use instead of this one.
        let new_credential = match &self.token_store {
            Some(store) => {
                crate::provider::store_refreshed_credential(
                    store,
                    &self.credential_key,
                    &derived_from,
                    refreshed,
                )
                .await
            }
            None => refreshed,
        };

        let (token_value, account_id) = match &new_credential {
            AuthCredential::OAuthToken {
                access_token,
                account_id,
                ..
            } => (access_token.clone(), account_id.clone()),
            // Unreachable as things stand, and kept anyway. `store_refreshed_credential` only
            // hands back a credential of the same kind this refresh was derived from, and that is
            // an `OAuthToken` at every path reaching here. If that ever stops being true this is
            // the difference between a clear refusal and an empty bearer token on the wire.
            _ => {
                return Err(MekaError::Provider(
                    "the stored credential for this profile is not an OAuth token".to_string(),
                ));
            }
        };

        *self.credential.write().await = new_credential;
        Ok((token_value, account_id))
    }

    async fn refresh_oauth_token(&self, refresh_token: &str) -> Result<AuthCredential> {
        tracing::info!("refreshing Codex OAuth token");

        #[derive(Deserialize)]
        struct RefreshResponse {
            id_token: Option<String>,
            access_token: Option<String>,
            refresh_token: Option<String>,
        }

        // The exchange itself is shared with the Claude backend; what is this backend's own is
        // everything below, because ChatGPT's issuer states neither an expiry nor an account and
        // both have to be read out of the JWTs it returns.
        let data: RefreshResponse =
            crate::provider::exchange_refresh_token(crate::provider::RefreshExchange {
                client: &self.client,
                token_url: &self.oauth_token_url,
                client_id: &self.client_id,
                refresh_token,
                profile: &self.credential_key,
                context: "Codex OAuth token refresh",
            })
            .await?;

        let access_token = data.access_token.ok_or_else(|| {
            MekaError::Provider("Codex refresh response missing access_token".to_string())
        })?;

        // Re-extract `chatgpt_account_id` from the new id_token if the server returned one: the
        // workspace association can change.
        let account_id = match data.id_token.as_deref() {
            Some(id_token) => extract_account_id(id_token).ok().flatten(),
            None => None,
        };

        // expires_at comes from the access_token JWT's `exp` claim.
        let expires_at = Some(match extract_expiration_seconds(&access_token) {
            // A nonsense `exp` reads as "far future" and lets the 401 correct it, rather than
            // overflowing to a past instant and refreshing on every request.
            Ok(Some(seconds)) => seconds.checked_mul(1000).unwrap_or(i64::MAX),
            // A token carrying no readable `exp` gets an assumed lifetime rather than `None`, which
            // reads as due and would send every later request back through this whole path,
            // rotating the refresh token each time.
            _ => crate::provider::oauth_assumed_expiry(crate::provider::now_epoch_millis()),
        });

        Ok(AuthCredential::OAuthToken {
            access_token,
            refresh_token: data
                .refresh_token
                .or_else(|| Some(refresh_token.to_string())),
            expires_at,
            account_id,
        })
    }

    fn apply_headers(
        &self,
        request: reqwest::RequestBuilder,
        bearer: &str,
        account_id: Option<&str>,
    ) -> reqwest::RequestBuilder {
        let mut request = request
            .header("Authorization", format!("Bearer {}", bearer))
            .header("originator", ORIGINATOR)
            .header("User-Agent", &self.user_agent)
            .header("Accept", "text/event-stream")
            .header("Content-Type", "application/json");
        if let Some(account_id) = account_id {
            request = request.header("ChatGPT-Account-ID", account_id);
        }
        request
    }
}

#[async_trait]
impl Provider for ChatGptSubscriptionProvider {
    async fn complete(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<(Message, StopReason, TokenUsage, Vec<Notice>)> {
        // The Responses API on chatgpt.com is SSE-only; there is no non-streaming JSON shape to
        // parse. Satisfy the `complete` contract by consuming our own stream: drive `stream` into a
        // local channel and fold the events into the tuple a non-streaming provider would return.
        // Used by the provider-agnostic compaction path, which needs a full completion. Runs the
        // stream and the drain concurrently so a summary longer than the channel buffer can't
        // deadlock; the typed error (if any) surfaces from `stream_result`.
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

        let (bearer, account_id) = self.ensure_valid_credential().await?;

        let request = self
            .apply_headers(
                self.client.post(self.responses_url()),
                &bearer,
                account_id.as_deref(),
            )
            .json(&body);

        let response = request.send().await.map_err(|error| {
            crate::error::provider_transport_error("Codex HTTP request failed", &error, None)
        })?;

        drive_responses_sse_stream(response, event_sender, cancellation).await
    }

    fn name(&self) -> &str {
        "chatgpt-subscription"
    }

    fn resolved_effort(&self) -> Option<String> {
        self.wire_effort()
    }

    async fn fetch_usage(&self) -> Result<Option<AccountUsage>> {
        Ok(Some(self.fetch_wham_usage().await?.into_account_usage()))
    }

    async fn fetch_history(&self) -> Result<Option<UsageHistory>> {
        let (bearer, account_id) = self.ensure_valid_credential().await?;
        let mut request = self
            .client
            .get(self.profiles_url())
            .header("Authorization", format!("Bearer {}", bearer))
            .header("originator", ORIGINATOR)
            .header("User-Agent", &self.user_agent)
            .header("Accept", "application/json");
        if let Some(account_id) = account_id.as_deref() {
            request = request.header("ChatGPT-Account-ID", account_id);
        }
        let response = request.send().await.map_err(|error| {
            crate::error::provider_transport_error("Codex profile request failed", &error, None)
        })?;
        let status = response.status();
        let retry_after = crate::error::parse_retry_after(response.headers());
        let text = response.text().await.map_err(|error| {
            crate::error::provider_transport_error(
                "failed to read Codex profile response",
                &error,
                retry_after,
            )
        })?;
        if !status.is_success() {
            return Err(crate::error::provider_http_error(
                status,
                &text,
                retry_after,
                crate::error::ProviderRequest::Auxiliary,
            ));
        }
        let parsed: CodexProfileResponse = serde_json::from_str(&text).map_err(|error| {
            MekaError::Provider(format!("invalid Codex profile JSON: {}", error))
        })?;
        Ok(Some(parsed.into_history()))
    }

    async fn fetch_identity(&self) -> Result<Option<AccountIdentity>> {
        // The plan is the one identity field the usage payload carries; name/org/role need
        // `accounts/check` (a documented follow-up), so leave them `None` for now.
        let plan = self.fetch_wham_usage().await?.plan_type;
        Ok(Some(AccountIdentity {
            display_name: None,
            email: None,
            plan,
            tier: None,
            subscription_status: None,
            organization: None,
            role: None,
        }))
    }
}

/// Subset of the ChatGPT backend `GET /wham/usage` body that we render. Mirrors the fields the
/// Codex CLI reads (`RateLimitStatusPayload`), tolerant of absent/null buckets.
#[derive(Deserialize)]
struct CodexUsageResponse {
    #[serde(default)]
    plan_type: Option<String>,
    #[serde(default)]
    rate_limit: Option<CodexRateLimit>,
    #[serde(default)]
    credits: Option<CodexCredits>,
    #[serde(default)]
    spend_control: Option<CodexSpendControl>,
}

#[derive(Deserialize)]
struct CodexRateLimit {
    #[serde(default)]
    primary_window: Option<CodexWindow>,
    #[serde(default)]
    secondary_window: Option<CodexWindow>,
}

#[derive(Deserialize)]
struct CodexWindow {
    #[serde(default)]
    used_percent: Option<f64>,
    #[serde(default)]
    limit_window_seconds: Option<i64>,
    #[serde(default)]
    reset_at: Option<i64>,
}

#[derive(Deserialize)]
struct CodexCredits {
    #[serde(default)]
    has_credits: Option<bool>,
    /// Dollar string, e.g. `"9.99"` or `"$9.99"`.
    #[serde(default)]
    balance: Option<String>,
}

#[derive(Deserialize)]
struct CodexSpendControl {
    #[serde(default)]
    individual_limit: Option<CodexIndividualLimit>,
}

#[derive(Deserialize)]
struct CodexIndividualLimit {
    /// Dollar string of the amount spent against the cap.
    #[serde(default)]
    used: Option<String>,
    #[serde(default)]
    used_percent: Option<f64>,
}

impl CodexUsageResponse {
    fn into_account_usage(self) -> AccountUsage {
        let mut windows = Vec::new();
        if let Some(rate_limit) = self.rate_limit {
            push_codex_window(&mut windows, rate_limit.primary_window, "Primary");
            push_codex_window(&mut windows, rate_limit.secondary_window, "Secondary");
        }
        let note = self
            .plan_type
            .filter(|plan| !plan.is_empty())
            .map(|plan| format!("plan: {plan}"));
        AccountUsage {
            windows,
            extra_usage: codex_extra_usage(self.credits, self.spend_control),
            note,
        }
    }
}

/// Parse a dollar string like `"$9.99"` / `"9.99"` / `"1,234.50"` into an `f64`.
fn parse_dollars(value: &str) -> Option<f64> {
    value
        .trim()
        .trim_start_matches('$')
        .replace(',', "")
        .parse::<f64>()
        .ok()
}

/// Normalize Codex's `credits` + `spend_control` blocks into [`ExtraUsage`].
fn codex_extra_usage(
    credits: Option<CodexCredits>,
    spend_control: Option<CodexSpendControl>,
) -> Option<ExtraUsage> {
    if credits.is_none() && spend_control.is_none() {
        return None;
    }
    let (has_credits, balance) = match credits {
        Some(credits) => (
            credits.has_credits.unwrap_or(false),
            credits.balance.as_deref().and_then(parse_dollars),
        ),
        None => (false, None),
    };
    let (used, utilization) = match spend_control.and_then(|control| control.individual_limit) {
        Some(limit) => (
            limit.used.as_deref().and_then(parse_dollars),
            limit.used_percent,
        ),
        None => (None, None),
    };
    Some(ExtraUsage {
        // Extra usage is active if the account holds credits or has recorded spend against a cap;
        // keying only on `has_credits` would mislabel spend-only accounts as "disabled".
        enabled: has_credits || used.is_some(),
        utilization,
        used,
        balance,
        currency: None,
    })
}

fn push_codex_window(windows: &mut Vec<UsageWindow>, window: Option<CodexWindow>, fallback: &str) {
    if let Some(window) = window
        && let Some(used_percent) = window.used_percent
    {
        windows.push(UsageWindow {
            label: codex_window_label(window.limit_window_seconds, fallback),
            used_percent,
            resets_at: window.reset_at,
        });
    }
}

/// Subset of Codex's `GET /wham/profiles/me` body (`TokenUsageProfile`).
#[derive(Deserialize)]
struct CodexProfileResponse {
    #[serde(default)]
    stats: Option<CodexProfileStats>,
}

#[derive(Deserialize)]
struct CodexProfileStats {
    #[serde(default)]
    lifetime_tokens: Option<i64>,
    #[serde(default)]
    peak_daily_tokens: Option<i64>,
    #[serde(default)]
    current_streak_days: Option<i64>,
    #[serde(default)]
    longest_streak_days: Option<i64>,
    #[serde(default)]
    daily_usage_buckets: Vec<CodexDailyBucket>,
}

#[derive(Deserialize)]
struct CodexDailyBucket {
    start_date: String,
    tokens: i64,
}

impl CodexProfileResponse {
    fn into_history(self) -> UsageHistory {
        let stats = self.stats;
        UsageHistory {
            lifetime_tokens: stats.as_ref().and_then(|s| s.lifetime_tokens),
            peak_daily_tokens: stats.as_ref().and_then(|s| s.peak_daily_tokens),
            current_streak_days: stats.as_ref().and_then(|s| s.current_streak_days),
            longest_streak_days: stats.as_ref().and_then(|s| s.longest_streak_days),
            first_used: None,
            daily: stats
                .map(|s| {
                    s.daily_usage_buckets
                        .into_iter()
                        .map(|bucket| DailyUsage {
                            date: bucket.start_date,
                            tokens: bucket.tokens,
                        })
                        .collect()
                })
                .unwrap_or_default(),
        }
    }
}

/// Human label for a window from its duration (seconds). Common durations get friendly names; the
/// rest fall back to the primary/secondary position label.
fn codex_window_label(limit_window_seconds: Option<i64>, fallback: &str) -> String {
    let Some(minutes) = limit_window_seconds.map(|seconds| seconds / 60) else {
        return fallback.to_string();
    };
    match minutes {
        m if m == 7 * 24 * 60 => "Weekly".to_string(),
        m if m % (24 * 60) == 0 => format!("{}-day", m / (24 * 60)),
        m if m % 60 == 0 => format!("{}-hour", m / 60),
        m => format!("{m}-min"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_credential() -> AuthCredential {
        AuthCredential::OAuthToken {
            access_token: "access-test".to_string(),
            refresh_token: Some("refresh-test".to_string()),
            // 1 day in the future to avoid the refresh path during construction.
            expires_at: Some(crate::provider::now_epoch_millis() + 86_400_000),
            account_id: Some("workspace-test".to_string()),
        }
    }

    fn test_provider() -> ChatGptSubscriptionProvider {
        ChatGptSubscriptionProvider::new(
            test_credential(),
            "gpt-5".to_string(),
            None,
            None,
            None,
            None,
            "test".to_string(),
            Some("high".to_string()),
            None,
        )
        .expect("provider")
    }

    /// A stream that never started is retryable here too.
    ///
    /// The Codex sibling of the pair in `provider::anthropic::subscription`: its own `.send()`
    /// site, so its own wiring into [`crate::error::provider_transport_error`] and its own chance
    /// to regress to a bare `MekaError::Provider` that the agent loop discards. `test_credential`'s
    /// expiry is a day out, so `ensure_valid_credential` takes no refresh round trip first.
    #[tokio::test]
    async fn a_stream_that_could_not_start_reports_a_retryable_failure() {
        // Bound and dropped, so the port is refused rather than answered or hung.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind an ephemeral port");
        let port = listener.local_addr().expect("the bound address").port();
        drop(listener);

        let provider = ChatGptSubscriptionProvider::new(
            test_credential(),
            "gpt-5".to_string(),
            Some(format!("http://127.0.0.1:{port}/v1")),
            None,
            None,
            None,
            "test".to_string(),
            Some("high".to_string()),
            None,
        )
        .expect("provider");
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

    /// The history probe classifies a dead endpoint the same way the turn path does.
    ///
    /// Its own `.send()` and response-read, so its own chance to bypass the shared classifier.
    ///
    /// Nothing retries this -- its caller is `meka account stats`, outside any retry loop -- so
    /// classifying it changes only how the failure reads. Worth pinning anyway: it is the same
    /// classifier the turn path depends on, and a probe that stops calling it has grown its own
    /// private rules.
    #[tokio::test]
    async fn the_history_probe_reports_a_dead_endpoint_as_retryable() {
        // Bound and dropped, so the port is refused rather than answered or hung.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind an ephemeral port");
        let port = listener.local_addr().expect("the bound address").port();
        drop(listener);

        let provider = ChatGptSubscriptionProvider::new(
            test_credential(),
            "gpt-5".to_string(),
            Some(format!("http://127.0.0.1:{port}/v1")),
            None,
            None,
            None,
            "test".to_string(),
            Some("high".to_string()),
            None,
        )
        .expect("provider");
        let error = provider
            .fetch_history()
            .await
            .expect_err("nothing is listening there");

        assert!(
            matches!(error, MekaError::RetryableProvider { .. }),
            "{error}"
        );
    }

    /// This backend keeps the `include` its API-key sibling refuses to send.
    ///
    /// The other half of the split asserted in `openai-responses`: there, an OpenAI extension must
    /// never reach an endpoint that may not implement it; here, the endpoint is always ChatGPT and
    /// the first-party Codex client asks for the same thing, so reasoning survives the stateless
    /// round trip. Dropping it here would be silent -- the requests would still succeed, just
    /// without reasoning carried across turns.
    #[test]
    fn the_subscription_asks_chatgpt_to_round_trip_its_reasoning() {
        let body = test_provider().build_body("s", &[Message::user("hi")], &[]);
        assert_eq!(body["reasoning"]["effort"], "high");
        let include = body["include"].as_array().expect("include");
        assert!(
            include
                .iter()
                .any(|value| value == "reasoning.encrypted_content"),
            "{body}"
        );
    }

    /// The summary is the only part of the reasoning a person ever sees. Without it the model
    /// still thinks, the stream carries no summary deltas, and a long think renders as a hang.
    #[test]
    fn the_subscription_asks_chatgpt_to_summarise_its_reasoning() {
        let body = test_provider().build_body("s", &[Message::user("hi")], &[]);
        assert_eq!(body["reasoning"]["summary"], "auto", "{body}");
    }

    /// Neither ask may hinge on an effort being configured: with none, the shared body omits
    /// `reasoning` entirely, so there is nothing for the `include` to attach to, and the profile a
    /// user gets by default would ask ChatGPT for neither a summary nor encrypted reasoning. Codex
    /// sends `reasoning` on every request and omits only the fields it has no value for.
    #[test]
    fn an_unconfigured_profile_still_asks_for_a_summary_and_encrypted_reasoning() {
        let unconfigured = ChatGptSubscriptionProvider::new(
            test_credential(),
            "gpt-5.6-sol".to_string(),
            None,
            None,
            None,
            None,
            "test".to_string(),
            None,
            None,
        )
        .expect("provider");
        let body = unconfigured.build_body("s", &[Message::user("hi")], &[]);

        assert!(body["reasoning"].get("effort").is_none(), "{body}");
        assert_eq!(body["reasoning"]["summary"], "auto", "{body}");
        assert_eq!(
            body["include"],
            serde_json::json!(["reasoning.encrypted_content"]),
            "{body}"
        );
    }

    #[test]
    fn test_provider_name() {
        assert_eq!(test_provider().name(), "chatgpt-subscription");
    }

    #[test]
    fn an_unconfigured_profile_sends_no_reasoning_effort() {
        // Effort belongs to the provider: unset means the Responses API applies its own default,
        // which meka asks for by omitting the field rather than by naming a tier.
        let unconfigured = ChatGptSubscriptionProvider::new(
            test_credential(),
            "gpt-5.6-sol".to_string(),
            None,
            None,
            None,
            None,
            "test".to_string(),
            None,
            None,
        )
        .expect("provider");
        assert_eq!(unconfigured.resolved_effort(), None);

        let configured = ChatGptSubscriptionProvider::new(
            test_credential(),
            "gpt-5.6-sol".to_string(),
            None,
            None,
            None,
            None,
            "test".to_string(),
            Some("medium".to_string()),
            None,
        )
        .expect("provider");
        assert_eq!(configured.resolved_effort().as_deref(), Some("medium"));
    }

    #[test]
    fn test_usage_url_default_appends_backend_api_wham() {
        assert_eq!(
            test_provider().usage_url(),
            "https://chatgpt.com/backend-api/wham/usage"
        );
    }

    #[test]
    fn test_codex_usage_maps_windows_and_note() {
        // Shaped like the ChatGPT-backend `/wham/usage` body (RateLimitStatusPayload).
        let body = r#"{
            "plan_type": "plus",
            "rate_limit": {
                "allowed": true,
                "limit_reached": false,
                "primary_window": {"used_percent": 42, "limit_window_seconds": 18000, "reset_at": 123},
                "secondary_window": {"used_percent": 84, "limit_window_seconds": 604800, "reset_at": 456}
            },
            "credits": {"has_credits": true, "unlimited": false, "balance": "9.99"}
        }"#;
        let usage = serde_json::from_str::<CodexUsageResponse>(body)
            .expect("parse")
            .into_account_usage();
        assert_eq!(usage.windows.len(), 2);
        // 18000s = 300min -> 5-hour; 604800s = 10080min -> Weekly.
        assert_eq!(usage.windows[0].label, "5-hour");
        assert_eq!(usage.windows[0].used_percent, 42.0);
        assert_eq!(usage.windows[0].resets_at, Some(123));
        assert_eq!(usage.windows[1].label, "Weekly");
        // Plan stays in the note; credits move to extra_usage.
        assert_eq!(usage.note.as_deref(), Some("plan: plus"));
        let extra = usage.extra_usage.expect("extra_usage");
        assert!(extra.enabled);
        assert_eq!(extra.balance, Some(9.99));
    }

    #[test]
    fn test_codex_extra_usage_parses_spend_control() {
        let body = r#"{
            "plan_type": "pro",
            "credits": {"has_credits": true, "unlimited": false, "balance": "$5.00"},
            "spend_control": {"individual_limit": {"used": "$3.50", "used_percent": 70}}
        }"#;
        let extra = serde_json::from_str::<CodexUsageResponse>(body)
            .unwrap()
            .into_account_usage()
            .extra_usage
            .expect("extra_usage");
        assert!(extra.enabled);
        assert_eq!(extra.balance, Some(5.0));
        assert_eq!(extra.used, Some(3.5));
        assert_eq!(extra.utilization, Some(70.0));
    }

    #[test]
    fn test_codex_extra_usage_spend_only_is_enabled() {
        // No purchased credits, but recorded spend against a cap: must render as enabled, not
        // "disabled · $X spent".
        let body = r#"{
            "spend_control": {"individual_limit": {"used": "$3.50", "used_percent": 70}}
        }"#;
        let extra = serde_json::from_str::<CodexUsageResponse>(body)
            .unwrap()
            .into_account_usage()
            .extra_usage
            .expect("extra_usage");
        assert!(extra.enabled);
        assert_eq!(extra.used, Some(3.5));
    }

    #[test]
    fn test_codex_window_missing_used_percent_is_skipped_not_fatal() {
        // A partial window object (no `used_percent`) degrades to being dropped rather than failing
        // the whole payload; the complete sibling window still parses.
        let body = r#"{
            "rate_limit": {
                "primary_window": {"limit_window_seconds": 18000, "reset_at": 123},
                "secondary_window": {"used_percent": 84, "limit_window_seconds": 604800}
            }
        }"#;
        let usage = serde_json::from_str::<CodexUsageResponse>(body)
            .expect("partial window must not fail the parse")
            .into_account_usage();
        assert_eq!(usage.windows.len(), 1);
        assert_eq!(usage.windows[0].label, "Weekly");
        assert_eq!(usage.windows[0].used_percent, 84.0);
    }

    #[test]
    fn test_codex_profile_maps_history() {
        let body = r#"{
            "stats": {
                "lifetime_tokens": 1200000,
                "peak_daily_tokens": 45000,
                "current_streak_days": 3,
                "longest_streak_days": 12,
                "daily_usage_buckets": [
                    {"start_date": "2026-06-30", "tokens": 8100},
                    {"start_date": "2026-07-01", "tokens": 12300}
                ]
            }
        }"#;
        let history = serde_json::from_str::<CodexProfileResponse>(body)
            .unwrap()
            .into_history();
        assert_eq!(history.lifetime_tokens, Some(1_200_000));
        assert_eq!(history.current_streak_days, Some(3));
        assert_eq!(history.daily.len(), 2);
        assert_eq!(history.daily[1].date, "2026-07-01");
        assert_eq!(history.daily[1].tokens, 12300);
    }

    #[test]
    fn test_codex_usage_empty_rate_limit_is_no_windows() {
        let usage = serde_json::from_str::<CodexUsageResponse>(r#"{"plan_type": "pro"}"#)
            .unwrap()
            .into_account_usage();
        assert!(usage.windows.is_empty());
        assert_eq!(usage.note.as_deref(), Some("plan: pro"));
    }

    #[test]
    fn test_responses_url_default_appends_backend_api_codex() {
        let provider = test_provider();
        assert_eq!(
            provider.responses_url(),
            "https://chatgpt.com/backend-api/codex/responses"
        );
    }

    #[test]
    fn test_responses_url_user_supplied_backend_api_path_preserved() {
        let provider = ChatGptSubscriptionProvider::new(
            test_credential(),
            "gpt-5".to_string(),
            Some("https://example.com/backend-api/codex".to_string()),
            None,
            None,
            None,
            "test".to_string(),
            None,
            None,
        )
        .expect("provider");
        assert_eq!(
            provider.responses_url(),
            "https://example.com/backend-api/codex/responses"
        );
    }

    #[test]
    fn test_responses_url_strips_trailing_slash() {
        let provider = ChatGptSubscriptionProvider::new(
            test_credential(),
            "gpt-5".to_string(),
            Some("https://chatgpt.com/".to_string()),
            None,
            None,
            None,
            "test".to_string(),
            None,
            None,
        )
        .expect("provider");
        assert_eq!(
            provider.responses_url(),
            "https://chatgpt.com/backend-api/codex/responses"
        );
    }

    #[tokio::test]
    async fn test_ensure_valid_credential_returns_token_and_account_id() {
        let provider = test_provider();
        let (bearer, account_id) = provider
            .ensure_valid_credential()
            .await
            .expect("valid credential");
        assert_eq!(bearer, "access-test");
        assert_eq!(account_id.as_deref(), Some("workspace-test"));
    }

    #[tokio::test]
    async fn test_ensure_valid_credential_rejects_api_key() {
        let provider = ChatGptSubscriptionProvider::new(
            AuthCredential::ApiKey("sk-test".to_string()),
            "gpt-5".to_string(),
            None,
            None,
            None,
            None,
            "test".to_string(),
            None,
            None,
        )
        .expect("provider");
        let result = provider.ensure_valid_credential().await;
        assert!(matches!(result, Err(MekaError::Provider(_))));
    }

    #[tokio::test]
    async fn test_ensure_valid_credential_no_refresh_token_when_expired() {
        // Token already expired, no refresh available → error.
        let provider = ChatGptSubscriptionProvider::new(
            AuthCredential::OAuthToken {
                access_token: "old".to_string(),
                refresh_token: None,
                expires_at: Some(crate::provider::now_epoch_millis() - 1_000),
                account_id: None,
            },
            "gpt-5".to_string(),
            None,
            None,
            None,
            None,
            "test".to_string(),
            None,
            None,
        )
        .expect("provider");
        let result = provider.ensure_valid_credential().await;
        assert!(matches!(result, Err(MekaError::Provider(ref m)) if m.contains("expired")));
    }

    /// The Codex refresh path classifies by what the token endpoint answered, like its Claude twin.
    ///
    /// The sibling of `anthropic::subscription::tests::
    /// what_the_token_endpoint_answered_decides_whether_a_refresh_is_retried`. The classification
    /// they exercise lives in one place (`crate::error::oauth_refresh_error`, reached through
    /// `provider::exchange_refresh_token`), so this is a wiring test: what it proves is that *this*
    /// backend reaches that path, with its own `context`, and that its answer still comes back as
    /// this backend's own message. A bare `MekaError::Provider` here kills a turn
    /// `ensure_valid_credential` was called from the middle of, on an outage at the token endpoint.
    #[tokio::test]
    async fn what_the_codex_token_endpoint_answered_decides_whether_a_refresh_is_retried() {
        for (status_line, body, retryable) in [
            (
                "503 Service Unavailable",
                r#"{"error":"temporarily_unavailable"}"#,
                true,
            ),
            ("400 Bad Request", r#"{"error":"invalid_grant"}"#, false),
        ] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind mock OAuth endpoint");
            let local = listener.local_addr().expect("local addr");
            tokio::spawn(async move {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let mut scratch = [0u8; 4096];
                // One read is enough to know the request arrived; the response follows whatever
                // was sent, and the body is small enough to land in a single segment.
                if socket.read(&mut scratch).await.is_err() {
                    return;
                }
                let response = format!(
                    "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    status_line,
                    body.len(),
                    body
                );
                if socket.write_all(response.as_bytes()).await.is_err() {
                    return;
                }
                if socket.shutdown().await.is_err() {
                    tracing::debug!("mock Codex refresh endpoint did not shut down cleanly");
                }
            });

            let credential = AuthCredential::OAuthToken {
                access_token: "stale".to_string(),
                refresh_token: Some("rt".to_string()),
                expires_at: Some(crate::provider::now_epoch_millis()),
                account_id: None,
            };
            let provider = ChatGptSubscriptionProvider::new(
                credential,
                "gpt-5".to_string(),
                None,
                None,
                Some(format!("http://{}/", local)),
                None,
                "work".to_string(),
                None,
                None,
            )
            .expect("build test provider");

            let error = provider
                .ensure_valid_credential()
                .await
                .expect_err("the endpoint did not hand back a usable token");

            // The `Codex` prefix is asserted because both backends now compose their messages from
            // one shared exchange, and this is the half that would lose its own voice if the
            // `context` a call site passes stopped being read.
            match error {
                MekaError::RetryableProvider { message, .. } if retryable => assert!(
                    message.starts_with("Codex OAuth token refresh failed ("),
                    "{status_line}: {message}"
                ),
                MekaError::Provider(message) if !retryable => {
                    assert!(
                        message.starts_with("Codex OAuth token refresh"),
                        "{status_line}: {message}"
                    );
                    assert!(
                        message.contains("meka provider login work"),
                        "{status_line} should name the profile to log in to: {message}"
                    );
                }
                other => panic!("{status_line} was classified wrongly: {other}"),
            }
        }
    }

    #[tokio::test]
    async fn test_aggregate_stream_folds_events_into_message() {
        // chatgpt-subscription is streaming-only; it satisfies `complete` by folding its own SSE.
        // Feed the event sequence a summary turn would emit and assert it aggregates into
        // one assistant text message carrying the reported stop reason, with no spurious
        // notices.
        let (sender, receiver) = mpsc::channel::<StreamEvent>(16);
        sender
            .send(StreamEvent::TextDelta("Summary: ".to_string()))
            .await
            .unwrap();
        sender
            .send(StreamEvent::TextDelta("all done.".to_string()))
            .await
            .unwrap();
        sender
            .send(StreamEvent::MessageEnd {
                stop_reason: StopReason::EndTurn,
            })
            .await
            .unwrap();
        drop(sender);

        let (message, stop_reason, _usage, notices) = aggregate_stream(receiver).await;
        assert_eq!(message.text_content(), "Summary: all done.");
        assert!(matches!(stop_reason, StopReason::EndTurn));
        assert!(notices.is_empty());
    }
}
