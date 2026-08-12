//! LLM provider abstraction. Defines the [`Provider`] trait, the shared message/content/tool types,
//! and the [`ProviderBuilder`] that returns a concrete Claude or OpenAI-compatible implementation.

mod claude;
/// `meka provider` subcommand suite (add/list/use/remove/login) and the provider OAuth login flows.
pub mod cli;
/// Scripted provider used by the ACP integration test. Available in debug builds only; release
/// builds don't pay the binary-size cost. Activated by the `MEKA_ACP_MOCK_PROVIDER` env var inside
/// `acp::run_acp`; never reachable from production paths otherwise.
#[cfg(debug_assertions)]
pub(crate) mod mock;
/// Resolving a model's metadata (context window + reasoning effort) in one cached, post-build pass.
pub(crate) mod model_metadata;
pub(crate) mod openai;
/// Backoff policy for retrying [`crate::error::MekaError::RetryableProvider`] failures.
pub(crate) mod retry;

use std::sync::Arc;

use async_trait::async_trait;
pub(crate) use claude::model_supports_adaptive_thinking;
pub use claude::{ClaudeApiProvider, ClaudeOAuthProvider};
pub use openai::{OpenAiCodexProvider, OpenAiProvider};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    error::{MekaError, Result},
    session::TokenStore,
};

pub(crate) const DEFAULT_CLAUDE_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";

/// Codex's hardcoded OpenAI OAuth client ID. Mirrors the value used by the first-party CLI at
/// `temp/codex/codex-rs/login/src/auth/manager.rs:869`.
pub(crate) const DEFAULT_OPENAI_CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

pub const SUPPORTED_PROVIDERS: &[&str] =
    &["openai-api", "openai-codex", "claude-api", "claude-oauth"];

tokio::task_local! {
    /// True while a sub-agent's turn is running. The Claude OAuth provider is shared across the
    /// main agent and its sub-agents via a single `Arc`, so per-request sub-agent attribution can't
    /// live on the provider; it rides this task-local instead. Mirrors Claude Code's
    /// `AsyncLocalStorage`-based `cc_is_subagent` attribution. Set via [`scope_subagent`] around the
    /// sub-agent run; read via [`is_subagent`] when building the billing header.
    static IS_SUBAGENT: bool;
}

/// Whether the current task is executing a sub-agent's turn. Returns `false` outside any
/// [`scope_subagent`] (the main agent, tests, etc.).
pub(crate) fn is_subagent() -> bool {
    IS_SUBAGENT.try_with(|flag| *flag).unwrap_or(false)
}

/// Run `future` with the sub-agent attribution flag set. `tokio::task_local` scopes the value to
/// this specific future, so parallel sub-agents (and the main agent) stay isolated.
pub(crate) async fn scope_subagent<F: std::future::Future>(future: F) -> F::Output {
    IS_SUBAGENT.scope(true, future).await
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthCredential {
    ApiKey(String),
    OAuthToken {
        access_token: String,
        refresh_token: Option<String>,
        expires_at: Option<i64>,
        /// Provider-flavoured identity carried alongside the bearer token. Currently only
        /// `openai-codex` populates this, the `chatgpt_account_id` extracted from the id_token
        /// JWT, sent on every request as `ChatGPT-Account-ID`. Claude OAuth leaves it `None`.
        account_id: Option<String>,
    },
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

/// Model metadata resolved from a provider's models API (Codex's `/backend-api/codex/models`,
/// Anthropic's `/v1/models/{id}`) and cached per `(profile, model)`. Every field is optional
/// because providers report different subsets and an unknown model yields nothing. Drives the
/// context-window resolution when the built-in table doesn't recognize a model, and (for providers
/// whose catalog reports it) the reasoning-effort default via [`resolve_effort_with_catalog`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Maximum total context window in tokens, if the provider reported it.
    pub context_window: Option<u64>,
    /// Reasoning-effort tiers the provider's models catalog reports for this model, lowercased
    /// (e.g. `["low", "medium", "high", "xhigh"]`). `None` means no effort catalog was consulted
    /// (Anthropic's models API exposes none, so always `None` for Claude); `Some` means a catalog
    /// answered, where an empty list authoritatively marks a non-reasoning model (effort omitted).
    /// When present it is authoritative for the effort default; otherwise the provider's
    /// name-based predicates decide.
    pub effort_levels: Option<Vec<String>>,
}

/// The shared reasoning-effort policy, applied identically by every provider that exposes an effort
/// knob (Claude `output_config.effort`, OpenAI `reasoning.effort`). The provider supplies two
/// model-capability booleans; this decides the wire value:
///
/// - an explicit override -> passed through verbatim (lowercased), **absolute**: never clamped to a
///   lower tier and never dropped, even on a model meka wouldn't pick effort for by default. The
///   user owns correctness for their chosen model/endpoint;
/// - no override, model supports effort -> the strongest default tier (`xhigh` where available,
///   else `high`), so a fresh profile gets strong reasoning without configuration;
/// - no override, model has no effort knob (legacy Claude, non-reasoning / unrecognized OpenAI
///   models) -> `None`, so the caller omits the field.
pub(crate) fn resolve_effort_level(
    configured: Option<&str>,
    supports_effort: bool,
    supports_xhigh: bool,
) -> Option<String> {
    // A blank override (empty or whitespace-only) is treated as unset, falling through to the
    // model-aware default rather than sending an empty `effort` the API would reject. A non-blank
    // value passes through verbatim (trimmed + lowercased); the user owns its correctness.
    match configured.map(str::trim).filter(|value| !value.is_empty()) {
        Some(value) => Some(value.to_ascii_lowercase()),
        None if supports_xhigh => Some("xhigh".to_string()),
        None if supports_effort => Some("high".to_string()),
        None => None,
    }
}

/// Reasoning-effort tiers meka considers when picking a *default* from a catalog, ordered weakest
/// to strongest and capped at `xhigh`. `max`/`ultra` are deliberately omitted: they are never
/// auto-selected (they stay opt-in via an explicit override), matching the name-based default.
const DEFAULT_EFFORT_TIERS: [&str; 6] = ["none", "minimal", "low", "medium", "high", "xhigh"];

/// The strongest tier meka will default to from a catalog's advertised `levels`: the highest-ranked
/// member of [`DEFAULT_EFFORT_TIERS`] present in the list. `None` when the catalog lists none of
/// those tiers (an empty list for a non-reasoning model, or the degenerate case of only
/// `max`/`ultra` being offered), which the caller turns into an omitted field. Picking from the
/// list - rather than a constant - keeps the default in-catalog: a model that caps at `medium`
/// defaults to `medium`, not an out-of-catalog `high` the API would reject.
fn strongest_catalog_tier(levels: &[String]) -> Option<String> {
    DEFAULT_EFFORT_TIERS
        .iter()
        .rev()
        .find(|tier| levels.iter().any(|level| level == *tier))
        .map(|tier| (*tier).to_string())
}

/// Apply the shared effort policy with the *catalog* as the authoritative capability source when
/// present, falling back to the provider's name-based predicates otherwise. `fetched_levels` is a
/// model's [`ModelInfo::effort_levels`] (the provider's `/models` catalog): when `Some`, the
/// default is the strongest tier the catalog actually offers (see [`strongest_catalog_tier`]) - an
/// empty list means no effort knob, so the field is omitted; when `None` (no catalog: Claude
/// always, the public OpenAI API, an unlisted model) the `name_supports_*` booleans decide. An
/// explicit override is absolute either way. This is the single place the "catalog beats name
/// predicate" ladder lives; only Codex has a catalog, so only its `refine_effort` calls this (the
/// other providers, having no catalog, settle effort at construction via [`resolve_effort_level`]).
pub(crate) fn resolve_effort_with_catalog(
    configured: Option<&str>,
    fetched_levels: Option<&[String]>,
    name_supports_effort: bool,
    name_supports_xhigh: bool,
) -> Option<String> {
    // An explicit (non-blank) override is absolute: passed through verbatim (lowercased), whatever
    // the catalog or predicates say. A blank override is treated as unset (see
    // [`resolve_effort_level`]).
    if let Some(value) = configured.map(str::trim).filter(|value| !value.is_empty()) {
        return Some(value.to_ascii_lowercase());
    }
    match fetched_levels {
        Some(levels) => strongest_catalog_tier(levels),
        None => resolve_effort_level(None, name_supports_effort, name_supports_xhigh),
    }
}

/// Best-effort context-window inference from a model name. `Some(n)` for a recognized family;
/// `None` when the model is unknown, which signals the caller to probe the provider API (and floor
/// at 128k if that fails). This table is meka's *authoritative* source for recognized models: it
/// encodes each model's real window as meka's request receives it, so it reflects the window the
/// request truly gets and wins over the live `/models` probe - which runs only for models this
/// table returns `None` for. A config override still wins over everything. Returning `None` (not
/// `Some(128_000)`) for unknowns is deliberate: it stops the 128k default from masquerading as
/// knowledge and short-circuiting the probe.
pub(crate) fn context_window_for_model(model: &str) -> Option<u64> {
    if model.contains("claude") {
        // Opus 4.6/4.7/4.8/5, Sonnet 4.6/5, and the Fable/Mythos 5 family ship a 1M window; Haiku
        // and pre-4.6 models are 200k. `model_supports_adaptive_thinking` is the same era boundary
        // that marks the 1M models. 1M is the default on the direct Messages API with no beta
        // header for these models, so neither claude-api nor claude-oauth sends
        // `context-1m` (matching Claude Code 2.1.219); this value is the window the request
        // actually gets on both backends.
        if model_supports_adaptive_thinking(model) {
            Some(1_000_000)
        } else {
            Some(200_000)
        }
    } else if model.contains("gpt-4.1") {
        Some(1_047_576)
    } else if model.contains("gpt-4o") {
        Some(128_000)
    } else if model.contains("gpt-5.4") || model.contains("gpt-5.5") || model.contains("gpt-5.6") {
        // gpt-5.4 / 5.5 / 5.6 (incl. the 5.6 sol/terra/luna tiers). Above 272k input, OpenAI bills
        // a premium tier, but the window is still one request.
        Some(1_050_000)
    } else if model.contains("gpt-5") {
        // Legacy gpt-5, and unrecognized future gpt-5.x conservatively floored here rather than at
        // the larger 5.4+ window.
        Some(400_000)
    } else if model.contains("o3") || model.contains("o4-mini") || model.contains("o1") {
        Some(200_000)
    } else {
        None
    }
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
        thinking: String,
        signature: Option<String>,
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
        #[serde(deserialize_with = "deserialize_tool_result_content")]
        content: Vec<ToolResultContent>,
        is_error: bool,
    },
}

/// Deserializes ToolResult content from either a string (legacy format) or an array of
/// ToolResultContent (new format).
fn deserialize_tool_result_content<'de, D>(
    deserializer: D,
) -> std::result::Result<Vec<ToolResultContent>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrVec {
        String(String),
        Vec(Vec<ToolResultContent>),
    }

    match StringOrVec::deserialize(deserializer)? {
        StringOrVec::String(text) => Ok(vec![ToolResultContent::Text { text }]),
        StringOrVec::Vec(vec) => Ok(vec),
    }
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
        signature: Option<String>,
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
/// (a dim hint for `Info`, a warn-colored line for `Warn`). Today only `Info` is used by the
/// image-redaction path; `Warn` is reserved for future provider-side recoverable conditions.
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

    #[allow(dead_code)]
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
#[allow(dead_code)]
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

/// Abstraction over an LLM provider (Claude API/OAuth, OpenAI, etc.). Implementors are held behind
/// `Arc<dyn Provider>` and shared across concurrent tool dispatch; calls must be safe to make in
/// parallel from multiple sub-agents in one turn.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Single round-trip request. Returns the assistant message, stop-reason, token-usage metadata,
    /// and any user-visible notices that arose during the request (e.g. the redaction hint from
    /// `claude::shared::build_body_within_budget`). The caller is expected to forward each notice
    /// to the active frontend; an empty `Vec` means nothing to surface. No streaming; the agent
    /// awaits the full response.
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

    /// Override thinking for the next API call. `Some(false)` disables, `Some(true)` enables,
    /// `None` restores the default. Default impl is a silent no-op. Providers that don't support
    /// thinking should leave it that way; providers that do must override.
    fn set_thinking_override(&self, _enabled: Option<bool>) {}

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

    /// Fetch metadata for the provider's active model (`self.model`) from its models API, used to
    /// resolve the context window when the built-in table doesn't recognize the model. Returns
    /// `Ok(None)` for providers/models with no such endpoint (the default; the public OpenAI API
    /// exposes nothing here). Codex and the Anthropic backends override it.
    async fn fetch_model_info(&self) -> Result<Option<ModelInfo>> {
        Ok(None)
    }

    /// Resolve this model's reasoning-effort default from `fetched` (the provider's `/models`
    /// catalog, authoritative when it reports effort levels) or, absent that, the provider's
    /// name-based predicates, and store the result for the request body. Called once post-build by
    /// [`crate::provider::model_metadata::resolve_model_metadata`] before any turn, so the wire
    /// path reads a settled value instead of re-deriving it per request. Default: no-op
    /// (providers with no effort knob). Effort-sending providers override; the shared ladder is
    /// [`resolve_effort_with_catalog`].
    fn refine_effort(&self, _fetched: Option<&ModelInfo>) {}

    /// Whether [`crate::provider::model_metadata::resolve_model_metadata`] should probe the models
    /// catalog to settle this model's reasoning effort. `true` only when the provider *has* an
    /// effort catalog (Codex) *and* the effort isn't already pinned by an explicit override: in
    /// that case the resolver probes even when the context window is table-known, so effort is
    /// catalog-accurate rather than name-guessed (cached, so once per `(profile, model)` per TTL).
    /// Returns `false` when there is no catalog (Claude, the public OpenAI API) or when an explicit
    /// `effort` override already determines the value (the catalog would change nothing, so a
    /// table-known window needs no probe at all). Default: `false`.
    fn needs_effort_catalog(&self) -> bool {
        false
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
            match serde_json::from_str::<serde_json::Value>(&accumulator.arguments) {
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

/// Constructs a concrete [`Provider`] (Claude API, Claude OAuth, or OpenAI-compatible) from a bag
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
    thinking_enabled: bool,
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
            thinking_enabled: false,
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

    /// OAuth client ID. Only consumed by `claude-oauth`.
    pub fn client_id(mut self, value: Option<String>) -> Self {
        self.client_id = value;
        self
    }

    /// OAuth token endpoint. Only consumed by `claude-oauth`.
    pub fn oauth_token_url(mut self, value: Option<String>) -> Self {
        self.oauth_token_url = value;
        self
    }

    /// Sink for refreshed OAuth tokens. Only consumed by `claude-oauth`; when `None`, refreshed
    /// tokens are held in memory only.
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

    /// Claude-only: turn on extended thinking with the given budget cap. Ignored by `openai-api`.
    pub fn thinking(mut self, enabled: bool, budget_tokens: u64) -> Self {
        self.thinking_enabled = enabled;
        self.thinking_budget_tokens = budget_tokens;
        self
    }

    /// Stable device identity embedded in `metadata.user_id`. Only consumed by `claude-oauth`.
    pub fn device_id(mut self, value: String) -> Self {
        self.device_id = value;
        self
    }

    /// The user's explicit reasoning-effort override (`low` / `medium` / `high` / `xhigh` / `max`),
    /// or `None` to let the provider pick a model-aware default. Consumed by every backend: Claude
    /// maps it to `output_config.effort`, OpenAI to `reasoning.effort`.
    pub fn effort(mut self, value: Option<String>) -> Self {
        self.effort = value;
        self
    }

    /// Request `redacted_thinking` blocks. Only consumed by `claude-oauth`.
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
    /// by `claude-oauth` and `claude-api`.
    pub fn session_stats(mut self, value: Option<Arc<crate::stats::SessionStats>>) -> Self {
        self.session_stats = value;
        self
    }

    pub fn build(self) -> Result<Arc<dyn Provider>> {
        match self.provider_name.as_str() {
            "openai-api" => {
                let api_key = match self.credential {
                    AuthCredential::ApiKey(key) => key,
                    AuthCredential::OAuthToken { access_token, .. } => access_token,
                };
                Ok(Arc::new(OpenAiProvider::new(
                    api_key,
                    self.model,
                    self.base_url,
                    self.effort,
                    self.max_output_tokens,
                )))
            }
            "claude-api" => {
                let api_key = match self.credential {
                    AuthCredential::ApiKey(key) => key,
                    AuthCredential::OAuthToken { .. } => {
                        return Err(MekaError::Config(
                            "provider 'claude-api' requires an API key, not an OAuth token. \
                             Use 'claude-oauth' for Claude Code OAuth."
                                .to_string(),
                        ));
                    }
                };
                Ok(Arc::new(ClaudeApiProvider::new(
                    api_key,
                    self.model,
                    self.base_url,
                    self.thinking_enabled,
                    self.thinking_budget_tokens,
                    self.effort,
                    self.max_output_tokens,
                    self.session_stats,
                )))
            }
            "claude-oauth" => {
                if matches!(self.credential, AuthCredential::ApiKey(_)) {
                    return Err(MekaError::Config(
                        "provider 'claude-oauth' requires an OAuth token, not an API key. \
                         Use 'claude-api' for direct API access."
                            .to_string(),
                    ));
                }
                Ok(Arc::new(ClaudeOAuthProvider::new(
                    self.credential,
                    self.model,
                    self.base_url,
                    self.client_id,
                    self.oauth_token_url,
                    self.token_store,
                    self.credential_key
                        .unwrap_or_else(|| self.provider_name.clone()),
                    self.thinking_enabled,
                    self.thinking_budget_tokens,
                    self.device_id,
                    self.effort,
                    self.redact_thinking,
                    self.max_output_tokens,
                    self.session_stats,
                )))
            }
            "openai-codex" => {
                if matches!(self.credential, AuthCredential::ApiKey(_)) {
                    return Err(MekaError::Config(
                        "provider 'openai-codex' requires an OAuth token, not an API key. \
                         Use 'openai-api' for direct API access."
                            .to_string(),
                    ));
                }
                Ok(Arc::new(OpenAiCodexProvider::new(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_effort_with_catalog() {
        let levels = |items: &[&str]| Some(items.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        // Catalog present + contains xhigh → strongest default is xhigh, regardless of name
        // guesses.
        assert_eq!(
            resolve_effort_with_catalog(
                None,
                levels(&["low", "high", "xhigh"]).as_deref(),
                false,
                false
            )
            .as_deref(),
            Some("xhigh")
        );
        // Catalog present without xhigh → high, even if the name predicate wrongly claimed xhigh.
        assert_eq!(
            resolve_effort_with_catalog(
                None,
                levels(&["low", "medium", "high"]).as_deref(),
                true,
                true
            )
            .as_deref(),
            Some("high")
        );
        // Catalog capped BELOW high → its strongest present tier (medium), not an out-of-catalog
        // "high" the API would reject. This is the crux of catalog-authoritative defaults.
        assert_eq!(
            resolve_effort_with_catalog(None, levels(&["low", "medium"]).as_deref(), true, true)
                .as_deref(),
            Some("medium")
        );
        assert_eq!(
            resolve_effort_with_catalog(
                None,
                levels(&["none", "minimal", "low"]).as_deref(),
                true,
                true
            )
            .as_deref(),
            Some("low")
        );
        // Empty catalog → no effort support → omit (name guesses ignored).
        assert_eq!(
            resolve_effort_with_catalog(None, Some(&[][..]), true, true),
            None
        );
        // No catalog → fall back to the name predicates.
        assert_eq!(
            resolve_effort_with_catalog(None, None, true, true).as_deref(),
            Some("xhigh")
        );
        assert_eq!(
            resolve_effort_with_catalog(None, None, true, false).as_deref(),
            Some("high")
        );
        assert_eq!(resolve_effort_with_catalog(None, None, false, false), None);
        // An explicit override is absolute: passed through verbatim whatever the
        // catalog/predicates.
        assert_eq!(
            resolve_effort_with_catalog(
                Some("low"),
                levels(&["high", "xhigh"]).as_deref(),
                true,
                true
            )
            .as_deref(),
            Some("low")
        );
        assert_eq!(
            resolve_effort_with_catalog(Some("xhigh"), Some(&[][..]), false, false).as_deref(),
            Some("xhigh")
        );
        // A blank override (empty / whitespace) is treated as unset: it must not short-circuit to
        // an empty wire value; it falls through to the catalog default or the name
        // predicate.
        assert_eq!(
            resolve_effort_with_catalog(Some("  "), None, true, true).as_deref(),
            Some("xhigh")
        );
        assert_eq!(
            resolve_effort_with_catalog(
                Some(""),
                levels(&["low", "medium"]).as_deref(),
                true,
                true
            )
            .as_deref(),
            Some("medium")
        );
    }

    #[test]
    fn test_context_window_for_model() {
        // gpt-5.4 / 5.5 / 5.6 family (incl. the 5.6 sol/terra/luna tiers) = 1.05M.
        assert_eq!(context_window_for_model("gpt-5.6-sol"), Some(1_050_000));
        assert_eq!(context_window_for_model("gpt-5.6-terra"), Some(1_050_000));
        assert_eq!(context_window_for_model("gpt-5.5"), Some(1_050_000));
        assert_eq!(context_window_for_model("gpt-5.4"), Some(1_050_000));
        // Legacy / bare gpt-5 floors lower.
        assert_eq!(context_window_for_model("gpt-5"), Some(400_000));
        assert_eq!(context_window_for_model("gpt-5-codex"), Some(400_000));
        // Unchanged families.
        assert_eq!(context_window_for_model("gpt-4.1"), Some(1_047_576));
        assert_eq!(context_window_for_model("gpt-4o"), Some(128_000));
        assert_eq!(context_window_for_model("o3"), Some(200_000));
        assert_eq!(context_window_for_model("o4-mini"), Some(200_000));
        // Claude: Opus 4.6+/Sonnet 4.6/Fable 5 ship 1M; Haiku 4.5 and pre-4.6 stay at 200k.
        assert_eq!(context_window_for_model("claude-opus-4-6"), Some(1_000_000));
        assert_eq!(context_window_for_model("claude-opus-4-8"), Some(1_000_000));
        assert_eq!(context_window_for_model("claude-opus-5"), Some(1_000_000));
        assert_eq!(context_window_for_model("claude-sonnet-5"), Some(1_000_000));
        assert_eq!(
            context_window_for_model("claude-sonnet-4-6"),
            Some(1_000_000)
        );
        assert_eq!(context_window_for_model("claude-fable-5"), Some(1_000_000));
        assert_eq!(context_window_for_model("claude-haiku-4-5"), Some(200_000));
        assert_eq!(context_window_for_model("claude-sonnet-4-5"), Some(200_000));
        assert_eq!(context_window_for_model("claude-3-5-sonnet"), Some(200_000));
        // Unknown model → None (the resolver then probes the API / floors at 128k).
        assert_eq!(context_window_for_model("some-unknown-model"), None);
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

    #[test]
    fn test_without_tool_use_keeps_text_and_thinking_drops_tool_use() {
        let message = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    thinking: "hmm".to_string(),
                    signature: None,
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

    #[test]
    fn test_create_provider_openai_api() {
        let result = ProviderBuilder::new(
            "openai-api",
            AuthCredential::ApiKey("key".to_string()),
            "gpt-4o",
        )
        .device_id("a".repeat(64))
        .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_provider_claude_api() {
        let result = ProviderBuilder::new(
            "claude-api",
            AuthCredential::ApiKey("key".to_string()),
            "claude-sonnet-4-20250514",
        )
        .thinking(false, 10000)
        .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_provider_claude_oauth() {
        let result = ProviderBuilder::new(
            "claude-oauth",
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
    fn test_create_provider_claude_api_rejects_oauth_token() {
        let result = ProviderBuilder::new(
            "claude-api",
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
    fn test_create_provider_claude_oauth_rejects_api_key() {
        let result = ProviderBuilder::new(
            "claude-oauth",
            AuthCredential::ApiKey("sk-ant-api03-test".to_string()),
            "claude-sonnet-4-20250514",
        )
        .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_create_provider_openai_codex() {
        let result = ProviderBuilder::new(
            "openai-codex",
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
    fn test_create_provider_openai_codex_rejects_api_key() {
        let result = ProviderBuilder::new(
            "openai-codex",
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
