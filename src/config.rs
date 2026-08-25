//! Configuration: parses `~/.config/meka/config.toml`, layers CLI overrides and environment
//! variables on top, and produces a [`ResolvedConfig`] that the rest of the binary consumes.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{
    cli::Cli,
    permission::{EnabledPermissions, Permission},
    render::{RenderMode, ToolParams},
};

/// In-memory shape of `config.toml`. Each top-level `[section]` deserializes into its own
/// sub-struct; missing sections fall back to `Default`. This is the raw deserialized form;
/// `resolve_config` merges it with CLI flags and env vars to produce a [`ResolvedConfig`].
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ConfigFile {
    /// Name of the profile to use when no `--provider` flag is given.
    pub default_provider: Option<String>,
    /// Named provider profiles, parsed from `[providers.<name>]`. Each pins a backend `type` plus
    /// non-secret settings; credentials live in the DB, not here.
    #[serde(default)]
    pub providers: std::collections::BTreeMap<String, ProviderProfile>,
    pub display: Option<DisplayConfig>,
    pub web: Option<WebConfig>,
    pub shell: Option<ShellConfig>,
    pub session: Option<SessionConfig>,
    pub thinking: Option<ThinkingConfig>,
    pub mcp: Option<McpConfig>,
    pub tools: Option<ToolsConfig>,
    pub skills: Option<SkillsConfig>,
    pub memory: Option<MemoryConfig>,
    pub permissions: Option<PermissionsConfig>,
    pub schedule: Option<ScheduleConfig>,
    pub background: Option<BackgroundConfig>,
    pub subagents: Option<SubagentsConfig>,
    pub serve: Option<ServeConfig>,
}

/// `[skills]` table: the user-authored skill store ([`crate::skills`]).
///
/// Same shape and rationale as [`MemoryConfig`]: config-only, no env var, no CLI flag. Turning it
/// off keeps the `skill` tool's schema out of every request and stops the `[Skills]` section from
/// rendering, for an installation that never uses skills.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct SkillsConfig {
    pub enabled: Option<bool>,
    /// Whether the agent may author skills, adding `skill_write` and `skill_delete` to its
    /// registry. Default `false`.
    ///
    /// Off by default because for the ordinary terminal session the human authors and curates
    /// skills, and an agent rewriting that store is not something the user asked for. It earns its
    /// keep in the opposite deployment: a long-running dispatcher that spawns workers, where a
    /// skill is the only artifact that both outlives the session and can be handed to a sub-agent
    /// as its task, so writing one is how the agent gets a refined worker brief to the next worker
    /// without routing it through its own context window.
    ///
    /// Never granted to a sub-agent whatever this says; see the registration site in
    /// [`crate::tools`].
    pub agent_managed: Option<bool>,
    /// Additional directories to scan for skills, **read-only**. Default empty.
    ///
    /// meka writes only to its own `skills/` directory under the config dir; these roots are never
    /// created and never written to, so listing one that does not exist costs nothing and leaves
    /// no trace. That is what makes it safe to point at a shared location like
    /// `~/.agents/skills`, which other Agent Skills clients populate: meka reads what is
    /// already there rather than putting anything in `$HOME` itself.
    ///
    /// Empty by default, and deliberately not defaulted to the cross-client convention: whether a
    /// directory outside meka's own namespace should be read is the user's call, not meka's.
    ///
    /// A leading `~` is expanded (see [`expand_user_path`]). A relative path is resolved against
    /// the process working directory, which is why nothing here is scanned implicitly: meka does
    /// not auto-trust the cwd, so a project's skills are read only when a path names them.
    pub extra_paths: Option<Vec<String>>,
}

/// `[schedule]` table: agent-created wakeups ([`crate::schedule`]).
///
/// Config-only, like `[skills]` and `[memory]`: whether an agent may schedule its own future turns
/// is a property of the installation. Turning it off keeps the three `schedule_*` tool schemas out
/// of every request, stops the `[Scheduled]` section rendering, and leaves the scheduler unstarted,
/// so existing jobs stay on disk without firing.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ScheduleConfig {
    pub enabled: Option<bool>,
    /// How often the scheduler looks for due jobs. Accepts humantime strings like `"10s"`, `"1m"`.
    /// Default `"10s"`. This is the real resolution floor: a job with a shorter interval fires
    /// once per tick, not once per interval.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub poll_interval: Option<std::time::Duration>,
    /// How late a *one-shot* job may be and still fire after downtime. Accepts humantime strings
    /// like `"24h"`. Default `"24h"`.
    ///
    /// Recurring jobs need no equivalent: their occurrences are one period apart, so the most
    /// recent missed one is always less than a period old, and the scheduler coalesces the rest.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub missed_grace: Option<std::time::Duration>,
    /// Wall-clock budget for a gate command. Accepts humantime strings like `"30s"`. Default
    /// `"30s"`. A gate that overruns is a failure, not a silent skip.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub gate_timeout: Option<std::time::Duration>,
    /// Ceiling on jobs per session, refused at `schedule_create`. Default 50.
    pub max_jobs: Option<usize>,
    /// How many of one session's jobs may spend a turn in a single sweep. Default 5. Jobs past it
    /// keep their occurrence and fire on the next sweep.
    pub max_consecutive_fires: Option<usize>,
}

/// [`ScheduleConfig`] with every default filled in.
#[derive(Debug, Clone)]
pub struct ResolvedScheduleConfig {
    pub enabled: bool,
    /// The permission level of the process running the scheduler.
    ///
    /// The **fallback** for a session row that carries no per-session level: the REPL and ACP
    /// paths derive theirs from process config, so for those the host's level is the session's
    /// level.
    ///
    /// Not a ceiling. `meka serve --permission read` does not stop a job whose own row carries a
    /// higher level, because every session `serve` creates records its own; the host's level only
    /// answers for a row that has none.
    ///
    /// Deliberately *not* a ceiling, which is what this comment used to claim. `--permission` on
    /// `meka serve` is a starting default, and every session `serve` itself creates persists a
    /// level of its own, so restarting at `read` does not disarm a gate an `unrestricted` session
    /// authored. What does bound it is `enabled_permissions` below, which the operator can only
    /// narrow in `config.toml` and which no session can exceed.
    pub host_permission: crate::permission::Permission,
    /// The modes this installation permits at all.
    ///
    /// The fire-time gate check reads a session's recorded level straight out of the row, and a
    /// row outlives the configuration that produced it. Without this, narrowing
    /// `[permissions].enabled` and restarting left an `unrestricted` row firing its gate
    /// forever while the live session re-attached at `read` -- verified end to end, and the
    /// exact fail-open the live re-check was added to prevent, reached from the one of five
    /// readers that did not apply the filter.
    pub enabled_permissions: crate::permission::EnabledPermissions,
    pub poll_interval: std::time::Duration,
    pub missed_grace: std::time::Duration,
    pub gate_timeout: std::time::Duration,
    pub max_jobs: usize,
    pub max_consecutive_fires: usize,
}

impl ResolvedScheduleConfig {
    const DEFAULT_GATE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
    /// Five is well above what a healthy session has due at once, since coalescing means one job
    /// contributes one fire however long the outage was, so the budget only engages on a real
    /// backlog. What it buys there is interleaving, not throttling: sweeps do not overlap and the
    /// next begins as soon as the last ends, so a backlog produces the same number of turns either
    /// way -- but another session's due job is reached after five of this one's rather than after
    /// all of them.
    const DEFAULT_MAX_CONSECUTIVE_FIRES: usize = 5;
    const DEFAULT_MAX_JOBS: usize = 50;
    /// A day. Long enough that an overnight outage still delivers this morning's reminder, short
    /// enough that a laptop closed for a week does not wake to a pile of stale ones.
    const DEFAULT_MISSED_GRACE: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);
    /// Ten seconds trades scheduling precision against wakeups: a minute-granularity cron never
    /// needs better, and it keeps a `--oneshot` run from paying for a tick it will never use.
    const DEFAULT_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_secs(10);

    fn resolve(
        raw: Option<ScheduleConfig>,
        host_permission: crate::permission::Permission,
        enabled_permissions: crate::permission::EnabledPermissions,
    ) -> Self {
        let raw = raw.unwrap_or_default();
        Self {
            host_permission,
            enabled_permissions,
            enabled: raw.enabled.unwrap_or(true),
            poll_interval: raw.poll_interval.unwrap_or(Self::DEFAULT_POLL_INTERVAL),
            missed_grace: raw.missed_grace.unwrap_or(Self::DEFAULT_MISSED_GRACE),
            gate_timeout: raw.gate_timeout.unwrap_or(Self::DEFAULT_GATE_TIMEOUT),
            max_jobs: raw.max_jobs.unwrap_or(Self::DEFAULT_MAX_JOBS),
            max_consecutive_fires: raw
                .max_consecutive_fires
                .unwrap_or(Self::DEFAULT_MAX_CONSECUTIVE_FIRES),
        }
    }
}

impl Default for ResolvedScheduleConfig {
    /// `None` for the host level, which is the most restrictive: a default-constructed config is a
    /// test fixture or a caller that never said, and neither should be able to run a gate.
    fn default() -> Self {
        // `DEFAULT` for the enabled set rather than something narrower: it can only ever *narrow*
        // what a row already recorded, and `from_modes` guarantees a non-empty set anyway.
        Self::resolve(
            None,
            crate::permission::Permission::None,
            crate::permission::EnabledPermissions::DEFAULT,
        )
    }
}

/// `[background]` table: tool calls the agent starts and does not wait for ([`crate::background`]).
///
/// Config-only, like `[schedule]`, `[skills]`, and `[memory]`.
///
/// Alone among them it defaults **off**, which is deliberate rather than caution. Those three add
/// capability without changing when a turn ends; this changes the contract of the primary
/// interaction from "you asked, it answered" into "and something else may interrupt you later",
/// which is right for an unattended assistant and wrong for someone using the REPL as a command
/// line. A scheduled job also takes an explicit, visible act to create, whereas `background` is
/// reachable from any tool call, so an agent will reach for it unprompted.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct BackgroundConfig {
    pub enabled: Option<bool>,
    /// Ceiling on tasks running at once per session, refused at dispatch. Default 10.
    pub max_tasks: Option<usize>,
}

/// [`BackgroundConfig`] with every default filled in.
#[derive(Debug, Clone)]
pub struct ResolvedBackgroundConfig {
    pub enabled: bool,
    pub max_tasks: usize,
}

impl ResolvedBackgroundConfig {
    /// Enough for a fan-out of parallel work, low enough that a runaway loop is capped before it
    /// has spawned a hundred shells.
    const DEFAULT_MAX_TASKS: usize = 10;

    fn resolve(raw: Option<BackgroundConfig>) -> Self {
        let raw = raw.unwrap_or_default();
        Self {
            enabled: raw.enabled.unwrap_or(false),
            max_tasks: raw.max_tasks.unwrap_or(Self::DEFAULT_MAX_TASKS),
        }
    }
}

impl Default for ResolvedBackgroundConfig {
    fn default() -> Self {
        Self::resolve(None)
    }
}

/// `[memory]` table: the agent's durable note store ([`crate::memory`]).
///
/// Config-only, with no env var and no CLI flag: whether an agent keeps memories is a property of
/// the installation, not something to vary per run. Turning it off keeps the four `memory_*` tool
/// schemas out of every request, which is the point for someone running lean coding sessions.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct MemoryConfig {
    pub enabled: Option<bool>,
}

/// `[serve]` table: HTTP server config for `meka serve`. All fields optional with sensible
/// defaults, but at least one `[[serve.tokens]]` entry is required; the server refuses to
/// start without one.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ServeConfig {
    /// Listen address. Default `127.0.0.1:8080`: bind to loopback so a fresh deploy isn't
    /// accidentally world-reachable. Operators front with a reverse proxy (nginx, caddy) for
    /// TLS termination and put a public address there.
    pub bind: Option<String>,
    /// Idle-timeout for session eviction. Sessions with no turn activity for this long are
    /// dropped from the in-memory map by the GC scanner. Accepts humantime strings like
    /// `"24h"`, `"30m"`, `"86400s"`. Default `"24h"`.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub idle_timeout: Option<std::time::Duration>,
    /// How often the GC scanner sweeps the session map. Accepts humantime strings like
    /// `"5m"`, `"300s"`. Default `"5m"`.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub gc_scan_interval: Option<std::time::Duration>,
    /// When true, GC also deletes the SQLite row for idle sessions; default false (keep the
    /// row so a future request with the same session ID can re-attach, mirroring ACP's
    /// `session/load`).
    pub delete_on_idle: Option<bool>,
    /// On SIGTERM / SIGINT, wait at most this long for in-flight turns to finish before
    /// forcibly aborting. Accepts humantime strings like `"30s"`, `"1m"`. Default `"30s"`.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub shutdown_drain_timeout: Option<std::time::Duration>,
    /// Process-wide cap on concurrent in-flight turns across all sessions. None = unbounded
    /// (default). Returns 429 with `concurrency-limit` when exceeded.
    pub max_concurrent_turns: Option<usize>,
    /// Request body size limit (bytes). Default 10 MiB.
    pub max_body_bytes: Option<usize>,
    /// Whether to serve the Swagger UI and the OpenAPI document at `/v1/docs` and
    /// `/v1/openapi.json`. Default `false`.
    ///
    /// Off by default because they are unauthenticated -- as are the two health probes, which
    /// publish nothing -- and what they publish is the shape of every endpoint the deployment
    /// exposes. That is useful while
    /// building a client and pure reconnaissance value once the deployment is real. Turn it on
    /// deliberately, on a deployment where anyone who can reach the port is entitled to the map.
    pub docs: Option<bool>,
    /// How many SSE events per turn to retain so a client reconnecting with `Last-Event-ID` can
    /// replay what it missed. Default 256, matching the live broadcast channel's capacity.
    ///
    /// Raising it buys a longer reconnect window at the cost of holding more per-session memory
    /// during a turn; `0` switches replay off, so a reconnect gets only what happens from then on.
    pub stream_replay_events: Option<usize>,
    /// How long a streaming turn keeps running after its SSE consumer disconnects, waiting for a
    /// reconnect. Accepts humantime strings like `"30s"`. Default `"30s"`.
    ///
    /// `"0s"` cancels the turn the moment the stream drops. That spends fewer provider tokens on
    /// abandoned work, and makes re-attach useful only for turns that already finished.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub stream_reattach_grace: Option<std::time::Duration>,
    /// Bearer tokens configured for this deployment. An empty list is refused at startup:
    /// `meka serve` exits rather than binding a port nothing can authenticate against.
    pub tokens: Option<Vec<ServeTokenConfig>>,
    /// Outbound webhook endpoints. Empty (the default) means meka never makes an outbound request.
    pub webhooks: Option<Vec<WebhookConfig>>,
}

/// One entry in `[[serve.webhooks]]`.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct WebhookConfig {
    /// Where to POST. `http://` is accepted for loopback development but logged as a warning:
    /// deliveries are signed, not encrypted, so anything on the path can read them.
    pub url: String,
    /// Shared secret for the `X-Meka-Signature` HMAC. Supports `${ENV_VAR}` substitution.
    /// Mutually exclusive with `secret_file`.
    pub secret: Option<String>,
    /// Path to a file whose contents (trimmed) are the secret. chmod 0600 recommended.
    pub secret_file: Option<std::path::PathBuf>,
    /// Which events to deliver. Required and non-empty: an endpoint subscribed to nothing is
    /// almost certainly a mistake, and silently never firing is the worst way to find out.
    pub events: Vec<String>,
    /// Per-attempt request timeout. Accepts humantime strings like `"10s"`. Default `"10s"`.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub timeout: Option<std::time::Duration>,
    /// Retries after the first attempt, with exponential backoff. Default 3.
    pub max_retries: Option<u32>,
}

// Manual `Debug` so a secret cannot reach a log through the *raw* struct either. Nothing prints
// this today, but the derived one would have been the single unredacted path in the webhook chain,
// and that is exactly the kind of thing a later `dbg!` finds.
impl std::fmt::Debug for WebhookConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebhookConfig")
            .field("url", &self.url)
            .field("secret", &self.secret.as_ref().map(|_| "[REDACTED]"))
            .field("secret_file", &self.secret_file)
            .field("events", &self.events)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

/// Scheme + host of a webhook URL, for log lines that run at the default verbosity.
///
/// The path is dropped because it frequently *is* the secret (Slack, Discord, and every other
/// "unguessable URL" webhook). Enough to tell two endpoints apart; not enough to call one.
///
/// Parsed rather than split on `"://"` and `'/'`. Splitting only ever removed the *path*, and a
/// URL has three other places to keep a secret: a query (`?token=…`, which survived intact because
/// a URL with no path has no `/` to split on), a fragment, and userinfo (`user:pass@host`, which
/// was returned verbatim as part of the host). Every one of those reached a `warn!` that runs at
/// default verbosity, on the line whose own comment calls it the one that gets pasted into an issue
/// tracker. Reconstructing from the parsed components keeps only what is named here, so a component
/// nobody thought of cannot ride along.
pub(crate) fn webhook_host(url: &str) -> String {
    let Ok(parsed) = url::Url::parse(url) else {
        return "<malformed>".to_string();
    };
    let Some(host) = parsed.host_str() else {
        return "<malformed>".to_string();
    };
    match parsed.port() {
        Some(port) => format!("{}://{}:{}", parsed.scheme(), host, port),
        None => format!("{}://{}", parsed.scheme(), host),
    }
}

/// Upper bound on `[[serve.webhooks]] max_retries`. See where it is applied for the reasoning.
const MAX_WEBHOOK_RETRIES: u32 = 10;

/// Validated webhook endpoint, with `${ENV}` substitution applied and `secret_file` loaded.
#[derive(Clone)]
pub struct ResolvedWebhook {
    pub url: String,
    pub secret: Option<String>,
    pub events: Vec<String>,
    pub timeout: std::time::Duration,
    pub max_retries: u32,
}

// Manual `Debug` so a configured secret never reaches a log line, matching `ResolvedServeToken`.
impl std::fmt::Debug for ResolvedWebhook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedWebhook")
            .field("url", &self.url)
            .field(
                "secret",
                &self
                    .secret
                    .as_ref()
                    .map(|secret| format_args!("[REDACTED len={}]", secret.len()).to_string()),
            )
            .field("events", &self.events)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

impl ResolvedWebhook {
    fn resolve(raw: WebhookConfig) -> Result<Self, String> {
        if raw.url.trim().is_empty() {
            return Err("[serve.webhooks] entry has an empty `url`".into());
        }
        let (url, _) = substitute_env(&raw.url)?;
        if !url.starts_with("https://") && !url.starts_with("http://") {
            return Err(format!(
                "[serve.webhooks] url '{}' must start with https:// or http://",
                url
            ));
        }
        let secret = match (raw.secret, raw.secret_file) {
            (Some(_), Some(_)) => {
                return Err(
                    "[serve.webhooks] entry has both `secret` and `secret_file`; pick one".into(),
                );
            }
            (Some(inline), None) => {
                let (resolved, _) = substitute_env(&inline)?;
                Some(resolved)
            }
            (None, Some(path)) => {
                let resolved = std::fs::read_to_string(&path)
                    .map(|contents| contents.trim().to_string())
                    .map_err(|error| {
                        format!("failed to read `secret_file` {}: {}", path.display(), error)
                    })?;
                warn_if_world_readable(&path);
                Some(resolved)
            }
            (None, None) => None,
        };
        if raw.events.is_empty() {
            return Err(format!(
                "[serve.webhooks] entry for '{}' subscribes to no events; set `events` to one \
                 or more of: {}",
                url,
                crate::server::webhook::WebhookEvent::ALL.join(", ")
            ));
        }
        for event in &raw.events {
            if !crate::server::webhook::WebhookEvent::ALL.contains(&event.as_str()) {
                // Rejected rather than warned, unlike an unknown token scope: a scope that grants
                // nothing still leaves the token working for whatever else it holds, whereas an
                // endpoint whose only subscription is a typo is silently never called at all.
                return Err(format!(
                    "[serve.webhooks] unknown event '{}'; expected one of: {}",
                    event,
                    crate::server::webhook::WebhookEvent::ALL.join(", ")
                ));
            }
        }
        if secret.as_deref().is_some_and(str::is_empty) {
            // An empty key still produces a well-formed HMAC, so the receiver would see a valid
            // `X-Meka-Signature` computed over nothing and the "unsigned deliveries" warning would
            // not fire. Rejected rather than treated as absent: reaching here means the operator
            // meant to sign, and an env var that resolved to empty is the likeliest cause.
            return Err(format!(
                "[serve.webhooks] entry for '{}' has an empty `secret`; omit the field to send \
                 unsigned deliveries, or supply a non-empty key",
                url
            ));
        }
        if raw.timeout == Some(std::time::Duration::ZERO) {
            // reqwest treats a zero timeout as "already elapsed", so every delivery fails its
            // whole retry schedule and the only symptom is a warn per event with no hint why.
            return Err(format!(
                "[serve.webhooks] entry for '{}' has `timeout = \"0s\"`, which fails every \
                 delivery before it is sent. Omit the field for the default (10s).",
                url
            ));
        }
        Ok(Self {
            url,
            secret,
            events: raw.events,
            timeout: raw.timeout.unwrap_or(std::time::Duration::from_secs(10)),
            // Clamped rather than trusted: each retry holds a task and sleeps, so a config typo of
            // `max_retries = 100000` would keep one alive for days against an endpoint that is
            // plainly not coming back. Ten attempts spans about two minutes of backoff.
            max_retries: raw.max_retries.unwrap_or(3).min(MAX_WEBHOOK_RETRIES),
        })
    }
}

/// One entry in `[serve.tokens]`. Tokens identify callers; scopes gate what they can do. See
/// the Auth section of the HTTP API docs for the full scope catalogue.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ServeTokenConfig {
    /// Inline token value. Supports `${ENV_VAR}` substitution at config-load time. Mutually
    /// exclusive with `token_file`.
    pub token: Option<String>,
    /// Path to a file whose contents (trimmed) are the token. chmod 0600 recommended.
    pub token_file: Option<std::path::PathBuf>,
    /// Free-form description, surfaced in startup logs. Operators use it to remember which
    /// caller a token belongs to (e.g. "telegram bridge", "ci debug").
    pub description: Option<String>,
    /// Scopes granted to this token. See the HTTP API docs for the catalogue.
    pub scopes: Vec<String>,
}

// Manual `Debug` for the same reason as [`WebhookConfig`]'s, and more urgently: `ServeConfig` and
// `ResolvedConfig` both derive `Debug` and `ResolvedConfig` owns the whole `[serve]` table, so the
// derived impl made every bearer token reachable from a single `{:?}` on the config. Redacting only
// `ResolvedServeToken` left that path open, because the raw form survives resolution.
impl std::fmt::Debug for ServeTokenConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServeTokenConfig")
            .field(
                "token",
                &self
                    .token
                    .as_ref()
                    .map(|token| format_args!("[REDACTED len={}]", token.len()).to_string()),
            )
            .field("token_file", &self.token_file)
            .field("description", &self.description)
            .field("scopes", &self.scopes)
            .finish()
    }
}

/// `[permissions]` table: choose which modes are reachable at runtime and which mode the session
/// starts in. See `docs/book/src/usage/permissions.md`.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct PermissionsConfig {
    pub default: Option<String>,
    pub enabled: Option<Vec<String>>,
}

/// Built-in tool filters, mirroring the per-server knobs on [`McpServerConfig`]. Applied at
/// registration time by [`crate::tools::ToolRegistry`].
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ToolsConfig {
    pub allowed_tools: Option<Vec<String>>,
    pub disabled_tools: Option<Vec<String>>,
    pub tool_permissions: Option<HashMap<String, String>>,
}

/// `[subagents]` table: capabilities a sub-agent may never hold.
///
/// Deliberately only the two deny lists. The distinction that decides what belongs here: a
/// *capability* is something config can genuinely withhold, because a tool the registry never
/// registered cannot be reached however the parent phrases the task. *Context* -- the memory store,
/// the instructions file -- cannot be withheld the same way, because a parent holding it can copy
/// it into the worker's prompt; a config key promising otherwise would read as a boundary while
/// being none. Context is therefore granted per call by `agent_spawn`, defaulting to nothing, and
/// has no key here.
///
/// The forgetting failure these lists exist for is real: MCP servers are inherited by default, so a
/// parent that never considers the question hands a worker every tool it has -- including one that
/// messages the user. `agent_spawn`'s `deny_servers` / `deny_tools` union with these, so a parent
/// can restrict further; there is deliberately no call-site allow-list, which would let a parent
/// widen what config denied.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct SubagentsConfig {
    /// Whole MCP servers a sub-agent cannot see. The coarse lever and the one that matters: naming
    /// a server removes its tools, its resources, and its prompts.
    pub disabled_servers: Option<Vec<String>>,
    /// Individual tools a sub-agent cannot see, by registry name, so built-ins and namespaced MCP
    /// tools (`mcp__<server>__<tool>`) share one namespace.
    pub disabled_tools: Option<Vec<String>>,
}

/// How much of the memory store an agent may reach.
///
/// Not a config key: the primary agent always holds [`MemoryAccess::Write`], and a sub-agent holds
/// whatever its `agent_spawn` call granted, defaulting to [`MemoryAccess::None`]. Ordered
/// `None < Read < Write` so `min` reads as "the more restrictive of two", which is how a grant is
/// clamped against what the granting agent itself holds.
///
/// `Write` is deliberately unreachable from `agent_spawn` (see [`Self::parse_grant`]). A worker
/// that inferred something from one narrow task should not be able to write it into the store every
/// future turn then reasons from; the parent, which has the whole conversation, is better placed to
/// decide what is worth remembering, and can record it after reading the worker's report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryAccess {
    /// No memory tools at all.
    None,
    /// `memory_read` and `memory_search`; no `memory_write` or `memory_delete`.
    Read,
    /// All four memory tools. The primary agent only.
    Write,
}

impl MemoryAccess {
    /// `None`, as a `#[serde(default)]` target for a persisted spec that lost the field: an absent
    /// grant must cost the worker its memory tools rather than hand it the store.
    pub fn none() -> Self {
        Self::None
    }

    /// Parse a `agent_spawn` grant. `"write"` is refused with its reason rather than clamped, so a
    /// parent asking for it learns that no sub-agent can have it instead of silently getting
    /// `read`.
    pub fn parse_grant(text: &str) -> std::result::Result<Self, String> {
        match text.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "read" => Ok(Self::Read),
            "write" => Err(
                "memory = \"write\" is not available to sub-agents. Use \"read\" and record \
                 anything worth keeping yourself, from the worker's report."
                    .to_string(),
            ),
            other => Err(format!(
                "memory = \"{}\" is not a valid access level. Use \"none\" or \"read\".",
                other
            )),
        }
    }
}

/// Whether an agent was handed the installation's instructions file.
///
/// Like [`MemoryAccess`], granted per `agent_spawn` call rather than configured, and defaulting to
/// [`InstructionAccess::None`]. Instructions describe the top-level agent -- its persona, how to
/// address the user, what to volunteer -- so a worker inherits none of it unless the parent decides
/// this particular task needs the project's rules.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InstructionAccess {
    /// A clean slate: the worker's system prompt carries no user instructions.
    #[default]
    None,
    /// The worker receives the instructions verbatim, as the parent has them.
    Inherit,
}

impl InstructionAccess {
    /// Parse a `agent_spawn` grant.
    pub fn parse_grant(text: &str) -> std::result::Result<Self, String> {
        match text.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "inherit" => Ok(Self::Inherit),
            other => Err(format!(
                "instructions = \"{}\" is not a valid value. Use \"none\" or \"inherit\".",
                other
            )),
        }
    }
}

/// [`SubagentsConfig`] with defaults filled in.
#[derive(Debug, Clone, Default)]
pub struct ResolvedSubagentsConfig {
    pub disabled_servers: Vec<String>,
    pub disabled_tools: Vec<String>,
}

impl ResolvedSubagentsConfig {
    fn resolve(raw: Option<SubagentsConfig>) -> Self {
        let raw = raw.unwrap_or_default();
        Self {
            disabled_servers: raw.disabled_servers.unwrap_or_default(),
            disabled_tools: raw.disabled_tools.unwrap_or_default(),
        }
    }
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ThinkingConfig {
    /// Budget for [`crate::provider::ThinkingMode::Budgeted`]; ignored under the other modes.
    pub budget_tokens: Option<u64>,
    pub show_content: Option<bool>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct McpConfig {
    /// Fallback permission for MCP tools when nothing more specific applies (no `tool_permissions`
    /// override, no server-level `permission`, no `readOnlyHint` from the server). If this is also
    /// unset the hardcoded fallback is `Unrestricted`, i.e. strict.
    pub default_permission: Option<String>,
    pub servers: Option<Vec<McpServerConfig>>,
    /// Default for each server's [`McpServerConfig::required`]. When true, every enabled server
    /// gates the turn; when false (the default) only servers that opt in with `required = true`
    /// do. A gated turn is rejected outright rather than sent to the model.
    ///
    /// Defaults to false because whether a missing server should stop the turn is a property of
    /// that server, not of the installation: one that is essential on a workstation may be
    /// irrelevant inside a container that lacks its binary.
    pub strict: Option<bool>,
    /// Per-turn cap on how long to wait for still-`Pending` MCP servers to settle before the
    /// readiness gate decides. Default: 3.
    pub grace_seconds: Option<u64>,
    /// Per-server wrap around connect + `initialize` + `list_tools`. A hung stdio spawn or slow
    /// HTTPS handshake can't stall the whole fleet past this bound. Default: 30.
    pub connect_timeout_seconds: Option<u64>,
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct McpServerConfig {
    pub name: String,
    pub transport: McpTransport,
    pub command: Option<String>,
    pub args: Option<Vec<String>>,
    pub env: Option<std::collections::HashMap<String, String>>,
    pub url: Option<String>,
    pub auth_token: Option<String>,
    pub headers: Option<std::collections::HashMap<String, String>>,
    /// Optional path to an executable that, when run, prints dynamic HTTP headers to stdout in
    /// `Name: Value\n` form. Merged over [`Self::headers`] (dynamic wins). Useful for SSO flows
    /// where bearer tokens rotate. The script is spawned with `MEKA_MCP_SERVER_NAME` and
    /// `MEKA_MCP_SERVER_URL` in its environment so one helper can drive multiple servers. Non-zero
    /// exit fails the connect.
    pub headers_helper: Option<String>,
    pub auth: Option<McpAuthConfig>,
    pub permission: Option<String>,
    /// Optional allow-list of raw tool names (the server-advertised form, not the
    /// `mcp__<server>__<tool>` namespaced form). When set and non-empty, only these tools from
    /// this server are registered.
    pub allowed_tools: Option<Vec<String>>,
    /// Optional block-list of raw tool names. Applied after [`Self::allowed_tools`]. Tools listed
    /// here are never registered.
    pub disabled_tools: Option<Vec<String>>,
    /// Raw tool names (server-advertised, not the `mcp__<server>__<tool>` namespaced form) that
    /// should ship eager-loaded instead of deferred. Saves a `load_tool` round-trip and keeps the
    /// schema in the cacheable tools-array prefix. Names that don't match an advertised tool
    /// surface as a `warn!` via [`crate::mcp::warn_on_stale_tool_config`].
    pub eager_load_tools: Option<Vec<String>>,
    /// Optional per-tool permission overrides keyed by raw tool name. Beats the server-level
    /// `permission` and the server's `readOnlyHint` annotation when resolving a tool's required
    /// permission at registration time.
    pub tool_permissions: Option<std::collections::HashMap<String, String>>,
    /// Whether this server's `readOnlyHint` annotation may classify a tool as `read`. Defaults to
    /// true, so a server that says a tool only reads is believed.
    ///
    /// The hint is asserted by the server, not verified by meka, and MCP tools execute in the
    /// server's own process with no sandbox. A server that advertises `readOnlyHint: true` for a
    /// tool that in fact writes therefore gets to write while meka sits at `read`. That is the
    /// reason this knob exists: setting it to `false` makes the hint advisory for display only, so
    /// the tool falls through to the strict `Unrestricted` fallback, and nothing from this server
    /// is reachable at `read` without an explicit [`Self::tool_permissions`] or
    /// [`Self::permission`] entry.
    ///
    /// A refused hint deliberately skips `[mcp].default_permission` on the way. That is a global
    /// convenience and this is a per-server audit decision, so the per-server one wins, the same
    /// direction the two overrides above already run. Falling through to it meant that with
    /// `default_permission = "read"` the knob changed nothing at all: the tool landed back on
    /// `Read` and dispatched unapproved at `--permission read`, which is precisely the outcome
    /// setting it to `false` was meant to prevent.
    ///
    /// Defaulting to true keeps existing configurations working and keeps read mode useful with
    /// well-behaved servers; the trade is that read mode's filesystem guarantee covers meka's
    /// built-in tools plus whichever MCP servers the user has chosen to trust.
    pub trust_read_only_hint: Option<bool>,
    /// When true, this server is skipped at startup: no process is spawned, no HTTP connect
    /// attempt is made. Lets users mute a flaky or in-development server without removing the
    /// entry.
    #[serde(default)]
    pub disabled: bool,
    /// Whether a turn may proceed while this server is unavailable. `None` inherits
    /// [`McpConfig::strict`] (false by default), so a server is optional unless it says otherwise.
    ///
    /// The other half of the availability pair with [`Self::disabled`]: `disabled` says "don't
    /// even try", `required` says "if trying failed, stop the turn". Resolved once in
    /// [`ResolvedConfig::from_cli`], so every later consumer reads a plain `bool`.
    pub required: Option<bool>,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum McpTransport {
    Stdio,
    Http,
}

#[derive(Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum McpAuthConfig {
    ClientCredentials {
        client_id: String,
        client_secret: String,
        scopes: Option<Vec<String>>,
        resource: Option<String>,
    },
    ClientCredentialsJwt {
        client_id: String,
        signing_key_path: String,
        signing_algorithm: Option<String>,
        scopes: Option<Vec<String>>,
        resource: Option<String>,
    },
    #[serde(rename = "oauth")]
    OAuth {
        client_id: Option<String>,
        client_secret: Option<String>,
        scopes: Option<Vec<String>>,
        redirect_port: Option<u16>,
    },
}

/// Redacts the value of every header and environment entry, and `auth_token`.
///
/// `headers` and `env` are the two places a user is most likely to put a bearer token, and a
/// `{:?}` on this struct (in a connect error, a `tracing::debug!`, a panic message) printed them.
/// Names are kept: knowing that `Authorization` was set is the diagnostic; knowing its value is the
/// leak.
fn redact_map(
    map: &Option<std::collections::HashMap<String, String>>,
) -> Option<std::collections::BTreeMap<&str, String>> {
    map.as_ref().map(|entries| {
        entries
            .iter()
            .map(|(name, value)| (name.as_str(), format!("[REDACTED len={}]", value.len())))
            .collect()
    })
}

impl std::fmt::Debug for McpServerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpServerConfig")
            .field("name", &self.name)
            .field("transport", &self.transport)
            .field("command", &self.command)
            .field("args", &self.args)
            .field("env", &redact_map(&self.env))
            .field("url", &self.url)
            .field(
                "auth_token",
                &self
                    .auth_token
                    .as_ref()
                    .map(|token| format_args!("[REDACTED len={}]", token.len()).to_string()),
            )
            .field("headers", &redact_map(&self.headers))
            .field("headers_helper", &self.headers_helper)
            .field("auth", &self.auth)
            .field("permission", &self.permission)
            .field("allowed_tools", &self.allowed_tools)
            .field("disabled_tools", &self.disabled_tools)
            .field("eager_load_tools", &self.eager_load_tools)
            .field("tool_permissions", &self.tool_permissions)
            .field("trust_read_only_hint", &self.trust_read_only_hint)
            .field("disabled", &self.disabled)
            .field("required", &self.required)
            .finish()
    }
}

/// Redacts `client_secret`, for the same reason as [`McpServerConfig`]'s impl. The flow's name and
/// its `client_id` stay, because those are what an auth failure is diagnosed from.
impl std::fmt::Debug for McpAuthConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ClientCredentials {
                client_id,
                client_secret,
                scopes,
                resource,
            } => f
                .debug_struct("ClientCredentials")
                .field("client_id", client_id)
                .field(
                    "client_secret",
                    &format_args!("[REDACTED len={}]", client_secret.len()),
                )
                .field("scopes", scopes)
                .field("resource", resource)
                .finish(),
            Self::ClientCredentialsJwt {
                client_id,
                signing_key_path,
                signing_algorithm,
                scopes,
                resource,
            } => f
                .debug_struct("ClientCredentialsJwt")
                .field("client_id", client_id)
                .field("signing_key_path", signing_key_path)
                .field("signing_algorithm", signing_algorithm)
                .field("scopes", scopes)
                .field("resource", resource)
                .finish(),
            Self::OAuth {
                client_id,
                client_secret,
                scopes,
                redirect_port,
            } => f
                .debug_struct("OAuth")
                .field("client_id", client_id)
                .field(
                    "client_secret",
                    &client_secret
                        .as_ref()
                        .map(|secret| format_args!("[REDACTED len={}]", secret.len()).to_string()),
                )
                .field("scopes", scopes)
                .field("redirect_port", redirect_port)
                .finish(),
        }
    }
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct DisplayConfig {
    pub newline_before_prompt: Option<bool>,
    pub newline_after_prompt: Option<bool>,
    pub show_session_id_on_create: Option<bool>,
    pub show_session_id_on_exit: Option<bool>,
    pub show_path_in_prompt: Option<bool>,
    pub show_context_in_prompt: Option<bool>,
    pub show_token_usage: Option<bool>,
    pub render_mode: Option<RenderMode>,
    /// How much of a tool call's input the `[tool X]` indicator shows: `off`, `summary` (default),
    /// or `full`.
    pub tool_params: Option<ToolParams>,
    /// Widest line meka composes from model output, in terminal columns. Unset follows the
    /// terminal, which is what keeps a line from ever wrapping; a set value is honoured exactly,
    /// so one wider than the terminal will wrap. Covers meka's own output only: assistant
    /// markdown reflows to the real terminal through `render_mode` either way.
    pub max_width: Option<usize>,
    /// Style applied to the REPL input buffer so submitted prompts stand out in scrollback. Parsed
    /// by [`parse_input_style`]. Accepts `bold`, `dim`, `none`, or a colour name (`cyan`,
    /// `yellow`, …).
    pub input_style: Option<String>,
    /// When set to `Some(N)` with `N > 0`, resuming a session reprints the last `N` turns (user
    /// prompts plus the agent's response, styled like the live REPL). Unset reprints only the last
    /// assistant message.
    pub resume_show_recent: Option<usize>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct WebConfig {
    pub user_agent: Option<String>,
    pub request_timeout_seconds: Option<u64>,
    pub connect_timeout_seconds: Option<u64>,
    pub read_timeout_seconds: Option<u64>,
    /// Max number of redirects reqwest will follow. `0` disables redirects entirely. Default:
    /// `10`.
    pub max_redirects: Option<u64>,
    /// Proxy URL. Accepts `http://…`, `https://…`, `socks5://…`, `socks5h://…`. The literal string
    /// `"none"` explicitly disables env-var auto-detection (overriding `HTTP_PROXY` etc.). Unset
    /// honours the env vars.
    pub proxy: Option<String>,
    /// Path to a PEM file containing one or more root CAs to trust on top of the system trust
    /// store. Used for corporate MITM proxies or self-signed internal services.
    pub ca_cert_file: Option<String>,
    /// Reject plain `http://` URLs: only `https://` allowed.
    pub https_only: Option<bool>,
    /// Minimum TLS version: `"1.0"`, `"1.1"`, `"1.2"`, or `"1.3"`. Anything else logs a warn and
    /// falls back to reqwest's default.
    pub min_tls_version: Option<String>,
    /// DANGER: disable TLS certificate validation entirely. Allows MITM; only use against trusted
    /// local dev servers.
    pub danger_accept_invalid_certs: Option<bool>,
    /// DANGER: accept certificates whose hostname doesn't match. Allows MITM; only use against
    /// trusted local dev servers.
    pub danger_accept_invalid_hostnames: Option<bool>,
}

/// Minimum TLS version accepted by the web-tools client. Normalised from `[web].min_tls_version` so
/// the rest of the crate doesn't pass free-form strings around.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinTlsVersion {
    V1_0,
    V1_1,
    V1_2,
    V1_3,
}

impl MinTlsVersion {
    /// Parse the config-file string form. Returns `None` on unknown input; callers are expected to
    /// log and fall through to the reqwest backend default.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim() {
            "1.0" => Some(Self::V1_0),
            "1.1" => Some(Self::V1_1),
            "1.2" => Some(Self::V1_2),
            "1.3" => Some(Self::V1_3),
            _ => None,
        }
    }
}

/// Fully-resolved web-tools HTTP client configuration. Carried on [`ResolvedConfig`] and consumed
/// by `crate::tools::web::build_web_client` at registry-build time.
#[derive(Debug, Clone)]
pub struct WebClientConfig {
    pub user_agent: String,
    pub request_timeout: std::time::Duration,
    pub connect_timeout: Option<std::time::Duration>,
    pub read_timeout: Option<std::time::Duration>,
    /// `0` means "no redirects" (`Policy::none`); any other value becomes `Policy::limited(n)`.
    pub max_redirects: usize,
    pub proxy: Option<String>,
    pub ca_cert_file: Option<std::path::PathBuf>,
    pub https_only: bool,
    pub min_tls_version: Option<MinTlsVersion>,
    pub danger_accept_invalid_certs: bool,
    pub danger_accept_invalid_hostnames: bool,
}

impl Default for WebClientConfig {
    fn default() -> Self {
        Self {
            user_agent: DEFAULT_WEB_USER_AGENT.to_string(),
            request_timeout: std::time::Duration::from_secs(30),
            connect_timeout: None,
            read_timeout: None,
            max_redirects: 10,
            proxy: None,
            ca_cert_file: None,
            https_only: false,
            min_tls_version: None,
            danger_accept_invalid_certs: false,
            danger_accept_invalid_hostnames: false,
        }
    }
}

impl WebClientConfig {
    /// Build a resolved config from the TOML section. Invalid `min_tls_version` strings log a warn
    /// and fall through to reqwest's default rather than aborting startup.
    pub fn from_file(file: &WebConfig) -> Self {
        let user_agent = file
            .user_agent
            .clone()
            .unwrap_or_else(|| DEFAULT_WEB_USER_AGENT.to_string());

        let min_tls_version =
            file.min_tls_version
                .as_deref()
                .and_then(|raw| match MinTlsVersion::parse(raw) {
                    Some(v) => Some(v),
                    None => {
                        tracing::warn!(
                            "ignoring unknown [web].min_tls_version '{}' \
                         (expected '1.0', '1.1', '1.2', or '1.3')",
                            raw
                        );
                        None
                    }
                });

        Self {
            user_agent,
            request_timeout: file
                .request_timeout_seconds
                .filter(|n| *n > 0)
                .map(std::time::Duration::from_secs)
                .unwrap_or_else(|| std::time::Duration::from_secs(30)),
            connect_timeout: file
                .connect_timeout_seconds
                .filter(|n| *n > 0)
                .map(std::time::Duration::from_secs),
            read_timeout: file
                .read_timeout_seconds
                .filter(|n| *n > 0)
                .map(std::time::Duration::from_secs),
            max_redirects: file
                .max_redirects
                .map(|n| usize::try_from(n).unwrap_or(usize::MAX))
                .unwrap_or(10),
            proxy: file.proxy.clone(),
            ca_cert_file: file.ca_cert_file.clone().map(std::path::PathBuf::from),
            https_only: file.https_only.unwrap_or(false),
            min_tls_version,
            danger_accept_invalid_certs: file.danger_accept_invalid_certs.unwrap_or(false),
            danger_accept_invalid_hostnames: file.danger_accept_invalid_hostnames.unwrap_or(false),
        }
    }
}

/// Default UA for the web tools when `[web].user_agent` is unset. Kept in sync with what real
/// Chrome emits so anti-bot filters don't single out meka by default.
pub const DEFAULT_WEB_USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36";

/// Max conversation messages kept in the per-turn API window by default.
const DEFAULT_CONTEXT_MESSAGES: usize = 200;
/// Default extended-thinking token budget.
const DEFAULT_THINKING_BUDGET_TOKENS: u64 = 16_000;
/// Default maximum sub-agent recursion depth (root spawns down to grandchild).
const DEFAULT_SUBAGENT_MAX_DEPTH: usize = 3;

/// Narrowest `[display].max_width` that still leaves room for anything.
///
/// Every budget derived from it subtracts fixed chrome first (`[tool ` and its brackets, the
/// `Thinking... ` prefix, a block's indent), so below roughly this the subtraction leaves nothing
/// and output degrades to punctuation. Clamped rather than rejected: a too-narrow width is a
/// preference stated badly, not a broken config, and refusing to start over it would be
/// disproportionate.
const MIN_CONFIGURED_WIDTH: usize = 40;

/// Widest `[display].max_width` worth honouring.
///
/// Not a taste judgement about long lines. Fitting text to a column budget re-measures a growing
/// prefix, so the work to compose one line grows with the square of the width: a stray `max_width =
/// 100000` turns each elided line into billions of width computations and hangs the renderer. No
/// terminal is this wide, so nothing legitimate is refused.
const MAX_CONFIGURED_WIDTH: usize = 1000;

fn clamp_max_width(configured: usize) -> usize {
    if configured < MIN_CONFIGURED_WIDTH {
        tracing::warn!(
            "[display].max_width = {} is below the {}-column minimum; using {}",
            configured,
            MIN_CONFIGURED_WIDTH,
            MIN_CONFIGURED_WIDTH
        );
        return MIN_CONFIGURED_WIDTH;
    }
    if configured > MAX_CONFIGURED_WIDTH {
        tracing::warn!(
            "[display].max_width = {} is above the {}-column maximum; using {}",
            configured,
            MAX_CONFIGURED_WIDTH,
            MAX_CONFIGURED_WIDTH
        );
        return MAX_CONFIGURED_WIDTH;
    }
    configured
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ShellConfig {
    pub sandbox: Option<bool>,
    /// Linux-only choice between `"landlock"` and `"bubblewrap"`. When omitted, the resolver
    /// auto-picks bubblewrap if available and falls back to landlock with a one-shot warning (see
    /// `src/sandbox.rs` and `Warn 2` in `warn_if_sandbox_issues`).
    pub sandbox_backend: Option<SandboxBackend>,
}

/// Linux sandbox backend. Silently ignored on macOS and Windows
/// (sandbox-exec and Low-integrity respectively are the only options
/// on those platforms). The absence of a default impl is intentional:
/// an unset value is meaningful and triggers auto-resolution in
/// [`ResolvedConfig::from_cli`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SandboxBackend {
    Landlock,
    Bubblewrap,
}

impl SandboxBackend {
    /// Brand-cased name for user-facing prose (logs, errors). Also the `Display` impl's output.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Landlock => "Landlock",
            Self::Bubblewrap => "Bubblewrap",
        }
    }
}

impl std::fmt::Display for SandboxBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.display_name())
    }
}

impl std::str::FromStr for SandboxBackend {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "landlock" => Ok(SandboxBackend::Landlock),
            "bubblewrap" => Ok(SandboxBackend::Bubblewrap),
            other => Err(format!(
                "unknown sandbox backend '{other}' (expected 'landlock' or 'bubblewrap')"
            )),
        }
    }
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct SessionConfig {
    pub context_messages: Option<usize>,
    /// Delete sessions whose `updated_at` is older than this, at agent startup. Unset means keep
    /// everything: conversation history is not reproducible, so meka never discards it unasked.
    pub retention_days: Option<u64>,
    pub auto_compact: Option<bool>,
    /// Run a checkpoint turn before each compaction, letting the agent save what must survive and
    /// write the summary itself. Default `true`. Costs one extra model call per compaction; off
    /// falls back to the standalone summarizer, which has no tools and none of the agent's
    /// identity.
    pub compact_checkpoint: Option<bool>,
    pub context_window: Option<u64>,
    pub subagent_max_depth: Option<usize>,
}

/// One named provider profile from `[providers.<name>]`. Holds only non-secret settings; the
/// credential (API key or OAuth bundle) is stored in the DB keyed by the profile name and is
/// acquired via `meka provider add` / `login`.
#[derive(Debug, Deserialize, Default, Clone)]
#[serde(deny_unknown_fields)]
pub struct ProviderProfile {
    /// Backend kind: one of [`crate::provider::SUPPORTED_PROVIDERS`].
    #[serde(rename = "type")]
    pub backend: String,
    pub model: Option<String>,
    pub base_url: Option<String>,
    /// The model's context window (total tokens it can hold), used for the context gauge and
    /// auto-compaction. Falls back to `[session].context_window`, then to
    /// [`crate::provider::DEFAULT_CONTEXT_WINDOW`]. meka never infers this from the model name or
    /// asks the provider for it, so this is where a model smaller than the default gets stated.
    pub context_window: Option<u64>,
    /// Whether this profile's model accepts image input. Defaults to `true`; set `false` to stop
    /// the ACP frontend from advertising / accepting images for a text-only model.
    pub vision: Option<bool>,
    /// Override the per-request output (completion) token cap. When unset, each backend keeps its
    /// built-in default. On Claude the value must exceed the thinking budget.
    pub max_output_tokens: Option<u64>,
    pub oauth_token_url: Option<String>,
    pub device_id: Option<String>,
    /// OAuth client ID override (advanced; `claude-subscription` / `chatgpt-subscription`).
    pub client_id: Option<String>,
    /// The reasoning-effort knob for every backend: Claude's `output_config.effort` and OpenAI's
    /// `reasoning.effort`. Passed through verbatim (trimmed and lowercased); when unset the field
    /// is omitted entirely, which is how meka asks for the provider's own default. meka picks no
    /// tier of its own - see [`crate::provider::resolve_effort_level`].
    pub effort: Option<String>,
    /// Claude-only: how this profile encodes extended thinking - `adaptive`, `budgeted`, or `off`.
    /// Defaults to `adaptive`. The right value depends on the model *and* on what the endpoint
    /// implements, which meka can't infer, so it is stated rather than guessed; a profile that
    /// later changes `model` is the user's to keep correct.
    pub thinking: Option<crate::provider::ThinkingMode>,
    /// `claude-subscription` only: when true, meka sends the `redact-thinking-2026-02-12` beta
    /// header so the server returns `redacted_thinking` blocks instead of full thinking
    /// summaries (saves bandwidth, but the redacted payloads can't be replayed back to the
    /// server in multi-turn conversations). Defaults to true.
    pub redact_thinking: Option<bool>,
}

/// Which session a run should pick up, resolved from the mutually exclusive `--continue` and
/// `--resume` flags.
///
/// A real type rather than the `Option<String>` with a `"last"` sentinel this replaces: the
/// sentinel meant `-c` had to take an optional value, which made it the only flag on the root
/// command that could swallow the following argument. `meka -c "fix the bug"` read the prompt as a
/// session prefix and failed with either a confusing lookup error or, under `--oneshot`, a claim
/// that no prompt was given. Splitting the two intents into a boolean and a value-taking flag puts
/// both in line with every other root flag and removes the ambiguity at the parser rather than
/// guessing at it afterwards.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionResume {
    /// `--continue`: the most recently updated session.
    Last,
    /// `--resume <SESSION>`: a specific UUID, or a leading prefix of one.
    Id(String),
}

impl SessionResume {
    /// `None` when neither flag was given. Clap enforces that they are mutually exclusive, so the
    /// order of these checks is not load-bearing.
    fn from_cli(cli: &crate::cli::Cli) -> Option<Self> {
        if let Some(id) = &cli.resume {
            return Some(Self::Id(id.clone()));
        }
        cli.continue_last.then_some(Self::Last)
    }
}

/// Merged + validated runtime view of [`ConfigFile`], CLI flags, and env vars. This is what the
/// rest of the binary reads; `ConfigFile` is for deserialization only. Resolution lives in
/// `resolve_config` (Linux) and the non-Linux variant below it.
#[derive(Debug)]
pub struct ResolvedConfig {
    /// Backend type of the active profile (one of [`crate::provider::SUPPORTED_PROVIDERS`]);
    /// `None` when no profile could be selected.
    pub provider_name: Option<String>,
    /// Name of the active profile, used as the DB key for its credential.
    pub active_profile: Option<String>,
    /// Set when profile selection failed (no profiles, ambiguous, or an unknown name); surfaced by
    /// [`Self::validate`] with guidance to run `meka provider add` / `use`.
    pub provider_error: Option<String>,
    /// `config.toml` failed to read or parse. Raised by [`Self::validate`] before anything else,
    /// since every other field is a default that silently ignores what the user wrote.
    pub config_error: Option<String>,
    /// The instructions the user explicitly named could not be loaded, or the two environment
    /// variables that carry them contradict each other. Deferred like the two above so `from_cli`
    /// stays infallible, and raised rather than shrugged off because running without guidance the
    /// user believes they supplied is worse than not starting.
    pub instructions_error: Option<String>,
    pub model: Option<String>,
    pub base_url: Option<String>,
    pub client_id: Option<String>,
    pub oauth_token_url: Option<String>,
    pub permission: Permission,
    pub enabled_permissions: EnabledPermissions,
    pub streaming: bool,
    pub session_resume: Option<SessionResume>,
    pub prompt: Option<String>,
    pub oneshot: bool,
    pub newline_before_prompt: bool,
    pub newline_after_prompt: bool,
    pub show_session_id_on_create: bool,
    pub show_session_id_on_exit: bool,
    pub show_path_in_prompt: bool,
    pub show_context_in_prompt: bool,
    pub show_token_usage: bool,
    pub resume_show_recent: Option<usize>,
    pub web_client: WebClientConfig,
    pub sandbox: bool,
    /// Extra workspace roots from `--writable-root`, joined with the cwd (and, under ACP, the
    /// client's folders) to form the write boundary at `workspace` permission.
    ///
    /// CLI-only by design. Which folders a run may write is a per-run scope like the working
    /// directory, not a preference worth persisting, so there is deliberately no `config.toml`
    /// key and no env tier for it.
    pub writable_roots: Vec<std::path::PathBuf>,
    /// Resolved Linux sandbox backend. Auto-picked at startup when `[shell].sandbox_backend` was
    /// not set: prefers bubblewrap, falls back to landlock. Silently `Landlock` on macOS / Windows
    /// (those platforms have their own backend and ignore the field).
    pub sandbox_backend: SandboxBackend,
    /// True when [`Self::sandbox_backend`] was auto-resolved (i.e. the user did not pin a value in
    /// `[shell].sandbox_backend`). Used to gate the "stronger sandbox available; install bwrap"
    /// startup warn. We don't want to nag users who explicitly chose landlock.
    pub sandbox_auto_resolved: bool,
    /// Cached probe of the resolved backend. Consulted by `warn_if_sandbox_issues` and by the lazy
    /// hard-error path in `src/tools/shell.rs` when read-mode `execute_command` is invoked.
    pub backend_probe: crate::sandbox::BackendProbe,
    pub render_mode: RenderMode,
    pub tool_params: ToolParams,
    /// Resolved `[display].max_width`. `None` - the default - follows the terminal.
    pub max_width: Option<usize>,
    pub context_messages: Option<usize>,
    /// Resolved `[session].retention_days`. `None` - the default - disables startup cleanup.
    pub retention_days: Option<u64>,
    pub thinking: crate::provider::ThinkingMode,
    pub thinking_budget_tokens: u64,
    pub thinking_show_content: bool,
    /// Stable per-device identifier for `claude-subscription`'s `metadata.user_id`. Empty string
    /// for non-`claude-subscription` providers (the value is ignored downstream).
    pub device_id: String,
    /// The user's explicit reasoning-effort override for every backend (Claude
    /// `output_config.effort`, OpenAI `reasoning.effort`). Passed through verbatim; `None` leaves
    /// the field off the request, so the provider applies its own default.
    pub effort: Option<String>,
    /// `claude-subscription`: when true, request `redacted_thinking` blocks via
    /// `redact-thinking-2026-02-12` beta. Default true.
    pub redact_thinking: bool,
    pub auto_compact: bool,
    pub compact_checkpoint: bool,
    pub context_window: Option<u64>,
    /// Maximum sub-agent recursion depth. `1` lets the root agent spawn but denies its sub-agents
    /// the same; `0` disables `agent_spawn` entirely. Seeds the root `AgentSpawnTool`'s depth
    /// budget in `main.rs`.
    pub subagent_max_depth: usize,
    /// Whether the active profile's model accepts image input (the ACP `image` prompt capability).
    /// Resolved from `[providers.<name>].vision`, defaulting to `true`.
    pub vision: bool,
    /// Per-request output (completion) token cap from `[providers.<name>].max_output_tokens`.
    /// `None` leaves each backend's built-in default in place.
    pub max_output_tokens: Option<u64>,
    pub mcp_servers: Vec<McpServerConfig>,
    /// Parsed [`Permission`] from `[mcp].default_permission`, carried so per-turn tool-permission
    /// resolution in `src/mcp.rs` doesn't have to re-read the config file. `None` means "no
    /// `[mcp]` default configured"; resolution falls through to the hardcoded `Unrestricted`.
    pub mcp_default_permission: Option<Permission>,
    pub user_instructions: Option<String>,
    /// Where [`Self::user_instructions`] came from, rendered (`--instructions`,
    /// `MEKA_INSTRUCTIONS`, or the contributing file paths). Kept alongside the text so an
    /// operator asking "why is it behaving like that" over the API gets the provenance, not just
    /// the prose.
    pub user_instructions_source: Option<String>,
    /// Non-secret projection of the configured `[providers.<name>]` profiles: `(name, backend,
    /// model)`.
    ///
    /// A projection rather than the profiles themselves, so a field added to [`ProviderProfile`]
    /// later cannot start appearing on `GET /v1/providers` by accident. Credentials were never in
    /// here to begin with; they live in the database keyed by profile name.
    pub provider_profiles: Vec<(String, String, Option<String>)>,
    /// Whether the `skill_read` / `skill_search` tools are registered and the `[Skills]` index
    /// rendered. Defaults to `true`; see [`SkillsConfig`].
    pub skills_enabled: bool,
    /// Whether `skill_write` / `skill_delete` are additionally registered. Defaults to `false`;
    /// see [`SkillsConfig::agent_managed`].
    pub skills_agent_managed: bool,
    /// Read-only skill roots scanned in addition to meka's own; see
    /// [`SkillsConfig::extra_paths`]. Already tilde-expanded, and empty unless configured.
    ///
    /// Read through [`Self::skill_roots`] rather than directly, so the "native first, then these"
    /// order has one owner.
    pub skills_extra_paths: Vec<std::path::PathBuf>,
    /// Whether the `memory_*` tools are registered and the `[Memory]` index rendered. Defaults to
    /// `true`; see [`MemoryConfig`].
    pub memory_enabled: bool,
    /// `[schedule]` with defaults filled. Resolved here rather than at the call site because both
    /// the REPL and `meka serve` start a scheduler and would otherwise each re-derive it.
    pub schedule: ResolvedScheduleConfig,
    /// `[background]` with defaults filled. Off unless the installation asks for it; see
    /// [`BackgroundConfig`].
    pub background: ResolvedBackgroundConfig,
    /// `[subagents]` with defaults filled. Governs what a spawned worker does *not* inherit; see
    /// [`SubagentsConfig`].
    pub subagents: ResolvedSubagentsConfig,
    pub builtin_allowed_tools: Option<Vec<String>>,
    pub builtin_disabled_tools: Vec<String>,
    pub builtin_tool_permissions: HashMap<String, Permission>,
    pub input_style: nu_ansi_term::Style,
    /// First-turn await cap for still-connecting MCP servers.
    pub mcp_grace: std::time::Duration,
    /// Per-server connect+initialize timeout.
    pub mcp_connect_timeout: std::time::Duration,
    /// Raw `[serve]` section. Resolved (defaults filled, env vars substituted, token files
    /// read) at `meka serve` startup by [`ResolvedServeConfig::resolve`], not here, so
    /// `from_cli` can stay infallible.
    pub serve: Option<ServeConfig>,
    /// CLI-flag override for `[serve].bind` (`meka serve --bind ...`). Wins over the config
    /// file when set.
    pub serve_bind_override: Option<String>,
}

/// Validated, defaults-filled view of [`ServeConfig`]. Constructed at config-load time by
/// [`ResolvedServeConfig::resolve`].
// Individual fields mirror [`ServeConfig`]; see its per-field documentation.
#[derive(Debug, Clone)]
pub struct ResolvedServeConfig {
    pub bind: String,
    pub idle_timeout: std::time::Duration,
    pub gc_scan_interval: std::time::Duration,
    pub delete_on_idle: bool,
    pub shutdown_drain_timeout: std::time::Duration,
    pub max_concurrent_turns: Option<usize>,
    pub max_body_bytes: usize,
    /// Whether `/v1/docs` and `/v1/openapi.json` are served. Off unless asked for.
    pub docs: bool,
    /// How many SSE events per turn to retain for `Last-Event-ID` replay.
    pub stream_replay_events: usize,
    /// How long a streaming turn keeps running after its SSE consumer disconnects, waiting for a
    /// reconnect. Zero cancels the turn the moment the stream drops.
    pub stream_reattach_grace: std::time::Duration,
    pub tokens: Vec<ResolvedServeToken>,
    pub webhooks: Vec<ResolvedWebhook>,
}

/// Validated token entry. The `token` field carries the final secret value with `${ENV}`
/// substitution applied and `token_file` contents loaded; call sites compare against this
/// directly via constant-time equality.
#[derive(Clone)]
pub struct ResolvedServeToken {
    pub token: String,
    pub description: Option<String>,
    pub scopes: Vec<String>,
    /// Where the token value originated, used at startup to nudge operators away from inline
    /// plaintext. See [`TokenSource`]. Not security-sensitive itself.
    pub source: TokenSource,
}

impl std::fmt::Debug for ResolvedServeToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedServeToken")
            .field(
                "token",
                &format_args!("[REDACTED len={}]", self.token.len()),
            )
            .field("description", &self.description)
            .field("scopes", &self.scopes)
            .field("source", &self.source)
            .finish()
    }
}

/// Provenance of a configured token. Determines whether `meka serve` emits a `warn!` at startup
/// (inline plaintext) or file-backed.
#[derive(Debug, Clone)]
pub enum TokenSource {
    /// Literal value in `token = "..."` with no `${ENV}` markers, discouraged outside
    /// development.
    Inline,
    /// `token = "${ENV_VAR}"` substituted at config-load time.
    EnvVar,
    /// `token_file = "/path/to/token"` read at config-load time.
    File,
}

impl ResolvedServeConfig {
    /// Resolve a [`ServeConfig`] (or its absence) into a [`ResolvedServeConfig`] with all
    /// defaults filled and tokens read from disk / env. Errors are returned for caller
    /// configuration problems (both `token` and `token_file` set, env var unset, file missing).
    pub fn resolve(raw: Option<ServeConfig>) -> Result<Self, String> {
        let raw = raw.unwrap_or_default();
        // Reject zero-value `max_*` knobs at config time:
        //   - `max_concurrent_turns = 0` would 429 every turn
        //   - `max_body_bytes = 0` would 413 every request
        // Operators wanting "unbounded" omit the field instead.
        if let Some(0) = raw.max_concurrent_turns {
            return Err(
                "[serve] `max_concurrent_turns = 0` would block every turn. Omit the field \
                 to disable the cap, or set a positive integer."
                    .into(),
            );
        }
        if let Some(0) = raw.max_body_bytes {
            return Err(
                "[serve] `max_body_bytes = 0` would reject every request body. Omit the \
                 field to use the default (10 MiB), or set a positive integer."
                    .into(),
            );
        }
        if raw.gc_scan_interval == Some(std::time::Duration::ZERO) {
            // `tokio::time::interval(ZERO)` panics, and the GC handle is only ever aborted, never
            // joined, so the panic is swallowed: eviction silently never runs and every session
            // lock is held until the process exits. `[schedule].poll_interval` guards the same
            // shape for the same reason.
            return Err(
                "[serve] `gc_scan_interval = \"0s\"` would stop the session GC from ever running. \
                 Omit the field for the default (5m), or set a positive duration."
                    .into(),
            );
        }
        let tokens = raw
            .tokens
            .unwrap_or_default()
            .into_iter()
            .map(ResolvedServeToken::resolve)
            .collect::<Result<Vec<_>, _>>()?;
        let webhooks = raw
            .webhooks
            .unwrap_or_default()
            .into_iter()
            .map(ResolvedWebhook::resolve)
            .collect::<Result<Vec<_>, _>>()?;
        for webhook in &webhooks {
            if webhook.secret.is_none() {
                tracing::warn!(
                    // Host only. `warn` is the default level, so this line lands in logs an
                    // operator may well paste elsewhere, and for a Slack- or Discord-style
                    // endpoint the URL path *is* the credential. The full URL is at `info`.
                    endpoint = %webhook_host(&webhook.url),
                    "webhook has no `secret`; deliveries are unsigned and a receiver cannot \
                     tell them apart from anything else that can reach it",
                );
            }
            if webhook.url.starts_with("http://") {
                tracing::warn!(
                    endpoint = %webhook_host(&webhook.url),
                    "webhook uses plaintext http; deliveries are signed but not encrypted",
                );
            }
        }
        Ok(Self {
            bind: raw.bind.unwrap_or_else(|| "127.0.0.1:8080".to_string()),
            idle_timeout: raw
                .idle_timeout
                .unwrap_or(std::time::Duration::from_secs(24 * 60 * 60)),
            gc_scan_interval: raw
                .gc_scan_interval
                .unwrap_or(std::time::Duration::from_secs(5 * 60)),
            delete_on_idle: raw.delete_on_idle.unwrap_or(false),
            shutdown_drain_timeout: raw
                .shutdown_drain_timeout
                .unwrap_or(std::time::Duration::from_secs(30)),
            max_concurrent_turns: raw.max_concurrent_turns,
            max_body_bytes: raw.max_body_bytes.unwrap_or(10 * 1024 * 1024),
            docs: raw.docs.unwrap_or(false),
            // Matches the broadcast channel's capacity: retaining more than the live channel can
            // buffer would let a client replay events a *connected* consumer would have been
            // dropped for missing.
            stream_replay_events: raw.stream_replay_events.unwrap_or(256),
            stream_reattach_grace: raw
                .stream_reattach_grace
                .unwrap_or(std::time::Duration::from_secs(30)),
            tokens,
            webhooks,
        })
    }
}

impl ResolvedServeToken {
    fn resolve(raw: ServeTokenConfig) -> Result<Self, String> {
        let (token, source) = match (raw.token, raw.token_file) {
            (Some(_), Some(_)) => {
                return Err(
                    "[serve.tokens] entry has both `token` and `token_file`; pick one".into(),
                );
            }
            (Some(inline), None) => {
                let (resolved, substituted) = substitute_env(&inline)?;
                let source = if substituted {
                    TokenSource::EnvVar
                } else {
                    TokenSource::Inline
                };
                (resolved, source)
            }
            (None, Some(path)) => {
                let resolved = std::fs::read_to_string(&path)
                    .map(|s| s.trim().to_string())
                    .map_err(|error| {
                        format!("failed to read `token_file` {}: {}", path.display(), error)
                    })?;
                warn_if_world_readable(&path);
                (resolved, TokenSource::File)
            }
            (None, None) => {
                return Err("[serve.tokens] entry must set either `token` or `token_file`".into());
            }
        };
        if token.is_empty() {
            return Err("[serve.tokens] resolved token is empty".into());
        }
        crate::server::scope::warn_unknown(&raw.scopes, raw.description.as_deref());
        Ok(Self {
            token,
            description: raw.description,
            scopes: raw.scopes,
            source,
        })
    }
}

/// Log a warning if `path` is readable by group or others on Unix. No-op on non-Unix.
/// Matches the advisory guidance ("chmod 0600 recommended") without
/// refusing to start; the file has already been read successfully, so we just nudge.
fn warn_if_world_readable(path: &std::path::Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(metadata) = std::fs::metadata(path) {
            let mode = metadata.permissions().mode() & 0o777;
            if mode & 0o077 != 0 {
                tracing::warn!(
                    "[serve.tokens] token_file '{}' has permissions {:04o}; recommend chmod 0600 \
                     to prevent other users from reading the bearer token",
                    path.display(),
                    mode,
                );
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
}

/// Deserialize an optional humantime duration string (e.g. `"24h"`, `"5m"`, `"30s"`).
/// `serde(deserialize_with)` on `Option<Duration>` requires this wrapper because
/// `humantime_serde` only provides a `deserialize` for `Duration`, not `Option<Duration>`.
fn deserialize_optional_duration<'de, D>(
    deserializer: D,
) -> Result<Option<std::time::Duration>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt: Option<String> = Option::deserialize(deserializer)?;
    match opt {
        None => Ok(None),
        Some(s) => humantime_serde::re::humantime::parse_duration(&s)
            .map(Some)
            .map_err(serde::de::Error::custom),
    }
}

/// Replace `${VAR}` occurrences with the corresponding environment variable. `$$` is an escape
/// for a literal `$`. Unknown variables return an error rather than expanding to empty; silent
/// expansion to empty would produce a runtime auth-bypass-shaped configuration.
///
/// Returns the substituted string plus a boolean indicating whether at least one `${VAR}`
/// expansion happened; callers use this to classify token provenance (`TokenSource::EnvVar`
/// vs `TokenSource::Inline`) for the startup warning.
fn substitute_env(input: &str) -> Result<(String, bool), String> {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    let mut substituted = false;
    while let Some(c) = chars.next() {
        if c != '$' {
            out.push(c);
            continue;
        }
        match chars.peek() {
            Some('$') => {
                chars.next();
                out.push('$');
            }
            Some('{') => {
                chars.next();
                let mut name = String::new();
                let mut closed = false;
                for inner in chars.by_ref() {
                    if inner == '}' {
                        closed = true;
                        break;
                    }
                    name.push(inner);
                }
                if !closed {
                    return Err("unclosed `${` in a substituted config value".into());
                }
                let value = std::env::var(&name)
                    .map_err(|_| format!("env var `{}` referenced in config is unset", name))?;
                out.push_str(&value);
                substituted = true;
            }
            _ => out.push('$'),
        }
    }
    Ok((out, substituted))
}

/// Default input style: bold, white-ish foreground, slate-blue background. Uses truecolor RGB (not
/// palette indices or named colours) so the visual is consistent across terminals that remap the
/// standard 16 colours to match their theme.
pub fn default_input_style() -> nu_ansi_term::Style {
    use nu_ansi_term::{Color, Style};
    Style::new()
        .bold()
        .fg(Color::Rgb(240, 240, 240))
        .on(Color::Rgb(55, 75, 110))
}

/// Parse a `[display].input_style` value. `"default"` (or unset) yields [`default_input_style`];
/// `"none"` yields no styling; simple keywords pick a single colour or attribute. Unknown keywords
/// warn and fall back to the default so a typo doesn't lose the session to a panic.
pub fn parse_input_style(raw: &str) -> nu_ansi_term::Style {
    use nu_ansi_term::{Color, Style};
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "default" => default_input_style(),
        "none" => Style::default(),
        "reverse" => Style::new().reverse(),
        "bold" => Style::new().bold(),
        "dim" => Style::new().dimmed(),
        "italic" => Style::new().italic(),
        "underline" => Style::new().underline(),
        "black" => Style::new().fg(Color::Black),
        "red" => Style::new().fg(Color::Red),
        "green" => Style::new().fg(Color::Green),
        "yellow" => Style::new().fg(Color::Yellow),
        "blue" => Style::new().fg(Color::Blue),
        "magenta" | "purple" => Style::new().fg(Color::Magenta),
        "cyan" => Style::new().fg(Color::Cyan),
        "white" => Style::new().fg(Color::White),
        other => {
            tracing::warn!(
                "ignoring unknown [display].input_style '{}' (expected \
                 default, none, reverse, bold, dim, italic, underline, or a color name)",
                other
            );
            default_input_style()
        }
    }
}

/// Expand a leading `~` or `~/` in a user-supplied path to the home directory.
///
/// Returns `None` only when a tilde needs the home directory but it cannot be determined, which the
/// caller reports rather than silently treating `~` as a relative directory of that name.
///
/// Shared by the REPL's `/cd` (and its path completer) and by `[skills] extra_paths`, so a tilde
/// means the same thing wherever a user types a path. Deliberately does *not* expand `~user`: that
/// needs a password-database lookup, and no config surface here has ever accepted the form.
pub fn expand_user_path(target: &str) -> Option<PathBuf> {
    if target.is_empty() || target == "~" {
        dirs::home_dir()
    } else if let Some(rest) = target
        .strip_prefix('~')
        .and_then(|rest| rest.strip_prefix(std::path::is_separator))
    {
        // `is_separator`, not a literal `"~/"`. On Windows both `/` and `\` separate, and `~\` is
        // the spelling a user reaches for there because that is what PowerShell's own tilde
        // expansion produces. Matching only `~/` left `~\projects` unexpanded, so it became a
        // literal relative directory of that name and `/cd` reported the tilde back at the user as
        // though it were a folder. `[skills] extra_paths` took the same spelling and resolved to a
        // root that never existed.
        dirs::home_dir().map(|home| home.join(rest))
    } else {
        Some(PathBuf::from(target))
    }
}

/// Resolve `[skills] extra_paths` into the read-only roots discovery will scan.
///
/// A *function* rather than a chain inside `from_cli`, because each of the four things it drops has
/// its own reason and each needs saying out loud:
///
/// - An empty entry, because [`expand_user_path`] maps `""` to the home directory. That is right
///   for `/cd` (type `cd` and you go home) and catastrophic here: `$HOME` would become a skills
///   root and warn once per subdirectory on every rediscovery.
/// - An entry whose `~` cannot be expanded, rather than treating the tilde as a directory name.
/// - A repeat of an earlier entry, which discovery would otherwise walk twice and then report every
///   skill in it as shadowed *by itself* -- a warning naming one path twice, which an operator can
///   neither act on nor dismiss.
/// - meka's own root, for the same reason. It is always scanned first; naming it again adds a
///   second pass and the same self-shadowing report.
///
/// Compared as written after expansion, not canonicalised: resolving symlinks would touch the
/// filesystem, and the promise that a configured-but-absent root leaves no trace is worth more than
/// catching two spellings of one directory. Discovery's own first-wins rule handles that case.
fn resolve_skills_extra_paths(raw: &[String], native: Option<&Path>) -> Vec<PathBuf> {
    let mut resolved: Vec<PathBuf> = Vec::new();
    for entry in raw {
        if entry.trim().is_empty() {
            tracing::warn!("[skills] extra_paths contains an empty entry; ignoring it");
            continue;
        }
        let Some(path) = expand_user_path(entry) else {
            tracing::warn!(
                "[skills] extra_paths entry '{}' needs a home directory that could not be \
                 determined; skipping it",
                entry
            );
            continue;
        };
        if Some(path.as_path()) == native {
            tracing::warn!(
                "[skills] extra_paths entry '{}' is meka's own skills directory, which is always \
                 scanned first; ignoring it",
                entry
            );
            continue;
        }
        if resolved.contains(&path) {
            tracing::warn!(
                "[skills] extra_paths lists '{}' more than once; ignoring the repeat",
                entry
            );
            continue;
        }
        resolved.push(path);
    }
    resolved
}

/// Returns the meka config directory (the directory that contains `config.toml` and `skills/`).
/// Honours the `MEKA_CONFIG_DIR` env var, used by tests for per-run isolation and by power users
/// who want a non-standard location, before falling back to the platform-native
/// `dirs::config_dir().join("meka")`. The env-var route is the only reliable way to isolate state
/// on macOS and Windows, where `dirs::config_dir()` doesn't honour `XDG_CONFIG_HOME`.
pub fn meka_config_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MEKA_CONFIG_DIR") {
        let path = PathBuf::from(dir);
        // An empty value reads as unset rather than as the current directory. `MEKA_CONFIG_DIR=`
        // in a shell profile, or an exported-but-unset variable in a systemd unit, made meka load
        // `./config.toml` from whatever directory it happened to start in -- and then spawn that
        // file's MCP servers. A relative value has the same shape of problem: which config you get
        // depends on your cwd. `MEKA_DATA_DIR` refuses both for the same reasons, and more
        // sharply: `meka.db` holds every provider credential.
        if path.as_os_str().is_empty() {
            // `warn!`, not `debug!`: an override the user set and meka did not honour is exactly
            // the recoverable-fallback case, and the environment-variable documentation says a
            // warning is what happens.
            tracing::warn!("MEKA_CONFIG_DIR is empty; using the platform config directory");
        } else if !path.is_absolute() {
            tracing::warn!(
                "MEKA_CONFIG_DIR '{}' is not an absolute path; ignoring it and using the platform \
                 config directory",
                path.display()
            );
        } else {
            return Some(path);
        }
    }
    dirs::config_dir().map(|directory| directory.join("meka"))
}

pub(crate) fn config_file_path() -> Option<PathBuf> {
    meka_config_dir().map(|dir| dir.join("config.toml"))
}

/// Resolve the user's standing instructions across the tiers that can carry them, most specific
/// first: `--instructions`, then `MEKA_INSTRUCTIONS`, then `MEKA_INSTRUCTIONS_FILE`, then the
/// conventional path under the config directory.
///
/// Inline and file are two *transports* for one setting rather than two ways of saying the same
/// thing, which is why both survive. Which is natural follows from the channel: a filesystem
/// channel (the config directory) points at content, while a string channel (argv, environment)
/// carries it. Demanding a file from the latter can be impossible, not merely inconvenient: the
/// `mekabox` wrapper mounts the host config directory read-only and then overrides the instructions
/// for the container, which it can do with one `-e` and could not do at all if a path were the only
/// accepted form.
///
/// Setting both environment variables is refused rather than silently resolved. There is no reading
/// under which someone meant both, so picking one would just hide the mistake until the agent
/// behaved unexpectedly.
/// [`resolve_instructions`] over the persistent tiers only, for `meka instructions show`. A
/// per-run `--instructions` isn't part of what a standalone query is asking about.
pub(crate) fn resolve_instructions_for_display()
-> crate::error::Result<Option<crate::instructions::Instructions>> {
    resolve_instructions(None)
}

fn resolve_instructions(
    flag: Option<&str>,
) -> crate::error::Result<Option<crate::instructions::Instructions>> {
    if let Some(text) = flag {
        let text = text.trim();
        return Ok(
            (!text.is_empty()).then(|| crate::instructions::Instructions {
                text: text.to_string(),
                source: crate::instructions::InstructionsSource::Flag,
            }),
        );
    }

    let inline = std::env::var("MEKA_INSTRUCTIONS").ok();
    let from_file = std::env::var("MEKA_INSTRUCTIONS_FILE").ok();
    if inline.is_some() && from_file.is_some() {
        return Err(crate::error::MekaError::Config(
            "MEKA_INSTRUCTIONS and MEKA_INSTRUCTIONS_FILE are both set; keep one. The first \
             carries the text itself, the second a path to it."
                .to_string(),
        ));
    }

    if let Some(text) = inline {
        let text = text.trim();
        return Ok(
            (!text.is_empty()).then(|| crate::instructions::Instructions {
                text: text.to_string(),
                source: crate::instructions::InstructionsSource::Env,
            }),
        );
    }
    if let Some(path) = from_file {
        return crate::instructions::read_explicit(Path::new(&path)).map(Some);
    }
    Ok(crate::instructions::discover())
}

/// Resolve `path` through a final-component symlink, leaving anything else untouched.
///
/// Returns `path` unchanged when it is not a link, when the link cannot be read, or when the target
/// is itself a link that leads nowhere useful: every failure mode falls back to writing where the
/// caller asked, which is the previous behaviour. A relative link target is joined onto the link's
/// own directory, as the OS resolves it.
///
/// Chains are followed to a small depth so a link-to-a-link still lands on the real file, with the
/// bound there to stop a cycle (`a -> b -> a`) spinning.
fn resolve_symlink_target(path: &Path) -> std::path::PathBuf {
    const MAX_LINK_DEPTH: usize = 8;

    let mut current = path.to_path_buf();
    for _ in 0..MAX_LINK_DEPTH {
        let is_link = std::fs::symlink_metadata(&current)
            .map(|metadata| metadata.file_type().is_symlink())
            .unwrap_or(false);
        if !is_link {
            return current;
        }
        let Ok(target) = std::fs::read_link(&current) else {
            return current;
        };
        current = if target.is_absolute() {
            target
        } else {
            match current.parent() {
                Some(parent) => parent.join(target),
                None => return current,
            }
        };
    }
    tracing::warn!(
        "'{}' is a symlink chain deeper than {} links; writing to the link itself",
        path.display(),
        MAX_LINK_DEPTH
    );
    path.to_path_buf()
}

/// Exclusive cross-process lock over `config.toml`, held for a whole read-modify-write.
///
/// Every editor of the file reads it, mutates a `toml_edit` document, and writes the whole thing
/// back. Two of those interleaving means the second write is computed from a snapshot taken before
/// the first, so the first is silently discarded -- `write_file_atomic` makes each *write* atomic,
/// which does nothing for a lost update. This is not a hypothetical race between two humans running
/// CLI commands: `device_id::persist` runs from `ResolvedConfig::from_cli` on every start for a
/// `claude-subscription` profile without one, so an ordinary launch races `meka mcp add`.
///
/// The lock lives beside the file rather than on it, because the editors replace the file by rename
/// and a lock held on the replaced inode would guard nothing.
/// Reentrant within a thread. `flock` is associated with the open file description, so a second
/// `open` in the same process conflicts with the first exactly as another process would: nesting
/// `lock_config_file()` inside a held one self-deadlocks. That nesting is real -- `meka mcp add` on
/// an HTTPS URL holds the lock and then calls `persist_auth_block_for`, which wants it too -- so
/// the depth is tracked and only the outermost acquisition touches the file.
pub(crate) enum ConfigFileLock {
    /// The outermost acquisition on this thread; owns the `flock` and releases it on drop.
    Held {
        _guard: fd_lock::RwLockWriteGuard<'static, std::fs::File>,
        _lock: Box<fd_lock::RwLock<std::fs::File>>,
    },
    /// A nested acquisition. The outer guard still holds the file; this one only keeps the depth
    /// accurate so the outer release happens at the right moment.
    Reentrant,
}

thread_local! {
    /// How many [`ConfigFileLock`]s this thread currently holds.
    ///
    /// Thread-local while the `flock` underneath it is per open file description, and that pairing
    /// only holds because no `ConfigFileLock` is ever held across an `.await` on a task that can
    /// migrate. Every acquisition today runs inside `runtime.block_on` on the main thread, which
    /// never moves. If one were ever held across an await inside a `tokio::spawn`, the task could
    /// resume on a different worker and both halves of this would break at once: a nested
    /// acquisition there would see depth 0 and block on an `flock` this process already holds, and
    /// the guard would decrement the wrong thread's counter, leaving the original stuck above zero
    /// so every later acquisition on it silently took no lock at all.
    static CONFIG_LOCK_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

impl Drop for ConfigFileLock {
    fn drop(&mut self) {
        CONFIG_LOCK_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

/// Take [`ConfigFileLock`]. Blocks until the holder releases it, which is right for a CLI edit:
/// the alternative is failing a `meka mcp add` because a background `device_id` write happened to
/// be in flight.
pub(crate) fn lock_config_file() -> std::io::Result<ConfigFileLock> {
    if CONFIG_LOCK_DEPTH.with(|depth| depth.get()) > 0 {
        CONFIG_LOCK_DEPTH.with(|depth| depth.set(depth.get() + 1));
        return Ok(ConfigFileLock::Reentrant);
    }

    let directory = meka_config_dir()
        .ok_or_else(|| std::io::Error::other("could not determine the config directory"))?;
    std::fs::create_dir_all(&directory)?;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(directory.join(".config.toml.lock"))?;

    let mut lock = Box::new(fd_lock::RwLock::new(file));
    let guard = lock.write()?;
    // SAFETY: `guard` borrows from `*lock`. The box is moved, not the `RwLock` inside it, so the
    // lock's heap address is stable for as long as the box lives, and the field order above drops
    // `_guard` before `_lock`. Same shape as `FileLock` in `crate::session`.
    let guard: fd_lock::RwLockWriteGuard<'static, std::fs::File> =
        unsafe { std::mem::transmute(guard) };
    CONFIG_LOCK_DEPTH.with(|depth| depth.set(1));
    Ok(ConfigFileLock::Held {
        _guard: guard,
        _lock: lock,
    })
}

/// Write `content` to `path` atomically: serialise to a `<name>.<pid>.<seq>.tmp` beside it,
/// `sync_all` the fd, then `rename` over the target. Also creates the parent directory (0700 on
/// Unix, unless a symlink redirected the write out of meka's own tree) and holds the final file to
/// at most 0600 on Unix, so `auth_token` / OAuth-derived secrets aren't group- or world-readable
/// regardless of the user's umask or of what mode the target already had.
///
/// Not config-specific despite living here: the same durability and permission guarantees are what
/// the skill store under the config dir wants, so `meka skill` bodies go through it too.
///
/// A symlinked target is followed (see [`resolve_symlink_target`]), so writing through a
/// dotfile-manager link updates the tracked file instead of replacing the link.
pub(crate) fn write_file_atomic(path: &Path, content: &str) -> std::io::Result<()> {
    use std::io::Write as _;

    // Follow a symlink to its target before choosing where to write.
    //
    // `rename(2)` replaces the directory entry, so renaming over a symlink destroys the link itself
    // rather than updating what it points at. Dotfile managers (stow, chezmoi, yadm) leave
    // `~/.config/meka/config.toml` as a link into a tracked repo, and one `meka mcp add` turned it
    // into a plain file: the tracked copy went stale, every later edit diverged from it, and the
    // next `stow --restow` silently reverted the lot. Resolving first keeps the write on the file
    // the user actually manages, and the atomicity below is unaffected because the temp file is
    // then created beside the *target*.
    //
    // Only the link itself is resolved, not the whole path: `canonicalize` would also resolve
    // symlinked parent directories, which changes where the temp file lands for no benefit here.
    let resolved = resolve_symlink_target(path);
    // Whether the write is still landing inside meka's own tree. The 0700 tightening below is a
    // statement about a directory meka owns, and following a link takes the write somewhere it does
    // not: a dotfile manager's `~/dotfiles/config.toml` meant one `meka mcp disable` re-moded the
    // whole repository directory to 0700 and every unrelated file in it stayed put but newly
    // hidden -- a permission fact meka never established and cannot restore. Observed.
    // Consulted only by the Unix mode-tightening branch below; there is no mode to tighten
    // elsewhere, so computing it on those targets is dead work the compiler rightly flags.
    #[cfg(unix)]
    let redirected = resolved.as_path() != path;
    let path = resolved.as_path();

    if let Some(parent) = path.parent() {
        // Create newly-missing parents already at 0700 to avoid the umask window left by
        // `create_dir_all` followed by `set_permissions`. `DirBuilderExt::mode` passes the mode
        // straight to `mkdir(2)`.
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            std::fs::DirBuilder::new()
                .mode(0o700)
                .recursive(true)
                .create(parent)?;
        }
        #[cfg(not(unix))]
        std::fs::create_dir_all(parent)?;

        #[cfg(unix)]
        if !redirected {
            use std::os::unix::fs::PermissionsExt;
            // Pre-existing dirs may have a different mode (e.g. user pre-created `~/.config` at
            // 0755). Best-effort tighten to 0700; failure here gets a warning rather than aborting
            // the write.
            if let Ok(metadata) = std::fs::metadata(parent) {
                let mut perms = metadata.permissions();
                if perms.mode() & 0o777 != 0o700 {
                    perms.set_mode(0o700);
                    if let Err(error) = std::fs::set_permissions(parent, perms) {
                        tracing::warn!(
                            "failed to tighten '{}' to 0700: {}",
                            parent.display(),
                            error
                        );
                    }
                }
            }
        }
    }

    let file_name = path.file_name().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "path has no file name")
    })?;
    // Unique per writer, and created exclusively. A fixed `<file>.tmp` is shared by every writer of
    // the same path: both `open(O_TRUNC)` the same inode, both write from offset zero, and whoever
    // renames last publishes a byte-level splice of two documents as a success. Reproduced -- 4 MiB
    // of one body with 64 KiB of another laid over its front, renamed into place, `Ok(())` returned
    // to both. The pid separates processes and the counter separates threads within one;
    // `create_new` is what makes the pair a guarantee rather than a strong hope.
    //
    // `write_file_bytes` in `crate::tools::file` already carried the pid half of this for the same
    // reason. This is the primitive every *store* writes through, and it had neither half.
    static TEMP_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let (tmp_path, mut file) = loop {
        let mut tmp_name = file_name.to_os_string();
        tmp_name.push(format!(
            ".{}.{}.tmp",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let candidate = path.with_file_name(tmp_name);
        match options.open(&candidate) {
            Ok(file) => break (candidate, file),
            // Only a name collision is worth retrying; anything else is the real error.
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    };
    // Every failure arm removes the temp file. A unique name means a leftover is never reused, so
    // without this each interrupted write leaves one behind for ever.
    let write = (|| -> std::io::Result<()> {
        file.write_all(content.as_bytes())?;
        file.sync_all()
    })();
    drop(file);
    if let Err(error) = write {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(error);
    }

    // An existing file keeps whatever mode it had, *narrowed* to at most 0600.
    //
    // Two failures meet here and only this rule avoids both. Handing the target the temp's fresh
    // 0600 unconditionally re-moded a file meka did not create, which is wrong for a `config.toml`
    // a dotfile manager owns. But simply carrying the old mode across is worse: it left
    // `auth_token = "sk-..."` sitting in a world-readable file after an ordinary `meka mcp add`,
    // because the write follows the symlink out of meka's 0700 directory and the group and other
    // bits came along. Narrowing keeps a deliberately *tighter* mode like 0400 and refuses to
    // publish a looser one into a file meka is putting secrets in.
    //
    // Said out loud when it bites, because it is a change to something the user set.
    #[cfg(unix)]
    if let Ok(existing) = std::fs::metadata(path) {
        use std::os::unix::fs::PermissionsExt;
        let existing_mode = existing.permissions().mode() & 0o777;
        let narrowed = existing_mode & 0o600;
        if narrowed != existing_mode {
            tracing::warn!(
                "tightening '{}' from {:o} to {:o}: meka writes credentials through this path and \
                 will not leave them group- or world-readable",
                path.display(),
                existing_mode,
                narrowed
            );
        }
        if let Err(error) =
            std::fs::set_permissions(&tmp_path, std::fs::Permissions::from_mode(narrowed))
        {
            tracing::warn!(
                "failed to set the mode of '{}' on the replacement: {}",
                path.display(),
                error
            );
        }
    }

    std::fs::rename(&tmp_path, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp_path);
    })
}

/// Load `config.toml`, returning the parsed file and any error that stopped it from parsing.
///
/// A malformed config used to warn and fall back to `ConfigFile::default()`, which is the worst of
/// the options: one mistyped key silently reconfigured the whole agent, running it with no provider
/// profiles, no MCP servers and default permissions, off a single line the user could easily scroll
/// past. The error is carried instead and raised by [`ResolvedConfig::validate`], alongside the
/// profile-selection failures, so `from_cli` can stay infallible.
///
/// The line between failing and continuing is whether a command *consults* the parsed config. One
/// that does can only answer from empty defaults, which reads as fact: `meka provider list`
/// printing "No provider profiles configured." over a file full of them. Those callers use
/// [`load_config_file_or_err`] / [`ResolvedConfig::require_readable_config`].
///
/// Commands that instead edit the raw document through `toml_edit` are unaffected by an unknown key
/// and are how a broken config gets fixed from the CLI, so they run anyway: `meka mcp add` /
/// `remove` / `enable` / `disable` note the failure via
/// [`ResolvedConfig::warn_if_config_unreadable`], and `meka provider remove` never loads the parsed
/// config at all. Each re-reads the file itself, and each must fail on a *read* error rather than
/// substitute an empty document, or the write-back truncates what it couldn't read.
pub(crate) fn load_config_file() -> (ConfigFile, Option<String>) {
    let Some(path) = config_file_path() else {
        return (ConfigFile::default(), None);
    };

    match std::fs::read_to_string(&path) {
        Ok(contents) => match toml::from_str(&contents) {
            Ok(config) => (config, None),
            Err(error) => (
                ConfigFile::default(),
                Some(format!("failed to parse {}: {}", path.display(), error)),
            ),
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (ConfigFile::default(), None),
        Err(error) => (
            ConfigFile::default(),
            Some(format!("failed to read {}: {}", path.display(), error)),
        ),
    }
}

/// [`load_config_file`] for callers that read profiles but have no `validate()` to route the error
/// through: the `meka provider` and `meka account` subcommands.
///
/// Fails rather than falling back to an empty `ConfigFile`, because for these callers "empty" is
/// indistinguishable from "your profiles are gone". `meka provider add <existing>` is the sharp
/// edge: its duplicate guard is the parsed map, so an empty one lets the add through to
/// `upsert_profile_document`, which replaces the profile's table wholesale and drops every field
/// the flags didn't set. An unparseable config used to route the user straight into that ("no
/// provider profile named 'work'. Run `meka provider add work` to create it.").
pub(crate) fn load_config_file_or_err() -> crate::error::Result<ConfigFile> {
    let (config_file, error) = load_config_file();
    match error {
        Some(error) => Err(crate::error::MekaError::Config(error)),
        None => Ok(config_file),
    }
}

/// Resolve the runtime permission level and the set of enabled modes from the layered config
/// sources. CLI > env > config file > built-in defaults. Invalid entries warn and are dropped;
/// out-of-set overrides (e.g. `--permission ask` when `ask` is disabled) warn and clamp to the
/// configured default rather than refusing to start, mirroring the `[tools.tool_permissions]`
/// warn-and-skip pattern.
fn resolve_permission(
    cli_permission: Option<Permission>,
    env_permission: Option<&str>,
    file_default: Option<&str>,
    file_enabled: Option<&[String]>,
) -> (Permission, EnabledPermissions) {
    let enabled = match file_enabled {
        Some(list) => {
            let parsed: Vec<Permission> = list
                .iter()
                .filter_map(|raw| match raw.parse::<Permission>() {
                    Ok(mode) => Some(mode),
                    // The parser's own message, not a list restated here, so the two cannot drift.
                    Err(error) => {
                        tracing::warn!(
                            "ignoring invalid [permissions].enabled entry '{}': {}",
                            raw,
                            error
                        );
                        None
                    }
                })
                .collect();
            match EnabledPermissions::from_modes(parsed) {
                Some(set) => set,
                None => {
                    // `{read}`, not `DEFAULT`. The user wrote a list; every entry being
                    // unrecognised means their intent could not be honoured, and `DEFAULT` answers
                    // that by handing back *four* modes including `unrestricted`. A pre-split
                    // config pinning `enabled = ["write"]` -- exactly one mode, deliberately --
                    // came back with the whole ladder reachable by Shift+Tab. Falling back to the
                    // narrowest useful set keeps a failed parse from widening authority.
                    tracing::warn!(
                        "[permissions].enabled was empty after filtering, so none of the modes \
                         you listed could be used; falling back to `read` alone rather than to \
                         the default set, which would be wider than what you asked for"
                    );
                    EnabledPermissions::from_modes([Permission::Read])
                        .unwrap_or(EnabledPermissions::DEFAULT)
                }
            }
        }
        None => EnabledPermissions::DEFAULT,
    };

    let configured_default = file_default.and_then(|raw| match raw.parse::<Permission>() {
        Ok(mode) => Some(mode),
        Err(error) => {
            tracing::warn!(
                "ignoring invalid [permissions].default '{}': {}",
                raw,
                error
            );
            None
        }
    });

    let resolved_default = match configured_default {
        Some(mode) if enabled.is_enabled(mode) => mode,
        Some(mode) => {
            tracing::warn!(
                "[permissions].default = '{}' is not in [permissions].enabled; \
                 falling back",
                mode
            );
            if enabled.is_enabled(Permission::Read) {
                Permission::Read
            } else {
                enabled.lowest()
            }
        }
        None => {
            if enabled.is_enabled(Permission::Read) {
                Permission::Read
            } else {
                enabled.lowest()
            }
        }
    };

    let env_override = env_permission.and_then(|raw| match raw.parse::<Permission>() {
        Ok(mode) => Some(mode),
        Err(error) => {
            tracing::warn!("ignoring invalid MEKA_PERMISSION='{}': {}", raw, error);
            None
        }
    });

    let raw_choice = cli_permission.or(env_override);
    let permission = match raw_choice {
        Some(mode) if enabled.is_enabled(mode) => mode,
        Some(mode) => {
            tracing::warn!(
                "requested start mode '{}' is not in [permissions].enabled; using '{}'",
                mode,
                resolved_default
            );
            resolved_default
        }
        None => resolved_default,
    };

    (permission, enabled)
}

/// `MEKA_SANDBOX_BACKEND` overrides `[shell].sandbox_backend` for non-interactive / containerized
/// runs (the `mekabox` wrapper sets it to pin Landlock and silence the auto-resolve warning when
/// it mounts the host config read-only). Accepts `landlock` / `bubblewrap` (case-insensitive); an
/// unrecognized value is warned about and ignored. Like the config field, this only affects the
/// resolved backend on Linux.
fn sandbox_backend_override() -> Option<SandboxBackend> {
    parse_sandbox_backend_override(&std::env::var("MEKA_SANDBOX_BACKEND").ok()?)
}

/// Parse a `MEKA_SANDBOX_BACKEND` value (case-insensitive, trimmed). Empty or unrecognized values
/// yield `None`; unrecognized ones also warn. Split from [`sandbox_backend_override`] so the
/// parsing is unit-testable without mutating process env.
fn parse_sandbox_backend_override(value: &str) -> Option<SandboxBackend> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    // Env tolerates a bad value (warn and ignore); the CLI path uses `FromStr` directly and errors.
    match trimmed.parse() {
        Ok(backend) => Some(backend),
        Err(message) => {
            tracing::warn!("ignoring MEKA_SANDBOX_BACKEND: {}", message);
            None
        }
    }
}

/// Resolve the active Linux sandbox backend.
///
/// When the user pinned `[shell].sandbox_backend = "..."` in `config.toml`, that choice is binding,
/// no silent fallback at runtime; an unavailable explicit backend surfaces at use time via the
/// `BackendProbe::Missing` / `UserNamespaceDenied` variants.
///
/// When the value is unset (`None`), meka probes bubblewrap and picks it if available, falling back
/// to landlock otherwise. The `auto_resolved` flag is propagated so the startup warn helper can
/// nudge the user once toward installing bwrap (without nagging users who explicitly pinned
/// landlock).
#[cfg(target_os = "linux")]
fn resolve_sandbox_backend(
    configured: Option<SandboxBackend>,
) -> (SandboxBackend, bool, crate::sandbox::BackendProbe) {
    use crate::sandbox::{BackendProbe, probe_backend};

    // Probe Bubblewrap only when its result is load-bearing for the resolution: either the user
    // pinned it explicitly, or no value was configured (so we need the probe to decide whether to
    // auto-pick it). When the user pinned Landlock, the Bubblewrap smoke test would be pure waste
    // (~500 ms on every meka start).
    let (backend, auto_resolved, cached_bubblewrap_probe) = match configured {
        Some(explicit) => (explicit, false, None),
        None => {
            let probe = probe_backend(SandboxBackend::Bubblewrap);
            let picked = match &probe {
                BackendProbe::Ok(_) => SandboxBackend::Bubblewrap,
                _ => SandboxBackend::Landlock,
            };
            (picked, true, Some(probe))
        }
    };
    // The Landlock arm discards `cached_bubblewrap_probe` because the auto-resolve path that
    // populated it landed on Bubblewrap (it only falls through to Landlock when Bubblewrap probes
    // unavailable, and that probe isn't useful for the chosen backend's status).
    let backend_probe = match (backend, cached_bubblewrap_probe) {
        (SandboxBackend::Bubblewrap, Some(probe)) => probe,
        (SandboxBackend::Bubblewrap, None) => probe_backend(SandboxBackend::Bubblewrap),
        (SandboxBackend::Landlock, _) => probe_backend(SandboxBackend::Landlock),
    };
    (backend, auto_resolved, backend_probe)
}

/// Non-Linux platforms have a single platform-native sandbox (`sandbox-exec` on macOS,
/// Low-integrity on Windows, nothing elsewhere). `[shell].sandbox_backend` is documented as
/// Linux-only and is ignored here: the resolved capability comes from [`crate::sandbox::detect`]
/// and is surfaced through the same `BackendProbe::Ok` envelope so the downstream wiring in
/// `src/main.rs` doesn't need a platform branch.
#[cfg(not(target_os = "linux"))]
fn resolve_sandbox_backend(
    _configured: Option<SandboxBackend>,
) -> (SandboxBackend, bool, crate::sandbox::BackendProbe) {
    use crate::sandbox::{BackendProbe, SandboxCapability};

    let probe = match crate::sandbox::detect() {
        SandboxCapability::Unavailable => BackendProbe::Missing {
            reason: "no platform sandbox backend available".to_string(),
        },
        capability => BackendProbe::Ok(capability),
    };
    // `SandboxBackend::Landlock` is a stand-in here; the field exists for Linux config parity but
    // is never consulted on this platform.
    (SandboxBackend::Landlock, false, probe)
}

/// Merge `--eager-load-tool SERVER:TOOL` CLI values into the matching server's
/// [`McpServerConfig::eager_load_tools`] list. Malformed entries and unknown server names warn and
/// are skipped, same philosophy as `warn_on_stale_tool_config`. Appends to (never replaces) the
/// configured list, and deduplicates so a CLI flag that overlaps with `config.toml` doesn't grow
/// the list.
fn apply_cli_eager_load_overrides(raw_pairs: &[String], servers: &mut [McpServerConfig]) {
    for raw in raw_pairs {
        let (server_name, tool_name) = match raw.split_once(':') {
            Some((server, tool)) => {
                let server = server.trim();
                let tool = tool.trim();
                if server.is_empty() || tool.is_empty() {
                    tracing::warn!(
                        "ignoring --eager-load-tool '{}' (expected SERVER:TOOL with both \
                         parts non-empty)",
                        raw
                    );
                    continue;
                }
                (server, tool)
            }
            None => {
                tracing::warn!(
                    "ignoring --eager-load-tool '{}' (expected SERVER:TOOL format)",
                    raw
                );
                continue;
            }
        };
        match servers.iter_mut().find(|s| s.name == server_name) {
            Some(server) => {
                let list = server.eager_load_tools.get_or_insert_with(Vec::new);
                if !list.iter().any(|existing| existing == tool_name) {
                    list.push(tool_name.to_string());
                }
            }
            None => {
                tracing::warn!(
                    "--eager-load-tool '{}': no MCP server named '{}' is configured",
                    raw,
                    server_name
                );
            }
        }
    }
}

/// Resolve which provider profile is active given an explicit request (from `--provider` or
/// `default_provider`) and the configured profiles. Returns `(active_profile, provider_error)`:
/// exactly one is `Some`. A deferred error string (rather than a hard failure) keeps `from_cli`
/// infallible; `validate()` surfaces it later with guidance.
pub(crate) fn select_active_profile(
    requested: Option<String>,
    providers: &std::collections::BTreeMap<String, ProviderProfile>,
) -> (Option<String>, Option<String>) {
    match requested {
        Some(name) if providers.contains_key(&name) => (Some(name), None),
        Some(_) if providers.is_empty() => (
            None,
            Some("no provider profiles configured. Run `meka provider add <name>`.".to_string()),
        ),
        Some(name) => (
            None,
            Some(format!(
                "no provider profile named '{}' (configured: {}). Pass `--provider` or set \
                 `default_provider`.",
                name,
                providers.keys().cloned().collect::<Vec<_>>().join(", ")
            )),
        ),
        None => match providers.len() {
            0 => (
                None,
                Some(
                    "no provider configured. Run `meka provider add <name>` to set one up."
                        .to_string(),
                ),
            ),
            1 => (providers.keys().next().cloned(), None),
            _ => (
                None,
                Some(format!(
                    "multiple provider profiles configured ({}); set `default_provider` or pass \
                     `--provider <name>`.",
                    providers.keys().cloned().collect::<Vec<_>>().join(", ")
                )),
            ),
        },
    }
}

impl ResolvedConfig {
    pub fn from_cli(cli: &Cli) -> Self {
        let (config_file, config_error) = load_config_file();
        let providers = config_file.providers;
        // Select the active profile: `--provider` flag, else `default_provider`, else the sole
        // profile. Absence / ambiguity / unknown name becomes a deferred error surfaced by
        // `validate()` so `from_cli` stays infallible.
        let (active_profile, provider_error) = select_active_profile(
            cli.provider
                .clone()
                .or_else(|| config_file.default_provider.clone()),
            &providers,
        );
        let active = active_profile.as_ref().and_then(|name| providers.get(name));
        let file_display = config_file.display.unwrap_or_default();
        let file_web = config_file.web.unwrap_or_default();
        let file_shell = config_file.shell.unwrap_or_default();
        let file_session = config_file.session.unwrap_or_default();
        let file_thinking = config_file.thinking.unwrap_or_default();
        let file_tools = config_file.tools.unwrap_or_default();
        let file_skills = config_file.skills.unwrap_or_default();
        let skills_enabled = file_skills.enabled.unwrap_or(true);
        let skills_agent_managed = file_skills.agent_managed.unwrap_or(false);
        let skills_extra_paths = resolve_skills_extra_paths(
            file_skills.extra_paths.as_deref().unwrap_or_default(),
            crate::skills::skills_dir().as_deref(),
        );
        // The two keys read as independent but are not: `enabled = false` registers no skill tools
        // at all, so the authoring pair never appears however this is set. Saying so beats leaving
        // someone to conclude that `agent_managed` is broken.
        if skills_agent_managed && !skills_enabled {
            tracing::warn!(
                "[skills] agent_managed is on but enabled is false, so no skill tools are \
                 registered at all; set enabled = true to let the agent author skills"
            );
        }
        let memory_enabled = config_file
            .memory
            .unwrap_or_default()
            .enabled
            .unwrap_or(true);
        let background = ResolvedBackgroundConfig::resolve(config_file.background);
        let subagents = ResolvedSubagentsConfig::resolve(config_file.subagents);
        // Destructure the [mcp] table into its two independent fields so we don't have to re-open
        // the config file later for resolution.
        let (
            mcp_default_permission_str,
            mut mcp_servers,
            mcp_strict,
            mcp_grace,
            mcp_connect_timeout,
        ) = match config_file.mcp {
            Some(mcp) => (
                mcp.default_permission,
                mcp.servers.unwrap_or_default(),
                mcp.strict.unwrap_or(false),
                std::time::Duration::from_secs(mcp.grace_seconds.unwrap_or(3)),
                std::time::Duration::from_secs(mcp.connect_timeout_seconds.unwrap_or(30)),
            ),
            None => (
                None,
                Vec::new(),
                false,
                std::time::Duration::from_secs(3),
                std::time::Duration::from_secs(30),
            ),
        };
        apply_cli_eager_load_overrides(&cli.eager_load_tool, &mut mcp_servers);
        // Settle `required` here so no later consumer has to remember that `None` means "inherit
        // strict". Everything downstream reads a plain `Some(bool)`, and `strict` deliberately
        // isn't carried on `ResolvedConfig`: a second copy of the same answer is one a future
        // reader could gate on, believing it still decides turns on its own. It doesn't.
        for server in &mut mcp_servers {
            server.required = Some(server.required.unwrap_or(mcp_strict));
        }
        let mcp_default_permission = match mcp_default_permission_str.as_deref() {
            Some(raw) => match raw.parse::<Permission>() {
                Ok(permission) => Some(permission),
                Err(error) => {
                    tracing::warn!(
                        "ignoring invalid [mcp].default_permission '{}': {}",
                        raw,
                        error
                    );
                    None
                }
            },
            None => None,
        };

        let (instructions, instructions_error) =
            match resolve_instructions(cli.instructions.as_deref()) {
                Ok(found) => (found, None),
                Err(error) => (None, Some(error.to_string())),
            };
        if let Some(found) = &instructions {
            crate::instructions::warn_if_large(found);
            tracing::info!("instructions loaded from {}", found.source);
        }
        let user_instructions_source = instructions.as_ref().map(|found| found.source.to_string());
        let user_instructions = instructions.map(|found| found.text);
        let provider_profiles: Vec<(String, String, Option<String>)> = providers
            .iter()
            .map(|(name, profile)| (name.clone(), profile.backend.clone(), profile.model.clone()))
            .collect();

        let builtin_allowed_tools = file_tools
            .allowed_tools
            .filter(|list| !list.is_empty())
            .map(|list| list.into_iter().map(|s| s.trim().to_string()).collect());
        let builtin_disabled_tools = file_tools
            .disabled_tools
            .unwrap_or_default()
            .into_iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let builtin_tool_permissions = file_tools
            .tool_permissions
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(name, level)| {
                let name = name.trim().to_string();
                if name.is_empty() {
                    return None;
                }
                match level.parse::<Permission>() {
                    Ok(permission) => Some((name, permission)),
                    Err(error) => {
                        tracing::warn!(
                            "ignoring invalid [tools.tool_permissions] entry '{}' = '{}': {}",
                            name,
                            level,
                            error
                        );
                        None
                    }
                }
            })
            .collect();

        // Provider config comes from the active profile (no env tier); the credential is loaded
        // from the DB in main.rs by `active_profile`. CLI `--model` / `--base-url` override the
        // profile's values for this run.
        let provider_name = active.map(|profile| profile.backend.clone());

        let model = cli
            .model
            .clone()
            .or_else(|| active.and_then(|profile| profile.model.clone()));

        let base_url = cli
            .base_url
            .clone()
            .or_else(|| active.and_then(|profile| profile.base_url.clone()));

        let file_permissions = config_file.permissions.unwrap_or_default();
        let (permission, enabled_permissions) = resolve_permission(
            cli.permission,
            std::env::var("MEKA_PERMISSION").ok().as_deref(),
            file_permissions.default.as_deref(),
            file_permissions.enabled.as_deref(),
        );

        // After `resolve_permission`, because a gate needs the level this process actually starts
        // at: a `meka serve --permission read` must not run a gate a write-mode session authored.
        let schedule =
            ResolvedScheduleConfig::resolve(config_file.schedule, permission, enabled_permissions);

        // Compute device_id before the struct literal so we can borrow `provider_name` and
        // `active_profile` here without conflicting with the field moves below.
        let device_id = device_id::resolve(
            provider_name.as_deref(),
            active_profile.as_deref(),
            active.and_then(|profile| profile.device_id.as_deref()),
        );
        // Default on to match Claude Code, which sends `redact-thinking` for every capable model.
        // Profiles opt out with `redact_thinking = false` to keep interleaved thinking visible.
        let redact_thinking = active
            .and_then(|profile| profile.redact_thinking)
            .unwrap_or(true);

        // Only probe the sandbox backend when sandboxing is actually enabled. Skipping the probe
        // for `sandbox = false` saves the smoke-test cost on every invocation of subcommands that
        // don't touch the shell (`meka session list`, `meka session export`, `meka mcp list`, etc.)
        // when the user has disabled sandboxing globally. The placeholder probe is never
        // consulted in that state; the shell tool short-circuits on `sandbox_enabled =
        // false`, and the warn helper early- returns on `!state.enabled`.
        // Backend precedence: `--sandbox-backend` > `MEKA_SANDBOX_BACKEND` >
        // `[shell].sandbox_backend`. An explicit value from any tier also flips
        // `auto_resolved` off, suppressing the "install Bubblewrap" startup warning.
        let configured_backend = cli
            .sandbox_backend
            .or_else(sandbox_backend_override)
            .or(file_shell.sandbox_backend);
        let (sandbox_backend, sandbox_auto_resolved, backend_probe) =
            if file_shell.sandbox.unwrap_or(true) {
                resolve_sandbox_backend(configured_backend)
            } else {
                (
                    configured_backend.unwrap_or(SandboxBackend::Landlock),
                    false,
                    crate::sandbox::BackendProbe::Missing {
                        reason: "sandbox disabled in config".to_string(),
                    },
                )
            };

        // A root that does not resolve contributes nothing to the write boundary, because
        // `writable_roots` drops what it cannot canonicalise. Silently, until a write is refused
        // with a boundary the user believed included this path. The path is kept regardless of the
        // warning: a build directory that does not exist yet is a legitimate root, and the boundary
        // is recomputed on every write rather than frozen here.
        for root in &cli.writable_root {
            if let Err(error) = std::fs::canonicalize(root) {
                tracing::warn!(
                    "--writable-root {} does not resolve ({}); it grants nothing until it exists",
                    root.display(),
                    error
                );
            }
        }

        Self {
            writable_roots: cli.writable_root.clone(),
            provider_name,
            active_profile,
            provider_error,
            config_error,
            instructions_error,
            model,
            base_url,
            client_id: active.and_then(|profile| profile.client_id.clone()),
            oauth_token_url: active.and_then(|profile| profile.oauth_token_url.clone()),
            permission,
            enabled_permissions,
            streaming: !cli.no_stream,
            session_resume: SessionResume::from_cli(cli),
            prompt: cli.prompt.clone(),
            oneshot: cli.oneshot,
            newline_before_prompt: file_display.newline_before_prompt.unwrap_or(true),
            newline_after_prompt: file_display.newline_after_prompt.unwrap_or(true),
            show_session_id_on_create: file_display.show_session_id_on_create.unwrap_or(false),
            show_session_id_on_exit: file_display.show_session_id_on_exit.unwrap_or(true),
            show_token_usage: file_display.show_token_usage.unwrap_or(false),
            resume_show_recent: file_display.resume_show_recent,
            show_path_in_prompt: file_display.show_path_in_prompt.unwrap_or(true),
            show_context_in_prompt: file_display.show_context_in_prompt.unwrap_or(false),
            web_client: WebClientConfig::from_file(&file_web),
            sandbox: file_shell.sandbox.unwrap_or(true),
            sandbox_backend,
            sandbox_auto_resolved,
            backend_probe,
            render_mode: cli
                .render_mode
                .or_else(|| {
                    std::env::var("MEKA_RENDER_MODE")
                        .ok()
                        .and_then(|value| value.parse().ok())
                })
                .or(file_display.render_mode)
                .unwrap_or_default(),
            tool_params: file_display.tool_params.unwrap_or_default(),
            max_width: file_display.max_width.map(clamp_max_width),
            context_messages: file_session
                .context_messages
                .or(Some(DEFAULT_CONTEXT_MESSAGES)),
            retention_days: file_session.retention_days,
            thinking: cli
                .thinking
                .or_else(|| active.and_then(|profile| profile.thinking))
                .unwrap_or_default(),
            thinking_budget_tokens: cli.thinking_budget.unwrap_or_else(|| {
                file_thinking
                    .budget_tokens
                    .unwrap_or(DEFAULT_THINKING_BUDGET_TOKENS)
            }),
            thinking_show_content: file_thinking.show_content.unwrap_or(false),
            device_id,
            // Pure passthrough for every backend: whatever the profile sets goes to the provider
            // verbatim (the provider trims and lowercases it). Unset means the field is omitted
            // and the provider applies its own default. An invalid value is the user's to own.
            effort: active.and_then(|profile| profile.effort.clone()),
            redact_thinking,
            auto_compact: file_session.auto_compact.unwrap_or(true),
            compact_checkpoint: file_session.compact_checkpoint.unwrap_or(true),
            // Precedence: profile > `[session].context_window`. The call sites in `main.rs` apply
            // `DEFAULT_CONTEXT_WINDOW` when both are unset.
            context_window: active
                .and_then(|profile| profile.context_window)
                .or(file_session.context_window),
            subagent_max_depth: file_session
                .subagent_max_depth
                .unwrap_or(DEFAULT_SUBAGENT_MAX_DEPTH),
            vision: active.and_then(|profile| profile.vision).unwrap_or(true),
            max_output_tokens: active.and_then(|profile| profile.max_output_tokens),
            mcp_servers,
            mcp_default_permission,
            user_instructions,
            user_instructions_source,
            provider_profiles,
            skills_enabled,
            skills_agent_managed,
            skills_extra_paths,
            memory_enabled,
            schedule,
            background,
            subagents,
            builtin_allowed_tools,
            builtin_disabled_tools,
            builtin_tool_permissions,
            input_style: file_display
                .input_style
                .as_deref()
                .map(parse_input_style)
                .unwrap_or_else(default_input_style),
            mcp_grace,
            mcp_connect_timeout,
            serve: config_file.serve,
            serve_bind_override: None,
        }
    }

    /// Refuse to answer from empty defaults when `config.toml` didn't parse.
    ///
    /// For the subcommands that never reach [`Self::validate`] but do read the parsed config:
    /// `meka mcp list` over a config full of servers would otherwise print "(no MCP servers
    /// configured)", and `meka mcp get <name>` would say the server doesn't exist. Both are
    /// indistinguishable from the truthful answer, which is what makes them worth failing on.
    pub fn require_readable_config(&self) -> crate::error::Result<()> {
        match &self.config_error {
            Some(error) => Err(crate::error::MekaError::Config(error.clone())),
            None => Ok(()),
        }
    }

    /// Every directory skills are read from, in precedence order.
    ///
    /// A method rather than the four `skill_roots(&config.skills_extra_paths)` spellings it
    /// replaces: the derivation is "meka's own first, then the configured extras", and a caller
    /// that assembled it itself was a caller that could get the order wrong.
    pub fn skill_roots(&self) -> Vec<PathBuf> {
        crate::skills::skill_roots(&self.skills_extra_paths)
    }

    /// Note an unreadable `config.toml` without failing, for the commands that edit the raw
    /// document through `toml_edit` and so work fine on one meka can't parse. They are how the file
    /// gets repaired from the CLI, so they must run; the warning is there because their view of the
    /// config is empty and any message they print about it would otherwise mislead.
    pub fn warn_if_config_unreadable(&self) {
        if let Some(error) = &self.config_error {
            tracing::warn!("ignoring config file: {}", error);
        }
    }

    pub fn validate(&self) -> crate::error::Result<()> {
        // Profile-selection failure (none / ambiguous / unknown name) is reported first with its
        // specific guidance.
        // Ahead of the provider check: with an unparsed config there are no profiles at all, and
        // "no provider configured" would send the user chasing the wrong problem.
        if let Some(error) = &self.config_error {
            return Err(crate::error::MekaError::Config(error.clone()));
        }
        if let Some(error) = &self.provider_error {
            return Err(crate::error::MekaError::Config(error.clone()));
        }
        if let Some(error) = &self.instructions_error {
            return Err(crate::error::MekaError::Config(error.clone()));
        }
        // `retention_days = 0` means "delete anything not updated in the last zero days", i.e.
        // every session, on every startup. Nobody means that, and the cost of guessing wrong is
        // unrecoverable, so refuse rather than run it once and find out.
        if self.retention_days == Some(0) {
            return Err(crate::error::MekaError::Config(
                "[session].retention_days = 0 would delete every session on each startup. \
                 Remove the key to keep sessions forever, or use `meka session delete --all`."
                    .to_string(),
            ));
        }
        // `context_messages = 0` reads as "send no history", which is not a thing the provider APIs
        // accept: a request needs at least the current user message. It also used to index one past
        // the end of the message slice on the first turn and panic -- fatal in the REPL, and under
        // ACP it left the client's `session/prompt` waiting forever with no response.
        if self.context_messages == Some(0) {
            return Err(crate::error::MekaError::Config(format!(
                "[session].context_messages = 0 would send no conversation at all. Remove the \
                     key for the default of {}, or set it to the number of messages to keep.",
                DEFAULT_CONTEXT_MESSAGES
            )));
        }
        // `tokio::time::interval` panics on a zero period, so this would be a config value taking
        // the process down rather than a setting behaving oddly.
        if self.schedule.poll_interval.is_zero() {
            return Err(crate::error::MekaError::Config(
                "[schedule].poll_interval = 0 is not a valid tick. Remove the key for the \
                 default, or set an interval like \"10s\"."
                    .to_string(),
            ));
        }
        // A zero budget fails every gate before it can produce output, so every gated job would
        // report a broken watcher forever.
        if self.schedule.gate_timeout.is_zero() {
            return Err(crate::error::MekaError::Config(
                "[schedule].gate_timeout = 0 would time out every gate before it runs. Remove \
                 the key for the default, or set a budget like \"30s\"."
                    .to_string(),
            ));
        }
        if self.schedule.max_jobs == 0 {
            return Err(crate::error::MekaError::Config(
                "[schedule].max_jobs = 0 would refuse every job. Set `[schedule] enabled = false` \
                 to turn scheduling off instead."
                    .to_string(),
            ));
        }
        if self.schedule.max_consecutive_fires == 0 {
            return Err(crate::error::MekaError::Config(
                "[schedule].max_consecutive_fires = 0 would hold every due job over forever. Set \
                 `[schedule] enabled = false` to turn scheduling off instead."
                    .to_string(),
            ));
        }
        match self.provider_name.as_deref() {
            None => {
                return Err(crate::error::MekaError::Config(
                    "no provider configured. Run `meka provider add <name>` to set one up."
                        .to_string(),
                ));
            }
            Some(name) if !crate::provider::SUPPORTED_PROVIDERS.contains(&name) => {
                return Err(crate::error::MekaError::Config(format!(
                    "profile '{}' has unknown type '{}'. Supported types: {}",
                    self.active_profile.as_deref().unwrap_or("?"),
                    name,
                    crate::provider::SUPPORTED_PROVIDERS.join(", "),
                )));
            }
            Some(_) => {}
        }
        if self.model.is_none() {
            return Err(crate::error::MekaError::Config(format!(
                "no model configured for profile '{}'. Set `model` in its [providers.<name>] \
                 table, or pass --model.",
                self.active_profile.as_deref().unwrap_or("?"),
            )));
        }
        validate_max_output_tokens(
            self.provider_name.as_deref(),
            self.max_output_tokens,
            self.thinking,
            self.thinking_budget_tokens,
        )?;
        Ok(())
    }
}

/// Reject a `max_output_tokens` override that can't produce a valid Claude request: under
/// [`crate::provider::ThinkingMode::Budgeted`] the budget is drawn from `max_tokens`, so the cap
/// must exceed it. Surfaced as a config error with clear guidance rather than a provider 400
/// mid-turn. The other two modes send no `budget_tokens` and have no such constraint, and neither
/// do the OpenAI backends.
fn validate_max_output_tokens(
    provider_name: Option<&str>,
    max_output_tokens: Option<u64>,
    thinking: crate::provider::ThinkingMode,
    thinking_budget_tokens: u64,
) -> crate::error::Result<()> {
    let Some(max_output) = max_output_tokens else {
        return Ok(());
    };
    // The same predicate `/status` and `meka provider add` use, rather than a third copy of the
    // backend list: the invariant belongs to the Messages protocol, so every backend that speaks it
    // is in scope and a future one is in scope automatically. Missing a backend here is invisible
    // until a real request is rejected mid-turn, which is what this check exists to pre-empt.
    let speaks_messages = provider_name.is_some_and(crate::provider::backend_takes_thinking);
    let budgeted = matches!(thinking, crate::provider::ThinkingMode::Budgeted);
    if speaks_messages && budgeted && max_output <= thinking_budget_tokens {
        return Err(crate::error::MekaError::Config(format!(
            "max_output_tokens ({}) must exceed the thinking budget ({}) for an Anthropic Messages \
             profile with thinking = \"budgeted\"; raise max_output_tokens or lower \
             [thinking].budget_tokens.",
            max_output, thinking_budget_tokens,
        )));
    }
    Ok(())
}

/// Stable per-device identity for `claude-subscription` (embedded in `metadata.user_id`). Other
/// providers get an empty string; we don't write a stub config file just to hold an unused value.
mod device_id {
    use std::path::Path;

    use super::{config_file_path, write_file_atomic};

    /// Lookup order: configured → Claude Code's `~/.claude.json` userID → freshly generated. The
    /// claude.json fallback lets meka and Claude Code on the same machine share a device identity.
    /// A freshly seeded value is persisted into the active profile's `[providers.<name>].device_id`
    /// so it stays stable across runs.
    pub(super) fn resolve(
        backend: Option<&str>,
        profile_name: Option<&str>,
        configured: Option<&str>,
    ) -> String {
        if backend != Some("claude-subscription") {
            return String::new();
        }

        // A configured value (the active profile's `device_id`, already deserialized for us) wins;
        // no need to re-read the file for it.
        if let Some(id) = configured
            && !id.is_empty()
        {
            return id.to_string();
        }

        let (id, source) = match read_claude_code_user_id() {
            Some(id) => (id, "~/.claude.json"),
            None => (generate(), "random"),
        };
        tracing::info!("seeded claude-subscription device_id from {}", source);

        // Persist into the active profile's table. Skip quietly when we can't locate the config
        // path or the profile name; the id is still returned and used for this run, just
        // not saved.
        if let (Some(profile), Some(path)) = (profile_name, config_file_path())
            && let Err(error) = persist(&path, profile, &id)
        {
            tracing::warn!("failed to persist device_id: {}", error);
        }
        id
    }

    fn generate() -> String {
        use rand::RngExt;
        let mut bytes = [0u8; 32];
        rand::rng().fill(&mut bytes);
        bytes.iter().map(|byte| format!("{:02x}", byte)).collect()
    }

    fn read_claude_code_user_id() -> Option<String> {
        read_user_id_from(&dirs::home_dir()?.join(".claude.json"))
    }

    pub(super) fn read_user_id_from(path: &Path) -> Option<String> {
        let contents = match std::fs::read_to_string(path) {
            Ok(contents) => contents,
            Err(error) => {
                tracing::debug!("could not read user-id file {}: {}", path.display(), error);
                return None;
            }
        };
        let document: serde_json::Value = match serde_json::from_str(&contents) {
            Ok(document) => document,
            Err(error) => {
                tracing::debug!("could not parse user-id file {}: {}", path.display(), error);
                return None;
            }
        };
        let id = document.get("userID")?.as_str()?.trim();
        if id.is_empty() {
            return None;
        }
        Some(id.to_string())
    }

    pub(super) fn persist(path: &Path, profile: &str, id: &str) -> std::io::Result<()> {
        // This is the writer that made the race ordinary rather than theoretical: it runs from
        // `ResolvedConfig::from_cli` on every start for a `claude-subscription` profile without a
        // device id, so a plain `meka` launch competes with whatever `meka mcp add` is
        // doing in the next terminal.
        let _lock = super::lock_config_file()?;
        let contents = std::fs::read_to_string(path).unwrap_or_default();
        let mut doc: toml_edit::DocumentMut = contents
            .parse()
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;

        // The active profile's table already exists (it's how the profile was selected); update its
        // `device_id` in place. Bail quietly rather than synthesize a malformed inline table if the
        // structure isn't what we expect.
        let Some(table) = doc
            .get_mut("providers")
            .and_then(|item| item.get_mut(profile))
            .and_then(|item| item.as_table_mut())
        else {
            return Ok(());
        };
        table.insert("device_id", toml_edit::value(id));

        write_file_atomic(path, &doc.to_string())
    }
}

/// The one lock serialising every test in this crate that mutates `MEKA_CONFIG_DIR` or
/// `MEKA_DATA_DIR`.
///
/// The var is process-global and unit tests share a process, so a per-module lock only serialises a
/// module against itself. Two of them are worse than none: while `src/skills/cli.rs` held its own
/// lock, a config test's `remove_var` could land mid-test and send `skills::cli::run_add` at the
/// developer's real `~/.config/meka`. A `tokio::sync::Mutex` because the skills tests are async and
/// hold the guard across `.await`; the synchronous tests here take it with `blocking_lock`.
#[cfg(test)]
pub(crate) static CONFIG_DIR_ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[cfg(test)]
mod tests {
    use super::*;

    /// A tilde is expanded whichever separator follows it.
    ///
    /// `~\projects` is what a Windows user types, because it is what PowerShell's own expansion
    /// produces. Matching a literal `"~/"` left it alone, so `/cd` reported the tilde back as a
    /// missing directory and `[skills] extra_paths` resolved to a root that never existed.
    #[test]
    fn expand_user_path_accepts_either_separator_after_the_tilde() {
        let home = dirs::home_dir().expect("a home directory");

        assert_eq!(expand_user_path("~"), Some(home.clone()));
        assert_eq!(expand_user_path(""), Some(home.clone()));
        assert_eq!(expand_user_path("~/projects"), Some(home.join("projects")));
        #[cfg(windows)]
        assert_eq!(expand_user_path(r"~\projects"), Some(home.join("projects")));

        // Not a tilde-prefixed path: `~foo` is a user reference meka deliberately does not expand,
        // and it must stay a literal rather than silently becoming `$HOME/foo`.
        assert_eq!(
            expand_user_path("~user/projects"),
            Some(PathBuf::from("~user/projects"))
        );
    }

    /// The webhook redactor keeps the scheme, host and port, and nothing else.
    ///
    /// Every case below is one a splitting implementation let through, and each reached a `warn!`
    /// at default verbosity. The query one is the sharpest: a webhook URL with no path has no `/`
    /// to split on, so `?token=…` was reproduced in full by the function whose entire job is to
    /// remove the secret.
    #[test]
    fn webhook_host_keeps_only_the_origin() {
        for (url, expected) in [
            (
                "https://hooks.example.com/services/T00/B00/XXXX",
                "https://hooks.example.com",
            ),
            // No path, so nothing for a `'/'` split to cut.
            (
                "https://hooks.example.com?token=SECRET",
                "https://hooks.example.com",
            ),
            (
                "https://hooks.example.com/path?token=SECRET",
                "https://hooks.example.com",
            ),
            (
                "https://hooks.example.com#SECRET",
                "https://hooks.example.com",
            ),
            // Userinfo rode along as part of what the old split called the host.
            (
                "https://user:pass@hooks.example.com/x",
                "https://hooks.example.com",
            ),
            // A port distinguishes two endpoints and is not a secret, so it is kept.
            ("http://127.0.0.1:8080/hook", "http://127.0.0.1:8080"),
            ("not a url", "<malformed>"),
        ] {
            let rendered = webhook_host(url);
            assert_eq!(rendered, expected, "redacting {url}");
            for secret in ["SECRET", "pass", "T00", "XXXX"] {
                assert!(
                    !rendered.contains(secret),
                    "{rendered} still carries `{secret}` from {url}"
                );
            }
        }
    }

    /// `extra_paths` resolution drops what discovery cannot use, and says so each time.
    ///
    /// A repeat, or meka's own root listed again, makes discovery walk the directory twice and then
    /// report every skill in it as shadowed *by itself* -- a warning naming one path twice, which
    /// an operator can neither act on nor dismiss. The empty entry is the dangerous one: it would
    /// expand to `$HOME` and make the whole home directory a skills root.
    #[test]
    fn extra_paths_drops_repeats_the_native_root_and_empty_entries() {
        let native = PathBuf::from("/cfg/skills");
        let resolved = resolve_skills_extra_paths(
            &[
                "/a/skills".to_string(),
                "/a/skills".to_string(),
                "/cfg/skills".to_string(),
                "  ".to_string(),
                "/b/skills".to_string(),
            ],
            Some(native.as_path()),
        );
        assert_eq!(
            resolved,
            vec![PathBuf::from("/a/skills"), PathBuf::from("/b/skills")],
            "order is precedence, so it has to be preserved"
        );

        // With no native root there is nothing to compare against, and the rest still applies.
        assert_eq!(
            resolve_skills_extra_paths(&["/a".to_string(), "/a".to_string()], None),
            vec![PathBuf::from("/a")]
        );
    }

    /// The redacting `Debug` impls exist to keep a secret out of a log, and until this test there
    /// was nothing asserting any of them: all three could have been deleted and the suite would
    /// have stayed green while `{:?}` started printing bearer tokens again.
    #[test]
    fn a_secret_never_reaches_a_debug_rendering() {
        let mut server = fixture_server("s");
        server.auth_token = Some("tok-AUTHTOKEN".to_string());
        server.headers = Some(std::collections::HashMap::from([(
            "Authorization".to_string(),
            "Bearer HEADERSECRET".to_string(),
        )]));
        server.env = Some(std::collections::HashMap::from([(
            "API_KEY".to_string(),
            "ENVSECRET".to_string(),
        )]));
        server.auth = Some(McpAuthConfig::ClientCredentials {
            client_id: "public-client-id".to_string(),
            client_secret: "CLIENTSECRET".to_string(),
            scopes: None,
            resource: None,
        });

        let rendered = format!("{:?}", server);
        for secret in ["tok-AUTHTOKEN", "HEADERSECRET", "ENVSECRET", "CLIENTSECRET"] {
            assert!(
                !rendered.contains(secret),
                "'{secret}' leaked into {rendered}"
            );
        }
        // The names around the secrets are the diagnostic, and must survive.
        assert!(rendered.contains("Authorization"), "{rendered}");
        assert!(rendered.contains("API_KEY"), "{rendered}");
        assert!(rendered.contains("public-client-id"), "{rendered}");
        assert!(rendered.contains("REDACTED"), "{rendered}");

        let oauth = McpAuthConfig::OAuth {
            client_id: Some("public-client-id".to_string()),
            client_secret: Some("OAUTHSECRET".to_string()),
            scopes: None,
            redirect_port: None,
        };
        let rendered = format!("{:?}", oauth);
        assert!(!rendered.contains("OAUTHSECRET"), "{rendered}");

        let serve_token = ResolvedServeToken {
            token: "SERVETOKEN".to_string(),
            description: None,
            scopes: Default::default(),
            source: TokenSource::Inline,
        };
        let rendered = format!("{:?}", serve_token);
        assert!(!rendered.contains("SERVETOKEN"), "{rendered}");

        // The *raw* config form too, and by the route that actually reaches a log: `ResolvedConfig`
        // derives `Debug` and owns the whole `[serve]` table, so redacting only the resolved token
        // left every configured bearer token one `{:?}` away. Asserting on the raw struct alone
        // would not have caught that; this walks the containment chain.
        let raw_token = ServeTokenConfig {
            token: Some("RAWSERVETOKEN".to_string()),
            token_file: None,
            description: Some("ci".to_string()),
            scopes: vec!["sessions:r".to_string()],
        };
        let rendered = format!("{:?}", raw_token);
        assert!(!rendered.contains("RAWSERVETOKEN"), "{rendered}");
        assert!(
            rendered.contains("ci"),
            "the label must survive: {rendered}"
        );

        let serve = ServeConfig {
            tokens: Some(vec![raw_token]),
            ..Default::default()
        };
        let rendered = format!("{:?}", serve);
        assert!(
            !rendered.contains("RAWSERVETOKEN"),
            "a token must not reach a log through the enclosing [serve] table: {rendered}"
        );

        let webhook = WebhookConfig {
            url: "https://example/hook".to_string(),
            secret: Some("WEBHOOKSECRET".to_string()),
            secret_file: None,
            events: Vec::new(),
            timeout: None,
            max_retries: None,
        };
        let rendered = format!("{:?}", webhook);
        assert!(!rendered.contains("WEBHOOKSECRET"), "{rendered}");
    }

    fn fixture_server(name: &str) -> McpServerConfig {
        McpServerConfig {
            name: name.to_string(),
            transport: McpTransport::Http,
            command: None,
            args: None,
            env: None,
            url: Some("https://example".to_string()),
            auth_token: None,
            headers: None,
            headers_helper: None,
            auth: None,
            permission: None,
            allowed_tools: None,
            disabled_tools: None,
            eager_load_tools: None,
            tool_permissions: None,
            trust_read_only_hint: None,
            disabled: false,
            required: None,
        }
    }

    #[test]
    fn test_eager_load_override_appends_to_matching_server() {
        let mut servers = vec![fixture_server("notion"), fixture_server("github")];
        let raw = vec![
            "notion:search".to_string(),
            "github:create_issue".to_string(),
        ];
        apply_cli_eager_load_overrides(&raw, &mut servers);

        assert_eq!(
            servers[0].eager_load_tools.as_deref(),
            Some(&["search".to_string()][..])
        );
        assert_eq!(
            servers[1].eager_load_tools.as_deref(),
            Some(&["create_issue".to_string()][..])
        );
    }

    #[test]
    fn test_eager_load_override_appends_to_existing_list() {
        let mut servers = vec![fixture_server("notion")];
        servers[0].eager_load_tools = Some(vec!["search".to_string()]);
        apply_cli_eager_load_overrides(&["notion:fetch".to_string()], &mut servers);
        assert_eq!(
            servers[0].eager_load_tools.as_deref(),
            Some(&["search".to_string(), "fetch".to_string()][..])
        );
    }

    #[test]
    fn test_eager_load_override_dedupes_existing_entry() {
        let mut servers = vec![fixture_server("notion")];
        servers[0].eager_load_tools = Some(vec!["search".to_string()]);
        apply_cli_eager_load_overrides(&["notion:search".to_string()], &mut servers);
        assert_eq!(
            servers[0].eager_load_tools.as_deref(),
            Some(&["search".to_string()][..]),
            "duplicate tool name must not double the list"
        );
    }

    #[test]
    fn test_eager_load_override_skips_unknown_server() {
        let mut servers = vec![fixture_server("notion")];
        apply_cli_eager_load_overrides(&["nope:search".to_string()], &mut servers);
        // The matching `notion` entry must remain untouched; the unknown `nope` entry simply
        // produces a warn log (not captured here).
        assert!(servers[0].eager_load_tools.is_none());
    }

    #[test]
    fn test_eager_load_override_skips_malformed_values() {
        let mut servers = vec![fixture_server("notion")];
        let raw = vec![
            "no-colon".to_string(),
            ":missing-server".to_string(),
            "missing-tool:".to_string(),
            "".to_string(),
        ];
        apply_cli_eager_load_overrides(&raw, &mut servers);
        assert!(servers[0].eager_load_tools.is_none());
    }

    /// `DisplayConfig` denies unknown fields, so a name or spelling that does not match is a
    /// startup error rather than a silently ignored line. Worth pinning both the key and the three
    /// values, since the config file is the only way to reach this setting.
    #[test]
    fn test_display_tool_params_parses_each_value() {
        for (written, expected) in [
            ("off", ToolParams::Off),
            ("summary", ToolParams::Summary),
            ("full", ToolParams::Full),
        ] {
            let toml_str = format!("[display]\ntool_params = \"{}\"\n", written);
            let config: ConfigFile = toml::from_str(&toml_str).expect("parse toml");
            let display = config.display.expect("display present");
            assert_eq!(display.tool_params, Some(expected));
        }
    }

    /// Omitting it leaves today's one-line indicator, so an existing config keeps its output.
    #[test]
    fn test_display_tool_params_defaults_to_summary() {
        assert_eq!(ToolParams::default(), ToolParams::Summary);
    }

    /// Deserializing `DisplayConfig` is not the same as the setting working: a key that parses but
    /// never reaches `ResolvedConfig` is a dead feature that every test on either side still
    /// passes. This is the one that fails if the resolution line is dropped.
    ///
    /// Unset means "follow the terminal", so the resolved value stays `None` rather than becoming a
    /// number that would then be honoured exactly and pin output to a guess.
    #[test]
    fn test_display_max_width_reaches_the_resolved_config() {
        assert_eq!(
            resolve_with_config("[display]\nmax_width = 120\n").max_width,
            Some(120)
        );
        assert_eq!(resolve_with_config("[display]\n").max_width, None);
    }

    /// Every budget subtracts fixed chrome first, so below the floor the subtraction leaves nothing
    /// and output degrades to punctuation. Above the ceiling the cost of composing a line grows
    /// quadratically and the renderer stalls. Clamped rather than rejected at both ends: a width
    /// stated badly is a preference, not a broken config.
    #[test]
    fn test_display_max_width_is_clamped_to_what_can_be_rendered() {
        assert_eq!(
            resolve_with_config("[display]\nmax_width = 5\n").max_width,
            Some(MIN_CONFIGURED_WIDTH)
        );
        assert_eq!(
            resolve_with_config(&format!(
                "[display]\nmax_width = {}\n",
                MIN_CONFIGURED_WIDTH
            ))
            .max_width,
            Some(MIN_CONFIGURED_WIDTH)
        );
        assert_eq!(
            resolve_with_config("[display]\nmax_width = 100000\n").max_width,
            Some(MAX_CONFIGURED_WIDTH)
        );
        assert_eq!(
            resolve_with_config("[display]\nmax_width = 120\n").max_width,
            Some(120)
        );
    }

    /// The same resolution check for `tool_params`, whose default is a value rather than an
    /// absence: unset must land on `summary`, not on whatever `ToolParams` derives.
    #[test]
    fn test_display_tool_params_reaches_the_resolved_config() {
        assert_eq!(
            resolve_with_config("[display]\ntool_params = \"full\"\n").tool_params,
            ToolParams::Full
        );
        assert_eq!(
            resolve_with_config("[display]\n").tool_params,
            ToolParams::Summary
        );
    }

    #[test]
    fn test_eager_load_override_trims_whitespace() {
        let mut servers = vec![fixture_server("notion")];
        apply_cli_eager_load_overrides(&["  notion : search  ".to_string()], &mut servers);
        assert_eq!(
            servers[0].eager_load_tools.as_deref(),
            Some(&["search".to_string()][..])
        );
    }

    #[test]
    fn test_web_config_all_fields_parse() {
        let toml_str = r#"
[web]
user_agent = "meka-test"
request_timeout_seconds = 60
connect_timeout_seconds = 5
read_timeout_seconds = 10
max_redirects = 3
proxy = "socks5h://127.0.0.1:1080"
ca_cert_file = "/etc/ssl/corp.pem"
https_only = true
min_tls_version = "1.3"
danger_accept_invalid_certs = true
danger_accept_invalid_hostnames = true
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("parse toml");
        let web = config.web.expect("web present");
        assert_eq!(web.user_agent.as_deref(), Some("meka-test"));
        assert_eq!(web.request_timeout_seconds, Some(60));
        assert_eq!(web.connect_timeout_seconds, Some(5));
        assert_eq!(web.read_timeout_seconds, Some(10));
        assert_eq!(web.max_redirects, Some(3));
        assert_eq!(web.proxy.as_deref(), Some("socks5h://127.0.0.1:1080"));
        assert_eq!(web.ca_cert_file.as_deref(), Some("/etc/ssl/corp.pem"));
        assert_eq!(web.https_only, Some(true));
        assert_eq!(web.min_tls_version.as_deref(), Some("1.3"));
        assert_eq!(web.danger_accept_invalid_certs, Some(true));
        assert_eq!(web.danger_accept_invalid_hostnames, Some(true));
    }

    #[test]
    fn test_web_client_config_defaults_from_empty_file() {
        // Empty [web] → sensible defaults; no user-surprising failures.
        let file = WebConfig::default();
        let cfg = WebClientConfig::from_file(&file);
        assert_eq!(cfg.user_agent, DEFAULT_WEB_USER_AGENT);
        assert_eq!(cfg.request_timeout, std::time::Duration::from_secs(30));
        assert!(cfg.connect_timeout.is_none());
        assert!(cfg.read_timeout.is_none());
        assert_eq!(cfg.max_redirects, 10);
        assert!(cfg.proxy.is_none());
        assert!(cfg.ca_cert_file.is_none());
        assert!(!cfg.https_only);
        assert!(cfg.min_tls_version.is_none());
        assert!(!cfg.danger_accept_invalid_certs);
        assert!(!cfg.danger_accept_invalid_hostnames);
    }

    #[test]
    fn test_web_client_config_resolves_full_file() {
        let file = WebConfig {
            user_agent: Some("ua".to_string()),
            request_timeout_seconds: Some(60),
            connect_timeout_seconds: Some(5),
            read_timeout_seconds: Some(10),
            max_redirects: Some(0),
            proxy: Some("http://proxy.local:8080".to_string()),
            ca_cert_file: Some("/tmp/ca.pem".to_string()),
            https_only: Some(true),
            min_tls_version: Some("1.3".to_string()),
            danger_accept_invalid_certs: Some(true),
            danger_accept_invalid_hostnames: Some(true),
        };
        let cfg = WebClientConfig::from_file(&file);
        assert_eq!(cfg.user_agent, "ua");
        assert_eq!(cfg.request_timeout, std::time::Duration::from_secs(60));
        assert_eq!(cfg.connect_timeout, Some(std::time::Duration::from_secs(5)));
        assert_eq!(cfg.read_timeout, Some(std::time::Duration::from_secs(10)));
        assert_eq!(cfg.max_redirects, 0);
        assert_eq!(cfg.proxy.as_deref(), Some("http://proxy.local:8080"));
        assert_eq!(
            cfg.ca_cert_file.as_deref(),
            Some(std::path::Path::new("/tmp/ca.pem"))
        );
        assert!(cfg.https_only);
        assert_eq!(cfg.min_tls_version, Some(MinTlsVersion::V1_3));
        assert!(cfg.danger_accept_invalid_certs);
        assert!(cfg.danger_accept_invalid_hostnames);
    }

    #[test]
    fn test_min_tls_version_parse_accepts_all_valid() {
        assert_eq!(MinTlsVersion::parse("1.0"), Some(MinTlsVersion::V1_0));
        assert_eq!(MinTlsVersion::parse("1.1"), Some(MinTlsVersion::V1_1));
        assert_eq!(MinTlsVersion::parse("1.2"), Some(MinTlsVersion::V1_2));
        assert_eq!(MinTlsVersion::parse("1.3"), Some(MinTlsVersion::V1_3));
        // Whitespace trimming.
        assert_eq!(MinTlsVersion::parse("  1.2  "), Some(MinTlsVersion::V1_2));
    }

    #[test]
    fn test_min_tls_version_parse_rejects_invalid() {
        assert!(MinTlsVersion::parse("1.5").is_none());
        assert!(MinTlsVersion::parse("tls1.3").is_none());
        assert!(MinTlsVersion::parse("").is_none());
    }

    #[test]
    fn test_web_client_config_rejects_bad_min_tls_falls_back() {
        // Invalid min_tls_version string logs a warn but doesn't abort; we fall through to
        // reqwest's default rather than failing startup on a typo.
        let file = WebConfig {
            min_tls_version: Some("1.5".to_string()),
            ..WebConfig::default()
        };
        let cfg = WebClientConfig::from_file(&file);
        assert!(cfg.min_tls_version.is_none());
    }

    #[test]
    fn test_web_client_config_zero_timeout_uses_default() {
        // `0` in the config is treated as "fall through to default" so users can't accidentally set
        // request_timeout = 0 and disable timeouts entirely.
        let file = WebConfig {
            request_timeout_seconds: Some(0),
            connect_timeout_seconds: Some(0),
            read_timeout_seconds: Some(0),
            ..WebConfig::default()
        };
        let cfg = WebClientConfig::from_file(&file);
        assert_eq!(cfg.request_timeout, std::time::Duration::from_secs(30));
        assert!(cfg.connect_timeout.is_none());
        assert!(cfg.read_timeout.is_none());
    }

    #[test]
    fn test_mcp_runtime_fields_parse() {
        let toml_str = r#"
[mcp]
default_permission = "read"
strict = false
grace_seconds = 5
connect_timeout_seconds = 60
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("parse toml");
        let mcp = config.mcp.expect("mcp present");
        assert_eq!(mcp.strict, Some(false));
        assert_eq!(mcp.grace_seconds, Some(5));
        assert_eq!(mcp.connect_timeout_seconds, Some(60));
    }

    #[test]
    fn test_mcp_server_disabled_parses() {
        let toml_str = r#"
[[mcp.servers]]
name = "flaky"
transport = "stdio"
command = "npx"
disabled = true
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("parse toml");
        let servers = config.mcp.unwrap().servers.unwrap();
        assert_eq!(servers.len(), 1);
        assert!(servers[0].disabled);
    }

    #[test]
    fn test_mcp_server_disabled_defaults_false() {
        let toml_str = r#"
[[mcp.servers]]
name = "normal"
transport = "stdio"
command = "npx"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("parse toml");
        let servers = config.mcp.unwrap().servers.unwrap();
        assert!(!servers[0].disabled);
    }

    #[test]
    fn test_config_file_deserialization() {
        let toml_str = r#"
default_provider = "work"

[providers.work]
type = "openai-chat-completions"
model = "gpt-4o"
base_url = "https://api.openai.com/v1"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        assert_eq!(config.default_provider.as_deref(), Some("work"));
        let profile = config
            .providers
            .get("work")
            .expect("profile should be present");
        assert_eq!(profile.backend, "openai-chat-completions");
        assert_eq!(profile.model.as_deref(), Some("gpt-4o"));
        assert_eq!(
            profile.base_url.as_deref(),
            Some("https://api.openai.com/v1")
        );
    }

    #[test]
    fn test_empty_config_file() {
        let config: ConfigFile = toml::from_str("").expect("failed to parse empty toml");
        assert!(config.providers.is_empty());
        assert!(config.default_provider.is_none());
    }

    #[test]
    fn test_partial_config_file() {
        let toml_str = r#"
[providers.main]
type = "claude-subscription"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let profile = config
            .providers
            .get("main")
            .expect("profile should be present");
        assert_eq!(profile.backend, "claude-subscription");
        assert!(profile.model.is_none());
        assert!(profile.base_url.is_none());
    }

    fn profiles_from(
        backends: &[(&str, &str)],
    ) -> std::collections::BTreeMap<String, ProviderProfile> {
        backends
            .iter()
            .map(|(name, backend)| {
                (name.to_string(), ProviderProfile {
                    backend: backend.to_string(),
                    ..Default::default()
                })
            })
            .collect()
    }

    #[test]
    fn test_select_active_profile_explicit_request_wins() {
        let providers = profiles_from(&[
            ("work", "claude-subscription"),
            ("personal", "openai-chat-completions"),
        ]);
        let (active, error) = select_active_profile(Some("personal".to_string()), &providers);
        assert_eq!(active.as_deref(), Some("personal"));
        assert!(error.is_none());
    }

    #[test]
    fn test_select_active_profile_sole_profile_when_no_request() {
        let providers = profiles_from(&[("only", "anthropic-messages")]);
        let (active, error) = select_active_profile(None, &providers);
        assert_eq!(active.as_deref(), Some("only"));
        assert!(error.is_none());
    }

    #[test]
    fn test_select_active_profile_zero_profiles_errors() {
        let providers = profiles_from(&[]);
        let (active, error) = select_active_profile(None, &providers);
        assert!(active.is_none());
        assert!(error.expect("error expected").contains("provider add"));
    }

    #[test]
    fn test_select_active_profile_ambiguous_without_default_errors() {
        let providers = profiles_from(&[
            ("work", "claude-subscription"),
            ("personal", "openai-chat-completions"),
        ]);
        let (active, error) = select_active_profile(None, &providers);
        assert!(active.is_none());
        let error = error.expect("error expected");
        assert!(error.contains("multiple provider profiles"));
        // The error lists the configured names so the user knows what to pick.
        assert!(error.contains("work") && error.contains("personal"));
    }

    #[test]
    fn test_select_active_profile_unknown_name_errors() {
        let providers = profiles_from(&[("work", "claude-subscription")]);
        let (active, error) = select_active_profile(Some("missing".to_string()), &providers);
        assert!(active.is_none());
        assert!(
            error
                .expect("error expected")
                .contains("no provider profile named 'missing'")
        );
    }

    #[test]
    fn a_device_id_is_empty_for_every_backend_but_claude_subscription() {
        // Should not generate / persist anything when the provider doesn't need a device_id. Empty
        // string flows through but is ignored by non-claude-subscription providers.
        assert_eq!(
            device_id::resolve(Some("openai-chat-completions"), Some("work"), None),
            ""
        );
        assert_eq!(
            device_id::resolve(Some("anthropic-messages"), Some("work"), None),
            ""
        );
        assert_eq!(device_id::resolve(None, None, None), "");
        // Even an explicit configured value is suppressed when the provider isn't
        // claude-subscription; the field is provider-scoped.
        assert_eq!(
            device_id::resolve(
                Some("openai-chat-completions"),
                Some("work"),
                Some("explicit")
            ),
            ""
        );
    }

    #[test]
    fn test_resolve_device_id_uses_configured_value_for_claude_oauth() {
        // A configured value returns before the persist branch, so this never touches the FS.
        let id = "deadbeef".repeat(8);
        assert_eq!(
            device_id::resolve(Some("claude-subscription"), Some("work"), Some(&id)),
            id,
            "configured value must be used verbatim for claude-subscription"
        );
    }

    #[test]
    fn test_persist_device_id_writes_into_profile_table() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "default_provider = \"work\"\n\n[providers.work]\ntype = \"claude-subscription\"\nmodel = \"claude-opus-4-8\"\n",
        )
        .expect("seed config");

        device_id::persist(&path, "work", "abc123").expect("persist");

        let contents = std::fs::read_to_string(&path).expect("read back");
        let config: ConfigFile = toml::from_str(&contents).expect("re-parse");
        let persisted = config
            .providers
            .get("work")
            .and_then(|profile| profile.device_id.as_deref());
        assert_eq!(
            persisted,
            Some("abc123"),
            "device_id must be stored under the active profile"
        );
        // Existing profile fields are preserved.
        assert_eq!(
            config.providers.get("work").map(|p| p.backend.as_str()),
            Some("claude-subscription")
        );
        // Closing the loop: the reader feeds the persisted value back through `resolve`, which must
        // return it verbatim rather than regenerate a fresh id on the next run.
        assert_eq!(
            device_id::resolve(Some("claude-subscription"), Some("work"), persisted),
            "abc123",
            "a persisted device_id must be picked up on the next run, not regenerated"
        );
    }

    #[test]
    fn test_read_user_id_from_valid_claude_json() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("claude.json");
        std::fs::write(
            &path,
            r#"{"userID": "af5986c7cb3b5e8d00eaf3da3b81730c6f523b1e68e1720c7128a96167534be3", "other": "stuff"}"#,
        )
        .expect("write");
        assert_eq!(
            device_id::read_user_id_from(&path).as_deref(),
            Some("af5986c7cb3b5e8d00eaf3da3b81730c6f523b1e68e1720c7128a96167534be3")
        );
    }

    #[test]
    fn test_read_user_id_from_missing_file_returns_none() {
        let path = std::path::Path::new("/nonexistent/path/claude.json");
        assert!(device_id::read_user_id_from(path).is_none());
    }

    #[test]
    fn test_read_user_id_from_malformed_json_returns_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("claude.json");
        std::fs::write(&path, "{not valid json").expect("write");
        assert!(device_id::read_user_id_from(&path).is_none());
    }

    #[test]
    fn test_read_user_id_from_missing_field_returns_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("claude.json");
        std::fs::write(&path, r#"{"foo": "bar"}"#).expect("write");
        assert!(device_id::read_user_id_from(&path).is_none());
    }

    #[test]
    fn test_read_user_id_from_empty_string_returns_none() {
        // An empty `userID` in claude.json shouldn't override meka's own random-generation
        // fallback; treat it as "not configured".
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("claude.json");
        std::fs::write(&path, r#"{"userID": ""}"#).expect("write");
        assert!(device_id::read_user_id_from(&path).is_none());
    }

    #[test]
    fn test_read_user_id_from_whitespace_only_returns_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("claude.json");
        std::fs::write(&path, r#"{"userID": "   "}"#).expect("write");
        assert!(device_id::read_user_id_from(&path).is_none());
    }

    #[test]
    fn test_read_user_id_from_non_string_returns_none() {
        // A non-string `userID` (number, object, …) shouldn't crash; just decline to use the
        // value.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("claude.json");
        std::fs::write(&path, r#"{"userID": 12345}"#).expect("write");
        assert!(device_id::read_user_id_from(&path).is_none());
    }

    #[test]
    fn test_read_user_id_from_trims_surrounding_whitespace() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("claude.json");
        std::fs::write(&path, r#"{"userID": "  abcdef123  "}"#).expect("write");
        assert_eq!(
            device_id::read_user_id_from(&path).as_deref(),
            Some("abcdef123")
        );
    }

    #[test]
    fn test_provider_profile_deserializes_effort_and_redact_thinking() {
        let toml_str = r#"
[providers.work]
type = "claude-subscription"
model = "claude-opus-4-6-20250514"
effort = "medium"
thinking = "budgeted"
redact_thinking = true
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let profile = config
            .providers
            .get("work")
            .expect("profile should be present");
        assert_eq!(profile.effort.as_deref(), Some("medium"));
        assert_eq!(
            profile.thinking,
            Some(crate::provider::ThinkingMode::Budgeted)
        );
        assert_eq!(profile.redact_thinking, Some(true));
    }

    #[test]
    fn the_thinking_block_parses_its_keys() {
        let config: ConfigFile =
            toml::from_str("[thinking]\nbudget_tokens = 10000\nshow_content = true\n")
                .expect("parse");
        let thinking = config.thinking.expect("[thinking] present");
        assert_eq!(thinking.budget_tokens, Some(10_000));
        assert_eq!(thinking.show_content, Some(true));
    }

    #[test]
    fn test_unknown_config_keys_are_rejected() {
        // `deny_unknown_fields`: an unrecognised key errors at load instead of silently doing
        // nothing. An unknown key inside a provider profile (here `reasoning_effort`, which meka
        // does not accept because the provider owns that default).
        let stale_profile = r#"
[providers.work]
type = "chatgpt-subscription"
model = "gpt-5.6-sol"
reasoning_effort = "high"
"#;
        let error = toml::from_str::<ConfigFile>(stale_profile)
            .expect_err("unknown profile key must be rejected")
            .to_string();
        assert!(
            error.contains("reasoning_effort") || error.contains("unknown field"),
            "{error}"
        );
        // A typo'd key inside `[session]`.
        assert!(toml::from_str::<ConfigFile>("[session]\ncontex_window = 1000").is_err());
        // A typo'd top-level section.
        assert!(toml::from_str::<ConfigFile>("[sesion]\nfoo = 1").is_err());
        // The corrected config still parses.
        assert!(toml::from_str::<ConfigFile>("[session]\ncontext_window = 1000000").is_ok());
    }

    #[test]
    fn test_provider_profile_deserializes_capability_knobs() {
        let toml_str = r#"
[providers.work]
type = "openai-chat-completions"
model = "gpt-5.5"
context_window = 1000000
vision = false
max_output_tokens = 64000
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let profile = config
            .providers
            .get("work")
            .expect("profile should be present");
        assert_eq!(profile.context_window, Some(1_000_000));
        assert_eq!(profile.vision, Some(false));
        assert_eq!(profile.max_output_tokens, Some(64_000));
    }

    #[test]
    fn the_budget_invariant_is_checked_only_under_budgeted_thinking() {
        use crate::provider::ThinkingMode;

        // Budgeted draws the thinking budget out of `max_tokens`, so a cap at or below it can only
        // produce a 400. Catch it at config load, where the message can say what to change.
        //
        // Both Anthropic Messages backends, because the invariant belongs to the protocol rather
        // than to the account: dropping either from the gate is invisible until a real request is
        // rejected mid-turn, which is exactly what this check exists to pre-empt.
        for backend in ["claude-subscription", "anthropic-messages"] {
            let result = validate_max_output_tokens(
                Some(backend),
                Some(5_000),
                ThinkingMode::Budgeted,
                10_000,
            );
            assert!(result.is_err(), "{backend}");
            assert!(
                result.unwrap_err().to_string().contains("thinking budget"),
                "{backend}: the error must name the budget it conflicts with"
            );
        }
        assert!(
            validate_max_output_tokens(
                Some("claude-subscription"),
                Some(20_000),
                ThinkingMode::Budgeted,
                10_000
            )
            .is_ok()
        );

        // The other two modes send no `budget_tokens` at all, so the same cap is fine. Reading the
        // mode rather than guessing it from the model name is the point: a budget-free request is
        // no longer inferred, it is stated.
        for mode in [ThinkingMode::Adaptive, ThinkingMode::Off] {
            assert!(
                validate_max_output_tokens(Some("claude-subscription"), Some(5_000), mode, 10_000)
                    .is_ok(),
                "{mode:?}"
            );
        }
        // Non-Claude backends have no such constraint either, whatever the mode.
        assert!(
            validate_max_output_tokens(
                Some("openai-chat-completions"),
                Some(100),
                ThinkingMode::Budgeted,
                10_000
            )
            .is_ok()
        );
        // No override at all.
        assert!(
            validate_max_output_tokens(
                Some("anthropic-messages"),
                None,
                ThinkingMode::Budgeted,
                10_000
            )
            .is_ok()
        );
    }

    #[test]
    fn test_provider_profile_capability_knobs_default_to_none() {
        let toml_str = r#"
[providers.work]
type = "openai-chat-completions"
model = "gpt-5.5"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let profile = config.providers.get("work").expect("profile present");
        assert_eq!(profile.context_window, None);
        assert_eq!(profile.vision, None);
        assert_eq!(profile.max_output_tokens, None);
    }

    /// Deserialization only: `required` reaches the per-server config, and an omitted one stays
    /// `None` so resolution can seed it from `strict`. What that resolution then produces is
    /// covered by `test_strict_seeds_required_and_per_server_wins`.
    #[test]
    fn test_mcp_required_deserialization() {
        let config: ConfigFile = toml::from_str(
            r#"
[mcp]
strict = true

[[mcp.servers]]
name = "ida"
transport = "stdio"
command = "ida-mcp"
required = false

[[mcp.servers]]
name = "bridge"
transport = "http"
url = "http://127.0.0.1:9100/mcp"
"#,
        )
        .expect("parse");
        let mcp = config.mcp.expect("mcp table");
        assert_eq!(mcp.strict, Some(true));
        let servers = mcp.servers.expect("servers");
        assert_eq!(servers[0].required, Some(false), "explicit opt-out");
        assert_eq!(servers[1].required, None, "inherits strict at resolve time");
    }

    #[test]
    fn test_memory_and_skills_config_deserialization() {
        let config: ConfigFile = toml::from_str(
            r#"
[memory]
enabled = false

[skills]
enabled = true
"#,
        )
        .expect("failed to parse toml");
        assert_eq!(
            config.memory.expect("memory table present").enabled,
            Some(false)
        );
        assert_eq!(
            config.skills.expect("skills table present").enabled,
            Some(true)
        );
    }

    #[test]
    fn test_schedule_config_deserialization() {
        let config: ConfigFile = toml::from_str(
            r#"
[schedule]
enabled = true
poll_interval = "5s"
missed_grace = "2h"
gate_timeout = "45s"
max_jobs = 10
max_consecutive_fires = 3
"#,
        )
        .expect("failed to parse toml");
        let schedule = ResolvedScheduleConfig::resolve(
            config.schedule,
            crate::permission::Permission::Unrestricted,
            crate::permission::EnabledPermissions::DEFAULT,
        );
        assert!(schedule.enabled);
        assert_eq!(schedule.poll_interval, std::time::Duration::from_secs(5));
        assert_eq!(schedule.missed_grace, std::time::Duration::from_secs(7200));
        assert_eq!(schedule.gate_timeout, std::time::Duration::from_secs(45));
        assert_eq!(schedule.max_jobs, 10);
        assert_eq!(schedule.max_consecutive_fires, 3);
    }

    #[test]
    fn test_schedule_config_defaults_are_filled_when_absent() {
        let config: ConfigFile = toml::from_str("").expect("empty config parses");
        assert!(config.schedule.is_none());
        let schedule = ResolvedScheduleConfig::resolve(
            config.schedule,
            crate::permission::Permission::Unrestricted,
            crate::permission::EnabledPermissions::DEFAULT,
        );
        assert!(schedule.enabled, "scheduling is on unless turned off");
        assert!(!schedule.poll_interval.is_zero());
        assert!(!schedule.gate_timeout.is_zero());
        assert!(schedule.max_jobs > 0);
        // Pinned to the value rather than to non-zero: both doc pages state 5, and drifting either
        // way is silent -- large disables the interleaving, 1 serialises every session.
        assert_eq!(schedule.max_consecutive_fires, 5);
    }

    #[test]
    fn test_background_config_deserialization() {
        let config: ConfigFile = toml::from_str(
            "[background]
enabled = true
max_tasks = 3
",
        )
        .expect("config parses");
        let background = ResolvedBackgroundConfig::resolve(config.background);
        assert!(background.enabled);
        assert_eq!(background.max_tasks, 3);
    }

    #[test]
    fn test_subagents_config_deserialization() {
        let config: ConfigFile = toml::from_str(
            r#"[subagents]
disabled_servers = ["mekabridge"]
disabled_tools = ["mcp__notion__create_page", "write_file"]
"#,
        )
        .expect("config parses");
        let subagents = ResolvedSubagentsConfig::resolve(config.subagents);
        assert_eq!(subagents.disabled_servers, vec!["mekabridge".to_string()]);
        assert_eq!(subagents.disabled_tools, vec![
            "mcp__notion__create_page".to_string(),
            "write_file".to_string()
        ]);
    }

    /// Absent `[subagents]` denies nothing. What a worker *receives* is not configured at all: it
    /// is granted per `agent_spawn` call and defaults to nothing.
    #[test]
    fn test_subagents_defaults_deny_nothing() {
        let config: ConfigFile = toml::from_str("").expect("empty config parses");
        assert!(config.subagents.is_none());
        let subagents = ResolvedSubagentsConfig::resolve(config.subagents);
        assert!(subagents.disabled_servers.is_empty());
        assert!(subagents.disabled_tools.is_empty());
    }

    /// `[subagents]` deliberately has no `memory` key: config withholds capabilities, and the
    /// memory store is context a parent can copy into a prompt regardless. Setting one must be an
    /// error rather than silently ignored.
    #[test]
    fn test_subagents_has_no_memory_key() {
        let error = toml::from_str::<ConfigFile>("[subagents]\nmemory = \"none\"\n")
            .expect_err("an unrecognised key must not be silently ignored");
        assert!(error.to_string().contains("memory"), "{error}");
    }

    /// Sub-agents can be granted read access but never write: a worker recording what it inferred
    /// from one narrow task puts it in front of every future turn.
    #[test]
    fn test_memory_grant_parsing_refuses_write() {
        assert_eq!(MemoryAccess::parse_grant("none"), Ok(MemoryAccess::None));
        assert_eq!(MemoryAccess::parse_grant("  READ "), Ok(MemoryAccess::Read));
        let error = MemoryAccess::parse_grant("write").expect_err("write is not grantable");
        assert!(error.contains("not available to sub-agents"), "{error}");
        assert!(error.contains("yourself"), "and says what to do instead");
        assert!(MemoryAccess::parse_grant("readonly").is_err());
    }

    #[test]
    fn test_instruction_grant_parsing() {
        assert_eq!(
            InstructionAccess::parse_grant("inherit"),
            Ok(InstructionAccess::Inherit)
        );
        assert_eq!(
            InstructionAccess::parse_grant("None"),
            Ok(InstructionAccess::None)
        );
        assert!(InstructionAccess::parse_grant("yes").is_err());
        assert_eq!(InstructionAccess::default(), InstructionAccess::None);
    }

    /// `min` is "the more restrictive of two", which is what clamps a grant against what the
    /// granting agent itself holds.
    #[test]
    fn test_memory_access_orders_restrictive_first() {
        assert!(MemoryAccess::None < MemoryAccess::Read);
        assert!(MemoryAccess::Read < MemoryAccess::Write);
        assert_eq!(
            MemoryAccess::Read.min(MemoryAccess::None),
            MemoryAccess::None
        );
        assert_eq!(
            InstructionAccess::Inherit.min(InstructionAccess::None),
            InstructionAccess::None
        );
    }

    /// The one capability block that is off unless asked for. See [`BackgroundConfig`] for why it
    /// departs from `[schedule]` / `[skills]` / `[memory]`.
    #[test]
    fn test_background_is_off_by_default() {
        let config: ConfigFile = toml::from_str("").expect("empty config parses");
        assert!(config.background.is_none());
        let background = ResolvedBackgroundConfig::resolve(config.background);
        assert!(
            !background.enabled,
            "background tasks are off unless turned on"
        );
        assert!(background.max_tasks > 0);
    }

    #[test]
    fn test_background_config_rejects_unknown_keys() {
        assert!(
            toml::from_str::<ConfigFile>(
                "[background]
detach = true
"
            )
            .is_err()
        );
    }

    #[test]
    fn test_schedule_config_rejects_unknown_keys() {
        assert!(
            toml::from_str::<ConfigFile>(
                "[schedule]
on_exit = \"make build\"
"
            )
            .is_err()
        );
    }

    /// Absent tables must stay absent rather than deserialising to something opinionated; the
    /// default-on decision lives in `from_cli`'s `unwrap_or(true)`, not in the parsed shape.
    #[test]
    fn test_memory_and_skills_absent_by_default() {
        let config: ConfigFile = toml::from_str("").expect("empty config parses");
        assert!(config.memory.is_none());
        assert!(config.skills.is_none());
        assert!(MemoryConfig::default().enabled.is_none());
        assert!(SkillsConfig::default().enabled.is_none());
    }

    /// Both tables are `deny_unknown_fields`, so a typo is a startup error rather than a setting
    /// that silently does nothing.
    #[test]
    fn test_memory_and_skills_reject_unknown_keys() {
        assert!(
            toml::from_str::<ConfigFile>(
                "[memory]
enabeld = false
"
            )
            .is_err()
        );
        assert!(
            toml::from_str::<ConfigFile>(
                "[skills]
path = \"/tmp\"
"
            )
            .is_err()
        );
    }

    #[test]
    fn test_session_config_deserialization() {
        let toml_str = r#"
[session]
context_messages = 100
retention_days = 90
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let session = config.session.expect("session should be present");
        assert_eq!(session.context_messages, Some(100));
        assert_eq!(session.retention_days, Some(90));
    }

    /// Builds a `ResolvedConfig` from a real config file, so the resolution steps in `from_cli`
    /// are exercised rather than re-implemented in the test.
    fn resolve_with_config(body: &str) -> ResolvedConfig {
        use clap::Parser;
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.toml"), body).expect("write config");
        // SAFETY: `MEKA_CONFIG_DIR` is process-global; `CONFIG_DIR_ENV_LOCK` serialises every test
        // that touches it, and the guard is held across the whole set → resolve → clear
        // cycle.
        let _guard = CONFIG_DIR_ENV_LOCK.blocking_lock();
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let resolved = ResolvedConfig::from_cli(&crate::cli::Cli::parse_from(["meka"]));
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
        resolved
    }

    /// The registry gating is tested in `crate::tools`, but nothing joined the two: a key that
    /// never reaches `ResolvedConfig` leaves the tools correctly gated on a flag that is always
    /// false, and every test on either side still passes.
    #[test]
    fn test_skills_agent_managed_resolves_from_the_config_file() {
        let resolved = resolve_with_config("");
        assert!(resolved.skills_enabled, "skills default on");
        assert!(
            !resolved.skills_agent_managed,
            "authoring must be off unless asked for"
        );

        let resolved = resolve_with_config(
            r#"
[skills]
agent_managed = true
"#,
        );
        assert!(resolved.skills_agent_managed);
        assert!(
            resolved.skills_enabled,
            "agent_managed alone must not disturb the enabled default"
        );
    }

    /// `strict` is only a default for `required`; every consumer downstream reads the per-server
    /// flag, so if this resolution stopped happening `strict = true` would silently stop gating.
    #[test]
    fn test_strict_seeds_required_and_per_server_wins() {
        let resolved = resolve_with_config(
            r#"
[mcp]
strict = true

[[mcp.servers]]
name = "gates"
transport = "http"
url = "http://localhost/mcp"

[[mcp.servers]]
name = "optout"
transport = "http"
url = "http://localhost/mcp"
required = false
"#,
        );
        let by_name = |n: &str| {
            resolved
                .mcp_servers
                .iter()
                .find(|s| s.name == n)
                .unwrap_or_else(|| panic!("{n} present"))
                .required
        };
        assert_eq!(by_name("gates"), Some(true), "inherits strict");
        assert_eq!(by_name("optout"), Some(false), "explicit opt-out wins");
    }

    /// The default is off: an unavailable server degrades the session instead of stopping it.
    #[test]
    fn test_servers_are_optional_without_strict() {
        let resolved = resolve_with_config(
            r#"
[[mcp.servers]]
name = "ida"
transport = "stdio"
command = "ida-mcp"
"#,
        );
        assert_eq!(resolved.mcp_servers[0].required, Some(false));
        assert!(resolved.retention_days.is_none(), "no cleanup by default");
    }

    /// Zero is "delete everything, every startup". Refuse rather than discover it after the fact.
    /// `context_messages = 0` would send an empty conversation, which no provider accepts, and the
    /// assembly path indexes on the assumption that the window is non-empty. Rejecting it at load
    /// is what keeps that from becoming a panic mid-turn; nothing asserted the rejection.
    #[test]
    fn test_context_messages_zero_is_rejected() {
        let resolved = resolve_with_config(
            r#"
default_provider = "p"

[providers.p]
type = "openai-chat-completions"
model = "m"

[session]
context_messages = 0
"#,
        );
        assert_eq!(
            resolved.context_messages,
            Some(0),
            "fixture must actually set it"
        );
        let error = resolved
            .validate()
            .expect_err("context_messages = 0 must not be accepted");
        assert!(error.to_string().contains("context_messages"), "{error}");
    }

    #[test]
    fn test_retention_days_zero_is_rejected() {
        // Needs a usable provider: `validate` reports a missing one first, and rightly so - it
        // blocks the run outright, where retention only bites at the next startup sweep.
        let resolved = resolve_with_config(
            r#"
default_provider = "p"

[providers.p]
type = "openai-chat-completions"
model = "m"

[session]
retention_days = 0
"#,
        );
        assert_eq!(
            resolved.retention_days,
            Some(0),
            "fixture must actually set it"
        );
        let error = resolved
            .validate()
            .expect_err("retention_days = 0 must not be accepted");
        assert!(error.to_string().contains("retention_days"), "{error}");
    }

    /// `tokio::time::interval` panics on a zero period, so without this check a config value takes
    /// the whole process down at scheduler startup rather than merely behaving oddly.
    #[test]
    fn test_schedule_poll_interval_zero_is_rejected() {
        let resolved = resolve_with_config(
            r#"
default_provider = "p"

[providers.p]
type = "openai-chat-completions"
model = "m"

[schedule]
poll_interval = "0s"
"#,
        );
        assert!(
            resolved.schedule.poll_interval.is_zero(),
            "fixture must actually set it"
        );
        let error = resolved
            .validate()
            .expect_err("a zero poll interval must not be accepted");
        assert!(error.to_string().contains("poll_interval"), "{error}");
    }

    #[test]
    fn test_schedule_gate_timeout_and_max_jobs_zero_are_rejected() {
        for (key, value, needle) in [
            ("gate_timeout", "\"0s\"", "gate_timeout"),
            ("max_jobs", "0", "max_jobs"),
            ("max_consecutive_fires", "0", "max_consecutive_fires"),
        ] {
            let resolved = resolve_with_config(&format!(
                r#"
default_provider = "p"

[providers.p]
type = "openai-chat-completions"
model = "m"

[schedule]
{key} = {value}
"#
            ));
            let error = resolved
                .validate()
                .expect_err("a zero value must not be accepted");
            assert!(error.to_string().contains(needle), "{key}: {error}");
        }
    }

    /// A config that doesn't parse must stop meka, not be swapped for defaults. The old fallback
    /// meant one mistyped key silently ran the agent with no provider profiles, no MCP servers and
    /// default permissions, off one warn line among the rest of startup.
    #[test]
    fn test_unparseable_config_is_reported_not_ignored() {
        // `load_config_file` carries the parse failure; `validate` is what turns it into an exit.
        let (file, error) = {
            let dir = tempfile::tempdir().expect("tempdir");
            std::fs::write(dir.path().join("config.toml"), "[session]\nnot_a_key = 1\n")
                .expect("write config");
            // SAFETY: `MEKA_CONFIG_DIR` is process-global; `CONFIG_DIR_ENV_LOCK` serialises every
            // test that touches it, and the guard is held across the whole set → read →
            // clear cycle.
            let _guard = CONFIG_DIR_ENV_LOCK.blocking_lock();
            unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
            let loaded = load_config_file();
            unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
            loaded
        };
        assert!(
            file.session.is_none(),
            "a failed parse yields defaults, not partial config"
        );
        let error = error.expect("the parse failure must be reported");
        assert!(error.contains("not_a_key"), "{error}");
    }

    /// Carrying the parse failure is only half of it: `validate` is what turns it into an exit.
    /// Without this, deleting the `config_error` arm leaves the whole suite green while meka goes
    /// back to silently running the agent on defaults.
    #[test]
    fn test_validate_rejects_an_unparseable_config() {
        let resolved = resolve_with_config("[session]\nnot_a_key = 1\n");
        let error = resolved
            .validate()
            .expect_err("an unparseable config must not start the agent");
        assert!(error.to_string().contains("not_a_key"), "{error}");
        // Ahead of the provider check: this fixture has no profiles either, and "no provider
        // configured" would send the user chasing the wrong problem.
        assert!(resolved.provider_error.is_some(), "fixture has no profiles");
    }

    /// Session cleanup is driven by age alone, so `max_storage_bytes` is not a knob. It must be
    /// rejected outright rather than parsed and ignored, which would leave a config that asks for
    /// a size cap looking like it got one.
    #[test]
    fn test_size_based_cleanup_is_not_configurable() {
        let error = toml::from_str::<ConfigFile>("[session]\nmax_storage_bytes = 52428800\n")
            .expect_err("the key must not parse");
        assert!(error.to_string().contains("max_storage_bytes"), "{error}");
    }

    #[test]
    fn test_session_config_partial() {
        let toml_str = r#"
[session]
context_messages = 50
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let session = config.session.expect("session should be present");
        assert_eq!(session.context_messages, Some(50));
        assert!(session.retention_days.is_none());
    }

    #[test]
    fn test_session_defaults_applied() {
        let file_session = SessionConfig::default();
        let context_messages = file_session.context_messages.or(Some(200));
        let subagent_max_depth = file_session.subagent_max_depth.unwrap_or(3);

        assert_eq!(context_messages, Some(200));
        assert_eq!(subagent_max_depth, 3);
        // Retention has no default: unset means keep every session forever.
        assert!(file_session.retention_days.is_none());
    }

    #[test]
    fn test_mcp_config_deserialization() {
        let toml_str = r#"
[[mcp.servers]]
name = "postgres"
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-postgres"]
permission = "read"

[[mcp.servers]]
name = "web-api"
transport = "http"
url = "http://localhost:8080/mcp"
permission = "workspace"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let mcp = config.mcp.expect("mcp should be present");
        let servers = mcp.servers.expect("servers should be present");
        assert_eq!(servers.len(), 2);
        assert_eq!(servers[0].name, "postgres");
        assert_eq!(servers[0].transport, McpTransport::Stdio);
        assert_eq!(servers[0].command.as_deref(), Some("npx"));
        assert_eq!(
            servers[0].args.as_deref(),
            Some(
                ["-y", "@modelcontextprotocol/server-postgres"]
                    .map(String::from)
                    .as_slice()
            )
        );
        assert_eq!(servers[0].permission.as_deref(), Some("read"));
        assert_eq!(servers[1].name, "web-api");
        assert_eq!(servers[1].transport, McpTransport::Http);
        assert_eq!(servers[1].url.as_deref(), Some("http://localhost:8080/mcp"));
        assert_eq!(servers[1].permission.as_deref(), Some("workspace"));
    }

    #[test]
    fn test_mcp_config_empty() {
        let config: ConfigFile = toml::from_str("").expect("failed to parse empty toml");
        assert!(config.mcp.is_none());
    }

    #[test]
    fn test_session_config_overrides_defaults() {
        let toml_str = r#"
[session]
context_messages = 50
retention_days = 30
subagent_max_depth = 5
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let file_session = config.session.unwrap_or_default();
        let context_messages = file_session.context_messages.or(Some(200));
        let subagent_max_depth = file_session.subagent_max_depth.unwrap_or(3);

        assert_eq!(context_messages, Some(50));
        assert_eq!(file_session.retention_days, Some(30));
        assert_eq!(subagent_max_depth, 5);
    }

    #[test]
    fn test_mcp_auth_client_credentials() {
        let toml_str = r#"
[[mcp.servers]]
name = "api"
transport = "http"
url = "https://api.example.com/mcp"

[mcp.servers.auth]
type = "client_credentials"
client_id = "my-client"
client_secret = "my-secret"
scopes = ["read", "write"]
resource = "https://api.example.com"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let servers = config.mcp.unwrap().servers.unwrap();
        assert_eq!(servers.len(), 1);
        let auth = servers[0].auth.as_ref().expect("auth should be present");
        match auth {
            McpAuthConfig::ClientCredentials {
                client_id,
                client_secret,
                scopes,
                resource,
            } => {
                assert_eq!(client_id, "my-client");
                assert_eq!(client_secret, "my-secret");
                assert_eq!(
                    scopes.as_deref(),
                    Some(["read".to_string(), "write".to_string()].as_slice())
                );
                assert_eq!(resource.as_deref(), Some("https://api.example.com"));
            }
            other => panic!("expected ClientCredentials, got {:?}", other),
        }
    }

    #[test]
    fn test_mcp_auth_client_credentials_jwt() {
        let toml_str = r#"
[[mcp.servers]]
name = "api"
transport = "http"
url = "https://api.example.com/mcp"

[mcp.servers.auth]
type = "client_credentials_jwt"
client_id = "my-client"
signing_key_path = "/path/to/key.pem"
signing_algorithm = "ES256"
scopes = ["admin"]
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let servers = config.mcp.unwrap().servers.unwrap();
        let auth = servers[0].auth.as_ref().expect("auth should be present");
        match auth {
            McpAuthConfig::ClientCredentialsJwt {
                client_id,
                signing_key_path,
                signing_algorithm,
                scopes,
                resource,
            } => {
                assert_eq!(client_id, "my-client");
                assert_eq!(signing_key_path, "/path/to/key.pem");
                assert_eq!(signing_algorithm.as_deref(), Some("ES256"));
                assert_eq!(scopes.as_deref(), Some(["admin".to_string()].as_slice()));
                assert!(resource.is_none());
            }
            other => panic!("expected ClientCredentialsJwt, got {:?}", other),
        }
    }

    #[test]
    fn test_mcp_auth_oauth() {
        let toml_str = r#"
[[mcp.servers]]
name = "github"
transport = "http"
url = "https://mcp.example.com"

[mcp.servers.auth]
type = "oauth"
client_id = "my-app"
scopes = ["repo", "user"]
redirect_port = 9000
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let servers = config.mcp.unwrap().servers.unwrap();
        let auth = servers[0].auth.as_ref().expect("auth should be present");
        match auth {
            McpAuthConfig::OAuth {
                client_id,
                client_secret,
                scopes,
                redirect_port,
            } => {
                assert_eq!(client_id.as_deref(), Some("my-app"));
                assert!(client_secret.is_none());
                assert_eq!(
                    scopes.as_deref(),
                    Some(["repo".to_string(), "user".to_string()].as_slice())
                );
                assert_eq!(*redirect_port, Some(9000));
            }
            other => panic!("expected OAuth, got {:?}", other),
        }
    }

    #[test]
    fn test_mcp_auth_oauth_minimal() {
        let toml_str = r#"
[[mcp.servers]]
name = "api"
transport = "http"
url = "https://api.example.com/mcp"

[mcp.servers.auth]
type = "oauth"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let servers = config.mcp.unwrap().servers.unwrap();
        let auth = servers[0].auth.as_ref().expect("auth should be present");
        match auth {
            McpAuthConfig::OAuth {
                client_id,
                client_secret,
                scopes,
                redirect_port,
            } => {
                assert!(client_id.is_none());
                assert!(client_secret.is_none());
                assert!(scopes.is_none());
                assert!(redirect_port.is_none());
            }
            other => panic!("expected OAuth, got {:?}", other),
        }
    }

    #[test]
    fn test_mcp_no_auth() {
        let toml_str = r#"
[[mcp.servers]]
name = "simple"
transport = "http"
url = "https://api.example.com/mcp"
auth_token = "bearer-token"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let servers = config.mcp.unwrap().servers.unwrap();
        assert!(servers[0].auth.is_none());
        assert_eq!(servers[0].auth_token.as_deref(), Some("bearer-token"));
    }

    #[test]
    fn test_parse_input_style_known_values() {
        use nu_ansi_term::{Color, Style};
        assert_eq!(parse_input_style("bold"), Style::new().bold());
        assert_eq!(parse_input_style("BOLD"), Style::new().bold());
        assert_eq!(parse_input_style("dim"), Style::new().dimmed());
        assert_eq!(parse_input_style("cyan"), Style::new().fg(Color::Cyan));
        assert_eq!(parse_input_style("purple"), Style::new().fg(Color::Magenta));
    }

    #[test]
    fn test_parse_input_style_none_is_plain() {
        use nu_ansi_term::Style;
        assert_eq!(parse_input_style("none"), Style::default());
    }

    #[test]
    fn test_parse_input_style_default_and_empty_yield_preset() {
        let preset = default_input_style();
        assert_eq!(parse_input_style(""), preset);
        assert_eq!(parse_input_style("default"), preset);
        assert!(preset.is_bold);
        assert!(preset.foreground.is_some(), "default must set foreground");
        assert!(preset.background.is_some(), "default must set background");
    }

    #[test]
    fn test_parse_input_style_reverse() {
        assert!(parse_input_style("reverse").is_reverse);
    }

    #[test]
    fn test_parse_input_style_unknown_falls_back_to_default() {
        // Invalid keywords warn but must not panic; fall back to the same preset used when the key
        // is unset.
        assert_eq!(parse_input_style("superbold"), default_input_style());
    }

    #[test]
    fn test_tools_config_deserialization() {
        let toml_str = r#"
[tools]
allowed_tools = ["read_file", "find_files"]
disabled_tools = ["search_web"]

[tools.tool_permissions]
execute_command = "write"
read_file = "ask"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let tools = config.tools.expect("tools should be present");
        assert_eq!(
            tools.allowed_tools.as_deref(),
            Some(["read_file".to_string(), "find_files".to_string()].as_slice())
        );
        assert_eq!(
            tools.disabled_tools.as_deref(),
            Some(["search_web".to_string()].as_slice())
        );
        let perms = tools.tool_permissions.expect("tool_permissions set");
        assert_eq!(
            perms.get("execute_command").map(String::as_str),
            Some("write")
        );
        assert_eq!(perms.get("read_file").map(String::as_str), Some("ask"));
    }

    #[test]
    fn test_tools_config_missing_is_none() {
        let config: ConfigFile = toml::from_str("").expect("failed to parse empty toml");
        assert!(config.tools.is_none());
    }

    #[test]
    fn test_tools_config_invalid_permission_drops_entry() {
        // Drive the post-parse filter directly; ResolvedConfig::from_cli runs this loop. Checks
        // that a bad level string is filtered out without panicking and that valid entries still
        // land.
        let raw: HashMap<String, String> = [
            ("read_file".to_string(), "unrestricted".to_string()),
            ("write_file".to_string(), "superuser".to_string()),
            ("find_files".to_string(), "read".to_string()),
        ]
        .into_iter()
        .collect();
        let parsed: HashMap<String, Permission> = raw
            .into_iter()
            .filter_map(|(name, level)| level.parse::<Permission>().ok().map(|p| (name, p)))
            .collect();
        assert_eq!(
            parsed.get("read_file").copied(),
            Some(Permission::Unrestricted)
        );
        assert_eq!(parsed.get("find_files").copied(), Some(Permission::Read));
        assert!(
            !parsed.contains_key("write_file"),
            "invalid level 'superuser' must be dropped"
        );
    }

    #[test]
    fn test_session_resume_from_cli() {
        use clap::Parser as _;

        let parse = |args: &[&str]| SessionResume::from_cli(&crate::cli::Cli::parse_from(args));

        assert_eq!(parse(&["meka"]), None);
        assert_eq!(parse(&["meka", "-c"]), Some(SessionResume::Last));
        assert_eq!(
            parse(&["meka", "-r", "550e8400"]),
            Some(SessionResume::Id("550e8400".to_string())),
        );
        // A prompt alongside either flag is a prompt, not a session; that ambiguity is exactly what
        // splitting the old optional-value `-c` removed.
        assert_eq!(
            parse(&["meka", "-c", "fix the bug"]),
            Some(SessionResume::Last)
        );
        assert_eq!(
            parse(&["meka", "-r", "550e8400", "fix the bug"]),
            Some(SessionResume::Id("550e8400".to_string())),
        );
    }

    /// The conventional path is the tier with no configuration at all, so a regression here is
    /// invisible: instructions simply stop applying and nothing says so.
    #[test]
    fn test_instructions_resolve_from_the_conventional_file() {
        let _guard = CONFIG_DIR_ENV_LOCK.blocking_lock();
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("instructions.md"), "  from the file  \n")
            .expect("write instructions.md");

        // SAFETY: `CONFIG_DIR_ENV_LOCK` serialises this with every other env-var test.
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let resolved = resolve_instructions(None);
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let found = resolved.expect("ok").expect("some");
        assert_eq!(found.text, "from the file", "trimmed");
        assert!(matches!(
            found.source,
            crate::instructions::InstructionsSource::Files(_)
        ));
    }

    /// Splitting a grown `instructions.md` into a directory should be a rename, so the directory
    /// has to win rather than the two silently concatenating or the file shadowing it.
    #[test]
    fn test_instructions_directory_wins_over_the_single_file() {
        let _guard = CONFIG_DIR_ENV_LOCK.blocking_lock();
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("instructions.md"), "single file").expect("write file");
        std::fs::create_dir(dir.path().join("instructions")).expect("mkdir");
        std::fs::write(dir.path().join("instructions/10-a.md"), "split one").expect("write a");
        std::fs::write(dir.path().join("instructions/20-b.md"), "split two").expect("write b");

        // SAFETY: guarded above.
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let resolved = resolve_instructions(None);
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        assert_eq!(
            resolved.expect("ok").expect("some").text,
            "split one\n\nsplit two"
        );
    }

    /// The `mekabox` case: the config directory is mounted read-only, so the container's own
    /// instructions can only arrive as a string. If the env tier ever stopped beating the
    /// conventional file, that wrapper would silently run with the host's instructions instead.
    #[test]
    fn test_inline_env_beats_the_conventional_file() {
        let _guard = CONFIG_DIR_ENV_LOCK.blocking_lock();
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("instructions.md"), "FROM THE FILE").expect("write");

        // SAFETY: guarded above.
        unsafe {
            std::env::set_var("MEKA_CONFIG_DIR", dir.path());
            std::env::set_var("MEKA_INSTRUCTIONS", "FROM THE ENV");
        }
        let resolved = resolve_instructions(None);
        unsafe {
            std::env::remove_var("MEKA_CONFIG_DIR");
            std::env::remove_var("MEKA_INSTRUCTIONS");
        }

        let found = resolved.expect("ok").expect("some");
        assert_eq!(found.text, "FROM THE ENV");
        assert_eq!(found.source, crate::instructions::InstructionsSource::Env);
    }

    #[test]
    fn test_instructions_file_env_reads_the_named_path() {
        let _guard = CONFIG_DIR_ENV_LOCK.blocking_lock();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("elsewhere.md");
        std::fs::write(&path, "from the named path").expect("write");

        // SAFETY: guarded above.
        unsafe { std::env::set_var("MEKA_INSTRUCTIONS_FILE", &path) };
        let resolved = resolve_instructions(None);
        unsafe { std::env::remove_var("MEKA_INSTRUCTIONS_FILE") };

        assert_eq!(
            resolved.expect("ok").expect("some").text,
            "from the named path"
        );
    }

    /// No reading of "both set" means both, so resolving one silently would hide the mistake until
    /// the agent behaved unexpectedly.
    #[test]
    fn test_both_instruction_env_vars_is_an_error() {
        let _guard = CONFIG_DIR_ENV_LOCK.blocking_lock();
        // SAFETY: guarded above.
        unsafe {
            std::env::set_var("MEKA_INSTRUCTIONS", "inline");
            std::env::set_var("MEKA_INSTRUCTIONS_FILE", "/nonexistent.md");
        }
        let resolved = resolve_instructions(None);
        unsafe {
            std::env::remove_var("MEKA_INSTRUCTIONS");
            std::env::remove_var("MEKA_INSTRUCTIONS_FILE");
        }

        let error = resolved.expect_err("must refuse").to_string();
        assert!(error.contains("MEKA_INSTRUCTIONS"), "{error}");
        assert!(error.contains("MEKA_INSTRUCTIONS_FILE"), "{error}");
    }

    #[test]
    fn test_flag_beats_every_env_tier_and_trims() {
        let _guard = CONFIG_DIR_ENV_LOCK.blocking_lock();
        // SAFETY: guarded above.
        unsafe { std::env::set_var("MEKA_INSTRUCTIONS", "from the env") };
        let resolved = resolve_instructions(Some("  from the flag  "));
        let blank = resolve_instructions(Some("   \n\t "));
        unsafe { std::env::remove_var("MEKA_INSTRUCTIONS") };

        assert_eq!(resolved.expect("ok").expect("some").text, "from the flag");
        assert!(
            blank.expect("ok").is_none(),
            "an all-whitespace flag is no instructions, not empty ones"
        );
    }

    /// Touches process env, so it serializes against any other env-var test in this file via
    /// [`CONFIG_DIR_ENV_LOCK`].
    #[test]
    fn a_profiles_thinking_mode_reaches_the_resolved_config() {
        // The resolution step, not the parse: `ProviderProfile.thinking` deserializing is separate
        // from `from_cli` actually consulting it, and dropping the profile arm here is invisible to
        // every wire-level test - those construct providers directly.
        let _guard = CONFIG_DIR_ENV_LOCK.blocking_lock();

        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            r#"
default_provider = "local"

[providers.local]
type = "anthropic-messages"
model = "some-local-model"
thinking = "budgeted"
"#,
        )
        .expect("write config.toml");

        // SAFETY: `CONFIG_DIR_ENV_LOCK` serializes this with any other env-var test.
        unsafe {
            std::env::set_var("MEKA_CONFIG_DIR", dir.path());
        }
        use clap::Parser;
        let from_profile = ResolvedConfig::from_cli(&crate::cli::Cli::parse_from(["meka"]));
        // A CLI flag still wins over the profile.
        let from_flag =
            ResolvedConfig::from_cli(&crate::cli::Cli::parse_from(["meka", "--thinking", "off"]));
        // SAFETY: same as above; the guard is held for the full set→read→clear cycle.
        unsafe {
            std::env::remove_var("MEKA_CONFIG_DIR");
        }

        assert_eq!(
            from_profile.thinking,
            crate::provider::ThinkingMode::Budgeted
        );
        assert_eq!(from_flag.thinking, crate::provider::ThinkingMode::Off);
    }

    /// End-to-end check that `MEKA_INSTRUCTIONS` reaches `user_instructions` when
    /// `--instructions` is not on the CLI. Drives the actual `from_cli` path rather than
    /// `resolve_instructions` directly, to catch a regression where the resolved value never makes
    /// it onto `ResolvedConfig`.
    ///
    /// Touches process env, so it serializes against any other env-var test in this file via
    /// [`CONFIG_DIR_ENV_LOCK`].
    #[test]
    fn test_env_var_overrides_config_file_instructions() {
        // The module-level lock, not a private one: a function-local static only serialises the
        // test against itself, which let concurrent tests clobber each other's MEKA_CONFIG_DIR.
        let _guard = CONFIG_DIR_ENV_LOCK.blocking_lock();

        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("instructions.md"), "FROM THE FILE")
            .expect("write instructions.md");

        // SAFETY: `CONFIG_DIR_ENV_LOCK` serializes this with any other env-var test.
        unsafe {
            std::env::set_var("MEKA_CONFIG_DIR", dir.path());
            std::env::set_var("MEKA_INSTRUCTIONS", "FROM ENV VAR");
        }

        use clap::Parser;
        let cli = crate::cli::Cli::parse_from(["meka"]);
        let resolved = ResolvedConfig::from_cli(&cli);

        // SAFETY: same as above, the `CONFIG_DIR_ENV_LOCK` guard is held for the full
        // set→read→clear cycle.
        unsafe {
            std::env::remove_var("MEKA_CONFIG_DIR");
            std::env::remove_var("MEKA_INSTRUCTIONS");
        }

        assert_eq!(
            resolved.user_instructions.as_deref(),
            Some("FROM ENV VAR"),
            "MEKA_INSTRUCTIONS should override the conventional instructions file",
        );
    }

    #[test]
    fn test_parse_sandbox_backend_override() {
        assert_eq!(
            parse_sandbox_backend_override("landlock"),
            Some(SandboxBackend::Landlock)
        );
        assert_eq!(
            parse_sandbox_backend_override("Bubblewrap"),
            Some(SandboxBackend::Bubblewrap)
        );
        assert_eq!(
            parse_sandbox_backend_override("  LANDLOCK  "),
            Some(SandboxBackend::Landlock),
            "value is trimmed and case-insensitive"
        );
        assert_eq!(parse_sandbox_backend_override(""), None);
        assert_eq!(
            parse_sandbox_backend_override("nonsense"),
            None,
            "unrecognized values are ignored, not fatal"
        );
    }

    #[test]
    fn test_sandbox_backend_from_str() {
        assert_eq!(
            "landlock".parse::<SandboxBackend>(),
            Ok(SandboxBackend::Landlock)
        );
        assert_eq!(
            "Bubblewrap".parse::<SandboxBackend>(),
            Ok(SandboxBackend::Bubblewrap)
        );
        assert_eq!(
            "  LANDLOCK  ".parse::<SandboxBackend>(),
            Ok(SandboxBackend::Landlock),
            "trimmed and case-insensitive"
        );
        // Unlike the env path, the CLI parse surfaces an error for a bad value.
        assert!("bogus".parse::<SandboxBackend>().is_err());
    }

    #[test]
    fn test_write_file_atomic_writes_content_and_no_tmp_left() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("sub").join("config.toml");
        write_file_atomic(&path, "[x]\nk = 1\n").expect("atomic write");
        assert_eq!(
            std::fs::read_to_string(&path).expect("read back"),
            "[x]\nk = 1\n"
        );
        // The temporary file must not be left behind after a successful write.
        let tmp = dir.path().join("sub").join("config.toml.tmp");
        assert!(!tmp.exists(), "temp file should not remain: {:?}", tmp);
    }

    #[test]
    fn test_write_file_atomic_overwrites_existing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "old contents that are LONGER than the new ones").expect("seed file");
        write_file_atomic(&path, "new\n").expect("atomic overwrite");
        assert_eq!(std::fs::read_to_string(&path).expect("read back"), "new\n");
    }

    /// Dotfile managers (stow, chezmoi, yadm) leave `config.toml` as a link into a tracked repo.
    /// `rename(2)` replaces the directory entry, so writing without resolving first turned the link
    /// into a regular file: the tracked copy went stale, every later edit diverged from it, and the
    /// next `stow --restow` reverted them all. The write has to land on the target.
    #[cfg(unix)]
    /// `write_file_atomic` makes each *write* atomic, which does nothing for a lost update: two
    /// editors that each read, mutate and write back will silently discard whichever finished
    /// first. The lock is what makes the read and the write one critical section.
    #[tokio::test]
    async fn the_config_lock_serialises_a_read_modify_write() {
        let _env = CONFIG_DIR_ENV_LOCK.lock().await;
        let dir = tempfile::tempdir().expect("tempdir");
        // SAFETY: `MEKA_CONFIG_DIR` is process-global; the guard above serialises every test that
        // touches it and is held across the whole set → use → clear cycle.
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };

        let path = dir.path().join("config.toml");
        std::fs::write(&path, "counter = 0\n").expect("seed");

        // Each thread does the full read-modify-write under the lock. Without it the two reads
        // race and one increment is lost; with it the file ends at 2.
        let threads: Vec<_> = (0..2)
            .map(|_| {
                let path = path.clone();
                std::thread::spawn(move || {
                    let _lock = lock_config_file().expect("lock");
                    let contents = std::fs::read_to_string(&path).expect("read");
                    let current: u32 = contents
                        .trim()
                        .strip_prefix("counter = ")
                        .and_then(|value| value.parse().ok())
                        .expect("parse");
                    // Widen the window the lock has to cover, so an unlocked version loses
                    // reliably rather than occasionally.
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    write_file_atomic(&path, &format!("counter = {}\n", current + 1))
                        .expect("write");
                })
            })
            .collect();
        for thread in threads {
            thread.join().expect("thread");
        }

        let final_contents = std::fs::read_to_string(&path).expect("read");
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
        assert_eq!(
            final_contents.trim(),
            "counter = 2",
            "both increments must survive; one lost means the reads interleaved"
        );
    }

    /// `flock` is per open file description, so a second `open` in the same process conflicts with
    /// the first exactly as another process would. Nesting is not hypothetical: `meka mcp add` on
    /// an HTTPS URL holds the lock through `run_add` and then calls `persist_auth_block_for`,
    /// which wants it too. The first version of this lock hung there.
    #[tokio::test]
    async fn the_config_lock_can_be_taken_again_by_the_thread_already_holding_it() {
        let _env = CONFIG_DIR_ENV_LOCK.lock().await;
        let dir = tempfile::tempdir().expect("tempdir");
        // SAFETY: `MEKA_CONFIG_DIR` is process-global; the guard above serialises every test that
        // touches it and is held across the whole set -> use -> clear cycle.
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };

        let outer = lock_config_file().expect("outer");
        let inner = lock_config_file().expect("a nested acquisition must not block");
        drop(inner);
        // Still held by the outer guard, so a third nested take is also fine.
        let another = lock_config_file().expect("still reentrant");
        drop(another);
        drop(outer);

        // Once the outermost is dropped the file is free again, so a fresh acquisition takes the
        // real lock rather than believing it is still nested.
        let reacquired = lock_config_file().expect("reacquire after full release");
        assert!(matches!(reacquired, ConfigFileLock::Held { .. }));
        drop(reacquired);

        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
    }

    /// Concurrent writers of one path must not splice their contents together.
    ///
    /// A fixed `<file>.tmp` is shared by every writer: both `open(O_TRUNC)` the same inode, both
    /// write from offset zero, and whoever renames last publishes a mixture of the two as a
    /// success. Reproduced before the fix at 4 MiB against 64 KiB; the sizes here are the same
    /// shape, small enough to stay quick.
    #[test]
    fn write_file_atomic_never_publishes_two_writers_spliced_together() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("contended.txt");
        let long = "A".repeat(512 * 1024);
        let short = "B".repeat(4 * 1024);

        for _ in 0..40 {
            std::thread::scope(|scope| {
                for content in [&long, &short] {
                    let path = &path;
                    scope.spawn(move || {
                        write_file_atomic(path, content).expect("write");
                    });
                }
            });
            let published = std::fs::read_to_string(&path).expect("read back");
            assert!(
                published == long || published == short,
                "published a splice: {} bytes, {} A and {} B",
                published.len(),
                published.matches('A').count(),
                published.matches('B').count(),
            );
        }
        // And no temp files are left behind by a run that succeeded.
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .expect("read dir")
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "leftover temp files: {leftovers:?}");
    }

    /// An existing file's mode is preserved, but never left looser than 0600.
    ///
    /// Two failures meet here. Handing the target the temp's fresh 0600 unconditionally re-modes a
    /// file meka did not create. Carrying the old mode across verbatim is worse: it left an
    /// `auth_token` in a world-readable `config.toml` after an ordinary `meka mcp add`, because the
    /// write follows a dotfile manager's symlink out of meka's 0700 directory and the group and
    /// other bits came with it. Narrowing keeps a deliberately tighter mode and refuses a looser
    /// one.
    #[cfg(unix)]
    #[test]
    fn write_file_atomic_never_leaves_a_rewritten_file_looser_than_0600() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().expect("tempdir");
        for (seeded, expected) in [
            (0o644, 0o600),
            (0o400, 0o400),
            (0o600, 0o600),
            (0o660, 0o600),
        ] {
            let path = dir.path().join(format!("f{seeded:o}.toml"));
            std::fs::write(&path, "old\n").expect("seed");
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(seeded))
                .expect("chmod");

            write_file_atomic(&path, "new\n").expect("rewrite");

            assert_eq!(
                std::fs::metadata(&path).expect("stat").permissions().mode() & 0o777,
                expected,
                "seeded {seeded:o}"
            );
            assert_eq!(std::fs::read_to_string(&path).expect("read"), "new\n");
        }

        // A file meka creates still gets the restrictive mode.
        let fresh = dir.path().join("fresh.toml");
        write_file_atomic(&fresh, "new\n").expect("create");
        assert_eq!(
            std::fs::metadata(&fresh)
                .expect("stat")
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }

    /// Following a link out of meka's tree must not re-mode the directory it lands in.
    ///
    /// One `meka mcp disable` against a `config.toml` symlinked into `~/dotfiles` set that whole
    /// directory to 0700, hiding every unrelated file in it. Observed.
    #[cfg(unix)]
    #[test]
    fn write_file_atomic_does_not_tighten_a_directory_it_was_redirected_into() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().expect("tempdir");
        let dotfiles = dir.path().join("dotfiles");
        std::fs::create_dir_all(&dotfiles).expect("dotfiles");
        std::fs::set_permissions(&dotfiles, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        let target = dotfiles.join("config.toml");
        std::fs::write(&target, "old\n").expect("seed");
        let link = dir.path().join("config.toml");
        std::os::unix::fs::symlink(&target, &link).expect("symlink");

        write_file_atomic(&link, "new\n").expect("write through the link");

        assert_eq!(
            std::fs::read_to_string(&target).expect("read"),
            "new\n",
            "the write must still land on the target"
        );
        assert_eq!(
            std::fs::metadata(&dotfiles)
                .expect("stat")
                .permissions()
                .mode()
                & 0o777,
            0o755,
            "a directory meka was redirected into is not one it owns"
        );
    }

    // Unix-only: the body calls `std::os::unix::fs::symlink`, so without this the file does not
    // compile on Windows at all -- taking out the `lint` and `test` jobs that CI now runs on the
    // three-OS matrix, which is exactly the platform-only break that matrix was added to catch and
    // which cannot be reproduced from a Linux workstation.
    #[cfg(unix)]
    #[test]
    fn write_file_atomic_writes_through_a_symlink_instead_of_replacing_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("tracked.toml");
        let link = dir.path().join("config.toml");
        std::fs::write(&target, "old\n").expect("seed target");
        std::os::unix::fs::symlink(&target, &link).expect("symlink");

        write_file_atomic(&link, "new\n").expect("atomic write through link");

        assert!(
            std::fs::symlink_metadata(&link)
                .expect("stat link")
                .file_type()
                .is_symlink(),
            "the link itself must survive the write"
        );
        assert_eq!(
            std::fs::read_to_string(&target).expect("read target"),
            "new\n",
            "the tracked file is what should have changed"
        );
    }

    /// A chain still resolves to the real file, and a cycle must not spin.
    #[cfg(unix)]
    #[test]
    fn write_file_atomic_follows_a_symlink_chain_and_survives_a_cycle() {
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("real.toml");
        let middle = dir.path().join("middle.toml");
        let entry = dir.path().join("entry.toml");
        std::fs::write(&target, "old\n").expect("seed");
        std::os::unix::fs::symlink(&target, &middle).expect("link 1");
        std::os::unix::fs::symlink(&middle, &entry).expect("link 2");

        write_file_atomic(&entry, "new\n").expect("write through chain");
        assert_eq!(std::fs::read_to_string(&target).expect("read"), "new\n");

        // `a -> b -> a`: resolution gives up and the call still returns rather than hanging.
        let loop_a = dir.path().join("a.toml");
        let loop_b = dir.path().join("b.toml");
        std::os::unix::fs::symlink(&loop_b, &loop_a).expect("cycle a");
        std::os::unix::fs::symlink(&loop_a, &loop_b).expect("cycle b");
        let _ = write_file_atomic(&loop_a, "whatever\n");
    }

    #[cfg(unix)]
    #[test]
    fn test_write_file_atomic_sets_unix_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let parent = dir.path().join("meka");
        let path = parent.join("config.toml");
        write_file_atomic(&path, "x = 1\n").expect("atomic write");

        let file_mode = std::fs::metadata(&path)
            .expect("stat file")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(
            file_mode, 0o600,
            "config file should be 0600, got {:o}",
            file_mode
        );

        let dir_mode = std::fs::metadata(&parent)
            .expect("stat dir")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(
            dir_mode, 0o700,
            "config dir should be 0700, got {:o}",
            dir_mode
        );
    }

    fn enabled_set(modes: &[Permission]) -> EnabledPermissions {
        EnabledPermissions::from_modes(modes.iter().copied()).unwrap()
    }

    fn enabled_strings(modes: &[&str]) -> Vec<String> {
        modes.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_resolve_permission_no_config() {
        let (perm, enabled) = resolve_permission(None, None, None, None);
        assert_eq!(perm, Permission::Read);
        assert_eq!(enabled, EnabledPermissions::DEFAULT);
    }

    #[test]
    fn test_resolve_permission_explicit_enabled_all() {
        let list = enabled_strings(&["none", "read", "workspace", "ask", "unrestricted"]);
        let (_perm, enabled) = resolve_permission(None, None, None, Some(&list));
        assert!(enabled.is_enabled(Permission::Ask));
        assert_eq!(enabled.iter().count(), 5);
    }

    #[test]
    fn test_resolve_permission_invalid_entry_warns_and_drops() {
        let list = enabled_strings(&["read", "lol", "unrestricted"]);
        let (perm, enabled) = resolve_permission(None, None, None, Some(&list));
        assert_eq!(
            enabled,
            enabled_set(&[Permission::Read, Permission::Unrestricted])
        );
        assert_eq!(perm, Permission::Read);
    }

    /// A list that yields nothing must not resolve to a set *wider* than the one written.
    ///
    /// `EnabledPermissions::DEFAULT` was the old answer for both of these, and it holds four modes
    /// including `unrestricted`. So `enabled = []`, which asks for nothing, and `enabled =
    /// ["write"]`, which asks for exactly one now-retired mode, each came back with the full ladder
    /// reachable by Shift+Tab. The fallback has to move down the ladder, never up.
    #[test]
    fn an_enabled_list_that_yields_nothing_falls_back_to_read_alone() {
        for list in [enabled_strings(&[]), enabled_strings(&["write"])] {
            let (perm, enabled) = resolve_permission(None, None, None, Some(&list));
            assert_eq!(enabled, enabled_set(&[Permission::Read]), "for {list:?}");
            assert!(
                !enabled.is_enabled(Permission::Unrestricted),
                "a failed parse must not enable a mode the user did not write: {list:?}"
            );
            assert_eq!(perm, Permission::Read);
        }
    }

    #[test]
    fn test_resolve_permission_default_not_in_enabled_clamps() {
        let list = enabled_strings(&["read", "write"]);
        let (perm, _enabled) = resolve_permission(None, None, Some("ask"), Some(&list));
        // `ask` is filtered out → fall back to Read because it's enabled.
        assert_eq!(perm, Permission::Read);
    }

    #[test]
    fn test_resolve_permission_default_not_in_enabled_no_read_falls_to_lowest() {
        let list = enabled_strings(&["ask", "write"]);
        let (perm, _enabled) = resolve_permission(None, None, Some("none"), Some(&list));
        // none isn't enabled, Read isn't either → lowest enabled is Ask.
        assert_eq!(perm, Permission::Ask);
    }

    #[test]
    fn test_resolve_permission_invalid_default_falls_back() {
        let (perm, _enabled) = resolve_permission(None, None, Some("weird"), None);
        assert_eq!(perm, Permission::Read);
    }

    #[test]
    fn test_resolve_permission_explicit_default_used() {
        let (perm, _enabled) = resolve_permission(None, None, Some("unrestricted"), None);
        assert_eq!(perm, Permission::Unrestricted);
    }

    #[test]
    fn test_resolve_permission_cli_override_disabled_clamps_to_default() {
        // `ask` not enabled → CLI request for ask warns and clamps to the configured default
        // (Read).
        let (perm, _enabled) = resolve_permission(Some(Permission::Ask), None, None, None);
        assert_eq!(perm, Permission::Read);
    }

    #[test]
    fn test_resolve_permission_cli_override_enabled_wins() {
        let list = enabled_strings(&["none", "read", "workspace", "ask", "unrestricted"]);
        let (perm, _enabled) = resolve_permission(Some(Permission::Ask), None, None, Some(&list));
        assert_eq!(perm, Permission::Ask);
    }

    #[test]
    fn test_resolve_permission_env_override_used() {
        let (perm, _enabled) = resolve_permission(None, Some("unrestricted"), None, None);
        assert_eq!(perm, Permission::Unrestricted);
    }

    #[test]
    fn test_resolve_permission_env_override_disabled_clamps() {
        // env asks for ask, but ask isn't in DEFAULT enabled set.
        let (perm, _enabled) = resolve_permission(None, Some("ask"), None, None);
        assert_eq!(perm, Permission::Read);
    }

    /// The env value has to be one meka still accepts, or this proves nothing: an unparseable
    /// `MEKA_PERMISSION` is dropped before the precedence rule is ever consulted, so the CLI wins
    /// by default rather than by rule, and reversing the precedence leaves the test green.
    #[test]
    fn test_resolve_permission_cli_beats_env() {
        let (perm, _enabled) =
            resolve_permission(Some(Permission::None), Some("unrestricted"), None, None);
        assert_eq!(perm, Permission::None);
    }

    #[test]
    fn test_permissions_config_deserialization() {
        let toml_str = r#"
[permissions]
default = "workspace"
enabled = ["read", "workspace"]
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("parse toml");
        let perms = config.permissions.expect("permissions present");
        assert_eq!(perms.default.as_deref(), Some("workspace"));
        assert_eq!(
            perms.enabled.as_deref(),
            Some(&[String::from("read"), String::from("workspace")][..])
        );
    }

    /// `sandbox_backend = "bubblewrap"` and `"landlock"` deserialize cleanly. Any other value,
    /// including the obvious abbreviation `"bwrap"`, must be rejected; we don't want alias creep
    /// that would silently desync generated configs from hand-edited ones.
    #[test]
    fn test_sandbox_backend_deserializes_strict_values() {
        let bubblewrap: ShellConfig =
            toml::from_str(r#"sandbox_backend = "bubblewrap""#).expect("deserialize bubblewrap");
        assert_eq!(bubblewrap.sandbox_backend, Some(SandboxBackend::Bubblewrap));
        let landlock: ShellConfig =
            toml::from_str(r#"sandbox_backend = "landlock""#).expect("deserialize landlock");
        assert_eq!(landlock.sandbox_backend, Some(SandboxBackend::Landlock));
        // No aliases / case variants accepted.
        assert!(toml::from_str::<ShellConfig>(r#"sandbox_backend = "bwrap""#).is_err());
        assert!(toml::from_str::<ShellConfig>(r#"sandbox_backend = "Bubblewrap""#).is_err());
        assert!(toml::from_str::<ShellConfig>(r#"sandbox_backend = "none""#).is_err());
    }

    /// When the user pins `sandbox_backend = "..."` explicitly, the resolver returns that choice
    /// with `auto_resolved == false`: no silent fallback even if the probe would suggest
    /// otherwise.
    #[cfg(target_os = "linux")]
    #[test]
    fn test_resolve_sandbox_backend_explicit_value_is_binding() {
        let (backend, auto_resolved, _probe) =
            resolve_sandbox_backend(Some(SandboxBackend::Landlock));
        assert_eq!(backend, SandboxBackend::Landlock);
        assert!(!auto_resolved);

        let (backend, auto_resolved, _probe) =
            resolve_sandbox_backend(Some(SandboxBackend::Bubblewrap));
        assert_eq!(backend, SandboxBackend::Bubblewrap);
        assert!(!auto_resolved);
    }

    /// When the user has not pinned a backend, resolve_sandbox_backend must surface `auto_resolved
    /// == true`. The exact backend it picks depends on whether the host has bwrap installed and
    /// supports user namespaces, so we just assert the auto flag is set and one of the two backends
    /// came back.
    #[cfg(target_os = "linux")]
    #[test]
    fn test_resolve_sandbox_backend_auto_resolves_when_unset() {
        let (backend, auto_resolved, _probe) = resolve_sandbox_backend(None);
        assert!(auto_resolved);
        assert!(matches!(
            backend,
            SandboxBackend::Bubblewrap | SandboxBackend::Landlock
        ));
    }

    /// On macOS / Windows the `sandbox_backend` config field is documented as ignored. The resolver
    /// must still return a probe that reflects the platform's native sandbox capability rather than
    /// the never-applicable Linux defaults, so the downstream wiring in `src/main.rs` can map it to
    /// `SandboxCapability` for `sandbox-exec` / Low-integrity. Guards against the regression that
    /// surfaces when only the Linux probe paths are wired up.
    #[cfg(not(target_os = "linux"))]
    #[test]
    fn test_resolve_sandbox_backend_uses_platform_sandbox_on_non_linux() {
        use crate::sandbox::{BackendProbe, SandboxCapability};

        // Explicit `Some(...)` is ignored on non-Linux; the field is documented as Linux-only.
        let (_backend, auto_resolved, probe) =
            resolve_sandbox_backend(Some(SandboxBackend::Bubblewrap));
        assert!(!auto_resolved);
        // The probe should reflect what `detect()` reports for this host, surfaced as `Ok(...)` so
        // the consumer can drop into the platform's spawn path.
        match probe {
            BackendProbe::Ok(SandboxCapability::Unavailable) => {
                panic!("Ok(Unavailable) is incoherent: expected a real capability or Missing")
            }
            BackendProbe::Ok(_) | BackendProbe::Missing { .. } => {}
            other => panic!("unexpected probe variant on non-Linux: {:?}", other),
        }
    }
}
