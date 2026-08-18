//! Model Context Protocol (MCP) client integration. Manages the lifecycle of configured MCP servers
//! (stdio child processes or streamable HTTP), exposes their tools through the regular
//! [`crate::tools`] registry, and handles OAuth/JWT authentication for HTTP transports.

pub mod auth;
pub mod cli;
pub mod connector;
pub mod elicitation;
pub mod expand;
pub mod handler;
pub mod progress;
pub mod resource_updates;
pub mod sanitize;
pub mod transport;

use std::{
    collections::HashMap,
    sync::{Arc, OnceLock, Weak},
};

pub use handler::McpToolAdapter;
use rmcp::{
    Peer, RoleClient,
    model::{
        GetPromptRequestParams, GetPromptResult, Prompt, ReadResourceRequestParams,
        ReadResourceResult, Resource,
    },
    service::ServiceError,
};
use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;

use crate::{
    config::{McpServerConfig, McpTransport},
    error::{MekaError, Result},
    permission::Permission,
    session::TokenStore,
};

/// Cap MCP-provided text (tool descriptions, resource/prompt descriptions) to this many characters
/// so a chatty server can't blow up the system prompt. Mirrors Claude Code's
/// `MAX_MCP_DESCRIPTION_LENGTH`.
pub const MAX_MCP_DESCRIPTION_LENGTH: usize = 2048;

/// Cap on base64 payload size for an MCP image tool-result block. A server returning a giant image
/// would otherwise be cloned verbatim, forwarded to the provider, billed against the user's API
/// quota, and risk OOM. Mirrors the 10 MiB body cap on `fetch_url`.
pub const MAX_MCP_IMAGE_BYTES: usize = 10 * 1024 * 1024;

/// Tools one MCP server may advertise before the list is cut.
///
/// `list_all_tools` pages until the server stops offering a cursor, so a server that keeps offering
/// one keeps meka reading -- and every tool it returns costs a `ToolDefinition` resident for the
/// session plus a line in the catalogue the model reads on every turn. The cap is far above any
/// real server (the largest published ones advertise dozens) and exists so the ceiling belongs to
/// meka rather than to whatever is on the other end of the socket.
pub const MAX_MCP_TOOLS_PER_SERVER: usize = 512;

/// Keep at most [`MAX_MCP_TOOLS_PER_SERVER`] of what a server advertised, warning when it bites.
///
/// A free function rather than an inline block so the bound is assertable: reaching it through
/// `list_tools_bounded` needs a live server, so raising the constant to `usize::MAX` left every
/// suite green. The tool list is held per session and re-sent in every request's tools array, so
/// an unbounded one is resident cost on every turn, not just at connect.
fn cap_advertised_tools<T>(listed: Vec<T>, server_name: &str) -> Vec<T> {
    if listed.len() > MAX_MCP_TOOLS_PER_SERVER {
        tracing::warn!(
            "MCP server '{}' advertised {} tools; keeping the first {}",
            server_name,
            listed.len(),
            MAX_MCP_TOOLS_PER_SERVER
        );
        return listed.into_iter().take(MAX_MCP_TOOLS_PER_SERVER).collect();
    }
    listed
}

/// Bound on an MCP request made outside the connector, when no configured timeout is available.
///
/// Matches `[mcp].connect_timeout_seconds`'s own default, so a manager that never started a
/// connector behaves like one that did rather than waiting forever.
pub(crate) const DEFAULT_MCP_REQUEST_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(30);

/// Allow-list of image MIME types passed straight through to the provider. Anything else (notably
/// `image/svg+xml`, which can embed script/link elements) is converted to a text placeholder.
pub const ALLOWED_IMAGE_MIME_TYPES: &[&str] =
    &["image/png", "image/jpeg", "image/gif", "image/webp"];

pub(crate) type McpRunningService =
    rmcp::service::RunningService<RoleClient, handler::MekaClientHandler>;

tokio::task_local! {
    /// Per-task override for the frontend that should receive MCP-originated UI events fired
    /// during the in-flight tool call. Scoped by [`with_session_frontend`] from the agent dispatch
    /// site.
    ///
    /// **Important**: rmcp's notification / server-request callbacks run on *separately spawned*
    /// handler tasks (see `rmcp::service::spawn_service_task`), so this task-local is NOT visible
    /// from those callbacks directly. Instead, the [`crate::mcp::handler::McpToolAdapter`]
    /// snapshots [`current_session_frontend`] at its call site and stashes the value on the
    /// per-call progress-registry entry; the rmcp dispatch path then looks it up by token. So
    /// this task-local exists to source the frontend at the agent-driven call site only; the
    /// progress registry is what carries it across the rmcp task boundary.
    ///
    /// Outside any `with_session_frontend` scope (connection-time handshakes, REPL startup probes)
    /// [`current_session_frontend`] returns `None` and the caller falls back to either auto-decline
    /// (elicitation) or a tracing log (progress).
    static SESSION_FRONTEND: std::sync::Arc<dyn crate::frontend::Frontend>;
}

/// Read the per-session frontend currently in scope, if any. Returns `None` outside a
/// [`with_session_frontend`] block; callers must treat that as "no UI available" rather than
/// hitting a panic, because MCP callbacks can legitimately fire before any session exists
/// (connection-time handshakes) or under code paths that intentionally aren't session-scoped.
pub(crate) fn current_session_frontend() -> Option<std::sync::Arc<dyn crate::frontend::Frontend>> {
    SESSION_FRONTEND.try_with(|frontend| frontend.clone()).ok()
}

/// Scope `frontend` as the task-local override for the duration of `fut`. The agent dispatch site
/// installs this so MCP-originated UI events (progress, elicitation) route through the calling
/// session's `AcpFrontend` / `ReplFrontend` instead of through a process-global sink.
pub async fn with_session_frontend<F, T>(
    frontend: std::sync::Arc<dyn crate::frontend::Frontend>,
    fut: F,
) -> T
where
    F: std::future::Future<Output = T>,
{
    SESSION_FRONTEND.scope(frontend, fut).await
}

tokio::task_local! {
    /// Session the tool call executing on this task belongs to, scoped by
    /// `Agent::resolve_and_execute_tool`. Read by [`crate::mcp::handler::McpToolAdapter`] so
    /// `tools/call` can carry `meka/sessionId` in `_meta`, letting a server scope per-session state
    /// (a cache, a workspace, an audit trail) to the conversation that called it.
    ///
    /// A sub-agent runs under its own `Agent` with its own child session id, so a call it makes
    /// reports the child, which is the correct attribution.
    static SESSION_ID: uuid::Uuid;
}

/// The session id for the in-flight tool call, or `None` outside one. `None` is legitimate: MCP
/// callbacks fire during connection-time handshakes, before any session exists.
pub(crate) fn current_session_id() -> Option<uuid::Uuid> {
    SESSION_ID.try_with(|id| *id).ok()
}

/// Scope `session_id` as the session owning the tool call for the duration of `fut`.
pub async fn with_session_id<F, T>(session_id: uuid::Uuid, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    SESSION_ID.scope(session_id, fut).await
}

/// Total wall-clock budget for closing every MCP server on the way out. Serial teardown at up to
/// `CLOSE_TIMEOUT` per server would otherwise make exit latency scale with how many of them hang.
pub(crate) const SHUTDOWN_BUDGET: std::time::Duration = std::time::Duration::from_secs(10);

pub struct McpClientManager {
    servers: HashMap<String, Arc<ServerEntry>>,
    /// Global fallback permission from `[mcp].default_permission`. Consulted by
    /// `resolve_tool_permission` at tool-registration time when neither the server nor the user
    /// has configured a more specific permission and the server didn't advertise a
    /// `readOnlyHint`. `None` means "no user default": resolution falls through to the
    /// hardcoded strict `Write`.
    mcp_default_permission: Option<Permission>,
    /// Flipped to `true` by the background connector once every enabled entry has reached a
    /// terminal state (Connected or Failed). The turn gate watches this via
    /// [`Self::await_settled`] / [`Self::all_ready`].
    settled: tokio::sync::watch::Sender<bool>,
    /// Entries waiting to be connected by [`Self::start_connector`]. `None` once the connector has
    /// been started so a second call is a no-op; avoids re-spawning the connector if a test or the
    /// REPL re-enters the same manager.
    pending_entries: std::sync::Mutex<Option<Vec<Arc<ServerEntry>>>>,
    /// Live snapshot of every connected server's currently-registered tools. The connector writes
    /// here as each server reaches `Connected`, and `on_tool_list_changed` writes here on dynamic
    /// updates. New sessions read this snapshot at [`Self::attach_registry`] time to backfill MCP
    /// tools into their fresh per-session registry.
    tools_snapshot: tokio::sync::RwLock<HashMap<String, Vec<Arc<dyn crate::tools::Tool>>>>,
    /// Registries currently observing MCP tool updates. Sessions attach at `session/new` (or REPL
    /// startup) and detach at `session/close`. Updates from the connector or notification handler
    /// propagate to every entry.
    attached_registries: tokio::sync::RwLock<Vec<crate::tools::ToolRegistry>>,
    /// `[mcp].connect_timeout_seconds`, kept so the request paths that run *after* the connector
    /// (a `tools/list_changed` refresh, `meka mcp tools`) can bound themselves by the same value
    /// the connect did. Set by [`Self::start_connector`]; a manager that never started one
    /// (tests) falls back to [`DEFAULT_MCP_REQUEST_TIMEOUT`].
    connect_timeout: OnceLock<std::time::Duration>,
}

/// Lifecycle state of a single MCP server. Transitions:
/// - Built as `Disabled` (config says so) or `Pending` (will be connected by the background
///   connector).
/// - `Pending` → `Connected` on successful `initialize` + `list_tools`.
/// - `Pending` → `Failed` on connect error or connect-timeout.
/// - `Connected` → `Connected` (with a new `service` Arc) on reconnect.
#[derive(Clone)]
pub enum ServerState {
    Disabled,
    Pending,
    Connected {
        service: Arc<McpRunningService>,
    },
    Failed {
        error: String,
        #[allow(dead_code)]
        at: std::time::Instant,
    },
}

impl ServerState {
    /// Why this server can't serve a tool call right now, or `None` when it can.
    ///
    /// Deliberately terse and free of instructions: it states the condition and leaves the agent
    /// to decide what to do. "Still connecting" and "failed" are kept distinct because they call
    /// for opposite behaviour, and collapsing them produces an agent that either gives up too
    /// early or retries forever.
    pub fn unavailable_reason(&self) -> Option<String> {
        match self {
            ServerState::Connected { .. } => None,
            ServerState::Pending => Some("is still connecting".to_string()),
            ServerState::Failed { error, .. } => Some(format!("is unavailable: {}", error)),
            ServerState::Disabled => Some("is disabled in config".to_string()),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            ServerState::Disabled => "disabled",
            ServerState::Pending => "pending",
            ServerState::Connected { .. } => "connected",
            ServerState::Failed { .. } => "failed",
        }
    }
}

/// One enabled server that isn't `Connected`, as reported by
/// [`McpClientManager::enabled_not_connected`].
#[derive(Clone)]
pub struct NotConnected {
    pub name: String,
    /// Whether this server gates the turn. See [`crate::config::McpServerConfig::required`].
    pub required: bool,
    pub state: ServerState,
}

/// Holds the lifecycle state of a single MCP server plus reconnection machinery. Wrapped in an
/// [`Arc`] and shared between the manager, the per-server tool adapters, and the resource/prompt
/// builtin tools so every caller sees the current service (or the current failure) via
/// [`Self::require_connected`].
pub struct ServerEntry {
    pub(crate) server_name: String,
    pub(crate) config: McpServerConfig,
    pub(crate) token_store: Option<TokenStore>,
    pub(crate) client_context: Arc<McpClientContext>,
    pub(crate) state: RwLock<ServerState>,
    pub(crate) reconnect_lock: Mutex<()>,
    /// Optional `InitializeResult.instructions` captured on the first `Connected` transition.
    /// Immutable for the lifetime of the connection per the MCP spec, so reconnects don't reset
    /// it.
    pub(crate) instructions: OnceLock<Option<String>>,
    /// `[mcp].connect_timeout_seconds`, copied here so the request helpers that run outside the
    /// manager can honour it. Without it [`bounded`] fell back to its own constant and a
    /// configured timeout applied to `tools/list` but silently not to `resources/read` or
    /// `prompts/get`.
    pub(crate) request_timeout: OnceLock<std::time::Duration>,
    /// How many tools the last `tools/list` dropped to stay under [`MAX_MCP_TOOLS_PER_SERVER`].
    /// Recorded so the cap is *disclosed* rather than only logged: a tool that vanished between
    /// what the server offers and what meka registered is indistinguishable, from the outside,
    /// from a tool the server never had.
    pub(crate) dropped_tools: std::sync::atomic::AtomicUsize,
}

impl ServerEntry {
    /// The configured per-request bound, or [`DEFAULT_MCP_REQUEST_TIMEOUT`] before the connector
    /// has started (tests, and the window before `start_connector`).
    pub(crate) fn request_timeout(&self) -> std::time::Duration {
        self.request_timeout
            .get()
            .copied()
            .unwrap_or(DEFAULT_MCP_REQUEST_TIMEOUT)
    }

    /// Returns the server's `InitializeResult.instructions` (sanitised + truncated to
    /// [`MAX_MCP_DESCRIPTION_LENGTH`]) if the server advertised one during the handshake.
    pub fn instructions(&self) -> Option<&str> {
        self.instructions.get().and_then(|opt| opt.as_deref())
    }

    /// Snapshot of the current lifecycle state. `Connected` carries an `Arc<McpRunningService>`
    /// which is cheap to clone.
    pub async fn state(&self) -> ServerState {
        self.state.read().await.clone()
    }

    pub(crate) fn server_name(&self) -> &str {
        &self.server_name
    }

    /// `tools/list`, bounded in both time and count.
    ///
    /// Every caller wants the same two guarantees and none of them had both. `list_all_tools`
    /// follows the server's pagination cursor to exhaustion, so a server that answers slowly holds
    /// the caller open with no deadline, and one that keeps handing back cursors grows the tool set
    /// without limit. The connect path already wrapped its discovery in a timeout; the two others
    /// -- a `tools/list_changed` refresh and `meka mcp tools` -- did not.
    pub(crate) async fn list_tools_bounded(
        &self,
        timeout: std::time::Duration,
    ) -> Result<Vec<rmcp::model::Tool>> {
        let peer = self.require_connected().await?;
        let listed = tokio::time::timeout(timeout, peer.list_all_tools())
            .await
            .map_err(|_elapsed| MekaError::McpConnection {
                server_name: self.server_name.clone(),
                message: format!("tools/list timed out after {:?}", timeout),
            })?
            .map_err(|error| MekaError::McpConnection {
                server_name: self.server_name.clone(),
                message: format!("list_tools failed: {}", error),
            })?;

        let advertised = listed.len();
        let kept = cap_advertised_tools(listed, &self.server_name);
        self.dropped_tools.store(
            advertised.saturating_sub(kept.len()),
            std::sync::atomic::Ordering::Relaxed,
        );
        Ok(kept)
    }
}

impl ServerEntry {
    /// Return the live peer if the server is currently `Connected`; otherwise return an error
    /// describing the current lifecycle state. Every tool dispatch / list-call path funnels through
    /// this so the "MCP X not ready" error surfaces at one site.
    pub(crate) async fn require_connected(&self) -> Result<Peer<RoleClient>> {
        let state = self.state.read().await;
        if let ServerState::Connected { service } = &*state {
            return Ok(service.peer().clone());
        }
        // Same wording the unregistered-tool path uses, so one condition reads one way however the
        // agent reached it.
        Err(MekaError::McpConnection {
            server_name: self.server_name.clone(),
            message: state
                .unavailable_reason()
                .unwrap_or_else(|| "is unavailable".to_string()),
        })
    }

    /// Transport-close check used by [`Self::reconnect`]. Returns false if the server isn't
    /// `Connected` (there's nothing to reconnect).
    async fn needs_reconnect(&self) -> bool {
        match &*self.state.read().await {
            ServerState::Connected { service } => service.peer().is_transport_closed(),
            _ => false,
        }
    }

    /// Attempt to reconnect this server with exponential backoff. Serialised via `reconnect_lock`
    /// so concurrent tool calls don't stampede. If another caller already reopened the transport,
    /// returns immediately.
    ///
    /// Schedule: 1s, 2s, 4s, 8s, 16s, capped at 30s, max 5 attempts. Only remote (HTTP) transports
    /// go through backoff; a dead stdio child has to be respawned and retry-after-sleep doesn't
    /// help.
    ///
    /// The connect future itself can be `!Send` for OAuth-authenticated servers (rmcp 1.5 holds a
    /// `form_urlencoded::Serializer` across an await in its auth module, whose `Option<&dyn
    /// Fn(&str) -> Cow<[u8]>>` closure slot is not `Sync`). To keep `Tool::execute`'s `Send` bound
    /// satisfied, we drive the reconnect on a `spawn_blocking` thread using the outer runtime's
    /// `Handle`.
    pub(crate) async fn reconnect(self: &Arc<Self>) -> Result<()> {
        let _guard = self.reconnect_lock.lock().await;

        if !self.needs_reconnect().await {
            return Ok(());
        }

        tracing::warn!(
            "MCP server '{}' transport closed; attempting reconnect",
            self.server_name
        );

        let max_attempts: u32 = match self.config.transport {
            McpTransport::Stdio => 1,
            McpTransport::Http => 5,
        };
        let mut last_error: Option<MekaError> = None;
        for attempt in 0..max_attempts {
            if attempt > 0 {
                // 1s, 2s, 4s, 8s, 16s, capped at 30s.
                let delay_secs = std::cmp::min(30u64, 1u64 << (attempt - 1));
                tokio::time::sleep(std::time::Duration::from_secs(delay_secs)).await;
            }
            let handle = tokio::runtime::Handle::current();
            let server_name = self.server_name.clone();
            let config = self.config.clone();
            let token_store = self.token_store.clone();
            let client_context = Arc::clone(&self.client_context);

            let result = tokio::task::spawn_blocking(move || {
                handle.block_on(connector::connect_server(
                    &server_name,
                    &config,
                    token_store.as_ref(),
                    &client_context,
                ))
            })
            .await;

            match result {
                Ok(Ok(new_service)) => {
                    *self.state.write().await = ServerState::Connected {
                        service: Arc::new(new_service),
                    };
                    tracing::info!(
                        "reconnected to MCP server '{}' on attempt {}",
                        self.server_name,
                        attempt + 1
                    );
                    return Ok(());
                }
                Ok(Err(error)) => {
                    tracing::warn!(
                        "MCP server '{}' reconnect attempt {} failed: {}",
                        self.server_name,
                        attempt + 1,
                        error
                    );
                    last_error = Some(error);
                }
                Err(join_error) => {
                    tracing::warn!(
                        "MCP server '{}' reconnect task join error on attempt {}: {}",
                        self.server_name,
                        attempt + 1,
                        join_error
                    );
                    last_error = Some(MekaError::McpConnection {
                        server_name: self.server_name.clone(),
                        message: format!("reconnect task join error: {}", join_error),
                    });
                }
            }
        }
        Err(last_error.unwrap_or_else(|| MekaError::McpConnection {
            server_name: self.server_name.clone(),
            message: format!("exhausted {} reconnect attempts", max_attempts),
        }))
    }
}

/// Runtime tuning for the background MCP connector. Pulled from `ResolvedConfig` by the binary; the
/// manager uses it directly.
pub struct McpRuntimeConfig {
    /// Per-server wrap around connect + `initialize` + `list_tools`.
    pub connect_timeout: std::time::Duration,
    /// Max concurrent stdio spawns. Defaults to 3 (env `MEKA_MCP_STDIO_CONCURRENCY`).
    pub stdio_concurrency: usize,
    /// Max concurrent HTTP connects. Defaults to 20 (env `MEKA_MCP_HTTP_CONCURRENCY`).
    pub http_concurrency: usize,
}

impl McpRuntimeConfig {
    pub fn from_config(config: &crate::config::ResolvedConfig) -> Self {
        Self {
            connect_timeout: config.mcp_connect_timeout,
            stdio_concurrency: resolve_concurrency_env("MEKA_MCP_STDIO_CONCURRENCY", 3),
            http_concurrency: resolve_concurrency_env("MEKA_MCP_HTTP_CONCURRENCY", 20),
        }
    }
}

/// Parse a positive-integer concurrency override from `env_var`. Falls back to `default` when the
/// variable is unset, unparseable, or zero. Extracted from `McpRuntimeConfig::from_config` so tests
/// can exercise the env-var override path without constructing a full `ResolvedConfig`.
fn resolve_concurrency_env(env_var: &str, default: usize) -> usize {
    std::env::var(env_var)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(default)
}

impl McpClientManager {
    /// Validate configs and build a manager with every non-empty entry
    /// in `Disabled` or `Pending` state. Does NOT spawn any network /
    /// process work; that happens in [`Self::start_connector`].
    /// Callers typically:
    /// 1. `let manager = McpClientManager::prepare(...).await?;`
    /// 2. Register the manager on the `McpClientContext`.
    /// 3. Build the tool registry and call `manager.attach_registry(registry.clone()).await`.
    /// 4. `manager.start_connector(runtime);`
    ///
    /// The split exists so the connector can register MCP tools into attached registries as each
    /// server comes online, without forcing any registry to exist before config validation.
    pub async fn prepare(
        configs: &[McpServerConfig],
        mcp_default_permission: Option<Permission>,
        token_store: Option<TokenStore>,
        client_context: Arc<McpClientContext>,
    ) -> Result<Arc<Self>> {
        let mut servers = HashMap::new();
        let mut pending: Vec<Arc<ServerEntry>> = Vec::new();

        for original_config in configs {
            // Apply env-var substitution (`${VAR}` / `${VAR:-default}`) once, up-front, so the rest
            // of the pipeline sees only resolved values.
            let mut config = original_config.clone();
            let missing = crate::mcp::expand::expand_server_config(&mut config);
            if !missing.is_empty() {
                tracing::warn!(
                    "MCP server '{}': unresolved env vars {:?} left literal in config",
                    config.name,
                    missing
                );
            }

            if config.name.is_empty() {
                return Err(MekaError::McpConnection {
                    server_name: "(empty)".to_string(),
                    message: "server name must not be empty".to_string(),
                });
            }

            // Reject anything that would collide with meka-internal names or our
            // `mcp__<server>__<tool>` namespace separator.
            if crate::mcp::sanitize::is_reserved_server_name(&config.name) {
                return Err(MekaError::McpConnection {
                    server_name: config.name.clone(),
                    message: "server name is reserved (meka, ide, or mcp_*)".to_string(),
                });
            }

            let normalised = crate::mcp::sanitize::normalize_server_name(&config.name);
            if normalised != config.name {
                return Err(MekaError::McpConnection {
                    server_name: config.name.clone(),
                    message: format!(
                        "server name contains characters not allowed in tool prefixes (would normalize to '{}')",
                        normalised
                    ),
                });
            }

            if config.name.contains("__") {
                return Err(MekaError::McpConnection {
                    server_name: config.name.clone(),
                    message: "server name must not contain '__' (reserved as namespace separator)"
                        .to_string(),
                });
            }

            if servers.contains_key(&config.name) {
                return Err(MekaError::McpConnection {
                    server_name: config.name.clone(),
                    message: "duplicate server name".to_string(),
                });
            }

            let is_disabled = config.disabled;
            if is_disabled {
                tracing::info!("MCP server '{}' is disabled in config", config.name);
            }

            let entry = Arc::new(ServerEntry {
                server_name: config.name.clone(),
                config: config.clone(),
                token_store: token_store.clone(),
                client_context: Arc::clone(&client_context),
                state: RwLock::new(if is_disabled {
                    ServerState::Disabled
                } else {
                    ServerState::Pending
                }),
                reconnect_lock: Mutex::new(()),
                instructions: OnceLock::new(),
                request_timeout: OnceLock::new(),
                dropped_tools: std::sync::atomic::AtomicUsize::new(0),
            });
            if !is_disabled {
                pending.push(Arc::clone(&entry));
            }
            servers.insert(config.name.clone(), entry);
        }

        // Initialise the watch with `true` when nothing will ever be pending (all servers disabled,
        // or no servers configured) so callers of `all_ready` / `await_settled` short-circuit
        // immediately. `send` on a Sender with no receivers errors and drops the value, so the
        // initial-value path is the only safe pre-subscription way to publish settled.
        let initial_settled = pending.is_empty();
        let (settled_tx, _) = tokio::sync::watch::channel(initial_settled);
        let manager = Arc::new(Self {
            servers,
            mcp_default_permission,
            settled: settled_tx,
            pending_entries: std::sync::Mutex::new(Some(pending)),
            tools_snapshot: tokio::sync::RwLock::new(HashMap::new()),
            attached_registries: tokio::sync::RwLock::new(Vec::new()),
            connect_timeout: OnceLock::new(),
        });
        Ok(manager)
    }

    /// Update the live snapshot for one server's tools and propagate the change to every attached
    /// registry. Called by the connector when a server reaches `Connected` and by
    /// `on_tool_list_changed` when a server signals a dynamic update.
    ///
    /// The snapshot is what new sessions read at attach time; the propagation keeps existing
    /// sessions in sync without requiring them to re-attach.
    pub async fn update_server_tools(
        &self,
        server_name: &str,
        tools: Vec<Arc<dyn crate::tools::Tool>>,
    ) {
        self.tools_snapshot
            .write()
            .await
            .insert(server_name.to_string(), tools.clone());
        let registries = self.attached_registries.read().await;
        for registry in registries.iter() {
            registry.replace_server_tools(server_name, tools.clone());
        }
    }

    /// Attach a per-session registry to receive live MCP tool updates. Pushes the registry into the
    /// attached list *before* backfilling from the snapshot so any concurrent
    /// [`Self::update_server_tools`] either fans out to the new registry (push happened first) or
    /// has its result observed by the subsequent backfill (push happened second). The opposite
    /// ordering (read snapshot, then push) has a window where an update can land between the
    /// snapshot read and the push, with the registry missing it forever.
    ///
    /// `replace_server_tools` is idempotent, so the double-write when both paths fire is harmless.
    ///
    /// Sessions call this at `session/new` (after building their per-session
    /// [`crate::tools::ToolRegistry`]) and pair it with [`Self::detach_registry`] at
    /// `session/close`.
    /// Takes `&Arc<Self>` rather than `&self` so the registry can be handed a `Weak` back to the
    /// manager. `load_tool` needs it to explain that a name it can't find belongs to a server that
    /// isn't connected, rather than reporting it as unknown.
    pub async fn attach_registry(self: &Arc<Self>, registry: crate::tools::ToolRegistry) {
        registry.set_mcp_manager(Arc::downgrade(self));
        self.attached_registries
            .write()
            .await
            .push(registry.clone());
        let snapshot = self.tools_snapshot.read().await;
        for (server_name, tools) in snapshot.iter() {
            registry.replace_server_tools(server_name, tools.clone());
        }
    }

    /// Detach a registry from MCP tool updates. Identity is by inner `Arc` pointer (see
    /// [`crate::tools::ToolRegistry::same_inner`]) so clones of the same registry match. No-op if
    /// not attached.
    pub async fn detach_registry(&self, registry: &crate::tools::ToolRegistry) {
        let mut registries = self.attached_registries.write().await;
        registries.retain(|other| !crate::tools::ToolRegistry::same_inner(other, registry));
    }

    /// Mark a batch of tool names as deferred across every attached registry. Called after
    /// [`Self::update_server_tools`] when some of the newly-registered adapters are lazy-load only;
    /// the agent's tools-array build then skips them until they're explicitly requested.
    pub async fn mark_deferred_on_attached(&self, tool_names: &[String]) {
        let registries = self.attached_registries.read().await;
        for registry in registries.iter() {
            for name in tool_names {
                registry.mark_deferred(name);
            }
        }
    }

    /// Spawn the background connector. Consumes the `Pending` entry list stashed by
    /// [`Self::prepare`] so subsequent calls are no-ops. Safe to call on managers with no pending
    /// entries.
    ///
    /// The connector writes tool discoveries through [`Self::update_server_tools`], which fans out
    /// to every registry attached via [`Self::attach_registry`]. The caller does not pass a
    /// specific registry: attach yours first, then start the connector.
    /// The bound to put on an MCP request made outside the connector.
    fn request_timeout(&self) -> std::time::Duration {
        self.connect_timeout
            .get()
            .copied()
            .unwrap_or(DEFAULT_MCP_REQUEST_TIMEOUT)
    }

    pub fn start_connector(self: &Arc<Self>, runtime: McpRuntimeConfig) {
        // Recorded before the early return, so a second `start_connector` call still leaves the
        // timeout set for the request paths that read it.
        let _ = self.connect_timeout.set(runtime.connect_timeout);
        // Every entry gets the same bound, so the request helpers that only hold an
        // `Arc<ServerEntry>` honour the configured timeout rather than falling back to the
        // module default.
        for entry in self.servers.values() {
            let _ = entry.request_timeout.set(runtime.connect_timeout);
        }
        let Some(pending) = self
            .pending_entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
        else {
            return;
        };
        let manager = Arc::clone(self);
        let settled = self.settled.clone();
        let mcp_default_permission = self.mcp_default_permission;
        tokio::spawn(async move {
            connector::run_connector(pending, manager, mcp_default_permission, runtime, settled)
                .await;
        });
    }

    /// True when every enabled server has reached a terminal state (`Connected` or `Failed`).
    /// Returns `true` if there are no enabled servers configured. Non-blocking.
    pub fn all_ready(&self) -> bool {
        *self.settled.borrow()
    }

    /// Parks until the background connector finishes processing every enabled server. Returns
    /// immediately if already settled. Safe to call concurrently from multiple turn dispatches.
    pub async fn await_settled(&self) {
        let mut rx = self.settled.subscribe();
        if *rx.borrow() {
            return;
        }
        let _ = rx.wait_for(|done| *done).await;
    }

    /// Snapshot of enabled servers that are not currently `Connected` (still `Pending` or
    /// `Failed`), each paired with whether it is `required`. [`crate::agent`]'s turn gate stops
    /// only for the required ones; the rest are logged at `debug` and the session runs without
    /// them. Deliberately not `warn`: this is consulted on every turn, and a server that is down
    /// stays down, so warning here would print a line before every reply for the life of the
    /// session.
    pub async fn enabled_not_connected(&self) -> Vec<NotConnected> {
        let mut out = Vec::new();
        for (name, entry) in &self.servers {
            let state = entry.state().await;
            match state {
                ServerState::Connected { .. } | ServerState::Disabled => {}
                other => out.push(NotConnected {
                    name: name.clone(),
                    // `required` is settled in `ResolvedConfig::from_cli`, so `None` here can only
                    // mean a config built outside that path (tests, `meka mcp add`); treat it as
                    // optional, matching the default.
                    required: entry.config.required.unwrap_or(false),
                    state: other,
                }),
            }
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    /// Why a `mcp__<server>__<tool>` name isn't callable, when the reason is the server rather
    /// than the tool. `None` means this is not an MCP name, or names a server meka has never heard
    /// of - both genuinely "unknown tool".
    ///
    /// Exists because a server that never connected registers no tools, so its names fall through
    /// to the agent's unknown-tool arm and the model is told the tool does not exist. It does
    /// exist; it is unreachable. An agent told "unknown" reasonably stops asking, which is the
    /// wrong lesson when the server is still connecting or is one `meka mcp reconnect` away. The
    /// prompt-level instructions, a skill, or a resumed conversation can all name a tool whose
    /// server is currently down.
    pub async fn unavailable_tool_reason(&self, tool_name: &str) -> Option<String> {
        let rest = tool_name.strip_prefix("mcp__")?;
        // Server names cannot contain `__` (`sanitize::normalize_server_name`), so the first
        // occurrence splits server from tool.
        let (server_name, _tool) = rest.split_once("__")?;
        let entry = self.servers.get(server_name)?;
        Some(format!(
            "MCP server '{}' {}",
            server_name,
            entry.state().await.unavailable_reason()?
        ))
    }

    pub fn server_entry(&self, server_name: &str) -> Option<Arc<ServerEntry>> {
        self.servers.get(server_name).cloned()
    }

    pub fn server_names(&self) -> Vec<String> {
        self.servers.keys().cloned().collect()
    }

    /// Returns `(server_name, instructions)` pairs for every connected server that advertised an
    /// `InitializeResult.instructions` string during the handshake. Already sanitised and truncated
    /// to [`MAX_MCP_DESCRIPTION_LENGTH`]. Used by the agent loop to splice MCP server instructions
    /// into the per-turn context.
    pub fn server_instructions(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for (name, entry) in &self.servers {
            if let Some(text) = entry.instructions()
                && !text.trim().is_empty()
            {
                out.push((name.clone(), text.to_string()));
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    pub async fn discover_tools_for_server(
        &self,
        server_name: &str,
    ) -> Result<Vec<McpToolAdapter>> {
        let Some(entry) = self.servers.get(server_name) else {
            return Ok(Vec::new());
        };

        let server_config = &entry.config;

        let tools = entry.list_tools_bounded(self.request_timeout()).await?;

        // Collect advertised raw names up-front so we can flag stale `allowed_tools` /
        // `disabled_tools` / `tool_permissions` entries that no longer match anything the server
        // returns.
        let advertised: std::collections::HashSet<&str> =
            tools.iter().map(|t| t.name.as_ref()).collect();
        warn_on_stale_tool_config(server_name, server_config, &advertised);

        let mut adapters = Vec::new();
        for tool in tools {
            let raw_tool_name = tool.name.as_ref().to_string();

            if !tool_is_allowed(server_config, &raw_tool_name) {
                continue;
            }

            // Sanitise the tool's advertised name defensively. It is rare in the wild, but a server
            // returning `my.tool` or anything with Unicode would cause the provider to reject the
            // schema.
            let sanitised_tool_name = crate::mcp::sanitize::normalize_server_name(&raw_tool_name);
            let namespaced_name = format!("mcp__{}__{}", server_name, sanitised_tool_name);

            let raw_description = tool
                .description
                .as_ref()
                .map(|d| d.as_ref().to_string())
                .unwrap_or_default();
            let description = truncate(
                &crate::mcp::sanitize::sanitize_text(&raw_description),
                MAX_MCP_DESCRIPTION_LENGTH,
            );

            let parameters = match serde_json::to_value(&*tool.input_schema) {
                Ok(value) => value,
                Err(error) => {
                    tracing::warn!(
                        "MCP server '{}' tool '{}' has unserializable input schema ({}); \
                         skipping registration",
                        server_name,
                        raw_tool_name,
                        error
                    );
                    continue;
                }
            };

            // Per-tool permission via the layered resolution. Hints come from
            // `tool.annotations.readOnlyHint` as published by the server; the function handles all
            // the precedence rules.
            let permission = resolve_tool_permission(
                server_name,
                &raw_tool_name,
                tool.annotations.as_ref(),
                server_config,
                self.mcp_default_permission,
            )?;

            // Annotations carry permission hints (`readOnlyHint`, `destructiveHint`); silently
            // dropping them on a serialization failure could quietly relax permission resolution.
            // Matches `connector::build_mcp_adapters`, which this path duplicates: a hint lost
            // during a `tools/list_changed` refresh or a sub-agent spawn is exactly as
            // consequential as one lost at startup.
            let annotations = tool
                .annotations
                .as_ref()
                .and_then(|ann| match serde_json::to_value(ann) {
                    Ok(value) => Some(value),
                    Err(error) => {
                        tracing::warn!(
                            "failed to serialize annotations for tool '{}': {}",
                            namespaced_name,
                            error
                        );
                        None
                    }
                });
            let meta = tool
                .meta
                .as_ref()
                .and_then(|m| match serde_json::to_value(m) {
                    Ok(value) => Some(value),
                    Err(error) => {
                        tracing::warn!(
                            "failed to serialize meta for tool '{}': {}",
                            namespaced_name,
                            error
                        );
                        None
                    }
                });
            let title = tool
                .title
                .as_ref()
                .map(|t| crate::mcp::sanitize::sanitize_text(t));

            adapters.push(McpToolAdapter::new(
                namespaced_name,
                raw_tool_name,
                description,
                parameters,
                permission,
                Arc::clone(entry),
                annotations,
                meta,
                title,
            ));
        }

        Ok(adapters)
    }

    /// Install MCP tools onto a freshly-built sub-agent registry. Mirrors the startup wiring at
    /// `main.rs:create_agent_from_config` minus the `start_connector` spawn: only
    /// already-`Connected` servers contribute adapters; Pending / Failed servers are skipped
    /// silently and their tools simply don't appear in the sub-agent's catalogue. The resource /
    /// prompt meta-tools are registered unconditionally; they delegate through
    /// [`ServerEntry::require_connected`] themselves and tolerate non-connected servers until
    /// invoked.
    ///
    /// Mirrors the connector's deferred-mark step so a sub-agent sees the same eager-vs-deferred
    /// tool classification as the parent.
    ///
    /// Idempotent and safe to call concurrently from separate `agent_spawn` invocations operating
    /// on distinct sub-agent registries.
    pub async fn install_tools_on(self: &Arc<Self>, registry: &crate::tools::ToolRegistry) {
        use crate::tools::Tool as _;

        // Sub-agent registries come through here rather than `attach_registry`, so this is where
        // they pick up the back-reference `load_tool` needs to explain an unconnected server. A
        // sub-agent that reaches for a dead server's tool should get the same answer the parent
        // would, not a bare "not registered".
        registry.set_mcp_manager(Arc::downgrade(self));
        crate::tools::mcp_resources::register_all(registry, Arc::clone(self));
        for name in self.server_names() {
            // Skip the round trip entirely rather than discovering and then filtering: a denied
            // server should not even be listed, and `list_all_tools` on a server the sub-agent
            // cannot use is latency the spawn pays for nothing.
            if registry.denials().denies_server(&name) {
                tracing::info!("MCP server '{}' denied for sub-agent registry", name);
                continue;
            }
            let adapters = match self.discover_tools_for_server(&name).await {
                Ok(adapters) => adapters,
                Err(error) => {
                    // Pending / Failed servers fall through `require_connected` as Err; that's
                    // normal, not worth a warn. The sub-agent just won't see this server's tools
                    // until it next runs (and the parent's connector finishes the handshake).
                    tracing::debug!(
                        "MCP server '{}' skipped for sub-agent registry: {}",
                        name,
                        error
                    );
                    continue;
                }
            };
            if adapters.is_empty() {
                continue;
            }
            let deferred_names: Vec<String> = adapters
                .iter()
                .filter(|adapter| {
                    !crate::mcp::tool_should_eager_load(adapter.server_config(), adapter.raw_name())
                })
                .map(|adapter| adapter.definition().name.clone())
                .collect();
            let arc_adapters: Vec<Arc<dyn crate::tools::Tool>> = adapters
                .into_iter()
                .map(|adapter| Arc::new(adapter) as Arc<dyn crate::tools::Tool>)
                .collect();
            registry.replace_server_tools(&name, arc_adapters);
            for deferred in &deferred_names {
                registry.mark_deferred(deferred);
            }
        }
    }

    /// Heal one server on demand, picking the right repair for the state it is actually in.
    ///
    /// The dispatch is the whole point, because the two repairs are not interchangeable.
    /// [`ServerEntry::reconnect`] only swaps the transport, which is all a *previously connected*
    /// server needs: its tool adapters already exist and resolve the live peer at dispatch time.
    /// An entry that failed its initial connect never reached `discover_and_register_tools`, so
    /// healing it that way would leave a server reporting `connected` while exposing no tools at
    /// all. [`connector::connect_one`] re-registers tools and is the right call there.
    ///
    /// A `Failed` server is already being retried in the background with exponential backoff, so
    /// this is an impatience button rather than the only route back: it collapses the wait for an
    /// operator who has just fixed whatever was wrong.
    pub async fn reconnect_server(
        self: &Arc<Self>,
        server_name: &str,
        connect_timeout: std::time::Duration,
    ) -> Result<ServerState> {
        let Some(entry) = self.servers.get(server_name).cloned() else {
            return Err(MekaError::McpConnection {
                server_name: server_name.to_string(),
                message: format!("no MCP server named '{}'", server_name),
            });
        };
        match entry.state().await {
            // Refused, not honoured. `run_connector` owns every `Pending` entry and will connect
            // it without taking `reconnect_lock` (it iterates the list captured at `prepare`
            // time), so firing a second `connect_one` here races it: two child processes for a
            // stdio server, and if the second attempt loses, `record_connect_failure` overwrites a
            // working `Connected` with `Failed`. Startup ordering makes this reachable -- servers
            // past `stdio_concurrency` sit `Pending` for seconds, which is exactly when a
            // dashboard polling `GET /v1/mcp` would see "not connected" and try to help.
            // Defensive, and currently unreachable: the one caller
            // (`server::handlers::info::mcp_reconnect`) reads the state first and answers 200
            // `pending`, because over the wire a refusal reads as "the server failed" when nothing
            // was even attempted. Kept so a future caller cannot race `run_connector` into a
            // second `connect_one` on the same entry -- but the handler's own check is what
            // produces the 200, so do not delete it on the strength of this arm.
            ServerState::Pending => {
                return Err(MekaError::McpConnection {
                    server_name: server_name.to_string(),
                    message: format!(
                        "server '{}' is still being connected; wait for it to settle",
                        server_name
                    ),
                });
            }
            ServerState::Disabled => {
                return Err(MekaError::McpConnection {
                    server_name: server_name.to_string(),
                    message: format!(
                        "server '{}' is disabled in config; enable it with `meka mcp enable {}`",
                        server_name, server_name
                    ),
                });
            }
            ServerState::Connected { .. } => {
                // Reconnect is still the right call: it no-ops unless the transport has actually
                // closed underneath a state that still says `Connected`, which is exactly the case
                // an operator reaching for this button cannot see from outside.
                //
                // Bounded here rather than inside: `ServerEntry::reconnect` retries an HTTP
                // transport up to five times with its own backoff and wraps none of it in a
                // timeout, so an endpoint that blackholes connections would hold this request far
                // past the budget the caller passed in.
                tokio::time::timeout(connect_timeout, entry.reconnect())
                    .await
                    .map_err(|_| MekaError::McpConnection {
                        server_name: server_name.to_string(),
                        message: format!("reconnect did not complete within {:?}", connect_timeout),
                    })??;
            }
            ServerState::Failed { .. } => {
                // Under `reconnect_lock`, which is the same guard `retry_until_connected` takes
                // around this call. Without it two of these requests, or one racing the background
                // retry, drive two `connect_one`s into the same entry: two child processes for a
                // stdio server, both writing `state`, the loser's service orphaned, and
                // `update_server_tools` fanned out twice.
                let _guard = entry.reconnect_lock.lock().await;
                // Re-checked under the lock: whoever held it may have just connected this entry,
                // in which case a second connect would replace a healthy transport for nothing.
                if matches!(entry.state().await, ServerState::Failed { .. }) {
                    connector::connect_one(
                        Arc::clone(&entry),
                        Arc::clone(self),
                        self.mcp_default_permission,
                        connect_timeout,
                    )
                    .await;
                }
            }
        }
        Ok(entry.state().await)
    }

    /// Connect to the named server and list EVERY advertised tool, including ones currently
    /// filtered out by `allowed_tools` / `disabled_tools` so users editing those lists can see what
    /// names are available. Permission is resolved through the normal 5-step chain with the winning
    /// step recorded on each entry.
    ///
    /// Differs from [`Self::discover_tools_for_server`] by (a) not filtering by allow/block lists,
    /// (b) not registering adapters, and (c) capturing the resolution source for display.
    pub async fn list_advertised_tools(&self, server_name: &str) -> Result<Vec<AdvertisedTool>> {
        let Some(entry) = self.servers.get(server_name) else {
            return Err(MekaError::McpConnection {
                server_name: server_name.to_string(),
                message: format!("no MCP server named '{}'", server_name),
            });
        };

        let server_config = &entry.config;
        let tools = entry.list_tools_bounded(self.request_timeout()).await?;

        let mut out = Vec::with_capacity(tools.len());
        for tool in tools {
            let raw_name = tool.name.as_ref().to_string();
            let raw_description = tool
                .description
                .as_ref()
                .map(|d| d.as_ref().to_string())
                .unwrap_or_default();
            let description = truncate(
                &crate::mcp::sanitize::sanitize_text(&raw_description),
                MAX_MCP_DESCRIPTION_LENGTH,
            );
            let (resolved_permission, permission_source) = resolve_tool_permission_with_source(
                server_name,
                &raw_name,
                tool.annotations.as_ref(),
                server_config,
                self.mcp_default_permission,
            )?;
            let allowed = tool_is_allowed(server_config, &raw_name);
            let read_only_hint_declined =
                read_only_hint_was_declined(tool.annotations.as_ref(), permission_source);
            out.push(AdvertisedTool {
                raw_name,
                description,
                resolved_permission,
                permission_source,
                allowed,
                read_only_hint_declined,
            });
        }

        out.sort_by(|a, b| a.raw_name.cmp(&b.raw_name));
        Ok(out)
    }

    /// Shutdown helper for callers that hold the manager through an `Arc`. Just calls
    /// [`Self::shutdown_within`], which needs only `&self`.
    pub async fn shutdown_arc(self: Arc<Self>) {
        self.shutdown_within(SHUTDOWN_BUDGET).await;
    }

    /// [`Self::shutdown`] under a total wall-clock bound.
    ///
    /// The loop is serial and each server can spend up to `CLOSE_TIMEOUT`, so an exit's cost scales
    /// with the number of servers that refuse to answer. Every caller is on its way out and some of
    /// them are being timed by something else - systemd, a container runtime, a user holding a
    /// terminal - so the whole teardown gets one budget rather than each server getting its own.
    /// Overrunning it is not an error: the remaining servers fall to rmcp's drop guards, which is
    /// exactly where they were before any of this ran.
    pub async fn shutdown_within(&self, budget: std::time::Duration) {
        if tokio::time::timeout(budget, self.shutdown()).await.is_err() {
            tracing::warn!(
                "MCP shutdown exceeded {:?}; the servers still closing are left to their drop \
                 guards, and a stdio child that ignores both may outlive this process",
                budget
            );
        }
    }

    /// Close every connected server, in place.
    ///
    /// Takes `&self` deliberately. This used to consume `self`, so callers had to `try_unwrap` an
    /// `Arc<Self>` first and the whole graceful path was skipped whenever anything else still held
    /// a reference - which was always, because the manager holds the tool registries it serves
    /// (`attached_registries`) and those registries hold the six `mcp_resource_*` / `mcp_prompt_*`
    /// tools, each of which holds an `Arc` back to the manager. Sole ownership was unreachable by
    /// construction, so `close_with_timeout` never ran and stdio children were left to rmcp's drop
    /// guard, which spawns onto a runtime already tearing down.
    ///
    /// The service is taken out from under each entry's `state` lock rather than by owning the
    /// entry, so a `ServerEntry` clone held by an in-flight call doesn't block teardown either. The
    /// entry is left `Disabled` so a tool call arriving during teardown is refused rather than
    /// handed a service that is closing.
    ///
    /// This does *not* stop a racing connect. `connect_one` and `reconnect` write `Connected`
    /// unconditionally, and a `Pending` entry is left `Pending` here because it has nothing to
    /// close - so a connector still working through its queue at exit can bring a server up behind
    /// this loop and leave that child running. Shutting the connector down first is the fix, and is
    /// not attempted here.
    pub async fn shutdown(&self) {
        /// Max time to wait for in-flight tool calls to complete before we drop the shared service
        /// Arc and let the drop-guard cancel it.
        const SHUTDOWN_GRACE: std::time::Duration = std::time::Duration::from_millis(2000);
        /// Max time to wait for `RunningService::close` to finish after the shared references are
        /// released.
        const CLOSE_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(2000);

        for (server_name, entry) in &self.servers {
            // Only Connected entries have a service to close; Pending / Failed / Disabled entries
            // are tear-down no-ops. Taken under the write lock so a concurrent reconnect can't
            // hand this loop a service it is about to replace.
            let service = {
                let mut state = entry.state.write().await;
                match std::mem::replace(&mut *state, ServerState::Disabled) {
                    ServerState::Connected { service } => service,
                    other => {
                        *state = other;
                        continue;
                    }
                }
            };

            // Intended to let in-flight tool calls finish before the transport goes. It does not
            // currently wait: dispatch goes through `require_connected`, which clones
            // `service.peer()`, and rmcp 3.1's `Peer` holds channels rather than an
            // `Arc<RunningService>` - so the count is already 1 and the loop exits immediately.
            // Kept because the shape is right and the fix belongs in what dispatch
            // holds, not here; it was unreachable before shutdown ran at all, and is
            // merely ineffective now.
            let deadline = tokio::time::Instant::now() + SHUTDOWN_GRACE;
            while Arc::strong_count(&service) > 1 && tokio::time::Instant::now() < deadline {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }

            match Arc::try_unwrap(service) {
                Ok(mut owned_service) => {
                    match owned_service.close_with_timeout(CLOSE_TIMEOUT).await {
                        Ok(Some(_)) => {}
                        Ok(None) => {
                            tracing::warn!(
                                "MCP server '{}' shutdown timed out after {:?}",
                                server_name,
                                CLOSE_TIMEOUT
                            );
                        }
                        Err(error) => {
                            tracing::warn!(
                                "failed to shut down MCP server '{}': {}",
                                server_name,
                                error
                            );
                        }
                    }
                }
                Err(_arc) => {
                    tracing::debug!(
                        "MCP server '{}' still had in-flight calls after {:?} grace; \
                         relying on drop guard for cleanup",
                        server_name,
                        SHUTDOWN_GRACE
                    );
                }
            }
        }
    }
}

/// Decide whether a tool advertised by a server should be registered. Applies `allowed_tools`
/// (restrict-in, when set and non-empty) then `disabled_tools` (always-remove). Both fields can
/// coexist: the allow-list acts as a restriction, and the block-list subtracts from whatever
/// remains. A tool passes iff it survives both checks.
pub(crate) fn tool_is_allowed(server_config: &McpServerConfig, tool_raw_name: &str) -> bool {
    if let Some(allow) = server_config.allowed_tools.as_deref()
        && !allow.is_empty()
        && !allow.iter().any(|t| t == tool_raw_name)
    {
        return false;
    }
    if let Some(deny) = server_config.disabled_tools.as_deref()
        && deny.iter().any(|t| t == tool_raw_name)
    {
        return false;
    }
    true
}

/// Whether the given raw tool name is in this server's
/// [`eager_load_tools`][McpServerConfig::eager_load_tools] list. Mirrors [`tool_is_allowed`]'s
/// shape. When true, the registration sites skip `mark_deferred` so the tool ships in the cacheable
/// tools-array prefix from the first turn instead of after a `load_tool` round-trip.
pub(crate) fn tool_should_eager_load(server_config: &McpServerConfig, tool_raw_name: &str) -> bool {
    server_config
        .eager_load_tools
        .as_ref()
        .is_some_and(|list| list.iter().any(|n| n == tool_raw_name))
}

/// Emit a `warn!` once per entry in `allowed_tools` / `disabled_tools` / `eager_load_tools` /
/// `tool_permissions` that doesn't match anything the server currently advertises. Users get a
/// visible heads-up without failing the connect. Tool lists can change between server releases,
/// and forcing a hard error on every rename would be hostile. Also warns on the disabled∩eager-load
/// overlap, which is meaningless (disabled tools aren't registered, so eager-loading them is a
/// no-op).
pub(crate) fn warn_on_stale_tool_config(
    server_name: &str,
    server_config: &McpServerConfig,
    advertised: &std::collections::HashSet<&str>,
) {
    if let Some(allow) = server_config.allowed_tools.as_deref() {
        for name in allow {
            if !advertised.contains(name.as_str()) {
                tracing::warn!(
                    "MCP server '{}': allowed_tools entry '{}' doesn't match any advertised tool",
                    server_name,
                    name
                );
            }
        }
    }
    if let Some(deny) = server_config.disabled_tools.as_deref() {
        for name in deny {
            if !advertised.contains(name.as_str()) {
                tracing::warn!(
                    "MCP server '{}': disabled_tools entry '{}' doesn't match any advertised tool",
                    server_name,
                    name
                );
            }
        }
    }
    if let Some(eager) = server_config.eager_load_tools.as_deref() {
        let disabled = server_config.disabled_tools.as_deref().unwrap_or(&[]);
        for name in eager {
            if !advertised.contains(name.as_str()) {
                tracing::warn!(
                    "MCP server '{}': eager_load_tools entry '{}' doesn't match any advertised tool",
                    server_name,
                    name
                );
            }
            if disabled.iter().any(|d| d == name) {
                tracing::warn!(
                    "MCP server '{}': eager_load_tools entry '{}' is also in disabled_tools \
                     (the tool won't be registered at all, so eager-loading it is a no-op)",
                    server_name,
                    name
                );
            }
        }
    }
    if let Some(perms) = server_config.tool_permissions.as_ref() {
        for key in perms.keys() {
            if !advertised.contains(key.as_str()) {
                tracing::warn!(
                    "MCP server '{}': tool_permissions key '{}' doesn't match any advertised tool",
                    server_name,
                    key
                );
            }
        }
    }
}

/// Resolve the required permission for a single MCP tool. Applies the
/// layered policy documented in `docs/book/src/configuration/config-file.md`:
///
/// 1. `server.tool_permissions[tool]`: per-tool user override.
/// 2. `server.permission`: server-level user override.
/// 3. `tool.annotations.readOnlyHint` advertised by the server: `true` → Read, `false` → Write. The
///    `true` half is skipped when the server sets `trust_read_only_hint = false`.
/// 4. `mcp.default_permission`: global fallback when no hint exists.
/// 5. Hardcoded `Write`: ultimate strict fallback.
///
/// User config at steps 1/2 always beats the server's hints. Hints beat the global fallback so a
/// `readOnlyHint = false` destructive tool isn't silently promoted to Read just because the user
/// opted into a lenient global default.
pub(crate) fn resolve_tool_permission(
    server_name: &str,
    tool_raw_name: &str,
    tool_annotations: Option<&rmcp::model::ToolAnnotations>,
    server_config: &McpServerConfig,
    mcp_default: Option<Permission>,
) -> Result<Permission> {
    resolve_tool_permission_with_source(
        server_name,
        tool_raw_name,
        tool_annotations,
        server_config,
        mcp_default,
    )
    .map(|(permission, _)| permission)
}

/// Identifies which step of the 5-step resolution chain produced a tool's permission. Used by `meka
/// mcp tools <name>` so users can see which knob is driving each tool's classification when editing
/// allow/block lists or per-tool overrides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionSource {
    ToolOverride,
    ServerOverride,
    ReadOnlyHint,
    GlobalDefault,
    Fallback,
}

impl PermissionSource {
    /// Short human label matching the config keys users would edit.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ToolOverride => "tool_permission",
            Self::ServerOverride => "server_permission",
            Self::ReadOnlyHint => "readOnlyHint",
            Self::GlobalDefault => "default_permission",
            Self::Fallback => "fallback",
        }
    }
}

/// A tool advertised by an MCP server, paired with the resolved permission and the source step of
/// the resolution chain. Returned by [`McpClientManager::list_advertised_tools`] and printed by
/// `meka mcp tools <server>`.
pub struct AdvertisedTool {
    /// Raw name as advertised by the server. Use this value in `allowed_tools` / `disabled_tools`
    /// / `tool_permissions` config.
    pub raw_name: String,
    /// Sanitised + truncated description (same pipeline as registered tools).
    pub description: String,
    /// Output of the 5-step permission resolution.
    pub resolved_permission: Permission,
    /// Which step of the chain won.
    pub permission_source: PermissionSource,
    /// `false` if currently filtered out by `allowed_tools` / `disabled_tools`, i.e. the agent
    /// would never see this tool.
    pub allowed: bool,
    /// The server advertised `readOnlyHint: true` and `trust_read_only_hint = false` withheld it,
    /// so resolution fell through to the steps below.
    ///
    /// Carried separately because [`Self::permission_source`] names only what *won*, and a
    /// declined hint by definition did not. Without it the one thing the setting exists to do
    /// is invisible at the one place a user checks it: `meka mcp tools` showed
    /// `default_permission` either way, so a server advertising no hint and a server whose
    /// hint was refused read identically.
    pub read_only_hint_declined: bool,
}

/// Same resolution as [`resolve_tool_permission`] but also returns which step of the chain fired,
/// so `meka mcp tools` can show the user exactly why a given tool has its current permission.
fn resolve_tool_permission_with_source(
    server_name: &str,
    tool_raw_name: &str,
    tool_annotations: Option<&rmcp::model::ToolAnnotations>,
    server_config: &McpServerConfig,
    mcp_default: Option<Permission>,
) -> Result<(Permission, PermissionSource)> {
    // 1. Per-tool override.
    if let Some(map) = &server_config.tool_permissions
        && let Some(raw) = map.get(tool_raw_name)
    {
        let permission = raw
            .parse::<Permission>()
            .map_err(|_| MekaError::McpConnection {
                server_name: server_name.to_string(),
                message: format!(
                    "invalid tool_permissions['{}'] = '{}': expected \
                     'none', 'read', 'ask', or 'write'",
                    tool_raw_name, raw
                ),
            })?;
        return Ok((permission, PermissionSource::ToolOverride));
    }
    // 2. Server-level override.
    if let Some(raw) = server_config.permission.as_deref() {
        let permission = raw
            .parse::<Permission>()
            .map_err(|_| MekaError::McpConnection {
                server_name: server_name.to_string(),
                message: format!(
                    "invalid permission '{}': expected 'none', 'read', \
                     'ask', or 'write'",
                    raw
                ),
            })?;
        return Ok((permission, PermissionSource::ServerOverride));
    }
    // 3. Server-advertised readOnlyHint.
    //
    // The two directions are not symmetric, so they are gated differently. A hint of `false` only
    // ever *raises* the requirement to Write, so believing it costs nothing and it is always
    // honoured. A hint of `true` *lowers* the requirement to Read, and that is the direction in
    // which a wrong or dishonest hint matters: MCP tools run in the server's own process with no
    // sandbox, so a tool wrongly classified Read can write the user's tree while meka sits at
    // `read`. `trust_read_only_hint = false` withholds exactly that, leaving the hint advisory for
    // display and dropping the tool through to the strict fallback -- past the global default, for
    // the reason step 4 gives.
    let mut hint_declined = false;
    if let Some(annotations) = tool_annotations
        && let Some(hint) = annotations.read_only_hint
    {
        if !hint {
            return Ok((Permission::Write, PermissionSource::ReadOnlyHint));
        }
        if server_config.trust_read_only_hint.unwrap_or(true) {
            return Ok((Permission::Read, PermissionSource::ReadOnlyHint));
        }
        hint_declined = true;
    }
    // 4. Global [mcp].default_permission -- but not for a hint this server was refused.
    //
    // A declined hint skips straight to the strict fallback, because otherwise the knob is
    // display-only in exactly the configuration where it matters most. `default_permission =
    // "read"` sent a refused `readOnlyHint: true` back to `Read` here, which is bit-for-bit the
    // outcome of trusting it: the tool registers at `Read` and dispatches unapproved at
    // `--permission read`. `"none"` was worse, since a required level of `None` is permitted at
    // every tier. Either way the user set a per-server flag saying "do not take this server's word
    // for it" and a global convenience setting quietly took its word for it anyway.
    //
    // Per-server beats global, which is the direction the rest of this chain already runs: steps 1
    // and 2 are the per-server `tool_permissions` / `permission` overrides and they are checked
    // above. Those remain the way to put a distrusted server's tool back within reach of `read`.
    if !hint_declined && let Some(permission) = mcp_default {
        return Ok((permission, PermissionSource::GlobalDefault));
    }
    // 5. Hardcoded strict fallback.
    Ok((Permission::Write, PermissionSource::Fallback))
}

/// Whether a server offered `readOnlyHint: true` and resolution refused it.
///
/// A hint that *won* is reported as [`PermissionSource::ReadOnlyHint`], so anything else means the
/// hint was present and something below it decided. Only the `true` direction can be declined:
/// `readOnlyHint: false` only ever raises the requirement, so it is always honoured and always wins
/// when present.
fn read_only_hint_was_declined(
    tool_annotations: Option<&rmcp::model::ToolAnnotations>,
    source: PermissionSource,
) -> bool {
    tool_annotations
        .and_then(|annotations| annotations.read_only_hint)
        .unwrap_or(false)
        && !matches!(source, PermissionSource::ReadOnlyHint)
}

/// Shared context threaded into every [`handler::MekaClientHandler`] so notification callbacks and
/// server-to-client requests (`elicitation/create`, `tools/list_changed`) can reach the rest of the
/// agent. The manager slot is optional because the handler is constructed before the manager
/// exists; it is filled in post-construction via [`McpClientContext::set_manager`].
#[derive(Default)]
pub struct McpClientContext {
    /// Weak reference to the MCP manager so the notification callback can rediscover tools without
    /// creating an Arc cycle through the handler. Tool registry updates flow through the manager's
    /// attached registries; no per-context registry slot is needed.
    manager: OnceLock<Weak<McpClientManager>>,
}

impl McpClientContext {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn set_manager(&self, manager: Weak<McpClientManager>) {
        if self.manager.set(manager).is_err() {
            tracing::warn!("MCP client context: manager already set");
        }
    }

    pub(crate) fn manager(&self) -> Option<Weak<McpClientManager>> {
        self.manager.get().cloned()
    }
}

/// Truncate a string to `max_chars` Unicode scalar values, appending an ellipsis marker if
/// truncation occurred. Operates on `char` boundaries so the result is always valid UTF-8.
pub fn truncate(text: &str, max_chars: usize) -> String {
    let mut byte_end = text.len();
    for (count, (idx, _)) in text.char_indices().enumerate() {
        if count == max_chars {
            byte_end = idx;
            break;
        }
    }
    if byte_end < text.len() {
        let mut truncated = String::with_capacity(byte_end + 3);
        truncated.push_str(&text[..byte_end]);
        truncated.push_str("...");
        truncated
    } else {
        text.to_string()
    }
}

/// Bound one MCP round-trip in time and against the turn's cancellation.
///
/// Every helper below is a request to a process meka does not control, over a transport that can
/// accept and then go quiet. Without this a server that never answers parks the tool call, and with
/// it the turn, for the life of the process -- and pressing stop did not reach it either, because
/// the token the tool was handed went unused. `call_tool_once` has had both since it was written;
/// the resource and prompt helpers were the asymmetry.
///
/// `biased` so a token already fired wins over a response arriving in the same instant: once the
/// user has stopped the turn, the answer is not wanted whichever got there first.
async fn bounded<T>(
    entry: &Arc<ServerEntry>,
    what: &str,
    cancellation: &CancellationToken,
    work: impl std::future::Future<Output = Result<T>>,
) -> Result<T> {
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => Err(MekaError::Interrupted),
        outcome = tokio::time::timeout(entry.request_timeout(), work) => match outcome {
            Ok(result) => result,
            Err(_elapsed) => Err(MekaError::McpConnection {
                server_name: entry.server_name.clone(),
                message: format!("{} timed out after {:?}", what, entry.request_timeout()),
            }),
        },
    }
}

/// List all resources advertised by a server. Returned verbatim from the current peer; no caching
/// is done here.
pub async fn list_resources(
    entry: &Arc<ServerEntry>,
    cancellation: &CancellationToken,
) -> Result<Vec<Resource>> {
    bounded(entry, "resources/list", cancellation, async {
        let peer = entry.require_connected().await?;
        match peer.list_all_resources().await {
            Ok(resources) => Ok(resources),
            Err(ServiceError::TransportClosed) => {
                entry.reconnect().await?;
                let peer = entry.require_connected().await?;
                peer.list_all_resources()
                    .await
                    .map_err(|error| MekaError::McpConnection {
                        server_name: entry.server_name.clone(),
                        message: format!("list_resources failed: {}", error),
                    })
            }
            Err(error) => Err(MekaError::McpConnection {
                server_name: entry.server_name.clone(),
                message: format!("list_resources failed: {}", error),
            }),
        }
    })
    .await
}

pub async fn read_resource(
    entry: &Arc<ServerEntry>,
    uri: String,
    cancellation: &CancellationToken,
) -> Result<ReadResourceResult> {
    bounded(entry, "resources/read", cancellation, async {
        let params = ReadResourceRequestParams::new(uri.clone());
        let peer = entry.require_connected().await?;
        match peer.read_resource(params.clone()).await {
            Ok(result) => Ok(result),
            Err(ServiceError::TransportClosed) => {
                entry.reconnect().await?;
                let peer = entry.require_connected().await?;
                peer.read_resource(params)
                    .await
                    .map_err(|error| MekaError::McpConnection {
                        server_name: entry.server_name.clone(),
                        message: format!("read_resource({}) failed: {}", uri, error),
                    })
            }
            Err(error) => Err(MekaError::McpConnection {
                server_name: entry.server_name.clone(),
                message: format!("read_resource({}) failed: {}", uri, error),
            }),
        }
    })
    .await
}

pub async fn list_prompts(
    entry: &Arc<ServerEntry>,
    cancellation: &CancellationToken,
) -> Result<Vec<Prompt>> {
    bounded(entry, "prompts/list", cancellation, async {
        let peer = entry.require_connected().await?;
        match peer.list_all_prompts().await {
            Ok(prompts) => Ok(prompts),
            Err(ServiceError::TransportClosed) => {
                entry.reconnect().await?;
                let peer = entry.require_connected().await?;
                peer.list_all_prompts()
                    .await
                    .map_err(|error| MekaError::McpConnection {
                        server_name: entry.server_name.clone(),
                        message: format!("could not list prompts: {}", error),
                    })
            }
            Err(error) => Err(MekaError::McpConnection {
                server_name: entry.server_name.clone(),
                message: format!("could not list prompts: {}", error),
            }),
        }
    })
    .await
}

// rmcp 3.1 deprecates `subscribe` / `unsubscribe` in favour of `Peer::listen`, but that is a
// 2026-07-28 mechanism and meka negotiates 2025-11-25: it implements no `get_info`, so it takes
// `ClientInfo::default()`, whose protocol version is rmcp's own `ProtocolVersion::LATEST`. rmcp
// gates its 2026-07-28 features on the *server's* reported version being at least that, so at the
// version meka actually speaks `resources/subscribe` is the mechanism rather than a fallback, and
// reaching for `listen` here would ask servers for a method they never negotiated.
//
// Switching is not a local edit either: notifications routed to a `Subscription` are deliberately
// not delivered through `ClientHandler`, so `on_resource_updated` would stop firing and the
// updates `mcp_resource_poll` reads would need a per-server pump task feeding them instead. That
// belongs with the move to 2026-07-28, not ahead of it.
#[allow(deprecated)]
pub async fn subscribe_resource(
    entry: &Arc<ServerEntry>,
    uri: String,
    cancellation: &CancellationToken,
) -> Result<()> {
    bounded(entry, "resources/subscribe", cancellation, async {
        let peer = entry.require_connected().await?;
        let params = rmcp::model::SubscribeRequestParams::new(uri.clone());
        peer.subscribe(params)
            .await
            .map_err(|error| MekaError::McpConnection {
                server_name: entry.server_name.clone(),
                message: format!("subscribe({}) failed: {}", uri, error),
            })
    })
    .await
}

#[allow(deprecated)]
pub async fn unsubscribe_resource(
    entry: &Arc<ServerEntry>,
    uri: String,
    cancellation: &CancellationToken,
) -> Result<()> {
    bounded(entry, "resources/unsubscribe", cancellation, async {
        let peer = entry.require_connected().await?;
        let params = rmcp::model::UnsubscribeRequestParams::new(uri.clone());
        peer.unsubscribe(params)
            .await
            .map_err(|error| MekaError::McpConnection {
                server_name: entry.server_name.clone(),
                message: format!("unsubscribe({}) failed: {}", uri, error),
            })
    })
    .await
}

pub async fn get_prompt(
    entry: &Arc<ServerEntry>,
    name: String,
    arguments: Option<serde_json::Map<String, serde_json::Value>>,
    cancellation: &CancellationToken,
) -> Result<GetPromptResult> {
    bounded(entry, "prompts/get", cancellation, async {
        let mut params = GetPromptRequestParams::new(name.clone());
        params.arguments = arguments;

        let peer = entry.require_connected().await?;
        match peer.get_prompt(params.clone()).await {
            Ok(result) => Ok(result),
            Err(ServiceError::TransportClosed) => {
                entry.reconnect().await?;
                let peer = entry.require_connected().await?;
                peer.get_prompt(params)
                    .await
                    .map_err(|error| MekaError::McpConnection {
                        server_name: entry.server_name.clone(),
                        message: format!("could not render prompt '{}': {}", name, error),
                    })
            }
            Err(error) => Err(MekaError::McpConnection {
                server_name: entry.server_name.clone(),
                message: format!("could not render prompt '{}': {}", name, error),
            }),
        }
    })
    .await
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn bare_server_config(name: &str) -> McpServerConfig {
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

    fn annotations_with_read_only_hint(hint: Option<bool>) -> rmcp::model::ToolAnnotations {
        // `ToolAnnotations` is `#[non_exhaustive]`; use the builder.
        let mut ann = rmcp::model::ToolAnnotations::new();
        ann.read_only_hint = hint;
        ann
    }

    #[test]
    fn resolve_tool_permission_prefers_per_tool_override() {
        let mut server = bare_server_config("s");
        server.permission = Some("write".into());
        let mut per_tool = std::collections::HashMap::new();
        per_tool.insert("search".to_string(), "read".to_string());
        server.tool_permissions = Some(per_tool);

        // Per-tool override wins even when both the server default AND the server's hint disagree.
        let annotations = annotations_with_read_only_hint(Some(false));
        let resolved = resolve_tool_permission(
            "s",
            "search",
            Some(&annotations),
            &server,
            Some(Permission::Write),
        )
        .expect("should resolve");
        assert_eq!(resolved, Permission::Read);
    }

    #[test]
    fn resolve_tool_permission_falls_through_to_server_level() {
        let mut server = bare_server_config("s");
        server.permission = Some("read".into());
        // Server level beats the hint.
        let annotations = annotations_with_read_only_hint(Some(false));
        let resolved = resolve_tool_permission(
            "s",
            "any",
            Some(&annotations),
            &server,
            Some(Permission::Write),
        )
        .expect("should resolve");
        assert_eq!(resolved, Permission::Read);
    }

    #[test]
    fn resolve_tool_permission_honours_read_only_hint() {
        let server = bare_server_config("s");
        // readOnlyHint = true → Read, even though the global default would otherwise be Write.
        let annotations = annotations_with_read_only_hint(Some(true));
        let resolved = resolve_tool_permission(
            "s",
            "search",
            Some(&annotations),
            &server,
            Some(Permission::Write),
        )
        .expect("should resolve");
        assert_eq!(resolved, Permission::Read);

        // readOnlyHint = false → Write, even though the global default is the lenient Read.
        let annotations = annotations_with_read_only_hint(Some(false));
        let resolved = resolve_tool_permission(
            "s",
            "write-page",
            Some(&annotations),
            &server,
            Some(Permission::Read),
        )
        .expect("should resolve");
        assert_eq!(resolved, Permission::Write);
    }

    /// `trust_read_only_hint = false` is the knob that keeps an unsandboxed MCP tool out of the
    /// `read` tier on a server whose self-classification the user does not accept. Without it, a
    /// server advertising `readOnlyHint: true` for a tool that in fact writes gets to write while
    /// The advertised-tool bound is a residency bound: the list is held per session and re-sent in
    /// every request's tools array, so an unbounded one is a per-turn cost, not a one-off.
    #[test]
    fn an_over_advertising_server_is_capped_at_the_tool_ceiling() {
        let under: Vec<usize> = (0..MAX_MCP_TOOLS_PER_SERVER).collect();
        assert_eq!(
            cap_advertised_tools(under.clone(), "s").len(),
            MAX_MCP_TOOLS_PER_SERVER,
            "a server exactly at the ceiling keeps everything"
        );

        let over: Vec<usize> = (0..MAX_MCP_TOOLS_PER_SERVER + 250).collect();
        let capped = cap_advertised_tools(over, "s");
        assert_eq!(capped.len(), MAX_MCP_TOOLS_PER_SERVER);
        assert_eq!(
            capped.first().copied(),
            Some(0),
            "the kept ones are the first, not an arbitrary slice"
        );
    }

    /// meka sits at `read`, because MCP tools run in the server's process with no sandbox.
    #[test]
    fn a_declined_read_only_hint_cannot_reach_the_read_tier() {
        let mut server = bare_server_config("s");
        server.trust_read_only_hint = Some(false);
        let annotations = annotations_with_read_only_hint(Some(true));

        // Every global default, including the two that are themselves at or below `read`.
        //
        // Those two are the whole point. `default_permission = "read"` used to send a refused hint
        // straight back to `Read`, which is bit-for-bit what trusting it would have done, so the
        // knob changed nothing but a label. `"none"` was worse: a required level of `None` is
        // permitted at every tier, so the tool ran even at `--permission none`. This test asserted
        // the invariant in its name while only ever passing `Some(Write)` and `None`.
        for default in [
            Some(Permission::Write),
            Some(Permission::Ask),
            Some(Permission::Read),
            Some(Permission::None),
            None,
        ] {
            let resolved =
                resolve_tool_permission("s", "search", Some(&annotations), &server, default)
                    .expect("should resolve");
            assert_eq!(
                resolved,
                Permission::Write,
                "a declined hint must reach the strict fallback whatever the global default is, \
                 but with {:?} it resolved to {}",
                default,
                resolved
            );
        }
    }

    /// The knob is per-server, so it has to beat the global default; the per-server *overrides*
    /// still beat it in turn.
    ///
    /// Without this second half the fix would be a lockout: a distrusted server's tool could never
    /// be brought back within reach of `read` at all. `tool_permissions` and `permission` are
    /// checked before the hint, and they remain the documented escape hatch.
    #[test]
    fn an_explicit_override_still_outranks_a_declined_hint() {
        let mut server = bare_server_config("s");
        server.trust_read_only_hint = Some(false);
        let annotations = annotations_with_read_only_hint(Some(true));

        server.tool_permissions = Some(std::collections::HashMap::from([(
            "search".to_string(),
            "read".to_string(),
        )]));
        let resolved = resolve_tool_permission(
            "s",
            "search",
            Some(&annotations),
            &server,
            Some(Permission::Read),
        )
        .expect("should resolve");
        assert_eq!(
            resolved,
            Permission::Read,
            "an explicit per-tool override is how a distrusted server's tool is re-admitted"
        );
    }

    /// Declining the hint withholds only the direction that *lowers* the requirement. A
    /// `readOnlyHint: false` can only ever raise it, so believing it costs nothing and it stays
    /// honoured, including the attribution, so `meka mcp tools` still explains the classification.
    #[test]
    fn a_declined_read_only_hint_still_honours_the_raising_direction() {
        let mut server = bare_server_config("s");
        server.trust_read_only_hint = Some(false);
        let annotations = annotations_with_read_only_hint(Some(false));

        let (resolved, source) = resolve_tool_permission_with_source(
            "s",
            "write-page",
            Some(&annotations),
            &server,
            Some(Permission::Read),
        )
        .expect("should resolve");
        assert_eq!(resolved, Permission::Write);
        assert_eq!(source, PermissionSource::ReadOnlyHint);
    }

    /// Every MCP round-trip has to answer to the turn's cancellation and to a clock.
    ///
    /// A server can accept a request and then go quiet, and the resource and prompt helpers awaited
    /// that unconditionally: the tool call parked, the turn with it, and pressing stop did not
    /// reach it either, because the token the tool was handed went unused. `call_tool_once` had
    /// both bounds from the start; these six were the asymmetry.
    #[tokio::test]
    async fn an_mcp_round_trip_answers_to_cancellation_and_to_the_clock() {
        let entry = pending_entry("quiet-srv", McpTransport::Http);

        let cancelled = CancellationToken::new();
        cancelled.cancel();
        let outcome = bounded(&entry, "resources/read", &cancelled, async {
            std::future::pending::<Result<()>>().await
        })
        .await;
        assert!(
            matches!(outcome, Err(MekaError::Interrupted)),
            "a stopped turn must not wait on the server: {outcome:?}",
        );

        // And the clock, for a server nobody stopped waiting on.
        tokio::time::pause();
        let live = CancellationToken::new();
        let waiting = tokio::spawn({
            let entry = Arc::clone(&entry);
            async move {
                bounded(&entry, "resources/read", &live, async {
                    std::future::pending::<Result<()>>().await
                })
                .await
            }
        });
        tokio::time::advance(DEFAULT_MCP_REQUEST_TIMEOUT + std::time::Duration::from_secs(1)).await;
        let outcome = waiting.await.expect("join");
        assert!(
            matches!(outcome, Err(MekaError::McpConnection { .. })),
            "an unanswered request must end: {outcome:?}",
        );
    }

    /// A user who sets `trust_read_only_hint = false` has to be able to see what it moved.
    ///
    /// `meka mcp tools` reports the step that *won*, and a declined hint by definition did not, so
    /// a server advertising no hint at all and a server whose hint was refused both printed
    /// `default_permission` and read identically. The one place the setting is observable showed
    /// nothing of it.
    #[test]
    fn a_declined_read_only_hint_is_reported_as_declined() {
        let offered = annotations_with_read_only_hint(Some(true));

        assert!(
            read_only_hint_was_declined(Some(&offered), PermissionSource::GlobalDefault),
            "offered and outvoted by the global default is the case this exists for",
        );
        assert!(
            read_only_hint_was_declined(Some(&offered), PermissionSource::Fallback),
            "and the same when there is no global default to fall to",
        );
        assert!(
            !read_only_hint_was_declined(Some(&offered), PermissionSource::ReadOnlyHint),
            "a hint that won was not declined",
        );
        assert!(
            !read_only_hint_was_declined(None, PermissionSource::GlobalDefault),
            "and a server that offered nothing has nothing to decline",
        );
        assert!(
            !read_only_hint_was_declined(
                Some(&annotations_with_read_only_hint(Some(false))),
                PermissionSource::GlobalDefault,
            ),
            "only the lowering direction can be declined; `false` always wins when present",
        );
    }

    /// Declining the hint must not also discard the user's own overrides: steps 1 and 2 sit above
    /// the hint in the chain and are the documented way to make a distrusted server's tool usable.
    #[test]
    fn a_declined_read_only_hint_leaves_user_overrides_in_charge() {
        let mut server = bare_server_config("s");
        server.trust_read_only_hint = Some(false);
        server.tool_permissions = Some(
            [("search".to_string(), "read".to_string())]
                .into_iter()
                .collect(),
        );
        let annotations = annotations_with_read_only_hint(Some(true));

        let (resolved, source) =
            resolve_tool_permission_with_source("s", "search", Some(&annotations), &server, None)
                .expect("should resolve");
        assert_eq!(resolved, Permission::Read);
        assert_eq!(source, PermissionSource::ToolOverride);
    }

    #[test]
    fn resolve_tool_permission_falls_through_to_mcp_default() {
        let server = bare_server_config("s");
        // No user overrides, no hint → fall through to `[mcp].default`.
        let resolved = resolve_tool_permission("s", "any", None, &server, Some(Permission::Read))
            .expect("should resolve");
        assert_eq!(resolved, Permission::Read);
    }

    #[test]
    fn resolve_tool_permission_hardcoded_write_fallback() {
        let server = bare_server_config("s");
        // Nothing configured anywhere, no hint → hardcoded strict Write.
        let resolved =
            resolve_tool_permission("s", "any", None, &server, None).expect("should resolve");
        assert_eq!(resolved, Permission::Write);
    }

    #[test]
    fn resolve_tool_permission_rejects_invalid_tool_override() {
        let mut server = bare_server_config("s");
        let mut per_tool = std::collections::HashMap::new();
        per_tool.insert("search".to_string(), "typo".to_string());
        server.tool_permissions = Some(per_tool);
        let err = resolve_tool_permission("s", "search", None, &server, None)
            .expect_err("invalid level should error");
        assert!(format!("{}", err).contains("tool_permissions['search']"));
    }

    #[test]
    fn resolve_tool_permission_with_source_attributes_each_step() {
        // 1. Per-tool override.
        let mut server = bare_server_config("s");
        let mut per_tool = std::collections::HashMap::new();
        per_tool.insert("a".to_string(), "ask".to_string());
        server.tool_permissions = Some(per_tool);
        let (perm, source) =
            resolve_tool_permission_with_source("s", "a", None, &server, None).unwrap();
        assert_eq!(perm, Permission::Ask);
        assert_eq!(source, PermissionSource::ToolOverride);

        // 2. Server-level override.
        let mut server = bare_server_config("s");
        server.permission = Some("read".into());
        let (perm, source) =
            resolve_tool_permission_with_source("s", "b", None, &server, None).unwrap();
        assert_eq!(perm, Permission::Read);
        assert_eq!(source, PermissionSource::ServerOverride);

        // 3. readOnlyHint fires when no user override is set.
        let server = bare_server_config("s");
        let ann = annotations_with_read_only_hint(Some(true));
        let (perm, source) =
            resolve_tool_permission_with_source("s", "c", Some(&ann), &server, None).unwrap();
        assert_eq!(perm, Permission::Read);
        assert_eq!(source, PermissionSource::ReadOnlyHint);

        // 4. Global default when no hint.
        let server = bare_server_config("s");
        let (perm, source) =
            resolve_tool_permission_with_source("s", "d", None, &server, Some(Permission::Read))
                .unwrap();
        assert_eq!(perm, Permission::Read);
        assert_eq!(source, PermissionSource::GlobalDefault);

        // 5. Hardcoded fallback.
        let server = bare_server_config("s");
        let (perm, source) =
            resolve_tool_permission_with_source("s", "e", None, &server, None).unwrap();
        assert_eq!(perm, Permission::Write);
        assert_eq!(source, PermissionSource::Fallback);
    }

    #[test]
    fn permission_source_labels_match_config_keys() {
        // The labels printed by `meka mcp tools` must match the config keys users would edit to
        // change a classification.
        assert_eq!(PermissionSource::ToolOverride.as_str(), "tool_permission");
        assert_eq!(
            PermissionSource::ServerOverride.as_str(),
            "server_permission"
        );
        assert_eq!(PermissionSource::ReadOnlyHint.as_str(), "readOnlyHint");
        assert_eq!(
            PermissionSource::GlobalDefault.as_str(),
            "default_permission"
        );
        assert_eq!(PermissionSource::Fallback.as_str(), "fallback");
    }

    #[test]
    fn tool_is_allowed_default_passes_everything() {
        let server = bare_server_config("s");
        assert!(tool_is_allowed(&server, "search"));
        assert!(tool_is_allowed(&server, "create-page"));
    }

    #[test]
    fn tool_is_allowed_allowlist_restricts() {
        let mut server = bare_server_config("s");
        server.allowed_tools = Some(vec!["search".into(), "fetch".into()]);
        assert!(tool_is_allowed(&server, "search"));
        assert!(tool_is_allowed(&server, "fetch"));
        assert!(!tool_is_allowed(&server, "create-page"));
    }

    #[test]
    fn tool_is_allowed_empty_allowlist_means_all() {
        // An empty `allowed_tools` array is treated as "unset", i.e. no restriction. A totally
        // absent field behaves the same way.
        let mut server = bare_server_config("s");
        server.allowed_tools = Some(Vec::new());
        assert!(tool_is_allowed(&server, "anything"));
    }

    #[test]
    fn tool_is_allowed_blocklist_removes() {
        let mut server = bare_server_config("s");
        server.disabled_tools = Some(vec!["delete-page".into()]);
        assert!(tool_is_allowed(&server, "search"));
        assert!(!tool_is_allowed(&server, "delete-page"));
    }

    #[test]
    fn tool_is_allowed_both_lists_compose() {
        // allow restricts to {search, fetch, write-page}, then block subtracts {write-page}. Net
        // effect: only search + fetch.
        let mut server = bare_server_config("s");
        server.allowed_tools = Some(vec!["search".into(), "fetch".into(), "write-page".into()]);
        server.disabled_tools = Some(vec!["write-page".into()]);
        assert!(tool_is_allowed(&server, "search"));
        assert!(tool_is_allowed(&server, "fetch"));
        assert!(!tool_is_allowed(&server, "write-page"));
        assert!(!tool_is_allowed(&server, "delete-page")); // not in allow
    }

    #[test]
    fn warn_on_stale_tool_config_smoke() {
        // The function just emits `warn!` lines; we can't easily assert on tracing output from a
        // unit test. Smoke-test that the happy path (empty config) doesn't panic and that it
        // accepts a server_config with all four list fields populated plus tool_permissions.
        let mut server = bare_server_config("s");
        server.allowed_tools = Some(vec!["a".into(), "unknown".into()]);
        server.disabled_tools = Some(vec!["b".into(), "gone".into()]);
        server.eager_load_tools = Some(vec!["a".into(), "stale".into(), "b".into()]);
        let mut perms = std::collections::HashMap::new();
        perms.insert("a".to_string(), "read".to_string());
        perms.insert("missing".to_string(), "write".to_string());
        server.tool_permissions = Some(perms);

        let advertised: std::collections::HashSet<&str> =
            ["a", "b", "search"].into_iter().collect();
        // Just confirm the call doesn't panic; "stale" should warn (unknown), and "b" should warn
        // (disabled∩eager overlap).
        warn_on_stale_tool_config("s", &server, &advertised);
    }

    #[test]
    fn tool_should_eager_load_unset_returns_false() {
        let server = bare_server_config("s");
        assert!(!tool_should_eager_load(&server, "search"));
        assert!(!tool_should_eager_load(&server, "anything"));
    }

    #[test]
    fn tool_should_eager_load_empty_list_returns_false() {
        let mut server = bare_server_config("s");
        server.eager_load_tools = Some(Vec::new());
        assert!(!tool_should_eager_load(&server, "search"));
    }

    #[test]
    fn tool_should_eager_load_matching_name_returns_true() {
        let mut server = bare_server_config("s");
        server.eager_load_tools = Some(vec!["search".into(), "fetch".into()]);
        assert!(tool_should_eager_load(&server, "search"));
        assert!(tool_should_eager_load(&server, "fetch"));
    }

    #[test]
    fn tool_should_eager_load_nonmatching_returns_false() {
        let mut server = bare_server_config("s");
        server.eager_load_tools = Some(vec!["search".into()]);
        assert!(!tool_should_eager_load(&server, "create-page"));
    }

    #[test]
    fn tool_should_eager_load_uses_raw_not_namespaced_name() {
        // The check is against the server-advertised raw name; the namespaced `mcp__notion__search`
        // form must NOT match an entry of `"search"`; that would create a footgun where users
        // could accidentally over-match across servers.
        let mut server = bare_server_config("notion");
        server.eager_load_tools = Some(vec!["search".into()]);
        assert!(!tool_should_eager_load(&server, "mcp__notion__search"));
    }

    #[test]
    fn test_truncate_under_limit() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_at_limit() {
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_over_limit() {
        assert_eq!(truncate("hello world", 5), "hello...");
    }

    #[test]
    fn test_truncate_unicode_boundary() {
        // Three emoji, each multiple bytes: truncation should cut on char boundary.
        let input = "🦀🦀🦀🦀🦀";
        let out = truncate(input, 2);
        assert_eq!(out, "🦀🦀...");
    }

    /// Build a bare server entry in `Pending` state for pure-state tests. No network, no process
    /// spawn.
    /// The configured `connect_timeout` has to reach the request helpers, not just `tools/list`.
    ///
    /// `bounded` hardcoded the module default, so `[mcp].connect_timeout_seconds` governed
    /// discovery and silently not `resources/read`, `prompts/get` or any of the other four. An
    /// operator who raised it for a slow server still had those calls cut at the default, and one
    /// who lowered it still waited the default.
    #[tokio::test]
    async fn a_request_helper_waits_the_configured_timeout_not_the_default() {
        let entry = pending_entry("slow-srv", McpTransport::Http);
        let configured = std::time::Duration::from_secs(3);
        entry
            .request_timeout
            .set(configured)
            .expect("first and only set");
        assert_ne!(
            configured, DEFAULT_MCP_REQUEST_TIMEOUT,
            "the test is only meaningful while the two differ"
        );

        tokio::time::pause();
        let live = CancellationToken::new();
        let waiting = tokio::spawn({
            let entry = Arc::clone(&entry);
            async move {
                bounded(&entry, "resources/read", &live, async {
                    std::future::pending::<Result<()>>().await
                })
                .await
            }
        });

        // Measured, not merely awaited. A paused clock auto-advances to the next timer whenever
        // every task is idle, so *some* timeout always fires and asserting only on the error tells
        // the two apart not at all -- the default would fire too, just later in virtual time.
        // Elapsed virtual time is the thing that differs.
        let started = tokio::time::Instant::now();
        let outcome = waiting.await.expect("join");
        let waited = started.elapsed();

        assert!(
            matches!(outcome, Err(MekaError::McpConnection { .. })),
            "an unanswered request must end: {outcome:?}",
        );
        assert!(
            waited < DEFAULT_MCP_REQUEST_TIMEOUT,
            "waited {waited:?}, which is the module default rather than the configured {configured:?}",
        );
    }

    fn pending_entry(name: &str, transport: McpTransport) -> Arc<ServerEntry> {
        let mut config = bare_server_config(name);
        config.transport = transport;
        Arc::new(ServerEntry {
            server_name: name.to_string(),
            config,
            token_store: None,
            client_context: McpClientContext::new(),
            state: RwLock::new(ServerState::Pending),
            reconnect_lock: Mutex::new(()),
            instructions: OnceLock::new(),
            request_timeout: OnceLock::new(),
            dropped_tools: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Shutdown must run while the manager is still shared.
    ///
    /// It used to consume `self`, so the caller had to win an `Arc::try_unwrap` first - and never
    /// could: the manager holds the registries it serves, and each of those holds six
    /// `mcp_resource_*` / `mcp_prompt_*` tools that hold the manager back. Sole ownership was
    /// unreachable by construction, so the close handshake never ran on any launch that had an MCP
    /// server to close, and every exit warned about it instead.
    ///
    /// The cycle is built here rather than described, so a future change that reintroduces it is
    /// caught: `attach_registry` puts a registry inside the manager, `install_tools_on` puts tools
    /// holding the manager inside that registry, and shutdown still has to reach the entries.
    #[tokio::test]
    async fn shutdown_runs_while_the_manager_is_still_shared() {
        let manager = McpClientManager::prepare(
            &[bare_server_config("probe")],
            None,
            None,
            McpClientContext::new(),
        )
        .await
        .expect("prepare");
        let registry = crate::tools::ToolRegistry::new();
        manager.attach_registry(registry.clone()).await;
        manager.install_tools_on(&registry).await;

        // The cycle the old `try_unwrap` lost to.
        assert!(
            Arc::strong_count(&manager) > 1,
            "the manager must be shared for this test to mean anything"
        );

        // Runs anyway. That this compiles at all is half the guard: `shutdown` taking `&self` is
        // what makes it reachable, and a change back to `self` would fail here rather than silently
        // restoring the warn-and-skip behaviour at runtime.
        manager.shutdown().await;

        // A `Pending` entry has no service, so it is left as it was; only `Connected` entries carry
        // something to close, and those are left `Disabled` once their service has been taken.
        let entry = manager.server_entry("probe").expect("entry");
        assert!(matches!(entry.state().await, ServerState::Pending));
    }

    /// A server that never connected registers no tools, so its names reach the agent's
    /// unknown-tool arm. Answering "unknown" would be false and would teach the agent the
    /// capability is gone; this reports the server's state instead.
    #[tokio::test]
    async fn unavailable_tool_reason_names_the_server_state() {
        let manager = McpClientManager::prepare(
            &[bare_server_config("ida")],
            None,
            None,
            McpClientContext::new(),
        )
        .await
        .expect("prepare");

        let reason = manager
            .unavailable_tool_reason("mcp__ida__decompile")
            .await
            .expect("a configured server must explain itself");
        assert!(reason.contains("ida"), "{reason}");
        assert!(reason.contains("still connecting"), "{reason}");

        // Failed reads differently from Pending: the two call for opposite behaviour, and
        // collapsing them yields an agent that either gives up early or retries forever.
        *manager
            .server_entry("ida")
            .expect("entry")
            .state
            .write()
            .await = ServerState::Failed {
            error: "'ida-mcp' not found".to_string(),
            at: std::time::Instant::now(),
        };
        let reason = manager
            .unavailable_tool_reason("mcp__ida__decompile")
            .await
            .expect("failed server must explain itself");
        assert!(reason.contains("unavailable"), "{reason}");
        assert!(reason.contains("'ida-mcp' not found"), "{reason}");
    }

    /// Sub-agent registries come through `install_tools_on`, not `attach_registry`, so they need
    /// the manager back-reference wired there too - otherwise a sub-agent reaching for a dead
    /// server's tool gets a bare "not registered" while the parent gets the reason.
    #[tokio::test]
    async fn subagent_registry_also_explains_an_unconnected_server() {
        use crate::tools::ToolRegistry;

        let manager = McpClientManager::prepare(
            &[bare_server_config("ida")],
            None,
            None,
            McpClientContext::new(),
        )
        .await
        .expect("prepare");

        let registry = ToolRegistry::new();
        registry.register_load_tool_for_test();
        manager.install_tools_on(&registry).await;

        let load_tool = registry.get("load_tool").expect("load_tool registered");
        let output = load_tool
            .execute(
                serde_json::json!({"name": "mcp__ida__decompile"}),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .expect("load_tool returns Ok with an error payload");
        let text = format!("{:?}", output.content);
        assert!(text.contains("ida"), "{text}");
        assert!(!text.contains("not registered"), "{text}");

        // The same back-reference is what `Agent::resolve_and_execute_tool` reads when the model
        // calls the tool outright instead of loading it: a sub-agent has no `mcp_manager` of its
        // own, so the registry is the only route to the reason.
        let from_registry = registry
            .mcp_manager()
            .expect("install_tools_on must record the manager")
            .unavailable_tool_reason("mcp__ida__decompile")
            .await;
        assert!(from_registry.is_some_and(|reason| reason.contains("ida")));
    }

    /// `load_tool` is the path a model actually takes: the tool is absent from its catalogue, so
    /// it reaches for the documented way to load a deferred tool first. Found by watching a real
    /// model do exactly that and get "not registered" back.
    #[tokio::test]
    async fn load_tool_explains_an_unconnected_server() {
        use crate::tools::ToolRegistry;

        let manager = McpClientManager::prepare(
            &[bare_server_config("ida")],
            None,
            None,
            McpClientContext::new(),
        )
        .await
        .expect("prepare");

        let registry = ToolRegistry::new();
        registry.register_load_tool_for_test();
        manager.attach_registry(registry.clone()).await;

        let load_tool = registry.get("load_tool").expect("load_tool registered");
        let output = load_tool
            .execute(
                serde_json::json!({"name": "mcp__ida__decompile"}),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .expect("load_tool returns Ok with an error payload");

        assert!(output.is_error);
        let text = format!("{:?}", output.content);
        assert!(text.contains("ida"), "{text}");
        assert!(text.contains("still connecting"), "{text}");
        assert!(!text.contains("not registered"), "{text}");
    }

    /// Genuinely unknown names must stay unknown, or the agent loses the signal that it invented
    /// a tool.
    #[tokio::test]
    async fn unavailable_tool_reason_ignores_non_mcp_and_unconfigured_names() {
        let manager = McpClientManager::prepare(
            &[bare_server_config("ida")],
            None,
            None,
            McpClientContext::new(),
        )
        .await
        .expect("prepare");

        for name in [
            "read_file",
            "memory_write",
            "mcp__nosuch__tool",
            "mcp__malformed",
            "mcp__",
        ] {
            assert!(
                manager.unavailable_tool_reason(name).await.is_none(),
                "'{name}' must fall through to unknown-tool"
            );
        }
    }

    /// The gate needs to know which unavailable servers actually stop a turn.
    #[tokio::test]
    async fn enabled_not_connected_reports_required() {
        let mut optional = bare_server_config("ida");
        optional.required = Some(false);
        let mut gating = bare_server_config("bridge");
        gating.required = Some(true);

        let manager =
            McpClientManager::prepare(&[optional, gating], None, None, McpClientContext::new())
                .await
                .expect("prepare");

        let not_ready = manager.enabled_not_connected().await;
        assert_eq!(not_ready.len(), 2);
        let bridge = not_ready
            .iter()
            .find(|s| s.name == "bridge")
            .expect("bridge listed");
        assert!(bridge.required);
        let ida = not_ready
            .iter()
            .find(|s| s.name == "ida")
            .expect("ida listed");
        assert!(!ida.required);
    }

    #[tokio::test]
    async fn require_connected_errors_for_pending() {
        let entry = pending_entry("pending-srv", McpTransport::Http);
        let err = entry
            .require_connected()
            .await
            .expect_err("pending should not yield a peer");
        match err {
            MekaError::McpConnection {
                server_name,
                message,
            } => {
                assert_eq!(server_name, "pending-srv");
                assert!(message.contains("connecting"), "got: {}", message);
            }
            other => panic!("expected McpConnection, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn require_connected_errors_for_failed() {
        let entry = pending_entry("failed-srv", McpTransport::Http);
        *entry.state.write().await = ServerState::Failed {
            error: "boom".to_string(),
            at: std::time::Instant::now(),
        };
        let err = entry.require_connected().await.unwrap_err();
        assert!(matches!(err, MekaError::McpConnection { .. }));
    }

    #[tokio::test]
    async fn require_connected_errors_for_disabled() {
        let entry = pending_entry("off-srv", McpTransport::Http);
        *entry.state.write().await = ServerState::Disabled;
        let err = entry.require_connected().await.unwrap_err();
        match err {
            MekaError::McpConnection { message, .. } => assert!(message.contains("disabled")),
            other => panic!("expected McpConnection, got: {:?}", other),
        }
    }

    #[test]
    fn server_state_label_matches_variant() {
        assert_eq!(ServerState::Pending.label(), "pending");
        assert_eq!(ServerState::Disabled.label(), "disabled");
        assert_eq!(
            ServerState::Failed {
                error: "x".into(),
                at: std::time::Instant::now()
            }
            .label(),
            "failed"
        );
    }

    #[tokio::test]
    async fn prepare_all_disabled_publishes_settled_immediately() {
        let mut config = bare_server_config("off");
        config.disabled = true;
        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(&[config], None, None, context)
            .await
            .expect("prepare should succeed with a disabled-only config");
        assert!(manager.all_ready(), "manager should be settled immediately");
        let not_ready = manager.enabled_not_connected().await;
        assert!(
            not_ready.is_empty(),
            "disabled servers don't count as not-ready"
        );
    }

    #[tokio::test]
    async fn prepare_pending_entries_not_ready_until_connector_runs() {
        let config = bare_server_config("waiting");
        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(&[config], None, None, context)
            .await
            .expect("prepare should succeed");
        assert!(
            !manager.all_ready(),
            "pending server shouldn't be ready yet"
        );
        let not_ready = manager.enabled_not_connected().await;
        assert_eq!(not_ready.len(), 1);
        assert_eq!(not_ready[0].name, "waiting");
    }

    #[test]
    fn resolve_concurrency_env_uses_default_when_unset() {
        // Unique var names so parallel tests can't race on env state.
        let var = "MEKA_TEST_CONCURRENCY_UNSET";
        unsafe {
            std::env::remove_var(var);
        }
        assert_eq!(resolve_concurrency_env(var, 7), 7);
    }

    #[test]
    fn resolve_concurrency_env_parses_positive_override() {
        let var = "MEKA_TEST_CONCURRENCY_OVERRIDE";
        unsafe {
            std::env::set_var(var, "11");
        }
        assert_eq!(resolve_concurrency_env(var, 3), 11);
        unsafe {
            std::env::remove_var(var);
        }
    }

    #[test]
    fn resolve_concurrency_env_falls_back_on_garbage() {
        let var = "MEKA_TEST_CONCURRENCY_GARBAGE";
        unsafe {
            std::env::set_var(var, "not-a-number");
        }
        assert_eq!(resolve_concurrency_env(var, 5), 5);
        unsafe {
            std::env::remove_var(var);
        }
    }

    #[test]
    fn resolve_concurrency_env_rejects_zero() {
        // Zero would deadlock `buffer_unordered(0)`; must fall back.
        let var = "MEKA_TEST_CONCURRENCY_ZERO";
        unsafe {
            std::env::set_var(var, "0");
        }
        assert_eq!(resolve_concurrency_env(var, 4), 4);
        unsafe {
            std::env::remove_var(var);
        }
    }

    #[tokio::test]
    async fn await_settled_returns_immediately_when_already_settled() {
        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(&[], None, None, context)
            .await
            .expect("prepare with no servers should succeed");
        let res = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            manager.await_settled(),
        )
        .await;
        assert!(
            res.is_ok(),
            "await_settled blocked past the no-pending fast path"
        );
    }

    #[tokio::test]
    async fn await_settled_unblocks_when_connector_finishes() {
        // `/bin/false` exits immediately, so the connector reaches `settled.send(true)` via Failed
        // state on the first entry.
        let mut config = bare_server_config("quick-fail");
        config.transport = McpTransport::Stdio;
        config.command = Some("/bin/false".to_string());
        config.url = None;

        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(&[config], None, None, context)
            .await
            .expect("prepare should succeed");
        assert!(!manager.all_ready());

        manager.start_connector(McpRuntimeConfig {
            connect_timeout: std::time::Duration::from_secs(2),
            stdio_concurrency: 1,
            http_concurrency: 1,
        });

        let res =
            tokio::time::timeout(std::time::Duration::from_secs(5), manager.await_settled()).await;
        assert!(
            res.is_ok(),
            "await_settled didn't unblock after connector finished"
        );
        assert!(manager.all_ready());

        let entry = manager.server_entry("quick-fail").expect("entry");
        let state = entry.state().await;
        assert!(
            matches!(state, ServerState::Failed { .. }),
            "expected Failed, got: {}",
            state.label()
        );
    }

    /// Sub-agent registry inherits the parent's MCP resource / prompt meta-tools, even when no
    /// server is connected yet. The per-server adapters only show up for `Connected` servers; that
    /// case is covered separately by manual verification since spinning up a real stdio MCP server
    /// here is heavy.
    #[tokio::test]
    async fn install_tools_on_registers_resource_meta_tools() {
        let mut config = bare_server_config("subagent-fixture");
        // Disable so `prepare` skips entirely without spawning a connector. `server_names()` still
        // includes it, which is all `register_all` needs to gate on.
        config.disabled = true;

        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(&[config], None, None, context)
            .await
            .expect("prepare should succeed for a disabled server");

        let registry = crate::tools::ToolRegistry::new();
        manager.install_tools_on(&registry).await;

        for name in [
            "mcp_resource_list",
            "mcp_resource_read",
            "mcp_prompt_list",
            "mcp_prompt_get",
            "mcp_resource_subscribe",
            "mcp_resource_unsubscribe",
            "mcp_resource_updates_list",
        ] {
            assert!(
                registry.get(name).is_some(),
                "expected '{}' on sub-agent registry after install_tools_on",
                name
            );
        }
    }

    /// With zero servers configured, `register_all`'s `server_names().is_empty()` guard kicks in
    /// and nothing is registered.
    #[tokio::test]
    async fn install_tools_on_noop_without_servers() {
        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(&[], None, None, context)
            .await
            .expect("prepare with no servers should succeed");

        let registry = crate::tools::ToolRegistry::new();
        manager.install_tools_on(&registry).await;

        assert!(
            registry.get("mcp_resource_list").is_none(),
            "no MCP meta-tools should land on the registry when no servers configured"
        );
    }

    /// `update_server_tools` racing against `attach_registry` must not lose updates: every
    /// published tool list must reach every session that attaches before or during the publish,
    /// with no silent miss window. Regression guard for the race fixed in
    /// [`McpClientManager::attach_registry`] where the original "read snapshot → push registry"
    /// order let updates land in the gap.
    #[tokio::test]
    async fn attach_registry_races_with_update_without_losing_tools() {
        use std::sync::Arc;

        use crate::{
            permission::Permission,
            provider::ToolDefinition,
            tools::{Tool, ToolOutput},
        };

        // Minimal fixture so each server publishes a distinctively-named tool. An empty Vec to
        // `replace_server_tools` is a no-op and wouldn't actually exercise the propagation path.
        struct FixtureTool {
            name: String,
        }
        #[async_trait::async_trait]
        impl Tool for FixtureTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition::new(
                    self.name.clone(),
                    "race fixture".to_string(),
                    serde_json::json!({"type": "object", "properties": {}}),
                )
            }

            fn required_permission(&self) -> Permission {
                Permission::Read
            }

            async fn execute(
                &self,
                _input: serde_json::Value,
                _cancellation: tokio_util::sync::CancellationToken,
            ) -> crate::error::Result<ToolOutput> {
                Ok(ToolOutput::text("ok".to_string(), false))
            }
        }

        // Empty config: we don't need real servers to exercise the snapshot/registry plumbing,
        // just the manager methods.
        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(&[], None, None, context)
            .await
            .expect("prepare");

        let server_names: Vec<String> = (0..4).map(|index| format!("srv-{}", index)).collect();
        let registry_count = 8;
        let registries: Vec<crate::tools::ToolRegistry> = (0..registry_count)
            .map(|_| crate::tools::ToolRegistry::new())
            .collect();

        // Each updater publishes one tool named mcp__<server>__ping.
        let mut update_handles = Vec::new();
        for name in &server_names {
            let manager = Arc::clone(&manager);
            let name = name.clone();
            update_handles.push(tokio::spawn(async move {
                let tool: Arc<dyn Tool> = Arc::new(FixtureTool {
                    name: format!("mcp__{}__ping", name),
                });
                manager.update_server_tools(&name, vec![tool]).await;
            }));
        }
        let mut attach_handles = Vec::new();
        for registry in &registries {
            let manager = Arc::clone(&manager);
            let registry = registry.clone();
            attach_handles.push(tokio::spawn(async move {
                manager.attach_registry(registry).await;
            }));
        }

        for handle in update_handles {
            handle.await.expect("update task");
        }
        for handle in attach_handles {
            handle.await.expect("attach task");
        }

        // The snapshot is the source of truth for "what got published".  Every server's
        // update must land there, and every registry must hold every server's tool.
        let snapshot_keys: std::collections::HashSet<String> = manager
            .tools_snapshot
            .read()
            .await
            .keys()
            .cloned()
            .collect();
        assert_eq!(
            snapshot_keys.len(),
            server_names.len(),
            "every update_server_tools call should land in the snapshot",
        );
        for registry in &registries {
            for server in &server_names {
                let tool_name = format!("mcp__{}__ping", server);
                assert!(
                    registry.get(&tool_name).is_some(),
                    "registry missing '{}' after concurrent attach/update: race regressed",
                    tool_name,
                );
            }
        }
    }
}
