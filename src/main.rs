//! `meka`: a general-purpose AI agent harness where you describe what you want in natural language
//! and an LLM-backed agent decides which tools to run.
//!
//! The binary wires together: a [`provider`] (Claude or OpenAI), a [`session`] store backed by
//! SQLite, a [`tools`] registry, an MCP client manager, and a [`repl`] input loop. The [`agent`]
//! module owns the per-turn loop that streams provider output and dispatches tool calls.

// Production code shouldn't panic on unexpected input; the `Cargo.toml` `[lints.clippy]` block
// enforces that with `unwrap_used` / `expect_used` / `panic` at warn level (CI promotes warnings to
// errors). Tests use `.unwrap()` and `.expect()` heavily on purpose: a failed test should panic
// with a clear message rather than thread `Result` through every fixture. The cfg_attr below scopes
// the relaxation to test builds only.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

mod acp;
mod agent;
mod cli;
mod config;
mod context;
mod conversation;
mod error;
mod frontend;
mod history;
mod image;
mod mcp;
mod permission;
mod provider;
mod relay;
mod render;
mod repl;
mod sandbox;
mod server;
mod session;
mod skills;
mod stats;
mod tokens;
mod tools;

use std::sync::Arc;

use clap::Parser;
use tokio_util::sync::CancellationToken;

use crate::{
    agent::{Agent, AgentOptions},
    config::ResolvedConfig,
    permission::SharedPermission,
    provider::{AuthCredential, ProviderBuilder},
    repl::ReplEvent,
    session::{SessionManager, TokenStore},
    tools::ToolRegistry,
};

fn main() -> anyhow::Result<()> {
    let cli = cli::Cli::parse();

    let log_level = match cli.verbosity {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    // Route tracing through `relay::RELAY` so the REPL can later install a reedline
    // `ExternalPrinter` and have warnings printed *above* the live prompt instead of racing
    // reedline's redraw. Without a printer installed (non-interactive subcommands, pre-REPL startup
    // window) the relay falls back to plain stderr.
    let rust_log = std::env::var("RUST_LOG").ok();
    tracing_subscriber::fmt()
        .with_env_filter(build_log_filter(rust_log.as_deref(), log_level))
        .with_writer(relay::RELAY.clone())
        .init();

    let runtime = tokio::runtime::Runtime::new()?;
    let result = run_on_runtime(&runtime, cli);
    // Detach any lingering blocking threads instead of joining them on drop. `tokio::io::stdin()`
    // (used by the OAuth paste fallback) spawns a blocking worker that sits on a `read()` syscall
    // until stdin has bytes or EOF; when the user Ctrl-Cs during the wait, the future is dropped
    // but that worker can't be cancelled from the outside. Without this the default `Runtime::drop`
    // joins that thread and hangs the process after a clean rollback.
    runtime.shutdown_background();

    // User-initiated interrupts are already acknowledged by the rollback warn log ("interrupted;
    // rolling back …") and the shell typically echoes `^C` itself; anyhow's default "Error:
    // agent interrupted by user" on top of that is just noise. Exit with the conventional
    // SIGINT code (128 + 2) silently instead.
    if let Err(error) = &result
        && let Some(crate::error::MekaError::Interrupted) =
            error.downcast_ref::<crate::error::MekaError>()
    {
        std::process::exit(130);
    }
    result
}

fn run_on_runtime(runtime: &tokio::runtime::Runtime, cli: cli::Cli) -> anyhow::Result<()> {
    // `meka acp` and `meka serve` are heavyweight (full config + credential resolution + MCP
    // setup) so they route through `async_main` rather than the lightweight subcommand block
    // below.
    let acp_mode = matches!(cli.command, Some(cli::Command::Acp));
    let serve_mode = matches!(cli.command, Some(cli::Command::Serve { .. }));

    // Handle subcommands that don't need full config resolution.
    if let Some(command) = cli.command.as_ref()
        && !acp_mode
        && !serve_mode
    {
        let cli_ref = &cli;
        return runtime.block_on(async move {
            let session_manager = SessionManager::open(None).await?;
            match command {
                cli::Command::Provider { action } => {
                    let token_store = session_manager.token_store();
                    provider::cli::run(action, &token_store).await
                }
                cli::Command::Session { action } => {
                    run_session_subcommand(&session_manager, action).await
                }
                cli::Command::History { action } => {
                    run_history_subcommand(&session_manager, action).await
                }
                cli::Command::Mcp { action } => {
                    run_mcp_subcommand(&session_manager, action, cli_ref).await
                }
                cli::Command::Tools { action } => run_tools_subcommand(action, cli_ref).await,
                cli::Command::Skill { action } => run_skill_subcommand(action).await,
                cli::Command::Account { action } => {
                    run_account_subcommand(&session_manager, action).await
                }
                cli::Command::Acp | cli::Command::Serve { .. } => {
                    unreachable!("Acp / Serve route through async_main above");
                }
            }
        });
    }

    // --oneshot needs something to do; reject early before any setup.
    if cli.oneshot && cli.prompt.is_none() && cli.skill.is_none() {
        return Err(anyhow::anyhow!(
            "--oneshot requires a prompt argument or --skill"
        ));
    }

    // If --skill is set, validate and render the body upfront so an invalid name fails fast
    // before any session/MCP setup. The combined string (extra + body, mirroring the REPL's `/skill
    // <name> [extra...]`) then takes the place of cli.prompt as the first-turn input.
    let skill_prompt = runtime.block_on(build_skill_prompt(&cli))?;

    let mut config = ResolvedConfig::from_cli(&cli);
    if let Some(prompt) = skill_prompt {
        config.prompt = Some(prompt);
    }
    // `--bind` on `meka serve` overrides the config-file `[serve].bind`. Apply here so
    // `async_main` sees a single resolved binding without re-parsing the CLI.
    if let Some(cli::Command::Serve { bind: Some(bind) }) = cli.command.as_ref() {
        config.serve_bind_override = Some(bind.clone());
    }
    runtime.block_on(async_main(config, acp_mode, serve_mode))
}

/// Render a `--skill <name>` invocation into the user-message string that drives the first turn.
/// Returns `Ok(None)` when `--skill` is not set so callers can leave `cli.prompt` untouched.
///
/// Mirrors the REPL handler at `SlashCommand::SkillInvoke`: same lookup, same `user_invocable`
/// gate, same `format!("{extra}\n\n{body}")` order when the positional `[PROMPT]` is supplied.
async fn build_skill_prompt(cli: &cli::Cli) -> anyhow::Result<Option<String>> {
    let Some(name) = cli.skill.as_deref() else {
        return Ok(None);
    };
    let skill = skills::cli::require_skill(name)?;
    // Pass `None` for session_id: the session is created lazily on the first turn, so
    // `${MEKA_SESSION_ID}` would be unresolvable here. This matches the REPL's first-turn `/skill`
    // behaviour, where session_id is also None until run_turn populates it.
    let body = skills::load_skill_body(&skill, None)
        .await
        .map_err(|error| anyhow::anyhow!("failed to load skill '{}': {}", name, error))?;
    let combined = match cli.prompt.as_deref() {
        Some(extra) if !extra.is_empty() => format!("{}\n\n{}", extra, body),
        _ => body,
    };
    Ok(Some(combined))
}

/// Build the `tracing` filter for meka.
///
/// When the user sets `RUST_LOG`, we honour it verbatim; no hidden
/// overrides. Debugging with `RUST_LOG=rmcp=debug` works as expected.
/// Otherwise we start from `log_level` (derived from `-v` / `-vv`) and
/// add a single directive that downgrades rmcp's SSE-reconnect warning
/// to `error`:
///
/// MCP servers behind a CDN / edge (Cloudflare, Fastly, …) close idle HTTP streams after ~100 s,
/// which trips `rmcp::transport::common::client_side_sse`'s `warn!("sse stream error: …")` before
/// rmcp transparently reconnects via `Last-Event-ID`. The warn fires on every expected reconnect;
/// the real failure mode (`"max retry times reached"`) is emitted at `error!` from the same module,
/// so an `=error` floor keeps the useful signal and drops the noise. Verified against rmcp 1.5.
fn build_log_filter(rust_log: Option<&str>, log_level: &str) -> tracing_subscriber::EnvFilter {
    use tracing_subscriber::EnvFilter;
    if let Some(value) = rust_log
        && let Ok(filter) = EnvFilter::try_new(value)
    {
        return filter;
    }
    // The directive string is a compile-time literal in a known-good shape; `.parse()` failing
    // would mean we shipped a malformed directive, caught on first test.
    #[allow(clippy::expect_used)]
    let directive = "rmcp::transport::common::client_side_sse=error"
        .parse()
        .expect("valid tracing directive");
    EnvFilter::new(log_level).add_directive(directive)
}

async fn async_main(
    config: ResolvedConfig,
    acp_mode: bool,
    serve_mode: bool,
) -> anyhow::Result<()> {
    // Validate provider name and model before opening the session store or resolving credentials so
    // the user sees a clear "not configured" or "invalid value" message instead of the downstream
    // credential error.
    config.validate()?;

    // Warn once at startup about an unusable configured sandbox backend or an auto-fallback to
    // landlock that the user could improve by installing bubblewrap. Re-emitted at read-mode entry
    // boundaries below.
    crate::sandbox::warn_if_sandbox_issues(
        &crate::sandbox::SandboxState::from_config(&config),
        crate::sandbox::WarnContext::Startup,
    );

    let session_manager = SessionManager::open(None).await?;
    let token_store = session_manager.token_store();

    if let Some(retention_days) = config.retention_days {
        let deleted = session_manager
            .delete_expired_sessions(retention_days)
            .await?;
        if deleted > 0 {
            tracing::info!("deleted {} expired sessions", deleted);
        }
    }

    if let Some(max_bytes) = config.max_storage_bytes {
        // Startup eviction: no sessions are opened yet, so the active set is empty. The signature
        // is still required for mid-run callers that want to protect live sessions.
        let active: std::collections::HashSet<String> = std::collections::HashSet::new();
        let deleted = session_manager
            .enforce_storage_limit(max_bytes, &active)
            .await?;
        if deleted > 0 {
            tracing::info!("deleted {} sessions to enforce storage limit", deleted);
        }
    }

    let mcp_context = mcp::McpClientContext::new();
    let mcp_manager = if !config.mcp_servers.is_empty() {
        let manager = mcp::McpClientManager::prepare(
            &config.mcp_servers,
            config.mcp_default_permission,
            Some(token_store.clone()),
            Arc::clone(&mcp_context),
        )
        .await?;
        mcp_context.set_manager(Arc::downgrade(&manager));
        Some(manager)
    } else {
        None
    };

    // `meka acp` and `meka serve` reuse every step above (credential resolution, MCP setup,
    // session-manager housekeeping) and then enter their respective transport loops instead of
    // the REPL.
    if serve_mode {
        return server::run_serve(config, session_manager, mcp_manager, mcp_context).await;
    }
    if acp_mode {
        return acp::run_acp(config, session_manager, mcp_manager, mcp_context).await;
    }

    // `--oneshot` runs a single turn and exits; the prompt is required (validated at startup).
    // Without `--oneshot`, any provided prompt/skill becomes the first-turn input but the REPL
    // stays open afterwards.
    if config.oneshot {
        // `Cli` validation at startup rejects `--oneshot` without a prompt or `--skill`, so the
        // `Some` arm is the only reachable one here. `let-else { unreachable!() }` documents the
        // invariant in code rather than relying on a brittle string-tagged `expect`.
        let Some(prompt) = config.prompt.clone() else {
            unreachable!("--oneshot requires a prompt or --skill; rejected by Cli validation");
        };
        return run_oneshot(config, session_manager, token_store, prompt, mcp_manager).await;
    }

    let initial_prompt = config.prompt.clone();
    run_interactive(
        config,
        session_manager,
        token_store,
        initial_prompt,
        mcp_manager,
    )
    .await
}

/// Process-wide dependencies that every ACP session shares. Built once at `meka acp` startup by
/// [`build_shared_deps`]; sessions hold an [`Arc<SharedDeps>`] and read fields by reference.
/// Cheap to clone (every field is either an `Arc`, an owned-but-small value, or a clonable handle).
///
/// The REPL / oneshot paths don't use this; they go through [`create_agent_from_config`] which
/// bundles shared + per-session work into one call.
#[derive(Clone)]
pub struct SharedDeps {
    pub config: Arc<ResolvedConfig>,
    pub session_manager: SessionManager,
    pub provider: Arc<dyn provider::Provider>,
    pub mcp_manager: Option<Arc<mcp::McpClientManager>>,
    pub mcp_context: Arc<mcp::McpClientContext>,
    pub skills: Arc<skills::SkillCache>,
    pub builtin_filter: crate::tools::BuiltinToolFilter,
    pub sandbox_capability: crate::sandbox::SandboxCapability,
    pub sandboxed_shell: bool,
    pub agent_options: AgentOptions,
    pub session_stats: Arc<stats::SessionStats>,
}

/// Build the process-wide [`SharedDeps`] for `meka acp`. Sets up the provider, MCP wiring, skill
/// cache, sandbox capability probe, and the shared `agent_options` template. Each ACP session later
/// calls [`build_session_agent`] against the resulting struct to spin up its own per-session
/// `Agent` + `ToolRegistry`.
pub async fn build_shared_deps(
    config: ResolvedConfig,
    session_manager: SessionManager,
    credential: AuthCredential,
    mcp_manager: Option<Arc<mcp::McpClientManager>>,
    mcp_context: Arc<mcp::McpClientContext>,
) -> anyhow::Result<SharedDeps> {
    config.validate()?;

    let provider_name = config
        .provider_name
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("provider_name missing after validation"))?;
    let needs_token_store = matches!(credential, AuthCredential::OAuthToken { .. });
    let token_store = session_manager.token_store();

    let model = config
        .model
        .clone()
        .ok_or_else(|| anyhow::anyhow!("model missing after validation"))?;
    let session_stats = Arc::new(stats::SessionStats::default());
    let provider = ProviderBuilder::new(provider_name, credential, model)
        .base_url(config.base_url.clone())
        .client_id(config.client_id.clone())
        .credential_key(config.active_profile.clone())
        .oauth_token_url(config.oauth_token_url.clone())
        .token_store(if needs_token_store {
            Some(Arc::new(token_store))
        } else {
            None
        })
        .thinking(config.thinking_enabled, config.thinking_budget_tokens)
        .reasoning_effort(config.reasoning_effort.clone())
        .device_id(config.device_id.clone())
        .effort(config.effort.clone())
        .redact_thinking(config.redact_thinking)
        .max_output_tokens(config.max_output_tokens)
        .session_stats(Some(Arc::clone(&session_stats)))
        .build()?;

    let sandbox_capability: crate::sandbox::SandboxCapability = match &config.backend_probe {
        crate::sandbox::BackendProbe::Ok(capability) => capability.clone(),
        _ => crate::sandbox::SandboxCapability::Unavailable,
    };
    let sandboxed_shell = config.sandbox
        && !matches!(
            sandbox_capability,
            crate::sandbox::SandboxCapability::Unavailable
        );

    let skills = crate::skills::SkillCache::discover();
    let builtin_filter = crate::tools::BuiltinToolFilter::from_config(
        config.builtin_allowed_tools.clone(),
        config.builtin_disabled_tools.clone(),
        config.builtin_tool_permissions.clone(),
    );

    let agent_options = AgentOptions {
        streaming: config.streaming,
        sandboxed_shell,
        context_messages: config.context_messages,
        auto_compact: config.auto_compact,
        context_window: config.context_window.unwrap_or_else(|| {
            config
                .model
                .as_deref()
                .map(crate::config::context_window_for_model)
                .unwrap_or(128_000)
        }),
        user_instructions: config.user_instructions.clone(),
        mcp_strict: config.mcp_strict,
        mcp_grace: config.mcp_grace,
        system_prompt_override: None,
    };

    // Kick off the MCP background connector once for the whole process. The connector writes tool
    // discoveries through `update_server_tools`, which fans them out to every attached registry,
    // so per-session registries built later via `build_session_agent` see the tools as servers
    // come online. Idempotent on second call.
    if let Some(manager) = &mcp_manager {
        manager.start_connector(crate::mcp::McpRuntimeConfig::from_config(&config));
    }

    Ok(SharedDeps {
        config: Arc::new(config),
        session_manager,
        provider,
        mcp_manager,
        mcp_context,
        skills,
        builtin_filter,
        sandbox_capability,
        sandboxed_shell,
        agent_options,
        session_stats,
    })
}

/// Inputs both `build_session_agent` and `create_agent_from_config` hand into the unified
/// [`assemble_agent`] helper. Bundling them in a struct keeps the assembly call below readable and
/// lets both callers express "everything I built; turn it into an Agent" in one line.
struct AgentAssembly<'a> {
    web_client: crate::config::WebClientConfig,
    sandbox_enabled: bool,
    sandbox_capability: crate::sandbox::SandboxCapability,
    sandbox_backend: crate::config::SandboxBackend,
    backend_probe: crate::sandbox::BackendProbe,
    user_instructions: Option<String>,
    session_manager: SessionManager,
    provider: Arc<dyn provider::Provider>,
    mcp_manager: Option<&'a Arc<mcp::McpClientManager>>,
    skills: Arc<skills::SkillCache>,
    builtin_filter: crate::tools::BuiltinToolFilter,
    agent_options: AgentOptions,
    session_stats: Arc<stats::SessionStats>,
    /// Seeds the root `SpawnAgentTool`'s recursion budget from `session.subagent_max_depth`.
    subagent_max_depth: usize,
}

/// Per-session agent assembly used by both the ACP session builder and the REPL's
/// `create_agent_from_config`. Builds the shared todo list / scratchpad cell, the tool registry
/// (with the session's cwd / permission / frontend baked into the builtins), registers
/// `spawn_agent` and the MCP resource meta-tools, attaches the registry to the MCP manager, and
/// finally constructs the `Agent` itself.
///
/// **MCP attach-before-connector invariant**: the caller is expected to either (a) already have run
/// `start_connector` (ACP path: `build_shared_deps` does this once) or (b) call
/// `start_connector` *after* this returns (REPL path). Either way, the registry must be attached
/// before any connector activity, so initial tool-list discoveries reach this session's registry.
async fn assemble_agent(
    bundle: AgentAssembly<'_>,
    shared_permission: SharedPermission,
    frontend: Arc<dyn frontend::Frontend>,
    cwd: crate::agent::SharedCwd,
) -> anyhow::Result<(Agent, crate::tools::ToolRegistry)> {
    let todo_list: crate::tools::todo::SharedTodoList = std::sync::Arc::new(
        tokio::sync::RwLock::new(crate::tools::todo::TodoState::default()),
    );
    let shared_session_id: std::sync::Arc<tokio::sync::RwLock<Option<uuid::Uuid>>> =
        std::sync::Arc::new(tokio::sync::RwLock::new(None));

    let tool_registry = ToolRegistry::build_default(
        bundle.web_client.clone(),
        shared_permission.clone(),
        bundle.sandbox_enabled,
        bundle.sandbox_capability.clone(),
        bundle.sandbox_backend,
        bundle.backend_probe.clone(),
        todo_list.clone(),
        bundle.session_manager.clone(),
        shared_session_id.clone(),
        bundle.skills.clone(),
        bundle.builtin_filter.clone(),
        cwd.clone(),
        Arc::clone(&frontend),
    )?;

    // `subagent_max_depth == 0` disables sub-agents entirely (root gets no `spawn_agent`); `>= 1`
    // seeds the root's soft recursion budget. The `absolute_depth` starts at 0 for the root.
    if bundle.subagent_max_depth >= 1 && bundle.builtin_filter.admits("spawn_agent") {
        tool_registry.register(Arc::new(crate::tools::subagent::SpawnAgentTool {
            provider: Arc::clone(&bundle.provider),
            parent_permission: shared_permission.clone(),
            tool_builder_params: crate::tools::subagent::ToolBuilderParams {
                web_client: bundle.web_client.clone(),
                sandbox_enabled: bundle.sandbox_enabled,
                sandbox_capability: bundle.sandbox_capability.clone(),
                sandbox_backend: bundle.sandbox_backend,
                backend_probe: bundle.backend_probe.clone(),
                builtin_filter: bundle.builtin_filter.clone(),
                skills: bundle.skills.clone(),
                mcp_manager: bundle.mcp_manager.map(Arc::downgrade),
                session_manager: bundle.session_manager.clone(),
                parent_shared_session_id: shared_session_id.clone(),
                session_stats: Arc::clone(&bundle.session_stats),
                parent_options: bundle.agent_options.clone(),
                parent_cwd: Arc::clone(&cwd),
                parent_frontend: Arc::clone(&frontend),
            },
            user_instructions: bundle.user_instructions.clone(),
            remaining_depth: bundle.subagent_max_depth,
            absolute_depth: 0,
        }))?;
    }

    if let Some(manager) = bundle.mcp_manager {
        // Register MCP resource meta-tools upfront; they delegate through
        // `ServerEntry::require_connected` so they tolerate Pending / Failed servers until a
        // specific one is called.
        crate::tools::mcp_resources::register_all(&tool_registry, Arc::clone(manager));
        // Attach this session's registry so the MCP connector and tools/list_changed handler
        // propagate updates into it. Must happen before the connector kicks off; otherwise initial
        // server-state updates miss the registry.
        manager.attach_registry(tool_registry.clone()).await;
    }

    let mut agent = Agent::new(
        Arc::clone(&bundle.provider),
        tool_registry.clone(),
        bundle.session_manager.clone(),
        shared_permission,
        bundle.agent_options.clone(),
        todo_list,
        shared_session_id,
        bundle.skills.clone(),
        frontend,
        cwd,
        Arc::clone(&bundle.session_stats),
    );
    if let Some(manager) = bundle.mcp_manager {
        agent.set_mcp_manager(Arc::clone(manager));
    }

    Ok((agent, tool_registry))
}

/// Build a per-session `Agent` + `ToolRegistry` from the already-prepared [`SharedDeps`]. Each ACP
/// session gets a fresh todo list, scratchpad slot, tool registry (with the session's cwd /
/// permission / frontend baked into its builtin tools), and an Agent that owns those.
///
/// The returned `ToolRegistry` is the one already attached to the MCP manager; callers (the ACP
/// `session/new` handler) keep a handle so they can pass it to
/// [`crate::mcp::McpClientManager::detach_registry`] on `session/close`.
pub async fn build_session_agent(
    shared: &SharedDeps,
    shared_permission: SharedPermission,
    frontend: Arc<dyn frontend::Frontend>,
    cwd: crate::agent::SharedCwd,
) -> anyhow::Result<(Agent, crate::tools::ToolRegistry)> {
    let bundle = AgentAssembly {
        web_client: shared.config.web_client.clone(),
        sandbox_enabled: shared.config.sandbox,
        sandbox_capability: shared.sandbox_capability.clone(),
        sandbox_backend: shared.config.sandbox_backend,
        backend_probe: shared.config.backend_probe.clone(),
        user_instructions: shared.config.user_instructions.clone(),
        session_manager: shared.session_manager.clone(),
        provider: Arc::clone(&shared.provider),
        mcp_manager: shared.mcp_manager.as_ref(),
        skills: shared.skills.clone(),
        builtin_filter: shared.builtin_filter.clone(),
        agent_options: shared.agent_options.clone(),
        session_stats: Arc::clone(&shared.session_stats),
        subagent_max_depth: shared.config.subagent_max_depth,
    };
    assemble_agent(bundle, shared_permission, frontend, cwd).await
}

// Top-level entry point for assembling the agent; splitting its inputs further would force callers
// to pre-bundle unrelated collaborators (config, session manager, permission mode, credential, MCP
// plumbing, frontend) just to appease the arg-count lint.
#[allow(clippy::too_many_arguments)]
async fn create_agent_from_config(
    config: &ResolvedConfig,
    session_manager: SessionManager,
    shared_permission: SharedPermission,
    token_store: TokenStore,
    credential: AuthCredential,
    mcp_manager: Option<&Arc<mcp::McpClientManager>>,
    frontend: Arc<dyn frontend::Frontend>,
    cwd: crate::agent::SharedCwd,
    session_stats: Arc<stats::SessionStats>,
) -> anyhow::Result<Agent> {
    config.validate()?;

    let provider_name = config
        .provider_name
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("provider_name missing after validation"))?;
    let needs_token_store = matches!(credential, AuthCredential::OAuthToken { .. });

    let model = config
        .model
        .clone()
        .ok_or_else(|| anyhow::anyhow!("model missing after validation"))?;
    let provider = ProviderBuilder::new(provider_name, credential, model)
        .base_url(config.base_url.clone())
        .client_id(config.client_id.clone())
        .credential_key(config.active_profile.clone())
        .oauth_token_url(config.oauth_token_url.clone())
        .token_store(if needs_token_store {
            Some(Arc::new(token_store))
        } else {
            None
        })
        .thinking(config.thinking_enabled, config.thinking_budget_tokens)
        .reasoning_effort(config.reasoning_effort.clone())
        .device_id(config.device_id.clone())
        .effort(config.effort.clone())
        .redact_thinking(config.redact_thinking)
        .max_output_tokens(config.max_output_tokens)
        .session_stats(Some(Arc::clone(&session_stats)))
        .build()?;

    let sandbox_capability: crate::sandbox::SandboxCapability = match &config.backend_probe {
        crate::sandbox::BackendProbe::Ok(capability) => capability.clone(),
        _ => crate::sandbox::SandboxCapability::Unavailable,
    };
    let sandboxed_shell = config.sandbox
        && !matches!(
            sandbox_capability,
            crate::sandbox::SandboxCapability::Unavailable
        );

    // Discover skills once at startup. Any malformed `SKILL.md` emits its `tracing::warn!` here
    // (tracing is already initialized), so the user sees parse errors above the first prompt rather
    // than interleaved with their first turn's output. The cache also drives mid-session auto-
    // reload; `SkillCache::current()` re-snapshots on each turn and re-discovers only when the
    // on-disk state changes.
    let skills = crate::skills::SkillCache::discover();

    let builtin_filter = crate::tools::BuiltinToolFilter::from_config(
        config.builtin_allowed_tools.clone(),
        config.builtin_disabled_tools.clone(),
        config.builtin_tool_permissions.clone(),
    );

    // Build the parent's `AgentOptions` up-front so it can be cloned into `ToolBuilderParams` for
    // sub-agents to inherit `sandboxed_shell` / `context_messages` / `user_instructions` via
    // `Agent::new_subagent`.
    let agent_options = AgentOptions {
        streaming: config.streaming,
        sandboxed_shell,
        context_messages: config.context_messages,
        auto_compact: config.auto_compact,
        context_window: config.context_window.unwrap_or_else(|| {
            config
                .model
                .as_deref()
                .map(crate::config::context_window_for_model)
                .unwrap_or(128_000)
        }),
        user_instructions: config.user_instructions.clone(),
        mcp_strict: config.mcp_strict,
        mcp_grace: config.mcp_grace,
        // Parent builds its system prompt dynamically per-turn via context::build_system_prompt.
        // Sub-agents override.
        system_prompt_override: None,
    };

    let bundle = AgentAssembly {
        web_client: config.web_client.clone(),
        sandbox_enabled: config.sandbox,
        sandbox_capability,
        sandbox_backend: config.sandbox_backend,
        backend_probe: config.backend_probe.clone(),
        user_instructions: config.user_instructions.clone(),
        session_manager: session_manager.clone(),
        provider: Arc::clone(&provider),
        mcp_manager,
        skills: skills.clone(),
        builtin_filter: builtin_filter.clone(),
        agent_options: agent_options.clone(),
        session_stats: Arc::clone(&session_stats),
        subagent_max_depth: config.subagent_max_depth,
    };
    let (agent, _tool_registry) =
        assemble_agent(bundle, shared_permission, frontend, Arc::clone(&cwd)).await?;

    crate::tools::warn_on_stale_builtin_tool_config(&builtin_filter);

    if let Some(manager) = mcp_manager {
        // Kick off the background connector. Each server's adapters are pushed through
        // `manager.update_server_tools` and then fan out to every attached registry. Safe to call
        // after any number of `attach_registry`s; idempotent on second call. (The ACP path does
        // this once in `build_shared_deps`; the REPL path does it here, after `assemble_agent`
        // has attached the single registry.)
        manager.start_connector(crate::mcp::McpRuntimeConfig::from_config(config));
    }

    Ok(agent)
}

/// Run one agent turn with Ctrl+C wired to a fresh cancellation token. Spawns a `ctrl_c()` listener
/// for the turn's duration and aborts it afterward, so a SIGINT during the turn cancels it (and
/// every tool and sub-agent it spawned), while a SIGINT between turns is not consumed by a leaked
/// listener. Every `run_turn` callsite in the REPL / CLI path must go through here; a bare
/// `CancellationToken` with no signal source silently swallows Ctrl+C.
async fn run_turn_interruptible(
    agent: &Agent,
    session_id: &mut Option<uuid::Uuid>,
    messages: &mut conversation::Conversation,
    input: String,
) -> error::Result<()> {
    let cancellation = CancellationToken::new();
    let signal_handle = {
        let cancellation = cancellation.clone();
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                cancellation.cancel();
            }
        })
    };
    let result = agent
        .run_turn(session_id, messages, input, Vec::new(), cancellation)
        .await;
    signal_handle.abort();
    // REPL / `meka -p` callers don't surface a stop reason; they only care whether the turn
    // succeeded. Drop the `TurnOutcome`.
    result.map(|_| ())
}

async fn run_oneshot(
    config: ResolvedConfig,
    session_manager: SessionManager,
    token_store: TokenStore,
    prompt: String,
    mcp_manager: Option<Arc<mcp::McpClientManager>>,
) -> anyhow::Result<()> {
    let shared_permission = SharedPermission::new(config.permission, config.enabled_permissions);
    if config.permission == crate::permission::Permission::Read {
        crate::sandbox::warn_if_sandbox_issues(
            &crate::sandbox::SandboxState::from_config(&config),
            crate::sandbox::WarnContext::InitialReadMode,
        );
    }
    let credential = resolve_credential(&config, &token_store).await?;
    let session_stats = Arc::new(stats::SessionStats::default());
    // Oneshot has no REPL, so approval requests can't reach a human. The channel below is
    // intentionally disconnected on the receiver side: `ReplFrontend::request_permission`'s `send`
    // will fail, and the agent surfaces a `cancelled` tool result, same end behavior as the
    // pre-refactor `None` approval sender.
    let (noninteractive_sender, _) = std::sync::mpsc::channel::<repl::AgentToReplEvent>();
    let oneshot_frontend: Arc<dyn frontend::Frontend> =
        Arc::new(repl::ReplFrontend::new(repl::ReplFrontendConfig {
            render_mode: config.render_mode,
            newline_before_prompt: config.newline_before_prompt,
            newline_after_prompt: config.newline_after_prompt,
            show_session_id_on_create: config.show_session_id_on_create,
            show_token_usage: config.show_token_usage,
            thinking_show_content: config.thinking_show_content,
            agent_event_sender: noninteractive_sender,
        }));
    let cwd: crate::agent::SharedCwd = Arc::new(std::sync::RwLock::new(
        std::env::current_dir().unwrap_or_else(|error| {
            tracing::warn!("could not read process cwd at startup: {}", error);
            std::path::PathBuf::from(".")
        }),
    ));
    let agent = create_agent_from_config(
        &config,
        session_manager,
        shared_permission,
        token_store,
        credential,
        mcp_manager.as_ref(),
        oneshot_frontend,
        cwd,
        Arc::clone(&session_stats),
    )
    .await?;

    let mut session_id = None;
    let mut messages = conversation::Conversation::new();

    match run_turn_interruptible(&agent, &mut session_id, &mut messages, prompt).await {
        Ok(()) => {}
        Err(error::MekaError::Interrupted) => {
            eprintln!("\nInterrupted.");
        }
        Err(error) => return Err(error.into()),
    }

    if let Some(id) = session_id
        && config.show_session_id_on_exit
    {
        render::render_session_id("Leaving session", &id.to_string());
    }

    if let Some(manager) = mcp_manager {
        shutdown_mcp_manager(manager).await;
    }

    Ok(())
}

async fn run_interactive(
    config: ResolvedConfig,
    session_manager: SessionManager,
    token_store: TokenStore,
    initial_prompt: Option<String>,
    mcp_manager: Option<Arc<mcp::McpClientManager>>,
) -> anyhow::Result<()> {
    // Per-session working directory, initialised from process cwd at startup. Shared by reference
    // between the REPL (prompt + `/cd`) and the agent (file/shell/find/grep tools +
    // environment-context block). Process cwd is no longer mutated.
    let cwd: crate::agent::SharedCwd = Arc::new(std::sync::RwLock::new(
        std::env::current_dir().unwrap_or_else(|error| {
            tracing::warn!("could not read process cwd at startup: {}", error);
            std::path::PathBuf::from(".")
        }),
    ));

    let shared_permission = SharedPermission::new(config.permission, config.enabled_permissions);
    if config.permission == crate::permission::Permission::Read {
        crate::sandbox::warn_if_sandbox_issues(
            &crate::sandbox::SandboxState::from_config(&config),
            crate::sandbox::WarnContext::InitialReadMode,
        );
    }

    // Resolve session resumption BEFORE spawning the REPL so the "Resuming session" message appears
    // before the first prompt.
    let (mut session_id, mut messages, mut session_lock) =
        resolve_session_resume(&session_manager, &config).await?;

    if !messages.is_empty() {
        match config.resume_show_recent {
            Some(n) if n > 0 => {
                render::render_message_history(
                    render::last_n_turns(messages.as_slice(), n),
                    &history_render_options(&config),
                );
                // Match the live-turn-end convention: blank line between the rendered content and
                // the first REPL prompt. `reprint_last_message` does the same.
                if config.newline_before_prompt {
                    eprintln!();
                }
            }
            _ => reprint_last_message(messages.as_slice(), config.render_mode),
        }
    }

    let (input_sender, mut input_receiver) = tokio::sync::mpsc::unbounded_channel::<ReplEvent>();

    // If a prompt or skill was given without `--oneshot`, queue it as a synthetic user input so the
    // first turn runs immediately. The REPL takes over afterwards for follow-up turns. The send
    // cannot fail; the receiver was just constructed above. Tracking the flag separately tells the
    // REPL to wait for the synthetic turn's events before drawing its first prompt; otherwise
    // reedline's prompt collides with the agent's output.
    let initial_turn_pending = initial_prompt.is_some();
    if let Some(prompt) = initial_prompt {
        // Channel was constructed two lines above and the receiver is still live (we own it in
        // `input_receiver` below); `send` cannot fail under any runtime condition.
        #[allow(clippy::expect_used)]
        input_sender
            .send(ReplEvent::UserInput(prompt))
            .expect("freshly created input channel must accept first send");
    }
    let (agent_event_sender, agent_event_receiver) =
        std::sync::mpsc::channel::<repl::AgentToReplEvent>();
    // The REPL frontend forwards approval requests to the same channel the REPL thread already
    // reads from for `Done` / MCP elicitation / MCP progress events.
    let repl_frontend: Arc<dyn frontend::Frontend> =
        Arc::new(repl::ReplFrontend::new(repl::ReplFrontendConfig {
            render_mode: config.render_mode,
            newline_before_prompt: config.newline_before_prompt,
            newline_after_prompt: config.newline_after_prompt,
            show_session_id_on_create: config.show_session_id_on_create,
            show_token_usage: config.show_token_usage,
            thinking_show_content: config.thinking_show_content,
            agent_event_sender: agent_event_sender.clone(),
        }));

    // MCP progress / elicitation events now flow through the per-session `Frontend` trait, not the
    // process-global sinks they used to be wired through here. Progress:
    // `ReplFrontend::emit(McpProgress)` and the matching ACP impl carry the event to the right
    // UI. Elicitation: `Frontend::handle_elicitation` runs the round-trip on whichever frontend
    // the in-flight call's `progress::register` recorded. The agent_event_sender is still the
    // bridge between `ReplFrontend` (on the agent's task) and the blocking REPL thread; that
    // wiring happens inside `ReplFrontend` itself.

    let repl_permission = shared_permission.clone();
    let show_path_in_prompt = config.show_path_in_prompt;
    let input_style = config.input_style;
    let repl_sandbox_state = crate::sandbox::SandboxState::from_config(&config);
    let repl_cwd = Arc::clone(&cwd);
    let repl_mcp_server_names: Vec<String> = config
        .mcp_servers
        .iter()
        .map(|server| server.name.clone())
        .collect();
    let repl_history_db_path = Some(session_manager.database_path().to_path_buf());

    // Live context gauge for the prompt: a shared counter the agent writes after each turn and the
    // prompt reads each render. Created here (before the agent) so the REPL, spawned below, can
    // hold it; the agent adopts the same atomic via `set_context_tokens`. Seeded with an
    // estimate when resuming so the gauge isn't blank until the first new turn measures the
    // context exactly.
    let context_window = config.context_window.unwrap_or_else(|| {
        config
            .model
            .as_deref()
            .map(crate::config::context_window_for_model)
            .unwrap_or(128_000)
    });
    let context_tokens = Arc::new(std::sync::atomic::AtomicU64::new(0));
    if !messages.is_empty() {
        context_tokens.store(
            tokens::estimate_messages(messages.as_slice()),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    let context_indicator = config
        .show_context_in_prompt
        .then(|| (Arc::clone(&context_tokens), context_window));

    let repl_handle = tokio::task::spawn_blocking(move || {
        repl::run_repl(
            repl_permission,
            show_path_in_prompt,
            context_indicator,
            input_style,
            initial_turn_pending,
            repl_sandbox_state,
            input_sender,
            agent_event_receiver,
            repl_cwd,
            repl_mcp_server_names,
            repl_history_db_path,
        );
    });

    // Try to create the agent (may fail if config is incomplete)
    let credential = match resolve_credential(&config, &token_store).await {
        Ok(credential) => credential,
        Err(error) => {
            render::render_error(&error);
            render::render_provider_setup_hint();
            drop(agent_event_sender);
            repl_handle.await?;
            return Ok(());
        }
    };
    // A resumed session continues its lifetime `/status` totals; a fresh session (or a load
    // failure) starts empty.
    let session_stats = match session_id {
        Some(id) => match session_manager.load_session_stats(id).await {
            Ok(snapshot) => Arc::new(stats::SessionStats::from_snapshot(&snapshot)),
            Err(error) => {
                tracing::warn!("failed to load session stats, starting fresh: {}", error);
                Arc::new(stats::SessionStats::default())
            }
        },
        None => Arc::new(stats::SessionStats::default()),
    };
    let mut agent = match create_agent_from_config(
        &config,
        session_manager.clone(),
        shared_permission,
        token_store.clone(),
        credential,
        mcp_manager.as_ref(),
        Arc::clone(&repl_frontend),
        Arc::clone(&cwd),
        Arc::clone(&session_stats),
    )
    .await
    {
        Ok(agent) => agent,
        Err(error) => {
            render::render_error(&error);
            render::render_provider_setup_hint();
            drop(agent_event_sender);
            repl_handle.await?;
            return Ok(());
        }
    };
    // Point the agent's live context counter at the same atomic the REPL prompt holds, so the
    // prompt gauge tracks what the agent writes after each turn (and the resume seed above).
    agent.set_context_tokens(Arc::clone(&context_tokens));

    while let Some(event) = input_receiver.recv().await {
        match event {
            ReplEvent::UserInput(input) => {
                match run_turn_interruptible(&agent, &mut session_id, &mut messages, input).await {
                    Ok(()) => {}
                    Err(error::MekaError::Interrupted) => {
                        eprintln!("\nInterrupted.");
                        if config.newline_before_prompt {
                            eprintln!();
                        }
                    }
                    Err(error) => {
                        render::render_error(&error);
                        if config.newline_before_prompt {
                            eprintln!();
                        }
                    }
                }

                // The first turn creates the session if one wasn't resumed; claim the file lock as
                // soon as the ID is known so a second meka invocation can't attach to it.
                if session_lock.is_none()
                    && let Some(id) = session_id
                {
                    match session_manager.lock_session(id) {
                        Ok(lock) => session_lock = Some(lock),
                        Err(error) => render::render_error(&error),
                    }
                }

                if agent_event_sender
                    .send(repl::AgentToReplEvent::Done)
                    .is_err()
                {
                    break;
                }
            }
            ReplEvent::Command(command) => {
                match command {
                    repl::SlashCommand::Session => match &session_id {
                        Some(id) => render::render_session_id("Current session", &id.to_string()),
                        None => eprintln!("No active session yet."),
                    },
                    repl::SlashCommand::Compact => {
                        match agent.compact_session(&mut session_id, &mut messages).await {
                            Ok(()) => {
                                render::render_hint("Session compacted.");
                            }
                            Err(error) => {
                                render::render_error(&error);
                            }
                        }
                    }
                    repl::SlashCommand::Export => match &session_id {
                        Some(id) => {
                            if let Err(error) = export_session(
                                &session_manager,
                                *id,
                                None,
                                cli::SessionExportFormat::Markdown,
                            )
                            .await
                            {
                                render::render_error(&error);
                            }
                        }
                        None => eprintln!("No active session to export."),
                    },
                    repl::SlashCommand::McpList => {
                        if let Err(error) =
                            mcp::cli::run_list(&config.mcp_servers, mcp_manager.as_ref()).await
                        {
                            render::render_error(&error);
                        }
                    }
                    repl::SlashCommand::McpReconnect { server } => {
                        if let Err(error) =
                            mcp::cli::run_reconnect(&config.mcp_servers, &token_store, &server)
                                .await
                        {
                            render::render_error(&error);
                        }
                    }
                    repl::SlashCommand::McpLogin { server } => {
                        if let Err(error) =
                            mcp::cli::run_login(&config.mcp_servers, &token_store, &server).await
                        {
                            render::render_error(&error);
                        }
                    }
                    repl::SlashCommand::McpLogout { server } => {
                        if let Err(error) =
                            mcp::cli::run_logout(&config.mcp_servers, &token_store, &server).await
                        {
                            render::render_error(&error);
                        }
                    }
                    repl::SlashCommand::McpPrompt {
                        server,
                        prompt: prompt_name,
                        args,
                    } => match mcp_manager.as_ref() {
                        Some(manager) => {
                            let entry = manager.server_entry(&server);
                            let Some(entry) = entry else {
                                eprintln!(
                                    "unknown MCP server '{}'; configured: {:?}",
                                    server,
                                    manager.server_names()
                                );
                                continue;
                            };
                            // Map positional args to declared prompt argument names (lookup via
                            // prompts/list).
                            let arg_names = match mcp::list_prompts(&entry).await {
                                Ok(prompts) => prompts
                                    .into_iter()
                                    .find(|p| p.name == prompt_name)
                                    .and_then(|p| p.arguments)
                                    .map(|args| {
                                        args.into_iter().map(|a| a.name).collect::<Vec<_>>()
                                    })
                                    .unwrap_or_default(),
                                Err(error) => {
                                    eprintln!("list_prompts failed: {}", error);
                                    Vec::new()
                                }
                            };
                            let mut arguments: Option<serde_json::Map<String, serde_json::Value>> =
                                None;
                            if !arg_names.is_empty() {
                                let mut map = serde_json::Map::new();
                                for (i, name) in arg_names.iter().enumerate() {
                                    if let Some(value) = args.get(i) {
                                        map.insert(
                                            name.clone(),
                                            serde_json::Value::String(value.clone()),
                                        );
                                    }
                                }
                                arguments = Some(map);
                            }
                            match mcp::get_prompt(&entry, prompt_name.clone(), arguments).await {
                                Ok(result) => {
                                    // Render the prompt messages as a single user turn, same shape
                                    // as the `get_mcp_prompt` tool output.
                                    let mut body = String::new();
                                    for message in &result.messages {
                                        let role = match message.role {
                                            rmcp::model::Role::User => "user",
                                            rmcp::model::Role::Assistant => "assistant",
                                        };
                                        if let rmcp::model::ContentBlock::Text(text) =
                                            &message.content
                                        {
                                            body.push_str(&format!("{}: {}\n", role, text.text));
                                        }
                                    }
                                    let user_input = body.trim().to_string();
                                    if !user_input.is_empty() {
                                        match run_turn_interruptible(
                                            &agent,
                                            &mut session_id,
                                            &mut messages,
                                            user_input,
                                        )
                                        .await
                                        {
                                            Ok(()) => {}
                                            Err(error::MekaError::Interrupted) => {
                                                eprintln!("\nInterrupted.");
                                            }
                                            Err(error) => render::render_error(&error),
                                        }
                                    }
                                }
                                Err(error) => {
                                    eprintln!("get_prompt failed: {}", error);
                                }
                            }
                        }
                        None => {
                            eprintln!("no MCP servers configured");
                        }
                    },
                    repl::SlashCommand::SkillList => {
                        if let Err(error) = skills::cli::run_list().await {
                            render::render_error(&error);
                        }
                    }
                    repl::SlashCommand::SkillInvoke { name, extra } => 'invoke: {
                        // Labeled block so the early-exit error paths can `break 'invoke` out of
                        // the arm body without skipping the `AgentToReplEvent::Done` send below;
                        // `continue` would short-circuit the outer `while let`, leaving the REPL
                        // stuck in `wait_for_agent` and never drawing the next prompt.
                        let installed = agent.skills().current().await;
                        let Some(skill) = installed.iter().find(|s| s.name == name) else {
                            let available: Vec<&str> =
                                installed.iter().map(|s| s.name.as_str()).collect();
                            render::render_error(&format!(
                                "unknown skill '{}'; available: {:?}",
                                name, available
                            ));
                            break 'invoke;
                        };
                        let session_str = session_id.map(|id| id.to_string());
                        let body =
                            match skills::load_skill_body(skill, session_str.as_deref()).await {
                                Ok(body) => body,
                                Err(error) => {
                                    render::render_error(&format!(
                                        "failed to load skill '{}': {}",
                                        name, error
                                    ));
                                    break 'invoke;
                                }
                            };
                        // Prepend the user's free-form directive to the skill body when present.
                        // The blank-line separator gives the model a visual cue that the first
                        // paragraph is the user's "do this skill, but with this twist" and the rest
                        // is the skill's static body.
                        let body = if extra.is_empty() {
                            body
                        } else {
                            format!("{}\n\n{}", extra, body)
                        };
                        match run_turn_interruptible(&agent, &mut session_id, &mut messages, body)
                            .await
                        {
                            Ok(()) => {}
                            Err(error::MekaError::Interrupted) => {
                                eprintln!("\nInterrupted.");
                            }
                            Err(error) => render::render_error(&error),
                        }
                    }
                    repl::SlashCommand::Status => {
                        let snap = agent.session_stats_snapshot();
                        let (context_tokens, context_window) = agent.context_usage();
                        render::render_session_status(
                            &snap,
                            messages.len(),
                            context_tokens,
                            context_window,
                        );
                    }
                    repl::SlashCommand::Usage => match agent.fetch_usage().await {
                        Ok(Some(usage)) => render::render_account_usage(&usage),
                        Ok(None) => {
                            render::render_hint("Account usage isn't available for this provider.")
                        }
                        Err(error) => render::render_error(&error),
                    },
                    repl::SlashCommand::History(limit) => {
                        let materialised = messages.as_slice();
                        let slice = match limit {
                            Some(n) => render::last_n_turns(materialised, n),
                            None => materialised,
                        };
                        // Bracket the rendered history with the same blank-line spacing the live
                        // REPL puts around a regular turn: a `newline_after_prompt` blank between
                        // the user's `/history` line and the rendered content, plus a
                        // `newline_before_prompt` blank between the content and the next REPL
                        // prompt.
                        if config.newline_after_prompt {
                            eprintln!();
                        }
                        render::render_message_history(slice, &history_render_options(&config));
                        if config.newline_before_prompt {
                            eprintln!();
                        }
                    }
                    _ => {}
                }

                if agent_event_sender
                    .send(repl::AgentToReplEvent::Done)
                    .is_err()
                {
                    break;
                }
            }
            ReplEvent::Exit => {
                break;
            }
        }
    }

    drop(agent_event_sender);
    repl_handle.await?;

    if let Some(id) = session_id
        && config.show_session_id_on_exit
    {
        render::render_session_id("Leaving session", &id.to_string());
    }
    // Drop after the "Leaving session" message so the lock is held until the very end; the OS
    // releases the underlying flock when the FD closes.
    drop(session_lock);

    if let Some(manager) = mcp_manager {
        shutdown_mcp_manager(manager).await;
    }

    Ok(())
}

/// Unwrap the shared MCP manager and drive its shutdown. The manager is held behind an `Arc`
/// because resource/prompt tools keep clones of it; once the agent and tool registry have been
/// dropped, try_unwrap should succeed.
async fn shutdown_mcp_manager(manager: Arc<mcp::McpClientManager>) {
    match Arc::try_unwrap(manager) {
        Ok(manager) => manager.shutdown().await,
        Err(_arc) => {
            tracing::debug!(
                "MCP manager still referenced at shutdown; relying on drop guards for cleanup"
            );
        }
    }
}

/// On-wire format version for `meka session export --format json`. Bumped when the envelope shape
/// or the underlying [`crate::conversation::Event`] serialization changes incompatibly; `meka
/// session import` rejects versions it doesn't recognize.
const SESSION_EXPORT_FORMAT_VERSION: u32 = 1;

/// Root envelope for a JSON session export. Carries the session plus any sub-agent descendants as a
/// flat, root-first list; parent links are by original id and get remapped on import. Deliberately
/// secret-free: credentials live in separate global tables and the `token_id` fingerprint is
/// omitted.
#[derive(serde::Serialize, serde::Deserialize)]
struct SessionExport {
    format_version: u32,
    meka_version: String,
    exported_at: String,
    root_session_id: String,
    sessions: Vec<ExportedSession>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ExportedSession {
    id: String,
    parent_id: Option<String>,
    created_at: String,
    updated_at: String,
    cwd: Option<String>,
    permission: Option<String>,
    capabilities_json: Option<String>,
    stats: crate::stats::SessionStatsSnapshot,
    events: Vec<ExportedEvent>,
    tool_outputs: std::collections::BTreeMap<String, String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ExportedEvent {
    /// RFC 3339 timestamp the event row was persisted; preserved across import.
    at: String,
    event: crate::conversation::Event,
}

async fn export_session(
    session_manager: &SessionManager,
    session_id: uuid::Uuid,
    output: Option<&str>,
    format: cli::SessionExportFormat,
) -> anyhow::Result<()> {
    if !session_manager.session_exists(session_id).await? {
        anyhow::bail!("session not found: {}", session_id);
    }

    let (body, default_ext) = match format {
        cli::SessionExportFormat::Markdown => {
            // Export the full event log so pre-compaction turns are included. Compaction only hides
            // older turns from the model (it appends a boundary, never deletes), so the export
            // walks the raw log and renders every turn plus a marker at each compaction point.
            let events = session_manager.load_events(session_id).await?;
            let tool_outputs: std::collections::HashMap<String, String> = session_manager
                .load_all_tool_outputs(session_id)
                .await?
                .into_iter()
                .collect();
            (
                format_session_as_markdown(session_id, &events, &tool_outputs),
                "md",
            )
        }
        cli::SessionExportFormat::Json => {
            let export = build_session_export(session_manager, session_id).await?;
            (serde_json::to_string_pretty(&export)?, "json")
        }
    };

    match output {
        Some("-") => {
            print!("{}", body);
        }
        Some(path) => {
            std::fs::write(path, &body)?;
            tracing::info!("exported session to {}", path);
        }
        None => {
            let path = format!("session-{}.{}", session_id, default_ext);
            std::fs::write(&path, &body)?;
            tracing::info!("exported session to {}", path);
        }
    }

    Ok(())
}

/// Assemble the structured JSON export envelope for a session and every sub-agent descendant.
/// Per-event timestamps and cumulative stats are preserved; `token_id` is intentionally excluded.
async fn build_session_export(
    session_manager: &SessionManager,
    root: uuid::Uuid,
) -> anyhow::Result<SessionExport> {
    let tree = session_manager.load_session_tree(root).await?;
    let mut sessions = Vec::with_capacity(tree.len());
    for meta in tree {
        let events = session_manager
            .load_events_with_timestamps(meta.id)
            .await?
            .into_iter()
            .map(|(at, event)| ExportedEvent { at, event })
            .collect();
        let tool_outputs = session_manager
            .load_all_tool_outputs(meta.id)
            .await?
            .into_iter()
            .collect();
        let stats = session_manager.load_session_stats(meta.id).await?;
        sessions.push(ExportedSession {
            id: meta.id.to_string(),
            parent_id: meta.parent_id.map(|id| id.to_string()),
            created_at: meta.created_at,
            updated_at: meta.updated_at,
            cwd: meta.cwd,
            permission: meta.permission,
            capabilities_json: meta.capabilities_json,
            stats,
            events,
            tool_outputs,
        });
    }
    Ok(SessionExport {
        format_version: SESSION_EXPORT_FORMAT_VERSION,
        meka_version: env!("CARGO_PKG_VERSION").to_string(),
        exported_at: chrono::Utc::now().to_rfc3339(),
        root_session_id: root.to_string(),
        sessions,
    })
}

/// Import a session (and any sub-agent children) from a JSON export produced by
/// `meka session export --format json`. Reads `input` (a file path, or `-` for stdin), mints fresh
/// IDs for every session, rewires parent links, and persists the whole tree in one transaction.
/// Prints the new root session ID to stdout.
async fn import_session(session_manager: &SessionManager, input: &str) -> anyhow::Result<()> {
    let raw = if input == "-" {
        use std::io::Read as _;
        let mut buffer = String::new();
        std::io::stdin().read_to_string(&mut buffer)?;
        buffer
    } else {
        std::fs::read_to_string(input)
            .map_err(|error| anyhow::anyhow!("failed to read '{}': {}", input, error))?
    };

    let export: SessionExport = serde_json::from_str(&raw)
        .map_err(|error| anyhow::anyhow!("invalid session export JSON: {}", error))?;
    let (records, root_new_id) = plan_import(export)?;

    let count = records.len();
    session_manager.import_sessions(records).await?;
    tracing::info!("imported {} session(s) from {}", count, input);
    // Human-facing confirmation and resume guidance go to stderr; the bare root ID stays on stdout
    // so `id=$(meka session import ...)` and piping keep working. Plain (unstyled) to match the
    // other one-shot CLI messages; `render_hint`'s dark-grey styling is for the REPL.
    if count > 1 {
        eprintln!(
            "Imported session with {} sub-agent(s). Resume with: meka -c {}",
            count - 1,
            root_new_id
        );
    } else {
        eprintln!("Imported session. Resume with: meka -c {}", root_new_id);
    }
    println!("{}", root_new_id);
    Ok(())
}

/// Turn a deserialized [`SessionExport`] into the parents-first
/// [`crate::session::ImportSessionRecord`] list to persist, plus the freshly-minted root session
/// ID. Validates the format version, mints a new ID per session, and remaps parent links (a parent
/// pointing outside the exported set collapses to `None`, importing that session as a new top-level
/// session). Pure and I/O-free so the ID-remap and ordering are unit-testable.
fn plan_import(
    export: SessionExport,
) -> anyhow::Result<(Vec<crate::session::ImportSessionRecord>, uuid::Uuid)> {
    if export.format_version != SESSION_EXPORT_FORMAT_VERSION {
        anyhow::bail!(
            "unsupported session export format_version {} (this build supports {})",
            export.format_version,
            SESSION_EXPORT_FORMAT_VERSION
        );
    }
    if export.sessions.is_empty() {
        anyhow::bail!("session export contains no sessions");
    }

    let remap: std::collections::HashMap<String, uuid::Uuid> = export
        .sessions
        .iter()
        .map(|session| (session.id.clone(), uuid::Uuid::new_v4()))
        .collect();
    let root_new_id = remap
        .get(&export.root_session_id)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("root_session_id is not present in the sessions list"))?;

    let nodes: Vec<(String, Option<String>)> = export
        .sessions
        .iter()
        .map(|session| (session.id.clone(), session.parent_id.clone()))
        .collect();
    let order = parents_first_order(&nodes)?;

    let mut slots: Vec<Option<ExportedSession>> = export.sessions.into_iter().map(Some).collect();
    let mut records = Vec::with_capacity(order.len());
    for index in order {
        let session = slots[index]
            .take()
            .ok_or_else(|| anyhow::anyhow!("duplicate session index while ordering import"))?;
        let new_id = remap
            .get(&session.id)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("internal error: session id missing from ID remap"))?;
        let new_parent_id = session
            .parent_id
            .as_ref()
            .and_then(|parent| remap.get(parent).copied());
        records.push(crate::session::ImportSessionRecord {
            new_id,
            new_parent_id,
            created_at: session.created_at,
            updated_at: session.updated_at,
            cwd: session.cwd,
            permission: session.permission,
            capabilities_json: session.capabilities_json,
            stats: session.stats,
            events: session
                .events
                .into_iter()
                .map(|event| (event.at, event.event))
                .collect(),
            tool_outputs: session.tool_outputs.into_iter().collect(),
        });
    }

    Ok((records, root_new_id))
}

/// Order sessions parents-first (a topological sort over `parent_id` edges, considering only
/// parents present in the set) so an importer can insert each session after its parent and satisfy
/// the `parent_session_id` foreign key. Returns indices into `nodes`. Errors on a cyclic
/// relationship. Sessions whose parent is absent from the set are treated as roots.
fn parents_first_order(nodes: &[(String, Option<String>)]) -> anyhow::Result<Vec<usize>> {
    use std::collections::{HashMap, VecDeque};

    let index_of: HashMap<&str, usize> = nodes
        .iter()
        .enumerate()
        .map(|(index, (id, _))| (id.as_str(), index))
        .collect();
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    let mut indegree = vec![0usize; nodes.len()];
    for (index, (_, parent)) in nodes.iter().enumerate() {
        if let Some(parent) = parent
            && let Some(&parent_index) = index_of.get(parent.as_str())
        {
            children[parent_index].push(index);
            indegree[index] += 1;
        }
    }
    let mut queue: VecDeque<usize> = (0..nodes.len()).filter(|&i| indegree[i] == 0).collect();
    let mut order = Vec::with_capacity(nodes.len());
    while let Some(node) = queue.pop_front() {
        order.push(node);
        for &child in &children[node] {
            indegree[child] -= 1;
            if indegree[child] == 0 {
                queue.push_back(child);
            }
        }
    }
    if order.len() != nodes.len() {
        anyhow::bail!("session export has a cyclic parent relationship");
    }
    Ok(order)
}

async fn run_mcp_subcommand(
    session_manager: &SessionManager,
    action: &cli::McpAction,
    cli_args: &cli::Cli,
) -> anyhow::Result<()> {
    let config = ResolvedConfig::from_cli(cli_args);
    let token_store = session_manager.token_store();
    match action {
        cli::McpAction::List => mcp::cli::run_list(&config.mcp_servers, None).await?,
        cli::McpAction::Get { name } => mcp::cli::run_get(&config.mcp_servers, name).await?,
        cli::McpAction::Reconnect { name } => {
            mcp::cli::run_reconnect(&config.mcp_servers, &token_store, name).await?
        }
        cli::McpAction::Tools { name } => {
            mcp::cli::run_tools(
                &config.mcp_servers,
                config.mcp_default_permission,
                &token_store,
                name,
            )
            .await?
        }
        cli::McpAction::Login { name } => {
            mcp::cli::run_login(&config.mcp_servers, &token_store, name).await?
        }
        cli::McpAction::Logout { name } => {
            mcp::cli::run_logout(&config.mcp_servers, &token_store, name).await?
        }
        cli::McpAction::Add {
            name,
            location,
            args,
            transport,
            env,
            header,
            auth,
            auth_token,
            client_id,
            client_secret,
            signing_key,
            signing_algorithm,
            scope,
            redirect_port,
            permission,
            no_login,
            allow_tool,
            disable_tool,
            eager_load_tool,
            tool_permission,
            disabled,
        } => {
            mcp::cli::run_add(
                mcp::cli::AddArgs {
                    name: name.clone(),
                    location: location.clone(),
                    args: args.clone(),
                    transport: transport.clone(),
                    env: env.clone(),
                    header: header.clone(),
                    auth: auth.clone(),
                    auth_token: auth_token.clone(),
                    client_id: client_id.clone(),
                    client_secret: client_secret.clone(),
                    signing_key: signing_key.clone(),
                    signing_algorithm: signing_algorithm.clone(),
                    scope: scope.clone(),
                    redirect_port: *redirect_port,
                    permission: permission.clone(),
                    no_login: *no_login,
                    allow_tool: allow_tool.clone(),
                    disable_tool: disable_tool.clone(),
                    eager_load_tool: eager_load_tool.clone(),
                    tool_permission: tool_permission.clone(),
                    disabled: *disabled,
                },
                &token_store,
            )
            .await?
        }
        cli::McpAction::Remove { name } => mcp::cli::run_remove(name, &token_store).await?,
        cli::McpAction::Disable { name } => mcp::cli::run_disable(name).await?,
        cli::McpAction::Enable { name } => mcp::cli::run_enable(name).await?,
    }
    Ok(())
}

/// Handle `meka tools <action>`.
async fn run_tools_subcommand(
    action: &cli::ToolsAction,
    cli_args: &cli::Cli,
) -> anyhow::Result<()> {
    match action {
        cli::ToolsAction::List => {
            let config = ResolvedConfig::from_cli(cli_args);
            let filter = crate::tools::BuiltinToolFilter::from_config(
                config.builtin_allowed_tools.clone(),
                config.builtin_disabled_tools.clone(),
                config.builtin_tool_permissions.clone(),
            );
            crate::tools::warn_on_stale_builtin_tool_config(&filter);

            // Build with no filter so the catalogue carries every tool's hardcoded level; overlay
            // the real filter for status/source.
            let session_manager = SessionManager::open(None).await?;
            let shared_permission =
                SharedPermission::new(config.permission, config.enabled_permissions);
            let sandbox_capability = match &config.backend_probe {
                crate::sandbox::BackendProbe::Ok(capability) => capability.clone(),
                _ => crate::sandbox::SandboxCapability::Unavailable,
            };
            let todo_list: crate::tools::todo::SharedTodoList = std::sync::Arc::new(
                tokio::sync::RwLock::new(crate::tools::todo::TodoState::default()),
            );
            let shared_session_id: std::sync::Arc<tokio::sync::RwLock<Option<uuid::Uuid>>> =
                std::sync::Arc::new(tokio::sync::RwLock::new(None));
            let reference = ToolRegistry::build_default(
                config.web_client.clone(),
                shared_permission,
                config.sandbox,
                sandbox_capability,
                config.sandbox_backend,
                config.backend_probe.clone(),
                todo_list,
                session_manager,
                shared_session_id,
                // `meka tools list` only prints the tool catalogue; skill metadata isn't read, so
                // skip the filesystem walk.
                crate::skills::SkillCache::for_root(None),
                crate::tools::BuiltinToolFilter::default(),
                std::sync::Arc::new(std::sync::RwLock::new(std::path::PathBuf::from("."))),
                std::sync::Arc::new(crate::frontend::SilentFrontend),
            )?;

            let catalogue = reference.tool_catalogue();
            println!(
                "{:<20} {:<9} {:<9} {:<10} description",
                "NAME", "REQUIRED", "SOURCE", "VISIBILITY"
            );
            println!("{}", "-".repeat(78));
            for (name, description, required, is_deferred) in &catalogue {
                let override_entry = filter.permission_overrides.get(name);
                let effective = override_entry.copied().unwrap_or(*required);
                let source = if override_entry.is_some() {
                    "override"
                } else {
                    "builtin"
                };
                let visibility = if filter.admits(name) {
                    if *is_deferred { "deferred" } else { "enabled" }
                } else {
                    "disabled"
                };
                let short = description
                    .lines()
                    .next()
                    .unwrap_or("")
                    .chars()
                    .take(40)
                    .collect::<String>();
                println!(
                    "{:<20} {:<9} {:<9} {:<10} {}",
                    name,
                    effective.to_string(),
                    source,
                    visibility,
                    short
                );
            }
        }
    }
    Ok(())
}

async fn run_skill_subcommand(action: &cli::SkillAction) -> anyhow::Result<()> {
    match action {
        cli::SkillAction::List => skills::cli::run_list().await?,
        cli::SkillAction::Get { name } => skills::cli::run_get(name).await?,
        cli::SkillAction::Show { name } => skills::cli::run_show(name).await?,
        cli::SkillAction::Add {
            name,
            description,
            version,
            author,
            source_url,
            from_file,
            force,
            edit,
        } => {
            skills::cli::run_add(skills::cli::AddArgs {
                name,
                description: description.as_deref(),
                version: version.as_deref(),
                author: author.as_deref(),
                source_url: source_url.as_deref(),
                from_file: from_file.as_deref(),
                force: *force,
                edit: *edit,
            })
            .await?
        }
        cli::SkillAction::Remove { name } => skills::cli::run_remove(name).await?,
        cli::SkillAction::Update { name, all, yes } => {
            skills::cli::run_update(name.as_deref(), *all, *yes).await?
        }
    }
    Ok(())
}

/// Local (no-network) auth status for a stored credential: is the token valid, and when does it
/// expire. Serialized as the `auth` block of `meka account whoami --format json`.
#[derive(serde::Serialize)]
struct AuthStatus {
    valid: bool,
    /// Token expiry as Unix seconds (`None` for API keys / no expiry).
    expires_at: Option<i64>,
    /// Seconds until expiry (negative if already expired).
    expires_in_seconds: Option<i64>,
}

impl AuthStatus {
    fn from_credential(credential: &AuthCredential) -> Self {
        match credential {
            AuthCredential::OAuthToken { expires_at, .. } => {
                // `expires_at` is stored as epoch milliseconds.
                let expires_at = expires_at.map(|millis| millis / 1000);
                let expires_in_seconds =
                    expires_at.map(|secs| secs - chrono::Utc::now().timestamp());
                AuthStatus {
                    valid: expires_in_seconds.is_none_or(|remaining| remaining > 0),
                    expires_at,
                    expires_in_seconds,
                }
            }
            AuthCredential::ApiKey(_) => AuthStatus {
                valid: true,
                expires_at: None,
                expires_in_seconds: None,
            },
        }
    }
}

#[derive(serde::Serialize)]
struct UsageOutput<'a> {
    provider: &'a str,
    #[serde(flatten)]
    usage: &'a crate::provider::AccountUsage,
}

#[derive(serde::Serialize)]
struct WhoamiOutput<'a> {
    provider: &'a str,
    backend: &'a str,
    auth: AuthStatus,
    identity: Option<crate::provider::AccountIdentity>,
}

/// `meka account { usage, whoami }`: resolve a profile, build just that provider (no agent/MCP/
/// session), fetch the requested info, and print it (plain to stdout, or JSON). Requested data goes
/// to stdout; the "not available" / error notes go to stderr, so `… 2>/dev/null | jq` stays clean.
async fn run_account_subcommand(
    session_manager: &SessionManager,
    action: &cli::AccountAction,
) -> anyhow::Result<()> {
    let (profile_arg, format) = match action {
        cli::AccountAction::Usage { profile, format } => (profile.clone(), *format),
        cli::AccountAction::Whoami { profile, format } => (profile.clone(), *format),
        cli::AccountAction::Stats { profile, format } => (profile.clone(), *format),
    };

    let token_store = session_manager.token_store();
    let config_file = config::load_config_file();
    let requested = profile_arg.or_else(|| config_file.default_provider.clone());
    let (name, error) = config::select_active_profile(requested, &config_file.providers);
    let name = name.ok_or_else(|| {
        anyhow::anyhow!(error.unwrap_or_else(|| "no provider configured".to_string()))
    })?;
    let profile = config_file
        .providers
        .get(&name)
        .ok_or_else(|| anyhow::anyhow!("provider profile '{}' not found", name))?;
    let credential = token_store
        .load_provider_credential(&name)
        .await?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no stored credential for profile '{}'. Run `meka provider login {}`.",
                name,
                name
            )
        })?;

    let provider = ProviderBuilder::new(
        profile.backend.clone(),
        credential.clone(),
        profile.model.clone().unwrap_or_default(),
    )
    .base_url(profile.base_url.clone())
    .client_id(profile.client_id.clone())
    .oauth_token_url(profile.oauth_token_url.clone())
    .credential_key(Some(name.clone()))
    .token_store(Some(std::sync::Arc::new(session_manager.token_store())))
    .build()?;

    match action {
        cli::AccountAction::Usage { .. } => match provider.fetch_usage().await? {
            Some(usage) => match format {
                cli::OutputFormat::Plain => {
                    print!("{}", render::format_account_usage(&usage));
                }
                cli::OutputFormat::Json => {
                    let out = UsageOutput {
                        provider: &name,
                        usage: &usage,
                    };
                    println!("{}", serde_json::to_string_pretty(&out)?);
                }
            },
            None => {
                eprintln!("Account usage isn't available for provider '{}'.", name);
                std::process::exit(1);
            }
        },
        cli::AccountAction::Whoami { .. } => {
            // The identity call may refresh + rotate the token; re-read afterwards so the auth
            // block reflects the current expiry. A failed identity fetch (e.g. re-login needed)
            // still prints the local auth status so scripts can detect it.
            let identity = provider.fetch_identity().await;
            let fresh = session_manager
                .token_store()
                .load_provider_credential(&name)
                .await
                .ok()
                .flatten()
                .unwrap_or(credential);
            let auth = AuthStatus::from_credential(&fresh);
            let identity = match identity {
                Ok(identity) => identity,
                Err(error) => {
                    eprintln!("warning: could not fetch identity: {}", error);
                    None
                }
            };
            let out = WhoamiOutput {
                provider: &name,
                backend: &profile.backend,
                auth,
                identity,
            };
            match format {
                cli::OutputFormat::Plain => print!("{}", format_whoami_plain(&out)),
                cli::OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&out)?),
            }
            if !out.auth.valid {
                std::process::exit(1);
            }
        }
        cli::AccountAction::Stats { .. } => match provider.fetch_history().await? {
            Some(history) => {
                let out = StatsOutput {
                    provider: &name,
                    history: &history,
                };
                match format {
                    cli::OutputFormat::Plain => print!("{}", format_stats_plain(&out)),
                    cli::OutputFormat::Json => {
                        println!("{}", serde_json::to_string_pretty(&out)?)
                    }
                }
            }
            None => {
                eprintln!("Account history isn't available for provider '{}'.", name);
                std::process::exit(1);
            }
        },
    }
    Ok(())
}

#[derive(serde::Serialize)]
struct StatsOutput<'a> {
    provider: &'a str,
    #[serde(flatten)]
    history: &'a crate::provider::UsageHistory,
}

/// Plain-text (ANSI-free) rendering of `meka account stats`.
fn format_stats_plain(out: &StatsOutput<'_>) -> String {
    use std::fmt::Write as _;
    let history = out.history;
    let mut text = format!("Account history: {}\n", out.provider);
    let row_tokens = |text: &mut String, label: &str, value: Option<i64>| {
        if let Some(value) = value {
            let _ = writeln!(
                text,
                "  {label:<18} {}",
                render::format_token_count(value.max(0) as u64)
            );
        }
    };
    let row_days = |text: &mut String, label: &str, value: Option<i64>| {
        if let Some(value) = value {
            let _ = writeln!(text, "  {label:<18} {value} days");
        }
    };
    if let Some(first) = &history.first_used {
        // Trim an RFC 3339 timestamp to just the date for the human view.
        let date = first.split('T').next().unwrap_or(first);
        let _ = writeln!(text, "  {:<18} {date}", "First used:");
    }
    row_tokens(&mut text, "Lifetime tokens:", history.lifetime_tokens);
    row_tokens(&mut text, "Peak daily:", history.peak_daily_tokens);
    row_days(&mut text, "Current streak:", history.current_streak_days);
    row_days(&mut text, "Longest streak:", history.longest_streak_days);
    if !history.daily.is_empty() {
        let _ = writeln!(text, "  Recent:");
        for day in history.daily.iter().rev().take(7) {
            let _ = writeln!(
                text,
                "    {}  {}",
                day.date,
                render::format_token_count(day.tokens.max(0) as u64)
            );
        }
    }
    text
}

/// Plain-text (ANSI-free) rendering of `meka account whoami`.
fn format_whoami_plain(out: &WhoamiOutput<'_>) -> String {
    use std::fmt::Write as _;
    let mut text = format!("Account: {} ({})\n", out.provider, out.backend);
    let auth = match (out.auth.valid, out.auth.expires_in_seconds) {
        (true, Some(secs)) => format!("valid ({})", render::format_duration_short(secs.max(0))),
        (true, None) => "valid".to_string(),
        (false, _) => "EXPIRED: run `meka provider login`".to_string(),
    };
    let _ = writeln!(text, "  Auth:          {auth}");
    if let Some(identity) = &out.identity {
        let row = |text: &mut String, label: &str, value: &Option<String>| {
            if let Some(value) = value {
                let _ = writeln!(text, "  {label:<14} {value}");
            }
        };
        row(&mut text, "Name:", &identity.display_name);
        row(&mut text, "Email:", &identity.email);
        row(&mut text, "Plan:", &identity.plan);
        row(&mut text, "Tier:", &identity.tier);
        row(&mut text, "Subscription:", &identity.subscription_status);
        row(&mut text, "Organization:", &identity.organization);
        row(&mut text, "Role:", &identity.role);
    }
    text
}

async fn run_session_subcommand(
    session_manager: &SessionManager,
    action: &cli::SessionAction,
) -> anyhow::Result<()> {
    match action {
        cli::SessionAction::List {
            limit,
            include_children,
        } => list_sessions(session_manager, *limit, *include_children).await,
        cli::SessionAction::Export {
            session_id,
            output,
            format,
        } => export_session(session_manager, *session_id, output.as_deref(), *format).await,
        cli::SessionAction::Delete { session_ids, all } => {
            delete_sessions(session_manager, session_ids, *all).await
        }
        cli::SessionAction::Import { input } => import_session(session_manager, input).await,
    }
}

async fn run_history_subcommand(
    session_manager: &SessionManager,
    action: &cli::HistoryAction,
) -> anyhow::Result<()> {
    // Capacity only gates the write/prune path, so `0` is fine for read/clear. The table is created
    // lazily by `open`, so this is safe on a fresh database.
    let history = crate::history::PromptHistory::open(session_manager.database_path(), 0)?;
    match action {
        cli::HistoryAction::List { limit } => {
            let entries = history.recent(*limit as usize)?;
            if entries.is_empty() {
                println!("No input history.");
            } else {
                for entry in entries {
                    println!("{}", entry);
                }
            }
        }
        cli::HistoryAction::Clear => {
            let removed = history.clear_all()?;
            let noun = if removed == 1 { "entry" } else { "entries" };
            println!("Cleared {} input history {}.", removed, noun);
        }
    }
    Ok(())
}

async fn list_sessions(
    session_manager: &SessionManager,
    limit: u32,
    include_children: bool,
) -> anyhow::Result<()> {
    let (sessions, _next_cursor) = session_manager
        .list_sessions(limit, include_children, None, None)
        .await?;

    if sessions.is_empty() {
        println!("No sessions found.");
        return Ok(());
    }

    let rows: Vec<Vec<String>> = sessions
        .iter()
        .map(|session| {
            vec![
                session.id.to_string(),
                format_timestamp(&session.updated_at),
                session.preview.clone(),
            ]
        })
        .collect();
    print!(
        "{}",
        render::format_columns(&["ID", "Updated", "Preview"], &rows)
    );

    Ok(())
}

async fn delete_sessions(
    session_manager: &SessionManager,
    session_ids: &[uuid::Uuid],
    all: bool,
) -> anyhow::Result<()> {
    if all {
        let deleted = session_manager.delete_all_sessions().await?;
        tracing::info!("deleted {} session(s)", deleted);
        return Ok(());
    }

    if session_ids.is_empty() {
        anyhow::bail!("specify one or more session IDs, or use --all to delete all sessions");
    }

    let mut deleted = 0u64;
    for session_id in session_ids {
        if session_manager.delete_session(*session_id).await? {
            deleted += 1;
        } else {
            // User-facing error: they asked to delete a specific ID and we couldn't find it, so
            // stderr (not silent) is right.
            eprintln!("Session not found: {}", session_id);
        }
    }

    tracing::info!("deleted {} session(s)", deleted);
    Ok(())
}

fn format_timestamp(rfc3339: &str) -> String {
    chrono::DateTime::parse_from_rfc3339(rfc3339)
        .map(|datetime| datetime.format("%Y-%m-%d %H:%M:%S").to_string())
        .unwrap_or_else(|_| rfc3339.to_string())
}

pub(crate) fn format_session_as_markdown(
    session_id: uuid::Uuid,
    events: &[conversation::Event],
    tool_outputs: &std::collections::HashMap<String, String>,
) -> String {
    use std::fmt::Write;

    let mut output = String::new();
    writeln!(output, "# Session {}\n", session_id).ok();

    // Walk the raw event log so the full conversation is exported, including turns a compaction
    // later hid from the model. Each `CompactBoundary` becomes a marker; the turns it summarized
    // stay above it (the kept tail is re-appended after it, so the recent turns appear on both
    // sides of the marker, as stored).
    for event in events {
        match event {
            conversation::Event::Append(message) => {
                write_message_markdown(&mut output, message, tool_outputs);
            }
            conversation::Event::CompactBoundary { summary, .. } => {
                writeln!(output, "---\n").ok();
                writeln!(output, "<details>").ok();
                writeln!(
                    output,
                    "<summary>Session compaction (summary the model saw in place of the turns above)</summary>\n"
                )
                .ok();
                writeln!(output, "{}\n", summary.text_content()).ok();
                writeln!(output, "</details>\n").ok();
            }
        }
    }

    output
}

fn write_message_markdown(
    output: &mut String,
    message: &provider::Message,
    tool_outputs: &std::collections::HashMap<String, String>,
) {
    use std::fmt::Write;

    match message.role {
        provider::Role::User => {
            // A "user" message can be either a plain user turn or a tool_results envelope.
            // Inspect content blocks rather than role to decide.
            let has_tool_results = message
                .content
                .iter()
                .any(|block| matches!(block, provider::ContentBlock::ToolResult { .. }));
            if has_tool_results {
                for block in &message.content {
                    if let provider::ContentBlock::ToolResult {
                        content, is_error, ..
                    } = block
                    {
                        let label = if *is_error {
                            "Tool result (error)"
                        } else {
                            "Tool result"
                        };
                        writeln!(output, "<details>").ok();
                        writeln!(output, "<summary>{}</summary>\n", label).ok();
                        let text = provider::ContentBlock::tool_result_text_content(content);
                        let text = resolve_large_output_tags(&text, tool_outputs);
                        writeln!(output, "```\n{}\n```\n", text).ok();
                        writeln!(output, "</details>\n").ok();
                    }
                }
            } else {
                writeln!(output, "## User\n").ok();
                writeln!(output, "{}\n", message.text_content()).ok();
            }
        }
        provider::Role::Assistant => {
            writeln!(output, "## Assistant\n").ok();
            for block in &message.content {
                match block {
                    provider::ContentBlock::Text { text } => {
                        writeln!(output, "{}\n", text).ok();
                    }
                    provider::ContentBlock::ToolUse { name, input, .. } => {
                        let input_pretty = serde_json::to_string_pretty(input)
                            .unwrap_or_else(|_| input.to_string());
                        writeln!(output, "<details>").ok();
                        writeln!(output, "<summary>Tool call: {}</summary>\n", name).ok();
                        writeln!(output, "```json\n{}\n```\n", input_pretty).ok();
                        writeln!(output, "</details>\n").ok();
                    }
                    provider::ContentBlock::ToolResult { .. }
                    | provider::ContentBlock::Thinking { .. }
                    | provider::ContentBlock::RedactedThinking { .. }
                    | provider::ContentBlock::Image { .. } => {}
                }
            }
        }
    }
}

fn resolve_large_output_tags(
    text: &str,
    tool_outputs: &std::collections::HashMap<String, String>,
) -> String {
    let re = match regex::Regex::new(r#"<large-output name="([^"]+)"[^>]*>[\s\S]*?</large-output>"#)
    {
        Ok(re) => re,
        Err(_) => return text.to_string(),
    };

    re.replace_all(text, |caps: &regex::Captures| {
        let name = &caps[1];
        match tool_outputs.get(name) {
            Some(content) => content.clone(),
            None => caps[0].to_string(),
        }
    })
    .into_owned()
}

/// Translate the live-REPL display config into the options that [`render::render_message_history`]
/// consumes. Keeps the spacing / styling rules between live output and history rendering in sync
/// from a single source of truth.
fn history_render_options(config: &ResolvedConfig) -> render::HistoryRenderOptions {
    render::HistoryRenderOptions {
        render_mode: config.render_mode,
        show_thinking: config.thinking_show_content,
        input_style: config.input_style,
        newline_before_prompt: config.newline_before_prompt,
        newline_after_prompt: config.newline_after_prompt,
    }
}

async fn resolve_credential(
    config: &ResolvedConfig,
    token_store: &TokenStore,
) -> anyhow::Result<AuthCredential> {
    let Some(profile) = config.active_profile.as_deref() else {
        anyhow::bail!("no provider configured. Run `meka provider add <name>` to set one up.");
    };
    match token_store.load_provider_credential(profile).await? {
        Some(credential) => Ok(credential),
        None => Err(anyhow::anyhow!(
            "provider profile '{}' has no stored credential. Run `meka provider login {}` to \
             authenticate.",
            profile,
            profile
        )),
    }
}

fn reprint_last_message(messages: &[provider::Message], render_mode: render::RenderMode) {
    let Some(last) = messages.last() else {
        return;
    };

    let text = match last.role {
        provider::Role::Assistant => {
            let text = last.text_content();
            if text.is_empty() {
                return;
            }
            text
        }
        provider::Role::User => {
            let raw = last.text_content();
            let stripped = session::strip_context_tags(&raw);
            if stripped.is_empty() {
                return;
            }
            stripped.to_string()
        }
    };

    let mut renderer = render::StreamingRenderer::new(render_mode);
    if let Err(error) = renderer.push_delta(&text) {
        tracing::debug!("failed to render last message delta: {}", error);
    }
    if let Err(error) = renderer.finish() {
        tracing::debug!("failed to finish rendering last message: {}", error);
    }
    eprintln!();
}

async fn resolve_session_resume(
    session_manager: &SessionManager,
    config: &ResolvedConfig,
) -> anyhow::Result<(
    Option<uuid::Uuid>,
    conversation::Conversation,
    Option<session::SessionLock>,
)> {
    let Some(value) = &config.continue_session else {
        return Ok((None, conversation::Conversation::new(), None));
    };

    if value == "last" {
        match session_manager.last_session_id().await? {
            Some(id) => {
                let lock = session_manager.lock_session(id)?;
                render::render_session_id("Continuing session", &id.to_string());
                if config.newline_after_prompt {
                    eprintln!();
                }
                let messages = load_session_messages(session_manager, id).await?;
                Ok((Some(id), messages, Some(lock)))
            }
            None => Ok((None, conversation::Conversation::new(), None)),
        }
    } else {
        let id = resolve_session_id(session_manager, value).await?;
        let lock = session_manager.lock_session(id)?;
        render::render_session_id("Continuing session", &id.to_string());
        if config.newline_after_prompt {
            eprintln!();
        }
        let messages = load_session_messages(session_manager, id).await?;
        Ok((Some(id), messages, Some(lock)))
    }
}

/// Resolve `meka -c <value>` (where `value` is not "last") to a single session UUID. Tries a
/// full-UUID parse first; if that fails, falls back to a prefix lookup so users can type just the
/// leading hex chars.
///
/// Errors out cleanly when the prefix matches zero or multiple sessions.
async fn resolve_session_id(
    session_manager: &SessionManager,
    value: &str,
) -> anyhow::Result<uuid::Uuid> {
    if let Ok(id) = value.parse::<uuid::Uuid>() {
        if !session_manager.session_exists(id).await? {
            anyhow::bail!("session not found: {}", id);
        }
        return Ok(id);
    }

    let matches = session_manager.find_sessions_by_prefix(value).await?;
    match matches.len() {
        0 => anyhow::bail!("no session matches prefix '{}'", value),
        1 => Ok(matches[0]),
        _ => {
            let listing: Vec<String> = matches.iter().map(|id| id.to_string()).collect();
            anyhow::bail!(
                "ambiguous prefix '{}' matches {} sessions: {}",
                value,
                matches.len(),
                listing.join(", "),
            )
        }
    }
}

async fn load_session_messages(
    session_manager: &SessionManager,
    session_id: uuid::Uuid,
) -> anyhow::Result<conversation::Conversation> {
    // Hydrate the event log directly. Legacy databases (rows predating the event-log refactor)
    // decode their `user`/`assistant`/`tool_results` rows as `Event::Append` so resume is forward-
    // and backward- compatible without a schema migration.
    let events = session_manager.load_events(session_id).await?;
    let mut log = conversation::Conversation::from_events(events);

    // Drop assistant messages whose tool_use blocks lack matching tool_result blocks in the next
    // message. Anthropic's API rejects orphans; this sanitizes the log after a crash mid-tool-call.
    let dropped = log.sanitize_orphans();
    for message in &dropped {
        let tool_use_ids: Vec<String> = message
            .content
            .iter()
            .filter_map(|block| {
                if let provider::ContentBlock::ToolUse { id, .. } = block {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect();
        tracing::warn!(
            "dropping assistant message with orphaned tool_use IDs: {:?}",
            tool_use_ids,
        );
    }

    Ok(log)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parents_first_order_orders_parents_before_children() {
        // Given out of order (child, root, middle), each node must land after its parent.
        let nodes = vec![
            ("c".to_string(), Some("b".to_string())),
            ("a".to_string(), None),
            ("b".to_string(), Some("a".to_string())),
        ];
        let order = parents_first_order(&nodes).expect("order");
        let position = |id: &str| order.iter().position(|&i| nodes[i].0 == id).unwrap();
        assert!(position("a") < position("b"));
        assert!(position("b") < position("c"));
    }

    #[test]
    fn test_parents_first_order_treats_external_parent_as_root() {
        // A parent absent from the set (e.g. the exported root was itself a sub-agent) is not an
        // error; the node is ordered as a root.
        let nodes = vec![("only".to_string(), Some("outside".to_string()))];
        assert_eq!(parents_first_order(&nodes).expect("order"), vec![0]);
    }

    #[test]
    fn test_parents_first_order_rejects_cycle() {
        let nodes = vec![
            ("a".to_string(), Some("b".to_string())),
            ("b".to_string(), Some("a".to_string())),
        ];
        assert!(parents_first_order(&nodes).is_err());
    }

    #[test]
    fn test_plan_import_rejects_unknown_format_version() {
        let export = SessionExport {
            format_version: SESSION_EXPORT_FORMAT_VERSION + 1,
            meka_version: "test".into(),
            exported_at: "now".into(),
            root_session_id: "r".into(),
            sessions: Vec::new(),
        };
        assert!(plan_import(export).is_err());
    }

    #[tokio::test]
    async fn test_session_export_import_round_trip() {
        use std::path::Path;

        use crate::{
            conversation::Event,
            provider::{ContentBlock, ImageSource, Message, Role, ToolResultContent},
        };

        let manager = SessionManager::open(Some(Path::new(":memory:")))
            .await
            .expect("open");

        // Root session with a representative mix of events: plain text, an input image, a
        // tool_use/tool_result pair, and a compaction boundary.
        let root = manager.create_session(None).await.expect("root");
        let image = ImageSource {
            source_type: "base64".to_string(),
            media_type: "image/png".to_string(),
            data: "aGk=".to_string(),
        };
        let root_events = vec![
            Event::Append(Message::user("hello")),
            Event::Append(Message::user_with_images("look", vec![image])),
            Event::Append(Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "u1".to_string(),
                    name: "read".to_string(),
                    input: serde_json::json!({"path": "/x"}),
                }],
            }),
            Event::Append(Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "u1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "ok".to_string(),
                    }],
                    is_error: false,
                }],
            }),
            Event::CompactBoundary {
                summary: Message::user("[summary]"),
                replaced_count: 2,
                loaded_tools_snapshot: Default::default(),
            },
        ];
        for event in &root_events {
            manager
                .save_event(root, event)
                .await
                .expect("save root event");
        }
        manager
            .save_tool_output(root, "tool_1_output", "big output")
            .await
            .expect("tool output");
        let stats = crate::stats::SessionStatsSnapshot {
            turns: 3,
            input_tokens: 1000,
            ..Default::default()
        };
        manager
            .save_session_stats(root, &stats)
            .await
            .expect("stats");

        // A sub-agent child of the root.
        let child = manager
            .create_child_session(root, None)
            .await
            .expect("child");
        for event in [
            Event::Append(Message::user("sub task")),
            Event::Append(Message::assistant_text("sub done")),
        ] {
            manager.save_event(child, &event).await.expect("save child");
        }

        // Export -> JSON -> back.
        let export = build_session_export(&manager, root).await.expect("export");
        assert_eq!(export.sessions.len(), 2, "root + child");
        assert_eq!(export.sessions[0].id, root.to_string(), "root first");
        let json = serde_json::to_string_pretty(&export).expect("serialize");
        assert!(
            !json.contains("token_id"),
            "the fingerprint must not be exported"
        );
        let reparsed: SessionExport = serde_json::from_str(&json).expect("deserialize");

        // Import under fresh IDs.
        let (records, root_new_id) = plan_import(reparsed).expect("plan");
        assert_ne!(root_new_id, root, "import mints a new id");
        manager.import_sessions(records).await.expect("import");

        // The tree came back: root + child, with the child's parent rewired to the new root.
        let tree = manager.load_session_tree(root_new_id).await.expect("tree");
        assert_eq!(tree.len(), 2);
        let child_new = tree
            .iter()
            .find(|meta| meta.id != root_new_id)
            .expect("child present");
        assert_eq!(child_new.parent_id, Some(root_new_id));

        // The event log round-trips byte-for-byte against the untouched original.
        let imported = manager
            .load_events(root_new_id)
            .await
            .expect("load imported");
        let original = manager.load_events(root).await.expect("load original");
        assert_eq!(
            serde_json::to_string(&imported).unwrap(),
            serde_json::to_string(&original).unwrap(),
        );
        assert!(
            imported.iter().any(|event| match event {
                Event::Append(message) => message
                    .content
                    .iter()
                    .any(|block| matches!(block, ContentBlock::Image { .. })),
                _ => false,
            }),
            "the input image must survive the round trip",
        );

        // Child events, stats, and tool_outputs are preserved.
        assert_eq!(
            manager
                .load_events(child_new.id)
                .await
                .expect("load child events")
                .len(),
            2,
        );
        let imported_stats = manager
            .load_session_stats(root_new_id)
            .await
            .expect("load stats");
        assert_eq!(imported_stats.turns, 3);
        assert_eq!(imported_stats.input_tokens, 1000);
        assert_eq!(
            manager
                .load_all_tool_outputs(root_new_id)
                .await
                .expect("load outputs"),
            vec![("tool_1_output".to_string(), "big output".to_string())],
        );
    }

    #[test]
    fn test_auth_status_from_credential() {
        let future = AuthCredential::OAuthToken {
            access_token: "t".into(),
            refresh_token: None,
            // 1 hour out, in epoch millis.
            expires_at: Some((chrono::Utc::now().timestamp() + 3600) * 1000),
            account_id: None,
        };
        let status = AuthStatus::from_credential(&future);
        assert!(status.valid);
        assert!(status.expires_in_seconds.unwrap() > 3000);

        let expired = AuthCredential::OAuthToken {
            access_token: "t".into(),
            refresh_token: None,
            expires_at: Some((chrono::Utc::now().timestamp() - 60) * 1000),
            account_id: None,
        };
        assert!(!AuthStatus::from_credential(&expired).valid);

        // API keys never expire.
        let api = AuthCredential::ApiKey("k".into());
        let status = AuthStatus::from_credential(&api);
        assert!(status.valid);
        assert_eq!(status.expires_at, None);
    }

    fn user_msg(text: &str) -> provider::Message {
        provider::Message::user(text)
    }

    fn assistant_text(text: &str) -> provider::Message {
        provider::Message::assistant_text(text)
    }

    fn assistant_tool_use(id: &str, name: &str) -> provider::Message {
        provider::Message {
            role: provider::Role::Assistant,
            content: vec![provider::ContentBlock::ToolUse {
                id: id.to_string(),
                name: name.to_string(),
                input: serde_json::json!({}),
            }],
        }
    }

    fn tool_result(tool_use_id: &str) -> provider::Message {
        provider::Message {
            role: provider::Role::User,
            content: vec![provider::ContentBlock::ToolResult {
                tool_use_id: tool_use_id.to_string(),
                content: vec![provider::ToolResultContent::Text {
                    text: "ok".to_string(),
                }],
                is_error: false,
            }],
        }
    }

    fn build_log(messages: Vec<provider::Message>) -> conversation::Conversation {
        conversation::Conversation::from_vec(messages)
    }

    #[test]
    fn test_validate_valid_chain() {
        let mut log = build_log(vec![
            user_msg("hello"),
            assistant_tool_use("c1", "read_file"),
            tool_result("c1"),
            assistant_text("done"),
        ]);
        let dropped = log.sanitize_orphans();
        assert!(dropped.is_empty());
        assert_eq!(log.len(), 4);
    }

    #[test]
    fn test_validate_orphaned_tool_use_dropped() {
        let mut log = build_log(vec![
            user_msg("hello"),
            assistant_tool_use("c1", "read_file"),
            // Missing tool_result for c1
            assistant_text("done"),
        ]);
        let dropped = log.sanitize_orphans();
        assert_eq!(dropped.len(), 1);
        assert_eq!(log.len(), 2);
        let view = log.as_slice();
        assert_eq!(view[0].role, provider::Role::User);
        assert_eq!(view[1].role, provider::Role::Assistant);
        assert_eq!(view[1].text_content(), "done");
    }

    #[test]
    fn test_validate_orphaned_at_end() {
        let mut log = build_log(vec![
            user_msg("hello"),
            assistant_tool_use("c1", "read_file"),
        ]);
        log.sanitize_orphans();
        assert_eq!(log.len(), 1);
        assert_eq!(log.as_slice()[0].text_content(), "hello");
    }

    #[test]
    fn test_validate_mismatched_ids() {
        let mut log = build_log(vec![
            user_msg("hello"),
            assistant_tool_use("c1", "read_file"),
            tool_result("c2"), // Wrong ID
        ]);
        log.sanitize_orphans();
        // The assistant message is dropped because c1 has no matching result.
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_validate_text_only_preserved() {
        let mut log = build_log(vec![
            user_msg("hello"),
            assistant_text("hi"),
            user_msg("bye"),
        ]);
        log.sanitize_orphans();
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn test_validate_multiple_chains() {
        let mut log = build_log(vec![
            user_msg("start"),
            assistant_tool_use("c1", "read_file"),
            tool_result("c1"),
            assistant_tool_use("c2", "write_file"),
            // Missing tool_result for c2
            assistant_text("done"),
        ]);
        log.sanitize_orphans();
        // c2 should be dropped, rest preserved.
        assert_eq!(log.len(), 4);
        assert_eq!(log.as_slice()[3].text_content(), "done");
    }

    // -- log filter --

    /// The default filter (no `RUST_LOG`) floors rmcp's SSE-reconnect module at `error`. Guards
    /// against a future refactor silently dropping the directive and letting the noisy warning back
    /// in.
    #[test]
    fn default_log_filter_downgrades_rmcp_sse_warns() {
        let rendered = format!("{}", build_log_filter(None, "warn"));
        assert!(
            rendered.contains("rmcp::transport::common::client_side_sse=error"),
            "expected SSE-reconnect target to be floored at `error` in the default \
             filter, got: {}",
            rendered
        );
    }

    /// When the user sets `RUST_LOG`, we honour it verbatim (no hidden directive overlay), so
    /// debugging rmcp internals with e.g. `RUST_LOG=rmcp=debug` works as expected.
    #[test]
    fn explicit_rust_log_is_not_overridden() {
        let rendered = format!("{}", build_log_filter(Some("rmcp=debug"), "warn"));
        assert!(
            !rendered.contains("rmcp::transport::common::client_side_sse=error"),
            "explicit RUST_LOG must not be augmented; got: {}",
            rendered
        );
        assert!(
            rendered.contains("rmcp=debug"),
            "user's RUST_LOG should pass through unchanged; got: {}",
            rendered
        );
    }

    #[test]
    fn full_export_includes_pre_compaction_turns() {
        // A compacted session: the early turns are hidden from the model behind a CompactBoundary,
        // but `meka session export` must still render them. Build the same event log compaction
        // produces and assert the export contains both the summarized turns and a boundary marker.
        let mut log = conversation::Conversation::new();
        log.append(user_msg("first question"));
        log.append(assistant_text("first answer"));
        log.append(user_msg("second question"));
        log.append(assistant_text("second answer"));
        log.replace_for_compaction(
            user_msg("[Conversation summary from session compaction]\n\nYou discussed things."),
            vec![assistant_text("kept tail answer")],
            std::collections::HashSet::new(),
        );

        let markdown = format_session_as_markdown(
            uuid::Uuid::nil(),
            log.events(),
            &std::collections::HashMap::new(),
        );

        // Pre-compaction turns survive in the export even though the model no longer sees them.
        assert!(
            markdown.contains("first question") && markdown.contains("second answer"),
            "full export must include pre-compaction turns:\n{markdown}"
        );
        // The boundary is marked, and its summary is available (collapsed).
        assert!(
            markdown.contains("Session compaction") && markdown.contains("You discussed things."),
            "full export must mark the compaction boundary:\n{markdown}"
        );
        // The retained tail (re-appended after the boundary) is present.
        assert!(
            markdown.contains("kept tail answer"),
            "full export must include the retained tail:\n{markdown}"
        );
    }

    #[test]
    fn export_without_compaction_renders_plain_turns() {
        let mut log = conversation::Conversation::new();
        log.append(user_msg("hello"));
        log.append(assistant_text("hi there"));
        let markdown = format_session_as_markdown(
            uuid::Uuid::nil(),
            log.events(),
            &std::collections::HashMap::new(),
        );
        assert!(markdown.contains("## User") && markdown.contains("hello"));
        assert!(markdown.contains("## Assistant") && markdown.contains("hi there"));
        // No compaction happened, so no boundary marker.
        assert!(!markdown.contains("Session compaction"));
    }
}
