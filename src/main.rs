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
mod background;
mod cli;
mod config;
mod console;
mod context;
mod conversation;
mod error;
mod frontend;
mod history;
mod image;
mod instructions;
mod mcp;
mod memory;
mod permission;
mod provider;
mod relay;
mod render;
mod repl;
mod sandbox;
mod schedule;
mod server;
mod session;
mod skills;
mod stats;
mod store;
mod tokens;
mod tools;
mod workspace;

use std::sync::Arc;

use clap::Parser;
use tokio_util::sync::CancellationToken;

use crate::{
    agent::{Agent, AgentOptions},
    config::ResolvedConfig,
    permission::SharedPermission,
    repl::ReplEvent,
    session::{SessionManager, TokenStore},
    tools::ToolRegistry,
};

/// A failure whose message has already been printed in meka's own format.
///
/// Returning the error itself would print it twice, since `main`'s `anyhow::Result` prints whatever
/// it is given; returning `Ok(())` is what the interactive host used to do, which told every
/// supervisor and wrapper script that a session it had refused to open was a successful run. This
/// carries the exit status and nothing else, so the host keeps its own rendering (colour, and the
/// provider hint underneath) and still fails.
#[derive(Debug)]
struct AlreadyReported;

impl std::fmt::Display for AlreadyReported {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("already reported")
    }
}

impl std::error::Error for AlreadyReported {}

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

    // Multi-threaded, and one caller depends on that rather than merely preferring it:
    // `GateToolset::resolve` answers a scheduled gate's authority question synchronously and blocks
    // its worker on an MCP snapshot read. On a current-thread runtime that would park the only
    // thread the release has to run on. Anything that narrows this needs to make that resolver
    // async first.
    let runtime = tokio::runtime::Runtime::new()?;
    let result = run_on_runtime(&runtime, cli);
    // Detach any lingering blocking threads instead of joining them on drop. `tokio::io::stdin()`
    // (used by the OAuth paste fallback) spawns a blocking worker that sits on a `read()` syscall
    // until stdin has bytes or EOF; when the user Ctrl-Cs during the wait, the future is dropped
    // but that worker can't be cancelled from the outside. Without this the default `Runtime::drop`
    // joins that thread and hangs the process after a clean rollback.
    runtime.shutdown_background();

    // Ahead of the interrupt arm below, which exits without unwinding. Placed here rather than
    // duplicated into that arm because this is the funnel: every ordinary end of the process, clean
    // or interrupted, passes this line.
    crate::sandbox::release_process_grants();

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

    // Same shape, one line further along: the host has printed this one already, so all that is
    // left of it is the status. Both arms sit below `release_process_grants` for that reason.
    if let Err(error) = &result
        && error.downcast_ref::<AlreadyReported>().is_some()
    {
        std::process::exit(1);
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
            // Read off disk rather than from a `ResolvedConfig` this path deliberately does not
            // build. Opening the store is what migrates it, and a store carried forward by
            // `meka provider list` must record the same profile one carried forward by `meka`
            // would.
            //
            // Two values, and the difference matters. The ledger takes the one `default_provider`
            // picks, ignoring `--provider`: it stamps a profile onto every session that predates
            // meka recording one, once and irreversibly, and that must not turn on a flag the
            // first invocation after an upgrade happened to carry. `meka session import` takes the
            // flag-aware one, where choosing per run is exactly what `--provider` is for.
            // An unreadable `config.toml` is carried to the ledger as itself rather than collapsing
            // into "nothing resolved", because the two must not produce the same write: the adopt
            // step runs once and irreversibly, so a parse error used to strand every existing
            // session against no profile with nothing said. It is not turned into a hard error
            // here, though, because `meka mcp remove` and `meka provider remove` edit the raw
            // document through `toml_edit` and are how a user *repairs* such a file; refusing every
            // subcommand would close the only door out. The migration refuses instead, and only
            // when it actually has rows to stamp.
            let (default_profile, context) = match config::default_profile_on_disk(None) {
                Ok(adopted) => {
                    let flag_aware = config::default_profile_on_disk(cli_ref.provider.as_deref())?;
                    (
                        flag_aware,
                        session::migrations::Context::adopting(adopted.as_deref()),
                    )
                }
                Err(error) => {
                    tracing::warn!(
                        "config.toml could not be read, so this run cannot say which profile \
                         anything should adopt: {}",
                        error
                    );
                    (None, session::migrations::Context::on_unreadable_config())
                }
            };
            let session_manager = SessionManager::open(None, &context).await?;
            match command {
                cli::Command::Provider { action } => {
                    provider::cli::run(action, &session_manager).await
                }
                cli::Command::Session { action } => {
                    crate::session::cli::run_session_subcommand(
                        &session_manager,
                        action,
                        default_profile.as_deref(),
                    )
                    .await
                }
                cli::Command::History { action } => {
                    run_history_subcommand(&session_manager, action).await
                }
                cli::Command::Mcp { action } => {
                    run_mcp_subcommand(&session_manager, action, cli_ref).await
                }
                cli::Command::Tools { action } => run_tools_subcommand(action, cli_ref).await,
                cli::Command::Skill { action } => run_skill_subcommand(action, cli_ref).await,
                cli::Command::Memory { action } => {
                    run_memory_subcommand(&session_manager, action).await
                }
                cli::Command::Instructions { action } => run_instructions_subcommand(action),
                cli::Command::Schedule { action } => {
                    crate::schedule::cli::run(&session_manager, action, cli_ref).await
                }
                cli::Command::Account { action } => {
                    provider::cli::run_account_subcommand(&session_manager, action).await
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

    // `-c` used to take an optional session id, so `meka -c <uuid>` was the documented way to
    // resume one. It is now a boolean, which would silently read that id as the prompt and continue
    // the *most recent* session instead of the named one. Catch the old spelling rather than
    // quietly doing the wrong thing.
    if cli.continue_last
        && let Some(prompt) = cli.prompt.as_deref()
        && looks_like_session_id(prompt)
    {
        return Err(anyhow::anyhow!(
            "`-c` no longer takes a session id; use `-r {prompt}` to resume that session, \
             or `-c` alone to continue the most recent one"
        ));
    }

    // Refused rather than ignored. Both name *this run's session*, and a long-lived host has no
    // such thing: it creates a session per `session/new` or `POST /v1/sessions`. Accepting them
    // silently was worse than it sounds, because `-c` / `-r` set `session_resume`, which switches
    // off the default-profile check a host with no default needs most.
    //
    // `--provider` is deliberately not in this list: it selects which configured profile the host
    // defaults to, which is a property of the host rather than of one session.
    if acp_mode || serve_mode {
        let host = if acp_mode { "acp" } else { "serve" };
        let offending = [
            (cli.continue_last, "--continue"),
            (cli.resume.is_some(), "--resume"),
        ]
        .into_iter()
        .filter_map(|(given, flag)| given.then_some(flag))
        .collect::<Vec<_>>();
        if !offending.is_empty() {
            anyhow::bail!(
                "`meka {}` does not take {}: they name one run's session, and this host creates \
                 one per request. Name a `provider` per session instead.",
                host,
                offending.join(", ")
            );
        }
    }

    let mut config = ResolvedConfig::from_cli(&cli);

    // If --skill is set, validate and render the body upfront so an invalid name fails fast
    // before any session/MCP setup. The combined string (extra + body, mirroring the REPL's `/skill
    // <name> [extra...]`) then takes the place of cli.prompt as the first-turn input. Resolved
    // config comes first because `[skills] extra_paths` decides which roots the lookup sees.
    let skill_prompt = runtime.block_on(build_skill_prompt(&cli, &config.skill_roots()))?;

    if let Some(prompt) = skill_prompt {
        config.prompt = Some(prompt);
    }
    // `--bind` on `meka serve` overrides the config-file `[serve].bind`. Apply here so
    // `async_main` sees a single resolved binding without re-parsing the CLI.
    if let Some(cli::Command::Serve { bind: Some(bind) }) = cli.command.as_ref() {
        config.serve_bind_override = Some(bind.clone());
    }
    // Before anything renders. The renderers read this rather than taking it as a parameter because
    // the approval prompt sits several call sites below `run_repl`, which takes flat scalars; the
    // functions that compose a line still take an explicit width, so tests never touch it.
    render::set_max_width(config.max_width);
    runtime.block_on(async_main(config, acp_mode, serve_mode))
}

/// Render a `--skill <name>` invocation into the user-message string that drives the first turn.
/// Returns `Ok(None)` when `--skill` is not set so callers can leave `cli.prompt` untouched.
///
/// Mirrors the REPL's `SlashCommand::SkillInvoke` handler in what it composes: the same
/// `format!("{extra}\n\n{body}")` order when the positional `[PROMPT]` is supplied.
///
/// It does *not* mirror the lookup, and the difference is deliberate. This runs before the agent
/// exists, so it walks the roots itself; the REPL reads `agent.skills().current()`, which is the
/// live cache. Both resolve the same name to the same file, but only the REPL's sees a skill added
/// mid-session.
async fn build_skill_prompt(
    cli: &cli::Cli,
    roots: &[std::path::PathBuf],
) -> anyhow::Result<Option<String>> {
    let Some(name) = cli.skill.as_deref() else {
        return Ok(None);
    };
    let skill = skills::cli::require_skill(name, roots)?;
    let body = skills::load_skill_body(&skill)
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
/// add directives that quiet two rmcp log sites which fire on every retry:
///
/// 1. MCP servers behind a CDN / edge (Cloudflare, Fastly, …) close idle HTTP streams after ~100 s,
///    which trips `rmcp::transport::common::client_side_sse`'s `warn!("sse stream error: …")`
///    before rmcp transparently reconnects via `Last-Event-ID`. The warn fires on every expected
///    reconnect; the real failure mode (`"max retry times reached"`) is emitted at `error!` from
///    the same module, so an `=error` floor keeps the useful signal and drops the noise.
/// 2. `rmcp::transport::worker` emits `error!("worker quit with fatal: …")` each time a transport
///    fails to come up. A configured-but-unreachable server is retried in the background for the
///    life of the process, so that lands on the user's prompt every few minutes, at `error` level,
///    saying nothing meka hasn't already reported once itself through `record_connect_failure`.
///    Silenced outright rather than floored, because the noise *is* the error level.
///
/// Verified against rmcp 2.1. `RUST_LOG` short-circuits both, so nothing is permanently hidden.
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
    let sse = "rmcp::transport::common::client_side_sse=error"
        .parse()
        .expect("valid tracing directive");
    #[allow(clippy::expect_used)]
    let worker = "rmcp::transport::worker=off"
        .parse()
        .expect("valid tracing directive");
    EnvFilter::new(log_level)
        .add_directive(sse)
        .add_directive(worker)
}

async fn async_main(
    mut config: ResolvedConfig,
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

    let session_manager = SessionManager::open(
        None,
        &session::migrations::Context::adopting(config.configured_default_profile.as_deref()),
    )
    .await?;
    let token_store = session_manager.token_store();

    // Opt-in only, and never by size. Conversation history is not reproducible, and a byte budget
    // is unpredictable in a way a time window is not: which sessions it takes depends on the total
    // corpus, so one long conversation today can silently destroy an unrelated one from months
    // ago. `warn!` rather than `info!` because a deletion the user configured is still a deletion
    // they should see at the default log level.
    if let Some(retention_days) = config.retention_days {
        let sweep = session_manager
            .delete_expired_sessions(retention_days)
            .await?;
        if sweep.deleted > 0 {
            tracing::warn!(
                "deleted {} session(s) not updated in {} days ([session].retention_days)",
                sweep.deleted,
                retention_days
            );
        }
        // Only turns bump `updated_at`, so a REPL idle past the window looks expired while a human
        // is sitting in front of it. Saying nothing here would leave an operator wondering why
        // their retention setting never takes: the answer is that it did, and spared the one
        // session that was in use.
        if sweep.attached_elsewhere > 0 {
            tracing::info!(
                "spared {} session(s) another meka process has open",
                sweep.attached_elsewhere
            );
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

    // Attached before the host branch so all three hosts share one dispatcher, for the same reason
    // the MCP manager above is process-wide: a gate's tool probe is answered by the process that
    // picks the job up, not by the session that wrote it.
    config.schedule.gate_tools = Some(Arc::new(crate::tools::GateToolset::new(
        mcp_manager.clone(),
        &config,
        crate::tools::BuiltinToolFilter::from_config(
            config.builtin_allowed_tools.clone(),
            config.builtin_disabled_tools.clone(),
            config.builtin_tool_permissions.clone(),
        ),
    )));

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
    /// The profile a session created here takes when it names none.
    ///
    /// A `String` and not the `Option` on [`ResolvedConfig`], because `build_shared_deps` runs
    /// `validate()` first and a host that reaches this point has one. Resolving it once, where
    /// that guarantee is established, is what lets `session/new` and `POST /v1/sessions` take
    /// it without a branch for a state they cannot observe; both used to carry one, and
    /// neither could be tested or reached.
    pub default_profile: String,
    pub session_manager: SessionManager,
    /// Providers by profile, built on demand.
    ///
    /// A registry rather than one `Arc<dyn Provider>` because a session records the profile it
    /// runs with, and one `meka serve` may host sessions naming different ones.
    pub providers: Arc<provider::ProviderRegistry>,
    pub mcp_manager: Option<Arc<mcp::McpClientManager>>,
    pub mcp_context: Arc<mcp::McpClientContext>,
    pub skills: Arc<skills::SkillCache>,
    pub memories: Arc<memory::MemoryStore>,
    pub builtin_filter: crate::tools::BuiltinToolFilter,
    pub sandbox_capability: crate::sandbox::SandboxCapability,
    pub sandboxed_shell: bool,
    pub agent_options: AgentOptions,
    pub session_stats: Arc<stats::SessionStats>,
}

/// Warn once about `[tools]` and `[subagents]` entries that match nothing.
///
/// Called from both agent-assembly entry points. `meka acp` and `meka serve` build their agents
/// through `build_shared_deps` and so used to emit no warning at all: a typo in either block denied
/// nothing, silently, at every verbosity.
fn warn_on_stale_tool_config(
    config: &ResolvedConfig,
    builtin_filter: &crate::tools::BuiltinToolFilter,
) {
    crate::tools::warn_on_stale_builtin_tool_config(builtin_filter);
    crate::tools::warn_on_stale_subagent_config(
        &crate::tools::ToolDenials::new(
            config.subagents.disabled_servers.clone(),
            config.subagents.disabled_tools.clone(),
        ),
        &config
            .mcp_servers
            .iter()
            .map(|server| server.name.clone())
            .collect::<Vec<_>>(),
    );
}

/// The provider profile a session runs on.
///
/// One door, so every place that runs a turn answers the question the same way. A session that
/// exists names its profile on its row and that is what it gets; anything else would move the
/// conversation to a provider it was not having, drop the reasoning it recorded (a thinking block
/// is not replayed across providers) and bill a different account.
///
/// `None` is a session that does not exist yet, which takes the configured default and records it
/// the moment its row is written.
///
/// Takes the value it decides between rather than the whole [`ResolvedConfig`], so the decision can
/// be exercised on its own. Which of a recorded profile and the process default wins is the entire
/// question here, and a door that could only be reached through a fully resolved config could not
/// be asked it directly.
pub async fn resolve_session_provider(
    session_manager: &SessionManager,
    // The process default, or the reason there is not one. A reason rather than a bare absence
    // because it is the only useful thing to say when this falls through: "no profile could be
    // picked" is not actionable, while "multiple profiles configured (work, side); run
    // `meka provider use <name>`" is. `validate()` no longer raises it for a resume, so this is
    // where it surfaces.
    default_profile: std::result::Result<&str, &str>,
    session_id: Option<uuid::Uuid>,
) -> anyhow::Result<String> {
    if let Some(session_id) = session_id
        && let Some(recorded) = session_manager.recorded_provider(session_id).await?
    {
        return Ok(recorded);
    }
    Ok(default_profile
        .map_err(|reason| anyhow::anyhow!("{}", reason))?
        .to_string())
}

/// [`resolve_session_provider`] for a caller that has a whole [`ResolvedConfig`] to hand.
pub async fn provider_for_config(
    session_manager: &SessionManager,
    config: &ResolvedConfig,
    session_id: Option<uuid::Uuid>,
) -> anyhow::Result<String> {
    resolve_session_provider(
        session_manager,
        // Exactly one of the two is set; see `select_active_profile`. The fallback text is for a
        // shape that pairing rules out rather than for a case anyone expects to hit.
        config.active_profile.as_deref().ok_or_else(|| {
            config
                .provider_error
                .as_deref()
                .unwrap_or("no provider profile is configured; run `meka provider add`")
        }),
        session_id,
    )
    .await
}

/// The profile a session records, when `config.toml` no longer has it.
///
/// The one failure `--provider` is the fix for, and the only one worth naming a session in a hint
/// about: a profile that is configured but unusable (no stored credential, an endpoint that
/// refuses) is not moved by repinning the row.
///
/// The recorded name is compared against the configured set and nothing else, with no test for the
/// empty one a migrated store can hold. `""` is a name that resolves to nothing, which is exactly
/// what this asks, so it answers correctly without this function having to know where it came
/// from.
///
/// The name itself is not returned, because nothing needs it: the refusal already printed names the
/// profile, and the hint this gates adds only the repin command.
///
/// A read failure answers `false`: this runs only to decorate an error that has already been
/// printed, and failing the process over the decoration would replace a useful message with a
/// useless one.
async fn recorded_profile_is_gone(
    session_manager: &SessionManager,
    config: &ResolvedConfig,
    session_id: uuid::Uuid,
) -> bool {
    match session_manager.recorded_provider(session_id).await {
        Ok(Some(binding)) => !config.providers.contains_key(&binding),
        Ok(None) => false,
        Err(error) => {
            tracing::debug!(
                "could not read session {}'s recorded profile for the setup hint: {}",
                session_id,
                error
            );
            false
        }
    }
}

/// Turn a session's binding into the provider it names and the per-profile facts that come with it.
///
/// The one producer of [`agent::ResolvedBinding`], so building a session and moving one
/// mid-conversation cannot disagree about what a profile means. Before this existed, the window and
/// the vision flag were read once per process from the *default* profile: a session pinned to a
/// 32k profile gauged itself against the default's window, so auto-compaction never fired and the
/// provider rejected the turn instead.
pub async fn resolved_binding(
    providers: &provider::ProviderRegistry,
    binding: String,
) -> anyhow::Result<agent::ResolvedBinding> {
    let (provider, settings) = providers.build(&binding).await?;
    Ok(agent::ResolvedBinding {
        provider,
        // The documented default, not a guess at the model: meka does not infer a window from a
        // model name, so a profile that states none gets the one value the docs name.
        context_window: settings
            .context_window
            .unwrap_or(crate::provider::DEFAULT_CONTEXT_WINDOW),
        vision: settings.vision,
        binding,
    })
}

/// Whether a session's profile accepts image input, answered without building its provider.
///
/// For the hosts that must decide whether to admit an attachment before a turn exists. A profile
/// that cannot resolve answers `false`: that session's next turn is going to fail on the same
/// profile, and taking the attachment first would only add a second failure further in.
///
/// Reads the same `ProfileSettings` [`resolved_binding`] does, so the answer a host caches cannot
/// drift from the one the agent was built with.
pub fn binding_accepts_images(providers: &provider::ProviderRegistry, binding: &str) -> bool {
    providers
        .settings(binding)
        .map(|settings| settings.vision)
        .unwrap_or(false)
}

/// A session's context window, answered without building its provider.
///
/// The sibling of [`binding_accepts_images`], for a host that reports occupancy without reaching
/// through the runtime mutex an in-flight turn is holding. Same source, so the reported window is
/// the one the agent gauges against.
/// `None` for a binding that cannot resolve, which is not the same as the documented default: that
/// session's next turn is going to be refused by name, and answering `1000000` beside a refusal
/// invites a client to divide by a number meka has no reason to believe.
pub fn binding_context_window(
    providers: &provider::ProviderRegistry,
    binding: &str,
) -> Option<u64> {
    providers.settings(binding).ok().map(|settings| {
        settings
            .context_window
            .unwrap_or(crate::provider::DEFAULT_CONTEXT_WINDOW)
    })
}

/// The provider registry for the two CLI hosts (the REPL and `--oneshot`), built the way
/// [`build_shared_deps`] builds ACP's and `serve`'s so all four resolve a profile identically.
fn cli_provider_registry(
    config: &ResolvedConfig,
    token_store: TokenStore,
    session_stats: Arc<stats::SessionStats>,
) -> anyhow::Result<Arc<provider::ProviderRegistry>> {
    let providers = Arc::new(provider::ProviderRegistry::new(
        config,
        token_store,
        session_stats,
    ));

    // Debug-only, and the same install `run_acp` and `run_serve` make. It reaches the REPL and
    // `--oneshot` because the questions two `meka` processes raise about each other -- who holds a
    // session's lock while a first turn runs, whose background task the other sweeps -- are
    // questions about the CLI entry point, and no harness could ask them while the only scriptable
    // surfaces were ACP and HTTP.
    #[cfg(debug_assertions)]
    if std::env::var("MEKA_MOCK_PROVIDER").as_deref() == Ok("1") {
        let rounds = crate::provider::mock::load_script_from_env()?.unwrap_or_default();
        tracing::info!("MEKA_MOCK_PROVIDER=1: using scripted mock provider");
        providers.install_scripted(Arc::new(crate::provider::mock::MockProvider::from_rounds(
            rounds,
        )));
    }

    Ok(providers)
}

/// [`resolve_session_provider`] for the hosts that carry a [`SharedDeps`].
pub async fn provider_for_session(
    shared: &SharedDeps,
    session_id: Option<uuid::Uuid>,
) -> anyhow::Result<String> {
    provider_for_config(&shared.session_manager, &shared.config, session_id).await
}

/// Build the process-wide [`SharedDeps`] for `meka acp`. Sets up the provider, MCP wiring, skill
/// cache, sandbox capability probe, and the shared `agent_options` template. Each ACP session later
/// calls [`build_session_agent`] against the resulting struct to spin up its own per-session
/// `Agent` + `ToolRegistry`.
pub async fn build_shared_deps(
    config: ResolvedConfig,
    session_manager: SessionManager,
    mcp_manager: Option<Arc<mcp::McpClientManager>>,
    mcp_context: Arc<mcp::McpClientContext>,
) -> anyhow::Result<SharedDeps> {
    config.validate()?;
    // The one place the "a long-lived host has a default profile" guarantee becomes a type.
    // `validate()` has just enforced it: `validate_default_profile` is skipped only for a resume,
    // and `-c` / `-r` are refused for both hosts precisely so that exception cannot apply here.
    let default_profile = config.active_profile.clone().ok_or_else(|| {
        anyhow::anyhow!(config.provider_error.clone().unwrap_or_else(|| {
            "no provider profile is configured; run `meka provider add`".to_string()
        }))
    })?;

    let session_stats = Arc::new(stats::SessionStats::default());
    // Nothing is built here. The registry resolves a profile and loads its credential when a
    // session first asks, because which profiles this process will need is a property of the
    // sessions it ends up serving rather than of its configuration.
    let providers = Arc::new(provider::ProviderRegistry::new(
        &config,
        session_manager.token_store(),
        Arc::clone(&session_stats),
    ));

    let sandbox_capability: crate::sandbox::SandboxCapability = match &config.backend_probe {
        crate::sandbox::BackendProbe::Ok(capability) => capability.clone(),
        _ => crate::sandbox::SandboxCapability::Unavailable,
    };
    let sandboxed_shell = config.sandbox
        && !matches!(
            sandbox_capability,
            crate::sandbox::SandboxCapability::Unavailable
        );

    // Both stores are instance-scoped, so one cache each serves every session this process runs.
    // `disabled()` is distinct from an empty store: it keeps the subsystem's tools out of the
    // registry entirely, which is the point of the config switch.
    let skills = if config.skills_enabled {
        crate::skills::SkillCache::discover(config.skills_extra_paths.clone())
    } else {
        crate::skills::SkillCache::disabled()
    };
    // Always connected, `enabled` carrying the config switch: it gates the agent's tools, not the
    // operator's access to a store that already exists.
    let memories = session_manager.memory_store(config.memory_enabled);
    let builtin_filter = crate::tools::BuiltinToolFilter::from_config(
        config.builtin_allowed_tools.clone(),
        config.builtin_disabled_tools.clone(),
        config.builtin_tool_permissions.clone(),
    );
    warn_on_stale_tool_config(&config, &builtin_filter);

    let agent_options = AgentOptions {
        streaming: config.streaming,
        sandboxed_shell,
        gate_tools: config.schedule.gate_tools.clone(),
        context_messages: config.context_messages,
        auto_compact: config.auto_compact,
        compact_checkpoint: config.compact_checkpoint,
        user_instructions: config.user_instructions.clone(),
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
        default_profile,
        session_manager,
        providers,
        mcp_manager,
        mcp_context,
        skills,
        memories,
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
    session_manager: SessionManager,
    /// The whole binding rather than the provider and its profile separately, because
    /// [`assemble_agent`] has to publish it: the sub-agent and `context_*` tools hold a handle so
    /// a mid-session switch reaches them.
    resolved: agent::ResolvedBinding,
    mcp_manager: Option<&'a Arc<mcp::McpClientManager>>,
    skills: Arc<skills::SkillCache>,
    /// Whether this agent gets `skill_write` / `skill_delete`, from `[skills] agent_managed`.
    /// Never reaches a sub-agent registry; see `ToolRegistry::register_session_scoped_tools`.
    skills_agent_managed: bool,
    memories: Arc<memory::MemoryStore>,
    builtin_filter: crate::tools::BuiltinToolFilter,
    agent_options: AgentOptions,
    session_stats: Arc<stats::SessionStats>,
    /// Seeds the root `AgentSpawnTool`'s recursion budget from `session.subagent_max_depth`.
    subagent_max_depth: usize,
    /// Gates the `schedule_*` tools and supplies their ceilings. Sub-agent registries get `None`
    /// instead; see `ToolRegistry::register_session_scoped_tools`.
    schedule: crate::config::ResolvedScheduleConfig,
    /// Gates the `background` parameter and the `task_*` tools, and supplies the concurrency
    /// ceiling. Off by default; see `crate::config::BackgroundConfig`.
    background: crate::config::ResolvedBackgroundConfig,
    /// Capabilities a spawned worker may never hold. Seeds the root `AgentSpawnTool`'s deny lists;
    /// see `crate::config::SubagentsConfig`. What a worker *receives* is not configured at all,
    /// and is granted per `agent_spawn` call.
    subagents: crate::config::ResolvedSubagentsConfig,
    /// The live context counter, supplied by the caller rather than made here because a frontend
    /// gauge (the REPL prompt, ACP's `usage_update`) holds the same atomic and is constructed
    /// before the agent exists. The agent writes it after every provider response; `context_check`
    /// and the frontend read it.
    ///
    /// It has to arrive here rather than be set afterwards: `assemble_agent` builds the
    /// `context_check` gauge around this handle, and `Agent::set_context_tokens` *replaces* the
    /// agent's handle without touching the gauge. A caller that constructs its own atomic and
    /// re-points the agent later leaves the tool reading a counter nobody writes, which reports a
    /// serenely empty context forever.
    context_tokens: Arc<std::sync::atomic::AtomicU64>,
    /// Fixed overhead (system prompt + tool schemas), on the bundle for the same reason
    /// `context_tokens` is: `assemble_agent` builds the `context_check` gauge around this exact
    /// handle, so a caller wanting to read it later has to supply it rather than adopt one after
    /// the fact.
    context_overhead: Arc<std::sync::atomic::AtomicU64>,
    /// The live window, supplied for the same reason `context_tokens` is: the host's own gauge --
    /// the REPL prompt indicator, ACP's `usage_update` -- exists before the agent and must be the
    /// same cell rather than a copy re-stored by hand next to every provider switch. `serve` and
    /// `--oneshot` have no such gauge and pass a throwaway; `serve` takes the handle back out of
    /// the assembled agent instead, and `--oneshot` prints one answer and exits.
    context_window: Arc<std::sync::atomic::AtomicU64>,
}

/// Per-session agent assembly used by both the ACP session builder and the REPL's
/// `create_agent_from_config`. Builds the shared todo list / scratchpad cell, the tool registry
/// (with the session's cwd / permission / frontend baked into the builtins), registers
/// `agent_spawn` and the MCP resource meta-tools, attaches the registry to the MCP manager, and
/// finally constructs the `Agent` itself.
///
/// The order this runs in relative to `start_connector` does not matter. `build_shared_deps` runs
/// the connector once for ACP and `serve`, before any session exists; the REPL runs it after this
/// returns. Either way every attached registry converges on the same tool set, because
/// [`crate::mcp::McpClientManager::update_server_tools`] writes the snapshot before fanning out and
/// [`crate::mcp::McpClientManager::attach_registry`] replays the whole of it. What a late attach
/// costs is latency, not state: a session created while a slow server is still connecting sees that
/// server's tools when it lands.
async fn assemble_agent(
    bundle: AgentAssembly<'_>,
    shared_permission: SharedPermission,
    frontend: Arc<dyn frontend::Frontend>,
    cwd: crate::workspace::SharedCwd,
    roots: crate::workspace::SharedRoots,
) -> anyhow::Result<(Agent, crate::tools::ToolRegistry)> {
    let todo_list: crate::tools::todo::SharedTodoList = std::sync::Arc::new(
        tokio::sync::RwLock::new(crate::tools::todo::TodoState::default()),
    );
    let shared_session_id: std::sync::Arc<tokio::sync::RwLock<Option<uuid::Uuid>>> =
        std::sync::Arc::new(tokio::sync::RwLock::new(None));
    // One registry per session, shared by the agent that dispatches tasks, the `task_*` tools that
    // manage them, and the REPL's `/tasks` and Ctrl+C handling.
    let background_tasks = crate::background::BackgroundTasks::default();

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
        bundle.skills_agent_managed,
        bundle.memories.clone(),
        bundle.builtin_filter.clone(),
        cwd.clone(),
        Arc::clone(&roots),
        Arc::clone(&frontend),
        bundle.schedule.clone(),
        (bundle.background.clone(), background_tasks.clone()),
    )?;

    // Published before the tools that read it are registered, and handed to the agent below, which
    // is its only writer. This is what makes `/provider` and its two siblings reach `agent_spawn`
    // and the `context_*` gauge instead of moving the agent alone. The window cell comes from the
    // caller, so the host's own gauge *is* this one rather than a copy it has to re-store.
    let published_binding =
        agent::PublishedBinding::new(&bundle.resolved, Arc::clone(&bundle.context_window));

    // `subagent_max_depth == 0` disables sub-agents entirely (root gets no `agent_spawn`); `>= 1`
    // seeds the root's soft recursion budget, and `absolute_depth` starts at 0 for the root. The
    // predicate folds in the `[tools]` half too, and is named rather than written inline because
    // `meka tools list` has to reach the same answer with no provider to assemble a session with.
    if crate::tools::subagent::agent_tools_registered(
        &bundle.builtin_filter,
        bundle.subagent_max_depth,
    ) {
        crate::tools::subagent::register_subagent_tools(
            &tool_registry,
            crate::tools::subagent::AgentSpawnTool {
                parent_permission: shared_permission.clone(),
                tool_builder_params: crate::tools::subagent::ToolBuilderParams {
                    live_binding: published_binding.clone(),
                    web_client: bundle.web_client.clone(),
                    sandbox_enabled: bundle.sandbox_enabled,
                    sandbox_capability: bundle.sandbox_capability.clone(),
                    sandbox_backend: bundle.sandbox_backend,
                    backend_probe: bundle.backend_probe.clone(),
                    builtin_filter: bundle.builtin_filter.clone(),
                    skills: bundle.skills.clone(),
                    memories: bundle.memories.clone(),
                    // The primary agent holds the whole store, so that is the ceiling on what it
                    // can grant a worker. Whether a worker gets anything is
                    // decided per `agent_spawn` call, and defaults to nothing.
                    memory_access: crate::config::MemoryAccess::Write,
                    config_denials: crate::tools::ToolDenials::new(
                        bundle.subagents.disabled_servers.clone(),
                        bundle.subagents.disabled_tools.clone(),
                    ),
                    mcp_manager: bundle.mcp_manager.map(Arc::downgrade),
                    session_manager: bundle.session_manager.clone(),
                    parent_shared_session_id: shared_session_id.clone(),
                    session_stats: Arc::clone(&bundle.session_stats),
                    parent_options: bundle.agent_options.clone(),
                    parent_cwd: Arc::clone(&cwd),
                    parent_roots: Arc::clone(&roots),
                    parent_frontend: Arc::clone(&frontend),
                },
                inherited_denials: crate::tools::ToolDenials::new(
                    bundle.subagents.disabled_servers.clone(),
                    bundle.subagents.disabled_tools.clone(),
                ),
                remaining_depth: bundle.subagent_max_depth,
                absolute_depth: 0,
            },
        )?;
    }

    // The `context_*` tools and the agent must share one set of counters, so they are made here and
    // handed to both. Registered outside `build_default` for the same reason `agent_spawn` is: what
    // they read belongs to the agent, which does not exist yet.
    let pending_compaction: crate::tools::context::PendingCompaction =
        Arc::new(std::sync::Mutex::new(None));
    tool_registry.register_context_tools(
        crate::tools::context::ContextGauge {
            used: Arc::clone(&bundle.context_tokens),
            overhead: Arc::clone(&bundle.context_overhead),
            window: published_binding.window(),
            compact_at_percent: bundle
                .agent_options
                .auto_compact
                .then_some(crate::agent::AUTO_COMPACT_THRESHOLD_PERCENT),
        },
        Arc::clone(&pending_compaction),
        bundle.agent_options.compact_checkpoint,
        bundle.session_manager.clone(),
        shared_session_id.clone(),
    );

    if let Some(manager) = bundle.mcp_manager {
        // Register MCP resource meta-tools upfront; they delegate through
        // `ServerEntry::require_connected` so they tolerate Pending / Failed servers until a
        // specific one is called.
        crate::tools::mcp_resources::register_all(&tool_registry, Arc::clone(manager));
        // Attach this session's registry so the MCP connector and tools/list_changed handler
        // propagate updates into it, and so it picks up whatever has already been discovered.
        manager.attach_registry(tool_registry.clone()).await;
    }

    let mut agent = Agent::new(
        published_binding,
        tool_registry.clone(),
        bundle.session_manager.clone(),
        shared_permission,
        bundle.agent_options.clone(),
        todo_list,
        shared_session_id,
        bundle.skills.clone(),
        bundle.memories.clone(),
        frontend,
        cwd,
        roots,
        Arc::clone(&bundle.session_stats),
    );
    agent.set_context_tokens(Arc::clone(&bundle.context_tokens));
    agent.attach_context_tools(Arc::clone(&bundle.context_overhead), pending_compaction);
    if let Some(manager) = bundle.mcp_manager {
        agent.set_mcp_manager(Arc::clone(manager));
    }
    if bundle.background.enabled {
        agent.enable_background(background_tasks, bundle.background.max_tasks);
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
// The other top-level agent-assembly entry point, and the same reasoning as
// `create_agent_from_config`: splitting these up would force every host to pre-bundle unrelated
// collaborators just to appease the arg-count lint.
#[allow(clippy::too_many_arguments)]
pub async fn build_session_agent(
    shared: &SharedDeps,
    // Which session this agent serves, or `None` for one that does not exist yet. It decides which
    // provider profile the agent runs on: see `provider_for_session`.
    session_id: Option<uuid::Uuid>,
    shared_permission: SharedPermission,
    frontend: Arc<dyn frontend::Frontend>,
    cwd: crate::workspace::SharedCwd,
    roots: crate::workspace::SharedRoots,
    context_tokens: Arc<std::sync::atomic::AtomicU64>,
    // `context_overhead` is a parameter for the same reason `context_tokens` is: a caller that
    // wants to read the gauge without holding the session's runtime mutex has to own the handle,
    // because the `Agent` that writes it lives inside that mutex. `meka serve` retains both so
    // `GET /v1/sessions/{id}/context` never blocks on a turn.
    context_overhead: Arc<std::sync::atomic::AtomicU64>,
    // The cell this session's window is published into, so a host reporting occupancy holds the
    // same one the agent gauges against rather than a copy it re-stores on every switch.
    context_window: Arc<std::sync::atomic::AtomicU64>,
) -> anyhow::Result<(Agent, crate::tools::ToolRegistry)> {
    let resolved = resolved_binding(
        &shared.providers,
        provider_for_session(shared, session_id).await?,
    )
    .await?;
    let agent_options = shared.agent_options.clone();
    let bundle = AgentAssembly {
        schedule: shared.config.schedule.clone(),
        background: shared.config.background.clone(),
        web_client: shared.config.web_client.clone(),
        sandbox_enabled: shared.config.sandbox,
        sandbox_capability: shared.sandbox_capability.clone(),
        sandbox_backend: shared.config.sandbox_backend,
        backend_probe: shared.config.backend_probe.clone(),
        session_manager: shared.session_manager.clone(),
        resolved,
        mcp_manager: shared.mcp_manager.as_ref(),
        skills: shared.skills.clone(),
        skills_agent_managed: shared.config.skills_agent_managed,
        memories: shared.memories.clone(),
        builtin_filter: shared.builtin_filter.clone(),
        agent_options,
        session_stats: Arc::clone(&shared.session_stats),
        subagent_max_depth: shared.config.subagent_max_depth,
        subagents: shared.config.subagents.clone(),
        context_tokens,
        context_overhead,
        context_window,
    };
    assemble_agent(bundle, shared_permission, frontend, cwd, roots).await
}

// Top-level entry point for assembling the agent; splitting its inputs further would force callers
// to pre-bundle unrelated collaborators (config, session manager, permission mode, credential, MCP
// plumbing, frontend) just to appease the arg-count lint.
#[allow(clippy::too_many_arguments)]
async fn create_agent_from_config(
    config: &ResolvedConfig,
    // The session this agent is for, or `None` for a fresh one. A resumed session runs on the
    // profile it recorded, which is what keeps `meka -c` on the provider the conversation was had
    // with.
    session_id: Option<uuid::Uuid>,
    session_manager: SessionManager,
    shared_permission: SharedPermission,
    providers: &Arc<provider::ProviderRegistry>,
    mcp_manager: Option<&Arc<mcp::McpClientManager>>,
    frontend: Arc<dyn frontend::Frontend>,
    cwd: crate::workspace::SharedCwd,
    session_stats: Arc<stats::SessionStats>,
    context_tokens: Arc<std::sync::atomic::AtomicU64>,
    context_window: Arc<std::sync::atomic::AtomicU64>,
) -> anyhow::Result<Agent> {
    config.validate()?;

    let resolved = resolved_binding(
        providers,
        provider_for_config(&session_manager, config, session_id).await?,
    )
    .await?;

    let sandbox_capability: crate::sandbox::SandboxCapability = match &config.backend_probe {
        crate::sandbox::BackendProbe::Ok(capability) => capability.clone(),
        _ => crate::sandbox::SandboxCapability::Unavailable,
    };
    let sandboxed_shell = config.sandbox
        && !matches!(
            sandbox_capability,
            crate::sandbox::SandboxCapability::Unavailable
        );

    // Discover both stores once at startup. Any malformed entry emits its `tracing::warn!` here
    // (tracing is already initialized), so the user sees parse errors above the first prompt rather
    // than interleaved with their first turn's output. The caches also drive mid-session auto-
    // reload; `current()` re-snapshots on each turn and re-discovers only when the on-disk state
    // changes.
    //
    // `disabled()` is distinct from an empty store: it keeps the subsystem's tools out of the
    // registry entirely, which is the point of the config switch.
    let skills = if config.skills_enabled {
        crate::skills::SkillCache::discover(config.skills_extra_paths.clone())
    } else {
        crate::skills::SkillCache::disabled()
    };
    // Always connected, `enabled` carrying the config switch: it gates the agent's tools, not the
    // operator's access to a store that already exists.
    let memories = session_manager.memory_store(config.memory_enabled);

    let builtin_filter = crate::tools::BuiltinToolFilter::from_config(
        config.builtin_allowed_tools.clone(),
        config.builtin_disabled_tools.clone(),
        config.builtin_tool_permissions.clone(),
    );

    // Build the parent's `AgentOptions` up-front so it can be cloned into `ToolBuilderParams` for
    // sub-agents to inherit `sandboxed_shell` / `context_messages` / the auto-compaction settings
    // via `Agent::new_subagent`. `user_instructions` is deliberately not among them.
    let agent_options = AgentOptions {
        streaming: config.streaming,
        sandboxed_shell,
        gate_tools: config.schedule.gate_tools.clone(),
        context_messages: config.context_messages,
        auto_compact: config.auto_compact,
        compact_checkpoint: config.compact_checkpoint,
        user_instructions: config.user_instructions.clone(),
        mcp_grace: config.mcp_grace,
        // Parent builds its system prompt dynamically per-turn via context::build_system_prompt.
        // Sub-agents override.
        system_prompt_override: None,
    };

    let bundle = AgentAssembly {
        schedule: config.schedule.clone(),
        background: config.background.clone(),
        web_client: config.web_client.clone(),
        sandbox_enabled: config.sandbox,
        sandbox_capability,
        sandbox_backend: config.sandbox_backend,
        backend_probe: config.backend_probe.clone(),
        session_manager: session_manager.clone(),
        resolved,
        mcp_manager,
        skills: skills.clone(),
        skills_agent_managed: config.skills_agent_managed,
        memories: memories.clone(),
        builtin_filter: builtin_filter.clone(),
        agent_options: agent_options.clone(),
        session_stats: Arc::clone(&session_stats),
        subagent_max_depth: config.subagent_max_depth,
        subagents: config.subagents.clone(),
        context_tokens,
        // The REPL reads occupancy through `/status`, which goes via the agent it owns outright;
        // nothing here needs a separate handle on the overhead counter.
        context_overhead: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        context_window,
    };
    let (agent, _tool_registry) = assemble_agent(
        bundle,
        shared_permission,
        frontend,
        Arc::clone(&cwd),
        // ACP is the other source of extra workspace roots; here they come from
        // `--writable-root`. Both land in the same handle because they mean the same thing, so
        // a named folder is searched and, at `workspace` permission, writable.
        Arc::new(std::sync::RwLock::new(config.writable_roots.clone())),
    )
    .await?;

    warn_on_stale_tool_config(config, &builtin_filter);

    if let Some(manager) = mcp_manager {
        // Kick off the background connector. Each server's adapters are pushed through
        // `manager.update_server_tools`, which records them and fans them out to every attached
        // registry. Idempotent on second call. (The ACP path does this once in
        // `build_shared_deps`; the REPL path does it here, after `assemble_agent` has attached the
        // single registry.)
        manager.start_connector(crate::mcp::McpRuntimeConfig::from_config(config));
    }

    Ok(agent)
}

/// Say what a cancelled turn left running.
///
/// Ctrl+C stops the turn, not the detached work, which is the shell's contract and the right
/// default. But the user may not have registered that anything was detached, so what survived has
/// to be visible rather than merely discoverable.
async fn report_background_survivors(agent: &Agent) {
    let running = agent.background_tasks().running_count_all().await;
    if running > 0 {
        eprintln!(
            "{} background task(s) still running. Press Ctrl+C again during a turn to stop them, \
             or run /tasks cancel --all.",
            running
        );
    }
}

/// Claim this session's undelivered task outcomes, ready to be rendered into one turn.
///
/// Stamped delivered *before* the turn runs, matching the scheduler's own claim and for the same
/// reason: an outcome that reliably wedges the process would otherwise be redelivered on every
/// restart, turning one bad result into a boot loop. Losing one report is the cheaper failure.
///
/// Returns empty on a database error rather than propagating, because failing to *report* a
/// finished task must not also fail whatever else the caller was about to do.
async fn collect_background_outcomes(
    session_manager: &SessionManager,
    session_id: Option<uuid::Uuid>,
) -> Vec<crate::background::BackgroundTask> {
    let Some(session_id) = session_id else {
        return Vec::new();
    };
    let ready = match session_manager
        .background_store()
        .list_undelivered_background_tasks(session_id)
        .await
    {
        Ok(ready) => ready,
        Err(error) => {
            tracing::warn!("failed to load background task outcomes: {}", error);
            return Vec::new();
        }
    };
    if ready.is_empty() {
        return ready;
    }
    let ids: Vec<String> = ready.iter().map(|task| task.id.clone()).collect();
    if let Err(error) = session_manager
        .background_store()
        .mark_background_tasks_delivered(&ids)
        .await
    {
        tracing::warn!(
            "failed to stamp background outcomes as delivered: {}",
            error
        );
        // Not delivering is better than delivering forever: without the stamp the next tick would
        // repeat these, and every tick after it.
        return Vec::new();
    }
    ready
}

/// Run a `/compact` with Ctrl+C wired to a fresh cancellation token, aborting the listener
/// afterwards.
///
/// Compaction is not a turn, but it makes provider calls and - at `ask` permission - can block on
/// an approval prompt, so it needs a signal source for exactly the reason
/// [`run_turn_interruptible`] documents: a bare token silently swallows Ctrl+C. It has no
/// background tasks to reap, so it skips that function's second press and escalates straight from
/// cancel to exit.
/// The scheduler config to evaluate this REPL sweep against, with `host_permission` replaced by the
/// level the session is running at *now*.
///
/// [`crate::config::ResolvedScheduleConfig::host_permission`] is resolved once at startup, and a
/// REPL session carries no per-session level on its row, so a gate's live re-check would otherwise
/// compare against the level the process was launched with. Shift+Tab and `/permission` move only
/// the [`SharedPermission`] cell, so that snapshot is wrong from the first cycle onward, and wrong
/// in both directions: cycling up from the default `read` to author a gate left it refused forever,
/// and cycling down from `unrestricted` left an already-written gate firing, which is exactly the
/// withdrawal the re-check exists to perform.
///
/// A function rather than two lines at the call site so the substitution is assertable; the wiring
/// is the whole fix, and there is no REPL harness that could reach it otherwise.
fn schedule_config_at_live_permission(
    configured: &crate::config::ResolvedScheduleConfig,
    live: &SharedPermission,
) -> crate::config::ResolvedScheduleConfig {
    crate::config::ResolvedScheduleConfig {
        host_permission: live.get(),
        ..configured.clone()
    }
}

/// The process's Ctrl+C handling: one long-lived listener, not one per turn.
///
/// tokio installs its SIGINT handler on first use and never removes it, so a per-turn listener that
/// is aborted when the turn ends leaves that handler in place with nothing awaiting it. Every later
/// press is then captured and dropped, and a turn whose tool ignores cancellation -- a stuck child,
/// an MCP call that never returns -- became unkillable from its own terminal. A task that never
/// stops awaiting cannot drop one.
///
/// Escalation counts per *turn*, not per process: publishing a turn's token resets the count, so
/// the second press of the fifth turn means what the second press of the first one did.
///
/// Nothing here competes with the prompt. reedline reads Ctrl+C as a key event in raw mode, where
/// the terminal generates no SIGINT at all, so this listener only ever sees a press made while a
/// turn is running -- which is the only window it is about.
struct InterruptRelay {
    /// The running turn's token, or `None` between turns.
    current: std::sync::RwLock<Option<CancellationToken>>,
    presses: std::sync::atomic::AtomicUsize,
    /// Woken on every press, for the one caller that waits outside a turn.
    ///
    /// A second `tokio::signal::ctrl_c()` elsewhere in the process would be a second *handler*:
    /// tokio delivers each press to every awaiter, so one keystroke ran the escalation ladder here
    /// and printed an unrelated message there, racing each other's output and the outcome
    /// collection between them. Waiters listen to this instead, so the ladder stays the only
    /// reader of the signal.
    pressed: tokio::sync::Notify,
}

static INTERRUPT_RELAY: std::sync::LazyLock<InterruptRelay> =
    std::sync::LazyLock::new(|| InterruptRelay {
        current: std::sync::RwLock::new(None),
        presses: std::sync::atomic::AtomicUsize::new(0),
        pressed: tokio::sync::Notify::new(),
    });

/// Grace given to background tasks on the press that leaves. Long enough for a child to die and its
/// row to be written, short enough that a user who has pressed Ctrl+C three times is not made to
/// wait: the alternative was `exit` on the spot, which orphaned the process group and left the row
/// reading `running` forever -- the exact loss the REPL's own exit drain was added to prevent.
const INTERRUPT_DRAIN_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

impl InterruptRelay {
    /// Hand the relay the token of the turn about to run, and reset the escalation count.
    fn publish(token: CancellationToken) {
        match INTERRUPT_RELAY.current.write() {
            Ok(mut slot) => *slot = Some(token),
            Err(poisoned) => *poisoned.into_inner() = Some(token),
        }
        INTERRUPT_RELAY
            .presses
            .store(0, std::sync::atomic::Ordering::SeqCst);
    }

    /// Called when the turn ends. A press after this has no turn to cancel and falls through to the
    /// escalation, which is what makes a wedged *tool* still interruptible after its turn returns.
    fn clear() {
        match INTERRUPT_RELAY.current.write() {
            Ok(mut slot) => *slot = None,
            Err(poisoned) => *poisoned.into_inner() = None,
        }
    }

    fn current() -> Option<CancellationToken> {
        match INTERRUPT_RELAY.current.read() {
            Ok(slot) => slot.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }
}

/// Start the process's single SIGINT listener. Idempotent; every turn path calls it, and only the
/// first call spawns.
fn install_interrupt_handler(agent: &Agent) {
    static INSTALLED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    if INSTALLED.set(()).is_err() {
        return;
    }

    let tasks = agent.background_tasks();
    let session_manager = agent.session_manager();
    tokio::spawn(async move {
        loop {
            if tokio::signal::ctrl_c().await.is_err() {
                return;
            }
            INTERRUPT_RELAY.pressed.notify_waiters();
            let press = INTERRUPT_RELAY
                .presses
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                + 1;

            match press {
                // The shell's contract: the first SIGINT reaches the foreground job only.
                // Background work survives, because losing a twenty-minute build to a Ctrl+C aimed
                // at the answer on screen is unrecoverable and is not what the keystroke meant.
                1 => {
                    if let Some(token) = InterruptRelay::current() {
                        token.cancel();
                    }
                }
                2 => {
                    // Recorded before signalling, so what the agent hears is "you stopped it"
                    // rather than the `failed` its own interruption would otherwise write.
                    for id in tasks.task_ids().await {
                        if let Err(error) = session_manager
                            .background_store()
                            .finish_background_task(
                                &id,
                                crate::background::TaskStatus::Cancelled,
                                None,
                                None,
                            )
                            .await
                        {
                            tracing::warn!("could not record task {} as cancelled: {}", id, error);
                        }
                    }
                    let signalled = tasks.cancel_all().await;
                    if signalled > 0 {
                        render::render_annotation(&format!(
                            "stopping {} background task{}",
                            signalled,
                            if signalled == 1 { "" } else { "s" }
                        ));
                    }
                }
                // Leave -- but let what was already cancelled finish unwinding first.
                _ => {
                    render::render_annotation("interrupted");
                    if tokio::time::timeout(INTERRUPT_DRAIN_GRACE, tasks.wait_all())
                        .await
                        .is_err()
                    {
                        tracing::warn!(
                            "background tasks did not unwind within {:?}; exiting anyway",
                            INTERRUPT_DRAIN_GRACE
                        );
                    }
                    // This arm exits from inside a spawned task, so it never reaches the funnel in
                    // `main` and has to release the grants itself.
                    crate::sandbox::release_process_grants();
                    std::process::exit(130);
                }
            }
        }
    });
}

async fn compact_interruptible(
    agent: &Agent,
    session_id: &mut Option<uuid::Uuid>,
    messages: &mut conversation::Conversation,
    request: crate::agent::CompactRequest,
) -> error::Result<crate::agent::CompactOutcome> {
    let cancellation = CancellationToken::new();
    install_interrupt_handler(agent);
    InterruptRelay::publish(cancellation.clone());
    let result = agent
        .compact_session(session_id, messages, request, cancellation)
        .await;
    InterruptRelay::clear();
    result
}

/// Run one agent turn with Ctrl+C wired to a fresh cancellation token. Hands the token to
/// [`InterruptRelay`] for the turn's duration, so a SIGINT during the turn cancels it and every
/// tool and sub-agent it spawned. Every `run_turn` callsite in the REPL / CLI path must go through
/// here; a bare `CancellationToken` with no signal source silently swallows Ctrl+C.
async fn run_turn_interruptible(
    agent: &Agent,
    session_id: &mut Option<uuid::Uuid>,
    messages: &mut conversation::Conversation,
    input: String,
    retention: agent::PromptRetention,
) -> error::Result<()> {
    let cancellation = CancellationToken::new();
    install_interrupt_handler(agent);
    InterruptRelay::publish(cancellation.clone());
    let result = agent
        .run_turn_retaining(
            session_id,
            messages,
            input,
            Vec::new(),
            cancellation,
            retention,
        )
        .await;
    InterruptRelay::clear();
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
    // Same rejection the HTTP surface applies, for the same reason: an empty turn costs a provider
    // round-trip to produce nothing, and the model has no way to tell it apart from a prompt whose
    // content went missing somewhere upstream.
    if prompt.trim().is_empty() {
        anyhow::bail!("the prompt must be a non-empty string");
    }
    let session_stats = Arc::new(stats::SessionStats::default());
    // Oneshot has no REPL, so approval requests can't reach a human. The channel below is
    // intentionally disconnected on the receiver side: `ReplFrontend::request_permission`'s `send`
    // will fail, and the agent surfaces a `cancelled` tool result, same end behavior as the
    // pre-refactor `None` approval sender.
    let (noninteractive_sender, _) = std::sync::mpsc::channel::<repl::AgentToReplEvent>();
    // One episode for the whole run. Oneshot draws no prompt, so there is no line above to space
    // away from and no prompt below to space towards: both blanks are configured off here rather
    // than left to a bracket that would have nothing to bracket against. What the console still
    // buys is the rest of it -- one owner for the streaming renderer, and the row-settling that
    // keeps a turn's last paragraph off an MCP progress line.
    let console = Arc::new(std::sync::Mutex::new(console::Console::new(
        console::Spacing {
            newline_before_prompt: false,
            newline_after_prompt: false,
        },
        config.render_mode,
    )));
    with_console(&console, |console| {
        console.open_episode(console::RowState::Empty)
    });
    let oneshot_frontend: Arc<dyn frontend::Frontend> =
        Arc::new(repl::ReplFrontend::new(repl::ReplFrontendConfig {
            console: Arc::clone(&console),
            show_session_id_on_create: config.show_session_id_on_create,
            show_token_usage: config.show_token_usage,
            thinking_show_content: config.thinking_show_content,
            tool_params: config.tool_params,
            agent_event_sender: noninteractive_sender,
        }));
    let launch_cwd = std::env::current_dir().unwrap_or_else(|error| {
        tracing::warn!("could not read process cwd at startup: {}", error);
        std::path::PathBuf::from(".")
    });
    // Resolved before the agent is built, not after: which session this is decides which provider
    // profile the agent runs on, which level it runs at and which directory it opens in, and the
    // agent carries all three.
    let ResumedSession {
        mut session_id,
        mut messages,
        lock: _session_lock,
        permission: start_permission,
        repin,
        cwd: recorded_cwd,
    } = resolve_session_resume(&session_manager, &config, &console).await?;
    // A resumed session reopens where it was, not where this shell is. See
    // `resume_working_directory`.
    let cwd: crate::workspace::SharedCwd = Arc::new(std::sync::RwLock::new(
        resume_working_directory(recorded_cwd, &launch_cwd, session_id),
    ));

    let shared_permission = SharedPermission::new(start_permission, config.enabled_permissions);
    if start_permission == crate::permission::Permission::Read {
        crate::sandbox::warn_if_sandbox_issues(
            &crate::sandbox::SandboxState::from_config(&config),
            crate::sandbox::WarnContext::InitialReadMode,
        );
    }
    // `ask` has nowhere to ask from here: `oneshot_frontend` is built on a channel whose receiver
    // is dropped, so every approval request fails to send and the tool is refused. Say so once,
    // up front, rather than letting the run look like the model simply chose not to use its tools.
    //
    // Against the level the run actually starts at, which a resumed session brings with it, rather
    // than against the configured default.
    if start_permission == crate::permission::Permission::Ask {
        tracing::warn!(
            "permission is 'ask' but one-shot mode has no interactive prompt: every tool that \
             needs approval will be denied. Use --permission workspace or unrestricted, or drop \
             --oneshot."
        );
    }

    let providers = cli_provider_registry(&config, token_store, Arc::clone(&session_stats))?;
    // Before the agent, because the agent resolves the row: a repin that has not landed yet would
    // build this run on the binding the session is leaving.
    if let (Some(id), Some(binding)) = (session_id, repin) {
        apply_session_repin(&session_manager, &providers, id, binding).await?;
    }
    let agent = create_agent_from_config(
        &config,
        session_id,
        session_manager.clone(),
        shared_permission,
        &providers,
        mcp_manager.as_ref(),
        oneshot_frontend,
        cwd,
        Arc::clone(&session_stats),
        Arc::new(std::sync::atomic::AtomicU64::new(0)),
        // `--oneshot` prints one answer and exits; nothing reads a live gauge.
        Arc::new(std::sync::atomic::AtomicU64::new(0)),
    )
    .await?;

    match run_turn_interruptible(
        &agent,
        &mut session_id,
        &mut messages,
        prompt,
        agent::PromptRetention::Keep,
    )
    .await
    {
        Ok(()) => {}
        Err(error::MekaError::Interrupted) => {
            with_console(&console, |console| console.annotation("interrupted"));
        }
        // Closed before returning, so a turn that streamed a partial answer and then failed still
        // shows what it streamed. `TurnFinished` closes the happy path; nothing closed this one.
        Err(error) => {
            close_console_episode(&console);
            return Err(error.into());
        }
    }
    close_console_episode(&console);

    // A one-shot run exits with the turn, so there is no later turn to deliver an outcome into.
    // Waiting here degrades a background call into a slow synchronous one, which is a worse deal
    // than the agent asked for but an honest one; exiting instead would leave a promise nothing can
    // keep, and kill the work halfway through besides.
    if config.background.enabled
        && let Some(id) = session_id
    {
        let outstanding = agent.background_tasks().running_count(id).await;
        if outstanding > 0 {
            tracing::info!(
                "waiting for {} background task(s) before exiting; a one-shot run has no later \
                 turn to report them in",
                outstanding
            );
            // Interruptible. A one-shot has no REPL loop and no per-turn signal listener by this
            // point, so an unbounded await here made the process ignore Ctrl+C entirely whenever a
            // background task never finished. Racing the signal keeps the documented "wait for
            // outstanding work" behaviour while leaving the user a way out; the outcomes collected
            // just below still report whatever did finish.
            let tasks = agent.background_tasks();
            install_interrupt_handler(&agent);
            tokio::select! {
                _ = tasks.wait_for_session(id) => {}
                _ = INTERRUPT_RELAY.pressed.notified() => {
                    with_console(&console, |console| {
                        console.annotation("stopped waiting for background tasks")
                    });
                }
            }
        }
        // Collected unconditionally, not only when this process started something. Resuming a
        // session sweeps whatever the *last* process left running into `interrupted`, and without
        // this a one-shot resume would answer the prompt and exit while that report sat
        // undelivered.
        let outcomes = collect_background_outcomes(&session_manager, session_id).await;
        if !outcomes.is_empty() {
            // Printed rather than delivered as a turn: the agent's answer has already been given
            // and the process is on its way out, so this is for the human reading the output.
            eprintln!();
            eprint!("{}", crate::background::render_outcomes(&outcomes));
        }
    }

    if let Some(id) = session_id
        && config.show_session_id_on_exit
    {
        render::render_session_id("Leaving session", &id.to_string());
    }

    // Same pairing the REPL does: this path attached the registry to the manager when the agent was
    // built, so it detaches it here rather than leaving the cycle for the process teardown.
    if let Some(manager) = &mcp_manager {
        manager.detach_registry(agent.tool_registry()).await;
    }
    drop(agent);

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
    // Where the shell was. Kept separate from the per-session `cwd` below rather than folded into
    // it, because the two part company the moment a resumed session opens somewhere else, and a
    // bare `/cd` returns here.
    let launch_cwd = std::env::current_dir().unwrap_or_else(|error| {
        tracing::warn!("could not read process cwd at startup: {}", error);
        std::path::PathBuf::from(".")
    });

    // Resolve session resumption BEFORE spawning the REPL so the "Resuming session" message appears
    // before the first prompt, and before the permission and cwd cells exist because a resumed
    // session brings both.
    // Everything printed between two prompts, wherever it came from. One instance, shared by the
    // agent's frontend, the blocking REPL thread and this loop, because the blank lines that
    // bracket an episode follow from what the episode did rather than from which of the three
    // happened to answer it.
    //
    // Built before the session is resolved because the resume banner, the replayed history and any
    // prompt queued on the command line all belong to the episode that ends at the *first* prompt.
    // Opening it here is what gives that episode the same brackets every later one gets.
    let console = Arc::new(std::sync::Mutex::new(console::Console::new(
        console::Spacing {
            newline_before_prompt: config.newline_before_prompt,
            newline_after_prompt: config.newline_after_prompt,
        },
        config.render_mode,
    )));
    with_console(&console, |console| {
        console.open_episode(console::RowState::Empty)
    });
    let repl_console = Arc::clone(&console);

    let ResumedSession {
        mut session_id,
        mut messages,
        lock: resumed_lock,
        permission: start_permission,
        repin,
        cwd: recorded_cwd,
    } = resolve_session_resume(&session_manager, &config, &console).await?;

    // Per-session working directory, shared by reference between the REPL (prompt + `/cd`) and the
    // agent (file/shell/find/grep tools + environment-context block). Process cwd is never mutated.
    // A resumed session opens where it recorded rather than where this shell is; see
    // `resume_working_directory`.
    let cwd: crate::workspace::SharedCwd = Arc::new(std::sync::RwLock::new(
        resume_working_directory(recorded_cwd, &launch_cwd, session_id),
    ));

    let shared_permission = SharedPermission::new(start_permission, config.enabled_permissions);
    if start_permission == crate::permission::Permission::Read {
        crate::sandbox::warn_if_sandbox_issues(
            &crate::sandbox::SandboxState::from_config(&config),
            crate::sandbox::WarnContext::InitialReadMode,
        );
    }

    if !messages.is_empty() {
        match config.resume_show_recent {
            Some(n) if n > 0 => {
                // The replay is part of the episode that ends at the first prompt, so its blank
                // line is that episode's closing bracket rather than a rule of its own. Announcing
                // only when something rendered keeps the empty case (a tail of tool calls with no
                // text) unbracketed.
                if render::render_message_history(
                    render::last_n_turns(messages.as_slice(), n),
                    &history_render_options(&config),
                ) {
                    with_console(&console, |console| console.announce_foreign_output());
                }
            }
            _ => {
                if reprint_last_message(messages.as_slice(), config.render_mode) {
                    with_console(&console, |console| console.announce_foreign_output());
                }
            }
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
            console: Arc::clone(&console),
            show_session_id_on_create: config.show_session_id_on_create,
            show_token_usage: config.show_token_usage,
            thinking_show_content: config.thinking_show_content,
            tool_params: config.tool_params,
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
    let repl_skill_roots = config.skill_roots();
    let repl_history_db_path = Some(session_manager.database_path().to_path_buf());

    // Live context gauge for the prompt: a shared counter the agent writes after each turn and the
    // prompt reads each render. Created here (before the agent) so the REPL, spawned below, can
    // hold it; the agent adopts the same atomic via `set_context_tokens`. Seeded with an
    // estimate when resuming so the gauge isn't blank until the first new turn measures the
    // context exactly.
    // A handle, seeded from the process default and corrected to the session's own window as soon
    // as the agent below resolves it. It cannot be read from `config` and left alone: this session
    // may be pinned to another profile, and `/provider` may move it again, and a prompt dividing by
    // a window the agent is not gauging against contradicts `/status` on the very next line.
    let context_window_gauge = Arc::new(std::sync::atomic::AtomicU64::new(
        config
            .session_context_window
            .unwrap_or(crate::provider::DEFAULT_CONTEXT_WINDOW),
    ));
    let context_tokens = Arc::new(std::sync::atomic::AtomicU64::new(0));
    if !messages.is_empty() {
        context_tokens.store(
            tokens::estimate_messages(messages.as_slice()),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    let context_indicator = config.show_context_in_prompt.then(|| {
        (
            Arc::clone(&context_tokens),
            Arc::clone(&context_window_gauge),
        )
    });

    // Shared with reedline: the scheduler watcher sets it, `read_line` polls it and returns
    // `Signal::ExternalBreak` so a due job can interrupt an idle prompt.
    let schedule_wake = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let repl_wake = Arc::clone(&schedule_wake);

    // What `/provider` reports and rewrites. Seeded through the same door the agent resolves with,
    // so a resumed session shows the profile it recorded rather than the configured default. A
    // session that cannot resolve one never reaches a prompt at all, so the default keeps the seed
    // total.
    let current_provider = Arc::new(std::sync::RwLock::new(match &repin {
        // The repin has not been committed yet (it waits for the registry, below), but it is what
        // the row will say by the time anything reads this cell.
        Some(binding) => binding.clone(),
        None => provider_for_config(&session_manager, &config, session_id)
            .await
            .unwrap_or_default(),
    }));
    let repl_current_provider = Arc::clone(&current_provider);
    // Name and backend, so `/provider` can say what each profile *is* rather than only what it is
    // called. `providers` is a `BTreeMap`, so this is already in name order.
    let repl_configured_providers: Vec<(String, String)> = config
        .providers
        .iter()
        .map(|(name, profile)| (name.clone(), profile.backend.clone()))
        .collect();

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
    // Kept back from the move below so the scheduler can read the *current* level rather than the
    // one this process started at. See the `run_due` call in the `Wake` arm.
    let scheduler_permission = shared_permission.clone();
    // Held past agent construction, unlike the other hosts', because `/provider` rebuilds one
    // mid-session and the cache is what makes the profile it left reusable when it comes back.
    let providers =
        match cli_provider_registry(&config, token_store.clone(), Arc::clone(&session_stats)) {
            Ok(providers) => providers,
            Err(error) => {
                with_console(&console, |console| console.error(&error));
                return Err(AlreadyReported.into());
            }
        };
    // Before the agent, because the agent resolves the row: a repin that has not landed yet would
    // build this run on the binding the session is leaving.
    if let (Some(id), Some(binding)) = (session_id, repin)
        && let Err(error) = apply_session_repin(&session_manager, &providers, id, binding).await
    {
        with_console(&console, |console| console.error(&error));
        return Err(AlreadyReported.into());
    }
    let mut agent = match create_agent_from_config(
        &config,
        session_id,
        session_manager.clone(),
        shared_permission,
        &providers,
        mcp_manager.as_ref(),
        Arc::clone(&repl_frontend),
        Arc::clone(&cwd),
        Arc::clone(&session_stats),
        Arc::clone(&context_tokens),
        // The prompt's own gauge, handed in rather than seeded and corrected: it *is* the cell the
        // agent publishes into, so `/provider` cannot move one without the other.
        Arc::clone(&context_window_gauge),
    )
    .await
    {
        Ok(agent) => agent,
        Err(error) => {
            with_console(&console, |console| console.error(&error));
            let gone = match session_id {
                Some(id) => recorded_profile_is_gone(&session_manager, &config, id).await,
                None => false,
            };
            // The default when there is one, so the suggested move is the profile the rest of this
            // config already runs on. With no profile at all there is nowhere to move to, and the
            // generic setup example is the more useful answer.
            let move_to = config
                .active_profile
                .as_deref()
                .or_else(|| config.providers.keys().next().map(String::as_str));
            render::render_provider_setup_hint(session_id.filter(|_| gone).zip(move_to).map(
                |(session_id, move_to)| render::MissingSessionProfile {
                    session_id,
                    move_to,
                },
            ));
            return Err(AlreadyReported.into());
        }
    };
    // Point the agent's live context counter at the same atomic the REPL prompt holds, so the
    // prompt gauge tracks what the agent writes after each turn (and the resume seed above).
    agent.set_context_tokens(Arc::clone(&context_tokens));

    // Spawned once there is an agent to answer it, which is what makes every refusal above final.
    // Started before the agent, the prompt outlived a failed construction: the loop below never
    // ran, so `/provider` -- the one way to move a session off a profile that has left
    // `config.toml` -- was sent to nobody and waited for an answer that could not come, and the
    // user was left typing into a shell that ignored them.
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
            launch_cwd,
            repl_mcp_server_names,
            repl_skill_roots,
            repl_history_db_path,
            repl_wake,
            repl_current_provider,
            repl_configured_providers,
            repl_console,
        );
    });

    // One slot for the session lock from here on, whichever way the session was reached: the agent
    // fills it the moment it creates one, and a resumed session's lock -- taken above, before the
    // REPL thread existed -- moves into the same place. `/fork` replaces what is in it and the exit
    // path empties it, so neither has to know which of the two put it there.
    let session_lock = agent.session_lock_slot();
    hold_session_lock(&session_lock, resumed_lock);

    // Mirrors the loop's `session_id` for the watcher below, which runs on another task and cannot
    // borrow it. Written after every event, which is the only thing that can change it.
    let repl_shared_session_id = Arc::new(std::sync::RwLock::new(session_id));

    // Watcher, not a scheduler: it only nudges reedline awake. The agent loop below owns the
    // conversation, so it has to be the thing that evaluates gates and runs the turn -- otherwise
    // two tasks would be appending to `messages`. Background outcomes ride the same watcher and the
    // same flag for the same reason.
    let schedule_watcher = {
        let session_manager = session_manager.clone();
        let shared_session_id = Arc::clone(&repl_shared_session_id);
        let poll_interval = config.schedule.poll_interval;
        let schedule_enabled = config.schedule.enabled;
        let background_enabled = config.background.enabled;
        // The watcher asks whether a wake would produce work, and part of that answer is the
        // session's live permission resolved against this installation's enabled set.
        let watcher_schedule_config = config.schedule.clone();
        tokio::spawn(async move {
            if !schedule_enabled && !background_enabled {
                return;
            }
            let mut ticker = tokio::time::interval(poll_interval);
            ticker.tick().await;
            loop {
                ticker.tick().await;
                let Some(current) = shared_session_id
                    .read()
                    .map(|guard| *guard)
                    .unwrap_or_else(|poisoned| *poisoned.into_inner())
                else {
                    continue;
                };
                if schedule_enabled {
                    // The same question the fire door asks, as far as it can be answered without
                    // running a gate probe. Asking the weaker "is a row due" here is what let a
                    // parked job interrupt the prompt every `poll_interval` to run nothing.
                    match crate::schedule::wake_would_produce_work(
                        &session_manager,
                        &watcher_schedule_config,
                        current,
                        chrono::Utc::now(),
                    )
                    .await
                    {
                        Ok(true) => {
                            schedule_wake.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        Ok(false) => {}
                        Err(error) => tracing::warn!("scheduler watcher failed: {}", error),
                    }
                }
                if background_enabled {
                    match session_manager
                        .background_store()
                        .list_undelivered_background_tasks(current)
                        .await
                    {
                        Ok(ready) if !ready.is_empty() => {
                            schedule_wake.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        Ok(_) => {}
                        Err(error) => tracing::warn!("background watcher failed: {}", error),
                    }
                }
            }
        })
    };

    while let Some(event) = input_receiver.recv().await {
        match event {
            // Recorded on the row, then straight back to the prompt: this is not a turn and must
            // not look like one, so no `AgentToReplEvent::Done` and no spacing.
            //
            // The row is what a *scheduled gate* is re-checked against at fire time, and a row that
            // carries no level falls back to the polling process's own startup flag. Before this,
            // a REPL session's row was always NULL, so a `meka serve` sharing the data directory
            // answered that question with its own `--permission` -- and kept running a gate the
            // user had just withdrawn with Shift+Tab. `docs/book/src/usage/scheduling.md` documents
            // that withdrawal, so it was a promise the code did not keep whenever serve was up.
            ReplEvent::PermissionChanged(level) => {
                let Some(id) = session_id else {
                    // No row yet: the level the first turn creates the session with is this one, so
                    // there is nothing to correct.
                    continue;
                };
                if let Err(error) = session_manager
                    .update_session_permission(id, &level.to_string())
                    .await
                {
                    // `warn!` rather than `?`: the user's own level has already moved in this
                    // process, and failing the REPL over a bookkeeping write would be worse than
                    // the stale row. Loud because the consequence is precisely that another process
                    // may still act on the old level.
                    tracing::warn!(
                        "could not record permission `{}` on session {}: {}. Another meka process \
                         may still act on this session's previous level",
                        level,
                        id,
                        error
                    );
                }
            }
            // Recorded for the same reasons the level above is, and it lands here rather than in
            // the REPL thread because that thread is not async and holds no `SessionManager`. The
            // row is where the *next* resume opens the session, and it is the directory a scheduled
            // tool-gate is re-checked in; a `/cd` that stopped at the in-memory cell left both
            // answering with the directory the session was created in.
            ReplEvent::CwdChanged(path) => {
                record_session_cwd(&session_manager, session_id, &path).await;
            }
            // The row moves first. If recording the change failed but the agent had already
            // switched, the next `meka -c` would silently go back to the old profile, which is the
            // surprise this whole feature exists to remove.
            ReplEvent::ProviderChange(name) => {
                // A labelled block rather than `continue`, so every way out passes the `Done`
                // below. Without it this was the one forwarded command that did not hand the
                // prompt back, and the REPL thread painted the next prompt while this task was
                // still deciding what to print: the confirmation, or the error, landed on top of
                // the line the user had started typing.
                'switch: {
                    // Membership first, so a typo gets a message about a name the user just typed
                    // rather than `look_up_profile`'s, which is written for a session whose
                    // recorded profile went missing and speaks about restoring `config.toml`.
                    if !config.providers.contains_key(&name) {
                        with_console(&console, |console| {
                            console.error(&format!(
                                "no provider profile named '{}' (configured: {})",
                                name,
                                config
                                    .providers
                                    .keys()
                                    .cloned()
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ))
                        });
                        break 'switch;
                    }
                    // The profile as configured. `/provider` moves the session to that bundle
                    // entire, which is the only thing naming a profile can mean.
                    let resolved = match resolved_binding(&providers, name.clone()).await {
                        Ok(resolved) => resolved,
                        Err(error) => {
                            with_console(&console, |console| console.error(&error));
                            break 'switch;
                        }
                    };
                    if let Some(id) = session_id {
                        match session_manager
                            .set_recorded_provider(id, &resolved.binding)
                            .await
                        {
                            // No row: the session was deleted from under this process, so there is
                            // nothing to switch and the next turn will fail on its own terms.
                            Ok(false) => {
                                with_console(&console, |console| {
                                    console.error(&format!("session {} no longer exists", id))
                                });
                                break 'switch;
                            }
                            Ok(true) => {}
                            Err(error) => {
                                with_console(&console, |console| console.error(&error));
                                break 'switch;
                            }
                        }
                    }
                    // The prompt gauge is the cell the agent publishes into, so this moves it too.
                    agent.set_provider(resolved);
                    match current_provider.write() {
                        Ok(mut guard) => *guard = name.clone(),
                        Err(poisoned) => *poisoned.into_inner() = name.clone(),
                    }
                    with_console(&console, |console| {
                        console.line(&format!("Provider profile set to: {}", name))
                    });
                }
                if agent_event_sender
                    .send(repl::AgentToReplEvent::Done)
                    .is_err()
                {
                    break;
                }
            }
            ReplEvent::Wake => {
                let scope = match session_id {
                    Some(id) => crate::schedule::SchedulerScope::OneSession(id),
                    // Nothing can be due before the session exists; the watcher would not have
                    // woken us, but the loop must not assume that.
                    None => {
                        if agent_event_sender
                            .send(repl::AgentToReplEvent::Done)
                            .is_err()
                        {
                            break;
                        }
                        continue;
                    }
                };
                // Collected first, then run: `run_due` takes an `Fn`, and the turn needs `&mut`
                // access to `messages` and `session_id`, which belong to this loop alone. Gating
                // and stamping have already happened by the time a wakeup lands here.
                let fired = std::sync::Mutex::new(Vec::new());
                // Gated on the switch, not just on having been woken. `run_due` has no `enabled`
                // check of its own -- the flag is enforced by whoever decides to poll -- and this
                // arm is now also reached by a finished background task setting the same wake flag.
                // Without this, turning scheduling off while background calls are on would still
                // fire the jobs already in the database.
                // `host_permission` is resolved once from config, so under the REPL it is the level
                // this process *started* at, and a REPL session carries no per-session level on its
                // row (`ResolvedScheduleConfig::host_permission` documents that fallback). The
                // gate's live re-check therefore read a snapshot that Shift+Tab and `/permission`
                // never touch, and it failed in both directions: starting at the default `read` and
                // cycling up to `unrestricted` to author a gate left it refused forever, while
                // starting at `unrestricted` and cycling down to `read` kept firing
                // it -- which is the withdrawal the re-check exists for. Overriding
                // it here is the whole fix; `run_due` reads this clone, not the
                // process-wide one.
                let schedule_config =
                    schedule_config_at_live_permission(&config.schedule, &scheduler_permission);
                if config.schedule.enabled
                    && let Err(error) = crate::schedule::run_due(
                        &session_manager,
                        &schedule_config,
                        &scope,
                        &|wakeup: crate::schedule::Wakeup| {
                            if let Ok(mut collected) = fired.lock() {
                                collected.push(wakeup);
                            }
                            // The REPL owns this session outright, so it never defers: the turn
                            // runs below, on this same loop, before
                            // anything else is handled.
                            std::future::ready(crate::schedule::FireOutcome::Ran)
                        },
                    )
                    .await
                {
                    with_console(&console, |console| console.error(&error));
                }
                let fired = fired
                    .into_inner()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                // Close the prompt line, once, and only if something is actually going to run.
                //
                // reedline moves the cursor below the input area on its way out of `read_line`
                // (`move_cursor_to_end`), but only when it is genuinely exiting: the guard is
                // `suspended_state.is_none()`, and the break path sets `suspended_state` precisely
                // because the host is expected to print and come back. So unlike a submitted line,
                // a scheduled wake leaves the cursor parked at the end of the prompt, and the
                // turn's own `newline_after_prompt` blank gets spent terminating that line instead
                // of producing a gap. Emitting the terminator here starts the turn at column zero
                // of a fresh line, which is where a typed turn starts, so both `[display]` spacing
                // settings mean the same thing either way.
                // Outcomes are collected here, alongside the due jobs, so one wake delivers both
                // rather than one of them and then another tick.
                let outcomes = if config.background.enabled {
                    collect_background_outcomes(&session_manager, session_id).await
                } else {
                    Vec::new()
                };
                if !outcomes.is_empty() {
                    let prompt = crate::background::render_outcomes(&outcomes);
                    match run_turn_interruptible(
                        &agent,
                        &mut session_id,
                        &mut messages,
                        prompt,
                        agent::PromptRetention::Keep,
                    )
                    .await
                    {
                        Ok(()) => {}
                        Err(error::MekaError::Interrupted) => {
                            with_console(&console, |console| console.annotation("interrupted"));
                            report_background_survivors(&agent).await;
                        }
                        Err(error) => {
                            with_console(&console, |console| console.error(&error));
                        }
                    }
                }
                for wakeup in fired {
                    let prompt = wakeup.render_prompt();
                    match run_turn_interruptible(
                        &agent,
                        &mut session_id,
                        &mut messages,
                        prompt,
                        wakeup.job.prompt_retention(),
                    )
                    .await
                    {
                        Ok(()) => {}
                        Err(error::MekaError::Interrupted) => {
                            with_console(&console, |console| console.annotation("interrupted"));
                            report_background_survivors(&agent).await;
                        }
                        Err(error) => {
                            with_console(&console, |console| console.error(&error));
                        }
                    }
                }
                if agent_event_sender
                    .send(repl::AgentToReplEvent::Done)
                    .is_err()
                {
                    break;
                }
            }
            ReplEvent::UserInput(input) => {
                match run_turn_interruptible(
                    &agent,
                    &mut session_id,
                    &mut messages,
                    input,
                    agent::PromptRetention::Keep,
                )
                .await
                {
                    Ok(()) => {}
                    Err(error::MekaError::Interrupted) => {
                        with_console(&console, |console| console.annotation("interrupted"));
                        report_background_survivors(&agent).await;
                    }
                    Err(error) => {
                        with_console(&console, |console| console.error(&error));
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
                // Every command answered here says something, even if only that a list is empty,
                // and much of it prints through the `cli` modules the console cannot see. One
                // announcement covers all of them.
                //
                // There is deliberately no "does this one answer by running a turn" exception any
                // more. Announcing is idempotent within an episode -- the opening blank is spent
                // once, by whichever writer gets there first -- so a command that runs a turn is
                // spaced identically whether the turn happens or it bails first. The predicate
                // that used to make that distinction is what left `/skill nosuchskill` printing
                // its error flush against both the line above and the prompt below.
                with_console(&console, |console| console.announce_foreign_output());
                match command {
                    repl::SlashCommand::Session => match &session_id {
                        Some(id) => with_console(&console, |console| {
                            console.session_id("Current session", &id.to_string())
                        }),
                        None => eprintln!("No active session yet."),
                    },
                    repl::SlashCommand::Compact(instructions) => {
                        let request = crate::agent::CompactRequest {
                            origin: crate::agent::CompactOrigin::Manual,
                            instructions: instructions
                                .as_deref()
                                .map(str::trim)
                                .filter(|value| !value.is_empty())
                                .map(str::to_string),
                            keep_recent: None,
                        };
                        match compact_interruptible(&agent, &mut session_id, &mut messages, request)
                            .await
                        {
                            Ok(outcome) => {
                                with_console(&console, |console| {
                                    console.hint(&render::compaction_summary(&outcome))
                                });
                            }
                            Err(error) => {
                                with_console(&console, |console| console.error(&error));
                            }
                        }
                    }
                    repl::SlashCommand::Rewind(turns) => {
                        let turns = turns.unwrap_or(1);
                        // `rewind(0)` returns `None` unconditionally, so without this the `None`
                        // arm below would report "fewer than 0 turn(s)". The count is what's
                        // wrong, not the conversation.
                        let rewound = if turns == 0 {
                            None
                        } else {
                            messages.rewind(turns)
                        };
                        match (session_id, rewound) {
                            (Some(id), Some(event)) => {
                                if let Err(error) = session_manager.save_event(id, &event).await {
                                    // Put the turns back rather than leave memory and disk
                                    // disagreeing, which would resurrect them on the next resume
                                    // and make the rewind look like it silently un-did itself.
                                    messages.pop_repair();
                                    with_console(&console, |console| console.error(&error));
                                } else {
                                    agent.reset_conversation_markers().await;
                                    with_console(&console, |console| {
                                        console.hint(&format!(
                                            "Rewound {} turn(s). The model no longer sees them; \
                                         `meka session export` still does.",
                                            turns,
                                        ))
                                    });
                                }
                            }
                            // No session means nothing was ever persisted, so the in-memory rewind
                            // (which did happen) is the whole story.
                            (None, Some(_)) => {
                                agent.reset_conversation_markers().await;
                                with_console(&console, |console| {
                                    console.hint(&format!("Rewound {} turn(s).", turns))
                                });
                            }
                            (_, None) if turns == 0 => {
                                eprintln!(
                                    "Nothing to rewind: /rewind takes a turn count of 1 or more."
                                );
                            }
                            (_, None) => {
                                eprintln!(
                                    "Nothing to rewind: the conversation has fewer than {} turn(s).",
                                    turns
                                );
                            }
                        }
                    }
                    repl::SlashCommand::Export => match &session_id {
                        Some(id) => {
                            match crate::session::cli::export_session(
                                &session_manager,
                                *id,
                                None,
                                cli::SessionExportFormat::Markdown,
                            )
                            .await
                            {
                                // The name is generated from the session id and the file lands in
                                // the working directory, so a REPL user who is not told it has
                                // nowhere to look. `meka session export` stays quiet: there the
                                // shell is the one that knows.
                                Ok(Some(path)) => {
                                    // Shown absolute: `/cd` moves the session's directory while
                                    // the export lands in the process's, so a bare filename would
                                    // point at the wrong one.
                                    let shown = std::env::current_dir()
                                        .map(|dir| dir.join(&path))
                                        .unwrap_or(path);
                                    eprintln!("Exported session to {}", shown.display());
                                }
                                Ok(None) => {}
                                Err(error) => {
                                    with_console(&console, |console| console.error(&error))
                                }
                            }
                        }
                        None => eprintln!("No active session to export."),
                    },
                    repl::SlashCommand::Fork => match session_id {
                        Some(id) => match crate::session::cli::fork_and_lock(&session_manager, id)
                            .await
                        {
                            Ok(crate::session::cli::ForkHandoff::Switched { id, lock }) => {
                                // Replacing the slot's contents drops the original guard only now
                                // that the new one is held; see
                                // `crate::session::cli::fork_and_lock`.
                                hold_session_lock(&session_lock, Some(lock));
                                session_id = Some(id);
                                // `messages` is deliberately untouched, so the branch happens at
                                // the current head and the next turn continues in the copy.
                                with_console(&console, |console| {
                                    console.session_id("Forked session", &id.to_string())
                                });
                            }
                            Ok(crate::session::cli::ForkHandoff::LockFailed { id, error }) => {
                                with_console(&console, |console| console.error(&error));
                                with_console(&console, |console| {
                                    console.hint(&format!(
                                        "Staying in the original. The copy exists: {}",
                                        id
                                    ))
                                });
                            }
                            Ok(crate::session::cli::ForkHandoff::SourceGone) => {
                                eprintln!("Session no longer exists: {}", id);
                            }
                            Err(error) => eprintln!("Failed to fork session: {}", error),
                        },
                        None => eprintln!("No active session to fork."),
                    },
                    repl::SlashCommand::McpList => {
                        if let Err(error) = mcp::cli::run_list(
                            &config.mcp_servers,
                            mcp_manager.as_ref(),
                            &session_manager.token_store(),
                        )
                        .await
                        {
                            with_console(&console, |console| console.error(&error));
                        }
                    }
                    // These three report success at `info!` and print nothing, which is right for
                    // the `meka mcp …` CLI (the exit code carries it) and wrong here: a REPL
                    // command has no exit code, so silence is indistinguishable from the command
                    // never having run, and it leaves the `[display]` blank lines wrapped around
                    // an empty region. `/permission` sets the precedent for confirming a state
                    // change the user asked for.
                    repl::SlashCommand::McpReconnect { server } => {
                        match mcp::cli::run_reconnect(&config.mcp_servers, &token_store, &server)
                            .await
                        {
                            // "Connected", not "Reconnected": this is a smoke test on a throwaway
                            // client, and the session's own connection to that server is untouched.
                            Ok(()) => eprintln!("Connected to '{}'.", server),
                            Err(error) => with_console(&console, |console| console.error(&error)),
                        }
                    }
                    repl::SlashCommand::McpLogin { server } => {
                        match mcp::cli::run_login(&config.mcp_servers, &token_store, &server).await
                        {
                            Ok(()) => eprintln!("Authorized '{}'.", server),
                            Err(error) => with_console(&console, |console| console.error(&error)),
                        }
                    }
                    repl::SlashCommand::McpLogout { server } => {
                        match mcp::cli::run_logout(&config.mcp_servers, &token_store, &server).await
                        {
                            Ok(()) => eprintln!("Cleared credentials for '{}'.", server),
                            Err(error) => with_console(&console, |console| console.error(&error)),
                        }
                    }
                    repl::SlashCommand::McpPrompt {
                        server,
                        prompt: prompt_name,
                        args,
                    } => 'prompt: {
                        let Some(manager) = mcp_manager.as_ref() else {
                            eprintln!("No MCP servers configured.");
                            break 'prompt;
                        };
                        let entry = manager.server_entry(&server);
                        let Some(entry) = entry else {
                            // Labelled break, not `continue`: `continue` targets the agent
                            // loop, skipping the `AgentToReplEvent::Done` send below and
                            // leaving the REPL thread parked in `wait_for_agent` with no
                            // prompt, for good. Same reason as `SkillInvoke`'s `'invoke`.
                            eprintln!(
                                "Unknown MCP server '{}' (configured: {}).",
                                server,
                                manager.server_names().join(", ")
                            );
                            break 'prompt;
                        };
                        // Map positional args to declared prompt argument names (lookup via
                        // prompts/list).
                        let arg_names = match mcp::list_prompts(
                            &entry,
                            &tokio_util::sync::CancellationToken::new(),
                        )
                        .await
                        {
                            Ok(prompts) => prompts
                                .into_iter()
                                .find(|p| p.name == prompt_name)
                                .and_then(|p| p.arguments)
                                .map(|args| args.into_iter().map(|a| a.name).collect::<Vec<_>>())
                                .unwrap_or_default(),
                            Err(error) => {
                                // The `McpConnection` error already names the server and the
                                // operation, so wrapping it here would say both twice.
                                with_console(&console, |console| console.error(&error));
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
                        match mcp::get_prompt(
                            &entry,
                            prompt_name.clone(),
                            arguments,
                            &tokio_util::sync::CancellationToken::new(),
                        )
                        .await
                        {
                            Ok(result) => {
                                // Render the prompt messages as a single user turn, same shape
                                // as the `mcp_prompt_get` tool output.
                                let mut body = String::new();
                                for message in &result.messages {
                                    let role = match message.role {
                                        rmcp::model::Role::User => "user",
                                        rmcp::model::Role::Assistant => "assistant",
                                    };
                                    if let rmcp::model::ContentBlock::Text(text) = &message.content
                                    {
                                        body.push_str(&format!("{}: {}\n", role, text.text));
                                    }
                                }
                                let user_input = body.trim().to_string();
                                if user_input.is_empty() {
                                    // Every other exit from this arm prints; a server whose
                                    // prompt renders to nothing would otherwise return the user
                                    // straight to a fresh prompt, which reads as "the command
                                    // did nothing" rather than "the prompt was empty".
                                    eprintln!(
                                        "'{}:{}' rendered an empty prompt; nothing to send.",
                                        server, prompt_name
                                    );
                                } else {
                                    match run_turn_interruptible(
                                        &agent,
                                        &mut session_id,
                                        &mut messages,
                                        user_input,
                                        agent::PromptRetention::Keep,
                                    )
                                    .await
                                    {
                                        Ok(()) => {}
                                        Err(error::MekaError::Interrupted) => {
                                            with_console(&console, |console| {
                                                console.annotation("interrupted")
                                            });
                                        }
                                        Err(error) => {
                                            with_console(&console, |console| console.error(&error))
                                        }
                                    }
                                }
                            }
                            Err(error) => {
                                with_console(&console, |console| console.error(&error));
                            }
                        }
                    }
                    repl::SlashCommand::MemoryList => {
                        if let Err(error) = memory::cli::run_list(
                            &session_manager.memory_store(true),
                            memory::cli::ListDetail::TableOnly,
                        )
                        .await
                        {
                            with_console(&console, |console| console.error(&error));
                        }
                    }
                    repl::SlashCommand::MemoryShow { name } => {
                        if let Err(error) =
                            memory::cli::run_show(&session_manager.memory_store(true), &name).await
                        {
                            with_console(&console, |console| console.error(&error));
                        }
                    }
                    // Scoped to the session in the REPL, unlike `meka schedule list`, which has no
                    // conversation to be "this one" and so shows every session's jobs.
                    repl::SlashCommand::ScheduleList => match session_id {
                        Some(id) => {
                            if let Err(error) = crate::schedule::cli::run_list_for_session(
                                &session_manager,
                                id,
                                &config.schedule,
                            )
                            .await
                            {
                                with_console(&console, |console| console.error(&error));
                            }
                        }
                        None => eprintln!("No active session yet."),
                    },
                    repl::SlashCommand::ScheduleCancel { id } => match session_id {
                        Some(session) => {
                            match session_manager
                                .schedule_store()
                                .cancel_scheduled_job(session, &id)
                                .await
                            {
                                Ok(Some(cancelled)) => {
                                    eprintln!(
                                        "Cancelled job {}.",
                                        &cancelled[..8.min(cancelled.len())]
                                    );
                                }
                                Ok(None) => {
                                    eprintln!("No scheduled job matching '{}'.", id);
                                }
                                Err(error) => {
                                    with_console(&console, |console| console.error(&error))
                                }
                            }
                        }
                        None => eprintln!("No active session yet."),
                    },
                    repl::SlashCommand::TaskList => match session_id {
                        Some(id) => {
                            if let Err(error) =
                                crate::background::cli::run_list_for_session(&session_manager, id)
                                    .await
                            {
                                with_console(&console, |console| console.error(&error));
                            }
                        }
                        None => eprintln!("No active session yet."),
                    },
                    repl::SlashCommand::TaskCancel { id } => match session_id {
                        Some(session) => {
                            // Recorded first, then signalled: `finish_background_task` only
                            // overwrites a `running` row, so a task finishing in the same instant
                            // cannot report success after the user was told it stopped.
                            match crate::background::cli::cancel(
                                &session_manager,
                                session,
                                id.as_deref(),
                            )
                            .await
                            {
                                Ok(cancelled) if cancelled.is_empty() => {
                                    eprintln!("No running background tasks.")
                                }
                                Ok(cancelled) => {
                                    for task_id in &cancelled {
                                        agent.background_tasks().cancel(task_id).await;
                                    }
                                    eprintln!("Cancelling {} background task(s).", cancelled.len());
                                }
                                Err(error) => {
                                    with_console(&console, |console| console.error(&error))
                                }
                            }
                        }
                        None => eprintln!("No active session yet."),
                    },
                    repl::SlashCommand::SkillList => {
                        if let Err(error) =
                            skills::cli::run_list(&config.skill_roots(), false).await
                        {
                            with_console(&console, |console| console.error(&error));
                        }
                    }
                    repl::SlashCommand::SkillInvoke { name, extra } => 'invoke: {
                        // Labeled block so the early-exit error paths can `break 'invoke` out of
                        // the arm body without skipping the `AgentToReplEvent::Done` send below;
                        // `continue` would short-circuit the outer `while let`, leaving the REPL
                        // stuck in `wait_for_agent` and never drawing the next prompt.
                        let installed = agent.skills().current().await;
                        let Some(skill) = installed.find(&name) else {
                            // Prose, not `{:?}` on a `Vec<&str>`. This is user-facing output, and
                            // the debug rendering printed `["a", "b"]` -- quotes, brackets and all
                            // -- for a list a person is meant to read and pick from.
                            let message = match installed.skip_reason(&name) {
                                // The same distinction `skill_read` draws for the model, in the
                                // same words: a file that is there and unreadable is not a missing
                                // skill.
                                Some(_) => installed.unavailable(&name),
                                None if installed.skills.is_empty() => {
                                    format!("unknown skill '{}'; no skills are installed", name)
                                }
                                None => format!(
                                    "unknown skill '{}'; available: {}",
                                    name,
                                    installed
                                        .skills
                                        .iter()
                                        .map(|s| s.name.as_str())
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                ),
                            };
                            with_console(&console, |console| console.error(&message));
                            break 'invoke;
                        };
                        let body = match skills::load_skill_body(skill).await {
                            Ok(body) => body,
                            Err(error) => {
                                with_console(&console, |console| {
                                    console.error(&format!(
                                        "failed to load skill '{}': {}",
                                        name, error
                                    ))
                                });
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
                        match run_turn_interruptible(
                            &agent,
                            &mut session_id,
                            &mut messages,
                            body,
                            agent::PromptRetention::Keep,
                        )
                        .await
                        {
                            Ok(()) => {}
                            Err(error::MekaError::Interrupted) => {
                                with_console(&console, |console| console.annotation("interrupted"));
                            }
                            Err(error) => with_console(&console, |console| console.error(&error)),
                        }
                    }
                    repl::SlashCommand::Status => {
                        let snap = agent.session_stats_snapshot();
                        let (context_tokens, context_window) = agent.context_usage();
                        let effort = agent.resolved_effort();
                        // This session's profile, not the process default's. Reading `config` here
                        // reported the default profile's model and backend beside a window and an
                        // effort that came from the session's, so `/status` and `/provider`
                        // contradicted each other on any resume onto a non-default profile.
                        let binding = agent.provider_binding().clone();
                        let settings = providers.settings(&binding);
                        render::render_session_status(
                            &snap,
                            &render::ModelStatus {
                                model: settings
                                    .as_ref()
                                    .ok()
                                    .and_then(|settings| settings.model.as_deref()),
                                profile: Some(binding.as_str()),
                                backend: settings
                                    .as_ref()
                                    .ok()
                                    .map(|settings| settings.backend.as_str()),
                                effort: effort.as_deref(),
                                thinking: settings
                                    .as_ref()
                                    .map(|settings| settings.thinking)
                                    .unwrap_or_default(),
                            },
                            messages.len(),
                            context_tokens,
                            context_window,
                        );
                    }
                    repl::SlashCommand::Usage => match agent.fetch_usage().await {
                        Ok(Some(usage)) => render::render_account_usage(&usage),
                        Ok(None) => with_console(&console, |console| {
                            console.hint("Account usage isn't available for this provider.")
                        }),
                        Err(error) => with_console(&console, |console| console.error(&error)),
                    },
                    repl::SlashCommand::History(limit) => {
                        let materialised = messages.as_slice();
                        let slice = match limit {
                            Some(n) => render::last_n_turns(materialised, n),
                            None => materialised,
                        };
                        // Say so rather than printing nothing, like every other list command
                        // (`/tasks`, `/memory`, `/skill`). Silence here would be ambiguous between
                        // "no history" and "the command did not run", and it would leave the
                        // `[display]` blank lines bracketing an empty region. `/history 0` asks
                        // for nothing and gets the neutral wording: there may well be a
                        // conversation, it just wasn't what was asked for.
                        if !render::render_message_history(slice, &history_render_options(&config))
                        {
                            if materialised.is_empty() {
                                eprintln!("No conversation history yet.");
                            } else {
                                eprintln!("Nothing to show.");
                            }
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
        // The loop's own `session_id` is authoritative; the watcher reads this mirror. An event is
        // the only thing that can create or replace a session, so syncing here is sufficient.
        match repl_shared_session_id.write() {
            Ok(mut cell) => *cell = session_id,
            Err(poisoned) => *poisoned.into_inner() = session_id,
        }
    }

    schedule_watcher.abort();
    drop(agent_event_sender);
    repl_handle.await?;

    if let Some(id) = session_id
        && config.show_session_id_on_exit
    {
        with_console(&console, |console| {
            console.session_id("Leaving session", &id.to_string())
        });
    }
    // The last episode has no prompt after it, but closing it is still what settles the row and
    // flushes anything a turn left open. Nothing else runs after this.
    close_console_episode(&console);
    // Emptied after the "Leaving session" message so the lock is held until the very end; the OS
    // releases the underlying flock when the FD closes. Emptied rather than dropped: the slot is
    // shared with the agent, which is still alive here, so letting this handle fall out of scope
    // would release nothing.
    hold_session_lock(&session_lock, None);

    // Stop this process's background tasks on the way out.
    //
    // Nothing did before: `/exit` broke the loop and returned, and `BackgroundTasks` has no `Drop`,
    // so a detached `execute_command` kept running -- `setsid()`-ed, so not even a terminal hangup
    // reaches it -- with no meka process tracking it. Its row stayed `running` and the next session
    // open swept it to `interrupted`, telling the model work had died that was in fact still going,
    // possibly still writing to the workspace. The HTTP `DELETE /v1/sessions/{id}` handler already
    // does this, and its comment claimed "The REPL does the same thing on its way out", which is
    // what this makes true.
    let stopped = agent.background_tasks().cancel_all().await;
    if stopped > 0 {
        with_console(&console, |console| {
            console.annotation(&format!(
                "stopping {} background task{}",
                stopped,
                if stopped == 1 { "" } else { "s" }
            ))
        });
        // The shutdown notice is the last thing the terminal sees, so it gets the same closing
        // treatment as everything else. Closing twice is free.
        close_console_episode(&console);
        // Waited for, not just signalled. `run_on_runtime` returns into `shutdown_background`
        // immediately after this, which drops every task where it stands: a task parked at an
        // await is never polled again, so it runs neither `kill_child_tree` nor
        // `finish_background_task` and the cancel achieves exactly nothing. Bounded, because
        // cancelling only asks -- a task that does not answer must not hold the terminal, and its
        // row is swept to `interrupted` on the next open, which is what that sweep is for.
        if tokio::time::timeout(BACKGROUND_EXIT_GRACE, agent.background_tasks().wait_all())
            .await
            .is_err()
        {
            tracing::warn!(
                "background task(s) still running after {}s; leaving them to the next session open",
                BACKGROUND_EXIT_GRACE.as_secs()
            );
        }
    }

    // Detach before dropping the agent, so the manager stops holding the registry whose tools hold
    // the manager. Shutdown no longer depends on this -- it takes `&self` -- but every other
    // surface (ACP `session/close`, `meka serve`) pairs `attach_registry` with `detach_registry`,
    // and leaving the REPL as the one that attaches and never detaches means the cycle outlives the
    // session it belongs to.
    if let Some(manager) = &mcp_manager {
        manager.detach_registry(agent.tool_registry()).await;
    }
    drop(agent);

    if let Some(manager) = mcp_manager {
        shutdown_mcp_manager(manager).await;
    }

    Ok(())
}

/// Close the MCP servers on the way out.
///
/// [`mcp::McpClientManager::shutdown`] takes `&self`, so this runs regardless of how many owners
/// the `Arc` still has. It used to `try_unwrap` first and warn when that failed, which it always
/// did: the manager holds the registries it serves, and those registries hold six tools that each
/// hold the manager back, so the graceful path was unreachable and every run ended by leaving its
/// stdio children to rmcp's drop guard.
async fn shutdown_mcp_manager(manager: Arc<mcp::McpClientManager>) {
    manager.shutdown_within(mcp::SHUTDOWN_BUDGET).await;
}

/// How long leaving the REPL waits for its cancelled background tasks to unwind.
///
/// Long enough for the work a cancelled task actually has left -- signal its process group, write
/// one row -- and short enough that a task ignoring its token cannot hold the terminal. Whatever
/// overruns it is swept to `interrupted` when the session is next opened.
const BACKGROUND_EXIT_GRACE: std::time::Duration = std::time::Duration::from_secs(5);

/// Read a secret from stdin when its `--…-stdin` flag was passed, and return `None` when it was
/// not.
///
/// stdin is read to end, so exactly one secret can be taken per command; the flags that reach here
/// conflict with each other in clap for that reason. An empty stream is an error rather than
/// `None`: a caller that asked for a token and got nothing should hear it here, not from the server
/// later.
fn read_secret_from_stdin(from_stdin: bool, label: &str) -> anyhow::Result<Option<String>> {
    if !from_stdin {
        return Ok(None);
    }
    use std::io::Read as _;
    let mut buffer = String::new();
    std::io::stdin().read_to_string(&mut buffer)?;
    let secret = buffer.trim().to_string();
    if secret.is_empty() {
        anyhow::bail!("no {} was read from stdin", label);
    }
    Ok(Some(secret))
}

async fn run_mcp_subcommand(
    session_manager: &SessionManager,
    action: &cli::McpAction,
    cli_args: &cli::Cli,
) -> anyhow::Result<()> {
    let config = ResolvedConfig::from_cli(cli_args);
    // `validate()` never runs here, so an unparseable config has to be handled per action. The four
    // that edit `config.toml` through `toml_edit` never read `config.mcp_servers` and are how the
    // file gets repaired, so they run on a broken one; the rest would answer out of an empty server
    // list and state it as fact ("No MCP servers configured.", "no MCP server named 'x'").
    if matches!(
        action,
        cli::McpAction::Add { .. }
            | cli::McpAction::Remove { .. }
            | cli::McpAction::Enable { .. }
            | cli::McpAction::Disable { .. }
    ) {
        config.warn_if_config_unreadable();
    } else {
        config.require_readable_config()?;
    }
    let token_store = session_manager.token_store();
    match action {
        cli::McpAction::List => mcp::cli::run_list(&config.mcp_servers, None, &token_store).await?,
        cli::McpAction::Get { name } => {
            mcp::cli::run_get(&config.mcp_servers, name, &token_store).await?
        }
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
        cli::McpAction::Login {
            name,
            auth_token_stdin,
            client_secret_stdin,
        } => {
            use crate::session::McpCredentialKind;

            // Clap rejects both flags at once, so at most one of these reads stdin.
            let stored = match (
                read_secret_from_stdin(*auth_token_stdin, "auth token")?,
                read_secret_from_stdin(*client_secret_stdin, "client secret")?,
            ) {
                (Some(token), _) => Some((McpCredentialKind::Bearer, token)),
                (None, Some(secret)) => Some((McpCredentialKind::ClientSecret, secret)),
                (None, None) => None,
            };

            match stored {
                Some((kind, secret)) => {
                    mcp::cli::run_store_secret(
                        &config.mcp_servers,
                        &token_store,
                        name,
                        kind,
                        &secret,
                    )
                    .await?
                }
                None => mcp::cli::run_login(&config.mcp_servers, &token_store, name).await?,
            }
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
            auth_token_stdin,
            client_id,
            client_secret_stdin,
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
            required,
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
                    // Read here rather than in `run_add` because stdin is a process-wide resource
                    // and this is the layer that owns it. Clap has already rejected both flags at
                    // once, so at most one of these two reads the stream.
                    auth_token: read_secret_from_stdin(*auth_token_stdin, "auth token")?,
                    client_id: client_id.clone(),
                    client_secret: read_secret_from_stdin(*client_secret_stdin, "client secret")?,
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
                    required: *required,
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
            // The table's permission and status columns come from config, so rendering it off
            // defaults would misreport every tool the user has overridden.
            config.require_readable_config()?;
            let filter = crate::tools::BuiltinToolFilter::from_config(
                config.builtin_allowed_tools.clone(),
                config.builtin_disabled_tools.clone(),
                config.builtin_tool_permissions.clone(),
            );
            crate::tools::warn_on_stale_builtin_tool_config(&filter);

            // Build with no filter so the catalogue carries every tool's hardcoded level; overlay
            // the real filter for status/source.
            let session_manager = SessionManager::open(
                None,
                &session::migrations::Context::adopting(
                    config.configured_default_profile.as_deref(),
                ),
            )
            .await?;
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
                // `meka tools list` only prints the catalogue, so neither store's metadata is read
                // and the filesystem walk is skipped. The switches still have to be honoured:
                // this listing exists to show what a real session would have.
                if config.skills_enabled {
                    crate::skills::SkillCache::for_root(None)
                } else {
                    crate::skills::SkillCache::disabled()
                },
                config.skills_agent_managed,
                if config.memory_enabled {
                    crate::memory::MemoryStore::detached()
                } else {
                    crate::memory::MemoryStore::disabled()
                },
                crate::tools::BuiltinToolFilter::default(),
                std::sync::Arc::new(std::sync::RwLock::new(std::path::PathBuf::from("."))),
                std::sync::Arc::new(std::sync::RwLock::new(Vec::new())),
                std::sync::Arc::new(crate::frontend::SilentFrontend),
                config.schedule.clone(),
                (
                    config.background.clone(),
                    crate::background::BackgroundTasks::default(),
                ),
            )?;

            // The `context_*` family is registered by the `Agent`, not by `build_default`, because
            // its counters belong to a live conversation. This listing is what the docs point
            // people at to discover tool names, so it registers them here with idle handles: the
            // names, levels and descriptions are what a real session would show.
            reference.register_context_tools(
                crate::tools::context::ContextGauge {
                    used: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
                    overhead: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
                    window: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
                    compact_at_percent: None,
                },
                std::sync::Arc::new(std::sync::Mutex::new(None)),
                config.compact_checkpoint,
                SessionManager::open(
                    None,
                    &session::migrations::Context::adopting(
                        config.configured_default_profile.as_deref(),
                    ),
                )
                .await?,
                std::sync::Arc::new(tokio::sync::RwLock::new(None)),
            );

            let mut catalogue = reference.tool_catalogue();
            // The one family the registry above cannot hold: its tools carry an `Arc<dyn
            // Provider>`, so building them would make naming meka's own tools require a resolved
            // credential. Their definitions are constants, so the listing reads those instead of
            // describing the family in prose the table has no way to keep honest.
            let agent_family = crate::tools::subagent::agent_tool_catalogue();
            let agent_family_names: std::collections::HashSet<String> = agent_family
                .iter()
                .map(|(definition, _)| definition.name.clone())
                .collect();
            // Both conditions that remove the whole family, asked the way `assemble_agent` asks
            // them. A per-name `[tools]` entry is a separate question, applied below alongside
            // every other tool's.
            let agent_family_registered =
                crate::tools::subagent::agent_tools_registered(&filter, config.subagent_max_depth);
            catalogue.extend(agent_family.into_iter().map(|(definition, required)| {
                // Never deferred: `register_subagent_tools` registers all four eagerly and calls
                // no `mark_deferred`.
                (definition.name, definition.description, required, false)
            }));
            catalogue.sort_by(|left, right| left.0.cmp(&right.0));

            // `format_columns`, like every other listing meka prints. The fixed `{:<20}` this used
            // to hand-roll silently ran its columns together for any name longer than the width,
            // and a namespaced MCP tool (`mcp__mekabridge__send_file`) is 26
            // characters.
            let rows: Vec<Vec<String>> = catalogue
                .iter()
                .map(|(name, description, required, is_deferred)| {
                    let override_entry = filter.permission_overrides.get(name);
                    let effective = override_entry.copied().unwrap_or(*required);
                    let admitted = filter.admits(name)
                        && (agent_family_registered || !agent_family_names.contains(name));
                    vec![
                        name.clone(),
                        effective.to_string(),
                        if override_entry.is_some() {
                            "override".to_string()
                        } else {
                            "builtin".to_string()
                        },
                        if admitted {
                            if *is_deferred { "deferred" } else { "enabled" }
                        } else {
                            "disabled"
                        }
                        .to_string(),
                        description
                            .lines()
                            .next()
                            .unwrap_or("")
                            .chars()
                            .take(60)
                            .collect::<String>(),
                    ]
                })
                .collect();
            print!(
                "{}",
                render::format_columns(
                    &["Name", "Required", "Source", "Visibility", "Description"],
                    &rows
                )
            );
        }
    }
    Ok(())
}

/// `meka instructions`: answer "what is the model actually being told, and why".
///
/// With four tiers feeding one value and a conventional path that appears in no config file, that
/// question is otherwise only answerable by reading the source. Not async: every tier is either a
/// process environment read or a small synchronous file read.
fn run_instructions_subcommand(action: &cli::InstructionsAction) -> anyhow::Result<()> {
    match action {
        cli::InstructionsAction::Show => {
            // `--instructions` belongs to a run, not to this query, so it is deliberately not
            // consulted here; `None` resolves the persistent tiers only.
            match config::resolve_instructions_for_display()? {
                Some(found) => {
                    // Source to stderr, text to stdout: the text is the data you asked for, so
                    // `2>/dev/null` leaves something pipeable.
                    eprintln!("Source: {}", found.source);
                    eprintln!();
                    println!("{}", found.text);
                }
                None => eprintln!(
                    "No instructions configured. Write them to {} (or split them across {}).",
                    display_path(instructions::instructions_file()),
                    display_path(instructions::instructions_dir()),
                ),
            }
        }
        cli::InstructionsAction::Path => {
            for path in [
                instructions::instructions_dir(),
                instructions::instructions_file(),
            ]
            .into_iter()
            .flatten()
            {
                let state = if path.exists() { "present" } else { "absent" };
                println!("{}\t{}", path.display(), state);
            }
        }
    }
    Ok(())
}

fn display_path(path: Option<std::path::PathBuf>) -> String {
    path.map(|path| path.display().to_string())
        .unwrap_or_else(|| "<no config directory>".to_string())
}

async fn run_memory_subcommand(
    session_manager: &SessionManager,
    action: &cli::MemoryAction,
) -> anyhow::Result<()> {
    // Deliberately not gated on `[memory] enabled`. That switch decides whether an *agent* keeps
    // memories; it is not a reason to refuse to show, back up or delete what is already stored.
    let store = session_manager.memory_store(true);
    match action {
        cli::MemoryAction::List => {
            memory::cli::run_list(&store, memory::cli::ListDetail::WithDistribution).await?
        }
        cli::MemoryAction::Get { name } => memory::cli::run_get(&store, name).await?,
        cli::MemoryAction::Show { name } => memory::cli::run_show(&store, name).await?,
        cli::MemoryAction::Add {
            name,
            description,
            priority,
            tags,
            body,
            from_file,
            force,
        } => {
            memory::cli::run_add(&store, memory::cli::AddArgs {
                name,
                description,
                priority: *priority,
                tags,
                body: body.as_deref(),
                from_file: from_file.as_deref(),
                force: *force,
            })
            .await?
        }
        cli::MemoryAction::Edit { name } => memory::cli::run_edit(&store, name).await?,
        cli::MemoryAction::Remove { name } => memory::cli::run_remove(&store, name).await?,
        cli::MemoryAction::Verify { rebuild } => memory::cli::run_verify(&store, *rebuild).await?,
        cli::MemoryAction::Export { dir } => {
            let directory = dir.clone().unwrap_or_else(memory::cli::default_export_dir);
            memory::cli::run_export(&store, &directory).await?
        }
    }
    Ok(())
}

async fn run_skill_subcommand(
    action: &cli::SkillAction,
    cli_args: &cli::Cli,
) -> anyhow::Result<()> {
    // Resolved rather than defaulted: `[skills] extra_paths` decides which roots these read, and a
    // handler that ignored it would report a store the running agent does not have.
    let config = ResolvedConfig::from_cli(cli_args);
    config.require_readable_config()?;
    let roots = config.skill_roots();
    match action {
        cli::SkillAction::List { paths } => skills::cli::run_list(&roots, *paths).await?,
        cli::SkillAction::Get { name } => skills::cli::run_get(name, &roots).await?,
        cli::SkillAction::Show { name } => skills::cli::run_show(name, &roots).await?,
        cli::SkillAction::Add {
            name,
            description,
            priority,
            metadata,
            from_file,
            force,
            edit,
        } => {
            skills::cli::run_add(
                skills::cli::AddArgs {
                    name,
                    description: description.as_deref(),
                    priority: *priority,
                    metadata,
                    from_file: from_file.as_deref(),
                    force: *force,
                    edit: *edit,
                },
                &roots,
            )
            .await?
        }
        cli::SkillAction::Remove { name } => skills::cli::run_remove(name, &roots).await?,
    }
    Ok(())
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
                eprintln!("No input history.");
            } else {
                for entry in entries {
                    println!("{}", entry);
                }
            }
        }
        cli::HistoryAction::Clear => {
            let removed = history.clear_all()?;
            let noun = if removed == 1 { "entry" } else { "entries" };
            tracing::info!("cleared {} input history {}", removed, noun);
        }
    }
    Ok(())
}

/// Translate the live-REPL display config into the options that [`render::render_message_history`]
/// consumes. Keeps the spacing / styling rules between live output and history rendering in sync
/// from a single source of truth.
fn history_render_options(config: &ResolvedConfig) -> render::HistoryRenderOptions {
    render::HistoryRenderOptions {
        render_mode: config.render_mode,
        show_thinking: config.thinking_show_content,
        tool_params: config.tool_params,
        input_style: config.input_style,
        newline_before_prompt: config.newline_before_prompt,
        newline_after_prompt: config.newline_after_prompt,
    }
}

fn reprint_last_message(messages: &[provider::Message], render_mode: render::RenderMode) -> bool {
    let Some(last) = messages.last() else {
        return false;
    };

    let text = match last.role {
        provider::Role::Assistant => {
            let text = last.text_content();
            if text.is_empty() {
                return false;
            }
            text
        }
        provider::Role::User => {
            let raw = last.text_content();
            let stripped = session::strip_context_tags(&raw);
            if stripped.is_empty() {
                return false;
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
    true
}

/// Borrow the shared console for one synchronous run of writes.
///
/// A poisoned lock is recovered from rather than propagated: the console holds the terminal's
/// layout, and losing that is a worse outcome than continuing from a state one panicking writer may
/// have left mid-transition. No `.await` may appear inside `act` -- `clippy::await_holding_lock` is
/// deny-level and would catch it, but the reason is that the agent's frontend writes through the
/// same lock.
fn with_console<T>(
    console: &std::sync::Mutex<console::Console>,
    act: impl FnOnce(&mut console::Console) -> T,
) -> T {
    act(&mut console
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()))
}

fn close_console_episode(console: &std::sync::Mutex<console::Console>) {
    with_console(console, |console| console.close_episode());
}

/// Move a lock into the slot the agent and its host share, replacing whatever was there.
///
/// The replacement is what `/fork` depends on: the new lock is already held by the time this is
/// called, and the old guard is dropped only once the new one is in place, so the session is never
/// momentarily unheld. Passing `None` releases outright, which is how the REPL lets go on the way
/// out.
///
/// A poisoned mutex is recovered from rather than propagated. The slot holds one value and every
/// writer replaces it whole, so a panic cannot have left it half-updated -- and refusing to release
/// a lock because some unrelated thread panicked would be the worse failure.
fn hold_session_lock(slot: &session::SessionLockSlot, lock: Option<session::FileLock>) {
    match slot.lock() {
        Ok(mut held) => *held = lock,
        Err(poisoned) => *poisoned.into_inner() = lock,
    }
}

/// What `meka -c` / `-r` resolved to, and the settings the resumed session brings with it.
struct ResumedSession {
    session_id: Option<uuid::Uuid>,
    messages: conversation::Conversation,
    lock: Option<session::FileLock>,
    /// The level this run starts at: the session's recorded one, unless `--permission` asked for
    /// something else, and the configured default for a run that resumed nothing.
    permission: crate::permission::Permission,
    /// The profile `--provider` asked this session to move to, still uncommitted. Carried out
    /// rather than written here because the row must not move until the profile is known to
    /// produce a provider, and that needs a registry this runs before.
    repin: Option<String>,
    /// The directory the session recorded, which is where it reopens.
    ///
    /// Carried out rather than applied here because the caller owns the
    /// [`crate::workspace::SharedCwd`] and the fallback: `None` here means the row carried no
    /// directory (an imported archive may omit it) and the launch directory decides.
    ///
    /// **A resume never writes this column back.** The recorded directory is the session's, and at
    /// `workspace` it is also the writable boundary; correcting it to whatever shell happened to
    /// start the process would silently widen that boundary to, say, `$HOME`, and would let an
    /// unattended `--oneshot -c` from a unit at `/` repoint a session and every gate it holds.
    /// `meka serve` already reads the column this way ([`crate::server::reattach`]); ACP is the
    /// deliberate exception, because its client passes an authoritative project root per request.
    cwd: Option<std::path::PathBuf>,
}

async fn resolve_session_resume(
    session_manager: &SessionManager,
    config: &ResolvedConfig,
    console: &std::sync::Mutex<console::Console>,
) -> anyhow::Result<ResumedSession> {
    let fresh = || ResumedSession {
        session_id: None,
        messages: conversation::Conversation::new(),
        lock: None,
        permission: config.permission,
        // A run that resumed nothing has no row to repin: the flags become this session's binding
        // when `resolve_session_provider` builds it, and are recorded when the row is created.
        repin: None,
        // Nor a directory to reopen: a fresh session starts where the shell is, and records that.
        cwd: None,
    };
    let resolved = match &config.session_resume {
        None => return Ok(fresh()),
        // `--continue` on a store with no sessions yet is not an error: there is simply nothing to
        // pick up, so the run starts fresh.
        Some(crate::config::SessionResume::Last) => session_manager.last_session_id().await?,
        Some(crate::config::SessionResume::Id(value)) => {
            Some(resolve_session_id(session_manager, value).await?)
        }
    };
    let Some(id) = resolved else {
        return Ok(fresh());
    };

    let lock = session_manager.lock_session(id)?;
    // `--provider` on a resume repins the session rather than applying for this run alone. A
    // per-run override would leave the row disagreeing with the conversation it describes, and the
    // next resume would silently move back; rewriting it keeps the row the answer to "what does
    // this session run on".
    //
    // Only computed here. `apply_session_repin` commits it, once the profile is known to produce a
    // provider.
    let repin = if let Some(requested) = &config.requested_profile {
        if !config.providers.contains_key(requested) {
            anyhow::bail!(
                "provider profile `{}` is not configured; `meka provider list` shows the \
                 configured ones",
                requested
            );
        }
        Some(requested.clone())
    } else {
        None
    };

    // The level the session recorded, unless this run asked for a different one. Every other
    // surface already resolves permission this way -- ACP, the HTTP API, the scheduler's fire door
    // and `meka schedule` all read the row through `parse_recorded_permission` -- and both CLI
    // hosts ignored it, this function being what the REPL and `--oneshot` share, so a session
    // created at `unrestricted` came back at the config default while its row still claimed
    // otherwise. `--oneshot` is the worse half: a scripted run whose level silently differs from
    // the one the session was created with has nobody watching it.
    let recorded = session_manager.session_info(id).await?;
    let permission = config.requested_permission.or_else(|| {
        crate::permission::parse_recorded_permission(
            recorded
                .as_ref()
                .and_then(|info| info.permission.as_deref()),
            &format_args!("session {}", id),
        )
        .filter(|level| {
            // A level the operator has since removed from `[permissions].enabled` is not one this
            // run may take, so the session drops to the configured default rather than being
            // granted authority the configuration withdrew.
            if config.enabled_permissions.is_enabled(*level) {
                return true;
            }
            tracing::warn!(
                "session {} records permission '{}', which is no longer in \
                 [permissions].enabled; starting at '{}'",
                id,
                level,
                config.permission
            );
            false
        })
    });
    // `--permission` rewrites the row for the reason `/permission` already does: a scheduled gate
    // is re-checked against it, and leaving it stale means another process acts on a level the user
    // has moved away from.
    if let Some(requested) = config.requested_permission
        && recorded
            .as_ref()
            .and_then(|info| info.permission.as_deref())
            != Some(requested.to_string().as_str())
        && let Err(error) = session_manager
            .update_session_permission(id, &requested.to_string())
            .await
    {
        tracing::warn!(
            "could not record permission `{}` on session {}: {}. Another meka process may still \
             act on this session's previous level",
            requested,
            id,
            error
        );
    }

    with_console(console, |console| {
        console.session_id("Continuing session", &id.to_string())
    });
    let messages = load_session_messages(session_manager, id).await?;
    Ok(ResumedSession {
        session_id: Some(id),
        messages,
        lock: Some(lock),
        permission: permission.unwrap_or(config.permission),
        repin,
        // Read off the row already loaded above, not fetched again.
        cwd: recorded.and_then(|info| info.cwd),
    })
}

/// Record where `/cd` moved the session, so the row keeps saying where the session is.
///
/// A free function rather than a block inside the REPL's event loop because the loop is not
/// reachable from a test: reedline needs a terminal, and a piped stdin fails the read outright, so
/// everything inline there is verified only by running meka by hand.
///
/// Never returns an error. The directory has already moved in this process, so failing the REPL
/// over a bookkeeping write would be worse than the stale row; the `warn!` is loud because the
/// consequence is real -- this session's next resume, and any scheduled gate it holds, still read
/// the old directory.
async fn record_session_cwd(
    session_manager: &SessionManager,
    session_id: Option<uuid::Uuid>,
    path: &std::path::Path,
) {
    // No row yet: the directory the first turn creates the session with is read from the same cell
    // `/cd` has already written, so there is nothing to correct.
    let Some(id) = session_id else {
        return;
    };
    if let Err(error) = session_manager.update_session_cwd(id, path).await {
        tracing::warn!(
            "could not record working directory `{}` on session {}: {}. A resume and any scheduled \
             gate will still use this session's previous directory",
            path.display(),
            id,
            error
        );
    }
}

/// The directory a run opens in: the one its session recorded, or where meka was launched.
///
/// A session's directory is its own, so a resume reopens it rather than adopting whatever shell
/// started the process. At `workspace` that directory is also the writable boundary, and taking the
/// shell's instead would silently widen it -- resume a project session from `$HOME` and the whole
/// home directory becomes writable, with a scheduled job able to fire before the user can react.
/// `meka serve` already reads the column this way (`crate::server::reattach`); the REPL and
/// `--oneshot` were the two that did not.
///
/// The launch directory is the fallback for two cases, both of which have to keep the run going:
/// a row that carries no directory (`meka session import` stores an archive's value verbatim, and
/// an archive may omit it) and one naming a directory that has since been removed.
fn resume_working_directory(
    recorded: Option<std::path::PathBuf>,
    launch_cwd: &std::path::Path,
    session_id: Option<uuid::Uuid>,
) -> std::path::PathBuf {
    let Some(recorded) = recorded else {
        return launch_cwd.to_path_buf();
    };
    if recorded.is_dir() {
        return recorded;
    }
    // Warn rather than fail: the conversation is still worth resuming, and the alternative is
    // refusing to open a session because a directory moved. Matches how a scheduled gate reports
    // the same loss (`crate::schedule`'s `run_shell_probe`).
    tracing::warn!(
        "session {} records working directory '{}', which no longer exists; opening in '{}' \
         instead",
        session_id.map_or_else(|| "?".to_string(), |id| id.to_string()),
        recorded.display(),
        launch_cwd.display()
    );
    launch_cwd.to_path_buf()
}

/// Commit a `--provider` repin, once the profile it names is known to produce a provider.
///
/// The row moves last, deliberately. Writing it first and only then discovering that the profile
/// has no stored credential leaves the session pinned to something that cannot run, and the binding
/// it had is nowhere in the output, so the next plain resume fails the same way with nothing left
/// to say what it used to be. Both other surfaces that offer this switch build the provider before
/// they write: `PATCH /v1/sessions/{id}` and ACP's `session/set_config_option`.
async fn apply_session_repin(
    session_manager: &SessionManager,
    providers: &provider::ProviderRegistry,
    session_id: uuid::Uuid,
    binding: String,
) -> anyhow::Result<()> {
    let resolved = resolved_binding(providers, binding).await?;
    if !session_manager
        .set_recorded_provider(session_id, &resolved.binding)
        .await?
    {
        anyhow::bail!("session not found: {}", session_id);
    }
    tracing::info!(
        "repinned session {} to provider profile `{}`",
        session_id,
        resolved.binding
    );
    Ok(())
}

/// Whether a string is plausibly a session id rather than a prompt: hex digits and hyphens only,
/// and long enough to be a useful UUID prefix.
///
/// Only used to catch the old `meka -c <uuid>` spelling and point at `-r`. Deliberately
/// conservative: an English prompt of eight-plus characters with no spaces that happens to be pure
/// hex (`deadbeef`) would be caught, but that is a far better trade than silently continuing the
/// wrong session for someone following older docs.
fn looks_like_session_id(value: &str) -> bool {
    value.len() >= 8 && value.chars().all(|c| c.is_ascii_hexdigit() || c == '-')
}

/// Resolve `meka -r <value>` to a single session UUID. Tries a
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
    // Retire whatever the last process left running before hydrating anything else; see
    // `crate::background::claim_session`.
    crate::background::claim_session(session_manager, session_id).await;

    // Hydrate the event log directly: `load_events` decodes each stored row back into the `Event`
    // that wrote it, so resume rebuilds the log the last process ended with rather than a flattened
    // approximation of it.
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

    // Materializing the log also replaces images whose bytes contradict their declared media type.
    // Providers sniff and reject those with a 400, and since the block is already in the log that
    // 400 would repeat on every request, leaving the session unusable. A stored session carrying
    // such a block heals on the way in, for free; all that is left to do is say so.
    let replaced = log.invalid_images_replaced();
    if replaced > 0 {
        tracing::warn!(
            "replaced {} image(s) whose bytes did not match their declared media type",
            replaced,
        );
    }

    Ok(log)
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn store() -> SessionManager {
        // `:memory:`, spelled out. `None` is not "no path": it is *the default path*, so this
        // helper was creating sessions in the developer's own `~/.local/share/meka/meka.db` on
        // every `cargo test`, and migrating and backing it up on the way in.
        SessionManager::open(
            Some(std::path::Path::new(":memory:")),
            &session::migrations::Context::default(),
        )
        .await
        .expect("an in-memory store opens")
    }

    /// `/cd` reaches the row, which is what makes the recorded directory mean "where the session
    /// is" rather than "where it was created". Nothing else covers this: the REPL loop that sends
    /// the event needs a terminal, so a test cannot drive `/cd` itself.
    #[tokio::test]
    async fn a_cd_records_the_directory_on_the_session_row() {
        let manager = store().await;
        let temp = tempfile::tempdir().expect("tempdir");
        let moved = crate::workspace::canonical_for_test(temp.path());
        let id = manager
            .create_session(
                Some(std::path::PathBuf::from("/somewhere/else")),
                "p".to_string(),
            )
            .await
            .expect("create");

        record_session_cwd(&manager, Some(id), &moved).await;

        let recorded = manager
            .session_info(id)
            .await
            .expect("read the row")
            .and_then(|info| info.cwd);
        assert_eq!(
            recorded,
            Some(moved),
            "the row must say where `/cd` moved the session",
        );
    }

    /// Before the first turn there is no row to correct: the directory the creation snapshot reads
    /// is the cell `/cd` has already written. So this is a no-op, and in particular it must not
    /// reach for some other session's row -- a REPL sharing a store with a `meka serve` has plenty
    /// to choose from.
    #[tokio::test]
    async fn a_cd_before_the_first_turn_writes_nobody_elses_row() {
        let manager = store().await;
        let temp = tempfile::tempdir().expect("tempdir");
        let bystander = manager
            .create_session(
                Some(std::path::PathBuf::from("/its/own/place")),
                "p".to_string(),
            )
            .await
            .expect("create");

        record_session_cwd(&manager, None, temp.path()).await;

        let untouched = manager
            .session_info(bystander)
            .await
            .expect("read the row")
            .and_then(|info| info.cwd);
        assert_eq!(
            untouched,
            Some(std::path::PathBuf::from("/its/own/place")),
            "a `/cd` with no session of its own must leave every other row alone",
        );
    }

    /// A resumed session opens where it was recorded. At `workspace` that directory is also the
    /// writable boundary, so adopting the shell's would widen it behind the user's back.
    #[test]
    fn a_resume_prefers_the_recorded_directory_over_the_launch_directory() {
        let temp = tempfile::tempdir().expect("tempdir");
        let recorded = temp.path().join("project");
        let launch = temp.path().join("elsewhere");
        std::fs::create_dir_all(&recorded).expect("recorded dir");
        std::fs::create_dir_all(&launch).expect("launch dir");

        assert_eq!(
            resume_working_directory(Some(recorded.clone()), &launch, None),
            recorded,
        );
    }

    /// The two cases that have to keep the run going: a row carrying no directory (an imported
    /// archive may omit it) and one naming a directory that has since been removed.
    #[test]
    fn a_resume_falls_back_to_the_launch_directory_when_the_recording_cannot_serve() {
        let temp = tempfile::tempdir().expect("tempdir");
        let launch = temp.path().join("elsewhere");
        std::fs::create_dir_all(&launch).expect("launch dir");

        assert_eq!(resume_working_directory(None, &launch, None), launch);
        assert_eq!(
            resume_working_directory(Some(temp.path().join("gone")), &launch, None),
            launch,
        );
        // A file is not a directory to open in either, and `is_dir` is what separates them.
        let file = temp.path().join("a-file");
        std::fs::write(&file, b"x").expect("write file");
        assert_eq!(resume_working_directory(Some(file), &launch, None), launch);
    }

    /// The regression the whole feature exists for, at the one door every turn-running site goes
    /// through: a session that named a profile keeps it, whatever the process default now is.
    #[tokio::test]
    async fn a_recorded_binding_beats_the_process_default() {
        let manager = store().await;
        let id = manager
            .create_session(None, "openaiprof".to_string())
            .await
            .expect("create");

        let resolved = resolve_session_provider(&manager, Ok("claudeprof"), Some(id))
            .await
            .expect("resolves");

        assert_eq!(resolved, "openaiprof");
    }

    /// A session that does not exist yet is the only case the configured default answers.
    #[tokio::test]
    async fn a_session_that_does_not_exist_yet_takes_the_default() {
        let manager = store().await;

        let resolved = resolve_session_provider(&manager, Ok("claudeprof"), None)
            .await
            .expect("resolves");

        assert_eq!(resolved, "claudeprof");
    }

    /// With nothing configured there is no profile to fall back to, and inventing one would be the
    /// silent redirection this door exists to prevent.
    #[tokio::test]
    async fn no_configured_profile_is_an_error_rather_than_an_empty_one() {
        let manager = store().await;

        let error = resolve_session_provider(
            &manager,
            Err("no provider profiles configured. Run `meka provider add <name>`."),
            None,
        )
        .await
        .expect_err("nothing to resolve to");

        assert!(
            error.to_string().contains("meka provider add"),
            "the refusal should say how to fix it: {error}"
        );
    }

    /// The reason travels: `validate()` no longer raises an ambiguous default for a resume, so a
    /// resume that *does* fall through to needing one has to carry the message that says what to
    /// do rather than a generic "nothing configured".
    #[tokio::test]
    async fn a_resume_that_needs_a_default_reports_why_there_is_none() {
        let manager = store().await;
        let ambiguous = "multiple provider profiles configured (personal, side); run \
                         `meka provider use <name>` to pick a default, or pass `--provider <name>`.";

        // `None` session id: `-c` on a store with nothing to resume lands here.
        let error = resolve_session_provider(&manager, Err(ambiguous), None)
            .await
            .expect_err("no default to fall back to");

        assert_eq!(error.to_string(), ambiguous);
    }

    /// A REPL sweep is evaluated against the level the session is at now, not the one it launched
    /// with.
    ///
    /// This is the wiring half of the scheduled-gate fix, and it is the half that was wrong: the
    /// refusal logic in `schedule.rs` is well covered at both `read` and `unrestricted`, but every
    /// one of those tests *supplies* the host permission. Nothing checked that the REPL
    /// supplies a live one, so the gate compared against a startup snapshot that `Shift+Tab`
    /// never touches.
    #[test]
    fn a_repl_sweep_reads_the_permission_the_session_is_at_now() {
        let configured = crate::config::ResolvedScheduleConfig {
            gate_tools: None,
            claim_lease: std::time::Duration::from_secs(3600),
            enabled_permissions: crate::permission::EnabledPermissions::DEFAULT,
            enabled: true,
            host_permission: crate::permission::Permission::Read,
            poll_interval: std::time::Duration::from_secs(10),
            missed_grace: std::time::Duration::from_secs(60),
            gate_timeout: std::time::Duration::from_secs(30),
            max_jobs: 50,
            max_consecutive_fires: 5,
        };
        let live = SharedPermission::new(
            crate::permission::Permission::Read,
            crate::permission::EnabledPermissions::from_modes([
                crate::permission::Permission::Read,
                crate::permission::Permission::Unrestricted,
            ])
            .expect("a non-empty mode set"),
        );

        // Cycling up is what lets a gate authored now ever fire.
        live.try_set(crate::permission::Permission::Unrestricted)
            .expect("write is enabled");
        assert_eq!(
            schedule_config_at_live_permission(&configured, &live).host_permission,
            crate::permission::Permission::Unrestricted,
            "a gate authored after cycling up to write would be refused forever"
        );

        // And cycling down is what withdraws one already written, which is the security half.
        live.try_set(crate::permission::Permission::Read)
            .expect("read is enabled");
        assert_eq!(
            schedule_config_at_live_permission(&configured, &live).host_permission,
            crate::permission::Permission::Read,
            "an unattended shell command kept running after the authority behind it was withdrawn"
        );

        // Everything else is carried through untouched; only the permission is substituted.
        let derived = schedule_config_at_live_permission(&configured, &live);
        assert_eq!(derived.poll_interval, configured.poll_interval);
        assert_eq!(
            derived.max_consecutive_fires,
            configured.max_consecutive_fires
        );
        assert_eq!(derived.enabled, configured.enabled);
    }

    /// Zero days means "not updated since this instant", i.e. everything. Easy to type when you
    /// meant "today's", and unrecoverable, so it is refused rather than run.
    #[tokio::test]
    async fn test_delete_older_than_zero_days_is_refused() {
        let manager =
            SessionManager::open(Some(std::path::Path::new(":memory:")), &Default::default())
                .await
                .expect("in-memory db");
        let error = crate::session::cli::delete_sessions(&manager, &[], false, Some(0))
            .await
            .expect_err("zero must be refused");
        assert!(error.to_string().contains("--all"), "{error}");
    }

    /// The flag has to reach `delete_expired_sessions(days)` and nothing else: routing it to
    /// `delete_all_sessions` would pass every error-path test in this file while wiping the DB.
    #[tokio::test]
    async fn test_delete_older_than_days_deletes_only_the_old() {
        let manager =
            SessionManager::open(Some(std::path::Path::new(":memory:")), &Default::default())
                .await
                .expect("in-memory db");
        let old = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create old");
        let recent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create recent");
        let backdated = (chrono::Utc::now() - chrono::TimeDelta::days(100)).to_rfc3339();
        manager
            .set_session_updated_at_for_test(old, &backdated)
            .await
            .expect("backdate");

        crate::session::cli::delete_sessions(&manager, &[], false, Some(30))
            .await
            .expect("sweep");

        assert!(!manager.session_exists(old).await.expect("exists"));
        assert!(manager.session_exists(recent).await.expect("exists"));
    }

    /// No selector at all should say what the options are, not silently do nothing.
    #[tokio::test]
    async fn test_delete_with_no_selector_explains_itself() {
        let manager =
            SessionManager::open(Some(std::path::Path::new(":memory:")), &Default::default())
                .await
                .expect("in-memory db");
        let error = crate::session::cli::delete_sessions(&manager, &[], false, None)
            .await
            .expect_err("no selector must be an error");
        let text = error.to_string();
        assert!(text.contains("--older-than-days"), "{text}");
        assert!(text.contains("--all"), "{text}");
    }

    /// Both directives exist to stop a retried MCP connect from repeating itself on the user's
    /// prompt. `RUST_LOG` must still win outright, or there is no way to see them when debugging.
    #[test]
    fn test_log_filter_quiets_rmcp_retry_noise() {
        let filter = build_log_filter(None, "warn").to_string();
        assert!(
            filter.contains("rmcp::transport::worker=off"),
            "the per-attempt transport error must be silenced: {filter}"
        );
        assert!(
            filter.contains("rmcp::transport::common::client_side_sse=error"),
            "the per-reconnect sse warning must be floored: {filter}"
        );

        let overridden = build_log_filter(Some("rmcp=debug"), "warn").to_string();
        assert!(
            !overridden.contains("rmcp::transport::worker=off"),
            "RUST_LOG must replace the defaults wholesale: {overridden}"
        );
    }
    #[test]
    fn test_parents_first_order_orders_parents_before_children() {
        // Given out of order (child, root, middle), each node must land after its parent.
        let nodes = vec![
            ("c".to_string(), Some("b".to_string())),
            ("a".to_string(), None),
            ("b".to_string(), Some("a".to_string())),
        ];
        let order = crate::session::cli::parents_first_order(&nodes).expect("order");
        let position = |id: &str| order.iter().position(|&i| nodes[i].0 == id).unwrap();
        assert!(position("a") < position("b"));
        assert!(position("b") < position("c"));
    }

    #[test]
    fn test_parents_first_order_treats_external_parent_as_root() {
        // A parent absent from the set (e.g. the exported root was itself a sub-agent) is not an
        // error; the node is ordered as a root.
        let nodes = vec![("only".to_string(), Some("outside".to_string()))];
        assert_eq!(
            crate::session::cli::parents_first_order(&nodes).expect("order"),
            vec![0]
        );
    }

    /// `meka -c <uuid>` was the documented way to resume a specific session before `-c` became a
    /// boolean. It now parses as "continue the most recent session, with this id as the prompt",
    /// which is silently the wrong session, so the old spelling has to be caught rather than run.
    #[test]
    fn test_looks_like_session_id_catches_the_old_spelling() {
        assert!(looks_like_session_id(
            "550e8400-e29b-41d4-a716-446655440000"
        ));
        assert!(looks_like_session_id("550e8400"));
        // Real prompts are not mistaken for ids.
        assert!(!looks_like_session_id("fix the bug"));
        assert!(!looks_like_session_id("explain"));
        assert!(!looks_like_session_id("why?"));
        // Too short to be a useful prefix, so treated as a prompt.
        assert!(!looks_like_session_id("550e"));
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
    fn export_without_compaction_renders_plain_turns() {
        let mut log = conversation::Conversation::new();
        log.append(user_msg("hello"));
        log.append(assistant_text("hi there"));
        let markdown = crate::session::cli::format_session_as_markdown(
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
