//! Clap-derived CLI definition. Owns the top-level argument struct, the subcommand enum
//! (`provider`, `session`, `history`, `mcp`, `tools`, `skill`, `account`, `acp`, `serve`), and the
//! small parsers for permission/render-mode/output-format flag values.

use clap::Parser;

use crate::permission::Permission;

// `Mcp { action: McpAction }` is bigger than every other variant because `McpAction::Add` holds
// every CLI flag inline, but the enum is only ever constructed once per process by clap and held on
// the stack of `main`, so the few extra words of padding on the other variants aren't worth the
// indirection cost of boxing.
#[allow(clippy::large_enum_variant)]
#[derive(clap::Subcommand, Debug)]
pub enum Command {
    /// Manage provider profiles
    Provider {
        #[command(subcommand)]
        action: ProviderAction,
    },
    /// Manage stored sessions
    Session {
        #[command(subcommand)]
        action: SessionAction,
    },
    /// View or clear REPL input history
    History {
        #[command(subcommand)]
        action: HistoryAction,
    },
    /// Manage MCP servers
    Mcp {
        #[command(subcommand)]
        action: McpAction,
    },
    /// Inspect built-in tool filters
    Tools {
        #[command(subcommand)]
        action: ToolsAction,
    },
    /// Manage user skills
    Skill {
        #[command(subcommand)]
        action: SkillAction,
    },
    /// Manage the agent's saved memories
    Memory {
        #[command(subcommand)]
        action: MemoryAction,
    },
    /// Show the standing instructions the agent is given
    Instructions {
        #[command(subcommand)]
        action: InstructionsAction,
    },
    /// Inspect and cancel scheduled jobs
    Schedule {
        #[command(subcommand)]
        action: ScheduleAction,
    },
    /// Show account usage, identity, and history for scripting
    Account {
        #[command(subcommand)]
        action: AccountAction,
    },
    /// Run meka as an ACP (Agent Client Protocol) agent over stdio
    ///
    /// Speaks newline-framed JSON-RPC on stdin/stdout so ACP clients (Zed, JetBrains, Neovim, VS
    /// Code via the ACP extension, etc.) can drive meka turns directly. Diagnostic output stays on
    /// stderr; stdout is reserved for the protocol.
    Acp,
    /// Run meka as a long-lived HTTP service
    ///
    /// Exposes the agent over HTTP+JSON for programmatic clients (bots, scripts, web UIs).
    /// See the HTTP API docs for the full spec. Auth, session GC, and SSE streaming are configured
    /// under `[serve]` in config.toml.
    Serve {
        /// Override `[serve].bind` (e.g. `0.0.0.0:8080`)
        #[arg(long)]
        bind: Option<String>,
    },
}

#[derive(clap::Subcommand, Debug)]
pub enum ToolsAction {
    /// List every built-in tool with its effective permission and status
    List,
}

#[derive(clap::Subcommand, Debug)]
pub enum SessionAction {
    /// List past sessions
    List {
        /// Maximum number of sessions to show
        #[arg(short = 'n', long, default_value = "20")]
        limit: u32,
        /// Include sub-agent sessions in the listing
        ///
        /// Hidden by default, so the view stays on conversations you started.
        #[arg(long)]
        include_children: bool,
    },
    /// Export a session as Markdown or JSON
    Export {
        /// Session UUID to export
        session_id: uuid::Uuid,
        /// Output file (`-` for stdout)
        ///
        /// Defaults to `session-<id>.md` for markdown or `session-<id>.json`
        /// for json, written to the current directory.
        #[arg(short, long)]
        output: Option<String>,
        /// Export format: markdown or json
        ///
        /// `json` is structured and round-trippable via `meka session import`,
        /// and includes any sub-agent child sessions. `markdown` is rendered
        /// and covers the single session only.
        #[arg(long, value_parser = parse_session_export_format, default_value = "markdown")]
        format: SessionExportFormat,
    },
    /// Delete one or more sessions
    ///
    /// Examples:
    ///   meka session delete 0e5f… 7a21…
    ///   meka session delete --older-than-days 90
    ///   meka session delete --all
    #[command(verbatim_doc_comment)]
    Delete {
        /// Session UUIDs to delete
        session_ids: Vec<uuid::Uuid>,
        /// Delete all sessions
        #[arg(long, conflicts_with = "older_than_days")]
        all: bool,
        /// Delete sessions not updated in this many days
        ///
        /// Conflicts with explicit IDs rather than ignoring them: a listed session younger than
        /// the window would otherwise be silently spared.
        #[arg(
            long = "older-than-days",
            value_name = "DAYS",
            conflicts_with = "session_ids"
        )]
        older_than_days: Option<u64>,
    },
    /// Import a session from a JSON export
    ///
    /// Recreates the session and any sub-agent children under fresh IDs so it
    /// can be resumed with `meka -r <new-id>`. Prints the new root session ID.
    Import {
        /// Export file to read (`-` for stdin)
        input: String,
    },
    /// Fork a session into an independent copy
    ///
    /// The copy carries the original's full conversation and continues from
    /// there; the original is untouched. Prints the new session ID.
    Fork {
        /// Session UUID to fork
        session_id: uuid::Uuid,
    },
    /// Drop the most recent turns from a session
    ///
    /// Cuts at a clean user boundary so no tool call is separated from its
    /// result. The log is append-only, so `meka session export` still shows
    /// what was dropped. Use this to recover a session the provider refuses.
    Rewind {
        /// Session UUID to rewind
        session_id: uuid::Uuid,
        /// Number of turns to drop
        #[arg(short = 'n', long, default_value = "1")]
        turns: usize,
    },
}

#[derive(clap::Subcommand, Debug)]
pub enum HistoryAction {
    /// List recorded input history
    List {
        /// Max entries to show (0 = all)
        #[arg(short = 'n', long, default_value = "50")]
        limit: u32,
    },
    /// Delete all recorded input history
    Clear,
}

// `Add` is the outlier with several flags inline; same one-shot CLI dispatch reasoning as the
// other action enums.
#[allow(clippy::large_enum_variant)]
#[derive(clap::Subcommand, Debug)]
pub enum ProviderAction {
    /// Add a provider profile and authenticate it
    ///
    /// Prompts for any of type/model/base-url not passed as flags, offers an
    /// optional advanced step (thinking, context window, effort), then acquires
    /// the secret (OAuth login for the subscription backends, API-key prompt for
    /// the rest) and stores it in the database. Becomes the default provider
    /// whenever no `default_provider` is set, not only for the first profile.
    Add {
        /// Profile name (e.g. `work`, `personal`).
        name: String,
        /// Backend type: the wire protocol
        ///
        /// One of: anthropic-messages, chatgpt-subscription,
        /// claude-subscription, openai-chat-completions, openai-responses.
        #[arg(long = "type")]
        r#type: Option<String>,
        /// API base URL
        ///
        /// Any endpoint serving the chosen protocol.
        #[arg(long = "base-url", value_name = "URL")]
        base_url: Option<String>,
        /// OAuth token endpoint override (advanced)
        ///
        /// Used for the initial code exchange and every refresh. For a proxied
        /// route out; the backend's own endpoint is used when this is unset.
        #[arg(long = "oauth-token-url", value_name = "URL")]
        oauth_token_url: Option<String>,
        /// OAuth client ID override (advanced)
        ///
        /// Subscription backends only. Like --oauth-token-url, it replaces a
        /// value meka has to hardcode but does not own.
        #[arg(long = "client-id", value_name = "ID")]
        client_id: Option<String>,
        /// Model name.
        #[arg(long)]
        model: Option<String>,
        /// Context window in tokens
        ///
        /// Skips the advanced prompt for this setting. Defaults to 1000000. Set the model's real
        /// window so compaction fires at the right point; meka never infers or probes for it.
        #[arg(long = "context-window", value_name = "TOKENS")]
        context_window: Option<u64>,
        /// Per-request output token cap
        ///
        /// Unset leaves the backend's own default. On Claude with thinking =
        /// budgeted this must exceed the thinking budget.
        #[arg(long = "max-output-tokens", value_name = "TOKENS")]
        max_output_tokens: Option<u64>,
        /// Reasoning effort (e.g. low, high)
        ///
        /// Skips the advanced prompt for this setting. Unset sends nothing, so the provider
        /// applies its own default.
        #[arg(long, value_name = "LEVEL")]
        effort: Option<String>,
        /// Accept image input (default: true)
        ///
        /// Set false for a text-only model, so the ACP frontend stops offering
        /// images it cannot use.
        #[arg(long, hide_possible_values = true, value_name = "BOOL")]
        vision: Option<bool>,
        /// Thinking mode: adaptive, budgeted, off
        ///
        /// Skips the advanced prompt for this setting. Anthropic Messages backends only; the rest
        /// ignore it. Defaults to adaptive.
        #[arg(long, value_enum, hide_possible_values = true, value_name = "MODE")]
        thinking: Option<crate::provider::ThinkingMode>,
        /// Token budget when thinking = budgeted
        ///
        /// Ignored by the other two modes, which send no budget. Defaults to
        /// `[thinking].budget_tokens`, then to 16000.
        #[arg(long = "thinking-budget", value_name = "TOKENS")]
        thinking_budget: Option<u64>,
        /// Redact thinking blocks (default: true)
        ///
        /// claude-subscription only. Saves bandwidth, but redacted payloads
        /// cannot be replayed to the server across turns.
        #[arg(
            long = "redact-thinking",
            hide_possible_values = true,
            value_name = "BOOL"
        )]
        redact_thinking: Option<bool>,
        /// Read the API key from stdin
        ///
        /// Non-interactive alternative to the key prompt. API backends only.
        #[arg(long = "api-key-stdin")]
        api_key_stdin: bool,
    },
    /// List configured provider profiles
    List,
    /// Change one setting on a provider profile
    ///
    /// Keeps every other setting and every comment, and leaves the profile's
    /// keys in meka's documented order. This is how a profile's model changes;
    /// there is no per-run flag for it, because a profile is an indivisible
    /// bundle and a session records which one it runs on, not a rewritten copy.
    ///
    /// Keys: base_url, oauth_token_url, client_id, model, context_window,
    /// max_output_tokens, effort, vision, thinking, thinking_budget,
    /// redact_thinking. `type` is not settable, because the stored credential
    /// was acquired for the current backend.
    ///
    /// Examples:
    ///   meka provider set work model claude-opus-5
    ///   meka provider set work context_window 200000
    ///   meka provider set work effort --unset
    #[command(verbatim_doc_comment)]
    Set {
        /// Profile name to change.
        name: String,
        /// Setting to write.
        key: String,
        /// New value. Omit it and pass --unset to remove the setting instead.
        value: Option<String>,
        /// Remove the setting, so the profile falls back to the default
        ///
        /// The inverse of setting a value, and not the same as setting an
        /// empty one: an absent key follows whatever meka's default becomes,
        /// which is what an unstated setting has always meant.
        #[arg(long, conflicts_with = "value")]
        unset: bool,
    },
    /// Set the default provider profile
    Use {
        /// Profile name to make the default.
        name: String,
    },
    /// Remove a provider profile and clear its stored credential
    Remove {
        /// Profile name to remove.
        name: String,
    },
    /// Re-authenticate an existing provider profile
    ///
    /// Keeps every other setting on the profile. Rotating a key with
    /// `remove` + `add` instead rebuilds the whole table, discarding
    /// `context_window`, `effort`, `thinking` and the rest.
    Login {
        /// Profile name to re-authenticate.
        name: String,
        /// Read the API key from stdin instead of prompting
        ///
        /// For scripted key rotation, the same as on `meka provider add`.
        /// API-key backends only; the subscription backends authenticate
        /// through the browser and have no key to read.
        #[arg(long = "api-key-stdin")]
        api_key_stdin: bool,
    },
}

// `Add` is the outlier with several flags inline; same one-shot CLI dispatch reasoning as
// [`Command`] and [`McpAction`] above.
#[allow(clippy::large_enum_variant)]
#[derive(clap::Subcommand, Debug)]
pub enum SkillAction {
    /// List installed skills
    List {
        /// Also show where each skill is on disk
        #[arg(long)]
        paths: bool,
    },
    /// Print one skill's frontmatter and on-disk paths
    Get { name: String },
    /// Print the rendered skill body
    Show { name: String },
    /// Scaffold a new skill at `~/.config/meka/skills/<name>/SKILL.md`
    ///
    /// Examples:
    ///   meka skill add demo --description "X"
    ///   meka skill add demo --description "X" --metadata author="Jane Doe"
    ///   meka skill add custom --from-file ./template.md
    #[command(verbatim_doc_comment)]
    Add {
        /// Unique skill name (lowercase letters, digits, hyphens)
        name: String,

        /// One-line description for the system prompt
        #[arg(long)]
        description: Option<String>,

        /// Listing rank 0-9, lower first. Defaults to 5
        #[arg(long, value_parser = clap::value_parser!(u8).range(0..=9))]
        priority: Option<u8>,

        /// Frontmatter metadata as key=value. Repeatable
        #[arg(long, value_name = "KEY=VALUE")]
        metadata: Vec<String>,

        /// Copy this file instead of the template
        #[arg(long = "from-file", value_name = "PATH")]
        from_file: Option<std::path::PathBuf>,

        /// Overwrite the skill directory if it exists
        #[arg(long)]
        force: bool,

        /// Open the new SKILL.md in $EDITOR afterwards
        #[arg(long)]
        edit: bool,
    },
    /// Remove a skill's directory
    Remove { name: String },
}

/// Inspect and cancel the wakeups the agent scheduled for itself through the `schedule_*` tools.
/// Read-and-cancel only: creating a job needs a session to attach it to, which is the agent's job.
#[derive(clap::Subcommand, Debug)]
pub enum ScheduleAction {
    /// List scheduled jobs
    ///
    /// Examples:
    ///   meka schedule list
    ///   meka schedule list --session 0b5c...
    #[command(verbatim_doc_comment)]
    List {
        /// Only this session's jobs (default: every session)
        #[arg(long)]
        session: Option<String>,
    },
    /// Cancel a job by id or unique prefix
    Cancel {
        /// Job id, or any unique prefix of one
        id: String,
    },
}

#[derive(clap::Subcommand, Debug)]
pub enum InstructionsAction {
    /// Print the resolved instructions and where they came from
    ///
    /// Resolution order is `MEKA_INSTRUCTIONS`, `MEKA_INSTRUCTIONS_FILE`,
    /// then `instructions.md` (or `instructions/`) in the config directory.
    /// `--instructions` belongs to a run, so it is not consulted here. The
    /// text goes to stdout and the source to stderr, so
    /// `meka instructions show 2>/dev/null` pipes cleanly.
    #[command(verbatim_doc_comment)]
    Show,
    /// Print the paths checked for instructions, and whether each exists
    Path,
}

/// Inspect and curate the agent's durable notes. The agent maintains these itself through the
/// `memory_*` tools; these subcommands are for reading, auditing, and pruning them by hand.
#[derive(clap::Subcommand, Debug)]
pub enum MemoryAction {
    /// List saved memories and the priority distribution
    List,
    /// Print one memory's stored fields
    Get { name: String },
    /// Print a memory's body
    Show { name: String },
    /// Write a memory by hand
    ///
    /// Examples:
    ///   meka memory add tz --description "K4YT3X is in UTC+8"
    ///   meka memory add rules --description "House rules" --priority 1
    #[command(verbatim_doc_comment)]
    Add {
        /// Unique name (alphanumerics, `-`, `_` only)
        name: String,

        /// Fact shown in every session's memory index
        #[arg(long)]
        description: String,

        /// 0 is most important, 9 least; defaults to 5
        #[arg(long, value_parser = clap::value_parser!(u8).range(0..=9))]
        priority: Option<u8>,

        /// Label for grouping and filtering; repeatable
        #[arg(long = "tag", value_name = "TAG")]
        tags: Vec<String>,

        /// Detail loaded only on memory_read
        #[arg(long)]
        body: Option<String>,

        /// Read the body from this file, not --body
        #[arg(long = "from-file", value_name = "PATH")]
        from_file: Option<std::path::PathBuf>,

        /// Update an existing memory instead of refusing
        #[arg(long)]
        force: bool,
    },
    /// Open a memory's body in $EDITOR
    ///
    /// The body only. To change a description, priority or tags, use
    /// `meka memory add <name> --force`, which keeps whatever it does
    /// not mention.
    #[command(verbatim_doc_comment)]
    Edit {
        /// Name of the memory to edit
        name: String,
    },
    /// Delete a memory permanently
    Remove { name: String },
    /// Check the search index against the stored memories
    ///
    /// The index is derived from the memories and can be regenerated,
    /// so neither answer here risks losing a note.
    #[command(verbatim_doc_comment)]
    Verify {
        /// Regenerate the index instead of only checking it
        #[arg(long)]
        rebuild: bool,
    },
    /// Write every memory out as Markdown, one file per memory
    ///
    /// The backup, grep and git answer for a store that lives in
    /// meka's database.
    ///
    /// Examples:
    ///   meka memory export
    ///   meka memory export --dir ~/backup/memory
    #[command(verbatim_doc_comment)]
    Export {
        /// Directory to write into; must be new or empty
        #[arg(long, value_name = "PATH")]
        dir: Option<std::path::PathBuf>,
    },
}

/// Read-only account introspection, for scripting (e.g. an i3blocks status bar). Both subcommands
/// take an optional profile (defaults to the active provider) and a `--format`.
#[derive(clap::Subcommand, Debug)]
pub enum AccountAction {
    /// Show account rate-limit usage (session / weekly windows)
    Usage {
        /// Provider profile (defaults to the active provider)
        profile: Option<String>,
        /// Output format
        #[arg(long, value_parser = parse_output_format, default_value = "plain")]
        format: OutputFormat,
    },
    /// Show account identity (plan, tier, org, role) and local auth status
    Whoami {
        /// Provider profile (defaults to the active provider)
        profile: Option<String>,
        /// Output format
        #[arg(long, value_parser = parse_output_format, default_value = "plain")]
        format: OutputFormat,
    },
    /// Show historical usage (lifetime tokens, streaks, per-day counts)
    Stats {
        /// Provider profile (defaults to the active provider)
        profile: Option<String>,
        /// Output format
        #[arg(long, value_parser = parse_output_format, default_value = "plain")]
        format: OutputFormat,
    },
}

/// Output format for `meka account` subcommands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Human-readable text.
    Plain,
    /// JSON (stable shape, for scripts).
    Json,
}

fn parse_output_format(s: &str) -> std::result::Result<OutputFormat, String> {
    match s.to_ascii_lowercase().as_str() {
        "plain" | "text" => Ok(OutputFormat::Plain),
        "json" => Ok(OutputFormat::Json),
        other => Err(format!(
            "unknown format '{}' (expected plain or json)",
            other
        )),
    }
}

/// Output format for `meka session export`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionExportFormat {
    /// Rendered Markdown (single session).
    Markdown,
    /// Structured JSON (round-trippable; includes sub-agent children).
    Json,
}

fn parse_session_export_format(s: &str) -> std::result::Result<SessionExportFormat, String> {
    match s.to_ascii_lowercase().as_str() {
        "markdown" | "md" => Ok(SessionExportFormat::Markdown),
        "json" => Ok(SessionExportFormat::Json),
        other => Err(format!(
            "unknown format '{}' (expected markdown or json)",
            other
        )),
    }
}

// Same reasoning as `Command` above: `Add` is the outlier and the enum lives on `main`'s stack for
// exactly one dispatch, not in a hot collection, so boxing would trade clarity for nothing.
#[allow(clippy::large_enum_variant)]
#[derive(clap::Subcommand, Debug)]
pub enum McpAction {
    /// List all configured MCP servers
    List,
    /// Print the configuration for one server
    Get { name: String },
    /// Connect once and exit non-zero if the handshake fails
    Reconnect { name: String },
    /// List a server's advertised tools with their resolved permissions
    Tools { name: String },
    /// Authenticate a server interactively (OAuth assumed for HTTP)
    ///
    /// With neither flag, runs the OAuth authorization-code flow. With one, stores the secret read
    /// from stdin and exits, which is also how an existing one is rotated.
    Login {
        /// Name of a server in config.toml
        name: String,

        /// Store a static bearer token read from stdin
        #[arg(long = "auth-token-stdin", conflicts_with = "client_secret_stdin")]
        auth_token_stdin: bool,

        /// Store an OAuth client secret read from stdin
        #[arg(long = "client-secret-stdin")]
        client_secret_stdin: bool,
    },
    /// Clear every stored credential for a server, revoking OAuth first
    Logout { name: String },
    /// Add a server to config.toml
    ///
    /// Examples:
    ///   meka mcp add pg npx -y @modelcontextprotocol/server-postgres
    ///   meka mcp add notion https://mcp.notion.com/mcp
    ///   meka mcp add api https://api.example.com/mcp --auth-token-stdin
    ///   meka mcp add notion https://mcp.notion.com/mcp --auth oauth
    // `rustdoc::bare_urls` normally turns URLs like https://example into auto-links, but these doc
    // lines are ALSO the text clap prints for `meka mcp add --help`. Angle-brackets would leak into
    // the CLI help. Allow bare URLs just on this variant.
    #[allow(rustdoc::bare_urls)]
    // Preserve line breaks in the `Examples:` block; clap's default joins consecutive `///` lines
    // into one re-wrapped paragraph.
    #[command(verbatim_doc_comment)]
    Add {
        /// Unique server name (alphanumerics, `-`, `_` only)
        name: String,
        /// URL (HTTP) or executable path (stdio); transport auto-detected
        location: Option<String>,
        /// Arguments to pass to the stdio command.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,

        /// Force transport (stdio or http); auto-detected otherwise
        #[arg(long, value_parser = parse_mcp_transport)]
        transport: Option<crate::config::McpTransport>,

        /// Environment variable for stdio (KEY=VALUE, repeatable)
        #[arg(long = "env", value_name = "KEY=VALUE")]
        env: Vec<String>,

        /// HTTP header (KEY=VALUE, repeatable)
        #[arg(long = "header", value_name = "KEY=VALUE")]
        header: Vec<String>,

        /// Authentication: oauth | client-credentials | client-credentials-jwt
        #[arg(long, value_parser = parse_mcp_auth_kind)]
        auth: Option<McpAuthKind>,

        /// Read a static bearer token from stdin (excludes --auth)
        ///
        /// The token is kept in meka's database, never in config.toml. There is no flag that takes
        /// it as an argument, because an argument is visible in `ps` output and in the shell
        /// history of every user on the machine.
        #[arg(long = "auth-token-stdin", conflicts_with = "client_secret_stdin")]
        auth_token_stdin: bool,

        /// OAuth / client-credentials client ID
        #[arg(long)]
        client_id: Option<String>,

        /// Read the OAuth client secret from stdin
        ///
        /// Kept in meka's database rather than config.toml, and read from stdin rather than an
        /// argument, for the same reasons as the bearer token above. One command reads one secret,
        /// so this cannot be combined with --auth-token-stdin.
        #[arg(long = "client-secret-stdin")]
        client_secret_stdin: bool,

        /// JWT signing key path (for client-credentials-jwt)
        #[arg(long)]
        signing_key: Option<String>,

        /// JWT signing algorithm (RS256, RS384, RS512, ES256, ES384)
        #[arg(long)]
        signing_algorithm: Option<String>,

        /// OAuth scope (repeatable)
        #[arg(long = "scope", value_name = "SCOPE")]
        scope: Vec<String>,

        /// Fixed OAuth redirect port (default: ephemeral)
        #[arg(long)]
        redirect_port: Option<u16>,

        /// Permission: none, read, workspace, ask, unrestricted (default: read)
        #[arg(long)]
        permission: Option<String>,

        /// Raw tool name to allow (repeatable; restricts which register)
        #[arg(long = "allow-tool", value_name = "TOOL")]
        allow_tool: Vec<String>,

        /// Raw tool name to block (repeatable; applied after --allow-tool)
        #[arg(long = "disable-tool", value_name = "TOOL")]
        disable_tool: Vec<String>,

        /// Raw tool name to eager-load (repeatable; skips load_tool)
        #[arg(long = "eager-load-tool", value_name = "TOOL")]
        eager_load_tool: Vec<String>,

        /// Per-tool permission override (TOOL=LEVEL, repeatable)
        #[arg(long = "tool-permission", value_name = "TOOL=LEVEL")]
        tool_permission: Vec<String>,

        /// Skip post-add auto-login; run `meka mcp login <name>` later
        #[arg(long = "no-login")]
        no_login: bool,

        /// Persist with disabled=true; re-enable via `meka mcp enable`
        #[arg(long = "disabled")]
        disabled: bool,

        /// Gate turns on this server: reject the turn if it is not connected
        #[arg(long = "required")]
        required: bool,
    },
    /// Remove a server from config.toml and clear stored creds
    Remove { name: String },
    /// Temporarily turn off a server without removing it from config
    Disable { name: String },
    /// Turn a disabled server back on
    Enable { name: String },
}

/// Authentication flavours selectable from the CLI. Maps onto the [`crate::config::McpAuthConfig`]
/// variants, except `None` which means "no `[auth]` block at all" (static token or
/// unauthenticated).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum McpAuthKind {
    OAuth,
    ClientCredentials,
    ClientCredentialsJwt,
}

fn parse_mcp_transport(s: &str) -> std::result::Result<crate::config::McpTransport, String> {
    match s.to_ascii_lowercase().as_str() {
        "stdio" => Ok(crate::config::McpTransport::Stdio),
        "http" => Ok(crate::config::McpTransport::Http),
        other => Err(format!(
            "unknown transport '{}' (expected stdio or http)",
            other
        )),
    }
}

fn parse_mcp_auth_kind(s: &str) -> std::result::Result<McpAuthKind, String> {
    match s.to_ascii_lowercase().as_str() {
        "oauth" => Ok(McpAuthKind::OAuth),
        "client-credentials" | "client_credentials" => Ok(McpAuthKind::ClientCredentials),
        "client-credentials-jwt" | "client_credentials_jwt" => {
            Ok(McpAuthKind::ClientCredentialsJwt)
        }
        other => Err(format!(
            "unknown auth '{}' (expected oauth, client-credentials, or client-credentials-jwt)",
            other
        )),
    }
}

#[derive(Parser, Debug)]
#[command(name = "meka", version, about = "A general-purpose AI agent harness")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Prompt for the first turn; add --oneshot to exit after it
    pub prompt: Option<String>,

    /// Continue the most recent session
    #[arg(short = 'c', long = "continue", conflicts_with = "resume")]
    pub continue_last: bool,

    /// Resume a session by UUID or leading prefix
    #[arg(short = 'r', long = "resume", value_name = "SESSION")]
    pub resume: Option<String>,

    /// Initial permission mode (none, read, workspace, ask, unrestricted)
    #[arg(long = "permission", value_parser = parse_permission)]
    pub permission: Option<Permission>,

    /// Extra directory writable at `workspace` permission (repeatable)
    ///
    /// The working directory is always writable at that level, as are any folders an
    /// ACP client supplies. This adds to them.
    ///
    /// Deliberately a flag rather than a config key: which folders this run may write
    /// is a per-run scope, like the working directory itself, not a preference to
    /// persist.
    #[arg(long = "writable-root", value_name = "PATH")]
    pub writable_root: Vec<std::path::PathBuf>,

    /// Provider profile for this session; on a resume, repins it
    ///
    /// The only provider flag on a run. A profile is an indivisible bundle of a backend, an
    /// endpoint, a credential and every model-tied setting, so this selects one rather than
    /// rewriting part of one. Change a field with `meka provider set`.
    #[arg(short = 'p', long = "provider")]
    pub provider: Option<String>,

    /// Linux sandbox backend: landlock or bubblewrap
    #[arg(long = "sandbox-backend", value_parser = parse_sandbox_backend)]
    pub sandbox_backend: Option<crate::config::SandboxBackend>,

    /// Disable streaming mode
    #[arg(long = "no-stream")]
    pub no_stream: bool,

    /// Markdown render mode: termimad (default), syntect, raw, or silent
    #[arg(long = "render-mode", value_parser = parse_render_mode)]
    pub render_mode: Option<crate::render::RenderMode>,

    /// Standing instructions for this run, replacing the discovered ones
    #[arg(long = "instructions", value_name = "STRING")]
    pub instructions: Option<String>,

    /// Invoke a user-invocable skill on the first turn.
    #[arg(long = "skill", value_name = "NAME")]
    pub skill: Option<String>,

    /// Exit after the first turn finishes (requires a prompt or `--skill`).
    #[arg(long = "oneshot")]
    pub oneshot: bool,

    /// Eager-load an MCP tool this session (raw SERVER:TOOL, repeatable)
    #[arg(long = "eager-load-tool", value_name = "SERVER:TOOL")]
    pub eager_load_tool: Vec<String>,

    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count)]
    pub verbosity: u8,
}

fn parse_permission(s: &str) -> std::result::Result<Permission, String> {
    s.parse()
}

fn parse_render_mode(s: &str) -> std::result::Result<crate::render::RenderMode, String> {
    s.parse()
}

fn parse_sandbox_backend(s: &str) -> std::result::Result<crate::config::SandboxBackend, String> {
    s.parse()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_defaults() {
        let cli = Cli::parse_from(["meka"]);
        assert!(cli.command.is_none());
        assert!(cli.prompt.is_none());
        assert!(!cli.continue_last);
        assert!(cli.resume.is_none());
        assert!(cli.permission.is_none());
        assert!(cli.provider.is_none());
        assert!(!cli.no_stream);
        assert!(cli.render_mode.is_none());
        assert!(cli.skill.is_none());
        assert!(!cli.oneshot);
        assert!(cli.eager_load_tool.is_empty());
        assert_eq!(cli.verbosity, 0);
    }

    #[test]
    fn test_cli_eager_load_tool_repeatable() {
        let cli = Cli::parse_from([
            "meka",
            "--eager-load-tool",
            "notion:search",
            "--eager-load-tool",
            "github:create_issue",
        ]);
        assert_eq!(cli.eager_load_tool, vec![
            "notion:search".to_string(),
            "github:create_issue".to_string()
        ]);
    }

    #[test]
    fn test_cli_oneshot_flag() {
        let cli = Cli::parse_from(["meka", "--oneshot", "do thing"]);
        assert!(cli.oneshot);
        assert_eq!(cli.prompt.as_deref(), Some("do thing"));
    }

    #[test]
    fn test_cli_oneshot_prompt() {
        let cli = Cli::parse_from(["meka", "hello world"]);
        assert_eq!(cli.prompt.as_deref(), Some("hello world"));
    }

    #[test]
    fn test_cli_skill_flag_alone() {
        let cli = Cli::parse_from(["meka", "--skill", "demo"]);
        assert_eq!(cli.skill.as_deref(), Some("demo"));
        assert!(cli.prompt.is_none());
    }

    #[test]
    fn test_cli_skill_flag_with_extra_prompt() {
        let cli = Cli::parse_from(["meka", "--skill", "demo", "extra context"]);
        assert_eq!(cli.skill.as_deref(), Some("demo"));
        assert_eq!(cli.prompt.as_deref(), Some("extra context"));
    }

    #[test]
    fn test_cli_continue_last() {
        let cli = Cli::parse_from(["meka", "-c"]);
        assert!(cli.continue_last);
        assert!(cli.resume.is_none());
    }

    #[test]
    fn test_cli_resume_specific_session() {
        let cli = Cli::parse_from(["meka", "-r", "550e8400-e29b-41d4-a716-446655440000"]);
        assert_eq!(
            cli.resume.as_deref(),
            Some("550e8400-e29b-41d4-a716-446655440000")
        );
        assert!(!cli.continue_last);
    }

    /// The reason `-c` stopped taking an optional value: it was the only root flag that could
    /// swallow the next argument, so a prompt after it was read as a session prefix.
    #[test]
    fn test_cli_continue_does_not_consume_the_prompt() {
        let cli = Cli::parse_from(["meka", "-c", "fix the bug"]);
        assert!(cli.continue_last);
        assert_eq!(cli.prompt.as_deref(), Some("fix the bug"));
    }

    #[test]
    fn test_cli_resume_takes_the_id_and_leaves_the_prompt() {
        let cli = Cli::parse_from(["meka", "-r", "550e8400", "fix the bug"]);
        assert_eq!(cli.resume.as_deref(), Some("550e8400"));
        assert_eq!(cli.prompt.as_deref(), Some("fix the bug"));
    }

    #[test]
    fn test_cli_continue_and_resume_are_mutually_exclusive() {
        assert!(Cli::try_parse_from(["meka", "-c", "-r", "550e8400"]).is_err());
    }

    #[test]
    fn test_cli_flags() {
        let cli = Cli::parse_from([
            "meka",
            "--provider",
            "openai-chat-completions",
            "--no-stream",
            "-c",
            "-vv",
        ]);
        assert_eq!(cli.provider.as_deref(), Some("openai-chat-completions"));
        assert!(cli.no_stream);
        assert!(cli.continue_last);
        assert_eq!(cli.verbosity, 2);
    }

    #[test]
    fn test_cli_provider_short_form() {
        let cli = Cli::parse_from(["meka", "-p", "work", "fix the bug"]);
        assert_eq!(cli.provider.as_deref(), Some("work"));
        assert_eq!(cli.prompt.as_deref(), Some("fix the bug"));
    }

    #[test]
    fn test_cli_permission_flag() {
        let cli = Cli::parse_from(["meka", "--permission", "workspace"]);
        assert_eq!(cli.permission, Some(Permission::Workspace));
        let cli = Cli::parse_from(["meka", "--permission", "unrestricted"]);
        assert_eq!(cli.permission, Some(Permission::Unrestricted));
        // The single-letter aliases are the indicator characters, so what the prompt shows is
        // always something the flag accepts.
        let cli = Cli::parse_from(["meka", "--permission", "u"]);
        assert_eq!(cli.permission, Some(Permission::Unrestricted));
    }

    /// A mode meka does not have must fail *loudly* at the CLI rather than resolve to anything.
    ///
    /// This is the surface where an invocation is most likely to be automated, and a hard exit is
    /// the good outcome there: a nonzero status stops a script, where quietly picking a mode would
    /// let it run on with authority nobody chose.
    #[test]
    fn permission_flag_refuses_a_mode_it_does_not_have() {
        let error = Cli::try_parse_from(["meka", "--permission", "elevated"])
            .expect_err("an unknown mode must not parse");
        let rendered = error.to_string();
        for mode in ["none", "read", "workspace", "ask", "unrestricted"] {
            assert!(rendered.contains(mode), "clap must list {mode}: {rendered}");
        }
    }

    #[test]
    fn test_cli_continue_long_form() {
        let cli = Cli::parse_from(["meka", "--continue"]);
        assert!(cli.continue_last);
    }

    #[test]
    fn test_cli_resume_long_form() {
        let cli = Cli::parse_from(["meka", "--resume", "550e8400-e29b-41d4-a716-446655440000"]);
        assert_eq!(
            cli.resume.as_deref(),
            Some("550e8400-e29b-41d4-a716-446655440000")
        );
    }

    #[test]
    fn test_cli_provider_add_subcommand() {
        let cli = Cli::parse_from([
            "meka",
            "provider",
            "add",
            "work",
            "--type",
            "claude-subscription",
        ]);
        match cli.command {
            Some(Command::Provider {
                action: ProviderAction::Add { name, r#type, .. },
            }) => {
                assert_eq!(name, "work");
                assert_eq!(r#type.as_deref(), Some("claude-subscription"));
            }
            other => panic!("expected provider add, got {:?}", other),
        }
    }

    #[test]
    fn test_cli_session_list_subcommand() {
        let cli = Cli::parse_from(["meka", "session", "list"]);
        match cli.command {
            Some(Command::Session {
                action:
                    SessionAction::List {
                        limit,
                        include_children,
                    },
            }) => {
                assert_eq!(limit, 20);
                assert!(!include_children);
            }
            other => panic!("expected session list, got {:?}", other),
        }
    }

    #[test]
    fn test_cli_session_delete_all_subcommand() {
        let id = "550e8400-e29b-41d4-a716-446655440000";
        let cli = Cli::parse_from(["meka", "session", "delete", id, "--all"]);
        match cli.command {
            Some(Command::Session {
                action:
                    SessionAction::Delete {
                        session_ids, all, ..
                    },
            }) => {
                assert_eq!(session_ids.len(), 1);
                assert!(all);
            }
            other => panic!("expected session delete, got {:?}", other),
        }
    }

    /// The manual replacement for size-based auto-cleanup, so it has to actually parse.
    #[test]
    fn test_cli_session_delete_older_than_days() {
        let cli = Cli::parse_from(["meka", "session", "delete", "--older-than-days", "90"]);
        match cli.command {
            Some(Command::Session {
                action:
                    SessionAction::Delete {
                        session_ids,
                        all,
                        older_than_days,
                    },
            }) => {
                assert!(session_ids.is_empty());
                assert!(!all);
                assert_eq!(older_than_days, Some(90));
            }
            other => panic!("expected session delete, got {:?}", other),
        }
    }

    /// Both other selectors must be refused alongside it. `--all` because the two windows
    /// disagree, and explicit IDs because a listed session younger than the window would be
    /// silently spared while the user watched a different count come back.
    #[test]
    fn test_cli_session_delete_older_than_days_conflicts() {
        let id = "550e8400-e29b-41d4-a716-446655440000";
        for args in [
            vec![
                "meka",
                "session",
                "delete",
                "--older-than-days",
                "90",
                "--all",
            ],
            vec!["meka", "session", "delete", "--older-than-days", "90", id],
        ] {
            assert!(
                Cli::try_parse_from(&args).is_err(),
                "{args:?} must be rejected"
            );
        }
    }

    #[test]
    fn test_cli_session_export_stdout_subcommand() {
        let id = "550e8400-e29b-41d4-a716-446655440000";
        let cli = Cli::parse_from(["meka", "session", "export", id, "-o", "-"]);
        match cli.command {
            Some(Command::Session {
                action: SessionAction::Export { output, .. },
            }) => assert_eq!(output.as_deref(), Some("-")),
            other => panic!("expected session export, got {:?}", other),
        }
    }

    #[test]
    fn test_cli_session_fork_subcommand() {
        let id = "550e8400-e29b-41d4-a716-446655440000";
        let cli = Cli::parse_from(["meka", "session", "fork", id]);
        match cli.command {
            Some(Command::Session {
                action: SessionAction::Fork { session_id },
            }) => assert_eq!(session_id.to_string(), id),
            other => panic!("expected session fork, got {:?}", other),
        }
    }

    #[test]
    fn test_cli_history_list_subcommand() {
        let cli = Cli::parse_from(["meka", "history", "list", "-n", "10"]);
        match cli.command {
            Some(Command::History {
                action: HistoryAction::List { limit },
            }) => assert_eq!(limit, 10),
            other => panic!("expected history list, got {:?}", other),
        }
    }

    #[test]
    fn test_cli_history_clear_subcommand() {
        let cli = Cli::parse_from(["meka", "history", "clear"]);
        assert!(matches!(
            cli.command,
            Some(Command::History {
                action: HistoryAction::Clear
            })
        ));
    }
}
