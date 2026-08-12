//! Per-turn agent loop: streams provider output, dispatches tool calls, and persists the resulting
//! messages to the session store. Also handles mid-conversation auto-compaction when the
//! input-token budget is exceeded.

use std::{
    path::PathBuf,
    sync::{Arc, RwLock},
};

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// Why an [`Agent::run_turn`] invocation finished cleanly. Callers that drive a user-facing
/// protocol (e.g. the ACP `session/prompt` response) use this to map to a protocol-level stop
/// reason; REPL and one-shot callers discard it. `Interrupted` is not represented here. It
/// surfaces as `Err(MekaError::Interrupted)` so the success-path return type stays straightforward.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnOutcome {
    /// The model returned a natural end-of-turn (or an unrecognised stop reason, treated as
    /// end-of-turn since we have nothing better to surface).
    EndTurn,
    /// The provider stopped because the model hit its maximum output tokens. The assistant message
    /// may be truncated; clients can reflect this in their UI.
    MaxTokens,
    /// The model refused to comply with the request (Claude `stop_reason: "refusal"`, OpenAI
    /// equivalent). The string carries the model's refusal text when available so clients can
    /// render it instead of a generic "request failed."
    Refusal(String),
}

/// Per-session working directory, shared by reference between the agent, every file-touching tool,
/// the REPL prompt, the `/cd` slash command, and the per-turn environment-context block.
/// `std::sync::RwLock` (rather than `tokio::sync::RwLock`) so the synchronous REPL prompt can read
/// it without entering an async context; reads/writes are microseconds (a `PathBuf` clone or
/// replace), never held across `.await`.
pub type SharedCwd = Arc<RwLock<PathBuf>>;

/// Read the current value of [`SharedCwd`]. Recovers from a poisoned lock by extracting the inner
/// value; meka never panics with the cwd lock held, so the only way to see a poisoned lock is a
/// separate bug that already triggered, and falling back to the stored value beats crashing the
/// agent on every subsequent tool call.
pub fn cwd_snapshot(cwd: &SharedCwd) -> PathBuf {
    cwd.read()
        .map(|guard| guard.clone())
        .unwrap_or_else(|poisoned| poisoned.into_inner().clone())
}

/// Resolve a tool-input path against the per-session [`SharedCwd`]. Absolute paths pass through
/// unchanged; relative paths are joined to the current cwd value. Tools use this at the top of
/// their `execute` methods to decouple from process `cwd`.
pub fn resolve_against_cwd(cwd: &SharedCwd, input: impl AsRef<std::path::Path>) -> PathBuf {
    let input = input.as_ref();
    if input.is_absolute() {
        input.to_path_buf()
    } else {
        cwd_snapshot(cwd).join(input)
    }
}

/// Workspace roots beyond [`SharedCwd`], as supplied by an ACP client's `additionalDirectories`.
///
/// A separate handle rather than a field on `SharedCwd` because only the two search tools and the
/// environment-context block care: widening [`resolve_against_cwd`] would touch every file tool's
/// constructor to serve two callers. `cwd` remains the base for relative paths, per the ACP spec,
/// so these expand *discovery* scope only.
///
/// Empty for the REPL, the HTTP API, and any ACP client that sends no extra roots.
pub type SharedRoots = Arc<RwLock<Vec<PathBuf>>>;

/// Read the current value of [`SharedRoots`], with the same poisoned-lock recovery as
/// [`cwd_snapshot`] and for the same reason.
pub fn roots_snapshot(roots: &SharedRoots) -> Vec<PathBuf> {
    roots
        .read()
        .map(|guard| guard.clone())
        .unwrap_or_else(|poisoned| poisoned.into_inner().clone())
}

/// The ordered set of roots a **recursive** search should sweep when the caller named no explicit
/// path: `cwd` first, then each additional root, with anything already covered by another root
/// dropped.
///
/// Only correct for a walker that descends, which today means `search_contents`. A tool that
/// anchors a pattern at each root instead wants [`glob_roots`]; dropping a contained root would
/// drop the files under it.
///
/// A root is dropped when some other root *contains* it, which subsumes exact duplicates. Both
/// shapes are things a client legitimately sends: Zed may repeat `cwd` inside
/// `additionalDirectories`, and nothing stops a client naming a folder nested inside another. Left
/// in, the overlapping tree is walked twice, so every file under it is reported twice, consumes two
/// slots of the result cap, and spends the shared walk budget twice.
///
/// Containment is checked in both directions, so a root that is an *ancestor* of `cwd` wins and
/// `cwd` drops out of the search set. A descending walk from the ancestor still reaches everything
/// under `cwd`, and this does not affect `cwd`'s real job: it remains the base for relative paths
/// and the shell's working directory regardless of what this returns.
///
/// Paths are compared as given. A symlink pointing at another root, or a path containing `..`, is
/// not detected; canonicalising to catch those would resolve symlinked roots to targets the client
/// never named, which is a worse trade than an occasional duplicate.
pub fn search_roots(cwd: &SharedCwd, roots: &SharedRoots) -> Vec<PathBuf> {
    let mut kept: Vec<PathBuf> = Vec::new();
    for path in std::iter::once(cwd_snapshot(cwd)).chain(roots_snapshot(roots)) {
        if kept.iter().any(|existing| path.starts_with(existing)) {
            continue;
        }
        // This root is broader than ones already kept, so those become redundant.
        kept.retain(|existing| !existing.starts_with(&path));
        kept.push(path);
    }
    kept
}

/// The ordered set of roots to anchor a glob at when the caller named no explicit path: `cwd`
/// first, then each additional root, with only *exact* repeats dropped.
///
/// The counterpart to [`search_roots`] for a tool that builds one rooted pattern per root rather
/// than descending from it. Containment must not drop anything here: `find_files` turns each root
/// into `<root>/<pattern>`, and a glob's `*` does not cross `/`, so a workspace of `/work` plus
/// `cwd = /work/main` would answer `*.md` from `/work/*.md` alone and miss
/// `/work/main/README.md` entirely. That is the exact "the agent says a file you can see doesn't
/// exist" failure multi-root support was added to prevent, so nested roots are all kept and the
/// caller deduplicates the matches instead.
pub fn glob_roots(cwd: &SharedCwd, roots: &SharedRoots) -> Vec<PathBuf> {
    let mut kept: Vec<PathBuf> = Vec::new();
    for path in std::iter::once(cwd_snapshot(cwd)).chain(roots_snapshot(roots)) {
        if !kept.contains(&path) {
            kept.push(path);
        }
    }
    kept
}

/// Construct a fresh [`SharedCwd`] pointing at the process cwd, for use in tests that need to
/// instantiate a tool but don't exercise the per-session cwd resolution path. Tests using absolute
/// paths or `tempdir()` are unaffected by the value here.
#[cfg(test)]
pub fn test_cwd() -> SharedCwd {
    Arc::new(RwLock::new(
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    ))
}

/// Construct an empty [`SharedRoots`], for tools under test that don't exercise multi-root search.
#[cfg(test)]
pub fn test_roots() -> SharedRoots {
    Arc::new(RwLock::new(Vec::new()))
}

use crate::{
    context,
    conversation::Conversation,
    error::{MekaError, Result},
    frontend::{Frontend, FrontendEvent, PermissionOutcome, PermissionRequest},
    memory::MemoryCache,
    permission::SharedPermission,
    provider::{
        ContentBlock, ImageSource, Message, Provider, Role, StopReason, StreamEvent,
        ToolDefinition, ToolResultContent,
    },
    session::SessionManager,
    skills::SkillCache,
    tools::{ToolRegistry, todo::SharedTodoList},
};

/// Trigger auto-compaction once a turn's input tokens exceed this fraction of the configured
/// context window.
const AUTO_COMPACT_THRESHOLD_PERCENT: u64 = 80;

/// How many times a single turn may emergency-compact-and-retry after the provider reports a
/// context-window overflow before giving up. One pass shrinks the request dramatically; if it still
/// overflows, looping won't help.
const MAX_OVERFLOW_RETRIES: u32 = 1;

/// How many times a single turn may degrade-and-retry after the provider rejects the request as
/// malformed. One pass strips every non-text block meka added since the last accepted request, so a
/// second pass would have nothing left to remove.
const MAX_REQUEST_REPAIRS: u32 = 1;

/// Per-turn configuration knobs for [`Agent`]. Constructed once by `main` from the
/// [`crate::config::ResolvedConfig`] and held immutably for the agent's lifetime; mid-session
/// permission cycling and tool loading are handled by shared state (see [`SharedPermission`] and
/// [`ToolRegistry`]) rather than by mutating fields here.
#[derive(Clone)]
pub struct AgentOptions {
    /// When true, assistant responses stream token-by-token via `Provider::stream`; otherwise the
    /// agent uses the blocking `Provider::complete`.
    pub streaming: bool,
    /// Whether read-mode `execute_command` calls run inside the platform sandbox. Forced off when
    /// no sandbox backend is available.
    pub sandboxed_shell: bool,
    /// Cap on messages sent to the provider per turn. `None` = unlimited; the agent walks back to
    /// a safe boundary so tool-result chains stay intact (see
    /// `truncate_messages_for_context`).
    pub context_messages: Option<usize>,
    /// When true, the agent auto-compacts the conversation once a turn's input tokens cross
    /// [`AUTO_COMPACT_THRESHOLD_PERCENT`] of [`Self::context_window`]. Requires `context_window >
    /// 0`.
    pub auto_compact: bool,
    /// Provider's advertised context window in tokens. Drives auto-compact.
    pub context_window: u64,
    /// User-authored instructions, surfaced in the system prompt and to sub-agents. Per-run
    /// `--instructions` overrides the config-file value.
    pub user_instructions: Option<String>,
    /// Max time to wait for still-`Pending` MCP servers to settle before the readiness gate
    /// decides. Which servers actually gate is per-server (`[[mcp.servers]].required`), so there
    /// is no strictness flag here.
    pub mcp_grace: std::time::Duration,
    /// When `Some`, `run_turn` uses this string verbatim instead of invoking
    /// [`crate::context::build_system_prompt`]. Sub-agents set this to their stripped-down prompt
    /// from `build_subagent_system_prompt`. The override is static; it does not see per-turn todo
    /// updates or permission changes, which is fine for one-shot sub-agents whose tool list and
    /// permission level are fixed at spawn time.
    pub system_prompt_override: Option<String>,
}

/// Driver for a single conversation. One [`Agent`] handles one or more sequential turns against a
/// single provider, with a shared tool registry, shared permission state, and a persistent SQLite
/// session. A turn fans out tool calls (in parallel via `join_all`) and persists every assistant
/// and tool-result message to the session store.
///
/// `Agent` is held across turns but not across providers; switching providers requires a fresh
/// instance.
pub struct Agent {
    provider: Arc<dyn Provider>,
    tool_registry: ToolRegistry,
    session_manager: SessionManager,
    shared_permission: SharedPermission,
    options: AgentOptions,
    todo_list: SharedTodoList,
    /// Last todo state pushed to the frontend, so a no-op `todo` call (e.g. a read with no
    /// arguments, or a rewrite that changes nothing) doesn't re-render the list. Private to this
    /// `Agent`; sub-agents route through `Agent::new` and so get their own.
    last_rendered_todo: tokio::sync::RwLock<Option<crate::tools::todo::TodoState>>,
    /// The tool/skill/MCP picture the model was last shown, plus the conversation length at which
    /// it was shown. `None` means "tell it everything": a fresh agent, or a compaction that may
    /// have summarized the earlier rendering away. Same shape and reasoning as
    /// [`Self::last_rendered_todo`].
    ///
    /// The length matters because the render lives in a single user message, and
    /// [`truncate_messages_for_context`] sends only the most recent `context_messages` entries.
    /// Once that message falls out of the window the model can no longer see the catalogue,
    /// the skill list, or any MCP server's instructions, so the picture has to be restated.
    /// Tracking where it landed means that costs a full render roughly once per window rather
    /// than once per turn.
    last_rendered_world: tokio::sync::RwLock<Option<(crate::context::WorldSnapshot, usize)>>,
    shared_session_id: Arc<tokio::sync::RwLock<Option<uuid::Uuid>>>,
    /// Shared skill cache. Re-checks the on-disk snapshot at the top of each turn and re-discovers
    /// when something changed, so adds / removes / frontmatter edits land without restart.
    /// Body-only edits take effect even sooner; `load_skill_body` re-reads from disk on every
    /// invocation regardless of cache state.
    skills: Arc<SkillCache>,
    /// Shared memory cache, same contract as `skills`: re-checked at the top of each turn so a
    /// memory the agent writes mid-turn appears in the very next turn's index.
    memories: Arc<MemoryCache>,
    /// Where streaming output, todo-list renders, token-usage summaries,
    /// and tool-approval requests flow. Concrete impls today:
    /// [`crate::repl::ReplFrontend`], [`crate::acp::AcpFrontend`],
    /// [`crate::frontend::SilentFrontend`], and [`crate::frontend::PermissionForwardingFrontend`].
    frontend: Arc<dyn Frontend>,
    /// Per-session working directory. Initialised from `std::env::current_dir()` at startup;
    /// updated by `/cd`; read by the file/shell/find/grep tools, the REPL prompt, and the per-turn
    /// environment-context block. Process `cwd` is no longer mutated.
    cwd: SharedCwd,
    /// Workspace roots beyond [`Self::cwd`], from an ACP client's `additionalDirectories`. Read by
    /// the per-turn environment-context block; the search tools hold the same handle. Empty for
    /// every non-ACP session.
    roots: SharedRoots,
    /// Total tokens of this agent's most recent provider round: the live, cache-write, and
    /// cache-read input tiers plus output. That equals everything in context as of the last
    /// exchange, i.e. the size the next request re-sends minus the new user prompt. Drives
    /// auto-compact and the `/status` context gauge, and is shared (`Arc`) with the REPL prompt
    /// for the optional live indicator. Seeded by an estimate after `/compact` and on resume
    /// until the next real response corrects it. Per-`Agent`, so sub-agents (own counter) are
    /// excluded from a parent's reading.
    last_context_tokens: Arc<std::sync::atomic::AtomicU64>,
    /// Per-turn map of `tool_use_id` → scratchpad-name hint. Populated by MCP tool adapters so
    /// oversized-output persistence uses `mcp_<server>_<tool>` instead of the plain tool name.
    /// Cleared between turns by `persist_oversized_results`.
    scratchpad_hints: Arc<tokio::sync::RwLock<std::collections::HashMap<String, String>>>,
    /// Tools that have already been the subject of a [`Self::schema_advisory`]. Never cleared: the
    /// advisory lives on in the conversation, so a second copy teaches nothing and costs context
    /// on every later call.
    schema_advisories_sent: Arc<std::sync::Mutex<std::collections::HashSet<String>>>,
    /// Live handles for background tool calls this agent started. Empty and inert unless
    /// `[background] enabled`; shared with the `task_*` tools and the REPL so all three act on one
    /// set. See [`crate::background`].
    background_tasks: crate::background::BackgroundTasks,
    /// Ceiling on this session's concurrent background tasks, from `[background] max_tasks`. Zero
    /// when the feature is off, which is also what refuses a call that somehow arrives anyway.
    background_max_tasks: usize,
    /// Optional MCP client manager; used to read server-supplied `InitializeResult.instructions`
    /// for inclusion in the system prompt.
    mcp_manager: Option<Arc<crate::mcp::McpClientManager>>,
    /// Counters surfaced by `/status`. Shared with the Claude providers, which increment the
    /// redaction-related fields when oversized request bodies trigger image-block redaction.
    session_stats: Arc<crate::stats::SessionStats>,
    /// Whether this agent persists `session_stats` onto its session row after each turn. True for
    /// the primary agent; false for sub-agents, which share the parent's `SessionStats` Arc but
    /// own a child session row (so only the primary writes the parent-inclusive totals).
    persist_session_stats: bool,
    /// Conversation length at the time of the most recent request the provider *accepted*, or
    /// [`LAST_ACCEPTED_UNKNOWN`] before the first one. Everything appended past it is what a
    /// `MekaError::InvalidRequest` is allowed to blame: the failing request differs from the last
    /// good one by exactly those messages, which is how `run_turn` locates the offending content
    /// without parsing the provider's error path (Anthropic's `messages.34.content.0…`), a shape
    /// no other backend produces and none of them map cleanly back through context truncation.
    ///
    /// Carried across turns rather than reset per turn so a turn that failed *after* appending its
    /// user message leaves that message a suspect on the retry; a fresh-per-turn floor would put
    /// it out of reach and leave the session stuck. The cost is that it must be invalidated
    /// whenever the conversation is rewritten under the agent, which
    /// [`Self::reset_conversation_markers`] and `compact_session` are responsible for.
    ///
    /// Atomic rather than `&mut` because `run_turn` takes `&self`; never shared between agents
    /// (sub-agents construct their own through [`Self::new`]).
    last_accepted_len: std::sync::atomic::AtomicUsize,
}

/// Sentinel for [`Agent::last_accepted_len`] before any request has come back 2xx, and after a
/// compaction rewrites the conversation and makes earlier lengths incomparable.
const LAST_ACCEPTED_UNKNOWN: usize = usize::MAX;

impl Agent {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provider: Arc<dyn Provider>,
        tool_registry: ToolRegistry,
        session_manager: SessionManager,
        shared_permission: SharedPermission,
        options: AgentOptions,
        todo_list: SharedTodoList,
        shared_session_id: Arc<tokio::sync::RwLock<Option<uuid::Uuid>>>,
        skills: Arc<SkillCache>,
        memories: Arc<MemoryCache>,
        frontend: Arc<dyn Frontend>,
        cwd: SharedCwd,
        roots: SharedRoots,
        session_stats: Arc<crate::stats::SessionStats>,
    ) -> Self {
        Self {
            provider,
            tool_registry,
            session_manager,
            shared_permission,
            options,
            todo_list,
            last_rendered_todo: tokio::sync::RwLock::new(None),
            last_rendered_world: tokio::sync::RwLock::new(None),
            shared_session_id,
            skills,
            memories,
            frontend,
            cwd,
            roots,
            last_context_tokens: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            scratchpad_hints: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
            schema_advisories_sent: Arc::new(std::sync::Mutex::new(
                std::collections::HashSet::new(),
            )),
            // Off until `enable_background` says otherwise, so every path that forgets to configure
            // it gets today's synchronous behaviour rather than a half-wired detach.
            background_tasks: crate::background::BackgroundTasks::default(),
            background_max_tasks: 0,
            mcp_manager: None,
            session_stats,
            persist_session_stats: true,
            last_accepted_len: std::sync::atomic::AtomicUsize::new(LAST_ACCEPTED_UNKNOWN),
        }
    }

    /// Swap the provider after construction. Used by the ACP integration test path
    /// (`MEKA_ACP_MOCK_PROVIDER=1`) so the test can drive a scripted
    /// [`crate::provider::mock::MockProvider`] without going through the credential / HTTP-client
    /// setup that `create_agent_from_config` performs for real providers. Debug builds only;
    /// release builds don't include it.
    #[cfg(debug_assertions)]
    pub fn set_provider(&mut self, provider: Arc<dyn Provider>) {
        self.provider = provider;
    }

    /// Shared handle to the agent's session-scoped working directory. Public so frontends can
    /// observe live cwd changes via the same `Arc` the `/cd` handler mutates; currently unused
    /// because main.rs / acp.rs build the `SharedCwd` themselves and pass it in. Kept
    /// allow(dead_code) until a frontend reaches for it.
    #[allow(dead_code)]
    pub fn cwd(&self) -> &SharedCwd {
        &self.cwd
    }

    /// Build an `Agent` configured for sub-agent use: no compaction, no MCP readiness gate.
    /// Inherits `sandboxed_shell`, `context_messages`, and `user_instructions` from the parent's
    /// options.
    ///
    /// `sub_system_prompt` is the pre-built sub-agent system prompt (typically from
    /// `build_subagent_system_prompt`); `run_turn` uses it verbatim instead of building one
    /// dynamically.
    ///
    /// `frontend` decides where the sub-agent's output and permission requests go. The standard
    /// caller (the `spawn_agent` tool) uses [`crate::frontend::PermissionForwardingFrontend`]
    /// wrapping the parent's frontend. That wrapper drops emits (the sub-agent's report flows back
    /// via the tool result) but forwards permission prompts so the user is asked in their original
    /// UI. Tests can pass [`crate::frontend::SilentFrontend`] for fully-isolated sub-agent
    /// runs.
    ///
    /// Doesn't call `set_mcp_manager`. MCP tool dispatch from the sub-agent's registry works
    /// without an attached manager because the adapters delegate through `Arc<ServerEntry>`
    /// directly, and the paths that do need the manager (`load_tool`, the unknown-tool
    /// explanation) reach it through the registry, which
    /// [`crate::mcp::McpClientManager::install_tools_on`] wires up.
    #[allow(clippy::too_many_arguments)]
    pub fn new_subagent(
        provider: Arc<dyn Provider>,
        tool_registry: ToolRegistry,
        session_manager: SessionManager,
        shared_permission: SharedPermission,
        parent_options: &AgentOptions,
        sub_system_prompt: String,
        todo_list: SharedTodoList,
        shared_session_id: Arc<tokio::sync::RwLock<Option<uuid::Uuid>>>,
        skills: Arc<SkillCache>,
        memories: Arc<MemoryCache>,
        parent_cwd: &SharedCwd,
        parent_roots: &SharedRoots,
        frontend: Arc<dyn Frontend>,
        session_stats: Arc<crate::stats::SessionStats>,
    ) -> Self {
        let options = AgentOptions {
            sandboxed_shell: parent_options.sandboxed_shell,
            context_messages: parent_options.context_messages,
            user_instructions: parent_options.user_instructions.clone(),
            // Sub-agents run silent + one-shot: no streaming UI, no auto-compact, no MCP readiness
            // gate.
            streaming: false,
            auto_compact: false,
            context_window: 0,
            mcp_grace: std::time::Duration::ZERO,
            system_prompt_override: Some(sub_system_prompt),
        };
        // Snapshot the parent's cwd at spawn time. The sub-agent has no `/cd` of its own (no REPL)
        // so this `Arc` is effectively immutable, but giving the sub-agent its own `Arc` rather
        // than sharing the parent's prevents a parent `/cd` mid-sub-agent-turn from changing the
        // sub-agent's resolution mid-flight.
        let parent_path = parent_cwd
            .read()
            .map(|guard| guard.clone())
            .unwrap_or_else(|poisoned| poisoned.into_inner().clone());
        let sub_cwd: SharedCwd = Arc::new(RwLock::new(parent_path));
        let mut agent = Self::new(
            provider,
            tool_registry,
            session_manager,
            shared_permission,
            options,
            todo_list,
            shared_session_id,
            skills,
            memories,
            frontend,
            sub_cwd,
            Arc::clone(parent_roots),
            session_stats,
        );
        // Sub-agents share the parent's `SessionStats` Arc but own a child session row; only the
        // primary agent persists, so the parent-inclusive totals aren't stamped onto a child.
        agent.persist_session_stats = false;
        agent
    }

    /// Snapshot of the per-session counters used by `/status`. Called from the REPL on demand.
    pub fn session_stats_snapshot(&self) -> crate::stats::SessionStatsSnapshot {
        self.session_stats.snapshot()
    }

    /// Live context occupancy for `/status`: `(tokens_in_context, context_window)`.
    ///
    /// `tokens_in_context` is the total tokens of this agent's most recent provider round (all
    /// input tiers + output) = what the next request re-sends minus the new prompt; `0` before
    /// the first turn. It is per-`Agent`, so sub-agents are excluded; a sub-agent's *returned
    /// result* counts only insofar as it became a tool result in this agent's own context.
    /// `context_window` is the resolved window for the active model (`0` if unknown).
    pub fn context_usage(&self) -> (u64, u64) {
        (
            self.last_context_tokens
                .load(std::sync::atomic::Ordering::Relaxed),
            self.options.context_window,
        )
    }

    /// The reasoning-effort value this agent's provider will send on the wire, or `None` when it
    /// sends none. Used by the `/status` model block.
    pub fn resolved_effort(&self) -> Option<String> {
        self.provider.resolved_effort()
    }

    /// Fetch the account's rate-limit usage from the active provider, for the `/usage` command.
    /// `Ok(None)` when the provider has no per-account usage endpoint.
    pub async fn fetch_usage(&self) -> Result<Option<crate::provider::AccountUsage>> {
        self.provider.fetch_usage().await
    }

    /// Tell the agent that the conversation it holds was rewritten out from under it, as `/rewind`
    /// does. Both markers the agent keeps against message *positions* stop meaning anything and are
    /// cleared: the accepted-prefix length a rejection is measured back from, and the index where
    /// the world-state delta was last rendered.
    ///
    /// Leaving either stale is silently wrong rather than loud. A stale accepted-prefix makes the
    /// degrade-and-retry recovery compute an empty suspect window and quietly not fire; a stale
    /// world-state index makes `run_turn` believe it already told the model about a tool or MCP
    /// server whose announcement the rewind just deleted, so it never mentions it again.
    ///
    /// `compact_session` clears the same two inline, since it rewrites the conversation itself.
    pub async fn reset_conversation_markers(&self) {
        self.last_accepted_len
            .store(LAST_ACCEPTED_UNKNOWN, std::sync::atomic::Ordering::Relaxed);
        *self.last_rendered_world.write().await = None;
    }

    /// Turn on background tool calls and hand this agent the shared task registry.
    ///
    /// Called only on the primary agent, before the first turn. Sub-agents are deliberately never
    /// enabled: a sub-agent's session ends with the one turn that spawned it, so a task outliving
    /// that turn would have no conversation left to report into.
    pub fn enable_background(
        &mut self,
        tasks: crate::background::BackgroundTasks,
        max_tasks: usize,
    ) {
        self.tool_registry.enable_background();
        self.background_tasks = tasks;
        self.background_max_tasks = max_tasks;
    }

    /// The shared registry, for the REPL's `/tasks` command and its Ctrl+C handling.
    pub fn background_tasks(&self) -> crate::background::BackgroundTasks {
        self.background_tasks.clone()
    }

    /// This agent's session store, so a signal handler can record a terminal outcome without the
    /// REPL threading a second handle through every call site.
    pub fn session_manager(&self) -> SessionManager {
        self.session_manager.clone()
    }

    /// Point this agent's live context counter at an externally-owned atomic so the REPL prompt
    /// (constructed before the agent) can read the same value the agent writes after each turn.
    /// Safe to call only before the first turn; the primary REPL path uses it, sub-agents don't.
    pub fn set_context_tokens(&mut self, handle: Arc<std::sync::atomic::AtomicU64>) {
        self.last_context_tokens = handle;
    }

    /// Shared handle to the auto-refreshing skill cache. The REPL's `/skill <name>` dispatch reads
    /// from this so the agent's system prompt and the user-invocable list never diverge.
    pub fn skills(&self) -> &Arc<SkillCache> {
        &self.skills
    }

    /// Attach the MCP client manager so server-supplied `initialize` instructions can be injected
    /// into each turn's context block.
    pub fn set_mcp_manager(&mut self, manager: Arc<crate::mcp::McpClientManager>) {
        self.mcp_manager = Some(manager);
    }

    /// Per-turn MCP readiness gate. Applies to every turn (not just the
    /// first) so mid-session reconnects also gate cleanly. Awaits
    /// `grace` for Pending servers to finish connecting, then hands whatever is still not
    /// `Connected` to [`gate_on_required_servers`], which rejects the turn only if one of them is
    /// `required`.
    ///
    /// No-op when no MCP manager is attached (e.g. sub-agents).
    async fn await_mcp_ready(&self) -> Result<()> {
        let Some(manager) = self.mcp_manager.as_ref() else {
            return Ok(());
        };
        if manager.all_ready() {
            let not_ready = manager.enabled_not_connected().await;
            if not_ready.is_empty() {
                return Ok(());
            }
            return self.handle_mcp_not_ready(not_ready);
        }

        // Best-effort grace wait. We re-check readiness below regardless of whether
        // `await_settled` returned in time. The timeout result is intentionally discarded.
        let _ = tokio::time::timeout(self.options.mcp_grace, manager.await_settled()).await;

        let not_ready = manager.enabled_not_connected().await;
        if not_ready.is_empty() {
            return Ok(());
        }
        self.handle_mcp_not_ready(not_ready)
    }

    fn handle_mcp_not_ready(&self, not_ready: Vec<crate::mcp::NotConnected>) -> Result<()> {
        gate_on_required_servers(not_ready)
    }

    pub async fn run_turn(
        &self,
        session_id: &mut Option<Uuid>,
        messages: &mut Conversation,
        user_input: String,
        images: Vec<ImageSource>,
        cancellation: CancellationToken,
    ) -> Result<TurnOutcome> {
        // Gate on MCP readiness BEFORE touching session state / message history so a rejected turn
        // leaves no trace in the conversation.
        self.await_mcp_ready().await?;

        if session_id.is_none() {
            let id = self
                .session_manager
                .create_session(Some(cwd_snapshot(&self.cwd)))
                .await?;
            *session_id = Some(id);
            self.frontend
                .emit(FrontendEvent::SessionStarted { id })
                .await;
        }

        self.frontend.emit(FrontendEvent::TurnStarted).await;

        let sid = session_id.ok_or(MekaError::Config("session_id not set".into()))?;

        // Keep the shared session ID in sync so scratchpad tools can access it.
        *self.shared_session_id.write().await = Some(sid);

        // Auto-compact if the last turn's context occupancy exceeded the threshold fraction of the
        // context window. This runs between turns (not mid-tool-loop) so the stable base_messages
        // invariant is preserved.
        if self.options.auto_compact && self.options.context_window > 0 {
            let last_tokens = self
                .last_context_tokens
                .load(std::sync::atomic::Ordering::Relaxed);
            let threshold = self.options.context_window * AUTO_COMPACT_THRESHOLD_PERCENT / 100;
            if last_tokens > threshold && messages.len() > 1 {
                tracing::info!(
                    "auto-compacting: {} input tokens exceeds {}% of {} context window",
                    last_tokens,
                    AUTO_COMPACT_THRESHOLD_PERCENT,
                    self.options.context_window
                );
                tracing::info!("auto-compacting conversation");
                if let Err(error) = self.compact_session(session_id, messages).await {
                    tracing::warn!("auto-compact failed: {}", error);
                }
            }
        }

        let permission = self.shared_permission.get();

        let catalogue = self.tool_registry.tool_catalogue();
        let skills = self.skills.current().await;
        let memories = self.memories.current().await;
        let mcp_instructions = self
            .mcp_manager
            .as_ref()
            .map(|manager| manager.server_instructions())
            .unwrap_or_default();

        // Tools, skills, and MCP instructions all move mid-session, so they ride in the user
        // message rather than the system prompt. Render only what changed since the model was last
        // told; an unchanged session emits nothing here. Read before the block is built because the
        // block carries it.
        //
        // `world_state_rollback` is the snapshot as it was before this turn claimed to have
        // announced the change. A turn that fails early pops its user message (see the error arm at
        // the end of this function), and that message is the only place the announcement lives, so
        // the claim has to be withdrawn with it. Without this a server that connects during a turn
        // that then fails is never mentioned again for the rest of the session.
        //
        // Skipped entirely for sub-agents. They run on a `system_prompt_override` that already
        // lists their tools (`build_subagent_system_prompt`), and that prompt is built once at
        // spawn for a fixed tool set and permission, so there is nothing here for them to
        // learn and rendering it would bill every `spawn_agent` for a second copy of the
        // catalogue.
        // Read fresh each turn and rendered outside the world-state diff: running tasks are live
        // state, like the todo list, not a record of what the model has been told. Skipped entirely
        // when the `task_*` tools are unregistered, which is the default.
        let background_tasks =
            match (*session_id).filter(|_| context::background_index_is_live(&catalogue)) {
                Some(id) => self
                    .session_manager
                    .list_running_background_tasks(id)
                    .await
                    .unwrap_or_else(|error| {
                        tracing::warn!("failed to load background tasks for context: {}", error);
                        Vec::new()
                    }),
                None => Vec::new(),
            };

        let (world_state, world_state_rollback) = if self.options.system_prompt_override.is_some() {
            (String::new(), None)
        } else {
            // Read fresh rather than cached: a job can be added or cancelled by `meka schedule`,
            // by another attached client, or by the scheduler retiring a fired one-shot, none of
            // which pass through this agent.
            // Skipped outright when the tool that opens the index is not registered: without this
            // an installation with `[schedule] enabled = false` pays a database round trip on every
            // single turn for a section that will be discarded.
            let scheduled =
                match (*session_id).filter(|_| context::schedule_index_is_live(&catalogue)) {
                    Some(id) => self
                        .session_manager
                        .list_scheduled_jobs(id)
                        .await
                        .unwrap_or_else(|error| {
                            tracing::warn!("failed to load scheduled jobs for context: {}", error);
                            Vec::new()
                        }),
                    None => Vec::new(),
                };
            let current = context::WorldSnapshot::new(
                &catalogue,
                skills.as_slice(),
                memories.as_slice(),
                &mcp_instructions,
                &scheduled,
            );
            let mut last = self.last_rendered_world.write().await;
            // Treat a render that has scrolled out of the API window as never having happened. The
            // window keeps the last `context_messages` entries, so a render at index `i` is gone
            // once the conversation grows past `i + limit`. Rendering in full then puts a fresh
            // copy at the new tail, good for another window's worth of turns.
            let still_visible = last.as_ref().filter(|(_, rendered_at)| {
                world_state_still_visible(
                    *rendered_at,
                    messages.len(),
                    self.options.context_messages,
                )
            });
            let rendered = context::render_world_state(&current, still_visible.map(|(s, _)| s));
            // This turn's user message is about to be appended, so that is where the render lands.
            let previous = last.replace((current, messages.len()));
            (rendered, previous)
        };

        let augmented_input = {
            let todos = self.todo_list.read().await;
            let cwd_snapshot = cwd_snapshot(&self.cwd);
            let roots_snapshot = roots_snapshot(&self.roots);
            let block = context::build_turn_context(
                permission,
                &todos,
                &cwd_snapshot,
                &roots_snapshot,
                &world_state,
                Some(context::ContextBudget {
                    used: self
                        .last_context_tokens
                        .load(std::sync::atomic::Ordering::Relaxed),
                    window: self.options.context_window,
                    compact_at_percent: self
                        .options
                        .auto_compact
                        .then_some(AUTO_COMPACT_THRESHOLD_PERCENT),
                }),
                &background_tasks,
            );
            format!("{}\n\n{}", block, user_input)
        };
        // Build the user message once (text preamble + any input images) and reuse it for both the
        // in-memory append and every persist path below, so attached images survive resume.
        let user_message = Message::user_with_images(augmented_input, images);
        // Where this turn's additions begin, captured before the append so the user message (which
        // may carry the attached images) is inside the window a rejection can blame. Distinct from
        // `turn_start_len` below, which marks the start of the *loop's* additions and so excludes
        // it.
        let mut suspect_floor = messages.len();
        messages.append(user_message.clone());
        // Persist the user message eagerly, before the first provider call.  A crash
        // during the provider roundtrip would otherwise lose it from disk.  On transient
        // DB failure the lazy save path below retries; `user_eagerly_saved` suppresses
        // double-writes on the happy path.
        let user_event = crate::conversation::Event::Append(user_message.clone());
        let user_eagerly_saved = match self.session_manager.save_event(sid, &user_event).await {
            Ok(()) => true,
            Err(error) => {
                tracing::warn!(
                    "failed to persist user message eagerly: {}; falling back to lazy \
                     persist on the first provider response",
                    error,
                );
                false
            }
        };
        let system_prompt: Arc<str> = match &self.options.system_prompt_override {
            Some(prompt) => Arc::from(prompt.as_str()),
            None => Arc::from(context::build_system_prompt(
                self.options.sandboxed_shell,
                self.options.user_instructions.as_deref(),
            )),
        };

        // Proactive pre-send compaction. The reactive check at the top of the turn reads the
        // *previous* round's reported usage, so a turn whose own input jumps over the window (a
        // huge paste, a large tool result carried in) would be sent uncompacted and
        // hard-fail. Project this request locally (conversation + system prompt) and
        // compact before sending if it would cross the threshold. `estimate_messages`
        // under-reads (no tool schemas), so this is a floor that complements, not replaces,
        // the reactive check and the overflow recovery below.
        if self.options.auto_compact && self.options.context_window > 0 && messages.len() > 1 {
            let projected = crate::tokens::estimate_messages(messages.as_slice())
                .saturating_add(crate::tokens::estimate_text(&system_prompt));
            let threshold = self.options.context_window * AUTO_COMPACT_THRESHOLD_PERCENT / 100;
            if projected > threshold {
                tracing::info!(
                    "proactive compaction: projected {} input tokens exceeds {}% of {} window",
                    projected,
                    AUTO_COMPACT_THRESHOLD_PERCENT,
                    self.options.context_window
                );
                if let Err(error) = self.compact_session(session_id, messages).await {
                    tracing::warn!("proactive compaction failed: {}", error);
                }
            }
        }

        // Wrapped in `Arc` once so the no-tool-progress branch below can share it with a cheap
        // `Arc::clone` instead of a deep `Vec` clone on every loop iteration. `mut` so an overflow
        // recovery (compact-and-retry, below) can rebuild it from the compacted conversation.
        let mut base_messages: Arc<[Message]> = Arc::from(truncate_messages_for_context(
            messages.as_slice(),
            self.options.context_messages,
        ));
        let mut turn_start_len = messages.len();
        // Bounds the emergency compact-and-retry on a `ContextOverflow` so a request that stays too
        // large after one compaction fails cleanly instead of looping.
        let mut overflow_retries = 0u32;
        // Bounds the degrade-and-retry on a `MekaError::InvalidRequest`, in the same spirit.
        let mut repairs_used = 0u32;
        // A repair applied to the in-memory conversation but not yet proven good by a 2xx, so not
        // yet persisted. Dropped back into the log on success, undone on a second rejection.
        let mut pending_repair: Option<crate::conversation::Event> = None;

        let mut user_saved = user_eagerly_saved;
        // Set once we've nudged the model for a user-visible response this turn, so the recovery
        // fires at most once and can't loop (see `should_nudge_thinking_only`).
        let mut thinking_only_nudged = false;
        // Accumulate token usage across every provider call within this turn so the per-turn
        // display reflects the whole turn (including tool-execution loops), not just the final
        // round-trip.
        let mut turn_usage = crate::provider::TokenUsage::default();

        let result: Result<TurnOutcome> = 'turn: {
            loop {
                if cancellation.is_cancelled() {
                    break 'turn Err(MekaError::Interrupted);
                }
                // Bail out if the frontend has noticed its client went away (e.g. ACP stdio
                // disconnect). No point burning more provider tokens for an audience that won't see
                // the output. REPL frontends report `false` here, so this is a no-op for them.
                if self.frontend.client_disconnected() {
                    break 'turn Err(MekaError::Interrupted);
                }

                // Conversation length behind this request, stamped onto `last_accepted_len` when
                // the provider takes it.
                let sent_len = messages.len();

                let api_messages: Arc<[Message]> = if messages.len() > turn_start_len {
                    let mut combined = base_messages.to_vec();
                    combined.extend_from_slice(&messages.as_slice()[turn_start_len..]);
                    Arc::from(combined)
                } else {
                    Arc::clone(&base_messages)
                };

                // Recompute the active tool set every iteration so a `load_tool` call earlier in
                // this turn becomes visible to the model on the very next request, without
                // mutating any registry state. Append-only growth keeps the tools array's cache
                // prefix stable.
                //
                // Read from events (not the materialized slice) so the deferred-tool snapshot
                // stored on `Event::CompactBoundary` survives across compaction; otherwise tools
                // the model loaded pre-compaction would silently drop out of the active set on the
                // next turn.
                let loaded =
                    crate::conversation::extract_loaded_tool_names_from_events(messages.events());
                let tools: Arc<[ToolDefinition]> =
                    Arc::from(self.tool_registry.definitions_active_with_loaded(&loaded));

                // Streaming and blocking paths converge on `(Message, StopReason, TokenUsage)`. The
                // blocking provider call surfaces notices in its return tuple (no event channel);
                // we forward them to the frontend here so the user sees the same advisories the
                // streaming path emits inline via `StreamEvent::Notice`.
                let call_result: Result<(Message, StopReason, crate::provider::TokenUsage)> =
                    if self.options.streaming {
                        self.run_streaming(
                            Arc::clone(&system_prompt),
                            api_messages,
                            tools,
                            cancellation.clone(),
                        )
                        .await
                    } else {
                        // Non-streaming is fully atomic (nothing is visible until this returns
                        // `Ok`), so `content_started` is always `false` here — every retryable
                        // failure is retried up to the cap regardless of prior attempts.
                        let mut retries = 0u32;
                        loop {
                            match self
                                .provider
                                .complete(&system_prompt, &api_messages, &tools)
                                .await
                            {
                                Ok((message, stop_reason, usage, notices)) => {
                                    for notice in notices {
                                        self.frontend.emit(FrontendEvent::Notice(notice)).await;
                                    }
                                    break Ok((message, stop_reason, usage));
                                }
                                Err(error) => {
                                    match should_retry_provider_error(&error, false, retries) {
                                        Some(delay) => {
                                            retries += 1;
                                            tracing::warn!(
                                                "provider request failed transiently (attempt \
                                                 {}/{}), retrying in {:?}: {}",
                                                retries,
                                                crate::provider::retry::MAX_PROVIDER_RETRIES,
                                                delay,
                                                error
                                            );
                                            tokio::select! {
                                                _ = tokio::time::sleep(delay) => {}
                                                _ = cancellation.cancelled() => break Err(MekaError::Interrupted),
                                            }
                                        }
                                        None => break Err(error),
                                    }
                                }
                            }
                        }
                    };

                let (mut assistant_message, stop_reason, usage) = match call_result {
                    Ok(value) => value,
                    // Context overflow despite the proactive check (the local estimate under-counts
                    // tool schemas). Compact once and retry rather than fail the turn; the
                    // compacted conversation already holds this turn's tool
                    // results, so the retry rebuilds `base_messages` from it
                    // and re-sends.
                    Err(MekaError::ContextOverflow(message))
                        if self.options.auto_compact
                            && self.options.context_window > 0
                            && messages.len() > 1
                            && overflow_retries < MAX_OVERFLOW_RETRIES =>
                    {
                        overflow_retries += 1;
                        tracing::warn!(
                            "provider reported context overflow; compacting and retrying ({})",
                            message
                        );
                        if let Err(compact_error) = self.compact_session(session_id, messages).await
                        {
                            tracing::warn!("emergency compaction failed: {}", compact_error);
                            break 'turn Err(MekaError::ContextOverflow(message));
                        }
                        base_messages = Arc::from(truncate_messages_for_context(
                            messages.as_slice(),
                            self.options.context_messages,
                        ));
                        turn_start_len = messages.len();
                        // Compaction rewrote everything before this point, so the old floor no
                        // longer marks anything.
                        suspect_floor = messages.len();
                        continue;
                    }
                    // The retry after a repair was refused too, so the repair was not the fix.
                    // Undo it and report what the provider actually said, leaving the conversation
                    // byte-identical to before the attempt: the cost of guessing wrong has to be
                    // one round trip, never a destroyed tool result.
                    Err(MekaError::InvalidRequest(message)) if pending_repair.is_some() => {
                        if messages.pop_repair() {
                            tracing::warn!(
                                "degrading this turn's content did not satisfy the provider; \
                                 restored it unchanged"
                            );
                        }
                        break 'turn Err(MekaError::InvalidRequest(message));
                    }
                    // The provider refused the request as malformed. Retrying it unchanged is
                    // pointless (a 400 is deterministic on the body), and failing outright is worse
                    // than it looks: the content is already committed to the session, so every
                    // later request carries it and dies the same way, leaving the session
                    // unusable. Strip the non-text content appended since the last accepted request
                    // and try once more, telling the model what happened via the tool result it is
                    // already equipped to read.
                    Err(MekaError::InvalidRequest(message))
                        if repairs_used < MAX_REQUEST_REPAIRS =>
                    {
                        let suspect_start = match self
                            .last_accepted_len
                            .load(std::sync::atomic::Ordering::Relaxed)
                        {
                            LAST_ACCEPTED_UNKNOWN => suspect_floor,
                            accepted => accepted.min(messages.len()),
                        };
                        let Some(degraded) = degrade_rejected_content(
                            &messages.as_slice()[suspect_start..],
                            &message,
                        ) else {
                            // Nothing to strip, so this is not a content problem: a `max_tokens`
                            // over the model's ceiling, an unknown header, a bad `tool_choice`.
                            break 'turn Err(MekaError::InvalidRequest(message));
                        };
                        repairs_used += 1;
                        let replaced_count = messages.len() - suspect_start;
                        tracing::warn!(
                            "provider rejected the request; degrading {} message(s) appended since \
                             the last accepted one and retrying ({})",
                            replaced_count,
                            message,
                        );
                        self.frontend
                            .emit(FrontendEvent::Notice(crate::provider::Notice::warn(
                                format!(
                                    "provider rejected content in this turn; retrying without it: \
                                     {}",
                                    elide_reason(&message)
                                ),
                            )))
                            .await;
                        pending_repair = Some(messages.replace_tail(replaced_count, degraded));
                        base_messages = Arc::from(truncate_messages_for_context(
                            messages.as_slice(),
                            self.options.context_messages,
                        ));
                        turn_start_len = messages.len();
                        continue;
                    }
                    Err(error) => break 'turn Err(error),
                };

                // The provider accepted this body, so everything in it is known-good and only what
                // comes after can be blamed for a later rejection.
                self.last_accepted_len
                    .store(sent_len, std::sync::atomic::Ordering::Relaxed);

                // Total of all tiers including output = everything in context as of this exchange,
                // which is what the next request re-sends (minus the new user prompt). Summing the
                // input tiers + output (Claude reports cached tokens in separate fields) is the
                // true occupancy and what the `/status` gauge and auto-compact
                // threshold read.
                self.last_context_tokens.store(
                    usage
                        .input_tokens
                        .saturating_add(usage.cache_creation_input_tokens)
                        .saturating_add(usage.cache_read_input_tokens)
                        .saturating_add(usage.output_tokens),
                    std::sync::atomic::Ordering::Relaxed,
                );
                turn_usage.input_tokens =
                    turn_usage.input_tokens.saturating_add(usage.input_tokens);
                turn_usage.output_tokens =
                    turn_usage.output_tokens.saturating_add(usage.output_tokens);
                turn_usage.cache_creation_input_tokens = turn_usage
                    .cache_creation_input_tokens
                    .saturating_add(usage.cache_creation_input_tokens);
                turn_usage.cache_read_input_tokens = turn_usage
                    .cache_read_input_tokens
                    .saturating_add(usage.cache_read_input_tokens);

                if !user_saved {
                    let user_event = crate::conversation::Event::Append(user_message.clone());
                    if let Err(error) = self.session_manager.save_event(sid, &user_event).await {
                        break 'turn Err(error);
                    }
                    user_saved = true;
                }

                // Persist the repair this response just vindicated, after the user message is
                // guaranteed on disk and before anything else is appended. `Event::Repair` replaces
                // the *trailing* messages on replay, so any row written between the messages it
                // repairs and the repair itself would be swallowed instead.
                if let Some(event) = pending_repair.take()
                    && let Err(error) = self.session_manager.save_event(sid, &event).await
                {
                    // The in-memory conversation is repaired either way, so this turn still
                    // completes; the cost is that a resume re-reads the rejected content and pays
                    // one more round trip to heal it again.
                    tracing::warn!("failed to persist content repair: {}", error);
                }

                if cancellation.is_cancelled() {
                    // Interrupted mid-stream. Persist the partial assistant text so it survives
                    // resume instead of being discarded, but drop any `tool_use` blocks first: no
                    // tools run on an interrupt, so a persisted `tool_use` would be orphaned (no
                    // matching `tool_result`) and the provider would reject the next request. Only
                    // persist when text actually streamed; a partial with no text (interrupted
                    // before any output, or mid-thinking) has nothing worth restoring.
                    let partial = assistant_message.without_tool_use();
                    if partial
                        .content
                        .iter()
                        .any(|block| matches!(block, ContentBlock::Text { .. }))
                    {
                        messages.append(partial.clone());
                        if let Err(error) = self
                            .session_manager
                            .save_events_atomic(sid, vec![crate::conversation::Event::Append(
                                partial,
                            )])
                            .await
                        {
                            tracing::error!(
                                "failed to persist interrupted partial assistant message: {}",
                                error
                            );
                        }
                    }
                    break 'turn Err(MekaError::Interrupted);
                }

                // Run tools based on the *presence* of tool-call blocks, not the reported stop
                // reason: stop reasons are advisory and providers sometimes mislabel a tool turn as
                // a plain end, but any tool call the model made must be answered with a result or
                // the next request is invalid. Only complete tool calls reach the content blocks,
                // so executing whatever is present is safe.
                let has_tool_calls = assistant_message
                    .content
                    .iter()
                    .any(|block| matches!(block, ContentBlock::ToolUse { .. }));
                let has_visible_text = has_visible_text(&assistant_message.content);

                // A turn that makes no tool call and produces no visible text (a thinking-only
                // response, or an empty one) would otherwise end silently. Mirror Claude Code's
                // `query_thinking_only_response`: record the turn, then nudge the model once for a
                // user-visible response and continue. The nudge is appended *after* the assistant
                // message so the thinking-only turn isn't the trailing assistant message - Claude
                // strips trailing thinking blocks only from the last assistant turn, so keeping it
                // non-last preserves its thinking block on the retry request.
                if should_nudge_thinking_only(
                    has_tool_calls,
                    has_visible_text,
                    &stop_reason,
                    thinking_only_nudged,
                ) {
                    messages.append(assistant_message.clone());
                    let assistant_event =
                        crate::conversation::Event::Append(assistant_message.clone());
                    let nudge = Message {
                        role: Role::User,
                        content: vec![ContentBlock::Text {
                            text: THINKING_ONLY_NUDGE.to_string(),
                        }],
                    };
                    let nudge_event = crate::conversation::Event::Append(nudge.clone());
                    if let Err(error) = self
                        .session_manager
                        .save_events_atomic(sid, vec![assistant_event, nudge_event])
                        .await
                    {
                        break 'turn Err(error);
                    }
                    messages.append(nudge);
                    thinking_only_nudged = true;
                    tracing::info!(
                        "thinking-only response (no visible text, stop_reason {:?}); nudging once",
                        stop_reason,
                    );
                    continue;
                }

                // No tool call and no visible text, and the nudge above didn't fire (already used
                // this turn, or a stop reason with its own handling such as refusal / max tokens).
                // Surface a stand-in in the assistant's place and persist it so the message is
                // non-empty: an empty content array is invalid on the next request and breaks
                // resume, and a silent turn leaves the user with nothing.
                if !has_tool_calls && !has_visible_text {
                    let notice = empty_turn_notice(&stop_reason);
                    self.frontend
                        .emit(FrontendEvent::AssistantTextDelta(notice.clone()))
                        .await;
                    assistant_message
                        .content
                        .push(ContentBlock::Text { text: notice });
                }

                // Append in memory now so the next iteration sees the full state; defer the DB save
                // to the branches below (atomic with results on the tool path, standalone
                // otherwise).
                messages.append(assistant_message.clone());
                let assistant_event = crate::conversation::Event::Append(assistant_message.clone());

                if has_tool_calls {
                    // Surface a provider that mislabeled the stop reason - the bug this presence
                    // check guards against.
                    if !matches!(stop_reason, StopReason::ToolUse) {
                        tracing::warn!(
                            "assistant message carries tool calls but stop_reason is {:?}; executing them anyway so each tool call gets a result",
                            stop_reason,
                        );
                    }

                    let mut tool_results = self
                        .execute_tool_calls(&assistant_message, &loaded, cancellation.clone())
                        .await;

                    if let Err(error) = crate::tools::scratchpad::save_explicit_scratchpad_results(
                        &self.session_manager,
                        sid,
                        &assistant_message,
                        &mut tool_results,
                    )
                    .await
                    {
                        tracing::warn!("failed to save explicit scratchpad results: {}", error);
                    }

                    // Take the per-turn hints. This both snapshots them for the call below and
                    // clears them, so a long session doesn't accumulate entries for tool calls that
                    // already ran. No clone needed.
                    let hints_snapshot = std::mem::take(&mut *self.scratchpad_hints.write().await);
                    if let Err(error) = crate::tools::scratchpad::persist_oversized_results(
                        &self.session_manager,
                        sid,
                        &assistant_message,
                        &mut tool_results,
                        &hints_snapshot,
                    )
                    .await
                    {
                        tracing::warn!("failed to persist oversized tool results: {}", error);
                    }

                    let result_message = Message {
                        role: Role::User,
                        content: tool_results,
                    };

                    // Save assistant + tool-results together in one transaction. Both rows commit
                    // or neither does: no dangling assistant-with-tool_use that the provider would
                    // reject on the next iteration.
                    let result_event = crate::conversation::Event::Append(result_message.clone());
                    if let Err(error) = self
                        .session_manager
                        .save_events_atomic(sid, vec![assistant_event, result_event])
                        .await
                    {
                        break 'turn Err(error);
                    }

                    messages.append(result_message);
                } else {
                    // No tool calls: the assistant message stands alone and ends the turn. Save it
                    // before breaking so the persistent log includes it.
                    if let Err(error) = self
                        .session_manager
                        .save_events_atomic(sid, vec![assistant_event])
                        .await
                    {
                        break 'turn Err(error);
                    }
                    break 'turn match stop_reason {
                        StopReason::MaxTokens => Ok(TurnOutcome::MaxTokens),
                        StopReason::Refusal(text) if !text.is_empty() => {
                            Ok(TurnOutcome::Refusal(text))
                        }
                        // An empty refusal body carries no text, so fall back to the assistant
                        // message's text (the model's own refusal, or the stand-in above).
                        StopReason::Refusal(_) => {
                            Ok(TurnOutcome::Refusal(assistant_message.text_content()))
                        }
                        _ => Ok(TurnOutcome::EndTurn),
                    };
                }
            }
        };

        if result.is_ok() {
            // Roll the turn into the session-level counters surfaced by `/status`. Done here (not
            // inside the inner loop) so a single `/status` reading reflects whole turns, not
            // partial state.
            self.session_stats.record_turn(&turn_usage);
            // Persist the cumulative counters onto the session row so `/status` survives resume.
            // Best-effort: a DB hiccup must not fail the turn. Only the primary agent writes; a
            // sub-agent shares the parent's `SessionStats` (rolling its usage into the parent's
            // totals) but owns a child session row, so letting it write would stamp the
            // parent-inclusive totals onto the child.
            if self.persist_session_stats
                && let Err(error) = self
                    .session_manager
                    .save_session_stats(sid, &self.session_stats.snapshot())
                    .await
            {
                tracing::warn!("failed to persist session stats: {}", error);
            }
            self.frontend
                .emit(FrontendEvent::TokenUsage(turn_usage))
                .await;
            self.frontend.emit(FrontendEvent::TurnFinished).await;
        }

        match &result {
            Err(MekaError::Interrupted) if !user_saved => {
                let user_event = crate::conversation::Event::Append(user_message.clone());
                if let Err(error) = self.session_manager.save_event(sid, &user_event).await {
                    tracing::error!("failed to save user message on interruption: {}", error);
                }
            }
            Err(error) if !matches!(error, MekaError::Interrupted) && !user_saved => {
                messages.pop_unsaved();
                // The popped message carried this turn's world-state announcement, so put the
                // snapshot back to what the model has actually seen. The next turn then re-renders
                // the change rather than assuming it was already delivered.
                *self.last_rendered_world.write().await = world_state_rollback;
            }
            _ => {}
        }

        result
    }

    /// Streaming provider call with bounded retry-with-backoff on transient failures
    /// ([`MekaError::RetryableProvider`]) — 429/5xx (including Anthropic's 529 "overloaded") and,
    /// for Claude, a mid-stream `event: error` of a retryable type. Each attempt runs
    /// [`Self::run_streaming_attempt`] fresh (new channel, new spawned task, all accumulator state
    /// reinitialized); retries only fire when that attempt reports `content_started == false`, i.e.
    /// nothing has been forwarded to the frontend yet this attempt — retrying after the user has
    /// already seen partial output would duplicate or corrupt what's on screen. Nothing is
    /// persisted to the session DB until the whole turn's result is resolved (see `run_turn`),
    /// so a discarded attempt never leaves a partial write behind.
    async fn run_streaming(
        &self,
        system_prompt: Arc<str>,
        messages: Arc<[Message]>,
        tools: Arc<[ToolDefinition]>,
        cancellation: CancellationToken,
    ) -> Result<(Message, StopReason, crate::provider::TokenUsage)> {
        let mut retries = 0u32;
        loop {
            let mut content_started = false;
            match self
                .run_streaming_attempt(
                    Arc::clone(&system_prompt),
                    Arc::clone(&messages),
                    Arc::clone(&tools),
                    cancellation.clone(),
                    &mut content_started,
                )
                .await
            {
                Ok(value) => return Ok(value),
                Err(error) => match should_retry_provider_error(&error, content_started, retries) {
                    Some(delay) => {
                        retries += 1;
                        tracing::warn!(
                            "provider stream failed transiently (attempt {}/{}), retrying in \
                             {:?}: {}",
                            retries,
                            crate::provider::retry::MAX_PROVIDER_RETRIES,
                            delay,
                            error
                        );
                        tokio::select! {
                            _ = tokio::time::sleep(delay) => {}
                            _ = cancellation.cancelled() => return Err(MekaError::Interrupted),
                        }
                    }
                    None => return Err(error),
                },
            }
        }
    }

    /// A single streaming attempt: spawns `provider.stream(...)`, drains its `StreamEvent`s into a
    /// `Message`. `content_started` is set the instant anything user-visible is forwarded to the
    /// frontend — see [`Self::run_streaming`], which reads it back to decide whether a failure is
    /// safe to retry.
    async fn run_streaming_attempt(
        &self,
        system_prompt: Arc<str>,
        messages: Arc<[Message]>,
        tools: Arc<[ToolDefinition]>,
        cancellation: CancellationToken,
        content_started: &mut bool,
    ) -> Result<(Message, StopReason, crate::provider::TokenUsage)> {
        // Bounded so a provider streaming faster than the renderer consumes can't grow memory
        // without limit. 1024 is far above any realistic in-flight backlog, so backpressure
        // effectively never engages.
        let (event_sender, mut event_receiver) = mpsc::channel::<StreamEvent>(1024);

        let provider = Arc::clone(&self.provider);
        let cancellation_clone = cancellation.clone();

        let stream_handle = tokio::spawn(async move {
            provider
                .stream(
                    &system_prompt,
                    &messages,
                    &tools,
                    event_sender,
                    cancellation_clone,
                )
                .await
        });

        let mut content_blocks: Vec<ContentBlock> = Vec::new();
        let mut current_text = String::new();
        let mut current_thinking = String::new();
        let mut current_tool_id = String::new();
        let mut current_tool_name = String::new();
        let mut current_tool_input_json = String::new();
        let mut stop_reason = StopReason::EndTurn;
        let mut token_usage = crate::provider::TokenUsage::default();

        while let Some(event) = event_receiver.recv().await {
            match event {
                StreamEvent::ThinkingDelta(text) => {
                    current_thinking.push_str(&text);
                }
                StreamEvent::ThinkingComplete { signature } => {
                    let content = std::mem::take(&mut current_thinking);
                    // Keep the block whenever it carries replayable state: visible text and/or a
                    // signature. Under `redact-thinking` the text is empty but the signature must
                    // survive to continue the reasoning chain on the next turn.
                    if !content.is_empty() || signature.is_some() {
                        if !content.is_empty() {
                            *content_started = true;
                            self.frontend
                                .emit(FrontendEvent::ThinkingBlock {
                                    content: content.clone(),
                                    signature: signature.clone(),
                                })
                                .await;
                        }
                        content_blocks.push(ContentBlock::Thinking {
                            thinking: content,
                            signature,
                        });
                    }
                }
                StreamEvent::RedactedThinking { data } => {
                    *content_started = true;
                    self.frontend
                        .emit(FrontendEvent::ThinkingBlock {
                            content: "[redacted thinking]".to_string(),
                            signature: None,
                        })
                        .await;
                    content_blocks.push(ContentBlock::RedactedThinking { data });
                }
                StreamEvent::TextDelta(text) => {
                    *content_started = true;
                    current_text.push_str(&text);
                    self.frontend
                        .emit(FrontendEvent::AssistantTextDelta(text))
                        .await;
                }
                StreamEvent::ToolUseStart { id, name } => {
                    if !current_text.is_empty() {
                        content_blocks.push(ContentBlock::Text {
                            text: std::mem::take(&mut current_text),
                        });
                    }
                    current_tool_id = id;
                    current_tool_name = name;
                    current_tool_input_json.clear();
                }
                StreamEvent::ToolInputDelta(delta) => {
                    current_tool_input_json.push_str(&delta);
                }
                StreamEvent::ToolUseEnd { input } => {
                    *content_started = true;
                    let schema = self
                        .tool_registry
                        .get(&current_tool_name)
                        .map(|t| t.definition().parameters);
                    let display_summary = crate::render::resolve_primary_param(
                        &current_tool_name,
                        &input,
                        schema.as_ref(),
                    );
                    self.frontend
                        .emit(FrontendEvent::ToolCallStarted {
                            id: current_tool_id.clone(),
                            name: current_tool_name.clone(),
                            input: input.clone(),
                            display_summary,
                        })
                        .await;

                    content_blocks.push(ContentBlock::ToolUse {
                        id: std::mem::take(&mut current_tool_id),
                        name: std::mem::take(&mut current_tool_name),
                        input,
                    });
                    current_tool_input_json.clear();
                }
                StreamEvent::ToolCallRejected { id, name, reason } => {
                    // A malformed tool-call arrived (bad JSON). Emit a `ToolUse` block with a
                    // sentinel marker so the shape of the assistant message stays valid for the API
                    // round-trip, but `resolve_and_execute_tool` sees the marker and surfaces an
                    // error back to the model rather than running the tool on a silently-empty
                    // argument object.
                    *content_started = true;
                    let marker_input = serde_json::json!({
                        crate::provider::INVALID_TOOL_ARGS_MARKER: reason,
                    });
                    let schema = self
                        .tool_registry
                        .get(&name)
                        .map(|t| t.definition().parameters);
                    let display_summary =
                        crate::render::resolve_primary_param(&name, &marker_input, schema.as_ref());
                    self.frontend
                        .emit(FrontendEvent::ToolCallStarted {
                            id: id.clone(),
                            name: name.clone(),
                            input: marker_input.clone(),
                            display_summary,
                        })
                        .await;
                    content_blocks.push(ContentBlock::ToolUse {
                        id,
                        name,
                        input: marker_input,
                    });
                    current_tool_id.clear();
                    current_tool_name.clear();
                    current_tool_input_json.clear();
                }
                StreamEvent::MessageEnd {
                    stop_reason: reason,
                } => {
                    stop_reason = reason;
                }
                StreamEvent::Usage(usage) => {
                    // Merge rather than overwrite: Anthropic streams the input/cache tiers on
                    // `message_start` and the output on `message_delta`, so last-event-wins would
                    // drop the input count. The non-zero merge keeps each tier from whichever event
                    // reported it.
                    token_usage.merge_stream(&usage);
                }
                StreamEvent::Notice(notice) => {
                    // Forward provider-side advisories (image redaction, etc.) to the frontend
                    // alongside the stream. Emitted inline so the user sees them in order with the
                    // assistant text that follows.
                    *content_started = true;
                    self.frontend.emit(FrontendEvent::Notice(notice)).await;
                }
                StreamEvent::Error(error) => {
                    // Log only; deliberately don't return here. Every producer of this event sends
                    // it immediately before its own typed `Err` return (see
                    // `drive_claude_sse_stream`/`drive_responses_sse_stream`), so the task finishes
                    // right after this, the channel closes, and this loop ends naturally. The
                    // `stream_handle.await` join below then surfaces the ORIGINAL typed error
                    // (e.g. `RetryableProvider` vs. plain `Provider`) — returning a generic
                    // `MekaError::Provider(error)` here would discard that classification before
                    // `run_streaming`'s retry logic ever saw it.
                    tracing::error!("stream error: {}", error);
                }
            }
        }

        if !current_text.is_empty() {
            content_blocks.push(ContentBlock::Text { text: current_text });
        }

        match stream_handle.await {
            Ok(Ok(())) => {}
            Ok(Err(MekaError::Interrupted)) => {
                // Interrupted. Fall through to return partial content. The caller detects
                // interruption via the cancellation token.
            }
            Ok(Err(error)) => return Err(error),
            Err(join_error) => {
                return Err(MekaError::Provider(format!(
                    "stream task panicked: {}",
                    join_error
                )));
            }
        }

        let message = Message {
            role: Role::Assistant,
            content: content_blocks,
        };

        Ok((message, stop_reason, token_usage))
    }

    /// `loaded` is the turn's active-tool set, used only to tell a call made against a schema the
    /// model has actually seen from one made blind; see [`Self::schema_advisory`].
    async fn execute_tool_calls(
        &self,
        assistant_message: &Message,
        loaded: &[String],
        cancellation: CancellationToken,
    ) -> Vec<ContentBlock> {
        // Emit tool-call indicators in source order. The streaming path already emitted these as
        // `ToolUseEnd` events; this loop only fires for the blocking provider path. Serial so
        // concurrent execution below can't interleave indicators.
        let mut planned: Vec<(String, String, serde_json::Value)> = Vec::new();
        for block in &assistant_message.content {
            if let ContentBlock::ToolUse { id, name, input } = block {
                if !self.options.streaming {
                    let schema = self
                        .tool_registry
                        .get(name)
                        .map(|t| t.definition().parameters);
                    let display_summary =
                        crate::render::resolve_primary_param(name, input, schema.as_ref());
                    self.frontend
                        .emit(FrontendEvent::ToolCallStarted {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                            display_summary,
                        })
                        .await;
                }
                planned.push((id.clone(), name.clone(), input.clone()));
            }
        }

        // Dispatch concurrently. `join_all` preserves input ordering so the i-th output corresponds
        // to the i-th planned call.
        let futures = planned.iter().map(|(id, name, input)| {
            self.resolve_and_execute_tool(
                id.as_str(),
                name.as_str(),
                input,
                loaded,
                cancellation.clone(),
            )
        });
        let outputs = futures::future::join_all(futures).await;

        // Serial pass to accumulate scratchpad hints, emit per-tool completion events in source
        // order, build ToolResult blocks, and emit a single TodoListUpdated event if any `todo`
        // call landed and actually changed the rendered state.
        let mut results = Vec::with_capacity(planned.len());
        let mut todo_fired = false;
        for ((id, name, _), output) in planned.into_iter().zip(outputs) {
            if name == "todo" {
                todo_fired = true;
            }
            if let Some(hint) = output.scratchpad_hint.clone() {
                self.scratchpad_hints.write().await.insert(id.clone(), hint);
            }
            // Notify the frontend of completion BEFORE building the ToolResult content block so ACP
            // `tool_call_update` notifications arrive before the next assistant turn's text starts
            // streaming.
            self.frontend
                .emit(FrontendEvent::ToolCallCompleted {
                    id: id.clone(),
                    name: name.clone(),
                    is_error: output.is_error,
                    content: output.content.clone(),
                    metadata: output.frontend_metadata.clone(),
                })
                .await;
            results.push(ContentBlock::ToolResult {
                tool_use_id: id,
                content: output.content,
                is_error: output.is_error,
            });
        }
        if todo_fired {
            let state = self.todo_list.read().await.clone();
            // Suppress re-renders for reads and rewrites that change nothing. Drop the guard before
            // awaiting the emit.
            let should_emit = {
                let mut last = self.last_rendered_todo.write().await;
                let changed = last.as_ref() != Some(&state);
                if changed {
                    *last = Some(state.clone());
                }
                // An empty list renders nothing, so emitting it would be a no-op event that also
                // corrupts REPL spacing; require something to show.
                !state.items.is_empty() && changed
            };
            if should_emit {
                self.frontend
                    .emit(FrontendEvent::TodoListUpdated {
                        title: state.title,
                        items: state.items,
                    })
                    .await;
            }
        }

        results
    }

    async fn resolve_and_execute_tool(
        &self,
        tool_call_id: &str,
        name: &str,
        input: &serde_json::Value,
        loaded: &[String],
        cancellation: CancellationToken,
    ) -> crate::tools::ToolOutput {
        // If the stream layer couldn't parse this tool call's JSON arguments, it marked the input
        // with a sentinel. Bail out with an error so the model sees the parse failure instead of us
        // silently invoking the tool on a default-filled object.
        if let Some(reason) = input
            .get(crate::provider::INVALID_TOOL_ARGS_MARKER)
            .and_then(|v| v.as_str())
        {
            return crate::tools::ToolOutput::text(format!("Tool call rejected: {}", reason), true);
        }

        let Some(tool) = self.tool_registry.get(name) else {
            // A tool from a server that never connected was never registered, so it lands here.
            // Saying "unknown" would be false - it exists and is unreachable - and would teach the
            // agent to stop asking for a capability that may be seconds from returning.
            //
            // Asks the registry rather than `self.mcp_manager`: a sub-agent has no manager of its
            // own (see `new_subagent`) but its registry does, and it deserves the same answer.
            if let Some(manager) = self.tool_registry.mcp_manager()
                && let Some(reason) = manager.unavailable_tool_reason(name).await
            {
                return crate::tools::ToolOutput::text(reason, true);
            }
            // Namespaced MCP names are long and easy to mangle, and the commonest slip is dropping
            // the `mcp__<server>__` prefix entirely, which no amount of re-reading the catalogue
            // fixes if the reply is a bare "unknown".
            let registered = self.tool_registry.registered_tool_names();
            let hint = crate::tools::did_you_mean_hint(name, registered.iter().map(String::as_str));
            return crate::tools::ToolOutput::text(
                format!("Unknown tool: '{}'.{}", name, hint),
                true,
            );
        };

        // Read the current permission once, at the enforcement site, so a permission cycle via
        // Shift+Tab during dispatch can't leave us acting on a stale snapshot captured earlier in
        // the loop.
        let permission = self.shared_permission.get();
        let required = self
            .tool_registry
            .required_permission_for(name)
            .unwrap_or_else(|| tool.required_permission());
        if !permission.allows(required) {
            return crate::tools::ToolOutput::text(
                format!(
                    "Permission denied: '{}' requires `{}` permission, current level is `{}`. \
                     Ask the user to run `/permission {}` (or press Shift+Tab) to enable it.",
                    name, required, permission, required
                ),
                true,
            );
        }

        // Scope the id across both dispatch paths, so a tool that has to correlate itself with the
        // client's view of this call -- `spawn_agent`, routing its sub-agent's activity back into
        // the tool call already on screen -- can read it without every other tool's signature
        // growing a parameter it ignores.
        //
        // The session id rides alongside for the same reason, so an MCP `tools/call` can name the
        // conversation it came from. `run_turn` populates `shared_session_id` before the tool loop,
        // so this is only `None` on paths that never established a session.
        let session_id = *self.shared_session_id.read().await;
        let schema = tool.definition().parameters;

        // Refused before anything runs, rather than read as absent. `background` and `scratchpad`
        // are consumed by this loop, not by the tool, and each decides what the call *does*, so a
        // wrong type that we shrugged off would be a silent no-op the model has no way to notice:
        // a detach that quietly blocked, or output it asked to keep that was never kept.
        if let Some(complaint) = crate::tools::meka_parameter_error(input, &schema) {
            return crate::tools::ToolOutput::text(complaint, true);
        }

        // `background` is meka's own, spliced into the schema by the registry and consumed here, so
        // it is taken out of the arguments before any tool (least of all a remote MCP server) sees
        // a key it never advertised. The schema goes along to settle whose parameter it is: a tool
        // that declares `background` itself never received the splice and keeps its argument.
        let (input, detach) = crate::tools::take_background_flag(input, &schema);
        let input = &input;

        // Approval resolves *before* a detach, never inside it. A prompt surfacing minutes after
        // the turn that caused it, with nothing on screen to explain it, is worse than the
        // round trip it would save.
        if permission == crate::permission::Permission::Ask
            && let Some(denial) = self
                .request_approval(name, input, &schema, &cancellation)
                .await
        {
            return denial;
        }

        let mut output = if detach {
            self.start_background_call(&tool, tool_call_id, name, input, session_id)
                .await
        } else {
            let dispatch = crate::tools::with_tool_call_id(tool_call_id.to_string(), async move {
                Self::run_tool(&*tool, input, cancellation, &self.frontend).await
            });
            match session_id {
                Some(id) => crate::mcp::with_session_id(id, dispatch).await,
                None => dispatch.await,
            }
        };
        if let Some(advisory) = self.schema_advisory(name, input, &schema, loaded, !output.is_error)
        {
            output.append_notice(&advisory);
        }
        output
    }

    /// The advisory to append to this call's result when the arguments and the tool's advertised
    /// schema disagree, or `None`.
    ///
    /// Deferred tools are dispatchable whether or not the model loaded them (see
    /// [`Self::resolve_and_execute_tool`]), and a model that never loaded one has only ever seen
    /// the truncated one-line summary from `[Tool discovery]`. That is how a `send_file` call
    /// landed every image as a download attachment for want of an `as_photo` flag nothing had
    /// mentioned: the call succeeded, so there was no error to read, and the wrong default was
    /// invisible.
    ///
    /// Emitted at most once per tool per process. The result stays in the conversation, so
    /// repeating it buys nothing and costs context on every subsequent call.
    ///
    /// `ran` gates only the *bookkeeping*: a call that never executed (an interactive permission
    /// denial, say) still gets the advisory, but must not spend the tool's one slot, or the retry
    /// after the user grants permission would be the silent call this exists to prevent.
    fn schema_advisory(
        &self,
        name: &str,
        input: &serde_json::Value,
        schema: &serde_json::Value,
        loaded: &[String],
        ran: bool,
    ) -> Option<String> {
        let blind =
            self.tool_registry.is_deferred(name) && !loaded.iter().any(|entry| entry == name);
        let advisory = crate::tools::schema_disagreement(name, input, schema, blind)?;
        if !ran {
            return Some(advisory);
        }
        let first_time = self
            .schema_advisories_sent
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(name.to_string());
        first_time.then_some(advisory)
    }

    /// Ask the user to approve one call in `ask` mode. `None` means run it; `Some` is the result to
    /// return instead.
    ///
    /// Split out of the dispatch path so approval can be settled *before* a `background` call
    /// detaches. Left inline, the prompt would surface minutes later with nothing on screen to
    /// explain what it belonged to.
    async fn request_approval(
        &self,
        name: &str,
        input: &serde_json::Value,
        schema: &serde_json::Value,
        cancellation: &CancellationToken,
    ) -> Option<crate::tools::ToolOutput> {
        let primary_param = crate::render::resolve_primary_param(name, input, Some(schema));
        let outcome = self
            .frontend
            .request_permission(PermissionRequest {
                tool_name: name.to_string(),
                primary_param,
                cancellation: cancellation.clone(),
            })
            .await;
        match outcome {
            PermissionOutcome::Allow => None,
            PermissionOutcome::Deny => Some(crate::tools::ToolOutput::text(
                "User denied tool execution.".to_string(),
                true,
            )),
            PermissionOutcome::Cancelled => Some(crate::tools::ToolOutput::text(
                "Approval request was cancelled.".to_string(),
                true,
            )),
        }
    }

    /// Detach one call: record it, spawn it, and hand the model a task id instead of a result.
    ///
    /// The spawned work gets a **fresh** cancellation token rather than the turn's. Sharing the
    /// turn's would kill the task the instant the turn ended, which is the whole thing this exists
    /// to avoid; the cost is that Ctrl+C no longer reaches it, which is why `task_cancel` and the
    /// second-press escalation exist.
    async fn start_background_call(
        &self,
        tool: &Arc<dyn crate::tools::Tool>,
        tool_call_id: &str,
        name: &str,
        input: &serde_json::Value,
        session_id: Option<Uuid>,
    ) -> crate::tools::ToolOutput {
        let Some(session_id) = session_id else {
            return crate::tools::ToolOutput::text(
                "Error: background calls need a session to report back into. Run this one \
                 normally, without `background`."
                    .to_string(),
                true,
            );
        };
        if self.background_max_tasks == 0 {
            return crate::tools::ToolOutput::text(
                "Error: background calls are disabled on this installation. Run this one normally, \
                 without `background`."
                    .to_string(),
                true,
            );
        }
        let schema = tool.definition().parameters;
        let label = crate::render::resolve_primary_param(name, input, Some(&schema))
            .unwrap_or_else(|| name.to_string());
        let task = crate::background::BackgroundTask {
            id: Uuid::new_v4().to_string(),
            session_id,
            tool_name: name.to_string(),
            label,
            status: crate::background::TaskStatus::Running,
            outcome: None,
            scratchpad_name: None,
            started_at: chrono::Utc::now(),
            finished_at: None,
            delivered_at: None,
        };
        // Claim a slot before anything else. Atomic against the sibling calls in this same
        // assistant message, which `execute_tool_calls` dispatches concurrently: a
        // count-then-register would let four calls all read "zero running" and every one of
        // them start.
        let cancellation = CancellationToken::new();
        if !self
            .background_tasks
            .try_reserve(
                task.id.clone(),
                session_id,
                cancellation.clone(),
                self.background_max_tasks,
            )
            .await
        {
            return crate::tools::ToolOutput::text(
                format!(
                    "Error: {} background tasks are already running, which is the limit. Wait for \
                     one to report, cancel one with `task_cancel`, or run this call without \
                     `background`.",
                    self.background_max_tasks
                ),
                true,
            );
        }

        // Recorded before the spawn, so a process that dies in between leaves a `running` row the
        // sweep can retire rather than work nobody knows happened.
        if let Err(error) = self.session_manager.start_background_task(&task).await {
            // Hand the slot back, or a failed start would shrink the ceiling for the session's
            // lifetime.
            self.background_tasks.forget(&task.id).await;
            return crate::tools::ToolOutput::text(
                format!("Error: could not record the background task: {}", error),
                true,
            );
        }

        let join = tokio::spawn({
            let tool = Arc::clone(tool);
            let input = input.clone();
            let frontend = Arc::clone(&self.frontend);
            let session_manager = self.session_manager.clone();
            let tasks = self.background_tasks.clone();
            let cancellation = cancellation.clone();
            let tool_call_id = tool_call_id.to_string();
            let task_id = task.id.clone();
            let tool_name = task.tool_name.clone();
            async move {
                let run = crate::tools::with_tool_call_id(tool_call_id, async move {
                    Self::run_tool(&*tool, &input, cancellation.clone(), &frontend).await
                });
                // A panic must not escape this task. Nothing awaits its `JoinHandle` outside
                // `--oneshot`, so an unwind here would skip both the outcome write and the slot
                // release: the agent would wait forever on a report that is never coming, and the
                // ceiling would be permanently one lower. Turning it into a `failed` outcome is
                // what the rest of the machinery already knows how to deliver.
                use futures::FutureExt;
                let output = match std::panic::AssertUnwindSafe(crate::mcp::with_session_id(
                    session_id, run,
                ))
                .catch_unwind()
                .await
                {
                    Ok(output) => output,
                    Err(_) => {
                        tracing::error!("background task {} panicked", task_id);
                        crate::tools::ToolOutput::text(
                            "The tool panicked while running in the background.".to_string(),
                            true,
                        )
                    }
                };

                let text = crate::provider::ContentBlock::tool_result_text_content(&output.content);
                let (inline, spilled) = crate::background::split_outcome(&text);
                let mut scratchpad_name = None;
                if let Some(full) = spilled {
                    let name = crate::background::spill_entry_name(&task_id, &tool_name);
                    match session_manager
                        .save_tool_output(session_id, &name, &full)
                        .await
                    {
                        Ok(()) => scratchpad_name = Some(name),
                        // Not fatal: the head still reaches the model, and losing the tail is far
                        // better than losing the whole report.
                        Err(error) => tracing::warn!(
                            "background task {}: could not spill output to the scratchpad: {}",
                            task_id,
                            error
                        ),
                    }
                }

                let status = if output.is_error {
                    crate::background::TaskStatus::Failed
                } else {
                    crate::background::TaskStatus::Completed
                };
                if let Err(error) = session_manager
                    .finish_background_task(&task_id, status, Some(inline), scratchpad_name)
                    .await
                {
                    tracing::warn!(
                        "background task {} finished but could not be recorded: {}",
                        task_id,
                        error
                    );
                }
                tasks.forget(&task_id).await;
            }
        });
        self.background_tasks.attach(&task.id, join).await;

        crate::tools::ToolOutput::text(
            format!(
                "Started in the background as task {} ({}). It is still running; its result will \
                 be delivered to you when it finishes. Do not wait for it here. Use `task_list` to \
                 check on it and `task_cancel` with \"{}\" to stop it.",
                task.short_id(),
                task.label,
                task.short_id(),
            ),
            false,
        )
    }

    /// Invoke a tool, scoping the per-session frontend into a task-local so MCP-originated
    /// callbacks fired during the call (`notifications/progress`, `elicitation/create`) reach
    /// the calling session's UI rather than the process default. Built-in tools ignore the
    /// task-local (they never produce MCP callbacks), so the wrap is cheap on those paths.
    async fn run_tool(
        tool: &dyn crate::tools::Tool,
        input: &serde_json::Value,
        cancellation: CancellationToken,
        frontend: &Arc<dyn Frontend>,
    ) -> crate::tools::ToolOutput {
        let input = input.clone();
        let frontend = Arc::clone(frontend);
        crate::mcp::with_session_frontend(frontend, async move {
            match tool.execute(input, cancellation).await {
                Ok(output) => output,
                Err(MekaError::Interrupted) => {
                    crate::tools::ToolOutput::text("Tool execution interrupted.".to_string(), true)
                }
                Err(error) => {
                    crate::tools::ToolOutput::text(format!("Tool error: {}", error), true)
                }
            }
        })
        .await
    }

    pub async fn compact_session(
        &self,
        session_id: &mut Option<Uuid>,
        messages: &mut Conversation,
    ) -> Result<()> {
        let Some(sid) = *session_id else {
            return Err(MekaError::Config(
                "no active session to compact".to_string(),
            ));
        };

        if messages.is_empty() {
            return Err(MekaError::Config("no messages to compact".to_string()));
        }

        let system_prompt = "You are a conversation summarizer. Produce a structured summary \
             that will replace the conversation. Write in second person \
             (\"You were working on...\").\n\n\
             Cover these sections (skip any that don't apply):\n\n\
             1. **Primary task**: What the user asked for and the overall goal.\n\
             2. **Current state**: What has been completed, what is in progress, what remains.\n\
             3. **Key files**: Files read, created, or modified (list paths).\n\
             4. **Key decisions**: Important choices made and their rationale.\n\
             5. **Errors and fixes**: Problems encountered and how they were resolved.\n\
             6. **User preferences and constraints**: Feedback or corrections about how to \
             work. Preserve any security-relevant instructions verbatim (sensitive files or \
             data to avoid, operations that must not be performed, secret-handling rules) so \
             they keep applying after compaction.\n\
             7. **All user requests**: Every distinct request the user made, in order, so none \
             of their intent is lost.\n\
             8. **Next step**: The immediate next action. If a task was mid-flight, quote the \
             user's most recent request verbatim so the work does not drift.";

        // Split into a head to summarize and a recent tail to keep verbatim. The tail is the
        // largest recent suffix that fits a token budget (~10% of the window, capped), snapped back
        // to a clean user boundary so tool_use/tool_result pairs are never orphaned.
        let keep_budget = (self.options.context_window / 10).clamp(4_000, 16_000);
        let (to_summarize, to_keep) = compute_compaction_split(messages.as_slice(), keep_budget);

        // Clone and preprocess messages for the summarizer: strip images and truncate large text
        // blocks to avoid overwhelming the summary call.
        let mut compact_messages = to_summarize;
        for message in &mut compact_messages {
            strip_images_and_truncate(&mut message.content);
        }

        // Append a user message so the conversation ends with a user turn.
        compact_messages.push(Message::user(
            "Summarize this conversation into a concise context message.",
        ));

        self.provider.set_thinking_override(Some(false));
        let compact_result = self
            .provider
            .complete(system_prompt, &compact_messages, &[])
            .await;
        self.provider.set_thinking_override(None);
        let (summary_message, _stop_reason, _usage, notices) = compact_result?;
        // Surface any provider notices from the summary call (e.g. image redaction on a very large
        // compaction window). Rare in practice; emitting before we mutate the conversation keeps
        // the user-facing order stable.
        for notice in notices {
            self.frontend.emit(FrontendEvent::Notice(notice)).await;
        }

        let summary_text = summary_message.text_content();
        if summary_text.is_empty() {
            return Err(MekaError::Provider(
                "LLM returned an empty summary".to_string(),
            ));
        }

        // Build post-compact context: environment, todos, scratchpad inventory.
        let post_context = self.build_post_compact_context(sid).await;

        let mut context_message =
            format!("[Conversation summary from session compaction]\n\n{summary_text}");
        if !post_context.is_empty() {
            context_message.push_str(&format!("\n\n[Post-compaction context]\n\n{post_context}"));
        }
        // Behavioural directive (always last, most salient): pick the work back up rather than
        // narrate the summary. Without it, the turn after an auto-compaction tends to open with
        // "Based on the summary, I'll continue..." preambles that waste output and add nothing.
        context_message.push_str(
            "\n\n[Continue the work directly from the summary above. Do not acknowledge or recap \
             this summary; resume as if the conversation had not been interrupted.]",
        );

        // Snapshot the deferred-tool active set BEFORE compaction so the `CompactBoundary` event
        // carries it forward; otherwise tools the model loaded pre-compaction would silently drop
        // out of the active set on the next turn.
        let loaded_tools_snapshot = crate::tools::extract_loaded_tool_names(messages.as_slice());

        let summary_user_message = Message::user(&context_message);
        messages.replace_for_compaction(
            summary_user_message,
            to_keep.clone(),
            loaded_tools_snapshot,
        );

        // Persist the new compaction-boundary event and the re-appended tail. Pre-compaction rows
        // stay in the DB unchanged; the event log on disk grows append-only.
        let boundary_event = messages
            .events()
            .iter()
            .rev()
            .find(|e| matches!(e, crate::conversation::Event::CompactBoundary { .. }))
            .cloned()
            .ok_or_else(|| {
                MekaError::Internal(
                    "compact boundary missing after replace_for_compaction".to_string(),
                )
            })?;
        self.session_manager
            .save_event(sid, &boundary_event)
            .await?;
        for message in &to_keep {
            self.session_manager
                .save_event(sid, &crate::conversation::Event::Append(message.clone()))
                .await?;
        }

        // Pre-boundary events are now fully superseded and already persisted; drop them so the
        // in-memory log doesn't grow unbounded across repeated compactions.
        messages.prune_compacted_events();

        // Compaction rewrites the conversation, so a length recorded against the old one no longer
        // identifies any particular message. Cleared here rather than at the call sites so
        // `/compact` and both auto-compact paths are covered by construction.
        self.last_accepted_len
            .store(LAST_ACCEPTED_UNKNOWN, std::sync::atomic::Ordering::Relaxed);

        // The model's view of which files it has read is reset by the summary; drop the
        // read-tracker so `edit_file` re-reads rather than trusting a pre-compaction read (also
        // bounds its growth).
        self.tool_registry.clear_read_tracker().await;

        // Same reasoning for the tool/skill/MCP picture: the turns that carried it are now behind
        // the boundary and may have been summarized away, so forget what the model was told and let
        // the next turn re-state it in full. Compaction re-caches the conversation anyway, so the
        // extra tokens cost nothing that wasn't already spent.
        *self.last_rendered_world.write().await = None;

        // Seed the live context gauge with an estimate of the compacted working set so `/status`
        // (and the prompt indicator) immediately reflect the smaller size; the next real turn
        // overwrites it with the exact provider-reported total.
        self.last_context_tokens.store(
            crate::tokens::estimate_messages(messages.as_slice()),
            std::sync::atomic::Ordering::Relaxed,
        );

        Ok(())
    }

    async fn build_post_compact_context(&self, session_id: Uuid) -> String {
        let permission = self.shared_permission.get();
        let todos = self.todo_list.read().await.clone();
        let entries = self
            .session_manager
            .list_tool_outputs(session_id)
            .await
            .unwrap_or_default();
        context::build_post_compact_context(
            permission,
            &todos,
            &entries,
            &cwd_snapshot(&self.cwd),
            &roots_snapshot(&self.roots),
        )
    }
}

/// Whether a world-state render made when the conversation held `rendered_at` messages is still
/// inside the window [`truncate_messages_for_context`] will send.
///
/// The render lives in exactly one user message, at index `rendered_at`. The window keeps the last
/// `context_messages` entries, so that message survives while `current_len - rendered_at` stays
/// within the limit. Once it falls out, the model can no longer see the tool catalogue, the skill
/// list, or any MCP server's instructions, and the picture has to be restated in full.
///
/// Deliberately one turn conservative (`<` rather than `<=`), for two reasons: `current_len` is
/// read before this turn's own message is appended, and `truncate_messages_for_context` walks
/// backward from the cut to land on a user-message boundary, which can only keep *more*. Restating
/// a turn early costs tokens once per window; restating a turn late means a request with no
/// catalogue in it at all.
fn world_state_still_visible(
    rendered_at: usize,
    current_len: usize,
    context_messages: Option<usize>,
) -> bool {
    context_messages.is_none_or(|limit| current_len.saturating_sub(rendered_at) < limit)
}

fn truncate_messages_for_context(
    messages: &[Message],
    context_messages: Option<usize>,
) -> Vec<Message> {
    let Some(limit) = context_messages else {
        return messages.to_vec();
    };

    if messages.len() <= limit {
        return messages.to_vec();
    }

    let mut start_index = messages.len().saturating_sub(limit);

    // Walk backward to find a safe cut point: a user message that is NOT a tool_results message.
    // This avoids splitting assistant(ToolUse) → user(ToolResult) chains and ensures the first
    // message has role User (required by Claude API).
    loop {
        if start_index == 0 {
            break;
        }

        let message = &messages[start_index];
        if message.role == Role::User && !has_tool_results(&message.content) {
            break;
        }

        start_index -= 1;
    }

    messages[start_index..].to_vec()
}

fn has_tool_results(content: &[ContentBlock]) -> bool {
    content
        .iter()
        .any(|block| matches!(block, ContentBlock::ToolResult { .. }))
}

/// How much of the provider's rejection text is carried into the conversation. Long enough to keep
/// the specific complaint (Anthropic's runs to about 150 characters), short enough that a provider
/// echoing the request body back can't flood the window.
const REJECTION_REASON_LIMIT: usize = 600;

/// Rewrite `messages` so nothing the provider can refuse on content grounds survives, replacing
/// every non-text block with a note carrying `reason`. Returns `None` when there was nothing to
/// rewrite, which the caller treats as "this rejection isn't about content, don't retry".
///
/// Structure is preserved rather than pruned. A `tool_use` whose result is dropped would be an
/// orphan the provider rejects in a *new* way, and dropping the `tool_use` itself is worse still:
/// the tool has already run, side effects and all, so erasing the record invites the model to run
/// it again. Instead the `tool_result` keeps its `tool_use_id` and is marked `is_error`, which is
/// exactly the shape meka already uses for a tool that failed outright, so the model needs no new
/// concept to understand it and no frontend needs new rendering.
fn degrade_rejected_content(messages: &[Message], reason: &str) -> Option<Vec<Message>> {
    let reason = elide_reason(reason);
    let mut changed = false;
    let degraded: Vec<Message> = messages
        .iter()
        .map(|message| {
            let content = message
                .content
                .iter()
                .map(|block| match block {
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } if content
                        .iter()
                        .any(|item| !matches!(item, ToolResultContent::Text { .. })) =>
                    {
                        changed = true;
                        let mut kept: Vec<ToolResultContent> = content
                            .iter()
                            .filter(|item| matches!(item, ToolResultContent::Text { .. }))
                            .cloned()
                            .collect();
                        kept.push(ToolResultContent::Text {
                            text: format!(
                                "[meka] The provider refused this tool result, so its non-text \
                                 content was removed to keep the conversation usable: {}. Do not \
                                 repeat this call unchanged.",
                                reason
                            ),
                        });
                        ContentBlock::ToolResult {
                            tool_use_id: tool_use_id.clone(),
                            content: kept,
                            is_error: true,
                        }
                    }
                    ContentBlock::Image { .. } => {
                        changed = true;
                        ContentBlock::Text {
                            text: format!(
                                "[meka] An image attached to this message was removed because the \
                                 provider refused it: {}.",
                                reason
                            ),
                        }
                    }
                    other => other.clone(),
                })
                .collect();
            Message {
                role: message.role.clone(),
                content,
            }
        })
        .collect();

    changed.then_some(degraded)
}

fn elide_reason(reason: &str) -> String {
    if reason.chars().count() <= REJECTION_REASON_LIMIT {
        return reason.to_string();
    }
    let kept: String = reason.chars().take(REJECTION_REASON_LIMIT).collect();
    format!("{}…", kept)
}

/// Whether a failed provider call should be retried, and if so, after how long. Pure and
/// sleep-free so it's unit-testable in isolation from the async retry loops in `run_streaming` and
/// `run_turn`'s non-streaming branch, which both call this with their current `retries` count
/// (0-indexed, incremented by the caller only when this returns `Some`). `content_started` must
/// always be `false` for the non-streaming path — nothing is ever partially visible there, so every
/// retryable failure is retryable regardless of prior attempts within the same call.
fn should_retry_provider_error(
    error: &MekaError,
    content_started: bool,
    retries: u32,
) -> Option<std::time::Duration> {
    match error {
        MekaError::RetryableProvider { retry_after, .. }
            if !content_started && retries < crate::provider::retry::MAX_PROVIDER_RETRIES =>
        {
            Some(crate::provider::retry::backoff_delay(
                retries + 1,
                *retry_after,
            ))
        }
        // A mid-stream transport failure (SSE decode error, dropped connection, idle timeout) is
        // transient: retry it with backoff, but only before any output reached the frontend, so a
        // retry can't double-emit. Mirrors codex's `retry_transport` behaviour.
        MekaError::StreamError(_)
            if !content_started && retries < crate::provider::retry::MAX_PROVIDER_RETRIES =>
        {
            Some(crate::provider::retry::backoff_delay(retries + 1, None))
        }
        _ => None,
    }
}

/// Split a conversation for compaction into `(to_summarize, to_keep)`. The kept tail is the largest
/// recent suffix whose estimated tokens stay within `keep_budget`, then snapped backward to a clean
/// `User`-without-`tool_results` boundary so a tool_use/tool_result pair is never orphaned and the
/// kept window starts on a valid user turn. If that leaves fewer than `MIN_SUMMARIZE` messages to
/// summarize, the whole conversation is summarized and no tail is kept (a smaller head saves too
/// little to be worth a boundary).
fn compute_compaction_split(view: &[Message], keep_budget: u64) -> (Vec<Message>, Vec<Message>) {
    const MIN_SUMMARIZE: usize = 4;
    if view.len() <= MIN_SUMMARIZE {
        return (view.to_vec(), Vec::new());
    }

    // Grow the tail from the end while it fits the budget; always keep at least the last message.
    let mut split = view.len();
    let mut tail_tokens = 0u64;
    while split > 0 {
        let candidate =
            tail_tokens.saturating_add(crate::tokens::estimate_message(&view[split - 1]));
        if candidate > keep_budget && split < view.len() {
            break;
        }
        tail_tokens = candidate;
        split -= 1;
    }

    // Snap back to a clean user boundary. This only grows the tail, so it never orphans a
    // tool_result and guarantees the kept window starts on a User turn.
    while split > 0 {
        let message = &view[split];
        if message.role == Role::User && !has_tool_results(&message.content) {
            break;
        }
        split -= 1;
    }

    if split >= MIN_SUMMARIZE {
        (view[..split].to_vec(), view[split..].to_vec())
    } else {
        (view.to_vec(), Vec::new())
    }
}

/// Split the unavailable MCP servers into the ones that stop the turn and the ones that don't.
///
/// Only `required` servers gate. Whether a missing server should halt work is a property of that
/// server, not of the installation - the same config runs on a workstation that has the binary and
/// in a container that doesn't - so a single installation-wide switch could only ever be right for
/// one of them. `[mcp].strict` survives as the default each server inherits.
///
/// A free function rather than a method because it reads nothing from the agent, which also makes
/// the gating decision directly testable.
fn gate_on_required_servers(not_ready: Vec<crate::mcp::NotConnected>) -> Result<()> {
    let (required, optional): (Vec<_>, Vec<_>) =
        not_ready.into_iter().partition(|server| server.required);

    if !optional.is_empty() {
        let names: Vec<&str> = optional.iter().map(|s| s.name.as_str()).collect();
        // `debug!`, not `warn!`: this runs on *every* turn, and a server that is down stays down,
        // so at warn level a single unreachable server would print a line before every reply for
        // the life of the session. The connector already reports the failure once, and `/mcp list`
        // in the REPL shows live state on demand; repeating it per turn is noise.
        tracing::debug!(
            "mcp: proceeding without {} optional server(s): {:?}",
            names.len(),
            names
        );
    }

    if required.is_empty() {
        return Ok(());
    }
    Err(MekaError::McpTurnGated {
        servers: required
            .iter()
            .map(|server| {
                // Carry the cause, not just the label. This message is the only thing the user gets
                // when a required server blocks every turn, and the connector's warn fires once at
                // startup and then stays quiet, so "ida (failed)" on its own leaves them with
                // nothing to act on. The cause replaces the label rather than joining it: every
                // one of them already describes a failure ("failed to spawn process: …"), so
                // prefixing would render as "failed: failed to …".
                let detail = match &server.state {
                    crate::mcp::ServerState::Failed { error, .. } => error.clone(),
                    other => other.label().to_string(),
                };
                (server.name.clone(), detail)
            })
            .collect(),
    })
}

/// Whether an assistant turn produced any user-visible text: a `Text` block with non-whitespace
/// content. `Thinking`, `ToolUse`, `ToolResult`, and `Image` blocks are not user-visible prose, so
/// a thinking-only turn returns `false` here.
fn has_visible_text(content: &[ContentBlock]) -> bool {
    content
        .iter()
        .any(|block| matches!(block, ContentBlock::Text { text } if !text.trim().is_empty()))
}

/// Human-readable stand-in for a terminal turn that produced no content (e.g. a hard refusal, or an
/// empty `max_tokens` / `end_turn`). Used both as the persisted assistant text (so the message
/// isn't empty) and as the line surfaced to the user.
fn empty_turn_notice(stop_reason: &StopReason) -> String {
    match stop_reason {
        StopReason::Refusal(text) if !text.is_empty() => text.clone(),
        StopReason::Refusal(_) => "[The model declined to respond to this request.]".to_string(),
        StopReason::MaxTokens => {
            "[The model reached its output limit before producing a response.]".to_string()
        }
        // Surface the raw reason so an unrecognised stop reason is visible instead of being
        // swallowed as a blank turn.
        StopReason::Unknown(reason) => {
            format!("[The model returned an empty response (stop reason: {reason}).]")
        }
        _ => "[The model returned an empty response.]".to_string(),
    }
}

/// Meta message injected to coax a user-visible response out of a turn that produced only thinking
/// (or nothing). Mirrors Claude Code's `query_thinking_only_response` nudge.
const THINKING_ONLY_NUDGE: &str = "[Your previous response contained no visible output. Please \
                                   continue and produce a user-visible response.]";

/// Whether to nudge the model for a user-visible response after a turn that made no tool call and
/// produced no visible text (e.g. a thinking-only turn). Mirrors Claude Code's
/// `query_thinking_only_response`: fire at most once per turn, and only for a terminal stop reason
/// without its own handling - `MaxTokens` and `Refusal` carry their own outcomes, so a no-text turn
/// under those reasons falls through to [`empty_turn_notice`] instead of being retried.
fn should_nudge_thinking_only(
    has_tool_calls: bool,
    has_visible_text: bool,
    stop_reason: &StopReason,
    already_nudged: bool,
) -> bool {
    !has_tool_calls
        && !has_visible_text
        && !already_nudged
        && matches!(stop_reason, StopReason::EndTurn | StopReason::Unknown(_))
}

/// Preprocess message content blocks for the compaction summarizer:
/// replace images with "[image]" markers and truncate large text blocks.
fn strip_images_and_truncate(content: &mut [ContentBlock]) {
    use crate::provider::ToolResultContent;

    const MAX_TEXT_CHARS: usize = 2000;
    const HEAD_CHARS: usize = 1000;
    const TAIL_CHARS: usize = 500;

    for block in content.iter_mut() {
        match block {
            ContentBlock::ToolResult {
                content: tool_content,
                ..
            } => {
                for item in tool_content.iter_mut() {
                    match item {
                        ToolResultContent::Image { .. } => {
                            *item = ToolResultContent::Text {
                                text: "[image]".to_string(),
                            };
                        }
                        ToolResultContent::Text { text } => {
                            if text.len() > MAX_TEXT_CHARS {
                                let head_end = text.floor_char_boundary(HEAD_CHARS);
                                let tail_start =
                                    text.floor_char_boundary(text.len().saturating_sub(TAIL_CHARS));
                                *text = format!(
                                    "{}\n... (truncated for compaction) ...\n{}",
                                    &text[..head_end],
                                    &text[tail_start..],
                                );
                            }
                        }
                    }
                }
            }
            ContentBlock::Text { text } if text.len() > MAX_TEXT_CHARS => {
                let head_end = text.floor_char_boundary(HEAD_CHARS);
                let tail_start = text.floor_char_boundary(text.len().saturating_sub(TAIL_CHARS));
                *text = format!(
                    "{}\n... (truncated for compaction) ...\n{}",
                    &text[..head_end],
                    &text[tail_start..],
                );
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ToolResultContent;

    /// Minimal in-memory agent driving `provider`: no tools, no skills, no memories, silent
    /// frontend. Enough to exercise `run_turn`'s recovery arms, which touch none of that.
    async fn test_agent(provider: Arc<dyn Provider>) -> (Agent, SessionManager) {
        test_agent_with_registry(provider, crate::tools::ToolRegistry::new()).await
    }

    async fn test_agent_with_registry(
        provider: Arc<dyn Provider>,
        registry: crate::tools::ToolRegistry,
    ) -> (Agent, SessionManager) {
        let session_manager = SessionManager::open(Some(std::path::Path::new(":memory:")))
            .await
            .expect("in-memory db");
        let options = AgentOptions {
            streaming: true,
            sandboxed_shell: false,
            context_messages: None,
            auto_compact: false,
            context_window: 0,
            user_instructions: None,
            mcp_grace: std::time::Duration::from_secs(0),
            system_prompt_override: Some("test".to_string()),
        };
        let agent = Agent::new(
            provider,
            registry,
            session_manager.clone(),
            SharedPermission::new(
                crate::permission::Permission::Read,
                crate::permission::EnabledPermissions::ALL,
            ),
            options,
            crate::tools::todo::SharedTodoList::default(),
            Arc::new(tokio::sync::RwLock::new(None)),
            crate::skills::SkillCache::disabled(),
            crate::memory::MemoryCache::disabled(),
            Arc::new(crate::frontend::SilentFrontend),
            Arc::new(RwLock::new(std::env::temp_dir())),
            Arc::new(RwLock::new(Vec::new())),
            Arc::new(crate::stats::SessionStats::default()),
        );
        (agent, session_manager)
    }

    fn image_source() -> ImageSource {
        ImageSource {
            source_type: "base64".to_string(),
            media_type: "image/png".to_string(),
            data: "QUJD".to_string(),
        }
    }

    const REJECTION: &str = "API returned status 400 Bad Request: the image was specified using \
                             the image/png media type, but the image appears to be a image/jpeg \
                             image";

    /// The whole point of the feature: a rejection of content meka just appended must not end the
    /// turn, and the repair must be persisted so a resume doesn't walk back into it.
    #[tokio::test]
    async fn test_run_turn_degrades_rejected_content_and_continues() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![MockEvent::FailInvalidRequest {
                message: REJECTION.to_string(),
            }],
            vec![
                MockEvent::Text {
                    text: "I could not see that image.".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
        ]));
        let (agent, session_manager) = test_agent(provider).await;

        let mut session_id = None;
        let mut messages = Conversation::new();
        let outcome = agent
            .run_turn(
                &mut session_id,
                &mut messages,
                "look at this".to_string(),
                vec![image_source()],
                CancellationToken::new(),
            )
            .await
            .expect("the turn recovers instead of dying");
        assert_eq!(outcome, TurnOutcome::EndTurn);

        // The image is gone from the live conversation, replaced by an explanation.
        let user = &messages.as_slice()[0];
        assert!(
            user.content
                .iter()
                .all(|block| !matches!(block, ContentBlock::Image { .. })),
            "the refused image must not survive in the conversation"
        );
        assert!(
            user.text_content().contains("image/jpeg"),
            "carries the reason"
        );

        // And it is gone on disk too, or the next resume would re-poison the session.
        let sid = session_id.expect("session created");
        let reloaded =
            Conversation::from_events(session_manager.load_events(sid).await.expect("load events"));
        assert!(
            reloaded
                .iter()
                .flat_map(|message| message.content.iter())
                .all(|block| !matches!(block, ContentBlock::Image { .. })),
            "the repair must be persisted, not just applied in memory"
        );
    }

    /// A rejection that degrading doesn't fix must cost one round trip and nothing else.
    #[tokio::test]
    async fn test_run_turn_restores_content_when_the_repair_does_not_help() {
        use crate::provider::mock::{MockEvent, MockProvider};

        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![MockEvent::FailInvalidRequest {
                message: REJECTION.to_string(),
            }],
            vec![MockEvent::FailInvalidRequest {
                message: REJECTION.to_string(),
            }],
        ]));
        let (agent, _session_manager) = test_agent(provider).await;

        let mut session_id = None;
        let mut messages = Conversation::new();
        let error = agent
            .run_turn(
                &mut session_id,
                &mut messages,
                "look at this".to_string(),
                vec![image_source()],
                CancellationToken::new(),
            )
            .await
            .expect_err("both attempts were refused");
        assert!(
            matches!(&error, MekaError::InvalidRequest(message) if message.contains("image/jpeg")),
            "the provider's own error surfaces, not one about the repair: {error}"
        );

        assert!(
            messages.as_slice()[0]
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::Image { .. })),
            "a repair that didn't help must leave the conversation untouched"
        );
        assert!(
            !messages
                .events()
                .iter()
                .any(|event| matches!(event, crate::conversation::Event::Repair { .. })),
            "and must leave no repair behind in the log"
        );
    }

    /// `/rewind` shortens the conversation behind a live agent, which invalidates the length the
    /// recovery measures its suspect window back from. Left stale, that length lands at or past the
    /// end of the shortened conversation, the window comes out empty, and the recovery silently
    /// never fires again for the rest of the session.
    #[tokio::test]
    async fn test_recovery_still_fires_after_the_conversation_is_rewound() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let text_round = |text: &str| {
            vec![
                MockEvent::Text {
                    text: text.to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ]
        };
        let provider = Arc::new(MockProvider::from_rounds(vec![
            text_round("first answer"),
            vec![MockEvent::FailInvalidRequest {
                message: REJECTION.to_string(),
            }],
            text_round("second answer, without the image"),
        ]));
        let (agent, _session_manager) = test_agent(provider).await;

        let mut session_id = None;
        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut session_id,
                &mut messages,
                "first".to_string(),
                Vec::new(),
                CancellationToken::new(),
            )
            .await
            .expect("first turn succeeds");

        assert!(messages.rewind(1).is_some(), "the turn is rewound away");
        agent.reset_conversation_markers().await;

        agent
            .run_turn(
                &mut session_id,
                &mut messages,
                "look at this".to_string(),
                vec![image_source()],
                CancellationToken::new(),
            )
            .await
            .expect("the recovery must still fire on a rewound conversation");
        assert!(
            messages
                .iter()
                .flat_map(|message| message.content.iter())
                .all(|block| !matches!(block, ContentBlock::Image { .. })),
            "the refused image should have been degraded away"
        );
    }

    /// A deferred tool with a documented optional parameter, standing in for mekabridge's
    /// `send_file`.
    struct SendFileFixture;

    #[async_trait::async_trait]
    impl crate::tools::Tool for SendFileFixture {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "mcp__bridge__send_file".to_string(),
                description: "Send a file to a conversation.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file"},
                        "as_photo": {
                            "type": "boolean",
                            "default": false,
                            "description": "Send as a viewable photo rather than a document.",
                        },
                    },
                    "required": ["path"]
                }),
                ..Default::default()
            }
        }

        fn required_permission(&self) -> crate::permission::Permission {
            crate::permission::Permission::Read
        }

        async fn execute(
            &self,
            _input: serde_json::Value,
            _cancellation: CancellationToken,
        ) -> Result<crate::tools::ToolOutput> {
            Ok(crate::tools::ToolOutput::text(
                "Sent (message id 1)".to_string(),
                false,
            ))
        }
    }

    fn send_file_registry() -> crate::tools::ToolRegistry {
        let registry = crate::tools::ToolRegistry::new();
        registry.register_load_tool_for_test();
        registry
            .register(Arc::new(SendFileFixture))
            .expect("register fixture");
        registry.mark_deferred("mcp__bridge__send_file");
        registry
    }

    fn send_file_round(input: serde_json::Value) -> Vec<crate::provider::mock::MockEvent> {
        use crate::provider::mock::{MockEvent, MockStopReason};
        vec![
            MockEvent::ToolUseStart {
                id: "call-1".to_string(),
                name: "mcp__bridge__send_file".to_string(),
            },
            MockEvent::ToolUseEnd { input },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::ToolUse,
            },
        ]
    }

    /// The whole incident, end to end: a deferred tool is callable without `load_tool`, so a model
    /// working from the truncated `[Tool discovery]` summary takes a silently wrong default. The
    /// result has to say so, because the call itself succeeds and there is no error to read.
    #[tokio::test]
    async fn test_blind_deferred_call_is_told_what_it_omitted() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let provider = Arc::new(MockProvider::from_rounds(vec![
            send_file_round(serde_json::json!({"path": "/tmp/a.png"})),
            vec![
                MockEvent::Text {
                    text: "sent".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
        ]));
        let (agent, _session_manager) =
            test_agent_with_registry(provider, send_file_registry()).await;

        let mut session_id = None;
        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut session_id,
                &mut messages,
                "send the picture".to_string(),
                Vec::new(),
                CancellationToken::new(),
            )
            .await
            .expect("turn succeeds");

        let results: String = messages
            .iter()
            .flat_map(|message| message.content.iter())
            .filter_map(|block| match block {
                ContentBlock::ToolResult { content, .. } => {
                    Some(ContentBlock::tool_result_text_content(content))
                }
                _ => None,
            })
            .collect();

        assert!(results.contains("Sent (message id 1)"), "{results}");
        assert!(results.contains("as_photo"), "the omitted flag: {results}");
        assert!(results.contains("load_tool"), "{results}");
    }

    /// A tool that blocks until cancelled, standing in for a long build.
    struct SlowFixture;

    #[async_trait::async_trait]
    impl crate::tools::Tool for SlowFixture {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "execute_command".to_string(),
                description: "Run a shell command.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The command"}
                    },
                    "required": ["command"]
                }),
                ..Default::default()
            }
        }

        fn required_permission(&self) -> crate::permission::Permission {
            crate::permission::Permission::Read
        }

        async fn execute(
            &self,
            _input: serde_json::Value,
            cancellation: CancellationToken,
        ) -> Result<crate::tools::ToolOutput> {
            cancellation.cancelled().await;
            Err(MekaError::Interrupted)
        }
    }

    /// A tool that returns at once, for the ordinary completion path.
    struct QuickFixture;

    #[async_trait::async_trait]
    impl crate::tools::Tool for QuickFixture {
        fn definition(&self) -> ToolDefinition {
            SlowFixture.definition()
        }

        fn required_permission(&self) -> crate::permission::Permission {
            crate::permission::Permission::Read
        }

        async fn execute(
            &self,
            _input: serde_json::Value,
            _cancellation: CancellationToken,
        ) -> Result<crate::tools::ToolOutput> {
            Ok(crate::tools::ToolOutput::text(
                "42 passed".to_string(),
                false,
            ))
        }
    }

    async fn background_agent(
        provider: Arc<dyn Provider>,
        tool: Arc<dyn crate::tools::Tool>,
    ) -> (Agent, SessionManager) {
        let registry = crate::tools::ToolRegistry::new();
        registry.enable_background();
        registry.register(tool).expect("register fixture");
        let (mut agent, session_manager) = test_agent_with_registry(provider, registry).await;
        agent.enable_background(crate::background::BackgroundTasks::default(), 2);
        (agent, session_manager)
    }

    fn background_round(command: &str) -> Vec<crate::provider::mock::MockEvent> {
        use crate::provider::mock::{MockEvent, MockStopReason};
        vec![
            MockEvent::ToolUseStart {
                id: "call-1".to_string(),
                name: "execute_command".to_string(),
            },
            MockEvent::ToolUseEnd {
                input: serde_json::json!({"command": command, "background": true}),
            },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::ToolUse,
            },
        ]
    }

    fn text_round(text: &str) -> Vec<crate::provider::mock::MockEvent> {
        use crate::provider::mock::{MockEvent, MockStopReason};
        vec![
            MockEvent::Text {
                text: text.to_string(),
            },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::EndTurn,
            },
        ]
    }

    /// The whole point: the turn ends without waiting, and the model is handed a task id rather
    /// than a result.
    #[tokio::test]
    async fn test_a_background_call_returns_a_handle_and_ends_the_turn() {
        use crate::provider::mock::MockProvider;

        let provider = Arc::new(MockProvider::from_rounds(vec![
            background_round("sleep 600"),
            text_round("started it"),
        ]));
        let (agent, session_manager) = background_agent(provider, Arc::new(SlowFixture)).await;

        let mut session_id = None;
        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut session_id,
                &mut messages,
                "run the suite".to_string(),
                Vec::new(),
                CancellationToken::new(),
            )
            .await
            .expect("the turn must not block on the task");

        let results: String = messages
            .iter()
            .flat_map(|message| message.content.iter())
            .filter_map(|block| match block {
                ContentBlock::ToolResult { content, .. } => {
                    Some(ContentBlock::tool_result_text_content(content))
                }
                _ => None,
            })
            .collect();
        assert!(results.contains("Started in the background"), "{results}");
        assert!(results.contains("task_cancel"), "{results}");

        let running = session_manager
            .list_running_background_tasks(session_id.expect("session"))
            .await
            .expect("list running");
        assert_eq!(running.len(), 1);
        assert_eq!(running[0].label, "sleep 600");

        // Leave nothing behind for the next test's runtime to trip over.
        agent.background_tasks().cancel_all().await;
    }

    /// With `[background] enabled = false` the parameter is never offered, so a model asking for it
    /// is guessing. It still must not be silently ignored: running a twenty-minute command in the
    /// foreground because a flag was dropped is exactly the surprise the whole feature exists to
    /// avoid, and the refusal tells the model to reissue the call plainly.
    #[tokio::test]
    async fn test_background_is_refused_when_the_installation_disabled_it() {
        use crate::provider::mock::MockProvider;

        let registry = crate::tools::ToolRegistry::new();
        registry
            .register(Arc::new(QuickFixture))
            .expect("register fixture");
        // Deliberately no `enable_background`: this is what a default installation looks like.
        let provider = Arc::new(MockProvider::from_rounds(vec![
            background_round("cargo test"),
            text_round("understood"),
        ]));
        let (agent, session_manager) = test_agent_with_registry(provider, registry).await;

        let mut session_id = None;
        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut session_id,
                &mut messages,
                "run the suite".to_string(),
                Vec::new(),
                CancellationToken::new(),
            )
            .await
            .expect("turn succeeds");

        let results: String = messages
            .iter()
            .flat_map(|message| message.content.iter())
            .filter_map(|block| match block {
                ContentBlock::ToolResult { content, .. } => {
                    Some(ContentBlock::tool_result_text_content(content))
                }
                _ => None,
            })
            .collect();
        assert!(
            results.contains("background calls are disabled"),
            "{results}"
        );
        assert!(results.contains("without `background`"), "{results}");
        assert!(
            !results.contains("42 passed"),
            "the call must be refused, not quietly run in the foreground: {results}"
        );
        assert!(
            session_manager
                .list_background_tasks(session_id.expect("session"))
                .await
                .expect("list")
                .is_empty(),
            "a disabled installation must not record tasks"
        );
    }

    /// A finished task's outcome has to reach the conversation, and exactly once.
    #[tokio::test]
    async fn test_a_finished_task_is_recorded_for_delivery_once() {
        use crate::provider::mock::MockProvider;

        let provider = Arc::new(MockProvider::from_rounds(vec![
            background_round("cargo test"),
            text_round("started it"),
        ]));
        let (agent, session_manager) = background_agent(provider, Arc::new(QuickFixture)).await;

        let mut session_id = None;
        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut session_id,
                &mut messages,
                "run the suite".to_string(),
                Vec::new(),
                CancellationToken::new(),
            )
            .await
            .expect("turn succeeds");
        let session_id = session_id.expect("session");
        agent.background_tasks().wait_for_session(session_id).await;

        let ready = session_manager
            .list_undelivered_background_tasks(session_id)
            .await
            .expect("list undelivered");
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].status, crate::background::TaskStatus::Completed);
        assert_eq!(ready[0].outcome.as_deref(), Some("42 passed"));

        let rendered = crate::background::render_outcomes(&ready);
        assert!(rendered.contains("42 passed"), "{rendered}");
        assert!(rendered.contains("cargo test"), "{rendered}");

        let ids: Vec<String> = ready.iter().map(|task| task.id.clone()).collect();
        session_manager
            .mark_background_tasks_delivered(&ids)
            .await
            .expect("stamp");
        assert!(
            session_manager
                .list_undelivered_background_tasks(session_id)
                .await
                .expect("list undelivered")
                .is_empty(),
            "an outcome must reach the conversation once, not on every tick"
        );
    }

    /// Nothing awaits a background task's `JoinHandle` outside `--oneshot`, so an unwind that
    /// escaped would skip both the outcome write and the slot release: a report that never comes,
    /// and a ceiling permanently one lower.
    #[tokio::test]
    async fn test_a_panicking_background_tool_still_reports_and_frees_its_slot() {
        use crate::provider::mock::MockProvider;

        struct PanicFixture;

        #[async_trait::async_trait]
        impl crate::tools::Tool for PanicFixture {
            fn definition(&self) -> ToolDefinition {
                SlowFixture.definition()
            }

            fn required_permission(&self) -> crate::permission::Permission {
                crate::permission::Permission::Read
            }

            async fn execute(
                &self,
                _input: serde_json::Value,
                _cancellation: CancellationToken,
            ) -> Result<crate::tools::ToolOutput> {
                panic!("tool exploded");
            }
        }

        let provider = Arc::new(MockProvider::from_rounds(vec![
            background_round("boom"),
            text_round("started"),
        ]));
        let (agent, session_manager) = background_agent(
            provider,
            Arc::new(PanicFixture) as Arc<dyn crate::tools::Tool>,
        )
        .await;

        let mut session_id = None;
        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut session_id,
                &mut messages,
                "run it".to_string(),
                Vec::new(),
                CancellationToken::new(),
            )
            .await
            .expect("turn succeeds");
        let session_id = session_id.expect("session");
        agent.background_tasks().wait_for_session(session_id).await;

        let ready = session_manager
            .list_undelivered_background_tasks(session_id)
            .await
            .expect("list undelivered");
        assert_eq!(ready.len(), 1, "the panic must still produce a report");
        assert_eq!(ready[0].status, crate::background::TaskStatus::Failed);

        assert_eq!(
            agent.background_tasks().running_count(session_id).await,
            0,
            "the slot must be released, or the ceiling shrinks for the session's lifetime"
        );
    }

    /// The concurrency ceiling refuses rather than silently queueing, so the model can decide
    /// whether to wait or run the call in the foreground.
    #[tokio::test]
    async fn test_the_task_ceiling_refuses_with_something_actionable() {
        use crate::provider::mock::MockProvider;

        let provider = Arc::new(MockProvider::from_rounds(vec![
            background_round("sleep 1"),
            background_round("sleep 2"),
            background_round("sleep 3"),
            text_round("done"),
        ]));
        let (agent, _session_manager) = background_agent(provider, Arc::new(SlowFixture)).await;

        let mut session_id = None;
        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut session_id,
                &mut messages,
                "start three".to_string(),
                Vec::new(),
                CancellationToken::new(),
            )
            .await
            .expect("turn succeeds");

        let results: String = messages
            .iter()
            .flat_map(|message| message.content.iter())
            .filter_map(|block| match block {
                ContentBlock::ToolResult { content, .. } => {
                    Some(ContentBlock::tool_result_text_content(content))
                }
                _ => None,
            })
            .collect();
        assert!(results.contains("which is the limit"), "{results}");
        assert!(results.contains("without `background`"), "{results}");

        agent.background_tasks().cancel_all().await;
    }

    /// `background` is meka's own. A tool must never see it, least of all a remote MCP server that
    /// never advertised the key.
    #[tokio::test]
    async fn test_the_background_flag_never_reaches_the_tool() {
        use crate::provider::mock::MockProvider;

        struct RecordingFixture(Arc<std::sync::Mutex<Option<serde_json::Value>>>);

        #[async_trait::async_trait]
        impl crate::tools::Tool for RecordingFixture {
            fn definition(&self) -> ToolDefinition {
                SlowFixture.definition()
            }

            fn required_permission(&self) -> crate::permission::Permission {
                crate::permission::Permission::Read
            }

            async fn execute(
                &self,
                input: serde_json::Value,
                _cancellation: CancellationToken,
            ) -> Result<crate::tools::ToolOutput> {
                *self.0.lock().unwrap_or_else(|p| p.into_inner()) = Some(input);
                Ok(crate::tools::ToolOutput::text("ok".to_string(), false))
            }
        }

        let seen = Arc::new(std::sync::Mutex::new(None));
        let provider = Arc::new(MockProvider::from_rounds(vec![
            background_round("make"),
            text_round("started"),
        ]));
        let (agent, _session_manager) = background_agent(
            provider,
            Arc::new(RecordingFixture(Arc::clone(&seen))) as Arc<dyn crate::tools::Tool>,
        )
        .await;

        let mut session_id = None;
        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut session_id,
                &mut messages,
                "build".to_string(),
                Vec::new(),
                CancellationToken::new(),
            )
            .await
            .expect("turn succeeds");
        agent
            .background_tasks()
            .wait_for_session(session_id.expect("session"))
            .await;

        let seen = seen.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let seen = seen.expect("the tool ran");
        assert_eq!(seen, serde_json::json!({"command": "make"}));
    }

    /// A wrong-typed `background` refuses the call and says what to send, rather than running it in
    /// the foreground. Models that stringify every argument (GLM through OpenRouter does) would
    /// otherwise get a twenty-minute block where they asked for a detach, with nothing in the
    /// transcript accounting for it; here they are told, and the retry is theirs to make.
    #[tokio::test]
    async fn test_a_wrong_typed_background_flag_refuses_the_call() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let ran = Arc::new(std::sync::atomic::AtomicBool::new(false));

        struct WitnessFixture(Arc<std::sync::atomic::AtomicBool>);

        #[async_trait::async_trait]
        impl crate::tools::Tool for WitnessFixture {
            fn definition(&self) -> ToolDefinition {
                SlowFixture.definition()
            }

            fn required_permission(&self) -> crate::permission::Permission {
                crate::permission::Permission::Read
            }

            async fn execute(
                &self,
                _input: serde_json::Value,
                _cancellation: CancellationToken,
            ) -> Result<crate::tools::ToolOutput> {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok(crate::tools::ToolOutput::text("ok".to_string(), false))
            }
        }

        let stringified = vec![
            MockEvent::ToolUseStart {
                id: "call-1".to_string(),
                name: "execute_command".to_string(),
            },
            MockEvent::ToolUseEnd {
                input: serde_json::json!({"command": "make", "background": "true"}),
            },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::ToolUse,
            },
        ];
        let provider = Arc::new(MockProvider::from_rounds(vec![
            stringified,
            text_round("understood"),
        ]));
        let (agent, _session_manager) = background_agent(
            provider,
            Arc::new(WitnessFixture(Arc::clone(&ran))) as Arc<dyn crate::tools::Tool>,
        )
        .await;

        let mut session_id = None;
        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut session_id,
                &mut messages,
                "build".to_string(),
                Vec::new(),
                CancellationToken::new(),
            )
            .await
            .expect("turn succeeds");

        assert!(
            !ran.load(std::sync::atomic::Ordering::SeqCst),
            "a call meka could not read must not run at all, least of all in the foreground",
        );
        let results: String = messages
            .iter()
            .flat_map(|message| message.content.iter())
            .filter_map(|block| match block {
                ContentBlock::ToolResult { content, .. } => {
                    Some(ContentBlock::tool_result_text_content(content))
                }
                _ => None,
            })
            .collect();
        assert!(results.contains("background"), "{results}");
        assert!(results.contains("boolean"), "{results}");
    }

    /// A call that never executed must not spend the tool's one advisory. Otherwise a permission
    /// denial swallows the hint, and the retry after the user grants permission is exactly the
    /// silent call this machinery exists to prevent.
    #[tokio::test]
    async fn test_a_call_that_did_not_run_keeps_its_advisory_slot() {
        use crate::provider::mock::MockProvider;

        let (agent, _session_manager) = test_agent_with_registry(
            Arc::new(MockProvider::from_rounds(vec![])),
            send_file_registry(),
        )
        .await;
        let name = "mcp__bridge__send_file";
        let schema = crate::tools::Tool::definition(&SendFileFixture).parameters;
        let input = serde_json::json!({"path": "/tmp/a.png"});

        let denied = agent.schema_advisory(name, &input, &schema, &[], false);
        assert!(denied.is_some(), "a denied call is still told");

        let retried = agent.schema_advisory(name, &input, &schema, &[], true);
        assert!(retried.is_some(), "the denial must not have spent the slot");

        let third = agent.schema_advisory(name, &input, &schema, &[], true);
        assert!(third.is_none(), "but a call that ran spends it");
    }

    /// Once the model has loaded the schema, the same call is its own business.
    #[tokio::test]
    async fn test_loaded_tool_call_gets_no_advisory() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let registry = send_file_registry();
        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![
                MockEvent::ToolUseStart {
                    id: "load-1".to_string(),
                    name: crate::tools::LOAD_TOOL_NAME.to_string(),
                },
                MockEvent::ToolUseEnd {
                    input: serde_json::json!({"name": "mcp__bridge__send_file"}),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::ToolUse,
                },
            ],
            send_file_round(serde_json::json!({"path": "/tmp/a.png"})),
            vec![
                MockEvent::Text {
                    text: "sent".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
        ]));
        let (agent, _session_manager) = test_agent_with_registry(provider, registry).await;

        let mut session_id = None;
        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut session_id,
                &mut messages,
                "send the picture".to_string(),
                Vec::new(),
                CancellationToken::new(),
            )
            .await
            .expect("turn succeeds");

        let results: String = messages
            .iter()
            .flat_map(|message| message.content.iter())
            .filter_map(|block| match block {
                ContentBlock::ToolResult { content, .. } => {
                    Some(ContentBlock::tool_result_text_content(content))
                }
                _ => None,
            })
            .collect();

        assert!(results.contains("Sent (message id 1)"), "{results}");
        assert!(
            !results.contains("[meka]"),
            "the schema was loaded; nothing to advise: {results}"
        );
    }

    /// A 400 that isn't about content (`max_tokens` over the ceiling, a bad header) has nothing to
    /// degrade, so it must fail immediately rather than spend a retry.
    #[tokio::test]
    async fn test_run_turn_does_not_retry_a_rejection_with_nothing_to_degrade() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![MockEvent::FailInvalidRequest {
                message: "max_tokens: 999999 > 8192, the maximum for this model".to_string(),
            }],
            // Reaching this round would mean a retry was spent on an unrepairable request.
            vec![
                MockEvent::Text {
                    text: "should never run".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
        ]));
        let (agent, _session_manager) = test_agent(provider).await;

        let mut session_id = None;
        let mut messages = Conversation::new();
        let error = agent
            .run_turn(
                &mut session_id,
                &mut messages,
                "plain text only".to_string(),
                Vec::new(),
                CancellationToken::new(),
            )
            .await
            .expect_err("nothing to repair, so the turn fails");
        assert!(matches!(error, MekaError::InvalidRequest(_)), "{error}");
    }

    fn retryable_error() -> MekaError {
        MekaError::RetryableProvider {
            message: "overloaded".to_string(),
            retry_after: None,
        }
    }

    fn tool_result_with_image(tool_use_id: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: tool_use_id.to_string(),
                content: vec![
                    ToolResultContent::Text {
                        text: "[Image: smoketest.png]".to_string(),
                    },
                    ToolResultContent::Image {
                        source: ImageSource {
                            source_type: "base64".to_string(),
                            media_type: "image/png".to_string(),
                            data: "QUJD".to_string(),
                        },
                    },
                ],
                is_error: false,
            }],
        }
    }

    #[test]
    fn test_degrade_rejected_content_replaces_the_image_and_keeps_the_pairing() {
        let assistant = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
                input: serde_json::json!({"path": "smoketest.png"}),
            }],
        };
        let degraded = degrade_rejected_content(
            &[assistant, tool_result_with_image("call_1")],
            "the image appears to be a image/jpeg image",
        )
        .expect("there was non-text content to degrade");

        assert_eq!(degraded.len(), 2, "the message count must not change");
        // The tool_use survives: the tool already ran, so erasing the record would invite a rerun.
        assert!(matches!(
            &degraded[0].content[0],
            ContentBlock::ToolUse { id, .. } if id == "call_1"
        ));
        match &degraded[1].content[0] {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                assert_eq!(tool_use_id, "call_1", "the pairing must survive");
                assert!(is_error, "the model has to see this as a failed call");
                assert!(
                    content
                        .iter()
                        .all(|item| matches!(item, ToolResultContent::Text { .. })),
                    "no non-text content may remain"
                );
                let text: String = content
                    .iter()
                    .map(|item| match item {
                        ToolResultContent::Text { text } => text.clone(),
                        _ => String::new(),
                    })
                    .collect();
                assert!(
                    text.contains("[Image: smoketest.png]"),
                    "keeps the text: {text}"
                );
                assert!(text.contains("image/jpeg"), "carries the reason: {text}");
                assert!(
                    text.contains("Do not repeat this call unchanged"),
                    "tells the model not to loop: {text}"
                );
            }
            other => panic!("expected ToolResult, got {:?}", other),
        }
    }

    #[test]
    fn test_degrade_rejected_content_replaces_a_user_input_image() {
        let attached = Message::user_with_images("look at this".to_string(), vec![ImageSource {
            source_type: "base64".to_string(),
            media_type: "image/png".to_string(),
            data: "QUJD".to_string(),
        }]);
        let degraded = degrade_rejected_content(&[attached], "refused").expect("degraded");
        assert!(
            degraded[0]
                .content
                .iter()
                .all(|block| matches!(block, ContentBlock::Text { .. }))
        );
    }

    /// The signal that a rejection is *not* about content, which is what stops the loop from
    /// spending a retry on a `max_tokens` or bad-header error.
    #[test]
    fn test_degrade_rejected_content_reports_nothing_to_do_for_text_only() {
        let messages = vec![
            Message::user("plain text"),
            Message::assistant_text("also plain"),
        ];
        assert!(degrade_rejected_content(&messages, "refused").is_none());
        assert!(degrade_rejected_content(&[], "refused").is_none());
    }

    #[test]
    fn test_elide_reason_caps_a_provider_echoing_the_request_body() {
        let long = "x".repeat(REJECTION_REASON_LIMIT * 2);
        let elided = elide_reason(&long);
        assert_eq!(elided.chars().count(), REJECTION_REASON_LIMIT + 1);
        assert!(elided.ends_with('…'));
        assert_eq!(elide_reason("short"), "short");
    }

    /// Multi-byte input must not be sliced mid-character.
    #[test]
    fn test_elide_reason_respects_char_boundaries() {
        let long = "é".repeat(REJECTION_REASON_LIMIT + 10);
        assert_eq!(
            elide_reason(&long).chars().count(),
            REJECTION_REASON_LIMIT + 1
        );
    }

    #[test]
    fn test_should_retry_provider_error_retries_when_no_content_and_under_cap() {
        let delay = should_retry_provider_error(&retryable_error(), false, 0);
        assert!(delay.is_some());
    }

    #[test]
    fn test_should_retry_provider_error_stops_once_content_started() {
        // The core safety property: once the user has seen any output this attempt, a retryable
        // error must not trigger a retry (would duplicate/corrupt what's already shown).
        assert_eq!(
            should_retry_provider_error(&retryable_error(), true, 0),
            None
        );
    }

    #[test]
    fn test_should_retry_provider_error_stops_at_retry_cap() {
        assert_eq!(
            should_retry_provider_error(
                &retryable_error(),
                false,
                crate::provider::retry::MAX_PROVIDER_RETRIES
            ),
            None
        );
        // One below the cap still retries.
        assert!(
            should_retry_provider_error(
                &retryable_error(),
                false,
                crate::provider::retry::MAX_PROVIDER_RETRIES - 1
            )
            .is_some()
        );
    }

    #[test]
    fn test_should_retry_provider_error_retries_stream_error_before_output() {
        // A mid-stream transport failure (SSE decode error, dropped connection, idle timeout) is
        // retryable under the same content-started / cap guards as a RetryableProvider error.
        let stream_error = MekaError::StreamError("error decoding response body".to_string());
        assert!(should_retry_provider_error(&stream_error, false, 0).is_some());
        assert_eq!(should_retry_provider_error(&stream_error, true, 0), None);
        assert_eq!(
            should_retry_provider_error(
                &stream_error,
                false,
                crate::provider::retry::MAX_PROVIDER_RETRIES
            ),
            None
        );
    }

    #[test]
    fn test_should_retry_provider_error_ignores_non_retryable_errors() {
        assert_eq!(
            should_retry_provider_error(&MekaError::Provider("bad request".to_string()), false, 0),
            None
        );
        assert_eq!(
            should_retry_provider_error(
                &MekaError::ContextOverflow("too long".to_string()),
                false,
                0
            ),
            None
        );
    }

    #[test]
    fn test_should_retry_provider_error_uses_retry_after_hint() {
        let error = MekaError::RetryableProvider {
            message: "rate limited".to_string(),
            retry_after: Some(std::time::Duration::from_secs(5)),
        };
        assert_eq!(
            should_retry_provider_error(&error, false, 0),
            Some(std::time::Duration::from_secs(5))
        );
    }

    #[test]
    fn test_empty_turn_notice_includes_unknown_stop_reason() {
        assert_eq!(
            empty_turn_notice(&StopReason::Refusal("custom refusal".to_string())),
            "custom refusal"
        );
        assert!(
            empty_turn_notice(&StopReason::Refusal(String::new())).contains("declined to respond")
        );
        assert!(empty_turn_notice(&StopReason::MaxTokens).contains("output limit"));
        // The raw reason of an unrecognised stop reason must be surfaced, not swallowed.
        let notice = empty_turn_notice(&StopReason::Unknown("pause_turn".to_string()));
        assert!(notice.contains("pause_turn"), "got: {notice}");
        assert!(empty_turn_notice(&StopReason::EndTurn).contains("empty response"));
    }

    #[test]
    fn test_has_visible_text() {
        assert!(!has_visible_text(&[]));
        assert!(!has_visible_text(&[ContentBlock::Thinking {
            thinking: "pondering".to_string(),
            signature: None,
        }]));
        // Whitespace-only text is not visible output.
        assert!(!has_visible_text(&[ContentBlock::Text {
            text: "   \n".to_string(),
        }]));
        assert!(!has_visible_text(&[ContentBlock::ToolUse {
            id: "call_1".to_string(),
            name: "read_file".to_string(),
            input: serde_json::json!({}),
        }]));
        assert!(has_visible_text(&[ContentBlock::Text {
            text: "hello".to_string(),
        }]));
        // A thinking block followed by real text still counts as visible.
        assert!(has_visible_text(&[
            ContentBlock::Thinking {
                thinking: "pondering".to_string(),
                signature: None,
            },
            ContentBlock::Text {
                text: "answer".to_string(),
            },
        ]));
    }

    fn not_connected(name: &str, required: bool) -> crate::mcp::NotConnected {
        crate::mcp::NotConnected {
            name: name.to_string(),
            required,
            state: crate::mcp::ServerState::Failed {
                error: "boom".to_string(),
                at: std::time::Instant::now(),
            },
        }
    }

    /// The point of the change: an optional server that is down must not stop the session. A
    /// container without `ida-mcp` should still run every turn that doesn't need IDA.
    #[test]
    fn test_optional_servers_do_not_gate_the_turn() {
        assert!(gate_on_required_servers(vec![]).is_ok());
        assert!(
            gate_on_required_servers(vec![
                not_connected("ida", false),
                not_connected("exa", false)
            ])
            .is_ok()
        );
    }

    /// The rejection must name the cause, not just "failed": it is the only thing the user sees
    /// when a required server blocks every turn, and the connector's warn fires once and stops.
    #[test]
    fn test_required_server_gates_the_turn() {
        let error = gate_on_required_servers(vec![not_connected("bridge", true)])
            .expect_err("a required server must gate");
        match error {
            MekaError::McpTurnGated { servers } => {
                assert_eq!(servers.len(), 1);
                assert_eq!(servers[0].0, "bridge");
                // The cause, not the "failed" label: every cause already reads as a failure, so
                // the label would only produce "failed: failed to ...".
                assert_eq!(servers[0].1, "boom");
            }
            other => panic!("expected McpTurnGated, got {other:?}"),
        }
        assert!(
            gate_on_required_servers(vec![crate::mcp::NotConnected {
                name: "slow".to_string(),
                required: true,
                state: crate::mcp::ServerState::Pending,
            }])
            .expect_err("pending required server still gates")
            .to_string()
            .contains("pending")
        );
    }

    /// A mixed fleet gates on the required one and names only it: listing the optional servers
    /// would imply they are the problem.
    #[test]
    fn test_gate_names_only_the_required_servers() {
        let error = gate_on_required_servers(vec![
            not_connected("ida", false),
            not_connected("bridge", true),
            not_connected("exa", false),
        ])
        .expect_err("a required server must gate even alongside optional ones");
        match error {
            MekaError::McpTurnGated { servers } => {
                let names: Vec<&str> = servers.iter().map(|(n, _)| n.as_str()).collect();
                assert_eq!(names, vec!["bridge"]);
            }
            other => panic!("expected McpTurnGated, got {other:?}"),
        }
    }

    #[test]
    fn test_should_nudge_thinking_only() {
        // Thinking-only end_turn: nudge once.
        assert!(should_nudge_thinking_only(
            false,
            false,
            &StopReason::EndTurn,
            false
        ));
        // Thinking-only unrecognized reason (e.g. pause_turn): nudge once.
        assert!(should_nudge_thinking_only(
            false,
            false,
            &StopReason::Unknown("pause_turn".to_string()),
            false,
        ));
        // Already nudged this turn: no second nudge (prevents loops).
        assert!(!should_nudge_thinking_only(
            false,
            false,
            &StopReason::EndTurn,
            true
        ));
        // Visible text present: nothing to recover.
        assert!(!should_nudge_thinking_only(
            false,
            true,
            &StopReason::EndTurn,
            false
        ));
        // Tool calls present: the tool path drives continuation.
        assert!(!should_nudge_thinking_only(
            true,
            false,
            &StopReason::EndTurn,
            false
        ));
        // MaxTokens and Refusal carry their own outcomes; don't retry them.
        assert!(!should_nudge_thinking_only(
            false,
            false,
            &StopReason::MaxTokens,
            false
        ));
        assert!(!should_nudge_thinking_only(
            false,
            false,
            &StopReason::Refusal(String::new()),
            false,
        ));
    }

    fn user_msg(text: &str) -> Message {
        Message::user(text)
    }

    fn assistant_msg(text: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
        }
    }

    fn assistant_tool_use() -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
                input: serde_json::json!({"path": "/tmp/test"}),
            }],
        }
    }

    fn tool_result_msg() -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: vec![ToolResultContent::Text {
                    text: "file contents".to_string(),
                }],
                is_error: false,
            }],
        }
    }

    /// `cwd` leads and duplicates are dropped: a client is free to repeat `cwd` inside
    /// `additionalDirectories`, and a repeated root would double every search result and spend the
    /// shared walk budget twice on the same tree.
    #[test]
    fn test_search_roots_puts_cwd_first_and_dedupes() {
        let cwd: SharedCwd = Arc::new(RwLock::new(PathBuf::from("/work/main")));
        let roots: SharedRoots = Arc::new(RwLock::new(vec![
            PathBuf::from("/work/shared"),
            PathBuf::from("/work/main"),
            PathBuf::from("/work/shared"),
            PathBuf::from("/work/docs"),
        ]));

        assert_eq!(search_roots(&cwd, &roots), vec![
            PathBuf::from("/work/main"),
            PathBuf::from("/work/shared"),
            PathBuf::from("/work/docs"),
        ]);
    }

    /// A root nested inside another is covered by it, so keeping both walks that tree twice and
    /// reports every file in it twice.
    #[test]
    fn test_search_roots_drops_roots_nested_in_another() {
        let cwd: SharedCwd = Arc::new(RwLock::new(PathBuf::from("/work/main")));
        let roots: SharedRoots = Arc::new(RwLock::new(vec![
            PathBuf::from("/work/main/nested"),
            PathBuf::from("/work/other"),
        ]));

        assert_eq!(search_roots(&cwd, &roots), vec![
            PathBuf::from("/work/main"),
            PathBuf::from("/work/other"),
        ]);
    }

    /// And the inverse: a root that *contains* `cwd` wins, because its walk already reaches
    /// everything under `cwd`. Dropping `cwd` from the search set is safe; it stays the base for
    /// relative paths and the shell either way.
    #[test]
    fn test_search_roots_lets_an_ancestor_root_subsume_cwd() {
        let cwd: SharedCwd = Arc::new(RwLock::new(PathBuf::from("/work/main")));
        let roots: SharedRoots = Arc::new(RwLock::new(vec![PathBuf::from("/work")]));
        assert_eq!(search_roots(&cwd, &roots), vec![PathBuf::from("/work")]);
    }

    /// A shared prefix is not containment: `/work/main2` is not inside `/work/main`.
    #[test]
    fn test_search_roots_keeps_sibling_with_shared_prefix() {
        let cwd: SharedCwd = Arc::new(RwLock::new(PathBuf::from("/work/main")));
        let roots: SharedRoots = Arc::new(RwLock::new(vec![PathBuf::from("/work/main2")]));
        assert_eq!(search_roots(&cwd, &roots), vec![
            PathBuf::from("/work/main"),
            PathBuf::from("/work/main2"),
        ]);
    }

    /// The single-root case has to stay exactly one path: that is every REPL and HTTP session, and
    /// every ACP client that sends no extra roots.
    #[test]
    fn test_search_roots_without_extras_is_just_cwd() {
        let cwd: SharedCwd = Arc::new(RwLock::new(PathBuf::from("/work/main")));
        let roots: SharedRoots = Arc::new(RwLock::new(Vec::new()));
        assert_eq!(search_roots(&cwd, &roots), vec![PathBuf::from(
            "/work/main"
        )]);
    }

    #[test]
    fn test_resolve_against_cwd_passes_absolute_paths_through() {
        let cwd: SharedCwd = Arc::new(RwLock::new(PathBuf::from("/home/agent")));
        let absolute = std::path::Path::new("/etc/hosts");
        let resolved = resolve_against_cwd(&cwd, absolute);
        assert_eq!(resolved, PathBuf::from("/etc/hosts"));
    }

    #[test]
    fn test_resolve_against_cwd_joins_relative_paths_to_session_cwd() {
        let cwd: SharedCwd = Arc::new(RwLock::new(PathBuf::from("/home/agent/project")));
        let resolved = resolve_against_cwd(&cwd, "src/main.rs");
        assert_eq!(resolved, PathBuf::from("/home/agent/project/src/main.rs"));
    }

    #[test]
    fn test_resolve_against_cwd_follows_subsequent_writes() {
        // Confirms multiple sessions in one process would observe their own cwds: a write to the
        // shared lock is visible on the next resolve, without touching process cwd.
        let cwd: SharedCwd = Arc::new(RwLock::new(PathBuf::from("/tmp/a")));
        let first = resolve_against_cwd(&cwd, "foo.txt");
        *cwd.write().expect("cwd lock") = PathBuf::from("/tmp/b");
        let second = resolve_against_cwd(&cwd, "foo.txt");
        assert_eq!(first, PathBuf::from("/tmp/a/foo.txt"));
        assert_eq!(second, PathBuf::from("/tmp/b/foo.txt"));
    }

    #[test]
    fn test_truncate_no_limit() {
        let messages = vec![user_msg("hello"), assistant_msg("hi")];
        let result = truncate_messages_for_context(&messages, None);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_truncate_under_limit() {
        let messages = vec![user_msg("hello"), assistant_msg("hi")];
        let result = truncate_messages_for_context(&messages, Some(10));
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_truncate_over_limit() {
        let messages = vec![
            user_msg("first"),
            assistant_msg("response1"),
            user_msg("second"),
            assistant_msg("response2"),
            user_msg("third"),
            assistant_msg("response3"),
        ];
        let result = truncate_messages_for_context(&messages, Some(4));
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].role, Role::User);
    }

    #[test]
    fn test_truncate_does_not_split_tool_chain() {
        let messages = vec![
            user_msg("first"),
            assistant_msg("response1"),
            user_msg("second"),
            assistant_tool_use(),
            tool_result_msg(),
            assistant_msg("final"),
        ];
        // Limit 3 would naively start at index 3 (assistant_tool_use), but that splits the tool
        // chain. It should walk back to index 2 (user "second").
        let result = truncate_messages_for_context(&messages, Some(3));
        assert_eq!(result[0].role, Role::User);
        assert!(!has_tool_results(&result[0].content));
        assert!(result.len() >= 3);
    }

    #[test]
    fn test_truncate_starts_with_user() {
        let messages = vec![
            user_msg("first"),
            assistant_msg("response1"),
            assistant_msg("response2"),
            user_msg("second"),
            assistant_msg("response3"),
        ];
        // Limit 2 would naively start at index 3, which is a user message
        let result = truncate_messages_for_context(&messages, Some(2));
        assert_eq!(result[0].role, Role::User);
    }

    #[test]
    fn test_truncate_walks_back_past_tool_result() {
        let messages = vec![
            user_msg("first"),
            assistant_tool_use(),
            tool_result_msg(),
            assistant_msg("response"),
            user_msg("second"),
            assistant_msg("response2"),
        ];
        // Limit 4 would naively start at index 2 (tool_result_msg), should walk back to index 0
        // (user "first")
        let result = truncate_messages_for_context(&messages, Some(4));
        assert_eq!(result[0].role, Role::User);
        assert!(!has_tool_results(&result[0].content));
    }

    #[test]
    fn test_compaction_split_small_summarizes_all() {
        let messages = vec![user_msg("a"), assistant_msg("b"), user_msg("c")];
        let (head, tail) = compute_compaction_split(&messages, 10_000);
        assert_eq!(head.len(), 3);
        assert!(tail.is_empty());
    }

    #[test]
    fn test_compaction_split_keeps_recent_tail_within_budget() {
        let mut messages = Vec::new();
        for i in 0..6 {
            messages.push(user_msg(&format!("user {i}")));
            messages.push(assistant_msg(&format!("assistant {i}")));
        }
        let (head, tail) = compute_compaction_split(&messages, 30);
        // The split partitions the whole conversation.
        assert_eq!(head.len() + tail.len(), messages.len());
        // A small budget keeps only a recent slice, leaving a real head to summarize.
        assert!(head.len() >= 4);
        assert!(!tail.is_empty() && tail.len() < messages.len());
        // The kept window starts on a clean user boundary.
        assert_eq!(tail[0].role, Role::User);
        assert!(!has_tool_results(&tail[0].content));
    }

    #[test]
    fn test_compaction_split_does_not_orphan_tool_results() {
        let messages = vec![
            user_msg("first"),
            assistant_msg("r1"),
            user_msg("second"),
            assistant_msg("r2"),
            user_msg("third"),
            assistant_tool_use(),
            tool_result_msg(),
            assistant_msg("final"),
        ];
        // A budget that naively cuts inside the assistant(tool_use)->user(tool_result) chain must
        // snap back to the user boundary before it.
        let (head, tail) = compute_compaction_split(&messages, 20);
        assert_eq!(head.len() + tail.len(), messages.len());
        assert_eq!(tail[0].role, Role::User);
        assert!(!has_tool_results(&tail[0].content));
    }

    // Cache prefix stability tests. These tests simulate the agent's message-assembly logic (stable
    // base + appended tool-loop messages) to verify that the prefix sent to the API remains
    // identical across iterations of the tool-use loop.  This is the core invariant required for KV
    // cache reuse.

    fn assistant_tool_use_named(id: &str, name: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: id.to_string(),
                name: name.to_string(),
                input: serde_json::json!({"path": "/tmp/test"}),
            }],
        }
    }

    fn tool_result_for(tool_use_id: &str, content: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: tool_use_id.to_string(),
                content: vec![ToolResultContent::Text {
                    text: content.to_string(),
                }],
                is_error: false,
            }],
        }
    }

    /// Compares two message slices for semantic equality (same role, same content blocks).  This is
    /// what determines whether the KV cache prefix is reusable.
    fn assert_messages_equal(a: &[Message], b: &[Message], context: &str) {
        assert_eq!(a.len(), b.len(), "{}: length mismatch", context);
        for (i, (ma, mb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                ma.role, mb.role,
                "{}: role mismatch at index {}",
                context, i
            );
            assert_eq!(
                ma.content.len(),
                mb.content.len(),
                "{}: content block count mismatch at index {}",
                context,
                i
            );
            let json_a = serde_json::to_string(&ma.content).unwrap();
            let json_b = serde_json::to_string(&mb.content).unwrap();
            assert_eq!(
                json_a, json_b,
                "{}: content mismatch at index {}",
                context, i
            );
        }
    }

    /// Simulates the tool-loop message assembly logic from `run_turn`:
    ///   base_messages = truncate(messages, limit)   // computed once
    ///   turn_start_len = messages.len()
    ///   loop { api_messages = base + messages[turn_start_len..] }
    fn build_api_messages(
        messages: &[Message],
        base_messages: &[Message],
        turn_start_len: usize,
    ) -> Vec<Message> {
        if messages.len() > turn_start_len {
            let mut combined = base_messages.to_vec();
            combined.extend_from_slice(&messages[turn_start_len..]);
            combined
        } else {
            base_messages.to_vec()
        }
    }

    #[test]
    fn test_stable_base_during_tool_loop() {
        // Simulate a conversation with history, then a tool loop that adds 3 tool call/result
        // pairs.  The base prefix (everything before the tool loop) must be identical across all
        // iterations.
        let mut messages = vec![
            user_msg("first question"),
            assistant_msg("first answer"),
            user_msg("second question"),
        ];

        let base_messages = truncate_messages_for_context(&messages, None);
        let turn_start_len = messages.len();

        let api_iter0 = build_api_messages(&messages, &base_messages, turn_start_len);
        assert_eq!(api_iter0.len(), 3);

        // Iteration 1: model calls a tool
        messages.push(assistant_tool_use_named("t1", "read_file"));
        messages.push(tool_result_for("t1", "file contents"));

        let api_iter1 = build_api_messages(&messages, &base_messages, turn_start_len);
        assert_eq!(api_iter1.len(), 5);

        // The first 3 messages (the base) must be identical.
        assert_messages_equal(&api_iter0[..3], &api_iter1[..3], "iter0→iter1 base");

        // Iteration 2: model calls another tool
        messages.push(assistant_tool_use_named("t2", "execute_command"));
        messages.push(tool_result_for("t2", "command output"));

        let api_iter2 = build_api_messages(&messages, &base_messages, turn_start_len);
        assert_eq!(api_iter2.len(), 7);

        // Base is still identical.
        assert_messages_equal(&api_iter0[..3], &api_iter2[..3], "iter0→iter2 base");
        // And the first 5 (base + iter1's additions) are identical too.
        assert_messages_equal(&api_iter1[..5], &api_iter2[..5], "iter1→iter2 prefix");

        // Iteration 3: yet another tool call
        messages.push(assistant_tool_use_named("t3", "read_file"));
        messages.push(tool_result_for("t3", "more contents"));

        let api_iter3 = build_api_messages(&messages, &base_messages, turn_start_len);
        assert_eq!(api_iter3.len(), 9);

        assert_messages_equal(&api_iter2[..7], &api_iter3[..7], "iter2→iter3 prefix");
        assert_messages_equal(&api_iter0[..3], &api_iter3[..3], "iter0→iter3 base");
    }

    #[test]
    fn test_truncation_boundary_does_not_shift_during_tool_loop() {
        // This is the critical test for the fix: when context_messages is set and we're near the
        // limit, adding tool results within the loop must NOT cause the truncated prefix to shift.
        // Before the fix, truncation was recomputed inside the loop, causing prefix instability.
        let limit = Some(6);

        // Start with 5 messages (under the limit of 6).
        let mut messages = vec![
            user_msg("msg-1"),
            assistant_msg("resp-1"),
            user_msg("msg-2"),
            assistant_msg("resp-2"),
            user_msg("msg-3"),
        ];

        // Compute the stable base ONCE before the loop (as run_turn does).
        let base_messages = truncate_messages_for_context(&messages, limit);
        let turn_start_len = messages.len();

        // All 5 messages fit within the limit; no truncation yet.
        assert_eq!(base_messages.len(), 5);

        let api_iter0 = build_api_messages(&messages, &base_messages, turn_start_len);
        assert_eq!(api_iter0.len(), 5);

        // Iteration 1: add tool call + result → 7 messages total, over limit. With the old code,
        // truncation would kick in and drop messages from the front.  With the new code, the base
        // is frozen.
        messages.push(assistant_tool_use_named("t1", "read_file"));
        messages.push(tool_result_for("t1", "data"));

        let api_iter1 = build_api_messages(&messages, &base_messages, turn_start_len);
        // Should be base(5) + new(2) = 7
        assert_eq!(api_iter1.len(), 7);

        // The first 5 messages must be identical to iter0.
        assert_messages_equal(&api_iter0[..5], &api_iter1[..5], "iter0→iter1 base");

        // Iteration 2: add another tool call → 9 total, well over limit.
        messages.push(assistant_tool_use_named("t2", "execute_command"));
        messages.push(tool_result_for("t2", "output"));

        let api_iter2 = build_api_messages(&messages, &base_messages, turn_start_len);
        assert_eq!(api_iter2.len(), 9);

        // The first 7 messages must match iter1 exactly.
        assert_messages_equal(&api_iter1[..7], &api_iter2[..7], "iter1→iter2 prefix");
        // And the base (first 5) is still untouched.
        assert_messages_equal(&api_iter0[..5], &api_iter2[..5], "iter0→iter2 base");
    }

    #[test]
    fn test_truncation_with_tool_chain_near_boundary() {
        // Verify that when the conversation includes a tool chain right at the truncation boundary,
        // the base is computed correctly and stays stable.
        let limit = Some(4);

        let mut messages = vec![
            user_msg("old-msg"),
            assistant_msg("old-resp"),
            user_msg("current question"),
            assistant_tool_use_named("t0", "read_file"),
            tool_result_for("t0", "initial data"),
            assistant_msg("here is the data"),
            user_msg("follow-up"),
        ];

        let base_messages = truncate_messages_for_context(&messages, limit);
        let turn_start_len = messages.len();

        // The truncation should keep a safe cut point; verify it starts with a user message and
        // doesn't split tool chains.
        assert_eq!(base_messages[0].role, Role::User);
        assert!(!has_tool_results(&base_messages[0].content));

        let api_iter0 = build_api_messages(&messages, &base_messages, turn_start_len);

        // Add tool loop messages
        messages.push(assistant_tool_use_named("t1", "read_file"));
        messages.push(tool_result_for("t1", "more data"));

        let api_iter1 = build_api_messages(&messages, &base_messages, turn_start_len);

        // The base portion must be identical.
        let base_len = base_messages.len();
        assert_messages_equal(
            &api_iter0[..base_len],
            &api_iter1[..base_len],
            "base stable after tool loop",
        );
    }

    /// Regression: the tool catalogue, skill list, and MCP instructions used to live in the system
    /// prompt, which is sent unconditionally. They now live in one user message, which
    /// `truncate_messages_for_context` will drop once the conversation outgrows
    /// `context_messages` (200 by default). Without this check the snapshot would still claim the
    /// model had been told, and a long session would run with no catalogue at all.
    #[test]
    fn test_world_state_is_restated_once_it_scrolls_out_of_the_window() {
        // Rendered at index 0, window of 200.
        assert!(
            world_state_still_visible(0, 10, Some(200)),
            "a fresh render is visible"
        );
        assert!(
            world_state_still_visible(0, 199, Some(200)),
            "still inside the window one message before the cut"
        );
        assert!(
            !world_state_still_visible(0, 200, Some(200)),
            "the render has reached the edge and must be restated"
        );
        assert!(
            !world_state_still_visible(0, 5_000, Some(200)),
            "a long session must not run on a render that scrolled away"
        );

        // A restatement lands at the current tail and buys another window.
        assert!(world_state_still_visible(4_900, 5_000, Some(200)));

        // No limit means nothing is ever dropped, so a single render lasts the session.
        assert!(world_state_still_visible(0, 100_000, None));
    }

    #[test]
    fn test_no_limit_produces_full_prefix() {
        // With no context_messages limit, base_messages includes everything, and tool loop
        // additions are appended without any truncation.
        let mut messages = vec![user_msg("a"), assistant_msg("b"), user_msg("c")];

        let base_messages = truncate_messages_for_context(&messages, None);
        let turn_start_len = messages.len();

        assert_eq!(base_messages.len(), 3);

        let api_iter0 = build_api_messages(&messages, &base_messages, turn_start_len);
        assert_eq!(api_iter0.len(), 3);

        // Add many tool calls
        for i in 0..5 {
            messages.push(assistant_tool_use_named(&format!("t{}", i), "read_file"));
            messages.push(tool_result_for(
                &format!("t{}", i),
                &format!("result {}", i),
            ));
        }

        let api_final = build_api_messages(&messages, &base_messages, turn_start_len);
        assert_eq!(api_final.len(), 13); // 3 base + 10 tool messages

        // Base prefix still matches.
        assert_messages_equal(&api_iter0[..3], &api_final[..3], "full prefix stable");
    }

    #[test]
    fn test_multi_turn_with_truncation_each_turn_gets_stable_base() {
        // Simulate multiple turns, each computing its own stable base. Verify that within each
        // turn's tool loop the base stays fixed, and that across turns the overlapping messages are
        // consistent.
        let limit = Some(6);

        // -- Turn 1 --
        let mut messages: Vec<Message> = vec![user_msg("turn-1 question")];
        let base_t1 = truncate_messages_for_context(&messages, limit);
        let start_t1 = messages.len();

        // Tool loop: 2 iterations
        messages.push(assistant_tool_use_named("t1a", "read_file"));
        messages.push(tool_result_for("t1a", "data-a"));
        let api_t1_iter1 = build_api_messages(&messages, &base_t1, start_t1);

        messages.push(assistant_msg("here's your answer"));
        let api_t1_iter2 = build_api_messages(&messages, &base_t1, start_t1);

        // Base is stable within turn 1.
        assert_messages_equal(
            &api_t1_iter1[..base_t1.len()],
            &api_t1_iter2[..base_t1.len()],
            "turn 1 base stable",
        );

        // -- Turn 2 --
        messages.push(user_msg("turn-2 question"));

        let base_t2 = truncate_messages_for_context(&messages, limit);
        let start_t2 = messages.len();

        messages.push(assistant_tool_use_named("t2a", "execute_command"));
        messages.push(tool_result_for("t2a", "output"));
        let api_t2_iter1 = build_api_messages(&messages, &base_t2, start_t2);

        messages.push(assistant_tool_use_named("t2b", "read_file"));
        messages.push(tool_result_for("t2b", "more"));
        let api_t2_iter2 = build_api_messages(&messages, &base_t2, start_t2);

        // Base is stable within turn 2.
        assert_messages_equal(
            &api_t2_iter1[..base_t2.len()],
            &api_t2_iter2[..base_t2.len()],
            "turn 2 base stable",
        );

        // And the tool-loop prefix from iter1 is preserved in iter2.
        let shared = api_t2_iter1.len();
        assert_messages_equal(
            &api_t2_iter1[..shared],
            &api_t2_iter2[..shared],
            "turn 2 iter1→iter2 prefix",
        );
    }
}
