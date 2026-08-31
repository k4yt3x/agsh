//! `meka acp` subcommand. Speaks the Agent Client Protocol (ACP) on stdio so editor / web /
//! messenger clients can drive a meka turn end to end.
//!
//! The advertised capability surface, the delegation rules and the wire shapes are documented in
//! `docs/book/src/usage/acp.md`.
//!
//! **`execute_command` is never delegated to the client's `terminal/*`**, whatever it advertises
//! and whatever the permission mode. meka owns the process so its Landlock / bwrap / sandbox-exec /
//! Low-Integrity jail, env scrub, cwd resolution and process-group kill keep applying; the client's
//! terminal offers no equivalent, so routing through it runs `ask` unsandboxed, a mode meka treats
//! as sandboxed.
//!
//! Any number of sessions coexist in one process, each with its own cwd, permission cell,
//! conversation, cancellation token, `Agent` and `AcpFrontend`, over process-wide dependencies held
//! by `Arc`. Nothing serialises turns, so two `session/prompt` calls run in parallel. A sub-agent
//! reaches the parent's client through [`crate::frontend::PermissionForwardingFrontend`], so its
//! permission prompts and fs delegates surface in the parent session's editor UI.

mod elicitation;
mod schedule;

use std::{
    path::PathBuf,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use agent_client_protocol::{
    Agent as AcpAgentRole, ByteStreams, Client, ConnectionTo,
    schema::v1::{
        AgentCapabilities, AvailableCommand, AvailableCommandInput, AvailableCommandsUpdate,
        CancelNotification, ClientCapabilities, CloseSessionRequest, CloseSessionResponse,
        ConfigOptionUpdate, ContentBlock, ContentChunk, CurrentModeUpdate, Diff, EmbeddedResource,
        EmbeddedResourceResource, ForkSessionRequest, ForkSessionResponse, ImageContent,
        Implementation, InitializeRequest, InitializeResponse, ListSessionsRequest,
        ListSessionsResponse, LoadSessionRequest, LoadSessionResponse, NewSessionRequest,
        NewSessionResponse, PermissionOption, PermissionOptionKind, Plan, PlanEntry,
        PlanEntryPriority, PlanEntryStatus, PromptCapabilities, PromptRequest, PromptResponse,
        ReadTextFileRequest, RequestPermissionOutcome, RequestPermissionRequest,
        ResumeSessionRequest, ResumeSessionResponse, SessionAdditionalDirectoriesCapabilities,
        SessionCapabilities, SessionCloseCapabilities, SessionConfigOption,
        SessionConfigOptionCategory, SessionConfigSelectOption, SessionConfigValueId,
        SessionForkCapabilities, SessionId, SessionInfo, SessionInfoUpdate,
        SessionListCapabilities, SessionMode, SessionModeId, SessionModeState, SessionNotification,
        SessionResumeCapabilities, SessionUpdate, SetSessionConfigOptionRequest,
        SetSessionConfigOptionResponse, SetSessionModeRequest, SetSessionModeResponse, StopReason,
        ToolCall, ToolCallContent, ToolCallLocation, ToolCallStatus, ToolCallUpdate,
        ToolCallUpdateFields, ToolKind, UnstructuredCommandInput, Usage, UsageUpdate,
        WriteTextFileRequest,
    },
};
use async_trait::async_trait;
use futures::io::AsyncRead;
use tokio::sync::Mutex;
use tokio_util::{
    compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt},
    sync::CancellationToken,
};

use crate::{
    agent::Agent,
    config::ResolvedConfig,
    conversation::Conversation,
    error::MekaError,
    frontend::{
        Frontend, FrontendError, FrontendEvent, PermissionOutcome, PermissionRequest,
        ToolOutputMetadata,
    },
    mcp,
    permission::{Permission, SharedPermission},
    provider::{ContentBlock as MekaContentBlock, Role, ToolResultContent},
    session::SessionManager,
    skills::SkillCache,
    tools::todo::{TodoItem, TodoStatus},
    workspace::{SharedCwd, SharedRoots, resolve_against_cwd},
};

/// Build a JSON-RPC `InvalidParams` error (`-32602`) with a free-form human-readable message in the
/// `data` field. Mirrors [`agent_client_protocol::util::internal_error`] but for the
/// input-validation cases (unknown sessionId, malformed UUID, unsupported mode, non-text content).
/// Clients can rely on the JSON-RPC code to distinguish "bad input" from "server failure".
fn invalid_params_error(message: impl ToString) -> agent_client_protocol::Error {
    agent_client_protocol::Error::invalid_params().data(message.to_string())
}

/// Classify an `fs/*` failure into the one distinction the file tools route on.
///
/// `ResourceNotFound` (`-32002`) is the protocol's way for a client to say it will not serve a
/// path. Which paths those are is the client's business and differs between editors -- Zed answers
/// it for anything outside the project it has open, another client may serve any absolute path --
/// which is exactly why meka asks per path instead of modelling any editor's rule. Every other
/// code (transport, timeout, an internal error inside the client) leaves open the possibility that
/// the client owns the file and holds unsaved changes for it, so it must not route around it.
fn classify_fs_error(method: &str, error: &agent_client_protocol::Error) -> FrontendError {
    let message = format!("{} failed: {}", method, error);
    if error.code == agent_client_protocol::ErrorCode::ResourceNotFound {
        FrontendError::unservable_path(message)
    } else {
        FrontendError::new(message)
    }
}

/// Late-bound view of everything the connected client told us on `initialize`: its advertised
/// capabilities and its self-identifying `Implementation` (name + version). Default is the
/// all-`false` `ClientCapabilities` and a `None` identity, so an `AcpFrontend` constructed before
/// `initialize` arrives correctly reports "delegation unavailable" and "client unknown" until the
/// handler fills it in.
#[derive(Clone, Default)]
pub struct SharedClientState {
    inner: Arc<std::sync::RwLock<ClientStateInner>>,
}

#[derive(Clone, Default)]
struct ClientStateInner {
    capabilities: ClientCapabilities,
    /// Logged once on `initialize`. Read only in tests today; the `#[allow(dead_code)]` stays
    /// until a production reader (e.g. surfacing the client name in response `_meta`) lands.
    #[allow(dead_code)]
    info: Option<Implementation>,
}

impl SharedClientState {
    /// Record both halves of the client-side initialize payload in one write. Called exactly once
    /// per process today (the `initialize` handler), but tolerant of re-initialisation if a future
    /// client ever resends.
    fn record_initialize(&self, capabilities: ClientCapabilities, info: Option<Implementation>) {
        let mut guard = self
            .inner
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *guard = ClientStateInner { capabilities, info };
    }

    fn capabilities(&self) -> ClientCapabilities {
        self.inner
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .capabilities
            .clone()
    }

    /// Whether the client renders agent-owned terminals from meka's `_meta` frames.
    ///
    /// Deliberately not the typed `terminal` capability: that one says the client implements
    /// `terminal/*` so an agent can run commands *in the client*, which meka never does. A client
    /// can offer that and still have no idea what [`META_TERMINAL_INFO`] means.
    fn renders_agent_terminals(&self) -> bool {
        self.capabilities()
            .meta
            .as_ref()
            .and_then(|meta| meta.get(CAP_TERMINAL_OUTPUT))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    }

    #[cfg(test)]
    fn client_info(&self) -> Option<Implementation> {
        self.inner
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .info
            .clone()
    }
}

/// Render an `Implementation` as a `"name version"` pair for the `initialize` log line. `None`
/// renders as `"<unknown> <unknown>"` so the log shape is stable across clients that omit
/// `client_info` entirely.
fn describe_client(info: Option<&Implementation>) -> String {
    match info {
        Some(implementation) => format!("{} {}", implementation.name, implementation.version),
        None => "<unknown> <unknown>".to_string(),
    }
}

/// ACP-side [`Frontend`] impl. Converts the agent loop's streamed events into ACP `session/update`
/// notifications and runs the `session/request_permission` round-trip for tool approvals.
/// Constructed per-session: every field is fully populated at build time, so there's no "not yet
/// bound" `Option` state to handle.
pub struct AcpFrontend {
    connection: ConnectionTo<Client>,
    session_id: SessionId,
    cwd: SharedCwd,
    /// Sticky `allow_always` set; symmetric `never_allowed` below for `reject_always`. Both
    /// short-circuit `request_permission` so the user isn't re-prompted for the same tool in this
    /// session. Per-session (one `AcpFrontend` per session); not persisted.
    always_allowed: std::sync::Mutex<std::collections::HashSet<String>>,
    never_allowed: std::sync::Mutex<std::collections::HashSet<String>>,
    client_state: SharedClientState,
    /// Stdio-level "transport is dead" latch, shared across every per-session `AcpFrontend` in the
    /// process. When `send_notification` fails on any session, we set the latch so every other
    /// session's agent loop short-circuits on its next iteration instead of burning provider
    /// tokens until its own emit also fails.
    ///
    /// This is correct *for stdio*: one closed pipe affects every session in the process, so the
    /// global signal carries no false positives. When a per-session transport (e.g. WebSocket-ACP
    /// or a TCP-multiplexed successor) lands, this field needs a per-session sibling (read both
    /// in `client_disconnected()` and OR them) so a single session's drop doesn't take the
    /// process down with it. Grep for `transport_dead` to find the migration points.
    transport_dead: Arc<std::sync::atomic::AtomicBool>,
    /// Live context-occupancy counter shared with this session's agent (it writes the value after
    /// each round via [`Agent::set_context_tokens`]); read on every `TokenUsage` event to emit an
    /// ACP `usage_update` so editors show "tokens used / context window".
    context_tokens: Arc<std::sync::atomic::AtomicU64>,
    /// The resolved window for the `usage_update` `size` field, and the same cell the agent
    /// publishes into on every provider switch. `0` until the agent is built, which suppresses the
    /// update.
    ///
    /// Shared rather than pushed. ACP was the one host that kept its own copy and had it re-stored
    /// by hand from `session/set_config_option`, which meant a mid-turn switch reported occupancy
    /// measured against the profile the turn was still running on, divided by the window of the
    /// one it had not moved to yet.
    context_window: Arc<std::sync::atomic::AtomicU64>,
    /// Accumulated live output per in-flight tool call, keyed by `tool_use_id`. ACP replaces a
    /// tool call's whole `content` array on each update rather than appending to it, so the
    /// running total has to be kept somewhere; the emitter sends deltas, and this is where
    /// they are added up. Entries are dropped when the call completes.
    live_output: std::sync::Mutex<std::collections::HashMap<String, LiveOutput>>,
    /// The same cell [`SessionEntry::cancellation`] holds, so a client round-trip started by this
    /// frontend can be abandoned when `session/cancel` fires. Shared rather than copied: the
    /// prompt handler rewrites the token at every turn start, and a frontend holding a stale
    /// clone would race against a token nobody signals.
    cancellation: Arc<std::sync::RwLock<CancellationToken>>,
}

/// How a running command's output is shown to this client.
///
/// The two modes are not cosmetic variants of each other. In [`Self::Terminal`] the client owns a
/// scrollback buffer that meka appends to, so the whole output is available and rendered as a real
/// terminal (ANSI colours, selection, its own scrolling). In [`Self::Text`] the only lever is
/// replacing the tool call's `content`, so meka has to keep the running text itself and can only
/// afford to re-send a window of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LiveOutputMode {
    /// Client advertised `terminal`, so it understands the agent-owned terminal `_meta` channel.
    Terminal,
    /// Fallback: a `console` code block, replaced on each update.
    Text,
}

/// Per-tool-call state for relaying a running command's output.
struct LiveOutput {
    mode: LiveOutputMode,
    /// Text still to show. In [`LiveOutputMode::Text`] this is the whole (tail-capped) output,
    /// re-sent every tick. In [`LiveOutputMode::Terminal`] it is only the bytes not yet appended,
    /// and it is drained on each send, because the client keeps the scrollback.
    text: String,
    /// `None` until the first update goes out, so the first chunk is never delayed.
    last_sent: Option<std::time::Instant>,
}

/// Shortest gap between two `tool_call_update`s for the same tool call. A chatty build writes
/// thousands of small chunks a second, and one notification per read syscall would spend more time
/// on the wire than on the build. Coalescing is why both modes need a buffer at all.
const LIVE_OUTPUT_INTERVAL: std::time::Duration = std::time::Duration::from_millis(150);

/// How much of the tail to keep visible while a command runs, in [`LiveOutputMode::Text`] only.
/// That mode re-sends its whole buffer on every tick, so an uncapped buffer would make a chatty
/// command cost quadratic in its own output. The terminal mode appends and needs no cap.
const LIVE_OUTPUT_TAIL_BYTES: usize = 8 * 1024;

impl LiveOutput {
    fn new(mode: LiveOutputMode) -> Self {
        Self {
            mode,
            text: String::new(),
            last_sent: None,
        }
    }

    /// Append a delta and return what to send now, or `None` while throttled.
    fn push(&mut self, chunk: &str, now: std::time::Instant) -> Option<String> {
        self.text.push_str(chunk);
        if self.mode == LiveOutputMode::Text && self.text.len() > LIVE_OUTPUT_TAIL_BYTES {
            let mut cut = self.text.len() - LIVE_OUTPUT_TAIL_BYTES;
            // A byte count can land inside a multi-byte character, and both slicing and draining
            // there panic. Advance to a boundary before either.
            while cut < self.text.len() && !self.text.is_char_boundary(cut) {
                cut += 1;
            }
            // Prefer opening the view at a line start rather than mid-line.
            if let Some(offset) = self.text[cut..].find('\n') {
                cut += offset + 1;
            }
            self.text.drain(..cut);
        }
        if let Some(last) = self.last_sent
            && now.duration_since(last) < LIVE_OUTPUT_INTERVAL
        {
            return None;
        }
        self.last_sent = Some(now);
        match self.mode {
            // Appending: hand over what has accumulated and start empty again, so the same bytes
            // are never sent twice.
            LiveOutputMode::Terminal => {
                if self.text.is_empty() {
                    None
                } else {
                    Some(std::mem::take(&mut self.text))
                }
            }
            LiveOutputMode::Text => Some(self.text.clone()),
        }
    }

    /// Whatever is still buffered, for the final flush before a call completes. The throttle can
    /// swallow the last chunk of a command that exits right after printing, and in terminal mode
    /// those bytes exist nowhere else.
    fn take_pending(&mut self) -> Option<String> {
        match self.mode {
            LiveOutputMode::Terminal if !self.text.is_empty() => {
                Some(std::mem::take(&mut self.text))
            }
            _ => None,
        }
    }
}

/// `clientCapabilities._meta` key a client sets to say it renders agent-owned terminals from the
/// `_meta` frames below. Distinct from the typed `terminal` capability, which means "I implement
/// `terminal/*` requests" -- a client can do that without understanding these frames at all. Zed
/// advertises both; gating on the wrong one would send terminal content blocks to a client that
/// resolves them to nothing and so displays no output, which is the failure this whole path exists
/// to fix.
const CAP_TERMINAL_OUTPUT: &str = "terminal_output";

/// `_meta` key naming an agent-owned terminal so the client registers it. Without this the client
/// has nothing to attach output to and buffers it against an id it was never told about (Zed parks
/// it in `pending_terminal_output`, drained only on a matching create), leaving an empty terminal.
///
/// Must ride on the `tool_call` that opens the call, not a later `tool_call_update`: the client
/// reads it only off the former.
const META_TERMINAL_INFO: &str = "terminal_info";
/// `_meta` key carrying a chunk of output to append to an agent-owned terminal.
const META_TERMINAL_OUTPUT: &str = "terminal_output";
/// `_meta` key marking an agent-owned terminal as finished, with its exit status.
const META_TERMINAL_EXIT: &str = "terminal_exit";

/// Build the `_meta` map for one agent-owned-terminal frame.
///
/// This is an extension, not ACP proper: it originates in codex-acp, claude-agent-acp emits the
/// same shape, and Zed consumes it. ACP v2 standardises the idea as `terminal_update` /
/// `terminal_output_chunk`, which is what this should become once a client speaks v2. `_meta` is
/// specified as ignorable (every `_meta` field deserialises with `DefaultOnError`), so a client
/// that doesn't know these keys drops them rather than failing.
fn terminal_meta(key: &str, payload: serde_json::Value) -> agent_client_protocol::schema::v1::Meta {
    let mut meta = agent_client_protocol::schema::v1::Meta::new();
    meta.insert(key.to_string(), payload);
    meta
}

impl AcpFrontend {
    #[allow(clippy::too_many_arguments)]
    fn new(
        connection: ConnectionTo<Client>,
        session_id: SessionId,
        cwd: SharedCwd,
        client_state: SharedClientState,
        transport_dead: Arc<std::sync::atomic::AtomicBool>,
        context_tokens: Arc<std::sync::atomic::AtomicU64>,
        context_window: Arc<std::sync::atomic::AtomicU64>,
        cancellation: Arc<std::sync::RwLock<CancellationToken>>,
    ) -> Self {
        Self {
            connection,
            session_id,
            cwd,
            always_allowed: std::sync::Mutex::new(std::collections::HashSet::new()),
            never_allowed: std::sync::Mutex::new(std::collections::HashSet::new()),
            client_state,
            transport_dead,
            context_tokens,
            context_window,
            live_output: std::sync::Mutex::new(std::collections::HashMap::new()),
            cancellation,
        }
    }

    /// The cell itself, so the owning [`SessionEntry`] shares it rather than minting a second one.
    /// Both sides must see the token the prompt handler installs, or `session/cancel` signals one
    /// cell while the frontend waits on another.
    fn cancellation_cell(&self) -> Arc<std::sync::RwLock<CancellationToken>> {
        Arc::clone(&self.cancellation)
    }

    /// The current turn's cancellation token, cloned out of the cell the prompt handler rewrites at
    /// each turn start.
    fn current_cancellation(&self) -> CancellationToken {
        // A detached call's own token wins over the session's. See
        // `crate::frontend::scope_call_cancellation`: without this, cancelling any later turn
        // abandoned a background task's `fs/*` round-trip mid-flight.
        if let Some(call) = crate::frontend::current_call_cancellation() {
            return call;
        }
        match self.cancellation.read() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    /// Await a client round-trip, giving up if the turn is cancelled.
    ///
    /// `session/cancel` is the user pressing stop, and a client is entitled to drop an outstanding
    /// `fs/*` or elicitation request rather than answer it once it has cancelled. Without this race
    /// the future never resolves: the prompt returns no `stopReason` at all and every later prompt
    /// on that session is rejected as "already has a prompt in flight", so one stop wedges the
    /// session for the life of the process. `request_permission` has always done this; these paths
    /// were the asymmetry.
    ///
    /// The cancelled arm is a [`FrontendError`] rather than a `None` on purpose. Both callers
    /// return `Option<Result<_, FrontendError>>`, where `None` already means "this frontend has no
    /// delegate, do it locally" -- so a `?` on an `Option` here turned pressing stop into a local
    /// write, computing the edit from on-disk bytes and overwriting whatever the editor still held
    /// unsaved. Returning a `Result` makes that `?` a type error instead of a silent one.
    async fn until_cancelled<T>(
        &self,
        what: &str,
        work: impl std::future::Future<Output = T>,
    ) -> std::result::Result<T, FrontendError> {
        race_against_cancellation(what, &self.current_cancellation(), work).await
    }

    /// Mark the stdio transport as dead. Called from `emit` and the `session/load` replay loop
    /// whenever `send_notification` reports an error. Idempotent. The trait-level
    /// `client_disconnected()` read below surfaces the same flag back to the agent loop.
    fn mark_transport_dead(&self) {
        self.transport_dead
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Show a scheduled job's prompt in the transcript, before the turn that answers it runs.
    ///
    /// Goes out as a `UserMessageChunk` rather than a notice because that is what it is: the turn
    /// has a prompt, it just came from a timer instead of a keystroke. Without it the editor shows
    /// a reply with nothing above it explaining the question.
    pub(crate) fn push_scheduled_prompt(&self, wakeup: &crate::schedule::Wakeup) {
        self.push_out_of_band_prompt(&wakeup.render_prompt());
    }

    /// Show any prompt the user did not type, for the same reason: a reply with nothing above it
    /// explaining the question reads as the agent talking to itself. Used by scheduled jobs and by
    /// background-task outcome reports.
    pub(crate) fn push_out_of_band_prompt(&self, prompt: &str) {
        send_session_update(
            &self.connection,
            &self.session_id,
            super::acp::schedule::out_of_band_prompt_update(prompt),
        );
    }

    /// Push one `session/update` to the client, latching the transport as dead if it fails.
    ///
    /// `send_notification` is synchronous, which is what lets the live-output path hold its buffer
    /// lock across a send to keep chunks ordered.
    fn send_update(&self, update: SessionUpdate) {
        if let Err(error) = self
            .connection
            .send_notification(SessionNotification::new(self.session_id.clone(), update))
        {
            tracing::debug!("AcpFrontend send_notification failed: {}", error);
            self.mark_transport_dead();
        }
    }

    /// Recover from a poisoned lock rather than propagating it. A panic under this lock would
    /// otherwise disable live output *and* the tool-call completion update for the rest of the
    /// session, leaving the client on a spinner that never resolves; the buffer is display state,
    /// so continuing with whatever it holds is strictly better than going silent.
    fn live_output(
        &self,
    ) -> std::sync::MutexGuard<'_, std::collections::HashMap<String, LiveOutput>> {
        self.live_output
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Open a live-output view for a starting tool call and return how it will be rendered, or
    /// `None` for calls that produce no streamed output.
    ///
    /// Only `execute_command` qualifies: it is the one tool whose result can be minutes away, and
    /// the terminal frames below would be nonsense for anything that isn't a command. The mode is
    /// decided once, here, so the client can't be told about a terminal it never registered.
    fn begin_live_output(&self, id: &str, tool_name: &str) -> Option<LiveOutputMode> {
        if tool_name != "execute_command" {
            return None;
        }
        let mode = if self.client_state.renders_agent_terminals() {
            LiveOutputMode::Terminal
        } else {
            LiveOutputMode::Text
        };
        // The single most useful line when a user reports "I see the command but not its output":
        // it says whether the client asked for terminal rendering, which is the whole branch point.
        tracing::debug!("execute_command {} live output: {:?}", id, mode);
        self.live_output()
            .insert(id.to_string(), LiveOutput::new(mode));
        Some(mode)
    }

    fn is_always_allowed(&self, tool_name: &str) -> bool {
        self.always_allowed
            .lock()
            .map(|guard| guard.contains(tool_name))
            .unwrap_or(false)
    }

    fn is_never_allowed(&self, tool_name: &str) -> bool {
        self.never_allowed
            .lock()
            .map(|guard| guard.contains(tool_name))
            .unwrap_or(false)
    }

    fn remember_allow(&self, tool_name: &str) {
        if let Ok(mut guard) = self.always_allowed.lock() {
            guard.insert(tool_name.to_string());
        }
    }

    fn remember_deny(&self, tool_name: &str) {
        if let Ok(mut guard) = self.never_allowed.lock() {
            guard.insert(tool_name.to_string());
        }
    }
}

#[async_trait]
impl Frontend for AcpFrontend {
    async fn emit(&self, event: FrontendEvent) {
        let update = match event {
            FrontendEvent::AssistantTextDelta(text) => {
                SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
                    agent_client_protocol::schema::v1::TextContent::new(text),
                )))
            }
            // Nothing to show, and no transient UI to close: ACP thought chunks accumulate, so a
            // counter meant to be drawn over and replaced would leave a trail of stale figures in
            // the thread. Matched explicitly rather than left to the catch-all below, which is for
            // REPL signage an editor has its own UI for -- these two are the opposite case, signals
            // this frontend structurally cannot represent.
            FrontendEvent::ThinkingProgress { .. } | FrontendEvent::ThinkingEnded => return,
            FrontendEvent::ThinkingBlock { content, .. } => {
                SessionUpdate::AgentThoughtChunk(ContentChunk::new(ContentBlock::Text(
                    agent_client_protocol::schema::v1::TextContent::new(content),
                )))
            }
            // ACP has a `pending` status this could drive, but a call is announced to the client
            // exactly once, and that announcement carries the title, locations and arguments an
            // editor renders -- none of which exist yet here. Sending a name-only call first would
            // mean two `session/update: tool_call` notifications for one id.
            FrontendEvent::ToolCallComposing { .. } => return,
            FrontendEvent::ToolCallStarted {
                id,
                name,
                input,
                display_summary,
            } => {
                // No separate `pending` state in the agent loop, so the in-progress emit is the
                // first one the client sees. The title carries the resolved primary argument (the
                // command, file path, URL, ...) so editors show what's actually running instead of
                // a bare tool name; `raw_input` still carries the full argument object.
                let locations = tool_locations(&name, &input, &self.cwd);
                let title = tool_call_title(&name, display_summary.as_deref());
                let mut call = ToolCall::new(id.clone(), title)
                    .kind(tool_kind_for(&name))
                    .status(ToolCallStatus::InProgress)
                    .locations(locations)
                    .raw_input(input);
                // Claim the rendering mode for this call up front, because the terminal has to be
                // announced before any output references it. The tool call's own id doubles as the
                // terminal id: it is unique per call and is what the client already correlates on.
                if self.begin_live_output(&id, &name) == Some(LiveOutputMode::Terminal) {
                    // `cwd` is optional in the frame but worth sending: the client labels the
                    // terminal with it, and meka's per-session cwd (which `/cd` moves) is not
                    // something the client could otherwise know.
                    let cwd = crate::workspace::cwd_snapshot(&self.cwd);
                    call = call
                        .content(vec![ToolCallContent::Terminal(
                            agent_client_protocol::schema::v1::Terminal::new(id.clone()),
                        )])
                        .meta(terminal_meta(
                            META_TERMINAL_INFO,
                            serde_json::json!({ "terminal_id": id, "cwd": cwd }),
                        ));
                }
                SessionUpdate::ToolCall(call)
            }
            FrontendEvent::ToolCallOutputDelta { id, chunk } => {
                // stdout and stderr drain on separate tasks, so two deltas for the same call can
                // be in flight on two threads at once. Terminal mode appends whatever it is handed,
                // so draining the buffer and sending it have to be one atomic step: release the
                // lock in between and the two tasks can swap order, interleaving the terminal's
                // contents. Safe to hold across the send because `send_notification` is
                // synchronous -- nothing is awaited under the lock.
                let mut buffers = self.live_output();
                // Absent means the call never opened a live view (not `execute_command`, or it
                // already completed), so there is nothing to attach this to.
                let Some(entry) = buffers.get_mut(&id) else {
                    return;
                };
                let mode = entry.mode;
                // `None` means this tick is throttled away; the buffer keeps the bytes and the next
                // tick that clears the interval sends them.
                let Some(text) = entry.push(&chunk, std::time::Instant::now()) else {
                    return;
                };
                self.send_update(SessionUpdate::ToolCallUpdate(live_output_update(
                    &id, mode, &text,
                )));
                return;
            }
            FrontendEvent::ToolCallCompleted {
                id,
                name,
                is_error,
                content,
                metadata,
            } => {
                let live = {
                    // The completion update carries the authoritative output, so the live view has
                    // done its job either way; take the mode and any bytes the throttle swallowed.
                    self.live_output()
                        .remove(&id)
                        .map(|mut entry| (entry.mode, entry.take_pending()))
                };
                // A command that exits immediately after printing can have its last chunk still
                // inside the throttle window. In terminal mode those bytes live nowhere else -- the
                // client's scrollback is the only copy -- so flush them before marking the call
                // done.
                if let Some((LiveOutputMode::Terminal, Some(pending))) = &live {
                    self.send_update(SessionUpdate::ToolCallUpdate(live_output_update(
                        &id,
                        LiveOutputMode::Terminal,
                        pending,
                    )));
                }
                let status = if is_error {
                    ToolCallStatus::Failed
                } else {
                    ToolCallStatus::Completed
                };
                let mut fields = ToolCallUpdateFields::new().status(status);
                let mut update_meta = None;
                if let Some((LiveOutputMode::Terminal, _)) = live {
                    // Keep the terminal as the call's content: it already holds the full
                    // scrollback, and replacing it with a text block here would
                    // swap a live, scrollable, colour-rendered view for a
                    // flattened copy at the moment the command finishes.
                    fields = fields.content(vec![ToolCallContent::Terminal(
                        agent_client_protocol::schema::v1::Terminal::new(id.clone()),
                    )]);
                    update_meta = Some(terminal_meta(
                        META_TERMINAL_EXIT,
                        serde_json::json!({
                            "terminal_id": id,
                            "exit_code": command_exit_code(&metadata, is_error),
                            "signal": command_signal(&metadata),
                        }),
                    ));
                } else {
                    fields = fields.content(build_completion_content(&name, &content, metadata));
                }
                // Surface the structured tool output too, so clients (e.g. Zed's tool-call detail
                // view) can introspect the result beyond the rendered `content` blocks.
                if let Ok(raw) = serde_json::to_value(&content) {
                    fields = fields.raw_output(raw);
                }
                let mut update = ToolCallUpdate::new(id, fields);
                if let Some(meta) = update_meta {
                    update = update.meta(meta);
                }
                SessionUpdate::ToolCallUpdate(update)
            }
            FrontendEvent::SubAgentActivity {
                tool_call_id,
                summary,
            } => {
                // ACP has no sub-agent primitive -- no nested sessions, no nested tool calls -- so
                // a sub-agent is one tool call and its progress is that call's content. Updating
                // content while the call is still `in_progress` is what turns an opaque spinner
                // into a live view of what the sub-agent is doing. The client replaces the content
                // array on each update, which is why `summary` is the whole block.
                let fields = ToolCallUpdateFields::new().content(vec![text_content_block(summary)]);
                SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(tool_call_id, fields))
            }
            FrontendEvent::Notice(notice) => {
                // No dedicated ACP primitive for advisories; surface inline as an assistant-message
                // chunk with an `[meka]` prefix so editor transcripts record the side-effect and
                // clients can filter or style by that prefix. `notice.level` is unused on the wire
                // today; when ACP grows a typed notice variant, branch on it here.
                //
                // Deliberately not [`crate::conversation::HARNESS_NOTE`], despite the resemblance.
                // That marker is longer because a *model* has to read it and should not have to
                // recall what meka is; this one is read by a person looking at their own editor,
                // where the product name needs no gloss, and by clients matching on the prefix,
                // which a rename would break.
                let text = format!("[meka] {}", notice.text);
                SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
                    agent_client_protocol::schema::v1::TextContent::new(text),
                )))
            }
            FrontendEvent::McpProgress(update) => {
                // ACP has no protocol primitive for tool-progress streams. The REPL renders these
                // inline as a carriage-return-overwrite status line; in the ACP world the editor
                // already has its own visibility into MCP server activity (or can subscribe to the
                // stderr log stream of the spawned agent). Log at info so `-v` users can still see
                // them; don't pollute the assistant-message transcript with per-tick status text.
                tracing::info!(
                    "MCP '{}' {} progress: {}{}{}",
                    update.server_name,
                    update.tool_name,
                    update.progress,
                    update.total.map(|t| format!("/{}", t)).unwrap_or_default(),
                    update
                        .message
                        .as_deref()
                        .map(|m| format!(", {}", m))
                        .unwrap_or_default()
                );
                return;
            }
            FrontendEvent::TodoListUpdated { items, .. } => {
                // The `todo` tool's list maps onto ACP's plan panel. The REPL-only `title` has no
                // `Plan` analogue and is dropped. Note: the agent loop suppresses emission of an
                // emptied list (`agent.rs`), so a cleared plan is not pushed - parity with the
                // REPL.
                SessionUpdate::Plan(Plan::new(todo_items_to_plan(&items)))
            }
            FrontendEvent::TokenUsage(_) => {
                // Mirror the REPL's context gauge as an ACP `usage_update` so editors (e.g. Zed)
                // show "tokens used / context window". `used` is read from the shared atomic the
                // agent updates each round (current occupancy: all input tiers + output) rather
                // than the event's per-turn total, which over-counts multi-round tool turns;
                // `size` is the resolved window. Suppress until both are known.
                let used = self
                    .context_tokens
                    .load(std::sync::atomic::Ordering::Relaxed);
                let size = self
                    .context_window
                    .load(std::sync::atomic::Ordering::Relaxed);
                if used == 0 || size == 0 {
                    return;
                }
                SessionUpdate::UsageUpdate(UsageUpdate::new(used, size))
            }
            FrontendEvent::TurnStarted => {
                // Tool calls begin and end inside a turn, so anything still open here belongs to a
                // previous one that never delivered its completion (a cancelled turn, or a stream
                // retried after announcing a tool call). Those entries would otherwise accumulate
                // for the life of the session. `TurnFinished` is not a substitute: the agent loop
                // only emits it when the turn succeeded, which is exactly when there is nothing to
                // clean up.
                self.live_output().clear();
                return;
            }
            // REPL-specific signage (lifecycle, diffs).
            _ => return,
        };

        self.send_update(update);
    }

    fn client_disconnected(&self) -> bool {
        self.transport_dead
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    async fn request_permission(&self, request: PermissionRequest) -> PermissionOutcome {
        // Honor sticky decisions from earlier `*_always` selections.
        if self.is_always_allowed(&request.tool_name) {
            return PermissionOutcome::Allow;
        }
        if self.is_never_allowed(&request.tool_name) {
            return PermissionOutcome::Deny;
        }

        let connection = self.connection.clone();
        let session_id = self.session_id.clone();

        // The sticky options name the *tool*, because that is their scope: the decision is keyed on
        // the tool name alone and applies to every later call to it, whatever its arguments. The
        // prompt's title beside them is `<tool> <primary_param>` -- for `execute_command` that is
        // the specific command line -- so a bare "Always allow" reads as approving the command the
        // user just read, when it actually approves every shell command for the rest of the
        // session. Spelling the tool out is what makes the affordance and the semantics
        // agree.
        let options = vec![
            PermissionOption::new(OPTION_ALLOW_ONCE, "Allow", PermissionOptionKind::AllowOnce),
            PermissionOption::new(
                OPTION_ALLOW_ALWAYS,
                sticky_option_label("allow", &request.tool_name),
                PermissionOptionKind::AllowAlways,
            ),
            PermissionOption::new(OPTION_REJECT_ONCE, "Deny", PermissionOptionKind::RejectOnce),
            PermissionOption::new(
                OPTION_REJECT_ALWAYS,
                sticky_option_label("deny", &request.tool_name),
                PermissionOptionKind::RejectAlways,
            ),
        ];

        // Synthetic id: the permission round-trip is its own space, not correlated with the
        // streaming tool_call lifecycle.
        let tool_call_id = format!("perm-{}", uuid::Uuid::new_v4());
        let title = match &request.primary_param {
            Some(param) if !param.is_empty() => format!("{} {}", request.tool_name, param),
            _ => request.tool_name.clone(),
        };
        let fields = ToolCallUpdateFields::new()
            .kind(tool_kind_for(&request.tool_name))
            .title(title)
            .status(ToolCallStatus::Pending);
        let tool_call = ToolCallUpdate::new(tool_call_id, fields);

        let req = RequestPermissionRequest::new(session_id, tool_call, options);
        // Race the round-trip against the per-turn cancellation token. If `session/cancel` fires
        // while we're waiting for the client to answer the permission prompt, we resolve as
        // `Cancelled` instead of holding the runtime mutex forever, which would block
        // `session/close` and `session/set_mode` too.
        let response = tokio::select! {
            biased;
            _ = request.cancellation.cancelled() => {
                return PermissionOutcome::Cancelled;
            }
            // The backstop the cancellation race alone does not provide. A client that is
            // *connected* but never answers -- an editor whose UI thread is wedged, a headless
            // harness that speaks ACP but implements no prompt -- fires no cancellation, so the
            // race above waits on it forever and the turn holds the runtime mutex with it. Deny on
            // expiry rather than allow: an unanswered prompt is not consent.
            _ = tokio::time::sleep(PERMISSION_PROMPT_TIMEOUT) => {
                tracing::warn!(
                    "the client did not answer the permission prompt for '{}' within {:?}; \
                     denying it",
                    request.tool_name,
                    PERMISSION_PROMPT_TIMEOUT,
                );
                return PermissionOutcome::Deny;
            }
            result = connection.send_request(req).block_task() => match result {
                Ok(resp) => resp,
                Err(error) => {
                    tracing::debug!("request_permission send_request failed: {}", error);
                    // Spec-conformant clients always reply with a `Selected` or `Cancelled`
                    // outcome, so an `Err` here is almost certainly transport-level. Mark the
                    // connection dropped so the agent loop short-circuits on the next pre-iteration
                    // check instead of running a tool, emitting a denied result, and only then
                    // discovering the client is gone via the next emit. The FS delegates
                    // intentionally don't do this: those paths legitimately receive JSON-RPC error
                    // responses (a path the client won't serve), which would produce false-positive
                    // disconnects.
                    self.mark_transport_dead();
                    return PermissionOutcome::Deny;
                }
            },
        };

        translate_permission_outcome(
            response.outcome,
            &request.tool_name,
            |sticky| match sticky {
                StickyDecision::AllowAlways => self.remember_allow(&request.tool_name),
                StickyDecision::RejectAlways => self.remember_deny(&request.tool_name),
            },
        )
    }

    async fn delegate_fs_read(
        &self,
        path: &std::path::Path,
        line: Option<u32>,
        limit: Option<u32>,
    ) -> Option<Result<String, FrontendError>> {
        let caps = self.client_state.capabilities();
        if !caps.fs.read_text_file {
            return None;
        }
        let connection = self.connection.clone();
        let session_id = self.session_id.clone();
        let mut request = ReadTextFileRequest::new(session_id, path.to_path_buf());
        if let Some(line) = line {
            request = request.line(line);
        }
        if let Some(limit) = limit {
            request = request.limit(limit);
        }
        let outcome = match self
            .until_cancelled(
                "fs/read_text_file",
                connection.send_request(request).block_task(),
            )
            .await
        {
            Ok(outcome) => outcome,
            Err(cancelled) => return Some(Err(cancelled)),
        };
        Some(match outcome {
            Ok(response) => Ok(response.content),
            Err(error) => Err(classify_fs_error("fs/read_text_file", &error)),
        })
    }

    async fn delegate_fs_write(
        &self,
        path: &std::path::Path,
        content: &str,
    ) -> Option<Result<(), FrontendError>> {
        let caps = self.client_state.capabilities();
        if !caps.fs.write_text_file {
            return None;
        }
        let connection = self.connection.clone();
        let session_id = self.session_id.clone();
        let request =
            WriteTextFileRequest::new(session_id, path.to_path_buf(), content.to_string());
        let outcome = match self
            .until_cancelled(
                "fs/write_text_file",
                connection.send_request(request).block_task(),
            )
            .await
        {
            Ok(outcome) => outcome,
            Err(cancelled) => return Some(Err(cancelled)),
        };
        Some(match outcome {
            Ok(_) => Ok(()),
            Err(error) => Err(classify_fs_error("fs/write_text_file", &error)),
        })
    }

    async fn handle_elicitation(
        &self,
        prompt: crate::mcp::elicitation::ElicitationPrompt,
    ) -> crate::mcp::elicitation::ElicitationResponse {
        use crate::mcp::elicitation::{ElicitationKind, ElicitationResponse};

        let kind = match &prompt.kind {
            ElicitationKind::Form { .. } => "form",
            ElicitationKind::Url { .. } => "url",
        };
        // Form and URL support are advertised independently, so check the mode actually being
        // asked for. An unadvertised mode must not be sent: the client would reject it, and the
        // MCP call is blocked on the answer meanwhile.
        let supported = self
            .client_state
            .capabilities()
            .elicitation
            .is_some_and(|caps| match &prompt.kind {
                ElicitationKind::Form { .. } => caps.form.is_some(),
                ElicitationKind::Url { .. } => caps.url.is_some(),
            });
        if !supported {
            tracing::warn!(
                "MCP elicitation from '{}' ({}) declined: the client does not support this \
                 elicitation mode; prompt was '{}'",
                prompt.server_name,
                kind,
                prompt.message,
            );
            return ElicitationResponse::Decline;
        }

        let Some(request) = elicitation::to_acp_request(&prompt, &self.session_id) else {
            tracing::warn!(
                "MCP elicitation from '{}' declined: its schema uses a field type ACP cannot \
                 express, so the client could not render the form; prompt was '{}'",
                prompt.server_name,
                prompt.message,
            );
            return ElicitationResponse::Decline;
        };

        // Raced against the turn's cancellation, like every other client round-trip on this
        // frontend. A bare await here was the last one left: an MCP `call_tool` is blocked on this
        // answer, so a client that drops the request rather than answering it -- which it is
        // entitled to do once the user has pressed stop -- left the tool call, the turn, and every
        // later prompt on that session waiting for the life of the process.
        //
        // Declining on cancellation rather than propagating it, because the return type has no
        // third state and the MCP server needs an answer either way. The turn is stopping; what the
        // server does with the refusal no longer changes what the user sees.
        let outcome = match self
            .until_cancelled(
                "elicitation/create",
                self.connection.clone().send_request(request).block_task(),
            )
            .await
        {
            Ok(outcome) => outcome,
            Err(_cancelled) => {
                tracing::debug!(
                    "MCP elicitation from '{}' ({}) declined: the turn was cancelled",
                    prompt.server_name,
                    kind,
                );
                return ElicitationResponse::Decline;
            }
        };
        match outcome {
            Ok(response) => elicitation::from_acp_action(response.action),
            Err(error) => {
                tracing::warn!(
                    "MCP elicitation from '{}' ({}) declined: elicitation/create failed: {}",
                    prompt.server_name,
                    kind,
                    error,
                );
                ElicitationResponse::Decline
            }
        }
    }
}

/// Stable string IDs for the four permission options. The agent and the client must agree on these;
/// picking them as `const`s keeps the match arm in [`translate_permission_outcome`] honest.
const OPTION_ALLOW_ONCE: &str = "allow_once";
const OPTION_ALLOW_ALWAYS: &str = "allow_always";
const OPTION_REJECT_ONCE: &str = "reject_once";
const OPTION_REJECT_ALWAYS: &str = "reject_always";

/// Label for a sticky (`*Always`) permission option, naming the tool the decision actually covers.
///
/// A function rather than two `format!`s at the call site so the wording is assertable. The sticky
/// options are keyed on the tool name alone, and the prompt beside them shows one specific
/// invocation, so a bare "Always allow" reads as approving the command on screen when it approves
/// every call to that tool for the session. The tool name is the part that must not go missing.
fn sticky_option_label(verb: &str, tool_name: &str) -> String {
    format!("Always {} any {}", verb, tool_name)
}

/// The cancellation race itself, taking the token rather than reading it off an `AcpFrontend`, so
/// it can be exercised without standing up a connection to a client.
///
/// `biased` matters: a turn cancelled while the request is already outstanding must lose the race
/// deterministically, not half the time. The client owes no answer to a request the user withdrew,
/// and before this every `fs/*` round trip and every elicitation could outlive the stop button
/// indefinitely.
async fn race_against_cancellation<T>(
    what: &str,
    cancellation: &CancellationToken,
    work: impl std::future::Future<Output = T>,
) -> std::result::Result<T, FrontendError> {
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => {
            tracing::debug!("{} abandoned: the turn was cancelled", what);
            Err(FrontendError::cancelled(what))
        }
        result = work => Ok(result),
    }
}

/// Indicates which sticky bucket the user just opted into, so the caller can update its set.
/// Internal to the permission flow.
enum StickyDecision {
    AllowAlways,
    RejectAlways,
}

/// Map an ACP outcome to meka's [`PermissionOutcome`] and fire `record_sticky` when the user picked
/// one of the `*_always` options. Pure function so it's easy to unit-test.
fn translate_permission_outcome<F>(
    outcome: RequestPermissionOutcome,
    tool_name: &str,
    mut record_sticky: F,
) -> PermissionOutcome
where
    F: FnMut(StickyDecision),
{
    match outcome {
        RequestPermissionOutcome::Cancelled => PermissionOutcome::Cancelled,
        RequestPermissionOutcome::Selected(selected) => {
            let option_id: &str = selected.option_id.0.as_ref();
            match option_id {
                OPTION_ALLOW_ONCE => PermissionOutcome::Allow,
                OPTION_ALLOW_ALWAYS => {
                    record_sticky(StickyDecision::AllowAlways);
                    PermissionOutcome::Allow
                }
                OPTION_REJECT_ONCE => PermissionOutcome::Deny,
                OPTION_REJECT_ALWAYS => {
                    record_sticky(StickyDecision::RejectAlways);
                    PermissionOutcome::Deny
                }
                other => {
                    tracing::debug!(
                        "request_permission for '{}' returned unknown option_id '{}'; \
                         defaulting to Deny",
                        tool_name,
                        other,
                    );
                    PermissionOutcome::Deny
                }
            }
        }
        // ACP's `RequestPermissionOutcome` is `#[non_exhaustive]`; any future variant we haven't
        // taught the agent about should fail closed.
        other => {
            tracing::debug!(
                "request_permission for '{}' returned unknown outcome {:?}; \
                 defaulting to Deny",
                tool_name,
                other,
            );
            PermissionOutcome::Deny
        }
    }
}

/// Map meka's tool name to ACP's [`ToolKind`] so clients can pick the right icon and grouping.
/// MCP-loaded tools (named `mcp__server__tool`) and anything unknown fall through to `Other`.
fn tool_kind_for(name: &str) -> ToolKind {
    match name {
        "read_file" | "todo" => ToolKind::Read,
        "edit_file" | "write_file" => ToolKind::Edit,
        "find_files" | "search_contents" => ToolKind::Search,
        "execute_command" => ToolKind::Execute,
        "fetch_url" | "search_web" => ToolKind::Fetch,
        "agent_spawn" => ToolKind::Think,
        // skill, memory_*, scratchpad_*, render_image, load_tool, mcp__*, and any
        // future built-ins.
        _ => ToolKind::Other,
    }
}

/// Build the human-readable `title` for a tool call from the resolved primary argument
/// (`display_summary`: the command for `execute_command`, the path for `read_file`, the URL for
/// `fetch_url`, ...). Mirrors claude-agent-acp: editors should show what's running, not the bare
/// tool name. `raw_input` still carries the full argument object for clients that want it.
fn tool_call_title(name: &str, display_summary: Option<&str>) -> String {
    let arg = display_summary.map(str::trim).filter(|s| !s.is_empty());
    let raw = match (name, arg) {
        ("execute_command", Some(command)) => command.to_string(),
        ("read_file", Some(path)) => format!("Read {path}"),
        ("edit_file", Some(path)) => format!("Edit {path}"),
        ("write_file", Some(path)) => format!("Write {path}"),
        ("find_files", Some(pattern)) => format!("Find {pattern}"),
        ("search_contents", Some(pattern)) => format!("Search {pattern}"),
        ("fetch_url", Some(url)) => format!("Fetch {url}"),
        ("search_web", Some(query)) => format!("Web search: {query}"),
        ("agent_spawn", Some(task)) => format!("Sub-agent: {task}"),
        // Any other built-in or MCP (`mcp__server__tool`) tool: show its primary argument when one
        // was resolved (via the tool's JSON Schema), else fall back to the bare tool name.
        (other, Some(argument)) => format!("{other}: {argument}"),
        (other, None) => other.to_string(),
    };
    sanitize_title(&raw)
}

/// Collapse internal whitespace (so a multi-line command becomes a one-line title) and cap the
/// length so an editor never gets an unwieldy title. Mirrors claude-agent-acp's `sanitizeTitle`.
fn sanitize_title(text: &str) -> String {
    const MAX_TITLE_CHARS: usize = 256;
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= MAX_TITLE_CHARS {
        collapsed
    } else {
        let truncated: String = collapsed.chars().take(MAX_TITLE_CHARS - 1).collect();
        format!("{truncated}…")
    }
}

/// Convert meka's `todo` tool list into ACP [`PlanEntry`] rows for [`SessionUpdate::Plan`]. meka's
/// `Cancelled` status has no ACP analogue, so it maps to `Completed` ("no longer active") to keep
/// the entry count stable against the model's own todo list. meka tracks no per-item priority, so
/// every entry is reported as `Medium`.
fn todo_items_to_plan(items: &[TodoItem]) -> Vec<PlanEntry> {
    items
        .iter()
        .map(|item| {
            let status = match item.status {
                TodoStatus::Pending => PlanEntryStatus::Pending,
                TodoStatus::InProgress => PlanEntryStatus::InProgress,
                TodoStatus::Completed | TodoStatus::Cancelled => PlanEntryStatus::Completed,
            };
            PlanEntry::new(item.text.clone(), PlanEntryPriority::Medium, status)
        })
        .collect()
}

/// Append one prompt content block to `prompt_text`, inserting a newline separator between blocks.
fn push_prompt_block(prompt_text: &mut String, block: &str) {
    if !prompt_text.is_empty() {
        prompt_text.push('\n');
    }
    prompt_text.push_str(block);
}

/// Append a ` mime="..."` attribute to a resource/resource_link tag when one is present.
fn push_mime_attr(tag: &mut String, mime: &Option<String>) {
    if let Some(mime) = mime {
        tag.push_str(&format!(" mime=\"{}\"", mime));
    }
}

/// Render an ACP embedded resource (an @-mention's inlined contents) as a `<resource>` tag for the
/// prompt body. Text resources inline their contents; binary (blob) resources emit a self-closing
/// marker without the (potentially huge) payload, so the model still learns the reference exists.
///
/// A distinct `<resource>` tag (not `<context>`) is deliberate: the stored user message is wrapped
/// by the agent's own `<context>...</context>` preamble, and [`crate::session::strip_context_tags`]
/// keys on that first `</context>`. A `<resource>` tag therefore round-trips through history replay
/// exactly like `<resource_link>` does.
fn format_embedded_resource(embedded: &EmbeddedResource) -> String {
    match &embedded.resource {
        EmbeddedResourceResource::TextResourceContents(text) => {
            let mut tag = format!("<resource uri=\"{}\"", text.uri);
            push_mime_attr(&mut tag, &text.mime_type);
            tag.push('>');
            tag.push_str(&text.text);
            tag.push_str("</resource>");
            tag
        }
        EmbeddedResourceResource::BlobResourceContents(blob) => {
            let mut tag = format!("<resource uri=\"{}\"", blob.uri);
            push_mime_attr(&mut tag, &blob.mime_type);
            tag.push_str(" encoding=\"base64\"/>");
            tag
        }
        // `EmbeddedResourceResource` is `#[non_exhaustive]`; a future variant we can't introspect
        // still gets a bare marker so the prompt stays well-formed.
        _ => "<resource/>".to_string(),
    }
}

/// Validate a client's `additionalDirectories`, rejecting any relative entry.
///
/// The spec requires each to be absolute, and meka has no defensible base to resolve a relative one
/// against: joining to `cwd` would invent a root the client never named. Failing the request is the
/// honest answer, and it matches how `cwd` itself is validated.
fn validate_additional_roots(roots: &[PathBuf]) -> Result<(), agent_client_protocol::Error> {
    for root in roots {
        if !root.is_absolute() {
            return Err(invalid_params_error(format!(
                "additionalDirectories entries must be absolute paths; got `{}`",
                root.display()
            )));
        }
    }
    Ok(())
}

/// Decode an ACP image content block into meka's internal [`crate::provider::ImageSource`] via the
/// shared client-image pipeline, so ACP and the HTTP API enforce the same limits.
///
/// Off the runtime, for the reason `read_file` and `fetch_url` document at their own call sites:
/// the pipeline base64-decodes and then decodes the image to verify it, which is tens of
/// milliseconds of pure CPU on a multi-megapixel screenshot, and on the runtime it blocks every
/// other task on that worker. The editor pasting one attachment must not stall an unrelated
/// session's stream.
async fn decode_acp_image(image: &ImageContent) -> Result<crate::provider::ImageSource, String> {
    let data = image.data.clone();
    let mime_type = image.mime_type.clone();
    tokio::task::spawn_blocking(move || crate::image::decode_base64_image(&data, &mime_type))
        .await
        .map_err(|error| format!("image decode task failed: {}", error))?
}

/// Compute the `locations` entries for a tool call. For tools whose primary argument is a path,
/// resolve it against the agent's per-session cwd (ACP requires absolute paths). Anything else
/// returns an empty list; clients fall back to the `raw_input` field.
fn tool_locations(name: &str, input: &serde_json::Value, cwd: &SharedCwd) -> Vec<ToolCallLocation> {
    let raw = match name {
        "read_file" | "edit_file" | "write_file" | "find_files" | "search_contents" => {
            input.get("path").and_then(|v| v.as_str())
        }
        _ => None,
    };
    raw.map(|path| {
        let mut location = ToolCallLocation::new(resolve_against_cwd(cwd, path));
        // For `read_file`, point the client at the first line being read. meka's `offset` is
        // 0-based; ACP line numbers are 1-based.
        if name == "read_file"
            && let Some(offset) = input.get("offset").and_then(|value| value.as_u64())
        {
            location = location.line(u32::try_from(offset.saturating_add(1)).unwrap_or(u32::MAX));
        }
        vec![location]
    })
    .unwrap_or_default()
}

/// Wrap a string as a plain-text [`ToolCallContent`] block.
fn text_content_block(text: impl Into<String>) -> ToolCallContent {
    ToolCallContent::from(ContentBlock::Text(
        agent_client_protocol::schema::v1::TextContent::new(text.into()),
    ))
}

/// Build the `tool_call_update` that carries a running command's output, in whichever shape the
/// client understands. Terminal mode appends `text` to the client's scrollback and repeats the
/// content block so the terminal stays attached; text mode replaces the content with the window
/// meka is holding.
fn live_output_update(id: &str, mode: LiveOutputMode, text: &str) -> ToolCallUpdate {
    match mode {
        LiveOutputMode::Terminal => {
            let fields = ToolCallUpdateFields::new().content(vec![ToolCallContent::Terminal(
                agent_client_protocol::schema::v1::Terminal::new(id.to_string()),
            )]);
            ToolCallUpdate::new(id.to_string(), fields).meta(terminal_meta(
                META_TERMINAL_OUTPUT,
                serde_json::json!({ "terminal_id": id, "data": text }),
            ))
        }
        LiveOutputMode::Text => {
            let fields = ToolCallUpdateFields::new().content(vec![console_content_block(text)]);
            ToolCallUpdate::new(id.to_string(), fields)
        }
    }
}

/// Exit code for a finished command's terminal frame. Falls back to the coarse "did it fail" bit
/// when the tool didn't report a code (a signal kill, or a tool error raised before the spawn).
fn command_exit_code(metadata: &Option<ToolOutputMetadata>, is_error: bool) -> Option<i32> {
    match metadata {
        Some(ToolOutputMetadata::CommandExit { exit_code, .. }) => *exit_code,
        _ => Some(i32::from(is_error)),
    }
}

/// Signal name for a finished command's terminal frame, when it was killed rather than exiting.
fn command_signal(metadata: &Option<ToolOutputMetadata>) -> Option<String> {
    match metadata {
        Some(ToolOutputMetadata::CommandExit { signal, .. }) => signal.clone(),
        _ => None,
    }
}

/// Wrap shell output in a `console` code block so editors render it monospaced (mirrors
/// claude-agent-acp's no-terminal fallback). Shared by the live view a command streams while it
/// runs and the final one emitted when it exits, so output doesn't reflow when the call completes.
fn console_content_block(output: &str) -> ToolCallContent {
    text_content_block(format!("```console\n{}\n```", output.trim_end()))
}

/// Build the `content` array of a `tool_call_update` from meka's tool output. A populated `Diff`
/// metadata wins (so clients like Zed get the structured diff for apply-UI). `execute_command`
/// output is wrapped in a `console` code block so editors render it monospaced (mirrors
/// claude-agent-acp's no-terminal fallback). Other tools pass their text and image blocks through
/// unchanged, so a tool that looked at an image (`read_file` on a PNG, `render_image`, `fetch_url`)
/// shows the human the same picture the model saw.
fn build_completion_content(
    tool_name: &str,
    content: &[ToolResultContent],
    metadata: Option<ToolOutputMetadata>,
) -> Vec<ToolCallContent> {
    if let Some(ToolOutputMetadata::Diff {
        path,
        old_text,
        new_text,
    }) = metadata
    {
        let mut diff = Diff::new(path, new_text);
        if let Some(old) = old_text {
            diff = diff.old_text(old);
        }
        return vec![ToolCallContent::Diff(diff)];
    }

    if tool_name == "execute_command" {
        // Reuse the canonical text-flattening; `execute_command` output is text-only, so the
        // `[Image]` marker `tool_result_text_content` would emit for images never appears here.
        let combined = MekaContentBlock::tool_result_text_content(content);
        if combined.trim_end().is_empty() {
            return Vec::new();
        }
        return vec![console_content_block(&combined)];
    }

    content
        .iter()
        .map(|block| match block {
            ToolResultContent::Text { text } => text_content_block(text.clone()),
            // The payload is already provider-normalized (size-capped, converted to a native
            // format) by the time it reaches here, so it can go out as-is.
            ToolResultContent::Image { source } => ToolCallContent::from(ContentBlock::Image(
                ImageContent::new(source.data.clone(), source.media_type.clone()),
            )),
        })
        .collect()
}

/// Walk a hydrated [`Conversation`] and emit one `session/update` notification per content
/// block, mirroring the streaming shape the client would have seen had it been connected during
/// the original turn. Used by `session/load` so an editor that just reopened a session replays the
/// full history into its UI.
///
/// `<context>...</context>` preambles meka prepends to each user message are stripped before emit
/// so the client sees only what the user actually typed.
///
/// Tool calls track open `tool_use_id`s; any tool that never received a matching `ToolResult` (e.g.
/// a crashed turn) is closed out with a `failed` `tool_call_update` so the client doesn't render a
/// stuck spinner.
fn replay_session_updates(
    connection: &ConnectionTo<Client>,
    session_id: &SessionId,
    cwd: &SharedCwd,
    messages: &Conversation,
) {
    // Map each open `tool_use_id` to its tool name so the result update can format output per tool
    // and the orphan sweep can close stragglers.
    let mut open_tools: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for message in messages.as_slice() {
        match message.role {
            Role::User => {
                for block in &message.content {
                    match block {
                        MekaContentBlock::Text { text } => {
                            let stripped = crate::session::strip_context_tags(text);
                            if !stripped.is_empty() {
                                send_session_update(
                                    connection,
                                    session_id,
                                    SessionUpdate::UserMessageChunk(ContentChunk::new(
                                        ContentBlock::Text(
                                            agent_client_protocol::schema::v1::TextContent::new(
                                                stripped.to_string(),
                                            ),
                                        ),
                                    )),
                                );
                            }
                        }
                        MekaContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => {
                            let status = if *is_error {
                                ToolCallStatus::Failed
                            } else {
                                ToolCallStatus::Completed
                            };
                            let tool_name = open_tools
                                .get(tool_use_id)
                                .map(String::as_str)
                                .unwrap_or("");
                            let acp_content = build_completion_content(tool_name, content, None);
                            let fields = ToolCallUpdateFields::new()
                                .status(status)
                                .content(acp_content);
                            send_session_update(
                                connection,
                                session_id,
                                SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                                    tool_use_id.clone(),
                                    fields,
                                )),
                            );
                            open_tools.remove(tool_use_id);
                        }
                        // Re-emit input images so a reopened session shows the attachment.
                        MekaContentBlock::Image { source } => {
                            send_session_update(
                                connection,
                                session_id,
                                SessionUpdate::UserMessageChunk(ContentChunk::new(
                                    ContentBlock::Image(ImageContent::new(
                                        source.data.clone(),
                                        source.media_type.clone(),
                                    )),
                                )),
                            );
                        }
                        _ => {}
                    }
                }
            }
            Role::Assistant => {
                for block in &message.content {
                    match block {
                        MekaContentBlock::Text { text } => {
                            send_session_update(
                                connection,
                                session_id,
                                SessionUpdate::AgentMessageChunk(ContentChunk::new(
                                    ContentBlock::Text(
                                        agent_client_protocol::schema::v1::TextContent::new(
                                            text.clone(),
                                        ),
                                    ),
                                )),
                            );
                        }
                        MekaContentBlock::Thinking { thinking, .. } => {
                            send_session_update(
                                connection,
                                session_id,
                                SessionUpdate::AgentThoughtChunk(ContentChunk::new(
                                    ContentBlock::Text(
                                        agent_client_protocol::schema::v1::TextContent::new(
                                            thinking.clone(),
                                        ),
                                    ),
                                )),
                            );
                        }
                        MekaContentBlock::RedactedThinking { .. } => {
                            send_session_update(
                                connection,
                                session_id,
                                SessionUpdate::AgentThoughtChunk(ContentChunk::new(
                                    ContentBlock::Text(
                                        agent_client_protocol::schema::v1::TextContent::new(
                                            "[redacted thinking]".to_string(),
                                        ),
                                    ),
                                )),
                            );
                        }
                        MekaContentBlock::ToolUse { id, name, input } => {
                            let locations = tool_locations(name, input, cwd);
                            // Match the live path's rich title. No tool schema is available on
                            // replay, so only built-in tools resolve a primary argument; MCP tools
                            // fall back to the bare name.
                            let display_summary =
                                crate::render::resolve_primary_param(name, input, None);
                            let title = tool_call_title(name, display_summary.as_deref());
                            let call = ToolCall::new(id.clone(), title)
                                .kind(tool_kind_for(name))
                                .status(ToolCallStatus::InProgress)
                                .locations(locations)
                                .raw_input(input.clone());
                            send_session_update(
                                connection,
                                session_id,
                                SessionUpdate::ToolCall(call),
                            );
                            open_tools.insert(id.clone(), name.clone());
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Tool calls without a matching result: close them as failed so the client's "tool running"
    // indicator doesn't get stuck.
    for orphan_id in open_tools.into_keys() {
        let fields = ToolCallUpdateFields::new().status(ToolCallStatus::Failed);
        send_session_update(
            connection,
            session_id,
            SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(orphan_id, fields)),
        );
    }
}

fn send_session_update(
    connection: &ConnectionTo<Client>,
    session_id: &SessionId,
    update: SessionUpdate,
) {
    if let Err(error) =
        connection.send_notification(SessionNotification::new(session_id.clone(), update))
    {
        // Every `session/update` goes out through here, not just `session/load` replay, so name the
        // notification rather than one caller: this is the line to look at when a client reports
        // that updates aren't arriving.
        tracing::debug!("session/update send_notification failed: {}", error);
    }
}

/// The first user message's preview text (the basis for the session title), or `None` if the
/// conversation carries no user text yet. The stored text still has the agent's `<context>`
/// preamble, which [`crate::session::truncate_preview`] strips.
fn first_user_preview(messages: &Conversation) -> Option<String> {
    for message in messages.as_slice() {
        if message.role != Role::User {
            continue;
        }
        for block in &message.content {
            if let MekaContentBlock::Text { text } = block {
                let preview = crate::session::truncate_preview(text, 80);
                if !preview.is_empty() {
                    return Some(preview);
                }
            }
        }
    }
    None
}

/// Emit a `session_info_update` carrying the session title exactly once. The title is the first
/// user message preview, which never changes after the first turn, so `title_sent` guards against
/// re-emission across the first prompt and any later load/resume of the same session.
fn maybe_emit_session_title(
    connection: &ConnectionTo<Client>,
    session_id: &SessionId,
    title_sent: &std::sync::atomic::AtomicBool,
    messages: &Conversation,
) {
    use std::sync::atomic::Ordering;
    if title_sent.load(Ordering::Acquire) {
        return;
    }
    let Some(title) = first_user_preview(messages) else {
        return;
    };
    // Claim the one-shot before sending; if a concurrent path beat us to it, skip.
    if title_sent.swap(true, Ordering::AcqRel) {
        return;
    }
    send_session_update(
        connection,
        session_id,
        SessionUpdate::SessionInfoUpdate(SessionInfoUpdate::new().title(title)),
    );
}

/// Map a meka [`Permission`] to its ACP [`SessionModeId`] string: the same lowercase word
/// `Permission::Display` produces, so the id a client reads back is the one it sees in
/// `config.toml` and on the `--permission` flag.
fn mode_id_for(permission: Permission) -> SessionModeId {
    SessionModeId::from(permission.to_string())
}

/// Parse a `SessionModeId` (treated as a `&str`) into the matching `Permission`. Returns `None` for
/// any unrecognised mode id, which the caller turns into an error response.
///
/// Delegates to [`Permission`]'s [`std::str::FromStr`] rather than keeping its own table. A second
/// hand-maintained copy is the shape that grants a client a mode it did not ask for: the two have
/// to stay in lock-step, and when they drift it is an id silently mapping to the wrong rung. One
/// table cannot drift from itself.
fn parse_mode_id(id: &str) -> Option<Permission> {
    id.parse().ok()
}

/// Human-readable label for a permission mode, shown in editor mode pickers next to each option.
/// Kept in lock-step with the REPL's `/permission` output and the `[permissions]` keys in
/// `config.toml` so users see the same vocabulary everywhere.
fn mode_display_name(permission: Permission) -> &'static str {
    match permission {
        Permission::None => "None",
        Permission::Read => "Read",
        Permission::Workspace => "Workspace",
        Permission::Ask => "Ask",
        Permission::Unrestricted => "Unrestricted",
    }
}

/// One-line description of what a permission mode lets the agent do. Shown beneath the mode label
/// in editor pickers.
///
/// `Unrestricted` is described by its *reach*, not by the absence of approval prompts: "all tools
/// without per-call approval" is equally true of `Workspace`, so it never distinguished the two.
fn mode_description(permission: Permission) -> &'static str {
    match permission {
        Permission::None => "No tools available.",
        Permission::Read => "File reads and searches only. No writes, no shell.",
        Permission::Workspace => "Writes confined to the workspace roots. No approval prompts.",
        Permission::Ask => "Every write or shell command requires approval.",
        Permission::Unrestricted => "Writes and shell commands reach anywhere on the machine.",
    }
}

/// Build the `SessionModeState` advertised on every session-creation response (`session/new`,
/// `session/load`, `session/resume`). Only modes in [`SharedPermission::enabled`] are exposed:
/// picking a non-enabled mode through `session/set_mode` later would just error out, so we don't
/// surface them in the first place.
fn build_mode_state(permission: &SharedPermission) -> SessionModeState {
    let modes: Vec<SessionMode> = permission
        .enabled()
        .iter()
        .map(|mode| {
            SessionMode::new(mode_id_for(mode), mode_display_name(mode))
                .description(mode_description(mode))
        })
        .collect();
    SessionModeState::new(mode_id_for(permission.get()), modes)
}

/// The `configOptions` id for the permission picker, which duplicates the legacy `modes` field.
///
/// Both are advertised, the way the reference adapter does it: `modes` and `configOptions` are
/// separate response fields rendered in separate places, so a client that only understands `modes`
/// keeps its picker, and one that understands `configOptions` gets permission and provider side by
/// side rather than in two unrelated menus.
const PERMISSION_CONFIG_ID: &str = "permission";

/// The `configOptions` id for the provider picker. There is no legacy field for this one.
const PROVIDER_CONFIG_ID: &str = "provider";

/// The permission picker as a `configOptions` entry. The same enabled set [`build_mode_state`]
/// exposes, for the same reason: a level the client cannot actually be granted has no business in
/// the list.
fn permission_config_option(permission: &SharedPermission) -> SessionConfigOption {
    SessionConfigOption::select(
        PERMISSION_CONFIG_ID,
        "Permission",
        SessionConfigValueId::from(permission.get().to_string()),
        permission
            .enabled()
            .iter()
            .map(|mode| {
                SessionConfigSelectOption::new(
                    SessionConfigValueId::from(mode.to_string()),
                    mode_display_name(mode),
                )
                .description(mode_description(mode))
            })
            .collect::<Vec<_>>(),
    )
    .category(SessionConfigOptionCategory::Mode)
    .description("What the agent may do without asking.")
}

/// The provider picker as a `configOptions` entry.
///
/// `current` may name no configured profile, which is what a session whose profile was deleted from
/// `config.toml` looks like. Nothing is invented for it: no option matches, so a client renders
/// "nothing selected", which is the truth. Inventing a selection would show a profile the session
/// is not going to run on.
fn provider_config_option(
    profiles: &std::collections::BTreeMap<String, crate::config::ProviderProfile>,
    current: &str,
) -> SessionConfigOption {
    SessionConfigOption::select(
        PROVIDER_CONFIG_ID,
        "Provider",
        SessionConfigValueId::from(current.to_string()),
        profiles
            .iter()
            .map(|(name, profile)| {
                let option = SessionConfigSelectOption::new(
                    SessionConfigValueId::from(name.clone()),
                    name.clone(),
                );
                // The model, when the profile names one. A profile that leaves it to the provider
                // has nothing truthful to put here, and inventing a label would be meka asserting
                // a fact about someone else's system.
                match &profile.model {
                    Some(model) => option.description(model.clone()),
                    None => option,
                }
            })
            .collect::<Vec<_>>(),
    )
    .category(SessionConfigOptionCategory::Model)
    .description("The provider profile this session runs on.")
}

/// Build the `configOptions` list advertised on every session-creation response and returned by
/// `session/set_config_option`.
///
/// The provider's current value is read from the session row rather than from the live agent: the
/// row is what the next turn resolves against, and the agent is behind the runtime mutex that an
/// in-flight prompt holds.
async fn build_config_options(
    shared: &crate::SharedDeps,
    permission: &SharedPermission,
    session_uuid: Option<uuid::Uuid>,
) -> Vec<SessionConfigOption> {
    let current_provider = match session_uuid {
        Some(session_uuid) => shared
            .session_manager
            .recorded_provider(session_uuid)
            .await
            .unwrap_or_else(|error| {
                tracing::warn!(
                    "could not read the recorded provider for session {}: {}",
                    session_uuid,
                    error
                );
                None
            })
            .unwrap_or_default(),
        None => String::new(),
    };
    vec![
        permission_config_option(permission),
        provider_config_option(&shared.config.providers, &current_provider),
    ]
}

/// Whether a session's profile accepts image input.
///
/// Read from the row on every prompt rather than cached on the session entry, because the row is
/// what [`apply_recorded_binding`] puts that same prompt's turn on: a copy taken when the session
/// was created described whichever profile it started on, and every writer of the row had to
/// remember to push a new value at it.
///
/// **`serve` answers the same question differently, on purpose.** It reads
/// `SessionEntry::binding`, the cell the agent publishes into
/// (`crate::server::handlers::turn`), because its `PATCH` moves the agent under the runtime mutex
/// before returning, so the published binding and the row cannot disagree. ACP cannot do that: it
/// must not block the dispatch loop on that mutex, so a switch made mid-turn leaves the agent
/// behind until the next turn applies it, and a published binding would report the profile the
/// session is *leaving*. The row is the only thing both hosts agree is authoritative, and it is
/// the one ACP has to read.
///
/// A session whose profile cannot resolve answers `false`; its next prompt fails on that same
/// profile either way, and taking an attachment first would only add a second failure.
async fn session_accepts_images(state: &ServerState, session_uuid: uuid::Uuid) -> bool {
    match crate::provider::provider_for_config(
        &state.shared.session_manager,
        &state.shared.config,
        Some(session_uuid),
    )
    .await
    {
        Ok(binding) => crate::provider::binding_accepts_images(&state.shared.providers, &binding),
        Err(error) => {
            tracing::warn!(
                "could not resolve the provider for session {} to decide image support: {}",
                session_uuid,
                error
            );
            false
        }
    }
}

/// Move a session's agent onto whatever its row currently names, before a turn runs on it.
///
/// **The row is the carrier, and the only one.** `session/set_config_option` writes it before it
/// reaches for the runtime mutex, precisely so a switch it cannot apply is not lost; this is what
/// applies it. A resolved binding parked on the session entry is drained only by `session/prompt`,
/// so a scheduled fire or a background-outcome turn runs on, and bills, the profile the user has
/// left, while the row, both pickers and the reported window all say otherwise. A parked value was
/// a second carrier of a fact the row already held, and it could lose to any other writer of that
/// row.
///
/// Cheap when nothing has changed: one indexed row read and a comparison, with no resolution at all
/// unless the two differ.
///
/// Must be called under the runtime mutex, which is what makes "the agent this turn is about to
/// use" the thing being moved.
async fn apply_recorded_binding(
    state: &ServerState,
    runtime: &mut SessionRuntime,
) -> anyhow::Result<()> {
    let Some(recorded) = state
        .shared
        .session_manager
        .recorded_provider(runtime.session_uuid)
        .await?
    else {
        // No row, so nothing names a profile to move to. Reachable only for a session deleted from
        // under a live entry; its turn is going to fail on the write either way.
        return Ok(());
    };
    if &recorded == runtime.agent.provider_binding() {
        return Ok(());
    }
    let profile = recorded.clone();
    let resolved = crate::provider::resolved_binding(&state.shared.providers, recorded).await?;
    runtime.agent.set_provider(resolved);
    tracing::info!(
        "session {} moved onto provider profile `{}`",
        runtime.session_uuid,
        profile
    );
    Ok(())
}

/// Push a `config_option_update` so a client's pickers reflect a change it did not make, whether
/// that was another surface repinning the session or meka's own `session/set_mode` handler.
async fn emit_config_options(
    state: &ServerState,
    entry: &SessionEntry,
    session_uuid: Option<uuid::Uuid>,
) {
    let options = build_config_options(&state.shared, &entry.permission, session_uuid).await;
    send_session_update(
        &entry.frontend.connection,
        &entry.frontend.session_id,
        SessionUpdate::ConfigOptionUpdate(ConfigOptionUpdate::new(options)),
    );
}

/// Slash commands meka handles entirely agent-side: they render text and end the turn with no model
/// call, unlike skills (which resolve to prompt text and run the model). `(name, description)`.
const LOCAL_COMMANDS: &[(&str, &str)] = &[
    (
        "status",
        "Show session status: model, context usage, tokens, mode",
    ),
    (
        "mcp",
        "List configured MCP servers and their connection status",
    ),
    (
        "usage",
        "Show account rate-limit usage (subscription providers)",
    ),
];

/// If `prompt_text` is a [`LOCAL_COMMANDS`] invocation, render its output; otherwise `None` so the
/// prompt falls through to skill resolution / the model. Text after the command name is ignored
/// (these commands take no arguments).
async fn try_local_command(
    prompt_text: &str,
    runtime: &SessionRuntime,
    shared: &crate::SharedDeps,
) -> Option<String> {
    let (name, _extra) = split_acp_slash(prompt_text)?;
    match name.as_str() {
        "status" => Some(build_status_text(runtime, shared)),
        "mcp" => Some(build_mcp_list_text(shared).await),
        "usage" => Some(build_usage_text(runtime).await),
        _ => None,
    }
}

/// Plain-text `/usage` output for ACP clients, reusing the shared `render::format_account_usage`.
async fn build_usage_text(runtime: &SessionRuntime) -> String {
    match runtime.agent.fetch_usage().await {
        Ok(Some(usage)) => crate::render::format_account_usage(&usage),
        Ok(None) => "Account usage isn't available for this provider.".to_string(),
        Err(error) => format!("Error fetching usage: {error}"),
    }
}

/// Plain-text `/status` output: the REPL's numbers in a narrower envelope, plus the permission mode
/// an ACP client may not otherwise surface.
///
/// Deliberately not [`crate::render::format_session_status`], though that now exists for exactly
/// this kind of caller. This block is read inside an editor pane rather than a terminal, so it uses
/// shorter labels and drops the fields an ACP client already shows in its own chrome (provider,
/// thinking, redactions). Sharing the formatter would change what every ACP client displays, which
/// is a product decision rather than a refactor. If the two should converge, converge them on
/// purpose; the shared formatter is there when that call is made.
fn build_status_text(runtime: &SessionRuntime, shared: &crate::SharedDeps) -> String {
    use std::fmt::Write as _;
    let snap = runtime.agent.session_stats_snapshot();
    let (used, window) = runtime.agent.context_usage();
    let mut out = String::from("meka session status\n");
    // This session's profile, not the process default's: two ACP sessions on one connection may sit
    // on different profiles, and the effort and context lines below already come from this one.
    let binding = runtime.agent.provider_binding();
    if let Ok(settings) = shared.providers.settings(binding)
        && let Some(model) = settings.model.as_deref()
    {
        let _ = writeln!(out, "  Model:    {model}");
    }
    if let Some(effort) = runtime.agent.resolved_effort() {
        let _ = writeln!(out, "  Effort:   {effort}");
    }
    let _ = writeln!(
        out,
        "  Mode:     {}",
        mode_display_name(runtime.permission.get())
    );
    if window > 0 && used > 0 {
        let pct = ((used as f64 / window as f64) * 100.0).round() as u64;
        let left = window.saturating_sub(used);
        let _ = writeln!(
            out,
            "  Context:  {} / {} ({pct}% used, {} left)",
            crate::render::format_token_count(used),
            crate::render::format_token_count(window),
            crate::render::format_token_count(left),
        );
    }
    let _ = writeln!(
        out,
        "  Tokens:   in {} (cache hit {}%) / out {}",
        crate::render::format_token_count(snap.total_input_tokens()),
        snap.cache_hit_pct(),
        crate::render::format_token_count(snap.output_tokens),
    );
    let _ = writeln!(out, "  Turns:    {}", snap.turns);
    let _ = write!(out, "  Messages: {}", runtime.messages.len());
    out
}

/// Plain-text `/mcp` output: each configured MCP server and its live connection state.
async fn build_mcp_list_text(shared: &crate::SharedDeps) -> String {
    use std::fmt::Write as _;
    let Some(manager) = shared.mcp_manager.as_ref() else {
        return "No MCP servers configured.".to_string();
    };
    let names = manager.server_names();
    if names.is_empty() {
        return "No MCP servers configured.".to_string();
    }
    let mut out = String::from("MCP servers\n");
    for name in names {
        let status = match manager.server_entry(&name) {
            Some(entry) => match entry.state().await {
                crate::mcp::ServerState::Failed { error, .. } => {
                    format!("failed: {}", error.lines().next().unwrap_or("").trim())
                }
                other => other.label().to_string(),
            },
            None => "unknown".to_string(),
        };
        let _ = writeln!(out, "  {name}: {status}");
    }
    out.truncate(out.trim_end().len());
    out
}

/// Emit a `session/update: available_commands_update` listing meka's built-in local commands
/// ([`LOCAL_COMMANDS`]) followed by every installed skill, each as an [`AvailableCommand`]. Editor
/// clients render these as slash commands in their prompt input; picking one inserts `/<name> `.
/// Skills whose name collides with a built-in command are dropped so the palette has no duplicates.
///
/// `SkillCache::current` is mtime-cached, so calling this at the top of every prompt is cheap (one
/// `read_dir`, no parsing on the warm path).
async fn emit_available_commands(
    connection: &ConnectionTo<Client>,
    session_id: &SessionId,
    skills: &Arc<SkillCache>,
) {
    let snapshot = skills.current().await;
    let mut commands: Vec<AvailableCommand> = LOCAL_COMMANDS
        .iter()
        .map(|(name, description)| AvailableCommand::new(*name, *description))
        .collect();
    commands.extend(
        snapshot
            .skills
            .iter()
            .filter(|skill| !LOCAL_COMMANDS.iter().any(|(name, _)| *name == skill.name))
            .map(|skill| {
                // Sanitised, like every other place a skill description is shown. The store hands
                // back the file's bytes now, so this is a render boundary: a description carrying a
                // bidi override or a control character would otherwise reach the editor's command
                // palette over JSON-RPC and be drawn by whatever renders it.
                AvailableCommand::new(
                    skill.name.clone(),
                    crate::memory::render_description_for_model(&skill.description),
                )
                .input(AvailableCommandInput::Unstructured(
                    UnstructuredCommandInput::new("additional context (optional)"),
                ))
            }),
    );
    send_session_update(
        connection,
        session_id,
        SessionUpdate::AvailableCommandsUpdate(AvailableCommandsUpdate::new(commands)),
    );
}

/// Outcome of running a slash-command parse against an ACP `session/prompt`'s text. Carries enough
/// detail for the prompt handler to either continue with the resolved text or surface a JSON-RPC
/// error explaining what went wrong.
#[derive(Debug)]
enum SlashInvocationError {
    /// The already-composed reason from [`crate::skills::SkillIndex::unavailable`], so this path
    /// distinguishes a name nobody wrote from a `SKILL.md` that will not parse rather than calling
    /// both "unknown skill".
    SkillNotFound(String),
    SkillLoadFailed {
        name: String,
        source: String,
    },
}

impl std::fmt::Display for SlashInvocationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlashInvocationError::SkillNotFound(reason) => write!(f, "{}", reason),
            SlashInvocationError::SkillLoadFailed { name, source } => {
                write!(f, "failed to load skill '{}': {}", name, source)
            }
        }
    }
}

/// Split an ACP prompt that looks like `/<name> [extra]` into the command name and the remainder.
/// Returns `None` if the input isn't in that shape, i.e. doesn't start with `/`, has only
/// whitespace after the slash, or contains a newline before the first whitespace (heuristic: a real
/// slash command never spans lines, but pasted content might).
fn split_acp_slash(prompt_text: &str) -> Option<(String, String)> {
    let rest = prompt_text.strip_prefix('/')?;
    if rest.is_empty() || rest.starts_with(char::is_whitespace) {
        return None;
    }
    Some(match rest.split_once(char::is_whitespace) {
        Some((name, extra)) => (name.to_string(), extra.trim().to_string()),
        None => (rest.to_string(), String::new()),
    })
}

/// Intercept `/<skill-name> [extra]` invocations in an ACP prompt's
/// text. Returns the text the agent should actually run with:
///
/// - Non-slash input: returned unchanged.
/// - Slash followed by a name that isn't a syntactically valid skill identifier (e.g. a pasted path
///   like `/etc/hosts` or a `//` comment): returned unchanged so the model can see it.
/// - `/<skill-name>` matching an installed skill: returns `extra\n\n{body}` where `body` is
///   [`crate::skills::load_skill_body`]'s output (the skill's base-directory header followed by its
///   body verbatim). Empty `extra` collapses to just `body`. Same composition the REPL's
///   `SlashCommand::SkillInvoke` handler uses; named rather than cited by line, because a line
///   number is a cross-reference that rots on the next edit and this one already had.
/// - `/<name>` with a syntactically valid skill name but no installed skill of that name: error.
///   The shape is too deliberate to be a paste, so a typo deserves a clear "unknown skill" rather
///   than silently going to the model.
async fn slash_to_prompt_text(
    prompt_text: String,
    skills: &Arc<SkillCache>,
) -> Result<String, SlashInvocationError> {
    let Some((name, extra)) = split_acp_slash(&prompt_text) else {
        return Ok(prompt_text);
    };
    // Anything that doesn't even look like a skill identifier was never going to match. Pass
    // through so pasted paths and code comments reach the model unchanged.
    //
    // Narrower than the rule the delete doors apply, on purpose: those decide whether a name is
    // safe to act on, and this decides whether the user meant a skill at all. A skill whose name
    // predates the spec still resolves (`/My_Skill`), but a line like `/v1.2 of the API` stays
    // prose rather than becoming "no such skill".
    if !crate::skills::looks_like_skill_invocation(&name) {
        return Ok(prompt_text);
    }
    let snapshot = skills.current().await;
    let Some(skill) = snapshot.find(&name) else {
        return Err(SlashInvocationError::SkillNotFound(
            snapshot.unavailable(&name),
        ));
    };
    let body = crate::skills::load_skill_body(skill)
        .await
        .map_err(|source| SlashInvocationError::SkillLoadFailed {
            name: name.clone(),
            source,
        })?;
    Ok(if extra.is_empty() {
        body
    } else {
        format!("{}\n\n{}", extra, body)
    })
}

/// Aborts a background task when the value is dropped. `run_acp` has several exit paths, and a
/// scheduler that outlived them would keep running turns against a connection nobody is reading.
struct AbortOnDrop(tokio::task::JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Process-wide ACP server state. The outer `sessions` `RwLock` is held only for map insert /
/// lookup / remove; per-session mutable state lives behind each entry's inner `Mutex` so a
/// long-running prompt on one session never blocks operations on another.
struct ServerState {
    shared: Arc<crate::SharedDeps>,
    client_state: SharedClientState,
    sessions: Arc<tokio::sync::RwLock<std::collections::HashMap<String, SessionEntry>>>,
    /// Shared with every per-session `AcpFrontend`; see the field on `AcpFrontend` for the
    /// stdio-level rationale.
    transport_dead: Arc<std::sync::atomic::AtomicBool>,
    /// The default profile's `vision` flag, and the only thing the advertised `image` prompt
    /// capability can be built from: `initialize` is answered before any session exists, so the
    /// capability is necessarily a property of the connection. Whether a given `session/prompt`
    /// *accepts* an image block is per session, from that session's own row; see
    /// [`session_accepts_images`].
    vision: bool,
}

/// Per-session map entry. Most fields live outside `runtime` so the cancel / set_mode / close
/// handlers can act without waiting for the long-held runtime mutex.
#[derive(Clone)]
struct SessionEntry {
    runtime: Arc<Mutex<SessionRuntime>>,
    /// In-flight turn's cancellation token. Rewritten at turn start inside `runtime`'s lock;
    /// cancel handler reads-and-clones it without touching `runtime`.
    cancellation: Arc<std::sync::RwLock<CancellationToken>>,
    /// Latch for cancels that arrive between turns. The prompt handler checks-and-clears it under
    /// the runtime lock after installing the new token, so a between-turn cancel signal isn't
    /// lost. See `acp_session_cancel_between_turns_applied_to_next_prompt`.
    cancel_pending: Arc<std::sync::atomic::AtomicBool>,
    /// Set once the session's `session_info_update` title has been emitted. The title is the first
    /// user message preview, stable after the first turn, so it is pushed exactly once (after that
    /// first turn, or at load/resume when history already carries it).
    title_sent: Arc<std::sync::atomic::AtomicBool>,
    /// Hoisted out of `SessionRuntime` so `set_mode` can flip the permission without waiting on
    /// the runtime mutex.
    permission: SharedPermission,
    /// Hoisted for the same reason as `permission`: `set_mode` / `close` need the connection to
    /// emit notifications without blocking on the runtime mutex.
    frontend: Arc<AcpFrontend>,
    /// Held purely for its `Drop` side-effect: dropping releases the OS file lock on the persisted
    /// session row. Without this, a second `meka` process could attach to the same id.
    #[allow(dead_code)]
    session_lock: Arc<crate::session::FileLock>,
    /// When this session was last used, for the idle sweep.
    ///
    /// `session/close` is optional in ACP, and an editor is entitled never to send one. Every
    /// session it opens therefore keeps an `Agent`, a `ToolRegistry` the MCP manager holds a
    /// strong clone of, and an open file lock, for as long as the connection lives -- so an
    /// editor that opens one session per file it looks at eventually runs the process out of
    /// descriptors, and the session cannot be reopened elsewhere meanwhile. Monotonic, so a
    /// wall-clock adjustment cannot make a live session look ancient.
    last_activity: Arc<std::sync::RwLock<std::time::Instant>>,
}

impl SessionEntry {
    /// Install the cancellation token for a turn this session is about to run, so `session/cancel`
    /// reaches it. Mirrors what the prompt handler does; a scheduled turn needs it for the same
    /// reason, and without it the editor's stop button would do nothing.
    fn publish_cancellation(&self, token: CancellationToken) {
        match self.cancellation.write() {
            Ok(mut slot) => *slot = token,
            Err(poisoned) => *poisoned.into_inner() = token,
        }
    }

    /// Mark this session as used, so the idle sweep leaves it alone.
    fn touch(&self) {
        let now = std::time::Instant::now();
        match self.last_activity.write() {
            Ok(mut slot) => *slot = now,
            Err(poisoned) => *poisoned.into_inner() = now,
        }
    }

    /// Whether nothing has touched this session for `timeout` *and* no turn is running.
    ///
    /// The runtime lock is the liveness check: the prompt handler holds it for the whole turn, so a
    /// long turn on a session whose `last_activity` predates it cannot be evicted out from under
    /// itself.
    fn is_idle(&self, timeout: std::time::Duration) -> bool {
        session_is_idle(&self.runtime, &self.last_activity, timeout)
    }
}

/// The idle test itself, generic over what the runtime mutex guards so it can be exercised without
/// standing up an `Agent`.
///
/// The busy check comes first and is not an optimisation: a session mid-turn has a `last_activity`
/// from before the turn started, so on a long turn the timestamp alone says "idle" while the agent
/// is working. Evicting there would drop the entry out from under a running turn.
fn session_is_idle<T>(
    runtime: &Mutex<T>,
    last_activity: &std::sync::RwLock<std::time::Instant>,
    timeout: std::time::Duration,
) -> bool {
    if runtime.try_lock().is_err() {
        return false;
    }
    let last = last_activity
        .read()
        .map(|slot| *slot)
        .unwrap_or_else(|poisoned| *poisoned.into_inner());
    last.elapsed() >= timeout
}

/// Per-session state held under `SessionEntry.runtime`. Held inside a `Mutex` because
/// `Agent::run_turn` mutates the conversation. The `frontend` field duplicates
/// `SessionEntry.frontend` so the agent (which only knows `Arc<dyn Frontend>`) can reach the
/// connection.
struct SessionRuntime {
    /// Duplicates `frontend.session_id.0`; string form retained for handlers that need it without
    /// re-extracting from the schema.
    #[allow(dead_code)]
    session_id_str: String,
    session_uuid: uuid::Uuid,
    messages: Conversation,
    cwd: SharedCwd,
    permission: SharedPermission,
    agent: Agent,
    #[allow(dead_code)]
    frontend: Arc<AcpFrontend>,
    tool_registry: crate::tools::ToolRegistry,
}

/// `futures::io::AsyncRead` wrapper over the ACP stdin transport that fires `eof` (a
/// `CancellationToken`) when the underlying reader reports end-of-stream. The
/// `agent-client-protocol` connection future does not resolve on idle stdin EOF by itself (its
/// outgoing actor stays alive while we hold `ConnectionTo` handles), so we observe EOF here and let
/// `acp_run_until_disconnect` use it to shut down. Without this, a `meka acp` whose client
/// disconnected lingers forever holding its session `flock`, and reopening that session later fails
/// with `SessionLocked`.
struct EofSignalingRead<R> {
    inner: R,
    eof: CancellationToken,
}

impl<R: AsyncRead + Unpin> AsyncRead for EofSignalingRead<R> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        let this = self.get_mut();
        let result = Pin::new(&mut this.inner).poll_read(cx, buf);
        // A zero-length read into a non-empty buffer is end-of-stream: the client closed stdio (or
        // the parent died, closing the pipe). Fire the shutdown token; `cancel()` is idempotent so
        // repeated EOF reads are harmless.
        if matches!(result, Poll::Ready(Ok(0))) && !buf.is_empty() {
            this.eof.cancel();
        }
        result
    }
}

/// Max time to wait for in-flight turns to unwind during ACP shutdown before abandoning them. They
/// are abandoned safely regardless (the OS releases the session `flock` when the process exits),
/// but the grace window lets a running turn reach its interrupt path and persist its partial output
/// first.
const ACP_DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// `connect_with` `main_fn`: resolve (shutting the connection down) when the ACP client disconnects
/// (stdin EOF, signalled via `stdin_eof`) or a termination signal arrives. Before returning it
/// cancels every in-flight turn and waits briefly so a running turn can persist its partial output.
/// The connection's spawned turns run inside the `background` future this return races against, so
/// the drain must happen here, before we return and that future is dropped.
async fn acp_run_until_disconnect(
    state: Arc<ServerState>,
    stdin_eof: CancellationToken,
) -> std::result::Result<(), agent_client_protocol::Error> {
    tokio::select! {
        _ = stdin_eof.cancelled() => {
            tracing::info!("ACP client disconnected (stdin EOF); shutting down");
        }
        _ = acp_shutdown_signal() => {
            tracing::info!("received termination signal; shutting down ACP server");
        }
    }
    drain_acp_sessions(&state).await;
    if tokio::time::timeout(ACP_DRAIN_TIMEOUT, wait_for_sessions_idle(&state))
        .await
        .is_err()
    {
        tracing::warn!("ACP shutdown drain timed out; abandoning in-flight turn(s)");
    }

    // Reclaim every session the client never closed.
    //
    // `session/close` is an *optional* capability, so a client that simply exits leaves each entry
    // resident: an `Agent`, a `ToolRegistry` the MCP manager holds a strong clone of, and an open
    // file lock. Dropping the map here releases the flock -- which is what lets the same session be
    // reopened by the next `meka` without a `SessionLocked` error -- and lets the registry go, so
    // `tools/list_changed` stops fanning out to sessions that no longer exist.
    let abandoned = {
        let mut sessions = state.sessions.write().await;
        std::mem::take(&mut *sessions)
    };
    if !abandoned.is_empty() {
        if let Some(manager) = &state.shared.mcp_manager {
            for entry in abandoned.values() {
                // `try_lock`: an in-flight turn that outlived the drain timeout still holds the
                // runtime, and blocking here would trade a leaked registry for a hung shutdown.
                if let Ok(runtime) = entry.runtime.try_lock() {
                    manager.detach_registry(&runtime.tool_registry).await;
                }
            }
        }
        tracing::info!(
            "released {} session(s) the client did not close",
            abandoned.len()
        );
    }
    drop(abandoned);

    Ok(())
}

/// How long to wait for a client to answer a permission prompt before denying it.
///
/// Generous, because the thing being waited on is a human reading a prompt and deciding: this is a
/// backstop against a client that will never answer at all, not a deadline on the user. A prompt
/// still on screen after this long has been abandoned, and the turn holding a runtime mutex open
/// for it blocks `session/close` and `session/set_mode` behind it.
const PERMISSION_PROMPT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30 * 60);

/// How long an ACP session may sit untouched before it is released.
///
/// Matches `[serve].idle_timeout`'s default. Not configurable, deliberately: a day is long past the
/// point where an editor still means to use a session, and a knob here would be a setting nobody
/// sets for a mechanism nobody should notice.
const ACP_SESSION_IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);

/// How often the idle sweep runs. Matches `[serve].gc_scan_interval`'s default.
const ACP_IDLE_SCAN_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5 * 60);

/// Release sessions the editor opened and stopped using.
///
/// `session/close` is an *optional* ACP capability, so an editor is entitled never to send one, and
/// several do not. Each session it opens holds an `Agent`, a `ToolRegistry` the MCP manager keeps a
/// strong clone of, and an open file lock; over a long editing session that is a descriptor per
/// file the user glanced at, none of them released, and the session cannot be reopened from
/// anywhere else meanwhile.
///
/// Only the in-memory entry goes. The row stays, so `session/load` brings the conversation back
/// exactly as it does for a session from a previous run -- which is the same trade `meka serve`
/// makes with `delete_on_idle = false`.
fn spawn_idle_session_sweep(state: Arc<ServerState>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(ACP_IDLE_SCAN_INTERVAL);
        ticker.tick().await;
        loop {
            ticker.tick().await;
            // Collected under the read lock and evicted under the write lock, with the liveness
            // check inside `is_idle` rather than out here: a turn that starts between the two
            // re-takes the runtime mutex, and the second `is_idle` below sees it.
            let candidates: Vec<String> = {
                let sessions = state.sessions.read().await;
                sessions
                    .iter()
                    .filter(|(_, entry)| entry.is_idle(ACP_SESSION_IDLE_TIMEOUT))
                    .map(|(id, _)| id.clone())
                    .collect()
            };
            for id in candidates {
                let evicted = {
                    let mut sessions = state.sessions.write().await;
                    match sessions.get(&id) {
                        Some(entry) if entry.is_idle(ACP_SESSION_IDLE_TIMEOUT) => {
                            sessions.remove(&id)
                        }
                        _ => None,
                    }
                };
                let Some(entry) = evicted else { continue };
                // Same teardown `session/close` does, and for the same reason: without it the
                // manager keeps fanning `tools/list_changed` out to a registry nobody reads.
                let registry = {
                    let runtime = entry.runtime.lock().await;
                    runtime.tool_registry.clone()
                };
                if let Some(manager) = &state.shared.mcp_manager {
                    manager.detach_registry(&registry).await;
                }
                tracing::info!(
                    "released idle ACP session {} after {:?}; `session/load` reopens it",
                    id,
                    ACP_SESSION_IDLE_TIMEOUT
                );
                drop(entry);
            }
        }
    })
}

/// Cancel every active session's in-flight turn. Mirrors `crate::server`'s drain. Clones each token
/// out before any `await` so no lock guard is held across an await point.
async fn drain_acp_sessions(state: &ServerState) {
    let tokens: Vec<CancellationToken> = {
        let sessions = state.sessions.read().await;
        sessions
            .values()
            .map(|entry| {
                entry
                    .cancellation
                    .read()
                    .map(|guard| guard.clone())
                    .unwrap_or_else(|poisoned| poisoned.into_inner().clone())
            })
            .collect()
    };
    for token in tokens {
        token.cancel();
    }
}

/// Resolve once no session is running a turn. The prompt handler holds `entry.runtime`'s lock for
/// the whole turn, so a successful `try_lock` on every session means all turns have unwound.
async fn wait_for_sessions_idle(state: &ServerState) {
    loop {
        let all_idle = {
            let sessions = state.sessions.read().await;
            sessions
                .values()
                .all(|entry| entry.runtime.try_lock().is_ok())
        };
        if all_idle {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
}

/// Wait for a cross-platform termination signal: SIGTERM or Ctrl-C on unix, Ctrl-C elsewhere.
/// Mirrors `crate::server`'s `shutdown_signal`.
async fn acp_shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut terminate) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = terminate.recv() => {}
                }
            }
            Err(error) => {
                tracing::warn!(
                    "failed to install SIGTERM handler: {}; relying on Ctrl+C only",
                    error
                );
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

/// Run meka as an ACP agent over stdio. Returns (and the process then exits) when the client
/// disconnects (stdin EOF) or a termination signal arrives.
pub async fn run_acp(
    config: ResolvedConfig,
    session_manager: SessionManager,
    mcp_manager: Option<Arc<mcp::McpClientManager>>,
    mcp_context: Arc<mcp::McpClientContext>,
) -> anyhow::Result<()> {
    // Capture the default profile's vision flag before `config` is moved into `build_shared_deps`.
    // It gates the advertised `image` prompt capability, which `initialize` has to answer before
    // any session exists. Whether a given `session/prompt` admits an image is per session and comes
    // off that session's row; this is only what a prompt naming an id that is not a UUID falls back
    // to.
    let vision = config.vision;

    // Build process-wide shared deps once. Sessions hold an `Arc<SharedDeps>` and read fields by
    // reference; no work happens here that needs to be re-run per session.
    let shared = Arc::new(
        super::build_shared_deps(config, session_manager, mcp_manager, mcp_context).await?,
    );

    // Test-only: hand the registry a scripted provider to return for every profile. Only compiled
    // in debug builds. Installed rather than swapped into a rebuilt `SharedDeps`, so a harness
    // driving sessions on different profiles gets the script for all of them.
    #[cfg(debug_assertions)]
    if std::env::var("MEKA_MOCK_PROVIDER").as_deref() == Ok("1") {
        let rounds = crate::provider::mock::load_script_from_env()?.unwrap_or_default();
        shared.providers.install_scripted(Arc::new(
            crate::provider::mock::MockProvider::from_rounds(rounds),
        ));
        tracing::info!("MEKA_MOCK_PROVIDER=1: using scripted mock provider");
    }

    let client_state = SharedClientState::default();
    let transport_dead = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let state = Arc::new(ServerState {
        shared: Arc::clone(&shared),
        client_state: client_state.clone(),
        sessions: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
        transport_dead,
        vision,
    });

    // Fires jobs for whichever sessions the editor currently has open; see `acp::schedule`. Aborted
    // on the way out so a scheduled turn cannot outlive the connection it would report to.
    let scheduler_handle = schedule::spawn(Arc::clone(&state));
    let _scheduler_guard = AbortOnDrop(scheduler_handle);

    // Reports background-task outcomes into whichever sessions the editor has open, on the same
    // terms and with the same abort-on-drop lifetime.
    let background_handle = schedule::spawn_background_poller(Arc::clone(&state));
    let _background_guard = AbortOnDrop(background_handle);

    // Releases sessions the editor opened and never closed; see `spawn_idle_session_sweep`.
    let idle_handle = spawn_idle_session_sweep(Arc::clone(&state));
    let _idle_guard = AbortOnDrop(idle_handle);

    // Observe stdin EOF so the connection shuts down when the client disconnects (or the parent
    // dies). The connection future does not resolve on idle EOF by itself, so wrap the incoming
    // side; `acp_run_until_disconnect` (the `connect_with` closure below) waits on this token.
    // tokio stdio + `tokio_util::compat` provide the `futures::io` byte streams the transport wants
    // without pulling in the `blocking` crate.
    let stdin_eof = CancellationToken::new();
    let transport = ByteStreams::new(tokio::io::stdout().compat_write(), EofSignalingRead {
        inner: tokio::io::stdin().compat(),
        eof: stdin_eof.clone(),
    });

    let acp_result = AcpAgentRole
        .builder()
        .name("meka")
        .on_receive_request(
            {
                let client_state = client_state.clone();
                async move |req: InitializeRequest, responder, _cx| {
                    // Stash the client's advertised capabilities (so `AcpFrontend`'s delegate_*
                    // methods can gate on them) and the client's self-identifying `Implementation`
                    // (logged here, available for diagnostics elsewhere). Both are small clones.
                    tracing::info!(
                        "ACP client connected: {}",
                        describe_client(req.client_info.as_ref())
                    );
                    client_state.record_initialize(
                        req.client_capabilities.clone(),
                        req.client_info.clone(),
                    );

                    // Advertise the optional session methods. Each marker is an empty struct;
                    // presence signals support.
                    let session_caps = SessionCapabilities::new()
                        .additional_directories(Some(
                            SessionAdditionalDirectoriesCapabilities::new(),
                        ))
                        .list(Some(SessionListCapabilities::new()))
                        .resume(Some(SessionResumeCapabilities::new()))
                        .fork(Some(SessionForkCapabilities::new()))
                        .close(Some(SessionCloseCapabilities::new()));
                    // meka accepts `text`, `resource_link`, and embedded `resource` (@-mention)
                    // blocks in `session/prompt`, so `embedded_context` is advertised true. `image`
                    // follows the active profile's `vision` flag (default true; set false for a
                    // text-only model). `audio` stays false. Each field is set explicitly so the
                    // contract is visible in the initialize response and a future SDK default
                    // change can't quietly flip it.
                    //
                    // `mcp_capabilities` is intentionally omitted:
                    // meka sources MCP servers from its own config
                    // file and does not yet connect to servers passed
                    // through `session/new`'s `mcpServers` array.
                    // Advertising `{ http: true, sse: true }` while
                    // ignoring client-provided servers was misleading;
                    // the marker is dropped until client-MCP
                    // support lands.
                    let capabilities = AgentCapabilities::new()
                        .load_session(true)
                        .session_capabilities(session_caps)
                        .prompt_capabilities(
                            PromptCapabilities::new()
                                .image(vision)
                                .audio(false)
                                .embedded_context(true),
                        );
                    // Reject the V0 sentinel explicitly. The schema uses V0 as the "couldn't parse
                    // the requested version" fallback; a clamped `min(V0, LATEST)` would silently
                    // echo it back and let the handshake proceed against a malformed input.
                    if req.protocol_version == agent_client_protocol::schema::ProtocolVersion::V0 {
                        return responder.respond_with_error(invalid_params_error(
                            "protocolVersion 0 is the schema's parse-failure sentinel; \
                             specify a supported version",
                        ));
                    }
                    // Negotiate the protocol version per the ACP spec:
                    // respond with the requested version if we
                    // support it, otherwise pin to the latest stable
                    // version we know about. A naive echo lets a
                    // future client think we support a version we
                    // haven't shipped yet.
                    let negotiated = std::cmp::min(
                        req.protocol_version,
                        agent_client_protocol::schema::ProtocolVersion::LATEST,
                    );
                    let response = InitializeResponse::new(negotiated)
                        .agent_capabilities(capabilities)
                        .agent_info(Implementation::new("meka", env!("CARGO_PKG_VERSION")));
                    responder.respond(response)
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |req: NewSessionRequest, responder, cx: ConnectionTo<Client>| {
                    if !req.cwd.is_absolute() {
                        return responder.respond_with_error(invalid_params_error(format!(
                            "cwd must be an absolute path; got `{}`",
                            req.cwd.display()
                        )));
                    }
                    if let Err(error) = validate_additional_roots(&req.additional_directories) {
                        return responder.respond_with_error(error);
                    }
                    // A new session runs on the host's default. `session/load` reads the one the
                    // session recorded instead, which is what stops a resume moving the
                    // conversation to another provider.
                    let profile = state.shared.default_profile.clone();
                    // Created and locked in one step, the lock taken *before* the row exists: a
                    // row committed ahead of its lock is one `meka session delete --all` can
                    // enumerate and sweep out from under this handler. See
                    // `SessionManager::create_session_locked`.
                    let (session_uuid, session_lock) = match state
                        .shared
                        .session_manager
                        .create_session_locked(Some(req.cwd.clone()), None, None, None, profile)
                        .await
                    {
                        Ok((created, lock)) => (created.id, lock),
                        Err(error) => {
                            return responder.respond_with_error(
                                agent_client_protocol::util::internal_error(format!(
                                    "failed to create meka session: {}",
                                    error
                                )),
                            );
                        }
                    };
                    // Persist the roots so `session/list` can report the workspace shape this
                    // session was opened with. Non-fatal: a session that runs with the right roots
                    // but forgets them across a restart is far better than refusing to start.
                    if let Err(error) = state
                        .shared
                        .session_manager
                        .update_session_roots(session_uuid, &req.additional_directories)
                        .await
                    {
                        tracing::warn!(
                            "session/new: failed to persist additional roots: {}",
                            error
                        );
                    }

                    // The lock taken above, which a second `meka acp` process (or a REPL) needs to
                    // be unable to take. `None` means the claim could not be made at all -- an
                    // unwritable lock directory -- and an editor session that cannot be held alone
                    // is one this host must not open.
                    let session_lock = match session_lock {
                        Ok(lock) => Arc::new(lock),
                        Err(error) => {
                            return responder.respond_with_error(
                                agent_client_protocol::util::internal_error(format!(
                                    "failed to lock session: {}",
                                    error
                                )),
                            );
                        }
                    };
                    let session_id_str = session_uuid.to_string();
                    let session_id: SessionId = session_id_str.clone().into();

                    let runtime = match build_session_runtime(
                        &state.shared,
                        &state.client_state,
                        &state.transport_dead,
                        cx.clone(),
                        session_id.clone(),
                        session_id_str.clone(),
                        session_uuid,
                        req.cwd.clone(),
                        req.additional_directories.clone(),
                        Conversation::new(),
                    )
                    .await
                    {
                        Ok(runtime) => runtime,
                        Err(error) => {
                            return responder.respond_with_error(
                                agent_client_protocol::util::internal_error(format!(
                                    "failed to build session runtime: {}",
                                    error
                                )),
                            );
                        }
                    };

                    let permission = runtime.permission.clone();
                    let frontend = Arc::clone(&runtime.frontend);
                    let cancellation = runtime.frontend.cancellation_cell();
                    let entry = SessionEntry {
                        runtime: Arc::new(Mutex::new(runtime)),
                        cancellation,
                        cancel_pending: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                        title_sent: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                        permission: permission.clone(),
                        frontend,
                        session_lock,
                        last_activity: Arc::new(std::sync::RwLock::new(std::time::Instant::now())),
                    };
                    state.sessions.write().await.insert(session_id_str, entry);

                    if !req.mcp_servers.is_empty() {
                        tracing::warn!(
                            "session/new: client provided {} mcpServers, \
                             ignored (config-driven MCP servers are still \
                             active)",
                            req.mcp_servers.len(),
                        );
                    }

                    // Push the initial skill palette + the configured mode picker so the editor's
                    // UI is populated before the user types their first prompt.
                    let modes = build_mode_state(&permission);
                    let config_options =
                        build_config_options(&state.shared, &permission, Some(session_uuid)).await;
                    emit_available_commands(&cx, &session_id, &state.shared.skills).await;

                    responder.respond(
                        NewSessionResponse::new(session_id)
                            .modes(modes)
                            .config_options(config_options),
                    )
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |req: PromptRequest, responder, cx: ConnectionTo<Client>| {
                    let state_for_spawn = Arc::clone(&state);
                    cx.spawn(
                        async move { run_prompt_turn(state_for_spawn, req, responder).await },
                    )?;
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |req: LoadSessionRequest, responder, cx: ConnectionTo<Client>| {
                    handle_load_session(Arc::clone(&state), req, responder, cx).await
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |req: ListSessionsRequest, responder, _cx: ConnectionTo<Client>| {
                    handle_list_sessions(Arc::clone(&state), req, responder).await
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |req: ResumeSessionRequest, responder, cx: ConnectionTo<Client>| {
                    handle_resume_session(Arc::clone(&state), req, responder, cx).await
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |req: ForkSessionRequest, responder, cx: ConnectionTo<Client>| {
                    handle_fork_session(Arc::clone(&state), req, responder, cx).await
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |req: CloseSessionRequest, responder, cx: ConnectionTo<Client>| {
                    // Spawned, exactly like the prompt handler above, because this one waits on a
                    // lock an in-flight turn holds.
                    //
                    // Handler callbacks run on the SDK's dispatch loop, and that loop is also what
                    // routes *responses to meka's own outgoing requests*. A turn blocked on
                    // `fs/read_text_file` cannot release the runtime mutex until its response
                    // arrives, and the response cannot arrive while the loop is parked inside this
                    // handler waiting for that same mutex. Running inline deadlocked every session
                    // in the process, recoverable only by killing the client.
                    let state_for_spawn = Arc::clone(&state);
                    cx.spawn(async move {
                        handle_close_session(state_for_spawn, req, responder).await
                    })?;
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |req: SetSessionModeRequest, responder, _cx: ConnectionTo<Client>| {
                    handle_set_session_mode(Arc::clone(&state), req, responder).await
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |req: SetSessionConfigOptionRequest,
                            responder,
                            cx: ConnectionTo<Client>| {
                    // Spawned rather than run inline for the reason `session/close` spells out:
                    // this one reaches the session runtime, and parking the dispatch loop on
                    // anything a turn holds deadlocks every session in the process.
                    let state_for_spawn = Arc::clone(&state);
                    cx.spawn(async move {
                        handle_set_session_config_option(state_for_spawn, req, responder).await
                    })?;
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_notification(
            {
                let state = Arc::clone(&state);
                async move |notif: CancelNotification, _cx: ConnectionTo<Client>| {
                    // Cancel fires through the sibling `cancellation` cell on the `SessionEntry`;
                    // we never touch the per-session runtime mutex, which the prompt handler
                    // holds for the duration of the turn.
                    //
                    // We also set `cancel_pending`: if the cancel arrives between turns (the cell
                    // still holds a stale token from the previous turn, which is now a no-op), the
                    // next prompt handler will check this flag right after installing its fresh
                    // token and cancel it immediately. Without the latch, the cancel signal is
                    // lost.
                    let entry = {
                        let sessions = state.sessions.read().await;
                        sessions.get(notif.session_id.0.as_ref()).cloned()
                    };
                    if let Some(entry) = entry {
                        entry
                            .cancel_pending
                            .store(true, std::sync::atomic::Ordering::SeqCst);
                        let token = entry
                            .cancellation
                            .read()
                            .map(|guard| guard.clone())
                            .unwrap_or_else(|poisoned| poisoned.into_inner().clone());
                        token.cancel();
                    }
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(transport, {
            let state = Arc::clone(&state);
            async move |_cx: ConnectionTo<Client>| acp_run_until_disconnect(state, stdin_eof).await
        })
        .await;

    // The connection has unwound, so nothing will issue another tool call. An editor that quits
    // takes meka's stdio with it but not the grandchildren meka spawned, which is what this closes.
    if let Some(manager) = &state.shared.mcp_manager {
        manager.shutdown_within(crate::mcp::SHUTDOWN_BUDGET).await;
    }

    acp_result.map_err(|error| anyhow::anyhow!("ACP server error: {}", error))
}

/// Body of the `session/prompt` spawn. Extracted so the closure stays thin. Owns `responder` and
/// replies exactly once.
///
/// Lock ordering: take the outer `sessions` read lock briefly to clone the per-session
/// `Arc<Mutex<SessionRuntime>>`, drop it, then hold *only* the per-session mutex for the duration
/// of the turn. Cancel and other sessions remain unblocked.
async fn run_prompt_turn(
    state: Arc<ServerState>,
    req: PromptRequest,
    responder: agent_client_protocol::Responder<PromptResponse>,
) -> Result<(), agent_client_protocol::Error> {
    // This session's profile decides whether an image block is admissible, not the connection's:
    // the `image` prompt capability is answered at `initialize`, before any session exists, but
    // what a session can actually read follows the provider it recorded.
    //
    // An id that is not a uuid falls back to the advertised flag. Every session key in the map is
    // `session_uuid.to_string()`, so such an id names no session and the turn is refused either
    // way; the fallback only picks which refusal the client is given, never whether an attachment
    // is admitted.
    //
    // Read before the runtime mutex, and so before `apply_recorded_binding` puts the turn on the
    // profile the row names. Both read the same row, so a `session/set_config_option` landing
    // between them could admit an image against the outgoing profile's flag. That window is one
    // request wide and strictly narrower than the cached flag this replaced, which was stale from
    // the switch until the following turn.
    let session_vision = match uuid::Uuid::parse_str(req.session_id.0.as_ref()) {
        Ok(session_uuid) => session_accepts_images(&state, session_uuid).await,
        Err(_) => state.vision,
    };
    // Accept `text` + `resource_link` (the ACP baseline) + embedded `resource` and, when the
    // profile has vision enabled, `image`. Other content variants get rejected below.
    let mut prompt_text = String::new();
    let mut images: Vec<crate::provider::ImageSource> = Vec::new();
    for block in &req.prompt {
        match block {
            ContentBlock::Text(text) => {
                push_prompt_block(&mut prompt_text, &text.text);
            }
            // `ResourceLink` is part of the ACP baseline that every agent MUST support (alongside
            // `Text`). meka doesn't fetch the resource server-side; the model sees the reference as
            // a structured tag carrying the link's name, uri, and (optional) description so it can
            // decide what to do with it.
            ContentBlock::ResourceLink(link) => {
                let mut tag =
                    format!("<resource_link name=\"{}\" uri=\"{}\"", link.name, link.uri,);
                push_mime_attr(&mut tag, &link.mime_type);
                tag.push('>');
                if let Some(description) = &link.description {
                    tag.push_str(description);
                }
                tag.push_str("</resource_link>");
                push_prompt_block(&mut prompt_text, &tag);
            }
            // `Resource` carries an @-mention's inlined contents (the `embedded_context`
            // capability). meka surfaces it to the model as a `<resource>` tag rather than fetching
            // anything server-side, mirroring `ResourceLink`.
            ContentBlock::Resource(embedded) => {
                push_prompt_block(&mut prompt_text, &format_embedded_resource(embedded));
            }
            // `Image` is accepted only when the active profile advertised the `image` capability
            // (vision on). Normalize the payload through the shared image pipeline so the size cap
            // and format conversion match tool-result images.
            ContentBlock::Image(image) if session_vision => match decode_acp_image(image).await {
                Ok(source) => images.push(source),
                Err(message) => {
                    return responder.respond_with_error(invalid_params_error(format!(
                        "invalid image content block: {}",
                        message
                    )));
                }
            },
            _ => {
                return responder.respond_with_error(invalid_params_error(
                    "meka acp accepts `text`, `resource_link`, `resource`, and (when the \
                     profile has vision enabled) `image` content blocks in `prompt`; `audio` is \
                     not supported",
                ));
            }
        }
    }

    // Same rejection the HTTP surface applies at `handlers::turn`: a prompt of nothing but
    // whitespace costs a provider round-trip to produce nothing, and a client that dropped its
    // content on the floor should hear about it rather than get an empty answer back.
    if prompt_text.trim().is_empty() && images.is_empty() {
        return responder.respond_with_error(invalid_params_error(
            "`prompt` must contain non-empty text, or at least one image",
        ));
    }

    // Look up the target session by id under the outer read lock, clone the entry (cheap, two
    // `Arc`s), drop the outer guard. From here on, only the per-session runtime mutex is held; the
    // sibling cancellation cell is accessible to the cancel handler throughout the turn.
    let session_id_str = req.session_id.0.as_ref().to_string();
    let entry = {
        let sessions = state.sessions.read().await;
        match sessions.get(&session_id_str) {
            Some(entry) => {
                entry.touch();
                entry.clone()
            }
            None => {
                return responder.respond_with_error(invalid_params_error(format!(
                    "unknown sessionId: {}",
                    session_id_str
                )));
            }
        }
    };

    // Acquire the runtime mutex non-blocking. If another prompt is already in flight for this
    // session, reject explicitly: ACP models one prompt at a time per session and silent queueing
    // also enables a race against the sibling cancellation cell (the second prompt would overwrite
    // the first's token before the first finishes, so `session/cancel` would target the wrong
    // turn). The lock guard is held for the entire turn so the token written below cannot be
    // overwritten by a sibling request, and per-session pre-work serialises naturally.
    let mut runtime = match entry.runtime.try_lock() {
        Ok(guard) => guard,
        Err(_) => {
            return responder.respond_with_error(invalid_params_error(
                "session already has a prompt in flight",
            ));
        }
    };

    // Under the lock and before the first round, so a switch made while the previous turn held this
    // mutex takes effect on this one. Refused rather than run on the old profile: the client asked
    // for a specific account, and answering from another one silently is the failure the whole
    // per-session binding exists to prevent.
    if let Err(error) = apply_recorded_binding(&state, &mut runtime).await {
        return responder.respond_with_error(invalid_params_error(format!(
            "cannot run this turn on the provider profile this session is recorded against: {}",
            error
        )));
    }

    // Install a fresh cancellation token inside the locked scope so the cancel handler (which reads
    // the sibling cell) always sees the token for the turn currently using the runtime.
    let cancellation = CancellationToken::new();
    {
        let mut guard = entry
            .cancellation
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *guard = cancellation.clone();
    }

    // Close the between-turns race: if a `session/cancel` arrived after the previous turn finished
    // but before we installed this turn's token, the cancel handler set `cancel_pending` and fired
    // the now-dead previous token. Apply the latched signal to the freshly installed token so the
    // spec-mandated cancel isn't lost. `swap` provides the read-and-clear in one step; SeqCst pairs
    // with the same ordering in the cancel handler.
    if entry
        .cancel_pending
        .swap(false, std::sync::atomic::Ordering::SeqCst)
    {
        cancellation.cancel();
    }

    // Refresh the slash-command palette before the prompt body resolves. This uses the per-session
    // frontend so the notification routes to the right ACP connection.
    let frontend = Arc::clone(&runtime.frontend);
    emit_available_commands(
        &frontend.connection,
        &frontend.session_id,
        &state.shared.skills,
    )
    .await;

    // Local slash commands (`/status`, `/mcp`, `/usage`) render text and end the turn with no model
    // call. Checked before skill resolution so they aren't misread as unknown skills.
    if let Some(output) = try_local_command(&prompt_text, &runtime, &state.shared).await {
        // `agent_message_chunk` is rendered as Markdown, where a bare newline is a soft break (it
        // collapses to a space) and small indents are stripped. Wrap the preformatted table in a
        // fenced code block so the column alignment and line breaks survive, matching how
        // `execute_command` output is rendered.
        let body = format!("```\n{output}\n```");
        send_session_update(
            &frontend.connection,
            &frontend.session_id,
            SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
                agent_client_protocol::schema::v1::TextContent::new(body),
            ))),
        );
        // Reported here too, even though a local command consumes no tokens: `usage` is
        // session-cumulative, so omitting it would make a client's running total flicker away on
        // every `/status` turn.
        return responder.respond(
            PromptResponse::new(StopReason::EndTurn).usage(session_usage(&runtime.agent)),
        );
    }

    let original_prompt_text = prompt_text.clone();
    let prompt_text = match slash_to_prompt_text(prompt_text, &state.shared.skills).await {
        Ok(text) => text,
        Err(SlashInvocationError::SkillNotFound(name)) => {
            // `slash_to_prompt_text` only returns `SkillNotFound` for strings whose first token is
            // a syntactically-valid skill name. That's deliberately a narrow filter, but it still
            // false-positives on pasted text like `/usr local lib` (parses as name=`usr`,
            // extra=`local lib`). Treat "no such skill" as "this wasn't a skill invocation after
            // all" and feed the original text to the model. It can respond with "I don't know that
            // command" if the user really meant `/<name>`. The alternative (hard-error) breaks
            // paste UX for any string starting with `/word`.
            tracing::debug!(
                "session/prompt: '/{}' didn't match a registered skill; passing through",
                name,
            );
            original_prompt_text
        }
        Err(error @ SlashInvocationError::SkillLoadFailed { .. }) => {
            // The skill name was valid and matched an installed skill; the failure is a server-side
            // problem reading the body (disk I/O, permission, etc.). JSON-RPC `InternalError` is
            // the correct classification; `InvalidParams` would mislead the client into thinking
            // the user's request was malformed.
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                error.to_string(),
            ));
        }
    };

    let SessionRuntime {
        agent,
        session_uuid,
        messages,
        ..
    } = &mut *runtime;
    // ACP sessions always have a UUID pre-allocated at `session/new`, so `run_turn` never mutates
    // this `Option`. Pass it through anyway for API compatibility with the REPL path that does
    // lazy-create sessions on first prompt.
    let mut session_uuid_opt = Some(*session_uuid);
    // Clone the cancellation token so we can probe `is_cancelled()` after the call returns. The
    // spec mandates that any cancel arriving during a turn must surface as `StopReason::Cancelled`,
    // even when the cancellation manifests as a provider / tool error rather than the clean
    // `MekaError::Interrupted` path.
    let cancel_probe = cancellation.clone();
    let result = agent
        .run_turn(
            &mut session_uuid_opt,
            messages,
            prompt_text,
            images,
            cancellation,
        )
        .await;

    let stop_reason = match result {
        Ok(crate::agent::TurnOutcome::EndTurn) => StopReason::EndTurn,
        Ok(crate::agent::TurnOutcome::MaxTokens) => StopReason::MaxTokens,
        Ok(crate::agent::TurnOutcome::Refusal(_)) => StopReason::Refusal,
        Err(MekaError::Interrupted) => StopReason::Cancelled,
        Err(error) => {
            if cancel_probe.is_cancelled() {
                StopReason::Cancelled
            } else {
                return responder.respond_with_error(agent_client_protocol::util::internal_error(
                    format!("meka turn failed: {}", error),
                ));
            }
        }
    };

    // The first user message defines the session title; push it once now that the turn has run and
    // that message is in the conversation.
    maybe_emit_session_title(
        &frontend.connection,
        &frontend.session_id,
        &entry.title_sent,
        messages,
    );

    responder.respond(PromptResponse::new(stop_reason).usage(session_usage(agent)))
}

/// Session-cumulative token counts for `session/prompt`'s response. Complements the per-turn
/// `usage_update` notification, which carries the context gauge rather than these totals.
fn session_usage(agent: &Agent) -> Usage {
    let snapshot = agent.session_stats_snapshot();
    let mut usage = Usage::new(
        snapshot
            .total_input_tokens()
            .saturating_add(snapshot.output_tokens),
        snapshot.input_tokens,
        snapshot.output_tokens,
    );
    usage.cached_read_tokens = Some(snapshot.cache_read_input_tokens);
    usage.cached_write_tokens = Some(snapshot.cache_creation_input_tokens);
    // `thought_tokens` is left at its `None` default: meka doesn't meter reasoning separately from
    // output, and reporting a made-up split would be worse than reporting none.
    usage
}

/// `session/load`: reopen a previously persisted session and add it to the active sessions map.
/// Replays the persisted history as `session/update` notifications so the client's UI rebuilds the
/// conversation before the response goes out.
async fn handle_load_session(
    state: Arc<ServerState>,
    req: LoadSessionRequest,
    responder: agent_client_protocol::Responder<LoadSessionResponse>,
    cx: ConnectionTo<Client>,
) -> Result<(), agent_client_protocol::Error> {
    let session_id_str = req.session_id.0.as_ref().to_string();
    let session_uuid = match uuid::Uuid::parse_str(&session_id_str) {
        Ok(uuid) => uuid,
        Err(_) => {
            return responder.respond_with_error(invalid_params_error(format!(
                "malformed sessionId: {}",
                session_id_str
            )));
        }
    };

    // Refuse if a session with the same id is already loaded. Collisions between different
    // connections aren't possible (one process serves one ACP client) but a re-load of the same
    // session would discard in-flight state.
    if state.sessions.read().await.contains_key(&session_id_str) {
        return responder.respond_with_error(invalid_params_error(
            "session is already loaded; call session/close first",
        ));
    }

    if !req.cwd.is_absolute() {
        return responder.respond_with_error(invalid_params_error(format!(
            "cwd must be an absolute path; got `{}`",
            req.cwd.display()
        )));
    }
    if let Err(error) = validate_additional_roots(&req.additional_directories) {
        return responder.respond_with_error(error);
    }

    let summary = match state
        .shared
        .session_manager
        .session_info(session_uuid)
        .await
    {
        Ok(Some(summary)) => summary,
        Ok(None) => {
            return responder.respond_with_error(invalid_params_error(format!(
                "unknown sessionId: {}",
                session_uuid
            )));
        }
        Err(error) => {
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to look up session: {}", error),
            ));
        }
    };

    // Before the lock, and before the three writes below. `build_session_runtime` refuses a
    // sub-agent at the end of this handler, but by then it has taken the worker's file lock,
    // rewritten its `cwd` and replaced its `additional_roots_json` -- durable writes to a session
    // the caller may not drive, followed by an *internal* error for something the caller got wrong.
    // `cwd` is the writable boundary at `workspace` and the directory a scheduled gate is
    // re-checked in, so moving it is not cosmetic; the test below proves it moves without this.
    //
    // The `claim_session` call in between is *not* part of the harm, though it looks like it: its
    // sweep is keyed on this id, and a worker has no `background_tasks` rows because
    // `Agent::new_subagent` never enables them.
    //
    // Both load doors, not one. `session/resume` got this first and `session/load` did not, which
    // left the guard on the method editors reach for least: every side effect above stayed
    // reachable, and the changelog and upgrade guide both said otherwise. Through the shared
    // predicate rather than `summary.parent_id`, so the doors cannot drift and an imported worker
    // with no surviving parent is refused here too.
    if let Err(error) =
        crate::refuse_a_spawned_session(&state.shared.session_manager, Some(session_uuid)).await
    {
        return responder.respond_with_error(invalid_params_error(error.to_string()));
    }

    // Take the on-disk lock now so a concurrent process can't write events while we replay history.
    let session_lock = match state.shared.session_manager.lock_session(session_uuid) {
        Ok(lock) => Arc::new(lock),
        Err(error) => {
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to lock session: {}", error),
            ));
        }
    };

    // The client's cwd wins (consistent with `session/new`'s captured cwd); update the DB row when
    // it differs so `session/list` reflects the live state.
    if summary.cwd.as_deref() != Some(req.cwd.as_path())
        && let Err(error) = state
            .shared
            .session_manager
            .update_session_cwd(session_uuid, &req.cwd)
            .await
    {
        tracing::warn!(
            "session/load: failed to update persisted cwd to {}: {}",
            req.cwd.display(),
            error,
        );
    }

    // Whatever the last owner left running is ours to retire now; see
    // `crate::background::claim_session`.
    crate::background::claim_session(&state.shared.session_manager, session_uuid).await;

    let events = match state.shared.session_manager.load_events(session_uuid).await {
        Ok(events) => events,
        Err(error) => {
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to load session events: {}", error),
            ));
        }
    };
    let mut conversation = Conversation::from_events(events);
    // Drop an orphaned `tool_use` (no following `tool_result`) before adopting the session; the
    // provider rejects orphans on the next request. Mirrors the REPL resume path.
    let dropped = conversation.sanitize_orphans();
    if !dropped.is_empty() {
        tracing::warn!(
            "dropped {} orphaned assistant message(s) with unmatched tool calls while loading session {}",
            dropped.len(),
            session_uuid,
        );
    }
    let session_id: SessionId = session_id_str.clone().into();

    // Replaces the stored list, including with empty: see `update_session_roots`.
    if let Err(error) = state
        .shared
        .session_manager
        .update_session_roots(session_uuid, &req.additional_directories)
        .await
    {
        tracing::warn!(
            "session/load: failed to persist additional roots: {}",
            error
        );
    }

    let runtime = match build_session_runtime(
        &state.shared,
        &state.client_state,
        &state.transport_dead,
        cx.clone(),
        session_id.clone(),
        session_id_str.clone(),
        session_uuid,
        req.cwd.clone(),
        req.additional_directories.clone(),
        conversation,
    )
    .await
    {
        Ok(runtime) => runtime,
        Err(error) => {
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to build session runtime: {}", error),
            ));
        }
    };

    // Replay before inserting so the client sees the rebuild stream before any new turn-related
    // update could race in.
    replay_session_updates(&cx, &session_id, &runtime.cwd, &runtime.messages);

    let permission = runtime.permission.clone();

    // Restore the level this session was last set to, not the one this process starts at.
    //
    // `build_session_runtime` seeds from `shared.config.permission`, which is right for
    // `session/new` and wrong here: the row carries what the user last chose via
    // `session/set_mode`. Leaving it out is not merely a lost preference. The scheduler's live gate
    // re-check reads the row, so a session whose row said `unrestricted` while its live cell sat at
    // config default would have its gates evaluated against authority the session is not running
    // at -- the same fail-open the re-check exists to prevent, reached from the other side.
    //
    // `try_set` validates against the enabled set, so a row naming a mode this configuration no
    // longer enables cannot escalate the session: it is refused and the default stands.
    if let Some(persisted) = crate::permission::parse_recorded_permission(
        summary.permission.as_deref(),
        &format_args!("session {}", summary.id),
    ) && let Err(disabled) = permission.try_set(persisted)
    {
        tracing::debug!(
            "session was last set to '{}', which this configuration no longer enables; keeping the \
             default",
            disabled.0
        );
    }
    let frontend = Arc::clone(&runtime.frontend);
    // History already carries the first user message, so the title is known; push it once now,
    // sharing the flag with the entry so a later prompt won't re-emit it.
    let title_sent = Arc::new(std::sync::atomic::AtomicBool::new(false));
    maybe_emit_session_title(&cx, &session_id, &title_sent, &runtime.messages);
    let cancellation = runtime.frontend.cancellation_cell();
    let entry = SessionEntry {
        runtime: Arc::new(Mutex::new(runtime)),
        cancellation,
        cancel_pending: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        title_sent,
        permission: permission.clone(),
        frontend,
        session_lock,
        last_activity: Arc::new(std::sync::RwLock::new(std::time::Instant::now())),
    };
    state.sessions.write().await.insert(session_id_str, entry);

    // Refresh the palette + advertise the current mode set: the editor was reopened, so its UI
    // starts blank.
    let modes = build_mode_state(&permission);
    let config_options = build_config_options(&state.shared, &permission, Some(session_uuid)).await;
    emit_available_commands(&cx, &session_id, &state.shared.skills).await;

    responder.respond(
        LoadSessionResponse::new()
            .modes(modes)
            .config_options(config_options),
    )
}

/// `session/list`: paginated index of persisted sessions, filtered by cwd when the client asks.
/// Sub-agent sessions are excluded; they're internal audit rows, not user-facing conversations.
async fn handle_list_sessions(
    state: Arc<ServerState>,
    req: ListSessionsRequest,
    responder: agent_client_protocol::Responder<ListSessionsResponse>,
) -> Result<(), agent_client_protocol::Error> {
    const PAGE_SIZE: u32 = 50;
    let cwd_filter = req.cwd.as_deref();
    let cursor = req.cursor.as_deref();
    let (rows, next_cursor) = match state
        .shared
        .session_manager
        .list_sessions(PAGE_SIZE, false, cwd_filter, cursor)
        .await
    {
        Ok(pair) => pair,
        Err(error) => {
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to list sessions: {}", error),
            ));
        }
    };
    // Fallback for a row carrying no `cwd`, which `meka session import` produces when the archive
    // omits it. The process cwd matches what the agent would use for relative-path resolution if
    // the client picked one of these to load. That is better than refusing to surface them.
    let fallback_cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    let sessions = rows
        .into_iter()
        .map(|summary| {
            let cwd = summary.cwd.unwrap_or_else(|| fallback_cwd.clone());
            let mut info =
                SessionInfo::new(summary.id.to_string(), cwd).updated_at(summary.updated_at);
            if !summary.additional_roots.is_empty() {
                info = info.additional_directories(summary.additional_roots);
            }
            if !summary.preview.is_empty() {
                info = info.title(summary.preview);
            }
            info
        })
        .collect::<Vec<_>>();

    let mut response = ListSessionsResponse::new(sessions);
    if let Some(token) = next_cursor {
        response = response.next_cursor(token);
    }
    responder.respond(response)
}

/// `session/resume`: adopt an existing session as active without replaying. Used when the client
/// already has the history in its UI and just wants the agent to pick up the conversation context.
async fn handle_resume_session(
    state: Arc<ServerState>,
    req: ResumeSessionRequest,
    responder: agent_client_protocol::Responder<ResumeSessionResponse>,
    cx: ConnectionTo<Client>,
) -> Result<(), agent_client_protocol::Error> {
    let session_id_str = req.session_id.0.as_ref().to_string();
    let session_uuid = match uuid::Uuid::parse_str(&session_id_str) {
        Ok(uuid) => uuid,
        Err(_) => {
            return responder.respond_with_error(invalid_params_error(format!(
                "malformed sessionId: {}",
                session_id_str
            )));
        }
    };

    if state.sessions.read().await.contains_key(&session_id_str) {
        return responder.respond_with_error(invalid_params_error(
            "session is already loaded; call session/close first",
        ));
    }

    if !req.cwd.is_absolute() {
        return responder.respond_with_error(invalid_params_error(format!(
            "cwd must be an absolute path; got `{}`",
            req.cwd.display()
        )));
    }
    if let Err(error) = validate_additional_roots(&req.additional_directories) {
        return responder.respond_with_error(error);
    }

    let summary = match state
        .shared
        .session_manager
        .session_info(session_uuid)
        .await
    {
        Ok(Some(summary)) => summary,
        Ok(None) => {
            return responder.respond_with_error(invalid_params_error(format!(
                "unknown sessionId: {}",
                session_uuid
            )));
        }
        Err(error) => {
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to look up session: {}", error),
            ));
        }
    };

    // Before the lock, and before any of the four writes below. `build_session_runtime` refuses a
    // sub-agent at the end of this handler, but by then this has taken the worker's file lock,
    // rewritten its `cwd`, retired the background work its parent left running, and replaced its
    // `additional_roots_json` -- four side effects on a session the caller may not drive, followed
    // by an *internal* error for something the caller got wrong. `session/fork` refuses up front
    // for the same reason; this is its sibling.
    //
    // Through the shared predicate rather than `summary.parent_id`, so the two doors cannot drift
    // and so an imported worker with no surviving parent is refused here too.
    if let Err(error) =
        crate::refuse_a_spawned_session(&state.shared.session_manager, Some(session_uuid)).await
    {
        return responder.respond_with_error(invalid_params_error(error.to_string()));
    }

    let session_lock = match state.shared.session_manager.lock_session(session_uuid) {
        Ok(lock) => Arc::new(lock),
        Err(error) => {
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to lock session: {}", error),
            ));
        }
    };

    if summary.cwd.as_deref() != Some(req.cwd.as_path())
        && let Err(error) = state
            .shared
            .session_manager
            .update_session_cwd(session_uuid, &req.cwd)
            .await
    {
        tracing::warn!(
            "session/resume: failed to update persisted cwd to {}: {}",
            req.cwd.display(),
            error,
        );
    }

    // Whatever the last owner left running is ours to retire now; see
    // `crate::background::claim_session`.
    crate::background::claim_session(&state.shared.session_manager, session_uuid).await;

    let events = match state.shared.session_manager.load_events(session_uuid).await {
        Ok(events) => events,
        Err(error) => {
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to load session events: {}", error),
            ));
        }
    };
    let mut conversation = Conversation::from_events(events);
    // Drop an orphaned `tool_use` (no following `tool_result`) before adopting the session; the
    // provider rejects orphans on the next request. Mirrors the REPL resume path.
    let dropped = conversation.sanitize_orphans();
    if !dropped.is_empty() {
        tracing::warn!(
            "dropped {} orphaned assistant message(s) with unmatched tool calls while resuming session {}",
            dropped.len(),
            session_uuid,
        );
    }
    let session_id: SessionId = session_id_str.clone().into();

    // Replaces the stored list, including with empty: see `update_session_roots`.
    if let Err(error) = state
        .shared
        .session_manager
        .update_session_roots(session_uuid, &req.additional_directories)
        .await
    {
        tracing::warn!(
            "session/resume: failed to persist additional roots: {}",
            error
        );
    }

    let runtime = match build_session_runtime(
        &state.shared,
        &state.client_state,
        &state.transport_dead,
        cx.clone(),
        session_id.clone(),
        session_id_str.clone(),
        session_uuid,
        req.cwd.clone(),
        req.additional_directories.clone(),
        conversation,
    )
    .await
    {
        Ok(runtime) => runtime,
        Err(error) => {
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to build session runtime: {}", error),
            ));
        }
    };

    let permission = runtime.permission.clone();

    // Restore the level this session was last set to, not the one this process starts at.
    //
    // `build_session_runtime` seeds from `shared.config.permission`, which is right for
    // `session/new` and wrong here: the row carries what the user last chose via
    // `session/set_mode`. Leaving it out is not merely a lost preference. The scheduler's live gate
    // re-check reads the row, so a session whose row said `unrestricted` while its live cell sat at
    // config default would have its gates evaluated against authority the session is not running
    // at -- the same fail-open the re-check exists to prevent, reached from the other side.
    //
    // `try_set` validates against the enabled set, so a row naming a mode this configuration no
    // longer enables cannot escalate the session: it is refused and the default stands.
    if let Some(persisted) = crate::permission::parse_recorded_permission(
        summary.permission.as_deref(),
        &format_args!("session {}", summary.id),
    ) && let Err(disabled) = permission.try_set(persisted)
    {
        tracing::debug!(
            "session was last set to '{}', which this configuration no longer enables; keeping the \
             default",
            disabled.0
        );
    }
    let frontend = Arc::clone(&runtime.frontend);
    // History already carries the first user message, so the title is known; push it once now,
    // sharing the flag with the entry so a later prompt won't re-emit it.
    let title_sent = Arc::new(std::sync::atomic::AtomicBool::new(false));
    maybe_emit_session_title(&cx, &session_id, &title_sent, &runtime.messages);
    let cancellation = runtime.frontend.cancellation_cell();
    let entry = SessionEntry {
        runtime: Arc::new(Mutex::new(runtime)),
        cancellation,
        cancel_pending: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        title_sent,
        permission: permission.clone(),
        frontend,
        session_lock,
        last_activity: Arc::new(std::sync::RwLock::new(std::time::Instant::now())),
    };
    state.sessions.write().await.insert(session_id_str, entry);

    let modes = build_mode_state(&permission);
    let config_options = build_config_options(&state.shared, &permission, Some(session_uuid)).await;
    emit_available_commands(&cx, &session_id, &state.shared.skills).await;

    responder.respond(
        ResumeSessionResponse::new()
            .modes(modes)
            .config_options(config_options),
    )
}

/// Delete a fork whose runtime could not be built, so a failed `session/fork` doesn't leave a full
/// copy of the conversation in the database under an id the client was never told.
///
/// `session/new` has the same failure shape but leaves its row behind; there the orphan is an empty
/// session, whereas a fork's is an entire transcript, and an auto-retrying client would multiply it
/// on every attempt. Best-effort: a failed cleanup is worth a warning, not a second error replacing
/// the one the client needs to see.
async fn discard_failed_fork(state: &Arc<ServerState>, session_uuid: uuid::Uuid) {
    match state
        .shared
        .session_manager
        .delete_session(session_uuid)
        .await
    {
        Ok(_) => tracing::info!("session/fork: discarded unusable fork {}", session_uuid),
        Err(error) => tracing::warn!(
            "session/fork: failed to discard unusable fork {}: {}",
            session_uuid,
            error,
        ),
    }
}

/// `session/fork`: copy an existing session's conversation into a new session and adopt the copy as
/// active. The source is left open and untouched.
///
/// **UNSTABLE** in the protocol: gated behind the SDK's `unstable_session_fork` feature and subject
/// to change.
///
/// Shaped like [`handle_resume_session`], with one difference that matters: ACP models fork as a
/// session-*creation* request, so `cwd` and `additionalDirectories` come from the request rather
/// than the source session, and may legitimately differ from it.
async fn handle_fork_session(
    state: Arc<ServerState>,
    req: ForkSessionRequest,
    responder: agent_client_protocol::Responder<ForkSessionResponse>,
    cx: ConnectionTo<Client>,
) -> Result<(), agent_client_protocol::Error> {
    let source_id_str = req.session_id.0.as_ref().to_string();
    let source_uuid = match uuid::Uuid::parse_str(&source_id_str) {
        Ok(uuid) => uuid,
        Err(_) => {
            return responder.respond_with_error(invalid_params_error(format!(
                "malformed sessionId: {}",
                source_id_str
            )));
        }
    };

    if !req.cwd.is_absolute() {
        return responder.respond_with_error(invalid_params_error(format!(
            "cwd must be an absolute path; got `{}`",
            req.cwd.display()
        )));
    }
    if let Err(error) = validate_additional_roots(&req.additional_directories) {
        return responder.respond_with_error(error);
    }

    // Before the copy, for the reason `crate::server::handlers::sessions::fork_session` refuses
    // there: a fork of a sub-agent is a sibling under the same parent, so the copy is a worker too
    // and `build_session_runtime` below refuses to build it. That refusal is correct but arrives
    // far too late to say anything useful -- it is reported as an *internal* error, for something
    // the caller got wrong, and it names the copy's id, which the client has never seen and which
    // `discard_failed_fork` has already deleted by the time it reads it.
    //
    // `spawn_terms` and not the parent link, so the same rows the builders refuse are refused
    // here. Keyed on the link alone, an imported worker fell straight through this check into the
    // failure it exists to prevent.
    match state.shared.session_manager.spawn_terms(source_uuid).await {
        Ok(Some(terms)) => {
            return responder.respond_with_error(invalid_params_error(match terms.parent {
                Some(parent) => format!(
                    "session {source_uuid} is a sub-agent of {parent}, so a copy of it is another \
                     sub-agent and there is no session to hand back. Continue the conversation \
                     with `agent_followup` from {parent}."
                ),
                None => format!(
                    "session {source_uuid} is a sub-agent whose parent is not in this store, so a \
                     copy of it is another sub-agent and there is no session to hand back."
                ),
            }));
        }
        // Not this door's refusal to make: `fork_session_locked` answers an unknown id below, with
        // the wording the rest of this handler uses.
        Ok(None) => {}
        Err(error) => {
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to read session: {}", error),
            ));
        }
    }

    // Locked before the copy's row exists. Otherwise a sweep between the two takes the copy, this
    // handler locks the vanished id successfully, `load_events` returns empty, and the editor is
    // handed a silently blank fork. See `SessionManager::fork_session_locked`.
    let (forked, forked_lock) = match state
        .shared
        .session_manager
        .fork_session_locked(source_uuid, crate::session::ForkOverrides {
            cwd: Some(req.cwd.clone()),
            // Always `Some`: per the spec an omitted or empty list means "no additional roots are
            // activated", which is an override to none rather than a request to inherit.
            additional_roots: Some(req.additional_directories.clone()),
            token_id: None,
        })
        .await
    {
        Ok(Some(pair)) => pair,
        Ok(None) => {
            return responder.respond_with_error(invalid_params_error(format!(
                "unknown sessionId: {}",
                source_uuid
            )));
        }
        Err(error) => {
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to fork session: {}", error),
            ));
        }
    };

    let session_uuid = forked.id;
    let session_id_str = session_uuid.to_string();
    let session_id: SessionId = session_id_str.clone().into();

    let session_lock = match forked_lock {
        Ok(lock) => Arc::new(lock),
        Err(error) => {
            discard_failed_fork(&state, session_uuid).await;
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to lock session: {}", error),
            ));
        }
    };

    // Whatever the last owner left running is ours to retire now; see
    // `crate::background::claim_session`.
    crate::background::claim_session(&state.shared.session_manager, session_uuid).await;

    let events = match state.shared.session_manager.load_events(session_uuid).await {
        Ok(events) => events,
        Err(error) => {
            discard_failed_fork(&state, session_uuid).await;
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to load session events: {}", error),
            ));
        }
    };
    let mut conversation = Conversation::from_events(events);
    // Same reasoning as resume: an orphaned `tool_use` copied from a source that was interrupted
    // mid-turn would make the fork's first prompt fail at the provider.
    let dropped = conversation.sanitize_orphans();
    if !dropped.is_empty() {
        tracing::warn!(
            "dropped {} orphaned assistant message(s) with unmatched tool calls while forking session {}",
            dropped.len(),
            source_uuid,
        );
    }

    let runtime = match build_session_runtime(
        &state.shared,
        &state.client_state,
        &state.transport_dead,
        cx.clone(),
        session_id.clone(),
        session_id_str.clone(),
        session_uuid,
        req.cwd.clone(),
        req.additional_directories.clone(),
        conversation,
    )
    .await
    {
        Ok(runtime) => runtime,
        Err(error) => {
            discard_failed_fork(&state, session_uuid).await;
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to build session runtime: {}", error),
            ));
        }
    };

    if !req.mcp_servers.is_empty() {
        tracing::warn!(
            "session/fork: client provided {} mcpServers, ignored (config-driven MCP servers are \
             still active)",
            req.mcp_servers.len(),
        );
    }

    let permission = runtime.permission.clone();
    let frontend = Arc::clone(&runtime.frontend);
    // The copied history already carries the first user message, so the title is known now.
    let title_sent = Arc::new(std::sync::atomic::AtomicBool::new(false));
    maybe_emit_session_title(&cx, &session_id, &title_sent, &runtime.messages);
    let cancellation = runtime.frontend.cancellation_cell();
    let entry = SessionEntry {
        runtime: Arc::new(Mutex::new(runtime)),
        cancellation,
        cancel_pending: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        title_sent,
        permission: permission.clone(),
        frontend,
        session_lock,
        last_activity: Arc::new(std::sync::RwLock::new(std::time::Instant::now())),
    };
    state.sessions.write().await.insert(session_id_str, entry);

    tracing::info!("session/fork: {} forked into {}", source_uuid, session_uuid);

    let modes = build_mode_state(&permission);
    let config_options = build_config_options(&state.shared, &permission, Some(session_uuid)).await;
    emit_available_commands(&cx, &session_id, &state.shared.skills).await;

    responder.respond(
        ForkSessionResponse::new(session_id)
            .modes(modes)
            .config_options(config_options),
    )
}

/// `session/close`: remove a session from the active map. Cancels any in-flight prompt for that
/// session before removing it from the map so the agent loop unwinds. Detaches the session's tool
/// registry from the MCP manager so live `tools/list_changed` updates stop targeting it.
async fn handle_close_session(
    state: Arc<ServerState>,
    req: CloseSessionRequest,
    responder: agent_client_protocol::Responder<CloseSessionResponse>,
) -> Result<(), agent_client_protocol::Error> {
    let session_id_str = req.session_id.0.as_ref().to_string();
    let removed = state.sessions.write().await.remove(&session_id_str);
    let Some(entry) = removed else {
        return responder.respond_with_error(invalid_params_error("no such session"));
    };
    // Fire cancel via the sibling cell; never blocks on the runtime mutex (which an in-flight
    // prompt may hold for the whole turn).
    let token = entry
        .cancellation
        .read()
        .map(|guard| guard.clone())
        .unwrap_or_else(|poisoned| poisoned.into_inner().clone());
    token.cancel();
    // Detach the session's tool registry from the MCP manager so tools/list_changed updates stop
    // targeting it.
    //
    // This waits on the runtime mutex, which an in-flight prompt holds for the whole turn. That is
    // safe only because the handler is `cx.spawn`ed: on the dispatch loop it would starve the very
    // response the turn is waiting for. The cancel above does not make the wait short either --
    // `read_file` and the `fs/*` delegates do not observe the token -- so this genuinely blocks
    // until the turn ends, off the loop, which is the correct place to do it.
    let registry = {
        let runtime = entry.runtime.lock().await;
        runtime.tool_registry.clone()
    };
    if let Some(manager) = &state.shared.mcp_manager {
        manager.detach_registry(&registry).await;
    }
    // The inner Arcs live until any in-flight prompt's lock guard drops; the agent loop sees the
    // cancel and returns. The map entry is gone, so further requests for this session id error.
    drop(entry);
    responder.respond(CloseSessionResponse::new())
}

/// `session/set_mode`: switch the active session to a different permission level. Validates against
/// the configured enabled set; modes outside it become JSON-RPC errors rather than silently
/// failing. On success, emit `current_mode_update` so every connected client (the picker UI)
/// reflects the new state.
async fn handle_set_session_mode(
    state: Arc<ServerState>,
    req: SetSessionModeRequest,
    responder: agent_client_protocol::Responder<SetSessionModeResponse>,
) -> Result<(), agent_client_protocol::Error> {
    let session_id_str = req.session_id.0.as_ref().to_string();
    let entry = {
        let sessions = state.sessions.read().await;
        match sessions.get(&session_id_str) {
            Some(entry) => {
                entry.touch();
                entry.clone()
            }
            None => {
                return responder.respond_with_error(invalid_params_error("no such session"));
            }
        }
    };
    let permission = match parse_mode_id(req.mode_id.0.as_ref()) {
        Some(p) => p,
        None => {
            return responder.respond_with_error(invalid_params_error(format!(
                "unknown mode id: {}",
                req.mode_id.0.as_ref()
            )));
        }
    };
    // No runtime mutex acquired: `SharedPermission` is `Arc<AtomicU8>` and the frontend cell holds
    // the connection. A user's mid-turn mode change takes effect on the next tool-call permission
    // probe without waiting for the in-flight turn to finish.
    if let Err(disabled) = entry.permission.try_set(permission) {
        return responder.respond_with_error(invalid_params_error(format!(
            "mode '{}' is not enabled in this configuration",
            disabled.0
        )));
    }
    // Persist alongside the in-memory cell, the way `PATCH /v1/sessions/{id}` already does.
    //
    // The scheduler's live gate re-check reads the session *row*, falling back to the process's
    // startup level when the column is null (see `ResolvedScheduleConfig::host_permission`). An ACP
    // session that only ever moved its in-memory cell left that column null forever, so cycling to
    // `unrestricted` in the editor and authoring a gate left the gate refused, and cycling back
    // down to `read` did not withdraw one already written. The row is also what `session/list`
    // reports, so it was misreporting the mode for the same reason.
    //
    // In-memory first and best-effort here: the mode change the user asked for has already taken
    // effect on the next tool call, and failing the whole request over a database write would be a
    // worse answer than a stale row plus a warning.
    // Parsed from the request id rather than read off `entry.runtime`: that mutex is held for the
    // whole of an in-flight prompt, and blocking the dispatch loop on it is the deadlock
    // `session/close` documents at length.
    match uuid::Uuid::parse_str(&session_id_str) {
        Ok(session_uuid) => {
            if let Err(error) = state
                .shared
                .session_manager
                .update_session_permission(session_uuid, &permission.to_string())
                .await
            {
                tracing::warn!(
                    "could not persist the new permission for session {}: {}",
                    session_uuid,
                    error
                );
            }
        }
        Err(error) => {
            tracing::warn!(
                "session id '{}' is not a UUID, so the new permission was not persisted: {}",
                session_id_str,
                error
            );
        }
    }
    // The canonical id for the mode that was actually set, not the string the client sent.
    //
    // `parse_mode_id` accepts what `--permission` accepts, so `W`, `Workspace` and `workspace` all
    // reach the same rung. Echoing the request verbatim reported a `currentMode` matching none of
    // the ids advertised in `availableModes`, which is what an editor compares against to tick the
    // right entry in its mode picker.
    send_session_update(
        &entry.frontend.connection,
        &entry.frontend.session_id,
        SessionUpdate::CurrentModeUpdate(CurrentModeUpdate::new(mode_id_for(permission))),
    );
    // The same change again for clients reading `configOptions` rather than `modes`. Both are
    // advertised, so both have to be kept current or the two pickers disagree about the level the
    // session is at.
    emit_config_options(&state, &entry, uuid::Uuid::parse_str(&session_id_str).ok()).await;
    responder.respond(SetSessionModeResponse::new())
}

/// `session/set_config_option`: the `configOptions` counterpart to `session/set_mode`, which is how
/// a client changes the session's provider profile.
///
/// The permission option does the same three things `session/set_mode` does -- set the cell, record
/// the row, push a `current_mode_update` -- so a client driving either one gets the same result and
/// both pickers agree. It does not *call* that handler: this one answers with the refreshed
/// `configOptions` list rather than pushing it, which is the response shape the method has.
async fn handle_set_session_config_option(
    state: Arc<ServerState>,
    req: SetSessionConfigOptionRequest,
    responder: agent_client_protocol::Responder<SetSessionConfigOptionResponse>,
) -> Result<(), agent_client_protocol::Error> {
    let session_id_str = req.session_id.0.as_ref().to_string();
    let entry = {
        let sessions = state.sessions.read().await;
        match sessions.get(&session_id_str) {
            Some(entry) => {
                entry.touch();
                entry.clone()
            }
            None => {
                return responder.respond_with_error(invalid_params_error("no such session"));
            }
        }
    };
    let Some(value) = req.value.as_value_id() else {
        return responder.respond_with_error(invalid_params_error(
            "both configuration options take a string value",
        ));
    };
    let value = value.0.as_ref().to_string();
    let session_uuid = uuid::Uuid::parse_str(&session_id_str).ok();

    match req.config_id.0.as_ref() {
        PERMISSION_CONFIG_ID => {
            let Some(permission) = parse_mode_id(&value) else {
                return responder.respond_with_error(invalid_params_error(format!(
                    "unknown mode id: {}",
                    value
                )));
            };
            if let Err(disabled) = entry.permission.try_set(permission) {
                return responder.respond_with_error(invalid_params_error(format!(
                    "mode '{}' is not enabled in this configuration",
                    disabled.0
                )));
            }
            // The row matters here for the same reason it does in `session/set_mode`: a scheduled
            // gate is re-checked against it, and `session/list` reports it.
            if let Some(session_uuid) = session_uuid
                && let Err(error) = state
                    .shared
                    .session_manager
                    .update_session_permission(session_uuid, &permission.to_string())
                    .await
            {
                tracing::warn!(
                    "could not persist the new permission for session {}: {}",
                    session_uuid,
                    error
                );
            }
            // `modes` is still advertised, so its picker has to hear about a change made through
            // the other one.
            send_session_update(
                &entry.frontend.connection,
                &entry.frontend.session_id,
                SessionUpdate::CurrentModeUpdate(CurrentModeUpdate::new(mode_id_for(permission))),
            );
        }
        PROVIDER_CONFIG_ID => {
            let Some(session_uuid) = session_uuid else {
                return responder.respond_with_error(invalid_params_error(
                    "this session has no row, so its provider cannot be changed",
                ));
            };
            if !state.shared.config.providers.contains_key(&value) {
                // Names the configured set, as the REPL and the HTTP API both do. A client picking
                // from `configOptions` cannot reach this, but one setting the id from a script or a
                // stale config can, and "not configured" alone gives it nothing to correct to.
                return responder.respond_with_error(invalid_params_error(format!(
                    "provider profile `{}` is not configured (configured: {})",
                    value,
                    state
                        .shared
                        .config
                        .providers
                        .keys()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                )));
            }
            // The profile as configured. Picking one through the picker moves the session to that
            // bundle entire, which is the only thing a provider selection can mean now that a
            // profile is indivisible.
            let resolved =
                match crate::provider::resolved_binding(&state.shared.providers, value.clone())
                    .await
                {
                    Ok(resolved) => resolved,
                    Err(error) => {
                        return responder.respond_with_error(invalid_params_error(format!(
                            "cannot use provider profile `{}`: {}",
                            value, error
                        )));
                    }
                };
            // Setting the option to what it already is writes nothing. A picker re-sends its
            // current value readily -- a client rebuilding its UI, a user reselecting the
            // highlighted row -- and each write bumps `updated_at`, which is what the GC scanner's
            // idle timer reads, so a client polling its own pickers could keep a session resident
            // forever. `PATCH /v1/sessions/{id}` already filters the no-op; this is the same rule.
            let recorded = match state
                .shared
                .session_manager
                .recorded_provider(session_uuid)
                .await
            {
                Ok(recorded) => recorded,
                Err(error) => {
                    return responder.respond_with_error(invalid_params_error(format!(
                        "could not read the recorded provider: {}",
                        error
                    )));
                }
            };
            // The row first, so a write that fails leaves the session on the profile it was
            // already recorded against rather than on one no later resume would resolve.
            if recorded.as_ref() != Some(&resolved.binding) {
                match state
                    .shared
                    .session_manager
                    .set_recorded_provider(session_uuid, &resolved.binding)
                    .await
                {
                    Ok(true) => {}
                    Ok(false) => {
                        return responder.respond_with_error(invalid_params_error(
                            "this session no longer exists",
                        ));
                    }
                    Err(error) => {
                        return responder.respond_with_error(invalid_params_error(format!(
                            "could not record the provider: {}",
                            error
                        )));
                    }
                }
            }
            // Unlike permission, which lives in an atomic the tool-call path reads live, the
            // provider is owned by the `Agent` behind the runtime mutex. An in-flight prompt holds
            // that mutex for the whole turn, and blocking the dispatch loop on it is the deadlock
            // `session/close` documents; `try_lock` turns the wait into an answer the client can
            // act on.
            //
            // Nothing is parked on a lost `try_lock`, and nothing needs to be: the row is already
            // written, and every turn entry point reads it through `apply_recorded_binding` before
            // its first round. A parked binding was a second carrier of a fact the row already
            // held, only one of the three entry points drained it, and it could lose to any other
            // writer of that row.
            match entry.runtime.try_lock() {
                Ok(mut runtime) => runtime.agent.set_provider(resolved),
                Err(_) => tracing::info!(
                    "session {} has a turn in flight; the provider change takes effect on the next \
                     turn",
                    session_uuid
                ),
            }
        }
        unknown => {
            return responder.respond_with_error(invalid_params_error(format!(
                "unknown configuration option: {}",
                unknown
            )));
        }
    }

    let options = build_config_options(&state.shared, &entry.permission, session_uuid).await;
    responder.respond(SetSessionConfigOptionResponse::new(options))
}

/// Build a fresh [`SessionRuntime`] from the process-wide
/// [`crate::SharedDeps`]. Called from `session/new`, `session/load`,
/// and `session/resume`. Each follows the same shape:
/// 1. Construct the per-session `AcpFrontend` bound to this connection + session id.
/// 2. Build a per-session `SharedPermission` cell seeded from config defaults.
/// 3. Build the per-session `Agent` + `ToolRegistry` via [`crate::build_session_agent`], which also
///    attaches the registry to the MCP manager.
/// 4. Bundle everything into a `SessionRuntime`.
#[allow(clippy::too_many_arguments)]
async fn build_session_runtime(
    shared: &Arc<crate::SharedDeps>,
    client_state: &SharedClientState,
    transport_dead: &Arc<std::sync::atomic::AtomicBool>,
    connection: ConnectionTo<Client>,
    session_id: SessionId,
    session_id_str: String,
    session_uuid: uuid::Uuid,
    cwd_path: PathBuf,
    additional_roots: Vec<PathBuf>,
    messages: Conversation,
) -> anyhow::Result<SessionRuntime> {
    let cwd: SharedCwd = Arc::new(std::sync::RwLock::new(cwd_path));
    // `--writable-root` is a flag on the process the editor launched, so it belongs to every
    // session that process serves, not only the REPL's. Merged into the live handle and
    // deliberately not into the persisted row: the row is what `session/load` hands back to the
    // client as its `additionalDirectories`, and reporting a folder the client never asked for
    // would misdescribe its own request to it.
    let roots: SharedRoots = Arc::new(std::sync::RwLock::new(
        additional_roots
            .into_iter()
            .chain(shared.config.writable_roots.iter().cloned())
            .collect(),
    ));
    let permission =
        SharedPermission::new(shared.config.permission, shared.config.enabled_permissions);

    // Shared with the agent (adopted inside `build_session_agent`) so the frontend can read the
    // current context occupancy when emitting `usage_update`.
    let context_tokens = Arc::new(std::sync::atomic::AtomicU64::new(0));
    // The window the same `usage_update` divides by, made here for the same reason: the frontend
    // exists before the agent and has to hold the cell the agent publishes into, rather than a copy
    // something has to remember to re-store beside every provider switch.
    let context_window = Arc::new(std::sync::atomic::AtomicU64::new(0));
    // Created here rather than beside the `SessionEntry` so the frontend and the entry share one
    // cell: the entry's cancel handler writes it, the frontend's client round-trips read it.
    let cancellation = Arc::new(std::sync::RwLock::new(CancellationToken::new()));
    let acp_frontend = Arc::new(AcpFrontend::new(
        connection,
        session_id,
        Arc::clone(&cwd),
        client_state.clone(),
        Arc::clone(transport_dead),
        Arc::clone(&context_tokens),
        Arc::clone(&context_window),
        cancellation,
    ));
    let frontend: Arc<dyn Frontend> = acp_frontend.clone();

    let (agent, tool_registry) = crate::build_session_agent(
        shared,
        // The row already exists by here, for `session/new` as well as `session/load`, so a loaded
        // session resolves the profile it recorded rather than whatever this process defaults to.
        Some(session_uuid),
        permission.clone(),
        frontend,
        Arc::clone(&cwd),
        Arc::clone(&roots),
        Arc::clone(&context_tokens),
        // ACP reports occupancy through `usage_update`, driven by the counter above; it has no
        // separate reader for the fixed overhead, so a fresh handle is all it needs.
        Arc::new(std::sync::atomic::AtomicU64::new(0)),
        context_window,
    )
    .await?;

    Ok(SessionRuntime {
        session_id_str,
        session_uuid,
        messages,
        cwd,
        permission,
        agent,
        frontend: acp_frontend,
        tool_registry,
    })
}

#[cfg(test)]
mod tests {
    use base64::Engine as _;

    use super::*;
    use crate::frontend::PermissionOutcome;

    // `AcpFrontend` itself can't be unit-tested (requires a live `ConnectionTo<Client>`);
    // per-session behaviour is covered end-to-end in `tests/acp.rs`. The pure helpers below are
    // what this unit-test module owns.

    /// A request the user has already stopped must not wait on the client's answer.
    ///
    /// Every `fs/read_text_file`, `fs/write_text_file` and elicitation is a round trip to an editor
    /// that owes no reply once the turn is cancelled, so without the race the stop button left the
    /// turn parked on a request nobody was going to answer.
    #[tokio::test]
    async fn a_cancelled_turn_abandons_a_request_the_client_has_not_answered() {
        let cancellation = CancellationToken::new();
        cancellation.cancel();

        // Bounded so a lost race fails the test rather than hanging it: the work below models a
        // client that never answers, so without the race there is nothing to wait for.
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            race_against_cancellation(
                "fs/read_text_file",
                &cancellation,
                std::future::pending::<()>(),
            ),
        )
        .await
        .expect("a cancelled turn must not wait on the client");

        let error = outcome.expect_err("a cancelled turn must not wait on the client");
        assert!(
            error.is_cancelled(),
            "the caller has to be able to tell a stop from a failure: {error}"
        );
    }

    /// The other half: an uncancelled turn must still get its answer, or the race would make every
    /// client round trip fail.
    #[tokio::test]
    async fn a_live_turn_receives_the_clients_answer() {
        let cancellation = CancellationToken::new();

        let outcome =
            race_against_cancellation("fs/read_text_file", &cancellation, async { "contents" })
                .await;

        assert_eq!(
            outcome.expect("a live turn must get its answer"),
            "contents"
        );
    }

    /// A sticky permission option must name the tool it actually covers.
    ///
    /// This ships as a `**Breaking:**` changelog line and had no test: the whole point of the
    /// relabel is that the option is keyed on the tool name and applies for the rest of the
    /// session, while the prompt beside it names one specific command. Reverting to a bare "Always
    /// allow" left every suite green, which is exactly how an affordance drifts back out of step
    /// with its semantics.
    #[test]
    fn a_sticky_permission_option_names_the_tool_it_covers() {
        assert_eq!(
            sticky_option_label("allow", "execute_command"),
            "Always allow any execute_command"
        );
        assert_eq!(
            sticky_option_label("deny", "write_file"),
            "Always deny any write_file"
        );
        for verb in ["allow", "deny"] {
            assert!(
                sticky_option_label(verb, "execute_command").contains("execute_command"),
                "dropping the tool name makes the option read as approving the one call on screen"
            );
        }
    }

    /// A session running a turn is never idle, whatever its timestamp says.
    ///
    /// `last_activity` is stamped when a request arrives, so a long turn leaves it older than the
    /// eviction timeout while the agent is still working. Testing the timestamp alone would evict
    /// the entry out from under the turn -- dropping its `Agent`, its registry and its file
    /// lock mid-tool-call -- which is why the busy check comes first rather than second.
    #[test]
    fn a_session_running_a_turn_is_not_idle_however_old_its_timestamp() {
        let runtime = Mutex::new(());
        let timeout = std::time::Duration::from_secs(1);
        // Derived from the timeout rather than an evocative hour. `Instant` is measured from boot
        // on both Linux and Windows, and subtracting more than the host's uptime panics
        // with "overflow when subtracting duration from instant" -- which is how this
        // failed on a Windows box 55 minutes after a reboot, having passed on the same box
        // earlier the same day. Any age past the timeout proves the same thing, so the test
        // asks for the smallest one that does.
        let age = timeout * 10;
        let ancient = std::sync::RwLock::new(
            std::time::Instant::now()
                .checked_sub(age)
                .expect("the host must have been up longer than a few seconds"),
        );

        assert!(
            session_is_idle(&runtime, &ancient, timeout),
            "untouched for longer than the timeout and nothing running: evictable",
        );

        let _turn = runtime.try_lock().expect("nothing holds it yet");
        assert!(
            !session_is_idle(&runtime, &ancient, timeout),
            "a turn holds the runtime lock, so the session is in use",
        );
    }

    /// And a session used recently stays, so the sweep does not evict the one the editor is on.
    #[test]
    fn a_recently_touched_session_is_not_idle() {
        let runtime = Mutex::new(());
        let just_now = std::sync::RwLock::new(std::time::Instant::now());
        assert!(!session_is_idle(
            &runtime,
            &just_now,
            std::time::Duration::from_secs(60 * 60)
        ));
    }

    #[test]
    fn test_tool_kind_for_covers_builtins() {
        assert_eq!(tool_kind_for("read_file"), ToolKind::Read);
        assert_eq!(tool_kind_for("edit_file"), ToolKind::Edit);
        assert_eq!(tool_kind_for("write_file"), ToolKind::Edit);
        assert_eq!(tool_kind_for("find_files"), ToolKind::Search);
        assert_eq!(tool_kind_for("search_contents"), ToolKind::Search);
        assert_eq!(tool_kind_for("execute_command"), ToolKind::Execute);
        assert_eq!(tool_kind_for("fetch_url"), ToolKind::Fetch);
        assert_eq!(tool_kind_for("agent_spawn"), ToolKind::Think);
        // MCP-loaded tools and anything else fall through.
        assert_eq!(tool_kind_for("mcp__github__create_issue"), ToolKind::Other);
        assert_eq!(tool_kind_for("scratchpad_write"), ToolKind::Other);
        assert_eq!(tool_kind_for("totally_unknown"), ToolKind::Other);
    }

    #[test]
    fn test_todo_items_to_plan_maps_status_and_priority() {
        let items = vec![
            TodoItem {
                text: "first".to_string(),
                status: TodoStatus::Pending,
            },
            TodoItem {
                text: "second".to_string(),
                status: TodoStatus::InProgress,
            },
            TodoItem {
                text: "third".to_string(),
                status: TodoStatus::Completed,
            },
            TodoItem {
                text: "fourth".to_string(),
                status: TodoStatus::Cancelled,
            },
        ];
        let entries = todo_items_to_plan(&items);
        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0].content, "first");
        assert_eq!(entries[0].status, PlanEntryStatus::Pending);
        assert_eq!(entries[1].status, PlanEntryStatus::InProgress);
        assert_eq!(entries[2].status, PlanEntryStatus::Completed);
        // Cancelled has no ACP analogue; it collapses to Completed.
        assert_eq!(entries[3].status, PlanEntryStatus::Completed);
        // meka tracks no per-item priority, so every entry is Medium.
        assert!(
            entries
                .iter()
                .all(|entry| entry.priority == PlanEntryPriority::Medium)
        );
    }

    /// A real PNG, because the payload has to survive a decode: a client's attachment goes through
    /// the same [`crate::image::prepare_image_source`] door a tool result does.
    fn tiny_png() -> Vec<u8> {
        let mut out = Vec::new();
        image::RgbaImage::from_pixel(2, 2, image::Rgba([1, 2, 3, 255]))
            .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
            .expect("encode");
        out
    }

    #[tokio::test]
    async fn test_decode_acp_image_passes_through_within_cap() {
        let data = base64::engine::general_purpose::STANDARD.encode(tiny_png());
        let image = ImageContent::new(data, "image/png".to_string());
        let source = decode_acp_image(&image).await.expect("decode");
        assert_eq!(source.source_type, "base64");
        assert_eq!(source.media_type, "image/png");
    }

    /// A client's attachment is refused for the same reason a `read_file` result is: forwarding
    /// bytes that only look like a PNG lands the provider's rejection inside a committed message.
    #[tokio::test]
    async fn test_decode_acp_image_rejects_a_payload_that_does_not_decode() {
        let mut truncated = tiny_png();
        truncated.truncate(16);
        let data = base64::engine::general_purpose::STANDARD.encode(&truncated);
        let image = ImageContent::new(data, "image/png".to_string());
        let error = decode_acp_image(&image)
            .await
            .expect_err("should reject undecodable bytes");
        assert!(error.contains("decode"), "got: {error}");
    }

    #[tokio::test]
    async fn test_decode_acp_image_rejects_oversized() {
        let raw = vec![0u8; crate::image::MAX_IMAGE_RAW_BYTES + 1];
        let data = base64::engine::general_purpose::STANDARD.encode(&raw);
        let image = ImageContent::new(data, "image/png".to_string());
        let error = decode_acp_image(&image)
            .await
            .expect_err("should reject oversized");
        assert!(error.contains("too large"), "got: {error}");
    }

    #[tokio::test]
    async fn test_decode_acp_image_rejects_bad_base64() {
        let image = ImageContent::new("not%%%valid".to_string(), "image/png".to_string());
        assert!(decode_acp_image(&image).await.is_err());
    }

    #[test]
    fn test_format_embedded_resource_text_inlines_contents() {
        let embedded = EmbeddedResource::new(EmbeddedResourceResource::TextResourceContents(
            agent_client_protocol::schema::v1::TextResourceContents::new(
                "fn main() {}",
                "file:///proj/src/main.rs",
            )
            .mime_type("text/x-rust".to_string()),
        ));
        let tag = format_embedded_resource(&embedded);
        assert_eq!(
            tag,
            "<resource uri=\"file:///proj/src/main.rs\" mime=\"text/x-rust\">fn main() {}</resource>"
        );
    }

    #[test]
    fn test_format_embedded_resource_blob_emits_marker_without_payload() {
        let embedded = EmbeddedResource::new(EmbeddedResourceResource::BlobResourceContents(
            agent_client_protocol::schema::v1::BlobResourceContents::new(
                "QUJD",
                "file:///proj/logo.png",
            )
            .mime_type("image/png".to_string()),
        ));
        let tag = format_embedded_resource(&embedded);
        // The base64 payload must NOT be inlined; only a self-closing marker.
        assert_eq!(
            tag,
            "<resource uri=\"file:///proj/logo.png\" mime=\"image/png\" encoding=\"base64\"/>"
        );
        assert!(!tag.contains("QUJD"));
    }

    #[test]
    fn test_embedded_resource_tag_survives_context_wrapper_strip() {
        // A `<resource>` tag is part of the user's prompt body. Once wrapped by the agent's
        // `<context>...</context>` preamble and stripped on replay, the tag must remain intact.
        let embedded = EmbeddedResource::new(EmbeddedResourceResource::TextResourceContents(
            agent_client_protocol::schema::v1::TextResourceContents::new(
                "hello",
                "file:///note.txt",
            ),
        ));
        let prompt_body = format!("see this\n{}", format_embedded_resource(&embedded));
        let wrapped = format!(
            "<context>\n[Environment context]\n</context>\n\n{}",
            prompt_body
        );
        assert_eq!(crate::session::strip_context_tags(&wrapped), prompt_body);
    }

    #[test]
    fn test_first_user_preview_strips_context_and_truncates() {
        let mut convo = Conversation::new();
        convo.append(crate::provider::Message::user(
            "<context>\n[Environment context]\n</context>\n\nfind all rust files",
        ));
        convo.append(crate::provider::Message::assistant_text("ok"));
        assert_eq!(
            first_user_preview(&convo).as_deref(),
            Some("find all rust files")
        );
    }

    #[test]
    fn test_first_user_preview_none_when_no_user_text() {
        let convo = Conversation::new();
        assert!(first_user_preview(&convo).is_none());
    }

    #[test]
    fn test_tool_locations_resolves_relative_against_cwd() {
        let cwd: SharedCwd = Arc::new(std::sync::RwLock::new(PathBuf::from("/home/agent/proj")));
        let input = serde_json::json!({"path": "src/main.rs"});
        let locations = tool_locations("read_file", &input, &cwd);
        assert_eq!(locations.len(), 1);
        assert_eq!(
            locations[0].path,
            PathBuf::from("/home/agent/proj/src/main.rs")
        );
    }

    #[test]
    fn test_tool_locations_passes_absolute_paths_through() {
        let cwd: SharedCwd = Arc::new(std::sync::RwLock::new(PathBuf::from("/some/other/dir")));
        let input = serde_json::json!({"path": "/etc/hosts"});
        let locations = tool_locations("edit_file", &input, &cwd);
        assert_eq!(locations[0].path, PathBuf::from("/etc/hosts"));
    }

    #[test]
    fn test_tool_locations_empty_for_non_path_tools() {
        let cwd: SharedCwd = Arc::new(std::sync::RwLock::new(PathBuf::from("/")));
        let input = serde_json::json!({"command": "ls"});
        assert!(tool_locations("execute_command", &input, &cwd).is_empty());
        assert!(tool_locations("search_web", &input, &cwd).is_empty());
    }

    #[test]
    fn test_tool_locations_read_file_line_from_offset() {
        let cwd: SharedCwd = Arc::new(std::sync::RwLock::new(PathBuf::from("/home/agent/proj")));
        // `read_file` offset is 0-based; ACP `line` is 1-based.
        let input = serde_json::json!({"path": "src/main.rs", "offset": 41});
        let locations = tool_locations("read_file", &input, &cwd);
        assert_eq!(locations.len(), 1);
        assert_eq!(locations[0].line, Some(42));
        // No offset -> no line.
        let no_offset = serde_json::json!({"path": "src/main.rs"});
        assert_eq!(tool_locations("read_file", &no_offset, &cwd)[0].line, None);
        // Other path tools never set a line, even with an offset present.
        let edit = serde_json::json!({"path": "src/main.rs", "offset": 41});
        assert_eq!(tool_locations("edit_file", &edit, &cwd)[0].line, None);
    }

    #[test]
    fn test_build_completion_content_prefers_diff_metadata() {
        let metadata = Some(ToolOutputMetadata::Diff {
            path: PathBuf::from("/tmp/foo.txt"),
            old_text: Some("old".to_string()),
            new_text: "new".to_string(),
        });
        let content = vec![ToolResultContent::Text {
            text: "ignored".to_string(),
        }];
        let blocks = build_completion_content("edit_file", &content, metadata);
        assert_eq!(blocks.len(), 1);
        assert!(matches!(blocks[0], ToolCallContent::Diff(_)));
    }

    #[test]
    fn test_tool_call_title_per_tool() {
        assert_eq!(
            tool_call_title("execute_command", Some("git status && git diff")),
            "git status && git diff"
        );
        assert_eq!(
            tool_call_title("read_file", Some("src/main.rs")),
            "Read src/main.rs"
        );
        assert_eq!(
            tool_call_title("edit_file", Some("src/lib.rs")),
            "Edit src/lib.rs"
        );
        assert_eq!(
            tool_call_title("write_file", Some("out.txt")),
            "Write out.txt"
        );
        assert_eq!(
            tool_call_title("find_files", Some("**/*.rs")),
            "Find **/*.rs"
        );
        assert_eq!(
            tool_call_title("search_contents", Some("TODO")),
            "Search TODO"
        );
        assert_eq!(
            tool_call_title("fetch_url", Some("https://example.com")),
            "Fetch https://example.com"
        );
        assert_eq!(
            tool_call_title("search_web", Some("rust acp")),
            "Web search: rust acp"
        );
        // MCP / unknown tool with a resolved primary argument.
        assert_eq!(
            tool_call_title("mcp__exa__web_search_exa", Some("query")),
            "mcp__exa__web_search_exa: query"
        );
        // No primary argument resolved -> bare tool name.
        assert_eq!(tool_call_title("read_file", None), "read_file");
    }

    #[test]
    fn test_tool_call_title_sanitizes_whitespace_and_length() {
        // A multi-line command collapses to a single line.
        assert_eq!(
            tool_call_title("execute_command", Some("git status\n  && git diff")),
            "git status && git diff"
        );
        // Over-long titles are truncated with an ellipsis.
        let long = "x".repeat(400);
        let title = tool_call_title("execute_command", Some(&long));
        assert!(title.chars().count() <= 256);
        assert!(title.ends_with('…'));
    }

    #[test]
    fn test_build_completion_content_execute_command_wraps_console() {
        let content = vec![ToolResultContent::Text {
            text: "hello\nworld\n".to_string(),
        }];
        let blocks = build_completion_content("execute_command", &content, None);
        assert_eq!(blocks.len(), 1);
        let ToolCallContent::Content(chunk) = &blocks[0] else {
            panic!("expected ToolCallContent::Content; got {:?}", blocks[0]);
        };
        let ContentBlock::Text(text) = &chunk.content else {
            panic!("expected ContentBlock::Text; got {:?}", chunk.content);
        };
        assert_eq!(text.text, "```console\nhello\nworld\n```");
    }

    #[test]
    fn test_build_completion_content_execute_command_empty_output_no_block() {
        let content = vec![ToolResultContent::Text {
            text: "   \n".to_string(),
        }];
        assert!(build_completion_content("execute_command", &content, None).is_empty());
    }

    fn empty_live_output() -> LiveOutput {
        LiveOutput::new(LiveOutputMode::Text)
    }

    /// The first chunk goes out immediately (a command that prints one line and exits must still
    /// show it), and chunks inside the interval accumulate silently rather than being dropped.
    #[test]
    fn test_live_output_throttles_but_keeps_every_byte() {
        let start = std::time::Instant::now();
        let mut live = empty_live_output();

        assert_eq!(live.push("first\n", start).as_deref(), Some("first\n"));
        assert_eq!(
            live.push("swallowed\n", start + LIVE_OUTPUT_INTERVAL / 2),
            None,
            "a second chunk inside the interval must not produce an update",
        );
        assert_eq!(
            live.push("later\n", start + LIVE_OUTPUT_INTERVAL * 2)
                .as_deref(),
            Some("first\nswallowed\nlater\n"),
            "the throttled chunk must reappear in the next update, not be lost",
        );
    }

    /// The live view is capped, so a command that dumps far more than the cap doesn't make every
    /// subsequent update carry the whole history. Cutting must land on a line boundary and must
    /// never split a multi-byte character (slicing off one panics).
    #[test]
    fn test_live_output_trims_to_a_tail_on_a_line_boundary() {
        let start = std::time::Instant::now();
        let mut live = empty_live_output();
        // Multi-byte content so a naive byte-offset cut would panic rather than merely look wrong.
        let line = "ünïcödé filler line to push past the cap\n";
        let mut now = start;
        for _ in 0..(LIVE_OUTPUT_TAIL_BYTES / line.len() + 10) {
            now += LIVE_OUTPUT_INTERVAL * 2;
            live.push(line, now);
        }
        let tail = live
            .push("final\n", now + LIVE_OUTPUT_INTERVAL * 2)
            .expect("update");
        assert!(
            tail.len() <= LIVE_OUTPUT_TAIL_BYTES + line.len(),
            "tail should stay near the cap; got {} bytes",
            tail.len(),
        );
        assert!(tail.ends_with("final\n"), "the newest output must survive");
        assert!(
            tail.starts_with(line),
            "the cut must land at a line start; got {:?}",
            &tail[..line.len().min(tail.len())],
        );
    }

    /// Terminal mode appends into a buffer the client owns, so each send must carry only what has
    /// arrived since the last one. Re-sending the running total (which is what text mode does)
    /// would make the terminal show every line duplicated more times the longer the command ran.
    #[test]
    fn test_live_output_terminal_mode_sends_each_byte_once() {
        let start = std::time::Instant::now();
        let mut live = LiveOutput::new(LiveOutputMode::Terminal);

        assert_eq!(live.push("alpha\n", start).as_deref(), Some("alpha\n"));
        assert_eq!(
            live.push("beta\n", start + LIVE_OUTPUT_INTERVAL * 2)
                .as_deref(),
            Some("beta\n"),
            "the second send must not repeat the first chunk",
        );
        // Throttled chunks coalesce into the next send rather than being dropped or re-sent.
        assert_eq!(live.push("gamma\n", start + LIVE_OUTPUT_INTERVAL * 2), None);
        assert_eq!(
            live.push("delta\n", start + LIVE_OUTPUT_INTERVAL * 4)
                .as_deref(),
            Some("gamma\ndelta\n"),
        );
    }

    /// The throttle can swallow the final chunk of a command that exits right after printing. In
    /// terminal mode the client's scrollback is the only copy of those bytes, so completion has to
    /// flush them; in text mode the completion update re-sends everything anyway.
    #[test]
    fn test_live_output_terminal_mode_flushes_what_the_throttle_held_back() {
        let start = std::time::Instant::now();
        let mut live = LiveOutput::new(LiveOutputMode::Terminal);
        live.push("first\n", start).expect("first send");
        assert_eq!(live.push("last gasp\n", start), None, "inside the interval");
        assert_eq!(live.take_pending().as_deref(), Some("last gasp\n"));
        assert_eq!(live.take_pending(), None, "flushing twice would duplicate");

        let mut text_mode = LiveOutput::new(LiveOutputMode::Text);
        text_mode.push("buffered\n", start);
        assert_eq!(
            text_mode.take_pending(),
            None,
            "text mode's completion update carries the whole output already",
        );
    }

    /// The tail cap exists only because text mode re-sends its buffer every tick. Applying it to
    /// terminal mode would silently drop output the client can no longer recover.
    #[test]
    fn test_live_output_terminal_mode_never_drops_output_to_the_tail_cap() {
        let start = std::time::Instant::now();
        let mut live = LiveOutput::new(LiveOutputMode::Terminal);
        let line = "x".repeat(1024) + "\n";
        let mut now = start;
        let mut delivered = String::new();
        let rounds = (LIVE_OUTPUT_TAIL_BYTES / line.len()) + 8;
        for _ in 0..rounds {
            now += LIVE_OUTPUT_INTERVAL * 2;
            if let Some(chunk) = live.push(&line, now) {
                delivered.push_str(&chunk);
            }
        }
        assert_eq!(
            delivered.len(),
            line.len() * rounds,
            "every byte must reach the client exactly once, past the text-mode cap",
        );
    }

    /// A live update reports content only. Setting a status would tell the client the call had
    /// finished while the command is still running.
    #[test]
    fn test_live_output_update_carries_no_status() {
        let mut live = empty_live_output();
        let text = live
            .push("building...\n", std::time::Instant::now())
            .expect("first push always emits");
        let fields = ToolCallUpdateFields::new().content(vec![console_content_block(&text)]);
        let update = ToolCallUpdate::new("call_1", fields);
        let wire = serde_json::to_value(&update).expect("serialize");
        assert!(
            wire["status"].is_null(),
            "live updates must not carry a status; got {wire}",
        );
        assert_eq!(
            wire["content"][0]["content"]["text"], "```console\nbuilding...\n```",
            "the live view must render the same way the completed one does",
        );
    }

    #[test]
    fn test_translate_permission_outcome_maps_each_option() {
        use agent_client_protocol::schema::v1::SelectedPermissionOutcome;

        // Capture sticky pushes via a `Cell` so each call site borrows it fresh; this sidesteps the
        // closure-vs-direct-read borrow conflict that comes from sharing one `&mut Vec`.
        let sticky: std::cell::RefCell<Vec<&'static str>> = std::cell::RefCell::new(Vec::new());
        let record = |s: StickyDecision| {
            sticky.borrow_mut().push(match s {
                StickyDecision::AllowAlways => "allow",
                StickyDecision::RejectAlways => "deny",
            });
        };

        assert_eq!(
            translate_permission_outcome(
                RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                    OPTION_ALLOW_ONCE,
                )),
                "read_file",
                record,
            ),
            PermissionOutcome::Allow,
        );
        assert!(
            sticky.borrow().is_empty(),
            "allow_once must not record a sticky"
        );

        assert_eq!(
            translate_permission_outcome(
                RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                    OPTION_ALLOW_ALWAYS,
                )),
                "read_file",
                record,
            ),
            PermissionOutcome::Allow,
        );
        assert_eq!(sticky.borrow().last().copied(), Some("allow"));

        assert_eq!(
            translate_permission_outcome(
                RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                    OPTION_REJECT_ONCE,
                )),
                "write_file",
                record,
            ),
            PermissionOutcome::Deny,
        );
        assert_eq!(
            sticky.borrow().last().copied(),
            Some("allow"),
            "reject_once must not push"
        );

        assert_eq!(
            translate_permission_outcome(
                RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                    OPTION_REJECT_ALWAYS,
                )),
                "write_file",
                record,
            ),
            PermissionOutcome::Deny,
        );
        assert_eq!(sticky.borrow().last().copied(), Some("deny"));

        assert_eq!(
            translate_permission_outcome(RequestPermissionOutcome::Cancelled, "read_file", record,),
            PermissionOutcome::Cancelled,
        );
    }

    #[test]
    fn test_translate_permission_outcome_unknown_option_denies() {
        use agent_client_protocol::schema::v1::SelectedPermissionOutcome;
        let result = translate_permission_outcome(
            RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new("future_option")),
            "read_file",
            &mut |_| {},
        );
        assert_eq!(result, PermissionOutcome::Deny);
    }

    #[test]
    fn test_build_completion_content_falls_back_to_text() {
        let content = vec![ToolResultContent::Text {
            text: "hello".to_string(),
        }];
        let blocks = build_completion_content("read_file", &content, None);
        assert_eq!(blocks.len(), 1);
        assert!(matches!(blocks[0], ToolCallContent::Content(_)));
    }

    /// Image tool results go out as ACP `image` content blocks rather than a text marker, so the
    /// client can render the picture the model was shown. Walks into the block to confirm the
    /// payload and MIME type survive the conversion intact.
    #[test]
    fn test_build_completion_content_forwards_image_content() {
        use crate::provider::ImageSource;
        let content = vec![ToolResultContent::Image {
            source: ImageSource {
                source_type: "base64".to_string(),
                media_type: "image/png".to_string(),
                data: "aGVsbG8=".to_string(),
            },
        }];
        let blocks = build_completion_content("read_file", &content, None);
        assert_eq!(blocks.len(), 1);
        let ToolCallContent::Content(chunk) = &blocks[0] else {
            panic!("expected ToolCallContent::Content; got {:?}", blocks[0]);
        };
        let ContentBlock::Image(image) = &chunk.content else {
            panic!("expected ContentBlock::Image; got {:?}", chunk.content);
        };
        assert_eq!(image.data, "aGVsbG8=");
        assert_eq!(image.mime_type, "image/png");
    }

    /// A tool whose output interleaves text and an image keeps both blocks, in order: the text
    /// marker `read_file` emits alongside an image (`[Image: path]`) is what names the file in the
    /// transcript, so dropping either half loses information.
    #[test]
    fn test_build_completion_content_preserves_mixed_text_and_image() {
        use crate::provider::ImageSource;
        let content = vec![
            ToolResultContent::Text {
                text: "[Image: logo.png]".to_string(),
            },
            ToolResultContent::Image {
                source: ImageSource {
                    source_type: "base64".to_string(),
                    media_type: "image/webp".to_string(),
                    data: "d2VicA==".to_string(),
                },
            },
        ];
        let blocks = build_completion_content("read_file", &content, None);
        assert_eq!(blocks.len(), 2);
        let ToolCallContent::Content(first) = &blocks[0] else {
            panic!("expected ToolCallContent::Content; got {:?}", blocks[0]);
        };
        assert!(
            matches!(&first.content, ContentBlock::Text(text) if text.text == "[Image: logo.png]")
        );
        let ToolCallContent::Content(second) = &blocks[1] else {
            panic!("expected ToolCallContent::Content; got {:?}", blocks[1]);
        };
        assert!(
            matches!(&second.content, ContentBlock::Image(image) if image.mime_type == "image/webp")
        );
    }

    #[test]
    fn test_parse_mode_id_covers_all_levels() {
        assert_eq!(parse_mode_id("none"), Some(Permission::None));
        assert_eq!(parse_mode_id("read"), Some(Permission::Read));
        assert_eq!(parse_mode_id("workspace"), Some(Permission::Workspace));
        assert_eq!(parse_mode_id("ask"), Some(Permission::Ask));
        assert_eq!(
            parse_mode_id("unrestricted"),
            Some(Permission::Unrestricted)
        );
    }

    /// An id naming no mode is refused, rather than resolving to some rung the client did not ask
    /// for.
    #[test]
    fn test_parse_mode_id_rejects_garbage() {
        // Case-insensitive since the parser became shared with `--permission`, where `Read` and
        // `read` have always meant the same thing. Nothing is granted by it: every id still has to
        // name a real mode, and `session/set_mode` separately refuses one outside the enabled set.
        assert_eq!(parse_mode_id("READ"), Some(Permission::Read));
        for unknown in ["admin", "write", "elevated", ""] {
            assert!(
                parse_mode_id(unknown).is_none(),
                "'{unknown}' names no mode and must not resolve"
            );
        }
    }

    #[test]
    fn test_build_mode_state_lists_only_enabled_modes() {
        use crate::permission::{EnabledPermissions, SharedPermission};
        let enabled =
            EnabledPermissions::from_modes([Permission::Read, Permission::Ask]).expect("non-empty");
        let permission = SharedPermission::new(Permission::Read, enabled);

        let state = build_mode_state(&permission);
        let ids: Vec<&str> = state
            .available_modes
            .iter()
            .map(|m| m.id.0.as_ref())
            .collect();
        assert_eq!(ids, vec!["read", "ask"]);
        assert_eq!(state.current_mode_id.0.as_ref(), "read");
        // Descriptions populated.
        assert!(
            state
                .available_modes
                .iter()
                .all(|m| m.description.is_some()),
            "every mode advertised must carry a description"
        );
    }

    #[test]
    fn test_build_mode_state_reflects_current_after_set() {
        use crate::permission::{EnabledPermissions, SharedPermission};
        let permission = SharedPermission::new(Permission::Read, EnabledPermissions::ALL);
        permission
            .try_set(Permission::Unrestricted)
            .expect("unrestricted enabled");
        assert_eq!(
            build_mode_state(&permission).current_mode_id.0.as_ref(),
            "unrestricted"
        );
    }

    fn select_options(option: &SessionConfigOption) -> Vec<(String, Option<String>)> {
        use agent_client_protocol::schema::v1::{SessionConfigKind, SessionConfigSelectOptions};
        match &option.kind {
            SessionConfigKind::Select(select) => match &select.options {
                SessionConfigSelectOptions::Ungrouped(options) => options
                    .iter()
                    .map(|option| {
                        (
                            option.value.0.as_ref().to_string(),
                            option.description.clone(),
                        )
                    })
                    .collect(),
                other => panic!("expected an ungrouped select, got {:?}", other),
            },
            other => panic!("expected a select option, got {:?}", other),
        }
    }

    fn current_value(option: &SessionConfigOption) -> String {
        use agent_client_protocol::schema::v1::SessionConfigKind;
        match &option.kind {
            SessionConfigKind::Select(select) => select.current_value.0.as_ref().to_string(),
            other => panic!("expected a select option, got {:?}", other),
        }
    }

    fn test_profiles(
        entries: &[(&str, Option<&str>)],
    ) -> std::collections::BTreeMap<String, crate::config::ProviderProfile> {
        entries
            .iter()
            .map(|(name, model)| {
                (name.to_string(), crate::config::ProviderProfile {
                    backend: "anthropic-messages".to_string(),
                    model: model.map(str::to_string),
                    ..Default::default()
                })
            })
            .collect()
    }

    /// The `configOptions` permission entry must offer exactly what the `modes` picker offers, or a
    /// client driving one of the two pickers is looking at a different set of levels from a client
    /// driving the other.
    #[test]
    fn the_permission_config_option_matches_the_mode_picker() {
        use crate::permission::{EnabledPermissions, SharedPermission};
        let enabled =
            EnabledPermissions::from_modes([Permission::Read, Permission::Ask]).expect("non-empty");
        let permission = SharedPermission::new(Permission::Read, enabled);

        let option = permission_config_option(&permission);
        let offered: Vec<String> = select_options(&option)
            .into_iter()
            .map(|(value, _description)| value)
            .collect();
        let modes: Vec<String> = build_mode_state(&permission)
            .available_modes
            .iter()
            .map(|mode| mode.id.0.as_ref().to_string())
            .collect();

        assert_eq!(offered, modes);
        assert_eq!(current_value(&option), "read");
        assert_eq!(option.id.0.as_ref(), PERMISSION_CONFIG_ID);
    }

    /// A profile that states no model must not acquire one here: the description is shown to the
    /// user as a fact about the profile, and meka does not know what an unstated model resolves to.
    #[test]
    fn a_provider_option_describes_only_a_stated_model() {
        let profiles = test_profiles(&[("work", Some("claude-opus-5")), ("personal", None)]);

        let option = provider_config_option(&profiles, "personal");

        assert_eq!(option.id.0.as_ref(), PROVIDER_CONFIG_ID);
        assert_eq!(current_value(&option), "personal");
        assert_eq!(select_options(&option), vec![
            ("personal".to_string(), None),
            ("work".to_string(), Some("claude-opus-5".to_string())),
        ],);
    }

    /// A session pinned to a profile that has since left `config.toml` selects nothing rather than
    /// silently presenting some other profile as the one it runs on.
    #[test]
    fn a_provider_no_longer_configured_selects_nothing() {
        let profiles = test_profiles(&[("work", None)]);

        let option = provider_config_option(&profiles, "retired");

        assert_eq!(current_value(&option), "retired");
        assert!(
            !select_options(&option)
                .iter()
                .any(|(value, _description)| value == "retired"),
            "a profile that is gone must not appear among the choices"
        );
    }

    #[tokio::test]
    async fn test_slash_to_prompt_text_passes_through_non_slash() {
        let cache = SkillCache::for_root(None);
        let out = slash_to_prompt_text("just a normal prompt".to_string(), &cache)
            .await
            .expect("ok");
        assert_eq!(out, "just a normal prompt");
    }

    #[tokio::test]
    async fn test_slash_to_prompt_text_passes_through_paste_shaped_input() {
        // A pasted path like `/etc/hosts is a config file` has an invalid skill-name first token
        // (slash inside the name), so the helper must NOT touch it.
        let cache = SkillCache::for_root(None);
        let out = slash_to_prompt_text("/etc/hosts is the config file".to_string(), &cache)
            .await
            .expect("pass-through");
        assert_eq!(out, "/etc/hosts is the config file");
    }

    #[tokio::test]
    async fn test_slash_to_prompt_text_passes_through_double_slash_comment() {
        // `//foo` parses as name="/foo", which is invalid; pass through.
        let cache = SkillCache::for_root(None);
        let out = slash_to_prompt_text("//comment line".to_string(), &cache)
            .await
            .expect("pass-through");
        assert_eq!(out, "//comment line");
    }

    #[tokio::test]
    async fn test_slash_to_prompt_text_unknown_but_valid_name_errors() {
        // A clean `/<name>` shape with a syntactically valid skill name that isn't installed:
        // error, since the only realistic source of this shape is a typo'd palette pick.
        let cache = SkillCache::for_root(None);
        let err = slash_to_prompt_text("/nonexistent".to_string(), &cache)
            .await
            .expect_err("should error");
        assert!(
            matches!(err, SlashInvocationError::SkillNotFound(ref reason)
                if reason == "no skill named 'nonexistent'"),
            "an absent name reads as absent, not as an unreadable file: {err}"
        );
    }

    /// `/name` for a skill whose `SKILL.md` will not parse reports the file, not "unknown skill".
    ///
    /// This is a person typing a name they know exists, so answering that it does not sends them
    /// looking for something they already have.
    #[tokio::test]
    async fn test_slash_to_prompt_text_reports_a_broken_skill_as_broken() {
        let temp = tempfile::tempdir().expect("tempdir");
        let skill_dir = temp.path().join("wrecked");
        std::fs::create_dir_all(&skill_dir).expect("mkdir skill");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: wrecked\ndescription: [unclosed\n---\nbody\n",
        )
        .expect("write skill");
        let cache = SkillCache::for_root(Some(temp.path().to_path_buf()));

        let err = slash_to_prompt_text("/wrecked".to_string(), &cache)
            .await
            .expect_err("should error");
        let message = err.to_string();
        assert!(
            message.contains("could not be read"),
            "a file that is right there is not an unknown skill: {message}"
        );
        assert!(message.contains("frontmatter"), "{message}");
    }

    #[tokio::test]
    async fn test_slash_to_prompt_text_known_skill_composes_body() {
        // Drop a SKILL.md under a tempdir, point a fresh cache at it.
        let temp = tempfile::tempdir().expect("tempdir");
        let skill_dir = temp.path().join("demo");
        std::fs::create_dir_all(&skill_dir).expect("mkdir skill");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\ndescription: demo skill\n---\nrun ls in scripts/\n",
        )
        .expect("write SKILL.md");

        let cache = SkillCache::for_root(Some(temp.path().to_path_buf()));
        let out = slash_to_prompt_text("/demo only fetch UK news".to_string(), &cache)
            .await
            .expect("ok");
        assert!(
            out.starts_with("only fetch UK news\n\n"),
            "extra context must lead: {}",
            out
        );
        assert!(
            out.contains("run ls in scripts/"),
            "body must be passed through verbatim: {}",
            out
        );
        assert!(
            out.contains("Base directory for this skill"),
            "skill_context_header must be present: {}",
            out
        );
    }

    #[tokio::test]
    async fn test_slash_to_prompt_text_known_skill_no_extra() {
        let temp = tempfile::tempdir().expect("tempdir");
        let skill_dir = temp.path().join("ping");
        std::fs::create_dir_all(&skill_dir).expect("mkdir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\ndescription: ping\n---\npong\n",
        )
        .expect("write");

        let cache = SkillCache::for_root(Some(temp.path().to_path_buf()));
        let out = slash_to_prompt_text("/ping".to_string(), &cache)
            .await
            .expect("ok");
        // No `extra\n\n` prefix when the user passed only the skill name; the body stands alone.
        assert!(
            !out.starts_with("\n\n"),
            "bare /skill must not have a leading newline: {:?}",
            out
        );
        assert!(out.contains("pong"));
    }

    #[test]
    fn test_shared_client_state_round_trip() {
        let shared = SharedClientState::default();
        // Default snapshot has every capability false and no client identity recorded.
        let initial = shared.capabilities();
        assert!(!initial.fs.read_text_file);
        assert!(!initial.fs.write_text_file);
        assert!(!initial.terminal);
        assert!(shared.client_info().is_none());

        let updated_caps = ClientCapabilities::new()
            .fs(
                agent_client_protocol::schema::v1::FileSystemCapabilities::new()
                    .read_text_file(true)
                    .write_text_file(true),
            )
            .terminal(true);
        let updated_info = Implementation::new("test-editor", "9.9.9");
        shared.record_initialize(updated_caps, Some(updated_info));

        let after_caps = shared.capabilities();
        assert!(after_caps.fs.read_text_file);
        assert!(after_caps.fs.write_text_file);
        assert!(after_caps.terminal);
        let after_info = shared.client_info().expect("info present");
        assert_eq!(after_info.name, "test-editor");
        assert_eq!(after_info.version, "9.9.9");
    }

    #[test]
    fn test_describe_client_formats_known_and_unknown() {
        assert_eq!(describe_client(None), "<unknown> <unknown>");
        let info = Implementation::new("zed", "0.999.0");
        assert_eq!(describe_client(Some(&info)), "zed 0.999.0");
    }
}
