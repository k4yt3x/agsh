//! `meka acp` subcommand. Speaks the Agent Client Protocol (ACP) on stdio so editor / web /
//! messenger clients can drive a meka turn end to end.
//!
//! # Capability surface
//!
//! - **Lifecycle**: `initialize` (with `clientInfo` capture + `agentInfo` advertisement),
//!   `session/new`, `session/load` (replays the persisted conversation as `session/update`
//!   notifications), `session/list` (cwd-filtered, cursor-paginated; sub-agent sessions hidden),
//!   `session/resume`, `session/close`, `session/cancel`, `session/set_mode`.
//! - **Turn**: `session/prompt` streams `agent_message_chunk`, `agent_thought_chunk`, the full
//!   `tool_call` + `tool_call_update` lifecycle (with diff content blocks from
//!   [`crate::frontend::ToolOutputMetadata::Diff`]), and ends with `end_turn` / `cancelled` stop
//!   reasons. `session/request_permission` handles `ask`-mode tool approvals; per-session sticky
//!   always/never sets short-circuit subsequent requests.
//! - **Commands + modes**: built-in local commands (`/status`, `/mcp`) and installed skills surface
//!   as `available_commands_update` palette entries. Local commands render text and end the turn
//!   with no model call; skills resolve `/<skill-name> [extra]` prompts to the rendered skill body
//!   before the turn. `Permission` levels map 1:1 to ACP `SessionMode` ids, advertised on every
//!   session-creation response and mutated live via `session/set_mode`.
//! - **Delegation**: `read_file` / `write_file` / `edit_file` / `execute_command` route through the
//!   client's `fs/read_text_file`, `fs/write_text_file`, and `terminal/*` when the matching
//!   capability is offered, falling back to local syscalls otherwise. `read` permission mode
//!   bypasses `execute_command` delegation so the local Landlock / bwrap / sandbox-exec /
//!   Low-Integrity jail stays in place.
//!
//! Multi-session: any number of sessions can coexist in one `meka acp` process. Each session has
//! its own cwd, permission cell, conversation, cancellation token, and per-session `Agent` +
//! `AcpFrontend`. Sessions share process-wide dependencies (provider, MCP manager, session DB,
//! skill cache) via `Arc`. Two sessions can run `session/prompt` calls in parallel; there is no
//! global mutex serialising turns. Sub-agents reach the parent's client through
//! [`crate::frontend::PermissionForwardingFrontend`], so their permission prompts, fs delegates,
//! and terminal delegates all flow through the parent session's editor UI.

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
        ContentBlock, ContentChunk, CreateTerminalRequest, CurrentModeUpdate, Diff,
        EmbeddedResource, EmbeddedResourceResource, EnvVariable, ImageContent, Implementation,
        InitializeRequest, InitializeResponse, KillTerminalRequest, ListSessionsRequest,
        ListSessionsResponse, LoadSessionRequest, LoadSessionResponse, NewSessionRequest,
        NewSessionResponse, PermissionOption, PermissionOptionKind, Plan, PlanEntry,
        PlanEntryPriority, PlanEntryStatus, PromptCapabilities, PromptRequest, PromptResponse,
        ReadTextFileRequest, ReleaseTerminalRequest, RequestPermissionOutcome,
        RequestPermissionRequest, ResumeSessionRequest, ResumeSessionResponse, SessionCapabilities,
        SessionCloseCapabilities, SessionId, SessionInfo, SessionInfoUpdate,
        SessionListCapabilities, SessionMode, SessionModeId, SessionModeState, SessionNotification,
        SessionResumeCapabilities, SessionUpdate, SetSessionModeRequest, SetSessionModeResponse,
        StopReason, TerminalOutputRequest, ToolCall, ToolCallContent, ToolCallLocation,
        ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind, UnstructuredCommandInput,
        UsageUpdate, WaitForTerminalExitRequest, WriteTextFileRequest,
    },
};
use async_trait::async_trait;
use base64::Engine;
use futures::io::AsyncRead;
use tokio::sync::Mutex;
use tokio_util::{
    compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt},
    sync::CancellationToken,
};

use crate::{
    agent::{Agent, SharedCwd, resolve_against_cwd},
    config::ResolvedConfig,
    conversation::Conversation,
    error::MekaError,
    frontend::{
        DelegatedExecOutput, DelegatedExecSpec, Frontend, FrontendError, FrontendEvent,
        PermissionOutcome, PermissionRequest, ToolOutputMetadata,
    },
    mcp,
    permission::{Permission, SharedPermission},
    provider::{AuthCredential, ContentBlock as MekaContentBlock, Role, ToolResultContent},
    session::SessionManager,
    skills::SkillCache,
    tools::todo::{TodoItem, TodoStatus},
};

/// Build a JSON-RPC `InvalidParams` error (`-32602`) with a free-form human-readable message in the
/// `data` field. Mirrors [`agent_client_protocol::util::internal_error`] but for the
/// input-validation cases (unknown sessionId, malformed UUID, unsupported mode, non-text content).
/// Clients can rely on the JSON-RPC code to distinguish "bad input" from "server failure".
fn invalid_params_error(message: impl ToString) -> agent_client_protocol::Error {
    agent_client_protocol::Error::invalid_params().data(message.to_string())
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
    /// ACP `usage_update` so editors show "tokens used / context window". `context_window` holds
    /// the resolved window for the `size` field, set once the agent is built (`0` until then,
    /// which suppresses the update).
    context_tokens: Arc<std::sync::atomic::AtomicU64>,
    context_window: std::sync::atomic::AtomicU64,
}

impl AcpFrontend {
    fn new(
        connection: ConnectionTo<Client>,
        session_id: SessionId,
        cwd: SharedCwd,
        client_state: SharedClientState,
        transport_dead: Arc<std::sync::atomic::AtomicBool>,
        context_tokens: Arc<std::sync::atomic::AtomicU64>,
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
            context_window: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Record the resolved context-window size for this session's model, used as the `size` of the
    /// ACP `usage_update`. Called once the agent is built (its window resolution is authoritative).
    fn set_context_window(&self, window: u64) {
        self.context_window
            .store(window, std::sync::atomic::Ordering::Relaxed);
    }

    /// Mark the stdio transport as dead. Called from `emit` and the `session/load` replay loop
    /// whenever `send_notification` reports an error. Idempotent. The trait-level
    /// `client_disconnected()` read below surfaces the same flag back to the agent loop.
    fn mark_transport_dead(&self) {
        self.transport_dead
            .store(true, std::sync::atomic::Ordering::Relaxed);
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
        let connection = self.connection.clone();
        let session_id = self.session_id.clone();

        let update = match event {
            FrontendEvent::AssistantTextDelta(text) => {
                SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
                    agent_client_protocol::schema::v1::TextContent::new(text),
                )))
            }
            FrontendEvent::ThinkingBlock { content, .. } => {
                SessionUpdate::AgentThoughtChunk(ContentChunk::new(ContentBlock::Text(
                    agent_client_protocol::schema::v1::TextContent::new(content),
                )))
            }
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
                let call = ToolCall::new(id, title)
                    .kind(tool_kind_for(&name))
                    .status(ToolCallStatus::InProgress)
                    .locations(locations)
                    .raw_input(input);
                SessionUpdate::ToolCall(call)
            }
            FrontendEvent::ToolCallCompleted {
                id,
                name,
                is_error,
                content,
                metadata,
            } => {
                let status = if is_error {
                    ToolCallStatus::Failed
                } else {
                    ToolCallStatus::Completed
                };
                let acp_content = build_completion_content(&name, &content, metadata);
                let mut fields = ToolCallUpdateFields::new()
                    .status(status)
                    .content(acp_content);
                // Surface the structured tool output too, so clients (e.g. Zed's tool-call detail
                // view) can introspect the result beyond the rendered `content` blocks.
                if let Ok(raw) = serde_json::to_value(&content) {
                    fields = fields.raw_output(raw);
                }
                SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(id, fields))
            }
            FrontendEvent::Notice(notice) => {
                // No dedicated ACP primitive for advisories; surface inline as an assistant-message
                // chunk with an `[meka]` prefix so editor transcripts record the side-effect and
                // clients can filter or style by that prefix. `notice.level` is unused on the wire
                // today; when ACP grows a typed notice variant, branch on it here.
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
            // REPL-specific signage (lifecycle, diffs).
            _ => return,
        };

        if let Err(error) =
            connection.send_notification(SessionNotification::new(session_id, update))
        {
            tracing::debug!("AcpFrontend send_notification failed: {}", error);
            self.mark_transport_dead();
        }
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

        let options = vec![
            PermissionOption::new(OPTION_ALLOW_ONCE, "Allow", PermissionOptionKind::AllowOnce),
            PermissionOption::new(
                OPTION_ALLOW_ALWAYS,
                "Always allow",
                PermissionOptionKind::AllowAlways,
            ),
            PermissionOption::new(OPTION_REJECT_ONCE, "Deny", PermissionOptionKind::RejectOnce),
            PermissionOption::new(
                OPTION_REJECT_ALWAYS,
                "Always deny",
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
            result = connection.send_request(req).block_task() => match result {
                Ok(resp) => resp,
                Err(error) => {
                    tracing::debug!("request_permission send_request failed: {}", error);
                    // Spec-conformant clients always reply with a `Selected` or `Cancelled`
                    // outcome, so an `Err` here is almost certainly transport-level. Mark the
                    // connection dropped so the agent loop short-circuits on the next pre-iteration
                    // check instead of running a tool, emitting a denied result, and only then
                    // discovering the client is gone via the next emit. The FS / execute delegates
                    // intentionally don't do this: those paths legitimately receive JSON-RPC error
                    // responses (e.g. terminal/create denied), which would produce false-positive
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
        Some(match connection.send_request(request).block_task().await {
            Ok(response) => Ok(response.content),
            Err(error) => Err(FrontendError::new(format!(
                "fs/read_text_file failed: {}",
                error
            ))),
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
        Some(match connection.send_request(request).block_task().await {
            Ok(_) => Ok(()),
            Err(error) => Err(FrontendError::new(format!(
                "fs/write_text_file failed: {}",
                error
            ))),
        })
    }

    async fn delegate_execute(
        &self,
        spec: DelegatedExecSpec,
    ) -> Option<Result<DelegatedExecOutput, FrontendError>> {
        let caps = self.client_state.capabilities();
        if !caps.terminal {
            return None;
        }
        let connection = self.connection.clone();
        let session_id = self.session_id.clone();
        Some(run_delegated_execute(connection, session_id, spec).await)
    }

    async fn handle_elicitation(
        &self,
        prompt: crate::mcp::elicitation::ElicitationPrompt,
    ) -> crate::mcp::elicitation::ElicitationResponse {
        // ACP has no protocol primitive for arbitrary server forms. The pragmatic stance until one
        // lands is to auto-decline with an info-level log so editor users can see in their agent
        // stderr that an elicitation arrived and was passed back to the server. A future per-ACP
        // path could synthesize a `session/request_permission` round-trip (the only existing
        // round-trip primitive) by mapping form fields to permission options, but that conflates
        // tool approval and form input, which is the kind of overload the protocol is likely to
        // rule out as it grows a proper elicitation surface.
        tracing::warn!(
            "ACP session received MCP elicitation from '{}' ({}); auto-declining (no ACP \
             primitive for form/URL prompts yet): {}",
            prompt.server_name,
            match &prompt.kind {
                crate::mcp::elicitation::ElicitationKind::Form { .. } => "form",
                crate::mcp::elicitation::ElicitationKind::Url { .. } => "url",
            },
            prompt.message,
        );
        crate::mcp::elicitation::ElicitationResponse::Decline
    }
}

/// Stable string IDs for the four permission options. The agent and the client must agree on these;
/// picking them as `const`s keeps the match arm in [`translate_permission_outcome`] honest.
const OPTION_ALLOW_ONCE: &str = "allow_once";
const OPTION_ALLOW_ALWAYS: &str = "allow_always";
const OPTION_REJECT_ONCE: &str = "reject_once";
const OPTION_REJECT_ALWAYS: &str = "reject_always";

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
        "fetch_url" | "web_search" => ToolKind::Fetch,
        "spawn_agent" => ToolKind::Think,
        // skill, scratchpad_*, render_image, load_tool, mcp__*, and any future
        // built-ins.
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
        ("web_search", Some(query)) => format!("Web search: {query}"),
        ("spawn_agent", Some(task)) => format!("Sub-agent: {task}"),
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

/// Decode an ACP image content block into meka's internal [`crate::provider::ImageSource`],
/// normalizing through the shared image pipeline: base64-decode the payload, classify the format
/// (by the declared MIME type, falling back to the magic bytes), then enforce the size cap and
/// convert unsupported formats to PNG. Returns a human-readable message on failure.
fn decode_acp_image(image: &ImageContent) -> Result<crate::provider::ImageSource, String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(image.data.as_bytes())
        .map_err(|error| format!("base64 decode failed: {}", error))?;
    let handling = match crate::image::classify_content_type(&image.mime_type) {
        crate::image::ImageHandling::Unsupported => crate::image::classify_bytes(&bytes),
        handling => handling,
    };
    crate::image::prepare_image_source(handling, &bytes)
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

/// Build the `content` array of a `tool_call_update` from meka's tool output. A populated `Diff`
/// metadata wins (so clients like Zed get the structured diff for apply-UI). `execute_command`
/// output is wrapped in a `console` code block so editors render it monospaced (mirrors
/// claude-agent-acp's no-terminal fallback). Other tools pass their text/image blocks through.
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
        let trimmed = combined.trim_end();
        if trimmed.is_empty() {
            return Vec::new();
        }
        return vec![text_content_block(format!("```console\n{trimmed}\n```"))];
    }

    content
        .iter()
        .map(|block| {
            let text = match block {
                ToolResultContent::Text { text } => text.clone(),
                // No ACP analogue for image content yet; collapse to a text marker so the wire
                // stays valid.
                ToolResultContent::Image { .. } => "[image]".to_string(),
            };
            text_content_block(text)
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
        tracing::debug!("session/load replay send_notification failed: {}", error);
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

/// Drive the ACP `terminal/*` four-call dance for a delegated execute:
/// `terminal/create` → wait for exit (raced against the agent's
/// cancellation token + the spec's timeout) → `terminal/output` →
/// `terminal/release`. On cancel or timeout we send `terminal/kill`
/// first and then still read whatever output the editor buffered.
async fn run_delegated_execute(
    connection: ConnectionTo<Client>,
    session_id: SessionId,
    spec: DelegatedExecSpec,
) -> Result<DelegatedExecOutput, FrontendError> {
    // Build CreateTerminalRequest. Empty `args` / `env` / unset `cwd` / unset `output_byte_limit`
    // are all fine; the builder leaves them at defaults.
    let mut create = CreateTerminalRequest::new(session_id.clone(), spec.command.clone());
    if !spec.args.is_empty() {
        create = create.args(spec.args.clone());
    }
    if !spec.env.is_empty() {
        let env_vars: Vec<EnvVariable> = spec
            .env
            .iter()
            .map(|(name, value)| EnvVariable::new(name.clone(), value.clone()))
            .collect();
        create = create.env(env_vars);
    }
    if let Some(cwd) = spec.cwd.clone() {
        create = create.cwd(cwd);
    }
    if let Some(limit) = spec.output_byte_limit {
        create = create.output_byte_limit(limit);
    }

    let terminal_id = match connection.send_request(create).block_task().await {
        Ok(response) => response.terminal_id,
        Err(error) => {
            return Err(FrontendError::new(format!(
                "terminal/create failed: {}",
                error
            )));
        }
    };

    // Wait for exit, racing the agent's cancellation token + the spec's timeout. On race-loss we
    // kill the terminal first; the follow-up `terminal/output` still returns whatever was buffered.
    //
    // Default cap of 15 minutes when the caller didn't supply one; interactive tools can override
    // per-call. The agent's cancel token is still the primary escape hatch; the timeout is just the
    // worst-case bound if both the cancel path and the underlying process get wedged.
    let timeout = spec
        .timeout
        .unwrap_or_else(|| std::time::Duration::from_secs(60 * 15));
    let wait_request = WaitForTerminalExitRequest::new(session_id.clone(), terminal_id.clone());
    let killed = tokio::select! {
        result = connection.send_request(wait_request).block_task() => {
            match result {
                Ok(_response) => false,
                Err(error) => {
                    return Err(FrontendError::new(format!(
                        "terminal/wait_for_exit failed: {}",
                        error
                    )));
                }
            }
        }
        _ = spec.cancellation.cancelled() => true,
        _ = tokio::time::sleep(timeout) => true,
    };
    if killed
        && let Err(error) = connection
            .send_request(KillTerminalRequest::new(
                session_id.clone(),
                terminal_id.clone(),
            ))
            .block_task()
            .await
    {
        tracing::debug!("terminal/kill failed: {}", error);
    }

    // Fetch the final output regardless of which arm won.
    let output_request = TerminalOutputRequest::new(session_id.clone(), terminal_id.clone());
    let output_response = match connection.send_request(output_request).block_task().await {
        Ok(response) => response,
        Err(error) => {
            // Try to release before bubbling the error so we don't leak the terminal handle on the
            // client side.
            if let Err(release_error) = connection
                .send_request(ReleaseTerminalRequest::new(
                    session_id.clone(),
                    terminal_id.clone(),
                ))
                .block_task()
                .await
            {
                tracing::debug!(
                    "terminal/release after output failure also failed: {}",
                    release_error,
                );
            }
            return Err(FrontendError::new(format!(
                "terminal/output failed: {}",
                error
            )));
        }
    };

    // Best-effort release; errors are non-fatal (the editor cleans up on disconnect anyway). Log at
    // debug for diagnostics.
    if let Err(error) = connection
        .send_request(ReleaseTerminalRequest::new(session_id, terminal_id))
        .block_task()
        .await
    {
        tracing::debug!("terminal/release failed: {}", error);
    }

    let (exit_code, signal) = match output_response.exit_status {
        Some(status) => (
            // ACP wire protocol uses u32 exit codes; meka's `DelegatedExecOutput.exit_code` is
            // i32. Real exit codes are 0-255, so `try_from` always succeeds; the
            // explicit fallback to -1 documents the choice instead of doing a lossy
            // `as`-cast.
            status
                .exit_code
                .map(|code| i32::try_from(code).unwrap_or(-1)),
            status.signal.clone(),
        ),
        None if killed => (None, Some("SIGTERM".to_string())),
        None => (None, None),
    };

    Ok(DelegatedExecOutput {
        output: output_response.output,
        exit_code,
        signal,
        truncated: output_response.truncated,
    })
}

/// Map a meka [`Permission`] to its ACP [`SessionModeId`] string. The mapping is the lowercase
/// debug name (`none` / `read` / `ask` / `write`), the same string `Permission::Display` produces.
/// It is kept as a dedicated function so the inverse parser ([`parse_mode_id`]) reads as the
/// obvious complement.
fn mode_id_for(permission: Permission) -> SessionModeId {
    SessionModeId::from(permission.to_string())
}

/// Parse a `SessionModeId` (treated as a `&str`) into the matching `Permission`. Returns `None` for
/// any unrecognised mode id. The match arms must stay in lock-step with [`mode_id_for`].
fn parse_mode_id(id: &str) -> Option<Permission> {
    match id {
        "none" => Some(Permission::None),
        "read" => Some(Permission::Read),
        "ask" => Some(Permission::Ask),
        "write" => Some(Permission::Write),
        _ => None,
    }
}

/// Human-readable label for a permission mode, shown in editor mode pickers next to each option.
/// Kept in lock-step with the REPL's `/permission` output and the `[permissions]` keys in
/// `config.toml` so users see the same vocabulary everywhere.
fn mode_display_name(permission: Permission) -> &'static str {
    match permission {
        Permission::None => "None",
        Permission::Read => "Read",
        Permission::Ask => "Ask",
        Permission::Write => "Write",
    }
}

/// One-line description of what a permission mode lets the agent do. Shown beneath the mode label
/// in editor pickers.
fn mode_description(permission: Permission) -> &'static str {
    match permission {
        Permission::None => "No tools available.",
        Permission::Read => "File reads and searches only. No writes, no shell.",
        Permission::Ask => "Every write or shell command requires approval.",
        Permission::Write => "All tools allowed without per-call approval.",
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

/// Plain-text `/status` output, mirroring the REPL's `render_session_status` plus the model and
/// mode an ACP client may not otherwise surface.
fn build_status_text(runtime: &SessionRuntime, shared: &crate::SharedDeps) -> String {
    use std::fmt::Write as _;
    let snap = runtime.agent.session_stats_snapshot();
    let (used, window) = runtime.agent.context_usage();
    let mut out = String::from("meka session status\n");
    if let Some(model) = shared.config.model.as_deref() {
        let _ = writeln!(out, "  Model:    {model}");
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
            .iter()
            .filter(|skill| !LOCAL_COMMANDS.iter().any(|(name, _)| *name == skill.name))
            .map(|skill| {
                AvailableCommand::new(skill.name.clone(), skill.description.clone()).input(
                    AvailableCommandInput::Unstructured(UnstructuredCommandInput::new(
                        "additional context (optional)",
                    )),
                )
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
    SkillNotFound(String),
    SkillLoadFailed { name: String, source: String },
}

impl std::fmt::Display for SlashInvocationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlashInvocationError::SkillNotFound(name) => {
                write!(f, "unknown skill '{}'", name)
            }
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
///   [`crate::skills::load_skill_body`]'s output (with `${MEKA_SKILL_DIR}` / `${MEKA_SESSION_ID}`
///   substituted). Empty `extra` collapses to just `body`. Same composition the REPL uses at
///   `main.rs:1004`.
/// - `/<name>` with a syntactically valid skill name but no installed skill of that name: error.
///   The shape is too deliberate to be a paste, so a typo deserves a clear "unknown skill" rather
///   than silently going to the model.
async fn slash_to_prompt_text(
    prompt_text: String,
    skills: &Arc<SkillCache>,
    session_id: &str,
) -> Result<String, SlashInvocationError> {
    let Some((name, extra)) = split_acp_slash(&prompt_text) else {
        return Ok(prompt_text);
    };
    // Anything that doesn't even look like a skill identifier was never going to match. Pass
    // through so pasted paths and code comments reach the model unchanged.
    if crate::skills::validate_skill_name(&name).is_err() {
        return Ok(prompt_text);
    }
    let snapshot = skills.current().await;
    let Some(skill) = snapshot.iter().find(|skill| skill.name == name) else {
        return Err(SlashInvocationError::SkillNotFound(name));
    };
    let body = crate::skills::load_skill_body(skill, Some(session_id))
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
    /// Resolved per-profile vision flag (`[providers.<name>].vision`, default `true`). Gates the
    /// advertised `image` prompt capability and whether `session/prompt` accepts image blocks.
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
    session_lock: Arc<crate::session::SessionLock>,
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
    Ok(())
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
                tracing::debug!("failed to install SIGTERM handler ({error}); using Ctrl-C only");
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
    // Resolve provider credentials the same way the REPL path does.
    let credential = resolve_credential_for_acp(&config, &session_manager.token_store()).await?;

    // Capture the resolved per-profile vision flag before `config` is moved into
    // `build_shared_deps`. It gates the advertised `image` prompt capability and image ingest
    // below.
    let vision = config.vision;

    // Build process-wide shared deps once. Sessions hold an `Arc<SharedDeps>` and read fields by
    // reference; no work happens here that needs to be re-run per session.
    let shared = Arc::new(
        super::build_shared_deps(
            config,
            session_manager,
            credential,
            mcp_manager,
            mcp_context,
        )
        .await?,
    );

    // Test-only: swap in a scripted provider when the integration harness asks for it. The real
    // provider built above is dropped unused. Only compiled in debug builds. We rebuild SharedDeps
    // with the mock provider before installing it.
    #[cfg(debug_assertions)]
    let shared = if std::env::var("MEKA_ACP_MOCK_PROVIDER").as_deref() == Ok("1") {
        let rounds = crate::provider::mock::load_script_from_env()?.unwrap_or_default();
        let mock = Arc::new(crate::provider::mock::MockProvider::from_rounds(rounds));
        // Replace just the provider field, inheriting the rest from the real SharedDeps.
        // `SharedDeps: Clone` keeps this one-line and means future field additions are picked up
        // automatically; Rust still enforces the exhaustive struct literal at compile time on top.
        let new_inner = crate::SharedDeps {
            provider: mock,
            ..(*shared).clone()
        };
        tracing::info!("MEKA_ACP_MOCK_PROVIDER=1: using scripted mock provider");
        Arc::new(new_inner)
    } else {
        shared
    };

    let client_state = SharedClientState::default();
    let transport_dead = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let state = Arc::new(ServerState {
        shared: Arc::clone(&shared),
        client_state: client_state.clone(),
        sessions: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
        transport_dead,
        vision,
    });

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
                        .list(Some(SessionListCapabilities::new()))
                        .resume(Some(SessionResumeCapabilities::new()))
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
                    let session_uuid = match state
                        .shared
                        .session_manager
                        .create_session(Some(req.cwd.clone()))
                        .await
                    {
                        Ok(uuid) => uuid,
                        Err(error) => {
                            return responder.respond_with_error(
                                agent_client_protocol::util::internal_error(format!(
                                    "failed to create meka session: {}",
                                    error
                                )),
                            );
                        }
                    };
                    // Take the OS file lock on the newly created session row so a second `meka acp`
                    // process (or an `meka repl`) can't open the same id and interleave events.
                    let session_lock = match state.shared.session_manager.lock_session(session_uuid)
                    {
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
                    let entry = SessionEntry {
                        runtime: Arc::new(Mutex::new(runtime)),
                        cancellation: Arc::new(std::sync::RwLock::new(CancellationToken::new())),
                        cancel_pending: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                        title_sent: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                        permission: permission.clone(),
                        frontend,
                        session_lock,
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
                    emit_available_commands(&cx, &session_id, &state.shared.skills).await;

                    responder.respond(NewSessionResponse::new(session_id).modes(modes))
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
                async move |req: CloseSessionRequest, responder, _cx: ConnectionTo<Client>| {
                    handle_close_session(Arc::clone(&state), req, responder).await
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
            ContentBlock::Image(image) if state.vision => match decode_acp_image(image) {
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

    // Look up the target session by id under the outer read lock, clone the entry (cheap, two
    // `Arc`s), drop the outer guard. From here on, only the per-session runtime mutex is held; the
    // sibling cancellation cell is accessible to the cancel handler throughout the turn.
    let session_id_str = req.session_id.0.as_ref().to_string();
    let entry = {
        let sessions = state.sessions.read().await;
        match sessions.get(&session_id_str) {
            Some(entry) => entry.clone(),
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
        return responder.respond(PromptResponse::new(StopReason::EndTurn));
    }

    let original_prompt_text = prompt_text.clone();
    let prompt_text = match slash_to_prompt_text(
        prompt_text,
        &state.shared.skills,
        session_id_str.as_str(),
    )
    .await
    {
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

    responder.respond(PromptResponse::new(stop_reason))
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

    let runtime = match build_session_runtime(
        &state.shared,
        &state.client_state,
        &state.transport_dead,
        cx.clone(),
        session_id.clone(),
        session_id_str.clone(),
        session_uuid,
        req.cwd.clone(),
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
    let frontend = Arc::clone(&runtime.frontend);
    // History already carries the first user message, so the title is known; push it once now,
    // sharing the flag with the entry so a later prompt won't re-emit it.
    let title_sent = Arc::new(std::sync::atomic::AtomicBool::new(false));
    maybe_emit_session_title(&cx, &session_id, &title_sent, &runtime.messages);
    let entry = SessionEntry {
        runtime: Arc::new(Mutex::new(runtime)),
        cancellation: Arc::new(std::sync::RwLock::new(CancellationToken::new())),
        cancel_pending: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        title_sent,
        permission: permission.clone(),
        frontend,
        session_lock,
    };
    state.sessions.write().await.insert(session_id_str, entry);

    // Refresh the palette + advertise the current mode set: the editor was reopened, so its UI
    // starts blank.
    let modes = build_mode_state(&permission);
    emit_available_commands(&cx, &session_id, &state.shared.skills).await;

    responder.respond(LoadSessionResponse::new().modes(modes))
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
    // Fallback cwd for legacy rows that predate the cwd column. The process cwd matches what the
    // agent would use for relative-path resolution if the client picked one of these to load. That
    // is better than refusing to surface them.
    let fallback_cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    let sessions = rows
        .into_iter()
        .map(|summary| {
            let cwd = summary.cwd.unwrap_or_else(|| fallback_cwd.clone());
            let mut info =
                SessionInfo::new(summary.id.to_string(), cwd).updated_at(summary.updated_at);
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

    let runtime = match build_session_runtime(
        &state.shared,
        &state.client_state,
        &state.transport_dead,
        cx.clone(),
        session_id.clone(),
        session_id_str.clone(),
        session_uuid,
        req.cwd.clone(),
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
    let frontend = Arc::clone(&runtime.frontend);
    // History already carries the first user message, so the title is known; push it once now,
    // sharing the flag with the entry so a later prompt won't re-emit it.
    let title_sent = Arc::new(std::sync::atomic::AtomicBool::new(false));
    maybe_emit_session_title(&cx, &session_id, &title_sent, &runtime.messages);
    let entry = SessionEntry {
        runtime: Arc::new(Mutex::new(runtime)),
        cancellation: Arc::new(std::sync::RwLock::new(CancellationToken::new())),
        cancel_pending: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        title_sent,
        permission: permission.clone(),
        frontend,
        session_lock,
    };
    state.sessions.write().await.insert(session_id_str, entry);

    let modes = build_mode_state(&permission);
    emit_available_commands(&cx, &session_id, &state.shared.skills).await;

    responder.respond(ResumeSessionResponse::new().modes(modes))
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
    // targeting it. Briefly lock the runtime to read the registry handle; the in-flight prompt (if
    // any) was just cancelled and will release the lock shortly.
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
            Some(entry) => entry.clone(),
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
    send_session_update(
        &entry.frontend.connection,
        &entry.frontend.session_id,
        SessionUpdate::CurrentModeUpdate(CurrentModeUpdate::new(req.mode_id.clone())),
    );
    responder.respond(SetSessionModeResponse::new())
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
    messages: Conversation,
) -> anyhow::Result<SessionRuntime> {
    let cwd: SharedCwd = Arc::new(std::sync::RwLock::new(cwd_path));
    let permission =
        SharedPermission::new(shared.config.permission, shared.config.enabled_permissions);

    // Shared with the agent (adopted via `set_context_tokens` below) so the frontend can read the
    // current context occupancy when emitting `usage_update`.
    let context_tokens = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let acp_frontend = Arc::new(AcpFrontend::new(
        connection,
        session_id,
        Arc::clone(&cwd),
        client_state.clone(),
        Arc::clone(transport_dead),
        Arc::clone(&context_tokens),
    ));
    let frontend: Arc<dyn Frontend> = acp_frontend.clone();

    let (mut agent, tool_registry) =
        crate::build_session_agent(shared, permission.clone(), frontend, Arc::clone(&cwd)).await?;
    // Point the agent's context counter at the shared atomic and capture its resolved window, so
    // the frontend's `usage_update` reports the same occupancy/size the REPL gauge would.
    agent.set_context_tokens(Arc::clone(&context_tokens));
    acp_frontend.set_context_window(agent.context_usage().1);

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

/// Mirrors `main::resolve_credential` but stays in this module to avoid widening `main`'s
/// visibility for an ACP-only call site.
async fn resolve_credential_for_acp(
    config: &ResolvedConfig,
    token_store: &crate::session::TokenStore,
) -> anyhow::Result<AuthCredential> {
    // Debug-only: when the integration harness sets `MEKA_ACP_MOCK_PROVIDER=1`, `run_acp` swaps in
    // a scripted provider and discards the real one built from this credential. Return a
    // placeholder so the harness needn't seed a credential into the database.
    #[cfg(debug_assertions)]
    if std::env::var("MEKA_ACP_MOCK_PROVIDER").as_deref() == Ok("1") {
        return Ok(AuthCredential::ApiKey("mock-acp-provider".to_string()));
    }

    let Some(profile) = config.active_profile.as_deref() else {
        anyhow::bail!("meka acp requires a configured provider; run `meka provider add <name>`");
    };
    token_store
        .load_provider_credential(profile)
        .await?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "provider profile '{}' has no stored credential; run `meka provider login {}`",
                profile,
                profile
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::PermissionOutcome;

    // `AcpFrontend` itself can't be unit-tested (requires a live `ConnectionTo<Client>`);
    // per-session behaviour is covered end-to-end in `tests/acp.rs`. The pure helpers below are
    // what this unit-test module owns.

    #[test]
    fn test_tool_kind_for_covers_builtins() {
        assert_eq!(tool_kind_for("read_file"), ToolKind::Read);
        assert_eq!(tool_kind_for("edit_file"), ToolKind::Edit);
        assert_eq!(tool_kind_for("write_file"), ToolKind::Edit);
        assert_eq!(tool_kind_for("find_files"), ToolKind::Search);
        assert_eq!(tool_kind_for("search_contents"), ToolKind::Search);
        assert_eq!(tool_kind_for("execute_command"), ToolKind::Execute);
        assert_eq!(tool_kind_for("fetch_url"), ToolKind::Fetch);
        assert_eq!(tool_kind_for("spawn_agent"), ToolKind::Think);
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

    #[test]
    fn test_decode_acp_image_passes_through_within_cap() {
        // The PassThrough path validates size only, so small arbitrary bytes labelled `image/png`
        // round-trip into an `ImageSource` without needing a real PNG.
        let raw = vec![0u8; 128];
        let data = base64::engine::general_purpose::STANDARD.encode(&raw);
        let image = ImageContent::new(data, "image/png".to_string());
        let source = decode_acp_image(&image).expect("decode");
        assert_eq!(source.source_type, "base64");
        assert_eq!(source.media_type, "image/png");
    }

    #[test]
    fn test_decode_acp_image_rejects_oversized() {
        let raw = vec![0u8; crate::image::MAX_IMAGE_RAW_BYTES + 1];
        let data = base64::engine::general_purpose::STANDARD.encode(&raw);
        let image = ImageContent::new(data, "image/png".to_string());
        let error = decode_acp_image(&image).expect_err("should reject oversized");
        assert!(error.contains("too large"), "got: {error}");
    }

    #[test]
    fn test_decode_acp_image_rejects_bad_base64() {
        let image = ImageContent::new("not%%%valid".to_string(), "image/png".to_string());
        assert!(decode_acp_image(&image).is_err());
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
        assert!(tool_locations("web_search", &input, &cwd).is_empty());
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
            tool_call_title("web_search", Some("rust acp")),
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

    /// Image content has no ACP analogue today, so `build_completion_content` collapses it to a
    /// `[image]` text marker. Walks the resulting `ContentBlock::Text` to confirm the literal. This
    /// guards against accidentally swapping in the `ImageContent` ACP variant before the wire
    /// format is wired through end-to-end.
    #[test]
    fn test_build_completion_content_image_falls_back_to_marker() {
        use crate::provider::ImageSource;
        let content = vec![ToolResultContent::Image {
            source: ImageSource {
                source_type: "base64".to_string(),
                media_type: "image/png".to_string(),
                data: "irrelevant".to_string(),
            },
        }];
        let blocks = build_completion_content("read_file", &content, None);
        assert_eq!(blocks.len(), 1);
        let ToolCallContent::Content(chunk) = &blocks[0] else {
            panic!("expected ToolCallContent::Content; got {:?}", blocks[0]);
        };
        let ContentBlock::Text(text) = &chunk.content else {
            panic!("expected ContentBlock::Text; got {:?}", chunk.content);
        };
        assert_eq!(text.text, "[image]");
    }

    #[test]
    fn test_parse_mode_id_covers_all_levels() {
        assert_eq!(parse_mode_id("none"), Some(Permission::None));
        assert_eq!(parse_mode_id("read"), Some(Permission::Read));
        assert_eq!(parse_mode_id("ask"), Some(Permission::Ask));
        assert_eq!(parse_mode_id("write"), Some(Permission::Write));
    }

    #[test]
    fn test_parse_mode_id_rejects_garbage() {
        assert!(parse_mode_id("READ").is_none(), "case-sensitive");
        assert!(parse_mode_id("admin").is_none());
        assert!(parse_mode_id("").is_none());
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
            .try_set(Permission::Write)
            .expect("write enabled");
        assert_eq!(
            build_mode_state(&permission).current_mode_id.0.as_ref(),
            "write"
        );
    }

    #[tokio::test]
    async fn test_slash_to_prompt_text_passes_through_non_slash() {
        let cache = SkillCache::for_root(None);
        let out = slash_to_prompt_text("just a normal prompt".to_string(), &cache, "sid")
            .await
            .expect("ok");
        assert_eq!(out, "just a normal prompt");
    }

    #[tokio::test]
    async fn test_slash_to_prompt_text_passes_through_paste_shaped_input() {
        // A pasted path like `/etc/hosts is a config file` has an invalid skill-name first token
        // (slash inside the name), so the helper must NOT touch it.
        let cache = SkillCache::for_root(None);
        let out = slash_to_prompt_text("/etc/hosts is the config file".to_string(), &cache, "sid")
            .await
            .expect("pass-through");
        assert_eq!(out, "/etc/hosts is the config file");
    }

    #[tokio::test]
    async fn test_slash_to_prompt_text_passes_through_double_slash_comment() {
        // `//foo` parses as name="/foo", which is invalid; pass through.
        let cache = SkillCache::for_root(None);
        let out = slash_to_prompt_text("//comment line".to_string(), &cache, "sid")
            .await
            .expect("pass-through");
        assert_eq!(out, "//comment line");
    }

    #[tokio::test]
    async fn test_slash_to_prompt_text_unknown_but_valid_name_errors() {
        // A clean `/<name>` shape with a syntactically valid skill name that isn't installed:
        // error, since the only realistic source of this shape is a typo'd palette pick.
        let cache = SkillCache::for_root(None);
        let err = slash_to_prompt_text("/nonexistent".to_string(), &cache, "sid")
            .await
            .expect_err("should error");
        assert!(
            matches!(err, SlashInvocationError::SkillNotFound(ref name) if name == "nonexistent")
        );
    }

    #[tokio::test]
    async fn test_slash_to_prompt_text_known_skill_composes_body() {
        // Drop a SKILL.md under a tempdir, point a fresh cache at it.
        let temp = tempfile::tempdir().expect("tempdir");
        let skill_dir = temp.path().join("demo");
        std::fs::create_dir_all(&skill_dir).expect("mkdir skill");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\ndescription: demo skill\n---\nrun ls in ${MEKA_SKILL_DIR}\n",
        )
        .expect("write SKILL.md");

        let cache = SkillCache::for_root(Some(temp.path().to_path_buf()));
        let out = slash_to_prompt_text("/demo only fetch UK news".to_string(), &cache, "sid-xyz")
            .await
            .expect("ok");
        assert!(
            out.starts_with("only fetch UK news\n\n"),
            "extra context must lead: {}",
            out
        );
        assert!(
            out.contains("run ls in ") && out.contains("demo"),
            "body must include the substituted skill dir: {}",
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
        let out = slash_to_prompt_text("/ping".to_string(), &cache, "sid")
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
