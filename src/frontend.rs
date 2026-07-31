//! `Frontend`: the swappable driver for agent output and approval round-trips.
//!
//! `Agent::run_turn` emits its user-facing output (streamed assistant text, thinking blocks,
//! tool-call indicators, todo lists, token usage) and its tool-approval requests through `Arc<dyn
//! Frontend>` instead of calling `render::*` and `std::sync::mpsc` directly. The REPL today is one
//! impl ([`crate::repl::ReplFrontend`]); ACP, a Telegram bridge, or a web UI become additional
//! impls without touching the agent core.
//!
//! This module owns the trait, the event/permission types, and the two UI-agnostic impls
//! ([`SilentFrontend`], [`PermissionForwardingFrontend`]). Concrete UI impls live with their UI
//! (`ReplFrontend` in `crate::repl`, `AcpFrontend` in `crate::acp`) so the abstraction layer never
//! depends on a specific frontend by name.
//!
//! The event-based shape mirrors ACP's `session/update` notification: one channel for every kind
//! of agent-emitted output, discriminated by the [`FrontendEvent`] variant.

// `Mutex` is consumed only by the `#[cfg(test)] mod testing` block below; gating its import
// keeps non-test builds warning-clean now that `ReplFrontend` (the production user of
// `std::sync::Mutex`) has moved into `crate::repl`.
#[cfg(test)]
use std::sync::Mutex;
use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{provider::TokenUsage, tools::todo::TodoItem};

/// Trait the agent loop talks through to surface output and ask the user to approve tool calls.
/// Implementations are responsible for rendering mode, newline spacing, and any inter-event
/// formatting.
#[async_trait]
pub trait Frontend: Send + Sync {
    /// Emit a one-way UI event. Implementations must tolerate any order of events but may assume
    /// `TurnStarted` precedes any per-turn activity and `TurnFinished` closes it.
    async fn emit(&self, event: FrontendEvent);

    /// Round-trip request for user approval of a tool call. Used only when
    /// [`crate::permission::Permission::Ask`] is active. [`PermissionOutcome::Cancelled`] is
    /// distinct from [`PermissionOutcome::Deny`]; it indicates the user cancelled the enclosing
    /// turn (Ctrl+C, `session/cancel`), which ACP will surface later. Today's REPL collapses it to
    /// deny semantics.
    async fn request_permission(&self, request: PermissionRequest) -> PermissionOutcome;

    /// Delegate a file read to whatever filesystem the frontend owns (typically the ACP client's
    /// in-buffer view of the file). `Some(Ok(content))` means the frontend handled it;
    /// `Some(Err(_))` means delegation was attempted and failed; `None` means "no delegate
    /// available, do it locally".
    ///
    /// The file tools route on the [`DelegateFailure`] carried by `Some(Err(_))`, under one rule:
    /// **[`DelegateFailure::UnservablePath`] means the local filesystem is the only route and is
    /// used; anything else means the frontend may own the file, and the operation fails rather
    /// than routing around it.** Which paths a frontend will serve is its own business and differs
    /// between editors, so meka models none of them -- it asks per path and believes the answer.
    ///
    /// `line` and `limit` follow ACP's 1-based line / line-count convention.
    async fn delegate_fs_read(
        &self,
        _path: &Path,
        _line: Option<u32>,
        _limit: Option<u32>,
    ) -> Option<Result<String, FrontendError>> {
        None
    }

    /// Delegate a file write. Same `None` / `Some(Err)` / `Some(Ok)` semantics as
    /// [`Self::delegate_fs_read`].
    async fn delegate_fs_write(
        &self,
        _path: &Path,
        _content: &str,
    ) -> Option<Result<(), FrontendError>> {
        None
    }

    /// Delegate a shell command to the frontend's hosted terminal (e.g. ACP `terminal/*`). Same
    /// `None` / `Some(Err)` / `Some(Ok)` semantics as [`Self::delegate_fs_read`].
    async fn delegate_execute(
        &self,
        _spec: DelegatedExecSpec,
    ) -> Option<Result<DelegatedExecOutput, FrontendError>> {
        None
    }

    /// Returns `true` if the frontend has observed that its client is no longer reachable (e.g. an
    /// ACP client has closed its stdio connection, so every `session/update` notification returns
    /// an error). The agent loop checks this at every loop iteration and short-circuits with
    /// [`crate::error::MekaError::Interrupted`] so it doesn't keep burning provider / MCP cycles
    /// for an audience that's gone away.
    ///
    /// REPL and silent frontends never disconnect in this sense, so the default `false` is correct
    /// for them.
    fn client_disconnected(&self) -> bool {
        false
    }

    /// Handle an MCP `elicitation/create` request: the server asked the user for input (either a
    /// structured form or a URL-consent flow). The frontend is responsible for prompting the user
    /// and returning their response. The default impl declines: the safe behavior when no human
    /// is reachable (non-interactive subcommands, `SilentFrontend`, the test-only
    /// `RecordingFrontend`).
    ///
    /// Called via the task-local installed in `Agent::run_tool`; see
    /// [`crate::mcp::current_session_frontend`]. Concrete impls today:
    /// [`crate::repl::ReplFrontend`] (routes through the REPL thread), `crate::acp::AcpFrontend`
    /// (issues ACP `elicitation/create`, declining when the client doesn't advertise the mode),
    /// and [`PermissionForwardingFrontend`] (hands a sub-agent's elicitation to the parent).
    async fn handle_elicitation(
        &self,
        _prompt: crate::mcp::elicitation::ElicitationPrompt,
    ) -> crate::mcp::elicitation::ElicitationResponse {
        crate::mcp::elicitation::ElicitationResponse::Decline
    }
}

/// Why a delegated operation failed, in the only distinction the routing rule turns on.
///
/// Editors differ in which paths they will serve -- some only the project they have open, some
/// anything absolute -- so meka does not model any editor's rule. It asks, and this is the answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelegateFailure {
    /// The frontend will not serve this path at all (ACP `ResourceNotFound`). It holds no buffer
    /// for the file and never will, so the local filesystem is not a degraded substitute for the
    /// delegate here -- it is the same bytes, and the only route.
    UnservablePath,
    /// Anything else: transport, timeout, an internal error inside the client. The frontend may
    /// well own this file and hold unsaved changes for it, so falling back to the local filesystem
    /// could read stale bytes or overwrite the user's unsaved work.
    Transient,
}

/// Error from a frontend-delegated operation ([`Frontend::delegate_fs_read`],
/// [`Frontend::delegate_fs_write`], [`Frontend::delegate_execute`]). Carries the underlying
/// transport's message in a stringly form so tools can splice it into their `ToolOutput` text
/// without depending on the transport crate, plus the [`DelegateFailure`] the routing rule needs.
#[derive(Debug, Clone)]
pub struct FrontendError {
    message: String,
    failure: DelegateFailure,
}

impl FrontendError {
    /// Construct a [`DelegateFailure::Transient`] error -- the conservative default, since it is
    /// the classification that never routes around the frontend.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            failure: DelegateFailure::Transient,
        }
    }

    /// Construct a [`DelegateFailure::UnservablePath`] error. Only a transport that can tell the
    /// two apart on the wire may call this.
    pub fn unservable_path(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            failure: DelegateFailure::UnservablePath,
        }
    }

    /// Whether the local filesystem is a safe route for this path: true only when the frontend
    /// said it cannot serve the path at all.
    pub fn is_unservable_path(&self) -> bool {
        self.failure == DelegateFailure::UnservablePath
    }
}

impl std::fmt::Display for FrontendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for FrontendError {}

/// Description of the command a delegated `execute_command` should run. The frontend is responsible
/// for spawning the process (via ACP `terminal/create`, an MCP equivalent, etc.) and returning the
/// assembled output via [`DelegatedExecOutput`].
#[derive(Debug, Clone)]
pub struct DelegatedExecSpec {
    /// The executable to run. meka always picks a shell (e.g. `sh` / `powershell.exe`) and passes
    /// the user-supplied command as an argument, so the frontend doesn't need to do its own shell-
    /// quoting.
    pub command: String,
    pub args: Vec<String>,
    /// Process environment to set in addition to whatever the frontend supplies as its baseline.
    /// meka forwards a filtered subset of the agent's env so things like `PATH` / `LANG` are
    /// preserved.
    pub env: Vec<(String, String)>,
    /// Working directory for the spawned process. Almost always `Some(_)`: meka's per-session cwd
    /// snapshot at the call site.
    pub cwd: Option<PathBuf>,
    /// Hard timeout. The frontend should attempt to kill the process and return whatever output
    /// accumulated. `None` defers to the frontend's own default.
    pub timeout: Option<Duration>,
    /// Maximum bytes of output the frontend should retain. The frontend signals truncation via
    /// [`DelegatedExecOutput::truncated`].
    pub output_byte_limit: Option<u64>,
    /// Cancellation token from the agent loop. The frontend may use this to short-circuit
    /// `wait_for_exit` and issue a kill.
    pub cancellation: CancellationToken,
}

/// Output of a delegated execute_command. ACP's `terminal/*` returns one combined output stream; we
/// flatten any stdout/stderr separation into the single [`Self::output`] field. meka's local
/// execute_command renders the same way (stderr is appended to stdout with a separator), so this
/// matches.
#[derive(Debug, Clone)]
pub struct DelegatedExecOutput {
    pub output: String,
    pub exit_code: Option<i32>,
    /// Signal name (e.g. `"SIGTERM"`) when the process was killed. Mutually exclusive with
    /// [`Self::exit_code`] in practice.
    pub signal: Option<String>,
    /// True iff the frontend dropped bytes past [`DelegatedExecSpec::output_byte_limit`].
    pub truncated: bool,
}

/// One-way UI event emitted by the agent loop.
#[derive(Debug, Clone)]
pub enum FrontendEvent {
    /// A new session was created. Carries the session UUID.
    SessionStarted { id: Uuid },
    /// The agent is about to start a turn. REPL uses this to emit the `newline_after_prompt` blank
    /// line.
    TurnStarted,
    /// The agent finished a turn cleanly. REPL uses this to flush any open streaming renderer and
    /// emit the `newline_before_prompt` blank line.
    TurnFinished,
    /// A streamed chunk of assistant text. Multiple deltas concatenate into one logical text run;
    /// any non-text event closes the run.
    AssistantTextDelta(String),
    /// A complete thinking block. Emitted after the provider's `ThinkingComplete` stream event.
    ThinkingBlock {
        content: String,
        /// Provider-opaque signature blob (Claude's extended-thinking signature). Unread today;
        /// kept on the event so future session/load replay can round-trip it without an event
        /// shape change.
        #[allow(dead_code)]
        signature: Option<String>,
    },
    /// A tool call is about to be dispatched. `id` is the `tool_use_id` assigned by the provider;
    /// frontends use it to correlate this announcement with the matching
    /// [`Self::ToolCallCompleted`]. `display_summary` is the agent-resolved primary argument for
    /// display (e.g. the path for `read_file`, the command for `execute_command`), pre-computed via
    /// [`crate::render::resolve_primary_param`] so frontends don't need the tool's JSON Schema to
    /// render the indicator. `None` means "no obvious primary arg". Render the bare tool name.
    ToolCallStarted {
        id: String,
        name: String,
        input: serde_json::Value,
        display_summary: Option<String>,
    },
    /// A previously-announced tool call has finished. Emitted once per tool in source order after
    /// the parallel dispatch settles. The REPL impl ignores this today (tool results render through
    /// the model's next assistant message); the ACP impl translates it to `session/update:
    /// tool_call_update` with `status: completed | failed`.
    ToolCallCompleted {
        id: String,
        /// Tool name (matches the [`Self::ToolCallStarted`] `name`). Lets a frontend format the
        /// output per tool, e.g. wrapping `execute_command` output in a console code block.
        name: String,
        is_error: bool,
        content: Vec<crate::provider::ToolResultContent>,
        /// Tool-specific structured side-channel. `edit_file` / `write_file` populate
        /// [`ToolOutputMetadata::Diff`] so ACP can emit a proper `diff` content block (and Zed can
        /// render its apply-diff UI). `None` for tools that have nothing extra.
        metadata: Option<ToolOutputMetadata>,
    },
    /// The shared todo list changed via the `todo` tool. Emitted by the agent loop after the tool
    /// succeeds and only when the rendered state actually changed; the REPL renders the list and
    /// the agent's per-turn `OutputSpacing` is advanced. `title` is the heading the agent set
    /// for the list.
    TodoListUpdated {
        title: Option<String>,
        items: Vec<TodoItem>,
    },
    /// A sub-agent running under the `spawn_agent` tool call `tool_call_id` did something worth
    /// showing. `summary` is the *whole* rolling activity block, not a delta, because ACP's
    /// `tool_call_update` replaces a tool call's content rather than appending to it.
    ///
    /// Emitted by [`PermissionForwardingFrontend`], which is the only place that knows both the
    /// sub-agent's events and the parent tool call they belong to.
    SubAgentActivity {
        tool_call_id: String,
        summary: String,
    },
    /// End-of-turn token-usage summary.
    TokenUsage(TokenUsage),
    /// User-visible advisory surfaced by the provider layer (e.g. image redaction when the request
    /// body would exceed the API limit). `ReplFrontend` renders via [`crate::render::render_hint`];
    /// `AcpFrontend` forwards as an `AgentMessageChunk` with a `[meka] ` prefix so the editor's
    /// transcript records the side-effect. `SilentFrontend` drops them.
    Notice(crate::provider::Notice),
    /// Incremental progress from an in-flight MCP tool (`notifications/progress`). Routed
    /// per-session via the task-local frontend installed in `Agent::run_tool`. See
    /// [`crate::mcp::current_session_frontend`]. `ReplFrontend` renders an inline status line
    /// (carriage-return overwrite); `AcpFrontend` logs at `info!` today (no protocol primitive
    /// yet). `SilentFrontend` drops them.
    McpProgress(crate::mcp::progress::ProgressUpdate),
}

/// Structured side-channel a tool can attach to its [`crate::tools::ToolOutput`] for frontends that
/// know how to render it. Frontends that don't understand a variant ignore it (the regular
/// `content` text is still the source of truth for the model and the REPL).
#[derive(Debug, Clone)]
pub enum ToolOutputMetadata {
    /// Pre/post file content produced by `edit_file` / `write_file`. `old_text == None` means the
    /// file did not exist before the call (the write created it).
    Diff {
        path: std::path::PathBuf,
        old_text: Option<String>,
        new_text: String,
    },
}

/// Round-trip request for tool-call approval.
#[derive(Debug, Clone)]
pub struct PermissionRequest {
    pub tool_name: String,
    /// The most user-meaningful argument for display in the prompt (e.g. the file path for
    /// `read_file`, the command for `execute_command`). Resolved via
    /// [`crate::render::resolve_primary_param`].
    pub primary_param: Option<String>,
    /// Per-turn cancellation token. ACP frontends race their `session/request_permission`
    /// round-trip against this so a `session/cancel` during an `Ask`-mode prompt resolves promptly
    /// instead of hanging until the client replies.
    pub cancellation: tokio_util::sync::CancellationToken,
}

/// Outcome of a [`Frontend::request_permission`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionOutcome {
    Allow,
    Deny,
    /// The enclosing turn was cancelled while the request was in flight. The ACP frontend surfaces
    /// this as `{outcome: cancelled}`; the REPL collapses it to a deny-shaped tool error.
    Cancelled,
}

/// Frontend wrapper used by sub-agents when the parent is interactive enough to host permission
/// prompts. Streaming output (text, thinking, todos, token usage) is dropped; sub-agents' final
/// reports flow back through the parent's `spawn_agent` tool result, not through this frontend.
/// The exceptions are:
///
/// - `Notice` — provider-side advisories the user should still see, e.g. a redaction during a
///   sub-agent's turn.
/// - `request_permission` and `handle_elicitation` — round-trips forwarded so the user is asked in
///   their original UI (REPL approval line, ACP `session/request_permission` /
///   `elicitation/create`).
/// - `ToolCallStarted` — not forwarded as-is, but rolled up into
///   [`FrontendEvent::SubAgentActivity`] against the parent's `spawn_agent` call so a long
///   delegated task shows its progress instead of an opaque spinner.
///
/// Constructed in [`crate::tools::subagent::SpawnAgentTool`] with the parent agent's frontend as
/// the delegate.
pub struct PermissionForwardingFrontend {
    delegate: Arc<dyn Frontend>,
    /// The parent's `tool_use_id` for the `spawn_agent` call this sub-agent is running under, when
    /// one is in scope. `None` outside a tool call (tests, direct construction), which disables
    /// activity forwarding rather than guessing at a correlation id.
    tool_call_id: Option<String>,
    /// Rolling record of what the sub-agent has done, oldest first, capped at
    /// [`Self::MAX_ACTIVITY_LINES`].
    activity: std::sync::Mutex<VecDeque<String>>,
}

impl PermissionForwardingFrontend {
    /// How many activity lines the parent tool call shows. The whole block is resent on every
    /// update (ACP replaces content), so this bounds per-update payload as well as height.
    const MAX_ACTIVITY_LINES: usize = 20;

    pub fn new(delegate: Arc<dyn Frontend>, tool_call_id: Option<String>) -> Self {
        Self {
            delegate,
            tool_call_id,
            activity: std::sync::Mutex::new(VecDeque::new()),
        }
    }

    /// Append `line` and return the whole block to send.
    fn record_activity(&self, line: String) -> Option<String> {
        let mut activity = self.activity.lock().ok()?;
        if activity.len() == Self::MAX_ACTIVITY_LINES {
            activity.pop_front();
        }
        activity.push_back(line);
        Some(
            activity
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join("\n"),
        )
    }
}

#[async_trait]
impl Frontend for PermissionForwardingFrontend {
    async fn emit(&self, event: FrontendEvent) {
        match event {
            // Provider advisories about the sub-agent's request belong in the user's primary UI.
            FrontendEvent::Notice(_) => self.delegate.emit(event).await,
            // Roll the sub-agent's tool calls up into the parent's `spawn_agent` tool call, so a
            // long-running sub-agent shows what it is doing instead of an opaque spinner. Only the
            // call being *started* is recorded: it answers "where is it now", which is the
            // question an unattended run leaves open, and results still arrive in the report.
            FrontendEvent::ToolCallStarted {
                name,
                display_summary,
                ..
            } => {
                // Without a parent call there is nothing to attach the activity to, so it is
                // dropped rather than sent against a guessed correlation id.
                let Some(tool_call_id) = self.tool_call_id.clone() else {
                    return;
                };
                let line = match display_summary {
                    Some(summary) => format!("{}: {}", name, summary),
                    None => name,
                };
                if let Some(summary) = self.record_activity(line) {
                    self.delegate
                        .emit(FrontendEvent::SubAgentActivity {
                            tool_call_id,
                            summary,
                        })
                        .await;
                }
            }
            // A nested sub-agent's activity is already summarised as a `spawn_agent` line in this
            // sub-agent's own record; forwarding it too would have two writers fighting over one
            // tool call's content.
            FrontendEvent::SubAgentActivity { .. } => {}
            // Everything else (text deltas, thinking, tool results, todos, token usage, session
            // lifecycle) is sub-agent chrome the user shouldn't see.
            _ => {}
        }
    }

    fn client_disconnected(&self) -> bool {
        // Sub-agents must observe the parent's disconnect so their own run_turn loop
        // short-circuits; without this forward, a sub-agent under a dropped ACP connection
        // keeps burning provider tokens.
        self.delegate.client_disconnected()
    }

    async fn request_permission(&self, request: PermissionRequest) -> PermissionOutcome {
        self.delegate.request_permission(request).await
    }

    async fn delegate_fs_read(
        &self,
        path: &Path,
        line: Option<u32>,
        limit: Option<u32>,
    ) -> Option<Result<String, FrontendError>> {
        self.delegate.delegate_fs_read(path, line, limit).await
    }

    async fn delegate_fs_write(
        &self,
        path: &Path,
        content: &str,
    ) -> Option<Result<(), FrontendError>> {
        self.delegate.delegate_fs_write(path, content).await
    }

    async fn delegate_execute(
        &self,
        spec: DelegatedExecSpec,
    ) -> Option<Result<DelegatedExecOutput, FrontendError>> {
        self.delegate.delegate_execute(spec).await
    }

    /// Forwarded for the same reason as [`Self::request_permission`]: an MCP server called from a
    /// sub-agent needs to reach the same human, and the trait default would decline on their
    /// behalf without ever asking.
    async fn handle_elicitation(
        &self,
        prompt: crate::mcp::elicitation::ElicitationPrompt,
    ) -> crate::mcp::elicitation::ElicitationResponse {
        self.delegate.handle_elicitation(prompt).await
    }
}

/// Fully-silent frontend: drops every emit and denies every permission request. Used by tests and
/// `meka tools list`'s reference registry. Both want a frontend that never reaches out to a user.
/// Sub-agents use [`PermissionForwardingFrontend`] instead so their permission prompts surface in
/// the parent's UI.
pub struct SilentFrontend;

#[async_trait]
impl Frontend for SilentFrontend {
    async fn emit(&self, _event: FrontendEvent) {}

    async fn request_permission(&self, _request: PermissionRequest) -> PermissionOutcome {
        // No human to ask: safest default.
        PermissionOutcome::Deny
    }
}

/// Test-only frontend that records every event it receives. Available to the rest of the crate's
/// test suite via `crate::frontend::testing::RecordingFrontend`.
#[cfg(test)]
pub mod testing {
    use super::*;

    pub struct RecordingFrontend {
        events: Mutex<Vec<FrontendEvent>>,
        permission_response: Mutex<PermissionOutcome>,
    }

    impl RecordingFrontend {
        pub fn new() -> Self {
            Self {
                events: Mutex::new(Vec::new()),
                permission_response: Mutex::new(PermissionOutcome::Allow),
            }
        }

        pub fn with_permission(response: PermissionOutcome) -> Self {
            Self {
                events: Mutex::new(Vec::new()),
                permission_response: Mutex::new(response),
            }
        }

        pub fn events(&self) -> Vec<FrontendEvent> {
            self.events.lock().unwrap().clone()
        }
    }

    impl Default for RecordingFrontend {
        fn default() -> Self {
            Self::new()
        }
    }

    #[async_trait]
    impl Frontend for RecordingFrontend {
        async fn emit(&self, event: FrontendEvent) {
            self.events.lock().unwrap().push(event);
        }

        async fn request_permission(&self, _request: PermissionRequest) -> PermissionOutcome {
            self.permission_response.lock().unwrap().clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{testing::RecordingFrontend, *};

    #[tokio::test]
    async fn test_silent_frontend_emit_is_no_op_and_does_not_panic() {
        let frontend = SilentFrontend;
        frontend.emit(FrontendEvent::TurnStarted).await;
        frontend
            .emit(FrontendEvent::AssistantTextDelta("hello".to_string()))
            .await;
        frontend.emit(FrontendEvent::TurnFinished).await;
    }

    #[tokio::test]
    async fn test_silent_frontend_request_permission_denies() {
        let frontend = SilentFrontend;
        let outcome = frontend
            .request_permission(PermissionRequest {
                tool_name: "read_file".to_string(),
                primary_param: Some("/tmp/foo".to_string()),
                cancellation: tokio_util::sync::CancellationToken::new(),
            })
            .await;
        assert_eq!(outcome, PermissionOutcome::Deny);
    }

    #[tokio::test]
    async fn test_recording_frontend_records_events_in_order() {
        let frontend = RecordingFrontend::new();
        frontend.emit(FrontendEvent::TurnStarted).await;
        frontend
            .emit(FrontendEvent::AssistantTextDelta("hi".to_string()))
            .await;
        frontend.emit(FrontendEvent::TurnFinished).await;
        let events = frontend.events();
        assert_eq!(events.len(), 3);
        assert!(matches!(events[0], FrontendEvent::TurnStarted));
        assert!(matches!(events[1], FrontendEvent::AssistantTextDelta(ref s) if s == "hi"));
        assert!(matches!(events[2], FrontendEvent::TurnFinished));
    }

    /// Locks in the contract for the `display_summary` field on
    /// [`FrontendEvent::ToolCallStarted`]: the agent loop is expected to pre-resolve the primary
    /// argument via [`crate::render::resolve_primary_param`] and ship the resulting `String` (or
    /// `None`) on the event. Frontends rely on this so they never need the tool's JSON Schema
    /// themselves.
    ///
    /// This test exercises both the helper that the agent calls and the event shape that carries
    /// the result, so a future refactor that changes either side is caught here. End-to-end
    /// emission from `Agent::run_turn` is covered by `tests/acp.rs`.
    #[tokio::test]
    async fn test_tool_call_started_carries_resolved_display_summary() {
        let recorder = RecordingFrontend::new();
        let input = serde_json::json!({"path": "/etc/hosts"});
        let display_summary = crate::render::resolve_primary_param("read_file", &input, None);
        assert_eq!(display_summary.as_deref(), Some("/etc/hosts"));
        recorder
            .emit(FrontendEvent::ToolCallStarted {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
                input: input.clone(),
                display_summary: display_summary.clone(),
            })
            .await;
        let events = recorder.events();
        assert_eq!(events.len(), 1);
        match &events[0] {
            FrontendEvent::ToolCallStarted {
                display_summary, ..
            } => {
                assert_eq!(display_summary.as_deref(), Some("/etc/hosts"));
            }
            other => panic!("expected ToolCallStarted; got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_recording_frontend_returns_configured_permission_outcome() {
        let frontend = RecordingFrontend::with_permission(PermissionOutcome::Deny);
        let outcome = frontend
            .request_permission(PermissionRequest {
                tool_name: "execute_command".to_string(),
                primary_param: Some("rm -rf /".to_string()),
                cancellation: tokio_util::sync::CancellationToken::new(),
            })
            .await;
        assert_eq!(outcome, PermissionOutcome::Deny);
    }

    #[tokio::test]
    async fn test_permission_forwarding_frontend_drops_sub_agent_chrome() {
        // Sub-agent chrome (text, lifecycle, tool indicators) must NOT bubble up to the parent's
        // UI; the sub-agent's report flows back via the spawn_agent tool result instead.
        let recorder = Arc::new(RecordingFrontend::new());
        let delegate: Arc<dyn Frontend> = recorder.clone();
        let forwarder = PermissionForwardingFrontend::new(delegate, None);

        forwarder.emit(FrontendEvent::TurnStarted).await;
        forwarder
            .emit(FrontendEvent::AssistantTextDelta("ignored".into()))
            .await;

        assert!(
            recorder.events().is_empty(),
            "sub-agent chrome must not forward to the delegate",
        );
    }

    #[tokio::test]
    async fn test_permission_forwarding_frontend_rolls_up_sub_agent_tool_calls() {
        let recorder = Arc::new(RecordingFrontend::new());
        let delegate: Arc<dyn Frontend> = recorder.clone();
        let forwarder = PermissionForwardingFrontend::new(delegate, Some("toolu_parent".into()));

        for (name, summary) in [
            ("read_file", Some("/etc/hosts".to_string())),
            ("find_files", Some("**/*.rs".to_string())),
            ("todo", None),
        ] {
            forwarder
                .emit(FrontendEvent::ToolCallStarted {
                    id: format!("toolu_{}", name),
                    name: name.to_string(),
                    input: serde_json::json!({}),
                    display_summary: summary,
                })
                .await;
        }

        let events = recorder.events();
        assert_eq!(events.len(), 3, "one update per sub-agent tool call");
        // Each update carries the whole block, not a delta: ACP replaces tool call content.
        match &events[2] {
            FrontendEvent::SubAgentActivity {
                tool_call_id,
                summary,
            } => {
                assert_eq!(tool_call_id, "toolu_parent");
                assert_eq!(summary, "read_file: /etc/hosts\nfind_files: **/*.rs\ntodo");
            }
            other => panic!("expected SubAgentActivity; got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_permission_forwarding_frontend_caps_activity_lines() {
        let recorder = Arc::new(RecordingFrontend::new());
        let delegate: Arc<dyn Frontend> = recorder.clone();
        let forwarder = PermissionForwardingFrontend::new(delegate, Some("toolu_parent".into()));

        let total = PermissionForwardingFrontend::MAX_ACTIVITY_LINES + 5;
        for i in 0..total {
            forwarder
                .emit(FrontendEvent::ToolCallStarted {
                    id: format!("toolu_{}", i),
                    name: "read_file".to_string(),
                    input: serde_json::json!({}),
                    display_summary: Some(format!("/file{}", i)),
                })
                .await;
        }

        let events = recorder.events();
        let FrontendEvent::SubAgentActivity { summary, .. } = events.last().expect("an event")
        else {
            panic!("expected SubAgentActivity");
        };
        let lines: Vec<&str> = summary.lines().collect();
        assert_eq!(
            lines.len(),
            PermissionForwardingFrontend::MAX_ACTIVITY_LINES
        );
        assert_eq!(
            lines[0],
            format!(
                "read_file: /file{}",
                total - PermissionForwardingFrontend::MAX_ACTIVITY_LINES
            ),
            "the oldest lines are dropped, not the newest"
        );
    }

    #[tokio::test]
    async fn test_permission_forwarding_frontend_without_tool_call_id_stays_silent() {
        // Outside a tool call there is nothing to correlate the activity with, so it is dropped
        // rather than sent against a guessed id.
        let recorder = Arc::new(RecordingFrontend::new());
        let delegate: Arc<dyn Frontend> = recorder.clone();
        let forwarder = PermissionForwardingFrontend::new(delegate, None);

        forwarder
            .emit(FrontendEvent::ToolCallStarted {
                id: "toolu_1".into(),
                name: "read_file".into(),
                input: serde_json::json!({}),
                display_summary: Some("/etc/hosts".into()),
            })
            .await;

        assert!(recorder.events().is_empty());
    }

    #[tokio::test]
    async fn test_permission_forwarding_frontend_drops_nested_sub_agent_activity() {
        // A nested sub-agent's roll-up must not reach the parent: it already appears as a
        // `spawn_agent` line in this level's own record, and two writers on one tool call's
        // content would overwrite each other.
        let recorder = Arc::new(RecordingFrontend::new());
        let delegate: Arc<dyn Frontend> = recorder.clone();
        let forwarder = PermissionForwardingFrontend::new(delegate, Some("toolu_parent".into()));

        forwarder
            .emit(FrontendEvent::SubAgentActivity {
                tool_call_id: "toolu_nested".into(),
                summary: "read_file: /deep".into(),
            })
            .await;

        assert!(recorder.events().is_empty());
    }

    /// Notices are the one event that *does* forward through `PermissionForwardingFrontend`. Image
    /// redaction during a sub-agent's provider call is a side-effect the *user* needs to see; the
    /// sub-agent's report has no place to surface it, and silent redaction without operator
    /// awareness is exactly the bypass this whole refactor is meant to close.
    #[tokio::test]
    async fn test_permission_forwarding_frontend_forwards_notice() {
        let recorder = Arc::new(RecordingFrontend::new());
        let delegate: Arc<dyn Frontend> = recorder.clone();
        let forwarder = PermissionForwardingFrontend::new(delegate, None);
        forwarder
            .emit(FrontendEvent::Notice(crate::provider::Notice::info(
                "redacted 2 images",
            )))
            .await;
        let events = recorder.events();
        assert_eq!(events.len(), 1, "exactly one event should forward");
        match &events[0] {
            FrontendEvent::Notice(notice) => {
                assert_eq!(notice.text, "redacted 2 images");
                assert_eq!(notice.level, crate::provider::NoticeLevel::Info);
            }
            other => panic!("expected Notice, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_permission_forwarding_frontend_delegates_request_permission() {
        let delegate: Arc<dyn Frontend> =
            Arc::new(RecordingFrontend::with_permission(PermissionOutcome::Allow));
        let forwarder = PermissionForwardingFrontend::new(delegate, None);
        let outcome = forwarder
            .request_permission(PermissionRequest {
                tool_name: "write_file".into(),
                primary_param: Some("/tmp/foo".into()),
                cancellation: tokio_util::sync::CancellationToken::new(),
            })
            .await;
        assert_eq!(outcome, PermissionOutcome::Allow);
    }

    #[tokio::test]
    async fn test_silent_frontend_default_delegate_methods_return_none() {
        // Default impls signal "no delegate available, do it locally".
        let frontend = SilentFrontend;
        assert!(
            frontend
                .delegate_fs_read(Path::new("/tmp/x"), None, None)
                .await
                .is_none()
        );
        assert!(
            frontend
                .delegate_fs_write(Path::new("/tmp/x"), "hi")
                .await
                .is_none()
        );
        assert!(
            frontend
                .delegate_execute(DelegatedExecSpec {
                    command: "true".to_string(),
                    args: Vec::new(),
                    env: Vec::new(),
                    cwd: None,
                    timeout: None,
                    output_byte_limit: None,
                    cancellation: tokio_util::sync::CancellationToken::new(),
                })
                .await
                .is_none()
        );
    }

    /// Test fixture that records what arguments each delegate method was called with, and lets the
    /// test pick the response.
    pub(super) struct DelegatingRecorder {
        pub fs_reads: Mutex<Vec<PathBuf>>,
        pub fs_writes: Mutex<Vec<(PathBuf, String)>>,
        pub execs: Mutex<Vec<DelegatedExecSpec>>,
        pub fs_read_response: Mutex<Option<Result<String, FrontendError>>>,
        pub fs_write_response: Mutex<Option<Result<(), FrontendError>>>,
        pub exec_response: Mutex<Option<Result<DelegatedExecOutput, FrontendError>>>,
    }

    impl DelegatingRecorder {
        fn new() -> Self {
            Self {
                fs_reads: Mutex::new(Vec::new()),
                fs_writes: Mutex::new(Vec::new()),
                execs: Mutex::new(Vec::new()),
                fs_read_response: Mutex::new(Some(Ok("from-delegate".to_string()))),
                fs_write_response: Mutex::new(Some(Ok(()))),
                exec_response: Mutex::new(Some(Ok(DelegatedExecOutput {
                    output: "delegate-out".to_string(),
                    exit_code: Some(0),
                    signal: None,
                    truncated: false,
                }))),
            }
        }
    }

    #[async_trait]
    impl Frontend for DelegatingRecorder {
        async fn emit(&self, _event: FrontendEvent) {}

        async fn request_permission(&self, _request: PermissionRequest) -> PermissionOutcome {
            PermissionOutcome::Allow
        }

        async fn delegate_fs_read(
            &self,
            path: &Path,
            _line: Option<u32>,
            _limit: Option<u32>,
        ) -> Option<Result<String, FrontendError>> {
            self.fs_reads.lock().unwrap().push(path.to_path_buf());
            self.fs_read_response.lock().unwrap().take()
        }

        async fn delegate_fs_write(
            &self,
            path: &Path,
            content: &str,
        ) -> Option<Result<(), FrontendError>> {
            self.fs_writes
                .lock()
                .unwrap()
                .push((path.to_path_buf(), content.to_string()));
            self.fs_write_response.lock().unwrap().take()
        }

        async fn delegate_execute(
            &self,
            spec: DelegatedExecSpec,
        ) -> Option<Result<DelegatedExecOutput, FrontendError>> {
            self.execs.lock().unwrap().push(spec);
            self.exec_response.lock().unwrap().take()
        }
    }

    #[tokio::test]
    async fn test_permission_forwarding_frontend_forwards_fs_read() {
        let recorder = Arc::new(DelegatingRecorder::new());
        let delegate: Arc<dyn Frontend> = recorder.clone();
        let forwarder = PermissionForwardingFrontend::new(delegate, None);
        let outcome = forwarder
            .delegate_fs_read(Path::new("/tmp/sub.txt"), None, None)
            .await
            .expect("delegate result");
        assert_eq!(outcome.expect("ok"), "from-delegate");
        assert_eq!(recorder.fs_reads.lock().unwrap().as_slice(), &[
            PathBuf::from("/tmp/sub.txt")
        ],);
    }

    #[tokio::test]
    async fn test_permission_forwarding_frontend_forwards_fs_write() {
        let recorder = Arc::new(DelegatingRecorder::new());
        let delegate: Arc<dyn Frontend> = recorder.clone();
        let forwarder = PermissionForwardingFrontend::new(delegate, None);
        forwarder
            .delegate_fs_write(Path::new("/tmp/sub.txt"), "hi from sub-agent")
            .await
            .expect("delegate result")
            .expect("ok");
        let recorded = recorder.fs_writes.lock().unwrap().clone();
        assert_eq!(recorded, vec![(
            PathBuf::from("/tmp/sub.txt"),
            "hi from sub-agent".to_string()
        )]);
    }

    #[tokio::test]
    async fn test_permission_forwarding_frontend_forwards_execute() {
        let recorder = Arc::new(DelegatingRecorder::new());
        let delegate: Arc<dyn Frontend> = recorder.clone();
        let forwarder = PermissionForwardingFrontend::new(delegate, None);
        let spec = DelegatedExecSpec {
            command: "ls".to_string(),
            args: vec!["-la".to_string()],
            env: Vec::new(),
            cwd: Some(PathBuf::from("/tmp")),
            timeout: None,
            output_byte_limit: None,
            cancellation: tokio_util::sync::CancellationToken::new(),
        };
        let outcome = forwarder
            .delegate_execute(spec)
            .await
            .expect("delegate result")
            .expect("ok");
        assert_eq!(outcome.output, "delegate-out");
        assert_eq!(recorder.execs.lock().unwrap().len(), 1);
    }
}
