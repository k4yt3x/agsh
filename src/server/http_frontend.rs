//! `HttpFrontend`, the `meka serve` impl of [`crate::frontend::Frontend`].
//!
//! Blocking mode buffers every emitted event into the turn's recorder; mid-turn pause primitives
//! (permission approval, MCP elicitation) short-circuit to their safe defaults (`Deny`,
//! `Decline`) and append a diagnostic `Notice` so the caller can detect the misconfiguration.
//!
//! Streaming mode (`stream: true`) additionally publishes translated `SseEvent`s on a per-turn
//! `broadcast::Sender`, and the same pause primitives park on a `oneshot::Receiver` until the
//! client POSTs to `/v1/sessions/{id}/responses/{request_id}`.
//!
//! The HTTP API deliberately omits frontend-tool delegation (`delegate_fs_read` / `_fs_write` /
//! `_execute`). See the HTTP API docs. The
//! `Frontend` trait defaults already return `None`, which is the correct behaviour (the agent
//! falls back to local I/O).

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
    time::Duration,
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, oneshot};

use super::sse::{EventIdGenerator, SseEvent, SseEventType, translate};
use crate::{
    frontend::{Frontend, FrontendEvent, PermissionOutcome, PermissionRequest},
    mcp::elicitation::{ElicitationPrompt, ElicitationResponse},
    provider::Notice,
};

/// 60s timeout matching MCP elicitation. Mid-turn `permission_required` / `elicitation_required`
/// requests time out after this duration and resolve to their safe defaults (Deny / Decline).
const MID_TURN_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// How often the parked `request_permission` poll checks whether the SSE consumer has
/// disconnected. `tokio::sync::broadcast::Sender` has no async "wait for subscriber count
/// change" primitive, so we poll `client_disconnected()` on a short interval. 500ms is fast
/// enough to feel instant to a human operator while consuming negligible CPU.
const DISCONNECT_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Frontend impl bound to one in-flight turn. Constructed by the turn handler immediately
/// before calling `Agent::run_turn`; dropped after the handler reads the recorded events out of
/// it to assemble the JSON response body.
///
/// One HttpFrontend per session. The blocking-mode recorder is a [`Mutex`] around
/// [`Recorder`]; in streaming mode a per-turn `tokio::sync::broadcast` channel is installed
/// on top via [`Self::install_stream`], and `emit` fans events out to both the recorder and
/// the channel.
pub struct HttpFrontend {
    recorder: Mutex<Recorder>,
    /// Per-turn streaming channel. Set by the turn handler before calling `run_turn` (via
    /// [`Self::install_stream`]) and cleared after (via [`Self::clear_stream`]). When `Some`,
    /// every emitted event is translated into an `SseEvent` and published on the broadcast.
    /// `None` means blocking-mode: events go only to the recorder.
    stream: Mutex<Option<StreamSink>>,
    /// In-memory parking lot for mid-turn pause primitives (`request_permission` and
    /// `handle_elicitation`). The HTTP turn handler emits an SSE event with the `request_id`,
    /// then `POST /v1/sessions/{id}/responses/{request_id}` pushes the resolution through the
    /// matching oneshot.
    pending: Arc<Mutex<HashMap<String, PermissionPending>>>,
    /// Per-session capabilities, declared at session creation. Controls SSE event filtering
    /// (`supports_reasoning_stream`) and whether a gated tool parks for approval or is denied
    /// outright (`supports_permission_prompts`).
    capabilities: SessionCapabilities,
    /// Sticky `allow_always` set: tools for which the client has chosen "always allow" in this
    /// session short-circuit `request_permission` to `Allow` without ever re-emitting the SSE
    /// pause event. Mirrors `AcpFrontend.always_allowed`. Per-session, never persisted.
    always_allowed: Mutex<HashSet<String>>,
    /// Symmetric `deny_always` set. Tools the client has chosen "always deny" for short-circuit
    /// to `Deny`.
    never_allowed: Mutex<HashSet<String>>,
}

/// Per-session capabilities flags declared at create time. Defaults match the bot/bridge use
/// case (server handles everything locally; SSE clients get assistant text + tool calls but not
/// thinking deltas). See the HTTP API docs § "Capabilities".
///
/// `Serialize` / `Deserialize` are derived so the value can be persisted on the session row and
/// re-hydrated by `reattach::ensure_session_loaded` when a GC-evicted session is reconstructed.
/// `ToSchema` is derived so the field can ride on `SessionResponse` in the OpenAPI spec.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(default)]
pub struct SessionCapabilities {
    /// When `true`, the SSE stream includes `thinking.delta` events for extended-thinking
    /// content. Default `false` so chat-transcript clients (Telegram bridges, etc.) don't
    /// surface reasoning text inline.
    pub supports_reasoning_stream: bool,
    /// When `false`, a mid-turn permission request is denied immediately with a notice instead of
    /// parking on the SSE channel for [`MID_TURN_REQUEST_TIMEOUT`].
    ///
    /// Streaming mode otherwise assumes the consumer can answer a prompt, which is wrong for a
    /// service-to-service client: it wants streaming for liveness on a long turn and has no
    /// interface to put an approval in front of, so an `ask`-mode session stalls a full minute per
    /// gated call and then denies anyway. That reads as a hang rather than a misconfiguration.
    /// Setting this `false` gets blocking mode's behaviour (immediate deny plus an explanatory
    /// notice), which is the same outcome without the stall and legible in the response rather
    /// than only in the timing.
    ///
    /// Defaults to `true`, so existing clients and sessions persisted before this field existed
    /// keep parking as before.
    pub supports_permission_prompts: bool,
}

impl Default for SessionCapabilities {
    fn default() -> Self {
        Self {
            supports_reasoning_stream: false,
            supports_permission_prompts: true,
        }
    }
}

#[derive(Clone)]
struct StreamSink {
    sender: broadcast::Sender<SseEvent>,
    ids: Arc<EventIdGenerator>,
}

/// One parked permission request. Carries `tool_name` so the resolve handler can record sticky
/// "always allow / deny" decisions against the right key.
pub struct PermissionPending {
    pub sender: oneshot::Sender<PermissionOutcome>,
    pub tool_name: String,
}

/// Outcome carried by `POST /responses/{request_id}` for permission resolution. `*_always`
/// records the sticky decision before unblocking the agent.
#[derive(Debug, Clone, Copy)]
pub enum PermissionResolution {
    Allow,
    AllowAlways,
    Deny,
    DenyAlways,
}

/// What a blocking-mode turn collects on its way to producing the JSON response. The turn
/// handler reads this back after `run_turn` returns.
pub type Recorder = Vec<FrontendEvent>;

impl HttpFrontend {
    pub fn new() -> Self {
        Self::with_capabilities(SessionCapabilities::default())
    }

    pub fn with_capabilities(capabilities: SessionCapabilities) -> Self {
        Self {
            recorder: Mutex::new(Recorder::default()),
            stream: Mutex::new(None),
            pending: Arc::new(Mutex::new(HashMap::new())),
            capabilities,
            always_allowed: Mutex::new(HashSet::new()),
            never_allowed: Mutex::new(HashSet::new()),
        }
    }

    fn is_always_allowed(&self, tool_name: &str) -> bool {
        super::poisoned::lock(&self.always_allowed, "http_frontend::is_always_allowed")
            .contains(tool_name)
    }

    fn is_never_allowed(&self, tool_name: &str) -> bool {
        super::poisoned::lock(&self.never_allowed, "http_frontend::is_never_allowed")
            .contains(tool_name)
    }

    fn remember_allow(&self, tool_name: &str) {
        super::poisoned::lock(&self.always_allowed, "http_frontend::remember_allow")
            .insert(tool_name.to_string());
    }

    fn remember_deny(&self, tool_name: &str) {
        super::poisoned::lock(&self.never_allowed, "http_frontend::remember_deny")
            .insert(tool_name.to_string());
    }

    /// True when an SSE consumer is currently attached. Drives the mid-turn-pause branch
    /// selection: streaming → park in `pending`, blocking → short-circuit to safe default.
    fn is_streaming(&self) -> bool {
        let guard = super::poisoned::lock(&self.stream, "http_frontend::is_streaming");
        guard.is_some()
    }

    /// Resolve a pending mid-turn permission request by `request_id`. Returns true iff the entry
    /// existed and the variant matched. Called by `POST /v1/sessions/{id}/responses/{request_id}`.
    /// `*Always` resolutions also record a sticky decision keyed on the tool name so the next
    /// `request_permission` for the same tool short-circuits without re-emitting an SSE pause.
    pub fn resolve_permission(&self, request_id: &str, resolution: PermissionResolution) -> bool {
        let entry = {
            let mut guard =
                super::poisoned::lock(&self.pending, "http_frontend::resolve_permission");
            guard.remove(request_id)
        };
        match entry {
            Some(pending) => {
                let outcome = match resolution {
                    PermissionResolution::Allow => PermissionOutcome::Allow,
                    PermissionResolution::AllowAlways => {
                        self.remember_allow(&pending.tool_name);
                        PermissionOutcome::Allow
                    }
                    PermissionResolution::Deny => PermissionOutcome::Deny,
                    PermissionResolution::DenyAlways => {
                        self.remember_deny(&pending.tool_name);
                        PermissionOutcome::Deny
                    }
                };
                pending.sender.send(outcome).is_ok()
            }
            None => false,
        }
    }

    /// Swap the recorder out for an empty one and return what was collected. Called by the
    /// turn handler after `run_turn` returns; the per-session `HttpFrontend` lives across
    /// turns so consuming `self` isn't an option.
    pub fn drain(&self) -> Recorder {
        let mut guard = super::poisoned::lock(&self.recorder, "http_frontend::drain");
        std::mem::take(&mut *guard)
    }

    /// Install a broadcast sink so subsequent `emit()` calls publish translated SSE events on
    /// it (in addition to recording into the blocking-mode recorder). Returns a `Receiver` the
    /// turn handler subscribes to *before* the broadcast is installed, so no events between
    /// install and first subscribe are lost.
    pub fn install_stream(
        &self,
        capacity: usize,
    ) -> (broadcast::Receiver<SseEvent>, Arc<EventIdGenerator>) {
        let (sender, receiver) = broadcast::channel::<SseEvent>(capacity);
        let ids = Arc::new(EventIdGenerator::default());
        let sink = StreamSink {
            sender,
            ids: ids.clone(),
        };
        let mut guard = super::poisoned::lock(&self.stream, "http_frontend::install_stream");
        *guard = Some(sink);
        (receiver, ids)
    }

    /// Tear down the broadcast sink. Subsequent `emit()` calls go back to recording into the
    /// blocking-mode recorder only. Called by the streaming turn handler after `run_turn`
    /// returns.
    pub fn clear_stream(&self) {
        let mut guard = super::poisoned::lock(&self.stream, "http_frontend::clear_stream");
        *guard = None;
    }

    /// Drop SSE events the per-session capabilities don't enable. Currently only the
    /// `thinking.delta` event is gated (clients opt in via
    /// `capabilities.supports_reasoning_stream`). Returns true if the event should reach the
    /// broadcast.
    fn event_passes_capability_filter(&self, event: &FrontendEvent) -> bool {
        match event {
            FrontendEvent::ThinkingBlock { .. } => self.capabilities.supports_reasoning_stream,
            _ => true,
        }
    }

    /// Surface a `warn`-level diagnostic notice from a safe-default short-circuit (e.g.
    /// auto-denied Ask-mode permission check, auto-declined MCP elicitation). The notice ends
    /// up in *both* sinks (recorder for blocking-mode JSON, broadcast for SSE).
    async fn record_warn_notice(&self, text: String) {
        self.emit(FrontendEvent::Notice(Notice::warn(text))).await;
    }
}

impl Default for HttpFrontend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Frontend for HttpFrontend {
    async fn emit(&self, event: FrontendEvent) {
        // Push to the broadcast (if streaming) BEFORE recording, so a slow blocking-mode Mutex
        // can't delay live subscribers.
        //
        // The stream lock is held across the entire broadcast block (`ids.next()` + `send`) so
        // concurrent emitters can't reorder monotonic ids. `broadcast::Sender::send` is
        // synchronous, so there's no await-under-lock hazard.
        {
            let guard = super::poisoned::lock(&self.stream, "http_frontend::emit_stream");
            if let Some(sink) = guard.as_ref()
                && self.event_passes_capability_filter(&event)
                && let Some((event_type, data)) = translate(event.clone(), self.capabilities)
            {
                let sse = SseEvent {
                    id: sink.ids.next(),
                    event_type,
                    data,
                };
                // `send` returns Err only when there are no subscribers: that means the SSE
                // client has disconnected. The recorder still gets the event so the turn
                // handler can produce the blocking-mode JSON fallback (or in the streaming
                // case, just discard the events after run_turn returns).
                let _ = sink.sender.send(sse);
            }
        }

        let mut guard = super::poisoned::lock(&self.recorder, "http_frontend::emit_record");
        guard.push(event);
    }

    async fn request_permission(&self, request: PermissionRequest) -> PermissionOutcome {
        // Honour sticky decisions recorded earlier this session; they short-circuit before any
        // SSE pause event so the client never sees the same tool prompted twice.
        if self.is_always_allowed(&request.tool_name) {
            return PermissionOutcome::Allow;
        }
        if self.is_never_allowed(&request.tool_name) {
            return PermissionOutcome::Deny;
        }

        if !self.is_streaming() {
            // Blocking mode: no SSE channel to ask through. Auto-deny and surface the
            // misconfiguration signal in the response so the operator notices.
            self.record_warn_notice(format!(
                "Permission for '{}' auto-denied: session is in Ask mode but the caller \
                 requested stream=false, which has no human-in-loop channel. Configure the \
                 session with permission=write to allow these tools, or use stream=true.",
                request.tool_name
            ))
            .await;
            return PermissionOutcome::Deny;
        }

        if !self.capabilities.supports_permission_prompts {
            // Streaming, but the client told us it has nowhere to show a prompt. Parking would
            // burn the full timeout and deny anyway; do it now and say why.
            self.record_warn_notice(format!(
                "Permission for '{}' auto-denied: the session declared \
                 supports_permission_prompts=false, so there is no channel to approve on. \
                 Configure the session with permission=write to allow these tools.",
                request.tool_name
            ))
            .await;
            return PermissionOutcome::Deny;
        }

        // Streaming mode: emit a permission_required event and park on a oneshot for the
        // matching POST /responses/{request_id}. Race against per-turn cancellation and the
        // 60s timeout so the agent loop never blocks indefinitely.
        let request_id = format!("req_{}", uuid::Uuid::new_v4());
        let (sender, receiver) = oneshot::channel::<PermissionOutcome>();
        {
            let mut guard = super::poisoned::lock(&self.pending, "http_frontend::park_permission");
            guard.insert(request_id.clone(), PermissionPending {
                sender,
                tool_name: request.tool_name.clone(),
            });
        }

        // Hold the stream lock across `ids.next()` + `sender.send()` to preserve monotonic id
        // ordering, mirroring `emit()`.
        {
            let guard = super::poisoned::lock(&self.stream, "http_frontend::emit_pause");
            if let Some(sink) = guard.as_ref() {
                let payload = serde_json::json!({
                    "request_id": request_id,
                    "tool_name": request.tool_name,
                    "expires_in_seconds": MID_TURN_REQUEST_TIMEOUT.as_secs(),
                });
                let event = SseEvent {
                    id: sink.ids.next(),
                    event_type: SseEventType::PermissionRequired,
                    data: payload,
                };
                let _ = sink.sender.send(event);
            }
        }

        // Poll-based disconnect detection: `broadcast::Sender` has no async "wait for
        // subscriber count change", so we check `client_disconnected()` on a short interval.
        // Without this, a client that drops the SSE connection while the turn is parked here
        // leaves the session stuck in `TurnInFlight` until the 60s timeout or a manual
        // `POST /cancel`.
        let disconnect_poll = async {
            loop {
                tokio::time::sleep(DISCONNECT_POLL_INTERVAL).await;
                if self.client_disconnected() {
                    break;
                }
            }
        };

        let outcome = tokio::select! {
            biased;
            _ = request.cancellation.cancelled() => PermissionOutcome::Cancelled,
            _ = disconnect_poll => {
                tracing::info!(
                    "SSE consumer disconnected while permission_required for '{}' was pending; \
                     auto-cancelling",
                    request.tool_name,
                );
                PermissionOutcome::Cancelled
            },
            _ = tokio::time::sleep(MID_TURN_REQUEST_TIMEOUT) => PermissionOutcome::Deny,
            response = receiver => response.unwrap_or(PermissionOutcome::Cancelled),
        };
        // Remove the entry if it's still there (timeout, cancellation, or disconnect paths).
        let mut guard = super::poisoned::lock(&self.pending, "http_frontend::cleanup_permission");
        guard.remove(&request_id);
        outcome
    }

    async fn handle_elicitation(&self, prompt: ElicitationPrompt) -> ElicitationResponse {
        // The HTTP API doesn't expose MCP elicitation in either mode: service-to-service
        // callers can't render interactive prompts (see HTTP API docs § Ask mode). The
        // notice surfaces the auto-decline so operators can spot misconfigured servers that
        // expect to drive elicitation interactively.
        self.record_warn_notice(format!(
            "MCP elicitation from '{}' auto-declined: the HTTP frontend does not expose \
             interactive MCP prompts.",
            prompt.server_name
        ))
        .await;
        ElicitationResponse::Decline
    }

    // `delegate_fs_read` / `_fs_write` / `_execute` keep the trait defaults (all return `None`).
    // The HTTP frontend does not expose client-hosted tool delegation. Returning `None`
    // routes the call to the agent's local I/O path.

    /// SSE-mode disconnect detection. When a `StreamSink` is installed and the broadcast has zero
    /// remaining subscribers, the SSE consumer has dropped; the agent loop should short-circuit
    /// at its next iteration so we don't keep burning provider tokens for an audience that's
    /// gone away. Blocking mode (no `StreamSink`) has no transport-level disconnect to observe
    /// until the response writes complete, so we keep the trait default `false` there.
    fn client_disconnected(&self) -> bool {
        let stream = super::poisoned::lock(&self.stream, "http_frontend::client_disconnected");
        match stream.as_ref() {
            Some(sink) => sink.sender.receiver_count() == 0,
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        frontend::{Frontend, FrontendEvent},
        mcp::elicitation::{ElicitationKind, ElicitationPrompt},
    };

    #[tokio::test]
    async fn emit_buffers_events_in_order() {
        let frontend = HttpFrontend::new();
        frontend.emit(FrontendEvent::TurnStarted).await;
        frontend
            .emit(FrontendEvent::AssistantTextDelta("hello".into()))
            .await;
        frontend.emit(FrontendEvent::TurnFinished).await;
        let recorder = frontend.drain();
        assert_eq!(recorder.len(), 3);
        assert!(matches!(recorder[0], FrontendEvent::TurnStarted));
        assert!(matches!(recorder[2], FrontendEvent::TurnFinished));
    }

    #[tokio::test]
    async fn request_permission_returns_deny_and_records_notice() {
        let frontend = HttpFrontend::new();
        let outcome = frontend
            .request_permission(PermissionRequest {
                tool_name: "execute_command".into(),
                primary_param: Some("rm /tmp/x".into()),
                cancellation: tokio_util::sync::CancellationToken::new(),
            })
            .await;
        assert_eq!(outcome, PermissionOutcome::Deny);
        let recorder = frontend.drain();
        let notice_count = recorder
            .iter()
            .filter(|event| {
                matches!(
                    event,
                    FrontendEvent::Notice(notice)
                        if notice.text.contains("execute_command")
                            && notice.text.contains("auto-denied")
                )
            })
            .count();
        assert_eq!(
            notice_count, 1,
            "blocking-mode deny must surface exactly one diagnostic Notice"
        );
    }

    /// A streaming client that declared it cannot show prompts must be denied immediately rather
    /// than parked: the default path burns the full 60s `MID_TURN_REQUEST_TIMEOUT` before denying
    /// anyway, which is indistinguishable from a hang. Bounded well under that timeout so a
    /// regression to the parking path fails here instead of passing slowly.
    #[tokio::test]
    async fn streaming_denies_immediately_when_prompts_are_unsupported() {
        let frontend = Arc::new(HttpFrontend::with_capabilities(SessionCapabilities {
            supports_permission_prompts: false,
            ..Default::default()
        }));
        let (_receiver, _ids) = frontend.install_stream(16);

        let outcome = tokio::time::timeout(
            Duration::from_secs(5),
            frontend.request_permission(PermissionRequest {
                tool_name: "execute_command".into(),
                primary_param: Some("rm /tmp/x".into()),
                cancellation: tokio_util::sync::CancellationToken::new(),
            }),
        )
        .await
        .expect("must not park on the SSE channel");

        assert_eq!(outcome, PermissionOutcome::Deny);
        assert!(
            frontend
                .pending
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .is_empty(),
            "nothing should have been parked"
        );
        let recorder = frontend.drain();
        assert!(
            recorder.iter().any(|event| matches!(
                event,
                FrontendEvent::Notice(notice)
                    if notice.text.contains("supports_permission_prompts")
            )),
            "the deny must explain itself in the response, not just in the timing"
        );
    }

    /// Sessions persisted before this field existed deserialize without it, and must keep the
    /// old parking behaviour rather than silently switching to auto-deny.
    #[test]
    fn capabilities_from_legacy_json_default_to_supporting_prompts() {
        let legacy: SessionCapabilities =
            serde_json::from_str(r#"{"supports_reasoning_stream":true}"#).expect("deserialize");
        assert!(legacy.supports_reasoning_stream);
        assert!(legacy.supports_permission_prompts);
    }

    #[tokio::test]
    async fn sticky_allow_short_circuits_subsequent_requests() {
        let frontend = HttpFrontend::new();
        frontend.remember_allow("read_file");
        let outcome = frontend
            .request_permission(PermissionRequest {
                tool_name: "read_file".into(),
                primary_param: None,
                cancellation: tokio_util::sync::CancellationToken::new(),
            })
            .await;
        assert_eq!(
            outcome,
            PermissionOutcome::Allow,
            "tools in always_allowed must not re-prompt"
        );
        let recorder = frontend.drain();
        // No diagnostic notice should be emitted: the sticky path bypasses both the streaming
        // SSE pause and the blocking-mode auto-deny.
        assert!(
            !recorder
                .iter()
                .any(|event| matches!(event, FrontendEvent::Notice(_))),
            "sticky allow must not emit a diagnostic Notice"
        );
    }

    #[tokio::test]
    async fn sticky_deny_short_circuits_subsequent_requests() {
        let frontend = HttpFrontend::new();
        frontend.remember_deny("execute_command");
        let outcome = frontend
            .request_permission(PermissionRequest {
                tool_name: "execute_command".into(),
                primary_param: None,
                cancellation: tokio_util::sync::CancellationToken::new(),
            })
            .await;
        assert_eq!(outcome, PermissionOutcome::Deny);
    }

    #[tokio::test]
    async fn resolve_permission_allow_always_records_sticky() {
        let frontend = HttpFrontend::new();
        // Install a stream so request_permission parks instead of blocking-mode short-circuit.
        let (_receiver, _ids) = frontend.install_stream(16);

        let pending_handle = {
            let frontend_clone = Arc::new(frontend);
            let frontend_inner = Arc::clone(&frontend_clone);
            let request = PermissionRequest {
                tool_name: "write_file".into(),
                primary_param: Some("/tmp/x".into()),
                cancellation: tokio_util::sync::CancellationToken::new(),
            };
            let join =
                tokio::spawn(async move { frontend_inner.request_permission(request).await });
            // Yield so request_permission has registered the pending entry.
            tokio::time::sleep(Duration::from_millis(20)).await;

            // Resolve via the AllowAlways path.
            let pending_request_id = {
                let guard = frontend_clone
                    .pending
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                guard.keys().next().cloned().expect("pending entry exists")
            };
            assert!(
                frontend_clone
                    .resolve_permission(&pending_request_id, PermissionResolution::AllowAlways,)
            );
            let outcome = join.await.expect("join");
            assert_eq!(outcome, PermissionOutcome::Allow);
            assert!(
                frontend_clone.is_always_allowed("write_file"),
                "AllowAlways must record the tool in always_allowed"
            );
            assert!(
                !frontend_clone.is_never_allowed("write_file"),
                "AllowAlways must not also touch never_allowed"
            );
            frontend_clone
        };
        drop(pending_handle);
    }

    #[tokio::test]
    async fn thinking_delta_is_filtered_when_capability_is_off() {
        let frontend = HttpFrontend::with_capabilities(SessionCapabilities::default());
        let (mut receiver, _ids) = frontend.install_stream(16);
        frontend
            .emit(FrontendEvent::ThinkingBlock {
                content: "musing".into(),
                signature: None,
            })
            .await;
        frontend
            .emit(FrontendEvent::AssistantTextDelta("answer".into()))
            .await;
        // Drop the stream to close the broadcast and drain.
        frontend.clear_stream();
        let mut events = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            events.push(event);
        }
        assert_eq!(
            events.len(),
            1,
            "only the assistant delta should reach the SSE stream when reasoning is off"
        );
        assert_eq!(
            events[0].event_type,
            super::SseEventType::AssistantTextDelta
        );

        // The recorder still has both events (blocking-mode JSON path is unaffected).
        let recorder = frontend.drain();
        assert_eq!(recorder.len(), 2);
    }

    #[tokio::test]
    async fn thinking_delta_streams_when_capability_is_on() {
        let frontend = HttpFrontend::with_capabilities(SessionCapabilities {
            supports_reasoning_stream: true,
            ..Default::default()
        });
        let (mut receiver, _ids) = frontend.install_stream(16);
        frontend
            .emit(FrontendEvent::ThinkingBlock {
                content: "musing".into(),
                signature: None,
            })
            .await;
        frontend.clear_stream();
        let event = receiver.try_recv().expect("thinking event should stream");
        assert_eq!(event.event_type, super::SseEventType::ThinkingDelta);
    }

    /// When the SSE consumer disconnects (all broadcast receivers dropped) while
    /// `request_permission` is parked, the permission wait should resolve to `Cancelled`
    /// within `DISCONNECT_POLL_INTERVAL` instead of blocking until the 60s timeout.
    #[tokio::test]
    async fn request_permission_detects_sse_disconnect() {
        let frontend = Arc::new(HttpFrontend::new());
        // Install a stream so request_permission takes the streaming (park) path.
        let (receiver, _ids) = frontend.install_stream(16);

        let frontend_inner = Arc::clone(&frontend);
        let join = tokio::spawn(async move {
            frontend_inner
                .request_permission(PermissionRequest {
                    tool_name: "execute_command".into(),
                    primary_param: Some("echo hi".into()),
                    cancellation: tokio_util::sync::CancellationToken::new(),
                })
                .await
        });

        // Let the permission request register in the pending map.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(
            frontend
                .pending
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .len(),
            1,
            "one pending permission request should be registered"
        );

        // Simulate SSE client disconnect by dropping the broadcast receiver.
        drop(receiver);
        assert!(
            frontend.client_disconnected(),
            "client_disconnected() should return true after receiver is dropped"
        );

        // The permission wait should resolve within a few poll intervals.
        let outcome = tokio::time::timeout(Duration::from_secs(5), join)
            .await
            .expect("should resolve well before 60s timeout")
            .expect("task should not panic");
        assert_eq!(
            outcome,
            PermissionOutcome::Cancelled,
            "SSE disconnect must resolve the parked permission to Cancelled"
        );

        // The pending map should be cleaned up.
        assert_eq!(
            frontend
                .pending
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .len(),
            0,
            "pending entry should be cleaned up after disconnect"
        );
    }

    #[tokio::test]
    async fn handle_elicitation_returns_decline_and_records_notice() {
        let frontend = HttpFrontend::new();
        let prompt = ElicitationPrompt {
            server_name: "github".into(),
            kind: ElicitationKind::Url {
                url: "https://example.com".into(),
            },
            message: "Open this URL?".into(),
        };
        let response = frontend.handle_elicitation(prompt).await;
        assert!(matches!(response, ElicitationResponse::Decline));
        let recorder = frontend.drain();
        assert!(
            recorder.iter().any(|event| matches!(
                event,
                FrontendEvent::Notice(notice) if notice.text.contains("github")
            )),
            "elicitation decline must surface a diagnostic Notice"
        );
    }
}
