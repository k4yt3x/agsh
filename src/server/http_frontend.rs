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
    /// The current (or most recent) turn's SSE stream. Set by the turn handler before calling
    /// `run_turn` (via [`Self::install_stream`]) and ended after (via [`Self::end_stream`]). While
    /// its sender is live, every emitted event is translated into an `SseEvent`, published on the
    /// broadcast, and recorded in the replay ring. `None` means no streaming turn has run on this
    /// session yet; a `Some` whose sender is `None` means the last one has finished and only its
    /// tail is being held for a late reconnect.
    stream: Mutex<Option<TurnStream>>,
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
    /// Event ids, monotonic across the *session* rather than restarting per turn.
    ///
    /// Per-turn ids look tidier and make `Last-Event-ID` unusable: a client holding id 5 from one
    /// turn and reconnecting during the next cannot be told apart from one that is up to date,
    /// because the new turn issues id 5 too. Filtering against it then discards the entire backlog
    /// as already-delivered. Session-scoped ids make a stale position sort strictly below
    /// everything the current turn emitted, so the ordinary `event.id > last` filter is correct
    /// without needing to know which turn the id came from.
    ids: Arc<EventIdGenerator>,
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
    /// Defaults to `true`, so a client that declares nothing and an imported `capabilities_json`
    /// that omits the flag both park rather than auto-deny.
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

/// One turn's SSE event stream, retained past the end of the turn so a client that reconnects can
/// still collect the tail.
///
/// The ring is what `Last-Event-ID` resumption is built on. Everything else on this stream is
/// additive, so a client that misses an event still holds a prefix of the truth and could limp
/// along; the *terminal* event is not, because a client that never receives one waits forever. So
/// the terminal is recorded here too, by the spawned turn task rather than by the response stream.
/// That distinction is load-bearing: in the case re-attach exists for, the client's connection has
/// dropped and axum has already discarded the response stream, so anything only that future
/// computes is computed for nobody.
pub struct TurnStream {
    turn_id: uuid::Uuid,
    ids: Arc<EventIdGenerator>,
    /// `None` once the turn has ended. Dropping the sender is what closes every live subscriber,
    /// and is also how [`HttpFrontend::is_streaming`] reports the turn as over while the ring is
    /// still held for late reconnects.
    sender: Option<broadcast::Sender<SseEvent>>,
    /// Recent events, oldest first, capped at `replay_capacity`.
    replay: std::collections::VecDeque<SseEvent>,
    replay_capacity: usize,
    /// The turn's terminal event, once known.
    terminal: Option<SseEvent>,
    /// When the last subscriber went away, or `None` while one is attached.
    ///
    /// Zero subscribers used to mean "cancel the turn, nobody is listening". That is the right
    /// instinct (a turn with no audience is burning provider tokens for nobody) and the wrong
    /// deadline, because the case re-attach exists for looks identical for its first instant: a
    /// client whose connection dropped and is about to come back. The stamp turns the check into a
    /// grace period, so a reconnect inside the window finds the turn still running.
    disconnected_since: Option<std::time::Instant>,
    /// How long to hold a turn open for a reconnect before treating the client as gone.
    reattach_grace: Duration,
}

impl TurnStream {
    fn record(&mut self, event: SseEvent) {
        if self.replay_capacity == 0 {
            return;
        }
        while self.replay.len() >= self.replay_capacity {
            self.replay.pop_front();
        }
        self.replay.push_back(event);
    }
}

/// What a re-attaching client gets: the backlog it missed, plus a live subscription, taken
/// together under one lock so nothing can be emitted in the gap between them.
pub struct StreamAttachment {
    pub turn_id: uuid::Uuid,
    /// Buffered events with an id greater than the client's `Last-Event-ID`, oldest first.
    pub backlog: Vec<SseEvent>,
    /// `None` when the turn has already ended; the backlog and `terminal` are then the whole
    /// story.
    pub receiver: Option<broadcast::Receiver<SseEvent>>,
    /// Present once the turn is over. A client that reconnects after the fact gets it immediately
    /// rather than waiting on a stream that will never produce another event.
    pub terminal: Option<SseEvent>,
    /// True when the client's `Last-Event-ID` is older than the oldest event still buffered, so
    /// the replay has a hole in it. Reported rather than papered over: a transcript with a silent
    /// gap is worse than one the client knows is incomplete.
    pub gap: bool,
    /// The position resumption should actually use, after discarding a `Last-Event-ID` that this
    /// turn never issued.
    ///
    /// `None` means "send everything you have".
    ///
    /// Ids run monotonically across the whole session, so an id from an *earlier* turn sorts below
    /// this turn's backlog and filters nothing -- that is the case this field is designed to let
    /// through. What it discards is an id at or above the high-water mark: one fabricated, or
    /// carried over from a different session by a browser `EventSource` that re-sends its stored
    /// id automatically. Honouring such an id would filter the entire backlog, and the terminal
    /// with it, as already-delivered, leaving the client waiting on a turn it can never see end.
    pub resume_from: Option<u64>,
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
            ids: Arc::new(EventIdGenerator::default()),
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
        // The sender, not the slot: a `TurnStream` outlives its turn so a late reconnect can still
        // read the tail, and treating that as "streaming" would send the next blocking turn's
        // permission prompt to a channel nobody is listening on.
        guard.as_ref().is_some_and(|stream| stream.sender.is_some())
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
                    PermissionResolution::Allow | PermissionResolution::AllowAlways => {
                        PermissionOutcome::Allow
                    }
                    PermissionResolution::Deny | PermissionResolution::DenyAlways => {
                        PermissionOutcome::Deny
                    }
                };

                // Deliver first, and only record the sticky decision if the waiter actually took
                // it.
                //
                // `request_permission` resolves through a `tokio::select!`, so its
                // `oneshot::Receiver` is already dropped once the 60s timeout
                // expires, the turn is cancelled, or the SSE client disconnects.
                // Recording before the send meant a reply that lost that race was
                // reported to the caller as `404 request-not-found` -- the tool call denied, the
                // client told nothing was resolved -- while the tool had nonetheless been written
                // into `always_allowed` for the rest of the session, silently
                // short-circuiting every later call to it with no prompt and no SSE
                // event. An answer nobody received must not grant anything.
                let delivered = pending.sender.send(outcome).is_ok();
                if delivered {
                    match resolution {
                        PermissionResolution::AllowAlways => {
                            self.remember_allow(&pending.tool_name)
                        }
                        PermissionResolution::DenyAlways => self.remember_deny(&pending.tool_name),
                        PermissionResolution::Allow | PermissionResolution::Deny => {}
                    }
                }
                delivered
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
        replay_capacity: usize,
        reattach_grace: Duration,
        turn_id: uuid::Uuid,
    ) -> (broadcast::Receiver<SseEvent>, Arc<EventIdGenerator>) {
        let (sender, receiver) = broadcast::channel::<SseEvent>(capacity);
        // The session's generator, not a fresh one: see the field docs. Ids continue across turns.
        let ids = Arc::clone(&self.ids);
        let mut guard = super::poisoned::lock(&self.stream, "http_frontend::install_stream");
        // Replaces any retained previous turn outright: the previous turn's tail is superseded the
        // moment a new one starts, and a client reconnecting now wants the live one.
        *guard = Some(TurnStream {
            turn_id,
            ids: ids.clone(),
            sender: Some(sender),
            replay: std::collections::VecDeque::with_capacity(replay_capacity.min(64)),
            replay_capacity,
            terminal: None,
            disconnected_since: None,
            reattach_grace,
        });
        (receiver, ids)
    }

    /// Record the turn's terminal event, so a client that re-attaches learns how it ended.
    ///
    /// Called from the spawned turn task, deliberately, and not from the response stream: see the
    /// note on [`TurnStream`]. Assigning the id here rather than at the call site keeps the
    /// per-turn sequence dense even when the response stream was dropped long ago.
    pub fn record_terminal(&self, event_type: SseEventType, data: serde_json::Value) -> SseEvent {
        let mut guard = super::poisoned::lock(&self.stream, "http_frontend::record_terminal");
        let Some(stream) = guard.as_mut() else {
            // Unreachable today: the only caller is the streaming turn task, which installs a
            // stream before it spawns. Still drawn from the session generator rather than
            // hardcoded, because ids are session-scoped now and a fabricated `0` would collide
            // with the session's genuine first event if this ever did fire.
            return SseEvent {
                id: self.ids.next(),
                event_type,
                data,
            };
        };
        let event = SseEvent {
            id: stream.ids.next(),
            event_type,
            data,
        };
        stream.record(event.clone());
        stream.terminal = Some(event.clone());
        event
    }

    /// How many SSE consumers are attached to the live turn right now.
    ///
    /// Zero once the turn's stream has ended or was never installed. Read by the stream task to
    /// decide whether one lagging consumer speaks for the whole turn.
    pub fn subscriber_count(&self) -> usize {
        let guard = super::poisoned::lock(&self.stream, "http_frontend::subscriber_count");
        guard
            .as_ref()
            .and_then(|stream| stream.sender.as_ref())
            .map_or(0, |sender| sender.receiver_count())
    }

    /// End the turn's stream: drop the sender so live subscribers see the close, keeping the ring
    /// and the terminal for a late reconnect. Called by the streaming turn handler's guard after
    /// `run_turn` returns.
    pub fn end_stream(&self) {
        let mut guard = super::poisoned::lock(&self.stream, "http_frontend::end_stream");
        if let Some(stream) = guard.as_mut() {
            stream.sender = None;
        }
    }

    /// The terminal event of the most recent turn, once it has one.
    ///
    /// Re-read after a live subscription closes rather than trusted from the attachment snapshot:
    /// a client that attached mid-turn captured `terminal: None` because the turn had not ended
    /// yet, and [`Self::record_terminal`] deliberately does not broadcast (the primary stream
    /// yields its own copy from the join handle, and a broadcast one would race it into a
    /// duplicate). So the only way a live re-attacher learns the outcome is to ask again.
    /// Scoped to `turn_id`, because `install_stream` replaces the whole [`TurnStream`], terminal
    /// included. A re-attacher wakes on its broadcast closing and asks again; if the next turn has
    /// already started by then, an unscoped read would hand it the new turn's terminal (or `None`
    /// for a turn that actually succeeded). Returning `None` on a mismatch lets the caller say what
    /// is true: the turn ended and its outcome is no longer held here.
    pub fn recorded_terminal(&self, turn_id: uuid::Uuid) -> Option<SseEvent> {
        let guard = super::poisoned::lock(&self.stream, "http_frontend::recorded_terminal");
        guard
            .as_ref()
            .filter(|stream| stream.turn_id == turn_id)
            .and_then(|stream| stream.terminal.clone())
    }

    /// Attach to the current turn's stream, replaying anything after `last_event_id`.
    ///
    /// The backlog snapshot and the `subscribe()` happen under one lock, and [`Self::emit`] takes
    /// the same lock to append. That is what makes the replay gap-free: without it an event
    /// emitted between the snapshot and the subscribe would be in neither.
    ///
    /// `None` when no turn has ever streamed on this session.
    pub fn attach_stream(&self, last_event_id: Option<u64>) -> Option<StreamAttachment> {
        let mut guard = super::poisoned::lock(&self.stream, "http_frontend::attach_stream");
        let stream = guard.as_mut()?;
        // Someone is listening again, so the grace clock restarts. Cleared here and not only in
        // `client_disconnected`, which the agent loop reaches at provider-round boundaries: a
        // client that reconnects and drops again before the next boundary would otherwise have its
        // second grace measured from the *first* disconnect, and so get almost none of it.
        stream.disconnected_since = None;
        // Ids are session-monotonic, so an id at or above the high-water mark was never issued
        // here at all -- a fabricated value, or one carried over from a different session. Discard
        // it rather than filter against it, which would silently deliver nothing.
        let stale = last_event_id.is_some_and(|last| last >= stream.ids.peek());
        let resume_from = if stale { None } else { last_event_id };
        // Taking `pending` while holding `stream` is safe in this order only: `request_permission`
        // releases `pending` before it acquires `stream` (see `park_permission` / `emit_pause`),
        // and `resolve_permission` never touches `stream` at all, so there is no inversion.
        let still_pending = super::poisoned::lock(&self.pending, "http_frontend::attach_pending");
        let backlog: Vec<SseEvent> = stream
            .replay
            .iter()
            .filter(|event| resume_from.is_none_or(|last| event.id > last))
            // A pause is stateful, not additive. Replaying one the client already answered (or
            // that timed out) would put an approval prompt back on screen for a request that no
            // longer exists, and any decision sent for it comes back 404. Replay it only while it
            // is still actionable, which is exactly while it is still parked.
            .filter(|event| {
                if event.event_type != SseEventType::PermissionRequired {
                    return true;
                }
                event
                    .data
                    .get("request_id")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|request_id| still_pending.contains_key(request_id))
            })
            .cloned()
            .collect();
        drop(still_pending);
        // A hole exists only relative to a position the client actually claims. `id + 1` because
        // resuming from exactly the oldest retained id is contiguous.
        //
        // A client that names no `Last-Event-ID` is joining, not resuming, and has lost nothing:
        // warning it about the events before it arrived would fire on every first attach, since
        // the ring never holds the `turn.started` the response stream generates for itself. A
        // *stale* id is always a gap, because whatever the client was following has ended.
        let gap = stale
            || match (resume_from, stream.replay.front()) {
                (Some(last), Some(oldest)) => oldest.id > last.saturating_add(1),
                // Replay is switched off (`stream_replay_events = 0`), so a client resuming from a
                // position has been handed nothing between there and now. Reporting no gap would
                // be the silent truncation the notice exists to rule out.
                (Some(_), None) => stream.replay_capacity == 0,
                _ => false,
            };
        Some(StreamAttachment {
            turn_id: stream.turn_id,
            backlog,
            receiver: stream.sender.as_ref().map(|sender| sender.subscribe()),
            terminal: stream.terminal.clone(),
            gap,
            resume_from,
        })
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
            let mut guard = super::poisoned::lock(&self.stream, "http_frontend::emit_stream");
            if let Some(stream) = guard.as_mut()
                && let Some(sender) = stream.sender.clone()
                && self.event_passes_capability_filter(&event)
                && let Some((event_type, data)) = translate(event.clone(), self.capabilities)
            {
                let sse = SseEvent {
                    id: stream.ids.next(),
                    event_type,
                    data,
                };
                // Recorded before sending, so an event is in the replay ring by the time any
                // subscriber can observe it. The reverse order would let a client re-attach in
                // the window between and miss it in both places.
                stream.record(sse.clone());
                // `send` returns Err only when there are no subscribers: that means the SSE
                // client has disconnected. The recorder still gets the event so the turn
                // handler can produce the blocking-mode JSON fallback (or in the streaming
                // case, just discard the events after run_turn returns). The ring keeps it
                // either way, which is what a re-attaching client reads back.
                let _ = sender.send(sse);
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
            let mut guard = super::poisoned::lock(&self.stream, "http_frontend::emit_pause");
            if let Some(stream) = guard.as_mut()
                && let Some(sender) = stream.sender.clone()
            {
                let payload = serde_json::json!({
                    "request_id": request_id,
                    "tool_name": request.tool_name,
                    "expires_in_seconds": MID_TURN_REQUEST_TIMEOUT.as_secs(),
                });
                let event = SseEvent {
                    id: stream.ids.next(),
                    event_type: SseEventType::PermissionRequired,
                    data: payload,
                };
                // Recorded like any other event: a client that reconnects mid-pause has to learn
                // that the turn is waiting on it, or the turn sits there until the timeout.
                stream.record(event.clone());
                let _ = sender.send(event);
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

    /// SSE-mode disconnect detection, with a reconnect grace period.
    ///
    /// Zero remaining subscribers means the SSE consumer has dropped, and the agent loop
    /// short-circuits so we don't keep burning provider tokens for an audience that has gone away.
    /// It reports the disconnect only once the count has been zero for
    /// [`TurnStream::reattach_grace`], because a client whose connection dropped a moment ago and
    /// one that is never coming back are the same observation until the window expires. A
    /// reconnect through [`Self::attach_stream`] clears the stamp on its next poll.
    ///
    /// Blocking mode (no stream installed) has no transport-level disconnect to observe until the
    /// response writes complete, so the trait default `false` stands there.
    fn client_disconnected(&self) -> bool {
        let mut guard = super::poisoned::lock(&self.stream, "http_frontend::client_disconnected");
        let Some(stream) = guard.as_mut() else {
            return false;
        };
        let Some(sender) = stream.sender.as_ref() else {
            return false;
        };
        if sender.receiver_count() > 0 {
            stream.disconnected_since = None;
            return false;
        }
        // Stamped and evaluated in one step so a zero grace means exactly no grace, rather than
        // "one poll interval": the first observation would otherwise always report `false`.
        let since = *stream
            .disconnected_since
            .get_or_insert_with(std::time::Instant::now);
        since.elapsed() >= stream.reattach_grace
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        frontend::{Frontend, FrontendEvent},
        mcp::elicitation::{ElicitationKind, ElicitationPrompt},
    };

    /// `subscriber_count` has to see a re-attached second consumer, because that count is the only
    /// thing standing between one slow reader and everyone else's turn.
    ///
    /// The SSE stream task cancels the turn when a consumer lags, and it used to do so
    /// unconditionally: turn events are a broadcast, so a re-attached client or a second consumer
    /// is a separate receiver, and one slow reader killed the turn out from under the client that
    /// was keeping up. The guard is `subscriber_count() <= 1`, and reverting it left all four
    /// suites green. This pins the count's semantics, including the part the guard depends on --
    /// that the lagging receiver, which is about to be dropped, is still included while it lives,
    /// which is why the threshold is `<= 1` rather than `== 0`.
    #[tokio::test]
    async fn subscriber_count_sees_every_live_consumer_of_a_turn() {
        let frontend = HttpFrontend::new();
        assert_eq!(
            frontend.subscriber_count(),
            0,
            "no stream installed yet, so nobody is reading"
        );

        let (first, _ids) = frontend.install_stream(
            16,
            16,
            Duration::from_secs(1),
            uuid::Uuid::from_u128(0xfeed),
        );
        assert_eq!(frontend.subscriber_count(), 1, "the turn's own consumer");

        let second = frontend
            .attach_stream(None)
            .expect("a live stream accepts a re-attach");
        assert_eq!(
            frontend.subscriber_count(),
            2,
            "a re-attached client is a second receiver; cancelling on the first one's lag would \
             take the turn away from it"
        );

        drop(second);
        assert_eq!(
            frontend.subscriber_count(),
            1,
            "and once it goes, the lagging consumer speaks for the whole turn again"
        );
        drop(first);
        assert_eq!(frontend.subscriber_count(), 0);
    }

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
                input: serde_json::Value::Null,
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
        let (_receiver, _ids) =
            frontend.install_stream(16, 16, Duration::from_secs(30), uuid::Uuid::nil());

        let outcome = tokio::time::timeout(
            Duration::from_secs(5),
            frontend.request_permission(PermissionRequest {
                tool_name: "execute_command".into(),
                primary_param: Some("rm /tmp/x".into()),
                input: serde_json::Value::Null,
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

    /// `meka session import` stores `capabilities_json` verbatim from a user-supplied archive, so
    /// a hand-written or third-party one can be missing a flag. An absent flag has to mean parking:
    /// silently auto-denying every gated call in an imported session is a worse failure than a
    /// stall the operator can see.
    #[test]
    fn capabilities_json_missing_a_flag_defaults_to_supporting_prompts() {
        let partial: SessionCapabilities =
            serde_json::from_str(r#"{"supports_reasoning_stream":true}"#).expect("deserialize");
        assert!(partial.supports_reasoning_stream);
        assert!(partial.supports_permission_prompts);
    }

    #[tokio::test]
    async fn sticky_allow_short_circuits_subsequent_requests() {
        let frontend = HttpFrontend::new();
        frontend.remember_allow("read_file");
        let outcome = frontend
            .request_permission(PermissionRequest {
                tool_name: "read_file".into(),
                primary_param: None,
                input: serde_json::Value::Null,
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
                input: serde_json::Value::Null,
                cancellation: tokio_util::sync::CancellationToken::new(),
            })
            .await;
        assert_eq!(outcome, PermissionOutcome::Deny);
    }

    #[tokio::test]
    async fn resolve_permission_allow_always_records_sticky() {
        let frontend = HttpFrontend::new();
        // Install a stream so request_permission parks instead of blocking-mode short-circuit.
        let (_receiver, _ids) =
            frontend.install_stream(16, 16, Duration::from_secs(30), uuid::Uuid::nil());

        let pending_handle = {
            let frontend_clone = Arc::new(frontend);
            let frontend_inner = Arc::clone(&frontend_clone);
            let request = PermissionRequest {
                tool_name: "write_file".into(),
                primary_param: Some("/tmp/x".into()),
                input: serde_json::Value::Null,
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
        let (mut receiver, _ids) =
            frontend.install_stream(16, 16, Duration::from_secs(30), uuid::Uuid::nil());
        frontend
            .emit(FrontendEvent::ThinkingBlock {
                content: "musing".into(),
            })
            .await;
        frontend
            .emit(FrontendEvent::AssistantTextDelta("answer".into()))
            .await;
        // Drop the stream to close the broadcast and drain.
        frontend.end_stream();
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
        let (mut receiver, _ids) =
            frontend.install_stream(16, 16, Duration::from_secs(30), uuid::Uuid::nil());
        frontend
            .emit(FrontendEvent::ThinkingBlock {
                content: "musing".into(),
            })
            .await;
        frontend.end_stream();
        let event = receiver.try_recv().expect("thinking event should stream");
        assert_eq!(event.event_type, super::SseEventType::ThinkingDelta);
    }

    /// When the SSE consumer disconnects (all broadcast receivers dropped) while
    /// `request_permission` is parked, the permission wait should resolve to `Cancelled`
    /// within `DISCONNECT_POLL_INTERVAL` instead of blocking until the 60s timeout.
    ///
    /// Zero grace, so this tests the resolution path rather than the reconnect window;
    /// `client_disconnected_waits_out_the_reattach_grace` covers the window itself.
    /// A reply that loses the race must grant nothing.
    ///
    /// `resolve_permission` answers the HTTP caller `404 request-not-found` when the waiter has
    /// already gone -- cancelled, timed out, or disconnected. Recording the sticky decision
    /// *before* the send meant that same call still wrote the tool into `always_allowed` for
    /// the rest of the session: the caller was told nothing was resolved, the tool call was
    /// denied, and every later call to that tool was silently approved with no prompt and no
    /// SSE event. Ordering the record after a successful delivery is the whole fix, and nothing
    /// pinned it.
    #[tokio::test]
    async fn a_reply_that_arrives_too_late_grants_nothing() {
        let frontend = Arc::new(HttpFrontend::new());
        let (_receiver, _ids) = frontend.install_stream(16, 16, Duration::ZERO, uuid::Uuid::nil());

        let frontend_inner = Arc::clone(&frontend);
        let cancellation = tokio_util::sync::CancellationToken::new();
        let waiter = cancellation.clone();
        let join = tokio::spawn(async move {
            frontend_inner
                .request_permission(PermissionRequest {
                    tool_name: "execute_command".into(),
                    primary_param: Some("rm -rf /".into()),
                    input: serde_json::Value::Null,
                    cancellation: waiter,
                })
                .await
        });

        // Let it register, then drop the waiter *without* letting it tidy up.
        //
        // Cancelling instead would not reach the race: `request_permission` removes its own entry
        // after the select, so a resolve arriving later finds nothing and returns at the
        // unknown-request arm, never reaching the record at all. Aborting drops the future
        // mid-await -- the receiver dies, the removal never runs -- which leaves exactly
        // the state the fix is about: an entry still in the map whose reader is gone, so
        // `send` fails.
        tokio::time::sleep(Duration::from_millis(50)).await;
        let request_id = frontend
            .pending
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .keys()
            .next()
            .cloned()
            .expect("the request must be registered before it is answered");
        drop(cancellation);
        join.abort();
        assert!(
            join.await.is_err(),
            "the waiter must be gone before the reply arrives"
        );

        // The user's "always allow" arrives after the waiter is gone.
        let delivered = frontend.resolve_permission(&request_id, PermissionResolution::AllowAlways);
        assert!(
            !delivered,
            "the caller has to be told the reply landed nowhere"
        );
        assert!(
            !frontend.is_always_allowed("execute_command"),
            "and a reply nobody received must not grant the tool for the rest of the session"
        );
    }

    #[tokio::test]
    async fn request_permission_detects_sse_disconnect() {
        let frontend = Arc::new(HttpFrontend::new());
        // Install a stream so request_permission takes the streaming (park) path.
        let (receiver, _ids) = frontend.install_stream(16, 16, Duration::ZERO, uuid::Uuid::nil());

        let frontend_inner = Arc::clone(&frontend);
        let join = tokio::spawn(async move {
            frontend_inner
                .request_permission(PermissionRequest {
                    tool_name: "execute_command".into(),
                    primary_param: Some("echo hi".into()),
                    input: serde_json::Value::Null,
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

    /// A dropped consumer is not immediately a departed one. Re-attach exists because networks
    /// drop connections, and for the first instant those two look identical; reporting a disconnect
    /// straight away would cancel exactly the turns a client is about to rejoin.
    #[tokio::test]
    async fn client_disconnected_waits_out_the_reattach_grace() {
        let frontend = HttpFrontend::new();
        let (receiver, _ids) =
            frontend.install_stream(16, 16, Duration::from_millis(150), uuid::Uuid::nil());
        assert!(
            !frontend.client_disconnected(),
            "a live consumer is attached"
        );

        drop(receiver);
        assert!(
            !frontend.client_disconnected(),
            "the first zero-subscriber observation starts the grace period, it does not end it"
        );
        assert!(!frontend.client_disconnected(), "still inside the window");
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            frontend.client_disconnected(),
            "past the window with nobody attached, the client really is gone"
        );
    }

    /// Reconnecting inside the window clears the stamp, so the turn keeps running.
    #[tokio::test]
    async fn reattaching_cancels_a_pending_disconnect() {
        let frontend = HttpFrontend::new();
        let (receiver, _ids) =
            frontend.install_stream(16, 16, Duration::from_millis(100), uuid::Uuid::nil());
        drop(receiver);
        assert!(!frontend.client_disconnected(), "grace period starts");

        let attachment = frontend.attach_stream(None).expect("a stream is installed");
        let _receiver = attachment.receiver.expect("the turn is still live");
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(
            !frontend.client_disconnected(),
            "a reconnect inside the window must clear the pending disconnect, not merely delay it"
        );
    }

    /// The ring is what `Last-Event-ID` replay reads. It must drop oldest-first at capacity and
    /// hand back only what the client has not already seen.
    #[tokio::test]
    async fn replay_ring_is_bounded_and_resumes_after_the_given_id() {
        let frontend = HttpFrontend::new();
        let (_receiver, _ids) =
            frontend.install_stream(64, 3, Duration::from_secs(30), uuid::Uuid::nil());
        for index in 0..5 {
            frontend
                .emit(FrontendEvent::AssistantTextDelta(format!("chunk{index}")))
                .await;
        }

        let all = frontend.attach_stream(None).expect("stream installed");
        assert_eq!(
            all.backlog.len(),
            3,
            "the ring holds its capacity, not the history"
        );
        assert_eq!(
            all.backlog[0].id, 2,
            "oldest-first eviction keeps the newest three"
        );
        assert!(
            !all.gap,
            "a client naming no Last-Event-ID is joining, not resuming, so it has lost nothing"
        );

        let resumed = frontend.attach_stream(Some(3)).expect("stream installed");
        let ids: Vec<u64> = resumed.backlog.iter().map(|event| event.id).collect();
        assert_eq!(ids, vec![4], "resume delivers strictly after the given id");
        assert!(
            !resumed.gap,
            "id 3 is still buffered, so the replay is contiguous"
        );

        let stale = frontend.attach_stream(Some(0)).expect("stream installed");
        assert!(
            stale.gap,
            "resuming from id 0 when the ring starts at 2 skips event 1, and must say so"
        );
    }

    /// A client that reconnects after the turn ended still has to learn how it ended.
    #[tokio::test]
    async fn attach_after_the_turn_ends_yields_the_terminal() {
        let frontend = HttpFrontend::new();
        let (_receiver, _ids) =
            frontend.install_stream(16, 16, Duration::from_secs(30), uuid::Uuid::nil());
        frontend
            .emit(FrontendEvent::AssistantTextDelta("hi".into()))
            .await;
        frontend.record_terminal(
            SseEventType::TurnFinished,
            serde_json::json!({"stop_reason": "end_turn"}),
        );
        frontend.end_stream();

        let attachment = frontend
            .attach_stream(None)
            .expect("stream retained past the turn");
        assert!(
            attachment.receiver.is_none(),
            "the turn is over; there is nothing live left to subscribe to"
        );
        let terminal = attachment.terminal.expect("terminal must be retained");
        assert_eq!(terminal.event_type, SseEventType::TurnFinished);
        assert!(
            attachment
                .backlog
                .iter()
                .any(|event| event.event_type.is_terminal()),
            "the terminal is in the ring too, so a replaying client receives it in order"
        );
    }

    /// A pause the client already answered must not come back on reconnect: it would put an
    /// approval prompt on screen for a request that no longer exists, and a decision sent for it
    /// returns 404. Additive events replay; stateful ones only while they are still true.
    #[tokio::test]
    async fn replay_drops_permission_prompts_that_are_no_longer_pending() {
        let frontend = Arc::new(HttpFrontend::new());
        let (_receiver, _ids) =
            frontend.install_stream(16, 16, Duration::from_secs(30), uuid::Uuid::nil());

        let asking = Arc::clone(&frontend);
        let join = tokio::spawn(async move {
            asking
                .request_permission(PermissionRequest {
                    tool_name: "execute_command".into(),
                    primary_param: Some("echo hi".into()),
                    input: serde_json::Value::Null,
                    cancellation: tokio_util::sync::CancellationToken::new(),
                })
                .await
        });
        tokio::time::sleep(Duration::from_millis(50)).await;

        // While parked, the prompt is real and must be replayed.
        let parked = frontend.attach_stream(None).expect("stream installed");
        assert!(
            parked
                .backlog
                .iter()
                .any(|event| event.event_type == SseEventType::PermissionRequired),
            "a live prompt must reach a reconnecting client, or the turn stalls until timeout"
        );
        let request_id = parked
            .backlog
            .iter()
            .find(|event| event.event_type == SseEventType::PermissionRequired)
            .and_then(|event| event.data.get("request_id"))
            .and_then(serde_json::Value::as_str)
            .expect("prompt carries a request_id")
            .to_string();

        assert!(frontend.resolve_permission(&request_id, PermissionResolution::Allow));
        assert_eq!(join.await.expect("task"), PermissionOutcome::Allow);

        let after = frontend.attach_stream(None).expect("stream installed");
        assert!(
            !after
                .backlog
                .iter()
                .any(|event| event.event_type == SseEventType::PermissionRequired),
            "an answered prompt must not be replayed"
        );
    }

    /// A finished-but-retained stream must not read as "streaming", or the next blocking turn
    /// would park its permission prompt on a channel nobody is listening to.
    #[tokio::test]
    async fn a_retained_stream_is_not_reported_as_streaming() {
        let frontend = HttpFrontend::new();
        let (_receiver, _ids) =
            frontend.install_stream(16, 16, Duration::from_secs(30), uuid::Uuid::nil());
        assert!(frontend.is_streaming());
        frontend.end_stream();
        assert!(
            !frontend.is_streaming(),
            "the ring outlives the turn; the streaming mode must not"
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
