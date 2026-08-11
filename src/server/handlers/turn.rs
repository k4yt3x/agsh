//! `POST /v1/sessions/{id}/turn`: submit a turn. Two response shapes:
//!
//! - **Blocking** (default, `stream: false`): `application/json` with the assembled
//!   [`TurnResponse`] once `Agent::run_turn` returns. The client gets the full transcript, tool
//!   calls, usage counters, and stop reason in one body.
//! - **Streaming** (`stream: true`): `text/event-stream` carrying live `turn.started` /
//!   `assistant_text.delta` / `tool_call.*` / `turn.finished` events (the full taxonomy is in the
//!   HTTP API docs § SSE events). Lifecycle events are 0-based and monotonic per turn.
//!
//! Both modes share an idempotency cache (Stripe-style, `Idempotency-Key` header). The cache
//! key is `(token_id, key)` and stores the *blocking* JSON envelope, so a replay of a
//! previously-streaming request returns the cached blocking body. Mid-turn permission gates
//! are handled out-of-band via `POST /v1/responses/{request_id}` on a side channel; the
//! streaming client sees a `permission_required` event and resolves via that endpoint without
//! interrupting the SSE response.

use std::{convert::Infallible, sync::Arc};

use axum::{
    Extension, Json,
    body::Bytes,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use futures::Stream;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;
use utoipa::ToSchema;
use uuid::Uuid;

use crate::{
    agent::TurnOutcome,
    frontend::FrontendEvent,
    provider::{Notice, NoticeLevel, ToolResultContent},
    server::{
        auth::Principal,
        errors::{ErrorKind, ProblemDetail},
        http_frontend::Recorder,
        idempotency::{LookupOutcome, hash_body},
        reattach::ensure_session_loaded,
        state::{ServerState, TurnGuard},
    },
};

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct TurnRequest {
    pub message: String,
    /// Image attachments for this turn. A sibling of `message` rather than a member of
    /// [`TurnOptions`] because these are user content, not a per-turn knob. Requires the active
    /// provider profile to have vision enabled; see [`decode_turn_images`].
    #[serde(default)]
    pub images: Vec<ImageInput>,
    /// `false` (default) → blocking JSON response. `true` → SSE.
    #[serde(default)]
    pub stream: bool,
    /// Per-turn knobs. See [`TurnOptions`]. Omitting the field is the same as `{}`.
    #[serde(default)]
    pub options: TurnOptions,
}

/// One inline image attachment. Base64 rather than a path or URL because the API is a network
/// surface: `[serve].bind` may be non-loopback, so the caller generally shares no filesystem with
/// the agent and cannot name a file for it to read.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageInput {
    /// Declared MIME type, e.g. `image/png`. Used as the primary format hint; the payload's magic
    /// bytes win if this doesn't name a supported format.
    pub media_type: String,
    /// Base64-encoded image bytes (standard alphabet, padding required).
    pub data: String,
}

/// Per-turn options. `#[serde(deny_unknown_fields)]` here (and only here) so a typo in
/// `option.skil` surfaces as a 422 rather than being silently dropped.
#[derive(Debug, Deserialize, ToSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct TurnOptions {
    /// Optional installed-skill name. When set, `message` is combined with the skill's body
    /// (user text first, then the skill body) before the agent runs, matching
    /// `/skill <name> <prompt>` in the REPL and the `--skill` CLI flag. Unknown skill → 422.
    #[serde(default)]
    pub skill: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct TurnResponse {
    pub turn_id: Uuid,
    pub session_id: Uuid,
    pub stop_reason: String,
    /// Concatenated assistant text produced this turn. **Excludes** refusal explanation:
    /// when the model refuses, the refusal text rides on the dedicated `refusal_text` field
    /// instead. Clients that just want "what the user sees" should consume both:
    /// `final_text` for the normal response, `refusal_text` when `stop_reason == "refusal"`.
    pub final_text: String,
    /// Refusal explanation when `stop_reason == "refusal"`; `None` otherwise. Mirrors the
    /// `refusal_text` field on the streaming `turn.finished` SSE event, so blocking and
    /// streaming clients share the same shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal_text: Option<String>,
    /// Structured view of the assistant's message(s) produced this turn. Per the spec, this is
    /// the "richer access" companion to `final_text`. Clients that want the text plus its
    /// formatting context (text/thinking content blocks) consume this; clients that just want
    /// a single string consume `final_text`. Tool calls live in their own `tool_calls` array.
    pub messages: Vec<crate::server::handlers::messages::MessageView>,
    pub tool_calls: Vec<ToolCallView>,
    pub usage: UsageView,
    pub notices: Vec<NoticeView>,
}

#[derive(Debug, Serialize, Default, ToSchema)]
pub struct UsageView {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_read_input_tokens: u64,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ToolCallView {
    pub id: String,
    pub name: String,
    #[schema(value_type = Object)]
    pub input: serde_json::Value,
    /// Serialized as `null` when the agent didn't produce a summary, per spec. Contrast with
    /// `refusal_text` which is omitted (not `null`) when absent.
    pub display_summary: Option<String>,
    pub is_error: bool,
    pub content: Vec<ToolCallContentView>,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolCallContentView {
    Text { text: String },
    Image { media_type: String },
}

#[derive(Debug, Serialize, ToSchema)]
pub struct NoticeView {
    pub level: String,
    pub text: String,
}

#[utoipa::path(
    post,
    path = "/v1/sessions/{id}/turn",
    tag = "turn",
    params(
        ("id" = Uuid, Path, description = "Session UUID"),
        ("Idempotency-Key" = Option<String>, Header, description = "Stripe-style replay key. Same key + same body returns the cached response; same key + different body returns 409."),
    ),
    request_body = TurnRequest,
    responses(
        (status = 200, description = "Blocking turn response (stream=false) or live SSE stream (stream=true). The application/json schema applies only to blocking mode.", body = TurnResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
        (status = 409, description = "Turn already in flight OR Idempotency-Key body mismatch", body = ProblemDetail),
        (status = 422, description = "Invalid body", body = ProblemDetail),
        (status = 429, description = "Concurrency limit reached or idempotency-key cache full", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn submit_turn(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    Path(session_id): Path<Uuid>,
    headers: HeaderMap,
    raw_body: Bytes,
) -> Result<Response, ProblemDetail> {
    if !principal.has_scope("sessions:w") {
        return Err(ProblemDetail::new(
            ErrorKind::AuthScope,
            StatusCode::FORBIDDEN,
            "scope `sessions:w` is required",
        ));
    }

    // Parse the header + body before consulting the idempotency cache. A malformed header /
    // body returns 422 cheaply; a successful parse lets us peek `stream` so we can skip
    // idempotency entirely for SSE replays.
    let idempotency_key = idempotency_header(&headers)?;
    let body: TurnRequest = serde_json::from_slice(&raw_body).map_err(|error| {
        ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            format!("failed to parse turn request body: {}", error),
        )
    })?;

    // Streaming turns can't be replayed (no envelope to cache), so the key is silently ignored
    // there.
    //
    // `lookup_and_mark` atomically inserts a `Pending` sentinel on miss and hands us a
    // ticket; concurrent same-keyed requests see `InFlight` and 409.  The ticket commits on
    // completion via the rollback-on-Drop pattern so a panic doesn't block retries forever.
    let body_hash = hash_body(&raw_body);
    let cacheable_key = if body.stream { None } else { idempotency_key };
    let idempotency_ticket: Option<crate::server::idempotency::IdempotencyTicket> =
        if let Some(key) = cacheable_key.as_deref() {
            match state
                .idempotency
                .lookup_and_mark(&principal.token_id, key, &body_hash)
                .await
            {
                LookupOutcome::Hit(entry) => {
                    tracing::debug!(
                        "idempotency hit: token={} key={} bytes={}",
                        principal.token_id,
                        key,
                        entry.body.len()
                    );
                    return Ok(cached_response_into_axum(entry));
                }
                LookupOutcome::Conflict => {
                    return Err(ProblemDetail::new(
                        ErrorKind::Idempotency,
                        StatusCode::CONFLICT,
                        "Idempotency-Key has been used with a different request body; replays \
                         must be byte-identical",
                    ));
                }
                LookupOutcome::InFlight => {
                    return Err(ProblemDetail::new(
                        ErrorKind::Idempotency,
                        StatusCode::CONFLICT,
                        "Idempotency-Key is in flight on a concurrent request; retry after it \
                         completes",
                    ));
                }
                LookupOutcome::CapExceeded => {
                    let mut problem = ProblemDetail::new(
                        ErrorKind::Idempotency,
                        StatusCode::TOO_MANY_REQUESTS,
                        "per-token idempotency-key cache is full; reduce the rate of unique \
                         keys or wait for in-flight requests to complete",
                    )
                    .with_retry_after(60);
                    // Override the generic "conflict" title: this is cache pressure, not a
                    // body-mismatch conflict.
                    problem.title = "Idempotency-Key cache capacity exceeded".to_string();
                    return Err(problem);
                }
                LookupOutcome::Miss(ticket) => Some(ticket),
            }
        } else {
            None
        };

    // Hold the sessions read-lock across both the map lookup and `TurnGuard::acquire` to
    // close the TOCTOU gap: DELETE's write-lock blocks behind any reader, so by the time
    // it fires we've already bumped `in_flight > 0` and DELETE's re-check returns 409.
    let (entry, turn_guard) = {
        let map = state.sessions.read().await;
        if let Some(entry) = map.get(&session_id).cloned() {
            let guard = TurnGuard::acquire(
                Arc::clone(&state.concurrent_turns),
                Arc::clone(&entry.in_flight),
                state.config.max_concurrent_turns,
            )?;
            (entry, guard)
        } else {
            drop(map);
            let entry = ensure_session_loaded(&state, session_id).await?;
            let guard = TurnGuard::acquire(
                Arc::clone(&state.concurrent_turns),
                Arc::clone(&entry.in_flight),
                state.config.max_concurrent_turns,
            )?;
            (entry, guard)
        }
    };

    // Reject a turn with nothing in it before hitting the provider. An image with no text is
    // allowed: against prior context "look at this" is a complete request, and ACP permits the
    // same (`run_prompt_turn` has no empty-text check).
    if body.message.trim().is_empty() && body.images.is_empty() {
        return Err(ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            "`message` must be a non-empty string, or at least one image must be attached",
        ));
    }
    let images = decode_turn_images(&body.images, state.shared.config.vision)?;
    let message = if let Some(skill_name) = body.options.skill.as_deref() {
        let snapshot = state.shared.skills.current().await;
        let skill = snapshot
            .iter()
            .find(|skill| skill.name == skill_name)
            .ok_or_else(|| {
                ProblemDetail::new(
                    ErrorKind::InvalidBody,
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!("unknown skill `{}`", skill_name),
                )
            })?;
        let skill_body = crate::skills::load_skill_body(skill)
            .await
            .map_err(|error| {
                ProblemDetail::new(
                    ErrorKind::Internal,
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to load skill `{}`: {}", skill_name, error),
                )
            })?;
        format!("{}\n\n{}", body.message, skill_body)
    } else {
        body.message
    };

    if body.stream {
        // SSE responses are streamed live and aren't a single envelope we can cache.
        run_streaming_turn(state, entry, session_id, message, images, turn_guard).await
    } else {
        let response_result =
            run_blocking_turn(entry, session_id, message, images, turn_guard).await;
        // Cache success (2xx) and client-error (4xx) envelopes. Skip server-side errors
        // (5xx) and TurnInFlight: a transient provider 502 or internal 500 would otherwise
        // be replayed for the full 24h TTL, defeating the point of idempotent retries.
        // TurnInFlight means the turn was never attempted at all.
        if let Some(ticket) = idempotency_ticket {
            let skip_cache = matches!(
                &response_result,
                Err(problem) if problem.status >= 500
                    || problem.type_uri == ErrorKind::TurnInFlight.type_uri()
            );
            if skip_cache {
                tracing::debug!(
                    session_id = %session_id,
                    "skipping idempotency cache commit for a 5xx or pre-attempt turn-in-flight \
                     response; ticket Drop clears the Pending entry so retries re-execute"
                );
            } else {
                let (status, bytes) = match &response_result {
                    Ok(json) => (StatusCode::OK, serde_json::to_vec(&json.0)),
                    Err(problem) => (
                        StatusCode::from_u16(problem.status)
                            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                        serde_json::to_vec(problem),
                    ),
                };
                if let Ok(bytes) = bytes {
                    ticket.commit(status.as_u16(), bytes).await;
                }
                // If serialization failed (extraordinarily unlikely; TurnResponse /
                // ProblemDetail are both pure-data serde types), drop the ticket without
                // commit so the Pending entry is removed and clients can retry instead of
                // hitting a permanent 409.
            }
        }
        response_result.map(IntoResponse::into_response)
    }
}

/// Validate and normalize a turn's image attachments through the shared client-image pipeline
/// ([`crate::image::decode_base64_image`]), so an HTTP attachment gets exactly the size cap and
/// format conversion an ACP `image` content block does.
///
/// `vision` is the active profile's resolved `[providers.<name>].vision` flag. Attachments are
/// refused outright when it is off, mirroring ACP, which rejects `image` content blocks with
/// `InvalidParams` for a text-only profile. This gates on configuration only: whether the named
/// model *actually* understands images is left to the provider to complain about, which is the
/// stance documented for ACP.
///
/// Every failure is a 422: these are all malformed input, not server faults.
///
/// The `allow` matches the rest of this module's validation helpers: `ProblemDetail` is a large
/// struct by design (RFC 9457 members plus an extensions map) and boxing it here alone would make
/// the error type inconsistent with every other handler.
#[allow(clippy::result_large_err)]
fn decode_turn_images(
    images: &[ImageInput],
    vision: bool,
) -> Result<Vec<crate::provider::ImageSource>, ProblemDetail> {
    if images.is_empty() {
        return Ok(Vec::new());
    }
    if !vision {
        return Err(ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            "image attachments require a provider profile with vision enabled; set `vision = \
             true` under `[providers.<name>]` or omit `images`",
        ));
    }
    images
        .iter()
        .enumerate()
        .map(|(index, image)| {
            crate::image::decode_base64_image(&image.data, &image.media_type).map_err(|message| {
                ProblemDetail::new(
                    ErrorKind::InvalidBody,
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!("`images[{}]` is invalid: {}", index, message),
                )
            })
        })
        .collect()
}

/// Extract the `Idempotency-Key` header, validating that it isn't empty and stays within
/// reasonable size bounds. Returns `Ok(None)` when the header is absent.
///
/// All validation failures map to 422 `invalid-body` so the status code is consistent with the
/// body-parse error path and matches the spec's error-catalogue table for `invalid-body`.
#[allow(clippy::result_large_err)]
fn idempotency_header(headers: &HeaderMap) -> Result<Option<String>, ProblemDetail> {
    let Some(value) = headers.get("idempotency-key") else {
        return Ok(None);
    };
    let value = value.to_str().map_err(|_| {
        ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            "Idempotency-Key header must be ASCII",
        )
    })?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            "Idempotency-Key header must not be empty",
        ));
    }
    if trimmed.len() > 255 {
        return Err(ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            "Idempotency-Key header is too long (max 255 chars)",
        ));
    }
    Ok(Some(trimmed.to_string()))
}

/// Re-build an `axum::Response` from a [`crate::server::idempotency::CachedResponse`]. Sets the
/// same status code the original handler returned, and picks the content-type to match: a 2xx
/// body is a serialised `TurnResponse` (`application/json`); a 4xx/5xx body is a serialised
/// `ProblemDetail` (`application/problem+json` per RFC 9457).
fn cached_response_into_axum(entry: crate::server::idempotency::CachedResponse) -> Response {
    let status = StatusCode::from_u16(entry.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let content_type = if status.is_success() {
        "application/json"
    } else {
        "application/problem+json"
    };
    (
        status,
        [(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static(content_type),
        )],
        axum::body::Body::from(entry.body),
    )
        .into_response()
}

/// RAII guard that clears the per-session `StreamSink` on drop so both normal completion
/// and panics reset the cell. Without this, a panic leaves a zero-subscriber sink that
/// causes subsequent blocking turns to 500 via `client_disconnected()`.
struct StreamGuard {
    frontend: Arc<crate::server::http_frontend::HttpFrontend>,
}

impl StreamGuard {
    fn new(frontend: Arc<crate::server::http_frontend::HttpFrontend>) -> Self {
        Self { frontend }
    }
}

impl Drop for StreamGuard {
    fn drop(&mut self) {
        self.frontend.clear_stream();
    }
}

async fn run_blocking_turn(
    entry: crate::server::state::SessionEntry,
    session_id: Uuid,
    message: String,
    images: Vec<crate::provider::ImageSource>,
    _turn_guard: TurnGuard,
) -> Result<Json<TurnResponse>, ProblemDetail> {
    let mut runtime = entry.runtime.try_lock().map_err(|_| {
        ProblemDetail::new(
            ErrorKind::TurnInFlight,
            StatusCode::CONFLICT,
            "another turn is already in flight on this session",
        )
        .with("session_id", session_id.to_string())
    })?;

    let _stale = entry.frontend.drain();

    // Publish the cancellation token *after* acquiring the mutex. Publishing before the lock
    // would let a rejected Turn B overwrite a running Turn A's token, making Turn A
    // uncancellable. The brief window between lock-acquire and this publish is harmless:
    // POST /cancel reading the old (session-creation or prior-turn) token is a no-op on an
    // already-finished turn.
    let cancellation = CancellationToken::new();
    {
        let mut guard = crate::server::poisoned::write(
            &entry.cancellation,
            "turn::blocking::publish_cancellation",
        );
        *guard = cancellation.clone();
    }
    let turn_id = uuid::Uuid::new_v4();
    let mut session_uuid_opt = Some(runtime.session_uuid);
    let runtime_inner = &mut *runtime;
    let outcome = runtime_inner
        .agent
        .run_turn(
            &mut session_uuid_opt,
            &mut runtime_inner.messages,
            message,
            images,
            cancellation,
        )
        .await;

    let recorder = entry.frontend.drain();
    entry.touch();

    match outcome {
        Ok(turn_outcome) => Ok(Json(assemble_response(
            turn_id,
            session_id,
            turn_outcome,
            recorder,
            entry.capabilities,
        ))),
        Err(error) => Err((&error).into()),
    }
}

/// Run a turn with `stream: true`. Returns an SSE response that emits events live as the agent
/// produces them, plus a terminal `turn.finished` (or `turn.failed` / `turn.cancelled`) event
/// before closing.
async fn run_streaming_turn(
    state: ServerState,
    entry: crate::server::state::SessionEntry,
    session_id: Uuid,
    message: String,
    images: Vec<crate::provider::ImageSource>,
    turn_guard: TurnGuard,
) -> Result<Response, ProblemDetail> {
    // Acquire the runtime mutex up front via `try_lock_owned`. The `OwnedMutexGuard` moves
    // into the spawned task so the lock holds continuously from the admission check to the
    // end of the turn.
    let owned_runtime = Arc::clone(&entry.runtime).try_lock_owned().map_err(|_| {
        ProblemDetail::new(
            ErrorKind::TurnInFlight,
            StatusCode::CONFLICT,
            "another turn is already in flight on this session",
        )
        .with("session_id", session_id.to_string())
    })?;

    // Subscribe to the broadcast BEFORE installing: install_stream returns a receiver that
    // captures the first event onwards.
    let _stale = entry.frontend.drain();
    let (receiver, ids) = entry.frontend.install_stream(256);

    // Publish after the lock succeeds. Same rationale as `run_blocking_turn`.
    let cancellation = CancellationToken::new();
    {
        let mut guard = crate::server::poisoned::write(
            &entry.cancellation,
            "turn::streaming::publish_cancellation",
        );
        *guard = cancellation.clone();
    }

    let turn_id = uuid::Uuid::new_v4();
    let entry_for_task = entry.clone();
    let cancel_for_task = cancellation.clone();

    // Spawn the turn so the SSE response can return immediately.
    //
    // Declare `runtime` last so it drops first (reverse order), keeping the mutex held
    // while `_stream_guard` and `_turn_guard` clean up.
    let join = tokio::spawn(async move {
        let _turn_guard = turn_guard;
        let mut runtime = owned_runtime;
        let _stream_guard = StreamGuard::new(Arc::clone(&entry_for_task.frontend));
        let runtime_inner = &mut *runtime;
        let mut session_uuid_opt = Some(runtime_inner.session_uuid);
        let outcome = runtime_inner
            .agent
            .run_turn(
                &mut session_uuid_opt,
                &mut runtime_inner.messages,
                message,
                images,
                cancel_for_task,
            )
            .await;
        entry_for_task.touch();
        let usage = drain_recorder_and_extract_usage(&entry_for_task.frontend);
        (outcome, usage)
    });

    // Build the SSE stream. Emits the per-FrontendEvent events from the broadcast, then a
    // terminal turn.finished/failed/cancelled when the join handle resolves. The shutdown
    // token tells the loop to emit `turn.cancelled{reason:"server_shutdown"}` during a
    // graceful drain; the per-session cancellation token races so `POST /cancel` closes the
    // stream promptly without waiting for the broadcast to drain.
    let stream = build_sse_stream(
        turn_id,
        session_id,
        receiver,
        join,
        cancellation,
        state.shutdown.clone(),
        ids,
    );
    let sse = Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(20))
            .text("keep-alive"),
    );

    let mut response = sse.into_response();
    response.headers_mut().insert(
        "X-Accel-Buffering",
        axum::http::HeaderValue::from_static("no"),
    );
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        axum::http::HeaderValue::from_static("no-cache, no-transform"),
    );
    // No explicit `Connection: keep-alive`: it's forbidden on HTTP/2 (RFC 9113 §8.2.2)
    // and hyper sets it automatically on HTTP/1.1.
    Ok(response)
}

#[allow(clippy::too_many_arguments)]
fn build_sse_stream(
    turn_id: Uuid,
    session_id: Uuid,
    mut receiver: tokio::sync::broadcast::Receiver<crate::server::sse::SseEvent>,
    join: tokio::task::JoinHandle<(crate::error::Result<TurnOutcome>, UsageView)>,
    cancellation: CancellationToken,
    shutdown: CancellationToken,
    ids: Arc<crate::server::sse::EventIdGenerator>,
) -> impl Stream<Item = Result<Event, Infallible>> + Send {
    async_stream::stream! {
        // Per spec §SSE production-concerns: hint clients to reconnect after 3s on disconnect.
        // Must be the first thing on the wire (before any `id:`/`event:` lines). The `retry:`
        // field has no `id:` by SSE spec.
        yield Ok::<_, Infallible>(Event::default().retry(std::time::Duration::from_secs(3)));

        // Emit the turn.started lifecycle event. The agent's own TurnStarted is filtered out
        // by sse.rs::translate; this richer envelope carries turn_id + session_id + started_at
        // for clients building lifecycle timelines. The id is drawn from the same generator the
        // broadcast events use so per-turn ids stay monotonic and dense.
        let started_at = chrono::Utc::now().to_rfc3339();
        let lifecycle = Event::default()
            .id(ids.next().to_string())
            .event("turn.started")
            .json_data(serde_json::json!({
                "turn_id": turn_id,
                "session_id": session_id,
                "started_at": started_at,
            }))
            .unwrap_or_else(|_| Event::default().comment("turn.started serialize-failed"));
        yield Ok(lifecycle);

        let mut join = Box::pin(join);
        let mut emitted_cancel: Option<&'static str> = None;
        loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled(), if emitted_cancel.is_none() => {
                    // Server-initiated drain. Trip the per-session cancel so the agent loop
                    // unwinds quickly, then mark the cancellation reason. The actual
                    // turn.cancelled event is emitted once the join handle resolves so a
                    // late content-block delta doesn't appear after it.
                    cancellation.cancel();
                    emitted_cancel = Some("server_shutdown");
                }
                _ = cancellation.cancelled(), if emitted_cancel.is_none() => {
                    // Client-initiated cancel (POST /cancel) or transitive from shutdown.
                    emitted_cancel = Some("client");
                }
                event = receiver.recv() => {
                    match event {
                        Ok(sse) => yield Ok(sse.into_axum()),
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                            tracing::warn!(
                                "SSE consumer lagged, skipped {} events; terminating stream",
                                skipped
                            );
                            // Cancel the turn so the agent doesn't keep burning provider
                            // tokens for a consumer that's lost data and will need to retry.
                            cancellation.cancel();
                            yield Ok(Event::default()
                                .id(ids.next().to_string())
                                .event("turn.failed")
                                .json_data(serde_json::json!({
                                    "turn_id": turn_id.to_string(),
                                    "session_id": session_id.to_string(),
                                    "error": {
                                        "type": "https://meka.so/errors/sse-lag",
                                        "title": "SSE consumer lagged",
                                        "status": 500,
                                        "detail": format!(
                                            "SSE consumer fell behind; {} event(s) were dropped. \
                                             The stream is terminated to prevent an incomplete \
                                             transcript. Retry the turn.",
                                            skipped
                                        ),
                                    },
                                }))
                                .unwrap_or_else(|_| Event::default().comment("lag-failed serialize-failed")));
                            break;
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                            let turn_result = (&mut join).await;
                            let (outcome, usage) = match turn_result {
                                Ok((outcome, usage)) => (Ok(outcome), usage),
                                Err(panic) => (Err(panic), UsageView::default()),
                            };
                            yield Ok(terminal_event_from_join(
                                outcome,
                                emitted_cancel,
                                usage,
                                &ids,
                                turn_id,
                                session_id,
                            ));
                            break;
                        }
                    }
                }
                turn_result = &mut join => {
                    // Agent finished; flush remaining buffered events before the terminal one.
                    while let Ok(sse) = receiver.try_recv() {
                        yield Ok(sse.into_axum());
                    }
                    let (outcome, usage) = match turn_result {
                        Ok((outcome, usage)) => (Ok(outcome), usage),
                        Err(panic) => (Err(panic), UsageView::default()),
                    };
                    yield Ok(terminal_event_from_join(
                        outcome,
                        emitted_cancel,
                        usage,
                        &ids,
                        turn_id,
                        session_id,
                    ));
                    break;
                }
            }
        }
    }
}

/// Resolve a join handle outcome into the matching terminal SSE event. A successful agent
/// outcome always wins over a concurrent cancel signal so that a race between completion
/// and cancellation doesn't discard an already-persisted result.
fn terminal_event_from_join(
    turn_result: std::result::Result<crate::error::Result<TurnOutcome>, tokio::task::JoinError>,
    cancel_reason: Option<&'static str>,
    usage: UsageView,
    ids: &Arc<crate::server::sse::EventIdGenerator>,
    turn_id: Uuid,
    session_id: Uuid,
) -> Event {
    if let Ok(Ok(outcome)) = &turn_result {
        return terminal_event_for_outcome(outcome, usage, ids, turn_id, session_id);
    }
    if let Some(reason) = cancel_reason {
        return cancelled_event(reason, ids, turn_id, session_id);
    }
    match turn_result {
        Ok(Ok(_)) => unreachable!("already handled above"),
        Ok(Err(crate::error::MekaError::Interrupted)) => {
            cancelled_event("client", ids, turn_id, session_id)
        }
        Ok(Err(error)) => {
            let instance = format!("/v1/sessions/{}/turn", session_id);
            let problem =
                crate::server::errors::ProblemDetail::from(&error).instance(instance.clone());
            Event::default()
                .id(ids.next().to_string())
                .event("turn.failed")
                .json_data(serde_json::json!({
                    "turn_id": turn_id.to_string(),
                    "session_id": session_id.to_string(),
                    "error": serde_json::to_value(problem)
                        .unwrap_or(serde_json::Value::Null),
                }))
                .unwrap_or_else(|_| Event::default().comment("failed serialize-failed"))
        }
        Err(panic) => {
            tracing::error!("turn task panicked: {:?}", panic);
            Event::default()
                .id(ids.next().to_string())
                .event("turn.failed")
                .json_data(serde_json::json!({
                    "turn_id": turn_id.to_string(),
                    "session_id": session_id.to_string(),
                    "error": {
                        "type": "https://meka.so/errors/internal",
                        "title": "Internal server error",
                        "status": 500,
                        "detail": "turn task panicked",
                        "instance": format!("/v1/sessions/{}/turn", session_id),
                    },
                }))
                .unwrap_or_else(|_| Event::default().comment("panic serialize-failed"))
        }
    }
}

fn cancelled_event(
    reason: &'static str,
    ids: &Arc<crate::server::sse::EventIdGenerator>,
    turn_id: Uuid,
    session_id: Uuid,
) -> Event {
    Event::default()
        .id(ids.next().to_string())
        .event("turn.cancelled")
        .json_data(serde_json::json!({
            "turn_id": turn_id.to_string(),
            "session_id": session_id.to_string(),
            "reason": reason,
        }))
        .unwrap_or_else(|_| Event::default().comment("cancelled serialize-failed"))
}

/// Wire `stop_reason` string for a finished turn. Shared by the blocking (`assemble_response`)
/// and streaming (`terminal_event_for_outcome`) paths so the two can't drift.
fn stop_reason_str(outcome: &TurnOutcome) -> &'static str {
    match outcome {
        TurnOutcome::EndTurn => "end_turn",
        TurnOutcome::MaxTokens => "max_tokens",
        TurnOutcome::Refusal(_) => "refusal",
    }
}

fn terminal_event_for_outcome(
    outcome: &TurnOutcome,
    usage: UsageView,
    ids: &Arc<crate::server::sse::EventIdGenerator>,
    turn_id: Uuid,
    session_id: Uuid,
) -> Event {
    let stop_reason = stop_reason_str(outcome);
    let mut data = serde_json::json!({
        "turn_id": turn_id.to_string(),
        "session_id": session_id.to_string(),
        "stop_reason": stop_reason,
    });
    if let TurnOutcome::Refusal(text) = outcome
        && !text.is_empty()
        && let Some(obj) = data.as_object_mut()
    {
        obj.insert(
            "refusal_text".into(),
            serde_json::Value::String(text.clone()),
        );
    }
    // Always emit `usage` so clients don't have to handle a conditionally-absent field.
    if let Some(obj) = data.as_object_mut()
        && let Ok(value) = serde_json::to_value(&usage)
    {
        obj.insert("usage".into(), value);
    }
    Event::default()
        .id(ids.next().to_string())
        .event("turn.finished")
        .json_data(data)
        .unwrap_or_else(|_| Event::default().comment("finished serialize-failed"))
}

/// Drain the per-session recorder at end-of-turn and pluck the most recent `TokenUsage` event
/// off the back. Mirrors what `run_blocking_turn` does explicitly via `entry.frontend.drain()`.
/// Both transport branches reset the recorder so the next turn starts clean. Returns `None`
/// when the turn never reported usage (mock provider tests, refused turns, server-shutdown
/// cancel before the agent emitted anything).
///
/// Only one of the two select-arm callers ever runs per turn (terminal events break the loop),
/// so the drain happens exactly once.
fn drain_recorder_and_extract_usage(
    frontend: &Arc<crate::server::http_frontend::HttpFrontend>,
) -> UsageView {
    let recorder = frontend.drain();
    recorder
        .into_iter()
        .rev()
        .find_map(|event| {
            if let FrontendEvent::TokenUsage(usage) = event {
                Some(UsageView {
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                    cache_creation_input_tokens: usage.cache_creation_input_tokens,
                    cache_read_input_tokens: usage.cache_read_input_tokens,
                })
            } else {
                None
            }
        })
        .unwrap_or_default()
}

fn assemble_response(
    turn_id: Uuid,
    session_id: Uuid,
    outcome: TurnOutcome,
    recorder: Recorder,
    capabilities: crate::server::http_frontend::SessionCapabilities,
) -> TurnResponse {
    let stop_reason = stop_reason_str(&outcome).to_string();

    let mut final_text = String::new();
    let mut tool_calls_by_id: std::collections::HashMap<String, ToolCallView> =
        std::collections::HashMap::new();
    let mut tool_call_order: Vec<String> = Vec::new();
    let mut usage = UsageView::default();
    let mut notices: Vec<NoticeView> = Vec::new();
    let mut thinking_segments: Vec<String> = Vec::new();

    for event in recorder {
        match event {
            FrontendEvent::AssistantTextDelta(text) => {
                final_text.push_str(&text);
            }
            FrontendEvent::ThinkingBlock { content, .. }
                if capabilities.supports_reasoning_stream =>
            {
                thinking_segments.push(content);
            }
            FrontendEvent::ToolCallStarted {
                id,
                name,
                input,
                display_summary,
            } => {
                tool_call_order.push(id.clone());
                tool_calls_by_id.insert(id.clone(), ToolCallView {
                    id,
                    name,
                    input,
                    display_summary,
                    is_error: false,
                    content: Vec::new(),
                });
            }
            FrontendEvent::ToolCallCompleted {
                id,
                is_error,
                content,
                ..
            } => {
                if let Some(view) = tool_calls_by_id.get_mut(&id) {
                    view.is_error = is_error;
                    view.content = content
                        .into_iter()
                        .map(|item| match item {
                            ToolResultContent::Text { text } => ToolCallContentView::Text { text },
                            ToolResultContent::Image { source } => ToolCallContentView::Image {
                                media_type: source.media_type,
                            },
                        })
                        .collect();
                }
            }
            FrontendEvent::TokenUsage(token_usage) => {
                // Last-wins assignment: the agent emits exactly one `TokenUsage` per turn
                // (accumulated total). If that ever changes, switch to `saturating_add`.
                usage.input_tokens = token_usage.input_tokens;
                usage.output_tokens = token_usage.output_tokens;
                usage.cache_creation_input_tokens = token_usage.cache_creation_input_tokens;
                usage.cache_read_input_tokens = token_usage.cache_read_input_tokens;
            }
            FrontendEvent::Notice(notice) => {
                notices.push(NoticeView::from(notice));
            }
            // Remaining lifecycle / UI-chrome variants (TurnStarted/Finished, TodoListUpdated,
            // McpProgress, SessionStarted) aren't part of the blocking JSON envelope.
            _ => {}
        }
    }

    // Mark orphan tool calls (started but never completed) as errors so clients can
    // distinguish "tool returned nothing" from "interrupted mid-execution".
    for view in tool_calls_by_id.values_mut() {
        if !view.is_error && view.content.is_empty() {
            view.is_error = true;
            view.content = vec![ToolCallContentView::Text {
                text: "tool execution interrupted before completion".to_string(),
            }];
        }
    }

    let refusal_text = match &outcome {
        TurnOutcome::Refusal(text) if !text.is_empty() => Some(text.clone()),
        _ => None,
    };

    let tool_calls = tool_call_order
        .into_iter()
        .filter_map(|id| tool_calls_by_id.remove(&id))
        .collect();

    let mut content_blocks: Vec<crate::server::handlers::messages::ContentBlockView> = Vec::new();
    for segment in thinking_segments {
        content_blocks.push(
            crate::server::handlers::messages::ContentBlockView::Thinking { thinking: segment },
        );
    }
    if !final_text.is_empty() {
        content_blocks.push(crate::server::handlers::messages::ContentBlockView::Text {
            text: final_text.clone(),
        });
    }
    let messages = if content_blocks.is_empty() {
        Vec::new()
    } else {
        vec![crate::server::handlers::messages::MessageView {
            role: "assistant".to_string(),
            content: content_blocks,
            // Not available yet: the DB write may still be in progress.
            created_at: None,
            // Only the current message is available; full history index lives on
            // `GET /v1/sessions/{id}/messages`.
            turn_id: None,
        }]
    };

    TurnResponse {
        turn_id,
        session_id,
        stop_reason,
        final_text,
        refusal_text,
        messages,
        tool_calls,
        usage,
        notices,
    }
}

impl From<Notice> for NoticeView {
    fn from(notice: Notice) -> Self {
        Self {
            level: match notice.level {
                NoticeLevel::Info => "info".to_string(),
                NoticeLevel::Warn => "warn".to_string(),
            },
            text: notice.text,
        }
    }
}

/// `POST /v1/sessions/{id}/cancel`: interrupt the in-flight turn (if any) by firing the
/// session's cancellation token. Always returns 204 even if no turn is in flight (the
/// operation is idempotent and absence is observationally indistinguishable from a turn that
/// finished microseconds before the cancel arrived).
#[utoipa::path(
    post,
    path = "/v1/sessions/{id}/cancel",
    tag = "turn",
    params(("id" = Uuid, Path, description = "Session UUID")),
    responses(
        (status = 204, description = "Cancellation token fired (idempotent)"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn cancel_turn(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    Path(session_id): Path<Uuid>,
) -> Result<StatusCode, ProblemDetail> {
    if !principal.has_scope("sessions:w") {
        return Err(ProblemDetail::new(
            ErrorKind::AuthScope,
            StatusCode::FORBIDDEN,
            "scope `sessions:w` is required",
        ));
    }
    // Fast-path: look up the in-memory session map directly. If the session was GC-evicted
    // (no in-memory entry), there's no in-flight turn to cancel. Return 204 idempotently
    // instead of re-attaching from disk (which would build an unconnected cancellation token
    // and waste a file-lock + DB load).
    let entry = state.sessions.read().await.get(&session_id).cloned();

    if let Some(entry) = entry {
        let token =
            crate::server::poisoned::read(&entry.cancellation, "cancel::read_token").clone();
        token.cancel();
    }
    // 204 whether or not there was anything to cancel: POST /cancel is idempotent.
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{frontend::FrontendEvent, provider::Notice};

    #[test]
    fn assemble_response_concatenates_text_deltas() {
        let recorder: Recorder = vec![
            FrontendEvent::AssistantTextDelta("Hello ".into()),
            FrontendEvent::AssistantTextDelta("world".into()),
        ];
        let response = assemble_response(
            Uuid::nil(),
            Uuid::nil(),
            TurnOutcome::EndTurn,
            recorder,
            crate::server::http_frontend::SessionCapabilities::default(),
        );
        assert_eq!(response.final_text, "Hello world");
        assert_eq!(response.stop_reason, "end_turn");
        assert!(response.tool_calls.is_empty());
    }

    #[test]
    fn assemble_response_pairs_tool_calls_with_completion() {
        let input = serde_json::json!({"path": "src/main.rs"});
        let recorder: Recorder = vec![
            FrontendEvent::ToolCallStarted {
                id: "tu_1".into(),
                name: "read_file".into(),
                input: input.clone(),
                display_summary: Some("src/main.rs".into()),
            },
            FrontendEvent::ToolCallCompleted {
                id: "tu_1".into(),
                name: "read_file".into(),
                is_error: false,
                content: vec![ToolResultContent::Text {
                    text: "fn main() {}".into(),
                }],
                metadata: None,
            },
        ];
        let response = assemble_response(
            Uuid::nil(),
            Uuid::nil(),
            TurnOutcome::EndTurn,
            recorder,
            crate::server::http_frontend::SessionCapabilities::default(),
        );
        assert_eq!(response.tool_calls.len(), 1);
        let call = &response.tool_calls[0];
        assert_eq!(call.id, "tu_1");
        assert_eq!(call.name, "read_file");
        assert_eq!(call.input, input);
        assert!(!call.is_error);
        match &call.content[0] {
            ToolCallContentView::Text { text } => assert_eq!(text, "fn main() {}"),
            other => panic!("expected text content, got {:?}", other),
        }
    }

    #[test]
    fn assemble_response_surfaces_notices() {
        let recorder: Recorder = vec![FrontendEvent::Notice(Notice::warn("auto-denied tool"))];
        let response = assemble_response(
            Uuid::nil(),
            Uuid::nil(),
            TurnOutcome::EndTurn,
            recorder,
            crate::server::http_frontend::SessionCapabilities::default(),
        );
        assert_eq!(response.notices.len(), 1);
        assert_eq!(response.notices[0].level, "warn");
        assert_eq!(response.notices[0].text, "auto-denied tool");
    }

    #[test]
    fn assemble_response_separates_refusal_text_from_final_text() {
        let recorder: Recorder = vec![FrontendEvent::AssistantTextDelta(
            "I cannot help with that.".into(),
        )];
        let response = assemble_response(
            Uuid::nil(),
            Uuid::nil(),
            TurnOutcome::Refusal("policy violation".into()),
            recorder,
            crate::server::http_frontend::SessionCapabilities::default(),
        );
        assert_eq!(response.stop_reason, "refusal");
        assert_eq!(response.final_text, "I cannot help with that.");
        assert_eq!(response.refusal_text.as_deref(), Some("policy violation"));
    }

    #[test]
    fn assemble_response_omits_refusal_text_on_normal_stop() {
        let recorder: Recorder = vec![FrontendEvent::AssistantTextDelta("hello".into())];
        let response = assemble_response(
            Uuid::nil(),
            Uuid::nil(),
            TurnOutcome::EndTurn,
            recorder,
            crate::server::http_frontend::SessionCapabilities::default(),
        );
        assert_eq!(response.refusal_text, None);
        assert_eq!(response.final_text, "hello");
    }

    fn png_input() -> ImageInput {
        use base64::Engine as _;
        let image = image::RgbaImage::from_pixel(4, 4, image::Rgba([10, 20, 30, 255]));
        let mut bytes = Vec::new();
        image
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .expect("encode png");
        ImageInput {
            media_type: "image/png".to_string(),
            data: base64::engine::general_purpose::STANDARD.encode(&bytes),
        }
    }

    #[test]
    fn decode_turn_images_accepts_a_png() {
        let decoded = decode_turn_images(&[png_input()], true).expect("should decode");
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].media_type, "image/png");
        assert_eq!(decoded[0].source_type, "base64");
        assert!(!decoded[0].data.is_empty());
    }

    #[test]
    fn decode_turn_images_is_a_noop_without_attachments() {
        // The vision flag is irrelevant when nothing is attached: a text-only profile must still
        // be able to take ordinary turns.
        assert!(
            decode_turn_images(&[], false)
                .expect("no images")
                .is_empty()
        );
    }

    #[test]
    fn decode_turn_images_rejects_attachments_when_vision_is_off() {
        let problem = decode_turn_images(&[png_input()], false).expect_err("should reject");
        assert_eq!(problem.status, 422);
        assert!(problem.detail.unwrap_or_default().contains("vision"));
    }

    #[test]
    fn decode_turn_images_rejects_invalid_base64() {
        let bad = ImageInput {
            media_type: "image/png".to_string(),
            data: "!!!not-base64!!!".to_string(),
        };
        let problem = decode_turn_images(&[bad], true).expect_err("should reject");
        assert_eq!(problem.status, 422);
        assert!(problem.detail.unwrap_or_default().contains("images[0]"));
    }

    /// The offending index is named so a client sending several attachments knows which one to
    /// fix, rather than being told only that "an image" was bad.
    #[test]
    fn decode_turn_images_names_the_failing_index() {
        use base64::Engine as _;
        let garbage = ImageInput {
            media_type: "application/octet-stream".to_string(),
            data: base64::engine::general_purpose::STANDARD.encode(b"not an image"),
        };
        let problem = decode_turn_images(&[png_input(), garbage], true).expect_err("should reject");
        assert_eq!(problem.status, 422);
        let detail = problem.detail.unwrap_or_default();
        assert!(detail.contains("images[1]"), "{}", detail);
    }

    /// A declared MIME type that names no supported format still decodes when the payload's magic
    /// bytes do, so a client that labels its upload `application/octet-stream` isn't stuck.
    #[test]
    fn decode_turn_images_falls_back_to_magic_bytes() {
        let mut input = png_input();
        input.media_type = "application/octet-stream".to_string();
        let decoded = decode_turn_images(&[input], true).expect("should decode via magic bytes");
        assert_eq!(decoded[0].media_type, "image/png");
    }

    #[test]
    fn decode_turn_images_rejects_oversized_payloads() {
        use base64::Engine as _;
        let raw = vec![0u8; crate::image::MAX_IMAGE_RAW_BYTES + 1];
        let oversized = ImageInput {
            media_type: "image/png".to_string(),
            data: base64::engine::general_purpose::STANDARD.encode(&raw),
        };
        let problem = decode_turn_images(&[oversized], true).expect_err("should reject");
        assert_eq!(problem.status, 422);
    }
}
