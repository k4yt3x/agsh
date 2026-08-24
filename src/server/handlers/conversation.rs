//! Operations that reshape or read out a session's conversation log: compaction, rewind, context
//! occupancy, and export / import.
//!
//! What the four share is that they are the HTTP surface for machinery the server already runs
//! (auto-compaction) or the CLI already exposes (`meka session export`), and that the REPL reaches
//! through `/compact`, `/rewind` and `/export`. None of them are new capability; they are the
//! missing way to ask for it over the wire.
//!
//! Compaction and rewind mutate the *live* conversation, so both take the session runtime mutex
//! and refuse while a turn is in flight. Reading the DB copy and writing it back would be wrong in
//! a way that is silent: a resident session holds its own `Conversation` in memory and would
//! overwrite the change on its next turn. `meka session rewind` guards the same hazard with an
//! on-disk `lock_session` because it runs in a separate process (`src/main.rs`).

use axum::{
    Extension, Json,
    body::Bytes,
    extract::{Path, Query, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;
use utoipa::ToSchema;
use uuid::Uuid;

use crate::{
    agent::{CompactOrigin, CompactRequest, CompactSource},
    server::{
        auth::Principal,
        errors::{ErrorKind, ProblemDetail},
        handlers::sessions::turn_in_flight_conflict,
        reattach::{ensure_session_loaded, require_session_exists},
        scope,
        state::ServerState,
    },
};

#[derive(Debug, Deserialize, ToSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct CompactRequestBody {
    /// Free-text guidance on what to preserve or drop, the wire equivalent of
    /// `/compact <instructions>`. Reaches the checkpoint turn and the fallback summariser alike.
    #[serde(default)]
    pub instructions: Option<String>,
    /// Whether to keep the most recent turns verbatim after the summary. Omit to let meka decide;
    /// the checkpoint turn overrides this when it knows better, having just read the conversation.
    #[serde(default)]
    pub keep_recent: Option<bool>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct CompactResponse {
    pub session_id: Uuid,
    /// Which strategy produced the summary: `checkpoint`, `checkpoint_text`, or `summarizer`.
    /// Reported because they differ in fidelity, not just in mechanism: `summarizer` means the
    /// checkpoint turn was disabled, failed, or produced nothing usable.
    pub source: String,
    /// Memories the checkpoint turn wrote, observed from its `memory_write` calls rather than
    /// self-reported, so this cannot disagree with what actually landed on disk.
    pub memories_written: Vec<String>,
    /// Whether the recent turns were kept verbatim after the summary.
    pub kept_recent: bool,
    pub messages_before: usize,
    pub messages_after: usize,
}

/// `POST /v1/sessions/{id}/compact`: summarise the conversation now.
#[utoipa::path(
    post,
    path = "/v1/sessions/{id}/compact",
    tag = "conversation",
    params(("id" = Uuid, Path, description = "Session UUID")),
    request_body = CompactRequestBody,
    responses(
        (status = 200, description = "Compaction completed", body = CompactResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
        (status = 409, description = "Turn in flight; cancel first", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid body, or nothing to compact", body = ProblemDetail),
        (status = 502, description = "Provider call failed", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn compact(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    Path(id): Path<Uuid>,
    raw_body: Bytes,
) -> Result<Json<CompactResponse>, ProblemDetail> {
    scope::require(&principal, "sessions:w")?;

    // An empty body means "compact with no guidance", which is the common case; `serde_json` would
    // reject the zero-length slice, so treat it as `{}` rather than making every client send one.
    let body: CompactRequestBody = if raw_body.is_empty() {
        CompactRequestBody::default()
    } else {
        serde_json::from_slice(&raw_body).map_err(|error| {
            ProblemDetail::new(
                ErrorKind::InvalidBody,
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("invalid compact request body: {}", error),
            )
        })?
    };

    // Acquired under the sessions read-lock, the way `submit_turn` does it (see the note there).
    // `DELETE`'s write-lock blocks behind any reader, so taking the guard inside the read lock is
    // what makes DELETE's own `in_flight` re-check see this operation. Acquiring it after the lock
    // is dropped would let a concurrent DELETE remove the row and the map entry first, leaving
    // this to run a multi-minute checkpoint against a session that no longer exists and to persist
    // a boundary event for it.
    let (entry, _in_flight) = {
        let map = state.sessions.read().await;
        match map.get(&id).cloned() {
            Some(entry) => {
                let guard = crate::server::state::InFlightGuard::acquire(&entry).map_err(|_| {
                    turn_in_flight_conflict(
                        id,
                        "session is busy; cancel the turn first via POST /v1/sessions/{id}/cancel \
                         before compacting",
                    )
                })?;
                (entry, guard)
            }
            // Not resident: `ensure_session_loaded` below re-attaches it, and a session that is
            // not in the map cannot be racing a turn, so there is nothing to guard against yet.
            None => {
                drop(map);
                let entry = ensure_session_loaded(&state, id).await?;
                let guard = crate::server::state::InFlightGuard::acquire(&entry).map_err(|_| {
                    turn_in_flight_conflict(
                        id,
                        "session is busy; cancel the turn first via POST /v1/sessions/{id}/cancel \
                         before compacting",
                    )
                })?;
                (entry, guard)
            }
        }
    };

    // `try_lock`, not `lock().await`. The CAS above catches a turn that started first, but a turn
    // that starts in the window between the CAS and here wins the mutex, and blocking would then
    // park this request for the length of that turn -- and, for rewind, drop the turn that just
    // succeeded instead of the one the caller meant. Out-of-band turns (the scheduler,
    // background-outcome delivery) take the mutex before marking themselves busy, so this is the
    // check that catches one in that window.
    let mut runtime = entry.runtime.try_lock().map_err(|_| {
        turn_in_flight_conflict(
            id,
            "session is busy; cancel the turn first via POST /v1/sessions/{id}/cancel before \
             compacting",
        )
    })?;
    let messages_before = runtime.messages.len();
    let request = CompactRequest {
        // `Manual` and not `Requested`: the origin distinguishes who asked, and an API caller is
        // standing in for the human at the keyboard, not for the model asking on its own behalf.
        origin: CompactOrigin::Manual,
        instructions: body.instructions,
        keep_recent: body.keep_recent,
    };
    // A *fresh* token, published the way `submit_turn` publishes one, rather than a clone of
    // whatever the last turn left behind.
    //
    // `entry.cancellation` holds the previous turn's token, and a turn that was cancelled leaves
    // it in the fired state. Inheriting it would start the checkpoint turn already cancelled, so
    // `run_checkpoint_turn` returns immediately and compaction silently falls back to the
    // standalone summariser -- no memories written, a worse summary, and a `warn` as the only
    // trace. "Cancel the slow turn, then compact to free the window" is an ordinary thing to do.
    //
    // Publishing it keeps `POST /cancel` and the shutdown drain working: both fire whatever token
    // is in this cell, which is now this compaction's.
    let cancellation = CancellationToken::new();
    {
        let mut guard =
            crate::server::poisoned::write(&entry.cancellation, "compact::publish_cancellation");
        *guard = cancellation.clone();
    }
    let session_uuid = runtime.session_uuid;
    let mut session_id = Some(session_uuid);
    let (outcome, messages_after) = {
        let runtime = &mut *runtime;
        let outcome = runtime
            .agent
            .compact_session(
                &mut session_id,
                &mut runtime.messages,
                request,
                cancellation,
            )
            .await
            .map_err(|error| ProblemDetail::from(&error).with("session_id", id.to_string()))?;
        (outcome, runtime.messages.len())
    };
    // Releases the guard, not a reborrow of it: `drop(runtime)` on the `&mut *runtime` above is a
    // no-op, which is what the `dropping_references` lint caught.
    drop(runtime);
    // The checkpoint turn emits provider notices and the `Compacted` event into the session's
    // recorder, and nothing here consumes them. Every other path that runs a turn outside a
    // request drains for the same reason (`schedule::run_prompt_in_session`): left alone they
    // accumulate across repeated compactions and then surface in whichever turn drains next.
    let _checkpoint_events = entry.frontend.drain();
    // Same reason every turn path touches: this both advances the `updated_at` clients poll for
    // change detection, which the boundary write already moved on the DB row, and resets the GC
    // idle timer. Without it a session compacted just shy of `idle_timeout` is evicted on the next
    // scan, throwing away the context gauge `compact_session` has just re-seeded.
    entry.touch();

    tracing::info!(
        "compacted session {} via HTTP: {} -> {} messages",
        id,
        messages_before,
        messages_after
    );

    Ok(Json(CompactResponse {
        session_id: id,
        source: match outcome.source {
            CompactSource::Checkpoint => "checkpoint",
            CompactSource::CheckpointText => "checkpoint_text",
            CompactSource::Summarizer => "summarizer",
        }
        .to_string(),
        memories_written: outcome.memories_written,
        kept_recent: outcome.kept_recent,
        messages_before,
        messages_after,
    }))
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ContextResponse {
    pub session_id: Uuid,
    /// Tokens behind the most recent provider round, or `null` when no turn has run since this
    /// process loaded the session.
    ///
    /// Null rather than zero, deliberately. A re-attached session has a full conversation and an
    /// unmeasured window, and reporting `0` there would read as "empty" to every client that
    /// divides by `window`. Run a turn, or read `total_input_tokens` for the cumulative figure
    /// that does survive a restart.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub used: Option<u64>,
    /// The model's context window, or `null` when meka has no metadata for it. A percentage of an
    /// unknown denominator is worse than silence, so clients should suppress occupancy rather than
    /// assume a default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window: Option<u64>,
    /// Occupancy percent, present only when both `used` and `window` are known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub used_percent: Option<u64>,
    /// Estimated system prompt + tool schemas: the part of the window compaction cannot reclaim.
    ///
    /// Absent, not zero, when nothing has measured it. Same reasoning as `used`: the counter is
    /// only stamped mid-turn, so a session evicted or re-attached since its last turn has a real
    /// overhead of several thousand tokens and a recorded one of zero. Reporting `0` would make a
    /// client computing `used - overhead` confidently wrong rather than visibly uninformed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overhead: Option<u64>,
    /// Occupancy at which auto-compaction fires, or `null` when it is switched off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compact_at_percent: Option<u64>,
    /// How many times this session has already been compacted. Fidelity degrades with each pass,
    /// so a client deciding whether to fork rather than compact again wants this.
    pub generation: u64,
    /// Messages currently in the materialised window (post-compaction), not the full history.
    ///
    /// Absent while a turn holds the conversation. Everything else here is read from atomics and
    /// the database, so occupancy stays answerable during a turn; this one field genuinely needs
    /// the log, and blocking the whole response on it would defeat the point of asking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_count: Option<usize>,
    /// Cumulative turns and tokens for this session, read from the DB and so unaffected by
    /// eviction or restart.
    pub totals: SessionTotals,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct SessionTotals {
    pub turns: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_read_input_tokens: u64,
}

/// `GET /v1/sessions/{id}/context`: live window occupancy plus cumulative usage.
#[utoipa::path(
    get,
    path = "/v1/sessions/{id}/context",
    tag = "conversation",
    params(("id" = Uuid, Path, description = "Session UUID")),
    responses(
        (status = 200, description = "Context occupancy", body = ContextResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn context(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    Path(id): Path<Uuid>,
) -> Result<Json<ContextResponse>, ProblemDetail> {
    scope::require(&principal, "sessions:r")?;

    // Deliberately does NOT revive. Re-attaching takes the session's cross-process file lock and
    // holds it until the GC evicts the entry, which defaults to 24 hours -- so a `sessions:r`
    // token calling this once would lock the operator out of `meka -r` on their own session for a
    // day, and could do it to every session it can list. A read scope must not be able to seize a
    // write-exclusive resource. An evicted session still answers from the database; only the live
    // counters are missing, and they are already `Option` for exactly that reason.
    require_session_exists(&state, id).await?;
    let entry = state.sessions.read().await.get(&id).cloned();
    let stats = state
        .shared
        .session_manager
        .load_session_stats(id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to load session stats", error)
                .with("session_id", id.to_string())
        })?;
    // Counted from the database rather than the agent's cache, because the cache lives behind the
    // runtime mutex and this is the same figure `Agent::compaction_generation` would seed itself
    // from.
    let generation = state
        .shared
        .session_manager
        .count_compactions(id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to count compactions", error)
                .with("session_id", id.to_string())
        })?;

    // Atomics, so no lock. `try_lock` for the message count alone: a turn in flight is exactly
    // when headroom is worth asking about, and waiting on it would turn this into a request that
    // hangs for the length of a turn.
    let used_raw = entry
        .as_ref()
        .map(|entry| {
            entry
                .context_used
                .load(std::sync::atomic::Ordering::Relaxed)
        })
        .unwrap_or(0);
    let overhead_raw = entry
        .as_ref()
        .map(|entry| {
            entry
                .context_overhead
                .load(std::sync::atomic::Ordering::Relaxed)
        })
        .unwrap_or(0);
    let message_count = entry
        .as_ref()
        .and_then(|entry| entry.runtime.try_lock().ok())
        .map(|runtime| runtime.messages.len());

    // `agent_options`, not the raw config: this is the window the agent actually runs with, after
    // profile precedence and the default. Reading `config.context_window` would report the
    // unresolved `Option` and disagree with the gauge it is being divided into.
    let options = &state.shared.agent_options;
    let compact_at_percent = options
        .auto_compact
        .then_some(crate::agent::AUTO_COMPACT_THRESHOLD_PERCENT);
    let used = (used_raw > 0).then_some(used_raw);
    let overhead = (overhead_raw > 0).then_some(overhead_raw);
    let window = (options.context_window > 0).then_some(options.context_window);
    Ok(Json(ContextResponse {
        session_id: id,
        used,
        window,
        used_percent: match (used, window) {
            (Some(used), Some(window)) => Some(used.saturating_mul(100) / window),
            _ => None,
        },
        overhead,
        compact_at_percent,
        generation,
        message_count,
        totals: SessionTotals {
            turns: stats.turns,
            input_tokens: stats.input_tokens,
            output_tokens: stats.output_tokens,
            cache_creation_input_tokens: stats.cache_creation_input_tokens,
            cache_read_input_tokens: stats.cache_read_input_tokens,
        },
    }))
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct RewindRequestBody {
    /// How many trailing turns to drop. A turn starts at a user message that is not a tool result,
    /// so dropping one removes that message and every assistant / tool-result message after it.
    #[serde(default = "default_turns")]
    pub turns: usize,
}

fn default_turns() -> usize {
    1
}

#[derive(Debug, Serialize, ToSchema)]
pub struct RewindResponse {
    pub session_id: Uuid,
    pub turns_removed: usize,
    pub messages_before: usize,
    pub messages_after: usize,
}

/// `POST /v1/sessions/{id}/rewind`: drop trailing turns from the conversation.
#[utoipa::path(
    post,
    path = "/v1/sessions/{id}/rewind",
    tag = "conversation",
    params(("id" = Uuid, Path, description = "Session UUID")),
    request_body = RewindRequestBody,
    responses(
        (status = 200, description = "Turns removed", body = RewindResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
        (status = 409, description = "Turn in flight; cancel first", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid body, or fewer turns than requested", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn rewind(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    Path(id): Path<Uuid>,
    raw_body: Bytes,
) -> Result<Json<RewindResponse>, ProblemDetail> {
    scope::require(&principal, "sessions:w")?;

    let body: RewindRequestBody = if raw_body.is_empty() {
        RewindRequestBody { turns: 1 }
    } else {
        serde_json::from_slice(&raw_body).map_err(|error| {
            ProblemDetail::new(
                ErrorKind::InvalidBody,
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("invalid rewind request body: {}", error),
            )
        })?
    };
    if body.turns == 0 {
        return Err(ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            "`turns` must be at least 1",
        ));
    }

    // Acquired under the sessions read-lock, the way `submit_turn` does it (see the note there).
    // `DELETE`'s write-lock blocks behind any reader, so taking the guard inside the read lock is
    // what makes DELETE's own `in_flight` re-check see this operation. Acquiring it after the lock
    // is dropped would let a concurrent DELETE remove the row and the map entry first, leaving
    // this to run a multi-minute checkpoint against a session that no longer exists and to persist
    // a boundary event for it.
    let (entry, _in_flight) = {
        let map = state.sessions.read().await;
        match map.get(&id).cloned() {
            Some(entry) => {
                let guard = crate::server::state::InFlightGuard::acquire(&entry).map_err(|_| {
                    turn_in_flight_conflict(
                        id,
                        "session is busy; cancel the turn first via POST /v1/sessions/{id}/cancel \
                         before rewinding",
                    )
                })?;
                (entry, guard)
            }
            // Not resident: `ensure_session_loaded` below re-attaches it, and a session that is
            // not in the map cannot be racing a turn, so there is nothing to guard against yet.
            None => {
                drop(map);
                let entry = ensure_session_loaded(&state, id).await?;
                let guard = crate::server::state::InFlightGuard::acquire(&entry).map_err(|_| {
                    turn_in_flight_conflict(
                        id,
                        "session is busy; cancel the turn first via POST /v1/sessions/{id}/cancel \
                         before rewinding",
                    )
                })?;
                (entry, guard)
            }
        }
    };

    // `try_lock` for the same reason as `compact`; see the note there. Rewind is the sharper case:
    // blocking here would drop whichever turn won the race rather than the one the caller saw.
    let mut runtime = entry.runtime.try_lock().map_err(|_| {
        turn_in_flight_conflict(
            id,
            "session is busy; cancel the turn first via POST /v1/sessions/{id}/cancel before \
             rewinding",
        )
    })?;
    let messages_before = runtime.messages.len();
    let Some(event) = runtime.messages.rewind(body.turns) else {
        return Err(ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            format!(
                "nothing to rewind: session has fewer than {} turn(s)",
                body.turns
            ),
        )
        .with("session_id", id.to_string()));
    };
    let messages_after = runtime.messages.len();
    // The in-memory log is already rewound; persisting is what makes it survive eviction.
    if let Err(error) = state.shared.session_manager.save_event(id, &event).await {
        // Put the turns back rather than leave memory and disk disagreeing, exactly as the REPL's
        // `/rewind` does. Left diverged, `GET /messages` reads the DB and still shows the turns
        // with `revision` unmoved -- so the counter added to make a rewrite detectable reports
        // that nothing happened -- while the model no longer sees them, and a client retrying the
        // 500 eats another turn per attempt. `pop_repair` is the exact inverse of the
        // `replace_tail` that `rewind` just performed.
        runtime.messages.pop_repair();
        return Err(
            ProblemDetail::internal_sanitized("failed to persist rewind event", error)
                .with("session_id", id.to_string()),
        );
    }
    // The conversation was rewritten under the agent, which indexes two markers by message
    // position. Left stale, `last_accepted_len` makes the degrade-and-retry repair compute an
    // empty suspect window and silently stop firing for the rest of the session, and
    // `last_rendered_world` makes `run_turn` believe it already announced a tool or MCP server
    // whose announcement the rewind just deleted. `compact_session` clears both inline; this is
    // the other path that rewrites the log, and the REPL's `/rewind` has always called this.
    runtime.agent.reset_conversation_markers().await;
    drop(runtime);
    // See the note in `compact`. `save_event` has already moved `updated_at` on the row, so
    // without this the resident entry reports an older timestamp than `meka session list` does for
    // the same session, and only agrees again once the GC evicts it.
    entry.touch();

    tracing::info!(
        "rewound {} turn(s) from session {} via HTTP: {} -> {} messages",
        body.turns,
        id,
        messages_before,
        messages_after
    );

    Ok(Json(RewindResponse {
        session_id: id,
        turns_removed: body.turns,
        messages_before,
        messages_after,
    }))
}

#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct ExportQuery {
    /// `markdown` (default) or `json`. Markdown is a rendered transcript for a human; JSON is the
    /// round-trippable envelope `POST /v1/sessions/import` accepts.
    #[serde(default)]
    pub format: Option<String>,
}

/// `GET /v1/sessions/{id}/export`: the full conversation, including pre-compaction turns.
///
/// Reads the raw event log rather than the materialised view, so turns that compaction hid from
/// the model are still in the export. That is the whole point of having it: `GET /messages` shows
/// what the model can see, and this shows what actually happened.
#[utoipa::path(
    get,
    path = "/v1/sessions/{id}/export",
    tag = "conversation",
    params(
        ("id" = Uuid, Path, description = "Session UUID"),
        ExportQuery,
    ),
    responses(
        (status = 200, description = "Rendered transcript (text/markdown) or export envelope (application/json)"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
        (status = 422, description = "Unknown format", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn export(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    Path(id): Path<Uuid>,
    Query(query): Query<ExportQuery>,
) -> Result<Response, ProblemDetail> {
    scope::require(&principal, "sessions:r")?;
    // Read-only: never revive the runtime. Exporting a spawn tree would otherwise rebuild an agent
    // per sub-agent session just to read rows back out of SQLite.
    require_session_exists(&state, id).await?;

    let manager = &state.shared.session_manager;
    match query.format.as_deref().unwrap_or("markdown") {
        "markdown" | "md" => {
            let events = manager.load_events(id).await.map_err(|error| {
                ProblemDetail::internal_sanitized("failed to load session events", error)
                    .with("session_id", id.to_string())
            })?;
            let tool_outputs: std::collections::HashMap<String, String> = manager
                .load_all_tool_outputs(id)
                .await
                .map_err(|error| {
                    ProblemDetail::internal_sanitized("failed to load tool outputs", error)
                        .with("session_id", id.to_string())
                })?
                .into_iter()
                .collect();
            let body = crate::session::cli::format_session_as_markdown(id, &events, &tool_outputs);
            Ok((
                [(header::CONTENT_TYPE, "text/markdown; charset=utf-8")],
                body,
            )
                .into_response())
        }
        "json" => {
            let export = crate::session::cli::build_session_export(manager, id)
                .await
                .map_err(|error| {
                    ProblemDetail::internal_sanitized("failed to build session export", error)
                        .with("session_id", id.to_string())
                })?;
            Ok(Json(export).into_response())
        }
        other => Err(ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            format!(
                "unknown export format '{}'; expected 'markdown' or 'json'",
                other
            ),
        )),
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ImportResponse {
    /// Freshly minted id for the imported root. Import never reuses the exported ids, so the same
    /// envelope can be imported twice without collision.
    pub session_id: Uuid,
    /// Total sessions written, including sub-agent descendants.
    pub sessions_imported: usize,
}

/// Refuse an envelope large enough to hold the database against every other request.
fn store_too_large(count: usize) -> ProblemDetail {
    ProblemDetail::new(
        ErrorKind::InvalidBody,
        StatusCode::UNPROCESSABLE_ENTITY,
        format!(
            "session export contains {} sessions, more than the {} this server imports in one \
             request; import it with `meka session import`, which has no such limit",
            count,
            crate::session::cli::MAX_IMPORT_SESSIONS
        ),
    )
}

/// `POST /v1/sessions/import`: recreate a session tree from an export envelope.
#[utoipa::path(
    post,
    path = "/v1/sessions/import",
    tag = "conversation",
    request_body(content = String, description = "A `format=json` export envelope", content_type = "application/json"),
    responses(
        (status = 201, description = "Session tree imported", body = ImportResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Malformed or unsupported envelope", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn import(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    raw_body: Bytes,
) -> Result<(StatusCode, Json<ImportResponse>), ProblemDetail> {
    scope::require(&principal, "sessions:w")?;

    let export: crate::session::cli::SessionExport =
        serde_json::from_slice(&raw_body).map_err(|error| {
            ProblemDetail::new(
                ErrorKind::InvalidBody,
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("invalid session export JSON: {}", error),
            )
        })?;
    if export.sessions.len() > crate::session::cli::MAX_IMPORT_SESSIONS {
        return Err(store_too_large(export.sessions.len()));
    }
    // Version mismatch and an empty session list both surface here, as 422 rather than 500: the
    // envelope is the caller's, so a rejection is a statement about their input.
    let (records, root_new_id) = crate::session::cli::plan_import(export).map_err(|error| {
        ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            error.to_string(),
        )
    })?;
    let count = records.len();
    state
        .shared
        .session_manager
        .import_sessions(records)
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to import sessions", error))?;

    tracing::info!(
        "imported {} session(s) via HTTP as root {}",
        count,
        root_new_id
    );
    Ok((
        StatusCode::CREATED,
        Json(ImportResponse {
            session_id: root_new_id,
            sessions_imported: count,
        }),
    ))
}
