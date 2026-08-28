//! Scheduled jobs and background tasks: the two kinds of work that outlive the request that
//! started them.
//!
//! `meka serve` is already the durable host for both. It runs the scheduler over *every* job in
//! the database rather than only those belonging to an open conversation, because it can revive any
//! session on demand (`crate::server::schedule`), and it polls background-task outcomes on its own
//! timer. Until now it was also the only surface with no way to ask what it was about to run.
//!
//! Scheduled jobs are keyed by `schedule:r` / `schedule:w` rather than `sessions:*`: a job survives
//! the conversation that created it and fires unattended, so the ability to plant one is a
//! materially different grant from the ability to run a turn. Background tasks stay on `sessions:*`
//! because they live and die inside one session's runtime.

use axum::{
    Extension, Json,
    body::Bytes,
    extract::{Path, State},
    http::StatusCode,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

use crate::{
    background::TaskStatus,
    permission::Permission,
    schedule::{Gate, GatePredicate, GateProbe, Schedule, ScheduledJob},
    server::{
        auth::Principal,
        errors::{ErrorKind, ProblemDetail},
        reattach::require_session_exists,
        scope,
        state::ServerState,
    },
};

#[derive(Debug, Serialize, ToSchema)]
pub struct ScheduledJobView {
    pub id: String,
    pub session_id: Uuid,
    /// Human-readable rendering of the schedule, e.g. `every 30m`, `cron 0 9 * * 1-5`, or an
    /// RFC 3339 instant for a one-shot.
    pub schedule: String,
    pub prompt: String,
    /// Present when the job is gated, on a shell command or a tool call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gate: Option<GateView>,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_fired_at: Option<String>,
    pub next_fire_at: String,
    /// Why this job will not fire on its next occurrence, when something is holding it back.
    ///
    /// Absent means it will, as far as this server can establish. A held job and a healthy watcher
    /// with nothing to report are otherwise identical from outside: neither fires, and
    /// `last_fired_at` is absent for a brand-new job too. The agent gets the same sentence in its
    /// `[Scheduled]` block and `meka schedule list` gets a `Held` column; this is the third
    /// reader, and it is the one that can create a job on a session that cannot run it.
    ///
    /// Computed per request from the session's *current* level, not stored, so it tracks a `PATCH
    /// /v1/sessions/{id}` without the job being rewritten.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub withheld: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct GateView {
    /// What the gate runs: a shell command, or a tool name. `None` when the caller may not see it.
    ///
    /// Withheld from a token that does not also hold `sessions:r`. A gate command is an
    /// `execute_command` line that runs unattended, and the webhook path already omits the same
    /// field on the stated grounds that a command line is "the highest-entropy field in the system
    /// and the one most likely to carry a credential someone pasted into a `curl`". `GET
    /// /v1/schedule` is server-wide, so leaving it at `schedule:r` handed every gate on the box to
    /// a calendar bridge scoped to the read half of a scope invented so that schedule access would
    /// *not* imply session access. A tool gate is withheld on the same terms, though it discloses
    /// less either way: this field carries [`crate::schedule::GateProbe::summary`], which is the
    /// bare tool name. Its arguments, which are where a pasted credential would sit, reach only
    /// `schedule_list` and therefore only the agent that wrote them.
    ///
    /// The gate's kind and its condition stay visible either way, so a client can still tell a
    /// gated job from an ungated one, and a shell gate from a tool gate, without being told what
    /// it runs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub check: Option<String>,
    /// `shell` or `tool`.
    pub kind: String,
    /// The fire condition, as `changed`, `succeeded`, `matches /…/` or `/pointer not-empty`.
    pub when: String,
}

impl ScheduledJobView {
    /// Render a job for a caller, disclosing the gate command only when `reveal_command` is set.
    ///
    /// Not a `From` impl any more: the rendering depends on the caller's scopes, and a conversion
    /// that cannot see them is exactly how the command came to be disclosed at `schedule:r`.
    fn render(job: ScheduledJob, reveal_command: bool, withheld: Option<String>) -> Self {
        Self {
            withheld,
            id: job.id,
            session_id: job.session_id,
            schedule: job.schedule.describe(),
            prompt: job.prompt,
            gate: job.gate.map(|gate| GateView {
                check: reveal_command.then(|| gate.probe.summary()),
                kind: gate.probe.kind_str().to_string(),
                when: gate.predicate.summary(),
            }),
            created_at: job.created_at.to_rfc3339(),
            last_fired_at: job.last_fired_at.map(|at| at.to_rfc3339()),
            next_fire_at: job.next_fire_at.to_rfc3339(),
        }
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ScheduledJobsResponse {
    pub jobs: Vec<ScheduledJobView>,
}

/// `GET /v1/schedule`: every scheduled job in the database, across all sessions.
#[utoipa::path(
    get,
    path = "/v1/schedule",
    tag = "schedule",
    responses(
        (status = 200, description = "All scheduled jobs", body = ScheduledJobsResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn list_all(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
) -> Result<Json<ScheduledJobsResponse>, ProblemDetail> {
    scope::require(&principal, "schedule:r")?;
    let reveal_command = principal.has_scope("sessions:r");
    let jobs = state
        .shared
        .session_manager
        .schedule_store()
        .list_all_scheduled_jobs()
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to list scheduled jobs", error)
        })?;
    Ok(Json(ScheduledJobsResponse {
        jobs: render_batch(&state, jobs, reveal_command).await,
    }))
}

/// `GET /v1/sessions/{id}/schedule`: jobs belonging to one session.
#[utoipa::path(
    get,
    path = "/v1/sessions/{id}/schedule",
    tag = "schedule",
    params(("id" = Uuid, Path, description = "Session UUID")),
    responses(
        (status = 200, description = "Scheduled jobs for this session", body = ScheduledJobsResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn list_for_session(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    Path(id): Path<Uuid>,
) -> Result<Json<ScheduledJobsResponse>, ProblemDetail> {
    scope::require(&principal, "schedule:r")?;
    let reveal_command = principal.has_scope("sessions:r");
    require_session_exists(&state, id).await?;
    let jobs = state
        .shared
        .session_manager
        .schedule_store()
        .list_scheduled_jobs(id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to list scheduled jobs", error)
                .with("session_id", id.to_string())
        })?;
    Ok(Json(ScheduledJobsResponse {
        jobs: render_batch(&state, jobs, reveal_command).await,
    }))
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct CreateJobRequest {
    /// What the agent is asked to do when the job fires.
    pub prompt: String,
    /// One-shot time: an RFC 3339 instant, or a duration from now (`"20m"`, `"1h 30m"`).
    #[serde(default)]
    pub at: Option<String>,
    /// Recurring interval (`"30m"`, `"6h"`).
    #[serde(default)]
    pub every: Option<String>,
    /// 5-field cron pattern, evaluated in the host's local time.
    #[serde(default)]
    pub cron: Option<String>,
    /// Guard the job on a probe: a shell command, or a call to a read-only tool.
    ///
    /// The level required depends on which. A shell command runs unattended and with no sandbox,
    /// so it needs `unrestricted` and no level that promises a boundary can honestly authorise it.
    /// A tool probe needs only what the tool itself resolves to, which a gate requires to be
    /// `read`.
    #[serde(default)]
    pub gate: Option<CreateGate>,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct CreateGate {
    /// What to run: `{"command": "…"}` for a shell command, or `{"tool": "mcp__server__tool",
    /// "arguments": {…}}` for a tool call.
    pub check: serde_json::Value,
    /// When to fire: `"changed"` (the default), `"succeeded"`, `{"matches": "<regex>"}`, or
    /// `{"at": "<json pointer>", "is": "not-empty" | "empty" | "changed"}`.
    ///
    /// Untyped here because one parser answers for this field on every door
    /// ([`crate::schedule::GatePredicate::parse_request`]); a second, derived shape would drift
    /// from the tool schema and give different errors for the same mistake.
    #[serde(default)]
    pub when: Option<serde_json::Value>,
}

/// `POST /v1/sessions/{id}/schedule`: plant a job on a session.
#[utoipa::path(
    post,
    path = "/v1/sessions/{id}/schedule",
    tag = "schedule",
    params(("id" = Uuid, Path, description = "Session UUID")),
    request_body = CreateJobRequest,
    responses(
        (status = 201, description = "Job created", body = ScheduledJobView),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope, or a session that cannot do unattended work", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid schedule, or the session's job limit is reached", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn create(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    Path(id): Path<Uuid>,
    raw_body: Bytes,
) -> Result<(StatusCode, Json<ScheduledJobView>), ProblemDetail> {
    scope::require(&principal, "schedule:w")?;

    // Refused rather than accepted-and-ignored. With `[schedule] enabled = false` the scheduler
    // task is a no-op and the `schedule_*` tools are not registered at all, so this endpoint is the
    // only way a job can still be created -- and it would be created into a server that will never
    // run it, persisted, and listed forever with a `next_fire_at` receding into the past. Listing
    // and cancelling stay open, because clearing out jobs left over from before the flag was
    // flipped is exactly what an operator wants then.
    if !state.shared.config.schedule.enabled {
        return Err(ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            "scheduling is disabled on this server (`[schedule] enabled = false`), so a job \
             created here would never fire",
        )
        .with("session_id", id.to_string()));
    }

    let body: CreateJobRequest = serde_json::from_slice(&raw_body).map_err(|error| {
        ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            format!("invalid schedule request body: {}", error),
        )
    })?;
    if body.prompt.trim().is_empty() {
        return Err(ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            "`prompt` cannot be empty",
        ));
    }

    // Read, not revived. Reviving would take the session's cross-process file lock for up to
    // `idle_timeout` and hand a `schedule:w` token the same lock-pinning reach a read token was
    // just denied on `GET /context`. One read answers all three questions below -- that the session
    // exists, that it is one a job may belong to, and what level a gate would be authorised
    // against -- so the row is fetched once here rather than per check.
    let summary = state
        .shared
        .session_manager
        .session_info(id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to look up session", error)
                .with("session_id", id.to_string())
        })?
        .ok_or_else(|| {
            ProblemDetail::new(
                ErrorKind::SessionNotFound,
                StatusCode::NOT_FOUND,
                format!("session '{}' does not exist", id),
            )
            .with("session_id", id.to_string())
        })?;

    if let Some(refusal) = refuse_subagent_session(id, summary.parent_id) {
        return Err(refusal);
    }

    let resident = state.sessions.read().await.get(&id).cloned();

    let now = Utc::now();
    let schedule = parse_schedule(&body, now)?;

    // The session's level, for both checks below rather than only the gate's.
    //
    // The shared predicate rather than a comparison, exactly as `schedule_create` does:
    // `Permission`'s `Ord` is for display order, and `Ask` is precisely the level a gate must not
    // run at, because there is nobody to approve it.
    let permission = match &resident {
        Some(entry) => entry.permission.get(),
        None => permission_from_summary(&state, id, Some(&summary)),
    };

    // Refused for an *ungated* job too, which is the case a gate-shaped check misses.
    //
    // The fire door declines every job on a session at `none`, so accepting one here creates a row
    // that can never run. This endpoint is the only door that could: `schedule_create` requires
    // `read` to dispatch at all, so the agent cannot reach it, and a token's scopes say nothing
    // about the session's level. A client that means to raise the session later can do so and then
    // create the job; one that does not now finds out at the point it can act on it, rather than by
    // noticing months later that nothing ever fired.
    if !permission.allows_unattended_work() {
        return Err(ProblemDetail::new(
            ErrorKind::SessionPermission,
            StatusCode::FORBIDDEN,
            format!(
                "this session is at {}, where no tool is executable, so a scheduled turn could \
                 neither act on the job nor cancel it. Raise the session with `PATCH \
                 /v1/sessions/{{id}}` first.",
                permission
            ),
        )
        .with("session_id", id.to_string()));
    }

    let gate = match &body.gate {
        None => None,
        Some(requested) => {
            // `sessions:w` on top of `schedule:w`, because a gate is not really a scheduling
            // feature: the command runs on a timer, before the turn, as the server's user, and it
            // runs whether or not the turn works at all -- so a job whose gate is the payload
            // needs no provider, no credit and no model. `schedule:w` alone is meant to say "may
            // plant work on a session", and `GET /v1/schedule` already hands out every session id
            // in the database, so without this an operator who scoped a calendar bridge to
            // `schedule:*` and nothing else has in fact granted it unattended arbitrary shell.
            //
            // Requiring `sessions:w` puts gates in the tier that can already drive the agent, and
            // leaves the ordinary prompt-only job reachable by a schedule-only token.
            scope::require(&principal, "sessions:w").map_err(|_| {
                ProblemDetail::new(
                    ErrorKind::AuthScope,
                    StatusCode::FORBIDDEN,
                    "a `gate` runs a probe unattended, so it needs `sessions:w` as well as \
                     `schedule:w`. Create the job without `gate` and check the condition inside \
                     the prompt instead.",
                )
                .with("session_id", id.to_string())
            })?;
            let invalid = |message: String| {
                ProblemDetail::new(
                    ErrorKind::InvalidBody,
                    StatusCode::UNPROCESSABLE_ENTITY,
                    message,
                )
            };
            let probe = GateProbe::parse_request(Some(&requested.check)).map_err(invalid)?;
            let predicate =
                GatePredicate::parse_request(requested.when.as_ref()).map_err(invalid)?;

            // Authority is re-checked at fire time as well; this is the early, specific refusal so
            // a client learns at `POST` rather than from a job that silently never fires.
            if let Err(refusal) = crate::schedule::gate_probe_is_authorised(
                &probe,
                permission,
                state.shared.config.schedule.gate_tools.as_deref(),
            ) {
                // Routed by what would actually fix it, because that is what clients switch on.
                //
                // Never `AuthScope`: the token is fine and a better one will never help. Reporting
                // a scope failure sends a client off to re-provision a token it already holds.
                //
                // `SessionPermission` only where raising the session is the remedy the docs promise
                // for that type. A misspelled tool, or one that resolves above `read`, is a bad
                // request: no level and no token changes the answer, and `PATCH /v1/sessions/{id}`
                // is a wild goose chase. Those are `InvalidBody`, alongside the malformed-`check`
                // refusals the parser above already returns that way.
                let (kind, status) = match refusal {
                    crate::schedule::GateRefusal::ShellNeedsUnrestricted
                    | crate::schedule::GateRefusal::SessionBelowTool => {
                        (ErrorKind::SessionPermission, StatusCode::FORBIDDEN)
                    }
                    crate::schedule::GateRefusal::ToolUnavailable
                    | crate::schedule::GateRefusal::ToolNotReadOnly(_) => {
                        (ErrorKind::InvalidBody, StatusCode::UNPROCESSABLE_ENTITY)
                    }
                };
                // A server mid-handshake resolves nothing, and every tool it provides is
                // `ToolUnavailable` for the second that takes. The status stays 422 rather than
                // gaining a taxonomy entry for a sub-second window, but the message says so: a
                // client that retries once is right, and one that concludes the tool does not
                // exist is wrong.
                let transient = matches!(refusal, crate::schedule::GateRefusal::ToolUnavailable)
                    && state
                        .shared
                        .config
                        .schedule
                        .gate_tools
                        .as_ref()
                        .is_some_and(|tools| tools.is_still_connecting(&probe.summary()));
                let advice = if transient {
                    "Its MCP server has not finished connecting; retry shortly."
                } else {
                    "Create the job without `gate` and check the condition inside the prompt \
                     instead."
                };
                return Err(ProblemDetail::new(
                    kind,
                    status,
                    format!("{}. {}", refusal.explain(&probe, permission), advice),
                )
                .with("session_id", id.to_string()));
            }
            Some(Gate {
                probe,
                predicate,
                last_output: None,
                // See `schedule_create`: the level is recorded so `prepare` can re-check it at fire
                // time. The guard above has admitted this level for this probe, but the session's
                // is mutable through `PATCH /v1/sessions/{id}` and a tool's resolved level moves
                // with config, so the check above cannot stand in for one made when the probe
                // actually runs.
                permission,
            })
        }
    };

    let existing = state
        .shared
        .session_manager
        .schedule_store()
        .list_scheduled_jobs(id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to count scheduled jobs", error)
                .with("session_id", id.to_string())
        })?
        .len();
    let max_jobs = state.shared.config.schedule.max_jobs;
    if existing >= max_jobs {
        return Err(ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            format!(
                "session already has {} scheduled jobs (the configured limit). Cancel one first.",
                existing
            ),
        )
        .with("session_id", id.to_string()));
    }

    let next_fire_at = schedule.next_after(now).ok_or_else(|| {
        ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            "that schedule has no next occurrence; a one-shot time must be in the future",
        )
    })?;

    let job = ScheduledJob {
        id: Uuid::new_v4().to_string(),
        session_id: id,
        schedule,
        prompt: body.prompt,
        gate,
        created_at: now,
        last_fired_at: None,
        next_fire_at,
        attempts: 0,
    };
    state
        .shared
        .session_manager
        .schedule_store()
        .create_scheduled_job(&job)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to create scheduled job", error)
                .with("session_id", id.to_string())
        })?;

    tracing::info!(
        "created scheduled job {} on session {} via HTTP, next fire {}",
        job.id,
        id,
        job.next_fire_at.to_rfc3339()
    );
    // The caller just supplied this command, so echoing it back discloses nothing new. `withheld`
    // is `None` by construction: both refusals above have already run against this same level, so a
    // job that reached here can fire.
    Ok((
        StatusCode::CREATED,
        Json(ScheduledJobView::render(job, true, None)),
    ))
}

/// Render a batch of jobs, each paired with why it cannot fire.
///
/// One permission lookup per distinct *session* rather than per job: a listing of fifty jobs on one
/// session is the ordinary case, and the level is a property of the session. The resident copy wins
/// where there is one, matching the creation door two functions up.
///
/// A session whose level cannot be read does not silence the answer. `job_withheld` asks the
/// questions that need no level -- parking, above all -- before it asks for one, so a job that is
/// provably dead is still reported as such. Passing the level as an `Option` is what lets that
/// happen; resolving it to a default first, or skipping the call, reports a parked job as healthy
/// on the one surface a client would ask.
///
/// (The two paragraphs that used to open this comment described
/// `session_permission_from_row` below, and were left attached here when this function was
/// inserted above it. Rustdoc showed them as this function's summary and left that one
/// undocumented.)
async fn render_batch(
    state: &ServerState,
    jobs: Vec<ScheduledJob>,
    reveal_command: bool,
) -> Vec<ScheduledJobView> {
    // Snapshotted rather than held: the loop below awaits a database read, and keeping the sessions
    // lock across that would put a listing in the way of every attach and detach.
    let resident: std::collections::HashMap<Uuid, Permission> = {
        let sessions = state.sessions.read().await;
        sessions
            .iter()
            .map(|(id, entry)| (*id, entry.permission.get()))
            .collect()
    };
    let tools = state.shared.config.schedule.gate_tools.as_deref();
    let mut levels: std::collections::HashMap<Uuid, Option<Permission>> =
        std::collections::HashMap::new();
    let mut out = Vec::with_capacity(jobs.len());
    for job in jobs {
        let level = match levels.get(&job.session_id) {
            Some(level) => *level,
            None => {
                let level = match resident.get(&job.session_id) {
                    Some(level) => Some(*level),
                    None => session_permission_from_row(state, job.session_id)
                        .await
                        .ok(),
                };
                levels.insert(job.session_id, level);
                level
            }
        };
        let withheld = match crate::schedule::job_withheld(&job, level, tools) {
            crate::schedule::Withheld::Yes(reason) => Some(reason),
            crate::schedule::Withheld::No | crate::schedule::Withheld::Undetermined => None,
        };
        out.push(ScheduledJobView::render(
            job,
            reveal_command,
            withheld_for_scope(withheld, reveal_command),
        ));
    }
    out
}

/// The withheld reason as far as this reader's scope allows.
///
/// The reason is not a neutral sentence. `GateRefusal::explain` embeds `probe.summary()`, which for
/// a tool gate *is* the tool name that `check` beside it is nulled to hide; the level refusals name
/// the session's permission, which is otherwise behind `sessions:r`; and a standing probe failure
/// carries the first line of the check's own output. Attaching it unconditionally therefore handed
/// a `schedule:r` token everything the field next to it withholds, and one MCP server failing to
/// connect turned a server-wide endpoint into a listing of every internal tool name on the box.
///
/// The bare fact survives at that scope, because it is why a client polls this at all, and it
/// discloses nothing `kind` does not already.
fn withheld_for_scope(reason: Option<String>, reveal_command: bool) -> Option<String> {
    match reveal_command {
        true => reason,
        false => reason
            .map(|_| "this job cannot currently fire; the reason needs `sessions:r`".to_string()),
    }
}

/// Refuse a job on a session that is somebody's sub-agent, returning the problem when it is one.
///
/// A job belongs to a top-level session, and this endpoint is the only door that could plant one
/// elsewhere. A sub-agent has no `schedule_*` tools by construction
/// ([`crate::tools::ToolRegistry::build_for_subagent`] passes no schedule config), so until this
/// the rule was held by omission at the tool door and by nothing at all here.
///
/// What it prevents is an authority escalation rather than an oddity. A worker's restrictions live
/// in `sessions.subagent_spec_json` and are applied by `build_for_subagent`, but the fire path
/// rebuilds a session through [`crate::build_session_agent`], which never reads that column. A job
/// keyed to a worker would therefore wake it with the full built-in set, none of its `[subagents]`
/// denials, and none of its memory or instruction grants. Its row also records no `permission`, so
/// the turn would run at the *host's* level rather than the lower one it was spawned under.
///
/// **This closes the scheduling route to that, not the weakness itself.** Every caller of
/// `reattach::ensure_session_loaded` rebuilds a worker the same unrestricted way, and `POST
/// /v1/sessions/{id}/turn` does it on demand for anyone holding `sessions:w`. Closing that needs a
/// re-attach path that reads the spec and routes to `build_for_subagent`, which is a larger change
/// than a refusal. What this door earns meanwhile is that the cheaper `schedule:w` grant cannot
/// reach it at all, and that nobody can leave a *standing* job that reaches it on a timer.
///
/// `InvalidBody` rather than `AuthScope` or `SessionPermission`: no token and no permission level
/// changes the answer, so routing it at either would send a client off to re-provision a token or
/// to `PATCH /v1/sessions/{id}` for a refusal neither can lift. [`create`] routes its gate refusals
/// by the same rule.
fn refuse_subagent_session(id: Uuid, parent: Option<Uuid>) -> Option<ProblemDetail> {
    let parent = parent?;
    Some(
        ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            format!(
                "session '{}' is a sub-agent of '{}', and a sub-agent runs only while the agent \
                 that spawned it is waiting on it. Schedule the job on '{}' instead and let its \
                 turn dispatch the worker.",
                id, parent, parent
            ),
        )
        .with("session_id", id.to_string()),
    )
}

/// The level a non-resident session's row asks for, narrowed to what this installation permits.
///
/// Split from the fetch so a caller that already holds the row does not read it again. A row that
/// records nothing (REPL, ACP and sub-agent rows all leave `permission` NULL) falls back to the
/// host's configured level, which is also what an unparseable value resolves to.
fn permission_from_summary(
    state: &ServerState,
    id: Uuid,
    summary: Option<&crate::session::SessionSummary>,
) -> Permission {
    let persisted = crate::permission::parse_recorded_permission(
        summary.and_then(|summary| summary.permission.as_deref()),
        &format_args!("session {id}"),
    )
    .unwrap_or(state.shared.config.permission);
    if state
        .shared
        .config
        .enabled_permissions
        .is_enabled(persisted)
    {
        persisted
    } else {
        state.shared.config.permission
    }
}

async fn session_permission_from_row(
    state: &ServerState,
    id: Uuid,
) -> Result<Permission, ProblemDetail> {
    let summary = state
        .shared
        .session_manager
        .session_info(id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to read session permission", error)
                .with("session_id", id.to_string())
        })?;
    Ok(permission_from_summary(state, id, summary.as_ref()))
}

/// Resolve exactly one of `at` / `every` / `cron`.
///
/// Ambiguity is refused rather than resolved by precedence, matching `schedule_create`: silently
/// honouring one and dropping the other would produce a job firing on a schedule nobody asked for.
fn parse_schedule(
    body: &CreateJobRequest,
    now: chrono::DateTime<Utc>,
) -> Result<Schedule, ProblemDetail> {
    let given: Vec<(&str, &str)> = [
        ("at", body.at.as_deref()),
        ("every", body.every.as_deref()),
        ("cron", body.cron.as_deref()),
    ]
    .into_iter()
    .filter_map(|(key, value)| {
        value
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| (key, value))
    })
    .collect();

    let invalid = |message: String| {
        ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            message,
        )
    };
    match given.as_slice() {
        [("at", value)] => Schedule::parse_at(value, now).map_err(invalid),
        [("every", value)] => Schedule::parse_every(value).map_err(invalid),
        [("cron", value)] => Schedule::parse_cron(value).map_err(invalid),
        [] => Err(invalid(
            "give one of `at` (once), `every` (interval), or `cron` (expression)".to_string(),
        )),
        several => Err(invalid(format!(
            "give exactly one schedule, got {}",
            several
                .iter()
                .map(|(key, _)| *key)
                .collect::<Vec<_>>()
                .join(" and ")
        ))),
    }
}

/// `DELETE /v1/schedule/{job_id}`: cancel a job.
///
/// Keyed on the job id alone rather than nested under its session, because that is how a client
/// that read `GET /v1/schedule` holds it.
///
/// Takes a unique id prefix as well as the full id, and 404s when nothing matches. Both halves
/// matter, and for the same reason: the 8-character short form is what every surface that renders a
/// job to a human shows (`meka schedule list`, the REPL's `/schedule`, the `schedule_list` tool),
/// so an operator will paste one here, and answering 204 to an id that matched nothing would report
/// a still-firing job as cancelled. A gated job kept alive that way goes on running a shell command
/// unattended. `schedule_cancel` and `meka schedule cancel` already resolve prefixes and already
/// report a miss; this is the surface that did not.
#[utoipa::path(
    delete,
    path = "/v1/schedule/{job_id}",
    tag = "schedule",
    params(("job_id" = String, Path, description = "Scheduled job id, or a unique prefix of one")),
    responses(
        (status = 204, description = "Job cancelled"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No job matches that id", body = ProblemDetail),
        (status = 422, description = "The prefix matches more than one job", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn cancel(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    Path(job_id): Path<String>,
) -> Result<StatusCode, ProblemDetail> {
    scope::require(&principal, "schedule:w")?;
    let jobs = state
        .shared
        .session_manager
        .schedule_store()
        .list_all_scheduled_jobs()
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to resolve scheduled job", error)
                .with("job_id", job_id.clone())
        })?;
    let matches: Vec<&ScheduledJob> = jobs
        .iter()
        .filter(|job| job.id.starts_with(&job_id))
        .collect();
    let resolved = match matches.as_slice() {
        [job] => job.id.clone(),
        [] => {
            return Err(ProblemDetail::new(
                ErrorKind::NotFound,
                StatusCode::NOT_FOUND,
                format!("no scheduled job matches '{}'", job_id),
            )
            .with("job_id", job_id.clone()));
        }
        several => {
            return Err(ProblemDetail::new(
                ErrorKind::InvalidBody,
                StatusCode::UNPROCESSABLE_ENTITY,
                format!(
                    "'{}' matches {} scheduled jobs; use a longer id",
                    job_id,
                    several.len()
                ),
            )
            .with("job_id", job_id.clone()));
        }
    };

    let removed = state
        .shared
        .session_manager
        .schedule_store()
        .delete_scheduled_job(&resolved)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to cancel scheduled job", error)
                .with("job_id", resolved.clone())
        })?;
    // The listing above and the delete are two statements, and a scheduler sweep can retire the row
    // in between. `204` then reported a cancellation this request did not perform, which is exactly
    // what a client polls this endpoint to establish. `404` is the same answer it would have got a
    // moment earlier, and is true.
    if !removed {
        return Err(ProblemDetail::new(
            ErrorKind::NotFound,
            StatusCode::NOT_FOUND,
            format!("scheduled job '{}' was already gone", resolved),
        )
        .with("job_id", resolved));
    }
    tracing::info!("cancelled scheduled job {} via HTTP", resolved);
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Debug, Serialize, ToSchema)]
pub struct BackgroundTaskView {
    pub id: String,
    pub session_id: Uuid,
    /// The tool that was backgrounded, e.g. `execute_command`.
    pub tool_name: String,
    /// Human-readable summary of what was started.
    pub label: String,
    /// `running`, `completed`, `failed`, `cancelled`, or `interrupted`.
    pub status: String,
    /// The tool's output for a terminal task, truncated when it was also spilled to the
    /// scratchpad.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
    /// Scratchpad entry holding the full output, when it was too large to carry inline.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scratchpad_name: Option<String>,
    pub started_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<String>,
    /// When the outcome was handed to the agent. `null` means it is still waiting to be delivered
    /// on the session's next turn.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delivered_at: Option<String>,
}

impl From<crate::background::BackgroundTask> for BackgroundTaskView {
    fn from(task: crate::background::BackgroundTask) -> Self {
        Self {
            id: task.id,
            session_id: task.session_id,
            tool_name: task.tool_name,
            label: task.label,
            status: task.status.as_str().to_string(),
            outcome: task.outcome,
            scratchpad_name: task.scratchpad_name,
            started_at: task.started_at.to_rfc3339(),
            finished_at: task.finished_at.map(|at| at.to_rfc3339()),
            delivered_at: task.delivered_at.map(|at| at.to_rfc3339()),
        }
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub struct BackgroundTasksResponse {
    pub tasks: Vec<BackgroundTaskView>,
}

/// `GET /v1/sessions/{id}/tasks`: this session's background tasks, newest first.
#[utoipa::path(
    get,
    path = "/v1/sessions/{id}/tasks",
    tag = "tasks",
    params(("id" = Uuid, Path, description = "Session UUID")),
    responses(
        (status = 200, description = "Background tasks", body = BackgroundTasksResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn list_tasks(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    Path(id): Path<Uuid>,
) -> Result<Json<BackgroundTasksResponse>, ProblemDetail> {
    scope::require(&principal, "sessions:r")?;
    require_session_exists(&state, id).await?;
    let tasks = state
        .shared
        .session_manager
        .background_store()
        .list_background_tasks(id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to list background tasks", error)
                .with("session_id", id.to_string())
        })?;
    Ok(Json(BackgroundTasksResponse {
        tasks: tasks.into_iter().map(BackgroundTaskView::from).collect(),
    }))
}

/// `DELETE /v1/sessions/{id}/tasks/{task_id}`: stop a running background task.
#[utoipa::path(
    delete,
    path = "/v1/sessions/{id}/tasks/{task_id}",
    tag = "tasks",
    params(
        ("id" = Uuid, Path, description = "Session UUID"),
        ("task_id" = String, Path, description = "Background task id"),
    ),
    responses(
        (status = 204, description = "Task cancelled, or already terminal"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session or task not found", body = ProblemDetail),
        (status = 422, description = "The prefix matches more than one task", body = ProblemDetail),
    ),
    security(("bearerAuth" = []))
)]
pub async fn cancel_task(
    State(state): State<ServerState>,
    Extension(principal): Extension<Principal>,
    Path((id, task_id)): Path<(Uuid, String)>,
) -> Result<StatusCode, ProblemDetail> {
    scope::require(&principal, "sessions:w")?;

    let Some(task) = state
        .shared
        .session_manager
        .background_store()
        .resolve_background_task(id, &task_id)
        .await
        .map_err(|error| match &error {
            // `resolve_background_task` accepts an id prefix, and reports an ambiguous one as
            // `Config`. That is a statement about the caller's input, so it is a 422; routing it
            // through `internal_sanitized` would report "use a longer id" as a server fault and
            // hide the one detail that fixes it.
            crate::error::MekaError::Config(message) => ProblemDetail::new(
                ErrorKind::InvalidBody,
                StatusCode::UNPROCESSABLE_ENTITY,
                message.clone(),
            )
            .with("session_id", id.to_string()),
            _ => ProblemDetail::internal_sanitized("failed to resolve background task", error)
                .with("session_id", id.to_string()),
        })?
    else {
        return Err(ProblemDetail::new(
            ErrorKind::NotFound,
            StatusCode::NOT_FOUND,
            format!("no background task in session {} matches '{}'", id, task_id),
        )
        .with("session_id", id.to_string())
        .with("task_id", task_id.clone()));
    };
    if task.status.is_terminal() {
        return Ok(StatusCode::NO_CONTENT);
    }

    // Recorded before signalling, exactly as `task_cancel` does: `finish_background_task` only
    // writes over a `running` row, so a task finishing in the same instant cannot report success
    // after the caller was told it was stopped.
    state
        .shared
        .session_manager
        .background_store()
        .finish_background_task(&task.id, TaskStatus::Cancelled, None, None)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to record task cancellation", error)
                .with("task_id", task.id.clone())
        })?;

    // Read off the entry, not through `runtime.agent`: the registry is hoisted onto `SessionEntry`
    // precisely so this path does not wait on the mutex an in-flight turn holds, which is the
    // state a session is in whenever anyone actually wants to stop a detached task.
    //
    // A task belonging to an evicted session has a row here but no live handle. Recording the
    // cancellation is still right, and is what stops a resumed session waiting on an outcome that
    // will never arrive.
    let entry = state.sessions.read().await.get(&id).cloned();
    let signalled = match entry {
        Some(entry) => entry.background_tasks.cancel(&task.id).await,
        None => false,
    };
    if !signalled {
        // `warn`, not `debug`: this is the one outcome that differs from what the 204 claims. The
        // row now says `cancelled` and `GET /tasks` will agree, but nothing in this process could
        // reach the task, so if it is still running it will run to completion unnoticed -- and a
        // second cancel short-circuits on the now-terminal status. Worth seeing by default.
        tracing::warn!(
            "background task {} had no live handle in this process; recorded as cancelled, but \
             anything still running was not signalled",
            task.id
        );
    }
    tracing::info!("cancelled background task {} via HTTP", task.id);
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A job may be planted on a top-level session and on nothing else.
    ///
    /// The refusal is what stands between a `schedule:w` token and an authority escalation:
    /// `GET /v1/sessions?include_children=true` hands out sub-agent ids to any `sessions:r`
    /// holder, and the fire path rebuilds a session through `build_session_agent`, which knows
    /// nothing of `subagent_spec_json`. Without this the worker wakes unrestricted, at the host's
    /// permission rather than its own.
    ///
    /// Both arms are asserted. A guard that refused everything would pass a
    /// refusal-only test while breaking every ordinary job.
    #[test]
    fn a_job_is_refused_on_a_sub_agent_session_and_admitted_on_a_top_level_one() {
        let worker = Uuid::new_v4();
        let parent = Uuid::new_v4();

        assert!(
            refuse_subagent_session(worker, None).is_none(),
            "a session with no parent is the ordinary case and must be admitted"
        );

        let refusal = refuse_subagent_session(worker, Some(parent))
            .expect("a session with a parent cannot own a job");
        assert_eq!(refusal.status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(refusal.type_uri, ErrorKind::InvalidBody.type_uri());
        // Both ids, because the remedy is to re-issue the request against the parent and a client
        // that was handed the worker's id may not know what spawned it.
        let detail = refusal.detail.as_deref().unwrap_or_default();
        assert!(
            detail.contains(&worker.to_string()) && detail.contains(&parent.to_string()),
            "the refusal must name the worker and the session to use instead: {detail}"
        );
    }

    /// Nothing about *why* a job is held escapes to a token that may not see what the gate runs.
    ///
    /// Every refusal sentence carries something the scope withholds elsewhere: a tool name, a
    /// session's permission level, or a line of the check's own output. Asserting on the exact
    /// wording would be brittle, so this asserts the property -- none of the reason survives --
    /// against the three shapes the reasons actually take.
    #[test]
    fn a_low_scope_reader_learns_that_a_job_is_held_and_nothing_else() {
        let reasons = [
            "no gate tool named `mcp__internal_bridge__unseen`. A gate can call a read-only tool",
            "a gate command runs unattended with no sandbox, so it needs `unrestricted` \
             (currently workspace)",
            "its gate keeps failing and cannot say whether to fire: gate points at `/x` but the \
             probe did not return JSON: sk-live-DO-NOT-DISCLOSE",
        ];
        for reason in reasons {
            let redacted = withheld_for_scope(Some(reason.to_string()), false)
                .expect("the fact that it is held still reaches the client");
            for leaked in [
                "mcp__internal_bridge__unseen",
                "workspace",
                "sk-live-DO-NOT-DISCLOSE",
                "unrestricted",
            ] {
                assert!(
                    !redacted.contains(leaked),
                    "{leaked:?} reached a reader that cannot see `check`: {redacted:?}"
                );
            }
            assert_eq!(
                withheld_for_scope(Some(reason.to_string()), true).as_deref(),
                Some(reason),
                "and a reader that can see `check` still gets the whole answer"
            );
        }
        assert_eq!(
            withheld_for_scope(None, false),
            None,
            "a job that can fire is not reported as held to anyone"
        );
    }
}
