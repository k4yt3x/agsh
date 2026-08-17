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
    schedule::{Gate, GateFire, Schedule, ScheduledJob},
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
    /// Present when the job is gated on a shell command.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gate: Option<GateView>,
    /// Whether the job runs in a fresh sub-agent session rather than the conversation that created
    /// it.
    pub isolated: bool,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_fired_at: Option<String>,
    pub next_fire_at: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct GateView {
    pub command: String,
    /// `on-change` or `on-success`.
    pub fire: String,
}

impl From<ScheduledJob> for ScheduledJobView {
    fn from(job: ScheduledJob) -> Self {
        Self {
            id: job.id,
            session_id: job.session_id,
            schedule: job.schedule.describe(),
            prompt: job.prompt,
            gate: job.gate.map(|gate| GateView {
                command: gate.command,
                fire: gate.fire.as_str().to_string(),
            }),
            isolated: job.isolated,
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
        jobs: jobs.into_iter().map(ScheduledJobView::from).collect(),
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
        jobs: jobs.into_iter().map(ScheduledJobView::from).collect(),
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
    /// Guard the job on a shell command. Requires the session to be at `write` permission, for the
    /// same reason `schedule_create` does: the command runs unattended and unsandboxed.
    #[serde(default)]
    pub gate: Option<CreateGate>,
    /// Run in a fresh sub-agent session rather than replaying the parent's history every fire.
    #[serde(default)]
    pub isolated: bool,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct CreateGate {
    pub command: String,
    /// `on-change` (fire when stdout differs from last run) or `on-success` (fire while the
    /// command exits 0). Defaults to `on-change`.
    #[serde(default)]
    pub fire: Option<String>,
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
        (status = 403, description = "Insufficient scope, or a gate below write permission", body = ProblemDetail),
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

    // Existence-checked, not revived. Reviving would take the session's cross-process file lock
    // for up to `idle_timeout` and hand a `schedule:w` token the same lock-pinning reach a read
    // token was just denied on `GET /context`. A gate is authorised against the session's
    // permission, which is on the row as well as the entry, so the resident copy is used when
    // there is one and the row answers otherwise.
    require_session_exists(&state, id).await?;
    let resident = state.sessions.read().await.get(&id).cloned();

    let now = Utc::now();
    let schedule = parse_schedule(&body, now)?;
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
                    "a `gate` runs a shell command unattended, so it needs `sessions:w` as well as \
                     `schedule:w`. Create the job without `gate` and check the condition inside \
                     the prompt instead.",
                )
                .with("session_id", id.to_string())
            })?;
            // Matched explicitly rather than compared, exactly as `schedule_create` does:
            // `Permission`'s `Ord` is for clamping a sub-agent to its parent, and `Ask` is
            // precisely the level a gate must not run at, because there is nobody to approve it.
            let permission = match &resident {
                Some(entry) => entry.permission.get(),
                None => session_permission_from_row(&state, id).await?,
            };
            if !matches!(permission, Permission::Write) {
                // `SessionPermission`, not `AuthScope`: the token is fine and a better one will
                // never help. The docs tell clients to route on `type`, so reporting this as a
                // scope failure sends them to re-provision a token when the fix is to raise the
                // session's permission.
                return Err(ProblemDetail::new(
                    ErrorKind::SessionPermission,
                    StatusCode::FORBIDDEN,
                    format!(
                        "a gate runs a shell command unattended, so it needs the session to be at \
                         write permission (currently {}). Create the job without `gate` and check \
                         the condition inside the prompt instead.",
                        permission
                    ),
                )
                .with("session_id", id.to_string()));
            }
            if requested.command.trim().is_empty() {
                return Err(ProblemDetail::new(
                    ErrorKind::InvalidBody,
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "`gate.command` cannot be empty",
                ));
            }
            let fire = match requested.fire.as_deref().unwrap_or("on-change") {
                "on-change" => GateFire::OnChange,
                "on-success" => GateFire::OnSuccess,
                other => {
                    return Err(ProblemDetail::new(
                        ErrorKind::InvalidBody,
                        StatusCode::UNPROCESSABLE_ENTITY,
                        format!(
                            "unknown gate.fire '{}'; expected 'on-change' or 'on-success'",
                            other
                        ),
                    ));
                }
            };
            Some(Gate {
                command: requested.command.clone(),
                fire,
                last_output: None,
                // See `schedule_create`: the level is recorded so `prepare` can re-check it at fire
                // time. `permission` is `Write` by the guard above, and the session's level is
                // mutable through `PATCH /v1/sessions/{id}`, so the check above cannot stand in for
                // one made when the command actually runs.
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
        isolated: body.isolated,
        created_at: now,
        last_fired_at: None,
        next_fire_at,
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
    Ok((StatusCode::CREATED, Json(ScheduledJobView::from(job))))
}

/// The permission an evicted session would come back at, read from its row.
///
/// Mirrors `reattach::ensure_session_loaded`'s resolution so a gate authorised here is authorised
/// against the level the job will actually fire at: a persisted level that is no longer in the
/// enabled set falls back to the process default, exactly as re-attach would.
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
    let persisted = summary
        .and_then(|summary| summary.permission)
        .and_then(|value| value.parse::<Permission>().ok())
        .unwrap_or(state.shared.config.permission);
    Ok(
        if state
            .shared
            .config
            .enabled_permissions
            .is_enabled(persisted)
        {
            persisted
        } else {
            state.shared.config.permission
        },
    )
}

/// Resolve exactly one of `at` / `every` / `cron`.
///
/// Ambiguity is refused rather than resolved by precedence, matching `schedule_create`: silently
/// honouring one and dropping the other would produce a job firing on a schedule nobody asked for.
#[allow(clippy::result_large_err)]
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

    state
        .shared
        .session_manager
        .schedule_store()
        .delete_scheduled_job(&resolved)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to cancel scheduled job", error)
                .with("job_id", resolved.clone())
        })?;
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
