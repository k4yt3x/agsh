//! The `schedule_*` tools: the agent arranging its own future turns ([`crate::schedule`]).
//!
//! Creating and listing gate at [`Permission::Read`], matching `memory_*` and `scratchpad_*`: a
//! scheduled prompt writes to a store meka owns, and the turn it eventually produces is permission
//! -checked when it runs, so scheduling one grants nothing the session did not already have.
//!
//! The `gate` field is the exception, and is checked inside [`ScheduleCreateTool::execute`]
//! against [`crate::schedule::gate_probe_is_authorised`]. A gate runs unattended, on a timer, until
//! someone cancels it -- persistent in a way a tool call inside a turn is not, since that at least
//! ends with the turn that made it. The bar depends on what the gate runs: a shell command needs
//! `unrestricted` because it is unsandboxed, while a read-only tool call needs only what that tool
//! needs. [`Tool::required_permission`] is per-tool and cannot vary by argument, so the check has
//! to live in the body either way.
//!
//! Sub-agents deliberately get none of these tools. A sub-agent's session is ephemeral, so a job
//! keyed to it would outlive the only conversation that could run it.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{Local, Utc};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::{
    Tool, ToolOutput,
    util::{require_str, resolve_session_id},
};
use crate::{
    config::ResolvedScheduleConfig,
    error::{MekaError, Result},
    permission::Permission,
    provider::ToolDefinition,
    schedule::{Gate, GatePredicate, GateProbe, Schedule, ScheduledJob},
    session::SessionManager,
};

/// Shared by all three tools.
pub(super) struct ScheduleContext {
    session_manager: SessionManager,
    session_id: Arc<RwLock<Option<Uuid>>>,
}

/// Absolute local time plus a relative offset, e.g.
/// `Wed 2026-08-12 08:57 CEST (in 17h 35m)`.
///
/// Both halves earn their place: the absolute form is what the user can check against a calendar,
/// and the relative form is what catches a schedule that parsed successfully but means something
/// other than intended. A model that writes `0 9 * * 1-5` believing it fires "in a few minutes"
/// sees `in 17h 35m` and can correct itself before the user ever finds out.
fn describe_fire_time(at: chrono::DateTime<Utc>) -> String {
    let local = at.with_timezone(&Local);
    let delta = at - Utc::now();
    let relative = match delta.to_std() {
        Ok(std) => format!(
            "in {}",
            humantime_serde::re::humantime::format_duration(std::time::Duration::from_secs(
                std.as_secs()
            ))
        ),
        // Negative: the instant has already passed, which the scheduler will treat as due.
        Err(_) => "overdue".to_string(),
    };
    format!("{} ({})", local.format("%a %Y-%m-%d %H:%M %Z"), relative)
}

pub(super) struct ScheduleCreateTool {
    pub session_manager: SessionManager,
    pub session_id: Arc<RwLock<Option<Uuid>>>,
    pub config: ResolvedScheduleConfig,
    pub shared_permission: crate::permission::SharedPermission,
}

#[async_trait]
impl Tool for ScheduleCreateTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "schedule_create".to_string(),
            description: "Arrange for a prompt to be delivered to you at a future time, so you can \
                act without the user asking again. Use for reminders (\"remind me in 20 minutes\"), \
                recurring work (\"summarise my calendar every weekday morning\"), and watching \
                something change. Give exactly one of `at`, `every`, or `cron`. The prompt is \
                delivered as a turn with no human present, so write it as an instruction to \
                yourself, including any context you will need and no longer have."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "What to do when the job fires. Self-contained: the \
                                        conversation that created it may be long gone."
                    },
                    "at": {
                        "type": "string",
                        "description": "One-shot. Either a duration from now ('20m', '2h', \
                                        '1h 30m') or an RFC 3339 timestamp. Note 'm' is minutes \
                                        and 'M' is months. Fires once, then deletes itself."
                    },
                    "every": {
                        "type": "string",
                        "description": "Recurring fixed interval ('30m', '1h', '1d'). Runs until \
                                        cancelled."
                    },
                    "cron": {
                        "type": "string",
                        "description": "Recurring 5-field cron expression in local time \
                                        ('0 9 * * 1-5' = 09:00 on weekdays). No seconds field."
                    },
                    "gate": {
                        "type": "object",
                        "description": "Optional. Check something first and only take a turn if it \
                                        says something happened. Turns an expensive poll into a \
                                        cheap one, so a short `every` becomes affordable.",
                        "properties": {
                            "check": {
                                "type": "object",
                                "description": "What to run. Either {\"command\": \"...\"} for a \
                                                shell command, which needs `unrestricted` because \
                                                it runs unsandboxed, or {\"tool\": \"name\", \
                                                \"arguments\": {...}} to call a read-only tool, \
                                                which needs only what that tool needs. Prefer the \
                                                tool form when one exists: it is available at lower \
                                                permission and returns structured data you can \
                                                point `when.at` into.",
                                "properties": {
                                    "command": {"type": "string"},
                                    "tool": {"type": "string"},
                                    "arguments": {"type": "object"}
                                }
                            },
                            "when": {
                                "description": "What counts as 'something happened'. \"changed\" \
                                                (default) fires when the whole result differs from \
                                                last time. \"succeeded\" fires while the command \
                                                exits 0 or the tool call does not error. \
                                                {\"matches\": \"regex\"} fires when the result \
                                                matches. {\"at\": \"/json/pointer\", \"is\": \
                                                \"not-empty\"|\"empty\"|\"changed\"} judges one \
                                                field. Use `at` for anything returning JSON: a \
                                                result carrying a timestamp or an id differs on \
                                                every call, so \"changed\" over the whole of it \
                                                fires every tick and costs the turns the gate is \
                                                meant to save."
                            }
                        },
                        "required": ["check"]
                    }
                },
                "required": ["prompt"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let tool_name = "schedule_create";
        let session_id = resolve_session_id(&self.session_id, tool_name).await?;
        let prompt = require_str(&input, "prompt", tool_name)?;

        let now = Utc::now();
        let schedule = parse_schedule(&input, now, tool_name)?;
        let gate = self.parse_gate(&input, tool_name)?;

        let existing = self
            .session_manager
            .schedule_store()
            .list_scheduled_jobs(session_id)
            .await?
            .len();
        if existing >= self.config.max_jobs {
            return Err(MekaError::ToolExecution {
                tool_name: tool_name.to_string(),
                message: format!(
                    "this session already has {} scheduled jobs (the limit). Cancel one with \
                     schedule_cancel first.",
                    existing
                ),
            });
        }

        let next_fire_at = schedule
            .next_after(now)
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: tool_name.to_string(),
                message: "that schedule has no next occurrence. A one-shot time must be in the \
                          future."
                    .to_string(),
            })?;

        let job = ScheduledJob {
            id: Uuid::new_v4().to_string(),
            session_id,
            schedule,
            prompt: prompt.to_string(),
            gate,
            created_at: now,
            last_fired_at: None,
            next_fire_at,
            attempts: 0,
        };
        self.session_manager
            .schedule_store()
            .create_scheduled_job(&job)
            .await?;

        tracing::info!(
            "scheduled job {} ({}), next fire {}",
            job.short_id(),
            job.schedule.describe(),
            next_fire_at.to_rfc3339()
        );

        let mut summary = format!(
            "Created job {} ({}). Next fire: {}.",
            job.short_id(),
            job.schedule.describe(),
            describe_fire_time(next_fire_at)
        );
        if job.gate.is_some() {
            summary.push_str(" Gated: a turn happens only when the gate says so.");
        }
        summary.push_str(&format!(
            " Cancel with schedule_cancel(\"{}\").",
            job.short_id()
        ));
        Ok(ToolOutput::text(summary, false))
    }
}

impl ScheduleCreateTool {
    fn parse_gate(&self, input: &serde_json::Value, tool_name: &str) -> Result<Option<Gate>> {
        let Some(raw) = input.get("gate").filter(|value| !value.is_null()) else {
            return Ok(None);
        };
        let refuse = |message: String| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message,
        };
        let probe = GateProbe::parse_request(raw.get("check")).map_err(refuse)?;
        let predicate = GatePredicate::parse_request(raw.get("when")).map_err(refuse)?;

        // A gate outlives the turn that created it and runs with no supervision, which is a
        // stronger grant than a tool call inside a turn. Checked here rather than through
        // `required_permission` so an ungated reminder still works at read, and because the bar
        // depends on the probe: a shell command needs `unrestricted`, a read-only tool call needs
        // only what that tool needs.
        let permission = self.shared_permission.get();
        if let Err(refusal) = crate::schedule::gate_probe_is_authorised(
            &probe,
            permission,
            self.config.gate_tools.as_deref(),
        ) {
            return Err(refuse(format!(
                "{}. Create the job without `gate` and check the condition inside the prompt \
                 instead.",
                refusal.explain(&probe, permission)
            )));
        }

        Ok(Some(Gate {
            probe,
            predicate,
            last_output: None,
            // Recorded, not re-derived. The check above proves the level *now*; the row will be
            // executed by some other process on some later day, and `prepare` re-checks both this
            // and the live level before running anything.
            permission,
        }))
    }
}

/// Pull exactly one of `at` / `every` / `cron` out of the tool input.
fn parse_schedule(
    input: &serde_json::Value,
    now: chrono::DateTime<Utc>,
    tool_name: &str,
) -> Result<Schedule> {
    let field = |key: &str| {
        input
            .get(key)
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
    };
    let given: Vec<(&str, &str)> = ["at", "every", "cron"]
        .into_iter()
        .filter_map(|key| field(key).map(|value| (key, value)))
        .collect();

    let (kind, value) = match given.as_slice() {
        [single] => *single,
        [] => {
            return Err(MekaError::ToolExecution {
                tool_name: tool_name.to_string(),
                message: "give one of 'at' (once), 'every' (interval), or 'cron' (expression)"
                    .to_string(),
            });
        }
        // Ambiguity is refused rather than resolved by precedence: silently honouring one and
        // dropping the other would produce a job that fires on a schedule nobody asked for.
        several => {
            return Err(MekaError::ToolExecution {
                tool_name: tool_name.to_string(),
                message: format!(
                    "give exactly one schedule, got {}",
                    several
                        .iter()
                        .map(|(key, _)| *key)
                        .collect::<Vec<_>>()
                        .join(" and ")
                ),
            });
        }
    };

    let parsed = match kind {
        "at" => Schedule::parse_at(value, now),
        "every" => Schedule::parse_every(value),
        _ => Schedule::parse_cron(value),
    };
    parsed.map_err(|message| MekaError::ToolExecution {
        tool_name: tool_name.to_string(),
        message,
    })
}

pub(super) struct ScheduleListTool {
    pub context: ScheduleContext,
    /// Carried only to resolve a gate's tool when reporting whether the gate can still fire. The
    /// listing itself creates nothing, so none of the creation limits in here apply to it.
    pub config: ResolvedScheduleConfig,
    pub shared_permission: crate::permission::SharedPermission,
}

#[async_trait]
impl Tool for ScheduleListTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "schedule_list".to_string(),
            description: "List the scheduled jobs for this session, with their next fire times, \
                gates, and full prompts. Your per-turn context already carries a short index, so \
                reach for this when you need a job's exact prompt or gate check."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        _input: serde_json::Value,
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let session_id = resolve_session_id(&self.context.session_id, "schedule_list").await?;
        let jobs = self
            .context
            .session_manager
            .schedule_store()
            .list_scheduled_jobs(session_id)
            .await?;
        if jobs.is_empty() {
            return Ok(ToolOutput::text(
                "No scheduled jobs in this session.".to_string(),
                false,
            ));
        }

        let mut rendered = String::new();
        for job in &jobs {
            rendered.push_str(&format!(
                "{} ({})\n  next: {}\n",
                job.short_id(),
                job.schedule.describe(),
                describe_fire_time(job.next_fire_at)
            ));
            if let Some(fired) = job.last_fired_at {
                rendered.push_str(&format!(
                    "  last fired: {}\n",
                    fired.with_timezone(&Local).format("%Y-%m-%d %H:%M %Z")
                ));
            }
            if let Some(gate) = &job.gate {
                // `detail`, not `summary`: this is the surface the model authored the job on, and
                // it cannot otherwise read back the arguments it wrote.
                rendered.push_str(&format!(
                    "  gate ({}): {}\n",
                    gate.predicate.summary(),
                    gate.probe.detail()
                ));
            }
            // A job that cannot fire is the difference between one quietly waiting and one that is
            // dead, and the two look identical without this line: both simply never fire, and
            // `last fired` is absent for a brand-new job too. Outside the `gate` branch because an
            // ungated job on a session at `none` is held as well. Reported to the model because it
            // can act on it -- `schedule_cancel` needs only `read` -- where until now the only
            // trace was a `warn!` in the operator's log.
            if let Some(reason) = crate::schedule::job_withheld_reason(
                job,
                self.shared_permission.get(),
                self.config.gate_tools.as_deref(),
            ) {
                rendered.push_str(&format!("  NOT FIRING: {}\n", reason));
            }
            rendered.push_str(&format!("  prompt: {}\n", job.prompt));
        }
        Ok(ToolOutput::text(rendered, false))
    }
}

pub(super) struct ScheduleCancelTool {
    pub context: ScheduleContext,
}

#[async_trait]
impl Tool for ScheduleCancelTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "schedule_cancel".to_string(),
            description: "Cancel a scheduled job so it stops firing. Takes the id from \
                schedule_create or schedule_list; a unique prefix is enough."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Job id, or any unique prefix of one"
                    }
                },
                "required": ["id"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let tool_name = "schedule_cancel";
        let session_id = resolve_session_id(&self.context.session_id, tool_name).await?;
        let id = require_str(&input, "id", tool_name)?;

        match self
            .context
            .session_manager
            .schedule_store()
            .cancel_scheduled_job(session_id, &id)
            .await?
        {
            Some(cancelled) => {
                tracing::info!("cancelled scheduled job {}", cancelled);
                Ok(ToolOutput::text(
                    format!("Cancelled job {}.", &cancelled[..8.min(cancelled.len())]),
                    false,
                ))
            }
            None => Ok(ToolOutput::text(
                format!(
                    "No scheduled job matching '{}' in this session. Use schedule_list to see \
                     what is there.",
                    id
                ),
                false,
            )),
        }
    }
}

/// Build the three tools. Kept here so `crate::tools` does not need to know their field shapes.
pub(super) fn build(
    session_manager: SessionManager,
    session_id: Arc<RwLock<Option<Uuid>>>,
    config: ResolvedScheduleConfig,
    shared_permission: crate::permission::SharedPermission,
) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(ScheduleCreateTool {
            session_manager: session_manager.clone(),
            session_id: session_id.clone(),
            config: config.clone(),
            shared_permission: shared_permission.clone(),
        }),
        Arc::new(ScheduleListTool {
            context: ScheduleContext {
                session_manager: session_manager.clone(),
                session_id: session_id.clone(),
            },
            config,
            shared_permission,
        }),
        Arc::new(ScheduleCancelTool {
            context: ScheduleContext {
                session_manager,
                session_id,
            },
        }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::tests::text_content;

    async fn harness() -> (
        ScheduleCreateTool,
        Arc<RwLock<Option<Uuid>>>,
        SessionManager,
    ) {
        let manager =
            SessionManager::open(Some(std::path::Path::new(":memory:")), &Default::default())
                .await
                .expect("open in-memory database");
        let session = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let session_id = Arc::new(RwLock::new(Some(session)));
        let tool = ScheduleCreateTool {
            session_manager: manager.clone(),
            session_id: session_id.clone(),
            config: ResolvedScheduleConfig::default(),
            shared_permission: crate::permission::SharedPermission::new(
                Permission::Unrestricted,
                crate::permission::EnabledPermissions::ALL,
            ),
        };
        (tool, session_id, manager)
    }

    #[tokio::test]
    async fn test_create_reports_the_resolved_fire_time() {
        let (tool, _session_id, _manager) = harness().await;
        let output = tool
            .execute(
                serde_json::json!({"prompt": "check the deploy", "at": "20m"}),
                CancellationToken::new(),
            )
            .await
            .expect("creates");
        let text = text_content(&output);
        assert!(text.contains("Next fire:"), "{text}");
        // The relative half is the part that catches a schedule meaning something other than
        // intended, so it must actually be present.
        assert!(text.contains("in 19m"), "{text}");
    }

    #[tokio::test]
    async fn test_create_requires_exactly_one_schedule() {
        let (tool, _session_id, _manager) = harness().await;
        let none = tool
            .execute(serde_json::json!({"prompt": "x"}), CancellationToken::new())
            .await
            .expect_err("no schedule is an error");
        assert!(none.to_string().contains("one of"), "{none}");

        let both = tool
            .execute(
                serde_json::json!({"prompt": "x", "at": "20m", "every": "1h"}),
                CancellationToken::new(),
            )
            .await
            .expect_err("two schedules is an error");
        assert!(both.to_string().contains("exactly one"), "{both}");
    }

    /// The permission rule that cannot live in `required_permission`, since that is per-tool.
    #[tokio::test]
    async fn test_a_gate_is_refused_below_write_but_the_reminder_is_not() {
        // Every level below `unrestricted`, not just `read`.
        //
        // `workspace` is the one that matters: it used to *pass* this door, and that was a
        // one-call escape -- `schedule_create` with a gate ran arbitrary unconfined commands from
        // inside the confined mode within one poll interval. `ask` matters for the other reason:
        // nobody is present at fire time to answer the prompt its safety rests on. Exercising only
        // `read` left both of those unguarded at the tool door.
        for level in [
            Permission::None,
            Permission::Read,
            Permission::Workspace,
            Permission::Ask,
        ] {
            refuses_a_gate_at(level).await;
        }
    }

    async fn refuses_a_gate_at(level: Permission) {
        let (mut tool, _session_id, _manager) = harness().await;
        tool.shared_permission = crate::permission::SharedPermission::new(
            level,
            crate::permission::EnabledPermissions::ALL,
        );

        let refused = tool
            .execute(
                serde_json::json!({
                    "prompt": "x",
                    "every": "1h",
                    "gate": {"check": {"command": "true"}}
                }),
                CancellationToken::new(),
            )
            .await
            .expect_err("a gate needs an unattended-write level");
        assert!(
            refused.to_string().contains("`unrestricted`"),
            "the refusal at {level} must name the level a gate needs: {refused}"
        );

        // The same job without a gate is fine at read: scheduling a prompt grants nothing, since
        // the turn it produces is permission-checked when it runs.
        tool.execute(
            serde_json::json!({"prompt": "x", "every": "1h"}),
            CancellationToken::new(),
        )
        .await
        .unwrap_or_else(|error| panic!("an ungated reminder is allowed at {level}: {error}"));
    }

    #[tokio::test]
    async fn test_create_enforces_the_job_ceiling() {
        let (mut tool, _session_id, _manager) = harness().await;
        tool.config = ResolvedScheduleConfig {
            max_jobs: 1,
            ..ResolvedScheduleConfig::default()
        };
        tool.execute(
            serde_json::json!({"prompt": "first", "every": "1h"}),
            CancellationToken::new(),
        )
        .await
        .expect("first fits");
        let error = tool
            .execute(
                serde_json::json!({"prompt": "second", "every": "1h"}),
                CancellationToken::new(),
            )
            .await
            .expect_err("second exceeds the ceiling");
        assert!(error.to_string().contains("limit"), "{error}");
    }

    #[tokio::test]
    async fn test_create_rejects_a_one_shot_in_the_past() {
        let (tool, _session_id, _manager) = harness().await;
        let error = tool
            .execute(
                serde_json::json!({"prompt": "x", "at": "2020-01-01T00:00:00Z"}),
                CancellationToken::new(),
            )
            .await
            .expect_err("a past instant has no next occurrence");
        assert!(error.to_string().contains("future"), "{error}");
    }

    /// The model can read back the arguments it wrote into a tool gate.
    ///
    /// `schedule_list` is the only surface that shows them: the HTTP view and `meka schedule list`
    /// are read by parties who did not author the job, and arguments are where a pasted credential
    /// would sit. Without this the model could not tell a gate it built with the wrong `folder`
    /// from a correct one.
    #[tokio::test]
    async fn list_shows_a_tool_gate_s_arguments_to_the_model_that_wrote_them() {
        let (_tool, session_id, manager) = harness().await;
        // Planted through the store rather than `schedule_create`: the creation door needs a live
        // tool dispatcher to authorise a tool gate, and what is under test here is the rendering.
        let id = session_id.read().await.expect("harness made a session");
        let schedule = Schedule::parse_every("1h").expect("parses");
        let now = Utc::now();
        manager
            .schedule_store()
            .create_scheduled_job(&ScheduledJob {
                attempts: 0,
                id: uuid::Uuid::new_v4().to_string(),
                session_id: id,
                schedule: schedule.clone(),
                prompt: "watch it".to_string(),
                gate: Some(Gate {
                    probe: GateProbe::Tool {
                        name: "mcp__bridge__unseen".to_string(),
                        arguments: serde_json::json!({"folder": "sentinel-9c3f"}),
                    },
                    predicate: GatePredicate::Succeeded,
                    last_output: None,
                    permission: Permission::Read,
                }),
                created_at: now,
                last_fired_at: None,
                next_fire_at: schedule.next_after(now).expect("has a next fire"),
            })
            .await
            .expect("plants the job");

        let list = ScheduleListTool {
            context: ScheduleContext {
                session_manager: manager.clone(),
                session_id: session_id.clone(),
            },
            config: ResolvedScheduleConfig::default(),
            shared_permission: crate::permission::SharedPermission::new(
                Permission::Unrestricted,
                crate::permission::EnabledPermissions::DEFAULT,
            ),
        };
        let listed = text_content(
            &list
                .execute(serde_json::json!({}), CancellationToken::new())
                .await
                .expect("lists"),
        );
        assert!(listed.contains("mcp__bridge__unseen"), "{listed}");
        assert!(
            listed.contains("sentinel-9c3f"),
            "the arguments the model wrote must come back to it: {listed}"
        );
    }

    /// `schedule_list` is where the model looks when it wants detail, so it is where the reason a
    /// job is dead belongs. The listing used to show the gate's *definition* and stop there, which
    /// reads as healthy however long the gate has been refused.
    #[tokio::test]
    async fn list_reports_a_gate_that_cannot_currently_fire() {
        let (tool, session_id, manager) = harness().await;
        tool.execute(
            serde_json::json!({
                "prompt": "watch the build",
                "every": "1h",
                "gate": {"check": {"command": "gh pr checks"}, "when": "changed"}
            }),
            CancellationToken::new(),
        )
        .await
        .expect("creates at unrestricted");

        // What the operator did afterwards: dropped the session to `read`.
        let list = ScheduleListTool {
            context: ScheduleContext {
                session_manager: manager.clone(),
                session_id: session_id.clone(),
            },
            config: ResolvedScheduleConfig::default(),
            shared_permission: crate::permission::SharedPermission::new(
                Permission::Read,
                crate::permission::EnabledPermissions::DEFAULT,
            ),
        };
        let listed = text_content(
            &list
                .execute(serde_json::json!({}), CancellationToken::new())
                .await
                .expect("lists"),
        );
        assert!(
            listed.contains("NOT FIRING"),
            "a withheld gate must say so: {listed}"
        );
        assert!(
            listed.contains("unrestricted"),
            "and name what it needs: {listed}"
        );
    }

    #[tokio::test]
    async fn test_list_and_cancel_round_trip() {
        let (tool, session_id, manager) = harness().await;
        tool.execute(
            serde_json::json!({"prompt": "watch the build", "every": "1h"}),
            CancellationToken::new(),
        )
        .await
        .expect("creates");

        let list = ScheduleListTool {
            context: ScheduleContext {
                session_manager: manager.clone(),
                session_id: session_id.clone(),
            },
            config: ResolvedScheduleConfig::default(),
            shared_permission: crate::permission::SharedPermission::new(
                Permission::Unrestricted,
                crate::permission::EnabledPermissions::DEFAULT,
            ),
        };
        let listed = text_content(
            &list
                .execute(serde_json::json!({}), CancellationToken::new())
                .await
                .expect("lists"),
        );
        assert!(listed.contains("watch the build"), "{listed}");

        let short = listed
            .split_whitespace()
            .next()
            .expect("id is the first token")
            .to_string();
        let cancel = ScheduleCancelTool {
            context: ScheduleContext {
                session_manager: manager,
                session_id,
            },
        };
        let cancelled = text_content(
            &cancel
                .execute(serde_json::json!({"id": short}), CancellationToken::new())
                .await
                .expect("cancels"),
        );
        assert!(cancelled.contains("Cancelled job"), "{cancelled}");

        let after = text_content(
            &list
                .execute(serde_json::json!({}), CancellationToken::new())
                .await
                .expect("lists"),
        );
        assert!(after.contains("No scheduled jobs"), "{after}");
    }

    #[tokio::test]
    async fn test_cancel_reports_a_miss_rather_than_failing() {
        let (_tool, session_id, manager) = harness().await;
        let cancel = ScheduleCancelTool {
            context: ScheduleContext {
                session_manager: manager,
                session_id,
            },
        };
        let text = text_content(
            &cancel
                .execute(
                    serde_json::json!({"id": "deadbeef"}),
                    CancellationToken::new(),
                )
                .await
                .expect("a miss is not an error"),
        );
        assert!(text.contains("No scheduled job matching"), "{text}");
    }
}
