//! The `schedule_*` tools: the agent arranging its own future turns ([`crate::schedule`]).
//!
//! Creating and listing gate at [`Permission::Read`], matching `memory_*` and `scratchpad_*`: a
//! scheduled prompt writes to a store meka owns, and the turn it eventually produces is permission
//! -checked when it runs, so scheduling one grants nothing the session did not already have.
//!
//! The `gate` field is the exception, and is rejected below [`Permission::Write`] inside
//! [`ScheduleCreateTool::execute`]. A gate is a shell command that runs unattended, on a timer,
//! until someone cancels it -- persistent in a way `execute_command` is not, since that at least
//! ends with the turn that called it. [`Tool::required_permission`] is per-tool and cannot vary by
//! argument, so the check has to live in the body.
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
    schedule::{Gate, GateFire, Schedule, ScheduledJob},
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
                        "description": "Optional. Run a shell command first and only take a turn \
                                        if it says something happened. Turns an expensive poll \
                                        into a cheap one, so a short `every` becomes affordable. \
                                        Requires write permission.",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Shell command. A fast, read-only \
                                                check whose output changes when, and \
                                                only when, the watched thing does. \
                                                Output that moves on its own (a \
                                                timestamp) fires every tick; output \
                                                that can return to an earlier value \
                                                between polls (a bare count) hides \
                                                the change in between."
                            },
                            "fire": {
                                "type": "string",
                                "enum": ["on-change", "on-success"],
                                "description": "'on-change' (default) fires when stdout differs \
                                                from last time, for 'tell me when X changes'. \
                                                'on-success' fires while the command exits 0, for \
                                                'is X true yet'."
                            }
                        },
                        "required": ["command"]
                    },
                    "isolated": {
                        "type": "boolean",
                        "description": "Run in a fresh session instead of this conversation. \
                                        Cheaper for a recurring job, since this conversation's \
                                        history is not replayed on every fire and the fires do \
                                        not pile up in it. The trade is continuity: the turn \
                                        recalls nothing said here and its result does not \
                                        appear here. Leave it false when the job depends on \
                                        that. Only `meka serve` honours it."
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
        let isolated = input["isolated"].as_bool().unwrap_or(false);

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
            isolated,
            created_at: now,
            last_fired_at: None,
            next_fire_at,
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
            summary.push_str(" Gated: a turn happens only when the gate command says so.");
        }
        if job.isolated {
            summary.push_str(" Runs in a fresh session, so its result will not appear here.");
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
        // A gate outlives the turn that created it and runs with no supervision, which is a
        // stronger grant than `execute_command`'s one-shot use. Checked here rather than through
        // `required_permission` so an ungated reminder still works at read.
        // Matched explicitly rather than compared. `Permission`'s `Ord` exists to clamp a
        // sub-agent's level to its parent's and its own docs warn against reusing it for
        // authorization; the capability predicate `allows` is the usual tool, but it treats `Ask`
        // and `Write` as equal, and `Ask` is precisely the level a gate must not run at -- there is
        // nobody to approve an unattended command.
        let permission = self.shared_permission.get();
        if !matches!(permission, Permission::Write) {
            let reason = match permission {
                Permission::Ask => {
                    "a gate runs unattended, with nobody present to approve it, so \
                                    `ask` is not enough"
                }
                _ => "a gate runs a shell command unattended on a schedule",
            };
            return Err(MekaError::ToolExecution {
                tool_name: tool_name.to_string(),
                message: format!(
                    "{}; it needs write permission (currently {}). Create the job without `gate` \
                     and check the condition inside the prompt instead.",
                    reason, permission
                ),
            });
        }

        let command = require_str(raw, "command", tool_name)?;
        let fire = match raw.get("fire").and_then(serde_json::Value::as_str) {
            None => GateFire::OnChange,
            Some(text) => GateFire::parse(text).map_err(|message| MekaError::ToolExecution {
                tool_name: tool_name.to_string(),
                message,
            })?,
        };
        Ok(Some(Gate {
            command: command.to_string(),
            fire,
            last_output: None,
            // Recorded, not re-derived. The check above proves the level *now*; the row will be
            // executed by some other process on some later day, and `prepare` re-reads this to
            // confirm the authority still stands. `permission` is `Write` here by the guard above.
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
}

#[async_trait]
impl Tool for ScheduleListTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "schedule_list".to_string(),
            description: "List the scheduled jobs for this session, with their next fire times, \
                gates, and full prompts. Your per-turn context already carries a short index, so \
                reach for this when you need a job's exact prompt or gate command."
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
                rendered.push_str(&format!(
                    "  gate ({}): {}\n",
                    gate.fire.as_str(),
                    gate.command
                ));
            }
            if job.isolated {
                rendered.push_str("  runs in a fresh session\n");
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
            config,
            shared_permission,
        }),
        Arc::new(ScheduleListTool {
            context: ScheduleContext {
                session_manager: session_manager.clone(),
                session_id: session_id.clone(),
            },
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
        let manager = SessionManager::open(Some(std::path::Path::new(":memory:")))
            .await
            .expect("open in-memory database");
        let session = manager.create_session(None).await.expect("create session");
        let session_id = Arc::new(RwLock::new(Some(session)));
        let tool = ScheduleCreateTool {
            session_manager: manager.clone(),
            session_id: session_id.clone(),
            config: ResolvedScheduleConfig::default(),
            shared_permission: crate::permission::SharedPermission::new(
                Permission::Write,
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
        let (mut tool, _session_id, _manager) = harness().await;
        tool.shared_permission = crate::permission::SharedPermission::new(
            Permission::Read,
            crate::permission::EnabledPermissions::ALL,
        );

        let refused = tool
            .execute(
                serde_json::json!({
                    "prompt": "x",
                    "every": "1h",
                    "gate": {"command": "true"}
                }),
                CancellationToken::new(),
            )
            .await
            .expect_err("a gate needs write");
        assert!(
            refused.to_string().contains("write permission"),
            "{refused}"
        );

        // The same job without a gate is fine at read: scheduling a prompt grants nothing, since
        // the turn it produces is permission-checked when it runs.
        tool.execute(
            serde_json::json!({"prompt": "x", "every": "1h"}),
            CancellationToken::new(),
        )
        .await
        .expect("an ungated reminder is allowed at read");
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
