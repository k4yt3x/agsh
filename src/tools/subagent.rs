//! `spawn_agent` tool: delegates a self-contained research/exploration task to a fresh sub-agent
//! with its own conversation, returning the sub-agent's final report as a single tool result.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::{BuiltinToolFilter, Tool, ToolOutput, ToolRegistry};
use crate::{
    agent::{Agent, AgentOptions},
    context::build_environment_context,
    conversation::Conversation,
    error::{MekaError, Result},
    permission::{EnabledPermissions, Permission, SharedPermission},
    provider::{Provider, ToolDefinition},
    session::SessionManager,
};

/// Hard ceiling on sub-agent nesting depth, independent of the tunable
/// `session.subagent_max_depth` budget. Guarantees recursion always terminates even if an agent
/// re-grants `max_depth` at every level: no agent nested deeper than this is given a `spawn_agent`
/// tool, so the tree can never exceed this height.
const SUBAGENT_ABSOLUTE_MAX_DEPTH: usize = 16;

/// Parameters needed to build a fresh ToolRegistry for sub-agents.
#[derive(Clone)]
pub struct ToolBuilderParams {
    pub web_client: crate::config::WebClientConfig,
    pub sandbox_enabled: bool,
    pub sandbox_capability: crate::sandbox::SandboxCapability,
    pub sandbox_backend: crate::config::SandboxBackend,
    pub backend_probe: crate::sandbox::BackendProbe,
    /// Parent's `[tools]` filter; sub-agents inherit it.
    pub builtin_filter: BuiltinToolFilter,
    /// Shared skill cache. Sub-agents read from the same cache as the parent so their system
    /// prompts stay consistent and pick up the same auto-reloads.
    pub skills: Arc<crate::skills::SkillCache>,
    /// Parent's MCP client manager, if any servers are configured. When `Some`, every
    /// `spawn_agent` invocation calls [`crate::mcp::McpClientManager::install_tools_on`] on
    /// the freshly-built sub-agent registry so sub-agents see the same MCP resource meta-tools
    /// and per-server adapters as the parent. `None` is the no-MCP-configured case.
    ///
    /// Stored as a `Weak` to break the strong reference cycle that would otherwise form:
    /// `McpClientManager.attached_registries` holds each session's `ToolRegistry`, which holds
    /// this `SpawnAgentTool`, which holds the manager. Without a `Weak`, a session that drops
    /// without `session/close` calling `detach_registry` leaks the entire chain until process
    /// exit.
    pub mcp_manager: Option<std::sync::Weak<crate::mcp::McpClientManager>>,
    /// Shared `SessionManager` so sub-agents can create their own DB session at spawn time and
    /// persist their conversation under it.
    pub session_manager: SessionManager,
    /// Parent agent's session ID. Read at spawn time so the new sub-agent session's
    /// `parent_session_id` column points back here; cascade-on- delete in
    /// `SessionManager::delete_session` then sweeps sub-agent rows when the parent is deleted.
    pub parent_shared_session_id: Arc<RwLock<Option<Uuid>>>,
    /// Parent's session-level counters. Shared so sub-agent token usage rolls up into the same
    /// `/status` totals; operators see the full cost of a session including everything its
    /// sub-agents consumed.
    pub session_stats: Arc<crate::stats::SessionStats>,
    /// Parent's options, used to derive the sub-agent's inherited fields (`sandboxed_shell`,
    /// `context_messages`, `user_instructions`) inside [`Agent::new_subagent`].
    pub parent_options: AgentOptions,
    /// Parent's per-session working directory. Sub-agents snapshot the current value at spawn time
    /// so a parent `/cd` mid-sub-agent-turn can't change the sub-agent's path resolution
    /// mid-flight.
    pub parent_cwd: crate::agent::SharedCwd,
    /// Parent's frontend. Sub-agents wrap it in a
    /// [`crate::frontend::PermissionForwardingFrontend`] so their permission prompts surface
    /// in the parent's UI (REPL line or ACP `session/request_permission`). Without this,
    /// sub-agents have no human to ask and would have to refuse Ask-mode tools outright.
    pub parent_frontend: Arc<dyn crate::frontend::Frontend>,
}

pub struct SpawnAgentTool {
    pub provider: Arc<dyn Provider>,
    pub parent_permission: SharedPermission,
    pub tool_builder_params: ToolBuilderParams,
    pub user_instructions: Option<String>,
    /// Soft, agent-tunable recursion budget for sub-agents spawned by this tool. The root tool is
    /// seeded from `session.subagent_max_depth`; each level hands the child `remaining_depth - 1`
    /// unless the caller overrides it via the `max_depth` param. A nested `spawn_agent` is granted
    /// only while the child's budget is `>= 1`.
    pub remaining_depth: usize,
    /// Monotonic absolute nesting depth of the agent holding this tool (root = 0). Unlike
    /// `remaining_depth` it can't be reset by `max_depth`, so it bounds real recursion at
    /// [`SUBAGENT_ABSOLUTE_MAX_DEPTH`] regardless of what the agent requests.
    pub absolute_depth: usize,
}

#[async_trait]
impl Tool for SpawnAgentTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "spawn_agent".to_string(),
            description: "Spawn a sub-agent to perform a research, analysis, or delegated task. \
                          The sub-agent inherits the parent's permission level, has its own \
                          private todo list and scratchpad, and returns a single text report. \
                          Multiple spawn_agent calls in one turn run in parallel. Pass `skill` \
                          to run an installed skill in the sub-agent. The skill's instructions \
                          become the sub-agent's task; supply at least one of `prompt` or \
                          `skill`. Use `inherit_scratchpad` to grant read-only access to \
                          specific parent scratchpad entries by name so the sub-agent can \
                          consume large captured output via `scratchpad_read` without you \
                          re-inlining it in the prompt. Tip: when you expect to hand output to a \
                          sub-agent later, set the `scratchpad` parameter on the originating \
                          tool call (e.g. `execute_command({command: \"...\", scratchpad: \
                          \"build_log\"})`) so the entry has a semantic name you can pass \
                          through `inherit_scratchpad`. Sub-agents may themselves spawn further \
                          sub-agents up to a configured depth; tune a subtree's depth with \
                          `max_depth`. Pass `permission` to run the sub-agent at a more \
                          restricted level than your own (you can restrict but never escalate)."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task description for the sub-agent. Optional when \
                                        `skill` is given; otherwise required."
                    },
                    "skill": {
                        "type": "string",
                        "description": "Name of an installed skill to run in the sub-agent. The \
                                        skill's instructions become the sub-agent's task; \
                                        `prompt`, if also given, is prepended as extra direction."
                    },
                    "scratchpad": {
                        "type": "string",
                        "description": "If provided, save the sub-agent's final report to the \
                                        parent's scratchpad under this name instead of returning \
                                        it inline."
                    },
                    "inherit_scratchpad": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Names of the parent's scratchpad entries the sub-agent \
                                        is allowed to read. The sub-agent's `scratchpad_read` \
                                        falls back to the parent for these names; \
                                        `scratchpad_list` shows them with origin `inherited`. \
                                        Read-only: `scratchpad_write` / `_edit` / `_delete` \
                                        targeting an inherited name return an error so the \
                                        sub-agent can't silently shadow your copy. Names that \
                                        don't exist in the parent are silently skipped."
                    },
                    "permission": {
                        "type": "string",
                        "enum": ["none", "read", "ask", "write"],
                        "description": "Permission level for the sub-agent, clamped to your own \
                                        level as a ceiling: you can restrict but never escalate. \
                                        Defaults to your current level. Use a lower level (e.g. \
                                        \"read\") to sandbox untrusted or risky work."
                    },
                    "max_depth": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Override how many further levels of sub-agents this \
                                        sub-agent may itself spawn. Defaults to one less than your \
                                        own remaining budget. 0 forbids it from spawning further; \
                                        larger values are still bounded by a built-in absolute \
                                        recursion cap."
                    }
                }
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
        cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        // Both `prompt` and `skill` are optional, but at least one must be present. This mirrors
        // the CLI's `--oneshot` guard in `src/main.rs`. An empty/whitespace `prompt` counts
        // as absent.
        let prompt = input["prompt"]
            .as_str()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_string);
        let skill_name = input["skill"]
            .as_str()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_string);
        if prompt.is_none() && skill_name.is_none() {
            return Err(MekaError::ToolExecution {
                tool_name: "spawn_agent".to_string(),
                message: "spawn_agent requires 'prompt', 'skill', or both".to_string(),
            });
        }

        // Resolve the skill against the shared cache up front, before any session is created, so a
        // bad name fails fast without leaving an orphan child session behind.
        let skill = match &skill_name {
            Some(name) => {
                let installed = self.tool_builder_params.skills.current().await;
                match installed.iter().find(|skill| &skill.name == name) {
                    Some(skill) => Some(skill.clone()),
                    None => {
                        let available: Vec<&str> =
                            installed.iter().map(|skill| skill.name.as_str()).collect();
                        let hint = if available.is_empty() {
                            "No skills are installed.".to_string()
                        } else {
                            format!("Available skills: {}", available.join(", "))
                        };
                        return Err(MekaError::ToolExecution {
                            tool_name: "spawn_agent".to_string(),
                            message: format!("skill '{}' not found. {}", name, hint),
                        });
                    }
                }
            }
            None => None,
        };

        // `inherit_scratchpad`: optional array of parent-scratchpad names. Non-string entries are
        // silently skipped so a partially- malformed array doesn't tank the whole spawn.
        let inherited_scratchpad: Vec<String> = input
            .get("inherit_scratchpad")
            .and_then(|value| value.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();

        // Resolve the sub-agent's permission: an optional `permission` param clamped to the
        // parent's level as a ceiling (restrict-only, never escalate); absent keeps the parent's
        // level. Ask-mode prompts route through `PermissionForwardingFrontend` so they surface in
        // the parent's UI.
        let requested_permission = input["permission"]
            .as_str()
            .map(str::trim)
            .filter(|text| !text.is_empty());
        let sub_perm =
            resolve_subagent_permission(requested_permission, self.parent_permission.get())?;

        // Optional `max_depth`: the caller's override for how deep this sub-agent's own subtree may
        // recurse. Consumed by `child_spawn_depth` when deciding whether to grant a nested
        // `spawn_agent` below.
        let max_depth_override = input
            .get("max_depth")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize);

        // Resolve parent session ID. By the time a tool runs, `Agent::run_turn` has already written
        // `shared_session_id` before dispatching tools. A missing value here means an agent ran a
        // tool without first creating its session, an internal invariant break worth surfacing
        // rather than silently producing an orphan.
        let parent_sid = self
            .tool_builder_params
            .parent_shared_session_id
            .read()
            .await
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "spawn_agent".to_string(),
                message: "parent session ID not yet assigned (run_turn invariant)".to_string(),
            })?;

        // Create the sub-agent's own DB session, linked back to the parent via `parent_session_id`.
        // Cascade-on-delete in `delete_session` sweeps it when the parent is removed.
        let sub_session_id = self
            .tool_builder_params
            .session_manager
            .create_child_session(
                parent_sid,
                Some(crate::agent::cwd_snapshot(
                    &self.tool_builder_params.parent_cwd,
                )),
            )
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "spawn_agent".to_string(),
                message: format!("failed to create sub-agent session: {}", error),
            })?;
        let sub_shared_session_id: Arc<RwLock<Option<Uuid>>> =
            Arc::new(RwLock::new(Some(sub_session_id)));
        tracing::info!(
            "spawning sub-agent: parent={} child={}",
            parent_sid,
            sub_session_id
        );

        // Render the skill body now that the sub-agent's session ID exists, so `${MEKA_SESSION_ID}`
        // resolves to the sub-agent's own session. `load_skill_body` also prepends the
        // base-directory header so bundled-file references resolve.
        let skill_body = match &skill {
            Some(skill) => Some(
                crate::skills::load_skill_body(skill, Some(&sub_session_id.to_string()))
                    .await
                    .map_err(|error| MekaError::ToolExecution {
                        tool_name: "spawn_agent".to_string(),
                        message: format!("failed to load skill: {}", error),
                    })?,
            ),
            None => None,
        };

        // Build a sub-agent tool registry with a fresh, private todo list so the sub-agent's `todo`
        // calls don't touch the parent's task tracking; scratchpad and render_image use the new
        // sub-session ID. A nested `spawn_agent` is added after the registry is built, when the
        // recursion budget allows (see `child_spawn_depth` below). The enabled-permission set is
        // clamped to the sub-agent's level too, so the ceiling can't be re-escalated.
        let sub_shared_perm = SharedPermission::new(
            sub_perm,
            clamp_enabled_permissions(self.parent_permission.enabled(), sub_perm),
        );
        let sub_todo_list: super::todo::SharedTodoList =
            Arc::new(tokio::sync::RwLock::new(super::todo::TodoState::default()));
        // Snapshot the parent's cwd at sub-agent build time so a parent `/cd` mid-sub-agent
        // execution can't shift the sub-agent's path resolution mid-flight. The sub-agent's tool
        // registry sees this snapshot; `Agent::new_subagent` makes the same snapshot for
        // `Agent::cwd()`.
        let sub_cwd: crate::agent::SharedCwd = {
            let parent_path = self
                .tool_builder_params
                .parent_cwd
                .read()
                .map(|guard| guard.clone())
                .unwrap_or_else(|poisoned| poisoned.into_inner().clone());
            Arc::new(std::sync::RwLock::new(parent_path))
        };
        let sub_registry = ToolRegistry::build_for_subagent(
            self.tool_builder_params.web_client.clone(),
            sub_shared_perm.clone(),
            self.tool_builder_params.sandbox_enabled,
            self.tool_builder_params.sandbox_capability.clone(),
            self.tool_builder_params.sandbox_backend,
            self.tool_builder_params.backend_probe.clone(),
            self.tool_builder_params.builtin_filter.clone(),
            sub_todo_list.clone(),
            self.tool_builder_params.session_manager.clone(),
            sub_shared_session_id.clone(),
            self.tool_builder_params.skills.clone(),
            if inherited_scratchpad.is_empty() {
                None
            } else {
                Some(parent_sid)
            },
            inherited_scratchpad.clone(),
            sub_cwd.clone(),
            Arc::clone(&self.tool_builder_params.parent_frontend),
        )
        .map_err(|error| MekaError::ToolExecution {
            tool_name: "spawn_agent".to_string(),
            message: format!("failed to build sub-agent tool registry: {}", error),
        })?;

        // Inherit the parent's MCP toolset. Skipped silently when no MCP manager is attached (no
        // servers configured) or when the parent's servers are still Pending / Failed at spawn
        // time. `install_tools_on` is non-spawning and idempotent. See
        // `src/mcp.rs:install_tools_on`.
        if let Some(weak) = self.tool_builder_params.mcp_manager.as_ref() {
            // Upgrade only if the manager is still alive. If the parent's `meka acp` process is
            // mid-shutdown, the Arc may already be gone. Skip silently.
            if let Some(manager) = weak.upgrade() {
                manager.install_tools_on(&sub_registry).await;
            }
        }

        // Grant the sub-agent its own `spawn_agent` when its recursion budget allows, so it can
        // orchestrate a team of its own sub-agents. Registered here (before the tool catalogue is
        // snapshotted for the system prompt below) and outside `build_for_subagent`, mirroring the
        // root registration in `assemble_agent` (main.rs). `child_spawn_depth` bounds nesting with
        // two counters: `remaining_depth` is the soft, `max_depth`-tunable budget; `absolute_depth`
        // is the hard cap that guarantees termination.
        let (child_remaining_depth, child_absolute_depth, allow_nested_spawn) = child_spawn_depth(
            self.remaining_depth,
            self.absolute_depth,
            max_depth_override,
        );
        if allow_nested_spawn
            && self
                .tool_builder_params
                .builtin_filter
                .admits("spawn_agent")
        {
            let child_params = ToolBuilderParams {
                parent_shared_session_id: sub_shared_session_id.clone(),
                parent_cwd: sub_cwd,
                ..self.tool_builder_params.clone()
            };
            sub_registry.register(Arc::new(SpawnAgentTool {
                provider: Arc::clone(&self.provider),
                parent_permission: sub_shared_perm.clone(),
                tool_builder_params: child_params,
                user_instructions: self.user_instructions.clone(),
                remaining_depth: child_remaining_depth,
                absolute_depth: child_absolute_depth,
            }))?;
        }

        // Build the sub-agent's system prompt against the fully-loaded registry (registry now
        // includes MCP adapters). The override on `AgentOptions` is static, so this single build
        // captures the full tool catalogue visible to the sub-agent.
        let tools = sub_registry.definitions_for_permission(sub_perm);
        let sub_system_prompt = build_subagent_system_prompt(
            sub_perm,
            &tools,
            self.user_instructions.as_deref(),
            &inherited_scratchpad,
        );

        // Compose the first-turn task: parent directive first, skill body second. The at-least-one
        // check above guarantees a `Some`.
        let task =
            compose_subagent_task(prompt.as_deref(), skill_body.as_deref()).ok_or_else(|| {
                MekaError::ToolExecution {
                    tool_name: "spawn_agent".to_string(),
                    message: "spawn_agent requires 'prompt', 'skill', or both".to_string(),
                }
            })?;
        let sub_cwd_snapshot = crate::agent::cwd_snapshot(&self.tool_builder_params.parent_cwd);
        let environment_context = build_environment_context(sub_perm, &sub_cwd_snapshot);
        let augmented_prompt = format!("{}\n{}", environment_context, task);

        // Wrap so permission prompts surface in the parent's UI while emits stay silent (sub-agent
        // output flows back via the spawn_agent tool result, not as live notifications).
        let sub_frontend: Arc<dyn crate::frontend::Frontend> =
            Arc::new(crate::frontend::PermissionForwardingFrontend::new(
                Arc::clone(&self.tool_builder_params.parent_frontend),
            ));

        let sub_agent = Agent::new_subagent(
            Arc::clone(&self.provider),
            sub_registry,
            self.tool_builder_params.session_manager.clone(),
            sub_shared_perm,
            &self.tool_builder_params.parent_options,
            sub_system_prompt,
            sub_todo_list,
            sub_shared_session_id,
            self.tool_builder_params.skills.clone(),
            &self.tool_builder_params.parent_cwd,
            sub_frontend,
            self.tool_builder_params.session_stats.clone(),
        );

        // Run the sub-agent's single turn via the shared `Agent::run_turn` path. Conversation
        // persistence (user message, assistant messages, tool results) happens inside `run_turn`
        // against the sub-session, so the audit trail is identical to a primary agent's. Silent
        // rendering and the omitted MCP gate are baked into the options via `new_subagent`.
        let mut messages = Conversation::new();
        let mut session_id_opt = Some(sub_session_id);
        // Mark every provider request made during this run as a sub-agent request so the Claude
        // OAuth billing header carries `cc_is_subagent=true;` (the provider is a shared `Arc`, so
        // the flag rides a task-local rather than provider state).
        crate::provider::scope_subagent(sub_agent.run_turn(
            &mut session_id_opt,
            &mut messages,
            augmented_prompt,
            Vec::new(),
            cancellation,
        ))
        .await?;

        let report = messages
            .last_assistant_text()
            .unwrap_or_else(|| "(sub-agent produced no final text)".to_string());
        Ok(ToolOutput::text(report, false))
    }
}

/// Parse an optional caller-supplied permission string and clamp it to the parent's level as a
/// ceiling. `None` keeps the parent's level (inherit verbatim). A sub-agent can only ever run at an
/// equal-or-more-restricted level than its parent (`min` over the discriminant order
/// `None < Read < Ask < Write`), so a parent turn can hand risky work to a locked-down sub-agent
/// but can never escalate one. An unrecognized string is a hard error, not a silent fallback.
fn resolve_subagent_permission(requested: Option<&str>, parent: Permission) -> Result<Permission> {
    match requested {
        Some(text) => {
            let requested =
                text.parse::<Permission>()
                    .map_err(|message| MekaError::ToolExecution {
                        tool_name: "spawn_agent".to_string(),
                        message,
                    })?;
            Ok(requested.min(parent))
        }
        None => Ok(parent),
    }
}

/// Restrict an `EnabledPermissions` set to modes at or below `ceiling`. Defense-in-depth for the
/// permission clamp: sub-agents have no runtime permission-switch path today, so their initial
/// level is what governs, but bounding the enabled set means any future switch path can't climb a
/// sub-agent back above the ceiling its parent set. Falls back to a singleton `{ceiling}` set when
/// the intersection is empty (e.g. the parent enabled only higher modes).
fn clamp_enabled_permissions(
    enabled: EnabledPermissions,
    ceiling: Permission,
) -> EnabledPermissions {
    EnabledPermissions::from_modes(enabled.iter().filter(|mode| *mode <= ceiling)).unwrap_or_else(
        || EnabledPermissions::from_modes([ceiling]).unwrap_or(EnabledPermissions::DEFAULT),
    )
}

/// Compute the recursion budget for a sub-agent one level below a `SpawnAgentTool` with the given
/// `remaining_depth` / `absolute_depth`, honoring an optional `max_depth` override.
///
/// Returns `(child_remaining, child_absolute, allow_nested)`. `remaining_depth` is the soft,
/// agent-tunable budget seeded from `session.subagent_max_depth`; `max_depth` may raise it.
/// `absolute_depth` is the monotonic hard counter: it always increments and, once it reaches
/// [`SUBAGENT_ABSOLUTE_MAX_DEPTH`], no further `spawn_agent` is granted, so recursion terminates
/// even if the agent re-grants `max_depth` at every level. A nested `spawn_agent` is granted only
/// when both budgets allow it.
fn child_spawn_depth(
    remaining_depth: usize,
    absolute_depth: usize,
    max_depth_override: Option<usize>,
) -> (usize, usize, bool) {
    let child_remaining = match max_depth_override {
        Some(requested) => requested,
        None => remaining_depth.saturating_sub(1),
    };
    let child_absolute = absolute_depth + 1;
    let allow_nested = child_remaining >= 1 && child_absolute < SUBAGENT_ABSOLUTE_MAX_DEPTH;
    (child_remaining, child_absolute, allow_nested)
}

/// Compose the sub-agent's first-turn task from an optional parent directive and an optional
/// rendered skill body. Mirrors the CLI's `--skill` ordering (`build_skill_prompt` in
/// `src/main.rs`): the parent directive comes first, the skill body second. Returns `None` only
/// when both inputs are absent; the caller treats that as an error.
fn compose_subagent_task(prompt: Option<&str>, skill_body: Option<&str>) -> Option<String> {
    match (prompt, skill_body) {
        (Some(prompt), Some(body)) => Some(format!("{}\n\n{}", prompt, body)),
        (Some(prompt), None) => Some(prompt.to_string()),
        (None, Some(body)) => Some(body.to_string()),
        (None, None) => None,
    }
}

fn build_subagent_system_prompt(
    permission: Permission,
    tools: &[ToolDefinition],
    user_instructions: Option<&str>,
    inherited_scratchpad: &[String],
) -> String {
    let mut prompt = String::new();
    prompt.push_str(
        "You are a research sub-agent. Complete the assigned task using the \
         available tools, then produce a concise final report summarizing your \
         findings. Do not ask follow-up questions. Work with what you have. \
         For multi-step work, use the `todo` tool to plan and track progress: \
         pass `items` together with a `title` to (re)write the list, `set` to \
         update statuses by task number, and call `todo` with no arguments to \
         read the current list. Your todo list is private to this sub-agent.\n\n",
    );

    prompt.push_str(&format!("## Permission Level: {}\n\n", permission));

    if let Some(instructions) = user_instructions
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        prompt.push_str("## User Instructions\n\n");
        prompt.push_str(
            "These are installation-specific rules set by the user. Treat them as \
             hard constraints unless they conflict with safety requirements.\n\n",
        );
        prompt.push_str(instructions);
        prompt.push_str("\n\n");
    }

    if !inherited_scratchpad.is_empty() {
        prompt.push_str("## Inherited Scratchpad Entries\n\n");
        prompt.push_str(
            "Your parent agent has granted you read-only access to the following \
             scratchpad entries from its own session. Use `scratchpad_read` with \
             the exact names below to load them on demand. Do not assume their \
             contents without reading. `scratchpad_write`, `_edit`, and `_delete` \
             against these names will return an error; if you need to derive new \
             state, save it under a different name (e.g. `<name>_local`).\n\n",
        );
        for name in inherited_scratchpad {
            prompt.push_str(&format!("- {}\n", name));
        }
        prompt.push('\n');
    }

    if !tools.is_empty() {
        prompt.push_str("## Available Tools\n\n");
        for tool in tools {
            prompt.push_str(&format!("- **{}**: {}\n", tool.name, tool.description));
        }
        prompt.push('\n');
    }

    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subagent_system_prompt_reflects_inherited_permission() {
        let prompt = build_subagent_system_prompt(Permission::Write, &[], None, &[]);
        assert!(
            prompt.contains(&format!("## Permission Level: {}", Permission::Write)),
            "expected Write level in prompt, got: {}",
            prompt
        );

        let read_prompt = build_subagent_system_prompt(Permission::Read, &[], None, &[]);
        assert!(read_prompt.contains(&format!("## Permission Level: {}", Permission::Read)));
    }

    #[test]
    fn test_subagent_system_prompt_mentions_todo_tools() {
        let prompt = build_subagent_system_prompt(Permission::Read, &[], None, &[]);
        assert!(
            prompt.contains("`todo` tool"),
            "expected todo tool mention in prompt, got: {}",
            prompt
        );
    }

    #[test]
    fn test_subagent_system_prompt_omits_inheritance_section_when_empty() {
        let prompt = build_subagent_system_prompt(Permission::Read, &[], None, &[]);
        assert!(
            !prompt.contains("Inherited Scratchpad"),
            "no inherited section expected for empty allowlist, got: {}",
            prompt
        );
    }

    #[test]
    fn test_subagent_system_prompt_lists_inherited_names() {
        let names = vec!["captured_output".to_string(), "research_notes".to_string()];
        let prompt = build_subagent_system_prompt(Permission::Read, &[], None, &names);
        assert!(prompt.contains("## Inherited Scratchpad Entries"));
        assert!(prompt.contains("- captured_output"));
        assert!(prompt.contains("- research_notes"));
        assert!(prompt.contains("scratchpad_read"));
    }

    #[test]
    fn test_subagent_system_prompt_warns_inherited_writes_will_error() {
        let names = vec!["build_log".to_string()];
        let prompt = build_subagent_system_prompt(Permission::Read, &[], None, &names);
        assert!(
            prompt.contains("will return an error"),
            "expected write-rejection wording, got: {}",
            prompt,
        );
        assert!(
            prompt.contains("_local"),
            "expected naming suggestion, got: {}",
            prompt,
        );
    }

    #[test]
    fn test_compose_subagent_task_combinations() {
        assert_eq!(
            compose_subagent_task(Some("focus on UK news"), Some("skill body")),
            Some("focus on UK news\n\nskill body".to_string()),
            "parent directive must come first, skill body second",
        );
        assert_eq!(
            compose_subagent_task(Some("just a prompt"), None),
            Some("just a prompt".to_string()),
        );
        assert_eq!(
            compose_subagent_task(None, Some("skill body")),
            Some("skill body".to_string()),
        );
        assert_eq!(compose_subagent_task(None, None), None);
    }

    #[test]
    fn test_resolve_subagent_permission_inherits_when_absent() {
        assert_eq!(
            resolve_subagent_permission(None, Permission::Write).unwrap(),
            Permission::Write
        );
        assert_eq!(
            resolve_subagent_permission(None, Permission::Read).unwrap(),
            Permission::Read
        );
    }

    #[test]
    fn test_resolve_subagent_permission_clamps_to_parent_ceiling() {
        // Requesting a higher level than the parent is clamped down: a sub-agent can never be
        // escalated above its parent.
        assert_eq!(
            resolve_subagent_permission(Some("write"), Permission::Read).unwrap(),
            Permission::Read
        );
        assert_eq!(
            resolve_subagent_permission(Some("ask"), Permission::Read).unwrap(),
            Permission::Read
        );
        // Requesting a lower level restricts the sub-agent below the parent.
        assert_eq!(
            resolve_subagent_permission(Some("read"), Permission::Write).unwrap(),
            Permission::Read
        );
        assert_eq!(
            resolve_subagent_permission(Some("none"), Permission::Write).unwrap(),
            Permission::None
        );
    }

    #[test]
    fn test_resolve_subagent_permission_rejects_invalid() {
        assert!(resolve_subagent_permission(Some("admin"), Permission::Write).is_err());
    }

    #[test]
    fn test_clamp_enabled_permissions_drops_higher_modes() {
        let clamped = clamp_enabled_permissions(EnabledPermissions::ALL, Permission::Read);
        assert!(clamped.is_enabled(Permission::None));
        assert!(clamped.is_enabled(Permission::Read));
        assert!(!clamped.is_enabled(Permission::Ask));
        assert!(!clamped.is_enabled(Permission::Write));
    }

    #[test]
    fn test_clamp_enabled_permissions_falls_back_to_singleton() {
        // Parent enabled only Write; clamping to Read leaves an empty intersection, so we fall back
        // to a singleton set of the ceiling itself rather than an invalid empty set.
        let only_write = EnabledPermissions::from_modes([Permission::Write]).unwrap();
        let clamped = clamp_enabled_permissions(only_write, Permission::Read);
        assert!(clamped.is_enabled(Permission::Read));
        assert!(!clamped.is_enabled(Permission::Write));
    }

    #[test]
    fn test_child_spawn_depth_natural_decrement() {
        let (remaining, absolute, allow) = child_spawn_depth(3, 0, None);
        assert_eq!(remaining, 2);
        assert_eq!(absolute, 1);
        assert!(allow);
    }

    #[test]
    fn test_child_spawn_depth_leaf_when_budget_exhausted() {
        // remaining_depth = 1 reproduces the historical "root spawns, sub-agents can't" behavior:
        // the child gets 0 and is not granted a nested spawn_agent.
        let (remaining, _absolute, allow) = child_spawn_depth(1, 0, None);
        assert_eq!(remaining, 0);
        assert!(!allow);
    }

    #[test]
    fn test_child_spawn_depth_override_raises_soft_budget() {
        // An explicit max_depth raises the soft budget above the natural decrement.
        let (remaining, _absolute, allow) = child_spawn_depth(1, 0, Some(5));
        assert_eq!(remaining, 5);
        assert!(allow);
        // max_depth = 0 explicitly forbids the sub-agent from spawning further.
        let (_remaining, _absolute, allow_zero) = child_spawn_depth(3, 0, Some(0));
        assert!(!allow_zero);
    }

    #[test]
    fn test_child_spawn_depth_absolute_cap_forces_leaf() {
        // Even with a large soft budget, the monotonic absolute counter stops recursion at the cap.
        let (_remaining, absolute, allow) =
            child_spawn_depth(100, SUBAGENT_ABSOLUTE_MAX_DEPTH - 1, Some(100));
        assert_eq!(absolute, SUBAGENT_ABSOLUTE_MAX_DEPTH);
        assert!(!allow);
    }

    async fn test_session_manager() -> SessionManager {
        SessionManager::open(Some(std::path::Path::new(":memory:")))
            .await
            .expect("in-memory session manager")
    }

    // (Permission gating and "Unknown tool" fold-into-ToolOutput semantics that used to live in
    // `run_subagent_tool` are now exercised by the shared `Agent::run_turn` path's tool-dispatch
    // logic, covered by `src/agent.rs` and `src/tools.rs` test suites.)

    #[tokio::test]
    async fn test_subagent_registry_has_independent_todo_list() {
        use crate::{
            sandbox::{BackendProbe, SandboxCapability},
            tools::BuiltinToolFilter,
        };

        let parent_list: super::super::todo::SharedTodoList = Arc::new(tokio::sync::RwLock::new(
            super::super::todo::TodoState::default(),
        ));
        let sub_list: super::super::todo::SharedTodoList = Arc::new(tokio::sync::RwLock::new(
            super::super::todo::TodoState::default(),
        ));

        let sub_registry = ToolRegistry::build_for_subagent(
            crate::config::WebClientConfig::default(),
            SharedPermission::new(Permission::Read, crate::permission::EnabledPermissions::ALL),
            true,
            SandboxCapability::Unavailable,
            crate::config::SandboxBackend::Landlock,
            BackendProbe::Missing {
                reason: "test fixture".to_string(),
            },
            BuiltinToolFilter::default(),
            sub_list.clone(),
            test_session_manager().await,
            Arc::new(tokio::sync::RwLock::new(None)),
            crate::skills::SkillCache::for_root(None),
            None,
            Vec::new(),
            crate::agent::test_cwd(),
            Arc::new(crate::frontend::SilentFrontend),
        )
        .expect("subagent registry should build");

        let todo = sub_registry.get("todo").expect("subagent should have todo");
        todo.execute(
            serde_json::json!({ "title": "Sub work", "items": ["sub task"] }),
            CancellationToken::new(),
        )
        .await
        .expect("todo should succeed");

        assert_eq!(sub_list.read().await.items.len(), 1);
        assert!(
            parent_list.read().await.items.is_empty(),
            "parent list must remain untouched"
        );
    }
}
