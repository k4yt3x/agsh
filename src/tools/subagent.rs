//! `agent_spawn` tool: delegates a self-contained research/exploration task to a fresh sub-agent
//! with its own conversation, returning the sub-agent's final report as a single tool result.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::{BuiltinToolFilter, Tool, ToolDenials, ToolOutput, ToolRegistry};
use crate::{
    agent::{Agent, AgentOptions},
    config::{InstructionAccess, MemoryAccess},
    context::build_environment_context,
    conversation::Conversation,
    error::{MekaError, Result},
    permission::{EnabledPermissions, Permission, SharedPermission},
    provider::{Provider, ToolDefinition},
    session::SessionManager,
};

/// Hard ceiling on sub-agent nesting depth, independent of the tunable
/// `session.subagent_max_depth` budget. Guarantees recursion always terminates even if an agent
/// re-grants `max_depth` at every level: no agent nested deeper than this is given a `agent_spawn`
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
    /// Shared memory cache, reached at whatever level `memory_access` allows. The cache itself is
    /// shared because memory is scoped to the meka instance, not to a conversation.
    pub memories: Arc<crate::memory::MemoryStore>,
    /// How much of the store the agent doing the spawning holds, which is the ceiling on what it
    /// can grant. `Write` for the primary agent; for a worker, whatever its own spawn call
    /// granted. A grant is clamped against this, so authority only ever narrows going down the
    /// tree.
    pub memory_access: MemoryAccess,
    /// `[subagents].disabled_servers` / `disabled_tools` as config reads *now*.
    ///
    /// Distinct from `AgentSpawnTool::inherited_denials`, which is the accumulated set for this
    /// agent's children. This one exists for the follow-up path: a worker is rebuilt from the
    /// terms it was spawned under, so without re-applying current config, an operator who adds
    /// a denial and resumes a session would find their existing workers still reaching what
    /// they just took away. Restrictions are combined, never replaced, so this can only ever
    /// narrow.
    pub config_denials: ToolDenials,
    /// Parent's MCP client manager, if any servers are configured. When `Some`, every
    /// `agent_spawn` invocation calls [`crate::mcp::McpClientManager::install_tools_on`] on
    /// the freshly-built sub-agent registry so sub-agents see the same MCP resource meta-tools
    /// and per-server adapters as the parent. `None` is the no-MCP-configured case.
    ///
    /// Stored as a `Weak` to break the strong reference cycle that would otherwise form:
    /// `McpClientManager.attached_registries` holds each session's `ToolRegistry`, which holds
    /// this `AgentSpawnTool`, which holds the manager. Without a `Weak`, a session that drops
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
    /// `context_messages`, the auto-compaction settings) inside [`Agent::new_subagent`].
    /// `user_instructions` is deliberately *not* among them; see
    /// [`build_subagent_system_prompt`].
    pub parent_options: AgentOptions,
    /// Parent's per-session working directory. Sub-agents snapshot the current value at spawn time
    /// so a parent `/cd` mid-sub-agent-turn can't change the sub-agent's path resolution
    /// mid-flight.
    pub parent_cwd: crate::workspace::SharedCwd,
    /// Parent's extra workspace roots, so a delegated search sees the same folders the parent does
    /// and, at `workspace` permission, may write to the same ones.
    ///
    /// Shared rather than snapshotted, unlike `parent_cwd`: that one is copied into a fresh `Arc`
    /// at spawn time so a mid-flight `/cd` in the parent can't move a running sub-agent. Roots
    /// have no such hazard today because ACP fixes them for a session runtime's lifetime and
    /// nothing mutates them after construction. If that ever changes, this needs the same
    /// snapshot.
    pub parent_roots: crate::workspace::SharedRoots,
    /// Parent's frontend. Sub-agents wrap it in a
    /// [`crate::frontend::PermissionForwardingFrontend`] so their permission prompts surface
    /// in the parent's UI (REPL line or ACP `session/request_permission`). Without this,
    /// sub-agents have no human to ask and would have to refuse Ask-mode tools outright.
    pub parent_frontend: Arc<dyn crate::frontend::Frontend>,
}

/// The terms a sub-agent was spawned under, persisted as JSON on its session row
/// (`sessions.subagent_spec_json`).
///
/// **A follow-up rebuilds the worker from this, never from the parent's current state.** That is
/// the whole reason it exists. Rebuilding from the parent would mean `agent_spawn({permission:
/// "read"})` followed by `agent_followup` runs the same worker at whatever the parent is now,
/// turning a second question into a one-call privilege escalation. The same holds for the deny
/// lists and the memory level: every restriction the spawn call chose has to outlive the call.
///
/// Every field except `permission` carries a `#[serde(default)]`, so a spec written by a build with
/// fewer fields still loads; a spec missing `permission` is a hard decode error, which
/// `agent_followup` turns into a refusal. Each default is the *restrictive* value, so a spec that
/// loses a field loses authority rather than gaining it.
///
/// What is deliberately *not* here: the cwd (already on the session row) and the task (already in
/// the event log).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SubagentSpec {
    /// The level the worker ran at, already clamped against its parent at spawn time.
    pub permission: Permission,
    /// The clamped enabled-permission set, so a rebuilt worker cannot climb past the ceiling its
    /// spawn call set even if a future runtime switch path appears.
    #[serde(default)]
    pub enabled_permissions: Vec<Permission>,
    #[serde(default)]
    pub denied_servers: Vec<String>,
    #[serde(default)]
    pub denied_tools: Vec<String>,
    /// A spec that somehow lacks the field costs the worker its memory tools rather than handing
    /// it the store. [`MemoryAccess`] has no `Default` for the same reason: the only sensible one
    /// would be `Write`, which is the primary agent's level and the wrong answer everywhere else.
    #[serde(default = "MemoryAccess::none")]
    pub memory: MemoryAccess,
    /// Whether the worker was handed the installation's instructions file. Defaults to `None` for
    /// a spec that lost the field, for the same fail-closed reason as `memory`.
    #[serde(default)]
    pub instructions: InstructionAccess,
    /// Parent scratchpad names the worker may read. Persisted because it is baked into both the
    /// registry and the system prompt at spawn; without it a follow-up would silently lose the
    /// entries the first turn was working from.
    #[serde(default)]
    pub inherited_scratchpad: Vec<String>,
    /// The worker's own recursion budgets, so a followed-up worker can spawn exactly what it could
    /// have spawned on its first turn.
    #[serde(default)]
    pub remaining_depth: usize,
    #[serde(default)]
    pub absolute_depth: usize,
}

impl SubagentSpec {
    fn denials(&self) -> ToolDenials {
        ToolDenials::new(self.denied_servers.clone(), self.denied_tools.clone())
    }

    /// The worker's `SharedPermission`, clamped to `ceiling` on top of what the spec recorded.
    ///
    /// Two ceilings, because there are two ways a worker could end up with authority it should not
    /// have. The spec stops a follow-up *escalating* a worker its spawn call deliberately
    /// restricted. `ceiling` -- the parent's level right now -- stops a worker *outliving* a
    /// restriction: spawn at `unrestricted`, switch the session to `read`, and without this the
    /// worker would still run at `unrestricted` on the next follow-up, so the user's downgrade
    /// would silently not reach the work being done on their behalf. The effective level is the
    /// lower of the two.
    ///
    /// Falls back to a singleton set when the persisted list is empty or invalid, rather than to
    /// `EnabledPermissions::ALL`: an unreadable spec must not widen what the worker can do.
    ///
    /// Test-only. Production goes through [`Self::shared_permission_bounded`], which applies this
    /// same clamp and then stays bound to the parent; this one is kept because the clamp is worth
    /// asserting in isolation from the tracking.
    #[cfg(test)]
    fn shared_permission(&self, ceiling: Permission) -> SharedPermission {
        let effective = self.effective_permission(ceiling);
        SharedPermission::new(effective, self.clamped_enabled(effective))
    }

    /// Same, but bounded by the parent's *live* level rather than a snapshot of it.
    ///
    /// This is what production uses. The spawn-time clamp is still applied (a worker granted `read`
    /// under a `write` parent stays at `read`), and the ceiling then tracks the parent afterwards,
    /// so cycling the parent down to `none` stops the worker on its next tool call instead of
    /// letting it run to completion at the level it started with.
    fn shared_permission_bounded(&self, parent: &SharedPermission) -> SharedPermission {
        let effective = self.effective_permission(parent.get());
        SharedPermission::with_ceiling(effective, self.clamped_enabled(effective), parent)
    }

    fn clamped_enabled(&self, effective: Permission) -> EnabledPermissions {
        let enabled = EnabledPermissions::from_modes(self.enabled_permissions.iter().copied())
            .unwrap_or_else(|| {
                EnabledPermissions::from_modes([effective]).unwrap_or(EnabledPermissions::DEFAULT)
            });
        clamp_enabled_permissions(enabled, effective)
    }

    /// The memory level this worker actually gets, capped at `Read`.
    ///
    /// `MemoryAccess::parse_grant` refuses `"write"` at the `agent_spawn` boundary, but a spec is
    /// persisted JSON and `meka session import` writes `subagent_spec_json` verbatim from a
    /// user-supplied archive, where `Write` deserializes fine. The documented guarantee is that no
    /// sub-agent can write to the store, so it is enforced where the level is *consumed* rather
    /// than resting on every writer having validated first -- the same shape as
    /// [`clamp_enabled_permissions`], which bounds a persisted permission set for the same reason.
    fn granted_memory(&self) -> MemoryAccess {
        self.memory.min(MemoryAccess::Read)
    }

    /// The level this worker actually runs at, given the parent's current ceiling.
    /// What a **replayed** grant resolves to under the parent's current level.
    ///
    /// `greatest_within_both`, not `clamp_to`: a follow-up must never run the worker at more than
    /// the spawn call asked for. `clamp_to` would resolve a recorded `workspace` under an `ask`
    /// parent to `ask`, handing the worker whole-filesystem reach that the original
    /// `agent_spawn({permission: "workspace"})` explicitly declined. See the helper for why spawn
    /// and replay are different questions.
    fn effective_permission(&self, ceiling: Permission) -> Permission {
        self.permission.greatest_within_both(ceiling)
    }
}

pub struct AgentSpawnTool {
    pub provider: Arc<dyn Provider>,
    pub parent_permission: SharedPermission,
    pub tool_builder_params: ToolBuilderParams,
    /// Everything a sub-agent spawned by *this* tool is denied. Seeded from `[subagents]` at the
    /// root and, for a nested `agent_spawn`, from its own parent's effective set unioned with what
    /// that spawn call added.
    ///
    /// Carried on the tool rather than looked up per call because it has to accumulate: a worker
    /// that could spawn a grandchild free of its own denials would make every restriction one
    /// `agent_spawn` deep.
    pub inherited_denials: ToolDenials,
    /// Soft, agent-tunable recursion budget for sub-agents spawned by this tool. The root tool is
    /// seeded from `session.subagent_max_depth`; each level hands the child `remaining_depth - 1`
    /// unless the caller overrides it via the `max_depth` param. A nested `agent_spawn` is granted
    /// only while the child's budget is `>= 1`.
    pub remaining_depth: usize,
    /// Monotonic absolute nesting depth of the agent holding this tool (root = 0). Unlike
    /// `remaining_depth` it can't be reset by `max_depth`, so it bounds real recursion at
    /// [`SUBAGENT_ABSOLUTE_MAX_DEPTH`] regardless of what the agent requests.
    pub absolute_depth: usize,
}

#[async_trait]
impl Tool for AgentSpawnTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "agent_spawn".to_string(),
            description: "Spawn a sub-agent to perform a research, analysis, or delegated task. \
                          The sub-agent inherits the parent's permission level, has its own \
                          private todo list and scratchpad, and returns a single text report. \
                          Multiple agent_spawn calls in one turn run in parallel. Pass `skill` \
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
                          restricted level than your own (you can restrict but never escalate), \
                          and `deny_servers` / `deny_tools` to withhold MCP servers or individual \
                          tools it would otherwise inherit. Restrictions only ever accumulate: \
                          these add to whatever the installation already denies sub-agents, and \
                          there is no way to grant back."
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
                        "enum": ["none", "read", "workspace", "ask", "unrestricted"],
                        "description": "Permission level for the sub-agent, bounded by your own: \
                                        the worker never gets more reach than you have. Defaults \
                                        to your current level. Use a lower level (e.g. \"read\") \
                                        to sandbox untrusted or risky work. Note that \
                                        \"workspace\" and \"ask\" do not contain each other, so \
                                        asking for one while you are at the other gives the worker \
                                        \"read\" -- the most both of you vouch for. The result is \
                                        reported back to you."
                    },
                    "memory": {
                        "type": "string",
                        "enum": ["none", "read"],
                        "description": "Grant the sub-agent read access to your memory store. \
                                        Defaults to \"none\": a worker starts with a clean slate, \
                                        since memories from unrelated work are context it pays for \
                                        and reasons from. Grant \"read\" when the task genuinely \
                                        depends on what you have recorded. Sub-agents can never \
                                        write to the store; record anything worth keeping \
                                        yourself, from the worker's report."
                    },
                    "instructions": {
                        "type": "string",
                        "enum": ["none", "inherit"],
                        "description": "Give the sub-agent the installation's instructions file. \
                                        Defaults to \"none\", because those instructions describe \
                                        you -- your persona, how to address the user -- and a \
                                        worker is not you. Pass \"inherit\" when the task needs \
                                        the project's standing rules verbatim and quoting the \
                                        relevant ones into `prompt` would be lossy or expensive."
                    },
                    "deny_servers": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "MCP server names the sub-agent must not see. Removes \
                                        everything the server offers: its tools, its resources, \
                                        and its prompts. Use this when a server exists to act on \
                                        your behalf or to talk to the user, so a worker cannot \
                                        speak as you."
                    },
                    "deny_tools": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Individual tool names the sub-agent must not see, as they \
                                        appear in your own tool list (e.g. \"write_file\", \
                                        \"mcp__notion__create_page\"). For a whole server, prefer \
                                        `deny_servers`."
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
                tool_name: "agent_spawn".to_string(),
                message: "agent_spawn requires 'prompt', 'skill', or both".to_string(),
            });
        }

        // Resolve the skill against the shared cache up front, before any session is created, so a
        // bad name fails fast without leaving an orphan child session behind.
        let skill = match &skill_name {
            Some(name) => {
                let installed = self.tool_builder_params.skills.current().await;
                match installed.find(name) {
                    Some(skill) => Some(skill.clone()),
                    // A skill whose `SKILL.md` will not parse is in no index, so listing what *is*
                    // available answers a question the caller did not ask and invites it to pick a
                    // substitute. Name the file instead, the way `skill_read` does.
                    None if installed.skip_reason(name).is_some() => {
                        return Ok(ToolOutput::text(
                            format!(
                                "Error: {}. Tell the user; they need to fix that file. Do not \
                                 delegate a procedure you invented in its place.",
                                installed.unavailable(name)
                            ),
                            true,
                        ));
                    }
                    None => {
                        let available: Vec<&str> = installed
                            .skills
                            .iter()
                            .map(|skill| skill.name.as_str())
                            .collect();
                        let hint = if available.is_empty() {
                            "No skills are installed.".to_string()
                        } else {
                            format!("Available skills: {}", available.join(", "))
                        };
                        return Err(MekaError::ToolExecution {
                            tool_name: "agent_spawn".to_string(),
                            message: format!("skill '{}' not found. {}", name, hint),
                        });
                    }
                }
            }
            None => None,
        };

        // `inherit_scratchpad`: optional array of parent-scratchpad names.
        let inherited_scratchpad = string_array(&input, "inherit_scratchpad");

        // Resolve the sub-agent's permission: an optional `permission` param clamped to the
        // parent's level as a ceiling (restrict-only, never escalate); absent keeps the parent's
        // level. Ask-mode prompts route through `PermissionForwardingFrontend` so they surface in
        // the parent's UI.
        let requested_permission = optional_str(&input, "permission", "agent_spawn")?;
        let sub_perm =
            resolve_subagent_permission(requested_permission, self.parent_permission.get())?;

        // Union, never replace: the call site adds to what config (and, when nested, this agent's
        // own parent) already denied. There is deliberately no allow-list parameter, because one
        // would let a parent hand a worker something the installation took away.
        let call_site_denials = ToolDenials::new(
            string_array(&input, "deny_servers"),
            string_array(&input, "deny_tools"),
        );
        // A name that matches nothing denies nothing, and the model gets no signal either way: it
        // asked for a sandboxed worker and would receive an unsandboxed one believing otherwise.
        // Warned rather than refused, because "deny it if it is there" is a legitimate thing to
        // write against a server list that varies by machine.
        if let Some(weak) = self.tool_builder_params.mcp_manager.as_ref()
            && let Some(manager) = weak.upgrade()
        {
            let configured = manager.server_names();
            for name in call_site_denials.server_list() {
                if !configured.contains(&name) {
                    tracing::warn!(
                        "agent_spawn deny_servers entry '{}' matches no configured MCP server, so \
                         it denies nothing",
                        name
                    );
                }
            }
        }
        let effective_denials = self.inherited_denials.union(&call_site_denials);

        // Context grants. Both default to nothing and are clamped against what this agent itself
        // holds, so a worker can never hand a grandchild more than it was given. Unlike the deny
        // lists, these are grants rather than restrictions: config cannot meaningfully withhold
        // them (a parent holding the text can copy it into the prompt), so the decision is the
        // parent's, and the safe state is the default.
        let memory_access = match optional_str(&input, "memory", "agent_spawn")? {
            Some(text) => {
                let requested = MemoryAccess::parse_grant(text).map_err(|message| {
                    MekaError::ToolExecution {
                        tool_name: "agent_spawn".to_string(),
                        message,
                    }
                })?;
                requested.min(self.tool_builder_params.memory_access)
            }
            None => MemoryAccess::None,
        };
        // The clamp for instructions is the text itself: a worker spawned without them has `None`
        // here (see `build_subagent`), so asking to pass them on is silently a no-op rather than a
        // hole. An installation with no instructions file behaves the same way.
        let parent_has_instructions = self
            .tool_builder_params
            .parent_options
            .user_instructions
            .as_deref()
            .is_some_and(|text| !text.trim().is_empty());
        let instructions = match optional_str(&input, "instructions", "agent_spawn")? {
            Some(text) => {
                let requested = InstructionAccess::parse_grant(text).map_err(|message| {
                    MekaError::ToolExecution {
                        tool_name: "agent_spawn".to_string(),
                        message,
                    }
                })?;
                if parent_has_instructions {
                    requested
                } else {
                    InstructionAccess::None
                }
            }
            None => InstructionAccess::None,
        };

        // Optional `max_depth`: the caller's override for how deep this sub-agent's own subtree may
        // recurse. Consumed by `child_spawn_depth` when deciding whether to grant a nested
        // `agent_spawn` below.
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
                tool_name: "agent_spawn".to_string(),
                message: "parent session ID not yet assigned (run_turn invariant)".to_string(),
            })?;

        // Read the skill body before any row is written, for the same reason the name was resolved
        // up front: an unreadable or oversized skill file fails here, and failing after
        // `create_child_session` would leave a childless session row that `agent_list` then
        // advertises as a worker you can follow up on.
        //
        // `load_skill_body` prepends the base-directory header so the skill's relative references
        // resolve against the skill rather than the sub-agent's working directory.
        let skill_body = match &skill {
            Some(skill) => {
                let body = crate::skills::load_skill_body(skill)
                    .await
                    .map_err(|error| MekaError::ToolExecution {
                        tool_name: "agent_spawn".to_string(),
                        message: format!("failed to load skill: {}", error),
                    })?;
                Some(body)
            }
            None => None,
        };

        // Compose the first-turn task: parent directive first, skill body second. The at-least-one
        // check above guarantees a `Some`, but it is resolved here, before the row exists, so that
        // every fallible step of a spawn happens while there is still nothing to clean up.
        let task =
            compose_subagent_task(prompt.as_deref(), skill_body.as_deref()).ok_or_else(|| {
                MekaError::ToolExecution {
                    tool_name: "agent_spawn".to_string(),
                    message: "agent_spawn requires 'prompt', 'skill', or both".to_string(),
                }
            })?;

        // Bound the sub-agent's own recursion budget before the spec is written:
        // `child_spawn_depth` turns this tool's counters plus the optional `max_depth`
        // override into the counters the *child* holds, and the spec has to carry those so
        // a followed-up worker can spawn exactly what it could have spawned on its first
        // turn.
        let (child_remaining_depth, child_absolute_depth, _allow_nested_spawn) = child_spawn_depth(
            self.remaining_depth,
            self.absolute_depth,
            max_depth_override,
        );

        // Snapshot the parent's cwd once, here, so a parent `/cd` mid-sub-agent execution can't
        // shift the sub-agent's path resolution mid-flight. The same value is written to the
        // child's session row, handed to its tool registry, and used to render its
        // environment context; a follow-up reads it back off the row.
        let sub_cwd_snapshot = crate::workspace::cwd_snapshot(&self.tool_builder_params.parent_cwd);
        let sub_cwd: crate::workspace::SharedCwd =
            Arc::new(std::sync::RwLock::new(sub_cwd_snapshot.clone()));

        let spec = SubagentSpec {
            permission: sub_perm,
            enabled_permissions: clamp_enabled_permissions(
                self.parent_permission.enabled(),
                sub_perm,
            )
            .iter()
            .collect(),
            denied_servers: effective_denials.server_list(),
            denied_tools: effective_denials.tool_list(),
            memory: memory_access,
            instructions,
            inherited_scratchpad: inherited_scratchpad.clone(),
            remaining_depth: child_remaining_depth,
            absolute_depth: child_absolute_depth,
        };
        let spec_json = serde_json::to_string(&spec).map_err(|error| MekaError::ToolExecution {
            tool_name: "agent_spawn".to_string(),
            message: format!("failed to encode sub-agent spec: {}", error),
        })?;

        // Create the sub-agent's own DB session, linked back to the parent via `parent_session_id`.
        // Cascade-on-delete in `delete_session` sweeps it when the parent is removed.
        let (sub_session_id, sub_session_lock) = self
            .tool_builder_params
            .session_manager
            .create_child_session(parent_sid, Some(sub_cwd_snapshot.clone()), Some(spec_json))
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "agent_spawn".to_string(),
                message: format!("failed to create sub-agent session: {}", error),
            })?;
        // Held for the whole of the worker's run, then released with this scope. A sub-agent's row
        // used to be locked by nothing at all, so a concurrent `meka session delete --all` could
        // take it and cascade the conversation away mid-turn. A failure to claim is a warning
        // rather than a refusal, matching the primary agent's own creation path: the id is one
        // nobody else can be holding, so the only way here is a filesystem problem, and refusing to
        // spawn over that would break installations that work today.
        let _sub_session_lock = match sub_session_lock {
            Ok(lock) => Some(lock),
            Err(error) => {
                tracing::warn!(
                    "sub-agent session {} is running unlocked: {}",
                    sub_session_id,
                    error
                );
                None
            }
        };
        tracing::info!(
            "spawning sub-agent {} for parent {}",
            sub_session_id,
            parent_sid
        );

        let sub_roots_snapshot =
            crate::workspace::roots_snapshot(&self.tool_builder_params.parent_roots);
        let environment_context =
            build_environment_context(sub_perm, &sub_cwd_snapshot, &sub_roots_snapshot);
        let augmented_prompt = format!("{}\n{}", environment_context, task);

        // The last step that can fail before the worker exists in its own right. Nothing here is
        // reachable in practice -- the web client is built from config the root already used, and a
        // fresh registry cannot collide -- but the row is already on disk, so a failure would leave
        // a childless session that `agent_list` advertises and `agent_followup` would resume into
        // an empty conversation. Roll it back rather than rely on the failure staying unreachable.
        //
        // Deliberately *not* extended to `run_turn` below: once the worker has started, a provider
        // error or a cancellation leaves a real conversation the parent may still want to read or
        // follow up on, and deleting that would discard work.
        let sub_agent = match build_subagent(
            &self.tool_builder_params,
            &self.provider,
            &spec,
            // The parent's live handle. `resolve_subagent_permission` has already clamped the
            // spec against it, so this re-derives the same starting level -- but it also stays
            // bound to the parent afterwards, which a snapshot could not do.
            &self.parent_permission,
            parent_sid,
            sub_session_id,
            sub_cwd,
            "agent_spawn",
        )
        .await
        {
            Ok(agent) => agent,
            Err(error) => {
                if let Err(cleanup) = self
                    .tool_builder_params
                    .session_manager
                    .delete_session(sub_session_id)
                    .await
                {
                    tracing::warn!(
                        "sub-agent {} could not be built and its session row could not be removed \
                         either: {}",
                        sub_session_id,
                        cleanup,
                    );
                }
                return Err(error);
            }
        };

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
        // Lead with the id so the report stays the tail of the output, where a model reading a long
        // result looks for the conclusion.
        //
        // Skipped when the caller redirected the report to a scratchpad. That redirect is universal
        // (`scratchpad::save_explicit_scratchpad_results` keys off the `scratchpad` argument, not
        // the tool) and stores the *whole* result text, so a header here would be written into the
        // entry and handed to whatever later reads it -- while the model, which now sees only a
        // reference, would not get the id anyway. A scratchpad holds output the parent means to
        // pass around; the id is metadata about the call, and `agent_list` is where to find it.
        let output = if input.get("scratchpad").is_some() {
            report
        } else {
            format!("agent: {}\n\n{}", sub_session_id, report)
        };
        Ok(ToolOutput::text(output, false))
    }
}

/// Register `agent_spawn` and the three lifecycle tools that operate on what it produced.
///
/// One function so the four always arrive together. `agent_followup` and `agent_delete` are useless
/// without `agent_spawn`, and `agent_spawn` without them is the one-shot worker this replaced: a
/// registry with three of the four is a shape nobody wants.
///
/// Filtering goes through `registry.admits`, which asks the registry *being written to* rather than
/// the tool being written. The distinction is the whole point: `AgentSpawnTool::inherited_denials`
/// is what this agent's future *children* are denied, and on the root agent that is the
/// `[subagents]` config. Reading it here would make `[subagents] disabled_tools = ["agent_spawn"]`
/// -- the natural way to write "workers may not spawn workers" -- delete `agent_spawn` from the
/// top-level agent and turn delegation off entirely.
///
/// Takes the already-built `AgentSpawnTool` because its depth counters differ between the root
/// (seeded from config) and a nested level (derived from the parent's).
pub fn register_subagent_tools(registry: &ToolRegistry, spawn: AgentSpawnTool) -> Result<()> {
    let params = spawn.tool_builder_params.clone();
    let provider = Arc::clone(&spawn.provider);
    // The permission of the agent this registry belongs to. `AgentSpawnTool` clamps new workers
    // against it; `AgentFollowupTool` clamps rehydrated ones against it too.
    let spawn_permission = spawn.parent_permission.clone();
    // One shared map per registry, so two parallel `agent_followup` calls on the same worker see
    // each other. A map per tool would make the guard a no-op.
    let in_flight: InFlightFollowups =
        Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));
    let delete_in_flight = Arc::clone(&in_flight);

    let mut tools: Vec<(&str, Arc<dyn Tool>)> = vec![("agent_spawn", Arc::new(spawn))];
    tools.push((
        "agent_list",
        Arc::new(AgentListTool {
            tool_builder_params: params.clone(),
        }),
    ));
    tools.push((
        "agent_followup",
        Arc::new(AgentFollowupTool {
            provider,
            parent_permission: spawn_permission,
            tool_builder_params: params.clone(),
            in_flight,
        }),
    ));
    tools.push((
        "agent_delete",
        Arc::new(AgentDeleteTool {
            tool_builder_params: params,
            in_flight: delete_in_flight,
        }),
    ));

    // All four or none, gated on `agent_spawn`. The three lifecycle tools only ever operate on what
    // `agent_spawn` produced, so an agent that cannot delegate has nothing for them to act on --
    // and `disabled_tools = ["agent_spawn"]` means "no delegation", not "no *new* delegation while
    // keeping the ability to drive workers a previous run left behind".
    if !registry.admits("agent_spawn") {
        return Ok(());
    }
    for (name, tool) in tools {
        if !registry.admits(name) {
            continue;
        }
        registry.register(tool)?;
    }
    Ok(())
}

/// Parse the `agent` argument, without checking ownership.
///
/// Split out so a caller can claim the per-worker guard *before* verifying ownership: verifying
/// first leaves an await boundary between the check and the claim, which is exactly long enough for
/// a concurrent `agent_delete` to remove the row the caller just validated.
fn parse_agent_id(input: &serde_json::Value, tool_name: &'static str) -> Result<Uuid> {
    let raw = super::util::require_str(input, "agent", tool_name)?;
    Uuid::parse_str(raw.trim()).map_err(|_| MekaError::ToolExecution {
        tool_name: tool_name.to_string(),
        message: format!("'{}' is not a valid agent id", raw),
    })
}

/// Refuse an `agent` argument that isn't a live child of the session running the tool.
///
/// One check, three failures it has to catch: an id that was never a sub-agent (a fabricated or
/// mistyped UUID), a sub-agent belonging to a *different* parent, and a fork holding ids it does
/// not own (`fork_session` copies the conversation, which names the children, but not the children
/// themselves, so a forked parent's log advertises sessions that are still linked to the original).
/// Letting any of those through would let one session drive or delete another's workers.
async fn require_child_session(
    params: &ToolBuilderParams,
    tool_name: &'static str,
    input: &serde_json::Value,
) -> Result<(Uuid, crate::session::SessionMetaRow)> {
    let agent_id = parse_agent_id(input, tool_name)?;
    let parent_sid = current_session_id(params, tool_name).await?;
    let children = params
        .session_manager
        .load_session_tree(parent_sid)
        .await
        .map_err(|error| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!("failed to list sub-agents: {}", error),
        })?;
    children
        .into_iter()
        .find(|row| row.id == agent_id && row.parent_id == Some(parent_sid))
        .map(|row| (parent_sid, row))
        .ok_or_else(|| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!(
                "no sub-agent '{}' belongs to this session. Use `agent_list` to see the ones that \
                 do.",
                agent_id
            ),
        })
}

/// The session id of the agent running the tool. By the time a tool runs, `Agent::run_turn` has
/// written `shared_session_id`; a missing value means an agent ran a tool without first creating
/// its session, an internal invariant break worth surfacing rather than papering over.
async fn current_session_id(params: &ToolBuilderParams, tool_name: &'static str) -> Result<Uuid> {
    params
        .parent_shared_session_id
        .read()
        .await
        .ok_or_else(|| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: "session ID not yet assigned (run_turn invariant)".to_string(),
        })
}

/// Lists the sub-agents this session has spawned and can still follow up on.
pub struct AgentListTool {
    pub tool_builder_params: ToolBuilderParams,
}

#[async_trait]
impl Tool for AgentListTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "agent_list".to_string(),
            description: "List the sub-agents you have spawned in this session, with each one's \
                          id, working directory, turn count, and last activity. Pass an id to \
                          `agent_followup` to ask it another question, or to `agent_delete` to \
                          discard it and free what it held."
                .to_string(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
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
        let session_id = current_session_id(&self.tool_builder_params, "agent_list").await?;
        let rows = self
            .tool_builder_params
            .session_manager
            .load_session_tree(session_id)
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "agent_list".to_string(),
                message: format!("failed to list sub-agents: {}", error),
            })?;

        // Direct children only. Grandchildren belong to the worker that spawned them and are
        // reachable through *its* `agent_list`; listing them here would advertise ids this session
        // cannot follow up on.
        let mut lines = Vec::new();
        for row in rows.iter().filter(|row| row.parent_id == Some(session_id)) {
            // Tool results are persisted as user-role messages too (`Agent::run_turn` wraps them in
            // one), so counting every user message would report a single task that took four tool
            // rounds as five turns. A real turn is a user message that carries something other than
            // tool results.
            let turns = self
                .tool_builder_params
                .session_manager
                .load_events(row.id)
                .await
                .map(|events| {
                    events
                        .iter()
                        .filter(|event| match event {
                            crate::conversation::Event::Append(message) => {
                                message.role == crate::provider::Role::User
                                    && !message.content.iter().all(|block| {
                                        matches!(
                                            block,
                                            crate::provider::ContentBlock::ToolResult { .. }
                                        )
                                    })
                            }
                            _ => false,
                        })
                        .count()
                })
                .unwrap_or(0);
            lines.push(format!(
                "{}\t{}\tturns={}\tlast_active={}",
                row.id,
                row.cwd.as_deref().unwrap_or(""),
                turns,
                row.updated_at,
            ));
        }

        if lines.is_empty() {
            return Ok(ToolOutput::text(
                "(no sub-agents spawned in this session)".to_string(),
                false,
            ));
        }
        Ok(ToolOutput::text(lines.join("\n"), false))
    }
}

/// Asks a sub-agent another question, on top of everything it already did.
pub struct AgentFollowupTool {
    pub provider: Arc<dyn Provider>,
    /// The level of the agent holding this tool, read live at each call. A worker never runs above
    /// it, so switching the session down to `read` reaches workers spawned while it was at
    /// `unrestricted`.
    pub parent_permission: SharedPermission,
    pub tool_builder_params: ToolBuilderParams,
    /// Sub-agents currently running a follow-up, keyed by session id. Shared with every other
    /// `agent_followup` on this registry.
    pub in_flight: InFlightFollowups,
}

/// Sub-agent sessions with a follow-up in progress.
///
/// Parallel tool calls in one turn run concurrently, so two follow-ups on the same worker would
/// interleave: both hydrate the same event log, both append to it, and the second overwrites the
/// first's view of what happened. A `std::sync::Mutex` over a set is enough because every critical
/// section is a set insert or removal, never an await.
pub type InFlightFollowups = Arc<std::sync::Mutex<std::collections::HashSet<Uuid>>>;

/// Claims a sub-agent for the duration of one follow-up, releasing it on drop so an error or a
/// cancellation mid-turn cannot leave the worker permanently marked busy.
struct FollowupGuard {
    in_flight: InFlightFollowups,
    agent_id: Uuid,
}

impl FollowupGuard {
    fn claim(in_flight: &InFlightFollowups, agent_id: Uuid) -> Option<Self> {
        let mut guard = in_flight
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !guard.insert(agent_id) {
            return None;
        }
        drop(guard);
        Some(Self {
            in_flight: Arc::clone(in_flight),
            agent_id,
        })
    }
}

impl Drop for FollowupGuard {
    fn drop(&mut self) {
        self.in_flight
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&self.agent_id);
    }
}

/// The spec a follow-up actually runs under: the recorded grant, narrowed by whatever the operator
/// has denied since.
///
/// Named rather than inlined at the call site because the narrowing is the security property and it
/// was not observable there. The stored spec is deliberately left alone, so the only assertion a
/// test could make against the call site was that the *recording* had not changed -- which is true
/// whether or not the narrowing happened. A mutation dropping `denied_servers` from the expression
/// therefore survived the whole suite, and would have let a worker keep reaching an MCP server the
/// operator had since denied.
///
/// The spec is a floor on restriction, never a licence: config can only narrow it, and `..spec`
/// carries everything config has no opinion about.
fn combined_for_followup(
    spec: SubagentSpec,
    config_denials: &ToolDenials,
    memory_access: MemoryAccess,
) -> SubagentSpec {
    SubagentSpec {
        denied_servers: spec.denials().union(config_denials).server_list(),
        denied_tools: spec.denials().union(config_denials).tool_list(),
        memory: spec.memory.min(memory_access),
        ..spec
    }
}

#[async_trait]
impl Tool for AgentFollowupTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "agent_followup".to_string(),
            description: "Ask a sub-agent you already spawned another question. It keeps its own \
                          conversation, so it still remembers what it found and can build on it \
                          rather than starting over from a summary. Returns its new report. The \
                          sub-agent runs under the terms it was spawned with (same permission \
                          level, same restrictions), which your current settings cannot widen. \
                          Get ids from `agent_spawn`'s result or from `agent_list`."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "description": "The sub-agent's id, as returned by `agent_spawn` or \
                                        `agent_list`."
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The follow-up question or task."
                    },
                    "scratchpad": {
                        "type": "string",
                        "description": "If provided, save the sub-agent's new report to your \
                                        scratchpad under this name instead of returning it inline."
                    }
                },
                "required": ["agent", "prompt"]
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
        let prompt = super::util::require_str(&input, "prompt", "agent_followup")?;
        // Claimed before the ownership check, not after: the check awaits, and a concurrent
        // `agent_delete` finishing inside that window would leave this turn writing to a session
        // that no longer exists.
        let agent_id = parse_agent_id(&input, "agent_followup")?;
        let Some(_guard) = FollowupGuard::claim(&self.in_flight, agent_id) else {
            return Err(MekaError::ToolExecution {
                tool_name: "agent_followup".to_string(),
                message: format!(
                    "sub-agent '{}' is busy. Wait for the call already running against it to \
                     return before asking again.",
                    agent_id
                ),
            });
        };
        let (parent_sid, row) =
            require_child_session(&self.tool_builder_params, "agent_followup", &input).await?;
        debug_assert_eq!(row.id, agent_id);

        // The terms come off the session row, never from this agent's current state. See
        // `SubagentSpec`.
        let spec_json = self
            .tool_builder_params
            .session_manager
            .load_subagent_spec(agent_id)
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "agent_followup".to_string(),
                message: format!("failed to load sub-agent spec: {}", error),
            })?
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "agent_followup".to_string(),
                message: format!(
                    "sub-agent '{}' has no recorded spawn terms (it predates follow-up support), \
                     so it cannot be resumed safely. Spawn a new one.",
                    agent_id
                ),
            })?;
        let spec: SubagentSpec =
            serde_json::from_str(&spec_json).map_err(|error| MekaError::ToolExecution {
                tool_name: "agent_followup".to_string(),
                message: format!("sub-agent '{}' has an unreadable spec: {}", agent_id, error),
            })?;
        // The spec records what the worker was granted; it is not a licence to ignore what applies
        // now. Config may have gained deny lists since the spawn -- most likely across the restart
        // an old worker had to survive to be here -- and the memory grant is re-clamped against
        // what *this* agent currently holds, so a worker cannot outlive its granter's own limits.
        // Both combines take the more restrictive side, so neither can widen the worker.
        let spec = combined_for_followup(
            spec,
            &self.tool_builder_params.config_denials,
            self.tool_builder_params.memory_access,
        );

        // The worker's own cwd, as recorded when it was spawned, not the parent's current one: a
        // `/cd` between the spawn and the follow-up must not move a worker mid-task.
        let sub_cwd: crate::workspace::SharedCwd = Arc::new(std::sync::RwLock::new(
            row.cwd
                .as_deref()
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| {
                    crate::workspace::cwd_snapshot(&self.tool_builder_params.parent_cwd)
                }),
        ));
        let sub_cwd_snapshot = crate::workspace::cwd_snapshot(&sub_cwd);

        let ceiling = self.parent_permission.get();
        let effective_permission = spec.effective_permission(ceiling);
        // `!=`, not `<`. The derived `Ord` is display order, which the enum doc says must not
        // decide authority, and it cannot see a sideways move at all: a `workspace` spec resolving
        // to `ask` compares as *greater* and reported nothing, even though the worker changed
        // level. Any difference from what was recorded is worth saying out loud.
        if effective_permission != spec.permission {
            tracing::info!(
                "sub-agent {} runs at {} rather than its recorded {}: this session has since been \
                 restricted",
                agent_id,
                effective_permission,
                spec.permission,
            );
        }

        let sub_agent = build_subagent(
            &self.tool_builder_params,
            &self.provider,
            &spec,
            &self.parent_permission,
            parent_sid,
            agent_id,
            Arc::clone(&sub_cwd),
            "agent_followup",
        )
        .await?;

        // Rehydrate the worker's own conversation: the same three calls the REPL's resume path
        // makes. `from_events` arms the resume notice, and it is left armed deliberately -- every
        // follow-up really is a fresh registry, a fresh read tracker and an empty todo list, so the
        // worker is being told something true each time rather than a stale banner.
        let events = self
            .tool_builder_params
            .session_manager
            .load_events(agent_id)
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "agent_followup".to_string(),
                message: format!("failed to load sub-agent conversation: {}", error),
            })?;
        let mut messages = Conversation::from_events(events);
        for dropped in messages.sanitize_orphans() {
            tracing::warn!(
                "sub-agent {}: dropped an assistant message with orphaned tool_use blocks while \
                 rehydrating ({} blocks)",
                agent_id,
                dropped.content.len(),
            );
        }

        let roots_snapshot =
            crate::workspace::roots_snapshot(&self.tool_builder_params.parent_roots);
        let environment_context =
            build_environment_context(effective_permission, &sub_cwd_snapshot, &roots_snapshot);
        let augmented_prompt = format!("{}\n{}", environment_context, prompt);

        // Held for this turn, as `agent_spawn` holds it for the spawn. A follow-up runs a full turn
        // against a row nothing else claims, so without this the worker sat unlocked for seconds to
        // minutes and a concurrent `meka session delete --all` could take it and cascade the
        // conversation away mid-run -- the same exposure `create_child_session` closed for spawn,
        // still open on the sibling door.
        //
        // A refusal here, where spawn only warns, and the asymmetry is the point: spawn's id is
        // brand new, so a failure can only be a filesystem problem, while this id already exists
        // and a refusal genuinely means somebody else is running a turn on this worker. Two turns
        // interleaved into one conversation is the thing the lock is for.
        let _worker_lock = self
            .tool_builder_params
            .session_manager
            .lock_session(agent_id)
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "agent_followup".to_string(),
                message: format!("cannot follow up on sub-agent {}: {}", agent_id, error),
            })?;

        let mut session_id_opt = Some(agent_id);
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

/// Discards a sub-agent and everything it accumulated.
pub struct AgentDeleteTool {
    pub tool_builder_params: ToolBuilderParams,
    /// Shared with this registry's `agent_followup`, so the two cannot run on one worker at once.
    pub in_flight: InFlightFollowups,
}

#[async_trait]
impl Tool for AgentDeleteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "agent_delete".to_string(),
            description: "Delete a sub-agent you spawned, discarding its conversation, its \
                          scratchpad entries, and any sub-agents it spawned in turn. Use this once \
                          you have what you needed from a worker, so a long session doesn't carry \
                          every worker it ever ran. This removes only meka's own record of that \
                          sub-agent and its descendants: files it wrote to disk, and your own \
                          conversation and scratchpad, are untouched."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "description": "The sub-agent's id, as returned by `agent_spawn` or \
                                        `agent_list`."
                    }
                },
                "required": ["agent"]
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
        // Claimed before the ownership check, mirroring `agent_followup`, so the two serialise on
        // one worker whichever arrives first. Parallel tool calls in a single turn make this
        // reachable: without it, a delete completing inside a follow-up's ownership check leaves
        // that follow-up writing to a deleted session and failing on a foreign-key violation --
        // a raw database error, on a path where the model did nothing wrong.
        let agent_id = parse_agent_id(&input, "agent_delete")?;
        let Some(_guard) = FollowupGuard::claim(&self.in_flight, agent_id) else {
            return Err(MekaError::ToolExecution {
                tool_name: "agent_delete".to_string(),
                message: format!(
                    "sub-agent '{}' is busy. Wait for the call already running against it to \
                     return before deleting it.",
                    agent_id
                ),
            });
        };
        let (_parent_sid, row) =
            require_child_session(&self.tool_builder_params, "agent_delete", &input).await?;
        // One statement; `sessions.parent_session_id`, `messages.session_id` and
        // `tool_outputs.session_id` all carry `ON DELETE CASCADE`, so the worker's messages, its
        // scratchpad entries and its own descendants go with it.
        self.tool_builder_params
            .session_manager
            .delete_session(row.id)
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "agent_delete".to_string(),
                message: format!("failed to delete sub-agent: {}", error),
            })?;
        tracing::info!("deleted sub-agent session {}", row.id);
        Ok(ToolOutput::text(format!("deleted agent {}", row.id), false))
    }
}

/// Build the worker described by `spec`: its tool registry, its inherited MCP toolset, its own
/// `agent_spawn` when the recursion budget allows, its system prompt, and the `Agent` over all of
/// it.
///
/// Shared by `agent_spawn` and `agent_followup` on purpose. The two differ only in where the spec
/// comes from (freshly built vs. read off the session row) and what conversation the agent is
/// handed (empty vs. rehydrated). Anything that drifted between two copies of this would be a
/// follow-up that quietly runs under different terms than the spawn did, which is the exact failure
/// the spec exists to prevent.
///
/// `params` supplies the *ambient* collaborators (provider, caches, session manager, frontend) and
/// `spec` supplies every restriction. `params.memory_access` is deliberately not read here: the
/// spec's copy is authoritative, because config may have changed since the spawn.
#[allow(clippy::too_many_arguments)]
async fn build_subagent(
    params: &ToolBuilderParams,
    provider: &Arc<dyn Provider>,
    spec: &SubagentSpec,
    // The parent's live handle, not a snapshot of its level. Taking a `Permission` here is what
    // let a worker outlive its parent's downgrade: the value was read once and frozen into a fresh
    // atomic, so nothing the user did afterwards could reach the running child.
    parent_permission: &SharedPermission,
    parent_session_id: Uuid,
    sub_session_id: Uuid,
    sub_cwd: crate::workspace::SharedCwd,
    tool_name: &'static str,
) -> Result<Agent> {
    let sub_shared_perm = spec.shared_permission_bounded(parent_permission);
    let effective_permission = spec.effective_permission(parent_permission.get());
    let denials = spec.denials();
    // Resolved once: it feeds both this worker's system prompt and what its own children can be
    // given. `None` here is what makes nesting self-enforcing -- a worker that was not granted the
    // instructions has no copy to pass on, so the restriction propagates through the data rather
    // than through a check every future call site has to remember.
    let memory_access = spec.granted_memory();
    let granted_instructions = match spec.instructions {
        InstructionAccess::Inherit => params.parent_options.user_instructions.clone(),
        InstructionAccess::None => None,
    };
    // A fresh, private todo list so the worker's `todo` calls don't touch the parent's task
    // tracking. Not persisted, so a follow-up starts with an empty one; the resume notice the
    // rehydrated conversation carries is what tells the worker its tool state is gone.
    let sub_todo_list: super::todo::SharedTodoList =
        Arc::new(tokio::sync::RwLock::new(super::todo::TodoState::default()));
    let sub_shared_session_id: Arc<RwLock<Option<Uuid>>> =
        Arc::new(RwLock::new(Some(sub_session_id)));

    let sub_registry = ToolRegistry::build_for_subagent(
        params.web_client.clone(),
        sub_shared_perm.clone(),
        params.sandbox_enabled,
        params.sandbox_capability.clone(),
        params.sandbox_backend,
        params.backend_probe.clone(),
        params.builtin_filter.clone(),
        denials.clone(),
        sub_todo_list.clone(),
        params.session_manager.clone(),
        sub_shared_session_id.clone(),
        params.skills.clone(),
        params.memories.clone(),
        memory_access,
        if spec.inherited_scratchpad.is_empty() {
            None
        } else {
            Some(parent_session_id)
        },
        spec.inherited_scratchpad.clone(),
        sub_cwd.clone(),
        Arc::clone(&params.parent_roots),
        Arc::clone(&params.parent_frontend),
    )
    .map_err(|error| MekaError::ToolExecution {
        tool_name: tool_name.to_string(),
        message: format!("failed to build sub-agent tool registry: {}", error),
    })?;

    // Inherit the parent's MCP toolset, minus anything the spec denies (`install_tools_on` reads
    // the denials back off the registry). Skipped silently when no MCP manager is attached (no
    // servers configured) or when the parent's servers are still Pending / Failed. Non-spawning and
    // idempotent; see `src/mcp.rs:install_tools_on`.
    if let Some(weak) = params.mcp_manager.as_ref() {
        // Upgrade only if the manager is still alive. If the parent's `meka acp` process is
        // mid-shutdown, the Arc may already be gone. Skip silently.
        if let Some(manager) = weak.upgrade() {
            manager.install_tools_on(&sub_registry).await;
        }
    }

    // Grant the sub-agent its own `agent_spawn` when its recursion budget allows, so it can
    // orchestrate a team of its own. Registered here (before the tool catalogue is snapshotted for
    // the system prompt below) and outside `build_for_subagent`, mirroring the root registration in
    // `assemble_agent` (main.rs). Two counters bound nesting: `remaining_depth` is the soft,
    // `max_depth`-tunable budget; `absolute_depth` is the hard cap that guarantees termination.
    // The three lifecycle tools ride the same gate: a worker that cannot spawn has no children to
    // list, follow up on, or delete.
    let allow_nested_spawn =
        spec.remaining_depth >= 1 && spec.absolute_depth < SUBAGENT_ABSOLUTE_MAX_DEPTH;
    if allow_nested_spawn {
        let child_params = ToolBuilderParams {
            parent_shared_session_id: sub_shared_session_id.clone(),
            parent_cwd: Arc::clone(&sub_cwd),
            parent_roots: Arc::clone(&params.parent_roots),
            // The worker's own granted level, not its parent's. `params.memory_access` is what the
            // *spawning* agent holds, so letting the spread supply it would let a worker grant its
            // children up to its parent's level rather than its own -- reaching through a child
            // what it was denied directly. The deny lists come from `spec` for the same reason.
            memory_access,
            // Both ceilings for whatever this worker spawns in turn. `parent_options` carries the
            // instruction text, so overwriting it with the grant is what stops a worker handing a
            // grandchild something it was not given itself.
            parent_options: AgentOptions {
                user_instructions: granted_instructions.clone(),
                ..params.parent_options.clone()
            },
            ..params.clone()
        };
        register_subagent_tools(&sub_registry, AgentSpawnTool {
            provider: Arc::clone(provider),
            // The worker's own (already clamped) permission is the ceiling for anything it spawns,
            // so a downgrade the parent made reaches the whole subtree.
            parent_permission: sub_shared_perm.clone(),
            tool_builder_params: child_params,
            inherited_denials: denials,
            remaining_depth: spec.remaining_depth,
            absolute_depth: spec.absolute_depth,
        })?;
    }

    // Build the system prompt against the fully-loaded registry (which now includes MCP adapters).
    // The override on `AgentOptions` is static, so this single build captures the whole catalogue
    // the sub-agent can see.
    let tools = sub_registry.definitions_for_permission(effective_permission);
    // Gated on the registry rather than on the spec alone: `[memory] enabled = false` or a
    // `[tools]` filter can leave a granted worker without `memory_read`, and an index describing
    // memories it has no tool to open is pure cost. This mirrors how the primary agent's index is
    // gated on the same tool being in its catalogue.
    let memory_index = if sub_registry.get("memory_read").is_some() {
        render_subagent_memory_index(&params.memories.index().await.unwrap_or_else(|error| {
            // Search still works, and the worker is told what it has. A store that cannot be read
            // is not a reason to refuse to spawn.
            tracing::warn!("could not read the memory index for a sub-agent: {}", error);
            Vec::new()
        }))
    } else {
        String::new()
    };
    let sub_system_prompt = build_subagent_system_prompt(
        effective_permission,
        &tools,
        &spec.inherited_scratchpad,
        granted_instructions.as_deref(),
        &memory_index,
    );

    // Wrap so permission prompts surface in the parent's UI while emits stay silent (the
    // sub-agent's output flows back as this tool's result, not as live notifications). The one
    // exception is the sub-agent's tool calls, which are rolled up into this call's own display
    // so a long run isn't an opaque spinner -- hence the tool-call id.
    let sub_frontend: Arc<dyn crate::frontend::Frontend> =
        Arc::new(crate::frontend::PermissionForwardingFrontend::new(
            Arc::clone(&params.parent_frontend),
            crate::tools::current_tool_call_id(),
        ));

    Ok(Agent::new_subagent(
        Arc::clone(provider),
        sub_registry,
        params.session_manager.clone(),
        sub_shared_perm,
        &params.parent_options,
        sub_system_prompt,
        sub_todo_list,
        sub_shared_session_id,
        params.skills.clone(),
        params.memories.clone(),
        &sub_cwd,
        &params.parent_roots,
        sub_frontend,
        params.session_stats.clone(),
    ))
}

/// Read an optional string-valued parameter, refusing a value of the wrong type.
///
/// The obvious `input[key].as_str()` treats a non-string as absent, which for a restriction is the
/// wrong way to fail: `permission: 0` would silently run the worker at the parent's own level
/// rather than the restricted one the caller was reaching for. Absent and explicitly-null still
/// mean "not specified"; anything else that is not a string is an error naming the parameter.
fn optional_str<'a>(
    input: &'a serde_json::Value,
    key: &str,
    tool_name: &'static str,
) -> Result<Option<&'a str>> {
    match input.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(text)) => {
            let trimmed = text.trim();
            Ok(if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            })
        }
        Some(other) => Err(MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!(
                "'{}' must be a string, got {}. Leave it out to use the default.",
                key, other
            ),
        }),
    }
}

/// Pull a `Vec<String>` out of an optional array parameter. Non-string entries are silently
/// skipped, so a partially malformed array doesn't tank the whole spawn; a missing or non-array
/// value yields an empty list.
fn string_array(input: &serde_json::Value, key: &str) -> Vec<String> {
    input
        .get(key)
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::trim))
                .filter(|text| !text.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
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
                        tool_name: "agent_spawn".to_string(),
                        message,
                    })?;
            // `clamp_to`, not `min`. The ladder is a partial order: `Workspace` and `Ask` are
            // incomparable, and `min` over the discriminants hands a child at `workspace` to a
            // parent at `ask` because 2 < 3. That is not merely a mislabel. `SharedPermission::
            // with_ceiling` flattens a grandchild's ceiling to the *root* cell, on the stated
            // precondition that this clamp already folded the intermediate level in; with `min`
            // that precondition is false, so an `ask` child spawning a `workspace` grandchild under
            // an `unrestricted` root produced a worker that wrote unattended beneath a parent whose
            // whole safety was the approval prompt.
            Ok(requested.clamp_to(parent))
        }
        None => Ok(parent),
    }
}

/// Restrict an `EnabledPermissions` set to the modes *contained by* `ceiling`.
///
/// Not "at or below", which is what this used to say and is a different question. The ladder is a
/// partial order and the predicate is [`Permission::is_within`], so `workspace` and `ask` exclude
/// each other in both directions: neither is within the other, and an `ask` ceiling therefore
/// drops `workspace` from the set rather than keeping it as something lower down. Reading the doc
/// as a `<=` over the discriminants would predict the opposite for exactly the pair this release
/// introduced.
///
/// Defense-in-depth for the permission clamp: sub-agents have no runtime permission-switch path
/// today, so their initial level is what governs, but bounding the enabled set means any future
/// switch path cannot climb a sub-agent back past the ceiling its parent set. Falls back to a
/// singleton `{ceiling}` set when the intersection is empty, which the incomparable pair makes
/// reachable rather than theoretical: a parent that enabled only `workspace` and hands down an
/// `ask` ceiling leaves nothing behind.
fn clamp_enabled_permissions(
    enabled: EnabledPermissions,
    ceiling: Permission,
) -> EnabledPermissions {
    EnabledPermissions::from_modes(enabled.iter().filter(|mode| mode.is_within(ceiling)))
        .unwrap_or_else(|| {
            EnabledPermissions::from_modes([ceiling]).unwrap_or(EnabledPermissions::DEFAULT)
        })
}

/// Compute the recursion budget for a sub-agent one level below a `AgentSpawnTool` with the given
/// `remaining_depth` / `absolute_depth`, honoring an optional `max_depth` override.
///
/// Returns `(child_remaining, child_absolute, allow_nested)`. `remaining_depth` is the budget
/// seeded from `session.subagent_max_depth`; `max_depth` may lower it but never raise it.
/// `absolute_depth` is the monotonic hard counter: it always increments and, once it reaches
/// [`SUBAGENT_ABSOLUTE_MAX_DEPTH`], no further `agent_spawn` is granted. A nested `agent_spawn` is
/// granted only when both budgets allow it.
///
/// The override is clamped because `[session] subagent_max_depth` is documented as a ceiling
/// ("`subagent_max_depth = 1` means sub-agents cannot spawn further sub-agents"), and an
/// unclamped override made it merely a default: one `agent_spawn` passing `max_depth: 15` handed a
/// worker a budget the operator had explicitly denied, and each level could re-grant it. Recursion
/// was still bounded by [`SUBAGENT_ABSOLUTE_MAX_DEPTH`], so this is about the config key meaning
/// what it says rather than about termination.
fn child_spawn_depth(
    remaining_depth: usize,
    absolute_depth: usize,
    max_depth_override: Option<usize>,
) -> (usize, usize, bool) {
    let inherited = remaining_depth.saturating_sub(1);
    let child_remaining = match max_depth_override {
        Some(requested) => requested.min(inherited),
        None => inherited,
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

/// Ceiling on the memory index handed to a granted worker. Smaller than the primary agent's 8 KiB
/// budget on purpose: a worker was spawned for one task, and the store is background for it rather
/// than the running context it is for the agent that owns the session.
const SUBAGENT_MEMORY_INDEX_MAX_BYTES: usize = 4_096;

/// Render the memory index for a worker granted `memory: "read"`.
///
/// Separate from the primary agent's `[Memory]` section rather than shared with it, for two
/// reasons. A sub-agent's system prompt is a static override, so it never receives the per-turn
/// world state the parent's index rides in. And the parent's header tells the reader to call
/// `memory_write` when it learns something durable, which a worker cannot do -- pointing it at a
/// tool it does not have is how a model burns a turn discovering the tool is missing.
fn render_subagent_memory_index(memories: &[crate::memory::Memory]) -> String {
    if memories.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "## Memory\n\nDurable notes the agent that spawned you has saved, most important first. \
         Call `memory_read` with a name to load one in full, or `memory_search` to search across \
         all of them. You cannot add to or change this store: if you learn something worth \
         keeping, say so in your report and let the agent that spawned you decide.\n\n",
    );
    let mut shown = 0;
    for memory in memories {
        // Sanitised at the boundary, like the primary agent's index: the store hands back stored
        // bytes, and this is a worker's context.
        //
        // Elided too, which the parent's index and both search renderers already did and this one
        // did not. Descriptions are unbounded at the write door, so a single 4,000-character one
        // at the top of the store exceeded the whole budget on its own.
        let line = format!(
            "- **{}**: {}\n",
            memory.name,
            crate::store::elide_description_for_index(
                &crate::memory::render_description_for_model(&memory.description)
            )
        );
        // Always emit the first, for the same reason `render_hits` does. The elide above is what
        // actually makes a collapse to zero entries unreachable -- `MAX_DESCRIPTION_CHARS` bounds
        // one line far below this budget -- so this branch is the belt to that brace, and holds if
        // that bound ever moves. It is deliberately not something the tests can distinguish.
        if shown > 0 && out.len() + line.len() > SUBAGENT_MEMORY_INDEX_MAX_BYTES {
            break;
        }
        out.push_str(&line);
        shown += 1;
    }
    // Same reason the parent states its remainder: a silently truncated index reads as "this is
    // everything", which is what turns a full store into a confidently incomplete answer.
    let remaining = memories.len() - shown;
    if remaining > 0 {
        out.push_str(&format!(
            "\n{} more not listed here. Use `memory_search` to reach them.\n",
            remaining
        ));
    }
    out.push('\n');
    out
}

/// The sub-agent's system prompt.
///
/// `user_instructions` is `None` unless the `agent_spawn` call asked for them, which is the reverse
/// of how this used to work. Instructions are installation-wide and describe the top-level agent:
/// its persona, how it should address the user, what it should volunteer. A worker handed a task by
/// another agent is not that agent, and inheriting the persona unasked is how a sub-agent ends up
/// talking to the user as though it were the one they are speaking to.
///
/// They remain *grantable* because they are also where project conventions live, and a parent that
/// judges a task needs the standing rules can hand them over verbatim rather than paraphrasing them
/// into the prompt.
fn build_subagent_system_prompt(
    permission: Permission,
    tools: &[ToolDefinition],
    inherited_scratchpad: &[String],
    user_instructions: Option<&str>,
    memory_index: &str,
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
            "These are installation-specific rules, handed to you by the agent that spawned you. \
             Treat them as hard constraints unless they conflict with safety requirements. They \
             describe that agent's own conduct, so where they concern how to address the user or \
             what to volunteer, they are context rather than instructions to you: your output goes \
             back to that agent as a report, not to the user.\n\n",
        );
        prompt.push_str(instructions);
        prompt.push_str("\n\n");
    }

    prompt.push_str(memory_index);

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
        let prompt = build_subagent_system_prompt(Permission::Unrestricted, &[], &[], None, "");
        assert!(
            prompt.contains(&format!(
                "## Permission Level: {}",
                Permission::Unrestricted
            )),
            "expected Write level in prompt, got: {}",
            prompt
        );

        let read_prompt = build_subagent_system_prompt(Permission::Read, &[], &[], None, "");
        assert!(read_prompt.contains(&format!("## Permission Level: {}", Permission::Read)));
    }

    /// The installation's instructions describe the top-level agent, not a worker one of its turns
    /// handed a task to. A sub-agent that inherited them would answer the user in the leader's
    /// voice, under rules written for a conversation it is not part of.
    #[test]
    fn test_subagent_system_prompt_carries_no_user_instructions() {
        let prompt = build_subagent_system_prompt(Permission::Unrestricted, &[], &[], None, "");
        assert!(!prompt.contains("User Instructions"));
        assert!(!prompt.contains("installation-specific"));
    }

    #[test]
    fn test_subagent_system_prompt_mentions_todo_tools() {
        let prompt = build_subagent_system_prompt(Permission::Read, &[], &[], None, "");
        assert!(
            prompt.contains("`todo` tool"),
            "expected todo tool mention in prompt, got: {}",
            prompt
        );
    }

    #[test]
    fn test_subagent_system_prompt_omits_inheritance_section_when_empty() {
        let prompt = build_subagent_system_prompt(Permission::Read, &[], &[], None, "");
        assert!(
            !prompt.contains("Inherited Scratchpad"),
            "no inherited section expected for empty allowlist, got: {}",
            prompt
        );
    }

    #[test]
    fn test_subagent_system_prompt_lists_inherited_names() {
        let names = vec!["captured_output".to_string(), "research_notes".to_string()];
        let prompt = build_subagent_system_prompt(Permission::Read, &[], &names, None, "");
        assert!(prompt.contains("## Inherited Scratchpad Entries"));
        assert!(prompt.contains("- captured_output"));
        assert!(prompt.contains("- research_notes"));
        assert!(prompt.contains("scratchpad_read"));
    }

    #[test]
    fn test_subagent_system_prompt_warns_inherited_writes_will_error() {
        let names = vec!["build_log".to_string()];
        let prompt = build_subagent_system_prompt(Permission::Read, &[], &names, None, "");
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
            resolve_subagent_permission(None, Permission::Unrestricted).unwrap(),
            Permission::Unrestricted
        );
        assert_eq!(
            resolve_subagent_permission(None, Permission::Read).unwrap(),
            Permission::Read
        );
    }

    /// `agent_followup` resolves a stored grant with the meet, not the spawn clamp.
    ///
    /// Reverting `effective_permission` to `clamp_to` passed every sub-agent test, so which of the
    /// two operations the replay path uses was unguarded. The difference is not academic: it is
    /// whether a worker that was deliberately confined to the workspace comes back able to write
    /// anywhere once the parent moves to `ask`.
    #[test]
    fn a_followup_resolves_a_stored_grant_without_widening_it() {
        let spec = SubagentSpec {
            permission: Permission::Workspace,
            enabled_permissions: Vec::new(),
            denied_servers: Vec::new(),
            denied_tools: Vec::new(),
            memory: MemoryAccess::None,
            instructions: InstructionAccess::None,
            inherited_scratchpad: Vec::new(),
            remaining_depth: 2,
            absolute_depth: 1,
        };

        assert_eq!(
            spec.effective_permission(Permission::Ask),
            Permission::Read,
            "a `workspace` worker replayed under an `ask` parent must not gain whole-filesystem \
             reach; `clamp_to` would have said `ask` here"
        );
        assert_eq!(
            spec.effective_permission(Permission::Unrestricted),
            Permission::Workspace,
            "a parent that still holds the recorded level replays it unchanged"
        );
        assert_eq!(
            spec.effective_permission(Permission::Read),
            Permission::Read,
            "and a parent that has dropped below it narrows the worker with it"
        );
    }

    /// The incomparable pair, at the depth where nothing downstream re-clamps it.
    ///
    /// `Workspace` and `Ask` are incomparable, and `min` picks `Workspace` because 2 < 3. At depth
    /// 1 that is invisible: `SharedPermission::get` re-clamps against the parent and returns `Ask`.
    /// At depth 2 it is not, because `with_ceiling` flattens a grandchild's ceiling to the *root*
    /// cell on the stated precondition that the spawn-time clamp already folded the intermediate
    /// level in. With `min` that precondition is false, so an `ask` child under an `unrestricted`
    /// root could spawn a `workspace` grandchild that wrote unattended -- beneath a parent whose
    /// entire safety was the approval prompt.
    ///
    /// The pre-existing ceiling test only exercised pairs the total order already gets right
    /// (`unrestricted`/`read`, `ask`/`read`), which is why this shipped.
    ///
    /// **What this does not cover**, and the honest limit of the guarantee: the root cell here
    /// never moves. `with_ceiling`'s precondition is taken once, at spawn, so cycling the root
    /// `unrestricted -> workspace -> unrestricted` between two spawns leaves a grandchild bound to
    /// a root that has changed shape underneath it, and it can sit at `workspace` under an `ask`
    /// parent. Nothing exceeds the *root*, which is the human's own level, so the headline
    /// invariant holds and the next `agent_followup` re-clamps it -- but the direct-parent bound
    /// does not hold across that sequence, and no test asserts it does.
    #[test]
    fn a_grandchild_cannot_escape_an_intermediate_ask_parent() {
        assert_eq!(
            resolve_subagent_permission(Some("workspace"), Permission::Ask).unwrap(),
            Permission::Ask,
            "a `workspace` request under an `ask` parent must resolve to the parent's own level"
        );
        assert_eq!(
            resolve_subagent_permission(Some("ask"), Permission::Workspace).unwrap(),
            Permission::Workspace,
            "and the same in the other direction"
        );

        // The full chain, through the handles production actually builds.
        fn spec_at(permission: Permission, parent_enabled: EnabledPermissions) -> SubagentSpec {
            SubagentSpec {
                permission,
                enabled_permissions: clamp_enabled_permissions(parent_enabled, permission)
                    .iter()
                    .collect(),
                denied_servers: Vec::new(),
                denied_tools: Vec::new(),
                memory: MemoryAccess::None,
                instructions: InstructionAccess::None,
                inherited_scratchpad: Vec::new(),
                remaining_depth: 2,
                absolute_depth: 1,
            }
        }

        let root = SharedPermission::new(Permission::Unrestricted, EnabledPermissions::ALL);
        let child_spec = spec_at(
            resolve_subagent_permission(Some("ask"), root.get()).unwrap(),
            root.enabled(),
        );
        let child = child_spec.shared_permission_bounded(&root);
        assert_eq!(
            child.get(),
            Permission::Ask,
            "child holds what it asked for"
        );

        let grandchild_spec = spec_at(
            resolve_subagent_permission(Some("workspace"), child.get()).unwrap(),
            child.enabled(),
        );
        let grandchild = grandchild_spec.shared_permission_bounded(&child);
        assert_eq!(
            grandchild.get(),
            Permission::Ask,
            "a grandchild must not reach `workspace` past an `ask` parent, however the ceiling \
             cell is flattened"
        );
        assert!(
            !grandchild.enabled().is_enabled(Permission::Workspace),
            "nor may `workspace` remain switchable in its enabled set"
        );
    }

    #[test]
    fn test_resolve_subagent_permission_clamps_to_parent_ceiling() {
        // Requesting a higher level than the parent is clamped down: a sub-agent can never be
        // escalated above its parent.
        assert_eq!(
            resolve_subagent_permission(Some("unrestricted"), Permission::Read).unwrap(),
            Permission::Read
        );
        assert_eq!(
            resolve_subagent_permission(Some("ask"), Permission::Read).unwrap(),
            Permission::Read
        );
        // Requesting a lower level restricts the sub-agent below the parent.
        assert_eq!(
            resolve_subagent_permission(Some("read"), Permission::Unrestricted).unwrap(),
            Permission::Read
        );
        assert_eq!(
            resolve_subagent_permission(Some("none"), Permission::Unrestricted).unwrap(),
            Permission::None
        );
    }

    #[test]
    fn test_resolve_subagent_permission_rejects_invalid() {
        assert!(resolve_subagent_permission(Some("admin"), Permission::Unrestricted).is_err());
    }

    /// A restriction passed with the wrong type is refused, not read as absent. Reading it as
    /// absent is the dangerous direction for `permission`, where "not specified" means "inherit the
    /// parent's level" -- so a malformed restriction would hand the worker *more* than the caller
    /// was reaching for.
    #[test]
    fn test_optional_str_refuses_a_wrong_typed_value() {
        let input = serde_json::json!({
            "permission": 0,
            "memory": true,
            "good": "read",
            "blank": "   ",
            "explicit_null": null,
        });
        for key in ["permission", "memory"] {
            let error = optional_str(&input, key, "agent_spawn")
                .expect_err("a non-string restriction must be refused");
            assert!(error.to_string().contains(key), "{error}");
            assert!(error.to_string().contains("must be a string"), "{error}");
        }
        assert_eq!(
            optional_str(&input, "good", "agent_spawn").expect("string"),
            Some("read")
        );
        // Absent, null, and whitespace all mean "not specified", which is what lets the defaults
        // apply without the caller having to say so.
        for key in ["missing", "blank", "explicit_null"] {
            assert_eq!(optional_str(&input, key, "agent_spawn").expect("ok"), None);
        }
    }

    #[test]
    fn test_string_array_skips_non_strings_and_blanks() {
        let input = serde_json::json!({
            "deny_servers": ["notion", 7, "  ", "  linear  ", null],
        });
        assert_eq!(string_array(&input, "deny_servers"), vec![
            "notion".to_string(),
            "linear".to_string()
        ],);
        assert!(string_array(&input, "absent").is_empty());
        // A non-array value is not a one-element list.
        assert!(string_array(&serde_json::json!({"x": "notion"}), "x").is_empty());
    }

    #[test]
    fn test_clamp_enabled_permissions_drops_higher_modes() {
        let clamped = clamp_enabled_permissions(EnabledPermissions::ALL, Permission::Read);
        assert!(clamped.is_enabled(Permission::None));
        assert!(clamped.is_enabled(Permission::Read));
        assert!(!clamped.is_enabled(Permission::Ask));
        assert!(!clamped.is_enabled(Permission::Unrestricted));
    }

    #[test]
    fn test_clamp_enabled_permissions_falls_back_to_singleton() {
        // Parent enabled only Write; clamping to Read leaves an empty intersection, so we fall back
        // to a singleton set of the ceiling itself rather than an invalid empty set.
        let only_write = EnabledPermissions::from_modes([Permission::Unrestricted]).unwrap();
        let clamped = clamp_enabled_permissions(only_write, Permission::Read);
        assert!(clamped.is_enabled(Permission::Read));
        assert!(!clamped.is_enabled(Permission::Unrestricted));
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
        // the child gets 0 and is not granted a nested agent_spawn.
        let (remaining, _absolute, allow) = child_spawn_depth(1, 0, None);
        assert_eq!(remaining, 0);
        assert!(!allow);
    }

    /// `max_depth` narrows the child's budget and can never widen it. `[session]
    /// subagent_max_depth` is documented as a ceiling, so a model asking for more than the operator
    /// allowed gets the operator's answer.
    #[test]
    fn test_child_spawn_depth_override_only_narrows() {
        // Asking for more than is left yields what is left, not what was asked for.
        let (remaining, _absolute, allow) = child_spawn_depth(3, 0, Some(9));
        assert_eq!(remaining, 2);
        assert!(allow);

        // At the documented `subagent_max_depth = 1`, no override can grant a grandchild.
        let (remaining, _absolute, allow) = child_spawn_depth(1, 0, Some(5));
        assert_eq!(remaining, 0);
        assert!(!allow, "subagent_max_depth = 1 must forbid nesting");

        // Asking for less than is left is honoured: the agent may still restrict itself.
        let (remaining, _absolute, _allow) = child_spawn_depth(5, 0, Some(1));
        assert_eq!(remaining, 1);

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
            ToolDenials::default(),
            sub_list.clone(),
            test_session_manager().await,
            Arc::new(tokio::sync::RwLock::new(None)),
            crate::skills::SkillCache::for_root(None),
            crate::memory::MemoryStore::detached(),
            MemoryAccess::Write,
            None,
            Vec::new(),
            crate::workspace::test_cwd(),
            crate::workspace::test_roots(),
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

    fn test_spec(permission: Permission) -> SubagentSpec {
        SubagentSpec {
            permission,
            enabled_permissions: vec![Permission::None, permission],
            denied_servers: vec!["mekabridge".to_string()],
            denied_tools: vec!["write_file".to_string()],
            memory: MemoryAccess::None,
            instructions: InstructionAccess::Inherit,
            inherited_scratchpad: vec!["build_log".to_string()],
            remaining_depth: 2,
            absolute_depth: 1,
        }
    }

    #[test]
    fn test_subagent_spec_round_trips_through_json() {
        let spec = test_spec(Permission::Read);
        let encoded = serde_json::to_string(&spec).expect("encode");
        let decoded: SubagentSpec = serde_json::from_str(&encoded).expect("decode");
        assert_eq!(spec, decoded);
        // The two enums persist as the lowercase words config uses, so a stored spec is readable.
        assert!(encoded.contains("\"permission\":\"read\""), "{encoded}");
        assert!(encoded.contains("\"memory\":\"none\""), "{encoded}");
    }

    /// A spec missing fields still decodes rather than failing, but every absent field takes its
    /// *restrictive* value: losing a field must cost the worker authority, never grant it.
    #[test]
    fn test_subagent_spec_decodes_a_minimal_document_and_fails_closed() {
        let decoded: SubagentSpec =
            serde_json::from_str(r#"{"permission":"unrestricted"}"#).expect("decode");
        assert_eq!(decoded.permission, Permission::Unrestricted);
        assert_eq!(
            decoded.memory,
            MemoryAccess::None,
            "an absent memory level must cost the worker the store, not hand it over"
        );
        assert_eq!(decoded.remaining_depth, 0, "and must not let it spawn");
        // The deny lists are the one pair that can't fail closed on their own (an empty list is
        // indistinguishable from "nothing was denied"), which is why `agent_followup` re-unions
        // them with current config rather than trusting the spec alone.
        assert!(decoded.denied_servers.is_empty());
    }

    /// `permission` is the one field with no default: a spec that lost it cannot be second-guessed,
    /// so the decode fails and `agent_followup` refuses the worker outright.
    #[test]
    fn test_subagent_spec_without_a_permission_refuses_to_decode() {
        assert!(serde_json::from_str::<SubagentSpec>(r#"{"memory":"write"}"#).is_err());
    }

    /// The core follow-up invariant: the worker is rebuilt at the level its spawn call chose. A
    /// parent that has since moved to `unrestricted` does not drag the worker up with it.
    #[test]
    fn test_spec_permission_survives_a_parent_that_has_since_escalated() {
        let spec = test_spec(Permission::Read);
        let rebuilt = spec.shared_permission(Permission::Unrestricted);
        assert_eq!(rebuilt.get(), Permission::Read);
        assert!(!rebuilt.enabled().is_enabled(Permission::Unrestricted));
        assert!(!rebuilt.enabled().is_enabled(Permission::Ask));
    }

    /// And the other direction, which matters more: a worker spawned at `unrestricted` must drop to
    /// `read` when the user switches the session down. Otherwise pressing Shift+Tab would stop
    /// the agent writing while leaving every worker it already has free to write on its behalf.
    #[test]
    fn test_spec_permission_follows_a_parent_that_has_since_been_restricted() {
        let spec = SubagentSpec {
            enabled_permissions: vec![Permission::Read, Permission::Unrestricted],
            ..test_spec(Permission::Unrestricted)
        };
        let rebuilt = spec.shared_permission(Permission::Read);
        assert_eq!(rebuilt.get(), Permission::Read);
        assert!(
            !rebuilt.enabled().is_enabled(Permission::Unrestricted),
            "the downgrade has to reach the enabled set too, or a switch path could climb back"
        );
        // None is a floor like any other.
        assert_eq!(
            spec.shared_permission(Permission::None).get(),
            Permission::None
        );
        // Unrestricted parent: the spec's own level still governs.
        assert_eq!(
            spec.shared_permission(Permission::Unrestricted).get(),
            Permission::Unrestricted
        );
    }

    /// The clamp above happens once, at build time. This is the half that was missing: a worker
    /// already running when the user presses Shift+Tab has to see the new level on its next tool
    /// call, not finish at the level it started with. Previously `shared_permission` minted a fresh
    /// atomic from a snapshot, so a downgrade reached the parent's next call and nothing else --
    /// while `permissions.md` presents cycling the parent as the way to restrict sub-agents.
    #[test]
    fn a_running_sub_agent_sees_a_parent_downgrade() {
        let parent = SharedPermission::new(Permission::Unrestricted, EnabledPermissions::ALL);
        let spec = SubagentSpec {
            enabled_permissions: vec![Permission::Read, Permission::Unrestricted],
            ..test_spec(Permission::Unrestricted)
        };

        let worker = spec.shared_permission_bounded(&parent);
        assert_eq!(
            worker.get(),
            Permission::Unrestricted,
            "spawned under a write parent"
        );

        // The user cycles the session down mid-run.
        parent.set_unchecked(Permission::None);
        assert_eq!(
            worker.get(),
            Permission::None,
            "the worker must not outlive the authority it was granted under"
        );

        // And back up: the rule is min(own grant, what the human currently permits), read in both
        // directions. The worker never exceeds its own grant either way.
        parent.set_unchecked(Permission::Unrestricted);
        assert_eq!(worker.get(), Permission::Unrestricted);
    }

    /// A worker granted less than its parent keeps its own lower level when the parent is raised:
    /// the ceiling bounds from above and never lifts.
    #[test]
    fn a_parent_raise_does_not_lift_a_worker_above_its_own_grant() {
        let parent = SharedPermission::new(Permission::Read, EnabledPermissions::ALL);
        let spec = test_spec(Permission::Read);

        let worker = spec.shared_permission_bounded(&parent);
        parent.set_unchecked(Permission::Unrestricted);

        assert_eq!(
            worker.get(),
            Permission::Read,
            "the spec's own grant is still the worker's ceiling"
        );
    }

    /// A spec whose enabled set is empty or unparseable must not fall back to "everything". The
    /// safe floor is the recorded level alone.
    #[test]
    fn test_spec_permission_falls_back_narrow_not_wide() {
        let spec = SubagentSpec {
            enabled_permissions: Vec::new(),
            ..test_spec(Permission::Read)
        };
        let rebuilt = spec.shared_permission(Permission::Unrestricted);
        assert_eq!(rebuilt.get(), Permission::Read);
        assert!(rebuilt.enabled().is_enabled(Permission::Read));
        assert!(!rebuilt.enabled().is_enabled(Permission::Unrestricted));
    }

    #[test]
    fn test_spec_denials_reconstruct_both_lists() {
        let denials = test_spec(Permission::Read).denials();
        assert!(denials.denies_server("mekabridge"));
        assert!(denials.denies_tool("mcp__mekabridge__send_message"));
        assert!(denials.denies_tool("write_file"));
        assert!(!denials.denies_tool("read_file"));
    }

    /// One round of scripted assistant text, for a sub-agent turn.
    fn text_round(text: &str) -> Vec<crate::provider::mock::MockEvent> {
        vec![
            crate::provider::mock::MockEvent::Text {
                text: text.to_string(),
            },
            crate::provider::mock::MockEvent::MessageEnd {
                stop_reason: crate::provider::mock::MockStopReason::EndTurn,
            },
        ]
    }

    /// A parent agent's `ToolBuilderParams` pointing at `session_manager`, with `parent_permission`
    /// as the ceiling and no MCP manager attached.
    fn test_params(
        session_manager: SessionManager,
        parent_session: Arc<RwLock<Option<Uuid>>>,
    ) -> ToolBuilderParams {
        ToolBuilderParams {
            web_client: crate::config::WebClientConfig::default(),
            sandbox_enabled: false,
            sandbox_capability: crate::sandbox::SandboxCapability::Unavailable,
            sandbox_backend: crate::config::SandboxBackend::Landlock,
            backend_probe: crate::sandbox::BackendProbe::Missing {
                reason: "test fixture".to_string(),
            },
            builtin_filter: BuiltinToolFilter::default(),
            skills: crate::skills::SkillCache::for_root(None),
            memories: crate::memory::MemoryStore::detached(),
            // A root agent: holds the whole store, and has instructions it could pass on.
            memory_access: MemoryAccess::Write,
            config_denials: ToolDenials::default(),
            mcp_manager: None,
            session_manager,
            parent_shared_session_id: parent_session,
            session_stats: Arc::new(crate::stats::SessionStats::default()),
            parent_options: AgentOptions {
                streaming: false,
                sandboxed_shell: false,
                context_messages: None,
                auto_compact: false,
                compact_checkpoint: false,
                context_window: 0,
                user_instructions: Some("never inherited by a worker".to_string()),
                mcp_grace: std::time::Duration::ZERO,
                system_prompt_override: None,
            },
            parent_cwd: crate::workspace::test_cwd(),
            parent_roots: crate::workspace::test_roots(),
            parent_frontend: Arc::new(crate::frontend::SilentFrontend),
        }
    }

    /// The whole Phase 4 loop against a scripted provider: spawn returns an id, `agent_list`
    /// reports the worker, a follow-up sees the first turn's history, and `agent_delete`
    /// removes it.
    #[tokio::test]
    async fn test_spawn_followup_and_delete_round_trip() {
        let session_manager = test_session_manager().await;
        let parent_sid = session_manager
            .create_session(None)
            .await
            .expect("parent session");
        let parent_session = Arc::new(RwLock::new(Some(parent_sid)));
        let provider: Arc<dyn Provider> =
            Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
                text_round("first answer"),
                text_round("second answer"),
            ]));
        let params = test_params(session_manager.clone(), Arc::clone(&parent_session));

        let spawn = AgentSpawnTool {
            provider: Arc::clone(&provider),
            parent_permission: SharedPermission::new(
                Permission::Unrestricted,
                EnabledPermissions::ALL,
            ),
            tool_builder_params: params.clone(),
            inherited_denials: ToolDenials::default(),
            remaining_depth: 1,
            absolute_depth: 0,
        };
        let output = spawn
            .execute(
                serde_json::json!({ "prompt": "look into it", "permission": "read" }),
                CancellationToken::new(),
            )
            .await
            .expect("spawn succeeds");
        let text = super::super::tests::text_content(&output);
        assert!(text.contains("first answer"), "{text}");

        let agent_id: Uuid = text
            .lines()
            .find_map(|line| line.strip_prefix("agent: "))
            .and_then(|id| Uuid::parse_str(id.trim()).ok())
            .unwrap_or_else(|| panic!("spawn must return a usable agent id, got: {text}"));

        // The spawn call restricted the worker below the parent; the persisted spec says so, and
        // that is what a follow-up rebuilds from.
        let spec: SubagentSpec = serde_json::from_str(
            &session_manager
                .load_subagent_spec(agent_id)
                .await
                .expect("load spec")
                .expect("a spawned worker has a spec"),
        )
        .expect("spec decodes");
        assert_eq!(spec.permission, Permission::Read);

        let list = AgentListTool {
            tool_builder_params: params.clone(),
        };
        let listed = super::super::tests::text_content(
            &list
                .execute(serde_json::json!({}), CancellationToken::new())
                .await
                .expect("list succeeds"),
        );
        assert!(listed.contains(&agent_id.to_string()), "{listed}");

        let followup = AgentFollowupTool {
            provider: Arc::clone(&provider),
            parent_permission: SharedPermission::new(
                Permission::Unrestricted,
                EnabledPermissions::ALL,
            ),
            tool_builder_params: params.clone(),
            in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        };
        let second = super::super::tests::text_content(
            &followup
                .execute(
                    serde_json::json!({ "agent": agent_id.to_string(), "prompt": "and then?" }),
                    CancellationToken::new(),
                )
                .await
                .expect("followup succeeds"),
        );
        assert!(second.contains("second answer"), "{second}");

        // The worker's own history carried into the second turn rather than starting over: its log
        // now holds both tasks and both answers.
        let events = session_manager.load_events(agent_id).await.expect("events");
        let transcript = format!("{:?}", events);
        assert!(transcript.contains("look into it"), "{transcript}");
        assert!(transcript.contains("first answer"), "{transcript}");
        assert!(transcript.contains("and then?"), "{transcript}");

        let delete = AgentDeleteTool {
            tool_builder_params: params,
            in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        };
        delete
            .execute(
                serde_json::json!({ "agent": agent_id.to_string() }),
                CancellationToken::new(),
            )
            .await
            .expect("delete succeeds");
        assert!(
            session_manager
                .load_session_tree(parent_sid)
                .await
                .expect("tree")
                .iter()
                .all(|row| row.id != agent_id),
            "the deleted worker must be gone from the parent's tree"
        );
        // The parent itself is untouched.
        assert!(
            session_manager
                .load_session_tree(parent_sid)
                .await
                .expect("tree")
                .iter()
                .any(|row| row.id == parent_sid)
        );
    }

    /// End to end: a worker spawned while the session was at `unrestricted` must not still be there
    /// after the user restricts the session to Read. Asserted through the worker's registry
    /// rather than A follow-up holds the worker's session for the length of its turn.
    ///
    /// `agent_spawn` gained this and `agent_followup` did not, which left the exposure open on the
    /// door that reaches an *existing* worker: a follow-up runs a full turn against a row nothing
    /// claims, for seconds to minutes, and a concurrent `meka session delete --all` takes the lock
    /// nobody holds and cascades the conversation away. The follow-up's next message insert then
    /// dies on a foreign-key violation with the worker's output lost.
    ///
    /// Refused rather than warned, unlike spawn: this id already exists, so a lock it cannot take
    /// genuinely means somebody else is running a turn on this worker, and two turns interleaved
    /// into one conversation is the thing the lock is for.
    #[tokio::test]
    async fn a_followup_is_refused_while_something_else_holds_the_worker() {
        let session_manager = test_session_manager().await;
        let parent_sid = session_manager.create_session(None).await.expect("parent");
        let parent_session = Arc::new(RwLock::new(Some(parent_sid)));
        let params = test_params(session_manager.clone(), Arc::clone(&parent_session));
        // Never reached: the refusal happens before any turn runs, which is the point.
        let provider: Arc<dyn Provider> =
            Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
                text_round("unreachable"),
            ]));
        // The claim `create_child_session` takes *is* the contention: holding it here is exactly
        // what a second meka looks like from the follow-up's side, since `flock` conflicts across
        // open file descriptions rather than across processes.
        // A real spec, or the follow-up refuses at the resumability check before it ever reaches
        // the lock.
        let spec = SubagentSpec {
            permission: Permission::Read,
            enabled_permissions: vec![Permission::Read],
            denied_servers: Vec::new(),
            denied_tools: Vec::new(),
            memory: MemoryAccess::None,
            instructions: InstructionAccess::None,
            inherited_scratchpad: Vec::new(),
            remaining_depth: 0,
            absolute_depth: 1,
        };
        let (worker, held) = session_manager
            .create_child_session(
                parent_sid,
                None,
                Some(serde_json::to_string(&spec).expect("serialize the spec")),
            )
            .await
            .expect("child");
        let _held = held.expect("the spawn's own claim");

        let followup = AgentFollowupTool {
            provider: Arc::clone(&provider),
            parent_permission: SharedPermission::new(
                Permission::Unrestricted,
                EnabledPermissions::ALL,
            ),
            tool_builder_params: params.clone(),
            in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        };
        let refused = followup
            .execute(
                serde_json::json!({ "agent": worker.to_string(), "prompt": "carry on" }),
                CancellationToken::new(),
            )
            .await
            .expect_err("a worker somebody else is running must not be run again");
        assert!(
            refused.to_string().contains("cannot follow up"),
            "and the refusal has to name the worker: {refused}"
        );
    }

    /// A follow-up turn runs the worker at the parent's *current* level, not at the one it was
    /// spawned with.
    ///
    /// Asserted against the worker's persisted conversation rather than against its spec, since the
    /// registry is what actually decides what it can do.
    #[tokio::test]
    async fn test_followup_drops_a_worker_when_the_session_is_restricted() {
        let session_manager = test_session_manager().await;
        let parent_sid = session_manager.create_session(None).await.expect("parent");
        let parent_session = Arc::new(RwLock::new(Some(parent_sid)));
        // A path unique to this run: a shared one would leave a file behind on the failing case and
        // make the *next* run fail for the wrong reason.
        let temporary = tempfile::tempdir().expect("tempdir");
        let target = temporary.path().join("escalation-probe.txt");
        // The worker tries to write on its follow-up turn, then reports.
        let provider: Arc<dyn Provider> =
            Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
                vec![
                    crate::provider::mock::MockEvent::ToolUseStart {
                        id: "call-1".into(),
                        name: "write_file".into(),
                    },
                    crate::provider::mock::MockEvent::ToolUseEnd {
                        input: serde_json::json!({
                            "path": target.to_string_lossy(),
                            "content": "escalated",
                        }),
                    },
                    crate::provider::mock::MockEvent::MessageEnd {
                        stop_reason: crate::provider::mock::MockStopReason::ToolUse,
                    },
                ],
                text_round("could not write"),
            ]));
        let params = test_params(session_manager.clone(), Arc::clone(&parent_session));

        // Spawned at `unrestricted`.
        let spec = SubagentSpec {
            permission: Permission::Unrestricted,
            enabled_permissions: vec![Permission::Read, Permission::Unrestricted],
            denied_servers: Vec::new(),
            denied_tools: Vec::new(),
            memory: MemoryAccess::Write,
            instructions: InstructionAccess::None,
            inherited_scratchpad: Vec::new(),
            remaining_depth: 0,
            absolute_depth: 1,
        };
        let child = session_manager
            .create_child_session(
                parent_sid,
                None,
                Some(serde_json::to_string(&spec).expect("encode")),
            )
            .await
            .expect("child")
            .0;

        // The session is then restricted to Read, as `/permission` or Shift+Tab would.
        let restricted = SharedPermission::new(
            Permission::Read,
            EnabledPermissions::from_modes([Permission::Read]).expect("set"),
        );
        let followup = AgentFollowupTool {
            provider,
            parent_permission: restricted,
            tool_builder_params: params,
            in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        };
        followup
            .execute(
                serde_json::json!({ "agent": child.to_string(), "prompt": "write the file" }),
                CancellationToken::new(),
            )
            .await
            .expect("followup runs");

        // The worker really was refused, by the dispatcher, at Read. Asserted against its persisted
        // conversation rather than against the spec: what matters is what the worker was able to
        // *do*, and a spec-only assertion would still pass if the clamp never reached the registry.
        let transcript = format!(
            "{:?}",
            session_manager.load_events(child).await.expect("events")
        );
        assert!(
            transcript.contains("Permission denied") && transcript.contains("write_file"),
            "the worker should have been refused write_file at Read, got: {transcript}"
        );
        assert!(!target.exists(), "and nothing should have been written");

        // The recorded spec is untouched, so the worker returns to `unrestricted` if the session
        // does: the clamp is a live ceiling, not a rewrite of the spawn terms.
        let reloaded: SubagentSpec = serde_json::from_str(
            &session_manager
                .load_subagent_spec(child)
                .await
                .expect("load")
                .expect("spec"),
        )
        .expect("decode");
        assert_eq!(reloaded.permission, Permission::Unrestricted);
    }

    /// A restriction added to config after a worker was spawned still reaches it on follow-up. The
    /// spec is a floor on restriction, not a licence to ignore what the operator has since decided.
    #[tokio::test]
    async fn test_followup_applies_denials_config_gained_since_the_spawn() {
        let session_manager = test_session_manager().await;
        let parent_sid = session_manager.create_session(None).await.expect("parent");
        let provider: Arc<dyn Provider> =
            Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
                text_round("ok"),
            ]));
        let mut params = test_params(
            session_manager.clone(),
            Arc::new(RwLock::new(Some(parent_sid))),
        );
        // Config now denies a server and all memory access; the spec predates both.
        params.config_denials = ToolDenials::new(vec!["mekabridge".to_string()], Vec::new());
        params.memory_access = MemoryAccess::None;

        let spec = SubagentSpec {
            permission: Permission::Read,
            enabled_permissions: vec![Permission::Read],
            denied_servers: Vec::new(),
            denied_tools: Vec::new(),
            memory: MemoryAccess::Write,
            instructions: InstructionAccess::None,
            inherited_scratchpad: Vec::new(),
            remaining_depth: 0,
            absolute_depth: 1,
        };
        let child = session_manager
            .create_child_session(
                parent_sid,
                None,
                Some(serde_json::to_string(&spec).expect("encode")),
            )
            .await
            .expect("child")
            .0;

        let followup = AgentFollowupTool {
            provider,
            parent_permission: SharedPermission::new(Permission::Read, EnabledPermissions::ALL),
            tool_builder_params: params,
            in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        };
        followup
            .execute(
                serde_json::json!({ "agent": child.to_string(), "prompt": "go" }),
                CancellationToken::new(),
            )
            .await
            .expect("followup runs");

        // The combine happens in memory; the stored spec still records the original spawn terms, so
        // loosening config later restores them rather than leaving the worker permanently narrowed.
        let reloaded: SubagentSpec = serde_json::from_str(
            &session_manager
                .load_subagent_spec(child)
                .await
                .expect("load")
                .expect("spec"),
        )
        .expect("decode");
        assert!(reloaded.denied_servers.is_empty());
        assert_eq!(reloaded.memory, MemoryAccess::Write);
    }

    /// Config denials narrow a recorded grant on every axis, not just the ones a test happened to
    /// look at.
    ///
    /// Asserted against the combined spec directly. The call site deliberately leaves the *stored*
    /// spec alone so a later loosening restores the original terms, which means the only thing
    /// observable there is that the recording did not change -- true whether or not the narrowing
    /// ran. Dropping `denied_servers` from the expression survived the whole suite on exactly that
    /// gap, while `denied_tools` beside it was caught.
    #[test]
    fn a_followup_narrows_a_recorded_grant_by_config_on_every_axis() {
        let spec = SubagentSpec {
            permission: Permission::Read,
            enabled_permissions: vec![Permission::Read],
            denied_servers: Vec::new(),
            denied_tools: Vec::new(),
            memory: MemoryAccess::Write,
            instructions: InstructionAccess::None,
            inherited_scratchpad: Vec::new(),
            remaining_depth: 0,
            absolute_depth: 1,
        };
        let config = ToolDenials::new(vec!["mekabridge".to_string()], vec![
            "web_search".to_string(),
        ]);

        let combined = combined_for_followup(spec.clone(), &config, MemoryAccess::None);

        assert!(
            combined.denied_servers.contains(&"mekabridge".to_string()),
            "a server denied since the spawn must reach the worker: {:?}",
            combined.denied_servers
        );
        assert!(
            combined.denied_tools.contains(&"web_search".to_string()),
            "and so must a tool: {:?}",
            combined.denied_tools
        );
        assert_eq!(
            combined.memory,
            MemoryAccess::None,
            "memory narrows to the lesser of the grant and config"
        );
        // Never the other direction: config cannot hand back what the spawn call withheld.
        assert_eq!(combined.permission, Permission::Read);
        assert_eq!(
            combined_for_followup(spec, &ToolDenials::default(), MemoryAccess::Write).memory,
            MemoryAccess::Write,
            "an empty config leaves the recorded grant exactly as it was"
        );
    }

    /// Drive a spawn and hand back the spec that was recorded for the worker.
    async fn spawn_and_read_spec(
        params: ToolBuilderParams,
        parent_permission: Permission,
        parent_sid: Uuid,
        input: serde_json::Value,
    ) -> Result<SubagentSpec> {
        let session_manager = params.session_manager.clone();
        let spawn = AgentSpawnTool {
            provider: Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
                text_round("done"),
            ])),
            parent_permission: SharedPermission::new(parent_permission, EnabledPermissions::ALL),
            tool_builder_params: params,
            inherited_denials: ToolDenials::default(),
            remaining_depth: 1,
            absolute_depth: 0,
        };
        let output = spawn.execute(input, CancellationToken::new()).await?;
        let text = super::super::tests::text_content(&output);
        let agent_id: Uuid = text
            .lines()
            .find_map(|line| line.strip_prefix("agent: "))
            .and_then(|id| Uuid::parse_str(id.trim()).ok())
            .unwrap_or_else(|| panic!("no agent id in: {text}"));
        let _ = parent_sid;
        let json = session_manager
            .load_subagent_spec(agent_id)
            .await?
            .expect("a spawned worker has a spec");
        Ok(serde_json::from_str(&json).expect("spec decodes"))
    }

    /// A worker starts with nothing it was not given. Both grants default to the restrictive end,
    /// so a parent that never considers the question produces a clean slate rather than a copy of
    /// itself.
    #[tokio::test]
    async fn test_grants_default_to_nothing() {
        let session_manager = test_session_manager().await;
        let parent_sid = session_manager.create_session(None).await.expect("parent");
        let params = test_params(session_manager, Arc::new(RwLock::new(Some(parent_sid))));

        let spec = spawn_and_read_spec(
            params,
            Permission::Unrestricted,
            parent_sid,
            serde_json::json!({ "prompt": "go" }),
        )
        .await
        .expect("spawn");
        assert_eq!(spec.memory, MemoryAccess::None);
        assert_eq!(spec.instructions, InstructionAccess::None);
    }

    /// And gets exactly what it was given when the parent asks.
    #[tokio::test]
    async fn test_grants_are_recorded_when_asked_for() {
        let session_manager = test_session_manager().await;
        let parent_sid = session_manager.create_session(None).await.expect("parent");
        let params = test_params(session_manager, Arc::new(RwLock::new(Some(parent_sid))));

        let spec = spawn_and_read_spec(
            params,
            Permission::Unrestricted,
            parent_sid,
            serde_json::json!({ "prompt": "go", "memory": "read", "instructions": "inherit" }),
        )
        .await
        .expect("spawn");
        assert_eq!(spec.memory, MemoryAccess::Read);
        assert_eq!(spec.instructions, InstructionAccess::Inherit);
    }

    /// `write` is refused rather than clamped to `read`, so a parent asking for it learns that no
    /// sub-agent can have it instead of quietly getting something else.
    #[tokio::test]
    async fn test_memory_write_is_refused_not_clamped() {
        let session_manager = test_session_manager().await;
        let parent_sid = session_manager.create_session(None).await.expect("parent");
        let params = test_params(
            session_manager.clone(),
            Arc::new(RwLock::new(Some(parent_sid))),
        );

        let error = spawn_and_read_spec(
            params,
            Permission::Unrestricted,
            parent_sid,
            serde_json::json!({ "prompt": "go", "memory": "write" }),
        )
        .await
        .expect_err("write is not grantable");
        assert!(error.to_string().contains("not available to sub-agents"));
        // A refused spawn leaves nothing behind.
        assert_eq!(
            session_manager
                .load_session_tree(parent_sid)
                .await
                .expect("tree")
                .len(),
            1
        );
    }

    /// A worker cannot hand a grandchild more than it holds. Memory clamps against the spawning
    /// agent's own level; instructions clamp against whether it has the text at all.
    #[tokio::test]
    async fn test_a_worker_cannot_grant_more_than_it_holds() {
        let session_manager = test_session_manager().await;
        let parent_sid = session_manager.create_session(None).await.expect("parent");
        let mut params = test_params(session_manager, Arc::new(RwLock::new(Some(parent_sid))));
        // Stand in for a worker that was itself granted nothing: no memory, and no copy of the
        // instructions to pass on. This is exactly what `build_subagent` hands a nested
        // `AgentSpawnTool`.
        params.memory_access = MemoryAccess::None;
        params.parent_options.user_instructions = None;

        let spec = spawn_and_read_spec(
            params,
            Permission::Unrestricted,
            parent_sid,
            serde_json::json!({ "prompt": "go", "memory": "read", "instructions": "inherit" }),
        )
        .await
        .expect("spawn");
        assert_eq!(
            spec.memory,
            MemoryAccess::None,
            "a worker with no memory cannot grant read"
        );
        assert_eq!(
            spec.instructions,
            InstructionAccess::None,
            "and one with no instructions cannot pass them on"
        );
    }

    /// The same clamp, but through `build_subagent` rather than a hand-built `ToolBuilderParams`.
    ///
    /// A worker granted nothing spawns a grandchild and asks for everything. Written this way
    /// because the clamp is only as good as the params the nested `AgentSpawnTool` is handed, and a
    /// test that sets those params itself would pass even if `build_subagent` set them wrong.
    #[tokio::test]
    async fn test_a_worker_granted_nothing_cannot_grant_its_own_child_anything() {
        use crate::provider::mock::{MockEvent, MockStopReason};

        let session_manager = test_session_manager().await;
        let parent_sid = session_manager.create_session(None).await.expect("parent");
        let params = test_params(
            session_manager.clone(),
            Arc::new(RwLock::new(Some(parent_sid))),
        );

        // Worker turn 1 asks for a grandchild with both grants; then the grandchild runs; then the
        // worker reports. All three drain from the one shared script, in that order.
        let provider: Arc<dyn Provider> =
            Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
                vec![
                    MockEvent::ToolUseStart {
                        id: "nest-1".into(),
                        name: "agent_spawn".into(),
                    },
                    MockEvent::ToolUseEnd {
                        input: serde_json::json!({
                            "prompt": "grandchild task",
                            "memory": "read",
                            "instructions": "inherit",
                        }),
                    },
                    MockEvent::MessageEnd {
                        stop_reason: MockStopReason::ToolUse,
                    },
                ],
                text_round("grandchild done"),
                text_round("worker done"),
            ]));

        let spawn = AgentSpawnTool {
            provider,
            parent_permission: SharedPermission::new(
                Permission::Unrestricted,
                EnabledPermissions::ALL,
            ),
            tool_builder_params: params,
            inherited_denials: ToolDenials::default(),
            // Deep enough that the worker gets its own `agent_spawn`.
            remaining_depth: 2,
            absolute_depth: 0,
        };
        // The worker itself is granted nothing, which is the default.
        spawn
            .execute(
                serde_json::json!({ "prompt": "worker task" }),
                CancellationToken::new(),
            )
            .await
            .expect("spawn");

        let tree = session_manager
            .load_session_tree(parent_sid)
            .await
            .expect("tree");
        assert_eq!(tree.len(), 3, "parent, worker, grandchild");
        let worker = tree
            .iter()
            .find(|row| row.parent_id == Some(parent_sid))
            .expect("worker");
        let grandchild = tree
            .iter()
            .find(|row| row.parent_id == Some(worker.id))
            .expect("the worker really spawned one");

        let spec: SubagentSpec = serde_json::from_str(
            &session_manager
                .load_subagent_spec(grandchild.id)
                .await
                .expect("load")
                .expect("spec"),
        )
        .expect("decode");
        assert_eq!(
            spec.memory,
            MemoryAccess::None,
            "the worker held no memory, so it had none to grant"
        );
        assert_eq!(
            spec.instructions,
            InstructionAccess::None,
            "and no copy of the instructions to pass on"
        );
    }

    /// The report handed to a scratchpad must be the report, with no `agent:` header bolted on.
    ///
    /// The redirect is universal and stores the whole result text, so a header would end up inside
    /// the entry that `inherit_scratchpad` later hands to another worker — and the model would not
    /// even receive the id, since it sees only a reference once the redirect fires.
    #[tokio::test]
    async fn test_a_redirected_report_carries_no_agent_header() {
        let session_manager = test_session_manager().await;
        let parent_sid = session_manager.create_session(None).await.expect("parent");
        let params = test_params(
            session_manager.clone(),
            Arc::new(RwLock::new(Some(parent_sid))),
        );
        let spawn = AgentSpawnTool {
            provider: Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
                text_round("the findings"),
                text_round("the findings"),
            ])),
            parent_permission: SharedPermission::new(
                Permission::Unrestricted,
                EnabledPermissions::ALL,
            ),
            tool_builder_params: params,
            inherited_denials: ToolDenials::default(),
            remaining_depth: 1,
            absolute_depth: 0,
        };

        let redirected = super::super::tests::text_content(
            &spawn
                .execute(
                    serde_json::json!({ "prompt": "go", "scratchpad": "findings" }),
                    CancellationToken::new(),
                )
                .await
                .expect("spawn"),
        );
        assert_eq!(
            redirected.trim(),
            "the findings",
            "a redirected result is the report alone"
        );
        assert!(!redirected.contains("agent:"), "{redirected}");

        // Without the redirect the id leads, which is how a parent reaches the worker again.
        let inline = super::super::tests::text_content(
            &spawn
                .execute(
                    serde_json::json!({ "prompt": "go" }),
                    CancellationToken::new(),
                )
                .await
                .expect("spawn"),
        );
        assert!(inline.starts_with("agent: "), "{inline}");
        assert!(inline.contains("the findings"));
    }

    /// A spec claiming `Write` cannot produce a worker that can write. `parse_grant` refuses it at
    /// the `agent_spawn` boundary, but a spec is persisted JSON and `meka session import` writes it
    /// verbatim from a user-supplied archive, so the guarantee is enforced where it is consumed.
    #[test]
    fn test_a_spec_claiming_write_is_capped_at_read() {
        let forged = SubagentSpec {
            memory: MemoryAccess::Write,
            ..test_spec(Permission::Unrestricted)
        };
        assert_eq!(forged.granted_memory(), MemoryAccess::Read);
        // And the honest values pass through untouched.
        assert_eq!(
            SubagentSpec {
                memory: MemoryAccess::Read,
                ..test_spec(Permission::Read)
            }
            .granted_memory(),
            MemoryAccess::Read
        );
        assert_eq!(
            test_spec(Permission::Read).granted_memory(),
            MemoryAccess::None
        );
    }

    /// A `memory: "read"` grant has to arrive with an index, or it is only half a grant:
    fn test_memory(name: &str, priority: u8, description: &str) -> crate::memory::Memory {
        crate::memory::Memory {
            name: name.to_string(),
            description: description.to_string(),
            priority,
            tags: Vec::new(),
            recorded_at: std::time::SystemTime::UNIX_EPOCH,
            updated_at: std::time::SystemTime::UNIX_EPOCH,
            read_count: 0,
            body: None,
        }
    }

    /// `memory_read` takes an exact name, and a sub-agent never receives the per-turn world state
    /// the primary agent's `[Memory]` section rides in. Without this the worker holds two tools and
    /// no idea what to call them with.
    #[test]
    fn test_a_granted_worker_gets_a_usable_memory_index() {
        let index = [
            test_memory("build-incantation", 1, "How this project is built"),
            test_memory("review-style", 2, "What the user wants from a review"),
        ];

        let rendered = render_subagent_memory_index(&index);
        assert!(rendered.contains("build-incantation"));
        assert!(rendered.contains("How this project is built"));
        assert!(rendered.contains("review-style"));
        assert!(rendered.contains("memory_read"), "and how to open one");
        // A worker cannot write, so it must not be told to. The primary agent's header says to call
        // `memory_write`, which is exactly why this renders separately.
        assert!(
            !rendered.contains("memory_write"),
            "must not point a read-only worker at a tool it lacks: {rendered}"
        );
        assert!(rendered.contains("report"), "it reports instead");

        // An empty store contributes nothing at all rather than an empty heading.
        assert!(render_subagent_memory_index(&[]).is_empty());
    }

    /// A truncated index must say so. Reading as "this is everything" is what turns a full store
    /// into a confidently incomplete answer.
    #[test]
    fn test_a_truncated_memory_index_states_its_remainder() {
        let memories: Vec<crate::memory::Memory> = (0..400)
            .map(|n| {
                test_memory(
                    &format!("memory-{n:03}"),
                    1,
                    &"a description long enough to make the budget bite".repeat(3),
                )
            })
            .collect();
        let rendered = render_subagent_memory_index(&memories);
        assert!(
            rendered.len() <= SUBAGENT_MEMORY_INDEX_MAX_BYTES + 200,
            "budget respected"
        );
        assert!(rendered.contains("more not listed here"), "{rendered}");
        assert!(rendered.contains("memory_search"), "and how to reach them");
    }

    /// One enormous description must not empty a granted worker's whole index.
    ///
    /// Descriptions are unbounded at the write door, and this was the one memory render that did
    /// not elide them. With the budget checked before *every* push and nothing exempt, a single
    /// 4,000-character description at the top of the store produced a header promising memories
    /// followed by "N more not listed here" -- in a worker that had been deliberately granted
    /// access to them. The parent's index and both search renderers already guarded both halves.
    #[test]
    fn one_enormous_description_does_not_empty_the_subagent_index() {
        let mut memories = vec![test_memory("enormous", 1, &"x".repeat(4_000))];
        memories.extend(
            (0..3).map(|n| test_memory(&format!("ordinary-{n}"), 3, "a short description")),
        );

        let rendered = render_subagent_memory_index(&memories);

        assert!(
            rendered.contains("**enormous**"),
            "the first entry is always emitted: {rendered}"
        );
        assert!(
            !rendered.contains(&"x".repeat(4_000)),
            "and its description is elided rather than carried whole"
        );
        assert!(
            rendered.contains("**ordinary-0**"),
            "which leaves room for the rest of the store: {rendered}"
        );
        assert!(
            rendered.len() <= SUBAGENT_MEMORY_INDEX_MAX_BYTES + 200,
            "budget still respected: {} bytes",
            rendered.len()
        );
    }

    /// The grant reaches the prompt, and its absence leaves no trace of the section.
    #[test]
    fn test_system_prompt_carries_instructions_only_when_granted() {
        let ungranted = build_subagent_system_prompt(Permission::Read, &[], &[], None, "");
        assert!(!ungranted.contains("User Instructions"));

        let granted = build_subagent_system_prompt(
            Permission::Read,
            &[],
            &[],
            Some("Never use pip. Always prefer uv."),
            "",
        );
        assert!(granted.contains("## User Instructions"));
        assert!(granted.contains("Never use pip. Always prefer uv."));
        // The worker is told whose rules these are, so persona clauses read as context rather than
        // as an instruction to address the user directly.
        assert!(granted.contains("report"), "{granted}");

        // Whitespace-only instructions are treated as absent, matching the primary agent.
        assert!(
            !build_subagent_system_prompt(Permission::Read, &[], &[], Some("  \n "), "")
                .contains("User Instructions")
        );
    }

    /// Denying `agent_spawn` means "no delegation", so the three tools that only ever act on what
    /// it produced go with it. Leaving them behind would give an agent that cannot spawn a worker
    /// the ability to drive workers a previous run left in the database.
    #[tokio::test]
    async fn test_denying_agent_spawn_takes_the_lifecycle_tools_with_it() {
        let session_manager = test_session_manager().await;
        let parent_sid = session_manager.create_session(None).await.expect("parent");
        let params = test_params(
            session_manager.clone(),
            Arc::new(RwLock::new(Some(parent_sid))),
        );
        let spawn = |params: ToolBuilderParams| AgentSpawnTool {
            provider: Arc::new(crate::provider::mock::MockProvider::from_rounds(Vec::new())),
            parent_permission: SharedPermission::new(
                Permission::Unrestricted,
                EnabledPermissions::ALL,
            ),
            tool_builder_params: params,
            inherited_denials: ToolDenials::default(),
            remaining_depth: 1,
            absolute_depth: 0,
        };

        let permitted = ToolRegistry::new();
        register_subagent_tools(&permitted, spawn(params.clone())).expect("register");
        for name in [
            "agent_spawn",
            "agent_list",
            "agent_followup",
            "agent_delete",
        ] {
            assert!(permitted.get(name).is_some(), "expected '{name}'");
        }

        let filtered = ToolRegistry::new_with_filter(BuiltinToolFilter::from_config(
            None,
            vec!["agent_spawn".to_string()],
            std::collections::HashMap::new(),
        ));
        register_subagent_tools(&filtered, spawn(params)).expect("register");
        for name in [
            "agent_spawn",
            "agent_list",
            "agent_followup",
            "agent_delete",
        ] {
            assert!(
                filtered.get(name).is_none(),
                "'{name}' must go with agent_spawn"
            );
        }
    }

    /// `agent_delete` shares the follow-up guard, so the two cannot run on one worker at once.
    /// Without it, a delete landing between a follow-up's ownership check and its first write makes
    /// the follow-up fail on a foreign-key violation -- a raw database error, on a path where the
    /// model did nothing wrong.
    #[tokio::test]
    async fn test_delete_is_refused_while_a_followup_holds_the_worker() {
        let session_manager = test_session_manager().await;
        let parent_sid = session_manager.create_session(None).await.expect("parent");
        let child = session_manager
            .create_child_session(
                parent_sid,
                None,
                Some(r#"{"permission":"read"}"#.to_string()),
            )
            .await
            .expect("child")
            .0;
        let params = test_params(
            session_manager.clone(),
            Arc::new(RwLock::new(Some(parent_sid))),
        );
        let in_flight: InFlightFollowups =
            Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));

        let delete = AgentDeleteTool {
            tool_builder_params: params,
            in_flight: Arc::clone(&in_flight),
        };

        // Stand in for a follow-up in progress on this worker.
        let held = FollowupGuard::claim(&in_flight, child).expect("claim");
        let error = delete
            .execute(
                serde_json::json!({ "agent": child.to_string() }),
                CancellationToken::new(),
            )
            .await
            .expect_err("a busy worker must not be deleted mid-follow-up");
        assert!(error.to_string().contains("busy"), "{error}");
        assert_eq!(
            session_manager
                .load_session_tree(parent_sid)
                .await
                .expect("tree")
                .len(),
            2,
            "and it is still there"
        );

        // Once the follow-up returns, the delete goes through.
        drop(held);
        delete
            .execute(
                serde_json::json!({ "agent": child.to_string() }),
                CancellationToken::new(),
            )
            .await
            .expect("delete succeeds once the worker is free");
    }

    /// `turns` counts tasks, not messages. `Agent::run_turn` persists tool results as user-role
    /// messages, so a single task that took three tool rounds must still read as one turn.
    #[tokio::test]
    async fn test_agent_list_turns_excludes_tool_result_messages() {
        use crate::{
            conversation::Event,
            provider::{ContentBlock, Message, Role, ToolResultContent},
        };

        let session_manager = test_session_manager().await;
        let parent_sid = session_manager.create_session(None).await.expect("parent");
        let child = session_manager
            .create_child_session(
                parent_sid,
                None,
                Some(r#"{"permission":"read"}"#.to_string()),
            )
            .await
            .expect("child")
            .0;

        let tool_result = Event::Append(Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "u1".to_string(),
                content: vec![ToolResultContent::Text {
                    text: "ok".to_string(),
                }],
                is_error: false,
            }],
        });
        for event in [
            Event::Append(Message::user("the one and only task")),
            Event::Append(Message::assistant_text("working")),
            tool_result.clone(),
            Event::Append(Message::assistant_text("still working")),
            tool_result,
            Event::Append(Message::assistant_text("done")),
        ] {
            session_manager
                .save_event(child, &event)
                .await
                .expect("save");
        }

        let list = AgentListTool {
            tool_builder_params: test_params(
                session_manager,
                Arc::new(RwLock::new(Some(parent_sid))),
            ),
        };
        let rendered = super::super::tests::text_content(
            &list
                .execute(serde_json::json!({}), CancellationToken::new())
                .await
                .expect("list"),
        );
        assert!(
            rendered.contains("turns=1"),
            "one task through two tool rounds is one turn, got: {rendered}"
        );
    }

    /// A skill that resolves by name but fails to load must not leave a childless session row
    /// behind: `agent_list` would advertise it as a worker, and following it up would resume a
    /// conversation that never happened.
    ///
    /// Unix-only because it needs a directory the process cannot read.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_a_failed_skill_load_leaves_no_orphan_session() {
        use std::os::unix::fs::PermissionsExt;

        let temporary = tempfile::tempdir().expect("tempdir");
        let root = temporary.path().join("skills");
        let skill_dir = root.join("broken");
        std::fs::create_dir_all(&skill_dir).expect("skill dir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: broken\ndescription: a skill whose body goes away\n---\n\nbody\n",
        )
        .expect("write skill");

        let skills = crate::skills::SkillCache::for_root(Some(root.clone()));
        assert_eq!(
            skills.current().await.skills.len(),
            1,
            "skill is discoverable"
        );
        // Making the root unreadable makes `disk_snapshot` return `None`, and `SkillCache::current`
        // then serves its cached list rather than wiping it. So the name still resolves and only
        // the body read fails -- the ordering this test is about, and a real race with a
        // `git checkout` or an editor moving a skill mid-turn. (Removing the root instead would not
        // do: a *missing* root is deliberately read as an empty store, which fails resolution.)
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o000))
            .expect("seal skills root");
        if std::fs::read_dir(&root).is_ok() {
            // Running as root, where the mode is advisory. Nothing to assert.
            return;
        }
        assert_eq!(
            skills.current().await.skills.len(),
            1,
            "resolution still succeeds"
        );

        let session_manager = test_session_manager().await;
        let parent_sid = session_manager.create_session(None).await.expect("parent");
        let mut params = test_params(
            session_manager.clone(),
            Arc::new(RwLock::new(Some(parent_sid))),
        );
        params.skills = skills;

        let spawn = AgentSpawnTool {
            provider: Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
                text_round("never runs"),
            ])),
            parent_permission: SharedPermission::new(Permission::Read, EnabledPermissions::ALL),
            tool_builder_params: params,
            inherited_denials: ToolDenials::default(),
            remaining_depth: 0,
            absolute_depth: 0,
        };
        let error = spawn
            .execute(
                serde_json::json!({ "skill": "broken" }),
                CancellationToken::new(),
            )
            .await
            .expect_err("an unreadable skill body must fail the spawn");
        assert!(
            error.to_string().contains("failed to load skill"),
            "{error}"
        );

        assert_eq!(
            session_manager
                .load_session_tree(parent_sid)
                .await
                .expect("tree")
                .len(),
            1,
            "the parent alone: a failed spawn must not leave a child row behind"
        );

        // Let the tempdir clean itself up.
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700))
            .expect("unseal skills root");
    }

    /// Delegating a skill whose `SKILL.md` will not parse must say so, not offer substitutes.
    ///
    /// The `skill_read` wording exists because a model told "not found" improvises the procedure.
    /// `agent_spawn` is the same audience with more at stake -- the improvisation runs in a worker,
    /// out of sight -- and it kept answering "skill 'x' not found. Available skills: ...", which
    /// reads as an invitation to pick one of those instead.
    #[tokio::test]
    async fn spawning_a_broken_skill_names_the_file_rather_than_offering_alternatives() {
        let temporary = tempfile::tempdir().expect("tempdir");
        let root = temporary.path().join("skills");
        for (name, body) in [
            (
                "wrecked",
                "---\nname: wrecked\ndescription: [unclosed\n---\nbody\n",
            ),
            (
                "fine",
                "---\nname: fine\ndescription: a working one\n---\nbody\n",
            ),
        ] {
            std::fs::create_dir_all(root.join(name)).expect("skill dir");
            std::fs::write(root.join(name).join("SKILL.md"), body).expect("write skill");
        }

        let session_manager = test_session_manager().await;
        let parent_sid = session_manager.create_session(None).await.expect("parent");
        let mut params = test_params(
            session_manager.clone(),
            Arc::new(RwLock::new(Some(parent_sid))),
        );
        params.skills = crate::skills::SkillCache::for_root(Some(root));

        let spawn = AgentSpawnTool {
            provider: Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
                text_round("never runs"),
            ])),
            parent_permission: SharedPermission::new(Permission::Read, EnabledPermissions::ALL),
            tool_builder_params: params,
            inherited_denials: ToolDenials::default(),
            remaining_depth: 1,
            absolute_depth: 0,
        };
        let output = spawn
            .execute(
                serde_json::json!({ "skill": "wrecked" }),
                CancellationToken::new(),
            )
            .await
            .expect("a broken skill is a tool result, not a tool error");
        assert!(output.is_error);
        let text = super::super::tests::text_content(&output);
        assert!(
            text.contains("could not be read"),
            "a present-but-unparseable file must not read as absent: {text}"
        );
        assert!(
            !text.contains("Available skills"),
            "naming substitutes invites the model to delegate one: {text}"
        );
        assert_eq!(
            session_manager
                .load_session_tree(parent_sid)
                .await
                .expect("tree")
                .len(),
            1,
            "the refusal happens before any child session exists"
        );
    }

    /// A session may only drive its own workers. This also covers a forked parent, whose copied
    /// conversation names children that are still linked to the original session.
    #[tokio::test]
    async fn test_followup_and_delete_refuse_a_session_that_is_not_the_parent() {
        let session_manager = test_session_manager().await;
        let owner = session_manager.create_session(None).await.expect("owner");
        let stranger = session_manager
            .create_session(None)
            .await
            .expect("stranger");
        let child = session_manager
            .create_child_session(owner, None, Some("{\"permission\":\"read\"}".to_string()))
            .await
            .expect("child")
            .0;

        let provider: Arc<dyn Provider> =
            Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
                text_round("should never run"),
            ]));
        let params = test_params(
            session_manager.clone(),
            Arc::new(RwLock::new(Some(stranger))),
        );

        let followup = AgentFollowupTool {
            provider,
            parent_permission: SharedPermission::new(
                Permission::Unrestricted,
                EnabledPermissions::ALL,
            ),
            tool_builder_params: params.clone(),
            in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        };
        let error = followup
            .execute(
                serde_json::json!({ "agent": child.to_string(), "prompt": "hello" }),
                CancellationToken::new(),
            )
            .await
            .expect_err("a stranger's worker must be refused");
        assert!(error.to_string().contains("belongs to this session"));

        let delete = AgentDeleteTool {
            tool_builder_params: params,
            in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        };
        assert!(
            delete
                .execute(
                    serde_json::json!({ "agent": child.to_string() }),
                    CancellationToken::new()
                )
                .await
                .is_err(),
            "and must not be deletable either"
        );
        // Still there.
        assert!(
            session_manager
                .load_session_tree(owner)
                .await
                .expect("tree")
                .iter()
                .any(|row| row.id == child)
        );
    }

    /// A worker spawned before the spec column existed has no recorded terms. Refuse rather than
    /// rebuild it from the parent, which is exactly the escalation the spec exists to prevent.
    #[tokio::test]
    async fn test_followup_refuses_a_worker_with_no_recorded_spec() {
        let session_manager = test_session_manager().await;
        let parent_sid = session_manager.create_session(None).await.expect("parent");
        let child = session_manager
            .create_child_session(parent_sid, None, None)
            .await
            .expect("child")
            .0;

        let provider: Arc<dyn Provider> =
            Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
                text_round("should never run"),
            ]));
        let followup = AgentFollowupTool {
            provider,
            parent_permission: SharedPermission::new(
                Permission::Unrestricted,
                EnabledPermissions::ALL,
            ),
            tool_builder_params: test_params(
                session_manager,
                Arc::new(RwLock::new(Some(parent_sid))),
            ),
            in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        };
        let error = followup
            .execute(
                serde_json::json!({ "agent": child.to_string(), "prompt": "hello" }),
                CancellationToken::new(),
            )
            .await
            .expect_err("a spec-less worker must be refused");
        assert!(
            error.to_string().contains("no recorded spawn terms"),
            "{error}"
        );
    }

    /// Two parallel follow-ups on one worker would hydrate the same event log and append to it
    /// independently, so the second's view of the conversation is already stale when it starts. The
    /// guard refuses rather than interleaving.
    #[test]
    fn test_followup_guard_admits_one_holder_at_a_time() {
        let in_flight: InFlightFollowups =
            Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));
        let agent = Uuid::new_v4();
        let other = Uuid::new_v4();

        let first = FollowupGuard::claim(&in_flight, agent).expect("first claim succeeds");
        assert!(
            FollowupGuard::claim(&in_flight, agent).is_none(),
            "a second follow-up on the same worker must be refused"
        );
        // A different worker is unaffected: the guard is per-agent, not a global lock.
        let sibling = FollowupGuard::claim(&in_flight, other);
        assert!(sibling.is_some());

        // Released on drop, so a turn that errors or is cancelled doesn't strand the worker.
        drop(first);
        assert!(FollowupGuard::claim(&in_flight, agent).is_some());
    }
}

#[cfg(test)]
mod clamp_enabled_permissions_doc {
    use super::*;

    /// The set a ceiling leaves behind is what `is_within` contains, not what a `<=` would keep.
    ///
    /// Written because the doc on `clamp_enabled_permissions` claimed "at or below", which is a
    /// different question on a partial order and predicts the opposite answer for the one pair this
    /// release introduced: `workspace` and `ask` exclude each other in both directions, so an `ask`
    /// ceiling drops `workspace` entirely rather than keeping it as something lower down.
    #[test]
    fn an_ask_ceiling_drops_workspace_rather_than_keeping_it_as_lower() {
        let all = EnabledPermissions::ALL;
        let under_ask = clamp_enabled_permissions(all, Permission::Ask);
        assert!(
            !under_ask.is_enabled(Permission::Workspace),
            "`workspace` is not within `ask`, so an `ask` ceiling must not leave it enabled"
        );
        assert!(under_ask.is_enabled(Permission::Ask));
        assert!(under_ask.is_enabled(Permission::Read));

        // And the reverse, which is the half a `<=` reading would get right by accident.
        let under_workspace = clamp_enabled_permissions(all, Permission::Workspace);
        assert!(!under_workspace.is_enabled(Permission::Ask));
        assert!(under_workspace.is_enabled(Permission::Workspace));

        // The empty-intersection fallback is reachable, not theoretical: a set holding only
        // `workspace` under an `ask` ceiling has nothing left, so the ceiling itself stands in.
        let only_workspace = EnabledPermissions::from_modes([Permission::Workspace])
            .expect("a one-mode set is valid");
        let collapsed = clamp_enabled_permissions(only_workspace, Permission::Ask);
        assert!(collapsed.is_enabled(Permission::Ask));
        assert!(!collapsed.is_enabled(Permission::Workspace));
    }
}
