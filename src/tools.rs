//! Tool registry and built-in tool modules. Owns the [`ToolRegistry`] type that the agent loop
//! consults to resolve tool names to executable handlers, plus the per-tool submodules (file, find,
//! grep, scratchpad, shell, etc.).

pub(crate) mod background;
pub(crate) mod context;
mod conversation;
mod file;
mod find;
mod grep;
mod load_tool;
pub(crate) mod mcp_resources;
mod memory;
mod render_image;
mod schedule;
pub(crate) mod scratchpad;
mod shell;
mod skill;
pub(crate) mod subagent;
pub(crate) mod todo;
mod util;
mod web;

use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::Arc,
};

use async_trait::async_trait;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

type DeferredSet = Arc<std::sync::RwLock<HashSet<String>>>;

/// Name of the meta-tool that loads a deferred tool's schema. Calls to this tool are scanned out of
/// the conversation to compute the per-turn active tool set; see [`extract_loaded_tool_names`].
pub const LOAD_TOOL_NAME: &str = "load_tool";

use crate::{
    error::Result,
    permission::Permission,
    provider::{ContentBlock, Message, ToolDefinition, ToolResultContent},
    session::SessionManager,
};

/// Most unused parameters one advisory will name before it starts costing more context than the
/// discovery is worth.
const MAX_ADVISED_PARAMETERS: usize = 5;

/// The universal output-redirection parameter every tool accepts, handled by the agent loop rather
/// than by the tool itself. See [`crate::tools::scratchpad::save_explicit_scratchpad_results`].
const SCRATCHPAD_PARAMETER: &str = "scratchpad";

/// The universal detach parameter, consumed by the agent loop. Where `scratchpad` changes *where* a
/// tool's output goes, this changes *when* it arrives. See [`crate::background`].
pub const BACKGROUND_PARAMETER: &str = "background";

/// Whether `background` in this call means the *tool's* `background`, not meka's.
///
/// [`offer_background`] refuses to shadow a name the tool already advertises, so a tool whose
/// schema declares `background` at all never received meka's splice and owns the name outright.
/// The consumption side has to ask the same question: an image tool with `background:
/// "transparent"`, or an exec server with a detach flag of its own, would otherwise have its
/// argument eaten before dispatch and the call quietly detached meka's way instead.
fn tool_owns_background(schema: &serde_json::Value) -> bool {
    declared_type(schema, BACKGROUND_PARAMETER).is_some()
}

/// Whether `scratchpad` in this call means the tool's own parameter.
///
/// A weaker test than [`tool_owns_background`], because it has to be: meka's builtins declare
/// `scratchpad` themselves, as a string, so its presence in a schema proves nothing. Only a
/// different declared *type* says the tool means something else by the name.
fn tool_owns_scratchpad(schema: &serde_json::Value) -> bool {
    matches!(declared_type(schema, SCRATCHPAD_PARAMETER), Some(kind) if kind != "string")
}

/// The `type` a schema declares for one property, or `None` when it declares neither.
fn declared_type<'a>(schema: &'a serde_json::Value, name: &str) -> Option<&'a str> {
    let property = schema.get("properties")?.as_object()?.get(name)?;
    // A property declared with no `type` (or a union) is still declared. Report it as the empty
    // string so `tool_owns_background` sees a declaration while `tool_owns_scratchpad`, which is
    // looking for a conflicting type, also treats it as the tool's.
    Some(
        property
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(""),
    )
}

/// Reject a call whose meka-owned parameters are the wrong type, naming what was expected.
///
/// These two are validated where every other argument is left to the tool because meka is the one
/// that consumes them, and both change what a call *does* rather than what it is called with:
/// `background` decides whether the turn waits for the result, `scratchpad` decides whether the
/// output is kept. Reading a wrong type as "absent" makes each of those a silent no-op, so the
/// model asks for a detached call and blocks for twenty minutes, or asks for its output to be saved
/// and finds nothing saved, with no way to discover why. An error costs one round trip and says
/// exactly what to send instead.
///
/// Guessing is the third option and the worst one: reading `"true"` as `true` would leave `"yes"`,
/// `1` and `"1"` still silently false, so the contract becomes "some wrong types work", which is
/// harder to write against than either strict rule.
///
/// A tool's *own* arguments are deliberately not checked here. The tool, or a remote MCP server, is
/// the authority on what it accepts, and refusing a call the server would have honoured would take
/// away capability to enforce a schema meka does not own. Those get an advisory instead; see
/// [`schema_disagreement`]. That includes these two names when the tool turns out to own them: see
/// [`tool_owns_background`] and [`tool_owns_scratchpad`].
///
/// `null` counts as absent throughout: models emit it for optional arguments they are not using,
/// and refusing that would be pedantry rather than a bug caught.
pub fn meka_parameter_error(
    input: &serde_json::Value,
    schema: &serde_json::Value,
) -> Option<String> {
    let object = input.as_object()?;
    let wrong_type = |key: &str, want: &str, value: &serde_json::Value| {
        format!(
            "Error: `{}` must be {}, but this call sent {}. Retry the call with a real {}, or \
             leave `{}` out.",
            key,
            want,
            json_type_name(value),
            want,
            key,
        )
    };

    if let Some(value) = object.get(BACKGROUND_PARAMETER)
        && !tool_owns_background(schema)
        && !value.is_null()
        && !value.is_boolean()
    {
        return Some(wrong_type(
            BACKGROUND_PARAMETER,
            "a boolean (`true` or `false`, unquoted)",
            value,
        ));
    }
    if let Some(value) = object.get(SCRATCHPAD_PARAMETER)
        && !tool_owns_scratchpad(schema)
        && !value.is_null()
        && !value.is_string()
    {
        return Some(wrong_type(
            SCRATCHPAD_PARAMETER,
            "a string naming the entry to save under",
            value,
        ));
    }
    None
}

/// What a JSON value is, for an error message aimed at whoever sent it.
fn json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

/// Split `background` out of a call's arguments, returning what the tool should receive and whether
/// it asked to be detached.
///
/// Removing rather than ignoring is the point: the property is spliced in by
/// [`ToolRegistry::definitions_active_with_loaded`] and is meka's, so forwarding it would hand an
/// MCP server a key it never advertised.
///
/// Unless the tool advertised it first, in which case the argument is passed straight through and
/// the call does not detach. [`offer_background`] declines to shadow a name a tool already uses,
/// and stripping what it declined to splice would be the same collision one step later, with the
/// tool losing an argument it does declare.
///
/// Only a real boolean detaches. A wrong type never reaches here, having been refused by
/// [`meka_parameter_error`]; `null` and absent both mean the call was not asking to detach.
pub fn take_background_flag(
    input: &serde_json::Value,
    schema: &serde_json::Value,
) -> (serde_json::Value, bool) {
    let Some(object) = input.as_object() else {
        return (input.clone(), false);
    };
    if !object.contains_key(BACKGROUND_PARAMETER) || tool_owns_background(schema) {
        return (input.clone(), false);
    }
    let mut object = object.clone();
    let detach = object
        .remove(BACKGROUND_PARAMETER)
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    (serde_json::Value::Object(object), detach)
}

/// Splice `background` into one tool's schema.
///
/// Done here, on the definitions handed to the provider, rather than declared per-tool the way
/// `scratchpad` is. One insertion point instead of seven, config-gated in one place, and it reaches
/// MCP tools too, which matters because a slow MCP call is exactly the kind worth detaching.
///
/// That last part is a deliberate exception to passing an MCP server's `input_schema` through
/// verbatim (see `crate::mcp`): the property is meka's own, and
/// `Agent::resolve_and_execute_tool` strips it from the arguments before the adapter forwards them,
/// so no server ever sees a key it did not advertise.
fn offer_background(parameters: &mut serde_json::Value) {
    let Some(object) = parameters.as_object_mut() else {
        return;
    };
    // A schema with no `properties` describes a tool taking no arguments in the shape every
    // provider expects; creating the map here would change what the tool advertises.
    let Some(properties) = object
        .get_mut("properties")
        .and_then(serde_json::Value::as_object_mut)
    else {
        return;
    };
    // Never shadow a real parameter. A server that already advertises `background` owns the name,
    // and silently overwriting its meaning would be a far worse bug than losing the feature on that
    // one tool.
    if properties.contains_key(BACKGROUND_PARAMETER) {
        return;
    }
    properties.insert(
        BACKGROUND_PARAMETER.to_string(),
        serde_json::json!({
            "type": "boolean",
            "default": false,
            "description":
                "Run this call in the background. It returns a task id immediately and its result \
                 is delivered to you when it finishes, so use it for work that takes minutes (long \
                 builds, test suites, large downloads) and not for anything you need in order to \
                 continue this turn. Track it with `task_list` and stop it with `task_cancel`.",
        }),
    );
}

/// An advisory appended to a `tool_result` when the call and the tool's advertised schema disagree,
/// or `None` when they don't.
///
/// Two disagreements are worth reporting, and they are mirror images:
///
/// - **Parameters the schema documents that the call omitted.** Only raised for a deferred tool the
///   model never loaded, because that is the case where it was working from a truncated one-line
///   summary and could not have known. This is the `send_file(as_photo)` failure: the call
///   succeeds, the default is silently wrong, and nothing anywhere says a knob existed.
/// - **Arguments the schema doesn't declare.** Servers almost universally ignore unknown keys, so
///   without this the model concludes the parameter "didn't work" rather than "was never
///   delivered", which is what a server binary older than its own source looks like from the
///   outside.
pub fn schema_disagreement(
    tool_name: &str,
    input: &serde_json::Value,
    schema: &serde_json::Value,
    report_unused: bool,
) -> Option<String> {
    let properties = schema.get("properties")?.as_object()?;
    // A schema that declares no properties at all describes nothing to disagree with. Servers do
    // ship these for tools that genuinely take arguments, so treating every key as undeclared would
    // be noise rather than a finding.
    if properties.is_empty() {
        return None;
    }
    let passed: std::collections::BTreeSet<&str> = input
        .as_object()
        .map(|object| object.keys().map(String::as_str).collect())
        .unwrap_or_default();

    let mut lines: Vec<String> = Vec::new();

    let undeclared: Vec<&str> = passed
        .iter()
        .copied()
        .filter(|key| !properties.contains_key(*key))
        // `scratchpad` and `background` are meka's own, accepted on every tool and consumed by the
        // agent loop rather than by the tool. Neither appears in the raw definition this function
        // reads, so flagging them would fire on documented features.
        .filter(|key| *key != SCRATCHPAD_PARAMETER && *key != BACKGROUND_PARAMETER)
        .collect();
    // An explicit `additionalProperties: true` means the server documented that it takes more than
    // it lists, so an undeclared key there is intentional rather than a mistake.
    let open = schema
        .get("additionalProperties")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    if !undeclared.is_empty() && !open {
        lines.push(format!(
            "Sent but not declared in this tool's schema, so it was most likely ignored: {}.",
            undeclared.join(", "),
        ));
    }

    if report_unused {
        let mut unused: Vec<String> = properties
            .iter()
            .filter(|(key, _)| !passed.contains(key.as_str()))
            .filter_map(|(key, spec)| {
                let description = spec
                    .get("description")
                    .and_then(serde_json::Value::as_str)?;
                let default = spec
                    .get("default")
                    .map(|value| format!(", default {}", value))
                    .unwrap_or_default();
                let kind = spec
                    .get("type")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("any");
                Some(format!("{} ({}{}): {}", key, kind, default, description))
            })
            .collect();
        if !unused.is_empty() {
            let dropped = unused.len().saturating_sub(MAX_ADVISED_PARAMETERS);
            unused.truncate(MAX_ADVISED_PARAMETERS);
            let tail = if dropped > 0 {
                format!("\n  (+{} more)", dropped)
            } else {
                String::new()
            };
            lines.push(format!(
                "Called without loading its schema, so these documented parameters took their \
                 defaults:\n  {}{}",
                unused.join("\n  "),
                tail,
            ));
        }
    }

    if lines.is_empty() {
        return None;
    }
    // Only point at `load_tool` when the schema is genuinely still hidden; telling a model to load
    // something it already loaded reads as noise and trains it to skim these.
    if report_unused {
        lines.push(format!(
            "Call `load_tool` with \"{}\" for the full contract.",
            tool_name,
        ));
    }
    Some(format!("\n\n[meka] {}", lines.join("\n")))
}

/// `" Did you mean `a` or `b`?"` for a tool name that isn't registered, or `""` when nothing is
/// close. The leading space lets it slot into a sentence unconditionally.
///
/// Plain edit distance is a poor fit for this namespace: omitting the `mcp__<server>__` prefix is
/// the likeliest mistake by far and puts the right answer seventeen edits away. So an exact match
/// on the final `__` segment is tried first and wins outright; distance is only the fallback for
/// genuine typos.
pub fn did_you_mean_hint<'a>(target: &str, candidates: impl Iterator<Item = &'a str>) -> String {
    const MAX_SUGGESTIONS: usize = 3;
    let needle = target.to_ascii_lowercase();
    let needle_tail = needle.rsplit("__").next().unwrap_or(needle.as_str());
    let threshold = (needle.chars().count() / 3).clamp(1, 5);

    let mut by_segment: Vec<&str> = Vec::new();
    let mut by_distance: Vec<(usize, &str)> = Vec::new();
    for candidate in candidates {
        let lowered = candidate.to_ascii_lowercase();
        if lowered == needle
            || lowered.rsplit("__").next() == Some(needle.as_str())
            || lowered == needle_tail
        {
            by_segment.push(candidate);
            continue;
        }
        let distance = edit_distance(&needle, &lowered);
        if distance <= threshold {
            by_distance.push((distance, candidate));
        }
    }

    let mut suggestions: Vec<&str> = if by_segment.is_empty() {
        by_distance.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
        by_distance.into_iter().map(|(_, name)| name).collect()
    } else {
        by_segment.sort_unstable();
        by_segment
    };
    suggestions.truncate(MAX_SUGGESTIONS);

    if suggestions.is_empty() {
        return String::new();
    }
    let rendered: Vec<String> = suggestions
        .iter()
        .map(|name| format!("`{}`", name))
        .collect();
    format!(" Did you mean {}?", rendered.join(" or "))
}

/// Levenshtein distance in chars, two-row. Only ever runs on an error path, so the quadratic cost
/// over the registry is not worth optimising away.
fn edit_distance(left: &str, right: &str) -> usize {
    let left: Vec<char> = left.chars().collect();
    let right: Vec<char> = right.chars().collect();
    if left.is_empty() {
        return right.len();
    }
    if right.is_empty() {
        return left.len();
    }
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    let mut current: Vec<usize> = vec![0; right.len() + 1];
    for (i, left_char) in left.iter().enumerate() {
        current[0] = i + 1;
        for (j, right_char) in right.iter().enumerate() {
            let substitution = usize::from(left_char != right_char);
            current[j + 1] = (previous[j + 1] + 1)
                .min(current[j] + 1)
                .min(previous[j] + substitution);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()]
}

/// Most tools one `load_tool` call will render. A batch past this is more likely a model loading a
/// whole server speculatively than a task that genuinely needs them all, and each schema is
/// unbounded in size.
pub const MAX_LOAD_TOOL_BATCH: usize = 10;

/// The tool names one `load_tool` call refers to, in order, deduplicated, and capped at
/// [`MAX_LOAD_TOOL_BATCH`].
///
/// Accepts a bare string or an array of strings: a task needing three tools off one server should
/// cost one round trip, not three. Shared with [`extract_loaded_tool_names`] so the active set is
/// derived from exactly the names the tool acted on, cap included. A name dropped by the cap must
/// not become active, since its schema was never rendered.
pub fn load_tool_names(input: &serde_json::Value) -> Vec<String> {
    let mut names = requested_tool_names(input);
    names.truncate(MAX_LOAD_TOOL_BATCH);
    names
}

/// Every distinct name the call asked for, before [`MAX_LOAD_TOOL_BATCH`] is applied. `load_tool`
/// compares this against [`load_tool_names`] so an over-long batch is reported rather than quietly
/// half-honoured, which is the same class of silent shortfall this whole advisory machinery exists
/// to eliminate.
pub fn requested_tool_names(input: &serde_json::Value) -> Vec<String> {
    let mut names: Vec<String> = match input.get("name") {
        Some(serde_json::Value::String(name)) => vec![name.clone()],
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|item| item.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    };
    let mut seen = HashSet::new();
    names.retain(|name| seen.insert(name.clone()));
    names
}

/// Walk the conversation and collect the names of tools that have been loaded via successful
/// `load_tool` calls. A `load_tool` `tool_use` block counts only when paired with a non-error
/// `tool_result` whose `tool_use_id` matches; this excludes errored loads (unknown name, malformed
/// args) and orphan `tool_use` blocks awaiting their result.
///
/// The active set is a pure function of the message slice, so the tools array sent to the Claude
/// API is deterministic given the conversation state. Resumed sessions reconstruct the exact active
/// set their suspend time had, with no out-of-band state.
///
/// A batch that resolved some names and not others returns a non-error result, so every name it
/// carried is recorded here. Harmless: a name with no registry entry matches nothing when the
/// active set is assembled (see [`ToolRegistry::definitions_active_with_loaded`]).
pub fn extract_loaded_tool_names(messages: &[Message]) -> HashSet<String> {
    let mut pending: HashMap<String, Vec<String>> = HashMap::new();
    let mut loaded: HashSet<String> = HashSet::new();
    for message in messages {
        for block in &message.content {
            match block {
                ContentBlock::ToolUse { id, name, input } if name == LOAD_TOOL_NAME => {
                    let names = load_tool_names(input);
                    if !names.is_empty() {
                        pending.insert(id.clone(), names);
                    }
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    is_error,
                    ..
                } => {
                    if let Some(loaded_names) = pending.remove(tool_use_id)
                        && !is_error
                    {
                        loaded.extend(loaded_names);
                    }
                }
                _ => {}
            }
        }
    }
    loaded
}

/// Built-in tool policy from `[tools]` in `config.toml`. Mirrors the three knobs
/// [`crate::config::McpServerConfig`] exposes for MCP tools.
#[derive(Debug, Clone, Default)]
pub struct BuiltinToolFilter {
    pub allowed: Option<HashSet<String>>,
    pub disabled: HashSet<String>,
    pub permission_overrides: HashMap<String, Permission>,
}

impl BuiltinToolFilter {
    pub fn from_config(
        allowed: Option<Vec<String>>,
        disabled: Vec<String>,
        permission_overrides: HashMap<String, Permission>,
    ) -> Self {
        // Empty allow-list → None so `admits` treats it as "no restriction".
        let allowed = allowed.and_then(|list| {
            if list.is_empty() {
                None
            } else {
                Some(list.into_iter().collect())
            }
        });
        Self {
            allowed,
            disabled: disabled.into_iter().collect(),
            permission_overrides,
        }
    }

    pub fn admits(&self, name: &str) -> bool {
        if self.denies(name) {
            return false;
        }
        match &self.allowed {
            Some(list) => list.contains(name),
            None => true,
        }
    }

    /// The block-list half alone, for tools that `allowed_tools` was never able to reach.
    ///
    /// `allowed_tools` is exhaustive: naming five tools removes everything else. Anyone who wrote
    /// one before the MCP meta-tools were filterable wrote it against a world where those seven
    /// registered unconditionally, so applying the allow-list to them now would delete
    /// `mcp_resource_read` and friends from working installations on upgrade, with nothing in the
    /// config to explain it. `disabled_tools` has no such problem: naming a tool there has always
    /// meant "remove this one", so honouring it is what the user already asked for.
    pub fn denies(&self, name: &str) -> bool {
        self.disabled.contains(name)
    }
}

/// What a sub-agent registry refuses to register, from `[subagents]` unioned with the
/// `deny_servers` / `deny_tools` of the `agent_spawn` call that built it. Empty on the primary
/// agent's registry, which is what makes this a sub-agent concept rather than a second `[tools]`
/// filter.
///
/// Denials only ever accumulate ([`Self::union`]): a nested `agent_spawn` inherits its parent's
/// effective set and adds to it, so no depth of nesting can hand a descendant something config took
/// away.
#[derive(Debug, Clone, Default)]
pub struct ToolDenials {
    servers: HashSet<String>,
    tools: HashSet<String>,
}

impl ToolDenials {
    pub fn new(
        servers: impl IntoIterator<Item = String>,
        tools: impl IntoIterator<Item = String>,
    ) -> Self {
        Self {
            servers: servers.into_iter().collect(),
            tools: tools.into_iter().collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.servers.is_empty() && self.tools.is_empty()
    }

    pub fn denies_server(&self, server_name: &str) -> bool {
        self.servers.contains(server_name)
    }

    /// A tool is denied by its own registry name, or because it belongs to a denied server. The
    /// second arm is what makes `disabled_servers` cover tools registered through paths that never
    /// see a server name, so the two keys can't disagree.
    pub fn denies_tool(&self, name: &str) -> bool {
        if self.tools.contains(name) {
            return true;
        }
        match server_of_tool(name) {
            Some(server) => self.servers.contains(server),
            None => false,
        }
    }

    /// Everything either set denies. Used when a `agent_spawn` call adds to what config already
    /// denied, and again when the child's own `agent_spawn` inherits the result.
    pub fn union(&self, other: &Self) -> Self {
        Self {
            servers: self.servers.union(&other.servers).cloned().collect(),
            tools: self.tools.union(&other.tools).cloned().collect(),
        }
    }

    /// Sorted, for the persisted spawn spec and for tests. `HashSet` iteration order is not stable,
    /// and a spec that round-trips differently every time is one nobody can diff.
    pub fn server_list(&self) -> Vec<String> {
        let mut names: Vec<String> = self.servers.iter().cloned().collect();
        names.sort();
        names
    }

    pub fn tool_list(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tools.iter().cloned().collect();
        names.sort();
        names
    }
}

/// Server name out of a namespaced MCP tool name, or `None` for a built-in. Server names cannot
/// contain `__` (`mcp::sanitize::normalize_server_name`), so the first occurrence splits them.
pub fn server_of_tool(tool_name: &str) -> Option<&str> {
    let rest = tool_name.strip_prefix("mcp__")?;
    rest.split_once("__").map(|(server, _tool)| server)
}

/// The MCP resource and prompt meta-tools, which `mcp_resources::register_all` registers directly
/// rather than through `register_builtin`. Deniable via `disabled_tools`, but deliberately outside
/// `allowed_tools` -- see [`BuiltinToolFilter::denies`].
pub const MCP_META_TOOL_NAMES: &[&str] = &[
    "mcp_prompt_get",
    "mcp_prompt_list",
    "mcp_resource_list",
    "mcp_resource_read",
    "mcp_resource_subscribe",
    "mcp_resource_unsubscribe",
    "mcp_resource_updates_list",
];

/// Canonical built-in names for the stale-entry warning pass, sorted.
///
/// Every name a user may legitimately put in `[tools]` or `[subagents]`, which is wider than
/// [`ToolRegistry::build_default`]'s own list: the conditionally-registered families
/// (`schedule_*`, `task_*`) and the tools registered outside `register_builtin` (the MCP
/// meta-tools) are all deniable, so leaving them out made the warning fire on correct entries.
/// Update when adding any new built-in.
pub const BUILTIN_TOOL_NAMES: &[&str] = &[
    "agent_delete",
    "agent_followup",
    "agent_list",
    "agent_spawn",
    "context_check",
    "context_compact",
    "conversation_read",
    "conversation_search",
    "edit_file",
    "execute_command",
    "fetch_url",
    "find_files",
    "load_tool",
    "mcp_prompt_get",
    "mcp_prompt_list",
    "mcp_resource_list",
    "mcp_resource_read",
    "mcp_resource_subscribe",
    "mcp_resource_unsubscribe",
    "mcp_resource_updates_list",
    "memory_delete",
    "memory_read",
    "memory_search",
    "memory_write",
    "read_file",
    "render_image",
    "schedule_cancel",
    "schedule_create",
    "schedule_list",
    "scratchpad_delete",
    "scratchpad_edit",
    "scratchpad_list",
    "scratchpad_load_file",
    "scratchpad_merge",
    "scratchpad_read",
    "scratchpad_rename",
    "scratchpad_save_file",
    "scratchpad_write",
    "search_contents",
    "search_web",
    "skill",
    "task_cancel",
    "task_list",
    "todo",
    "write_file",
];

/// Tools a checkpoint turn may reach, on top of the `context_replace` it is given.
///
/// An allow-list, because the guiding rule is that **a checkpoint can save, but not act**: the turn
/// exists to preserve what already happened, not to do more work while the window is about to be
/// rewritten. So no `execute_command`, no `write_file` / `edit_file`, no `agent_spawn`, no
/// `schedule_*`, and no MCP tools.
///
/// The read tools *are* here, because deciding what is worth keeping sometimes means checking
/// something first. The two delete tools are not: deleting is not saving, and a mistaken
/// `memory_delete` in an unattended checkpoint is unrecoverable, while the agent can still delete
/// on any ordinary turn.
///
/// Kept sorted, and every entry must exist in [`BUILTIN_TOOL_NAMES`]; a test enforces both.
pub const CHECKPOINT_TOOL_NAMES: &[&str] = &[
    "conversation_read",
    "conversation_search",
    "find_files",
    "memory_read",
    "memory_search",
    "memory_write",
    "read_file",
    "scratchpad_edit",
    "scratchpad_list",
    "scratchpad_read",
    "scratchpad_write",
    "search_contents",
    "todo",
];

/// Warn (never fail) on `[tools]` entries that don't match any known built-in. Mirrors MCP's
/// `warn_on_stale_tool_config()`.
pub fn warn_on_stale_builtin_tool_config(filter: &BuiltinToolFilter) {
    let known: HashSet<&str> = BUILTIN_TOOL_NAMES.iter().copied().collect();
    if let Some(allowed) = filter.allowed.as_ref() {
        for name in allowed {
            if !known.contains(name.as_str()) {
                tracing::warn!(
                    "[tools].allowed_tools entry '{}' doesn't match any built-in tool",
                    name
                );
            } else if MCP_META_TOOL_NAMES.contains(&name.as_str()) {
                // These register outside the allow-list (see
                // `ToolRegistry::admits_infrastructure`), so naming one here is
                // inert. Worth saying: the entry looks like it is keeping the tool,
                // and before the meta-tools were listed in `BUILTIN_TOOL_NAMES` at all this
                // case at least warned as unrecognised.
                tracing::warn!(
                    "[tools].allowed_tools entry '{}' has no effect: the MCP meta-tools are not \
                     subject to the allow-list. Use disabled_tools to remove one.",
                    name
                );
            }
        }
    }
    for name in &filter.disabled {
        if !known.contains(name.as_str()) {
            tracing::warn!(
                "[tools].disabled_tools entry '{}' doesn't match any built-in tool",
                name
            );
        }
    }
    for name in filter.permission_overrides.keys() {
        if !known.contains(name.as_str()) {
            tracing::warn!(
                "[tools.tool_permissions] entry '{}' doesn't match any built-in tool",
                name
            );
        }
    }
}

/// Warn (never fail) on `[subagents]` entries that match nothing. A typo here denies nothing at all
/// while reading as a restriction, which is the worst of both: the user believes a worker cannot
/// reach a server it can reach.
///
/// Checked against configured server names rather than advertised tool names, because this runs at
/// startup before any server has completed its handshake. So `mcp__notion__craete_page` with
/// `notion` configured is accepted here and never warned about: nothing else checks it either
/// (`mcp::warn_on_stale_tool_config` only inspects the per-server `[[mcp.servers]]` lists, against
/// raw un-namespaced names). Catching it would mean deferring this pass until every server has
/// connected, which is a different shape of startup. Server names, which are the coarse lever and
/// the more consequential half, are checked exactly.
pub fn warn_on_stale_subagent_config(denials: &ToolDenials, configured_servers: &[String]) {
    let servers: HashSet<&str> = configured_servers.iter().map(String::as_str).collect();
    for name in denials.server_list() {
        if !servers.contains(name.as_str()) {
            tracing::warn!(
                "[subagents].disabled_servers entry '{}' doesn't match any configured MCP server",
                name
            );
        }
    }
    let known: HashSet<&str> = BUILTIN_TOOL_NAMES.iter().copied().collect();
    for name in denials.tool_list() {
        if known.contains(name.as_str()) {
            continue;
        }
        if server_of_tool(&name).is_some_and(|server| servers.contains(server)) {
            continue;
        }
        tracing::warn!(
            "[subagents].disabled_tools entry '{}' matches no built-in tool and no configured MCP \
             server",
            name
        );
    }
}

/// What a file looked like when a tool last read it, so a later `edit_file` can tell "you never
/// read this" from "this moved under you".
///
/// Metadata rather than a content hash: both fields come off the `stat` the read already performs,
/// so recording them is free, and any edit that changes a file changes at least one of them in
/// practice. A hash would be exact but would mean re-reading every file on every edit.
/// Compared against whatever the *same source* says at edit time, which is the whole reason this is
/// an enum. A file read through an editor's hosted filesystem and a file read off the disk are two
/// different documents that happen to share a path, and checking one against the other produces a
/// false alarm every time the user saves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadStamp {
    /// Read from the filesystem. Metadata rather than a content hash: both fields come off the
    /// `stat` the read already performs, so recording them is free, and any edit that changes a
    /// file changes at least one of them in practice.
    ///
    /// `mtime` is `None` on the platforms and filesystems that don't report one; two `None` stamps
    /// compare equal, so such a filesystem degrades to length-only detection rather than to a false
    /// "changed on disk" on every edit.
    Disk {
        mtime: Option<std::time::SystemTime>,
        len: u64,
    },
    /// Served by the frontend's hosted filesystem (see [`crate::frontend::Frontend`]), which is the
    /// editor's document and not the bytes on disk.
    ///
    /// Fingerprints what was served, because that is the only thing the disk cannot answer for. An
    /// editor serves its buffer for any file it owns, saved or not, so a disk comparison here is
    /// wrong in both directions: it fires when the user saves (the buffer did not change) and stays
    /// silent when the user types (the disk did not change).
    Delegated { fingerprint: u64 },
}

impl ReadStamp {
    pub fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        Self::Disk {
            mtime: metadata.modified().ok(),
            len: metadata.len(),
        }
    }

    /// The stamp for a path, or `None` when it can't be stated.
    pub async fn of_path(path: &std::path::Path) -> Option<Self> {
        tokio::fs::metadata(path)
            .await
            .ok()
            .map(|metadata| Self::from_metadata(&metadata))
    }

    /// The stamp for text the frontend served, or that meka wrote back through it.
    pub fn of_delegated(text: &str) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        Self::Delegated {
            fingerprint: hasher.finish(),
        }
    }
}

/// Files a tool has read this session, and what they looked like at the time.
///
/// A set of paths would answer "has this been read", which is the question `edit_file` used to ask.
/// It could not answer "is that read still valid", so an edit against a file rewritten in between
/// (by a shell command, a concurrent agent, or the user's editor) silently clobbered.
pub type ReadTracker = Arc<RwLock<HashMap<PathBuf, ReadStamp>>>;

#[derive(Debug, Default)]
pub struct ToolOutput {
    pub content: Vec<ToolResultContent>,
    pub is_error: bool,
    /// When `persist_oversized_results` has to spill to the scratchpad, use this name instead of
    /// the caller-supplied tool name. Set by MCP tool adapters so the persisted blob is namespaced
    /// as `mcp_<server>_<remote_tool>` for easier debugging.
    pub scratchpad_hint: Option<String>,
    /// Tool-specific structured side-channel for frontends that know how to render it (e.g. ACP's
    /// `diff` content block). Tools that don't produce extra structure leave this as `None`; the
    /// regular `content` text remains the source of truth for the model.
    pub frontend_metadata: Option<crate::frontend::ToolOutputMetadata>,
}

impl ToolOutput {
    pub fn text(content: String, is_error: bool) -> Self {
        Self {
            content: vec![ToolResultContent::Text { text: content }],
            is_error,
            scratchpad_hint: None,
            frontend_metadata: None,
        }
    }

    /// Attach structured frontend metadata to an existing output, e.g. the pre/post text from a
    /// successful `edit_file`. Chains after any other builder so the call site reads as
    /// `ToolOutput::text(...).with_metadata(diff)`.
    #[must_use]
    pub fn with_metadata(mut self, metadata: crate::frontend::ToolOutputMetadata) -> Self {
        self.frontend_metadata = Some(metadata);
        self
    }

    /// Append harness-authored text to the model-visible result, extending the trailing text block
    /// rather than adding one. An MCP result can be a mix of text and images, and a bare text block
    /// tacked onto the end of that is easy to read as part of the tool's own output.
    pub fn append_notice(&mut self, notice: &str) {
        match self.content.last_mut() {
            Some(ToolResultContent::Text { text }) => text.push_str(notice),
            _ => self.content.push(ToolResultContent::Text {
                text: notice.trim_start().to_string(),
            }),
        }
    }
}

tokio::task_local! {
    /// The provider's `tool_use_id` for the call executing on this task, scoped by
    /// `Agent::resolve_and_execute_tool` around the tool's own execution, inline and backgrounded
    /// alike. Approval is deliberately outside it: `Agent::request_approval` runs before a
    /// `background` call detaches, and nothing on that path reads this (ACP mints its own
    /// `perm-<uuid>` for the permission request).
    /// A task-local rather than a [`Tool::execute`] parameter because exactly
    /// one tool needs it -- `agent_spawn`, to route its sub-agent's activity back into the tool
    /// call the client is already displaying -- and threading it through every implementor to
    /// serve one of them is the wrong trade.
    ///
    /// Tool dispatch is concurrent but each call gets its own task, so the value is per-call.
    static CURRENT_TOOL_CALL_ID: String;
}

/// The `tool_use_id` of the in-flight tool call, or `None` outside one (tests, direct tool
/// construction). Callers must treat `None` as "no correlation available" rather than guessing.
pub(crate) fn current_tool_call_id() -> Option<String> {
    CURRENT_TOOL_CALL_ID.try_with(|id| id.clone()).ok()
}

/// Scope `id` as the in-flight tool call for the duration of `fut`.
pub(crate) async fn with_tool_call_id<F, T>(id: String, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    CURRENT_TOOL_CALL_ID.scope(id, fut).await
}

/// A callable tool surfaced to the model. Built-in tools live under `src/tools/`; MCP tools are
/// wrapped at registration time. Implementors must be safe to invoke concurrently; the dispatch
/// loop runs all tool calls in a single assistant message in parallel via `join_all`.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Schema surfaced to the model (name + description + JSON-schema for parameters). Called once
    /// per registry build, not per call.
    fn definition(&self) -> ToolDefinition;
    /// Lowest permission level that may invoke this tool. The dispatch loop short-circuits with a
    /// "permission denied" tool error when the current level is below this.
    fn required_permission(&self) -> Permission;
    /// Run the tool. Long-running implementations must observe `cancellation` (e.g. via
    /// `tokio::select!`) so a user interrupt or turn-level abort unblocks promptly.
    async fn execute(
        &self,
        input: serde_json::Value,
        cancellation: CancellationToken,
    ) -> Result<ToolOutput>;
}

type ToolSet = Arc<std::sync::RwLock<Vec<Arc<dyn Tool>>>>;

/// Tool registry. Backed by an `Arc<RwLock<Vec<Arc<dyn Tool>>>>` so MCP notification handlers can
/// swap a server's tools in place on `tools/list_changed`. Individual registrations only hold the
/// write lock briefly; dispatch clones the matching `Arc<dyn Tool>` out of the lock before awaiting
/// `execute`, so no lock is held across `.await`.
#[derive(Clone)]
pub struct ToolRegistry {
    tools: ToolSet,
    deferred: DeferredSet,
    /// Per-tool overrides from `[tools.tool_permissions]`. Immutable after construction so the
    /// cached system-prompt prefix stays byte-stable across `/permission` toggles.
    permission_overrides: Arc<HashMap<String, Permission>>,
    /// Built-in allow/block-list. MCP tools have their own per-server filtering in `src/mcp.rs`
    /// and bypass this.
    builtin_filter: Arc<BuiltinToolFilter>,
    /// `[subagents]` denials, empty on the primary agent's registry. Unlike `builtin_filter` this
    /// covers MCP tools too, and is read back out by
    /// [`crate::mcp::McpClientManager::install_tools_on`] and
    /// [`mcp_resources::register_all`] so the registry stays the single place the answer lives.
    denials: Arc<ToolDenials>,
    /// Files read this session, shared with the file tools so `edit_file` can require a prior
    /// read. Cleared on conversation compaction.
    read_tracker: ReadTracker,
    /// Back-reference to the MCP manager, filled in by
    /// [`crate::mcp::McpClientManager::attach_registry`] for session registries and
    /// [`crate::mcp::McpClientManager::install_tools_on`] for sub-agent ones. Both `load_tool` and
    /// `Agent::resolve_and_execute_tool` read it to distinguish "no such tool" from "that tool's
    /// server isn't connected". `Weak` because the manager owns the registry list, not the other
    /// way round.
    mcp_manager: Arc<std::sync::OnceLock<std::sync::Weak<crate::mcp::McpClientManager>>>,
    /// Whether the `background` property is spliced into the schemas this registry hands the
    /// provider. Off leaves it out entirely rather than refusing it at dispatch: a parameter the
    /// model can see but cannot use is worse than one it cannot see.
    background_enabled: Arc<std::sync::atomic::AtomicBool>,
}

impl ToolRegistry {
    /// Empty registry with the default filter: no built-ins, no MCP tools. Used by out-of-band CLI
    /// commands that spin up a manager for a single RPC (`meka mcp reconnect`, `meka mcp tools`)
    /// and don't need a populated registry. The `dead_code` allow keeps the helper available for
    /// future CLI subcommands that need a throwaway registry.
    #[allow(dead_code)]
    pub(crate) fn new() -> Self {
        Self::new_with_filter(BuiltinToolFilter::default())
    }

    fn new_with_filter(filter: BuiltinToolFilter) -> Self {
        Self::new_with_filter_and_denials(filter, ToolDenials::default())
    }

    fn new_with_filter_and_denials(filter: BuiltinToolFilter, denials: ToolDenials) -> Self {
        let overrides = filter.permission_overrides.clone();
        Self {
            tools: Arc::new(std::sync::RwLock::new(Vec::new())),
            deferred: Arc::new(std::sync::RwLock::new(HashSet::new())),
            permission_overrides: Arc::new(overrides),
            builtin_filter: Arc::new(filter),
            denials: Arc::new(denials),
            mcp_manager: Arc::new(std::sync::OnceLock::new()),
            background_enabled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            read_tracker: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// What this registry refuses to register. Read by the MCP installation paths so they can skip
    /// a denied server's discovery round trip and its resource/prompt meta-tools, rather than each
    /// caller threading its own copy of the deny lists.
    pub(crate) fn denials(&self) -> &ToolDenials {
        &self.denials
    }

    /// Whether this registry will accept a tool by that name: it has to pass both the `[tools]`
    /// filter and the sub-agent deny list.
    ///
    /// The single predicate behind [`Self::register_builtin`] *and* the two paths that call
    /// [`Self::register`] directly ([`mcp_resources::register_all`] and
    /// [`subagent::register_subagent_tools`]). Those two exist because their tools are built from
    /// collaborators the generic builder doesn't have, and routing them past `register_builtin`
    /// used to mean routing them past the filters with it: naming an MCP meta-tool or
    /// `agent_spawn` in a deny list did nothing at all. Any future direct registration should
    /// come through here too.
    pub(crate) fn admits(&self, name: &str) -> bool {
        if !self.builtin_filter.admits(name) {
            tracing::info!("skipping tool '{}' (disabled by [tools] config)", name);
            return false;
        }
        self.not_denied(name)
    }

    /// Like [`Self::admits`] but consulting only `[tools].disabled_tools`, not `allowed_tools`.
    ///
    /// For the MCP meta-tools, which an exhaustive `allowed_tools` never used to reach. See
    /// [`BuiltinToolFilter::denies`] for why widening it now would break working configs.
    pub(crate) fn admits_infrastructure(&self, name: &str) -> bool {
        if self.builtin_filter.denies(name) {
            tracing::info!("skipping tool '{}' (disabled by [tools] config)", name);
            return false;
        }
        self.not_denied(name)
    }

    fn not_denied(&self, name: &str) -> bool {
        if self.denials.denies_tool(name) {
            tracing::info!("skipping tool '{}' for sub-agent (denied by config)", name);
            return false;
        }
        true
    }

    /// Clear the read-tracker. Called on conversation compaction: the model's context is reset, so
    /// a follow-up `edit_file` should re-read the file rather than trust a pre-compaction read.
    pub async fn clear_read_tracker(&self) {
        self.read_tracker.write().await.clear();
    }

    /// Identity check by inner `Arc` pointer. `ToolRegistry` is `Clone` over its inner
    /// `Arc<RwLock<Vec<Arc<dyn Tool>>>>`, so cloned registries match. Used by
    /// [`crate::mcp::McpClientManager::detach_registry`] to find the right entry when a session
    /// closes.
    pub fn same_inner(a: &Self, b: &Self) -> bool {
        Arc::ptr_eq(&a.tools, &b.tools)
    }

    /// Register a tool. Returns an error if another tool with the same name is already registered.
    /// Callers that know the tool is unique (e.g. core builtins) may `.expect()` the result; MCP
    /// registration should log and continue so one bad server can't break startup.
    pub fn register(&self, tool: Arc<dyn Tool>) -> Result<()> {
        let name = tool.definition().name.clone();
        let mut tools = self
            .tools
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if tools.iter().any(|t| t.definition().name == name) {
            return Err(crate::error::MekaError::ToolRegistration {
                message: format!("tool name '{}' is already registered", name),
            });
        }
        tools.push(tool);
        Ok(())
    }

    /// Replace every tool whose name starts with `mcp__<server_name>__` with the supplied set. Used
    /// by `MekaClientHandler::on_tool_list_changed` to hot-swap a server's tools without restarting
    /// the agent. Deferred markers for removed tool names are cleared so the registry's deferred
    /// set doesn't grow unbounded.
    pub fn replace_server_tools(&self, server_name: &str, new_tools: Vec<Arc<dyn Tool>>) {
        // Filter here rather than only at the call sites: a `tools/list_changed` notification
        // arriving mid-run goes through this same path, so a denied server would otherwise walk its
        // tools back in the moment it re-advertised them.
        let new_tools: Vec<Arc<dyn Tool>> = if self.denials.is_empty() {
            new_tools
        } else {
            new_tools
                .into_iter()
                .filter(|tool| !self.denials.denies_tool(&tool.definition().name))
                .collect()
        };
        let prefix = format!("mcp__{}__", server_name);
        let mut tools = self
            .tools
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let removed: Vec<String> = tools
            .iter()
            .filter(|t| t.definition().name.starts_with(&prefix))
            .map(|t| t.definition().name)
            .collect();
        tools.retain(|t| !t.definition().name.starts_with(&prefix));
        tools.extend(new_tools);
        drop(tools);

        if !removed.is_empty() {
            let mut deferred = self
                .deferred
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for name in &removed {
                deferred.remove(name);
            }
        }
    }

    /// Register just `load_tool`, for tests that exercise its unfindable-name path against a real
    /// registry + manager pair without building the whole builtin set.
    #[cfg(test)]
    pub(crate) fn register_load_tool_for_test(&self) {
        self.register_builtin(Arc::new(load_tool::LoadToolTool {
            tools: Arc::downgrade(&self.tools),
            deferred: Arc::downgrade(&self.deferred),
            mcp_manager: Arc::downgrade(&self.mcp_manager),
        }));
    }

    /// Record the manager whose servers back this registry's MCP tools. Called by
    /// [`crate::mcp::McpClientManager::attach_registry`] (session registries) and
    /// [`crate::mcp::McpClientManager::install_tools_on`] (sub-agent registries); a second call is
    /// ignored, since a registry is only ever wired to one manager.
    pub(crate) fn set_mcp_manager(&self, manager: std::sync::Weak<crate::mcp::McpClientManager>) {
        // `set` fails only when the slot is already filled, which is the documented no-op.
        let _already_set = self.mcp_manager.set(manager);
    }

    /// The manager recorded by [`Self::set_mcp_manager`], if it is still alive.
    ///
    /// The registry is the right place to ask "which manager owns these tools": a sub-agent's
    /// [`crate::agent::Agent`] has no `mcp_manager` of its own, but its registry does.
    pub(crate) fn mcp_manager(&self) -> Option<Arc<crate::mcp::McpClientManager>> {
        self.mcp_manager.get()?.upgrade()
    }

    /// Mark a tool as deferred. Deferred tools live in the registry but are hidden from the
    /// per-turn tools array until the model explicitly loads them via the `load_tool` meta-tool.
    /// Discoverability is preserved by the `[Tool discovery]` section of the per-turn `<context>`
    /// block (built from `tool_catalogue()`), and the active set is recomputed per turn from the
    /// conversation, not from registry state.
    pub fn mark_deferred(&self, name: &str) {
        self.deferred
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(name.to_string());
    }

    /// Offer `background` on every tool this registry hands the provider.
    ///
    /// Set once, from `[background] enabled`. Sub-agent registries are deliberately left off: a
    /// sub-agent's session ends with the single turn that spawned it, so a task outliving that turn
    /// would have no conversation to report back into.
    pub fn enable_background(&self) {
        self.background_enabled
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn background_enabled(&self) -> bool {
        self.background_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Every registered tool name. Unlike [`Self::tool_catalogue`] this clones no descriptions and
    /// imposes no order, which is all a "did you mean" lookup needs.
    pub fn registered_tool_names(&self) -> Vec<String> {
        self.tools
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .map(|tool| tool.definition().name)
            .collect()
    }

    /// Whether `name` is deferred, i.e. whether the model has to have loaded it to have ever seen
    /// its schema.
    pub fn is_deferred(&self, name: &str) -> bool {
        self.deferred
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains(name)
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .find(|tool| tool.definition().name == name)
            .cloned()
    }

    /// Effective required permission: override wins, else the tool's hardcoded
    /// `Tool::required_permission()`. `None` if not registered.
    pub fn required_permission_for(&self, name: &str) -> Option<Permission> {
        if let Some(permission) = self.permission_overrides.get(name) {
            return Some(*permission);
        }
        self.get(name).map(|tool| tool.required_permission())
    }

    /// Returns tool definitions for the API call, excluding deferred tools. Permission-filtered
    /// view, used by sub-agents which run at a fixed permission. The main agent uses
    /// [`Self::definitions_active_with_loaded`] so the tools array remains byte-identical across
    /// mid-session `/permission` toggles, keeping the Claude prompt cache warm on subsequent turns.
    pub fn definitions_for_permission(&self, permission: Permission) -> Vec<ToolDefinition> {
        let deferred = self
            .deferred
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.tools
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .filter(|tool| {
                let definition = tool.definition();
                let required = self
                    .permission_overrides
                    .get(&definition.name)
                    .copied()
                    .unwrap_or_else(|| tool.required_permission());
                permission.allows(required) && !deferred.contains(&definition.name)
            })
            .map(|tool| tool.definition())
            .collect()
    }

    /// Slice-based convenience wrapper for tests: composes [`extract_loaded_tool_names`] with
    /// [`Self::definitions_active_with_loaded`]. Production code goes through the events-aware path
    /// (see [`crate::conversation::extract_loaded_tool_names_from_events`]) so
    /// `Event::CompactBoundary::loaded_tools_snapshot` survives across compaction; a slice-only
    /// scan loses the snapshot.
    #[cfg(test)]
    pub fn definitions_active(&self, messages: &[Message]) -> Vec<ToolDefinition> {
        // Sorted so the slice-only path is deterministic; the events-aware path preserves real load
        // order, which is what the append-only cache prefix depends on.
        let mut loaded: Vec<String> = extract_loaded_tool_names(messages).into_iter().collect();
        loaded.sort();
        self.definitions_active_with_loaded(&loaded)
    }

    /// Returns every active tool definition regardless of the caller's current permission. The
    /// active set is the union of non-deferred tools and deferred tools whose schema has been
    /// loaded via the `load_tool` meta-tool. `loaded` is computed by the caller (via
    /// [`crate::conversation::extract_loaded_tool_names_from_events`] for the agent loop,
    /// [`extract_loaded_tool_names`] for tests).
    ///
    /// Blocked calls are rejected at dispatch; keeping the tools array permission-independent is
    /// what preserves the prompt cache prefix across `/permission` toggles (breakpoint 3 in the
    /// Claude provider's cache layout).
    /// `loaded` must be in load order (see
    /// [`crate::conversation::extract_loaded_tool_names_from_events`]).
    ///
    /// Emits always-active tools first in registration order, then loaded-deferred tools in the
    /// order they were loaded. Both halves grow only at the tail as a session proceeds, which is
    /// what keeps this array a stable cache prefix: `load_tool` only ever appends to the
    /// conversation, so the second half only ever gains entries.
    ///
    /// Filtering the registration-ordered list in place would *not* be append-only. With tools
    /// registered `[a, b]`, loading `b` then `a` yields `[b]` and then `[a, b]`, inserting `a`
    /// ahead of `b`. The array precedes every message, so that edit re-caches the whole
    /// conversation.
    ///
    /// The one thing that does disturb the first half is [`Self::replace_server_tools`], which
    /// removes an MCP server's tools and re-appends the new set at the tail. That genuinely
    /// rewrites part of the array and cannot be avoided when a server withdraws a tool, but it is
    /// confined here: the system prompt ahead of the array is unaffected, so the blast radius is
    /// the array rather than the whole conversation.
    pub fn definitions_active_with_loaded(&self, loaded: &[String]) -> Vec<ToolDefinition> {
        let deferred = self
            .deferred
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let tools = self
            .tools
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let mut definitions: Vec<ToolDefinition> = tools
            .iter()
            .filter(|tool| !deferred.contains(&tool.definition().name))
            .map(|tool| tool.definition())
            .collect();

        for name in loaded {
            // Names that aren't deferred are already in the active half; a name with no registry
            // entry was loaded and later removed (an MCP `tools/list_changed` swap) and simply
            // drops out.
            if !deferred.contains(name) {
                continue;
            }
            if let Some(tool) = tools.iter().find(|tool| &tool.definition().name == name) {
                definitions.push(tool.definition());
            }
        }

        if self.background_enabled() {
            for definition in &mut definitions {
                // `context_compact` is excluded: it does not do work, it parks a request that
                // `run_turn` drains once the tool loop ends. Detaching it would race that drain -
                // the spawned call may set the request after the drain has already run - leaving it
                // to fire against the next, unrelated turn, which is the exact hazard the drain is
                // written to prevent. It would also persist a `background_tasks` row for an
                // operation that takes microseconds.
                if definition.name == "context_compact" {
                    continue;
                }
                offer_background(&mut definition.parameters);
            }
        }

        definitions
    }

    /// Returns (name, description, required_permission, is_deferred) for every registered tool.
    /// Drives the permission-independent `[Available tools]` catalogue plus the per-turn
    /// `[Permission context]` block that names currently-blocked tools. Sorted by (name) for
    /// deterministic output.
    pub fn tool_catalogue(&self) -> Vec<(String, String, Permission, bool)> {
        let deferred = self
            .deferred
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut entries: Vec<(String, String, Permission, bool)> = self
            .tools
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .map(|tool| {
                let def = tool.definition();
                let is_deferred = deferred.contains(&def.name);
                let required = self
                    .permission_overrides
                    .get(&def.name)
                    .copied()
                    .unwrap_or_else(|| tool.required_permission());
                (def.name, def.description, required, is_deferred)
            })
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        entries
    }

    /// Register the core tools shared by the main agent and sub-agents:
    /// file I/O, search, web, and shell execution.
    #[allow(clippy::too_many_arguments)]
    fn register_core_tools(
        &self,
        web_client_config: &crate::config::WebClientConfig,
        shared_permission: crate::permission::SharedPermission,
        sandbox_enabled: bool,
        sandbox_capability: crate::sandbox::SandboxCapability,
        sandbox_backend: crate::config::SandboxBackend,
        backend_probe: crate::sandbox::BackendProbe,
        cwd: crate::agent::SharedCwd,
        roots: crate::agent::SharedRoots,
        frontend: Arc<dyn crate::frontend::Frontend>,
    ) -> Result<()> {
        let read_tracker = self.read_tracker.clone();
        self.register_builtin(Arc::new(file::ReadFileTool {
            read_tracker: read_tracker.clone(),
            cwd: cwd.clone(),
            frontend: Arc::clone(&frontend),
        }));
        self.register_builtin(Arc::new(file::EditFileTool {
            read_tracker: read_tracker.clone(),
            cwd: cwd.clone(),
            frontend: Arc::clone(&frontend),
        }));
        self.register_builtin(Arc::new(file::WriteFileTool {
            read_tracker,
            cwd: cwd.clone(),
            frontend: Arc::clone(&frontend),
        }));
        self.register_builtin(Arc::new(find::FindFilesTool {
            cwd: cwd.clone(),
            roots: roots.clone(),
        }));
        self.register_builtin(Arc::new(grep::SearchContentsTool {
            cwd: cwd.clone(),
            roots,
        }));
        // A malformed proxy URL or unreadable CA file surfaces as a startup error rather than
        // silently falling back to an unconfigured client (which would ignore the user's intent).
        let web_client = web::build_web_client(web_client_config)?;
        self.register_builtin(Arc::new(web::FetchUrlTool {
            client: web_client.clone(),
        }));
        self.register_builtin(Arc::new(web::WebSearchTool { client: web_client }));
        self.register_builtin(Arc::new(shell::ExecuteCommandTool {
            sandbox_capability,
            sandbox_backend,
            backend_probe,
            shared_permission,
            sandbox_enabled,
            cwd,
            frontend,
        }));
        Ok(())
    }

    /// Register a builtin. Collisions panic (programmer error). Tools rejected by the `[tools]`
    /// filter are silently skipped.
    fn register_builtin(&self, tool: Arc<dyn Tool>) {
        let name = tool.definition().name;
        if !self.admits(&name) {
            return;
        }
        // A collision here means two builtin tools share a name (a coding bug, not a runtime
        // condition). Panic so the test suite catches it on the first build instead of letting the
        // second registration silently fail.
        #[allow(clippy::expect_used)]
        self.register(tool).expect("builtin tool name collision");
    }

    /// Register the session-scoped tools (load_tool, skill, render_image, todo_*, scratchpad_*) on
    /// the registry. Shared between [`Self::build_default`] and [`Self::build_for_subagent`] so
    /// adding a new such tool to the parent automatically gives it to sub-agents too. Todo-list
    /// rendering is the [`crate::frontend::Frontend`]'s concern now, not the tool's.
    ///
    /// `parent_session_id` + `inherited_scratchpad_names` configure read-only scratchpad
    /// inheritance for sub-agents. Both are `None`/empty on the primary agent's registry, so no
    /// fallback path is taken there.
    #[allow(clippy::too_many_arguments)]
    fn register_session_scoped_tools(
        &self,
        session_manager: SessionManager,
        shared_session_id: Arc<RwLock<Option<Uuid>>>,
        todo_list: todo::SharedTodoList,
        skills: Arc<crate::skills::SkillCache>,
        memories: Arc<crate::memory::MemoryCache>,
        // How much of the memory store this agent may reach. Always `Write` for the primary agent;
        // for a sub-agent, whatever its `agent_spawn` call granted, which defaults to nothing.
        memory_access: crate::config::MemoryAccess,
        parent_session_id: Option<Uuid>,
        inherited_scratchpad_names: Vec<String>,
        cwd: crate::agent::SharedCwd,
        // `None` for sub-agents. A sub-agent's session is ephemeral, so a job keyed to it would
        // outlive the only conversation able to run it -- the same reason Claude Code refuses
        // durable crons for its teammates.
        schedule: Option<(
            crate::config::ResolvedScheduleConfig,
            crate::permission::SharedPermission,
        )>,
        // `None` for sub-agents. A sub-agent runs only while its parent is waiting on it or
        // following up, so work that outlives the turn has nobody to report to: the parent has
        // already returned, and the child is not scheduled again on its own.
        background: Option<crate::background::BackgroundTasks>,
    ) {
        self.register_builtin(Arc::new(load_tool::LoadToolTool {
            tools: Arc::downgrade(&self.tools),
            deferred: Arc::downgrade(&self.deferred),
            mcp_manager: Arc::downgrade(&self.mcp_manager),
        }));
        // Both stores register their tools only when the subsystem is switched on. Skipping is
        // the whole point of the config switch: a disabled subsystem must keep its schemas out of
        // every request, not ship tools that can only ever fail.
        if skills.enabled() {
            self.register_builtin(Arc::new(skill::SkillTool { skills }));
        }
        // Two independent gates: `enabled()` is whether this installation keeps memories at all,
        // `memory_access` is how much of that store the agent in front of us may reach.
        if memories.enabled() && memory_access != crate::config::MemoryAccess::None {
            // Registration order is the order the tools reach the provider, and the tool array
            // heads the prompt-cache prefix. Interleaving the gates rather than grouping them
            // keeps a full-access agent's order identical to what it was before the levels
            // existed, so no existing installation pays a cache miss for a set that did not
            // actually change.
            if memory_access == crate::config::MemoryAccess::Write {
                self.register_builtin(Arc::new(memory::MemoryWriteTool {
                    memories: memories.clone(),
                }));
            }
            self.register_builtin(Arc::new(memory::MemoryReadTool {
                memories: memories.clone(),
            }));
            self.register_builtin(Arc::new(memory::MemorySearchTool {
                memories: memories.clone(),
            }));
            if memory_access == crate::config::MemoryAccess::Write {
                self.register_builtin(Arc::new(memory::MemoryDeleteTool { memories }));
            }
        }
        self.register_builtin(Arc::new(render_image::RenderImageTool {
            session_id: shared_session_id.clone(),
            session_manager: session_manager.clone(),
        }));
        if let Some((schedule_config, shared_permission)) = schedule
            && schedule_config.enabled
        {
            for tool in schedule::build(
                session_manager.clone(),
                shared_session_id.clone(),
                schedule_config,
                shared_permission,
            ) {
                self.register_builtin(tool);
            }
        }
        if let Some(tasks) = background
            && self.background_enabled()
        {
            for tool in background::build(session_manager.clone(), shared_session_id.clone(), tasks)
            {
                self.register_builtin(tool);
            }
        }
        self.register_builtin(Arc::new(todo::TodoTool { todo_list }));
        self.register_builtin(Arc::new(scratchpad::ScratchpadWriteTool {
            session_manager: session_manager.clone(),
            session_id: shared_session_id.clone(),
            inherited_names: inherited_scratchpad_names.clone(),
        }));
        self.register_builtin(Arc::new(scratchpad::ScratchpadReadTool {
            session_manager: session_manager.clone(),
            session_id: shared_session_id.clone(),
            parent_session_id,
            inherited_names: inherited_scratchpad_names.clone(),
        }));
        self.register_builtin(Arc::new(conversation::ConversationSearchTool {
            session_manager: session_manager.clone(),
            session_id: shared_session_id.clone(),
        }));
        self.register_builtin(Arc::new(conversation::ConversationReadTool {
            session_manager: session_manager.clone(),
            session_id: shared_session_id.clone(),
        }));
        self.register_builtin(Arc::new(scratchpad::ScratchpadEditTool {
            session_manager: session_manager.clone(),
            session_id: shared_session_id.clone(),
            inherited_names: inherited_scratchpad_names.clone(),
        }));
        self.register_builtin(Arc::new(scratchpad::ScratchpadListTool {
            session_manager: session_manager.clone(),
            session_id: shared_session_id.clone(),
            parent_session_id,
            inherited_names: inherited_scratchpad_names.clone(),
        }));
        self.register_builtin(Arc::new(scratchpad::ScratchpadMergeTool {
            session_manager: session_manager.clone(),
            session_id: shared_session_id.clone(),
            parent_session_id,
            inherited_names: inherited_scratchpad_names.clone(),
        }));
        self.register_builtin(Arc::new(scratchpad::ScratchpadDeleteTool {
            session_manager: session_manager.clone(),
            session_id: shared_session_id.clone(),
            inherited_names: inherited_scratchpad_names.clone(),
        }));
        self.register_builtin(Arc::new(scratchpad::ScratchpadRenameTool {
            session_manager: session_manager.clone(),
            session_id: shared_session_id.clone(),
            inherited_names: inherited_scratchpad_names.clone(),
        }));
        self.register_builtin(Arc::new(scratchpad::ScratchpadLoadFileTool {
            session_manager: session_manager.clone(),
            session_id: shared_session_id.clone(),
            inherited_names: inherited_scratchpad_names.clone(),
            cwd: cwd.clone(),
        }));
        self.register_builtin(Arc::new(scratchpad::ScratchpadSaveFileTool {
            session_manager,
            session_id: shared_session_id,
            parent_session_id,
            inherited_names: inherited_scratchpad_names,
            cwd,
        }));
    }

    /// Register the always-on half of the `context_*` family.
    ///
    /// Outside [`Self::build_default`] for the same reason `agent_spawn` is: the counters these
    /// read are owned by the `Agent`, which does not exist when the registry is built. The caller
    /// makes them, hands them here, and points the agent at the same handles.
    ///
    /// `context_replace` is deliberately absent. It only exists inside a checkpoint turn, which
    /// builds its own tool list; registering it here would offer the model a way to blank its
    /// context during ordinary work.
    ///
    /// Sub-agents get neither. A worker's window is its own, but the compaction it would be asking
    /// for happens inside a turn its parent is waiting on, and the parent owns that conversation.
    pub fn register_context_tools(
        &self,
        gauge: context::ContextGauge,
        pending: context::PendingCompaction,
        checkpoint_enabled: bool,
        session_manager: SessionManager,
        shared_session_id: Arc<RwLock<Option<Uuid>>>,
    ) {
        self.register_builtin(Arc::new(context::ContextCheckTool {
            gauge,
            session_manager,
            session_id: shared_session_id,
        }));
        self.register_builtin(Arc::new(context::ContextCompactTool {
            pending,
            checkpoint_enabled,
        }));
    }

    /// The tools a checkpoint turn may use: this registry's, filtered to
    /// [`CHECKPOINT_TOOL_NAMES`] and the caller's permission, plus a fresh `context_replace` bound
    /// to `slot`.
    ///
    /// Read *out of* the live registry rather than built fresh, so a tool the user disabled in
    /// `[tools]` stays disabled here, and every tool keeps the session's cwd, permission and
    /// frontend already baked into it.
    pub fn checkpoint_tools(
        &self,
        permission: Permission,
        slot: context::SubmissionSlot,
    ) -> Vec<Arc<dyn Tool>> {
        let mut tools: Vec<Arc<dyn Tool>> = CHECKPOINT_TOOL_NAMES
            .iter()
            .filter_map(|name| self.get(name))
            .filter(|tool| {
                let definition = tool.definition();
                let required = self
                    .permission_overrides
                    .get(&definition.name)
                    .copied()
                    .unwrap_or_else(|| tool.required_permission());
                permission.allows(required)
            })
            .collect();
        // Unconditional, deliberately outside the permission filter above. `context_replace` has
        // no effect outside this process - it writes a summary into a slot the caller owns - and
        // without it a checkpoint has no way to finish, so filtering it out would silently
        // downgrade every compaction at `none` permission to the fallback summarizer. That leaves
        // a `none`-permission checkpoint holding exactly one tool, which is the right shape: it
        // can still write the summary, it just has nowhere to save anything.
        tools.push(Arc::new(context::ContextReplaceTool { slot }));
        tools
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_default(
        web_client_config: crate::config::WebClientConfig,
        shared_permission: crate::permission::SharedPermission,
        sandbox_enabled: bool,
        sandbox_capability: crate::sandbox::SandboxCapability,
        sandbox_backend: crate::config::SandboxBackend,
        backend_probe: crate::sandbox::BackendProbe,
        todo_list: todo::SharedTodoList,
        session_manager: SessionManager,
        shared_session_id: Arc<RwLock<Option<Uuid>>>,
        skills: Arc<crate::skills::SkillCache>,
        memories: Arc<crate::memory::MemoryCache>,
        builtin_filter: BuiltinToolFilter,
        cwd: crate::agent::SharedCwd,
        roots: crate::agent::SharedRoots,
        frontend: Arc<dyn crate::frontend::Frontend>,
        schedule: crate::config::ResolvedScheduleConfig,
        background: (
            crate::config::ResolvedBackgroundConfig,
            crate::background::BackgroundTasks,
        ),
    ) -> Result<Self> {
        let registry = Self::new_with_filter(builtin_filter);
        // Before the tools are registered, because both the `task_*` registration and the schema
        // splice read this flag.
        let (background_config, background_tasks) = background;
        if background_config.enabled {
            registry.enable_background();
        }
        registry.register_core_tools(
            &web_client_config,
            shared_permission.clone(),
            sandbox_enabled,
            sandbox_capability,
            sandbox_backend,
            backend_probe,
            cwd.clone(),
            roots,
            frontend,
        )?;
        registry.register_session_scoped_tools(
            session_manager,
            shared_session_id,
            todo_list,
            skills,
            memories,
            crate::config::MemoryAccess::Write,
            None,
            Vec::new(),
            cwd,
            Some((schedule, shared_permission)),
            Some(background_tasks),
        );
        Ok(registry)
    }

    /// Build a tool registry for sub-agents. Sub-agents get the same session-scoped tools as the
    /// parent (load_tool, skill, memory_*, render_image, todo, scratchpad_*) scoped to their own
    /// ephemeral child session.
    ///
    /// `agent_spawn` is deliberately not registered here, but sub-agents *can* nest: the caller
    /// adds it afterwards when the recursion budget allows (`AgentSpawnTool::execute`), because
    /// the child's depth counters aren't known until the parent's `max_depth` override has been
    /// resolved. Registering it outside this builder mirrors the root's own registration in
    /// `assemble_agent`.
    ///
    /// `parent_session_id` + `inherited_scratchpad_names` enable read-only scratchpad inheritance:
    /// `scratchpad_read` falls back to the parent for allowlisted names, and `scratchpad_list`
    /// enumerates them in an `(inherited)` section. Pass `None`/`Vec::new()` to opt out.
    ///
    /// `denials` and `memory_access` are what makes a sub-agent registry narrower than its parent's
    /// rather than a copy of it. Both are already unioned/resolved by the caller: this builder
    /// applies them, it does not decide them.
    #[allow(clippy::too_many_arguments)]
    pub fn build_for_subagent(
        web_client_config: crate::config::WebClientConfig,
        shared_permission: crate::permission::SharedPermission,
        sandbox_enabled: bool,
        sandbox_capability: crate::sandbox::SandboxCapability,
        sandbox_backend: crate::config::SandboxBackend,
        backend_probe: crate::sandbox::BackendProbe,
        builtin_filter: BuiltinToolFilter,
        denials: ToolDenials,
        todo_list: todo::SharedTodoList,
        session_manager: SessionManager,
        shared_session_id: Arc<RwLock<Option<Uuid>>>,
        skills: Arc<crate::skills::SkillCache>,
        memories: Arc<crate::memory::MemoryCache>,
        memory_access: crate::config::MemoryAccess,
        parent_session_id: Option<Uuid>,
        inherited_scratchpad_names: Vec<String>,
        cwd: crate::agent::SharedCwd,
        roots: crate::agent::SharedRoots,
        frontend: Arc<dyn crate::frontend::Frontend>,
    ) -> Result<Self> {
        let registry = Self::new_with_filter_and_denials(builtin_filter, denials);
        registry.register_core_tools(
            &web_client_config,
            shared_permission,
            sandbox_enabled,
            sandbox_capability,
            sandbox_backend,
            backend_probe,
            cwd.clone(),
            roots,
            frontend,
        )?;
        registry.register_session_scoped_tools(
            session_manager,
            shared_session_id,
            todo_list,
            skills,
            memories,
            memory_access,
            parent_session_id,
            inherited_scratchpad_names,
            cwd,
            None,
            None,
        );
        Ok(registry)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::path::Path;

    use super::*;

    /// Test helper: pull the concatenated text content out of a `ToolOutput`. Used across every
    /// per-tool test module.
    pub(crate) fn text_content(output: &ToolOutput) -> String {
        ContentBlock::tool_result_text_content(&output.content)
    }

    fn test_shared_permission() -> crate::permission::SharedPermission {
        crate::permission::SharedPermission::new(
            Permission::Write,
            crate::permission::EnabledPermissions::ALL,
        )
    }

    fn test_todo_list() -> todo::SharedTodoList {
        Arc::new(RwLock::new(todo::TodoState::default()))
    }

    /// Every tool that takes a meaningful argument must be able to show one in the tool-call
    /// indicator.
    ///
    /// `resolve_primary_param` tries the built-in map first and falls back to `required[0]`, so a
    /// tool with arguments but no `required` array falls through both and renders as a bare name.
    /// `task_cancel` shipped exactly that way: an id-or-`all` schema with nothing required, so a
    /// cancellation displayed without saying what it cancelled.
    #[tokio::test]
    async fn test_every_tool_with_arguments_can_show_a_primary_param() {
        // meka's own universal parameters. A tool whose schema is only these takes no argument of
        // its own and has nothing to display.
        const UNIVERSAL: &[&str] = &[SCRATCHPAD_PARAMETER, BACKGROUND_PARAMETER];

        let registry = test_registry().await;
        registry.enable_background();
        for tool in background::build(
            SessionManager::open(Some(std::path::Path::new(":memory:")))
                .await
                .expect("in-memory db"),
            Arc::new(RwLock::new(None)),
            crate::background::BackgroundTasks::default(),
        ) {
            registry.register(tool).expect("register task tool");
        }

        for definition in registry.definitions_active_with_loaded(&[]) {
            let own_properties = definition
                .parameters
                .get("properties")
                .and_then(|properties| properties.as_object())
                .map(|properties| {
                    properties
                        .keys()
                        .filter(|key| !UNIVERSAL.contains(&key.as_str()))
                        .count()
                })
                .unwrap_or(0);
            if own_properties == 0 {
                continue;
            }
            let has_required = definition
                .parameters
                .get("required")
                .and_then(|required| required.as_array())
                .is_some_and(|required| !required.is_empty());
            // Probe the built-in map with an empty object: a mapped tool answers for at least one
            // shape of input, which is what distinguishes "has a rule" from "falls through".
            let mapped = crate::render::has_primary_param_rule(&definition.name);
            assert!(
                has_required || mapped,
                "{} takes arguments but has no `required` and no built-in rule, so its tool-call \
                 indicator would render with no argument",
                definition.name,
            );
        }
    }

    async fn test_registry() -> ToolRegistry {
        build_test_registry(BuiltinToolFilter::default()).await
    }

    async fn build_test_registry(filter: BuiltinToolFilter) -> ToolRegistry {
        let session_manager = SessionManager::open(Some(Path::new(":memory:")))
            .await
            .expect("failed to open in-memory database");
        let shared_session_id = Arc::new(RwLock::new(None));
        let sandbox_capability = crate::sandbox::detect();
        let backend_probe = crate::sandbox::BackendProbe::Ok(sandbox_capability.clone());
        ToolRegistry::build_default(
            crate::config::WebClientConfig::default(),
            test_shared_permission(),
            true,
            sandbox_capability,
            crate::config::SandboxBackend::Landlock,
            backend_probe,
            test_todo_list(),
            session_manager,
            shared_session_id,
            crate::skills::SkillCache::for_root(None),
            crate::memory::MemoryCache::for_root(None),
            filter,
            crate::agent::test_cwd(),
            crate::agent::test_roots(),
            Arc::new(crate::frontend::SilentFrontend),
            crate::config::ResolvedScheduleConfig::default(),
            (
                crate::config::ResolvedBackgroundConfig::default(),
                crate::background::BackgroundTasks::default(),
            ),
        )
        .expect("default web client config should build cleanly")
    }

    use crate::provider::Role;

    fn load_tool_use(id: &str, target: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: id.to_string(),
                name: LOAD_TOOL_NAME.to_string(),
                input: serde_json::json!({ "name": target }),
            }],
        }
    }

    fn tool_result(use_id: &str, body: &str, is_error: bool) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: use_id.to_string(),
                content: vec![ToolResultContent::Text {
                    text: body.to_string(),
                }],
                is_error,
            }],
        }
    }

    /// A no-op tool used as a deferred-tool fixture: registering it after `build_default` and
    /// calling `mark_deferred` lets tests exercise the load_tool flow against a tool that is
    /// genuinely deferred, instead of re-deferring a production tool that ships active.
    pub(crate) struct FixtureDeferredTool {
        pub name: String,
    }

    #[async_trait::async_trait]
    impl Tool for FixtureDeferredTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new(
                self.name.clone(),
                format!("test fixture: deferred tool {}", self.name),
                serde_json::json!({"type": "object", "properties": {}}),
            )
        }

        fn required_permission(&self) -> Permission {
            Permission::Read
        }

        async fn execute(
            &self,
            _input: serde_json::Value,
            _cancellation: CancellationToken,
        ) -> crate::error::Result<ToolOutput> {
            Ok(ToolOutput::text("ok".to_string(), false))
        }
    }

    /// Register a deferred fixture tool on the registry. Returns the name the test should use when
    /// invoking `load_tool`.
    pub(crate) fn register_deferred_fixture(registry: &ToolRegistry, name: &str) {
        registry
            .register(Arc::new(FixtureDeferredTool {
                name: name.to_string(),
            }))
            .expect("fixture tool name must be unique");
        registry.mark_deferred(name);
    }

    #[test]
    fn test_extract_loaded_tool_names_empty() {
        assert!(extract_loaded_tool_names(&[]).is_empty());
    }

    #[test]
    fn test_extract_loaded_tool_names_single_success() {
        let messages = vec![
            load_tool_use("u1", "scratchpad_read"),
            tool_result("u1", "loaded", false),
        ];
        let loaded = extract_loaded_tool_names(&messages);
        assert_eq!(loaded.len(), 1);
        assert!(loaded.contains("scratchpad_read"));
    }

    #[test]
    fn test_extract_loaded_tool_names_from_a_batch() {
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "u1".to_string(),
                    name: LOAD_TOOL_NAME.to_string(),
                    input: serde_json::json!({"name": ["alpha", "beta"]}),
                }],
            },
            tool_result("u1", "loaded", false),
        ];
        let loaded = extract_loaded_tool_names(&messages);
        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains("alpha") && loaded.contains("beta"));
    }

    #[test]
    fn test_load_tool_names_dedups_and_caps() {
        let input = serde_json::json!({"name": ["a", "a", "b"]});
        assert_eq!(load_tool_names(&input), vec![
            "a".to_string(),
            "b".to_string()
        ]);

        let many: Vec<String> = (0..MAX_LOAD_TOOL_BATCH + 5)
            .map(|index| format!("tool_{index}"))
            .collect();
        let input = serde_json::json!({"name": many});
        assert_eq!(load_tool_names(&input).len(), MAX_LOAD_TOOL_BATCH);
    }

    #[test]
    fn test_load_tool_names_rejects_a_non_string_name() {
        assert!(load_tool_names(&serde_json::json!({})).is_empty());
        assert!(load_tool_names(&serde_json::json!({"name": 7})).is_empty());
        assert!(load_tool_names(&serde_json::json!({"name": [7, false]})).is_empty());
    }

    /// The `send_file` schema, as mekabridge advertises it.
    fn send_file_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "conversation": {"type": "string", "description": "Target conversation"},
                "path": {"type": "string", "description": "Path to the file"},
                "caption": {"type": ["string", "null"], "description": "Optional caption"},
                "as_photo": {
                    "type": "boolean",
                    "default": false,
                    "description": "Send as a viewable photo rather than a downloadable document.",
                },
            },
            "required": ["conversation", "path"]
        })
    }

    /// The regression in full: a call that succeeds, takes a silently wrong default, and until now
    /// said nothing about the flag that would have fixed it.
    #[test]
    fn test_schema_disagreement_names_the_unused_parameter() {
        let input = serde_json::json!({"conversation": "telegram:1", "path": "/tmp/a.png"});
        let advisory = schema_disagreement(
            "mcp__mekabridge__send_file",
            &input,
            &send_file_schema(),
            true,
        )
        .expect("blind call omitting as_photo must be advised");

        assert!(
            advisory.contains("as_photo (boolean, default false)"),
            "{advisory}"
        );
        assert!(advisory.contains("viewable photo"), "{advisory}");
        assert!(advisory.contains("caption"), "{advisory}");
        assert!(advisory.contains("load_tool"), "{advisory}");
        // Parameters the call did supply are not worth repeating back.
        assert!(!advisory.contains("\n  path ("), "{advisory}");
    }

    /// The same call, once the model has actually seen the schema, is not worth a word.
    #[test]
    fn test_schema_disagreement_is_silent_for_a_loaded_tool() {
        let input = serde_json::json!({"conversation": "telegram:1", "path": "/tmp/a.png"});
        assert!(
            schema_disagreement(
                "mcp__mekabridge__send_file",
                &input,
                &send_file_schema(),
                false
            )
            .is_none()
        );
    }

    /// What a server binary older than its own source looks like from the caller's side.
    #[test]
    fn test_schema_disagreement_flags_an_undeclared_argument() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path"}},
        });
        let input = serde_json::json!({"path": "/tmp/a.png", "as_photo": true});
        let advisory = schema_disagreement("send_file", &input, &schema, false).expect("advised");

        assert!(
            advisory.contains("not declared in this tool's schema"),
            "{advisory}"
        );
        assert!(advisory.contains("as_photo"), "{advisory}");
        // Nothing to load: this tool's schema is already in hand.
        assert!(!advisory.contains("load_tool"), "{advisory}");
    }

    /// `scratchpad` works on every tool and is consumed by the agent loop, so an MCP schema that
    /// doesn't mention it is not a disagreement.
    #[test]
    fn test_schema_disagreement_ignores_the_scratchpad_parameter() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path"}},
        });
        let input = serde_json::json!({"path": "/tmp/a.png", "scratchpad": "out"});
        assert!(schema_disagreement("mcp__x__read", &input, &schema, false).is_none());
    }

    #[test]
    fn test_schema_disagreement_respects_additional_properties() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path"}},
            "additionalProperties": true,
        });
        let input = serde_json::json!({"path": "/tmp/a.png", "extra": 1});
        assert!(schema_disagreement("open", &input, &schema, false).is_none());
    }

    #[test]
    fn test_schema_disagreement_ignores_schemas_with_nothing_to_say() {
        let input = serde_json::json!({"anything": 1});
        assert!(schema_disagreement("x", &input, &serde_json::json!({}), true).is_none());
        assert!(
            schema_disagreement(
                "x",
                &input,
                &serde_json::json!({"type": "object", "properties": {}}),
                true
            )
            .is_none()
        );
    }

    #[test]
    fn test_schema_disagreement_caps_the_list() {
        let mut properties = serde_json::Map::new();
        for index in 0..MAX_ADVISED_PARAMETERS + 3 {
            properties.insert(
                format!("option_{index}"),
                serde_json::json!({"type": "string", "description": "An option"}),
            );
        }
        let schema = serde_json::json!({"type": "object", "properties": properties});
        let advisory =
            schema_disagreement("x", &serde_json::json!({}), &schema, true).expect("some");
        assert!(advisory.contains("(+3 more)"), "{advisory}");
    }

    /// Undocumented parameters are skipped: naming a knob without saying what it does is not enough
    /// to act on, and the schema is one `load_tool` away.
    #[test]
    fn test_schema_disagreement_skips_undocumented_parameters() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path"},
                "mystery": {"type": "string"},
            },
        });
        assert!(
            schema_disagreement("x", &serde_json::json!({"path": "/a"}), &schema, true).is_none()
        );
    }

    #[tokio::test]
    async fn test_background_is_absent_until_enabled_and_present_after() {
        let registry = ToolRegistry::new();
        register_deferred_fixture(&registry, "fixture_alpha");
        let has_background = |registry: &ToolRegistry| {
            registry
                .definitions_active_with_loaded(&["fixture_alpha".to_string()])
                .iter()
                .any(|definition| {
                    definition
                        .parameters
                        .get("properties")
                        .and_then(|properties| properties.get(BACKGROUND_PARAMETER))
                        .is_some()
                })
        };

        assert!(
            !has_background(&registry),
            "a parameter the model can see but cannot use is worse than one it cannot see"
        );
        registry.enable_background();
        assert!(has_background(&registry));
    }

    /// A server that already advertises `background` owns the name. Silently redefining it would be
    /// a far worse bug than losing the feature on that one tool.
    #[test]
    fn test_offer_background_never_shadows_a_real_parameter() {
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "background": {"type": "string", "description": "Wallpaper to set"}
            }
        });
        offer_background(&mut schema);
        assert_eq!(
            schema["properties"]["background"]["type"],
            serde_json::json!("string")
        );
    }

    #[test]
    fn test_offer_background_leaves_a_no_argument_schema_alone() {
        let mut schema = serde_json::json!({"type": "object"});
        offer_background(&mut schema);
        assert!(schema.get("properties").is_none());
    }

    /// A schema for a tool that declares neither of meka's parameters, which is the ordinary case
    /// and the one where meka owns both names.
    fn plain_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {"command": {"type": "string"}}
        })
    }

    /// The flag is meka's, so it must never reach the tool, least of all a remote MCP server that
    /// never advertised the key.
    #[test]
    fn test_take_background_flag_removes_it_from_the_arguments() {
        let schema = plain_schema();
        let (input, detach) = take_background_flag(
            &serde_json::json!({"command": "make", "background": true}),
            &schema,
        );
        assert!(detach);
        assert_eq!(input, serde_json::json!({"command": "make"}));

        let (input, detach) = take_background_flag(
            &serde_json::json!({"command": "make", "background": false}),
            &schema,
        );
        assert!(!detach);
        assert_eq!(input, serde_json::json!({"command": "make"}));

        let (input, detach) =
            take_background_flag(&serde_json::json!({"command": "make"}), &schema);
        assert!(!detach);
        assert_eq!(input, serde_json::json!({"command": "make"}));
    }

    #[test]
    fn test_take_background_flag_treats_a_non_boolean_as_absent() {
        for value in [
            serde_json::json!("yes"),
            serde_json::json!(1),
            serde_json::json!(null),
        ] {
            let (input, detach) =
                take_background_flag(&serde_json::json!({"background": value}), &plain_schema());
            assert!(!detach, "a wrong-typed flag must not detach by accident");
            assert_eq!(input, serde_json::json!({}));
        }
    }

    /// A tool that advertises `background` itself owns the name: `offer_background` declines to
    /// shadow it, so the model is looking at the *tool's* parameter and the argument is the tool's
    /// to receive. An image tool taking `background: "transparent"`, or an exec server with a
    /// detach flag of its own, would otherwise have the argument eaten before dispatch.
    #[test]
    fn test_a_tool_that_declares_background_keeps_its_own_argument() {
        let owned = serde_json::json!({
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "background": {"type": "string", "description": "Backdrop colour"}
            }
        });

        // Not spliced, so the model never saw meka's version.
        let mut offered = owned.clone();
        offer_background(&mut offered);
        assert_eq!(offered, owned, "meka must not shadow the tool's parameter");

        // Not stripped, not detached, and not refused for being the wrong type for a flag it is
        // not.
        let call = serde_json::json!({"prompt": "a cat", "background": "transparent"});
        let (input, detach) = take_background_flag(&call, &owned);
        assert!(!detach, "the tool's parameter is not a detach request");
        assert_eq!(
            input, call,
            "the tool must receive the argument it declared"
        );
        assert_eq!(meka_parameter_error(&call, &owned), None);
    }

    /// The same question for `scratchpad`, which needs a weaker test: meka's own builtins declare
    /// it, as a string, so presence proves nothing and only a conflicting *type* means the tool
    /// has its own idea of the name.
    #[test]
    fn test_scratchpad_ownership_turns_on_the_declared_type() {
        let builtin_style = serde_json::json!({
            "type": "object",
            "properties": {"scratchpad": {"type": "string"}}
        });
        let complaint = meka_parameter_error(&serde_json::json!({"scratchpad": 7}), &builtin_style);
        assert!(
            complaint.is_some(),
            "a tool declaring it as a string agrees with meka, so the check still applies",
        );

        let someone_elses = serde_json::json!({
            "type": "object",
            "properties": {"scratchpad": {"type": "integer"}}
        });
        assert_eq!(
            meka_parameter_error(&serde_json::json!({"scratchpad": 7}), &someone_elses),
            None,
            "a tool that means a number by the name is entitled to be sent one",
        );
    }

    /// Some models emit every argument as a string whatever the schema says; GLM does it through
    /// OpenRouter, which is where this came up. Both of meka's own parameters decide what a call
    /// does rather than what it is called with, so the answer is to say so and let the model retry,
    /// not to read the wrong type as absent (a silent no-op) or to guess at it (a contract where
    /// `"true"` works and `"yes"` does not).
    #[test]
    fn test_a_wrong_typed_meka_parameter_is_refused_by_name() {
        for value in [
            serde_json::json!("true"),
            serde_json::json!("yes"),
            serde_json::json!(1),
        ] {
            let complaint =
                meka_parameter_error(&serde_json::json!({"background": value}), &plain_schema())
                    .expect("a wrong-typed background flag must be refused, not silently ignored");
            assert!(complaint.contains("background"), "{complaint}");
            assert!(complaint.contains("boolean"), "{complaint}");
        }

        let complaint =
            meka_parameter_error(&serde_json::json!({"scratchpad": 7}), &plain_schema())
                .expect("a wrong-typed scratchpad name must be refused");
        assert!(complaint.contains("scratchpad"), "{complaint}");
        assert!(complaint.contains("string"), "{complaint}");
    }

    /// Models emit `null` for optional arguments they are not using, and a tool's own arguments
    /// belong to the tool: refusing either would cost capability without catching a bug.
    #[test]
    fn test_well_formed_and_absent_parameters_pass() {
        for input in [
            serde_json::json!({"command": "make", "background": true}),
            serde_json::json!({"command": "make", "background": false}),
            serde_json::json!({"command": "make", "background": null}),
            serde_json::json!({"command": "make", "scratchpad": "build-log"}),
            serde_json::json!({"command": "make", "scratchpad": null}),
            serde_json::json!({"command": "make"}),
            // The tool's own arguments, wrong-typed on purpose: not meka's to police.
            serde_json::json!({"command": 12, "timeout": "30"}),
            serde_json::json!("not an object at all"),
        ] {
            assert_eq!(
                meka_parameter_error(&input, &plain_schema()),
                None,
                "{input}"
            );
        }
    }

    /// `background` is spliced into the definitions sent to the provider, not into the raw one this
    /// check reads, so without an exemption it would be reported as an undeclared argument.
    #[test]
    fn test_schema_disagreement_ignores_the_background_parameter() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Shell command"}},
        });
        let input = serde_json::json!({"command": "make", "background": true});
        assert!(schema_disagreement("execute_command", &input, &schema, false).is_none());
    }

    /// The commonest slip: naming the tool without its `mcp__<server>__` prefix. Edit distance
    /// alone puts the answer seventeen operations away, so the segment match has to carry it.
    #[test]
    fn test_did_you_mean_matches_on_the_final_segment() {
        let registered = ["mcp__mekabridge__send_file", "read_file"];
        let hint = did_you_mean_hint("send_file", registered.into_iter());
        assert_eq!(hint, " Did you mean `mcp__mekabridge__send_file`?");
    }

    #[test]
    fn test_did_you_mean_catches_a_typo() {
        let registered = ["read_file", "write_file"];
        let hint = did_you_mean_hint("raed_file", registered.into_iter());
        assert_eq!(hint, " Did you mean `read_file`?");
    }

    #[test]
    fn test_did_you_mean_is_silent_when_nothing_is_close() {
        let registered = ["read_file", "run_shell"];
        assert_eq!(
            did_you_mean_hint("frobnicate_widget", registered.into_iter()),
            ""
        );
    }

    #[test]
    fn test_did_you_mean_lists_every_server_offering_the_segment() {
        let registered = ["mcp__a__send_file", "mcp__b__send_file"];
        let hint = did_you_mean_hint("send_file", registered.into_iter());
        assert_eq!(
            hint,
            " Did you mean `mcp__a__send_file` or `mcp__b__send_file`?"
        );
    }

    #[test]
    fn test_extract_loaded_tool_names_error_excluded() {
        let messages = vec![
            load_tool_use("u1", "missing_tool"),
            tool_result("u1", "Error: not registered", true),
        ];
        assert!(extract_loaded_tool_names(&messages).is_empty());
    }

    #[test]
    fn test_extract_loaded_tool_names_orphan_use() {
        // load_tool was issued but the tool_result hasn't arrived yet.
        let messages = vec![load_tool_use("u1", "scratchpad_read")];
        assert!(extract_loaded_tool_names(&messages).is_empty());
    }

    #[test]
    fn test_extract_loaded_tool_names_ignores_other_tools() {
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "u1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({ "name": "anything" }),
                }],
            },
            tool_result("u1", "ok", false),
        ];
        assert!(extract_loaded_tool_names(&messages).is_empty());
    }

    #[test]
    fn test_extract_loaded_tool_names_malformed_input() {
        // load_tool called with no `name` field: must not panic, must not pollute the active set.
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "u1".to_string(),
                    name: LOAD_TOOL_NAME.to_string(),
                    input: serde_json::json!({}),
                }],
            },
            tool_result("u1", "Error", true),
        ];
        assert!(extract_loaded_tool_names(&messages).is_empty());
    }

    #[test]
    fn test_extract_loaded_tool_names_multiple_loads_dedup() {
        let messages = vec![
            load_tool_use("u1", "scratchpad_read"),
            tool_result("u1", "ok", false),
            load_tool_use("u2", "scratchpad_edit"),
            tool_result("u2", "ok", false),
            load_tool_use("u3", "scratchpad_read"),
            tool_result("u3", "already available", false),
        ];
        let loaded = extract_loaded_tool_names(&messages);
        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains("scratchpad_read"));
        assert!(loaded.contains("scratchpad_edit"));
    }

    #[test]
    fn test_extract_loaded_tool_names_multi_block_message() {
        // The model can emit several `tool_use` blocks in one assistant message; the matching
        // `tool_result`s come back as separate blocks of one user message. Both must be processed.
        let assistant = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::ToolUse {
                    id: "u1".to_string(),
                    name: LOAD_TOOL_NAME.to_string(),
                    input: serde_json::json!({"name": "scratchpad_read"}),
                },
                ContentBlock::ToolUse {
                    id: "u2".to_string(),
                    name: LOAD_TOOL_NAME.to_string(),
                    input: serde_json::json!({"name": "scratchpad_edit"}),
                },
            ],
        };
        let user_results = Message {
            role: Role::User,
            content: vec![
                ContentBlock::ToolResult {
                    tool_use_id: "u1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "ok".to_string(),
                    }],
                    is_error: false,
                },
                ContentBlock::ToolResult {
                    tool_use_id: "u2".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "ok".to_string(),
                    }],
                    is_error: false,
                },
            ],
        };
        let loaded = extract_loaded_tool_names(&[assistant, user_results]);
        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains("scratchpad_read"));
        assert!(loaded.contains("scratchpad_edit"));
    }

    #[test]
    fn test_extract_loaded_tool_names_mismatched_id() {
        // tool_result references an id that no `load_tool` use claimed. The result is dropped; the
        // orphan use stays unmatched and is not added to the active set.
        let messages = vec![
            load_tool_use("u1", "scratchpad_read"),
            tool_result("u_other", "ok", false),
        ];
        assert!(extract_loaded_tool_names(&messages).is_empty());
    }

    #[test]
    fn test_extract_loaded_tool_names_interleaved_with_other_tool_calls() {
        // load_tool calls share the message stream with regular tool calls; the scanner must pair
        // on tool_use_id, not on positional adjacency.
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::ToolUse {
                        id: "u1".to_string(),
                        name: "read_file".to_string(),
                        input: serde_json::json!({"path": "/tmp/x"}),
                    },
                    ContentBlock::ToolUse {
                        id: "u2".to_string(),
                        name: LOAD_TOOL_NAME.to_string(),
                        input: serde_json::json!({"name": "scratchpad_read"}),
                    },
                ],
            },
            Message {
                role: Role::User,
                content: vec![
                    ContentBlock::ToolResult {
                        tool_use_id: "u1".to_string(),
                        content: vec![ToolResultContent::Text {
                            text: "x contents".to_string(),
                        }],
                        is_error: false,
                    },
                    ContentBlock::ToolResult {
                        tool_use_id: "u2".to_string(),
                        content: vec![ToolResultContent::Text {
                            text: "loaded".to_string(),
                        }],
                        is_error: false,
                    },
                ],
            },
        ];
        let loaded = extract_loaded_tool_names(&messages);
        assert_eq!(loaded.len(), 1);
        assert!(loaded.contains("scratchpad_read"));
    }

    #[tokio::test]
    async fn test_tool_registry() {
        let registry = test_registry().await;
        assert!(registry.get("read_file").is_some());
        assert!(registry.get("write_file").is_some());
        assert!(registry.get("edit_file").is_some());
        assert!(registry.get("find_files").is_some());
        assert!(registry.get("search_contents").is_some());
        assert!(registry.get("execute_command").is_some());
        assert!(registry.get("fetch_url").is_some());
        assert!(registry.get("search_web").is_some());
        assert!(registry.get("todo").is_some());
        assert!(registry.get("scratchpad_write").is_some());
        assert!(registry.get("scratchpad_read").is_some());
        assert!(registry.get("scratchpad_edit").is_some());
        assert!(registry.get("scratchpad_list").is_some());
        assert!(registry.get("scratchpad_delete").is_some());
        assert!(registry.get("skill").is_some());
        assert!(registry.get("memory_write").is_some());
        assert!(registry.get("memory_read").is_some());
        assert!(registry.get("memory_search").is_some());
        assert!(registry.get("memory_delete").is_some());
        assert!(registry.get("render_image").is_some());
        assert!(registry.get("load_tool").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    /// A store with no root is an *empty* store, not a disabled one: its tools still register, so
    /// the agent can write the first memory into a directory that doesn't exist yet. Conflating
    /// the two is what made `meka tools list` hide tools a real session would have had.
    #[tokio::test]
    async fn test_empty_store_still_registers_its_tools() {
        let registry = test_registry().await;
        assert!(registry.get("skill").is_some());
        assert!(registry.get("memory_write").is_some());
    }

    /// Disabling a subsystem keeps its schemas out of the request entirely, which is the whole
    /// point of the config switch.
    #[tokio::test]
    async fn test_disabled_stores_register_no_tools() {
        let session_manager = SessionManager::open(Some(Path::new(":memory:")))
            .await
            .expect("failed to open in-memory database");
        let sandbox_capability = crate::sandbox::detect();
        let backend_probe = crate::sandbox::BackendProbe::Ok(sandbox_capability.clone());
        let registry = ToolRegistry::build_default(
            crate::config::WebClientConfig::default(),
            test_shared_permission(),
            true,
            sandbox_capability,
            crate::config::SandboxBackend::Landlock,
            backend_probe,
            test_todo_list(),
            session_manager,
            Arc::new(RwLock::new(None)),
            crate::skills::SkillCache::disabled(),
            crate::memory::MemoryCache::disabled(),
            BuiltinToolFilter::default(),
            crate::agent::test_cwd(),
            crate::agent::test_roots(),
            Arc::new(crate::frontend::SilentFrontend),
            crate::config::ResolvedScheduleConfig::default(),
            (
                crate::config::ResolvedBackgroundConfig::default(),
                crate::background::BackgroundTasks::default(),
            ),
        )
        .expect("default web client config should build cleanly");

        assert!(registry.get("skill").is_none());
        for name in [
            "memory_write",
            "memory_read",
            "memory_search",
            "memory_delete",
        ] {
            assert!(registry.get(name).is_none(), "{name} must not register");
        }
        // Unrelated built-ins are untouched.
        assert!(registry.get("read_file").is_some());
    }

    #[tokio::test]
    async fn test_permission_filtering() {
        let registry = test_registry().await;

        let none_tools = registry.definitions_for_permission(Permission::None);
        assert!(none_tools.is_empty());

        let read_tools = registry.definitions_for_permission(Permission::Read);
        assert!(read_tools.iter().any(|t| t.name == "read_file"));
        assert!(read_tools.iter().any(|t| t.name == "find_files"));
        assert!(read_tools.iter().any(|t| t.name == "execute_command"));
        assert!(!read_tools.iter().any(|t| t.name == "write_file"));

        let write_tools = registry.definitions_for_permission(Permission::Write);
        assert!(write_tools.iter().any(|t| t.name == "read_file"));
        assert!(write_tools.iter().any(|t| t.name == "write_file"));
        assert!(write_tools.iter().any(|t| t.name == "execute_command"));
    }

    #[tokio::test]
    async fn test_definitions_active_includes_write_tools() {
        let registry = test_registry().await;
        let active = registry.definitions_active(&[]);
        assert!(active.iter().any(|t| t.name == "read_file"));
        assert!(active.iter().any(|t| t.name == "write_file"));
        assert!(active.iter().any(|t| t.name == "edit_file"));
        assert!(active.iter().any(|t| t.name == "execute_command"));
        // All five scratchpad tools ship default: no `load_tool` round-trip.
        assert!(active.iter().any(|t| t.name == "scratchpad_write"));
        assert!(active.iter().any(|t| t.name == "scratchpad_read"));
        assert!(active.iter().any(|t| t.name == "scratchpad_edit"));
        assert!(active.iter().any(|t| t.name == "scratchpad_list"));
        assert!(active.iter().any(|t| t.name == "scratchpad_delete"));
    }

    /// The tools array precedes every message in the request, so it must only ever grow at the
    /// tail: a mid-array insertion re-caches the entire conversation behind it.
    ///
    /// Loading in reverse registration order is the case that breaks a naive implementation.
    /// Filtering the registration-ordered list in place puts `alpha` in front of `omega` once both
    /// are loaded, even though `omega` was loaded first.
    #[tokio::test]
    async fn test_definitions_active_grows_only_at_the_tail() {
        let registry = test_registry().await;
        register_deferred_fixture(&registry, "alpha_fixture");
        register_deferred_fixture(&registry, "omega_fixture");

        let names = |loaded: &[String]| -> Vec<String> {
            registry
                .definitions_active_with_loaded(loaded)
                .into_iter()
                .map(|definition| definition.name)
                .collect()
        };

        // Reverse registration order: `omega_fixture` was registered second but loads first.
        let none = names(&[]);
        let one = names(&["omega_fixture".to_string()]);
        let two = names(&["omega_fixture".to_string(), "alpha_fixture".to_string()]);

        assert_eq!(
            one[..none.len()],
            none[..],
            "loading a tool must not disturb the tools already in the array"
        );
        assert_eq!(
            two[..one.len()],
            one[..],
            "loading a second tool must not reorder the first: got {:?} after {:?}",
            two,
            one,
        );
        assert_eq!(two[none.len()..], [
            "omega_fixture".to_string(),
            "alpha_fixture".to_string()
        ]);
    }

    #[tokio::test]
    async fn test_definitions_active_stable_across_permissions() {
        let registry = test_registry().await;
        let a = registry.definitions_active(&[]);
        let b = registry.definitions_active(&[]);
        assert_eq!(a.len(), b.len());
        let a_names: Vec<_> = a.iter().map(|t| t.name.clone()).collect();
        let b_names: Vec<_> = b.iter().map(|t| t.name.clone()).collect();
        assert_eq!(a_names, b_names);
    }

    #[tokio::test]
    async fn test_definitions_active_exposes_loaded_deferred_tool() {
        // End-to-end: a successful load_tool call in the conversation promotes the named tool into
        // the active set on the next call.
        let registry = test_registry().await;
        register_deferred_fixture(&registry, "fixture_alpha");
        register_deferred_fixture(&registry, "fixture_beta");

        let baseline = registry.definitions_active(&[]);
        assert!(!baseline.iter().any(|t| t.name == "fixture_alpha"));
        assert!(!baseline.iter().any(|t| t.name == "fixture_beta"));

        let messages = vec![
            load_tool_use("u1", "fixture_alpha"),
            tool_result("u1", "ok", false),
        ];
        let after_load = registry.definitions_active(&messages);
        assert!(after_load.iter().any(|t| t.name == "fixture_alpha"));
        // Append-only: the tools array gains exactly one entry.
        assert_eq!(after_load.len(), baseline.len() + 1);
        // Sibling deferred fixtures remain hidden.
        assert!(!after_load.iter().any(|t| t.name == "fixture_beta"));
    }

    #[tokio::test]
    async fn test_definitions_active_errored_load_stays_hidden() {
        // A load_tool call that ended in an error tool_result must NOT expose the deferred tool:
        // the model's parameter shape was wrong, so the schema was not delivered.
        let registry = test_registry().await;
        register_deferred_fixture(&registry, "fixture_alpha");

        let messages = vec![
            load_tool_use("u1", "fixture_alpha"),
            tool_result("u1", "Error", true),
        ];
        let active = registry.definitions_active(&messages);
        assert!(!active.iter().any(|t| t.name == "fixture_alpha"));
    }

    #[tokio::test]
    async fn test_definitions_active_load_tool_itself_always_visible() {
        // load_tool is the bootstrap meta-tool. It must appear in the active set for an empty
        // conversation; otherwise the model has no way to discover deferred tools.
        let registry = test_registry().await;
        let active = registry.definitions_active(&[]);
        assert!(active.iter().any(|t| t.name == "load_tool"));
    }

    #[tokio::test]
    async fn test_definitions_active_unknown_load_silently_dropped() {
        // load_tool was called for a tool that isn't registered. The scanner records the (errored)
        // result as not loaded, and even if it were loaded, the registry just doesn't contain a
        // tool by that name: no crash, no spurious entry.
        let registry = test_registry().await;
        let messages = vec![
            load_tool_use("u1", "no_such_tool"),
            tool_result("u1", "Error: not registered", true),
        ];
        let active = registry.definitions_active(&messages);
        assert!(!active.iter().any(|t| t.name == "no_such_tool"));
    }

    #[tokio::test]
    async fn test_tool_catalogue_covers_active_and_deferred() {
        let registry = test_registry().await;
        register_deferred_fixture(&registry, "fixture_alpha");

        let entries = registry.tool_catalogue();
        let names: std::collections::HashSet<_> = entries.iter().map(|(n, ..)| n.clone()).collect();
        assert!(names.contains("write_file"));
        assert!(names.contains("scratchpad_read"));
        assert!(names.contains("fixture_alpha"));

        let by_name: std::collections::HashMap<_, _> =
            entries.iter().map(|(n, _, _, d)| (n.clone(), *d)).collect();
        assert!(
            by_name["fixture_alpha"],
            "deferred fixture must be flagged deferred"
        );
        assert!(
            !by_name["scratchpad_read"],
            "scratchpad_read ships active and must not be flagged deferred"
        );
        assert!(!by_name["write_file"], "write_file is an active builtin");

        let required: std::collections::HashMap<_, _> =
            entries.iter().map(|(n, _, p, _)| (n.clone(), *p)).collect();
        assert_eq!(required["read_file"], Permission::Read);
        assert_eq!(required["write_file"], Permission::Write);
    }

    #[tokio::test]
    async fn test_scratchpad_tools_default_to_active() {
        // Regression: feedback agents kept tripping on the asymmetry where scratchpad_write was
        // active but _read/_edit/_list/_delete were deferred behind load_tool. All five must ship
        // default now.
        let registry = test_registry().await;
        let entries = registry.tool_catalogue();
        for name in [
            "scratchpad_write",
            "scratchpad_read",
            "scratchpad_edit",
            "scratchpad_list",
            "scratchpad_merge",
            "scratchpad_delete",
            "scratchpad_rename",
            "scratchpad_load_file",
            "scratchpad_save_file",
        ] {
            let entry = entries
                .iter()
                .find(|(n, ..)| n == name)
                .unwrap_or_else(|| panic!("{} missing from catalogue", name));
            assert!(
                !entry.3,
                "{} must not be deferred (would force a load_tool round-trip)",
                name,
            );
        }
    }

    #[tokio::test]
    async fn test_tool_catalogue_is_sorted() {
        let registry = test_registry().await;
        let entries = registry.tool_catalogue();
        let names: Vec<_> = entries.iter().map(|(n, ..)| n.clone()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted, "tool_catalogue must return sorted entries");
    }

    /// `CHECKPOINT_TOOL_NAMES` is an allow-list read at compaction time, so an entry that is not a
    /// real built-in is silently inert rather than an error: `checkpoint_tools` just finds nothing
    /// under that name and the checkpoint quietly loses a capability.
    #[test]
    fn test_checkpoint_tool_names_are_sorted_and_real() {
        let mut sorted = CHECKPOINT_TOOL_NAMES.to_vec();
        sorted.sort_unstable();
        assert_eq!(CHECKPOINT_TOOL_NAMES, sorted.as_slice());

        let known: HashSet<&str> = BUILTIN_TOOL_NAMES.iter().copied().collect();
        for name in CHECKPOINT_TOOL_NAMES {
            assert!(known.contains(name), "{name} is not a built-in tool");
        }
    }

    /// The rule the allow-list encodes: a checkpoint can save, but not act. Asserted by name
    /// because the list is the only thing standing between an unattended turn and the shell.
    #[test]
    fn test_checkpoint_tool_names_exclude_acting_and_deleting() {
        for name in [
            "execute_command",
            "write_file",
            "edit_file",
            "agent_spawn",
            "schedule_create",
            "fetch_url",
            "search_web",
            "memory_delete",
            "scratchpad_delete",
        ] {
            assert!(
                !CHECKPOINT_TOOL_NAMES.contains(&name),
                "{name} must not be reachable from a checkpoint turn"
            );
        }
    }

    /// `context_replace` is always supplied, even against a registry holding nothing else, because
    /// a checkpoint with no way to submit could only ever fall through to the text tier.
    #[tokio::test]
    async fn test_checkpoint_tools_are_the_allow_list_plus_context_replace() {
        let registry = test_registry().await;
        let slot = Arc::new(std::sync::Mutex::new(None));
        let names: HashSet<String> = registry
            .checkpoint_tools(Permission::Write, slot)
            .iter()
            .map(|tool| tool.definition().name)
            .collect();

        assert!(names.contains("context_replace"));
        assert!(names.contains("memory_write"));
        assert!(names.contains("read_file"));
        assert!(!names.contains("execute_command"));
        assert!(!names.contains("write_file"));
        assert!(!names.contains("agent_spawn"));
    }

    /// A tool the user switched off in `[tools]` must stay off inside a checkpoint too: the list is
    /// read out of the live registry precisely so config keeps applying.
    #[tokio::test]
    async fn test_checkpoint_tools_respect_disabled_tools() {
        let registry = build_test_registry(BuiltinToolFilter::from_config(
            None,
            vec!["memory_write".to_string()],
            std::collections::HashMap::new(),
        ))
        .await;
        let slot = Arc::new(std::sync::Mutex::new(None));
        let names: HashSet<String> = registry
            .checkpoint_tools(Permission::Write, slot)
            .iter()
            .map(|tool| tool.definition().name)
            .collect();

        assert!(!names.contains("memory_write"));
        assert!(names.contains("context_replace"));
    }

    /// `context_replace` is deliberately absent from `BUILTIN_TOOL_NAMES`. Listing it would make
    /// `disabled_tools = ["context_replace"]` a valid-looking entry that silently downgrades every
    /// compaction to the fallback summarizer; left out, it warns as unrecognised instead.
    #[test]
    fn test_context_replace_is_not_a_configurable_builtin() {
        assert!(!BUILTIN_TOOL_NAMES.contains(&"context_replace"));
        assert!(BUILTIN_TOOL_NAMES.contains(&"context_check"));
        assert!(BUILTIN_TOOL_NAMES.contains(&"context_compact"));
    }

    #[tokio::test]
    async fn test_register_duplicate_returns_error() {
        struct DummyTool;
        #[async_trait::async_trait]
        impl Tool for DummyTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition::new(
                    "dup_tool".to_string(),
                    "dummy".to_string(),
                    serde_json::json!({}),
                )
            }

            fn required_permission(&self) -> Permission {
                Permission::Read
            }

            async fn execute(
                &self,
                _input: serde_json::Value,
                _cancellation: CancellationToken,
            ) -> crate::error::Result<ToolOutput> {
                Ok(ToolOutput::text(String::new(), false))
            }
        }

        let registry = ToolRegistry::new();
        registry
            .register(Arc::new(DummyTool) as Arc<dyn Tool>)
            .expect("first registration succeeds");
        let err = registry
            .register(Arc::new(DummyTool) as Arc<dyn Tool>)
            .expect_err("second registration with same name must fail");
        let message = format!("{}", err);
        assert!(
            message.contains("dup_tool"),
            "error message should mention the duplicate name, got: {}",
            message
        );
    }

    #[test]
    fn test_builtin_filter_default_admits_everything() {
        let filter = BuiltinToolFilter::default();
        assert!(filter.admits("read_file"));
        assert!(filter.admits("write_file"));
        assert!(filter.admits("anything_else"));
    }

    #[test]
    fn test_builtin_filter_allow_list_restricts() {
        let allowed: HashSet<String> = ["read_file", "find_files"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let filter = BuiltinToolFilter {
            allowed: Some(allowed),
            ..Default::default()
        };
        assert!(filter.admits("read_file"));
        assert!(filter.admits("find_files"));
        assert!(!filter.admits("write_file"));
        assert!(!filter.admits("execute_command"));
    }

    #[test]
    fn test_builtin_filter_block_list_wins_over_allow_list() {
        let allowed: HashSet<String> = ["read_file", "write_file"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let disabled: HashSet<String> = ["write_file"].iter().map(|s| s.to_string()).collect();
        let filter = BuiltinToolFilter {
            allowed: Some(allowed),
            disabled,
            ..Default::default()
        };
        assert!(filter.admits("read_file"));
        assert!(!filter.admits("write_file"));
    }

    #[test]
    fn test_builtin_filter_from_config_empty_allow_list_is_none() {
        let filter = BuiltinToolFilter::from_config(Some(Vec::new()), Vec::new(), HashMap::new());
        assert!(
            filter.allowed.is_none(),
            "empty allow-list should drop to None"
        );
        assert!(filter.admits("read_file"));
    }

    #[tokio::test]
    async fn test_registry_filter_drops_disabled_tools() {
        let filter = BuiltinToolFilter::from_config(
            None,
            vec!["search_web".to_string(), "fetch_url".to_string()],
            HashMap::new(),
        );
        let registry = build_test_registry(filter).await;
        assert!(registry.get("read_file").is_some());
        assert!(registry.get("write_file").is_some());
        assert!(
            registry.get("search_web").is_none(),
            "search_web should be filtered out"
        );
        assert!(
            registry.get("fetch_url").is_none(),
            "fetch_url should be filtered out"
        );
    }

    #[tokio::test]
    async fn test_registry_filter_allow_list_keeps_only_listed() {
        let filter = BuiltinToolFilter::from_config(
            Some(vec!["read_file".to_string(), "find_files".to_string()]),
            Vec::new(),
            HashMap::new(),
        );
        let registry = build_test_registry(filter).await;
        assert!(registry.get("read_file").is_some());
        assert!(registry.get("find_files").is_some());
        assert!(registry.get("write_file").is_none());
        assert!(registry.get("execute_command").is_none());
        assert!(registry.get("search_web").is_none());
    }

    #[tokio::test]
    async fn test_registry_permission_override_applied() {
        let mut overrides = HashMap::new();
        overrides.insert("read_file".to_string(), Permission::Write);
        let filter = BuiltinToolFilter::from_config(None, Vec::new(), overrides);
        let registry = build_test_registry(filter).await;

        // Override wins over the Tool impl's hardcoded `Read`.
        assert_eq!(
            registry.required_permission_for("read_file"),
            Some(Permission::Write)
        );
        // Non-overridden tool returns its hardcoded level.
        assert_eq!(
            registry.required_permission_for("write_file"),
            Some(Permission::Write)
        );
        // Catalogue must reflect the override too (the world-state block reads from it).
        let catalogue = registry.tool_catalogue();
        let read_file_required = catalogue
            .iter()
            .find(|(name, ..)| name == "read_file")
            .map(|(_, _, perm, _)| *perm);
        assert_eq!(read_file_required, Some(Permission::Write));
    }

    #[tokio::test]
    async fn test_registry_permission_override_excludes_tool_from_lower_level() {
        let mut overrides = HashMap::new();
        overrides.insert("read_file".to_string(), Permission::Write);
        let filter = BuiltinToolFilter::from_config(None, Vec::new(), overrides);
        let registry = build_test_registry(filter).await;

        // At Read permission, read_file should now be excluded from the permission-filtered
        // definitions because the override raised it to Write.
        let read_defs = registry.definitions_for_permission(Permission::Read);
        assert!(!read_defs.iter().any(|t| t.name == "read_file"));

        let write_defs = registry.definitions_for_permission(Permission::Write);
        assert!(write_defs.iter().any(|t| t.name == "read_file"));
    }

    #[tokio::test]
    async fn test_subagent_registry_honours_filter() {
        let filter =
            BuiltinToolFilter::from_config(None, vec!["search_web".to_string()], HashMap::new());
        let sandbox_capability = crate::sandbox::detect();
        let backend_probe = crate::sandbox::BackendProbe::Ok(sandbox_capability.clone());
        let session_manager = SessionManager::open(Some(Path::new(":memory:")))
            .await
            .expect("failed to open in-memory database");
        let shared_session_id = Arc::new(RwLock::new(None));
        let registry = ToolRegistry::build_for_subagent(
            crate::config::WebClientConfig::default(),
            crate::permission::SharedPermission::new(
                Permission::Read,
                crate::permission::EnabledPermissions::ALL,
            ),
            true,
            sandbox_capability,
            crate::config::SandboxBackend::Landlock,
            backend_probe,
            filter,
            ToolDenials::default(),
            test_todo_list(),
            session_manager,
            shared_session_id,
            crate::skills::SkillCache::for_root(None),
            crate::memory::MemoryCache::for_root(None),
            crate::config::MemoryAccess::Write,
            None,
            Vec::new(),
            crate::agent::test_cwd(),
            crate::agent::test_roots(),
            Arc::new(crate::frontend::SilentFrontend),
        )
        .expect("default web client config should build cleanly");
        assert!(registry.get("read_file").is_some());
        assert!(registry.get("search_web").is_none());
        assert!(registry.get("todo").is_some());
        assert!(registry.get("agent_spawn").is_none());
    }

    /// Minimal named tool, for registry-level tests that only care whether a name is present.
    struct StubTool {
        name: String,
    }

    #[async_trait::async_trait]
    impl Tool for StubTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new(self.name.clone(), "stub".to_string(), serde_json::json!({}))
        }

        fn required_permission(&self) -> Permission {
            Permission::Read
        }

        async fn execute(
            &self,
            _input: serde_json::Value,
            _cancellation: CancellationToken,
        ) -> crate::error::Result<ToolOutput> {
            Ok(ToolOutput::text(String::new(), false))
        }
    }

    /// Test helper for the sub-agent registry: everything but the two knobs under test held at its
    /// default, so a case reads as "denials X, memory Y" rather than nineteen positional arguments.
    async fn subagent_registry(
        denials: ToolDenials,
        memory_access: crate::config::MemoryAccess,
    ) -> ToolRegistry {
        let sandbox_capability = crate::sandbox::detect();
        let backend_probe = crate::sandbox::BackendProbe::Ok(sandbox_capability.clone());
        let session_manager = SessionManager::open(Some(Path::new(":memory:")))
            .await
            .expect("failed to open in-memory database");
        ToolRegistry::build_for_subagent(
            crate::config::WebClientConfig::default(),
            crate::permission::SharedPermission::new(
                Permission::Write,
                crate::permission::EnabledPermissions::ALL,
            ),
            true,
            sandbox_capability,
            crate::config::SandboxBackend::Landlock,
            backend_probe,
            BuiltinToolFilter::default(),
            denials,
            test_todo_list(),
            session_manager,
            Arc::new(RwLock::new(None)),
            crate::skills::SkillCache::for_root(None),
            crate::memory::MemoryCache::for_root(None),
            memory_access,
            None,
            Vec::new(),
            crate::agent::test_cwd(),
            crate::agent::test_roots(),
            Arc::new(crate::frontend::SilentFrontend),
        )
        .expect("subagent registry should build")
    }

    const MEMORY_TOOLS: [&str; 4] = [
        "memory_read",
        "memory_search",
        "memory_write",
        "memory_delete",
    ];

    #[tokio::test]
    async fn test_subagent_memory_access_none_registers_no_memory_tools() {
        let registry =
            subagent_registry(ToolDenials::default(), crate::config::MemoryAccess::None).await;
        for name in MEMORY_TOOLS {
            assert!(
                registry.get(name).is_none(),
                "memory = \"none\" must not register '{}'",
                name
            );
        }
        // The gate is memory-specific, not a blanket refusal.
        assert!(registry.get("read_file").is_some());
    }

    #[tokio::test]
    async fn test_subagent_memory_access_read_registers_exactly_the_readers() {
        let registry =
            subagent_registry(ToolDenials::default(), crate::config::MemoryAccess::Read).await;
        assert!(registry.get("memory_read").is_some());
        assert!(registry.get("memory_search").is_some());
        assert!(
            registry.get("memory_write").is_none(),
            "read access must not let a worker edit the store the parent reasons from"
        );
        assert!(registry.get("memory_delete").is_none());
    }

    #[tokio::test]
    async fn test_subagent_memory_access_write_registers_all_four() {
        let registry =
            subagent_registry(ToolDenials::default(), crate::config::MemoryAccess::Write).await;
        for name in MEMORY_TOOLS {
            assert!(registry.get(name).is_some(), "expected '{}'", name);
        }
    }

    #[tokio::test]
    async fn test_subagent_denied_builtin_is_absent() {
        let registry = subagent_registry(
            ToolDenials::new(Vec::new(), vec!["write_file".to_string()]),
            crate::config::MemoryAccess::Write,
        )
        .await;
        assert!(registry.get("write_file").is_none());
        assert!(
            registry.get("edit_file").is_some(),
            "denying one tool must not take its neighbours"
        );
    }

    /// A denied server's tools never reach the registry even when handed to it directly, which is
    /// the path a mid-run `tools/list_changed` takes.
    #[tokio::test]
    async fn test_subagent_denied_server_tools_are_dropped_on_replace() {
        let registry = subagent_registry(
            ToolDenials::new(vec!["blocked".to_string()], Vec::new()),
            crate::config::MemoryAccess::Write,
        )
        .await;
        registry.replace_server_tools("blocked", vec![Arc::new(StubTool {
            name: "mcp__blocked__send".to_string(),
        })]);
        registry.replace_server_tools("allowed", vec![Arc::new(StubTool {
            name: "mcp__allowed__send".to_string(),
        })]);
        assert!(registry.get("mcp__blocked__send").is_none());
        assert!(registry.get("mcp__allowed__send").is_some());
    }

    /// Denials accumulate. Two `agent_spawn` levels each adding a restriction must end up with
    /// both, or a worker could shed its parent's limits by spawning one more level down.
    #[test]
    fn test_tool_denials_union_accumulates() {
        let parent = ToolDenials::new(vec!["one".to_string()], vec!["alpha".to_string()]);
        let child = parent.union(&ToolDenials::new(vec!["two".to_string()], vec![
            "beta".to_string(),
        ]));
        assert_eq!(child.server_list(), vec!["one", "two"]);
        assert_eq!(child.tool_list(), vec!["alpha", "beta"]);
        assert!(parent.denies_server("one") && !parent.denies_server("two"));
    }

    /// Denying a server denies its tools without naming each one, so the two config keys can't
    /// disagree about a server the user has already ruled out.
    #[test]
    fn test_tool_denials_server_covers_its_tools() {
        let denials = ToolDenials::new(vec!["notion".to_string()], Vec::new());
        assert!(denials.denies_tool("mcp__notion__create_page"));
        assert!(!denials.denies_tool("mcp__linear__create_issue"));
        assert!(!denials.denies_tool("write_file"));
    }

    /// `[subagents]` restricts workers, not the agent doing the delegating. The root registry's own
    /// denial set is empty, so `admits` is what keeps `disabled_tools = ["agent_spawn"]` -- the
    /// natural way to write "workers may not spawn workers" -- from deleting the top-level agent's
    /// ability to delegate at all.
    #[tokio::test]
    async fn test_registry_admits_is_answered_by_the_registry_not_the_config() {
        let root =
            subagent_registry(ToolDenials::default(), crate::config::MemoryAccess::Write).await;
        assert!(root.admits("agent_spawn"));
        assert!(root.admits("agent_delete"));

        let worker = subagent_registry(
            ToolDenials::new(Vec::new(), vec!["agent_spawn".to_string()]),
            crate::config::MemoryAccess::Write,
        )
        .await;
        assert!(!worker.admits("agent_spawn"));
        assert!(worker.admits("agent_list"));
    }

    /// The MCP meta-tools are registered outside `register_builtin`, so `admits` is the only thing
    /// standing between them and a deny list that names them. Before it, `disabled_tools =
    /// ["mcp_resource_read"]` was accepted and silently did nothing.
    #[tokio::test]
    async fn test_registry_admits_covers_the_directly_registered_tools() {
        let registry = subagent_registry(
            ToolDenials::new(Vec::new(), vec!["mcp_resource_read".to_string()]),
            crate::config::MemoryAccess::Write,
        )
        .await;
        assert!(!registry.admits("mcp_resource_read"));
        assert!(registry.admits("mcp_resource_list"));
    }

    /// `allowed_tools` is exhaustive, and before the MCP meta-tools were filterable at all they
    /// registered regardless of it. Applying it to them now would delete seven tools from every
    /// install that has an allow-list, on upgrade, with nothing in the config naming them. The
    /// block-list half does apply, because naming a tool there has always meant "remove this one".
    #[tokio::test]
    async fn test_allowed_tools_does_not_reach_the_mcp_meta_tools() {
        // Asserted through `register_all` rather than the predicate alone: the regression this
        // guards was in the wiring (which predicate `register_all` calls), so a test that only
        // exercised `admits_infrastructure` would pass with the wiring wrong.
        let manager = crate::mcp::McpClientManager::prepare(
            &[crate::mcp::tests::bare_server_config("notes")],
            None,
            None,
            crate::mcp::McpClientContext::new(),
        )
        .await
        .expect("prepare");

        let allow_listed = ToolRegistry::new_with_filter(BuiltinToolFilter::from_config(
            Some(vec!["read_file".to_string()]),
            Vec::new(),
            HashMap::new(),
        ));
        mcp_resources::register_all(&allow_listed, std::sync::Arc::clone(&manager));
        assert!(
            allow_listed.get("mcp_resource_read").is_some(),
            "an exhaustive allowed_tools must not silently take the MCP meta-tools"
        );
        assert!(
            !allow_listed.admits("write_file"),
            "while still biting the tools it always did"
        );

        let block_listed = ToolRegistry::new_with_filter(BuiltinToolFilter::from_config(
            None,
            vec!["mcp_resource_read".to_string()],
            HashMap::new(),
        ));
        mcp_resources::register_all(&block_listed, manager);
        assert!(
            block_listed.get("mcp_resource_read").is_none(),
            "naming one explicitly does remove it"
        );
        assert!(block_listed.get("mcp_prompt_get").is_some());
    }

    /// Every meta-tool name must be a real one, and must be in `BUILTIN_TOOL_NAMES` too — the
    /// allow-list warning treats "not in `BUILTIN_TOOL_NAMES`" and "is a meta-tool" as different
    /// cases, so a name in neither list would fall through both.
    #[test]
    fn test_mcp_meta_tool_names_are_a_subset_of_the_builtins() {
        let known: HashSet<&str> = BUILTIN_TOOL_NAMES.iter().copied().collect();
        for name in MCP_META_TOOL_NAMES {
            assert!(
                known.contains(name),
                "BUILTIN_TOOL_NAMES is missing '{name}'"
            );
        }
        let mut sorted = MCP_META_TOOL_NAMES.to_vec();
        sorted.sort_unstable();
        assert_eq!(
            MCP_META_TOOL_NAMES,
            sorted.as_slice(),
            "keep the list sorted"
        );
        assert_eq!(MCP_META_TOOL_NAMES.len(), 7);
    }

    /// Every name a user may put in `[tools]` or `[subagents]` has to be in `BUILTIN_TOOL_NAMES`,
    /// or the stale-entry warning fires on a correct entry and invites them to "fix" it.
    #[test]
    fn test_builtin_tool_names_covers_the_deniable_families() {
        let known: HashSet<&str> = BUILTIN_TOOL_NAMES.iter().copied().collect();
        for name in [
            "conversation_search",
            "conversation_read",
            "schedule_create",
            "task_list",
            "mcp_resource_read",
            "mcp_resource_updates_list",
            "agent_followup",
        ] {
            assert!(
                known.contains(name),
                "BUILTIN_TOOL_NAMES is missing '{name}'"
            );
        }
        let mut sorted = BUILTIN_TOOL_NAMES.to_vec();
        sorted.sort_unstable();
        assert_eq!(
            BUILTIN_TOOL_NAMES,
            sorted.as_slice(),
            "keep the list sorted"
        );
    }

    #[test]
    fn test_server_of_tool_splits_on_the_first_separator() {
        assert_eq!(server_of_tool("mcp__notion__create_page"), Some("notion"));
        // Tool names may themselves contain `__`; server names may not.
        assert_eq!(server_of_tool("mcp__notion__a__b"), Some("notion"));
        assert_eq!(server_of_tool("write_file"), None);
        assert_eq!(server_of_tool("mcp__malformed"), None);
    }

    #[test]
    fn test_builtin_tool_names_covers_canonical_set() {
        // Guard against forgetting to add a new built-in to the canonical list that drives
        // stale-entry warnings. Update this assertion deliberately when adding a tool in
        // register_core_tools.
        let names: HashSet<&str> = BUILTIN_TOOL_NAMES.iter().copied().collect();
        for expected in &[
            "read_file",
            "write_file",
            "edit_file",
            "find_files",
            "search_contents",
            "execute_command",
            "fetch_url",
            "search_web",
            "todo",
            "scratchpad_read",
            "scratchpad_write",
            "scratchpad_edit",
            "scratchpad_list",
            "scratchpad_delete",
            "skill",
            "render_image",
            "agent_spawn",
            "load_tool",
        ] {
            assert!(
                names.contains(expected),
                "BUILTIN_TOOL_NAMES missing '{}'",
                expected
            );
        }
    }

    /// Stub tool that sleeps for a known duration before returning a payload derived from its
    /// input. Observes the cancellation token via `select!` so cancellation tests can assert early
    /// exit. Used to verify the parent and sub-agent dispatch loops actually run their `join_all`
    /// futures in parallel and propagate cancellation correctly.
    struct SleepTool {
        name: String,
        delay: std::time::Duration,
    }

    #[async_trait]
    impl Tool for SleepTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: self.name.clone(),
                description: "test sleep tool".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "label": { "type": "string" }
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
            let label = input
                .get("label")
                .and_then(|v| v.as_str())
                .unwrap_or("default")
                .to_string();
            tokio::select! {
                _ = tokio::time::sleep(self.delay) => {
                    Ok(ToolOutput::text(format!("done:{}", label), false))
                }
                _ = cancellation.cancelled() => {
                    Ok(ToolOutput::text(format!("cancelled:{}", label), true))
                }
            }
        }
    }

    /// Guards against a regression where parallel tool dispatch is replaced by sequential
    /// `.await`-in-a-loop. Two tools each sleep ~200 ms; the total wall-clock must be much less
    /// than the sum.
    #[tokio::test]
    async fn test_parallel_dispatch_runs_tools_concurrently() {
        let registry = ToolRegistry::new_with_filter(BuiltinToolFilter::default());
        registry
            .register(Arc::new(SleepTool {
                name: "sleep_one".to_string(),
                delay: std::time::Duration::from_millis(200),
            }))
            .expect("registration should succeed");
        registry
            .register(Arc::new(SleepTool {
                name: "sleep_two".to_string(),
                delay: std::time::Duration::from_millis(200),
            }))
            .expect("registration should succeed");

        let tools = [
            ("a", "sleep_one", serde_json::json!({ "label": "first" })),
            ("b", "sleep_two", serde_json::json!({ "label": "second" })),
        ];
        let cancellation = CancellationToken::new();

        let start = std::time::Instant::now();
        let futures = tools.iter().map(|(_, name, input)| {
            let tool = registry.get(name).expect("tool registered above");
            let cancellation = cancellation.clone();
            async move { tool.execute(input.clone(), cancellation).await }
        });
        let outputs: Vec<_> = futures::future::join_all(futures).await;
        let elapsed = start.elapsed();

        // 500ms gives ~300ms headroom over the parallel ~200ms baseline while still being well
        // below the ~400ms serial-dispatch case. The wide margin absorbs scheduler jitter on slow
        // CI runners without losing the parallel-vs-sequential discrimination.
        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "expected parallel execution (<500ms), got {:?}",
            elapsed
        );
        assert_eq!(outputs.len(), 2);
        let first = outputs[0].as_ref().expect("first should succeed");
        let second = outputs[1].as_ref().expect("second should succeed");
        assert_eq!(text_content(first), "done:first");
        assert_eq!(text_content(second), "done:second");
    }

    /// Verifies that the cancellation token threads through `tool.execute(...)` calls when many are
    /// in flight. Cancelling mid-batch should cause every running tool to observe the cancellation
    /// and return early.
    #[tokio::test]
    async fn test_parallel_dispatch_respects_cancellation() {
        let registry = ToolRegistry::new_with_filter(BuiltinToolFilter::default());
        registry
            .register(Arc::new(SleepTool {
                name: "long_one".to_string(),
                delay: std::time::Duration::from_secs(10),
            }))
            .expect("registration should succeed");
        registry
            .register(Arc::new(SleepTool {
                name: "long_two".to_string(),
                delay: std::time::Duration::from_secs(10),
            }))
            .expect("registration should succeed");

        let tools = [
            ("a", "long_one", serde_json::json!({ "label": "first" })),
            ("b", "long_two", serde_json::json!({ "label": "second" })),
        ];
        let cancellation = CancellationToken::new();

        let cancel_handle = {
            let cancellation = cancellation.clone();
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                cancellation.cancel();
            })
        };

        let start = std::time::Instant::now();
        let futures = tools.iter().map(|(_, name, input)| {
            let tool = registry.get(name).expect("tool registered above");
            let cancellation = cancellation.clone();
            async move { tool.execute(input.clone(), cancellation).await }
        });
        let outputs: Vec<_> = futures::future::join_all(futures).await;
        let elapsed = start.elapsed();
        cancel_handle.await.expect("cancel task should not panic");

        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "expected early exit on cancellation (<500ms), got {:?}",
            elapsed
        );
        for (i, output) in outputs.iter().enumerate() {
            let output = output.as_ref().expect("execute should not error");
            assert!(
                output.is_error,
                "tool {} should report cancellation as is_error=true",
                i
            );
            assert!(
                text_content(output).starts_with("cancelled:"),
                "tool {} should report cancellation, got {:?}",
                i,
                output.content
            );
        }
    }
}
