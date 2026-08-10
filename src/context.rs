//! Builds the system prompt and per-turn context: permission state, environment info (PWD, date,
//! shell, OS), todo list, tool catalogue, and skill summaries.
//!
//! The split between the two is a caching decision, not an organisational one. Prompt caching is
//! prefix-based and the system prompt heads that prefix, so anything rendered into it that later
//! changes re-caches the tools array and the whole conversation behind it. [`build_system_prompt`]
//! therefore takes only inputs that are fixed for a session, and everything mutable travels in the
//! per-turn `<context>` block, which is appended to the conversation like any other message.
//!
//! What lives in the block, and why each would otherwise invalidate the prefix:
//!
//! - **permission level** and the tools it blocks: `/permission` and Shift+Tab change it mid-turn.
//! - **cwd and workspace roots**: `/cd`, and an ACP client re-sending `additionalDirectories`.
//! - **todo list**: rewritten by the `todo` tool.
//! - **tools, skills, MCP server instructions** ([`WorldSnapshot`]): skills are re-read from disk
//!   every turn, MCP servers connect late and can hot-swap their tool lists.
//!
//! The last group is diffed rather than re-sent: an unchanged turn renders nothing at all.

use crate::{
    memory::Memory,
    permission::Permission,
    session::ToolOutputSummary,
    skills::Skill,
    tools::todo::{self, TodoState},
};

/// A tool's entry in the catalogue rendered into the per-turn `<context>` block. Tuple:
/// `(name, description, required_permission, is_deferred)`. Produced by
/// [`crate::tools::ToolRegistry::tool_catalogue`].
pub type ToolCatalogueEntry = (String, String, Permission, bool);

/// Per-entry cap for a deferred tool's one-line summary. Keeps the rendered catalogue bounded when
/// MCP servers advertise 2 KB descriptions.
const TOOL_SUMMARY_MAX_CHARS: usize = 160;

/// Names of the seven built-in MCP-resource helper tools (defined in `src/tools/mcp_resources.rs`).
/// They share no common simple prefix, so they're enumerated explicitly. Used to group deferred
/// entries into the `MCP resource tools` subsection of `[Tool discovery]`.
const MCP_RESOURCE_TOOLS: &[&str] = &[
    "list_mcp_resources",
    "read_mcp_resource",
    "list_mcp_prompts",
    "get_mcp_prompt",
    "subscribe_mcp_resource",
    "unsubscribe_mcp_resource",
    "list_mcp_resource_updates",
];

/// The mutable half of what the model knows: which tools exist, which skills are installed, and
/// what each connected MCP server said about itself.
///
/// Kept out of the system prompt because all three change mid-session. Skills are re-read from disk
/// every turn, an MCP server can connect late, and `tools/list_changed` swaps a server's tools
/// wholesale. Rendering any of that into the cached prefix means a re-cache of the whole
/// conversation the first time it moves; rendering it into the per-turn `<context>` block costs an
/// append instead.
///
/// Tools, skills, and MCP instructions are `BTreeMap`s, so a snapshot has one canonical form and
/// equality is a real "did the model's picture change" test rather than an ordering accident.
/// Memories are a `Vec` because their order is meaningful (see [`WorldSnapshot::memories`]) and
/// already canonical when it arrives.
/// One memory's line in the `[Memory]` index. Carries `mtime` rather than a rendered age so the
/// snapshot compares equal across a midnight boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
struct MemoryIndexEntry {
    name: String,
    description: String,
    priority: u8,
    mtime: std::time::SystemTime,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WorldSnapshot {
    /// Tool name → `(required permission, deferred, one-line summary)`.
    tools: std::collections::BTreeMap<String, (Permission, bool, String)>,
    /// Skill name → description.
    skills: std::collections::BTreeMap<String, String>,
    /// Memory name → index entry, in the order [`crate::memory::sort_for_index`] produced.
    ///
    /// A `Vec`, not a map, because the order *is* the ranking and the budget cuts from the end.
    /// The entry holds `mtime` rather than a rendered age so snapshot equality is stable across
    /// days: rendering "14 days ago" into the snapshot would make every midnight look like a
    /// world change and force a full re-render.
    memories: Vec<MemoryIndexEntry>,
    /// MCP server name → its `initialize` instructions.
    mcp_instructions: std::collections::BTreeMap<String, String>,
}

/// The tool each store's index exists to drive. An index is a menu: without the tool that opens an
/// entry, listing the entries is a promise the model cannot act on.
const SKILL_INDEX_TOOL: &str = "skill";
const MEMORY_INDEX_TOOL: &str = "memory_read";

/// Whether `name` is registered, deferred or not. A deferred tool still counts: its schema is
/// withheld until `load_tool` fetches it, but the model can reach it.
fn catalogue_has(catalogue: &[ToolCatalogueEntry], name: &str) -> bool {
    catalogue.iter().any(|(entry, ..)| entry == name)
}

impl WorldSnapshot {
    /// Build the picture the model will be shown.
    ///
    /// Each store's index is dropped when the tool that opens it is not registered. That happens
    /// through `[skills] enabled` / `[memory] enabled`, which also empty the caches, but equally
    /// through `[tools] disabled_tools = ["skill"]`, which does not - and without this filter the
    /// section would keep instructing the model to call a tool that no longer exists. Gating here
    /// rather than at render time means the snapshot records what the model was *told*, so the
    /// diff and the equality check stay honest.
    pub fn new(
        catalogue: &[ToolCatalogueEntry],
        skills: &[Skill],
        memories: &[Memory],
        mcp_server_instructions: &[(String, String)],
    ) -> Self {
        let skills: &[Skill] = if catalogue_has(catalogue, SKILL_INDEX_TOOL) {
            skills
        } else {
            &[]
        };
        let memories: &[Memory] = if catalogue_has(catalogue, MEMORY_INDEX_TOOL) {
            memories
        } else {
            &[]
        };
        Self {
            tools: catalogue
                .iter()
                .map(|(name, description, required, deferred)| {
                    (
                        name.clone(),
                        (*required, *deferred, short_description(description)),
                    )
                })
                .collect(),
            skills: skills
                .iter()
                .map(|skill| (skill.name.clone(), skill.description.clone()))
                .collect(),
            memories: memories
                .iter()
                .map(|memory| MemoryIndexEntry {
                    name: memory.name.clone(),
                    description: memory.description.clone(),
                    priority: memory.priority,
                    mtime: memory.mtime,
                })
                .collect(),
            mcp_instructions: mcp_server_instructions
                .iter()
                .map(|(server, body)| (server.clone(), body.trim_end().to_string()))
                .collect(),
        }
    }
}

/// Bucket deferred catalogue entries by source for the `[Tool discovery]`
/// section. Returns `(heading, entries)` pairs in a deterministic order:
/// scratchpad operations, MCP resource tools, then per-MCP-server groups
/// alphabetically, then a catch-all bucket for any deferred tool that
/// matches none of those classifiers.
fn group_deferred_entries<'a>(
    deferred: &[&'a ToolCatalogueEntry],
) -> Vec<(String, Vec<&'a ToolCatalogueEntry>)> {
    let mcp_resource_set: std::collections::HashSet<&str> =
        MCP_RESOURCE_TOOLS.iter().copied().collect();

    let mut scratchpad: Vec<&ToolCatalogueEntry> = Vec::new();
    let mut mcp_resource: Vec<&ToolCatalogueEntry> = Vec::new();
    let mut mcp_servers: std::collections::BTreeMap<String, Vec<&ToolCatalogueEntry>> =
        std::collections::BTreeMap::new();
    let mut other: Vec<&ToolCatalogueEntry> = Vec::new();

    for entry in deferred {
        let name = entry.0.as_str();
        if name.starts_with("scratchpad_") {
            scratchpad.push(entry);
        } else if mcp_resource_set.contains(name) {
            mcp_resource.push(entry);
        } else if let Some(rest) = name.strip_prefix("mcp__") {
            // Format: `mcp__<server>__<tool>`. Split on the first `__` to isolate the server name;
            // tools without the second separator are unexpected but bucketed under the literal
            // first segment so we don't lose them.
            let server = rest.split("__").next().unwrap_or(rest).to_string();
            mcp_servers.entry(server).or_default().push(entry);
        } else {
            other.push(entry);
        }
    }

    let mut groups: Vec<(String, Vec<&ToolCatalogueEntry>)> = Vec::new();
    if !scratchpad.is_empty() {
        groups.push(("Scratchpad operations".to_string(), scratchpad));
    }
    if !mcp_resource.is_empty() {
        groups.push(("MCP resource tools".to_string(), mcp_resource));
    }
    for (server, entries) in mcp_servers {
        groups.push((format!("MCP server: {}", server), entries));
    }
    if !other.is_empty() {
        groups.push(("Other".to_string(), other));
    }
    groups
}

/// Collapse whitespace, keep the first sentence, clamp to [`TOOL_SUMMARY_MAX_CHARS`], append `…` if
/// clipped.
fn short_description(description: &str) -> String {
    let collapsed: String = {
        let mut out = String::with_capacity(description.len());
        let mut prev_space = false;
        for ch in description.chars() {
            if ch.is_whitespace() {
                if !prev_space && !out.is_empty() {
                    out.push(' ');
                }
                prev_space = true;
            } else {
                out.push(ch);
                prev_space = false;
            }
        }
        out.trim_end().to_string()
    };

    if collapsed.is_empty() {
        return collapsed;
    }

    // Find the first sentence terminator followed by whitespace or EOS. Walks by char to avoid
    // slicing a multi-byte UTF-8 scalar, and recognises CJK fullwidth punctuation (。！？)
    // alongside ASCII so descriptions in non-Western scripts get the same treatment.
    let mut sentence_end_byte: Option<usize> = None;
    let mut prev_term: Option<(char, usize)> = None;
    for (byte_idx, ch) in collapsed.char_indices() {
        if let Some((_, term_end)) = prev_term {
            if ch.is_whitespace() {
                sentence_end_byte = Some(term_end);
                break;
            }
            prev_term = None;
        }
        if matches!(ch, '.' | '!' | '?' | '。' | '！' | '？') {
            prev_term = Some((ch, byte_idx + ch.len_utf8()));
        }
    }
    // Terminator at end-of-string counts as a sentence boundary.
    if sentence_end_byte.is_none()
        && let Some((_, term_end)) = prev_term
        && term_end == collapsed.len()
    {
        sentence_end_byte = Some(term_end);
    }

    let candidate = match sentence_end_byte {
        Some(end) => collapsed[..end].to_string(),
        None => collapsed.clone(),
    };

    if candidate.chars().count() <= TOOL_SUMMARY_MAX_CHARS {
        return candidate;
    }

    // Char-cap fallback. Walking by char preserves UTF-8 boundaries without relying on the unstable
    // `floor_char_boundary`.
    let clipped: String = candidate.chars().take(TOOL_SUMMARY_MAX_CHARS).collect();
    format!("{}…", clipped.trim_end())
}

/// OS description for the system prompt's environment block, detected once. Probing the OS is
/// blocking I/O (`sw_vers` subprocess on macOS, `/etc/os-release` read on Linux); the system prompt
/// is rebuilt every turn from the async agent loop, so the result is cached process-wide.
static OS_DESCRIPTION: std::sync::LazyLock<Option<String>> =
    std::sync::LazyLock::new(detect_os_description);

fn detect_os_description() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let info = std::fs::read_to_string("/etc/os-release").ok()?;
        info.lines()
            .find_map(|line| line.strip_prefix("PRETTY_NAME="))
            .map(|name| name.trim_matches('"').to_string())
    }
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("sw_vers")
            .arg("-productVersion")
            .output()
            .ok()?;
        let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
        (!version.is_empty()).then(|| format!("macOS {}", version))
    }
    #[cfg(target_os = "windows")]
    {
        Some("Windows".to_string())
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

/// Build the session-level system prompt: role, permission model, user instructions, guidelines,
/// and environment info.
///
/// **Every input here is fixed for the lifetime of a session**, which is the point. The system
/// prompt is the head of the cached prefix, and prompt caching is prefix-based, so a single byte
/// changing here re-caches the tools array and the entire conversation behind it. Both parameters
/// come from [`crate::agent::AgentOptions`], which is constructed once and never rebuilt, so this
/// function cannot render differently twice in one session.
///
/// The tool catalogue, skills, and MCP server instructions used to live here and are the reason
/// that guarantee did not hold: all three can change mid-session. They now travel in the per-turn
/// `<context>` block via [`WorldSnapshot`], which is appended rather than mutated. The narrow
/// signature is the enforcement mechanism: there is nothing dynamic left to pass in.
pub fn build_system_prompt(sandboxed_shell: bool, user_instructions: Option<&str>) -> String {
    let mut prompt = String::new();

    prompt.push_str(
        "You are meka, a general-purpose AI agent. The user communicates with you \
         in natural language, and you execute their requests using the available tools.\n\n",
    );

    prompt.push_str("## Permission Model\n\n");
    prompt.push_str(
        "meka runs at a graduated permission level that the user can change mid-session \
         by pressing Shift+Tab or typing `/permission <level>`. Levels, from least to \
         most powerful:\n\n",
    );
    prompt.push_str("- `none`: text-only, no tools may execute.\n");
    if sandboxed_shell {
        prompt.push_str(
            "- `read`: read-only tools (file reads, search, web fetch). `execute_command` \
             runs with the filesystem mounted read-only. Commands that write to disk fail.\n",
        );
    } else {
        prompt.push_str(
            "- `read`: read-only tools (file reads, search, web fetch). `execute_command` \
             is blocked at this level.\n",
        );
    }
    prompt.push_str(
        "- `ask`: full tool access; each tool call is presented to the user for \
         approval before execution.\n",
    );
    prompt.push_str("- `write`: full tool access, no approval required.\n\n");
    prompt.push_str(
        "The current level, and the set of tools it does NOT allow, is delivered in \
         the per-turn `[Permission context]` block of each user message. If the user \
         asks for an operation their current level blocks, name the required tool and \
         suggest they run `/permission <level>` (or Shift+Tab) to enable it. For \
         potentially destructive operations at `write`, briefly explain what you will \
         do before proceeding.\n\n",
    );

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

    prompt.push_str("## Guidelines\n\n");
    prompt.push_str("- Format your responses in Markdown.\n");
    prompt.push_str("- When executing shell commands, show the command you are about to run.\n");
    prompt.push_str(
        "- For potentially destructive operations, explain what you will do before proceeding.\n",
    );
    prompt.push_str(
        "- If a tool returns an error, explain the error to the user and suggest alternatives.\n",
    );
    prompt.push_str("- Be concise but thorough.\n\n");

    prompt.push_str("## Environment\n\n");

    if let Ok(shell) = std::env::var("SHELL") {
        prompt.push_str(&format!("- Shell: {}\n", shell));
    }

    if let Some(os) = &*OS_DESCRIPTION {
        prompt.push_str(&format!("- OS: {}\n", os));
    }

    prompt
}

/// Render `current` for the per-turn `<context>` block, relative to what the model was last told.
///
/// - `previous == Some(same)` → `""`. The steady-state path, and the one that matters: an unchanged
///   session must add nothing per turn, or this would cost more than the cache it protects.
/// - `previous == None` → the full picture. Used on the first turn of a session and again after a
///   compaction, which rewrites the head of the conversation and can summarize the earlier
///   rendering away.
/// - otherwise → only what changed, carrying explicit replacement wording so the model treats the
///   new text as superseding the old rather than adding to it.
pub fn render_world_state(current: &WorldSnapshot, previous: Option<&WorldSnapshot>) -> String {
    let Some(previous) = previous else {
        return render_world_state_full(current);
    };
    if previous == current {
        return String::new();
    }
    render_world_state_diff(current, previous)
}

/// The whole picture, for a session's first turn and for the turn after a compaction.
fn render_world_state_full(current: &WorldSnapshot) -> String {
    let catalogue: Vec<ToolCatalogueEntry> = current
        .tools
        .iter()
        .map(|(name, (required, deferred, summary))| {
            (name.clone(), summary.clone(), *required, *deferred)
        })
        .collect();
    let active: Vec<&ToolCatalogueEntry> = catalogue.iter().filter(|(_, _, _, d)| !d).collect();
    let deferred: Vec<&ToolCatalogueEntry> = catalogue.iter().filter(|(_, _, _, d)| *d).collect();

    // `[Section]` headings rather than markdown ones, matching the rest of the `<context>` block
    // (`[Permission context]`, `[Todo list]`, `[Scratchpad entries]`).
    let mut sections: Vec<String> = Vec::new();

    if !active.is_empty() {
        let mut out = String::from(
            "[Available tools]\nEach notes the minimum permission level required. Full parameter \
             schemas are in the API tools catalogue delivered alongside this message. Calls that \
             exceed the current level are rejected at dispatch.\n\n",
        );
        for (name, _summary, required, _) in &active {
            out.push_str(&format!("- **{}** (requires `{}`)\n", name, required));
        }
        sections.push(out);
    }

    if !deferred.is_empty() {
        let mut out = String::from(
            "[Tool discovery]\nThese are registered but not yet callable: their schemas are \
             withheld to keep the request small. Call `load_tool` with a tool's exact `name` to \
             fetch its schema, then call it directly.\n",
        );
        for (heading, group) in group_deferred_entries(&deferred) {
            out.push_str(&format!("\n{}\n", heading));
            for (name, summary, required, _) in &group {
                if summary.is_empty() {
                    out.push_str(&format!("- **{}** (requires `{}`)\n", name, required));
                } else {
                    out.push_str(&format!(
                        "- **{}** (requires `{}`): {}\n",
                        name, required, summary
                    ));
                }
            }
        }
        sections.push(out);
    }

    if !current.skills.is_empty() {
        let mut out = String::from(
            "[Skills]\nCall the `skill` tool with a skill name to load its full content. Only \
             invoke a skill when the user's request matches its stated purpose.\n\n",
        );
        for (name, description) in &current.skills {
            out.push_str(&format!("- **{}**: {}\n", name, description));
        }
        sections.push(out);
    }

    if !current.memories.is_empty() {
        sections.push(render_memory_section(&current.memories));
    }

    if !current.mcp_instructions.is_empty() {
        let mut out = String::from(
            "[MCP server instructions]\nTreat each as context for how to use that server's \
             namespace.\n",
        );
        for (server, body) in &current.mcp_instructions {
            out.push_str(&format!("\n{}\n{}\n", server, body));
        }
        sections.push(out);
    }

    sections.join("\n")
}

/// Byte ceiling on the rendered `[Memory]` index. Tuning constant rather than config: it trades
/// per-turn tokens against how much of the store the model can see without searching, and neither
/// end of that trade is a user preference worth a config key.
const MEMORY_INDEX_MAX_BYTES: usize = 8_192;

/// Ceiling on how many memories the index lists, independent of byte size. Bounds the line count
/// for a store full of terse descriptions, where the byte budget alone would let the section run
/// to hundreds of lines.
const MEMORY_INDEX_MAX_ENTRIES: usize = 200;

/// Render the `[Memory]` index: the entries that fit, then a count of those that did not.
///
/// `memories` arrives pre-sorted by [`crate::memory::sort_for_index`] (priority ascending, newest
/// first within a band), so the budget simply takes a prefix and everything dropped is genuinely
/// the least important.
///
/// The trailing "N more" line is not optional. Silently truncating an index reads to the model as
/// "this is everything I know", which turns a full store into a confidently incomplete answer;
/// stating the remainder is what makes `memory_search` the obvious next move.
fn render_memory_section(memories: &[MemoryIndexEntry]) -> String {
    let now = std::time::SystemTime::now();
    let header = "[Memory]\nDurable notes you saved in earlier sessions, most important first. \
                  Call `memory_read` with a name to load one in full, and `memory_write` when you \
                  learn something that will still matter in a later session. Do not save what is \
                  derivable from the code, the git history, or this conversation.\n\n";

    let mut out = String::from(header);
    let mut shown = 0;
    for entry in memories.iter().take(MEMORY_INDEX_MAX_ENTRIES) {
        let line = format!(
            "- **{}** (p{}, {}): {}\n",
            entry.name,
            entry.priority,
            crate::memory::render_age(entry.mtime, now),
            entry.description
        );
        // Always emit at least one entry: a single pathological description longer than the whole
        // budget should still be visible rather than collapsing the section to a bare count.
        if shown > 0 && out.len() + line.len() > MEMORY_INDEX_MAX_BYTES {
            break;
        }
        out.push_str(&line);
        shown += 1;
    }

    let hidden = memories.len().saturating_sub(shown);
    if hidden > 0 {
        out.push_str(&format!(
            "\n{} more {} not shown here — use `memory_search` to find {}.\n",
            hidden,
            if hidden == 1 { "memory" } else { "memories" },
            if hidden == 1 { "it" } else { "them" },
        ));
    }
    out
}

/// Only what moved since the model was last told, phrased so the new text supersedes the old.
fn render_world_state_diff(current: &WorldSnapshot, previous: &WorldSnapshot) -> String {
    let mut lines: Vec<String> = Vec::new();

    let newly_callable: Vec<&String> = current
        .tools
        .iter()
        .filter(|(name, (_, deferred, _))| {
            !deferred && !matches!(previous.tools.get(*name), Some((_, false, _)))
        })
        .map(|(name, _)| name)
        .collect();
    // Described rather than merely named: a deferred tool's schema is withheld, so this one-line
    // summary is all the model has to decide whether the tool is worth a `load_tool` round trip.
    // A late-connecting MCP server can announce eighty tools at once, and eighty bare names are
    // not a catalogue. Newly *callable* tools need no summary here because their full schema ships
    // in the API tools array; only the permission, which is meka's own concept, has to be stated.
    let newly_deferred: Vec<String> = current
        .tools
        .iter()
        .filter(|(name, (_, deferred, _))| {
            *deferred && !matches!(previous.tools.get(*name), Some((_, true, _)))
        })
        .map(|(name, facts)| describe_tool(name, facts))
        .collect();
    let gone: Vec<&String> = previous
        .tools
        .keys()
        .filter(|name| !current.tools.contains_key(*name))
        .collect();
    // A tool whose name and callability both held still but whose facts moved underneath: an MCP
    // server reconnecting with a reworded description for the same tool. Without this bucket the
    // snapshot would advance while nothing was said, leaving the model working from a stale
    // summary for the rest of the session.
    let restated: Vec<String> = current
        .tools
        .iter()
        .filter_map(|(name, facts)| {
            let before = previous.tools.get(name)?;
            (before != facts && before.1 == facts.1).then(|| describe_tool(name, facts))
        })
        .collect();

    if !newly_callable.is_empty() {
        lines.push(format!(
            "- Now callable: {}",
            join_names(newly_callable.into_iter())
        ));
    }
    if !newly_deferred.is_empty() {
        lines.push(format!(
            "- Now registered but not yet callable (use `load_tool`): {}",
            newly_deferred.join("; "),
        ));
    }
    if !gone.is_empty() {
        lines.push(format!(
            "- No longer available, do not call: {}",
            join_names(gone.into_iter())
        ));
    }
    if !restated.is_empty() {
        lines.push(format!("- Redescribed: {}", restated.join("; ")));
    }

    let added_skills: Vec<String> = current
        .skills
        .iter()
        .filter(|(name, description)| previous.skills.get(*name) != Some(description))
        .map(|(name, description)| format!("{} ({})", name, description))
        .collect();
    let removed_skills: Vec<&String> = previous
        .skills
        .keys()
        .filter(|name| !current.skills.contains_key(*name))
        .collect();
    if !added_skills.is_empty() {
        lines.push(format!(
            "- Skills added or updated: {}",
            added_skills.join("; ")
        ));
    }
    if !removed_skills.is_empty() {
        lines.push(format!(
            "- Skills no longer available: {}",
            join_names(removed_skills.into_iter())
        ));
    }

    // Memories move whenever the agent writes one, which is often, so the diff carries the delta
    // rather than re-listing the index. Priority is included because it decides where the entry
    // will sit when the index is next stated in full.
    let previous_memories: std::collections::HashMap<&str, &MemoryIndexEntry> = previous
        .memories
        .iter()
        .map(|entry| (entry.name.as_str(), entry))
        .collect();
    let changed_memories: Vec<String> = current
        .memories
        .iter()
        .filter(|entry| {
            // Compare only what the model was told - priority and description - not `mtime`.
            // Rewriting a memory with identical content moves its mtime, and re-announcing "saved
            // or updated" for a note that reads exactly as before is noise. The mtime still rides
            // in the snapshot, because it decides ordering the next time the index renders in full.
            previous_memories
                .get(entry.name.as_str())
                .is_none_or(|before| {
                    before.priority != entry.priority || before.description != entry.description
                })
        })
        .map(|entry| {
            format!(
                "{} (p{}: {})",
                entry.name, entry.priority, entry.description
            )
        })
        .collect();
    let removed_memories: Vec<&String> = previous
        .memories
        .iter()
        .filter(|entry| {
            !current
                .memories
                .iter()
                .any(|candidate| candidate.name == entry.name)
        })
        .map(|entry| &entry.name)
        .collect();
    if !changed_memories.is_empty() {
        lines.push(format!(
            "- Memories saved or updated: {}",
            changed_memories.join("; ")
        ));
    }
    if !removed_memories.is_empty() {
        lines.push(format!(
            "- Memories deleted: {}",
            join_names(removed_memories.into_iter())
        ));
    }

    let mut server_blocks: Vec<String> = Vec::new();
    for (server, body) in &current.mcp_instructions {
        if previous.mcp_instructions.get(server) != Some(body) {
            server_blocks.push(format!(
                "Instructions from MCP server {} (these replace any instructions it provided \
                 earlier):\n{}",
                server, body,
            ));
        }
    }
    let dropped_servers: Vec<&String> = previous
        .mcp_instructions
        .keys()
        .filter(|server| !current.mcp_instructions.contains_key(*server))
        .collect();
    if !dropped_servers.is_empty() {
        lines.push(format!(
            "- Instructions from these MCP servers no longer apply: {}",
            join_names(dropped_servers.into_iter()),
        ));
    }

    if lines.is_empty() && server_blocks.is_empty() {
        return String::new();
    }

    let mut out = String::from(
        "[Tool and skill changes]\nThe following supersedes what was stated earlier in this \
         conversation.\n",
    );
    if !lines.is_empty() {
        out.push('\n');
        out.push_str(&lines.join("\n"));
        out.push('\n');
    }
    for block in server_blocks {
        out.push('\n');
        out.push_str(&block);
        out.push('\n');
    }
    out
}

/// One deferred-tool line, matching the `[Tool discovery]` shape of the full render so the model
/// reads the same format whether it arrived as an initial listing or as a later change.
fn describe_tool(name: &str, (required, _, summary): &(Permission, bool, String)) -> String {
    if summary.is_empty() {
        format!("`{}` (requires `{}`)", name, required)
    } else {
        format!("`{}` (requires `{}`): {}", name, required, summary)
    }
}

fn join_names<'a>(names: impl Iterator<Item = &'a String>) -> String {
    names
        .map(|name| format!("`{}`", name))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Build the per-turn `[Permission context]` block. Names the current permission level plus a
/// one-line statement of what tools can execute at that level. The `[Available tools]` catalogue
/// already lists every tool's required level, so the per-turn block stays short and bounded
/// regardless of how many tools are registered. Permission-dependent content lives here, NOT in
/// the system prompt, so `/permission` toggles don't invalidate the cached prefix.
pub fn build_permission_context(permission: Permission) -> String {
    let summary = match permission {
        Permission::None => "No tools are executable.",
        Permission::Read => "Only read-only tools are executable.",
        Permission::Ask => "All tools are executable, but each call requires user approval.",
        Permission::Write => "All tools are executable.",
    };
    format!(
        "[Permission context]\nCurrent permission level: {}\n{}\n",
        permission, summary
    )
}

/// Build the per-turn environment context block (pwd, extra workspace roots, date). Returns an
/// empty string in `None` permission mode so system info isn't leaked. The `cwd` argument is the
/// agent's per-session working directory; passing it explicitly (rather than reading process state)
/// lets multiple sessions in one process report their own cwds correctly.
///
/// `roots` are the workspace roots beyond `cwd` (an ACP client's `additionalDirectories`). Naming
/// them is the whole point of tracking them: without this line the model has no way to learn the
/// other folders exist, and would report a file it cannot find as absent rather than looking. Emits
/// nothing when the list is empty, so single-root output is unchanged.
pub fn build_environment_context(
    permission: Permission,
    cwd: &std::path::Path,
    roots: &[std::path::PathBuf],
) -> String {
    if permission == Permission::None {
        return String::new();
    }

    let mut context = String::from("[Environment context]\n");
    context.push_str(&format!("Working directory: {}\n", cwd.display()));

    if !roots.is_empty() {
        context.push_str(
            "Additional workspace roots (searched alongside the working directory; \
             relative paths still resolve against the working directory):\n",
        );
        for root in roots {
            context.push_str(&format!("  {}\n", root.display()));
        }
    }

    let now = chrono::Local::now().to_rfc2822();
    context.push_str(&format!("Date: {}\n", now));

    context
}

/// Build the `<context>...</context>` block that wraps per-turn user input with permission state,
/// the active todo list, environment info, and any world-state change. The `[Permission context]`
/// section is always included so the model sees the current level on every turn.
///
/// `world_state` comes from [`render_world_state`] and is empty on a turn where nothing changed,
/// which is the normal case. Everything here rides inside the user's own message, so it is appended
/// to the conversation rather than mutating the cached prefix ahead of it.
pub fn build_turn_context(
    permission: Permission,
    todos: &TodoState,
    cwd: &std::path::Path,
    roots: &[std::path::PathBuf],
    world_state: &str,
) -> String {
    let mut sections = Vec::new();

    sections.push(build_permission_context(permission));

    if !todos.items.is_empty() {
        sections.push(todo::format_todo_state(todos));
    }

    let environment_context = build_environment_context(permission, cwd, roots);
    if !environment_context.is_empty() {
        sections.push(environment_context);
    }

    if !world_state.is_empty() {
        sections.push(world_state.to_string());
    }

    format!("<context>\n{}</context>", sections.join("\n"))
}

/// Build the post-compaction context block summarizing live session state (environment, todos,
/// scratchpad inventory) that must persist across the compacted message window.
pub fn build_post_compact_context(
    permission: Permission,
    todos: &TodoState,
    scratchpad_entries: &[ToolOutputSummary],
    cwd: &std::path::Path,
    roots: &[std::path::PathBuf],
) -> String {
    let mut parts = Vec::new();

    let env = build_environment_context(permission, cwd, roots);
    if !env.is_empty() {
        parts.push(env);
    }

    if !todos.items.is_empty() {
        parts.push(todo::format_todo_state(todos));
    }

    if !scratchpad_entries.is_empty() {
        let mut listing = String::from("[Scratchpad entries]\n");
        for entry in scratchpad_entries {
            listing.push_str(&format!(
                "- \"{}\" ({})\n",
                entry.name,
                crate::tools::scratchpad::format_size(entry.size),
            ));
        }
        parts.push(listing);
    }

    // The summary above replaces the earlier turns; if a needed detail is missing from it, the full
    // history is still searchable. `recall` is a Read-tier tool, so the nudge only applies when
    // tools can run at all (not in `none` mode).
    if permission != Permission::None {
        parts.push(
            "[Earlier turns were summarized above. If you need a detail the summary omitted, use \
             the `recall` tool to search the full conversation history and `recall_read` to read a \
             specific turn.]"
                .to_string(),
        );
    }

    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_memory(name: &str, priority: u8, description: &str, age_days: u64) -> Memory {
        Memory {
            name: name.to_string(),
            description: description.to_string(),
            priority,
            path: std::path::PathBuf::from("/tmp").join(format!("{}.md", name)),
            mtime: std::time::SystemTime::now() - std::time::Duration::from_secs(age_days * 86_400),
        }
    }

    /// A memory index that fits the budget lists everything and adds no "N more" line.
    #[test]
    fn test_memory_section_lists_everything_under_budget() {
        let memories = [
            sample_memory("standing-rule", 1, "Always reply in kind", 0),
            sample_memory("a-fact", 5, "The NAS is at nas.lan", 14),
        ];
        let rendered = world_state_for_memories(&memories);

        assert!(rendered.contains("[Memory]"), "{rendered}");
        assert!(
            rendered.contains("**standing-rule** (p1, today)"),
            "{rendered}"
        );
        assert!(
            rendered.contains("**a-fact** (p5, 14 days ago)"),
            "{rendered}"
        );
        assert!(
            !rendered.contains("more memories"),
            "nothing was dropped, so no truncation notice belongs here: {rendered}"
        );
    }

    /// Truncation must always announce itself. A silently shortened index reads to the model as
    /// "this is everything I know", which turns a full store into a confidently incomplete answer.
    #[test]
    fn test_memory_section_always_announces_truncation() {
        // Long descriptions so the byte budget bites well before the entry cap.
        let filler = "x".repeat(400);
        let memories: Vec<Memory> = (0..100)
            .map(|index| sample_memory(&format!("memory-{index:03}"), 5, &filler, index))
            .collect();
        let rendered = world_state_for_memories(&memories);

        assert!(
            rendered.len() < 12_000,
            "budget must bite: {}",
            rendered.len()
        );
        let notice = rendered
            .lines()
            .find(|line| line.contains("more memories not shown"))
            .unwrap_or_else(|| panic!("truncation must be announced; got:\n{rendered}"));
        assert!(notice.contains("memory_search"), "{notice}");

        // The count has to be right, or the model can't tell how much it is missing.
        let listed = memory_lines(&rendered);
        assert!(
            notice.contains(&(100 - listed).to_string()),
            "notice must name the number withheld ({} listed): {notice}",
            listed
        );
    }

    /// The entry cap bounds line count independently of the byte cap, for a store full of terse
    /// descriptions that would otherwise run to hundreds of lines inside the byte budget.
    #[test]
    fn test_memory_section_caps_entry_count() {
        let memories: Vec<Memory> = (0..MEMORY_INDEX_MAX_ENTRIES + 50)
            .map(|index| sample_memory(&format!("m{index:04}"), 5, "x", index as u64))
            .collect();
        let rendered = world_state_for_memories(&memories);
        let listed = memory_lines(&rendered);
        assert!(listed <= MEMORY_INDEX_MAX_ENTRIES, "listed {listed}");
        assert!(rendered.contains("more memories not shown"), "{rendered}");
    }

    /// Snapshot equality must not depend on the wall clock. The index renders ages as "14 days
    /// ago", but the snapshot stores `mtime`, so crossing midnight cannot masquerade as a world
    /// change and force a full re-render every day.
    #[test]
    fn test_memory_snapshot_equality_ignores_elapsed_time() {
        let memory = sample_memory("stable", 5, "unchanged", 3);
        let catalogue = catalogue_with(MEMORY_INDEX_TOOL);
        let before = WorldSnapshot::new(&catalogue, &[], std::slice::from_ref(&memory), &[]);
        let after = WorldSnapshot::new(&catalogue, &[], std::slice::from_ref(&memory), &[]);
        assert_eq!(before, after);
        assert_eq!(
            render_world_state(&after, Some(&before)),
            "",
            "an unchanged world must render nothing at all"
        );
    }

    /// A newly saved memory reaches the model through the diff, without re-listing the whole index.
    #[test]
    fn test_memory_diff_reports_saves_and_deletions() {
        let catalogue = catalogue_with(MEMORY_INDEX_TOOL);
        let before = WorldSnapshot::new(
            &catalogue,
            &[],
            &[sample_memory("kept", 5, "still true", 1)],
            &[],
        );
        let after = WorldSnapshot::new(
            &catalogue,
            &[],
            &[
                sample_memory("kept", 5, "still true", 1),
                sample_memory("fresh", 2, "just learned", 0),
            ],
            &[],
        );

        let diff = render_world_state(&after, Some(&before));
        assert!(diff.contains("fresh (p2: just learned)"), "{diff}");
        assert!(
            !diff.contains("kept"),
            "unchanged entries must stay quiet: {diff}"
        );

        let removed = render_world_state(&before, Some(&after));
        assert!(removed.contains("Memories deleted: `fresh`"), "{removed}");
    }

    /// The feature's load-bearing claim is that memory survives compaction. Two pieces make that
    /// true: `Agent::compact_session` drops `last_rendered_world`, and a `None` previous renders
    /// the world in full. This pins the second - an index already shown is restated verbatim, not
    /// diffed into silence - so a change to the diff path can't quietly make a post-compaction
    /// turn forget what the agent knows.
    #[test]
    fn test_full_render_restates_the_memory_index_after_compaction() {
        let memories = [
            sample_memory("standing-rule", 1, "Always reply in kind", 0),
            sample_memory("a-fact", 5, "The NAS is at nas.lan", 14),
        ];
        let snapshot = WorldSnapshot::new(&catalogue_with(MEMORY_INDEX_TOOL), &[], &memories, &[]);

        // Mid-session, an unchanged world says nothing.
        assert_eq!(render_world_state(&snapshot, Some(&snapshot)), "");

        // Post-compaction the previous render is forgotten, and the same snapshot renders whole.
        let restated = render_world_state(&snapshot, None);
        assert!(restated.contains("[Memory]"), "{restated}");
        assert!(restated.contains("standing-rule"), "{restated}");
        assert!(restated.contains("a-fact"), "{restated}");
    }

    /// An index whose opening tool is gone is a menu with no kitchen: the model would be told to
    /// call `skill` / `memory_read` and get an unknown-tool error. `[tools] disabled_tools` can
    /// reach that state without going through the `enabled` switches, so the filter lives in
    /// `WorldSnapshot::new` rather than beside them.
    #[test]
    fn test_index_is_dropped_when_its_opening_tool_is_unregistered() {
        let skills = [sample_skill("setup-server")];
        let memories = [sample_memory("a-note", 5, "a durable fact", 0)];

        // Only `read_file` registered: neither index has a way to be opened.
        let catalogue = catalogue_with("read_file");
        let rendered = render_world_state(
            &WorldSnapshot::new(&catalogue, &skills, &memories, &[]),
            None,
        );
        assert!(!rendered.contains("[Skills]"), "{rendered}");
        assert!(!rendered.contains("setup-server"), "{rendered}");
        assert!(!rendered.contains("[Memory]"), "{rendered}");
        assert!(!rendered.contains("a-note"), "{rendered}");

        // Each appears exactly when its own tool does, and not because of the other's.
        let rendered = render_world_state(
            &WorldSnapshot::new(&catalogue_with(SKILL_INDEX_TOOL), &skills, &memories, &[]),
            None,
        );
        assert!(rendered.contains("setup-server"), "{rendered}");
        assert!(!rendered.contains("[Memory]"), "{rendered}");

        let rendered = render_world_state(
            &WorldSnapshot::new(&catalogue_with(MEMORY_INDEX_TOOL), &skills, &memories, &[]),
            None,
        );
        assert!(rendered.contains("a-note"), "{rendered}");
        assert!(!rendered.contains("[Skills]"), "{rendered}");
    }

    /// A deferred tool still counts. Its schema is withheld until `load_tool` fetches it, but the
    /// model can reach it, so the index it opens is still actionable.
    #[test]
    fn test_deferred_opening_tool_still_renders_the_index() {
        let deferred = vec![(
            MEMORY_INDEX_TOOL.to_string(),
            "Load a saved memory".to_string(),
            Permission::Read,
            true,
        )];
        let memories = [sample_memory("a-note", 5, "a durable fact", 0)];
        let rendered =
            render_world_state(&WorldSnapshot::new(&deferred, &[], &memories, &[]), None);
        assert!(rendered.contains("[Memory]"), "{rendered}");
        assert!(rendered.contains("a-note"), "{rendered}");
    }

    /// Losing the opening tool mid-session has to reach the model as a deletion, not silence: the
    /// entries it was told about earlier are no longer usable.
    #[test]
    fn test_losing_the_opening_tool_reports_the_index_as_gone() {
        let memories = [sample_memory("a-note", 5, "a durable fact", 0)];
        let before = WorldSnapshot::new(&catalogue_with(MEMORY_INDEX_TOOL), &[], &memories, &[]);
        let after = WorldSnapshot::new(&catalogue_with("read_file"), &[], &memories, &[]);

        let diff = render_world_state(&after, Some(&before));
        assert!(diff.contains("Memories deleted: `a-note`"), "{diff}");
    }

    /// The `[Memory]` section alone. Counting index lines across the whole render would also
    /// catch `[Available tools]` entries, which share the `- **name**` shape.
    fn memory_section_of(rendered: &str) -> &str {
        let start = rendered
            .find("[Memory]")
            .unwrap_or_else(|| panic!("no [Memory] section in:\n{rendered}"));
        &rendered[start..]
    }

    fn memory_lines(rendered: &str) -> usize {
        memory_section_of(rendered)
            .lines()
            .filter(|line| line.starts_with("- **"))
            .count()
    }

    fn world_state_for_memories(memories: &[Memory]) -> String {
        render_world_state(
            &WorldSnapshot::new(&catalogue_with(MEMORY_INDEX_TOOL), &[], memories, &[]),
            None,
        )
    }

    fn sample_skill(name: &str) -> Skill {
        Skill {
            name: name.to_string(),
            source_dir: std::path::PathBuf::from("/tmp").join(name),
            description: format!("{} description", name),
            version: None,
            author: None,
            source_url: None,
            body_path: std::path::PathBuf::from("/tmp").join(name).join("SKILL.md"),
        }
    }

    fn sample_todo(text: &str, status: todo::TodoStatus) -> todo::TodoItem {
        todo::TodoItem {
            text: text.to_string(),
            status,
        }
    }

    fn sample_scratchpad_entry(name: &str, size: usize) -> ToolOutputSummary {
        ToolOutputSummary {
            name: name.to_string(),
            size,
            created_at: "2026-04-17T00:00:00Z".to_string(),
        }
    }

    fn sample_catalogue() -> Vec<ToolCatalogueEntry> {
        vec![
            (
                "read_file".to_string(),
                "Read file contents".to_string(),
                Permission::Read,
                false,
            ),
            (
                "write_file".to_string(),
                "Write text content to a file".to_string(),
                Permission::Write,
                false,
            ),
            (
                "execute_command".to_string(),
                "Run a shell command".to_string(),
                Permission::Read,
                false,
            ),
            (
                "scratchpad_read".to_string(),
                "Read a scratchpad entry".to_string(),
                Permission::Read,
                true,
            ),
            // `WorldSnapshot::new` drops each store's index unless the tool that opens it is
            // registered, so a catalogue meant to render those sections has to carry both.
            (
                SKILL_INDEX_TOOL.to_string(),
                "Load a skill's instructions".to_string(),
                Permission::Read,
                false,
            ),
            (
                MEMORY_INDEX_TOOL.to_string(),
                "Load a saved memory".to_string(),
                Permission::Read,
                false,
            ),
        ]
    }

    /// The smallest catalogue that lets `name`'s index render.
    fn catalogue_with(name: &str) -> Vec<ToolCatalogueEntry> {
        vec![(
            name.to_string(),
            "opens the index".to_string(),
            Permission::Read,
            false,
        )]
    }

    /// Full world-state render, as a first turn or a post-compaction turn sees it.
    fn world_state_for(
        catalogue: &[ToolCatalogueEntry],
        skills: &[Skill],
        mcp_server_instructions: &[(String, String)],
    ) -> String {
        render_world_state(
            &WorldSnapshot::new(catalogue, skills, &[], mcp_server_instructions),
            None,
        )
    }

    #[test]
    fn test_system_prompt_describes_permission_model() {
        let prompt = build_system_prompt(false, None);
        assert!(prompt.contains("## Permission Model"));
        assert!(prompt.contains("`none`"));
        assert!(prompt.contains("`read`"));
        assert!(prompt.contains("`ask`"));
        assert!(prompt.contains("`write`"));
        assert!(prompt.contains("`[Permission context]`"));
        assert!(prompt.contains("Shift+Tab"));
    }

    #[test]
    fn test_system_prompt_sandbox_note_read_mode() {
        let prompt = build_system_prompt(true, None);
        assert!(prompt.contains("filesystem mounted read-only"));
    }

    #[test]
    fn test_system_prompt_no_sandbox_note_without_flag() {
        let prompt = build_system_prompt(false, None);
        assert!(!prompt.contains("filesystem mounted read-only"));
        assert!(prompt.contains("`execute_command` is blocked"));
    }

    #[test]
    fn test_world_state_lists_active_tools_with_required_level() {
        let catalogue = sample_catalogue();
        let prompt = world_state_for(&catalogue, &[], &[]);
        assert!(prompt.contains("[Available tools]"));
        assert!(prompt.contains("**read_file** (requires `read`)"));
        assert!(prompt.contains("**write_file** (requires `write`)"));
        assert!(prompt.contains("**execute_command** (requires `read`)"));
    }

    #[test]
    fn test_world_state_omits_active_tool_descriptions() {
        // Active tools' descriptions already live in the API tools array. The system prompt
        // catalogue is now name + permission only, so the description string must not appear in the
        // `## Available Tools` section.
        let catalogue = sample_catalogue();
        let prompt = world_state_for(&catalogue, &[], &[]);
        let active_header = prompt.find("[Available tools]").unwrap();
        let next_section = prompt[active_header..]
            .find("\n[")
            .map(|idx| active_header + idx)
            .unwrap_or(prompt.len());
        let active_section = &prompt[active_header..next_section];
        assert!(!active_section.contains("Read file contents"));
        assert!(!active_section.contains("Write text content to a file"));
    }

    #[test]
    fn test_world_state_separates_deferred_tools() {
        let catalogue = sample_catalogue();
        let prompt = world_state_for(&catalogue, &[], &[]);
        assert!(prompt.contains("[Tool discovery]"));
        assert!(prompt.contains("Scratchpad operations"));
        assert!(prompt.contains("**scratchpad_read** (requires `read`)"));
        // The deferred tool must NOT appear in the active "Available Tools" section.
        let active_header = prompt.find("[Available tools]").unwrap();
        let deferred_header = prompt.find("[Tool discovery]").unwrap();
        let active_section = &prompt[active_header..deferred_header];
        assert!(!active_section.contains("scratchpad_read"));
    }

    #[test]
    fn test_world_state_truncates_deferred_tool_descriptions() {
        // A 2 KB MCP description must collapse to a one-liner; the full description still flows
        // through the tool schema once `load_tool` exposes it, so the only loss is the prose repeat
        // in the system prompt.
        let big_desc = "x".repeat(2048);
        let catalogue: Vec<ToolCatalogueEntry> = vec![(
            "mcp__notion__search".to_string(),
            big_desc,
            Permission::Read,
            true,
        )];
        let prompt = world_state_for(&catalogue, &[], &[]);
        let deferred_header = prompt.find("[Tool discovery]").unwrap();
        let section_end = prompt[deferred_header..]
            .find("\n[")
            .map(|idx| deferred_header + idx)
            .unwrap_or(prompt.len());
        let section = &prompt[deferred_header..section_end];
        let entry_line = section
            .lines()
            .find(|line| line.starts_with("- **mcp__notion__search**"))
            .expect("mcp__notion__search entry present");
        // Summary char cap + one-line prose scaffolding; well under the 2048 char full description
        // that used to ship here.
        let line_len = entry_line.chars().count();
        assert!(
            line_len <= TOOL_SUMMARY_MAX_CHARS + 60,
            "deferred entry line too long: {} chars",
            line_len
        );
        assert!(entry_line.ends_with('…'));
    }

    #[test]
    fn test_world_state_load_tool_itself_is_active_not_deferred() {
        // load_tool is the bootstrap meta-tool. Listing it under `## Tool Discovery` would create
        // a chicken-and-egg problem, so it must always be in the active `## Available Tools`
        // section, never in the deferred catalogue.
        let catalogue: Vec<ToolCatalogueEntry> = vec![
            (
                "load_tool".to_string(),
                "Load a deferred tool's schema.".to_string(),
                Permission::Read,
                false,
            ),
            (
                "scratchpad_read".to_string(),
                "Read a scratchpad entry".to_string(),
                Permission::Read,
                true,
            ),
        ];
        let prompt = world_state_for(&catalogue, &[], &[]);
        let active_header = prompt.find("[Available tools]").unwrap();
        let discovery_header = prompt.find("[Tool discovery]").unwrap();
        let active_section = &prompt[active_header..discovery_header];
        let discovery_section = &prompt[discovery_header..];
        assert!(active_section.contains("**load_tool**"));
        assert!(!discovery_section.contains("**load_tool**"));
    }

    #[test]
    fn test_world_state_groups_mcp_servers() {
        let catalogue: Vec<ToolCatalogueEntry> = vec![
            (
                "mcp__notion__search".to_string(),
                "Search Notion".to_string(),
                Permission::Read,
                true,
            ),
            (
                "mcp__notion__fetch".to_string(),
                "Fetch a Notion page".to_string(),
                Permission::Read,
                true,
            ),
            (
                "mcp__github__create_issue".to_string(),
                "Open a GitHub issue".to_string(),
                Permission::Write,
                true,
            ),
            (
                "scratchpad_read".to_string(),
                "Read scratchpad entry".to_string(),
                Permission::Read,
                true,
            ),
            (
                "list_mcp_resources".to_string(),
                "List MCP resources".to_string(),
                Permission::Read,
                true,
            ),
        ];
        let prompt = world_state_for(&catalogue, &[], &[]);
        assert!(prompt.contains("Scratchpad operations"));
        assert!(prompt.contains("MCP resource tools"));
        assert!(prompt.contains("MCP server: github"));
        assert!(prompt.contains("MCP server: notion"));
        // notion subsection lists both notion tools.
        let notion_header = prompt.find("MCP server: notion").unwrap();
        let notion_section_end = prompt[notion_header..]
            .find("\n### ")
            .or_else(|| prompt[notion_header..].find("\n["))
            .map(|idx| notion_header + idx)
            .unwrap_or(prompt.len());
        let notion_section = &prompt[notion_header..notion_section_end];
        assert!(notion_section.contains("mcp__notion__search"));
        assert!(notion_section.contains("mcp__notion__fetch"));
        assert!(!notion_section.contains("mcp__github__"));
    }

    #[test]
    fn test_short_description_first_sentence() {
        let s = "Read a scratchpad entry. Extra info follows that we drop.";
        assert_eq!(short_description(s), "Read a scratchpad entry.");
    }

    #[test]
    fn test_short_description_passes_through_short_text() {
        let s = "Read a scratchpad entry.";
        assert_eq!(short_description(s), "Read a scratchpad entry.");
    }

    #[test]
    fn test_short_description_char_cap() {
        let long = "a".repeat(300);
        let out = short_description(&long);
        assert!(out.chars().count() <= TOOL_SUMMARY_MAX_CHARS + 1);
        assert!(out.ends_with('…'));
    }

    #[test]
    fn test_short_description_collapses_whitespace() {
        let s = "Line one.\n\n  Line   two.  ";
        assert_eq!(short_description(s), "Line one.");
    }

    #[test]
    fn test_short_description_no_sentence_terminator_short() {
        // A short description without an ASCII/CJK sentence terminator is the complete description,
        // no ellipsis suffix, since the model is seeing the whole text already.
        let s = "no terminator at all here just words";
        assert_eq!(short_description(s), "no terminator at all here just words");
    }

    #[test]
    fn test_short_description_no_sentence_terminator_long_gets_ellipsis() {
        let s = "word ".repeat(200);
        let out = short_description(&s);
        assert!(out.ends_with('…'));
        assert!(out.chars().count() <= TOOL_SUMMARY_MAX_CHARS + 1);
    }

    #[test]
    fn test_short_description_empty_input() {
        assert_eq!(short_description(""), "");
        assert_eq!(short_description("   \n  "), "");
    }

    #[test]
    fn test_short_description_utf8_safe() {
        let s = "读取一个文件。附加内容在这里。";
        let out = short_description(s);
        assert_eq!(out, "读取一个文件。附加内容在这里。");
    }

    #[test]
    fn test_world_state_render_is_deterministic() {
        let catalogue = sample_catalogue();
        let a = world_state_for(&catalogue, &[], &[]);
        let b = world_state_for(&catalogue, &[], &[]);
        assert_eq!(a, b);
    }

    /// The invariant this whole design exists for. The system prompt heads the cached prefix, so a
    /// byte moving here re-caches the tools array and every message behind it. Registering a tool,
    /// installing a skill, and connecting an MCP server are the three things that used to move it.
    #[test]
    fn test_system_prompt_ignores_everything_that_changes_mid_session() {
        let baseline = build_system_prompt(true, Some("Never use pip."));

        // The parameters are the whole story: there is nothing dynamic left to pass. Feeding a
        // catalogue, a skill, and a server's instructions through the world-state path proves they
        // render somewhere, and that somewhere is not here.
        let catalogue = sample_catalogue();
        let skills = vec![Skill {
            name: "deploy-app".to_string(),
            source_dir: std::path::PathBuf::from("/tmp"),
            description: "deploy".to_string(),
            version: None,
            author: None,
            source_url: None,
            body_path: std::path::PathBuf::from("/tmp/SKILL.md"),
        }];
        let instructions = vec![("fs".to_string(), "Read before write.".to_string())];
        let world = world_state_for(&catalogue, &skills, &instructions);

        assert_eq!(
            baseline,
            build_system_prompt(true, Some("Never use pip.")),
            "the system prompt must be byte-identical across a session",
        );
        assert!(!baseline.contains("read_file"), "tools must not be in it");
        assert!(!baseline.contains("deploy-app"), "skills must not be in it");
        assert!(
            !baseline.contains("Read before write."),
            "MCP instructions must not be in it",
        );
        // ...and all three are accounted for in the block that is appended instead.
        assert!(world.contains("read_file"));
        assert!(world.contains("deploy-app"));
        assert!(world.contains("Read before write."));
    }

    /// Walks a whole session the way `run_turn` does, asserting what each turn actually renders.
    /// The shape across turns is the deliverable: full render once, silence while nothing moves,
    /// a delta naming only the change, then silence again.
    #[test]
    fn test_world_state_across_a_session() {
        let catalogue = sample_catalogue();
        let mut last: Option<WorldSnapshot> = None;

        // Turn 1: nothing has been said yet, so the model gets the whole picture.
        let current = WorldSnapshot::new(&catalogue, &[], &[], &[]);
        let turn1 = render_world_state(&current, last.as_ref());
        last = Some(current);
        assert!(turn1.contains("[Available tools]"), "got: {}", turn1);
        assert!(turn1.contains("**read_file**"));

        // Turn 2: nothing changed. This is the steady state and must cost nothing.
        let current = WorldSnapshot::new(&catalogue, &[], &[], &[]);
        let turn2 = render_world_state(&current, last.as_ref());
        last = Some(current);
        assert_eq!(turn2, "", "an unchanged turn must render nothing");

        // Turn 3: an MCP server connects, bringing a tool and instructions.
        let mut grown = catalogue.clone();
        grown.push((
            "mcp__fs__read".to_string(),
            "Read a file over MCP".to_string(),
            Permission::Read,
            true,
        ));
        let instructions = vec![("fs".to_string(), "Read before write.".to_string())];
        let current = WorldSnapshot::new(&grown, &[], &[], &instructions);
        let turn3 = render_world_state(&current, last.as_ref());
        last = Some(current);
        assert!(turn3.contains("`mcp__fs__read`"), "got: {}", turn3);
        assert!(turn3.contains("Read before write."), "got: {}", turn3);
        // A delta, not a re-listing: tools that did not move must not be repeated.
        assert!(
            !turn3.contains("**read_file**"),
            "unchanged tools must not be re-sent every time something else moves; got: {}",
            turn3,
        );

        // Turn 4: quiet again.
        let current = WorldSnapshot::new(&grown, &[], &[], &instructions);
        assert_eq!(render_world_state(&current, last.as_ref()), "");

        // Turn 5: compaction forgets what the model was told (`compact_session` clears the stored
        // snapshot), so the picture is restated whole rather than diffed against something the
        // model can no longer see.
        let after_compact = render_world_state(&current, None);
        assert!(after_compact.contains("[Available tools]"));
        assert!(after_compact.contains("**read_file**"));
        assert!(after_compact.contains("Read before write."));
    }

    /// The steady-state path. An unchanged session must add nothing per turn, or the delta
    /// machinery would cost more than the cache it protects.
    #[test]
    fn test_world_state_is_silent_when_nothing_changed() {
        // Two snapshots built independently from the same inputs, not one compared with itself:
        // the latter would pass on reference equality alone and prove nothing about the fields.
        let catalogue = sample_catalogue();
        let skills = [Skill {
            name: "deploy-app".to_string(),
            source_dir: std::path::PathBuf::from("/tmp"),
            description: "Ship it".to_string(),
            version: None,
            author: None,
            source_url: None,
            body_path: std::path::PathBuf::from("/tmp/SKILL.md"),
        }];
        let instructions = [("fs".to_string(), "Read before write.".to_string())];
        let before = WorldSnapshot::new(&catalogue, &skills, &[], &instructions);
        let after = WorldSnapshot::new(&catalogue, &skills, &[], &instructions);
        assert_eq!(render_world_state(&after, Some(&before)), "");
    }

    #[test]
    fn test_world_state_diff_reports_added_and_removed_tools() {
        let before = WorldSnapshot::new(
            &[(
                "mcp__fs__read".to_string(),
                "Read".to_string(),
                Permission::Read,
                true,
            )],
            &[],
            &[],
            &[],
        );
        let after = WorldSnapshot::new(
            &[(
                "mcp__fs__write".to_string(),
                "Write".to_string(),
                Permission::Write,
                false,
            )],
            &[],
            &[],
            &[],
        );
        let diff = render_world_state(&after, Some(&before));

        assert!(diff.contains("supersedes"), "got: {}", diff);
        assert!(
            diff.contains("Now callable: `mcp__fs__write`"),
            "got: {}",
            diff
        );
        assert!(
            diff.contains("No longer available, do not call: `mcp__fs__read`"),
            "a removed tool must be named, not silently dropped; got: {}",
            diff,
        );
        // A delta is not a re-listing: nothing that stayed put should reappear.
        assert!(!diff.contains("[Available tools]"), "got: {}", diff);
    }

    /// `None` is how compaction asks for a re-statement: the turns carrying the earlier rendering
    /// are behind the boundary and may have been summarized away, so the model needs the whole
    /// picture again rather than a delta against something it can no longer see.
    #[test]
    fn test_world_state_renders_in_full_when_previous_is_forgotten() {
        let catalogue = sample_catalogue();
        let snapshot = WorldSnapshot::new(&catalogue, &[], &[], &[]);
        let full = render_world_state(&snapshot, None);

        assert!(full.contains("[Available tools]"));
        assert!(full.contains("**read_file**"));
        assert!(
            !full.contains("supersedes"),
            "a full render is not a delta; got: {}",
            full,
        );
        assert_eq!(
            full,
            render_world_state(&snapshot, None),
            "and it must be stable, so two post-compaction turns agree",
        );
    }

    /// The general form of the guarantee, over every ordered pair of representative snapshots:
    /// identical snapshots render nothing, and differing snapshots always render something.
    ///
    /// Doubles as a drift guard. A field added to [`WorldSnapshot`] without a matching branch in
    /// `render_world_state_diff` makes some pair differ while rendering nothing, and this fails
    /// rather than the model silently working from stale facts.
    #[test]
    fn test_world_state_diff_never_advances_silently() {
        let tool = |name: &str, summary: &str, deferred: bool| {
            (
                name.to_string(),
                summary.to_string(),
                Permission::Read,
                deferred,
            )
        };
        let skill = |name: &str, description: &str| Skill {
            name: name.to_string(),
            source_dir: std::path::PathBuf::from("/tmp"),
            description: description.to_string(),
            version: None,
            author: None,
            source_url: None,
            body_path: std::path::PathBuf::from("/tmp/SKILL.md"),
        };

        let snapshots = vec![
            ("empty", WorldSnapshot::default()),
            (
                "one tool",
                WorldSnapshot::new(&[tool("a", "does a", false)], &[], &[], &[]),
            ),
            (
                "same tool, deferred",
                WorldSnapshot::new(&[tool("a", "does a", true)], &[], &[], &[]),
            ),
            (
                "same tool, reworded",
                WorldSnapshot::new(&[tool("a", "does a differently", false)], &[], &[], &[]),
            ),
            (
                "two tools",
                WorldSnapshot::new(
                    &[tool("a", "does a", false), tool("b", "does b", false)],
                    &[],
                    &[],
                    &[],
                ),
            ),
            (
                "one tool and a skill",
                WorldSnapshot::new(
                    &[tool("a", "does a", false)],
                    &[skill("s", "ships")],
                    &[],
                    &[],
                ),
            ),
            (
                "one tool and a reworded skill",
                WorldSnapshot::new(
                    &[tool("a", "does a", false)],
                    &[skill("s", "ships fast")],
                    &[],
                    &[],
                ),
            ),
            (
                "one tool and a server",
                WorldSnapshot::new(&[tool("a", "does a", false)], &[], &[], &[(
                    "fs".to_string(),
                    "guidance".to_string(),
                )]),
            ),
            (
                "one tool and a rewritten server",
                WorldSnapshot::new(&[tool("a", "does a", false)], &[], &[], &[(
                    "fs".to_string(),
                    "new guidance".to_string(),
                )]),
            ),
        ];

        for (from_label, from) in &snapshots {
            for (to_label, to) in &snapshots {
                let rendered = render_world_state(to, Some(from));
                if from == to {
                    assert!(
                        rendered.is_empty(),
                        "{} -> {} is a no-op but rendered: {:?}",
                        from_label,
                        to_label,
                        rendered,
                    );
                } else {
                    assert!(
                        !rendered.is_empty(),
                        "{} -> {} changed the model's picture but rendered nothing",
                        from_label,
                        to_label,
                    );
                }
            }
        }
    }

    /// Every snapshot difference must produce something for the model to read. A change that
    /// silently advances the snapshot is worse than a noisy one: the model then works from stale
    /// facts for the rest of the session with no way to notice.
    #[test]
    fn test_world_state_diff_reports_a_reworded_tool() {
        let entry = |summary: &str| {
            vec![(
                "mcp__fs__read".to_string(),
                summary.to_string(),
                Permission::Read,
                false,
            )]
        };
        let before = WorldSnapshot::new(&entry("Reads a file."), &[], &[], &[]);
        let after = WorldSnapshot::new(&entry("Reads a file, following symlinks."), &[], &[], &[]);
        assert_ne!(before, after, "the fixture must actually differ");

        let diff = render_world_state(&after, Some(&before));
        assert!(
            diff.contains("following symlinks"),
            "a changed description must reach the model; got: {:?}",
            diff,
        );
    }

    #[test]
    fn test_world_state_diff_reports_mcp_instruction_changes() {
        let before = WorldSnapshot::new(&[], &[], &[], &[
            ("fs".to_string(), "Old guidance.".to_string()),
            ("db".to_string(), "Read only.".to_string()),
        ]);
        let after = WorldSnapshot::new(&[], &[], &[], &[(
            "fs".to_string(),
            "New guidance.".to_string(),
        )]);
        let diff = render_world_state(&after, Some(&before));

        assert!(diff.contains("New guidance."), "got: {}", diff);
        assert!(
            diff.contains("replace any instructions it provided earlier"),
            "a changed server block must supersede, not stack; got: {}",
            diff,
        );
        assert!(
            diff.contains("no longer apply: `db`"),
            "a disconnected server must be retracted; got: {}",
            diff,
        );
    }

    #[test]
    fn test_system_prompt_always_has_environment() {
        let prompt = build_system_prompt(false, None);
        assert!(prompt.contains("## Environment"));
    }

    #[test]
    fn test_world_state_lists_skills() {
        let skills = vec![sample_skill("setup-server"), sample_skill("deploy-app")];
        let prompt = world_state_for(&catalogue_with(SKILL_INDEX_TOOL), &skills, &[]);
        assert!(prompt.contains("[Skills]"));
        assert!(prompt.contains("**setup-server**"));
        assert!(prompt.contains("setup-server description"));
        assert!(prompt.contains("**deploy-app**"));
    }

    #[test]
    fn test_world_state_omits_skills_section_when_empty() {
        let rendered = world_state_for(&sample_catalogue(), &[], &[]);
        assert!(
            !rendered.contains("[Skills]"),
            "an empty skill list must not emit a heading; got: {}",
            rendered,
        );
        assert!(
            rendered.contains("[Available tools]"),
            "and the rest of the render must still be there, or this proves nothing",
        );
    }

    #[test]
    fn test_system_prompt_includes_user_instructions() {
        let prompt = build_system_prompt(false, Some("Never use pip. Always prefer uv."));
        assert!(prompt.contains("## User Instructions"));
        assert!(prompt.contains("Never use pip. Always prefer uv."));
        assert!(prompt.contains("installation-specific rules"));
    }

    #[test]
    fn test_system_prompt_omits_user_instructions_when_none() {
        let prompt = build_system_prompt(false, None);
        assert!(!prompt.contains("## User Instructions"));
    }

    #[test]
    fn test_system_prompt_omits_user_instructions_when_whitespace() {
        let prompt = build_system_prompt(false, Some("   \n"));
        assert!(!prompt.contains("## User Instructions"));
    }

    #[test]
    fn test_permission_context_read_is_terse() {
        let context = build_permission_context(Permission::Read);
        assert!(context.contains("[Permission context]"));
        assert!(context.contains("Current permission level: read"));
        assert!(context.contains("Only read-only tools are executable."));
        // The per-turn block must NOT enumerate individual tools; that duplicates the static
        // system-prompt catalogue and balloons with MCP-tool count. Regression-guards the O(1) size
        // invariant.
        assert!(!context.contains("write_file"));
        assert!(!context.contains("requires `"));
    }

    #[test]
    fn test_permission_context_write_shows_all_accessible() {
        let context = build_permission_context(Permission::Write);
        assert!(context.contains("Current permission level: write"));
        assert!(context.contains("All tools are executable."));
    }

    #[test]
    fn test_permission_context_ask_mentions_approval() {
        let context = build_permission_context(Permission::Ask);
        assert!(context.contains("Current permission level: ask"));
        assert!(context.contains("user approval"));
    }

    #[test]
    fn test_permission_context_none_is_terse() {
        let context = build_permission_context(Permission::None);
        assert!(context.contains("Current permission level: none"));
        assert!(context.contains("No tools are executable."));
        assert!(!context.contains("read_file"));
    }

    #[test]
    fn test_permission_context_size_bounded_regardless_of_catalogue() {
        // Whatever the registered tool count, the block's token cost stays constant; this is the
        // whole point of the trim.
        for level in [
            Permission::None,
            Permission::Read,
            Permission::Ask,
            Permission::Write,
        ] {
            let ctx = build_permission_context(level);
            assert!(
                ctx.len() < 200,
                "permission context for {:?} grew past 200 bytes: {}",
                level,
                ctx.len()
            );
        }
    }

    #[test]
    fn test_environment_context() {
        let context = build_environment_context(Permission::Read, std::path::Path::new("."), &[]);
        assert!(context.contains("[Environment context]"));
        assert!(context.contains("Working directory:"));
        assert!(context.contains("Date:"));
    }

    #[test]
    fn test_environment_context_none_mode() {
        let context = build_environment_context(Permission::None, std::path::Path::new("."), &[]);
        assert!(context.is_empty());
    }

    /// Single-root output must be untouched: every REPL, HTTP, and single-folder ACP session goes
    /// through here, and an extra line would change the prompt for all of them.
    #[test]
    fn test_environment_context_omits_roots_when_empty() {
        let context =
            build_environment_context(Permission::Read, std::path::Path::new("/work"), &[]);
        assert!(context.contains("Working directory: /work"));
        assert!(!context.contains("Additional workspace roots"));
    }

    /// Naming the extra roots is the point of tracking them: without this the model cannot learn
    /// the other folders exist and reports files in them as missing.
    #[test]
    fn test_environment_context_names_additional_roots() {
        let roots = vec![
            std::path::PathBuf::from("/work/shared"),
            std::path::PathBuf::from("/work/docs"),
        ];
        let context =
            build_environment_context(Permission::Read, std::path::Path::new("/work/main"), &roots);
        assert!(context.contains("Working directory: /work/main"));
        assert!(context.contains("Additional workspace roots"));
        assert!(context.contains("/work/shared"));
        assert!(context.contains("/work/docs"));
    }

    #[test]
    fn test_turn_context_always_has_permission_context() {
        let context = build_turn_context(
            Permission::None,
            &TodoState::default(),
            std::path::Path::new("."),
            &[],
            "",
        );
        assert!(context.starts_with("<context>\n"));
        assert!(context.ends_with("</context>"));
        assert!(context.contains("[Permission context]"));
    }

    #[test]
    fn test_turn_context_has_environment_in_read_mode() {
        let context = build_turn_context(
            Permission::Read,
            &TodoState::default(),
            std::path::Path::new("."),
            &[],
            "",
        );
        assert!(context.contains("[Permission context]"));
        assert!(context.contains("[Environment context]"));
    }

    #[test]
    fn test_turn_context_includes_todos() {
        let todos = TodoState {
            items: vec![sample_todo("write tests", todo::TodoStatus::InProgress)],
            ..Default::default()
        };
        let context =
            build_turn_context(Permission::Read, &todos, std::path::Path::new("."), &[], "");
        assert!(context.contains("write tests"));
        assert!(context.contains("[Environment context]"));
        assert!(context.contains("[Permission context]"));
    }

    #[test]
    fn test_turn_context_none_mode_omits_environment() {
        let todos = TodoState {
            items: vec![sample_todo("do a thing", todo::TodoStatus::Pending)],
            ..Default::default()
        };
        let context =
            build_turn_context(Permission::None, &todos, std::path::Path::new("."), &[], "");
        assert!(context.contains("do a thing"));
        assert!(context.contains("[Permission context]"));
        assert!(!context.contains("[Environment context]"));
    }

    #[test]
    fn test_post_compact_context_empty_in_none_mode_no_state() {
        let result = build_post_compact_context(
            Permission::None,
            &TodoState::default(),
            &[],
            std::path::Path::new("."),
            &[],
        );
        assert!(result.is_empty());
    }

    #[test]
    fn test_post_compact_context_includes_env_todos_scratchpad() {
        let todos = TodoState {
            items: vec![sample_todo("keep working", todo::TodoStatus::Pending)],
            ..Default::default()
        };
        let entries = vec![sample_scratchpad_entry("notes", 1024)];
        let result = build_post_compact_context(
            Permission::Read,
            &todos,
            &entries,
            std::path::Path::new("."),
            &[],
        );
        assert!(result.contains("[Environment context]"));
        assert!(result.contains("keep working"));
        assert!(result.contains("[Scratchpad entries]"));
        assert!(result.contains("\"notes\""));
    }

    #[test]
    fn test_post_compact_context_scratchpad_only() {
        let entries = vec![sample_scratchpad_entry("log", 500)];
        let result = build_post_compact_context(
            Permission::None,
            &TodoState::default(),
            &entries,
            std::path::Path::new("."),
            &[],
        );
        assert!(result.contains("[Scratchpad entries]"));
        assert!(result.contains("\"log\""));
        assert!(!result.contains("[Environment context]"));
    }

    #[test]
    fn test_world_state_includes_mcp_server_instructions() {
        let instructions = vec![
            (
                "fs".to_string(),
                "Call `fs__read` before `fs__write`.".to_string(),
            ),
            (
                "db".to_string(),
                "All queries run in read-only mode.".to_string(),
            ),
        ];
        let prompt = world_state_for(&[], &[], &instructions);
        assert!(prompt.contains("[MCP server instructions]"));
        assert!(prompt.contains("\nfs\n"));
        assert!(prompt.contains("Call `fs__read` before `fs__write`."));
        assert!(prompt.contains("\ndb\n"));
        assert!(prompt.contains("All queries run in read-only mode."));
    }

    #[test]
    fn test_system_prompt_has_no_mcp_server_instructions() {
        let prompt = build_system_prompt(false, None);
        assert!(!prompt.contains("## MCP Server Instructions"));
    }
}
