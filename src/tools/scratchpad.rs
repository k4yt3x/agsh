//! Scratchpad: session-scoped, persisted key/value text store. Oversized tool outputs are
//! automatically redirected here and replaced inline with a preview + handle, keeping the
//! conversation context bounded. Provides write/read/edit/list/delete operations plus a regex
//! search mode.

use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::{
    Tool, ToolOutput,
    util::{MAX_SEARCH_MATCHES, resolve_session_id, search_lines},
};
use crate::{
    error::{MekaError, Result},
    permission::Permission,
    provider::{ContentBlock, Message, ToolDefinition, ToolResultContent},
    session::{SessionManager, ToolOutputSummary},
};

/// Tool result text blocks larger than this (in bytes) are persisted to the database and replaced
/// with a preview + handle in the conversation context.
pub const MAX_INLINE_RESULT_BYTES: usize = 30_000;

/// Number of bytes included in the inline preview.
const PREVIEW_BYTES: usize = 2_000;

/// Default byte limit when reading back a persisted output.
const DEFAULT_READ_LIMIT: usize = 30_000;

pub(crate) fn format_size(bytes: usize) -> String {
    if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1_024 {
        format!("{:.1} KB", bytes as f64 / 1_024.0)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Build a map from tool_use_id to (tool_name, input) for the ToolUse blocks in an assistant
/// message.
fn build_tool_use_map(assistant_message: &Message) -> HashMap<String, (String, serde_json::Value)> {
    let mut map = HashMap::new();
    for block in &assistant_message.content {
        if let ContentBlock::ToolUse { id, name, input } = block {
            map.insert(id.clone(), (name.clone(), input.clone()));
        }
    }
    map
}

fn build_scratchpad_reference(name: &str, size: usize) -> String {
    format!(
        "Output saved to scratchpad \"{}\" ({} bytes). \
         Use scratchpad_read to access it.",
        name, size,
    )
}

fn build_large_output_preview(name: &str, text: &str) -> String {
    let size = text.len();
    let preview_end = text.floor_char_boundary(PREVIEW_BYTES.min(size));
    let preview = &text[..preview_end];
    let has_more = preview_end < size;

    let mut replacement = format!(
        "<large-output name=\"{}\" size=\"{}\">\n\
         Output too large ({}). Read with `scratchpad_read`. Use \
         `limit: {}` to load the full content in one call, or page \
         with `offset`/`limit` if a partial read is enough.\n\n\
         Preview (first {} bytes):\n\
         {}",
        name,
        size,
        format_size(size),
        size,
        preview_end,
        preview,
    );
    if has_more {
        replacement.push_str("\n...");
    }
    replacement.push_str("\n</large-output>");
    replacement
}

/// Save tool results to the scratchpad when the agent explicitly requested it via the `scratchpad`
/// parameter on a tool call. Replaces the inline result with a brief reference.
pub async fn save_explicit_scratchpad_results(
    session_manager: &SessionManager,
    session_id: Uuid,
    assistant_message: &Message,
    results: &mut [ContentBlock],
) -> Result<()> {
    let tool_use_map = build_tool_use_map(assistant_message);

    for block in results.iter_mut() {
        if let ContentBlock::ToolResult {
            tool_use_id,
            content,
            ..
        } = block
        {
            let scratchpad_name = tool_use_map
                .get(tool_use_id.as_str())
                .and_then(|(_, input)| input.get("scratchpad"))
                .and_then(|v| v.as_str());

            let Some(name) = scratchpad_name else {
                continue;
            };

            let text = ContentBlock::tool_result_text_content(content);
            if text.is_empty() {
                continue;
            }

            let size = text.len();
            session_manager
                .save_tool_output(session_id, name, &text)
                .await?;

            *content = vec![ToolResultContent::Text {
                text: build_scratchpad_reference(name, size),
            }];
        }
    }
    Ok(())
}

/// Check each text block in tool results. If oversized, persist to DB and replace with a preview +
/// handle. Names are derived from the tool call's `scratchpad_hint` (MCP adapters) or the tool name
/// otherwise, with a numeric suffix on collision. `hints` is typically the per-turn map owned by
/// the agent; empty is fine.
pub async fn persist_oversized_results(
    session_manager: &SessionManager,
    session_id: Uuid,
    assistant_message: &Message,
    results: &mut [ContentBlock],
    hints: &std::collections::HashMap<String, String>,
) -> Result<()> {
    let tool_use_map = build_tool_use_map(assistant_message);
    let mut counter: usize = 0;

    for block in results.iter_mut() {
        if let ContentBlock::ToolResult {
            tool_use_id,
            content,
            ..
        } = block
        {
            let base_name = hints.get(tool_use_id.as_str()).cloned().unwrap_or_else(|| {
                tool_use_map
                    .get(tool_use_id.as_str())
                    .map(|(name, _)| name.clone())
                    .unwrap_or_else(|| "unknown".to_string())
            });

            for item in content.iter_mut() {
                if let ToolResultContent::Text { text } = item {
                    if text.len() <= MAX_INLINE_RESULT_BYTES {
                        continue;
                    }

                    counter += 1;
                    // Unique for the life of the session, not just within this call.
                    //
                    // `counter` restarts at zero on every invocation -- this runs once per
                    // assistant message -- so the name was `<tool>_1` for the
                    // first oversized result of *every*
                    // turn. `save_tool_output` is `INSERT OR REPLACE` keyed on `(session, name)`,
                    // so turn 3's 150 KB command output silently overwrote turn
                    // 1's, and the model reading back the handle it was given
                    // in turn 1 got turn 3's bytes with no error and no size
                    // complaint. MCP tools were worse: their hint is a stable
                    // `mcp_<server>_<tool>`, so every large fetch from one server collapsed onto
                    // one name. The `tool_use_id` is provider-generated and
                    // unique per call, which is exactly the scope the name
                    // needs.
                    let name = format!("{}_{}_{}", base_name, short_call_id(tool_use_id), counter);

                    session_manager
                        .save_tool_output(session_id, &name, text)
                        .await?;

                    *text = build_large_output_preview(&name, text);
                }
            }
        }
    }
    Ok(())
}

/// A short, name-safe slice of a provider tool-call id, for disambiguating scratchpad entries.
///
/// Ids run to ~30 characters (`toolu_01A9…`, `call_abc…`) and the whole thing in every entry name
/// would make `scratchpad_list` unreadable and the handles tedious for the model to quote back. The
/// tail is used rather than the head because providers put their fixed prefix at the front, so the
/// last few characters carry the entropy. Restricted to `[A-Za-z0-9]` because the name is also an
/// entry key.
fn short_call_id(tool_use_id: &str) -> String {
    let cleaned: String = tool_use_id
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .collect();
    let tail: String = cleaned
        .chars()
        .rev()
        .take(6)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if tail.is_empty() {
        "call".to_string()
    } else {
        tail
    }
}

pub(super) struct ScratchpadWriteTool {
    pub session_manager: SessionManager,
    pub session_id: Arc<RwLock<Option<Uuid>>>,
    /// Names the parent has lent this sub-agent read-only. Writing to any of these is rejected so
    /// the child can't silently shadow the parent's copy. Empty on the primary agent's registry.
    pub inherited_names: Vec<String>,
}

fn inherited_write_error(name: &str) -> Result<ToolOutput> {
    Ok(ToolOutput::text(
        format!(
            "Scratchpad entry \"{}\" is inherited read-only from the parent. \
             Pick a different name (e.g. \"{}_local\") for your own scratchpad state.",
            name, name,
        ),
        true,
    ))
}

fn is_inherited(inherited_names: &[String], name: &str) -> bool {
    inherited_names.iter().any(|candidate| candidate == name)
}

#[async_trait]
impl Tool for ScratchpadWriteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "scratchpad_write".to_string(),
            description: "Store content in the scratchpad under the given name \u{2014} a \
                session-scoped working memory that persists across turns without consuming \
                conversation context. If the name already exists, the content is overwritten. \
                Use this to save intermediate results, extracted text, accumulated data, or \
                research notes. You can also save tool output directly by adding a 'scratchpad' \
                parameter to any tool call. When you are a sub-agent, names inherited \
                read-only from the parent are rejected here. Use a different name (e.g. \
                'name_local') for your own state."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name for the scratchpad entry"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to store"
                    }
                },
                "required": ["name", "content"]
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
        let name = input["name"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_write".to_string(),
                message: "missing 'name' parameter".to_string(),
            })?;
        let content = input["content"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_write".to_string(),
                message: "missing 'content' parameter".to_string(),
            })?;

        if is_inherited(&self.inherited_names, name) {
            return inherited_write_error(name);
        }

        let session_id = resolve_session_id(&self.session_id, "scratchpad_write").await?;

        self.session_manager
            .save_tool_output(session_id, name, content)
            .await?;

        Ok(ToolOutput::text(
            format!(
                "Stored {} bytes as scratchpad entry \"{}\"",
                content.len(),
                name,
            ),
            false,
        ))
    }
}

pub(super) struct ScratchpadReadTool {
    pub session_manager: SessionManager,
    pub session_id: Arc<RwLock<Option<Uuid>>>,
    /// Sub-agent fallback: when the read misses the active (child) session, retry against this
    /// parent session for names listed in [`Self::inherited_names`]. `None` on the primary agent's
    /// registry; no fallback path is taken.
    pub parent_session_id: Option<Uuid>,
    /// Allowlist of parent-scoped scratchpad names the sub-agent is permitted to read. Empty on
    /// the primary agent.
    pub inherited_names: Vec<String>,
}

#[async_trait]
impl Tool for ScratchpadReadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "scratchpad_read".to_string(),
            description: format!(
                "Read or search a scratchpad entry by name. Default returns {} \
                 bytes from offset; pass a larger `limit` (no hard cap) to load the full \
                 entry in one call, or page with `offset`/`limit` for partial reads. Provide \
                 `regex` to return matching lines (max {}) instead of a byte range. Also \
                 used to access content referenced by <large-output> tags. Pass the `size` \
                 value from the tag as `limit` when you intend to read everything. When this \
                 is a sub-agent and the name is not found locally, looks up names from the \
                 parent's inherited allowlist (see the system-prompt section if any).",
                DEFAULT_READ_LIMIT, MAX_SEARCH_MATCHES,
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The scratchpad entry name"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Character offset to start reading from. Default: 0."
                    },
                    "limit": {
                        "type": "integer",
                        "description": format!("Maximum bytes to return. Default: {}.", DEFAULT_READ_LIMIT)
                    },
                    "regex": {
                        "type": "string",
                        "description": format!(
                            "If provided, search the entry with this regex pattern \
                             and return matching lines (max {} matches) instead of a character range.",
                            MAX_SEARCH_MATCHES,
                        )
                    }
                },
                "required": ["name"]
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
        let name = input["name"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_read".to_string(),
                message: "missing 'name' parameter".to_string(),
            })?;

        let session_id = resolve_session_id(&self.session_id, "scratchpad_read").await?;

        let mut content = self
            .session_manager
            .load_tool_output(session_id, name)
            .await?;

        // Sub-agent inheritance: fall back to the parent's scratchpad if the child miss matches an
        // allowlisted name. Read-only: writes still target the child session, so the parent's
        // audit trail is untouched.
        if content.is_none()
            && let Some(parent_sid) = self.parent_session_id
            && self.inherited_names.iter().any(|n| n == name)
        {
            content = self
                .session_manager
                .load_tool_output(parent_sid, name)
                .await?;
        }

        let content = content.ok_or_else(|| MekaError::ToolExecution {
            tool_name: "scratchpad_read".to_string(),
            message: format!("scratchpad entry \"{}\" not found", name),
        })?;

        if let Some(pattern) = input.get("regex").and_then(|v| v.as_str()) {
            return search_lines(&content, pattern, "scratchpad_read");
        }

        read_mode(&content, &input)
    }
}

fn read_mode(content: &str, input: &serde_json::Value) -> Result<ToolOutput> {
    let offset = usize::try_from(input["offset"].as_u64().unwrap_or(0)).unwrap_or(usize::MAX);
    let limit = usize::try_from(input["limit"].as_u64().unwrap_or(DEFAULT_READ_LIMIT as u64))
        .unwrap_or(usize::MAX);
    let total = content.len();

    if offset >= total {
        return Ok(ToolOutput::text(
            format!("Offset {} exceeds content length ({} bytes)", offset, total),
            true,
        ));
    }

    let start = content.floor_char_boundary(offset);
    let end = content.floor_char_boundary(start.saturating_add(limit).min(total));
    let slice = &content[start..end];

    Ok(ToolOutput::text(
        format!(
            "{}\n\n(showing bytes {}..{} of {})",
            slice, start, end, total
        ),
        false,
    ))
}

pub(super) struct ScratchpadEditTool {
    pub session_manager: SessionManager,
    pub session_id: Arc<RwLock<Option<Uuid>>>,
    /// See [`ScratchpadWriteTool::inherited_names`].
    pub inherited_names: Vec<String>,
}

#[async_trait]
impl Tool for ScratchpadEditTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "scratchpad_edit".to_string(),
            description: "Edit a scratchpad entry in place. Provide 'content' to fully \
                overwrite, or 'old_string'/'new_string' for targeted string replacement \
                (like edit_file). When you are a sub-agent, names inherited read-only \
                from the parent are rejected. Copy the content into your own entry first \
                if you need to mutate it."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The scratchpad entry name to edit"
                    },
                    "content": {
                        "type": "string",
                        "description": "Full replacement content (mutually exclusive with old_string/new_string)"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to find and replace"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement string"
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "If true, replace all occurrences. Defaults to false."
                    }
                },
                "required": ["name"]
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
        let name = input["name"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_edit".to_string(),
                message: "missing 'name' parameter".to_string(),
            })?;

        if is_inherited(&self.inherited_names, name) {
            return inherited_write_error(name);
        }

        let session_id = resolve_session_id(&self.session_id, "scratchpad_edit").await?;

        // Full overwrite mode
        if let Some(new_content) = input.get("content").and_then(|v| v.as_str()) {
            let updated = self
                .session_manager
                .update_tool_output(session_id, name, new_content)
                .await?;

            return if updated {
                Ok(ToolOutput::text(
                    format!(
                        "Scratchpad entry \"{}\" overwritten ({} bytes)",
                        name,
                        new_content.len()
                    ),
                    false,
                ))
            } else {
                Ok(ToolOutput::text(
                    format!("Scratchpad entry \"{}\" not found", name),
                    true,
                ))
            };
        }

        // String replacement mode
        let old_string = input
            .get("old_string")
            .and_then(|v| v.as_str())
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_edit".to_string(),
                message: "provide either 'content' for full overwrite \
                    or 'old_string'/'new_string' for replacement"
                    .to_string(),
            })?;
        let new_string = input
            .get("new_string")
            .and_then(|v| v.as_str())
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_edit".to_string(),
                message: "missing 'new_string' parameter".to_string(),
            })?;
        let replace_all = input["replace_all"].as_bool().unwrap_or(false);

        let existing = self
            .session_manager
            .load_tool_output(session_id, name)
            .await?
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_edit".to_string(),
                message: format!("scratchpad entry \"{}\" not found", name),
            })?;

        if !existing.contains(old_string) {
            return Ok(ToolOutput::text(
                format!(
                    "Error: '{}' not found in scratchpad entry \"{}\"",
                    super::util::truncate_string(old_string, 100),
                    name,
                ),
                true,
            ));
        }

        let (updated_content, count) = if replace_all {
            let count = existing.matches(old_string).count();
            (existing.replace(old_string, new_string), count)
        } else {
            (existing.replacen(old_string, new_string, 1), 1)
        };

        self.session_manager
            .update_tool_output(session_id, name, &updated_content)
            .await?;

        Ok(ToolOutput::text(
            format!(
                "Scratchpad entry \"{}\": replaced {} occurrence(s) ({} bytes)",
                name,
                count,
                updated_content.len(),
            ),
            false,
        ))
    }
}

pub(super) struct ScratchpadListTool {
    pub session_manager: SessionManager,
    pub session_id: Arc<RwLock<Option<Uuid>>>,
    /// Sub-agent fallback: list also enumerates parent entries filtered by
    /// [`Self::inherited_names`], rendered in a trailing `(inherited)` section. `None` on the
    /// primary agent.
    pub parent_session_id: Option<Uuid>,
    /// Allowlist of parent-scoped scratchpad names visible to this sub-agent. Empty on the primary
    /// agent.
    pub inherited_names: Vec<String>,
}

#[async_trait]
impl Tool for ScratchpadListTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "scratchpad_list".to_string(),
            description: "List scratchpad entries in the current session with their name, size, \
                creation time, and origin. Sub-agent entries inherited read-only from the \
                parent session appear in the same table with origin `inherited`."
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
        let session_id = resolve_session_id(&self.session_id, "scratchpad_list").await?;

        let own = self.session_manager.list_tool_outputs(session_id).await?;

        // Inherited (parent) entries, filtered to the allowlist so the sub-agent never sees parent
        // state it wasn't explicitly granted.
        let mut rows: Vec<(ToolOutputSummary, &'static str)> =
            own.into_iter().map(|entry| (entry, "own")).collect();
        if let Some(parent_sid) = self.parent_session_id
            && !self.inherited_names.is_empty()
        {
            let parent_entries = self.session_manager.list_tool_outputs(parent_sid).await?;
            rows.extend(
                parent_entries
                    .into_iter()
                    .filter(|entry| self.inherited_names.iter().any(|n| n == &entry.name))
                    .map(|entry| (entry, "inherited")),
            );
        }

        if rows.is_empty() {
            return Ok(ToolOutput::text("Scratchpad is empty.".to_string(), false));
        }

        let table_rows: Vec<Vec<String>> = rows
            .iter()
            .map(|(entry, origin)| {
                vec![
                    entry.name.clone(),
                    format_size(entry.size),
                    entry.created_at[..19.min(entry.created_at.len())].to_string(),
                    origin.to_string(),
                ]
            })
            .collect();
        let mut output =
            crate::render::format_columns(&["Name", "Size", "Created", "Origin"], &table_rows);
        output.push_str(&format!("\n{} entries total", rows.len()));

        Ok(ToolOutput::text(output, false))
    }
}

pub(super) struct ScratchpadMergeTool {
    pub session_manager: SessionManager,
    pub session_id: Arc<RwLock<Option<Uuid>>>,
    /// See [`ScratchpadReadTool::parent_session_id`]. Sources may be inherited from the parent's
    /// scratchpad.
    pub parent_session_id: Option<Uuid>,
    /// See [`ScratchpadWriteTool::inherited_names`]. The `target` is blocked if listed here;
    /// sources are also matched against this set for the parent-fallback read.
    pub inherited_names: Vec<String>,
}

#[async_trait]
impl Tool for ScratchpadMergeTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "scratchpad_merge".to_string(),
            description: "Combine multiple scratchpad entries into one without routing the \
                bytes through the conversation. Useful for collecting parallel sub-agent \
                reports or any accumulated scratchpad data. `format` controls how entries \
                are joined: `concat_with_headers` (default, prepends `--- name ---` before \
                each entry), `concat` (plain join), or `json_array` (each source parsed as \
                JSON if valid, else quoted as a string; result is a compact JSON array). \
                Sub-agents cannot merge into a name inherited read-only from the parent. \
                Source entries may be inherited."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "sources": {
                        "type": "array",
                        "items": { "type": "string" },
                        "minItems": 1,
                        "description": "Names of scratchpad entries to combine, in order."
                    },
                    "target": {
                        "type": "string",
                        "description": "Name to store the merged result under. Overwrites if it already exists."
                    },
                    "format": {
                        "type": "string",
                        "enum": ["concat_with_headers", "concat", "json_array"],
                        "description": "How to join the source entries. Defaults to `concat_with_headers`."
                    }
                },
                "required": ["sources", "target"]
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
        let target = input["target"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_merge".to_string(),
                message: "missing 'target' parameter".to_string(),
            })?;

        if is_inherited(&self.inherited_names, target) {
            return inherited_write_error(target);
        }

        let sources: Vec<String> = input["sources"]
            .as_array()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_merge".to_string(),
                message: "missing 'sources' parameter (expected non-empty array of strings)"
                    .to_string(),
            })?
            .iter()
            .filter_map(|item| item.as_str().map(str::to_string))
            .collect();

        if sources.is_empty() {
            return Ok(ToolOutput::text(
                "scratchpad_merge: 'sources' must contain at least one entry name".to_string(),
                true,
            ));
        }

        let format = input
            .get("format")
            .and_then(|value| value.as_str())
            .unwrap_or("concat_with_headers");

        let session_id = resolve_session_id(&self.session_id, "scratchpad_merge").await?;

        // Resolve each source. Inheritance-aware: child first, then parent if the name is
        // allowlisted; same fallback path as ScratchpadReadTool. Any missing source aborts the
        // merge so we never partially write the target.
        let mut loaded: Vec<(String, String)> = Vec::with_capacity(sources.len());
        for name in &sources {
            let mut content = self
                .session_manager
                .load_tool_output(session_id, name)
                .await?;
            if content.is_none()
                && let Some(parent_sid) = self.parent_session_id
                && self.inherited_names.iter().any(|n| n == name)
            {
                content = self
                    .session_manager
                    .load_tool_output(parent_sid, name)
                    .await?;
            }
            let content = content.ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_merge".to_string(),
                message: format!("scratchpad entry \"{}\" not found", name),
            })?;
            loaded.push((name.clone(), content));
        }

        let merged = match format {
            "concat" => loaded
                .iter()
                .map(|(_, body)| body.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
            "json_array" => {
                let values: Vec<serde_json::Value> = loaded
                    .iter()
                    .map(|(_, body)| {
                        serde_json::from_str::<serde_json::Value>(body)
                            .unwrap_or_else(|_| serde_json::Value::String(body.clone()))
                    })
                    .collect();
                serde_json::to_string(&values).map_err(|error| MekaError::ToolExecution {
                    tool_name: "scratchpad_merge".to_string(),
                    message: format!("failed to serialize merged JSON array: {}", error),
                })?
            }
            "concat_with_headers" => {
                let mut output = String::new();
                for (idx, (name, body)) in loaded.iter().enumerate() {
                    if idx > 0 {
                        output.push('\n');
                    }
                    output.push_str(&format!("--- {} ---\n", name));
                    output.push_str(body);
                }
                output
            }
            other => {
                return Ok(ToolOutput::text(
                    format!(
                        "scratchpad_merge: unknown `format` value '{}' (expected \
                         'concat_with_headers', 'concat', or 'json_array')",
                        other,
                    ),
                    true,
                ));
            }
        };

        let merged_size = merged.len();
        self.session_manager
            .save_tool_output(session_id, target, &merged)
            .await?;

        Ok(ToolOutput::text(
            format!(
                "Merged {} entries into scratchpad entry \"{}\" ({} bytes)",
                loaded.len(),
                target,
                merged_size,
            ),
            false,
        ))
    }
}

pub(super) struct ScratchpadDeleteTool {
    pub session_manager: SessionManager,
    pub session_id: Arc<RwLock<Option<Uuid>>>,
    /// See [`ScratchpadWriteTool::inherited_names`].
    pub inherited_names: Vec<String>,
}

#[async_trait]
impl Tool for ScratchpadDeleteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "scratchpad_delete".to_string(),
            description: "Delete a scratchpad entry by name to free up space. When you are \
                a sub-agent, names inherited read-only from the parent cannot be deleted."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The scratchpad entry name to delete"
                    }
                },
                "required": ["name"]
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
        let name = input["name"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_delete".to_string(),
                message: "missing 'name' parameter".to_string(),
            })?;

        if is_inherited(&self.inherited_names, name) {
            return inherited_write_error(name);
        }

        let session_id = resolve_session_id(&self.session_id, "scratchpad_delete").await?;

        let deleted = self
            .session_manager
            .delete_tool_output(session_id, name)
            .await?;

        if deleted {
            Ok(ToolOutput::text(
                format!("Scratchpad entry \"{}\" deleted", name),
                false,
            ))
        } else {
            Ok(ToolOutput::text(
                format!("Scratchpad entry \"{}\" not found", name),
                true,
            ))
        }
    }
}

pub(super) struct ScratchpadRenameTool {
    pub session_manager: SessionManager,
    pub session_id: Arc<RwLock<Option<Uuid>>>,
    /// See [`ScratchpadWriteTool::inherited_names`].
    pub inherited_names: Vec<String>,
}

#[async_trait]
impl Tool for ScratchpadRenameTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "scratchpad_rename".to_string(),
            description: "Rename a scratchpad entry from `old` to `new` without round-tripping \
                the content through the conversation. Errors if `old` doesn't exist, if `new` \
                already exists, or (for sub-agents) if either name is inherited read-only from \
                the parent."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "old": {
                        "type": "string",
                        "description": "Current scratchpad entry name"
                    },
                    "new": {
                        "type": "string",
                        "description": "Replacement scratchpad entry name"
                    }
                },
                "required": ["old", "new"]
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
        let old = input["old"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_rename".to_string(),
                message: "missing 'old' parameter".to_string(),
            })?;
        let new = input["new"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_rename".to_string(),
                message: "missing 'new' parameter".to_string(),
            })?;

        // Block both ends. Renaming away from an inherited source would be a no-op against the
        // parent's row but implies the sub-agent owns the name; renaming to an inherited target
        // would create a child shadow. Either way: reject early with the same error text the other
        // mutators use, naming the offending entry.
        if is_inherited(&self.inherited_names, old) {
            return inherited_write_error(old);
        }
        if is_inherited(&self.inherited_names, new) {
            return inherited_write_error(new);
        }

        if old == new {
            return Ok(ToolOutput::text(
                format!("Scratchpad entry \"{}\" is already named that", old),
                true,
            ));
        }

        let session_id = resolve_session_id(&self.session_id, "scratchpad_rename").await?;

        let outcome = self
            .session_manager
            .rename_tool_output(session_id, old, new)
            .await?;

        match outcome {
            crate::session::RenameOutcome::Renamed => Ok(ToolOutput::text(
                format!("Renamed scratchpad entry \"{}\" to \"{}\"", old, new),
                false,
            )),
            crate::session::RenameOutcome::NotFound => Ok(ToolOutput::text(
                format!("Scratchpad entry \"{}\" not found", old),
                true,
            )),
            crate::session::RenameOutcome::TargetExists => Ok(ToolOutput::text(
                format!(
                    "Scratchpad entry \"{}\" already exists; delete it first or pick a \
                     different name",
                    new,
                ),
                true,
            )),
        }
    }
}

pub(super) struct ScratchpadLoadFileTool {
    pub session_manager: SessionManager,
    pub session_id: Arc<RwLock<Option<Uuid>>>,
    /// See [`ScratchpadWriteTool::inherited_names`].
    pub inherited_names: Vec<String>,
    /// Per-session cwd, so a relative `path` resolves the same way `read_file` does (against the
    /// agent's `/cd` directory, not the process cwd).
    pub cwd: crate::workspace::SharedCwd,
}

#[async_trait]
impl Tool for ScratchpadLoadFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "scratchpad_load_file".to_string(),
            description: "Read a file's contents directly into the scratchpad without routing \
                the bytes through the conversation. Useful for staging a large captured log or \
                document that you want to hand to sub-agents via `inherit_scratchpad` on \
                `agent_spawn` \u{2014} the model never sees the payload. UTF-8 text only; binary \
                files are rejected with the detected MIME type. For binary content, pass the \
                file path directly to whatever tool will consume it; sub-agents inherit the \
                parent's filesystem access. Overwrites an existing entry of the same name. \
                Sub-agents cannot load into a name inherited read-only from the parent."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to read"
                    },
                    "name": {
                        "type": "string",
                        "description": "Name to store the contents under in the scratchpad"
                    }
                },
                "required": ["path", "name"]
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
        let path = input["path"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_load_file".to_string(),
                message: "missing 'path' parameter".to_string(),
            })?
            .to_string();
        let name = input["name"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_load_file".to_string(),
                message: "missing 'name' parameter".to_string(),
            })?;

        if is_inherited(&self.inherited_names, name) {
            return inherited_write_error(name);
        }

        let session_id = resolve_session_id(&self.session_id, "scratchpad_load_file").await?;

        // Resolve a relative path against the session cwd before canonicalizing, so `/cd` is
        // honoured (canonicalize alone would resolve relative to the process cwd).
        let resolved = crate::workspace::resolve_against_cwd(&self.cwd, &path);
        let canonical =
            super::util::canonicalize_for_tool("scratchpad_load_file", &resolved).await?;

        // We read the file as raw bytes (rather than directly as a String) so that on a UTF-8
        // failure we can run a single content sniff and tell the model what kind of binary it just
        // tried to load. The happy path then incurs one extra allocation; for the sizes this tool
        // is meant for (tens of MB), that's negligible.
        let bytes = super::file::read_file_bytes(&canonical)
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "scratchpad_load_file".to_string(),
                message: format!("failed to read '{}': {}", path, error),
            })?;

        let content = match String::from_utf8(bytes) {
            Ok(text) => text,
            Err(utf8_error) => {
                let bytes = utf8_error.as_bytes();
                let detected = infer::get(bytes)
                    .map(|kind| format!(" Detected MIME type: {}.", kind.mime_type()))
                    .unwrap_or_default();
                return Ok(ToolOutput::text(
                    format!(
                        "'{}' is not valid UTF-8: {}.{}",
                        path,
                        utf8_error.utf8_error(),
                        detected,
                    ),
                    true,
                ));
            }
        };

        let byte_count = content.len();
        self.session_manager
            .save_tool_output(session_id, name, &content)
            .await?;

        Ok(ToolOutput::text(
            format!(
                "Loaded {} bytes from '{}' into scratchpad entry \"{}\"",
                byte_count, path, name,
            ),
            false,
        ))
    }
}

pub(super) struct ScratchpadSaveFileTool {
    /// The write boundary, shared with `write_file`. This tool reads as the scratchpad's
    /// `write_file` and lands bytes at a path the user named, so it is fenced the same way.
    pub scope: crate::workspace::WriteScope,
    pub session_manager: SessionManager,
    pub session_id: Arc<RwLock<Option<Uuid>>>,
    /// See [`ScratchpadReadTool::parent_session_id`].
    pub parent_session_id: Option<Uuid>,
    /// See [`ScratchpadReadTool::inherited_names`].
    pub inherited_names: Vec<String>,
    /// Per-session cwd, so a relative `path` resolves the same way `write_file` does (against the
    /// agent's `/cd` directory, not the process cwd).
    pub cwd: crate::workspace::SharedCwd,
    /// The same tracker `write_file` and `edit_file` stamp.
    ///
    /// This tool is described to the model as the scratchpad's `write_file`, and it lands bytes at
    /// a path the user named -- so it has to leave the same record. Without it the tracker still
    /// held the pre-save stamp, and the next `write_file` or `edit_file` on that path was refused
    /// with "Something else wrote to it (a shell command, another agent, or the user)". The writer
    /// was meka, one call earlier.
    pub read_tracker: crate::tools::ReadTracker,
}

#[async_trait]
impl Tool for ScratchpadSaveFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "scratchpad_save_file".to_string(),
            description: "Write the contents of a scratchpad entry to a file on disk without \
                routing the bytes through the conversation. Useful for persisting a sub-agent's \
                report or a large extracted result. Mirrors `write_file`: creates parent \
                directories, overwrites by default, UTF-8 only. Sub-agents can save inherited \
                entries (read from parent, write to disk) without copying through the model."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The scratchpad entry to read from"
                    },
                    "path": {
                        "type": "string",
                        "description": "The file path to write to"
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Replace the file if it already exists. Without this, \
                                        saving over an existing file is refused"
                    }
                },
                "required": ["name", "path"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Workspace
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let name = input["name"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_save_file".to_string(),
                message: "missing 'name' parameter".to_string(),
            })?;
        let path = input["path"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_save_file".to_string(),
                message: "missing 'path' parameter".to_string(),
            })?
            .to_string();

        let session_id = resolve_session_id(&self.session_id, "scratchpad_save_file").await?;

        // Inherited-fallback read: mirror ScratchpadReadTool. Lets a sub- agent flush a parent's
        // allowlisted entry to disk without having to first copy it into its own scratchpad.
        let mut content = self
            .session_manager
            .load_tool_output(session_id, name)
            .await?;
        if content.is_none()
            && let Some(parent_sid) = self.parent_session_id
            && self.inherited_names.iter().any(|n| n == name)
        {
            content = self
                .session_manager
                .load_tool_output(parent_sid, name)
                .await?;
        }
        let content = content.ok_or_else(|| MekaError::ToolExecution {
            tool_name: "scratchpad_save_file".to_string(),
            message: format!("scratchpad entry \"{}\" not found", name),
        })?;

        // Shared with `write_file` rather than mirrored, because the two must agree on the file
        // they name and on the lock they take, and a copy of the resolution here agreed on neither.
        // Both are dispatched concurrently from one assistant message and both write through a temp
        // file derived from the target, so two calls naming one path could interleave.
        let (target, _write_guard) = super::file::resolve_write_target(
            "scratchpad_save_file",
            &self.cwd,
            &self.scope,
            &path,
        )
        .await?;

        // Asked *under* the write lock, so the answer is still true when the write happens.
        //
        // This tool reads as the scratchpad's `write_file` and its description says so, but it had
        // none of that tool's protections: no read of the target, so no staleness check to fail; no
        // `force`, so nothing to bypass and nothing to consult; and a confirmation that named the
        // bytes written while never mentioning the bytes replaced. A path the model named by
        // mistake was gone with a success message on top of it. `force` is the same escape hatch
        // `write_file` and `edit_file` offer, spelled the same way.
        let force = input
            .get("force")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let replaced = tokio::fs::metadata(&target)
            .await
            .ok()
            .map(|meta| meta.len());
        if let Some(existing_bytes) = replaced
            && !force
        {
            return Ok(ToolOutput::text(
                format!(
                    "Error: '{}' already exists ({} bytes) and saving would replace it. Read it \
                     first if you need what it says, pick another path, or set force=true.",
                    path, existing_bytes
                ),
                true,
            ));
        }

        let byte_count = content.len();
        super::file::write_file_bytes(&target, content.as_bytes())
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "scratchpad_save_file".to_string(),
                message: format!("failed to write '{}': {}", path, error),
            })?;

        // Stamped like any other write meka performs, so the next `write_file` or `edit_file` on
        // this path does not mistake meka's own bytes for someone else's. `FileRoute::Local`
        // because this wrote to disk directly rather than through an editor delegate.
        super::file::record_write(
            &self.read_tracker,
            target.clone(),
            super::file::FileRoute::Local,
            &content,
        )
        .await;

        Ok(ToolOutput::text(
            format!(
                "Saved {} bytes from scratchpad entry \"{}\" to '{}'{}",
                byte_count,
                name,
                path,
                match replaced {
                    Some(existing_bytes) => format!(", replacing {} bytes", existing_bytes),
                    None => String::new(),
                }
            ),
            false,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::{
        provider::{ContentBlock, Role},
        tools::tests::text_content,
    };

    async fn test_manager() -> SessionManager {
        SessionManager::open(Some(Path::new(":memory:")), &Default::default())
            .await
            .expect("failed to open in-memory database")
    }

    fn test_session_id(uuid: Uuid) -> Arc<RwLock<Option<Uuid>>> {
        Arc::new(RwLock::new(Some(uuid)))
    }

    fn make_assistant_message(tool_calls: Vec<(&str, &str, serde_json::Value)>) -> Message {
        Message {
            role: Role::Assistant,
            content: tool_calls
                .into_iter()
                .map(|(id, name, input)| ContentBlock::ToolUse {
                    id: id.to_string(),
                    name: name.to_string(),
                    input,
                })
                .collect(),
        }
    }

    // -- persist_oversized_results --

    /// `scratchpad_save_file` is fenced by the scope it was built with, and stamps the tracker.
    ///
    /// Two gaps in one place. Every other test here builds the tool with
    /// `WriteScope::unconfined()`, so replacing `&self.scope` with a fresh unconfined scope left
    /// the suite green -- and this tool is described to the model as the scratchpad's `write_file`,
    /// so that is a full write door outside the boundary at `workspace`. Separately it never
    /// recorded its write, so the next `write_file`/`edit_file` on the same path was refused with
    /// "Something else wrote to it", blaming a shell command or the user for meka's own bytes.
    #[tokio::test]
    async fn saving_a_file_is_fenced_by_its_scope_and_records_the_write() {
        let temp = tempfile::tempdir().expect("tempdir");
        let workspace = crate::workspace::canonical_for_test(temp.path()).join("work");
        std::fs::create_dir_all(&workspace).expect("workspace");
        let outside = crate::workspace::canonical_for_test(temp.path()).join("outside");
        std::fs::create_dir_all(&outside).expect("outside");

        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        manager
            .save_tool_output(session_id, "report", "FINDINGS")
            .await
            .expect("seed");

        let read_tracker: crate::tools::ReadTracker =
            std::sync::Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new()));
        let tool = ScratchpadSaveFileTool {
            read_tracker: std::sync::Arc::clone(&read_tracker),
            scope: crate::workspace::WriteScope::confined(vec![workspace.clone()]),
            session_manager: manager,
            session_id: test_session_id(session_id),
            parent_session_id: None,
            inherited_names: Vec::new(),
            cwd: std::sync::Arc::new(std::sync::RwLock::new(workspace.clone())),
        };

        // Outside the one root: refused, and nothing lands.
        let escaped = outside.join("leaked.txt");
        let refused = tool
            .execute(
                serde_json::json!({
                    "name": "report",
                    "path": escaped.to_str().expect("path"),
                    "force": true,
                }),
                CancellationToken::new(),
            )
            .await;
        let refused_text = match refused {
            Err(error) => error.to_string(),
            Ok(output) => crate::provider::ContentBlock::tool_result_text_content(&output.content),
        };
        assert!(
            !escaped.exists(),
            "a save outside the workspace must not land: {refused_text}"
        );

        // Inside it: lands, and the tracker records that meka was the writer.
        let inside = workspace.join("report.txt");
        tool.execute(
            serde_json::json!({
                "name": "report",
                "path": inside.to_str().expect("path"),
                "force": true,
            }),
            CancellationToken::new(),
        )
        .await
        .expect("a save inside the workspace must succeed");
        assert_eq!(
            std::fs::read_to_string(&inside).expect("read back"),
            "FINDINGS"
        );
        assert!(
            read_tracker.read().await.contains_key(&inside),
            "the write must be stamped, or the next write_file blames someone else for it"
        );
    }

    #[tokio::test]
    async fn test_persist_oversized_results_replaces_large_text() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let large_text = "x".repeat(MAX_INLINE_RESULT_BYTES + 1000);
        let assistant_msg =
            make_assistant_message(vec![("call-1", "execute_command", serde_json::json!({}))]);
        let mut results = vec![ContentBlock::ToolResult {
            tool_use_id: "call-1".to_string(),
            content: vec![ToolResultContent::Text {
                text: large_text.clone(),
            }],
            is_error: false,
        }];

        persist_oversized_results(
            &manager,
            session_id,
            &assistant_msg,
            &mut results,
            &std::collections::HashMap::new(),
        )
        .await
        .expect("persist");

        if let ContentBlock::ToolResult { content, .. } = &results[0] {
            let text = ContentBlock::tool_result_text_content(content);
            assert!(text.contains("<large-output"));
            // The tool name still leads, so the handle stays recognisable; the call-id tail after
            // it is what makes it unique (see the collision test below).
            assert!(text.contains("name=\"execute_command_"), "{}", text);
            assert!(text.contains("scratchpad_read"));
            assert!(!text.contains(&large_text));
        } else {
            panic!("expected ToolResult");
        }

        let name = format!("execute_command_{}_1", short_call_id("call-1"));
        let loaded = manager
            .load_tool_output(session_id, &name)
            .await
            .expect("load");
        assert_eq!(loaded, Some(large_text));
    }

    /// The counter restarts on every call, so a name built from it alone repeated across turns --
    /// and `save_tool_output` is `INSERT OR REPLACE`, so the later spill silently destroyed the
    /// earlier one. The model still held the first handle from its conversation and would read back
    /// the *second* result under it, with nothing to signal the substitution.
    #[tokio::test]
    async fn spilled_outputs_from_different_calls_do_not_overwrite_each_other() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let mut names = Vec::new();
        for (call_id, body) in [("call-turn1", 'a'), ("call-turn3", 'b')] {
            let text = body.to_string().repeat(MAX_INLINE_RESULT_BYTES + 100);
            let assistant_msg =
                make_assistant_message(vec![(call_id, "execute_command", serde_json::json!({}))]);
            let mut results = vec![ContentBlock::ToolResult {
                tool_use_id: call_id.to_string(),
                content: vec![ToolResultContent::Text { text: text.clone() }],
                is_error: false,
            }];
            persist_oversized_results(
                &manager,
                session_id,
                &assistant_msg,
                &mut results,
                &std::collections::HashMap::new(),
            )
            .await
            .expect("persist");
            names.push((
                format!("execute_command_{}_1", short_call_id(call_id)),
                text,
            ));
        }

        assert_ne!(names[0].0, names[1].0, "two calls must not share a name");
        for (name, expected) in &names {
            assert_eq!(
                manager
                    .load_tool_output(session_id, name)
                    .await
                    .expect("load"),
                Some(expected.clone()),
                "entry {} was overwritten",
                name
            );
        }
    }

    #[tokio::test]
    async fn test_persist_oversized_results_leaves_small_text() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let small_text = "hello world".to_string();
        let assistant_msg =
            make_assistant_message(vec![("call-1", "execute_command", serde_json::json!({}))]);
        let mut results = vec![ContentBlock::ToolResult {
            tool_use_id: "call-1".to_string(),
            content: vec![ToolResultContent::Text {
                text: small_text.clone(),
            }],
            is_error: false,
        }];

        persist_oversized_results(
            &manager,
            session_id,
            &assistant_msg,
            &mut results,
            &std::collections::HashMap::new(),
        )
        .await
        .expect("persist");

        if let ContentBlock::ToolResult { content, .. } = &results[0] {
            assert_eq!(ContentBlock::tool_result_text_content(content), small_text);
        }
    }

    // -- save_explicit_scratchpad_results --

    #[tokio::test]
    async fn test_explicit_scratchpad_save() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let assistant_msg = make_assistant_message(vec![(
            "call-1",
            "execute_command",
            serde_json::json!({"command": "echo hi", "scratchpad": "cmd_output"}),
        )]);

        let mut results = vec![ContentBlock::ToolResult {
            tool_use_id: "call-1".to_string(),
            content: vec![ToolResultContent::Text {
                text: "hi\n".to_string(),
            }],
            is_error: false,
        }];

        save_explicit_scratchpad_results(&manager, session_id, &assistant_msg, &mut results)
            .await
            .expect("save");

        // Result should be replaced with a reference
        if let ContentBlock::ToolResult { content, .. } = &results[0] {
            let text = ContentBlock::tool_result_text_content(content);
            assert!(text.contains("cmd_output"));
            assert!(text.contains("scratchpad_read"));
            assert!(!text.contains("hi\n"));
        }

        // Content should be in the DB
        let loaded = manager
            .load_tool_output(session_id, "cmd_output")
            .await
            .expect("load");
        assert_eq!(loaded, Some("hi\n".to_string()));
    }

    #[tokio::test]
    async fn test_explicit_scratchpad_not_requested() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let assistant_msg = make_assistant_message(vec![(
            "call-1",
            "execute_command",
            serde_json::json!({"command": "echo hi"}),
        )]);

        let mut results = vec![ContentBlock::ToolResult {
            tool_use_id: "call-1".to_string(),
            content: vec![ToolResultContent::Text {
                text: "hi\n".to_string(),
            }],
            is_error: false,
        }];

        save_explicit_scratchpad_results(&manager, session_id, &assistant_msg, &mut results)
            .await
            .expect("save");

        // Result should be unchanged
        if let ContentBlock::ToolResult { content, .. } = &results[0] {
            assert_eq!(ContentBlock::tool_result_text_content(content), "hi\n");
        }
    }

    #[tokio::test]
    async fn test_explicit_scratchpad_ignores_non_scratchpad_keys() {
        // Regression test for the `render_image` clobber bug: when a tool uses `from_scratchpad`
        // (as an input-source param) instead of `scratchpad` (the output-destination convention),
        // the agent-layer save must not touch the pre-existing scratchpad entry.
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_tool_output(session_id, "img", "original base64 data")
            .await
            .expect("seed");

        let assistant_msg = make_assistant_message(vec![(
            "call-1",
            "render_image",
            serde_json::json!({"from_scratchpad": "img"}),
        )]);

        let mut results = vec![ContentBlock::ToolResult {
            tool_use_id: "call-1".to_string(),
            content: vec![ToolResultContent::Text {
                text: "[Image rendered from scratchpad \"img\"]".to_string(),
            }],
            is_error: false,
        }];

        save_explicit_scratchpad_results(&manager, session_id, &assistant_msg, &mut results)
            .await
            .expect("save");

        // Pre-existing scratchpad entry should be untouched.
        let loaded = manager
            .load_tool_output(session_id, "img")
            .await
            .expect("load");
        assert_eq!(loaded.as_deref(), Some("original base64 data"));

        // Result content should also be unchanged (no rewriting to a reference).
        if let ContentBlock::ToolResult { content, .. } = &results[0] {
            assert_eq!(
                ContentBlock::tool_result_text_content(content),
                "[Image rendered from scratchpad \"img\"]"
            );
        }
    }

    // -- scratchpad_write --

    #[tokio::test]
    async fn test_scratchpad_write() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let tool = ScratchpadWriteTool {
            session_manager: manager.clone(),
            session_id: test_session_id(session_id),
            inherited_names: Vec::new(),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "greeting", "content": "hello world"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        assert!(!result.is_error);
        assert!(text_content(&result).contains("greeting"));

        let loaded = manager
            .load_tool_output(session_id, "greeting")
            .await
            .expect("load");
        assert_eq!(loaded, Some("hello world".to_string()));
    }

    #[tokio::test]
    async fn test_scratchpad_write_overwrites() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_tool_output(session_id, "notes", "old content")
            .await
            .expect("save");

        let tool = ScratchpadWriteTool {
            session_manager: manager.clone(),
            session_id: test_session_id(session_id),
            inherited_names: Vec::new(),
        };

        tool.execute(
            serde_json::json!({"name": "notes", "content": "new content"}),
            CancellationToken::new(),
        )
        .await
        .expect("execute");

        let loaded = manager
            .load_tool_output(session_id, "notes")
            .await
            .expect("load");
        assert_eq!(loaded, Some("new content".to_string()));
    }

    // -- scratchpad_read --

    #[tokio::test]
    async fn test_scratchpad_read() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_tool_output(session_id, "data", "line1\nline2\nline3\n")
            .await
            .expect("save");

        let tool = ScratchpadReadTool {
            session_manager: manager,
            session_id: test_session_id(session_id),
            parent_session_id: None,
            inherited_names: Vec::new(),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "data"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        assert!(!result.is_error);
        let text = text_content(&result);
        assert!(text.contains("line1"));
        assert!(text.contains("line3"));
    }

    #[tokio::test]
    async fn test_scratchpad_read_with_offset_and_limit() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_tool_output(session_id, "abc", "abcdefghij")
            .await
            .expect("save");

        let tool = ScratchpadReadTool {
            session_manager: manager,
            session_id: test_session_id(session_id),
            parent_session_id: None,
            inherited_names: Vec::new(),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "abc", "offset": 3, "limit": 4}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        let text = text_content(&result);
        assert!(text.contains("defg"));
        assert!(text.contains("showing bytes 3..7 of 10"));
    }

    #[tokio::test]
    async fn test_scratchpad_read_search_mode() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_tool_output(
                session_id,
                "fruits",
                "apple\nbanana\napricot\ncherry\navocado\n",
            )
            .await
            .expect("save");

        let tool = ScratchpadReadTool {
            session_manager: manager,
            session_id: test_session_id(session_id),
            parent_session_id: None,
            inherited_names: Vec::new(),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "fruits", "regex": "^a"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        let text = text_content(&result);
        assert!(text.contains("1:apple"));
        assert!(text.contains("3:apricot"));
        assert!(text.contains("5:avocado"));
        assert!(!text.contains("banana"));
    }

    #[tokio::test]
    async fn test_scratchpad_read_not_found() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let tool = ScratchpadReadTool {
            session_manager: manager,
            session_id: test_session_id(session_id),
            parent_session_id: None,
            inherited_names: Vec::new(),
        };

        let result = tool.execute(
            serde_json::json!({"name": "nonexistent"}),
            CancellationToken::new(),
        );
        assert!(result.await.is_err());
    }

    // -- scratchpad_edit --

    #[tokio::test]
    async fn test_scratchpad_edit_overwrite() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_tool_output(session_id, "doc", "old content")
            .await
            .expect("save");

        let tool = ScratchpadEditTool {
            session_manager: manager.clone(),
            session_id: test_session_id(session_id),
            inherited_names: Vec::new(),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "doc", "content": "new content"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        assert!(!result.is_error);
        assert!(text_content(&result).contains("overwritten"));

        let loaded = manager
            .load_tool_output(session_id, "doc")
            .await
            .expect("load");
        assert_eq!(loaded, Some("new content".to_string()));
    }

    #[tokio::test]
    async fn test_scratchpad_edit_replacement() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_tool_output(session_id, "doc", "hello world hello")
            .await
            .expect("save");

        let tool = ScratchpadEditTool {
            session_manager: manager.clone(),
            session_id: test_session_id(session_id),
            inherited_names: Vec::new(),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "doc", "old_string": "hello", "new_string": "hi"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        assert!(text_content(&result).contains("1 occurrence(s)"));

        let loaded = manager
            .load_tool_output(session_id, "doc")
            .await
            .expect("load");
        assert_eq!(loaded, Some("hi world hello".to_string()));
    }

    #[tokio::test]
    async fn test_scratchpad_edit_replace_all() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_tool_output(session_id, "doc", "foo bar foo baz foo")
            .await
            .expect("save");

        let tool = ScratchpadEditTool {
            session_manager: manager.clone(),
            session_id: test_session_id(session_id),
            inherited_names: Vec::new(),
        };

        let result = tool
            .execute(
                serde_json::json!({
                    "name": "doc",
                    "old_string": "foo",
                    "new_string": "qux",
                    "replace_all": true
                }),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        assert!(text_content(&result).contains("3 occurrence(s)"));

        let loaded = manager
            .load_tool_output(session_id, "doc")
            .await
            .expect("load");
        assert_eq!(loaded, Some("qux bar qux baz qux".to_string()));
    }

    // -- scratchpad_list --

    #[tokio::test]
    async fn test_scratchpad_list_empty() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let tool = ScratchpadListTool {
            session_manager: manager,
            session_id: test_session_id(session_id),
            parent_session_id: None,
            inherited_names: Vec::new(),
        };

        let result = tool
            .execute(serde_json::json!({}), CancellationToken::new())
            .await
            .expect("execute");

        assert!(text_content(&result).contains("empty"));
    }

    #[tokio::test]
    async fn test_scratchpad_list_with_entries() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_tool_output(session_id, "notes", "content one")
            .await
            .expect("save");
        manager
            .save_tool_output(session_id, "data", "content two")
            .await
            .expect("save");

        let tool = ScratchpadListTool {
            session_manager: manager,
            session_id: test_session_id(session_id),
            parent_session_id: None,
            inherited_names: Vec::new(),
        };

        let result = tool
            .execute(serde_json::json!({}), CancellationToken::new())
            .await
            .expect("execute");

        let text = text_content(&result);
        assert!(text.contains("notes"));
        assert!(text.contains("data"));
        assert!(text.contains("2 entries total"));
    }

    // -- scratchpad_delete --

    #[tokio::test]
    async fn test_scratchpad_delete() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_tool_output(session_id, "temp", "temp data")
            .await
            .expect("save");

        let tool = ScratchpadDeleteTool {
            session_manager: manager.clone(),
            session_id: test_session_id(session_id),
            inherited_names: Vec::new(),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "temp"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        assert!(!result.is_error);
        assert!(text_content(&result).contains("deleted"));

        let loaded = manager
            .load_tool_output(session_id, "temp")
            .await
            .expect("load");
        assert_eq!(loaded, None);
    }

    #[tokio::test]
    async fn test_scratchpad_delete_not_found() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let tool = ScratchpadDeleteTool {
            session_manager: manager,
            session_id: test_session_id(session_id),
            inherited_names: Vec::new(),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "nonexistent"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        assert!(result.is_error);
    }

    // -- sub-agent inheritance --

    #[tokio::test]
    async fn test_inherited_scratchpad_read_falls_back_to_parent() {
        let manager = test_manager().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(parent, None, None)
            .await
            .expect("child")
            .0;

        manager
            .save_tool_output(parent, "captured", "parent payload")
            .await
            .expect("seed parent");

        let tool = ScratchpadReadTool {
            session_manager: manager.clone(),
            session_id: test_session_id(child),
            parent_session_id: Some(parent),
            inherited_names: vec!["captured".to_string()],
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "captured"}),
                CancellationToken::new(),
            )
            .await
            .expect("read inherited");

        assert!(!result.is_error);
        assert!(text_content(&result).contains("parent payload"));
    }

    #[tokio::test]
    async fn test_inherited_scratchpad_prefers_child_when_both_present() {
        let manager = test_manager().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(parent, None, None)
            .await
            .expect("child")
            .0;

        manager
            .save_tool_output(parent, "shared", "parent value")
            .await
            .expect("seed parent");
        manager
            .save_tool_output(child, "shared", "child value")
            .await
            .expect("seed child");

        let tool = ScratchpadReadTool {
            session_manager: manager,
            session_id: test_session_id(child),
            parent_session_id: Some(parent),
            inherited_names: vec!["shared".to_string()],
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "shared"}),
                CancellationToken::new(),
            )
            .await
            .expect("read shadowed");

        assert!(text_content(&result).contains("child value"));
    }

    #[tokio::test]
    async fn test_inherited_scratchpad_read_blocks_names_not_in_allowlist() {
        let manager = test_manager().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(parent, None, None)
            .await
            .expect("child")
            .0;

        manager
            .save_tool_output(parent, "secret", "do not leak")
            .await
            .expect("seed parent");

        // allowlist only mentions a different name; the parent's "secret" entry must stay
        // invisible.
        let tool = ScratchpadReadTool {
            session_manager: manager,
            session_id: test_session_id(child),
            parent_session_id: Some(parent),
            inherited_names: vec!["unrelated".to_string()],
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "secret"}),
                CancellationToken::new(),
            )
            .await;
        assert!(result.is_err(), "secret name must not be readable");
    }

    #[tokio::test]
    async fn test_inherited_scratchpad_list_respects_allowlist() {
        let manager = test_manager().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(parent, None, None)
            .await
            .expect("child")
            .0;

        manager
            .save_tool_output(parent, "shared_research", "p1")
            .await
            .expect("seed parent");
        manager
            .save_tool_output(parent, "private_note", "p2")
            .await
            .expect("seed parent");
        manager
            .save_tool_output(child, "own_note", "c1")
            .await
            .expect("seed child");

        let tool = ScratchpadListTool {
            session_manager: manager,
            session_id: test_session_id(child),
            parent_session_id: Some(parent),
            inherited_names: vec!["shared_research".to_string()],
        };

        let result = tool
            .execute(serde_json::json!({}), CancellationToken::new())
            .await
            .expect("list");

        let text = text_content(&result);
        assert!(text.contains("own_note"));
        assert!(text.contains("shared_research"));
        assert!(
            !text.contains("private_note"),
            "non-allowlisted parent entry must not appear, got: {}",
            text
        );
        // Unified-table contract: one Origin header, one totals line, no separate "inherited from
        // parent" section.
        assert_eq!(text.matches("Origin").count(), 1);
        assert!(text.contains("2 entries total"));
        assert!(!text.contains("inherited from parent"));
    }

    #[tokio::test]
    async fn test_inherited_scratchpad_list_handles_child_only_when_empty_allowlist() {
        // When no inheritance is configured, the list behaves exactly as before: no extra section,
        // no parent enumeration.
        let manager = test_manager().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(parent, None, None)
            .await
            .expect("child")
            .0;

        manager
            .save_tool_output(parent, "private", "do not leak")
            .await
            .expect("seed parent");

        let tool = ScratchpadListTool {
            session_manager: manager,
            session_id: test_session_id(child),
            parent_session_id: None,
            inherited_names: Vec::new(),
        };

        let result = tool
            .execute(serde_json::json!({}), CancellationToken::new())
            .await
            .expect("list");

        let text = text_content(&result);
        assert!(!text.contains("private"));
        assert!(!text.contains("(inherited"));
    }

    #[tokio::test]
    async fn test_inherited_scratchpad_list_unified_table_has_origin_column() {
        let manager = test_manager().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(parent, None, None)
            .await
            .expect("child")
            .0;

        manager
            .save_tool_output(parent, "build_log", "p")
            .await
            .expect("seed parent");
        manager
            .save_tool_output(child, "analysis", "c")
            .await
            .expect("seed child");

        let tool = ScratchpadListTool {
            session_manager: manager,
            session_id: test_session_id(child),
            parent_session_id: Some(parent),
            inherited_names: vec!["build_log".to_string()],
        };

        let text = text_content(
            &tool
                .execute(serde_json::json!({}), CancellationToken::new())
                .await
                .expect("list"),
        );

        // Single header (not two sections) with the new `Origin` column.
        assert_eq!(
            text.matches("Origin").count(),
            1,
            "expected one Origin header, got: {}",
            text,
        );
        assert!(text.contains("analysis"));
        assert!(text.contains("own"));
        assert!(text.contains("build_log"));
        assert!(text.contains("inherited"));
        // Old multi-section markers must be gone.
        assert!(!text.contains("inherited from parent"));
        assert!(!text.contains("inherited entries"));
        assert!(text.contains("2 entries total"));
    }

    #[tokio::test]
    async fn test_inherited_scratchpad_write_is_rejected() {
        let manager = test_manager().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(parent, None, None)
            .await
            .expect("child")
            .0;

        manager
            .save_tool_output(parent, "captured", "parent data")
            .await
            .expect("seed");

        let tool = ScratchpadWriteTool {
            session_manager: manager.clone(),
            session_id: test_session_id(child),
            inherited_names: vec!["captured".to_string()],
        };
        let result = tool
            .execute(
                serde_json::json!({"name": "captured", "content": "child override"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        assert!(result.is_error, "write to inherited name must error");
        assert!(text_content(&result).contains("inherited read-only"));
        // Parent untouched.
        assert_eq!(
            manager.load_tool_output(parent, "captured").await.unwrap(),
            Some("parent data".to_string())
        );
        // No child shadow row created.
        assert_eq!(
            manager.load_tool_output(child, "captured").await.unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn test_inherited_scratchpad_edit_is_rejected() {
        let manager = test_manager().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(parent, None, None)
            .await
            .expect("child")
            .0;

        manager
            .save_tool_output(parent, "captured", "parent data")
            .await
            .expect("seed");

        let tool = ScratchpadEditTool {
            session_manager: manager.clone(),
            session_id: test_session_id(child),
            inherited_names: vec!["captured".to_string()],
        };
        let result = tool
            .execute(
                serde_json::json!({"name": "captured", "content": "child override"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        assert!(result.is_error);
        assert!(text_content(&result).contains("inherited read-only"));
        assert_eq!(
            manager.load_tool_output(parent, "captured").await.unwrap(),
            Some("parent data".to_string())
        );
    }

    #[tokio::test]
    async fn test_inherited_scratchpad_delete_is_rejected() {
        let manager = test_manager().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(parent, None, None)
            .await
            .expect("child")
            .0;

        manager
            .save_tool_output(parent, "captured", "parent data")
            .await
            .expect("seed");

        let tool = ScratchpadDeleteTool {
            session_manager: manager.clone(),
            session_id: test_session_id(child),
            inherited_names: vec!["captured".to_string()],
        };
        let result = tool
            .execute(
                serde_json::json!({"name": "captured"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        assert!(result.is_error);
        assert!(text_content(&result).contains("inherited read-only"));
        assert_eq!(
            manager.load_tool_output(parent, "captured").await.unwrap(),
            Some("parent data".to_string())
        );
    }

    #[tokio::test]
    async fn test_write_to_unlisted_name_succeeds_without_touching_parent() {
        // Even when the parent has a same-named entry, if it isn't on the sub-agent's inherit
        // allowlist the child still writes its own independent row. (The block fires only on names
        // the parent actually granted; otherwise child sessions stay free to use any name.)
        // Parent's row is untouched.
        let manager = test_manager().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(parent, None, None)
            .await
            .expect("child")
            .0;

        manager
            .save_tool_output(parent, "shared", "parent original")
            .await
            .expect("seed");

        let write_tool = ScratchpadWriteTool {
            session_manager: manager.clone(),
            session_id: test_session_id(child),
            inherited_names: Vec::new(),
        };
        write_tool
            .execute(
                serde_json::json!({"name": "shared", "content": "child override"}),
                CancellationToken::new(),
            )
            .await
            .expect("write");

        // Parent copy untouched.
        assert_eq!(
            manager
                .load_tool_output(parent, "shared")
                .await
                .expect("load parent"),
            Some("parent original".to_string())
        );
        // Child copy now holds its own version.
        assert_eq!(
            manager
                .load_tool_output(child, "shared")
                .await
                .expect("load child"),
            Some("child override".to_string())
        );
    }

    // -- session isolation --

    #[tokio::test]
    async fn test_sessions_have_independent_scratchpads() {
        let manager = test_manager().await;
        let session1 = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let session2 = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_tool_output(session1, "shared_name", "session1 data")
            .await
            .expect("save");
        manager
            .save_tool_output(session2, "shared_name", "session2 data")
            .await
            .expect("save");

        let loaded1 = manager
            .load_tool_output(session1, "shared_name")
            .await
            .expect("load");
        let loaded2 = manager
            .load_tool_output(session2, "shared_name")
            .await
            .expect("load");

        assert_eq!(loaded1, Some("session1 data".to_string()));
        assert_eq!(loaded2, Some("session2 data".to_string()));
    }

    // -- session lifecycle --

    #[tokio::test]
    async fn test_delete_session_removes_scratchpad() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_tool_output(session_id, "data", "content")
            .await
            .expect("save");

        manager.delete_session(session_id).await.expect("delete");

        let outputs = manager
            .load_all_tool_outputs(session_id)
            .await
            .expect("load");
        assert!(outputs.is_empty());
    }

    #[tokio::test]
    async fn test_clear_messages_removes_scratchpad() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_tool_output(session_id, "data", "content")
            .await
            .expect("save");

        manager.clear_messages(session_id).await.expect("clear");

        let outputs = manager
            .load_all_tool_outputs(session_id)
            .await
            .expect("load");
        assert!(outputs.is_empty());
        assert!(manager.session_exists(session_id).await.expect("exists"));
    }

    // -- integration --

    #[tokio::test]
    async fn test_write_edit_read_list_delete_integration() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let sid = test_session_id(session_id);

        let write_tool = ScratchpadWriteTool {
            session_manager: manager.clone(),
            session_id: sid.clone(),
            inherited_names: Vec::new(),
        };
        write_tool
            .execute(
                serde_json::json!({"name": "test", "content": "hello world"}),
                CancellationToken::new(),
            )
            .await
            .expect("write");

        let edit_tool = ScratchpadEditTool {
            session_manager: manager.clone(),
            session_id: sid.clone(),
            inherited_names: Vec::new(),
        };
        edit_tool
            .execute(
                serde_json::json!({"name": "test", "old_string": "world", "new_string": "rust"}),
                CancellationToken::new(),
            )
            .await
            .expect("edit");

        let read_tool = ScratchpadReadTool {
            session_manager: manager.clone(),
            session_id: sid.clone(),
            parent_session_id: None,
            inherited_names: Vec::new(),
        };
        let result = read_tool
            .execute(
                serde_json::json!({"name": "test"}),
                CancellationToken::new(),
            )
            .await
            .expect("read");
        assert!(text_content(&result).contains("hello rust"));

        let list_tool = ScratchpadListTool {
            session_manager: manager.clone(),
            session_id: sid.clone(),
            parent_session_id: None,
            inherited_names: Vec::new(),
        };
        let result = list_tool
            .execute(serde_json::json!({}), CancellationToken::new())
            .await
            .expect("list");
        assert!(text_content(&result).contains("1 entries total"));

        let delete_tool = ScratchpadDeleteTool {
            session_manager: manager.clone(),
            session_id: sid.clone(),
            inherited_names: Vec::new(),
        };
        delete_tool
            .execute(
                serde_json::json!({"name": "test"}),
                CancellationToken::new(),
            )
            .await
            .expect("delete");

        let result = list_tool
            .execute(serde_json::json!({}), CancellationToken::new())
            .await
            .expect("list");
        assert!(text_content(&result).contains("empty"));
    }

    // -- scratchpad_load_file / scratchpad_save_file --

    #[tokio::test]
    async fn test_scratchpad_load_file_happy_path() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("input.txt");
        tokio::fs::write(&path, "hello scratchpad")
            .await
            .expect("write input");

        let tool = ScratchpadLoadFileTool {
            cwd: crate::workspace::test_cwd(),
            session_manager: manager.clone(),
            session_id: test_session_id(session_id),
            inherited_names: Vec::new(),
        };
        let result = tool
            .execute(
                serde_json::json!({"path": path.to_str().unwrap(), "name": "loaded"}),
                CancellationToken::new(),
            )
            .await
            .expect("load");

        assert!(!result.is_error, "got: {}", text_content(&result));
        assert!(text_content(&result).contains("Loaded 16 bytes"));
        let stored = manager
            .load_tool_output(session_id, "loaded")
            .await
            .expect("load_tool_output");
        assert_eq!(stored.as_deref(), Some("hello scratchpad"));
    }

    #[tokio::test]
    async fn test_scratchpad_load_file_resolves_relative_path_against_cwd() {
        // Regression: a relative path must resolve against the session cwd (like `read_file`), not
        // the process cwd, so `scratchpad_load_file` tracks `/cd`.
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let dir = tempfile::tempdir().expect("tempdir");
        tokio::fs::write(dir.path().join("rel.txt"), "relative contents")
            .await
            .expect("write input");
        let cwd: crate::workspace::SharedCwd =
            std::sync::Arc::new(std::sync::RwLock::new(dir.path().to_path_buf()));

        let tool = ScratchpadLoadFileTool {
            cwd,
            session_manager: manager.clone(),
            session_id: test_session_id(session_id),
            inherited_names: Vec::new(),
        };
        let result = tool
            .execute(
                serde_json::json!({"path": "rel.txt", "name": "loaded"}),
                CancellationToken::new(),
            )
            .await
            .expect("load");

        assert!(!result.is_error, "got: {}", text_content(&result));
        assert_eq!(
            manager
                .load_tool_output(session_id, "loaded")
                .await
                .expect("load_tool_output")
                .as_deref(),
            Some("relative contents"),
        );
    }

    #[tokio::test]
    async fn test_scratchpad_save_file_resolves_relative_path_against_cwd() {
        // Regression: a relative save path must land under the session cwd, not the process cwd.
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        manager
            .save_tool_output(session_id, "report", "final analysis")
            .await
            .expect("seed");
        let dir = tempfile::tempdir().expect("tempdir");
        let cwd: crate::workspace::SharedCwd =
            std::sync::Arc::new(std::sync::RwLock::new(dir.path().to_path_buf()));

        let tool = ScratchpadSaveFileTool {
            read_tracker: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
            scope: crate::workspace::WriteScope::unconfined(),
            cwd,
            session_manager: manager,
            session_id: test_session_id(session_id),
            parent_session_id: None,
            inherited_names: Vec::new(),
        };
        let result = tool
            .execute(
                serde_json::json!({"name": "report", "path": "out.txt"}),
                CancellationToken::new(),
            )
            .await
            .expect("save");

        assert!(!result.is_error, "got: {}", text_content(&result));
        let written = tokio::fs::read_to_string(dir.path().join("out.txt"))
            .await
            .expect("read back");
        assert_eq!(written, "final analysis");
    }

    /// `scratchpad_save_file` has to take the same per-path write lock `write_file` does.
    ///
    /// It carried its own copy of the path resolution and then called `write_file_bytes` directly,
    /// so the two shared no lock at all. Both are dispatched concurrently from one assistant
    /// message and both write through a temp file derived from the target, so two calls naming one
    /// path could interleave their write-then-rename and publish a spliced file. Asserted by
    /// holding the file tool's lock and showing the save cannot proceed behind it.
    #[tokio::test]
    async fn test_scratchpad_save_file_contends_with_write_file_for_the_same_path() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        manager
            .save_tool_output(session_id, "report", "final analysis")
            .await
            .expect("seed");
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("out.txt");
        tokio::fs::write(&target, "existing")
            .await
            .expect("seed target");

        let tool = ScratchpadSaveFileTool {
            read_tracker: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
            scope: crate::workspace::WriteScope::unconfined(),
            cwd: std::sync::Arc::new(std::sync::RwLock::new(dir.path().to_path_buf())),
            session_manager: manager,
            session_id: test_session_id(session_id),
            parent_session_id: None,
            inherited_names: Vec::new(),
        };

        // Whoever holds it, holds it against both tools: this is the lock `write_file` takes.
        let (_, held) = super::super::file::resolve_write_target(
            "write_file",
            &tool.cwd,
            &crate::workspace::WriteScope::unconfined(),
            target.to_str().expect("path"),
        )
        .await
        .expect("take the write lock");

        let blocked = tokio::time::timeout(
            std::time::Duration::from_millis(250),
            tool.execute(
                serde_json::json!({"name": "report", "path": "out.txt"}),
                CancellationToken::new(),
            ),
        )
        .await;
        assert!(
            blocked.is_err(),
            "the save must wait behind a write_file holding the same path",
        );

        drop(held);
        // `force`, because the fixture seeded `out.txt` before taking the lock and saving over an
        // existing file is refused without it. What this test is about is the lock, not the
        // clobber.
        let result = tool
            .execute(
                serde_json::json!({"name": "report", "path": "out.txt", "force": true}),
                CancellationToken::new(),
            )
            .await
            .expect("save once the lock is free");
        assert!(!result.is_error, "got: {}", text_content(&result));
        assert_eq!(
            tokio::fs::read_to_string(&target).await.expect("read back"),
            "final analysis",
        );
    }

    /// Saving over a file that already exists is refused, and the confirmation says what it
    /// replaced when `force` allows it.
    ///
    /// This tool reads as the scratchpad's `write_file` and its description says so, but it had
    /// none of that tool's protections: no read of the target, so no staleness check to fail; no
    /// `force`, so nothing to consult; and a confirmation naming the bytes written while never
    /// mentioning the bytes replaced. A path the model named by mistake was gone with a success
    /// message on top of it.
    #[tokio::test]
    async fn test_scratchpad_save_file_refuses_to_replace_an_existing_file() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        manager
            .save_tool_output(session_id, "report", "scratchpad bytes")
            .await
            .expect("seed");
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("out.txt");
        tokio::fs::write(&target, "THE USER WROTE THIS")
            .await
            .expect("seed target");

        let tool = ScratchpadSaveFileTool {
            read_tracker: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
            scope: crate::workspace::WriteScope::unconfined(),
            cwd: std::sync::Arc::new(std::sync::RwLock::new(dir.path().to_path_buf())),
            session_manager: manager,
            session_id: test_session_id(session_id),
            parent_session_id: None,
            inherited_names: Vec::new(),
        };

        let refused = tool
            .execute(
                serde_json::json!({"name": "report", "path": "out.txt"}),
                CancellationToken::new(),
            )
            .await
            .expect("a refusal is a result, not an error");
        assert!(refused.is_error, "got: {}", text_content(&refused));
        assert!(
            text_content(&refused).contains("already exists"),
            "got: {}",
            text_content(&refused)
        );
        assert_eq!(
            tokio::fs::read_to_string(&target).await.expect("read"),
            "THE USER WROTE THIS",
            "the file must survive the refusal"
        );

        let forced = tool
            .execute(
                serde_json::json!({"name": "report", "path": "out.txt", "force": true}),
                CancellationToken::new(),
            )
            .await
            .expect("force writes");
        assert!(!forced.is_error, "got: {}", text_content(&forced));
        assert!(
            text_content(&forced).contains("replacing 19 bytes"),
            "the confirmation must say what it replaced: {}",
            text_content(&forced)
        );
    }

    #[tokio::test]
    async fn test_scratchpad_load_file_rejects_inherited_name() {
        let manager = test_manager().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(parent, None, None)
            .await
            .expect("child")
            .0;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("input.txt");
        tokio::fs::write(&path, "irrelevant")
            .await
            .expect("write input");

        let tool = ScratchpadLoadFileTool {
            cwd: crate::workspace::test_cwd(),
            session_manager: manager.clone(),
            session_id: test_session_id(child),
            inherited_names: vec!["captured".to_string()],
        };
        let result = tool
            .execute(
                serde_json::json!({"path": path.to_str().unwrap(), "name": "captured"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        assert!(result.is_error);
        assert!(text_content(&result).contains("inherited read-only"));
        assert_eq!(
            manager
                .load_tool_output(child, "captured")
                .await
                .expect("load"),
            None,
            "no child shadow row should be created"
        );
    }

    #[tokio::test]
    async fn test_scratchpad_load_file_rejects_image_with_mime() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("pic.png");
        // Minimal PNG signature + IHDR chunk bytes, enough for `infer` to fingerprint without
        // needing a syntactically valid image.
        let png_bytes: &[u8] = &[
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk header
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1 dimensions
            0x08, 0x00, 0x00, 0x00, 0x00, // depth, type, etc.
            0xFF, 0xFE, 0xFD, 0xFC, // CRC placeholder + body, non-UTF-8
        ];
        tokio::fs::write(&path, png_bytes).await.expect("write png");

        let tool = ScratchpadLoadFileTool {
            cwd: crate::workspace::test_cwd(),
            session_manager: manager.clone(),
            session_id: test_session_id(session_id),
            inherited_names: Vec::new(),
        };
        let result = tool
            .execute(
                serde_json::json!({"path": path.to_str().unwrap(), "name": "img"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        let text = text_content(&result);
        assert!(result.is_error, "got: {}", text);
        assert!(text.contains("not valid UTF-8"), "msg: {}", text);
        assert!(text.contains("image/png"), "msg: {}", text);
        assert_eq!(
            manager.load_tool_output(session_id, "img").await.unwrap(),
            None,
            "binary file must not produce a scratchpad row"
        );
    }

    #[tokio::test]
    async fn test_scratchpad_load_file_unknown_binary_has_no_mime_line() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("blob.bin");
        // A short run of 0xFF bytes won't match any `infer` signature.
        tokio::fs::write(&path, &[0xFF_u8; 8])
            .await
            .expect("write blob");

        let tool = ScratchpadLoadFileTool {
            cwd: crate::workspace::test_cwd(),
            session_manager: manager.clone(),
            session_id: test_session_id(session_id),
            inherited_names: Vec::new(),
        };
        let result = tool
            .execute(
                serde_json::json!({"path": path.to_str().unwrap(), "name": "blob"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        let text = text_content(&result);
        assert!(result.is_error);
        assert!(text.contains("not valid UTF-8"));
        assert!(
            !text.contains("Detected MIME"),
            "no MIME line expected for unknown binary, msg: {}",
            text,
        );
    }

    #[tokio::test]
    async fn test_scratchpad_save_file_happy_path() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        manager
            .save_tool_output(session_id, "report", "final analysis")
            .await
            .expect("seed");

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("subdir").join("out.txt");

        let tool = ScratchpadSaveFileTool {
            read_tracker: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
            scope: crate::workspace::WriteScope::unconfined(),
            cwd: crate::workspace::test_cwd(),
            session_manager: manager,
            session_id: test_session_id(session_id),
            parent_session_id: None,
            inherited_names: Vec::new(),
        };
        let result = tool
            .execute(
                serde_json::json!({"name": "report", "path": path.to_str().unwrap()}),
                CancellationToken::new(),
            )
            .await
            .expect("save");

        assert!(!result.is_error, "got: {}", text_content(&result));
        let written = tokio::fs::read_to_string(&path).await.expect("read back");
        assert_eq!(written, "final analysis");
    }

    #[tokio::test]
    async fn test_scratchpad_save_file_reads_inherited_from_parent() {
        let manager = test_manager().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(parent, None, None)
            .await
            .expect("child")
            .0;
        manager
            .save_tool_output(parent, "build_log", "parent-only payload")
            .await
            .expect("seed");

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("log.txt");

        let tool = ScratchpadSaveFileTool {
            read_tracker: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
            scope: crate::workspace::WriteScope::unconfined(),
            cwd: crate::workspace::test_cwd(),
            session_manager: manager,
            session_id: test_session_id(child),
            parent_session_id: Some(parent),
            inherited_names: vec!["build_log".to_string()],
        };
        let result = tool
            .execute(
                serde_json::json!({"name": "build_log", "path": path.to_str().unwrap()}),
                CancellationToken::new(),
            )
            .await
            .expect("save");

        assert!(!result.is_error, "got: {}", text_content(&result));
        let written = tokio::fs::read_to_string(&path).await.expect("read back");
        assert_eq!(written, "parent-only payload");
    }

    #[tokio::test]
    async fn test_scratchpad_save_file_missing_entry_errors() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("out.txt");

        let tool = ScratchpadSaveFileTool {
            read_tracker: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
            scope: crate::workspace::WriteScope::unconfined(),
            cwd: crate::workspace::test_cwd(),
            session_manager: manager,
            session_id: test_session_id(session_id),
            parent_session_id: None,
            inherited_names: Vec::new(),
        };
        let result = tool
            .execute(
                serde_json::json!({"name": "missing", "path": path.to_str().unwrap()}),
                CancellationToken::new(),
            )
            .await;
        assert!(result.is_err(), "missing entry should propagate an error");
    }

    // -- scratchpad_rename --

    #[tokio::test]
    async fn test_scratchpad_rename_happy_path() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        manager
            .save_tool_output(session_id, "draft", "payload")
            .await
            .expect("seed");

        let tool = ScratchpadRenameTool {
            session_manager: manager.clone(),
            session_id: test_session_id(session_id),
            inherited_names: Vec::new(),
        };
        let result = tool
            .execute(
                serde_json::json!({"old": "draft", "new": "final"}),
                CancellationToken::new(),
            )
            .await
            .expect("rename");

        assert!(!result.is_error, "got: {}", text_content(&result));
        assert_eq!(
            manager.load_tool_output(session_id, "draft").await.unwrap(),
            None,
            "old name must be gone after rename"
        );
        assert_eq!(
            manager.load_tool_output(session_id, "final").await.unwrap(),
            Some("payload".to_string()),
        );
    }

    #[tokio::test]
    async fn test_scratchpad_rename_source_not_found() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let tool = ScratchpadRenameTool {
            session_manager: manager.clone(),
            session_id: test_session_id(session_id),
            inherited_names: Vec::new(),
        };
        let result = tool
            .execute(
                serde_json::json!({"old": "absent", "new": "whatever"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        assert!(result.is_error);
        assert!(text_content(&result).contains("not found"));
        // No row should appear under the target name.
        assert_eq!(
            manager
                .load_tool_output(session_id, "whatever")
                .await
                .unwrap(),
            None,
        );
    }

    #[tokio::test]
    async fn test_scratchpad_rename_target_already_exists() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        manager
            .save_tool_output(session_id, "src", "src-content")
            .await
            .expect("seed src");
        manager
            .save_tool_output(session_id, "dst", "dst-content")
            .await
            .expect("seed dst");

        let tool = ScratchpadRenameTool {
            session_manager: manager.clone(),
            session_id: test_session_id(session_id),
            inherited_names: Vec::new(),
        };
        let result = tool
            .execute(
                serde_json::json!({"old": "src", "new": "dst"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        assert!(result.is_error);
        assert!(text_content(&result).contains("already exists"));
        // Both rows must be untouched.
        assert_eq!(
            manager.load_tool_output(session_id, "src").await.unwrap(),
            Some("src-content".to_string()),
        );
        assert_eq!(
            manager.load_tool_output(session_id, "dst").await.unwrap(),
            Some("dst-content".to_string()),
        );
    }

    #[tokio::test]
    async fn test_scratchpad_rename_blocks_inherited_source() {
        let manager = test_manager().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(parent, None, None)
            .await
            .expect("child")
            .0;
        manager
            .save_tool_output(parent, "captured", "parent-data")
            .await
            .expect("seed");

        let tool = ScratchpadRenameTool {
            session_manager: manager.clone(),
            session_id: test_session_id(child),
            inherited_names: vec!["captured".to_string()],
        };
        let result = tool
            .execute(
                serde_json::json!({"old": "captured", "new": "mine"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        assert!(result.is_error);
        assert!(text_content(&result).contains("inherited read-only"));
        // Parent's row stays intact, child has no shadow under either name.
        assert_eq!(
            manager.load_tool_output(parent, "captured").await.unwrap(),
            Some("parent-data".to_string()),
        );
        assert_eq!(manager.load_tool_output(child, "mine").await.unwrap(), None,);
    }

    #[tokio::test]
    async fn test_scratchpad_rename_blocks_inherited_target() {
        let manager = test_manager().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(parent, None, None)
            .await
            .expect("child")
            .0;
        manager
            .save_tool_output(child, "mine", "child-data")
            .await
            .expect("seed");

        let tool = ScratchpadRenameTool {
            session_manager: manager.clone(),
            session_id: test_session_id(child),
            inherited_names: vec!["captured".to_string()],
        };
        let result = tool
            .execute(
                serde_json::json!({"old": "mine", "new": "captured"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        assert!(result.is_error);
        assert!(text_content(&result).contains("inherited read-only"));
        // Child's source row must stay intact, no shadow under inherited name.
        assert_eq!(
            manager.load_tool_output(child, "mine").await.unwrap(),
            Some("child-data".to_string()),
        );
        assert_eq!(
            manager.load_tool_output(child, "captured").await.unwrap(),
            None,
        );
    }

    // -- scratchpad_merge --

    #[tokio::test]
    async fn test_scratchpad_merge_concat_with_headers_default() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        for (name, body) in [("a", "first"), ("b", "second"), ("c", "third")] {
            manager
                .save_tool_output(session_id, name, body)
                .await
                .expect("seed");
        }

        let tool = ScratchpadMergeTool {
            session_manager: manager.clone(),
            session_id: test_session_id(session_id),
            parent_session_id: None,
            inherited_names: Vec::new(),
        };
        let result = tool
            .execute(
                serde_json::json!({"sources": ["a", "b", "c"], "target": "merged"}),
                CancellationToken::new(),
            )
            .await
            .expect("merge");

        assert!(!result.is_error, "got: {}", text_content(&result));
        let stored = manager
            .load_tool_output(session_id, "merged")
            .await
            .expect("load")
            .expect("present");
        assert!(stored.contains("--- a ---"));
        assert!(stored.contains("--- b ---"));
        assert!(stored.contains("--- c ---"));
        assert!(stored.contains("first"));
        assert!(stored.contains("second"));
        assert!(stored.contains("third"));
    }

    #[tokio::test]
    async fn test_scratchpad_merge_json_array_parses_valid_and_quotes_invalid() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        manager
            .save_tool_output(session_id, "obj", r#"{"k":1}"#)
            .await
            .expect("seed obj");
        manager
            .save_tool_output(session_id, "num", "42")
            .await
            .expect("seed num");
        manager
            .save_tool_output(session_id, "plain", "not json")
            .await
            .expect("seed plain");

        let tool = ScratchpadMergeTool {
            session_manager: manager.clone(),
            session_id: test_session_id(session_id),
            parent_session_id: None,
            inherited_names: Vec::new(),
        };
        tool.execute(
            serde_json::json!({
                "sources": ["obj", "num", "plain"],
                "target": "combined",
                "format": "json_array"
            }),
            CancellationToken::new(),
        )
        .await
        .expect("merge");

        let stored = manager
            .load_tool_output(session_id, "combined")
            .await
            .expect("load")
            .expect("present");
        let parsed: serde_json::Value = serde_json::from_str(&stored).expect("valid JSON");
        let array = parsed.as_array().expect("array");
        assert_eq!(array.len(), 3);
        assert_eq!(array[0]["k"], serde_json::json!(1));
        assert_eq!(array[1], serde_json::json!(42));
        assert_eq!(array[2], serde_json::json!("not json"));
    }

    #[tokio::test]
    async fn test_scratchpad_merge_blocks_inherited_target() {
        let manager = test_manager().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(parent, None, None)
            .await
            .expect("child")
            .0;
        manager
            .save_tool_output(child, "src", "data")
            .await
            .expect("seed");

        let tool = ScratchpadMergeTool {
            session_manager: manager.clone(),
            session_id: test_session_id(child),
            parent_session_id: Some(parent),
            inherited_names: vec!["shadow".to_string()],
        };
        let result = tool
            .execute(
                serde_json::json!({"sources": ["src"], "target": "shadow"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");
        assert!(result.is_error);
        assert!(text_content(&result).contains("inherited read-only"));
        assert_eq!(
            manager.load_tool_output(child, "shadow").await.unwrap(),
            None,
        );
    }

    #[tokio::test]
    async fn test_scratchpad_merge_reads_inherited_sources() {
        let manager = test_manager().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(parent, None, None)
            .await
            .expect("child")
            .0;
        manager
            .save_tool_output(parent, "shared", "parent payload")
            .await
            .expect("seed parent");
        manager
            .save_tool_output(child, "mine", "child payload")
            .await
            .expect("seed child");

        let tool = ScratchpadMergeTool {
            session_manager: manager.clone(),
            session_id: test_session_id(child),
            parent_session_id: Some(parent),
            inherited_names: vec!["shared".to_string()],
        };
        tool.execute(
            serde_json::json!({"sources": ["shared", "mine"], "target": "combined"}),
            CancellationToken::new(),
        )
        .await
        .expect("merge");

        let stored = manager
            .load_tool_output(child, "combined")
            .await
            .expect("load")
            .expect("present");
        assert!(stored.contains("parent payload"));
        assert!(stored.contains("child payload"));
    }

    #[tokio::test]
    async fn test_scratchpad_merge_missing_source_aborts_without_writing() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        manager
            .save_tool_output(session_id, "real", "stuff")
            .await
            .expect("seed");

        let tool = ScratchpadMergeTool {
            session_manager: manager.clone(),
            session_id: test_session_id(session_id),
            parent_session_id: None,
            inherited_names: Vec::new(),
        };
        let result = tool
            .execute(
                serde_json::json!({"sources": ["real", "missing"], "target": "out"}),
                CancellationToken::new(),
            )
            .await;
        assert!(result.is_err(), "missing source must propagate an error");
        // Target must not have been written.
        assert_eq!(
            manager.load_tool_output(session_id, "out").await.unwrap(),
            None,
            "target row must not exist after a failed merge",
        );
    }
}
