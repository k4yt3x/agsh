//! The `memory_*` tools: the agent's read/write access to its own durable notes
//! ([`crate::memory`]).
//!
//! All four gate at [`Permission::Read`], matching `scratchpad` and `todo`: these write to a store
//! meka owns under its own config directory, not to the user's tree, and the motivating deployment
//! runs at read permission permanently. Gating them at `Write` would mean an agent that can never
//! remember anything, which defeats the feature.
//!
//! That makes [`crate::memory::validate_memory_name`] load-bearing rather than cosmetic: these are
//! the first tools in meka that touch the *filesystem* at read permission (scratchpad and todo go
//! to the database), so the name check is what keeps `memory_write` from being an arbitrary-file
//! -write primitive.

use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::{
    Tool, ToolOutput,
    util::{MAX_SEARCH_MATCHES, compile_user_regex},
};
use crate::{
    error::{MekaError, Result},
    memory::{self, MemoryCache},
    permission::Permission,
    provider::ToolDefinition,
};

/// Shared by every tool here: resolve the memory root, or fail with a message that names the cause
/// rather than reporting an empty store.
fn require_root(cache: &MemoryCache, tool_name: &str) -> Result<std::path::PathBuf> {
    cache
        .root()
        .map(|root| root.to_path_buf())
        .ok_or_else(|| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: "memory is disabled or the meka config directory could not be resolved"
                .to_string(),
        })
}

fn require_str<'a>(input: &'a serde_json::Value, key: &str, tool_name: &str) -> Result<&'a str> {
    input[key]
        .as_str()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!("missing '{}' parameter", key),
        })
}

pub(super) struct MemoryWriteTool {
    pub memories: Arc<MemoryCache>,
}

#[async_trait]
impl Tool for MemoryWriteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "memory_write".to_string(),
            description: "Save a durable note that outlives this session. Use when you learn \
                something that will still matter in a later conversation: who someone is, how \
                they want you to work, a standing decision, or where something external lives. \
                Writing to a name that already exists replaces it, so this is also how you \
                correct or refine an existing memory. Do not save what is derivable from the \
                code, git history, or the current conversation. The description is what you will \
                see in every future session, so make it stand on its own."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Identifier, letters/digits/-/_ only (e.g. 'alice-timezone')"
                    },
                    "description": {
                        "type": "string",
                        "description": "One line stating the fact itself, shown in every future \
                                        session's memory index"
                    },
                    "priority": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 9,
                        "description": "Lower sorts higher in the index. 0-1 standing directives \
                                        that always apply, 2-4 durable facts, 5 default, 6-9 \
                                        situational or short-lived"
                    },
                    "body": {
                        "type": "string",
                        "description": "Optional detail, loaded only when memory_read is called"
                    }
                },
                "required": ["name", "description"]
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
        let root = require_root(&self.memories, "memory_write")?;
        let name = require_str(&input, "name", "memory_write")?;
        let description = require_str(&input, "description", "memory_write")?;
        let body = input["body"].as_str().unwrap_or("");
        // Same clamp the frontmatter path uses, so a priority reaches the same value whether the
        // agent passed it here or a human typed it into the file. Reading it as `i64` matters:
        // `as_u64` would reject a negative outright, where the file path clamps it to 0.
        let priority = match input.get("priority") {
            Some(serde_json::Value::Null) | None => memory::DEFAULT_PRIORITY,
            Some(value) => {
                let raw = value.as_i64().ok_or_else(|| MekaError::ToolExecution {
                    tool_name: "memory_write".to_string(),
                    message: format!("'priority' must be a whole number, got {}", value),
                })?;
                memory::parse_priority(Some(raw), name)
            }
        };

        let path =
            memory::write_memory(&root, name, description, priority, body).map_err(|message| {
                MekaError::ToolExecution {
                    tool_name: "memory_write".to_string(),
                    message,
                }
            })?;

        tracing::info!("saved memory to {}", path.display());
        Ok(ToolOutput::text(
            format!(
                "Saved memory '{}' (priority {}). It will appear in your memory index from the \
                 next turn on.",
                name, priority
            ),
            false,
        ))
    }
}

pub(super) struct MemoryReadTool {
    pub memories: Arc<MemoryCache>,
}

#[async_trait]
impl Tool for MemoryReadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "memory_read".to_string(),
            description: "Load one saved memory in full. The memory index in your context lists \
                each memory's name and description; call this when a memory's description \
                suggests it holds detail you need."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the memory, as listed in the memory index"
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
        let name = require_str(&input, "name", "memory_read")?;
        let memories = self.memories.current().await;
        let entry = memories
            .iter()
            .find(|entry| entry.name == name)
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "memory_read".to_string(),
                message: format!("no memory named '{}'", name),
            })?;

        let body =
            memory::load_memory_body(entry)
                .await
                .map_err(|message| MekaError::ToolExecution {
                    tool_name: "memory_read".to_string(),
                    message,
                })?;

        // Age is stated on the way out, not just in the index: a memory is a point-in-time
        // observation, and detail that was true months ago is exactly what gets asserted as
        // current fact without a nudge.
        let age = memory::render_age(entry.mtime, std::time::SystemTime::now());
        Ok(ToolOutput::text(
            format!(
                "# {}\n\n{}\n\nSaved {}. This is what you recorded then, not live state; verify \
                 before relying on it.\n\n{}",
                entry.name,
                entry.description,
                age,
                body.trim()
            ),
            false,
        ))
    }
}

pub(super) struct MemorySearchTool {
    pub memories: Arc<MemoryCache>,
}

#[async_trait]
impl Tool for MemorySearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "memory_search".to_string(),
            description: "Search the full text of every saved memory with a regular expression, \
                including memories too old or low-priority to appear in the index. Use this when \
                the index says memories are not shown, or when you suspect you know something \
                that is not currently listed."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Rust regex matched against each line of every memory"
                    }
                },
                "required": ["pattern"]
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
        let pattern = require_str(&input, "pattern", "memory_search")?;
        let regex = compile_user_regex(pattern, "memory_search")?;
        let memories = self.memories.current().await;

        let mut matches = Vec::new();
        let mut truncated = false;
        for entry in memories.iter() {
            // Search the whole file, frontmatter included, so a hit on the description surfaces
            // even when the body says nothing.
            let content = match tokio::fs::read_to_string(&entry.path).await {
                Ok(content) => content,
                Err(error) => {
                    tracing::warn!("memory_search skipping {}: {}", entry.path.display(), error);
                    continue;
                }
            };
            for (index, line) in content.lines().enumerate() {
                if !regex.is_match(line) {
                    continue;
                }
                if matches.len() >= MAX_SEARCH_MATCHES {
                    truncated = true;
                    break;
                }
                matches.push(format!("{}:{}: {}", entry.name, index + 1, line.trim()));
            }
            if truncated {
                break;
            }
        }

        if matches.is_empty() {
            return Ok(ToolOutput::text(
                "No memories matched that pattern.".to_string(),
                false,
            ));
        }

        let mut out = matches.join("\n");
        if truncated {
            out.push_str(&format!(
                "\n\n(stopped at {} matches; narrow the pattern to see the rest)",
                MAX_SEARCH_MATCHES
            ));
        }
        Ok(ToolOutput::text(out, false))
    }
}

pub(super) struct MemoryDeleteTool {
    pub memories: Arc<MemoryCache>,
}

#[async_trait]
impl Tool for MemoryDeleteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "memory_delete".to_string(),
            description: "Delete a saved memory permanently. Use when something you recorded has \
                turned out to be wrong or no longer applies. To revise a memory rather than drop \
                it, call memory_write with the same name instead."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the memory to delete"
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
        let root = require_root(&self.memories, "memory_delete")?;
        let name = require_str(&input, "name", "memory_delete")?;
        // Validate before joining: the name reaches the filesystem here exactly as it does in
        // `memory_write`, so it needs the same guard.
        memory::validate_memory_name(name).map_err(|message| MekaError::ToolExecution {
            tool_name: "memory_delete".to_string(),
            message,
        })?;

        let path = memory::memory_file_in(&root, name);
        if !path.is_file() {
            return Err(MekaError::ToolExecution {
                tool_name: "memory_delete".to_string(),
                message: format!("no memory named '{}'", name),
            });
        }
        tokio::fs::remove_file(&path)
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "memory_delete".to_string(),
                message: format!("failed to delete {}: {}", path.display(), error),
            })?;

        tracing::info!("deleted memory {}", path.display());
        Ok(ToolOutput::text(
            format!("Deleted memory '{}'.", name),
            false,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache_at(root: &std::path::Path) -> Arc<MemoryCache> {
        MemoryCache::for_root(Some(root.to_path_buf()))
    }

    #[tokio::test]
    async fn test_write_then_read_round_trip() {
        let temp = tempfile::tempdir().expect("tempdir");
        let memories = cache_at(temp.path());

        let write = MemoryWriteTool {
            memories: memories.clone(),
        };
        write
            .execute(
                serde_json::json!({
                    "name": "alice-timezone",
                    "description": "Alice is in JST",
                    "priority": 2,
                    "body": "She works 10:00-19:00 JST."
                }),
                CancellationToken::new(),
            )
            .await
            .expect("write");

        let read = MemoryReadTool { memories };
        let output = read
            .execute(
                serde_json::json!({"name": "alice-timezone"}),
                CancellationToken::new(),
            )
            .await
            .expect("read");
        let text = format!("{:?}", output.content);
        assert!(text.contains("10:00-19:00 JST"), "{text}");
        // The freshness caveat must ride along; a recalled memory asserted as live state is the
        // failure this guards against.
        assert!(text.contains("not live state"), "{text}");
    }

    /// The tools run at `Permission::Read`, so a name that escapes the memory directory would be
    /// an arbitrary-file write available in read-only mode.
    /// A priority must land on the same value whichever door it came through. `as_u64` used to
    /// reject a negative outright while the frontmatter path clamped it to 0.
    #[tokio::test]
    async fn test_write_clamps_priority_like_the_file_path() {
        let temp = tempfile::tempdir().expect("tempdir");
        let memories = cache_at(temp.path());
        let write = MemoryWriteTool {
            memories: memories.clone(),
        };

        let cases = [
            ("low", -5, memory::MIN_PRIORITY),
            ("high", 99, memory::MAX_PRIORITY),
            ("mid", 3, 3),
        ];
        for (name, given, _) in cases {
            write
                .execute(
                    serde_json::json!({"name": name, "description": "d", "priority": given}),
                    CancellationToken::new(),
                )
                .await
                .unwrap_or_else(|error| panic!("priority {given} must be accepted: {error}"));
        }

        let found = memories.current().await;
        for (name, given, expected) in cases {
            let entry = found
                .iter()
                .find(|entry| entry.name == name)
                .unwrap_or_else(|| panic!("'{name}' must have been written"));
            assert_eq!(entry.priority, expected, "priority {given}");
        }

        // A non-integer is still a hard error rather than a silent default.
        assert!(
            write
                .execute(
                    serde_json::json!({"name": "bad", "description": "d", "priority": "high"}),
                    CancellationToken::new()
                )
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_write_rejects_path_traversal() {
        let temp = tempfile::tempdir().expect("tempdir");
        let write = MemoryWriteTool {
            memories: cache_at(temp.path()),
        };

        for bad in ["../escape", "a/b", "/abs", ".hidden"] {
            let result = write
                .execute(
                    serde_json::json!({"name": bad, "description": "x"}),
                    CancellationToken::new(),
                )
                .await;
            assert!(result.is_err(), "'{bad}' must be rejected");
        }
        assert!(
            !temp
                .path()
                .parent()
                .map(|p| p.join("escape.md").exists())
                .unwrap_or(false),
            "nothing may be written outside the memory root"
        );
    }

    #[tokio::test]
    async fn test_delete_rejects_traversal_and_missing() {
        let temp = tempfile::tempdir().expect("tempdir");
        let delete = MemoryDeleteTool {
            memories: cache_at(temp.path()),
        };
        assert!(
            delete
                .execute(
                    serde_json::json!({"name": "../escape"}),
                    CancellationToken::new()
                )
                .await
                .is_err()
        );
        assert!(
            delete
                .execute(
                    serde_json::json!({"name": "absent"}),
                    CancellationToken::new()
                )
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_search_finds_body_and_description() {
        let temp = tempfile::tempdir().expect("tempdir");
        let memories = cache_at(temp.path());
        let write = MemoryWriteTool {
            memories: memories.clone(),
        };
        write
            .execute(
                serde_json::json!({
                    "name": "deploy-host",
                    "description": "mekabridge runs on the NAS",
                    "body": "Hostname is nas.lan."
                }),
                CancellationToken::new(),
            )
            .await
            .expect("write");

        let search = MemorySearchTool { memories };
        let hit = search
            .execute(
                serde_json::json!({"pattern": "nas\\.lan"}),
                CancellationToken::new(),
            )
            .await
            .expect("search");
        assert!(format!("{:?}", hit.content).contains("deploy-host:"));

        let miss = search
            .execute(
                serde_json::json!({"pattern": "nothing-matches-this"}),
                CancellationToken::new(),
            )
            .await
            .expect("search");
        assert!(format!("{:?}", miss.content).contains("No memories matched"));
    }

    #[tokio::test]
    async fn test_all_memory_tools_gate_at_read() {
        let temp = tempfile::tempdir().expect("tempdir");
        let memories = cache_at(temp.path());
        // Read permission is what mekabridge runs at; anything stricter means an agent that can
        // never remember.
        assert_eq!(
            MemoryWriteTool {
                memories: memories.clone()
            }
            .required_permission(),
            Permission::Read
        );
        assert_eq!(
            MemoryReadTool {
                memories: memories.clone()
            }
            .required_permission(),
            Permission::Read
        );
        assert_eq!(
            MemorySearchTool {
                memories: memories.clone()
            }
            .required_permission(),
            Permission::Read
        );
        assert_eq!(
            MemoryDeleteTool { memories }.required_permission(),
            Permission::Read
        );
    }
}
