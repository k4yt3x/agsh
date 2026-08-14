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
                Writing to a name that already exists updates it, so this is also how you \
                correct or refine an existing memory, or change just its priority: omit body and \
                whatever the memory already said is kept. Do not save what is derivable from the \
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
                        "description": "Optional detail, loaded only when memory_read is called. \
                                        Omit it to leave an existing memory's body untouched; \
                                        pass an empty string to clear it"
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
        // Validated here as well as inside `write_memory`, because the existence check below joins
        // the name onto the root and an unvalidated one would stat outside it. The same guard
        // `memory_delete` applies, for the same reason.
        memory::validate_memory_name(name).map_err(|message| MekaError::ToolExecution {
            tool_name: "memory_write".to_string(),
            message,
        })?;
        let description = require_str(&input, "description", "memory_write")?;
        // An absent `body` means "leave it alone", not "make it empty". The schema has always
        // marked the field optional, so the call that changes only a priority is one the tool
        // invites -- and rendering the absence as `""` made that call silently delete everything
        // the memory said. `Some("")` is still an explicit request to clear it.
        let body = input.get("body").and_then(serde_json::Value::as_str);
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

        // Read before the write, since the write is what makes the file exist. Without this the
        // confirmation would claim to have kept the body of a memory that had none, on the very
        // call that created it.
        let kept_existing_body = body.is_none() && memory::memory_file_in(&root, name).is_file();

        let path =
            memory::write_memory(&root, name, description, priority, body).map_err(|message| {
                MekaError::ToolExecution {
                    tool_name: "memory_write".to_string(),
                    message,
                }
            })?;
        // The memory snapshot keys on mtime alone, so *any* second write inside one clock tick is
        // invisible to it. Without this the agent's own note would not be in the index it reads
        // back on the very next turn.
        self.memories.invalidate().await;

        tracing::info!("saved memory to {}", path.display());
        Ok(ToolOutput::text(
            format!(
                "Saved memory '{}' (priority {}){}. It will appear in your memory index from the \
                 next turn on.",
                name,
                priority,
                // Stated so the two calls are distinguishable from the result alone: a metadata
                // update and a rewrite otherwise report identically, and the difference is the
                // whole body of the note.
                if kept_existing_body {
                    ", keeping the existing body"
                } else {
                    ""
                }
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
        let index = self.memories.current().await;
        let entry = index
            .memories
            .iter()
            .find(|entry| entry.name == name)
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "memory_read".to_string(),
                // A name whose file exists but would not parse is reported as such. Answering "no
                // memory named x" there sends the reader off to write the memory again, when what
                // it needs to know is that its note is on disk, unread, and not in effect.
                message: match index.skip_reason(name) {
                    Some(reason) => format!(
                        "memory '{}' is on disk but could not be read: {}. Until it is fixed it is \
                         not in your index and nothing it says is in effect.",
                        name, reason
                    ),
                    None => format!("no memory named '{}'", name),
                },
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
        let index = self.memories.current().await;

        let mut matches = Vec::new();
        let mut truncated = false;
        for entry in index.memories.iter() {
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
        // Before the `is_file` check below, which follows symlinks: a link pointing at a real file
        // would pass it, and `remove_file` would then take the link and leave the target. Calling
        // that a deleted memory misreports what happened, and the link was put there deliberately.
        crate::store::reject_symlinked_path(&path, "memory").map_err(|message| {
            MekaError::ToolExecution {
                tool_name: "memory_delete".to_string(),
                message,
            }
        })?;
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
        // See the note in `memory_write`.
        self.memories.invalidate().await;

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

    fn output_text(output: &ToolOutput) -> String {
        output
            .content
            .iter()
            .map(|block| match block {
                crate::provider::ToolResultContent::Text { text } => text.clone(),
                _ => String::new(),
            })
            .collect()
    }

    /// `is_file` follows symlinks, so a link pointing at a real file passed the existence check and
    /// `remove_file` then took the link and left the target: a deleted memory that still exists.
    /// Both tools refuse instead, keeping read permission's promise that nothing outside meka's own
    /// store changes.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_write_and_delete_refuse_a_symlinked_memory() {
        let temp = tempfile::tempdir().expect("tempdir");
        let victim = temp.path().join("victim.txt");
        std::fs::write(&victim, "ORIGINAL").expect("victim");
        std::os::unix::fs::symlink(&victim, temp.path().join("evil.md")).expect("symlink");

        let write = MemoryWriteTool {
            memories: cache_at(temp.path()),
        };
        let error = write
            .execute(
                serde_json::json!({"name": "evil", "description": "d", "body": "PWNED"}),
                CancellationToken::new(),
            )
            .await
            .expect_err("must refuse to write through a symlink");
        assert!(error.to_string().contains("symlink"), "{error}");

        let delete = MemoryDeleteTool {
            memories: cache_at(temp.path()),
        };
        let error = delete
            .execute(
                serde_json::json!({"name": "evil"}),
                CancellationToken::new(),
            )
            .await
            .expect_err("must refuse to delete through a symlink");
        assert!(error.to_string().contains("symlink"), "{error}");

        assert_eq!(
            std::fs::read_to_string(&victim).expect("read"),
            "ORIGINAL",
            "the target must survive both"
        );
        assert!(
            temp.path().join("evil.md").is_symlink(),
            "the link itself must survive too"
        );
    }

    /// The reported failure, at the point it was felt: four files on disk and an agent told four
    /// times that no memory by those names existed. "Not found" sends the reader off to write the
    /// note again; what it needs to know is that its note is right there and unread.
    #[tokio::test]
    async fn test_read_explains_a_file_it_could_not_parse() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("mica-policy.md"), "no frontmatter here\n")
            .expect("write broken memory");

        let read = MemoryReadTool {
            memories: cache_at(temp.path()),
        };
        let error = read
            .execute(
                serde_json::json!({"name": "mica-policy"}),
                CancellationToken::new(),
            )
            .await
            .expect_err("an unreadable memory must not read as absent");
        let message = error.to_string();
        assert!(message.contains("missing YAML frontmatter"), "{message}");
        assert!(
            message.contains("nothing it says is in effect"),
            "{message}"
        );

        // A name with no file at all still gets the plain answer.
        let error = read
            .execute(
                serde_json::json!({"name": "never-written"}),
                CancellationToken::new(),
            )
            .await
            .expect_err("absent is still absent");
        assert!(error.to_string().contains("no memory named"), "{error}");
    }

    /// `body` has always been optional, so a priority change is a call the schema invites. It used
    /// to render the absence as an empty body and delete everything the memory said.
    #[tokio::test]
    async fn test_write_without_a_body_keeps_the_existing_one() {
        let temp = tempfile::tempdir().expect("tempdir");
        let memories = cache_at(temp.path());
        let write = MemoryWriteTool {
            memories: memories.clone(),
        };

        // Creating a memory without a body keeps nothing, and must not say it did.
        let output = write
            .execute(
                serde_json::json!({"name": "bare", "description": "No body"}),
                CancellationToken::new(),
            )
            .await
            .expect("create without a body");
        assert!(
            !output_text(&output).contains("keeping"),
            "{}",
            output_text(&output)
        );

        write
            .execute(
                serde_json::json!({
                    "name": "policy",
                    "description": "How to reply",
                    "body": "Always answer in kind."
                }),
                CancellationToken::new(),
            )
            .await
            .expect("initial write");

        let output = write
            .execute(
                serde_json::json!({
                    "name": "policy",
                    "description": "How to reply",
                    "priority": 0
                }),
                CancellationToken::new(),
            )
            .await
            .expect("metadata-only write");
        // Said out loud, so the two calls are distinguishable from their results alone.
        assert!(
            output_text(&output).contains("keeping the existing body"),
            "{}",
            output_text(&output)
        );

        let read = MemoryReadTool { memories };
        let output = read
            .execute(
                serde_json::json!({"name": "policy"}),
                CancellationToken::new(),
            )
            .await
            .expect("read");
        assert!(
            output_text(&output).contains("Always answer in kind."),
            "{}",
            output_text(&output)
        );

        // Clearing is still possible; it just has to be asked for.
        write
            .execute(
                serde_json::json!({
                    "name": "policy",
                    "description": "How to reply",
                    "body": ""
                }),
                CancellationToken::new(),
            )
            .await
            .expect("explicit clear");
        let output = read
            .execute(
                serde_json::json!({"name": "policy"}),
                CancellationToken::new(),
            )
            .await
            .expect("read");
        assert!(
            !output_text(&output).contains("Always answer in kind."),
            "{}",
            output_text(&output)
        );
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
                .memories
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
