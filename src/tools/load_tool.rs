//! `load_tool` meta-tool: makes a deferred tool's full schema visible to the model on subsequent
//! turns. The active tool set is derived by scanning the conversation for successful `load_tool`
//! calls ([`super::extract_loaded_tool_names`]); this tool's `execute` only renders the description
//! and schema as `tool_result` text. It never mutates the registry.

use std::{
    collections::HashSet,
    sync::{Arc, RwLock, Weak},
};

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::{LOAD_TOOL_NAME, Tool, ToolOutput, util::require_str};
use crate::{error::Result, permission::Permission, provider::ToolDefinition};

/// Meta-tool that makes a deferred tool's schema visible for use. Held by the
/// [`super::ToolRegistry`] like any other tool, so the same `Arc` lifecycle applies. The `Weak`
/// handles avoid a self-referential cycle (registry → `Arc<dyn Tool>` → `Arc<RwLock<…>>` →
/// registry).
pub(super) struct LoadToolTool {
    pub(super) tools: Weak<RwLock<Vec<std::sync::Arc<dyn Tool>>>>,
    pub(super) deferred: Weak<RwLock<HashSet<String>>>,
    /// Filled once the registry is attached to an MCP manager. Lets an unfindable name be
    /// explained by its server's state instead of reported as unknown; a server that never
    /// connected registers no tools, so `load_tool` is the first place the agent hears about it.
    pub(super) mcp_manager: Weak<std::sync::OnceLock<Weak<crate::mcp::McpClientManager>>>,
}

impl LoadToolTool {
    /// The state of the MCP server behind `name`, when the name is unfindable *because* its
    /// server isn't connected. `None` for every other reason, leaving the generic message.
    async fn unavailable_server_reason(&self, name: &str) -> Option<String> {
        let slot = self.mcp_manager.upgrade()?;
        let manager = slot.get()?.upgrade()?;
        manager.unavailable_tool_reason(name).await
    }

    /// `" Did you mean …?"` against the registry, for a name that resolved to nothing. Takes the
    /// already-upgraded handle so the caller's read lock discipline stays in one place.
    fn near_miss_hint(
        &self,
        name: &str,
        tools: &std::sync::Arc<RwLock<Vec<Arc<dyn Tool>>>>,
    ) -> String {
        let registered: Vec<String> = tools
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .map(|tool| tool.definition().name)
            .collect();
        crate::tools::did_you_mean_hint(name, registered.iter().map(String::as_str))
    }
}

#[async_trait]
impl Tool for LoadToolTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: LOAD_TOOL_NAME.to_string(),
            description: "Load the full schema for one or more deferred tools listed \
                          under `[Tool discovery]` in the conversation context. After a \
                          successful call, each tool's full schema becomes available on \
                          your next turn. Invoke the tools by name as usual. Pass exact \
                          tool names (e.g. `mcp__notion__fetch`), either one as a string \
                          or several as an array."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": ["string", "array"],
                        "items": {"type": "string"},
                        "description": format!(
                            "Exact name of the tool to load, or an array of up to {} names",
                            crate::tools::MAX_LOAD_TOOL_BATCH,
                        ),
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
        let names = crate::tools::load_tool_names(&input);
        if names.is_empty() {
            // Re-run the scalar extraction purely to raise its error, so a missing or non-string
            // `name` reports the same way it always has.
            require_str(&input, "name", LOAD_TOOL_NAME)?;
        }

        let Some(tools) = self.tools.upgrade() else {
            return Ok(ToolOutput::text(
                "Error: tool registry is no longer available.".to_string(),
                true,
            ));
        };

        let mut sections: Vec<String> = Vec::new();
        let mut resolved = 0usize;
        for name in &names {
            let definition = {
                let guard = tools
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                guard
                    .iter()
                    .find(|t| t.definition().name == *name)
                    .map(|t| t.definition())
            };

            let Some(definition) = definition else {
                sections.push(match self.unavailable_server_reason(name).await {
                    Some(reason) => reason,
                    None => format!(
                        "Error: tool '{}' is not registered.{} Check the names listed under \
                         `[Tool discovery]` in the conversation context.",
                        name,
                        self.near_miss_hint(name, &tools),
                    ),
                });
                continue;
            };

            // Tools that aren't deferred are already part of the active tool set. Treat this as a
            // no-op success so the scanner harmlessly records the name (it was already there). The
            // model gets a clear hint to call the tool directly next time without an extra round
            // trip.
            let is_deferred = self
                .deferred
                .upgrade()
                .map(|d| {
                    d.read()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .contains(name)
                })
                .unwrap_or(false);

            resolved += 1;
            if !is_deferred {
                sections.push(format!(
                    "Tool '{}' is already available. Call it directly.",
                    name
                ));
                continue;
            }

            let schema = serde_json::to_string_pretty(&definition.parameters)
                .unwrap_or_else(|_| definition.parameters.to_string());
            sections.push(format!(
                "# {}\n\n{}\n\n## Schema\n\n```json\n{}\n```",
                name, definition.description, schema,
            ));
        }

        let plural = if names.len() == 1 {
            "schema is"
        } else {
            "schemas are"
        };
        // Say so when the cap bit. Loading 10 of 15 while reporting success would leave the model
        // believing it holds five schemas it has never seen, which is the exact failure this tool's
        // advisories exist to prevent.
        let dropped = crate::tools::requested_tool_names(&input)
            .len()
            .saturating_sub(names.len());
        let capped = if dropped > 0 {
            format!(
                " Only the first {} names were loaded; {} more were not. Call `load_tool` again \
                 for those.",
                crate::tools::MAX_LOAD_TOOL_BATCH,
                dropped,
            )
        } else {
            String::new()
        };
        // Only claimed when something actually loaded. The trailer used to be appended
        // unconditionally, so a call naming one unregistered tool came back as an error followed
        // by "The full schema is now available on your next turn" -- a flat contradiction, and the
        // one sentence a model reads to decide whether to call the tool. Found by a sub-agent that
        // tried to load `memory_write` it had not been granted, and said the line was misleading.
        let body = if resolved == 0 {
            sections.join("\n\n---\n\n")
        } else {
            format!(
                "{}\n\nThe full {} now available on your next turn. Call the tools directly with \
                 the parameters above.{}",
                sections.join("\n\n---\n\n"),
                plural,
                capped,
            )
        };
        // Errors are reported per name, but the call only *fails* when nothing resolved: a batch
        // that loaded three of four tools must stay non-error so the three are recorded as active.
        Ok(ToolOutput::text(body, resolved == 0))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::provider::ContentBlock;

    /// Minimal fake tool for testing the registry-lookup paths of `LoadToolTool` without dragging
    /// in `ToolRegistry::build_default`.
    struct FakeTool {
        name: String,
        description: String,
        schema: serde_json::Value,
    }

    #[async_trait]
    impl Tool for FakeTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: self.name.clone(),
                description: self.description.clone(),
                parameters: self.schema.clone(),
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
        ) -> Result<crate::tools::ToolOutput> {
            Ok(crate::tools::ToolOutput::text(String::new(), false))
        }
    }

    type ToolStorage = Arc<RwLock<Vec<Arc<dyn Tool>>>>;
    type DeferredStorage = Arc<RwLock<HashSet<String>>>;

    /// Test fixture: holds the strong `Arc`s for `tools` and `deferred` so the `Weak`s inside
    /// `LoadToolTool` stay live for the duration of a test. `take()` either field to simulate
    /// registry teardown.
    struct Fixture {
        tools: Option<ToolStorage>,
        deferred: Option<DeferredStorage>,
        load_tool: LoadToolTool,
    }

    fn fake_tool(name: &str) -> Arc<dyn Tool> {
        Arc::new(FakeTool {
            name: name.to_string(),
            description: format!("Fixture tool {}.", name),
            schema: serde_json::json!({
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Page URL"}},
                "required": ["url"]
            }),
        }) as Arc<dyn Tool>
    }

    fn build_test_tool(registered: Vec<Arc<dyn Tool>>, deferred_names: &[&str]) -> Fixture {
        let tools: ToolStorage = Arc::new(RwLock::new(registered));
        let deferred: DeferredStorage = Arc::new(RwLock::new(
            deferred_names.iter().map(|n| n.to_string()).collect(),
        ));
        let load_tool = LoadToolTool {
            tools: Arc::downgrade(&tools),
            deferred: Arc::downgrade(&deferred),
            // No manager attached: these fixtures exercise the plain registry paths, so an
            // unfindable name must still produce the generic "not registered" message.
            mcp_manager: std::sync::Weak::new(),
        };
        Fixture {
            tools: Some(tools),
            deferred: Some(deferred),
            load_tool,
        }
    }

    #[tokio::test]
    async fn test_load_tool_unknown_name() {
        let fixture = build_test_tool(Vec::new(), &[]);
        let load_tool = &fixture.load_tool;
        let result = load_tool
            .execute(
                serde_json::json!({"name": "nonexistent"}),
                CancellationToken::new(),
            )
            .await
            .expect("should return Ok");
        assert!(result.is_error);
        let text = ContentBlock::tool_result_text_content(&result.content);
        assert!(text.contains("not registered"));
        assert!(text.contains("[Tool discovery]"));
        // And it must not also claim the schema arrived. The trailer used to be appended whatever
        // happened, so a failed load read as an error immediately contradicted by "The full schema
        // is now available on your next turn" -- the one sentence the model uses to decide whether
        // to go ahead and call the tool.
        assert!(
            !text.contains("next turn"),
            "a load that resolved nothing must not promise a schema: {text}"
        );
    }

    #[tokio::test]
    async fn test_load_tool_missing_name_field() {
        let fixture = build_test_tool(Vec::new(), &[]);
        let load_tool = &fixture.load_tool;
        let result = load_tool
            .execute(serde_json::json!({}), CancellationToken::new())
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_load_tool_returns_schema_for_deferred_tool() {
        let fake = Arc::new(FakeTool {
            name: "mcp__notion__fetch".to_string(),
            description: "Fetch a Notion page by URL or ID.".to_string(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Page URL"}
                },
                "required": ["url"]
            }),
        }) as Arc<dyn Tool>;
        let fixture = build_test_tool(vec![fake], &["mcp__notion__fetch"]);
        let load_tool = &fixture.load_tool;

        let result = load_tool
            .execute(
                serde_json::json!({"name": "mcp__notion__fetch"}),
                CancellationToken::new(),
            )
            .await
            .expect("should return Ok");

        assert!(!result.is_error, "deferred-tool load should succeed");
        let text = ContentBlock::tool_result_text_content(&result.content);
        assert!(text.contains("mcp__notion__fetch"));
        assert!(text.contains("Fetch a Notion page"));
        assert!(text.contains("## Schema"));
        // The schema body must be the actual tool's schema, not a placeholder.
        assert!(text.contains("\"url\""));
        assert!(text.contains("\"required\""));
        assert!(text.contains("next turn"));
    }

    #[tokio::test]
    async fn test_load_tool_already_available_tool() {
        // Registered but not in the deferred set: model should be told to call it directly.
        // Returned as success so the scanner records the name harmlessly (it was already in the
        // active set).
        let fake = Arc::new(FakeTool {
            name: "read_file".to_string(),
            description: "Read a file from disk.".to_string(),
            schema: serde_json::json!({"type": "object"}),
        }) as Arc<dyn Tool>;
        let fixture = build_test_tool(vec![fake], &[]);
        let load_tool = &fixture.load_tool;

        let result = load_tool
            .execute(
                serde_json::json!({"name": "read_file"}),
                CancellationToken::new(),
            )
            .await
            .expect("should return Ok");

        assert!(!result.is_error);
        let text = ContentBlock::tool_result_text_content(&result.content);
        assert!(text.contains("already available"));
        assert!(text.contains("read_file"));
        // Must NOT render the schema block; the model already has it.
        assert!(!text.contains("## Schema"));
    }

    #[tokio::test]
    async fn test_load_tool_accepts_an_array_of_names() {
        let fixture = build_test_tool(
            vec![
                fake_tool("mcp__notion__fetch"),
                fake_tool("mcp__notion__search"),
            ],
            &["mcp__notion__fetch", "mcp__notion__search"],
        );

        let result = fixture
            .load_tool
            .execute(
                serde_json::json!({"name": ["mcp__notion__fetch", "mcp__notion__search"]}),
                CancellationToken::new(),
            )
            .await
            .expect("should return Ok");

        assert!(!result.is_error);
        let text = ContentBlock::tool_result_text_content(&result.content);
        assert!(text.contains("# mcp__notion__fetch"), "{text}");
        assert!(text.contains("# mcp__notion__search"), "{text}");
        assert!(text.contains("schemas are"), "plural wording: {text}");
    }

    /// A batch must not lose the tools that did resolve just because one name was wrong: the
    /// non-error result is what records them in the active set.
    #[tokio::test]
    async fn test_load_tool_batch_survives_one_bad_name() {
        let fixture = build_test_tool(vec![fake_tool("mcp__notion__fetch")], &[
            "mcp__notion__fetch",
        ]);

        let result = fixture
            .load_tool
            .execute(
                serde_json::json!({"name": ["mcp__notion__fetch", "nope"]}),
                CancellationToken::new(),
            )
            .await
            .expect("should return Ok");

        assert!(!result.is_error, "one resolved, so the call succeeded");
        let text = ContentBlock::tool_result_text_content(&result.content);
        assert!(text.contains("# mcp__notion__fetch"), "{text}");
        assert!(text.contains("'nope' is not registered"), "{text}");
    }

    /// Half-honouring an over-long batch while reporting plain success would leave the model
    /// believing it holds schemas it has never seen.
    #[tokio::test]
    async fn test_load_tool_reports_names_dropped_by_the_cap() {
        let names: Vec<String> = (0..crate::tools::MAX_LOAD_TOOL_BATCH + 3)
            .map(|index| format!("tool_{index}"))
            .collect();
        let registered: Vec<Arc<dyn Tool>> = names.iter().map(|name| fake_tool(name)).collect();
        let deferred: Vec<&str> = names.iter().map(String::as_str).collect();
        let fixture = build_test_tool(registered, &deferred);

        let result = fixture
            .load_tool
            .execute(
                serde_json::json!({ "name": names }),
                CancellationToken::new(),
            )
            .await
            .expect("should return Ok");

        assert!(!result.is_error);
        let text = ContentBlock::tool_result_text_content(&result.content);
        assert!(text.contains("3 more were not"), "{text}");
        assert!(!text.contains("# tool_12"), "past the cap: {text}");
    }

    /// Dropping the `mcp__<server>__` prefix is the likeliest way to get a tool name wrong, and
    /// pure edit distance would never suggest the right answer.
    #[tokio::test]
    async fn test_load_tool_suggests_the_namespaced_name() {
        let fixture = build_test_tool(vec![fake_tool("mcp__notion__fetch")], &[
            "mcp__notion__fetch",
        ]);

        let result = fixture
            .load_tool
            .execute(
                serde_json::json!({"name": "fetch"}),
                CancellationToken::new(),
            )
            .await
            .expect("should return Ok");

        assert!(result.is_error);
        let text = ContentBlock::tool_result_text_content(&result.content);
        assert!(
            text.contains("Did you mean `mcp__notion__fetch`?"),
            "{text}"
        );
    }

    #[tokio::test]
    async fn test_load_tool_registry_dropped() {
        // Simulate the registry going away while the LoadToolTool is still held somewhere. Both
        // Weak upgrades should fail gracefully, returning a plain error tool_result, not
        // panicking.
        let mut fixture = build_test_tool(Vec::new(), &[]);
        fixture.tools.take();
        fixture.deferred.take();

        let result = fixture
            .load_tool
            .execute(
                serde_json::json!({"name": "anything"}),
                CancellationToken::new(),
            )
            .await
            .expect("should return Ok with error tool_result");
        assert!(result.is_error);
        let text = ContentBlock::tool_result_text_content(&result.content);
        assert!(text.contains("no longer available"));
    }
}
