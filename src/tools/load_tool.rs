//! `load_tool` meta-tool: makes a deferred tool's full schema visible to the model on subsequent
//! turns. The active tool set is derived by scanning the conversation for successful `load_tool`
//! calls ([`super::extract_loaded_tool_names`]); this tool's `execute` only renders the description
//! and schema as `tool_result` text. It never mutates the registry.

use std::{
    collections::HashSet,
    sync::{RwLock, Weak},
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
}

#[async_trait]
impl Tool for LoadToolTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: LOAD_TOOL_NAME.to_string(),
            description: "Load the full schema for a deferred tool listed under \
                          `[Tool discovery]` in the conversation context. After a \
                          successful call, the tool's full schema becomes available on \
                          your next turn. Invoke the tool by name as usual. Pass the \
                          exact tool name (e.g. `mcp__notion__fetch`)."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Exact name of the tool to load",
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
        let name = require_str(&input, "name", LOAD_TOOL_NAME)?;

        let Some(tools) = self.tools.upgrade() else {
            return Ok(ToolOutput::text(
                "Error: tool registry is no longer available.".to_string(),
                true,
            ));
        };

        let definition = {
            let guard = tools
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard
                .iter()
                .find(|t| t.definition().name == name)
                .map(|t| t.definition())
        };

        let Some(definition) = definition else {
            if let Some(reason) = self.unavailable_server_reason(&name).await {
                return Ok(ToolOutput::text(reason, true));
            }
            return Ok(ToolOutput::text(
                format!(
                    "Error: tool '{}' is not registered. Check the names listed under \
                     `[Tool discovery]` in the conversation context.",
                    name
                ),
                true,
            ));
        };

        // Tools that aren't deferred are already part of the active tool set. Treat this as a no-op
        // success so the scanner harmlessly records the name (it was already there). The model
        // gets a clear hint to call the tool directly next time without an extra round trip.
        let is_deferred = self
            .deferred
            .upgrade()
            .map(|d| {
                d.read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .contains(&name)
            })
            .unwrap_or(false);

        if !is_deferred {
            return Ok(ToolOutput::text(
                format!("Tool '{}' is already available. Call it directly.", name),
                false,
            ));
        }

        let schema = serde_json::to_string_pretty(&definition.parameters)
            .unwrap_or_else(|_| definition.parameters.to_string());
        let body = format!(
            "# {}\n\n{}\n\n## Schema\n\n```json\n{}\n```\n\n\
             The tool's full schema is now available on your next turn. Call \
             `{}` directly with the parameters above.",
            name, definition.description, schema, name,
        );
        Ok(ToolOutput::text(body, false))
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
