//! Builtin tools exposing MCP resources and prompts to the agent:
//! `list_mcp_resources`, `read_mcp_resource`, `list_mcp_prompts`, and
//! `get_mcp_prompt`. Each tool routes through a shared [`McpClientManager`]
//! so it can target any configured server by name.

use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::{Tool, ToolOutput, util::require_str};
use crate::{
    error::{MekaError, Result},
    mcp::{MAX_MCP_DESCRIPTION_LENGTH, McpClientManager, sanitize::sanitize_text, truncate},
    permission::Permission,
    provider::ToolDefinition,
};

/// Cap on total bytes returned by `read_mcp_resource` across all content
/// chunks from a single server response. Mirrors `MAX_MCP_IMAGE_BYTES`:
/// servers can return large blob or text resources that would otherwise be
/// cloned verbatim into the provider request and blown through the user's
/// API quota (or OOM the agent).
pub const MAX_MCP_RESOURCE_BYTES: usize = 10 * 1024 * 1024;

fn no_such_server(tool_name: &str, server: &str, available: &[String]) -> MekaError {
    MekaError::ToolExecution {
        tool_name: tool_name.to_string(),
        message: format!(
            "unknown MCP server '{}' (configured: {})",
            server,
            if available.is_empty() {
                "<none>".to_string()
            } else {
                available.join(", ")
            }
        ),
    }
}

pub(crate) struct ListMcpResourcesTool {
    pub manager: Arc<McpClientManager>,
}

#[async_trait]
impl Tool for ListMcpResourcesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "list_mcp_resources".to_string(),
            description: "List resources advertised by MCP servers. If `server` is provided, \
                 list only that server's resources; otherwise list all configured \
                 servers. Each row is `<server>\\t<uri>\\t<name>\\t<mime>\\t<description>`."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "Optional MCP server name. If omitted, lists resources from every configured server."
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
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let server_filter = input
            .get("server")
            .and_then(|v| v.as_str())
            .map(String::from);

        let names: Vec<String> = if let Some(name) = &server_filter {
            if self.manager.server_entry(name).is_none() {
                return Err(no_such_server(
                    "list_mcp_resources",
                    name,
                    &self.manager.server_names(),
                ));
            }
            vec![name.clone()]
        } else {
            self.manager.server_names()
        };

        if names.is_empty() {
            return Ok(ToolOutput::text(
                "(no MCP servers configured)".to_string(),
                false,
            ));
        }

        let mut lines = Vec::new();
        let mut any_error = false;

        for name in names {
            let Some(entry) = self.manager.server_entry(&name) else {
                continue;
            };
            match crate::mcp::list_resources(&entry).await {
                Ok(resources) => {
                    for resource in resources {
                        // rmcp 2.1 flattened `Resource` (the fields that used to sit under `.raw`
                        // are now directly on the struct).
                        let raw = &resource;
                        let mime = raw.mime_type.as_deref().unwrap_or("");
                        let description = raw.description.as_deref().unwrap_or("");
                        let description =
                            truncate(&sanitize_text(description), MAX_MCP_DESCRIPTION_LENGTH);
                        lines.push(format!(
                            "{}\t{}\t{}\t{}\t{}",
                            name,
                            sanitize_text(&raw.uri),
                            sanitize_text(&raw.name),
                            sanitize_text(mime),
                            description
                        ));
                    }
                }
                Err(error) => {
                    any_error = true;
                    lines.push(format!("{}\t<error>\t\t\t{}", name, error));
                }
            }
        }

        if lines.is_empty() {
            return Ok(ToolOutput::text(
                "(no resources advertised)".to_string(),
                false,
            ));
        }

        Ok(ToolOutput::text(lines.join("\n"), any_error))
    }
}

pub(crate) struct ReadMcpResourceTool {
    pub manager: Arc<McpClientManager>,
}

#[async_trait]
impl Tool for ReadMcpResourceTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read_mcp_resource".to_string(),
            description: "Read an MCP resource by URI from a specific server. Text \
                          resources are returned inline; binary resources are \
                          returned base64-encoded with their declared MIME type."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "MCP server name that advertises the resource."
                    },
                    "uri": {
                        "type": "string",
                        "description": "Resource URI (e.g. file:///path/to/file). Exactly as listed by `list_mcp_resources`."
                    }
                },
                "required": ["server", "uri"]
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
        let server = require_str(&input, "server", "read_mcp_resource")?;
        let uri = require_str(&input, "uri", "read_mcp_resource")?;

        let entry = self.manager.server_entry(&server).ok_or_else(|| {
            no_such_server("read_mcp_resource", &server, &self.manager.server_names())
        })?;

        let result = crate::mcp::read_resource(&entry, uri.clone()).await?;

        let chunks = format_resource_contents(&result.contents, MAX_MCP_RESOURCE_BYTES);

        if chunks.is_empty() {
            return Ok(ToolOutput::text(
                format!("resource '{}' returned no content", uri),
                false,
            ));
        }

        Ok(ToolOutput::text(chunks.join("\n\n"), false))
    }
}

/// Render MCP `ResourceContents` into formatted chunks with Unicode sanitisation applied to all
/// server-supplied strings (URIs, MIME types, text bodies) and a hard byte budget across the whole
/// response. Split from `ReadMcpResourceTool::execute` so it's exercisable from tests.
fn format_resource_contents(
    contents: &[rmcp::model::ResourceContents],
    max_bytes: usize,
) -> Vec<String> {
    let mut chunks = Vec::with_capacity(contents.len());
    let mut total_bytes: usize = 0;
    let mut truncated = false;
    for entry in contents {
        if truncated {
            break;
        }
        match entry {
            rmcp::model::ResourceContents::TextResourceContents {
                uri: content_uri,
                mime_type,
                text,
                ..
            } => {
                if total_bytes.saturating_add(text.len()) > max_bytes {
                    chunks.push(format!(
                        "--- {} [truncated: would exceed {} byte limit]",
                        sanitize_text(content_uri),
                        max_bytes
                    ));
                    truncated = true;
                    continue;
                }
                total_bytes = total_bytes.saturating_add(text.len());
                chunks.push(format!(
                    "--- {} [{}]\n{}",
                    sanitize_text(content_uri),
                    sanitize_text(mime_type.as_deref().unwrap_or("text")),
                    sanitize_text(text)
                ));
            }
            rmcp::model::ResourceContents::BlobResourceContents {
                uri: content_uri,
                mime_type,
                blob,
                ..
            } => {
                if total_bytes.saturating_add(blob.len()) > max_bytes {
                    chunks.push(format!(
                        "--- {} [truncated: blob would exceed {} byte limit]",
                        sanitize_text(content_uri),
                        max_bytes
                    ));
                    truncated = true;
                    continue;
                }
                total_bytes = total_bytes.saturating_add(blob.len());
                chunks.push(format!(
                    "--- {} [{}] (base64, {} bytes encoded)\n{}",
                    sanitize_text(content_uri),
                    sanitize_text(mime_type.as_deref().unwrap_or("application/octet-stream")),
                    blob.len(),
                    blob
                ));
            }
            // `ResourceContents` is non-exhaustive; skip any content kind this build doesn't know.
            _ => {}
        }
    }
    chunks
}

pub(crate) struct ListMcpPromptsTool {
    pub manager: Arc<McpClientManager>,
}

#[async_trait]
impl Tool for ListMcpPromptsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "list_mcp_prompts".to_string(),
            description: "List prompts advertised by MCP servers. If `server` is provided, \
                 list only that server's prompts; otherwise list all configured \
                 servers. Each row is `<server>\\t<name>\\t<description>\\t<args>`."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "Optional MCP server name. If omitted, lists prompts from every configured server."
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
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let server_filter = input
            .get("server")
            .and_then(|v| v.as_str())
            .map(String::from);

        let names: Vec<String> = if let Some(name) = &server_filter {
            if self.manager.server_entry(name).is_none() {
                return Err(no_such_server(
                    "list_mcp_prompts",
                    name,
                    &self.manager.server_names(),
                ));
            }
            vec![name.clone()]
        } else {
            self.manager.server_names()
        };

        if names.is_empty() {
            return Ok(ToolOutput::text(
                "(no MCP servers configured)".to_string(),
                false,
            ));
        }

        let mut lines = Vec::new();
        let mut any_error = false;

        for name in names {
            let Some(entry) = self.manager.server_entry(&name) else {
                continue;
            };
            match crate::mcp::list_prompts(&entry).await {
                Ok(prompts) => {
                    for prompt in prompts {
                        let description = prompt.description.unwrap_or_default();
                        let description =
                            truncate(&sanitize_text(&description), MAX_MCP_DESCRIPTION_LENGTH);
                        let args = prompt
                            .arguments
                            .unwrap_or_default()
                            .into_iter()
                            .map(|a| {
                                let sanitised = sanitize_text(&a.name);
                                if a.required == Some(true) {
                                    format!("{}!", sanitised)
                                } else {
                                    sanitised
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(",");
                        lines.push(format!(
                            "{}\t{}\t{}\t{}",
                            name,
                            sanitize_text(&prompt.name),
                            description,
                            args
                        ));
                    }
                }
                Err(error) => {
                    any_error = true;
                    lines.push(format!("{}\t<error>\t{}\t", name, error));
                }
            }
        }

        if lines.is_empty() {
            return Ok(ToolOutput::text(
                "(no prompts advertised)".to_string(),
                false,
            ));
        }

        Ok(ToolOutput::text(lines.join("\n"), any_error))
    }
}

pub(crate) struct GetMcpPromptTool {
    pub manager: Arc<McpClientManager>,
}

#[async_trait]
impl Tool for GetMcpPromptTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "get_mcp_prompt".to_string(),
            description: "Render an MCP prompt by name from a specific server. \
                          Returns the prompt's messages serialised as `<role>: \
                          <text>` lines. `arguments` are passed verbatim to the \
                          server; see `list_mcp_prompts` for each prompt's \
                          declared arguments."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "MCP server name that advertises the prompt."
                    },
                    "name": {
                        "type": "string",
                        "description": "Prompt name, as returned by `list_mcp_prompts`."
                    },
                    "arguments": {
                        "type": "object",
                        "description": "Arguments to pass to the prompt. Keys match the prompt's declared argument names.",
                        "additionalProperties": true
                    }
                },
                "required": ["server", "name"]
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
        let server = require_str(&input, "server", "get_mcp_prompt")?;
        let name = require_str(&input, "name", "get_mcp_prompt")?;

        let arguments = input.get("arguments").and_then(|v| v.as_object()).cloned();

        let entry = self.manager.server_entry(&server).ok_or_else(|| {
            no_such_server("get_mcp_prompt", &server, &self.manager.server_names())
        })?;

        let result = crate::mcp::get_prompt(&entry, name.clone(), arguments).await?;

        let description = result
            .description
            .map(|d| truncate(&d, MAX_MCP_DESCRIPTION_LENGTH))
            .unwrap_or_default();

        let mut lines = Vec::new();
        if !description.is_empty() {
            lines.push(format!("# {}", description));
        }

        for message in &result.messages {
            // rmcp 2.1: `PromptMessage` carries a plain `Role` and a `ContentBlock` (the same
            // content enum used everywhere else), replacing the old `PromptMessageRole` /
            // `PromptMessageContent` types.
            let role_label = match message.role {
                rmcp::model::Role::User => "user",
                rmcp::model::Role::Assistant => "assistant",
            };
            match &message.content {
                rmcp::model::ContentBlock::Text(text) => {
                    lines.push(format!("{}: {}", role_label, sanitize_text(&text.text)));
                }
                rmcp::model::ContentBlock::Image(_) => {
                    lines.push(format!("{}: [image content]", role_label));
                }
                rmcp::model::ContentBlock::Audio(_) => {
                    lines.push(format!("{}: [audio content]", role_label));
                }
                rmcp::model::ContentBlock::Resource(embedded) => {
                    lines.push(format!(
                        "{}: [embedded resource: {:?}]",
                        role_label, embedded.resource
                    ));
                }
                rmcp::model::ContentBlock::ResourceLink(link) => {
                    lines.push(format!(
                        "{}: [resource link: {}]",
                        role_label,
                        sanitize_text(&link.uri)
                    ));
                }
                _ => lines.push(format!("{}: [unsupported content]", role_label)),
            }
        }

        if lines.is_empty() {
            return Ok(ToolOutput::text(
                format!("prompt '{}' returned no messages", name),
                false,
            ));
        }

        Ok(ToolOutput::text(lines.join("\n"), false))
    }
}

// `expect()` on each registration: a collision means two builtins share a name (a coding bug,
// not a runtime condition). We want the first build to fail loudly rather than silently dropping
// the second registration. Scoped to this function because the same justification applies to every
// `.register(...).expect(...)` site below.
#[allow(clippy::expect_used)]
pub(crate) fn register_all(registry: &super::ToolRegistry, manager: Arc<McpClientManager>) {
    // Skip registration if no servers are configured. These tools rely on the manager and there's
    // nothing useful to do without at least one.
    if manager.server_names().is_empty() {
        return;
    }
    registry
        .register(Arc::new(ListMcpResourcesTool {
            manager: Arc::clone(&manager),
        }))
        .expect("builtin list_mcp_resources tool name collision");
    registry
        .register(Arc::new(ReadMcpResourceTool {
            manager: Arc::clone(&manager),
        }))
        .expect("builtin read_mcp_resource tool name collision");
    registry
        .register(Arc::new(ListMcpPromptsTool {
            manager: Arc::clone(&manager),
        }))
        .expect("builtin list_mcp_prompts tool name collision");
    registry
        .register(Arc::new(GetMcpPromptTool {
            manager: Arc::clone(&manager),
        }))
        .expect("builtin get_mcp_prompt tool name collision");
    registry
        .register(Arc::new(SubscribeMcpResourceTool {
            manager: Arc::clone(&manager),
        }))
        .expect("builtin subscribe_mcp_resource tool name collision");
    registry
        .register(Arc::new(UnsubscribeMcpResourceTool {
            manager: Arc::clone(&manager),
        }))
        .expect("builtin unsubscribe_mcp_resource tool name collision");
    registry
        .register(Arc::new(ListMcpResourceUpdatesTool))
        .expect("builtin list_mcp_resource_updates tool name collision");
    drop(manager);

    // These tools are discovery-style helpers; mark them deferred so they don't clutter the tool
    // list until a prompt/resource-focused flow is activated. The registry's auto-activate path
    // already promotes them to the API when invoked.
    registry.mark_deferred("list_mcp_resources");
    registry.mark_deferred("read_mcp_resource");
    registry.mark_deferred("list_mcp_prompts");
    registry.mark_deferred("get_mcp_prompt");
    registry.mark_deferred("subscribe_mcp_resource");
    registry.mark_deferred("unsubscribe_mcp_resource");
    registry.mark_deferred("list_mcp_resource_updates");
}

pub(crate) struct SubscribeMcpResourceTool {
    pub manager: Arc<McpClientManager>,
}

#[async_trait]
impl Tool for SubscribeMcpResourceTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "subscribe_mcp_resource".to_string(),
            description: "Subscribe to change notifications for an MCP resource. After \
                          subscribing, the server will send resources/updated notifications \
                          that meka records; query them with `list_mcp_resource_updates`."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "server": {"type": "string", "description": "MCP server name"},
                    "uri": {"type": "string", "description": "Resource URI to subscribe to"}
                },
                "required": ["server", "uri"]
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
        let server = require_str(&input, "server", "subscribe_mcp_resource")?;
        let uri = require_str(&input, "uri", "subscribe_mcp_resource")?;
        let entry = self.manager.server_entry(&server).ok_or_else(|| {
            no_such_server(
                "subscribe_mcp_resource",
                &server,
                &self.manager.server_names(),
            )
        })?;
        crate::mcp::subscribe_resource(&entry, uri.clone())
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "subscribe_mcp_resource".to_string(),
                message: format!("subscribe failed: {}", error),
            })?;
        Ok(ToolOutput::text(
            format!("subscribed to '{}' on server '{}'", uri, server),
            false,
        ))
    }
}

pub(crate) struct UnsubscribeMcpResourceTool {
    pub manager: Arc<McpClientManager>,
}

#[async_trait]
impl Tool for UnsubscribeMcpResourceTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "unsubscribe_mcp_resource".to_string(),
            description: "Cancel a prior subscription to an MCP resource.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "server": {"type": "string", "description": "MCP server name"},
                    "uri": {"type": "string", "description": "Resource URI to unsubscribe from"}
                },
                "required": ["server", "uri"]
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
        let server = require_str(&input, "server", "unsubscribe_mcp_resource")?;
        let uri = require_str(&input, "uri", "unsubscribe_mcp_resource")?;
        let entry = self.manager.server_entry(&server).ok_or_else(|| {
            no_such_server(
                "unsubscribe_mcp_resource",
                &server,
                &self.manager.server_names(),
            )
        })?;
        crate::mcp::unsubscribe_resource(&entry, uri.clone())
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "unsubscribe_mcp_resource".to_string(),
                message: format!("unsubscribe failed: {}", error),
            })?;
        Ok(ToolOutput::text(
            format!("unsubscribed from '{}' on server '{}'", uri, server),
            false,
        ))
    }
}

pub(crate) struct ListMcpResourceUpdatesTool;

#[async_trait]
impl Tool for ListMcpResourceUpdatesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "list_mcp_resource_updates".to_string(),
            description: "List all resources that have been reported as updated since \
                          this meka session started. Rows are `<server>\\t<uri>\\t<unix_ts>`."
                .to_string(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
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
        let updates = crate::mcp::resource_updates::snapshot();
        if updates.is_empty() {
            return Ok(ToolOutput::text(
                "(no MCP resource updates recorded)".to_string(),
                false,
            ));
        }
        let body = updates
            .into_iter()
            .map(|(server, uri, stamp)| format!("{}\t{}\t{}", server, uri, stamp))
            .collect::<Vec<_>>()
            .join("\n");
        Ok(ToolOutput::text(body, false))
    }
}

#[cfg(test)]
mod tests {
    use rmcp::model::ResourceContents;

    use super::*;

    fn text_contents(uri: &str, mime: Option<&str>, body: &str) -> ResourceContents {
        ResourceContents::TextResourceContents {
            uri: uri.to_string(),
            mime_type: mime.map(str::to_string),
            text: body.to_string(),
            meta: None,
        }
    }

    fn blob_contents(uri: &str, mime: Option<&str>, blob: &str) -> ResourceContents {
        ResourceContents::BlobResourceContents {
            uri: uri.to_string(),
            mime_type: mime.map(str::to_string),
            blob: blob.to_string(),
            meta: None,
        }
    }

    #[test]
    fn format_resource_contents_strips_ansi_and_rtl_from_text() {
        let contents = vec![text_contents(
            "file:///evil.txt",
            Some("text/plain"),
            "before\x1b[2Jafter\u{202E}rtl",
        )];
        let out = format_resource_contents(&contents, 1_000_000).join("\n");
        assert!(!out.contains('\x1b'), "ANSI escape leaked: {:?}", out);
        assert!(!out.contains('\u{202E}'), "RTL override leaked: {:?}", out);
        assert!(out.contains("before"));
        assert!(out.contains("after"));
        assert!(out.contains("rtl"));
    }

    #[test]
    fn format_resource_contents_strips_control_chars_from_uri_and_mime() {
        let contents = vec![text_contents(
            "file:///a\u{200B}b",
            Some("text/\x07plain"),
            "body",
        )];
        let out = format_resource_contents(&contents, 1_000_000).join("\n");
        assert!(!out.contains('\u{200B}'));
        assert!(!out.contains('\x07'));
        assert!(out.contains("file:///ab"));
    }

    #[test]
    fn format_resource_contents_truncates_when_text_exceeds_cap() {
        // 1 MiB text body with a 1 KiB cap: must bail out, not emit the body.
        let body = "X".repeat(1024 * 1024);
        let contents = vec![text_contents("file:///big.txt", Some("text/plain"), &body)];
        let out = format_resource_contents(&contents, 1024).join("\n");
        assert!(out.contains("truncated"));
        assert!(out.contains("1024"));
        // The giant body itself must NOT have been emitted.
        assert!(
            !out.contains(&"X".repeat(2048)),
            "truncation failed to omit body"
        );
    }

    #[test]
    fn format_resource_contents_truncates_when_blob_exceeds_cap() {
        let blob = "B".repeat(1024 * 1024);
        let contents = vec![blob_contents(
            "file:///big.bin",
            Some("application/octet-stream"),
            &blob,
        )];
        let out = format_resource_contents(&contents, 1024).join("\n");
        assert!(out.contains("truncated"));
        assert!(out.contains("blob would exceed"));
        assert!(
            !out.contains(&"B".repeat(2048)),
            "blob leaked despite truncation"
        );
    }

    #[test]
    fn format_resource_contents_stops_after_first_truncation() {
        // First entry fills the budget, second should be dropped entirely.
        let first = "Y".repeat(2048);
        let contents = vec![
            text_contents("file:///first.txt", Some("text/plain"), &first),
            text_contents("file:///second.txt", Some("text/plain"), "short"),
        ];
        let out = format_resource_contents(&contents, 1024);
        // The truncation marker for the first chunk is present, and no second-chunk line should
        // appear.
        let joined = out.join("\n");
        assert!(joined.contains("first.txt"));
        assert!(joined.contains("truncated"));
        assert!(!joined.contains("second.txt"));
        assert!(!joined.contains("short"));
    }

    #[test]
    fn format_resource_contents_under_cap_emits_all_chunks() {
        let contents = vec![
            text_contents("file:///a.txt", Some("text/plain"), "alpha"),
            text_contents("file:///b.txt", Some("text/plain"), "beta"),
        ];
        let out = format_resource_contents(&contents, 1_000_000);
        assert_eq!(out.len(), 2);
        assert!(out[0].contains("alpha"));
        assert!(out[1].contains("beta"));
    }

    /// Regression guard for the scratchpad un-defer change: the seven MCP resource tools must STILL
    /// be deferred after registration. We only relaxed `scratchpad_*` deferral; if a future
    /// refactor accidentally drops `mark_deferred` calls here too, every MCP-using session would
    /// see seven extra tool schemas in its tools array on the first turn.
    #[tokio::test]
    async fn test_mcp_resource_tools_remain_deferred() {
        use crate::{
            config::{McpServerConfig, McpTransport},
            mcp::{McpClientContext, McpClientManager},
            tools::ToolRegistry,
        };

        let server_config = McpServerConfig {
            name: "fixture-srv".to_string(),
            transport: McpTransport::Stdio,
            command: Some("/bin/false".to_string()),
            args: None,
            env: None,
            url: None,
            auth_token: None,
            headers: None,
            headers_helper: None,
            auth: None,
            permission: None,
            allowed_tools: None,
            disabled_tools: None,
            eager_load_tools: None,
            tool_permissions: None,
            disabled: false,
            required: None,
        };
        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(&[server_config], None, None, context)
            .await
            .expect("prepare with one server should succeed");

        let registry = ToolRegistry::new();
        register_all(&registry, manager);

        let entries = registry.tool_catalogue();
        let by_name: std::collections::HashMap<_, _> =
            entries.iter().map(|(n, _, _, d)| (n.clone(), *d)).collect();

        for name in [
            "list_mcp_resources",
            "read_mcp_resource",
            "list_mcp_prompts",
            "get_mcp_prompt",
            "subscribe_mcp_resource",
            "unsubscribe_mcp_resource",
            "list_mcp_resource_updates",
        ] {
            assert!(
                by_name.contains_key(name),
                "MCP resource tool {} not registered",
                name
            );
            assert!(
                by_name[name],
                "MCP resource tool {} should still be deferred (would otherwise \
                 bloat the tools array on every MCP-enabled session)",
                name,
            );
        }
    }
}
