//! Client-side MCP handler: dispatches server-initiated `elicitation/create` requests to the rest
//! of the agent, forwards `tools/list_changed` notifications through the manager, and adapts the
//! remote tool list into the `crate::tools` trait so the provider loop can call them like any other
//! tool.
//!
//! meka does not implement the MCP sampling / roots / logging handlers: those features are
//! deprecated by SEP-2577 and slated for removal from the protocol, so the rmcp defaults apply
//! (sampling → `method_not_found`, roots → empty, logging → ignored).

use std::sync::Arc;

use async_trait::async_trait;
use rmcp::{
    ErrorData as McpError, Peer, RoleClient,
    handler::client::ClientHandler,
    model::{
        CallToolRequest, CallToolRequestParams, CancelledNotificationParam, ClientRequest,
        ConstString, CustomNotification, ElicitRequestParams, ElicitResult,
        ElicitationResponseNotificationMethod, ProgressNotificationParam, RequestMetaObject,
        ServerResult,
    },
    service::{NotificationContext, PeerRequestOptions, RequestContext, ServiceError},
};
use tokio_util::sync::CancellationToken;

use super::{ALLOWED_IMAGE_MIME_TYPES, MAX_MCP_IMAGE_BYTES, McpClientContext, ServerEntry};
use crate::{
    error::{MekaError, Result},
    permission::Permission,
    provider::ToolDefinition,
    tools::{Tool, ToolOutput},
};

/// Client-side MCP handler. Dispatches server-initiated `elicitation/create` requests and
/// notifications (`tools/list_changed`, progress, etc.) to the rest of the agent via the shared
/// [`McpClientContext`]. Sampling / roots / logging are intentionally not handled (SEP-2577).
#[derive(Clone)]
pub struct MekaClientHandler {
    server_name: Arc<str>,
    context: Arc<McpClientContext>,
}

impl MekaClientHandler {
    pub fn new(server_name: String, context: Arc<McpClientContext>) -> Self {
        Self {
            server_name: Arc::from(server_name),
            context,
        }
    }
}

impl ClientHandler for MekaClientHandler {
    fn on_tool_list_changed(
        &self,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        let server_name: String = self.server_name.as_ref().to_string();
        let manager = self.context.manager().and_then(|weak| weak.upgrade());

        async move {
            tracing::debug!("MCP server '{}' sent tools/list_changed", server_name);
            let Some(manager) = manager else {
                tracing::debug!(
                    "tool list refresh skipped: manager not yet wired for '{}'",
                    server_name
                );
                return;
            };

            // Tool-permission resolution reads the server config and `mcp_default_permission` from
            // the manager itself; no explicit permission needs to be threaded here.
            match manager.discover_tools_for_server(&server_name).await {
                Ok(adapters) => {
                    // Match the initial-registration path: only mark non-eager tools deferred.
                    // Compute the deferred set before we erase the adapters into `Arc<dyn Tool>`,
                    // since `raw_name`/`server_config` live on the concrete type.
                    let deferred_names: Vec<String> = adapters
                        .iter()
                        .filter(|adapter| {
                            !crate::mcp::tool_should_eager_load(
                                adapter.server_config(),
                                adapter.raw_name(),
                            )
                        })
                        .map(|adapter| adapter.definition().name)
                        .collect();
                    let new_tools: Vec<Arc<dyn Tool>> = adapters
                        .into_iter()
                        .map(|a| Arc::new(a) as Arc<dyn Tool>)
                        .collect();
                    // Routes through every attached registry so all active sessions observe the
                    // updated tool set.
                    manager.update_server_tools(&server_name, new_tools).await;
                    if !deferred_names.is_empty() {
                        manager.mark_deferred_on_attached(&deferred_names).await;
                    }
                    tracing::info!("MCP server '{}' tool registry refreshed", server_name);
                }
                Err(error) => {
                    tracing::warn!(
                        "failed to refresh tools for MCP server '{}': {}",
                        server_name,
                        error
                    );
                }
            }
        }
    }

    fn on_resource_list_changed(
        &self,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        let server = Arc::clone(&self.server_name);
        async move {
            tracing::debug!("MCP server '{}' sent resources/list_changed", server);
        }
    }

    fn on_prompt_list_changed(
        &self,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        let server = Arc::clone(&self.server_name);
        async move {
            tracing::debug!("MCP server '{}' sent prompts/list_changed", server);
        }
    }

    fn on_resource_updated(
        &self,
        params: rmcp::model::ResourceUpdatedNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        let server = Arc::clone(&self.server_name);
        async move {
            tracing::info!(
                "MCP server '{}' reported resource updated: {}",
                server,
                params.uri
            );
            crate::mcp::resource_updates::record(server.as_ref(), &params.uri);
        }
    }

    // Keep the explicit `impl Future` return type: other handlers in this trait impl have
    // non-trivial captures (`Arc<str>` clones, server name in logging, etc.) and use the same
    // signature shape. Staying uniform makes the module easier to read than mixing `async fn` and
    // the manual-future form.
    #[allow(clippy::manual_async_fn)]
    fn on_progress(
        &self,
        params: ProgressNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        async move {
            crate::mcp::progress::dispatch(params).await;
        }
    }

    /// Notification that a server-side URL elicitation the user was sent to complete has finished.
    /// rmcp 3.1 removed the typed hook this used to have, so it now arrives as an unrecognised
    /// method: the wire notification still exists, but the SDK no longer routes it anywhere
    /// specific. meka's [`Self::create_elicitation`] already returned its response synchronously,
    /// so nothing needs to drive here; log it for observability.
    fn on_custom_notification(
        &self,
        notification: CustomNotification,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        let server = Arc::clone(&self.server_name);
        async move {
            if notification.method != ElicitationResponseNotificationMethod::VALUE {
                return;
            }
            let elicitation_id = notification
                .params
                .as_ref()
                .and_then(|params| params.get("elicitationId"))
                .and_then(|id| id.as_str())
                .unwrap_or("<unknown>");
            tracing::debug!(
                "MCP server '{}' completed URL elicitation '{}'",
                server,
                elicitation_id
            );
        }
    }

    fn create_elicitation(
        &self,
        request: ElicitRequestParams,
        _context: RequestContext<RoleClient>,
    ) -> impl Future<Output = std::result::Result<ElicitResult, McpError>> + Send + '_ {
        let server = Arc::clone(&self.server_name);
        async move {
            use crate::mcp::elicitation::{
                ElicitationKind, ElicitationPrompt, ElicitationResponse,
            };

            let (kind, message) = match &request {
                ElicitRequestParams::FormElicitationParams {
                    message,
                    requested_schema,
                    ..
                } => {
                    let schema = serde_json::to_value(requested_schema)
                        .unwrap_or(serde_json::json!({"type": "object", "properties": {}}));
                    (ElicitationKind::Form { schema }, message.clone())
                }
                ElicitRequestParams::UrlElicitationParams { message, url, .. } => {
                    (ElicitationKind::Url { url: url.clone() }, message.clone())
                }
                // Forward-compat: an elicitation kind this build doesn't recognize falls back to a
                // generic form prompt rather than failing the request.
                _ => (
                    ElicitationKind::Form {
                        schema: serde_json::json!({"type": "object", "properties": {}}),
                    },
                    "unsupported elicitation request".to_string(),
                ),
            };

            let prompt = ElicitationPrompt {
                server_name: server.as_ref().to_string(),
                kind,
                message,
            };

            // Correlate the elicitation back to the in-flight call's frontend via the per-server
            // lookup on the progress registry. When no call from `server` is in flight (the server
            // elicited outside of a tool call, or the progress guard already dropped), there's no
            // human to ask. Auto-decline matches the safe pre-refactor "no shell sink installed"
            // behaviour.
            let frontend = crate::mcp::progress::find_frontend_for_server(server.as_ref());
            let Some(frontend) = frontend else {
                tracing::warn!(
                    "MCP server '{}' requested elicitation but no in-flight call's frontend was \
                     registered; declining",
                    server
                );
                return Ok(ElicitationResponse::Decline.into_result());
            };

            // User-response timeout so a distracted user can't stall an MCP tool call forever.
            // Matches `MID_TURN_REQUEST_TIMEOUT`, the deadline the HTTP frontend gives a pending
            // permission request. Elicitations are standard MCP *requests*, so a `Decline`
            // response IS how the server learns the user didn't answer; no separate
            // `notifications/cancelled` is appropriate here (cancellation notifications are for
            // long-running requests we started, not for server-initiated elicitations).
            const ELICITATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
            let response = match tokio::time::timeout(
                ELICITATION_TIMEOUT,
                frontend.handle_elicitation(prompt),
            )
            .await
            {
                Ok(response) => response,
                Err(_) => {
                    tracing::warn!(
                        "MCP server '{}' elicitation timed out after {}s; declining",
                        server,
                        ELICITATION_TIMEOUT.as_secs()
                    );
                    ElicitationResponse::Decline
                }
            };

            Ok(response.into_result())
        }
    }
}

pub struct McpToolAdapter {
    namespaced_name: String,
    remote_tool_name: String,
    description: String,
    parameters: serde_json::Value,
    permission: Permission,
    entry: Arc<ServerEntry>,
    /// `tool.annotations` and `tool.meta` captured from the remote server. Surfaced to the
    /// provider as hints (read-only / destructive) and round-tripped back in `_meta` so the
    /// MCP server can correlate client-side context.
    annotations: Option<serde_json::Value>,
    meta: Option<serde_json::Value>,
    title: Option<String>,
}

impl McpToolAdapter {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        namespaced_name: String,
        remote_tool_name: String,
        description: String,
        parameters: serde_json::Value,
        permission: Permission,
        entry: Arc<ServerEntry>,
        annotations: Option<serde_json::Value>,
        meta: Option<serde_json::Value>,
        title: Option<String>,
    ) -> Self {
        Self {
            namespaced_name,
            remote_tool_name,
            description,
            parameters,
            permission,
            entry,
            annotations,
            meta,
            title,
        }
    }

    /// Raw, server-advertised tool name (not the `mcp__<server>__<tool>` namespaced form). Used to
    /// look the tool up in per-server config fields like `eager_load_tools`.
    pub(crate) fn raw_name(&self) -> &str {
        &self.remote_tool_name
    }

    /// The server config that produced this adapter. Used to read per-server policy (eager-load,
    /// permission overrides, …) without rediscovering the manager.
    pub(crate) fn server_config(&self) -> &crate::config::McpServerConfig {
        &self.entry.config
    }

    /// Resolves a per-call tool-call timeout. Respects `MEKA_MCP_TOOL_TIMEOUT` (milliseconds) when
    /// set, otherwise falls back to 600 seconds, long enough for a database index rebuild but
    /// short enough that a hung server isn't invisible.
    fn tool_call_timeout() -> std::time::Duration {
        std::env::var("MEKA_MCP_TOOL_TIMEOUT")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(std::time::Duration::from_millis)
            .unwrap_or(std::time::Duration::from_secs(600))
    }

    async fn call_tool_once(
        &self,
        mut params: CallToolRequestParams,
        cancellation: CancellationToken,
        tool_use_id: Option<String>,
    ) -> std::result::Result<rmcp::model::CallToolResult, ServiceError> {
        // Per-call progress token: allows the server to emit `notifications/progress` updates that
        // route back to our shell UI. The frontend snapshot is taken from the task-local installed
        // by `Agent::run_tool` and stored on the registry entry so the rmcp notification handler
        // (which runs on a separately-spawned task; see `rmcp::service::spawn_service_task`) can
        // look it up by token. `None` outside an agent-driven call site falls through to a debug
        // log in `dispatch`.
        let frontend_for_progress = crate::mcp::current_session_frontend();
        let (progress_token, _progress_guard) = crate::mcp::progress::register(
            self.entry.server_name().to_string(),
            self.remote_tool_name.clone(),
            tool_use_id.clone(),
            frontend_for_progress,
        );
        let mut meta = RequestMetaObject::new();
        meta.set_progress_token(progress_token);
        if let Some(id) = &tool_use_id {
            meta.0
                .insert("meka/toolUseId".to_string(), serde_json::json!(id));
        }
        // Lets a server scope per-session state (a cache, a workspace, a connection pool, an audit
        // trail) to the conversation the call came from. `_meta` is the spec's extension point and
        // already carries `meka/toolUseId`, so this adds no new wire contract.
        if let Some(session_id) = crate::mcp::current_session_id() {
            meta.0.insert(
                "meka/sessionId".to_string(),
                serde_json::json!(session_id.to_string()),
            );
        }
        params.meta = Some(meta);

        // Same error surface as an actually-closed transport. The upstream retry logic already
        // handles `TransportClosed` by attempting a reconnect.
        let peer: Peer<RoleClient> = self
            .entry
            .require_connected()
            .await
            .map_err(|_| ServiceError::TransportClosed)?;
        let request = ClientRequest::CallToolRequest(CallToolRequest::new(params));
        let handle = peer
            .send_cancellable_request(request, PeerRequestOptions::no_options())
            .await?;
        let request_id = handle.id.clone();

        let timeout = Self::tool_call_timeout();
        // Cap how long we wait on the best-effort cancellation notification so a hung transport
        // can't block Ctrl-C handling or shutdown.
        const CANCEL_NOTIFY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);
        let notify_cancel = |reason: &'static str| {
            let peer = peer.clone();
            let request_id = request_id.clone();
            let server_name = self.entry.server_name().to_string();
            async move {
                let send = peer.notify_cancelled(CancelledNotificationParam::new(
                    Some(request_id),
                    Some(reason.to_string()),
                ));
                match tokio::time::timeout(CANCEL_NOTIFY_TIMEOUT, send).await {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        tracing::debug!(
                            "failed to send cancellation notification to '{}': {}",
                            server_name,
                            error
                        );
                    }
                    Err(_) => {
                        tracing::debug!(
                            "cancellation notification to '{}' timed out after {}s",
                            server_name,
                            CANCEL_NOTIFY_TIMEOUT.as_secs()
                        );
                    }
                }
            }
        };

        tokio::select! {
            response = handle.await_response() => {
                match response? {
                    ServerResult::CallToolResult(result) => Ok(result),
                    _ => Err(ServiceError::UnexpectedResponse),
                }
            }
            _ = cancellation.cancelled() => {
                notify_cancel("user interrupt").await;
                Err(ServiceError::Cancelled {
                    reason: Some("user interrupt".to_string()),
                })
            }
            _ = tokio::time::sleep(timeout) => {
                notify_cancel("timeout").await;
                Err(ServiceError::Cancelled {
                    reason: Some(format!("timed out after {}s", timeout.as_secs())),
                })
            }
        }
    }
}

#[async_trait]
impl Tool for McpToolAdapter {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.namespaced_name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
            title: self.title.clone(),
            annotations: self.annotations.clone(),
            meta: self.meta.clone(),
        }
    }

    fn required_permission(&self) -> Permission {
        self.permission
    }

    /// An MCP call runs in the server's own process, which meka spawns but does not sandbox.
    fn runs_outside_confinement(&self) -> bool {
        true
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let arguments = forwarded_arguments(&input, &self.parameters);

        let params = {
            let mut p = CallToolRequestParams::new(self.remote_tool_name.clone());
            p.arguments = arguments;
            p
        };

        let is_timeout = |error: &ServiceError| matches!(error, ServiceError::Cancelled { reason: Some(reason) } if reason.starts_with("timed out"));

        // Correlates this MCP call with the provider's tool-use entry, for `meka/toolUseId` in
        // `_meta` and for the progress registry. `None` outside an agent-driven dispatch.
        let tool_use_id = crate::tools::current_tool_call_id();

        // First attempt. On TransportClosed, reconnect and retry once.
        let result = match self
            .call_tool_once(params.clone(), cancellation.clone(), tool_use_id.clone())
            .await
        {
            Ok(result) => result,
            Err(ServiceError::Cancelled { reason })
                if reason.as_deref() == Some("user interrupt") =>
            {
                return Err(MekaError::Interrupted);
            }
            Err(error) if is_timeout(&error) => {
                return Err(MekaError::McpToolExecution {
                    server_name: self.entry.server_name().to_string(),
                    tool_name: self.remote_tool_name.clone(),
                    message: error.to_string(),
                });
            }
            Err(ServiceError::TransportClosed) => {
                self.entry.reconnect().await?;
                match self.call_tool_once(params, cancellation, tool_use_id).await {
                    Ok(result) => result,
                    Err(ServiceError::Cancelled { reason })
                        if reason.as_deref() == Some("user interrupt") =>
                    {
                        return Err(MekaError::Interrupted);
                    }
                    Err(error) => {
                        return Err(MekaError::McpToolExecution {
                            server_name: self.entry.server_name().to_string(),
                            tool_name: self.remote_tool_name.clone(),
                            message: error.to_string(),
                        });
                    }
                }
            }
            Err(error) => {
                // Matched on the text because the transport surfaces the status inside the error
                // rather than as a field, so a bare `Unauthorized` with no code still counts.
                let text = error.to_string().to_ascii_lowercase();
                if text.contains("401") || text.contains("unauthorized") {
                    tracing::warn!(
                        "MCP server '{}' rejected the call as unauthorized; run `meka mcp login {}` to re-authenticate",
                        self.entry.server_name(),
                        self.entry.server_name()
                    );
                }
                return Err(MekaError::McpToolExecution {
                    server_name: self.entry.server_name().to_string(),
                    tool_name: self.remote_tool_name.clone(),
                    message: error.to_string(),
                });
            }
        };

        Ok(tool_output_from_result(
            &result,
            format!("mcp_{}_{}", self.entry.server_name(), self.remote_tool_name),
        ))
    }
}

/// Convert a server's `CallToolResult` into meka's [`ToolOutput`].
///
/// Split out of `execute` so it can be tested without a live server: everything above it in
/// `execute` is transport and retry, and none of that bears on how a result is shaped.
fn tool_output_from_result(
    result: &rmcp::model::CallToolResult,
    scratchpad_hint: String,
) -> ToolOutput {
    let mut content = convert_tool_result_content(&result.content);

    // If the server included structured_content, append it as a fenced JSON block so providers
    // can reason over it without needing a dedicated ToolResultContent variant. Matches Claude
    // Code's pragmatic passthrough.
    //
    // This block is for the model to *read*. Callers that compute on the result take the
    // `structured` field below instead, so the wording and fencing here stay free to change
    // without altering what a scheduled job's gate predicate decides.
    if let Some(structured) = &result.structured_content {
        let pretty = serde_json::to_string_pretty(structured).unwrap_or_default();
        if !pretty.is_empty() {
            let appended = format!("\n\n---\n**Structured content:**\n```json\n{}\n```", pretty);
            content.push(crate::provider::ToolResultContent::Text { text: appended });
        }
    }

    // Unicode sanitisation on every text block that came from the server.
    for block in content.iter_mut() {
        if let crate::provider::ToolResultContent::Text { text } = block {
            *text = crate::mcp::sanitize::sanitize_text(text);
        }
    }

    ToolOutput {
        content,
        is_error: result.is_error.unwrap_or(false),
        scratchpad_hint: Some(scratchpad_hint),
        frontend_metadata: None,
        structured: result.structured_content.clone(),
    }
}

/// Map MCP `CallToolResult.content` items to meka's provider-layer `ToolResultContent` blocks. Text
/// stays text; images pass through as multimodal blocks so providers like Claude and GPT-4o can see
/// them; audio, embedded resources, and resource links collapse to informative text placeholders
/// (no provider accepts them as tool-result blocks yet).
fn convert_tool_result_content(
    items: &[rmcp::model::ContentBlock],
) -> Vec<crate::provider::ToolResultContent> {
    use crate::provider::{ImageSource, ToolResultContent};

    // The one unbounded thing a server controls in a tool result. Images have had a ceiling since
    // they were added; text was appended until the server stopped sending, and every byte then went
    // into the session and back to the provider on every subsequent turn. Generous enough that no
    // real result reaches it: four megabytes is roughly a million tokens.
    //
    // Past it the excess is *dropped*, not spilled, which is a knowingly weaker guarantee than the
    // one `tools::shell` gives: an overflowing command writes every byte to a file and hands the
    // model its path. Two things stand in the way of matching it here. This function is a pure
    // transform over content blocks with no session to spill into, and
    // `scratchpad::persist_oversized_results` -- which would preserve the rest -- runs on the
    // `ToolOutput` *after* `execute` returns, so it never sees what was cut. Closing the gap means
    // threading the scratchpad through the conversion, and is worth doing when a real server is
    // observed hitting four megabytes; none has been. The loss is disclosed either way, which is
    // the property that actually matters: the model is told the result was cut rather than
    // answering from a silent truncation.
    const MAX_MCP_TEXT_BYTES: usize = 4 * 1024 * 1024;

    let mut blocks: Vec<ToolResultContent> = Vec::new();
    let mut text_buf = String::new();
    let mut text_dropped: usize = 0;
    // Counted across the whole result, not per buffer.
    //
    // The ceiling used to be compared against `text_buf.len()` alone, and the accepted-image arm
    // calls `flush_text`, which `mem::take`s the buffer. A result shaped `[4 MiB text][small
    // PNG][4 MiB text][PNG]...` therefore passed the guard on every round and the cap bounded
    // nothing: the resident total is the sum of the flushed blocks, which is what the model is
    // sent.
    let mut text_kept: usize = 0;

    let flush_text = |buf: &mut String, out: &mut Vec<ToolResultContent>| {
        if !buf.is_empty() {
            out.push(ToolResultContent::Text {
                text: std::mem::take(buf),
            });
        }
    };

    for item in items {
        match item {
            rmcp::model::ContentBlock::Text(text_content) => {
                if text_kept >= MAX_MCP_TEXT_BYTES {
                    text_dropped += text_content.text.len();
                    continue;
                }
                if !text_buf.is_empty() {
                    text_buf.push('\n');
                    text_kept += 1;
                }
                let room = MAX_MCP_TEXT_BYTES - text_kept;
                if text_content.text.len() <= room {
                    text_buf.push_str(&text_content.text);
                    text_kept += text_content.text.len();
                } else {
                    let cut = text_content.text.floor_char_boundary(room);
                    text_buf.push_str(&text_content.text[..cut]);
                    text_kept += cut;
                    text_dropped += text_content.text.len() - cut;
                }
            }
            rmcp::model::ContentBlock::Image(image) => {
                // The server's declared `mime_type` is a hint; what gets forwarded is decided by
                // the bytes. Providers sniff and reject a mismatch with a 400, and that rejection
                // lands inside a `tool_result` already committed to the session, where it fails
                // every later request. Only the first few characters are decoded, so this costs
                // nothing on a multi-megabyte payload.
                let sniffed = match crate::image::classify_base64_prefix(&image.data) {
                    crate::image::ImageHandling::PassThrough(format) => Some(format.to_mime_type()),
                    // A format needing transcoding (TIFF, ICO, ...) would mean decoding and
                    // re-encoding the whole payload, which is not worth it for a server that
                    // mislabelled its own output.
                    _ => None,
                }
                .filter(|mime| {
                    ALLOWED_IMAGE_MIME_TYPES
                        .iter()
                        .any(|allowed| allowed.eq_ignore_ascii_case(mime))
                });
                // Two ceilings, and both matter. `MAX_MCP_IMAGE_BYTES` is meka's own memory and
                // quota guard. The second is the providers' limit, which every other image
                // producer gets for free by going through `prepare_image_payload`; this path
                // builds its `ImageSource` directly, so without it an MCP image between the two
                // ceilings is forwarded only for the provider to answer 400. Base64 carries 3
                // bytes per 4 characters, which is exact enough to compare against a cap.
                let decoded_len = image.data.len() / 4 * 3;
                let oversize = if image.data.len() > MAX_MCP_IMAGE_BYTES {
                    Some(format!(
                        "{} base64 bytes exceeds {} byte limit",
                        image.data.len(),
                        MAX_MCP_IMAGE_BYTES
                    ))
                } else if decoded_len > crate::image::MAX_IMAGE_RAW_BYTES {
                    Some(format!(
                        "~{} decoded bytes exceeds the {} byte ceiling providers accept",
                        decoded_len,
                        crate::image::MAX_IMAGE_RAW_BYTES
                    ))
                } else {
                    None
                };
                if let Some(reason) = oversize {
                    if !text_buf.is_empty() {
                        text_buf.push('\n');
                    }
                    text_buf.push_str(&format!("[image suppressed: {}]", reason));
                } else if let Some(media_type) = sniffed {
                    flush_text(&mut text_buf, &mut blocks);
                    blocks.push(ToolResultContent::Image {
                        source: ImageSource {
                            source_type: "base64".to_string(),
                            media_type: media_type.to_string(),
                            data: image.data.clone(),
                        },
                    });
                } else {
                    if !text_buf.is_empty() {
                        text_buf.push('\n');
                    }
                    text_buf.push_str(&format!(
                        "[image suppressed: declared '{}', but the bytes are not an allowed image \
                         format]",
                        image.mime_type
                    ));
                }
            }
            rmcp::model::ContentBlock::Audio(audio) => {
                if !text_buf.is_empty() {
                    text_buf.push('\n');
                }
                text_buf.push_str(&format!(
                    "[audio content: {}, {} base64 bytes; meka does not yet pass audio to the provider]",
                    audio.mime_type,
                    audio.data.len()
                ));
            }
            rmcp::model::ContentBlock::Resource(resource) => {
                if !text_buf.is_empty() {
                    text_buf.push('\n');
                }
                match &resource.resource {
                    rmcp::model::ResourceContents::TextResourceContents { uri, text, .. } => {
                        text_buf.push_str(&format!("--- {}\n{}", uri, text));
                    }
                    rmcp::model::ResourceContents::BlobResourceContents {
                        uri,
                        mime_type,
                        blob,
                        ..
                    } => {
                        text_buf.push_str(&format!(
                            "[embedded blob resource: {} ({}), {} base64 bytes]",
                            uri,
                            mime_type.as_deref().unwrap_or("application/octet-stream"),
                            blob.len()
                        ));
                    }
                    _ => text_buf.push_str("[embedded resource omitted]"),
                }
            }
            rmcp::model::ContentBlock::ResourceLink(link) => {
                if !text_buf.is_empty() {
                    text_buf.push('\n');
                }
                text_buf.push_str(&format!("[resource link: {}]", link.uri));
            }
            // `ContentBlock` is non-exhaustive; a block kind this build doesn't recognize collapses
            // to a placeholder rather than being dropped silently.
            _ => {
                if !text_buf.is_empty() {
                    text_buf.push('\n');
                }
                text_buf.push_str("[unsupported content omitted]");
            }
        }
    }

    if text_dropped > 0 {
        if !text_buf.is_empty() {
            text_buf.push('\n');
        }
        text_buf.push_str(&format!(
            "\n... ({} further bytes of text were dropped; this server's result exceeded the {} \
             byte ceiling)",
            text_dropped, MAX_MCP_TEXT_BYTES
        ));
    }
    flush_text(&mut text_buf, &mut blocks);
    if blocks.is_empty() {
        blocks.push(ToolResultContent::Text {
            text: String::new(),
        });
    }
    blocks
}

/// The arguments an MCP call actually carries: everything the model sent, minus meka's own.
///
/// `scratchpad` and `background` are accepted on every tool and consumed by the agent loop, so a
/// remote server never declared them. Forwarding them sends a property the server did not ask for,
/// which a strict schema validator on the far side rejects outright -- failing a call whose only
/// fault was that the model used a meka feature. The tool documentation already said this happened;
/// it did not.
///
/// `schema` is the tool's own advertised `input_schema`, and a name it declares belongs to *it*.
/// `offer_background` in `src/tools.rs` already refuses to splice `background` onto a tool that
/// advertises the name, precisely so a server owning it keeps its meaning -- but this side stripped
/// unconditionally, so the value the model sent for the *server's* parameter was deleted on the way
/// out and the call arrived missing an argument it had asked for. Both halves have to consult the
/// schema or the pair is incoherent.
fn forwarded_arguments(
    input: &serde_json::Value,
    schema: &serde_json::Value,
) -> Option<serde_json::Map<String, serde_json::Value>> {
    let declares = |name: &str| {
        schema
            .get("properties")
            .and_then(serde_json::Value::as_object)
            .is_some_and(|properties| properties.contains_key(name))
    };
    let strip_scratchpad = !declares(crate::tools::SCRATCHPAD_PARAMETER);
    let strip_background = !declares(crate::tools::BACKGROUND_PARAMETER);
    input.as_object().map(|object| {
        object
            .iter()
            .filter(|(key, _)| {
                !((strip_scratchpad && key.as_str() == crate::tools::SCRATCHPAD_PARAMETER)
                    || (strip_background && key.as_str() == crate::tools::BACKGROUND_PARAMETER))
            })
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use base64::Engine as _;

    use super::*;

    /// A server's structured output reaches callers as data, not only as rendered prose.
    ///
    /// The fenced block is what the model reads and is deliberately presentational, so anything
    /// deciding *on* a result -- a scheduled job's gate predicate is the caller this exists for --
    /// has to take the field. Recovering the JSON by parsing the block back out would make that
    /// format string a wire format between two parts of meka while it reads as formatting, and a
    /// readability edit would then silently change what a gate decides. Both halves are asserted
    /// here so neither can quietly stop happening.
    #[test]
    fn structured_content_is_carried_as_data_and_still_rendered_for_the_model() {
        let structured = serde_json::json!({ "chats": [{ "id": "a" }], "checked_at": "now" });
        let mut result =
            rmcp::model::CallToolResult::success(vec![rmcp::model::ContentBlock::text(
                "1 unseen chat",
            )]);
        result.structured_content = Some(structured.clone());

        let output = tool_output_from_result(&result, "mcp_bridge_unseen".to_string());

        assert_eq!(
            output.structured.as_ref(),
            Some(&structured),
            "a predicate must reach the value without parsing the rendering"
        );
        let rendered = output
            .content
            .iter()
            .filter_map(|block| match block {
                crate::provider::ToolResultContent::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert!(
            rendered.contains("**Structured content:**") && rendered.contains("\"chats\""),
            "the model still sees the fenced block: {rendered}"
        );
    }

    /// The common case: a server that sends only text leaves `structured` empty rather than
    /// inventing a value, so a pointer predicate knows to fall back to parsing the text itself.
    #[test]
    fn a_text_only_result_carries_no_structured_value() {
        let result = rmcp::model::CallToolResult::success(vec![rmcp::model::ContentBlock::text(
            "{\"chats\": []}",
        )]);

        let output = tool_output_from_result(&result, "mcp_bridge_unseen".to_string());

        assert!(output.structured.is_none());
    }

    /// A remote server sees the model's arguments and nothing of meka's.
    ///
    /// `scratchpad` and `background` are accepted on every tool and consumed here, so no server
    /// declares them; sending one is an undeclared property, and a server validating its schema
    /// strictly refuses the call over it. `tools/overview.md` documented this stripping before the
    /// code did it.
    /// A server that declares `background` or `scratchpad` itself owns the name, and the value the
    /// model sent for it must reach the server.
    ///
    /// `offer_background` (src/tools.rs) already declines to splice `background` onto a tool that
    /// advertises it, exactly so the server keeps the name. This side stripped unconditionally, so
    /// the pair disagreed: meka left the server's own parameter in the schema the model reads, then
    /// deleted the model's answer on the way out, and the call arrived missing a required argument.
    #[test]
    fn a_parameter_the_server_declares_is_forwarded_not_stripped() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "prompt": {}, "background": { "type": "string" } }
        });
        let arguments = forwarded_arguments(
            &serde_json::json!({
                "prompt": "a cat",
                "background": "transparent",
                "scratchpad": "out",
            }),
            &schema,
        )
        .expect("an object of arguments");

        assert_eq!(
            arguments.get("background").and_then(|v| v.as_str()),
            Some("transparent"),
            "the server declared `background`, so it is the server's parameter"
        );
        assert!(
            !arguments.contains_key("scratchpad"),
            "`scratchpad` is still meka's here, and is still stripped"
        );
    }

    #[test]
    fn meka_only_parameters_are_not_forwarded_to_the_server() {
        let arguments = forwarded_arguments(
            &serde_json::json!({
                "query": "rust",
                "limit": 10,
                "scratchpad": "results",
                "background": true,
            }),
            // A schema declaring neither name: the ordinary case, where both are meka's.
            &serde_json::json!({"type": "object", "properties": {"query": {}, "limit": {}}}),
        )
        .expect("an object of arguments");

        assert!(!arguments.contains_key("scratchpad"));
        assert!(!arguments.contains_key("background"));
        assert_eq!(arguments.get("query"), Some(&serde_json::json!("rust")));
        assert_eq!(arguments.get("limit"), Some(&serde_json::json!(10)));
    }

    /// A tool taking no arguments still sends `{}` rather than nothing, and a non-object input is
    /// passed through as "no arguments" the way it always was.
    #[test]
    fn stripping_leaves_an_argumentless_call_intact() {
        assert_eq!(
            forwarded_arguments(&serde_json::json!({}), &serde_json::json!({})),
            Some(serde_json::Map::new()),
        );
        assert_eq!(
            forwarded_arguments(&serde_json::Value::Null, &serde_json::json!({})),
            None
        );
    }

    /// Base64 of a real 4x4 image in `format`. The handler classifies from the bytes now, so a
    /// placeholder string is no longer a usable fixture for the accept path.
    fn base64_image(format: image::ImageFormat) -> String {
        let mut bytes = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut bytes);
        // JPEG has no alpha channel, so it needs an RGB source.
        if format == image::ImageFormat::Jpeg {
            image::RgbImage::from_pixel(4, 4, image::Rgb([128, 64, 200]))
                .write_to(&mut cursor, format)
                .expect("encode");
        } else {
            image::RgbaImage::from_pixel(4, 4, image::Rgba([128, 64, 200, 255]))
                .write_to(&mut cursor, format)
                .expect("encode");
        }
        base64::engine::general_purpose::STANDARD.encode(&bytes)
    }

    #[test]
    fn test_convert_tool_result_content_text_only() {
        use rmcp::model::ContentBlock;
        let items = vec![ContentBlock::text("hello"), ContentBlock::text("world")];
        let blocks = convert_tool_result_content(&items);
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            crate::provider::ToolResultContent::Text { text } => {
                assert_eq!(text, "hello\nworld");
            }
            other => panic!("expected Text, got {:?}", other),
        }
    }

    #[test]
    fn test_convert_tool_result_content_image_passthrough() {
        use rmcp::model::ContentBlock;
        let data = base64_image(image::ImageFormat::Png);
        let items = vec![ContentBlock::image(data.clone(), "image/png")];
        let blocks = convert_tool_result_content(&items);
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            crate::provider::ToolResultContent::Image { source } => {
                assert_eq!(source.source_type, "base64");
                assert_eq!(source.media_type, "image/png");
                assert_eq!(source.data, data);
            }
            other => panic!("expected Image, got {:?}", other),
        }
    }

    /// The bug this whole path guards: a server that declares one format and sends another. The
    /// declared type must not reach the provider, which sniffs and answers 400.
    #[test]
    fn test_convert_tool_result_content_image_media_type_comes_from_bytes() {
        use rmcp::model::ContentBlock;
        let items = vec![ContentBlock::image(
            base64_image(image::ImageFormat::Jpeg),
            "image/png",
        )];
        let blocks = convert_tool_result_content(&items);
        match &blocks[0] {
            crate::provider::ToolResultContent::Image { source } => {
                assert_eq!(source.media_type, "image/jpeg");
            }
            other => panic!("expected Image, got {:?}", other),
        }
    }

    /// The allow-list applies to what the bytes actually are. BMP decodes fine but isn't a format
    /// we forward, so a real BMP is suppressed no matter what the server called it.
    #[test]
    fn test_convert_tool_result_content_image_rejects_disallowed_format() {
        use rmcp::model::ContentBlock;
        let items = vec![ContentBlock::image(
            base64_image(image::ImageFormat::Bmp),
            "image/png",
        )];
        let blocks = convert_tool_result_content(&items);
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            crate::provider::ToolResultContent::Text { text } => {
                assert!(text.contains("image suppressed"), "{}", text);
            }
            other => panic!("expected Text placeholder, got {:?}", other),
        }
    }

    #[test]
    fn test_convert_tool_result_content_image_rejects_non_image_bytes() {
        use rmcp::model::ContentBlock;
        let items = vec![ContentBlock::image("BASE64DATA", "image/svg+xml")];
        let blocks = convert_tool_result_content(&items);
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            crate::provider::ToolResultContent::Text { text } => {
                assert!(text.contains("image suppressed"));
                assert!(text.contains("image/svg+xml"));
            }
            other => panic!("expected Text placeholder, got {:?}", other),
        }
    }

    #[test]
    fn test_convert_tool_result_content_image_rejects_oversize() {
        use rmcp::model::ContentBlock;
        let oversized = "X".repeat(MAX_MCP_IMAGE_BYTES + 1);
        let items = vec![ContentBlock::image(oversized, "image/png")];
        let blocks = convert_tool_result_content(&items);
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            crate::provider::ToolResultContent::Text { text } => {
                assert!(text.contains("image suppressed"));
                assert!(text.contains("exceeds"));
            }
            other => panic!("expected Text placeholder, got {:?}", other),
        }
    }

    /// Between meka's own memory guard and the ceiling providers accept there used to be a band
    /// where an MCP image was forwarded purely so the provider could answer 400.
    #[test]
    fn test_convert_tool_result_content_image_rejects_over_the_provider_ceiling() {
        use rmcp::model::ContentBlock;
        // Comfortably over the provider ceiling, comfortably under meka's own cap.
        let base64_len = crate::image::MAX_IMAGE_RAW_BYTES / 3 * 4 + 4096;
        assert!(base64_len < MAX_MCP_IMAGE_BYTES, "must sit between the two");
        let items = vec![ContentBlock::image("A".repeat(base64_len), "image/png")];
        let blocks = convert_tool_result_content(&items);
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            crate::provider::ToolResultContent::Text { text } => {
                assert!(text.contains("image suppressed"), "{}", text);
                assert!(text.contains("providers accept"), "{}", text);
            }
            other => panic!("expected Text placeholder, got {:?}", other),
        }
    }

    #[test]
    fn test_convert_tool_result_content_mixed_keeps_ordering() {
        use rmcp::model::ContentBlock;
        let items = vec![
            ContentBlock::text("before"),
            ContentBlock::image(base64_image(image::ImageFormat::Png), "image/png"),
            ContentBlock::text("after"),
        ];
        let blocks = convert_tool_result_content(&items);
        assert_eq!(blocks.len(), 3);
        assert!(matches!(
            blocks[0],
            crate::provider::ToolResultContent::Text { .. }
        ));
        assert!(matches!(
            blocks[1],
            crate::provider::ToolResultContent::Image { .. }
        ));
        assert!(matches!(
            blocks[2],
            crate::provider::ToolResultContent::Text { .. }
        ));
    }
}
