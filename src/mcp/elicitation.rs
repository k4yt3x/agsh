//! MCP `elicitation/create` handling. When a server asks for user input (either a structured form
//! or a URL-consent flow), we route the request through the per-session
//! [`crate::frontend::Frontend`] that initiated the in-flight tool call so the right UI (REPL
//! prompt, ACP form, etc.) gets to drive the response. The correlation goes through
//! [`crate::mcp::progress::find_frontend_for_server`] because rmcp dispatches the handler on a
//! separately-spawned task (see `rmcp::service::spawn_service_task`), so a task-local on the
//! agent's caller task isn't visible here.
//!
//! When no in-flight call from the originating server is registered (the server somehow elicited
//! outside of a tool call, or the call's progress guard already dropped), the request is auto-
//! declined: the safe default, matching the pre-refactor "no shell sink installed" behaviour.

use rmcp::model::{ElicitResult, ElicitationAction};

/// User-facing payload the frontend renders.
#[derive(Debug)]
pub struct ElicitationPrompt {
    pub server_name: String,
    pub kind: ElicitationKind,
    pub message: String,
}

#[derive(Debug)]
pub enum ElicitationKind {
    /// Structured form: the server sent a JSON schema of fields to fill.
    Form { schema: serde_json::Value },
    /// URL consent: the server wants the user to visit a URL (e.g. to log in to a third-party
    /// service).
    Url { url: String },
}

/// Frontend's response back to the MCP handler.
#[derive(Debug, Clone)]
pub enum ElicitationResponse {
    Accept { content: Option<serde_json::Value> },
    Decline,
    Cancel,
}

impl ElicitationResponse {
    pub fn into_result(self) -> ElicitResult {
        match self {
            ElicitationResponse::Accept {
                content: Some(content),
            } => ElicitResult::new(ElicitationAction::Accept).with_content(content),
            ElicitationResponse::Accept { content: None } => {
                ElicitResult::new(ElicitationAction::Accept)
            }
            ElicitationResponse::Decline => ElicitResult::new(ElicitationAction::Decline),
            ElicitationResponse::Cancel => ElicitResult::new(ElicitationAction::Cancel),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decline_maps_to_decline_action() {
        match ElicitationResponse::Decline.into_result().action {
            ElicitationAction::Decline => {}
            other => panic!("expected Decline, got {:?}", other),
        }
    }

    #[test]
    fn accept_preserves_content() {
        let content = serde_json::json!({"k": "v"});
        let result = ElicitationResponse::Accept {
            content: Some(content.clone()),
        }
        .into_result();
        assert!(matches!(result.action, ElicitationAction::Accept));
        assert_eq!(result.content, Some(content));
    }
}
