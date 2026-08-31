//! Plumbing for forwarding MCP `notifications/progress` updates from servers up into the active
//! frontend. The MCP protocol requires the client to announce a per-request `progressToken`; the
//! server then emits progress notifications carrying that token. We keep a process-wide map from
//! token → (entry, frontend) so the rmcp notification dispatch (which runs on a separately-spawned
//! task; see `rmcp::service::spawn_service_task`) can find the per-session UI even though it
//! can't read the caller's task-local. RAII guards remove the entry when the in-flight call
//! finishes so orphaned tokens don't pile up.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use rmcp::model::{NumberOrString, ProgressNotificationParam, ProgressToken};

use crate::frontend::{Frontend, FrontendEvent};

/// Single progress update forwarded to the frontend.
#[derive(Clone, Debug)]
pub struct ProgressUpdate {
    pub server_name: String,
    pub tool_name: String,
    /// The provider's `tool_use_id` for the in-flight call, when one was supplied. Reserved for a
    /// future renderer that correlates progress lines to the tool-use entry the LLM is waiting on.
    /// Not currently consumed by the shell, but always populated at the emit site so downstream
    /// code doesn't need a plumbing migration later.
    #[allow(dead_code)]
    pub tool_use_id: Option<String>,
    pub progress: f64,
    pub total: Option<f64>,
    pub message: Option<String>,
}

/// Process-wide progress registry. Per-call entries are keyed by progress token; the entry also
/// owns the per-session frontend so the dispatch path can route events to the correct UI without
/// relying on task-local propagation (which doesn't survive rmcp's spawned-handler tasks).
struct Registry {
    entries: Mutex<HashMap<String, Entry>>,
}

struct Entry {
    server_name: String,
    tool_name: String,
    tool_use_id: Option<String>,
    /// The session's frontend that initiated this tool call. `None` outside an agent-driven call
    /// site (e.g. an MCP connection probe). When `None`, `dispatch` falls back to a tracing log
    /// and the user sees nothing, same as the pre-refactor "no UI sink installed" behaviour.
    frontend: Option<Arc<dyn Frontend>>,
}

fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(|| Registry {
        entries: Mutex::new(HashMap::new()),
    })
}

/// Register a freshly-generated progress token for an in-flight tool call. The frontend is sourced
/// at the agent's call site (where the per-session task-local is in scope) and stored alongside
/// the call context so the rmcp notification handler can later look it up by token. Returns a
/// [`ProgressGuard`] that removes the entry when dropped, the standard `_progress_guard` binding
/// the MCP tool adapter holds for the duration of `peer.call_tool().await`.
pub fn register(
    server_name: String,
    tool_name: String,
    tool_use_id: Option<String>,
    frontend: Option<Arc<dyn Frontend>>,
) -> (ProgressToken, ProgressGuard) {
    let token_str = format!("meka-{}", uuid::Uuid::new_v4());
    let token = ProgressToken(NumberOrString::String(token_str.clone().into()));
    registry()
        .entries
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(token_str.clone(), Entry {
            server_name,
            tool_name,
            tool_use_id,
            frontend,
        });
    (token, ProgressGuard {
        key: Some(token_str),
    })
}

/// RAII guard that removes the progress-token entry when dropped.
pub struct ProgressGuard {
    key: Option<String>,
}

impl Drop for ProgressGuard {
    fn drop(&mut self) {
        if let Some(key) = self.key.take()
            && let Ok(mut entries) = registry().entries.lock()
        {
            entries.remove(&key);
        }
    }
}

/// Called from `MekaClientHandler::on_progress` (the `ClientHandler` trait impl in
/// `src/mcp/handler.rs`). Looks up the registered context by token, forwards the update to the
/// per-session frontend, and emits an `info!` log line for the off-frontend audience (e.g. a tail
/// on the meka stderr).
pub async fn dispatch(params: ProgressNotificationParam) {
    let key = match &params.progress_token.0 {
        NumberOrString::String(s) => s.to_string(),
        NumberOrString::Number(n) => n.to_string(),
    };
    let snapshot = {
        let entries = registry()
            .entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        entries.get(&key).map(|e| {
            (
                e.server_name.clone(),
                e.tool_name.clone(),
                e.tool_use_id.clone(),
                e.frontend.clone(),
            )
        })
    };
    let Some((server_name, tool_name, tool_use_id, frontend)) = snapshot else {
        tracing::debug!(
            "MCP progress for unknown token '{}' (likely cancelled); ignored",
            key
        );
        return;
    };
    let update = ProgressUpdate {
        server_name,
        tool_name,
        tool_use_id,
        progress: params.progress,
        total: params.total,
        message: params.message,
    };
    tracing::info!(
        "MCP '{}' {} progress: {}{}{}",
        update.server_name,
        update.tool_name,
        update.progress,
        update.total.map(|t| format!("/{}", t)).unwrap_or_default(),
        update
            .message
            .as_deref()
            .map(|m| format!(", {}", m))
            .unwrap_or_default()
    );
    if let Some(frontend) = frontend {
        frontend.emit(FrontendEvent::McpProgress(update)).await;
    }
}

/// Best-effort lookup: find the frontend of any in-flight tool call targeting `server_name`. Used
/// by the elicitation handler, which has no `progressToken` correlation of its own. The server's
/// elicitation request lands on the rmcp handler task with only its own request id and the
/// originating server identity. Scanning the registry for a matching in-flight call is the
/// pragmatic best we can do without protocol-level help.
///
/// Returns the first match (HashMap iteration order is arbitrary). In a multi-session ACP process
/// where two sessions race calls to the same server, an elicitation arriving during both calls
/// routes to whichever entry the scan picks. `AcpFrontend` issues a real `elicitation/create` on
/// its own connection, so a mis-pick surfaces the prompt in the wrong session's editor. Narrowing
/// it needs protocol-level help: MCP's elicitation is a server-initiated request with no link back
/// to the tool call that provoked it, so there is nothing here to correlate on.
pub(crate) fn find_frontend_for_server(server_name: &str) -> Option<Arc<dyn Frontend>> {
    registry()
        .entries
        .lock()
        .ok()?
        .values()
        .find(|entry| entry.server_name == server_name)
        .and_then(|entry| entry.frontend.clone())
}

/// Test helper: check whether a specific progress-token key is in the registry. Race-free against
/// concurrent tests because the key is a UUID only the caller knows.
#[cfg(test)]
pub(crate) fn is_registered(key: &str) -> bool {
    registry()
        .entries
        .lock()
        .map(|e| e.contains_key(key))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::frontend::{Frontend, FrontendEvent, testing::RecordingFrontend};

    #[test]
    fn guard_cleans_up_on_drop() {
        // Cargo runs unit tests in parallel and the registry is a shared global. Checking
        // `outstanding_count()` deltas races with other tests registering/dropping tokens
        // concurrently. Scan for this specific token's UUID key instead, which only this test
        // knows about.
        let token_key: String;
        {
            let (token, _guard) = register("srv".into(), "tool".into(), None, None);
            token_key = match &token.0 {
                NumberOrString::String(s) => s.to_string(),
                NumberOrString::Number(n) => n.to_string(),
            };
            assert!(
                is_registered(&token_key),
                "token missing from registry while guard is still alive: {}",
                token_key
            );
        }
        assert!(
            !is_registered(&token_key),
            "token '{}' lingered after guard dropped: ProgressGuard::drop didn't clean up",
            token_key
        );
    }

    #[tokio::test]
    async fn dispatch_without_registration_is_noop() {
        dispatch(ProgressNotificationParam::new(
            ProgressToken(NumberOrString::String(Arc::from("unknown"))),
            1.0,
        ))
        .await;
        // If it didn't panic we're fine.
    }

    /// The new contract: progress dispatch routes through the per-call frontend stored on the
    /// registry entry, not a process-global sink. This proves session A's frontend can receive an
    /// update while a hypothetical session B running the same test concurrently doesn't.
    #[tokio::test]
    async fn dispatch_forwards_to_per_call_frontend() {
        let recorder: Arc<RecordingFrontend> = Arc::new(RecordingFrontend::new());
        let frontend: Arc<dyn Frontend> = recorder.clone();

        let (token, _guard) = register(
            "srv".into(),
            "tool".into(),
            Some("tu-1".into()),
            Some(frontend),
        );
        let key = match &token.0 {
            NumberOrString::String(s) => s.to_string(),
            NumberOrString::Number(n) => n.to_string(),
        };

        dispatch(
            ProgressNotificationParam::new(
                ProgressToken(NumberOrString::String(Arc::from(key))),
                0.5,
            )
            .with_total(1.0)
            .with_message("halfway"),
        )
        .await;

        let events = recorder.events();
        let update = events
            .iter()
            .filter_map(|event| match event {
                FrontendEvent::McpProgress(update) => Some(update),
                _ => None,
            })
            .next_back()
            .expect("frontend should have received exactly one McpProgress");
        assert_eq!(update.tool_use_id.as_deref(), Some("tu-1"));
        assert_eq!(update.progress, 0.5);
        assert_eq!(update.message.as_deref(), Some("halfway"));
    }

    /// Multi-session isolation: two concurrent calls with distinct frontends must each receive
    /// only their own progress updates. This is the property that makes per-session ACP routing
    /// safe: session A's MCP server progress can't leak to session B.
    #[tokio::test]
    async fn dispatch_isolates_concurrent_calls() {
        let recorder_a: Arc<RecordingFrontend> = Arc::new(RecordingFrontend::new());
        let recorder_b: Arc<RecordingFrontend> = Arc::new(RecordingFrontend::new());
        let frontend_a: Arc<dyn Frontend> = recorder_a.clone();
        let frontend_b: Arc<dyn Frontend> = recorder_b.clone();

        let (token_a, _guard_a) = register(
            "srv-a".into(),
            "tool-a".into(),
            Some("ta".into()),
            Some(frontend_a),
        );
        let (token_b, _guard_b) = register(
            "srv-b".into(),
            "tool-b".into(),
            Some("tb".into()),
            Some(frontend_b),
        );

        let key_a = match &token_a.0 {
            NumberOrString::String(s) => s.to_string(),
            NumberOrString::Number(n) => n.to_string(),
        };
        let key_b = match &token_b.0 {
            NumberOrString::String(s) => s.to_string(),
            NumberOrString::Number(n) => n.to_string(),
        };

        dispatch(
            ProgressNotificationParam::new(
                ProgressToken(NumberOrString::String(Arc::from(key_a))),
                1.0,
            )
            .with_message("for-a"),
        )
        .await;
        dispatch(
            ProgressNotificationParam::new(
                ProgressToken(NumberOrString::String(Arc::from(key_b))),
                2.0,
            )
            .with_message("for-b"),
        )
        .await;

        let events_a = recorder_a.events();
        let events_b = recorder_b.events();

        // A received exactly one progress event for itself.
        let a_progress: Vec<_> = events_a
            .iter()
            .filter_map(|event| match event {
                FrontendEvent::McpProgress(update) => Some(update.message.as_deref()),
                _ => None,
            })
            .collect();
        assert_eq!(
            a_progress,
            vec![Some("for-a")],
            "frontend A must receive only its own update"
        );

        let b_progress: Vec<_> = events_b
            .iter()
            .filter_map(|event| match event {
                FrontendEvent::McpProgress(update) => Some(update.message.as_deref()),
                _ => None,
            })
            .collect();
        assert_eq!(
            b_progress,
            vec![Some("for-b")],
            "frontend B must receive only its own update"
        );
    }

    /// `find_frontend_for_server` returns the frontend from any in-flight entry that matches the
    /// server name. Used by the elicitation handler when it can't correlate via the progress
    /// token.
    #[test]
    fn find_frontend_for_server_returns_matching_entry() {
        let recorder: Arc<RecordingFrontend> = Arc::new(RecordingFrontend::new());
        let frontend: Arc<dyn Frontend> = recorder.clone();
        let (_token, _guard) = register(
            "unique-srv-for-test".into(),
            "tool".into(),
            None,
            Some(frontend),
        );
        let found =
            find_frontend_for_server("unique-srv-for-test").expect("entry should be findable");
        // We can't easily compare `Arc<dyn Frontend>` for identity, but we *can* observe that
        // emitting through `found` lands on the recorder we registered with.
        let runtime = tokio::runtime::Runtime::new().expect("runtime");
        runtime.block_on(async {
            found
                .emit(FrontendEvent::Notice(crate::provider::Notice::info(
                    "probe",
                )))
                .await;
        });
        assert!(
            recorder
                .events()
                .iter()
                .any(|e| matches!(e, FrontendEvent::Notice(n) if n.text == "probe")),
            "the found frontend must be the one originally registered",
        );
    }
}
