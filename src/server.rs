//! `meka serve` subcommand entry point. Exposes the agent over HTTP+JSON for programmatic
//! clients (bots, scripts, web UIs). See the HTTP API docs for the full wire
//! specification; this module owns the implementation.
//!
//! The HTTP transport is a third [`crate::frontend::Frontend`] impl alongside
//! [`crate::repl::ReplFrontend`] and [`crate::acp::AcpFrontend`]. The agent core, MCP plumbing,
//! session DB, permission model, and tool dispatch are all reused unchanged.

pub(crate) mod auth;
pub(crate) mod errors;
pub(crate) mod gc;
pub(crate) mod handlers;
pub(crate) mod http_frontend;
pub(crate) mod idempotency;
pub(crate) mod openapi;
pub(crate) mod poisoned;
pub(crate) mod reattach;
pub(crate) mod schedule;
pub(crate) mod scope;
pub(crate) mod sse;
pub(crate) mod state;
pub(crate) mod webhook;

use std::sync::Arc;

use axum::{
    Router, middleware,
    response::IntoResponse,
    routing::{delete, get, patch, post, put},
};
use tower_http::{limit::RequestBodyLimitLayer, trace::TraceLayer};

use crate::{
    config::{ResolvedConfig, ResolvedServeConfig},
    mcp,
    server::{
        auth::AuthRegistry,
        errors::{ErrorKind, ProblemDetail},
        state::ServerState,
    },
    session::SessionManager,
};

/// Resolve the provider credential for `meka serve` from the active profile's database entry.
///
/// Debug-only: when the integration harness sets `MEKA_ACP_MOCK_PROVIDER=1`, `run_serve` swaps in a
/// scripted provider and discards the real one built from this credential, so a placeholder is
/// returned and the harness needn't seed a credential into the database.
async fn resolve_serve_credential(
    config: &ResolvedConfig,
    session_manager: &SessionManager,
) -> anyhow::Result<crate::provider::AuthCredential> {
    #[cfg(debug_assertions)]
    if std::env::var("MEKA_ACP_MOCK_PROVIDER").as_deref() == Ok("1") {
        return Ok(crate::provider::AuthCredential::ApiKey(
            "mock-acp-provider".to_string(),
        ));
    }

    match config.active_profile.as_deref() {
        Some(profile) => session_manager
            .token_store()
            .load_provider_credential(profile)
            .await?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "provider profile '{}' has no stored credential; run `meka provider login {}`",
                    profile,
                    profile
                )
            }),
        None => anyhow::bail!("meka serve requires a configured provider; run `meka provider add`"),
    }
}

/// Run meka as an HTTP server until the listener stops accepting (e.g. on SIGTERM after the
/// graceful-shutdown drain completes). The signature mirrors [`crate::acp::run_acp`] for
/// consistency with the existing dispatch in `main::async_main`.
pub async fn run_serve(
    mut config: ResolvedConfig,
    session_manager: SessionManager,
    mcp_manager: Option<Arc<mcp::McpClientManager>>,
    mcp_context: Arc<mcp::McpClientContext>,
) -> anyhow::Result<()> {
    let mut serve = ResolvedServeConfig::resolve(config.serve.take())
        .map_err(|error| anyhow::anyhow!("invalid [serve] config: {}", error))?;
    if let Some(bind_override) = config.serve_bind_override.take() {
        serve.bind = bind_override;
    }
    if serve.tokens.is_empty() {
        anyhow::bail!(
            "[serve] is configured but has no tokens; add at least one `[[serve.tokens]]` \
             entry with `scopes` so callers can authenticate"
        );
    }
    for token in &serve.tokens {
        if matches!(token.source, crate::config::TokenSource::Inline) {
            tracing::warn!(
                description = token.description.as_deref().unwrap_or("(no description)"),
                "inline plaintext token configured; prefer ${{ENV_VAR}} or token_file for production",
            );
        }
    }

    let max_body_bytes = serve.max_body_bytes;
    let bind_addr = serve.bind.clone();

    let credential = resolve_serve_credential(&config, &session_manager).await?;

    let shared = Arc::new(
        crate::build_shared_deps(
            config,
            session_manager,
            credential,
            mcp_manager,
            mcp_context,
        )
        .await?,
    );

    #[cfg(debug_assertions)]
    let shared = if std::env::var("MEKA_ACP_MOCK_PROVIDER").as_deref() == Ok("1") {
        let rounds = crate::provider::mock::load_script_from_env()
            .map_err(|error| anyhow::anyhow!("load mock provider script: {}", error))?
            .unwrap_or_default();
        let mock = Arc::new(crate::provider::mock::MockProvider::from_rounds(rounds));
        let new_inner = crate::SharedDeps {
            provider: mock,
            ..(*shared).clone()
        };
        tracing::info!("MEKA_ACP_MOCK_PROVIDER=1: using scripted mock provider");
        Arc::new(new_inner)
    } else {
        shared
    };

    if !serve.webhooks.is_empty() {
        // Named at startup because an outbound request is the one thing meka does that leaves the
        // machine unprompted, and an operator inheriting a config should not have to read it to
        // find out that it does.
        tracing::info!(
            "{} webhook endpoint(s) configured: {}",
            serve.webhooks.len(),
            serve
                .webhooks
                .iter()
                .map(|hook| format!("{} ({})", hook.url, hook.events.join(", ")))
                .collect::<Vec<_>>()
                .join("; ")
        );
    }

    let auth = AuthRegistry::new(serve.tokens.clone());
    let serve_arc = Arc::new(serve);
    let idempotency_cache = idempotency::IdempotencyCache::standard();
    let shutdown_drain_timeout = serve_arc.shutdown_drain_timeout;
    let state = ServerState::new(shared, serve_arc, idempotency_cache.clone());

    let gc_handle = gc::spawn(state.clone());
    let scheduler_handle = schedule::spawn(state.clone());
    let background_handle = schedule::spawn_background_poller(state.clone());
    let pruner_handle = idempotency_cache.spawn_pruner();

    let router = build_router(state.clone(), auth, max_body_bytes);

    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .map_err(|error| anyhow::anyhow!("failed to bind {}: {}", bind_addr, error))?;
    let local = listener.local_addr()?;
    tracing::info!("listening on {}", local);

    // The timeout wraps only the post-signal drain, not the entire serve future.
    // Wrapping the whole future would start the timer at construction, causing the
    // server to exit after `shutdown_drain_timeout` of total uptime.
    let (drain_tx, drain_rx) = tokio::sync::oneshot::channel::<()>();
    let serve_future = axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            // Signal-watch + drain orchestration runs outside this closure so the
            // timeout can wrap it independently of the accept loop's lifetime.
            let _ = drain_rx.await;
        })
        .into_future();
    let serve_handle = tokio::spawn(serve_future);

    shutdown_signal().await;
    state.shutdown.cancel();
    drain_active_sessions(&state).await;
    let _ = drain_tx.send(());

    // The drain waits for the turns as well as for the accept loop. A turn runs on a task the
    // handler spawns rather than inside the handler itself, and `stream_reattach_grace` exists to
    // keep one running with no client attached, so axum's graceful shutdown finds no in-flight
    // request to wait for and returns while the work is still going. Awaiting only that was
    // therefore a drain in name: `handlers::turn` documents at length what a turn dropped
    // mid-flight costs (an orphaned process group, an assistant `tool_use` whose result never
    // lands), and every one of those was still on the table at shutdown.
    let drain_result = tokio::time::timeout(shutdown_drain_timeout, async {
        let (join_result, ()) = tokio::join!(serve_handle, wait_for_turns_to_unwind(&state));
        join_result
    })
    .await;
    gc_handle.abort();
    scheduler_handle.abort();
    background_handle.abort();
    pruner_handle.abort();
    // Flush the SQLite WAL before exit so a quick restart doesn't pay WAL-replay cost.
    // Best-effort, SQLite recovers from an unflushed WAL automatically.
    if let Err(error) = state.shared.session_manager.checkpoint().await {
        tracing::warn!("WAL checkpoint on shutdown failed: {}", error);
    } else {
        tracing::info!("WAL checkpoint complete");
    }
    match drain_result {
        Ok(join_result) => join_result
            .map_err(|error| anyhow::anyhow!("server task panicked: {}", error))?
            .map_err(|error| anyhow::anyhow!("server error: {}", error))?,
        Err(_elapsed) => {
            tracing::warn!(
                "drain exceeded {}s, forcing exit",
                shutdown_drain_timeout.as_secs()
            );
            // Non-zero exit so systemd / container orchestrators can distinguish forced
            // abort from a clean drain. Same semantics as `meka acp`.
            std::process::exit(1);
        }
    }
    Ok(())
}

/// Fire every session's cancellation token during a graceful drain.
///
/// This is now the only thing that stops an in-flight turn on shutdown. The streaming handler used
/// to watch `state.shutdown` in its own `select!` as well, which was redundant with this and has
/// been dropped; the turn task still reads the shutdown token, but only to label its terminal event
/// `server_shutdown` rather than `client`.
async fn drain_active_sessions(state: &ServerState) {
    let sessions = state.sessions.read().await;
    for entry in sessions.values() {
        let token =
            crate::server::poisoned::read(&entry.cancellation, "drain::session_cancel").clone();
        token.cancel();
    }
}

/// Resolve once every turn this process is running has finished unwinding.
///
/// Cancelling a turn is not the same as waiting for one: the token stops the agent at its next
/// check, and what follows is the commit of the partial assistant message, the tool result the
/// round already produced, and the frontend teardown. That tail is what a drain exists to protect,
/// and it is measured in database round-trips, not instants.
///
/// Both counters are consulted because neither covers everything. The process-wide one still counts
/// a client turn whose session has since been evicted from the map. The per-session one counts the
/// work that never takes a [`crate::server::state::TurnGuard`]: a scheduled fire, a
/// background-outcome delivery, a compaction or rewind. The latter run on the scheduler and poller
/// tasks that the caller aborts as soon as this returns, so leaving them out would abandon
/// precisely the unattended turns nobody is watching.
async fn wait_for_turns_to_unwind(state: &ServerState) {
    loop {
        let idle = state
            .concurrent_turns
            .load(std::sync::atomic::Ordering::Acquire)
            == 0
            && {
                let sessions = state.sessions.read().await;
                sessions
                    .values()
                    .all(|entry| entry.in_flight.load(std::sync::atomic::Ordering::Acquire) == 0)
            };
        if idle {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
}

fn build_router(state: ServerState, auth: AuthRegistry, max_body_bytes: usize) -> Router {
    let authenticated = Router::new()
        .route("/v1/sessions", post(handlers::sessions::create_session))
        .route("/v1/sessions", get(handlers::sessions::list_sessions))
        .route("/v1/sessions/{id}", get(handlers::sessions::get_session))
        .route(
            "/v1/sessions/{id}",
            patch(handlers::sessions::patch_session),
        )
        .route(
            "/v1/sessions/{id}",
            delete(handlers::sessions::delete_session),
        )
        .route(
            "/v1/sessions/{id}/messages",
            get(handlers::messages::list_messages),
        )
        .route(
            "/v1/sessions/{id}/fork",
            post(handlers::sessions::fork_session),
        )
        .route("/v1/sessions/{id}/turn", post(handlers::turn::submit_turn))
        .route(
            "/v1/sessions/{id}/cancel",
            post(handlers::turn::cancel_turn),
        )
        .route(
            "/v1/sessions/{id}/stream",
            get(handlers::turn::stream_turn),
        )
        .route(
            "/v1/sessions/{id}/responses/{request_id}",
            post(handlers::responses::respond),
        )
        // Conversation-shaping operations. `/v1/sessions/import` is a static segment, so matchit
        // prefers it over `/v1/sessions/{id}` regardless of registration order.
        .route(
            "/v1/sessions/import",
            post(handlers::conversation::import),
        )
        .route(
            "/v1/sessions/{id}/compact",
            post(handlers::conversation::compact),
        )
        .route(
            "/v1/sessions/{id}/context",
            get(handlers::conversation::context),
        )
        .route(
            "/v1/sessions/{id}/rewind",
            post(handlers::conversation::rewind),
        )
        .route(
            "/v1/sessions/{id}/export",
            get(handlers::conversation::export),
        )
        .route("/v1/sessions/{id}/tasks", get(handlers::jobs::list_tasks))
        .route(
            "/v1/sessions/{id}/tasks/{task_id}",
            delete(handlers::jobs::cancel_task),
        )
        .route("/v1/schedule", get(handlers::jobs::list_all))
        .route("/v1/schedule/{job_id}", delete(handlers::jobs::cancel))
        .route(
            "/v1/sessions/{id}/schedule",
            get(handlers::jobs::list_for_session),
        )
        .route(
            "/v1/sessions/{id}/schedule",
            post(handlers::jobs::create),
        )
        .route("/v1/sessions/{id}/tools", get(handlers::stores::list_tools))
        .route("/v1/info", get(handlers::info::info))
        .route("/v1/skills", get(handlers::info::skills))
        .route("/v1/skills/{name}", get(handlers::stores::get_skill))
        .route("/v1/skills/{name}", put(handlers::stores::put_skill))
        .route("/v1/skills/{name}", delete(handlers::stores::delete_skill))
        .route("/v1/memory", get(handlers::stores::list_memories))
        .route("/v1/memory/{name}", get(handlers::stores::get_memory))
        .route("/v1/memory/{name}", put(handlers::stores::put_memory))
        .route("/v1/memory/{name}", delete(handlers::stores::delete_memory))
        .route("/v1/instructions", get(handlers::stores::instructions))
        .route("/v1/providers", get(handlers::stores::providers))
        .route("/v1/mcp", get(handlers::info::mcp))
        .route("/v1/mcp/{name}/tools", get(handlers::info::mcp_tools))
        .route(
            "/v1/mcp/{name}/reconnect",
            post(handlers::info::mcp_reconnect),
        )
        .layer(middleware::from_fn_with_state(
            auth.clone(),
            crate::server::auth::bearer_auth,
        ));

    let public = Router::new()
        .route("/v1/health/live", get(handlers::discovery::live))
        .route("/v1/health/ready", get(handlers::discovery::ready));

    // `/v1/docs` and `/v1/openapi.json` are the only unauthenticated routes that describe the
    // deployment rather than report on it, so they are opt-in; see `[serve].docs`.
    let documentation = if state.config.docs {
        openapi::router()
    } else {
        Router::new()
    };

    authenticated
        .merge(public)
        .merge(documentation)
        // `RequestBodyLimitLayer` is the only authority on body size. Without disabling axum's
        // own default, the `Bytes` extractor every handler uses applies a 2 MiB cap of its own, so
        // any `max_body_bytes` above that was silently inert -- and the 413 this middleware
        // rewrites would name a limit that had not fired. The docs tell operators to raise
        // `max_body_bytes` for multi-image turns, which only works now.
        .layer(axum::extract::DefaultBodyLimit::disable())
        .layer(RequestBodyLimitLayer::new(max_body_bytes))
        .layer(middleware::from_fn_with_state(
            max_body_bytes,
            rewrite_payload_too_large,
        ))
        .layer(middleware::from_fn(rewrite_plain_bad_request))
        .layer(middleware::from_fn(inject_problem_instance))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Convert tower-http's plain-text 413 response to a Problem Detail. Runs as a middleware so the
/// rewrite happens once for every layered route, handlers themselves never produce 413, so any
/// 413 the middleware observes came from the body-limit layer.
async fn rewrite_payload_too_large(
    axum::extract::State(max_body_bytes): axum::extract::State<usize>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let path = request.uri().path().to_string();
    let response = next.run(request).await;
    if response.status() != axum::http::StatusCode::PAYLOAD_TOO_LARGE {
        return response;
    }
    // Don't double-wrap if a handler somehow returned 413 with the spec content-type already.
    let already_problem = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("application/problem+json"));
    if already_problem {
        return response;
    }
    ProblemDetail::new(
        ErrorKind::PayloadTooLarge,
        axum::http::StatusCode::PAYLOAD_TOO_LARGE,
        format!(
            "request body exceeds the configured limit of {} bytes",
            max_body_bytes,
        ),
    )
    .instance(path)
    .with("max_body_bytes", serde_json::Value::from(max_body_bytes))
    .into_response()
}

/// Convert axum's plain-text extractor rejections into Problem Details.
///
/// A malformed path segment or query parameter is rejected by the `Path` / `Query` extractor before
/// any handler runs, and axum answers `400 text/plain`. Every other error on this surface is RFC
/// 9457, so a client that parses `application/problem+json` had one response shape it could not
/// read, for the most ordinary mistake there is. Handled as middleware for the same reason as the
/// 413 rewrite: the alternative is a custom rejection type threaded through every handler
/// signature.
async fn rewrite_plain_bad_request(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let path = request.uri().path().to_string();
    let response = next.run(request).await;
    if response.status() != axum::http::StatusCode::BAD_REQUEST {
        return response;
    }
    let is_plain = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_none_or(|value| value.starts_with("text/plain"));
    if !is_plain {
        return response;
    }

    // The rejection text names which parameter failed and why, which is exactly what the caller
    // needs; it is generated by axum from the route definition, not from anything the caller sent.
    let (_, body) = response.into_parts();
    let detail = match axum::body::to_bytes(body, 4096).await {
        Ok(bytes) => String::from_utf8_lossy(&bytes).trim().to_string(),
        Err(_) => String::new(),
    };
    let detail = if detail.is_empty() {
        "invalid path or query parameter".to_string()
    } else {
        detail
    };
    ProblemDetail::new(
        ErrorKind::InvalidBody,
        axum::http::StatusCode::BAD_REQUEST,
        detail,
    )
    .instance(path)
    .into_response()
}

/// Inject RFC 9457's `instance` member into every Problem Detail response body that doesn't
/// already have one. Handled as middleware rather than per-handler to avoid threading a
/// `RequestPath` extractor through every error site.
async fn inject_problem_instance(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let path = request.uri().path().to_string();
    let response = next.run(request).await;
    let is_problem = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("application/problem+json"));
    if !is_problem {
        return response;
    }
    let (mut parts, body) = response.into_parts();
    // Strip the stale Content-Length so hyper recomputes it for the rewritten body,
    // which is longer than the original due to the injected `instance` field.
    parts.headers.remove(axum::http::header::CONTENT_LENGTH);
    // Problem Details are sub-KB in practice; the 64 KB cap is a safety net.
    // On failure (body exceeds the limit or the stream errors), we return
    // the status + headers with an empty body; the original stream is already
    // consumed and can't be replayed. This is acceptable because meka never
    // produces a Problem Detail anywhere near this size.
    const PROBLEM_DETAIL_BUFFER_LIMIT: usize = 64 * 1024;
    let bytes = match axum::body::to_bytes(body, PROBLEM_DETAIL_BUFFER_LIMIT).await {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::warn!("failed to buffer a problem detail response body: {}", error);
            return axum::response::Response::from_parts(parts, axum::body::Body::empty());
        }
    };
    let mut value: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(_) => {
            // Body claimed `application/problem+json` but isn't valid JSON; pass through
            // untouched rather than mangle the response.
            return axum::response::Response::from_parts(parts, axum::body::Body::from(bytes));
        }
    };
    if let Some(object) = value.as_object_mut()
        && !object.contains_key("instance")
    {
        object.insert("instance".to_string(), serde_json::Value::String(path));
    }
    let rewritten = match serde_json::to_vec(&value) {
        Ok(bytes) => axum::body::Body::from(bytes),
        Err(error) => {
            tracing::warn!(
                "failed to re-serialize a problem detail response body: {}",
                error
            );
            axum::body::Body::from(bytes)
        }
    };
    axum::response::Response::from_parts(parts, rewritten)
}

/// Wait for SIGTERM / SIGINT, then return so `axum::serve(...).with_graceful_shutdown(...)`
/// can begin draining. On Windows, only Ctrl+C is observed.
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut term =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(stream) => stream,
                Err(error) => {
                    tracing::warn!(
                        "failed to install SIGTERM handler: {}; relying on Ctrl+C only",
                        error
                    );
                    let _ = tokio::signal::ctrl_c().await;
                    return;
                }
            };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => tracing::info!("SIGINT received, draining"),
            _ = term.recv() => tracing::info!("SIGTERM received, draining"),
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("Ctrl+C received, draining");
    }
}
