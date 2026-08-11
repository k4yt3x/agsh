//! Health-probe endpoints: `/v1/health/live` (process-up) and `/v1/health/ready`
//! (subsystems-healthy). The other discovery endpoints (`/v1/info`, `/v1/skills`, `/v1/mcp`)
//! live in [`super::info`].

use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};
use serde::Serialize;
use utoipa::ToSchema;

use crate::server::state::ServerState;

#[derive(Serialize, ToSchema)]
pub struct LiveResponse {
    pub status: String,
}

/// Liveness probe. Always returns 200 if the process is up and the listener is accepting; the
/// response handler running at all is sufficient proof. Used by Kubernetes / systemd to distinguish
/// "process crashed" from "process degraded".
#[utoipa::path(
    get,
    path = "/v1/health/live",
    tag = "health",
    responses((status = 200, description = "Process is up and accepting connections", body = LiveResponse))
)]
pub async fn live() -> Json<LiveResponse> {
    Json(LiveResponse {
        status: "ok".to_string(),
    })
}

#[derive(Serialize, ToSchema)]
pub struct ReadyResponse {
    pub status: String,
    /// Per-subsystem readiness flags. `false` here means the subsystem is in a state that
    /// would fail a real request; a `503` is returned in that case.
    pub session_db: bool,
    pub provider_configured: bool,
    /// `true` unless some MCP server marked `required` is stuck in `Failed`. Optional servers
    /// don't count: they can't stop a turn, so they can't make the instance unready either.
    /// Per-server detail (names, connection states) is available via `GET /v1/mcp`
    /// (requires auth). It is deliberately omitted here because `/v1/health/ready` is
    /// unauthenticated and server names leak infrastructure topology.
    pub mcp_servers_healthy: bool,
}

/// Whether any MCP server that gates turns has failed to connect.
///
/// Only a `required` server counts, matching the turn gate
/// (`crate::agent::gate_on_required_servers`). A failed optional server doesn't stop turns, so
/// reporting 503 for one would pull a perfectly serviceable deployment out of rotation over a
/// capability it was explicitly told it could run without: exactly the container-missing-the-binary
/// case `required` exists to allow. `Pending` doesn't count either, since it resolves on its own
/// and a probe that flaps during startup is worse than one that waits.
fn any_required_server_failed(not_connected: &[crate::mcp::NotConnected]) -> bool {
    not_connected.iter().any(|server| {
        server.required && matches!(server.state, crate::mcp::ServerState::Failed { .. })
    })
}

/// Readiness probe. Returns 200 iff the server is in a state where new turn requests can
/// reasonably be expected to succeed (session DB queryable, provider configured, no *required* MCP
/// server stuck in `Failed`). Returns 503 with a body that names which subsystem is the blocker.
#[utoipa::path(
    get,
    path = "/v1/health/ready",
    tag = "health",
    responses(
        (status = 200, description = "All dependencies healthy", body = ReadyResponse),
        (status = 503, description = "One or more subsystems unavailable", body = ReadyResponse),
    )
)]
pub async fn ready(State(state): State<ServerState>) -> impl IntoResponse {
    // Touch the session DB with a cheap read. `session_exists(nil_uuid)` runs one statement
    // and returns Ok(false); any error means the connection is broken / DB is gone.
    let session_db = state
        .shared
        .session_manager
        .session_exists(uuid::Uuid::nil())
        .await
        .is_ok();
    let provider_configured = state.shared.config.provider_name.is_some();

    let mcp_healthy = match state.shared.mcp_manager.as_ref() {
        Some(manager) => !any_required_server_failed(&manager.enabled_not_connected().await),
        None => true,
    };

    let healthy = session_db && provider_configured && mcp_healthy;
    let body = ReadyResponse {
        status: if healthy {
            "ok".to_string()
        } else {
            "degraded".to_string()
        },
        session_db,
        provider_configured,
        mcp_servers_healthy: mcp_healthy,
    };
    let status = if healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(body))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::{NotConnected, ServerState};

    fn failed(name: &str, required: bool) -> NotConnected {
        NotConnected {
            name: name.to_string(),
            required,
            state: ServerState::Failed {
                error: "boom".to_string(),
                at: std::time::Instant::now(),
            },
        }
    }

    /// The point of `required`: a container without an optional server's binary still serves. If
    /// this regressed, a readiness probe would pull the deployment for a capability it was told it
    /// could run without.
    #[test]
    fn optional_failures_do_not_affect_readiness() {
        assert!(!any_required_server_failed(&[]));
        assert!(!any_required_server_failed(&[
            failed("ida", false),
            failed("exa", false)
        ]));
    }

    #[test]
    fn a_failed_required_server_is_unready() {
        assert!(any_required_server_failed(&[
            failed("ida", false),
            failed("bridge", true)
        ]));
    }

    /// Pending resolves on its own; treating it as unready makes the probe flap during startup.
    #[test]
    fn a_pending_required_server_is_still_ready() {
        assert!(!any_required_server_failed(&[NotConnected {
            name: "bridge".to_string(),
            required: true,
            state: ServerState::Pending,
        }]));
    }
}
