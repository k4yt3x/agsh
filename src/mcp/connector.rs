//! Background connector that drives `Pending` MCP server entries through initial handshake + tool
//! discovery + registration. Split into a stdio stream and an HTTP stream, each bounded by its own
//! concurrency cap.
//!
//! Entries that fail their initial connect are retried here in the background until they come up;
//! see [`retry_until_connected`].

use std::{sync::Arc, time::Duration};

use super::{
    MAX_MCP_DESCRIPTION_LENGTH, McpClientContext, McpClientManager, McpRunningService,
    McpRuntimeConfig, ServerEntry, ServerState,
    handler::{McpToolAdapter, MekaClientHandler},
    resolve_tool_permission, tool_is_allowed,
    transport::{build_http_transport_config, build_stdio_command},
    truncate, warn_on_stale_tool_config,
};
use crate::{
    config::{McpServerConfig, McpTransport},
    error::{MekaError, Result},
    permission::Permission,
    session::TokenStore,
};

/// First delay after a failed initial connect. Doubles per attempt up to [`MAX_RETRY_BACKOFF`].
const INITIAL_RETRY_BACKOFF: Duration = Duration::from_secs(5);

/// Ceiling on the retry delay. A server that is down for a long time is polled every five minutes
/// rather than abandoned, because the alternative is a meka that stays broken until someone
/// restarts it.
const MAX_RETRY_BACKOFF: Duration = Duration::from_secs(300);

/// Drive the actual connect work for every `Pending` entry, split into a stdio stream and an HTTP
/// stream, each bounded by its own concurrency cap. Runs in a spawned task so
/// [`super::McpClientManager::start_connector`] can return immediately and the REPL paints without
/// waiting.
///
/// When both streams drain, flips the `settled` watch so the turn gate can short-circuit, then
/// hands any entry that failed to [`retry_until_connected`]. Settling first is deliberate: the
/// grace wait in `Agent::await_mcp_ready` must not be held open by a server that is going to take
/// minutes to appear.
pub(super) async fn run_connector(
    pending: Vec<Arc<ServerEntry>>,
    manager: Arc<McpClientManager>,
    mcp_default_permission: Option<Permission>,
    runtime: McpRuntimeConfig,
    settled: tokio::sync::watch::Sender<bool>,
) {
    use futures::StreamExt;

    if pending.is_empty() {
        let _ = settled.send(true);
        return;
    }

    // Kept so the post-settle sweep below can find whichever entries ended up `Failed`; the
    // partition consumes `pending`. `Arc` clones, so this is a vector of pointers.
    let all_entries = pending.clone();

    let (stdio_entries, http_entries): (Vec<_>, Vec<_>) = pending
        .into_iter()
        .partition(|entry| matches!(entry.config.transport, McpTransport::Stdio));

    let stdio_limit = runtime.stdio_concurrency.max(1);
    let http_limit = runtime.http_concurrency.max(1);
    let timeout = runtime.connect_timeout;
    let stdio_manager = Arc::clone(&manager);
    let http_manager = Arc::clone(&manager);

    let stdio_stream = futures::stream::iter(stdio_entries)
        .map(move |entry| {
            let manager = Arc::clone(&stdio_manager);
            async move {
                connect_one(entry, manager, mcp_default_permission, timeout).await;
            }
        })
        .buffer_unordered(stdio_limit)
        .for_each(|_| async {});

    let http_stream = futures::stream::iter(http_entries)
        .map(move |entry| {
            let manager = Arc::clone(&http_manager);
            async move {
                connect_one(entry, manager, mcp_default_permission, timeout).await;
            }
        })
        .buffer_unordered(http_limit)
        .for_each(|_| async {});

    tokio::join!(stdio_stream, http_stream);
    let _ = settled.send(true);

    // `Failed` is only ever set by `connect_one`, and only from this function, so once both
    // streams have drained the failed set is complete and final. Without the retry below nothing
    // ever transitions out of `Failed`: `require_connected` rejects, `needs_reconnect` ignores any
    // state but `Connected`, and `meka mcp reconnect` builds a throwaway manager rather than
    // healing this one. A server that was thirty seconds slow to boot would stay dead for the life
    // of the process, and take every turn down with it if it is `required`.
    for entry in all_entries {
        if matches!(entry.state().await, ServerState::Failed { .. }) {
            tokio::spawn(retry_until_connected(
                entry,
                Arc::downgrade(&manager),
                mcp_default_permission,
                timeout,
                INITIAL_RETRY_BACKOFF,
            ));
        }
    }
}

/// Retry one cold-start failure until it connects, with exponential backoff from `initial_backoff`
/// to [`MAX_RETRY_BACKOFF`]. Returns once the entry leaves `Failed`.
///
/// `initial_backoff` is a parameter rather than a direct read of [`INITIAL_RETRY_BACKOFF`] so tests
/// can drive the loop in milliseconds; the sole production call site passes the constant.
///
/// This goes through [`connect_one`] rather than [`ServerEntry::reconnect`], which does nothing for
/// an entry that is not `Connected`: it exists to reopen a transport that has closed under a state
/// still claiming otherwise, and a cold-start failure has no transport to reopen.
///
/// `connect_one` registers tools through `McpClientManager::register_server_tools`, which fans out
/// to every attached per-session registry, so sessions created while the server was down pick up
/// its tools when it recovers.
///
/// The manager is held weakly so this loop can't outlive it. `mcp::cli::run_reconnect` builds a
/// throwaway manager per invocation, and it is reachable from the REPL's `/mcp reconnect` in a
/// process that keeps running; a strong reference there would leave a task respawning a failing
/// server every five minutes for a manager nobody is using.
async fn retry_until_connected(
    entry: Arc<ServerEntry>,
    manager: std::sync::Weak<McpClientManager>,
    mcp_default_permission: Option<Permission>,
    connect_timeout: Duration,
    initial_backoff: Duration,
) {
    let mut backoff = initial_backoff;
    loop {
        tokio::time::sleep(backoff).await;

        // Nothing holds the manager any more, so there is no registry left to heal into.
        let Some(manager) = manager.upgrade() else {
            return;
        };

        // Cheap pre-check before taking the lock; something else may have healed the entry.
        if !matches!(entry.state().await, ServerState::Failed { .. }) {
            return;
        }

        tracing::debug!(
            "retrying initial connect to MCP server '{}' after {:?}",
            entry.server_name(),
            backoff,
        );

        // Serialise against `ServerEntry::reconnect`, which takes the same lock, so a tool call
        // and this loop can't drive two connects into the same entry at once. Re-check under the
        // lock: the winner of a race leaves the entry `Connected` and the loser must not clobber
        // it with a second connection.
        {
            let _guard = entry.reconnect_lock.lock().await;
            if !matches!(entry.state().await, ServerState::Failed { .. }) {
                return;
            }
            connect_one(
                Arc::clone(&entry),
                manager,
                mcp_default_permission,
                connect_timeout,
            )
            .await;
        }

        if !matches!(entry.state().await, ServerState::Failed { .. }) {
            tracing::info!(
                "MCP server '{}' recovered after a failed initial connect",
                entry.server_name(),
            );
            return;
        }

        backoff = std::cmp::min(backoff.saturating_mul(2), MAX_RETRY_BACKOFF);
    }
}

/// Whether `cause` is the same failure the entry is already sitting on, and so not worth
/// repeating. A *changed* cause is new information (a server that went from "not found" to
/// "connection refused" is a different problem) and is announced.
fn is_repeat_failure(state: &ServerState, cause: &str) -> bool {
    matches!(state, ServerState::Failed { error, .. } if error == cause)
}

/// Record a failed connect on `entry`, announcing it only when it tells the user something new.
///
/// A server that is down stays down, and [`retry_until_connected`] keeps trying every five minutes
/// for the life of the process. Logging each attempt at `warn!` meant a missing binary or an
/// unreachable endpoint printed forever, on top of an idle REPL prompt, long after the user had
/// read the message the first time. The first failure - and any *change* of cause - is worth
/// saying; a repeat of the same one is not.
async fn record_connect_failure(entry: &Arc<ServerEntry>, server_name: &str, cause: String) {
    let repeat = {
        let state = entry.state.read().await;
        is_repeat_failure(&state, &cause)
    };
    if repeat {
        tracing::debug!("MCP server '{}' still unavailable: {}", server_name, cause);
    } else {
        tracing::warn!(
            "failed to connect to MCP server '{}': {}",
            server_name,
            cause
        );
    }
    *entry.state.write().await = ServerState::Failed {
        error: cause,
        at: std::time::Instant::now(),
    };
}

/// Connect a single `Pending` server: wrap the existing `connect_server` in a per-server timeout,
/// capture instructions, discover + register tools into the registry, and flip the entry's state to
/// `Connected` on success or `Failed` on error. Never panics; errors are logged and reflected in
/// [`ServerState::Failed`] so the turn gate can surface them.
/// **Precondition:** the caller holds `entry.reconnect_lock`, or is the initial sweep in
/// [`run_connector`], which owns every `Pending` entry outright. Two concurrent calls against one
/// entry spawn two transports and let the loser's `record_connect_failure` overwrite the winner's
/// `Connected` state. [`retry_until_connected`] and `McpClientManager::reconnect_server` both take
/// the lock for this reason.
pub(crate) async fn connect_one(
    entry: Arc<ServerEntry>,
    manager: Arc<McpClientManager>,
    mcp_default_permission: Option<Permission>,
    connect_timeout: std::time::Duration,
) {
    let server_name = entry.server_name.clone();

    // connect_server's future can be `!Send` for OAuth-authenticated servers (rmcp 1.5 holds a
    // `form_urlencoded::Serializer` across an await in its auth module, whose `Option<&dyn Fn(&str)
    // -> Cow<[u8]>>` closure slot is not `Sync`). Drive it on a `spawn_blocking` thread using the
    // outer runtime's `Handle`, the same approach `reconnect` uses.
    let handle = tokio::runtime::Handle::current();
    let entry_for_connect = Arc::clone(&entry);
    let server_name_for_task = server_name.clone();
    let connect_task = tokio::task::spawn_blocking(move || {
        handle.block_on(async move {
            tokio::time::timeout(
                connect_timeout,
                connect_server(
                    &server_name_for_task,
                    &entry_for_connect.config,
                    entry_for_connect.token_store.as_ref(),
                    &entry_for_connect.client_context,
                ),
            )
            .await
        })
    });

    let connected = match connect_task.await {
        Ok(Ok(Ok(service))) => service,
        // Store the bare cause. `MekaError::McpConnection` renders as
        // "MCP connection error: <server>: <message>", and every consumer of this string
        // (`meka mcp list`, the turn-gate summary, the agent-facing tool error) already names the
        // server, so keeping the prefix would say it twice.
        Ok(Ok(Err(error))) => {
            let cause = match &error {
                MekaError::McpConnection { message, .. } => message.clone(),
                other => other.to_string(),
            };
            record_connect_failure(&entry, &server_name, cause).await;
            return;
        }
        Ok(Err(_elapsed)) => {
            let cause = format!("connect timed out after {:?}", connect_timeout);
            record_connect_failure(&entry, &server_name, cause).await;
            return;
        }
        Err(join_error) => {
            let cause = format!("connect task join error: {}", join_error);
            record_connect_failure(&entry, &server_name, cause).await;
            return;
        }
    };

    tracing::info!("connected to MCP server '{}'", server_name);

    // rmcp 2.1: `peer_info()` returns `Option<Arc<InitializeResult>>` (owned) rather than a borrow,
    // so the instructions string is cloned out of the `Arc`.
    entry.record_instructions(
        connected
            .peer()
            .peer_info()
            .and_then(|info| info.instructions.clone()),
    );

    // Flip state to Connected BEFORE tool registration so `list_all_tools` below goes through the
    // live peer via `require_connected`.
    let service_arc = Arc::new(connected);
    *entry.state.write().await = ServerState::Connected {
        service: Arc::clone(&service_arc),
    };

    // Discover + register tools. Any error here doesn't undo the Connected state. The server is
    // reachable, just its tool list failed. Surface it as a warn and leave tool set empty.
    //
    // Bounded by the same `connect_timeout` as the connect itself: `tools/list` is a request to the
    // server just made, and unbounded, a server that accepts the connection and then never answers
    // holds this task open for the life of the process.
    let discovery = tokio::time::timeout(
        connect_timeout,
        discover_and_register_tools(&entry, mcp_default_permission, &manager),
    )
    .await
    .unwrap_or_else(|_elapsed| {
        Err(MekaError::McpConnection {
            server_name: server_name.clone(),
            message: format!("tool discovery timed out after {:?}", connect_timeout),
        })
    });
    match discovery {
        Ok(count) => {
            tracing::info!("MCP server '{}' registered {} tool(s)", server_name, count);
        }
        Err(error) => {
            tracing::warn!(
                "MCP server '{}' connected but tool discovery failed: {}",
                server_name,
                error
            );
        }
    }
}

/// Fetch `list_tools` from a just-connected server and route the resulting adapters through
/// [`McpClientManager::register_server_tools`], which records which of them ship deferred and
/// carries both facts to every registry -- the ones attached now and the ones that attach later.
async fn discover_and_register_tools(
    entry: &Arc<ServerEntry>,
    mcp_default_permission: Option<Permission>,
    manager: &Arc<McpClientManager>,
) -> Result<usize> {
    let adapters = build_mcp_adapters(entry, mcp_default_permission).await?;
    let registered_count = adapters.len();
    manager
        .register_server_tools(&entry.server_name, adapters)
        .await;
    Ok(registered_count)
}

/// Core adapter-construction logic shared between initial discovery (via the connector) and ad-hoc
/// discovery (via [`super::McpClientManager::discover_tools_for_server`]).
async fn build_mcp_adapters(
    entry: &Arc<ServerEntry>,
    mcp_default_permission: Option<Permission>,
) -> Result<Vec<McpToolAdapter>> {
    let server_name = entry.server_name.clone();
    let server_config = &entry.config;
    // Bounded like every other `tools/list`. This one already sat inside the connect timeout, but
    // the tool *count* is unbounded there too, and the cap belongs to meka rather than to the
    // server's pagination.
    let tools = entry
        .list_tools_bounded(super::DEFAULT_MCP_REQUEST_TIMEOUT)
        .await?;

    let advertised: std::collections::HashSet<&str> =
        tools.iter().map(|t| t.name.as_ref()).collect();
    warn_on_stale_tool_config(&server_name, server_config, &advertised);

    let mut adapters = Vec::new();
    for tool in tools {
        let raw_tool_name = tool.name.as_ref().to_string();
        if !tool_is_allowed(server_config, &raw_tool_name) {
            continue;
        }

        let sanitised_tool_name = crate::mcp::sanitize::normalize_server_name(&raw_tool_name);
        let namespaced_name = format!("mcp__{}__{}", server_name, sanitised_tool_name);

        let raw_description = tool
            .description
            .as_ref()
            .map(|d| d.as_ref().to_string())
            .unwrap_or_default();
        let description = truncate(
            &crate::mcp::sanitize::sanitize_text(&raw_description),
            MAX_MCP_DESCRIPTION_LENGTH,
        );

        let parameters = match serde_json::to_value(&*tool.input_schema) {
            Ok(value) => value,
            Err(error) => {
                tracing::warn!(
                    "MCP server '{}' tool '{}' has unserializable input schema ({}); \
                     skipping registration",
                    server_name,
                    raw_tool_name,
                    error
                );
                continue;
            }
        };

        let permission = resolve_tool_permission(
            &server_name,
            &raw_tool_name,
            tool.annotations.as_ref(),
            server_config,
            mcp_default_permission,
        )?;

        // Annotations carry permission hints (`readOnlyHint`, `destructiveHint`); silently dropping
        // them on a serialization failure could quietly relax permission resolution. Log so the
        // failure shows up at default verbosity.
        let annotations =
            tool.annotations
                .as_ref()
                .and_then(|ann| match serde_json::to_value(ann) {
                    Ok(value) => Some(value),
                    Err(error) => {
                        tracing::warn!(
                            "failed to serialize annotations for tool '{}': {}",
                            namespaced_name,
                            error
                        );
                        None
                    }
                });
        let meta = tool
            .meta
            .as_ref()
            .and_then(|m| match serde_json::to_value(m) {
                Ok(value) => Some(value),
                Err(error) => {
                    tracing::warn!(
                        "failed to serialize meta for tool '{}': {}",
                        namespaced_name,
                        error
                    );
                    None
                }
            });
        let title = tool
            .title
            .as_ref()
            .map(|t| crate::mcp::sanitize::sanitize_text(t));

        adapters.push(McpToolAdapter::new(
            namespaced_name,
            raw_tool_name,
            description,
            parameters,
            permission,
            Arc::clone(entry),
            annotations,
            meta,
            title,
        ));
    }

    Ok(adapters)
}

/// Drain a stdio MCP child's stderr line by line into meka's tracing stream. Many MCP servers log
/// diagnostics to stderr; with rmcp's default inherited stderr those lines land directly on meka's
/// terminal and corrupt the REPL. Emitting at `debug!` keeps them off the terminal at default
/// verbosity while still surfacing under `-v` / `RUST_LOG`, tagged with the server name. The task
/// is detached: it ends on its own when the child exits and stderr reaches EOF.
fn forward_child_stderr(server_name: String, stderr: tokio::process::ChildStderr) {
    use tokio::io::AsyncReadExt;

    // Split into lines by hand rather than with `BufReader::lines`, which grows one `String` until
    // it finds a newline: a child that emits none hands meka an unbounded allocation fed by a pipe
    // it controls. Splitting here caps what one line may cost without capping how much the child
    // may log over its life, which a `take` on the whole stream would have done -- and a full pipe
    // blocks the child rather than meka.
    const MAX_STDERR_LINE_BYTES: usize = 4096;

    fn emit(server_name: &str, line: &[u8], overlong: bool) {
        // The text is the child's, so it can carry escapes that would repaint the terminal of
        // anyone running with `-v`.
        let text = crate::mcp::sanitize::sanitize_text(&String::from_utf8_lossy(line));
        if overlong {
            tracing::debug!(server = %server_name, "{}... (line truncated)", text);
        } else {
            tracing::debug!(server = %server_name, "{}", text);
        }
    }

    tokio::spawn(async move {
        let mut stderr = stderr;
        let mut buffer = [0u8; 8192];
        let mut line: Vec<u8> = Vec::new();
        let mut overlong = false;
        loop {
            let read = match stderr.read(&mut buffer).await {
                Ok(0) => break,
                Ok(read) => read,
                Err(error) => {
                    tracing::debug!(server = %server_name, "stderr read error: {}", error);
                    break;
                }
            };
            for &byte in &buffer[..read] {
                if byte == b'\n' {
                    emit(&server_name, &line, overlong);
                    line.clear();
                    overlong = false;
                } else if line.len() < MAX_STDERR_LINE_BYTES {
                    line.push(byte);
                } else {
                    overlong = true;
                }
            }
        }
        if !line.is_empty() {
            emit(&server_name, &line, overlong);
        }
    });
}

/// Which stored bearer, if any, belongs on the transport: the `[auth]` block wins, and a bearer
/// beside one is dropped rather than sent alongside it.
///
/// rmcp consults the authorization flow only when the transport carries no `auth_header`, so
/// passing both would send the static bearer on every request and leave the token the flow obtained
/// unused -- a server that fails while holding a valid credential. `mcp add` and `mcp login` both
/// refuse to *create* that pairing, but `config.toml` is a supported surface: adding an `[auth]`
/// block by hand to a server that already has a stored bearer reaches it without passing either
/// door. This is the point where the ambiguity would do harm, so it is resolved here, in favour of
/// the block the user can see in their config over the row they cannot.
///
/// A free function rather than three lines inside [`connect_server`], because that function needs a
/// live server and a store to reach and nothing could test the decision in place. The rule is worth
/// more than the wiring around it.
fn bearer_for_transport(
    server_name: &str,
    bearer: Option<String>,
    auth: Option<&crate::config::McpAuthConfig>,
) -> Option<String> {
    match (bearer, auth) {
        (Some(_), Some(_)) => {
            tracing::warn!(
                "server '{}' has both a stored bearer and an [auth] block; ignoring the bearer and \
                 authenticating through the block. Run `meka mcp logout {}` to drop the bearer, or \
                 remove the [auth] block to use it",
                server_name,
                server_name
            );
            None
        }
        (bearer, _) => bearer,
    }
}

/// Connect to an MCP server, dispatching to the auth or no-auth path.
///
/// The returned future is `!Send` when the server config uses OAuth, because rmcp's auth module
/// holds a `!Sync` closure across an await. Callers that need a `Send` future (e.g. `Tool::execute`
/// during reconnect) drive this on a `spawn_blocking` thread via [`ServerEntry::reconnect`].
pub(super) async fn connect_server(
    server_name: &str,
    config: &McpServerConfig,
    token_store: Option<&TokenStore>,
    client_context: &Arc<McpClientContext>,
) -> Result<McpRunningService> {
    use rmcp::ServiceExt;

    let handler = MekaClientHandler::new(server_name.to_string(), Arc::clone(client_context));

    match config.transport {
        McpTransport::Stdio => {
            let command_str =
                config
                    .command
                    .as_deref()
                    .ok_or_else(|| MekaError::McpConnection {
                        server_name: server_name.to_string(),
                        message: "stdio transport requires 'command' field".to_string(),
                    })?;

            let args_vec: Vec<String> = config.args.clone().unwrap_or_default();
            let command = build_stdio_command(command_str, &args_vec);
            let mut command = command;
            // Scrub the environment before layering the server's own `env` on top.
            //
            // A stdio MCP server is a child process that talks to the network, and an inherited
            // environment hands it every variable meka was started with: `ANTHROPIC_API_KEY`,
            // `AWS_*`, `GITHUB_TOKEN`, the lot. Configuring a server is a decision to run its code,
            // not a decision to hand it every credential on the machine. What survives is the same
            // curated base a read-mode shell gets (`PATH`, `HOME`, locale), so servers still
            // resolve their own binaries; a server that genuinely needs a secret asks for it by
            // name, and `${VAR}` in the `env` table still reads it from meka's environment.
            command.env_clear();
            command.envs(crate::sandbox::sandbox_child_env());
            if let Some(env) = &config.env {
                command.envs(env);
            }

            // No bound on an incoming line, and rmcp 3.1 offers no way to set one here.
            //
            // `TokioChildProcess` builds an `AsyncRwTransport` internally with no seam to
            // configure, and that transport's `receive` does its own `read_until(b'\n')` into an
            // unbounded `Vec` -- it never consults `JsonRpcMessageCodec::max_length`, which is the
            // only length knob the crate exposes and applies to the *write* side. So a stdio server
            // that emits no newline grows one buffer until the process dies.
            //
            // Left as is deliberately. Bounding it means spawning the child and constructing the
            // transport here, which also means reimplementing `ChildWithCleanup` -- the drop guard
            // that kills the child rather than leaving a zombie. Getting that wrong is a worse
            // failure than the one being fixed, and the exposure is narrow: a stdio server is a
            // program the user configured and chose to run, already executing arbitrary code, so
            // this bounds a *buggy* server rather than a hostile one. Revisit when rmcp exposes the
            // read length.
            //
            // rmcp's `TokioChildProcess::new` leaves the child's stderr inherited, so an MCP server
            // that logs to stderr (many `tracing`/`log`-based servers do) writes straight onto
            // meka's terminal and corrupts the REPL display. Pipe it and drain it into our own
            // tracing stream instead: quiet by default, visible under `-v` / `RUST_LOG`, and
            // attributed to the server rather than mixed into the live prompt.
            let (transport, stderr) = rmcp::transport::TokioChildProcess::builder(command)
                .stderr(std::process::Stdio::piped())
                .spawn()
                .map_err(|error| MekaError::McpConnection {
                    server_name: server_name.to_string(),
                    // `NotFound` from spawn means the program isn't on PATH, which reads as a
                    // bare "No such file or directory" without saying which file. Name it.
                    message: if error.kind() == std::io::ErrorKind::NotFound {
                        format!("failed to spawn process: '{}' not found", command_str)
                    } else {
                        format!("failed to spawn process: {}", error)
                    },
                })?;
            if let Some(stderr) = stderr {
                forward_child_stderr(server_name.to_string(), stderr);
            }

            handler
                .serve(transport)
                .await
                .map_err(|error| MekaError::McpConnection {
                    server_name: server_name.to_string(),
                    message: format!("handshake failed: {}", error),
                })
        }
        McpTransport::Http => {
            let url = config
                .url
                .as_deref()
                .ok_or_else(|| MekaError::McpConnection {
                    server_name: server_name.to_string(),
                    message: "http transport requires 'url' field".to_string(),
                })?;

            // The server's static bearer, if it has one. Absent for a server that authenticates
            // through a flow instead, and absent when this host has no store at all (the mock
            // harness), which is the same "send no Authorization header" as before.
            let bearer = match token_store {
                Some(store) => {
                    store
                        .load_mcp_credentials(
                            server_name,
                            crate::session::McpCredentialKind::Bearer,
                        )
                        .await?
                }
                None => None,
            };

            let bearer = bearer_for_transport(server_name, bearer, config.auth.as_ref());
            let transport_config =
                build_http_transport_config(server_name, config, bearer.as_deref())?;

            if let Some(auth_config) = &config.auth {
                super::auth::connect_http_with_oauth(
                    server_name,
                    url,
                    auth_config,
                    transport_config,
                    token_store,
                    handler,
                )
                .await
            } else {
                let transport =
                    rmcp::transport::StreamableHttpClientTransport::from_config(transport_config);

                handler
                    .serve(transport)
                    .await
                    .map_err(|error| MekaError::McpConnection {
                        server_name: server_name.to_string(),
                        message: format!("HTTP connection failed: {}", error),
                    })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::sync::{Mutex, RwLock};

    use super::*;
    use crate::config::{McpAuthConfig, McpServerConfig, McpTransport};

    /// A stored bearer and an `[auth]` block cannot both be honoured, and rmcp resolves the tie the
    /// wrong way round: it consults the authorization flow only when the transport carries no
    /// `auth_header`, so passing both sends the bearer and wastes the token the flow obtained.
    ///
    /// `mcp add` and `mcp login` refuse to create the pairing, but hand-editing `config.toml`
    /// reaches it without passing either. This is the last door, and the only one that decides what
    /// actually goes on the wire.
    #[test]
    fn an_auth_block_suppresses_a_stored_bearer() {
        let oauth = McpAuthConfig::OAuth {
            client_id: None,
            scopes: None,
            redirect_port: None,
        };

        assert_eq!(
            bearer_for_transport("api", Some("bearer-not-a-real-token".to_string()), None),
            Some("bearer-not-a-real-token".to_string()),
            "with no [auth] block the bearer is the whole authentication and must be sent"
        );
        assert_eq!(
            bearer_for_transport(
                "api",
                Some("bearer-not-a-real-token".to_string()),
                Some(&oauth)
            ),
            None,
            "beside an [auth] block it must be dropped, or it shadows the flow's own token"
        );
        assert_eq!(
            bearer_for_transport("api", None, Some(&oauth)),
            None,
            "and a server with no bearer sends no Authorization header of its own"
        );
        assert_eq!(bearer_for_transport("api", None, None), None);
    }

    /// A dead server is retried every five minutes for the life of the process. Announcing each
    /// attempt buried an idle REPL prompt in identical warnings long after the user had read the
    /// first one, so only a new or changed cause is worth saying out loud.
    #[test]
    fn repeat_failure_is_recognised_and_a_changed_cause_is_not() {
        let same = ServerState::Failed {
            error: "'ida-mcp' not found".to_string(),
            at: std::time::Instant::now(),
        };
        assert!(is_repeat_failure(&same, "'ida-mcp' not found"));
        assert!(
            !is_repeat_failure(&same, "connection refused"),
            "a different cause is new information and must be announced"
        );
        // The first failure of all, from Pending, is never a repeat.
        assert!(!is_repeat_failure(
            &ServerState::Pending,
            "'ida-mcp' not found"
        ));
        assert!(!is_repeat_failure(&ServerState::Disabled, "anything"));
    }

    #[tokio::test]
    async fn record_connect_failure_stores_the_bare_cause() {
        let entry = bare_entry("ida");
        record_connect_failure(&entry, "ida", "'ida-mcp' not found".to_string()).await;
        match &*entry.state.read().await {
            ServerState::Failed { error, .. } => assert_eq!(error, "'ida-mcp' not found"),
            other => panic!("expected Failed, got {}", other.label()),
        }
        // A second identical failure keeps the state and stays quiet.
        record_connect_failure(&entry, "ida", "'ida-mcp' not found".to_string()).await;
        assert!(matches!(
            &*entry.state.read().await,
            ServerState::Failed { .. }
        ));
    }

    fn bare_entry(name: &str) -> Arc<ServerEntry> {
        Arc::new(ServerEntry {
            server_name: name.to_string(),
            config: bare_server_config(name),
            token_store: None,
            client_context: McpClientContext::new(),
            state: RwLock::new(ServerState::Pending),
            reconnect_lock: Mutex::new(()),
            instructions: std::sync::RwLock::new(None),
            request_timeout: std::sync::OnceLock::new(),
            dropped_tools: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    fn bare_server_config(name: &str) -> McpServerConfig {
        McpServerConfig {
            name: name.to_string(),
            transport: McpTransport::Http,
            command: None,
            args: None,
            env: None,
            url: Some("https://example".to_string()),
            headers: None,
            headers_helper: None,
            auth: None,
            permission: None,
            allowed_tools: None,
            disabled_tools: None,
            eager_load_tools: None,
            tool_permissions: None,
            trust_read_only_hint: None,
            disabled: false,
            required: None,
        }
    }

    /// The container case: the configured command simply isn't installed. A bare "No such file or
    /// directory" doesn't say *which* file, and that is the whole diagnosis.
    ///
    /// Unix-only: on Windows a shim command (`npx`, `*.cmd`) is launched through `cmd /c`, which
    /// spawns successfully and fails later in the handshake, so the spawn-time name is not
    /// available to report.
    #[cfg(unix)]
    #[tokio::test]
    async fn missing_stdio_binary_names_the_command() {
        let mut config = bare_server_config("ida");
        config.transport = McpTransport::Stdio;
        config.command = Some("meka-no-such-binary-xyz".to_string());
        config.url = None;

        let entry = Arc::new(ServerEntry {
            server_name: "ida".to_string(),
            config,
            token_store: None,
            client_context: McpClientContext::new(),
            state: RwLock::new(ServerState::Pending),
            reconnect_lock: Mutex::new(()),
            instructions: std::sync::RwLock::new(None),
            request_timeout: std::sync::OnceLock::new(),
            dropped_tools: std::sync::atomic::AtomicUsize::new(0),
        });
        let manager = McpClientManager::prepare(&[], None, None, McpClientContext::new())
            .await
            .expect("empty manager");
        connect_one(
            Arc::clone(&entry),
            manager,
            None,
            std::time::Duration::from_secs(5),
        )
        .await;

        match &*entry.state.read().await {
            ServerState::Failed { error, .. } => {
                assert!(error.contains("meka-no-such-binary-xyz"), "{error}");
                assert!(error.contains("not found"), "{error}");
            }
            other => panic!("expected Failed, got {}", other.label()),
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn connect_one_timeout_marks_entry_failed() {
        use std::sync::OnceLock;
        // A hung stdio process (`sleep 999`) forces `connect_server`'s initialize handshake to
        // never complete. With a 50 ms timeout, `connect_one` must bail and mark the entry Failed.
        let mut config = bare_server_config("hung");
        config.transport = McpTransport::Stdio;
        config.command = Some("/bin/sleep".to_string());
        config.args = Some(vec!["999".to_string()]);
        config.url = None;

        let entry = Arc::new(ServerEntry {
            server_name: "hung".to_string(),
            config,
            token_store: None,
            client_context: McpClientContext::new(),
            state: RwLock::new(ServerState::Pending),
            reconnect_lock: Mutex::new(()),
            instructions: std::sync::RwLock::new(None),
            request_timeout: OnceLock::new(),
            dropped_tools: std::sync::atomic::AtomicUsize::new(0),
        });

        // The test never reaches tool discovery (the connect itself times out), so the manager
        // isn't observed; build a minimal one just to satisfy the signature.
        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(&[], None, None, context)
            .await
            .expect("empty manager");
        connect_one(
            Arc::clone(&entry),
            manager,
            None,
            std::time::Duration::from_millis(50),
        )
        .await;

        let state = entry.state().await;
        match state {
            ServerState::Failed { error, .. } => {
                assert!(
                    error.contains("timed out"),
                    "expected 'timed out' in Failed error, got: {}",
                    error
                );
            }
            other => panic!("expected Failed, got: {}", other.label()),
        }
    }

    /// Build a `Failed` entry whose stdio command is `command`, plus a minimal manager. The
    /// manager is returned strongly so a test can decide when to drop it.
    #[cfg(unix)]
    async fn failed_entry(name: &str, command: &str) -> (Arc<ServerEntry>, Arc<McpClientManager>) {
        use std::sync::OnceLock;
        let mut config = bare_server_config(name);
        config.transport = McpTransport::Stdio;
        config.command = Some(command.to_string());
        config.url = None;

        let entry = Arc::new(ServerEntry {
            server_name: name.to_string(),
            config,
            token_store: None,
            client_context: McpClientContext::new(),
            state: RwLock::new(ServerState::Failed {
                error: "initial connect failed".to_string(),
                at: std::time::Instant::now(),
            }),
            reconnect_lock: Mutex::new(()),
            instructions: std::sync::RwLock::new(None),
            request_timeout: OnceLock::new(),
            dropped_tools: std::sync::atomic::AtomicUsize::new(0),
        });
        let manager = McpClientManager::prepare(&[], None, None, McpClientContext::new())
            .await
            .expect("empty manager");
        (entry, manager)
    }

    /// The whole point of the retry: a server that stays down must keep being retried rather than
    /// being abandoned after some attempt count. `/bin/false` exits immediately, so several
    /// attempts elapse inside the window and the task must still be looping at the end of it.
    #[cfg(unix)]
    #[tokio::test]
    async fn retry_keeps_going_while_the_server_stays_down() {
        let (entry, manager) = failed_entry("still-down", "/bin/false").await;
        let handle = tokio::spawn(retry_until_connected(
            Arc::clone(&entry),
            Arc::downgrade(&manager),
            None,
            std::time::Duration::from_millis(50),
            std::time::Duration::from_millis(10),
        ));

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        assert!(
            !handle.is_finished(),
            "retry loop gave up on a server that is still down"
        );
        assert!(matches!(entry.state().await, ServerState::Failed { .. }));
        handle.abort();
    }

    /// The retry must not outlive its manager. `mcp::cli::run_reconnect` builds a throwaway
    /// manager per invocation and is reachable from the REPL's `/mcp reconnect`, so a strong
    /// reference here would leave a task respawning a failing server for the life of the process.
    #[cfg(unix)]
    #[tokio::test]
    async fn retry_stops_when_the_manager_is_dropped() {
        let (entry, manager) = failed_entry("orphaned", "/bin/false").await;
        let handle = tokio::spawn(retry_until_connected(
            Arc::clone(&entry),
            Arc::downgrade(&manager),
            None,
            std::time::Duration::from_millis(50),
            std::time::Duration::from_millis(10),
        ));

        drop(manager);

        let joined = tokio::time::timeout(std::time::Duration::from_secs(5), handle).await;
        assert!(
            joined.is_ok(),
            "retry loop outlived the manager it was healing into"
        );
        // And it gave up rather than connecting behind the dropped manager's back.
        assert!(matches!(entry.state().await, ServerState::Failed { .. }));
    }

    /// And it must stop once the entry is no longer `Failed`, so a recovered server doesn't leave a
    /// task polling for the life of the process. `Disabled` stands in for the recovered state
    /// because constructing `Connected` needs a live service.
    #[cfg(unix)]
    #[tokio::test]
    async fn retry_stops_once_the_entry_leaves_failed() {
        let (entry, manager) = failed_entry("recovers", "/bin/false").await;
        let handle = tokio::spawn(retry_until_connected(
            Arc::clone(&entry),
            Arc::downgrade(&manager),
            None,
            std::time::Duration::from_millis(50),
            std::time::Duration::from_millis(10),
        ));

        *entry.state.write().await = ServerState::Disabled;

        let joined = tokio::time::timeout(std::time::Duration::from_secs(5), handle).await;
        assert!(
            joined.is_ok(),
            "retry loop kept polling an entry that is no longer Failed"
        );
    }
}
