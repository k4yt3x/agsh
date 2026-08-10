//! `meka mcp …` management subcommands.

use std::sync::Arc;

use crate::{
    config::{McpAuthConfig, McpServerConfig, McpTransport},
    error::{MekaError, Result},
    mcp::{McpClientContext, McpClientManager},
    session::TokenStore,
};

fn config_err(message: impl Into<String>) -> MekaError {
    MekaError::Config(message.into())
}

/// Run `meka mcp list`. Prints configured servers + their transport/URL.
///
/// When `manager` is `Some`, the output also carries a `state` column showing each server's live
/// lifecycle state (`pending` / `connected` / `failed` / `disabled`). The out-of-band `meka mcp
/// list` CLI doesn't have a running manager and gets the no-state view; the REPL's `/mcp list`
/// passes the live manager for the richer output.
pub async fn run_list(
    servers: &[McpServerConfig],
    manager: Option<&std::sync::Arc<crate::mcp::McpClientManager>>,
) -> Result<()> {
    if servers.is_empty() {
        println!("(no MCP servers configured)");
        return Ok(());
    }

    // Resolve state once up front so the table output stays consistent even if a server connects
    // mid-print.
    let mut states: std::collections::HashMap<&str, &'static str> =
        std::collections::HashMap::new();
    if let Some(manager) = manager {
        for config in servers {
            let label = match manager.server_entry(&config.name) {
                Some(entry) => entry.state().await.label(),
                None => "(unknown)",
            };
            states.insert(config.name.as_str(), label);
        }
    }

    // The live-manager variant adds a `State` column; both share the shared column formatter so the
    // table aligns regardless of how long a server name or target gets.
    let with_state = manager.is_some();

    let rows: Vec<Vec<String>> = servers
        .iter()
        .map(|config| {
            let target = match config.transport {
                McpTransport::Stdio => {
                    let args = config
                        .args
                        .as_ref()
                        .map(|a| a.join(" "))
                        .unwrap_or_default();
                    format!(
                        "{} {}",
                        config.command.as_deref().unwrap_or("(no command)"),
                        args
                    )
                    .trim()
                    .to_string()
                }
                McpTransport::Http => config.url.clone().unwrap_or_else(|| "(no url)".to_string()),
            };
            let transport_label = match config.transport {
                McpTransport::Stdio => "stdio",
                McpTransport::Http => "http",
            };
            let perm_label = config.permission.as_deref().unwrap_or("read");

            let mut row = vec![config.name.clone(), transport_label.to_string()];
            if with_state {
                let state_label = states
                    .get(config.name.as_str())
                    .copied()
                    .unwrap_or("(unknown)");
                row.push(state_label.to_string());
            }
            row.push(perm_label.to_string());
            row.push(target);
            row
        })
        .collect();

    let headers: &[&str] = if with_state {
        &["Name", "Transport", "State", "Permission", "Target"]
    } else {
        &["Name", "Transport", "Permission", "Target"]
    };
    print!("{}", crate::render::format_columns(headers, &rows));
    Ok(())
}

/// Run `meka mcp get <name>`. Prints a single server config in detail.
pub async fn run_get(servers: &[McpServerConfig], name: &str) -> Result<()> {
    let config = servers
        .iter()
        .find(|c| c.name == name)
        .ok_or_else(|| config_err(format!("no MCP server named '{}'", name)))?;
    println!("name:        {}", config.name);
    println!("transport:   {}", match config.transport {
        McpTransport::Stdio => "stdio",
        McpTransport::Http => "http",
    });
    println!(
        "permission:  {}",
        config.permission.as_deref().unwrap_or("read")
    );
    if let Some(command) = &config.command {
        println!("command:     {}", command);
    }
    if let Some(args) = &config.args {
        println!("args:        {:?}", args);
    }
    if let Some(env) = &config.env {
        println!("env:         {} keys", env.len());
        for key in env.keys() {
            println!("  - {}", key);
        }
    }
    if let Some(url) = &config.url {
        println!("url:         {}", url);
    }
    if let Some(headers) = &config.headers {
        println!("headers:     {} entries", headers.len());
    }
    if config.auth_token.is_some() {
        println!("auth_token:  (set)");
    }
    if let Some(auth) = &config.auth {
        println!("auth:        {:?}", std::mem::discriminant(auth));
    }
    if let Some(allowed) = config.allowed_tools.as_deref() {
        println!("allowed_tools: {}", allowed.join(", "));
    }
    if let Some(disabled) = config.disabled_tools.as_deref() {
        println!("disabled_tools: {}", disabled.join(", "));
    }
    if let Some(perms) = config.tool_permissions.as_ref()
        && !perms.is_empty()
    {
        println!("tool_permissions:");
        let mut keys: Vec<&String> = perms.keys().collect();
        keys.sort();
        for key in keys {
            println!("  - {} = {}", key, perms[key]);
        }
    }
    Ok(())
}

/// Run `meka mcp reconnect <name>`: connect once as a smoke test, print `ok` on success and the
/// error otherwise. Does not mutate config. Run `meka mcp tools <name>`: connect to the server,
/// list every advertised tool, resolve permissions, and print a column-aligned table.
/// Disabled-by-allow/block tools are still shown (marked `blocked`) so users can edit their config
/// without leaving the CLI to discover names.
pub async fn run_tools(
    servers: &[McpServerConfig],
    mcp_default: Option<crate::permission::Permission>,
    token_store: &TokenStore,
    name: &str,
) -> Result<()> {
    let config = servers
        .iter()
        .find(|c| c.name == name)
        .ok_or_else(|| config_err(format!("no MCP server named '{}'", name)))?
        .clone();

    let context = McpClientContext::new();
    let manager = McpClientManager::prepare(
        std::slice::from_ref(&config),
        mcp_default,
        Some(token_store.clone()),
        Arc::clone(&context),
    )
    .await?;
    manager.start_connector(crate::mcp::McpRuntimeConfig {
        connect_timeout: std::time::Duration::from_secs(30),
        stdio_concurrency: 1,
        http_concurrency: 1,
    });
    manager.await_settled().await;

    let connected = if let Some(entry) = manager.server_entry(&config.name) {
        matches!(
            &*entry.state.read().await,
            crate::mcp::ServerState::Connected { .. }
        )
    } else {
        false
    };
    if !connected {
        return Err(config_err(format!(
            "failed to connect to '{}'; see logs above",
            config.name
        )));
    }

    let tools = manager.list_advertised_tools(&config.name).await?;
    manager.shutdown_arc().await;

    if tools.is_empty() {
        println!("(server '{}' advertises no tools)", config.name);
        return Ok(());
    }

    let rows: Vec<Vec<String>> = tools
        .iter()
        .map(|tool| {
            vec![
                tool.raw_name.clone(),
                tool.resolved_permission.to_string(),
                tool.permission_source.as_str().to_string(),
                if tool.allowed { "allowed" } else { "blocked" }.to_string(),
                describe_one_line(&tool.description),
            ]
        })
        .collect();
    print!(
        "{}",
        crate::render::format_columns(
            &["Name", "Permission", "Source", "Status", "Description"],
            &rows,
        )
    );

    let total = tools.len();
    let allowed = tools.iter().filter(|t| t.allowed).count();
    println!();
    println!(
        "{} tool{} total, {} allowed, {} blocked",
        total,
        if total == 1 { "" } else { "s" },
        allowed,
        total - allowed
    );
    Ok(())
}

/// Collapse a (possibly multi-line) description into one short line so the table stays legible. MCP
/// descriptions can be kilobytes; the first sentence or ~80 chars is enough for a listing.
fn describe_one_line(description: &str) -> String {
    const MAX: usize = 80;
    let mut collapsed = String::with_capacity(description.len().min(MAX + 8));
    let mut prev_space = false;
    for ch in description.chars() {
        if ch.is_whitespace() {
            if !prev_space && !collapsed.is_empty() {
                collapsed.push(' ');
            }
            prev_space = true;
        } else {
            collapsed.push(ch);
            prev_space = false;
        }
        if collapsed.chars().count() > MAX {
            break;
        }
    }
    let trimmed = collapsed.trim_end();
    if trimmed.chars().count() > MAX {
        let clipped: String = trimmed.chars().take(MAX).collect();
        format!("{}…", clipped.trim_end())
    } else {
        trimmed.to_string()
    }
}

pub async fn run_reconnect(
    servers: &[McpServerConfig],
    token_store: &TokenStore,
    name: &str,
) -> Result<()> {
    let config = servers
        .iter()
        .find(|c| c.name == name)
        .ok_or_else(|| config_err(format!("no MCP server named '{}'", name)))?
        .clone();

    let context = McpClientContext::new();
    // Per-tool permission resolution uses `[mcp].default_permission` as its global fallback.
    // `reconnect` is a smoke-test command run from outside the main agent loop, so we don't have a
    // `ResolvedConfig` in scope. Pass `None` and let resolution fall through to the hardcoded
    // strict default. Any user-specific per-server / per-tool config still applies.
    let manager = McpClientManager::prepare(
        std::slice::from_ref(&config),
        None,
        Some(token_store.clone()),
        Arc::clone(&context),
    )
    .await?;
    manager.start_connector(crate::mcp::McpRuntimeConfig {
        connect_timeout: std::time::Duration::from_secs(30),
        stdio_concurrency: 1,
        http_concurrency: 1,
    });
    manager.await_settled().await;

    let connected = if let Some(entry) = manager.server_entry(&config.name) {
        matches!(
            &*entry.state.read().await,
            crate::mcp::ServerState::Connected { .. }
        )
    } else {
        false
    };

    if connected {
        tracing::info!("connected to '{}'", config.name);
        manager.shutdown_arc().await;
        Ok(())
    } else {
        Err(config_err(format!(
            "failed to connect to '{}'; see logs above",
            config.name
        )))
    }
}

/// Run `meka mcp logout <name>`: clear any stored OAuth credentials for the given server, and
/// clear the auth-probe cache entry (if any).
pub async fn run_logout(
    servers: &[McpServerConfig],
    token_store: &TokenStore,
    name: &str,
) -> Result<()> {
    // Best-effort revocation. Cleared from stored creds regardless.
    if let Some(config) = servers
        .iter()
        .find(|c| c.name == name && matches!(c.transport, McpTransport::Http))
        && let Err(error) = crate::mcp::auth::revoke_stored_token(token_store, &config.name).await
    {
        tracing::warn!(
            "failed to revoke token at server '{}': {} (continuing)",
            config.name,
            error
        );
    }

    token_store.clear_mcp_credentials(name).await?;
    token_store.clear_auth_probe(name).await?;
    tracing::info!("cleared credentials for '{}'", name);
    Ok(())
}

/// Run `meka mcp login <name>`. Drives an interactive OAuth flow.
///
/// If the config has an explicit `[auth]` block, it's honoured as-is. If the server is HTTP with no
/// `auth` set, we assume `type = "oauth"` (authorization-code grant with dynamic client
/// registration), run the flow, and on success persist the synthesised auth block back to
/// `config.toml` so future runs skip the assumption. stdio servers without `auth` can't be logged
/// in to and error.
pub async fn run_login(
    servers: &[McpServerConfig],
    token_store: &TokenStore,
    name: &str,
) -> Result<()> {
    let base_config = servers
        .iter()
        .find(|c| c.name == name)
        .ok_or_else(|| config_err(format!("no MCP server named '{}'", name)))?
        .clone();

    let (config, needs_persist) = if base_config.auth.is_some() {
        (base_config, false)
    } else {
        match base_config.transport {
            McpTransport::Http => {
                let mut assumed = base_config.clone();
                assumed.auth = Some(McpAuthConfig::OAuth {
                    client_id: None,
                    client_secret: None,
                    scopes: None,
                    redirect_port: None,
                });
                tracing::info!(
                    "no [auth] block for '{}'; assuming OAuth authorization_code.",
                    name
                );
                (assumed, true)
            }
            McpTransport::Stdio => {
                return Err(config_err(format!(
                    "server '{}' is stdio and has no 'auth' configured; nothing to log in to",
                    name
                )));
            }
        }
    };

    token_store.clear_mcp_credentials(name).await?;
    token_store.clear_auth_probe(name).await?;

    let context = McpClientContext::new();
    // `login` is also out-of-band from the main agent loop; see the note in `run_reconnect` for why
    // we pass `None` here.
    let manager = McpClientManager::prepare(
        std::slice::from_ref(&config),
        None,
        Some(token_store.clone()),
        context,
    )
    .await?;
    manager.start_connector(crate::mcp::McpRuntimeConfig {
        connect_timeout: std::time::Duration::from_secs(30),
        stdio_concurrency: 1,
        http_concurrency: 1,
    });
    manager.await_settled().await;

    let connected = if let Some(entry) = manager.server_entry(&config.name) {
        matches!(
            &*entry.state.read().await,
            crate::mcp::ServerState::Connected { .. }
        )
    } else {
        false
    };
    if !connected {
        return Err(config_err(format!(
            "OAuth flow did not complete for '{}'",
            config.name
        )));
    }
    manager.shutdown_arc().await;

    if needs_persist && let Err(error) = persist_auth_block_for(name) {
        // Login worked. Don't fail the whole command if we can't write the config back; just
        // surface the issue so the user can decide whether to hand-edit.
        tracing::warn!(
            "'{}' is authorised, but failed to write 'auth = oauth' back to config.toml: {}",
            name,
            error
        );
    }

    tracing::info!("authorized '{}'", config.name);
    Ok(())
}

/// Write `[mcp.servers.auth] type = "oauth"` for a named server if the entry doesn't already have
/// an `auth` key. Used by [`run_login`] to make the "assumed OAuth" path a one-time thing rather
/// than silently reapplying the assumption on every future run.
fn persist_auth_block_for(name: &str) -> Result<()> {
    let path = crate::config::config_file_path()
        .ok_or_else(|| config_err("could not determine config directory"))?;
    let existing = std::fs::read_to_string(&path)
        .map_err(|error| config_err(format!("failed to read config: {}", error)))?;
    let mut document = existing
        .parse::<toml_edit::DocumentMut>()
        .map_err(|error| config_err(format!("failed to parse config: {}", error)))?;

    let mutated = {
        let servers = document
            .get_mut("mcp")
            .and_then(|m| m.as_table_mut())
            .and_then(|t| t.get_mut("servers"))
            .and_then(|s| s.as_array_of_tables_mut())
            .ok_or_else(|| config_err("config has no [[mcp.servers]] entries".to_string()))?;

        let mut target = None;
        for entry in servers.iter_mut() {
            if entry.get("name").and_then(|v| v.as_str()) == Some(name) {
                target = Some(entry);
                break;
            }
        }
        let entry = target.ok_or_else(|| {
            config_err(format!(
                "server '{}' not found in [[mcp.servers]] after login",
                name
            ))
        })?;
        if entry.contains_key("auth") {
            false
        } else {
            let mut auth_table = toml_edit::Table::new();
            auth_table.insert("type", toml_edit::value("oauth"));
            entry.insert("auth", toml_edit::Item::Table(auth_table));
            true
        }
    };

    if mutated {
        crate::config::write_file_atomic(&path, &document.to_string())
            .map_err(|error| config_err(format!("failed to write config: {}", error)))?;
    }
    Ok(())
}

/// Inputs for `meka mcp add`. Parsed into a [`ResolvedAddArgs`] by [`resolve_add_args`] which is
/// where transport auto-detection, flag compatibility, and the `McpAuthKind` → `McpAuthConfig`
/// mapping live. Keep this struct plain-data so the clap layer in `cli.rs` and the CLI integration
/// tests can both build one.
pub struct AddArgs {
    pub name: String,
    pub location: Option<String>,
    pub args: Vec<String>,
    pub transport: Option<McpTransport>,
    /// Raw `KEY=VALUE` entries from the CLI.
    pub env: Vec<String>,
    /// Raw `KEY=VALUE` entries from the CLI.
    pub header: Vec<String>,
    pub auth: Option<crate::cli::McpAuthKind>,
    pub auth_token: Option<String>,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub signing_key: Option<String>,
    pub signing_algorithm: Option<String>,
    pub scope: Vec<String>,
    pub redirect_port: Option<u16>,
    pub permission: Option<String>,
    /// Skip the auto-login that runs when the probe reports auth-required or when `--auth oauth`
    /// was explicitly set.
    pub no_login: bool,
    /// Raw tool names to allow-list (only these register).
    pub allow_tool: Vec<String>,
    /// Raw tool names to block-list (never register).
    pub disable_tool: Vec<String>,
    /// Raw tool names to eager-load (skip `load_tool` round-trip).
    pub eager_load_tool: Vec<String>,
    /// Raw `NAME=LEVEL` pairs for per-tool permission overrides.
    pub tool_permission: Vec<String>,
    /// Persist with `disabled = true` so the server is skipped at startup until the user runs
    /// `meka mcp enable <name>`.
    pub disabled: bool,
}

/// What `add` looks like after validation: transport is chosen, mutually-exclusive flag
/// combinations have been rejected, and the `[auth]` block (if any) has been reduced to an
/// [`McpAuthConfig`] ready to be serialised into TOML.
#[cfg_attr(test, derive(Debug))]
struct ResolvedAddArgs {
    name: String,
    transport: McpTransport,
    /// Present iff `transport == Stdio`.
    command: Option<String>,
    /// Present iff `transport == Stdio` and there were trailing args.
    stdio_args: Vec<String>,
    /// Present iff `transport == Stdio` and `--env` was given.
    env: Vec<(String, String)>,
    /// Present iff `transport == Http`.
    url: Option<String>,
    /// Present iff `transport == Http` and `--header` was given.
    headers: Vec<(String, String)>,
    auth_token: Option<String>,
    auth: Option<McpAuthConfig>,
    permission: Option<String>,
    allowed_tools: Option<Vec<String>>,
    disabled_tools: Option<Vec<String>>,
    eager_load_tools: Option<Vec<String>>,
    tool_permissions: Option<std::collections::HashMap<String, String>>,
    no_login: bool,
    disabled: bool,
}

/// Run `meka mcp add …`.
///
/// Persists the server into `config.toml`, then for HTTP servers:
///   1. Probes the endpoint (RFC 6750 / RFC 9728) to see if auth is required.
///   2. If the probe says auth is required (or the user explicitly passed `--auth oauth`) and
///      `--no-login` wasn't set, runs the OAuth authorization_code flow immediately so the whole
///      setup is "add + authorise" in a single command.
///   3. If that OAuth flow fails, rolls back by purging the entry we just wrote. The CLI exit is
///      non-zero.
pub async fn run_add(args: AddArgs, token_store: &TokenStore) -> Result<()> {
    use crate::mcp::sanitize::{is_reserved_server_name, normalize_server_name};

    let normalized = normalize_server_name(&args.name);
    if normalized != args.name {
        return Err(config_err(format!(
            "server name '{}' contains invalid characters (would normalise to '{}')",
            args.name, normalized
        )));
    }
    if is_reserved_server_name(&args.name) {
        return Err(config_err(format!(
            "server name '{}' is reserved",
            args.name
        )));
    }
    let resolved = resolve_add_args(args)?;

    let path = crate::config::config_file_path()
        .ok_or_else(|| config_err("could not determine config directory"))?;

    // Propagate every read error except "the file does not exist yet". The previous
    // `unwrap_or_default()` would happily silently overwrite a file we merely lacked permission to
    // read.
    let existing = match std::fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => {
            return Err(config_err(format!(
                "failed to read existing config at {}: {}",
                path.display(),
                error
            )));
        }
    };
    let mut document = existing
        .parse::<toml_edit::DocumentMut>()
        .map_err(|error| config_err(format!("failed to parse existing config: {}", error)))?;

    let servers_array = document
        .entry("mcp")
        .or_insert(toml_edit::Item::Table(toml_edit::Table::new()))
        .as_table_mut()
        .ok_or_else(|| config_err("config 'mcp' is not a table"))?
        .entry("servers")
        .or_insert(toml_edit::Item::ArrayOfTables(
            toml_edit::ArrayOfTables::new(),
        ))
        .as_array_of_tables_mut()
        .ok_or_else(|| config_err("config 'mcp.servers' is not an array of tables"))?;

    for existing_entry in servers_array.iter() {
        if existing_entry
            .get("name")
            .and_then(|v| v.as_str())
            .is_some_and(|n| n == resolved.name)
        {
            return Err(config_err(format!(
                "server '{}' already exists in config",
                resolved.name
            )));
        }
    }

    let table = build_server_table(&resolved);
    servers_array.push(table);

    crate::config::write_file_atomic(&path, &document.to_string())
        .map_err(|error| config_err(format!("failed to write config: {}", error)))?;
    tracing::info!("added '{}' to {}", resolved.name, path.display());

    // Decide whether to probe and/or auto-login. Stdio has no auth surface; HTTP servers with a
    // pre-configured static bearer don't need one either. Everything else gets the probe. These
    // short-circuits return `Ok(())` before we'd need Ctrl-C protection.
    if resolved.transport != McpTransport::Http {
        return Ok(());
    }
    if resolved.auth_token.is_some() {
        return Ok(());
    }
    if resolved.url.is_none() {
        return Ok(());
    }

    // From here on, anything that goes wrong (natural failure, timeout, Ctrl-C) should leave
    // config.toml in the "never added" state. Race the probe + auto-login block against SIGINT so
    // an interrupted user ends up in exactly the same place as a clean `mcp remove`.
    let result: Result<()> = tokio::select! {
        biased;
        _ = tokio::signal::ctrl_c() => Err(MekaError::Interrupted),
        r = probe_then_login(&resolved, token_store) => r,
    };

    if let Err(error) = result {
        match &error {
            MekaError::Interrupted => {
                tracing::warn!("interrupted; rolling back '{}'.", resolved.name);
            }
            other => tracing::warn!(
                "authorisation failed for '{}': {}; rolling back the config entry.",
                resolved.name,
                other
            ),
        }
        if let Err(purge_err) = purge_server(&resolved.name, token_store).await {
            tracing::warn!(
                "rollback of '{}' also failed: {}; you may need to edit config.toml by hand.",
                resolved.name,
                purge_err
            );
        }
        return Err(error);
    }

    Ok(())
}

/// The "everything that can fail post-persist" block: probe, decide whether to auto-login, and run
/// the OAuth flow when warranted. Extracted so [`run_add`] can race it against a SIGINT handler and
/// roll back on either error path from one place.
async fn probe_then_login(resolved: &ResolvedAddArgs, token_store: &TokenStore) -> Result<()> {
    // Caller has already verified HTTP + no auth_token + url.is_some(). Use let-else with an
    // explicit `unreachable!` so the caller-contract invariant is documented in code, not just
    // hidden behind an `expect` string.
    let Some(url) = resolved.url.as_deref() else {
        unreachable!("probe_then_login caller must verify url.is_some()");
    };

    let wants_oauth_already = matches!(resolved.auth, Some(McpAuthConfig::OAuth { .. }));
    let should_login = match probe_and_announce(&resolved.name, url).await {
        // Probe says unauthenticated access works, and the user didn't explicitly ask for OAuth →
        // nothing to log in to.
        ProbeOutcome::Open => wants_oauth_already,
        ProbeOutcome::AuthRequired => true,
        ProbeOutcome::Inconclusive => wants_oauth_already,
    };

    if !should_login {
        return Ok(());
    }
    if resolved.no_login {
        tracing::info!(
            "skipping auto-login (--no-login). Run `meka mcp login {}` when ready.",
            resolved.name
        );
        return Ok(());
    }

    // Synthesised server config for the login, equivalent to parsing the entry we just wrote but
    // without round-tripping through disk.
    let server_config = resolved_to_server_config(resolved);
    tracing::info!(
        "running OAuth authorisation for '{}' (use --no-login to skip).",
        resolved.name
    );
    run_login(
        std::slice::from_ref(&server_config),
        token_store,
        &resolved.name,
    )
    .await
}

/// What we need to know about the probe from `run_add`'s perspective. The full `McpAuthProbe`
/// detail is printed to the user (so they still see "server reachable" / "couldn't reach …"), but
/// the auto-login decision only needs the three-state summary.
#[derive(Debug, PartialEq, Eq)]
enum ProbeOutcome {
    Open,
    AuthRequired,
    Inconclusive,
}

/// Probe the HTTP endpoint, print a one-line hint, and collapse the probe result into a
/// login-decision summary.
async fn probe_and_announce(name: &str, url: &str) -> ProbeOutcome {
    use crate::mcp::auth::McpAuthProbe;

    match crate::mcp::auth::probe_http_auth(url).await {
        McpAuthProbe::Open => {
            tracing::info!("probe: '{}' reachable and does not require auth.", name);
            ProbeOutcome::Open
        }
        McpAuthProbe::AuthRequired { resource_metadata } => {
            tracing::info!("probe: '{}' requires OAuth.", name);
            if let Some(meta) = resource_metadata {
                tracing::debug!("resource_metadata advertised by '{}': {}", name, meta);
            }
            ProbeOutcome::AuthRequired
        }
        McpAuthProbe::Unexpected { status } => {
            tracing::warn!(
                "probe: '{}' answered HTTP {}; couldn't infer auth state.",
                name,
                status
            );
            ProbeOutcome::Inconclusive
        }
        McpAuthProbe::Unreachable { message } => {
            tracing::warn!("probe: couldn't reach '{}' ({}).", url, message);
            ProbeOutcome::Inconclusive
        }
    }
}

/// Turn the raw CLI [`AddArgs`] into a validated [`ResolvedAddArgs`].
///
/// Auto-detects transport from the positional `location` when `--transport` is not given
/// (`http[s]://…` → http, anything else → stdio). Rejects every illegal flag combination (stdio
/// with http-only flags, `--auth-token` together with `--auth`, OAuth flags without an OAuth-family
/// auth kind, etc.) at add time so bad configurations never land in `config.toml`.
fn resolve_add_args(args: AddArgs) -> Result<ResolvedAddArgs> {
    let AddArgs {
        name,
        location,
        args: tail,
        transport,
        env,
        header,
        auth,
        auth_token,
        client_id,
        client_secret,
        signing_key,
        signing_algorithm,
        scope,
        redirect_port,
        permission,
        no_login,
        allow_tool,
        disable_tool,
        eager_load_tool,
        tool_permission,
        disabled,
    } = args;

    let looks_like_url = location
        .as_deref()
        .map(|s| s.starts_with("http://") || s.starts_with("https://"))
        .unwrap_or(false);
    let transport = transport.unwrap_or(if looks_like_url {
        McpTransport::Http
    } else {
        McpTransport::Stdio
    });

    // Permission allow-list matches `parse_server_permission` upstream.
    if let Some(perm) = permission.as_deref() {
        match perm {
            "none" | "read" | "ask" | "write" => {}
            other => {
                return Err(config_err(format!(
                    "unknown permission '{}' (expected none, read, ask, or write)",
                    other
                )));
            }
        }
    }

    // Per-tool permission overrides arrive as `NAME=LEVEL` strings. Parse
    // + validate here so bad input never lands in config.toml.
    let tool_permissions = if tool_permission.is_empty() {
        None
    } else {
        let mut map = std::collections::HashMap::with_capacity(tool_permission.len());
        for entry in &tool_permission {
            let (tool, level) = entry.split_once('=').ok_or_else(|| {
                config_err(format!(
                    "--tool-permission expects NAME=LEVEL, got '{}'",
                    entry
                ))
            })?;
            let tool = tool.trim();
            let level = level.trim();
            if tool.is_empty() {
                return Err(config_err(format!(
                    "--tool-permission '{}' has an empty tool name",
                    entry
                )));
            }
            match level {
                "none" | "read" | "ask" | "write" => {}
                other => {
                    return Err(config_err(format!(
                        "--tool-permission '{}' has unknown level '{}' \
                         (expected none, read, ask, or write)",
                        entry, other
                    )));
                }
            }
            map.insert(tool.to_string(), level.to_string());
        }
        Some(map)
    };
    let allowed_tools = if allow_tool.is_empty() {
        None
    } else {
        Some(allow_tool)
    };
    let disabled_tools = if disable_tool.is_empty() {
        None
    } else {
        Some(disable_tool)
    };
    let eager_load_tools = if eager_load_tool.is_empty() {
        None
    } else {
        Some(eager_load_tool)
    };

    let auth_flags_present = client_id.is_some()
        || client_secret.is_some()
        || signing_key.is_some()
        || signing_algorithm.is_some()
        || !scope.is_empty()
        || redirect_port.is_some();

    if auth_token.is_some() && auth.is_some() {
        return Err(config_err("--auth-token is mutually exclusive with --auth"));
    }

    match transport {
        McpTransport::Stdio => {
            let command = location.ok_or_else(|| {
                config_err(
                    "stdio transport needs an executable (pass it as the positional argument)",
                )
            })?;
            if !header.is_empty() {
                return Err(config_err("--header is HTTP-only"));
            }
            if auth_token.is_some() || auth.is_some() || auth_flags_present {
                return Err(config_err("auth flags are HTTP-only"));
            }

            let env = parse_kv_pairs("--env", &env)?;

            Ok(ResolvedAddArgs {
                name,
                transport,
                command: Some(command),
                stdio_args: tail,
                env,
                url: None,
                headers: Vec::new(),
                auth_token: None,
                auth: None,
                permission,
                allowed_tools,
                disabled_tools,
                eager_load_tools,
                tool_permissions,
                no_login,
                disabled,
            })
        }
        McpTransport::Http => {
            let url = location.ok_or_else(|| {
                config_err("http transport needs a URL (pass it as the positional argument)")
            })?;
            if !tail.is_empty() {
                return Err(config_err(
                    "http transport doesn't take trailing positional args",
                ));
            }
            if !env.is_empty() {
                return Err(config_err("--env is stdio-only"));
            }

            let headers = parse_kv_pairs("--header", &header)?;

            let auth_config = resolve_auth_config(
                auth,
                &auth_token,
                client_id,
                client_secret,
                signing_key,
                signing_algorithm,
                scope,
                redirect_port,
            )?;

            Ok(ResolvedAddArgs {
                name,
                transport,
                command: None,
                stdio_args: Vec::new(),
                env: Vec::new(),
                url: Some(url),
                headers,
                auth_token,
                auth: auth_config,
                permission,
                allowed_tools,
                disabled_tools,
                eager_load_tools,
                tool_permissions,
                no_login,
                disabled,
            })
        }
    }
}

/// Convert the CLI's auth-related flags into an [`McpAuthConfig`] (or `None` if the user chose
/// static-token / no auth). Validates the per-variant required fields so "oauth" doesn't silently
/// accept an unrelated `--signing-key` and ship a malformed config.
///
/// The eight inputs are the independent auth-related CLI flags; grouping them behind a newtype
/// would just shuffle the destructuring burden one step upstream without adding meaning.
#[allow(clippy::too_many_arguments)]
fn resolve_auth_config(
    auth: Option<crate::cli::McpAuthKind>,
    auth_token: &Option<String>,
    client_id: Option<String>,
    client_secret: Option<String>,
    signing_key: Option<String>,
    signing_algorithm: Option<String>,
    scope: Vec<String>,
    redirect_port: Option<u16>,
) -> Result<Option<McpAuthConfig>> {
    use crate::cli::McpAuthKind;

    let auth_flags_present = client_id.is_some()
        || client_secret.is_some()
        || signing_key.is_some()
        || signing_algorithm.is_some()
        || !scope.is_empty()
        || redirect_port.is_some();

    match auth {
        None => {
            if auth_flags_present && auth_token.is_none() {
                return Err(config_err(
                    "OAuth-family flags require --auth (oauth, client-credentials, or \
                     client-credentials-jwt)",
                ));
            }
            Ok(None)
        }
        Some(McpAuthKind::OAuth) => {
            if signing_key.is_some() || signing_algorithm.is_some() {
                return Err(config_err(
                    "--signing-key and --signing-algorithm are only valid with \
                     --auth client-credentials-jwt",
                ));
            }
            Ok(Some(McpAuthConfig::OAuth {
                client_id,
                client_secret,
                scopes: if scope.is_empty() { None } else { Some(scope) },
                redirect_port,
            }))
        }
        Some(McpAuthKind::ClientCredentials) => {
            let client_id = client_id
                .ok_or_else(|| config_err("--auth client-credentials requires --client-id"))?;
            let client_secret = client_secret
                .ok_or_else(|| config_err("--auth client-credentials requires --client-secret"))?;
            if signing_key.is_some() || signing_algorithm.is_some() {
                return Err(config_err(
                    "--signing-key and --signing-algorithm are only valid with \
                     --auth client-credentials-jwt",
                ));
            }
            if redirect_port.is_some() {
                return Err(config_err(
                    "--redirect-port is only valid with --auth oauth",
                ));
            }
            Ok(Some(McpAuthConfig::ClientCredentials {
                client_id,
                client_secret,
                scopes: if scope.is_empty() { None } else { Some(scope) },
                resource: None,
            }))
        }
        Some(McpAuthKind::ClientCredentialsJwt) => {
            let client_id = client_id
                .ok_or_else(|| config_err("--auth client-credentials-jwt requires --client-id"))?;
            let signing_key_path = signing_key.ok_or_else(|| {
                config_err("--auth client-credentials-jwt requires --signing-key")
            })?;
            if client_secret.is_some() {
                return Err(config_err(
                    "--client-secret is for --auth client-credentials, not -jwt",
                ));
            }
            if redirect_port.is_some() {
                return Err(config_err(
                    "--redirect-port is only valid with --auth oauth",
                ));
            }
            Ok(Some(McpAuthConfig::ClientCredentialsJwt {
                client_id,
                signing_key_path,
                signing_algorithm,
                scopes: if scope.is_empty() { None } else { Some(scope) },
                resource: None,
            }))
        }
    }
}

/// Parse a list of `KEY=VALUE` strings from the CLI into pairs. Surfaces the originating flag name
/// in errors so users can tell `--env` and `--header` apart when both are wrong at once.
fn parse_kv_pairs(flag: &str, pairs: &[String]) -> Result<Vec<(String, String)>> {
    let mut out = Vec::with_capacity(pairs.len());
    for entry in pairs {
        let (k, v) = entry
            .split_once('=')
            .ok_or_else(|| config_err(format!("{} expects KEY=VALUE, got '{}'", flag, entry)))?;
        if k.is_empty() {
            return Err(config_err(format!(
                "{} entry '{}' has an empty key",
                flag, entry
            )));
        }
        out.push((k.to_string(), v.to_string()));
    }
    Ok(out)
}

/// Build an [`McpServerConfig`] equivalent to what parsing the entry we just wrote into
/// `config.toml` would yield. Lets the auto-login path in [`run_add`] call into [`run_login`]
/// without round-tripping through disk.
fn resolved_to_server_config(resolved: &ResolvedAddArgs) -> McpServerConfig {
    let env = if resolved.env.is_empty() {
        None
    } else {
        Some(resolved.env.iter().cloned().collect())
    };
    let headers = if resolved.headers.is_empty() {
        None
    } else {
        Some(resolved.headers.iter().cloned().collect())
    };
    let args = if resolved.stdio_args.is_empty() {
        None
    } else {
        Some(resolved.stdio_args.clone())
    };
    McpServerConfig {
        name: resolved.name.clone(),
        transport: resolved.transport.clone(),
        command: resolved.command.clone(),
        args,
        env,
        url: resolved.url.clone(),
        auth_token: resolved.auth_token.clone(),
        headers,
        headers_helper: None,
        auth: resolved.auth.clone(),
        permission: resolved.permission.clone(),
        allowed_tools: resolved.allowed_tools.clone(),
        disabled_tools: resolved.disabled_tools.clone(),
        eager_load_tools: resolved.eager_load_tools.clone(),
        tool_permissions: resolved.tool_permissions.clone(),
        disabled: resolved.disabled,
    }
}

/// Serialise a validated [`ResolvedAddArgs`] into a TOML table ready to push onto `mcp.servers`.
/// Only the fields the user actually supplied are emitted so hand-edited config files stay
/// readable.
fn build_server_table(resolved: &ResolvedAddArgs) -> toml_edit::Table {
    let mut table = toml_edit::Table::new();
    table.insert("name", toml_edit::value(resolved.name.clone()));
    table.insert(
        "transport",
        toml_edit::value(match resolved.transport {
            McpTransport::Stdio => "stdio",
            McpTransport::Http => "http",
        }),
    );

    if let Some(command) = &resolved.command {
        table.insert("command", toml_edit::value(command.clone()));
    }
    if !resolved.stdio_args.is_empty() {
        let mut arr = toml_edit::Array::new();
        for arg in &resolved.stdio_args {
            arr.push(arg.as_str());
        }
        table.insert("args", toml_edit::Item::Value(toml_edit::Value::Array(arr)));
    }
    if !resolved.env.is_empty() {
        let mut env_table = toml_edit::InlineTable::new();
        for (k, v) in &resolved.env {
            env_table.insert(k, toml_edit::Value::from(v.as_str()));
        }
        table.insert(
            "env",
            toml_edit::Item::Value(toml_edit::Value::InlineTable(env_table)),
        );
    }
    if let Some(url) = &resolved.url {
        table.insert("url", toml_edit::value(url.clone()));
    }
    if !resolved.headers.is_empty() {
        let mut h = toml_edit::InlineTable::new();
        for (k, v) in &resolved.headers {
            h.insert(k, toml_edit::Value::from(v.as_str()));
        }
        table.insert(
            "headers",
            toml_edit::Item::Value(toml_edit::Value::InlineTable(h)),
        );
    }
    if let Some(token) = &resolved.auth_token {
        table.insert("auth_token", toml_edit::value(token.clone()));
    }
    if let Some(permission) = &resolved.permission {
        table.insert("permission", toml_edit::value(permission.clone()));
    }
    if resolved.disabled {
        table.insert("disabled", toml_edit::value(true));
    }
    if let Some(auth) = &resolved.auth {
        table.insert("auth", toml_edit::Item::Table(auth_to_toml(auth)));
    }
    if let Some(allowed) = resolved.allowed_tools.as_deref() {
        let mut arr = toml_edit::Array::new();
        for name in allowed {
            arr.push(name.as_str());
        }
        table.insert(
            "allowed_tools",
            toml_edit::Item::Value(toml_edit::Value::Array(arr)),
        );
    }
    if let Some(disabled) = resolved.disabled_tools.as_deref() {
        let mut arr = toml_edit::Array::new();
        for name in disabled {
            arr.push(name.as_str());
        }
        table.insert(
            "disabled_tools",
            toml_edit::Item::Value(toml_edit::Value::Array(arr)),
        );
    }
    if let Some(perms) = resolved.tool_permissions.as_ref()
        && !perms.is_empty()
    {
        let mut tpt = toml_edit::Table::new();
        // Stable key order so the TOML diff is review-friendly.
        let mut keys: Vec<&String> = perms.keys().collect();
        keys.sort();
        for key in keys {
            tpt.insert(key, toml_edit::value(perms[key].clone()));
        }
        table.insert("tool_permissions", toml_edit::Item::Table(tpt));
    }
    table
}

fn auth_to_toml(auth: &McpAuthConfig) -> toml_edit::Table {
    let mut t = toml_edit::Table::new();
    match auth {
        McpAuthConfig::OAuth {
            client_id,
            client_secret,
            scopes,
            redirect_port,
        } => {
            t.insert("type", toml_edit::value("oauth"));
            if let Some(id) = client_id {
                t.insert("client_id", toml_edit::value(id.clone()));
            }
            if let Some(secret) = client_secret {
                t.insert("client_secret", toml_edit::value(secret.clone()));
            }
            insert_string_array(&mut t, "scopes", scopes.as_deref());
            if let Some(port) = redirect_port {
                t.insert("redirect_port", toml_edit::value(*port as i64));
            }
        }
        McpAuthConfig::ClientCredentials {
            client_id,
            client_secret,
            scopes,
            resource,
        } => {
            t.insert("type", toml_edit::value("client_credentials"));
            t.insert("client_id", toml_edit::value(client_id.clone()));
            t.insert("client_secret", toml_edit::value(client_secret.clone()));
            insert_string_array(&mut t, "scopes", scopes.as_deref());
            if let Some(resource) = resource {
                t.insert("resource", toml_edit::value(resource.clone()));
            }
        }
        McpAuthConfig::ClientCredentialsJwt {
            client_id,
            signing_key_path,
            signing_algorithm,
            scopes,
            resource,
        } => {
            t.insert("type", toml_edit::value("client_credentials_jwt"));
            t.insert("client_id", toml_edit::value(client_id.clone()));
            t.insert(
                "signing_key_path",
                toml_edit::value(signing_key_path.clone()),
            );
            if let Some(alg) = signing_algorithm {
                t.insert("signing_algorithm", toml_edit::value(alg.clone()));
            }
            insert_string_array(&mut t, "scopes", scopes.as_deref());
            if let Some(resource) = resource {
                t.insert("resource", toml_edit::value(resource.clone()));
            }
        }
    }
    t
}

fn insert_string_array(table: &mut toml_edit::Table, key: &str, values: Option<&[String]>) {
    let Some(values) = values else {
        return;
    };
    if values.is_empty() {
        return;
    }
    let mut arr = toml_edit::Array::new();
    for v in values {
        arr.push(v.as_str());
    }
    table.insert(key, toml_edit::Item::Value(toml_edit::Value::Array(arr)));
}

/// Wipe every trace of `name`: the `[[mcp.servers]]` entry in `config.toml`, any stored OAuth
/// credentials (revoked server-side via RFC 7009 first, best-effort), the auth-probe cache row, and
/// any resource-update ledger entries. Silent: callers print their own user-facing line. Used by
/// both `run_remove` (user-invoked) and `run_add`'s auto-login rollback path (on OAuth failure
/// after the config entry has already been written).
async fn purge_server(name: &str, token_store: &TokenStore) -> Result<std::path::PathBuf> {
    let path = crate::config::config_file_path()
        .ok_or_else(|| config_err("could not determine config directory"))?;
    let existing = std::fs::read_to_string(&path)
        .map_err(|error| config_err(format!("failed to read config: {}", error)))?;
    let mut document = existing
        .parse::<toml_edit::DocumentMut>()
        .map_err(|error| config_err(format!("failed to parse config: {}", error)))?;

    let servers = document
        .get_mut("mcp")
        .and_then(|m| m.as_table_mut())
        .and_then(|t| t.get_mut("servers"))
        .and_then(|s| s.as_array_of_tables_mut());

    let Some(servers) = servers else {
        return Err(config_err(
            "no MCP servers in config; nothing to remove".to_string(),
        ));
    };

    let original_len = servers.len();
    servers.retain(|entry| entry.get("name").and_then(|v| v.as_str()) != Some(name));
    if servers.len() == original_len {
        return Err(config_err(format!("no server named '{}' in config", name)));
    }

    crate::config::write_file_atomic(&path, &document.to_string())
        .map_err(|error| config_err(format!("failed to write config: {}", error)))?;

    // Best-effort OAuth token revocation per RFC 7009 before we drop the local credentials. If the
    // caller is rolling back a never-succeeded login there won't be any credentials;
    // `revoke_stored_token` early- returns in that case so the call is safe regardless.
    if let Err(error) = crate::mcp::auth::revoke_stored_token(token_store, name).await {
        tracing::warn!(
            "failed to revoke token at server '{}' during purge: {} (continuing)",
            name,
            error
        );
    }

    token_store.clear_mcp_credentials(name).await?;
    token_store.clear_auth_probe(name).await?;
    crate::mcp::resource_updates::clear_for_server(name);
    Ok(path)
}

/// Run `meka mcp remove <name>`: delete the entry from config.toml, best-effort revoke OAuth
/// tokens at the provider, clear local state.
pub async fn run_remove(name: &str, token_store: &TokenStore) -> Result<()> {
    let path = purge_server(name, token_store).await?;
    tracing::info!("removed '{}' from {}", name, path.display());
    Ok(())
}

/// Set `disabled = <value>` on a server entry in config.toml, preserving other fields and
/// formatting. Backs `meka mcp disable|enable`. Writes atomically via
/// [`crate::config::write_file_atomic`].
async fn set_server_disabled(name: &str, disabled: bool) -> Result<std::path::PathBuf> {
    let path = crate::config::config_file_path()
        .ok_or_else(|| config_err("could not determine config directory"))?;
    let existing = std::fs::read_to_string(&path)
        .map_err(|error| config_err(format!("failed to read config: {}", error)))?;
    let mut document = existing
        .parse::<toml_edit::DocumentMut>()
        .map_err(|error| config_err(format!("failed to parse config: {}", error)))?;

    let servers = document
        .get_mut("mcp")
        .and_then(|m| m.as_table_mut())
        .and_then(|t| t.get_mut("servers"))
        .and_then(|s| s.as_array_of_tables_mut())
        .ok_or_else(|| config_err("no MCP servers in config"))?;

    let entry = servers
        .iter_mut()
        .find(|entry| entry.get("name").and_then(|v| v.as_str()) == Some(name))
        .ok_or_else(|| config_err(format!("no server named '{}' in config", name)))?;

    if disabled {
        entry.insert("disabled", toml_edit::value(true));
    } else {
        // Remove the key entirely rather than setting `false`. This keeps minimal diffs for users
        // who never enabled the flag before.
        entry.remove("disabled");
    }

    crate::config::write_file_atomic(&path, &document.to_string())
        .map_err(|error| config_err(format!("failed to write config: {}", error)))?;
    Ok(path)
}

/// Run `meka mcp disable <name>`. Sets `disabled = true` in config.toml. The currently-running meka
/// session (if any) keeps its state; the change takes effect on the next start.
pub async fn run_disable(name: &str) -> Result<()> {
    let path = set_server_disabled(name, true).await?;
    tracing::info!("disabled '{}' in {}", name, path.display());
    Ok(())
}

/// Run `meka mcp enable <name>`. Clears `disabled` from the server entry in config.toml.
pub async fn run_enable(name: &str) -> Result<()> {
    let path = set_server_disabled(name, false).await?;
    tracing::info!("enabled '{}' in {}", name, path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bare_add(name: &str, location: Option<&str>) -> AddArgs {
        AddArgs {
            name: name.to_string(),
            location: location.map(str::to_string),
            args: Vec::new(),
            transport: None,
            env: Vec::new(),
            header: Vec::new(),
            auth: None,
            auth_token: None,
            client_id: None,
            client_secret: None,
            signing_key: None,
            signing_algorithm: None,
            scope: Vec::new(),
            redirect_port: None,
            permission: None,
            no_login: false,
            allow_tool: Vec::new(),
            disable_tool: Vec::new(),
            eager_load_tool: Vec::new(),
            tool_permission: Vec::new(),
            disabled: false,
        }
    }

    #[test]
    fn resolve_autodetects_http_from_url() {
        let resolved = resolve_add_args(bare_add("notion", Some("https://mcp.notion.com/mcp")))
            .expect("should resolve");
        assert_eq!(resolved.transport, McpTransport::Http);
        assert_eq!(resolved.url.as_deref(), Some("https://mcp.notion.com/mcp"));
        assert!(resolved.command.is_none());
        assert!(resolved.auth.is_none());
    }

    #[test]
    fn resolve_autodetects_stdio_from_command() {
        let mut args = bare_add("pg", Some("npx"));
        args.args = vec![
            "-y".to_string(),
            "@modelcontextprotocol/server-postgres".to_string(),
        ];
        let resolved = resolve_add_args(args).expect("should resolve");
        assert_eq!(resolved.transport, McpTransport::Stdio);
        assert_eq!(resolved.command.as_deref(), Some("npx"));
        assert_eq!(resolved.stdio_args.len(), 2);
        assert!(resolved.url.is_none());
    }

    #[test]
    fn resolve_http_requires_location() {
        let err = resolve_add_args({
            let mut args = bare_add("srv", None);
            args.transport = Some(McpTransport::Http);
            args
        })
        .expect_err("http with no URL should error");
        assert!(format!("{}", err).contains("http transport needs a URL"));
    }

    #[test]
    fn resolve_stdio_requires_command() {
        let err = resolve_add_args({
            let mut args = bare_add("srv", None);
            args.transport = Some(McpTransport::Stdio);
            args
        })
        .expect_err("stdio with no command should error");
        assert!(format!("{}", err).contains("stdio transport needs an executable"));
    }

    #[test]
    fn resolve_http_rejects_trailing_args() {
        let mut args = bare_add("srv", Some("https://example.com"));
        args.args = vec!["extra".to_string()];
        let err = resolve_add_args(args).expect_err("should reject trailing args on http");
        assert!(format!("{}", err).contains("trailing positional args"));
    }

    #[test]
    fn resolve_rejects_env_on_http() {
        let mut args = bare_add("srv", Some("https://example.com"));
        args.env = vec!["K=V".to_string()];
        let err = resolve_add_args(args).expect_err("env on http should error");
        assert!(format!("{}", err).contains("--env is stdio-only"));
    }

    #[test]
    fn resolve_rejects_header_on_stdio() {
        let mut args = bare_add("srv", Some("/usr/bin/mcp"));
        args.header = vec!["X-Custom=1".to_string()];
        let err = resolve_add_args(args).expect_err("header on stdio should error");
        assert!(format!("{}", err).contains("--header is HTTP-only"));
    }

    #[test]
    fn resolve_rejects_auth_token_with_auth_flag() {
        let mut args = bare_add("srv", Some("https://example.com"));
        args.auth_token = Some("tok".to_string());
        args.auth = Some(crate::cli::McpAuthKind::OAuth);
        let err = resolve_add_args(args).expect_err("mutually exclusive flags");
        assert!(format!("{}", err).contains("mutually exclusive"));
    }

    #[test]
    fn resolve_client_credentials_requires_id_and_secret() {
        let mut args = bare_add("srv", Some("https://example.com"));
        args.auth = Some(crate::cli::McpAuthKind::ClientCredentials);
        let err = resolve_add_args(args).expect_err("missing client-id");
        assert!(format!("{}", err).contains("--client-id"));
    }

    #[test]
    fn resolve_oauth_builds_empty_config_when_no_flags() {
        let mut args = bare_add("notion", Some("https://mcp.notion.com/mcp"));
        args.auth = Some(crate::cli::McpAuthKind::OAuth);
        let resolved = resolve_add_args(args).expect("should resolve");
        match resolved.auth {
            Some(McpAuthConfig::OAuth {
                client_id,
                client_secret,
                scopes,
                redirect_port,
            }) => {
                assert!(client_id.is_none());
                assert!(client_secret.is_none());
                assert!(scopes.is_none());
                assert!(redirect_port.is_none());
            }
            other => panic!("expected OAuth auth, got {:?}", other.is_some()),
        }
    }

    #[test]
    fn resolve_rejects_oauth_flags_without_auth() {
        let mut args = bare_add("srv", Some("https://example.com"));
        args.client_id = Some("id".to_string());
        let err = resolve_add_args(args).expect_err("orphan oauth flag");
        assert!(format!("{}", err).contains("OAuth-family flags"));
    }

    #[test]
    fn resolve_auth_token_works_alone() {
        let mut args = bare_add("srv", Some("https://example.com"));
        args.auth_token = Some("bearer-xyz".to_string());
        let resolved = resolve_add_args(args).expect("should resolve");
        assert_eq!(resolved.auth_token.as_deref(), Some("bearer-xyz"));
        assert!(resolved.auth.is_none());
    }

    #[test]
    fn parse_kv_pairs_rejects_missing_separator() {
        let err = parse_kv_pairs("--env", &["bad".to_string()]).expect_err("no = should error");
        assert!(format!("{}", err).contains("--env"));
        assert!(format!("{}", err).contains("KEY=VALUE"));
    }

    #[test]
    fn parse_kv_pairs_rejects_empty_key() {
        let err =
            parse_kv_pairs("--header", &["=val".to_string()]).expect_err("empty key should error");
        assert!(format!("{}", err).contains("empty key"));
    }

    #[test]
    fn build_server_table_emits_oauth_block() {
        let mut args = bare_add("notion", Some("https://mcp.notion.com/mcp"));
        args.auth = Some(crate::cli::McpAuthKind::OAuth);
        args.scope = vec!["read".to_string(), "write".to_string()];
        args.redirect_port = Some(8400);
        let resolved = resolve_add_args(args).expect("resolve");
        // Parse it back through toml to confirm the schema matches what ResolvedConfig expects.
        // Checking the textual rendering is fragile because toml_edit decides when to emit a
        // standalone `[mcp.servers.auth]` header.
        let mut doc = toml_edit::DocumentMut::new();
        let mut servers = toml_edit::ArrayOfTables::new();
        servers.push(build_server_table(&resolved));
        doc.insert(
            "mcp",
            toml_edit::Item::Table({
                let mut t = toml_edit::Table::new();
                t.insert("servers", toml_edit::Item::ArrayOfTables(servers));
                t
            }),
        );
        let parsed: crate::config::ConfigFile =
            toml::from_str(&doc.to_string()).expect("valid config");
        let servers = parsed.mcp.expect("mcp").servers.expect("servers");
        assert_eq!(servers.len(), 1);
        let server = &servers[0];
        assert_eq!(server.name, "notion");
        assert!(matches!(
            &server.auth,
            Some(crate::config::McpAuthConfig::OAuth { redirect_port: Some(8400), scopes: Some(s), .. })
            if s == &vec!["read".to_string(), "write".to_string()]
        ));
    }

    #[test]
    fn describe_one_line_collapses_whitespace() {
        assert_eq!(describe_one_line("one\n\ntwo  three"), "one two three");
    }

    #[test]
    fn describe_one_line_short_input_passes_through() {
        assert_eq!(describe_one_line("Read a file."), "Read a file.");
    }

    #[test]
    fn describe_one_line_caps_at_80_chars_with_ellipsis() {
        let long = "a".repeat(200);
        let out = describe_one_line(&long);
        assert!(out.ends_with('…'));
        assert!(out.chars().count() <= 81);
    }

    #[test]
    fn describe_one_line_empty_passes_through() {
        assert_eq!(describe_one_line(""), "");
    }
}
