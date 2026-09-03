//! OAuth / JWT authentication and HTTP callback handling for MCP HTTP transports. Also houses the
//! probe that classifies an unauthenticated response (RFC 6750 + RFC 9728), the best-effort token
//! revocation path (RFC 7009), and the SQLite-backed credential store threaded into `rmcp`'s
//! `AuthorizationManager`.

use async_trait::async_trait;
use rmcp::{
    ServiceExt,
    transport::{
        AuthClient, AuthError, AuthorizationManager, ClientCredentialsConfig, CredentialStore,
        StoredCredentials, auth::OAuthState,
    },
};

use super::{McpRunningService, handler::MekaClientHandler};
use crate::{
    config::McpAuthConfig,
    error::{MekaError, Result},
    session::TokenStore,
};

/// The server URL a stored OAuth bundle was issued for, if it records one.
///
/// rmcp writes `server_url` (older bundles: `issuer`) into the credential it hands the store, so
/// the origin has always been on disk -- nothing surfaced it. Which matters because the row is
/// keyed by `(server_name, kind)` and nothing else: point `[[mcp_servers]]` named `docs` at a
/// different host and the token minted for the old one is presented to the new one, with no
/// indication in `meka mcp get` that the two disagree.
///
/// Best-effort by design, and `None` in three different ways that all mean the same thing to a
/// caller: no bundle, a bundle that is not JSON, or one from a version of rmcp that recorded
/// neither key. This decorates a listing; it must not fail one.
pub async fn stored_credential_origin(
    token_store: &TokenStore,
    server_name: &str,
) -> Option<String> {
    let json = match token_store
        .load_mcp_credentials(server_name, crate::session::McpCredentialKind::OAuth)
        .await
    {
        Ok(json) => json?,
        // A store meka cannot read is not the same fact as a server with no stored grant, and this
        // function's `None` says the second. Logged rather than propagated: the caller is a display
        // line, and losing one row of `mcp get` is not worth failing the command over -- but a
        // silent drop would present a locked store as "never logged in".
        Err(error) => {
            tracing::debug!(
                "could not read the stored OAuth credential for {} to report its origin: {}",
                server_name,
                error
            );
            return None;
        }
    };
    let parsed: serde_json::Value = serde_json::from_str(&json).ok()?;
    origin_of(&parsed).map(str::to_string)
}

/// Whether two URLs name the same server, for the purpose of "is this the grant for this host".
///
/// Scheme, host and port, which is the tuple that decides who receives the bearer token. Path is
/// deliberately excluded: an MCP endpoint at `/mcp` and the issuer metadata at the site root are
/// the same server, and comparing paths would report a mismatch on every correctly configured
/// OAuth server. Both unparseable falls back to string equality, so a comparison meka cannot make
/// says "the same" rather than crying wolf on a listing.
pub(crate) fn same_origin(left: &str, right: &str) -> bool {
    match (reqwest::Url::parse(left), reqwest::Url::parse(right)) {
        (Ok(left), Ok(right)) => {
            left.scheme() == right.scheme()
                && left.host_str() == right.host_str()
                && left.port_or_known_default() == right.port_or_known_default()
        }
        _ => left == right,
    }
}

/// The issuer field out of a parsed bundle, in the order [`revoke_stored_token`] reads them.
///
/// Shared so the two cannot disagree about which key is authoritative: a bundle carrying both must
/// revoke against the same host `mcp get` names, or the report is about a different endpoint than
/// the one the token would be sent to.
fn origin_of(parsed: &serde_json::Value) -> Option<&str> {
    parsed
        .get("server_url")
        .and_then(|value| value.as_str())
        .or_else(|| parsed.get("issuer").and_then(|value| value.as_str()))
}

/// Best-effort revoke of a stored OAuth access/refresh token for an MCP server. Looks up the stored
/// credentials, discovers the provider's revocation endpoint via the OAuth authorization server
/// metadata, and posts `token=…&token_type_hint=access_token` per RFC 7009. Errors are propagated
/// so the caller can log them; local credential cleanup should run regardless.
pub async fn revoke_stored_token(
    token_store: &TokenStore,
    server_name: &str,
) -> std::result::Result<(), String> {
    let Some(json) = token_store
        .load_mcp_credentials(server_name, crate::session::McpCredentialKind::OAuth)
        .await
        .map_err(|error| format!("load credentials: {}", error))?
    else {
        return Ok(());
    };
    let parsed: serde_json::Value = serde_json::from_str(&json)
        .map_err(|error| format!("stored credentials are not valid JSON: {}", error))?;
    let issuer = origin_of(&parsed)
        .ok_or_else(|| "stored credentials missing issuer/server_url".to_string())?;
    let access_token = parsed
        .get("tokens")
        .and_then(|t| t.get("access_token"))
        .and_then(|v| v.as_str());
    let client_id = parsed
        .get("client")
        .and_then(|c| c.get("client_id"))
        .and_then(|v| v.as_str());

    let Some(access_token) = access_token else {
        return Ok(());
    };

    // Discover the revocation endpoint. RFC 8414 says it lives under
    // /.well-known/oauth-authorization-server; many providers also expose it under
    // /.well-known/openid-configuration. Try OAuth first.
    //
    // Threat model: the `issuer` URL comes from credentials we stored during the original auth
    // flow, so we trust the origin. We do NOT trust the network path or any redirect: reqwest
    // follows redirects by default, which would let a MITM redirect the metadata fetch to an
    // attacker host and coax us into POSTing the access token there. Redirects are turned off, the
    // response body is size-capped, and the returned `revocation_endpoint` is pinned to the same
    // host as the issuer.
    const METADATA_BODY_CAP: usize = 256 * 1024;
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|error| format!("build http client: {}", error))?;

    let base = issuer.trim_end_matches('/');
    let candidates = [
        format!("{}/.well-known/oauth-authorization-server", base),
        format!("{}/.well-known/openid-configuration", base),
    ];
    let mut revocation_endpoint: Option<String> = None;
    for url in &candidates {
        let Ok(response) = http.get(url).send().await else {
            continue;
        };
        if !response.status().is_success() {
            continue;
        }
        // Read bytes before parsing so we can size-cap: reqwest's own Content-Length is
        // server-supplied and therefore untrusted.
        let Ok(bytes) = response.bytes().await else {
            continue;
        };
        if let Some(endpoint) = extract_revocation_endpoint(&bytes, METADATA_BODY_CAP) {
            revocation_endpoint = Some(endpoint);
            break;
        }
    }

    let Some(endpoint) = revocation_endpoint else {
        return Err(format!(
            "server '{}' does not advertise a revocation_endpoint",
            server_name
        ));
    };

    validate_revocation_endpoint_origin(issuer, &endpoint)?;

    // Build application/x-www-form-urlencoded body manually so we don't need an extra dependency.
    // `form_urlencoded` uses %-encoded UTF-8, same as `percent_encoding::NON_ALPHANUMERIC`.
    use percent_encoding::{NON_ALPHANUMERIC, utf8_percent_encode};
    let enc = |s: &str| utf8_percent_encode(s, NON_ALPHANUMERIC).to_string();
    let mut body = format!("token={}&token_type_hint=access_token", enc(access_token));
    if let Some(id) = client_id {
        body.push_str("&client_id=");
        body.push_str(&enc(id));
    }

    let response = http
        .post(&endpoint)
        .header(
            reqwest::header::CONTENT_TYPE,
            "application/x-www-form-urlencoded",
        )
        .body(body)
        .send()
        .await
        .map_err(|error| {
            format!(
                "revoke POST failed: {}",
                crate::error::format_reqwest_error(&error)
            )
        })?;
    // RFC 7009: successful revocation is 200 OK with an empty body; unknown tokens also return 200
    // OK. Non-2xx is a genuine failure.
    if !response.status().is_success() {
        return Err(format!(
            "revoke POST returned HTTP {}",
            response.status().as_u16()
        ));
    }
    Ok(())
}

/// Classification of an unauthenticated probe to an HTTP MCP endpoint. The MCP authorization spec
/// (2025-03-26) layers on RFC 6750 + RFC 9728: a server that requires auth answers unauthenticated
/// requests with `401` and a `WWW-Authenticate: Bearer …` challenge, optionally advertising a
/// `resource_metadata` URL we can fetch to learn which authorization servers + scopes to use.
#[derive(Debug, PartialEq, Eq)]
pub enum McpAuthProbe {
    /// Server answered 2xx: reachable and doesn't require auth.
    Open,
    /// Server answered 401 / 403 with a `Bearer` challenge. The optional URL is the RFC 9728
    /// protected-resource-metadata document.
    AuthRequired { resource_metadata: Option<String> },
    /// Reachable but some other status (405, 404, …). Record it so the caller can surface it
    /// without claiming auth is or isn't needed.
    Unexpected { status: u16 },
    /// Couldn't even talk to the server (DNS, TLS, timeout, …).
    Unreachable { message: String },
}

/// Probe an MCP HTTP endpoint to see whether it requires OAuth.
///
/// Runs an unauthenticated `GET` with a 3 s wall-clock timeout and redirects disabled; we never
/// follow off-origin so a compromised DNS can't bait us into treating an attacker host as
/// authoritative about the real server. The body is ignored; the verdict comes entirely from the
/// status line and the `WWW-Authenticate` header.
pub async fn probe_http_auth(url: &str) -> McpAuthProbe {
    let http = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .redirect(reqwest::redirect::Policy::none())
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            return McpAuthProbe::Unreachable {
                message: format!("build http client: {}", error),
            };
        }
    };
    let response = match http.get(url).send().await {
        Ok(response) => response,
        Err(error) => {
            return McpAuthProbe::Unreachable {
                message: crate::error::format_reqwest_error(&error),
            };
        }
    };
    let status = response.status().as_u16();
    let www_authenticate = response
        .headers()
        .get(reqwest::header::WWW_AUTHENTICATE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    classify_probe_response(status, www_authenticate.as_deref())
}

/// Pure classifier for the probe: takes the status code + optional `WWW-Authenticate` header and
/// returns the [`McpAuthProbe`] verdict. Extracted so the RFC-6750 / RFC-9728 parsing can be
/// unit-tested without a live HTTP server.
fn classify_probe_response(status: u16, www_authenticate: Option<&str>) -> McpAuthProbe {
    if (200..300).contains(&status) {
        return McpAuthProbe::Open;
    }
    if status == 401 || status == 403 {
        let header = www_authenticate.unwrap_or("");
        // RFC 6750 §3: the challenge must start with `Bearer`. Match case-insensitively; tolerate
        // the scheme with or without any `key=value` parameters following.
        let first = header.split(',').next().unwrap_or("").trim();
        let is_bearer = first.eq_ignore_ascii_case("Bearer")
            || first
                .split_once(|c: char| c.is_whitespace())
                .map(|(scheme, _)| scheme.eq_ignore_ascii_case("Bearer"))
                .unwrap_or(false);
        if !is_bearer {
            return McpAuthProbe::Unexpected { status };
        }
        return McpAuthProbe::AuthRequired {
            resource_metadata: extract_bearer_param(header, "resource_metadata"),
        };
    }
    McpAuthProbe::Unexpected { status }
}

/// Extract a quoted parameter value from an RFC 6750 `WWW-Authenticate:
/// Bearer …` challenge. Handles the forms seen in the wild:
/// `key="value"`, `key=value`, trailing commas, mixed whitespace.
/// Returns `None` if the key isn't present.
fn extract_bearer_param(header: &str, key: &str) -> Option<String> {
    // Drop the `Bearer` scheme prefix; everything after is a comma-separated parameter list.
    let idx = header.find(|c: char| c.is_whitespace())?;
    let params = &header[idx..];
    for pair in params.split(',') {
        let pair = pair.trim();
        let Some((k, v)) = pair.split_once('=') else {
            continue;
        };
        if !k.trim().eq_ignore_ascii_case(key) {
            continue;
        }
        let v = v.trim();
        let unquoted = v
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .unwrap_or(v);
        return Some(unquoted.to_string());
    }
    None
}

/// Parse an OAuth authorization-server metadata JSON document and return the `revocation_endpoint`
/// string, if any. Rejects bodies larger than `max_bytes` and invalid JSON. Split from
/// [`revoke_stored_token`] so the size-cap and extraction logic are testable without a live HTTP
/// server.
fn extract_revocation_endpoint(bytes: &[u8], max_bytes: usize) -> Option<String> {
    if bytes.len() > max_bytes {
        return None;
    }
    let metadata = serde_json::from_slice::<serde_json::Value>(bytes).ok()?;
    metadata
        .get("revocation_endpoint")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Verify that `endpoint` has the same scheme, host, and effective port as `issuer`. Prevents a
/// compromised metadata document from redirecting the access-token POST to an attacker-controlled
/// host.
fn validate_revocation_endpoint_origin(
    issuer: &str,
    endpoint: &str,
) -> std::result::Result<(), String> {
    let issuer_url = reqwest::Url::parse(issuer)
        .map_err(|error| format!("stored issuer '{}' is not a valid URL: {}", issuer, error))?;
    let endpoint_url = reqwest::Url::parse(endpoint).map_err(|error| {
        format!(
            "revocation_endpoint '{}' is not a valid URL: {}",
            endpoint, error
        )
    })?;
    if endpoint_url.scheme() != issuer_url.scheme()
        || endpoint_url.host_str() != issuer_url.host_str()
        || endpoint_url.port_or_known_default() != issuer_url.port_or_known_default()
    {
        return Err(format!(
            "revocation_endpoint '{}' is on a different origin than issuer '{}'; refusing to send token",
            endpoint, issuer
        ));
    }
    Ok(())
}

pub(super) async fn connect_http_with_oauth(
    server_name: &str,
    url: &str,
    auth_config: &McpAuthConfig,
    transport_config: rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig,
    token_store: Option<&TokenStore>,
    handler: MekaClientHandler,
) -> Result<McpRunningService> {
    // The client half of this server's identity, from the store. Loaded once here rather than in
    // each arm: `client-credentials` requires one, `oauth` takes one only for a confidential
    // client, and `client-credentials-jwt` signs instead of sharing a secret and wants none.
    let client_secret = match token_store {
        Some(store) => {
            store
                .load_mcp_credentials(server_name, crate::session::McpCredentialKind::ClientSecret)
                .await?
        }
        None => None,
    };

    let auth_manager = match auth_config {
        McpAuthConfig::ClientCredentials {
            client_id,
            scopes,
            resource,
        } => {
            // Required for this flow, and there is nothing to fall back to: without it meka cannot
            // prove which client it is and the token endpoint will refuse.
            let secret = client_secret.as_deref().ok_or_else(|| MekaError::McpAuth {
                server_name: server_name.to_string(),
                message: format!(
                    "no client secret stored for '{}'. Run `meka mcp login {} \
                     --client-secret-stdin`",
                    server_name, server_name
                ),
            })?;
            authenticate_client_credentials(
                server_name,
                url,
                client_id,
                secret,
                scopes.as_deref(),
                resource.as_deref(),
            )
            .await?
        }
        McpAuthConfig::ClientCredentialsJwt {
            client_id,
            signing_key_path,
            signing_algorithm,
            scopes,
            resource,
        } => {
            authenticate_client_credentials_jwt(
                server_name,
                url,
                client_id,
                signing_key_path,
                signing_algorithm.as_deref(),
                scopes.as_deref(),
                resource.as_deref(),
            )
            .await?
        }
        McpAuthConfig::OAuth {
            client_id,
            scopes,
            redirect_port,
        } => {
            // Optional here: a public client has none, and a confidential one's is whatever the
            // store holds. Either way this is the only place it can come from now.
            authenticate_oauth_authorization_code(
                server_name,
                url,
                client_id.as_deref(),
                client_secret.as_deref(),
                scopes.as_deref(),
                *redirect_port,
                token_store,
            )
            .await?
        }
    };

    let auth_client = AuthClient::new(reqwest::Client::new(), auth_manager);
    let transport =
        rmcp::transport::StreamableHttpClientTransport::with_client(auth_client, transport_config);

    handler
        .serve(transport)
        .await
        .map_err(|error| MekaError::McpConnection {
            server_name: server_name.to_string(),
            message: format!("HTTP connection failed: {}", error),
        })
}

async fn authenticate_client_credentials(
    server_name: &str,
    url: &str,
    client_id: &str,
    client_secret: &str,
    scopes: Option<&[String]>,
    resource: Option<&str>,
) -> Result<AuthorizationManager> {
    let config = ClientCredentialsConfig::ClientSecret {
        client_id: client_id.to_string(),
        client_secret: client_secret.to_string(),
        scopes: scopes.map(|s| s.to_vec()).unwrap_or_default(),
        resource: resource.map(|s| s.to_string()),
    };

    let mut oauth_state = OAuthState::new(url, None)
        .await
        .map_err(|error| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("failed to initialize OAuth: {}", error),
        })?;

    oauth_state
        .authenticate_client_credentials(config)
        .await
        .map_err(|error| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("client credentials authentication failed: {}", error),
        })?;

    oauth_state
        .into_authorization_manager()
        .ok_or_else(|| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: "unexpected OAuth state after client credentials authentication".to_string(),
        })
}

async fn authenticate_client_credentials_jwt(
    server_name: &str,
    url: &str,
    client_id: &str,
    signing_key_path: &str,
    signing_algorithm: Option<&str>,
    scopes: Option<&[String]>,
    resource: Option<&str>,
) -> Result<AuthorizationManager> {
    // Open the key once and check permissions on the open fd to close the stat-then-read TOCTOU
    // window: a separate `metadata(path)` followed by `read(path)` could land on a different inode
    // if the path was swapped between syscalls. `File::metadata` walks the open descriptor.
    let mut key_file =
        std::fs::File::open(signing_key_path).map_err(|error| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: format!(
                "failed to open signing key '{}': {}",
                signing_key_path, error
            ),
        })?;
    require_private_key_permissions_on_fd(server_name, signing_key_path, &key_file)?;
    let mut signing_key = Vec::new();
    {
        use std::io::Read;
        key_file
            .read_to_end(&mut signing_key)
            .map_err(|error| MekaError::McpAuth {
                server_name: server_name.to_string(),
                message: format!(
                    "failed to read signing key '{}': {}",
                    signing_key_path, error
                ),
            })?;
    }

    let algorithm = parse_jwt_signing_algorithm(server_name, signing_algorithm)?;

    let config = ClientCredentialsConfig::PrivateKeyJwt {
        client_id: client_id.to_string(),
        signing_key,
        signing_algorithm: algorithm,
        scopes: scopes.map(|s| s.to_vec()).unwrap_or_default(),
        resource: resource.map(|s| s.to_string()),
        token_endpoint_audience: None,
    };

    let mut oauth_state = OAuthState::new(url, None)
        .await
        .map_err(|error| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("failed to initialize OAuth: {}", error),
        })?;

    oauth_state
        .authenticate_client_credentials(config)
        .await
        .map_err(|error| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("JWT client credentials authentication failed: {}", error),
        })?;

    oauth_state
        .into_authorization_manager()
        .ok_or_else(|| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: "unexpected OAuth state after JWT authentication".to_string(),
        })
}

/// On Unix, refuse to read a JWT signing key that is group- or world- accessible. Matches the
/// 0600-only policy already applied to the session DB and config.toml: if the key can be read by
/// another local user, a local attacker can forge JWTs to the MCP server and impersonate us.
///
/// Takes the open `File` so the permission check and the subsequent read share the same inode. A
/// stat-then-read pair on the path could be swapped between syscalls.
///
/// No-op on non-Unix: Windows uses ACLs and we don't try to audit them.
fn require_private_key_permissions_on_fd(
    server_name: &str,
    path: &str,
    file: &std::fs::File,
) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let metadata = file.metadata().map_err(|error| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("failed to stat signing key '{}': {}", path, error),
        })?;
        let mode = metadata.permissions().mode() & 0o777;
        if mode & 0o077 != 0 {
            return Err(MekaError::McpAuth {
                server_name: server_name.to_string(),
                message: format!(
                    "signing key '{}' has permissions {:o}; must be 0600 (group/other bits must be clear)",
                    path, mode
                ),
            });
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (server_name, path, file);
    }
    Ok(())
}

fn parse_jwt_signing_algorithm(
    server_name: &str,
    algorithm: Option<&str>,
) -> Result<rmcp::transport::JwtSigningAlgorithm> {
    use rmcp::transport::JwtSigningAlgorithm;

    match algorithm.unwrap_or("RS256") {
        "RS256" => Ok(JwtSigningAlgorithm::RS256),
        "RS384" => Ok(JwtSigningAlgorithm::RS384),
        "RS512" => Ok(JwtSigningAlgorithm::RS512),
        "ES256" => Ok(JwtSigningAlgorithm::ES256),
        "ES384" => Ok(JwtSigningAlgorithm::ES384),
        other => Err(MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: format!(
                "unsupported signing algorithm '{}': expected RS256, RS384, RS512, ES256, or ES384",
                other
            ),
        }),
    }
}

async fn authenticate_oauth_authorization_code(
    server_name: &str,
    url: &str,
    client_id: Option<&str>,
    client_secret: Option<&str>,
    scopes: Option<&[String]>,
    redirect_port: Option<u16>,
    token_store: Option<&TokenStore>,
) -> Result<AuthorizationManager> {
    // Bind the callback listener up-front so we can support a random ephemeral port (`redirect_port
    // = None` → bind 0) and learn the actual port before constructing `redirect_uri`. This avoids
    // the "port 8400 already in use" failure mode and lets multiple concurrent meka sessions
    // coexist.
    let bind_port = redirect_port.unwrap_or(0);
    let callback_listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", bind_port))
        .await
        .map_err(|error| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: format!(
                "failed to bind callback server on port {}: {}",
                bind_port, error
            ),
        })?;
    let actual_port = callback_listener
        .local_addr()
        .map_err(|error| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("callback listener local_addr failed: {}", error),
        })?
        .port();
    let redirect_uri = format!("http://127.0.0.1:{}/callback", actual_port);
    let scope_strings: Vec<String> = scopes.map(|s| s.to_vec()).unwrap_or_default();

    let mut manager = AuthorizationManager::new(url)
        .await
        .map_err(|error| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("failed to initialize OAuth manager: {}", error),
        })?;

    // Set up persistent credential storage if available. The cell is held here as well as inside
    // the adapter, so the interactive flow below can clear it; see the reset after
    // `initialize_from_store`.
    let last_read = std::sync::Arc::new(std::sync::Mutex::new(LastRead::Nothing));
    if let Some(store) = token_store {
        manager.set_credential_store(SqliteCredentialStore {
            token_store: store.clone(),
            server_name: server_name.to_string(),
            last_read: std::sync::Arc::clone(&last_read),
        });
    }

    // Try loading existing credentials from the store
    let has_stored_credentials =
        manager
            .initialize_from_store()
            .await
            .map_err(|error| MekaError::McpAuth {
                server_name: server_name.to_string(),
                message: format!("failed to load stored credentials: {}", error),
            })?;

    if has_stored_credentials {
        tracing::info!(
            "loaded stored OAuth credentials for MCP server '{}'",
            server_name
        );
        return Ok(manager);
    }

    // No stored credentials; run the interactive browser flow.
    //
    // Cleared first, and this is load-bearing. `initialize_from_store` can itself write -- on an
    // issuer change it loads, saves cleared tokens, and returns `false` -- and if that write loses
    // its swap the adapter is left `Superseded`, which refuses every later write. The interactive
    // flow performs no further `load`, so the credential the user is about to authorise in a
    // browser would be silently dropped and the next launch would ask again. Nothing this flow
    // mints is derived from a stored value, so there is nothing here for a swap to protect.
    forget_what_was_read(&last_read);
    tracing::info!(
        "starting OAuth authorization flow for MCP server '{}'",
        server_name
    );

    // Wrap in OAuthState to use its start_authorization flow which handles metadata discovery,
    // dynamic client registration, and PKCE setup
    let mut oauth_state = OAuthState::Unauthorized(manager);

    // If we have a pre-configured client_id, configure it before starting
    if let Some(id) = client_id {
        // We need to discover metadata first, then configure the client
        if let OAuthState::Unauthorized(ref mut manager) = oauth_state {
            let resolution =
                manager
                    .resolve_metadata()
                    .await
                    .map_err(|error| MekaError::McpAuth {
                        server_name: server_name.to_string(),
                        message: format!("OAuth metadata discovery failed: {}", error),
                    })?;
            manager.set_metadata(resolution.metadata);

            let mut oauth_client_config =
                rmcp::transport::auth::OAuthClientConfig::new(id.to_string(), redirect_uri.clone());
            if let Some(secret) = client_secret {
                oauth_client_config = oauth_client_config.with_client_secret(secret.to_string());
            }
            if !scope_strings.is_empty() {
                oauth_client_config = oauth_client_config.with_scopes(scope_strings.clone());
            }
            manager
                .configure_client(oauth_client_config)
                .map_err(|error| MekaError::McpAuth {
                    server_name: server_name.to_string(),
                    message: format!("failed to configure OAuth client: {}", error),
                })?;

            let scope_refs: Vec<&str> = scope_strings.iter().map(|s| s.as_str()).collect();
            let auth_url = manager
                .get_authorization_url(&scope_refs)
                .await
                .map_err(|error| MekaError::McpAuth {
                    server_name: server_name.to_string(),
                    message: format!("failed to get authorization URL: {}", error),
                })?;

            let session = rmcp::transport::AuthorizationSession::for_scope_upgrade(
                std::mem::replace(
                    manager,
                    AuthorizationManager::new("http://localhost")
                        .await
                        .map_err(|error| MekaError::McpAuth {
                            server_name: server_name.to_string(),
                            message: format!("internal error: {}", error),
                        })?,
                ),
                auth_url,
                &redirect_uri,
            );
            oauth_state = OAuthState::Session(session);
        }
    } else {
        // No client_id configured; use dynamic registration via start_authorization
        let request = rmcp::transport::auth::AuthorizationRequest::new(redirect_uri.clone())
            .with_scopes(scope_strings.clone())
            .with_client_name("meka");
        oauth_state
            .start_authorization(request)
            .await
            .map_err(|error| MekaError::McpAuth {
                server_name: server_name.to_string(),
                message: format!("failed to start OAuth authorization: {}", error),
            })?;
    }

    let auth_url =
        oauth_state
            .get_authorization_url()
            .await
            .map_err(|error| MekaError::McpAuth {
                server_name: server_name.to_string(),
                message: format!("failed to get authorization URL: {}", error),
            })?;

    // Print the URL exactly once and try to open the browser silently. Browser-launch failures are
    // expected on headless hosts (SSH, CI, containers), so they stay at `debug`. The user has the
    // URL and can copy it either way.
    crate::render::write_stderr_line(format!(
        "open this URL in your browser to authorize:\n\n{auth_url}\n"
    ));
    if let Err(error) = open::that(&auth_url) {
        tracing::debug!("open::that failed to launch browser: {}", error);
    }

    // Wait for the authorization code on our pre-bound listener.
    let (code, state) = await_oauth_callback(callback_listener)
        .await
        .map_err(|error| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("OAuth callback failed: {}", error),
        })?;

    // Exchange the authorization code for tokens
    oauth_state
        .handle_callback(&code, &state)
        .await
        .map_err(|error| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("OAuth token exchange failed: {}", error),
        })?;

    oauth_state
        .into_authorization_manager()
        .ok_or_else(|| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: "unexpected OAuth state after authorization".to_string(),
        })
}

/// Max bytes we're willing to read from a single HTTP callback request before giving up. Large
/// enough to handle big `Cookie:` headers (which can exceed 4 KiB), small enough to cap a
/// resource-exhaustion attempt.
const CALLBACK_READ_CAP: usize = 64 * 1024;
/// End-of-headers marker for HTTP/1.x.
const CRLF_CRLF: &[u8] = b"\r\n\r\n";

/// Overall wall-clock budget for the OAuth callback wait, shared by both the TCP accept path and
/// the paste-URL fallback.
const OAUTH_CALLBACK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Wait for the authorization code.
///
/// The common case is that the OAuth provider redirects the user's browser to our localhost
/// listener and we pick the code out of the HTTP request. But when meka runs on a different host
/// than the browser (SSH sessions, containers, remote Codespaces), the browser can't reach back,
/// so we race the TCP accept against a stdin prompt that lets the user paste the full callback URL
/// (it's visible in the browser's address bar even when the connection is refused). Paste mode is
/// only offered when stdin is a TTY.
async fn await_oauth_callback(
    listener: tokio::net::TcpListener,
) -> std::result::Result<(String, String), String> {
    let deadline = tokio::time::Instant::now() + OAUTH_CALLBACK_TIMEOUT;
    let paste_enabled = std::io::IsTerminal::is_terminal(&std::io::stdin());

    if paste_enabled {
        crate::render::write_stderr_line(format!(
            "waiting up to {}s for the callback, or paste the callback URL here and press Enter:",
            OAUTH_CALLBACK_TIMEOUT.as_secs()
        ));
    } else {
        crate::render::write_stderr_line(format!(
            "waiting up to {}s for the callback.",
            OAUTH_CALLBACK_TIMEOUT.as_secs()
        ));
        return accept_http_callback(listener, deadline).await;
    }

    tokio::select! {
        result = accept_http_callback(listener, deadline) => result,
        result = read_pasted_callback(deadline) => result,
    }
}

/// Accept one HTTP request on the bound listener, validate it's the OAuth callback, extract `code`
/// and `state`, and send back a success page. Loops past non-callback requests (favicons,
/// preflights) until the shared deadline elapses.
async fn accept_http_callback(
    listener: tokio::net::TcpListener,
    overall_deadline: tokio::time::Instant,
) -> std::result::Result<(String, String), String> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    loop {
        let remaining = overall_deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return Err(format!(
                "authorization timed out after {}s",
                OAUTH_CALLBACK_TIMEOUT.as_secs()
            ));
        }

        let accept_result = tokio::time::timeout(remaining, listener.accept()).await;
        let (mut stream, _addr) = match accept_result {
            Err(_) => {
                return Err(format!(
                    "authorization timed out after {}s",
                    OAUTH_CALLBACK_TIMEOUT.as_secs()
                ));
            }
            Ok(Err(error)) => return Err(format!("failed to accept connection: {}", error)),
            Ok(Ok(pair)) => pair,
        };

        // Read until we've seen CRLF-CRLF (end of request headers) or hit the byte cap. Browsers
        // sometimes send favicon / preflight requests to the callback origin; if the path isn't
        // `/callback?...`, respond with a minimal 404 so the browser stops retrying and keep
        // waiting.
        let mut buffer = Vec::with_capacity(4096);
        let mut temp = [0u8; 4096];
        let headers_complete = loop {
            if buffer.windows(CRLF_CRLF.len()).any(|w| w == CRLF_CRLF) {
                break true;
            }
            if buffer.len() >= CALLBACK_READ_CAP {
                break false;
            }
            let read_remaining =
                overall_deadline.saturating_duration_since(tokio::time::Instant::now());
            if read_remaining.is_zero() {
                return Err(format!(
                    "authorization timed out after {}s",
                    OAUTH_CALLBACK_TIMEOUT.as_secs()
                ));
            }
            match tokio::time::timeout(read_remaining, stream.read(&mut temp)).await {
                Err(_) => {
                    return Err(format!(
                        "authorization timed out after {}s",
                        OAUTH_CALLBACK_TIMEOUT.as_secs()
                    ));
                }
                Ok(Err(error)) => return Err(format!("failed to read request: {}", error)),
                Ok(Ok(0)) => break buffer.windows(CRLF_CRLF.len()).any(|w| w == CRLF_CRLF),
                Ok(Ok(n)) => buffer.extend_from_slice(&temp[..n]),
            }
        };

        if !headers_complete {
            tracing::debug!(
                "OAuth callback: dropped request with incomplete/oversized headers \
                 ({} bytes)",
                buffer.len()
            );
            let _ = stream
                .write_all(b"HTTP/1.1 431 Request Header Fields Too Large\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .await;
            continue;
        }

        let request = String::from_utf8_lossy(&buffer);
        match parse_callback_query(&request) {
            Ok((code, state)) => {
                let response_body = "<!DOCTYPE html><html><body>\
                    <h1>Authorization successful</h1>\
                    <p>You can close this tab and return to meka.</p>\
                    </body></html>";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                if let Err(error) = stream.write_all(response.as_bytes()).await {
                    tracing::debug!("failed to send callback response: {}", error);
                }
                return Ok((code, state));
            }
            Err(CallbackParseError::NotCallbackPath) => {
                // Almost certainly a browser preflight or favicon request. Respond 404 and keep
                // waiting for the real callback.
                tracing::debug!("OAuth callback: ignored non-callback request on callback port");
                let _ = stream
                    .write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    )
                    .await;
                continue;
            }
            Err(CallbackParseError::Malformed(message)) => {
                let _ = stream
                    .write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                    .await;
                return Err(message);
            }
        }
    }
}

/// Paste-URL fallback for the OAuth callback: prompt on stderr, read a line from stdin, extract
/// `code` + `state` from the pasted URL. Used when the browser can't reach back to our bound
/// listener (e.g. meka is on an SSH host and the browser is on the user's laptop).
async fn read_pasted_callback(
    deadline: tokio::time::Instant,
) -> std::result::Result<(String, String), String> {
    use tokio::io::{AsyncBufReadExt, BufReader};

    let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
    if remaining.is_zero() {
        return Err(format!(
            "authorization timed out after {}s",
            OAUTH_CALLBACK_TIMEOUT.as_secs()
        ));
    }
    let mut reader = BufReader::new(tokio::io::stdin());
    let mut line = String::new();
    match tokio::time::timeout(remaining, reader.read_line(&mut line)).await {
        Err(_) => Err(format!(
            "authorization timed out after {}s",
            OAUTH_CALLBACK_TIMEOUT.as_secs()
        )),
        Ok(Err(error)) => Err(format!("stdin read failed: {}", error)),
        // `read_line` returning 0 means EOF: stdin was closed before the user pasted anything.
        // Don't treat this as a fatal error; let the TCP branch of the `select!` continue waiting.
        Ok(Ok(0)) => std::future::pending().await,
        Ok(Ok(_)) => parse_pasted_callback(&line),
    }
}

/// Extract `(code, state)` from a pasted callback URL. Accepts either the full URL or just the
/// query string, percent-decodes the values, and surfaces the `error=…` parameter (sanitised) when
/// the authorization server declines.
fn parse_pasted_callback(input: &str) -> std::result::Result<(String, String), String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err("no callback URL pasted".to_string());
    }
    // Drop any URL fragment, then narrow to whatever sits after `?`.
    let before_hash = trimmed.split('#').next().unwrap_or(trimmed);
    let query = match before_hash.find('?') {
        Some(idx) => &before_hash[idx + 1..],
        None => before_hash,
    };
    let mut code = None;
    let mut state = None;
    let mut error_param: Option<String> = None;
    for pair in query.split('&') {
        let Some((key, value)) = pair.split_once('=') else {
            continue;
        };
        // Strict UTF-8: silently mangling a security-sensitive parameter (e.g. swapping invalid
        // bytes for U+FFFD) could let a tampered `state` value match the expected one despite
        // differing bytes.
        let decoded = percent_encoding::percent_decode_str(value)
            .decode_utf8()
            .map_err(|error| {
                format!(
                    "OAuth callback parameter '{}' is not valid UTF-8: {}",
                    key, error
                )
            })?
            .into_owned();
        match key {
            "code" => code = Some(decoded),
            "state" => state = Some(decoded),
            "error" => error_param = Some(decoded),
            _ => {}
        }
    }
    if let Some(error) = error_param {
        return Err(format!(
            "authorization server returned error: {}",
            crate::mcp::sanitize::sanitize_text(&error)
        ));
    }
    let code = code.ok_or_else(|| "missing 'code' parameter in pasted URL".to_string())?;
    let state = state.ok_or_else(|| "missing 'state' parameter in pasted URL".to_string())?;
    Ok((code, state))
}

#[derive(Debug)]
enum CallbackParseError {
    /// Request wasn't to /callback at all (e.g. /favicon.ico, /).
    NotCallbackPath,
    /// Request targets /callback but failed to parse (missing code/state, etc).
    Malformed(String),
}

fn parse_callback_query(
    request: &str,
) -> std::result::Result<(String, String), CallbackParseError> {
    let first_line = request
        .lines()
        .next()
        .ok_or_else(|| CallbackParseError::Malformed("empty HTTP request".into()))?;

    let path = first_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| CallbackParseError::Malformed("malformed HTTP request line".into()))?;

    // Compare path component only, case-insensitive, anchored to /callback. `/` or `/favicon.ico`
    // fall through to `NotCallbackPath`.
    let (path_component, query_string) = match path.split_once('?') {
        Some((p, q)) => (p, q),
        None => (path, ""),
    };
    if !path_component.eq_ignore_ascii_case("/callback") {
        return Err(CallbackParseError::NotCallbackPath);
    }
    if query_string.is_empty() {
        return Err(CallbackParseError::Malformed(
            "no query parameters in callback URL".into(),
        ));
    }

    let mut code = None;
    let mut state = None;
    let mut error_param: Option<String> = None;

    // `form_urlencoded`, not a hand-rolled `split('&')` + `percent_decode_str`. A redirect query is
    // `application/x-www-form-urlencoded`, where `+` means a space; `percent_decode_str` leaves it
    // as a literal `+`, so a server that form-encodes a value containing a space hands back a code
    // that is then re-encoded as `%2B` and rejected at the token exchange with nothing to point at.
    for (key, value) in url::form_urlencoded::parse(query_string.as_bytes()) {
        // Stricter than the `Cow<str>` this yields for invalid UTF-8: reject rather than accept
        // a value with U+FFFD substituted into it. Same rationale as `parse_pasted_callback`.
        if value.contains('\u{FFFD}') {
            return Err(CallbackParseError::Malformed(format!(
                "OAuth callback parameter '{key}' is not valid UTF-8"
            )));
        }
        let decoded = value.into_owned();
        match key.as_ref() {
            "code" => code = Some(decoded),
            "state" => state = Some(decoded),
            "error" => error_param = Some(decoded),
            _ => {}
        }
    }

    if let Some(error) = error_param {
        // Strip Cc/Cf so a hostile authorization server can't inject ANSI escapes or RTL overrides
        // through the error message.
        return Err(CallbackParseError::Malformed(format!(
            "authorization server returned error: {}",
            crate::mcp::sanitize::sanitize_text(&error)
        )));
    }

    let code = code.ok_or_else(|| {
        CallbackParseError::Malformed("missing 'code' parameter in callback".into())
    })?;
    let state = state.ok_or_else(|| {
        CallbackParseError::Malformed("missing 'state' parameter in callback".into())
    })?;

    Ok((code, state))
}

/// What this adapter knows about the row it writes to, which is what decides how it writes.
///
/// rmcp's `CredentialStore` is a `load`/`save` pair with nothing in between, so a refresh arrives
/// as a bare "store this" with no way to say what it was derived from. Tracking the read is what
/// supplies that.
#[derive(Debug, Clone)]
enum LastRead {
    /// Nothing has been read, so there is nothing to conflict with: the interactive authorisation
    /// flow, where replacing whatever is there is the point of it.
    Nothing,
    /// The JSON last handed to rmcp. A save compares against it and replaces only that.
    Json(String),
    /// A save found the row holding something else.
    ///
    /// Distinct from [`Self::Nothing`], and the distinction is the whole of it. Both mean "there
    /// is nothing to compare against", but they want opposite writes: `Nothing` wants an
    /// unconditional one, and this wants none at all. Collapsing the two -- which is what
    /// recording a lost swap as "unread" did -- meant a process reverted to blind upserts for the
    /// rest of its run after losing a single race, quietly undoing the guarantee the swap exists
    /// for. Every token rmcp mints from here descends from a credential another process has
    /// already superseded, and that is true of the next one and the one after it, not just the one
    /// that lost.
    Superseded,
}

struct SqliteCredentialStore {
    token_store: TokenStore,
    server_name: String,
    /// What this adapter last saw in the row. See [`LastRead`].
    last_read: std::sync::Arc<std::sync::Mutex<LastRead>>,
}

/// Put a credential cell back to "nothing has been read", for a flow that is about to mint a
/// credential from scratch rather than derive one from what is stored.
///
/// A poisoned mutex is recovered from for the same reason [`SqliteCredentialStore::remember`]
/// recovers: the cell holds one value that every writer replaces whole.
fn forget_what_was_read(cell: &std::sync::Arc<std::sync::Mutex<LastRead>>) {
    match cell.lock() {
        Ok(mut cell) => *cell = LastRead::Nothing,
        Err(poisoned) => *poisoned.into_inner() = LastRead::Nothing,
    }
}

impl SqliteCredentialStore {
    /// Record what the row holds, after a read, a write that did not land, or a clear.
    ///
    /// A poisoned mutex is recovered from rather than propagated: the cell holds one value that
    /// every writer replaces whole, and refusing to track a read because an unrelated thread
    /// panicked would only make the next save unconditional.
    fn remember(&self, last: LastRead) {
        match self.last_read.lock() {
            Ok(mut cell) => *cell = last,
            Err(poisoned) => *poisoned.into_inner() = last,
        }
    }

    fn last_read(&self) -> LastRead {
        match self.last_read.lock() {
            Ok(cell) => cell.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }
}

#[async_trait]
impl CredentialStore for SqliteCredentialStore {
    async fn load(&self) -> std::result::Result<Option<StoredCredentials>, AuthError> {
        match self
            .token_store
            .load_mcp_credentials(&self.server_name, crate::session::McpCredentialKind::OAuth)
            .await
        {
            Ok(Some(json)) => {
                let credentials: StoredCredentials =
                    serde_json::from_str(&json).map_err(|error| {
                        AuthError::InternalError(format!(
                            "failed to deserialize stored credentials: {}",
                            error
                        ))
                    })?;
                self.remember(LastRead::Json(json));
                Ok(Some(credentials))
            }
            // Nothing stored means nothing to conflict with, and that has to be *recorded*: a
            // `Superseded` left over from a swap this adapter lost would otherwise outlive the row
            // it was about, and refuse to write to a profile that is now empty.
            Ok(None) => {
                self.remember(LastRead::Nothing);
                Ok(None)
            }
            Err(error) => Err(AuthError::InternalError(format!(
                "failed to load credentials from database: {}",
                error
            ))),
        }
    }

    async fn save(&self, credentials: StoredCredentials) -> std::result::Result<(), AuthError> {
        let json = serde_json::to_string(&credentials).map_err(|error| {
            AuthError::InternalError(format!("failed to serialize credentials: {}", error))
        })?;

        let stored = match self.last_read() {
            LastRead::Json(expected) => self
                .token_store
                .replace_mcp_credentials(&self.server_name, &expected, &json)
                .await
                .map_err(|error| {
                    AuthError::InternalError(format!(
                        "failed to save credentials to database: {}",
                        error
                    ))
                })?,
            LastRead::Nothing => {
                self.token_store
                    .save_mcp_credentials(
                        &self.server_name,
                        crate::session::McpCredentialKind::OAuth,
                        &json,
                    )
                    .await
                    .map_err(|error| {
                        AuthError::InternalError(format!(
                            "failed to save credentials to database: {}",
                            error
                        ))
                    })?;
                true
            }
            // Nothing this adapter can mint is writable any more: it all descends from a
            // credential the row has already moved past. Saying so once was enough; repeating it
            // per refresh would turn one fact into a line an hour.
            LastRead::Superseded => {
                tracing::debug!(
                    "not persisting a refreshed token for MCP server '{}': it descends from a \
                     credential the store has moved past",
                    self.server_name
                );
                return Ok(());
            }
        };

        if stored {
            self.remember(LastRead::Json(json));
        } else {
            // Reported rather than raised. rmcp holds the token it just minted whatever this says,
            // and failing the save would abort a connection that is otherwise working; what matters
            // is that the *stored* credential stays the newest one, which is what the next process
            // to start will load.
            tracing::info!(
                "MCP server '{}' was re-authenticated elsewhere while this token was being \
                 refreshed; the stored credential was left as it is, and this process will not \
                 write to it again",
                self.server_name
            );
            self.remember(LastRead::Superseded);
        }
        Ok(())
    }

    async fn clear(&self) -> std::result::Result<(), AuthError> {
        // Back to `Nothing` rather than to `Superseded`: a clear is this process deciding the row
        // should hold nothing, so a fresh authorisation after it is entitled to write.
        self.remember(LastRead::Nothing);
        // Only the bundle this adapter owns. rmcp calls `clear` when the authorization server's
        // issuer changes, meaning the tokens it holds are bound to an issuer that is no longer the
        // right one -- a statement about the tokens and nothing else. A confidential client's
        // `client_secret` is not bound to an issuer, was typed by the user, and after this change
        // meka is its only holder, so clearing every kind here would send them back to the
        // provider to reissue one that was never invalid.
        self.token_store
            .clear_mcp_credentials_of_kind(
                &self.server_name,
                crate::session::McpCredentialKind::OAuth,
            )
            .await
            .map_err(|error| {
                AuthError::InternalError(format!(
                    "failed to clear credentials from database: {}",
                    error
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The origin a stored bundle names, and what happens when it names none.
    ///
    /// rmcp has recorded `server_url` in the bundle since before meka stored it, so the fact was
    /// always on disk; what was missing was any reader. `issuer` is the older key and still the
    /// fallback, and the two must be read in the same order here as in `revoke_stored_token` --
    /// otherwise `meka mcp get` reports one host while the revoke posts to another.
    #[tokio::test]
    async fn a_stored_bundle_reports_the_host_it_was_issued_for() {
        let manager = crate::session::SessionManager::open(
            Some(std::path::Path::new(":memory:")),
            &Default::default(),
        )
        .await
        .expect("memory store");
        let token_store = manager.token_store();

        assert_eq!(
            stored_credential_origin(&token_store, "docs").await,
            None,
            "no bundle is not a mismatch; it is nothing to report"
        );

        token_store
            .save_mcp_credentials(
                "docs",
                crate::session::McpCredentialKind::OAuth,
                r#"{"server_url":"https://a.example/mcp","tokens":{"access_token":"x"}}"#,
            )
            .await
            .expect("seed");
        assert_eq!(
            stored_credential_origin(&token_store, "docs")
                .await
                .as_deref(),
            Some("https://a.example/mcp")
        );

        token_store
            .save_mcp_credentials(
                "older",
                crate::session::McpCredentialKind::OAuth,
                r#"{"issuer":"https://b.example","tokens":{"access_token":"x"}}"#,
            )
            .await
            .expect("seed");
        assert_eq!(
            stored_credential_origin(&token_store, "older")
                .await
                .as_deref(),
            Some("https://b.example"),
            "the older key is still read, or every pre-existing bundle reports nothing"
        );

        token_store
            .save_mcp_credentials(
                "broken",
                crate::session::McpCredentialKind::OAuth,
                "not json at all",
            )
            .await
            .expect("seed");
        assert_eq!(
            stored_credential_origin(&token_store, "broken").await,
            None,
            "this decorates a listing and must never fail one"
        );
    }

    /// Two URLs are the same server when the bearer token would reach the same host.
    #[test]
    fn same_origin_compares_who_receives_the_token() {
        assert!(
            same_origin("https://a.example/mcp", "https://a.example/"),
            "the issuer sits at the root and the endpoint under a path; comparing paths would \
             report a mismatch on every correctly configured server"
        );
        assert!(same_origin(
            "https://a.example",
            "https://a.example:443/mcp"
        ));
        assert!(
            !same_origin("https://a.example/mcp", "https://b.example/mcp"),
            "a different host is the whole point of reporting this"
        );
        assert!(
            !same_origin("https://a.example/mcp", "http://a.example/mcp"),
            "and so is a downgraded scheme"
        );
        assert!(
            !same_origin("https://a.example", "https://a.example:8443"),
            "a different port is a different server"
        );
        assert!(
            same_origin("not a url", "not a url"),
            "a comparison meka cannot make says `the same` rather than crying wolf"
        );
    }

    /// Losing one compare-and-swap must not turn every later write into an unconditional one.
    ///
    /// The state after a lost swap must not be recorded as "nothing was read", which is also the
    /// state the interactive authorisation flow is in, and that state takes the blind-upsert arm.
    /// So a process that lost a single race went back to overwriting the row for the rest of its
    /// run, with every token it wrote descended from a credential another process had already
    /// superseded. The invariant [`TokenStore::replace_mcp_credentials`] states -- that the stored
    /// credential never moves backwards -- held only until the first race.
    #[tokio::test]
    async fn a_lost_swap_does_not_return_the_adapter_to_blind_writes() {
        let manager = crate::session::SessionManager::open(
            Some(std::path::Path::new(":memory:")),
            &Default::default(),
        )
        .await
        .expect("memory store");
        let token_store = manager.token_store();
        // Built by deserializing rather than by a struct literal: `StoredCredentials` is
        // `#[non_exhaustive]`, so rmcp reserves the right to add a field and only its own crate may
        // name them all.
        let credentials = |client_id: &str| -> StoredCredentials {
            serde_json::from_value(serde_json::json!({ "client_id": client_id }))
                .expect("the shape rmcp stores")
        };
        let json =
            |client_id: &str| serde_json::to_string(&credentials(client_id)).expect("serialize");
        token_store
            .save_mcp_credentials(
                "docs",
                crate::session::McpCredentialKind::OAuth,
                &json("original"),
            )
            .await
            .expect("seed");

        let adapter = SqliteCredentialStore {
            token_store: token_store.clone(),
            server_name: "docs".to_string(),
            last_read: std::sync::Arc::new(std::sync::Mutex::new(LastRead::Nothing)),
        };
        adapter.load().await.expect("load").expect("seeded");

        // Another process re-authenticates while this adapter holds what it read.
        token_store
            .save_mcp_credentials(
                "docs",
                crate::session::McpCredentialKind::OAuth,
                &json("theirs"),
            )
            .await
            .expect("the other process writes");

        adapter
            .save(credentials("ours"))
            .await
            .expect("a lost swap is reported, not raised");
        assert_eq!(
            token_store
                .load_mcp_credentials("docs", crate::session::McpCredentialKind::OAuth)
                .await
                .expect("load")
                .as_deref(),
            Some(json("theirs").as_str()),
            "the swap must lose"
        );

        // The next refresh in the same run. Its token descends from the same superseded
        // credential, so it is no more writable than the one that lost.
        adapter.save(credentials("ours-again")).await.expect("save");
        assert_eq!(
            token_store
                .load_mcp_credentials("docs", crate::session::McpCredentialKind::OAuth)
                .await
                .expect("load")
                .as_deref(),
            Some(json("theirs").as_str()),
            "and every write after it must lose too, rather than reverting to a blind upsert"
        );
    }

    /// rmcp calls `clear` when the authorization server's issuer changes. That is a statement
    /// about the tokens it holds, not about the client secret they were obtained with.
    ///
    /// The secret was typed by the user and, since it stopped being a config key, meka is its only
    /// holder. Clearing every kind here would make an issuer migration at the provider silently
    /// cost the user a credential that was never invalid, discovered only when the next login
    /// fails for want of it.
    #[tokio::test]
    async fn the_adapter_clears_only_the_bundle_it_owns() {
        use crate::session::McpCredentialKind;

        let manager = crate::session::SessionManager::open(
            Some(std::path::Path::new(":memory:")),
            &Default::default(),
        )
        .await
        .expect("memory store");
        let token_store = manager.token_store();
        for (kind, secret) in [
            (McpCredentialKind::ClientSecret, "cs-not-a-real-secret"),
            (McpCredentialKind::Bearer, "bearer-not-a-real-token"),
            (McpCredentialKind::OAuth, r#"{"client_id":"x"}"#),
        ] {
            token_store
                .save_mcp_credentials("docs", kind, secret)
                .await
                .expect("seed");
        }

        let adapter = SqliteCredentialStore {
            token_store: token_store.clone(),
            server_name: "docs".to_string(),
            last_read: std::sync::Arc::new(std::sync::Mutex::new(LastRead::Nothing)),
        };
        adapter.clear().await.expect("clear");

        assert!(
            token_store
                .load_mcp_credentials("docs", McpCredentialKind::OAuth)
                .await
                .expect("load")
                .is_none(),
            "the bundle bound to the old issuer must go"
        );
        assert_eq!(
            token_store
                .load_mcp_credentials("docs", McpCredentialKind::ClientSecret)
                .await
                .expect("load")
                .as_deref(),
            Some("cs-not-a-real-secret"),
            "the client secret is not bound to an issuer and must survive"
        );
        assert_eq!(
            token_store
                .load_mcp_credentials("docs", McpCredentialKind::Bearer)
                .await
                .expect("load")
                .as_deref(),
            Some("bearer-not-a-real-token"),
            "nor is a bearer this adapter never owned"
        );
    }

    /// The other half: a save that should land, does.
    ///
    /// Both of the adapter's arms are exercised, because a `save` that quietly writes nothing is
    /// invisible for the life of the process, which keeps using the token it holds in memory. The
    /// user finds out at the next start, when the credential loaded is the one from before the
    /// refresh, expired.
    #[tokio::test]
    async fn a_save_that_should_land_reaches_the_store() {
        let manager = crate::session::SessionManager::open(
            Some(std::path::Path::new(":memory:")),
            &Default::default(),
        )
        .await
        .expect("memory store");
        let token_store = manager.token_store();
        let credentials = |client_id: &str| -> StoredCredentials {
            serde_json::from_value(serde_json::json!({ "client_id": client_id }))
                .expect("the shape rmcp stores")
        };
        let json =
            |client_id: &str| serde_json::to_string(&credentials(client_id)).expect("serialize");
        let stored = async || {
            token_store
                .load_mcp_credentials("docs", crate::session::McpCredentialKind::OAuth)
                .await
                .expect("load")
        };

        let adapter = SqliteCredentialStore {
            token_store: token_store.clone(),
            server_name: "docs".to_string(),
            last_read: std::sync::Arc::new(std::sync::Mutex::new(LastRead::Nothing)),
        };

        // The blind-upsert arm: the interactive flow depositing a first bundle.
        adapter.save(credentials("first")).await.expect("save");
        assert_eq!(
            stored().await.as_deref(),
            Some(json("first").as_str()),
            "the first bundle must reach the store"
        );

        // The compare-and-swap arm: a refresh of what this adapter last read.
        adapter.load().await.expect("load").expect("just written");
        adapter.save(credentials("refreshed")).await.expect("save");
        assert_eq!(
            stored().await.as_deref(),
            Some(json("refreshed").as_str()),
            "an uncontended refresh must land"
        );
    }

    #[test]
    fn test_parse_callback_query_valid() {
        let request = "GET /callback?code=abc123&state=xyz789 HTTP/1.1\r\nHost: localhost\r\n";
        let (code, state) = parse_callback_query(request).expect("should parse");
        assert_eq!(code, "abc123");
        assert_eq!(state, "xyz789");
    }

    #[test]
    fn test_parse_callback_query_reversed_order() {
        let request = "GET /callback?state=xyz789&code=abc123 HTTP/1.1\r\n";
        let (code, state) = parse_callback_query(request).expect("should parse");
        assert_eq!(code, "abc123");
        assert_eq!(state, "xyz789");
    }

    #[test]
    fn test_parse_callback_query_missing_code() {
        let request = "GET /callback?state=xyz789 HTTP/1.1\r\n";
        let err = parse_callback_query(request).expect_err("should fail");
        match err {
            CallbackParseError::Malformed(m) => assert!(m.contains("code")),
            other => panic!("expected Malformed, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_callback_query_missing_state() {
        let request = "GET /callback?code=abc123 HTTP/1.1\r\n";
        let err = parse_callback_query(request).expect_err("should fail");
        match err {
            CallbackParseError::Malformed(m) => assert!(m.contains("state")),
            other => panic!("expected Malformed, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_callback_query_no_query_string() {
        let request = "GET /callback HTTP/1.1\r\n";
        let err = parse_callback_query(request).expect_err("should fail");
        match err {
            CallbackParseError::Malformed(_) => {}
            other => panic!("expected Malformed, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_callback_query_ignores_favicon() {
        let request = "GET /favicon.ico HTTP/1.1\r\n";
        let err = parse_callback_query(request).expect_err("should fail");
        assert!(matches!(err, CallbackParseError::NotCallbackPath));
    }

    #[test]
    fn test_parse_callback_query_ignores_root() {
        let request = "GET / HTTP/1.1\r\n";
        let err = parse_callback_query(request).expect_err("should fail");
        assert!(matches!(err, CallbackParseError::NotCallbackPath));
    }

    #[test]
    fn test_parse_callback_query_url_decodes_state() {
        let request = "GET /callback?code=abc&state=xyz%3D%3D HTTP/1.1\r\n";
        let (code, state) = parse_callback_query(request).expect("should parse");
        assert_eq!(code, "abc");
        assert_eq!(state, "xyz==");
    }

    #[test]
    fn test_parse_callback_query_rejects_invalid_utf8_state() {
        // %FF and %FE are invalid as UTF-8; lossy decoding would silently turn them into U+FFFD,
        // which can let a tampered `state` parameter match a stored one despite differing bytes.
        // Strict mode rejects.
        let request = "GET /callback?code=abc&state=%FF%FE HTTP/1.1\r\n";
        let err = parse_callback_query(request).expect_err("non-utf8 state must be rejected");
        match err {
            CallbackParseError::Malformed(m) => {
                assert!(
                    m.contains("not valid UTF-8"),
                    "unexpected error message: {}",
                    m
                );
            }
            other => panic!("expected Malformed, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_callback_query_surfaces_oauth_error() {
        let request = "GET /callback?error=access_denied HTTP/1.1\r\n";
        let err = parse_callback_query(request).expect_err("should fail");
        match err {
            CallbackParseError::Malformed(m) => {
                assert!(m.contains("access_denied"));
            }
            other => panic!("expected Malformed with error, got {:?}", other),
        }
    }

    #[test]
    fn extract_revocation_endpoint_returns_string() {
        let body = br#"{"revocation_endpoint":"https://auth.example.com/revoke","other":"x"}"#;
        assert_eq!(
            extract_revocation_endpoint(body, 1024),
            Some("https://auth.example.com/revoke".to_string())
        );
    }

    #[test]
    fn extract_revocation_endpoint_rejects_oversize_body() {
        let body = br#"{"revocation_endpoint":"https://auth.example.com/revoke"}"#;
        assert!(extract_revocation_endpoint(body, 8).is_none());
    }

    #[test]
    fn extract_revocation_endpoint_returns_none_without_field() {
        let body = br#"{"issuer":"https://auth.example.com"}"#;
        assert!(extract_revocation_endpoint(body, 1024).is_none());
    }

    #[test]
    fn extract_revocation_endpoint_rejects_malformed_json() {
        assert!(extract_revocation_endpoint(b"not json", 1024).is_none());
    }

    #[test]
    fn validate_revocation_endpoint_accepts_same_origin() {
        validate_revocation_endpoint_origin(
            "https://auth.example.com",
            "https://auth.example.com/revoke",
        )
        .expect("same host should pass");
    }

    #[test]
    fn validate_revocation_endpoint_accepts_same_origin_with_path() {
        validate_revocation_endpoint_origin(
            "https://auth.example.com/realms/r",
            "https://auth.example.com/realms/r/protocol/openid-connect/revoke",
        )
        .expect("subpath on same host should pass");
    }

    #[test]
    fn validate_revocation_endpoint_rejects_different_host() {
        let err = validate_revocation_endpoint_origin(
            "https://auth.example.com",
            "https://attacker.com/steal",
        )
        .expect_err("cross-host should be rejected");
        assert!(err.contains("different origin"));
    }

    #[test]
    fn validate_revocation_endpoint_rejects_http_to_https_downgrade() {
        let err = validate_revocation_endpoint_origin(
            "https://auth.example.com",
            "http://auth.example.com/revoke",
        )
        .expect_err("scheme change should be rejected");
        assert!(err.contains("different origin"));
    }

    #[test]
    fn validate_revocation_endpoint_rejects_different_port() {
        let err = validate_revocation_endpoint_origin(
            "https://auth.example.com",
            "https://auth.example.com:8443/revoke",
        )
        .expect_err("port change should be rejected");
        assert!(err.contains("different origin"));
    }

    #[cfg(unix)]
    #[test]
    fn require_private_key_permissions_rejects_world_readable() {
        use std::{io::Write, os::unix::fs::PermissionsExt};
        let dir = tempfile::tempdir().expect("tempdir");
        let key_path = dir.path().join("key.pem");
        let mut f = std::fs::File::create(&key_path).expect("create key");
        f.write_all(b"---BEGIN---").expect("write");
        drop(f);
        // 0644: readable by other users.
        std::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(0o644))
            .expect("chmod 0644");
        let file = std::fs::File::open(&key_path).expect("open key");
        let err = require_private_key_permissions_on_fd(
            "srv",
            key_path.to_str().expect("utf-8 path"),
            &file,
        )
        .expect_err("loose perms must be rejected");
        let message = format!("{}", err);
        assert!(
            message.contains("0600") || message.contains("permissions"),
            "unexpected error: {}",
            message
        );
    }

    #[cfg(unix)]
    #[test]
    fn require_private_key_permissions_accepts_0600() {
        use std::{io::Write, os::unix::fs::PermissionsExt};
        let dir = tempfile::tempdir().expect("tempdir");
        let key_path = dir.path().join("key.pem");
        let mut f = std::fs::File::create(&key_path).expect("create key");
        f.write_all(b"---BEGIN---").expect("write");
        drop(f);
        std::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(0o600))
            .expect("chmod 0600");
        let file = std::fs::File::open(&key_path).expect("open key");
        require_private_key_permissions_on_fd("srv", key_path.to_str().expect("utf-8 path"), &file)
            .expect("0600 must pass");
    }

    #[cfg(unix)]
    #[test]
    fn require_private_key_permissions_reports_missing_file() {
        // Missing file errors at `File::open` rather than inside the permissions check; assert the
        // open-side error message matches the user-facing wording in
        // `authenticate_client_credentials_jwt`.
        let err = std::fs::File::open("/nonexistent/key.pem").expect_err("missing file must error");
        let message = format!("{}", err);
        assert!(message.contains("No such file") || message.contains("not found"));
    }

    #[test]
    fn validate_revocation_endpoint_rejects_malformed_urls() {
        assert!(
            validate_revocation_endpoint_origin("not-a-url", "https://auth.example.com/revoke")
                .is_err()
        );
        assert!(
            validate_revocation_endpoint_origin("https://auth.example.com", "also-not-a-url")
                .is_err()
        );
    }

    #[test]
    fn test_parse_callback_query_sanitises_oauth_error() {
        // A malicious authorization server includes an ANSI escape and an RTL override in its
        // `error` parameter. The resulting error message must not carry those codepoints to the
        // terminal.
        let request = "GET /callback?error=bad%1B%5B2Jstuff%E2%80%AErtl HTTP/1.1\r\n";
        let err = parse_callback_query(request).expect_err("should fail");
        match err {
            CallbackParseError::Malformed(m) => {
                assert!(!m.contains('\u{001B}'), "ANSI escape leaked: {:?}", m);
                assert!(!m.contains('\u{202E}'), "RTL override leaked: {:?}", m);
                assert!(m.contains("bad"));
                assert!(m.contains("stuff"));
                assert!(m.contains("rtl"));
            }
            other => panic!("expected Malformed with error, got {:?}", other),
        }
    }

    #[test]
    fn classify_probe_open_on_2xx() {
        assert_eq!(classify_probe_response(200, None), McpAuthProbe::Open);
        assert_eq!(classify_probe_response(204, None), McpAuthProbe::Open);
        assert_eq!(
            classify_probe_response(200, Some("Bearer realm=\"x\"")),
            McpAuthProbe::Open,
            "2xx always wins over any WWW-Authenticate header"
        );
    }

    #[test]
    fn classify_probe_auth_required_with_resource_metadata() {
        // What Notion-style MCP servers actually emit.
        let header = r#"Bearer realm="mcp", resource_metadata="https://mcp.notion.com/.well-known/oauth-protected-resource""#;
        assert_eq!(
            classify_probe_response(401, Some(header)),
            McpAuthProbe::AuthRequired {
                resource_metadata: Some(
                    "https://mcp.notion.com/.well-known/oauth-protected-resource".to_string()
                ),
            }
        );
    }

    #[test]
    fn classify_probe_auth_required_without_resource_metadata() {
        assert_eq!(
            classify_probe_response(401, Some("Bearer realm=\"mcp\"")),
            McpAuthProbe::AuthRequired {
                resource_metadata: None,
            }
        );
    }

    #[test]
    fn classify_probe_auth_required_bare_bearer() {
        // Some servers emit just `Bearer` with no parameters.
        assert_eq!(
            classify_probe_response(401, Some("Bearer")),
            McpAuthProbe::AuthRequired {
                resource_metadata: None,
            }
        );
    }

    #[test]
    fn classify_probe_is_case_insensitive_on_scheme() {
        assert_eq!(
            classify_probe_response(401, Some("bearer realm=\"x\"")),
            McpAuthProbe::AuthRequired {
                resource_metadata: None,
            }
        );
    }

    #[test]
    fn classify_probe_401_without_bearer_is_unexpected() {
        // A 401 with e.g. Basic / Digest auth is not MCP-spec compliant. Surface it as Unexpected
        // rather than pretending it's OAuth.
        assert_eq!(
            classify_probe_response(401, Some("Basic realm=\"x\"")),
            McpAuthProbe::Unexpected { status: 401 }
        );
        assert_eq!(
            classify_probe_response(401, None),
            McpAuthProbe::Unexpected { status: 401 }
        );
    }

    #[test]
    fn classify_probe_403_with_bearer_is_auth_required() {
        // Some implementations return 403 for missing auth.
        assert_eq!(
            classify_probe_response(403, Some("Bearer realm=\"mcp\"")),
            McpAuthProbe::AuthRequired {
                resource_metadata: None,
            }
        );
    }

    #[test]
    fn classify_probe_other_statuses_are_unexpected() {
        assert_eq!(
            classify_probe_response(405, None),
            McpAuthProbe::Unexpected { status: 405 }
        );
        assert_eq!(
            classify_probe_response(500, None),
            McpAuthProbe::Unexpected { status: 500 }
        );
    }

    #[test]
    fn extract_bearer_param_handles_quoting_and_spacing() {
        let header = r#"Bearer realm="x",resource_metadata="https://y/z""#;
        assert_eq!(
            extract_bearer_param(header, "resource_metadata"),
            Some("https://y/z".to_string())
        );
        let unquoted = r#"Bearer realm=x, resource_metadata=https://y/z"#;
        assert_eq!(
            extract_bearer_param(unquoted, "resource_metadata"),
            Some("https://y/z".to_string())
        );
    }

    #[test]
    fn extract_bearer_param_returns_none_when_missing() {
        assert_eq!(
            extract_bearer_param("Bearer realm=\"x\"", "resource_metadata"),
            None
        );
    }

    /// A form-encoded space arrives as a space, not a literal `+`.
    ///
    /// A redirect query is `application/x-www-form-urlencoded`, where `+` is a space.
    /// `percent_decode_str` does not know that, so a hand-rolled split returned the `+` verbatim
    /// and the token exchange re-encoded it as `%2B` -- a failure the server reports opaquely and
    /// nothing local explains. `form_urlencoded::parse` is the decoder that matches the encoder.
    #[test]
    fn callback_query_decodes_form_encoding_not_just_percent_escapes() {
        let (code, state) = parse_callback_query(
            "GET /callback?code=a+b&state=c%20d HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
        .expect("a well-formed callback");
        assert_eq!(code, "a b", "`+` is a space in a form-encoded query");
        assert_eq!(state, "c d", "and `%20` still is too");
    }

    /// Invalid UTF-8 in a callback parameter is refused rather than silently substituted.
    ///
    /// `form_urlencoded::parse` yields a lossy `Cow`, so without this check a `state` carrying
    /// invalid bytes would come back with U+FFFD in place of them and be compared against the
    /// stored value as though it were what the server sent.
    #[test]
    fn callback_query_refuses_invalid_utf8_rather_than_substituting() {
        let result = parse_callback_query(
            "GET /callback?code=ok&state=%FF%FE HTTP/1.1\r\nHost: localhost\r\n\r\n",
        );
        assert!(
            matches!(result, Err(CallbackParseError::Malformed(_))),
            "expected a malformed-parameter refusal, got {result:?}"
        );
    }

    #[test]
    fn parse_pasted_callback_accepts_full_url() {
        // Exact shape returned by Notion after the user authorises.
        let input = "http://127.0.0.1:46437/callback?code=1d5d872b-594c-8153-a5e0-0002d8f4be0f%3AGEE3YdaPJhHZpMMa%3AjZ2YV0BC0TYheYBtoSB16LmRDTgIZ6zM&state=82Nw67m4su5AeMSfCFcXAw";
        let (code, state) = parse_pasted_callback(input).expect("should parse");
        // Percent-encoded colons in the code must be decoded.
        assert_eq!(
            code,
            "1d5d872b-594c-8153-a5e0-0002d8f4be0f:GEE3YdaPJhHZpMMa:jZ2YV0BC0TYheYBtoSB16LmRDTgIZ6zM"
        );
        assert_eq!(state, "82Nw67m4su5AeMSfCFcXAw");
    }

    #[test]
    fn parse_pasted_callback_accepts_query_only() {
        let (code, state) = parse_pasted_callback("code=abc&state=xyz").expect("should parse");
        assert_eq!(code, "abc");
        assert_eq!(state, "xyz");
    }

    #[test]
    fn parse_pasted_callback_trims_whitespace_and_fragment() {
        let (code, state) =
            parse_pasted_callback("   http://127.0.0.1:1/callback?code=a&state=b#fragment  \n")
                .expect("should parse");
        assert_eq!(code, "a");
        assert_eq!(state, "b");
    }

    #[test]
    fn parse_pasted_callback_rejects_empty_input() {
        let err = parse_pasted_callback("   \n").expect_err("empty input should fail");
        assert!(err.contains("no callback URL"));
    }

    #[test]
    fn parse_pasted_callback_surfaces_server_error_sanitised() {
        let input = "http://127.0.0.1:1/callback?error=bad%1B%5B2Jstuff%E2%80%AErtl&state=z";
        let err = parse_pasted_callback(input).expect_err("should surface error");
        assert!(!err.contains('\u{001B}'), "ANSI leaked: {}", err);
        assert!(!err.contains('\u{202E}'), "RTL leaked: {}", err);
        assert!(err.contains("bad"));
    }

    #[test]
    fn parse_pasted_callback_missing_code() {
        let err = parse_pasted_callback("state=xyz").expect_err("should fail");
        assert!(err.contains("missing 'code'"));
    }

    #[test]
    fn parse_pasted_callback_missing_state() {
        let err = parse_pasted_callback("code=abc").expect_err("should fail");
        assert!(err.contains("missing 'state'"));
    }

    #[test]
    fn test_parse_jwt_signing_algorithm_defaults() {
        let alg = parse_jwt_signing_algorithm("test", None).expect("should parse");
        assert!(matches!(alg, rmcp::transport::JwtSigningAlgorithm::RS256));
    }

    #[test]
    fn test_parse_jwt_signing_algorithm_all_values() {
        for name in &["RS256", "RS384", "RS512", "ES256", "ES384"] {
            assert!(parse_jwt_signing_algorithm("test", Some(name)).is_ok());
        }
    }

    #[test]
    fn test_parse_jwt_signing_algorithm_invalid() {
        assert!(parse_jwt_signing_algorithm("test", Some("HS256")).is_err());
    }
}
