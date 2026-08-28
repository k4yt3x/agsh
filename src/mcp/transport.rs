//! Transport construction helpers for MCP: building stdio child commands and the streamable-HTTP
//! transport config (headers, auth tokens, dynamic header helpers).

use tokio::process::Command;

use crate::{
    config::McpServerConfig,
    error::{MekaError, Result},
};

/// Build a [`Command`] for a stdio MCP server, wrapping shell shims in `cmd /c` on Windows so
/// `npx`, `*.cmd`, and `*.bat` executables can be launched directly as a command string. Unix paths
/// pass through unchanged.
pub fn build_stdio_command(command_str: &str, args: &[String]) -> Command {
    #[cfg(windows)]
    {
        let lower = command_str.to_ascii_lowercase();
        let is_shim = lower == "npx"
            || lower == "yarn"
            || lower == "pnpm"
            || lower.ends_with(".cmd")
            || lower.ends_with(".bat")
            || lower.ends_with(".ps1");
        if is_shim {
            // `cmd /c <command> <args...>`: Windows wraps argument quoting. We don't try to
            // shell-quote the args; the `Command` API does the OS-appropriate escaping via
            // CreateProcess's lpCommandLine.
            let mut cmd = Command::new("cmd");
            cmd.arg("/c").arg(command_str).args(args);
            return cmd;
        }
    }
    let _ = (command_str, args);
    let mut cmd = Command::new(command_str);
    cmd.args(args);
    cmd
}

/// Build the shared HTTP transport config (URL, bearer token, custom headers) used by both the auth
/// and no-auth paths.
pub(super) fn build_http_transport_config(
    server_name: &str,
    config: &McpServerConfig,
    // The server's stored bearer, already read from `mcp_credentials`. Passed in rather than
    // looked up here because this function is sync and the store is not, and because a
    // transport has no business knowing where a secret lives.
    bearer: Option<&str>,
) -> Result<rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig> {
    let url = config
        .url
        .as_deref()
        .ok_or_else(|| MekaError::McpConnection {
            server_name: server_name.to_string(),
            message: "http transport requires 'url' field".to_string(),
        })?;

    let mut transport_config =
        rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig::with_uri(url);

    if let Some(token) = bearer {
        transport_config = transport_config.auth_header(token.to_string());
    }

    // Merge dynamic headers from the optional `headers_helper` script on top of the static
    // `headers` map (dynamic values override static ones).
    let mut merged_headers: std::collections::HashMap<String, String> =
        config.headers.clone().unwrap_or_default();
    if let Some(script) = &config.headers_helper {
        let dynamic = run_headers_helper(server_name, url, script)?;
        merged_headers.extend(dynamic);
    }

    if !merged_headers.is_empty() {
        let mut header_map = std::collections::HashMap::new();
        for (key, value) in &merged_headers {
            let header_name =
                reqwest::header::HeaderName::from_bytes(key.as_bytes()).map_err(|error| {
                    MekaError::McpConnection {
                        server_name: server_name.to_string(),
                        message: format!("invalid header name '{}': {}", key, error),
                    }
                })?;
            let header_value = reqwest::header::HeaderValue::from_str(value).map_err(|error| {
                MekaError::McpConnection {
                    server_name: server_name.to_string(),
                    message: format!("invalid header value for '{}': {}", key, error),
                }
            })?;
            header_map.insert(header_name, header_value);
        }
        transport_config = transport_config.custom_headers(header_map);
    }

    Ok(transport_config)
}

/// Execute `headers_helper` and parse its stdout as an `Name: Value\n` stream, returning a map
/// merged into the HTTP transport's custom headers.
///
/// The script is spawned synchronously (it's a startup-path helper, not called per-request) with a
/// 15-second wall-clock timeout. `MEKA_MCP_SERVER_NAME` and `MEKA_MCP_SERVER_URL` are injected so
/// one helper can serve multiple servers.
fn run_headers_helper(
    server_name: &str,
    url: &str,
    script: &str,
) -> Result<std::collections::HashMap<String, String>> {
    use std::process::Stdio;
    let err_ctx = |msg: String| MekaError::McpConnection {
        server_name: server_name.to_string(),
        message: msg,
    };

    // Resolve the script path. If it's relative and doesn't exist as-is, try resolving against the
    // meka config directory for safety (same place config.toml lives).
    let script_path = std::path::Path::new(script);
    let resolved: std::path::PathBuf = if script_path.is_absolute() || script_path.exists() {
        script_path.to_path_buf()
    } else if let Some(config_dir) = crate::config::meka_config_dir() {
        let candidate = config_dir.join(script);
        if candidate.exists() {
            candidate
        } else {
            script_path.to_path_buf()
        }
    } else {
        script_path.to_path_buf()
    };

    let mut command = std::process::Command::new(&resolved);
    command
        .env("MEKA_MCP_SERVER_NAME", server_name)
        .env("MEKA_MCP_SERVER_URL", url)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().map_err(|error| {
        err_ctx(format!(
            "headers_helper '{}' spawn failed: {}",
            script, error
        ))
    })?;

    // Poll for exit with a 15-second budget. std::process::Child doesn't expose a blocking
    // wait_timeout, so loop on try_wait with a short sleep.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    return Err(err_ctx(format!(
                        "headers_helper '{}' timed out after 15s",
                        script
                    )));
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(error) => {
                return Err(err_ctx(format!(
                    "headers_helper '{}' wait failed: {}",
                    script, error
                )));
            }
        }
    };

    // Caps on how much helper output we're willing to buffer. stdout is the header list (rarely
    // more than a few KiB); stderr is surfaced verbatim in the error message so keep it tight.
    const MAX_HELPER_STDOUT_BYTES: u64 = 64 * 1024;
    const MAX_HELPER_STDERR_BYTES: u64 = 4 * 1024;

    if !status.success() {
        let mut stderr_buf = Vec::new();
        if let Some(stderr) = child.stderr.take() {
            use std::io::Read;
            let _ = stderr
                .take(MAX_HELPER_STDERR_BYTES)
                .read_to_end(&mut stderr_buf);
        }
        let stderr_text = String::from_utf8_lossy(&stderr_buf);
        return Err(err_ctx(format!(
            "headers_helper '{}' exited with status {}: {}",
            script,
            status.code().unwrap_or(-1),
            stderr_text.trim()
        )));
    }

    let mut stdout_buf = Vec::new();
    if let Some(pipe) = child.stdout.take() {
        use std::io::Read;
        pipe.take(MAX_HELPER_STDOUT_BYTES)
            .read_to_end(&mut stdout_buf)
            .map_err(|error| {
                err_ctx(format!(
                    "headers_helper '{}' stdout read failed: {}",
                    script, error
                ))
            })?;
    }
    let stdout = String::from_utf8_lossy(&stdout_buf);

    parse_header_lines(&stdout)
        .map_err(|msg| err_ctx(format!("headers_helper '{}' output: {}", script, msg)))
}

fn parse_header_lines(
    text: &str,
) -> std::result::Result<std::collections::HashMap<String, String>, String> {
    let mut out = std::collections::HashMap::new();
    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| format!("line {} missing ':' separator", line_no + 1))?;
        out.insert(name.trim().to_string(), value.trim().to_string());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;

    use super::*;

    fn http_server(name: &str) -> McpServerConfig {
        // Parsed rather than constructed field-by-field, so a field added to `McpServerConfig`
        // does not silently make this fixture unrepresentative of what a user actually writes.
        let toml = format!(
            "name = \"{}\"\ntransport = \"http\"\nurl = \"https://api.example.com/mcp\"\n",
            name
        );
        toml::from_str(&toml).expect("the fixture parses")
    }

    /// The one place a stored bearer becomes a request header. Everything upstream of here can be
    /// right and the server still sees no `Authorization` if this drops it.
    #[test]
    fn a_stored_bearer_becomes_the_authorization_header() {
        let config = build_http_transport_config(
            "api",
            &http_server("api"),
            Some("bearer-not-a-real-token"),
        )
        .expect("builds");

        assert_eq!(&*config.uri, "https://api.example.com/mcp");
        assert_eq!(
            config.auth_header.as_deref(),
            Some("bearer-not-a-real-token")
        );
    }

    /// A server that authenticates through a flow has no stored bearer, and must send no
    /// `Authorization` of its own: rmcp's auth layer sets that header itself.
    #[test]
    fn no_bearer_means_no_authorization_header() {
        let config = build_http_transport_config("api", &http_server("api"), None).expect("builds");
        assert!(config.auth_header.is_none());
    }

    /// The bearer and `headers` are separate inputs and both have to survive.
    #[test]
    fn a_bearer_and_custom_headers_coexist() {
        let mut server = http_server("api");
        server.headers = Some(
            [("X-Tenant-Id".to_string(), "acme".to_string())]
                .into_iter()
                .collect(),
        );

        let config = build_http_transport_config("api", &server, Some("bearer-not-a-real-token"))
            .expect("builds");

        assert_eq!(
            config.auth_header.as_deref(),
            Some("bearer-not-a-real-token")
        );
        assert_eq!(
            config
                .custom_headers
                .get(&reqwest::header::HeaderName::from_static("x-tenant-id"))
                .and_then(|value| value.to_str().ok()),
            Some("acme")
        );
    }

    #[test]
    fn parse_header_lines_basic() {
        let map = parse_header_lines("X-Token: abc\nX-Env: prod\n").expect("parses");
        assert_eq!(map.get("X-Token").map(String::as_str), Some("abc"));
        assert_eq!(map.get("X-Env").map(String::as_str), Some("prod"));
    }

    #[test]
    fn parse_header_lines_skips_blank_and_comment_lines() {
        let map = parse_header_lines("# a comment\n\n  \nX-Real: yes\n").expect("parses");
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("X-Real").map(String::as_str), Some("yes"));
    }

    #[test]
    fn parse_header_lines_trims_surrounding_whitespace() {
        let map = parse_header_lines("  Name  :   value with spaces  \n").expect("parses");
        assert_eq!(
            map.get("Name").map(String::as_str),
            Some("value with spaces")
        );
    }

    #[test]
    fn parse_header_lines_value_may_contain_colons() {
        // Only the first ':' separates; URL values must survive intact.
        let map = parse_header_lines("Location: https://example.com/x\n").expect("parses");
        assert_eq!(
            map.get("Location").map(String::as_str),
            Some("https://example.com/x")
        );
    }

    #[test]
    fn parse_header_lines_rejects_missing_separator() {
        let err = parse_header_lines("Valid: ok\nbroken line\n").expect_err("must fail");
        assert!(err.contains("line 2"), "error should cite line 2: {}", err);
    }

    #[test]
    fn build_stdio_command_passes_through_program_and_args() {
        let args = vec!["--flag".to_string(), "value".to_string()];
        let cmd = build_stdio_command("my-server", &args);
        let std_cmd = cmd.as_std();
        assert_eq!(std_cmd.get_program(), OsStr::new("my-server"));
        let collected: Vec<_> = std_cmd.get_args().collect();
        assert_eq!(collected, vec![OsStr::new("--flag"), OsStr::new("value")]);
    }

    #[cfg(windows)]
    #[test]
    fn build_stdio_command_wraps_npx_in_cmd_shim() {
        let args = vec!["my-pkg".to_string()];
        let cmd = build_stdio_command("npx", &args);
        let std_cmd = cmd.as_std();
        assert_eq!(std_cmd.get_program(), OsStr::new("cmd"));
        let collected: Vec<_> = std_cmd.get_args().collect();
        assert_eq!(collected, vec![
            OsStr::new("/c"),
            OsStr::new("npx"),
            OsStr::new("my-pkg")
        ]);
    }
}
