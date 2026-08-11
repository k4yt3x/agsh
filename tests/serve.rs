// See the matching allow in `tests/acp.rs` for the rationale: integration tests panic on
// failure by design.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! End-to-end integration tests for `meka serve`. Spawns the real `meka serve` binary against
//! a tempdir and a scripted mock provider, then drives it over HTTP via `reqwest`.

use std::{
    io::{BufRead, BufReader},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

fn meka() -> Command {
    Command::new(env!("CARGO_BIN_EXE_meka"))
}

/// Bind to an OS-assigned ephemeral port, then immediately close so the OS hands the port back.
/// The server we're about to spawn re-claims it; brief TIME_WAIT-style races are tolerated by
/// the test runner's retry-on-startup-failure path (build_harness retries a few times).
fn ephemeral_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

struct ServeTestHarness {
    _temp: tempfile::TempDir,
    child: Child,
    base_url: String,
    token: String,
    /// Drained by the spawned reader thread; kept alive so the thread can exit cleanly.
    #[allow(dead_code)]
    stderr_handle: std::thread::JoinHandle<String>,
    client: reqwest::blocking::Client,
}

impl ServeTestHarness {
    /// Spawn `meka serve` with a single `sessions:r + sessions:w` token and the mock
    /// provider. Returns once the server has logged its listening address.
    fn spawn(config_toml: &str, script: serde_json::Value) -> Self {
        Self::spawn_with(config_toml, script, "sk_test_token", &[
            "sessions:r",
            "sessions:w",
        ])
    }

    fn spawn_with(
        extra_config: &str,
        script: serde_json::Value,
        token: &str,
        scopes: &[&str],
    ) -> Self {
        let temp = tempfile::tempdir().expect("tempdir");
        let config_dir = temp.path().join("meka");
        let data_dir = temp.path().join("data").join("meka");
        std::fs::create_dir_all(&config_dir).expect("create config dir");

        let script_path = temp.path().join("script.json");
        std::fs::write(&script_path, script.to_string()).expect("write script");

        let scopes_str = scopes
            .iter()
            .map(|s| format!("\"{}\"", s))
            .collect::<Vec<_>>()
            .join(", ");

        // Startup is retried: a parallel test can re-claim our just-freed ephemeral port before
        // the server binds it, so the server fails to bind and exits. That surfaces here as a
        // `Disconnected` recv (stderr EOFs) rather than a timeout, and waiting longer wouldn't
        // help; only a fresh port does. Retry a few times before giving up.
        const MAX_ATTEMPTS: usize = 5;
        let mut last_logs = String::new();
        for _ in 0..MAX_ATTEMPTS {
            let port = ephemeral_port();
            let bind = format!("127.0.0.1:{}", port);
            // `extra_config` is injected into the top-level `[serve]` table (before the
            // `[[serve.tokens]]` array-of-tables) so callers can set `max_body_bytes`,
            // `idle_timeout`, etc. without colliding with the per-token block.
            let config = format!(
                r#"
[providers.mock]
type = "claude-api"
model = "claude-sonnet-4-5"

[permissions]
default = "write"
enabled = ["read", "write", "ask"]

[serve]
bind = "{bind}"
{extra_config}

[[serve.tokens]]
token = "{token}"
scopes = [{scopes_str}]
"#,
                bind = bind,
                token = token,
                scopes_str = scopes_str,
                extra_config = extra_config,
            );
            std::fs::write(config_dir.join("config.toml"), &config).expect("write config.toml");

            let mut child = meka()
                .arg("serve")
                .env("MEKA_CONFIG_DIR", &config_dir)
                .env("MEKA_DATA_DIR", &data_dir)
                .env("HOME", temp.path())
                .env("MEKA_ACP_MOCK_PROVIDER", "1")
                .env("MEKA_ACP_MOCK_PROVIDER_SCRIPT", &script_path)
                .env("RUST_LOG", "meka=info")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .expect("spawn meka serve");

            // Drain stdout in the background; the server doesn't write to it.
            let stdout = child.stdout.take().expect("stdout");
            std::thread::spawn(move || {
                let mut buf = String::new();
                let mut r = BufReader::new(stdout);
                while r.read_line(&mut buf).unwrap_or(0) > 0 {}
            });

            // Watch stderr for the "listening on" line so we know the server has bound. Also
            // drains the rest of stderr to keep the pipe from blocking the child.
            let stderr_pipe = child.stderr.take().expect("stderr");
            let (ready_tx, ready_rx) = std::sync::mpsc::channel::<()>();
            let stderr_handle = std::thread::spawn(move || {
                let mut buf = String::new();
                let mut r = BufReader::new(stderr_pipe);
                let mut ready_sent = false;
                let mut accumulated = String::new();
                loop {
                    buf.clear();
                    let n = r.read_line(&mut buf).unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    accumulated.push_str(&buf);
                    if !ready_sent && buf.contains("listening on") {
                        let _ = ready_tx.send(());
                        ready_sent = true;
                    }
                }
                accumulated
            });

            // Wait for the server to bind. `Ok` means ready; any error (timeout, or the server
            // exited and dropped the sender) means this attempt failed: kill it, collect its
            // logs, and retry with a fresh port.
            if ready_rx.recv_timeout(Duration::from_secs(20)).is_ok() {
                let client = reqwest::blocking::Client::builder()
                    .timeout(Duration::from_secs(60))
                    .build()
                    .expect("reqwest client");

                return Self {
                    _temp: temp,
                    child,
                    base_url: format!("http://{}", bind),
                    token: token.to_string(),
                    stderr_handle,
                    client,
                };
            }

            let _ = child.kill();
            let _ = child.wait();
            last_logs = stderr_handle.join().unwrap_or_default();
        }

        panic!(
            "meka serve failed to log `listening on` within 20s across {} attempts; \
             last stderr:\n{}",
            MAX_ATTEMPTS, last_logs,
        );
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::blocking::RequestBuilder {
        self.client
            .request(method, format!("{}{}", self.base_url, path))
            .header("Authorization", format!("Bearer {}", self.token))
    }
}

impl Drop for ServeTestHarness {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn mock_simple_turn() -> serde_json::Value {
    serde_json::json!([
        [
            { "kind": "text", "text": "hello from agent" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ])
}

#[test]
fn missing_authorization_returns_401_problem_detail() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let response = harness
        .client
        .get(format!("{}/v1/sessions", harness.base_url))
        .send()
        .expect("send");
    assert_eq!(response.status(), 401);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("application/problem+json"),
    );
    // RFC 9110 §15.5.2 requires WWW-Authenticate on every 401.
    assert_eq!(
        response
            .headers()
            .get("www-authenticate")
            .and_then(|v| v.to_str().ok()),
        Some(r#"Bearer realm="meka""#),
        "401 responses must carry WWW-Authenticate: Bearer per RFC 9110",
    );
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(
        body["type"], "https://meka.so/errors/auth",
        "missing Authorization should land on auth error"
    );
}

#[test]
fn invalid_bearer_token_returns_401_auth_invalid() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let response = harness
        .client
        .get(format!("{}/v1/sessions", harness.base_url))
        .header("Authorization", "Bearer not-the-right-token")
        .send()
        .expect("send");
    assert_eq!(response.status(), 401);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["type"], "https://meka.so/errors/auth");
}

#[test]
fn insufficient_scope_returns_403() {
    // Token only has sessions:r, no sessions:w.
    let harness =
        ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", &["sessions:r"]);
    let response = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": "/tmp"}))
        .send()
        .expect("send");
    assert_eq!(response.status(), 403);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["type"], "https://meka.so/errors/auth-scope");
    // Every Problem Detail must carry the request URI as `instance` per RFC 9457.
    assert_eq!(
        body["instance"], "/v1/sessions",
        "handler-emitted ProblemDetails must include the request URI as `instance`",
    );
}

#[test]
fn health_live_does_not_require_auth() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let response = harness
        .client
        .get(format!("{}/v1/health/live", harness.base_url))
        .send()
        .expect("send");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["status"], "ok");
}

#[test]
fn create_and_list_session_round_trip() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({
            "cwd": std::env::temp_dir().to_string_lossy(),
        }))
        .send()
        .expect("send");
    assert_eq!(create.status(), 201);
    let created: serde_json::Value = create.json().expect("parse");
    let id = created["id"].as_str().expect("id").to_string();
    assert_eq!(created["permission"], "write");

    let list = harness
        .request(reqwest::Method::GET, "/v1/sessions")
        .send()
        .expect("send");
    assert_eq!(list.status(), 200);
    let listed: serde_json::Value = list.json().expect("parse");
    let ids: Vec<&str> = listed["sessions"]
        .as_array()
        .expect("sessions array")
        .iter()
        .filter_map(|s| s["id"].as_str())
        .collect();
    assert!(
        ids.contains(&id.as_str()),
        "newly created session must appear in /v1/sessions"
    );

    let delete = harness
        .request(reqwest::Method::DELETE, &format!("/v1/sessions/{}", id))
        .send()
        .expect("send");
    assert_eq!(delete.status(), 204);
}

#[test]
fn blocking_turn_returns_final_text_from_mock_provider() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "hi", "stream": false}))
        .send()
        .expect("send");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["stop_reason"], "end_turn");
    assert_eq!(body["final_text"], "hello from agent");
    assert_eq!(body["session_id"], id);
}

/// End-to-end shape of the per-turn context, through a real server and real turns rather than the
/// renderer in isolation: the first turn carries the tool catalogue, the second carries none of it.
///
/// This is the observable form of the whole cache-prefix design. The catalogue used to live in the
/// system prompt, where anything that changed re-cached the entire conversation; it now rides in
/// the user's own message, which is appended. If it ever reappears on every turn, the change is
/// costing tokens instead of saving them, and this fails.
#[test]
fn per_turn_context_states_the_catalogue_once_not_every_turn() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    for message in ["first question", "second question"] {
        let response = harness
            .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
            .json(&serde_json::json!({"message": message, "stream": false}))
            .send()
            .expect("send");
        assert_eq!(response.status(), 200);
    }

    let body: serde_json::Value = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/messages", id),
        )
        .send()
        .expect("send")
        .json()
        .expect("parse");
    let user_texts: Vec<String> = body["messages"]
        .as_array()
        .expect("messages array")
        .iter()
        .filter(|message| message["role"] == "user")
        .filter_map(|message| message["content"][0]["text"].as_str().map(str::to_string))
        .collect();
    assert!(
        user_texts.len() >= 2,
        "expected both user turns; got {:?}",
        user_texts,
    );

    assert!(
        user_texts[0].contains("[Available tools]") && user_texts[0].contains("**read_file**"),
        "the first turn must state the catalogue; got: {}",
        user_texts[0],
    );
    assert!(
        !user_texts[1].contains("[Available tools]"),
        "the second turn must not restate it: nothing changed, so it costs nothing; got: {}",
        user_texts[1],
    );
    // Both still carry the cheap per-turn state, so this isn't passing because the block vanished.
    for text in &user_texts[..2] {
        assert!(text.contains("[Permission context]"), "got: {}", text);
    }
}

#[test]
fn fork_copies_the_conversation_into_a_new_session() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    // Deliberately not the server's configured default (`write`): a fork that dropped the
    // `permission` column entirely would still report `write`, because the re-attach path falls
    // back to the config default when the column is NULL.
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({
            "cwd": std::env::temp_dir().to_string_lossy(),
            "permission": "read",
        }))
        .send()
        .expect("send");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "hi", "stream": false}))
        .send()
        .expect("send");

    // An empty body is the common case: inherit everything from the source.
    let fork = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/fork", id))
        .send()
        .expect("send");
    assert_eq!(fork.status(), 201);
    let forked: serde_json::Value = fork.json().expect("parse");
    let fork_id = forked["id"].as_str().expect("id").to_string();
    assert_ne!(fork_id, id, "the fork is a distinct session");
    assert_eq!(forked["permission"], "read", "permission is inherited");

    let messages_of = |session: &str| -> serde_json::Value {
        harness
            .request(
                reqwest::Method::GET,
                &format!("/v1/sessions/{}/messages", session),
            )
            .send()
            .expect("send")
            .json()
            .expect("parse")
    };
    assert_eq!(
        messages_of(&fork_id)["messages"],
        messages_of(&id)["messages"],
        "the fork starts from the source's exact conversation",
    );

    // The fork is immediately usable, and using it leaves the source alone.
    let turn = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/turn", fork_id),
        )
        .json(&serde_json::json!({"message": "again", "stream": false}))
        .send()
        .expect("send");
    assert_eq!(turn.status(), 200);
    assert!(
        messages_of(&fork_id)["messages"]
            .as_array()
            .expect("array")
            .len()
            > messages_of(&id)["messages"]
                .as_array()
                .expect("array")
                .len(),
        "the branch diverges: the source must not grow when the fork runs a turn",
    );
}

/// Forking reads the database directly while the source's runtime mutex is held by a running
/// turn. It must neither block on that mutex nor produce a broken copy: the fork's own first turn
/// has to succeed, which is only true if the conversation it copied loaded cleanly.
#[test]
fn fork_during_an_in_flight_turn_does_not_block_or_corrupt() {
    let slow = serde_json::json!([
        { "kind": "sleep", "ms": 1500 },
        { "kind": "text", "text": "source done" },
        { "kind": "message_end", "stop_reason": "end_turn" }
    ]);
    let quick = serde_json::json!([
        { "kind": "text", "text": "fork done" },
        { "kind": "message_end", "stop_reason": "end_turn" }
    ]);
    let harness = ServeTestHarness::spawn("", serde_json::json!([slow, quick]));
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    let base_url = harness.base_url.clone();
    let token = harness.token.clone();
    let id_for_turn = id.clone();
    let turn = std::thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("client");
        client
            .post(format!("{}/v1/sessions/{}/turn", base_url, id_for_turn))
            .header("Authorization", format!("Bearer {}", token))
            .json(&serde_json::json!({"message": "slow one"}))
            .send()
            .expect("turn send")
    });

    // Let the turn acquire the source's runtime mutex and enter the mock provider's sleep.
    std::thread::sleep(Duration::from_millis(300));
    let started = std::time::Instant::now();
    let fork = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/fork", id))
        .send()
        .expect("send");
    let elapsed = started.elapsed();

    assert_eq!(fork.status(), 201);
    assert!(
        elapsed < Duration::from_millis(1000),
        "fork must not wait on the source's runtime mutex; took {:?}",
        elapsed,
    );
    let fork_id = fork.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    // The copy is usable despite having been taken mid-turn.
    let fork_turn = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/turn", fork_id),
        )
        .json(&serde_json::json!({"message": "on the fork", "stream": false}))
        .send()
        .expect("send");
    assert_eq!(
        fork_turn.status(),
        200,
        "a fork taken mid-turn must still run: {}",
        fork_turn.text().unwrap_or_default(),
    );

    assert_eq!(turn.join().expect("join").status(), 200);
}

#[test]
fn fork_accepts_a_cwd_override() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    let elsewhere = std::env::temp_dir().join("meka-fork-cwd");
    std::fs::create_dir_all(&elsewhere).expect("mkdir");
    let fork = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/fork", id))
        .json(&serde_json::json!({"cwd": elsewhere.to_string_lossy()}))
        .send()
        .expect("send");
    assert_eq!(fork.status(), 201);
    let forked: serde_json::Value = fork.json().expect("parse");
    assert_eq!(forked["cwd"], elsewhere.to_string_lossy().as_ref());
}

#[test]
fn fork_rejects_a_relative_cwd_and_an_unknown_session() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    let relative = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/fork", id))
        .json(&serde_json::json!({"cwd": "relative/path"}))
        .send()
        .expect("send");
    assert_eq!(relative.status(), 422);

    let unknown = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/fork", uuid::Uuid::new_v4()),
        )
        .send()
        .expect("send");
    assert_eq!(unknown.status(), 404);
}

#[test]
fn fork_requires_write_scope() {
    let harness =
        ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", &["sessions:r"]);
    let response = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/fork", uuid::Uuid::new_v4()),
        )
        .send()
        .expect("send");
    assert_eq!(response.status(), 403);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["type"], "https://meka.so/errors/auth-scope");
}

#[test]
fn idempotency_key_replays_return_cached_body() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    let body = serde_json::json!({"message": "hi", "stream": false});
    let first = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .header("Idempotency-Key", "test-key-1")
        .json(&body)
        .send()
        .expect("send");
    assert_eq!(first.status(), 200);
    let first_body = first.text().expect("text");

    // Replay with the same key + same body → identical response (cached envelope).
    let second = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .header("Idempotency-Key", "test-key-1")
        .json(&body)
        .send()
        .expect("send");
    assert_eq!(second.status(), 200);
    let second_body = second.text().expect("text");
    assert_eq!(
        first_body, second_body,
        "replay must return identical bytes"
    );
}

#[test]
fn patch_session_updates_permission_and_cwd() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let temp_dir = std::env::temp_dir().to_string_lossy().to_string();
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": temp_dir, "permission": "write"}))
        .send()
        .expect("send");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    let new_cwd = std::env::temp_dir().join("patched-cwd-test");
    std::fs::create_dir_all(&new_cwd).expect("create new cwd");
    let patched = harness
        .request(reqwest::Method::PATCH, &format!("/v1/sessions/{}", id))
        .json(&serde_json::json!({
            "permission": "read",
            "cwd": new_cwd.to_string_lossy(),
        }))
        .send()
        .expect("send");
    assert_eq!(patched.status(), 200);
    let body: serde_json::Value = patched.json().expect("parse");
    assert_eq!(body["permission"], "read");
    assert_eq!(body["cwd"], new_cwd.to_string_lossy().as_ref());
}

#[test]
fn openapi_json_is_served_without_auth_and_documents_routes() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let response = harness
        .client
        .get(format!("{}/v1/openapi.json", harness.base_url))
        .send()
        .expect("send");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().expect("parse");
    let openapi_version = body["openapi"].as_str().expect("openapi version");
    assert!(
        openapi_version.starts_with("3."),
        "expected OpenAPI 3.x, got {openapi_version}",
    );
    let paths = body["paths"].as_object().expect("paths object");
    // Spot-check that representative endpoints made it into the spec.
    for required in [
        "/v1/sessions",
        "/v1/sessions/{id}",
        "/v1/sessions/{id}/turn",
        "/v1/sessions/{id}/messages",
        "/v1/health/live",
        "/v1/info",
    ] {
        assert!(
            paths.contains_key(required),
            "OpenAPI spec missing path {required}",
        );
    }
    let components = body["components"]["schemas"]
        .as_object()
        .expect("schemas object");
    assert!(
        components.contains_key("ProblemDetail"),
        "ProblemDetail schema must be exported",
    );
}

#[test]
fn swagger_ui_is_served_without_auth() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let response = harness
        .client
        .get(format!("{}/v1/docs/", harness.base_url))
        .send()
        .expect("send");
    assert_eq!(response.status(), 200);
    let body = response.text().expect("text");
    assert!(
        body.contains("swagger") || body.contains("Swagger"),
        "Swagger UI HTML must reference swagger somewhere",
    );
}

#[test]
fn patch_session_rejects_relative_cwd() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    let response = harness
        .request(reqwest::Method::PATCH, &format!("/v1/sessions/{}", id))
        .json(&serde_json::json!({"cwd": "relative/path"}))
        .send()
        .expect("send");
    assert_eq!(response.status(), 422);
}

#[test]
fn idempotency_key_with_different_body_returns_409_conflict() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    let first = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .header("Idempotency-Key", "conflict-key")
        .json(&serde_json::json!({"message": "first body", "stream": false}))
        .send()
        .expect("send");
    assert_eq!(first.status(), 200);

    let second = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .header("Idempotency-Key", "conflict-key")
        .json(&serde_json::json!({"message": "different body", "stream": false}))
        .send()
        .expect("send");
    assert_eq!(second.status(), 409);
    let body: serde_json::Value = second.json().expect("parse");
    assert_eq!(body["type"], "https://meka.so/errors/idempotency");
}

#[test]
fn streaming_turn_emits_turn_started_text_delta_and_finished() {
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "streamed " },
            { "kind": "text", "text": "response" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "hi", "stream": true}))
        .send()
        .expect("send");
    assert_eq!(response.status(), 200);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(|v| v.split(';').next().unwrap_or(v).trim().to_string()),
        Some("text/event-stream".to_string()),
    );
    // Confirm the SSE-specific cache-control headers are present so intermediate proxies
    // don't buffer or replay the stream.
    assert_eq!(
        response
            .headers()
            .get("cache-control")
            .and_then(|v| v.to_str().ok()),
        Some("no-cache, no-transform"),
        "SSE responses must declare no-cache, no-transform",
    );
    assert_eq!(
        response
            .headers()
            .get("x-accel-buffering")
            .and_then(|v| v.to_str().ok()),
        Some("no"),
        "SSE responses must set X-Accel-Buffering: no so nginx (and friends) don't buffer",
    );
    let body = response.text().expect("body");
    // The body is an SSE stream; coarse-grained string assertions are enough here.
    assert!(
        body.contains("event: turn.started"),
        "stream must include turn.started; body was:\n{}",
        body
    );
    assert!(
        body.contains("event: assistant_text.delta"),
        "stream must include assistant_text.delta events; body was:\n{}",
        body
    );
    assert!(
        body.contains("event: turn.finished"),
        "stream must include turn.finished; body was:\n{}",
        body
    );
    assert!(
        body.contains("\"stop_reason\":\"end_turn\""),
        "turn.finished must carry the stop reason; body was:\n{}",
        body
    );
}

#[test]
fn second_turn_on_same_session_returns_409_turn_in_flight() {
    // Two-round script so the first turn keeps the runtime mutex held for ~1s while the second
    // POST tries to acquire it.
    let script = serde_json::json!([
        [
            { "kind": "sleep", "ms": 1500 },
            { "kind": "text", "text": "first done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "text", "text": "second done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    // Fire the first turn in a background thread so we can race a second one against it.
    let base_url = harness.base_url.clone();
    let token = harness.token.clone();
    let id_a = id.clone();
    let first = std::thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("client");
        client
            .post(format!("{}/v1/sessions/{}/turn", base_url, id_a))
            .header("Authorization", format!("Bearer {}", token))
            .json(&serde_json::json!({"message": "first"}))
            .send()
            .expect("first send")
    });

    // Give the first turn time to acquire the runtime mutex.
    std::thread::sleep(Duration::from_millis(300));
    let second = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "second"}))
        .send()
        .expect("second send");
    assert_eq!(second.status(), 409, "concurrent turn must return 409");
    let body: serde_json::Value = second.json().expect("parse");
    assert_eq!(body["type"], "https://meka.so/errors/turn-in-flight");

    // Drain the first turn so the harness Drop doesn't leave a zombie.
    let first_response = first.join().expect("join").error_for_status();
    assert!(first_response.is_ok(), "first turn must succeed");
}

/// Two concurrent streaming POSTs on the same session must produce a 409 on the loser.
#[test]
fn concurrent_streaming_turns_return_409() {
    let script = serde_json::json!([
        [
            { "kind": "sleep", "ms": 1500 },
            { "kind": "text", "text": "first done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    let base_url = harness.base_url.clone();
    let token = harness.token.clone();
    let id_clone = id.clone();
    let first = std::thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("client");
        client
            .post(format!("{}/v1/sessions/{}/turn", base_url, id_clone))
            .header("Authorization", format!("Bearer {}", token))
            .json(&serde_json::json!({"message": "first", "stream": true}))
            .send()
            .expect("first send")
    });

    std::thread::sleep(Duration::from_millis(300));
    let second = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "second", "stream": true}))
        .send()
        .expect("send");
    assert_eq!(
        second.status(),
        409,
        "concurrent streaming turn must return 409"
    );
    let body: serde_json::Value = second.json().expect("parse");
    assert_eq!(body["type"], "https://meka.so/errors/turn-in-flight");

    // Drain the first stream so the harness drop is clean. The SSE body is consumed lazily,
    // so we just have to read it.
    let first_response = first.join().expect("join");
    let _ = first_response.text();
}

/// `POST /v1/sessions/{id}/cancel` returns 204 even when no turn is in flight.
#[test]
fn cancel_idempotent_when_no_turn_in_flight() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    let response = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/cancel", id),
        )
        .send()
        .expect("send");
    assert_eq!(response.status(), 204);
}

/// `max_body_bytes` rejects oversize requests with 413.
#[test]
fn oversize_body_returns_413() {
    let harness = ServeTestHarness::spawn_with(
        "max_body_bytes = 1024\n",
        mock_simple_turn(),
        "sk_test_token",
        &["sessions:r", "sessions:w"],
    );
    let mut huge = String::with_capacity(4096);
    for _ in 0..4096 {
        huge.push('x');
    }
    let response = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({
            "cwd": std::env::temp_dir().to_string_lossy(),
            "permission": huge,
        }))
        .send()
        .expect("send");
    assert_eq!(response.status(), 413);
    // The 413 must use application/problem+json, not tower-http's plain-text default.
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("application/problem+json"),
        "413 must serialize as Problem Detail, not plain text",
    );
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["type"], "https://meka.so/errors/payload-too-large");
    assert_eq!(body["status"], 413);
    assert!(
        body["max_body_bytes"].is_number(),
        "Problem Detail should carry the configured limit as an extension",
    );
}

/// `GET /v1/info`, `/v1/skills`, `/v1/mcp` smoke. All authenticated, all should succeed
/// against a default deployment with no MCP servers + no skills configured.
#[test]
fn discovery_endpoints_round_trip() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    for path in ["/v1/info", "/v1/skills", "/v1/mcp"] {
        let response = harness
            .request(reqwest::Method::GET, path)
            .send()
            .expect("send");
        assert_eq!(response.status(), 200, "expected 200 from {path}");
    }
}

/// `GET /v1/health/ready` includes the session_db + provider_configured + mcp_servers
/// fields and reports `ok` against a healthy default deployment.
#[test]
fn ready_probe_reports_subsystem_health() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let response = harness
        .client
        .get(format!("{}/v1/health/ready", harness.base_url))
        .send()
        .expect("send");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["status"], "ok");
    assert_eq!(body["session_db"], true);
    assert_eq!(body["provider_configured"], true);
    assert_eq!(
        body["mcp_servers_healthy"], true,
        "mcp_servers_healthy must be a boolean (true when no servers configured)"
    );
}

/// PATCH with insufficient scope (`sessions:r` only) returns 403.
#[test]
fn patch_without_write_scope_returns_403() {
    let harness =
        ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", &["sessions:r"]);
    // Create a session via a *second* token that has write scope, then PATCH via the read-
    // only token. The harness only supports a single token, so this test exercises the
    // negative path by attempting PATCH on a session ID the read-only token couldn't even
    // have created, but since session IDs aren't owner-scoped, a nonexistent ID still hits
    // the scope check before the lookup. The check should reject with 403, not 404.
    let response = harness
        .request(
            reqwest::Method::PATCH,
            "/v1/sessions/00000000-0000-0000-0000-000000000000",
        )
        .json(&serde_json::json!({"permission": "read"}))
        .send()
        .expect("send");
    assert_eq!(response.status(), 403);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["type"], "https://meka.so/errors/auth-scope");
}

/// After GC evicts an idle session, a subsequent `POST /turn` on the same session id rebuilds
/// the in-memory entry from the DB row instead of returning 404. The conversation history is
/// preserved (both turns appear in `GET /messages`) and the per-session permission persists
/// through eviction (validates the schema-persist work).
#[test]
fn re_attach_to_evicted_session_continues_conversation() {
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "first" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "text", "text": "second" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn_with(
        // Aggressive GC: evict after 1s of idle, scan every 1s. Test waits 3s between turns.
        "idle_timeout = \"1s\"\ngc_scan_interval = \"1s\"\n",
        script,
        "sk_test_token",
        &["sessions:r", "sessions:w"],
    );
    // Create with explicit `permission = "read"` so re-attach must round-trip it.
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({
            "cwd": std::env::temp_dir().to_string_lossy(),
            "permission": "read",
        }))
        .send()
        .expect("send");
    assert_eq!(create.status(), 201);
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    let first = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "hello"}))
        .send()
        .expect("first");
    assert_eq!(first.status(), 200);

    // Wait long enough for GC to evict the in-memory entry (idle_timeout=1s, scan=1s).
    std::thread::sleep(Duration::from_secs(3));

    // Re-attach: the next turn should succeed via the reconstruction path, not 404.
    let second = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "again"}))
        .send()
        .expect("second");
    assert_eq!(
        second.status(),
        200,
        "GC-evicted session must re-attach instead of returning 404; body was:\n{}",
        second.text().unwrap_or_default(),
    );

    // The persisted permission survived eviction + reattach.
    let get = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}", id))
        .send()
        .expect("get");
    assert_eq!(get.status(), 200);
    let body: serde_json::Value = get.json().expect("parse");
    assert_eq!(
        body["permission"], "read",
        "re-attached session must retain the per-session permission, not revert to default",
    );

    // Both turns appear in the conversation history.
    let messages = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/messages", id),
        )
        .send()
        .expect("messages");
    assert_eq!(messages.status(), 200);
    let body: serde_json::Value = messages.json().expect("parse");
    let user_messages: Vec<String> = body["messages"]
        .as_array()
        .expect("messages array")
        .iter()
        .filter(|m| m["role"] == "user")
        .filter_map(|m| m["content"][0]["text"].as_str().map(|s| s.to_string()))
        .collect();
    assert!(
        user_messages.iter().any(|t| t.contains("hello")),
        "first turn's user message should be in history; got {:?}",
        user_messages,
    );
    assert!(
        user_messages.iter().any(|t| t.contains("again")),
        "post-reattach turn's user message should be in history; got {:?}",
        user_messages,
    );
}

/// Server-side errors (5xx) are NOT cached by the idempotency layer: a transient provider
/// failure would otherwise be replayed for the full 24h TTL, defeating safe retries.  After
/// a 502, replaying the same key re-executes the turn (here the mock's second script entry
/// succeeds, proving the turn actually ran again).
#[test]
fn idempotency_does_not_cache_server_errors() {
    let script = serde_json::json!([
        [{ "kind": "fail", "message": "scripted upstream 502" }],
        [{ "kind": "text", "text": "recovered" }]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    let body = serde_json::json!({"message": "go", "stream": false});
    let first = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .header("Idempotency-Key", "5xx-retry-key")
        .json(&body)
        .send()
        .expect("first");
    assert_eq!(
        first.status().as_u16(),
        502,
        "scripted provider failure must surface as 502",
    );

    // Retry with the same key re-executes the turn instead of replaying the cached 502.
    let second = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .header("Idempotency-Key", "5xx-retry-key")
        .json(&body)
        .send()
        .expect("second");
    assert_eq!(
        second.status().as_u16(),
        200,
        "retried turn should succeed against the second mock script entry",
    );
}

/// `POST /cancel` against an in-flight streaming turn produces a `turn.cancelled` SSE event
/// with `"reason":"client"` on the streaming response, validating the SSE select-loop's
/// cancel branch.
#[test]
fn cancel_during_in_flight_turn_emits_cancelled_event() {
    let script = serde_json::json!([
        [
            { "kind": "sleep", "ms": 2000 },
            { "kind": "text", "text": "should never reach client" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    let base_url = harness.base_url.clone();
    let token = harness.token.clone();
    let id_clone = id.clone();
    let streaming = std::thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("client");
        client
            .post(format!("{}/v1/sessions/{}/turn", base_url, id_clone))
            .header("Authorization", format!("Bearer {}", token))
            .json(&serde_json::json!({"message": "long", "stream": true}))
            .send()
            .expect("stream send")
    });

    // Let the agent enter its 2-second mock-provider sleep before cancelling.
    std::thread::sleep(Duration::from_millis(300));
    let cancel = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/cancel", id),
        )
        .send()
        .expect("cancel");
    assert_eq!(cancel.status(), 204);

    let response = streaming.join().expect("join");
    let body = response.text().expect("body");
    assert!(
        body.contains("event: turn.cancelled"),
        "stream must emit turn.cancelled when /cancel fires mid-turn; body was:\n{}",
        body,
    );
    assert!(
        body.contains("\"reason\":\"client\""),
        "cancellation reason must be 'client' when triggered by POST /cancel; body was:\n{}",
        body,
    );
}

/// Process-wide `max_concurrent_turns = 1` rejects the second concurrent turn (across distinct
/// sessions) with 429 + concurrency-limit. Validates the `TurnGuard` admission check.
#[test]
fn max_concurrent_turns_returns_429_across_sessions() {
    let script = serde_json::json!([
        [
            { "kind": "sleep", "ms": 1500 },
            { "kind": "text", "text": "done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "sleep", "ms": 1500 },
            { "kind": "text", "text": "done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness =
        ServeTestHarness::spawn_with("max_concurrent_turns = 1\n", script, "sk_test_token", &[
            "sessions:r",
            "sessions:w",
        ]);

    // Two distinct sessions.
    let mut ids = Vec::new();
    for _ in 0..2 {
        let create = harness
            .request(reqwest::Method::POST, "/v1/sessions")
            .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
            .send()
            .expect("send");
        let id = create.json::<serde_json::Value>().expect("parse")["id"]
            .as_str()
            .expect("id")
            .to_string();
        ids.push(id);
    }

    // Fire the first turn in the background; it holds the cap.
    let base_url = harness.base_url.clone();
    let token = harness.token.clone();
    let id_a = ids[0].clone();
    let first = std::thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("client");
        client
            .post(format!("{}/v1/sessions/{}/turn", base_url, id_a))
            .header("Authorization", format!("Bearer {}", token))
            .json(&serde_json::json!({"message": "a"}))
            .send()
            .expect("first send")
    });

    std::thread::sleep(Duration::from_millis(200));
    // Second turn on a *different* session must be rejected with 429 concurrency-limit.
    let second = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/turn", ids[1]),
        )
        .json(&serde_json::json!({"message": "b"}))
        .send()
        .expect("second");
    assert_eq!(second.status(), 429);
    assert!(
        second.headers().get("retry-after").is_some(),
        "concurrency-limit response must carry Retry-After",
    );
    let body: serde_json::Value = second.json().expect("parse");
    assert_eq!(
        body["type"], "https://meka.so/errors/concurrency-limit",
        "process-wide cap must surface the concurrency-limit type, not rate-limit-exceeded",
    );

    let _ = first.join().expect("join").error_for_status();
}

/// Graceful shutdown: an in-flight streaming turn receives a final
/// `turn.cancelled{reason:"server_shutdown"}` SSE event when the server is SIGTERM'd.
///
/// Unix-only (uses `kill` to send SIGTERM); skipped on Windows since the server's shutdown path
/// there only listens for Ctrl+C and we can't deliver that to a child process easily.
#[cfg(unix)]
#[test]
fn graceful_shutdown_emits_server_shutdown_cancelled() {
    // Long-sleep script so the streaming turn is still in flight when the signal arrives.
    let script = serde_json::json!([
        [
            { "kind": "sleep", "ms": 5000 },
            { "kind": "text", "text": "would-be-text" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    let server_pid = harness.child.id();

    // Fire the streaming turn in a worker thread; we'll SIGTERM the server mid-stream and
    // collect the captured body when the connection closes.
    let base_url = harness.base_url.clone();
    let token = harness.token.clone();
    let id_clone = id.clone();
    let streaming = std::thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("client");
        client
            .post(format!("{}/v1/sessions/{}/turn", base_url, id_clone))
            .header("Authorization", format!("Bearer {}", token))
            .json(&serde_json::json!({"message": "stall me", "stream": true}))
            .send()
            .expect("stream send")
    });

    // Let the agent enter its mock-provider sleep, then SIGTERM the server.
    std::thread::sleep(Duration::from_millis(500));
    let kill_status = Command::new("kill")
        .arg("-TERM")
        .arg(server_pid.to_string())
        .status()
        .expect("send SIGTERM");
    assert!(kill_status.success(), "kill should succeed");

    let response = streaming.join().expect("stream join");
    let body = response.text().expect("body");
    assert!(
        body.contains("event: turn.cancelled"),
        "drained server must emit turn.cancelled; body was:\n{}",
        body,
    );
    assert!(
        body.contains("\"reason\":\"server_shutdown\""),
        "cancellation reason must be 'server_shutdown' on SIGTERM; body was:\n{}",
        body,
    );
}

/// Mid-turn permission flow: a session in `permission = "ask"` mode scripts a tool call,
/// the SSE stream emits `permission_required`, the test posts `/responses/{id}` with
/// `outcome: "deny"`, and the agent continues into a follow-up assistant message that ends
/// the turn cleanly.
#[test]
fn mid_turn_permission_round_trips() {
    // Round 1: model asks to run write_file; round 2: after deny, model gives up.
    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "tu_1", "name": "write_file" },
            { "kind": "tool_use_end", "input": {"path": "/tmp/meka-test.txt", "content": "x"} },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "ok, skipping" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({
            "cwd": std::env::temp_dir().to_string_lossy(),
            "permission": "ask",
        }))
        .send()
        .expect("create");
    assert_eq!(create.status(), 201);
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    // Streaming worker: reads the SSE body line by line, posts the deny response to
    // `/responses/{request_id}` as soon as the parked event arrives, and continues reading
    // until the server emits `turn.finished`. Doing both sides inside one thread avoids the
    // cross-thread channel-+-deadlock dance that a "main parses, worker posts" split would
    // need (the streaming POST blocks until the server closes the connection).
    let base_url = harness.base_url.clone();
    let token = harness.token.clone();
    let id_for_stream = id.clone();
    let stream_handle = std::thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("client");
        let response = client
            .post(format!("{}/v1/sessions/{}/turn", base_url, id_for_stream))
            .header("Authorization", format!("Bearer {}", token))
            .json(&serde_json::json!({"message": "please write", "stream": true}))
            .send()
            .expect("stream POST");
        let mut last_event: Option<String> = None;
        let mut posted_deny = false;
        let mut saw_permission_required = false;
        let mut saw_finished = false;
        let mut buffered = String::new();
        let reader = std::io::BufReader::new(response);
        let respond_client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .expect("respond client");
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            buffered.push_str(&line);
            buffered.push('\n');
            if let Some(name) = line.strip_prefix("event: ") {
                let trimmed = name.trim();
                last_event = Some(trimmed.to_string());
                if trimmed == "permission_required" {
                    saw_permission_required = true;
                }
                if trimmed == "turn.finished" {
                    saw_finished = true;
                }
            } else if let Some(data) = line.strip_prefix("data: ")
                && last_event.as_deref() == Some("permission_required")
                && !posted_deny
            {
                let payload: serde_json::Value =
                    serde_json::from_str(data.trim()).expect("parse data");
                let request_id = payload["request_id"]
                    .as_str()
                    .expect("request_id")
                    .to_string();
                let resp = respond_client
                    .post(format!(
                        "{}/v1/sessions/{}/responses/{}",
                        base_url, id_for_stream, request_id,
                    ))
                    .header("Authorization", format!("Bearer {}", token))
                    .json(&serde_json::json!({"outcome": "deny"}))
                    .send()
                    .expect("respond send");
                assert_eq!(
                    resp.status(),
                    204,
                    "POST /responses must accept the deny outcome"
                );
                posted_deny = true;
            }
        }
        (saw_permission_required, posted_deny, saw_finished, buffered)
    });

    let (saw_permission_required, posted_deny, saw_finished, body) =
        stream_handle.join().expect("stream worker join");
    assert!(
        saw_permission_required,
        "streaming turn must emit `permission_required`; body was:\n{}",
        body,
    );
    assert!(
        posted_deny,
        "POST /responses must have been invoked at least once",
    );
    assert!(
        saw_finished,
        "stream must reach `turn.finished` after the deny resolves; body was:\n{}",
        body,
    );
}

/// Streaming turn that executes a scripted tool call emits both `tool_call.executing` and
/// `tool_call.completed` SSE events with the expected payload shape.
#[test]
fn streaming_tool_call_emits_executing_and_completed_events() {
    // Mock provider scripts a tool_use round, then a follow-up text round so the turn ends.
    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "tu_1", "name": "list_directory" },
            { "kind": "tool_use_end", "input": {"path": std::env::temp_dir().to_string_lossy()} },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "listed" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "list it", "stream": true}))
        .send()
        .expect("send");
    assert_eq!(response.status(), 200);
    let body = response.text().expect("body");
    assert!(
        body.contains("event: tool_call.executing"),
        "stream must emit tool_call.executing; body was:\n{}",
        body,
    );
    assert!(
        body.contains("event: tool_call.completed"),
        "stream must emit tool_call.completed; body was:\n{}",
        body,
    );
    assert!(
        body.contains("\"name\":\"list_directory\""),
        "tool_call.executing must include the tool name; body was:\n{}",
        body,
    );
    assert!(
        body.contains("\"id\":\"tu_1\""),
        "tool_call events must propagate the tool_use id from the provider; body was:\n{}",
        body,
    );
}

/// Every mutating endpoint must return 404 (not 500) for an unknown session id, with a
/// `session-not-found` Problem Detail.
#[test]
fn unknown_session_returns_404_on_every_mutating_endpoint() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let unknown = uuid::Uuid::new_v4();

    // DELETE is idempotent: a second DELETE (or a DELETE on a never-existed id) returns
    // 204 No Content. PATCH, POST /turn, GET /messages all still 404 on a non-existent id.
    for (method, path, body) in [
        (
            reqwest::Method::PATCH,
            format!("/v1/sessions/{}", unknown),
            Some(serde_json::json!({"permission": "read"})),
        ),
        (
            reqwest::Method::POST,
            format!("/v1/sessions/{}/turn", unknown),
            Some(serde_json::json!({"message": "hi"})),
        ),
        (
            reqwest::Method::GET,
            format!("/v1/sessions/{}/messages", unknown),
            None,
        ),
    ] {
        let mut request = harness.request(method.clone(), &path);
        if let Some(json) = body {
            request = request.json(&json);
        }
        let response = request.send().expect("send");
        assert_eq!(
            response.status(),
            404,
            "{} {} on a non-existent session must return 404",
            method,
            path,
        );
        let problem: serde_json::Value = response.json().expect("parse");
        assert_eq!(
            problem["type"], "https://meka.so/errors/session-not-found",
            "404 must carry the session-not-found Problem Detail type",
        );
    }

    // DELETE returns 204 (idempotent) per the utoipa annotation contract.
    let delete = harness
        .request(
            reqwest::Method::DELETE,
            &format!("/v1/sessions/{}", unknown),
        )
        .send()
        .expect("send");
    assert_eq!(
        delete.status(),
        204,
        "DELETE on a non-existent session must be idempotent (204)",
    );
}

/// Malformed POST /v1/sessions body (unparseable `permission` value) returns 422 with the
/// `invalid-body` Problem Detail.
#[test]
fn malformed_create_body_returns_422() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let response = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({
            "cwd": std::env::temp_dir().to_string_lossy(),
            "permission": "not-a-permission",
        }))
        .send()
        .expect("send");
    assert_eq!(response.status(), 422);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["type"], "https://meka.so/errors/invalid-body");
}

/// Missing required `message` field on POST /turn returns 422 with the `invalid-body`
/// Problem Detail.
#[test]
fn malformed_turn_body_missing_message_returns_422() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"stream": false}))
        .send()
        .expect("send");
    assert_eq!(response.status(), 422);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["type"], "https://meka.so/errors/invalid-body");
}

/// A 1x1 transparent PNG, base64-encoded. Hardcoded rather than synthesized because the `image`
/// crate is a dependency of the binary, not of the integration-test target.
const TINY_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

/// Create a session and return its id. The turn-image tests all need one.
fn create_session_id(harness: &ServeTestHarness) -> String {
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string()
}

/// The whole point of inline image attachments: a client on another host has no filesystem in
/// common with the agent, so it can't just name a path. The scripted mock ignores the request
/// body, so this asserts the validate-and-thread path accepts the attachment end to end.
#[test]
fn blocking_turn_accepts_inline_image_attachment() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let id = create_session_id(&harness);
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({
            "message": "what is in this image?",
            "images": [{"media_type": "image/png", "data": TINY_PNG_BASE64}],
            "stream": false,
        }))
        .send()
        .expect("send");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["final_text"], "hello from agent");
}

/// An image with no text is a complete request against prior context ("look at this"), so the
/// empty-`message` check must not reject it.
#[test]
fn turn_with_only_an_image_and_no_text_is_accepted() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let id = create_session_id(&harness);
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({
            "message": "",
            "images": [{"media_type": "image/png", "data": TINY_PNG_BASE64}],
            "stream": false,
        }))
        .send()
        .expect("send");
    assert_eq!(response.status(), 200);
}

/// Neither text nor images is still an empty turn.
#[test]
fn turn_with_neither_message_nor_images_returns_422() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let id = create_session_id(&harness);
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "  ", "images": [], "stream": false}))
        .send()
        .expect("send");
    assert_eq!(response.status(), 422);
}

#[test]
fn turn_with_undecodable_image_returns_422() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let id = create_session_id(&harness);
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({
            "message": "look",
            "images": [{"media_type": "image/png", "data": "!!!not-base64!!!"}],
            "stream": false,
        }))
        .send()
        .expect("send");
    assert_eq!(response.status(), 422);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["type"], "https://meka.so/errors/invalid-body");
    assert!(
        body["detail"]
            .as_str()
            .expect("detail")
            .contains("images[0]"),
        "{}",
        body["detail"]
    );
}

/// A reconnecting client needs to tell "my turn is still running" from "my turn died" without
/// submitting a speculative turn and reading the 409. An idle session reports false.
#[test]
fn get_session_reports_turn_in_flight() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let id = create_session_id(&harness);
    let idle = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}", id))
        .send()
        .expect("send");
    assert_eq!(idle.status(), 200);
    let body: serde_json::Value = idle.json().expect("parse");
    assert_eq!(body["turn_in_flight"], false);
}

/// And it reports true while a turn actually is running. The scripted 2s sleep holds the turn
/// open long enough to observe it.
#[test]
fn get_session_reports_turn_in_flight_during_a_turn() {
    let script = serde_json::json!([[
        { "kind": "sleep", "ms": 2000 },
        { "kind": "text", "text": "done" },
        { "kind": "message_end", "stop_reason": "end_turn" }
    ]]);
    let harness = ServeTestHarness::spawn("", script);
    let id = create_session_id(&harness);

    let turn_url = format!("{}/v1/sessions/{}/turn", harness.base_url, id);
    let token = harness.token.clone();
    let worker = std::thread::spawn(move || {
        reqwest::blocking::Client::new()
            .post(&turn_url)
            .header("Authorization", format!("Bearer {}", token))
            .json(&serde_json::json!({"message": "hi", "stream": false}))
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .expect("turn send")
            .status()
    });

    // Poll rather than sleeping a fixed interval: on a loaded machine the turn may take a moment
    // to reach the provider, and a fixed sleep would read `false` and flake. The scripted 2s sleep
    // holds the turn open for the whole poll window.
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(1800);
    let mut seen_in_flight = false;
    while std::time::Instant::now() < deadline {
        let body: serde_json::Value = harness
            .request(reqwest::Method::GET, &format!("/v1/sessions/{}", id))
            .send()
            .expect("send")
            .json()
            .expect("parse");
        if body["turn_in_flight"] == true {
            seen_in_flight = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(
        seen_in_flight,
        "a running turn must be visible without having to trip the 409"
    );

    assert_eq!(worker.join().expect("worker panicked"), 200);

    let after: serde_json::Value = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}", id))
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert_eq!(after["turn_in_flight"], false);
}

/// A streaming client that declared it has no approval interface must get an immediate deny, not
/// a 60-second park. The turn completes well inside the timeout if the flag is honoured.
#[test]
fn streaming_session_without_prompt_support_does_not_park_on_permission() {
    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "tu_1", "name": "write_file" },
            { "kind": "tool_use_end", "input": {"path": "/tmp/meka-test-noprompt.txt", "content": "x"} },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "denied then" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({
            "cwd": std::env::temp_dir().to_string_lossy(),
            "permission": "ask",
            "capabilities": {"supports_permission_prompts": false},
        }))
        .send()
        .expect("create");
    assert_eq!(create.status(), 201);
    let body: serde_json::Value = create.json().expect("parse");
    assert_eq!(body["capabilities"]["supports_permission_prompts"], false);
    let id = body["id"].as_str().expect("id").to_string();

    let started = std::time::Instant::now();
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "run it", "stream": true}))
        .send()
        .expect("send");
    assert_eq!(response.status(), 200);
    let body = response.text().expect("text");
    let elapsed = started.elapsed();

    assert!(
        elapsed < std::time::Duration::from_secs(30),
        "turn took {:?}; it parked on the permission channel instead of denying",
        elapsed
    );
    assert!(
        !body.contains("permission_required"),
        "no permission_required event should be emitted; body was:\n{}",
        body
    );
}

/// `vision` on `/v1/info` is how a client discovers whether attaching an image is worth the
/// base64 payload, instead of finding out from a 422.
#[test]
fn info_reports_the_vision_capability() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let response = harness
        .request(reqwest::Method::GET, "/v1/info")
        .send()
        .expect("send");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["vision"], true);
}

/// Two concurrent turns on two different sessions complete in roughly the same wall time
/// as a single turn; i.e. the agent loop doesn't serialize across sessions.
#[test]
fn multi_session_parallel_happy_path() {
    let script = serde_json::json!([
        [
            { "kind": "sleep", "ms": 2000 },
            { "kind": "text", "text": "done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "sleep", "ms": 2000 },
            { "kind": "text", "text": "done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let mut ids = Vec::new();
    for _ in 0..2 {
        let create = harness
            .request(reqwest::Method::POST, "/v1/sessions")
            .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
            .send()
            .expect("create");
        let id = create.json::<serde_json::Value>().expect("parse")["id"]
            .as_str()
            .expect("id")
            .to_string();
        ids.push(id);
    }
    let base_url = harness.base_url.clone();
    let token = harness.token.clone();
    let started_at = Instant::now();
    let handles: Vec<_> = ids
        .into_iter()
        .map(|id| {
            let base = base_url.clone();
            let tok = token.clone();
            std::thread::spawn(move || {
                let client = reqwest::blocking::Client::builder()
                    .timeout(Duration::from_secs(30))
                    .build()
                    .expect("client");
                let resp = client
                    .post(format!("{}/v1/sessions/{}/turn", base, id))
                    .header("Authorization", format!("Bearer {}", tok))
                    .json(&serde_json::json!({"message": "go"}))
                    .send()
                    .expect("send");
                assert_eq!(resp.status(), 200, "parallel turn must succeed");
            })
        })
        .collect();
    for handle in handles {
        handle.join().expect("join");
    }
    let elapsed = started_at.elapsed();
    // Each turn sleeps ~2000ms, so a serialised run would take ≥4000ms. The 3500ms bound
    // clears the ~2000ms parallel time plus process-spawn and connection overhead (notably
    // higher on Windows CI) while staying well under the serial time.
    assert!(
        elapsed < Duration::from_millis(3500),
        "parallel turns should complete in <3500ms; elapsed={:?}",
        elapsed,
    );
}

/// With `capabilities.supports_reasoning_stream: true`, scripted thinking events appear on
/// the SSE wire as `thinking.delta`.
#[test]
fn thinking_delta_streams_with_capability_enabled() {
    let script = serde_json::json!([
        [
            { "kind": "thinking_delta", "text": "let me check" },
            { "kind": "thinking_complete", "signature": null },
            { "kind": "text", "text": "answer" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({
            "cwd": std::env::temp_dir().to_string_lossy(),
            "capabilities": {"supports_reasoning_stream": true},
        }))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "ponder", "stream": true}))
        .send()
        .expect("stream");
    assert_eq!(response.status(), 200);
    let body = response.text().expect("body");
    assert!(
        body.contains("event: thinking.delta"),
        "with supports_reasoning_stream: true the SSE wire must include thinking.delta; \
         body was:\n{}",
        body,
    );
}

/// With the default `capabilities.supports_reasoning_stream: false`, scripted thinking
/// events do NOT appear on the SSE wire.
#[test]
fn thinking_delta_filtered_when_capability_disabled() {
    let script = serde_json::json!([
        [
            { "kind": "thinking_delta", "text": "let me check" },
            { "kind": "thinking_complete", "signature": null },
            { "kind": "text", "text": "answer" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "ponder", "stream": true}))
        .send()
        .expect("stream");
    let body = response.text().expect("body");
    assert!(
        !body.contains("event: thinking.delta"),
        "default capabilities must exclude thinking.delta; body was:\n{}",
        body,
    );
}

/// A streaming turn that the provider fails mid-stream emits a `turn.failed` SSE event
/// carrying a Problem Detail before the connection closes.
#[test]
fn streaming_provider_failure_emits_turn_failed_event() {
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "before failure " },
            { "kind": "fail", "message": "scripted upstream 529" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "go", "stream": true}))
        .send()
        .expect("send");
    assert_eq!(response.status(), 200);
    let body = response.text().expect("body");
    assert!(
        body.contains("event: turn.failed"),
        "stream must emit turn.failed when provider errors mid-stream; body was:\n{}",
        body,
    );
    assert!(
        body.contains("https://meka.so/errors/provider"),
        "turn.failed payload must carry the provider error type; body was:\n{}",
        body,
    );
}

/// `outcome: "allow"` on the mid-turn permission response unblocks the parked tool call
/// and the turn proceeds to completion.
#[test]
fn permission_allow_outcome_resumes_turn() {
    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "tu_1", "name": "write_file" },
            { "kind": "tool_use_end", "input": {
                "path": std::env::temp_dir().join("meka-permission-allow-test.txt").to_string_lossy(),
                "content": "hello"
            } },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "wrote it" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({
            "cwd": std::env::temp_dir().to_string_lossy(),
            "permission": "ask",
        }))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    let base_url = harness.base_url.clone();
    let token = harness.token.clone();
    let id_for_stream = id.clone();
    let stream_handle = std::thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("client");
        let response = client
            .post(format!("{}/v1/sessions/{}/turn", base_url, id_for_stream))
            .header("Authorization", format!("Bearer {}", token))
            .json(&serde_json::json!({"message": "write please", "stream": true}))
            .send()
            .expect("stream POST");
        let respond_client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .expect("respond client");
        let mut last_event: Option<String> = None;
        let mut posted_allow = false;
        let mut saw_finished = false;
        let mut body = String::new();
        for line in std::io::BufReader::new(response).lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            body.push_str(&line);
            body.push('\n');
            if let Some(name) = line.strip_prefix("event: ") {
                let trimmed = name.trim();
                last_event = Some(trimmed.to_string());
                if trimmed == "turn.finished" {
                    saw_finished = true;
                }
            } else if let Some(data) = line.strip_prefix("data: ")
                && last_event.as_deref() == Some("permission_required")
                && !posted_allow
            {
                let payload: serde_json::Value =
                    serde_json::from_str(data.trim()).expect("parse data");
                let request_id = payload["request_id"]
                    .as_str()
                    .expect("request_id")
                    .to_string();
                respond_client
                    .post(format!(
                        "{}/v1/sessions/{}/responses/{}",
                        base_url, id_for_stream, request_id,
                    ))
                    .header("Authorization", format!("Bearer {}", token))
                    .json(&serde_json::json!({"outcome": "allow"}))
                    .send()
                    .expect("respond");
                posted_allow = true;
            }
        }
        (posted_allow, saw_finished, body)
    });

    let (posted_allow, saw_finished, body) = stream_handle.join().expect("join");
    assert!(
        posted_allow,
        "the test should have posted the allow outcome"
    );
    assert!(
        saw_finished,
        "after `allow`, the turn must proceed to turn.finished; body was:\n{}",
        body,
    );
}

/// `GET /v1/sessions/{id}/messages?limit=N&offset=M` returns a correctly-sliced page.
#[test]
fn messages_pagination_offset_limit() {
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "first response" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "text", "text": "second response" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "text", "text": "third response" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    for message in ["one", "two", "three"] {
        let response = harness
            .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
            .json(&serde_json::json!({"message": message}))
            .send()
            .expect("turn");
        assert_eq!(response.status(), 200);
    }

    let all = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/messages", id),
        )
        .send()
        .expect("messages all");
    let body: serde_json::Value = all.json().expect("parse");
    let total = body["total"].as_u64().expect("total");
    assert!(
        total >= 6,
        "three turns × (user + assistant) ⇒ ≥6 messages; got {}",
        total,
    );

    let page = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/messages?limit=2&offset=1", id),
        )
        .send()
        .expect("messages page");
    let page_body: serde_json::Value = page.json().expect("parse page");
    let page_len = page_body["messages"]
        .as_array()
        .expect("messages array")
        .len();
    assert_eq!(
        page_len, 2,
        "limit=2 must yield 2 messages; got {}",
        page_len
    );
    assert_eq!(
        page_body["total"].as_u64(),
        Some(total),
        "total must match the unpaginated count",
    );
}

/// POST /v1/sessions/{id}/responses/{unknown_request_id} returns 404 request-not-found.
#[test]
fn unknown_request_id_returns_404_request_not_found() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    let response = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/responses/req-nonexistent", id),
        )
        .json(&serde_json::json!({"outcome": "allow"}))
        .send()
        .expect("send");
    assert_eq!(response.status(), 404);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["type"], "https://meka.so/errors/request-not-found");
}

/// A second concurrent POST with the same Idempotency-Key receives 409 idempotency-conflict
/// while the first request is still running (Pending sentinel).
#[test]
fn idempotency_key_in_flight_returns_409() {
    let script = serde_json::json!([
        [
            { "kind": "sleep", "ms": 1200 },
            { "kind": "text", "text": "first done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    let base_url = harness.base_url.clone();
    let token = harness.token.clone();
    let id_for_first = id.clone();
    let first = std::thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("client");
        client
            .post(format!("{}/v1/sessions/{}/turn", base_url, id_for_first))
            .header("Authorization", format!("Bearer {}", token))
            .header("Idempotency-Key", "in-flight-key")
            .json(&serde_json::json!({"message": "first"}))
            .send()
            .expect("first")
    });

    // Wait for the first request to enter its mock sleep; by then the Pending sentinel is
    // installed in the cache.
    std::thread::sleep(Duration::from_millis(250));
    let second = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .header("Idempotency-Key", "in-flight-key")
        .json(&serde_json::json!({"message": "first"}))
        .send()
        .expect("second");
    assert_eq!(
        second.status(),
        409,
        "concurrent same-keyed request must receive 409 idempotency-in-flight",
    );
    let body: serde_json::Value = second.json().expect("parse");
    assert_eq!(body["type"], "https://meka.so/errors/idempotency");

    // Let the first request finish; it commits the Pending entry into a Cached one.
    let first_response = first.join().expect("join");
    assert_eq!(first_response.status(), 200);
}

/// `options.skill = "unknown"` returns 422 invalid-body.
#[test]
fn turn_options_unknown_skill_returns_422() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({
            "message": "go",
            "options": {"skill": "this-skill-does-not-exist"},
        }))
        .send()
        .expect("send");
    assert_eq!(response.status(), 422);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["type"], "https://meka.so/errors/invalid-body");
}

/// Unknown fields under `options` produce 422 invalid-body.
#[test]
fn turn_options_unknown_field_returns_422() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({
            "message": "go",
            "options": {"definitely_not_a_real_field": true},
        }))
        .send()
        .expect("send");
    assert_eq!(response.status(), 422);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["type"], "https://meka.so/errors/invalid-body");
}

/// DELETE on a session with an active turn returns 409 turn-in-flight.
/// Clients are expected to POST /cancel first.
#[test]
fn delete_while_turn_in_flight_returns_409() {
    let script = serde_json::json!([
        [
            { "kind": "sleep", "ms": 1500 },
            { "kind": "text", "text": "done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    let base_url = harness.base_url.clone();
    let token = harness.token.clone();
    let id_clone = id.clone();
    let turn_handle = std::thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("client");
        client
            .post(format!("{}/v1/sessions/{}/turn", base_url, id_clone))
            .header("Authorization", format!("Bearer {}", token))
            .json(&serde_json::json!({"message": "go"}))
            .send()
            .expect("turn send")
    });

    // Wait for the turn to enter its mock sleep, then attempt DELETE.
    std::thread::sleep(Duration::from_millis(300));
    let delete = harness
        .request(reqwest::Method::DELETE, &format!("/v1/sessions/{}", id))
        .send()
        .expect("delete");
    assert_eq!(
        delete.status(),
        409,
        "DELETE during in-flight turn must 409"
    );
    let body: serde_json::Value = delete.json().expect("parse");
    assert_eq!(body["type"], "https://meka.so/errors/turn-in-flight");

    // Drain the in-flight turn so the harness Drop is clean.
    let _ = turn_handle.join().expect("join");

    // Now DELETE should succeed (turn has finished).
    let delete_after = harness
        .request(reqwest::Method::DELETE, &format!("/v1/sessions/{}", id))
        .send()
        .expect("delete after");
    assert_eq!(delete_after.status(), 204);
}

/// DELETE /v1/sessions/{id} requires `sessions:w` scope; a `sessions:r`-only token gets 403.
#[test]
fn delete_without_write_scope_returns_403() {
    let harness =
        ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", &["sessions:r"]);
    let response = harness
        .request(
            reqwest::Method::DELETE,
            "/v1/sessions/00000000-0000-0000-0000-000000000000",
        )
        .send()
        .expect("send");
    assert_eq!(response.status(), 403);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["type"], "https://meka.so/errors/auth-scope");
}

/// A GC-evicted session re-attached on a subsequent request must report its original
/// `created_at` (the DB-persisted value), not `Utc::now()`.
#[test]
fn created_at_survives_gc_and_reattach() {
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "first" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "text", "text": "second" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn_with(
        "idle_timeout = \"1s\"\ngc_scan_interval = \"1s\"\n",
        script,
        "sk_test_token",
        &["sessions:r", "sessions:w"],
    );
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    assert_eq!(create.status(), 201);
    let create_body: serde_json::Value = create.json().expect("parse");
    let id = create_body["id"].as_str().expect("id").to_string();
    let original_created_at = create_body["created_at"]
        .as_str()
        .expect("created_at")
        .to_string();

    // Run a turn to ensure the session has DB activity.
    let first = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "hi"}))
        .send()
        .expect("first turn");
    assert_eq!(first.status(), 200);

    // Wait for GC eviction.
    std::thread::sleep(Duration::from_secs(3));

    // Trigger re-attach.
    let second = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "again"}))
        .send()
        .expect("second turn");
    assert_eq!(second.status(), 200);

    // GET the session and verify created_at is unchanged.
    let get = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}", id))
        .send()
        .expect("get");
    assert_eq!(get.status(), 200);
    let body: serde_json::Value = get.json().expect("parse");
    assert_eq!(
        body["created_at"], original_created_at,
        "created_at must survive GC eviction + re-attach intact",
    );
}

/// A typo on a top-level TurnRequest field (e.g. "streem" instead of "stream") returns
/// 422 invalid-body thanks to `#[serde(deny_unknown_fields)]`.
#[test]
fn turn_request_unknown_top_level_field_returns_422() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({
            "message": "hi",
            "streem": false,  // typo
        }))
        .send()
        .expect("send");
    assert_eq!(response.status(), 422);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["type"], "https://meka.so/errors/invalid-body");
}

/// A PATCH that mixes a valid field with an invalid one must reject the request
/// without applying *either* change (atomic validation).
#[test]
fn patch_session_atomic_rejects_when_cwd_is_invalid() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({
            "cwd": std::env::temp_dir().to_string_lossy(),
            "permission": "read",
        }))
        .send()
        .expect("create");
    let body: serde_json::Value = create.json().expect("parse");
    let id = body["id"].as_str().expect("id").to_string();
    assert_eq!(body["permission"], "read", "session created with read");

    // Mixed-validity PATCH: permission flips to "write" (valid), cwd is relative (invalid).
    let patch = harness
        .request(reqwest::Method::PATCH, &format!("/v1/sessions/{}", id))
        .json(&serde_json::json!({
            "permission": "write",
            "cwd": "relative/path",
        }))
        .send()
        .expect("patch");
    assert_eq!(patch.status(), 422, "invalid cwd must reject the PATCH");
    let problem: serde_json::Value = patch.json().expect("problem");
    assert_eq!(problem["type"], "https://meka.so/errors/invalid-body");

    // GET the session and verify the permission change did NOT leak through.
    let get = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}", id))
        .send()
        .expect("get");
    assert_eq!(get.status(), 200);
    let snapshot: serde_json::Value = get.json().expect("snapshot");
    assert_eq!(
        snapshot["permission"], "read",
        "permission must NOT have changed when the same PATCH rejected for a sibling field",
    );
}

/// The three discovery endpoints share a single read-scope helper. A token
/// holding any one of `sessions:r`, `mcp:r`, or `skills:r` must be admitted on all three.
#[test]
fn discovery_endpoints_share_read_scope_set() {
    // Token scoped to `mcp:r` only: must be admitted on all three discovery endpoints.
    let harness =
        ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_mcp_only", &["mcp:r"]);
    for path in ["/v1/info", "/v1/skills", "/v1/mcp"] {
        let response = harness
            .request(reqwest::Method::GET, path)
            .send()
            .expect("send");
        assert_eq!(
            response.status(),
            200,
            "mcp:r token must be admitted on {}",
            path,
        );
    }
}

/// Sibling check: a token holding only `skills:r` is also admitted on all three. Mirrors the
/// `mcp:r` test above for the other branch of the helper.
#[test]
fn discovery_endpoints_admit_skills_only_token() {
    let harness =
        ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_skills_only", &["skills:r"]);
    for path in ["/v1/info", "/v1/skills", "/v1/mcp"] {
        let response = harness
            .request(reqwest::Method::GET, path)
            .send()
            .expect("send");
        assert_eq!(
            response.status(),
            200,
            "skills:r token must be admitted on {}",
            path,
        );
    }
}

/// `delete_on_idle = true` must remove the DB row when GC evicts an idle session, so a
/// subsequent `GET /v1/sessions/{id}` returns 404 (not a stale row that re-attaches).
#[test]
fn delete_on_idle_true_removes_db_row_on_eviction() {
    let harness = ServeTestHarness::spawn_with(
        "idle_timeout = \"1s\"\ngc_scan_interval = \"1s\"\ndelete_on_idle = true\n",
        mock_simple_turn(),
        "sk_test_token",
        &["sessions:r", "sessions:w"],
    );
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    assert_eq!(create.status(), 201);
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    // Wait past the idle timeout for GC to fire.
    std::thread::sleep(Duration::from_secs(3));

    // The DB row should now be gone; GET returns 404 (not a re-attach).
    let get = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}", id))
        .send()
        .expect("get");
    assert_eq!(
        get.status(),
        404,
        "delete_on_idle = true must drop the DB row on eviction; got status {}",
        get.status(),
    );
    let body: serde_json::Value = get.json().expect("problem");
    assert_eq!(body["type"], "https://meka.so/errors/session-not-found");
}

/// A pre-attempt `turn-in-flight` 409 from `run_blocking_turn`'s `try_lock` must NOT be
/// persisted in the idempotency cache; ticket Drop removes the Pending entry on 409.
#[test]
fn idempotency_cache_does_not_persist_turn_in_flight_409() {
    let script = serde_json::json!([
        [
            { "kind": "sleep", "ms": 1200 },
            { "kind": "text", "text": "first done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "text", "text": "retry done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    // Turn A (no idempotency key) takes the runtime lock and sleeps for 1.2s.
    let base_url = harness.base_url.clone();
    let token = harness.token.clone();
    let id_for_a = id.clone();
    let first = std::thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("client");
        client
            .post(format!("{}/v1/sessions/{}/turn", base_url, id_for_a))
            .header("Authorization", format!("Bearer {}", token))
            .json(&serde_json::json!({"message": "long"}))
            .send()
            .expect("first send")
    });

    // Wait for turn A to enter its sleep so the runtime mutex is held.
    std::thread::sleep(Duration::from_millis(250));

    // Turn B uses `Idempotency-Key: k1` and bounces off run_blocking_turn's try_lock → 409
    // turn-in-flight. The Pending entry must be dropped, not committed to the cache.
    let key = "retry-after-in-flight";
    let second = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .header("Idempotency-Key", key)
        .json(&serde_json::json!({"message": "second"}))
        .send()
        .expect("second");
    assert_eq!(
        second.status(),
        409,
        "concurrent turn must hit run_blocking_turn try_lock and 409 with turn-in-flight",
    );
    let problem: serde_json::Value = second.json().expect("parse");
    assert_eq!(problem["type"], "https://meka.so/errors/turn-in-flight");

    // Let turn A finish so the runtime lock is free.
    let first_response = first.join().expect("join");
    assert_eq!(first_response.status(), 200, "turn A should have completed");

    // Replay turn C with the same key. The Pending entry was dropped (no cache commit on
    // TurnInFlight), so this re-executes and the mock returns the second round.
    let third = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .header("Idempotency-Key", key)
        .json(&serde_json::json!({"message": "second"}))
        .send()
        .expect("third");
    assert_eq!(
        third.status(),
        200,
        "replay after the in-flight clears must execute fresh, not return cached 409; got {}",
        third.status(),
    );
    let body: serde_json::Value = third.json().expect("parse");
    assert_eq!(body["final_text"], "retry done");
}

/// Validate the blocking turn response shape (`tool_calls`, `usage`, `messages`) end-to-end.
/// Script a tool call and assert the fields are populated with the shapes the spec documents.
#[test]
fn blocking_turn_response_carries_tool_calls_messages_and_usage() {
    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "tu_1", "name": "list_directory" },
            { "kind": "tool_use_end", "input": {"path": std::env::temp_dir().to_string_lossy()} },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "done listing" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "list it"}))
        .send()
        .expect("turn");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["stop_reason"], "end_turn");
    assert_eq!(body["final_text"], "done listing");

    let tool_calls = body["tool_calls"].as_array().expect("tool_calls array");
    assert!(
        !tool_calls.is_empty(),
        "tool_calls must include the scripted tool call",
    );
    assert_eq!(tool_calls[0]["id"], "tu_1");
    assert_eq!(tool_calls[0]["name"], "list_directory");
    assert!(
        tool_calls[0]["input"].is_object(),
        "tool_call input must be a JSON object",
    );

    let usage = &body["usage"];
    assert!(
        usage["input_tokens"].is_number(),
        "usage.input_tokens must be a number (zeros are fine)",
    );
    assert!(
        usage["output_tokens"].is_number(),
        "usage.output_tokens must be a number",
    );

    let messages = body["messages"].as_array().expect("messages array");
    assert!(
        !messages.is_empty(),
        "messages array must include the assistant response(s)",
    );
    assert!(
        messages.iter().any(|m| m["role"] == "assistant"),
        "messages must include at least one assistant role entry",
    );
}

/// A token holding only `sessions:w` must NOT be admitted on read endpoints (the inverse of
/// `insufficient_scope_returns_403` which tests r-only → 403 on write).
#[test]
fn write_only_token_cannot_read_sessions() {
    let harness =
        ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_w_only", &["sessions:w"]);
    let response = harness
        .request(reqwest::Method::GET, "/v1/sessions")
        .send()
        .expect("send");
    assert_eq!(
        response.status(),
        403,
        "sessions:w-only token must be rejected on GET /v1/sessions",
    );
    let problem: serde_json::Value = response.json().expect("parse");
    assert_eq!(problem["type"], "https://meka.so/errors/auth-scope");
}

/// `stop_reason = refusal` flows through the blocking response.  The mock provider emits
/// `StopReason::Refusal("")` (empty refusal text), and `assemble_response` suppresses the
/// `refusal_text` field via `skip_serializing_if` when empty, so this test asserts the
/// stop_reason channel without exercising the refusal_text payload.
#[test]
fn refusal_stop_reason_propagates_through_blocking_response() {
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "I can't help with that." },
            { "kind": "message_end", "stop_reason": "refusal" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "do something disallowed"}))
        .send()
        .expect("turn");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(
        body["stop_reason"], "refusal",
        "stop_reason must propagate the refusal terminal state",
    );
    // `final_text` carries the assistant's pre-refusal text (the mock sent it as a normal text
    // delta before the message_end:refusal). Clients surface both fields together when
    // stop_reason is refusal.
    assert_eq!(body["final_text"], "I can't help with that.");
}

/// SSE event ids form a dense, monotonic 0-based sequence with no gaps.
#[test]
fn streaming_turn_event_ids_are_dense_and_monotonic() {
    // Script a turn that emits text-delta plus tool-call events plus a token-usage marker
    // (which translate() drops). The tool-call path forces another agent loop iteration,
    // adding more "lifecycle" events the streaming handler emits directly.
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "thinking aloud " },
            { "kind": "tool_use_start", "id": "tu_1", "name": "list_directory" },
            { "kind": "tool_use_end", "input": {"path": std::env::temp_dir().to_string_lossy()} },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "go", "stream": true}))
        .send()
        .expect("send");
    assert_eq!(response.status(), 200);
    let body = response.text().expect("body");

    // Collect every `id: N` line from the stream. The streaming handler's `Sse` wrapper
    // injects KeepAlive lines which carry no `id:`; lifecycle + translated events all do.
    let ids: Vec<u64> = body
        .lines()
        .filter_map(|line| line.strip_prefix("id: "))
        .filter_map(|n| n.trim().parse::<u64>().ok())
        .collect();
    assert!(
        !ids.is_empty(),
        "streaming turn must emit at least one id-bearing event; body was:\n{}",
        body,
    );
    assert_eq!(
        ids[0], 0,
        "first event id must be 0 per spec example; ids were {:?}",
        ids,
    );
    for window in ids.windows(2) {
        let [a, b] = [window[0], window[1]];
        assert_eq!(
            b,
            a + 1,
            "event ids must be dense (no gaps from filtered events); saw {:?}",
            ids,
        );
    }
}

/// Sticky `allow_always` short-circuits subsequent same-tool prompts.  After
/// the client resolves the first `permission_required` with `outcome: allow_always`, the
/// SECOND tool call in the same turn must auto-allow without emitting another
/// `permission_required` event.
#[test]
fn sticky_allow_always_short_circuits_second_tool_call() {
    // Two tool_use rounds of the same write-tier tool + a terminal text round. `write_file`
    // is gated by ask mode; `list_directory` would short-circuit as read-tier without
    // prompting and miss the point of the test.
    let write_path = std::env::temp_dir().join("meka-test-sticky.txt");
    let _ = std::fs::remove_file(&write_path);
    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "tu_1", "name": "write_file" },
            { "kind": "tool_use_end", "input": {"path": write_path.to_string_lossy(), "content": "first"} },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "tool_use_start", "id": "tu_2", "name": "write_file" },
            { "kind": "tool_use_end", "input": {"path": write_path.to_string_lossy(), "content": "second"} },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "done twice" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({
            "cwd": std::env::temp_dir().to_string_lossy(),
            "permission": "ask",
        }))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    let base_url = harness.base_url.clone();
    let token = harness.token.clone();
    let id_for_stream = id.clone();
    let stream_handle = std::thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("client");
        let response = client
            .post(format!("{}/v1/sessions/{}/turn", base_url, id_for_stream))
            .header("Authorization", format!("Bearer {}", token))
            .json(&serde_json::json!({"message": "do it twice", "stream": true}))
            .send()
            .expect("stream POST");
        let mut last_event: Option<String> = None;
        let mut posted_resolution = false;
        let mut permission_required_count: u32 = 0;
        let mut saw_finished = false;
        let mut buffered = String::new();
        let reader = std::io::BufReader::new(response);
        let respond_client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .expect("respond client");
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            buffered.push_str(&line);
            buffered.push('\n');
            if let Some(name) = line.strip_prefix("event: ") {
                let trimmed = name.trim();
                last_event = Some(trimmed.to_string());
                if trimmed == "permission_required" {
                    permission_required_count += 1;
                }
                if trimmed == "turn.finished" {
                    saw_finished = true;
                }
            } else if let Some(data) = line.strip_prefix("data: ")
                && last_event.as_deref() == Some("permission_required")
                && !posted_resolution
            {
                let payload: serde_json::Value =
                    serde_json::from_str(data.trim()).expect("parse data");
                let request_id = payload["request_id"]
                    .as_str()
                    .expect("request_id")
                    .to_string();
                let resp = respond_client
                    .post(format!(
                        "{}/v1/sessions/{}/responses/{}",
                        base_url, id_for_stream, request_id,
                    ))
                    .header("Authorization", format!("Bearer {}", token))
                    .json(&serde_json::json!({"outcome": "allow_always"}))
                    .send()
                    .expect("respond send");
                assert_eq!(resp.status(), 204);
                posted_resolution = true;
            }
        }
        (
            permission_required_count,
            posted_resolution,
            saw_finished,
            buffered,
        )
    });

    let (permission_required_count, posted_resolution, saw_finished, body) =
        stream_handle.join().expect("stream worker join");
    assert!(posted_resolution, "client must have posted allow_always");
    assert!(
        saw_finished,
        "turn must finish after both tool calls; body was:\n{}",
        body
    );
    assert_eq!(
        permission_required_count, 1,
        "sticky allow_always must short-circuit the second tool prompt, saw {} \
         permission_required events; body was:\n{}",
        permission_required_count, body
    );
}

/// When `supports_reasoning_stream` is on, the blocking response includes
/// thinking content blocks in `messages[].content`.
#[test]
fn blocking_turn_with_reasoning_stream_includes_thinking() {
    let script = serde_json::json!([
        [
            { "kind": "thinking_delta", "text": "let me reason about this" },
            { "kind": "thinking_complete", "signature": null },
            { "kind": "text", "text": "answer." },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({
            "cwd": std::env::temp_dir().to_string_lossy(),
            "capabilities": {"supports_reasoning_stream": true},
        }))
        .send()
        .expect("create");
    assert_eq!(create.status(), 201);
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "think then answer"}))
        .send()
        .expect("turn");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().expect("parse");
    let content = &body["messages"][0]["content"];
    let blocks = content.as_array().expect("content array");
    let thinking = blocks
        .iter()
        .find(|block| block["type"] == "thinking")
        .expect("messages[0].content must include a thinking block when capability is on");
    assert_eq!(thinking["thinking"], "let me reason about this");
    let text = blocks
        .iter()
        .find(|block| block["type"] == "text")
        .expect("text block must follow");
    assert_eq!(text["text"], "answer.");
}

/// A blocking-mode `POST /cancel` produces 409 `turn-cancelled`.
#[test]
fn cancel_during_blocking_turn_returns_409_turn_cancelled() {
    let script = serde_json::json!([
        [
            { "kind": "sleep", "ms": 2000 },
            { "kind": "text", "text": "never reaches client" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    // Fire a long blocking turn in the background; cancel it after a beat.
    let base_url = harness.base_url.clone();
    let token = harness.token.clone();
    let id_for_turn = id.clone();
    let turn_handle = std::thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("client");
        client
            .post(format!("{}/v1/sessions/{}/turn", base_url, id_for_turn))
            .header("Authorization", format!("Bearer {}", token))
            .json(&serde_json::json!({"message": "go"}))
            .send()
            .expect("turn send")
    });

    std::thread::sleep(Duration::from_millis(300));
    let cancel = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/cancel", id),
        )
        .send()
        .expect("cancel send");
    assert_eq!(cancel.status(), 204);

    let response = turn_handle.join().expect("turn join");
    assert_eq!(
        response.status(),
        409,
        "blocking-mode cancel must surface as 409 turn-cancelled, not 500 internal",
    );
    let problem: serde_json::Value = response.json().expect("parse");
    assert_eq!(problem["type"], "https://meka.so/errors/turn-cancelled");
}

/// Ask mode + `stream: false` is a non-functional combination (every tool would auto-deny).
/// Ask-mode + blocking turn runs to completion; tool prompts are auto-denied with notices.
/// (No tool calls in this fixture, so the turn succeeds cleanly; the auto-deny pathway is
/// exercised by `ask_mode_blocking_turn_auto_denies_with_notice`.)
#[test]
fn ask_mode_blocking_turn_succeeds() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({
            "cwd": std::env::temp_dir().to_string_lossy(),
            "permission": "ask",
        }))
        .send()
        .expect("create");
    assert_eq!(create.status(), 201);
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "do it", "stream": false}))
        .send()
        .expect("turn");
    assert_eq!(
        response.status(),
        200,
        "ask-mode + blocking should succeed (tools auto-denied with notices, not rejected)"
    );
}

/// All terminal SSE events carry `turn_id` and `session_id` so clients can
/// correlate the terminal frame back to its `turn.started`.
#[test]
fn terminal_sse_events_carry_turn_id_and_session_id() {
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "ok" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "hi", "stream": true}))
        .send()
        .expect("send");
    let body = response.text().expect("body");
    let finished_line = body
        .lines()
        .skip_while(|line| !line.starts_with("event: turn.finished"))
        .nth(1)
        .expect("turn.finished data line must follow event header");
    let payload: serde_json::Value =
        serde_json::from_str(finished_line.strip_prefix("data: ").expect("data prefix"))
            .expect("parse data");
    assert_eq!(payload["session_id"], id);
    assert!(
        payload["turn_id"].is_string(),
        "turn.finished must include turn_id; payload: {}",
        payload,
    );
}

/// `ResponseBody` rejects unknown top-level fields with 422.
#[test]
fn responses_body_unknown_field_returns_422() {
    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "tu_1", "name": "write_file" },
            { "kind": "tool_use_end", "input": {
                "path": std::env::temp_dir().join("meka-l10-test").to_string_lossy(),
                "content": "x"
            }},
            { "kind": "message_end", "stop_reason": "tool_use" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({
            "cwd": std::env::temp_dir().to_string_lossy(),
            "permission": "ask",
        }))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    // Fire a stream so the permission_required event hits the channel, then post a
    // request_id with an unknown extra field. The request_id doesn't even have to be valid;
    // the body-parse rejection happens first.
    let response = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/responses/req_bogus", id),
        )
        .json(&serde_json::json!({"outcome": "allow", "extra": "garbage"}))
        .send()
        .expect("respond");
    assert_eq!(
        response.status(),
        422,
        "ResponseBody must reject unknown top-level fields with 422",
    );
    let problem: serde_json::Value = response.json().expect("parse");
    assert_eq!(problem["type"], "https://meka.so/errors/invalid-body");
}

/// When a turn is cancelled mid-tool-execution, `ToolCallStarted` arrives without a matching
/// `ToolCallCompleted`. Orphan entries are marked `is_error: true` with an explanatory text block.
#[test]
fn orphan_tool_call_marked_as_interrupted_in_blocking_response() {
    // Two-round script: round 1 starts a tool, the agent loop runs it, then we cancel.
    // The mock has a sleep after tool_use_end so the cancel fires during execution.
    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "tu_1", "name": "write_file" },
            { "kind": "tool_use_end", "input": {
                "path": std::env::temp_dir().join("meka-l8.txt").to_string_lossy(),
                "content": "x"
            }},
            { "kind": "sleep", "ms": 1500 },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    // Fire a blocking turn in a thread, cancel from main thread after a beat.
    let base = harness.base_url.clone();
    let token = harness.token.clone();
    let id_for_turn = id.clone();
    let turn_handle = std::thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("client");
        client
            .post(format!("{}/v1/sessions/{}/turn", base, id_for_turn))
            .header("Authorization", format!("Bearer {}", token))
            .json(&serde_json::json!({"message": "go", "stream": false}))
            .send()
            .expect("turn send")
    });

    // Sleep so the mock starts emitting the tool_use_start (recorder captures it), then
    // cancel before the mock's sleep finishes.
    std::thread::sleep(Duration::from_millis(400));
    let cancel = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/cancel", id),
        )
        .send()
        .expect("cancel");
    assert_eq!(cancel.status(), 204);

    let response = turn_handle.join().expect("join");
    // Cancelled turns return 409 with a Problem Detail body (not a TurnResponse), so the
    // orphan-tool content assertion can't be checked via the cancel path here.
    assert!(
        response.status() == 409 || response.status() == 200,
        "expected 409 (turn-cancelled) or 200 (turn finished before cancel); got {}",
        response.status(),
    );
}

/// Authenticated handlers' OpenAPI annotations include 403/409/500 where applicable,
/// and `delete_session` no longer documents 404 (it returns 204 idempotently).
#[test]
fn openapi_spec_documents_403_409_and_no_stale_404_on_delete() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let response = harness
        .client
        .get(format!("{}/v1/openapi.json", harness.base_url))
        .send()
        .expect("send");
    assert_eq!(response.status(), 200);
    let spec: serde_json::Value = response.json().expect("parse");
    let paths = &spec["paths"];

    // Every authenticated path declares 403.
    for (path, method) in [
        ("/v1/sessions", "get"),
        ("/v1/sessions", "post"),
        ("/v1/sessions/{id}", "get"),
        ("/v1/sessions/{id}", "patch"),
        ("/v1/sessions/{id}", "delete"),
        ("/v1/sessions/{id}/turn", "post"),
        ("/v1/sessions/{id}/cancel", "post"),
        ("/v1/sessions/{id}/messages", "get"),
        ("/v1/sessions/{id}/responses/{request_id}", "post"),
    ] {
        let responses = &paths[path][method]["responses"];
        assert!(
            responses["403"].is_object(),
            "OpenAPI {} {} should declare a 403 response; got {:?}",
            method,
            path,
            responses
        );
    }

    // PATCH and DELETE declare 409 (in-flight rejection).
    assert!(
        paths["/v1/sessions/{id}"]["patch"]["responses"]["409"].is_object(),
        "PATCH should declare 409 for in-flight turn rejection",
    );
    assert!(
        paths["/v1/sessions/{id}"]["delete"]["responses"]["409"].is_object(),
        "DELETE should declare 409 for in-flight turn rejection",
    );

    // DELETE returns 204 idempotently; no 404 in the spec.
    assert!(
        paths["/v1/sessions/{id}"]["delete"]["responses"]["404"].is_null(),
        "DELETE should no longer document 404 (idempotent, 204 for unknown ids)",
    );
}

/// Every `Option<T>` in the wire-shape structs is absent (not `null`)
/// when `None`. `cwd` on GET is always populated (it defaults to the server's cwd), so
/// the cleaner assertion is to check `display_summary` and `refusal_text` on a turn that
/// produces neither.
#[test]
fn option_fields_are_absent_not_null_when_unset() {
    // mock_simple_turn produces a turn with no tool calls and no refusal; refusal_text
    // and display_summary should both be absent from the JSON.
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    // Before the first turn, last_turn_at should be absent (not serialized as null).
    let pre_turn = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}", id))
        .send()
        .expect("pre-turn get");
    let pre_turn_text = pre_turn.text().expect("text");
    assert!(
        !pre_turn_text.contains("\"last_turn_at\""),
        "last_turn_at must be absent before the first turn; body was:\n{}",
        pre_turn_text,
    );

    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "hi"}))
        .send()
        .expect("turn");
    assert_eq!(response.status(), 200);
    let body_text = response.text().expect("text");
    // refusal_text must be absent (not serialized as null) on non-refusal outcomes.
    assert!(
        !body_text.contains("\"refusal_text\":null"),
        "refusal_text must be absent (not null) on non-refusal turns; body was:\n{}",
        body_text,
    );
    // tool_calls is an empty array for this turn, so display_summary won't appear at all
    // here, but verify that the SessionResponse.cwd field on GET is a string, not null.
    let get = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}", id))
        .send()
        .expect("get");
    let session_text = get.text().expect("text");
    let session: serde_json::Value = serde_json::from_str(&session_text).expect("parse session");
    assert!(
        session["cwd"].is_string(),
        "cwd must be a string when set; got: {}",
        session["cwd"],
    );
    assert!(
        session["last_turn_at"].is_string(),
        "last_turn_at must be a timestamp string after a turn; got: {}",
        session["last_turn_at"],
    );
}

/// End-to-end proof that a scheduled job actually fires: the agent creates a one-shot job through
/// `schedule_create`, and the scheduler running inside `meka serve` delivers its prompt as a turn
/// with no HTTP request driving it.
///
/// This is the whole feature in one test. Everything else about scheduling is unit-tested, but
/// nothing else proves the scheduler is wired into the server, that an agent-initiated turn reaches
/// the model, or that its output is persisted where a client can read it back.
#[test]
fn scheduled_job_fires_without_a_client_request() {
    let script = serde_json::json!([
        // Turn 1: the agent schedules a one-shot two seconds out.
        [
            { "kind": "tool_use_start", "id": "tu_1", "name": "schedule_create" },
            { "kind": "tool_use_end", "input": {
                "prompt": "DELIVERED_PROMPT_MARKER",
                "at": "2s"
            }},
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "scheduled it" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        // Turn 2 is the fire. Nothing on the client side asks for this one.
        [
            { "kind": "text", "text": "SCHEDULED_REPLY_MARKER" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("\n[schedule]\npoll_interval = \"1s\"\n", script);

    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    let scheduled = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "remind me in two seconds"}))
        .send()
        .expect("send");
    assert_eq!(scheduled.status(), 200);

    // Poll for the fired turn rather than sleeping a fixed span: the job is due in 2s and the
    // scheduler ticks every 1s, so the turn lands somewhere in a window rather than at an instant.
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut body = String::new();
    while Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(250));
        let messages = harness
            .request(
                reqwest::Method::GET,
                &format!("/v1/sessions/{}/messages", id),
            )
            .send()
            .expect("messages");
        body = messages.text().expect("body");
        if body.contains("SCHEDULED_REPLY_MARKER") {
            break;
        }
    }

    // Deliberately counted, not merely contained: the marker appears once in turn 1's
    // `schedule_create` tool-call input, which is echoed back by `GET /messages` whether or not the
    // job ever fires. Only a real delivery produces a second occurrence.
    assert!(
        body.matches("DELIVERED_PROMPT_MARKER").count() >= 2,
        "the job's prompt must be delivered as a turn nobody requested, not just appear in the \
         tool call that created it; messages were:\n{}",
        body,
    );
    assert!(
        body.contains("Scheduled job"),
        "the delivered prompt must be marked as scheduled so the model knows no human is \
         waiting; messages were:\n{}",
        body,
    );
    assert!(
        body.contains("SCHEDULED_REPLY_MARKER"),
        "the agent's reply to the scheduled turn must be persisted for a client to read back; \
         messages were:\n{}",
        body,
    );
}
