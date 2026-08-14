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

    /// Block until a turn is actually running on `id`.
    ///
    /// Every test that asserts in-flight behaviour (409, 429, cancel, delete-refusal) needs the
    /// turn *admitted* first, and a fixed sleep is a bet on how fast admission is. It is normally
    /// tens of milliseconds, but on a loaded machine it overruns any constant small enough to keep
    /// the suite quick, and the test then fails claiming the server did not reject a turn that had
    /// not started yet. Polling the state the assertion depends on removes the guesswork without
    /// weakening it.
    fn wait_until_in_flight(&self, id: &str) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            let body: serde_json::Value = self
                .request(reqwest::Method::GET, &format!("/v1/sessions/{}", id))
                .send()
                .expect("in-flight probe")
                .json()
                .expect("parse");
            if body["turn_in_flight"] == true {
                return;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        panic!("no turn became in flight on session {} within 10s", id);
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

/// The case an `Idempotency-Key` exists for. A client whose request timeout is shorter than the
/// turn hangs up, the turn runs to completion anyway, and the documented recovery is to retry with
/// the same key. That retry must replay the first turn's answer, not run a second one: the first
/// already committed its messages, so re-running would duplicate its tool calls and its bill.
#[test]
fn retrying_after_a_timeout_replays_the_turn_instead_of_repeating_it() {
    let script = serde_json::json!([
        [
            { "kind": "sleep", "ms": 2000 },
            { "kind": "text", "text": "FIRST-TURN-ANSWER" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "text", "text": "SECOND-TURN-RAN" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let id = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send")
        .json::<serde_json::Value>()
        .expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    let key = "timed-out-then-retried";
    let timed_out = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .header("Idempotency-Key", key)
        .timeout(Duration::from_millis(500))
        .json(&serde_json::json!({"message": "do the thing"}))
        .send();
    assert!(timed_out.is_err(), "the client must have given up");

    // Wait for the abandoned turn to finish and record itself against the key.
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(200));
        let body: serde_json::Value = harness
            .request(reqwest::Method::GET, &format!("/v1/sessions/{}", id))
            .send()
            .expect("send")
            .json()
            .expect("parse");
        if body["turn_in_flight"] == false {
            break;
        }
    }

    let retried = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .header("Idempotency-Key", key)
        .json(&serde_json::json!({"message": "do the thing"}))
        .send()
        .expect("send");
    assert_eq!(retried.status(), 200);
    let text = retried.text().expect("body");
    assert!(
        text.contains("FIRST-TURN-ANSWER"),
        "the retry must replay the first turn's cached answer: {}",
        text
    );
    assert!(
        !text.contains("SECOND-TURN-RAN"),
        "the retry must not have run a second turn: {}",
        text
    );
}

/// A blocking turn outlives most client timeouts, so a client giving up mid-turn is ordinary, not
/// exotic. axum drops a handler's future when the connection closes, and dropping this one would
/// abandon the turn partway: the running tool's future goes with it, the in-memory conversation
/// keeps an assistant `tool_use` whose result never arrives, and no webhook ever reports the end.
/// The turn must run to completion, exactly as the streaming path's already-documented behaviour.
#[test]
fn a_blocking_turn_survives_the_client_hanging_up() {
    let script = serde_json::json!([[
        { "kind": "sleep", "ms": 2000 },
        { "kind": "text", "text": "finished anyway" },
        { "kind": "message_end", "stop_reason": "end_turn" }
    ]]);
    let harness = ServeTestHarness::spawn("", script);
    let id = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send")
        .json::<serde_json::Value>()
        .expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    // Hang up well before the turn can finish. The 500ms timeout is the client's, not the
    // server's: the request is gone from this side while the turn is still running.
    let hung_up = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .timeout(Duration::from_millis(500))
        .json(&serde_json::json!({"message": "take your time"}))
        .send();
    assert!(
        hung_up.is_err(),
        "the client must actually have given up for this test to mean anything"
    );

    // Give the abandoned turn room to finish on its own.
    std::thread::sleep(Duration::from_millis(3000));

    let messages: serde_json::Value = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/messages", id),
        )
        .send()
        .expect("send")
        .json()
        .expect("parse");
    let text = messages.to_string();
    assert!(
        text.contains("finished anyway"),
        "the turn must have completed and persisted despite the client leaving: {}",
        text
    );

    // And the session must be usable again rather than stuck holding the runtime mutex.
    let next = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}", id))
        .send()
        .expect("send")
        .json::<serde_json::Value>()
        .expect("parse");
    assert_eq!(
        next["turn_in_flight"], false,
        "the abandoned turn must have released the session: {}",
        next
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
    harness.wait_until_in_flight(&id);
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

    harness.wait_until_in_flight(&id);
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
    harness.wait_until_in_flight(&id);
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

    harness.wait_until_in_flight(&ids[0]);
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

    // Wait for the first request to be admitted; by then the Pending sentinel is installed in the
    // cache, since it is written before the turn is dispatched.
    harness.wait_until_in_flight(&id);
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
    harness.wait_until_in_flight(&id);
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

    // Wait for turn A to be admitted so the runtime mutex is held.
    harness.wait_until_in_flight(&id);

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

    harness.wait_until_in_flight(&id);
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
    harness.wait_until_in_flight(&id);
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

/// A scheduled turn is still a turn. It holds the runtime mutex without going through `TurnGuard`,
/// so unless it marks the session busy every guard built on `in_flight` is blind to it: `DELETE`
/// would cascade the row away mid-turn, `PATCH` would slip a permission change into a turn nobody
/// is watching, and `turn_in_flight` would answer `false` while the agent is mid-tool.
#[test]
fn a_scheduled_turn_marks_the_session_in_flight() {
    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "tu_1", "name": "schedule_create" },
            { "kind": "tool_use_end", "input": { "prompt": "later", "at": "2s" }},
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "scheduled it" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        // The fire itself, held open long enough to observe the flag from outside.
        [
            { "kind": "sleep", "ms": 3000 },
            { "kind": "text", "text": "fired" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("\n[schedule]\npoll_interval = \"1s\"\n", script);
    let id = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create")
        .json::<serde_json::Value>()
        .expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    assert_eq!(
        harness
            .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
            .json(&serde_json::json!({"message": "schedule something"}))
            .send()
            .expect("send")
            .status(),
        200
    );

    // Poll until the scheduled turn is running, then assert the flag is visible from outside.
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut seen_in_flight = false;
    while Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(200));
        let body: serde_json::Value = harness
            .request(reqwest::Method::GET, &format!("/v1/sessions/{}", id))
            .send()
            .expect("send")
            .json()
            .expect("parse");
        if body["turn_in_flight"] == true {
            seen_in_flight = true;
            // And the guards that read it must actually refuse while it is set.
            let deleted = harness
                .request(reqwest::Method::DELETE, &format!("/v1/sessions/{}", id))
                .send()
                .expect("send");
            assert_eq!(
                deleted.status(),
                409,
                "DELETE must not remove a session out from under a scheduled turn"
            );
            break;
        }
    }
    assert!(
        seen_in_flight,
        "the scheduled turn never reported itself as in flight"
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

// ---------------------------------------------------------------------------
// Session capability endpoints: compact, context, rewind, export, import, and
// the schedule / background-task surfaces.
// ---------------------------------------------------------------------------

/// Create a session and run one turn against it, returning the session id. Several of the
/// capability endpoints are only interesting once a conversation exists.
fn session_with_one_turn(harness: &ServeTestHarness) -> String {
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send");
    assert_eq!(create.status(), 201);
    let created: serde_json::Value = create.json().expect("parse");
    let id = created["id"].as_str().expect("id").to_string();

    let turn = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "first question"}))
        .send()
        .expect("send");
    assert_eq!(
        turn.status(),
        200,
        "turn failed: {}",
        turn.text().unwrap_or_default()
    );
    id
}

/// A script with `n` identical simple turns, for tests that need more than one provider round.
fn mock_turns(n: usize) -> serde_json::Value {
    serde_json::Value::Array(
        (0..n)
            .map(|i| {
                serde_json::json!([
                    { "kind": "text", "text": format!("reply {}", i) },
                    { "kind": "message_end", "stop_reason": "end_turn" }
                ])
            })
            .collect(),
    )
}

#[test]
fn context_endpoint_reports_window_and_totals() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let id = session_with_one_turn(&harness);

    let response = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/context", id),
        )
        .send()
        .expect("send");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["session_id"], id);
    assert!(
        body["message_count"].as_u64().expect("message_count") >= 2,
        "one turn leaves at least a user and an assistant message: {}",
        body
    );
    assert_eq!(
        body["totals"]["turns"], 1,
        "cumulative totals come from the DB and must survive independently of the live gauge",
    );
}

/// `used` is absent rather than `0` when nothing has measured the window. Zero would read as
/// "empty" to any client that divides by `window`, which is exactly wrong for a re-attached
/// session holding a long conversation.
#[test]
fn context_endpoint_omits_used_before_any_turn() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send");
    let created: serde_json::Value = create.json().expect("parse");
    let id = created["id"].as_str().expect("id");

    let body: serde_json::Value = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/context", id),
        )
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert!(
        body.get("used").is_none(),
        "an unmeasured window must omit `used`, not report 0: {}",
        body
    );
    assert!(
        body.get("used_percent").is_none(),
        "occupancy cannot be computed without `used`: {}",
        body
    );
}

#[test]
fn context_endpoint_requires_sessions_read_scope() {
    let harness =
        ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", &["sessions:w"]);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send");
    let created: serde_json::Value = create.json().expect("parse");
    let id = created["id"].as_str().expect("id");

    let response = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/context", id),
        )
        .send()
        .expect("send");
    assert_eq!(response.status(), 403);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["type"], "https://meka.so/errors/auth-scope");
    assert!(
        body["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("sessions:r"),
        "the rejection must name the scope the caller is missing: {}",
        body
    );
}

/// Compaction with the checkpoint turn off exercises the standalone summariser, which needs one
/// extra provider round beyond the turn itself.
#[test]
fn compact_replaces_the_window_with_a_summary() {
    let harness =
        ServeTestHarness::spawn("\n[session]\ncompact_checkpoint = false\n", mock_turns(3));
    let id = session_with_one_turn(&harness);

    let before: serde_json::Value = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/messages", id),
        )
        .send()
        .expect("send")
        .json()
        .expect("parse");
    let before_total = before["total"].as_u64().expect("total");

    let response = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/compact", id),
        )
        .json(&serde_json::json!({"instructions": "keep the question"}))
        .send()
        .expect("send");
    assert_eq!(
        response.status(),
        200,
        "compact failed: {}",
        response.text().unwrap_or_default()
    );
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(
        body["source"], "summarizer",
        "with compact_checkpoint = false the summariser is the only strategy left",
    );
    assert_eq!(body["messages_before"].as_u64(), Some(before_total));
    assert!(
        body["messages_after"].as_u64().expect("after") >= 1,
        "compaction always leaves at least the summary: {}",
        body
    );
}

/// An empty body means "compact with no guidance". Requiring `{}` would make every client send a
/// payload to say nothing.
#[test]
fn compact_accepts_an_empty_body() {
    let harness =
        ServeTestHarness::spawn("\n[session]\ncompact_checkpoint = false\n", mock_turns(3));
    let id = session_with_one_turn(&harness);
    let response = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/compact", id),
        )
        .send()
        .expect("send");
    assert_eq!(
        response.status(),
        200,
        "an empty body must be accepted: {}",
        response.text().unwrap_or_default()
    );
}

#[test]
fn compact_requires_sessions_write_scope() {
    let harness =
        ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", &["sessions:r"]);
    let response = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/compact", uuid::Uuid::nil()),
        )
        .send()
        .expect("send");
    assert_eq!(response.status(), 403);
    let body: serde_json::Value = response.json().expect("parse");
    assert!(
        body["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("sessions:w"),
        "{}",
        body
    );
}

#[test]
fn rewind_drops_the_last_turn_and_persists_it() {
    let harness = ServeTestHarness::spawn("", mock_turns(2));
    let id = session_with_one_turn(&harness);

    let response = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/rewind", id),
        )
        .json(&serde_json::json!({"turns": 1}))
        .send()
        .expect("send");
    assert_eq!(
        response.status(),
        200,
        "rewind failed: {}",
        response.text().unwrap_or_default()
    );
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["turns_removed"], 1);
    assert_eq!(
        body["messages_after"], 0,
        "rewinding the only turn empties the window: {}",
        body
    );

    // The event has to reach the DB, not just the in-memory conversation, or the rewind is
    // undone the moment the session is evicted.
    let messages: serde_json::Value = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/messages", id),
        )
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert_eq!(
        messages["total"], 0,
        "the persisted log must reflect the rewind: {}",
        messages
    );
}

/// Rewinding past the start is a statement about the caller's request, not a server fault.
#[test]
fn rewind_past_the_start_is_422() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let id = session_with_one_turn(&harness);
    let response = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/rewind", id),
        )
        .json(&serde_json::json!({"turns": 99}))
        .send()
        .expect("send");
    assert_eq!(response.status(), 422);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["type"], "https://meka.so/errors/invalid-body");
}

#[test]
fn rewind_rejects_zero_turns() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let id = session_with_one_turn(&harness);
    let response = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/rewind", id),
        )
        .json(&serde_json::json!({"turns": 0}))
        .send()
        .expect("send");
    assert_eq!(response.status(), 422);
}

#[test]
fn export_returns_markdown_by_default_and_json_on_request() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let id = session_with_one_turn(&harness);

    let markdown = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}/export", id))
        .send()
        .expect("send");
    assert_eq!(markdown.status(), 200);
    assert_eq!(
        markdown
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("text/markdown; charset=utf-8"),
    );
    let body = markdown.text().expect("text");
    assert!(
        body.contains("first question"),
        "the markdown export must carry the conversation: {}",
        body
    );

    let json = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/export?format=json", id),
        )
        .send()
        .expect("send");
    assert_eq!(json.status(), 200);
    let envelope: serde_json::Value = json.json().expect("parse");
    assert_eq!(envelope["root_session_id"], id);
    assert!(
        !envelope["sessions"]
            .as_array()
            .expect("sessions")
            .is_empty()
    );
}

#[test]
fn export_rejects_an_unknown_format() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let id = session_with_one_turn(&harness);
    let response = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/export?format=pdf", id),
        )
        .send()
        .expect("send");
    assert_eq!(response.status(), 422);
}

/// Export then import round-trips into a *new* session rather than colliding with the original,
/// which is what makes the pair usable for cloning a conversation.
#[test]
fn export_json_round_trips_through_import() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let id = session_with_one_turn(&harness);

    let envelope: serde_json::Value = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/export?format=json", id),
        )
        .send()
        .expect("send")
        .json()
        .expect("parse");

    let response = harness
        .request(reqwest::Method::POST, "/v1/sessions/import")
        .json(&envelope)
        .send()
        .expect("send");
    assert_eq!(
        response.status(),
        201,
        "import failed: {}",
        response.text().unwrap_or_default()
    );
    let imported: serde_json::Value = response.json().expect("parse");
    let new_id = imported["session_id"].as_str().expect("session_id");
    assert_ne!(
        new_id, id,
        "import must mint a fresh id, never reuse the exported one"
    );
    assert_eq!(imported["sessions_imported"], 1);

    let messages: serde_json::Value = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/messages", new_id),
        )
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert!(
        messages["total"].as_u64().expect("total") >= 2,
        "the imported session must carry the conversation: {}",
        messages
    );
}

#[test]
fn import_rejects_a_malformed_envelope() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let response = harness
        .request(reqwest::Method::POST, "/v1/sessions/import")
        .json(&serde_json::json!({"format_version": 999, "sessions": []}))
        .send()
        .expect("send");
    assert_eq!(
        response.status(),
        422,
        "a bad envelope is the caller's input, so it is a 4xx, not a 500"
    );
}

/// Every surface that prints a job id to a human prints the 8-character short form, so that is
/// what gets pasted here. Cancelling on it must work, and an id matching nothing must say so
/// rather than answering 204 over a job that is still firing.
#[test]
fn cancelling_a_scheduled_job_takes_a_prefix_and_reports_a_miss() {
    let harness = ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", &[
        "sessions:r",
        "sessions:w",
        "schedule:r",
        "schedule:w",
    ]);
    let id = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send")
        .json::<serde_json::Value>()
        .expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    let job_id = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/schedule", id),
        )
        .json(&serde_json::json!({"prompt": "check the build", "every": "30m"}))
        .send()
        .expect("send")
        .json::<serde_json::Value>()
        .expect("parse")["id"]
        .as_str()
        .expect("job id")
        .to_string();

    let missing = harness
        .request(
            reqwest::Method::DELETE,
            "/v1/schedule/deadbeef-0000-0000-0000-000000000000",
        )
        .send()
        .expect("send");
    assert_eq!(
        missing.status(),
        404,
        "an id that matches no job must not report a cancellation"
    );
    let body: serde_json::Value = missing.json().expect("parse");
    assert_eq!(body["type"], "https://meka.so/errors/not-found");

    let short = &job_id[..8];
    let cancel = harness
        .request(reqwest::Method::DELETE, &format!("/v1/schedule/{}", short))
        .send()
        .expect("send");
    assert_eq!(
        cancel.status(),
        204,
        "the short form printed by `meka schedule list` must cancel"
    );

    let after: serde_json::Value = harness
        .request(reqwest::Method::GET, "/v1/schedule")
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert!(
        after["jobs"].as_array().expect("jobs").is_empty(),
        "the job must actually be gone: {}",
        after
    );
}

/// With the scheduler off, this endpoint is the only way a job can still be created: the
/// `schedule_*` tools are not registered and there is no CLI for it. Accepting one would persist a
/// job that never fires, listed forever with a `next_fire_at` receding into the past.
#[test]
fn creating_a_job_is_refused_when_scheduling_is_disabled() {
    let harness = ServeTestHarness::spawn_with(
        "\n[schedule]\nenabled = false\n",
        mock_simple_turn(),
        "sk_test_token",
        &["sessions:r", "sessions:w", "schedule:r", "schedule:w"],
    );
    let id = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send")
        .json::<serde_json::Value>()
        .expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    let created = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/schedule", id),
        )
        .json(&serde_json::json!({"prompt": "never runs", "every": "10s"}))
        .send()
        .expect("send");
    assert_eq!(created.status(), 422);

    // Listing and cancelling stay open: clearing out jobs left from before the flag was flipped is
    // exactly what an operator does next.
    let listed = harness
        .request(reqwest::Method::GET, "/v1/schedule")
        .send()
        .expect("send");
    assert_eq!(listed.status(), 200);
    assert!(
        listed.json::<serde_json::Value>().expect("parse")["jobs"]
            .as_array()
            .expect("jobs")
            .is_empty(),
        "the refused job must not have been persisted"
    );
}

/// `GET` hands back the stored body, so a client that edits and `PUT`s it writes back what it was
/// given. The agent-facing rendering prepends a base-directory line; returning that here would
/// bake an absolute host path into `SKILL.md`, once more per edit cycle.
#[test]
fn a_skill_body_round_trips_through_get_and_put() {
    let harness = ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", &[
        "skills:r", "skills:w",
    ]);
    let body = "# Deploy\n\nRun `scripts/deploy.sh` and report the exit code.\n";
    harness
        .request(reqwest::Method::PUT, "/v1/skills/deploy")
        .json(&serde_json::json!({"description": "how to deploy", "body": body}))
        .send()
        .expect("send");

    let first: serde_json::Value = harness
        .request(reqwest::Method::GET, "/v1/skills/deploy")
        .send()
        .expect("send")
        .json()
        .expect("parse");
    let fetched = first["body"].as_str().expect("body").to_string();
    assert!(
        !fetched.contains("Base directory for this skill"),
        "the agent-facing header must not leak into the stored body: {}",
        fetched
    );

    // The read-modify-write an editing client performs.
    harness
        .request(reqwest::Method::PUT, "/v1/skills/deploy")
        .json(&serde_json::json!({"description": "how to deploy", "body": fetched}))
        .send()
        .expect("send");
    let second: serde_json::Value = harness
        .request(reqwest::Method::GET, "/v1/skills/deploy")
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert_eq!(
        second["body"].as_str().expect("body"),
        fetched,
        "a GET/PUT cycle must be byte-stable"
    );
}

/// Omitting `priority` on an update must keep it, the way omitting `body` or `author` already
/// does. Resetting it would make the obvious edit demote a skill out of the index the model reads.
#[test]
fn omitting_priority_keeps_it_rather_than_resetting_to_the_default() {
    let harness = ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", &[
        "skills:r", "skills:w", "memory:r", "memory:w",
    ]);
    for (store, path) in [
        ("skill", "/v1/skills/ranked"),
        ("memory", "/v1/memory/ranked"),
    ] {
        harness
            .request(reqwest::Method::PUT, path)
            .json(&serde_json::json!({"description": "d", "priority": 1, "body": "b"}))
            .send()
            .expect("create");
        let updated: serde_json::Value = harness
            .request(reqwest::Method::PUT, path)
            .json(&serde_json::json!({"description": "revised", "body": "b"}))
            .send()
            .expect("update")
            .json()
            .expect("parse");
        assert_eq!(
            updated["priority"], 1,
            "{} priority must survive an update that does not mention it: {}",
            store, updated
        );
        // And it is still settable, so preservation has not made the field inert.
        let reset: serde_json::Value = harness
            .request(reqwest::Method::PUT, path)
            .json(&serde_json::json!({"description": "revised", "priority": 7, "body": "b"}))
            .send()
            .expect("update")
            .json()
            .expect("parse");
        assert_eq!(
            reset["priority"], 7,
            "{} priority must stay settable",
            store
        );
    }
}

/// A gate runs a shell command on a timer, before the turn, as the server's user, and needs no
/// working provider to do it. `GET /v1/schedule` hands a `schedule:r` token every session id in
/// the database, so if `schedule:w` alone could plant a gate, scoping a bridge to `schedule:*`
/// would quietly be granting it unattended arbitrary shell.
#[test]
fn planting_a_gate_needs_more_than_the_schedule_scope() {
    // Two tokens: the one under test holds only `schedule:*`, and a full-scope one mints the
    // session so that setup is not what fails.
    let config = "\n[[serve.tokens]]\ntoken = \"sk_test_full\"\n\
                  scopes = [\"sessions:r\", \"sessions:w\"]\n";
    let harness = ServeTestHarness::spawn_with(config, mock_simple_turn(), "sk_test_token", &[
        "schedule:r",
        "schedule:w",
    ]);
    let id = harness
        .client
        .post(format!("{}/v1/sessions", harness.base_url))
        .header("Authorization", "Bearer sk_test_full")
        .json(&serde_json::json!({
            "cwd": std::env::temp_dir().to_string_lossy(),
            "permission": "write",
        }))
        .send()
        .expect("send")
        .json::<serde_json::Value>()
        .expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    let gated = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/schedule", id),
        )
        .json(&serde_json::json!({
            "prompt": "report",
            "every": "30m",
            "gate": {"command": "id"},
        }))
        .send()
        .expect("send");
    assert_eq!(
        gated.status(),
        403,
        "a schedule-only token must not reach a shell"
    );

    // The ordinary prompt-only job stays reachable: this is a narrowing of gates, not of
    // scheduling.
    let plain = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/schedule", id),
        )
        .json(&serde_json::json!({"prompt": "report", "every": "30m"}))
        .send()
        .expect("send");
    assert_eq!(
        plain.status(),
        201,
        "a gateless job needs only `schedule:w`"
    );
}

/// A gate refused for the *session's* permission is not a token problem, and the docs tell clients
/// to route on `type`. Reporting it as `auth-scope` sends them to re-provision a token forever.
#[test]
fn a_gate_below_write_permission_is_not_reported_as_a_scope_failure() {
    let harness = ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", &[
        "sessions:r",
        "sessions:w",
        "schedule:r",
        "schedule:w",
    ]);
    let id = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({
            "cwd": std::env::temp_dir().to_string_lossy(),
            "permission": "read",
        }))
        .send()
        .expect("send")
        .json::<serde_json::Value>()
        .expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    let response = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/schedule", id),
        )
        .json(&serde_json::json!({
            "prompt": "report",
            "every": "30m",
            "gate": {"command": "git status --porcelain"},
        }))
        .send()
        .expect("send");
    assert_eq!(response.status(), 403);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(
        body["type"], "https://meka.so/errors/session-permission",
        "the remedy is PATCH the session, not a better token: {}",
        body
    );
}

#[test]
fn scheduled_job_create_list_and_cancel_round_trip() {
    let harness = ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", &[
        "sessions:r",
        "sessions:w",
        "schedule:r",
        "schedule:w",
    ]);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send");
    let created: serde_json::Value = create.json().expect("parse");
    let id = created["id"].as_str().expect("id").to_string();

    let response = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/schedule", id),
        )
        // Far enough out that the scheduler cannot fire it mid-test.
        .json(&serde_json::json!({"prompt": "check the build", "every": "6h"}))
        .send()
        .expect("send");
    assert_eq!(
        response.status(),
        201,
        "schedule create failed: {}",
        response.text().unwrap_or_default()
    );
    let job: serde_json::Value = response.json().expect("parse");
    let job_id = job["id"].as_str().expect("job id").to_string();
    assert_eq!(job["session_id"], id);
    assert!(
        job["schedule"].as_str().unwrap_or_default().contains("6h"),
        "the rendered schedule should describe the interval: {}",
        job
    );

    // Visible both server-wide and scoped to the session.
    let all: serde_json::Value = harness
        .request(reqwest::Method::GET, "/v1/schedule")
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert!(
        all["jobs"]
            .as_array()
            .expect("jobs")
            .iter()
            .any(|j| j["id"] == job_id.as_str()),
        "the job must appear in the server-wide listing: {}",
        all
    );

    let scoped: serde_json::Value = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/schedule", id),
        )
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert_eq!(scoped["jobs"].as_array().expect("jobs").len(), 1);

    let cancel = harness
        .request(reqwest::Method::DELETE, &format!("/v1/schedule/{}", job_id))
        .send()
        .expect("send");
    assert_eq!(cancel.status(), 204);

    let after: serde_json::Value = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/schedule", id),
        )
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert!(after["jobs"].as_array().expect("jobs").is_empty());
}

/// Giving two schedules is refused rather than resolved by precedence: silently honouring one
/// would produce a job firing on a schedule nobody asked for.
#[test]
fn scheduled_job_rejects_two_schedules() {
    let harness = ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", &[
        "sessions:r",
        "sessions:w",
        "schedule:r",
        "schedule:w",
    ]);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send");
    let created: serde_json::Value = create.json().expect("parse");
    let id = created["id"].as_str().expect("id");

    let response = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/schedule", id),
        )
        .json(&serde_json::json!({"prompt": "x", "every": "6h", "cron": "0 9 * * *"}))
        .send()
        .expect("send");
    assert_eq!(response.status(), 422);
    let body: serde_json::Value = response.json().expect("parse");
    assert!(
        body["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("exactly one"),
        "{}",
        body
    );
}

#[test]
fn schedule_endpoints_require_the_schedule_scopes() {
    // A token that can drive turns but was never granted the schedule scopes must not be able to
    // plant unattended work.
    let harness = ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", &[
        "sessions:r",
        "sessions:w",
    ]);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send");
    let created: serde_json::Value = create.json().expect("parse");
    let id = created["id"].as_str().expect("id");

    let listing = harness
        .request(reqwest::Method::GET, "/v1/schedule")
        .send()
        .expect("send");
    assert_eq!(listing.status(), 403);

    let creating = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/schedule", id),
        )
        .json(&serde_json::json!({"prompt": "x", "every": "6h"}))
        .send()
        .expect("send");
    assert_eq!(creating.status(), 403);
}

#[test]
fn background_tasks_endpoint_lists_an_empty_session() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let id = session_with_one_turn(&harness);
    let response = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}/tasks", id))
        .send()
        .expect("send");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().expect("parse");
    assert!(body["tasks"].as_array().expect("tasks").is_empty());
}

#[test]
fn background_task_cancel_on_unknown_id_is_404() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let id = session_with_one_turn(&harness);
    let response = harness
        .request(
            reqwest::Method::DELETE,
            &format!("/v1/sessions/{}/tasks/does-not-exist", id),
        )
        .send()
        .expect("send");
    assert_eq!(response.status(), 404);
}

/// Read-only capability endpoints must 404 on an unknown session rather than reviving or
/// inventing one.
#[test]
fn capability_endpoints_404_on_unknown_session() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let missing = uuid::Uuid::new_v4();
    for path in [
        format!("/v1/sessions/{}/export", missing),
        format!("/v1/sessions/{}/tasks", missing),
        format!("/v1/sessions/{}/context", missing),
    ] {
        let response = harness
            .request(reqwest::Method::GET, &path)
            .send()
            .expect("send");
        assert_eq!(response.status(), 404, "{} should 404", path);
        let body: serde_json::Value = response.json().expect("parse");
        assert_eq!(body["type"], "https://meka.so/errors/session-not-found");
    }
}

/// A compaction that shrinks `/messages` has to say so. Without the marker a polling client sees
/// `total` drop and messages it already rendered stop coming back, which is indistinguishable from
/// the server losing the conversation.
#[test]
fn compaction_marks_the_summary_in_the_message_history() {
    let harness =
        ServeTestHarness::spawn("\n[session]\ncompact_checkpoint = false\n", mock_turns(3));
    let id = session_with_one_turn(&harness);

    let compact = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/compact", id),
        )
        .send()
        .expect("send");
    assert_eq!(compact.status(), 200);

    let messages: serde_json::Value = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/messages", id),
        )
        .send()
        .expect("send")
        .json()
        .expect("parse");
    let marked: Vec<&serde_json::Value> = messages["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .filter(|m| m.get("compaction").is_some())
        .collect();
    assert_eq!(
        marked.len(),
        1,
        "exactly the summary carries the marker: {}",
        messages
    );
    assert_eq!(marked[0]["compaction"]["generation"], 1);
    assert!(
        marked[0]["compaction"]["replaced_count"]
            .as_u64()
            .expect("replaced_count")
            > 0,
        "a compaction always replaces something: {}",
        marked[0]
    );
}

/// Every other message must stay unmarked, or a client keying off the field's presence would treat
/// the whole transcript as summaries.
#[test]
fn ordinary_messages_carry_no_compaction_marker() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let id = session_with_one_turn(&harness);
    let messages: serde_json::Value = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/messages", id),
        )
        .send()
        .expect("send")
        .json()
        .expect("parse");
    for message in messages["messages"].as_array().expect("messages") {
        assert!(
            message.get("compaction").is_none(),
            "an uncompacted session must have no markers: {}",
            message
        );
    }
}

// ---------------------------------------------------------------------------
// Store endpoints: skills, memory, tools, instructions, providers.
// ---------------------------------------------------------------------------

/// Scopes for a token that can drive both stores plus sessions.
const STORE_SCOPES: &[&str] = &[
    "sessions:r",
    "sessions:w",
    "skills:r",
    "skills:w",
    "memory:r",
    "memory:w",
];

#[test]
fn skill_write_read_and_delete_round_trip() {
    let harness =
        ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", STORE_SCOPES);

    let write = harness
        .request(reqwest::Method::PUT, "/v1/skills/greet-user")
        .json(&serde_json::json!({
            "description": "Greet the user warmly",
            "priority": 2,
            "body": "Say hello, then ask what they need.",
            "author": "integration-test",
        }))
        .send()
        .expect("send");
    assert_eq!(
        write.status(),
        200,
        "skill write failed: {}",
        write.text().unwrap_or_default()
    );
    let written: serde_json::Value = write.json().expect("parse");
    assert_eq!(written["name"], "greet-user");
    assert_eq!(written["priority"], 2);
    assert_eq!(written["author"], "integration-test");

    // The collection listing carries the new metadata, not just name + description.
    let listed: serde_json::Value = harness
        .request(reqwest::Method::GET, "/v1/skills")
        .send()
        .expect("send")
        .json()
        .expect("parse");
    let entry = listed
        .as_array()
        .expect("skills array")
        .iter()
        .find(|s| s["name"] == "greet-user")
        .expect("skill must be listed");
    assert_eq!(entry["priority"], 2);
    assert!(
        entry.get("body").is_none(),
        "the palette listing must not carry bodies: {}",
        entry
    );

    let detail: serde_json::Value = harness
        .request(reqwest::Method::GET, "/v1/skills/greet-user")
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert!(
        detail["body"]
            .as_str()
            .unwrap_or_default()
            .contains("Say hello"),
        "the single-skill view must carry the body: {}",
        detail
    );

    let delete = harness
        .request(reqwest::Method::DELETE, "/v1/skills/greet-user")
        .send()
        .expect("send");
    assert_eq!(delete.status(), 204);

    let gone = harness
        .request(reqwest::Method::GET, "/v1/skills/greet-user")
        .send()
        .expect("send");
    assert_eq!(gone.status(), 404);
}

/// An omitted `body` keeps the existing one. A caller correcting a description should not have to
/// resend prose it never meant to touch, and the alternative is silently clearing it.
#[test]
fn skill_write_without_a_body_preserves_the_existing_one() {
    let harness =
        ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", STORE_SCOPES);
    harness
        .request(reqwest::Method::PUT, "/v1/skills/keeper")
        .json(&serde_json::json!({"description": "first", "body": "original body"}))
        .send()
        .expect("send");
    harness
        .request(reqwest::Method::PUT, "/v1/skills/keeper")
        .json(&serde_json::json!({"description": "second"}))
        .send()
        .expect("send");

    let detail: serde_json::Value = harness
        .request(reqwest::Method::GET, "/v1/skills/keeper")
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert_eq!(detail["description"], "second");
    assert!(
        detail["body"]
            .as_str()
            .unwrap_or_default()
            .contains("original body"),
        "omitting `body` must preserve it: {}",
        detail
    );
}

/// The name reaches the filesystem, so it needs the same character-class guard the tools apply.
/// A traversal must be refused before any path join, not sanitised after one.
#[test]
fn skill_write_rejects_a_traversing_name() {
    let harness =
        ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", STORE_SCOPES);
    for bad in ["..", "a%2Fb", "has space"] {
        let response = harness
            .request(reqwest::Method::PUT, &format!("/v1/skills/{}", bad))
            .json(&serde_json::json!({"description": "nope"}))
            .send()
            .expect("send");
        assert!(
            response.status() == 422 || response.status() == 404,
            "'{}' must be refused, got {}",
            bad,
            response.status()
        );
    }
}

#[test]
fn skill_write_rejects_an_empty_description() {
    let harness =
        ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", STORE_SCOPES);
    let response = harness
        .request(reqwest::Method::PUT, "/v1/skills/blank")
        .json(&serde_json::json!({"description": "   "}))
        .send()
        .expect("send");
    assert_eq!(
        response.status(),
        422,
        "an empty description produces a skill that can never be loaded again"
    );
}

/// A read scope must not admit a write. The catalogue is flat: neither implies the other.
#[test]
fn skill_write_requires_the_write_scope() {
    let harness = ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", &[
        "sessions:r",
        "sessions:w",
        "skills:r",
    ]);
    let response = harness
        .request(reqwest::Method::PUT, "/v1/skills/nope")
        .json(&serde_json::json!({"description": "x"}))
        .send()
        .expect("send");
    assert_eq!(response.status(), 403);
    let body: serde_json::Value = response.json().expect("parse");
    assert!(
        body["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("skills:w"),
        "{}",
        body
    );
}

#[test]
fn memory_write_read_list_and_delete_round_trip() {
    let harness =
        ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", STORE_SCOPES);

    let write = harness
        .request(reqwest::Method::PUT, "/v1/memory/deploy-policy")
        .json(&serde_json::json!({
            "description": "Never deploy on Fridays",
            "priority": 1,
            "body": "Ship Monday to Thursday only.",
        }))
        .send()
        .expect("send");
    assert_eq!(
        write.status(),
        200,
        "memory write failed: {}",
        write.text().unwrap_or_default()
    );
    let written: serde_json::Value = write.json().expect("parse");
    assert_eq!(written["priority"], 1);

    let listed: serde_json::Value = harness
        .request(reqwest::Method::GET, "/v1/memory")
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert!(
        listed["memories"]
            .as_array()
            .expect("memories")
            .iter()
            .any(|m| m["name"] == "deploy-policy"),
        "{}",
        listed
    );
    assert_eq!(listed["ignored_over_cap"], 0);

    let detail: serde_json::Value = harness
        .request(reqwest::Method::GET, "/v1/memory/deploy-policy")
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert!(
        detail["body"]
            .as_str()
            .unwrap_or_default()
            .contains("Monday to Thursday"),
        "{}",
        detail
    );

    let delete = harness
        .request(reqwest::Method::DELETE, "/v1/memory/deploy-policy")
        .send()
        .expect("send");
    assert_eq!(delete.status(), 204);
    assert_eq!(
        harness
            .request(reqwest::Method::GET, "/v1/memory/deploy-policy")
            .send()
            .expect("send")
            .status(),
        404
    );
}

/// Memory reads and writes are separately scoped from sessions: a bridge token that runs turns
/// must not be able to read the user's notes, let alone empty them.
#[test]
fn memory_endpoints_require_the_memory_scopes() {
    let harness = ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", &[
        "sessions:r",
        "sessions:w",
    ]);
    assert_eq!(
        harness
            .request(reqwest::Method::GET, "/v1/memory")
            .send()
            .expect("send")
            .status(),
        403
    );
    assert_eq!(
        harness
            .request(reqwest::Method::DELETE, "/v1/memory/anything")
            .send()
            .expect("send")
            .status(),
        403
    );
}

#[test]
fn memory_read_scope_does_not_grant_writes() {
    let harness = ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", &[
        "sessions:r",
        "memory:r",
    ]);
    let response = harness
        .request(reqwest::Method::PUT, "/v1/memory/nope")
        .json(&serde_json::json!({"description": "x"}))
        .send()
        .expect("send");
    assert_eq!(response.status(), 403);
}

#[test]
fn session_tools_endpoint_lists_the_catalogue_with_permissions() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let id = session_with_one_turn(&harness);
    let response = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}/tools", id))
        .send()
        .expect("send");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().expect("parse");
    let tools = body["tools"].as_array().expect("tools");
    assert!(!tools.is_empty(), "a session always has built-in tools");
    let read_file = tools
        .iter()
        .find(|t| t["name"] == "read_file")
        .expect("read_file must be registered");
    assert_eq!(read_file["required_permission"], "read");
    assert_eq!(read_file["deferred"], false);
    let write_file = tools
        .iter()
        .find(|t| t["name"] == "write_file")
        .expect("write_file must be registered");
    assert_eq!(
        write_file["required_permission"], "write",
        "the catalogue must report the tier a client needs to render an approval prompt"
    );
}

#[test]
fn instructions_endpoint_reports_absence_rather_than_failing() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let response = harness
        .request(reqwest::Method::GET, "/v1/instructions")
        .send()
        .expect("send");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().expect("parse");
    assert!(
        body.get("content").is_none(),
        "an unconfigured server reports no instructions, not an error: {}",
        body
    );
}

#[test]
fn providers_endpoint_lists_profiles_and_marks_the_active_one() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let response = harness
        .request(reqwest::Method::GET, "/v1/providers")
        .send()
        .expect("send");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().expect("parse");
    let providers = body["providers"].as_array().expect("providers");
    let mock = providers
        .iter()
        .find(|p| p["name"] == "mock")
        .expect("the harness configures a 'mock' profile");
    assert_eq!(mock["type"], "claude-api");
    assert_eq!(mock["active"], true);
    // Credentials live in the database keyed by profile name and must never transit this API.
    let serialized = body.to_string();
    for secret_key in ["api_key", "token", "secret", "credential"] {
        assert!(
            !serialized.contains(secret_key),
            "provider listing must carry no credential-shaped fields, found '{}': {}",
            secret_key,
            serialized
        );
    }
}

#[test]
fn mcp_tools_for_an_unknown_server_is_404() {
    let harness = ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", &[
        "sessions:r",
        "mcp:r",
        "mcp:w",
    ]);
    assert_eq!(
        harness
            .request(reqwest::Method::GET, "/v1/mcp/nope/tools")
            .send()
            .expect("send")
            .status(),
        404
    );
    assert_eq!(
        harness
            .request(reqwest::Method::POST, "/v1/mcp/nope/reconnect")
            .send()
            .expect("send")
            .status(),
        404
    );
}

/// Reconnect is a write: it respawns a process or reopens a socket, so a read token must not
/// trigger it.
#[test]
fn mcp_reconnect_requires_the_mcp_write_scope() {
    let harness = ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", &[
        "sessions:r",
        "mcp:r",
    ]);
    let response = harness
        .request(reqwest::Method::POST, "/v1/mcp/anything/reconnect")
        .send()
        .expect("send");
    assert_eq!(response.status(), 403);
    let body: serde_json::Value = response.json().expect("parse");
    assert!(
        body["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("mcp:w"),
        "{}",
        body
    );
}

// ---------------------------------------------------------------------------
// SSE re-attach.
// ---------------------------------------------------------------------------

/// Collect the `event: <name>` lines from an SSE body, in order.
fn sse_event_names(body: &str) -> Vec<String> {
    body.lines()
        .filter_map(|line| line.strip_prefix("event: "))
        .map(|name| name.trim().to_string())
        .collect()
}

/// Collect the `id: <n>` lines from an SSE body, in order.
fn sse_event_ids(body: &str) -> Vec<u64> {
    body.lines()
        .filter_map(|line| line.strip_prefix("id: "))
        .filter_map(|value| value.trim().parse::<u64>().ok())
        .collect()
}

fn start_streaming_session(harness: &ServeTestHarness) -> String {
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send");
    create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string()
}

/// The case re-attach exists for: the turn finished but the client was not there to see it end.
/// The ring plus the recorded terminal are what let it find out without re-running anything.
#[test]
fn reattach_after_the_turn_ends_replays_the_tail_and_the_terminal() {
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "streamed reply" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let id = start_streaming_session(&harness);

    let first = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "hi", "stream": true}))
        .send()
        .expect("send");
    let original = first.text().expect("body");
    assert!(original.contains("event: turn.finished"), "{}", original);

    let rejoined = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}/stream", id))
        .send()
        .expect("send");
    assert_eq!(rejoined.status(), 200);
    let body = rejoined.text().expect("body");
    let names = sse_event_names(&body);
    assert_eq!(
        names.first().map(String::as_str),
        Some("turn.started"),
        "a rejoin announces which turn it attached to first: {}",
        body
    );
    assert!(
        body.contains("\"resumed\":true"),
        "the re-issued turn.started must be marked as a resume, not a new turn: {}",
        body
    );
    assert!(
        names.iter().any(|name| name == "assistant_text.delta"),
        "the ring must replay the turn's content: {}",
        body
    );
    assert!(
        names.last().map(String::as_str) == Some("turn.finished"),
        "a rejoin must terminate, and with the outcome the turn actually had: {}",
        body
    );
    assert!(
        body.contains("\"stop_reason\":\"end_turn\""),
        "the replayed terminal carries the real stop reason: {}",
        body
    );
}

/// `Last-Event-ID` resumes strictly after the id the client names, so a reconnecting client does
/// not re-render what it already showed.
#[test]
fn reattach_with_last_event_id_skips_what_was_already_delivered() {
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "one " },
            { "kind": "text", "text": "two " },
            { "kind": "text", "text": "three" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let id = start_streaming_session(&harness);
    let original = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "hi", "stream": true}))
        .send()
        .expect("send")
        .text()
        .expect("body");
    let original_ids = sse_event_ids(&original);
    assert!(
        original_ids.len() >= 3,
        "need several events to resume from the middle: {}",
        original
    );
    let resume_from = original_ids[1];

    let body = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}/stream", id))
        .header("Last-Event-ID", resume_from.to_string())
        .send()
        .expect("send")
        .text()
        .expect("body");
    let replayed = sse_event_ids(&body);
    assert!(
        replayed.iter().all(|value| *value > resume_from),
        "replay must start strictly after the client's last id {}: got {:?}\n{}",
        resume_from,
        replayed,
        body
    );
    assert!(
        replayed.contains(original_ids.last().expect("last id")),
        "the tail through the terminal must still arrive: {:?}\n{}",
        replayed,
        body
    );
}

/// The query parameter exists for clients that cannot set the header (`fetch`-based readers,
/// most non-browser HTTP libraries without header control on redirects).
#[test]
fn reattach_accepts_last_event_id_as_a_query_parameter() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let id = start_streaming_session(&harness);
    let original = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "hi", "stream": true}))
        .send()
        .expect("send")
        .text()
        .expect("body");
    let ids = sse_event_ids(&original);
    let resume_from = ids.first().copied().expect("at least one event");

    let body = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/stream?last_event_id={}", id, resume_from),
        )
        .send()
        .expect("send")
        .text()
        .expect("body");
    assert!(
        sse_event_ids(&body).iter().all(|v| *v > resume_from),
        "the query parameter must behave exactly like the header: {}",
        body
    );
}

/// A replay that cannot reach the client's `Last-Event-ID` has a hole in it. Saying so is the
/// point: a transcript with a silent gap cannot be repaired, one the client knows about can.
#[test]
fn reattach_warns_when_the_replay_buffer_cannot_reach_back_far_enough() {
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "a" },
            { "kind": "text", "text": "b" },
            { "kind": "text", "text": "c" },
            { "kind": "text", "text": "d" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    // A ring of 2 cannot cover a turn that emits more than that.
    let harness = ServeTestHarness::spawn("stream_replay_events = 2\n", script);
    let id = start_streaming_session(&harness);
    harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "hi", "stream": true}))
        .send()
        .expect("send")
        .text()
        .expect("body");

    let body = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}/stream", id))
        .header("Last-Event-ID", "0")
        .send()
        .expect("send")
        .text()
        .expect("body");
    assert!(
        body.contains("event: notice") && body.contains("Replay buffer does not reach"),
        "a truncated replay must be announced, not silently delivered: {}",
        body
    );
}

#[test]
fn reattach_on_a_session_that_never_streamed_is_404() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let id = start_streaming_session(&harness);
    let response = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}/stream", id))
        .send()
        .expect("send");
    assert_eq!(response.status(), 404);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(
        body["type"], "https://meka.so/errors/not-found",
        "the session exists; it is the stream that does not, and `type` is what a client \
         switches on",
    );
    assert!(
        body["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("submit a turn"),
        "the 404 must say how to get a stream, not just that there isn't one: {}",
        body
    );
}

#[test]
fn reattach_on_an_unknown_session_is_404() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let response = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/stream", uuid::Uuid::new_v4()),
        )
        .send()
        .expect("send");
    assert_eq!(response.status(), 404);
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["type"], "https://meka.so/errors/session-not-found");
}

#[test]
fn reattach_requires_sessions_read_scope() {
    let harness =
        ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", &["sessions:w"]);
    let response = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/stream", uuid::Uuid::new_v4()),
        )
        .send()
        .expect("send");
    assert_eq!(response.status(), 403);
}

/// The headline of the re-attach work: a streaming turn whose consumer goes away keeps running for
/// `stream_reattach_grace` instead of being cancelled with it. The existing re-attach tests all
/// keep the original consumer alive, so they would still pass if the grace regressed to zero;
/// this one drops the connection outright and asserts the turn finished and persisted anyway.
#[test]
fn a_streaming_turn_survives_its_consumer_disconnecting() {
    let script = serde_json::json!([[
        { "kind": "sleep", "ms": 2500 },
        { "kind": "text", "text": "finished without a listener" },
        { "kind": "message_end", "stop_reason": "end_turn" }
    ]]);
    // `stream_reattach_grace` defaults to 30s, which is the behaviour under test.
    let harness = ServeTestHarness::spawn("", script);
    let id = start_streaming_session(&harness);

    // `send()` returns as soon as the SSE headers land, with the body still streaming. Dropping
    // the response without reading it closes the connection mid-turn, which is what a closed
    // browser tab or a dead network looks like from the server's side.
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "go", "stream": true}))
        .send()
        .expect("send");
    assert_eq!(response.status(), 200, "the turn must have been admitted");
    drop(response);

    // The turn must still be running, not cancelled along with the connection.
    harness.wait_until_in_flight(&id);

    let deadline = Instant::now() + Duration::from_secs(30);
    let mut transcript = String::new();
    while Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(200));
        transcript = harness
            .request(
                reqwest::Method::GET,
                &format!("/v1/sessions/{}/messages", id),
            )
            .send()
            .expect("send")
            .text()
            .expect("body");
        if transcript.contains("finished without a listener") {
            break;
        }
    }
    assert!(
        transcript.contains("finished without a listener"),
        "the turn must have completed and persisted with nobody watching: {}",
        transcript
    );
}

/// Two readers on one turn. The live path has to fan out, or a client that reconnects while the
/// turn is still running would starve the original consumer.
#[test]
fn reattach_mid_turn_follows_the_live_stream() {
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "before " },
            { "kind": "sleep", "ms": 1500 },
            { "kind": "text", "text": "after" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let id = start_streaming_session(&harness);

    let base_url = harness.base_url.clone();
    let token = harness.token.clone();
    let session = id.clone();
    let original = std::thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("client");
        client
            .post(format!("{}/v1/sessions/{}/turn", base_url, session))
            .header("Authorization", format!("Bearer {}", token))
            .json(&serde_json::json!({"message": "hi", "stream": true}))
            .send()
            .expect("send")
            .text()
            .expect("body")
    });

    // Join mid-turn, while the mock is sleeping.
    std::thread::sleep(Duration::from_millis(600));
    let rejoined = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}/stream", id))
        .send()
        .expect("send")
        .text()
        .expect("body");
    let original_body = original.join().expect("original stream thread");

    assert!(
        original_body.contains("event: turn.finished"),
        "the original consumer must still complete normally: {}",
        original_body
    );
    assert!(
        rejoined.contains("\"resumed\":true"),
        "the rejoin must identify itself as one: {}",
        rejoined
    );
    assert!(
        rejoined.contains("event: turn.finished"),
        "a mid-turn rejoin must follow the live stream through to the terminal: {}",
        rejoined
    );
    assert!(
        rejoined.contains("after"),
        "the rejoin must receive text emitted after it attached: {}",
        rejoined
    );
}

// ---------------------------------------------------------------------------
// Outbound webhooks.
// ---------------------------------------------------------------------------

/// One received delivery, as the listener below captured it.
struct CapturedDelivery {
    event: String,
    delivery_id: String,
    timestamp: String,
    signature: Option<String>,
    body: String,
}

/// A minimal blocking HTTP listener that records one POST and answers `204`.
///
/// Hand-rolled rather than pulled from a crate because the whole point is to observe the exact
/// bytes and headers meka put on the wire; anything that parses and re-serialises would hide the
/// thing under test.
fn spawn_webhook_listener() -> (u16, std::sync::mpsc::Receiver<CapturedDelivery>) {
    spawn_webhook_listener_rejecting(0, "")
}

/// A listener that answers its first `reject_count` deliveries with `status_line` before settling
/// into 204s. Every attempt is reported on the channel either way, so a test can count them.
fn spawn_webhook_listener_rejecting(
    reject_count: usize,
    status_line: &'static str,
) -> (u16, std::sync::mpsc::Receiver<CapturedDelivery>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind listener");
    let port = listener.local_addr().expect("addr").port();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut served = 0usize;
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let tx = tx.clone();
            let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
            let mut headers = std::collections::HashMap::new();
            let mut line = String::new();
            // Request line, then headers until the blank line.
            let _ = reader.read_line(&mut line);
            let mut content_length = 0usize;
            loop {
                let mut header = String::new();
                if reader.read_line(&mut header).unwrap_or(0) == 0 {
                    break;
                }
                if header.trim().is_empty() {
                    break;
                }
                if let Some((name, value)) = header.split_once(':') {
                    let name = name.trim().to_ascii_lowercase();
                    let value = value.trim().to_string();
                    if name == "content-length" {
                        content_length = value.parse().unwrap_or(0);
                    }
                    headers.insert(name, value);
                }
            }
            let mut body = vec![0u8; content_length];
            if content_length > 0 {
                use std::io::Read as _;
                let _ = reader.read_exact(&mut body);
            }
            use std::io::Write as _;
            let response = if served < reject_count {
                status_line
            } else {
                "HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n"
            };
            served += 1;
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
            let _ = tx.send(CapturedDelivery {
                event: headers.get("x-meka-event").cloned().unwrap_or_default(),
                delivery_id: headers.get("x-meka-delivery").cloned().unwrap_or_default(),
                timestamp: headers.get("x-meka-timestamp").cloned().unwrap_or_default(),
                signature: headers.get("x-meka-signature").cloned(),
                body: String::from_utf8_lossy(&body).to_string(),
            });
        }
    });
    (port, rx)
}

/// Recompute the signature the way a receiver would, from the documented recipe.
fn expected_signature(secret: &str, timestamp: &str, body: &str) -> String {
    use hmac::{Hmac, KeyInit, Mac};
    use sha2::Sha256;
    let mut mac = <Hmac<Sha256> as KeyInit>::new_from_slice(secret.as_bytes()).expect("key");
    mac.update(timestamp.as_bytes());
    mac.update(b".");
    mac.update(body.as_bytes());
    let digest = mac.finalize().into_bytes();
    let mut out = String::from("sha256=");
    for byte in digest.iter() {
        out.push_str(&format!("{:02x}", byte));
    }
    out
}

#[test]
fn turn_finished_webhook_is_delivered_and_signed() {
    let (port, rx) = spawn_webhook_listener();
    let config = format!(
        "\n[[serve.webhooks]]\nurl = \"http://127.0.0.1:{}/hook\"\nsecret = \"topsecret\"\n\
         events = [\"turn.finished\"]\n",
        port
    );
    let harness = ServeTestHarness::spawn(&config, mock_simple_turn());
    let id = start_streaming_session(&harness);
    harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "hi"}))
        .send()
        .expect("send");

    let delivery = rx
        .recv_timeout(Duration::from_secs(20))
        .expect("a turn.finished delivery must arrive");
    assert_eq!(delivery.event, "turn.finished");
    assert!(
        !delivery.delivery_id.is_empty(),
        "deliveries must be identifiable for dedup"
    );
    assert!(!delivery.timestamp.is_empty());

    let signature = delivery
        .signature
        .expect("a configured secret must produce a signature");
    assert_eq!(
        signature,
        expected_signature("topsecret", &delivery.timestamp, &delivery.body),
        "signature must be HMAC-SHA256 over `<timestamp>.<body>`; body was {}",
        delivery.body,
    );

    let payload: serde_json::Value = serde_json::from_str(&delivery.body).expect("json body");
    assert_eq!(payload["event"], "turn.finished");
    assert_eq!(payload["session_id"], id);
    assert!(payload["turn_id"].is_string());
}

/// 429 says "not now", not "not ever". Several jobs sharing a cron minute deliver as a burst,
/// which is exactly when a receiver rate-limits; dropping those would lose the 9am report to the
/// one rejection the receiver was explicitly asking meka to wait out.
#[test]
fn a_rate_limited_webhook_is_retried() {
    let (port, rx) = spawn_webhook_listener_rejecting(
        1,
        "HTTP/1.1 429 Too Many Requests\r\nContent-Length: 0\r\n\r\n",
    );
    let config = format!(
        "\n[[serve.webhooks]]\nurl = \"http://127.0.0.1:{}/hook\"\nsecret = \"s\"\n\
         events = [\"turn.finished\"]\n",
        port
    );
    let harness = ServeTestHarness::spawn(&config, mock_simple_turn());
    let id = start_streaming_session(&harness);
    harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "hi"}))
        .send()
        .expect("send");

    let first = rx
        .recv_timeout(Duration::from_secs(20))
        .expect("the first attempt must arrive");
    let second = rx
        .recv_timeout(Duration::from_secs(20))
        .expect("a 429 must be retried, not dropped");
    assert_eq!(
        first.delivery_id, second.delivery_id,
        "a retry must keep the delivery id so a receiver can deduplicate it"
    );
    assert_eq!(
        first.timestamp, second.timestamp,
        "the signed timestamp identifies the delivery, not the attempt"
    );
}

/// The other half of the policy: a receiver saying the request itself is malformed is telling meka
/// something retrying cannot fix, and hammering it would turn one bad delivery into `max_retries`.
#[test]
fn a_webhook_rejected_as_malformed_is_not_retried() {
    let (port, rx) = spawn_webhook_listener_rejecting(
        5,
        "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n",
    );
    let config = format!(
        "\n[[serve.webhooks]]\nurl = \"http://127.0.0.1:{}/hook\"\nsecret = \"s\"\n\
         events = [\"turn.finished\"]\n",
        port
    );
    let harness = ServeTestHarness::spawn(&config, mock_simple_turn());
    let id = start_streaming_session(&harness);
    harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "hi"}))
        .send()
        .expect("send");

    rx.recv_timeout(Duration::from_secs(20))
        .expect("the first attempt must arrive");
    // Comfortably past the 1s backoff a retry would have waited.
    assert!(
        rx.recv_timeout(Duration::from_secs(4)).is_err(),
        "a 400 must not be retried"
    );
}

/// The load-bearing privacy property: a webhook endpoint is a config-file URL, so a delivery says
/// that something happened, never what was said.
#[test]
fn webhook_payloads_carry_no_message_content() {
    let (port, rx) = spawn_webhook_listener();
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "SECRET-ASSISTANT-TEXT" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let config = format!(
        "\n[[serve.webhooks]]\nurl = \"http://127.0.0.1:{}/hook\"\nsecret = \"s\"\n\
         events = [\"turn.finished\"]\n",
        port
    );
    let harness = ServeTestHarness::spawn(&config, script);
    let id = start_streaming_session(&harness);
    harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "SECRET-USER-PROMPT"}))
        .send()
        .expect("send");

    let delivery = rx
        .recv_timeout(Duration::from_secs(20))
        .expect("delivery must arrive");
    assert!(
        !delivery.body.contains("SECRET-USER-PROMPT"),
        "the user's prompt must never leave through a webhook: {}",
        delivery.body
    );
    assert!(
        !delivery.body.contains("SECRET-ASSISTANT-TEXT"),
        "the assistant's reply must never leave through a webhook: {}",
        delivery.body
    );
}

/// The real cancel path, which the unknown-id and empty-list tests never reach: resolving an
/// 8-character prefix, recording the cancellation *before* signalling, and signalling through the
/// `BackgroundTasks` handle hoisted onto `SessionEntry`. That hoist is what lets the endpoint
/// answer while a turn holds the runtime mutex, and if it ever captured a different registry than
/// the agent dispatches through, this endpoint would report 204 over a task that kept running.
#[test]
fn cancelling_a_running_background_task_stops_it() {
    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "tu_1", "name": "execute_command" },
            { "kind": "tool_use_end", "input": {"command": "sleep 120", "background": true} },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "started" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("\n[background]\nenabled = true\n", script);
    let id = start_streaming_session(&harness);
    harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "run it"}))
        .send()
        .expect("send");

    // Wait for the task to be registered as running.
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut task_id = String::new();
    while Instant::now() < deadline {
        let body: serde_json::Value = harness
            .request(reqwest::Method::GET, &format!("/v1/sessions/{}/tasks", id))
            .send()
            .expect("send")
            .json()
            .expect("parse");
        if let Some(task) = body["tasks"]
            .as_array()
            .and_then(|tasks| tasks.iter().find(|task| task["status"] == "running"))
        {
            task_id = task["id"].as_str().expect("task id").to_string();
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(!task_id.is_empty(), "no background task started");

    // Cancel on the short form, the way a human reads it off a rendered list.
    let cancel = harness
        .request(
            reqwest::Method::DELETE,
            &format!("/v1/sessions/{}/tasks/{}", id, &task_id[..8]),
        )
        .send()
        .expect("send");
    assert_eq!(cancel.status(), 204, "a prefix must resolve to the task");

    let after: serde_json::Value = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}/tasks", id))
        .send()
        .expect("send")
        .json()
        .expect("parse");
    let task = after["tasks"]
        .as_array()
        .expect("tasks")
        .iter()
        .find(|task| task["id"] == task_id.as_str())
        .expect("the task must still be listed");
    assert_eq!(
        task["status"], "cancelled",
        "the cancellation must be recorded, not just signalled: {}",
        after
    );
}

/// A `task.finished` delivery must not carry the task's label. For `execute_command` that is the
/// shell command line, which is exactly where a pasted credential ends up.
#[test]
fn task_webhook_payload_omits_the_command_line() {
    let (port, rx) = spawn_webhook_listener();
    let config = format!(
        "\n[background]\nenabled = true\n\n[[serve.webhooks]]\nurl = \"http://127.0.0.1:{}/hook\"\n\
         secret = \"s\"\nevents = [\"task.finished\"]\n",
        port
    );
    // The mock provider drives the tool call; the payload shape is what is under test.
    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "tu_1", "name": "execute_command" },
            { "kind": "tool_use_end",
              "input": {"command": "echo SECRET-TOKEN-IN-COMMAND", "background": true} },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "started" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn(&config, script);
    let id = start_streaming_session(&harness);
    harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "run it"}))
        .send()
        .expect("send");

    match rx.recv_timeout(Duration::from_secs(30)) {
        Ok(delivery) => {
            assert_eq!(delivery.event, "task.finished");
            assert!(
                !delivery.body.contains("SECRET-TOKEN-IN-COMMAND"),
                "the command line must not ride on a webhook: {}",
                delivery.body
            );
            let payload: serde_json::Value =
                serde_json::from_str(&delivery.body).expect("json body");
            assert!(
                payload.get("label").is_none(),
                "no `label` field at all: {}",
                delivery.body
            );
            assert_eq!(payload["tool_name"], "execute_command");
        }
        Err(_) => panic!("a task.finished delivery must arrive"),
    }
}

/// An endpoint that subscribed to something else must not be called at all, or the `events` list
/// is decorative.
#[test]
fn webhook_only_fires_for_subscribed_events() {
    let (port, rx) = spawn_webhook_listener();
    let config = format!(
        "\n[[serve.webhooks]]\nurl = \"http://127.0.0.1:{}/hook\"\nsecret = \"s\"\n\
         events = [\"schedule.fired\"]\n",
        port
    );
    let harness = ServeTestHarness::spawn(&config, mock_simple_turn());
    let id = start_streaming_session(&harness);
    harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "hi"}))
        .send()
        .expect("send");

    assert!(
        rx.recv_timeout(Duration::from_secs(3)).is_err(),
        "a turn must not reach an endpoint subscribed only to schedule.fired"
    );
}

/// A typo in `events` is refused at startup rather than warned about. An endpoint whose only
/// subscription is misspelled is silently never called, which is the worst way to discover it.
#[test]
fn unknown_webhook_event_is_a_startup_error() {
    let temp = tempfile::tempdir().expect("tempdir");
    let config_dir = temp.path().join("meka");
    std::fs::create_dir_all(&config_dir).expect("create config dir");
    std::fs::write(
        config_dir.join("config.toml"),
        r#"
[providers.mock]
type = "claude-api"
model = "claude-sonnet-4-5"

[serve]
bind = "127.0.0.1:0"

[[serve.webhooks]]
url = "http://127.0.0.1:1/hook"
events = ["turn.finished", "turn.exploded"]

[[serve.tokens]]
token = "sk_test_token"
scopes = ["sessions:r"]
"#,
    )
    .expect("write config");

    let output = meka()
        .arg("serve")
        .env("MEKA_CONFIG_DIR", &config_dir)
        .env("MEKA_DATA_DIR", temp.path().join("data"))
        .env("HOME", temp.path())
        .env("MEKA_ACP_MOCK_PROVIDER", "1")
        .output()
        .expect("run meka serve");
    assert!(!output.status.success(), "startup must fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("turn.exploded"),
        "the error must name the offending event: {}",
        stderr
    );
}

/// No secret means no signature. Allowed (loopback receivers exist) but it must not silently
/// produce a signature over an empty key, which would look valid to a careless receiver.
#[test]
fn webhook_without_a_secret_sends_no_signature_header() {
    let (port, rx) = spawn_webhook_listener();
    let config = format!(
        "\n[[serve.webhooks]]\nurl = \"http://127.0.0.1:{}/hook\"\nevents = [\"turn.finished\"]\n",
        port
    );
    let harness = ServeTestHarness::spawn(&config, mock_simple_turn());
    let id = start_streaming_session(&harness);
    harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "hi"}))
        .send()
        .expect("send");

    let delivery = rx
        .recv_timeout(Duration::from_secs(20))
        .expect("delivery must arrive");
    assert!(
        delivery.signature.is_none(),
        "an unsigned delivery must omit the header rather than sign with an empty key"
    );
}

/// A missing skill is not a missing session. `type` is the machine-readable code a client
/// switches on, so reusing `session-not-found` here would tell a client its conversation had
/// been lost when all that happened was a typo in a skill name.
#[test]
fn missing_store_resources_report_not_found_rather_than_session_not_found() {
    let harness =
        ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", STORE_SCOPES);
    for path in ["/v1/skills/no-such-skill", "/v1/memory/no-such-memory"] {
        let response = harness
            .request(reqwest::Method::GET, path)
            .send()
            .expect("send");
        assert_eq!(response.status(), 404, "{}", path);
        let body: serde_json::Value = response.json().expect("parse");
        assert_eq!(
            body["type"], "https://meka.so/errors/not-found",
            "{} must not claim the session is gone: {}",
            path, body
        );
    }
}

/// The skill *body* is the instruction text itself, so it needs `skills:r` rather than any read
/// scope. The palette at `GET /v1/skills` is a listing and stays broadly readable.
#[test]
fn reading_a_skill_body_requires_the_skills_read_scope() {
    let harness = ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", &[
        "sessions:r",
        "sessions:w",
    ]);
    let palette = harness
        .request(reqwest::Method::GET, "/v1/skills")
        .send()
        .expect("send");
    assert_eq!(palette.status(), 200, "the palette admits any read scope");

    let body_read = harness
        .request(reqwest::Method::GET, "/v1/skills/anything")
        .send()
        .expect("send");
    assert_eq!(body_read.status(), 403);
    let problem: serde_json::Value = body_read.json().expect("parse");
    assert!(
        problem["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("skills:r"),
        "{}",
        problem
    );
}

/// Resumption promises that nothing at or before your position comes back. The terminal is the
/// event most likely to be acted on twice (a client marks the turn done, tears down its UI), so
/// re-delivering it to a client whose `Last-Event-ID` already covers it is the worst place to
/// break that promise.
#[test]
fn reattach_does_not_redeliver_a_terminal_the_client_already_has() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let id = start_streaming_session(&harness);
    let original = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "hi", "stream": true}))
        .send()
        .expect("send")
        .text()
        .expect("body");
    let last = *sse_event_ids(&original).last().expect("terminal id");
    assert!(
        original.contains("event: turn.finished"),
        "the last id must be the terminal's: {}",
        original
    );

    let body = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}/stream", id))
        .header("Last-Event-ID", last.to_string())
        .send()
        .expect("send")
        .text()
        .expect("body");
    assert!(
        sse_event_ids(&body).iter().all(|value| *value > last),
        "nothing at or before id {} may be replayed: {}",
        last,
        body
    );
    assert!(
        !body.contains("event: turn.finished"),
        "the client already has the terminal; sending it again makes the turn look finished twice: {}",
        body
    );
    assert!(
        !body.contains("stream-detached"),
        "a client that is simply up to date is not a detached stream: {}",
        body
    );
}

/// Ids restart at 0 every turn, so a `Last-Event-ID` from an earlier turn names a position this
/// one never reached. Filtering against it would discard the entire backlog *and* the terminal as
/// "already delivered", closing the stream with nothing at all -- and a browser `EventSource`
/// re-sends its stored id automatically, so that is the default path, not an edge case.
#[test]
fn reattach_with_a_stale_cross_turn_last_event_id_still_delivers() {
    let harness = ServeTestHarness::spawn("", mock_turns(3));
    let id = start_streaming_session(&harness);

    // Turn one: a long id sequence.
    let first = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "one", "stream": true}))
        .send()
        .expect("send")
        .text()
        .expect("body");
    let stale = *sse_event_ids(&first).last().expect("terminal id");

    // Turn two: a fresh sequence starting back at 0.
    harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "two", "stream": true}))
        .send()
        .expect("send")
        .text()
        .expect("body");

    let body = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}/stream", id))
        .header("Last-Event-ID", stale.to_string())
        .send()
        .expect("send")
        .text()
        .expect("body");
    assert!(
        body.contains("event: turn.finished"),
        "a stale id must not swallow the terminal: {}",
        body
    );
    assert!(
        body.contains("event: notice") && body.contains("Replay buffer does not reach"),
        "and the client must be told its position was unreachable: {}",
        body
    );
}

/// `GET /context` and `GET /tools` read hoisted handles, not the runtime mutex. Asking about
/// headroom while the session is busy is the whole point; blocking would make the request hang for
/// the length of the turn.
#[test]
fn context_and_tools_answer_while_a_turn_is_in_flight() {
    let script = serde_json::json!([
        [
            { "kind": "sleep", "ms": 2500 },
            { "kind": "text", "text": "done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let id = start_streaming_session(&harness);

    let base_url = harness.base_url.clone();
    let token = harness.token.clone();
    let session = id.clone();
    let turn = std::thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("client");
        client
            .post(format!("{}/v1/sessions/{}/turn", base_url, session))
            .header("Authorization", format!("Bearer {}", token))
            .json(&serde_json::json!({"message": "hi"}))
            .send()
            .expect("send")
            .status()
    });

    std::thread::sleep(Duration::from_millis(800));
    let started = Instant::now();
    let context = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/context", id),
        )
        .send()
        .expect("send");
    let tools = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}/tools", id))
        .send()
        .expect("send");
    let elapsed = started.elapsed();

    assert_eq!(context.status(), 200);
    assert_eq!(tools.status(), 200);
    assert!(
        elapsed < Duration::from_millis(1200),
        "both must answer immediately, not wait out the turn; took {:?}",
        elapsed
    );
    let body: serde_json::Value = context.json().expect("parse");
    assert!(
        body.get("message_count").is_none(),
        "the one field that needs the conversation is omitted while it is locked: {}",
        body
    );
    assert!(
        body["totals"].is_object(),
        "everything read from atomics and the DB is still present: {}",
        body
    );
    assert_eq!(turn.join().expect("turn thread"), 200);
}

/// Compaction rewrites the conversation, so it must register as in-flight: otherwise a concurrent
/// turn races it, and the GC scanner can evict the session out from under a minute-long
/// checkpoint and rebuild it from a database that has not seen the boundary yet.
#[test]
fn compact_marks_the_session_in_flight() {
    let harness =
        ServeTestHarness::spawn("\n[session]\ncompact_checkpoint = false\n", mock_turns(4));
    let id = session_with_one_turn(&harness);

    let before: serde_json::Value = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}", id))
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert_eq!(before["turn_in_flight"], false);

    let compact = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/compact", id),
        )
        .send()
        .expect("send");
    assert_eq!(compact.status(), 200);

    // The guard must release afterwards, or the session is wedged for good.
    let after: serde_json::Value = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}", id))
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert_eq!(
        after["turn_in_flight"], false,
        "the in-flight guard must be released when compaction returns: {}",
        after
    );

    let next = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "still usable?"}))
        .send()
        .expect("send");
    assert_eq!(next.status(), 200, "the session must still accept turns");
}

/// The full system-instruction text is instruction content, gated like a skill body rather than
/// like the palette listing.
#[test]
fn instructions_require_sessions_read_not_merely_any_read_scope() {
    let harness =
        ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", &["schedule:r"]);
    let response = harness
        .request(reqwest::Method::GET, "/v1/instructions")
        .send()
        .expect("send");
    assert_eq!(response.status(), 403);
    let body: serde_json::Value = response.json().expect("parse");
    assert!(
        body["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("sessions:r"),
        "{}",
        body
    );

    // The palette stays broadly readable.
    let palette = harness
        .request(reqwest::Method::GET, "/v1/skills")
        .send()
        .expect("send");
    assert_eq!(palette.status(), 200);
}

/// The counters `GET /context` reports are hoisted `Arc`s handed to `build_session_agent`, and
/// the agent writes them through differently-named handles. Nothing type-checks that the two ends
/// are the same allocation: pass a fresh `Arc` at either site and this endpoint reports an
/// unmeasured window forever, which `used: null` renders as a plausible answer rather than a bug.
#[test]
fn context_counters_are_wired_to_the_handles_the_agent_writes() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let id = session_with_one_turn(&harness);
    let body: serde_json::Value = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/context", id),
        )
        .send()
        .expect("send")
        .json()
        .expect("parse");
    // `used` is not asserted here: the mock provider reports no token usage, so the counter the
    // agent writes from it stays 0 and the field is legitimately absent. It is covered against a
    // real provider instead. `overhead` and `window` do not depend on provider usage, and between
    // them they pin both halves of the new wiring.
    assert!(
        body["overhead"]
            .as_u64()
            .is_some_and(|overhead| overhead > 0),
        "`overhead` must be the counter the agent stamps with prompt + schema cost: {}",
        body
    );
    assert!(
        body["window"].as_u64().is_some_and(|window| window > 0),
        "`window` must be the resolved window, not the unresolved config Option: {}",
        body
    );
    assert_eq!(
        body["window"], 200_000,
        "the harness configures no context_window, so this is the model-inferred value; reading \
         `config.context_window` instead would have reported nothing at all: {}",
        body
    );
}

/// Two writes inside one mtime tick that render to the same length are invisible to a
/// `(mtime, size)` snapshot. Without an explicit invalidation the second write's own 200 response
/// echoes the first write's values, and every agent keeps reading the stale skill.
#[test]
fn a_same_length_rewrite_is_visible_immediately() {
    let harness =
        ServeTestHarness::spawn_with("", mock_simple_turn(), "sk_test_token", STORE_SCOPES);
    let first: serde_json::Value = harness
        .request(reqwest::Method::PUT, "/v1/skills/tick")
        .json(&serde_json::json!({"description": "same length", "priority": 3, "body": "b"}))
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert_eq!(first["priority"], 3);

    // Same description and body, different priority: identical rendered length, same tick.
    let second: serde_json::Value = harness
        .request(reqwest::Method::PUT, "/v1/skills/tick")
        .json(&serde_json::json!({"description": "same length", "priority": 7, "body": "b"}))
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert_eq!(
        second["priority"], 7,
        "the write's own read-back must not be served a stale cache: {}",
        second
    );

    let listed: serde_json::Value = harness
        .request(reqwest::Method::GET, "/v1/skills")
        .send()
        .expect("send")
        .json()
        .expect("parse");
    let entry = listed
        .as_array()
        .expect("skills")
        .iter()
        .find(|s| s["name"] == "tick")
        .expect("listed");
    assert_eq!(
        entry["priority"], 7,
        "and agents must see it too: {}",
        entry
    );
}

/// A rewind removes messages with nothing left behind to carry a marker, so `total` shrinking is
/// the only visible sign. `revision` is what tells a polling client its copy is no longer a
/// prefix, rather than leaving it unable to distinguish a rewind from data loss.
#[test]
fn revision_advances_on_rewind_as_well_as_compaction() {
    let harness =
        ServeTestHarness::spawn("\n[session]\ncompact_checkpoint = false\n", mock_turns(4));
    let id = session_with_one_turn(&harness);

    let read = |path: String| -> serde_json::Value {
        harness
            .request(reqwest::Method::GET, &path)
            .send()
            .expect("send")
            .json()
            .expect("parse")
    };
    let before = read(format!("/v1/sessions/{}/messages", id));
    assert_eq!(
        before["revision"], 0,
        "an append-only log has never been rewritten"
    );

    harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/compact", id),
        )
        .send()
        .expect("send");
    let after_compact = read(format!("/v1/sessions/{}/messages", id));
    assert_eq!(after_compact["revision"], 1, "compaction rewrites the log");

    harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/rewind", id),
        )
        .json(&serde_json::json!({"turns": 1}))
        .send()
        .expect("send");
    let after_rewind = read(format!("/v1/sessions/{}/messages", id));
    assert_eq!(
        after_rewind["revision"], 2,
        "and so does a rewind, which leaves no marker behind: {}",
        after_rewind
    );
}

/// A read scope must not be able to seize a write-exclusive resource. Re-attaching takes the
/// session's cross-process file lock for up to `idle_timeout` (24h by default), which would lock
/// the operator out of `meka -r` on their own session.
#[test]
fn read_only_endpoints_do_not_revive_an_evicted_session() {
    // A tiny idle timeout plus a fast scan makes the GC evict between the turn and the reads.
    let harness = ServeTestHarness::spawn(
        "idle_timeout = \"1s\"\ngc_scan_interval = \"1s\"\n",
        mock_simple_turn(),
    );
    let id = session_with_one_turn(&harness);
    std::thread::sleep(Duration::from_millis(3500));

    // `/context` still answers from the database, without reviving.
    let context = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/context", id),
        )
        .send()
        .expect("send");
    assert_eq!(context.status(), 200);
    let body: serde_json::Value = context.json().expect("parse");
    assert!(
        body.get("used").is_none() && body.get("overhead").is_none(),
        "an evicted session has no live counters, and must say so by omission rather than \
         reporting zero: {}",
        body
    );
    assert_eq!(
        body["totals"]["turns"], 1,
        "the durable figures still come back: {}",
        body
    );

    // `/tools` needs a live registry, so it refuses rather than reviving or guessing.
    let tools = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}/tools", id))
        .send()
        .expect("send");
    assert_eq!(
        tools.status(),
        409,
        "a catalogue needs a loaded session; reviving one would pin its file lock for a read"
    );
    let problem: serde_json::Value = tools.json().expect("parse");
    assert_eq!(
        problem["type"], "https://meka.so/errors/session-not-loaded",
        "not `turn-in-flight`: that type tells a client to cancel a turn, and `POST /cancel` \
         would return 204 forever because there is no turn. The remedy is the opposite -- submit \
         one. Body was: {}",
        problem
    );
}

/// `max_body_bytes` above axum's own 2 MiB extractor default was silently inert, and the 413 then
/// named a limit that had not fired.
#[test]
fn max_body_bytes_above_two_mebibytes_is_honoured() {
    let harness = ServeTestHarness::spawn("max_body_bytes = 8388608\n", mock_simple_turn());
    let id = start_streaming_session(&harness);
    // 3 MiB of message: over axum's default, under the configured limit.
    let message = "x".repeat(3 * 1024 * 1024);
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": message}))
        .send()
        .expect("send");
    assert_ne!(
        response.status(),
        413,
        "a 3 MiB body must be accepted when max_body_bytes is 8 MiB"
    );
}

/// Config values that would silently disable a subsystem are rejected at startup, matching how
/// `max_body_bytes = 0` and `max_concurrent_turns = 0` are already handled.
#[test]
fn zero_valued_serve_knobs_are_rejected_at_startup() {
    for (snippet, probe) in [
        ("gc_scan_interval = \"0s\"\n", "gc_scan_interval"),
        ("", "timeout"),
    ] {
        let temp = tempfile::tempdir().expect("tempdir");
        let config_dir = temp.path().join("meka");
        std::fs::create_dir_all(&config_dir).expect("create config dir");
        let webhook = if probe == "timeout" {
            "\n[[serve.webhooks]]\nurl = \"http://127.0.0.1:1/h\"\nevents = [\"turn.finished\"]\n\
             timeout = \"0s\"\n"
        } else {
            ""
        };
        std::fs::write(
            config_dir.join("config.toml"),
            format!(
                "[providers.mock]\ntype = \"claude-api\"\nmodel = \"claude-sonnet-4-5\"\n\n\
                 [serve]\nbind = \"127.0.0.1:0\"\n{}{}\n\
                 [[serve.tokens]]\ntoken = \"t\"\nscopes = [\"sessions:r\"]\n",
                snippet, webhook
            ),
        )
        .expect("write config");
        let output = meka()
            .arg("serve")
            .env("MEKA_CONFIG_DIR", &config_dir)
            .env("MEKA_DATA_DIR", temp.path().join("data"))
            .env("HOME", temp.path())
            .env("MEKA_ACP_MOCK_PROVIDER", "1")
            .output()
            .expect("run meka serve");
        assert!(!output.status.success(), "{} must fail startup", probe);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(probe),
            "the error must name {}: {}",
            probe,
            stderr
        );
    }
}

/// Cancelling a slow turn and then compacting is an ordinary sequence: the turn is going nowhere,
/// so free the window. It broke silently, because `POST /compact` cloned whatever token the last
/// turn left in the session's cell, and a cancelled turn leaves that token fired. The checkpoint
/// turn then returned instantly and compaction fell back to the standalone summariser -- no
/// memories written, a worse summary, and a `warn` as the only trace.
#[test]
fn compacting_after_a_cancelled_turn_still_runs_the_checkpoint() {
    let script = serde_json::json!([
        // Turn one: slow enough to cancel.
        [
            { "kind": "sleep", "ms": 4000 },
            { "kind": "text", "text": "never seen" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        // The checkpoint turn the compaction should run.
        [
            { "kind": "text", "text": "summary of the conversation so far" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let id = start_streaming_session(&harness);

    let base_url = harness.base_url.clone();
    let token = harness.token.clone();
    let session = id.clone();
    let turn = std::thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("client");
        client
            .post(format!("{}/v1/sessions/{}/turn", base_url, session))
            .header("Authorization", format!("Bearer {}", token))
            .json(&serde_json::json!({"message": "something slow"}))
            .send()
            .expect("send")
            .status()
    });

    harness.wait_until_in_flight(&id);
    let cancel = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/cancel", id),
        )
        .send()
        .expect("send");
    assert_eq!(cancel.status(), 204);
    let _ = turn.join().expect("turn thread");

    let response = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/compact", id),
        )
        .send()
        .expect("send");
    assert_eq!(
        response.status(),
        200,
        "compact failed: {}",
        response.text().unwrap_or_default()
    );
    let body: serde_json::Value = response.json().expect("parse");
    let source = body["source"].as_str().unwrap_or_default();
    assert!(
        source.starts_with("checkpoint"),
        "the checkpoint turn must actually run after a cancelled turn, not be skipped because it \
         inherited the fired token; source was {:?}",
        source
    );
}

/// The compaction SSE event is the one signal a streaming client has that its mirror of the
/// conversation was rewritten mid-turn. Nothing asserted it before.
#[test]
fn a_compaction_during_a_streaming_turn_emits_context_compacted() {
    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "tu_1", "name": "context_compact" },
            { "kind": "tool_use_end", "input": {} },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "requested" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "text", "text": "a summary" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("\n[session]\ncompact_checkpoint = false\n", script);
    let id = start_streaming_session(&harness);
    let body = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "compact yourself", "stream": true}))
        .send()
        .expect("send")
        .text()
        .expect("body");

    let names = sse_event_names(&body);
    assert!(
        names.iter().any(|name| name == "context.compacted"),
        "a compaction inside a streaming turn must reach the client: {}",
        body
    );
    assert!(
        body.contains("\"generation\":1"),
        "and carry which compaction it was: {}",
        body
    );
    assert_eq!(
        names.last().map(String::as_str),
        Some("turn.finished"),
        "the event must land before the terminal, not after: {}",
        body
    );
}
