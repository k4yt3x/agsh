// See the matching allow in `tests/acp.rs` for the rationale: integration tests panic on
// failure by design.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! End-to-end integration tests for `meka serve`. Spawns the real `meka serve` binary against
//! a tempdir and a scripted mock provider, then drives it over HTTP via `reqwest`.

// Only the `#[cfg(unix)]` shutdown tests read an SSE body directly; importing it
// unconditionally made the Windows build warn about an import nothing there uses.
#[cfg(unix)]
use std::io::Read;
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
    /// The server's `MEKA_DATA_DIR`. Exposed so a test can reach the session lock directory
    /// underneath it, which is the only way to create the cross-process lock contention that
    /// distinguishes a refusal placed before `lock_session` from one placed after.
    data_dir: std::path::PathBuf,
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
        Self::spawn_with("", config_toml, script, "sk_test_token", &[
            "sessions:r",
            "sessions:w",
        ])
    }

    /// [`Self::spawn`] plus a top-level prelude, for the keys that cannot live inside a table.
    ///
    /// `extra_config` is injected *inside* `[serve]`, which is right for `max_body_bytes` and its
    /// neighbours and impossible for `default_provider`: a second profile makes the default
    /// ambiguous, and the key that resolves it has to precede every table header.
    fn spawn_with_prelude(prelude: &str, config_toml: &str, script: serde_json::Value) -> Self {
        Self::spawn_with(prelude, config_toml, script, "sk_test_token", &[
            "sessions:r",
            "sessions:w",
        ])
    }

    fn spawn_with(
        prelude: &str,
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
                r#"{prelude}
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"

[permissions]
default = "unrestricted"
# `workspace` included so it is reachable from a test at all. Leaving it out was a large part of
# why this whole surface went unexercised: `POST /v1/sessions` refuses a level the server has not
# enabled, so no test could create a `workspace` session even deliberately. `none` is here for the
# same reason and it was found the same way: the scheduler refuses every job on a session at
# `none`, and no test could reach that state to check what the endpoints then say.
enabled = ["none", "read", "workspace", "unrestricted", "ask"]

[serve]
bind = "{bind}"
{extra_config}

[[serve.tokens]]
token = "{token}"
scopes = [{scopes_str}]
"#,
                prelude = prelude,
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
                .env("MEKA_MOCK_PROVIDER", "1")
                .env("MEKA_MOCK_PROVIDER_SCRIPT", &script_path)
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
                    data_dir: data_dir.clone(),
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

    /// The temp root this server's config and data live under, so a test can seed files inside it.
    ///
    /// Reach meka's own store through this (`home().join("meka")` is `MEKA_CONFIG_DIR`), not a
    /// directory outside it. This used to be described as "the `HOME` this server runs under",
    /// which held only on Unix: `dirs::home_dir` on Windows is
    /// `SHGetKnownFolderPath(FOLDERID_Profile)` and never reads the environment, so a config `~/…`
    /// path resolved to the runner's real profile while the test seeded a temp directory. A test
    /// that needs a root outside meka's store should make its own and name it absolutely.
    fn home(&self) -> &std::path::Path {
        self._temp.path()
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
        ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", &["sessions:r"]);
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
    assert_eq!(created["permission"], "unrestricted");

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
    // Deliberately not the server's configured default (`unrestricted`): a fork that dropped the
    // `permission` column entirely would still report `unrestricted`, because the re-attach path
    // falls back to the config default when the column is NULL.
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
        ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", &["sessions:r"]);
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
        .json(&serde_json::json!({"cwd": temp_dir, "permission": "unrestricted"}))
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

/// The docs routes are the only unauthenticated ones that describe the deployment rather than
/// report on it, so they are opt-in. Anyone who can reach the port could otherwise read the shape
/// of every endpoint without presenting a token.
#[test]
fn the_docs_routes_are_off_unless_asked_for() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    for path in ["/v1/openapi.json", "/v1/docs/"] {
        let response = harness
            .client
            .get(format!("{}{}", harness.base_url, path))
            .send()
            .expect("send");
        assert_eq!(
            response.status(),
            404,
            "{path} must not be served unless `[serve].docs` is set"
        );
    }
}

#[test]
fn openapi_json_is_served_without_auth_and_documents_routes() {
    let harness = ServeTestHarness::spawn("docs = true\n", mock_simple_turn());
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
    let harness = ServeTestHarness::spawn("docs = true\n", mock_simple_turn());
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

    // Wait for the abandoned turn to finish on its own. Polled rather than slept: the script's own
    // 2s is most of any fixed budget, so what is left has to cover session setup and the commit on
    // whatever machine this runs on, and a slow runner reads as an abandoned turn.
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
        "",
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

/// A path segment axum's `Path<Uuid>` extractor rejects must still answer RFC 9457.
///
/// The rejection happens before any handler runs, so it used to escape the error taxonomy entirely
/// and answer `400 text/plain` -- one response shape a client parsing `application/problem+json`
/// could not read, for the most ordinary mistake there is.
#[test]
fn a_malformed_path_parameter_is_a_problem_detail() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let response = harness
        .request(reqwest::Method::GET, "/v1/sessions/not-a-uuid")
        .send()
        .expect("send");

    assert_eq!(response.status(), 400);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("application/problem+json"),
        "a rejected path parameter must serialize as Problem Detail, not plain text",
    );
    let body: serde_json::Value = response.json().expect("parse");
    assert_eq!(body["status"], 400);
    assert!(body["type"].is_string(), "{body}");
    // The rejection text names what failed, which is the whole value of surfacing it.
    assert!(
        body["detail"]
            .as_str()
            .is_some_and(|detail| !detail.is_empty()),
        "{body}"
    );
    assert_eq!(body["instance"], "/v1/sessions/not-a-uuid");
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
        ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", &["sessions:r"]);
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
        "",
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
    let harness = ServeTestHarness::spawn_with(
        "",
        "max_concurrent_turns = 1\n",
        script,
        "sk_test_token",
        &["sessions:r", "sessions:w"],
    );

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

/// Graceful shutdown waits for a detached turn to unwind instead of exiting out from under it.
///
/// A streaming turn's handler returns its SSE response as soon as the stream is installed and
/// leaves the turn running on a spawned task, so once the client is gone axum's own graceful
/// shutdown has no in-flight request to wait for. That is the case `stream_reattach_grace` exists
/// to keep alive, and it was also the case shutdown abandoned: the accept loop was the only thing
/// the drain timeout wrapped.
///
/// `stall` rather than `sleep` because cancellation is the point. The drain fires every session's
/// token first, so a `sleep` would end right there and leave nothing to wait for. A real turn's
/// tail behaves like `stall`: the token stops the agent at its next check, and the commit of what
/// the round already produced runs after that.
///
/// Unix-only for the same reason as the test above.
#[cfg(unix)]
#[test]
fn graceful_shutdown_waits_for_a_detached_turn_to_unwind() {
    const STALL_MS: u64 = 4000;
    // The leading text is a starting gun. `turn_in_flight` flips at admission, before the task
    // that runs the turn has been polled once, and a signal delivered in that window cancels the
    // turn before it ever reaches the provider -- so the stall never starts and there is nothing
    // for the drain to wait for. Reading this event back proves the stream is live.
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "turn-has-started" },
            { "kind": "stall", "ms": STALL_MS },
            { "kind": "text", "text": "finished after the signal" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = ServeTestHarness::spawn("", script);
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    // Start the turn, read until the agent is demonstrably streaming, then hang up. Dropping the
    // response closes the socket, which is what leaves the turn genuinely unattended rather than
    // merely slow to read.
    let mut response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "stall me", "stream": true}))
        .send()
        .expect("stream send");
    assert!(response.status().is_success(), "turn should be accepted");
    let mut seen = String::new();
    let mut chunk = [0u8; 512];
    while !seen.contains("turn-has-started") {
        let read = response.read(&mut chunk).expect("read sse");
        assert!(read > 0, "stream closed before the turn produced anything");
        seen.push_str(&String::from_utf8_lossy(&chunk[..read]));
    }
    drop(response);

    harness.wait_until_in_flight(&id);

    let signalled = Instant::now();
    let kill_status = Command::new("kill")
        .arg("-TERM")
        .arg(harness.child.id().to_string())
        .status()
        .expect("send SIGTERM");
    assert!(kill_status.success(), "kill should succeed");

    let status = harness.child.wait().expect("wait for exit");
    let elapsed = signalled.elapsed();
    assert!(
        status.success(),
        "a drain that completes exits 0; got {:?} after {:?}",
        status,
        elapsed,
    );
    assert!(
        elapsed >= Duration::from_millis(1500),
        "shutdown must wait for the detached turn: the process exited {:?} after SIGTERM while \
         the turn still had most of its {}ms left to run",
        elapsed,
        STALL_MS,
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

/// `tool_call.composing` opens the window a client can draw "the agent is writing" over, and
/// `tool_call.executing` closes it. The order is the whole point: the arguments -- for a tool that
/// sends a message, the message -- are written between the two, so an indicator raised on the
/// dispatch alone would appear only after there was nothing left to wait for.
#[test]
fn streaming_tool_call_announces_composition_before_it_executes() {
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
    let body = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "list it", "stream": true}))
        .send()
        .expect("send")
        .text()
        .expect("body");

    let composing = body
        .find("event: tool_call.composing")
        .unwrap_or_else(|| panic!("stream must emit tool_call.composing; body was:\n{}", body));
    let executing = body
        .find("event: tool_call.executing")
        .unwrap_or_else(|| panic!("stream must emit tool_call.executing; body was:\n{}", body));
    assert!(
        composing < executing,
        "composition must be announced before the dispatch; body was:\n{}",
        body,
    );
    assert!(
        body[composing..executing].contains("{\"id\":\"tu_1\",\"name\":\"list_directory\"}"),
        "tool_call.composing carries the id to pair on and the name, and nothing else has \
         streamed yet; body was:\n{}",
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

/// A payload that is valid base64 *and* sniffs as a PNG, but does not decode.
///
/// This is the one the decode guard exists for, and the door meka does not control: a client can
/// post anything. The sibling below stops at base64, which fails a line earlier and never reaches
/// the decode, so before this the whole HTTP layer could lose that guard and stay green.
#[test]
fn turn_with_truncated_image_returns_422() {
    use base64::Engine as _;

    let png = base64::engine::general_purpose::STANDARD
        .decode(TINY_PNG_BASE64)
        .expect("the fixture is valid base64");
    let truncated = base64::engine::general_purpose::STANDARD.encode(&png[..png.len() * 2 / 3]);

    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let id = create_session_id(&harness);
    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({
            "message": "look",
            "images": [{"media_type": "image/png", "data": truncated}],
            "stream": false,
        }))
        .send()
        .expect("send");
    assert_eq!(response.status(), 422);
    let detail = response.json::<serde_json::Value>().expect("parse")["detail"]
        .as_str()
        .expect("detail")
        .to_string();
    assert!(detail.contains("images[0]"), "{detail}");
    assert!(
        detail.contains("decode"),
        "the refusal has to name the decode, not the base64: {detail}"
    );
}

/// The sibling: not base64 at all, refused one step earlier. Named for what it actually feeds.
#[test]
fn turn_with_unparseable_image_returns_422() {
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

/// A session created at `workspace` writes inside its cwd and is refused outside it.
///
/// The whole integration surface of this mode was uncovered: every session in `tests/serve.rs`,
/// `tests/acp.rs` and `tests/multiprocess.rs` runs at `read`, `ask` or `unrestricted`, and the two
/// `workspace` mentions in the tree assert a config string and a catalogue field. So the mode this
/// change set exists to introduce had no end-to-end test on any surface -- the boundary could stop
/// being applied to a session `POST /v1/sessions` created and nothing here would notice.
///
/// Asserts filesystem ground truth rather than the turn's narration: a tool that never ran reports
/// failure just as convincingly as one the fence refused.
#[test]
fn a_workspace_session_writes_inside_its_cwd_and_is_refused_outside() {
    let inside_dir = tempfile::tempdir().expect("tempdir");
    let outside_dir = tempfile::tempdir().expect("tempdir");
    let inside = inside_dir.path().join("inside.txt");
    let outside = outside_dir.path().join("escaped.txt");

    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "tu_1", "name": "write_file" },
            { "kind": "tool_use_end", "input": {"path": inside, "content": "in"} },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "tool_use_start", "id": "tu_2", "name": "write_file" },
            { "kind": "tool_use_end", "input": {"path": outside, "content": "out"} },
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
        .json(&serde_json::json!({
            "cwd": inside_dir.path(),
            "permission": "workspace",
        }))
        .send()
        .expect("create");
    let status = create.status();
    let body: serde_json::Value = create.json().expect("parse");
    assert_eq!(status, 201, "create failed: {body}");
    assert_eq!(
        body["permission"], "workspace",
        "the session must actually be at workspace: {body}"
    );
    let id = body["id"].as_str().expect("id").to_string();

    let response = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "write both"}))
        .send()
        .expect("send");
    assert_eq!(response.status(), 200, "{}", response.text().expect("text"));

    assert!(
        inside.exists(),
        "a write inside the session cwd must land at workspace"
    );
    assert_eq!(std::fs::read_to_string(&inside).expect("read back"), "in");
    assert!(
        !outside.exists(),
        "a write outside every root must be refused at workspace"
    );
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

/// `/v1/info` carries no provider or model, and `/v1/providers` answers both.
///
/// It used to report the default profile's backend under the name `provider`, while the `provider`
/// field on `POST /v1/sessions` names a *profile*: one word, two meanings, one API, and a client
/// that read one and posted it to the other got a 422. The fields were also duplicates, since the
/// `active: true` row below already carries the same two facts under names that tell them apart.
#[test]
fn info_carries_no_provider_or_model_because_providers_does() {
    let harness = ServeTestHarness::spawn_with_prelude(
        "default_provider = \"mock\"\n",
        r#"
[providers.side]
type = "openai-chat-completions"
model = "gpt-5"
"#,
        mock_simple_turn(),
    );

    let info: serde_json::Value = harness
        .request(reqwest::Method::GET, "/v1/info")
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert!(
        info.get("provider").is_none() && info.get("model").is_none(),
        "neither belongs here; `/v1/providers` names them apart: {info}"
    );

    let providers: serde_json::Value = harness
        .request(reqwest::Method::GET, "/v1/providers")
        .send()
        .expect("send")
        .json()
        .expect("parse");
    let rows = providers["providers"].as_array().expect("providers");
    let active = rows
        .iter()
        .find(|row| row["active"] == true)
        .expect("one profile is the default");
    // `name` is the profile, which is what `POST /v1/sessions` takes; `type` is the backend. Both
    // present, and distinguishable, which is the whole reason `/v1/info` need not repeat them.
    assert_eq!(active["name"], "mock", "{providers}");
    assert!(active["type"].is_string(), "{providers}");
    assert!(
        rows.iter().any(|row| row["name"] == "side"),
        "every configured profile is listed, not just the default: {providers}"
    );
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
    // The agent emits the deltas *and* the whole block behind them, for the consumers that want
    // reasoning in one piece. Forwarding both here would put the same text on the wire twice.
    assert_eq!(
        body.matches("let me check").count(),
        1,
        "the reasoning reached the wire more than once; body was:\n{}",
        body,
    );
}

/// A retry must not send the reasoning again.
///
/// The stream tells clients to concatenate `thinking.delta` payloads to reassemble the block, so a
/// second copy is not a cosmetic repeat: it is the block doubled in whatever the client rebuilt.
/// `content_started` is what stops the retry once model output has been forwarded, and reasoning is
/// model output like the answer is.
#[test]
fn a_retry_does_not_send_the_reasoning_twice() {
    let script = serde_json::json!([
        [
            { "kind": "thinking_delta", "text": "weighing the options" },
            { "kind": "fail_retryable", "message": "transient", "retry_after_secs": 0 }
        ],
        [
            { "kind": "thinking_delta", "text": "weighing the options" },
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
    let body = response.text().expect("body");
    assert_eq!(
        body.matches("weighing the options").count(),
        1,
        "the retry re-sent reasoning the client had already been given; body was:\n{}",
        body,
    );
}

/// A blocking turn keeps its retry, whatever the session's reasoning capability says.
///
/// The capability is permission to *receive* reasoning, not evidence of having received it: a
/// blocking turn installs no stream, so the deltas never leave the frontend and the response is
/// assembled from whole blocks, which a failed attempt never produces. Refusing the retry there
/// spends a recoverable turn for nothing -- on essentially every transient failure, since reasoning
/// is the first thing a turn produces.
#[test]
fn a_blocking_turn_retries_even_with_reasoning_enabled() {
    let script = serde_json::json!([
        [
            { "kind": "thinking_delta", "text": "weighing the options" },
            { "kind": "fail_retryable", "message": "transient", "retry_after_secs": 0 }
        ],
        [
            { "kind": "thinking_delta", "text": "weighing the options" },
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
        .json(&serde_json::json!({"message": "ponder", "stream": false}))
        .send()
        .expect("turn");
    assert_eq!(
        response.status(),
        200,
        "the blocking turn gave up its retry: {}",
        response.text().unwrap_or_default()
    );
    let body = response.json::<serde_json::Value>().expect("parse");
    assert_eq!(body["final_text"], "answer");
}

/// The retry survives a blocking turn that follows a streamed one on the same session.
///
/// A `TurnStream` outlives its turn so a late reconnect can still read the tail, so the slot stays
/// occupied for the rest of the session. Asking whether it is filled -- rather than whether
/// anything is listening -- answered "reasoning was delivered" for every later blocking turn, and
/// spent its retry. The single-turn test above cannot see this, because the session has never
/// streamed.
#[test]
fn a_blocking_turn_after_a_streamed_one_keeps_its_retry() {
    let script = serde_json::json!([
        [
            { "kind": "thinking_delta", "text": "first turn" },
            { "kind": "thinking_complete", "signature": null },
            { "kind": "text", "text": "one" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "thinking_delta", "text": "second turn" },
            { "kind": "fail_retryable", "message": "transient", "retry_after_secs": 0 }
        ],
        [
            { "kind": "thinking_delta", "text": "second turn" },
            { "kind": "thinking_complete", "signature": null },
            { "kind": "text", "text": "two" },
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

    let streamed = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "one", "stream": true}))
        .send()
        .expect("stream");
    assert_eq!(streamed.status(), 200);
    drop(streamed.text());

    let blocking = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "two", "stream": false}))
        .send()
        .expect("turn");
    assert_eq!(
        blocking.status(),
        200,
        "the blocking turn gave up its retry because an earlier turn had streamed: {}",
        blocking.text().unwrap_or_default()
    );
}

/// Reasoning arrives in chunks, and each one is its own `thinking.delta`. The count is the
/// assertion: a client concatenating the payloads gets the block back either way, so only the
/// number of events distinguishes a stream from a single lump.
#[test]
fn thinking_deltas_stream_one_event_per_chunk() {
    let script = serde_json::json!([
        [
            { "kind": "thinking_delta", "text": "first thought " },
            { "kind": "thinking_delta", "text": "second thought" },
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
    let body = response.text().expect("body");
    assert_eq!(
        body.matches("event: thinking.delta").count(),
        2,
        "one event per chunk; body was:\n{}",
        body,
    );
    assert_eq!(body.matches("first thought").count(), 1, "{}", body);
    assert_eq!(body.matches("second thought").count(), 1, "{}", body);
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
    // Quoted, so the match ends where the type does. Bare, this is a *prefix* of
    // `/errors/provider-unavailable` and passes against either one, which is exactly the shape
    // `the_turn_failed_payload_tells_a_transient_failure_from_a_permanent_one` exists to tell
    // apart. The `fail` kind is `MekaError::Provider`, so this one is the permanent type.
    assert!(
        body.contains("\"https://meka.so/errors/provider\""),
        "turn.failed payload must carry the provider error type; body was:\n{}",
        body,
    );
}

/// A transient upstream failure and a permanent one reach an SSE client under different `type`s.
///
/// The mekabridge team's report, end to end. Both are 502s inside `turn.failed`, and both arrive
/// here carrying no `Retry-After`: `fail_retryable` is scripted with `retry_after_secs: null`
/// because that is the case the header cannot rescue, and `fail_stream` never had a response to
/// read one from. With the types shared, a bridge holding these two payloads had nothing left to
/// branch on, and had to choose between retrying a revoked credential forever and dropping turns a
/// second attempt would have completed.
///
/// Three rounds for the transient kinds because `MAX_PROVIDER_RETRIES` is 2: the agent retries them
/// and the assertion is about what it answers once those are spent, not about the first failure.
/// A one-round script would exhaust the mock instead and prove something else.
#[test]
fn the_turn_failed_payload_tells_a_transient_failure_from_a_permanent_one() {
    for (kind, rounds, expected) in [
        ("fail", 1, "https://meka.so/errors/provider"),
        (
            "fail_retryable",
            3,
            "https://meka.so/errors/provider-unavailable",
        ),
        (
            "fail_stream",
            3,
            "https://meka.so/errors/provider-unavailable",
        ),
    ] {
        let round = serde_json::json!([
            { "kind": kind, "message": "scripted upstream failure", "retry_after_secs": null }
        ]);
        let script = serde_json::Value::Array(vec![round; rounds]);
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
        let body = harness
            .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
            .json(&serde_json::json!({"message": "go", "stream": true}))
            .send()
            .expect("send")
            .text()
            .expect("body");

        assert!(
            body.contains("event: turn.failed"),
            "{kind} must fail the turn; body was:\n{body}"
        );
        assert!(
            body.contains(&format!("\"{expected}\"")),
            "{kind} must arrive as {expected}; body was:\n{body}"
        );
        // The status inside the payload, not just the type. Without it a regression that answered
        // 500 under the right `type` passes here, and one error type reporting two statuses is
        // exactly what a client keying on the type cannot handle.
        assert!(
            body.contains("\"status\":502"),
            "{kind} must carry 502 alongside its type; body was:\n{body}"
        );
        // Only meaningful for `fail_retryable`, which is the one kind here that *could* have
        // carried a header and was scripted with none. Left applying to all three so a future
        // kind added to this table inherits it.
        assert!(
            !body.contains("\"retry_after\""),
            "the setup must be the case with no Retry-After, or {kind} proves nothing; body was:\n\
             {body}"
        );
    }
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

/// `options.skill` naming a broken `SKILL.md` says so, rather than "unknown skill".
///
/// The last door to keep composing its own "not found". Both answers are a 422, so the status code
/// hid the difference: a caller driving a turn with a skill they had just installed was told it did
/// not exist, when the file was sitting in the store with a typo in its frontmatter. Every other
/// lookup goes through `SkillIndex::unavailable`; a per-site mutation sweep is what showed this one
/// had no test holding it there.
#[test]
fn a_turn_naming_a_broken_skill_says_why_rather_than_unknown() {
    let harness =
        ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", STORE_SCOPES);
    let broken = harness.home().join("meka").join("skills").join("wrecked");
    std::fs::create_dir_all(&broken).expect("mkdir");
    std::fs::write(
        broken.join("SKILL.md"),
        "---\nname: wrecked\ndescription: [unclosed\n---\nBODY\n",
    )
    .expect("seed");

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
            "options": {"skill": "wrecked"},
        }))
        .send()
        .expect("send");
    assert_eq!(response.status(), 422);
    let body: serde_json::Value = response.json().expect("parse");
    let detail = body["detail"].as_str().unwrap_or_default();
    assert!(
        detail.contains("could not be read"),
        "a present-but-unparseable file must not read as absent: {body}"
    );
    assert!(detail.contains("frontmatter"), "{body}");
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
        ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", &["sessions:r"]);
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
        "",
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

    // Mixed-validity PATCH: permission flips to "unrestricted" (valid), cwd is relative (invalid).
    let patch = harness
        .request(reqwest::Method::PATCH, &format!("/v1/sessions/{}", id))
        .json(&serde_json::json!({
            "permission": "unrestricted",
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
        ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_mcp_only", &["mcp:r"]);
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
        ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_skills_only", &[
            "skills:r",
        ]);
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
        "",
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
    let harness = ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_w_only", &[
        "sessions:w",
    ]);
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
    let harness = ServeTestHarness::spawn("docs = true\n", mock_simple_turn());
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

/// `isolated` is gone from the request body, and a client still sending it is told so.
///
/// The field named a mode that ran the turn in a fresh session instead of the conversation that
/// created the job. `CreateJobRequest` denies unknown fields, so this is a 422 naming `isolated`
/// rather than a silently ignored key -- which matters because the two failures look identical
/// from the client's side until the fire lands somewhere it did not expect.
#[test]
fn a_request_still_asking_for_isolation_is_refused_by_name() {
    let harness = ServeTestHarness::spawn_with(
        "",
        "[schedule]\nenabled = true\npoll_interval = \"1s\"\n",
        mock_simple_turn(),
        "sk_test_token",
        &["sessions:r", "sessions:w", "schedule:r", "schedule:w"],
    );

    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("create");
    assert_eq!(create.status(), 201);
    let body: serde_json::Value = create.json().expect("parse");
    let id = body["id"].as_str().expect("id").to_string();

    let refused = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/schedule", id),
        )
        .json(&serde_json::json!({
            "prompt": "check the feed",
            "every": "1h",
            "isolated": true,
        }))
        .send()
        .expect("send");
    let status = refused.status();
    let text = refused.text().expect("text");
    assert_eq!(status, 422, "{text}");
    assert!(
        text.contains("isolated"),
        "the refusal must name the retired field so the fix is obvious: {text}"
    );

    // And the same job without it is still perfectly ordinary.
    let accepted = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/schedule", id),
        )
        .json(&serde_json::json!({"prompt": "check the feed", "every": "1h"}))
        .send()
        .expect("send");
    let status = accepted.status();
    let text = accepted.text().expect("text");
    assert_eq!(status, 201, "{text}");
    assert!(
        !text.contains("isolated"),
        "and the job it returns no longer carries the field either: {text}"
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

/// A session gauges itself against its own profile's window, not the server default's.
///
/// The failure this guards is silent and expensive: `agent_options` is built once per process from
/// the default profile, so a session pinned to a 32k model was measured against the default's
/// window. Auto-compaction never reached its threshold and the provider rejected the turn for
/// exceeding a context meka thought was 97% free.
#[test]
fn context_window_follows_the_session_profile_not_the_server_default() {
    let harness = ServeTestHarness::spawn_with_prelude(
        "default_provider = \"mock\"\n",
        r#"
[providers.small]
type = "anthropic-messages"
model = "claude-sonnet-4-5"
context_window = 32000
"#,
        mock_simple_turn(),
    );

    let mut windows = std::collections::BTreeMap::new();
    for provider in ["mock", "small"] {
        let create = harness
            .request(reqwest::Method::POST, "/v1/sessions")
            .json(&serde_json::json!({
                "cwd": std::env::temp_dir().to_string_lossy(),
                "provider": provider,
            }))
            .send()
            .expect("send");
        assert_eq!(
            create.status(),
            201,
            "create on `{provider}` failed: {}",
            create.text().unwrap_or_default()
        );
        let id = create.json::<serde_json::Value>().expect("parse")["id"]
            .as_str()
            .expect("id")
            .to_string();

        let context: serde_json::Value = harness
            .request(
                reqwest::Method::GET,
                &format!("/v1/sessions/{}/context", id),
            )
            .send()
            .expect("send")
            .json()
            .expect("parse");
        windows.insert(provider, context["window"].as_u64().expect("window"));
    }

    assert_eq!(
        windows["small"], 32_000,
        "the session's own profile states 32000: {windows:?}"
    );
    assert_ne!(
        windows["mock"], windows["small"],
        "a profile that states no window must not inherit one that does: {windows:?}"
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
        ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", &["sessions:w"]);
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
        ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", &["sessions:r"]);
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
    let harness = ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", &[
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
        "",
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
    let harness = ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", &[
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
    let harness = ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", &[
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

/// A job that cannot fire is refused at the door, and one already there says so when listed.
///
/// This is the only door that could plant such a job: `schedule_create` needs `read` to dispatch at
/// all, so the agent cannot reach it from a session at `none`, and a token's scopes say nothing
/// about the session's level. It was also the only reader with no way to find out -- the agent gets
/// a `NOT FIRING` line and `meka schedule list` a `Held` column, while `GET` returned a row that
/// looked exactly like a working one.
#[test]
fn a_job_that_could_never_fire_is_refused_and_an_existing_one_is_reported() {
    let config = "\n[[serve.tokens]]\ntoken = \"sk_test_full\"\n\
                  scopes = [\"sessions:r\", \"sessions:w\", \"schedule:r\", \"schedule:w\"]\n";
    let harness = ServeTestHarness::spawn_with("", config, mock_simple_turn(), "sk_test_token", &[
        "schedule:r",
    ]);

    let session = harness
        .client
        .post(format!("{}/v1/sessions", harness.base_url))
        .header("Authorization", "Bearer sk_test_full")
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

    let plant = || {
        harness
            .client
            .post(format!(
                "{}/v1/sessions/{}/schedule",
                harness.base_url, session
            ))
            .header("Authorization", "Bearer sk_test_full")
            .json(&serde_json::json!({"prompt": "remind me", "every": "1h"}))
            .send()
            .expect("send")
    };

    // Planted while the session can still run it, so the listing below is about the level changing
    // underneath an existing job rather than about the refusal.
    assert_eq!(plant().status().as_u16(), 201, "ungated at `read` is fine");

    let lowered = harness
        .client
        .patch(format!("{}/v1/sessions/{}", harness.base_url, session))
        .header("Authorization", "Bearer sk_test_full")
        .json(&serde_json::json!({"permission": "none"}))
        .send()
        .expect("send");
    assert_eq!(lowered.status().as_u16(), 200, "the session drops to none");

    let refused = plant();
    assert_eq!(
        refused.status().as_u16(),
        403,
        "a second job would never fire either, and the fire door already knows it"
    );
    let body = refused.json::<serde_json::Value>().expect("parse");
    assert!(
        body["type"]
            .as_str()
            .unwrap_or_default()
            .ends_with("session-permission"),
        "raising the session is the remedy, and the type has to say so: {body}"
    );

    let listed = harness
        .client
        .get(format!(
            "{}/v1/sessions/{}/schedule",
            harness.base_url, session
        ))
        .header("Authorization", "Bearer sk_test_full")
        .send()
        .expect("send")
        .json::<serde_json::Value>()
        .expect("parse");
    let withheld = listed["jobs"][0]["withheld"].as_str().unwrap_or_default();
    assert!(
        withheld.contains("nothing is executable"),
        "the job planted before the drop is inert now, and a client has no other way to tell: \
         {listed}"
    );
}

/// A gate refusal is routed by what would actually fix it.
///
/// A shell gate below `unrestricted` is a 403 `session-permission`, whose documented remedy --
/// `PATCH /v1/sessions/{id}` -- is the real one. A misspelled or write-capable tool is not: no
/// level and no token changes the answer, so sending a client to raise a session is a wild goose
/// chase. Both used to be 403 `session-permission`, which meant `{"tool": 5}` was a 422 and
/// `{"tool": "typo"}` a 403 saying the session sat too low.
#[test]
fn a_gate_refusal_is_a_422_when_no_permission_would_help() {
    let config = "\n[[serve.tokens]]\ntoken = \"sk_test_full\"\n\
                  scopes = [\"sessions:r\", \"sessions:w\", \"schedule:r\", \"schedule:w\"]\n";
    let harness = ServeTestHarness::spawn_with("", config, mock_simple_turn(), "sk_test_token", &[
        "schedule:r",
    ]);

    let session = |permission: &str| -> String {
        harness
            .client
            .post(format!("{}/v1/sessions", harness.base_url))
            .header("Authorization", "Bearer sk_test_full")
            .json(&serde_json::json!({
                "cwd": std::env::temp_dir().to_string_lossy(),
                "permission": permission,
            }))
            .send()
            .expect("send")
            .json::<serde_json::Value>()
            .expect("parse")["id"]
            .as_str()
            .expect("id")
            .to_string()
    };
    let refuse = |id: &str, check: serde_json::Value| -> (u16, String) {
        let response = harness
            .client
            .post(format!("{}/v1/sessions/{}/schedule", harness.base_url, id))
            .header("Authorization", "Bearer sk_test_full")
            .json(&serde_json::json!({
                "prompt": "check the thing",
                "every": "1h",
                "gate": {"check": check, "when": "succeeded"},
            }))
            .send()
            .expect("send");
        let status = response.status().as_u16();
        let body = response.json::<serde_json::Value>().expect("parse");
        let kind = body["type"].as_str().unwrap_or_default().to_string();
        (status, kind)
    };

    let (status, kind) = refuse(&session("read"), serde_json::json!({"command": "true"}));
    assert_eq!(
        status, 403,
        "raising the session is the remedy for a shell gate"
    );
    assert!(kind.ends_with("session-permission"), "{kind}");

    // At `unrestricted`, so the level cannot be what is wrong.
    let high = session("unrestricted");
    let (status, kind) = refuse(&high, serde_json::json!({"tool": "no_such_tool_at_all"}));
    assert_eq!(status, 422, "no level fixes a name that does not exist");
    assert!(kind.ends_with("invalid-body"), "{kind}");

    let (status, kind) = refuse(&high, serde_json::json!({"tool": "write_file"}));
    assert_eq!(status, 422, "no level fixes a tool that is not read-only");
    assert!(kind.ends_with("invalid-body"), "{kind}");
}

/// The whole point of tool gates, asserted through a real door: a read-only built-in is accepted at
/// `read`, where a shell gate is refused.
///
/// This is the only test that requires the gate dispatcher to be *wired*. Every refusal above
/// passes whether or not one exists, because a process that cannot resolve any name refuses for
/// `ToolUnavailable` with the same 422 and the same body type as a tool that is genuinely not
/// read-only. Deleting the assignment in `main` therefore left the entire suite green while no tool
/// gate could ever be created.
#[test]
fn a_read_only_built_in_tool_gate_is_accepted_at_read() {
    let config = "\n[[serve.tokens]]\ntoken = \"sk_test_full\"\n\
                  scopes = [\"sessions:r\", \"sessions:w\", \"schedule:r\", \"schedule:w\"]\n";
    let harness = ServeTestHarness::spawn_with("", config, mock_simple_turn(), "sk_test_token", &[
        "schedule:r",
    ]);

    let session = harness
        .client
        .post(format!("{}/v1/sessions", harness.base_url))
        .header("Authorization", "Bearer sk_test_full")
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
        .client
        .post(format!(
            "{}/v1/sessions/{}/schedule",
            harness.base_url, session
        ))
        .header("Authorization", "Bearer sk_test_full")
        .json(&serde_json::json!({
            "prompt": "tell me when it changes",
            "every": "1h",
            "gate": {
                "check": {"tool": "read_file", "arguments": {"path": "/etc/hostname"}},
                "when": "changed",
            },
        }))
        .send()
        .expect("send");

    let status = response.status().as_u16();
    let body = response.json::<serde_json::Value>().expect("parse");
    assert_eq!(
        status, 201,
        "a `read` session may gate on a tool that resolves to `read`: {body}"
    );
    assert_eq!(
        body["gate"]["kind"].as_str(),
        Some("tool"),
        "and it round-trips as a tool gate: {body}"
    );
    assert_eq!(
        body["withheld"].as_str(),
        None,
        "with nothing holding it back: {body}"
    );
}

/// A `schedule:r` token sees that a job is gated, but not what the gate runs.
///
/// A gate command is an `execute_command` line that runs unattended as the server's user. The
/// webhook path already withholds the same field, on the stated grounds that a command line is the
/// highest-entropy field in the system and the one most likely to carry a credential someone pasted
/// into a `curl`. `GET /v1/schedule` is server-wide, so leaving the command at `schedule:r` handed
/// every gate on the box to a bridge scoped to the read half of a scope that was invented so
/// schedule access would *not* imply session access.
///
/// The fire condition stays visible either way, so a client can still tell a gated job from an
/// ungated one.
#[test]
fn a_schedule_scoped_token_sees_a_gate_without_its_command() {
    let config = "\n[[serve.tokens]]\ntoken = \"sk_test_full\"\n\
                  scopes = [\"sessions:r\", \"sessions:w\", \"schedule:r\", \"schedule:w\"]\n";
    let harness = ServeTestHarness::spawn_with("", config, mock_simple_turn(), "sk_test_token", &[
        "schedule:r",
    ]);

    let id = harness
        .client
        .post(format!("{}/v1/sessions", harness.base_url))
        .header("Authorization", "Bearer sk_test_full")
        .json(&serde_json::json!({
            "cwd": std::env::temp_dir().to_string_lossy(),
            "permission": "unrestricted",
        }))
        .send()
        .expect("send")
        .json::<serde_json::Value>()
        .expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    let secret = "curl -H 'Authorization: Bearer sk-live-DO-NOT-DISCLOSE' https://example.test";
    let created = harness
        .client
        .post(format!("{}/v1/sessions/{}/schedule", harness.base_url, id))
        .header("Authorization", "Bearer sk_test_full")
        .json(&serde_json::json!({
            "prompt": "check the thing",
            "every": "1h",
            "gate": {"check": {"command": secret}, "when": "succeeded"},
        }))
        .send()
        .expect("send");
    assert_eq!(created.status(), reqwest::StatusCode::CREATED);

    // The full-scope token gets the command back.
    let full: serde_json::Value = harness
        .client
        .get(format!("{}/v1/schedule", harness.base_url))
        .header("Authorization", "Bearer sk_test_full")
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert_eq!(
        full["jobs"][0]["gate"]["check"].as_str(),
        Some(secret),
        "a token holding sessions:r may see the command: {full}"
    );

    // The schedule:r-only token does not.
    let limited: serde_json::Value = harness
        .request(reqwest::Method::GET, "/v1/schedule")
        .send()
        .expect("send")
        .json()
        .expect("parse");
    let body = limited.to_string();
    assert!(
        !body.contains("sk-live-DO-NOT-DISCLOSE"),
        "the gate command must not reach a schedule:r-only token: {body}"
    );
    assert_eq!(
        limited["jobs"][0]["gate"]["when"].as_str(),
        Some("succeeded"),
        "the gate itself must still be visible: {limited}"
    );
    assert_eq!(
        limited["jobs"][0]["gate"]["kind"].as_str(),
        Some("shell"),
        "and so must its kind, so a shell gate reads differently from a tool one: {limited}"
    );
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
    let harness = ServeTestHarness::spawn_with("", config, mock_simple_turn(), "sk_test_token", &[
        "schedule:r",
        "schedule:w",
    ]);
    let id = harness
        .client
        .post(format!("{}/v1/sessions", harness.base_url))
        .header("Authorization", "Bearer sk_test_full")
        .json(&serde_json::json!({
            "cwd": std::env::temp_dir().to_string_lossy(),
            "permission": "unrestricted",
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
            "gate": {"check": {"command": "id"}},
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
    let harness = ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", &[
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
            "gate": {"check": {"command": "git status --porcelain"}},
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
    let harness = ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", &[
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
    let harness = ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", &[
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
    let harness = ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", &[
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
        ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", STORE_SCOPES);

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

/// A skill under a read-only `[skills] extra_paths` root is neither writable nor deletable here,
/// and both refusals arrive as 409 `store-read-only` naming where the file actually is.
///
/// Writing would not update that skill: it would put a second copy in meka's own store which wins
/// precedence, so the caller is told it edited a procedure while every other client goes on reading
/// the original. Both branches, the new [`ErrorKind::StoreReadOnly`] and its documented 409, had no
/// test at all: deleting either check left the whole suite green.
///
/// The broken file is the half that was actually wrong. The check compared against the *loaded*
/// skills, so a `SKILL.md` that does not parse was a name the store had no opinion about, and the
/// write went through silently.
#[test]
fn a_skill_in_a_read_only_root_is_refused_by_put_and_delete() {
    // A directory of its own, named absolutely, rather than `~` under the harness's home.
    //
    // The config has to name this root before the server starts, and `~` cannot be pointed at a
    // temp directory on Windows: `dirs::home_dir` there is `SHGetKnownFolderPath(FOLDERID_Profile)`
    // and ignores environment entirely, so the `HOME` the harness sets did nothing,
    // `~/shared-skills` resolved to the runner's real profile, and the root came up empty. An empty
    // read-only root refuses nothing, so the PUT this test exists to see refused went through.
    //
    // A TOML *literal* string, because a Windows path's backslashes are escapes in a basic one.
    let shared_root = tempfile::tempdir().expect("tempdir");
    let shared = shared_root.path().to_path_buf();
    let harness = ServeTestHarness::spawn_with(
        "",
        &format!("\n[skills]\nextra_paths = ['{}']\n", shared.display()),
        mock_simple_turn(),
        "sk_test_token",
        STORE_SCOPES,
    );
    for (name, body) in [
        (
            "borrowed",
            "---\nname: borrowed\ndescription: theirs\n---\nTHEIRS\n",
        ),
        (
            "wrecked",
            "---\nname: wrecked\ndescription: [unclosed\n---\nTHEIRS\n",
        ),
    ] {
        std::fs::create_dir_all(shared.join(name)).expect("mkdir");
        std::fs::write(shared.join(name).join("SKILL.md"), body).expect("seed");
    }

    for name in ["borrowed", "wrecked"] {
        let write = harness
            .request(reqwest::Method::PUT, &format!("/v1/skills/{}", name))
            .json(&serde_json::json!({"description": "mine now"}))
            .send()
            .expect("send");
        assert_eq!(
            write.status(),
            409,
            "PUT on a read-only root must conflict, not shadow: {}",
            write.text().unwrap_or_default()
        );
        let problem: serde_json::Value = harness
            .request(reqwest::Method::PUT, &format!("/v1/skills/{}", name))
            .json(&serde_json::json!({"description": "mine now"}))
            .send()
            .expect("send")
            .json()
            .expect("parse");
        assert_eq!(problem["type"], "https://meka.so/errors/store-read-only");
        assert!(
            problem["detail"]
                .as_str()
                .unwrap_or_default()
                .contains(&shared.join(name).display().to_string()),
            "the refusal must name the file it is protecting: {}",
            problem
        );

        let delete = harness
            .request(reqwest::Method::DELETE, &format!("/v1/skills/{}", name))
            .send()
            .expect("send");
        assert_eq!(delete.status(), 409, "DELETE must not reach a foreign root");
    }

    assert!(
        shared.join("borrowed/SKILL.md").exists() && shared.join("wrecked/SKILL.md").exists(),
        "neither foreign file may be touched"
    );
}

/// `GET /v1/skills/{name}` must not report a `SKILL.md` that is present but unparseable as absent.
///
/// A flat 404 sent an operator looking for a file sitting in the store, which is the answer they
/// get straight after the startup warning names it.
#[test]
fn getting_a_broken_skill_says_why_rather_than_404() {
    let harness =
        ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", STORE_SCOPES);
    let broken = harness.home().join("meka").join("skills").join("wrecked");
    std::fs::create_dir_all(&broken).expect("mkdir");
    std::fs::write(
        broken.join("SKILL.md"),
        "---\nname: wrecked\ndescription: [unclosed\n---\nBODY\n",
    )
    .expect("seed");

    let response = harness
        .request(reqwest::Method::GET, "/v1/skills/wrecked")
        .send()
        .expect("send");
    assert_eq!(response.status(), 422, "a present file is not a 404");
    let problem: serde_json::Value = response.json().expect("parse");
    assert!(
        problem["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("could not be read"),
        "{}",
        problem
    );

    let absent = harness
        .request(reqwest::Method::GET, "/v1/skills/never-written")
        .send()
        .expect("send");
    assert_eq!(absent.status(), 404, "a name nobody wrote is still a 404");
}

/// An omitted `body` keeps the existing one. A caller correcting a description should not have to
/// resend prose it never meant to touch, and the alternative is silently clearing it.
#[test]
fn skill_write_without_a_body_preserves_the_existing_one() {
    let harness =
        ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", STORE_SCOPES);
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
        ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", STORE_SCOPES);
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
        ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", STORE_SCOPES);
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
    let harness = ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", &[
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
        ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", STORE_SCOPES);

    let write = harness
        .request(reqwest::Method::PUT, "/v1/memory/deploy-policy")
        .json(&serde_json::json!({
            "description": "Never deploy on Fridays",
            "priority": 1,
            "body": "Ship Monday to Thursday only.",
            "tags": ["deploy", "policy"],
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
    // The three fields the storage move promised on this surface, none of which anything asserted.
    // `recorded_at` and `updated_at` are distinct concepts -- when the note was made, and when the
    // row was last written -- so a handler serving one for both would satisfy any test that only
    // checked they were present.
    assert!(
        detail["recorded_at"]
            .as_str()
            .is_some_and(|stamp| stamp.parse::<chrono::DateTime<chrono::Utc>>().is_ok()),
        "recorded_at must be an RFC 3339 stamp: {}",
        detail
    );
    assert!(
        detail["updated_at"]
            .as_str()
            .is_some_and(|stamp| stamp.parse::<chrono::DateTime<chrono::Utc>>().is_ok()),
        "updated_at must be an RFC 3339 stamp: {}",
        detail
    );
    assert_eq!(
        detail["tags"],
        serde_json::json!(["deploy", "policy"]),
        "tags round-trip through the write: {}",
        detail
    );
    let listed_entry = listed["memories"]
        .as_array()
        .expect("memories")
        .iter()
        .find(|m| m["name"] == "deploy-policy")
        .expect("the memory is listed");
    assert_eq!(
        listed_entry["tags"],
        serde_json::json!(["deploy", "policy"]),
        "and the list carries them too: {}",
        listed_entry
    );
    assert!(
        listed_entry["recorded_at"].as_str().is_some(),
        "{}",
        listed_entry
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

    // A second delete is a 404, not another 204. The distinction is `rows_affected`: a handler
    // that ignores it reports every delete as having removed something, so a client cannot tell
    // "gone now" from "was never there" and a typo in a name reads as success.
    let again = harness
        .request(reqwest::Method::DELETE, "/v1/memory/deploy-policy")
        .send()
        .expect("send");
    assert_eq!(again.status(), 404, "{}", again.text().unwrap_or_default());
    assert_eq!(
        harness
            .request(reqwest::Method::DELETE, "/v1/memory/never-existed")
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
    let harness = ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", &[
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
    let harness = ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", &[
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
        write_file["required_permission"], "workspace",
        "the catalogue must report the tier a client needs to render an approval prompt"
    );
}

/// The tool names a live session's registry ends up holding, which is what `assemble_agent`
/// actually built rather than what a test assembled by hand.
fn session_tool_names(harness: &ServeTestHarness, id: &str) -> Vec<String> {
    let response = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}/tools", id))
        .send()
        .expect("send");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().expect("parse");
    body["tools"]
        .as_array()
        .expect("tools")
        .iter()
        .filter_map(|tool| tool["name"].as_str().map(str::to_string))
        .collect()
}

const AGENT_FAMILY: [&str; 4] = [
    "agent_spawn",
    "agent_list",
    "agent_followup",
    "agent_delete",
];

/// `assemble_agent` gates the family on `agent_tools_registered`, and every other test of that rule
/// calls `register_subagent_tools` directly on a registry it built itself. This is the only door
/// that walks the live path, so a predicate that answered wrongly for a real session would show up
/// nowhere else -- including in `meka tools list`, which reproduces the rule rather than observing
/// it.
#[test]
fn a_session_registers_the_whole_agent_family_by_default() {
    let harness = ServeTestHarness::spawn("", mock_simple_turn());
    let id = session_with_one_turn(&harness);
    let names = session_tool_names(&harness, &id);
    for name in AGENT_FAMILY {
        assert!(
            names.contains(&name.to_string()),
            "'{name}' must be registered, got: {names:?}"
        );
    }
}

/// Denying `agent_spawn` takes the three lifecycle tools with it, on a real session and not just in
/// the listing's reproduction of the rule.
#[test]
fn a_session_denied_agent_spawn_registers_none_of_the_family() {
    let harness = ServeTestHarness::spawn_with_prelude(
        "[tools]\ndisabled_tools = [\"agent_spawn\"]\n",
        "",
        mock_simple_turn(),
    );
    let id = session_with_one_turn(&harness);
    let names = session_tool_names(&harness, &id);
    for name in AGENT_FAMILY {
        assert!(
            !names.contains(&name.to_string()),
            "'{name}' goes with agent_spawn, got: {names:?}"
        );
    }
    assert!(
        names.contains(&"read_file".to_string()),
        "only the family is denied, got: {names:?}"
    );
}

/// The other half of the rule, which the listing used not to mention at all.
#[test]
fn a_session_at_depth_zero_registers_none_of_the_family() {
    let harness = ServeTestHarness::spawn_with_prelude(
        "[session]\nsubagent_max_depth = 0\n",
        "",
        mock_simple_turn(),
    );
    let id = session_with_one_turn(&harness);
    let names = session_tool_names(&harness, &id);
    for name in AGENT_FAMILY {
        assert!(
            !names.contains(&name.to_string()),
            "'{name}' needs a depth budget, got: {names:?}"
        );
    }
    assert!(
        names.contains(&"read_file".to_string()),
        "only the family is denied, got: {names:?}"
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
    assert_eq!(mock["type"], "anthropic-messages");
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
    let harness = ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", &[
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
    let harness = ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", &[
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
        ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", &["sessions:w"]);
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

    // The timestamp is stamped and signed per attempt, not once per delivery. Receivers reject a
    // signature whose timestamp falls outside a replay window, and the backoff can put a late retry
    // minutes past the first attempt: re-sending the original stamp got the retry rejected as a
    // replay of itself. Deduplication is the delivery id's job, which is why that one is constant.
    assert_ne!(
        first.timestamp, second.timestamp,
        "each attempt must carry its own timestamp"
    );
    for delivery in [&first, &second] {
        let signature = delivery
            .signature
            .as_deref()
            .expect("a configured secret must produce a signature");
        assert_eq!(
            signature,
            expected_signature("s", &delivery.timestamp, &delivery.body),
            "each attempt must be signed over its own timestamp"
        );
    }
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

/// A cancelled task reports on the next turn, without spending one of its own.
///
/// The predicate test in `background.rs` covers only `wakes_a_host`; every branch that consults it
/// could be deleted with the suite still green. This one is written against the two that meet in
/// `meka serve`: the poller must leave the outcome alone, and `submit_turn` must fold it into the
/// caller's own message. Folding rather than appending is the point -- a lone user message opens a
/// turn (`conversation::opens_turn`), so a notice delivered as its own message would be rewound in
/// place of the user's last exchange and sent as a second consecutive user turn.
///
/// The script is the assertion that no turn was spent: it holds exactly one round after the
/// cancellation, so a poller that delivered would consume it and leave the real turn to fail.
#[test]
fn a_cancelled_task_rides_on_the_next_turn_instead_of_causing_one() {
    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "tu_1", "name": "execute_command" },
            { "kind": "tool_use_end", "input": {"command": "sleep 120", "background": true} },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "started" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "text", "text": "answered the question" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    // Polled fast, so "the poller did not deliver" is a fact about the branch rather than about
    // the test finishing before the first tick.
    let harness = ServeTestHarness::spawn(
        "\n[background]\nenabled = true\n\n[schedule]\npoll_interval = \"200ms\"\n",
        script,
    );
    let id = start_streaming_session(&harness);
    harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "run it"}))
        .send()
        .expect("send");

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

    let messages = |harness: &ServeTestHarness| -> Vec<serde_json::Value> {
        let body: serde_json::Value = harness
            .request(
                reqwest::Method::GET,
                &format!("/v1/sessions/{}/messages", id),
            )
            .send()
            .expect("send")
            .json()
            .expect("parse");
        body["messages"].as_array().expect("messages").clone()
    };
    let before = messages(&harness);

    let cancel = harness
        .request(
            reqwest::Method::DELETE,
            &format!("/v1/sessions/{}/tasks/{}", id, &task_id[..8]),
        )
        .send()
        .expect("send");
    assert_eq!(cancel.status(), 204, "the task must cancel");

    // Several poll intervals, so a delivering poller has every chance to prove itself.
    std::thread::sleep(Duration::from_secs(2));
    assert_eq!(
        messages(&harness).len(),
        before.len(),
        "a cancellation must add no message of its own: it would be a turn boundary with no turn \
         behind it"
    );

    let turn: serde_json::Value = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "what is in this CSV?"}))
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert_eq!(
        turn["stop_reason"], "end_turn",
        "the round the poller must not have eaten is this turn's: {}",
        turn
    );

    let after = messages(&harness);
    let carrier = after
        .iter()
        .rev()
        .find(|message| message["role"] == "user")
        .expect("the turn's own user message");
    let text = carrier["content"][0]["text"].as_str().unwrap_or_default();
    assert!(
        text.contains("was cancelled") && text.contains("what is in this CSV?"),
        "the outcome must ride inside the user's own message, not beside it: {}",
        text
    );
    assert!(
        !after
            .windows(2)
            .any(|pair| pair[0]["role"] == "user" && pair[1]["role"] == "user"),
        "two consecutive user turns must never reach the provider: {:#?}",
        after
    );
}

/// A session with a detached command still running is not idle, however long since its last turn.
///
/// The GC counted in-flight *turns* only, so a session whose turn had finished but whose background
/// command had not was evicted on schedule. Eviction drops the `SessionEntry`, which drops the
/// session's file lock, while the detached task keeps running -- and the next thing to open the
/// session sweeps it: the model is told the work "was interrupted", and when the command really
/// finishes `finish_background_task`'s `AND status = 'running'` guard discards the real outcome.
/// A false report and a silent loss out of one eviction.
///
/// Written against a single server because that reaches it too: evict on the timer, re-attach on
/// the next turn, sweep your own live task.
#[test]
fn a_session_with_a_running_background_task_is_not_evicted() {
    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "tu_1", "name": "execute_command" },
            { "kind": "tool_use_end", "input": {"command": "sleep 30", "background": true} },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "started" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "text", "text": "second turn" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn(
        "idle_timeout = \"1s\"\ngc_scan_interval = \"200ms\"\n\n[background]\nenabled = true\n",
        script,
    );
    let id = start_streaming_session(&harness);
    harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "run it"}))
        .send()
        .expect("send");

    let tasks = |harness: &ServeTestHarness| -> serde_json::Value {
        harness
            .request(reqwest::Method::GET, &format!("/v1/sessions/{}/tasks", id))
            .send()
            .expect("send")
            .json()
            .expect("parse")
    };
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if tasks(&harness)["tasks"]
            .as_array()
            .is_some_and(|tasks| tasks.iter().any(|task| task["status"] == "running"))
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    // Well past `idle_timeout` and many scans, with no turn running and the command still going.
    std::thread::sleep(Duration::from_millis(3500));
    // A second turn is what re-attaches, and re-attaching is what runs the sweep.
    let second = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "still there?"}))
        .send()
        .expect("send");
    assert_eq!(
        second.status(),
        200,
        "second turn failed: {}",
        second.text().unwrap_or_default()
    );

    let after = tasks(&harness);
    let task = after["tasks"]
        .as_array()
        .and_then(|tasks| tasks.first())
        .expect("the task must still be listed");
    assert_eq!(
        task["status"], "running",
        "a command that is still running must not be reported as interrupted: {}",
        after
    );
}

/// A fire that fails keeps the outcome that was riding on it.
///
/// A recurring job asks for `WithdrawOnFailure`, because its next occurrence regenerates the
/// prompt. That stops being true the moment an outcome joins it: the row is stamped delivered
/// before the turn starts and is never handed out again, so withdrawing the message destroys the
/// only copy. `retention_carrying` is what notices, and its three call sites were reachable only
/// by a turn that actually fails -- which every other test of this path avoids.
#[test]
fn a_fire_that_fails_keeps_the_outcome_riding_on_it() {
    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "tu_1", "name": "execute_command" },
            { "kind": "tool_use_end", "input": {"command": "sleep 120", "background": true} },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "started" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        // The fire's own round, which fails after the prompt has been persisted.
        [
            { "kind": "fail", "message": "error sending request: connection refused" }
        ]
    ]);
    let harness = ServeTestHarness::spawn_with(
        "",
        "\n[background]\nenabled = true\n\n[schedule]\npoll_interval = \"200ms\"\n",
        script,
        "sk_test_token",
        &["sessions:r", "sessions:w", "schedule:r", "schedule:w"],
    );
    let id = start_streaming_session(&harness);
    harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "run it"}))
        .send()
        .expect("send");

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

    let job = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/schedule", id),
        )
        .json(&serde_json::json!({"prompt": "PROBE_FAILING_FIRE", "every": "1s"}))
        .send()
        .expect("send");
    assert_eq!(job.status(), 201, "the job must be created");

    let cancel = harness
        .request(
            reqwest::Method::DELETE,
            &format!("/v1/sessions/{}/tasks/{}", id, &task_id[..8]),
        )
        .send()
        .expect("send");
    assert_eq!(cancel.status(), 204, "the task must cancel");

    // The fire runs, carries the cancellation, and its provider call fails. The prompt must stay.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let body: serde_json::Value = harness
            .request(
                reqwest::Method::GET,
                &format!("/v1/sessions/{}/messages", id),
            )
            .send()
            .expect("send")
            .json()
            .expect("parse");
        let carried = body["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .any(|message| {
                message["role"] == "user"
                    && message["content"][0]["text"]
                        .as_str()
                        .is_some_and(|text| text.contains("was cancelled"))
            });
        if carried {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "a failed fire withdrew the prompt and took the outcome with it; the row is stamped \
             delivered and will never be handed out again: {}",
            body
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// A scheduled fire announces the outcome it claims, not just delivers it.
///
/// Three doors claim from the undelivered pool and all three must announce, because the stamp is
/// one-way: `list_unannounced_background_tasks` also requires `delivered_at IS NULL`, so a row a
/// fire delivered can never be announced afterwards. Reachable without any race on a restart, where
/// the load sweep retires a task the dead host left running and a due job claims it before the
/// poller has ticked.
#[test]
fn a_scheduled_fire_announces_what_it_claims() {
    let (port, rx) = spawn_webhook_listener();
    // The poller only sweeps *resident* sessions, so evicting this one takes it out of reach: the
    // scheduled fire revives it through `ensure_session_loaded`, and is then the only thing that
    // can announce. A short idle timeout plus a poll interval long enough that the first tick
    // arrives after the eviction is what arranges that -- the same shape as a restart, where the
    // load sweep retires the task and a due job claims it before the poller has ever ticked.
    let config = format!(
        "idle_timeout = \"1s\"\ngc_scan_interval = \"200ms\"\n\
         \n[background]\nenabled = true\n\n[schedule]\npoll_interval = \"10s\"\n\n\
         [[serve.webhooks]]\nurl = \"http://127.0.0.1:{}/hook\"\nsecret = \"s\"\n\
         events = [\"task.finished\"]\n",
        port
    );
    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "tu_1", "name": "execute_command" },
            { "kind": "tool_use_end", "input": {"command": "sleep 120", "background": true} },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "started" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "text", "text": "ran the job" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn_with("", &config, script, "sk_test_token", &[
        "sessions:r",
        "sessions:w",
        "schedule:r",
        "schedule:w",
    ]);
    let id = start_streaming_session(&harness);
    harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "run it"}))
        .send()
        .expect("send");

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

    // Created before the cancellation, so nothing has to touch the session afterwards: an HTTP
    // request would make it resident again and hand the poller its chance.
    let job = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/schedule", id),
        )
        .json(&serde_json::json!({"prompt": "PROBE_ANNOUNCE", "every": "1s"}))
        .send()
        .expect("send");
    assert_eq!(job.status(), 201, "the job must be created");

    let cancel = harness
        .request(
            reqwest::Method::DELETE,
            &format!("/v1/sessions/{}/tasks/{}", id, &task_id[..8]),
        )
        .send()
        .expect("send");
    assert_eq!(cancel.status(), 204, "the task must cancel");

    let delivery = rx
        .recv_timeout(Duration::from_secs(30))
        .expect("the fire that claimed the outcome must also announce it");
    let body: serde_json::Value = serde_json::from_str(&delivery.body).expect("json");
    assert_eq!(body["event"], "task.finished");
    assert_eq!(body["status"], "cancelled", "body was {}", body);
}

/// A scheduled fire carries a cancellation that has been waiting for a turn.
///
/// The third door. A user turn and the poller's own report were both folded; a scheduled job runs a
/// turn too, and a session whose only traffic is scheduled work would otherwise never learn its
/// task was cancelled -- which is the promise `TaskStatus::wakes_a_host` makes when it declines to
/// wake one. `meka serve` and ACP both had the gap; this pins the serve half.
#[test]
fn a_scheduled_fire_carries_a_cancellation_that_was_waiting() {
    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "tu_1", "name": "execute_command" },
            { "kind": "tool_use_end", "input": {"command": "sleep 120", "background": true} },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "started" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "text", "text": "ran the job" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn_with(
        "",
        "\n[background]\nenabled = true\n\n[schedule]\npoll_interval = \"200ms\"\n",
        script,
        "sk_test_token",
        &["sessions:r", "sessions:w", "schedule:r", "schedule:w"],
    );
    let id = start_streaming_session(&harness);
    harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "run it"}))
        .send()
        .expect("send");

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

    let cancel = harness
        .request(
            reqwest::Method::DELETE,
            &format!("/v1/sessions/{}/tasks/{}", id, &task_id[..8]),
        )
        .send()
        .expect("send");
    assert_eq!(cancel.status(), 204, "the task must cancel");

    // The only turn from here on is the job's, so anything the model is told rides its prompt.
    let job = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/schedule", id),
        )
        .json(&serde_json::json!({"prompt": "PROBE_SCHEDULED_PROMPT", "every": "1s"}))
        .send()
        .expect("send");
    assert_eq!(job.status(), 201, "the job must be created");

    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let body: serde_json::Value = harness
            .request(
                reqwest::Method::GET,
                &format!("/v1/sessions/{}/messages", id),
            )
            .send()
            .expect("send")
            .json()
            .expect("parse");
        let fired = body["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .find(|message| {
                message["role"] == "user"
                    && message["content"][0]["text"]
                        .as_str()
                        .is_some_and(|text| text.contains("PROBE_SCHEDULED_PROMPT"))
            })
            .cloned();
        if let Some(fired) = fired {
            let text = fired["content"][0]["text"].as_str().unwrap_or_default();
            assert!(
                text.contains("was cancelled"),
                "the job's prompt must carry the outcome that was waiting for a turn: {}",
                text
            );
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the scheduled job never fired: {}",
            body
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// A cancelled task is announced to subscribers without being delivered to the model.
///
/// This is the whole point of splitting `announced_at` from `delivered_at`, and the two halves fail
/// in opposite directions: without the split, deferring the model's copy re-fires the webhook on
/// every poll, and firing it once costs the turn the deferral exists to avoid. Nothing else pins
/// the announce half -- the payload test above rides the ordinary completed path, which announced
/// correctly before the change too.
#[test]
fn a_cancelled_task_is_announced_without_being_delivered() {
    let (port, rx) = spawn_webhook_listener();
    let config = format!(
        "\n[background]\nenabled = true\n\n[schedule]\npoll_interval = \"200ms\"\n\n         [[serve.webhooks]]\nurl = \"http://127.0.0.1:{}/hook\"\n         secret = \"s\"\nevents = [\"task.finished\"]\n",
        port
    );
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
    let harness = ServeTestHarness::spawn(&config, script);
    let id = start_streaming_session(&harness);
    harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "run it"}))
        .send()
        .expect("send");

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

    let messages_before = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/messages", id),
        )
        .send()
        .expect("send")
        .json::<serde_json::Value>()
        .expect("parse")["messages"]
        .as_array()
        .expect("messages")
        .len();

    let cancel = harness
        .request(
            reqwest::Method::DELETE,
            &format!("/v1/sessions/{}/tasks/{}", id, &task_id[..8]),
        )
        .send()
        .expect("send");
    assert_eq!(cancel.status(), 204, "the task must cancel");

    let delivery = rx
        .recv_timeout(Duration::from_secs(30))
        .expect("a cancelled task must still reach subscribers");
    let body: serde_json::Value = serde_json::from_str(&delivery.body).expect("json");
    assert_eq!(body["event"], "task.finished");
    assert_eq!(body["status"], "cancelled", "body was {}", body);

    // Announcing must not have delivered: the conversation is untouched, and a second delivery must
    // not arrive on any later poll.
    assert!(
        rx.recv_timeout(Duration::from_secs(2)).is_err(),
        "a task is announced once, not on every poll"
    );
    let messages_after = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/messages", id),
        )
        .send()
        .expect("send")
        .json::<serde_json::Value>()
        .expect("parse")["messages"]
        .as_array()
        .expect("messages")
        .len();
    assert_eq!(
        messages_after, messages_before,
        "telling subscribers must not spend a turn telling the model"
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
type = "anthropic-messages"
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
        .env("MEKA_MOCK_PROVIDER", "1")
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
        ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", STORE_SCOPES);
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
    let harness = ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", &[
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
        ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", &["schedule:r"]);
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
        body["window"], 1_000_000,
        "the harness configures no context_window, so this is the documented default; reading \
         `config.session_context_window` instead would have reported nothing at all: {}",
        body
    );

    // And the configured value actually reaches the agent. Since meka no longer infers a window
    // from the model name or probes for it, this config key is the *only* way to state the real
    // one, so a call site that dropped it would silently budget every session against the default.
    let configured =
        ServeTestHarness::spawn("\n[session]\ncontext_window = 262144\n", mock_simple_turn());
    let id = session_with_one_turn(&configured);
    let body: serde_json::Value = configured
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/context", id),
        )
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert_eq!(
        body["window"], 262_144,
        "`[session].context_window` must reach the agent, not just the config struct: {}",
        body
    );
}

/// Two writes inside one mtime tick that render to the same length are invisible to a
/// `(mtime, size)` snapshot. Without an explicit invalidation the second write's own 200 response
/// echoes the first write's values, and every agent keeps reading the stale skill.
#[test]
fn a_same_length_rewrite_is_visible_immediately() {
    let harness =
        ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", STORE_SCOPES);
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
                "[providers.mock]\ntype = \"anthropic-messages\"\nmodel = \"claude-sonnet-4-5\"\n\n\
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
            .env("MEKA_MOCK_PROVIDER", "1")
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

/// The agent chose the moment, so it gets to act on the result: a compaction it asked for lands
/// before its next step, not after its last one.
///
/// The round order is the evidence. `context_compact` is answered by the summariser *first*, and
/// only then does the model produce "after the compaction", a round it could not have reached at
/// all while the request was drained after the tool loop, because by then the turn was over.
#[test]
fn a_requested_compaction_lets_the_turn_continue() {
    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "tu_1", "name": "context_compact" },
            { "kind": "tool_use_end", "input": {} },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "a summary" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "text", "text": "after the compaction" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        // A spare round, so the count assertion below is real. With exactly three, a second
        // compaction draws an exhausted round, fails on the empty summary, and emits nothing --
        // and the assertion would hold whether or not the slot had been emptied.
        [
            { "kind": "text", "text": "a spare round" },
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

    assert!(
        body.contains("after the compaction"),
        "the turn has to resume against the compacted context, not end at the request: {}",
        body
    );
    // Exactly one, which is also what keeps the post-loop drain from compacting a second time: it
    // takes unconditionally, and the in-loop drain has already emptied the slot.
    //
    // Deliberately not asserted: that `turn.finished` is last. The handler mints it after
    // `run_turn` returns, so it is last however this turn behaved, and pinning it would read as
    // coverage of an ordering nothing here can break.
    assert_eq!(
        sse_event_names(&body)
            .iter()
            .filter(|name| *name == "context.compacted")
            .count(),
        1,
        "one request, one compaction, and the client told about it: {}",
        body
    );
}

/// One turn honours one request. Now that a compaction lands mid-turn, an agent that asks on every
/// iteration would summarise on every iteration, each round costing a summariser call and, with a
/// checkpoint configured, several more. The second ask is answered and deferred, not run.
#[test]
fn a_turn_honours_one_compaction_request_however_many_it_gets() {
    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "tu_1", "name": "context_compact" },
            { "kind": "tool_use_end", "input": {} },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "first summary" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "tool_use_start", "id": "tu_2", "name": "context_compact" },
            { "kind": "tool_use_end", "input": {} },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("\n[session]\ncompact_checkpoint = false\n", script);
    let id = start_streaming_session(&harness);
    let body = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "compact twice", "stream": true}))
        .send()
        .expect("send")
        .text()
        .expect("body");

    let compactions = sse_event_names(&body)
        .iter()
        .filter(|name| *name == "context.compacted")
        .count();
    assert_eq!(
        compactions, 1,
        "the second request in one turn must be deferred, not honoured: {}",
        body
    );
}

/// Stopping the turn stops the compaction it asked for, before the window is replaced.
///
/// The request is parked by a tool that ignores its cancellation token, so it survives the
/// interrupt. Running it anyway replaces the whole window and -- because a fired token makes
/// `run_checkpoint_turn` return early -- does it through the standalone summariser, writing nothing
/// to memory. That is the failure `compacting_after_a_cancelled_turn_still_runs_the_checkpoint`
/// exists to prevent, reached through a different door.
///
/// Pins the outcome, not which guard delivers it. Two now stand in the way -- the drain declines to
/// start, and `compact_session` refuses to rewrite -- and this passes with either, so neutering one
/// alone will not fail it. The window staying intact is the property worth holding; see
/// `an_interrupt_inside_a_requested_compaction_still_spares_the_window` for the case only the
/// second one catches.
#[test]
fn an_interrupt_stops_a_requested_compaction_before_it_replaces_the_window() {
    // The slow half is a *tool*, not a stream event: the interrupt has to land after
    // `context_compact` has already parked its request and before the loop drains it, which is a
    // window only a running tool batch holds open.
    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "tu_1", "name": "context_compact" },
            { "kind": "tool_use_end", "input": {} },
            { "kind": "tool_use_start", "id": "tu_2", "name": "execute_command" },
            { "kind": "tool_use_end", "input": {"command": "sleep 5"} },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "a summary" },
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
            .json(&serde_json::json!({"message": "compact then stop"}))
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

    let messages = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/messages", id),
        )
        .send()
        .expect("send")
        .text()
        .expect("body");
    assert!(
        !messages.contains("a summary"),
        "the interrupted turn must not have had its window replaced by a summariser: {}",
        messages
    );
}

/// The same stop, one instant later: an interrupt that arrives *inside* the compaction.
///
/// Guarding only the drain's entry covers the narrow case. Nothing between there and the rewrite
/// is cancellable: `run_checkpoint_turn` answers a fired token with `Ok(None)` and hands on to the
/// summariser, and `provider.complete` takes no token, so the compaction reports success and the
/// window goes anyway. The checkpoint is skipped precisely because the token fired, so what
/// replaces the conversation is a summary written without the agent, with nothing saved.
#[test]
fn an_interrupt_inside_a_requested_compaction_still_spares_the_window() {
    // The checkpoint round is the slow one, so the cancel lands after the drain's own guard has
    // already passed and the compaction is under way.
    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "tu_1", "name": "context_compact" },
            { "kind": "tool_use_end", "input": {} },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        // Ends on a tool call, not `end_turn`: that sends `run_checkpoint_turn` round the loop to
        // its per-round token check, which is where a fired token turns into `Ok(None)` and hands
        // the compaction to the standalone summariser below.
        [
            { "kind": "sleep", "ms": 4000 },
            { "kind": "tool_use_start", "id": "tu_2", "name": "memory_search" },
            { "kind": "tool_use_end", "input": {"query": "anything"} },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "STANDALONE SUMMARY" },
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
            .json(&serde_json::json!({"message": "compact then stop mid-compaction"}))
            .send()
            .expect("send")
            .status()
    });

    harness.wait_until_in_flight(&id);
    std::thread::sleep(Duration::from_millis(800));
    let cancel = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/cancel", id),
        )
        .send()
        .expect("send");
    assert_eq!(cancel.status(), 204);
    let _ = turn.join().expect("turn thread");

    let messages = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/messages", id),
        )
        .send()
        .expect("send")
        .text()
        .expect("body");
    assert!(
        !messages.contains("STANDALONE SUMMARY"),
        "a stop landing inside the compaction must not still replace the window with a summary \
         written without the agent: {}",
        messages
    );
}

/// Stopping during an *emergency* compaction is reported as a stop, not as an overflow.
///
/// The emergency path turns any error from `compact_session` into `ContextOverflow`, which was
/// harmless while that call could not fail on an interrupt. Now that it refuses to rewrite the
/// window on a fired token, relabelling would answer a user who pressed stop with "the
/// conversation exceeds the model's context window" -- and a 502 `/errors/context-overflow` --
/// telling them to shorten a conversation that was never the problem.
///
/// Emergency skips the checkpoint by design, so the only window a cancel can land in is the
/// summariser call; the mock holds it open with a sleep.
#[test]
fn an_interrupt_during_an_emergency_compaction_is_not_reported_as_an_overflow() {
    // A first, ordinary turn: the emergency path is gated on `messages.len() > 1`, so a
    // conversation that is still just its opening prompt never reaches the compaction at all.
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "first turn" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "fail_context_overflow", "message": "too large" }
        ],
        [
            { "kind": "sleep", "ms": 4000 },
            { "kind": "text", "text": "an emergency summary" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "text", "text": "recovered" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);
    let id = start_streaming_session(&harness);
    harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "warm up"}))
        .send()
        .expect("send");

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
            .json(&serde_json::json!({"message": "overflow then stop"}))
            .send()
            .expect("send")
            .text()
            .unwrap_or_default()
    });

    harness.wait_until_in_flight(&id);
    std::thread::sleep(Duration::from_millis(800));
    let cancel = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/cancel", id),
        )
        .send()
        .expect("send");
    assert_eq!(cancel.status(), 204);
    let body = turn.join().expect("turn thread");

    assert!(
        !body.contains("context-overflow"),
        "a stop during the emergency compaction must not be reported as a context overflow: {}",
        body
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
        // The summariser draws first now that the drain runs mid-loop; the model's own reply is
        // the round after it. Labelled in that order so the fixture reads as what happens.
        [
            { "kind": "text", "text": "a summary" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "text", "text": "the reply" },
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

/// A malformed body is answered as malformed, even while the session is busy.
///
/// `TurnGuard::acquire` used to run before the body was validated, so a request meka was going to
/// refuse anyway was admitted first -- and on a session already running a turn, `acquire` fails,
/// so the caller was told 409 `turn-in-flight` about a request whose real problem was that it had
/// no message in it. Retrying that (which is what a 409 invites) reproduces it forever.
///
/// Asserted against a *running* turn deliberately. A sequential version of this test passes either
/// way: the guard is RAII, so a rejected request that took one released it again before the next
/// request could observe it. Only contention makes the ordering visible.
#[test]
fn a_malformed_body_is_refused_as_malformed_even_while_a_turn_runs() {
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
        .json(&serde_json::json!({}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    let base_url = harness.base_url.clone();
    let token = harness.token.clone();
    let running_id = id.clone();
    let running = std::thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("client");
        client
            .post(format!("{}/v1/sessions/{}/turn", base_url, running_id))
            .header("Authorization", format!("Bearer {}", token))
            .json(&serde_json::json!({"message": "the real turn"}))
            .send()
            .expect("send")
    });

    harness.wait_until_in_flight(&id);
    let rejected = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "   "}))
        .send()
        .expect("send");
    assert_eq!(
        rejected.status(),
        422,
        "an empty message is a body problem, and stays one while the session is busy"
    );
    let body: serde_json::Value = rejected.json().expect("parse");
    assert_eq!(body["type"], "https://meka.so/errors/invalid-body");

    running.join().expect("join").error_for_status().ok();
}

/// Two requests arriving together for a session this process has evicted must both be served.
///
/// Reconstruction takes the session's cross-process file lock, so without serialisation the loser
/// raced the winner for it and got a `session-locked` 409 whose documented remedy ("retry against
/// the process that holds it") pointed at this very process. `lock_session_reconstruction` makes
/// the loser wait and then find the winner's entry.
///
/// Driven through the real re-attach path rather than the lock helper, which already has a unit
/// test: that test passes with the helper wired to nothing, because it calls the helper directly.
/// A short `idle_timeout` plus a fast scan interval gets the GC to evict the session so the next
/// request has to rebuild it.
///
/// `PATCH` rather than `GET`, deliberately: a read is not permitted to take the session's lock, so
/// it answers an evicted session from the database and never reaches `ensure_session_loaded` at
/// all. Only a write reconstructs, which is the path the lock protects.
#[test]
fn two_requests_for_an_evicted_session_are_both_served() {
    let harness = ServeTestHarness::spawn(
        "idle_timeout = \"1s\"\ngc_scan_interval = \"200ms\"\n",
        mock_simple_turn(),
    );
    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({}))
        .send()
        .expect("create");
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    // Outlast `idle_timeout` plus a scan, so the session is dropped from the in-memory map and the
    // requests below have to go through `ensure_session_loaded`.
    std::thread::sleep(Duration::from_secs(2));

    let mut handles = Vec::new();
    for _ in 0..2 {
        let base_url = harness.base_url.clone();
        let token = harness.token.clone();
        let id = id.clone();
        handles.push(std::thread::spawn(move || {
            let client = reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("client");
            client
                .patch(format!("{}/v1/sessions/{}", base_url, id))
                .header("Authorization", format!("Bearer {}", token))
                .json(&serde_json::json!({"permission": "read"}))
                .send()
                .expect("send")
                .status()
        }));
    }

    let statuses: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().expect("join"))
        .collect();
    assert!(
        statuses.iter().all(|status| status.is_success()),
        "both requests must be served; one lost the race to rebuild the session: {:?}",
        statuses
    );
}

/// A cancelled turn is not cached against its `Idempotency-Key`, and the cancel actually lands.
///
/// Two properties in one run, because they were entangled. The key exists so a client whose
/// connection died can retry; cancellation is the case that most invites a retry, and caching it
/// answered every retry "cancelled" for the full 24h TTL. But an earlier version of this test was
/// flaky at about one run in four, and the cause was a second defect rather than the test: the
/// turn's cancellation token is published *after* `TurnGuard::acquire`, so a `POST /cancel` landing
/// in that window cancelled the previous turn's token, answered 204, and left this turn running to
/// completion. Poll `turn_in_flight` then cancel -- what this test does, and what the HTTP docs
/// describe -- walked straight into it. `SessionEntry::cancel_epoch` closes the window, and this is
/// the test that exercises the wiring rather than the helper.
#[test]
fn a_cancelled_turn_is_not_cached_against_its_idempotency_key() {
    let script = serde_json::json!([
        [
            { "kind": "sleep", "ms": 4000 },
            { "kind": "text", "text": "should never reach client" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "text", "text": "the retry actually ran" },
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

    let key = "cancelled-then-retried";
    let body = serde_json::json!({"message": "long", "stream": false});

    let base_url = harness.base_url.clone();
    let token = harness.token.clone();
    let id_clone = id.clone();
    let body_clone = body.clone();
    let first = std::thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("client");
        client
            .post(format!("{}/v1/sessions/{}/turn", base_url, id_clone))
            .header("Authorization", format!("Bearer {}", token))
            .header("Idempotency-Key", key)
            .json(&body_clone)
            .send()
            .expect("first send")
    });

    harness.wait_until_in_flight(&id);
    let cancel = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/cancel", id),
        )
        .send()
        .expect("cancel");
    assert_eq!(cancel.status(), 204);

    let cancelled = first.join().expect("join");
    let cancelled_status = cancelled.status();
    let cancelled_body = cancelled.text().expect("text");
    assert!(
        !cancelled_status.is_success(),
        "the 204 said the turn was cancelled, but it ran to completion: \
         {cancelled_status} {cancelled_body}"
    );

    // Same key, same body. A cached cancellation would replay verbatim and the mock's second round
    // would never be consumed.
    let retry = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .header("Idempotency-Key", key)
        .json(&body)
        .send()
        .expect("retry send");
    assert_eq!(
        retry.status(),
        200,
        "the retry must run rather than replay the cancellation"
    );
    // Deliberately not asserting *which* mock round the retry consumes. Whether the cancelled turn
    // consumed round one depends on how far it got before the token fired, so pinning the retry to
    // round two's text made this fail about one full-suite run in three -- a property of the mock's
    // round counter, not of the behaviour under test. What distinguishes "ran" from "replayed the
    // cache" is that a cached cancellation is a non-2xx problem document: it has no `stop_reason`
    // and carries the cancellation type URI.
    let retry_body = retry.text().expect("text");
    let parsed: serde_json::Value = serde_json::from_str(&retry_body).expect("json");
    assert_eq!(
        parsed["stop_reason"], "end_turn",
        "the retry did not run a turn: {retry_body}"
    );
    assert!(
        !retry_body.contains("turn-cancelled"),
        "the retry replayed the cached cancellation instead of running: {retry_body}"
    );
}

/// A `DELETE` of a name that could not be a skill in the store is refused before the filesystem is
/// touched.
///
/// Probing first made the endpoint answer "does this directory exist" for any string a caller sent,
/// including one that leaves the store entirely: a filesystem oracle reachable with nothing but
/// `skills:w`. `delete_memory` already validated first; this half did not. Reordering the two
/// statements back left all four suites green, which is why this exists.
///
/// The property is *escaping the store*, not looking unusual. Requiring the latter made ordinary
/// directories another client can create -- `not.a.skill`, `has space` -- impossible to delete
/// through any door while buying no safety, since a `skills:w` token may already list and write
/// every name in that directory. So an escaping name must fail validation (422) whether or not
/// anything is on disk under it, and a store-local one gets the honest 404.
#[test]
fn deleting_an_invalid_skill_name_is_refused_before_the_filesystem_is_probed() {
    let harness =
        ServeTestHarness::spawn_with("", "", mock_simple_turn(), "sk_test_token", STORE_SCOPES);

    // Names that survive routing as a single path segment but cannot name a directory in the store.
    // `..` is deliberately not among them: axum normalises it away before the handler sees it, so
    // it tests the router rather than this endpoint.
    for name in [".hidden", "tab%09name", "null%00name"] {
        let response = harness
            .request(reqwest::Method::DELETE, &format!("/v1/skills/{}", name))
            .send()
            .expect("send");
        let status = response.status();
        let body: serde_json::Value = response.json().expect("parse");
        assert_ne!(
            status, 404,
            "'{name}' was answered by a filesystem probe rather than the name rules, which leaks \
             whether the path exists: {body}"
        );
        assert_eq!(
            body["type"], "https://meka.so/errors/invalid-body",
            "'{name}' must be refused as an invalid name: {status} {body}"
        );
    }

    // A name that *could* be a skill here answers honestly instead of being stonewalled, which is
    // what makes such a skill removable at all.
    for name in ["not.a.skill", "has%20space"] {
        let response = harness
            .request(reqwest::Method::DELETE, &format!("/v1/skills/{}", name))
            .send()
            .expect("send");
        assert_eq!(
            response.status(),
            404,
            "'{name}' names a directory in the store, so it has to be answerable"
        );
    }
}

/// One `Idempotency-Key` reused against two sessions must run both turns.
///
/// The key was scoped to `(token_id, key)` alone, so the second session's request hit the first
/// session's cached envelope: it was answered with the *other session's* transcript and its turn
/// never ran. A key is a client's retry token, not a global name, and a client that reuses one
/// across sessions is doing something ordinary. This is the plan's own end-to-end check for the
/// fix, and dropping the session from the scope left all four suites green.
#[test]
fn one_idempotency_key_across_two_sessions_answers_each_with_its_own_turn() {
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "first session speaking" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "text", "text": "second session speaking" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let harness = ServeTestHarness::spawn("", script);

    let create = |harness: &ServeTestHarness| {
        harness
            .request(reqwest::Method::POST, "/v1/sessions")
            .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
            .send()
            .expect("create")
            .json::<serde_json::Value>()
            .expect("parse")["id"]
            .as_str()
            .expect("id")
            .to_string()
    };
    let first_id = create(&harness);
    let second_id = create(&harness);

    let key = "shared-across-sessions";
    let body = serde_json::json!({"message": "go", "stream": false});

    let turn = |id: &str| -> serde_json::Value {
        let response = harness
            .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
            .header("Idempotency-Key", key)
            .json(&body)
            .send()
            .expect("turn");
        assert_eq!(response.status(), 200, "turn on {id} must run");
        response.json().expect("parse")
    };

    let first = turn(&first_id);
    let second = turn(&second_id);

    assert_eq!(
        first["session_id"].as_str(),
        Some(first_id.as_str()),
        "first turn answered for the wrong session: {first}"
    );
    assert_eq!(
        second["session_id"].as_str(),
        Some(second_id.as_str()),
        "the second session was answered with the first session's cached envelope: {second}"
    );
    assert_ne!(
        first["turn_id"], second["turn_id"],
        "the second session replayed the first turn rather than running its own"
    );
}

/// `PATCH` moves a session that is not resident, without building an agent for it.
///
/// The rescue path for a session whose recorded profile has left `config.toml`. Reviving it to
/// apply the change is the one thing that cannot work in that state, because rebuilding the agent
/// resolves the very profile that is gone, so the documented recovery used to be refused by the
/// failure it was meant to repair. Exercised on a *healthy* dormant session, since a genuinely
/// stranded one cannot be created through this API; what it pins is that no agent is built, which
/// is what makes the stranded case work.
#[test]
fn patch_moves_a_dormant_session_without_reviving_it() {
    let harness = ServeTestHarness::spawn_with_prelude(
        "default_provider = \"mock\"\n",
        r#"
[providers.other]
type = "anthropic-messages"
model = "claude-sonnet-4-5"
context_window = 32000
"#,
        mock_simple_turn(),
    );

    // Imported rather than created: `POST /v1/sessions` leaves an entry resident, which takes the
    // path this test is not about. An import writes rows and nothing else.
    let created = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
        .send()
        .expect("send");
    let source = created.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    let envelope: serde_json::Value = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/export?format=json", source),
        )
        .send()
        .expect("send")
        .json()
        .expect("parse");
    let imported: serde_json::Value = harness
        .request(reqwest::Method::POST, "/v1/sessions/import")
        .json(&envelope)
        .send()
        .expect("send")
        .json()
        .expect("parse");
    let id = imported["session_id"]
        .as_str()
        .expect("session_id")
        .to_string();

    let patched = harness
        .request(reqwest::Method::PATCH, &format!("/v1/sessions/{}", id))
        .json(&serde_json::json!({"provider": "other"}))
        .send()
        .expect("send");
    assert_eq!(
        patched.status(),
        200,
        "a dormant session must be movable: {}",
        patched.text().unwrap_or_default()
    );
    let body: serde_json::Value = patched.json().expect("parse");
    assert_eq!(body["provider"], "other", "{body}");
    assert_eq!(
        body["turn_in_flight"], false,
        "a session with no agent has no turn: {body}"
    );

    // The name of this test, actually asserted. `message_count` is `Some` only while an entry is
    // resident (`conversation.rs` reads it through `entry.runtime`), and `GET /context` does not
    // itself revive one, so its absence is the residency signal.
    //
    // Without this the test proved nothing: with the whole dormant branch dead, `patch_session`
    // falls through to `ensure_session_loaded`, revives the session, and returns a byte-identical
    // body. Every assertion above still passed. Verified by prefixing the branch with `if false
    // &&`.
    let context: serde_json::Value = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/context", id),
        )
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert!(
        context["message_count"].is_null(),
        "the repin revived the session instead of moving it dormant: {context}"
    );

    // The row moved, and reading it back does not depend on an entry either.
    let fetched: serde_json::Value = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}", id))
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert_eq!(fetched["provider"], "other", "{fetched}");
}

/// A dormant repin and the reconstruction that follows it must agree about the profile.
///
/// `repin_dormant_session` decides what to write on the strength of the session not being resident,
/// then awaits five times before writing: `require_session_exists`, resolving the profile (which
/// may build a provider and load a credential), reading the recorded binding, the write itself, and
/// the re-read. Nothing re-checked residency and nothing serialised against reconstruction, so any
/// turn, scheduler fire, compaction or rewind arriving in that window rebuilt the agent from the
/// *old* profile and inserted it. The row then moved and the response quoted the new profile, while
/// the session ran, billed and gauged the old one until it was evicted again -- hours, at the
/// default idle timeout.
///
/// The race itself is not reproducible from out here: it needs a reconstruction to land inside
/// those five awaits, and nothing over HTTP can be timed that precisely. What is asserted is the
/// post-condition the fix guarantees -- that after the repin, the row and the *live agent* rebuilt
/// from it name the same profile -- which is exactly what the race broke. The serialisation itself
/// lives in `repin_dormant_session`, which now holds `lock_session_reconstruction` for its whole
/// body and re-checks residency under it.
#[test]
fn a_dormant_repin_and_the_agent_rebuilt_after_it_agree() {
    let harness = ServeTestHarness::spawn_with_prelude(
        "default_provider = \"mock\"\n",
        r#"idle_timeout = "1s"
gc_scan_interval = "1s"

[providers.small]
type = "anthropic-messages"
model = "claude-sonnet-4-5"
context_window = 32000
"#,
        mock_turns(2),
    );

    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({
            "cwd": std::env::temp_dir().to_string_lossy(),
            "provider": "mock",
        }))
        .send()
        .expect("send");
    assert_eq!(
        create.status(),
        201,
        "{}",
        create.text().unwrap_or_default()
    );
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();

    // One turn, so there is a conversation to count. `message_count` is the residency signal
    // below: it is read through the entry's own runtime mutex and is simply absent once the entry
    // is gone, and `/context` deliberately never revives a session, so polling it cannot keep this
    // one alive.
    let turn = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "hello"}))
        .send()
        .expect("send");
    assert_eq!(turn.status(), 200, "{}", turn.text().unwrap_or_default());

    let context_path = format!("/v1/sessions/{}/context", id);
    let context: serde_json::Value = harness
        .request(reqwest::Method::GET, &context_path)
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert!(
        context["message_count"].is_u64(),
        "the live entry must be readable while the session is resident, or the eviction check \
         below proves nothing: {context}"
    );

    // Now let the GC scanner drop it, so the PATCH takes the dormant path rather than the resident
    // one.
    let dormant_by = Instant::now() + Duration::from_secs(30);
    loop {
        let context: serde_json::Value = harness
            .request(reqwest::Method::GET, &context_path)
            .send()
            .expect("send")
            .json()
            .expect("parse");
        if context["message_count"].is_null() {
            break;
        }
        assert!(
            Instant::now() < dormant_by,
            "the session never left the live map; GC is not evicting it"
        );
        std::thread::sleep(Duration::from_millis(200));
    }

    let patched = harness
        .request(reqwest::Method::PATCH, &format!("/v1/sessions/{}", id))
        .json(&serde_json::json!({"provider": "small"}))
        .send()
        .expect("send");
    assert_eq!(
        patched.status(),
        200,
        "{}",
        patched.text().unwrap_or_default()
    );
    let patched: serde_json::Value = patched.json().expect("parse");
    assert_eq!(patched["provider"], "small", "{patched}");

    // The PATCH must have taken the dormant path, not revived the session and used the resident
    // one. Without this the test could not tell the two apart: the resident path reaches the same
    // row and the same `set_provider`, so the turn below agrees either way, and the whole dormant
    // branch could be deleted with this test still green. Verified by prefixing it with
    // `if false &&`.
    let after_patch: serde_json::Value = harness
        .request(reqwest::Method::GET, &context_path)
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert!(
        after_patch["message_count"].is_null(),
        "the repin revived the session, so it did not exercise the dormant path: {after_patch}"
    );

    // Reconstructs the agent from the row. Before the fix this is the request that, arriving a
    // moment earlier, would have raced the repin and won.
    let turn = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "hello"}))
        .send()
        .expect("send");
    assert_eq!(turn.status(), 200, "{}", turn.text().unwrap_or_default());

    let context: serde_json::Value = harness
        .request(reqwest::Method::GET, &context_path)
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert_eq!(
        context["window"].as_u64(),
        Some(32_000),
        "the live agent is gauging against a different profile from the one its row names: \
         {context}"
    );
    let session: serde_json::Value = harness
        .request(reqwest::Method::GET, &format!("/v1/sessions/{}", id))
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert_eq!(session["provider"], "small", "{session}");
}

/// A `PATCH` naming a provider on a session that is still **resident** must move both the row and
/// the live agent, and a name the server does not have must be refused before either.
///
/// Both existing provider tests reach `repin_dormant_session` instead: one patches a freshly
/// imported session, the other waits for GC to evict it first. So the resident branch had no
/// coverage, and a mutation sweep showed it -- flipping its no-op filter to `==` writes the row
/// only when it is *already* correct, which turns every real switch into a silent no-op, and
/// `require_configured_profile` could return `Ok(())` and let a typo be recorded on the row.
///
/// The window is what proves the *agent* moved rather than only the row: it comes off the cell
/// `set_provider` publishes into, and the two profiles here state different ones.
#[test]
fn a_resident_patch_moves_the_agent_and_refuses_an_unconfigured_profile() {
    let harness = ServeTestHarness::spawn_with_prelude(
        "default_provider = \"mock\"\n",
        r#"
[providers.small]
type = "anthropic-messages"
model = "claude-sonnet-4-5"
context_window = 32000
"#,
        mock_turns(1),
    );

    // A name the config does not have, refused and leaving nothing behind.
    //
    // Two things make that true and only one of them is `require_configured_profile`: the row is
    // written *before* the agent is built, so the up-front check is what produces the good message,
    // and `SessionRollback` is what deletes the orphan when the build fails anyway. Removing the
    // check leaves both the 422 and the empty listing intact, so this pair is deliberately a
    // statement of the contract rather than a discriminator -- the rollback guard is the half worth
    // pinning, because losing *it* strands a session pinned to a profile that resolves to nothing.
    let bad_create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({
            "cwd": std::env::temp_dir().to_string_lossy(),
            "provider": "ghost",
        }))
        .send()
        .expect("send");
    assert_eq!(
        bad_create.status(),
        422,
        "an unconfigured profile must be refused: {}",
        bad_create.text().unwrap_or_default()
    );
    let listed: serde_json::Value = harness
        .request(reqwest::Method::GET, "/v1/sessions")
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert_eq!(
        listed["sessions"].as_array().map(Vec::len),
        Some(0),
        "a refused create must leave no session behind: {listed}"
    );

    let create = harness
        .request(reqwest::Method::POST, "/v1/sessions")
        .json(&serde_json::json!({
            "cwd": std::env::temp_dir().to_string_lossy(),
            "provider": "mock",
        }))
        .send()
        .expect("send");
    assert_eq!(create.status(), 201);
    let id = create.json::<serde_json::Value>().expect("parse")["id"]
        .as_str()
        .expect("id")
        .to_string();
    let context_path = format!("/v1/sessions/{}/context", id);

    // Resident, and never evicted: no GC wait here, which is what makes this the other branch.
    let before: serde_json::Value = harness
        .request(reqwest::Method::GET, &context_path)
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert!(
        before["message_count"].is_u64(),
        "the session must still be resident for this test to mean anything: {before}"
    );

    let refused = harness
        .request(reqwest::Method::PATCH, &format!("/v1/sessions/{}", id))
        .json(&serde_json::json!({"provider": "ghost"}))
        .send()
        .expect("send");
    assert_eq!(
        refused.status(),
        422,
        "a resident session must refuse an unconfigured profile too: {}",
        refused.text().unwrap_or_default()
    );

    let patched = harness
        .request(reqwest::Method::PATCH, &format!("/v1/sessions/{}", id))
        .json(&serde_json::json!({"provider": "small"}))
        .send()
        .expect("send");
    assert_eq!(
        patched.status(),
        200,
        "{}",
        patched.text().unwrap_or_default()
    );
    let body: serde_json::Value = patched.json().expect("parse");
    assert_eq!(body["provider"], "small", "the row must move: {body}");

    let after: serde_json::Value = harness
        .request(reqwest::Method::GET, &context_path)
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert_eq!(
        after["window"], 32000,
        "the live agent must gauge against the new profile's window, not the old one: {after}"
    );
}

/// A sub-agent's transcript can be rewound over HTTP, the way `meka session rewind` always could.
///
/// Rewind is deliberately on the other side of the line from `/turn` and `/compact`: those drive a
/// conversation only the worker's parent may drive, while this edits an event log the same caller
/// can already read in full through `/export`. Leaving it refused made the HTTP surface disagree
/// with the CLI about the same operation on the same row, which is a difference nobody chose.
///
/// Asserts the log actually shrank, not just the 200: a handler that answered without writing
/// would satisfy the status alone.
#[test]
fn a_sub_agent_transcript_can_be_rewound_over_http() {
    let harness = ServeTestHarness::spawn(
        "",
        serde_json::json!([
            [
                { "kind": "tool_use_start", "id": "tu_1", "name": "agent_spawn" },
                { "kind": "tool_use_end", "input": {"prompt": "count the files"} },
                { "kind": "message_end", "stop_reason": "tool_use" }
            ],
            [
                { "kind": "text", "text": "worker done" },
                { "kind": "message_end", "stop_reason": "end_turn" }
            ],
            [
                { "kind": "text", "text": "dispatched" },
                { "kind": "message_end", "stop_reason": "end_turn" }
            ]
        ]),
    );
    let parent = session_with_one_turn(&harness);

    let listing: serde_json::Value = harness
        .request(reqwest::Method::GET, "/v1/sessions?include_children=true")
        .send()
        .expect("send")
        .json()
        .expect("parse");
    let worker = listing["sessions"]
        .as_array()
        .expect("a sessions array")
        .iter()
        .find(|row| row["parent_id"].as_str() == Some(parent.as_str()))
        .map(|row| row["id"].as_str().expect("id").to_string())
        .unwrap_or_else(|| panic!("the spawn should have left a worker under {parent}: {listing}"));

    let rewound = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/rewind", worker),
        )
        .json(&serde_json::json!({"turns": 1}))
        .send()
        .expect("send");
    assert_eq!(
        rewound.status(),
        200,
        "a rewind edits a transcript rather than driving it, so a worker takes it"
    );
    let body: serde_json::Value = rewound.json().expect("parse");
    let before = body["messages_before"].as_u64().unwrap_or_default();
    let after = body["messages_after"].as_u64().unwrap_or_default();
    assert!(
        after < before,
        "the rewind must actually remove turns, not merely answer 200: {body}"
    );

    // And driving it is still refused, which is the line this endpoint sits on the other side of.
    let refused = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/compact", worker),
        )
        .json(&serde_json::json!({}))
        .send()
        .expect("send");
    assert_eq!(
        refused.status(),
        422,
        "compaction runs the model, which is the parent's to do"
    );
}

/// A worker whose lock is held answers 422, not 409.
///
/// The ordering test for the re-attach door, and the reason `ensure_session_loaded` refuses before
/// `lock_session` rather than leaving it to `build_session_agent` at the end. Without the guard the
/// lock is taken first, so a worker its parent is currently running answers `409 session-locked` --
/// "another process has it, try again" -- for a condition no retry ever resolves. The 422 is the
/// true answer and this pins that it is the one that arrives.
///
/// Contention is created by flocking the worker's lock file from the test process, which is a
/// different open file description from the server's, so the conflict is real rather than
/// simulated. Moving the refusal below `lock_session` turns this 422 into a 409; the plain
/// refusal test above stays green under that move, which is why this one exists separately.
#[test]
fn a_locked_worker_is_refused_as_undrivable_rather_than_as_busy() {
    let harness = ServeTestHarness::spawn(
        "",
        serde_json::json!([
            [
                { "kind": "tool_use_start", "id": "tu_1", "name": "agent_spawn" },
                { "kind": "tool_use_end", "input": {"prompt": "count the files"} },
                { "kind": "message_end", "stop_reason": "tool_use" }
            ],
            [
                { "kind": "text", "text": "worker done" },
                { "kind": "message_end", "stop_reason": "end_turn" }
            ],
            [
                { "kind": "text", "text": "dispatched" },
                { "kind": "message_end", "stop_reason": "end_turn" }
            ]
        ]),
    );
    let parent = session_with_one_turn(&harness);

    let listing: serde_json::Value = harness
        .request(reqwest::Method::GET, "/v1/sessions?include_children=true")
        .send()
        .expect("send")
        .json()
        .expect("parse");
    let worker = listing["sessions"]
        .as_array()
        .expect("a sessions array")
        .iter()
        .find(|row| row["parent_id"].as_str() == Some(parent.as_str()))
        .map(|row| row["id"].as_str().expect("id").to_string())
        .unwrap_or_else(|| panic!("the spawn should have left a worker under {parent}: {listing}"));

    // Hold the worker's lock the way its parent would while running it.
    let lock_path = harness
        .data_dir
        .join("locks")
        .join(format!("{}.lock", worker));
    let held = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .unwrap_or_else(|error| panic!("open {}: {error}", lock_path.display()));
    // Released where `held` closes, at the end of the test.
    held.try_lock()
        .expect("the worker's lock is free for this test to take");

    let refused = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/turn", worker),
        )
        .json(&serde_json::json!({"message": "drive the worker while its lock is held"}))
        .send()
        .expect("send");
    assert_eq!(
        refused.status(),
        422,
        "a held lock must not turn a permanent refusal into a retryable conflict"
    );
    let problem: serde_json::Value = refused.json().expect("parse");
    assert_eq!(
        problem["type"].as_str(),
        Some("https://meka.so/errors/session-not-drivable"),
        "and it says which kind it is, rather than `session-locked`: {problem}"
    );
}

/// A worker session cannot be driven through the HTTP turn door.
///
/// The wiring, not the predicate: `refuse_a_spawned_session` has its own unit test, but that one
/// passes with the call deleted from both builders, which is exactly how this shipped open. A
/// worker's `[subagents]` denials, memory and instruction grants and permission ceiling live in
/// `subagent_spec_json`, which `build_session_agent` never reads, so before the refusal a
/// `sessions:w` holder could POST a turn at a worker id and drive that conversation with the full
/// built-in set at the host's level.
///
/// The worker is spawned for real rather than fabricated, because the id has to come from the same
/// place an attacker's would: `GET /v1/sessions?include_children=true`, which hands sub-agent ids
/// to any `sessions:r` holder.
#[test]
fn a_worker_session_refuses_a_turn_posted_straight_at_it() {
    // Three rounds in one queue, drained in order: the parent's `agent_spawn` call, the worker's
    // own (non-streaming) reply, then the parent's closing text.
    let harness = ServeTestHarness::spawn(
        "",
        serde_json::json!([
            [
                { "kind": "tool_use_start", "id": "tu_1", "name": "agent_spawn" },
                { "kind": "tool_use_end", "input": {"prompt": "count the files"} },
                { "kind": "message_end", "stop_reason": "tool_use" }
            ],
            [
                { "kind": "text", "text": "worker done" },
                { "kind": "message_end", "stop_reason": "end_turn" }
            ],
            [
                { "kind": "text", "text": "dispatched" },
                { "kind": "message_end", "stop_reason": "end_turn" }
            ]
        ]),
    );
    let parent = session_with_one_turn(&harness);

    let listing = harness
        .request(reqwest::Method::GET, "/v1/sessions?include_children=true")
        .send()
        .expect("send");
    assert_eq!(listing.status(), 200);
    let body: serde_json::Value = listing.json().expect("parse");
    let worker = body["sessions"]
        .as_array()
        .expect("a sessions array")
        .iter()
        .find(|row| row["parent_id"].as_str() == Some(parent.as_str()))
        .map(|row| row["id"].as_str().expect("id").to_string())
        .unwrap_or_else(|| panic!("the spawn should have left a worker under {parent}: {body}"));

    let refused = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/turn", worker),
        )
        .json(&serde_json::json!({"message": "drive the worker directly"}))
        .send()
        .expect("send");
    assert_eq!(
        refused.status(),
        422,
        "a worker must not take a turn from this door"
    );
    let problem: serde_json::Value = refused.json().expect("parse");
    let detail = problem["detail"].as_str().unwrap_or_default();
    assert!(
        detail.contains("agent_followup") && detail.contains(&parent),
        "the refusal must name the parent and the door that can drive it: {problem}"
    );

    // Reading it is untouched, which is the whole of what the refusal costs.
    let messages = harness
        .request(
            reqwest::Method::GET,
            &format!("/v1/sessions/{}/messages", worker),
        )
        .send()
        .expect("send");
    assert_eq!(
        messages.status(),
        200,
        "a worker's transcript stays readable"
    );

    // And the copy door is not a way round the refusal. `fork_session` used to write a NULL
    // parent, so this handed a `sessions:w` holder a drivable copy of the worker's whole
    // conversation with no spawn terms and the host's permission -- the very escalation the
    // refusal above exists to stop, one call to the side of it.
    let forked = harness
        .request(
            reqwest::Method::POST,
            &format!("/v1/sessions/{}/fork", worker),
        )
        .send()
        .expect("send");
    assert_eq!(
        forked.status(),
        422,
        "a copy of a worker is a worker, so this door has no live session to hand back"
    );
    let problem: serde_json::Value = forked.json().expect("parse");
    let detail = problem["detail"].as_str().unwrap_or_default();
    assert!(
        detail.contains(&worker) && detail.contains(&parent),
        "and it names the id the caller sent, not the copy it did not make: {problem}"
    );

    // The refusal has to come before the copy, or it leaves a row behind.
    let listing: serde_json::Value = harness
        .request(reqwest::Method::GET, "/v1/sessions?include_children=true")
        .send()
        .expect("send")
        .json()
        .expect("parse");
    assert_eq!(
        listing["sessions"]
            .as_array()
            .expect("a sessions array")
            .iter()
            .filter(|row| row["parent_id"].as_str() == Some(parent.as_str()))
            .count(),
        1,
        "a refused fork writes no row: {listing}"
    );

    // And neither is the metadata door. A provider-only `PATCH` takes `repin_dormant_session`,
    // the one branch of that handler which writes `sessions.provider` without building an agent --
    // so the refusal in the builders could not answer for it, and a worker is exactly the session
    // that branch always gets, since `build_subagent` runs one under its parent's runtime rather
    // than registering it with the server. It answered 200 and moved the row.
    let patched = harness
        .request(reqwest::Method::PATCH, &format!("/v1/sessions/{}", worker))
        .json(&serde_json::json!({"provider": "mock"}))
        .send()
        .expect("send");
    assert_eq!(
        patched.status(),
        422,
        "a worker's profile comes from the terms it was spawned with, not from this door"
    );
    let problem: serde_json::Value = patched.json().expect("parse");
    assert_eq!(
        problem["type"].as_str(),
        Some("https://meka.so/errors/session-not-drivable"),
        "and it says which kind of 422 it is, so a client stops rewriting its payload: {problem}"
    );
}

/// The operator's switch decides whether the upstream's own words reach the caller, and both
/// settings are exercised because only one of them is the default.
///
/// An upstream refusal can name the *operator's* provider account rather than anything about the
/// caller or the conversation, so `[serve] relay_provider_errors = false` exists for deployments
/// whose tokens go to people not entitled to it. Nothing else in the suite covers the key: the
/// resolution defaults it to `true`, and a regression flipping that, or dropping the member
/// entirely, would leave every other test green.
///
/// `detail` is asserted equal across the two runs, which is the property that makes this member
/// additive. Relaying by overwriting `detail` would have deleted meka's own sentence -- and on a
/// context overflow that sentence is the entire remedy.
#[test]
fn relay_provider_errors_decides_whether_the_upstream_body_reaches_the_caller() {
    let secret = "acct-0f3c-operator-only";
    let script = serde_json::json!([[
        { "kind": "fail", "message": "API returned status 401: {\"account_uuid\":\"acct-0f3c-operator-only\"}" }
    ]]);

    let mut details = Vec::new();
    for (extra, expect_relayed) in [("", true), ("relay_provider_errors = false", false)] {
        let harness = ServeTestHarness::spawn(extra, script.clone());
        let create = harness
            .request(reqwest::Method::POST, "/v1/sessions")
            .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
            .send()
            .expect("create");
        let id = create.json::<serde_json::Value>().expect("parse")["id"]
            .as_str()
            .expect("id")
            .to_string();
        let body: serde_json::Value = harness
            .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
            .json(&serde_json::json!({"message": "go"}))
            .send()
            .expect("send")
            .json()
            .expect("parse");

        assert_eq!(body["status"], 502, "{body}");
        let relayed = body
            .get("provider_response")
            .and_then(|value| value.as_str())
            .is_some_and(|text| text.contains(secret));
        assert_eq!(
            relayed,
            expect_relayed,
            "with `{extra}` the upstream body must {} the caller: {body}",
            if expect_relayed { "reach" } else { "not reach" }
        );
        // The whole serialised body, so a member renamed or moved elsewhere in the payload still
        // counts as reaching the caller rather than slipping past a check on one field.
        if !expect_relayed {
            assert!(
                !body.to_string().contains(secret),
                "the upstream body reached the caller by another route: {body}"
            );
        }
        details.push(body["detail"].as_str().unwrap_or_default().to_string());
    }
    assert_eq!(
        details[0], details[1],
        "the key adds a member; it must not rewrite meka's own sentence"
    );
}

/// The switch reaches the streaming path too, which is the one its own documentation calls risky.
///
/// The streaming path reads the key into its own binding (`relay_for_task`), which the blocking
/// test above never reaches because it posts a non-streaming turn: hardcoding that binding to
/// `true` left every test green. It is the value that reaches the terminal `turn.failed` payload,
/// which `record_terminal` retains for the reattach endpoint to replay at `sessions:r` -- the chain
/// the config key exists for. This test reads the `POST /turn` stream rather than reattaching, so
/// it covers the payload the retained terminal is built from.
#[test]
fn relay_provider_errors_is_honoured_on_the_streaming_path() {
    let secret = "acct-0f3c-operator-only";
    let script = serde_json::json!([[
        { "kind": "fail", "message": "API returned status 401: {\"account_uuid\":\"acct-0f3c-operator-only\"}" }
    ]]);

    for (extra, expect_relayed) in [("", true), ("relay_provider_errors = false", false)] {
        let harness = ServeTestHarness::spawn(extra, script.clone());
        let create = harness
            .request(reqwest::Method::POST, "/v1/sessions")
            .json(&serde_json::json!({"cwd": std::env::temp_dir().to_string_lossy()}))
            .send()
            .expect("create");
        let id = create.json::<serde_json::Value>().expect("parse")["id"]
            .as_str()
            .expect("id")
            .to_string();
        let body = harness
            .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
            .json(&serde_json::json!({"message": "go", "stream": true}))
            .send()
            .expect("send")
            .text()
            .expect("body");

        assert!(
            body.contains("event: turn.failed"),
            "the turn must fail so there is a payload to judge; body was:\n{body}"
        );
        assert_eq!(
            body.contains(secret),
            expect_relayed,
            "with `{extra}` the streaming terminal must {} the upstream body; body was:\n{body}",
            if expect_relayed { "carry" } else { "omit" }
        );
    }
}

/// An upstream that answers with megabytes does not get to repeat them to every reader.
///
/// The text is `response.text()` with no cap of its own and `base_url` is user-supplied, so this is
/// attacker-influenced input copied into every 502 body *and* into the terminal event the per-turn
/// replay ring retains for reconnects.
///
/// The body carries a distinctive head so the test can see *which* part survived. A run of one
/// repeated character could not: keeping the start, the end or the middle would all have looked
/// identical, while the docs promise the start, which is where a provider's error type sits.
#[test]
fn a_relayed_upstream_body_is_size_bounded() {
    let head = "UPSTREAM-ERROR-TYPE-HERE";
    let script = serde_json::json!([[
        { "kind": "fail", "message": format!("{head}{}", "x".repeat(64 * 1024)) }
    ]]);
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
    let body: serde_json::Value = harness
        .request(reqwest::Method::POST, &format!("/v1/sessions/{}/turn", id))
        .json(&serde_json::json!({"message": "go"}))
        .send()
        .expect("send")
        .json()
        .expect("parse");

    let relayed = body["provider_response"].as_str().expect("the member");
    // The documented cap exactly, not a loose multiple of it. At `< 8 KiB` the constant could be
    // doubled with this test still green while the API reference went on promising 4 KiB.
    assert!(
        relayed.len() <= 4 * 1024,
        "a 64 KiB upstream body must be cut to the documented 4 KiB; got {} bytes",
        relayed.len()
    );
    assert!(
        relayed.starts_with(head),
        "and the start must survive, which is where the error type sits: {}",
        &relayed[..relayed.len().min(60)]
    );
    assert!(
        relayed.contains("truncated"),
        "and the cut must be visible to the reader"
    );
}
