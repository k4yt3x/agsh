// Integration-test files are their own crate, so the `#![cfg_attr(test, allow(...))]` in
// `src/main.rs` doesn't reach here. Mirror it explicitly: tests rely on `.unwrap()` / `.expect()`
// for clear panic-on-failure semantics, and asserting against panics is the standard idiom.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! End-to-end ACP integration tests. Spawn the real `meka acp` binary with
//! `MEKA_MOCK_PROVIDER=1` so a scripted [`crate::provider::mock::MockProvider`] drives
//! deterministic `session/prompt` round-trips. Tests verify the tool-call lifecycle, permission
//! round-trip, session lifecycle (load / resume / list / close), slash-skill invocation, set_mode
//! flow, and the `fs/*` delegation path.
//!
//! # Test shape
//!
//! Tests should use [`AcpTestHarness`] (and its [`AcpTestHarnessBuilder`] for tests that need to
//! seed the tempdir before spawn). The harness collapses spawn / init / session/new boilerplate to
//! ~3 lines and the [`AcpTestHarnessBuilder::pre_spawn`] hook covers tests whose mock script must
//! reference an on-disk path inside the tempdir.
//!
//! A handful of legacy tests still use the inline
//! `tempfile::tempdir + Command::spawn + stdin/stdout pipes +
//! read_until` shape because the harness contract can't model
//! what they need:
//! - **Multi-spawn persistence tests** (`acp_session_load_replays_persisted_turn`,
//!   `acp_session_resume_adopts_without_replay`, `acp_session_list_filters_by_cwd`,
//!   `acp_session_list_paginates_across_cursor_boundary`) seed a second child process against the
//!   same on-disk session store the first child wrote to. The harness owns its tempdir and has no
//!   "respawn against this existing tempdir" hook.
//! - **Pre-initialize protocol tests** (`acp_initialize_clamps_far_future_version_to_latest`) need
//!   to send a non-default `protocolVersion`, which the harness bakes in during `build()`.
//! - **Bespoke timing tests** (`acp_session_cancel_interrupts_running_prompt`,
//!   `acp_multi_session_parallel_prompts_dont_serialize`) rely on precise `Instant::now()`
//!   measurements outside the harness's deadline window.
//!
//! Everything else (the tool-call lifecycle, permission flows, delegation paths, sub-agent
//! forwarding, and per-session isolation) sits on the harness.

use std::{
    io::{BufRead, BufReader, Write},
    path::Path,
    process::{Child, ChildStdin, Command, Stdio},
    time::{Duration, Instant},
};

fn meka_acp() -> Command {
    Command::new(env!("CARGO_BIN_EXE_meka"))
}

/// Test harness that owns the child process, stdio pipes, and a running deadline. Wraps the spawn /
/// `initialize` / `session/new` boilerplate so each test stays focused on the behavior it
/// exercises. See the module header for the inline-pattern exceptions.
struct AcpTestHarness {
    _temp: tempfile::TempDir,
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<std::process::ChildStdout>,
    /// Drained by the spawned reader thread; never read on the test side except in the (currently
    /// absent) panic-on-missing-response paths. Kept alive so the spawned thread can finish
    /// cleanly.
    #[allow(dead_code)]
    stderr_handle: std::thread::JoinHandle<String>,
    config_dir: std::path::PathBuf,
    next_id: u64,
    deadline: Instant,
}

/// Boxed pre-spawn hook. Type-aliased to keep the builder field declaration readable
/// (clippy::type_complexity).
type PreSpawnHook = Box<dyn FnOnce(&Path) -> serde_json::Value>;

/// Fluent builder for [`AcpTestHarness`]. Tests that need to pre-populate files inside the spawned
/// process's tempdir use [`Self::pre_spawn`] to run a closure with the resolved `config_dir`
/// *before* the child starts. The mock script can reference paths set up there.
#[allow(dead_code)]
#[derive(Default)]
struct AcpTestHarnessBuilder {
    config: String,
    script: Option<serde_json::Value>,
    capabilities: serde_json::Value,
    pre_spawn: Option<PreSpawnHook>,
    config_window: Option<Duration>,
}

#[allow(dead_code)]
impl AcpTestHarnessBuilder {
    fn config(mut self, toml: &str) -> Self {
        self.config = toml.to_string();
        self
    }

    fn script(mut self, value: serde_json::Value) -> Self {
        self.script = Some(value);
        self
    }

    fn script_opt(mut self, value: Option<serde_json::Value>) -> Self {
        self.script = value;
        self
    }

    fn capabilities(mut self, value: serde_json::Value) -> Self {
        self.capabilities = value;
        self
    }

    /// Run `f` with the resolved `config_dir` *before* spawn. The returned JSON value replaces the
    /// script (so the closure can reference real paths under the tempdir).
    fn pre_spawn<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&Path) -> serde_json::Value + 'static,
    {
        self.pre_spawn = Some(Box::new(f));
        self
    }

    /// Override the default 15s deadline.
    fn deadline(mut self, duration: Duration) -> Self {
        self.config_window = Some(duration);
        self
    }

    fn build(self) -> AcpTestHarness {
        let temp = tempfile::tempdir().expect("tempdir");
        let config_dir = temp.path().join("meka");
        let data_dir = temp.path().join("data").join("meka");
        std::fs::create_dir_all(&config_dir).expect("create config dir");
        std::fs::write(config_dir.join("config.toml"), &self.config).expect("write config.toml");

        let script = if let Some(f) = self.pre_spawn {
            f(&config_dir)
        } else {
            self.script
                .unwrap_or_else(|| serde_json::Value::Array(Vec::new()))
        };
        let script_path = temp.path().join("script.json");
        std::fs::write(&script_path, script.to_string()).expect("write script");

        let mut child = meka_acp()
            .arg("acp")
            .env("MEKA_CONFIG_DIR", &config_dir)
            .env("MEKA_DATA_DIR", &data_dir)
            .env("HOME", temp.path())
            .env("MEKA_MOCK_PROVIDER", "1")
            .env("MEKA_MOCK_PROVIDER_SCRIPT", &script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn meka acp");
        let stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let stderr_pipe = child.stderr.take().expect("stderr");
        let reader = BufReader::new(stdout);
        let stderr_handle = std::thread::spawn(move || {
            let mut buf = String::new();
            let mut r = BufReader::new(stderr_pipe);
            while r.read_line(&mut buf).unwrap_or(0) > 0 {}
            buf
        });
        let deadline = Instant::now() + self.config_window.unwrap_or(Duration::from_secs(15));
        let mut harness = AcpTestHarness {
            _temp: temp,
            child,
            stdin,
            reader,
            stderr_handle,
            config_dir,
            next_id: 0,
            deadline,
        };
        let _ = harness.request(
            "initialize",
            serde_json::json!({
                "protocolVersion": 1,
                "clientCapabilities": self.capabilities,
            }),
        );
        harness
    }
}

// Not every helper below is used by every test build (the suite uses different subsets);
// `dead_code` is silenced wholesale for the test-only utility surface.
#[allow(dead_code)]
impl AcpTestHarness {
    /// Spin up `meka acp` against a fresh tempdir with `config_toml` pre-written and
    /// `MEKA_MOCK_PROVIDER` enabled (with an empty script unless `script` is supplied).
    /// Initialise the connection but don't create a session yet.
    fn spawn(config_toml: &str, script: Option<serde_json::Value>) -> Self {
        Self::spawn_with_capabilities(config_toml, script, serde_json::json!({}))
    }

    /// As [`Self::spawn`], but pass `client_capabilities` to the `initialize` handler. Tests that
    /// exercise the `fs.*` or `terminal` delegation paths flip the relevant capability bits here.
    fn spawn_with_capabilities(
        config_toml: &str,
        script: Option<serde_json::Value>,
        client_capabilities: serde_json::Value,
    ) -> Self {
        Self::builder()
            .config(config_toml)
            .script_opt(script)
            .capabilities(client_capabilities)
            .build()
    }

    fn builder() -> AcpTestHarnessBuilder {
        AcpTestHarnessBuilder::default()
    }

    fn config_dir(&self) -> &Path {
        &self.config_dir
    }

    /// Send a JSON-RPC request and return the parsed response. Uses
    /// a monotonically increasing request id; tests don't need to
    /// pick ids themselves. Convenience wrapper over [`Self::send_request`]
    /// + [`Self::await_response`].
    fn request(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        let id = self.send_request(method, params);
        self.await_response(id)
    }

    /// Fire a JSON-RPC request and return the id without waiting. Use [`Self::await_response`] or
    /// [`Self::collect_updates`] to pick up the response later. Useful when a test fires a prompt,
    /// observes intermediate notifications, then collects the final response.
    fn send_request(&mut self, method: &str, params: serde_json::Value) -> u64 {
        self.next_id += 1;
        let id = self.next_id;
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        writeln!(self.stdin, "{}", request).expect("write request");
        id
    }

    /// Fire a JSON-RPC notification (no id, no response). Used for `session/cancel`.
    fn notify(&mut self, method: &str, params: serde_json::Value) {
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        writeln!(self.stdin, "{}", notification).expect("write notification");
    }

    /// Block until the response for `id` arrives. Side-channel notifications + meka-issued requests
    /// on the same connection are silently dropped (the latter is the right call only when the test
    /// isn't expected to provoke any).
    fn await_response(&mut self, id: u64) -> serde_json::Value {
        self.await_response_with_dispatch(id, |_| None)
    }

    /// Block until the response for `id` arrives, dispatching any meka-issued requests through
    /// `handler`. The handler returns `Some(response)` to answer, or `None` to ignore.
    fn await_response_with_dispatch<F>(&mut self, id: u64, mut handler: F) -> serde_json::Value
    where
        F: FnMut(&serde_json::Value) -> Option<serde_json::Value>,
    {
        let needle = format!("\"id\":{}", id);
        let lines = read_until_with_dispatch(
            &mut self.reader,
            &mut self.stdin,
            self.deadline,
            |value| handler(value),
            |line| response_matches(line, &needle),
        );
        let line = match lines.iter().find(|line| response_matches(line, &needle)) {
            Some(line) => line.clone(),
            None => {
                let collected = lines.join("");
                panic!("no response for id={}; transcript:\n{}", id, collected,);
            }
        };
        serde_json::from_str(&line).unwrap_or_else(|error| {
            panic!("response for id={} was not JSON ({}): {}", id, error, line);
        })
    }

    /// Drain every `session/update` for `sid` that meka has already emitted, with no prompt
    /// outstanding.
    ///
    /// Every other update helper is keyed to a pending `session/prompt`; a scheduled turn has none,
    /// which is the property under test. Termination comes from a `session/list` sent afterwards:
    /// stdio preserves order, so its response cannot arrive before the notifications queued ahead
    /// of it, and waiting on a reply meka is guaranteed to send means a regression fails the
    /// test instead of blocking the suite on a `read_line` that never returns.
    fn drain_unsolicited_updates(&mut self, sid: &str) -> Vec<serde_json::Value> {
        let id = self.send_request("session/list", serde_json::json!({}));
        let needle = format!("\"id\":{}", id);
        let sid_owned = sid.to_string();
        let lines = read_until_with_dispatch(
            &mut self.reader,
            &mut self.stdin,
            self.deadline,
            |_| None,
            |line| response_matches(line, &needle),
        );
        lines
            .iter()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|value| value["method"] == "session/update")
            .filter(|value| value["params"]["sessionId"] == sid_owned)
            .collect()
    }

    /// Collect every `session/update` notification for `sid` plus the eventual response for `id`.
    /// Captures everything the agent emits during a prompt turn, which is the single most reused
    /// pattern in the existing test file. Side-channel meka-issued requests are silently ignored;
    /// if a test expects them, use [`Self::collect_updates_with_dispatch`] instead.
    fn collect_updates(
        &mut self,
        sid: &str,
        id: u64,
    ) -> (Vec<serde_json::Value>, serde_json::Value) {
        self.collect_updates_with_dispatch(sid, id, |_| None)
    }

    /// As [`Self::collect_updates`], but dispatch meka-issued requests via `handler`. Used by tests
    /// that watch the session/update stream *and* answer fs / terminal delegation.
    ///
    /// The free [`read_until_with_dispatch`] only invokes its dispatch closure on JSON-RPC
    /// *requests* (those with both `method` and `id`). Notifications carry `method` but no `id`, so
    /// we can't piggy-back on it; drive a parallel loop here that also captures `session/update`
    /// notifications for the target `sid`.
    fn collect_updates_with_dispatch<F>(
        &mut self,
        sid: &str,
        id: u64,
        mut handler: F,
    ) -> (Vec<serde_json::Value>, serde_json::Value)
    where
        F: FnMut(&serde_json::Value) -> Option<serde_json::Value>,
    {
        let needle = format!("\"id\":{}", id);
        let mut updates: Vec<serde_json::Value> = Vec::new();
        let mut response: Option<serde_json::Value> = None;
        let mut transcript = String::new();
        while Instant::now() < self.deadline {
            let mut line = String::new();
            match self.reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {}
                Err(_) => break,
            }
            transcript.push_str(&line);
            let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
                continue;
            };
            if value["method"] == "session/update"
                && value["params"]["sessionId"].as_str() == Some(sid)
            {
                updates.push(value.clone());
                continue;
            }
            if value.get("method").is_some()
                && value.get("id").is_some()
                && let Some(reply) = handler(&value)
            {
                let _ = writeln!(self.stdin, "{}", reply);
                continue;
            }
            if line.contains(&needle) && response_matches(&line, &needle) {
                response = Some(value);
                break;
            }
        }
        let response = response
            .unwrap_or_else(|| panic!("no response for id={}; transcript:\n{}", id, transcript,));
        (updates, response)
    }

    /// Create a session in `config_dir` and return its id. Most tests do this once at start of a
    /// scenario.
    fn new_session(&mut self) -> String {
        let cwd = self.config_dir.clone();
        let response = self.request(
            "session/new",
            serde_json::json!({ "cwd": cwd, "mcpServers": [] }),
        );
        response["result"]["sessionId"]
            .as_str()
            .unwrap_or_else(|| panic!("session/new did not return a sessionId: {}", response))
            .to_string()
    }

    /// Fire a `session/prompt` against `sid` and return the request id. Pair with
    /// [`Self::collect_updates`] or [`Self::await_response`] to read the result.
    fn prompt(&mut self, sid: &str, text: &str) -> u64 {
        self.send_request(
            "session/prompt",
            serde_json::json!({
                "sessionId": sid,
                "prompt": [{ "type": "text", "text": text }],
            }),
        )
    }

    /// Fire a `session/cancel` notification for `sid`.
    fn cancel(&mut self, sid: &str) {
        self.notify("session/cancel", serde_json::json!({ "sessionId": sid }));
    }

    /// One-shot `session/set_mode` round-trip.
    fn set_mode(&mut self, sid: &str, mode: &str) -> serde_json::Value {
        self.request(
            "session/set_mode",
            serde_json::json!({
                "sessionId": sid,
                "modeId": mode,
            }),
        )
    }

    /// One-shot `session/close` round-trip.
    fn close_session(&mut self, sid: &str) -> serde_json::Value {
        self.request("session/close", serde_json::json!({ "sessionId": sid }))
    }
}

/// Returns `true` when `line` is the response (or error) message for the given `id` needle. Used by
/// [`AcpTestHarness`] helpers that stop reading once the awaited response arrives. The secondary
/// check on `result` / `error` filters out incoming meka-issued *requests* that happen to share an
/// id with our response we're awaiting (the dispatch loop renumbers those, but
/// belt-and-suspenders).
fn response_matches(line: &str, needle: &str) -> bool {
    if !line.contains(needle) {
        return false;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
        return false;
    };
    value.get("result").is_some() || value.get("error").is_some()
}

impl Drop for AcpTestHarness {
    fn drop(&mut self) {
        // Best-effort cleanup; we already closed stdin or killed the child in `drain_stderr` for
        // failing tests, but a happy-path test that returns normally still needs the child reaped.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Read lines until either `f` returns `true`, EOF, or the deadline elapses. Collects every line
/// read so test failures can dump the JSON-RPC stream for diagnosis.
fn read_until<R, F>(reader: &mut R, deadline: Instant, mut f: F) -> Vec<String>
where
    R: BufRead,
    F: FnMut(&str) -> bool,
{
    let mut lines = Vec::new();
    while Instant::now() < deadline {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                let stop = f(&line);
                lines.push(line);
                if stop {
                    return lines;
                }
            }
            Err(_) => break,
        }
    }
    lines
}

/// Variant of [`read_until`] that also answers incoming JSON-RPC *requests* from meka. Any line
/// that parses to a JSON object with both a `method` and an `id` field is treated as a meka-issued
/// request; `dispatch` is invoked with the parsed value and its `Some(response)` return value is
/// written back to meka's stdin. Tests use this to play the client side of the `fs/*` and
/// `terminal/*` round-trips.
fn read_until_with_dispatch<R, W, D, F>(
    reader: &mut R,
    stdin: &mut W,
    deadline: Instant,
    mut dispatch: D,
    mut stop: F,
) -> Vec<String>
where
    R: BufRead,
    W: Write,
    D: FnMut(&serde_json::Value) -> Option<serde_json::Value>,
    F: FnMut(&str) -> bool,
{
    let mut lines = Vec::new();
    while Instant::now() < deadline {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line)
                    && value.get("method").is_some()
                    && value.get("id").is_some()
                    && let Some(response) = dispatch(&value)
                {
                    // Failure to write to stdin means the child is gone; surface it via the
                    // deadline loop rather than panicking from inside the helper.
                    let _ = writeln!(stdin, "{}", response);
                }
                let should_stop = stop(&line);
                lines.push(line);
                if should_stop {
                    return lines;
                }
            }
            Err(_) => break,
        }
    }
    lines
}

#[test]
fn acp_tool_call_lifecycle_round_trips_through_mock_provider() {
    // Fake config + credential so `create_agent_from_config` builds a real provider stack. The
    // mock swap inside `run_acp` then replaces the provider before any HTTP call is attempted.
    let config_toml = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(|config_dir| {
            // Target file for the scripted `read_file` call. Real tool runs against this path, so
            // it must exist.
            let target = config_dir.join("target.txt");
            std::fs::write(&target, "hello from mock test\n").expect("write target");
            serde_json::json!([
                [
                    { "kind": "text", "text": "reading the file...\n" },
                    { "kind": "tool_use_start", "id": "call_1", "name": "read_file" },
                    { "kind": "tool_use_end", "input": { "path": target.to_str().unwrap() } },
                    { "kind": "message_end", "stop_reason": "tool_use" }
                ],
                [
                    { "kind": "text", "text": "done!" },
                    { "kind": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .build();
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "read the target file");
    let (updates, response) = harness.collect_updates(&sid, id);

    let saw_tool_call = updates.iter().any(|value| {
        let update = &value["params"]["update"];
        if update["sessionUpdate"] != "tool_call" {
            return false;
        }
        assert_eq!(
            update["kind"], "read",
            "expected tool kind 'read': {}",
            update
        );
        assert_eq!(
            update["status"], "in_progress",
            "expected tool_call status in_progress: {}",
            update,
        );
        // The title carries the resolved primary argument (the path), not the bare tool name.
        let title = update["title"].as_str().unwrap_or("");
        assert!(
            title.starts_with("Read ") && title.contains("target.txt"),
            "tool_call title should be 'Read <path>': {}",
            update,
        );
        true
    });
    assert!(
        saw_tool_call,
        "expected a session/update with sessionUpdate=tool_call; updates: {:?}",
        updates,
    );
    assert!(
        updates.iter().any(|value| {
            let update = &value["params"]["update"];
            update["sessionUpdate"] == "tool_call_update" && update["status"] == "completed"
        }),
        "expected a tool_call_update with status=completed; updates: {:?}",
        updates,
    );
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "expected stopReason=end_turn; full response: {}",
        response,
    );
}

/// The `todo` tool surfaces as a `plan` session/update with one entry per item.
#[test]
fn acp_todo_tool_emits_plan_update() {
    let config_toml = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(|_dir| {
            serde_json::json!([
                [
                    { "kind": "text", "text": "planning...\n" },
                    { "kind": "tool_use_start", "id": "call_todo", "name": "todo" },
                    {
                        "kind": "tool_use_end",
                        "input": { "title": "Work", "items": ["First", "Second"] }
                    },
                    { "kind": "message_end", "stop_reason": "tool_use" }
                ],
                [
                    { "kind": "text", "text": "done" },
                    { "kind": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .build();
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "make a plan");
    let (updates, response) = harness.collect_updates(&sid, id);

    let plan = updates
        .iter()
        .find(|value| value["params"]["update"]["sessionUpdate"] == "plan")
        .unwrap_or_else(|| panic!("expected a plan session/update; updates: {:?}", updates));
    let entries = plan["params"]["update"]["entries"]
        .as_array()
        .expect("plan entries array");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0]["content"], "First");
    assert_eq!(entries[0]["status"], "pending");
    assert_eq!(response["result"]["stopReason"], "end_turn");
}

/// The first turn of a fresh session emits a `session_info_update` carrying the title (the first
/// user message preview, with the agent's `<context>` preamble stripped).
#[test]
fn acp_first_turn_emits_session_info_update_title() {
    let script = serde_json::json!([[
        { "kind": "text", "text": "ok" },
        { "kind": "message_end", "stop_reason": "end_turn" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "explain the build system");
    let (updates, _response) = harness.collect_updates(&sid, id);

    let info = updates
        .iter()
        .find(|value| value["params"]["update"]["sessionUpdate"] == "session_info_update")
        .unwrap_or_else(|| panic!("expected a session_info_update; updates: {:?}", updates));
    assert_eq!(
        info["params"]["update"]["title"],
        "explain the build system"
    );
}

/// `session/prompt` reports session-cumulative token usage on the response, alongside the
/// per-turn `usage_update` notification that carries the context gauge.
#[test]
fn acp_prompt_response_carries_token_usage() {
    let script = serde_json::json!([[
        { "kind": "text", "text": "ok" },
        { "kind": "message_end", "stop_reason": "end_turn" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "hello");
    let (_updates, response) = harness.collect_updates(&sid, id);

    let usage = &response["result"]["usage"];
    assert!(
        usage.is_object(),
        "expected usage on the prompt response; full response: {}",
        response
    );
    // Every counter is reported, including the cache tiers, rather than left off the wire. The
    // values are all zero because the mock provider emits no token-usage events by design (see
    // `provider::mock`), so this pins the shape and the wiring, not the arithmetic.
    for field in [
        "totalTokens",
        "inputTokens",
        "outputTokens",
        "cachedReadTokens",
        "cachedWriteTokens",
    ] {
        assert!(
            usage[field].is_u64(),
            "expected {} on the usage object; usage: {}",
            field,
            usage
        );
    }
    // Omitted deliberately: meka doesn't meter reasoning separately from output.
    assert!(usage["thoughtTokens"].is_null(), "usage: {}", usage);
}

/// A turn that reports `tool_use` but carries no tool-call block leaves the agent with nothing to
/// run and nothing to show. It happens for real: an OpenAI-compatible endpoint that coalesces its
/// final delta into the `finish_reason` chunk used to have that delta dropped, so the tool call
/// vanished and only the stop reason survived. The client must still get a visible message and a
/// terminal stop reason rather than a silent turn it waits on forever.
#[test]
fn acp_turn_with_tool_use_stop_but_no_tool_call_still_reports_to_the_client() {
    let script = serde_json::json!([[
        { "kind": "message_end", "stop_reason": "tool_use" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "search the web");
    let (updates, response) = harness.collect_updates(&sid, id);

    let chunk = updates
        .iter()
        .find(|value| value["params"]["update"]["sessionUpdate"] == "agent_message_chunk")
        .unwrap_or_else(|| {
            panic!("expected an agent_message_chunk standing in for the empty turn; updates: {updates:?}")
        });
    let text = chunk["params"]["update"]["content"]["text"]
        .as_str()
        .unwrap_or_default();
    assert!(
        text.contains("empty response"),
        "the stand-in notice must reach the client verbatim; got {text:?}"
    );

    // No tool call was made, so the turn ends normally. A client that saw `tool_use` with no
    // `tool_call` update and no terminal stop reason would sit waiting on a call that never came.
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "full response: {response}"
    );
    assert!(
        !updates
            .iter()
            .any(|value| value["params"]["update"]["sessionUpdate"] == "tool_call"),
        "no tool call was made, so none must be announced; updates: {updates:?}"
    );
}

/// Outcome the test wants the synthetic ACP client to send back when the agent issues
/// `session/request_permission`.
#[derive(Debug, Clone, Copy)]
enum PermissionAnswer {
    AllowOnce,
    RejectOnce,
}

/// An ACP session at `workspace` writes inside its cwd and is refused outside it.
///
/// The ACP surface had no `workspace` session at all: its three mentions of the word are about ACP
/// *workspace folders*, which are unrelated. So the fence could stop being applied to a session
/// `session/new` created and nothing on this surface would notice.
///
/// Asserts disk state rather than the tool-call status, because a tool that never ran and one the
/// fence refused report the same way.
#[test]
fn an_acp_session_at_workspace_is_fenced_to_its_cwd() {
    let config_toml = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"

[permissions]
default = "workspace"
enabled = ["read", "workspace", "unrestricted"]
"#;
    let outside_dir = tempfile::tempdir().expect("tempdir");
    let outside = outside_dir.path().join("escaped.txt");
    let outside_for_script = outside.clone();

    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(move |config_dir| {
            // The session's cwd is the config dir, so this one is inside the boundary.
            let inside = config_dir.join("inside.txt");
            serde_json::json!([
                [
                    { "kind": "tool_use_start", "id": "call_in", "name": "write_file" },
                    {
                        "kind": "tool_use_end",
                        "input": { "path": inside.to_str().unwrap(), "content": "in" }
                    },
                    { "kind": "message_end", "stop_reason": "tool_use" }
                ],
                [
                    { "kind": "tool_use_start", "id": "call_out", "name": "write_file" },
                    {
                        "kind": "tool_use_end",
                        "input": {
                            "path": outside_for_script.to_str().unwrap(),
                            "content": "out"
                        }
                    },
                    { "kind": "message_end", "stop_reason": "tool_use" }
                ],
                [
                    { "kind": "text", "text": "done" },
                    { "kind": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .build();

    let inside = harness.config_dir().join("inside.txt");
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "write both");
    let _ = harness.await_response(id);

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

/// Drive a full `meka acp` permission round-trip with the mock provider. The scripted turn calls
/// `write_file` (which under `ask` triggers a
/// `session/request_permission`); the test auto-responds with the configured outcome and asserts
/// the resulting tool-call status.
fn run_permission_scenario(answer: PermissionAnswer) {
    // [permissions].default = "ask" puts the agent in `ask`, where a write triggers the round-trip
    // we want to exercise.
    let config_toml = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"

[permissions]
default = "ask"
enabled = ["read", "ask", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(|config_dir| {
            let target = config_dir.join("out.txt");
            serde_json::json!([
                [
                    { "kind": "text", "text": "writing the file...\n" },
                    { "kind": "tool_use_start", "id": "call_write", "name": "write_file" },
                    {
                        "kind": "tool_use_end",
                        "input": { "path": target.to_str().unwrap(), "content": "hello" }
                    },
                    { "kind": "message_end", "stop_reason": "tool_use" }
                ],
                [
                    { "kind": "text", "text": "done!" },
                    { "kind": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .build();
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "write the file");

    let option_id = match answer {
        PermissionAnswer::AllowOnce => "allow_once",
        PermissionAnswer::RejectOnce => "reject_once",
    };
    let mut saw_permission_request = false;
    let (updates, response) =
        harness.collect_updates_with_dispatch(&sid, id, |value| match value["method"].as_str() {
            Some("session/request_permission") => {
                saw_permission_request = true;
                Some(serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": value["id"].clone(),
                    "result": {
                        "outcome": { "outcome": "selected", "optionId": option_id }
                    }
                }))
            }
            _ => None,
        });

    assert!(
        saw_permission_request,
        "expected a session/request_permission from the agent; updates: {:?}",
        updates,
    );

    let status = updates
        .iter()
        .filter_map(|value| {
            let update = &value["params"]["update"];
            if update["sessionUpdate"] != "tool_call_update" {
                return None;
            }
            update["status"].as_str().map(str::to_string)
        })
        .next_back()
        .unwrap_or_else(|| {
            panic!(
                "expected a tool_call_update with a status; updates: {:?}",
                updates,
            )
        });
    match answer {
        PermissionAnswer::AllowOnce => assert_eq!(
            status, "completed",
            "allow_once should let write_file complete; updates: {:?}",
            updates,
        ),
        PermissionAnswer::RejectOnce => assert_eq!(
            status, "failed",
            "reject_once should fail the tool call; updates: {:?}",
            updates,
        ),
    }
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "expected stopReason=end_turn after permission outcome was handled; full response: {}",
        response,
    );
}

#[test]
fn acp_permission_allow_once_runs_tool_and_completes_turn() {
    // An `ask`-mode session where the client answers `allow_once` must actually run the gated tool.
    run_permission_scenario(PermissionAnswer::AllowOnce);
}

#[test]
fn acp_permission_reject_once_fails_tool_but_completes_turn() {
    run_permission_scenario(PermissionAnswer::RejectOnce);
}

/// a session that ran one prompt + tool round-trip is closed and then loaded by id; the load
/// handler must replay `user_message_chunk`, `agent_message_chunk`, `tool_call` (read kind), and a
/// `tool_call_update` with `status=completed` before responding with `LoadSessionResponse`.
#[test]
fn acp_session_load_replays_persisted_turn() {
    let temp = tempfile::tempdir().expect("tempdir");
    let config_dir = temp.path().join("meka");
    let data_dir = temp.path().join("data").join("meka");
    std::fs::create_dir_all(&config_dir).expect("create config dir");

    let config_toml = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"
"#;
    std::fs::write(config_dir.join("config.toml"), config_toml).expect("write config.toml");

    let target = config_dir.join("target.txt");
    std::fs::write(&target, "hello from reload test\n").expect("write target");

    let script = serde_json::json!([
        [
            { "kind": "text", "text": "reading the file...\n" },
            { "kind": "tool_use_start", "id": "call_1", "name": "read_file" },
            { "kind": "tool_use_end", "input": { "path": target.to_str().unwrap() } },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "done!" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let script_path = temp.path().join("script.json");
    std::fs::write(&script_path, script.to_string()).expect("write script.json");

    // First run: drive one prompt to populate the session, capture sessionId, then exit cleanly.
    let session_id = {
        let mut child = meka_acp()
            .arg("acp")
            .env("MEKA_CONFIG_DIR", &config_dir)
            .env("MEKA_DATA_DIR", &data_dir)
            .env("HOME", temp.path())
            .env("MEKA_MOCK_PROVIDER", "1")
            .env("MEKA_MOCK_PROVIDER_SCRIPT", &script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn meka acp");
        let mut stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let stderr_pipe = child.stderr.take().expect("stderr");
        let mut reader = BufReader::new(stdout);
        let stderr_handle = std::thread::spawn(move || {
            let mut buf = String::new();
            let mut r = BufReader::new(stderr_pipe);
            while r.read_line(&mut buf).unwrap_or(0) > 0 {}
            buf
        });
        let deadline = Instant::now() + Duration::from_secs(15);

        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
        )
        .expect("initialize");
        let _ = read_until(&mut reader, deadline, |line| line.contains("\"id\":1"));

        let new_req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": { "cwd": config_dir.clone(), "mcpServers": [] }
        });
        writeln!(stdin, "{}", new_req).expect("session/new");
        let new_lines = read_until(&mut reader, deadline, |line| line.contains("\"id\":2"));
        let new_line = new_lines
            .iter()
            .find(|line| line.contains("\"id\":2"))
            .expect("session/new response");
        let new_response: serde_json::Value =
            serde_json::from_str(new_line).expect("session/new JSON parses");
        let sid = new_response["result"]["sessionId"]
            .as_str()
            .expect("sessionId is a string")
            .to_string();

        let prompt_req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/prompt",
            "params": {
                "sessionId": sid,
                "prompt": [{ "type": "text", "text": "read the target file" }]
            }
        });
        writeln!(stdin, "{}", prompt_req).expect("write session/prompt");
        let _ = read_until(&mut reader, deadline, |line| line.contains("\"id\":3"));

        drop(stdin);
        let _ = child.kill();
        let _ = child.wait();
        let _ = stderr_handle.join();
        sid
    };

    // Second run: load the persisted session and assert the replay stream.
    let mut child = meka_acp()
        .arg("acp")
        .env("MEKA_CONFIG_DIR", &config_dir)
        .env("MEKA_DATA_DIR", &data_dir)
        .env("HOME", temp.path())
        .env("MEKA_MOCK_PROVIDER", "1")
        .env("MEKA_MOCK_PROVIDER_SCRIPT", &script_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn meka acp #2");

    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let stderr_pipe = child.stderr.take().expect("stderr");
    let mut reader = BufReader::new(stdout);
    let stderr_handle = std::thread::spawn(move || {
        let mut buf = String::new();
        let mut r = BufReader::new(stderr_pipe);
        while r.read_line(&mut buf).unwrap_or(0) > 0 {}
        buf
    });

    let deadline = Instant::now() + Duration::from_secs(15);

    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
    )
    .expect("initialize");
    let init_lines = read_until(&mut reader, deadline, |line| line.contains("\"id\":1"));
    // Confirm session-management capabilities were advertised.
    let init_response: serde_json::Value = serde_json::from_str(
        init_lines
            .iter()
            .find(|line| line.contains("\"id\":1"))
            .expect("init response"),
    )
    .expect("init parses");
    assert_eq!(
        init_response["result"]["agentCapabilities"]["loadSession"],
        true,
    );
    assert!(
        init_response["result"]["agentCapabilities"]["sessionCapabilities"]["list"].is_object(),
        "expected sessionCapabilities.list to be advertised; got: {}",
        init_response,
    );

    let load_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "session/load",
        "params": {
            "sessionId": session_id,
            "cwd": config_dir.clone(),
            "mcpServers": []
        }
    });
    writeln!(stdin, "{}", load_req).expect("session/load");
    let load_lines = read_until(&mut reader, deadline, |line| line.contains("\"id\":4"));

    let mut saw_user_chunk = false;
    let mut saw_agent_chunk = false;
    let mut saw_tool_call = false;
    let mut saw_tool_call_update_completed = false;
    let mut load_response: Option<serde_json::Value> = None;
    for line in &load_lines {
        let value: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(error) => panic!("stdout line is not valid JSON-RPC: {} ({})", line, error,),
        };
        if value["method"] == "session/update" {
            let update = &value["params"]["update"];
            match update["sessionUpdate"].as_str() {
                Some("user_message_chunk") => saw_user_chunk = true,
                Some("agent_message_chunk") => saw_agent_chunk = true,
                Some("tool_call") => {
                    assert_eq!(update["kind"], "read");
                    saw_tool_call = true;
                }
                Some("tool_call_update") if update["status"] == "completed" => {
                    saw_tool_call_update_completed = true;
                }
                _ => {}
            }
        }
        if value["id"] == 4 {
            load_response = Some(value);
        }
    }

    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();

    let dump = || load_lines.join("");

    assert!(
        saw_user_chunk,
        "replay must emit user_message_chunk; stream:\n{}\nSTDERR:\n{}",
        dump(),
        stderr_handle.join().unwrap_or_default(),
    );
    assert!(
        saw_agent_chunk,
        "replay must emit agent_message_chunk; stream:\n{}",
        dump()
    );
    assert!(
        saw_tool_call,
        "replay must emit tool_call; stream:\n{}",
        dump()
    );
    assert!(
        saw_tool_call_update_completed,
        "replay must emit tool_call_update completed; stream:\n{}",
        dump(),
    );

    let response =
        load_response.unwrap_or_else(|| panic!("no LoadSessionResponse; stream:\n{}", dump()));
    assert!(
        response["result"].is_object(),
        "expected an object result for session/load: {}",
        response,
    );
}

/// Poll `child.try_wait()` until the process exits or `timeout` elapses. Returns the exit status,
/// or `None` on timeout (the caller decides whether that's a failure and is responsible for killing
/// the child).
fn wait_for_exit(child: &mut Child, timeout: Duration) -> Option<std::process::ExitStatus> {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status),
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(50));
            }
            _ => return None,
        }
    }
}

/// Regression: `meka acp` must exit (releasing its session `flock`) when the client disconnects
/// (stdin EOF), instead of lingering as an orphan that pins the lock. Run 1 takes the lock via
/// `session/new` + a prompt, then drops stdin WITHOUT `session/close` or `kill`; the process must
/// exit on its own. Run 2 then loads the same session from a fresh process and must succeed (not
/// `SessionLocked`), proving run 1 released the lock by exiting.
#[test]
fn acp_exits_and_releases_lock_on_stdin_eof() {
    let temp = tempfile::tempdir().expect("tempdir");
    let config_dir = temp.path().join("meka");
    let data_dir = temp.path().join("data").join("meka");
    std::fs::create_dir_all(&config_dir).expect("create config dir");

    let config_toml = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"
"#;
    std::fs::write(config_dir.join("config.toml"), config_toml).expect("write config.toml");

    let script = serde_json::json!([
        [
            { "kind": "text", "text": "done!" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let script_path = temp.path().join("script.json");
    std::fs::write(&script_path, script.to_string()).expect("write script.json");

    // Run 1: take the session lock, then disconnect by dropping stdin (no session/close, no kill).
    let session_id = {
        let mut child = meka_acp()
            .arg("acp")
            .env("MEKA_CONFIG_DIR", &config_dir)
            .env("MEKA_DATA_DIR", &data_dir)
            .env("HOME", temp.path())
            .env("MEKA_MOCK_PROVIDER", "1")
            .env("MEKA_MOCK_PROVIDER_SCRIPT", &script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn meka acp");
        let mut stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let stderr_pipe = child.stderr.take().expect("stderr");
        let stderr_handle = std::thread::spawn(move || {
            let mut buf = String::new();
            let mut r = BufReader::new(stderr_pipe);
            while r.read_line(&mut buf).unwrap_or(0) > 0 {}
            buf
        });
        let mut reader = BufReader::new(stdout);
        let deadline = Instant::now() + Duration::from_secs(15);

        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
        )
        .expect("initialize");
        let _ = read_until(&mut reader, deadline, |line| line.contains("\"id\":1"));

        let new_req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": { "cwd": config_dir.clone(), "mcpServers": [] }
        });
        writeln!(stdin, "{}", new_req).expect("session/new");
        let new_lines = read_until(&mut reader, deadline, |line| line.contains("\"id\":2"));
        let sid = serde_json::from_str::<serde_json::Value>(
            new_lines
                .iter()
                .find(|line| line.contains("\"id\":2"))
                .expect("session/new response"),
        )
        .expect("parse")["result"]["sessionId"]
            .as_str()
            .expect("sessionId")
            .to_string();

        // One prompt so the session lock is definitely held.
        let prompt_req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/prompt",
            "params": { "sessionId": sid, "prompt": [{ "type": "text", "text": "hello" }] }
        });
        writeln!(stdin, "{}", prompt_req).expect("session/prompt");
        let _ = read_until(&mut reader, deadline, |line| line.contains("\"id\":3"));

        // Drain remaining stdout on a thread so a shutdown-time write can't block the child, then
        // disconnect by closing stdin. Crucially: NO `child.kill()` -- the process must exit
        // itself.
        let stdout_handle =
            std::thread::spawn(move || std::io::copy(&mut reader, &mut std::io::sink()));
        drop(stdin);

        let exited = wait_for_exit(&mut child, Duration::from_secs(10)).is_some();
        if !exited {
            let _ = child.kill();
            let _ = child.wait();
        }
        let _ = stdout_handle.join();
        assert!(
            exited,
            "meka acp did not exit within 10s of stdin EOF (orphaned, lock still held).\nSTDERR:\n{}",
            stderr_handle.join().unwrap_or_default(),
        );
        sid
    };

    // Run 2: a fresh process must be able to lock + load the same session.
    let mut child = meka_acp()
        .arg("acp")
        .env("MEKA_CONFIG_DIR", &config_dir)
        .env("MEKA_DATA_DIR", &data_dir)
        .env("HOME", temp.path())
        .env("MEKA_MOCK_PROVIDER", "1")
        .env("MEKA_MOCK_PROVIDER_SCRIPT", &script_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn meka acp #2");
    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let stderr_pipe = child.stderr.take().expect("stderr");
    let stderr_handle = std::thread::spawn(move || {
        let mut buf = String::new();
        let mut r = BufReader::new(stderr_pipe);
        while r.read_line(&mut buf).unwrap_or(0) > 0 {}
        buf
    });
    let mut reader = BufReader::new(stdout);
    let deadline = Instant::now() + Duration::from_secs(15);

    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
    )
    .expect("initialize");
    let _ = read_until(&mut reader, deadline, |line| line.contains("\"id\":1"));

    let load_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "session/load",
        "params": { "sessionId": session_id, "cwd": config_dir.clone(), "mcpServers": [] }
    });
    writeln!(stdin, "{}", load_req).expect("session/load");
    let load_lines = read_until(&mut reader, deadline, |line| line.contains("\"id\":4"));

    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();

    let response_line = load_lines
        .iter()
        .find(|line| line.contains("\"id\":4"))
        .unwrap_or_else(|| {
            panic!(
                "no session/load response; run 1 may not have released the lock.\nSTDERR:\n{}",
                stderr_handle.join().unwrap_or_default(),
            )
        });
    let response: serde_json::Value =
        serde_json::from_str(response_line).expect("parse session/load response");
    assert!(
        response.get("error").is_none(),
        "session/load must succeed after run 1 exited; got error (lock not released?): {}",
        response,
    );
}

/// a `session/list` with a `cwd` filter must only return sessions whose persisted cwd matches.
#[test]
fn acp_session_list_filters_by_cwd() {
    let temp = tempfile::tempdir().expect("tempdir");
    let config_dir = temp.path().join("meka");
    let data_dir = temp.path().join("data").join("meka");
    std::fs::create_dir_all(&config_dir).expect("create config dir");

    let config_toml = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"
"#;
    std::fs::write(config_dir.join("config.toml"), config_toml).expect("write config.toml");

    // Two distinct cwds that both physically exist (ACP server only stores the path; existence
    // doesn't matter for the filter, but tools later resolved against it would fail if absent).
    let cwd_a = temp.path().join("proj-a");
    let cwd_b = temp.path().join("proj-b");
    std::fs::create_dir_all(&cwd_a).expect("mkdir cwd_a");
    std::fs::create_dir_all(&cwd_b).expect("mkdir cwd_b");

    let script = serde_json::json!([
        [
            { "kind": "text", "text": "ack" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let script_path = temp.path().join("script.json");
    std::fs::write(&script_path, script.to_string()).expect("write script.json");

    // Helper: launch one `meka acp`, send initialize + session/new (with the given cwd) +
    // session/prompt, return the sessionId.
    let create_one = |session_cwd: &std::path::Path| -> String {
        let mut child = meka_acp()
            .arg("acp")
            .env("MEKA_CONFIG_DIR", &config_dir)
            .env("MEKA_DATA_DIR", &data_dir)
            .env("HOME", temp.path())
            .env("MEKA_MOCK_PROVIDER", "1")
            .env("MEKA_MOCK_PROVIDER_SCRIPT", &script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn");
        let mut stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let stderr_pipe = child.stderr.take().expect("stderr");
        let mut reader = BufReader::new(stdout);
        let _stderr_handle = std::thread::spawn(move || {
            let mut buf = String::new();
            let mut r = BufReader::new(stderr_pipe);
            while r.read_line(&mut buf).unwrap_or(0) > 0 {}
            buf
        });
        let deadline = Instant::now() + Duration::from_secs(15);
        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
        )
        .expect("init");
        let _ = read_until(&mut reader, deadline, |line| line.contains("\"id\":1"));

        let new_req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": { "cwd": session_cwd, "mcpServers": [] }
        });
        writeln!(stdin, "{}", new_req).expect("session/new");
        let new_lines = read_until(&mut reader, deadline, |line| line.contains("\"id\":2"));
        let new_line = new_lines
            .iter()
            .find(|line| line.contains("\"id\":2"))
            .expect("session/new response");
        let new_response: serde_json::Value =
            serde_json::from_str(new_line).expect("parse session/new");
        let sid = new_response["result"]["sessionId"]
            .as_str()
            .expect("sessionId")
            .to_string();

        // Drive a no-op prompt so the session is persisted with a message (otherwise the title
        // would be empty; not required by the assertion but matches the realistic shape).
        let prompt_req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/prompt",
            "params": {
                "sessionId": sid,
                "prompt": [{ "type": "text", "text": "ping" }]
            }
        });
        writeln!(stdin, "{}", prompt_req).expect("prompt");
        let _ = read_until(&mut reader, deadline, |line| line.contains("\"id\":3"));

        drop(stdin);
        let _ = child.kill();
        let _ = child.wait();
        sid
    };

    let id_a = create_one(&cwd_a);
    let _id_b = create_one(&cwd_b);

    // Second invocation issues session/list filtered to cwd_a.
    let mut child = meka_acp()
        .arg("acp")
        .env("MEKA_CONFIG_DIR", &config_dir)
        .env("MEKA_DATA_DIR", &data_dir)
        .env("HOME", temp.path())
        .env("MEKA_MOCK_PROVIDER", "1")
        .env("MEKA_MOCK_PROVIDER_SCRIPT", &script_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn list child");
    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let stderr_pipe = child.stderr.take().expect("stderr");
    let mut reader = BufReader::new(stdout);
    let _stderr_handle = std::thread::spawn(move || {
        let mut buf = String::new();
        let mut r = BufReader::new(stderr_pipe);
        while r.read_line(&mut buf).unwrap_or(0) > 0 {}
        buf
    });
    let deadline = Instant::now() + Duration::from_secs(15);

    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
    )
    .expect("init");
    let _ = read_until(&mut reader, deadline, |line| line.contains("\"id\":1"));

    let list_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "session/list",
        "params": { "cwd": cwd_a.clone() }
    });
    writeln!(stdin, "{}", list_req).expect("session/list");
    let list_lines = read_until(&mut reader, deadline, |line| line.contains("\"id\":5"));

    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();

    let list_line = list_lines
        .iter()
        .find(|line| line.contains("\"id\":5"))
        .expect("session/list response");
    let list_response: serde_json::Value =
        serde_json::from_str(list_line).expect("parse session/list");
    let sessions = list_response["result"]["sessions"]
        .as_array()
        .expect("sessions array");
    assert_eq!(
        sessions.len(),
        1,
        "expected exactly one session matching cwd_a; got: {}",
        list_response,
    );
    assert_eq!(sessions[0]["sessionId"], id_a);
}

/// `session/resume` adopts an existing session id without replaying. The handler should not emit
/// any `session/update` notifications, but should leave the slot populated so subsequent prompts
/// can proceed (smoke test: a follow-up prompt succeeds).
#[test]
fn acp_session_resume_adopts_without_replay() {
    let temp = tempfile::tempdir().expect("tempdir");
    let config_dir = temp.path().join("meka");
    let data_dir = temp.path().join("data").join("meka");
    std::fs::create_dir_all(&config_dir).expect("create config dir");

    let config_toml = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"
"#;
    std::fs::write(config_dir.join("config.toml"), config_toml).expect("write config.toml");

    // The script must serve two prompts: one for the first run, one for the follow-up after resume.
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "first response" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "text", "text": "follow up" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let script_path = temp.path().join("script.json");
    std::fs::write(&script_path, script.to_string()).expect("write script.json");

    // First run: create a session, run one prompt, capture the id.
    let session_id = {
        let mut child = meka_acp()
            .arg("acp")
            .env("MEKA_CONFIG_DIR", &config_dir)
            .env("MEKA_DATA_DIR", &data_dir)
            .env("HOME", temp.path())
            .env("MEKA_MOCK_PROVIDER", "1")
            .env("MEKA_MOCK_PROVIDER_SCRIPT", &script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn first");
        let mut stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let stderr_pipe = child.stderr.take().expect("stderr");
        let mut reader = BufReader::new(stdout);
        let _stderr_handle = std::thread::spawn(move || {
            let mut buf = String::new();
            let mut r = BufReader::new(stderr_pipe);
            while r.read_line(&mut buf).unwrap_or(0) > 0 {}
            buf
        });
        let deadline = Instant::now() + Duration::from_secs(15);
        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
        )
        .expect("init");
        let _ = read_until(&mut reader, deadline, |line| line.contains("\"id\":1"));

        let new_req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": { "cwd": config_dir.clone(), "mcpServers": [] }
        });
        writeln!(stdin, "{}", new_req).expect("session/new");
        let new_lines = read_until(&mut reader, deadline, |line| line.contains("\"id\":2"));
        let sid = serde_json::from_str::<serde_json::Value>(
            new_lines
                .iter()
                .find(|line| line.contains("\"id\":2"))
                .expect("session/new response"),
        )
        .expect("parse")["result"]["sessionId"]
            .as_str()
            .expect("sessionId")
            .to_string();

        let prompt_req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/prompt",
            "params": {
                "sessionId": sid,
                "prompt": [{ "type": "text", "text": "first" }]
            }
        });
        writeln!(stdin, "{}", prompt_req).expect("first prompt");
        let _ = read_until(&mut reader, deadline, |line| line.contains("\"id\":3"));

        drop(stdin);
        let _ = child.kill();
        let _ = child.wait();
        sid
    };

    // Second run: resume + a follow-up prompt.
    let mut child = meka_acp()
        .arg("acp")
        .env("MEKA_CONFIG_DIR", &config_dir)
        .env("MEKA_DATA_DIR", &data_dir)
        .env("HOME", temp.path())
        .env("MEKA_MOCK_PROVIDER", "1")
        .env("MEKA_MOCK_PROVIDER_SCRIPT", &script_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn resume");
    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let stderr_pipe = child.stderr.take().expect("stderr");
    let mut reader = BufReader::new(stdout);
    let stderr_handle = std::thread::spawn(move || {
        let mut buf = String::new();
        let mut r = BufReader::new(stderr_pipe);
        while r.read_line(&mut buf).unwrap_or(0) > 0 {}
        buf
    });
    let deadline = Instant::now() + Duration::from_secs(15);

    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
    )
    .expect("init");
    let _ = read_until(&mut reader, deadline, |line| line.contains("\"id\":1"));

    let resume_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 6,
        "method": "session/resume",
        "params": {
            "sessionId": session_id.clone(),
            "cwd": config_dir.clone(),
            "mcpServers": []
        }
    });
    writeln!(stdin, "{}", resume_req).expect("session/resume");
    let resume_lines = read_until(&mut reader, deadline, |line| line.contains("\"id\":6"));

    // The `available_commands_update` push is allowed (and expected) on resume. What must NOT
    // appear is a replay update: `user_message_chunk`, `agent_message_chunk`, `tool_call`, or
    // `tool_call_update`.
    let mut saw_replay_update = false;
    let mut resume_response: Option<serde_json::Value> = None;
    for line in &resume_lines {
        let value: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if value["method"] == "session/update" {
            let kind = value["params"]["update"]["sessionUpdate"]
                .as_str()
                .unwrap_or_default();
            if matches!(
                kind,
                "user_message_chunk"
                    | "agent_message_chunk"
                    | "agent_thought_chunk"
                    | "tool_call"
                    | "tool_call_update"
            ) {
                saw_replay_update = true;
            }
        }
        if value["id"] == 6 {
            resume_response = Some(value);
        }
    }
    assert!(
        !saw_replay_update,
        "session/resume must NOT emit replay updates; stream:\n{}",
        resume_lines.join(""),
    );
    let resume_response = resume_response.unwrap_or_else(|| {
        panic!(
            "no ResumeSessionResponse; stream:\n{}",
            resume_lines.join(""),
        )
    });
    assert!(
        resume_response["result"].is_object(),
        "resume must succeed: {}",
        resume_response,
    );

    // Follow-up prompt confirms the slot is active.
    let prompt_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 7,
        "method": "session/prompt",
        "params": {
            "sessionId": session_id.clone(),
            "prompt": [{ "type": "text", "text": "follow up" }]
        }
    });
    writeln!(stdin, "{}", prompt_req).expect("follow-up prompt");
    let prompt_lines = read_until(&mut reader, deadline, |line| line.contains("\"id\":7"));

    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();

    let prompt_response: serde_json::Value = serde_json::from_str(
        prompt_lines
            .iter()
            .find(|line| line.contains("\"id\":7"))
            .unwrap_or_else(|| {
                panic!(
                    "no follow-up prompt response; STDERR:\n{}",
                    stderr_handle.join().unwrap_or_default(),
                )
            }),
    )
    .expect("parse follow-up");
    assert_eq!(prompt_response["result"]["stopReason"], "end_turn");
}

/// `session/close` clears the active slot so a subsequent `session/new` succeeds within the same
/// process.
#[test]
fn acp_session_close_clears_slot_for_subsequent_new() {
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, None);
    let first_id = harness.new_session();

    // Second session/new without close: must succeed (multi-session ACP now allows N concurrent
    // sessions per process).
    let second_id = harness.new_session();
    assert_ne!(
        second_id, first_id,
        "second session/new must mint a fresh sessionId",
    );

    // Close the first session.
    let close_response = harness.close_session(&first_id);
    assert!(
        close_response["result"].is_object(),
        "expected ok result for session/close: {}",
        close_response,
    );

    // Re-closing the first session must error; it's gone.
    let reclose = harness.close_session(&first_id);
    assert!(
        reclose["error"].is_object(),
        "re-closing a removed session must error: {}",
        reclose,
    );
}

/// a skill installed under `$MEKA_CONFIG_DIR/skills/` shows up in the `available_commands_update`
/// push that follows `session/new`, AND the `NewSessionResponse` carries the configured mode
/// picker.
#[test]
fn acp_session_new_advertises_skills_and_modes() {
    // Provider stub + a non-default enabled set so we can assert exactly which modes get
    // advertised.
    let config_toml = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"

[permissions]
default = "read"
enabled = ["read", "ask", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(|config_dir| {
            let skill_dir = config_dir.join("skills").join("demo-skill");
            std::fs::create_dir_all(&skill_dir).expect("mkdir skill");
            std::fs::write(
                skill_dir.join("SKILL.md"),
                "---\ndescription: a demo skill\n---\ndo stuff\n",
            )
            .expect("write SKILL.md");
            // Empty script; we never run a turn.
            serde_json::json!([])
        })
        .build();

    // session/new fires before any session/update notifications, so we can't filter notifications
    // by sid up front. Send the request manually, then walk the stream picking up the intermediate
    // `available_commands_update` notification(s) and the eventual response together.
    let cwd = harness.config_dir().to_path_buf();
    let id = harness.send_request(
        "session/new",
        serde_json::json!({ "cwd": cwd, "mcpServers": [] }),
    );
    let needle = format!("\"id\":{}", id);
    let mut saw_skill = false;
    let mut new_response: Option<serde_json::Value> = None;
    let mut transcript = String::new();
    while Instant::now() < harness.deadline {
        let mut line = String::new();
        match harness.reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }
        transcript.push_str(&line);
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        if value["method"] == "session/update" {
            let update = &value["params"]["update"];
            if update["sessionUpdate"] == "available_commands_update"
                && let Some(cmds) = update["availableCommands"].as_array()
                && cmds.iter().any(|c| c["name"] == "demo-skill")
            {
                saw_skill = true;
            }
        }
        if line.contains(&needle) && response_matches(&line, &needle) {
            new_response = Some(value);
            break;
        }
    }
    assert!(
        saw_skill,
        "expected available_commands_update with demo-skill; transcript:\n{}",
        transcript,
    );
    let response = new_response.unwrap_or_else(|| {
        panic!("no session/new response; transcript:\n{}", transcript);
    });
    let modes = &response["result"]["modes"];
    let ids: Vec<String> = modes["availableModes"]
        .as_array()
        .expect("availableModes")
        .iter()
        .map(|m| m["id"].as_str().unwrap_or_default().to_string())
        .collect();
    assert_eq!(ids, vec!["read", "ask", "unrestricted"]);
    assert_eq!(modes["currentModeId"], "read");
}

/// a `session/prompt` whose text is `/<skill-name>` resolves to the rendered skill body before
/// being handed to the agent. Asserts the prompt completes successfully (the alternative, the
/// helper returning an error from an unknown skill, would surface as a JSON-RPC error response
/// instead of `stopReason=end_turn`).
#[test]
fn acp_session_prompt_invokes_skill_by_slash_name() {
    let config_toml = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(|config_dir| {
            let skill_dir = config_dir.join("skills").join("hello");
            std::fs::create_dir_all(&skill_dir).expect("mkdir");
            std::fs::write(
                skill_dir.join("SKILL.md"),
                "---\ndescription: say hi\n---\nrespond with a greeting\n",
            )
            .expect("write SKILL.md");
            serde_json::json!([[
                { "kind": "text", "text": "hello from agent" },
                { "kind": "message_end", "stop_reason": "end_turn" }
            ]])
        })
        .build();
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "/hello but be brief");
    let response = harness.await_response(id);
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "skill invocation should run a normal turn: {}",
        response,
    );
}

/// A `session/prompt` whose first text token *looks* like a skill invocation but doesn't match an
/// installed skill must pass through to the model rather than erroring. The original rejection
/// broke paste UX: pasted text like `/usr local lib` or the start of a sentence like `/etc and so
/// on` would be parsed as `name="usr"` / `name="etc"`, validated as a syntactically-OK skill name,
/// then rejected with `InvalidParams "unknown skill"`. The model can respond with "I don't know
/// that command" if the user genuinely meant `/skill-name`.
#[test]
fn acp_session_prompt_passes_through_unknown_skill_name() {
    let script = serde_json::json!([[
        { "kind": "text", "text": "ok" },
        { "kind": "message_end", "stop_reason": "end_turn" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "/unknown-skill but otherwise valid text");
    let response = harness.await_response(id);
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "unknown skill name must pass through to the model, not error: {}",
        response,
    );
}

/// A `/<skill>` invocation that resolves to an installed skill but whose body file becomes
/// unreadable between scan and invocation must surface as JSON-RPC `InternalError` (-32603), not
/// `InvalidParams` (-32602). The client's request was syntactically valid and named a real
/// skill; the failure is a server-side disk problem.
///
/// We trigger this by chmod-ing SKILL.md to 0 after the skills cache has registered it. The cache's
/// disk-snapshot key is the file's mtime (see `disk_snapshot` in `src/skills.rs`), which `chmod`
/// doesn't touch, so the cache happily serves the stale entry while `load_skill_body` fails with
/// EACCES.
#[cfg(unix)]
#[test]
fn acp_session_prompt_skill_body_unreadable_is_internal_error() {
    use std::os::unix::fs::PermissionsExt;

    let mut harness = AcpTestHarness::builder()
        .config(ACP_INVALID_PARAMS_CONFIG)
        .pre_spawn(|config_dir| {
            let skill_dir = config_dir.join("skills").join("doomed");
            std::fs::create_dir_all(&skill_dir).expect("mkdir skill");
            std::fs::write(
                skill_dir.join("SKILL.md"),
                "---\ndescription: will become unreadable\n---\nbody\n",
            )
            .expect("write SKILL.md");
            serde_json::json!([])
        })
        .build();
    let sid = harness.new_session();
    let skill_md = harness
        .config_dir()
        .join("skills")
        .join("doomed")
        .join("SKILL.md");
    // chmod 0 after the in-process cache has already scanned the file during `session/new`. Since
    // mtime is unchanged, the cache's snapshot-equality check serves the stale skill list and
    // load_skill_body hits EACCES.
    std::fs::set_permissions(&skill_md, std::fs::Permissions::from_mode(0o000))
        .expect("chmod 0 SKILL.md");

    let id = harness.prompt(&sid, "/doomed");
    let response = harness.await_response(id);

    // Restore perms before assertions so a panic doesn't break tempdir cleanup.
    let _ = std::fs::set_permissions(&skill_md, std::fs::Permissions::from_mode(0o644));

    let error = response["error"]
        .as_object()
        .unwrap_or_else(|| panic!("expected JSON-RPC error: {}", response));
    assert_eq!(
        error["code"].as_i64(),
        Some(-32603),
        "skill body load failure must map to InternalError (-32603), not InvalidParams; got: {}",
        response,
    );
    let data = error["data"].as_str().unwrap_or_else(|| {
        panic!(
            "expected error.data to carry the detail string: {}",
            response
        )
    });
    assert!(
        data.contains("failed to load skill 'doomed'"),
        "error.data should mention the doomed skill name and load failure; got: {}",
        data,
    );
}

/// `session/set_mode` flips the active permission level and emits `current_mode_update`. A request
/// for a mode outside the enabled set returns a JSON-RPC error.
#[test]
fn acp_session_set_mode_flips_permission_and_emits_update() {
    const CONFIG: &str = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"

[permissions]
default = "read"
enabled = ["read", "ask"]
"#;
    let mut harness = AcpTestHarness::spawn(CONFIG, None);

    // Confirm advertised modes match the enabled set.
    let new_response = harness.request(
        "session/new",
        serde_json::json!({
            "cwd": harness.config_dir().to_path_buf(),
            "mcpServers": []
        }),
    );
    let ids: Vec<String> = new_response["result"]["modes"]["availableModes"]
        .as_array()
        .expect("availableModes")
        .iter()
        .map(|m| m["id"].as_str().unwrap_or_default().to_string())
        .collect();
    assert_eq!(ids, vec!["read", "ask"]);
    let sid = new_response["result"]["sessionId"]
        .as_str()
        .expect("sessionId")
        .to_string();

    // Valid set_mode: read → ask. The `current_mode_update` notification arrives via the same
    // session/update channel, which `request` discards; collect it via a small ad-hoc loop instead
    // by issuing the request and watching for the notification before the response.
    let set_id = harness.send_request(
        "session/set_mode",
        serde_json::json!({ "sessionId": sid, "modeId": "ask" }),
    );
    let (updates, set_response) = harness.collect_updates(&sid, set_id);
    assert!(
        updates.iter().any(|u| {
            u["params"]["update"]["sessionUpdate"] == "current_mode_update"
                && u["params"]["update"]["currentModeId"] == "ask"
        }),
        "expected current_mode_update with currentModeId=ask; updates: {:?}",
        updates,
    );
    assert!(
        set_response["result"].is_object(),
        "set_mode must succeed: {}",
        set_response,
    );

    // Invalid set_mode: write is not in the enabled set.
    let bad_response = harness.set_mode(&sid, "unrestricted");
    assert!(
        bad_response["error"].is_object(),
        "set_mode for a disabled mode must error: {}",
        bad_response,
    );
}

/// when the client advertises `fs.read_text_file`, a `read_file` tool call delegates to
/// `fs/read_text_file` rather than touching the disk. The mock provider scripts the tool use; the
/// test harness intercepts the outgoing fs request and answers with canned content.
#[test]
fn acp_fs_read_text_file_is_delegated_when_capability_offered() {
    let config_toml = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .capabilities(serde_json::json!({
            "fs": { "readTextFile": true, "writeTextFile": false },
            "terminal": false
        }))
        .pre_spawn(|config_dir| {
            // Real on-disk file with one content; the delegate returns *different* content, so the
            // assertion proves the delegate path was used.
            let target = config_dir.join("delegated.txt");
            std::fs::write(&target, "ON DISK\n").expect("write target");
            serde_json::json!([
                [
                    { "kind": "text", "text": "reading..." },
                    { "kind": "tool_use_start", "id": "call_read", "name": "read_file" },
                    { "kind": "tool_use_end", "input": { "path": target.to_str().unwrap() } },
                    { "kind": "message_end", "stop_reason": "tool_use" }
                ],
                [
                    { "kind": "text", "text": "done" },
                    { "kind": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .build();
    let target = harness.config_dir().join("delegated.txt");
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "read it");

    let mut saw_fs_read_request = false;
    let _ = harness.await_response_with_dispatch(id, |value| {
        if value["method"] == "fs/read_text_file" {
            saw_fs_read_request = true;
            Some(serde_json::json!({
                "jsonrpc": "2.0",
                "id": value["id"].clone(),
                "result": { "content": "FROM EDITOR BUFFER\n" }
            }))
        } else {
            None
        }
    });

    assert!(
        saw_fs_read_request,
        "expected a fs/read_text_file request from meka",
    );

    // The on-disk file is untouched (we only wrote `ON DISK` once before the test). The delegate
    // returned different content, proving the tool used the delegate result rather than reading the
    // disk.
    assert_eq!(std::fs::read_to_string(&target).expect("read"), "ON DISK\n");
}

/// when the client advertises `fs.write_text_file`, a `write_file` tool call delegates to
/// `fs/write_text_file` and does NOT touch the local disk. The test harness intercepts the request,
/// replies ok, and asserts no local file was created.
#[test]
fn acp_fs_write_text_file_is_delegated_when_capability_offered() {
    let content_to_write = "hello from delegated write";
    // `write` mode so the agent's permission gate doesn't refuse `write_file` before we even reach
    // the delegation seam.
    let config_toml = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"

[permissions]
default = "unrestricted"
enabled = ["read", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .capabilities(serde_json::json!({
            "fs": { "readTextFile": true, "writeTextFile": true },
            "terminal": false
        }))
        .pre_spawn(move |config_dir| {
            let target = config_dir.join("delegated-write.txt");
            serde_json::json!([
                [
                    { "kind": "text", "text": "writing..." },
                    { "kind": "tool_use_start", "id": "call_write", "name": "write_file" },
                    {
                        "kind": "tool_use_end",
                        "input": { "path": target.to_str().unwrap(), "content": content_to_write }
                    },
                    { "kind": "message_end", "stop_reason": "tool_use" }
                ],
                [
                    { "kind": "text", "text": "done" },
                    { "kind": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .build();
    // `write_file` canonicalizes the parent directory before handing the path to the delegate, so
    // the expected path matches `/private/var/...` on macOS rather than the `/var/...` tempdir
    // returns from `config_dir()`.
    // Stripped of the `\\?\` prefix the way meka strips it, because this is compared against a
    // path meka reports rather than one the test constructs. `canonicalize` alone yields the
    // verbatim spelling, which is the one meka never emits, so the mismatch is Windows-only.
    let target_dir = {
        let canonical = std::fs::canonicalize(harness.config_dir()).expect("canonicalize tempdir");
        let text = canonical.to_string_lossy();
        match text.strip_prefix(r"\\?\") {
            Some(stripped) => std::path::PathBuf::from(stripped),
            None => canonical,
        }
    };
    let target = target_dir.join("delegated-write.txt");
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "write it");

    let mut saw_fs_write = false;
    let mut delegated_path: Option<String> = None;
    let mut delegated_content: Option<String> = None;
    let _ = harness.await_response_with_dispatch(id, |value| match value["method"].as_str() {
        Some("fs/read_text_file") => {
            // Pre-read for diff metadata: return file-not-found shaped error so write_file falls
            // back to None old_text.
            Some(serde_json::json!({
                "jsonrpc": "2.0",
                "id": value["id"].clone(),
                "error": { "code": -32603, "message": "file not open" }
            }))
        }
        Some("fs/write_text_file") => {
            saw_fs_write = true;
            delegated_path = value["params"]["path"].as_str().map(String::from);
            delegated_content = value["params"]["content"].as_str().map(String::from);
            Some(serde_json::json!({
                "jsonrpc": "2.0",
                "id": value["id"].clone(),
                "result": {}
            }))
        }
        _ => None,
    });

    assert!(saw_fs_write, "expected a fs/write_text_file request");
    assert_eq!(
        delegated_path.as_deref(),
        Some(target.to_str().unwrap()),
        "delegate received wrong path"
    );
    assert_eq!(
        delegated_content.as_deref(),
        Some(content_to_write),
        "delegate received wrong content"
    );
    // No local file on disk; the delegate handled the write.
    assert!(
        !target.exists(),
        "meka wrote a local file despite delegating: {}",
        target.display(),
    );
}

/// when the client does NOT advertise `fs.write_text_file`, `write_file` falls back to a local disk
/// write. No `fs/write_text_file` request should appear.
#[test]
fn acp_write_file_falls_back_to_local_when_no_capability() {
    let content_to_write = "wrote locally";
    let config_toml = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"

[permissions]
default = "unrestricted"
enabled = ["read", "unrestricted"]
"#;
    // No capabilities advertised; the harness default is `{}`.
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(move |config_dir| {
            let target = config_dir.join("local-write.txt");
            serde_json::json!([
                [
                    { "kind": "text", "text": "writing..." },
                    { "kind": "tool_use_start", "id": "call_write", "name": "write_file" },
                    {
                        "kind": "tool_use_end",
                        "input": { "path": target.to_str().unwrap(), "content": content_to_write }
                    },
                    { "kind": "message_end", "stop_reason": "tool_use" }
                ],
                [
                    { "kind": "text", "text": "done" },
                    { "kind": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .build();
    let target = harness.config_dir().join("local-write.txt");
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "write it");

    let mut saw_fs_write = false;
    let _ = harness.await_response_with_dispatch(id, |value| {
        if value["method"] == "fs/write_text_file" {
            saw_fs_write = true;
        }
        None
    });

    assert!(
        !saw_fs_write,
        "fs/write_text_file must NOT be issued without capability"
    );
    let written = std::fs::read_to_string(&target).expect("local write should have happened");
    assert_eq!(written, content_to_write);
}

/// The capabilities Zed advertises, in the shape that matters here: `terminal` says it implements
/// `terminal/*` requests, and the separate `_meta.terminal_output` key says it renders agent-owned
/// terminals from meka's `_meta` frames. Only the latter gates the terminal rendering.
fn zed_shaped_capabilities() -> serde_json::Value {
    serde_json::json!({
        "terminal": true,
        "_meta": { "terminal_output": true },
    })
}

/// `execute_command` always runs in meka's own (sandboxed) child process, never in the client's
/// terminal, whatever the permission mode and whatever the client advertises. `write` is the mode
/// that used to delegate, and `ask` is the mode where delegating was a sandbox bypass: meka treats
/// `ask` as sandboxed (it hard-errors when no sandbox backend is available) yet handed the command
/// to an unsandboxed editor terminal anyway. Guards both by asserting no `terminal/*` traffic and
/// that the output is the local shell's, not the client's.
#[test]
fn acp_execute_command_never_leaves_meka() {
    for mode in ["unrestricted", "ask"] {
        let config_toml = format!(
            r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"

[permissions]
default = "{mode}"
enabled = ["read", "ask", "unrestricted"]
"#
        );
        let script = serde_json::json!([
            [
                { "kind": "text", "text": "running..." },
                { "kind": "tool_use_start", "id": "call_exec", "name": "execute_command" },
                { "kind": "tool_use_end", "input": { "command": "echo ran-inside-meka" } },
                { "kind": "message_end", "stop_reason": "tool_use" }
            ],
            [
                { "kind": "text", "text": "done" },
                { "kind": "message_end", "stop_reason": "end_turn" }
            ]
        ]);
        let mut harness = AcpTestHarness::spawn_with_capabilities(
            &config_toml,
            Some(script),
            zed_shaped_capabilities(),
        );
        let sid = harness.new_session();
        let id = harness.prompt(&sid, "run it");

        let mut terminal_methods: Vec<String> = Vec::new();
        let (updates, _response) = harness.collect_updates_with_dispatch(&sid, id, |value| {
            if let Some(method) = value["method"].as_str()
                && method.starts_with("terminal/")
            {
                terminal_methods.push(method.to_string());
            }
            // Approve anything `ask` mode prompts for, so the tool actually runs.
            if value["method"] == "session/request_permission" {
                return Some(serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": value["id"].clone(),
                    "result": { "outcome": { "outcome": "selected", "optionId": "allow_once" } }
                }));
            }
            None
        });

        assert!(
            terminal_methods.is_empty(),
            "{mode} mode must run execute_command inside meka; saw {terminal_methods:?}",
        );
        // Either terminal status will do. Whether the command *succeeds* is a property of the
        // host's sandbox, not of this test: on a runner where the platform sandbox rejects the
        // profile, a read-mode-style run legitimately reports `failed`. What must hold everywhere
        // is that the call reached a terminal state without leaving meka.
        let status = updates
            .iter()
            .map(|u| &u["params"]["update"])
            .filter(|u| u["sessionUpdate"] == "tool_call_update")
            .filter_map(|u| u["status"].as_str())
            .next_back()
            .unwrap_or_else(|| {
                panic!("{mode}: no tool_call_update carried a status: {updates:#?}")
            });
        assert!(
            status == "completed" || status == "failed",
            "{mode}: unexpected terminal status {status:?}",
        );

        // When it did run, the output must be meka's own shell rather than a delegated one.
        if status == "completed" {
            let streamed: String = updates
                .iter()
                .filter_map(|u| {
                    u["params"]["update"]["_meta"]["terminal_output"]["data"]
                        .as_str()
                        .map(str::to_string)
                })
                .collect();
            assert!(
                streamed.contains("ran-inside-meka"),
                "{mode}: expected the local shell's output; got {streamed:?}",
            );
        }
    }
}

/// `unrestricted`, so `execute_command` is not subject to the sandbox's availability on the test
/// host.
const ACP_WRITE_MODE_CONFIG: &str = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"

[permissions]
default = "unrestricted"
enabled = ["read", "unrestricted"]
"#;

/// A command's output reaches the client while it is still running, not only when it exits. The
/// scripted command prints, sleeps past the throttle interval, then prints again, so a correct
/// implementation emits at least one `tool_call_update` carrying `first` but not yet `second`.
/// Before live output existed, the only update for a tool call was the one at completion, and a
/// long build showed a bare spinner for its whole duration.
#[test]
fn acp_execute_command_streams_output_while_running() {
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "running..." },
            { "kind": "tool_use_start", "id": "call_exec", "name": "execute_command" },
            {
                "kind": "tool_use_end",
                "input": { "command": "echo first; sleep 0.5; echo second" }
            },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = AcpTestHarness::spawn(ACP_WRITE_MODE_CONFIG, Some(script));
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "run it");
    let (updates, response) = harness.collect_updates_with_dispatch(&sid, id, |_value| None);

    assert_eq!(response["result"]["stopReason"], "end_turn");

    let exec_updates: Vec<&serde_json::Value> = updates
        .iter()
        .map(|u| &u["params"]["update"])
        .filter(|u| u["sessionUpdate"] == "tool_call_update" && u["toolCallId"] == "call_exec")
        .collect();

    let partial = exec_updates.iter().find(|u| {
        let rendered = u["content"].to_string();
        rendered.contains("first") && !rendered.contains("second")
    });
    assert!(
        partial.is_some(),
        "expected an in-progress update carrying only the first line; got: {exec_updates:#?}",
    );
    assert!(
        partial.expect("checked above")["status"].is_null(),
        "a live-output update must not claim the call finished",
    );

    let completed = exec_updates
        .iter()
        .find(|u| u["status"] == "completed")
        .unwrap_or_else(|| panic!("expected a completed update; got: {exec_updates:#?}"));
    let rendered = completed["content"].to_string();
    assert!(
        rendered.contains("first") && rendered.contains("second"),
        "the final update must carry the whole output; got {rendered}",
    );
}

/// A terminal-capable client gets the agent-owned terminal channel: the call announces a terminal
/// so the client can register it, each update carries only the *new* bytes (the client owns the
/// scrollback, so re-sending would duplicate), and the call ends with a real exit status.
///
/// This is what makes output visible in Zed. A `kind: execute` tool call whose content is a text
/// block renders with no expansion affordance at all, so the output is sent but never shown.
#[test]
fn acp_terminal_capable_client_gets_an_agent_owned_terminal() {
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "running..." },
            { "kind": "tool_use_start", "id": "call_exec", "name": "execute_command" },
            {
                "kind": "tool_use_end",
                "input": { "command": "echo alpha; sleep 0.4; echo beta; exit 7" }
            },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = AcpTestHarness::spawn_with_capabilities(
        ACP_WRITE_MODE_CONFIG,
        Some(script),
        zed_shaped_capabilities(),
    );
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "run it");
    let (updates, _response) = harness.collect_updates_with_dispatch(&sid, id, |_value| None);

    let for_call = |kind: &str| -> Vec<serde_json::Value> {
        updates
            .iter()
            .map(|u| &u["params"]["update"])
            .filter(|u| u["toolCallId"] == "call_exec")
            .filter(|u| !u["_meta"][kind].is_null())
            .cloned()
            .collect()
    };

    // The terminal has to be announced before anything references it; without a matching create,
    // clients park the output against an id they were never told about and show an empty terminal.
    let announced = for_call("terminal_info");
    assert_eq!(
        announced.len(),
        1,
        "expected exactly one terminal_info; got {announced:#?}",
    );
    assert_eq!(announced[0]["sessionUpdate"], "tool_call");
    assert_eq!(announced[0]["content"][0]["type"], "terminal");
    assert_eq!(announced[0]["content"][0]["terminalId"], "call_exec");

    // Output is appended, so no chunk may repeat what an earlier one already delivered.
    let output_frames = for_call("terminal_output");
    let chunks: Vec<&str> = output_frames
        .iter()
        .filter_map(|u| u["_meta"]["terminal_output"]["data"].as_str())
        .collect();
    assert!(
        chunks.len() >= 2,
        "expected the output to arrive in pieces while the command ran; got {chunks:?}",
    );
    assert_eq!(
        // The shell is PowerShell on Windows and `sh` elsewhere, and the two disagree about line
        // endings. What is being asserted is that the appended chunks reassemble to the command's
        // output exactly once, which is independent of that.
        chunks.concat().replace("\r\n", "\n"),
        "alpha\nbeta\n",
        "appended chunks must reassemble to the command's output exactly once",
    );
    assert!(
        chunks.iter().all(|chunk| !chunk.is_empty()),
        "an empty append is a wasted notification; got {chunks:?}",
    );

    // The real exit code, not a synthesised 1: a terminal shows the status, and 7 is only
    // available because the shell tool reports it structurally.
    let exits = for_call("terminal_exit");
    assert_eq!(exits.len(), 1, "expected exactly one terminal_exit");
    assert_eq!(exits[0]["_meta"]["terminal_exit"]["exit_code"], 7);
    assert_eq!(exits[0]["status"], "failed");
    assert_eq!(
        exits[0]["content"][0]["type"], "terminal",
        "the completed call must keep the terminal, not swap in a flattened copy",
    );
}

/// `terminal: true` alone must not trigger terminal rendering. That capability means "I implement
/// `terminal/*` requests", which is about running commands *in the client*; a client can offer it
/// and have no idea what meka's `_meta` terminal frames mean. Sending a terminal content block to
/// such a client resolves to nothing and displays no output at all, which is the exact failure the
/// terminal path exists to fix. Only `_meta.terminal_output` licenses it.
#[test]
fn acp_terminal_capability_alone_does_not_enable_terminal_rendering() {
    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "call_exec", "name": "execute_command" },
            { "kind": "tool_use_end", "input": { "command": "echo capability-check" } },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = AcpTestHarness::spawn_with_capabilities(
        ACP_WRITE_MODE_CONFIG,
        Some(script),
        // `terminal/*` implemented, agent-owned terminal frames not understood.
        serde_json::json!({ "terminal": true }),
    );
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "run it");
    let (updates, _response) = harness.collect_updates_with_dispatch(&sid, id, |_value| None);

    let call_updates: Vec<&serde_json::Value> = updates
        .iter()
        .map(|u| &u["params"]["update"])
        .filter(|u| u["toolCallId"] == "call_exec")
        .collect();
    assert!(
        call_updates
            .iter()
            .all(|u| u["content"][0]["type"] != "terminal"),
        "a terminal block here would render as nothing; got {call_updates:#?}",
    );
    assert!(
        call_updates.iter().any(|u| {
            u["content"][0]["content"]["text"]
                .as_str()
                .is_some_and(|text| text.contains("capability-check"))
        }),
        "expected the console text fallback; got {call_updates:#?}",
    );
}

/// A client that never advertised `terminal` keeps the text rendering, because a terminal block it
/// cannot resolve renders as nothing at all.
#[test]
fn acp_client_without_terminal_capability_gets_console_text() {
    let script = serde_json::json!([
        [
            { "kind": "tool_use_start", "id": "call_exec", "name": "execute_command" },
            { "kind": "tool_use_end", "input": { "command": "echo plain-text-path" } },
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = AcpTestHarness::spawn_with_capabilities(
        ACP_WRITE_MODE_CONFIG,
        Some(script),
        serde_json::json!({ "terminal": false }),
    );
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "run it");
    let (updates, _response) = harness.collect_updates_with_dispatch(&sid, id, |_value| None);

    let call_updates: Vec<&serde_json::Value> = updates
        .iter()
        .map(|u| &u["params"]["update"])
        .filter(|u| u["toolCallId"] == "call_exec")
        .collect();
    assert!(
        call_updates.iter().all(|u| u["_meta"].is_null()),
        "no terminal _meta may be sent to a client that can't resolve a terminal; got \
         {call_updates:#?}",
    );
    assert!(
        call_updates
            .iter()
            .all(|u| u["content"][0]["type"] != "terminal"),
        "no terminal content block either; got {call_updates:#?}",
    );
    let completed = call_updates
        .iter()
        .find(|u| u["status"] == "completed")
        .unwrap_or_else(|| panic!("expected a completed update; got {call_updates:#?}"));
    assert!(
        completed["content"][0]["content"]["text"]
            .as_str()
            .is_some_and(|text| text.contains("plain-text-path")),
        "expected a console text block carrying the output; got {completed:#?}",
    );
}

/// Build a JSON-RPC error response for a request the dispatch closure wants to fail.
fn jsonrpc_error(id: serde_json::Value, message: &str) -> serde_json::Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": -32603, "message": message }
    })
}

/// Spec: `session/cancel` MUST resolve the in-flight prompt with `stopReason: "cancelled"`. The
/// mock provider stalls mid-turn via a `Sleep` event so the test can fire the cancel notification
/// while the agent loop is parked inside `provider.stream`. Regression guard for the bug where
/// `Mutex<ServerState>` was held across `agent.run_turn().await`, which serialized the cancel
/// notification behind the prompt and made cancellation effectively useless.
#[test]
fn acp_session_cancel_interrupts_running_prompt() {
    let temp = tempfile::tempdir().expect("tempdir");
    let config_dir = temp.path().join("meka");
    let data_dir = temp.path().join("data").join("meka");
    std::fs::create_dir_all(&config_dir).expect("create config dir");

    let config_toml = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"
"#;
    std::fs::write(config_dir.join("config.toml"), config_toml).expect("write config.toml");

    // Round 1: a short "starting" delta so the test knows the turn started, then a 5s sleep that
    // races against cancel. If cancel arrives in time the mock returns early; the agent loop's
    // post-stream cancellation check breaks with Interrupted. If cancel is starved (bug regressed),
    // the sleep finishes, the text "done" + end_turn fires, and the response carries `end_turn`.
    // The assertion below catches that.
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "starting..." },
            { "kind": "sleep", "ms": 5000 },
            { "kind": "text", "text": "done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let script_path = temp.path().join("script.json");
    std::fs::write(&script_path, script.to_string()).expect("write script");

    let mut child = meka_acp()
        .arg("acp")
        .env("MEKA_CONFIG_DIR", &config_dir)
        .env("MEKA_DATA_DIR", &data_dir)
        .env("HOME", temp.path())
        .env("MEKA_MOCK_PROVIDER", "1")
        .env("MEKA_MOCK_PROVIDER_SCRIPT", &script_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let stderr_pipe = child.stderr.take().expect("stderr");
    let mut reader = BufReader::new(stdout);
    let stderr_handle = std::thread::spawn(move || {
        let mut buf = String::new();
        let mut r = BufReader::new(stderr_pipe);
        while r.read_line(&mut buf).unwrap_or(0) > 0 {}
        buf
    });
    let deadline = Instant::now() + Duration::from_secs(15);

    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
    )
    .expect("init");
    let _ = read_until(&mut reader, deadline, |line| line.contains("\"id\":1"));

    let new_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "session/new",
        "params": { "cwd": config_dir.clone(), "mcpServers": [] }
    });
    writeln!(stdin, "{}", new_req).expect("session/new");
    let new_lines = read_until(&mut reader, deadline, |line| line.contains("\"id\":2"));
    let sid = serde_json::from_str::<serde_json::Value>(
        new_lines
            .iter()
            .find(|line| line.contains("\"id\":2"))
            .expect("session/new"),
    )
    .expect("parse")["result"]["sessionId"]
        .as_str()
        .expect("sessionId")
        .to_string();

    let prompt_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "session/prompt",
        "params": {
            "sessionId": sid,
            "prompt": [{ "type": "text", "text": "stall then cancel" }]
        }
    });
    writeln!(stdin, "{}", prompt_req).expect("prompt");

    // Wait until we've seen the "starting..." chunk so we know the turn is actually parked inside
    // the mock's sleep; firing cancel any earlier might race the prompt setup.
    let start_deadline = Instant::now() + Duration::from_secs(5);
    let _ = read_until(&mut reader, start_deadline, |line| {
        line.contains("starting...")
    });

    // Fire session/cancel. If the bug regressed, this notification would queue behind the state
    // mutex; the assertion below would see stopReason=end_turn after the 5s sleep finishes.
    let cancel_notif = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "session/cancel",
        "params": { "sessionId": sid }
    });
    writeln!(stdin, "{}", cancel_notif).expect("cancel");

    // Tight deadline: cancel should resolve well before the 5s sleep would have completed
    // naturally. Allow generous slack for CI variance but well short of 5s.
    let response_deadline = Instant::now() + Duration::from_secs(3);
    let lines = read_until(&mut reader, response_deadline, |line| {
        line.contains("\"id\":3")
    });

    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();

    let response_line = lines
        .iter()
        .find(|line| line.contains("\"id\":3"))
        .unwrap_or_else(|| {
            panic!(
                "no PromptResponse before deadline, cancel was likely starved.\nSTDERR:\n{}\nstream:\n{}",
                stderr_handle.join().unwrap_or_default(),
                lines.join(""),
            )
        });
    let response: serde_json::Value =
        serde_json::from_str(response_line).expect("parse PromptResponse");
    assert_eq!(
        response["result"]["stopReason"], "cancelled",
        "session/cancel must resolve the in-flight prompt with cancelled; got: {}",
        response,
    );
}

/// Regression: a turn interrupted mid-stream must persist the partial assistant text so it survives
/// resume. Previously the partial was appended only in memory and discarded on exit, so resume
/// showed only the user prompt. Round 1 streams a partial answer then stalls in a `sleep`; the test
/// fires `session/cancel` once the partial has streamed. Round 2 loads the session and asserts the
/// replay carries the partial answer (and not the post-interrupt text, which never streamed).
#[test]
fn acp_interrupted_turn_persists_partial_assistant_text() {
    let temp = tempfile::tempdir().expect("tempdir");
    let config_dir = temp.path().join("meka");
    let data_dir = temp.path().join("data").join("meka");
    std::fs::create_dir_all(&config_dir).expect("create config dir");

    let config_toml = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"
"#;
    std::fs::write(config_dir.join("config.toml"), config_toml).expect("write config.toml");

    // A partial answer streams, then the turn stalls in a 5s sleep that races cancellation. The
    // text after the sleep must never stream once cancel fires, and so must never be persisted.
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "partial answer before interrupt" },
            { "kind": "sleep", "ms": 5000 },
            { "kind": "text", "text": "TEXT-AFTER-INTERRUPT-MUST-NOT-PERSIST" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let script_path = temp.path().join("script.json");
    std::fs::write(&script_path, script.to_string()).expect("write script.json");

    // Round 1: prompt, wait for the partial to stream, fire cancel, capture sessionId, exit
    // cleanly.
    let session_id = {
        let mut child = meka_acp()
            .arg("acp")
            .env("MEKA_CONFIG_DIR", &config_dir)
            .env("MEKA_DATA_DIR", &data_dir)
            .env("HOME", temp.path())
            .env("MEKA_MOCK_PROVIDER", "1")
            .env("MEKA_MOCK_PROVIDER_SCRIPT", &script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn meka acp");
        let mut stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let stderr_pipe = child.stderr.take().expect("stderr");
        let mut reader = BufReader::new(stdout);
        let stderr_handle = std::thread::spawn(move || {
            let mut buf = String::new();
            let mut r = BufReader::new(stderr_pipe);
            while r.read_line(&mut buf).unwrap_or(0) > 0 {}
            buf
        });
        let deadline = Instant::now() + Duration::from_secs(15);

        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
        )
        .expect("initialize");
        let _ = read_until(&mut reader, deadline, |line| line.contains("\"id\":1"));

        let new_req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": { "cwd": config_dir.clone(), "mcpServers": [] }
        });
        writeln!(stdin, "{}", new_req).expect("session/new");
        let new_lines = read_until(&mut reader, deadline, |line| line.contains("\"id\":2"));
        let sid = serde_json::from_str::<serde_json::Value>(
            new_lines
                .iter()
                .find(|line| line.contains("\"id\":2"))
                .expect("session/new response"),
        )
        .expect("parse")["result"]["sessionId"]
            .as_str()
            .expect("sessionId")
            .to_string();

        let prompt_req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/prompt",
            "params": {
                "sessionId": sid,
                "prompt": [{ "type": "text", "text": "interrupt me mid-turn" }]
            }
        });
        writeln!(stdin, "{}", prompt_req).expect("session/prompt");

        // Wait until the partial answer has streamed so the agent has it buffered, then cancel.
        let start_deadline = Instant::now() + Duration::from_secs(5);
        let _ = read_until(&mut reader, start_deadline, |line| {
            line.contains("partial answer before interrupt")
        });
        let cancel_notif = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "session/cancel",
            "params": { "sessionId": sid }
        });
        writeln!(stdin, "{}", cancel_notif).expect("cancel");

        // The prompt should resolve (cancelled) well before the 5s sleep would finish.
        let response_deadline = Instant::now() + Duration::from_secs(3);
        let lines = read_until(&mut reader, response_deadline, |line| {
            line.contains("\"id\":3")
        });
        let response: serde_json::Value = serde_json::from_str(
            lines
                .iter()
                .find(|line| line.contains("\"id\":3"))
                .unwrap_or_else(|| {
                    panic!(
                        "no PromptResponse before deadline; cancel was likely starved.\nSTDERR:\n{}",
                        stderr_handle.join().unwrap_or_default(),
                    )
                }),
        )
        .expect("parse PromptResponse");
        assert_eq!(
            response["result"]["stopReason"], "cancelled",
            "prompt must resolve as cancelled; got: {}",
            response,
        );

        drop(stdin);
        let _ = child.kill();
        let _ = child.wait();
        sid
    };

    // Round 2: load the persisted session; the replay must carry the partial answer.
    let mut child = meka_acp()
        .arg("acp")
        .env("MEKA_CONFIG_DIR", &config_dir)
        .env("MEKA_DATA_DIR", &data_dir)
        .env("HOME", temp.path())
        .env("MEKA_MOCK_PROVIDER", "1")
        .env("MEKA_MOCK_PROVIDER_SCRIPT", &script_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn meka acp #2");
    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let stderr_pipe = child.stderr.take().expect("stderr");
    let mut reader = BufReader::new(stdout);
    let stderr_handle = std::thread::spawn(move || {
        let mut buf = String::new();
        let mut r = BufReader::new(stderr_pipe);
        while r.read_line(&mut buf).unwrap_or(0) > 0 {}
        buf
    });
    let deadline = Instant::now() + Duration::from_secs(15);

    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
    )
    .expect("initialize");
    let _ = read_until(&mut reader, deadline, |line| line.contains("\"id\":1"));

    let load_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "session/load",
        "params": {
            "sessionId": session_id,
            "cwd": config_dir.clone(),
            "mcpServers": []
        }
    });
    writeln!(stdin, "{}", load_req).expect("session/load");
    let load_lines = read_until(&mut reader, deadline, |line| line.contains("\"id\":4"));

    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();

    let replay = load_lines.join("");
    assert!(
        replay.contains("partial answer before interrupt"),
        "session/load replay must carry the interrupted turn's partial answer; stream:\n{}\nSTDERR:\n{}",
        replay,
        stderr_handle.join().unwrap_or_default(),
    );
    assert!(
        !replay.contains("TEXT-AFTER-INTERRUPT-MUST-NOT-PERSIST"),
        "post-interrupt text must not be persisted; stream:\n{}",
        replay,
    );
}

/// when the client advertises both `fs.readTextFile` and `fs.writeTextFile`, `edit_file` delegates
/// both halves and does not touch the local disk. Covers the previously-untested delegated
/// read+write composition.
#[test]
fn acp_edit_file_delegates_when_both_fs_capabilities_offered() {
    let config_toml = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"

[permissions]
default = "unrestricted"
enabled = ["read", "unrestricted"]
"#;
    // edit_file canonicalizes the target before reaching the delegation seam, so the path must
    // exist on disk. Seed it with a *different* content from what the delegate will serve; this
    // proves the editor's in-buffer view (via fs/read_text_file) wins over the on-disk bytes.
    // `force=true` skips the read-before-edit gate (we're not testing that path).
    let disk_content = "this should NOT be edited\n";
    let editor_content = "alpha\nbeta\n";
    let expected_new_content = "alpha\nGAMMA\n";

    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .capabilities(serde_json::json!({
            "fs": { "readTextFile": true, "writeTextFile": true }
        }))
        .pre_spawn(move |config_dir| {
            let target = config_dir.join("delegated-edit.txt");
            std::fs::write(&target, disk_content).expect("seed local file");
            serde_json::json!([
                [
                    { "kind": "text", "text": "editing..." },
                    { "kind": "tool_use_start", "id": "call_edit", "name": "edit_file" },
                    {
                        "kind": "tool_use_end",
                        "input": {
                            "path": target.to_str().unwrap(),
                            "old_string": "beta",
                            "new_string": "GAMMA",
                            "force": true
                        }
                    },
                    { "kind": "message_end", "stop_reason": "tool_use" }
                ],
                [
                    { "kind": "text", "text": "done" },
                    { "kind": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .build();
    let target = harness.config_dir().join("delegated-edit.txt");
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "edit it");

    let mut saw_fs_read = false;
    let mut saw_fs_write = false;
    let mut delegated_content: Option<String> = None;
    let _ = harness.await_response_with_dispatch(id, |value| match value["method"].as_str() {
        Some("fs/read_text_file") => {
            saw_fs_read = true;
            Some(serde_json::json!({
                "jsonrpc": "2.0",
                "id": value["id"].clone(),
                "result": { "content": editor_content }
            }))
        }
        Some("fs/write_text_file") => {
            saw_fs_write = true;
            delegated_content = value["params"]["content"].as_str().map(String::from);
            Some(serde_json::json!({
                "jsonrpc": "2.0",
                "id": value["id"].clone(),
                "result": {}
            }))
        }
        _ => None,
    });

    assert!(saw_fs_read, "edit_file must delegate the read half");
    assert!(saw_fs_write, "edit_file must delegate the write half");
    assert_eq!(
        delegated_content.as_deref(),
        Some(expected_new_content),
        "delegated write content didn't reflect the edit \
         applied to the editor's in-buffer view"
    );
    // On-disk content must be untouched; delegation bypassed the local filesystem.
    let on_disk = std::fs::read_to_string(&target).expect("read seeded file");
    assert_eq!(
        on_disk, disk_content,
        "edit_file modified the local file despite delegating to fs/write_text_file"
    );
}

/// a sub-agent's permission prompt must forward through `PermissionForwardingFrontend` to the
/// parent's ACP connection. The parent triggers `agent_spawn`; the sub-agent runs in `ask` mode
/// (inherited) and attempts `write_file`, which fires a `session/request_permission` on the
/// *parent's* connection. Test answers `allow_once` and asserts the request was observed.
#[test]
fn acp_subagent_permission_forwards_to_parent_client() {
    let config_toml = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"

[permissions]
default = "ask"
enabled = ["read", "ask", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(|config_dir| {
            let target = config_dir.join("subagent-write.txt");
            // The mock provider is shared between parent and sub-agent (same `Arc<dyn Provider>`),
            // so rounds drain in the order they're consumed: parent → sub-agent → sub-agent →
            // parent.
            serde_json::json!([
                // Parent round 1: spawn the sub-agent.
                [
                    { "kind": "text", "text": "spawning sub-agent..." },
                    { "kind": "tool_use_start", "id": "call_spawn", "name": "agent_spawn" },
                    {
                        "kind": "tool_use_end",
                        "input": { "prompt": "write the file" }
                    },
                    { "kind": "message_end", "stop_reason": "tool_use" }
                ],
                // Sub-agent round 1: write_file → triggers permission.
                [
                    { "kind": "text", "text": "writing..." },
                    { "kind": "tool_use_start", "id": "call_write", "name": "write_file" },
                    {
                        "kind": "tool_use_end",
                        "input": { "path": target.to_str().unwrap(), "content": "subagent wrote me" }
                    },
                    { "kind": "message_end", "stop_reason": "tool_use" }
                ],
                // Sub-agent round 2: final report.
                [
                    { "kind": "text", "text": "wrote the file" },
                    { "kind": "message_end", "stop_reason": "end_turn" }
                ],
                // Parent round 2: final report.
                [
                    { "kind": "text", "text": "sub-agent finished" },
                    { "kind": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .deadline(Duration::from_secs(30))
        .build();
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "delegate the write");

    let mut saw_permission_request = false;
    let (_updates, _response) =
        harness.collect_updates_with_dispatch(&sid, id, |value| match value["method"].as_str() {
            Some("session/request_permission") => {
                saw_permission_request = true;
                Some(serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": value["id"].clone(),
                    "result": {
                        "outcome": { "outcome": "selected", "optionId": "allow_once" }
                    }
                }))
            }
            _ => None,
        });

    assert!(
        saw_permission_request,
        "sub-agent's write_file must forward a session/request_permission \
         through the parent connection",
    );
}

/// `session/list` paginates with an opaque cursor. Seed PAGE_SIZE + a few sessions in the same cwd;
/// first call must return a `nextCursor`; passing that cursor back must return the remaining rows;
/// both pages combined must equal the seeded set.
#[test]
fn acp_session_list_paginates_across_cursor_boundary() {
    let temp = tempfile::tempdir().expect("tempdir");
    let config_dir = temp.path().join("meka");
    let data_dir = temp.path().join("data").join("meka");
    std::fs::create_dir_all(&config_dir).expect("create config dir");

    let config_toml = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"
"#;
    std::fs::write(config_dir.join("config.toml"), config_toml).expect("write config.toml");

    let cwd = temp.path().join("proj");
    std::fs::create_dir_all(&cwd).expect("mkdir cwd");

    let script = serde_json::json!([
        [
            { "kind": "text", "text": "ack" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let script_path = temp.path().join("script.json");
    std::fs::write(&script_path, script.to_string()).expect("write script");

    // `acp::handle_list_sessions` uses PAGE_SIZE = 50. Seed PAGE_SIZE + 3 sessions so the second
    // page is non-empty but small enough to keep the test fast.
    const PAGE_SIZE: usize = 50;
    const TOTAL: usize = PAGE_SIZE + 3;

    let create_one = || -> String {
        let mut child = meka_acp()
            .arg("acp")
            .env("MEKA_CONFIG_DIR", &config_dir)
            .env("MEKA_DATA_DIR", &data_dir)
            .env("HOME", temp.path())
            .env("MEKA_MOCK_PROVIDER", "1")
            .env("MEKA_MOCK_PROVIDER_SCRIPT", &script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn");
        let mut stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let stderr_pipe = child.stderr.take().expect("stderr");
        let mut reader = BufReader::new(stdout);
        let _stderr_handle = std::thread::spawn(move || {
            let mut buf = String::new();
            let mut r = BufReader::new(stderr_pipe);
            while r.read_line(&mut buf).unwrap_or(0) > 0 {}
            buf
        });
        let deadline = Instant::now() + Duration::from_secs(15);

        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
        )
        .expect("init");
        let _ = read_until(&mut reader, deadline, |line| line.contains("\"id\":1"));

        let new_req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": { "cwd": cwd.clone(), "mcpServers": [] }
        });
        writeln!(stdin, "{}", new_req).expect("session/new");
        let new_lines = read_until(&mut reader, deadline, |line| line.contains("\"id\":2"));
        let sid = serde_json::from_str::<serde_json::Value>(
            new_lines
                .iter()
                .find(|line| line.contains("\"id\":2"))
                .expect("session/new"),
        )
        .expect("parse")["result"]["sessionId"]
            .as_str()
            .expect("sessionId")
            .to_string();

        // One trivial prompt so the session has a row to surface.
        let prompt_req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/prompt",
            "params": {
                "sessionId": sid,
                "prompt": [{ "type": "text", "text": "ping" }]
            }
        });
        writeln!(stdin, "{}", prompt_req).expect("prompt");
        let _ = read_until(&mut reader, deadline, |line| line.contains("\"id\":3"));

        drop(stdin);
        let _ = child.kill();
        let _ = child.wait();
        sid
    };

    let mut seeded: std::collections::HashSet<String> = std::collections::HashSet::new();
    for _ in 0..TOTAL {
        seeded.insert(create_one());
    }
    assert_eq!(seeded.len(), TOTAL, "test seeded duplicate session ids");

    // Now drive two session/list calls. The first returns the first page + a cursor; the second
    // uses the cursor to fetch the remainder.
    let mut child = meka_acp()
        .arg("acp")
        .env("MEKA_CONFIG_DIR", &config_dir)
        .env("MEKA_DATA_DIR", &data_dir)
        .env("HOME", temp.path())
        .env("MEKA_MOCK_PROVIDER", "1")
        .env("MEKA_MOCK_PROVIDER_SCRIPT", &script_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn list child");
    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let stderr_pipe = child.stderr.take().expect("stderr");
    let mut reader = BufReader::new(stdout);
    let _stderr_handle = std::thread::spawn(move || {
        let mut buf = String::new();
        let mut r = BufReader::new(stderr_pipe);
        while r.read_line(&mut buf).unwrap_or(0) > 0 {}
        buf
    });
    let deadline = Instant::now() + Duration::from_secs(30);

    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
    )
    .expect("init");
    let _ = read_until(&mut reader, deadline, |line| line.contains("\"id\":1"));

    let list_req_a = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "session/list",
        "params": { "cwd": cwd.clone() }
    });
    writeln!(stdin, "{}", list_req_a).expect("list page 1");
    let lines_a = read_until(&mut reader, deadline, |line| line.contains("\"id\":5"));
    let line_a = lines_a
        .iter()
        .find(|line| line.contains("\"id\":5"))
        .expect("list page 1 response");
    let response_a: serde_json::Value = serde_json::from_str(line_a).expect("parse");
    let sessions_a = response_a["result"]["sessions"]
        .as_array()
        .expect("sessions array")
        .clone();
    let cursor = response_a["result"]["nextCursor"]
        .as_str()
        .expect("page 1 must carry a nextCursor for TOTAL > PAGE_SIZE")
        .to_string();
    assert_eq!(
        sessions_a.len(),
        PAGE_SIZE,
        "page 1 should be exactly PAGE_SIZE rows"
    );

    let list_req_b = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 6,
        "method": "session/list",
        "params": { "cwd": cwd.clone(), "cursor": cursor }
    });
    writeln!(stdin, "{}", list_req_b).expect("list page 2");
    let lines_b = read_until(&mut reader, deadline, |line| line.contains("\"id\":6"));
    let line_b = lines_b
        .iter()
        .find(|line| line.contains("\"id\":6"))
        .expect("list page 2 response");
    let response_b: serde_json::Value = serde_json::from_str(line_b).expect("parse");
    let sessions_b = response_b["result"]["sessions"]
        .as_array()
        .expect("sessions array")
        .clone();

    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();

    assert_eq!(
        sessions_b.len(),
        TOTAL - PAGE_SIZE,
        "page 2 should be the remaining rows"
    );

    // Both pages combined must equal the seeded set (no dropped rows, no duplicates between pages).
    let mut combined: std::collections::HashSet<String> = std::collections::HashSet::new();
    for entry in sessions_a.iter().chain(sessions_b.iter()) {
        if let Some(id) = entry["sessionId"].as_str() {
            combined.insert(id.to_string());
        }
    }
    assert_eq!(
        combined, seeded,
        "paginated set must equal seeded set; missing or duplicated rows"
    );
}

/// `allow_always` sticks: a subsequent `session/prompt` that hits the same write tool MUST NOT
/// trigger another `session/request_permission` round-trip. Regression guard for the sticky-allow
/// store. Two consecutive prompts in one session.
#[test]
fn acp_permission_allow_always_skips_second_prompt() {
    let config_toml = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"

[permissions]
default = "ask"
enabled = ["read", "ask", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(|config_dir| {
            let target_a = config_dir.join("a.txt");
            let target_b = config_dir.join("b.txt");
            // Two complete turns; both invoke write_file. Only the first should provoke a
            // permission round-trip.
            serde_json::json!([
                // Turn 1 round 1.
                [
                    { "kind": "text", "text": "writing a..." },
                    { "kind": "tool_use_start", "id": "call_a", "name": "write_file" },
                    {
                        "kind": "tool_use_end",
                        "input": { "path": target_a.to_str().unwrap(), "content": "a" }
                    },
                    { "kind": "message_end", "stop_reason": "tool_use" }
                ],
                // Turn 1 round 2.
                [
                    { "kind": "text", "text": "done a" },
                    { "kind": "message_end", "stop_reason": "end_turn" }
                ],
                // Turn 2 round 1.
                [
                    { "kind": "text", "text": "writing b..." },
                    { "kind": "tool_use_start", "id": "call_b", "name": "write_file" },
                    {
                        "kind": "tool_use_end",
                        "input": { "path": target_b.to_str().unwrap(), "content": "b" }
                    },
                    { "kind": "message_end", "stop_reason": "tool_use" }
                ],
                // Turn 2 round 2.
                [
                    { "kind": "text", "text": "done b" },
                    { "kind": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .deadline(Duration::from_secs(30))
        .build();
    let sid = harness.new_session();

    // Turn 1: write_file → request_permission (allow_always).
    let id_1 = harness.prompt(&sid, "write a");
    let mut prompts_for_turn_1 = 0_usize;
    let _ = harness.await_response_with_dispatch(id_1, |value| {
        if value["method"] == "session/request_permission" {
            prompts_for_turn_1 += 1;
            Some(serde_json::json!({
                "jsonrpc": "2.0",
                "id": value["id"].clone(),
                "result": {
                    "outcome": { "outcome": "selected", "optionId": "allow_always" }
                }
            }))
        } else {
            None
        }
    });

    // Turn 2: same tool; sticky allow must suppress the round-trip.
    let id_2 = harness.prompt(&sid, "write b");
    let mut prompts_for_turn_2 = 0_usize;
    let _ = harness.await_response_with_dispatch(id_2, |value| {
        if value["method"] == "session/request_permission" {
            prompts_for_turn_2 += 1;
            // Defensive answer just in case, but the assertion below catches the sticky-allow
            // regression.
            Some(serde_json::json!({
                "jsonrpc": "2.0",
                "id": value["id"].clone(),
                "result": {
                    "outcome": { "outcome": "selected", "optionId": "allow_once" }
                }
            }))
        } else {
            None
        }
    });

    assert_eq!(
        prompts_for_turn_1, 1,
        "turn 1 should have triggered exactly one permission round-trip",
    );
    assert_eq!(
        prompts_for_turn_2, 0,
        "turn 2 should have skipped permission entirely (allow_always sticky)",
    );
}

// `acp_initialize_negotiates_unknown_protocol_version` was superseded by
// `acp_initialize_clamps_far_future_version_to_latest` (below); stricter: asserts the clamp lands
// on `ProtocolVersion::LATEST`, not just "some number".

/// Two `session/new` calls succeed and produce independent session ids. Each session has its own
/// conversation; prompts route to the right one via `session/update`'s `sessionId` field.
#[test]
fn acp_multi_session_create_and_isolate_messages() {
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "A says hello" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "text", "text": "B says hello" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid_a = harness.new_session();
    let sid_b = harness.new_session();
    assert_ne!(sid_a, sid_b, "second session/new must mint a distinct id",);

    for (sid, expected_text) in [(&sid_a, "A says hello"), (&sid_b, "B says hello")] {
        let id = harness.prompt(sid, "go");
        let (updates, _) = harness.collect_updates(sid, id);
        let saw_correct_chunk = updates.iter().any(|u| {
            u["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
                && u["params"]["update"]["content"]["text"].as_str() == Some(expected_text)
        });
        assert!(
            saw_correct_chunk,
            "session {} did not receive its expected agent_message_chunk; updates: {:?}",
            sid, updates,
        );
    }
}

/// Two sessions prompting in parallel: A stalls in a long sleep, B completes a fast prompt. With
/// multi-session ACP, B's response must arrive *well before* A's, proving the per-session mutex
/// design lets sessions parallelise (the single-session `Mutex<ServerState>` would have serialised
/// them).
#[test]
fn acp_multi_session_parallel_prompts_dont_serialize() {
    let temp = tempfile::tempdir().expect("tempdir");
    let config_dir = temp.path().join("meka");
    let data_dir = temp.path().join("data").join("meka");
    std::fs::create_dir_all(&config_dir).expect("create config dir");

    let config_toml = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"
"#;
    std::fs::write(config_dir.join("config.toml"), config_toml).expect("write config.toml");

    // Windows CI workers have noticeably slower stdio IPC, so give A a longer stall and B a more
    // generous threshold while keeping the parallelism check (A:B ratio still ≥ 2:1).
    let (a_stall_ms, b_threshold) = if cfg!(target_os = "windows") {
        (10_000_u64, Duration::from_secs(5))
    } else {
        (4_000_u64, Duration::from_secs(2))
    };

    // Two rounds, in the order they'll be drained:
    //   1. Session A's prompt: long sleep then "A done".
    //   2. Session B's prompt: short response.
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "A starting" },
            { "kind": "sleep", "ms": a_stall_ms },
            { "kind": "text", "text": "A done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "text", "text": "B done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let script_path = temp.path().join("script.json");
    std::fs::write(&script_path, script.to_string()).expect("write script");

    let mut child = meka_acp()
        .arg("acp")
        .env("MEKA_CONFIG_DIR", &config_dir)
        .env("MEKA_DATA_DIR", &data_dir)
        .env("HOME", temp.path())
        .env("MEKA_MOCK_PROVIDER", "1")
        .env("MEKA_MOCK_PROVIDER_SCRIPT", &script_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let stderr_pipe = child.stderr.take().expect("stderr");
    let mut reader = BufReader::new(stdout);
    let stderr_handle = std::thread::spawn(move || {
        let mut buf = String::new();
        let mut r = BufReader::new(stderr_pipe);
        while r.read_line(&mut buf).unwrap_or(0) > 0 {}
        buf
    });
    let deadline = Instant::now() + Duration::from_millis(a_stall_ms + 10_000);

    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
    )
    .expect("init");
    let _ = read_until(&mut reader, deadline, |line| line.contains("\"id\":1"));

    // Open two sessions.
    let mut session_ids = Vec::new();
    for id in [2, 3] {
        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "session/new",
            "params": { "cwd": config_dir.clone(), "mcpServers": [] }
        });
        writeln!(stdin, "{}", req).expect("session/new");
        let needle = format!("\"id\":{}", id);
        let lines = read_until(&mut reader, deadline, |line| line.contains(&needle));
        let line = lines
            .iter()
            .find(|line| line.contains(&needle))
            .expect("session/new response");
        let response: serde_json::Value = serde_json::from_str(line).expect("parse");
        session_ids.push(
            response["result"]["sessionId"]
                .as_str()
                .expect("sessionId")
                .to_string(),
        );
    }

    // Fire prompt A (will stall in 4s sleep), then immediately prompt B (should complete fast).
    let test_start = Instant::now();
    let prompt_a = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 100,
        "method": "session/prompt",
        "params": {
            "sessionId": session_ids[0].clone(),
            "prompt": [{ "type": "text", "text": "go A" }]
        }
    });
    writeln!(stdin, "{}", prompt_a).expect("prompt A");

    // Wait for A's "A starting" delta to surface before firing B, so we know A holds the runtime
    // mutex and isn't merely queued. A blind `sleep(300ms)` was the previous approach, but it was
    // flake-prone on loaded CI; the deterministic marker mirrors what
    // `acp_session_cancel_interrupts_running_prompt` already does.
    let barrier = Instant::now() + Duration::from_secs(5);
    let _ = read_until(&mut reader, barrier, |line| line.contains("A starting"));

    let prompt_b = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 101,
        "method": "session/prompt",
        "params": {
            "sessionId": session_ids[1].clone(),
            "prompt": [{ "type": "text", "text": "go B" }]
        }
    });
    writeln!(stdin, "{}", prompt_b).expect("prompt B");

    // Read responses for both. Track when each id is observed.
    let mut a_finish: Option<Duration> = None;
    let mut b_finish: Option<Duration> = None;
    while a_finish.is_none() || b_finish.is_none() {
        if Instant::now() > deadline {
            break;
        }
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
            if value["id"] == 100 {
                a_finish = Some(test_start.elapsed());
            } else if value["id"] == 101 {
                b_finish = Some(test_start.elapsed());
            }
        }
    }

    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();

    let a = a_finish.expect("session A never responded");
    let b = b_finish.expect("session B never responded");

    // B must finish *substantially* before A. A is stalled, so B should return well within the
    // threshold. If the design serialised B behind A, B would take ≥ a_stall_ms.
    assert!(
        b < b_threshold,
        "session B took {:?} (threshold {:?}), looks serialised behind A's {}ms stall;\nSTDERR:\n{}",
        b,
        b_threshold,
        a_stall_ms,
        stderr_handle.join().unwrap_or_default(),
    );
    assert!(
        a > b,
        "session A finished before B ({:?} vs {:?}), script ordering wrong?",
        a,
        b,
    );
}

/// Per-session permission cells: setting mode on session A doesn't leak to session B. Sessions use
/// the same builtin permission space but their `SharedPermission` cells are independent
/// per-session, so a `session/set_mode` on A only affects A.
#[test]
fn acp_multi_session_set_mode_isolated() {
    const CONFIG: &str = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"

[permissions]
default = "read"
enabled = ["read", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::spawn(CONFIG, None);
    let sid_a = harness.new_session();
    let sid_b = harness.new_session();

    // Use `collect_updates` against sid_a to gather A's notifications during the set_mode
    // round-trip. Notifications for sid_b appear in the same stream; collect a second pass
    // afterwards by inspecting the raw transcript to be sure none leaked.
    let set_id = harness.send_request(
        "session/set_mode",
        serde_json::json!({ "sessionId": sid_a.clone(), "modeId": "unrestricted" }),
    );
    // Track session-id of every current_mode_update we observe by inline-collecting alongside the
    // response.
    let sid_a_owned = sid_a.clone();
    let sid_b_owned = sid_b.clone();
    let mut saw_a_update_on_a = false;
    let mut saw_a_update_on_b = false;
    let needle = format!("\"id\":{}", set_id);
    while Instant::now() < harness.deadline {
        let mut line = String::new();
        match harness.reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line)
            && value["method"] == "session/update"
            && value["params"]["update"]["sessionUpdate"] == "current_mode_update"
        {
            match value["params"]["sessionId"].as_str() {
                Some(s) if s == sid_a_owned => saw_a_update_on_a = true,
                Some(s) if s == sid_b_owned => saw_a_update_on_b = true,
                _ => {}
            }
        }
        if line.contains(&needle) && response_matches(&line, &needle) {
            break;
        }
    }
    assert!(
        saw_a_update_on_a,
        "session A must receive current_mode_update for its own set_mode",
    );
    assert!(
        !saw_a_update_on_b,
        "session B must NOT receive A's current_mode_update, modes are per-session",
    );
}

/// `session/cancel` fires only the target session's token. Session A stalls, B prompts normally. We
/// cancel A; A resolves with `cancelled` while B continues to `end_turn`.
#[test]
fn acp_multi_session_cancel_fires_only_target_session() {
    let script = serde_json::json!([
        // Session A: stall 5s. Cancel arrives before sleep ends → cancelled.
        [
            { "kind": "text", "text": "A stalling" },
            { "kind": "sleep", "ms": 5000 },
            { "kind": "text", "text": "A done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        // Session B: short response.
        [
            { "kind": "text", "text": "B done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = AcpTestHarnessBuilder::default()
        .config(ACP_INVALID_PARAMS_CONFIG)
        .script(script)
        .deadline(Duration::from_secs(20))
        .build();
    let sid_a = harness.new_session();
    let sid_b = harness.new_session();

    let id_a = harness.prompt(&sid_a, "stall");
    // Wait for A to actually start streaming before firing cancel.
    let start_deadline = Instant::now() + Duration::from_secs(3);
    let _ = read_until(&mut harness.reader, start_deadline, |line| {
        line.contains("A stalling")
    });
    harness.cancel(&sid_a);
    let id_b = harness.prompt(&sid_b, "go");

    // Poll for both responses arriving in any order.
    let mut a_stop: Option<String> = None;
    let mut b_stop: Option<String> = None;
    while a_stop.is_none() || b_stop.is_none() {
        if Instant::now() > harness.deadline {
            break;
        }
        let mut line = String::new();
        if harness.reader.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
            if value["id"].as_u64() == Some(id_a)
                && let Some(reason) = value["result"]["stopReason"].as_str()
            {
                a_stop = Some(reason.to_string());
            }
            if value["id"].as_u64() == Some(id_b)
                && let Some(reason) = value["result"]["stopReason"].as_str()
            {
                b_stop = Some(reason.to_string());
            }
        }
    }

    assert_eq!(
        a_stop.as_deref(),
        Some("cancelled"),
        "session A must resolve cancelled",
    );
    assert_eq!(
        b_stop.as_deref(),
        Some("end_turn"),
        "session B's cancel must NOT have fired, only A was cancelled",
    );
}

/// `session/close` arriving while a prompt is still running must:
///   1. cancel the in-flight prompt (it resolves with `cancelled`),
///   2. return success on the close request itself,
///   3. cause re-close on the same id to error (slot is gone),
///   4. cause subsequent `session/prompt` against the closed id to error.
///
/// The architecture relies on the sibling cancellation cell so close can fire the token *without*
/// contending on the runtime mutex the in-flight prompt holds. If that wiring regressed, this test
/// would hang past the deadline.
#[test]
fn acp_session_close_while_prompt_in_flight_cancels_and_rejects_followups() {
    // Single round: a starting chunk, a 5s sleep that close should race against, then never-reached
    // completion text. If close doesn't cancel, the test will see end_turn after the full sleep.
    let script = serde_json::json!([[
        { "kind": "text", "text": "starting..." },
        { "kind": "sleep", "ms": 5000 },
        { "kind": "text", "text": "done" },
        { "kind": "message_end", "stop_reason": "end_turn" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid = harness.new_session();

    // Fire the stalled prompt, then wait for it to actually start streaming so we know the turn is
    // parked in the 5s sleep.
    let prompt_id = harness.prompt(&sid, "stall");
    let start_deadline = Instant::now() + Duration::from_secs(5);
    let _ = read_until(&mut harness.reader, start_deadline, |line| {
        line.contains("starting...")
    });

    // Fire close. The sibling cancellation cell pattern means this never blocks on the runtime
    // mutex.
    let close_id = harness.send_request(
        "session/close",
        serde_json::json!({ "sessionId": sid.clone() }),
    );

    // Both prompt_id (prompt cancelled) and close_id (close ok) must arrive well before the 5s
    // sleep would have finished.
    let response_deadline = Instant::now() + Duration::from_secs(3);
    let mut prompt_stop_reason: Option<String> = None;
    let mut close_result_seen = false;
    while prompt_stop_reason.is_none() || !close_result_seen {
        if Instant::now() > response_deadline {
            break;
        }
        let mut line = String::new();
        if harness.reader.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
            if value["id"].as_u64() == Some(prompt_id)
                && let Some(reason) = value["result"]["stopReason"].as_str()
            {
                prompt_stop_reason = Some(reason.to_string());
            }
            if value["id"].as_u64() == Some(close_id) && value["result"].is_object() {
                close_result_seen = true;
            }
        }
    }

    assert_eq!(
        prompt_stop_reason.as_deref(),
        Some("cancelled"),
        "in-flight prompt must resolve cancelled when session is closed mid-turn",
    );
    assert!(
        close_result_seen,
        "close request itself must return success even while a prompt is in flight",
    );

    // Re-close: must error.
    let re_close = harness.close_session(&sid);
    assert!(
        re_close["error"].is_object(),
        "re-closing a closed session must error: {}",
        re_close,
    );

    // Prompt against the closed id: must error.
    let stale_prompt_id = harness.prompt(&sid, "ghost");
    let stale = harness.await_response(stale_prompt_id);
    assert!(
        stale["error"].is_object(),
        "prompting a closed session must error: {}",
        stale,
    );
}

// === Input-validation error-path tests ==============================
//
// Each handler that takes a `sessionId`, `modeId`, or rich `prompt` content array must reject
// malformed input with a JSON-RPC `InvalidParams` (`-32602`) error, not a generic `InternalError`
// (`-32603`). Use the `AcpTestHarness` helper so the boilerplate stays out of these tests' way.

const ACP_INVALID_PARAMS: i64 = -32602;

fn assert_invalid_params(response: &serde_json::Value, context: &str) {
    let error = response["error"]
        .as_object()
        .unwrap_or_else(|| panic!("{}: expected error response, got: {}", context, response));
    let code = error
        .get("code")
        .and_then(|c| c.as_i64())
        .unwrap_or_else(|| panic!("{}: error missing numeric code: {}", context, response));
    assert_eq!(
        code, ACP_INVALID_PARAMS,
        "{}: expected -32602 InvalidParams, got code {}: {}",
        context, code, response,
    );
}

const ACP_INVALID_PARAMS_CONFIG: &str = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"
"#;

/// `session/prompt` against an unknown `sessionId` must error with `InvalidParams`. An `audio`
/// content block likewise: meka accepts `text` / `resource_link` / `resource` / `image` (when
/// vision is on) but never `audio`, so it is a client contract violation.
#[test]
fn acp_session_prompt_rejects_unknown_session_and_audio_block() {
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, None);

    let unknown = harness.request(
        "session/prompt",
        serde_json::json!({
            "sessionId": "00000000-0000-0000-0000-000000000000",
            "prompt": [{ "type": "text", "text": "hi" }]
        }),
    );
    assert_invalid_params(&unknown, "prompt with unknown sessionId");

    // Open a real session and send an audio block (an unsupported content type); must yield
    // InvalidParams during content parsing, before any turn work.
    let sid = harness.new_session();
    let bad_block = harness.request(
        "session/prompt",
        serde_json::json!({
            "sessionId": sid,
            "prompt": [{
                "type": "audio",
                "data": "AAAA",
                "mimeType": "audio/wav"
            }]
        }),
    );
    assert_invalid_params(&bad_block, "prompt with audio content block");
}

/// With `vision = false` on the active profile, meka advertises `image: false` and rejects image
/// content blocks with `InvalidParams` (the rejection happens during parsing, before any turn).
#[test]
fn acp_session_prompt_rejects_image_when_vision_disabled() {
    const NO_VISION_CONFIG: &str = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"
vision = false
"#;
    let mut harness = AcpTestHarness::spawn(NO_VISION_CONFIG, None);

    let sid = harness.new_session();
    let rejected = harness.request(
        "session/prompt",
        serde_json::json!({
            "sessionId": sid,
            "prompt": [{
                "type": "image",
                "data": "AAAA",
                "mimeType": "image/png"
            }]
        }),
    );
    assert_invalid_params(&rejected, "image block with vision disabled");
}

/// `session/load` rejects malformed UUIDs and refuses to re-load a session that's already open
/// (closing first is the correct flow). Both arms must report `InvalidParams`, not generic
/// `InternalError`.
#[test]
fn acp_session_load_rejects_malformed_uuid_and_already_loaded() {
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, None);

    let bad_uuid = harness.request(
        "session/load",
        serde_json::json!({
            "sessionId": "not-a-uuid-at-all",
            "cwd": harness.config_dir().to_path_buf(),
            "mcpServers": []
        }),
    );
    assert_invalid_params(&bad_uuid, "load with malformed UUID");

    // Open a real session and immediately try to reload it.
    let sid = harness.new_session();
    let already = harness.request(
        "session/load",
        serde_json::json!({
            "sessionId": sid,
            "cwd": harness.config_dir().to_path_buf(),
            "mcpServers": []
        }),
    );
    assert_invalid_params(&already, "load already-open session");
}

/// `session/resume` rejects malformed UUIDs, unknown ids, and ids for sessions already occupying
/// the active slot, all client-side mistakes that map to `InvalidParams`. The already-loaded guard
/// at `src/acp.rs:1695` mirrors the `session/load` one tested in
/// `acp_session_load_rejects_malformed_uuid_and_already_loaded`.
#[test]
fn acp_session_resume_rejects_malformed_uuid_unknown_and_already_loaded() {
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, None);

    let bad_uuid = harness.request(
        "session/resume",
        serde_json::json!({
            "sessionId": "not-a-uuid",
            "cwd": harness.config_dir().to_path_buf(),
            "mcpServers": []
        }),
    );
    assert_invalid_params(&bad_uuid, "resume with malformed UUID");

    let unknown = harness.request(
        "session/resume",
        serde_json::json!({
            "sessionId": "00000000-0000-0000-0000-000000000000",
            "cwd": harness.config_dir().to_path_buf(),
            "mcpServers": []
        }),
    );
    assert_invalid_params(&unknown, "resume with unknown UUID");

    // Open a real session and immediately try to resume it. The session is already in the active
    // map, so the resume guard rejects with `InvalidParams`.
    let sid = harness.new_session();
    let already = harness.request(
        "session/resume",
        serde_json::json!({
            "sessionId": sid,
            "cwd": harness.config_dir().to_path_buf(),
            "mcpServers": []
        }),
    );
    assert_invalid_params(&already, "resume already-active session");
}

/// `session/set_mode` rejects an unknown mode id and rejects a valid-but-disabled mode (configured
/// `enabled` array doesn't list it). Both arms are input validation: `InvalidParams`.
#[test]
fn acp_session_set_mode_rejects_unknown_and_disabled() {
    let config = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"

[permissions]
default = "read"
enabled = ["read"]
"#;
    let mut harness = AcpTestHarness::spawn(config, None);
    let sid = harness.new_session();

    let unknown = harness.request(
        "session/set_mode",
        serde_json::json!({
            "sessionId": sid,
            "modeId": "definitely-not-a-mode"
        }),
    );
    assert_invalid_params(&unknown, "set_mode with unknown mode id");

    // `write` is a valid mode id (parse_mode_id succeeds) but it's not in the configured `enabled`
    // list, so try_set rejects it.
    let disabled = harness.request(
        "session/set_mode",
        serde_json::json!({
            "sessionId": sid,
            "modeId": "unrestricted"
        }),
    );
    assert_invalid_params(&disabled, "set_mode with disabled mode");
}

/// Tightens the existing protocol-version test: meka must clamp far-future versions to
/// `ProtocolVersion::LATEST` (currently V1), not echo the requested value verbatim. A naive echo
/// would let a future client think we support a version we haven't shipped.
#[test]
fn acp_initialize_clamps_far_future_version_to_latest() {
    let temp = tempfile::tempdir().expect("tempdir");
    let config_dir = temp.path().join("meka");
    let data_dir = temp.path().join("data").join("meka");
    std::fs::create_dir_all(&config_dir).expect("create config dir");
    std::fs::write(config_dir.join("config.toml"), ACP_INVALID_PARAMS_CONFIG)
        .expect("write config.toml");

    let mut child = meka_acp()
        .arg("acp")
        .env("MEKA_CONFIG_DIR", &config_dir)
        .env("MEKA_DATA_DIR", &data_dir)
        .env("HOME", temp.path())
        .env("MEKA_MOCK_PROVIDER", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let stderr_pipe = child.stderr.take().expect("stderr");
    let mut reader = BufReader::new(stdout);
    let stderr_handle = std::thread::spawn(move || {
        let mut buf = String::new();
        let mut r = BufReader::new(stderr_pipe);
        while r.read_line(&mut buf).unwrap_or(0) > 0 {}
        buf
    });
    let deadline = Instant::now() + Duration::from_secs(10);

    // Far-future version, well past anything the schema crate would ever produce. Must come back
    // clamped to LATEST (V1).
    let init_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": { "protocolVersion": 9999 }
    });
    writeln!(stdin, "{}", init_req).expect("init");
    let lines = read_until(&mut reader, deadline, |line| line.contains("\"id\":1"));

    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();

    let line = lines
        .iter()
        .find(|line| line.contains("\"id\":1"))
        .unwrap_or_else(|| {
            panic!(
                "no initialize response.\nSTDERR:\n{}\nstdout:\n{}",
                stderr_handle.join().unwrap_or_default(),
                lines.join(""),
            )
        });
    let response: serde_json::Value = serde_json::from_str(line).expect("parse init");
    let result = response["result"]
        .as_object()
        .unwrap_or_else(|| panic!("initialize must succeed, got: {}", response));
    let negotiated = result
        .get("protocolVersion")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| panic!("missing numeric protocolVersion in: {}", response));
    // LATEST is currently V1; assert ≤ 1 so the test stays valid if the SDK ever introduces a
    // stable V2.
    assert!(
        negotiated <= 1,
        "negotiated version must be clamped to ≤ LATEST (V1 today); got {}",
        negotiated,
    );
}

// === Mock-provider-driven coverage ==================================
//
// These tests round-trip features that the mock provider couldn't emit before (`ThinkingDelta` /
// `ThinkingComplete` and the `MaxTokens` stop reason) through the full ACP pipeline.

/// A `ThinkingDelta` + `ThinkingComplete` pair from the provider maps to a `session/update`
/// notification with `sessionUpdate: "agent_thought_chunk"` carrying the thinking text. The text
/// body is the only thing the editor needs; the `signature` field is opaque pass-through and not
/// currently surfaced in the notification.
#[test]
fn acp_session_prompt_emits_agent_thought_chunk_for_thinking_block() {
    let script = serde_json::json!([[
        { "kind": "thinking_delta", "text": "weighing options... " },
        { "kind": "thinking_delta", "text": "considering safety" },
        { "kind": "thinking_complete", "signature": null },
        { "kind": "text", "text": "ok" },
        { "kind": "message_end", "stop_reason": "end_turn" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "think first");
    let (updates, response) = harness.collect_updates(&sid, id);

    let thought = updates
        .iter()
        .find(|u| u["params"]["update"]["sessionUpdate"] == "agent_thought_chunk")
        .unwrap_or_else(|| panic!("missing agent_thought_chunk; updates: {:?}", updates));
    let text = thought["params"]["update"]["content"]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("thought chunk missing text body: {}", thought));
    assert!(
        text.contains("weighing options") || text.contains("considering safety"),
        "agent_thought_chunk text should carry the scripted thinking content; got: {}",
        text,
    );
    assert_eq!(response["result"]["stopReason"], "end_turn");
}

/// Under `redact-thinking`, the provider streams a `thinking_complete` carrying only a signature
/// (no `thinking_delta`). The agent must retain that block for replay continuity without surfacing
/// a spurious empty `agent_thought_chunk`: the turn completes normally and the only visible output
/// is the assistant text.
#[test]
fn acp_session_prompt_keeps_empty_thinking_with_signature_quietly() {
    let script = serde_json::json!([[
        { "kind": "thinking_complete", "signature": "OPAQUE_SIGNATURE_BLOB" },
        { "kind": "text", "text": "answer" },
        { "kind": "message_end", "stop_reason": "end_turn" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "go");
    let (updates, response) = harness.collect_updates(&sid, id);

    // No empty thought chunk: the redacted (text-less) thinking block stays off-screen.
    assert!(
        !updates
            .iter()
            .any(|u| u["params"]["update"]["sessionUpdate"] == "agent_thought_chunk"),
        "empty thinking block must not emit an agent_thought_chunk; updates: {:?}",
        updates,
    );
    let saw_answer = updates.iter().any(|u| {
        u["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
            && u["params"]["update"]["content"]["text"]
                .as_str()
                .is_some_and(|t| t.contains("answer"))
    });
    assert!(
        saw_answer,
        "assistant text should reach the client; got: {:?}",
        updates
    );
    assert_eq!(response["result"]["stopReason"], "end_turn");
}

/// A thinking-only round (no visible text) ending in `end_turn` must not end the turn silently. The
/// agent nudges once for a user-visible response and re-issues, so the scripted second round's text
/// reaches the client. Mirrors Claude Code's `query_thinking_only_response` recovery. Without the
/// nudge the first round's `end_turn` would end the turn and the second round would never run.
#[test]
fn acp_session_prompt_nudges_thinking_only_turn() {
    let script = serde_json::json!([
        [
            { "kind": "thinking_delta", "text": "pondering silently" },
            { "kind": "thinking_complete", "signature": null },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "text", "text": "recovered answer" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "go");
    let (updates, response) = harness.collect_updates(&sid, id);

    // The second round runs only if the nudge fired after the thinking-only first round.
    let saw_recovered = updates.iter().any(|u| {
        u["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
            && u["params"]["update"]["content"]["text"]
                .as_str()
                .is_some_and(|text| text.contains("recovered answer"))
    });
    assert!(
        saw_recovered,
        "thinking-only turn must nudge and surface the second round's text; updates: {:?}",
        updates,
    );
    assert_eq!(response["result"]["stopReason"], "end_turn");
}

/// `MockStopReason::MaxTokens` propagates end-to-end as `stopReason: "max_tokens"` on the
/// `PromptResponse`. The mock already had the enum variant; this test plugs the gap in integration
/// coverage.
#[test]
fn acp_session_prompt_max_tokens_stop_reason() {
    let script = serde_json::json!([[
        { "kind": "text", "text": "truncated mid-thought" },
        { "kind": "message_end", "stop_reason": "max_tokens" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "go");
    let response = harness.await_response(id);
    assert_eq!(
        response["result"]["stopReason"], "max_tokens",
        "MaxTokens script → stopReason='max_tokens'; got: {}",
        response,
    );
}

/// Per-session frontend routing: a thinking block scripted for session A produces an
/// `agent_thought_chunk` on A's notifications and *not* on B's. Regression guard against
/// cross-session leakage of `session/update` traffic. The mock drains rounds in FIFO order across
/// sessions, so we send the only round to session A first then drive B's prompt through an empty
/// round (no `thinking_*` events).
#[test]
fn acp_session_prompt_thought_chunk_routes_per_session() {
    let script = serde_json::json!([
        [
            { "kind": "thinking_delta", "text": "session A only" },
            { "kind": "thinking_complete", "signature": null },
            { "kind": "text", "text": "A response" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "text", "text": "B response" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid_a = harness.new_session();
    let sid_b = harness.new_session();

    let id_a = harness.prompt(&sid_a, "go A");
    let (updates_a, _) = harness.collect_updates(&sid_a, id_a);

    let id_b = harness.prompt(&sid_b, "go B");
    let (updates_b, _) = harness.collect_updates(&sid_b, id_b);

    assert!(
        updates_a
            .iter()
            .any(|u| u["params"]["update"]["sessionUpdate"] == "agent_thought_chunk"),
        "session A must observe its agent_thought_chunk; updates: {:?}",
        updates_a,
    );
    assert!(
        updates_b
            .iter()
            .all(|u| u["params"]["update"]["sessionUpdate"] != "agent_thought_chunk"),
        "session B must not see A's agent_thought_chunk; updates: {:?}",
        updates_b,
    );
}

/// Non-Interrupted `Agent::run_turn` errors must surface as a JSON-RPC `error` on the
/// `session/prompt` response (the `Err(error)` arm at `src/acp.rs:1514`). Scripted via the `Fail`
/// mock event so no real provider call is made; the agent loop's stream handler turns the provider
/// error into `MekaError::Provider`, `run_turn` propagates it, and the ACP handler maps it to
/// `internal_error`.
#[test]
fn acp_session_prompt_surfaces_provider_error_as_jsonrpc_error() {
    let script = serde_json::json!([[
        { "kind": "fail", "message": "scripted provider failure" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "go");
    let response = harness.await_response(id);
    assert!(
        response.get("error").is_some(),
        "non-Interrupted run_turn error must surface as JSON-RPC error; got: {}",
        response,
    );
    assert!(
        response["result"].is_null(),
        "JSON-RPC response carries either result or error, not both; got: {}",
        response,
    );
    // `agent_client_protocol::util::internal_error` sets the standard `"Internal error"` JSON-RPC
    // message and stuffs the explanatory string into `data`.
    let message = response["error"]["message"]
        .as_str()
        .unwrap_or_else(|| panic!("error.message must be a string: {}", response));
    assert_eq!(message, "Internal error");
    let code = response["error"]["code"]
        .as_i64()
        .unwrap_or_else(|| panic!("error.code must be an integer: {}", response));
    assert_eq!(code, -32603, "internal_error → JSON-RPC code -32603");
    let data = response["error"]["data"]
        .as_str()
        .unwrap_or_else(|| panic!("error.data must carry the detail string: {}", response));
    assert!(
        data.contains("meka turn failed"),
        "error.data should be the internal_error prefix; got: {}",
        data,
    );
    assert!(
        data.contains("scripted provider failure"),
        "error.data should propagate the underlying provider error text; got: {}",
        data,
    );
}

/// A transient failure (`FailRetryable`) on the first attempt must be retried automatically:
/// `Agent::run_streaming` re-invokes `provider.stream()`, consuming the mock's next scripted round,
/// and the turn completes successfully with exactly the second round's content — no duplication,
/// no error surfaced to the client.
#[test]
fn acp_session_prompt_retries_transient_provider_error_then_succeeds() {
    let script = serde_json::json!([
        [{ "kind": "fail_retryable", "message": "overloaded", "retry_after_secs": null }],
        [
            { "kind": "text", "text": "recovered" },
            { "kind": "message_end", "stop_reason": "end_turn" },
        ],
    ]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "go");
    let (updates, response) = harness.collect_updates(&sid, id);

    assert!(
        response.get("error").is_none(),
        "the retry must be invisible to the client; got error: {}",
        response,
    );

    let text_chunks: Vec<&str> = updates
        .iter()
        .filter(|u| u["params"]["update"]["sessionUpdate"] == "agent_message_chunk")
        .filter_map(|u| u["params"]["update"]["content"]["text"].as_str())
        .collect();
    assert_eq!(
        text_chunks,
        vec!["recovered"],
        "exactly one clean copy of the recovered round's text, no duplication from the failed \
         first attempt; updates: {:?}",
        updates,
    );
}

/// `FailRetryable` repeated past `MAX_PROVIDER_RETRIES` must exhaust the retry budget and surface
/// the failure to the client as a normal JSON-RPC error, exactly like a non-retryable failure does.
#[test]
fn acp_session_prompt_exhausts_retries_then_surfaces_error() {
    // MAX_PROVIDER_RETRIES (3) retries + the initial attempt = 4 total attempts, all failing.
    let script = serde_json::json!([
        [{ "kind": "fail_retryable", "message": "overloaded 1", "retry_after_secs": null }],
        [{ "kind": "fail_retryable", "message": "overloaded 2", "retry_after_secs": null }],
        [{ "kind": "fail_retryable", "message": "overloaded 3", "retry_after_secs": null }],
        [{ "kind": "fail_retryable", "message": "overloaded 4", "retry_after_secs": null }],
    ]);
    // The default backoff (1s + 2s + 4s = 7s of sleeping) plus process overhead is comfortably
    // under 15s, but give this one extra headroom since it's the slowest test in the suite by
    // design.
    let mut harness = AcpTestHarness::builder()
        .config(ACP_INVALID_PARAMS_CONFIG)
        .script(script)
        .deadline(std::time::Duration::from_secs(25))
        .build();
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "go");
    let response = harness.await_response(id);

    assert!(
        response.get("error").is_some(),
        "exhausted retries must surface as a JSON-RPC error; got: {}",
        response,
    );
    let data = response["error"]["data"]
        .as_str()
        .unwrap_or_else(|| panic!("error.data must carry the detail string: {}", response));
    // The last attempt's message is what propagates.
    assert!(
        data.contains("overloaded 4"),
        "error.data should carry the final attempt's message; got: {}",
        data,
    );
}

/// A non-retryable failure (`Fail`, mapping to a plain `MekaError::Provider`) must still fail
/// immediately with no backoff delay — a regression guard against over-broadening the new retry
/// classification to errors that were never meant to retry.
#[test]
fn acp_session_prompt_plain_fail_is_not_retried() {
    let script = serde_json::json!([[{ "kind": "fail", "message": "permanent failure" }]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "go");
    let started = std::time::Instant::now();
    let response = harness.await_response(id);
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "a non-retryable failure must not incur any backoff delay; took {:?}",
        started.elapsed(),
    );
    assert!(
        response.get("error").is_some(),
        "plain Fail must still surface as a JSON-RPC error; got: {}",
        response,
    );
    let data = response["error"]["data"]
        .as_str()
        .unwrap_or_else(|| panic!("error.data must carry the detail string: {}", response));
    assert!(
        data.contains("permanent failure"),
        "error.data should propagate the underlying provider error text; got: {}",
        data,
    );
}

/// Once the frontend has already shown text this attempt, a subsequent transient error in the
/// SAME round must NOT trigger a retry — retrying after the user has seen partial output would
/// duplicate/corrupt what's on screen. The mock's per-round-is-one-`stream()`-call model means a
/// `FailRetryable` after a `Text` event in the same round exercises exactly this: the driver sends
/// the text (setting `content_started`), then fails, and `run_streaming` must propagate the error
/// immediately instead of consuming a second round.
#[test]
fn acp_session_prompt_does_not_retry_once_content_shown() {
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "partial" },
            { "kind": "fail_retryable", "message": "overloaded mid-stream", "retry_after_secs": null },
        ],
        [
            { "kind": "text", "text": "should never be reached" },
            { "kind": "message_end", "stop_reason": "end_turn" },
        ],
    ]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "go");
    let (updates, response) = harness.collect_updates(&sid, id);

    assert!(
        response.get("error").is_some(),
        "a transient error after content was shown must surface immediately, not retry; got: {}",
        response,
    );
    let text_chunks: Vec<&str> = updates
        .iter()
        .filter(|u| u["params"]["update"]["sessionUpdate"] == "agent_message_chunk")
        .filter_map(|u| u["params"]["update"]["content"]["text"].as_str())
        .collect();
    assert_eq!(
        text_chunks,
        vec!["partial"],
        "only the pre-failure text must appear; the second round must never be consumed: {:?}",
        updates,
    );
}

/// `session/request_permission` failure marks the connection as disconnected so the agent loop
/// bails out promptly. A spec-conformant client always answers `Selected` / `Cancelled`, so any
/// `Err` from `block_task` (channel closed or peer JSON-RPC error) signals a broken/malformed
/// client. The agent denies the tool call, completes the current iteration, then short-circuits the
/// next iteration via `Frontend::client_disconnected()`; the turn resolves `cancelled`, not
/// `end_turn`.
#[test]
fn acp_session_prompt_request_permission_failure_marks_disconnect() {
    let config_toml = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"

[permissions]
default = "ask"
enabled = ["read", "ask", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(|config_dir| {
            let target = config_dir.join("would-write.txt");
            serde_json::json!([
                [
                    { "kind": "text", "text": "writing..." },
                    { "kind": "tool_use_start", "id": "call_write", "name": "write_file" },
                    {
                        "kind": "tool_use_end",
                        "input": { "path": target.to_str().unwrap(), "content": "hi" }
                    },
                    { "kind": "message_end", "stop_reason": "tool_use" }
                ],
                // Second round would emit "done" + end_turn, but the disconnect-mark must
                // short-circuit the loop before it streams. If the mark wires up correctly this
                // round is never drained.
                [
                    { "kind": "text", "text": "done" },
                    { "kind": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .build();
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "write it");

    let (_updates, response) =
        harness.collect_updates_with_dispatch(&sid, id, |value| match value["method"].as_str() {
            Some("session/request_permission") => Some(jsonrpc_error(
                value["id"].clone(),
                "synthetic client error response",
            )),
            _ => None,
        });

    assert_eq!(
        response["result"]["stopReason"], "cancelled",
        "permission-Err must mark disconnect → next loop iter short-circuits; got: {}",
        response,
    );
}

// === fs/read_text_file line + limit =================================

/// `read_file` asks the editor for the whole document even when the model passed `offset`/`limit`,
/// and windows what comes back locally.
///
/// Pushing the window down to `fs/read_text_file` looked tidier and cost two things. The freshness
/// stamp recorded the slice rather than the document, so the next `edit_file` compared a slice
/// against the whole buffer and refused with a false "changed in the editor". And a response of
/// exactly `limit` lines was indistinguishable from a file that ended there, so a truncated read
/// was handed to the model with no notice.
#[test]
fn acp_fs_read_text_file_fetches_the_whole_document_and_windows_locally() {
    let on_disk_marker = "DO-NOT-READ-ME-FROM-DISK\n".repeat(100);
    let on_disk_marker_for_seed = on_disk_marker.clone();
    let mut harness = AcpTestHarness::builder()
        .config(ACP_INVALID_PARAMS_CONFIG)
        .capabilities(serde_json::json!({
            "fs": { "readTextFile": true, "writeTextFile": false }
        }))
        .pre_spawn(move |config_dir| {
            let target = config_dir.join("delegated-line-limit.txt");
            std::fs::write(&target, &on_disk_marker_for_seed).expect("write target");
            serde_json::json!([
                [
                    { "kind": "text", "text": "reading partial..." },
                    { "kind": "tool_use_start", "id": "call_read", "name": "read_file" },
                    {
                        "kind": "tool_use_end",
                        "input": { "path": target.to_string_lossy(), "offset": 9, "limit": 50 }
                    },
                    { "kind": "message_end", "stop_reason": "tool_use" }
                ],
                [
                    { "kind": "text", "text": "done" },
                    { "kind": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .build();
    let target = harness.config_dir().join("delegated-line-limit.txt");
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "read partial");

    let mut saw_request = false;
    let mut observed_line = serde_json::Value::Null;
    let mut observed_limit = serde_json::Value::Null;
    let _ = harness.await_response_with_dispatch(id, |value| match value["method"].as_str() {
        Some("fs/read_text_file") => {
            saw_request = true;
            observed_line = value["params"]["line"].clone();
            observed_limit = value["params"]["limit"].clone();
            Some(serde_json::json!({
                "jsonrpc": "2.0",
                "id": value["id"].clone(),
                "result": { "content": "DELEGATED CONTENT FROM EDITOR" }
            }))
        }
        _ => None,
    });

    assert!(saw_request, "the read must still be delegated");
    assert!(
        observed_line.is_null(),
        "the window is applied locally, so no line is sent: {observed_line}",
    );
    assert!(
        observed_limit.is_null(),
        "the window is applied locally, so no limit is sent: {observed_limit}",
    );
    // On-disk file is untouched, proving the delegate path won.
    assert_eq!(
        std::fs::read_to_string(&target).expect("read on-disk"),
        on_disk_marker,
        "on-disk file content should be unchanged",
    );
}

// === V2 protocol conformance ========================================

/// `ContentBlock::ResourceLink` is part of the ACP baseline. meka used to reject it with
/// `InvalidParams`; the new behavior flattens the link into a tag the model can see.
#[test]
fn acp_session_prompt_accepts_resource_link_baseline() {
    let script = serde_json::json!([[
        { "kind": "text", "text": "ack" },
        { "kind": "message_end", "stop_reason": "end_turn" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid = harness.new_session();
    // session/prompt with a resource_link block; assert no error.
    let id = harness.send_request(
        "session/prompt",
        serde_json::json!({
            "sessionId": sid,
            "prompt": [
                { "type": "text", "text": "describe this:" },
                {
                    "type": "resource_link",
                    "name": "README.md",
                    "uri": "file:///tmp/README.md",
                    "description": "project readme"
                }
            ]
        }),
    );
    let response = harness.await_response(id);
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "resource_link content block must be accepted, not error: {}",
        response,
    );
}

/// An embedded `resource` block (an @-mention's inlined contents) is accepted and flattened into a
/// `<resource>` tag, not rejected. meka advertises `embeddedContext: true`.
#[test]
fn acp_session_prompt_accepts_embedded_resource() {
    let script = serde_json::json!([[
        { "kind": "text", "text": "ack" },
        { "kind": "message_end", "stop_reason": "end_turn" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid = harness.new_session();
    let id = harness.send_request(
        "session/prompt",
        serde_json::json!({
            "sessionId": sid,
            "prompt": [
                { "type": "text", "text": "summarize:" },
                {
                    "type": "resource",
                    "resource": {
                        "uri": "file:///tmp/notes.txt",
                        "text": "the meeting notes",
                        "mimeType": "text/plain"
                    }
                }
            ]
        }),
    );
    let response = harness.await_response(id);
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "embedded resource block must be accepted, not error: {}",
        response,
    );
}

/// An `image` content block is accepted when the profile has vision on (the default), and the turn
/// runs to completion.
#[test]
fn acp_session_prompt_accepts_image_with_vision() {
    let script = serde_json::json!([[
        { "kind": "text", "text": "i see it" },
        { "kind": "message_end", "stop_reason": "end_turn" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid = harness.new_session();
    // A 1x1 transparent PNG, base64-encoded.
    let png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
    let id = harness.send_request(
        "session/prompt",
        serde_json::json!({
            "sessionId": sid,
            "prompt": [
                { "type": "text", "text": "what is this?" },
                { "type": "image", "data": png_b64, "mimeType": "image/png" }
            ]
        }),
    );
    let response = harness.await_response(id);
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "image block must be accepted when vision is on: {}",
        response,
    );
}

/// `session/cancel` yields a `Cancelled` stop reason even when the cancellation manifests as
/// a non-`Interrupted` provider error. Script a `Sleep` followed by a `Fail`; fire cancel during
/// the sleep; assert `stopReason: cancelled` rather than the JSON-RPC error the `Fail` would
/// otherwise produce.
#[test]
fn acp_session_prompt_cancelled_after_provider_error() {
    let script = serde_json::json!([[
        { "kind": "text", "text": "starting..." },
        { "kind": "sleep", "ms": 5000 },
        { "kind": "fail", "message": "would-be internal error" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "go");

    // Wait for "starting..." to confirm the turn is parked in the sleep, then fire cancel.
    let barrier = Instant::now() + Duration::from_secs(3);
    let _ = read_until(&mut harness.reader, barrier, |line| {
        line.contains("starting...")
    });
    harness.cancel(&sid);
    let response = harness.await_response(id);
    assert_eq!(
        response["result"]["stopReason"], "cancelled",
        "post-cancel error must surface as Cancelled, not internal_error: {}",
        response,
    );
}

/// `session/cancel` between turns is latched and applied to the next prompt. Without the
/// latch, the cancel handler fires on the previous turn's already-dead token and the signal is
/// lost.
#[test]
fn acp_session_cancel_between_turns_applied_to_next_prompt() {
    // First turn: short response (completes immediately). Second turn: a sleep so the cancel can
    // take effect.
    let script = serde_json::json!([
        [
            { "kind": "text", "text": "first done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "kind": "text", "text": "second starting..." },
            { "kind": "sleep", "ms": 5000 },
            { "kind": "text", "text": "second done" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid = harness.new_session();

    // Drive the first turn to completion.
    let id_1 = harness.prompt(&sid, "first");
    let response_1 = harness.await_response(id_1);
    assert_eq!(response_1["result"]["stopReason"], "end_turn");

    // Cancel arrives before the second prompt. It must be latched.
    harness.cancel(&sid);

    // Second prompt: the latched cancel applies immediately, so the turn resolves Cancelled even
    // before the Sleep finishes.
    let id_2 = harness.prompt(&sid, "second");
    let response_2 = harness.await_response(id_2);
    assert_eq!(
        response_2["result"]["stopReason"], "cancelled",
        "between-turn cancel must be latched and applied to the next prompt: {}",
        response_2,
    );
}

/// `session/set_mode` no longer needs the runtime mutex. Mid-turn mode change takes effect
/// without waiting for the turn to finish.
#[test]
fn acp_session_set_mode_during_long_prompt_does_not_block() {
    let script = serde_json::json!([[
        { "kind": "text", "text": "running..." },
        { "kind": "sleep", "ms": 2000 },
        { "kind": "text", "text": "done" },
        { "kind": "message_end", "stop_reason": "end_turn" }
    ]]);
    const CONFIG: &str = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"

[permissions]
default = "read"
enabled = ["read", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::spawn(CONFIG, Some(script));
    let sid = harness.new_session();
    let prompt_id = harness.prompt(&sid, "go");

    // Wait for the turn to start streaming before firing set_mode.
    let barrier = Instant::now() + Duration::from_secs(3);
    let _ = read_until(&mut harness.reader, barrier, |line| {
        line.contains("running...")
    });

    // set_mode while the turn is mid-sleep must return promptly (well under the sleep's 2s).
    let start = Instant::now();
    let set_response = harness.set_mode(&sid, "unrestricted");
    let elapsed = start.elapsed();
    assert!(
        set_response["result"].is_object(),
        "set_mode must succeed mid-turn: {}",
        set_response,
    );
    assert!(
        elapsed < Duration::from_secs(1),
        "set_mode must not block on the runtime mutex; took {:?}",
        elapsed,
    );

    let prompt_response = harness.await_response(prompt_id);
    assert_eq!(prompt_response["result"]["stopReason"], "end_turn");
}

/// `session/cancel` during an `Ask`-mode permission prompt resolves the turn promptly.
/// Without the race against the cancellation token, the agent hangs inside `request_permission`
/// until the client answers.
#[test]
fn acp_session_request_permission_cancelled_by_session_cancel() {
    let config_toml = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"

[permissions]
default = "ask"
enabled = ["read", "ask", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(|config_dir| {
            let target = config_dir.join("doomed.txt");
            serde_json::json!([[
                { "kind": "text", "text": "writing..." },
                { "kind": "tool_use_start", "id": "call_w", "name": "write_file" },
                {
                    "kind": "tool_use_end",
                    "input": { "path": target.to_str().unwrap(), "content": "x" }
                },
                { "kind": "message_end", "stop_reason": "tool_use" }
            ]])
        })
        .build();
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "write");

    // Watch for the permission request to fire, then cancel without answering. The turn should
    // resolve `cancelled`.
    let sid_clone = sid.clone();
    let mut saw_permission = false;
    let mut cancel_fired = false;
    let needle = format!("\"id\":{}", id);
    let mut response: Option<serde_json::Value> = None;
    while Instant::now() < harness.deadline {
        let mut line = String::new();
        if harness.reader.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        if value["method"] == "session/request_permission" {
            saw_permission = true;
            if !cancel_fired {
                let cancel_notif = serde_json::json!({
                    "jsonrpc": "2.0",
                    "method": "session/cancel",
                    "params": { "sessionId": sid_clone }
                });
                writeln!(harness.stdin, "{}", cancel_notif).expect("write cancel");
                cancel_fired = true;
            }
        }
        if line.contains(&needle) && response_matches(&line, &needle) {
            response = Some(value);
            break;
        }
    }
    let response = response.unwrap_or_else(|| panic!("no prompt response"));
    assert!(saw_permission, "permission request must have fired");
    assert_eq!(
        response["result"]["stopReason"], "cancelled",
        "cancel during request_permission must resolve as Cancelled: {}",
        response,
    );
}

/// `protocolVersion: 0` is the schema's parse-failure sentinel and is rejected with
/// `InvalidParams`, not silently clamped.
#[test]
fn acp_initialize_rejects_protocol_version_zero() {
    let temp = tempfile::tempdir().expect("tempdir");
    let config_dir = temp.path().join("meka");
    let data_dir = temp.path().join("data").join("meka");
    std::fs::create_dir_all(&config_dir).expect("create config dir");
    std::fs::write(config_dir.join("config.toml"), ACP_INVALID_PARAMS_CONFIG)
        .expect("write config.toml");

    let mut child = meka_acp()
        .arg("acp")
        .env("MEKA_CONFIG_DIR", &config_dir)
        .env("MEKA_DATA_DIR", &data_dir)
        .env("HOME", temp.path())
        .env("MEKA_MOCK_PROVIDER", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let stderr_pipe = child.stderr.take().expect("stderr");
    let mut reader = BufReader::new(stdout);
    let _stderr_handle = std::thread::spawn(move || {
        let mut buf = String::new();
        let mut r = BufReader::new(stderr_pipe);
        while r.read_line(&mut buf).unwrap_or(0) > 0 {}
        buf
    });
    let deadline = Instant::now() + Duration::from_secs(10);

    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":0}}}}"#,
    )
    .expect("init");
    let lines = read_until(&mut reader, deadline, |line| line.contains("\"id\":1"));
    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();

    let line = lines
        .iter()
        .find(|line| line.contains("\"id\":1"))
        .expect("init response");
    let response: serde_json::Value = serde_json::from_str(line).expect("parse");
    assert_eq!(
        response["error"]["code"].as_i64(),
        Some(-32602),
        "protocolVersion 0 must be rejected with InvalidParams; got: {}",
        response,
    );
}

/// Concurrent same-session prompt rejection. Two `session/prompt`s on the same `sessionId`: the
/// first stalls, the second must return `InvalidParams "session already has a prompt in flight"`
/// while the first still resolves normally.
#[test]
fn acp_session_prompt_rejects_concurrent_prompt_same_session() {
    let script = serde_json::json!([[
        { "kind": "text", "text": "stalling" },
        { "kind": "sleep", "ms": 2000 },
        { "kind": "text", "text": "done" },
        { "kind": "message_end", "stop_reason": "end_turn" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid = harness.new_session();
    let id_a = harness.prompt(&sid, "first");

    // Wait until A is mid-sleep, then fire B.
    let barrier = Instant::now() + Duration::from_secs(3);
    let _ = read_until(&mut harness.reader, barrier, |line| {
        line.contains("stalling")
    });

    let id_b = harness.prompt(&sid, "second");
    let response_b = harness.await_response(id_b);
    assert_invalid_params(&response_b, "second concurrent prompt");

    // A still completes normally.
    let response_a = harness.await_response(id_a);
    assert_eq!(response_a["result"]["stopReason"], "end_turn");
}

/// Refusal stop reason: Claude `stop_reason: "refusal"` is mapped to `MockStopReason::Refusal` and
/// surfaces as the spec's `refusal` stop reason in the response. Mock needs the variant; add it.
#[test]
fn acp_session_prompt_refusal_stop_reason() {
    let script = serde_json::json!([[
        { "kind": "text", "text": "I cannot help with that." },
        { "kind": "message_end", "stop_reason": "refusal" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "do something disallowed");
    let response = harness.await_response(id);
    assert_eq!(
        response["result"]["stopReason"], "refusal",
        "refusal stop_reason must surface as ACP `refusal`: {}",
        response,
    );
}

/// An *empty* refusal: Claude streams `stop_reason: "refusal"` with no body (as fable-5 does after
/// a search surfaces disallowed content). meka must still surface a visible stand-in message
/// instead of a blank turn. Regression for session fad3ed41, where the turn rendered nothing and
/// persisted an empty `[]` assistant message.
#[test]
fn acp_empty_refusal_surfaces_standin_message() {
    let script = serde_json::json!([[
        { "kind": "message_end", "stop_reason": "refusal" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "trigger an empty refusal");
    let (updates, response) = harness.collect_updates(&sid, id);
    assert_eq!(
        response["result"]["stopReason"], "refusal",
        "empty refusal must still surface as ACP `refusal`: {}",
        response,
    );
    let dump = format!("{:?}", updates);
    assert!(
        dump.contains("declined to respond"),
        "empty refusal must surface a stand-in agent_message_chunk; updates: {}",
        dump,
    );
}

/// Regression: tool calls must run off the *presence* of `tool_use` blocks, not the reported stop
/// reason. Providers mislabel it - OpenAI Codex reports `completed` for a tool turn, and Claude
/// occasionally reports `end_turn` with `tool_use` present. Here the mock emits a complete
/// `read_file` call but ends the turn with `stop_reason: "end_turn"`; meka must still execute the
/// tool (and the turn completes normally) instead of orphaning the call and breaking the next
/// request.
#[test]
fn acp_tool_calls_execute_despite_non_tool_use_stop_reason() {
    let config_toml = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(|config_dir| {
            let target = config_dir.join("target.txt");
            std::fs::write(&target, "hello from mock test\n").expect("write target");
            serde_json::json!([
                [
                    { "kind": "text", "text": "reading the file...\n" },
                    { "kind": "tool_use_start", "id": "call_1", "name": "read_file" },
                    { "kind": "tool_use_end", "input": { "path": target.to_str().unwrap() } },
                    // The bug condition: a complete tool call, but the stop reason is NOT "tool_use".
                    { "kind": "message_end", "stop_reason": "end_turn" }
                ],
                [
                    { "kind": "text", "text": "done!" },
                    { "kind": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .build();
    let sid = harness.new_session();
    let id = harness.prompt(&sid, "read the target file");
    let (updates, response) = harness.collect_updates(&sid, id);

    // The tool must have executed despite the end_turn stop reason.
    assert!(
        updates.iter().any(|value| {
            let update = &value["params"]["update"];
            update["sessionUpdate"] == "tool_call_update" && update["status"] == "completed"
        }),
        "tool must execute even with a non-tool_use stop reason; updates: {:?}",
        updates,
    );
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "turn should complete normally after the tool round; full response: {}",
        response,
    );
}

/// The capability is the switch that makes a multi-root client send anything at all: Zed reads
/// `sessionCapabilities.additionalDirectories` and, when it's absent, silently drops every
/// workspace folder but the first before it ever reaches meka.
#[test]
fn acp_advertises_additional_directories_capability() {
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, None);
    let response = harness.request("initialize", serde_json::json!({ "protocolVersion": 1 }));
    assert!(
        response["result"]["agentCapabilities"]["sessionCapabilities"]["additionalDirectories"]
            .is_object(),
        "expected additionalDirectories to be advertised; got: {}",
        response,
    );
}

/// A session created with extra roots reports them back on `session/list`, which is what a client
/// rebuilds its workspace from when picking a session out of history.
#[test]
fn acp_session_new_accepts_and_reports_additional_directories() {
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, None);
    harness.request("initialize", serde_json::json!({ "protocolVersion": 1 }));

    let cwd = harness.config_dir.clone();
    let extra = cwd.join("shared");
    std::fs::create_dir_all(&extra).expect("mkdir extra root");

    let response = harness.request(
        "session/new",
        serde_json::json!({
            "cwd": cwd,
            "mcpServers": [],
            "additionalDirectories": [extra],
        }),
    );
    let session_id = response["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("session/new failed: {}", response))
        .to_string();

    let listed = harness.request("session/list", serde_json::json!({}));
    let sessions = listed["result"]["sessions"]
        .as_array()
        .unwrap_or_else(|| panic!("session/list failed: {}", listed));
    let row = sessions
        .iter()
        .find(|row| row["sessionId"] == session_id.as_str())
        .unwrap_or_else(|| panic!("session not listed: {}", listed));
    assert_eq!(
        row["additionalDirectories"],
        serde_json::json!([extra]),
        "session/list must report the roots the session was opened with; got: {}",
        row,
    );
}

/// The spec requires absolute paths, and meka has no defensible base to resolve a relative one
/// against: joining to `cwd` would invent a root the client never named.
#[test]
fn acp_session_new_rejects_relative_additional_directory() {
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, None);
    harness.request("initialize", serde_json::json!({ "protocolVersion": 1 }));

    let response = harness.request(
        "session/new",
        serde_json::json!({
            "cwd": harness.config_dir.clone(),
            "mcpServers": [],
            "additionalDirectories": ["relative/path"],
        }),
    );
    assert_eq!(
        response["error"]["code"], -32602,
        "expected invalid_params; got: {}",
        response,
    );
}

#[test]
fn acp_advertises_fork_capability() {
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, None);
    let response = harness.request("initialize", serde_json::json!({ "protocolVersion": 1 }));
    assert!(
        response["result"]["agentCapabilities"]["sessionCapabilities"]["fork"].is_object(),
        "expected fork to be advertised; got: {}",
        response,
    );
}

/// `session/fork` mints a new session carrying the source's conversation, and leaves the source
/// open: both are addressable afterwards.
#[test]
fn acp_session_fork_creates_a_usable_copy() {
    let turn = [
        serde_json::json!({ "kind": "text", "text": "ok" }),
        serde_json::json!({ "kind": "message_end", "stop_reason": "end_turn" }),
    ];
    // One script entry per prompt: seed the source, then a turn on the fork, then one on the
    // source to prove forking left it usable.
    let script = serde_json::json!([turn, turn, turn]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let source_id = harness.new_session();
    let id = harness.prompt(&source_id, "hello");
    harness.collect_updates(&source_id, id);

    let forked = harness.request(
        "session/fork",
        serde_json::json!({
            "sessionId": source_id,
            "cwd": harness.config_dir.clone(),
            "mcpServers": [],
        }),
    );
    let fork_id = forked["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("session/fork failed: {}", forked))
        .to_string();
    assert_ne!(fork_id, source_id, "the fork is a distinct session");

    // The title is derived from the first user message, so matching titles mean the source's
    // conversation actually came across rather than the fork starting empty.
    let listed = harness.request("session/list", serde_json::json!({}));
    let sessions = listed["result"]["sessions"]
        .as_array()
        .unwrap_or_else(|| panic!("session/list failed: {}", listed));
    let title_of = |wanted: &str| -> serde_json::Value {
        sessions
            .iter()
            .find(|row| row["sessionId"] == wanted)
            .unwrap_or_else(|| panic!("session {} not listed: {}", wanted, listed))["title"]
            .clone()
    };
    assert_eq!(title_of(&fork_id), serde_json::json!("hello"));
    assert_eq!(title_of(&fork_id), title_of(&source_id));

    // The copy is immediately promptable, which is only true if its conversation loaded cleanly.
    let id = harness.prompt(&fork_id, "again");
    let (_updates, response) = harness.collect_updates(&fork_id, id);
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "the forked session must accept a prompt; got: {}",
        response,
    );

    // Forking does not close the source.
    let id = harness.prompt(&source_id, "still here");
    let (_updates, response) = harness.collect_updates(&source_id, id);
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "forking must leave the source session usable; got: {}",
        response,
    );
}

/// ACP models fork as a session-*creation* request, so the workspace comes from the request rather
/// than the source. The source here has no extra roots, so the fork reporting one proves the
/// request's list was applied instead of inherited.
#[test]
fn acp_session_fork_applies_the_requests_additional_directories() {
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, None);
    harness.request("initialize", serde_json::json!({ "protocolVersion": 1 }));

    let cwd = harness.config_dir.clone();
    let extra = cwd.join("shared");
    std::fs::create_dir_all(&extra).expect("mkdir extra root");

    let response = harness.request(
        "session/new",
        serde_json::json!({ "cwd": cwd, "mcpServers": [] }),
    );
    let source_id = response["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("session/new failed: {}", response))
        .to_string();

    let forked = harness.request(
        "session/fork",
        serde_json::json!({
            "sessionId": source_id,
            "cwd": cwd,
            "mcpServers": [],
            "additionalDirectories": [extra],
        }),
    );
    let fork_id = forked["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("session/fork failed: {}", forked))
        .to_string();

    let listed = harness.request("session/list", serde_json::json!({}));
    let sessions = listed["result"]["sessions"]
        .as_array()
        .unwrap_or_else(|| panic!("session/list failed: {}", listed));
    let row_of = |wanted: &str| {
        sessions
            .iter()
            .find(|row| row["sessionId"] == wanted)
            .unwrap_or_else(|| panic!("session {} not listed: {}", wanted, listed))
    };
    assert_eq!(
        row_of(&fork_id)["additionalDirectories"],
        serde_json::json!([extra]),
        "the fork must carry the roots the request named; got: {}",
        row_of(&fork_id),
    );
    assert!(
        row_of(&source_id)["additionalDirectories"].is_null(),
        "and must not write them back onto the source; got: {}",
        row_of(&source_id),
    );
}

/// The realistic client flow: pick a session out of `session/list` and branch from it without
/// opening it first. Fork reads the database, so the source never has to be in the active map --
/// and forking must not implicitly adopt it either, or the client would be left holding a session
/// it never asked to open (and its on-disk lock).
#[test]
fn acp_session_fork_works_on_a_session_that_was_never_loaded() {
    let turn = [
        serde_json::json!({ "kind": "text", "text": "ok" }),
        serde_json::json!({ "kind": "message_end", "stop_reason": "end_turn" }),
    ];
    let script = serde_json::json!([turn, turn]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));

    // Seed a session, then close it so only its persisted row remains.
    let source_id = harness.new_session();
    let id = harness.prompt(&source_id, "seed");
    harness.collect_updates(&source_id, id);
    let closed = harness.request(
        "session/close",
        serde_json::json!({ "sessionId": source_id }),
    );
    assert!(
        closed["error"].is_null(),
        "session/close failed: {}",
        closed
    );

    let forked = harness.request(
        "session/fork",
        serde_json::json!({
            "sessionId": source_id,
            "cwd": harness.config_dir.clone(),
            "mcpServers": [],
        }),
    );
    let fork_id = forked["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("forking an unloaded session failed: {}", forked))
        .to_string();

    // The fork is live...
    let id = harness.prompt(&fork_id, "go");
    let (_updates, response) = harness.collect_updates(&fork_id, id);
    assert_eq!(response["result"]["stopReason"], "end_turn");

    // ...and the source stayed closed rather than being silently adopted.
    let reclose = harness.request(
        "session/close",
        serde_json::json!({ "sessionId": source_id }),
    );
    assert!(
        !reclose["error"].is_null(),
        "forking must not adopt the source session; got: {}",
        reclose,
    );
}

#[test]
fn acp_session_fork_rejects_bad_input() {
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, None);
    harness.request("initialize", serde_json::json!({ "protocolVersion": 1 }));

    let cwd = harness.config_dir.clone();
    let response = harness.request(
        "session/new",
        serde_json::json!({ "cwd": cwd, "mcpServers": [] }),
    );
    let source_id = response["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("session/new failed: {}", response))
        .to_string();

    let unknown = harness.request(
        "session/fork",
        serde_json::json!({
            "sessionId": "9f1d4c2e-0000-4000-8000-000000000000",
            "cwd": cwd,
            "mcpServers": [],
        }),
    );
    assert_eq!(
        unknown["error"]["code"], -32602,
        "an unknown source must be invalid_params; got: {}",
        unknown,
    );

    let relative_root = harness.request(
        "session/fork",
        serde_json::json!({
            "sessionId": source_id,
            "cwd": cwd,
            "mcpServers": [],
            "additionalDirectories": ["relative/path"],
        }),
    );
    assert_eq!(
        relative_root["error"]["code"], -32602,
        "expected invalid_params for a relative additional root; got: {}",
        relative_root,
    );
}

const ACP_SCHEDULE_CONFIG: &str = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"

[permissions]
default = "unrestricted"
enabled = ["read", "unrestricted"]

[schedule]
poll_interval = "1s"
"#;

/// A scheduled job fires in an ACP session and its turn reaches the editor unsolicited: no
/// `session/prompt` is outstanding when the notifications arrive.
///
/// This is the property the whole ACP host rests on. Zed's `handle_session_notification` dispatches
/// on session id with no in-flight-prompt check, so an agent-initiated turn renders; if that ever
/// stopped being true the feature would run invisibly, and this test is what would notice.
#[test]
fn acp_scheduled_job_fires_without_a_prompt() {
    let script = serde_json::json!([
        // Turn 1: the agent schedules a one-shot two seconds out.
        [
            { "kind": "tool_use_start", "id": "call_sched", "name": "schedule_create" },
            { "kind": "tool_use_end", "input": {
                "prompt": "ACP_DELIVERED_MARKER",
                "at": "2s"
            }},
            { "kind": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "kind": "text", "text": "scheduled" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ],
        // Turn 2 is the fire. Nothing on the client side asks for it.
        [
            { "kind": "text", "text": "ACP_SCHEDULED_REPLY" },
            { "kind": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = AcpTestHarness::builder()
        .config(ACP_SCHEDULE_CONFIG)
        .script(script)
        .deadline(Duration::from_secs(45))
        .build();

    let sid = harness.new_session();
    let id = harness.prompt(&sid, "remind me in two seconds");
    let (_updates, _response) = harness.collect_updates(&sid, id);

    // Nothing is outstanding now: the prompt above has been answered. Anything that arrives from
    // here is agent-initiated. The job is due in 2s and the scheduler ticks every 1s, so this waits
    // past both before asking meka for anything.
    std::thread::sleep(Duration::from_secs(6));
    let updates = harness.drain_unsolicited_updates(&sid);

    let text_of = |kind: &str| -> String {
        updates
            .iter()
            .map(|update| &update["params"]["update"])
            .filter(|update| update["sessionUpdate"] == kind)
            .filter_map(|update| update["content"]["text"].as_str())
            .collect::<Vec<_>>()
            .join("")
    };

    // The prompt is shown as a user message, so the transcript explains the reply rather than
    // presenting an answer to a question the editor never saw asked.
    let user_text = text_of("user_message_chunk");
    assert!(
        user_text.contains("ACP_DELIVERED_MARKER"),
        "the job's prompt must be pushed as a user message; updates were:\n{:#?}",
        updates,
    );
    assert!(
        user_text.contains("Scheduled job"),
        "and it must be marked as scheduled; updates were:\n{:#?}",
        updates,
    );

    let agent_text = text_of("agent_message_chunk");
    assert!(
        agent_text.contains("ACP_SCHEDULED_REPLY"),
        "the agent's reply to the scheduled turn must reach the client; updates were:\n{:#?}",
        updates,
    );
}

/// The same, through `session/resume` rather than `session/load`.
///
/// `handle_resume_session` carries its own copy of the permission restore, and nothing exercised
/// it: `a_mode_set_through_acp_survives_a_reload` covers the twin in `handle_load_session`, and
/// `acp_session_resume_adopts_without_replay` asserts nothing about the mode. Deleting the resume
/// path's block left the suite green -- and the result is a session running at the config default
/// while its row claims a higher level, which the scheduler's fire-time re-check then trusts. Fail
/// open, reached from the side that re-check cannot see.
#[test]
fn a_mode_set_through_acp_survives_a_resume() {
    const CONFIG: &str = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"

[permissions]
default = "read"
enabled = ["read", "ask", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::spawn(CONFIG, None);
    let cwd = harness.config_dir().to_path_buf();

    let new_response = harness.request(
        "session/new",
        serde_json::json!({ "cwd": cwd, "mcpServers": [] }),
    );
    assert_eq!(new_response["result"]["modes"]["currentModeId"], "read");
    let sid = new_response["result"]["sessionId"]
        .as_str()
        .expect("sessionId")
        .to_string();

    let set = harness.set_mode(&sid, "unrestricted");
    assert!(set["result"].is_object(), "set_mode must succeed: {}", set);

    let closed = harness.request("session/close", serde_json::json!({ "sessionId": sid }));
    assert!(
        closed["result"].is_object(),
        "close must succeed: {}",
        closed
    );

    let resumed = harness.request(
        "session/resume",
        serde_json::json!({ "sessionId": sid, "cwd": cwd, "mcpServers": [] }),
    );
    assert_eq!(
        resumed["result"]["modes"]["currentModeId"], "unrestricted",
        "the resumed session fell back to the process default instead of the mode its row \
         records: {}",
        resumed
    );
}

/// A mode set through `session/set_mode` survives a reload.
///
/// Two halves of one invariant. `set_mode` only moved the in-memory cell, so the session row kept
/// whatever it was created with; the scheduler's live gate re-check reads that row, which meant a
/// gate authored after cycling to `write` was refused forever and one authored before cycling down
/// to `read` kept firing. Persisting alone is not enough either: `build_session_runtime` seeds the
/// permission from process config, so a reloaded session would run at the default while its row
/// claimed the level the gate check would then trust -- the same fail-open from the other side.
///
/// Asserted through `currentModeId`, which is the client-visible form of the live cell, so this
/// fails if either the write or the restore is dropped.
#[test]
fn a_mode_set_through_acp_survives_a_reload() {
    const CONFIG: &str = r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"

[permissions]
default = "read"
enabled = ["read", "ask", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::spawn(CONFIG, None);
    let cwd = harness.config_dir().to_path_buf();

    let new_response = harness.request(
        "session/new",
        serde_json::json!({ "cwd": cwd, "mcpServers": [] }),
    );
    assert_eq!(
        new_response["result"]["modes"]["currentModeId"], "read",
        "a fresh session starts at the configured default"
    );
    let sid = new_response["result"]["sessionId"]
        .as_str()
        .expect("sessionId")
        .to_string();

    let set = harness.set_mode(&sid, "unrestricted");
    assert!(set["result"].is_object(), "set_mode must succeed: {}", set);

    // Drop it from the live map so the reload rebuilds from the row rather than reusing the entry.
    let closed = harness.request("session/close", serde_json::json!({ "sessionId": sid }));
    assert!(
        closed["result"].is_object(),
        "session/close must succeed: {}",
        closed
    );

    let loaded = harness.request(
        "session/load",
        serde_json::json!({ "sessionId": sid, "cwd": cwd, "mcpServers": [] }),
    );
    assert_eq!(
        loaded["result"]["modes"]["currentModeId"], "unrestricted",
        "the reloaded session fell back to the process default instead of the mode it was set to: \
         {}",
        loaded
    );
}
