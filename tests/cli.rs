// See the matching allow in `tests/acp.rs` for the rationale: integration tests panic on failure
// by design.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! End-to-end CLI smoke tests. These shell out to the built `meka` binary
//! (`env!("CARGO_BIN_EXE_meka")`) so they exercise the same entry point users hit on the command
//! line. They cover surface-level invariants that unit tests can't reach: argument-parser wiring,
//! `--help` output, and the exit status of trivial subcommands.

use std::process::Command;

fn meka() -> Command {
    Command::new(env!("CARGO_BIN_EXE_meka"))
}

#[test]
fn version_flag_prints_version_and_exits_zero() {
    let output = meka()
        .arg("--version")
        .output()
        .expect("failed to spawn meka");
    assert!(
        output.status.success(),
        "meka --version exited non-zero: {:?}",
        output.status
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.starts_with("meka "),
        "expected version output to start with 'meka ', got: {}",
        stdout
    );
}

#[test]
fn help_flag_lists_subcommands() {
    let output = meka().arg("--help").output().expect("failed to spawn meka");
    assert!(output.status.success(), "meka --help exited non-zero");
    let stdout = String::from_utf8_lossy(&output.stdout);
    for expected in ["provider", "session", "history", "mcp", "acp"] {
        assert!(
            stdout.contains(expected),
            "--help output missing subcommand '{}':\n{}",
            expected,
            stdout
        );
    }
}

#[test]
fn session_subcommand_help_lists_actions() {
    let output = meka()
        .args(["session", "--help"])
        .output()
        .expect("failed to spawn meka");
    assert!(
        output.status.success(),
        "meka session --help exited non-zero"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    for expected in ["list", "export", "delete"] {
        assert!(
            stdout.contains(expected),
            "session --help missing action '{}':\n{}",
            expected,
            stdout
        );
    }
}

#[test]
fn history_subcommand_help_lists_actions() {
    let output = meka()
        .args(["history", "--help"])
        .output()
        .expect("failed to spawn meka");
    assert!(
        output.status.success(),
        "meka history --help exited non-zero"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    for expected in ["list", "clear"] {
        assert!(
            stdout.contains(expected),
            "history --help missing action '{}':\n{}",
            expected,
            stdout
        );
    }
}

#[test]
fn acp_subcommand_help_describes_protocol() {
    // Verifies the `acp` subcommand is wired up. Full JSON-RPC handshake coverage lives in
    // `tests/acp.rs` against the mock-provider build; this smoke test stops at `--help`.
    let output = meka()
        .args(["acp", "--help"])
        .output()
        .expect("failed to spawn meka acp --help");
    assert!(
        output.status.success(),
        "meka acp --help exited non-zero: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("ACP")
            || stdout.contains("Agent Client Protocol")
            || stdout.contains("stdio"),
        "meka acp --help should mention the protocol or transport:\n{}",
        stdout,
    );
}

#[test]
fn unknown_subcommand_exits_nonzero() {
    let output = meka()
        .arg("--definitely-not-a-flag")
        .output()
        .expect("failed to spawn meka");
    assert!(
        !output.status.success(),
        "meka accepted an unknown flag without erroring"
    );
}

/// Run `meka` with an isolated config + data directory so host state (e.g.
/// `~/.config/meka/config.toml`) doesn't leak in, and the test's writes don't spill out. Sets
/// `MEKA_CONFIG_DIR` and `MEKA_DATA_DIR`, the only env vars that work on every platform
/// (`dirs::config_dir()` and `dirs::data_dir()` ignore `XDG_*` on macOS/Windows). Without the
/// data-dir override, parallel CLI tests collide on a shared `%APPDATA%/meka/sessions.db` on
/// Windows and hit SQLite lock contention.
fn run_isolated(dir: &std::path::Path, args: &[&str]) -> std::process::Output {
    meka()
        .args(args)
        .env("MEKA_CONFIG_DIR", dir.join("meka"))
        .env("MEKA_DATA_DIR", dir.join("data").join("meka"))
        .env("XDG_CONFIG_HOME", dir)
        .env("HOME", dir)
        .env("XDG_DATA_HOME", dir.join("data"))
        .output()
        .unwrap_or_else(|err| panic!("failed to spawn meka {:?}: {}", args, err))
}

/// `Conversation::rewind(0)` returns `None` unconditionally, so without an explicit guard the
/// caller reports it as the session having "fewer than 0 turn(s)". Rejected before the session is
/// even looked up, which is why a nonexistent id still produces the argument error. The HTTP
/// surface already answers 422 here (`rewind_rejects_zero_turns` in `tests/serve.rs`); this keeps
/// the CLI in step.
#[test]
fn session_rewind_rejects_zero_turns_without_describing_the_conversation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let id = "00000000-0000-4000-8000-000000000000";
    let output = run_isolated(dir.path(), &["session", "rewind", id, "-n", "0"]);
    assert!(
        !output.status.success(),
        "-n 0 must fail, got: {:?}",
        output.status
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("-n must be 1 or more"),
        "expected the argument to be blamed, got: {}",
        stderr
    );
    assert!(
        !stderr.contains("fewer than 0"),
        "must not describe the conversation as having fewer than 0 turns: {}",
        stderr
    );
}

#[test]
fn mcp_list_with_empty_config_prints_no_servers_and_exits_zero() {
    // Isolate the config dir so the host's real `~/.config/meka` doesn't leak into the test.
    // `MEKA_CONFIG_DIR` is the only env var that works on every platform (see `run_isolated` for
    // details).
    let dir = tempfile::tempdir().expect("tempdir");
    let output = run_isolated(dir.path(), &["mcp", "list"]);
    assert!(
        output.status.success(),
        "meka mcp list exited non-zero: {:?}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    // The empty case is a status note, not the data a script asked for, so it goes to stderr and
    // stdout stays clean enough to pipe.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("No MCP servers configured."),
        "expected 'No MCP servers configured.' on stderr, got: {}",
        stderr
    );
    assert!(
        stdout.trim().is_empty(),
        "stdout must carry no placeholder row, got: {}",
        stdout
    );
}

#[test]
fn mcp_add_http_positional_url_persists_server() {
    // Notion-style happy path: positional URL, transport auto-detected from the URL scheme, no
    // --url flag required. `--no-login` keeps the test hermetic; we just want to confirm `add`
    // wrote the entry, not that we can drive an end-to-end OAuth flow.
    let dir = tempfile::tempdir().expect("tempdir");
    let output = run_isolated(dir.path(), &[
        "mcp",
        "add",
        "notion",
        "https://mcp.notion.com/mcp",
        "--no-login",
    ]);
    assert!(
        output.status.success(),
        "meka mcp add failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let list = run_isolated(dir.path(), &["mcp", "list"]);
    assert!(list.status.success());
    let stdout = String::from_utf8_lossy(&list.stdout);
    assert!(
        stdout.contains("notion") && stdout.contains("https://mcp.notion.com/mcp"),
        "mcp list should show the added server: {}",
        stdout
    );
}

#[test]
fn mcp_add_stdio_positional_command_and_args() {
    let dir = tempfile::tempdir().expect("tempdir");
    let output = run_isolated(dir.path(), &[
        "mcp",
        "add",
        "pg",
        "npx",
        "-y",
        "@modelcontextprotocol/server-postgres",
    ]);
    assert!(
        output.status.success(),
        "stdio add should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let get = run_isolated(dir.path(), &["mcp", "get", "pg"]);
    let stdout = String::from_utf8_lossy(&get.stdout);
    assert!(stdout.contains("transport:   stdio"), "{}", stdout);
    assert!(stdout.contains("npx"), "{}", stdout);
    assert!(
        stdout.contains("@modelcontextprotocol/server-postgres"),
        "{}",
        stdout
    );
}

#[test]
fn mcp_disable_sets_disabled_flag() {
    let dir = tempfile::tempdir().expect("tempdir");
    let add = run_isolated(dir.path(), &[
        "mcp",
        "add",
        "flaky",
        "npx",
        "-y",
        "mcp-flaky",
    ]);
    assert!(
        add.status.success(),
        "add: {}",
        String::from_utf8_lossy(&add.stderr)
    );

    let disable = run_isolated(dir.path(), &["mcp", "disable", "flaky"]);
    assert!(
        disable.status.success(),
        "disable: {}",
        String::from_utf8_lossy(&disable.stderr)
    );

    let config_path = dir.path().join("meka").join("config.toml");
    let toml_text = std::fs::read_to_string(&config_path).expect("read config");
    assert!(
        toml_text.contains("disabled = true"),
        "expected disabled = true in config, got:\n{}",
        toml_text
    );

    let enable = run_isolated(dir.path(), &["mcp", "enable", "flaky"]);
    assert!(
        enable.status.success(),
        "enable: {}",
        String::from_utf8_lossy(&enable.stderr)
    );
    let toml_text = std::fs::read_to_string(&config_path).expect("read config");
    assert!(
        !toml_text.contains("disabled = true"),
        "disabled flag should be cleared, got:\n{}",
        toml_text
    );
}

#[test]
fn mcp_add_with_disabled_flag_persists() {
    let dir = tempfile::tempdir().expect("tempdir");
    let output = run_isolated(dir.path(), &[
        "mcp",
        "add",
        "staging",
        "https://mcp.example.com/mcp",
        "--no-login",
        "--disabled",
    ]);
    assert!(
        output.status.success(),
        "add --disabled: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let config_path = dir.path().join("meka").join("config.toml");
    let toml_text = std::fs::read_to_string(&config_path).expect("read config");
    assert!(
        toml_text.contains("disabled = true"),
        "expected disabled = true from --disabled flag, got:\n{}",
        toml_text
    );
}

#[test]
fn mcp_add_http_without_url_fails() {
    let dir = tempfile::tempdir().expect("tempdir");
    let output = run_isolated(dir.path(), &["mcp", "add", "broken", "--transport", "http"]);
    assert!(
        !output.status.success(),
        "http without URL must be rejected, stdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("http transport needs a URL") || stderr.contains("URL"),
        "error should mention URL: {}",
        stderr
    );
}

#[test]
fn mcp_add_no_login_prints_skip_hint_when_probe_says_auth_required() {
    // Probing the real Notion endpoint classifies as AuthRequired; `--no-login` must surface the
    // "run `meka mcp login` later" hint rather than entering the OAuth flow. The hint goes to
    // tracing at info level; default filter is `warn`, so we pass `-v` to lift the floor and read
    // the message from stderr.
    let dir = tempfile::tempdir().expect("tempdir");
    let output = run_isolated(dir.path(), &[
        "-v",
        "mcp",
        "add",
        "notion",
        "https://mcp.notion.com/mcp",
        "--no-login",
    ]);
    assert!(
        output.status.success(),
        "mcp add should succeed even when probe says auth required: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("skipping auto-login"),
        "expected skip hint in stderr, got: {}",
        stderr
    );
    assert!(
        stderr.contains("meka mcp login notion"),
        "expected follow-up command in stderr, got: {}",
        stderr
    );
}

#[cfg(unix)]
#[test]
fn mcp_add_rollback_on_sigint_during_auto_login() {
    // Reproduces the "user hits Ctrl-C while the OAuth flow is waiting for the browser callback"
    // scenario: start `meka mcp add` without --no-login against a server that requires auth, wait
    // until the auto-login is clearly in progress, send SIGINT, then confirm nothing remains in
    // config.toml.
    use std::{
        io::{BufRead, BufReader},
        process::Stdio,
    };

    let dir = tempfile::tempdir().expect("tempdir");
    let mut child = meka()
        // `-v` so the `running OAuth authorization` info log is visible; we use it as the
        // "auto-login has started" signal before sending SIGINT.
        .args(["-v", "mcp", "add", "notion", "https://mcp.notion.com/mcp"])
        .env("MEKA_CONFIG_DIR", dir.path().join("meka"))
        .env("XDG_CONFIG_HOME", dir.path())
        .env("HOME", dir.path())
        .env("XDG_DATA_HOME", dir.path().join("data"))
        // Decouple stdin from the test harness so the paste-mode read doesn't hang waiting on a
        // terminal that isn't there.
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn meka mcp add");

    // Wait until we've seen the "running OAuth authorization" line so we know the child is past the
    // write + probe and is inside the SIGINT-covered post-persist section. The signpost now lives
    // on stderr (via tracing), not stdout. We drain into `captured` so the subsequent rollback log
    // lines are preserved across the SIGINT for the final assertion.
    let stderr = child.stderr.take().expect("child stderr");
    let mut reader = BufReader::new(stderr);
    let mut captured = String::new();
    let mut saw_running_line = false;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    while std::time::Instant::now() < deadline {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                captured.push_str(&line);
                if line.contains("running OAuth authorization") {
                    saw_running_line = true;
                    break;
                }
            }
            Err(_) => break,
        }
    }
    assert!(
        saw_running_line,
        "child never reached the auto-login stage within 15s; stderr so far:\n{}",
        captured
    );

    // Send SIGINT to the child, same signal a user gets from Ctrl-C.
    unsafe {
        libc::kill(child.id() as libc::pid_t, libc::SIGINT);
    }

    // Drain the rest of stderr until the child exits so we can assert on the rollback log lines.
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => captured.push_str(&line),
            Err(_) => break,
        }
    }

    let status = child.wait().expect("wait on meka");
    assert!(
        !status.success(),
        "meka should exit non-zero after SIGINT during auto-login"
    );
    assert!(
        captured.contains("interrupted") && captured.contains("rolling back"),
        "expected interrupted/rollback message in stderr, got:\n{}",
        captured
    );

    // Verify the entry was rolled out of config.toml.
    let config_path = dir.path().join("meka").join("config.toml");
    let config_contents = std::fs::read_to_string(&config_path).unwrap_or_default();
    assert!(
        !config_contents.contains("notion"),
        "rolled-back entry must not remain in config.toml; got:\n{}",
        config_contents
    );
}

#[test]
fn mcp_add_tool_filter_and_permission_flags_round_trip() {
    // --allow-tool, --disable-tool, and --tool-permission should land as allowed_tools,
    // disabled_tools, and a [tool_permissions] sub- table on the server entry in config.toml. We
    // also validate one parse error so the flag is actually enforced at add time.
    let dir = tempfile::tempdir().expect("tempdir");

    // Rejection path: missing '=' in --tool-permission.
    let bad = run_isolated(dir.path(), &[
        "mcp",
        "add",
        "broken",
        "https://mcp.example.com/mcp",
        "--no-login",
        "--tool-permission",
        "just-a-name",
    ]);
    assert!(
        !bad.status.success(),
        "bad --tool-permission should reject: {}",
        String::from_utf8_lossy(&bad.stdout)
    );

    // Happy path: all three fields populate correctly.
    let output = run_isolated(dir.path(), &[
        "mcp",
        "add",
        "notion",
        "https://mcp.notion.com/mcp",
        "--no-login",
        "--allow-tool",
        "notion-search",
        "--allow-tool",
        "notion-fetch",
        "--disable-tool",
        "notion-delete-pages",
        "--tool-permission",
        "notion-create-pages=unrestricted",
        "--tool-permission",
        "notion-update-page=unrestricted",
    ]);
    assert!(
        output.status.success(),
        "mcp add should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let config_path = dir.path().join("meka").join("config.toml");
    let contents = std::fs::read_to_string(&config_path).expect("read config");
    // Check the allow/block arrays and the nested permissions table.
    assert!(
        contents.contains("allowed_tools"),
        "config missing allowed_tools:\n{}",
        contents
    );
    assert!(
        contents.contains("notion-search") && contents.contains("notion-fetch"),
        "allowed_tools entries missing:\n{}",
        contents
    );
    assert!(
        contents.contains("disabled_tools") && contents.contains("notion-delete-pages"),
        "disabled_tools missing:\n{}",
        contents
    );
    assert!(
        contents.contains("tool_permissions"),
        "config missing [tool_permissions]:\n{}",
        contents
    );
    assert!(
        contents.contains("notion-create-pages")
            && contents.contains("notion-update-page")
            && contents.contains("unrestricted"),
        "tool_permissions entries missing:\n{}",
        contents
    );
}

#[test]
fn mcp_add_oauth_writes_auth_block() {
    let dir = tempfile::tempdir().expect("tempdir");
    let output = run_isolated(dir.path(), &[
        "mcp",
        "add",
        "notion",
        "https://mcp.notion.com/mcp",
        "--auth",
        "oauth",
        "--scope",
        "read",
        "--scope",
        "write",
        "--no-login",
    ]);
    assert!(
        output.status.success(),
        "oauth add should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Read back the config.toml we wrote.
    let config_path = dir.path().join("meka").join("config.toml");
    let contents = std::fs::read_to_string(&config_path).expect("read config");
    assert!(contents.contains("type = \"oauth\""), "{}", contents);
    assert!(contents.contains("read"), "{}", contents);
    assert!(contents.contains("write"), "{}", contents);
}

/// `meka skill remove` must wait on the store lock, like every other skill door.
///
/// It had its own `remove_dir_all` and never went through `delete_skill`, so it completed in 70 ms
/// against a lock every other door waited on — able to delete a skill directory while a
/// `skill_write` or `PUT /v1/skills` was composing and renaming `SKILL.md` inside it.
#[test]
fn skill_remove_waits_for_a_store_lock_another_process_holds() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = dir.path().join("config");
    let skills = config.join("skills");
    std::fs::create_dir_all(skills.join("victim")).expect("skill dir");
    std::fs::write(
        skills.join("victim").join("SKILL.md"),
        "---\nname: victim\ndescription: a skill\n---\nbody\n",
    )
    .expect("seed");

    let lock_file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(skills.join(".meka-store.lock"))
        .expect("open the store lock");
    let mut lock = fd_lock::RwLock::new(lock_file);
    let guard = lock.write().expect("hold the store lock");

    let mut blocked = meka()
        .env("MEKA_CONFIG_DIR", &config)
        .env("MEKA_DATA_DIR", dir.path().join("data"))
        .args(["skill", "remove", "victim"])
        .spawn()
        .expect("spawn meka skill remove");
    std::thread::sleep(std::time::Duration::from_millis(750));
    assert!(
        blocked.try_wait().expect("try_wait").is_none(),
        "the delete must wait for the lock this test is holding"
    );
    assert!(
        skills.join("victim").join("SKILL.md").exists(),
        "and must not have removed anything yet"
    );

    drop(guard);
    assert!(blocked.wait().expect("wait").success(), "then it completes");
    assert!(!skills.join("victim").exists(), "and the skill is gone");
}

/// Run `meka` isolated, with a config file, and a prompt that will fail on the missing provider.
///
/// The prompt is what forces full config resolution: `session list` and friends short-circuit
/// before `ResolvedConfig::from_cli` runs, so a flag that only affects the resolved config is
/// unobservable through them. Failing on "no provider profiles configured" is the expected end of
/// every call here; what the tests read is what was warned on the way there.
fn resolve_config(dir: &std::path::Path, config: &str, args: &[&str]) -> std::process::Output {
    let config_dir = dir.join("meka");
    std::fs::create_dir_all(&config_dir).expect("config dir");
    if !config.is_empty() {
        std::fs::write(config_dir.join("config.toml"), config).expect("write config");
    }
    let mut command = meka();
    command
        .args(args)
        .args(["-p", "hi"])
        .env("MEKA_CONFIG_DIR", &config_dir)
        .env("MEKA_DATA_DIR", dir.join("data"))
        .env("HOME", dir);
    command
        .output()
        .unwrap_or_else(|err| panic!("failed to spawn meka {args:?}: {err}"))
}

/// The retired `write` mode fails at every door it can be spelled at, and never resolves quietly.
///
/// `Permission` is read as a grant *and* as a requirement, so re-pointing `write` at one of the two
/// new modes would have silently admitted tools a rung earlier than their author intended. Retiring
/// it is what makes every stale config loud, and the value of that depends entirely on each surface
/// actually refusing rather than one of them keeping a private table that still maps it.
#[test]
fn the_retired_write_mode_is_refused_at_the_flag_and_in_the_config_file() {
    let dir = tempfile::tempdir().expect("tempdir");

    let flag = meka()
        .args(["--permission", "write", "session", "list"])
        .env("MEKA_CONFIG_DIR", dir.path().join("meka"))
        .env("MEKA_DATA_DIR", dir.path().join("data"))
        .output()
        .expect("spawn meka");
    assert!(!flag.status.success(), "--permission write must not start");
    let stderr = String::from_utf8_lossy(&flag.stderr);
    assert!(
        stderr.contains("was split")
            && stderr.contains("workspace")
            && stderr.contains("unrestricted"),
        "the refusal has to name both replacements, not just report 'invalid': {stderr}"
    );

    let file = resolve_config(
        dir.path(),
        "[permissions]\ndefault = \"write\"\nenabled = [\"read\", \"write\"]\n",
        &[],
    );
    let stderr = String::from_utf8_lossy(&file.stderr);
    for surface in ["[permissions].default", "[permissions].enabled"] {
        assert!(
            stderr.contains(surface),
            "{surface} must warn about the retired mode by name: {stderr}"
        );
    }
    assert!(
        stderr.contains("was split"),
        "and must carry the same explanation the flag gives: {stderr}"
    );
}

/// `workspace` is spellable everywhere the other modes are, and reaches config resolution.
#[test]
fn the_workspace_mode_is_accepted_at_the_flag_and_in_the_config_file() {
    let dir = tempfile::tempdir().expect("tempdir");

    let flag = resolve_config(dir.path(), "", &["--permission", "workspace"]);
    let stderr = String::from_utf8_lossy(&flag.stderr);
    assert!(
        stderr.contains("no provider profiles configured"),
        "resolution must get past the permission flag to the provider: {stderr}"
    );
    assert!(
        !stderr.contains("invalid value"),
        "`workspace` must not be rejected by the parser: {stderr}"
    );

    let file = resolve_config(
        dir.path(),
        "[permissions]\ndefault = \"workspace\"\nenabled = [\"read\", \"workspace\"]\n",
        &[],
    );
    let stderr = String::from_utf8_lossy(&file.stderr);
    assert!(
        !stderr.contains("ignoring invalid"),
        "neither [permissions] key may treat `workspace` as unknown: {stderr}"
    );
}

/// `--writable-root` naming a path that does not exist warns, and does not fail the run.
///
/// Both halves matter. A root that cannot be canonicalised is dropped from the boundary by
/// `writable_roots`, so without the warning the user learns about it from a refused write naming a
/// boundary they believed included the path. And a build directory that does not exist *yet* is a
/// legitimate root, so this cannot be an error: the boundary is recomputed on every write.
#[test]
fn an_unresolvable_writable_root_warns_without_failing_the_run() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing = dir.path().join("not-created-yet");
    let output = resolve_config(dir.path(), "", &[
        "--writable-root",
        missing.to_str().expect("path"),
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--writable-root") && stderr.contains("does not resolve"),
        "an unresolvable root must say so: {stderr}"
    );
    assert!(
        stderr.contains("no provider profiles configured"),
        "and must not be what stops the run: {stderr}"
    );

    // The existing case stays quiet, so the warning means something when it appears.
    std::fs::create_dir_all(&missing).expect("create the root");
    let output = resolve_config(dir.path(), "", &[
        "--writable-root",
        missing.to_str().expect("path"),
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("does not resolve"),
        "a root that exists must not warn: {stderr}"
    );
}
