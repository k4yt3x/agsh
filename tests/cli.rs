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

/// A mode meka does not have fails at every door it can be spelled at, and never resolves quietly.
///
/// `Permission` is read as a grant *and* as a requirement, so a surface that quietly mapped an
/// unknown string onto some mode would admit tools at authority nobody chose. The value of refusing
/// depends entirely on every surface doing it, rather than one of them keeping a private table.
#[test]
fn a_mode_meka_does_not_have_is_refused_at_the_flag_and_in_the_config_file() {
    let dir = tempfile::tempdir().expect("tempdir");

    let flag = meka()
        .args(["--permission", "elevated", "session", "list"])
        .env("MEKA_CONFIG_DIR", dir.path().join("meka"))
        .env("MEKA_DATA_DIR", dir.path().join("data"))
        .output()
        .expect("spawn meka");
    assert!(!flag.status.success(), "an unknown mode must not start");
    let stderr = String::from_utf8_lossy(&flag.stderr);
    assert!(
        stderr.contains("workspace") && stderr.contains("unrestricted"),
        "the refusal has to list the modes meka does have: {stderr}"
    );

    let file = resolve_config(
        dir.path(),
        "[permissions]\ndefault = \"elevated\"\nenabled = [\"read\", \"elevated\"]\n",
        &[],
    );
    let stderr = String::from_utf8_lossy(&file.stderr);
    for surface in ["[permissions].default", "[permissions].enabled"] {
        assert!(
            stderr.contains(surface),
            "{surface} must warn about the unknown mode by name: {stderr}"
        );
    }
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

/// `--continue` and `--resume` name *this run's session*, and neither long-lived host has one:
/// each creates a session per `session/new` or `POST /v1/sessions`. They used to parse and do
/// nothing, which was worse than it sounds: `-c` / `-r` set `session_resume`, which switches off
/// the default-profile check a host with no configured default needs most, so `meka -c acp` wrote
/// a session row naming the empty profile and failed its first turn complaining about a session it
/// had created moments earlier.
///
/// The list was longer before 0.44, when `--model` and `--base-url` were refused here for the same
/// reason. Those flags are gone entirely, so nothing about them needs refusing.
#[test]
fn the_long_lived_hosts_refuse_the_flags_that_name_one_session() {
    let dir = tempfile::tempdir().expect("tempdir");
    for host in ["acp", "serve"] {
        for flag in [vec!["--continue"], vec!["--resume", "0e5f"]] {
            // Isolated, like every other CLI test. A regression in the guard would otherwise reach
            // the host's real startup, and `meka serve` would bind the port in the *developer's*
            // `config.toml` and run until the harness gave up -- a hang rather than a failure.
            let mut args = flag.clone();
            args.push(host);
            let output = run_isolated(dir.path(), &args);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                !output.status.success(),
                "meka {} {host} should be refused, got: {stderr}",
                flag.join(" ")
            );
            assert!(
                stderr.contains(flag[0]) && stderr.contains("one run's session"),
                "meka {} {host} must say why: {stderr}",
                flag.join(" ")
            );
        }
    }
}

/// A `config.toml` meka cannot parse must not stop the commands that exist to repair one.
///
/// `meka mcp remove` and `meka provider remove` edit the raw document through `toml_edit` and never
/// parse it into a `ConfigFile`, which is exactly what makes them the way out of a config an
/// unknown key or a bad value has made unloadable. Gating the whole subcommand path on a readable
/// config closed that door: the fix for "the ledger must not adopt a profile it inferred from a
/// parse error" was briefly applied one level too high, and every subcommand refused.
///
/// The ledger's own protection is asserted where it lives, in
/// `session::migrations::tests::an_unreadable_config_refuses_to_stamp_carried_sessions_but_not_an_empty_store`:
/// a store with sessions to stamp is refused and left at its old version, and one with nothing to
/// stamp opens normally. That split is what lets both properties hold at once.
#[test]
fn an_unparseable_config_still_lets_the_commands_that_repair_it_run() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config_dir = dir.path().join("meka");
    std::fs::create_dir_all(&config_dir).expect("config dir");
    // Valid TOML that `serde` rejects, which is the shape the repair path is for:
    // `deny_unknown_fields` refuses the whole file over one stray key, while `toml_edit` still
    // parses it, so the document can be edited even though the config cannot be loaded. A
    // *syntax* error defeats `toml_edit` too and has never been repairable from the CLI; that
    // is not what this guards.
    let config = "default_provider = \"work\"\n\n[providers.work]\ntype = \
                  \"anthropic-messages\"\nmodel = \"some-model\"\nstray_unknown_key = 1\n";
    std::fs::write(config_dir.join("config.toml"), config).expect("write config.toml");

    let output = run_isolated(dir.path(), &["provider", "remove", "work"]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "`meka provider remove` must run on the very config it exists to repair: {stderr}"
    );
    let after = std::fs::read_to_string(config_dir.join("config.toml")).expect("read config back");
    assert!(
        !after.contains("[providers.work]"),
        "the profile was not actually removed, so the repair did not happen:\n{after}"
    );

    // The readers are the other half of the split and must keep refusing: answering "No MCP servers
    // configured" out of a file meka could not read would state something false.
    // `run_mcp_subcommand` branches on exactly this, and the two halves are what let a broken
    // config be both survivable and repairable.
    //
    // A second directory, because the repair above has by now *fixed* the first one: removing the
    // profile took the stray key with it.
    let unrepaired = tempfile::tempdir().expect("tempdir");
    let unrepaired_config = unrepaired.path().join("meka");
    std::fs::create_dir_all(&unrepaired_config).expect("config dir");
    std::fs::write(unrepaired_config.join("config.toml"), config).expect("write config.toml");
    let output = run_isolated(unrepaired.path(), &["mcp", "list"]);
    assert!(
        !output.status.success(),
        "`meka mcp list` must not answer out of a config it could not read"
    );
}

/// `--provider` is deliberately *not* refused above: it selects which configured profile the host
/// defaults to, which is a property of the host rather than of one session. A guard that lumped it
/// in with the four would take a real capability away.
#[test]
fn a_long_lived_host_still_takes_provider() {
    let dir = tempfile::tempdir().expect("tempdir");
    // One configured profile, so the refusal is "no profile named X" rather than "no profiles
    // configured". Seeded rather than inherited: run un-isolated, this test read the developer's
    // own `config.toml` and passed only because it happened to have a profile in it.
    let config_dir = dir.path().join("meka");
    std::fs::create_dir_all(&config_dir).expect("config dir");
    std::fs::write(
        config_dir.join("config.toml"),
        "default_provider = \"work\"\n\n[providers.work]\ntype = \"anthropic-messages\"\n\
         model = \"m\"\n",
    )
    .expect("write config");

    let output = run_isolated(dir.path(), &[
        "--provider",
        "definitely-not-configured",
        "serve",
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no provider profile named"),
        "--provider must reach profile selection rather than the flag guard: {stderr}"
    );
}

/// Write a `config.toml` with `profiles` configured and `default_provider` naming `default`.
///
/// Every endpoint is port 9, which discards, so a turn that got as far as the network could not
/// reach anything. The tests below never get that far: they run with the scripted provider.
fn write_provider_config(dir: &std::path::Path, default: &str, profiles: &[&str]) {
    let config_dir = dir.join("meka");
    std::fs::create_dir_all(&config_dir).expect("config dir");
    let mut config = format!(
        "default_provider = \"{default}\"\n\n[permissions]\ndefault = \"read\"\nenabled = \
         [\"read\"]\n"
    );
    for profile in profiles {
        config.push_str(&format!(
            "\n[providers.{profile}]\ntype = \"openai-chat-completions\"\nmodel = \
             \"{profile}-model\"\nbase_url = \"http://127.0.0.1:9/\"\n"
        ));
    }
    std::fs::write(config_dir.join("config.toml"), config).expect("write config.toml");
}

/// Run one `meka` turn against the scripted mock provider, which is how these tests get a session
/// row without a credential or a network.
///
/// The mock is compiled into debug builds only (`MEKA_MOCK_PROVIDER=1`), which is what `cargo test`
/// builds; `tests/multiprocess.rs` rests on the same thing.
fn run_scripted(dir: &std::path::Path, args: &[&str]) -> std::process::Output {
    let script = dir.join("script.json");
    std::fs::write(
        &script,
        r#"[[{"kind":"text","text":"ok"},{"kind":"message_end","stop_reason":"end_turn"}]]"#,
    )
    .expect("write the provider script");
    meka()
        .args(args)
        .env("MEKA_CONFIG_DIR", dir.join("meka"))
        .env("MEKA_DATA_DIR", dir.join("data").join("meka"))
        .env("XDG_CONFIG_HOME", dir)
        .env("HOME", dir)
        .env("XDG_DATA_HOME", dir.join("data"))
        .env("MEKA_MOCK_PROVIDER", "1")
        .env("MEKA_MOCK_PROVIDER_SCRIPT", &script)
        .output()
        .unwrap_or_else(|error| panic!("failed to spawn meka {:?}: {}", args, error))
}

/// The store an isolated run left behind, read directly: what is being set up is a row shape no
/// command produces on purpose.
fn store(dir: &std::path::Path) -> rusqlite::Connection {
    rusqlite::Connection::open(dir.join("data").join("meka").join("meka.db"))
        .expect("open the store")
}

/// The id of the one session a test made.
fn only_session(dir: &std::path::Path) -> String {
    store(dir)
        .query_row("SELECT id FROM sessions", [], |row| row.get::<_, String>(0))
        .expect("exactly one session")
}

/// Resuming a session whose recorded profile has left `config.toml` must fail the process, not just
/// print about it.
///
/// The interactive host rendered the refusal and returned `Ok(())`, so `meka -r <id>; echo $?` said
/// `0` for a session it had refused to open and every supervisor and wrapper script read that as
/// success. `--oneshot` on the same session, and a fresh session with an unresolvable
/// `default_provider` in either mode, all exited 1 already; the resume path in the REPL host was
/// alone in not doing so.
#[test]
fn resuming_a_session_whose_profile_is_gone_exits_nonzero() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_provider_config(dir.path(), "ghost", &["alpha", "ghost"]);
    let created = run_scripted(dir.path(), &["--oneshot", "hello"]);
    assert!(
        created.status.success(),
        "the first turn should have created a session: {}",
        String::from_utf8_lossy(&created.stderr)
    );
    let id = only_session(dir.path());

    // What `meka provider remove ghost` or a hand edit leaves behind: the row still names it.
    write_provider_config(dir.path(), "alpha", &["alpha"]);

    let refused = run_isolated(dir.path(), &["-r", &id]);
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(
        stderr.contains("is not configured"),
        "the resume should have been refused by name: {stderr}"
    );
    assert!(
        !refused.status.success(),
        "a refused resume must exit non-zero, got {:?}: {stderr}",
        refused.status
    );
    // The hint adds the one thing the refusal above it cannot: the session id, and the only command
    // that rewrites a row's binding.
    assert!(
        stderr.contains(&format!("meka -r {id} --provider alpha")),
        "the hint should give the command that repins this session: {stderr}"
    );
    // And it adds nothing else. `provider add` here would have to invent the deleted profile's
    // `--type` and `--model`, which meka never saw: `ghost` may have been `openai-responses` on
    // another model, so the command would create a different profile under the name the session
    // wants. The refusal above already says to restore it from config.toml, which is the honest
    // version of the same advice.
    assert!(
        !stderr.contains("provider add"),
        "the hint must not suggest recreating a profile whose type and model it cannot know: \
         {stderr}"
    );
}

/// `--provider` on a resume rewrites the row, which is the whole point of it being a repin rather
/// than a per-run override.
///
/// `apply_session_repin` could be replaced with `Ok(())` and every test stayed green: the resume
/// succeeded, the run used the new profile for its one turn, and the row silently kept the old one,
/// so the *next* resume went back. The row is the fact; this asserts the row.
#[test]
fn a_resume_with_provider_rewrites_the_row() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_provider_config(dir.path(), "alpha", &["alpha", "beta"]);
    let created = run_scripted(dir.path(), &["--oneshot", "hello"]);
    assert!(
        created.status.success(),
        "the first turn should have created a session: {}",
        String::from_utf8_lossy(&created.stderr)
    );
    let id = only_session(dir.path());
    assert_eq!(recorded_profile(dir.path(), &id), "alpha");

    let moved = run_scripted(dir.path(), &[
        "-r",
        &id,
        "--provider",
        "beta",
        "--oneshot",
        "hi",
    ]);
    assert!(
        moved.status.success(),
        "the repinned resume should run: {}",
        String::from_utf8_lossy(&moved.stderr)
    );
    assert_eq!(
        recorded_profile(dir.path(), &id),
        "beta",
        "the row must hold the new profile, or the next resume goes back to the old one"
    );
}

/// A `--provider` naming nothing configured is refused before anything is written, by the check
/// that reads the configured set rather than by the later failure to build a provider.
///
/// Both refuse, which is why this asserts the *message*: dropping the `!` from the membership test
/// inverts it, so a configured name bails and an unconfigured one falls through to fail later with
/// different wording. Exit code alone cannot tell those apart.
#[test]
fn a_resume_with_an_unconfigured_provider_is_refused_by_name() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_provider_config(dir.path(), "alpha", &["alpha"]);
    let created = run_scripted(dir.path(), &["--oneshot", "hello"]);
    assert!(created.status.success());
    let id = only_session(dir.path());

    let refused = run_scripted(dir.path(), &[
        "-r",
        &id,
        "--provider",
        "ghost",
        "--oneshot",
        "hi",
    ]);
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(!refused.status.success(), "must not run: {stderr}");
    assert!(
        stderr.contains("`meka provider list` shows the configured ones"),
        "the refusal must come from the membership check, not from a later build failure: {stderr}"
    );
    assert_eq!(
        recorded_profile(dir.path(), &id),
        "alpha",
        "a refused repin must leave the row alone"
    );
}

/// `meka provider set` is the successor to the retired `--model`: it writes the key, leaves the
/// rest of the file alone, and leaves behind a config the next process can still start on.
///
/// End to end rather than at `set_profile_field`, because the unit test cannot see the last of
/// those: a write that parses in isolation can still produce a file that fails at startup.
#[test]
fn provider_set_writes_the_key_and_leaves_a_config_the_next_run_can_start_on() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_provider_config(dir.path(), "alpha", &["alpha"]);

    // A comment beside the key, which is exactly what a whole-table rewrite would eat.
    let config_path = dir.path().join("meka").join("config.toml");
    let annotated = std::fs::read_to_string(&config_path)
        .expect("read config")
        .replace(
            "model = \"alpha-model\"",
            "model = \"alpha-model\" # the model, annotated",
        );
    std::fs::write(&config_path, annotated).expect("write config");

    let set = run_isolated(dir.path(), &[
        "provider",
        "set",
        "alpha",
        "model",
        "swapped-model",
    ]);
    assert!(
        set.status.success(),
        "set should succeed: {}",
        String::from_utf8_lossy(&set.stderr)
    );

    let written = std::fs::read_to_string(&config_path).expect("read config");
    assert!(
        written.contains("model = \"swapped-model\""),
        "the new model is written: {written}"
    );
    assert!(
        written.contains("# the model, annotated"),
        "the comment beside the changed key survives: {written}"
    );
    assert!(
        written.contains("base_url = \"http://127.0.0.1:9/\""),
        "the profile's other keys survive: {written}"
    );

    // The edited file still starts a process and runs a turn, which is what a whole class of
    // botched writes would break. Deliberately *not* a claim that the new model reached the wire:
    // `run_scripted`'s reply is fixed text, so nothing here can observe which model was built, and
    // saying otherwise would describe a guard this does not have.
    let turn = run_scripted(dir.path(), &["--oneshot", "hello"]);
    assert!(
        turn.status.success(),
        "the turn should run: {}",
        String::from_utf8_lossy(&turn.stderr)
    );

    // And `--unset` returns the profile to stating nothing, which a later run then refuses by name
    // rather than inventing a model for.
    let unset = run_isolated(dir.path(), &[
        "provider", "set", "alpha", "model", "--unset",
    ]);
    assert!(unset.status.success(), "unset should succeed");
    let cleared = std::fs::read_to_string(&config_path).expect("read config");
    assert!(
        !cleared.contains("swapped-model"),
        "the key is gone, not emptied: {cleared}"
    );
}

/// The retired per-profile override flags are gone from the parser, not merely ignored.
///
/// A flag that parses and does nothing is worse than one that does not parse: a script pinning a
/// model would keep exiting 0 while every turn ran on the profile's own. clap's unknown-argument
/// error is the honest answer, and it is what tells the user to look for the new door.
#[test]
fn the_retired_profile_override_flags_no_longer_parse() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_provider_config(dir.path(), "alpha", &["alpha"]);
    for flag in [
        vec!["--model", "pinned-model"],
        vec!["--base-url", "https://example.invalid"],
        vec!["--thinking", "off"],
        vec!["--thinking-budget", "2048"],
    ] {
        let mut args = flag.clone();
        args.extend(["--oneshot", "hi"]);
        let output = run_isolated(dir.path(), &args);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !output.status.success(),
            "meka {} must not run: {stderr}",
            flag.join(" ")
        );
        assert!(
            stderr.contains("unexpected argument") && stderr.contains(flag[0]),
            "meka {} must be refused by name: {stderr}",
            flag.join(" ")
        );
    }
}

/// A setup failure that is *not* a missing profile must not offer to repin the session.
///
/// The gate could be replaced with `true` and nothing noticed, which turns every failed start into
/// advice to move a session that is bound exactly where it belongs. Here the profile is configured
/// and merely has no credential, so repinning fixes nothing.
#[test]
fn a_credential_failure_does_not_advise_repinning_a_session() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_provider_config(dir.path(), "alpha", &["alpha"]);
    let created = run_scripted(dir.path(), &["--oneshot", "hello"]);
    assert!(created.status.success());
    let id = only_session(dir.path());

    // No `MEKA_MOCK_PROVIDER`, so the real credential lookup runs and finds nothing stored.
    let refused = run_isolated(dir.path(), &["-r", &id]);
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(
        stderr.contains("no stored credential"),
        "this test needs the credential failure, not some other one: {stderr}"
    );
    assert!(
        !stderr.contains("Move this session onto"),
        "the session's profile is configured, so repinning is the wrong advice: {stderr}"
    );
}

/// The profile a session's row currently names.
fn recorded_profile(dir: &std::path::Path, id: &str) -> String {
    store(dir)
        .query_row("SELECT provider FROM sessions WHERE id = ?1", [id], |row| {
            row.get(0)
        })
        .expect("the session row")
}
