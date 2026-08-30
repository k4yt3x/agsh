//! The REPL, driven through a real pseudo-terminal.
//!
//! Everything else in `tests/` runs meka with pipes, which the interactive shell refuses: reedline
//! needs a terminal, and additionally queries the cursor position (DSR, `ESC[6n`) during its first
//! paint, so even `script -qec` fails. That left `run_repl` and everything downstream of it -- the
//! whole `[display]` blank-line contract -- with no automated coverage at all, and a dozen spacing
//! defects accumulated in it unnoticed.
//!
//! This owns the master side of a pty: it answers DSR itself and feeds input whenever the child
//! goes quiet, which is the only "it is at a prompt again" signal available without parsing
//! reedline's paint. The captured bytes are then replayed through the handful of control sequences
//! meka and reedline actually emit, because a raw comparison would be reading carriage returns and
//! erase-to-end-of-line as though they were content.
//!
//! Unix only, and debug-only: the scripted provider (`MEKA_MOCK_PROVIDER`) is compiled out of a
//! release build, the same thing `tests/multiprocess.rs` rests on.

#![cfg(unix)]
// Same rationale as the other integration tests: a failed assumption here is a broken test, and
// panicking says so at the point it broke.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::{
    os::unix::io::RawFd,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

/// The debug binary under test.
fn meka_binary() -> PathBuf {
    let mut path = std::env::current_exe().expect("test binary path");
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.join("meka")
}

/// An isolated install: both env dirs redirected, and every endpoint on port 9, which discards. A
/// run here cannot read the developer's config or reach a provider.
struct Install {
    root: tempfile::TempDir,
}

impl Install {
    fn new(newline_before_prompt: bool, newline_after_prompt: bool, extra_display: &str) -> Self {
        let root = tempfile::tempdir().expect("temp dir");
        let path = root.path();
        std::fs::create_dir_all(path.join("meka")).expect("config dir");
        std::fs::create_dir_all(path.join("data").join("meka")).expect("data dir");
        std::fs::create_dir_all(path.join("work")).expect("work dir");
        std::fs::write(
            path.join("meka").join("config.toml"),
            format!(
                "default_provider = \"default\"\n\n\
                 [permissions]\ndefault = \"read\"\nenabled = [\"read\"]\n\n\
                 [display]\nnewline_before_prompt = {newline_before_prompt}\n\
                 newline_after_prompt = {newline_after_prompt}\n{extra_display}\n\
                 [providers.default]\ntype = \"openai-chat-completions\"\n\
                 model = \"mock-model\"\nbase_url = \"http://127.0.0.1:9/\"\n"
            ),
        )
        .expect("write config.toml");
        Self { root }
    }

    fn script(&self, json: &str) -> PathBuf {
        let path = self.root.path().join("script.json");
        std::fs::write(&path, json).expect("write the provider script");
        path
    }
}

/// Run one REPL session, sending `inputs` line by line, and return the rows the terminal would
/// show.
fn run_repl(install: &Install, script: &str, inputs: &[&str]) -> Vec<String> {
    let script = install.script(script);
    let root = install.root.path().to_path_buf();
    let captured = drive(&meka_binary(), &root, &script, inputs);
    replay(&captured)
}

/// Fork a pty, exec meka on the child side, and drive the master side to completion.
fn drive(binary: &Path, root: &Path, script: &Path, inputs: &[&str]) -> Vec<u8> {
    let mut master: RawFd = -1;
    // SAFETY: `forkpty` with null pointers for the optional out-params is the documented way to get
    // a pty pair plus a child. The child branch below does nothing but set state and `exec`, which
    // is the one thing that is defined after `fork` in a threaded process.
    let pid = unsafe {
        libc::forkpty(
            &mut master,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    assert!(pid >= 0, "forkpty failed");

    if pid == 0 {
        // SAFETY: single-threaded from here to `execv`; every call below is async-signal-safe or a
        // libc wrapper this process is allowed to use before exec.
        unsafe {
            // Drop ECHO so our own DSR replies do not come back as input.
            let mut attrs: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(0, &mut attrs) == 0 {
                attrs.c_lflag &= !libc::ECHO;
                libc::tcsetattr(0, libc::TCSANOW, &attrs);
            }
        }
        // Not `Command`: this process *is* the child, and `exec` replaces it.
        let error = exec_meka(binary, root, script);
        // Only reachable if exec failed.
        eprintln!("exec failed: {error}");
        std::process::exit(127);
    }

    let captured = pump(master, inputs);
    // SAFETY: `master` is the fd `forkpty` handed back and is still open here.
    unsafe {
        libc::close(master);
        libc::kill(pid, libc::SIGKILL);
        let mut status = 0;
        libc::waitpid(pid, &mut status, 0);
    }
    captured
}

fn exec_meka(binary: &Path, root: &Path, script: &Path) -> std::io::Error {
    use std::os::unix::process::CommandExt;
    std::process::Command::new(binary)
        .arg("-c")
        .current_dir(root.join("work"))
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", root)
        .env("TERM", "xterm-256color")
        .env("COLUMNS", "100")
        .env("LINES", "40")
        .env("MEKA_CONFIG_DIR", root.join("meka"))
        .env("MEKA_DATA_DIR", root.join("data").join("meka"))
        .env("XDG_CONFIG_HOME", root)
        .env("XDG_DATA_HOME", root.join("data"))
        .env("MEKA_MOCK_PROVIDER", "1")
        .env("MEKA_MOCK_PROVIDER_SCRIPT", script)
        .exec()
}

/// Read the master side, answering DSR and sending the next input whenever the child falls quiet.
fn pump(master: RawFd, inputs: &[&str]) -> Vec<u8> {
    let mut captured = Vec::new();
    let mut pending: Vec<&str> = inputs.to_vec();
    pending.reverse();
    let mut buffer = [0u8; 65536];
    let mut last_activity = Instant::now();
    // The first send waits longer: the child has a store to migrate and MCP probes to time out
    // before it draws anything.
    let mut quiet_needed = Duration::from_millis(2500);
    let deadline = Instant::now() + Duration::from_secs(90);

    while Instant::now() < deadline {
        // SAFETY: `master` is open, and the buffer outlives the call.
        let read = unsafe {
            let mut poll = libc::pollfd {
                fd: master,
                events: libc::POLLIN,
                revents: 0,
            };
            if libc::poll(&mut poll, 1, 250) > 0 && poll.revents & libc::POLLIN != 0 {
                libc::read(master, buffer.as_mut_ptr().cast(), buffer.len())
            } else {
                -1
            }
        };

        if read > 0 {
            let chunk = &buffer[..read as usize];
            captured.extend_from_slice(chunk);
            last_activity = Instant::now();
            if find(chunk, b"\x1b[6n") {
                // Any plausible position will do: reedline only needs the column to lay its prompt
                // out, and re-derives the row.
                write_all(master, b"\x1b[1;1R");
            }
            continue;
        }
        if read == 0 {
            break;
        }

        if last_activity.elapsed() > quiet_needed {
            match pending.pop() {
                Some(line) => {
                    write_all(master, line.as_bytes());
                    // A lone control byte is a keystroke, not a line: `^C` has to reach the tty's
                    // line discipline as itself, and a trailing carriage return would be a second
                    // keystroke the test did not ask for.
                    if line.len() > 1 || !line.starts_with(|c: char| c.is_control()) {
                        write_all(master, b"\r");
                    }
                    last_activity = Instant::now();
                    quiet_needed = Duration::from_millis(1500);
                }
                // Everything sent and the child has gone quiet: it has exited or is idle at a
                // prompt, and either way there is nothing left to capture.
                None if last_activity.elapsed() > Duration::from_secs(4) => break,
                None => {}
            }
        }
    }
    captured
}

fn write_all(fd: RawFd, bytes: &[u8]) {
    let mut written = 0;
    while written < bytes.len() {
        // SAFETY: writing a sub-slice of a live buffer to an open fd.
        let count =
            unsafe { libc::write(fd, bytes[written..].as_ptr().cast(), bytes.len() - written) };
        if count <= 0 {
            return;
        }
        written += count as usize;
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// Replay the capture onto a grid of rows, so a blank line means what it looks like.
///
/// Only the sequences meka and reedline actually emit are modelled: SGR (dropped), erase-to-end-of-
/// line, move-to-column, carriage return, backspace and newline. Anything else is skipped as a unit
/// rather than printed, which is what keeps escape bytes out of the rows a test asserts on.
fn replay(data: &[u8]) -> Vec<String> {
    let mut rows: Vec<Vec<u8>> = vec![Vec::new()];
    let mut column = 0usize;
    let mut index = 0usize;

    while index < data.len() {
        if data[index] == 0x1b && index + 1 < data.len() && data[index + 1] == b'[' {
            let mut end = index + 2;
            while end < data.len() && !data[end].is_ascii_alphabetic() {
                end += 1;
            }
            if end >= data.len() {
                break;
            }
            let parameters = &data[index + 2..end];
            match data[end] {
                // Erase to end of line: truncate the row at the cursor.
                b'K' => {
                    let row = rows.last_mut().expect("a row is always present");
                    row.truncate(column.min(row.len()));
                }
                b'G' => column = 0,
                _ => {}
            }
            let _ = parameters;
            index = end + 1;
            continue;
        }
        match data[index] {
            b'\r' => column = 0,
            b'\n' => {
                rows.push(Vec::new());
                column = 0;
            }
            0x08 => column = column.saturating_sub(1),
            // A lone escape (an OSC or a sequence we do not model) -- skip the byte.
            0x1b => {}
            byte => {
                let row = rows.last_mut().expect("a row is always present");
                while row.len() < column {
                    row.push(b' ');
                }
                if column < row.len() {
                    row[column] = byte;
                } else {
                    row.push(byte);
                }
                column += 1;
            }
        }
        index += 1;
    }

    rows.into_iter()
        .map(|row| String::from_utf8_lossy(&row).trim_end().to_string())
        .collect()
}

/// The rows between the line that contains `after` and the next one containing `before`.
fn between<'a>(rows: &'a [String], after: &str, before: &str) -> &'a [String] {
    let start = rows
        .iter()
        .position(|row| row.contains(after))
        .unwrap_or_else(|| panic!("no row contains {after:?} in {rows:#?}"));
    let end = rows[start + 1..]
        .iter()
        .position(|row| row.contains(before))
        .unwrap_or_else(|| panic!("no row after {after:?} contains {before:?} in {rows:#?}"))
        + start
        + 1;
    &rows[start + 1..end]
}

fn blanks(rows: &[String]) -> usize {
    rows.iter().filter(|row| row.is_empty()).count()
}

const TOOLS_THEN_TEXT: &str = r#"[
 [{"kind":"tool_use_start","id":"t1","name":"schedule_list"},
  {"kind":"tool_use_end","input":{}},
  {"kind":"message_end","stop_reason":"tool_use"}],
 [{"kind":"text","text":"All cleared."},
  {"kind":"message_end","stop_reason":"end_turn"}]
]"#;

const TWO_TURNS: &str = r#"[
 [{"kind":"text","text":"First answer."},
  {"kind":"message_end","stop_reason":"end_turn"}],
 [{"kind":"tool_use_start","id":"t1","name":"schedule_list"},
  {"kind":"tool_use_end","input":{}},
  {"kind":"message_end","stop_reason":"tool_use"}],
 [{"kind":"text","text":"Second answer."},
  {"kind":"message_end","stop_reason":"end_turn"}]
]"#;

const FAILS_MID_ANSWER: &str = r#"[
 [{"kind":"text","text":"partial answer before the failure"},
  {"kind":"fail","message":"provider exploded"}],
 [{"kind":"text","text":"recovered"},
  {"kind":"message_end","stop_reason":"end_turn"}]
]"#;

/// The shape every other case is a variation on: one blank after the line you typed, one before the
/// next prompt, and the tool indicator and the answer separated by the block machine.
#[test]
fn a_turn_is_bracketed_once_on_each_side() {
    let install = Install::new(true, true, "");
    let rows = run_repl(&install, TOOLS_THEN_TEXT, &["do the thing", "/exit"]);

    let body = between(&rows, "do the thing", "> /exit");
    assert_eq!(
        blanks(body),
        3,
        "one after the prompt, one between indicator and text, one before the next prompt: {body:#?}"
    );
    assert_eq!(body.first().map(String::as_str), Some(""));
    assert_eq!(body.last().map(String::as_str), Some(""));
}

/// The setting used to stop working after the first turn: the blank was suppressed but the block
/// machine was suppressed with it, so the next turn's first block saw the previous turn's last one
/// and asked for a separator that looked exactly like the blank the user had disabled.
#[test]
fn disabling_both_blanks_leaves_no_prompt_spacing_on_any_turn() {
    let install = Install::new(false, false, "");
    let rows = run_repl(&install, TWO_TURNS, &["first", "second", "/exit"]);

    let first = between(&rows, "> first", "> second");
    assert_eq!(blanks(first), 0, "turn one is unspaced: {first:#?}");

    let second = between(&rows, "> second", "> /exit");
    assert_eq!(
        blanks(second),
        1,
        "only the indicator-to-text separator, which is not a prompt bracket: {second:#?}"
    );
}

/// A command that answers without running a turn used to be bracketed by neither the dispatcher nor
/// the turn, printing its error flush against the line above *and* the prompt below.
#[test]
fn a_command_that_never_runs_a_turn_is_still_bracketed() {
    let install = Install::new(true, true, "");
    let rows = run_repl(&install, TWO_TURNS, &["/skill nosuchskill", "/exit"]);

    let body = between(&rows, "/skill nosuchskill", "> /exit");
    assert_eq!(blanks(body), 2, "one blank on each side: {body:#?}");
    assert!(
        body.iter().any(|row| row.contains("unknown skill")),
        "the error is what the brackets are around: {body:#?}"
    );
}

/// A turn that dies mid-answer holds whatever it streamed. Ending the *episode* flushes it, which
/// puts it under the turn it belongs to; it used to be flushed by the next turn's `TurnStarted`,
/// printing it beneath the following prompt as though it answered that.
#[test]
fn a_failed_turn_shows_its_partial_answer_in_its_own_turn() {
    let install = Install::new(true, true, "");
    let rows = run_repl(&install, FAILS_MID_ANSWER, &["boom", "again", "/exit"]);

    let failed = between(&rows, "> boom", "> again");
    assert!(
        failed.iter().any(|row| row.contains("partial answer")),
        "the partial answer belongs to the turn that produced it: {failed:#?}"
    );

    let recovered = between(&rows, "> again", "> /exit");
    assert!(
        !recovered.iter().any(|row| row.contains("partial answer")),
        "and must not reappear under the next prompt: {recovered:#?}"
    );
    assert_eq!(
        recovered.first().map(String::as_str),
        Some(""),
        "the blank after the prompt is still the first thing: {recovered:#?}"
    );
}

/// The session-id notice is emitted before the turn starts, so it used to print above the blank
/// that was supposed to separate it from the line the user typed.
#[test]
fn the_opening_blank_precedes_the_session_notice() {
    let install = Install::new(true, true, "show_session_id_on_create = true\n");
    let rows = run_repl(&install, TWO_TURNS, &["first", "/exit"]);

    let body = between(&rows, "> first", "> /exit");
    assert_eq!(
        body.first().map(String::as_str),
        Some(""),
        "nothing may slip above the opening blank: {body:#?}"
    );
    assert!(
        body.iter().any(|row| row.contains("Creating new session")),
        "the notice is still shown: {body:#?}"
    );
}

/// Most commands answer through the `cli` modules, which print for themselves and are invisible to
/// the console. They are bracketed by the dispatcher announcing on their behalf; without that the
/// blank lines follow only the output the console happens to render itself, and `/status` prints
/// none of it.
#[test]
fn a_command_that_prints_for_itself_is_still_bracketed() {
    let install = Install::new(true, true, "");
    let rows = run_repl(&install, TWO_TURNS, &["/status", "/exit"]);

    let body = between(&rows, "> /status", "> /exit");
    assert_eq!(
        body.first().map(String::as_str),
        Some(""),
        "a blank after the line typed: {body:#?}"
    );
    assert_eq!(
        body.last().map(String::as_str),
        Some(""),
        "and one before the next prompt: {body:#?}"
    );
    assert!(
        body.iter().any(|row| row.contains("Session status")),
        "the table is what the brackets are around: {body:#?}"
    );
}

/// `/help` is answered by the REPL thread rather than the agent loop, through a printer the console
/// cannot see. It is bracketed by the same rule as everything else.
#[test]
fn help_is_bracketed_by_the_repl_thread_too() {
    let install = Install::new(true, true, "");
    let rows = run_repl(&install, TWO_TURNS, &["/help", "/exit"]);

    let body = between(&rows, "> /help", "> /exit");
    assert_eq!(body.first().map(String::as_str), Some(""), "{body:#?}");
    assert_eq!(body.last().map(String::as_str), Some(""), "{body:#?}");
    assert!(
        body.iter().any(|row| row.contains("Shortcuts:")),
        "the help text is what the brackets are around: {body:#?}"
    );
}

/// The todo list paints its own surrounding blanks, so the block machine must not add more. This is
/// the one block that would double if it were spaced like the others.
#[test]
fn a_todo_list_is_not_double_spaced() {
    const TODO: &str = r#"[
 [{"kind":"tool_use_start","id":"t1","name":"todo"},
  {"kind":"tool_use_end","input":{"title":"Work","items":["First","Second"]}},
  {"kind":"message_end","stop_reason":"tool_use"}],
 [{"kind":"text","text":"Done."},
  {"kind":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = Install::new(true, true, "");
    let rows = run_repl(&install, TODO, &["plan it", "/exit"]);

    let body = between(&rows, "> plan it", "> /exit");
    assert!(
        body.iter().any(|row| row.contains("TODO: Work")),
        "the list rendered: {body:#?}"
    );
    assert!(
        !body
            .windows(2)
            .any(|pair| pair[0].is_empty() && pair[1].is_empty()),
        "no two blank lines ever sit together: {body:#?}"
    );
}

/// Thinking renders as its own block, inside the episode's brackets like any other.
#[test]
fn a_thinking_block_sits_inside_the_brackets() {
    const THINKING: &str = r#"[
 [{"kind":"thinking_delta","text":"weighing the options"},
  {"kind":"thinking_complete"},
  {"kind":"text","text":"Here is the answer."},
  {"kind":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = Install::new(true, true, "\n[thinking]\nshow_content = true\n");
    let rows = run_repl(&install, THINKING, &["think about it", "/exit"]);

    let body = between(&rows, "> think about it", "> /exit");
    assert_eq!(body.first().map(String::as_str), Some(""), "{body:#?}");
    assert_eq!(body.last().map(String::as_str), Some(""), "{body:#?}");
    assert!(
        body.iter().any(|row| row.contains("weighing the options")),
        "the thinking block rendered: {body:#?}"
    );
    assert!(
        body.iter().any(|row| row.contains("Here is the answer")),
        "and the answer after it: {body:#?}"
    );
}

/// A provider notice renders as a hint inside the turn, between the same brackets as everything
/// else rather than beside them.
#[test]
fn a_notice_renders_inside_the_brackets() {
    const NOTICE: &str = r#"[
 [{"kind":"notice","message":"the model dropped an unsupported parameter"},
  {"kind":"text","text":"Answered anyway."},
  {"kind":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = Install::new(true, true, "");
    let rows = run_repl(&install, NOTICE, &["ask", "/exit"]);

    let body = between(&rows, "> ask", "> /exit");
    assert_eq!(body.first().map(String::as_str), Some(""), "{body:#?}");
    assert_eq!(body.last().map(String::as_str), Some(""), "{body:#?}");
    assert!(
        body.iter().any(|row| row.contains("unsupported parameter")),
        "the notice rendered: {body:#?}"
    );
}

/// `/provider <name>` is typed at the REPL thread but answered on the agent's side, so its
/// confirmation is the one piece of command output that neither the dispatcher nor a turn prints.
#[test]
fn a_forwarded_provider_switch_is_bracketed() {
    let install = Install::new(true, true, "");
    // A second profile to switch to; the first is what the session starts on.
    let extra = "\n[providers.other]\ntype = \"openai-chat-completions\"\n\
                 model = \"other-model\"\nbase_url = \"http://127.0.0.1:9/\"\n";
    let config = install.root.path().join("meka").join("config.toml");
    let mut text = std::fs::read_to_string(&config).expect("read config");
    text.push_str(extra);
    std::fs::write(&config, text).expect("write config");

    let rows = run_repl(&install, TWO_TURNS, &["/provider other", "/exit"]);

    let body = between(&rows, "> /provider other", "> /exit");
    assert_eq!(body.first().map(String::as_str), Some(""), "{body:#?}");
    assert_eq!(body.last().map(String::as_str), Some(""), "{body:#?}");
    assert!(
        body.iter()
            .any(|row| row.contains("Provider profile set to")),
        "the confirmation is what the brackets are around: {body:#?}"
    );
}

/// `/provider` with no argument names the profile this session runs on, then lists every configured
/// one with the backend it speaks, under a heading styled like `/status`'s. It used to be a
/// comma-joined run of names with no heading, which stops fitting long before a user stops adding
/// accounts and never said what the list was for.
#[test]
fn provider_lists_one_profile_per_line_with_its_backend() {
    let install = Install::new(true, true, "");
    let config = install.root.path().join("meka").join("config.toml");
    let mut text = std::fs::read_to_string(&config).expect("read config");
    text.push_str(
        "\n[providers.zzz-last]\ntype = \"anthropic-messages\"\n\
         model = \"other-model\"\nbase_url = \"http://127.0.0.1:9/\"\n",
    );
    std::fs::write(&config, text).expect("write config");

    let rows = run_repl(&install, TWO_TURNS, &["/provider", "/exit"]);
    let body = between(&rows, "> /provider", "> /exit");

    let current = body
        .iter()
        .position(|row| row == "Current provider profile: default")
        .unwrap_or_else(|| panic!("the answer comes first: {body:#?}"));
    let heading = body
        .iter()
        .position(|row| row == "Configured profiles")
        .unwrap_or_else(|| panic!("the list says what it is: {body:#?}"));
    assert!(
        heading > current && body[heading - 1].is_empty(),
        "the heading follows, set apart from it: {body:#?}"
    );
    assert!(
        body[heading + 1].starts_with("- "),
        "with its list directly beneath, as `/status` heads its own block: {body:#?}"
    );
    assert!(
        body.iter()
            .any(|row| row == "- default (openai-chat-completions)"),
        "each profile is its own line, with the backend: {body:#?}"
    );
    assert!(
        body.iter()
            .any(|row| row == "- zzz-last (anthropic-messages)"),
        "including the ones that are not current: {body:#?}"
    );
    assert!(
        !body.iter().any(|row| row.contains("Configured:")),
        "the comma-joined line is gone: {body:#?}"
    );
}

/// A successful `/cd` prints nothing, so it gets no blank lines; its failure prints, so it does.
#[test]
fn cd_is_spaced_only_when_it_has_something_to_say() {
    let install = Install::new(true, true, "");
    let rows = run_repl(&install, TWO_TURNS, &[
        "/cd /nonexistent-xyz",
        "/cd /tmp",
        "/exit",
    ]);

    let failure = between(&rows, "/cd /nonexistent-xyz", "/cd /tmp");
    assert_eq!(blanks(failure), 2, "the error is bracketed: {failure:#?}");

    let success = between(&rows, "/cd /tmp", "> /exit");
    assert_eq!(
        blanks(success),
        0,
        "a silent command gets no blanks around nothing: {success:#?}"
    );
}

/// Ctrl+C during a turn. The notice is the one piece of chrome that used to open with its own
/// newline to terminate whatever row it landed on -- right when a row was open, a stray blank line
/// when the cursor was already at column zero, and no brackets at all when the turn had not printed
/// anything yet.
#[test]
fn an_interrupted_turn_is_annotated_and_bracketed() {
    const SLOW: &str = r#"[
 [{"kind":"text","text":"starting the long answer"},
  {"kind":"sleep","ms":8000},
  {"kind":"text","text":"never reached"},
  {"kind":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = Install::new(true, true, "");
    let rows = run_repl(&install, SLOW, &["slow", "\u{3}", "/exit"]);

    let body = between(&rows, "> slow", "> /exit");
    assert!(
        body.iter().any(|row| row.contains("(interrupted)")),
        "the interrupt is annotated, not announced as a sentence: {body:#?}"
    );
    assert!(
        !body.iter().any(|row| row.contains("never reached")),
        "the turn really was cut short: {body:#?}"
    );
    assert_eq!(
        body.first().map(String::as_str),
        Some(""),
        "still one blank after the line typed: {body:#?}"
    );
    assert_eq!(
        body.last().map(String::as_str),
        Some(""),
        "and one before the next prompt: {body:#?}"
    );
    assert!(
        !body
            .windows(2)
            .any(|pair| pair[0].is_empty() && pair[1].is_empty()),
        "and no stray blank from the notice's old leading newline: {body:#?}"
    );
}
