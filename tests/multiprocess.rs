// See the matching allow in `tests/acp.rs` for the rationale: integration tests panic on failure
// by design.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! What two `meka` processes do to each other.
//!
//! Every other "concurrent" test in this suite is in-process: `tests/serve.rs` drives two requests
//! at one server, `tests/acp.rs` two prompts at one session. Those cannot see the properties that
//! only exist between processes -- who holds a session's file lock, which host claims a scheduled
//! occurrence, whose credential write lands last -- and an audit found several of those properties
//! were not held at all. Six tests elsewhere are *named* for cross-process behaviour they
//! structurally cannot observe, and two of them enshrined the defect as intended behaviour.
//!
//! So these tests spawn real `meka` binaries against one shared `MEKA_DATA_DIR`. That is slow and
//! it is the point: nothing cheaper can fail when these guarantees break.
//!
//! # Shape
//!
//! [`Cluster`] owns the tempdir, the `config.toml` every process reads, and the database they
//! share. Processes come from [`Cluster::meka`] (a one-shot command) and [`Cluster::serve`] (a
//! long-lived server, waited on until it logs its bind address). Assertions read the database
//! directly through `rusqlite` rather than through meka, because what is being checked is what the
//! processes *left behind*, and asking meka would ask one of the processes under test.
//!
//! Turns run through the scripted mock provider (`MEKA_MOCK_PROVIDER=1`), so nothing here reaches
//! the network or needs a credential.

use std::{
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

/// A set of `meka` processes sharing one config and data directory.
struct Cluster {
    temp: tempfile::TempDir,
}

impl Cluster {
    /// Build the directories and the `config.toml` every process in the cluster reads, then open
    /// the database once so later inserts have a schema to insert into.
    ///
    /// `extra_config` is appended verbatim, which is how a test adds `[schedule]` or `[background]`
    /// settings without every other test carrying them.
    fn new(extra_config: &str) -> Self {
        let temp = tempfile::tempdir().expect("tempdir");
        let cluster = Self { temp };
        std::fs::create_dir_all(cluster.config_dir()).expect("create config dir");
        std::fs::write(
            cluster.config_dir().join("config.toml"),
            format!(
                r#"
[providers.mock]
type = "anthropic-messages"
model = "claude-sonnet-4-5"

[permissions]
default = "unrestricted"
enabled = ["read", "unrestricted"]
{extra_config}
"#
            ),
        )
        .expect("write config.toml");

        // Opening the store is a side effect of any subcommand that reads it, and this is the
        // cheapest one. Without it the first `rusqlite` connection a test opens would create an
        // empty file with no schema, and every later assertion would fail on a missing table
        // rather than on the behaviour under test.
        let opened = cluster
            .meka(&["session", "list"])
            .output()
            .expect("spawn meka session list");
        assert!(
            opened.status.success(),
            "could not open the store: {}",
            String::from_utf8_lossy(&opened.stderr)
        );
        cluster
    }

    fn config_dir(&self) -> PathBuf {
        self.temp.path().join("meka")
    }

    fn data_dir(&self) -> PathBuf {
        self.temp.path().join("data").join("meka")
    }

    fn database(&self) -> PathBuf {
        self.data_dir().join("meka.db")
    }

    /// A path inside the cluster's tempdir, for gate commands and probe files.
    fn path(&self, name: &str) -> PathBuf {
        self.temp.path().join(name)
    }

    /// A `meka` command pointed at this cluster, with the host's own config and data directories
    /// kept well clear.
    fn meka(&self, args: &[&str]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_meka"));
        command
            .args(args)
            .env("MEKA_CONFIG_DIR", self.config_dir())
            .env("MEKA_DATA_DIR", self.data_dir())
            .env("HOME", self.temp.path())
            .env("XDG_CONFIG_HOME", self.temp.path())
            .env("XDG_DATA_HOME", self.temp.path().join("data"))
            .env("MEKA_MOCK_PROVIDER", "1")
            .env("RUST_LOG", "meka=debug");
        command
    }

    /// Point the cluster's processes at a scripted set of provider rounds. One script serves every
    /// process; each gets its own queue, since each loads the file at startup.
    fn script(&self, rounds: serde_json::Value) -> &Self {
        std::fs::write(self.path("script.json"), rounds.to_string()).expect("write script");
        self
    }

    /// Run one `meka` process to completion.
    fn run(&self, args: &[&str]) -> std::process::Output {
        self.meka(args)
            .env("MEKA_MOCK_PROVIDER_SCRIPT", self.path("script.json"))
            .output()
            .unwrap_or_else(|error| panic!("spawn meka {:?}: {}", args, error))
    }

    /// Start one `meka` process and leave it running, for a test that needs to act while a turn is
    /// still going.
    ///
    /// Both pipes are drained by threads rather than left to fill. Nothing here reads them -- these
    /// tests assert against the database -- but a pipe nobody reads holds 64 KiB and then blocks
    /// the writer forever, and the caller is inside a `wait()` with no timeout under a `cargo test`
    /// with no timeout either. A `--oneshot` turn at `meka=debug` writes well under a kilobyte, so
    /// this is distance from a cliff rather than a fix, and the cliff is the kind that hangs a
    /// suite instead of failing it.
    fn start(&self, args: &[&str]) -> Child {
        let mut child = self
            .meka(args)
            .env("MEKA_MOCK_PROVIDER_SCRIPT", self.path("script.json"))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap_or_else(|error| panic!("spawn meka {:?}: {}", args, error));
        if let Some(stdout) = child.stdout.take() {
            drain(stdout);
        }
        if let Some(stderr) = child.stderr.take() {
            drain(stderr);
        }
        child
    }

    /// Spawn `meka serve` and return once it has logged its bind address.
    ///
    /// Each server gets its own ephemeral port and its own log file. No test here talks to the HTTP
    /// surface; `serve` is used because it is the only host that runs the scheduler for every
    /// session rather than one, which is what makes two of them contend.
    fn serve(&self, name: &str) -> ServeProcess {
        let port = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
            let port = listener.local_addr().expect("local_addr").port();
            drop(listener);
            port
        };
        // Written per server rather than shared, because the bind address differs and because a
        // test that fails wants to know which server did what.
        let config = self.config_dir().join(format!("{}.toml", name));
        let base = std::fs::read_to_string(self.config_dir().join("config.toml"))
            .expect("read base config");
        std::fs::write(
            &config,
            format!(
                "{base}\n[serve]\nbind = \"127.0.0.1:{port}\"\n\n[[serve.tokens]]\ntoken = \
                 \"sk_test_{name}\"\nscopes = [\"sessions:r\", \"sessions:w\"]\n"
            ),
        )
        .expect("write server config");
        // `MEKA_CONFIG_DIR` names a directory, not a file, so each server needs its own -- sharing
        // one would mean sharing a bind address.
        let config_dir = self.temp.path().join(format!("meka-{}", name));
        std::fs::create_dir_all(&config_dir).expect("create server config dir");
        std::fs::copy(&config, config_dir.join("config.toml")).expect("install server config");

        let mut child = Command::new(env!("CARGO_BIN_EXE_meka"))
            .arg("serve")
            .env("MEKA_CONFIG_DIR", &config_dir)
            .env("MEKA_DATA_DIR", self.data_dir())
            .env("HOME", self.temp.path())
            .env("MEKA_MOCK_PROVIDER", "1")
            .env("MEKA_MOCK_PROVIDER_SCRIPT", self.path("script.json"))
            .env("RUST_LOG", "meka=debug")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn meka serve");

        let stdout = child.stdout.take().expect("stdout");
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut sink = String::new();
            while reader.read_line(&mut sink).unwrap_or(0) > 0 {}
        });

        let stderr = child.stderr.take().expect("stderr");
        let (ready_sender, ready_receiver) = std::sync::mpsc::channel::<()>();
        let logs = std::thread::spawn(move || {
            let mut reader = BufReader::new(stderr);
            let mut line = String::new();
            let mut accumulated = String::new();
            let mut announced = false;
            loop {
                line.clear();
                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                    break;
                }
                accumulated.push_str(&line);
                if !announced && line.contains("listening on") {
                    let _ = ready_sender.send(());
                    announced = true;
                }
            }
            accumulated
        });

        let mut process = ServeProcess {
            child,
            logs: Some(logs),
        };
        if ready_receiver
            .recv_timeout(Duration::from_secs(30))
            .is_err()
        {
            panic!(
                "meka serve '{}' never logged its bind address:\n{}",
                name,
                process.stop()
            );
        }
        process
    }

    /// Read the database the cluster's processes share.
    fn read<T, F>(&self, read: F) -> T
    where
        F: FnOnce(&rusqlite::Connection) -> rusqlite::Result<T>,
    {
        let connection = rusqlite::Connection::open(self.database()).expect("open the store");
        read(&connection).expect("read the store")
    }

    /// The id of the one session the cluster has, for a test that made exactly one.
    fn only_session(&self) -> String {
        self.read(|connection| {
            connection.query_row("SELECT id FROM sessions", [], |row| row.get::<_, String>(0))
        })
    }

    fn session_count(&self) -> i64 {
        self.read(|connection| {
            connection.query_row("SELECT count(*) FROM sessions", [], |row| row.get(0))
        })
    }

    /// Every message role in the store, oldest first. The shape of a conversation two processes
    /// wrote into is the whole finding: `user, user, assistant, assistant` is what interleaving
    /// looks like, and the Anthropic Messages API refuses it outright, so the session is not merely
    /// muddled but unusable from that point on.
    fn message_roles(&self) -> Vec<String> {
        self.read(|connection| {
            connection
                .prepare("SELECT role FROM messages ORDER BY id ASC")?
                .query_map([], |row| row.get::<_, String>(0))?
                .collect()
        })
    }

    /// Every background task's status, oldest first.
    fn task_statuses(&self) -> Vec<String> {
        self.read(|connection| {
            connection
                .prepare("SELECT status FROM background_tasks ORDER BY started_at ASC")?
                .query_map([], |row| row.get::<_, String>(0))?
                .collect()
        })
    }

    /// Whether any lock file exists, which is the observable the audit measured directly: during a
    /// first turn the `locks/` directory held nothing at all.
    fn holds_a_session_lock(&self) -> bool {
        std::fs::read_dir(self.data_dir().join("locks"))
            .map(|entries| {
                entries.filter_map(Result::ok).any(|entry| {
                    entry
                        .path()
                        .extension()
                        .is_some_and(|extension| extension == "lock")
                        && entry.file_name() != "schema.lock"
                })
            })
            .unwrap_or(false)
    }

    /// Plant scheduled jobs directly, since creating one through meka needs an agent turn to ask
    /// for it and what is under test here is what happens to a job that already exists.
    ///
    /// `overdue_by` puts a job's occurrence that far in the past, so it is due the moment a
    /// scheduler looks. Pair it with a long `every` and exactly one occurrence exists for the whole
    /// life of the test: whatever advances the row puts the next one out of reach.
    ///
    /// All of them in one transaction, because a host polling every 200 ms can otherwise read a
    /// due list between two inserts. A list holding only some of the planted jobs is one where the
    /// interleaving the test is built around never happens, and the test then passes without
    /// exercising anything -- which is the worst outcome available to a race test.
    fn plant_jobs(&self, jobs: &[PlantedJob<'_>]) {
        // `chrono` is not a dev-dependency of the test crate, and SQLite renders the one shape
        // meka's own writer produces: UTC with an explicit offset.
        let mut connection = rusqlite::Connection::open(self.database()).expect("open the store");
        let transaction = connection.transaction().expect("begin");
        for job in jobs {
            let due = std::time::SystemTime::now() - job.overdue_by;
            let seconds = due
                .duration_since(std::time::UNIX_EPOCH)
                .expect("after the epoch")
                .as_secs();
            let due: String = transaction
                .query_row(
                    "SELECT strftime('%Y-%m-%dT%H:%M:%S', ?1, 'unixepoch') || '+00:00'",
                    [seconds],
                    |row| row.get(0),
                )
                .expect("render the timestamp");
            transaction
                .execute(
                    "INSERT INTO scheduled_jobs (id, session_id, kind, spec, prompt, \
                     gate_command, gate_fire, gate_permission, isolated, created_at, \
                     next_fire_at) \
                     VALUES (?1, ?2, 'every', ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                    rusqlite::params![
                        format!("job-{}", job.name),
                        job.session_id,
                        job.every,
                        job.prompt,
                        job.gate_command,
                        job.gate_command.map(|_| "on-success"),
                        job.gate_command.map(|_| "unrestricted"),
                        i64::from(job.isolated),
                        due,
                        due,
                    ],
                )
                .expect("plant the job");
        }
        transaction.commit().expect("commit");
    }
}

/// The arguments [`Cluster::plant_job`] takes, named rather than positional because five of the
/// seven are strings or bools and a call site of bare literals says nothing.
struct PlantedJob<'a> {
    name: &'a str,
    session_id: &'a str,
    every: &'a str,
    prompt: &'a str,
    /// `None` for an ungated job, which fires a turn every occurrence. A gate that exits non-zero
    /// under `on-success` declines instead, which claims the occurrence without spending a turn --
    /// the cheapest observable a scheduler test can have.
    gate_command: Option<&'a str>,
    isolated: bool,
    overdue_by: Duration,
}

/// A running `meka serve`, killed when the test drops it.
struct ServeProcess {
    child: Child,
    logs: Option<std::thread::JoinHandle<String>>,
}

impl ServeProcess {
    /// Kill the server and return everything it logged. Idempotent, so a test can read the logs
    /// mid-way and `Drop` can still run.
    fn stop(&mut self) -> String {
        let _ = self.child.kill();
        let _ = self.child.wait();
        match self.logs.take() {
            Some(handle) => handle.join().unwrap_or_default(),
            None => String::new(),
        }
    }
}

impl Drop for ServeProcess {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Read one of a child's pipes to EOF on its own thread and throw it away, so it cannot fill and
/// block the process writing to it. See [`Cluster::start`].
fn drain(pipe: impl std::io::Read + Send + 'static) {
    std::thread::spawn(move || {
        let mut reader = BufReader::new(pipe);
        let mut line = String::new();
        while reader.read_line(&mut line).unwrap_or(0) > 0 {
            line.clear();
        }
    });
}

/// A minimal script: one round that answers with a line of text and stops.
fn one_reply(text: &str) -> serde_json::Value {
    serde_json::json!([[
        {"kind": "text", "text": text},
        {"kind": "message_end", "stop_reason": "end_turn"},
    ]])
}

/// The same, but the provider takes its time, so another process can act mid-turn.
fn one_slow_reply(text: &str, thinking_for: Duration) -> serde_json::Value {
    serde_json::json!([[
        {"kind": "sleep", "ms": thinking_for.as_millis() as u64},
        {"kind": "text", "text": text},
        {"kind": "message_end", "stop_reason": "end_turn"},
    ]])
}

/// Wait until `check` holds, or give up. Polling rather than sleeping a fixed span: these tests
/// wait on another process reaching a state, and the only constant that is reliably long enough on
/// a loaded machine is one that makes the suite slow for everybody.
fn wait_until(what: &str, timeout: Duration, mut check: impl FnMut() -> bool) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if check() {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("timed out after {:?} waiting for {}", timeout, what);
}

/// A gate command that records one run and then declines, spelled for the host's own shell.
///
/// `execute_command` runs `powershell.exe -Command` on Windows and a POSIX shell elsewhere, and a
/// gate is an ordinary command. The POSIX spelling alone (`printf 'ran\n' >> …`) has no `printf` in
/// PowerShell, so on Windows the log stayed empty, [`gate_runs`] read zero forever, and the test
/// timed out waiting rather than failing on the claim semantics it exists to measure. `sleep` needs
/// no such treatment: PowerShell aliases it to `Start-Sleep`.
fn record_a_run_then_decline(log: &Path) -> String {
    if cfg!(windows) {
        format!(
            "Add-Content -LiteralPath '{}' -Value 'ran'; exit 1",
            log.display()
        )
    } else {
        format!("printf 'ran\\n' >> '{}'; exit 1", log.display())
    }
}

/// How many lines a gate command has appended to its log, which is how many times it ran.
fn gate_runs(log: &Path) -> usize {
    std::fs::read_to_string(log)
        .map(|text| text.lines().filter(|line| !line.is_empty()).count())
        .unwrap_or(0)
}

/// A session is locked from the instant its row exists, not from the end of its first turn.
///
/// The measured hole: for the whole of a first turn `locks/` held nothing, because the REPL claimed
/// the lock in its post-turn block and `--oneshot` never claimed one at all. A second `meka -c`
/// started in that window attached to the same session and wrote into it, ten times out of ten,
/// with rc=0 and no warning on either side. What came out was `user, user, assistant, assistant`,
/// which the Anthropic Messages API rejects for non-alternating roles -- so the session was left
/// permanently unusable, and nothing said so.
#[test]
fn a_session_is_locked_while_its_first_turn_runs() {
    let cluster = Cluster::new("");
    cluster.script(one_slow_reply("first", Duration::from_secs(5)));

    let mut first = cluster.start(&["--oneshot", "the first prompt"]);
    wait_until("the session row to exist", Duration::from_secs(20), || {
        cluster.session_count() >= 1
    });
    assert!(
        cluster.holds_a_session_lock(),
        "the lock must exist by the time the row does, not by the time the turn ends"
    );

    // Exactly what the audit ran: a second invocation, mid-turn, continuing the same session.
    let second = cluster.run(&["-c", "--oneshot", "the second prompt"]);
    let refusal = String::from_utf8_lossy(&second.stderr).to_string();

    let status = first.wait().expect("the first process exits");
    assert!(status.success(), "the first turn must still succeed");

    assert!(
        !second.status.success(),
        "a second process must be refused, not admitted: {}",
        refusal
    );
    assert!(
        refusal.contains("already attached by another process"),
        "and refused for the right reason: {}",
        refusal
    );
    assert_eq!(
        cluster.message_roles(),
        vec!["user".to_string(), "assistant".to_string()],
        "one process's turn, not two interleaved"
    );
}

/// `meka session delete` against a conversation another process is having.
///
/// The measured behaviour: rc=0 with no output whatsoever -- the count goes through
/// `tracing::info!`, invisible at the default level -- while the row and its messages cascaded away
/// and the live REPL carried on as though nothing had happened, until its next turn ran against the
/// provider and *then* failed on a foreign-key violation. Tokens spent, answer lost, and every
/// later turn failing the same way with no recovery.
#[test]
fn a_second_process_cannot_delete_a_session_in_use() {
    let cluster = Cluster::new("");
    cluster.script(one_slow_reply("thinking", Duration::from_secs(5)));

    let mut first = cluster.start(&["--oneshot", "a question"]);
    wait_until("the session row to exist", Duration::from_secs(20), || {
        cluster.session_count() >= 1
    });
    let session = cluster.only_session();

    let deleted = cluster.run(&["session", "delete", &session]);
    let refusal = String::from_utf8_lossy(&deleted.stderr).to_string();

    assert!(
        !deleted.status.success(),
        "deleting a session another process is mid-turn on must fail, not exit 0 in silence: {}",
        refusal
    );
    assert!(
        refusal.contains("already attached by another process"),
        "and say why: {}",
        refusal
    );
    assert_eq!(
        cluster.session_count(),
        1,
        "the conversation must survive the attempt"
    );

    let status = first.wait().expect("the first process exits");
    assert!(
        status.success(),
        "and finish its turn without a foreign-key violation"
    );
    assert_eq!(cluster.message_roles(), vec![
        "user".to_string(),
        "assistant".to_string()
    ]);
}

/// The consequence the first-turn hole had that costs work rather than coherence.
///
/// A `--oneshot` run that detaches a command stays alive waiting for it, and used to hold no lock
/// at any point. A second `meka -c --oneshot` therefore opened the same session, swept the
/// genuinely-running task to `interrupted`, and told its own model the work had died -- and when
/// the command really finished, `finish_background_task`'s `AND status = 'running'` guard threw
/// the real outcome away. Both halves were silent: two rc=0 processes, no warning either side.
#[test]
fn a_second_process_cannot_sweep_a_running_background_task() {
    let cluster = Cluster::new("[background]\nenabled = true\n");
    let proof = cluster.path("finished");
    cluster.script(serde_json::json!([
        [
            {"kind": "tool_use_start", "id": "tu_1", "name": "execute_command"},
            {"kind": "tool_use_end", "input": {
                "command": format!("sleep 4; printf done > '{}'", proof.display()),
                "background": true,
            }},
            {"kind": "message_end", "stop_reason": "tool_use"},
        ],
        [
            {"kind": "text", "text": "started it"},
            {"kind": "message_end", "stop_reason": "end_turn"},
        ],
    ]));

    let mut first = cluster.start(&["--oneshot", "kick off the build"]);
    wait_until("the task to be running", Duration::from_secs(30), || {
        cluster.task_statuses() == vec!["running".to_string()]
    });

    let second = cluster.run(&["-c", "--oneshot", "anything"]);
    assert!(
        !second.status.success(),
        "the second process must be refused while the first still owns the session"
    );

    let status = first.wait().expect("the first process exits");
    assert!(status.success(), "the first process must finish its wait");
    assert!(
        proof.exists(),
        "the command really did run to completion, which is what makes the row's status a claim \
         about reality"
    );
    assert_eq!(
        cluster.task_statuses(),
        vec!["completed".to_string()],
        "the outcome must be recorded, not discarded because someone else retired the row"
    );
}

/// `meka session fork` must not copy a conversation that is being written.
///
/// `Agent::run_turn` persists the user message eagerly, before the provider answers, so a fork
/// taken mid-turn copies a dangling user row. The copy then reads `user, user, assistant` from its
/// first resumed turn onward -- permanently, since nothing repairs it. This was 10/10 and 30/30
/// deterministic across two independent runs, not a race. `meka session rewind` was never affected
/// because it locks the source first; fork did not, and neither did `export`.
#[test]
fn a_conversation_being_written_cannot_be_forked_out_from_under_itself() {
    let cluster = Cluster::new("");
    cluster.script(one_slow_reply("answer", Duration::from_secs(5)));

    let mut first = cluster.start(&["--oneshot", "a question"]);
    wait_until("the session row to exist", Duration::from_secs(20), || {
        cluster.session_count() >= 1
    });
    let session = cluster.only_session();

    let forked = cluster.run(&["session", "fork", &session]);
    let refusal = String::from_utf8_lossy(&forked.stderr).to_string();
    assert!(
        !forked.status.success(),
        "forking a session mid-turn copies a half-written conversation: {}",
        refusal
    );
    assert!(
        refusal.contains("cannot be copied while it is being written"),
        "and the refusal has to say why: {}",
        refusal
    );

    // The same rule for the other door that copies a conversation.
    let exported = cluster.run(&["session", "export", &session, "-o", "-"]);
    assert!(
        !exported.status.success(),
        "nor may an export snapshot it: {}",
        String::from_utf8_lossy(&exported.stderr)
    );

    assert_eq!(
        cluster.session_count(),
        1,
        "and no copy was left behind by either"
    );
    let status = first.wait().expect("the first process exits");
    assert!(status.success());

    // Once the turn is done the conversation is whole, and both doors open.
    let forked = cluster.run(&["session", "fork", &session]);
    assert!(
        forked.status.success(),
        "a settled conversation forks: {}",
        String::from_utf8_lossy(&forked.stderr)
    );
    assert_eq!(cluster.session_count(), 2);
}

/// How long the decoy job's gate blocks the host that claimed it. Long enough that the other host
/// has tens of poll ticks inside the window, short enough that a test costs seconds.
const DECOY_GATE: Duration = Duration::from_secs(3);

/// Plant a decoy alongside the job under test, so the two hosts *must* contend for it.
///
/// Left to their own tickers, two servers rarely collide: the first to tick claims the occurrence
/// and puts the next one an hour out, and the second finds nothing due. Waiting for their phases to
/// align by chance is what makes a race test flaky, so this arranges the collision instead.
///
/// The decoy is more overdue than the real job, and `list_due_scheduled_jobs` orders by fire time,
/// so every host takes it first. Exactly one host can claim it; that host then sits inside the
/// decoy's gate command for [`DECOY_GATE`] while still holding the due list it read *before* the
/// claim -- a list in which the real job is unclaimed. The other host takes the real job during
/// that window. When the blocked host reaches the real job it is holding a copy of a row that has
/// moved, which is the exact interleaving two servers hit by chance in production.
///
/// Both rows land in one transaction, so no host can ever see a due list with one of them in it.
fn plant_a_decoy_and_the_job(cluster: &Cluster, session: &str, job: PlantedJob<'_>) {
    let decoy = format!("sleep {}; exit 1", DECOY_GATE.as_secs());
    cluster.plant_jobs(&[
        PlantedJob {
            name: "decoy",
            session_id: session,
            every: "1h",
            prompt: "never delivered: the gate declines",
            gate_command: Some(&decoy),
            isolated: false,
            // More overdue than the job under test, so it sorts first in every host's due list.
            overdue_by: Duration::from_secs(600),
        },
        job,
    ]);
}

/// Two servers, one overdue occurrence, one gate execution.
///
/// `every = "1h"` is what makes the count exact rather than statistical: the job is due once, and
/// whichever host claims it puts the next occurrence an hour out, so a second gate run can only be
/// a second claim of the *same* occurrence. Before the claim was a compare-and-swap both hosts ran
/// it -- 64 microseconds apart in the audit that found this -- and for an ungated job that is two
/// agent turns and two lots of spend, hourly, forever.
#[test]
fn two_servers_do_not_both_claim_one_scheduled_occurrence() {
    let cluster = Cluster::new("[schedule]\npoll_interval = \"200ms\"\n");
    cluster.script(one_reply("done"));
    let started = cluster.run(&["--oneshot", "make a session"]);
    assert!(
        started.status.success(),
        "the first turn failed: {}",
        String::from_utf8_lossy(&started.stderr)
    );
    let session = cluster.only_session();

    let mut first = cluster.serve("one");
    let mut second = cluster.serve("two");

    let log = cluster.path("gate.log");
    let gate = record_a_run_then_decline(&log);
    plant_a_decoy_and_the_job(&cluster, &session, PlantedJob {
        name: "shared",
        session_id: &session,
        every: "1h",
        prompt: "check the thing",
        gate_command: Some(&gate),
        isolated: false,
        overdue_by: Duration::from_secs(60),
    });

    wait_until(
        "the occurrence to be claimed",
        Duration::from_secs(30),
        || gate_runs(&log) >= 1,
    );
    // Past the decoy's gate, so the host it blocked has reached the job under test and been told
    // no. Without that answer it runs the gate a second time, here.
    std::thread::sleep(DECOY_GATE + Duration::from_secs(2));
    let logs = format!(
        "--- server one ---\n{}\n--- server two ---\n{}",
        first.stop(),
        second.stop()
    );

    assert_eq!(
        gate_runs(&log),
        1,
        "one occurrence must produce one gate execution, not one per host\n{}",
        logs
    );
}

/// The same arbitration for an `isolated` job, which reaches it by a different route: `runnable`
/// exempts isolated jobs from the session-lock probe entirely, so nothing before the claim declines
/// either host. Each fire creates a *new top-level session*, so a duplicate is not a repeated turn
/// that vanishes but a permanent extra row in `meka session list`.
#[test]
fn two_servers_do_not_both_run_one_isolated_job() {
    let cluster = Cluster::new("[schedule]\npoll_interval = \"200ms\"\n");
    cluster.script(one_reply("isolated run"));
    let started = cluster.run(&["--oneshot", "make a session"]);
    assert!(started.status.success(), "the first turn failed");
    let session = cluster.only_session();
    assert_eq!(cluster.session_count(), 1);

    let mut first = cluster.serve("one");
    let mut second = cluster.serve("two");

    plant_a_decoy_and_the_job(&cluster, &session, PlantedJob {
        name: "isolated",
        session_id: &session,
        every: "1h",
        prompt: "run somewhere else",
        gate_command: None,
        isolated: true,
        overdue_by: Duration::from_secs(60),
    });

    wait_until(
        "the isolated run to finish",
        Duration::from_secs(30),
        || cluster.session_count() >= 2,
    );
    std::thread::sleep(DECOY_GATE + Duration::from_secs(2));
    let logs = format!(
        "--- server one ---\n{}\n--- server two ---\n{}",
        first.stop(),
        second.stop()
    );

    assert_eq!(
        cluster.session_count(),
        2,
        "one occurrence must produce one isolated session, not one per host\n{}",
        logs
    );
}
