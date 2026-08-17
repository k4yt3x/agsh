//! `execute_command` tool. Spawns a shell process, optionally constrained by the platform sandbox
//! when permissions are read-only, and streams stdout/stderr back to the agent as it arrives.
//!
//! The sandbox is Landlock or Bubblewrap on Linux (see [`crate::sandbox`] for which is preferred),
//! `sandbox-exec` on macOS, and a low-integrity token on Windows, which is spawned through
//! `CreateProcessAsUserW` rather than `tokio::process` because the standard library offers no hook
//! for injecting one.

use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::{Tool, ToolOutput, util::require_str};
use crate::{
    error::{MekaError, Result},
    permission::Permission,
    provider::ToolDefinition,
};

/// Default `timeout_ms` applied when the caller doesn't pass one. Single source of truth for both
/// the parameter unwrap and the description shown to the agent.
const DEFAULT_TIMEOUT_MS: u64 = 30_000;

pub(super) struct ExecuteCommandTool {
    pub sandbox_capability: crate::sandbox::SandboxCapability,
    /// Backend chosen in config (or auto-resolved). Read only by the Linux hard-error message in
    /// [`Tool::execute`]; on macOS / Windows the field is populated but unused, so suppress the
    /// "never read" lint there without hiding regressions on Linux.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub sandbox_backend: crate::config::SandboxBackend,
    /// Probe outcome for [`Self::sandbox_backend`]. Drives the hard-error path in read mode when
    /// the backend isn't usable (bwrap missing, user namespaces denied, etc.). When `Ok(_)`,
    /// [`Self::sandbox_capability`] mirrors the inner capability and the spawn path runs normally.
    pub backend_probe: crate::sandbox::BackendProbe,
    pub shared_permission: crate::permission::SharedPermission,
    pub sandbox_enabled: bool,
    pub cwd: crate::workspace::SharedCwd,
    /// Sink for [`crate::frontend::FrontendEvent::ToolCallOutputDelta`], so a frontend can show
    /// output as the command produces it rather than only once it exits.
    pub frontend: Arc<dyn crate::frontend::Frontend>,
}

#[async_trait]
impl Tool for ExecuteCommandTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "execute_command".to_string(),
            description: "Execute a shell command and return its output. On Unix the \
                command runs via `sh -c <command>`. POSIX `$VAR` expansion applies; \
                quote with single quotes or `\\$` to pass a literal `$`. On Windows \
                the command runs via `powershell.exe -Command <command>`. Use \
                PowerShell syntax directly (e.g. `$var = ...`, `$env:PATH`); do NOT \
                wrap with another `powershell -Command` or the outer PowerShell will \
                expand your inner `$var` references to empty strings. In read mode \
                the command runs in a read-only sandbox where filesystem writes are \
                blocked. Multiple independent execute_command calls in one assistant \
                message run in parallel; use this for read-only commands and \
                serialize anything that mutates shared state (files, git, packages)."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": format!(
                            "Timeout in milliseconds. Defaults to {} ({} seconds).",
                            DEFAULT_TIMEOUT_MS,
                            DEFAULT_TIMEOUT_MS / 1000,
                        )
                    },
                    "scratchpad": {
                        "type": "string",
                        "description": "If provided, save the output to the scratchpad under this name instead of returning it inline."
                    }
                },
                "required": ["command"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        if self.sandbox_enabled
            && !matches!(
                self.sandbox_capability,
                crate::sandbox::SandboxCapability::Unavailable
            )
        {
            Permission::Read
        } else {
            Permission::Write
        }
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let command = require_str(&input, "command", "execute_command")?;
        let timeout_ms = input["timeout_ms"].as_u64().unwrap_or(DEFAULT_TIMEOUT_MS);
        let permission = self.shared_permission.get();
        let sandboxed = self.sandbox_enabled && permission != Permission::Write;

        if sandboxed {
            // Configured backend isn't usable on this host. Hard-error with the specific reason so
            // the model can surface it via `render::render_error` rather than treat the failure as
            // a tool result it could try to recover from.
            if let Some(reason) = crate::sandbox::backend_unavailable_reason(&self.backend_probe) {
                // `sandbox_backend` is Linux-only; on other platforms there's nothing to
                // reconfigure. The only escape hatch is write mode.
                #[cfg(target_os = "linux")]
                let message = format!(
                    "configured sandbox backend ({}) is unavailable: {}. \
                     Switch to write mode (Shift+Tab) to run shell commands \
                     without a sandbox, or update [shell].sandbox_backend in \
                     your config.",
                    self.sandbox_backend, reason
                );
                #[cfg(not(target_os = "linux"))]
                let message = format!(
                    "sandbox is unavailable: {}. Switch to write mode \
                     (Shift+Tab) to run shell commands without a sandbox.",
                    reason
                );
                return Err(MekaError::ToolExecution {
                    tool_name: "execute_command".to_string(),
                    message,
                });
            }
        }

        // Windows + sandboxed: spawn directly via CreateProcessAsUserW with a Low-integrity token.
        // This path can't go through tokio::process because the stdlib gives no hook for injecting
        // a custom token.
        #[cfg(windows)]
        if sandboxed
            && matches!(
                self.sandbox_capability,
                crate::sandbox::SandboxCapability::LowIntegrity
            )
        {
            let relay = OutputRelay::for_current_call(&self.frontend);
            return run_windows_low_integrity(&command, timeout_ms, cancellation, relay).await;
        }

        #[cfg(windows)]
        let mut command_builder = {
            // Wrap with the UTF-8 output prelude so pipe output matches what the sandboxed path
            // produces; both on Rust's side this is decoded as UTF-8. Without the wrap, PowerShell
            // 5.1 defaults to the legacy console code page and mangles non-ASCII characters into
            // `?`.
            let wrapped = crate::sandbox::wrap_command_with_utf8_output(&command);
            let mut cmd = tokio::process::Command::new("powershell.exe");
            cmd.arg("-NoProfile")
                .arg("-NonInteractive")
                .arg("-Command")
                .arg(&wrapped);
            cmd
        };

        #[cfg(target_os = "macos")]
        let mut command_builder = if sandboxed
            && matches!(
                self.sandbox_capability,
                crate::sandbox::SandboxCapability::SandboxExec
            ) {
            let mut cmd = tokio::process::Command::new(crate::sandbox::SANDBOX_EXEC_PATH);
            cmd.arg("-p")
                .arg(crate::sandbox::SANDBOX_PROFILE_READONLY)
                .arg("sh")
                .arg("-c")
                .arg(&command);
            cmd
        } else {
            let mut cmd = tokio::process::Command::new("sh");
            cmd.arg("-c").arg(&command);
            cmd
        };

        #[cfg(target_os = "linux")]
        let mut command_builder = if sandboxed
            && let crate::sandbox::SandboxCapability::Bubblewrap { bwrap_path } =
                &self.sandbox_capability
        {
            // Bubblewrap path: `--ro-bind /` enforces "no writes", `--unshare-*` cuts off PID /
            // user / UTS / IPC views, tmpfs masks over `/run`, `/tmp`, `/var/tmp`, and
            // `$XDG_RUNTIME_DIR` make the dbus and systemd-user sockets unreachable so the agent
            // can't `dbus-send` state-changing methods. `--unshare-net` is intentionally absent;
            // network must stay open for `curl | pdftotext` and similar pipelines.
            let mut cmd = tokio::process::Command::new(bwrap_path);
            cmd.args([
                "--new-session",
                "--die-with-parent",
                "--ro-bind",
                "/",
                "/",
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--tmpfs",
                "/tmp",
                "--tmpfs",
                "/run",
                "--tmpfs",
                "/var/tmp",
                "--unshare-user",
                "--unshare-pid",
                "--unshare-uts",
                "--unshare-ipc",
                "--unshare-cgroup-try",
            ]);
            if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR")
                && std::path::Path::new(&xdg).is_absolute()
            {
                cmd.arg("--tmpfs").arg(&xdg);
            }
            cmd.arg("--").arg("sh").arg("-c").arg(&command);
            cmd
        } else {
            let mut cmd = tokio::process::Command::new("sh");
            cmd.arg("-c").arg(&command);
            cmd
        };

        // Unix: place the child in its own session/process group via `setsid` so timeouts and
        // cancellation can kill the whole tree (including backgrounded grandchildren such as
        // `(sleep 3600 &)`) via `kill(-pgid, …)`. On Linux the Landlock setup runs in the same
        // closure; `pre_exec` overwrites rather than chains, so we fold both steps into one.
        // Landlock is applied ONLY for the Landlock capability; under Bubblewrap, the `--ro-bind /`
        // mount layer already enforces "no writes" and layering both is fragile to test across
        // kernels.
        #[cfg(unix)]
        {
            #[cfg(target_os = "linux")]
            let landlock_abi: Option<i32> = if sandboxed {
                if let crate::sandbox::SandboxCapability::Landlock { abi_version } =
                    self.sandbox_capability
                {
                    Some(abi_version)
                } else {
                    None
                }
            } else {
                None
            };

            unsafe {
                command_builder.pre_exec(move || {
                    // SAFETY: `setsid(2)` is async-signal-safe and has no preconditions beyond "the
                    // caller isn't already a process group leader", which is guaranteed for a
                    // freshly forked child process.
                    if libc::setsid() == -1 {
                        return Err(std::io::Error::last_os_error());
                    }
                    #[cfg(target_os = "linux")]
                    if let Some(abi) = landlock_abi {
                        crate::sandbox::apply_landlock_readonly(abi)
                            .map_err(std::io::Error::from_raw_os_error)?;
                    }
                    #[cfg(not(target_os = "linux"))]
                    let _ = (); // landlock_abi unused on non-Linux Unix
                    Ok(())
                });
            }
        }

        // Scrub env before spawn so secrets in the parent process (`ANTHROPIC_API_KEY`, `AWS_*`,
        // `GITHUB_TOKEN`, …) can't ride along into the read-mode child. Sandboxes block writes/IPC
        // but leave the network open, so leaked env is a live exfil vector under prompt injection.
        // Write mode keeps the parent env (trusted-operation path). The Windows sandboxed branch
        // applies the same scrub inside `spawn_low_integrity_command`.
        #[cfg(unix)]
        if sandboxed {
            command_builder.env_clear();
            command_builder.envs(crate::sandbox::sandbox_child_env());
        }

        // Resolve commands against the agent's per-session cwd, not the process cwd. `/cd` mutates
        // the agent's cwd; this is how it actually reaches the child.
        command_builder.current_dir(crate::workspace::cwd_snapshot(&self.cwd));

        let mut child = command_builder
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "execute_command".to_string(),
                message: format!("failed to spawn command: {}", error),
            })?;

        let timeout_duration = std::time::Duration::from_millis(timeout_ms);

        // Drain stdout/stderr on dedicated tasks that start *before* the wait.
        // `tokio::process::Child::wait()` does not read the pipes; a child writing past the OS pipe
        // buffer (~64 KiB) would block in `write()`, `wait()` would never return, and the call
        // would spuriously hit the timeout below. After the child's process group exits the pipe
        // write ends close and the drains hit EOF.
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let relay = OutputRelay::for_current_call(&self.frontend);
        let stdout_task = tokio::spawn({
            let relay = relay.clone();
            async move { read_to_string_best_effort(stdout, relay).await }
        });
        let stderr_task = tokio::spawn({
            let relay = relay.clone();
            async move { read_to_string_best_effort(stderr, relay).await }
        });

        // wait_with_output() consumes the child, so use wait() + manual stdout/stderr reading
        // instead to allow kill on cancellation.
        tokio::select! {
            _ = cancellation.cancelled() => {
                kill_child_tree(&mut child).await;
                stdout_task.abort();
                stderr_task.abort();
                Err(MekaError::Interrupted)
            }
            _ = tokio::time::sleep(timeout_duration) => {
                kill_child_tree(&mut child).await;
                stdout_task.abort();
                stderr_task.abort();
                // Timed out means we killed it, so report the kill rather than inventing an exit
                // code; a frontend rendering a terminal shows "terminated" instead of "exit 0".
                Ok(ToolOutput::text(
                    format!("Command timed out after {}ms", timeout_ms),
                    true,
                )
                .with_metadata(timed_out_exit_metadata()))
            }
            status = child.wait() => {
                let status = status.map_err(|error| MekaError::ToolExecution {
                    tool_name: "execute_command".to_string(),
                    message: format!("failed to wait for command: {}", error),
                })?;

                let exit_code = status.code().unwrap_or(-1);
                // A backgrounded grandchild can keep the pipe open past the direct child's exit;
                // cap the drain so the tool call can't hang, attaching a truncation note if the cap
                // fires.
                let (stdout_content, stdout_timed_out) =
                    join_drain_with_timeout(stdout_task, DRAIN_TIMEOUT).await;
                let (stderr_content, stderr_timed_out) =
                    join_drain_with_timeout(stderr_task, DRAIN_TIMEOUT).await;

                // No output-length truncation here: the agent layer's `persist_oversized_results`
                // auto-persists any oversized result to the scratchpad losslessly. Truncating here
                // would corrupt binary-in-base64 pipelines (see #1 in the trial feedback).
                let mut output =
                    assemble_command_output(&stdout_content, &stderr_content, exit_code);
                if stdout_timed_out || stderr_timed_out {
                    append_drain_truncation_note(&mut output, stdout_timed_out, stderr_timed_out);
                }
                Ok(output.with_metadata(command_exit_metadata(&status)))
            }
        }
    }
}

/// Terminate the child and, on Unix, its entire process group. Called on timeout and on
/// cancellation. On Unix we rely on the `setsid()` done in `pre_exec`: the child's pid is also its
/// pgid, so `kill(-pgid, …)` reaches every backgrounded descendant it spawned (e.g. `(sleep 3600
/// &)` survives a plain `child.kill()` but is caught here). The fallback `child.kill().await` is a
/// no-op on Unix once the group has been signaled but still the right primitive on Windows.
async fn kill_child_tree(child: &mut tokio::process::Child) {
    #[cfg(unix)]
    {
        if let Some(pid) = child.id() {
            let pgid = pid as libc::pid_t;
            // SAFETY: `kill(2)` is always safe to call; it just returns an error if the target is
            // gone. Sending to `-pgid` targets the whole process group. Errors here usually mean
            // the group already exited; log at debug so an unkillable group still leaves a trail
            // without spamming default verbosity.
            let term_result = unsafe { libc::kill(-pgid, libc::SIGTERM) };
            if term_result != 0 {
                tracing::debug!(
                    "libc::kill(-{}, SIGTERM) failed: {}",
                    pgid,
                    std::io::Error::last_os_error()
                );
            }
            // Brief grace period so well-behaved children can shut down cleanly before SIGKILL
            // lands.
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            let kill_result = unsafe { libc::kill(-pgid, libc::SIGKILL) };
            if kill_result != 0 {
                tracing::debug!(
                    "libc::kill(-{}, SIGKILL) failed: {}",
                    pgid,
                    std::io::Error::last_os_error()
                );
            }
        }
    }
    if let Err(error) = child.kill().await {
        tracing::debug!("failed to kill child process: {}", error);
    }
}

/// Upper bound on draining a child's stdout/stderr after it has exited. A backgrounded grandchild
/// that inherited the pipe write handle can keep the pipe open past the direct child's exit; rather
/// than block the tool call we cap the drain, abort it, and attach a truncation note.
const DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Forwards a running command's output to the frontend as it is produced. Cloned into both drain
/// tasks so stdout and stderr land in one stream in arrival order, which is what a terminal shows.
///
/// `None` when there is no tool call to correlate against (direct tool construction in tests), in
/// which case draining behaves exactly as it did before deltas existed.
#[derive(Clone)]
struct OutputRelay {
    frontend: Arc<dyn crate::frontend::Frontend>,
    tool_call_id: String,
}

impl OutputRelay {
    /// The id has to be captured here rather than inside the drain task: it lives in a task-local
    /// scoped by `Agent::resolve_and_execute_tool`, and `tokio::spawn` does not inherit
    /// task-locals.
    fn for_current_call(frontend: &Arc<dyn crate::frontend::Frontend>) -> Option<Self> {
        crate::tools::current_tool_call_id().map(|tool_call_id| Self {
            frontend: Arc::clone(frontend),
            tool_call_id,
        })
    }

    async fn send(&self, chunk: String) {
        self.frontend
            .emit(crate::frontend::FrontendEvent::ToolCallOutputDelta {
                id: self.tool_call_id.clone(),
                chunk,
            })
            .await;
    }
}

/// How much of one stream meka will hold in the turn's memory before moving it to a file.
///
/// There is no cap on how much a command may print, and there should not be: `execute_command` was
/// deliberately changed to stop truncating at 30 KB. But the drain used to accumulate one
/// unbounded `Vec<u8>`, and a command that writes faster than the turn ends (measured at 2.2 GB/s
/// for `cat /dev/zero`) took the process with it. Past this point the bytes go to disk and the
/// result names the file, so the output is still complete and still reachable, just not resident.
const MAX_RESIDENT_OUTPUT_BYTES: usize = 8 * 1024 * 1024;

/// How much of each end of an overflowing stream stays in the result inline. Enough that the model
/// can see how the command started and how it ended without opening the capture.
///
/// Read alongside [`crate::background::OUTCOME_INLINE_LIMIT`], which is eight times smaller and
/// keeps only the head. A backgrounded command's *delivered outcome* therefore shows less than its
/// tool result would have, and shows a different part of it. See that constant for why the
/// asymmetry is intended and what it costs.
const OUTPUT_WINDOW_BYTES: usize = 32 * 1024;

/// Where an overflowing stream's bytes are going.
///
/// Three states rather than an `Option`, because "not needed yet" and "tried and failed" call for
/// opposite responses at the next chunk and an `Option` collapsed them into one. A failure read as
/// "not capturing yet", so the ceiling was re-crossed 8 MiB later and the whole opening sequence
/// ran again: a second file, and `head` overwritten with a slice from the middle of the stream that
/// the result then presented as the beginning.
enum Capture {
    /// The stream still fits inline. The only state from which capture can begin.
    NotNeeded,
    Writing(std::path::PathBuf, tokio::fs::File),
    /// Capture was attempted and could not be relied on. Terminal: the notice discloses the loss
    /// rather than naming a file, and no second attempt is made.
    Failed,
}

/// Keep only the last `limit` bytes, dropping from the front. Returns how many were dropped.
///
/// The count is what keeps the relay cursor aligned. `relayed` is an index into this buffer, so a
/// trim has to shift it by exactly what was removed. Clamping it to the new length instead marked
/// the carried incomplete-UTF-8 bytes as already sent; the next chunk then began on continuation
/// bytes, `valid_up_to()` returned 0, and a whole read vanished from the live stream. It
/// self-corrected only when a read happened to end on a character boundary, so ASCII output never
/// showed it.
fn trim_front_to(buffer: &mut Vec<u8>, limit: usize) -> usize {
    if buffer.len() > limit {
        let dropped = buffer.len() - limit;
        buffer.drain(..dropped);
        dropped
    } else {
        0
    }
}

#[cfg(test)]
thread_local! {
    /// Test hook making the capture-failure arms reachable.
    ///
    /// There is no other way in. Both directories `capture_path` can pick fall back to each other
    /// by design, so no environment a test could arrange reliably fails the open, and those arms
    /// are exactly where the state machine used to go wrong. Thread-local rather than a static so a
    /// test that sets it cannot disturb the capture tests running beside it; `#[tokio::test]` is
    /// single-threaded, so the value is visible across the awaits below.
    static FORCE_CAPTURE_FAILURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// How long a command-output capture survives before the next overflow sweeps it.
const CAPTURE_RETENTION: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);

/// Delete captures older than [`CAPTURE_RETENTION`].
///
/// Runs on the overflow path rather than on a timer: an overflow is rare, so this costs a directory
/// read on the one occasion something is about to be written anyway, and a meka that never
/// overflows never needs the sweep. Every failure is ignored -- a capture that cannot be removed is
/// not a reason to fail the command whose output is about to be written beside it.
///
/// Matches only meka's own names, so a file someone else left in a shared temp directory is not
/// meka's to delete.
fn sweep_stale_captures(directory: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let is_ours = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| {
                (name.starts_with("command-output-") || name.starts_with("meka-command-output-"))
                    && name.ends_with(".log")
            });
        if !is_ours {
            continue;
        }
        let stale = entry
            .metadata()
            .and_then(|metadata| metadata.modified())
            .map(|modified| modified.elapsed().is_ok_and(|age| age > CAPTURE_RETENTION))
            .unwrap_or(false);
        if stale && let Err(error) = std::fs::remove_file(&path) {
            tracing::debug!(
                "could not remove stale capture '{}': {}",
                path.display(),
                error
            );
        }
    }
}

/// Where an overflowing stream is captured. One file per stream per command, named unguessably so
/// two concurrent commands cannot collide and a predictable name cannot be pre-created by something
/// else.
///
/// The cache directory rather than `std::env::temp_dir()`, because on most Linux systems `/tmp` is
/// a tmpfs: capturing there would move the bytes out of the heap and straight back into RAM, which
/// is the thing the capture exists to avoid.
///
/// Sweeps captures older than [`CAPTURE_RETENTION`] on the way past. These files are named in a
/// tool result the model has already read, so deleting one is deleting something a resumed session
/// may still refer to -- but they are 8 MiB or more each and nothing else ever removes them, so a
/// machine that runs long builds accumulates them until the disk notices. A day is well past the
/// point where the conversation that produced one is still acting on it.
fn capture_path() -> std::path::PathBuf {
    #[cfg(test)]
    if FORCE_CAPTURE_FAILURE.with(std::cell::Cell::get) {
        // A parent that does not exist, so `File::create` fails on every platform.
        return std::env::temp_dir()
            .join(format!("meka-absent-{}", uuid::Uuid::new_v4()))
            .join("capture.log");
    }

    // `MEKA_DATA_DIR` first, so a run isolated to a scratch directory keeps its captures there too
    // rather than dropping them in the real user's cache.
    //
    // Empty and relative values are rejected here rather than assumed away. The comment that used
    // to sit here claimed `default_database_path` guarantees absoluteness; it does not -- it
    // *warns* and falls back to the platform data dir (src/session.rs), so meka starts normally
    // with its database in the right place while a relative or empty value reached this join
    // and scattered capture files, holding whole command outputs, under whatever directory meka
    // happened to start in. Same guard, same reason, applied to the sibling that missed it.
    let directory = std::env::var_os("MEKA_DATA_DIR")
        .map(std::path::PathBuf::from)
        .filter(|path| {
            if path.as_os_str().is_empty() || !path.is_absolute() {
                tracing::warn!(
                    "MEKA_DATA_DIR '{}' is not an absolute path; keeping command-output captures \
                     in the cache directory instead",
                    path.display()
                );
                return false;
            }
            true
        })
        .map(|path| path.join("command-output"))
        .or_else(|| dirs::cache_dir().map(|directory| directory.join("meka")))
        .unwrap_or_else(std::env::temp_dir);
    sweep_stale_captures(&directory);
    if let Err(error) = std::fs::create_dir_all(&directory) {
        tracing::debug!(
            "could not create '{}' for command output capture ({}); using the temp directory",
            directory.display(),
            error
        );
        return std::env::temp_dir()
            .join(format!("meka-command-output-{}.log", uuid::Uuid::new_v4()));
    }
    directory.join(format!("command-output-{}.log", uuid::Uuid::new_v4()))
}

/// Create a capture file readable only by its owner.
///
/// A command's output is as sensitive as the command: `env`, a `curl -v` with an `Authorization`
/// header, a database dump. meka is careful to write its database at 0600 and its directories at
/// 0700; this file was created at whatever the umask allowed, in a directory shared with every
/// other user on the host on some configurations. Set before the first write, so the bytes are
/// never briefly visible at a looser mode.
async fn create_capture_file(path: &std::path::Path) -> std::io::Result<tokio::fs::File> {
    #[cfg(unix)]
    {
        tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
            .await
    }
    #[cfg(not(unix))]
    {
        // Windows ACLs inherit from the parent directory, which is the user's own cache directory.
        tokio::fs::File::create(path).await
    }
}

/// Read a child pipe to EOF, relaying each chunk as it arrives.
///
/// Reads bytes rather than `read_to_string` because a chunk boundary can fall inside a multi-byte
/// character: the trailing incomplete sequence is carried over to the next read instead of being
/// relayed as replacement characters. What is relayed still covers the whole stream, so the live
/// view is unaffected by how the reads happened to split.
///
/// The returned string is the whole stream unless it outgrew [`MAX_RESIDENT_OUTPUT_BYTES`], in
/// which case it is the two ends plus a line naming the file holding all of it.
async fn read_to_string_best_effort<R>(reader: Option<R>, relay: Option<OutputRelay>) -> String
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let Some(mut reader) = reader else {
        return String::new();
    };

    // `content` holds the whole stream until it outgrows the ceiling. After that it holds only the
    // trailing window, `head` holds the leading one, and `capture` has every byte.
    let mut content: Vec<u8> = Vec::new();
    let mut head: Vec<u8> = Vec::new();
    let mut capture = Capture::NotNeeded;
    let mut total: usize = 0;
    let mut relayed = 0usize;
    let mut buffer = [0u8; 8192];
    loop {
        match reader.read(&mut buffer).await {
            Ok(0) => break,
            Ok(read) => {
                let chunk = &buffer[..read];
                total += read;
                content.extend_from_slice(chunk);

                if let Some(relay) = &relay {
                    // Relay only the prefix that is complete UTF-8; whatever trails an incomplete
                    // sequence stays behind for the next read to finish.
                    let pending = &content[relayed..];
                    let valid = match std::str::from_utf8(pending) {
                        Ok(text) => text.len(),
                        Err(error) => error.valid_up_to(),
                    };
                    if valid > 0 {
                        // `valid` is the length of a verified-UTF-8 prefix, so the lossy decode
                        // never actually substitutes anything; it is just the panic-free spelling.
                        let text = String::from_utf8_lossy(&pending[..valid]).into_owned();
                        relayed += valid;
                        relay.send(text).await;
                    }
                }

                match &mut capture {
                    // Already capturing: the chunk goes to the file and `content` keeps only the
                    // trailing window. A write failure stops the capture rather than the command;
                    // the result then reports the truncation honestly instead of naming a file that
                    // does not hold what it claims.
                    Capture::Writing(path, file) => {
                        if let Some(error) = file.write_all(chunk).await.err() {
                            let path = path.clone();
                            tracing::warn!(
                                "failed to write command output capture '{}': {}",
                                path.display(),
                                error
                            );
                            capture = Capture::Failed;
                            // The notice below stops naming this file, so nothing would ever come
                            // back for it, and what it holds is a prefix of a stream that kept
                            // going. Leaving up to `MAX_RESIDENT_OUTPUT_BYTES` of it in the cache
                            // directory after a write failure -- most often a full disk -- is the
                            // wrong moment to be untidy.
                            if let Err(error) = tokio::fs::remove_file(&path).await {
                                tracing::debug!(
                                    "could not remove the partial capture '{}': {}",
                                    path.display(),
                                    error
                                );
                            }
                        }
                        relayed = relayed
                            .saturating_sub(trim_front_to(&mut content, OUTPUT_WINDOW_BYTES));
                    }
                    // Capture is off for the rest of the stream. Keep trimming anyway: the point of
                    // the ceiling is the residency bound, which holds whether or not the bytes are
                    // reaching a file.
                    Capture::Failed => {
                        relayed = relayed
                            .saturating_sub(trim_front_to(&mut content, OUTPUT_WINDOW_BYTES));
                    }
                    // The one crossing of the ceiling. `head` is taken here, before anything can
                    // fail, because this is the last moment `content` still starts at byte zero --
                    // and taking it here is what makes it *the* head. Re-entering this arm later
                    // (which a `None`-on-failure capture allowed, 8 MiB at a time) overwrote it
                    // with a mid-stream slice, and a capture that opened on the second attempt then
                    // held only the bytes from that point on while the notice called it complete.
                    Capture::NotNeeded if content.len() > MAX_RESIDENT_OUTPUT_BYTES => {
                        head = content[..OUTPUT_WINDOW_BYTES.min(content.len())].to_vec();
                        let path = capture_path();
                        capture = match create_capture_file(&path).await {
                            Ok(mut file) => match file.write_all(&content).await {
                                Ok(()) => Capture::Writing(path, file),
                                Err(error) => {
                                    tracing::warn!(
                                        "failed to write command output capture '{}': {}",
                                        path.display(),
                                        error
                                    );
                                    Capture::Failed
                                }
                            },
                            Err(error) => {
                                tracing::warn!(
                                    "failed to create command output capture '{}': {}",
                                    path.display(),
                                    error
                                );
                                Capture::Failed
                            }
                        };
                        relayed = relayed
                            .saturating_sub(trim_front_to(&mut content, OUTPUT_WINDOW_BYTES));
                    }
                    Capture::NotNeeded => {}
                }
            }
            Err(error) => {
                tracing::debug!("failed to read child output: {}", error);
                break;
            }
        }
    }

    // `tokio::fs::File` hands writes to the blocking pool and does not flush when it is dropped, so
    // without this the tail of the capture is lost and the file does not hold what the notice below
    // says it does.
    if let Capture::Writing(path, file) = &mut capture
        && let Err(error) = file.flush().await
    {
        let path = path.clone();
        tracing::warn!(
            "failed to flush command output capture '{}': {}",
            path.display(),
            error
        );
        capture = Capture::Failed;
        if let Err(error) = tokio::fs::remove_file(&path).await {
            tracing::debug!(
                "could not remove the unflushed capture '{}': {}",
                path.display(),
                error
            );
        }
    }

    // Lossy rather than a hard error: a command that emits a stray non-UTF-8 byte (a progress bar
    // in a foreign encoding, a binary blob on stderr) should still hand the model everything else
    // it printed.
    if head.is_empty() {
        return String::from_utf8_lossy(&content).into_owned();
    }

    let elided = total.saturating_sub(head.len() + content.len());
    let middle = match &capture {
        Capture::Writing(path, _) => format!(
            "\n\n... ({} bytes elided; the complete output is at {}) ...\n\n",
            elided,
            path.display()
        ),
        Capture::Failed | Capture::NotNeeded => format!(
            "\n\n... ({} bytes elided; capturing them to a file failed, see the log) ...\n\n",
            elided
        ),
    };
    format!(
        "{}{}{}",
        String::from_utf8_lossy(&head),
        middle,
        String::from_utf8_lossy(&content)
    )
}

/// Structured exit status for frontends that render a terminal. `ExitStatus::code()` is `None`
/// exactly when a signal ended the process, so the two are read as alternatives rather than both
/// being guessed from the same number.
fn command_exit_metadata(status: &std::process::ExitStatus) -> crate::frontend::ToolOutputMetadata {
    #[cfg(unix)]
    let signal = {
        use std::os::unix::process::ExitStatusExt;
        status.signal().map(signal_name)
    };
    #[cfg(not(unix))]
    let signal = None;
    crate::frontend::ToolOutputMetadata::CommandExit {
        exit_code: status.code(),
        signal,
    }
}

/// Symbolic name for the signals a command realistically dies from. The number alone would render
/// as `SIG9` in a client's terminal UI next to a `SIGKILL` we report elsewhere for the same kill,
/// so the two paths have to agree. Numbers outside this set are stable enough nowhere to name.
#[cfg(unix)]
fn signal_name(number: i32) -> String {
    match number {
        libc::SIGHUP => "SIGHUP".to_string(),
        libc::SIGINT => "SIGINT".to_string(),
        libc::SIGQUIT => "SIGQUIT".to_string(),
        libc::SIGABRT => "SIGABRT".to_string(),
        libc::SIGKILL => "SIGKILL".to_string(),
        libc::SIGSEGV => "SIGSEGV".to_string(),
        libc::SIGPIPE => "SIGPIPE".to_string(),
        libc::SIGTERM => "SIGTERM".to_string(),
        other => format!("SIG{}", other),
    }
}

/// Exit status for a command meka killed for exceeding its timeout. The child is torn down without
/// its status being reaped, so there is nothing to read it from. On Unix the kill really is a
/// signal ([`kill_child_tree`]); Windows has no signals, and claiming one there would put a name in
/// the client's terminal that never existed on that platform.
fn timed_out_exit_metadata() -> crate::frontend::ToolOutputMetadata {
    crate::frontend::ToolOutputMetadata::CommandExit {
        exit_code: None,
        #[cfg(unix)]
        signal: Some("SIGKILL".to_string()),
        #[cfg(not(unix))]
        signal: None,
    }
}

fn assemble_command_output(stdout: &str, stderr: &str, exit_code: i32) -> ToolOutput {
    let mut result_text = String::new();
    if !stdout.is_empty() {
        result_text.push_str(stdout);
    }
    if !stderr.is_empty() {
        if !result_text.is_empty() {
            result_text.push_str("\n--- stderr ---\n");
        }
        result_text.push_str(stderr);
    }
    if exit_code != 0 {
        result_text.push_str(&format!("\nExit code: {}", exit_code));
    }

    ToolOutput::text(
        if result_text.is_empty() {
            "(no output)".to_string()
        } else {
            result_text
        },
        exit_code != 0,
    )
}

/// Windows-only: spawn via `CreateProcessAsUserW` with a Low-integrity token, read stdout/stderr
/// from the pipe `File`s, and wait/kill through blocking tasks. Mirrors the timeout/cancellation
/// semantics of the standard path.
///
/// Stdout/stderr are drained on dedicated tasks that start *before* the child wait begins. Without
/// that, a child that writes more than the pipe buffer (1 MiB hinted; smaller if the kernel rounds
/// down) before anyone reads will block in `WriteFile`, the wait never returns, and the whole call
/// times out with truncated output. After the child exits or is killed, the pipe write ends close
/// and the drain tasks terminate at EOF.
///
/// # Drain timeouts
///
/// On Windows there is no atomic "kill process tree" primitive available in this code path (a
/// future refactor could wrap the child in a Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`).
/// Consequently, a grandchild that inherits the pipe write handles can keep the pipe alive past the
/// direct child's exit; the drain tasks would then block on `ReadFile` until the grandchild
/// finally exits. To bound the tool-call wall time we cap every drain await with [`DRAIN_TIMEOUT`];
/// on timeout the drain task is aborted, any output already read is lost, and we attach a
/// diagnostic note so the model can reason about truncation.
#[cfg(windows)]
async fn run_windows_low_integrity(
    command: &str,
    timeout_ms: u64,
    cancellation: CancellationToken,
    relay: Option<OutputRelay>,
) -> Result<ToolOutput> {
    use std::{sync::Arc, time::Duration};

    // Bound the post-kill cleanup wait so a stuck `TerminateProcess` or a drain task that somehow
    // fails to reach EOF can't hang the tool indefinitely. Two seconds is generous for kernel-side
    // teardown.
    const POST_KILL_TIMEOUT: Duration = Duration::from_secs(2);

    let mut sandboxed = crate::sandbox::spawn_low_integrity_command(command).map_err(|error| {
        MekaError::ToolExecution {
            tool_name: "execute_command".to_string(),
            message: format!("failed to spawn sandboxed command: {}", error),
        }
    })?;

    let stdout = sandboxed.take_stdout().map(tokio::fs::File::from_std);
    let stderr = sandboxed.take_stderr().map(tokio::fs::File::from_std);

    let child = Arc::new(sandboxed);
    let timeout_duration = Duration::from_millis(timeout_ms);

    let stdout_task = tokio::spawn({
        let relay = relay.clone();
        async move { read_to_string_best_effort(stdout, relay).await }
    });
    let stderr_task = tokio::spawn(async move { read_to_string_best_effort(stderr, relay).await });

    let wait_child = Arc::clone(&child);
    // `tokio::select!` requires the future passed to the happy-path branch (`join = ...`) to be
    // polled without consuming ownership of the handle, because the other two branches need to move
    // the same handle into `abort_after_timeout` if their future resolves first. Polling `&mut
    // wait_handle` satisfies `JoinHandle`'s `Future` impl (it has a `&mut self`-based `poll`)
    // without committing the move until we know which branch wins.
    let mut wait_handle = tokio::task::spawn_blocking(move || wait_child.wait_blocking());

    tokio::select! {
        _ = cancellation.cancelled() => {
            if let Err(error) = child.kill() {
                tracing::debug!("failed to kill sandboxed child: {}", error);
            }
            abort_after_timeout(wait_handle, POST_KILL_TIMEOUT).await;
            abort_after_timeout(stdout_task, POST_KILL_TIMEOUT).await;
            abort_after_timeout(stderr_task, POST_KILL_TIMEOUT).await;
            Err(MekaError::Interrupted)
        }
        _ = tokio::time::sleep(timeout_duration) => {
            if let Err(error) = child.kill() {
                tracing::debug!("failed to kill sandboxed child: {}", error);
            }
            abort_after_timeout(wait_handle, POST_KILL_TIMEOUT).await;
            abort_after_timeout(stdout_task, POST_KILL_TIMEOUT).await;
            abort_after_timeout(stderr_task, POST_KILL_TIMEOUT).await;
            Ok(ToolOutput::text(
                format!("Command timed out after {}ms", timeout_ms),
                true,
            )
            .with_metadata(timed_out_exit_metadata()))
        }
        join = &mut wait_handle => {
            let status = join
                .map_err(|error| MekaError::ToolExecution {
                    tool_name: "execute_command".to_string(),
                    message: format!("wait task panicked: {}", error),
                })?
                .map_err(|error| MekaError::ToolExecution {
                    tool_name: "execute_command".to_string(),
                    message: format!("failed to wait for command: {}", error),
                })?;

            let exit_code = status.code().unwrap_or(-1);
            let (stdout_content, stdout_timed_out) =
                join_drain_with_timeout(stdout_task, DRAIN_TIMEOUT).await;
            let (stderr_content, stderr_timed_out) =
                join_drain_with_timeout(stderr_task, DRAIN_TIMEOUT).await;
            if stdout_timed_out || stderr_timed_out {
                tracing::warn!(
                    "sandboxed command output drain timed out after {:?}; \
                     a background process may be holding the pipe open",
                    DRAIN_TIMEOUT
                );
            }
            let mut output =
                assemble_command_output(&stdout_content, &stderr_content, exit_code);
            if stdout_timed_out || stderr_timed_out {
                append_drain_truncation_note(
                    &mut output,
                    stdout_timed_out,
                    stderr_timed_out,
                );
            }
            Ok(output.with_metadata(command_exit_metadata(&status)))
        }
    }
}

/// Await a `JoinHandle<String>` up to `timeout`. If the timeout expires the task is aborted and an
/// empty string is returned alongside `timed_out=true` so the caller can surface a truncation note.
async fn join_drain_with_timeout(
    mut task: tokio::task::JoinHandle<String>,
    timeout: std::time::Duration,
) -> (String, bool) {
    tokio::select! {
        result = &mut task => match result {
            Ok(content) => (content, false),
            Err(error) => {
                tracing::debug!("drain task failed: {}", error);
                (String::new(), false)
            }
        },
        _ = tokio::time::sleep(timeout) => {
            task.abort();
            (String::new(), true)
        }
    }
}

/// Abort any pending `JoinHandle` after `timeout`. Used on cancel/timeout cleanup paths where we
/// don't need the task's output, just its termination.
#[cfg(windows)]
async fn abort_after_timeout<T: 'static>(
    mut handle: tokio::task::JoinHandle<T>,
    timeout: std::time::Duration,
) {
    tokio::select! {
        _ = &mut handle => {}
        _ = tokio::time::sleep(timeout) => {
            handle.abort();
        }
    }
}

fn append_drain_truncation_note(
    output: &mut ToolOutput,
    stdout_timed_out: bool,
    stderr_timed_out: bool,
) {
    let note = match (stdout_timed_out, stderr_timed_out) {
        (true, true) => {
            "\n(stdout/stderr drain timed out; output may be truncated: a background process likely held the pipe open past the child's exit)"
        }
        (true, false) => {
            "\n(stdout drain timed out; output may be truncated: a background process likely held the pipe open past the child's exit)"
        }
        (false, true) => {
            "\n(stderr drain timed out; output may be truncated: a background process likely held the pipe open past the child's exit)"
        }
        (false, false) => return,
    };
    if let Some(crate::provider::ToolResultContent::Text { text }) = output.content.last_mut() {
        text.push_str(note);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::tests::text_content;

    fn test_shared_permission() -> crate::permission::SharedPermission {
        crate::permission::SharedPermission::new(
            Permission::Write,
            crate::permission::EnabledPermissions::ALL,
        )
    }

    /// Construct an `ExecuteCommandTool` for tests with a backend probe matching whatever the host
    /// actually supports. Tests that need a specific probe state (e.g. exercising the "backend
    /// unavailable" hard-error path) should build `ExecuteCommandTool` directly with the desired
    /// `BackendProbe` rather than going through this helper.
    fn test_tool(
        shared_permission: crate::permission::SharedPermission,
        sandbox_enabled: bool,
    ) -> ExecuteCommandTool {
        let sandbox_capability = crate::sandbox::detect();
        let backend_probe = crate::sandbox::BackendProbe::Ok(sandbox_capability.clone());
        ExecuteCommandTool {
            sandbox_capability,
            sandbox_backend: crate::config::SandboxBackend::Landlock,
            backend_probe,
            shared_permission,
            sandbox_enabled,
            cwd: crate::workspace::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        }
    }

    /// A real signal kill and meka's own timeout kill must spell the same signal the same way; a
    /// client's terminal shows this string, and `SIG9` next to `SIGKILL` for the same event reads
    /// as two different failures.
    #[cfg(unix)]
    #[test]
    fn test_signal_naming_is_consistent_between_a_real_kill_and_a_timeout() {
        assert_eq!(signal_name(libc::SIGKILL), "SIGKILL");
        assert_eq!(signal_name(libc::SIGTERM), "SIGTERM");
        assert_eq!(
            signal_name(4242),
            "SIG4242",
            "unknown signals stay unambiguous"
        );

        let crate::frontend::ToolOutputMetadata::CommandExit { exit_code, signal } =
            timed_out_exit_metadata()
        else {
            panic!("timeout must report a command exit");
        };
        assert_eq!(
            exit_code, None,
            "a killed command never produced an exit code"
        );
        assert_eq!(
            signal.as_deref(),
            Some(signal_name(libc::SIGKILL).as_str()),
            "the timeout path must name the signal the same way the reaped-status path does",
        );
    }

    /// `execute_command` reports its exit status structurally, not only inside the prose the model
    /// reads, so a frontend rendering a terminal can show the real code instead of guessing from
    /// the error flag.
    #[tokio::test]
    async fn test_execute_command_reports_its_exit_code_as_metadata() {
        let tool = test_tool(test_shared_permission(), false);
        let result = tool
            .execute(
                serde_json::json!({"command": "exit 42"}),
                CancellationToken::new(),
            )
            .await
            .expect("tool runs");
        assert!(result.is_error, "a non-zero exit is a failed call");
        let Some(crate::frontend::ToolOutputMetadata::CommandExit { exit_code, signal }) =
            result.frontend_metadata
        else {
            panic!(
                "expected CommandExit metadata; got {:?}",
                result.frontend_metadata
            );
        };
        assert_eq!(exit_code, Some(42));
        assert_eq!(signal, None, "a clean exit carries no signal");
    }

    /// A read can end partway through a multi-byte character. Relaying the raw bytes would show
    /// the client replacement characters that were never in the output, so the incomplete tail has
    /// to wait for the read that completes it. Feeds the reader one byte at a time to force the
    /// split on every character.
    #[tokio::test]
    async fn test_reader_relays_chunks_without_splitting_characters() {
        #[derive(Default)]
        struct ChunkRecorder {
            chunks: std::sync::Mutex<Vec<String>>,
        }

        #[async_trait]
        impl crate::frontend::Frontend for ChunkRecorder {
            async fn emit(&self, event: crate::frontend::FrontendEvent) {
                if let crate::frontend::FrontendEvent::ToolCallOutputDelta { chunk, .. } = event
                    && let Ok(mut chunks) = self.chunks.lock()
                {
                    chunks.push(chunk);
                }
            }

            async fn request_permission(
                &self,
                _request: crate::frontend::PermissionRequest,
            ) -> crate::frontend::PermissionOutcome {
                crate::frontend::PermissionOutcome::Deny
            }
        }

        let recorder = Arc::new(ChunkRecorder::default());
        let frontend: Arc<dyn crate::frontend::Frontend> = recorder.clone();
        let relay = Some(OutputRelay {
            frontend,
            tool_call_id: "call_1".to_string(),
        });

        let source = "ünïcödé ✓ done\n";
        // `tokio::io::AsyncRead` over a byte slice yields whatever the caller's buffer allows, so
        // cap reads at one byte to guarantee every multi-byte character straddles a read.
        let reader = tokio::io::AsyncReadExt::take(source.as_bytes(), u64::MAX);
        let collected = read_to_string_best_effort(Some(OneByteAtATime(reader)), relay).await;

        assert_eq!(collected, source, "the full stream must survive intact");
        let chunks = recorder.chunks.lock().expect("lock").clone();
        assert_eq!(
            chunks.concat(),
            source,
            "the relayed chunks must reassemble to the same bytes",
        );
        assert!(
            !chunks.iter().any(|chunk| chunk.contains('\u{fffd}')),
            "no chunk may contain a replacement character; got {chunks:?}",
        );
    }

    /// The relay must survive the stream outgrowing the ceiling.
    ///
    /// `relayed` is an index into the same buffer the capture trims, and the trim used to clamp it
    /// to the new length rather than shift it by what was dropped. That marked the carried
    /// incomplete-UTF-8 bytes as sent, so the next read began mid-character, `valid_up_to()`
    /// returned 0, and its whole 8 KB vanished from the live stream while still reaching the
    /// capture. Only multi-byte output shows it, and only once past the ceiling: the two conditions
    /// this test puts together, which is why neither existing test caught it.
    #[tokio::test]
    async fn the_relay_loses_nothing_when_the_stream_outgrows_the_ceiling() {
        #[derive(Default)]
        struct ChunkRecorder {
            chunks: std::sync::Mutex<Vec<String>>,
        }

        #[async_trait]
        impl crate::frontend::Frontend for ChunkRecorder {
            async fn emit(&self, event: crate::frontend::FrontendEvent) {
                if let crate::frontend::FrontendEvent::ToolCallOutputDelta { chunk, .. } = event
                    && let Ok(mut chunks) = self.chunks.lock()
                {
                    chunks.push(chunk);
                }
            }

            async fn request_permission(
                &self,
                _request: crate::frontend::PermissionRequest,
            ) -> crate::frontend::PermissionOutcome {
                crate::frontend::PermissionOutcome::Deny
            }
        }

        // A 3-byte character repeated, so no power-of-two read size can land on a boundary and
        // every read carries a partial character into the next.
        let unit = "日";
        let source = unit.repeat(MAX_RESIDENT_OUTPUT_BYTES / unit.len() + 4096);
        assert!(source.len() > MAX_RESIDENT_OUTPUT_BYTES, "must overflow");

        let recorder = Arc::new(ChunkRecorder::default());
        let frontend: Arc<dyn crate::frontend::Frontend> = recorder.clone();
        let relay = Some(OutputRelay {
            frontend,
            tool_call_id: "call_1".to_string(),
        });

        let collected = read_to_string_best_effort(
            Some(std::io::Cursor::new(source.clone().into_bytes())),
            relay,
        )
        .await;

        let chunks = recorder.chunks.lock().expect("lock").clone();
        assert_eq!(
            chunks.concat(),
            source,
            "every byte printed must reach the client, capture or no capture",
        );
        assert!(
            !chunks.iter().any(|chunk| chunk.contains('\u{fffd}')),
            "and no chunk may be cut mid-character",
        );

        // Clean up the capture the overflow created.
        if let Some(start) = collected.find("the complete output is at ") {
            let start = start + "the complete output is at ".len();
            if let Some(end) = collected[start..].find(')') {
                let _ = std::fs::remove_file(std::path::Path::new(&collected[start..start + end]));
            }
        }
    }

    /// A command that outruns the turn must not take the process with it, and must not lose what
    /// it printed either. Past the ceiling the stream goes to a file and the result carries both
    /// ends plus the path, so the model sees the shape and can read the rest.
    #[tokio::test]
    async fn an_oversized_stream_is_captured_to_a_file_rather_than_held_in_memory() {
        // One byte more than the ceiling, with distinct ends so both are identifiable.
        let mut source = Vec::with_capacity(MAX_RESIDENT_OUTPUT_BYTES + 64);
        source.extend_from_slice(b"FIRST-LINE\n");
        source.resize(MAX_RESIDENT_OUTPUT_BYTES + 32, b'x');
        source.extend_from_slice(b"\nLAST-LINE\n");
        let total = source.len();

        let collected =
            read_to_string_best_effort(Some(std::io::Cursor::new(source.clone())), None).await;

        assert!(
            collected.len() < total,
            "the result must not carry the whole stream inline"
        );
        assert!(
            collected.starts_with("FIRST-LINE\n"),
            "the head must survive"
        );
        assert!(collected.ends_with("LAST-LINE\n"), "the tail must survive");
        assert!(
            collected.contains("bytes elided"),
            "the cut must be disclosed: {}",
            &collected[..collected.len().min(200)]
        );

        // Nothing is lost: the notice names a file holding every byte.
        let marker = "the complete output is at ";
        let start = collected.find(marker).expect("capture path") + marker.len();
        let end = collected[start..].find(')').expect("capture path end") + start;
        let path = std::path::PathBuf::from(&collected[start..end]);
        let captured = std::fs::read(&path).expect("read capture");
        std::fs::remove_file(&path).expect("clean up capture");
        assert_eq!(
            captured.len(),
            source.len(),
            "the capture must hold the whole stream"
        );
        assert_eq!(captured, source, "the capture must hold the whole stream");
    }

    /// Captures are 8 MiB or more each, and nothing else ever removes them: a machine that runs
    /// long builds accumulates one per overflow until the disk notices. The sweep runs on the
    /// overflow path, so it costs a directory read only when something is about to be written
    /// anyway, and it touches only meka's own names.
    #[test]
    fn stale_captures_are_swept_and_other_files_are_left_alone() {
        let directory = tempfile::tempdir().expect("tempdir");
        let old =
            std::time::SystemTime::now() - (CAPTURE_RETENTION + std::time::Duration::from_secs(60));

        let stale = directory.path().join("command-output-abc.log");
        let fresh = directory.path().join("command-output-def.log");
        let theirs = directory.path().join("someone-elses.log");
        for path in [&stale, &fresh, &theirs] {
            std::fs::write(path, b"x").expect("seed");
        }
        for path in [&stale, &theirs] {
            filetime::set_file_mtime(path, filetime::FileTime::from_system_time(old))
                .expect("age the file");
        }

        sweep_stale_captures(directory.path());

        assert!(!stale.exists(), "an old capture must go");
        assert!(
            fresh.exists(),
            "a recent one is still referenced by a live conversation"
        );
        assert!(
            theirs.exists(),
            "a file meka did not write is not meka's to delete, however old",
        );
    }

    /// A capture file holds a command's whole output, which is as sensitive as the command: `env`,
    /// a `curl -v` carrying an `Authorization` header, a database dump. It was created at whatever
    /// the umask allowed, in a cache directory that on some setups is world-traversable, while meka
    /// takes care to write its database at 0600 and its directories at 0700.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_capture_file_is_readable_only_by_its_owner() {
        use std::os::unix::fs::PermissionsExt;

        let mut source = Vec::with_capacity(MAX_RESIDENT_OUTPUT_BYTES + 64);
        source.extend_from_slice(b"FIRST-LINE\n");
        source.resize(MAX_RESIDENT_OUTPUT_BYTES + 32, b'x');
        source.extend_from_slice(b"\nLAST-LINE\n");

        let collected = read_to_string_best_effort(Some(std::io::Cursor::new(source)), None).await;

        let marker = "the complete output is at ";
        let start = collected.find(marker).expect("capture path") + marker.len();
        let end = collected[start..].find(')').expect("capture path end") + start;
        let path = std::path::PathBuf::from(&collected[start..end]);

        let mode = std::fs::metadata(&path)
            .expect("stat capture")
            .permissions()
            .mode()
            & 0o777;
        std::fs::remove_file(&path).expect("clean up capture");
        assert_eq!(mode, 0o600, "got {mode:o}");
    }

    /// When the capture cannot be opened, the head still has to be the head.
    ///
    /// A failed capture used to leave the state as "not capturing yet", so crossing the ceiling a
    /// second time 8 MiB later ran the whole opening sequence again and overwrote `head` with a
    /// slice from the middle of the stream -- which the result then printed as the beginning, under
    /// a notice that named the elision but not the lie. The same re-entry could also open a capture
    /// on the second attempt, holding only the bytes from that point on while the notice called it
    /// the complete output.
    #[tokio::test]
    async fn a_failed_capture_keeps_the_real_head_and_does_not_retry() {
        FORCE_CAPTURE_FAILURE.with(|forced| forced.set(true));

        // Past the ceiling twice, so the arm that opens the capture is reached more than once.
        let mut source = Vec::with_capacity(MAX_RESIDENT_OUTPUT_BYTES * 2 + 64);
        source.extend_from_slice(b"FIRST-LINE\n");
        source.resize(MAX_RESIDENT_OUTPUT_BYTES * 2 + 32, b'x');
        source.extend_from_slice(b"\nLAST-LINE\n");

        let collected = read_to_string_best_effort(Some(std::io::Cursor::new(source)), None).await;
        FORCE_CAPTURE_FAILURE.with(|forced| forced.set(false));

        assert!(
            collected.starts_with("FIRST-LINE\n"),
            "the head must still be the start of the stream, got: {}",
            &collected[..collected.len().min(80)],
        );
        assert!(collected.ends_with("LAST-LINE\n"), "the tail must survive");
        assert!(
            collected.contains("capturing them to a file failed"),
            "and the notice must say the bytes are gone rather than name a file: {}",
            &collected[..collected.len().min(200)],
        );
    }

    /// The common case must be untouched: no file, no notice, byte-for-byte what was printed.
    #[tokio::test]
    async fn an_ordinary_stream_is_returned_whole_with_no_capture() {
        let source = b"just a normal amount of output\n".to_vec();
        let collected =
            read_to_string_best_effort(Some(std::io::Cursor::new(source.clone())), None).await;
        assert_eq!(collected.as_bytes(), source.as_slice());
    }

    /// Reader that hands out at most one byte per `poll_read`, so every multi-byte character in
    /// the source is guaranteed to span a read boundary.
    struct OneByteAtATime<R>(R);

    impl<R: tokio::io::AsyncRead + Unpin> tokio::io::AsyncRead for OneByteAtATime<R> {
        fn poll_read(
            mut self: std::pin::Pin<&mut Self>,
            context: &mut std::task::Context<'_>,
            buffer: &mut tokio::io::ReadBuf<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            let mut single = [0u8; 1];
            let mut limited = tokio::io::ReadBuf::new(&mut single);
            match std::pin::Pin::new(&mut self.0).poll_read(context, &mut limited) {
                std::task::Poll::Ready(Ok(())) => {
                    let filled = limited.filled().to_vec();
                    buffer.put_slice(&filled);
                    std::task::Poll::Ready(Ok(()))
                }
                other => other,
            }
        }
    }

    #[tokio::test]
    async fn test_execute_command() {
        let tool = test_tool(test_shared_permission(), true);
        let result = tool
            .execute(
                serde_json::json!({"command": "echo hello"}),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert_eq!(text_content(&result).trim(), "hello");
    }

    /// Regression test for the orphaned-grandchild bug: a command that backgrounds a long-running
    /// helper (`(sleep 30 &)`) must have that helper killed when the tool times out, not outlive
    /// the agent. The child is placed in its own process group via `setsid` so the tool can signal
    /// the whole tree via `kill(-pgid, …)`.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_execute_command_timeout_kills_grandchild() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let marker = temp_dir.path().join("marker");
        let marker_str = marker.to_str().expect("utf-8 path").to_string();

        let tool = test_tool(test_shared_permission(), false);

        // The grandchild sleeps 3s then touches `marker`. If it survived the timeout, the marker
        // file will appear. The timeout is 300ms and we wait 5s below for a definitive "did it
        // survive?" answer.
        let script = format!(
            "( sleep 3 && : > '{}' ) & echo backgrounded; sleep 30",
            marker_str
        );
        let result = tool
            .execute(
                serde_json::json!({ "command": script, "timeout_ms": 300u64 }),
                CancellationToken::new(),
            )
            .await
            .expect("execute should not error");

        // Tool reports timeout.
        assert!(result.is_error);
        let text = text_content(&result);
        assert!(text.contains("timed out"), "got: {:?}", text);

        // Wait well past the grandchild's sleep-3s. If the marker materializes, the grandchild
        // wasn't killed; the bug is back.
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        assert!(
            !marker.exists(),
            "grandchild survived timeout and created marker at {:?}",
            marker
        );
    }

    #[tokio::test]
    async fn test_execute_command_failure() {
        let tool = test_tool(test_shared_permission(), true);
        let result = tool
            .execute(
                serde_json::json!({"command": "false"}),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(result.is_error);
    }

    /// When the configured sandbox backend isn't usable, read-mode `execute_command` must return
    /// `Err(MekaError::ToolExecution)`, *not* `Ok(ToolOutput { is_error: true })`. The hard error
    /// path is how the model is forced to surface the failure to the user rather than just retrying
    /// or describing it as a tool result.
    #[tokio::test]
    async fn test_execute_command_hard_errors_when_backend_unavailable() {
        let read_only_perm = crate::permission::SharedPermission::new(
            Permission::Read,
            crate::permission::EnabledPermissions::ALL,
        );
        let tool = ExecuteCommandTool {
            sandbox_capability: crate::sandbox::SandboxCapability::Unavailable,
            sandbox_backend: crate::config::SandboxBackend::Bubblewrap,
            backend_probe: crate::sandbox::BackendProbe::Missing {
                reason: "bwrap not found on PATH".to_string(),
            },
            shared_permission: read_only_perm,
            sandbox_enabled: true,
            cwd: crate::workspace::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({"command": "echo nope"}),
                CancellationToken::new(),
            )
            .await;
        match result {
            Err(MekaError::ToolExecution { tool_name, message }) => {
                assert_eq!(tool_name, "execute_command");
                // The Linux error path splices in the configured backend's display name
                // (`Bubblewrap`); the non-Linux variant drops the Linux-specific config reference
                // and reads "sandbox is unavailable: ...". Both must include the probe reason
                // verbatim.
                #[cfg(target_os = "linux")]
                assert!(
                    message.contains("Bubblewrap"),
                    "expected backend display name in error: {}",
                    message
                );
                assert!(
                    message.contains("bwrap not found on PATH"),
                    "expected probe reason in error: {}",
                    message
                );
            }
            Err(other) => panic!("expected ToolExecution, got {:?}", other),
            Ok(output) => panic!("expected hard error, got Ok({:?})", text_content(&output)),
        }
    }

    /// When the tool is invoked at Write permission, an unavailable sandbox backend must NOT
    /// short-circuit the spawn; the user has explicitly opted out of sandboxing for this command.
    #[tokio::test]
    async fn test_execute_command_runs_without_sandbox_when_write_mode() {
        let write_perm = crate::permission::SharedPermission::new(
            Permission::Write,
            crate::permission::EnabledPermissions::ALL,
        );
        let tool = ExecuteCommandTool {
            sandbox_capability: crate::sandbox::SandboxCapability::Unavailable,
            sandbox_backend: crate::config::SandboxBackend::Bubblewrap,
            backend_probe: crate::sandbox::BackendProbe::Missing {
                reason: "bwrap not found on PATH".to_string(),
            },
            shared_permission: write_perm,
            sandbox_enabled: true,
            cwd: crate::workspace::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({"command": "echo hello"}),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed in write mode");
        assert!(!result.is_error);
        assert_eq!(text_content(&result).trim(), "hello");
    }

    #[tokio::test]
    async fn test_execute_command_large_output_not_truncated() {
        // Output well over the old 30 KB cap: the tool must return it in full. The agent layer
        // handles oversize downstream.
        let tool = test_tool(test_shared_permission(), true);
        let result = tool
            .execute(
                // 50 000 "x" characters. POSIX-portable: uses `head` and `tr` instead of bash
                // brace expansion so it works under `dash` (Debian/Ubuntu's default `/bin/sh`) as
                // well as `bash`.
                serde_json::json!({
                    "command": "head -c 50000 /dev/zero | tr '\\0' x"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        let text = text_content(&result);
        assert!(
            !text.contains("(output truncated"),
            "no truncation marker expected, got: {:.200}...",
            text
        );
        assert!(
            text.trim().len() >= 50_000,
            "expected >= 50 000 chars, got {}",
            text.trim().len()
        );
    }

    /// Regression test for the stdout/stderr pipe deadlock on Unix: a command writing far more than
    /// the OS pipe buffer (~64 KiB on Linux) must complete without blocking. Before draining
    /// stdout/stderr on dedicated tasks that start *before* `child.wait()`, the child blocked in
    /// `write()`, `wait()` never returned, and the call hit a spurious timeout with truncated
    /// output.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_execute_command_large_output_no_deadlock() {
        let tool = test_tool(test_shared_permission(), true);
        // 5 MiB of 'x', two orders of magnitude past any pipe buffer.
        let result = tool
            .execute(
                serde_json::json!({
                    "command": "head -c 5242880 /dev/zero | tr '\\0' x"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error, "large output spuriously flagged as error");
        let text = text_content(&result);
        assert!(
            !text.contains("drain timed out"),
            "unexpected drain-timeout note: {:.200}",
            text
        );
        assert!(
            text.trim().len() >= 5_242_880,
            "expected >= 5 MiB of output, got {}",
            text.trim().len()
        );
    }

    #[cfg(windows)]
    mod windows_sandbox {
        use super::*;

        fn read_permission() -> crate::permission::SharedPermission {
            crate::permission::SharedPermission::new(
                Permission::Read,
                crate::permission::EnabledPermissions::ALL,
            )
        }

        /// Build an `ExecuteCommandTool` for the Low-integrity Windows path. Mirrors
        /// `super::test_tool` (which always calls `sandbox::detect()` and would resolve to
        /// `LowIntegrity` on Windows anyway) but constructs the fields explicitly so the tests
        /// document the intended state.
        fn windows_test_tool(
            shared_permission: crate::permission::SharedPermission,
        ) -> ExecuteCommandTool {
            let sandbox_capability = crate::sandbox::SandboxCapability::LowIntegrity;
            let backend_probe = crate::sandbox::BackendProbe::Ok(sandbox_capability.clone());
            ExecuteCommandTool {
                sandbox_capability,
                // `sandbox_backend` is Linux-only metadata; on Windows the value is never read but
                // the field must still be populated. `Landlock` is the conventional placeholder.
                sandbox_backend: crate::config::SandboxBackend::Landlock,
                backend_probe,
                shared_permission,
                sandbox_enabled: true,
                cwd: crate::workspace::test_cwd(),
                frontend: Arc::new(crate::frontend::SilentFrontend),
            }
        }

        /// Under Low integrity, writing to the user's profile directory must be denied by the OS.
        /// The test probes a path under `%USERPROFILE%` and asserts the file is never created.
        #[tokio::test]
        async fn test_windows_sandbox_blocks_write_to_userprofile() {
            let probe_path = format!(
                "{}\\meka-sandbox-probe.txt",
                std::env::var("USERPROFILE").expect("USERPROFILE must be set on Windows")
            );
            // Clean any stray file from an earlier failed run before starting.
            let _ = std::fs::remove_file(&probe_path);

            let tool = windows_test_tool(read_permission());
            let _ = tool
                .execute(
                    serde_json::json!({
                        "command": format!("echo hello > \"{}\"", probe_path),
                    }),
                    CancellationToken::new(),
                )
                .await
                .expect("execute should not error");

            let existed = std::path::Path::new(&probe_path).exists();
            // Defensive cleanup even if the assertion below fails.
            let _ = std::fs::remove_file(&probe_path);
            assert!(
                !existed,
                "Low-integrity sandbox should have blocked write to {}",
                probe_path
            );
        }

        /// A command that produces well over the default Windows pipe buffer (~4 KB) of output must
        /// complete without deadlocking and without truncation. Before the concurrent-drain fix,
        /// the child would block in `WriteFile` past the buffer, the wait would never return, and
        /// the tool would report a spurious timeout.
        #[tokio::test]
        async fn test_windows_sandbox_large_output_under_sandbox() {
            let tool = windows_test_tool(read_permission());
            // PowerShell builds a 262144-char string in memory then emits it as one line. Total
            // output is ~256 KB, well past any plausible pipe buffer.
            let result = tool
                .execute(
                    serde_json::json!({
                        "command": "'x' * 262144",
                        "timeout_ms": 60000u64,
                    }),
                    CancellationToken::new(),
                )
                .await
                .expect("execute should not error");

            assert!(
                !result.is_error,
                "large-output command should not be flagged as an error"
            );
            let text = text_content(&result);
            let x_count = text.matches('x').count();
            assert!(
                x_count >= 262144,
                "expected >= 262144 'x' characters in output, got {}",
                x_count
            );
        }

        /// The child's stdin must be connected to `NUL`, not inherited from the agent's TTY and not
        /// left as an invalid handle. `$input` enumerates pipeline input; piped from NUL it yields
        /// zero objects. The command must complete promptly rather than hanging on a dangling
        /// stdin.
        #[tokio::test]
        async fn test_windows_sandbox_stdin_is_null() {
            let tool = windows_test_tool(read_permission());
            let result = tokio::time::timeout(
                std::time::Duration::from_secs(10),
                tool.execute(
                    serde_json::json!({
                        "command": "($input | Measure-Object).Count",
                        "timeout_ms": 5000u64,
                    }),
                    CancellationToken::new(),
                ),
            )
            .await
            .expect("command must not hang waiting for stdin")
            .expect("execute should not error");

            assert!(!result.is_error);
            let text = text_content(&result);
            assert!(
                text.trim().starts_with('0'),
                "expected stdin-object count of 0, got {:?}",
                text
            );
        }

        /// Round-trip a grab-bag of tricky marker strings through PowerShell to confirm
        /// `quote_command_arg` + PowerShell's argv parser are inverses. Uses PowerShell
        /// single-quote literals internally so the test exercises our command-line encoding, not PS
        /// string rules.
        #[tokio::test]
        async fn test_windows_sandbox_quoting_roundtrip() {
            let tool = windows_test_tool(read_permission());

            let cases: &[&str] = &[
                "plain",
                "with spaces",
                r#"quotes "inside""#,
                r"back\slashes",
                "meta & chars | pipe > redir",
                "日本語 unicode",
            ];
            for marker in cases {
                // Escape ' as '' inside the PS single-quote literal.
                let script = format!("Write-Output '{}'", marker.replace('\'', "''"));
                let result = tool
                    .execute(
                        serde_json::json!({ "command": script, "timeout_ms": 10000u64 }),
                        CancellationToken::new(),
                    )
                    .await
                    .expect("execute should not error");
                assert!(!result.is_error, "command for marker {:?} errored", marker);
                let text = text_content(&result);
                assert!(
                    text.contains(marker),
                    "marker {:?} missing from output {:?}",
                    marker,
                    text
                );
            }
        }

        /// Regression test for the parent-env-inheritance leak: secrets set in the parent (API
        /// keys, OAuth tokens) must not appear in the sandboxed child's environment, because a
        /// Low-integrity child can still open outbound sockets and exfiltrate them.
        #[tokio::test]
        async fn test_windows_sandbox_scrubs_provider_api_keys() {
            // SAFETY: tests run under `cargo test`, which is single-threaded per target by default
            // for integration tests, and this env var is scoped to the test's probe command.
            // Acceptable for a test.
            unsafe {
                std::env::set_var("ANTHROPIC_API_KEY", "probe-12345-leaked");
            }

            let tool = windows_test_tool(read_permission());
            let result = tool
                .execute(
                    serde_json::json!({
                        "command": "$env:ANTHROPIC_API_KEY",
                        "timeout_ms": 10000u64,
                    }),
                    CancellationToken::new(),
                )
                .await
                .expect("execute should not error");

            unsafe {
                std::env::remove_var("ANTHROPIC_API_KEY");
            }

            let text = text_content(&result);
            assert!(
                !text.contains("probe-12345-leaked"),
                "parent API key leaked into sandboxed child env: {:?}",
                text
            );
        }

        /// Reads must still succeed under Low integrity. The hosts file is readable by Everyone on
        /// stock Windows, so it's a good probe.
        #[tokio::test]
        async fn test_windows_sandbox_allows_read() {
            let tool = windows_test_tool(read_permission());
            let result = tool
                .execute(
                    serde_json::json!({
                        "command": "type C:\\Windows\\System32\\drivers\\etc\\hosts",
                    }),
                    CancellationToken::new(),
                )
                .await
                .expect("execute should not error");

            assert!(
                !result.is_error,
                "reading %WINDIR%\\System32\\drivers\\etc\\hosts should succeed under Low integrity"
            );
        }
    }
}
