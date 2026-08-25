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

/// Every bwrap argument up to the `--` separator, for `writable` roots.
///
/// Factored out of the spawn so the ordering rule below is testable without a live child. Order is
/// the whole correctness argument here and it is invisible in the resulting mount namespace: bwrap
/// applies operations onto the new root in sequence and the last one to touch a path wins, so a
/// bind placed before the tmpfs masks is silently undone by them. A workspace under `/tmp` -- where
/// every test fixture and a fair number of real scratch directories live -- would come out
/// read-only with no error from bwrap, no error from meka, and a mode that quietly confines the
/// shell to nothing.
#[cfg(target_os = "linux")]
fn bwrap_args(writable: &[std::path::PathBuf], cwd: &std::path::Path) -> Vec<std::ffi::OsString> {
    // `--ro-bind /` enforces "no writes", `--unshare-*` cuts off PID / user / UTS / IPC views, and
    // the tmpfs masks over `/run`, `/tmp`, `/var/tmp` and `$XDG_RUNTIME_DIR` make the dbus and
    // systemd-user sockets unreachable so the agent cannot `dbus-send` state-changing methods.
    // `--unshare-net` is intentionally absent; network must stay open for `curl | pdftotext` and
    // similar pipelines.
    let mut args: Vec<std::ffi::OsString> = [
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
    ]
    .iter()
    .map(std::ffi::OsString::from)
    .collect();

    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR")
        && std::path::Path::new(&xdg).is_absolute()
    {
        args.push("--tmpfs".into());
        args.push(xdg.into());
    }

    // The working directory, read-only, after the masks and before the writable binds.
    //
    // Without it a session whose cwd is under a masked directory lost the directory entirely, and
    // bwrap's fallback is silent: `Command::current_dir` chdirs before `execve`, bwrap cannot
    // re-enter that path inside the new root, and it lands the child in `$HOME` instead. Measured
    // at `read` with a cwd under `/tmp`: `pwd` reported the user's home directory, `ls .` listed
    // `.ssh` and `.config`, the workspace was unreachable even by absolute path, and the command
    // exited 0 with empty stderr. Meanwhile `read_file` and `search_contents` run in-process and
    // saw the real files, so the model was handed two contradictory views of one session.
    //
    // Read-only because this is the `read`-mode fix: a writable root that happens to be the cwd is
    // bound read-write by the loop below, and a later mount wins.
    //
    // Skipped when the cwd *is* a masked directory, or an ancestor of one, because the same
    // last-mount-wins rule that makes this fix work would otherwise undo every mask above it.
    // Measured against real bwrap: with the bind unconditional, a session at `/tmp` saw 1330 host
    // entries instead of 0 and could reach the tmux socket; at `$XDG_RUNTIME_DIR` it reached the
    // session bus; at `/` it saw 455 host PIDs instead of 4, defeating `--unshare-pid` as well.
    // That is the escape `is_system_root` exists to prevent, arriving through the one door it does
    // not guard: it filters the *writable roots*, and the cwd is bound whether or not it is one.
    //
    // Nothing is lost by skipping it. `--chdir` below is unconditional, and a masked directory
    // still exists inside the sandbox as the empty tmpfs, so the child lands there and sees what
    // the mask intends rather than being relocated to `$HOME`. A path merely *under* a mask
    // (`/tmp/work`) is not a masked root, so it still gets its bind and still works.
    if !crate::workspace::is_system_root(cwd) {
        args.push("--ro-bind-try".into());
        args.push(cwd.into());
        args.push(cwd.into());
    }

    for root in writable {
        // `--bind-try`, not `--bind`. A root is canonicalised when the confinement is resolved and
        // mounted a moment later; a concurrent `execute_command` running `rm -rf` on it in between
        // makes plain `--bind` abort the *whole* spawn with a bwrap error the model cannot act on.
        // Landlock already degrades correctly here -- it skips a root it cannot open rather than
        // failing the command -- and this is the same rule spelled in bwrap's own vocabulary.
        args.push("--bind-try".into());
        args.push(root.into());
        args.push(root.into());
    }

    // Asked for explicitly rather than inherited through the pre-`execve` chdir, so that a cwd
    // bwrap cannot enter is a loud failure the model can read instead of a silent relocation to
    // `$HOME`. Last, so it applies to the mounts above it.
    args.push("--chdir".into());
    args.push(cwd.into());
    args
}

pub(super) struct ExecuteCommandTool {
    /// The process's workspace-ACE ledger on Windows.
    ///
    /// A handle on the one `process_grants()` singleton, not a per-tool ledger. It reads as a
    /// field because that is how the tool reaches it, and the distinction matters: an ACE is
    /// machine state, so a per-registry ledger had a sub-agent's teardown revoke the ACEs its
    /// parent was still writing through. Released by `release_process_grants` at process exit
    /// rather than by `Drop`; see [`crate::sandbox::windows_impl::WindowsGrants`] for what
    /// that does and does not cover.
    #[cfg(windows)]
    pub windows_grants: std::sync::Arc<crate::sandbox::windows_impl::WindowsGrants>,
    /// The write boundary, shared with `write_file`. The shell derives its sandbox allow-list from
    /// the same [`crate::workspace::WriteScope`] the file tools fence against, so the two cannot
    /// disagree about where a write may land.
    pub scope: crate::workspace::WriteScope,
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
            Permission::Unrestricted
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
        // Three states, resolved once: the rest of this function asks the `Confinement` rather than
        // re-deriving "is it sandboxed" from the level, which is how the read-only and
        // workspace-writable cases would drift apart.
        let confinement = crate::sandbox::Confinement::resolve(
            self.sandbox_enabled,
            permission,
            &self.scope,
            &self.cwd,
        );
        let sandboxed = confinement.is_sandboxed();

        // `[shell].sandbox = false` unconfines every level. That is right for the levels that never
        // promised a boundary and wrong for `workspace`, whose entire meaning is one. Left alone it
        // ran the shell with no confinement while the file tools stayed fenced, so a single config
        // key made the level mean two different things and the weaker meaning was the silent one.
        //
        // Refused rather than hidden. `required_permission` cannot hide it: `Workspace.allows` is
        // true for everything by design, because scope is meant to be enforced at the door rather
        // than by withholding tools. Refusing at that door is the same shape as the write fence,
        // and it can say what to do about it where a missing tool could not.
        // Any level that promises confinement, not just `workspace`.
        //
        // `ask` and `unrestricted` are the only two whose *intent* is `Unconfined`; every other
        // level reaching an unconfined spawn is a configuration that cannot deliver what the level
        // says. Keyed on `workspace` alone, the sibling case stayed open:
        // `[tools.tool_permissions]` overrides a tool's required level with no floor, so
        // `execute_command = "read"` plus `[shell].sandbox = false` ran a plain `sh -c` at
        // `read` -- with the full parent environment, since the scrub is gated on
        // `sandboxed` too.
        if !matches!(permission, Permission::Ask | Permission::Unrestricted) && !sandboxed {
            return Err(MekaError::ToolExecution {
                tool_name: "execute_command".to_string(),
                message: "[shell].sandbox = false leaves nothing to confine this command, \
                          which `workspace` requires. `unrestricted` runs it without a boundary; \
                          otherwise re-enable [shell].sandbox in the config."
                    .to_string(),
            });
        }

        if sandboxed {
            // Configured backend isn't usable on this host. Hard-error with the specific reason so
            // the model can surface it via `render::render_error` rather than treat the failure as
            // a tool result it could try to recover from.
            if let Some(reason) = crate::sandbox::backend_unavailable_reason(&self.backend_probe) {
                // `sandbox_backend` is Linux-only; on other platforms there's nothing to
                // reconfigure. The only escape hatch is `unrestricted`, which is also the only
                // level whose confinement is `Unconfined` and so never reaches this branch.
                #[cfg(target_os = "linux")]
                let message = format!(
                    "configured sandbox backend ({}) is unavailable: {}. \
                     `unrestricted` runs shell commands without a sandbox; otherwise \
                     update [shell].sandbox_backend in the config.",
                    self.sandbox_backend, reason
                );
                #[cfg(not(target_os = "linux"))]
                let message = format!(
                    "sandbox is unavailable: {}. `unrestricted` runs shell commands \
                     without a sandbox.",
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
            // Two different mechanisms, picked by what the level promises. A workspace confinement
            // needs the ACEs in place *before* the token names them, so the grant happens here
            // rather than inside the spawn.
            let windows_confinement = match confinement.writable() {
                [] => crate::sandbox::windows_impl::WindowsConfinement::LowIntegrity,
                roots => {
                    // Off the async executor. `ensure` calls `SetNamedSecurityInfoW`, which
                    // propagates the inheritable ACE over the *entire* existing tree -- seconds to
                    // minutes for a large workspace -- synchronously, while holding the ledger
                    // mutex. On a tokio worker that stalls streaming, every other `meka serve`
                    // session, and cancellation. `clippy::await_holding_lock` does not catch this
                    // shape: there is no `.await` inside the lock, just a long blocking syscall.
                    let grants = Arc::clone(&self.windows_grants);
                    let owned: Vec<std::path::PathBuf> = roots.to_vec();
                    let granted = tokio::task::spawn_blocking(move || {
                        for root in &owned {
                            if let Err(error) = grants.ensure(root) {
                                return Err((root.clone(), error));
                            }
                        }
                        Ok(())
                    })
                    .await
                    .map_err(|error| MekaError::ToolExecution {
                        tool_name: "execute_command".to_string(),
                        message: format!("granting workspace write access panicked: {error}"),
                    })?;

                    if let Err((root, error)) = &granted {
                        return Err(MekaError::ToolExecution {
                            tool_name: "execute_command".to_string(),
                            message: format!(
                                "could not make '{}' writable for the sandboxed shell: {}. \
                                 meka needs to own the directory to grant itself write access \
                                 there; a network share or another user's folder cannot be a \
                                 workspace root on Windows.",
                                root.display(),
                                error
                            ),
                        });
                    }
                    crate::sandbox::windows_impl::WindowsConfinement::WriteRestricted(
                        roots.to_vec(),
                    )
                }
            };
            let relay = OutputRelay::for_current_call(&self.frontend);
            return run_windows_sandboxed(
                &command,
                &windows_confinement,
                crate::workspace::cwd_snapshot(&self.cwd),
                timeout_ms,
                cancellation,
                relay,
            )
            .await;
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
            let (profile, params) = crate::sandbox::sandbox_profile_for(confinement.writable());
            let mut cmd = tokio::process::Command::new(crate::sandbox::SANDBOX_EXEC_PATH);
            cmd.arg("-p").arg(&profile);
            // `-D KEY=value` pairs, so a path never has to survive SBPL string quoting.
            cmd.args(&params);
            cmd.arg("sh").arg("-c").arg(&command);
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
            cmd.args(bwrap_args(
                confinement.writable(),
                &crate::workspace::cwd_snapshot(&self.cwd),
            ));
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
            // Built here, in the parent, because `pre_exec` runs after `fork` in a
            // single-threaded child where allocating is not async-signal-safe. A root whose bytes
            // contain a NUL cannot become a `CString`; dropping it leaves that root read-only,
            // which is the restrictive direction.
            #[cfg(target_os = "linux")]
            let landlock_writable: Vec<std::ffi::CString> = confinement
                .writable()
                .iter()
                .filter_map(|root| {
                    std::ffi::CString::new(std::os::unix::ffi::OsStrExt::as_bytes(root.as_os_str()))
                        .ok()
                })
                .collect();
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
                        crate::sandbox::apply_landlock(abi, &landlock_writable)
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
        // `ask` and `unrestricted` keep the full parent environment, and for `ask` that is a
        // deliberate widening rather than a consequence nobody noticed: this predicate is the same
        // `sandboxed` flag that decides confinement, so unconfining `ask`'s shell also unscrubbed
        // it. An approved command therefore reads `ANTHROPIC_API_KEY` and every other parent
        // secret. That is the chosen semantics -- an approved command should reach as far as an
        // approved `write_file`, environment included -- and it is what the prompt is buying: the
        // user has seen the command before it runs. It is stated in `docs/book/src/tools/shell.md`
        // rather than left to be discovered. The Windows sandboxed branch applies the same scrub
        // inside its own spawn.
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
async fn run_windows_sandboxed(
    command: &str,
    confinement: &crate::sandbox::windows_impl::WindowsConfinement,
    cwd: std::path::PathBuf,
    timeout_ms: u64,
    cancellation: CancellationToken,
    relay: Option<OutputRelay>,
) -> Result<ToolOutput> {
    use std::{sync::Arc, time::Duration};

    // Bound the post-kill cleanup wait so a stuck `TerminateProcess` or a drain task that somehow
    // fails to reach EOF can't hang the tool indefinitely. Two seconds is generous for kernel-side
    // teardown.
    const POST_KILL_TIMEOUT: Duration = Duration::from_secs(2);

    let mut sandboxed =
        crate::sandbox::windows_impl::spawn_sandboxed_command(command, confinement, &cwd).map_err(
            |error| MekaError::ToolExecution {
                tool_name: "execute_command".to_string(),
                message: format!("failed to spawn sandboxed command: {}", error),
            },
        )?;

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
            Permission::Unrestricted,
            crate::permission::EnabledPermissions::ALL,
        )
    }

    /// Construct an `ExecuteCommandTool` for tests with a backend probe matching whatever the host
    /// actually supports. Tests that need a specific probe state (e.g. exercising the "backend
    /// unavailable" hard-error path) should build `ExecuteCommandTool` directly with the desired
    /// `BackendProbe` rather than going through this helper.
    pub(super) fn test_tool(
        shared_permission: crate::permission::SharedPermission,
        sandbox_enabled: bool,
    ) -> ExecuteCommandTool {
        let sandbox_capability = crate::sandbox::detect();
        let backend_probe = crate::sandbox::BackendProbe::Ok(sandbox_capability.clone());
        ExecuteCommandTool {
            #[cfg(windows)]
            windows_grants: std::sync::Arc::new(
                crate::sandbox::windows_impl::WindowsGrants::default(),
            ),
            scope: crate::workspace::WriteScope::unconfined(),
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

    /// `workspace` refuses the shell outright when `[shell].sandbox = false` leaves nothing to
    /// confine it with.
    ///
    /// The failure this guards is silent and one-sided: the config key unconfines the shell while
    /// the file tools stay fenced, so `workspace` keeps reporting a boundary it is only half
    /// holding. `read` never had the problem, because `Read.allows(Unrestricted)` is false and the
    /// tool simply disappears; `Workspace.allows` is true for everything, so the refusal has to be
    /// here.
    #[tokio::test]
    async fn workspace_refuses_the_shell_when_the_sandbox_is_disabled() {
        // Every level that promises confinement, not just `workspace`.
        //
        // The guard is keyed on "any level that is not `ask` or `unrestricted`" precisely because
        // narrowing it to `workspace` alone left a hole: `[tools.tool_permissions]
        // execute_command = "read"` plus `[shell].sandbox = false` ran a plain `sh -c` at `read`,
        // with the full parent environment since the scrub is gated on the same flag. This test
        // only ever exercised `workspace`, so narrowing the guard back survived it.
        for level in [Permission::None, Permission::Read, Permission::Workspace] {
            refuses_at(level).await;
        }
    }

    async fn refuses_at(level: Permission) {
        let workspace_perm = crate::permission::SharedPermission::new(
            level,
            crate::permission::EnabledPermissions::ALL,
        );
        let tool = ExecuteCommandTool {
            #[cfg(windows)]
            windows_grants: std::sync::Arc::clone(crate::sandbox::windows_impl::process_grants()),
            scope: crate::workspace::WriteScope::confined(vec![]),
            sandbox_capability: crate::sandbox::SandboxCapability::Unavailable,
            sandbox_backend: crate::config::SandboxBackend::Bubblewrap,
            backend_probe: crate::sandbox::BackendProbe::Missing {
                reason: "sandbox disabled in config".to_string(),
            },
            shared_permission: workspace_perm,
            sandbox_enabled: false,
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
                assert!(
                    message.contains("[shell].sandbox = false"),
                    "the refusal must name the key responsible: {}",
                    message
                );
            }
            other => {
                panic!("expected a hard error at {level} with no sandbox, got {other:?}")
            }
        }
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
            #[cfg(windows)]
            windows_grants: std::sync::Arc::new(
                crate::sandbox::windows_impl::WindowsGrants::default(),
            ),
            scope: crate::workspace::WriteScope::unconfined(),
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

    /// At `unrestricted` an unavailable sandbox backend must NOT short-circuit the spawn: that
    /// level promises no boundary, so there is nothing for a missing backend to fail to provide.
    /// `workspace` and `read` are the opposite case and are refused outright, which is what makes
    /// this arm worth pinning separately.
    #[tokio::test]
    async fn test_execute_command_runs_without_sandbox_when_unrestricted() {
        let unrestricted_perm = crate::permission::SharedPermission::new(
            Permission::Unrestricted,
            crate::permission::EnabledPermissions::ALL,
        );
        let tool = ExecuteCommandTool {
            #[cfg(windows)]
            windows_grants: std::sync::Arc::new(
                crate::sandbox::windows_impl::WindowsGrants::default(),
            ),
            scope: crate::workspace::WriteScope::unconfined(),
            sandbox_capability: crate::sandbox::SandboxCapability::Unavailable,
            sandbox_backend: crate::config::SandboxBackend::Bubblewrap,
            backend_probe: crate::sandbox::BackendProbe::Missing {
                reason: "bwrap not found on PATH".to_string(),
            },
            shared_permission: unrestricted_perm,
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
            .expect("should succeed at unrestricted");
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
                // 50 000 "x" characters, in each host shell's own vocabulary. The Unix spelling
                // is POSIX-portable -- `head` and `tr` rather than bash brace expansion, so it
                // works under `dash` -- and the Windows one is PowerShell, which is the shell
                // `execute_command` actually invokes there.
                serde_json::json!({
                    "command": if cfg!(windows) {
                        "Write-Output ('x' * 50000)"
                    } else {
                        "head -c 50000 /dev/zero | tr '\\0' x"
                    }
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

        /// Where the process-read probe's parent leaves the target for its child.
        ///
        /// A file in the child's working directory rather than an environment variable, because
        /// the sandboxed spawn path hands the child a *curated* environment on purpose and a
        /// probe-specific variable has no business being added to that allow-list.
        const PROBE_HANDOFF: &str = "probe-target.txt";

        /// Marks the probe child's one line of output, so the parent can tell a real verdict from
        /// a child that never ran. Without it a host where the re-entry silently failed would read
        /// as "no read happened", which is the same shape as success. It earned its keep on the
        /// first hardware run, catching a filter that selected zero tests.
        const PROBE_VERDICT_PREFIX: &str = "MEKA-PROBE-VERDICT ";

        /// The probe child's libtest name, which `--exact` needs in full.
        ///
        /// Derived from [`module_path!`] rather than written out, so moving the module cannot leave
        /// a filter that silently matches nothing. `module_path!` is crate-qualified and libtest's
        /// names are not, so the leading crate segment comes off.
        fn probe_child_test_name() -> String {
            let module = module_path!()
                .split_once("::")
                .map_or(module_path!(), |(_crate_name, rest)| rest);
            format!("{module}::windows_process_read_probe_child")
        }

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
                scope: crate::workspace::WriteScope::unconfined(),
                windows_grants: std::sync::Arc::new(
                    crate::sandbox::windows_impl::WindowsGrants::default(),
                ),
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

        /// The `workspace` restricted-token path, end to end, on real Windows.
        ///
        /// This is the whole Windows half of the mode and nothing in the suite reached it: the two
        /// tests below drive the Low-integrity path, which is a different token, a different set of
        /// spawn flags, and no ACE at all. Everything specific to `workspace` -- the
        /// `WRITE_RESTRICTED` token, the synthesized capability SID, the inheritable ACE, the
        /// console a restricted child must inherit rather than create, and `lpCurrentDirectory` --
        /// only runs here.
        ///
        /// Both directions are checked in one command, because a token that denies *everything*
        /// would pass a refused-outside test on its own.
        #[tokio::test]
        async fn a_workspace_shell_writes_inside_the_root_and_is_refused_outside() {
            let temp = tempfile::tempdir().expect("tempdir");
            let base = crate::workspace::canonical_for_test(temp.path());
            let work = base.join("work");
            let outside = base.join("outside");
            std::fs::create_dir(&work).expect("work");
            std::fs::create_dir(&outside).expect("outside");

            let mut tool = windows_test_tool(crate::permission::SharedPermission::new(
                Permission::Workspace,
                crate::permission::EnabledPermissions::ALL,
            ));
            tool.cwd = std::sync::Arc::new(std::sync::RwLock::new(work.clone()));
            tool.scope = crate::workspace::WriteScope::confined(vec![work.clone()]);

            let result = tool
                .execute(
                    serde_json::json!({
                        "command": format!(
                            "Set-Content -Path '{}\\inside.txt' -Value 'in'; \
                             Set-Content -Path '{}\\escaped.txt' -Value 'out'",
                            work.display(),
                            outside.display()
                        ),
                    }),
                    CancellationToken::new(),
                )
                .await
                .expect("execute should not error");

            // Ground truth on disk, not the tool's narration: a shell that never started would
            // report failure just as convincingly as one the ACE confined.
            assert!(
                work.join("inside.txt").exists(),
                "a write inside the granted root must land, but did not: {result:?}"
            );
            assert!(
                !outside.join("escaped.txt").exists(),
                "a write outside every root must be refused by the token, not by meka: {result:?}"
            );

            tool.windows_grants.revoke_all();
        }

        /// The probe half of [`a_confined_child_reads_our_memory_at_workspace_and_cannot_at_read`].
        ///
        /// Re-entered as a child process rather than shelled out to, because the natural scripted
        /// probe cannot be trusted here: at `workspace` the `WRITE_RESTRICTED` token puts
        /// PowerShell into ConstrainedLanguage mode, where constructing the .NET types such a probe
        /// needs fails outright, and that failure is indistinguishable from a denied handle. This
        /// ran as a hand-cross-compiled C binary until the staging step became the only thing
        /// keeping the parent `#[ignore]`d.
        ///
        /// Stays ignored so a plain suite run never selects it, and no-ops when the handoff file is
        /// absent so selecting it by hand does nothing either. The parent passes the target through
        /// that file rather than the environment, because the spawn path hands the child a curated
        /// environment by design.
        #[tokio::test]
        #[ignore = "re-entered by its parent test; does nothing on its own"]
        async fn windows_process_read_probe_child() {
            let Ok(handoff) = std::fs::read_to_string(PROBE_HANDOFF) else {
                return;
            };
            let mut parts = handoff.split_whitespace();
            let (Some(pid), Some(address)) = (parts.next(), parts.next()) else {
                return;
            };
            let (Ok(pid), Ok(address)) = (pid.parse::<u32>(), usize::from_str_radix(address, 16))
            else {
                return;
            };

            use windows_sys::Win32::System::{
                Diagnostics::Debug::ReadProcessMemory,
                Threading::{OpenProcess, PROCESS_QUERY_INFORMATION, PROCESS_VM_READ},
            };

            // SAFETY: `pid` names a live process (the parent, which is blocked awaiting this
            // child), and the buffer is sized from itself. A denied handle comes back null and is
            // reported rather than dereferenced.
            let verdict = unsafe {
                let handle = OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, 0, pid);
                if handle.is_null() {
                    format!(
                        "OPEN_FAILED {}",
                        std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)
                    )
                } else {
                    let mut buffer = [0u8; 64];
                    let mut read = 0usize;
                    let ok = ReadProcessMemory(
                        handle,
                        address as *const std::ffi::c_void,
                        buffer.as_mut_ptr().cast(),
                        buffer.len(),
                        &mut read,
                    );
                    windows_sys::Win32::Foundation::CloseHandle(handle);
                    if ok == 0 {
                        format!(
                            "READ_FAILED {}",
                            std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)
                        )
                    } else {
                        let text: String = buffer[..read]
                            .iter()
                            .copied()
                            .take_while(|byte| byte.is_ascii_graphic() || *byte == b' ')
                            .map(char::from)
                            .collect();
                        format!("READ_OK {text}")
                    }
                }
            };
            println!("{PROBE_VERDICT_PREFIX}{verdict}");
        }

        /// A `workspace` child can read meka's own process memory; a `read` child cannot.
        ///
        /// The two levels are compared in one test because either result alone is uninformative: a
        /// token that denied everything would pass a "refused" assertion on its own merits, and a
        /// host where the probe simply failed to run would look like confinement.
        ///
        /// **The `read` leg is the guard.** Low integrity has to deny `OpenProcess` against meka,
        /// and nothing else in the suite defends that. **The `workspace` leg is a tripwire on the
        /// documentation.** It asserts the measured weakness still exists, so hardening the spawn
        /// path fails this test and forces `docs/book/src/usage/permissions.md` to be corrected
        /// rather than left quietly wrong in the safe direction.
        ///
        /// Why the weakness exists: `WRITE_RESTRICTED` intersects the restricting SIDs for write
        /// access only, and the `workspace` path deliberately leaves the integrity label alone so
        /// ordinary tooling keeps working. Neither restricts a *read*, and meka's memory holds
        /// provider credentials. Measured on hardware 2026-08-24: `workspace` returned the canary,
        /// `read` returned `ERROR_ACCESS_DENIED`.
        ///
        /// What this does **not** show is that an attacker could locate those credentials unaided.
        /// The parent hands the child the exact address, so this measures the capability -- a
        /// readable handle plus a successful read -- and not a search.
        #[tokio::test]
        async fn a_confined_child_reads_our_memory_at_workspace_and_cannot_at_read() {
            // Leaked so the marker stays mapped for as long as the child needs to read it.
            let secret: &'static str =
                Box::leak("MEKA-CANARY-7f3a91d4c8e2".to_string().into_boxed_str());
            let temp = tempfile::tempdir().expect("tempdir");
            let work = crate::workspace::canonical_for_test(temp.path());
            let runner = std::env::current_exe().expect("the test binary's own path");

            for (label, permission) in [
                ("workspace", Permission::Workspace),
                ("read", Permission::Read),
            ] {
                // Written fresh per leg, and inside the child's working directory, which is the
                // one path both confinements agree the child can reach.
                std::fs::write(
                    work.join(PROBE_HANDOFF),
                    format!("{} {:x}", std::process::id(), secret.as_ptr() as usize),
                )
                .expect("hand the target to the child");

                let mut tool = windows_test_tool(crate::permission::SharedPermission::new(
                    permission,
                    crate::permission::EnabledPermissions::ALL,
                ));
                tool.cwd = std::sync::Arc::new(std::sync::RwLock::new(work.clone()));
                tool.scope = crate::workspace::WriteScope::confined(vec![work.clone()]);

                let result = tool
                    .execute(
                        serde_json::json!({
                            "command": format!(
                                "& '{}' {} --ignored --exact --nocapture",
                                runner.display(),
                                probe_child_test_name()
                            ),
                        }),
                        CancellationToken::new(),
                    )
                    .await
                    .expect("execute should not error");

                let reported = format!("{result:?}");
                tool.windows_grants.revoke_all();
                assert!(
                    reported.contains(PROBE_VERDICT_PREFIX),
                    "the probe child did not report a verdict, so nothing below is meaningful: \
                     {reported}"
                );

                match permission {
                    // The canary itself, not just `READ_OK`: a read that succeeded against the
                    // wrong page and returned zeroes would satisfy the weaker check.
                    Permission::Workspace => assert!(
                        reported.contains("READ_OK") && reported.contains(secret),
                        "the `{label}` child could not read our memory. If the spawn path was \
                         hardened this is the good outcome, but permissions.md still documents \
                         the old one: {reported}"
                    ),
                    _ => assert!(
                        reported.contains("OPEN_FAILED"),
                        "`{label}` runs at Low integrity, which must refuse OpenProcess against \
                         meka: {reported}"
                    ),
                }
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

/// The Bubblewrap workspace binding, exercised against a real `bwrap`.
///
/// The Landlock dialect had a live confinement test and Bubblewrap had none, which left the
/// ordering rule in `bwrap_args` unguarded: bind before the tmpfs masks and every workspace under
/// `/tmp` silently comes out read-only, with bwrap reporting success. Both halves are checked here,
/// the order in the argument list and the outcome on disk, because the first is what a future edit
/// would break and the second is what a user would feel.
#[cfg(all(test, target_os = "linux"))]
mod bubblewrap_boundary {
    use std::path::PathBuf;

    /// The level `execute_command` needs depends on whether a sandbox can actually confine it.
    ///
    /// `read` only when the sandbox is both enabled *and* backed by a working backend; otherwise
    /// `unrestricted`, because a command meka cannot confine is a command only the boundary-free
    /// level may authorise. The conjunction is the whole rule and flipping it to `||` survived the
    /// suite: the tool would be offered at `read` with `[shell].sandbox = false`, or with the
    /// sandbox on but no usable backend. The runtime guard in `execute` still refuses the command
    /// in both cases, so this is a wrong catalogue entry rather than an escape -- but the catalogue
    /// is what the model plans against.
    #[test]
    fn the_shell_needs_unrestricted_whenever_nothing_can_confine_it() {
        use crate::{permission::Permission, sandbox::SandboxCapability, tools::Tool};

        let check = |sandbox_enabled: bool, capability: SandboxCapability, expected: Permission| {
            let mut tool = super::tests::test_tool(
                crate::permission::SharedPermission::new(
                    Permission::Read,
                    crate::permission::EnabledPermissions::ALL,
                ),
                sandbox_enabled,
            );
            tool.sandbox_capability = capability.clone();
            assert_eq!(
                tool.required_permission(),
                expected,
                "sandbox_enabled={sandbox_enabled}, capability={capability:?}"
            );
        };

        // Nothing can confine: either meka was told not to, or the host offers no backend.
        check(
            true,
            SandboxCapability::Unavailable,
            Permission::Unrestricted,
        );
        check(
            false,
            SandboxCapability::Unavailable,
            Permission::Unrestricted,
        );

        // Which backend is available does not enter the rule, and the variants are per-platform, so
        // the positive leg uses whatever this host actually has. Skipped rather than faked where
        // there is none: an invented variant would assert against a state meka cannot reach here.
        let available = crate::sandbox::detect();
        if matches!(available, SandboxCapability::Unavailable) {
            eprintln!("skipping the confined leg: no sandbox backend on this host");
            return;
        }
        check(true, available.clone(), Permission::Read);
        check(false, available, Permission::Unrestricted);
    }

    /// A cwd that *is* a masked directory must not be bound back over its own mask.
    ///
    /// The cwd bind and the tmpfs masks obey the same rule -- last mount wins -- so the fix that
    /// made a masked-directory session usable also handed it the host directory. Measured against
    /// real bwrap before the guard: a session at `/tmp` saw 1330 host entries instead of 0 and
    /// could `connect()` the tmux socket, one at `$XDG_RUNTIME_DIR` reached the session bus, and
    /// one at `/` saw 455 host PIDs instead of 4, which defeats `--unshare-pid` as well. A
    /// read-only bind does not help, because `connect(2)` on a socket inode is not a write.
    ///
    /// `/` is in the table because it is systemd's default working directory for a daemon, so `meka
    /// serve` under a unit file lands there without anyone choosing it.
    ///
    /// The counterpart is `the_child_is_given_a_working_directory_it_can_reach`: a path merely
    /// *under* a mask still needs its bind, and still gets one.
    #[test]
    fn a_masked_working_directory_is_not_bound_back_over_its_own_mask() {
        let masked = [
            std::path::PathBuf::from("/tmp"),
            std::path::PathBuf::from("/var/tmp"),
            std::path::PathBuf::from("/run"),
            std::path::PathBuf::from("/"),
        ];
        for cwd in masked {
            let text: Vec<String> = super::bwrap_args(&[], &cwd)
                .iter()
                .map(|arg| arg.to_string_lossy().into_owned())
                .collect();

            // The bind is what undoes the mask, so its absence is the property. Checked as an
            // adjacent pair rather than by searching for the path alone, because `/tmp` also
            // appears as a `--tmpfs` operand and `/` as the `--ro-bind / /` operand.
            let rebound = text.windows(3).any(|window| {
                window[0] == "--ro-bind-try"
                    && window[1] == cwd.to_string_lossy()
                    && window[2] == cwd.to_string_lossy()
            });
            assert!(
                !rebound,
                "binding {} back over its own mask restores the host directory the mask hides, \
                 which is the sandbox escape `is_system_root` exists to prevent: {text:?}",
                cwd.display()
            );

            // And the child still has somewhere to stand: the mask leaves an empty tmpfs at that
            // path, so `--chdir` succeeds and nothing is silently relocated to `$HOME`.
            let chdir = text
                .iter()
                .position(|arg| arg == "--chdir")
                .expect("the cwd must still be requested explicitly");
            assert_eq!(
                text.get(chdir + 1).map(String::as_str),
                Some(cwd.to_string_lossy().as_ref()),
                "the masked cwd is still where the child starts"
            );
        }
    }

    /// The child is told which directory to start in, and can read it.
    ///
    /// bwrap's fallback when it cannot enter the pre-`execve` cwd is silent and lands the child in
    /// `$HOME`. Confirmed against real bwrap with a cwd under `/tmp`: without these two arguments
    /// `pwd` reported the user's home directory and the workspace was unreachable even by absolute
    /// path, with exit 0 and empty stderr; with them `pwd` is correct, the file reads, and a write
    /// is still refused read-only at `read`.
    ///
    /// The bind sits after the masks and before the writable binds, so a cwd under `/tmp` is
    /// restored, and a cwd that is also a writable root is upgraded to read-write by the loop that
    /// follows -- last mount wins.
    #[test]
    #[cfg(target_os = "linux")]
    fn the_child_is_given_a_working_directory_it_can_reach() {
        let cwd = PathBuf::from("/tmp/session-cwd");
        let args = super::bwrap_args(&[], &cwd);
        let text: Vec<String> = args
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();

        let chdir = text.iter().position(|arg| arg == "--chdir");
        assert!(
            chdir.is_some(),
            "the cwd must be requested explicitly: {text:?}"
        );
        assert_eq!(
            text.get(chdir.expect("checked above") + 1)
                .map(String::as_str),
            Some("/tmp/session-cwd")
        );

        let bind = text
            .iter()
            .position(|arg| arg == "--ro-bind-try")
            .expect("the cwd must be bound back in, or read mode cannot see it");
        let last_mask = text
            .iter()
            .rposition(|arg| arg == "--tmpfs")
            .expect("the masks are always present");
        assert!(
            bind > last_mask,
            "a cwd bound before the masks is undone by them: {text:?}"
        );

        // A cwd that is also a writable root ends up read-write, because the rw bind comes later.
        let rw = super::bwrap_args(std::slice::from_ref(&cwd), &cwd);
        let rw: Vec<String> = rw
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        let ro_at = rw.iter().position(|arg| arg == "--ro-bind-try");
        let rw_at = rw.iter().position(|arg| arg == "--bind-try");
        assert!(
            ro_at < rw_at,
            "the writable bind must win over the read-only one: {rw:?}"
        );
    }

    #[test]
    fn every_workspace_bind_comes_after_every_mask() {
        let args = super::bwrap_args(
            &[PathBuf::from("/tmp/work")],
            std::path::Path::new("/tmp/work"),
        );
        let last_mask = args
            .iter()
            .rposition(|arg| arg == "--tmpfs")
            .expect("the masks are part of the recipe");
        let bind = args
            .iter()
            .position(|arg| arg == "--bind-try")
            .expect("the workspace root is bound");
        assert!(
            bind > last_mask,
            "a bind before a mask is undone by it, silently: {args:?}"
        );
    }

    #[test]
    fn a_bubblewrapped_shell_writes_inside_the_root_and_is_refused_outside() {
        let Some(bwrap) = which_bwrap() else {
            // Not `#[ignore]`: this must run wherever bwrap exists, and skipping loudly beats a
            // test that silently never runs on the machines that have the backend.
            eprintln!("skipping: bwrap is not on PATH");
            return;
        };

        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let work = base.join("work");
        std::fs::create_dir(&work).expect("work");

        // The "outside" target lives in `$HOME`, not beside the workspace.
        //
        // A sibling under the tempdir is itself under `/tmp`, which the recipe masks with a tmpfs,
        // so a write there fails with ENOENT -- the directory does not exist inside the namespace
        // at all. That is not the boundary refusing anything, and it would keep passing with the
        // boundary removed. `$HOME` is present and writable outside the sandbox, so a refusal there
        // is the ruleset's doing.
        let outside = crate::workspace::canonical_for_test(
            std::env::var_os("HOME")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(std::env::temp_dir),
        )
        .join(format!("meka-bwrap-probe-{}", uuid::Uuid::new_v4()));
        assert!(!outside.exists(), "the probe target must start absent");

        // `base` is under `/tmp` on virtually every machine, so this is also the regression: the
        // root has to survive the `--tmpfs /tmp` that the recipe applies before binding it.
        let script = format!(
            "echo in > {}/inside.txt 2>/dev/null || exit 3\n\
             if echo out > {} 2>/dev/null; then exit 4; fi\n\
             exit 0",
            work.display(),
            outside.display()
        );

        let status = std::process::Command::new(bwrap)
            .args(super::bwrap_args(std::slice::from_ref(&work), &work))
            .arg("--")
            .arg("sh")
            .arg("-c")
            .arg(&script)
            .status()
            .expect("spawn bwrap");

        match status.code() {
            Some(0) => {}
            Some(3) => panic!("the write inside the workspace root was refused"),
            Some(4) => panic!("the write outside every root was permitted"),
            other => panic!("bwrap did not run the command: exit {other:?}"),
        }
        // The bytes are visible outside the namespace, which is what makes the bind a bind rather
        // than a tmpfs the child happened to be able to write.
        assert_eq!(
            std::fs::read_to_string(work.join("inside.txt")).expect("read back"),
            "in\n"
        );
        assert!(
            !outside.exists(),
            "the write outside every root must not have landed in $HOME: {}",
            outside.display()
        );
        let _ = std::fs::remove_file(&outside);
    }

    fn which_bwrap() -> Option<std::path::PathBuf> {
        let path = std::env::var_os("PATH")?;
        std::env::split_paths(&path)
            .map(|dir| dir.join("bwrap"))
            .find(|candidate| candidate.is_file())
    }
}

/// `execute_command` at `workspace`, end to end, on a real Unix sandbox.
///
/// The Linux dialects are each tested through their *helpers* (`bwrap_args`, `apply_landlock`) with
/// a hand-built root list, so nothing exercised the wire between `Confinement` and the backend.
/// Cutting it -- `bwrap_args(&[])` and `apply_landlock(abi, &[])` -- left the whole suite green
/// while making the workspace shell silently read-only, which is the mode's central promise. The
/// only end-to-end test of this path was `#[cfg(windows)]`.
#[cfg(all(test, unix))]
mod workspace_shell_boundary {
    use std::sync::Arc;

    use tokio_util::sync::CancellationToken;

    use super::*;

    #[tokio::test]
    async fn a_workspace_shell_writes_inside_the_root_and_is_refused_outside() {
        // Every backend this host can actually run, not just the one `detect()` names.
        //
        // `detect()` on Linux only consults `probe_landlock`, so it never returns `Bubblewrap` --
        // while production resolves the backend through `resolve_sandbox_backend`, which
        // auto-prefers Bubblewrap whenever `bwrap` probes OK. Testing only what `detect()` returns
        // therefore left `Confinement::writable() -> bwrap_args` unexercised end to end on the
        // backend most hosts actually use: the argv could be cut entirely and this stayed green.
        let mut backends = Vec::new();
        let detected = crate::sandbox::detect();
        if !matches!(detected, crate::sandbox::SandboxCapability::Unavailable) {
            backends.push(detected);
        }
        backends.extend(a_backend_detect_does_not_name());
        // Skip rather than fail where no backend exists: this asserts what confinement does, and a
        // host without one has nothing to assert against. Loud, so it cannot silently never run.
        if backends.is_empty() {
            eprintln!("skipping: no usable sandbox backend on this host");
            return;
        }

        for capability in backends {
            eprintln!("workspace boundary against {:?}", capability);
            a_workspace_shell_boundary_holds_for(capability).await;
        }
    }

    /// A backend production would choose that [`crate::sandbox::detect`] does not name.
    ///
    /// Linux only. `detect()` there consults `probe_landlock` alone, so it never returns
    /// `Bubblewrap`, while production resolves through `resolve_sandbox_backend`, which
    /// auto-prefers Bubblewrap whenever `bwrap` probes OK.
    ///
    /// Split behind a `cfg` rather than pushed inline because `SandboxCapability::Bubblewrap` is
    /// itself `cfg(target_os = "linux")`: this module is `cfg(all(test, unix))`, so naming the
    /// variant unconditionally compiled here and failed the macOS build with `E0599`, which no
    /// amount of local Linux testing could have shown.
    #[cfg(target_os = "linux")]
    fn a_backend_detect_does_not_name() -> Option<crate::sandbox::SandboxCapability> {
        // Deliberately not `sandbox::bwrap_on_path`, which demands a root-owned binary: this only
        // answers "can this test spawn it", and a developer with a local build in `~/.local/bin`
        // should still get the leg run rather than silently skipped.
        let path = std::env::var_os("PATH")?;
        let bwrap_path = std::env::split_paths(&path)
            .map(|dir| dir.join("bwrap"))
            .find(|candidate| candidate.is_file())?;
        Some(crate::sandbox::SandboxCapability::Bubblewrap { bwrap_path })
    }

    /// macOS has one read-mode backend and `detect()` names it, so there is nothing to add.
    #[cfg(not(target_os = "linux"))]
    fn a_backend_detect_does_not_name() -> Option<crate::sandbox::SandboxCapability> {
        None
    }

    /// One backend's worth of the boundary check above.
    async fn a_workspace_shell_boundary_holds_for(capability: crate::sandbox::SandboxCapability) {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let work = base.join("work");
        let outside = base.join("outside");
        std::fs::create_dir(&work).expect("work");
        std::fs::create_dir(&outside).expect("outside");

        let mut tool = super::tests::test_tool(
            crate::permission::SharedPermission::new(
                Permission::Workspace,
                crate::permission::EnabledPermissions::ALL,
            ),
            true,
        );
        // The backend under test, not whatever `detect()` picked.
        tool.backend_probe = crate::sandbox::BackendProbe::Ok(capability.clone());
        tool.sandbox_capability = capability;
        tool.cwd = Arc::new(std::sync::RwLock::new(work.clone()));
        tool.scope = crate::workspace::WriteScope::confined(vec![work.clone()]);

        let result = tool
            .execute(
                serde_json::json!({
                    "command": format!(
                        "echo in > {}/inside.txt 2>/dev/null; \
                         echo out > {}/escaped.txt 2>/dev/null; true",
                        work.display(),
                        outside.display()
                    ),
                }),
                CancellationToken::new(),
            )
            .await
            .expect("the shell itself must run");

        // Ground truth on disk, not the tool's narration: a shell that never started would report
        // failure just as convincingly as one the sandbox confined.
        assert!(
            work.join("inside.txt").exists(),
            "a write inside the workspace root must land: {result:?}"
        );
        assert!(
            !outside.join("escaped.txt").exists(),
            "a write outside every root must be refused by the backend: {result:?}"
        );
    }
}
