# Shell Tool

## `execute_command`

Execute a shell command and return its output.

**Permission:** Read (sandboxed) / Write (unsandboxed)

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `command` | string | yes | The shell command to execute |
| `timeout_ms` | integer | no | Timeout in milliseconds (default: 30000) |
| `scratchpad` | string | no | Save output to the scratchpad under this name |

### Behavior

- Executes the command via `sh -c "<command>"` on Unix, or `powershell.exe -NoProfile -NonInteractive -Command "<command>"` on Windows (same shell in both sandboxed and unsandboxed mode).
- Captures both stdout and stderr.
- Returns the exit code along with the output if non-zero.
- Oversized output is losslessly persisted to the scratchpad by the agent layer; the tool itself never truncates.
- Default timeout is 30 seconds. If the command exceeds the timeout, it is killed (on Unix, via the process group so backgrounded grandchildren are caught too).
- Supports cancellation: pressing Ctrl+C while a command is running kills the child process.

### Shell-specific semantics

- **Unix (`sh -c`)**: POSIX `$VAR` expansion applies. Pass a literal `$` with single quotes (`'$foo'`) or backslash escape (`\$foo`).
- **Windows (`powershell.exe -Command`)**: The script body reaches PowerShell directly. Use PowerShell syntax (`$var = ...`, `$env:PATH`), and crucially, **do not** wrap your command in another `powershell -Command "..."`. The outer PowerShell will expand your inner `$var` references to empty strings before the inner shell runs, producing a parser error on mangled syntax. If you need to invoke a nested script, drop it into a `.ps1` file and run it by path, use `-EncodedCommand <base64>`, or escape each `$` as `` `$ ``.

### Read-Only Sandbox

In **read mode**, commands run inside a sandbox that blocks writes to the user's real data. Reads, program execution, and network access still work normally: the threat model is "no state mutation, but `curl http://x | pdftotext` must keep working."

#### What's blocked vs allowed (across all backends)

| Surface | Blocked | Allowed |
|---|---|---|
| Filesystem writes outside tmp / Low-integrity paths | ✓ | |
| Filesystem reads | | ✓ |
| Program execution | | ✓ |
| Outbound network (TCP/UDP) | | ✓ |
| dbus / systemd-user state mutations | Bubblewrap / macOS / Landlock on kernel 7.1+ | Landlock below kernel 7.1 / Windows |
| Mach IPC state mutation (launchd, pasteboard, LaunchServices) | macOS | Linux / Windows |
| COM / RPC to Low-integrity-accepting services (Windows) | | ✓ |
| Inheritance of sensitive parent env vars (API keys, OAuth tokens, …) | ✓ (all platforms) | |

The sandbox is not an adversarial containment boundary; it's defense-in-depth against an agent accidentally modifying user data. Set permission to `none` if you don't trust a turn at all.

#### Environment variable scrubbing

Read-mode sandboxes still permit outbound network (the threat model intentionally keeps `curl http://x | pdftotext`-style pipelines working), so any secret in the parent process's environment (`ANTHROPIC_API_KEY`, `AWS_SECRET_ACCESS_KEY`, `GITHUB_TOKEN`, OAuth tokens, etc.) would be a live exfiltration vector under prompt injection. meka scrubs the child environment at spawn time across every backend (Bubblewrap, Landlock, Seatbelt, Windows Low-integrity).

- **Unix (Linux + macOS): allow-list.** Only a curated set of vars survives into the read-mode child: `PATH`, `HOME`, `USER`, `LOGNAME`, `SHELL`, `PWD`, `TERM`, `COLORTERM`, `LANG`, `TMPDIR`, `TMP`, `TEMP`, plus everything matching the `LC_*` and `XDG_*` prefixes. Anything else is dropped, including credential-shaped vars (`AWS_*`, `GITHUB_TOKEN`, `OPENAI_API_KEY`, …) and credential-pointer vars (`SSH_AUTH_SOCK`, `KUBECONFIG`, `GNUPGHOME`, `NETRC`, `GIT_ASKPASS`, `GIT_SSH_COMMAND`, etc.) as well as benign-but-unlisted vars like `EDITOR`, `PAGER`, `DISPLAY`, custom toolchain vars, and so on. Unknown vars are dropped by default.
- **Windows: deny-list.** PowerShell pulls in a long tail of system vars (`PSModulePath`, `APPDATA`, `ProgramFiles`, etc.) that don't fit a tidy allow-list, so the Windows path lets everything through *except* names that match a heuristic deny-list. Dropped names include:
    - Credential-shaped substrings: `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `*PASSPHRASE*`, `*API_KEY*`, `*_KEY*`, `*BEARER*`, `*CREDENTIAL*`, etc.
    - Credential-pointer substrings: `SSH_AUTH_SOCK`, `KUBECONFIG`, `GNUPGHOME`, `NETRC`, `GIT_ASKPASS`, `SSH_ASKPASS`, `GIT_SSH_COMMAND`.
    - Provider / service prefixes: `ANTHROPIC_*`, `OPENAI_*`, `AWS_*`, `GCP_*`, `GOOGLE_*`, `AZURE_*`, `GITHUB_*`, `OPENROUTER_*`, `GROQ_*`, `MISTRAL_*`, `COHERE_*`, `DATABASE_*`, `POSTGRES_*`, `MONGO_*`, `STRIPE_*`, `CLOUDFLARE_*`, `VAULT_*`, `OAUTH_*`, `JWT_*`, `SENTRY_*`, `SLACK_*`, `DISCORD_*`, and others; see `is_sensitive_env_name` in `src/sandbox.rs` for the full list.

  The deny-list is intentionally aggressive on false positives (a legitimate `GITHUB_ACTOR` is dropped alongside `GITHUB_TOKEN`) because the cost of a missing env var is a confusing tool error, while the cost of a leaked credential is a live exfiltration channel.

**Write mode keeps the full parent environment.** Write mode is the trusted-operation path where users legitimately need `NPM_TOKEN` for `npm publish`, `AWS_*` creds for `aws s3 cp`, `GH_TOKEN` for `gh pr create`, etc. If you need a specific var inside a read-mode shell command, switch to write mode for that turn.

#### Linux: pick a backend

Linux supports two backends, selected via `[shell].sandbox_backend` in `config.toml`:

- **Bubblewrap** (`sandbox_backend = "bubblewrap"`, recommended): wraps the command in `bwrap` with `--ro-bind /`, tmpfs masks over `/run`, `/tmp`, `/var/tmp`, and `$XDG_RUNTIME_DIR`, plus `--unshare-user --unshare-pid --unshare-uts --unshare-ipc`. The tmpfs masks make the dbus session bus, systemd-user socket, and other socket-on-disk IPC paths unreachable, so `systemctl --user start <unit>`, `dbus-send`, and similar state-changing calls fail. Network is not unshared. Requires the `bubblewrap` package and a kernel with user-namespace creation enabled.
- **Landlock** (`sandbox_backend = "landlock"`, legacy / fallback): uses the [Landlock LSM](https://landlock.io/) (kernel 5.13+). Blocks filesystem writes via `landlock_restrict_self`. On kernel 7.1+ (Landlock ABI v9) it also blocks `connect()` to every Unix socket on disk, which closes the dbus / systemd-user route out of the sandbox but likewise breaks socket-based clients such as `docker` and `psql` in read mode. **Below kernel 7.1 the kernel offers no such right**, so a sandboxed shell can invoke state-mutating dbus methods and `systemd-run --user` escapes the filesystem restriction entirely. Prefer Bubblewrap, which removes those sockets on any kernel.

`sandbox_backend` is unset unless you pin it yourself; `meka provider add` does not write it. When unset, meka probes Bubblewrap once at startup and prefers it when available, falling back to Landlock with a one-shot warning that points at the install path and the suppress-this-warning escape hatch.

```toml
[shell]
sandbox = true                       # default; set to false to disable
sandbox_backend = "bubblewrap"       # or "landlock"; unset = auto-detect
```

#### macOS and Windows

- **macOS**: Uses `sandbox-exec` with a hardened SBPL profile (modeled after [Codex](https://github.com/openai/codex)'s vendored seatbelt policy, which is itself based on Chrome's renderer sandbox). The profile is closed-by-default: filesystem writes are blocked, Mach-lookup is restricted to a curated allow-list of safe services, and mutation paths (launchd job control, pasteboard, LaunchServices, distributed notifications) are not in the allow-list. Network and DNS resolution remain available. The `sandbox_backend` config key is ignored.
- **Windows**: Spawns the child with a duplicated primary token dropped to **Low integrity** (`SECURITY_MANDATORY_LOW_RID`) via `SetTokenInformation(TokenIntegrityLevel, …)`. Writes to the home directory, `%APPDATA%`, Program Files, and system directories (any location with Medium-or-higher integrity ACLs) are blocked by the kernel. Low integrity also strips token privileges, and the same env scrubbing applied on Unix runs here (see [Environment variable scrubbing](#environment-variable-scrubbing) above). The `sandbox_backend` config key is ignored.

Low integrity is not a total write-denial: the child can still write to the small residual Low-integrity-writable surface (`%LOCALAPPDATA%\Low`, `%TEMP%\Low`, any path with an explicit Low-integrity write ACE) and to files it creates itself.

#### When the configured backend is unavailable

If `sandbox_backend = "bubblewrap"` is set but `bwrap` isn't on `$PATH` (or user namespaces are denied), `execute_command` in read mode returns a hard error rather than silently falling back. The error names the configured backend and the specific failure reason. Either install `bubblewrap`, set `sandbox_backend = "landlock"`, or switch to write mode (Shift+Tab).

#### Disabling the sandbox entirely

To disable sandboxed shell execution in read mode altogether, set `sandbox = false` under `[shell]`. When disabled, shell commands require write mode.

```toml
[shell]
sandbox = false
```
