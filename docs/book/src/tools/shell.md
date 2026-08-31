# Shell Tool

## `execute_command`

Execute a shell command and return its output.

**Permission:** `read` (sandboxed read-only) / `workspace` (sandboxed, writable inside the workspace roots) / `ask` and `unrestricted` (unsandboxed)

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
- Oversized output is losslessly persisted to the scratchpad by the agent layer; the tool does not truncate what it returns to the agent, up to the residency ceiling below.
- There is no cap on how much a command may print, but there is a cap on how much of it meka holds in memory. Past 8 MiB on one stream the bytes are written to a file in the cache directory instead, and the tool result carries the first and last 32 KiB plus that file's path, so the whole capture stays reachable with `read_file`. This exists because a command that writes faster than the turn ends (`cat /dev/zero`, a runaway build log) previously grew one buffer until the process died.
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

#### Scratch space: one place the backends genuinely differ

A confined command may or may not get a writable temporary directory, and this is the one difference between backends big enough to change which commands work:

| Backend | Scratch space | Effect |
|---|---|---|
| Bubblewrap (`read`) | Private `/tmp` tmpfs | `mktemp`, `git`, `python`, `gpg`, `pip` all work |
| Landlock (`read`) | None | Anything that writes a temp file is denied |
| Windows `workspace` | None outside the roots | `New-TemporaryFile` is denied (measured) |
| macOS Seatbelt (`read`) | Per-backend; see below | |

Under Bubblewrap the child gets a private writable `/tmp`, so `mktemp` succeeds and the write goes nowhere real. Under Landlock there is no such directory and the write is simply denied, which takes `git`'s index lock, Python's `tempfile`, `gpg` and `pip` with it. The same is true of `workspace` on Windows outside the granted roots.

This divergence is deliberate. Granting a scratch directory under Landlock would weaken what `read` promises on the backend that currently keeps that promise strictly, so the narrower behaviour stays.

The practical cost is diagnostic: the model sees a bare `Permission denied` naming a path in `/tmp` (or `%TEMP%`), with nothing in the message connecting it to the sandbox, and cannot act on it. If a command fails that way and you expected it to work, install `bwrap` for Landlock hosts, or add the directory it wants as a writable root at `workspace`.

#### Environment variable scrubbing

Read-mode sandboxes still permit outbound network (the threat model intentionally keeps `curl http://x | pdftotext`-style pipelines working), so any secret in the parent process's environment (`ANTHROPIC_API_KEY`, `AWS_SECRET_ACCESS_KEY`, `GITHUB_TOKEN`, OAuth tokens, etc.) would be a live exfiltration vector under prompt injection. meka scrubs the child environment at spawn time across every backend (Bubblewrap, Landlock, Seatbelt, Windows Low-integrity).

- **Unix (Linux + macOS): allow-list.** Only a curated set of vars survives into the read-mode child: `PATH`, `HOME`, `USER`, `LOGNAME`, `SHELL`, `PWD`, `TERM`, `COLORTERM`, `LANG`, `TMPDIR`, `TMP`, `TEMP`, plus everything matching the `LC_*` and `XDG_*` prefixes. Because read mode intentionally keeps outbound network working, the proxy and CA-bundle vars survive too: `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`, `ALL_PROXY` (and their lowercase spellings), `SSL_CERT_FILE`, `SSL_CERT_DIR`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE` and `NODE_EXTRA_CA_CERTS` -- several of which redirect TLS trust, so treat them as part of the boundary. Anything else is dropped, including credential-shaped vars (`AWS_*`, `GITHUB_TOKEN`, `OPENAI_API_KEY`, …) and credential-pointer vars (`SSH_AUTH_SOCK`, `KUBECONFIG`, `GNUPGHOME`, `NETRC`, `GIT_ASKPASS`, `GIT_SSH_COMMAND`, etc.) as well as benign-but-unlisted vars like `EDITOR`, `PAGER`, `DISPLAY`, custom toolchain vars, and so on. Unknown vars are dropped by default.
- **Windows: deny-list.** PowerShell pulls in a long tail of system vars (`PSModulePath`, `APPDATA`, `ProgramFiles`, etc.) that don't fit a tidy allow-list, so the Windows path lets everything through *except* names that match a heuristic deny-list. Dropped names include:
    - Credential-shaped substrings: `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `*PASSPHRASE*`, `*API_KEY*`, `*_KEY*`, `*BEARER*`, `*CREDENTIAL*`, etc.
    - Credential-pointer substrings: `SSH_AUTH_SOCK`, `KUBECONFIG`, `GNUPGHOME`, `NETRC`, `GIT_ASKPASS`, `SSH_ASKPASS`, `GIT_SSH_COMMAND`.
    - Provider / service prefixes: `ANTHROPIC_*`, `OPENAI_*`, `AWS_*`, `GCP_*`, `GOOGLE_*`, `AZURE_*`, `GITHUB_*`, `OPENROUTER_*`, `GROQ_*`, `MISTRAL_*`, `COHERE_*`, `DATABASE_*`, `POSTGRES_*`, `MONGO_*`, `STRIPE_*`, `CLOUDFLARE_*`, `VAULT_*`, `OAUTH_*`, `JWT_*`, `SENTRY_*`, `SLACK_*`, `DISCORD_*`, and others; see `is_sensitive_env_name` in `src/sandbox.rs` for the full list.

  The deny-list is intentionally aggressive on false positives (a legitimate `GITHUB_ACTOR` is dropped alongside `GITHUB_TOKEN`) because the cost of a missing env var is a confusing tool error, while the cost of a leaked credential is a live exfiltration channel.

**`ask` and `unrestricted` keep the full parent environment.** These are the trusted-operation paths where users legitimately need `NPM_TOKEN` for `npm publish`, `AWS_*` creds for `aws s3 cp`, `GH_TOKEN` for `gh pr create`, etc. If you need a specific var inside a sandboxed shell command, switch to one of them for that turn.

For `ask` specifically this is worth stating outright, because the approval prompt shows you a *command* and not its environment: an approved `npm test` whose postinstall script reads `process.env` sees `ANTHROPIC_API_KEY` and every other secret in meka's environment, on a sandbox that deliberately leaves the network open. That is the same reach an approved `write_file` has, which is the point of the level -- the prompt is what you are trusting, not a scrub behind it. If you want the scrub, `workspace` keeps it and confines writes to the workspace roots.

#### Linux: pick a backend

Linux supports two backends, selected via `[shell].sandbox_backend` in `config.toml`:

- **Bubblewrap** (`sandbox_backend = "bubblewrap"`, recommended): wraps the command in `bwrap` with `--ro-bind /`, tmpfs masks over `/run`, `/tmp`, `/var/tmp`, and `$XDG_RUNTIME_DIR`, plus `--unshare-user --unshare-pid --unshare-uts --unshare-ipc`. The tmpfs masks make the dbus session bus, systemd-user socket, and other socket-on-disk IPC paths unreachable, so `systemctl --user start <unit>`, `dbus-send`, and similar state-changing calls fail. Network is not unshared. Requires the `bubblewrap` package and a kernel with user-namespace creation enabled.
- **Landlock** (`sandbox_backend = "landlock"`, legacy / fallback): uses the [Landlock LSM](https://landlock.io/). Blocks filesystem writes via `landlock_restrict_self`. **Requires ABI v3 (kernel 6.2+)**: below that the kernel does not mediate `truncate(2)`, so a sandboxed command could still empty an existing file despite every open-for-write being denied. meka reports Landlock unusable on those kernels rather than sandboxing with a ruleset that does not enforce what read mode promises, which means kernels 5.13–6.1 need Bubblewrap installed for read-mode shell. On kernel 7.1+ (ABI v9) Landlock also blocks `connect()` to every Unix socket on disk, which closes the dbus / systemd-user route out of the sandbox but likewise breaks socket-based clients such as `docker` and `psql` in read mode. **Between ABI v3 and v9 that right does not exist**, so a sandboxed shell can invoke state-mutating dbus methods and `systemd-run --user` escapes the filesystem restriction entirely; meka warns at startup naming exactly which mitigations the running ABI lacks. Prefer Bubblewrap, which removes those sockets on any kernel.

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

#### Windows at `workspace`

`workspace` uses a second mechanism, not the Low-integrity token above. meka derives a capability
SID from each workspace root, places an inheritable `GENERIC_WRITE | DELETE` ACE for it on that
root, and runs the shell under a `WRITE_RESTRICTED` token carrying that capability, so a write
succeeds exactly where one of those ACEs exists. Three consequences worth knowing before you use it:

- meka has to **own** the root, which is what lets it grant without elevation. A network share or
  another user's folder cannot be a workspace root.
- PowerShell runs in **ConstrainedLanguage** mode under a restricted token, so scripts that
  construct .NET types fail there while working at `unrestricted`. meka's UTF-8 output preamble is
  skipped for the same reason, so non-ASCII output may be mangled at `workspace`.
- The ACE is real, standing state on your directory, visible in `icacls`. It is released when the
  process exits, Ctrl+C included, but not after a crash or a kill.

See [Permissions](../usage/permissions.md#per-platform-enforcement) for the full account.

#### When the configured backend is unavailable

If `sandbox_backend = "bubblewrap"` is set but `bwrap` isn't on `$PATH` (or user namespaces are denied), `execute_command` in read mode returns a hard error rather than silently falling back. The error names the configured backend and the specific failure reason. Either install `bubblewrap`, set `sandbox_backend = "landlock"`, or switch to `unrestricted` (Shift+Tab).

#### Disabling the sandbox entirely

To disable sandboxed shell execution altogether, set `sandbox = false` under `[shell]`. When disabled, shell commands require `ask` or `unrestricted`: `read` loses the tool entirely, and `workspace` refuses it with an error naming the key, because there is no longer anything to hold the boundary that mode promises. Reach for `unrestricted` on those turns rather than expecting `workspace` to quietly run unconfined.

```toml
[shell]
sandbox = false
```
