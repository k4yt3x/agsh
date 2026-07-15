# Permissions

meka uses a four-level permission system to control what tools the agent can use. This gives you control over the agent's capabilities and prevents accidental modifications.

## Permission Levels

| Level | Indicator | Allowed Tools |
|-------|-----------|---------------|
| **None** | `[n]` (green) | No tools. The agent can only respond with text. |
| **Read** | `[r]` (yellow) | Read-only tools: `read_file`, `find_files`, `search_contents`, `fetch_url`, `web_search`, `execute_command` (sandboxed), `todo`, `spawn_agent`, scratchpad tools |
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
[ask] Shell ls -la (Y/n)
```

Press **Enter** or **y** to approve, or **n** to deny. If denied, the agent receives an error and may try an alternative approach.

This mode is useful when you want the agent to have full capabilities but want to review each action before it executes.

## How Permissions Work

When the agent attempts to use a tool, meka checks whether the current permission level allows it:

- If allowed, the tool executes normally.
- In ask mode, you are prompted to approve or deny.
- If denied, meka returns an error message to the agent explaining which level is required and suggests running `/permission <level>`.

### Telling the agent the current level

meka lists **every registered tool** in the system prompt with its required permission level inline (nothing is filtered out), and each user message carries a compact `[Permission context]` block:

```text
<context>
[Permission context]
Current permission level: read
Only read-only tools are executable.
...
</context>
```

That two-line block is the only permission-dependent content in the request. The system prompt and the tools-array schemas stay byte-identical across `/permission` toggles, so mid-session level changes don't invalidate the Claude prompt cache; the entire conversation stays warm.

### MCP tool permissions

MCP tools are classified through a 5-step resolution chain: per-tool override → server-level override → the server's own `readOnlyHint` → `[mcp].default_permission` → hardcoded `Write` fallback. See the *Permission resolution* section of the [Config File](../configuration/config-file.md) docs for the full rules and how to override a misclassified tool.

### Built-in tool permissions

Any built-in tool's required permission can be overridden from `config.toml` without editing code; see [`[tools]`: built-in tool filters](../configuration/config-file.md#tools-built-in-tool-filters). The same section documents how to allow-list or block-list specific built-ins (e.g. disabling `web_search` in a locked-down environment).

### Sub-agent permissions

Sub-agents spawned via `spawn_agent` inherit the parent's permission level by default. In write mode the sub-agent can call `write_file`, `edit_file`, and unsandboxed `execute_command`; in read mode it's confined to read-only tools. To run one delegated task with reduced privileges, pass the `permission` parameter (e.g. `spawn_agent({prompt: "...", permission: "read"})`): it is clamped to the parent's level as a ceiling, so a sub-agent can only ever be equal-or-more restricted, never escalated. Alternatively, cycle the parent into a lower mode before issuing the spawning prompt to restrict every sub-agent it spawns.

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

> **Note:** The read-only sandbox uses Landlock on Linux (kernel 5.13+) and sandbox-exec on macOS. On platforms where sandboxing is unavailable, shell commands are not available in read mode. You can disable sandboxed shell execution by setting `sandbox = false` under `[shell]` in the config file (see [Config File](../configuration/config-file.md)).

### Write Mode

```text
meka [w] > run cargo test and show me the output
```

The agent uses `execute_command` to run the tests and shows the results.
