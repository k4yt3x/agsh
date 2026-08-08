# Interactive Mode

Start meka without the `-p` flag to enter interactive mode:

```bash
meka
```

You get a prompt:

```text
meka [r] >
```

Type your instruction and press **Enter** to submit. The agent processes your request and prints its response (streamed in real time as Markdown). When it finishes, you get another prompt.

## Keybindings

meka uses Emacs-style keybindings (provided by reedline).

### Input

| Key | Action |
|-----|--------|
| Enter | Submit the current prompt |
| Alt+Enter | Insert a newline (for multi-line input) |
| Shift+Tab | Cycle the permission mode (none &rarr; read &rarr; ask &rarr; write &rarr; none) |

### Navigation

| Key | Action |
|-----|--------|
| Ctrl+A | Move cursor to start of line |
| Ctrl+E | Move cursor to end of line |
| Ctrl+F | Move cursor forward one character |
| Ctrl+B | Move cursor backward one character |
| Alt+F | Move cursor forward one word |
| Alt+B | Move cursor backward one word |
| Up / Down | Recall the previous / next input from history |

### Editing

| Key | Action |
|-----|--------|
| Ctrl+D | Delete character under cursor / exit on empty line |
| Ctrl+H, Backspace | Delete character before cursor |
| Ctrl+K | Kill text from cursor to end of line |
| Ctrl+U | Kill text from start of line to cursor |
| Ctrl+W | Kill word before cursor |
| Ctrl+Y | Yank (paste) killed text |

### Control

| Key | Action |
|-----|--------|
| Ctrl+C | Interrupt the running agent; clear the line if idle |
| Ctrl+D | Exit the shell (when the line is empty) |
| Ctrl+R | Reverse incremental search through history |
| Ctrl+L | Clear the screen |

### Input History

The prompts you type are saved to meka's SQLite database, so Up / Down and Ctrl+R recall what
you typed in **any previous run**. A brand-new `meka`, a resumed `meka -c`, and the current
session all share one history. Multi-line prompts are preserved intact, and only the most recent
entries are kept (older ones are pruned). This input history is separate from the conversation
shown by `/history`.

## Prompt Format

```text
meka [indicator] >
```

The indicator shows the current permission mode:

| Mode | Indicator | Color |
|------|-----------|-------|
| None | `[n]` | Green |
| Read | `[r]` | Yellow |
| Ask | `[a]` | Magenta |
| Write | `[w]` | Red |

The color provides a visual cue about the agent's current capabilities. Red means the agent can modify your system.

## Multi-Line Input

Press **Alt+Enter** to insert a newline instead of submitting. The prompt changes to show continuation:

```text
meka [r] > write a python script that
  ... prints hello world
  ... and saves it to hello.py
```

Press **Enter** on the last line to submit the entire multi-line input.

Pasting multi-line content also works seamlessly: all pasted lines appear in the buffer for review, and you press **Enter** to submit.

## Slash Commands

meka supports `/` prefix commands for controlling the shell:

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/exit` | Exit the shell |
| `/clear` | Clear the terminal screen |
| `/session` | Show the current session ID |
| `/permission [none\|read\|ask\|write]` | Show or set the permission level |
| `/compact` | Summarize and compact the session history |
| `/fork` | Branch into a copy of this session, freezing the original where you are |
| `/cd [path]` | Change working directory |
| `/mcp list` | List configured MCP servers with their live state (`pending` / `connected` / `failed` / `disabled`) |
| `/mcp reconnect <server>` | Smoke-test connect for one server |
| `/mcp login <server>` | Run the OAuth flow from the REPL |
| `/mcp logout <server>` | Revoke cached credentials for a server |
| `/mcp <server>:<prompt> [args...]` | Render a server-defined prompt and send it to the agent |
| `/status` | Show the resolved model/provider/effort/thinking, plus live context-window usage and cumulative turns, tokens, cache hit ratio, redactions, message count |
| `/usage` | Show the account's rate-limit usage (subscription providers): session/weekly windows, percent used, reset times |
| `/history [N]` | Reprint past conversation styled like the live REPL. Bare `/history` dumps everything; `/history N` shows the last `N` turns |

Press **Tab** after typing `/` to open a completion menu of command names, each shown with its description; keep typing to narrow it (`/comp` + Tab completes to `/compact`). Tab also completes arguments: permission levels for `/permission`, installed skill names for `/skill`, the subcommands and configured servers for `/mcp`, and directory paths for `/cd` (Tab again after a completed directory drills into its subdirectories). The leading command token is colored as you type: an accent color when it names a known command, an error color when it does not.

### `/history`

Replays prior messages in the current session so you can scroll back through context without exiting and re-resuming. `/history` with no argument dumps every materialised message; `/history 5` shows the last 5 turns (a *turn* = the user's prompt plus everything the agent did to respond). Any non-numeric argument (`/history all`, `/history foo`) falls back to the dump-everything path.

The renderer mimics the live REPL: assistant text flows through the same markdown highlighter, tool calls appear as `[tool ReadFile(...)]` indicators, and thinking blocks honour `[thinking].show_content`. User prompts are prefixed with a cyan `>` so they stand out from agent text.

For users who always want extra context at resume time, set [`display.resume_show_recent`](../configuration/config-file.md#displayresume_show_recent); the resume code path then renders the last N turns through the same function.

### `/status`

Print the session's resolved model parameters followed by its cumulative counters:

```
Session status
  Model:           claude-opus-4-8
  Provider:        claude-max (claude-oauth)
  Effort:          xhigh
  Thinking:        on
  Turns:           23
  Context:         128.4k / 1.0M (13% used, 871.6k left)
  Input tokens:    234.5k  (cache hit: 92%)
  Output tokens:   12.1k
  Redactions:      2 (12 images, ~38 MiB freed)
  Messages:        47
```

The top block reports what the session actually resolved to: the model, the active profile and its backend, the reasoning `Effort` sent on the wire (the line is omitted for models with no effort knob), and whether thinking is on. This is where a model-aware default such as `xhigh` becomes visible. The rest are cumulative counters for the session.

`Context` is the live context-window occupancy: the total tokens of the most recent exchange (all input tiers plus output, i.e. what the next request re-sends minus your new prompt), against the active model's context window, with the percent used and tokens remaining. Use it to decide whether to `/compact` before continuing; after `/compact` it drops to the compacted size immediately. It reflects this session only; sub-agents spawned via `spawn_agent` have their own context and are not counted (a sub-agent's returned result is counted only once it lands in this session as a tool result). The line is omitted until the first turn completes. Set [`display.show_context_in_prompt`](../configuration/config-file.md#displayshow_context_in_prompt) to show the same gauge in the prompt itself.

`Input tokens` (and the other cumulative counters) is the total billed across every turn of the whole session. These totals are persisted, so resuming a session with `meka -c` continues them rather than restarting at zero.

`Redactions` reports any times the Claude provider had to drop oldest tool-result image blocks because the request body would have exceeded Anthropic's 32 MiB ceiling. A non-zero count indicates the cache prefix was invalidated for the redacted messages. See [`display.show_token_usage`](../configuration/config-file.md#displayshow_token_usage) for a per-turn variant of the same data.

### `/usage`

Fetch the account's current rate-limit usage from the active provider and print each rolling window with its percentage used and reset time:

```
Account usage
  5-hour (session)   [#---------]   8% used  (resets in 4h 12m, 2026-07-02 02:10)
  Weekly             [----------]   2% used  (resets in 22h 50m, 2026-07-02 13:00)
```

This is distinct from `/status`, which reports this session's own token counters. `/usage` queries the provider for your whole-account subscription limits. It works only for OAuth subscription providers that expose a usage endpoint (`claude-oauth`'s 5-hour and weekly windows; `openai-codex`'s primary/secondary windows plus plan and credit balance). For API-key backends, OpenAI-compatible endpoints, and Ollama, it prints a short "not available for this provider" note instead. The same command is available under ACP.

### `/compact`

The `/compact` command asks the LLM to summarize the entire conversation, then replaces the messages the model sees with a single summary message followed by the recent tail. This is useful for long sessions that are approaching the context window limit or becoming expensive.

After compacting, the session continues with the summary as context. The pre-compaction messages are never deleted: they stay in the underlying event log on disk (the model just no longer sees them). `meka session export` walks that full log, so an export always contains the entire conversation including the compacted-away turns, with a marker at each compaction point.

### `/fork`

`/fork` copies the current session and switches you into the copy, printing its ID. Your conversation carries over untouched, so the branch happens exactly where you are; the original stops there and keeps everything up to that point.

Use it before trying a direction you might want to back out of, or before `/compact` if you'd rather keep the uncompacted conversation around. To go back, exit and resume the original with `meka -c <old-id>`.

The copy is a fully independent session with no link back to its source. See [Forking a Session](./sessions.md#forking-a-session) for exactly what it carries.

## Shell Escape

Prefix any input with `!` to execute it directly as a shell command, bypassing the LLM entirely:

```text
meka [r] > !pwd
/home/user/projects
meka [r] > !ls -la
total 32
drwxr-xr-x  5 user user 4096 Mar  4 10:00 .
...
meka [r] > !ping 1.1.1.1 -c 2
PING 1.1.1.1 (1.1.1.1) 56(84) bytes of data.
...
```

The command runs with inherited stdin/stdout/stderr, so it behaves exactly like a regular shell. This is useful for quick checks without waiting for the LLM.

## Exiting

You can exit meka in any of these ways:

- Type `/exit`
- Type `exit` or `quit`
- Press **Ctrl+D** on an empty line

## Interrupting the Agent

Press **Ctrl+C** while the agent is running to interrupt it. This cancels the current LLM request and kills any running shell commands that were spawned by the agent.
