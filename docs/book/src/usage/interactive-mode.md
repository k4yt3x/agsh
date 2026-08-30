# Interactive Mode

Start meka without `--oneshot` to enter interactive mode:

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
| Shift+Tab | Cycle the permission mode, skipping any not in `[permissions].enabled` (by default none &rarr; read &rarr; workspace &rarr; unrestricted &rarr; none) |

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
| Workspace | `[w]` | Orange |
| Unrestricted | `[u]` | Red |

The color provides a visual cue about the agent's current capabilities. Orange means the agent can modify your system inside the workspace roots; red means it can modify anything you can.

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
| `/permission [none\|read\|workspace\|ask\|unrestricted]` | Show or set the permission level |
| `/provider [profile]` | Show or change the provider profile this session runs on |
| `/compact` | Summarize and compact the session history |
| `/rewind [N]` | Drop the last `N` turns (default 1) from the conversation the model sees |
| `/fork` | Branch into a copy of this session, freezing the original where you are |
| `/export` | Write the session to `session-<id>.md` and print where it landed |
| `/cd [path]` | Change working directory; with no path, return to where meka was started (`~` still goes home) |
| `/schedule` | List this session's scheduled jobs |
| `/schedule cancel <id>` | Cancel a scheduled job by id or unique prefix |
| `/tasks` | List this session's [background tasks](./background.md) |
| `/tasks cancel <id>` | Stop one background task by id or unique prefix |
| `/tasks cancel --all` | Stop every running background task |
| `/mcp list` | List configured MCP servers with their live state (`pending` / `connected` / `failed` / `disabled`) |
| `/mcp reconnect <server>` | Smoke-test connect for one server |
| `/mcp login <server>` | Run the OAuth flow from the REPL |
| `/mcp logout <server>` | Revoke cached credentials for a server |
| `/mcp <server>:<prompt> [args...]` | Render a server-defined prompt and send it to the agent |
| `/status` | Show the resolved model/provider/effort/thinking, plus live context-window usage and cumulative turns, tokens, cache hit ratio, redactions, message count |
| `/usage` | Show the account's rate-limit usage (subscription providers): session/weekly windows, percent used, reset times |
| `/history [N]` | Reprint past conversation styled like the live REPL. Bare `/history` dumps everything; `/history N` shows the last `N` turns |

Press **Tab** after typing `/` to open a completion menu of command names, each shown with its description; keep typing to narrow it (`/comp` + Tab completes to `/compact`). Tab also completes arguments: permission levels for `/permission`, configured profile names for
`/provider`, installed skill names for `/skill`, the subcommands and configured servers for `/mcp`, and directory paths for `/cd` (Tab again after a completed directory drills into its subdirectories). The leading command token is colored as you type: an accent color when it names a known command, an error color when it does not.

### `/history`

Replays prior messages in the current session so you can scroll back through context without exiting and re-resuming. `/history` with no argument dumps every materialised message; `/history 5` shows the last 5 turns (a *turn* = the user's prompt plus everything the agent did to respond). Any non-numeric argument (`/history all`, `/history foo`) falls back to the dump-everything path.

The renderer mimics the live REPL: assistant text flows through the same markdown highlighter, tool calls honour [`display.tool_params`](../configuration/config-file.md#displaytool_params) (by default a one-line `[tool ReadFile(...)]` indicator), and thinking blocks honour `[thinking].show_content`. User prompts are prefixed with a cyan `>` so they stand out from agent text.

For users who always want extra context at resume time, set [`display.resume_show_recent`](../configuration/config-file.md#displayresume_show_recent); the resume code path then renders the last N turns through the same function.

### `/status`

Print the session's resolved model parameters followed by its cumulative counters:

```
Session status
  Provider:        claude-max (claude-subscription)
  Model:           claude-opus-4-8
  Context:         128.4k / 1.0M (13% used, 871.6k left)
  Effort:          xhigh
  Thinking:        adaptive
  Turns:           23
  Input tokens:    234.5k  (cache hit: 92%)
  Output tokens:   12.1k
  Redactions:      2 (12 images, ~38 MiB freed)
  Messages:        47
```

The top block reports what the session actually resolved to, in the order
[`[providers.<name>]`](../configuration/config-file.md#providers) declares the same fields,
so the two can be read side by side: the active profile and its backend (`type`), the `Model`, the
`Context` window, the reasoning `Effort` sent on the wire (omitted when nothing is sent, so the
provider applies its own default; `claude-subscription` sends `high` when the profile sets none),
and the `Thinking` mode. The rest are cumulative counters for the session.

`Context` is the live context-window occupancy: the total tokens of the most recent exchange (all input tiers plus output, i.e. what the next request re-sends minus your new prompt), against the active model's context window, with the percent used and tokens remaining. Use it to decide whether to `/compact` before continuing; after `/compact` it drops to the compacted size immediately. It reflects this session only; sub-agents spawned via `agent_spawn` have their own context and are not counted (a sub-agent's returned result is counted only once it lands in this session as a tool result). It is shown from the start, at `0 / <window>` before the first turn, since the window is your `context_window` setting (or the documented default) and this is where you confirm it took effect; it is omitted only when the window is unknown. Set [`display.show_context_in_prompt`](../configuration/config-file.md#displayshow_context_in_prompt) to show the same gauge in the prompt itself.

`Input tokens` (and the other cumulative counters) is the total billed across every turn of the whole session. These totals are persisted, so resuming a session with `meka -c` continues them rather than restarting at zero.

`cache hit` is the share of input tokens served from the prompt cache rather than re-sent at full price. It should climb quickly and stay high: meka keeps everything that changes mid-session out of the cached prefix, so a steady session re-reads the cache instead of rewriting it. Expect it to drop once after a `/compact` (which rewrites the head of the conversation) and to recover on the following turns.

`Redactions` reports any times the Claude provider had to drop oldest tool-result image blocks because the request body would have exceeded Anthropic's 32 MiB ceiling. A non-zero count indicates the cache prefix was invalidated for the redacted messages. See [`display.show_token_usage`](../configuration/config-file.md#displayshow_token_usage) for a per-turn variant of the same data.

### `/usage`

Fetch the account's current rate-limit usage from the active provider and print each rolling window with its percentage used and reset time:

```
Account usage
  5-hour (session)   [#---------]   8% used  (resets in 4h 12m, 2026-07-02 02:10)
  Weekly             [----------]   2% used  (resets in 22h 50m, 2026-07-02 13:00)
```

This is distinct from `/status`, which reports this session's own token counters. `/usage` queries the provider for your whole-account subscription limits. It works only for OAuth subscription providers that expose a usage endpoint (`claude-subscription`'s 5-hour and weekly windows; `chatgpt-subscription`'s primary/secondary windows plus plan and credit balance). For API-key backends, OpenAI-compatible endpoints, and Ollama, it prints a short "not available for this provider" note instead. The same command is available under ACP.

### `/compact`

The `/compact` command asks the LLM to summarize the entire conversation, then replaces the messages the model sees with a single summary message followed by the recent tail. This is useful for long sessions that are approaching the context window limit or becoming expensive.

After compacting, the session continues with the summary as context. The pre-compaction messages are never deleted: they stay in the underlying event log on disk (the model just no longer sees them). `meka session export` walks that full log, so an export always contains the entire conversation including the compacted-away turns, with a marker at each compaction point.

### `/rewind`

`/rewind` drops the most recent turn from the conversation, so the model no longer sees it or your prompt that started it. `/rewind N` drops the last `N`. The cut always lands on a turn boundary, so a tool call is never separated from its result.

Like `/compact`, nothing is deleted: the dropped turns stay in the event log on disk, and `meka session export` still shows them with a marker where the rewind happened.

Use it to take back a prompt that sent the agent down the wrong path without paying for a summary, or to recover a session the provider has started refusing. meka repairs a rejection it causes itself (see below), but content that entered the conversation earlier is out of its reach; rewinding past it is the way back. `meka session rewind <id>` does the same to a session you are not currently in.

### Recovering from a rejected message

Providers validate the whole conversation on every request, so one piece of content they refuse would otherwise fail every later turn as well, permanently. When that happens, meka strips the offending content from what it added this turn, retries once, and hands the model the provider's own complaint as a failed tool result so it can adapt rather than silently losing the data. If the retry is refused too, the original content goes back untouched and the turn reports the provider's error.

A mislabelled image already committed to the session is repaired when you resume it, without a provider round trip. For anything further back, use `/rewind`.

### Recovering from a call that got no answer

A refusal is one thing the provider says; a request that never got a usable reply at all is another. A connection that fails or is reset while the request is going out is retried with backoff (up to twice, waiting 1s then 2s), and so is a response body that could not be read back. The turn continues as if the failed attempt had not happened, and nothing about it enters the conversation. Only when the retries run out does the turn report the error.

Worth knowing what a retry can cost. When the failure was a body that could not be read, the provider had already generated the response and billed you for it, so the retry pays a second time. meka does it anyway, because the alternative is losing the turn for content you have already been charged for once, but it is not free.

Two failures are not retried, because the next attempt is known not to be worth making: a request meka could not build, and a URL that redirects in a loop. A redirect loop points at a misconfigured `base_url`; a request that could not be built points at whatever went into it, most often a `base_url` that is not a URL or a stored credential carrying a character that cannot go in a header.

Retrying is bounded by time as well as by count, and the time bound is the one that usually decides. A failure that takes the full read timeout to arrive costs five minutes, which spends the whole budget, so a call that hung is reported rather than tried again: retrying is for a failure that was cheap, and a provider that went silent for five minutes has already taken more of your turn than a second silence is worth. Without the bound at all, three slow failures would be fifteen minutes of waiting on a turn that fails anyway.

The bound stops a *new* attempt starting rather than capping the total, so the worst case is a failure arriving just under the five minutes and permitting one more full-length attempt after it, for about ten in total.

### `/fork`

`/fork` copies the current session and switches you into the copy, printing its ID. Your conversation carries over untouched, so the branch happens exactly where you are; the original stops there and keeps everything up to that point.

Use it before trying a direction you might want to back out of, or before `/compact` if you'd rather keep the uncompacted conversation around. To go back, exit and resume the original with `meka -r <old-id>`.

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
