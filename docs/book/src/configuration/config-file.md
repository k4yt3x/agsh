# Config File

meka looks for a TOML configuration file at a platform-specific location:

| Platform | Path |
|----------|------|
| Linux | `~/.config/meka/config.toml` (`$XDG_CONFIG_HOME/meka/config.toml`) |
| macOS | `~/Library/Application Support/meka/config.toml` |
| Windows | `%APPDATA%\meka\config.toml` |

The config file is optional. If it does not exist, meka silently skips it.

meka rejects unknown keys: a typo (`contex_window`) or a removed key (`reasoning_effort`) fails the load with an error naming the offending key, rather than being silently ignored. Fix or remove the key to continue.

The commands that *edit* the file are exempt, so a broken config can still be repaired from the CLI: `meka mcp add` / `remove` / `enable` / `disable` and `meka provider remove` work on the raw document and don't care about an unknown key elsewhere in it. Everything that *reads* config fails instead of answering from empty defaults, because "No MCP servers configured." over a file full of them is indistinguishable from the truth.

Those editors only reach the keys they own, so a bad key anywhere else (`[session]`, `[permissions]`, a top-level typo, a raw syntax error) has to be fixed in an editor. The error names the file, line, column, and offending key.

Set the `MEKA_CONFIG_DIR` environment variable to override the default location entirely. The value points at the `meka` directory itself (contains `config.toml` and `skills/`). Useful for tests, portable installs, and isolating a per-project config from your global one.

## Providers

Providers are configured as **named profiles** under `[providers.<name>]`. Each profile pins a
backend `type` plus its model and other non-secret knobs. You can keep several profiles side by
side (including multiple accounts of the same backend) and switch between them by name.

**Secrets are never stored in the config file.** API keys and OAuth token bundles live in meka's
database, keyed by profile name, and are acquired through the [`meka provider`](#meka-provider-cli)
command suite (`meka provider add` runs the API-key prompt or the OAuth login for you). The config
file holds only the non-secret settings shown below.

```toml
default_provider = "work"

[providers.work]
type  = "claude-subscription"
model = "claude-opus-5"

[providers.local]
type     = "openai-chat-completions"
base_url = "http://localhost:11434/v1"
model    = "llama3"
```

### Selecting the active profile

For each run meka picks one profile using this precedence:

1. `--provider <name>` CLI flag.
2. `default_provider` in the config file.
3. The sole profile, if exactly one is configured.

If none of these resolve (no profiles configured, or more than one with no `default_provider` /
`--provider`), meka errors and points you at `meka provider add` / `meka provider use`. Resuming a
session is the exception: it runs on the profile it recorded and never consults this, so an
ambiguous default does not block `meka -c`. There is no
environment-variable tier for provider selection; the config file (plus the per-run CLI flag) is the
source of truth.

### Timeouts

Every backend connects with a 30-second handshake deadline and fails a stream that produces nothing
for five minutes, which surfaces as a retryable error rather than a hung turn.

Neither is a limit on the turn. There is deliberately no cap on how long a turn may run, how many
tool calls it may make, or how many tokens it may spend: those ceilings belong to your API key and
your provider plan, not to the harness. What these bound is *silence*. A model that is still
thinking is still sending, so a stream that goes quiet for five minutes has died, and waiting on it
forever is not patience.

## `default_provider`

Top-level field naming the profile to use when `--provider` isn't passed. Set it with
`meka provider use <name>`; `meka provider add` sets it automatically when adding the first profile.

## Profile fields

### `type`

The backend the profile uses (required).

| Value | Protocol | Auth |
|-------|----------|------|
| `anthropic-messages` | Anthropic Messages, `POST {base}/v1/messages` | API key (`x-api-key`) |
| `claude-subscription` | Anthropic Messages, against `api.anthropic.com` | Claude subscription OAuth (fingerprinting + attestation) |
| `openai-chat-completions` | OpenAI Chat Completions, `POST {base}/chat/completions` | API key |
| `openai-responses` | OpenAI Responses, `POST {base}/responses` | API key |
| `chatgpt-subscription` | OpenAI Responses, against `chatgpt.com/backend-api/codex` | ChatGPT subscription OAuth |

A backend names the wire protocol it speaks, not a vendor. See [Providers Overview](../providers/overview.md) for why, and which servers implement which protocol.

### `model`

The model identifier to send to the provider, forwarded verbatim. meka does not gate which strings are valid, so an OpenAI-compatible endpoint accepts whatever that server exposes.

`meka provider add` suggests `claude-opus-5` for a Claude profile and `gpt-5.6-sol` for an OpenAI one. For the current line-ups, see [Anthropic's models overview](https://docs.claude.com/en/docs/about-claude/models/overview) and [OpenAI's models overview](https://platform.openai.com/docs/models); naming them here would go stale on someone else's schedule.

Override per-run with `--model`.

### `base_url`

Custom API base URL. Useful for:

- Self-hosted models via [Ollama](https://ollama.ai) (`http://localhost:11434/v1`)
- [OpenRouter](https://openrouter.ai) (`https://openrouter.ai/api/v1`)
- Other OpenAI-compatible API providers

If not set, defaults to:

- `https://api.openai.com/v1` for the `openai-chat-completions` and `openai-responses` backends
- `https://chatgpt.com` for the `chatgpt-subscription` backend (request path is `/backend-api/codex/responses`)
- `https://api.anthropic.com` for the `anthropic-messages` and `claude-subscription` backends

Override per-run with `--base-url`.

**The two API families end their base URL in different places, and that is not meka's choice.** An
OpenAI-compatible base includes the version segment, which is why every provider documents one
ending in `/v1` and why meka appends only `/chat/completions`. A Claude base is the host *root*,
because meka reaches two different roots off it: `/v1/messages` for the turn, and `/api/oauth/...`
for the subscription usage and profile endpoints. A base ending at `/v1` could not reach the second
set. The official SDKs draw the line the same way.

A gateway that fronts both APIs therefore publishes two URLs, and its Anthropic one is often written
with the `/v1` its OpenAI sibling needs (`https://api.synthetic.new/anthropic/v1`). Paste it as-is:
for a `anthropic-messages` or `claude-subscription` profile meka drops a trailing `/v1`, since it re-adds that
segment on every request and the alternative is a request to `/v1/v1/messages`. Only a trailing one
goes, so a base whose path legitimately contains `/v1` earlier
(`https://gateway.ai.cloudflare.com/v1/{account}/{gateway}/anthropic`) is left alone. Trailing
slashes are trimmed for every backend.

The reverse is not inferred: an `openai-chat-completions` base is used exactly as written, because a gateway
serving `/chat/completions` at its root is legitimate and meka cannot tell that apart from a missing
`/v1`. If an OpenAI-compatible endpoint 404s, check that the base carries the version segment its
documentation shows.

### `oauth_token_url`

Custom OAuth token refresh endpoint. Defaults:

- `https://api.anthropic.com/v1/oauth/token` for `claude-subscription`
- `https://auth.openai.com/oauth/token` for `chatgpt-subscription`

### `effort`

One knob for reasoning effort across every backend: Claude sends it as `output_config.effort` (`claude-subscription` under the `effort-2025-11-24` beta, `anthropic-messages` directly), OpenAI as `reasoning.effort` (with `max_completion_tokens` in place of `max_tokens`).

**When unset the field is omitted, and the provider applies its own default. `claude-subscription` is the exception: it sends `high`, matching Claude Code.** That is the point of leaving it unset: effort is a request parameter the provider owns, and omitting it is how you ask for whatever that provider considers right. meka picks no tier of its own, because it cannot know which tiers a given endpoint implements - `anthropic-messages` and `openai-chat-completions` reach any compatible server, including local ones serving weights that never had a reasoning knob, and a tier the backend doesn't implement is a rejected request rather than a graceful ignore.

An explicit value is absolute: sent verbatim (trimmed and lowercased), with no validation or clamping, whatever model it is aimed at. You own correctness for your model and endpoint; an invalid value is rejected by the API. A blank value reads as unset.

Typical values: `low`, `medium`, `high`, `xhigh`, `max`.

```toml
[providers.work]
type   = "claude-subscription"
effort = "xhigh"
```

### `redact_thinking`

`claude-subscription` only. Sends the `redact-thinking-2026-02-12` beta header for capable models, matching Claude Code, which enables it by default. With it on the server withholds the readable chain of thought: `thinking` blocks return with empty text plus a signature, and `redacted_thinking` blocks carry an opaque `data` payload. meka preserves and replays both verbatim, so multi-turn continuity holds. No reasoning text is shown for these models; in its place the REPL draws a live `Thinking... (150 tokens)` indicator from the server's running estimate, redrawn as the count climbs and left on screen when the phase ends, so a long silence reads as progress and stays legible afterwards. Defaults to `true`; set `false` to drop the beta and keep interleaved thinking visible.

```toml
[providers.work]
type            = "claude-subscription"
redact_thinking = false
```

### `context_window`

The model's context window (total tokens it can hold), used for the `/status` gauge and auto-compaction. Takes precedence over [`[session].context_window`](#sessioncontext_window); when neither is set, meka assumes **1000000**.

meka never infers this from the model name and never asks the provider for it, so this is where a model smaller than the default gets stated. It is a local budgeting number that is never sent on the wire, so a wrong value can't fail a request - but leaving it at 1M for a smaller model means planned compaction never fires, and every compaction instead happens after the provider rejects the request as too large, costing a wasted round trip each time.

The window belongs to the session, not to the process: each session is measured against the profile it recorded, so two sessions in one `meka serve` can sit on profiles with different windows.

```toml
[providers.work]
type           = "openai-chat-completions"
model          = "my-128k-model"
context_window = 131072
```

### `thinking`

Claude-only. How the request encodes extended thinking, and whether it asks for it at all:

| Value | Wire shape |
|-------|-----------|
| `adaptive` (default) | `thinking: {"type": "adaptive"}` - the model sets its own budget. Claude 4.6+ |
| `budgeted` | `thinking: {"type": "enabled", "budget_tokens": N}` from [`[thinking].budget_tokens`](#thinkingbudget_tokens). Required by pre-4.6 Claude, and the form most third-party Anthropic-compatible servers implement |
| `off` | No `thinking` field |

One knob rather than two: it replaces both the old on/off switch and the encoding meka used to infer from the model name. The right value depends on the model *and* on what the endpoint implements, which meka can't determine, so the profile states it - and a profile whose `model` later changes is yours to keep correct.

```toml
[providers.local]
type     = "anthropic-messages"
thinking = "budgeted"
```

### `vision`

Whether this profile's model accepts image input. Defaults to `true`. Set `false` for a text-only model so attachments are refused rather than sent to a model that cannot read them.

Refusal is per session, from the profile that session recorded, on both ACP and `POST /v1/sessions/{id}/turn`. What ACP *advertises* in `promptCapabilities.image` is necessarily per connection: `initialize` is answered before any session exists, so it reports the default profile's flag. A client on a vision-capable connection can still have its attachment refused by a session pinned to a text-only profile. See [ACP](../usage/acp.md).

```toml
[providers.local]
type   = "openai-chat-completions"
model  = "llama-3-8b"
vision = false
```

### `max_output_tokens`

Override the per-request output (completion) token cap. When unset, each backend keeps its built-in default (Claude 32k–64k depending on the [`thinking`](#thinking) mode; OpenAI 32k when an effort is set; otherwise the API default). Under `thinking = "budgeted"` the value must exceed `[thinking].budget_tokens` (validated at startup).

```toml
[providers.work]
type              = "anthropic-messages"
max_output_tokens = 16000
```

### `client_id`

OAuth client ID override (advanced; `claude-subscription` / `chatgpt-subscription` only). Leave unset to use meka's built-in default client IDs.

### `device_id`

`claude-subscription` only. Stable per-device identifier embedded in `metadata.user_id` to mirror Claude Code's `~/.claude.json` device ID (`getOrCreateUserID` in `utils/config.ts`).

If unset, meka first tries to adopt `userID` from `~/.claude.json` (so meka and Claude Code on the same machine look like the same device). If that file is missing or has no `userID`, meka generates a 64-character hex string. Either way, the resolved value is persisted back to the profile under `[providers.<name>].device_id`. This file write only happens for the `claude-subscription` backend; other backends don't need a device ID.

You can supply your own value if you want to control attribution explicitly:

```toml
[providers.work]
type      = "claude-subscription"
device_id = "your-stable-id-here"
```

## `meka provider` CLI

Add, switch, and remove profiles without editing `config.toml` by hand. The credential prompt /
OAuth login runs as part of `add` and `login`, and secrets are written to the database, never the
config file.

| Command | Action |
|---|---|
| `meka provider add <name> [--type T] [--model M] [--base-url U] [--api-key-stdin]` | Add a profile. Prompts for any of type/model interactively when not flagged (the model prompt offers a backend default: `claude-opus-5` for Claude, `gpt-5.6-sol` for OpenAI), then acquires the secret (OAuth login for `claude-subscription` / `chatgpt-subscription`, API-key prompt for `anthropic-messages` / `openai-chat-completions` / `openai-responses`). `--api-key-stdin` reads the key from stdin instead, and then needs `--type` and `--model` as flags too, since a prompt would consume the piped key; it is refused for the two subscription backends, which have no key to read. Becomes `default_provider` whenever none is set. |
| `meka provider list` | List configured profiles with type, model, the default marker, and whether each has a stored credential. Also names any stored credential that no profile claims (see [Leftover credentials](#leftover-credentials)). |
| `meka provider use <name>` | Set `default_provider` to this profile. |
| `meka provider login <name> [--api-key-stdin]` | Re-acquire the secret for an existing profile (re-authenticate, recover from a dead OAuth refresh token, or rotate an API key). `--api-key-stdin` reads the key from stdin for scripted rotation, and is refused on the subscription backends, which have no key to read. Every other setting on the profile is kept, which `remove` + `add` would not do. |
| `meka provider remove <name>` | Delete the stored credential from the database and remove the `[providers.<name>]` entry from the config file. Works on a name with only one of the two, so it can clean up after a hand-edit. Warns if it clears a `default_provider` that other profiles are still competing for, and if any sessions are pinned to the profile it deleted (those refuse to resume until it is configured again, or moved with `meka -r <id> --provider <name>`). |

`--api-key-stdin` reads the key from standard input instead of prompting, for scripted setup:

```console
$ printf '%s' "$OPENAI_API_KEY" | meka provider add local --type openai-chat-completions --model gpt-5.6-sol --api-key-stdin
```

### Leftover credentials

Adding a profile by hand works: write a `[providers.<name>]` block, then run `meka provider login
<name>` to attach the credential. Deleting one by hand is only half the job. Credentials live in the
database keyed by profile name, so removing the block takes the settings away and leaves the API key
or OAuth refresh token behind, still valid.

Nothing deletes it on your behalf. meka will not sweep the database against the config at startup:
`MEKA_CONFIG_DIR` and `MEKA_DATA_DIR` are independent, so a config read from the wrong place, or one
meka could not parse, would present as "no profiles configured" against a real database and take
every credential with it. Losing an OAuth refresh token that way means redoing the browser login for
each account.

Instead, `meka provider list` reports what it finds:

```console
$ meka provider list
Name  Type                Model          Authenticated  Default
work  anthropic-messages  claude-opus-5  yes            *

Stored credentials with no profile: archive
```

`meka provider remove archive` then deletes it. The same applies to MCP servers, reported by [`meka
mcp list`](#meka-mcp-cli) and cleaned by `meka mcp remove <name>`.

## Examples

### `claude-subscription`

```console
$ meka provider add work --type claude-subscription --model claude-opus-5
# Opens the browser for the OAuth login, then stores the token in the database.
```

### `anthropic-messages`

```console
$ meka provider add anthropic --type anthropic-messages --model claude-opus-5
# Prompts for your Anthropic API key (sk-ant-api03-...).
```

### `openai-chat-completions`

```console
$ meka provider add openai --type openai-chat-completions --model gpt-5.6-sol
# Prompts for your OpenAI API key (sk-...).
```

### `openai-responses`

```console
$ meka provider add openai --type openai-responses --model gpt-5.6-sol
# Prompts for your OpenAI API key (sk-...). Same key as openai-chat-completions,
# newer protocol; also reaches Ollama, vLLM, LM Studio and OpenRouter.
```

### `chatgpt-subscription`

```console
$ meka provider add chatgpt --type chatgpt-subscription --model gpt-5.6-sol
# Opens the browser for the ChatGPT OAuth login.
```

### Ollama (local, no key)

```console
$ printf 'unused' | meka provider add ollama --type openai-chat-completions --model llama3 \
    --base-url http://localhost:11434/v1 --api-key-stdin
```

### OpenRouter

```console
$ meka provider add openrouter --type openai-chat-completions --model anthropic/claude-sonnet-4.6 \
    --base-url https://openrouter.ai/api/v1
# Prompts for your OpenRouter key (sk-or-...).
```

## `[display]`

Settings for output formatting.

### `display.render_mode`

Output render mode. Equivalent to the `--render-mode` CLI flag.

| Value | Description |
|-------|-------------|
| `syntect` | Syntax-highlighted markdown source, incl. per-language code blocks; never reflowed |
| `termimad` | Rendered CommonMark, reflowed to the terminal: paragraphs re-wrap, wide tables wrap, markers are consumed. Same theme colours as `syntect`, and code blocks are highlighted by it. Alias: `rich` (default) |
| `raw` | Raw markdown printed verbatim with aligned tables |
| `silent` | No assistant output at all. For a run whose only product is its side effects |

Default: `termimad`

Reflowing only happens when there is a terminal to reflow to. With output redirected or piped,
`termimad` renders without wrapping, so a captured answer is not hard-wrapped to some fallback
width.

```toml
[display]
render_mode = "raw"
```

### `display.max_width`

Widest line meka composes from model output, in terminal columns.

Default: unset, meaning the terminal's own width, so nothing ever wraps.

Set it to pin the width instead:

```toml
[display]
max_width = 120
```

A set value is honoured exactly rather than clamped to the terminal, because pinning it is how you
get identical output across machines and a silent clamp would take that away on the narrow one. The
cost is that a value wider than your terminal wraps, and a wrapped row starts at column zero, where
meka's own output lives. Below 40 columns the value is clamped up and a warning is logged: every
budget subtracts fixed chrome first, and below roughly that the subtraction leaves nothing. Above
1000 it is clamped down, also with a warning, since no terminal is that wide and the value is far
more likely to be a typo than a request.

This covers meka's own output: tool indicators and their argument block, thinking previews, todo
lists, and the `ask` approval prompt. Assistant markdown is not affected and keeps reflowing to the
real terminal through [`display.render_mode`](#displayrender_mode). With output piped there is no
terminal to measure, so an unset width falls back to 100 columns and a captured run stays byte-stable.

A terminal narrower than 20 columns is treated as 20. That is not a legibility judgement: the
thinking block's own prefix is twelve columns, so below roughly that meka's chrome no longer fits and
the width stops meaning anything. Such a terminal wraps meka's output whatever the number says.

### `display.tool_params`

How much of a tool call's input the `[tool ...]` indicator shows.

This setting covers the indicator only. In `ask` permission mode the approval prompt always shows
every argument, whatever this is set to: the indicator is a notification, the prompt is a decision,
and setting `off` for a quiet scrollback must not leave you approving calls you cannot see.

| Value | Description |
|-------|-------------|
| `off` | Name only: `[tool Shell]`. No argument reaches your terminal |
| `summary` | Name plus the one argument that identifies the call: ``[tool Shell(`cargo test`)]`` (default) |
| `full` | Every argument, as an indented block under the name |

Default: `summary`

`full` writes each parameter on its own line. A value that fits on a line follows its key; one that
does not gets an indented block under a bare `key:`, so a multi-line `edit_file` argument stays
readable instead of collapsing into escaped newlines. Nesting is carried by indentation, with `-`
for array elements:

```
[tool EditFile]
  path: src/render.rs
  old_string:
    let first_line = thinking.lines().next().unwrap_or("");
    let truncated = truncate_display(first_line, 80);

[tool AgentSpawn]
  prompt: Audit the scheduler for missed-occurrence bugs
  tools:
    - read_file
    - search_contents
```

Consecutive calls are separated by a blank line under `full`, since each one is a block and running
them together reads as a single call with too many parameters. Under `summary` they stay flush, which
is what makes a run of them read as a list of steps.

This is a reading format, not a data format: quotes are dropped, so `timeout: 300` doesn't say
whether the model sent `300` or `"300"`. Four caps keep one call from filling the screen, and each
says what it hid:

| Cap | Limit | Marker |
|-----|-------|--------|
| One argument's value | 30 lines | `... N more lines`, indented under that argument |
| One argument's rows | 32 rows | `... N more rows`, indented under that argument |
| The block | 60 rows, checked at an argument boundary | `... N more arguments: name, name` |
| One line | [`display.max_width`](#displaymax_width) | `...` at the cut |

The first two caps look redundant and are not. A string value has lines to count, so it is trimmed
by line and the marker counts lines. An array or an object has none: it fans out one row per element,
so it needs a bound counted in rows, and the marker says rows rather than pretending they were lines.

The line cap is exact, brackets and indentation included. The block cap is not: it is checked before
an argument is rendered rather than after, so the block reaches at most the block cap plus one
argument's own budget plus the line naming what went — 93 rows.

The block cap drops whole arguments and names them rather than cutting wherever row 60 lands.
Knowing that `path` was passed but not shown beats seeing 60 rows of `content` and never learning
which file it was written to.

**A cut keeps the end.** Where a whole argument is dropped it is named; where rows are dropped the
last one is kept, so a long array still shows its final element and a trimmed value still shows how
it finishes. The reasoning is the same one that elides a long path from its middle rather than its
tail: the end of a thing too big to show is usually the half that identifies it.

When you need the exact JSON a tool was called with, `meka session export` has it, untruncated and
unflattened.

**`full` puts every argument on screen, secrets included.** `summary` shows only the one argument
that identifies a call (`write_file`'s path, `fetch_url`'s URL), so a request header carrying a token
or a file body carrying a key stayed off screen. `full` shows all of them, and replayed history
reprints them on every `/history` and every resume. meka never puts its own credentials into tool
arguments, so what appears is what the model itself passed, but that is worth knowing before turning
this on where somebody can read over your shoulder or your scrollback.

Values are escape-stripped, their newlines and carriage returns flattened, and Unicode format
characters (bidi overrides, soft hyphens, zero-width joiners) removed, so an argument cannot move
your cursor, reorder what you read, or place text at column zero where meka's own output lives.

No line exceeds [`display.max_width`](#displaymax_width), so by default nothing wraps and no row ever
begins with model text. Setting `max_width` wider than your terminal gives that up, which is the one
case where a long argument can still produce a row starting flush left.

One residual caveat: the `... N more lines`, `... N more rows` and `... N more arguments` markers are
ordinary text, so an argument whose content mimics one is indistinguishable from a real elision. That
does not let an argument run anything, but it can mislead a reader who is not expecting it.

Applies to the REPL, to one-shot runs (`meka --oneshot`), and to replayed history (`/history`,
`resume_show_recent`). ACP sends structured tool-call fields to the editor and the HTTP API's SSE
events already carry the raw input, so neither is affected.

```toml
[display]
tool_params = "full"
```

### `display.show_session_id_on_create`

Whether to display the session ID when a new session is created.

Default: `false`

### `display.show_session_id_on_exit`

Whether to display the session ID when meka exits.

Default: `true`

```toml
[display]
show_session_id_on_create = true
show_session_id_on_exit = false
```

### `display.show_path_in_prompt`

Whether to show the current working directory in the interactive prompt.

Default: `true`

### `display.show_context_in_prompt`

Whether to show a live context-window gauge in the interactive prompt, e.g. `128.4k/1.0M 13%` (tokens in context / model window / percent used). The figure comes from the most recent turn's reported usage (and an estimate right after `/compact` or on resume), the same value `/status` shows on its `Context:` line. Hidden until the first turn produces a measurement.

Default: `false`

### `display.newline_before_prompt`

Whether to add a blank line before the prompt, after whatever the previous line produced.

Default: `true`

### `display.newline_after_prompt`

Whether to add a blank line after the line you typed, before its output.

Default: `true`

Both apply to **anything printed between two prompts**, not only agent responses: a slash command's
output (`/tasks`, `/memory`, `/help`, …) is bracketed the same way a turn is. A command that answers
by running a turn, such as `/skill <name>`, is spaced once by the turn rather than twice.

The blank lines bracket output, so a command that prints nothing gets neither. In practice every
slash command says something, even if only that a list is empty, so there is always something to
bracket. Two exceptions: a successful `/cd` prints nothing at all, because the prompt itself is the
confirmation, and gets no blank lines either; `!command` is always bracketed, because meka hands the
terminal to the child process and never learns whether it wrote anything, so a silent `!touch file`
still gets its blank lines.

### `display.show_token_usage`

When `true`, meka prints a one-line per-turn token-usage summary to stderr after each turn:

```
[in 12.3k / cache hit 96% / out 1.2k]
```

The `in` column is the total of all three Anthropic input tiers (live, cache-write, cache-read); `cache hit %` is `cache_read / total_in`. Useful for monitoring caching effectiveness during long sessions. The `/status` slash command surfaces cumulative session stats in the same vein.

Default: `false`

### `display.resume_show_recent`

When set to a positive integer `N`, resuming a session reprints the **last `N` turns** (each turn = the user's prompt plus everything the agent did in response, styled to match the live REPL) instead of just the last assistant message.

Useful when you regularly resume long-running sessions and want more context than the single-message default. Inside a session, the `/history` slash command provides the same rendering on demand (`/history` dumps everything; `/history N` shows the last N turns).

Default: unset (resume reprints only the last assistant message, today's behaviour).

```toml
[display]
resume_show_recent = 3
```

### `display.input_style`

Visual style applied to a REPL prompt once it is submitted. Makes past prompts easy to spot when scrolling back through a long session. A line still being edited keeps the terminal's own colours; the style arrives on reedline's final paint, which is the one that lands in scrollback.

The leading `/command` token is a separate signal and is coloured as you type, green when meka recognises the command and red when it does not. This setting does not affect it.

Accepted values:
- `default` (or unset): bold white-ish foreground on a slate-blue background, rendered in truecolor RGB so it looks the same across terminal themes.
- `none`: disable styling entirely.
- `reverse`: reverse video (swaps the terminal's current foreground and background).
- `bold`, `dim`, `italic`, `underline`: single attribute, no colour change.
- A colour name (`black`, `red`, `green`, `yellow`, `blue`, `magenta` / `purple`, `cyan`, `white`): set only the foreground, mapped to the terminal's palette.

Unknown values warn at startup and fall back to `default`.

Default: the banner preset described above.

```toml
[display]
show_path_in_prompt = false
newline_before_prompt = false
newline_after_prompt = false
input_style = "none"    # or "cyan", "bold", "dim", etc.
```

## `[web]`

Settings for the HTTP client shared by `fetch_url` and `search_web`. All keys are optional; unset fields use the defaults shown below.

| Key | Type | Default | Purpose |
|---|---|---|---|
| `user_agent` | string | Real Chrome UA | Some search engines block non-browser UAs. Override if you need a specific identifier. |
| `request_timeout_seconds` | int | `30` | Total request budget (connect + TLS + read). `0` falls back to the default. |
| `connect_timeout_seconds` | int | unset | Separate cap on TCP + TLS handshake. Fail fast on unreachable hosts without shortening the whole request budget. |
| `read_timeout_seconds` | int | unset | Per-chunk idle timeout. Catches bodies that stall mid-stream. |
| `max_redirects` | int | `10` | Cap on 3xx hops. `0` disables redirects entirely. |
| `proxy` | string | unset (honours `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` env) | Proxy URL. Schemes: `http://`, `https://`, `socks5://`, `socks5h://`, `socks4://`. The literal string `"none"` explicitly disables env-var auto-detection. |
| `ca_cert_file` | path | unset | Extra PEM bundle to trust on top of the system store. Useful for corporate MITM proxies or self-signed internal services. Accepts single-cert and multi-cert files. |
| `https_only` | bool | `false` | Refuse plain `http://` URLs. |
| `min_tls_version` | string | unset (reqwest default) | Minimum TLS version. Accepts `"1.0"`, `"1.1"`, `"1.2"`, `"1.3"`. Unknown values log a warning and fall through. Note: the bundled rustls backend supports only TLS 1.2 and 1.3; `"1.0"` / `"1.1"` will surface a build error. |
| `danger_accept_invalid_certs` | bool | `false` | **DANGEROUS.** Disable TLS certificate validation entirely. Emits a `warn!` on every startup when enabled. Only use against trusted local dev servers. |
| `danger_accept_invalid_hostnames` | bool | `false` | **DANGEROUS.** Accept certificates whose hostname doesn't match. Emits a `warn!` on every startup when enabled. Only use against trusted local dev servers. |

### Example: corporate proxy with a private CA

```toml
[web]
proxy = "http://corp-proxy.internal:3128"
ca_cert_file = "/etc/ssl/corp-root-ca.pem"
min_tls_version = "1.2"
request_timeout_seconds = 60
```

### Example: local testing against self-signed certs

```toml
[web]
# Route everything through a local SOCKS proxy you control.
proxy = "socks5h://127.0.0.1:1080"
# Accept self-signed certs on dev.local, KEEP THIS OFF IN PROD.
danger_accept_invalid_certs = true
```

### Example: fail-fast timeouts

```toml
[web]
request_timeout_seconds = 5
connect_timeout_seconds = 2
max_redirects = 0
```

## `[shell]`

Settings for shell command execution.

### `shell.sandbox`

Whether to enable read-only filesystem sandboxing for shell commands in read mode. When enabled (default), shell commands can be executed at `read` and `workspace` but with the filesystem write-protected outside the workspace roots. When disabled, shell commands require `unrestricted`.

Default: `true`

```toml
[shell]
sandbox = false  # disable sandboxed shell in read mode
```

The sandbox uses one of two backends on Linux (see [`shell.sandbox_backend`](#shellsandbox_backend)), `sandbox-exec` on macOS, and a duplicated Low-integrity primary token on Windows. On platforms where no backend is usable, shell commands always require `unrestricted` regardless of this setting.

### `shell.sandbox_backend`

Linux-only choice between `"landlock"` and `"bubblewrap"`:

- **Bubblewrap** (`"bubblewrap"`) wraps the command in `bwrap` with read-only bind of `/`, tmpfs masks over `/run` / `/tmp` / `/var/tmp` / `$XDG_RUNTIME_DIR`, and `--unshare-user --unshare-pid --unshare-uts --unshare-ipc`. The tmpfs masks hide the dbus session bus and the systemd-user socket, so state-changing IPC calls like `systemctl --user start` and `dbus-send` fail. Network is intentionally not unshared so `curl http://x | pdftotext` still works. Requires the `bubblewrap` package and a kernel with user-namespace creation enabled.
- **Landlock** (`"landlock"`) uses the Landlock LSM to block filesystem writes, and requires **ABI v3 (kernel 6.2+)**: below that `truncate(2)` is unmediated, so a read-mode command could still empty a file, and meka reports the backend unusable instead. On kernel 7.1+ (ABI v9) it also blocks `connect()` to Unix sockets on disk, closing the dbus / systemd-user route out of the sandbox at the cost of socket-based clients like `docker` and `psql`. Between v3 and v9 that right does not exist, so a sandboxed shell can still invoke state-mutating dbus methods; meka warns at startup naming what the running ABI lacks. Kept as the lighter-weight fallback for hosts without Bubblewrap.

When omitted, meka probes Bubblewrap once at startup. If Bubblewrap is available it auto-picks it; otherwise it auto-picks Landlock and emits a one-shot warning nudging you to install `bubblewrap` for stronger protection. Set the field explicitly to either value (including `"landlock"`) to suppress that warning. `meka provider add` does not write this field; leave it unset to keep auto-detection.

If the configured backend can't be used at runtime (bwrap not installed, user namespaces denied, etc.), `execute_command` in read mode hard-errors with a message naming the configured backend and the specific failure reason. Read mode is not blocked for other tools; only `execute_command` requires a usable sandbox.

Overridable for one run with `meka --sandbox-backend landlock|bubblewrap`, and for a whole
environment with `MEKA_SANDBOX_BACKEND`. Precedence is flag, then environment, then this field.

Default: unset (auto-detect). Ignored on macOS and Windows.

```toml
[shell]
sandbox = true
sandbox_backend = "bubblewrap"  # or "landlock"
```

## `[permissions]`

Controls which permission modes are reachable at runtime and which mode the session starts in. See the [Permissions](../usage/permissions.md) page for what each mode does.

| Field | Required | Description |
|-------|----------|-------------|
| `default` | No | Mode the session starts in. One of `"none"`, `"read"`, `"workspace"`, `"ask"`, `"unrestricted"`. Default `"read"`. Overridden by `--permission` and `MEKA_PERMISSION`. |
| `enabled` | No | List of modes that can be reached at runtime via `/permission` and Shift+Tab. Default `["none", "read", "workspace", "unrestricted"]`; `"ask"` is opt-in. Disabled modes are skipped during Shift+Tab cycling and rejected by `/permission` with an error. |

If `default` is not in `enabled`, meka logs a warning and falls back to `read` if it's enabled, otherwise the lowest-discriminant enabled mode (in `none → read → workspace → ask → unrestricted` order). Same behavior if `--permission` or `MEKA_PERMISSION` selects a disabled mode: meka warns and starts in the configured default rather than refusing to launch.

```toml
[permissions]
default = "read"
enabled = ["none", "read", "workspace", "ask", "unrestricted"]  # opt back into ask
```

## `[session]`

Settings for session history retention and context window management.

### `session.context_messages`

Maximum number of messages to send to the LLM API per request. Older messages are truncated from the beginning while preserving tool call chain integrity. The full history remains stored in SQLite; only the API payload is limited.

The cap is applied to every request in a turn, not just the first, so a long tool loop cannot grow the payload past it mid-turn. It is a maximum rather than a target: the cut lands on the first message that is safe to start from, which means dropping a whole `tool_use` → `tool_result` pair rather than splitting one, and a request can end up under the limit as a result. A turn whose entire tail is one unbroken tool chain is the exception; there the payload runs over rather than be rejected by the provider.

Default: `200`. `0` is rejected at startup.

```toml
[session]
context_messages = 100
```

### `session.retention_days`

Delete sessions older than this many days, at agent startup. Uses `updated_at`, so an actively-resumed session is preserved even if created long ago. Deletions are reported at `warn` level.

Two kinds of session are spared whatever their timestamp says, and the sweep reports how many it left behind. A session another meka process has open is skipped — only turns bump `updated_at`, and resuming does not, so a REPL sitting at its prompt past the window looks expired while somebody is in front of it. And a session with a scheduled job still ahead of it is never expired, nor is any parent of one: a gated watcher that evaluates every tick and rarely fires looks untouched for exactly as long as it is working, and deleting it would take the schedule with it.

**Default: unset, meaning nothing is deleted.** Conversation history isn't reproducible, so meka keeps it until told otherwise. Use `meka session delete --older-than-days <DAYS>` to prune manually instead.

```toml
[session]
retention_days = 30
```

### `session.auto_compact`

Automatically compact the conversation when input tokens exceed 80% of the context window. Compaction summarizes older messages and preserves recent ones, the todo list, and scratchpad entries.

Default: `true`

```toml
[session]
auto_compact = false
```

### `session.compact_checkpoint`

Run a *checkpoint turn* before each compaction, in which the agent saves anything that must outlive the window and writes the replacement summary itself. See [Compacting a Session](../usage/sessions.md#compacting-a-session).

Costs one extra model call per compaction. Turning it off falls back to a standalone summarizer that has no tools and none of the agent's identity, so it cannot save to memory and cannot apply any judgment about what this particular agent is for.

Note that this applies to automatic compactions too, so an unattended checkpoint can write memory with nobody watching.

Default: `true`

```toml
[session]
compact_checkpoint = false
```

### `session.context_window`

Override the model's context window size (in tokens). Used for auto-compact threshold calculation. A per-profile `[providers.<name>].context_window` takes precedence over this.

When neither is set, meka assumes **1000000**. It does not infer the window from the model name, query the provider's models API, or cache anything: the window is a local budgeting number that is never sent on the wire, so a wrong value can't fail a request, and the user is the one who knows the truth.

1M suits the current flagship models and overshoots the smaller and older ones. Overshooting is survivable rather than free - planned compaction never fires, so those sessions compact only after the provider rejects an over-long request, paying a wasted round trip each time. Set the real window on any profile whose model is smaller.

```toml
[session]
context_window = 200000
```

### `session.subagent_max_depth`

Maximum recursion depth for sub-agents spawned via [`agent_spawn`](../tools/overview.md#agent_spawn). The root agent spawns at depth 1, its sub-agents at depth 2, and so on; each level below this limit is granted its own `agent_spawn`. `1` reproduces the historical behavior where sub-agents cannot spawn further sub-agents; `0` disables `agent_spawn` entirely. An agent can tune a subtree with the tool's `max_depth` parameter, but a built-in absolute cap always bounds real nesting so recursion can't run away.

Default: `3`

```toml
[session]
subagent_max_depth = 3
```

## `[thinking]`

Presentation and budget settings for extended thinking (`anthropic-messages` and `claude-subscription` providers). Whether thinking is on, and which wire encoding it uses, is the per-profile [`thinking`](#thinking) key - not a setting here.

While the model is thinking, the REPL draws a live `Thinking...` line so a long pause reads as work rather than as a hang. On `claude-subscription` it carries the server's own running estimate (`Thinking... (150 tokens)`), redrawn in place as the count climbs; `anthropic-messages` does not report one, so the line stays bare. The count is coarse -- a progress signal, not an accounting figure.

When the block ends the line stays on screen as a record that the phase happened; if the model returned readable reasoning, that text replaces the line instead. Nothing is drawn when output is piped or redirected, since there is no terminal to redraw on.

### `thinking.budget_tokens`

Maximum number of tokens the model can use for thinking. Read only under [`thinking = "budgeted"`](#thinking); the adaptive encoding lets the model set its own budget and sends no cap.

Default: `16000`

### `thinking.show_content`

Whether to show the whole text of a thinking block. When `false`, a block carrying readable reasoning is previewed as a single dimmed line, flattened across line breaks and cut to fit [`display.max_width`](#displaymax_width), and the history replayed on resume (`resume_show_recent`) omits it entirely. When `true`, the full block is printed under a dimmed header. Either way the block is still sent on subsequent turns, for reasoning continuity.

Default: `false`

```toml
[thinking]
budget_tokens = 20000
show_content = true
```

## Instructions

Standing instructions are **not** a config key. They live at a conventional path beside `config.toml`, because prose long enough to be worth writing is miserable to maintain inside a TOML string:

```
~/.config/meka/
├── config.toml
├── instructions.md      # or instructions/*.md
├── memory/
└── skills/
```

See [Instructions](../usage/instructions.md) for the full picture. In short: write `instructions.md`, or split a large set across `instructions/*.md`, and meka reads it at startup into the `## User Instructions` section of the system prompt. To pass the text as a string instead (containers, CI), use `MEKA_INSTRUCTIONS`, `MEKA_INSTRUCTIONS_FILE`, or `--instructions`.

## `[mcp]`

Settings for MCP (Model Context Protocol) tool servers. MCP allows meka to discover and use tools provided by external servers.

### `[[mcp.servers]]`

An array of MCP server configurations. Each entry defines a server to connect to at startup.

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique name for this server. Used as namespace prefix for tools (`name__tool`). Must match `[A-Za-z0-9_-]+`, must not contain `__`, and must not be `meka`, `ide`, or start with `mcp_`. |
| `transport` | Yes | Transport type: `"stdio"` (spawn subprocess) or `"http"` (streamable HTTP). |
| `command` | Stdio only | Path or name of the executable to spawn. On Windows, `npx` / `.cmd` / `.bat` / `.ps1` are auto-wrapped in `cmd /c`. |
| `args` | No | Arguments to pass to the command. |
| `env` | No | Environment variables to set for the spawned process (stdio only). The child does **not** inherit meka's environment; see below. |
| `url` | HTTP only | URL of the MCP server endpoint. |
| `auth` | No | OAuth authentication configuration (see below). Mutually exclusive with a stored bearer token. |
| `headers` | No | Custom HTTP headers to include with every request (HTTP only). |
| `headers_helper` | No | Path to an executable whose stdout (`Name: Value\n` lines) is merged over `headers` at connect-time (HTTP only). Executed with `MEKA_MCP_SERVER_NAME` / `MEKA_MCP_SERVER_URL` in env; 15 s timeout. |
| `permission` | No | Server-wide permission override. Applies to every tool on this server, beating the `readOnlyHint` the server advertises and the `[mcp].default_permission` global fallback. See *Permission resolution* below. |
| `allowed_tools` | No | Optional allow-list of raw tool names (the form the server advertises, not the `server__tool` namespaced form). When set and non-empty, only these tools are registered; all others from this server are ignored. |
| `disabled_tools` | No | Optional block-list of raw tool names. Applied **after** `allowed_tools`; tools listed here are never registered. Both lists can coexist; the net set is `allowed_tools \ disabled_tools`. |
| `eager_load_tools` | No | Raw tool names that should ship **eager-loaded** instead of deferred. Listed tools skip the `load_tool` round-trip and sit in the cacheable tools-array prefix from turn 1. Use this for tools the agent invokes constantly (search, fetch, …); leave others deferred so the tools array stays lean. |
| `tool_permissions` | No | Per-tool permission overrides keyed by raw tool name. Beats the server-level `permission` and the server's `readOnlyHint` when resolving a tool's required permission. |
| `trust_read_only_hint` | No | Whether this server's `readOnlyHint: true` may classify a tool as `read`. Defaults to `true`. Set `false` for a server you have not audited: its hints become advisory for display only, so its tools fall through to the strict `unrestricted` fallback, skipping `[mcp].default_permission` (a global convenience must not re-grant what a per-server audit decision refused). A `readOnlyHint: false` is still honoured either way, since it only raises the requirement. See *Permission resolution* below. |
| `disabled` | No | When `true`, the server is skipped entirely at startup: no process is spawned, no HTTP connect is attempted. Flip it back with `meka mcp enable <name>` or by editing the config. Defaults to `false`. |
| `required` | No | When `true`, a turn is rejected while this enabled server is not `Connected` (a `disabled` server is never started, so it never gates). When `false`, the session runs without it and its tools are simply absent. Defaults to `[mcp].strict` (itself `false`), so servers are optional unless they opt in. |

### `[mcp]` top-level table

| Field | Purpose |
|-------|---------|
| `default_permission` | Fallback permission for MCP tools whose server didn't advertise `readOnlyHint` and doesn't have a `permission` override. Accepts `"none"`, `"read"`, `"workspace"`, `"ask"`, or `"unrestricted"`. If unset the hardcoded fallback is `"unrestricted"` (strict). It stays there deliberately: an MCP server runs unsandboxed, so `workspace` cannot confine it. |
| `strict` | Default for every server's `required` flag. When `true`, all enabled servers gate the turn; when `false` (the default) only servers with `required = true` do. An unavailable optional server doesn't stop the turn; its failure is logged once when it happens, and its live state is shown by `/mcp list` in the REPL or probed with `meka mcp reconnect <name>`. |
| `grace_seconds` | Per-turn cap on how long to wait for still-`Pending` servers to connect before deciding. Default `3`. Set to `0` to skip waiting (useful for scripts that want to fail fast). |
| `connect_timeout_seconds` | Per-server timeout for connect + `initialize` + `list_tools`. A hung stdio spawn or slow HTTPS handshake can't stall the whole fleet past this bound. Default `30`. |

### Startup concurrency

MCP servers connect in parallel at startup, partitioned by transport so a fleet of stdio servers (process-spawn bound) doesn't fight a fleet of HTTP servers (network bound):

- stdio: `MEKA_MCP_STDIO_CONCURRENCY` (default `3`)
- http: `MEKA_MCP_HTTP_CONCURRENCY` (default `20`)

These env vars are tuning knobs: rarely needed, but useful if you're running ~30 stdio servers on a constrained box (lower it) or ~50 HTTP servers (raise it).

### Permission resolution

Every MCP tool's required permission is resolved through a five-step chain; the first match wins:

1. **`server.tool_permissions[<raw-tool>]`**: explicit per-tool override.
2. **`server.permission`**: explicit server-level override. Applies to every tool on that server regardless of what the server advertises.
3. **`tool.annotations.readOnlyHint`** from the server: `true` → `Read`, `false` → `Unrestricted`. The `true` half is skipped when the server sets `trust_read_only_hint = false`, and a hint skipped that way also bypasses step 4, landing on step 5.
4. **`[mcp].default_permission`**: global fallback. Not consulted for a hint that step 3 refused.
5. **Hardcoded `Unrestricted`**: strict ultimate fallback.

User-supplied config (1, 2, 4) always beats the server's self-classification; if a server lies about a tool, you can override. But when no user config says anything, the server's hint is trusted for that specific tool so `readOnlyHint = false` destructive tools don't silently become Read-accessible just because the user opted into a lenient global default.

**Hint spoofing**: `readOnlyHint` is asserted by the server and not verified by meka, and MCP tools run in the server's own process with **no sandbox**. A server that claims `readOnlyHint = true` for a tool that in fact writes therefore gets to write your tree while meka sits at `read`: MCP tools are outside the read-mode filesystem boundary that covers meka's built-ins (see [Permissions](../usage/permissions.md#mcp-tools-are-the-exception)).

Three defences, in increasing order of bluntness:

- `tool_permissions` on the specific tools you want pinned (step 1 wins).
- `trust_read_only_hint = false` on the server, which makes its hints advisory for display only. A refused hint drops straight to the strict `unrestricted` fallback, deliberately skipping `[mcp].default_permission`: that key is a global default, and letting it answer would mean `default_permission = "read"` silently re-granting exactly what the per-server flag refused. None of that server's hinted tools is reachable at `read` without an explicit override.
- `server.permission = "unrestricted"` on the whole server (step 2 wins), or `disabled_tools` to remove the tool entirely.

The hint is trusted by default because most servers annotate honestly and requiring per-tool config for every server would make read mode impractical. `trust_read_only_hint` is the switch for a server you have not audited.

**Stale config**: entries in `allowed_tools` / `disabled_tools` / `eager_load_tools` / `tool_permissions` that don't match any advertised tool get a `warn!` line at connect time. The server still connects; you just see a heads-up so you can clean up after the server renames a tool. A name that appears in both `eager_load_tools` and `disabled_tools` also warns: the disabled filter wins, so eager-loading the disabled tool is a no-op.

**Visibility across levels**: the resolved permission doesn't hide a tool from the agent. Every registered tool is listed in the per-turn context with its required level noted inline, and a `[Permission context]` section names the current level and states in one line what it allows (it does not enumerate tools; the per-tool levels are in the catalogue above it). The agent can still reason about an inaccessible tool and suggest `/permission <level>` to enable it; the permission gate is enforced at dispatch time. Keeping the tool catalogue visible across levels is also what lets the Claude prompt cache survive mid-session permission toggles.

#### The stdio server's environment

A stdio server is a child process that talks to the network, and it does **not** inherit meka's
environment. It receives the same curated base a read-mode shell gets (`PATH` so it can resolve its
own binaries, `HOME`, locale, `TMPDIR`), plus whatever the server's own `env` table sets.

Configuring a server is a decision to run its code, not a decision to hand it every credential on
the machine: without this, `ANTHROPIC_API_KEY`, `AWS_*` and `GITHUB_TOKEN` all rode along into every
server you had ever added.

The base also carries the machine's network configuration (`HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`,
`SSL_CERT_FILE`, `SSL_CERT_DIR`, `NODE_EXTRA_CA_CERTS` and the usual siblings), because a server
that cannot see them connects to nothing behind a corporate proxy and fails every call with an
error naming none of the cause. Those say where to go and whom to trust; they grant nothing.

Three families are deliberately left out and have to be requested per server: `SSH_AUTH_SOCK`, which
is a live credential agent; `NODE_OPTIONS`, which takes `--require` and therefore arbitrary code;
and the import paths `PYTHONPATH` / `NODE_PATH` / `VIRTUAL_ENV`, which change what a program loads.
A server that genuinely needs one takes it explicitly:

```toml
[mcp.servers.tooling.env]
PYTHONPATH = "${PYTHONPATH}"
```

A server that genuinely needs a secret asks for it by name, and `${VAR}` still reads meka's
environment at connect time:

```toml
[[mcp.servers]]
name = "github"
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
env = { GITHUB_PERSONAL_ACCESS_TOKEN = "${GITHUB_TOKEN}" }
```

#### Examples

**Exa**: reliable web search when the built-in DuckDuckGo scraper gets CAPTCHA'd. The free tier works without an API key; paste a key into the `headers` table for the paid tier:
```bash
# Free tier, no key required
meka mcp add exa https://mcp.exa.ai/mcp
```
```bash
# Paid tier, expands from EXA_API_KEY at connect time
meka mcp add exa https://mcp.exa.ai/mcp --header "x-api-key=${EXA_API_KEY}"
```

Well-annotated server: no config needed. Every tool is classified by its own `readOnlyHint` (read tools Read, write tools Write):
```toml
[[mcp.servers]]
name = "notion"
transport = "http"
url = "https://mcp.notion.com/mcp"
```

User-declared trust on an unannotated server (all tools accessible in Read):
```toml
[[mcp.servers]]
name       = "internal"
transport  = "http"
url        = "https://mcp.internal/…"
permission = "read"
```

Overriding a mis-annotated or distrusted tool (one specific tool requires `unrestricted`):
```toml
[[mcp.servers]]
name      = "notion"
transport = "http"
url       = "https://mcp.notion.com/mcp"

[mcp.servers.tool_permissions]
"notion-do-something-scary" = "unrestricted"
```

Subset of a server's tools (only `query` registers, all others are ignored):
```toml
[[mcp.servers]]
name          = "pg"
transport     = "stdio"
command       = "npx"
args          = ["-y", "@modelcontextprotocol/server-postgres"]
allowed_tools = ["query"]
```

Block-list with a narrow exception (all fs tools are Read-accessible except the two destructive ones, which are never registered):
```toml
[[mcp.servers]]
name           = "filesystem"
transport      = "stdio"
command        = "npx"
args           = ["-y", "@modelcontextprotocol/server-filesystem"]
permission     = "read"
disabled_tools = ["delete_file", "move_file"]
```

MCP tools are registered with namespaced names in the format `servername__toolname` to prevent collisions with built-in tools or between servers.

Tool and resource descriptions returned from MCP servers are truncated at 2048 characters to keep the rendered catalogue bounded.

### Environment variable substitution

Every string field listed above (command, args, env values, url, headers values) supports `${VAR}` and `${VAR:-default}` expansion from the process environment. Missing variables with no default leave the literal `${VAR}` in place and log a warning at startup. Use this to avoid committing secrets:

```toml
[[mcp.servers]]
name = "github"
transport = "http"
url = "https://mcp.github.com"
headers = { X-Api-Key = "${GITHUB_MCP_TOKEN}" }
```

`env`, `args` and `headers` may *contain* a secret, but they are not one: `env` sets a subprocess's whole environment, `args` carries connection strings, and `headers` carries `X-Tenant-Id` as readily as `X-Api-Key`. meka cannot tell which is which, so they stay in `config.toml` and `${VAR}` is how you keep a value out of it.

A bearer token and an OAuth client secret are unambiguously secrets, so they are not config at all. They live in meka's database and are set with `meka mcp add --auth-token-stdin` / `--client-secret-stdin`, or afterwards with `meka mcp login`. See [Credentials](#credentials).

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MEKA_MCP_TOOL_TIMEOUT` | `600000` ms (600 s) | Per-call timeout for MCP tools. Triggers `notifications/cancelled` on expiry. |

### `meka mcp` CLI

Manage configured servers without editing `config.toml` by hand:

| Command | Action |
|---|---|
| `meka mcp list` | Print all configured servers, plus any stored OAuth credential that no server claims (see [Leftover credentials](#leftover-credentials)). |
| `meka mcp get <name>` | Print full details for one server. |
| `meka mcp add <name> <url-or-command> [args...] [flags]` | Persist a server. Transport is auto-detected: a URL starting with `http[s]://` means HTTP, anything else means stdio. Preserves existing formatting/comments via `toml_edit`. |
| `meka mcp remove <name>` | Best-effort revoke stored OAuth tokens (RFC 7009) at the provider, then delete the server entry, clear stored credentials, and drop any resource-update ledger entries. A name with stored credentials but no config entry is cleaned rather than refused. |
| `meka mcp disable <name>` | Set `disabled = true` on the server entry. The next `meka` start skips it entirely. |
| `meka mcp enable <name>` | Clear the `disabled` flag, so the server connects on the next start. |
| `meka mcp reconnect <name>` | Smoke-test a connect; prints `ok` or the error. |
| `meka mcp tools <name>` | Connect and list every advertised tool with its resolved permission, the chain step that decided it, and whether the current config allows it. Useful for populating `--allow-tool`, `--disable-tool`, or `--tool-permission` overrides without leaving the CLI. |
| `meka mcp login <name>` | Drive interactive OAuth. If the server has no `[auth]` block and uses HTTP, assumes `type = "oauth"` and persists the block on success. With `--auth-token-stdin` or `--client-secret-stdin`, stores that secret and exits instead, which is also how you rotate one. |
| `meka mcp logout <name>` | Call the provider's `revocation_endpoint` (RFC 7009) best-effort, then clear every stored credential for the server. |

#### Credentials

An MCP server's bearer token, OAuth client secret and OAuth token bundle are stored in meka's database (`mcp_credentials`, keyed by server name and kind), never in `config.toml`. This is the same rule providers follow, and for the same reason: `config.toml` is a plaintext file people commit, sync and share.

Each is read from stdin so it never reaches `ps` output or your shell history. One command reads one secret, so `--auth-token-stdin` and `--client-secret-stdin` cannot be combined:

```console
$ pass show notion-token | meka mcp add notion https://mcp.notion.com/mcp --auth-token-stdin
$ pass show acme-secret | meka mcp login acme --client-secret-stdin
```

A confidential OAuth client holds two at once: the long-lived client secret it authenticates with, and the refreshable bundle it obtained. Store the secret first, then run `meka mcp login <name>` to complete the flow. Refreshing the bundle leaves the client secret alone.

`meka mcp get <name>` lists which kinds a server has, without printing any of them. `meka mcp list` names servers that have a stored credential but no `[[mcp.servers]]` entry, which is what a hand-edited config strands.

#### `meka mcp add` flags

| Flag | Purpose |
|------|---------|
| `--transport <stdio\|http>` | Override the auto-detected transport. |
| `--env KEY=VALUE` | Environment variable for stdio (repeatable). |
| `--header KEY=VALUE` | HTTP header (repeatable). |
| `--auth <oauth\|client-credentials\|client-credentials-jwt>` | Configure the `[auth]` block. |
| `--auth-token-stdin` | Read a static bearer token from stdin and store it. Mutually exclusive with `--auth`. |
| `--client-secret-stdin` | Read an OAuth client secret from stdin and store it. Required by `--auth client-credentials`. |
| `--client-id` | OAuth / client-credentials client identifier. Not a secret, so it goes in `config.toml`. |
| `--signing-key <PATH>`, `--signing-algorithm <ALG>` | JWT signing material (`client-credentials-jwt` only). |
| `--scope <SCOPE>` | OAuth scope (repeatable). |
| `--redirect-port <PORT>` | Fixed OAuth redirect port (default: ephemeral). |
| `--permission <none\|read\|workspace\|ask\|unrestricted>` | Per-server permission cap (applies to all tools on the server). |
| `--allow-tool <NAME>` | Raw tool name to allow (repeatable). When set, only listed tools register. |
| `--disable-tool <NAME>` | Raw tool name to block (repeatable). Applied after `--allow-tool`. |
| `--eager-load-tool <NAME>` | Raw tool name to eager-load (repeatable). Listed tools skip the `load_tool` round-trip and ship in the cacheable tools-array prefix from turn 1. |
| `--tool-permission <NAME=LEVEL>` | Per-tool permission override (repeatable). `LEVEL` is `none`/`read`/`workspace`/`ask`/`unrestricted`. |
| `--required` | Persist `required = true`, so a turn is rejected while this server isn't connected. Omitted, the server inherits `[mcp].strict` and is optional by default. |
| `--disabled` | Persist `disabled = true`, so the server is skipped entirely at startup. Re-enable with `meka mcp enable <name>`. |

#### Example: Notion

These signposts are `info` logs, so they need `-v`; at the default `warn` level the command
succeeds silently and the exit code carries the result. Timestamps and targets are elided here.

```console
$ meka -v mcp add notion https://mcp.notion.com/mcp
added 'notion' to ~/.config/meka/config.toml
probe: 'notion' requires OAuth
running OAuth authorization for 'notion' (use --no-login to skip)
no [auth] block for 'notion'; assuming OAuth authorization_code
…
authorized 'notion'
```

`meka mcp add` on an HTTP endpoint:

1. **Probe**: issues an unauthenticated `GET` (3 s timeout, redirects off) and classifies the response per the MCP authorization spec + RFC 6750 + RFC 9728:

   - `2xx` → server is open, no login needed.
   - `401` / `403` with `WWW-Authenticate: Bearer …` → OAuth required. The `resource_metadata="…"` attribute (RFC 9728) is captured at DEBUG.
   - Any other status → couldn't infer, prints the status code.
   - Network failure → prints the error.

2. **Auto-login**: if the probe says OAuth is required (or `--auth oauth` was explicitly set), the OAuth authorization_code flow runs immediately as though the user had chained `meka mcp login <name>` themselves. The synthesised `[auth] = oauth` block is written back to `config.toml` on success.

3. **Rollback on failure**: if the OAuth flow errors out, the entry we just wrote is purged from `config.toml` (alongside any partial credentials), leaving the user's config clean. The command exits non-zero.

4. **`--no-login`**: skips step 2. The entry is still persisted and the probe's hint is still printed; run `meka mcp login <name>` when ready. Useful for scripted setup or when you expect to edit `[auth]` by hand.

The probe and the auto-login only run for HTTP servers, and only when the user didn't provide `--auth-token-stdin` (static bearer) or `--auth` (other than `oauth`). Stdio servers skip both.

#### Remote hosts / SSH sessions

The OAuth flow redirects the browser to `http://127.0.0.1:<port>/callback`. When meka is running on a different host than the browser (SSH session, container, Codespace, WSL), the browser can't reach back and shows a "connection refused" error page. meka handles this automatically:

- While `meka mcp login <name>` waits for the callback it also watches stdin.
- The browser's address bar still contains the full callback URL (including `code` and `state`) even when the connection fails. Copy it, paste it into the meka prompt, and press Enter.
- Whichever completes first, the TCP callback or the pasted URL, wins.

meka opens the browser silently and prints the URL exactly once, so the flow works the same whether
or not a browser is reachable. The `authorized` line is an `info` log, shown here with `-v`.

```console
$ meka -v mcp login notion
open this URL in your browser to authorize:

https://mcp.notion.com/authorize?response_type=code&…

waiting up to 120s for the callback, or paste the callback URL here and press Enter:
http://127.0.0.1:46437/callback?code=…&state=…     ← paste here
authorized 'notion'
```

#### REPL parity

Inside the REPL:
- `/mcp list`: list configured servers.
- `/mcp reconnect <server>`: reconnect smoke-test.
- `/mcp login <server>` / `/mcp logout <server>`: run the auth flow or revoke.
- `/mcp <server>:<prompt> [args...]`: render a server-defined prompt as the next user turn.

### Resources and prompts

In addition to tools, meka exposes MCP resources and prompts through several builtin tools (deferred: the agent calls `load_tool` first to fetch the schema, then invokes them):

| Builtin | Purpose |
|---------|---------|
| `mcp_resource_list` | List resources from one or every configured server. |
| `mcp_resource_read` | Read a resource by `server` + `uri`; text inline, binary base64-encoded. |
| `mcp_prompt_list` | List prompts from one or every configured server, including their declared arguments. |
| `mcp_prompt_get` | Render a prompt by `server` + `name` with optional `arguments`; returns `<role>: <text>` lines. |
| `mcp_resource_subscribe` | Subscribe to `resources/updated` notifications for a specific URI. |
| `mcp_resource_unsubscribe` | Cancel a prior subscription. |
| `mcp_resource_updates_list` | Print every resource that has been reported as updated since the session started. |

### Connection lifecycle

- **Reconnection** is automatic for all transports (stdio, plain HTTP, OAuth-authenticated HTTP) when the transport closes mid-session. HTTP transports use exponential backoff (1s, 2s, 4s, 8s, 16s, capped 30s, max 5 attempts); stdio gets one immediate retry. The reconnect runs on a blocking thread to work around an upstream rmcp bug where the auth future is `!Send`.
- **Failed initial connect** is retried in the background with its own backoff (5s doubling to a 5 minute ceiling) until the server comes up, and the server's tools are registered into every live session when it does. A server that is slow to boot, or that starts after meka, therefore recovers on its own rather than staying `failed` for the life of the process. This matters most for a `required` server, where every turn is rejected until it connects.
- **Session-expired recovery**: rmcp transparently re-initialises HTTP sessions on 404 / JSON-RPC `-32001`. meka relies on this; no per-call handling is required.
- **Cancellation**: when the agent cancels a tool call (e.g. Ctrl-C), meka sends `notifications/cancelled` to the server with the in-flight request id so the server can stop work.
- **Timeouts**: tool calls default to 600 s; override with `MEKA_MCP_TOOL_TIMEOUT` in ms.
- **Tool list refresh**: on `tools/list_changed`, meka re-discovers the server's tools and hot-swaps them in the registry; no restart needed.
- **Progress notifications**: MCP tool calls attach a per-request `progressToken`; incoming `notifications/progress` render as a live status line under the tool invocation.
- **Call identity**: `tools/call` carries two extra keys in `_meta` alongside the progress token. `meka/sessionId` is the UUID of the session the call came from, letting a server scope per-session state (a cache, a workspace, an audit trail) to one conversation; a sub-agent reports its own child session id. `meka/toolUseId` is the provider's tool-use id for the call. Both are absent for calls made outside a session, such as connection-time handshakes.
- **Server instructions**: `InitializeResult.instructions` is captured once per connection and delivered in the per-turn context (sanitised + truncated to 2048 chars) under `[MCP server instructions]`. A server that connects late, or reconnects with different instructions, is announced as a change rather than rewriting anything already sent.
- **stdio server logs**: a stdio server's own stderr (many servers log there) is captured, not inherited, so it never corrupts the REPL display. Each line is re-emitted on meka's `tracing` stream at `debug` level tagged with the server name, so it stays silent at default verbosity and surfaces under `-v` / `RUST_LOG`.
- `resources/list_changed`, `prompts/list_changed`, and `resources/updated` notifications are logged at `info`/`debug` level.

### Server-to-client features

| Feature | meka behaviour |
|---------|----------------|
| `elicitation/create` | Routed to the calling session's frontend (REPL / ACP form or URL prompt) with a 60s timeout. Auto-declines when no in-flight tool call's frontend is registered or the user doesn't answer in time. |

### `[mcp.servers.auth]`

OAuth authentication for HTTP MCP servers. Set `type` to choose the authentication method. This is mutually exclusive with a stored bearer token.

The client secret is not a field here. It is a secret, so it lives in the database: set it with `meka mcp add --client-secret-stdin` or `meka mcp login <name> --client-secret-stdin`. See [Credentials](#credentials).

| Field | Required | Description |
|-------|----------|-------------|
| `type` | Yes | Auth method: `"client_credentials"`, `"client_credentials_jwt"`, or `"oauth"` |
| `client_id` | Varies | OAuth client ID (required for client_credentials/jwt, optional for oauth with dynamic registration) |
| `scopes` | No | OAuth scopes to request |
| `resource` | No | Resource parameter ([RFC 8707](https://datatracker.ietf.org/doc/html/rfc8707)), client_credentials only |
| `signing_key_path` | JWT only | Path to PEM private key file |
| `signing_algorithm` | No | JWT signing algorithm: `RS256` (default), `RS384`, `RS512`, `ES256`, `ES384` |
| `redirect_port` | No | Local port for OAuth authorization code callback. When omitted, meka binds to a random ephemeral port (recommended). `oauth` only. |

### Examples

#### Stdio server

```toml
[[mcp.servers]]
name = "postgres"
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]
permission = "unrestricted"
```

#### HTTP server

```toml
[[mcp.servers]]
name = "web-tools"
transport = "http"
url = "http://localhost:8080/mcp"
permission = "read"
```

#### HTTP server with authentication

The bearer token is not in the file. Store it once with `meka mcp add api https://api.example.com/mcp --auth-token-stdin`, or `meka mcp login api --auth-token-stdin` for a server that already exists.

```toml
[[mcp.servers]]
name = "api"
transport = "http"
url = "https://api.example.com/mcp"
permission = "unrestricted"

[mcp.servers.headers]
X-Custom-Header = "value"
```

#### Stdio server with environment variables

```toml
[[mcp.servers]]
name = "github"
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
permission = "read"

[mcp.servers.env]
GITHUB_TOKEN = "ghp_..."
```

#### Multiple servers

```toml
[[mcp.servers]]
name = "filesystem"
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
permission = "read"

[[mcp.servers]]
name = "github"
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
permission = "unrestricted"
```

#### HTTP server with OAuth client credentials

```toml
[[mcp.servers]]
name = "api"
transport = "http"
url = "https://api.example.com/mcp"
permission = "unrestricted"

[mcp.servers.auth]
type = "client_credentials"
client_id = "my-client-id"
scopes = ["read", "write"]
```

`client_credentials` needs a client secret, which is stored rather than written here: pass `--client-secret-stdin` to `meka mcp add`, or `meka mcp login api --client-secret-stdin` afterwards.

#### HTTP server with JWT client credentials

```toml
[[mcp.servers]]
name = "api"
transport = "http"
url = "https://api.example.com/mcp"

[mcp.servers.auth]
type = "client_credentials_jwt"
client_id = "my-client-id"
signing_key_path = "/path/to/private-key.pem"
signing_algorithm = "RS256"
scopes = ["admin"]
```

#### HTTP server with OAuth authorization code flow

On first connection, meka opens a browser for authorization and stores the token for future use.

```toml
[[mcp.servers]]
name = "github-mcp"
transport = "http"
url = "https://mcp.example.com"

[mcp.servers.auth]
type = "oauth"
client_id = "my-app-id"
scopes = ["repo", "user"]
redirect_port = 8400
```

If `client_id` is omitted, meka attempts [dynamic client registration](https://datatracker.ietf.org/doc/html/rfc7591) with the server.

## `[tools]`: built-in tool filters

The three knobs `[[mcp.servers]]` exposes for MCP tools also apply to meka's built-in tools (`read_file`, `write_file`, `execute_command`, `search_web`, etc.) via a top-level `[tools]` table. MCP per-server filtering is separate from this and keeps its own namespaces; this block only affects the built-ins.

| Key | Purpose |
|---|---|
| `allowed_tools` | Optional allow-list of built-in tool names. When set and non-empty, only these built-ins register, with one exception: the seven [MCP meta-tools](#resources-and-prompts) register regardless, because they are how the agent reaches a configured server's resources and prompts at all. Naming one here is inert and warns at startup; use `disabled_tools` to remove one. Use `meka tools list` to see the canonical names. |
| `disabled_tools` | Block-list of built-in tool names. Applied **after** `allowed_tools`; a tool here is never registered even if it also appears in the allow-list. |
| `tool_permissions` | Per-tool required-permission override keyed by built-in name. Beats the hardcoded required level from the tool's impl. Levels: `none`, `read`, `workspace`, `ask`, `unrestricted`. |

Stale entries (a name that doesn't match any built-in) emit a `warn!` at startup. meka still starts; the warning just flags a likely typo or a tool the binary renamed.

Restrict a session to read-only inspection:
```toml
[tools]
allowed_tools = ["read_file", "find_files", "search_contents", "fetch_url"]
```

Force `execute_command` to need `unrestricted` so `ask` mode prompts for every shell call:
```toml
[tools.tool_permissions]
execute_command = "unrestricted"
```

Disable web access entirely in a locked-down environment:
```toml
[tools]
disabled_tools = ["search_web", "fetch_url"]
```

Sub-agents spawned via `agent_spawn` inherit the same filter; a disabled built-in is disabled everywhere. To take something away from sub-agents *only*, use [`[subagents]`](#subagents). Run `meka tools list` to see every built-in's effective required permission, whether a `[tools.tool_permissions]` override is in effect, and whether the current config enables it.

## `[subagents]`

Capabilities a sub-agent may never hold. Where `[tools]` restricts everyone, this block restricts only workers.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `disabled_servers` | list | `[]` | MCP servers a sub-agent cannot see at all |
| `disabled_tools` | list | `[]` | Individual tool names a sub-agent cannot see |

```toml
[subagents]
disabled_servers = ["mekabridge"]
disabled_tools = ["mcp__notion__create_page"]
```

**`disabled_servers` is the one that matters.** Naming a server removes everything it offers from every sub-agent: its tools, its resources, and its prompts. Reach for it when a server exists to talk to *you* or to act on your behalf. The motivating case is a server that can message the user: without this, a worker three levels down can send a message the user has no way to distinguish from the one they are actually talking to.

`disabled_tools` takes names as they appear in the tool list, so built-ins (`write_file`) and namespaced MCP tools (`mcp__notion__create_page`) share one namespace. For a whole server, prefer `disabled_servers`: it covers the resource and prompt surfaces that a tool-name list cannot reach.

An entry matching nothing emits a `warn!` at startup, the same way `[tools]` does. A typo here denies nothing while reading as a restriction, which is worse than writing no config at all.

These are floors. An orchestrator can restrict a particular worker further with `agent_spawn`'s `deny_servers` / `deny_tools` parameters, and each level of nesting inherits everything above it, but nothing can grant back what this block took away. There is deliberately no call-site allow-list for that reason.

### Why memory and instructions are not configured here

Two things a sub-agent might inherit are deliberately absent: the memory store and the [instructions file](../usage/instructions.md). Both are granted per call by [`agent_spawn`](../tools/overview.md#agent_spawn) and default to nothing.

The distinction is what config can actually enforce. A *capability* can be withheld: a tool the registry never registered cannot be reached, however the parent phrases the task. *Context* cannot. An agent holding the instructions has them verbatim in its own system prompt, and one with `memory_read` can read any memory — so either can be copied into a worker's prompt whatever config says. A `[subagents].memory = "none"` key would look like a boundary while stopping only the worker's own browsing, not the content reaching it, and a control that reads as a guarantee but isn't one is worse than none.

The other half of the argument is that the config guardrail existed for a failure mode that no longer applies. It was there because the parent might *forget* — which only matters for things that are on by default. Both of these now default to off, so forgetting produces a clean worker.

## `[skills]`

Controls the skill store. See the [Skills](../usage/skills.md) guide.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Register `skill_read` / `skill_search` and render the skills index |
| `agent_managed` | bool | `false` | Additionally register `skill_write` / `skill_delete` |
| `extra_paths` | array | `[]` | Additional directories to scan, read-only |

```toml
[skills]
enabled = false
```

Setting `enabled = false` keeps every skill tool's schema out of every request and renders no skills section. Files already in `~/.config/meka/skills/` are left untouched.

`agent_managed = true` lets the agent author its own skills. It is off by default because you normally curate that store yourself; it exists for a long-running agent that dispatches sub-agents, where a skill is the only artifact that both survives the session and can be handed to a worker as its task. Sub-agents never receive the authoring tools whatever this is set to. See [Letting the Agent Manage Skills](../usage/skills.md#letting-the-agent-manage-skills).

`extra_paths` adds directories to the scan. They are strictly read-only: meka never creates them and never writes into them, so an entry that does not exist is simply skipped and leaves nothing behind. A leading `~` is expanded.

```toml
[skills]
extra_paths = ["~/.agents/skills"]
```

`~/.agents/skills` is the cross-client convention, so pointing at it makes skills installed by other Agent Skills clients visible here. It is not a default: reading a directory outside meka's own namespace is your call. meka's own store is searched first and wins a name collision. There is no automatic project-level scan, for the same reason meka does not read config or instructions from the working directory; name the path here if you want a project's skills read. See [Reading Skills from Other Directories](../usage/skills.md#reading-skills-from-other-directories).

An entry that repeats an earlier one, or that names meka's own skills directory, is dropped with a warning: it would otherwise be scanned twice and every skill in it reported as shadowed by itself. An empty string is dropped too, since it would expand to your home directory.

## `[memory]`

Controls the agent's durable note store. See the [Memory](../usage/memory.md) guide.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Register the `memory_*` tools and render the memory index |

```toml
[memory]
enabled = false
```

Setting `enabled = false` keeps the four `memory_*` tool schemas out of every request and renders no memory section, which is worth doing for lean sessions that will never use it. Memories already stored are left untouched, and `meka memory` still reaches them.

There is deliberately no environment variable and no CLI flag here: whether an agent keeps memories is a property of the installation, not something to vary per run.

## `[schedule]`

Controls the wakeups the agent schedules for itself. See the [Scheduling](../usage/scheduling.md) guide.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Register the `schedule_*` tools and run the scheduler |
| `poll_interval` | duration | `"10s"` | How often due jobs are checked |
| `missed_grace` | duration | `"24h"` | How late a one-shot job may be and still fire after downtime |
| `gate_timeout` | duration | `"30s"` | Wall-clock budget for a gate probe |
| `max_jobs` | int | `50` | Per-session ceiling, refused at `schedule_create` |
| `max_consecutive_fires` | int | `5` | Per-session ceiling on turns spent in one sweep |
| `claim_lease` | duration | `"1h"` | How long a host's claim on a due occurrence is good for |

```toml
[schedule]
enabled = true
poll_interval = "10s"
missed_grace = "24h"
gate_timeout = "30s"
max_jobs = 50
max_consecutive_fires = 5
claim_lease = "1h"
```

`poll_interval` is the real resolution floor: a job whose interval is shorter than the tick fires once per tick, not once per interval.

`missed_grace` applies only to one-shot jobs. Recurring jobs need no equivalent, because their occurrences are one period apart, so the most recent missed one is always less than a period old; the scheduler coalesces the rest into a single catch-up fire.

`claim_lease` is how long a crashed host's occurrence stays unavailable before another host takes it. A due job is leased rather than consumed, so the row survives until the turn is delivered and a host that dies mid-delivery costs a retry rather than the occurrence. Raise it only if a gate probe plus a turn could plausibly exceed an hour; lowering it below that risks a second host taking an occurrence the first is still running, which the session lock catches for an ordinary job but not for an `isolated` one. A host refuses to start on a value at or under `gate_timeout`, since a lease that cannot outlast the host's own probe is never right; that check does not cover the turn after the probe, which is unbounded, so leave headroom on top of it.

`max_consecutive_fires` interleaves sessions: without it, one session's whole backlog runs to completion before another session's single due job is reached. Jobs past the budget keep their occurrence, run no gate, and are taken by the next sweep most-overdue first. It bounds a batch rather than a rate — sweeps do not overlap and the next starts as soon as the last ends, so a backlog still produces one turn per job, just in interleaved groups. `0` is rejected, since it would hold every job over forever; use `enabled = false` to turn scheduling off.

Setting `enabled = false` keeps the three `schedule_*` tool schemas out of every request and leaves existing jobs on disk without firing.

As with `[skills]` and `[memory]`, there is no environment variable and no CLI flag: whether an agent may schedule its own turns is a property of the installation.

## `[background]`

Controls tool calls the agent starts and does not wait for. See the [Background Tasks](../usage/background.md) guide.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Offer the `background` parameter and register the `task_*` tools |
| `max_tasks` | int | `10` | Concurrent tasks per session, refused at dispatch |

```toml
[background]
enabled = true
max_tasks = 10
```

**Alone among the capability blocks, this one is off by default.** `[schedule]`, `[skills]`, and `[memory]` add capability without changing when a turn ends; this changes the contract of the primary interaction into "you asked, it answered, and something else may interrupt you later". That is right for an unattended assistant and wrong for someone using the REPL as a command line. A scheduled job also takes an explicit act to create, whereas `background` is reachable from any tool call, so an agent will reach for it unprompted.

Setting `enabled = false` keeps the `background` property out of every tool schema and the two `task_*` tools out of every request, rather than advertising a parameter that would only ever be refused.

Outcome delivery shares [`[schedule].poll_interval`](#schedule), so that key sets how long a finished task waits before it is reported, whether or not scheduling itself is enabled.

Config-only, like the blocks above: no environment variable, no CLI flag.

## `[serve]`

Configuration for `meka serve`, the HTTP API server. See the [HTTP API](../usage/http-api.md) usage guide for a full walkthrough.

### `serve.bind`

Address and port the HTTP server listens on.

| Type | Default |
|------|---------|
| `string` | `"127.0.0.1:8080"` |

```toml
[serve]
bind = "0.0.0.0:8080"
```

> **Security:** Binding to `0.0.0.0` exposes the server on all interfaces. In production, keep `127.0.0.1` and front with a TLS-terminating reverse proxy.

### `serve.max_body_bytes`

Maximum request body size in bytes. Requests exceeding this limit are rejected with `413 Payload Too Large`.

| Type | Default |
|------|---------|
| `integer` | `10485760` (10 MiB) |

### `serve.docs`

Whether to serve the Swagger UI at `/v1/docs` and the OpenAPI document at `/v1/openapi.json`.

Off by default. These are the only routes on the surface that take no bearer token *and* describe
the deployment rather than report on it: what they publish is the shape of every endpoint you
expose. That is exactly what you want while building a client against a local `meka serve`, and
exactly what you do not want reachable from anywhere else. Turn it on deliberately.

```toml
[serve]
docs = true
```

| Type | Default |
|------|---------|
| `boolean` | `false` |

### `serve.max_concurrent_turns`

Process-wide cap on in-flight turns across all sessions. When the cap is reached, new turn submissions return `429 Too Many Requests` with a `Retry-After` header. Leave it **unset** for no limit; `0` is rejected at startup, because a cap of zero would 429 every turn rather than mean "unlimited".

| Type | Default |
|------|---------|
| `integer` | unbounded |

### `serve.stream_replay_events`

How many SSE events per turn to retain so a client reconnecting to `GET /v1/sessions/{id}/stream` with `Last-Event-ID` can replay what it missed.

| Type | Default |
|------|---------|
| `integer` | `256` |

Matches the live broadcast channel's capacity: retaining more than the channel can buffer would let a reconnecting client replay events a *connected* consumer would have been dropped for missing. Raising it buys a longer reconnect window at the cost of per-session memory during a turn. `0` switches replay off, so a reconnect receives only what happens from then on and is told its replay is incomplete rather than being handed a silently truncated one.

### `serve.stream_reattach_grace`

How long a streaming turn keeps running after its SSE consumer disconnects, waiting for a reconnect. Accepts duration strings.

| Type | Default |
|------|---------|
| `string` (duration) | `"30s"` |

Zero subscribers means nobody is listening, and a turn with no audience is spending provider tokens for nothing. That is the right instinct and the wrong deadline: a client whose connection just dropped and one that is never coming back are the same observation until the window expires. Set `"0s"` to cancel a turn the moment its stream drops, which spends less on abandoned work and makes re-attach useful only for turns that already finished.

### `serve.idle_timeout`

How long a session can sit idle (no turns submitted) before the GC evicts it from memory. Accepts duration strings like `"24h"`, `"30m"`, `"7d"`. Set to `"0"` to disable idle GC.

| Type | Default |
|------|---------|
| `string` (duration) | `"24h"` |

Eviction drops the in-memory runtime but **preserves the SQLite row**; a later request transparently re-attaches. See `delete_on_idle` to also remove the DB row.

### `serve.gc_scan_interval`

How often the background GC scanner runs. Accepts duration strings.

| Type | Default |
|------|---------|
| `string` (duration) | `"5m"` |

### `serve.delete_on_idle`

When `true`, idle-evicted sessions also have their SQLite row deleted. When `false` (default), only the in-memory state is dropped and the session can be re-attached later.

| Type | Default |
|------|---------|
| `bool` | `false` |

### `serve.shutdown_drain_timeout`

Maximum time to wait for in-flight turns and tasks to finish during graceful shutdown (`SIGTERM` / `SIGINT`). After this timeout, remaining tasks are aborted and the process exits.

| Type | Default |
|------|---------|
| `string` (duration) | `"30s"` |

### `[[serve.tokens]]`

An array of bearer tokens for API authentication. At least one token is required.

| Key | Required | Description |
|-----|----------|-------------|
| `token` | Yes* | The bearer token value. Supports `${ENV_VAR}` substitution. Mutually exclusive with `token_file`. |
| `token_file` | Yes* | Path to a file containing the token (one line, trimmed). Mutually exclusive with `token`. A startup warning is logged if the file is world-readable. |
| `description` | No | Human-readable label for this token (appears in logs). |
| `scopes` | Yes | Array of scope strings. One `:r` and one `:w` per subsystem: `sessions`, `skills`, `memory`, `schedule`, `mcp`. |

\* Exactly one of `token` or `token_file` must be set.

Inline plaintext tokens log a startup warning; use `${ENV_VAR}` or `token_file` for production.

#### Examples

Development token (inline):

```toml
[[serve.tokens]]
token = "sk_dev_test123"
scopes = ["sessions:r", "sessions:w"]
```

Production token (environment variable):

```toml
[[serve.tokens]]
token = "${MEKA_BRIDGE_TOKEN}"
description = "telegram bridge"
scopes = ["sessions:r", "sessions:w"]
```

Production token (file-based):

```toml
[[serve.tokens]]
token_file = "/etc/meka/bridge.token"
description = "telegram bridge"
scopes = ["sessions:r", "sessions:w"]
```

Admin token with every scope:

```toml
[[serve.tokens]]
token = "${MEKA_ADMIN_TOKEN}"
description = "operator"
scopes = [
    "sessions:r", "sessions:w",
    "skills:r", "skills:w",
    "memory:r", "memory:w",
    "schedule:r", "schedule:w",
    "mcp:r", "mcp:w",
]
```

Scopes are flat: `memory:r` does not imply `memory:w`, and neither implies the other. See the [HTTP API scope table](../usage/http-api.md#scopes) for what each permits. An unrecognised scope logs a warning at startup and grants nothing, so a typo like `sessions:write` is visible rather than silently inert.

### `[[serve.webhooks]]`

Outbound endpoints meka POSTs to when something happens that no client is waiting on: a scheduled job firing, a background task finishing. Omit the block entirely and meka never makes an outbound request.

```toml
[[serve.webhooks]]
url = "https://bridge.example/meka-hook"
secret = "${MEKA_WEBHOOK_SECRET}"     # or secret_file = "/etc/meka/hook.secret"
events = ["turn.finished", "turn.failed", "task.finished", "schedule.fired"]
timeout = "10s"
max_retries = 3
```

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `url` | `string` | required | `https://` or `http://`; supports `${ENV_VAR}` |
| `secret` | `string` | none | HMAC key for `X-Meka-Signature`; supports `${ENV_VAR}` |
| `secret_file` | `path` | none | Mutually exclusive with `secret`; chmod 0600 |
| `events` | `array` | required | One or more of the four names above |
| `timeout` | `duration` | `"10s"` | Per attempt |
| `max_retries` | `integer` | `3` | Retries after the first attempt |

`events` is required and every name must be recognised. An unknown event is a startup **error**, not a warning, unlike an unknown token scope: a scope that grants nothing leaves the token working for whatever else it holds, whereas an endpoint whose only subscription is a typo is silently never called at all.

Payloads carry identifiers and metadata, never message content. Omitting `secret` sends unsigned deliveries and logs a warning. See [Webhooks](../usage/http-api.md#webhooks) for the payload shape and the signature-verification recipe.
