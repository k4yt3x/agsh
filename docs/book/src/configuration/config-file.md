# Config File

meka looks for a TOML configuration file at a platform-specific location:

| Platform | Path |
|----------|------|
| Linux | `~/.config/meka/config.toml` (`$XDG_CONFIG_HOME/meka/config.toml`) |
| macOS | `~/Library/Application Support/meka/config.toml` |
| Windows | `%APPDATA%\meka\config.toml` |

The config file is optional. If it does not exist, meka silently skips it.

meka rejects unknown keys: a typo (`contex_window`) or a removed key (`reasoning_effort`) fails the load with an error naming the offending key, rather than being silently ignored. Fix or remove the key to continue.

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
type  = "claude-oauth"
model = "claude-opus-4-6"

[providers.local]
type     = "openai-api"
base_url = "http://localhost:11434/v1"
model    = "llama3"
```

### Selecting the active profile

For each run meka picks one profile using this precedence:

1. `--provider <name>` CLI flag.
2. `default_provider` in the config file.
3. The sole profile, if exactly one is configured.

If none of these resolve (no profiles configured, or more than one with no `default_provider` /
`--provider`), meka errors and points you at `meka provider add` / `meka provider use`. There is no
environment-variable tier for provider selection; the config file (plus the per-run CLI flag) is the
source of truth.

## `default_provider`

Top-level field naming the profile to use when `--provider` isn't passed. Set it with
`meka provider use <name>`; `meka provider add` sets it automatically when adding the first profile.

## Profile fields

### `type`

The backend the profile uses (required).

| Value | Description |
|-------|-------------|
| `openai-api` | OpenAI Chat Completions API (also works with OpenAI-compatible APIs) |
| `openai-codex` | OpenAI Responses API via ChatGPT subscription OAuth, against `chatgpt.com/backend-api/codex` |
| `claude-api` | Claude Messages API with `x-api-key` auth |
| `claude-oauth` | Claude Messages API via Claude Code OAuth (fingerprinting + attestation) |

### `model`

The model identifier to send to the provider. Examples:

- `gpt-4o`, `gpt-4o-mini`, `gpt-5` (OpenAI)
- `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5` (Claude)
- Any model supported by an OpenAI-compatible endpoint

Override per-run with `--model`.

### `base_url`

Custom API base URL. Useful for:

- Self-hosted models via [Ollama](https://ollama.ai) (`http://localhost:11434/v1`)
- [OpenRouter](https://openrouter.ai) (`https://openrouter.ai/api/v1`)
- Other OpenAI-compatible API providers

If not set, defaults to:

- `https://api.openai.com/v1` for the `openai-api` backend
- `https://chatgpt.com` for the `openai-codex` backend (request path is `/backend-api/codex/responses`)
- `https://api.anthropic.com` for the `claude-api` and `claude-oauth` backends

Override per-run with `--base-url`.

### `oauth_token_url`

Custom OAuth token refresh endpoint. Defaults:

- `https://api.anthropic.com/v1/oauth/token` for `claude-oauth`
- `https://auth.openai.com/oauth/token` for `openai-codex`

### `effort`

One knob for reasoning effort across every backend: Claude sends it as `output_config.effort` (`claude-oauth` under the `effort-2025-11-24` beta, `claude-api` directly), OpenAI as `reasoning.effort` (with `max_completion_tokens` in place of `max_tokens`).

When unset, meka picks a model-aware default: `xhigh` on models that support it (Claude Opus 4.7+/Sonnet 5/Fable/Mythos 5; OpenAI gpt-5.2+, gpt-5.1-codex-max), `high` on effort-capable models without `xhigh` (Claude Opus 4.5/4.6, Sonnet 4.6; OpenAI o-series, gpt-5, gpt-5.1), and the field is omitted entirely on models with no effort knob (Claude Sonnet 4.0/4.5, Opus 4.1, Haiku 4.5; OpenAI gpt-4o, o1-mini, and unrecognised names such as local models served through `openai-api`).

For `openai-codex`, the supported tiers come from your account's models catalog (`/models`), which is authoritative: the default stays accurate even for models newer than meka, and the name-based lists above are only the fallback when the catalog is unavailable. Claude and the public OpenAI API expose no such catalog, so they always use the name-based lists. The lookup is one bounded, cached probe per model (see [`session.context_window`](#sessioncontext_window)).

An explicit value is absolute: sent verbatim, with no validation or clamping, and applied even to a model the default would omit it for. You own correctness for your model/endpoint; an invalid value is rejected by the API.

Typical values: `low`, `medium`, `high`, `xhigh`, `max`.

```toml
[providers.work]
type   = "claude-oauth"
effort = "xhigh"
```

### `redact_thinking`

`claude-oauth` only. Sends the `redact-thinking-2026-02-12` beta header for capable models, matching Claude Code, which enables it by default. With it on the server withholds the readable chain of thought: `thinking` blocks return with empty text plus a signature, and `redacted_thinking` blocks carry an opaque `data` payload. meka preserves and replays both verbatim, so multi-turn continuity holds; the visible effect is that live thinking output goes quiet for these models. Defaults to `true`; set `false` to drop the beta and keep interleaved thinking visible.

```toml
[providers.work]
type            = "claude-oauth"
redact_thinking = false
```

### `context_window`

Override the model's context window (total tokens it can hold), used for the `/status` gauge and auto-compaction. Takes precedence over `[session].context_window`, which in turn overrides the model-name inference. Leave unset to use the inferred default.

```toml
[providers.work]
type           = "openai-api"
model          = "my-128k-model"
context_window = 131072
```

### `vision`

Whether this profile's model accepts image input. Defaults to `true`. Set `false` for a text-only model so the ACP frontend stops advertising and accepting images (see [ACP](../usage/acp.md)).

```toml
[providers.local]
type   = "openai-api"
model  = "llama-3-8b"
vision = false
```

### `max_output_tokens`

Override the per-request output (completion) token cap. When unset, each backend keeps its built-in default (Claude 32k–64k depending on thinking; OpenAI 32k with reasoning effort; otherwise the API default). On Claude with thinking enabled, the value must exceed `[thinking].budget_tokens` (validated at startup).

```toml
[providers.work]
type              = "claude-api"
max_output_tokens = 16000
```

### `client_id`

OAuth client ID override (advanced; `claude-oauth` / `openai-codex` only). Leave unset to use meka's built-in default client IDs.

### `device_id`

`claude-oauth` only. Stable per-device identifier embedded in `metadata.user_id` to mirror Claude Code's `~/.claude.json` device ID (`getOrCreateUserID` in `utils/config.ts`).

If unset, meka first tries to adopt `userID` from `~/.claude.json` (so meka and Claude Code on the same machine look like the same device). If that file is missing or has no `userID`, meka generates a 64-character hex string. Either way, the resolved value is persisted back to the profile under `[providers.<name>].device_id`. This file write only happens for the `claude-oauth` backend; other backends don't need a device ID.

You can supply your own value if you want to control attribution explicitly:

```toml
[providers.work]
type      = "claude-oauth"
device_id = "your-stable-id-here"
```

## `meka provider` CLI

Add, switch, and remove profiles without editing `config.toml` by hand. The credential prompt /
OAuth login runs as part of `add` and `login`, and secrets are written to the database, never the
config file.

| Command | Action |
|---|---|
| `meka provider add <name> [--type T] [--model M] [--base-url U] [--api-key-stdin]` | Add a profile. Prompts for any of type/model interactively when not flagged (the model prompt offers a backend default: `claude-opus-4-8` for Claude, `gpt-5.6-sol` for OpenAI), then acquires the secret (OAuth login for `claude-oauth` / `openai-codex`, API-key prompt for `claude-api` / `openai-api`). Sets `default_provider` when it's the first profile. |
| `meka provider list` | List configured profiles with type, model, the default marker, and whether each has a stored credential. |
| `meka provider use <name>` | Set `default_provider` to this profile. |
| `meka provider login <name>` | Re-acquire the secret for an existing profile (re-authenticate, recover from a dead OAuth refresh token, or rotate an API key). |
| `meka provider remove <name>` | Logout + delete: best-effort revoke the OAuth token, delete the stored credential from the database, and remove the `[providers.<name>]` entry from the config file. |

`--api-key-stdin` reads the key from standard input instead of prompting, for scripted setup:

```console
$ printf '%s' "$OPENAI_API_KEY" | meka provider add local --type openai-api --model gpt-4o --api-key-stdin
```

## Examples

### Claude OAuth (Claude Code subscription)

```console
$ meka provider add work --type claude-oauth --model claude-opus-4-6
# Opens the browser for the OAuth login, then stores the token in the database.
```

### Claude API

```console
$ meka provider add anthropic --type claude-api --model claude-opus-4-6
# Prompts for your CLAUDE API key (sk-ant-api03-...).
```

### OpenAI API

```console
$ meka provider add openai --type openai-api --model gpt-4o
# Prompts for your OpenAI API key (sk-...).
```

### OpenAI Codex (ChatGPT subscription)

```console
$ meka provider add chatgpt --type openai-codex --model gpt-5
# Opens the browser for the ChatGPT OAuth login.
```

### Ollama (local, no key)

```console
$ printf 'unused' | meka provider add ollama --type openai-api --model llama3 \
    --base-url http://localhost:11434/v1 --api-key-stdin
```

### OpenRouter

```console
$ meka provider add openrouter --type openai-api --model anthropic/claude-sonnet-4.6 \
    --base-url https://openrouter.ai/api/v1
# Prompts for your OpenRouter key (sk-or-...).
```

## `[display]`

Settings for output formatting.

### `display.render_mode`

Output render mode. Equivalent to the `--render-mode` CLI flag.

| Value | Description |
|-------|-------------|
| `syntect` | Syntax-highlighted markdown via `syntect`, incl. per-language code blocks (default) |
| `termimad` | Terminal formatting via termimad (box-drawn code blocks, reflowed paragraphs). Alias: `rich` |
| `raw` | Raw markdown printed verbatim with aligned tables |

Default: `syntect`

```toml
[display]
render_mode = "raw"
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

Whether to add a blank line before the prompt after each agent response.

Default: `true`

### `display.newline_after_prompt`

Whether to add a blank line after the prompt (before the agent response).

Default: `true`

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

Visual style applied to text typed into the REPL prompt. Makes submitted prompts easy to spot when scrolling back through a long session. Reedline paints the buffer with this style on every repaint, including the final paint before the newline, so the styling lands in the terminal's scrollback alongside the literal text.

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

Settings for the HTTP client shared by `fetch_url` and `web_search`. All keys are optional; unset fields use the defaults shown below.

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

Whether to enable read-only filesystem sandboxing for shell commands in read mode. When enabled (default), shell commands can be executed in read mode but with the filesystem physically write-protected. When disabled, shell commands require write mode.

Default: `true`

```toml
[shell]
sandbox = false  # disable sandboxed shell in read mode
```

The sandbox uses one of two backends on Linux (see [`shell.sandbox_backend`](#shellsandbox_backend)), `sandbox-exec` on macOS, and a duplicated Low-integrity primary token on Windows. On platforms where no backend is usable, shell commands always require write mode regardless of this setting.

### `shell.sandbox_backend`

Linux-only choice between `"landlock"` and `"bubblewrap"`:

- **Bubblewrap** (`"bubblewrap"`) wraps the command in `bwrap` with read-only bind of `/`, tmpfs masks over `/run` / `/tmp` / `/var/tmp` / `$XDG_RUNTIME_DIR`, and `--unshare-user --unshare-pid --unshare-uts --unshare-ipc`. The tmpfs masks hide the dbus session bus and the systemd-user socket, so state-changing IPC calls like `systemctl --user start` and `dbus-send` fail. Network is intentionally not unshared so `curl http://x | pdftotext` still works. Requires the `bubblewrap` package and a kernel with user-namespace creation enabled.
- **Landlock** (`"landlock"`) uses the Landlock LSM (kernel 5.13+) to block filesystem writes. Does **not** block dbus / systemd-user IPC; a sandboxed shell can still invoke state-mutating dbus methods. Kept as the lighter-weight fallback for hosts without Bubblewrap.

When omitted, meka probes Bubblewrap once at startup. If Bubblewrap is available it auto-picks it; otherwise it auto-picks Landlock and emits a one-shot warning nudging you to install `bubblewrap` for stronger protection. Set the field explicitly to either value (including `"landlock"`) to suppress that warning. `meka provider add` does not write this field; leave it unset to keep auto-detection.

If the configured backend can't be used at runtime (bwrap not installed, user namespaces denied, etc.), `execute_command` in read mode hard-errors with a message naming the configured backend and the specific failure reason. Read mode is not blocked for other tools; only `execute_command` requires a usable sandbox.

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
| `default` | No | Mode the session starts in. One of `"none"`, `"read"`, `"ask"`, `"write"`. Default `"read"`. Overridden by `--permission` and `MEKA_PERMISSION`. |
| `enabled` | No | List of modes that can be reached at runtime via `/permission` and Shift+Tab. Default `["none", "read", "write"]`; `"ask"` is opt-in. Disabled modes are skipped during Shift+Tab cycling and rejected by `/permission` with an error. |

If `default` is not in `enabled`, meka logs a warning and falls back to `read` if it's enabled, otherwise the lowest-discriminant enabled mode (in `none → read → ask → write` order). Same behavior if `--permission` or `MEKA_PERMISSION` selects a disabled mode: meka warns and starts in the configured default rather than refusing to launch.

```toml
[permissions]
default = "read"
enabled = ["none", "read", "ask", "write"]  # opt back into ask
```

## `[session]`

Settings for session history retention and context window management.

### `session.context_messages`

Maximum number of messages to send to the LLM API per request. Older messages are truncated from the beginning while preserving tool call chain integrity. The full history remains stored in SQLite; only the API payload is limited.

Default: `200`

```toml
[session]
context_messages = 100
```

### `session.retention_days`

Automatically delete sessions older than this many days on startup. Uses the session's `updated_at` timestamp, so actively-resumed sessions are preserved even if originally created long ago.

Default: `90`

```toml
[session]
retention_days = 30
```

### `session.max_storage_bytes`

Maximum total byte size of all stored message content across all sessions. When exceeded on startup, the oldest sessions are deleted until the total is under the limit.

Default: `52428800` (50 MB)

```toml
[session]
max_storage_bytes = 10485760  # 10 MB
```

### `session.auto_compact`

Automatically compact the conversation when input tokens exceed 80% of the context window. Compaction summarizes older messages and preserves recent ones, the todo list, and scratchpad entries.

Default: `true`

```toml
[session]
auto_compact = false
```

### `session.context_window`

Override the model's context window size (in tokens). Used for auto-compact threshold calculation. A per-profile `[providers.<name>].context_window` takes precedence over this.

When neither is set, meka resolves the window automatically. A built-in table is authoritative for recognized models: it encodes each model's real window as meka's request receives it (for example 1M for Opus 4.8, which is the default on the direct API and what `claude-oauth` negotiates by mirroring Claude Code), so it reflects the window the request truly gets and wins over the live API. For a model the table doesn't recognize, meka makes one bounded query of the provider's models API (supported for `openai-codex` and the Claude backends; not the public OpenAI API), falling back to a 128k floor. That single probe also carries the reasoning-effort levels (for `openai-codex`), so window and [`effort`](#effort) share one lookup, and the whole result is cached in the session database keyed by profile and model so later runs don't re-query. Setting this option pins the window and skips the table and the API lookup for the window (a catalog probe may still run to resolve effort).

```toml
[session]
context_window = 200000
```

### `session.subagent_max_depth`

Maximum recursion depth for sub-agents spawned via [`spawn_agent`](../tools/overview.md#spawn_agent). The root agent spawns at depth 1, its sub-agents at depth 2, and so on; each level below this limit is granted its own `spawn_agent`. `1` reproduces the historical behavior where sub-agents cannot spawn further sub-agents; `0` disables `spawn_agent` entirely. An agent can tune a subtree with the tool's `max_depth` parameter, but a built-in absolute cap always bounds real nesting so recursion can't run away.

Default: `3`

```toml
[session]
subagent_max_depth = 3
```

## `[thinking]`

Settings for extended thinking (`claude-api` and `claude-oauth` providers). Claude 4.6+ models use adaptive thinking automatically; older models use a fixed token budget.

### `thinking.enabled`

Whether to enable extended thinking. When enabled, the model can use additional tokens for internal reasoning before responding.

Default: `true`

### `thinking.budget_tokens`

Maximum number of tokens the model can use for thinking (for non-adaptive models).

Default: `16000`

### `thinking.show_content`

Whether to render thinking blocks inline in the terminal as the model produces them. When `false`, thinking is silently consumed (still sent on subsequent turns for cache continuity, just not displayed). When `true`, thinking deltas are streamed under a dimmed header.

Default: `false`

```toml
[thinking]
enabled = true
budget_tokens = 20000
show_content = true
```

## `[prompt]`

Settings for injecting custom instructions into the system prompt. Use this to set installation-specific rules that should apply to every session: things the agent needs to know about your system, preferred tools, or policies.

### `prompt.instructions`

A string of custom instructions that meka will include in every system prompt, under a `## User Instructions` section. The model is told to treat them as hard constraints unless they conflict with safety requirements.

Suitable use cases:

- System-specific policies: "Never install Python packages globally with pip; always use `uv` or a venv."
- Installed tooling the agent should know about: "Poppler is available on this system; use `pdftotext` for PDFs."
- Workflow preferences: "Prefer ripgrep over grep; it's installed and faster."
- Signing / compliance rules: "Git commits on this system must use gpg signing."

Default: unset (no custom instructions).

```toml
[prompt]
instructions = """
Never install Python packages globally with pip. Always use `uv` or a venv.
Poppler is available on this system, use `pdftotext` for PDFs.
Prefer ripgrep over grep.
"""
```

Notes:

- Empty or whitespace-only strings are treated as unset.
- Instructions apply to sub-agents spawned via `spawn_agent` too.
- Instructions are included at all permission levels (including `none`) because they are authored by you.
- Per-run override: [`--instructions`](./cli-options.md#instructions-string) (or `MEKA_INSTRUCTIONS`) replaces this value for a single invocation.

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
| `env` | No | Environment variables to set for the spawned process (stdio only). |
| `url` | HTTP only | URL of the MCP server endpoint. |
| `auth_token` | No | Bearer token for HTTP authentication (sent as `Authorization: Bearer <token>`). |
| `auth` | No | OAuth authentication configuration (see below). Mutually exclusive with `auth_token`. |
| `headers` | No | Custom HTTP headers to include with every request (HTTP only). |
| `headers_helper` | No | Path to an executable whose stdout (`Name: Value\n` lines) is merged over `headers` at connect-time (HTTP only). Executed with `MEKA_MCP_SERVER_NAME` / `MEKA_MCP_SERVER_URL` in env; 15 s timeout. |
| `permission` | No | Server-wide permission override. Applies to every tool on this server, beating the `readOnlyHint` the server advertises and the `[mcp].default_permission` global fallback. See *Permission resolution* below. |
| `allowed_tools` | No | Optional allow-list of raw tool names (the form the server advertises, not the `server__tool` namespaced form). When set and non-empty, only these tools are registered; all others from this server are ignored. |
| `disabled_tools` | No | Optional block-list of raw tool names. Applied **after** `allowed_tools`; tools listed here are never registered. Both lists can coexist; the net set is `allowed_tools \ disabled_tools`. |
| `eager_load_tools` | No | Raw tool names that should ship **eager-loaded** instead of deferred. Listed tools skip the `load_tool` round-trip and sit in the cacheable tools-array prefix from turn 1. Use this for tools the agent invokes constantly (search, fetch, …); leave others deferred so the tools array stays lean. |
| `tool_permissions` | No | Per-tool permission overrides keyed by raw tool name. Beats the server-level `permission` and the server's `readOnlyHint` when resolving a tool's required permission. |
| `disabled` | No | When `true`, the server is skipped entirely at startup: no process is spawned, no HTTP connect is attempted. Flip it back with `meka mcp enable <name>` or by editing the config. Defaults to `false`. |

### `[mcp]` top-level table

| Field | Purpose |
|-------|---------|
| `default_permission` | Fallback permission for MCP tools whose server didn't advertise `readOnlyHint` and doesn't have a `permission` override. Accepts `"none"`, `"read"`, `"ask"`, or `"write"`. If unset the hardcoded fallback is `"write"` (strict). |
| `strict` | When `true` (default), every turn is gated on all enabled MCP servers being `Connected`. If any are not, the turn is rejected with a shell-style error instead of sending the request to the model. Set to `false` to proceed with whichever servers are ready (a warn log names the missing ones). |
| `grace_seconds` | Per-turn cap on how long to wait for still-`Pending` servers to connect before applying the strict check. Default `3`. Set to `0` to skip waiting (useful for scripts that want to fail fast). |
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
3. **`tool.annotations.readOnlyHint`** from the server: `true` → `Read`, `false` → `Write`.
4. **`[mcp].default_permission`**: global fallback.
5. **Hardcoded `Write`**: strict ultimate fallback.

User-supplied config (1, 2, 4) always beats the server's self-classification; if a server lies about a tool, you can override. But when no user config says anything, the server's hint is trusted for that specific tool so `readOnlyHint = false` destructive tools don't silently become Read-accessible just because the user opted into a lenient global default.

**Hint spoofing**: a compromised server could claim `readOnlyHint = true` on a destructive tool. Defend by setting `server.permission = "write"` on suspect servers (step 2 wins) or by listing the destructive tools explicitly in `tool_permissions` / `disabled_tools`.

**Stale config**: entries in `allowed_tools` / `disabled_tools` / `eager_load_tools` / `tool_permissions` that don't match any advertised tool get a `warn!` line at connect time. The server still connects; you just see a heads-up so you can clean up after the server renames a tool. A name that appears in both `eager_load_tools` and `disabled_tools` also warns: the disabled filter wins, so eager-loading the disabled tool is a no-op.

**Visibility across levels**: the resolved permission doesn't hide a tool from the agent. Every registered tool is listed in the system prompt with its required level noted inline, and a per-turn `[Permission context]` block names the current level plus any tools it blocks. The agent can still reason about an inaccessible tool and suggest `/permission <level>` to enable it; the permission gate is enforced at dispatch time. Keeping the tool catalogue visible across levels is also what lets the Claude prompt cache survive mid-session permission toggles.

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

Overriding a mis-annotated or distrusted tool (one specific tool requires Write):
```toml
[[mcp.servers]]
name      = "notion"
transport = "http"
url       = "https://mcp.notion.com/mcp"

[mcp.servers.tool_permissions]
"notion-do-something-scary" = "write"
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

Tool and resource descriptions returned from MCP servers are truncated at 2048 characters to keep the system prompt bounded.

### Environment variable substitution

Every string field listed above (command, args, env values, url, headers values, auth_token) supports `${VAR}` and `${VAR:-default}` expansion from the process environment. Missing variables with no default leave the literal `${VAR}` in place and log a warning at startup. Use this to avoid committing secrets:

```toml
[[mcp.servers]]
name = "github"
transport = "http"
url = "https://mcp.github.com"
auth_token = "${GITHUB_MCP_TOKEN}"
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MEKA_MCP_TOOL_TIMEOUT` | `600000` ms (600 s) | Per-call timeout for MCP tools. Triggers `notifications/cancelled` on expiry. |

### `meka mcp` CLI

Manage configured servers without editing `config.toml` by hand:

| Command | Action |
|---|---|
| `meka mcp list` | Print all configured servers. |
| `meka mcp get <name>` | Print full details for one server. |
| `meka mcp add <name> <url-or-command> [args...] [flags]` | Persist a server. Transport is auto-detected: a URL starting with `http[s]://` means HTTP, anything else means stdio. Preserves existing formatting/comments via `toml_edit`. |
| `meka mcp remove <name>` | Best-effort revoke stored OAuth tokens (RFC 7009) at the provider, then delete the server entry, clear stored credentials, and drop any resource-update ledger entries. |
| `meka mcp disable <name>` | Set `disabled = true` on the server entry. The next `meka` start skips it entirely. |
| `meka mcp enable <name>` | Clear the `disabled` flag, so the server connects on the next start. |
| `meka mcp reconnect <name>` | Smoke-test a connect; prints `ok` or the error. |
| `meka mcp tools <name>` | Connect and list every advertised tool with its resolved permission, the chain step that decided it, and whether the current config allows it. Useful for populating `--allow-tool`, `--disable-tool`, or `--tool-permission` overrides without leaving the CLI. |
| `meka mcp login <name>` | Drive interactive OAuth. If the server has no `[auth]` block and uses HTTP, assumes `type = "oauth"` and persists the block on success. |
| `meka mcp logout <name>` | Call the provider's `revocation_endpoint` (RFC 7009) best-effort, then clear stored credentials + auth-probe cache. |

#### `meka mcp add` flags

| Flag | Purpose |
|------|---------|
| `--transport <stdio\|http>` | Override the auto-detected transport. |
| `--env KEY=VALUE` | Environment variable for stdio (repeatable). |
| `--header KEY=VALUE` | HTTP header (repeatable). |
| `--auth <oauth\|client-credentials\|client-credentials-jwt>` | Configure the `[auth]` block. |
| `--auth-token <TOKEN>` | Static bearer token. Mutually exclusive with `--auth`. |
| `--client-id`, `--client-secret` | OAuth / client-credentials client identifiers. |
| `--signing-key <PATH>`, `--signing-algorithm <ALG>` | JWT signing material (`client-credentials-jwt` only). |
| `--scope <SCOPE>` | OAuth scope (repeatable). |
| `--redirect-port <PORT>` | Fixed OAuth redirect port (default: ephemeral). |
| `--permission <none\|read\|ask\|write>` | Per-server permission cap (applies to all tools on the server). |
| `--allow-tool <NAME>` | Raw tool name to allow (repeatable). When set, only listed tools register. |
| `--disable-tool <NAME>` | Raw tool name to block (repeatable). Applied after `--allow-tool`. |
| `--eager-load-tool <NAME>` | Raw tool name to eager-load (repeatable). Listed tools skip the `load_tool` round-trip and ship in the cacheable tools-array prefix from turn 1. |
| `--tool-permission <NAME=LEVEL>` | Per-tool permission override (repeatable). `LEVEL` is `none`/`read`/`ask`/`write`. |

#### Example: Notion

```console
$ meka mcp add notion https://mcp.notion.com/mcp
ok: added 'notion' to ~/.config/meka/config.toml
probe: server requires OAuth.
running OAuth authorisation for 'notion' (use --no-login to skip).
no [auth] block for 'notion', assuming OAuth authorization_code.
…
ok: authorized 'notion'
```

`meka mcp add` on an HTTP endpoint:

1. **Probe**: issues an unauthenticated `GET` (3 s timeout, redirects off) and classifies the response per the MCP authorization spec + RFC 6750 + RFC 9728:

   - `2xx` → server is open, no login needed.
   - `401` / `403` with `WWW-Authenticate: Bearer …` → OAuth required. The `resource_metadata="…"` attribute (RFC 9728) is captured at DEBUG.
   - Any other status → couldn't infer, prints the status code.
   - Network failure → prints the error.

2. **Auto-login**: if the probe says OAuth is required (or `--auth oauth` was explicitly set), the OAuth authorization_code flow runs immediately as though the user had chained `meka mcp login <name>` themselves. The synthesised `[auth] = oauth` block is written back to `config.toml` on success.

3. **Rollback on failure**: if the OAuth flow errors out, the entry we just wrote is purged from `config.toml` (alongside any partial credentials + probe cache), leaving the user's config clean. The command exits non-zero.

4. **`--no-login`**: skips step 2. The entry is still persisted and the probe's hint is still printed; run `meka mcp login <name>` when ready. Useful for scripted setup or when you expect to edit `[auth]` by hand.

The probe and the auto-login only run for HTTP servers, and only when the user didn't provide `--auth-token` (static bearer) or `--auth` (other than `oauth`). Stdio servers skip both.

#### Remote hosts / SSH sessions

The OAuth flow redirects the browser to `http://127.0.0.1:<port>/callback`. When meka is running on a different host than the browser (SSH session, container, Codespace, WSL), the browser can't reach back and shows a "connection refused" error page. meka handles this automatically:

- While `meka mcp login <name>` waits for the callback it also watches stdin.
- The browser's address bar still contains the full callback URL (including `code` and `state`) even when the connection fails. Copy it, paste it into the meka prompt, and press Enter.
- Whichever completes first, the TCP callback or the pasted URL, wins.

```console
$ meka mcp login notion
server 'notion' has no [auth] block; assuming OAuth authorization_code.
Opening browser for MCP server 'notion' OAuth authorization...
If the browser didn't open, visit:
  https://mcp.notion.com/authorize?response_type=code&…
Waiting for OAuth callback (up to 120s).
  If the browser can't reach this host (e.g. you're over SSH), paste the full
  callback URL here and press Enter.
http://127.0.0.1:46437/callback?code=…&state=…     ← paste here
ok: authorized 'notion'
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
| `list_mcp_resources` | List resources from one or every configured server. |
| `read_mcp_resource` | Read a resource by `server` + `uri`; text inline, binary base64-encoded. |
| `list_mcp_prompts` | List prompts from one or every configured server, including their declared arguments. |
| `get_mcp_prompt` | Render a prompt by `server` + `name` with optional `arguments`; returns `<role>: <text>` lines. |
| `subscribe_mcp_resource` | Subscribe to `resources/updated` notifications for a specific URI. |
| `unsubscribe_mcp_resource` | Cancel a prior subscription. |
| `list_mcp_resource_updates` | Print every resource that has been reported as updated since the session started. |

### Connection lifecycle

- **Reconnection** is automatic for all transports (stdio, plain HTTP, OAuth-authenticated HTTP) when the transport closes mid-session. HTTP transports use exponential backoff (1s, 2s, 4s, 8s, 16s, capped 30s, max 5 attempts); stdio gets one immediate retry. The reconnect runs on a blocking thread to work around an upstream rmcp bug where the auth future is `!Send`.
- **Session-expired recovery**: rmcp 1.5 transparently re-initialises HTTP sessions on 404 / JSON-RPC `-32001`. meka relies on this; no per-call handling is required.
- **Cancellation**: when the agent cancels a tool call (e.g. Ctrl-C), meka sends `notifications/cancelled` to the server with the in-flight request id so the server can stop work.
- **Timeouts**: tool calls default to 600 s; override with `MEKA_MCP_TOOL_TIMEOUT` in ms.
- **Tool list refresh**: on `tools/list_changed`, meka re-discovers the server's tools and hot-swaps them in the registry; no restart needed.
- **Progress notifications**: MCP tool calls attach a per-request `progressToken`; incoming `notifications/progress` render as a live status line under the tool invocation.
- **Server instructions**: `InitializeResult.instructions` is captured once per connection and spliced into the system prompt (sanitised + truncated to 2048 chars) under `## MCP Server Instructions`.
- **stdio server logs**: a stdio server's own stderr (many servers log there) is captured, not inherited, so it never corrupts the REPL display. Each line is re-emitted on meka's `tracing` stream at `debug` level tagged with the server name, so it stays silent at default verbosity and surfaces under `-v` / `RUST_LOG`.
- **Auth-probe cache**: 401 responses are cached for 15 minutes so a restart after a failed auth flow skips the unauthenticated probe and goes straight to OAuth. Cleared by `meka mcp logout`.
- `resources/list_changed`, `prompts/list_changed`, and `resources/updated` notifications are logged at `info`/`debug` level.

### Server-to-client features

| Feature | meka behaviour |
|---------|----------------|
| `elicitation/create` | Routed to the calling session's frontend (REPL / ACP form or URL prompt) with a 60s timeout. Auto-declines when no in-flight tool call's frontend is registered or the user doesn't answer in time. |

### `[mcp.servers.auth]`

OAuth authentication for HTTP MCP servers. Set `type` to choose the authentication method. This is mutually exclusive with `auth_token`.

| Field | Required | Description |
|-------|----------|-------------|
| `type` | Yes | Auth method: `"client_credentials"`, `"client_credentials_jwt"`, or `"oauth"` |
| `client_id` | Varies | OAuth client ID (required for client_credentials/jwt, optional for oauth with dynamic registration) |
| `client_secret` | Varies | Client secret (required for client_credentials, optional for oauth) |
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
permission = "write"
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

```toml
[[mcp.servers]]
name = "api"
transport = "http"
url = "https://api.example.com/mcp"
auth_token = "your-bearer-token"
permission = "write"

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
permission = "write"
```

#### HTTP server with OAuth client credentials

```toml
[[mcp.servers]]
name = "api"
transport = "http"
url = "https://api.example.com/mcp"
permission = "write"

[mcp.servers.auth]
type = "client_credentials"
client_id = "my-client-id"
client_secret = "my-client-secret"
scopes = ["read", "write"]
```

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

The three knobs `[[mcp.servers]]` exposes for MCP tools also apply to meka's built-in tools (`read_file`, `write_file`, `execute_command`, `web_search`, etc.) via a top-level `[tools]` table. MCP per-server filtering is separate from this and keeps its own namespaces; this block only affects the built-ins.

| Key | Purpose |
|---|---|
| `allowed_tools` | Optional allow-list of built-in tool names. When set and non-empty, only these built-ins register. Use `meka tools list` to see the canonical names. |
| `disabled_tools` | Block-list of built-in tool names. Applied **after** `allowed_tools`; a tool here is never registered even if it also appears in the allow-list. |
| `tool_permissions` | Per-tool required-permission override keyed by built-in name. Beats the hardcoded required level from the tool's impl. Levels: `none`, `read`, `ask`, `write`. |

Stale entries (a name that doesn't match any built-in) emit a `warn!` at startup. meka still starts; the warning just flags a likely typo or a tool the binary renamed.

Restrict a session to read-only inspection:
```toml
[tools]
allowed_tools = ["read_file", "find_files", "search_contents", "fetch_url"]
```

Force `execute_command` to need `write` so `ask` mode prompts for every shell call:
```toml
[tools.tool_permissions]
execute_command = "write"
```

Disable web access entirely in a locked-down environment:
```toml
[tools]
disabled_tools = ["web_search", "fetch_url"]
```

Sub-agents spawned via `spawn_agent` inherit the same filter; a disabled built-in is disabled everywhere. Run `meka tools list` to see every built-in's effective required permission, whether a `[tools.tool_permissions]` override is in effect, and whether the current config enables it.

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

### `serve.max_concurrent_turns`

Process-wide cap on in-flight turns across all sessions. When the cap is reached, new turn submissions return `429 Too Many Requests` with a `Retry-After` header. Unset or `0` means no limit.

| Type | Default |
|------|---------|
| `integer` | unbounded |

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
| `scopes` | Yes | Array of scope strings: `"sessions:r"`, `"sessions:w"`, `"skills:r"`, `"mcp:r"`. |

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

Admin token with all read scopes:

```toml
[[serve.tokens]]
token = "${MEKA_ADMIN_TOKEN}"
description = "operator"
scopes = ["sessions:r", "sessions:w", "mcp:r", "skills:r"]
```
