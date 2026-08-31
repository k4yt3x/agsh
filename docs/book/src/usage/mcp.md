# MCP

The [Model Context Protocol](https://modelcontextprotocol.io) is how meka reaches tools, resources
and prompts it does not implement itself. A server is a process meka spawns or an HTTP endpoint it
connects to; what it advertises is registered alongside the built-in tools and called the same way.

This page covers running servers: the command suite, where their secrets live, what happens on the
wire, and what the agent can reach. The keys themselves are in the
[config file reference](../configuration/config-file.md#mcp), which is also where tool permissions
are resolved.

## `meka mcp` CLI

Manage configured servers without editing `config.toml` by hand:

| Command | Action |
|---|---|
| `meka mcp list` | Print all configured servers, plus any stored OAuth credential that no server claims (see [Leftover credentials](../configuration/config-file.md#leftover-credentials)). |
| `meka mcp get <name>` | Print full details for one server. |
| `meka mcp add <name> <url-or-command> [args...] [flags]` | Persist a server. Transport is auto-detected: a URL starting with `http[s]://` means HTTP, anything else means stdio. Preserves existing formatting/comments via `toml_edit`. |
| `meka mcp remove <name>` | Best-effort revoke stored OAuth tokens (RFC 7009) at the provider, then delete the server entry, clear stored credentials, and drop any resource-update ledger entries. A name with stored credentials but no config entry is cleaned rather than refused. |
| `meka mcp disable <name>` | Set `disabled = true` on the server entry. The next `meka` start skips it entirely. |
| `meka mcp enable <name>` | Clear the `disabled` flag, so the server connects on the next start. |
| `meka mcp reconnect <name>` | Smoke-test a connect; prints `ok` or the error. |
| `meka mcp tools <name>` | Connect and list every advertised tool with its resolved permission, the chain step that decided it, and whether the current config allows it. Useful for populating `--allow-tool`, `--disable-tool`, or `--tool-permission` overrides without leaving the CLI. |
| `meka mcp login <name>` | Drive interactive OAuth. If the server has no `[auth]` block and uses HTTP, assumes `type = "oauth"` and persists the block on success. With `--auth-token-stdin` or `--client-secret-stdin`, stores that secret and exits instead, which is also how you rotate one. |
| `meka mcp logout <name>` | Call the provider's `revocation_endpoint` (RFC 7009) best-effort, then clear every stored credential for the server. |

### Credentials

An MCP server's bearer token, OAuth client secret and OAuth token bundle are stored in meka's database (`mcp_credentials`, keyed by server name and kind), never in `config.toml`. This is the same rule providers follow, and for the same reason: `config.toml` is a plaintext file people commit, sync and share.

Each is read from stdin so it never reaches `ps` output or your shell history. One command reads one secret, so `--auth-token-stdin` and `--client-secret-stdin` cannot be combined:

```console
$ pass show notion-token | meka mcp add notion https://mcp.notion.com/mcp --auth-token-stdin
$ pass show acme-secret | meka mcp login acme --client-secret-stdin
```

A confidential OAuth client holds two at once: the long-lived client secret it authenticates with, and the refreshable bundle it obtained. Store the secret first, then run `meka mcp login <name>` to complete the flow. Refreshing the bundle leaves the client secret alone.

`meka mcp get <name>` lists which kinds a server has, without printing any of them, and shows the origin an OAuth bundle was issued for as `issued for: <scheme>://<host>[:port]`. That flags the case a rotated `url` leaves behind: a bundle minted against the old host is still stored and still sent, so the line names a mismatch rather than letting the next call fail as a bare `401`. `meka mcp list` names servers that have a stored credential but no `[[mcp.servers]]` entry, which is what a hand-edited config strands.

### `meka mcp add` flags

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

### Example: Notion

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

### Remote hosts / SSH sessions

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

### REPL parity

Inside the REPL:
- `/mcp list`: list configured servers.
- `/mcp reconnect <server>`: reconnect smoke-test.
- `/mcp login <server>` / `/mcp logout <server>`: run the auth flow or revoke.
- `/mcp <server>:<prompt> [args...]`: render a server-defined prompt as the next user turn.

## Resources and prompts

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

## Startup concurrency

MCP servers connect in parallel at startup, partitioned by transport so a fleet of stdio servers (process-spawn bound) doesn't fight a fleet of HTTP servers (network bound):

- stdio: `MEKA_MCP_STDIO_CONCURRENCY` (default `3`)
- http: `MEKA_MCP_HTTP_CONCURRENCY` (default `20`)

These env vars are tuning knobs: rarely needed, but useful if you're running ~30 stdio servers on a constrained box (lower it) or ~50 HTTP servers (raise it).

## Connection lifecycle

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

## Server-to-client features

| Feature | meka behaviour |
|---------|----------------|
| `elicitation/create` | Routed to the calling session's frontend (REPL / ACP form or URL prompt) with a 60s timeout. Auto-declines when no in-flight tool call's frontend is registered or the user doesn't answer in time. |
