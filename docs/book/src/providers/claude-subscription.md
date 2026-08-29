# Claude subscription

The **Anthropic Messages API** billed to a Claude subscription. Authenticates by OAuth and mimics the Claude Code CLI's exact request shape, headers and request signing. Use this instead of a per-token Anthropic API key; for that, see [`anthropic-messages`](./anthropic-messages.md), which speaks the same protocol.

Named for the subscription rather than the protocol because that is what you are choosing: the endpoint is always `api.anthropic.com` and the client shape comes with the billing relationship.

> **Note:** This provider replicates Claude Code's fingerprinting and attestation machinery exactly. Modifying the request body, headers, or OAuth flow will cause requests to be rejected by Anthropic. If you hit 401/403 errors, verify that no middleware is rewriting the request.

## Configuration

| Setting | Value |
|---------|-------|
| Profile `type` | `claude-subscription` |
| Default base URL | `https://api.anthropic.com` |
| Credential | OAuth bundle stored in the database (acquired via `meka provider add` / `login`) |
| Auth method | `Authorization: Bearer <oauth_token>` |
| API version | `2023-06-01` |

### Quickest Start

```bash
meka provider add work --type claude-subscription --model claude-opus-5
```

`meka provider add` opens your browser, walks you through authorization, and saves the tokens to the
local database. It also writes the `[providers.work]` profile and sets it as the default.

### Config File

`meka provider add` writes this for you; you can also edit it by hand (secrets stay in the database):

```toml
default_provider = "work"

[providers.work]
type = "claude-subscription"
model = "claude-opus-5"
effort = "xhigh"         # optional; unset sends "high", as Claude Code does
thinking = "adaptive"    # optional; "adaptive"|"budgeted"|"off", default "adaptive"
redact_thinking = true   # optional; default on, matching Claude Code
# device_id, oauth_token_url, client_id are all optional overrides
```

See [Configuration → Config File](../configuration/config-file.md) for the full list of fields.

## Provider-specific knobs

### `effort`

Sent as `output_config.effort` under the `effort-2025-11-24` beta. When unset, meka sends `high`, which is what Claude Code does; only a model that takes no effort at all gets neither the field nor the beta. An explicit value is absolute: sent verbatim, with no validation or clamping, whatever model it is aimed at. Typical values: `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"`. See [Reasoning effort](#reasoning-effort).

### `thinking`

`adaptive` (the default) sends `thinking: {"type": "adaptive"}`; `budgeted` sends `{"type": "enabled", "budget_tokens": N}` from the profile's `thinking_budget` (falling back to [`[thinking].budget_tokens`](../configuration/config-file.md#thinkingbudget_tokens)), which pre-4.6 models require; `off` sends no thinking field. `temperature` follows whether thinking is on at all, not which encoding it uses. The betas do not: they are gated on the model alone.

### `redact_thinking`

Adds the `redact-thinking-2026-02-12` beta header for capable models, matching Claude Code, which sends it by default. With it on, the server withholds the readable chain of thought: `thinking` blocks come back with empty text plus a signature, and any `redacted_thinking` blocks carry an opaque `data` payload. meka preserves and replays both verbatim, so multi-turn reasoning continuity is maintained. The practical effect is that live thinking output goes quiet for these models (there is no readable text to show), exactly as in Claude Code. Defaults to `true`; set `redact_thinking = false` to drop the beta and keep interleaved thinking visible.

A stored block records that its signature is Claude's, so resuming the session under an OpenAI profile does not replay a Claude signature as encrypted reasoning. A session recorded by 0.41 holds its blocks under a shape that names no provider, and meka does not reshape them when it opens a session; the [one-shot upgrade script](../getting-started/upgrading.md) does. Until it runs, such a block keeps its readable text and loses its signature, so those turns are not replayed as verified reasoning.

### `device_id`

Stable per-machine identifier embedded in `metadata.user_id` to mirror Claude Code's `~/.claude.json` device ID (`getOrCreateUserID` in `utils/config.ts`).

If unset, meka first tries to adopt `userID` from `~/.claude.json` (so meka and Claude Code on the same machine present as the same device). If that file is missing or has no `userID`, meka generates a 64-character hex string. Either way the resolved value is persisted back to `[providers.<name>].device_id` in `config.toml`. Other backends ignore this field; no stub config file is written for them.

### `client_id`

Optional override for the OAuth client ID. Defaults to Claude Code's client ID; rarely needed.

## Authentication

### OAuth login

`meka provider add` (and `meka provider login <name>` to re-authenticate) performs an OAuth 2.0 Authorization Code flow with PKCE:

1. meka generates a PKCE challenge and opens your browser to Claude's authorization page.
2. You authorize the application in your browser.
3. You paste the authorization code back into meka (the redirect URI is the platform.claude.com hosted callback page, not a local listener).
4. meka exchanges the code for access + refresh tokens.
5. Tokens are stored in the local database and refreshed automatically.

The OAuth client ID defaults to Claude Code's client ID but can be overridden per profile via `client_id`.

### Token Lifecycle

1. Acquire the initial token with `meka provider add` / `login`.
2. The token bundle is stored in the database, keyed by the profile name.
3. On subsequent launches the token is loaded from the database.
4. meka refreshes the access token automatically when it's within 5 minutes of expiry; the new token is written back to the database under the same profile.
5. If the refresh token dies, run `meka provider login <name>` to re-authenticate. meka says so itself: a refresh the authorisation server *rejects* ends the turn with that command in the error, naming the profile. A refresh that fails because the token endpoint is rate-limited or down is retried with backoff instead, since neither answer means the grant is bad.

**Token refresh URL:** defaults to `https://api.anthropic.com/v1/oauth/token`. Configurable via `oauth_token_url` in the profile.

## Supported Models

Any model your Claude Code subscription exposes. For the current line-up and their retirement dates, see [Anthropic's models overview](https://docs.claude.com/en/docs/about-claude/models/overview) - `meka provider add` suggests `claude-opus-5` for new Claude profiles.

meka forwards the model string verbatim and doesn't gate which strings are valid. What is model-derived is a small set of gates, each pointed the way Claude Code points it. `temperature` is an allowlist, so an unrecognised model omits the field rather than earning a 400: it goes only to the models that still accept sampling params (Opus 4.6, Sonnet 4.6, Haiku 4.5, and older). `mid-conversation-system-2026-04-07` and `output_config.effort` are denylists, so an unrecognised model gets both: withholding the first would silently drop mid-conversation system messages, and effort is what a newer model is for. The `claude-code-20250219` beta is skipped for the Haiku tier. See [Beta header](#beta-header) and [Reasoning effort](#reasoning-effort).

## API Details

**Endpoint:** `POST {base_url}/v1/messages?beta=true`

**Authentication & identity headers:**

- `Authorization: Bearer <oauth_token>`
- `anthropic-version: 2023-06-01`
- `anthropic-beta: <comma-separated beta list>` (computed per request, see below)
- `x-app: cli`
- `User-Agent: claude-cli/<version> (external, cli)`
- `X-Claude-Code-Session-Id: <uuid>` (per-process)
- Stainless SDK identification headers (`x-stainless-*`)

### Beta header

Composed dynamically from the model + thinking settings, mirroring Claude Code's own assembly. Order is significant; the list below matches the Claude Code 2.1.241 interactive-CLI wire capture (tools present, thinking on) exactly:

| Beta | When |
|------|------|
| `claude-code-20250219` | All models *except* Haiku family |
| `oauth-2025-04-20` | Always (subscription auth) |
| `interleaved-thinking-2025-05-14` | Any modern Claude (4.x+) |
| `redact-thinking-2026-02-12` | Any modern Claude (4.x+); on by default, `redact_thinking = false` opts out |
| `thinking-token-count-2026-05-13` | Any modern Claude (4.x+) |
| `context-management-2025-06-27` | Any modern Claude (4.x+) |
| `prompt-caching-scope-2026-01-05` | Always |
| `mid-conversation-system-2026-04-07` | Everything except Claude 3.x, Opus 4.7 and older, Sonnet 4.6 and older, and Haiku 4.5 |
| `advanced-tool-use-2025-11-20` | When the request carries tools (meka always does) |
| `effort-2025-11-24` | Every model that takes an effort at all, whether or not the profile set one |
| `fallback-credit-2026-06-01` | Always. Claude Code latches it on every interactive turn; it only advertises that the server may answer with a fallback credit, and meka sends no `fallbacks` of its own |
| `extended-cache-ttl-2025-04-11` | Always (meka sends a 1h cache TTL) |

meka does **not** send `context-1m-2025-08-07`: Claude Code stopped sending it after 2.1.185, because 1M is the default context window (no beta header) on the current large-context models.

### System prompt

Sent as an array of three `text` blocks:

1. `x-anthropic-billing-header: cc_version=<version>.<fingerprint>; cc_entrypoint=cli; cch=<xxHash64-attestation>;` plus, when they apply, ` cc_is_subagent=true;`, ` cc_prev_req=<request id>;` and ` cc_prompt_id=<uuid>;`, in that order. The fingerprint suffix is a 3-character hex hash derived from the first user message (`SHA256(salt + msg[4] + msg[7] + msg[20] + version)[:3]`); the `cch` token is xxHash64 of a filtered copy of the serialized request body, computed and patched in just before send.

   `cc_prompt_id` identifies one human prompt and stays the same across every request that prompt produces, including the whole tool loop; a sub-agent inherits its spawner's. `cc_prev_req` names the `request-id` of the previous response in the same conversation, so it is absent on a conversation's first request. Both are absent from meka's own side queries, which is where Claude Code omits them too.
2. `You are Claude Code, Anthropic's official CLI for Claude.` (fixed identity prefix).
3. Your own system prompt, which carries `cache_control: {type: "ephemeral", ttl: "1h", scope: "global"}`.

Only block 3 is marked for caching, matching the captured Claude Code CLI wire; `scope: "global"` shares the cached prefix across sessions. Tools carry no `cache_control` (the rolling last-message breakpoint caches the tools+system prefix).

### Body key order

Keys are serialized in Claude Code's own order, which HTTP preserves:

```
model, messages, system, tools, metadata, max_tokens, thinking,
[temperature], [context_management], [output_config], stream
```

Nothing in meka depends on that order. `patch_request_body` finds the `cch=00000` placeholder by walking the JSON structurally to the *top-level* `system` key rather than by searching for the billing header, so a conversation that quotes one - which any session about this code does - cannot capture the attestation.

### Other body fields

- `metadata.user_id`: JSON-encoded `{"device_id": "...", "account_uuid": "...", "session_id": "..."}` (`device_id` from the profile's `device_id`; `account_uuid` from the OAuth token, empty until one is known; `session_id` is per-process).
- `context_management.edits = [{type: "clear_thinking_20251015", keep: "all"}]`: present when thinking is enabled on a context-management-capable model. Mirrors Claude Code's `apiMicrocompact`.
- `output_config.effort`: see [Reasoning effort](#reasoning-effort).
- `temperature: 1` (only when `thinking = "off"`, and only for models that still accept sampling params).
- `max_tokens`: `64_000` under `thinking = "adaptive"`, `max(thinking_budget * 2, 32_000)` under `budgeted`, `32_000` under `off`.

### Reasoning effort

Claude Code never leaves `output_config.effort` to the server on a model that takes one: it looks the model up in a table bundled in its binary, reads that model's `default_effort`, clamps it to what the model supports, and sends the result. meka also always sends a value, but one value rather than a per-model one, and sends the `effort-2025-11-24` beta alongside it.

| | sent |
|---|---|
| profile sets `effort` | that value, verbatim |
| profile sets nothing | `high` |
| model takes no effort | nothing, and no beta; a configured value is dropped with a warning |

One value for every model, not a copy of that table. `high` is what Claude Code's own resolution produces for almost every effort-capable model in the 2.1.241 table once the clamps have run, and it is what Claude Code falls back to for any model the table does not list. Carrying the per-model figures instead would add facts about Anthropic's data that go stale on their release schedule and buy nothing, because the server cannot tell a default meka chose from a value you configured. Models that take no effort at all are the Claude 3.x line, Opus 4.0/4.1, Sonnet 4.0/4.5 and Haiku 4.5.

A value you configure is absolute. Claude Code silently lowers `xhigh` or `max` to `high` on a model whose bundled entry lacks the capability; meka does not, because that table is a snapshot of someone else's system and quietly overriding what you asked for on the strength of it is worse than letting the API answer.

Only `claude-subscription` does this. `anthropic-messages` still omits `effort` when the profile sets none, because it can point at any Anthropic-compatible endpoint and has no standing to assert a default there.

### Cache control

The most recent message's last content block and the user system prompt carry `cache_control: {type: "ephemeral", ttl: "1h"}`. The 1h TTL is what an OAuth subscriber's Claude Code turn carries on the wire.

Caching is prefix-based: the system prompt precedes the tools array, which precedes the messages, so a byte changing early invalidates everything after it. meka is built so that nothing which changes mid-session sits in that prefix.

- **The system prompt is fixed for a session.** It carries only the role description, permission model, user instructions, guidelines, and OS/shell info, all resolved once at startup. The tool catalogue, skill list, and MCP server instructions live in the per-turn `<context>` block instead, because all three can change while a session runs.
- **The tools array only grows at the tail.** `load_tool` appends a schema rather than reordering, so the earlier entries stay byte-identical.
- **Permission toggles cost nothing.** See [Permissions](../usage/permissions.md).

Two things do legitimately invalidate it, both by necessity rather than oversight: compaction, which rewrites the head of the conversation, and an MCP server withdrawing a tool via `tools/list_changed`, which has to be removed from the tools array. The latter is confined to the array, leaving the system prompt ahead of it intact.

You can see the effect directly: `/status` reports the cache hit ratio, and reads should dominate from the second turn onward.

### Streaming

Server-Sent Events with the same event taxonomy as [`anthropic-messages`](./anthropic-messages.md): `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`. Reasoning streams as `thinking_delta` events; redacted thinking arrives as a `redacted_thinking` block carrying an opaque `data` payload and no signature, rendered as `[redacted thinking]`.
