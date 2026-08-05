# HTTP API

`meka serve` exposes meka as an HTTP API server so other programs can drive agent turns programmatically. Where [Interactive Mode](./interactive-mode.md) is for humans at a terminal and [ACP](./acp.md) is for editor integrations over stdio, the HTTP API is for **service-to-service** use cases:

- A Telegram or Discord bridge that connects a chat bot to an agent.
- A web or mobile UI that streams assistant responses in real time.
- A script or orchestrator that embeds meka as a sub-agent backend.
- Any cross-language client that speaks HTTP+JSON.

All three entry points (`meka`, `meka acp`, `meka serve`) drive the same agent core: same tools, same providers, same session persistence. The HTTP API is a transport layer on top.

## Starting the server

```bash
meka serve
```

The server reads the `[serve]` section from your `config.toml` (see [Configuration](#configuration) below). At minimum you need a bind address and at least one bearer token:

```toml
[serve]
bind = "127.0.0.1:8080"

[[serve.tokens]]
token = "${MEKA_API_TOKEN}"
scopes = ["sessions:r", "sessions:w"]
```

On startup the server logs the bind address and begins accepting requests. All endpoints (except health probes and OpenAPI docs) require a valid `Authorization: Bearer <token>` header.

> **TLS**: `meka serve` speaks plain HTTP. For production, front it with a TLS-terminating reverse proxy (nginx, Caddy, Cloudflare Tunnel).

## Quick example

### Blocking turn (simplest)

```bash
# Create a session
curl -s -X POST http://localhost:8080/v1/sessions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"cwd": "/home/user/project"}' | jq .id
# → "550e8400-e29b-41d4-a716-446655440000"

# Submit a turn
curl -s -X POST http://localhost:8080/v1/sessions/550e8400-e29b-41d4-a716-446655440000/turn \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "list the files in src/"}' | jq .final_text
# → "Here are the files in src/: ..."
```

### Streaming turn

```bash
curl -N -X POST http://localhost:8080/v1/sessions/$SESSION_ID/turn \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "explain this codebase", "stream": true}'
```

The response is a `text/event-stream` (SSE) that emits typed events as the agent works:

```
retry: 3000

event: turn.started
id: 0
data: {"turn_id":"...","session_id":"...","started_at":"2026-05-26T13:45:12Z"}

event: assistant_text.delta
id: 1
data: {"text":"This project is "}

event: assistant_text.delta
id: 2
data: {"text":"a Rust workspace that..."}

event: tool_call.executing
id: 3
data: {"id":"tu_1","name":"read_file","input":{"path":"src/main.rs"},"display_summary":"src/main.rs"}

event: tool_call.completed
id: 4
data: {"id":"tu_1","is_error":false,"content":[{"type":"text","text":"fn main() { ... }"}]}

event: turn.finished
id: 12
data: {"turn_id":"...","session_id":"...","stop_reason":"end_turn","usage":{"input_tokens":12340,"output_tokens":567,...}}
```

## Core concepts

### Sessions

A session is a persistent conversation with its own working directory, permission level, and message history. Sessions are stored in the same SQLite database as REPL and ACP sessions; they're interchangeable.

```
POST   /v1/sessions          Create a session
GET    /v1/sessions           List sessions (paginated)
GET    /v1/sessions/{id}      Get session details
PATCH  /v1/sessions/{id}      Update permission or cwd
DELETE /v1/sessions/{id}      Close and clean up
```

When creating a session, specify the working directory and optionally a permission level and capabilities:

```json
{
  "cwd": "/home/user/project",
  "permission": "write",
  "capabilities": {
    "supports_reasoning_stream": false,
    "supports_permission_prompts": true
  }
}
```

The `cwd` field is validated on create and patch:

- Must be an **absolute path** (no relative paths).
- Must **exist** on the server's filesystem.
- Must be a **directory** (not a file, device, or socket).
- Must not contain **null bytes** (which cause kernel/userspace path mismatch).

If `cwd` is omitted, it defaults to the server process's current working directory.

Sessions persist server-side until explicitly deleted or evicted by the idle timeout GC (see [Session lifecycle](#session-lifecycle)).

#### Capabilities

| Capability | Default | Meaning |
|------------|---------|---------|
| `supports_reasoning_stream` | `false` | Include `thinking.delta` events in the SSE stream |
| `supports_permission_prompts` | `true` | The client can answer a mid-turn `permission_required` event |

Set `supports_permission_prompts: false` if you stream but have no interface to show an approval
prompt on, which is the normal case for a service-to-service client streaming for liveness. Gated
tools are then denied immediately with an explanatory `notice`, the same as blocking mode. Leaving it
`true` means every gated call parks for 60 seconds and then denies anyway, which is hard to tell
apart from a hang. Better still, create the session with `permission: "write"` so nothing is gated.

#### Detecting an in-flight turn

Session responses carry `turn_in_flight`, a boolean saying whether a turn is running right now. It
exists so a client whose SSE stream dropped can tell "my turn is still running" from "my turn died"
without submitting a speculative turn and reading the `409`. A dropped stream does not cancel the
turn; the work continues server-side and resubmitting would duplicate a reply the user is about to
receive. Poll `GET /v1/sessions/{id}` and wait for it to go `false` rather than retrying blind.

### Turns

A turn is one round-trip: you send a user message, the agent processes it (potentially calling tools in a loop), and returns a result. Turns are ephemeral: they're not stored as their own resource, but the messages they produce are persisted in the session's conversation history.

```
POST   /v1/sessions/{id}/turn     Submit a turn
POST   /v1/sessions/{id}/cancel   Cancel an in-flight turn
```

**One turn at a time per session.** A second `POST /turn` while another is running returns `409 Conflict`. Across sessions, turns run fully concurrently.

The turn request body accepts four fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `message` | string | *(required)* | The user message. May be empty when `images` is non-empty |
| `images` | array | `[]` | Image attachments; see [Image attachments](#image-attachments) |
| `stream` | bool | `false` | `false` → single JSON response; `true` → SSE stream |
| `options.skill` | string \| null | `null` | When set, activates the named [skill](./skills.md) for this turn (equivalent to `/skill <name>` in the REPL) |

### Image attachments

Each entry in `images` is `{"media_type": "...", "data": "<base64>"}`. Images are inlined rather
than referenced by path because the API is a network surface: a client on another host shares no
filesystem with the agent, so it can't name a file for the agent to read.

```bash
curl -s -X POST http://localhost:8080/v1/sessions/$SESSION_ID/turn \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"what does this diagram show?\",
       \"images\": [{\"media_type\": \"image/png\", \"data\": \"$(base64 -w0 diagram.png)\"}]}"
```

- **Requires vision.** Attaching an image to a profile with `vision = false` returns `422`. Check
  `vision` on [`GET /v1/info`](#discovery-endpoints) before sending one.
- **`media_type` is a hint.** If it doesn't name a supported format, the payload's magic bytes are
  used instead, so `application/octet-stream` still works for a real image.
- **Formats.** PNG, JPEG, GIF, WebP, and BMP pass through; TIFF, ICO, HDR, EXR, TGA, PNM, QOI, DDS,
  and Farbfeld are converted to PNG. Anything else is a `422`.
- **Size.** Each image is capped at 3.75 MB decoded (~5 MB of base64). Note this interacts with
  `max_body_bytes`: the 10 MiB default comfortably fits one image, but a multi-image turn may need
  it raised.
- **Errors name the offender.** A bad attachment returns `422` with a detail like
  `` `images[1]` is invalid: unsupported image format ``.

### Messages

Read the conversation history for a session:

```
GET /v1/sessions/{id}/messages?offset=0&limit=50
```

Returns the full message list with role, content blocks, timestamps, and turn correlation IDs.

## Blocking response

With `stream: false` (the default), the server holds the connection until the turn completes, then returns a single JSON response:

```json
{
  "turn_id": "t_01J...",
  "session_id": "s_01J...",
  "stop_reason": "end_turn",
  "final_text": "Here are the files in src/: ...",
  "messages": [
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "..."}]
    }
  ],
  "tool_calls": [
    {
      "id": "tu_1",
      "name": "read_file",
      "input": {"path": "src/main.rs"},
      "display_summary": "src/main.rs",
      "is_error": false,
      "content": [{"type": "text", "text": "..."}]
    }
  ],
  "usage": {
    "input_tokens": 12340,
    "output_tokens": 567,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 8000
  },
  "notices": []
}
```

Key fields:

- **`final_text`**: concatenated assistant text. This is what most bots display to the user.
- **`messages`**: structured message array for clients that want richer rendering.
- **`tool_calls`**: every tool the agent called during the turn, with inputs and outputs.
- **`stop_reason`**: `end_turn`, `max_tokens`, or `refusal`.
- **`notices`**: provider advisories and auto-deny warnings.
- **`refusal_text`**: present only when `stop_reason` is `"refusal"`.

## Streaming response

With `stream: true`, the response is a `text/event-stream`. Every event has a monotonic `id`, a named `event` type, and a JSON `data` payload.

### Event types

#### Lifecycle

| Event | Payload | When |
|-------|---------|------|
| `turn.started` | `turn_id`, `session_id`, `started_at` | Turn begins |
| `turn.finished` | `stop_reason`, `usage`, optional `refusal_text` | Turn completed successfully |
| `turn.failed` | `error` (Problem Detail shape) | Turn failed mid-stream |
| `turn.cancelled` | `reason` (`"client"` or `"server_shutdown"`) | Turn was cancelled |

`turn.finished`, `turn.failed`, and `turn.cancelled` are **terminal**; the connection closes immediately after.

#### Content deltas

| Event | Payload | When |
|-------|---------|------|
| `assistant_text.delta` | `text` | Each chunk of assistant text |
| `thinking.delta` | `text` | Extended thinking content (only when `supports_reasoning_stream: true`) |

#### Tool execution

| Event | Payload | When |
|-------|---------|------|
| `tool_call.executing` | `id`, `name`, `input`, `display_summary` | Tool call starts |
| `tool_call.completed` | `id`, `is_error`, `content` | Tool call finishes |

#### Notices and pauses

| Event | Payload | When |
|-------|---------|------|
| `notice` | `level`, `text` | Provider advisories or warnings |
| `permission_required` | `request_id`, `tool_name`, `expires_in_seconds` | Permission approval needed (Ask mode) |

### Heartbeats

A `: keep-alive` comment is sent every 20 seconds. SSE clients ignore these automatically. The stream also sends `retry: 3000` as its first line, hinting clients to reconnect after 3 seconds on disconnect.

### SSE lag

The server buffers up to 256 events per SSE stream. If a consumer reads too slowly and falls behind, the server:

1. **Cancels the in-flight turn** to stop burning provider tokens.
2. **Emits a terminal `turn.failed`** event with error type `https://meka.so/errors/sse-lag`.
3. **Closes the stream.**

The client should retry by submitting a new turn. Use `GET /messages` to inspect what the agent completed before the lag occurred.

### Reconnection

There is no `Last-Event-ID` resumption. If the connection drops mid-turn, submit a new turn or use `GET /messages` to read what happened.

## Permission modes over HTTP

The same four [permission levels](./permissions.md) apply: `none`, `read`, `ask`, `write`. Set the level at session creation or update it via `PATCH /v1/sessions/{id}`.

### Ask mode

In `ask` mode with `stream: true`, the agent emits a `permission_required` SSE event when it needs to run a gated tool. The stream stays open while waiting. Your client resolves it by POSTing to the responses endpoint:

```
POST /v1/sessions/{id}/responses/{request_id}
Content-Type: application/json

{"outcome": "allow"}
```

Possible outcomes:

| Outcome | Effect |
|---------|--------|
| `allow` | Run this tool call |
| `deny` | Refuse this tool call |
| `allow_always` | Allow this and all future calls to this tool (session-scoped) |
| `deny_always` | Deny this and all future calls to this tool (session-scoped) |

If no response arrives within 60 seconds, the permission defaults to `deny`.

### Ask mode with blocking turns

When `stream: false` and the session is in `ask` mode, there is no SSE channel for permission prompts. The agent runs the turn with tool permissions **auto-denied**; each denied tool appends a `notice` to the response explaining what happened and suggesting `permission: "write"` or `stream: true`.

**MCP elicitations** (interactive form prompts from MCP servers) are always auto-declined over HTTP; there is no channel for interactive input. A `notice` event is emitted when this happens.

**Recommendation:** non-interactive callers (bots, bridges, scripts) should create sessions with `permission: "read"` or `permission: "write"` so auto-deny never triggers. Use `stream: true` if you need approval flow.

## Authentication

Every request (except health probes and `/v1/openapi.json`) requires `Authorization: Bearer <token>`.

### Scopes

Each token carries a set of scopes that control what it can access:

| Scope | Permits |
|-------|---------|
| `sessions:r` | List sessions, get session details, read messages |
| `sessions:w` | Create, modify, delete sessions; submit and cancel turns; respond to permission prompts |
| `skills:r` | List installed skills |
| `mcp:r` | List MCP server status |

Discovery endpoints (`/v1/info`, `/v1/skills`, `/v1/mcp`) accept any token with at least one read scope.

### Token configuration

Tokens are configured under `[[serve.tokens]]` in your config. Three forms are supported:

```toml
# Inline plaintext, development only (a startup warning is logged)
[[serve.tokens]]
token = "sk_dev_test123"
scopes = ["sessions:r", "sessions:w"]

# Environment variable substitution, recommended for CI/containers
[[serve.tokens]]
token = "${MEKA_BRIDGE_TOKEN}"
description = "telegram bridge"
scopes = ["sessions:r", "sessions:w"]

# File-based, recommended for production (chmod 0600)
[[serve.tokens]]
token_file = "/etc/meka/bridge.token"
description = "telegram bridge"
scopes = ["sessions:r", "sessions:w"]
```

Token comparison uses constant-time equality to prevent timing side-channel attacks. Tokens never appear in logs; only a truncated SHA-256 fingerprint is used for diagnostics.

## Idempotency

Blocking turn submissions (`stream: false`) support Stripe-style idempotency via the `Idempotency-Key` header:

```bash
curl -X POST http://localhost:8080/v1/sessions/$ID/turn \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: 7f8a9b0c-1234-5678-abcd-ef0123456789" \
  -d '{"message": "deploy to staging"}'
```

If the same key is replayed, the server returns the cached response. If the same key is sent with a different body, it returns `409 Conflict`. Keys are scoped per-token and expire after 24 hours.

Idempotency keys are **ignored for streaming responses**; streaming clients should reconnect by submitting a new turn.

## Error handling

All HTTP error responses use [RFC 9457 Problem Details](https://www.rfc-editor.org/rfc/rfc9457) with `Content-Type: application/problem+json`:

```json
{
  "type": "https://meka.so/errors/session-not-found",
  "title": "Session not found",
  "status": 404,
  "detail": "Session 's_xyz' does not exist.",
  "instance": "/v1/sessions/s_xyz/turn"
}
```

The `type` URI is the stable, machine-readable error code. Route error handling on `type`, not on `status` or `detail`.

> **Error detail redaction:** Validation errors (`422`) return a generic detail message (e.g. `"invalid session creation request body"`) rather than echoing internal field names or parser diagnostics. Consult the [OpenAPI spec](#endpoint-reference) for the expected request schema.

### Error types

| Type | Status | Meaning |
|------|--------|---------|
| `/errors/auth` | 401 | Missing or invalid bearer token |
| `/errors/auth-scope` | 403 | Token lacks the required scope |
| `/errors/session-not-found` | 404 | Unknown session ID |
| `/errors/session-locked` | 409 | Another meka process holds the session's DB lock (e.g. two `meka serve` instances sharing one DB); wait or restart the other process |
| `/errors/turn-in-flight` | 409 | A turn is already running on this session within *this* process; cancel it via `POST /cancel` first |
| `/errors/turn-cancelled` | 409 | Turn was cancelled |
| `/errors/request-not-found` | 404 | Unknown or expired `request_id` |
| `/errors/idempotency` | 409/429 | Key conflict (body mismatch: 409; cache cap: 429) |
| `/errors/invalid-body` | 422 | Request body validation failed |
| `/errors/payload-too-large` | 413 | Body exceeds `max_body_bytes` |
| `/errors/concurrency-limit` | 429 | Process-wide turn limit reached (`Retry-After` header included) |
| `/errors/sse-lag` | 500 | SSE consumer fell behind; stream terminated (see [SSE lag](#sse-lag)) |
| `/errors/provider` | 502 | Upstream provider call failed |
| `/errors/internal` | 500 | Unhandled server error |

Streaming turns that fail mid-stream emit a `turn.failed` SSE event with the same error shape, then close the connection.

## Discovery endpoints

These endpoints help clients inspect the server's capabilities at runtime.

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /v1/health/live` | None | Liveness probe: 200 if the process is up |
| `GET /v1/health/ready` | None | Readiness probe: 200 if provider, DB, and MCP servers are healthy. Returns `status`, `session_db`, `provider_configured`, and `mcp_servers_healthy` (boolean, no server names). |
| `GET /v1/info` | Any read scope | Server version, model, capabilities. `vision` reports whether `POST /turn` accepts [image attachments](#image-attachments) |
| `GET /v1/skills` | Any read scope | Installed skills |
| `GET /v1/mcp` | Any read scope | MCP server connection status |
| `GET /v1/openapi.json` | None | OpenAPI 3 spec |
| `GET /v1/docs` | None | Swagger UI |

## Session lifecycle

### Idle timeout and GC

A background garbage collector scans in-memory sessions and evicts those that have been idle longer than `idle_timeout`:

```toml
[serve]
idle_timeout = "24h"
gc_scan_interval = "5m"
```

Eviction drops the in-memory state (agent runtime, conversation buffer, cancellation tokens) but **keeps the SQLite row**. A later request with the same session ID transparently re-attaches and continues the conversation.

To also remove the DB row on eviction:

```toml
[serve]
delete_on_idle = true
```

Sessions with an in-flight turn are never evicted.

### Graceful shutdown

`meka serve` handles `SIGTERM` / `SIGINT` with a controlled drain:

1. Stop accepting new connections.
2. Cancel all in-flight turns (same mechanism as `POST /cancel`).
3. Emit `turn.cancelled` with `reason: "server_shutdown"` on open SSE streams.
4. Wait up to `shutdown_drain_timeout` for tasks to flush.
5. Exit.

```toml
[serve]
shutdown_drain_timeout = "30s"
```

## Concurrency

- **Per session:** one turn at a time. A second `POST /turn` returns 409.
- **Across sessions:** fully concurrent. Multiple sessions can run turns in parallel.
- **Process-wide cap (optional):** set `max_concurrent_turns` to limit total in-flight turns. Exceeding the cap returns 429 with a `Retry-After` header.

## Configuration

All settings live under `[serve]` in your `config.toml`. See the [`[serve]` section](../configuration/config-file.md#serve) of the config file reference for the full field list.

Minimal example:

```toml
[serve]
bind = "127.0.0.1:8080"

[[serve.tokens]]
token = "${MEKA_API_TOKEN}"
scopes = ["sessions:r", "sessions:w"]
```

Full example:

```toml
[serve]
bind = "0.0.0.0:8080"
max_body_bytes = 10485760           # 10 MiB (default)
max_concurrent_turns = 20
idle_timeout = "24h"
gc_scan_interval = "5m"
delete_on_idle = false
shutdown_drain_timeout = "30s"

# Bridge token, env var substitution
[[serve.tokens]]
token = "${BRIDGE_TOKEN}"
description = "telegram bridge"
scopes = ["sessions:r", "sessions:w"]

# Admin token, file-based
[[serve.tokens]]
token_file = "/etc/meka/admin.token"
description = "operator debugging"
scopes = ["sessions:r", "sessions:w", "mcp:r", "skills:r"]
```

## Client recipes

### Telegram bridge (Python)

```python
import httpx

MEKA_URL = "http://localhost:8080"
MEKA_TOKEN = os.environ["MEKA_TOKEN"]

async def handle_message(chat_id: str, text: str):
    session_id = await get_or_create_session(chat_id)

    resp = await httpx.AsyncClient().post(
        f"{MEKA_URL}/v1/sessions/{session_id}/turn",
        headers={"Authorization": f"Bearer {MEKA_TOKEN}"},
        json={"message": text},
        timeout=httpx.Timeout(600.0, connect=5.0),
    )
    resp.raise_for_status()
    return resp.json()["final_text"]
```

### Web UI (TypeScript, streaming)

```typescript
const resp = await fetch(`${MEKA_URL}/v1/sessions/${sessionId}/turn`, {
  method: "POST",
  headers: {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({ message: input, stream: true }),
});

const reader = resp.body!.getReader();
const decoder = new TextDecoder();
// ... parse SSE events from the stream
```

### Shell script

```bash
#!/usr/bin/env bash
set -euo pipefail

TOKEN="sk_..."
BASE="http://localhost:8080"

# Create a session
SESSION=$(curl -sf -X POST "$BASE/v1/sessions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"cwd\": \"$(pwd)\"}" | jq -r .id)

# Run a turn
RESULT=$(curl -sf -X POST "$BASE/v1/sessions/$SESSION/turn" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "summarize this project"}')

echo "$RESULT" | jq .final_text

# Clean up
curl -sf -X DELETE "$BASE/v1/sessions/$SESSION" \
  -H "Authorization: Bearer $TOKEN"
```

## Reverse proxy setup

For production deployments behind nginx:

```nginx
location /v1/ {
    proxy_pass http://127.0.0.1:8080;
    proxy_buffering off;
    proxy_cache off;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_read_timeout 600s;
}
```

Key points:
- **Disable buffering**: SSE events must not be buffered.
- **Extend read timeout**: turns can take minutes; the default 60s is too short.
- **Do not compress**: gzip/brotli on SSE responses swallow events. Exclude the `/turn` route from compression middleware.

## Endpoint reference

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/v1/health/live` | None | Liveness probe |
| GET | `/v1/health/ready` | None | Readiness probe |
| GET | `/v1/info` | read | Server info |
| GET | `/v1/skills` | read | Installed skills |
| GET | `/v1/mcp` | read | MCP server status |
| POST | `/v1/sessions` | `sessions:w` | Create session |
| GET | `/v1/sessions` | `sessions:r` | List sessions |
| GET | `/v1/sessions/{id}` | `sessions:r` | Get session |
| PATCH | `/v1/sessions/{id}` | `sessions:w` | Update session |
| DELETE | `/v1/sessions/{id}` | `sessions:w` | Delete session |
| GET | `/v1/sessions/{id}/messages` | `sessions:r` | List messages |
| POST | `/v1/sessions/{id}/turn` | `sessions:w` | Submit turn |
| POST | `/v1/sessions/{id}/cancel` | `sessions:w` | Cancel turn |
| POST | `/v1/sessions/{id}/responses/{request_id}` | `sessions:w` | Resolve permission prompt |
| GET | `/v1/openapi.json` | None | OpenAPI spec |
| GET | `/v1/docs` | None | Swagger UI |

For full request/response schemas, see `/v1/openapi.json` on a running server, or browse it interactively at `/v1/docs` (Swagger UI). Both endpoints are unauthenticated so CI pipelines and code generators can fetch the spec without a token.

### Exporting the spec

Save a local copy for offline use or code generation:

```bash
curl -s http://localhost:8080/v1/openapi.json -o openapi.json
```

### Code generation

Generate a typed client from the exported spec:

```bash
# Python (openapi-python-client)
openapi-python-client generate --path openapi.json

# TypeScript (openapi-typescript)
npx openapi-typescript openapi.json -o src/api.d.ts

# Go (oapi-codegen)
oapi-codegen -package api openapi.json > api/api.gen.go

# Rust (progenitor)
cargo progenitor-client openapi.json
```

### Import into tools

- **Postman / Insomnia:** Import → URL → `http://localhost:8080/v1/openapi.json`
- **Bruno:** Create collection from OpenAPI → paste the URL or a saved file.
- **Swagger Editor:** File → Import URL → `http://localhost:8080/v1/openapi.json`
