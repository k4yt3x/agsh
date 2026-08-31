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

Two flags are refused rather than ignored: `-c` and `-r`. Both name one run's session, and the server creates one per `POST /v1/sessions`, each naming its own provider profile. Pass `provider` on the create request instead. `--provider` is accepted, since it selects which configured profile a session gets when it names none, which is a property of the server rather than of one session.

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

event: tool_call.composing
id: 3
data: {"id":"tu_1","name":"read_file"}

event: tool_call.executing
id: 4
data: {"id":"tu_1","name":"read_file","input":{"path":"src/main.rs"},"display_summary":"src/main.rs"}

event: tool_call.completed
id: 5
data: {"id":"tu_1","is_error":false,"content":[{"type":"text","text":"fn main() { ... }"}]}

event: turn.finished
id: 12
data: {"turn_id":"...","session_id":"...","stop_reason":"end_turn","usage":{"input_tokens":12340,"output_tokens":567,...}}
```

## Core concepts

### Sessions

A session is a persistent conversation with its own working directory, permission level, and message history. Sessions are stored in the same SQLite database as REPL and ACP sessions; they're interchangeable.

```
POST   /v1/sessions           Create a session
GET    /v1/sessions           List sessions (paginated)
GET    /v1/sessions/{id}      Get session details
PATCH  /v1/sessions/{id}      Update permission, cwd or provider
DELETE /v1/sessions/{id}      Close and clean up
POST   /v1/sessions/{id}/fork Branch a copy off a session
```

When creating a session, specify the working directory and optionally a permission level, a provider
profile, and capabilities:

```json
{
  "cwd": "/home/user/project",
  "permission": "workspace",
  "provider": "work",
  "capabilities": {
    "supports_reasoning_stream": false,
    "supports_permission_prompts": true
  }
}
```

`provider` names a profile in the server's `config.toml`; `GET /v1/providers` lists them, and a name
that is not configured is a `422`. Omitted, it is the server's own default profile. The session keeps
it for the rest of its life and every session response echoes it back as `provider`, so a client can
confirm which account a session bills.

To move a live session onto another profile, `PATCH /v1/sessions/{id}` with `{"provider": "other"}`.
That rewrites the session's row, so it holds for a resume from any surface rather than for this
request. Switching mid-conversation is allowed and is your call: a thinking block is tagged with the
provider that produced it and is not replayed to a different one, so from the next turn the model no
longer sees the reasoning recorded under the old provider. Like the other `PATCH` fields, it is a
`409` when a turn is already in flight; cancel first. (One admitted between the check and the agent
swap makes the swap wait for that turn rather than fail, so the request can take as long as the turn
does. The row has already moved by then, and the agent follows when the turn ends.)

A `PATCH` naming a provider moves the session to that profile, and the profile is the whole story:
the model, the endpoint and every model-tied setting come from it, so there is nothing else on the
row to reconcile.

If you run more than one `meka` on the same store, send the `PATCH` to whichever process has the
session. A body naming only a provider is the one `PATCH` that works on a session this server has
not loaded, and it takes the session lock to do it, so a session another process is running answers
`409` `session-locked` rather than moving a row that process would go on ignoring. Only the host
holding a session may change what it runs on.

A body naming **only** `provider` is also the rescue for a session whose profile has left
`config.toml`: it moves the row without building an agent, so it works on a session that cannot
currently run. Adding `permission` or `cwd` to the same body loses that, because those need a loaded
session and loading one is exactly what fails; send the provider on its own first.

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
apart from a hang. Better still, create the session with `permission: "workspace"` so nothing is gated.

#### Forking a session

`POST /v1/sessions/{id}/fork` copies a session's conversation into a new session and returns it with
`201` and the usual session body. The copy starts with the source's full history and is immediately
usable; the source is left untouched, and does not have to be in memory, so a GC-evicted session
forks as well as a live one.

The body is optional and inherits everything by default. The only field is `cwd`, matching ACP's
`session/fork`, which likewise carries a workspace but no permission or capability fields:

```json
{ "cwd": "/home/user/other-project" }
```

Permission, capabilities and the provider profile are inherited and remain changeable afterwards via
`PATCH /v1/sessions/{id}`. Sub-agent child transcripts are not copied, and a fork of an ordinary
session records no link back to its source.

A sub-agent's own id is refused with `422`: the copy would keep that worker's parent and spawn terms,
so it is a sibling under the same parent rather than a session this endpoint could hand back. See
[Forking a Session](./sessions.md#forking-a-session) for the full semantics.

#### Sub-agent sessions cannot be driven through this API

`GET /v1/sessions?include_children=true` lists the sessions an `agent_spawn` created. Those ids are
readable through every endpoint on this page -- `/messages`, `/context`, `/export` -- and
drivable through none of them: `POST /v1/sessions/{id}/turn` answers `422` with
`/errors/session-not-drivable`, as do `/compact`, `/responses/{request_id}`, `/fork`, `/schedule`,
and `PATCH /v1/sessions/{id}`. A worker
runs under the tools, permission ceiling and provider profile its spawn call set, which live in its
spawn record and which only its parent can reconstruct, so the conversation is continued with the
`agent_followup` tool from the parent rather than over HTTP.

Two exceptions, both of which change a transcript without running anything on it. Teardown stays
open: `DELETE /v1/sessions/{id}` discards a worker and `DELETE /v1/sessions/{id}/tasks/{task_id}`
stops one of its background tasks, and the parent's own `agent_delete` does the same thing. So does
`POST /v1/sessions/{id}/rewind`, which truncates the event log the same caller can already read in
full through `/export`, and which `meka session rewind` has always allowed on a worker. The line is
whether the model runs: `/compact` is refused because compaction is a turn.

#### Importing an archive

`POST /v1/sessions/import` recreates a session tree from a `meka session export` archive under fresh
ids, on the same terms as the CLI's `meka session import`. An archive naming no provider profile
takes the server's default, the same one `POST /v1/sessions` applies to a body with no `provider`; a
long-lived host always has one, since it refuses to start without it.

One limit is the server's alone: an archive holding more than **1000** sessions is refused with a
`422` whose detail names the count and the cap, and points at `meka session import`. The whole tree
is written in one transaction on the process's single database connection, so a larger one would
hold every other in-flight request behind it. A one-shot `meka session import` restoring its own
backup has nothing to contend with and so carries no cap; it is the way to restore a tree this
large.

Everything else about the archive is honoured as the CLI honours it; see
[Exporting a Session](./sessions.md#exporting-a-session).

#### Detecting an in-flight turn

Session responses carry `turn_in_flight`, a boolean saying whether a turn is running right now. It
exists so a client whose SSE stream dropped can tell "my turn is still running" from "my turn died"
without submitting a speculative turn and reading the `409`. A dropped stream does not cancel the
turn; the work continues server-side and resubmitting would duplicate a reply the user is about to
receive. Poll `GET /v1/sessions/{id}` and wait for it to go `false` rather than retrying blind.

The same holds for a **blocking** turn whose client gives up: a request timeout on your side does
not stop the turn. It runs to completion, persists its messages, and fires its webhook; you just
never see the response body. Read the reply from `GET /v1/sessions/{id}/messages`. This is why a
client timeout shorter than your longest turn is safe, and why retrying on one duplicates work
rather than recovering it.

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

- **Requires vision.** Attaching an image to a session whose profile has `vision = false` returns
  `422`. The check is per session, from the profile that session recorded, so a session created with
  `provider` or moved by a `PATCH` follows that profile rather than the server default. `vision` on
  [`GET /v1/info`](#discovery-endpoints) reports the *default* profile's flag, which answers for a
  session created without naming one.
- **`media_type` is a hint.** If it doesn't name a supported format, the payload's magic bytes are
  used instead, so `application/octet-stream` still works for a real image.
- **Formats.** PNG, JPEG, GIF, WebP, and BMP pass through; TIFF, ICO, HDR, EXR, TGA, PNM, QOI, DDS,
  and Farbfeld are converted to PNG. Anything else is a `422`.
- **Size.** Each image is capped at 3.75 MB decoded (~5 MB of base64). Note this interacts with
  `max_body_bytes`: the 10 MiB default comfortably fits one image, but a multi-image turn may need
  it raised.
- **Errors name the offender.** A bad attachment returns `422` with a detail like
  `` `images[1]` is invalid: unsupported image format ``.

### Detecting a rewritten history

`GET /messages` returns the *materialised* view: what the model can currently see. Three things rewrite it rather than appending to it — compaction, `POST /rewind`, and a mid-turn repair of a malformed request — and after any of them your copy is no longer a prefix of the server's.

Two signals cover this:

- **`revision`** on the response increments on every rewrite. If it changed since your last poll, re-fetch rather than diff. This is the one to key on, because it covers all three causes.
- **`compaction`** on a message identifies a summary and says how many messages it replaced and which compaction it was. Only compaction leaves a message behind to carry it; a rewind removes messages with nothing in their place, which is why `revision` exists.

`total` alone is not enough: a shrinking `total` is indistinguishable from the server losing your conversation.

Note that neither `GET /context` nor `GET /v1/sessions/{id}/tools` will load an evicted session. Reading is not permitted to take the session's cross-process lock, which a write would hold for `idle_timeout`. `/context` answers from the database with the live counters omitted; `/tools` returns 409, since a catalogue needs a loaded session.

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
| `tool_call.composing` | `id`, `name` | The model started writing the call's arguments |
| `tool_call.executing` | `id`, `name`, `input`, `display_summary` | Tool call starts |
| `tool_call.completed` | `id`, `is_error`, `content` | Tool call finishes |

The arguments are written between `tool_call.composing` and `tool_call.executing` on the same `id`, which makes that interval the only thing on the stream that separates the agent *writing a message* from the agent doing anything else. Assistant text is usually narration around a call rather than the reply itself, and by `tool_call.executing` the arguments are already finished. A client drawing a typing indicator for a tool like an MCP `send_message` raises it on the first and drops it on the second. The payload is the id and the name because nothing else has streamed yet: which conversation a message is for is not known until `tool_call.executing`.

Three limits. The event exists only when meka streams from its provider, so a server started with `--no-stream` receives each call whole and emits `tool_call.executing` with nothing before it. The pairing is not guaranteed, because a turn that fails or is cancelled mid-call emits `tool_call.composing` with nothing after it, so close per-`id` state on the terminal event as well. And the interval is only wide on backends that stream a call as it is written (`anthropic-messages`, `claude-subscription`, `openai-responses`, `chatgpt-subscription`); `openai-chat-completions` resolves each call's name and arguments together when the stream ends, so there the two events arrive back to back.

#### Notices and pauses

| Event | Payload | When |
|-------|---------|------|
| `notice` | `level`, `text` | Provider advisories or warnings |
| `permission_required` | `request_id`, `tool_name`, `expires_in_seconds` | Permission approval needed (Ask mode) |

#### Context

| Event | Payload | When |
|-------|---------|------|
| `context.compacted` | `source`, `replaced_count`, `generation` | The conversation was summarised and the window replaced |

`context.compacted` is the one event on this stream that is not additive. Everything else appends, so a client that misses one still holds a prefix of the truth; a compaction *removes* messages the client has already rendered. `source` is `checkpoint`, `checkpoint_text`, or `summarizer` (they differ in fidelity, not just mechanism), `replaced_count` is how many messages the boundary removed from the view (the whole pre-compaction window, including the tail compaction re-appends verbatim), and `generation` counts compactions from 1.

The same information appears on `GET /messages`: the summary message carries a `compaction` object with `replaced_count` and `generation`, and every other message omits the field. Without it a polling client sees `total` shrink with no explanation, which is indistinguishable from the server losing the conversation.

### Heartbeats

A `: keep-alive` comment is sent every 20 seconds. SSE clients ignore these automatically. The stream also sends `retry: 3000` as its first line, hinting clients to reconnect after 3 seconds on disconnect.

### SSE lag

The server buffers up to 256 events per SSE stream. If a consumer reads too slowly and falls behind, the server closes that consumer's stream, and what it sends first depends on whether anyone else was still reading:

- **Nobody else was reading.** The turn is cancelled to stop burning provider tokens, and the stream ends with a terminal `turn.failed` carrying error type `https://meka.so/errors/sse-lag`. Retry by submitting a new turn.
- **Another consumer was keeping up.** The turn keeps running for them, so nothing has failed. The lagging stream ends with a `notice` explaining the drop and closes. **Re-attach with `Last-Event-ID`** rather than retrying: the turn is still in flight, so a new turn would be refused with `409 turn-in-flight`, and re-attaching recovers the dropped events instead of redoing the work.

Turn events are broadcast, so a re-attached client or a second consumer counts as a separate reader. Use `GET /messages` to inspect what the agent completed either way.

### Reconnection

`GET /v1/sessions/{id}/stream` rejoins the current turn. Send the last id you received as a `Last-Event-ID` header (browser `EventSource` does this automatically) or as a `?last_event_id=` query parameter, and the server replays what you missed before following the live stream.

```bash
curl -N -H "Authorization: Bearer $TOKEN" \
     -H "Last-Event-ID: 42" \
     "http://localhost:8080/v1/sessions/$SESSION/stream"
```

The stream opens with a `turn.started` carrying `"resumed": true` and the `turn_id` you actually rejoined, which is the only way to tell "my stream resumed" from "a newer turn started while I was away". That opening event is synthesised by the reconnect rather than replayed, so unlike the original it carries no `started_at` and no `id:` — a resumed stream must not move your stored resume position backwards before the replay has run. Every event after it is the real thing, ids included. The stream always terminates: if the turn has already finished, the buffered tail and its terminal event are delivered and the connection closes.

Three limits, all deliberate:

- **The replay buffer is bounded** by `[serve] stream_replay_events` (default 256). If your `Last-Event-ID` is older than the oldest retained event, you get a `notice` saying the replay has a hole rather than a transcript that silently skips. Read `GET /messages` to fill it.
- **Only the most recent turn is retained.** Reconnecting after a newer turn started gives you that turn.
- **A disconnected turn is not cancelled immediately.** It keeps running for `[serve] stream_reattach_grace` (default 30s) waiting for you to come back; after that the agent loop stops, since nobody is listening. Set `"0s"` to restore the older behaviour where a dropped stream cancels the turn at once, which spends fewer provider tokens on abandoned work.

## Webhooks

`meka serve` can POST to configured endpoints when something happens that no client is necessarily waiting on: a scheduled job firing overnight, a background task finishing long after the turn that started it.

```toml
[[serve.webhooks]]
url = "https://bridge.example/meka-hook"
secret = "${MEKA_WEBHOOK_SECRET}"     # or secret_file = "/etc/meka/hook.secret"
events = ["turn.finished", "turn.failed", "task.finished", "schedule.fired"]
timeout = "10s"                        # per attempt, default 10s
max_retries = 3                        # after the first attempt, default 3
```

`events` is required and every name must be recognised: an endpoint whose only subscription is a typo would be silently never called, so an unknown event is a startup error rather than a warning.

The four do not overlap. `turn.finished` and `turn.failed` cover turns submitted through `POST /turn`; a turn the *server* started fires `schedule.fired` (which carries its own `status`) or `task.finished` instead, so one occurrence never produces two deliveries. A client that wants to know about everything the agent did should subscribe to all four.

### Payloads

Every delivery carries `delivery_id`, `event`, `timestamp`, and event-specific identifiers:

```json
{
  "delivery_id": "6c1f...",
  "event": "schedule.fired",
  "timestamp": "2026-02-01T03:00:00Z",
  "job_id": "9f2c...",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed"
}
```

**Payloads never carry message content.** A webhook URL is a string in a config file: it can be mistyped, it can outlive whatever owned it, and anything that learns it can reach it. So a delivery tells you *what happened to which session*, and you fetch the conversation with your own bearer token over the API you already authenticate against. A compromised endpoint learns that a session was active, not what was said in it.

### Verifying a delivery

When `secret` is set, each request carries `X-Meka-Signature: sha256=<hex>`, an HMAC-SHA256 over `<timestamp>.<body>` keyed with the secret. The timestamp is *inside* the signed material, so a captured delivery cannot be replayed forever: reject anything whose `X-Meka-Timestamp` is too old and the window closes.

Each **attempt** carries its own timestamp and signature. A retry can land minutes after the first attempt, so re-sending the original stamp would have it rejected by that very window. Deduplicate on `X-Meka-Delivery`, which stays constant across a delivery's attempts.

```python
import hmac, hashlib

def verify(secret: str, timestamp: str, body: bytes, signature: str) -> bool:
    expected = "sha256=" + hmac.new(
        secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
```

Deliveries also carry `X-Meka-Event`, `X-Meka-Delivery` (unique per delivery, for deduplicating retries), and `X-Meka-Timestamp`.

`X-Meka-Timestamp` and the body's `timestamp` field differ on a retry, deliberately. The header is when *this attempt* was sent, re-stamped each time, because that is what your replay window is checking; a retry carrying the original time would be rejected as stale by the very check the header exists for. The body's field is when the *event* happened and stays fixed across attempts, so ordering and deduplication see one event rather than several.

Omitting `secret` is allowed for loopback receivers and logs a startup warning; no signature header is sent, rather than one computed over an empty key.

### Delivery semantics

Deliveries are notifications, not a durable queue. They are not persisted, not retried across a restart, and outstanding attempts are abandoned when the process exits: a delivery in flight during a `SIGTERM` is lost. That is the trade for never blocking the work that triggered it. Anything you cannot afford to miss should be reconciled by polling (`GET /v1/schedule`, `GET /v1/sessions/{id}/tasks`), with the webhook as the fast path rather than the only one.

Delivery is fire-and-forget on a detached task, so a slow or dead receiver never wedges the scheduler behind it. A `5xx` or a transport error is retried with exponential backoff (1s, 2s, 4s, capped at 30s) up to `max_retries`. A `4xx` is not, since retrying cannot fix a request the receiver considers malformed, with two exceptions: `429 Too Many Requests` and `408 Request Timeout` say "not now" rather than "not ever" and are retried like a `5xx`. That matters because several jobs sharing a cron minute deliver as a burst, which is exactly when a receiver rate-limits. `Retry-After` is not honoured; the backoff above is used regardless. After the last attempt meka logs one `warn` and gives up. Turn cancellations are not delivered: the client that cancelled already knows.

## Permission modes over HTTP

The same five [permission levels](./permissions.md) apply: `none`, `read`, `workspace`, `ask`, `unrestricted`. Set the level at session creation or update it via `PATCH /v1/sessions/{id}`.

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

When `stream: false` and the session is in `ask` mode, there is no SSE channel for permission prompts. The agent runs the turn with tool permissions **auto-denied**; each denied tool appends a `notice` to the response explaining what happened and suggesting `permission: "workspace"` or `stream: true`.

**MCP elicitations** (interactive form prompts from MCP servers) are always auto-declined over HTTP; there is no channel for interactive input. A `notice` event is emitted when this happens.

**Recommendation:** non-interactive callers (bots, bridges, scripts) should create sessions with `permission: "read"` or `permission: "workspace"` so auto-deny never triggers. Use `stream: true` if you need approval flow.

## Authentication

Every request requires `Authorization: Bearer <token>`, except the two health probes and, when `[serve].docs` is enabled, `/v1/openapi.json` and `/v1/docs`. Both of those are off by default, so on a default deployment they answer 404 rather than serving anything unauthenticated.

### Scopes

Each token carries a set of scopes that control what it can access:

| Scope | Permits |
|-------|---------|
| `sessions:r` | List sessions, get details, read messages, context occupancy, export, tools, background tasks, re-attach a stream |
| `sessions:w` | Create, modify, delete sessions; submit and cancel turns; compact, rewind, import; respond to permission prompts; cancel background tasks |
| `skills:r` | Read installed skills, including bodies |
| `skills:w` | Create, update, delete skills |
| `memory:r` | Read the memory store |
| `memory:w` | Create, update, delete memories |
| `schedule:r` | List scheduled jobs. `GET /v1/schedule` is server-wide and returns each job's full `prompt`, so this reads instruction text and not just schedules. A gate's `check` is withheld unless the token also holds `sessions:r` |
| `schedule:w` | Create and cancel scheduled jobs. **A job's `prompt` runs a full turn with tools**, so this is deferred turn execution, not just bookkeeping. A job's optional `gate` runs a shell command or a read-only tool call and additionally requires `sessions:w` (see below) |
| `mcp:r` | Read MCP server status and advertised tools |
| `mcp:w` | Reconnect an MCP server |

Discovery endpoints (`/v1/info`, `/v1/skills`, `/v1/mcp`, `/v1/providers`) accept any token with at least one read scope. Two deliberately do not: `GET /v1/skills/{name}` needs `skills:r` and `GET /v1/instructions` needs `sessions:r`, because both return instruction *text* rather than a listing.

Scopes are flat: `memory:r` does not imply `memory:w`, and neither implies the other. Operations *on a conversation* stay under `sessions:*`, because the thing being read or changed is one session. The process-wide stores carry their own scopes so a bridge token that runs turns cannot also empty the memory store or plant an unattended scheduled job.

An unrecognised scope logs a warning at startup and grants nothing, so a typo like `sessions:write` is visible rather than silently inert.

> **Note:** `[skills] agent_managed` and `[memory] access` govern what the *model* may do on its own initiative. They do not gate these endpoints. A token is the operator acting remotely, equivalent to running `meka skill add` in a shell, so a `skills:w` token writes skills even when `agent_managed = false`.

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

If the same key is replayed, the server returns the cached response. If the same key is sent with a different body, it returns `409 Conflict`.

Keys are scoped per-token **and per-session**, and expire after 24 hours. The session is part of the scope because an `Idempotency-Key` names *your* unit of work: sending the same key to two sessions is a reasonable thing to do, and it now runs both turns instead of answering the second with the first's transcript.

A turn that was cancelled is not cached, so the retry the cancellation invites can actually run. Neither is a 5xx, for the same reason.

The cache is bounded per token by both entry count and total bytes; a response too large to keep is not cached, and its retry re-executes.

Idempotency keys are **ignored for streaming responses**; streaming clients should reconnect by submitting a new turn.

### Which endpoints are safe to retry

`Idempotency-Key` covers blocking turns only. For everything else, know what a blind retry does before you configure one:

| Endpoint | Retry-safe | On a duplicate |
|---|---|---|
| `POST /turn` (blocking, with a key) | yes | cached response returned |
| `POST /cancel`, `DELETE /v1/sessions/{id}`, `DELETE /v1/sessions/{id}/tasks/{task_id}` | yes | already-done is the same state |
| `DELETE /v1/skills/{name}`, `/v1/memory/{name}`, `/v1/schedule/{job_id}` | yes, but | the resource is gone, so the retry answers **404**. Expected, not a failure — treat it as success if you are retrying blind |
| `PUT /v1/skills/{name}`, `PUT /v1/memory/{name}` | yes | same body writes the same skill file or memory row |
| `POST /compact` | mostly | a second compaction summarises the summary; fidelity drops, nothing is lost |
| `POST /rewind` | **no** | drops another turn. A client that retries on a connection error loses conversation |
| `POST /sessions/import` | **no** | creates a second copy of the tree under new ids |
| `POST /sessions/{id}/schedule` | **no** | creates a second job |

The three marked **no** are administrative operations meant to be driven deliberately. If your HTTP stack retries failed POSTs by default, exclude them, or check the outcome first: `POST /rewind` returns `messages_before` and `messages_after`, and `GET /messages` returns a `revision` that increments on every rewrite.

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
>
> A `502` on a turn says only that the provider rejected or failed it; it carries none of the provider's own response. An upstream refusal can contain account identifiers, rate-limit posture, and fragments of the request that triggered it, and none of that is meka's to publish to an HTTP caller. The full text goes to the server log, which is where an operator reads it. The same holds for the `503` a turn gets when a required MCP server is down: the server names travel, the connector's reason does not, since it has carried a command line and its path.
>
> The MCP endpoints under `/v1/mcp` are the deliberate exception, and the difference is who asked. A caller naming one server and asking why it will not connect is asking *for* that reason, so `GET /v1/mcp/{name}/tools` and `POST /v1/mcp/{name}/reconnect` do relay it. A caller asking for a completion is not, and does not get it.

### Error types

| Type | Status | Meaning |
|------|--------|---------|
| `/errors/auth` | 401 | Missing or invalid bearer token |
| `/errors/auth-scope` | 403 | Token lacks the required scope |
| `/errors/session-permission` | 403 | The token is fine; the *session* sits too low. Raise it with `PATCH /v1/sessions/{id}` — a better token will not help |
| `/errors/session-not-found` | 404 | Unknown session ID |
| `/errors/not-found` | 404 | Unknown skill, memory, MCP server, background task, or turn stream |
| `/errors/session-not-loaded` | 409 | The session exists but is not in memory; submit a turn to load it. Do **not** retry `POST /cancel` — there is no turn to cancel |
| `/errors/session-locked` | 409 | Another meka process holds the session's DB lock (e.g. two `meka serve` instances sharing one DB); wait or restart the other process |
| `/errors/turn-in-flight` | 409 | A turn is already running on this session within *this* process; cancel it via `POST /cancel` first |
| `/errors/turn-cancelled` | 409 | Turn was cancelled |
| `/errors/store-read-only` | 409 | The skill lives under a `[skills] extra_paths` root; meka reads those but never writes to them, so writing here would shadow the file rather than change it |
| `/errors/session-not-drivable` | 422 | The id names a sub-agent's conversation. Reading it is unaffected; continuing it means `agent_followup` from the parent, whose id the message names. **Do not retry with a corrected payload**: no body addressed at this id is accepted |
| `/errors/request-not-found` | 404 | Unknown or expired `request_id` |
| `/errors/idempotency` | 409/429 | Key conflict (body mismatch: 409; cache cap: 429) |
| `/errors/invalid-body` | 400/422 | Request body validation failed (422), or a path/query parameter the router rejected (400) |
| `/errors/payload-too-large` | 413 | Body exceeds `max_body_bytes` |
| `/errors/concurrency-limit` | 429 | Process-wide turn limit reached (`Retry-After` header included) |
| `/errors/sse-lag` | 500 | SSE consumer fell behind; stream terminated (see [SSE lag](#sse-lag)) |
| `/errors/stream-detached` | 500 | SSE-only. A re-attached stream ended with no recorded outcome because the turn's task died; read `GET /messages` for what completed |
| `/errors/provider` | 502 | Upstream provider call failed. Covers both a transient outage and a permanent one such as a bad credential or a misconfigured `base_url`, so it carries no promise that resending helps |
| `/errors/context-overflow` | 502 | The conversation exceeds the model's context window and could not be compacted further. **Do not retry unchanged**; shorten it first |
| `/errors/mcp-unavailable` | 503 | An MCP server marked `required` was not connected, so the turn was refused before reaching the provider. The `servers` extension names them; each one's reason is in the server log |
| `/errors/internal` | 500 | Unhandled server error |

Streaming turns that fail mid-stream emit a `turn.failed` SSE event with the same error shape, then close the connection.

> The two 502s are the ones worth branching on. `/errors/provider` says the upstream would not serve this turn, and is worth one backed-off resend; it does not say how many attempts meka made first, because it gives up as soon as any output has reached the stream and a cancelled turn abandons the sequence wherever it stands, so a permanent failure and an exhausted one look the same from here. `/errors/context-overflow` says the request itself no longer fits, and will not fit next time either: retrying it unchanged loops until your client gives up. Shorten the conversation with `POST /v1/sessions/{id}/compact` or send less.
>
> A `Retry-After` on a `/errors/provider` response is the upstream's own, relayed up to an hour; meka reads only the delta-seconds form, so an upstream that answers with an HTTP date sends none. Honour it in preference to your own backoff. `/errors/context-overflow` never carries one.

## Discovery endpoints

These endpoints help clients inspect the server's capabilities at runtime.

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /v1/health/live` | None | Liveness probe: 200 if the process is up |
| `GET /v1/health/ready` | None | Readiness probe: 200 if the DB is healthy, at least one provider profile is configured, and no `required` MCP server has failed. A failed *optional* server doesn't affect readiness, since it can't stop a turn either. Returns `status`, `session_db`, `provider_configured`, and `mcp_servers_healthy` (boolean, no server names). **`provider_configured` means a profile exists in `config.toml`, not that it has a usable credential**: a profile's credential is checked when a session first needs it, so a server can be ready and still answer 422 to `POST /v1/sessions`. |
| `GET /v1/providers` | Any read scope | Configured provider profiles: `name`, `type`, `model`, and `active: true` on the one a session gets when it names none. Read-only; profiles come from `config.toml` |
| `GET /v1/info` | Any read scope | Server version and permission surface. `vision` reports whether the *default* profile accepts [image attachments](#image-attachments); a session on another profile follows that one. Carries no provider or model: `GET /v1/providers` reports both per profile and marks the default with `active` |
| `GET /v1/skills` | Any read scope | Installed skills |
| `GET /v1/mcp` | Any read scope | MCP server connection status |
| `GET /v1/openapi.json` | None, and off unless `[serve].docs` is set | OpenAPI 3 spec |
| `GET /v1/docs` | None, and off unless `[serve].docs` is set | Swagger UI |

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
4. Wait up to `shutdown_drain_timeout` for every turn to finish unwinding, including scheduled
   fires, background-outcome deliveries, and turns whose client has already disconnected.
   Cancelling a turn is not the same as waiting for one: what follows the cancellation is the
   commit of the partial reply and of whatever the round already produced.
5. Exit `0`. A drain that hits the timeout instead logs a warning, abandons what is still
   running, and exits `1`, so a supervisor can tell the two apart.

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

## Scheduled jobs

`meka serve` is the durable host for [scheduled wakeups](./scheduling.md). It fires every job in the database, reviving evicted sessions on demand, so jobs keep running whether or not a client is connected and survive a restart of the server.

An agent-initiated turn has no HTTP request to respond to, so its output is persisted to the session like any other turn. Read it back with `GET /v1/sessions/{id}/messages`.

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
| GET | `/v1/info` | read | Server version and permission surface |
| GET | `/v1/skills` | read | Installed skills |
| GET | `/v1/mcp` | read | MCP server status |
| POST | `/v1/sessions` | `sessions:w` | Create session |
| GET | `/v1/sessions` | `sessions:r` | List sessions |
| GET | `/v1/sessions/{id}` | `sessions:r` | Get session |
| PATCH | `/v1/sessions/{id}` | `sessions:w` | Update session |
| DELETE | `/v1/sessions/{id}` | `sessions:w` | Delete session |
| POST | `/v1/sessions/{id}/fork` | `sessions:w` | Fork session |
| GET | `/v1/sessions/{id}/messages` | `sessions:r` | List messages |
| POST | `/v1/sessions/{id}/turn` | `sessions:w` | Submit turn |
| POST | `/v1/sessions/{id}/cancel` | `sessions:w` | Cancel turn |
| POST | `/v1/sessions/{id}/responses/{request_id}` | `sessions:w` | Resolve permission prompt |
| GET | `/v1/sessions/{id}/stream` | `sessions:r` | Re-attach to the current turn's SSE stream |
| POST | `/v1/sessions/{id}/compact` | `sessions:w` | Summarise the conversation now |
| GET | `/v1/sessions/{id}/context` | `sessions:r` | Context occupancy and cumulative usage |
| POST | `/v1/sessions/{id}/rewind` | `sessions:w` | Drop trailing turns |
| GET | `/v1/sessions/{id}/export` | `sessions:r` | Full transcript (`?format=markdown\|json`) |
| POST | `/v1/sessions/import` | `sessions:w` | Recreate a session tree from an export |
| GET | `/v1/sessions/{id}/tools` | `sessions:r` | Tool catalogue for this session (409 if not loaded) |
| GET | `/v1/sessions/{id}/tasks` | `sessions:r` | Background tasks |
| DELETE | `/v1/sessions/{id}/tasks/{task_id}` | `sessions:w` | Cancel a background task |
| GET | `/v1/schedule` | `schedule:r` | All scheduled jobs |
| GET | `/v1/sessions/{id}/schedule` | `schedule:r` | Scheduled jobs for one session |
| POST | `/v1/sessions/{id}/schedule` | `schedule:w` (+ `sessions:w` for a `gate`) | Create a scheduled job |
| DELETE | `/v1/schedule/{job_id}` | `schedule:w` | Cancel a scheduled job |
| GET | `/v1/skills/{name}` | `skills:r` | One skill, with its body |
| PUT | `/v1/skills/{name}` | `skills:w` | Create or update a skill |
| DELETE | `/v1/skills/{name}` | `skills:w` | Delete a skill |
| GET | `/v1/memory` | `memory:r` | Memory index |
| GET | `/v1/memory/{name}` | `memory:r` | One memory, with its body |
| PUT | `/v1/memory/{name}` | `memory:w` | Create or update a memory |
| DELETE | `/v1/memory/{name}` | `memory:w` | Delete a memory |
| GET | `/v1/mcp/{name}/tools` | `mcp:r` | Tools one MCP server advertises |
| POST | `/v1/mcp/{name}/reconnect` | `mcp:w` | Reconnect an MCP server |
| GET | `/v1/instructions` | `sessions:r` | Resolved system instructions |
| GET | `/v1/providers` | read | Configured provider profiles |
| GET | `/v1/openapi.json` | None, and off unless `[serve].docs` is set | OpenAPI spec |
| GET | `/v1/docs` | None, and off unless `[serve].docs` is set | Swagger UI |

`GET /v1/sessions` takes `include_children=true` to list sub-agent sessions alongside top-level ones, and `cwd=<path>` to filter by working directory. Every session record carries `parent_id`, which is what reconnects a spawned worker to the session that dispatched it.

A memory record carries both `updated_at` (when the row last changed) and `recorded_at` (when the memory was made, stamped once at creation), plus its `tags`. The two timestamps are deliberately separate: a description edit moves `updated_at` without the note saying anything new, and it is `recorded_at` that the model is shown as an age. `PUT /v1/memory/{name}` accepts `tags` with the same omit-to-keep rule as `body` — omit to leave an existing memory's labels alone, send `[]` to clear them.

`GET /v1/memory/{name}` answers **404** for a name that is not stored, with no 422 case: a memory is a row, so there is no file to be present but unparseable. Reading through this endpoint deliberately does *not* increment the memory's read count — an operator is not the agent recalling anything, and the count feeds search ranking.

Descriptions and bodies are returned **exactly as stored**, not as they are rendered into a model's context: this endpoint is a backup and inspection door, like `meka memory export`, and stripping characters out of a note on the way through would make a restore lossy. JSON escaping keeps that safe in transit, but a client that decodes and prints the text to a terminal should neutralise it, as meka does at its own render boundaries.

These four endpoints are **not** gated by `[memory] enabled`. That switch decides whether an agent keeps memories; a token is the operator, so it reaches a store that already exists exactly as `meka memory list` does in a shell.

A scheduled job's optional `gate` is the sharpest grant on this API, and how sharp depends on what it checks. It requires `sessions:w` in addition to `schedule:w` either way.

A **shell** gate (`"check": {"command": "…"}`) runs through `sh -c` as the user running `meka serve`, on a timer, *before* the turn and independently of it, so it needs no working provider and no model to execute. The session must be at `unrestricted`. This is the one grant `workspace` does not carry: the command runs outside the turn, so nothing confines it to the workspace roots, and the API's own 403 says `unrestricted`.

A **tool** gate (`"check": {"tool": "…", "arguments": {…}}`) is not held to that bar. It may only name a tool meka resolves to `read`, and the session need only be at `read`. Both facts are re-checked on every fire, so a tool that resolves higher after a config change stops being a gate.

`execute_command` is one such tool wherever a sandbox backend is usable, so a `read` session can plant an arbitrary command on a timer through the tool form. That is deliberate and it is not the same grant as the shell form: a gate dispatches at `read`, the level meka sandboxes, so the command runs read-only-confined rather than as a bare `sh -c`, and where no sandbox is available the tool resolves above `read` and the gate is refused instead. The confinement blocks writes, not the network. See [Scheduled jobs](./scheduling.md) for the longer version.

No job of any kind can be created on a session at `none`, gated or not: nothing is dispatchable there, so the turn could neither act on the job nor cancel it, and `POST /v1/sessions/{id}/schedule` answers 403 `session-permission` rather than creating a row that can never run. A job whose session drops to `none` afterwards keeps its row and reports itself: every job view carries a `withheld` field, present only when something is holding the job back, with the same sentence the agent is given. It is computed per request from the session's current level, so it tracks a `PATCH /v1/sessions/{id}` without the job being rewritten.

A `schedule:*`-only token can still plant ordinary prompt-only jobs; it cannot reach a gate at all. Scope a bridge accordingly, and note that `GET /v1/schedule` is server-wide, so `schedule:r` alone lists every session id in the database.

`DELETE /v1/schedule/{job_id}` and `DELETE /v1/sessions/{id}/tasks/{task_id}` both accept a unique id prefix as well as the full id, matching `meka schedule cancel` and the `schedule_cancel` / `task_cancel` tools — the 8-character short form those surfaces print is enough. An id matching nothing is a 404 and one matching several is a 422, so a typo is never reported as a cancellation. A job that a scheduler sweep retired between the lookup and the delete is a 404 as well, for the same reason: 204 means this request cancelled the job, not merely that it is gone.

Cancelling a background task records the cancellation and signals the running task, but only `meka serve` can signal work `meka serve` started. If the session is open in another process (a `meka -r` REPL, say), the row is marked `cancelled` and the command keeps running there until it ends on its own; its result is then discarded, because the row is no longer `running`.

`POST /v1/mcp/{name}/reconnect` answers 200 with where the server now stands, which is not the same as "it worked": **read `state`, not the status code**. An attempt that ran and failed is a 200 carrying `state: "failed"`, not a 502. A server the startup sweep is still connecting comes back as `state: "pending"` with no attempt made, so a dashboard polling `GET /v1/mcp` during startup does not mistake "still coming up" for "down". The two non-200s are narrow: 422 when the server is `disabled` in config, and 502 when an already-connected server's transport could not be re-established within `[mcp] connect_timeout_seconds`.

MCP OAuth login and logout are deliberately absent: the flow opens a browser and pastes back a callback, which does not belong on a service-to-service surface. Use `meka mcp login` on the host. `/v1/providers` is read-only for the same reason provider selection has no environment tier: an ambient value must never silently rebind which account a named profile bills.

For full request/response schemas, see `/v1/openapi.json` on a running server, or browse it interactively at `/v1/docs` (Swagger UI).

Both are **off unless you set [`[serve].docs`](../configuration/config-file.md#servedocs)**, and both are unauthenticated when on, so CI pipelines and code generators can fetch the spec without a token. That combination is what makes them opt-in: they take no token *and* they publish the shape of every endpoint the deployment exposes, which is useful on a workstation and reconnaissance anywhere else.

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
